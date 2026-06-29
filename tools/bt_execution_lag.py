"""bt_execution_lag.py — measure how sensitive a config's backtest is to EXECUTION LAG.

WHY (2026-06-29): the gated backtest fills instantly at the decision-bar close. Live can't
act until P1 (data download) + compute + the maker fill — historically ~36s+ (mostly a
full-history derivatives re-pull, fixed 2026-06-29 → ~7s). This tool re-fills the SAME
gated signals at the decision close + LAG seconds, using REAL spot 1-minute bars (fetched
fresh from Binance), and sweeps LAG. If the edge survives the realistic lag (~60-90s now),
the backtest is honest about timing.

SCOPE / honesty: this models the *time lag only* — it fills at the 1-minute CLOSE `lag`
seconds later. It does NOT model the maker order filling at the *bid* (below the 1m close)
while chasing a fast move — that fast-crash slippage needs bid-level data and is a separate
effect (it dominated the 2026-06-26 4h/4h live loss; pure time-lag did not). See the
CLAUDE.md "Backtest-vs-Live Fidelity" playbook.

The decision is computed ONCE per hour on the last closed bar (USE_CLOSED_BAR_FOR_INFERENCE);
lag changes only the fill price, never the buy/sell choice — so a lag sweep isolates pure
execution-price drift.

Validation built in: lag=0 should ~reproduce the hourly-close baseline (it fills the same
trades at the same bars, just sourced from 1m spot instead of the hourly CSV).

Usage:
  python tools/bt_execution_lag.py --detector price_sma72 --bull 4 --bear 4 \
      --bullconf 65 --bearconf 70 --csv models/crypto_ed_production.csv --replay 200
  # add --lags 0,30,60,90,120,180  (seconds)  ·  --asset ETH  ·  --start 2026-06-22
"""
import os
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
os.environ.setdefault('FAYE_CONFIG_DIR', 'config_faye_bt')
import sys, json, time, ssl, argparse, urllib.request
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'tools'))
os.chdir(HERE)
import pandas as pd
import bt_regime as B
from compare_prod_vs_4mo import _gate

SPOT_KLINES = "https://api.binance.com/api/v3/klines"  # SPOT — matches the hourly CSV + live fills


def fetch_spot_1m(symbol, start_ms, end_ms):
    ctx = ssl._create_unverified_context()
    rows, cur = [], start_ms
    while cur < end_ms:
        url = f"{SPOT_KLINES}?symbol={symbol}&interval=1m&startTime={cur}&limit=1000"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            data = json.loads(r.read().decode())
        if not data:
            break
        rows.extend(data)
        cur = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.15)
    m = pd.DataFrame(rows, columns=['t', 'o', 'h', 'l', 'c', 'v', 'ct', 'qv', 'n', 'tb', 'tq', 'ig'])
    m['dt'] = pd.to_datetime(m['t'], unit='ms')
    m['close'] = pd.to_numeric(m['c'])
    return m.drop_duplicates('dt').set_index('dt').sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', default='price_sma72')
    ap.add_argument('--bull', type=int, default=4)
    ap.add_argument('--bear', type=int, default=4)
    ap.add_argument('--bullconf', type=int, default=65)
    ap.add_argument('--bearconf', type=int, default=70)
    ap.add_argument('--csv', default='models/crypto_ed_production.csv')
    ap.add_argument('--asset', default='ETH')
    ap.add_argument('--replay', type=int, default=200)
    ap.add_argument('--start', default=None, help='only count trades on/after this date (YYYY-MM-DD)')
    ap.add_argument('--lags', default='0,30,60,90,120,180', help='comma sep seconds')
    a = ap.parse_args()

    horizons = sorted({a.bull, a.bear})
    det = {pd.Timestamp(k): v for k, v in B._bull_map_for(a.detector).items()}
    print(f"generating {horizons}h signals (live path, per-hour retrain)...", flush=True)
    sig_by_h = {h: {pd.Timestamp(k): v for k, v in B._gen(a.csv, h, a.replay).items()} for h in horizons}
    cfg = dict(bull_h=a.bull, bear_h=a.bear, bull_conf=a.bullconf, bear_conf=a.bearconf)

    all_keys = sorted(set().union(*[set(d) for d in sig_by_h.values()]))
    start = pd.Timestamp(a.start) if a.start else all_keys[0]
    keys = [k for k in all_keys if k >= start]
    print(f"signals: {len(keys)}  span {keys[0]} -> {keys[-1]}")

    sym = f"{a.asset}USDT"
    m = fetch_spot_1m(sym, int(keys[0].timestamp() * 1000),
                      int((keys[-1] + pd.Timedelta(hours=2)).timestamp() * 1000))
    print(f"fetched {len(m)} spot 1m bars: {m.index.min()} -> {m.index.max()}\n")

    def price_at(ts):
        sub = m.index[m.index <= ts]
        return float(m.loc[sub[-1], 'close']) if len(sub) else None

    # alignment check — sig.close (decision price) must match the 1m series at wall-clock k
    chk = []
    for k in keys[:3]:
        s = sig_by_h[a.bull].get(k) or sig_by_h[a.bear].get(k)
        if s:
            chk.append((str(k)[11:16], round(s['close'], 2), round(price_at(k) or 0, 2)))
    print("alignment (sig.close vs spot 1m@k):", chk, "\n")

    def sim(lag_s):
        cash, held, inpos, entry = 1000.0, 0.0, False, 0.0
        trades = wins = 0
        for k in keys:
            is_bull = bool(det.get(k, False))
            h = cfg['bull_h'] if is_bull else cfg['bear_h']
            mc = cfg['bull_conf'] if is_bull else cfg['bear_conf']
            s = sig_by_h.get(h, {}).get(k)
            if s is None:
                continue
            act = _gate(s['signal'], s['confidence'], mc)
            px = price_at(k + pd.Timedelta(seconds=lag_s))   # decision aligns to wall-clock k
            if px is None:
                continue
            if act == 'BUY' and not inpos:
                held = cash / px; cash = 0.0; inpos = True; entry = px; trades += 1
            elif act == 'SELL' and inpos:
                cash = held * px; wins += int(px > entry); held = 0.0; inpos = False
        if inpos:
            last = float(m['close'].iloc[-1]); cash = held * last; wins += int(last > entry)
        return (cash / 1000 - 1) * 100, trades, (wins / trades * 100 if trades else 0)

    print(f"=== EXECUTION-LAG SWEEP  {a.detector} bull {a.bull}h@{a.bullconf} / bear {a.bear}h@{a.bearconf} ===")
    print("exec lag | return  | trades | WR")
    for lag in [int(x) for x in a.lags.split(',')]:
        r, t, wr = sim(lag)
        tag = '  <- fill at bar close (validates vs hourly baseline)' if lag == 0 else \
              ('  <- ~realistic live now (~7s compute + maker fill)' if lag == 60 else '')
        print(f"  {lag:3d}s   | {r:+6.2f}% | {t:5d}  | {wr:3.0f}%{tag}")
    print("\nNOTE: models TIME lag only (fill at 1m close + lag); NOT maker bid-chase in fast moves.")


if __name__ == '__main__':
    main()
