"""C01 RETEST (rewritten 2026-07-01) — Volatility-Scaled Horizons on the LEAKAGE-FREE engine.

The SHELVED C01 idea: pick the trading horizon by VOL regime (high vol → shorter horizon)
instead of by a trend detector. Original verdict (2026-05-09): 2mo +5.02pp over the tsmom
baseline, but 4mo only +2.99pp (below the +5pp ship bar) → SHELVED as window-shopping.

Why retest (2026-07-01, user): the 2026-06-29 leak fix made the backtest honest (leakage-free
training edge), and the forming-bar-label leak scales WITH horizon (last train row = i-horizon)
— so a horizon-SWITCHING idea can interact with the leak asymmetrically and its verdict may
shift. This version fixes two staleness bugs in the old tool: (1) it imported the now-archived
`crypto_trading_system_ed`; (2) it compared against the DEFUNCT `tsmom_672h bull6h/bear8h`
baseline. Now: faye (leakage-free) + the CURRENT LIVE detector as the baseline.

Faithful gated A/B (F6.2): each horizon's signals are generated ONCE via faye's leakage-free
generate_signals; every strategy trades the SAME per-horizon signals — only the horizon PICKER
+ conf gate differ. Sub-period consistency (F6.3): a promising variant must beat live in EVERY
chunk, not just one window (that's what killed the 2mo→4mo replication).

Baseline = LIVE: detector sma168>sma480 → bull=6h@80% / bear=4h@65%.
Promotion bar: a vol variant must beat the LIVE baseline by >= +5pp in the full window AND in
every sub-period chunk. Else C01 STAYS SHELVED (confirmed leakage-free).

Run:  python tools/test_vol_scaled_horizon.py                 # 2mo (default)
      C01_REPLAY=2880 python tools/test_vol_scaled_horizon.py # 4mo replication test
"""
import sys, os, warnings
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
os.environ.setdefault('FAYE_MODELS_DIR', 'models')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
from crypto_trading_system_faye import (
    generate_signals, simulate_portfolio, load_data,
    _suppress_stderr, BACKTEST_FEE_PER_LEG,
)

ASSET = 'ETH'
REPLAY = int(os.environ.get('C01_REPLAY', '1440'))
PROD_CSV = 'models/crypto_ed_production.csv'
HORIZONS = [4, 5, 6, 7, 8]
K = 3  # sub-period chunks


def main():
    print("=" * 96)
    print(f"  C01 RETEST — VOL-SCALED HORIZONS (leakage-free faye) | {ASSET} | replay={REPLAY}h")
    print(f"  baseline = LIVE sma168>sma480 bull=6h@80% / bear=4h@65%   (promotion bar: +5pp AND every chunk)")
    print("=" * 96)

    df_models = pd.read_csv(PROD_CSV)
    sigs = {}
    for h in HORIZONS:
        rows = df_models[(df_models['coin'] == ASSET) & (df_models['horizon'] == h)]
        if len(rows) == 0:
            print(f"  no {h}h model — skip"); continue
        row = rows.iloc[0]
        feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
        gamma = float(row.get('gamma', 1.0))
        print(f"  generating {h}h signals (w={int(row['best_window'])} g={gamma}, {row['models']})...", flush=True)
        with _suppress_stderr():
            raw = generate_signals(ASSET, row['models'].split('+'), int(row['best_window']),
                                   REPLAY, feature_override=feats, horizon=h, gamma=gamma)
            raw = simulate_portfolio(raw)
        store = {}
        for s in raw:
            dt = s['datetime']
            if isinstance(dt, str):
                dt = datetime.strptime(dt, '%Y-%m-%d %H:%M'); s['datetime'] = dt
            store[dt] = s
        sigs[h] = store
        print(f"    {len(store)} signals")

    if not sigs:
        print("  no signals generated — abort"); return

    df = load_data(ASSET); df['datetime'] = pd.to_datetime(df['datetime'])
    di = df.set_index('datetime').sort_index()
    di['logret'] = np.log(di['close'] / di['close'].shift(1))
    di['vol24'] = di['logret'].rolling(24).std()
    di['volpct'] = di['vol24'].rolling(720, min_periods=100).rank(pct=True)   # causal vol percentile
    di['sma168'] = di['close'].rolling(168).mean()
    di['sma480'] = di['close'].rolling(480).mean()

    all_dts = sorted(set().union(*[s.keys() for s in sigs.values()]))
    bounds = np.linspace(0, len(all_dts), K + 1).astype(int)

    def _bull(dt):
        r = di.loc[dt]
        if pd.isna(r['sma168']) or pd.isna(r['sma480']):
            return True   # warmup → default bull (matches live fallback)
        return r['sma168'] > r['sma480']

    def _vp(dt):
        v = di.loc[dt, 'volpct']
        return v if pd.notna(v) else 0.0

    def simulate(picker, conf_of, dts):
        cash, held, inpos, entry = 1000.0, 0.0, False, 0.0
        trades, wins = 0, 0; fp = lp = None
        for dt in dts:
            if dt not in di.index:
                continue
            h = picker(dt); s = sigs.get(h, {}).get(dt)
            if s is None:
                continue
            price, sig, conf = s['close'], s['signal'], s['confidence']
            if fp is None: fp = price
            lp = price
            if sig == 'BUY' and conf >= conf_of(dt) and not inpos:
                held = cash * (1 - BACKTEST_FEE_PER_LEG) / price; cash = 0; inpos = True; entry = price; trades += 1
            elif sig == 'SELL' and inpos:
                cash = held * price * (1 - BACKTEST_FEE_PER_LEG); wins += 1 if price > entry else 0; held = 0; inpos = False
        if inpos and lp:
            cash = held * lp * (1 - BACKTEST_FEE_PER_LEG); wins += 1 if lp > entry else 0
        ret = (cash / 1000.0 - 1) * 100; wr = (wins / trades * 100) if trades else 0
        return ret, trades, wr

    def run_row(label, picker, conf_of):
        full = simulate(picker, conf_of, all_dts)
        chunks = [simulate(picker, conf_of, all_dts[bounds[i]:bounds[i + 1]])[0] for i in range(K)]
        print(f"  {label:<40} {full[0]:>+8.2f}% tr={full[1]:>3} wr={full[2]:>3.0f}%  | chunks "
              f"{chunks[0]:>+7.1f} {chunks[1]:>+7.1f} {chunks[2]:>+7.1f}")
        return full[0], chunks

    print(f"\n  {'strategy':<40} {'return':>9}          | sub-period chunks (window-shop guard)")
    print("  " + "-" * 94)
    base_ret, base_chunks = run_row("LIVE sma168>sma480 6h@80/4h@65", lambda dt: 6 if _bull(dt) else 4,
                                    lambda dt: 80 if _bull(dt) else 65)
    print("  --- single-horizon refs @80 ---")
    for h in HORIZONS:
        if h in sigs:
            run_row(f"{h}h-only @80", (lambda dt, hh=h: hh), (lambda dt: 80))

    print("  --- vol-scaled horizons (conf sweep) ---")
    vol_strats = [
        ("vol2 hi>4h lo>8h", lambda dt: 4 if _vp(dt) > 0.7 else 8),
        ("vol2 hi>4h lo>6h", lambda dt: 4 if _vp(dt) > 0.7 else 6),
        ("vol2 hi>6h lo>8h", lambda dt: 6 if _vp(dt) > 0.7 else 8),
        ("volMed hi>4h lo>6h", lambda dt: 4 if _vp(dt) > 0.5 else 6),
        ("volMed hi>6h lo>8h", lambda dt: 6 if _vp(dt) > 0.5 else 8),
        ("vol3 4/6/8", lambda dt: 4 if _vp(dt) > 0.8 else (6 if _vp(dt) > 0.4 else 8)),
        ("vol3 5/6/8", lambda dt: 5 if _vp(dt) > 0.8 else (6 if _vp(dt) > 0.4 else 8)),
    ]
    winners = []
    for conf in [65, 80, 85]:
        for label, picker in vol_strats:
            r, ch = run_row(f"{label} @{conf}", picker, (lambda dt, c=conf: c))
            if r >= base_ret + 5 and all(ch[i] >= base_chunks[i] for i in range(K)):
                winners.append((label, conf, r, r - base_ret))

    print("  " + "-" * 94)
    print(f"  LIVE baseline = {base_ret:+.2f}%   (chunks {base_chunks[0]:+.1f} {base_chunks[1]:+.1f} {base_chunks[2]:+.1f})")
    if winners:
        print("  ⚠ PROMISING — beats LIVE by >= +5pp AND wins EVERY sub-period chunk:")
        for lbl, c, r, d in winners:
            print(f"     {lbl} @{c}: {r:+.2f}% ({d:+.2f}pp over live)")
        print(f"  → confirm on the 4mo window (C01_REPLAY=2880) before any promotion (Rule F1/F6).")
    else:
        print("  → NO vol-scaled variant beats the LIVE baseline by +5pp with sub-period consistency.")
        print("    C01 STAYS SHELVED on the leakage-free engine (reconfirms the 2026-05 window-shop verdict).")


if __name__ == '__main__':
    main()
