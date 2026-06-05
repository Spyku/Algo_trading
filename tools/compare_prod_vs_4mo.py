"""
compare_prod_vs_4mo.py — Production (2mo HRST) vs 4mo HRST, over last-month + last-week
=====================================================================================
Purpose: decide whether a 2-MONTH or 4-MONTH HRST selection window produces the better
ETH model. Backtests BOTH regime configs over the last 720h (1 month) and 168h (1 week)
through the SAME FAYE near-live engine + maker-fee long/cash sim (Revolut spot, no
shorting). Both configs use models_faye/pysr_*.json for PySR (5-8h unchanged), so the
only thing that varies is the model selection itself -> a fair comparison.

  PRODUCTION (2mo, live):  tsmom_672h | bull 6h@65 | bear 5h@80
                           models/crypto_ed_production.csv
  4mo HRST (--replay 2880, log ed_v1_20260604_075223):
                           tsmom_672h | bull 5h@70 | bear 5h@65
                           models_faye/crypto_faye_production_noprod.csv

Gate = live rule (BUY below min_conf -> HOLD; SELL never gated). Closed-bar inference is
inherent to a backtest. Each generate_signals retrains per-hour (near-live step=1), so a
720h window is ~720 fits/horizon -> the full run is ~45-90 min on Desktop.

Run on Desktop or Laptop (venv active, from engine root):
    python tools/compare_prod_vs_4mo.py
    python tools/compare_prod_vs_4mo.py 720 336 168   # custom windows (space-separated h)

Note: this measures the LAST-MONTH/LAST-WEEK windows directly. Per Critical Rule 17,
short windows have wide error bars (168h at 5-6h horizon => only ~10-25 trades), so read
the 720h column as the signal and 168h as a tiebreaker, not the other way round.
"""
import os
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')  # skip FAYE's os.execv re-exec (Windows detach gotcha)
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crypto_trading_system_faye import generate_signals, BACKTEST_FEE_PER_LEG  # noqa: E402

ASSET = "ETH"
FEE = BACKTEST_FEE_PER_LEG
WINDOWS = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [720, 168]  # last month, last week

CONFIGS = {
    "PRODUCTION (2mo)": {
        "prod_csv": "models/crypto_ed_production.csv",
        "bull_h": 6, "bear_h": 5, "bull_conf": 65, "bear_conf": 80,
    },
    "4mo HRST": {
        "prod_csv": "models_faye/crypto_faye_production_noprod.csv",
        "bull_h": 5, "bear_h": 5, "bull_conf": 70, "bear_conf": 65,
    },
}
DETECTOR = "tsmom_672h"  # both configs use the same detector


def _cfg_row(csv, horizon):
    p = pd.read_csv(csv)
    p = p[p["coin"] == ASSET]
    sub = p[p["horizon"] == horizon]
    if sub.empty:
        raise SystemExit(f"  [!] {csv}: no {horizon}h row for {ASSET}")
    r = sub.iloc[0]
    feats = [f.strip() for f in str(r["optimal_features"]).split(",")
             if f.strip() and f.strip() != "nan"]
    return {"models": r["best_combo"].split("+"), "window": int(r["best_window"]),
            "gamma": float(r["gamma"]), "features": feats}


def _gen(csv, horizon, replay):
    c = _cfg_row(csv, horizon)
    sigs = generate_signals(asset_name=ASSET, model_names=c["models"], window_size=c["window"],
                            replay_hours=replay, feature_override=c["features"],
                            horizon=horizon, gamma=c["gamma"])
    return {s["datetime"]: s for s in sigs}


def _bull_map():
    """tsmom_672h: bull when log(close / close.shift(672)) > 0."""
    df = pd.read_csv("data/eth_hourly_data.csv", parse_dates=["datetime"]).sort_values("datetime")
    c = df["close"].reset_index(drop=True)
    bull = (np.log(c / c.shift(672)) > 0).values
    dts = df["datetime"].dt.strftime("%Y-%m-%d %H:%M").values
    return dict(zip(dts, bull))


def _gate(sig, conf, minconf):
    if sig == "BUY" and conf is not None and conf < minconf:
        return "HOLD"
    return sig


def _sim(cfg, sig_by_h, det_map):
    dts = sorted(set().union(*[set(d.keys()) for d in sig_by_h.values()]))
    cash, held, in_pos, entry = 1000.0, 0.0, False, 0.0
    trades = wins = 0
    first = last = None
    for dt in dts:
        is_bull = bool(det_map.get(dt, False))
        h = cfg["bull_h"] if is_bull else cfg["bear_h"]
        mc = cfg["bull_conf"] if is_bull else cfg["bear_conf"]
        s = sig_by_h.get(h, {}).get(dt)
        if s is None:
            continue
        px = s["close"]
        last = px
        if first is None:
            first = px
        act = _gate(s["signal"], s["confidence"], mc)
        if act == "BUY" and not in_pos:
            held = cash * (1 - FEE) / px
            cash = 0.0
            in_pos = True
            entry = px
            trades += 1
        elif act == "SELL" and in_pos:
            cash = held * px * (1 - FEE)
            wins += int(px > entry)
            held = 0.0
            in_pos = False
    if in_pos and last:
        cash = held * last * (1 - FEE)
        wins += int(last > entry)
    ret = (cash / 1000.0 - 1) * 100
    bh = (last / first - 1) * 100 if first and last else 0.0
    wr = (wins / trades * 100) if trades else 0.0
    return {"ret": ret, "trades": trades, "wr": wr, "bh": bh, "n": len(dts)}


def main():
    print("=" * 80)
    print(f"  PRODUCTION (2mo) vs 4mo HRST  —  {ASSET}, detector={DETECTOR}, maker {FEE*100:.2f}%/leg")
    print(f"  Windows: {', '.join(str(w) + 'h' for w in WINDOWS)}")
    print("=" * 80)
    det = _bull_map()
    results = {}
    for name, cfg in CONFIGS.items():
        horizons = sorted({cfg["bull_h"], cfg["bear_h"]})
        for w in WINDOWS:
            print(f"\n  >>> {name} @ {w}h  (bull {cfg['bull_h']}h@{cfg['bull_conf']} / "
                  f"bear {cfg['bear_h']}h@{cfg['bear_conf']}, horizons={horizons})")
            sig_by_h = {h: _gen(cfg["prod_csv"], h, w) for h in horizons}
            results[(name, w)] = _sim(cfg, sig_by_h, det)

    def cell(r):
        return f"{r['ret']:+6.1f}% WR{r['wr']:3.0f}% n{r['trades']:>2}"

    print("\n" + "=" * 80)
    print("  RESULT")
    print("=" * 80)
    head = f"  {'Model':<20}" + "".join(f"{str(w) + 'h':>22}" for w in WINDOWS)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name in CONFIGS:
        print(f"  {name:<20}" + "".join(f"{cell(results[(name, w)]):>22}" for w in WINDOWS))
    print(f"  {'Buy & Hold':<20}" + "".join(
        f"{results[(list(CONFIGS)[0], w)]['bh']:+.1f}%".rjust(22) for w in WINDOWS))
    print("=" * 80)
    print("\n  Verdict guide:")
    print("   - 4mo beats 2mo on BOTH windows -> a 4-month HRST is worth the ~7h.")
    print("   - 4mo wins 720h but loses 168h -> 4mo captures the slower regime; fine, prefer 4mo")
    print("     (168h is ~10-25 trades, too few to overrule the month).")
    print("   - 2mo wins both -> stick with the cheaper 2-month HRST.")


if __name__ == "__main__":
    main()
