"""
bt_am1_vs_B_regime.py — recent-month regime-switched backtest: a-m1 vs B
========================================================================
Compares two FULL regime configs over the last REPLAY hours, each with its OWN
regime detector + per-regime horizon + per-regime min_confidence, through the
same maker-fee long/cash sim (Revolut spot — no shorting). Closed-bar inference
is inherent to a backtest.

  a-m1 (1-month-window selection): detector sma24>sma100, bull 8h@65, bear 6h@65
  B    (2-month-window = production): detector tsmom_672h, bull 6h@65, bear 5h@80

a-m1 model params come from output/am1_prod_recentmonth.csv (its _noprod winner);
B's from the live production CSV. Gate = live rule (BUY below min_conf -> HOLD;
SELL never gated).
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crypto_trading_system_ed import generate_signals, BACKTEST_FEE_PER_LEG  # noqa: E402

REPLAY = 720
ASSET = "ETH"
FEE = BACKTEST_FEE_PER_LEG

CONFIGS = {
    "a-m1 (1mo-window)": {
        "prod_csv": "output/am1_prod_recentmonth.csv",
        "detector": "sma24>sma100", "bull_h": 8, "bear_h": 6,
        "bull_conf": 65, "bear_conf": 65,
    },
    "B (2mo-window=prod)": {
        "prod_csv": "models/crypto_ed_production.csv",
        "detector": "tsmom_672h", "bull_h": 6, "bear_h": 5,
        "bull_conf": 65, "bear_conf": 80,
    },
}


def _cfg_row(csv, horizon):
    p = pd.read_csv(csv)
    p = p[p["coin"] == ASSET]
    r = p[p["horizon"] == horizon].iloc[0]
    feats = [f.strip() for f in str(r["optimal_features"]).split(",")
             if f.strip() and f.strip() != "nan"]
    return {"models": r["best_combo"].split("+"), "window": int(r["best_window"]),
            "gamma": float(r["gamma"]), "features": feats}


def _gen(csv, horizon):
    c = _cfg_row(csv, horizon)
    print(f"\n  >>> generating {ASSET} {horizon}h from {csv} "
          f"({'+'.join(c['models'])} w={c['window']} g={c['gamma']} f={len(c['features'])})")
    sigs = generate_signals(asset_name=ASSET, model_names=c["models"], window_size=c["window"],
                            replay_hours=REPLAY, feature_override=c["features"],
                            horizon=horizon, gamma=c["gamma"])
    return {s["datetime"]: s for s in sigs}


def _detector_maps():
    df = pd.read_csv("data/eth_hourly_data.csv", parse_dates=["datetime"]).sort_values("datetime")
    c = df["close"].reset_index(drop=True)
    sma_bull = (c.rolling(24).mean() > c.rolling(100).mean()).values
    tsmom_bull = (np.log(c / c.shift(672)) > 0).values
    dts = df["datetime"].dt.strftime("%Y-%m-%d %H:%M").values
    return {"sma24>sma100": dict(zip(dts, sma_bull)),
            "tsmom_672h": dict(zip(dts, tsmom_bull))}


def _gate(sig, conf, minconf):
    if sig == "BUY" and conf is not None and conf < minconf:
        return "HOLD"
    return sig


def _sim(name, cfg, sig_by_h, det_map):
    dts = sorted(set().union(*[set(d.keys()) for d in sig_by_h.values()]))
    cash, held, in_pos, entry = 1000.0, 0.0, False, 0.0
    trades = wins = bull_hours = 0
    first = last = None
    for dt in dts:
        is_bull = bool(det_map.get(dt, False))
        bull_hours += is_bull
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
            held = cash * (1 - FEE) / px; cash = 0.0; in_pos = True; entry = px; trades += 1
        elif act == "SELL" and in_pos:
            cash = held * px * (1 - FEE); wins += int(px > entry); held = 0.0; in_pos = False
    if in_pos and last:
        cash = held * last * (1 - FEE); wins += int(last > entry)
    ret = (cash / 1000.0 - 1) * 100
    bh = (last / first - 1) * 100 if first and last else 0.0
    rt = trades  # round trips ~ entries
    wr = (wins / trades * 100) if trades else 0.0
    return {"name": name, "ret": ret, "trades": trades, "round_trips": rt, "wr": wr,
            "bh": bh, "bull_pct": 100 * bull_hours / max(len(dts), 1), "n": len(dts)}


def main():
    print("=" * 72)
    print(f"  RECENT-MONTH REGIME BACKTEST — a-m1 vs B  ({ASSET}, {REPLAY}h, "
          f"maker {FEE * 100:.2f}%/leg)")
    print("=" * 72)
    det = _detector_maps()
    results = []
    for name, cfg in CONFIGS.items():
        sig_by_h = {}
        for h in sorted({cfg["bull_h"], cfg["bear_h"]}):
            sig_by_h[h] = _gen(cfg["prod_csv"], h)
        res = _sim(name, cfg, sig_by_h, det[cfg["detector"]])
        results.append(res)

    print("\n" + "=" * 72)
    print("  RESULT — recent month, each config's own detector + per-regime conf")
    print("=" * 72)
    for r in results:
        print(f"  {r['name']:<22} return={r['ret']:+.2f}%  trades={r['trades']}  "
              f"round_trips={r['round_trips']}  WR={r['wr']:.0f}%  bull-time={r['bull_pct']:.0f}%")
    if len(results) == 2:
        d = results[0]["ret"] - results[1]["ret"]
        print(f"\n  a-m1 − B = {d:+.2f} pp     (Buy&Hold: {results[0]['bh']:+.2f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
