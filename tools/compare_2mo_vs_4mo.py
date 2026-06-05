"""
compare_2mo_vs_4mo.py — apples-to-apples: 2mo HRST vs 4mo HRST (same base).
=========================================================================
Both winners were selected as-of-now through the SAME FAYE near-live engine, so the
ONLY thing that differs is the SELECTION WINDOW (1440h vs 2880h). Each is backtested
over the eval windows through the same maker-fee long/cash regime sim. Fixes the old
compare_prod_vs_4mo flaw (which used the live production config with a *different*
PySR/base, so it wasn't isolating window length).

Inputs (saved winners — config + per-horizon model params):
    4mo : output/winner_4mo_regime.json   + output/winner_4mo_prod.csv   (saved already)
    2mo : output/winner_2mo_regime.json   + output/winner_2mo_prod.csv
          -> this script SNAPSHOTS the current *_noprod into winner_2mo_* on first run,
             so run it right after the 2mo HRST finishes (before anything overwrites noprod).

Still IN-SAMPLE for both (eval windows sit inside both selection windows) — but now it's a
*fair* in-sample comparison that isolates window length. The clean OOS version needs
past-anchored selections (data_asof_* + --no-data-update); this is the timely operational read.

Run:  python tools/compare_2mo_vs_4mo.py            # windows 720 336 168
      python tools/compare_2mo_vs_4mo.py 1440 720 168
"""
import os
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
import sys
import json
import shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crypto_trading_system_faye import generate_signals, BACKTEST_FEE_PER_LEG  # noqa: E402

ASSET = "ETH"
FEE = BACKTEST_FEE_PER_LEG
WINDOWS = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [720, 336, 168]

# Snapshot the just-run 2mo winner from _noprod (before anything overwrites it).
if not os.path.exists("output/winner_2mo_regime.json"):
    shutil.copy("config_faye/regime_config_faye_noprod.json", "output/winner_2mo_regime.json")
    shutil.copy("models_faye/crypto_faye_production_noprod.csv", "output/winner_2mo_prod.csv")
    print("  [snapshot] saved current _noprod -> output/winner_2mo_*  (the fresh 2mo winner)")

SOURCES = {
    "2mo HRST": ("output/winner_2mo_regime.json", "output/winner_2mo_prod.csv"),
    "4mo HRST": ("output/winner_4mo_regime.json", "output/winner_4mo_prod.csv"),
}


def load_cfg(regime_json, prod_csv):
    d = json.load(open(regime_json)).get(ASSET, {})
    return {
        "detector": d.get("regime_detector", {}).get("params", {}).get("name", "tsmom_672h"),
        "bull_h": d["bull"]["horizon"], "bear_h": d["bear"]["horizon"],
        "bull_conf": d["bull"]["min_confidence"], "bear_conf": d["bear"]["min_confidence"],
        "prod_csv": prod_csv,
    }


def _cfg_row(csv, horizon):
    p = pd.read_csv(csv); p = p[p["coin"] == ASSET]
    r = p[p["horizon"] == horizon].iloc[0]
    feats = [f.strip() for f in str(r["optimal_features"]).split(",") if f.strip() and f.strip() != "nan"]
    return {"models": r["best_combo"].split("+"), "window": int(r["best_window"]),
            "gamma": float(r["gamma"]), "features": feats}


def _gen(csv, horizon, replay):
    c = _cfg_row(csv, horizon)
    return {s["datetime"]: s for s in generate_signals(
        asset_name=ASSET, model_names=c["models"], window_size=c["window"], replay_hours=replay,
        feature_override=c["features"], horizon=horizon, gamma=c["gamma"])}


def _det_map(name):
    df = pd.read_csv("data/eth_hourly_data.csv", parse_dates=["datetime"]).sort_values("datetime")
    c = df["close"].reset_index(drop=True)
    if name == "tsmom_672h":
        bull = (np.log(c / c.shift(672)) > 0).values
    else:  # sma24>sma100
        bull = (c.rolling(24).mean() > c.rolling(100).mean()).values
    return dict(zip(df["datetime"].dt.strftime("%Y-%m-%d %H:%M").values, bull))


def _gate(sig, conf, mc):
    return "HOLD" if (sig == "BUY" and conf is not None and conf < mc) else sig


def _sim(cfg, sig_by_h, det_map):
    dts = sorted(set().union(*[set(d.keys()) for d in sig_by_h.values()]))
    cash, held, in_pos, entry = 1000.0, 0.0, False, 0.0
    trades = wins = 0
    first = last = None
    for dt in dts:
        h = cfg["bull_h"] if det_map.get(dt, False) else cfg["bear_h"]
        mc = cfg["bull_conf"] if det_map.get(dt, False) else cfg["bear_conf"]
        s = sig_by_h.get(h, {}).get(dt)
        if s is None:
            continue
        px = s["close"]; last = px
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
    return {"ret": ret, "trades": trades, "wr": (wins / trades * 100) if trades else 0.0, "bh": bh}


def main():
    cfgs = {name: load_cfg(*src) for name, src in SOURCES.items()}
    print("=" * 80)
    print(f"  2mo vs 4mo HRST (same base, FAYE engine, maker {FEE*100:.2f}%/leg) — isolates window length")
    for name, c in cfgs.items():
        print(f"    {name}: detector={c['detector']} bull {c['bull_h']}h@{c['bull_conf']} / bear {c['bear_h']}h@{c['bear_conf']}")
    print(f"  Windows: {', '.join(str(w)+'h' for w in WINDOWS)}")
    print("=" * 80)
    res = {}
    for name, c in cfgs.items():
        det = _det_map(c["detector"])
        for w in WINDOWS:
            print(f"  >>> {name} @ {w}h ...")
            sig = {h: _gen(c["prod_csv"], h, w) for h in sorted({c["bull_h"], c["bear_h"]})}
            res[(name, w)] = _sim(c, sig, det)
    print("\n" + "=" * 80 + "\n  RESULT")
    head = f"  {'Model':<14}" + "".join(f"{str(w)+'h':>22}" for w in WINDOWS)
    print(head + "\n  " + "-" * (len(head) - 2))
    for name in SOURCES:
        print(f"  {name:<14}" + "".join(f"{res[(name,w)]['ret']:+6.1f}% WR{res[(name,w)]['wr']:3.0f}% n{res[(name,w)]['trades']:>3}".rjust(22) for w in WINDOWS))
    print(f"  {'Buy & Hold':<14}" + "".join(f"{res[(list(SOURCES)[0],w)]['bh']:+.1f}%".rjust(22) for w in WINDOWS))
    print("=" * 80)
    print("  NOTE: in-sample for BOTH (eval inside both selection windows) — fair for isolating")
    print("  window length, but not OOS. 168h = ~10-25 trades (noise); read 720h+ as the signal.")


if __name__ == "__main__":
    main()
