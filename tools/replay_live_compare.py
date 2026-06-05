"""
replay_live_compare.py — LIVE-EQUIVALENT replay of PRODUCTION vs 4-MONTH config, last 168h.
Uses crypto_signal_core.compute_signal_core (the SAME core the live shadow/validator use:
ffill NaN policy, ternary signal) on the last CLOSED bar each hour -> confidences match the
live trader, not the backtest engine. Replicates crypto_live_shadow.py's data prep exactly.

Validation: production replay signal/conf vs the actual signal_log (sig_2/conf_2) so you can
see fidelity to live before trusting the 4mo numbers.

Bear 5h model only (week 100% bear). Gates: conf (prod 80 / 4mo 65), rally-cd (exact), max_hold=10.
Execution price = hist close at decision hour H. Maker 0.05%/leg. Deterministic (GPU). No rounding.
"""
import os
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
os.environ.setdefault('FAYE_MODELS_DIR', 'models')
import sys
import csv as _csv
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import crypto_live_trader_ed as LT          # noqa: E402  build_all_features/_compute_pysr/ALL_MODELS
import crypto_signal_core as CORE           # noqa: E402  compute_signal_core (shadow/validator core)
from crypto_trading_system_faye import BACKTEST_FEE_PER_LEG as FEE  # noqa: E402

ASSET, NOTIONAL, MAX_HOLD, HORIZON = "ETH", 14300.0, 10, 5
hist = LT.load_data(ASSET).sort_values("datetime").reset_index(drop=True)
keyfmt = hist["datetime"].dt.strftime("%Y-%m-%d %H:%M")
idx_at = {keyfmt.iloc[i]: i for i in range(len(hist))}
close_at = {keyfmt.iloc[i]: float(hist["close"].iloc[i]) for i in range(len(hist))}
cl = hist["close"].values


def prod_feats():
    for r in _csv.DictReader(open("models/crypto_ed_production.csv")):
        if r["coin"] == ASSET and r["horizon"] == "5":
            return [f.strip() for f in r["optimal_features"].split(",") if f.strip() and f.strip() != "nan"]


FOURMO_FEATS = ["deriv_basis", "pysr_5", "vol_ratio_12_48", "logret_240h", "adx_14h",
                "price_accel_12h", "logret_72h", "price_to_sma100h", "deriv_funding_zscore", "pysr_3"]

CFG = {
    "PRODUCTION": dict(combo=["RF", "LGBM"], window=150, gamma=0.999, feats=prod_feats(), conf=80,
                       rc=dict(h_short=10, h_long=12, t_short=5.5, t_long=2.0, cd=8)),
    "4-MONTH":   dict(combo=["RF", "LGBM"], window=250, gamma=0.999, feats=FOURMO_FEATS, conf=65,
                      rc=dict(h_short=20, h_long=36, t_short=7.0, t_long=9.0, cd=12)),
}

last_i = len(hist) - 1
decision = [keyfmt.iloc[i] for i in range(last_i - 167, last_i + 1)]

live_log = {}
for r in _csv.DictReader(open("config/signal_log.csv")):
    if r["asset"] == ASSET:
        h = r["timestamp"][:13].replace("T", " ") + ":00"
        live_log[h] = (r.get("sig_2", ""), r.get("conf_2", ""), r.get("action", ""))


def core_signal(spec, H):
    """Live-equivalent: build features on data strictly before H (last bar = H-1, the closed bar),
    ffill, train last `window` labeled rows, predict the closed bar via compute_signal_core."""
    i_h = idx_at[H]
    dft = hist.iloc[max(0, i_h - 2600):i_h].reset_index(drop=True)   # data < H
    f, c = LT.build_all_features(dft, asset_name=ASSET, horizon=HORIZON, verbose=False, keep_label_nan_tail=True)
    LT._compute_pysr_features(f, c, ASSET, HORIZON, verbose=False)
    fc = [x for x in spec["feats"] if x in c]
    if len(fc) != len(spec["feats"]):
        return None
    dtr = f.dropna(subset=["label"]).reset_index(drop=True)
    df = f.reset_index(drop=True).copy()
    df[fc] = df[fc].ffill().fillna(0.0)
    dtr[fc] = dtr[fc].ffill().fillna(0.0)
    if len(dtr) < spec["window"] + 50:
        return None
    ts = max(0, len(dtr) - spec["window"])
    X_train = dtr.iloc[ts:][fc].values
    y_train = dtr.iloc[ts:]["label"].values
    X_test = df.iloc[len(df) - 1:len(df)][fc].values
    mf = {n: LT.ALL_MODELS[n] for n in spec["combo"]}
    res = CORE.compute_signal_core(X_train=X_train, y_train=y_train, X_test=X_test,
                                   model_factories=mf, gamma=spec["gamma"], na_policy="ffill",
                                   return_probas=True, binary_signal=False)
    if not res or res.get("signal") is None:
        return None
    return res["signal"], float(res["confidence"])


def run(name, spec):
    g = spec["rc"]
    cooldown_until = -1
    in_pos = False
    entry = None
    trades = []
    sigrows = []
    forced = blocked = rc_trig = 0
    for H in decision:
        i = idx_at[H]
        sc = core_signal(spec, H)
        if sc is None:
            continue
        sig, conf = sc
        px = close_at[H]
        sigrows.append((H, sig, conf))
        rr_s = (cl[i] / cl[i - g["h_short"]] - 1) * 100
        rr_l = (cl[i] / cl[i - g["h_long"]] - 1) * 100
        if rr_s >= g["t_short"] or rr_l >= g["t_long"]:
            cooldown_until = max(cooldown_until, i + g["cd"])
            rc_trig += 1
        cd_active = i < cooldown_until
        if in_pos:
            held = i - entry["i"]
            why = "signal" if sig == "SELL" else ("max_hold" if held >= MAX_HOLD else None)
            if why == "max_hold":
                forced += 1
            if why:
                net = (1 - FEE) ** 2 * (px / entry["px"]) - 1
                trades.append({**entry, "ex": H, "expx": px, "exconf": conf, "net": net, "h": held, "why": why})
                in_pos, entry = False, None
        elif sig == "BUY":
            if conf < spec["conf"]:
                pass
            elif cd_active:
                blocked += 1
            else:
                in_pos, entry = True, {"bt": H, "i": i, "px": px, "conf": conf}
    open_t = None
    if in_pos:
        lp = close_at[decision[-1]]
        open_t = {**entry, "mark": lp, "net": (1 - FEE) ** 2 * (lp / entry["px"]) - 1}
    return dict(name=name, trades=trades, sigrows=sigrows, forced=forced, blocked=blocked,
                rc_trig=rc_trig, open_t=open_t, conf=spec["conf"])


R = {name: run(name, spec) for name, spec in CFG.items()}

print("\n" + "=" * 100)
print("  VALIDATION — PRODUCTION replay (compute_signal_core) vs ACTUAL signal_log (sig_2/conf_2)")
print("=" * 100)
pr = R["PRODUCTION"]
match = tot = 0
diffs = []
for H, sig, conf in pr["sigrows"]:
    if H in live_log and live_log[H][0]:
        lsig, lconf, _ = live_log[H]
        tot += 1
        d = conf - (float(lconf) if lconf else 0)
        match += (sig == lsig)
        diffs.append((H, sig, conf, lsig, lconf, d, sig == lsig))
print(f"  hours compared: {tot} | same raw bear signal: {match}/{tot} ({100*match/max(tot,1):.0f}%) | avg|conf delta|: {np.mean([abs(x[5]) for x in diffs]) if diffs else 0:.2f}")
print("  sample (last 10):")
for H, sig, conf, lsig, lconf, d, sm in diffs[-10:]:
    print(f"    {H}  replay={sig}({conf:.1f})  live={lsig}({lconf})  delta={d:+.1f}  {'OK' if sm else 'SIGDIFF'}")

for name in ("PRODUCTION", "4-MONTH"):
    r = R[name]
    print("\n" + "=" * 100)
    print(f"  {name} (live-equiv) bear5h@{r['conf']} | rc_trig={r['rc_trig']} blocked={r['blocked']} max_hold={r['forced']}")
    print("=" * 100)
    eq = 1.0
    for n, t in enumerate(r["trades"], 1):
        eq *= (1 + t["net"])
        print(f"  {n:>2} BUY {t['bt']} @{t['px']!r} cf{t['conf']:.0f} -> SELL {t['ex']} @{t['expx']!r} cf{t['exconf']:.0f} h{t['h']:>2} {t['why']:>8} net{t['net']*100:+.6f}% ${NOTIONAL*t['net']:+.2f}")
    r["comp"] = (eq - 1) * 100
    r["sum"] = sum(NOTIONAL * t["net"] for t in r["trades"])
    r["w"] = sum(1 for t in r["trades"] if t["net"] > 0)
    print(f"  -> {len(r['trades'])} trades {r['w']}W/{len(r['trades'])-r['w']}L | compounded {r['comp']:+.6f}% | sum ${r['sum']:+.2f}"
          + (f" | OPEN {r['open_t']['net']*100:+.4f}% (${NOTIONAL*r['open_t']['net']:+.2f})" if r["open_t"] else " | flat"))

print("\n" + "#" * 100)
print(f"  {'':<20}{'PRODUCTION':>18}{'4-MONTH':>18}")
print(f"  {'closed trades':<20}{len(R['PRODUCTION']['trades']):>18}{len(R['4-MONTH']['trades']):>18}")
print(f"  {'compounded %':<20}{R['PRODUCTION']['comp']:>18.4f}{R['4-MONTH']['comp']:>18.4f}")
print(f"  {'sum $@14300':<20}{R['PRODUCTION']['sum']:>18.2f}{R['4-MONTH']['sum']:>18.2f}")
print("#" * 100)
