"""
replay_4mo_confsweep.py — 4-MONTH config, LIVE-equivalent (compute_signal_core, ffill,
closed-bar), last 168h, swept over confidence gate 65/70/75/80%.
Generates the bear 5h model signal+conf ONCE per hour, then applies each gate -> trades.
Rally-cd (bear rr20h>=7 OR rr36h>=9 cd=12) + max_hold=10 applied. Maker 0.05%/leg. No rounding.
"""
import os
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
os.environ.setdefault('FAYE_MODELS_DIR', 'models')
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import crypto_live_trader_ed as LT          # noqa: E402
import crypto_signal_core as CORE           # noqa: E402
from crypto_trading_system_faye import BACKTEST_FEE_PER_LEG as FEE  # noqa: E402

ASSET, NOTIONAL, MAX_HOLD, HORIZON = "ETH", 14300.0, 10, 5
COMBO, WINDOW, GAMMA = ["RF", "LGBM"], 250, 0.999
FEATS = ["deriv_basis", "pysr_5", "vol_ratio_12_48", "logret_240h", "adx_14h",
         "price_accel_12h", "logret_72h", "price_to_sma100h", "deriv_funding_zscore", "pysr_3"]
RC = dict(h_short=20, h_long=36, t_short=7.0, t_long=9.0, cd=12)
CONFS = [65, 70, 75, 80]

hist = LT.load_data(ASSET).sort_values("datetime").reset_index(drop=True)
keyfmt = hist["datetime"].dt.strftime("%Y-%m-%d %H:%M")
idx_at = {keyfmt.iloc[i]: i for i in range(len(hist))}
close_at = {keyfmt.iloc[i]: float(hist["close"].iloc[i]) for i in range(len(hist))}
cl = hist["close"].values
last_i = len(hist) - 1
decision = [keyfmt.iloc[i] for i in range(last_i - 167, last_i + 1)]


def core_signal(H):
    i_h = idx_at[H]
    dft = hist.iloc[max(0, i_h - 2600):i_h].reset_index(drop=True)
    f, c = LT.build_all_features(dft, asset_name=ASSET, horizon=HORIZON, verbose=False, keep_label_nan_tail=True)
    LT._compute_pysr_features(f, c, ASSET, HORIZON, verbose=False)
    fc = [x for x in FEATS if x in c]
    if len(fc) != len(FEATS):
        return None
    dtr = f.dropna(subset=["label"]).reset_index(drop=True)
    df = f.reset_index(drop=True).copy()
    df[fc] = df[fc].ffill().fillna(0.0)
    dtr[fc] = dtr[fc].ffill().fillna(0.0)
    if len(dtr) < WINDOW + 50:
        return None
    ts = max(0, len(dtr) - WINDOW)
    res = CORE.compute_signal_core(X_train=dtr.iloc[ts:][fc].values, y_train=dtr.iloc[ts:]["label"].values,
                                   X_test=df.iloc[len(df) - 1:len(df)][fc].values,
                                   model_factories={n: LT.ALL_MODELS[n] for n in COMBO},
                                   gamma=GAMMA, na_policy="ffill", return_probas=True, binary_signal=False)
    if not res or res.get("signal") is None:
        return None
    return res["signal"], float(res["confidence"])


# --- generate signals ONCE ---
print("Generating 4mo bear-5h live-core signals for last 168h (one pass)...", flush=True)
cache = []
for H in decision:
    sc = core_signal(H)
    if sc:
        cache.append((H, idx_at[H], sc[0], sc[1], close_at[H]))
print(f"  got {len(cache)} hourly signals\n", flush=True)


def simulate(conf_gate):
    cooldown_until = -1
    in_pos = False
    entry = None
    trades = []
    blocked = forced = rc_trig = 0
    for H, i, sig, conf, px in cache:
        rr_s = (cl[i] / cl[i - RC["h_short"]] - 1) * 100
        rr_l = (cl[i] / cl[i - RC["h_long"]] - 1) * 100
        if rr_s >= RC["t_short"] or rr_l >= RC["t_long"]:
            cooldown_until = max(cooldown_until, i + RC["cd"])
            rc_trig += 1
        cd_active = i < cooldown_until
        if in_pos:
            held = i - entry["i"]
            why = "signal" if sig == "SELL" else ("max_hold" if held >= MAX_HOLD else None)
            if why == "max_hold":
                forced += 1
            if why:
                net = (1 - FEE) ** 2 * (px / entry["px"]) - 1
                trades.append({**entry, "ex": H, "expx": px, "net": net, "h": held, "why": why})
                in_pos, entry = False, None
        elif sig == "BUY":
            if conf < conf_gate:
                pass
            elif cd_active:
                blocked += 1
            else:
                in_pos, entry = True, {"bt": H, "i": i, "px": px, "conf": conf}
    open_t = None
    if in_pos:
        lp = cache[-1][4]
        open_t = {**entry, "net": (1 - FEE) ** 2 * (lp / entry["px"]) - 1}
    eq = 1.0
    for t in trades:
        eq *= (1 + t["net"])
    return dict(conf=conf_gate, trades=trades, comp=(eq - 1) * 100,
                sum=sum(NOTIONAL * t["net"] for t in trades),
                w=sum(1 for t in trades if t["net"] > 0), forced=forced, blocked=blocked, open_t=open_t)


results = [simulate(c) for c in CONFS]

for r in results:
    print("=" * 92)
    print(f"  4-MONTH @ conf>={r['conf']}%   (live-core, last 168h)")
    print("=" * 92)
    for n, t in enumerate(r["trades"], 1):
        print(f"   {n:>2} BUY {t['bt']} @{t['px']!r} cf{t['conf']:.0f} -> SELL {t['ex']} @{t['expx']!r} h{t['h']:>2} {t['why']:>8} "
              f"net{t['net']*100:+.6f}% ${NOTIONAL*t['net']:+.2f}")
    o = r["open_t"]
    print(f"   -> {len(r['trades'])} trades {r['w']}W/{len(r['trades'])-r['w']}L | compounded {r['comp']:+.6f}% | sum ${r['sum']:+.2f}"
          + (f" | OPEN {o['net']*100:+.4f}% (${NOTIONAL*o['net']:+.2f})" if o else " | flat"))
    print()

print("#" * 92)
print(f"  {'conf gate':>10}{'trades':>9}{'W/L':>8}{'compounded %':>16}{'sum $@14300':>16}{'open':>10}")
for r in results:
    o = r["open_t"]
    openstr = (f"{o['net']*100:+.2f}%" if o else "flat")
    wl = f"{r['w']}/{len(r['trades'])-r['w']}"
    print(f"  {str(r['conf'])+'%':>10}{len(r['trades']):>9}{wl:>8}{r['comp']:>16.4f}{r['sum']:>16.2f}{openstr:>10}")
print("#" * 92)
