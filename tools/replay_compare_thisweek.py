"""
replay_compare_thisweek.py — faithful per-trade replay of PRODUCTION vs 4-MONTH config
over the last 168h, same engine/method. Prints both trade lists + a side-by-side compare.

PRODUCTION (config/regime_config_ed.json + models/crypto_ed_production.csv — intact live files):
  tsmom_672h | bull 6h@65 / bear 5h@80 | bear 5h model RF+LGBM w=150 g=0.999 15f
  bear rally-cd: rr10h>=5.5 OR rr12h>=2.0 cd=8h ; max_hold=10h ; shields off ; min_sell_pnl=0
4-MONTH (logs/ed_v1_20260604_075223 winner; model HARDCODED — its CSV was overwritten 20:14):
  tsmom_672h | bull 5h@70 / bear 5h@65 | bear 5h model RF+LGBM w=250 g=0.999 10f
  bear rally-cd: rr20h>=7.0 OR rr36h>=9.0 cd=12h ; max_hold=10h ; shields off ; min_sell_pnl=0

The last 168h is 100% bear regime (tsmom_672h<0), confirmed in-script -> only the bear model/gate
binds for both. Gates replicate crypto_revolut_ed_v2.py exactly: conf gate, rally-cd trigger
(rr_short>=t_short OR rr_long>=t_long, >= compare, cooldown to trigger_bar+cd), max_hold.
Engine: FAYE generate_signals (near-live, embargo=horizon=5, step=1). Deterministic. No rounding.
"""
import os
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('FAYE_MODELS_DIR', 'models')
import sys
import csv as _csv
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crypto_trading_system_faye import generate_signals, BACKTEST_FEE_PER_LEG  # noqa: E402

ASSET, REPLAY, FEE, NOTIONAL, MAX_HOLD = "ETH", 168, BACKTEST_FEE_PER_LEG, 14300.0, 10


def prod_5h():
    for r in _csv.DictReader(open("models/crypto_ed_production.csv")):
        if r["coin"] == ASSET and r["horizon"] == "5":
            return (r["best_combo"].split("+"), int(r["best_window"]), float(r["gamma"]),
                    [f.strip() for f in r["optimal_features"].split(",") if f.strip() and f.strip() != "nan"])


CONFIGS = {
    "PRODUCTION (live)": {
        "bear_model": prod_5h(), "bear_conf": 80,
        "rc_bear": dict(h_short=10, h_long=12, t_short=5.5, t_long=2.0, cd=8),
    },
    "4-MONTH HRST": {
        "bear_model": (["RF", "LGBM"], 250, 0.999,
                       ["deriv_basis", "pysr_5", "vol_ratio_12_48", "logret_240h", "adx_14h",
                        "price_accel_12h", "logret_72h", "price_to_sma100h", "deriv_funding_zscore", "pysr_3"]),
        "bear_conf": 65,
        "rc_bear": dict(h_short=20, h_long=36, t_short=7.0, t_long=9.0, cd=12),
    },
}

hist = pd.read_csv("data/eth_hourly_data.csv", parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
close = hist["close"].values
idx_of = {hist["datetime"].iloc[i].strftime("%Y-%m-%d %H:%M"): i for i in range(len(hist))}


def replay(name, cfg):
    combo, window, gamma, feats = cfg["bear_model"]
    sigs = generate_signals(asset_name=ASSET, model_names=combo, window_size=window,
                            replay_hours=REPLAY, feature_override=feats, horizon=5, gamma=gamma)
    sd = {s["datetime"]: s for s in sigs}
    g = cfg["rc_bear"]
    cooldown_until = -1
    in_pos = False
    entry = None
    trades = []
    nbull = rc_trig = blocked = forced = 0
    for dt in sorted(sd):
        i = idx_of[dt]
        is_bull = bool(np.log(close[i] / close[i - 672]) > 0) if i >= 672 else False
        nbull += is_bull
        rr_s = (close[i] / close[i - g["h_short"]] - 1.0) * 100.0
        rr_l = (close[i] / close[i - g["h_long"]] - 1.0) * 100.0
        if rr_s >= g["t_short"] or rr_l >= g["t_long"]:
            cooldown_until = max(cooldown_until, i + g["cd"])
            rc_trig += 1
        cd_active = i < cooldown_until
        s = sd[dt]
        sig, conf, px = s["signal"], s["confidence"], float(s["close"])
        if in_pos:
            held = i - entry["i"]
            why = "signal" if sig == "SELL" else ("max_hold" if held >= MAX_HOLD else None)
            if why == "max_hold":
                forced += 1
            if why:
                net = (1 - FEE) * (1 - FEE) * (px / entry["px"]) - 1.0
                trades.append({**entry, "exit_dt": dt, "exit_px": px, "exit_conf": float(conf),
                               "net": net, "gross": px / entry["px"] - 1.0, "hold": held, "why": why})
                in_pos, entry = False, None
        elif sig == "BUY":
            if conf is None or conf < cfg["bear_conf"]:
                pass
            elif cd_active:
                blocked += 1
            else:
                in_pos, entry = True, {"dt": dt, "i": i, "px": px, "conf": float(conf)}
    open_t = None
    if in_pos and entry is not None:
        lp = float(sd[sorted(sd)[-1]]["close"])
        open_t = {**entry, "mark": lp, "net": (1 - FEE) * (1 - FEE) * (lp / entry["px"]) - 1.0}
    return dict(name=name, trades=trades, nbull=nbull, rc_trig=rc_trig, blocked=blocked,
                forced=forced, window=window, conf=cfg["bear_conf"], open_t=open_t,
                first=sorted(sd)[0], last=sorted(sd)[-1])


results = {name: replay(name, cfg) for name, cfg in CONFIGS.items()}

for R in results.values():
    print("\n" + "=" * 104)
    print(f"  {R['name']}  — ETH last {REPLAY}h ({R['first']} -> {R['last']})  bear 5h@{R['conf']} w={R['window']}")
    print(f"  regime bull={R['nbull']}h bear={REPLAY - R['nbull']}h | rally-cd triggers={R['rc_trig']} buys_blocked={R['blocked']} max_hold_sells={R['forced']}")
    print("=" * 104)
    print(f"  {'#':>2}  {'BUY (UTC)':<16} {'buy_px':>10} {'cf':>3}  {'SELL (UTC)':<16} {'sell_px':>10} {'cf':>3} {'h':>3} {'why':>8} {'net_%':>13} {'net_$':>12}")
    eq = 1.0
    for n, t in enumerate(R["trades"], 1):
        eq *= (1 + t["net"])
        print(f"  {n:>2}  {t['dt']:<16} {t['px']!r:>10} {t['conf']:>3.0f}  {t['exit_dt']:<16} {t['exit_px']!r:>10} "
              f"{t['exit_conf']:>3.0f} {t['hold']:>3} {t['why']:>8} {t['net']*100:>+13.8f} {NOTIONAL*t['net']:>+12.4f}")
    R["compound"] = (eq - 1.0) * 100
    R["sum_usd"] = sum(NOTIONAL * t["net"] for t in R["trades"])
    R["wins"] = sum(1 for t in R["trades"] if t["net"] > 0)
    print(f"  -> {len(R['trades'])} closed | {R['wins']}W/{len(R['trades'])-R['wins']}L | "
          f"COMPOUNDED net {R['compound']:+.8f}% | sum $@14300 {R['sum_usd']:+.4f}")
    if R["open_t"]:
        o = R["open_t"]
        print(f"  [OPEN] BUY {o['dt']} @ {o['px']!r} mark {R['last']} @ {o['mark']!r} -> unrealized {o['net']*100:+.8f}% (${NOTIONAL*o['net']:+.4f}) NOT in totals")

print("\n" + "#" * 104)
print("  SIDE-BY-SIDE (closed trades only)")
print("#" * 104)
print(f"  {'metric':<28} {'PRODUCTION (live)':>22} {'4-MONTH HRST':>22}")
P, F = results["PRODUCTION (live)"], results["4-MONTH HRST"]
for label, key in [("closed trades", "n"), ("wins / losses", "wl"), ("compounded net %", "compound"),
                   ("sum per-trade $@14300", "sum_usd"), ("rally-cd buys blocked", "blocked"),
                   ("max_hold forced sells", "forced")]:
    if key == "n":
        pv, fv = len(P["trades"]), len(F["trades"])
    elif key == "wl":
        pv, fv = f"{P['wins']}/{len(P['trades'])-P['wins']}", f"{F['wins']}/{len(F['trades'])-F['wins']}"
    elif key == "compound":
        pv, fv = f"{P['compound']:+.6f}", f"{F['compound']:+.6f}"
    elif key == "sum_usd":
        pv, fv = f"{P['sum_usd']:+.4f}", f"{F['sum_usd']:+.4f}"
    else:
        pv, fv = P[key], F[key]
    print(f"  {label:<28} {str(pv):>22} {str(fv):>22}")
for R in (P, F):
    o = R["open_t"]
    print(f"  open at end {R['name']:<16}: " + (f"{o['net']*100:+.6f}% (${NOTIONAL*o['net']:+.4f}) buy@{o['px']!r}" if o else "flat"))
print("#" * 104)
