"""
replay_4mo_thisweek.py — faithful per-trade replay of the 4-MONTH-HRST ETH config
over the last 168h. Prints every BUY/SELL with raw prices + exact per-trade PnL.

AUTHORITATIVE 4mo config (from logs/ed_v1_20260604_075223.log + config_faye/regime_config_faye.json):
  detector tsmom_672h | bull 5h@70 / bear 5h@65 | shields OFF | max_hold=10h | min_sell_pnl=0
  5h WINNER (log "D #5"): RF+LGBM w=250 g=0.999 f=10  (4mo return +70.01%, WR74%, 222 trades)
  rally-cooldown bull: rr12h>=2.0 OR rr30h>=4.0 cd=6h ; bear: rr20h>=7.0 OR rr36h>=9.0 cd=12h
NOTE: models_faye/crypto_faye_production_noprod.csv was OVERWRITTEN 2026-06-05 20:14 with the
am1 recent-month models, so the model spec below is HARDCODED from the authoritative sources
(not read from that file). The 10 feature names came from a direct read of the CSV while it
still held the 4mo winner (w=250 g=0.999 f=10 return=70.01 matched the log's D#5 exactly).

Gates applied (exact, replicating crypto_revolut_ed_v2.py):
  - conf gate: BUY blocked if conf < min_conf(regime)  [bull 70 / bear 65]
  - rally-cooldown: BUY blocked while cooldown active. Trigger at bar i if
      (close[i]/close[i-h_short]-1)*100 >= t_short  OR  (.../close[i-h_long]-1)*100 >= t_long
    -> cooldown until bar i + cd_hours (>= comparison; uses the ACTIVE regime's gate).
  - max_hold: force SELL when held >= 10h.
Engine: crypto_trading_system_faye generate_signals (FAYE near-live, embargo=horizon=5, step=1).
Deterministic (GPU LGBM reproducible, random_state=42). No rounding of stored floats.

Run:  python tools/replay_4mo_thisweek.py            # 168h (1 week)
"""
import os
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('FAYE_MODELS_DIR', 'models')  # PySR from live models/ (pysr_ETH_5h)
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crypto_trading_system_faye import generate_signals, BACKTEST_FEE_PER_LEG  # noqa: E402

ASSET = "ETH"
REPLAY = int(sys.argv[1]) if len(sys.argv) > 1 else 168
FEE = BACKTEST_FEE_PER_LEG
NOTIONAL = 14300.0
MAX_HOLD = 10

# --- HARDCODED 4mo 5h winner (D#5) ---
COMBO = ["RF", "LGBM"]
WINDOW = 250
GAMMA = 0.999
FEATS = ["deriv_basis", "pysr_5", "vol_ratio_12_48", "logret_240h", "adx_14h",
         "price_accel_12h", "logret_72h", "price_to_sma100h", "deriv_funding_zscore", "pysr_3"]
CONF = {"bull": 70, "bear": 65}
RC = {"bull": dict(h_short=12, h_long=30, t_short=2.0, t_long=4.0, cd=6),
      "bear": dict(h_short=20, h_long=36, t_short=7.0, t_long=9.0, cd=12)}

sigs = generate_signals(asset_name=ASSET, model_names=COMBO, window_size=WINDOW,
                        replay_hours=REPLAY, feature_override=FEATS, horizon=5, gamma=GAMMA)
sig_by_dt = {s["datetime"]: s for s in sigs}

hist = pd.read_csv("data/eth_hourly_data.csv", parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
close = hist["close"].values
key = hist["datetime"].dt.strftime("%Y-%m-%d %H:%M").values
idx_of = {key[i]: i for i in range(len(key))}

dts = sorted(sig_by_dt)
print("=" * 104)
print(f"  4-MONTH CONFIG FAITHFUL REPLAY — {ASSET}, last {REPLAY}h")
print(f"  model=RF+LGBM w={WINDOW} g={GAMMA} f={len(FEATS)}  conf bull{CONF['bull']}/bear{CONF['bear']}  max_hold={MAX_HOLD}h  maker fee={FEE*100:.4f}%/leg")
print(f"  features: {','.join(FEATS)}")
print(f"  signal window: {dts[0]}  ->  {dts[-1]}  ({len(dts)} hourly signals)")
print("=" * 104)

cooldown_until = -1   # bar index until which BUY is blocked (active if i < cooldown_until)
in_pos = False
entry = None
trades = []
n_bull = n_bear = 0
rc_triggers = 0
buys_blocked_cd = 0
forced_sells = 0

for dt in dts:
    i = idx_of[dt]
    is_bull = bool(np.log(close[i] / close[i - 672]) > 0) if i >= 672 else False
    reg = "bull" if is_bull else "bear"
    n_bull += is_bull
    n_bear += (not is_bull)
    g = RC[reg]
    rr_s = (close[i] / close[i - g["h_short"]] - 1.0) * 100.0
    rr_l = (close[i] / close[i - g["h_long"]] - 1.0) * 100.0
    if rr_s >= g["t_short"] or rr_l >= g["t_long"]:
        cooldown_until = max(cooldown_until, i + g["cd"])
        rc_triggers += 1
    cd_active = i < cooldown_until

    s = sig_by_dt[dt]
    sig = s["signal"]
    conf = s["confidence"]
    px = float(s["close"])

    if in_pos:
        held = i - entry["i"]
        reason = None
        if sig == "SELL":
            reason = "signal"
        elif held >= MAX_HOLD:
            reason = "max_hold"
            forced_sells += 1
        if reason:
            net = (1.0 - FEE) * (1.0 - FEE) * (px / entry["px"]) - 1.0
            gross = px / entry["px"] - 1.0
            trades.append({**entry, "exit_dt": dt, "exit_px": px, "exit_conf": float(conf),
                           "gross": gross, "net": net, "hold_h": float(held), "reason": reason})
            in_pos = False
            entry = None
    elif sig == "BUY":
        mc = CONF[reg]
        if conf is None or conf < mc:
            pass  # conf-gated
        elif cd_active:
            buys_blocked_cd += 1  # rally-cooldown blocked this BUY
        else:
            in_pos = True
            entry = {"dt": dt, "i": i, "px": px, "regime": reg, "conf": float(conf)}

equity = 1.0
print(f"\n  {'#':>2}  {'BUY (UTC)':<16} {'buy_px':>11} {'reg':>4} {'cf':>4}  {'SELL (UTC)':<16} {'sell_px':>11} "
      f"{'cf':>4} {'h':>3} {'why':>8} {'gross_%':>13} {'net_%':>13} {'net_$@14300':>13}")
for n, t in enumerate(trades, 1):
    equity *= (1.0 + t["net"])
    print(f"  {n:>2}  {t['dt']:<16} {t['px']!r:>11} {t['regime']:>4} {t['conf']:>4.0f}  "
          f"{t['exit_dt']:<16} {t['exit_px']!r:>11} {t['exit_conf']:>4.0f} {t['hold_h']:>3.0f} {t['reason']:>8} "
          f"{t['gross']*100:>+13.8f} {t['net']*100:>+13.8f} {NOTIONAL*t['net']:>+13.6f}")
print("=" * 104)
nwin = sum(1 for t in trades if t["net"] > 0)
print(f"  REGIME this week: bull={n_bull}h  bear={n_bear}h")
print(f"  rally-cooldown TRIGGERS: {rc_triggers}  | BUYs blocked by cooldown: {buys_blocked_cd}  | max_hold forced SELLs: {forced_sells}")
print(f"  CLOSED TRADES: {len(trades)}  | wins: {nwin}  | losses: {len(trades)-nwin}")
print(f"  COMPOUNDED net return = {(equity-1.0)*100:+.8f} %   (equity factor {equity!r})")
print(f"  SUM per-trade net%    = {sum(t['net'] for t in trades)*100:+.8f} %")
print(f"  SUM per-trade $@14300 = {sum(NOTIONAL*t['net'] for t in trades):+.6f}")
if in_pos and entry is not None:
    lp = float(sig_by_dt[dts[-1]]["close"])
    on = (1.0 - FEE) * (1.0 - FEE) * (lp / entry["px"]) - 1.0
    print(f"  [OPEN at end] BUY {entry['dt']} @ {entry['px']!r} ({entry['regime']}) mark {dts[-1]} @ {lp!r} "
          f"-> unrealized net {on*100:+.8f}% (${NOTIONAL*on:+.6f}) — NOT in totals")
else:
    print("  [flat at end of window]")
print("=" * 104)
