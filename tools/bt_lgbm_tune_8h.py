"""
bt_lgbm_tune_8h.py — REAL-engine backtest of LGBM regularization on ETH 8h.
=========================================================================
Tests, through generate_signals (the PRODUCTION model path — ALL_MODELS, n_est=300
GPU, hourly walk-forward) + the maker-fee regime sim, three arms:

  PROD : RF+LGBM   (the LIVE 8h incumbent)     current LGBM (min_child20 / reg_lambda0)
  A    : GB+LGBM   (today's new combo)         current LGBM (min_child20 / reg_lambda0)
  B    : GB+LGBM                               TUNED   LGBM (min_child30 / reg_lambda5)

Fixed 8h context from models/crypto_ed_production.csv (window, gamma, 24 features).
LGBM params are injected via the default-safe LGBM_MIN_CHILD / LGBM_REG_LAMBDA env hook
(_lgbm_hyperparam_overrides in the engine) — read at model-construction time, so each arm
gets its own params in one process. lr0.05 / depth4 / sub1.0 are unchanged across arms.

Regime (config/regime_config_ed.json, ETH): detector tsmom_672h, bull 8h @ conf65
(shield OFF), bear 8h @ conf70 (shield ON). Maker fee = BACKTEST_FEE_PER_LEG.

CAVEATS:
  * hold-shield NOT modeled (bear exits are signal-only here). It applies equally to all
    3 arms, so the RELATIVE ranking holds; bear absolute returns are approximate.
  * Both regimes use horizon 8 (same model), so one generate_signals run per arm; the
    regime only switches the confidence gate (65 bull / 70 bear).
  * Short windows = noise (336h ~5-15 trades). Read the 4320h / 1440h rows as the signal.

Run:  python tools/bt_lgbm_tune_8h.py                       # replay 4320, windows 4320 1440 720 336
      python tools/bt_lgbm_tune_8h.py --replay 1440 --windows 1440 720 336
"""
import os
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crypto_trading_system_faye import generate_signals, BACKTEST_FEE_PER_LEG  # noqa: E402

ASSET = "ETH"
HORIZON = 8
FEE = BACKTEST_FEE_PER_LEG
BULL_CONF = 65   # config/regime_config_ed.json ETH bull.min_confidence
BEAR_CONF = 70   # ETH bear.min_confidence

# (label, combo, lgbm_env_overrides)
ARMS = [
    ("PROD  RF+LGBM (cur)",       "RF+LGBM", {}),
    ("A     GB+LGBM (cur)",       "GB+LGBM", {}),
    ("B     GB+LGBM (mc30/reg5)", "GB+LGBM", {"LGBM_MIN_CHILD": "30", "LGBM_REG_LAMBDA": "5"}),
]
_LGBM_ENV_KEYS = ("LGBM_MIN_CHILD", "LGBM_REG_LAMBDA", "LGBM_LR", "LGBM_MAX_DEPTH")


def cfg_8h():
    p = pd.read_csv("models/crypto_ed_production.csv")
    p = p[p["coin"] == ASSET]
    r = p[p["horizon"] == HORIZON].iloc[0]
    feats = [f.strip() for f in str(r["optimal_features"]).split(",") if f.strip() and f.strip() != "nan"]
    return int(r["best_window"]), float(r["gamma"]), feats


def det_map():
    """tsmom_672h: bull when log(close/close[-672h]) > 0."""
    df = pd.read_csv("data/eth_hourly_data.csv", parse_dates=["datetime"]).sort_values("datetime")
    c = df["close"].reset_index(drop=True)
    bull = (np.log(c / c.shift(672)) > 0).values
    return dict(zip(df["datetime"].dt.strftime("%Y-%m-%d %H:%M").values, bull))


def _gate(sig, conf, mc):
    return "HOLD" if (sig == "BUY" and conf is not None and conf < mc) else sig


def sim(sigs, dmap, window_h=None):
    dts = sorted(sigs.keys())
    if window_h:
        cut = pd.Timestamp(dts[-1]) - pd.Timedelta(hours=window_h)
        dts = [d for d in dts if pd.Timestamp(d) >= cut]
    cash, held, in_pos, entry = 1000.0, 0.0, False, 0.0
    trades = wins = 0
    first = last_px = None
    for dt in dts:
        s = sigs[dt]
        px = s["close"]
        last_px = px
        if first is None:
            first = px
        mc = BULL_CONF if dmap.get(dt, False) else BEAR_CONF
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
    if in_pos and last_px:
        cash = held * last_px * (1 - FEE)
        wins += int(last_px > entry)
    ret = (cash / 1000.0 - 1) * 100
    bh = (last_px / first - 1) * 100 if first and last_px else 0.0
    return {"ret": ret, "trades": trades, "wr": (wins / trades * 100) if trades else 0.0, "bh": bh, "n": len(dts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=int, default=4320)
    ap.add_argument("--windows", type=int, nargs="+", default=[4320, 1440, 720, 336])
    args = ap.parse_args()
    windows = [w for w in args.windows if w <= args.replay]

    window, gamma, feats = cfg_8h()
    dmap = det_map()
    print("=" * 92)
    print(f"  REAL-engine backtest — ETH {HORIZON}h LGBM regularization (generate_signals n300 GPU + regime sim)")
    print(f"  context: window={window}  gamma={gamma}  {len(feats)} feats  |  maker {FEE*100:.2f}%/leg  "
          f"|  bull@{BULL_CONF} bear@{BEAR_CONF} (tsmom_672h)")
    print(f"  replay={args.replay}h  windows={windows}  |  hold-shield NOT modeled (relative ranking holds)")
    print("=" * 92)

    res = {}
    for label, combo, env in ARMS:
        for k in _LGBM_ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(env)
        tag = "default" if not env else " ".join(f"{k}={v}" for k, v in env.items())
        print(f"\n  >>> {label}  [{combo}, LGBM {tag}] ...")
        sigs = {s["datetime"]: s for s in generate_signals(
            asset_name=ASSET, model_names=combo.split("+"), window_size=window,
            replay_hours=args.replay, feature_override=feats, horizon=HORIZON, gamma=gamma)}
        res[label] = {w: sim(sigs, dmap, w) for w in windows}
    for k in _LGBM_ENV_KEYS:
        os.environ.pop(k, None)

    print("\n" + "=" * 92 + "\n  RESULT  (ret% / win% / n trades)")
    head = f"  {'Arm':<26}" + "".join(f"{str(w)+'h':>16}" for w in windows)
    print(head + "\n  " + "-" * (len(head) - 2))
    for label, _, _ in ARMS:
        print(f"  {label:<26}" + "".join(
            f"{res[label][w]['ret']:+6.1f}% {res[label][w]['wr']:3.0f}% n{res[label][w]['trades']:>2}".rjust(16) for w in windows))
    base = ARMS[0][0]
    print(f"  {'Buy & Hold':<26}" + "".join(f"{res[base][w]['bh']:+.1f}%".rjust(16) for w in windows))
    print("=" * 92)
    print("  NOTE: hold-shield not modeled (bear exits signal-only; equal across arms). Short windows = noise.")


if __name__ == "__main__":
    main()
