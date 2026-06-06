"""
dryrun_1_4h.py — SIMULTANEOUS forward DRY-RUN of ETH 1h / 2h / 3h / 4h models.
================================================================================
Paper-trades all four short horizons at once, hourly, using the LIVE inference path
(compute_signal_core: ffill, closed-bar, no-embargo) — NOT the backtest engine. This is
the leak-free forward test: it captures each decision's data at decision time, so there is
NO data revision and NO backtest overstatement. Use it to see whether the 1-3h backtest
leakage (97% WR) is real or evaporates live.

NO REAL TRADES. Logs only. Each horizon keeps an independent paper position (cash/long).

Per cycle (each hour, a few min after the candle closes):
  - refresh ETH OHLCV
  - for h in 1,2,3,4: build features (pysr from models_faye), train RF/LGBM on the last
    `window` labeled rows, predict the last CLOSED bar via compute_signal_core, gate on conf,
    update the paper position (BUY if conf>=gate & flat; SELL on SELL or max_hold).
  - append a row to output/dryrun_1_4h/dryrun_{h}h.csv and print a 4-horizon summary.

Model spec per horizon is auto-read from the latest models_faye/mode_d_full_ETH_{h}h_*.csv
(the DV top config: combo/window/gamma/n_features); features are ranked top-N by LGBM
importance once at startup and held fixed for the session (mirrors live's fixed features).
Edit SPEC_OVERRIDE / CONF_GATE / MAX_HOLD below to taste.

Run (venv active, from engine root):  python tools/dryrun_1_4h.py
  Ctrl-C to stop. Re-running resumes the CSVs (appends) but resets paper positions to cash.
"""
import os
os.environ.setdefault('FAYE_MODELS_DIR', 'models_faye')   # 1-4h PySR live here (Mode P output)
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
import sys
import csv as _csv
import glob
import time
import traceback
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import crypto_live_trader_ed as LT          # build_all_features / _compute_pysr_features / ALL_MODELS / load_data
import crypto_signal_core as CORE           # compute_signal_core (the live inference core)

ASSET = "ETH"
HORIZONS = [1, 2, 3, 4]
CONF_GATE = {1: 80, 2: 80, 3: 80, 4: 75}    # per-horizon BUY confidence gate (edit me)
MAX_HOLD = 10                               # force SELL after N hours in a position
FEE = 0.0011                                # round-trip-ish cost per leg (TRADING_FEE); paper only
OUTDIR = "output/dryrun_1_4h"
SPEC_OVERRIDE = {}                          # e.g. {1: dict(combo=['RF','LGBM'], window=150, gamma=0.999, n=15)}
DEFAULT_SPEC = dict(combo=["RF", "LGBM"], window=150, gamma=0.999, n=15)
os.makedirs(OUTDIR, exist_ok=True)


def _keep_awake():
    """Prevent Windows sleep / Modern Standby for the life of the dry run (mirror of FAYE
    sleep-guard, crypto_trading_system_faye.py:412). ES_CONTINUOUS holds the request until the
    process exits; atexit clears it. ES_AWAYMODE_REQUIRED blocks S0 idle (Modern Standby) which
    ignores powercfg timeouts; fall back without it on older Windows. Returns True if engaged."""
    try:
        import ctypes
        import atexit
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_AWAYMODE_REQUIRED = 0x80000000, 0x00000001, 0x00000040
        k32 = ctypes.windll.kernel32
        prev = k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
        if prev == 0:
            prev = k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        if prev != 0:
            atexit.register(lambda: k32.SetThreadExecutionState(ES_CONTINUOUS))
            return True
    except Exception:
        pass  # non-Windows / insufficient privileges — system may sleep
    return False


def read_dv_spec(h):
    """Auto-read combo/window/gamma/n_features from the latest mode_d_full grid for horizon h."""
    if h in SPEC_OVERRIDE:
        return {**DEFAULT_SPEC, **SPEC_OVERRIDE[h]}
    files = sorted(glob.glob(f"models_faye/mode_d_full_ETH_{h}h_*.csv"))
    if not files:
        return dict(DEFAULT_SPEC)
    rows = [r for r in _csv.DictReader(open(files[-1]))]
    rows = [r for r in rows if r.get("return_pct") not in (None, "", "nan")]
    if not rows:
        return dict(DEFAULT_SPEC)
    rows.sort(key=lambda r: float(r["return_pct"]), reverse=True)
    top = rows[0]
    return dict(combo=top["combo"].split("+"), window=int(top["window"]),
                gamma=float(top["gamma"]), n=int(top["n_features"]))


def build(h, df_raw):
    """Build features + pysr for horizon h on a recent slice; return (df_full, feature_cols)."""
    f, c = LT.build_all_features(df_raw.tail(3000).reset_index(drop=True), asset_name=ASSET,
                                 horizon=h, verbose=False, keep_label_nan_tail=True)
    LT._compute_pysr_features(f, c, ASSET, h, verbose=False)
    return f, c


def rank_features(f, cols, window, n):
    """Top-N features by LGBM importance on the last (window+300) labeled rows; ensure >=1 pysr."""
    sub = f.dropna(subset=["label"]).tail(window + 300).copy()
    X = sub[cols].ffill().fillna(0.0).values
    y = sub["label"].values
    m = lgb.LGBMClassifier(n_estimators=120, max_depth=4, learning_rate=0.05,
                           random_state=42, n_jobs=1, verbose=-1)
    m.fit(X, y)
    ranked = [c for c, _ in sorted(zip(cols, m.feature_importances_), key=lambda t: -t[1])]
    top = ranked[:n]
    pys = [c for c in cols if c.startswith("pysr_")]
    if pys and not any(c.startswith("pysr_") for c in top):
        top[-1] = pys[0]
    return top


def closed_bar_idx(f):
    """Index of the last FULLY-CLOSED bar (mirror of live fix #2)."""
    i = len(f) - 1
    try:
        last = pd.to_datetime(f["datetime"].iloc[i])
        if last.tzinfo is None:
            last = last.tz_localize("UTC")
        if (last + pd.Timedelta(hours=1)) > pd.Timestamp.now(tz="UTC"):
            i -= 1
    except Exception:
        pass
    return i


def live_signal(h, spec, feats, f, cols):
    fc = [x for x in feats if x in cols]
    if not fc:
        return None
    dtr = f.dropna(subset=["label"]).reset_index(drop=True)
    df = f.reset_index(drop=True).copy()
    df[fc] = df[fc].ffill().fillna(0.0)
    dtr[fc] = dtr[fc].ffill().fillna(0.0)
    if len(dtr) < spec["window"] + 50:
        return None
    i = closed_bar_idx(df)
    ts = max(0, len(dtr) - spec["window"])
    res = CORE.compute_signal_core(X_train=dtr.iloc[ts:][fc].values, y_train=dtr.iloc[ts:]["label"].values,
                                   X_test=df.iloc[i:i + 1][fc].values,
                                   model_factories={n: LT.ALL_MODELS[n] for n in spec["combo"]},
                                   gamma=spec["gamma"], na_policy="ffill", return_probas=True, binary_signal=False)
    if not res or res.get("signal") is None:
        return None
    bar_dt = str(df["datetime"].iloc[i])      # inference bar = last CLOSED bar (mirrors live fix #2)
    # EXECUTION price = latest available price (current/forming bar), NOT the closed-bar close.
    # The live trader infers on the closed bar but FILLS at the current market price. Using
    # close[i] would book a fill at a stale price whenever a cycle runs mid-bar (e.g. the
    # immediate cycle-0 launch ran 42 min after the last bar closed -> artificially cheap entry).
    # close[-1] is the freshest price; for aligned :02 cycles it ~equals close[i] anyway.
    exec_px = float(df["close"].iloc[-1])
    return res["signal"], float(res["confidence"]), bar_dt, exec_px


# paper position per horizon
POS = {h: dict(state="cash", entry_px=0.0, entry_t=None, entry_i=0, equity=1.0, trades=0, wins=0) for h in HORIZONS}
SPEC = {}
FEATS = {}


def log_row(h, ts, bar_dt, px, sig, conf, action, pnl):
    p = OUTDIR + f"/dryrun_{h}h.csv"
    new = not os.path.exists(p)
    with open(p, "a", newline="") as fh:
        w = _csv.writer(fh)
        if new:
            w.writerow(["logged_utc", "inference_bar", "price", "signal", "confidence",
                        "action", "state", "equity", "trade_pnl_pct", "trades", "wins"])
        pos = POS[h]
        w.writerow([ts, bar_dt, f"{px:.4f}", sig, f"{conf:.2f}", action, pos["state"],
                    f"{pos['equity']:.6f}", ("" if pnl is None else f"{pnl*100:.4f}"),
                    pos["trades"], pos["wins"]])


def cycle(cyc_i):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    try:
        LT.download_asset(ASSET, update_only=True)
    except Exception:
        pass
    df_raw = LT.load_data(ASSET)
    if df_raw is None:
        print(f"  [{ts}] no data — skip")
        return
    line = [f"[{ts}] DRY-RUN cycle {cyc_i}"]
    for h in HORIZONS:
        try:
            f, cols = build(h, df_raw)
            sc = live_signal(h, SPEC[h], FEATS[h], f, cols)
            if sc is None:
                line.append(f"  {h}h: (no signal)")
                continue
            sig, conf, bar_dt, px = sc
            pos = POS[h]
            action = "HOLD"
            pnl = None
            if pos["state"] == "long":
                held = 0
                try:
                    held = int((pd.to_datetime(bar_dt) - pd.to_datetime(pos["entry_t"])).total_seconds() // 3600)
                except Exception:
                    pass
                if sig == "SELL" or held >= MAX_HOLD:
                    pnl = (1 - FEE) ** 2 * (px / pos["entry_px"]) - 1
                    pos["equity"] *= (1 + pnl)
                    pos["trades"] += 1
                    pos["wins"] += int(pnl > 0)
                    pos["state"] = "cash"
                    action = "SELL" + ("(max_hold)" if sig != "SELL" else "")
            elif sig == "BUY" and conf >= CONF_GATE[h]:
                pos["state"] = "long"
                pos["entry_px"] = px
                pos["entry_t"] = bar_dt
                action = "BUY"
            log_row(h, ts, bar_dt, px, sig, conf, action, pnl)
            tag = "LONG" if pos["state"] == "long" else "cash"
            line.append(f"  {h}h: {sig}({conf:.0f}%)->{action:<13} {tag} eq={pos['equity']:.4f} "
                        f"({pos['wins']}/{pos['trades']}W)")
        except Exception as e:
            line.append(f"  {h}h: ERROR {type(e).__name__}: {e}")
    print("\n".join(line), flush=True)


def main():
    awake = _keep_awake()
    print("=" * 80)
    print("  DRY-RUN 1/2/3/4h — forward paper-trade (live core, NO real trades)")
    print(f"  sleep-guard: {'ON — system kept awake (display may still sleep)' if awake else 'OFF (failed/non-Windows; system may sleep)'}")
    print("=" * 80)
    df_raw = LT.load_data(ASSET)
    for h in HORIZONS:
        SPEC[h] = read_dv_spec(h)
        f, cols = build(h, df_raw)
        FEATS[h] = rank_features(f, cols, SPEC[h]["window"], SPEC[h]["n"])
        print(f"  {h}h: {'+'.join(SPEC[h]['combo'])} w={SPEC[h]['window']} g={SPEC[h]['gamma']} "
              f"n={len(FEATS[h])} gate={CONF_GATE[h]}% | feats={','.join(FEATS[h][:6])}...")
    print(f"\n  logging to {OUTDIR}/dryrun_{{1,2,3,4}}h.csv | aligned to :02 past each hour. Ctrl-C to stop.\n")
    cyc = 0
    # run one cycle immediately, then hourly at :02
    cycle(cyc)
    if "--once" in sys.argv:
        print("\n  --once: single cycle done. (Run without --once for the continuous hourly loop.)")
        return
    while True:
        now = datetime.now(timezone.utc)
        nxt = now.replace(minute=2, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(hours=1)
        time.sleep(max(5, (nxt - now).total_seconds()))
        cyc += 1
        cycle(cyc)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  stopped. Per-horizon paper results are in", OUTDIR)
