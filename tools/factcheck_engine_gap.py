"""
factcheck_engine_gap.py — MEASURE (not assert) the backtest-vs-live gap on the PRODUCTION
model (5h, the one your live trader actually ran), recent bars where signal_log is ground truth.

For each inference bar B, three confidences for the SAME bar:
  ACTUAL   = signal_log[B+1].conf_2/sig_2     (what your trader really computed)
  LIVECORE = compute_signal_core(infer on B)  (ffill, closed-bar; replay-now)
  BACKTEST = generate_signals[B]              (FAYE near-live: mean_last_10, same-bar)

Reports per-bar + summary: mean|delta| and signal-flip counts for BACKTEST-vs-ACTUAL,
LIVECORE-vs-ACTUAL, BACKTEST-vs-LIVECORE. This is the fact, not a claim.
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
import crypto_live_trader_ed as LT
import crypto_signal_core as CORE
from crypto_trading_system_faye import generate_signals

ASSET, HORIZON = "ETH", 5
N_BARS = 40  # most-recent inference bars to fact-check

# production 5h model
row = next(r for r in _csv.DictReader(open("models/crypto_ed_production.csv")) if r["coin"] == ASSET and r["horizon"] == "5")
COMBO = row["best_combo"].split("+")
WINDOW = int(row["best_window"])
GAMMA = float(row["gamma"])
FEATS = [f.strip() for f in row["optimal_features"].split(",") if f.strip() and f.strip() != "nan"]

hist = LT.load_data(ASSET).sort_values("datetime").reset_index(drop=True)
keyfmt = hist["datetime"].dt.strftime("%Y-%m-%d %H:%M")
idx_at = {keyfmt.iloc[i]: i for i in range(len(hist))}

# ACTUAL live (signal_log): decision D -> inferred on D-1; key by decision time
log = {}
for r in _csv.DictReader(open("config/signal_log.csv")):
    if r["asset"] == ASSET and r.get("sig_2"):
        d = r["timestamp"][:13].replace("T", " ") + ":00"
        log[d] = (r["sig_2"], float(r["conf_2"]))

# BACKTEST: generate_signals, keyed by inference-bar datetime
print("Running generate_signals (backtest engine) on production 5h ...", flush=True)
bt = {}
for s in generate_signals(asset_name=ASSET, model_names=COMBO, window_size=WINDOW, replay_hours=200,
                          feature_override=FEATS, horizon=HORIZON, gamma=GAMMA):
    bt[s["datetime"]] = (s["signal"], float(s["confidence"]))
print(f"  backtest signals: {len(bt)}", flush=True)


def livecore_infer_on(B_key):
    """compute_signal_core inferring on bar B (data up to and including B)."""
    iB = idx_at[B_key]
    dft = hist.iloc[max(0, iB - 2600):iB + 1].reset_index(drop=True)  # includes bar B as last row
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


# inference bars B = the last N bars that have a NEXT decision in the log AND a backtest value
last_i = len(hist) - 1
rows = []
for iB in range(last_i - N_BARS, last_i):
    B = keyfmt.iloc[iB]
    D = keyfmt.iloc[iB + 1]                 # decision hour that inferred on B
    if D not in log or B not in bt:
        continue
    lc = livecore_infer_on(B)
    if lc is None:
        continue
    rows.append((B, D, bt[B], lc, log[D]))

print("\n  inference_bar(B)   BACKTEST       LIVECORE       ACTUAL(live)   |bt-act| |lc-act|")
def f3(t): return f"{t[0]:>4}({t[1]:5.1f})"
da_bt = da_lc = da_bl = 0.0
flip_bt = flip_lc = n = 0
for B, D, b, l, a in rows:
    n += 1
    dba = b[1] - a[1]; dla = l[1] - a[1]; dbl = b[1] - l[1]
    da_bt += abs(dba); da_lc += abs(dla); da_bl += abs(dbl)
    flip_bt += (b[0] != a[0]); flip_lc += (l[0] != a[0])
    print(f"  {B}  {f3(b)}  {f3(l)}  {f3(a)}   {dba:+6.1f}  {dla:+6.1f}")

print("\n" + "=" * 78)
print(f"  bars compared: {n}")
print(f"  BACKTEST vs ACTUAL : mean|conf delta| = {da_bt/max(n,1):5.2f}  | signal flips = {flip_bt}/{n} ({100*flip_bt/max(n,1):.0f}%)")
print(f"  LIVECORE vs ACTUAL : mean|conf delta| = {da_lc/max(n,1):5.2f}  | signal flips = {flip_lc}/{n} ({100*flip_lc/max(n,1):.0f}%)")
print(f"  BACKTEST vs LIVECORE: mean|conf delta| = {da_bl/max(n,1):5.2f}")
print("=" * 78)
print("  Interpretation: if BACKTEST-vs-ACTUAL >> LIVECORE-vs-ACTUAL, the backtest ENGINE is the")
print("  inaccuracy (Layer 1). If both are similar/large, it's data revision (Layer 2), not the engine.")
