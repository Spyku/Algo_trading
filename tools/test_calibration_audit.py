"""test_calibration_audit.py — Audit LGBM probability calibration on current
production ETH 6h model. READ-ONLY — does not modify any production file or
write any tagged output that could collide with other tests (fracdiff etc).

What it does:
  1. Reads current ETH 6h prod row from crypto_ed_production.csv
  2. Trains the same LGBM (same combo, window, gamma, features)
  3. Splits the training window 80/20 chronologically
  4. Trains LGBM on first 80%, predicts on last 20% (held-out)
  5. Computes calibration curve (binned hit rates vs raw scores)
  6. Computes calibration AFTER Platt scaling and AFTER isotonic regression
  7. Reports: Brier score before/after, calibration table, max gap

Output:
  - Console table
  - logs/calibration_audit_<ts>.txt (saved summary)

Decision rule:
  - max gap < 5pp -> already calibrated, skip Step 2
  - max gap 5-10pp -> marginal, optional Step 2
  - max gap > 10pp -> miscalibrated, run Step 2 smoke test

Safe to run concurrently with fracdiff Mode D — no shared output paths.

Usage:
  python tools/test_calibration_audit.py
  python tools/test_calibration_audit.py --asset ETH --horizon 6
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ENGINE)
sys.path.insert(0, ENGINE)
warnings.filterwarnings('ignore')

PROD_CSV = os.path.join(ENGINE, 'models', 'crypto_ed_production.csv')


def load_prod_row(asset: str, horizon: int):
    df = pd.read_csv(PROD_CSV)
    rows = df[(df['coin'] == asset) & (df['horizon'] == horizon)]
    if len(rows) == 0:
        raise ValueError(f"No prod row for {asset} {horizon}h")
    return rows.sort_values('combined_score', ascending=False).iloc[0]


def build_dataset(asset: str, horizon: int, feature_override: list, window_size: int,
                  total_hours: int = 1440):
    """Replicates the engine's preprocessing pipeline for ONE training window."""
    from crypto_trading_system_ed import load_data, _build_features
    df_raw = load_data(asset)
    if df_raw is None:
        raise RuntimeError(f"Could not load {asset} hourly data")
    df_features, feature_cols = _build_features(df_raw, asset, feature_override=feature_override,
                                                horizon=horizon)
    n = len(df_features)
    # Take the most recent `total_hours` for analysis
    end = n - 1
    start = max(0, end - total_hours)
    return df_features.iloc[start:end+1].reset_index(drop=True), feature_cols


def calibration_table(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10):
    """Compute reliability diagram bins. Returns list of dicts."""
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({'bucket': f"{lo*100:.0f}-{hi*100:.0f}%", 'n': 0,
                         'avg_score': float('nan'), 'hit_rate': float('nan'), 'gap': float('nan')})
            continue
        avg_score = float(scores[mask].mean())
        hit_rate  = float(y_true[mask].mean())
        rows.append({'bucket': f"{lo*100:.0f}-{hi*100:.0f}%", 'n': n,
                     'avg_score': avg_score, 'hit_rate': hit_rate,
                     'gap': hit_rate - avg_score})
    return rows


def brier(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean((scores - y_true) ** 2))


def fit_combo_and_predict(X_train, y_train, X_test, sw, model_names: list):
    """Replicates the engine's per-cycle fit+predict loop. Returns averaged
    proba[1] across models."""
    from crypto_trading_system_ed import ALL_MODELS
    probas_per_model = []
    for mn in model_names:
        try:
            m = ALL_MODELS[mn]()
            m.fit(X_train, y_train, sample_weight=sw)
            p = m.predict_proba(X_test)[:, 1]
            probas_per_model.append(p)
        except Exception as e:
            print(f"  [warn] {mn} fit/predict failed: {e}")
            continue
    if not probas_per_model:
        return None
    return np.mean(np.stack(probas_per_model, axis=0), axis=0)


def fit_combo_with_calibration(X_train, y_train, X_test, sw, model_names: list,
                                method: str = 'platt'):
    """Train each model on first 80% of training data, fit calibrator on last 20%
    of training data using held-out predictions, refit each on full 100%, then
    apply calibration mapping at inference.

    Returns: (raw_avg_proba_test, calibrated_avg_proba_test).
    """
    from crypto_trading_system_ed import ALL_MODELS
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    n = len(X_train)
    split = int(0.8 * n)
    X_pre, y_pre, sw_pre = X_train.iloc[:split], y_train[:split], sw[:split]
    X_cal, y_cal         = X_train.iloc[split:], y_train[split:]

    raw_test, cal_test = [], []

    for mn in model_names:
        # 1) Pre-fit on first 80%
        m_pre = ALL_MODELS[mn]()
        m_pre.fit(X_pre, y_pre, sample_weight=sw_pre)
        cal_scores = m_pre.predict_proba(X_cal)[:, 1]

        # 2) Fit calibrator on held-out 20%
        if method == 'platt':
            calibrator = LogisticRegression()
            calibrator.fit(cal_scores.reshape(-1, 1), y_cal)
            mapper = lambda s: calibrator.predict_proba(s.reshape(-1, 1))[:, 1]
        elif method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(cal_scores, y_cal)
            mapper = lambda s: calibrator.predict(s)
        else:
            raise ValueError(f"unknown method: {method}")

        # 3) Refit on full 100%
        m_full = ALL_MODELS[mn]()
        m_full.fit(X_train, y_train, sample_weight=sw)
        raw = m_full.predict_proba(X_test)[:, 1]
        cal = mapper(raw)
        raw_test.append(raw)
        cal_test.append(cal)

    raw_avg = np.mean(np.stack(raw_test, axis=0), axis=0)
    cal_avg = np.mean(np.stack(cal_test, axis=0), axis=0)
    return raw_avg, cal_avg


def main():
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', default='ETH')
    ap.add_argument('--horizon', type=int, default=6)
    ap.add_argument('--folds', type=int, default=20,
                    help="Number of walk-forward cycles to evaluate (default 20). "
                         "More folds = more stable estimate.")
    args = ap.parse_args()

    print("=" * 100)
    print(f"  CALIBRATION AUDIT — {args.asset} {args.horizon}h, current production model")
    print("=" * 100)

    row = load_prod_row(args.asset, args.horizon)
    combo = str(row['models']).split('+')
    window = int(row['best_window'])
    gamma = float(row['gamma'])
    feats = [f.strip() for f in str(row['optimal_features']).split(',') if f.strip()]
    print(f"  Prod config: combo={'+'.join(combo)}  window={window}h  gamma={gamma}  "
          f"n_features={len(feats)}  reported_return={row.get('return_pct', 'NA')}%")

    print(f"\n  Loading data + building features (this calls the unmodified engine)...")
    df, feature_cols = build_dataset(args.asset, args.horizon, feats, window)
    print(f"  Features built: {len(df)} rows, {len(feature_cols)} feature columns")

    # Walk-forward: collect raw vs calibrated scores across the last `folds` cycles
    from crypto_trading_system_ed import get_decay_weights
    n = len(df)
    folds_y, folds_raw, folds_platt, folds_iso = [], [], [], []

    eval_start = n - args.folds - args.horizon
    print(f"\n  Running {args.folds} walk-forward cycles "
          f"(rows {eval_start}..{n-args.horizon-1})...")

    for k, i in enumerate(range(eval_start, n - args.horizon)):
        train_start = max(0, i - window)
        train_end   = max(train_start, i - args.horizon)
        train = df.iloc[train_start:train_end]
        X_train = train[feature_cols]
        y_train = train['label'].values
        X_test  = df.iloc[i:i+1][feature_cols]
        y_test  = df.iloc[i]['label']

        if len(np.unique(y_train)) < 2:
            continue
        if X_train.isnull().any().any() or X_test.isnull().any().any():
            continue

        # Same scaling as the engine
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
        X_te = pd.DataFrame(scaler.transform(X_test),     columns=feature_cols, index=X_test.index)
        sw = get_decay_weights(len(y_train), gamma)

        # Raw
        raw_avg = fit_combo_and_predict(X_tr, y_train, X_te, sw, combo)
        if raw_avg is None:
            continue

        # Platt calibrated
        _, platt_avg = fit_combo_with_calibration(X_tr, y_train, X_te, sw, combo, method='platt')
        # Isotonic calibrated
        _, iso_avg   = fit_combo_with_calibration(X_tr, y_train, X_te, sw, combo, method='isotonic')

        folds_y.append(int(y_test))
        folds_raw.append(float(raw_avg[0]))
        folds_platt.append(float(platt_avg[0]))
        folds_iso.append(float(iso_avg[0]))

        if (k + 1) % 5 == 0:
            print(f"    [{k+1}/{args.folds}] raw={raw_avg[0]:.3f}  "
                  f"platt={platt_avg[0]:.3f}  iso={iso_avg[0]:.3f}  y={int(y_test)}")

    if not folds_y:
        print("  ERROR: no valid folds")
        return

    y = np.array(folds_y)
    raw = np.array(folds_raw)
    platt = np.array(folds_platt)
    iso = np.array(folds_iso)

    n_pos = int(y.sum())
    base_rate = float(y.mean())
    print(f"\n  Collected {len(y)} valid cycles. Positives: {n_pos}/{len(y)} ({base_rate*100:.1f}%)")

    print(f"\n  Brier score (lower = better):")
    print(f"    Raw          {brier(y, raw):.4f}")
    print(f"    Platt        {brier(y, platt):.4f}")
    print(f"    Isotonic     {brier(y, iso):.4f}")

    print(f"\n  Reliability table (raw scores vs actual hit rate):")
    print(f"  {'Bucket':<14}{'n':>6}{'Avg score':>12}{'Hit rate':>11}{'Gap':>9}")
    for r in calibration_table(y, raw, n_bins=10):
        if r['n'] == 0:
            continue
        gap_str = f"{r['gap']*100:+.1f}pp"
        print(f"  {r['bucket']:<14}{r['n']:>6}{r['avg_score']:>12.3f}"
              f"{r['hit_rate']:>11.3f}{gap_str:>9}")

    print(f"\n  Reliability table after PLATT scaling:")
    print(f"  {'Bucket':<14}{'n':>6}{'Avg score':>12}{'Hit rate':>11}{'Gap':>9}")
    for r in calibration_table(y, platt, n_bins=10):
        if r['n'] == 0:
            continue
        gap_str = f"{r['gap']*100:+.1f}pp"
        print(f"  {r['bucket']:<14}{r['n']:>6}{r['avg_score']:>12.3f}"
              f"{r['hit_rate']:>11.3f}{gap_str:>9}")

    raw_gaps = [abs(r['gap']) for r in calibration_table(y, raw) if r['n'] > 0]
    platt_gaps = [abs(r['gap']) for r in calibration_table(y, platt) if r['n'] > 0]
    iso_gaps = [abs(r['gap']) for r in calibration_table(y, iso) if r['n'] > 0]

    raw_max = max(raw_gaps) * 100 if raw_gaps else 0
    platt_max = max(platt_gaps) * 100 if platt_gaps else 0
    iso_max = max(iso_gaps) * 100 if iso_gaps else 0
    raw_avg_gap = np.mean(raw_gaps) * 100 if raw_gaps else 0
    platt_avg_gap = np.mean(platt_gaps) * 100 if platt_gaps else 0
    iso_avg_gap = np.mean(iso_gaps) * 100 if iso_gaps else 0

    print(f"\n  Calibration gap summary:")
    print(f"    Raw:      max gap {raw_max:.1f}pp,  avg gap {raw_avg_gap:.1f}pp")
    print(f"    Platt:    max gap {platt_max:.1f}pp,  avg gap {platt_avg_gap:.1f}pp")
    print(f"    Isotonic: max gap {iso_max:.1f}pp,  avg gap {iso_avg_gap:.1f}pp")

    print(f"\n  --- VERDICT ---")
    if raw_max < 5:
        print(f"  RAW IS WELL-CALIBRATED (max gap {raw_max:.1f}pp < 5pp).")
        print(f"  Calibration not needed. Skip Idea #2 Step 2.")
    elif raw_max < 10:
        print(f"  RAW IS MARGINALLY MISCALIBRATED (max gap {raw_max:.1f}pp).")
        print(f"  Calibration may help slightly. Step 2 (Mode D smoke test) optional.")
    else:
        print(f"  RAW IS MISCALIBRATED (max gap {raw_max:.1f}pp >= 10pp).")
        print(f"  Calibration likely worthwhile. Run Step 2: tools/test_calibration_mode_d.py")
        improvement = raw_max - min(platt_max, iso_max)
        winner = 'Platt' if platt_max <= iso_max else 'Isotonic'
        print(f"  {winner} reduces max gap by {improvement:.1f}pp.")

    # Save summary
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(ENGINE, 'logs', f'calibration_audit_{args.asset}_{args.horizon}h_{ts}.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Calibration audit: {args.asset} {args.horizon}h\n")
        f.write(f"Folds: {len(y)}, Positives: {n_pos}/{len(y)} ({base_rate*100:.1f}%)\n")
        f.write(f"Brier raw={brier(y, raw):.4f}  platt={brier(y, platt):.4f}  "
                f"iso={brier(y, iso):.4f}\n")
        f.write(f"Max gap raw={raw_max:.1f}pp  platt={platt_max:.1f}pp  iso={iso_max:.1f}pp\n")
    print(f"\n  Summary saved: {out_path}")


if __name__ == '__main__':
    main()
