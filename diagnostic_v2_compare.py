"""
V1 vs V2 Feature Comparison Diagnostic
========================================
Runs the SAME 75 configs (15 model combos x 5 windows) on:
  - V1: original 20 technical features
  - V2: 101 features (20 base + 81 macro/sentiment/cross-asset)

Compares accuracy head-to-head. Same walk-forward validation.
If V2 doesn't beat V1, we delete it and move on.

Usage:
  python diagnostic_v2_compare.py
"""

import os, sys, time, warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

# Hardware config
try:
    from hardware_config import get_config
    hw = get_config()
    N_JOBS = hw['n_workers']
    print(f"  [{hw['machine']}] {hw['cpu_cores']} cores | {hw.get('gpu_name','no GPU')} | Parallel workers: {N_JOBS}")
except:
    N_JOBS = max(1, os.cpu_count() - 2)


# ============================================================
# MODELS (same as hourly_trading_system.py, CPU-only for parallel)
# ============================================================
ALL_MODELS = {
    'RF':   lambda: RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=1),
    'GB':   lambda: GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
    'LR':   lambda: LogisticRegression(max_iter=1000, random_state=42),
}

# Try adding LGBM (CPU mode for parallel safety)
try:
    import lightgbm as lgb
    ALL_MODELS['LGBM'] = lambda: lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, verbose=-1, n_jobs=1, device='cpu'
    )
except:
    pass

# 15 model combos
MODEL_COMBOS = [
    ['RF'], ['GB'], ['LR'], ['LGBM'],
    ['RF', 'GB'], ['RF', 'LR'], ['RF', 'LGBM'],
    ['GB', 'LR'], ['GB', 'LGBM'], ['LR', 'LGBM'],
    ['RF', 'GB', 'LR'], ['RF', 'GB', 'LGBM'],
    ['RF', 'LR', 'LGBM'], ['GB', 'LR', 'LGBM'],
    ['RF', 'GB', 'LR', 'LGBM'],
]

# Filter combos to only use available models
MODEL_COMBOS = [c for c in MODEL_COMBOS if all(m in ALL_MODELS for m in c)]

# 5 training windows
WINDOWS = [30, 50, 75, 100, 150]

PREDICTION_HORIZON = 4  # same as hourly system


# ============================================================
# WALK-FORWARD EVALUATION (single config)
# ============================================================
def evaluate_config(df, feature_cols, target_col, model_names, window_size, test_fraction=0.3):
    """
    Walk-forward evaluation for one (model_combo, window) config.
    Returns accuracy and sample count.
    """
    n = len(df)
    test_start = int(n * (1 - test_fraction))

    if test_start < window_size + 50:
        return None

    correct = 0
    total = 0

    for i in range(test_start, n):
        train_start = max(0, i - window_size)
        train = df.iloc[train_start:i]
        test_row = df.iloc[i:i+1]

        X_train = train[feature_cols].values
        y_train = train[target_col].values

        if len(np.unique(y_train)) < 2:
            continue

        X_test = test_row[feature_cols].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        votes = []
        for name in model_names:
            try:
                model = ALL_MODELS[name]()
                model.fit(X_train_s, y_train)
                pred = model.predict(X_test_s)[0]
                votes.append(int(pred))
            except:
                pass

        if not votes:
            continue

        ensemble_pred = 1 if sum(votes) > len(votes) / 2 else 0
        actual = int(test_row[target_col].values[0])

        if ensemble_pred == actual:
            correct += 1
        total += 1

    if total == 0:
        return None

    return {
        'models': '+'.join(model_names),
        'window': window_size,
        'accuracy': round(correct / total * 100, 2),
        'samples': total,
        'correct': correct,
    }


def run_grid(df, feature_cols, target_col, label=''):
    """Run all 75 configs in parallel, return results DataFrame."""
    configs = []
    for combo in MODEL_COMBOS:
        for window in WINDOWS:
            configs.append((combo, window))

    print(f"\n  [{label}] Running {len(configs)} configs ({len(feature_cols)} features, {len(df)} rows)...")
    t0 = time.time()

    results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(evaluate_config)(df, feature_cols, target_col, combo, window)
        for combo, window in configs
    )

    # Filter None results
    results = [r for r in results if r is not None]
    elapsed = time.time() - t0

    if not results:
        print(f"  [{label}] No valid results!")
        return None

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('accuracy', ascending=False).reset_index(drop=True)

    best = df_results.iloc[0]
    mean_acc = df_results['accuracy'].mean()
    median_acc = df_results['accuracy'].median()

    print(f"  [{label}] Done in {elapsed:.0f}s")
    print(f"  [{label}] Best:   {best['accuracy']:.1f}% ({best['models']}, w={best['window']}, n={best['samples']})")
    print(f"  [{label}] Mean:   {mean_acc:.1f}%")
    print(f"  [{label}] Median: {median_acc:.1f}%")
    print(f"  [{label}] Top 5:")
    for _, row in df_results.head(5).iterrows():
        print(f"           {row['accuracy']:5.1f}%  {row['models']:25s}  w={row['window']:>3d}  (n={row['samples']})")

    return df_results


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  V1 vs V2 FEATURE COMPARISON — DAX HOURLY")
    print("=" * 60)

    # Import the hourly system
    from hourly_trading_system import load_data, build_hourly_features, update_all_data

    # Update data
    print("\n  Updating DAX data...")
    try:
        update_all_data(['DAX'])
    except:
        pass

    # Load raw data
    df_raw = load_data('DAX')
    if df_raw is None:
        print("  ERROR: Cannot load DAX data")
        return

    # ---- V1: Original features ----
    print("\n" + "=" * 60)
    print("  BUILDING V1 FEATURES (original)")
    print("=" * 60)
    df_v1, cols_v1 = build_hourly_features(df_raw)
    # Drop NaN
    df_v1_clean = df_v1.dropna(subset=cols_v1 + ['label']).reset_index(drop=True)
    print(f"  V1: {len(cols_v1)} features, {len(df_v1_clean)} rows")

    # ---- V2: Macro-enhanced features ----
    print("\n" + "=" * 60)
    print("  BUILDING V2 FEATURES (macro + sentiment + cross-asset)")
    print("=" * 60)
    from features_v2 import build_features_v2_hourly
    df_v2, cols_v2 = build_features_v2_hourly(df_raw, original_builder=build_hourly_features)
    # Drop NaN — V2 has more columns so may have more NaN rows
    df_v2_clean = df_v2.dropna(subset=cols_v2 + ['label']).reset_index(drop=True)
    print(f"  V2: {len(cols_v2)} features, {len(df_v2_clean)} rows")

    # Ensure same test period for fair comparison
    # Use the shorter dataset length (V2 may lose rows due to NaN in macro features)
    min_rows = min(len(df_v1_clean), len(df_v2_clean))
    if len(df_v1_clean) > min_rows:
        df_v1_clean = df_v1_clean.tail(min_rows).reset_index(drop=True)
    if len(df_v2_clean) > min_rows:
        df_v2_clean = df_v2_clean.tail(min_rows).reset_index(drop=True)

    print(f"\n  Aligned to {min_rows} rows for fair comparison")

    # ---- Run grids ----
    print("\n" + "=" * 60)
    print("  RUNNING V1 DIAGNOSTIC GRID")
    print("=" * 60)
    results_v1 = run_grid(df_v1_clean, cols_v1, 'label', label='V1')

    print("\n" + "=" * 60)
    print("  RUNNING V2 DIAGNOSTIC GRID")
    print("=" * 60)
    results_v2 = run_grid(df_v2_clean, cols_v2, 'label', label='V2')

    if results_v1 is None or results_v2 is None:
        print("\n  Cannot compare — one or both grids failed")
        return

    # ---- HEAD-TO-HEAD COMPARISON ----
    print("\n" + "=" * 60)
    print("  HEAD-TO-HEAD COMPARISON")
    print("=" * 60)

    # Merge on (models, window)
    merged = results_v1.merge(results_v2, on=['models', 'window'], suffixes=('_v1', '_v2'))
    merged['diff'] = merged['accuracy_v2'] - merged['accuracy_v1']
    merged = merged.sort_values('diff', ascending=False)

    v2_wins = (merged['diff'] > 0).sum()
    v1_wins = (merged['diff'] < 0).sum()
    ties = (merged['diff'] == 0).sum()
    avg_diff = merged['diff'].mean()

    print(f"\n  V2 wins: {v2_wins} / {len(merged)} configs ({v2_wins/len(merged)*100:.0f}%)")
    print(f"  V1 wins: {v1_wins} / {len(merged)} configs ({v1_wins/len(merged)*100:.0f}%)")
    print(f"  Ties:    {ties}")
    print(f"  Avg accuracy change: {avg_diff:+.2f}%")
    print(f"\n  V1 best: {results_v1.iloc[0]['accuracy']:.1f}%")
    print(f"  V2 best: {results_v2.iloc[0]['accuracy']:.1f}%")
    print(f"  V1 mean: {results_v1['accuracy'].mean():.1f}%")
    print(f"  V2 mean: {results_v2['accuracy'].mean():.1f}%")

    # Show biggest improvements
    print(f"\n  Top 10 configs where V2 improved most:")
    print(f"  {'Models':25s} {'Window':>6s} {'V1':>6s} {'V2':>6s} {'Diff':>7s}")
    print(f"  {'-'*55}")
    for _, row in merged.head(10).iterrows():
        diff_str = f"{row['diff']:+.1f}%"
        color = '+++' if row['diff'] > 1 else '++' if row['diff'] > 0 else '--'
        print(f"  {row['models']:25s} {row['window']:>5d}  {row['accuracy_v1']:>5.1f}% {row['accuracy_v2']:>5.1f}% {diff_str:>7s} {color}")

    print(f"\n  Bottom 5 configs where V2 was worse:")
    for _, row in merged.tail(5).iterrows():
        diff_str = f"{row['diff']:+.1f}%"
        print(f"  {row['models']:25s} {row['window']:>5d}  {row['accuracy_v1']:>5.1f}% {row['accuracy_v2']:>5.1f}% {diff_str:>7s}")

    # ---- VERDICT ----
    print(f"\n{'='*60}")
    if avg_diff > 0.5 and v2_wins > v1_wins:
        print(f"  VERDICT: V2 WINS (+{avg_diff:.1f}% avg)")
        print(f"  Recommendation: Keep V2 features, update hourly system")
        # Save V2 best config
        best_v2 = results_v2.iloc[0]
        print(f"  Best V2 config: {best_v2['models']}, w={best_v2['window']}, acc={best_v2['accuracy']}%")
    elif avg_diff < -0.5 and v1_wins > v2_wins:
        print(f"  VERDICT: V1 WINS (V2 is {avg_diff:.1f}% worse)")
        print(f"  Recommendation: Delete V2, macro features not helping")
    else:
        print(f"  VERDICT: INCONCLUSIVE ({avg_diff:+.1f}% avg, {v2_wins} vs {v1_wins})")
        print(f"  Recommendation: Keep V2 code but use V1 for now. Retest with more data.")
    print(f"{'='*60}")

    # Save comparison results
    merged.to_csv('output/diagnostics/v1_vs_v2_comparison.csv', index=False)
    results_v1.to_csv('output/diagnostics/diagnostic_v1_results.csv', index=False)
    results_v2.to_csv('output/diagnostics/diagnostic_v2_results.csv', index=False)
    print(f"\n  Saved: v1_vs_v2_comparison.csv, diagnostic_v1_results.csv, diagnostic_v2_results.csv")


if __name__ == '__main__':
    main()
