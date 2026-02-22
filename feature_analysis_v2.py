"""
Feature Analysis V2 — DAX Hourly
==================================
Analyzes all 101 V2 features (20 base + 81 macro/sentiment/cross-asset)
to find the optimal subset.

Methods:
  1. Random Forest importance (Gini)
  2. Mutual Information (non-linear dependency)
  3. Permutation Importance (model-agnostic, walk-forward)
  4. Correlation-based redundancy removal
  5. Subset testing: walk-forward accuracy for top-N subsets

Outputs:
  - feature_analysis_v2_results.csv  (full ranking per method)
  - feature_analysis_v2_optimal.csv  (recommended feature list)
  - Console: recommended features + accuracy comparison

Usage:
  python feature_analysis_v2.py
"""

import os, sys, time, warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

# Hardware config
try:
    from hardware_config import get_config
    hw = get_config()
    N_JOBS = hw['n_workers']
    print(f"  [{hw['machine']}] {hw['cpu_cores']} cores | {hw.get('gpu_name','no GPU')} | Workers: {N_JOBS}")
except:
    N_JOBS = max(1, os.cpu_count() - 2)

try:
    import lightgbm as lgb
    HAS_LGBM = True
except:
    HAS_LGBM = False


# ============================================================
# 1. LOAD DATA & BUILD V2 FEATURES
# ============================================================
def load_v2_data():
    """Load DAX data and build V2 features."""
    from hourly_trading_system import load_data, build_hourly_features, update_all_data
    from features_v2 import build_features_v2_hourly

    print("\n  Loading DAX data...")
    try:
        update_all_data(['DAX'])
    except:
        pass

    df_raw = load_data('DAX')
    if df_raw is None:
        print("  ERROR: Cannot load DAX data")
        sys.exit(1)

    df, feature_cols = build_features_v2_hourly(df_raw, original_builder=build_hourly_features)

    # Clean NaN
    valid_cols = feature_cols + ['label']
    df_clean = df.dropna(subset=valid_cols).reset_index(drop=True)

    print(f"  Data: {len(df_clean)} rows, {len(feature_cols)} features")

    X = df_clean[feature_cols].values
    y = df_clean['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df_clean, feature_cols, X_scaled, y


# ============================================================
# 2. RANDOM FOREST IMPORTANCE (GINI)
# ============================================================
def rf_importance(X, y, feature_cols):
    """Feature importance from Random Forest (Gini impurity)."""
    print("\n  [1/5] Random Forest Importance (Gini)...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=N_JOBS)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"        Done in {time.time()-t0:.1f}s")
    return imp


# ============================================================
# 3. MUTUAL INFORMATION
# ============================================================
def mi_importance(X, y, feature_cols):
    """Non-linear dependency between each feature and target."""
    print("\n  [2/5] Mutual Information...")
    t0 = time.time()
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    imp = pd.Series(mi, index=feature_cols).sort_values(ascending=False)
    print(f"        Done in {time.time()-t0:.1f}s")
    return imp


# ============================================================
# 4. GRADIENT BOOSTING IMPORTANCE
# ============================================================
def gb_importance(X, y, feature_cols):
    """Feature importance from Gradient Boosting."""
    print("\n  [3/5] Gradient Boosting Importance...")
    t0 = time.time()
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
    gb.fit(X, y)
    imp = pd.Series(gb.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"        Done in {time.time()-t0:.1f}s")
    return imp


# ============================================================
# 5. LGBM IMPORTANCE (if available)
# ============================================================
def lgbm_importance(X, y, feature_cols):
    """Feature importance from LightGBM."""
    if not HAS_LGBM:
        return None
    print("\n  [4/5] LightGBM Importance...")
    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, verbose=-1, n_jobs=N_JOBS, device='cpu'
    )
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    # Normalize to 0-1 scale like the others
    imp = imp / imp.sum()
    print(f"        Done in {time.time()-t0:.1f}s")
    return imp


# ============================================================
# 6. CORRELATION ANALYSIS
# ============================================================
def correlation_analysis(X, feature_cols, threshold=0.85):
    """Find highly correlated feature pairs (redundancy)."""
    print(f"\n  [5/5] Correlation Analysis (threshold={threshold})...")
    t0 = time.time()
    corr_matrix = np.corrcoef(X.T)

    # Find pairs above threshold
    redundant_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            if abs(corr_matrix[i, j]) > threshold:
                redundant_pairs.append({
                    'feature_1': feature_cols[i],
                    'feature_2': feature_cols[j],
                    'correlation': round(corr_matrix[i, j], 3)
                })

    print(f"        Found {len(redundant_pairs)} highly correlated pairs")
    print(f"        Done in {time.time()-t0:.1f}s")

    return redundant_pairs, corr_matrix


# ============================================================
# 7. WALK-FORWARD SUBSET TESTING
# ============================================================
def walk_forward_accuracy(df, feature_subset, target_col='label', window=100, test_fraction=0.3):
    """
    Walk-forward accuracy for a given feature subset.
    Uses RF+GB ensemble (the best config from diagnostic).
    """
    n = len(df)
    test_start = int(n * (1 - test_fraction))
    correct = 0
    total = 0

    for i in range(test_start, n):
        train_start = max(0, i - window)
        train = df.iloc[train_start:i]
        test_row = df.iloc[i:i+1]

        X_train = train[feature_subset].values
        y_train = train[target_col].values

        if len(np.unique(y_train)) < 2:
            continue

        X_test = test_row[feature_subset].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        votes = []
        try:
            rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)
            rf.fit(X_train_s, y_train)
            votes.append(int(rf.predict(X_test_s)[0]))
        except:
            pass

        try:
            gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
            gb.fit(X_train_s, y_train)
            votes.append(int(gb.predict(X_test_s)[0]))
        except:
            pass

        if HAS_LGBM:
            try:
                model = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    random_state=42, verbose=-1, n_jobs=1, device='cpu'
                )
                model.fit(X_train_s, y_train)
                votes.append(int(model.predict(X_test_s)[0]))
            except:
                pass

        if not votes:
            continue

        ensemble_pred = 1 if sum(votes) > len(votes) / 2 else 0
        if ensemble_pred == int(test_row[target_col].values[0]):
            correct += 1
        total += 1

    return round(correct / total * 100, 2) if total > 0 else 0


def test_subsets(df, ranked_features, windows=[30, 100]):
    """
    Test top-N feature subsets to find optimal count.
    Tests: 10, 15, 20, 25, 30, 40, 50, 60, 75, all
    """
    subset_sizes = [10, 15, 20, 25, 30, 40, 50, 60, 75, len(ranked_features)]
    subset_sizes = sorted(set(s for s in subset_sizes if s <= len(ranked_features)))

    results = []
    total_tests = len(subset_sizes) * len(windows)
    test_num = 0

    print(f"\n  Testing {len(subset_sizes)} subset sizes × {len(windows)} windows = {total_tests} tests...")

    for n in subset_sizes:
        subset = ranked_features[:n]
        for window in windows:
            test_num += 1
            print(f"    [{test_num}/{total_tests}] top-{n} features, w={window}...", end=' ', flush=True)
            acc = walk_forward_accuracy(df, subset, window=window)
            print(f"{acc:.1f}%")
            results.append({
                'n_features': n,
                'window': window,
                'accuracy': acc,
                'features': ','.join(subset[:5]) + '...' if n > 5 else ','.join(subset),
            })

    return pd.DataFrame(results)


# ============================================================
# COMPOSITE RANKING
# ============================================================
def composite_ranking(importances_dict, feature_cols):
    """
    Create composite ranking by averaging normalized ranks across methods.
    """
    n = len(feature_cols)
    ranks = pd.DataFrame(index=feature_cols)

    for method, imp in importances_dict.items():
        if imp is None:
            continue
        # Rank: 1 = most important
        rank = imp.rank(ascending=False)
        # Normalize to 0-1 (1 = best)
        ranks[f'{method}_rank'] = 1 - (rank - 1) / (n - 1)
        ranks[f'{method}_imp'] = imp

    # Average normalized rank
    rank_cols = [c for c in ranks.columns if c.endswith('_rank')]
    ranks['avg_rank'] = ranks[rank_cols].mean(axis=1)
    ranks = ranks.sort_values('avg_rank', ascending=False)

    # Classify type
    def classify(feat):
        if feat.startswith(('m_', 'vix_')): return 'MACRO'
        if feat.startswith('fg_'): return 'SENTIMENT'
        if feat.startswith('xa_'): return 'CROSS-ASSET'
        return 'BASE'

    ranks['type'] = ranks.index.map(classify)

    return ranks


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 65)
    print("  FEATURE ANALYSIS V2 — DAX HOURLY")
    print("  101 features → find optimal subset")
    print("=" * 65)

    # Load data
    df, feature_cols, X, y = load_v2_data()

    # Run all importance methods
    imp_rf = rf_importance(X, y, feature_cols)
    imp_mi = mi_importance(X, y, feature_cols)
    imp_gb = gb_importance(X, y, feature_cols)
    imp_lgbm = lgbm_importance(X, y, feature_cols)

    # Correlation analysis
    redundant_pairs, corr_matrix = correlation_analysis(X, feature_cols)

    # Composite ranking
    print("\n" + "=" * 65)
    print("  COMPOSITE RANKING")
    print("=" * 65)

    importances = {
        'RF': imp_rf,
        'MI': imp_mi,
        'GB': imp_gb,
        'LGBM': imp_lgbm,
    }

    ranking = composite_ranking(importances, feature_cols)

    # Show top 40
    print(f"\n  {'Rank':>4s}  {'Feature':45s}  {'Type':>12s}  {'Composite':>9s}  {'RF':>6s}  {'MI':>6s}  {'GB':>6s}  {'LGBM':>6s}")
    print(f"  {'-'*100}")

    for i, (feat, row) in enumerate(ranking.head(40).iterrows()):
        rf_val = f"{row.get('RF_imp', 0):.4f}" if 'RF_imp' in row.index else '-'
        mi_val = f"{row.get('MI_imp', 0):.4f}" if 'MI_imp' in row.index else '-'
        gb_val = f"{row.get('GB_imp', 0):.4f}" if 'GB_imp' in row.index else '-'
        lgbm_val = f"{row.get('LGBM_imp', 0):.4f}" if 'LGBM_imp' in row.index else '-'
        bar = '█' * int(row['avg_rank'] * 30)
        print(f"  {i+1:>4d}  {feat:45s}  {row['type']:>12s}  {row['avg_rank']:>9.4f}  {rf_val}  {mi_val}  {gb_val}  {lgbm_val}  {bar}")

    # Category breakdown
    print(f"\n  Category breakdown (top 30):")
    top30_types = ranking.head(30)['type'].value_counts()
    for cat, count in top30_types.items():
        print(f"    {cat:15s}: {count:>2d} features ({count/30*100:.0f}%)")

    # Show redundant pairs
    if redundant_pairs:
        print(f"\n  Highly correlated pairs (|r| > 0.85) — candidates for removal:")
        for pair in sorted(redundant_pairs, key=lambda p: -abs(p['correlation']))[:20]:
            print(f"    {pair['feature_1']:40s} ↔ {pair['feature_2']:40s}  r={pair['correlation']:+.3f}")

    # ============================================================
    # SUBSET TESTING (walk-forward)
    # ============================================================
    print(f"\n{'='*65}")
    print(f"  SUBSET TESTING (walk-forward accuracy)")
    print(f"{'='*65}")

    ranked_features = ranking.index.tolist()

    # Remove one from each highly-correlated pair (keep the one ranked higher)
    features_to_drop = set()
    for pair in redundant_pairs:
        f1, f2 = pair['feature_1'], pair['feature_2']
        if f1 in ranking.index and f2 in ranking.index:
            if ranking.loc[f1, 'avg_rank'] >= ranking.loc[f2, 'avg_rank']:
                features_to_drop.add(f2)
            else:
                features_to_drop.add(f1)

    deduplicated = [f for f in ranked_features if f not in features_to_drop]
    print(f"\n  After removing {len(features_to_drop)} redundant features: {len(deduplicated)} remaining")

    subset_results = test_subsets(df, deduplicated, windows=[30, 100])

    # Show results
    print(f"\n  Subset Test Results:")
    print(f"  {'N features':>12s}  {'w=30':>8s}  {'w=100':>8s}  {'avg':>8s}")
    print(f"  {'-'*42}")

    for n in sorted(subset_results['n_features'].unique()):
        rows = subset_results[subset_results['n_features'] == n]
        w30 = rows[rows['window'] == 30]['accuracy'].values[0] if len(rows[rows['window'] == 30]) > 0 else '-'
        w100 = rows[rows['window'] == 100]['accuracy'].values[0] if len(rows[rows['window'] == 100]) > 0 else '-'
        avg = rows['accuracy'].mean()
        marker = ' ◀ BEST' if avg == subset_results.groupby('n_features')['accuracy'].mean().max() else ''
        w30_str = f"{w30:.1f}%" if isinstance(w30, float) else w30
        w100_str = f"{w100:.1f}%" if isinstance(w100, float) else w100
        print(f"  {n:>12d}  {w30_str:>8s}  {w100_str:>8s}  {avg:>7.1f}%{marker}")

    # Find optimal
    best_n = subset_results.groupby('n_features')['accuracy'].mean().idxmax()
    best_acc = subset_results.groupby('n_features')['accuracy'].mean().max()
    optimal_features = deduplicated[:best_n]

    # ============================================================
    # OUTPUT
    # ============================================================
    print(f"\n{'='*65}")
    print(f"  RECOMMENDATION")
    print(f"{'='*65}")
    print(f"\n  Optimal feature count: {best_n}")
    print(f"  Expected accuracy: {best_acc:.1f}%")
    print(f"\n  Recommended features ({best_n}):")

    for i, feat in enumerate(optimal_features):
        ftype = ranking.loc[feat, 'type'] if feat in ranking.index else '?'
        print(f"    {i+1:>3d}. {feat:45s}  [{ftype}]")

    # Count by type
    type_counts = pd.Series([ranking.loc[f, 'type'] for f in optimal_features]).value_counts()
    print(f"\n  Breakdown:")
    for cat, count in type_counts.items():
        print(f"    {cat:15s}: {count:>2d} features")

    # Save results
    ranking.to_csv('feature_analysis_v2_results.csv')
    print(f"\n  Saved: feature_analysis_v2_results.csv (full ranking)")

    optimal_df = pd.DataFrame({
        'rank': range(1, len(optimal_features) + 1),
        'feature': optimal_features,
        'type': [ranking.loc[f, 'type'] for f in optimal_features],
        'composite_score': [ranking.loc[f, 'avg_rank'] for f in optimal_features],
    })
    optimal_df.to_csv('feature_analysis_v2_optimal.csv', index=False)
    print(f"  Saved: feature_analysis_v2_optimal.csv (recommended {best_n} features)")

    subset_results.to_csv('feature_analysis_v2_subsets.csv', index=False)
    print(f"  Saved: feature_analysis_v2_subsets.csv (subset test results)")

    # Print the Python list for easy copy-paste into code
    print(f"\n  Copy-paste for hourly_trading_system.py:")
    print(f"  feature_cols_v2 = [")
    for feat in optimal_features:
        print(f"      '{feat}',")
    print(f"  ]")

    print(f"\n{'='*65}")
    print(f"  DONE — optimal: {best_n} features at {best_acc:.1f}% accuracy")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
