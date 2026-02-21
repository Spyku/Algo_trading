"""
Feature Importance & Selection Tool
=====================================
Analyzes which features matter and tests reduced feature sets.
Run this alongside the diagnostic to decide what to keep/drop.

Methods:
  1. LGBM feature importance (gain-based)
  2. Permutation importance (accuracy drop when shuffled)
  3. Correlation-based redundancy detection
  4. Forward selection test (build up from best features)
  5. Ablation test (drop one feature at a time)

Usage:
  python feature_analysis.py              # Analyze all indices
  python feature_analysis.py --asset SMI  # Analyze one index
"""

import sys
import os

os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import sklearn.utils.parallel
    sklearn.utils.parallel.warnings.warn = lambda *a, **kw: None
except Exception:
    pass

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from hardware_config import MACHINE, LGBM_DEVICE

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# CONFIGURATION (same as hourly system)
# ============================================================
ASSETS = {
    'SMI':   {'file': 'smi_hourly_data.csv'},
    'DAX':   {'file': 'dax_hourly_data.csv'},
    'CAC40': {'file': 'cac40_hourly_data.csv'},
}

PREDICTION_HORIZON = 4
TEST_WINDOW = 300       # Training window for tests
TEST_STEP = 24          # Fast step (1 per trading day)


# ============================================================
# DATA LOADING & FEATURES (from hourly_trading_system)
# ============================================================
def load_data(asset_name):
    filepath = ASSETS[asset_name]['file']
    if not os.path.exists(filepath):
        print(f"  {filepath} not found!")
        return None
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def build_hourly_features(df_hourly):
    df = df_hourly.copy()

    for period in [1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 120, 240]:
        df[f'logret_{period}h'] = np.log(df['close'] / df['close'].shift(period))

    df['spread_24h_4h']   = df['logret_24h']  - df['logret_4h']
    df['spread_48h_4h']   = df['logret_48h']  - df['logret_4h']
    df['spread_120h_8h']  = df['logret_120h'] - df['logret_8h']
    df['spread_240h_24h'] = df['logret_240h'] - df['logret_24h']
    df['spread_48h_12h']  = df['logret_48h']  - df['logret_12h']
    df['spread_120h_12h'] = df['logret_120h'] - df['logret_12h']

    df['sma20h']  = df['close'].rolling(20).mean()
    df['sma50h']  = df['close'].rolling(50).mean()
    df['sma100h'] = df['close'].rolling(100).mean()
    df['sma200h'] = df['close'].rolling(200).mean()
    df['price_to_sma20h']  = df['close'] / df['sma20h'] - 1
    df['price_to_sma50h']  = df['close'] / df['sma50h'] - 1
    df['price_to_sma100h'] = df['close'] / df['sma100h'] - 1
    df['sma20_to_sma50h']  = df['sma20h'] / df['sma50h'] - 1

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14h'] = 100 - (100 / (1 + rs))

    low14  = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['stoch_k_14h'] = 100 * (df['close'] - low14) / (high14 - low14)

    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position_20h'] = (df['close'] - (bb_mid - 2 * bb_std)) / (4 * bb_std)

    roll_mean = df['close'].rolling(50).mean()
    roll_std  = df['close'].rolling(50).std()
    df['zscore_50h'] = (df['close'] - roll_mean) / roll_std

    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    df['atr_pct_14h'] = tr.rolling(14).mean() / df['close']

    df['intraday_range'] = (df['high'] - df['low']) / df['close']

    df['volatility_12h'] = df['logret_1h'].rolling(12).std()
    df['volatility_48h'] = df['logret_1h'].rolling(48).std()
    df['vol_ratio_12_48'] = df['volatility_12h'] / df['volatility_48h']

    if df['volume'].sum() == 0 or df['volume'].isna().all():
        df['volume_ratio_h'] = 1.0
    else:
        df['volume'] = df['volume'].replace(0, np.nan).ffill().bfill()
        vol_sma = df['volume'].rolling(20).mean()
        df['volume_ratio_h'] = df['volume'] / vol_sma
        df['volume_ratio_h'] = df['volume_ratio_h'].fillna(1.0)

    hour = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    dow = df['datetime'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    future_return = df['close'].shift(-PREDICTION_HORIZON) / df['close'] - 1
    rolling_median = future_return.rolling(200, min_periods=50).median().shift(PREDICTION_HORIZON)
    df['label'] = (future_return > rolling_median).astype(int)

    feature_cols = [
        'logret_1h', 'logret_2h', 'logret_3h', 'logret_4h', 'logret_6h',
        'logret_8h', 'logret_12h', 'logret_24h', 'logret_48h', 'logret_72h',
        'logret_120h', 'logret_240h',
        'spread_24h_4h', 'spread_48h_4h', 'spread_120h_8h',
        'spread_240h_24h', 'spread_48h_12h', 'spread_120h_12h',
        'price_to_sma20h', 'price_to_sma50h', 'price_to_sma100h', 'sma20_to_sma50h',
        'rsi_14h', 'stoch_k_14h', 'bb_position_20h', 'zscore_50h',
        'atr_pct_14h', 'intraday_range',
        'volatility_12h', 'volatility_48h', 'vol_ratio_12_48',
        'volume_ratio_h',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    ]

    keep_cols = ['datetime', 'close'] + feature_cols + ['label']
    df = df[keep_cols].copy()
    df = df.dropna().reset_index(drop=True)
    return df, feature_cols


# ============================================================
# QUICK WALK-FORWARD ACCURACY TEST
# ============================================================
def quick_accuracy(df_features, feature_cols, window=TEST_WINDOW, step=TEST_STEP):
    """Fast walk-forward test with LGBM only. Returns accuracy."""
    n = len(df_features)
    min_start = window + 50
    if n < min_start + 30:
        return 0, 0

    correct = 0
    total = 0

    for i in range(min_start, n, step):
        train = df_features.iloc[max(0, i - window):i]
        test_row = df_features.iloc[i:i+1]

        X_train = train[feature_cols]
        y_train = train['label'].values
        X_test  = test_row[feature_cols]
        y_true  = test_row['label'].values[0]

        if len(np.unique(y_train)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train),
                                 columns=feature_cols, index=X_train.index)
        X_test_s = pd.DataFrame(scaler.transform(X_test),
                                columns=feature_cols, index=X_test.index)

        model = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            class_weight='balanced', verbose=-1, random_state=42,
            device=LGBM_DEVICE
        )
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)[0]

        if pred == y_true:
            correct += 1
        total += 1

    return (correct / total * 100 if total > 0 else 0), total


# ============================================================
# TEST 1: LGBM FEATURE IMPORTANCE (GAIN)
# ============================================================
def test_lgbm_importance(df_features, feature_cols):
    """Train LGBM on full data and extract feature importance."""
    print("\n  [1] LGBM Feature Importance (gain-based)")
    print("  " + "-" * 50)

    # Use last 70% as train to get representative importance
    n = len(df_features)
    train = df_features.iloc[:int(n * 0.7)]
    X = train[feature_cols]
    y = train['label'].values

    scaler = StandardScaler()
    X_s = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    model = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        class_weight='balanced', verbose=-1, random_state=42,
        device=LGBM_DEVICE
    )
    model.fit(X_s, y)

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance['pct'] = (importance['importance'] / importance['importance'].sum() * 100)
    importance['cumulative_pct'] = importance['pct'].cumsum()

    for _, row in importance.iterrows():
        bar = '#' * int(row['pct'] * 2)
        marker = ' <<<' if row['pct'] < 1.0 else ''
        print(f"    {row['feature']:22s} {row['pct']:5.1f}% {bar}{marker}")

    # Identify low-value features
    low_value = importance[importance['pct'] < 1.0]['feature'].tolist()
    print(f"\n    Low-value features (<1%): {len(low_value)}")
    for f in low_value:
        print(f"      - {f}")

    return importance


# ============================================================
# TEST 2: PERMUTATION IMPORTANCE
# ============================================================
def test_permutation_importance(df_features, feature_cols):
    """Shuffle each feature and measure accuracy drop."""
    print("\n  [2] Permutation Importance (accuracy drop when shuffled)")
    print("  " + "-" * 50)

    # Get baseline accuracy
    baseline_acc, n_tests = quick_accuracy(df_features, feature_cols)
    print(f"    Baseline accuracy: {baseline_acc:.1f}% (n={n_tests})")

    results = []
    for feat in feature_cols:
        df_shuffled = df_features.copy()
        df_shuffled[feat] = np.random.permutation(df_shuffled[feat].values)

        shuffled_acc, _ = quick_accuracy(df_shuffled, feature_cols)
        drop = baseline_acc - shuffled_acc
        results.append({'feature': feat, 'acc_drop': drop, 'shuffled_acc': shuffled_acc})
        print(f"    Shuffle {feat:22s} -> {shuffled_acc:5.1f}% (drop: {drop:+5.1f}%)")

    df_results = pd.DataFrame(results).sort_values('acc_drop', ascending=False)

    # Features that don't hurt when shuffled = useless
    useless = df_results[df_results['acc_drop'] <= 0.0]['feature'].tolist()
    print(f"\n    Features with NO accuracy drop when shuffled: {len(useless)}")
    for f in useless:
        print(f"      - {f}")

    return df_results


# ============================================================
# TEST 3: CORRELATION REDUNDANCY
# ============================================================
def test_correlation(df_features, feature_cols, threshold=0.90):
    """Find highly correlated feature pairs."""
    print(f"\n  [3] Correlation Redundancy (threshold: {threshold})")
    print("  " + "-" * 50)

    corr_matrix = df_features[feature_cols].corr().abs()

    pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            corr = corr_matrix.iloc[i, j]
            if corr >= threshold:
                pairs.append({
                    'feature_1': feature_cols[i],
                    'feature_2': feature_cols[j],
                    'correlation': round(corr, 3)
                })

    if pairs:
        df_pairs = pd.DataFrame(pairs).sort_values('correlation', ascending=False)
        print(f"    Found {len(pairs)} highly correlated pairs (>={threshold}):")
        for _, row in df_pairs.iterrows():
            print(f"    {row['feature_1']:22s} <-> {row['feature_2']:22s} "
                  f"r={row['correlation']:.3f}")

        # Suggest which to drop (drop the one with lower LGBM importance)
        print(f"\n    Suggestion: from each pair, drop the less important feature")
    else:
        print(f"    No pairs found above {threshold}")

    return pairs


# ============================================================
# TEST 4: ABLATION TEST (drop one at a time)
# ============================================================
def test_ablation(df_features, feature_cols):
    """Drop each feature one at a time and measure accuracy."""
    print("\n  [4] Ablation Test (drop one feature at a time)")
    print("  " + "-" * 50)

    baseline_acc, n_tests = quick_accuracy(df_features, feature_cols)
    print(f"    Baseline ({len(feature_cols)} features): {baseline_acc:.1f}%")

    results = []
    for feat in feature_cols:
        reduced_cols = [f for f in feature_cols if f != feat]
        acc, _ = quick_accuracy(df_features, reduced_cols)
        change = acc - baseline_acc
        results.append({'dropped': feat, 'accuracy': acc, 'change': change})

        marker = ' ** IMPROVES' if change > 0.3 else ''
        print(f"    Drop {feat:22s} -> {acc:5.1f}% ({change:+5.1f}%){marker}")

    df_results = pd.DataFrame(results).sort_values('change', ascending=False)

    # Features that improve accuracy when dropped = harmful
    harmful = df_results[df_results['change'] > 0.3]['dropped'].tolist()
    if harmful:
        print(f"\n    Features that IMPROVE accuracy when dropped:")
        for f in harmful:
            row = df_results[df_results['dropped'] == f].iloc[0]
            print(f"      - {f} (accuracy +{row['change']:.1f}%)")

    return df_results


# ============================================================
# TEST 5: REDUCED FEATURE SET COMPARISON
# ============================================================
def test_reduced_sets(df_features, feature_cols, importance_df):
    """Test accuracy with top-N features."""
    print("\n  [5] Reduced Feature Sets (top-N by importance)")
    print("  " + "-" * 50)

    ranked_features = importance_df['feature'].tolist()

    results = []
    for n_features in [5, 8, 10, 12, 15, 18, 20, 25, 30, len(feature_cols)]:
        if n_features > len(feature_cols):
            continue
        top_n = ranked_features[:n_features]
        acc, n_tests = quick_accuracy(df_features, top_n)
        results.append({'n_features': n_features, 'accuracy': acc})
        bar = '#' * int(acc * 0.5)
        print(f"    Top {n_features:2d} features: {acc:5.1f}% {bar}")

    # Find optimal
    df_results = pd.DataFrame(results)
    best_row = df_results.loc[df_results['accuracy'].idxmax()]
    print(f"\n    OPTIMAL: Top {int(best_row['n_features'])} features "
          f"-> {best_row['accuracy']:.1f}%")

    return df_results


# ============================================================
# GENERATE RECOMMENDATION
# ============================================================
def generate_recommendation(feature_cols, importance_df, ablation_df,
                            permutation_df, correlation_pairs, reduced_df):
    """Synthesize all tests into a clear recommendation."""
    print("\n" + "=" * 60)
    print("  RECOMMENDATION")
    print("=" * 60)

    # Score each feature
    scores = {}
    for f in feature_cols:
        scores[f] = 0

    # LGBM importance > 2% = +2, > 5% = +3
    for _, row in importance_df.iterrows():
        f = row['feature']
        if row['pct'] >= 5:
            scores[f] += 3
        elif row['pct'] >= 2:
            scores[f] += 2
        elif row['pct'] >= 1:
            scores[f] += 1
        else:
            scores[f] -= 1

    # Permutation: drop > 0.5% = +2, drop < 0 = -1
    if permutation_df is not None:
        for _, row in permutation_df.iterrows():
            f = row['feature']
            if row['acc_drop'] > 0.5:
                scores[f] += 2
            elif row['acc_drop'] > 0:
                scores[f] += 1
            else:
                scores[f] -= 1

    # Ablation: improves when dropped = -3
    if ablation_df is not None:
        for _, row in ablation_df.iterrows():
            f = row['dropped']
            if row['change'] > 0.3:
                scores[f] -= 3
            elif row['change'] > 0:
                scores[f] -= 1

    # Sort by score
    score_df = pd.DataFrame([
        {'feature': f, 'score': s} for f, s in scores.items()
    ]).sort_values('score', ascending=False)

    # Categorize
    keep = score_df[score_df['score'] >= 1]['feature'].tolist()
    maybe = score_df[(score_df['score'] >= -1) & (score_df['score'] < 1)]['feature'].tolist()
    drop = score_df[score_df['score'] < -1]['feature'].tolist()

    print(f"\n  KEEP ({len(keep)} features) - clearly useful:")
    for f in keep:
        print(f"    + {f} (score: {scores[f]})")

    print(f"\n  MAYBE ({len(maybe)} features) - marginal, test both ways:")
    for f in maybe:
        print(f"    ~ {f} (score: {scores[f]})")

    print(f"\n  DROP ({len(drop)} features) - likely noise:")
    for f in drop:
        print(f"    - {f} (score: {scores[f]})")

    # Find best reduced set size
    if reduced_df is not None and len(reduced_df) > 0:
        best_n = int(reduced_df.loc[reduced_df['accuracy'].idxmax(), 'n_features'])
        best_acc = reduced_df['accuracy'].max()
        full_acc = reduced_df[reduced_df['n_features'] == reduced_df['n_features'].max()]['accuracy'].values[0]
        print(f"\n  Best reduced set: Top {best_n} features -> {best_acc:.1f}% "
              f"(vs {full_acc:.1f}% with all {len(feature_cols)})")

    # Output the recommended feature list
    recommended = keep + maybe
    print(f"\n  RECOMMENDED FEATURE LIST ({len(recommended)} features):")
    print(f"  {recommended}")

    return score_df, keep, maybe, drop


# ============================================================
# MAIN
# ============================================================
def analyze_asset(asset_name):
    """Run full feature analysis for one asset."""
    print(f"\n{'='*60}")
    print(f"  FEATURE ANALYSIS: {asset_name}")
    print(f"{'='*60}")

    df_raw = load_data(asset_name)
    if df_raw is None:
        return

    df_features, feature_cols = build_hourly_features(df_raw)
    print(f"  Data: {len(df_features):,} hourly rows, {len(feature_cols)} features")

    if len(df_features) < 500:
        print(f"  Not enough data. Skipping.")
        return

    # Run all tests
    start = datetime.now()

    importance_df = test_lgbm_importance(df_features, feature_cols)
    permutation_df = test_permutation_importance(df_features, feature_cols)
    correlation_pairs = test_correlation(df_features, feature_cols)
    ablation_df = test_ablation(df_features, feature_cols)
    reduced_df = test_reduced_sets(df_features, feature_cols, importance_df)

    # Generate recommendation
    score_df, keep, maybe, drop = generate_recommendation(
        feature_cols, importance_df, ablation_df,
        permutation_df, correlation_pairs, reduced_df
    )

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n  Analysis completed in {elapsed/60:.1f} minutes")

    # Save results
    output_file = f'feature_analysis_{asset_name.lower()}.csv'
    score_df.to_csv(output_file, index=False)
    print(f"  Results saved to {output_file}")

    return keep, maybe, drop


def main():
    parser = argparse.ArgumentParser(description='Feature Analysis Tool')
    parser.add_argument('--asset', type=str, default=None,
                        help='Analyze specific asset (SMI, DAX, CAC40)')
    args = parser.parse_args()

    print("=" * 60)
    print("  FEATURE IMPORTANCE & SELECTION ANALYSIS")
    print("  Hourly Index Trading System")
    print("=" * 60)

    if args.asset:
        assets = [args.asset.upper()]
    else:
        print("\nWhich index to analyze?")
        print("  1. All (SMI, DAX, CAC40)")
        print("  2. SMI only (fastest, ~15 min)")
        print("  3. DAX only")
        print("  4. CAC40 only")
        choice = input("Enter choice: ").strip()

        if choice == '2':
            assets = ['SMI']
        elif choice == '3':
            assets = ['DAX']
        elif choice == '4':
            assets = ['CAC40']
        else:
            assets = ['SMI', 'DAX', 'CAC40']

    all_results = {}
    for asset in assets:
        result = analyze_asset(asset)
        if result:
            all_results[asset] = result

    # Cross-asset summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  CROSS-ASSET SUMMARY")
        print(f"{'='*60}")

        # Find features that are DROP across all assets
        all_drops = set(all_results[list(all_results.keys())[0]][2])
        for asset, (keep, maybe, drop) in all_results.items():
            all_drops = all_drops.intersection(set(drop))

        if all_drops:
            print(f"\n  Features to DROP across ALL indices:")
            for f in all_drops:
                print(f"    - {f}")

        # Find features that are KEEP across all
        all_keeps = set(all_results[list(all_results.keys())[0]][0])
        for asset, (keep, maybe, drop) in all_results.items():
            all_keeps = all_keeps.intersection(set(keep))

        if all_keeps:
            print(f"\n  Features that are ESSENTIAL across ALL indices:")
            for f in sorted(all_keeps):
                print(f"    + {f}")

    print(f"\n{'='*60}")
    print("  ANALYSIS COMPLETE")
    print("  Next: update feature_cols in hourly_trading_system.py")
    print("  and hourly_model_diagnostic.py with the recommended set")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
