"""
Feature Analysis — Daily System (Crypto + Indices)
====================================================
Analyzes which features matter in the daily trading system.
Tests: LGBM importance, permutation, correlation, ablation, top-N.

Usage:
  python feature_analysis_daily.py                  # All assets
  python feature_analysis_daily.py --asset BTC      # One asset
  python feature_analysis_daily.py --asset SMI
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
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from hardware_config import MACHINE, LGBM_DEVICE

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# CONFIGURATION (mirrors crypto_trading_system.py)
# ============================================================
ASSETS = {
    'BTC':   {'source': 'binance', 'file': 'btc_hourly_data.csv'},
    'ETH':   {'source': 'binance', 'file': 'eth_hourly_data.csv'},
    'SOL':   {'source': 'binance', 'file': 'sol_hourly_data.csv'},
    'XRP':   {'source': 'binance', 'file': 'xrp_hourly_data.csv'},
    'DOGE':  {'source': 'binance', 'file': 'doge_hourly_data.csv'},
    'SMI':   {'source': 'yfinance', 'file': 'smi_hourly_data.csv'},
    'DAX':   {'source': 'yfinance', 'file': 'dax_hourly_data.csv'},
    'CAC40': {'source': 'yfinance', 'file': 'cac40_hourly_data.csv'},
}

PREDICTION_HORIZON = 3  # 3-day prediction (same as daily system)
TEST_WINDOW = 200
TEST_STEP = 3


# ============================================================
# DATA LOADING & DAILY AGGREGATION
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


def hourly_to_daily(df):
    df = df.copy()
    df['date'] = df['datetime'].dt.date
    daily = df.groupby('date').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    ).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').reset_index(drop=True)
    return daily


def build_features(df_daily):
    """Build daily features — identical to crypto_trading_system.py."""
    df = df_daily.copy()

    # Log Returns
    for period in [1, 2, 3, 5, 7, 10, 14, 20, 30]:
        df[f'logret_{period}d'] = np.log(df['close'] / df['close'].shift(period))
    for period in [50, 100, 250]:
        df[f'logret_{period}d'] = np.log(df['close'] / df['close'].shift(period))

    # Log Return Spreads
    df['spread_log10_log2']   = df['logret_10d']  - df['logret_2d']
    df['spread_log20_log2']   = df['logret_20d']  - df['logret_2d']
    df['spread_log30_log2']   = df['logret_30d']  - df['logret_2d']
    df['spread_log30_log10']  = df['logret_30d']  - df['logret_10d']
    df['spread_log7_log3']    = df['logret_7d']   - df['logret_3d']
    df['spread_log250_log10'] = df['logret_250d'] - df['logret_10d']

    # Price-to-SMA
    df['sma20']  = df['close'].rolling(20).mean()
    df['sma50']  = df['close'].rolling(50).mean()
    df['sma200'] = df['close'].rolling(200).mean()
    df['price_to_sma20']  = df['close'] / df['sma20'] - 1
    df['price_to_sma50']  = df['close'] / df['sma50'] - 1
    df['price_to_sma200'] = df['close'] / df['sma200'] - 1
    df['sma20_to_sma50']  = df['sma20'] / df['sma50'] - 1

    # RSI 14
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Stochastic %K
    low14  = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low14) / (high14 - low14)

    # Bollinger Band Position
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - (bb_mid - 2 * bb_std)) / (4 * bb_std)

    # Z-Score 30d
    roll_mean = df['close'].rolling(30).mean()
    roll_std  = df['close'].rolling(30).std()
    df['zscore_30d'] = (df['close'] - roll_mean) / roll_std

    # ATR %
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    df['atr_pct'] = tr.rolling(14).mean() / df['close']

    # Volatility
    df['volatility_10d'] = df['logret_1d'].rolling(10).std()
    df['volatility_30d'] = df['logret_1d'].rolling(30).std()
    df['vol_ratio_10_30'] = df['volatility_10d'] / df['volatility_30d']

    # Volume Features
    if df['volume'].sum() == 0 or df['volume'].isna().all():
        df['volume_ratio'] = 1.0
        df['volume_change'] = 0.0
    else:
        df['volume'] = df['volume'].replace(0, np.nan).ffill().bfill()
        vol_sma20 = df['volume'].rolling(20).mean()
        df['volume_ratio']  = df['volume'] / vol_sma20
        df['volume_change'] = df['volume'].pct_change(5)
        df['volume_ratio']  = df['volume_ratio'].fillna(1.0)
        df['volume_change'] = df['volume_change'].fillna(0.0)

    # Day of Week
    dow = df['date'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # Labels
    future_return = df['close'].shift(-PREDICTION_HORIZON) / df['close'] - 1
    rolling_median = future_return.rolling(90, min_periods=30).median().shift(PREDICTION_HORIZON)
    df['label'] = (future_return > rolling_median).astype(int)

    feature_cols = [
        'logret_1d', 'logret_2d', 'logret_3d', 'logret_5d', 'logret_7d',
        'logret_10d', 'logret_14d', 'logret_20d', 'logret_30d',
        'spread_log10_log2', 'spread_log20_log2', 'spread_log30_log2',
        'spread_log30_log10', 'spread_log7_log3', 'spread_log250_log10',
        'price_to_sma20', 'price_to_sma50', 'price_to_sma200', 'sma20_to_sma50',
        'rsi_14', 'stoch_k', 'bb_position', 'zscore_30d', 'atr_pct',
        'volatility_10d', 'volatility_30d', 'vol_ratio_10_30',
        'volume_ratio', 'volume_change',
        'dow_sin', 'dow_cos',
    ]

    keep_cols = ['date', 'close', 'high', 'low', 'volume'] + feature_cols + ['label']
    df = df[keep_cols].copy()
    df = df.dropna().reset_index(drop=True)
    return df, feature_cols


# ============================================================
# QUICK WALK-FORWARD TEST
# ============================================================
def quick_accuracy(df_features, feature_cols, window=TEST_WINDOW, step=TEST_STEP):
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
# TEST 1: LGBM FEATURE IMPORTANCE
# ============================================================
def test_lgbm_importance(df_features, feature_cols):
    print("\n  [1] LGBM Feature Importance (gain-based)")
    print("  " + "-" * 50)

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

    low_value = importance[importance['pct'] < 1.0]['feature'].tolist()
    if low_value:
        print(f"\n    Low-value features (<1%): {len(low_value)}")
        for f in low_value:
            print(f"      - {f}")

    return importance


# ============================================================
# TEST 2: PERMUTATION IMPORTANCE
# ============================================================
def test_permutation_importance(df_features, feature_cols):
    print("\n  [2] Permutation Importance (accuracy drop when shuffled)")
    print("  " + "-" * 50)

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

    useless = df_results[df_results['acc_drop'] <= 0.0]['feature'].tolist()
    if useless:
        print(f"\n    Features with NO accuracy drop when shuffled: {len(useless)}")
        for f in useless:
            print(f"      - {f}")

    return df_results


# ============================================================
# TEST 3: CORRELATION REDUNDANCY
# ============================================================
def test_correlation(df_features, feature_cols, threshold=0.90):
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
    else:
        print(f"    No pairs found above {threshold}")

    return pairs


# ============================================================
# TEST 4: ABLATION TEST
# ============================================================
def test_ablation(df_features, feature_cols):
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

    harmful = df_results[df_results['change'] > 0.3]['dropped'].tolist()
    if harmful:
        print(f"\n    Features that IMPROVE accuracy when dropped:")
        for f in harmful:
            row = df_results[df_results['dropped'] == f].iloc[0]
            print(f"      - {f} (accuracy +{row['change']:.1f}%)")

    return df_results


# ============================================================
# TEST 5: REDUCED FEATURE SETS
# ============================================================
def test_reduced_sets(df_features, feature_cols, importance_df):
    print("\n  [5] Reduced Feature Sets (top-N by importance)")
    print("  " + "-" * 50)

    ranked_features = importance_df['feature'].tolist()

    results = []
    for n_features in [5, 8, 10, 12, 15, 18, 20, 25, len(feature_cols)]:
        if n_features > len(feature_cols):
            continue
        top_n = ranked_features[:n_features]
        acc, n_tests = quick_accuracy(df_features, top_n)
        results.append({'n_features': n_features, 'accuracy': acc})
        bar = '#' * int(acc * 0.5)
        print(f"    Top {n_features:2d} features: {acc:5.1f}% {bar}")

    df_results = pd.DataFrame(results)
    best_row = df_results.loc[df_results['accuracy'].idxmax()]
    print(f"\n    OPTIMAL: Top {int(best_row['n_features'])} features "
          f"-> {best_row['accuracy']:.1f}%")

    return df_results


# ============================================================
# RECOMMENDATION
# ============================================================
def generate_recommendation(feature_cols, importance_df, ablation_df,
                            permutation_df, correlation_pairs, reduced_df):
    print("\n" + "=" * 60)
    print("  RECOMMENDATION")
    print("=" * 60)

    scores = {f: 0 for f in feature_cols}

    # LGBM importance
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

    # Permutation
    if permutation_df is not None:
        for _, row in permutation_df.iterrows():
            f = row['feature']
            if row['acc_drop'] > 0.5:
                scores[f] += 2
            elif row['acc_drop'] > 0:
                scores[f] += 1
            else:
                scores[f] -= 1

    # Ablation
    if ablation_df is not None:
        for _, row in ablation_df.iterrows():
            f = row['dropped']
            if row['change'] > 0.3:
                scores[f] -= 3
            elif row['change'] > 0:
                scores[f] -= 1

    score_df = pd.DataFrame([
        {'feature': f, 'score': s} for f, s in scores.items()
    ]).sort_values('score', ascending=False)

    keep = score_df[score_df['score'] >= 1]['feature'].tolist()
    maybe = score_df[(score_df['score'] >= -1) & (score_df['score'] < 1)]['feature'].tolist()
    drop = score_df[score_df['score'] < -1]['feature'].tolist()

    print(f"\n  KEEP ({len(keep)} features):")
    for f in keep:
        print(f"    + {f} (score: {scores[f]})")

    print(f"\n  MAYBE ({len(maybe)} features):")
    for f in maybe:
        print(f"    ~ {f} (score: {scores[f]})")

    print(f"\n  DROP ({len(drop)} features):")
    for f in drop:
        print(f"    - {f} (score: {scores[f]})")

    if reduced_df is not None and len(reduced_df) > 0:
        best_n = int(reduced_df.loc[reduced_df['accuracy'].idxmax(), 'n_features'])
        best_acc = reduced_df['accuracy'].max()
        full_acc = reduced_df[reduced_df['n_features'] == reduced_df['n_features'].max()]['accuracy'].values[0]
        print(f"\n  Best reduced set: Top {best_n} features -> {best_acc:.1f}% "
              f"(vs {full_acc:.1f}% with all {len(feature_cols)})")

    recommended = keep + maybe
    print(f"\n  RECOMMENDED LIST ({len(recommended)} features):")
    print(f"  {recommended}")

    return score_df, keep, maybe, drop


# ============================================================
# MAIN
# ============================================================
def analyze_asset(asset_name):
    print(f"\n{'='*60}")
    print(f"  FEATURE ANALYSIS (DAILY): {asset_name}")
    print(f"{'='*60}")

    df_raw = load_data(asset_name)
    if df_raw is None:
        return None

    df_daily = hourly_to_daily(df_raw)
    print(f"  Raw: {len(df_raw):,} hourly -> {len(df_daily):,} daily candles")

    df_features, feature_cols = build_features(df_daily)
    print(f"  Usable: {len(df_features):,} rows, {len(feature_cols)} features")

    if len(df_features) < 300:
        print(f"  Not enough data. Skipping.")
        return None

    start = datetime.now()

    importance_df = test_lgbm_importance(df_features, feature_cols)
    permutation_df = test_permutation_importance(df_features, feature_cols)
    correlation_pairs = test_correlation(df_features, feature_cols)
    ablation_df = test_ablation(df_features, feature_cols)
    reduced_df = test_reduced_sets(df_features, feature_cols, importance_df)

    score_df, keep, maybe, drop = generate_recommendation(
        feature_cols, importance_df, ablation_df,
        permutation_df, correlation_pairs, reduced_df
    )

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n  Completed in {elapsed/60:.1f} minutes")

    output_file = f'feature_analysis_daily_{asset_name.lower()}.csv'
    score_df.to_csv(output_file, index=False)
    print(f"  Saved to {output_file}")

    return keep, maybe, drop


def main():
    parser = argparse.ArgumentParser(description='Daily Feature Analysis')
    parser.add_argument('--asset', type=str, default=None,
                        help='Analyze specific asset (BTC, ETH, SOL, XRP, DOGE, SMI, DAX, CAC40)')
    args = parser.parse_args()

    print("=" * 60)
    print("  FEATURE ANALYSIS — DAILY SYSTEM")
    print("  Crypto: BTC, ETH, SOL, XRP, DOGE")
    print("  Indices: SMI, DAX, CAC40")
    print("=" * 60)

    if args.asset:
        assets = [args.asset.upper()]
    else:
        print("\nWhich assets?")
        print("  1. All 8 assets")
        print("  2. Crypto only (BTC, ETH, SOL, XRP, DOGE)")
        print("  3. Indices only (SMI, DAX, CAC40)")
        print("  4. Just BTC (fastest test, ~10 min)")
        choice = input("Enter choice: ").strip()

        if choice == '2':
            assets = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']
        elif choice == '3':
            assets = ['SMI', 'DAX', 'CAC40']
        elif choice == '4':
            assets = ['BTC']
        else:
            assets = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'SMI', 'DAX', 'CAC40']

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

        # Consensus drops
        all_drops = None
        for asset, (keep, maybe, drop) in all_results.items():
            if all_drops is None:
                all_drops = set(drop)
            else:
                all_drops = all_drops.intersection(set(drop))

        if all_drops:
            print(f"\n  UNIVERSAL DROPS (noise across ALL assets):")
            for f in sorted(all_drops):
                print(f"    - {f}")

        # Consensus keeps
        all_keeps = None
        for asset, (keep, maybe, drop) in all_results.items():
            if all_keeps is None:
                all_keeps = set(keep)
            else:
                all_keeps = all_keeps.intersection(set(keep))

        if all_keeps:
            print(f"\n  ESSENTIAL (strong across ALL assets):")
            for f in sorted(all_keeps):
                print(f"    + {f}")

        # Per-asset summary table
        print(f"\n  Per-Asset Feature Counts:")
        print(f"  {'Asset':8s} {'Keep':6s} {'Maybe':6s} {'Drop':6s}")
        print(f"  {'-'*30}")
        for asset, (keep, maybe, drop) in all_results.items():
            print(f"  {asset:8s} {len(keep):6d} {len(maybe):6d} {len(drop):6d}")

    print(f"\n{'='*60}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
