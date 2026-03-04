"""
Crypto Feature Importance & Selection Tool (V2 — with Macro Features)
======================================================================
Analyzes which hourly features matter for crypto and tests reduced sets.
NOW INCLUDES V2 macro features (VIX, DXY, yields, F&G, cross-asset).

Tests ALL untrimmed technical features (36) + ALL macro features (~80)
to determine the optimal subset for the crypto trading system.

Methods:
  1. LGBM feature importance (gain-based)
  2. Permutation importance (accuracy drop when shuffled)
  3. Correlation-based redundancy detection
  4. Ablation test (drop one feature at a time)
  5. Reduced feature set comparison (top-N by importance)

Usage:
  python crypto_feature_analysis.py              # Interactive menu
  python crypto_feature_analysis.py --asset BTC  # Analyze one coin
  python crypto_feature_analysis.py --v1         # V1 only (no macro)
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
# CONFIGURATION
# ============================================================
ASSETS = {
    'BTC':   {'source': 'binance', 'file': 'btc_hourly_data.csv'},
    'ETH':   {'source': 'binance', 'file': 'eth_hourly_data.csv'},
    'SOL':   {'source': 'binance', 'file': 'sol_hourly_data.csv'},
    'XRP':   {'source': 'binance', 'file': 'xrp_hourly_data.csv'},
    'DOGE':  {'source': 'binance', 'file': 'doge_hourly_data.csv'},
}

PREDICTION_HORIZON = 4
TEST_WINDOW = 400       # Training window for tests (hours)
TEST_STEP = 72          # Fast step (every 3 days — matches diagnostic speed)


# ============================================================
# DATA LOADING
# ============================================================
def load_data(asset_name):
    filepath = ASSETS[asset_name]['file']
    if not os.path.exists(filepath):
        print(f"  {filepath} not found! Run crypto_trading_system.py first.")
        return None
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close']).reset_index(drop=True)
    return df


# ============================================================
# FEATURE ENGINEERING (ALL features — untrimmed)
# Builds the full 36-feature set so we can test what to keep/drop
# ============================================================
def build_all_features(df_hourly):
    """Build ALL possible hourly features (untrimmed) for analysis."""
    df = df_hourly.copy()

    # --- Log Returns (12 features) ---
    for period in [1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 120, 240]:
        df[f'logret_{period}h'] = np.log(df['close'] / df['close'].shift(period))

    # --- Log Return Spreads (6 features) ---
    df['spread_24h_4h']   = df['logret_24h']  - df['logret_4h']
    df['spread_48h_4h']   = df['logret_48h']  - df['logret_4h']
    df['spread_120h_8h']  = df['logret_120h'] - df['logret_8h']
    df['spread_240h_24h'] = df['logret_240h'] - df['logret_24h']
    df['spread_48h_12h']  = df['logret_48h']  - df['logret_12h']
    df['spread_120h_12h'] = df['logret_120h'] - df['logret_12h']

    # --- Price-to-SMA Ratios (4 features) ---
    df['sma20h']  = df['close'].rolling(20).mean()
    df['sma50h']  = df['close'].rolling(50).mean()
    df['sma100h'] = df['close'].rolling(100).mean()
    df['sma200h'] = df['close'].rolling(200).mean()
    df['price_to_sma20h']  = df['close'] / df['sma20h'] - 1
    df['price_to_sma50h']  = df['close'] / df['sma50h'] - 1
    df['price_to_sma100h'] = df['close'] / df['sma100h'] - 1
    df['sma20_to_sma50h']  = df['sma20h'] / df['sma50h'] - 1

    # --- RSI 14h ---
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14h'] = 100 - (100 / (1 + rs))

    # --- Stochastic %K (14h) ---
    low14  = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['stoch_k_14h'] = 100 * (df['close'] - low14) / (high14 - low14)

    # --- Bollinger Band Position (20h) ---
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position_20h'] = (df['close'] - (bb_mid - 2 * bb_std)) / (4 * bb_std)

    # --- Z-Score (50h) ---
    roll_mean = df['close'].rolling(50).mean()
    roll_std  = df['close'].rolling(50).std()
    df['zscore_50h'] = (df['close'] - roll_mean) / roll_std

    # --- ATR % (14h) ---
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    df['atr_pct_14h'] = tr.rolling(14).mean() / df['close']

    # --- Intraday Range ---
    df['intraday_range'] = (df['high'] - df['low']) / df['close']

    # --- Volatility (3 features) ---
    df['volatility_12h'] = df['logret_1h'].rolling(12).std()
    df['volatility_48h'] = df['logret_1h'].rolling(48).std()
    df['vol_ratio_12_48'] = df['volatility_12h'] / df['volatility_48h']

    # --- Volume Features (crypto has real volume) ---
    if df['volume'].sum() == 0 or df['volume'].isna().all():
        df['volume_ratio_h'] = 1.0
    else:
        df['volume'] = df['volume'].replace(0, np.nan).ffill().bfill()
        vol_sma = df['volume'].rolling(20).mean()
        df['volume_ratio_h'] = df['volume'] / vol_sma
        df['volume_ratio_h'] = df['volume_ratio_h'].fillna(1.0)

    # --- Time Encoding (4 features) ---
    hour = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    dow = df['datetime'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # --- Labels ---
    future_return = df['close'].shift(-PREDICTION_HORIZON) / df['close'] - 1
    rolling_median = future_return.rolling(200, min_periods=50).median().shift(PREDICTION_HORIZON)
    df['label'] = (future_return > rolling_median).astype(int)

    # ALL features (36 total — untrimmed for analysis)
    feature_cols = [
        # Log returns (12)
        'logret_1h', 'logret_2h', 'logret_3h', 'logret_4h', 'logret_6h',
        'logret_8h', 'logret_12h', 'logret_24h', 'logret_48h', 'logret_72h',
        'logret_120h', 'logret_240h',
        # Spreads (6)
        'spread_24h_4h', 'spread_48h_4h', 'spread_120h_8h',
        'spread_240h_24h', 'spread_48h_12h', 'spread_120h_12h',
        # SMA ratios (4)
        'price_to_sma20h', 'price_to_sma50h', 'price_to_sma100h', 'sma20_to_sma50h',
        # Oscillators (4)
        'rsi_14h', 'stoch_k_14h', 'bb_position_20h', 'zscore_50h',
        # ATR + range (2)
        'atr_pct_14h', 'intraday_range',
        # Volatility (3)
        'volatility_12h', 'volatility_48h', 'vol_ratio_12_48',
        # Volume (1)
        'volume_ratio_h',
        # Time (4)
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    ]

    keep_cols = ['datetime', 'close'] + feature_cols + ['label']
    df = df[keep_cols].copy()
    df = df.dropna().reset_index(drop=True)
    return df, feature_cols


# ============================================================
# V2: MACRO FEATURE BUILDER FOR ANALYSIS
# ============================================================
# Builds ALL 36 untrimmed technical + ALL macro/sentiment/cross-asset
# features so we can test everything and determine what to keep/drop.

MACRO_DIR = 'macro_data'
_macro_cache = {}

def _load_macro_csv(filename):
    """Load a CSV from macro_data/ with caching."""
    if filename in _macro_cache:
        return _macro_cache[filename]
    path = os.path.join(MACRO_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        _macro_cache[filename] = df
        return df
    except Exception as e:
        print(f"  Warning: Could not load {path}: {e}")
        return None


def _compute_macro_features(macro_df, prefix='m_'):
    """Derive features from raw macro data."""
    features = pd.DataFrame(index=macro_df.index)
    for col in macro_df.columns:
        if col in ('fear_greed_label',):
            continue
        s = macro_df[col].astype(float)
        tag = f"{prefix}{col.lower()}"
        roll_mean = s.rolling(50, min_periods=10).mean()
        roll_std = s.rolling(50, min_periods=10).std()
        features[f'{tag}_zscore'] = (s - roll_mean) / (roll_std + 1e-10)
        for window in [1, 5, 10]:
            features[f'{tag}_chg{window}d'] = s.pct_change(window) * 100
        for window in [5, 20]:
            features[f'{tag}_vol{window}d'] = s.pct_change().rolling(window, min_periods=3).std() * 100
    return features


def _compute_vix_regime(macro_df):
    """VIX-based regime features."""
    features = pd.DataFrame(index=macro_df.index)
    if 'VIX' not in macro_df.columns:
        return features
    vix = macro_df['VIX'].astype(float)
    features['vix_low'] = (vix < 15).astype(float)
    features['vix_normal'] = ((vix >= 15) & (vix < 25)).astype(float)
    features['vix_high'] = ((vix >= 25) & (vix < 35)).astype(float)
    features['vix_extreme'] = (vix >= 35).astype(float)
    features['vix_rising_5d'] = (vix.diff(5) > 0).astype(float)
    features['vix_spike'] = (vix.pct_change() > 0.15).astype(float)
    return features


def _compute_fear_greed_features(fg_df, prefix='fg_'):
    """Derive features from Fear & Greed Index."""
    features = pd.DataFrame(index=fg_df.index)
    fg = fg_df['fear_greed'].astype(float)
    features[f'{prefix}value'] = fg
    features[f'{prefix}zscore'] = (fg - fg.rolling(50, min_periods=10).mean()) / (fg.rolling(50, min_periods=10).std() + 1e-10)
    for w in [1, 5, 10]:
        features[f'{prefix}chg{w}d'] = fg.diff(w)
    for w in [5, 20]:
        features[f'{prefix}ma{w}d'] = fg.rolling(w, min_periods=3).mean()
    features[f'{prefix}extreme_fear'] = (fg < 20).astype(float)
    features[f'{prefix}extreme_greed'] = (fg > 80).astype(float)
    return features


def _compute_cross_asset_features(cross_df, target_col, prefix='xa_'):
    """Rolling correlations and relative strength."""
    features = pd.DataFrame(index=cross_df.index)
    if target_col not in cross_df.columns:
        return features
    target_ret = cross_df[target_col].pct_change()
    for col in cross_df.columns:
        if col == target_col:
            continue
        other_ret = cross_df[col].pct_change()
        tag = f"{prefix}{col.lower()}"
        for w in [10, 30]:
            features[f'{tag}_corr{w}d'] = target_ret.rolling(w, min_periods=5).corr(other_ret)
        features[f'{tag}_relstr5d'] = (
            target_ret.rolling(5, min_periods=3).mean() -
            other_ret.rolling(5, min_periods=3).mean()
        ) * 100
    return features


def _classify_feature(feat):
    """Classify a feature as BASE, MACRO, SENTIMENT, or CROSS-ASSET."""
    if feat.startswith(('m_', 'vix_')):
        return 'MACRO'
    if feat.startswith('fg_'):
        return 'SENTIMENT'
    if feat.startswith('xa_'):
        return 'CROSS-ASSET'
    return 'BASE'


def build_all_features_v2(df_hourly, asset_name='BTC'):
    """
    Build ALL untrimmed features (technical + macro) for analysis.
    
    Technical: 36 features (same as build_all_features)
    Macro: ~80 features (VIX, DXY, SP500, Gold, Oil, Yields, EUR/USD, F&G, cross-asset)
    
    Daily macro data is forward-filled into 24/7 crypto hourly timestamps.
    """
    # Step 1: Build all 36 technical features
    df, base_cols = build_all_features(df_hourly)
    v2_cols = list(base_cols)
    added = 0

    if not os.path.exists(MACRO_DIR):
        print(f"    V2: {MACRO_DIR}/ not found -- returning base features only")
        print(f"    Run download_macro_data.py first to add macro features!")
        return df, v2_cols

    # Prepare date-based merge key
    df['_merge_date'] = pd.to_datetime(df['datetime']).dt.normalize()

    # Step 2: Macro indicators
    macro_df = _load_macro_csv('macro_daily.csv')
    if macro_df is not None:
        macro_feats = _compute_macro_features(macro_df, prefix='m_')
        vix_feats = _compute_vix_regime(macro_df)
        macro_all = pd.concat([macro_feats, vix_feats], axis=1)
        macro_all['_merge_date'] = macro_all.index.normalize()

        df = df.merge(macro_all, on='_merge_date', how='left')
        new_cols = [c for c in macro_all.columns if c != '_merge_date']
        v2_cols.extend(new_cols)
        added += len(new_cols)
        print(f"    Macro indicators: +{len(new_cols)} features")

    # Step 3: Fear & Greed
    fg_df = _load_macro_csv('fear_greed.csv')
    if fg_df is not None:
        fg_feats = _compute_fear_greed_features(fg_df)
        fg_feats['_merge_date'] = fg_feats.index.normalize()

        df = df.merge(fg_feats, on='_merge_date', how='left')
        new_cols = [c for c in fg_feats.columns if c != '_merge_date']
        v2_cols.extend(new_cols)
        added += len(new_cols)
        print(f"    Fear & Greed: +{len(new_cols)} features")

    # Step 4: Cross-asset correlations
    cross_df = _load_macro_csv('cross_asset.csv')
    if cross_df is not None:
        asset_map = {
            'BTC': 'BTC_USD', 'ETH': 'ETH_USD',
            'SOL': 'BTC_USD', 'XRP': 'BTC_USD', 'DOGE': 'BTC_USD',
        }
        target_col = asset_map.get(asset_name, 'BTC_USD')
        xa_feats = _compute_cross_asset_features(cross_df, target_col, prefix='xa_')
        if len(xa_feats.columns) > 0:
            xa_feats['_merge_date'] = xa_feats.index.normalize()
            df = df.merge(xa_feats, on='_merge_date', how='left')
            new_cols = [c for c in xa_feats.columns if c != '_merge_date']
            v2_cols.extend(new_cols)
            added += len(new_cols)
            print(f"    Cross-asset ({target_col}): +{len(new_cols)} features")

    # Cleanup
    df = df.drop(columns=['_merge_date'], errors='ignore')
    if '_merge_date' in v2_cols:
        v2_cols.remove('_merge_date')

    v2_cols = list(dict.fromkeys(v2_cols))
    v2_cols = [c for c in v2_cols if c in df.columns]

    # Forward-fill macro NaN then drop remaining NaN rows
    macro_cols = [c for c in v2_cols if c not in base_cols]
    if macro_cols:
        df[macro_cols] = df[macro_cols].ffill()

    # Drop rows that still have NaN (leading rows before macro data starts)
    df = df.dropna(subset=v2_cols + ['label']).reset_index(drop=True)

    n_base = len(base_cols)
    n_macro = len([c for c in v2_cols if _classify_feature(c) == 'MACRO'])
    n_sent = len([c for c in v2_cols if _classify_feature(c) == 'SENTIMENT'])
    n_xa = len([c for c in v2_cols if _classify_feature(c) == 'CROSS-ASSET'])
    print(f"    V2 total: {len(v2_cols)} features "
          f"({n_base} base + {n_macro} macro + {n_sent} sentiment + {n_xa} cross-asset)")
    print(f"    Rows after dropna: {len(df)}")

    return df, v2_cols


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
        ftype = _classify_feature(row['feature'])
        print(f"    {row['feature']:30s} {row['pct']:5.1f}% [{ftype:>10s}] {bar}{marker}")

    low_value = importance[importance['pct'] < 1.0]['feature'].tolist()
    print(f"\n    Low-value features (<1%): {len(low_value)}")
    for f in low_value:
        ftype = _classify_feature(f)
        print(f"      - {f} [{ftype}]")

    # Category importance breakdown
    from collections import Counter
    types = {f: _classify_feature(f) for f in feature_cols}
    type_imp = {}
    for _, row in importance.iterrows():
        t = types[row['feature']]
        type_imp[t] = type_imp.get(t, 0) + row['pct']
    print(f"\n    Importance by category:")
    for cat in ['BASE', 'MACRO', 'SENTIMENT', 'CROSS-ASSET']:
        if cat in type_imp:
            print(f"      {cat:15s}: {type_imp[cat]:5.1f}%")

    return importance


# ============================================================
# TEST 2: PERMUTATION IMPORTANCE
# ============================================================
def test_permutation_importance(df_features, feature_cols):
    """Shuffle each feature and measure accuracy drop."""
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
    for n_features in [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 75, len(feature_cols)]:
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
# GENERATE RECOMMENDATION
# ============================================================
def generate_recommendation(feature_cols, importance_df, ablation_df,
                            permutation_df, correlation_pairs, reduced_df):
    """Synthesize all tests into a clear recommendation."""
    print("\n" + "=" * 60)
    print("  RECOMMENDATION")
    print("=" * 60)

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

    score_df = pd.DataFrame([
        {'feature': f, 'score': s} for f, s in scores.items()
    ]).sort_values('score', ascending=False)

    keep = score_df[score_df['score'] >= 1]['feature'].tolist()
    maybe = score_df[(score_df['score'] >= -1) & (score_df['score'] < 1)]['feature'].tolist()
    drop = score_df[score_df['score'] < -1]['feature'].tolist()

    print(f"\n  KEEP ({len(keep)} features) - clearly useful:")
    for f in keep:
        ftype = _classify_feature(f)
        print(f"    + {f} (score: {scores[f]}) [{ftype}]")

    print(f"\n  MAYBE ({len(maybe)} features) - marginal, test both ways:")
    for f in maybe:
        ftype = _classify_feature(f)
        print(f"    ~ {f} (score: {scores[f]}) [{ftype}]")

    print(f"\n  DROP ({len(drop)} features) - likely noise:")
    for f in drop:
        ftype = _classify_feature(f)
        print(f"    - {f} (score: {scores[f]}) [{ftype}]")

    if reduced_df is not None and len(reduced_df) > 0:
        best_n = int(reduced_df.loc[reduced_df['accuracy'].idxmax(), 'n_features'])
        best_acc = reduced_df['accuracy'].max()
        full_acc = reduced_df[reduced_df['n_features'] == reduced_df['n_features'].max()]['accuracy'].values[0]
        print(f"\n  Best reduced set: Top {best_n} features -> {best_acc:.1f}% "
              f"(vs {full_acc:.1f}% with all {len(feature_cols)})")

    recommended = keep + maybe
    print(f"\n  RECOMMENDED FEATURE LIST ({len(recommended)} features):")
    print(f"  {recommended}")

    # Category breakdown
    from collections import Counter
    type_counts = Counter(_classify_feature(f) for f in recommended)
    print(f"\n  Category breakdown (recommended set):")
    for cat in ['BASE', 'MACRO', 'SENTIMENT', 'CROSS-ASSET']:
        if type_counts.get(cat, 0) > 0:
            print(f"    {cat:15s}: {type_counts[cat]:>2d} features")

    keep_types = Counter(_classify_feature(f) for f in keep)
    drop_types = Counter(_classify_feature(f) for f in drop)
    print(f"\n  Category survival rates:")
    all_types = Counter(_classify_feature(f) for f in feature_cols)
    for cat in ['BASE', 'MACRO', 'SENTIMENT', 'CROSS-ASSET']:
        total_cat = all_types.get(cat, 0)
        kept_cat = keep_types.get(cat, 0) + Counter(_classify_feature(f) for f in maybe).get(cat, 0)
        if total_cat > 0:
            print(f"    {cat:15s}: {kept_cat}/{total_cat} survived ({kept_cat/total_cat*100:.0f}%)")

    # Print copy-paste ready feature_cols
    print(f"\n  Copy-paste for crypto_trading_system.py:")
    print(f"  feature_cols = [")
    for i, f in enumerate(recommended):
        comma = ',' if i < len(recommended) - 1 else ''
        ftype = _classify_feature(f)
        print(f"      '{f}'{comma}  # {ftype}")
    print(f"  ]")

    return score_df, keep, maybe, drop


# ============================================================
# MAIN
# ============================================================
def analyze_asset(asset_name, diag_years=2, use_v2=True):
    """Run full feature analysis for one crypto asset."""
    version = "V2 (technical + macro)" if use_v2 else "V1 (technical only)"
    print(f"\n{'='*60}")
    print(f"  FEATURE ANALYSIS: {asset_name} [{version}]")
    print(f"{'='*60}")

    df_raw = load_data(asset_name)
    if df_raw is None:
        return

    if use_v2 and os.path.exists(MACRO_DIR):
        df_features, feature_cols = build_all_features_v2(df_raw, asset_name=asset_name)
    else:
        df_features, feature_cols = build_all_features(df_raw)
        if use_v2:
            print(f"  WARNING: macro_data/ not found -- falling back to V1")
            print(f"  Run download_macro_data.py first to enable V2 macro features")

    total_rows = len(df_features)

    # Trim to last N years
    diag_hours = diag_years * 365 * 24
    if total_rows > diag_hours:
        df_features = df_features.tail(diag_hours).reset_index(drop=True)
        print(f"  Trimmed: {total_rows:,} -> {len(df_features):,} rows (last {diag_years}y)")

    print(f"  Data: {len(df_features):,} hourly rows, {len(feature_cols)} features")

    if len(df_features) < 500:
        print(f"  Not enough data. Skipping.")
        return

    # Run all 5 tests
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
    version_tag = 'v2' if use_v2 and os.path.exists(MACRO_DIR) else 'v1'
    output_file = f'crypto_feature_analysis_{asset_name.lower()}_{version_tag}.csv'
    score_df.to_csv(output_file, index=False)
    print(f"  Results saved to {output_file}")

    return keep, maybe, drop


def main():
    parser = argparse.ArgumentParser(description='Crypto Feature Analysis Tool (V2 with Macro)')
    parser.add_argument('--asset', type=str, default=None,
                        help='Analyze specific asset (BTC, ETH, SOL, XRP, DOGE)')
    parser.add_argument('--years', type=int, default=None,
                        help='Years of data to analyze (1, 2, or 4)')
    parser.add_argument('--v1', action='store_true',
                        help='V1 only: skip macro features, analyze technical features only')
    args = parser.parse_args()

    use_v2 = not args.v1
    has_macro = os.path.exists(MACRO_DIR)

    print("=" * 60)
    print("  CRYPTO FEATURE IMPORTANCE & SELECTION ANALYSIS")
    if use_v2 and has_macro:
        print("  Mode: V2 (36 technical + ~80 macro/sentiment/cross-asset)")
    elif use_v2 and not has_macro:
        print("  Mode: V2 requested but macro_data/ not found -- will fall back to V1")
        print("  Tip: run download_macro_data.py first to add macro features")
    else:
        print("  Mode: V1 (36 technical features only)")
    print("  Hourly Crypto Trading System")
    print("=" * 60)

    # Asset selection
    if args.asset:
        assets = [args.asset.upper()]
    else:
        print("\nWhich crypto to analyze?")
        print("  1. All (BTC, ETH, SOL, XRP, DOGE)")
        print("  2. BTC only (recommended first)")
        print("  3. ETH only")
        print("  4. SOL only")
        print("  5. Choose specific")
        choice = input("Enter choice: ").strip()

        if choice == '2':
            assets = ['BTC']
        elif choice == '3':
            assets = ['ETH']
        elif choice == '4':
            assets = ['SOL']
        elif choice == '5':
            print(f"Available: {', '.join(ASSETS.keys())}")
            selected = input("Enter comma-separated names: ").strip().upper()
            assets = [a.strip() for a in selected.split(',') if a.strip() in ASSETS]
        else:
            assets = list(ASSETS.keys())

    # Data range selection
    if args.years:
        diag_years = args.years
    else:
        print("\nData range (crypto has years of 24/7 data -- trimming speeds it up):")
        print("  1. Last 4 years  (most data, slowest)")
        print("  2. Last 2 years  (recommended)")
        print("  3. Last 1 year   (fastest)")
        range_choice = input("Enter choice (1-3): ").strip()

        if range_choice == '1':
            diag_years = 4
        elif range_choice == '3':
            diag_years = 1
        else:
            diag_years = 2

    print(f"\nAssets: {', '.join(assets)}")
    print(f"Data range: last {diag_years} year{'s' if diag_years > 1 else ''}")
    print(f"Feature mode: {'V2 (technical + macro)' if use_v2 else 'V1 (technical only)'}")
    print(f"Running on: {MACHINE}")

    all_results = {}
    for asset in assets:
        result = analyze_asset(asset, diag_years=diag_years, use_v2=use_v2)
        if result:
            all_results[asset] = result

    # Cross-asset summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  CROSS-ASSET SUMMARY")
        print(f"{'='*60}")

        # Features to DROP across ALL cryptos
        all_drops = set(all_results[list(all_results.keys())[0]][2])
        for asset, (keep, maybe, drop) in all_results.items():
            all_drops = all_drops.intersection(set(drop))

        if all_drops:
            print(f"\n  Features to DROP across ALL cryptos:")
            for f in sorted(all_drops):
                ftype = _classify_feature(f)
                print(f"    - {f} [{ftype}]")

        # Features ESSENTIAL across ALL cryptos
        all_keeps = set(all_results[list(all_results.keys())[0]][0])
        for asset, (keep, maybe, drop) in all_results.items():
            all_keeps = all_keeps.intersection(set(keep))

        if all_keeps:
            print(f"\n  Features ESSENTIAL across ALL cryptos:")
            for f in sorted(all_keeps):
                ftype = _classify_feature(f)
                print(f"    + {f} [{ftype}]")

        # Category summary across assets
        from collections import Counter
        print(f"\n  Category survival summary:")
        for asset, (keep, maybe, drop) in all_results.items():
            recommended = keep + maybe
            types = Counter(_classify_feature(f) for f in recommended)
            parts = [f"{cat}: {types.get(cat, 0)}" for cat in ['BASE', 'MACRO', 'SENTIMENT', 'CROSS-ASSET'] if types.get(cat, 0) > 0]
            print(f"    {asset:6s}: {len(recommended)} features -> {', '.join(parts)}")

    print(f"\n{'='*60}")
    print("  ANALYSIS COMPLETE")
    print("  Next: update feature_cols in crypto_trading_system.py")
    print("  with the recommended set, then re-run diagnostic (Mode A)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
