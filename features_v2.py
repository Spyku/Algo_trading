"""
Features V2 — Macro & Sentiment Enhanced
==========================================
Wraps the original feature engineering and adds:
  - Macro indicators (VIX, DXY, S&P500, Gold, US10Y, EUR/USD, Oil)
  - Crypto Fear & Greed Index
  - Cross-asset correlations (BTC vs Nasdaq, etc.)
  - Derived macro features (changes, regime flags, rolling correlations)

The original features are UNTOUCHED. V2 simply appends new columns.
If macro data files are missing, falls back to original features silently.

Usage:
  from features_v2 import build_features_v2_daily, build_features_v2_hourly
  
  # For crypto daily system:
  df, feature_cols = build_features_v2_daily(df_raw, asset_name='BTC')
  
  # For DAX hourly system:
  df, feature_cols = build_features_v2_hourly(df_raw)

Test mode:
  python features_v2.py --test
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

MACRO_DIR = 'macro_data'


# ============================================================
# LOAD MACRO DATA (cached in memory after first load)
# ============================================================
_macro_cache = {}

def _load_csv(filename):
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


def load_macro_daily():
    return _load_csv('macro_daily.csv')

def load_macro_hourly():
    return _load_csv('macro_hourly.csv')

def load_fear_greed():
    return _load_csv('fear_greed.csv')

def load_cross_asset():
    return _load_csv('cross_asset.csv')


# ============================================================
# MACRO FEATURE ENGINEERING
# ============================================================
def compute_macro_features(macro_df, prefix=''):
    """
    Derive features from raw macro data.
    
    For each indicator (VIX, DXY, etc.), creates:
      - Level (raw value, z-scored)
      - 1d, 5d, 10d changes (momentum)
      - 5d, 20d rolling std (volatility of the indicator)
      - Regime flags (VIX>25, DXY trend, etc.)
    
    Returns DataFrame with feature columns.
    """
    features = pd.DataFrame(index=macro_df.index)

    for col in macro_df.columns:
        if col in ('fear_greed_label',):
            continue

        s = macro_df[col].astype(float)
        tag = f"{prefix}{col.lower()}"

        # Z-score of level (rolling 50-period)
        roll_mean = s.rolling(50, min_periods=10).mean()
        roll_std = s.rolling(50, min_periods=10).std()
        features[f'{tag}_zscore'] = (s - roll_mean) / (roll_std + 1e-10)

        # Momentum: % change over various windows
        for window in [1, 5, 10]:
            features[f'{tag}_chg{window}d'] = s.pct_change(window) * 100

        # Volatility of the indicator itself
        for window in [5, 20]:
            features[f'{tag}_vol{window}d'] = s.pct_change().rolling(window, min_periods=3).std() * 100

    return features


def compute_fear_greed_features(fg_df, prefix='fg_'):
    """
    Derive features from Fear & Greed Index.
    
    Creates:
      - Raw value (0-100)
      - Z-score
      - 1d, 5d, 10d changes
      - Extreme flags (< 20 = extreme fear, > 80 = extreme greed)
      - 5d, 20d moving averages
    """
    features = pd.DataFrame(index=fg_df.index)

    fg = fg_df['fear_greed'].astype(float)

    features[f'{prefix}value'] = fg
    features[f'{prefix}zscore'] = (fg - fg.rolling(50, min_periods=10).mean()) / (fg.rolling(50, min_periods=10).std() + 1e-10)

    for w in [1, 5, 10]:
        features[f'{prefix}chg{w}d'] = fg.diff(w)

    for w in [5, 20]:
        features[f'{prefix}ma{w}d'] = fg.rolling(w, min_periods=3).mean()

    # Extreme flags
    features[f'{prefix}extreme_fear'] = (fg < 20).astype(float)
    features[f'{prefix}extreme_greed'] = (fg > 80).astype(float)

    return features


def compute_cross_asset_features(cross_df, target_asset, prefix='xa_'):
    """
    Compute rolling correlations between target asset and other assets.
    
    For crypto: correlation with Nasdaq, S&P500
    For DAX: correlation with S&P500, USD pairs
    """
    features = pd.DataFrame(index=cross_df.index)

    if target_asset not in cross_df.columns:
        return features

    target_ret = cross_df[target_asset].pct_change()

    for col in cross_df.columns:
        if col == target_asset:
            continue

        other_ret = cross_df[col].pct_change()
        tag = f"{prefix}{col.lower()}"

        # Rolling correlation (10d, 30d)
        for w in [10, 30]:
            features[f'{tag}_corr{w}d'] = target_ret.rolling(w, min_periods=5).corr(other_ret)

        # Relative strength (target vs other, 5d)
        features[f'{tag}_relstr5d'] = (
            target_ret.rolling(5, min_periods=3).mean() -
            other_ret.rolling(5, min_periods=3).mean()
        ) * 100

    return features


# ============================================================
# VIX-SPECIFIC REGIME FEATURES
# ============================================================
def compute_vix_regime(macro_df):
    """Special VIX-based regime features that apply to all assets."""
    features = pd.DataFrame(index=macro_df.index)

    if 'VIX' not in macro_df.columns:
        return features

    vix = macro_df['VIX'].astype(float)

    # Regime buckets
    features['vix_low'] = (vix < 15).astype(float)        # complacency
    features['vix_normal'] = ((vix >= 15) & (vix < 25)).astype(float)
    features['vix_high'] = ((vix >= 25) & (vix < 35)).astype(float)  # elevated fear
    features['vix_extreme'] = (vix >= 35).astype(float)    # panic

    # VIX term structure proxy: is VIX rising or falling?
    features['vix_rising_5d'] = (vix.diff(5) > 0).astype(float)
    features['vix_spike'] = (vix.pct_change() > 0.15).astype(float)  # >15% daily spike

    return features


# ============================================================
# MAIN V2 BUILDERS
# ============================================================
def build_features_v2_daily(df_raw, asset_name='BTC', original_builder=None):
    """
    Build V2 features for daily crypto system.
    
    1. Call original build_features() to get base features
    2. Merge macro indicators (date-aligned)
    3. Merge Fear & Greed Index
    4. Merge cross-asset correlations
    5. Return (df, feature_cols) with all V2 features appended
    
    Args:
        df_raw: Raw OHLCV DataFrame
        asset_name: 'BTC', 'ETH', 'SOL', etc.
        original_builder: Function to call for base features.
                          Should return (df, feature_cols).
    """
    # Step 1: Original features
    if original_builder is None:
        from trading_system import build_features
        df, base_cols = build_features(df_raw)
    else:
        df, base_cols = original_builder(df_raw)

    v2_cols = list(base_cols)  # copy
    added = 0

    # Ensure we have a date index for merging
    if 'date' in df.columns:
        df['_merge_date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
    elif hasattr(df.index, 'date'):
        df['_merge_date'] = pd.to_datetime(df.index).tz_localize(None).normalize()
    else:
        print("  V2: Cannot determine date column for macro merge")
        return df, base_cols

    # Step 2: Macro indicators
    macro_df = load_macro_daily()
    if macro_df is not None:
        macro_feats = compute_macro_features(macro_df, prefix='m_')
        vix_feats = compute_vix_regime(macro_df)
        macro_all = pd.concat([macro_feats, vix_feats], axis=1)
        macro_all['_merge_date'] = macro_all.index.normalize()

        before = len(df.columns)
        df = df.merge(macro_all, on='_merge_date', how='left')
        new_cols = [c for c in macro_all.columns if c != '_merge_date']
        v2_cols.extend(new_cols)
        added += len(new_cols)

    # Step 3: Fear & Greed
    fg_df = load_fear_greed()
    if fg_df is not None:
        fg_feats = compute_fear_greed_features(fg_df)
        fg_feats['_merge_date'] = fg_feats.index.normalize()

        df = df.merge(fg_feats, on='_merge_date', how='left')
        new_cols = [c for c in fg_feats.columns if c != '_merge_date']
        v2_cols.extend(new_cols)
        added += len(new_cols)

    # Step 4: Cross-asset correlations
    cross_df = load_cross_asset()
    if cross_df is not None:
        # Map asset names to cross_asset column names
        asset_map = {
            'BTC': 'BTC_USD', 'ETH': 'ETH_USD',
            'SOL': 'BTC_USD', 'XRP': 'BTC_USD', 'DOGE': 'BTC_USD',  # use BTC as proxy
        }
        target = asset_map.get(asset_name, 'BTC_USD')
        xa_feats = compute_cross_asset_features(cross_df, target, prefix='xa_')
        if len(xa_feats.columns) > 0:
            xa_feats['_merge_date'] = xa_feats.index.normalize()
            df = df.merge(xa_feats, on='_merge_date', how='left')
            new_cols = [c for c in xa_feats.columns if c != '_merge_date']
            v2_cols.extend(new_cols)
            added += len(new_cols)

    # Cleanup
    if '_merge_date' in df.columns:
        df = df.drop(columns=['_merge_date'])
    if '_merge_date' in v2_cols:
        v2_cols.remove('_merge_date')

    # Remove any duplicate column names
    v2_cols = list(dict.fromkeys(v2_cols))

    # Only keep columns that actually exist in df
    v2_cols = [c for c in v2_cols if c in df.columns]

    print(f"  V2 features: {len(base_cols)} base + {added} macro/sentiment = {len(v2_cols)} total")

    return df, v2_cols


def build_features_v2_hourly(df_raw, original_builder=None):
    """
    Build V2 features for hourly DAX system.
    
    Same approach but uses hourly macro proxy (forward-filled daily data).
    
    Args:
        df_raw: Raw OHLCV DataFrame with 'datetime' column
        original_builder: Function to call for base features.
                          Should return (df, feature_cols).
    """
    # Step 1: Original features
    if original_builder is None:
        from hourly_trading_system import build_hourly_features
        df, base_cols = build_hourly_features(df_raw)
    else:
        df, base_cols = original_builder(df_raw)

    v2_cols = list(base_cols)
    added = 0

    # Get datetime for merging
    if 'datetime' in df.columns:
        df['_merge_dt'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    else:
        print("  V2 hourly: Cannot determine datetime column")
        return df, base_cols

    # Use hourly macro data (forward-filled from daily)
    macro_h = load_macro_hourly()
    if macro_h is not None:
        macro_feats = compute_macro_features(macro_h, prefix='m_')
        vix_feats = compute_vix_regime(macro_h)
        macro_all = pd.concat([macro_feats, vix_feats], axis=1)
        macro_all['_merge_dt'] = macro_all.index

        df = df.merge(macro_all, on='_merge_dt', how='left')
        new_cols = [c for c in macro_all.columns if c != '_merge_dt']
        v2_cols.extend(new_cols)
        added += len(new_cols)

    # Daily Fear & Greed (forward-fill into hourly)
    fg_df = load_fear_greed()
    if fg_df is not None:
        fg_feats = compute_fear_greed_features(fg_df)
        # Reindex to hourly
        hourly_idx = df['_merge_dt']
        fg_hourly = fg_feats.reindex(hourly_idx, method='ffill')
        fg_hourly.index = df.index  # align index

        for col in fg_hourly.columns:
            df[col] = fg_hourly[col].values
            v2_cols.append(col)
            added += 1

    # Cross-asset (DAX vs S&P500, Nasdaq)
    cross_df = load_cross_asset()
    if cross_df is not None:
        xa_feats = compute_cross_asset_features(cross_df, 'DAX', prefix='xa_')
        if len(xa_feats.columns) > 0:
            # Daily cross-asset, forward-fill to hourly
            xa_feats['_merge_date'] = xa_feats.index.normalize()
            df['_merge_date'] = df['_merge_dt'].dt.normalize()
            df = df.merge(xa_feats, on='_merge_date', how='left')
            new_cols = [c for c in xa_feats.columns if c != '_merge_date']
            v2_cols.extend(new_cols)
            added += len(new_cols)
            df = df.drop(columns=['_merge_date'], errors='ignore')

    # Cleanup
    if '_merge_dt' in df.columns:
        df = df.drop(columns=['_merge_dt'])
    if '_merge_dt' in v2_cols:
        v2_cols.remove('_merge_dt')
    if '_merge_date' in v2_cols:
        v2_cols.remove('_merge_date')

    v2_cols = list(dict.fromkeys(v2_cols))
    v2_cols = [c for c in v2_cols if c in df.columns]

    print(f"  V2 features: {len(base_cols)} base + {added} macro/sentiment = {len(v2_cols)} total")

    return df, v2_cols


# ============================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================
def analyze_feature_importance(df, feature_cols, target_col='label', top_n=30):
    """
    Quick feature importance analysis using Random Forest.
    Shows which V2 features matter most.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # Drop NaN rows
    valid = df[feature_cols + [target_col]].dropna()
    if len(valid) < 100:
        print(f"  Not enough valid rows ({len(valid)}) for importance analysis")
        return

    X = valid[feature_cols].values
    y = valid[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)

    importance = pd.Series(rf.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)

    print(f"\n  Top {top_n} features by importance:")
    print(f"  {'Feature':45s} {'Importance':>10s} {'Type':>10s}")
    print(f"  {'-'*67}")
    for feat, imp in importance.head(top_n).items():
        ftype = 'MACRO' if feat.startswith(('m_', 'vix_')) else \
                'SENTIMENT' if feat.startswith('fg_') else \
                'CROSS' if feat.startswith('xa_') else 'BASE'
        bar = '█' * int(imp * 200)
        print(f"  {feat:45s} {imp:>10.4f} {ftype:>10s} {bar}")

    # Summary
    base_imp = importance[[c for c in importance.index if not c.startswith(('m_', 'vix_', 'fg_', 'xa_'))]].sum()
    macro_imp = importance[[c for c in importance.index if c.startswith(('m_', 'vix_'))]].sum()
    sent_imp = importance[[c for c in importance.index if c.startswith('fg_')]].sum()
    cross_imp = importance[[c for c in importance.index if c.startswith('xa_')]].sum()
    total = base_imp + macro_imp + sent_imp + cross_imp

    print(f"\n  Importance by category:")
    print(f"    Base (technical):     {base_imp/total*100:5.1f}%")
    print(f"    Macro (VIX/DXY/...): {macro_imp/total*100:5.1f}%")
    print(f"    Sentiment (F&G):     {sent_imp/total*100:5.1f}%")
    print(f"    Cross-asset:         {cross_imp/total*100:5.1f}%")

    return importance


# ============================================================
# TEST MODE
# ============================================================
def test_v2():
    """Quick test: load data, build V2 features, show what we got."""
    print("=" * 60)
    print("  FEATURES V2 — TEST")
    print("=" * 60)

    # Check macro data exists
    if not os.path.exists(MACRO_DIR):
        print(f"\n  ERROR: {MACRO_DIR}/ not found. Run download_macro_data.py first!")
        return

    files = os.listdir(MACRO_DIR)
    print(f"\n  Macro data files: {files}")

    # Test daily (crypto)
    print(f"\n  --- Testing DAILY (BTC) ---")
    try:
        from trading_system import load_data, build_features
        df_raw = load_data('BTC')
        if df_raw is not None:
            df_v2, cols_v2 = build_features_v2_daily(df_raw, 'BTC', original_builder=build_features)
            print(f"  Shape: {df_v2.shape}")
            print(f"  Features: {len(cols_v2)}")
            print(f"  NaN in last row: {df_v2[cols_v2].iloc[-1].isna().sum()}")

            # Quick importance
            if 'label' in df_v2.columns:
                analyze_feature_importance(df_v2, cols_v2, 'label')
            elif 'target' in df_v2.columns:
                analyze_feature_importance(df_v2, cols_v2, 'target')
    except Exception as e:
        print(f"  Daily test error: {e}")

    # Test hourly (DAX)
    print(f"\n  --- Testing HOURLY (DAX) ---")
    try:
        from hourly_trading_system import load_data, build_hourly_features
        df_raw = load_data('DAX')
        if df_raw is not None:
            df_v2, cols_v2 = build_features_v2_hourly(df_raw, original_builder=build_hourly_features)
            print(f"  Shape: {df_v2.shape}")
            print(f"  Features: {len(cols_v2)}")
            print(f"  NaN in last row: {df_v2[cols_v2].iloc[-1].isna().sum()}")

            if 'label' in df_v2.columns:
                analyze_feature_importance(df_v2, cols_v2, 'label')
    except Exception as e:
        print(f"  Hourly test error: {e}")

    print(f"\n{'='*60}")
    print(f"  TEST COMPLETE")
    print(f"  If results look good, run full diagnostic with --v2 flag")
    print(f"{'='*60}")


if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        test_v2()
    else:
        print("Usage:")
        print("  python features_v2.py --test    # Quick test with feature importance")
        print("")
        print("Or import in your code:")
        print("  from features_v2 import build_features_v2_daily, build_features_v2_hourly")
