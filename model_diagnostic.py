"""
Model Diagnostic Tool
=====================
Tests all model combinations across rolling windows for all assets.
Outputs best_models.csv with optimal configuration per asset.
Models: RF, GB, LR, LGBM (no XGB)
Assets: BTC, ETH, SOL, XRP, DOGE, SMI, DAX, CAC40
"""

import sys
import os

# ============================================================
# SUPPRESS ALL WARNINGS (must be before other imports)
# ============================================================
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from hardware_config import (
    MACHINE, N_JOBS_PARALLEL, LGBM_DEVICE,
    get_cpu_models, get_gpu_models, get_all_models, get_diagnostic_models,
)

# Windows UTF-8 fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================================
# ASSET CONFIGURATION
# ============================================================
ASSETS = {
    'BTC':   {'source': 'binance', 'ticker': 'BTC/USDT',  'file': 'btc_hourly_data.csv'},
    'ETH':   {'source': 'binance', 'ticker': 'ETH/USDT',  'file': 'eth_hourly_data.csv'},
    'SOL':   {'source': 'binance', 'ticker': 'SOL/USDT',  'file': 'sol_hourly_data.csv'},
    'XRP':   {'source': 'binance', 'ticker': 'XRP/USDT',  'file': 'xrp_hourly_data.csv'},
    'DOGE':  {'source': 'binance', 'ticker': 'DOGE/USDT', 'file': 'doge_hourly_data.csv'},
    'SMI':   {'source': 'yfinance', 'ticker': '^SSMI',    'file': 'smi_hourly_data.csv'},
    'DAX':   {'source': 'yfinance', 'ticker': '^GDAXI',   'file': 'dax_hourly_data.csv'},
    'CAC40': {'source': 'yfinance', 'ticker': '^FCHI',     'file': 'cac40_hourly_data.csv'},
}

WINDOWS = [30, 50, 70, 90, 100]
STEP = 3  # Predict every 3rd day for speed
PREDICTION_HORIZON = 3  # 3-day prediction

# ============================================================
# DATA LOADING
# ============================================================
def load_data(asset_name):
    """Load hourly CSV data for an asset."""
    config = ASSETS[asset_name]
    filepath = config['file']
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found. Skipping {asset_name}.")
        return None
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def hourly_to_daily(df):
    """Aggregate hourly candles to daily OHLCV."""
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


# ============================================================
# FEATURE ENGINEERING (31 features)
# ============================================================
def build_features(df_daily):
    """
    Build all normalized features from daily OHLCV data.
    Returns DataFrame with features + labels, NaN rows dropped.
    """
    df = df_daily.copy()

    # --- Log Returns (9 direct features + 3 extended for spreads) ---
    for period in [1, 2, 3, 5, 7, 10, 14, 20, 30]:
        df[f'logret_{period}d'] = np.log(df['close'] / df['close'].shift(period))
    for period in [50, 100, 250]:
        df[f'logret_{period}d'] = np.log(df['close'] / df['close'].shift(period))

    # --- Log Return Spreads (6 features) ---
    df['spread_log10_log2']   = df['logret_10d']  - df['logret_2d']
    df['spread_log20_log2']   = df['logret_20d']  - df['logret_2d']
    df['spread_log30_log2']   = df['logret_30d']  - df['logret_2d']
    df['spread_log30_log10']  = df['logret_30d']  - df['logret_10d']
    df['spread_log7_log3']    = df['logret_7d']   - df['logret_3d']
    df['spread_log250_log10'] = df['logret_250d'] - df['logret_10d']

    # --- Price-to-SMA Ratios (4 features) ---
    df['sma20']  = df['close'].rolling(20).mean()
    df['sma50']  = df['close'].rolling(50).mean()
    df['sma200'] = df['close'].rolling(200).mean()
    df['price_to_sma20']  = df['close'] / df['sma20'] - 1
    df['price_to_sma50']  = df['close'] / df['sma50'] - 1
    df['price_to_sma200'] = df['close'] / df['sma200'] - 1
    df['sma20_to_sma50']  = df['sma20'] / df['sma50'] - 1

    # --- RSI 14 ---
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # --- Stochastic %K (14 periods) ---
    low14  = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low14) / (high14 - low14)

    # --- Bollinger Band Position ---
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - (bb_mid - 2 * bb_std)) / (4 * bb_std)

    # --- Z-Score (30 day) ---
    roll_mean = df['close'].rolling(30).mean()
    roll_std  = df['close'].rolling(30).std()
    df['zscore_30d'] = (df['close'] - roll_mean) / roll_std

    # --- ATR % ---
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    df['atr_pct'] = tr.rolling(14).mean() / df['close']

    # --- Volatility ---
    df['volatility_10d'] = df['logret_1d'].rolling(10).std()
    df['volatility_30d'] = df['logret_1d'].rolling(30).std()
    df['vol_ratio_10_30'] = df['volatility_10d'] / df['volatility_30d']

    # --- Volume Features (handle indices with 0/NaN volume) ---
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

    # --- Day of Week (cyclical encoding) ---
    dow = df['date'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # --- Labels: adaptive threshold (rolling 90-day median return) ---
    future_return = df['close'].shift(-PREDICTION_HORIZON) / df['close'] - 1
    rolling_median = future_return.rolling(90, min_periods=30).median().shift(PREDICTION_HORIZON)
    df['label'] = (future_return > rolling_median).astype(int)

    # --- Select feature columns ---
    feature_cols = [
        # Log returns (9)
        'logret_1d', 'logret_2d', 'logret_3d', 'logret_5d', 'logret_7d',
        'logret_10d', 'logret_14d', 'logret_20d', 'logret_30d',
        # Spreads (6)
        'spread_log10_log2', 'spread_log20_log2', 'spread_log30_log2',
        'spread_log30_log10', 'spread_log7_log3', 'spread_log250_log10',
        # Technical (14)
        'price_to_sma20', 'price_to_sma50', 'price_to_sma200', 'sma20_to_sma50',
        'rsi_14', 'stoch_k', 'bb_position', 'zscore_30d', 'atr_pct',
        'volatility_10d', 'volatility_30d', 'vol_ratio_10_30',
        'volume_ratio', 'volume_change',
        # Day of week (2)
        'dow_sin', 'dow_cos',
    ]

    # Keep only what we need
    keep_cols = ['date', 'close'] + feature_cols + ['label']
    df = df[keep_cols].copy()

    # Drop rows with NaN
    # Debug: show NaN counts before dropping
    nan_counts = df[feature_cols + ['label']].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"    Rows before dropna: {len(df)}")
        print(f"    Columns with NaN: {dict(nan_cols)}")

    df = df.dropna().reset_index(drop=True)
    print(f"    Rows after dropna: {len(df)}")

    return df, feature_cols


# ============================================================
# MODELS (4 models, no XGB)
# ============================================================
def get_models():
    """Return dict of model factory functions (from hardware_config)."""
    return get_all_models()


def get_all_combos(model_names):
    """Return all non-empty subsets of model names (15 combos for 4 models)."""
    combos = []
    for r in range(1, len(model_names) + 1):
        for combo in combinations(model_names, r):
            combos.append(list(combo))
    return combos


# ============================================================
# WALK-FORWARD EVALUATION (parallelized)
# ============================================================
def _eval_one_config_daily(features_np, labels_np, combo, window_size, n, step, model_factories):
    """Worker: evaluate one (window, combo) config using numpy arrays."""
    min_start = window_size + 50

    if n < min_start + 30:
        return None

    correct = 0
    total = 0

    for i in range(min_start, n, step):
        train_start = max(0, i - window_size)
        X_train = features_np[train_start:i]
        y_train = labels_np[train_start:i]
        X_test  = features_np[i:i+1]
        y_true  = labels_np[i]

        if len(np.unique(y_train)) < 2:
            continue
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            continue

        mean = X_train.mean(axis=0)
        std  = X_train.std(axis=0)
        std[std == 0] = 1.0
        X_train_s = (X_train - mean) / std
        X_test_s  = (X_test - mean) / std

        votes = []
        for model_name in combo:
            try:
                model = model_factories[model_name]()
                model.fit(X_train_s, y_train)
                pred = model.predict(X_test_s)[0]
                votes.append(pred)
            except Exception:
                continue

        if not votes:
            continue

        ensemble_pred = 1 if sum(votes) > len(votes) / 2 else 0
        if ensemble_pred == y_true:
            correct += 1
        total += 1

    if total == 0:
        return None
    return ('+'.join(combo), window_size, correct / total, total)


# Diagnostic models: ALL on CPU for parallel execution
DIAG_MODELS = get_diagnostic_models()


# ============================================================
# RUN DIAGNOSTIC FOR ONE ASSET (CPU parallel + GPU sequential)
# ============================================================
def run_diagnostic_for_asset(asset_name):
    """Test all configs: ALL in parallel on CPU."""
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC: {asset_name}")
    print(f"{'='*60}")

    df_raw = load_data(asset_name)
    if df_raw is None:
        return []

    print(f"  Loaded {len(df_raw):,} hourly candles")

    df_daily = hourly_to_daily(df_raw)
    print(f"  Aggregated to {len(df_daily):,} daily candles")

    df_features, feature_cols = build_features(df_daily)
    print(f"  Features built: {len(df_features):,} rows, {len(feature_cols)} features")

    if len(df_features) < 200:
        print(f"  WARNING: Not enough data for diagnostic. Skipping.")
        return []

    model_names = list(get_models().keys())
    combos = get_all_combos(model_names)

    all_configs = []
    for window in WINDOWS:
        for combo in combos:
            all_configs.append((combo, window))

    print(f"  {len(all_configs)} configs, ALL parallel ({N_JOBS_PARALLEL} workers)...")

    n = len(df_features)
    features_np = df_features[feature_cols].values.astype(np.float64)
    labels_np = df_features['label'].values.astype(np.int32)

    raw_results = Parallel(n_jobs=N_JOBS_PARALLEL, verbose=5)(
        delayed(_eval_one_config_daily)(
            features_np, labels_np, combo, window, n, STEP, DIAG_MODELS
        )
        for combo, window in all_configs
    )

    results = []
    best_acc = 0
    best_config = None

    for result in raw_results:
        if result is None:
            continue
        combo_name, window, acc, n_total = result
        results.append({
            'asset': asset_name,
            'window': window,
            'combo': combo_name,
            'accuracy': round(acc * 100, 2),
            'models': combo_name,
            'n_models': len(combo_name.split('+')),
        })
        if acc > best_acc:
            best_acc = acc
            best_config = (window, combo_name, acc)

        print(f"    w={window:3d} | {combo_name:20s} -> {acc*100:.1f}% (n={n_total})"
              f"{'  <-- BEST' if acc == best_acc else ''}")

    if best_config:
        print(f"\n  BEST for {asset_name}: window={best_config[0]}d, "
              f"combo={best_config[1]}, accuracy={best_config[2]*100:.1f}%")

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  MODEL DIAGNOSTIC TOOL")
    print("  Testing RF, GB, LR, LGBM (15 combos x 5 windows)")
    print("=" * 60)

    # Check which assets have data files
    available = []
    missing = []
    for name, config in ASSETS.items():
        if os.path.exists(config['file']):
            available.append(name)
        else:
            missing.append(name)

    print(f"\nAvailable assets: {', '.join(available)}")
    if missing:
        print(f"Missing data files: {', '.join(missing)} (run trading system to download)")

    if not available:
        print("No data files found. Please download data first.")
        return

    # Ask which assets to test
    print(f"\nWhich assets to diagnose?")
    print(f"  1. All available ({', '.join(available)})")
    print(f"  2. Crypto only (BTC, ETH, SOL, XRP, DOGE)")
    print(f"  3. Indices only (SMI, DAX, CAC40)")
    print(f"  4. Choose specific")
    choice = input("Enter choice (1-4): ").strip()

    if choice == '2':
        assets_to_test = [a for a in available if a in ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']]
    elif choice == '3':
        assets_to_test = [a for a in available if a in ['SMI', 'DAX', 'CAC40']]
    elif choice == '4':
        print(f"Available: {', '.join(available)}")
        selected = input("Enter comma-separated names (e.g., BTC,ETH,DAX): ").strip().upper()
        assets_to_test = [a.strip() for a in selected.split(',') if a.strip() in available]
    else:
        assets_to_test = available

    print(f"\nWill test: {', '.join(assets_to_test)}")
    print(f"Estimated time: ~{len(assets_to_test) * 5}-{len(assets_to_test) * 15} minutes")

    # Run diagnostic for each asset
    all_results = []
    for asset_name in assets_to_test:
        results = run_diagnostic_for_asset(asset_name)
        all_results.extend(results)

    if not all_results:
        print("\nNo results generated.")
        return

    # Create full results DataFrame
    df_all = pd.DataFrame(all_results)
    df_all.to_csv('diagnostic_full_results.csv', index=False)
    print(f"\nFull results saved to: diagnostic_full_results.csv ({len(df_all)} rows)")

    # Extract best config per asset
    best_rows = []
    for asset_name in assets_to_test:
        asset_results = df_all[df_all['asset'] == asset_name]
        if asset_results.empty:
            continue
        best_idx = asset_results['accuracy'].idxmax()
        best = asset_results.loc[best_idx]
        best_rows.append({
            'coin': asset_name,
            'best_window': int(best['window']),
            'best_combo': best['combo'],
            'accuracy': best['accuracy'],
            'models': best['models'],
        })

    df_best = pd.DataFrame(best_rows)
    df_best.to_csv('best_models.csv', index=False)

    print(f"\n{'='*60}")
    print(f"  BEST MODELS (saved to best_models.csv)")
    print(f"{'='*60}")
    for _, row in df_best.iterrows():
        print(f"  {row['coin']:6s} | window={row['best_window']:3d}d | "
              f"{row['best_combo']:20s} | accuracy={row['accuracy']:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
