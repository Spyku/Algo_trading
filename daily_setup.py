"""
Daily Setup — Data Update + Model Diagnostic
==============================================
Run once daily (or before a trading session) to:
  1. Download/update hourly price data from Yahoo Finance
  2. Run 75-config walk-forward diagnostic (5 windows x 15 combos)
  3. Find best model + window per asset using optimal V2 features
  4. Export results:
     - data/hourly_best_models.csv
     - data/setup_config.json (features + settings for generate_signals.py)

Usage:
  python daily_setup.py

After running this, use generate_signals.py for signal generation & backtest.
"""

import sys
import os

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from features_v2 import build_features_v2_hourly
from hardware_config import (
    MACHINE, N_JOBS_PARALLEL, LGBM_DEVICE,
    get_cpu_models, get_gpu_models, get_all_models, get_diagnostic_models,
)

try:
    import sklearn.utils.parallel
    sklearn.utils.parallel.warnings.warn = lambda *a, **kw: None
except Exception:
    pass

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# CONFIGURATION
# ============================================================
ASSETS = {
    'SMI':   {'ticker': '^SSMI',  'file': 'data/indices/smi_hourly_data.csv'},
    'DAX':   {'ticker': '^GDAXI', 'file': 'data/indices/dax_hourly_data.csv'},
    'CAC40': {'ticker': '^FCHI',  'file': 'data/indices/cac40_hourly_data.csv'},
}

PREDICTION_HORIZON = 4    # Predict 4 hours ahead
DIAG_STEP = 24            # Evaluate every 24h in diagnostic
DIAG_WINDOWS = [100, 200, 300, 500, 700]

# Optimal V2 features (from feature_analysis_v2.py — 76.1% accuracy)
OPTIMAL_V2_FEATURES = [
    'logret_240h',          # BASE - 10-day momentum
    'm_sp500_vol20d',       # MACRO - S&P500 volatility
    'm_vix_zscore',         # MACRO - VIX normalized
    'm_sp500_zscore',       # MACRO - S&P500 normalized
    'logret_24h',           # BASE - 1-day return
    'volatility_48h',       # BASE - 2-day volatility
    'xa_sp500_relstr5d',    # CROSS-ASSET - relative strength vs S&P
    'atr_pct_14h',          # BASE - ATR as % of price
    'sma20_to_sma50h',      # BASE - MA crossover ratio
    'zscore_50h',           # BASE - price z-score
    'xa_sp500_corr30d',     # CROSS-ASSET - 30d correlation with S&P
    'spread_120h_8h',       # BASE - fast/slow spread
    'xa_nasdaq_corr10d',    # CROSS-ASSET - 10d correlation with Nasdaq
    'm_gold_vol20d',        # MACRO - gold volatility
    'fg_zscore',            # SENTIMENT - Fear & Greed normalized
]


# ============================================================
# DATA DOWNLOAD (yfinance)
# ============================================================
def _normalize_yf_chunk(chunk):
    """Normalize a yfinance DataFrame chunk to flat columns."""
    df = chunk.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.name or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    rename_map = {}
    for col in df.columns:
        cl = str(col).strip().lower()
        if cl in ('datetime', 'date', 'index', 'timestamp'):
            rename_map[col] = 'datetime'
        elif cl == 'open':   rename_map[col] = 'open'
        elif cl == 'high':   rename_map[col] = 'high'
        elif cl == 'low':    rename_map[col] = 'low'
        elif cl == 'close':  rename_map[col] = 'close'
        elif cl == 'volume': rename_map[col] = 'volume'
    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()]

    if 'datetime' not in df.columns:
        for fallback in ['Datetime', 'Date', 'index']:
            if fallback in df.columns:
                df = df.rename(columns={fallback: 'datetime'})
                break
    if 'datetime' not in df.columns:
        return None

    df['datetime'] = pd.to_datetime(df['datetime'])
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_convert(None)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            df[col] = 0

    df['timestamp'] = ((df['datetime'] - pd.Timestamp('1970-01-01')).dt.total_seconds() * 1000).astype(int)
    return df[['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]


def download_yfinance(asset_name, update_only=True):
    """Download or update hourly data from Yahoo Finance."""
    import yfinance as yf
    config = ASSETS[asset_name]
    filepath = config['file']
    ticker = config['ticker']

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if os.path.exists(filepath) and update_only:
        df_existing = pd.read_csv(filepath)
        df_existing['datetime'] = pd.to_datetime(df_existing['datetime'])
        last_dt = df_existing['datetime'].max()
        print(f"  Updating {asset_name} from {last_dt}...")

        start = last_dt - timedelta(days=1)
        end = datetime.now()
        all_chunks = []

        while start < end:
            chunk_end = min(start + timedelta(days=59), end)
            try:
                chunk = yf.download(ticker, start=start, end=chunk_end,
                                    interval='1h', progress=False)
                if len(chunk) > 0:
                    normalized = _normalize_yf_chunk(chunk)
                    if normalized is not None:
                        all_chunks.append(normalized)
            except Exception as e:
                print(f"    Warning: {e}")
            start = chunk_end

        if all_chunks:
            df_new = pd.concat(all_chunks, ignore_index=True)
            combined = pd.concat([df_existing, df_new], ignore_index=True)
            combined['datetime'] = pd.to_datetime(combined['datetime'])
            combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            combined.to_csv(filepath, index=False)
            print(f"  {asset_name}: {len(combined):,} total hourly candles")
        else:
            print(f"  {asset_name}: No new data")
    else:
        print(f"  Downloading {asset_name} full history (~2 years hourly)...")
        end = datetime.now()
        start = end - timedelta(days=730)
        all_chunks = []
        current = start

        while current < end:
            chunk_end = min(current + timedelta(days=59), end)
            try:
                chunk = yf.download(ticker, start=current, end=chunk_end,
                                    interval='1h', progress=False)
                if len(chunk) > 0:
                    normalized = _normalize_yf_chunk(chunk)
                    if normalized is not None:
                        all_chunks.append(normalized)
                        print(f"    Chunk: {current.date()} to {chunk_end.date()} -> {len(normalized)} rows")
            except Exception as e:
                print(f"    Warning: {e}")
            current = chunk_end
            time.sleep(0.3)

        if not all_chunks:
            print(f"  ERROR: No data downloaded for {asset_name}")
            return

        df_new = pd.concat(all_chunks, ignore_index=True)
        df_new = df_new.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        df_new.to_csv(filepath, index=False)
        print(f"  {asset_name}: {len(df_new):,} total hourly candles saved")


def update_all_data(assets_list):
    """Update data files for requested assets."""
    print("\n" + "=" * 60)
    print("  UPDATING DATA")
    print("=" * 60)
    for asset_name in assets_list:
        config = ASSETS[asset_name]
        file_exists = os.path.exists(config['file'])
        if file_exists:
            print(f"\n[{asset_name}] File exists. Updating...")
        else:
            print(f"\n[{asset_name}] File not found. Downloading fresh...")
        download_yfinance(asset_name, update_only=file_exists)
    print("\nData update complete.")


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


# ============================================================
# HOURLY FEATURE ENGINEERING (V1 base — used by features_v2)
# ============================================================
def build_hourly_features(df_hourly):
    """
    Build base features from hourly OHLCV data.
    Returns (DataFrame with features + labels, feature_cols list).
    """
    df = df_hourly.copy()

    # Log returns
    for period in [1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 120, 240]:
        df[f'logret_{period}h'] = np.log(df['close'] / df['close'].shift(period))

    # Spreads
    df['spread_24h_4h']   = df['logret_24h']  - df['logret_4h']
    df['spread_48h_4h']   = df['logret_48h']  - df['logret_4h']
    df['spread_120h_8h']  = df['logret_120h'] - df['logret_8h']
    df['spread_240h_24h'] = df['logret_240h'] - df['logret_24h']
    df['spread_48h_12h']  = df['logret_48h']  - df['logret_12h']
    df['spread_120h_12h'] = df['logret_120h'] - df['logret_12h']

    # SMA ratios
    df['sma20h']  = df['close'].rolling(20).mean()
    df['sma50h']  = df['close'].rolling(50).mean()
    df['sma100h'] = df['close'].rolling(100).mean()
    df['sma200h'] = df['close'].rolling(200).mean()
    df['price_to_sma20h']  = df['close'] / df['sma20h'] - 1
    df['price_to_sma50h']  = df['close'] / df['sma50h'] - 1
    df['price_to_sma100h'] = df['close'] / df['sma100h'] - 1
    df['sma20_to_sma50h']  = df['sma20h'] / df['sma50h'] - 1

    # RSI 14
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14h'] = 100 - (100 / (1 + rs))

    # Stochastic %K
    low14  = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['stoch_k_14h'] = 100 * (df['close'] - low14) / (high14 - low14)

    # Bollinger Band Position
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position_20h'] = (df['close'] - (bb_mid - 2 * bb_std)) / (4 * bb_std)

    # Z-Score
    roll_mean = df['close'].rolling(50).mean()
    roll_std  = df['close'].rolling(50).std()
    df['zscore_50h'] = (df['close'] - roll_mean) / roll_std

    # ATR %
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    df['atr_pct_14h'] = tr.rolling(14).mean() / df['close']

    # Intraday range
    df['intraday_range'] = (df['high'] - df['low']) / df['close']

    # Volatility
    df['volatility_12h'] = df['logret_1h'].rolling(12).std()
    df['volatility_48h'] = df['logret_1h'].rolling(48).std()
    df['vol_ratio_12_48'] = df['volatility_12h'] / df['volatility_48h']

    # Volume
    if df['volume'].sum() == 0 or df['volume'].isna().all():
        df['volume_ratio_h'] = 1.0
    else:
        df['volume'] = df['volume'].replace(0, np.nan).ffill().bfill()
        vol_sma = df['volume'].rolling(20).mean()
        df['volume_ratio_h'] = df['volume'] / vol_sma
        df['volume_ratio_h'] = df['volume_ratio_h'].fillna(1.0)

    # Time encoding
    hour = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    dow = df['datetime'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # Labels: predict direction in PREDICTION_HORIZON hours
    future_return = df['close'].shift(-PREDICTION_HORIZON) / df['close'] - 1
    rolling_median = future_return.rolling(200, min_periods=50).median().shift(PREDICTION_HORIZON)
    df['label'] = (future_return > rolling_median).astype(int)

    # V1 base feature list (20 features)
    feature_cols = [
        'logret_1h', 'logret_2h', 'logret_3h', 'logret_6h',
        'logret_12h', 'logret_24h', 'logret_48h', 'logret_72h',
        'logret_240h', 'spread_120h_8h', 'sma20_to_sma50h',
        'rsi_14h', 'bb_position_20h', 'zscore_50h', 'atr_pct_14h',
        'intraday_range', 'volatility_48h',
        'hour_sin', 'dow_sin', 'dow_cos',
    ]

    display_cols = ['spread_24h_4h']
    keep_cols = ['datetime', 'close', 'high', 'low', 'volume'] + feature_cols + display_cols + ['label']
    df = df[keep_cols].copy()

    nan_counts = df[feature_cols + ['label']].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"    Rows before dropna: {len(df)}")
        top_nan = sorted({k: v for k, v in dict(nan_cols).items() if v > 0}.items(), key=lambda x: -x[1])[:5]
        print(f"    Top NaN columns: {dict(top_nan)}")

    df = df.dropna().reset_index(drop=True)
    print(f"    Rows after dropna: {len(df)}")
    return df, feature_cols


# ============================================================
# MODELS
# ============================================================
ALL_MODELS = get_all_models()
DIAG_MODELS = get_diagnostic_models()


# ============================================================
# DIAGNOSTIC ENGINE
# ============================================================
def _eval_one_config(features_np, labels_np, combo, window, n, diag_step, model_factories):
    """Worker: evaluate one (window, combo) config using numpy arrays."""
    min_start = window + 50
    if n < min_start + 50:
        return None

    correct, total = 0, 0
    for i in range(min_start, n, diag_step):
        train_start = max(0, i - window)
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
                votes.append(model.predict(X_test_s)[0])
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
    return ('+'.join(combo), window, correct / total, total)


def run_diagnostic_for_asset(asset_name, df_features, feature_cols):
    """Run diagnostic: ALL configs in parallel."""
    combos = []
    model_names = list(ALL_MODELS.keys())
    for r in range(1, len(model_names) + 1):
        for combo in combinations(model_names, r):
            combos.append(list(combo))

    all_configs = [(combo, w) for w in DIAG_WINDOWS for combo in combos]
    print(f"  {len(all_configs)} configs, parallel ({N_JOBS_PARALLEL} workers, step={DIAG_STEP})...")

    n = len(df_features)
    features_np = df_features[feature_cols].values.astype(np.float64)
    labels_np = df_features['label'].values.astype(np.int32)

    all_results = Parallel(n_jobs=N_JOBS_PARALLEL, verbose=5)(
        delayed(_eval_one_config)(
            features_np, labels_np, combo, window, n, DIAG_STEP, DIAG_MODELS
        )
        for combo, window in all_configs
    )

    best_acc, best_config = 0, None
    for result in all_results:
        if result is None:
            continue
        combo_name, window, acc, n_total = result
        if acc > best_acc:
            best_acc = acc
            best_config = {
                'coin': asset_name,
                'best_window': window,
                'best_combo': combo_name,
                'accuracy': round(acc * 100, 2),
                'models': combo_name,
            }
        print(f"    w={window:4d}h | {combo_name:20s} | acc={acc*100:5.1f}% (n={n_total})"
              f"{'  <-- BEST' if acc == best_acc else ''}")

    return best_config


def run_full_diagnostic(assets_list):
    """Run diagnostic across all assets with V2 features. Save results."""
    print("\n" + "=" * 60)
    print("  RUNNING DIAGNOSTIC (V2 optimal features)")
    print("=" * 60)

    best_models = []
    for asset_name in assets_list:
        print(f"\n--- {asset_name} ---")
        df_raw = load_data(asset_name)
        if df_raw is None:
            continue

        # Build V2 features, use optimal 15
        df_v2, _all_cols = build_features_v2_hourly(df_raw, original_builder=build_hourly_features)
        feature_cols = [c for c in OPTIMAL_V2_FEATURES if c in df_v2.columns]
        missing = [c for c in OPTIMAL_V2_FEATURES if c not in df_v2.columns]
        if missing:
            print(f"  WARNING: Missing features: {missing}")

        df_features = df_v2.dropna(subset=feature_cols + ['label']).reset_index(drop=True)

        if len(df_features) < 500:
            print(f"  Not enough data ({len(df_features)} rows). Need 500+. Skipping.")
            continue

        print(f"  {len(df_features):,} hourly rows, {len(feature_cols)} features")

        best_config = run_diagnostic_for_asset(asset_name, df_features, feature_cols)
        if best_config:
            best_models.append(best_config)
            print(f"\n  >>> BEST: window={best_config['best_window']}h, "
                  f"combo={best_config['best_combo']}, "
                  f"acc={best_config['accuracy']:.1f}%")

    if best_models:
        os.makedirs('data', exist_ok=True)
        df_best = pd.DataFrame(best_models)
        df_best.to_csv('data/hourly_best_models.csv', index=False)
        print(f"\n{'='*60}")
        print("  DIAGNOSTIC RESULTS (saved to data/hourly_best_models.csv)")
        print(f"{'='*60}")
        for row in best_models:
            print(f"  {row['coin']:6s} | window={row['best_window']:4d}h | "
                  f"{row['best_combo']:20s} | {row['accuracy']:.1f}%")
    else:
        print("\nNo diagnostic results.")

    return best_models


def export_setup_config(best_models):
    """Export setup config JSON for generate_signals.py."""
    config = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prediction_horizon': PREDICTION_HORIZON,
        'optimal_features': OPTIMAL_V2_FEATURES,
        'n_features': len(OPTIMAL_V2_FEATURES),
        'best_models': best_models,
    }
    os.makedirs('data', exist_ok=True)
    with open('data/setup_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Setup config saved to data/setup_config.json")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  DAILY SETUP — Data + Diagnostic")
    print(f"  Indices: {', '.join(ASSETS.keys())}")
    print(f"  Prediction: {PREDICTION_HORIZON}h ahead")
    print(f"  Features: {len(OPTIMAL_V2_FEATURES)} optimal V2")
    print("=" * 60)

    assets_list = list(ASSETS.keys())

    # Step 1: Update data
    update_all_data(assets_list)

    # Step 2: Run diagnostic
    best_models = run_full_diagnostic(assets_list)

    # Step 3: Export config
    if best_models:
        export_setup_config(best_models)

    print(f"\n{'='*60}")
    print("  DAILY SETUP COMPLETE")
    print("  Next: python generate_signals.py")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
