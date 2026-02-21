"""
Hourly Index ML Trading System
================================
Trades European indices (SMI, DAX, CAC40) on hourly candles.
Features adapted for intraday patterns.
Modes:
  A: Update data -> Diagnostic -> Best models -> Signals -> Chart
  B: Update data -> Load hourly_best_models.csv -> Signals -> Chart
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

import time
import json
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

# Monkey-patch sklearn parallel warning
try:
    import sklearn.utils.parallel
    sklearn.utils.parallel.warnings.warn = lambda *a, **kw: None
except Exception:
    pass

# Windows UTF-8 fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# CONFIGURATION
# ============================================================
ASSETS = {
    'SMI':   {'ticker': '^SSMI',  'file': 'smi_hourly_data.csv'},
    'DAX':   {'ticker': '^GDAXI', 'file': 'dax_hourly_data.csv'},
    'CAC40': {'ticker': '^FCHI',  'file': 'cac40_hourly_data.csv'},
}

PREDICTION_HORIZON = 4    # Predict 4 hours ahead (~half a trading day)
DEFAULT_WINDOW = 400      # Default training window (hours)
REPLAY_HOURS = 200        # Hours of history for chart (~25 trading days)
DIAG_STEP = 24             # Diagnostic: evaluate every 24 hours (~1 per trading day)
DIAG_WINDOWS = [100, 200, 300, 500, 700]  # Training windows to test


# ============================================================
# DATA DOWNLOAD (yfinance)
# ============================================================
def _normalize_yf_chunk(chunk):
    """Normalize a yfinance DataFrame chunk to flat columns: datetime, open, high, low, close, volume."""
    df = chunk.copy()

    # Step 1: Flatten MultiIndex columns (yfinance 1.2+ returns these)
    if isinstance(df.columns, pd.MultiIndex):
        # Drop the ticker level, keep the price level (Close, Open, etc.)
        df.columns = df.columns.get_level_values(0)

    # Step 2: Reset index to get datetime as a column
    if df.index.name or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Step 3: Normalize column names to lowercase
    rename_map = {}
    for col in df.columns:
        cl = str(col).strip().lower()
        if cl in ('datetime', 'date', 'index', 'timestamp'):
            rename_map[col] = 'datetime'
        elif cl == 'open':
            rename_map[col] = 'open'
        elif cl == 'high':
            rename_map[col] = 'high'
        elif cl == 'low':
            rename_map[col] = 'low'
        elif cl == 'close':
            rename_map[col] = 'close'
        elif cl == 'volume':
            rename_map[col] = 'volume'
    df = df.rename(columns=rename_map)

    # Step 4: Remove duplicate columns (yfinance sometimes returns 'Price' etc.)
    df = df.loc[:, ~df.columns.duplicated()]

    # Step 5: Ensure datetime exists
    if 'datetime' not in df.columns:
        # Try common fallbacks
        for fallback in ['Datetime', 'Date', 'index']:
            if fallback in df.columns:
                df = df.rename(columns={fallback: 'datetime'})
                break

    if 'datetime' not in df.columns:
        return None

    df['datetime'] = pd.to_datetime(df['datetime'])
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_convert(None)

    # Step 6: Ensure all required columns
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
        # Full download (~730 days in 59-day chunks)
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
                    else:
                        print(f"    Chunk: {current.date()} to {chunk_end.date()} -> FAILED to normalize")
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


# ============================================================
# HOURLY FEATURE ENGINEERING
# ============================================================
def build_hourly_features(df_hourly):
    """
    Build features from hourly OHLCV data.
    Adapted for intraday trading on indices.
    Returns (DataFrame with features + labels, feature_cols list).
    """
    df = df_hourly.copy()

    # --- Log Returns (hourly periods) ---
    for period in [1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 120, 240]:
        df[f'logret_{period}h'] = np.log(df['close'] / df['close'].shift(period))

    # --- Log Return Spreads ---
    df['spread_24h_4h']   = df['logret_24h']  - df['logret_4h']
    df['spread_48h_4h']   = df['logret_48h']  - df['logret_4h']
    df['spread_120h_8h']  = df['logret_120h'] - df['logret_8h']
    df['spread_240h_24h'] = df['logret_240h'] - df['logret_24h']
    df['spread_48h_12h']  = df['logret_48h']  - df['logret_12h']
    df['spread_120h_12h'] = df['logret_120h'] - df['logret_12h']

    # --- Price-to-SMA Ratios ---
    df['sma20h']  = df['close'].rolling(20).mean()
    df['sma50h']  = df['close'].rolling(50).mean()
    df['sma100h'] = df['close'].rolling(100).mean()
    df['sma200h'] = df['close'].rolling(200).mean()
    df['price_to_sma20h']  = df['close'] / df['sma20h'] - 1
    df['price_to_sma50h']  = df['close'] / df['sma50h'] - 1
    df['price_to_sma100h'] = df['close'] / df['sma100h'] - 1
    df['sma20_to_sma50h']  = df['sma20h'] / df['sma50h'] - 1

    # --- RSI 14 (hourly) ---
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

    # --- Volatility ---
    df['volatility_12h'] = df['logret_1h'].rolling(12).std()
    df['volatility_48h'] = df['logret_1h'].rolling(48).std()
    df['vol_ratio_12_48'] = df['volatility_12h'] / df['volatility_48h']

    # --- Volume Features (indices have 0 volume, set neutral) ---
    if df['volume'].sum() == 0 or df['volume'].isna().all():
        df['volume_ratio_h'] = 1.0
    else:
        df['volume'] = df['volume'].replace(0, np.nan).ffill().bfill()
        vol_sma = df['volume'].rolling(20).mean()
        df['volume_ratio_h'] = df['volume'] / vol_sma
        df['volume_ratio_h'] = df['volume_ratio_h'].fillna(1.0)

    # --- Hour of Day (cyclical) ---
    hour = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # --- Day of Week (cyclical) ---
    dow = df['datetime'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # --- Labels: predict direction in PREDICTION_HORIZON hours ---
    future_return = df['close'].shift(-PREDICTION_HORIZON) / df['close'] - 1
    rolling_median = future_return.rolling(200, min_periods=50).median().shift(PREDICTION_HORIZON)
    df['label'] = (future_return > rolling_median).astype(int)

    # Feature columns (20 features — trimmed via feature_analysis.py)
    # Dropped: volume_ratio_h (always 1), vol_ratio_12_48, hour_cos (noise),
    #   logret_4h/8h/120h, spread_24h_4h/48h_4h/48h_12h/120h_12h/240h_24h (redundant),
    #   price_to_sma20h/50h/100h, stoch_k_14h, volatility_12h (redundant twins)
    feature_cols = [
        # Log returns (8)
        'logret_1h', 'logret_2h', 'logret_3h', 'logret_6h',
        'logret_12h', 'logret_24h', 'logret_48h', 'logret_72h',
        'logret_240h',
        # Spreads (1 — best survivor)
        'spread_120h_8h',
        # SMA (1)
        'sma20_to_sma50h',
        # Oscillators & indicators (4)
        'rsi_14h', 'bb_position_20h', 'zscore_50h',
        'atr_pct_14h',
        # Range (1)
        'intraday_range',
        # Volatility (1)
        'volatility_48h',
        # Time encoding (3)
        'hour_sin', 'dow_sin', 'dow_cos',
    ]

    # Extra cols needed for signal display (not model features)
    display_cols = ['spread_24h_4h']
    keep_cols = ['datetime', 'close', 'high', 'low', 'volume'] + feature_cols + display_cols + ['label']
    df = df[keep_cols].copy()

    # Debug: show NaN counts before dropping
    nan_counts = df[feature_cols + ['label']].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"    Rows before dropna: {len(df)}")
        nan_summary = {k: v for k, v in dict(nan_cols).items() if v > 0}
        top_nan = sorted(nan_summary.items(), key=lambda x: -x[1])[:5]
        print(f"    Top NaN columns: {dict(top_nan)}")

    df = df.dropna().reset_index(drop=True)
    print(f"    Rows after dropna: {len(df)}")

    return df, feature_cols


# ============================================================
# MODELS
# ============================================================
# MODELS (from hardware_config -- auto-detects LAPTOP vs DESKTOP)
ALL_MODELS = get_all_models()


# ============================================================
# SIGNAL GENERATION (hourly walk-forward)
# ============================================================
def generate_signals(asset_name, model_names, window_size, replay_hours=REPLAY_HOURS):
    """
    Generate hourly signals using walk-forward training.
    Returns list of signal dicts.
    """
    print(f"\n  Generating hourly signals for {asset_name} "
          f"(models={'+'.join(model_names)}, window={window_size}h, "
          f"replay={replay_hours}h)...")

    df_raw = load_data(asset_name)
    if df_raw is None:
        return []

    df_features, feature_cols = build_hourly_features(df_raw)

    n = len(df_features)
    start_idx = max(window_size + 50, n - replay_hours)

    signals = []
    count = 0

    for i in range(start_idx, n):
        row = df_features.iloc[i]
        dt_str = row['datetime'].strftime('%Y-%m-%d %H:%M')

        # Training data
        train_start = max(0, i - window_size)
        train = df_features.iloc[train_start:i]
        X_train = train[feature_cols]
        y_train = train['label'].values

        X_test = df_features.iloc[i:i+1][feature_cols]

        if len(np.unique(y_train)) < 2:
            continue
        if X_train.isnull().any().any() or X_test.isnull().any().any():
            continue

        # Scale (keep as DataFrame with feature names)
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
        X_test_s  = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

        # Get predictions from each model
        votes = []
        probas = []

        for model_name in model_names:
            try:
                model = ALL_MODELS[model_name]()
                model.fit(X_train_s, y_train)
                pred = model.predict(X_test_s)[0]
                proba = model.predict_proba(X_test_s)[0]
                votes.append(pred)
                probas.append(proba[1])
            except Exception:
                continue

        if not votes:
            continue

        # Ensemble: majority vote
        buy_votes = sum(votes)
        total_votes = len(votes)
        buy_ratio = buy_votes / total_votes

        if buy_ratio > 0.5:
            signal = 'BUY'
        elif buy_ratio == 0:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        # Confidence
        avg_proba = np.mean(probas)
        if signal == 'SELL':
            confidence = (1 - avg_proba) * 100
        else:
            confidence = avg_proba * 100

        # Check actual outcome
        actual = None
        if i + PREDICTION_HORIZON < n:
            future_close = df_features.iloc[i + PREDICTION_HORIZON]['close']
            actual_return = (future_close / row['close'] - 1) * 100
            actual = 'UP' if actual_return > 0 else 'DOWN'

        signals.append({
            'datetime': dt_str,
            'close': round(float(row['close']), 2),
            'signal': signal,
            'confidence': round(float(confidence), 1),
            'buy_votes': int(buy_votes),
            'total_votes': int(total_votes),
            'rsi': round(float(row['rsi_14h']), 1),
            'bb_position': round(float(row['bb_position_20h']), 3),
            'hourly_change': round(float(row['logret_1h'] * 100), 3),
            'intraday_range': round(float(row['intraday_range'] * 100), 3),
            'spread_24h_4h': round(float(row['spread_24h_4h'] * 100), 2),
            'spread_120h_8h': round(float(row['spread_120h_8h'] * 100), 2),
            'actual': actual,
        })

        count += 1
        if count % 50 == 0:
            print(f"    [{count}] {dt_str}: {signal} ({confidence:.0f}%) "
                  f"| price={row['close']:,.2f}")

    print(f"  Generated {len(signals)} hourly signals for {asset_name}")
    return signals


# ============================================================
# PORTFOLIO SIMULATION ($1000 starting)
# ============================================================
def simulate_portfolio(signals, initial=1000):
    """
    Simulate a $1000 portfolio on hourly signals.
    BUY = invest, SELL = cash, HOLD = maintain.
    """
    if not signals:
        return signals

    portfolio = initial
    hold_value = initial
    position = 'cash'
    entry_price = None
    start_price = signals[0]['close']

    for sig in signals:
        price = sig['close']
        hold_value = initial * (price / start_price)

        if sig['signal'] == 'BUY' and position == 'cash':
            position = 'invested'
            entry_price = price
        elif sig['signal'] == 'SELL' and position == 'invested':
            pnl_ratio = price / entry_price
            portfolio *= pnl_ratio
            position = 'cash'
            entry_price = None

        if position == 'invested' and entry_price:
            current_value = portfolio * (price / entry_price)
        else:
            current_value = portfolio

        sig['portfolio_value'] = round(current_value, 2)
        sig['hold_value'] = round(hold_value, 2)

    return signals


# ============================================================
# DIAGNOSTIC (CPU combos parallel, GPU combos sequential)
# ============================================================
def _eval_one_config(features_np, labels_np, combo, window, n, diag_step, model_factories):
    """Worker: evaluate one (window, combo) config using numpy arrays."""
    min_start = window + 50
    if n < min_start + 50:
        return None

    correct = 0
    total = 0

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

        # Scale with numpy (much faster than sklearn+DataFrame)
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
    return ('+'.join(combo), window, correct / total, total)


# Diagnostic models: ALL on CPU for parallel execution (GPU overhead kills small data)
DIAG_MODELS = get_diagnostic_models()


def run_diagnostic_for_asset(asset_name, df_features, feature_cols):
    """Run diagnostic: ALL configs in parallel on CPU (faster than GPU for small data)."""
    combos = []
    model_names = list(ALL_MODELS.keys())
    for r in range(1, len(model_names) + 1):
        for combo in combinations(model_names, r):
            combos.append(list(combo))

    all_configs = []
    for window in DIAG_WINDOWS:
        for combo in combos:
            all_configs.append((combo, window))

    print(f"  {len(all_configs)} configs, ALL parallel ({N_JOBS_PARALLEL} workers, step={DIAG_STEP})...")

    # Pre-convert to numpy
    n = len(df_features)
    features_np = df_features[feature_cols].values.astype(np.float64)
    labels_np = df_features['label'].values.astype(np.int32)

    # Single parallel phase — all configs on CPU
    all_results = Parallel(n_jobs=N_JOBS_PARALLEL, verbose=5)(
        delayed(_eval_one_config)(
            features_np, labels_np, combo, window, n, DIAG_STEP, DIAG_MODELS
        )
        for combo, window in all_configs
    )

    # Find best
    best_acc = 0
    best_config = None

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
    """Run diagnostic across all assets, save hourly_best_models.csv."""
    print("\n" + "=" * 60)
    print("  RUNNING HOURLY DIAGNOSTIC")
    print("=" * 60)

    best_models = []

    for asset_name in assets_list:
        print(f"\n--- {asset_name} ---")

        df_raw = load_data(asset_name)
        if df_raw is None:
            continue

        df_features, feature_cols = build_hourly_features(df_raw)

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
        df_best = pd.DataFrame(best_models)
        df_best.to_csv('hourly_best_models.csv', index=False)
        print(f"\n{'='*60}")
        print("  HOURLY DIAGNOSTIC RESULTS (saved to hourly_best_models.csv)")
        print(f"{'='*60}")
        for row in best_models:
            print(f"  {row['coin']:6s} | window={row['best_window']:4d}h | "
                  f"{row['best_combo']:20s} | {row['accuracy']:.1f}%")
    else:
        print("\nNo diagnostic results.")

    return best_models


# ============================================================
# CHART DATA EXPORT
# ============================================================
def export_chart_data(all_signals, output_file='hourly_chart_data.json'):
    """Export all hourly signal data as JSON for chart."""
    chart_data = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'type': 'hourly',
        'prediction_horizon': f'{PREDICTION_HORIZON}h',
        'assets': {}
    }

    for asset_name, signals in all_signals.items():
        chart_data['assets'][asset_name] = signals

    with open(output_file, 'w') as f:
        json.dump(chart_data, f, indent=2)

    print(f"\nHourly chart data saved to {output_file}")
    print(f"  Assets: {', '.join(all_signals.keys())}")
    total_signals = sum(len(s) for s in all_signals.values())
    print(f"  Total signals: {total_signals}")

    return output_file


# ============================================================
# MODE A: Full Review
# ============================================================
def run_mode_a(assets_list):
    """Full review: update -> diagnostic -> best models -> signals -> chart."""
    print("\n" + "=" * 60)
    print("  MODE A: FULL HOURLY REVIEW")
    print("=" * 60)

    # Step 1: Update data
    update_all_data(assets_list)

    # Step 2: Run diagnostic
    best_models = run_full_diagnostic(assets_list)

    if not best_models:
        print("No diagnostic results. Aborting.")
        return

    # Step 3: Generate signals using best models
    print("\n" + "=" * 60)
    print("  GENERATING HOURLY SIGNALS (best models)")
    print("=" * 60)

    all_signals = {}
    for config in best_models:
        asset_name = config['coin']
        model_names = config['models'].split('+')
        window = config['best_window']

        signals = generate_signals(asset_name, model_names, window, REPLAY_HOURS)
        signals = simulate_portfolio(signals)
        all_signals[asset_name] = signals

        if signals:
            latest = signals[-1]
            print(f"\n  >> {asset_name} LATEST: {latest['signal']} ({latest['confidence']:.0f}%) "
                  f"| price={latest['close']:,.2f}")

    # Step 4: Export
    export_chart_data(all_signals)

    print("\n" + "=" * 60)
    print("  MODE A (HOURLY) COMPLETE")
    print("=" * 60)


# ============================================================
# MODE B: Quick Run
# ============================================================
def run_mode_b(assets_list):
    """Quick run: update -> read hourly_best_models.csv -> signals -> chart."""
    print("\n" + "=" * 60)
    print("  MODE B: QUICK HOURLY RUN (using hourly_best_models.csv)")
    print("=" * 60)

    if not os.path.exists('hourly_best_models.csv'):
        print("\nERROR: hourly_best_models.csv not found!")
        print("Please run Mode A first to generate best model configurations.")
        return

    df_best = pd.read_csv('hourly_best_models.csv')
    print("\nLoaded hourly_best_models.csv:")
    for _, row in df_best.iterrows():
        print(f"  {row['coin']:6s} | window={row['best_window']:4d}h | {row['best_combo']}")

    available_in_csv = set(df_best['coin'].values)
    assets_to_run = [a for a in assets_list if a in available_in_csv]
    missing = [a for a in assets_list if a not in available_in_csv]

    if missing:
        print(f"\nWARNING: No hourly best model for: {', '.join(missing)}")

    if not assets_to_run:
        print("No assets to process.")
        return

    # Step 1: Update
    update_all_data(assets_to_run)

    # Step 2: Signals
    print("\n" + "=" * 60)
    print("  GENERATING HOURLY SIGNALS")
    print("=" * 60)

    all_signals = {}
    for asset_name in assets_to_run:
        row = df_best[df_best['coin'] == asset_name].iloc[0]
        model_names = row['models'].split('+')
        window = int(row['best_window'])

        signals = generate_signals(asset_name, model_names, window, REPLAY_HOURS)
        signals = simulate_portfolio(signals)
        all_signals[asset_name] = signals

        if signals:
            latest = signals[-1]
            print(f"\n  >> {asset_name} LATEST: {latest['signal']} ({latest['confidence']:.0f}%) "
                  f"| price={latest['close']:,.2f}")

    # Step 3: Export
    export_chart_data(all_signals)

    print("\n" + "=" * 60)
    print("  MODE B (HOURLY) COMPLETE")
    print("=" * 60)


# ============================================================
# MAIN MENU
# ============================================================
def main():
    print("=" * 60)
    print("  HOURLY INDEX ML TRADING SYSTEM")
    print("  Indices: SMI, DAX, CAC40")
    print(f"  Prediction: {PREDICTION_HORIZON}h ahead")
    print("=" * 60)

    # Mode selection
    print("\nChoose mode:")
    print("  A. Full review (update + diagnostic + signals + chart)")
    print("  B. Quick run (update + use hourly_best_models.csv + chart)")
    mode = input("\nEnter A or B: ").strip().upper()

    if mode not in ('A', 'B'):
        print("Invalid choice. Defaulting to B.")
        mode = 'B'

    # Asset selection
    print("\nWhich indices?")
    print("  1. All (SMI, DAX, CAC40)")
    print("  2. Choose specific")
    choice = input("Enter choice (1-2): ").strip()

    if choice == '2':
        print(f"Available: {', '.join(ASSETS.keys())}")
        selected = input("Enter comma-separated names: ").strip().upper()
        assets_list = [a.strip() for a in selected.split(',') if a.strip() in ASSETS]
    else:
        assets_list = list(ASSETS.keys())

    print(f"\nAssets: {', '.join(assets_list)}")
    print(f"Mode: {'A (Full Review)' if mode == 'A' else 'B (Quick Run)'}")

    if mode == 'A':
        run_mode_a(assets_list)
    else:
        run_mode_b(assets_list)

    print("\nDone! Hourly chart data saved to hourly_chart_data.json")


if __name__ == '__main__':
    main()
