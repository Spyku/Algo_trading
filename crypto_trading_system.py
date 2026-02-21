"""
Crypto & Index ML Trading System
==================================
Unified system for crypto (Binance) and stock indices (Yahoo Finance).
Features:
  - Mode A: Update data -> Run diagnostic -> Best models -> Signals -> Chart data
  - Mode B: Update data -> Load best_models.csv -> Signals -> Chart data
Assets: BTC, ETH, SOL, XRP, DOGE, SMI, DAX, CAC40
Models: RF, GB, LR, LGBM (no XGB)
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

# Windows UTF-8 fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================================
# ASSET CONFIGURATION
# ============================================================
ASSETS = {
    'BTC':   {'source': 'binance', 'ticker': 'BTC/USDT',  'file': 'btc_hourly_data.csv',  'start': '2017-08-01T00:00:00Z'},
    'ETH':   {'source': 'binance', 'ticker': 'ETH/USDT',  'file': 'eth_hourly_data.csv',  'start': '2017-08-01T00:00:00Z'},
    'SOL':   {'source': 'binance', 'ticker': 'SOL/USDT',  'file': 'sol_hourly_data.csv',  'start': '2020-08-01T00:00:00Z'},
    'XRP':   {'source': 'binance', 'ticker': 'XRP/USDT',  'file': 'xrp_hourly_data.csv',  'start': '2018-05-01T00:00:00Z'},
    'DOGE':  {'source': 'binance', 'ticker': 'DOGE/USDT', 'file': 'doge_hourly_data.csv', 'start': '2019-07-01T00:00:00Z'},
    'SMI':   {'source': 'yfinance', 'ticker': '^SSMI',    'file': 'smi_hourly_data.csv',  'start': None},
    'DAX':   {'source': 'yfinance', 'ticker': '^GDAXI',   'file': 'dax_hourly_data.csv',  'start': None},
    'CAC40': {'source': 'yfinance', 'ticker': '^FCHI',     'file': 'cac40_hourly_data.csv','start': None},
}

PREDICTION_HORIZON = 3   # 3-day prediction
DEFAULT_WINDOW = 200     # Default training window for signal generation
REPLAY_DAYS = 35         # Days of history for chart
STEP = 3                 # Diagnostic step size

# ============================================================
# DATA DOWNLOAD: BINANCE (ccxt)
# ============================================================
def download_binance(asset_name, update_only=True):
    """Download or update hourly data from Binance."""
    import ccxt
    config = ASSETS[asset_name]
    filepath = config['file']
    exchange = ccxt.binance()

    symbol = config['ticker']
    timeframe = '1h'
    limit = 1000

    # If file exists and update_only, just fetch new data
    if os.path.exists(filepath) and update_only:
        df_existing = pd.read_csv(filepath)
        df_existing['datetime'] = pd.to_datetime(df_existing['datetime'])
        last_ts = int(df_existing['timestamp'].max()) + 1
        print(f"  Updating {asset_name} from {df_existing['datetime'].max()}...")
    else:
        df_existing = None
        last_ts = exchange.parse8601(config['start'])
        print(f"  Downloading full history for {asset_name}...")

    end_ts = exchange.milliseconds()
    all_candles = []
    current_ts = last_ts
    batch_count = 0

    while current_ts < end_ts:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=limit)
            if not candles:
                break
            all_candles.extend(candles)
            batch_count += 1
            current_ts = candles[-1][0] + 1

            if batch_count % 10 == 0:
                current_date = exchange.iso8601(candles[-1][0])
                print(f"    Batch {batch_count}: up to {current_date}")

            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"    Error: {e}. Retrying in 5s...")
            time.sleep(5)
            continue

    if not all_candles:
        print(f"  No new data for {asset_name}.")
        return

    # Build DataFrame
    df_new = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_new['datetime'] = pd.to_datetime(df_new['timestamp'], unit='ms')
    df_new = df_new[['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Merge with existing
    if df_existing is not None:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined = df_combined.drop_duplicates(subset=['timestamp'])
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

    df_combined.to_csv(filepath, index=False)
    print(f"  {asset_name}: {len(df_combined):,} total candles saved to {filepath}")


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
        elif cl == 'open': rename_map[col] = 'open'
        elif cl == 'high': rename_map[col] = 'high'
        elif cl == 'low': rename_map[col] = 'low'
        elif cl == 'close': rename_map[col] = 'close'
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
    """Download or update hourly data from Yahoo Finance (stock indices)."""
    import yfinance as yf
    config = ASSETS[asset_name]
    filepath = config['file']
    ticker = config['ticker']

    end_date = datetime.now()

    if os.path.exists(filepath) and update_only:
        df_existing = pd.read_csv(filepath)
        df_existing['datetime'] = pd.to_datetime(df_existing['datetime'])
        start_date = df_existing['datetime'].max() - timedelta(days=1)
        print(f"  Updating {asset_name} from {start_date.date()}...")
    else:
        df_existing = None
        start_date = end_date - timedelta(days=729)
        print(f"  Downloading {asset_name} hourly data (last 2 years)...")

    all_chunks = []
    chunk_start = start_date
    chunk_num = 0

    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=59), end_date)
        try:
            data = yf.download(
                ticker,
                start=chunk_start.strftime('%Y-%m-%d'),
                end=chunk_end.strftime('%Y-%m-%d'),
                interval='1h',
                progress=False,
                auto_adjust=True
            )
            if len(data) > 0:
                normalized = _normalize_yf_chunk(data)
                if normalized is not None:
                    all_chunks.append(normalized)
                    chunk_num += 1
                    if chunk_num % 3 == 0:
                        print(f"    Chunk {chunk_num}: up to {chunk_end.date()}")
        except Exception as e:
            print(f"    Error downloading chunk: {e}")

        chunk_start = chunk_end
        time.sleep(0.5)

    if not all_chunks:
        print(f"  No data downloaded for {asset_name}.")
        return

    df_new = pd.concat(all_chunks, ignore_index=True)

    if df_existing is not None:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined = df_combined.drop_duplicates(subset=['timestamp'])
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

    df_combined.to_csv(filepath, index=False)
    print(f"  {asset_name}: {len(df_combined):,} total candles saved to {filepath}")


def download_asset(asset_name, update_only=True):
    """Download data for any asset (auto-selects source)."""
    config = ASSETS[asset_name]
    if config['source'] == 'binance':
        download_binance(asset_name, update_only)
    elif config['source'] == 'yfinance':
        download_yfinance(asset_name, update_only)


def update_all_data(assets_list=None):
    """Download/update data for all (or specified) assets."""
    if assets_list is None:
        assets_list = list(ASSETS.keys())

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
        download_asset(asset_name, update_only=file_exists)

    print("\nData update complete.")


# ============================================================
# DATA LOADING & FEATURE ENGINEERING
# ============================================================
def load_data(asset_name):
    """Load hourly CSV data for an asset."""
    config = ASSETS[asset_name]
    filepath = config['file']
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found.")
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


def build_features(df_daily):
    """
    Build all normalized features from daily OHLCV data.
    Returns (DataFrame with features + labels, feature_cols list).
    """
    df = df_daily.copy()

    # --- Log Returns ---
    for period in [1, 2, 3, 5, 7, 10, 14, 20, 30]:
        df[f'logret_{period}d'] = np.log(df['close'] / df['close'].shift(period))
    for period in [50, 100, 250]:
        df[f'logret_{period}d'] = np.log(df['close'] / df['close'].shift(period))

    # --- Log Return Spreads ---
    df['spread_log10_log2']   = df['logret_10d']  - df['logret_2d']
    df['spread_log20_log2']   = df['logret_20d']  - df['logret_2d']
    df['spread_log30_log2']   = df['logret_30d']  - df['logret_2d']
    df['spread_log30_log10']  = df['logret_30d']  - df['logret_10d']
    df['spread_log7_log3']    = df['logret_7d']   - df['logret_3d']
    df['spread_log250_log10'] = df['logret_250d'] - df['logret_10d']

    # --- Price-to-SMA Ratios ---
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

    # --- Stochastic %K ---
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

    # --- Day of Week (cyclical) ---
    dow = df['date'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # --- Labels ---
    future_return = df['close'].shift(-PREDICTION_HORIZON) / df['close'] - 1
    rolling_median = future_return.rolling(90, min_periods=30).median().shift(PREDICTION_HORIZON)
    df['label'] = (future_return > rolling_median).astype(int)

    # Feature columns
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
# MODELS
# ============================================================
# MODELS (from hardware_config — auto-detects LAPTOP vs DESKTOP)
ALL_MODELS = get_all_models()


# ============================================================
# SIGNAL GENERATION
# ============================================================
def generate_signals(asset_name, model_names, window_size, replay_days=REPLAY_DAYS):
    """
    Generate signals for the last `replay_days` using walk-forward training.
    Returns list of signal dicts suitable for chart data.
    """
    print(f"\n  Generating signals for {asset_name} "
          f"(models={'+'.join(model_names)}, window={window_size}d, "
          f"replay={replay_days}d)...")

    df_raw = load_data(asset_name)
    if df_raw is None:
        return []

    df_daily = hourly_to_daily(df_raw)

    # Feature columns defined here for scaler DataFrame wrapping
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

    df_features, feature_cols = build_features(df_daily)

    n = len(df_features)
    start_idx = max(window_size + 50, n - replay_days)

    signals = []

    for i in range(start_idx, n):
        row = df_features.iloc[i]
        date_str = row['date'].strftime('%Y-%m-%d')

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
        model_details = {}

        for model_name in model_names:
            try:
                model = ALL_MODELS[model_name]()
                model.fit(X_train_s, y_train)
                pred = model.predict(X_test_s)[0]
                proba = model.predict_proba(X_test_s)[0]
                votes.append(pred)
                probas.append(proba[1])  # probability of class 1 (BUY)
                model_details[model_name] = {
                    'prediction': int(pred),
                    'probability': float(proba[1])
                }
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

        # Confidence: average probability of the winning side
        avg_proba = np.mean(probas)
        if signal == 'SELL':
            confidence = (1 - avg_proba) * 100
        else:
            confidence = avg_proba * 100

        # Check actual outcome (if we have future data)
        actual = None
        if i + PREDICTION_HORIZON < n:
            future_close = df_features.iloc[i + PREDICTION_HORIZON]['close']
            actual_return = (future_close / row['close'] - 1) * 100
            actual = 'UP' if actual_return > 0 else 'DOWN'

        signals.append({
            'date': date_str,
            'close': float(row['close']),
            'signal': signal,
            'confidence': round(float(confidence), 1),
            'buy_votes': int(buy_votes),
            'total_votes': int(total_votes),
            'rsi': round(float(row['rsi_14']), 1),
            'bb_position': round(float(row['bb_position']), 3),
            'volume_ratio': round(float(row['volume_ratio']), 2),
            'daily_change': round(float(row['logret_1d'] * 100), 2),
            'spread_log20_log2': round(float(row['spread_log20_log2'] * 100), 2),
            'spread_log250_log10': round(float(row.get('spread_log250_log10', 0) * 100), 2),
            'actual': actual,
        })

        if (i - start_idx) % 10 == 0:
            print(f"    {date_str}: {signal} ({confidence:.0f}%) | "
                  f"price=${row['close']:,.2f}")

    print(f"  Generated {len(signals)} signals for {asset_name}")
    return signals


# ============================================================
# PORTFOLIO SIMULATION ($1000 starting)
# ============================================================
def simulate_portfolio(signals, initial=1000):
    """
    Simulate a $1000 portfolio based on signals.
    BUY = invest, SELL = go to cash, HOLD = keep current position.
    Returns signals list with portfolio_value and hold_value added.
    """
    if not signals:
        return signals

    portfolio = initial
    hold_value = initial
    position = 'cash'  # 'invested' or 'cash'
    entry_price = None
    start_price = signals[0]['close']

    for sig in signals:
        price = sig['close']

        # Buy & Hold benchmark
        hold_value = initial * (price / start_price)

        # ML strategy
        if sig['signal'] == 'BUY' and position == 'cash':
            position = 'invested'
            entry_price = price
        elif sig['signal'] == 'SELL' and position == 'invested':
            # Realize gains/losses
            pnl_ratio = price / entry_price
            portfolio *= pnl_ratio
            position = 'cash'
            entry_price = None

        # Current portfolio value
        if position == 'invested' and entry_price:
            current_value = portfolio * (price / entry_price)
        else:
            current_value = portfolio

        sig['portfolio_value'] = round(current_value, 2)
        sig['hold_value'] = round(hold_value, 2)

    return signals


# ============================================================
# DIAGNOSTIC (CPU parallel, GPU sequential)
# ============================================================
DIAG_WINDOWS = [30, 50, 70, 90, 100]

def _eval_one_config_daily(features_np, labels_np, combo, window, n, step, model_factories):
    """Worker: evaluate one (window, combo) config using numpy arrays."""
    min_start = window + 50
    if n < min_start + 30:
        return None

    correct = 0
    total = 0

    for i in range(min_start, n, step):
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
    """Run diagnostic: ALL configs in parallel on CPU."""
    combos = []
    model_names = list(ALL_MODELS.keys())
    for r in range(1, len(model_names) + 1):
        for combo in combinations(model_names, r):
            combos.append(list(combo))

    all_configs = []
    for window in DIAG_WINDOWS:
        for combo in combos:
            all_configs.append((combo, window))

    print(f"  {len(all_configs)} configs, ALL parallel ({N_JOBS_PARALLEL} workers, step={STEP})...")

    n = len(df_features)
    features_np = df_features[feature_cols].values.astype(np.float64)
    labels_np = df_features['label'].values.astype(np.int32)

    all_results = Parallel(n_jobs=N_JOBS_PARALLEL, verbose=5)(
        delayed(_eval_one_config_daily)(
            features_np, labels_np, combo, window, n, STEP, DIAG_MODELS
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
        print(f"    w={window:3d} | {combo_name:20s} | acc={acc*100:.1f}% (n={n_total})"
              f"{'  <-- BEST' if acc == best_acc else ''}")

    return best_config


def run_full_diagnostic(assets_list):
    """Run diagnostic across all assets, save best_models.csv."""
    print("\n" + "=" * 60)
    print("  RUNNING MODEL DIAGNOSTIC")
    print("=" * 60)

    best_models = []

    for asset_name in assets_list:
        print(f"\n--- {asset_name} ---")

        df_raw = load_data(asset_name)
        if df_raw is None:
            continue

        df_daily = hourly_to_daily(df_raw)
        df_features, feature_cols = build_features(df_daily)

        if len(df_features) < 200:
            print(f"  Not enough data ({len(df_features)} rows). Skipping.")
            continue

        print(f"  {len(df_features):,} daily rows, {len(feature_cols)} features")

        best_config = run_diagnostic_for_asset(asset_name, df_features, feature_cols)
        if best_config:
            best_models.append(best_config)
            print(f"\n  BEST: window={best_config['best_window']}d, "
                  f"combo={best_config['best_combo']}, "
                  f"acc={best_config['accuracy']:.1f}%")

    if best_models:
        df_best = pd.DataFrame(best_models)
        df_best.to_csv('best_models.csv', index=False)
        print(f"\n{'='*60}")
        print("  RESULTS (saved to best_models.csv)")
        print(f"{'='*60}")
        for row in best_models:
            print(f"  {row['coin']:6s} | window={row['best_window']:3d}d | "
                  f"{row['best_combo']:20s} | {row['accuracy']:.1f}%")

    return best_models


# ============================================================
# CHART DATA EXPORT
# ============================================================
def export_chart_data(all_signals, output_file='chart_data.json'):
    """
    Export all signal data as JSON for chart generation.
    Structure: { "generated": "...", "assets": { "BTC": [...], ... } }
    """
    chart_data = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'assets': {}
    }

    for asset_name, signals in all_signals.items():
        chart_data['assets'][asset_name] = signals

    with open(output_file, 'w') as f:
        json.dump(chart_data, f, indent=2)

    print(f"\nChart data saved to {output_file}")
    print(f"  Assets: {', '.join(all_signals.keys())}")
    total_signals = sum(len(s) for s in all_signals.values())
    print(f"  Total signals: {total_signals}")

    return output_file


# ============================================================
# MODE A: Full Review
# ============================================================
def run_mode_a(assets_list):
    """Full review: update data -> diagnostic -> best models -> signals -> chart."""
    print("\n" + "=" * 60)
    print("  MODE A: FULL REVIEW")
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
    print("  GENERATING SIGNALS (best models)")
    print("=" * 60)

    all_signals = {}
    for config in best_models:
        asset_name = config['coin']
        model_names = config['models'].split('+')
        window = config['best_window']

        signals = generate_signals(asset_name, model_names, window, REPLAY_DAYS)
        signals = simulate_portfolio(signals)
        all_signals[asset_name] = signals

        # Print latest signal
        if signals:
            latest = signals[-1]
            print(f"\n  >> {asset_name} LATEST: {latest['signal']} ({latest['confidence']:.0f}%) "
                  f"| price=${latest['close']:,.2f}")

    # Step 4: Export chart data
    export_chart_data(all_signals)

    print("\n" + "=" * 60)
    print("  MODE A COMPLETE")
    print("=" * 60)


# ============================================================
# MODE B: Quick Run (best models from CSV)
# ============================================================
def run_mode_b(assets_list):
    """Quick run: update data -> read best_models.csv -> signals -> chart."""
    print("\n" + "=" * 60)
    print("  MODE B: QUICK RUN (using best_models.csv)")
    print("=" * 60)

    # Check for best_models.csv
    if not os.path.exists('best_models.csv'):
        print("\nERROR: best_models.csv not found!")
        print("Please run Mode A first to generate best model configurations.")
        return

    df_best = pd.read_csv('best_models.csv')
    print("\nLoaded best_models.csv:")
    for _, row in df_best.iterrows():
        print(f"  {row['coin']:6s} | window={row['best_window']:3d}d | {row['best_combo']}")

    # Filter to requested assets
    available_in_csv = set(df_best['coin'].values)
    assets_to_run = [a for a in assets_list if a in available_in_csv]
    missing = [a for a in assets_list if a not in available_in_csv]

    if missing:
        print(f"\nWARNING: No best model config for: {', '.join(missing)}")
        print("Run Mode A to include these assets in the diagnostic.")

    if not assets_to_run:
        print("No assets to process.")
        return

    # Step 1: Update data
    update_all_data(assets_to_run)

    # Step 2: Generate signals
    print("\n" + "=" * 60)
    print("  GENERATING SIGNALS")
    print("=" * 60)

    all_signals = {}
    for asset_name in assets_to_run:
        row = df_best[df_best['coin'] == asset_name].iloc[0]
        model_names = row['models'].split('+')
        window = int(row['best_window'])

        signals = generate_signals(asset_name, model_names, window, REPLAY_DAYS)
        signals = simulate_portfolio(signals)
        all_signals[asset_name] = signals

        # Print latest signal
        if signals:
            latest = signals[-1]
            print(f"\n  >> {asset_name} LATEST: {latest['signal']} ({latest['confidence']:.0f}%) "
                  f"| price=${latest['close']:,.2f}")

    # Step 3: Export chart data
    export_chart_data(all_signals)

    print("\n" + "=" * 60)
    print("  MODE B COMPLETE")
    print("=" * 60)


# ============================================================
# MAIN MENU
# ============================================================
def main():
    print("=" * 60)
    print("  ML TRADING SYSTEM")
    print("  Crypto: BTC, ETH, SOL, XRP, DOGE")
    print("  Indices: SMI, DAX, CAC40")
    print("=" * 60)

    # Mode selection
    print("\nChoose mode:")
    print("  A. Full review (update data + diagnostic + best models + chart)")
    print("  B. Quick run (update data + use best_models.csv + chart)")
    mode = input("\nEnter A or B: ").strip().upper()

    if mode not in ('A', 'B'):
        print("Invalid choice. Defaulting to B.")
        mode = 'B'

    # Asset selection
    print("\nWhich assets?")
    print("  1. All (crypto + indices)")
    print("  2. Crypto only (BTC, ETH, SOL, XRP, DOGE)")
    print("  3. Indices only (SMI, DAX, CAC40)")
    print("  4. Choose specific")
    choice = input("Enter choice (1-4): ").strip()

    if choice == '2':
        assets_list = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']
    elif choice == '3':
        assets_list = ['SMI', 'DAX', 'CAC40']
    elif choice == '4':
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

    print("\nDone! Chart data saved to chart_data.json")


if __name__ == '__main__':
    main()
