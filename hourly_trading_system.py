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
from features_v2 import build_features_v2_hourly
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
    'SMI':   {'ticker': '^SSMI',  'file': 'data/indices/smi_hourly_data.csv'},
    'DAX':   {'ticker': '^GDAXI', 'file': 'data/indices/dax_hourly_data.csv'},
    'CAC40': {'ticker': '^FCHI',  'file': 'data/indices/cac40_hourly_data.csv'},
}

PREDICTION_HORIZON = 4    # Predict 4 hours ahead (~half a trading day)

# ============================================================
# OPTIMAL V2 FEATURES (from feature_analysis_v2.py)
# 15 features -> 76.1% accuracy (best subset of 101)
# ============================================================
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
DEFAULT_WINDOW = 400      # Default training window (hours)
REPLAY_HOURS = 200        # Hours of history for chart (~25 trading days)
DIAG_STEP = 24             # Diagnostic: evaluate every 24 hours (~1 per trading day)
DIAG_WINDOWS = [100, 200, 300, 500, 700]  # Training windows to test

# Strategy V2: 5-tier signal thresholds
CONFIDENCE_THRESHOLD = 70  # % confidence for STRONG signals


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
def generate_signals(asset_name, model_names, window_size, replay_hours=REPLAY_HOURS, strategy='v1'):
    """
    Generate hourly signals using walk-forward training.
    Returns list of signal dicts.
    """
    print(f"\n  Generating hourly signals for {asset_name} "
          f"(models={'+'.join(model_names)}, window={window_size}h, "
          f"replay={replay_hours}h, strategy={strategy})...")

    df_raw = load_data(asset_name)
    if df_raw is None:
        return []

    # Build V2 features (all 101), then use only optimal 15
    df_v2_all, _all_v2_cols = build_features_v2_hourly(df_raw, original_builder=build_hourly_features)
    feature_cols = [c for c in OPTIMAL_V2_FEATURES if c in df_v2_all.columns]
    if len(feature_cols) < len(OPTIMAL_V2_FEATURES):
        missing = [c for c in OPTIMAL_V2_FEATURES if c not in df_v2_all.columns]
        print(f"    WARNING: Missing V2 features: {missing}")
    df_features = df_v2_all.dropna(subset=feature_cols + ['label']).reset_index(drop=True)

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
        avg_proba = np.mean(probas)

        if strategy in ('v2', 'v3'):
            signal, confidence = classify_signal_v2(buy_votes, total_votes, avg_proba)
        else:
            # V1: original 3-tier
            if buy_ratio > 0.5:
                signal = 'BUY'
            elif buy_ratio == 0:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            # Confidence
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
            'rsi': round(float(row.get('rsi_14h', 0)), 1),
            'bb_position': round(float(row.get('bb_position_20h', 0)), 3),
            'hourly_change': round(float(row['logret_1h'] * 100), 3),
            'intraday_range': round(float(row['intraday_range'] * 100), 3),
            'spread_24h_4h': round(float(row.get('spread_24h_4h', 0) * 100), 2),
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
# STRATEGY V2: 5-TIER GRADUATED SIGNALS
# ============================================================
def classify_signal_v2(buy_votes, total_votes, avg_proba):
    """
    5-tier signal classification based on vote unanimity + confidence.
    Returns (signal, confidence).

    STRONG BUY:  unanimous BUY  + confidence >= threshold
    BUY:         majority BUY
    HOLD:        split / no clear majority
    SELL:        unanimous SELL (no BUY votes)
    STRONG SELL: unanimous SELL + confidence >= threshold
    """
    buy_ratio = buy_votes / total_votes

    if buy_ratio == 1.0:  # unanimous BUY
        confidence = avg_proba * 100
        if confidence >= CONFIDENCE_THRESHOLD:
            return 'STRONG BUY', confidence
        return 'BUY', confidence
    elif buy_ratio > 0.5:  # majority BUY (not unanimous)
        confidence = avg_proba * 100
        return 'BUY', confidence
    elif buy_ratio == 0:  # unanimous SELL
        confidence = (1 - avg_proba) * 100
        if confidence >= CONFIDENCE_THRESHOLD:
            return 'STRONG SELL', confidence
        return 'SELL', confidence
    else:  # split (0 < buy_ratio <= 0.5)
        confidence = max(avg_proba, 1 - avg_proba) * 100
        return 'HOLD', confidence


def simulate_portfolio_v2(signals, initial=1000):
    """
    Graduated position sizing portfolio simulation.

    Target allocation per signal:
      STRONG BUY  -> 100% invested
      BUY         ->  50% invested
      HOLD        ->  no change
      SELL        ->   0% invested (cash)
      STRONG SELL ->   0% invested (cash)

    Tracks: allocation (0.0, 0.5, 1.0), entry prices per tranche.
    """
    if not signals:
        return signals

    portfolio_cash = initial     # cash portion
    portfolio_invested = 0.0     # market value of invested portion
    allocation = 0.0             # 0.0, 0.5, or 1.0
    entry_price = None

    hold_value = initial
    start_price = signals[0]['close']

    for sig in signals:
        price = sig['close']
        hold_value = initial * (price / start_price)

        # Update invested portion to current market value
        if allocation > 0 and entry_price:
            portfolio_invested = portfolio_invested * (price / entry_price)
            entry_price = price

        signal = sig['signal']

        # Determine target allocation
        if signal == 'STRONG BUY':
            target = 1.0
        elif signal == 'BUY':
            target = 0.5
        elif signal == 'HOLD':
            target = allocation  # no change
        else:  # SELL or STRONG SELL
            target = 0.0

        # Rebalance if target changed
        if target != allocation:
            total_value = portfolio_cash + portfolio_invested

            portfolio_invested = total_value * target
            portfolio_cash = total_value * (1 - target)
            allocation = target
            entry_price = price if target > 0 else None

        current_value = portfolio_cash + portfolio_invested
        sig['portfolio_value'] = round(current_value, 2)
        sig['hold_value'] = round(hold_value, 2)
        sig['allocation'] = allocation

    return signals


# ============================================================
# STRATEGY V3: COMPARISON (V1 + V2 on same 5-tier signals)
# ============================================================
def simulate_portfolio_v3(signals, initial=1000):
    """
    Runs BOTH V1 and V2 portfolio simulations on the same V2 signals.
    V1 mapping: STRONG BUY/BUY -> invested, HOLD -> stay, SELL/STRONG SELL -> cash
    V2: graduated as before (STRONG=100%, BUY=50%, HOLD=stay, SELL/STRONG SELL=0%)
    Stores portfolio_v1, portfolio_v2, hold_value in each signal.
    """
    if not signals:
        return signals

    start_price = signals[0]['close']

    # V1 state
    v1_portfolio = initial
    v1_position = 'cash'
    v1_entry_price = None

    # V2 state
    v2_cash = initial
    v2_invested = 0.0
    v2_allocation = 0.0
    v2_entry_price = None

    for sig in signals:
        price = sig['close']
        hold_value = initial * (price / start_price)
        signal = sig['signal']

        # --- V1: all-in / all-out ---
        v1_signal = 'BUY' if signal in ('STRONG BUY', 'BUY') else \
                    'SELL' if signal in ('STRONG SELL', 'SELL') else 'HOLD'

        if v1_signal == 'BUY' and v1_position == 'cash':
            v1_position = 'invested'
            v1_entry_price = price
        elif v1_signal == 'SELL' and v1_position == 'invested':
            v1_portfolio *= price / v1_entry_price
            v1_position = 'cash'
            v1_entry_price = None

        if v1_position == 'invested' and v1_entry_price:
            v1_current = v1_portfolio * (price / v1_entry_price)
        else:
            v1_current = v1_portfolio

        # --- V2: graduated ---
        if v2_allocation > 0 and v2_entry_price:
            v2_invested = v2_invested * (price / v2_entry_price)
            v2_entry_price = price

        if signal == 'STRONG BUY':
            target = 1.0
        elif signal == 'BUY':
            target = 0.5
        elif signal == 'HOLD':
            target = v2_allocation
        else:
            target = 0.0

        if target != v2_allocation:
            total = v2_cash + v2_invested
            v2_invested = total * target
            v2_cash = total * (1 - target)
            v2_allocation = target
            v2_entry_price = price if target > 0 else None

        v2_current = v2_cash + v2_invested

        # Store all three
        sig['portfolio_v1'] = round(v1_current, 2)
        sig['portfolio_v2'] = round(v2_current, 2)
        sig['portfolio_value'] = round(v2_current, 2)
        sig['hold_value'] = round(hold_value, 2)
        sig['allocation'] = v2_allocation

    return signals


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

        # Build V2 features, use optimal 15
        df_v2_all, _all_v2_cols = build_features_v2_hourly(df_raw, original_builder=build_hourly_features)
        feature_cols = [c for c in OPTIMAL_V2_FEATURES if c in df_v2_all.columns]
        df_features = df_v2_all.dropna(subset=feature_cols + ['label']).reset_index(drop=True)

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
        df_best.to_csv('data/hourly_best_models.csv', index=False)
        print(f"\n{'='*60}")
        print("  HOURLY DIAGNOSTIC RESULTS (saved to data/hourly_best_models.csv)")
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
def export_chart_data(all_signals, output_file='output/charts/hourly_chart_data.json'):
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
# HTML DASHBOARD EXPORT
# ============================================================
def export_html_dashboard(all_signals, strategy='v1'):
    """Generate a self-contained HTML dashboard from signal data."""
    suffix = {'v1': '', 'v2': '_v2', 'v3': '_v3'}[strategy]
    html_file = f'output/dashboards/hourly_dashboard{suffix}.html'
    strat_label = {'v1': 'V1 (3-tier)', 'v2': 'V2 (5-tier graduated)', 'v3': 'V3 (V1 vs V2 comparison)'}[strategy]

    chart_data = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'type': 'hourly',
        'strategy': strategy,
        'prediction_horizon': f'{PREDICTION_HORIZON}h',
        'assets': {}
    }
    for asset_name, signals in all_signals.items():
        chart_data['assets'][asset_name] = signals

    generated_ts = chart_data['generated']
    total_sigs = sum(len(s) for s in all_signals.values())
    per_asset = total_sigs // max(len(all_signals), 1)

    data_json = json.dumps(chart_data)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hourly Trading Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg-primary: #0a0e17;
    --bg-card: #111827;
    --bg-card-hover: #1a2332;
    --border: #1e293b;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --green: #22c55e;
    --green-bg: rgba(34,197,94,0.08);
    --red: #ef4444;
    --red-bg: rgba(239,68,68,0.08);
    --blue: #3b82f6;
    --amber: #f59e0b;
    --cyan: #06b6d4;
    --purple: #a78bfa;
    --accent: #3b82f6;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: var(--bg-primary); color: var(--text-primary);
    font-family: 'DM Sans', sans-serif; min-height: 100vh; overflow-x: hidden;
  }}
  .noise-overlay {{
    position:fixed; inset:0; z-index:0; pointer-events:none; opacity:0.03;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  }}
  .container {{ position:relative; z-index:1; max-width:1400px; margin:0 auto; padding:24px 20px; }}
  .header {{
    display:flex; justify-content:space-between; align-items:flex-start;
    margin-bottom:28px; padding-bottom:20px; border-bottom:1px solid var(--border);
  }}
  .header h1 {{
    font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:600;
    letter-spacing:-0.3px; color:var(--text-primary);
  }}
  .header h1 span {{ color:var(--accent); }}
  .header-meta {{
    font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--text-muted);
    text-align:right; line-height:1.7;
  }}
  .header-meta .live-dot {{
    display:inline-block; width:6px; height:6px; background:var(--green);
    border-radius:50%; margin-right:4px; animation:pulse 2s infinite;
  }}
  @keyframes pulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.3;}} }}
  .summary-row {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:24px; }}
  .summary-card {{
    background:var(--bg-card); border:1px solid var(--border); border-radius:8px;
    padding:16px 18px; transition:background 0.2s; cursor:pointer;
    position:relative; overflow:hidden;
  }}
  .summary-card:hover {{ background:var(--bg-card-hover); }}
  .summary-card.active {{
    border-color:var(--accent);
    box-shadow:0 0 0 1px var(--accent), 0 0 20px rgba(59,130,246,0.08);
  }}
  .summary-card .asset-name {{
    font-family:'JetBrains Mono',monospace; font-size:13px; font-weight:600;
    color:var(--text-secondary); margin-bottom:10px; display:flex; align-items:center; gap:8px;
  }}
  .summary-card .asset-name .tag {{
    font-size:9px; font-weight:500; padding:2px 6px; border-radius:3px; letter-spacing:0.5px;
  }}
  .tag-buy {{ background:var(--green-bg); color:var(--green); }}
  .tag-sell {{ background:var(--red-bg); color:var(--red); }}
  .tag-hold {{ background:rgba(245,158,11,0.08); color:var(--amber); }}
  .summary-card .stat-row {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px; }}
  .stat-label {{ font-size:11px; color:var(--text-muted); }}
  .stat-value {{ font-family:'JetBrains Mono',monospace; font-size:13px; font-weight:500; }}
  .stat-value.positive {{ color:var(--green); }}
  .stat-value.negative {{ color:var(--red); }}
  .mini-bar {{ height:3px; border-radius:2px; margin-top:10px; background:var(--border); overflow:hidden; }}
  .mini-bar-fill {{ height:100%; border-radius:2px; transition:width 0.6s ease; }}
  .tab-bar {{
    display:flex; gap:2px; margin-bottom:20px; background:var(--bg-card);
    border-radius:6px; padding:3px; border:1px solid var(--border); width:fit-content;
  }}
  .tab-btn {{
    font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:500; padding:6px 14px;
    background:none; border:none; color:var(--text-muted); cursor:pointer; border-radius:4px;
    transition:all 0.2s;
  }}
  .tab-btn:hover {{ color:var(--text-secondary); }}
  .tab-btn.active {{ background:var(--accent); color:#fff; }}
  .chart-panel {{
    background:var(--bg-card); border:1px solid var(--border); border-radius:8px;
    padding:20px; margin-bottom:16px;
  }}
  .chart-panel h3 {{
    font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:500;
    color:var(--text-secondary); margin-bottom:14px; letter-spacing:0.5px;
  }}
  .chart-wrapper {{ position:relative; width:100%; }}
  .chart-wrapper.main {{ height:340px; }}
  .chart-wrapper.secondary {{ height:200px; }}
  .signal-table-wrap {{
    background:var(--bg-card); border:1px solid var(--border); border-radius:8px;
    padding:16px 18px; max-height:400px; overflow-y:auto;
  }}
  .signal-table-wrap h3 {{
    font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:500;
    color:var(--text-secondary); margin-bottom:12px;
  }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{
    font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:500;
    color:var(--text-muted); text-align:left; padding:6px 8px;
    border-bottom:1px solid var(--border); position:sticky; top:0;
    background:var(--bg-card); text-transform:uppercase; letter-spacing:0.5px;
  }}
  td {{
    font-family:'JetBrains Mono',monospace; font-size:11px; padding:5px 8px;
    border-bottom:1px solid rgba(30,41,59,0.5); color:var(--text-secondary);
  }}
  tr:hover td {{ background:rgba(59,130,246,0.03); }}
  .sig-buy {{ color:var(--green); font-weight:600; }}
  .sig-sell {{ color:var(--red); font-weight:600; }}
  .sig-hold {{ color:var(--amber); font-weight:600; }}
  .sig-strong-buy {{ color:var(--green); font-weight:700; text-shadow:0 0 6px rgba(34,197,94,0.3); }}
  .sig-strong-sell {{ color:var(--red); font-weight:700; text-shadow:0 0 6px rgba(239,68,68,0.3); }}
  .actual-up {{ color:var(--green); }}
  .actual-down {{ color:var(--red); }}
  .correct {{ background:rgba(34,197,94,0.06); }}
  .wrong {{ background:rgba(239,68,68,0.06); }}
  .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }}
  ::-webkit-scrollbar {{ width:5px; }}
  ::-webkit-scrollbar-track {{ background:var(--bg-card); }}
  ::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:3px; }}
  @media (max-width:900px) {{
    .summary-row {{ grid-template-columns:1fr; }}
    .two-col {{ grid-template-columns:1fr; }}
  }}
</style>
</head>
<body>
<div class="noise-overlay"></div>
<div class="container">
  <div class="header">
    <div>
      <h1><span>//</span> HOURLY TRADING DASHBOARD</h1>
      <div style="font-size:12px;color:var(--text-muted);margin-top:4px;">
        {PREDICTION_HORIZON}h prediction &middot; Walk-forward ML ensemble &middot; {strat_label}
      </div>
    </div>
    <div class="header-meta">
      <div><span class="live-dot"></span>Generated {generated_ts}</div>
      <div>{per_asset} signals per asset</div>
    </div>
  </div>
  <div class="summary-row" id="summaryRow"></div>
  <div class="tab-bar" id="tabBar">
    <button class="tab-btn active" data-tab="portfolio">PORTFOLIO</button>
    <button class="tab-btn" data-tab="price">PRICE + SIGNALS</button>
    <button class="tab-btn" data-tab="indicators">INDICATORS</button>
    <button class="tab-btn" data-tab="table">SIGNAL LOG</button>
  </div>
  <div id="tabContent"></div>
</div>
<script>
const CHART_DATA = {data_json};
const STRATEGY = "{strategy}";

const ASSET_COLORS = {{
  SMI:   {{ main:'#3b82f6', dim:'#2563eb' }},
  DAX:   {{ main:'#a78bfa', dim:'#7c3aed' }},
  CAC40: {{ main:'#06b6d4', dim:'#0891b2' }},
}};

let activeAsset = Object.keys(CHART_DATA.assets)[0];
let activeTab = 'portfolio';
let charts = {{}};

function getAssetSummary(name, signals) {{
  const last = signals[signals.length - 1];
  const pnl = ((last.portfolio_value - 1000) / 10).toFixed(1);
  const holdPnl = ((last.hold_value - 1000) / 10).toFixed(1);
  const v1Pnl = last.portfolio_v1 ? (((last.portfolio_v1 - 1000) / 10).toFixed(1)) : pnl;
  const v2Pnl = last.portfolio_v2 ? (((last.portfolio_v2 - 1000) / 10).toFixed(1)) : pnl;
  const withActual = signals.filter(s => s.actual === 'UP' || s.actual === 'DOWN');
  const correct = withActual.filter(s =>
    ((s.signal === 'BUY' || s.signal === 'STRONG BUY') && s.actual === 'UP') ||
    ((s.signal === 'SELL' || s.signal === 'STRONG SELL') && s.actual === 'DOWN')
  );
  const acc = withActual.length > 0 ? ((correct.length / withActual.length) * 100).toFixed(1) : 'N/A';
  return {{ pnl, holdPnl, v1Pnl, v2Pnl, acc, lastSignal: last.signal, lastClose: last.close, portfolio: last.portfolio_value, allocation: last.allocation }};
}}

function sigTagClass(sig) {{
  if (sig.includes('BUY')) return 'tag-buy';
  if (sig.includes('SELL')) return 'tag-sell';
  return 'tag-hold';
}}

function sigCssClass(sig) {{
  if (sig === 'STRONG BUY') return 'sig-strong-buy';
  if (sig === 'STRONG SELL') return 'sig-strong-sell';
  if (sig === 'BUY') return 'sig-buy';
  if (sig === 'SELL') return 'sig-sell';
  return 'sig-hold';
}}

function renderSummaryCards() {{
  const row = document.getElementById('summaryRow');
  row.innerHTML = '';
  for (const [name, signals] of Object.entries(CHART_DATA.assets)) {{
    const s = getAssetSummary(name, signals);
    const isActive = name === activeAsset;

    const card = document.createElement('div');
    card.className = 'summary-card' + (isActive ? ' active' : '');
    card.onclick = () => {{ activeAsset = name; renderAll(); }};

    let body = '<div class="asset-name">' + name + ' <span class="tag ' + sigTagClass(s.lastSignal) + '">' + s.lastSignal + '</span></div>';

    if (STRATEGY === 'v3') {{
      const v1Class = parseFloat(s.v1Pnl) >= 0 ? 'positive' : 'negative';
      const v2Class = parseFloat(s.v2Pnl) >= 0 ? 'positive' : 'negative';
      const v1Alpha = parseFloat(s.v1Pnl) - parseFloat(s.holdPnl);
      const v2Alpha = parseFloat(s.v2Pnl) - parseFloat(s.holdPnl);
      body +=
        '<div class="stat-row"><span class="stat-label">V1 All-in/out</span><span class="stat-value ' + v1Class + '">' + (parseFloat(s.v1Pnl)>=0?'+':'') + s.v1Pnl + '% <small style="color:' + (v1Alpha>=0?'var(--green)':'var(--red)') + '">(a:' + (v1Alpha>=0?'+':'') + v1Alpha.toFixed(1) + ')</small></span></div>' +
        '<div class="stat-row"><span class="stat-label">V2 Graduated</span><span class="stat-value ' + v2Class + '">' + (parseFloat(s.v2Pnl)>=0?'+':'') + s.v2Pnl + '% <small style="color:' + (v2Alpha>=0?'var(--green)':'var(--red)') + '">(a:' + (v2Alpha>=0?'+':'') + v2Alpha.toFixed(1) + ')</small></span></div>' +
        '<div class="stat-row"><span class="stat-label">Buy & Hold</span><span class="stat-value" style="color:var(--text-secondary)">' + (parseFloat(s.holdPnl)>=0?'+':'') + s.holdPnl + '%</span></div>';
    }} else {{
      const pnlClass = parseFloat(s.pnl) >= 0 ? 'positive' : 'negative';
      const alpha = parseFloat(s.pnl) - parseFloat(s.holdPnl);
      const alphaClass = alpha >= 0 ? 'positive' : 'negative';
      body +=
        '<div class="stat-row"><span class="stat-label">Portfolio</span><span class="stat-value ' + pnlClass + '">' + (parseFloat(s.pnl)>=0?'+':'') + s.pnl + '%</span></div>' +
        '<div class="stat-row"><span class="stat-label">Buy & Hold</span><span class="stat-value" style="color:var(--text-secondary)">' + (parseFloat(s.holdPnl)>=0?'+':'') + s.holdPnl + '%</span></div>' +
        '<div class="stat-row"><span class="stat-label">Alpha</span><span class="stat-value ' + alphaClass + '">' + (alpha>=0?'+':'') + alpha.toFixed(1) + '%</span></div>';
    }}

    body += '<div class="stat-row"><span class="stat-label">Accuracy</span><span class="stat-value" style="color:var(--cyan)">' + s.acc + '%</span></div>';

    if (STRATEGY !== 'v1' && s.allocation !== undefined) {{
      body += '<div class="stat-row"><span class="stat-label">Allocation</span><span class="stat-value" style="color:var(--purple)">' + (s.allocation * 100).toFixed(0) + '%</span></div>';
    }}

    body += '<div class="mini-bar"><div class="mini-bar-fill" style="width:' + s.acc + '%;background:' + (ASSET_COLORS[name]||{{main:'#3b82f6'}}).main + '"></div></div>';
    card.innerHTML = body;
    row.appendChild(card);
  }}
}}

function renderTabs() {{
  document.querySelectorAll('.tab-btn').forEach(btn => {{
    btn.classList.toggle('active', btn.dataset.tab === activeTab);
    btn.onclick = () => {{ activeTab = btn.dataset.tab; renderAll(); }};
  }});
}}

function destroyCharts() {{
  Object.values(charts).forEach(c => c.destroy());
  charts = {{}};
}}

function renderContent() {{
  destroyCharts();
  const container = document.getElementById('tabContent');
  const signals = CHART_DATA.assets[activeAsset];
  const color = ASSET_COLORS[activeAsset] || {{ main:'#3b82f6' }};

  if (activeTab === 'portfolio') {{
    container.innerHTML =
      '<div class="chart-panel"><h3>' + (STRATEGY==='v3' ? 'V1 vs V2 vs BUY & HOLD' : 'PORTFOLIO VALUE vs BUY & HOLD') + ' -- ' + activeAsset + '</h3><div class="chart-wrapper main"><canvas id="chartPortfolio"></canvas></div></div>' +
      '<div class="two-col">' +
        '<div class="chart-panel"><h3>SIGNAL CONFIDENCE</h3><div class="chart-wrapper secondary"><canvas id="chartConf"></canvas></div></div>' +
        '<div class="chart-panel"><h3>SIGNAL DISTRIBUTION</h3><div class="chart-wrapper secondary"><canvas id="chartDist"></canvas></div></div>' +
      '</div>';
    renderPortfolioChart(signals, color);
    renderConfidenceChart(signals, color);
    renderDistributionChart(signals);
  }} else if (activeTab === 'price') {{
    container.innerHTML =
      '<div class="chart-panel"><h3>PRICE ACTION + SIGNALS -- ' + activeAsset + '</h3><div class="chart-wrapper main"><canvas id="chartPrice"></canvas></div></div>';
    renderPriceChart(signals, color);
  }} else if (activeTab === 'indicators') {{
    container.innerHTML =
      '<div class="chart-panel"><h3>RSI (14h) -- ' + activeAsset + '</h3><div class="chart-wrapper secondary"><canvas id="chartRSI"></canvas></div></div>' +
      '<div class="chart-panel"><h3>BOLLINGER BAND POSITION -- ' + activeAsset + '</h3><div class="chart-wrapper secondary"><canvas id="chartBB"></canvas></div></div>';
    renderRSIChart(signals, color);
    renderBBChart(signals, color);
  }} else if (activeTab === 'table') {{
    renderSignalTable(signals);
  }}
}}

const chartDefaults = {{
  responsive:true, maintainAspectRatio:false,
  plugins: {{
    legend: {{ display:true, labels:{{ color:'#94a3b8', font:{{ family:'JetBrains Mono',size:10 }}, boxWidth:10, padding:15 }} }},
    tooltip: {{
      backgroundColor:'#1e293b', titleColor:'#e2e8f0', bodyColor:'#94a3b8',
      titleFont:{{ family:'JetBrains Mono',size:11 }}, bodyFont:{{ family:'JetBrains Mono',size:10 }},
      borderColor:'#334155', borderWidth:1, padding:10, cornerRadius:4,
    }}
  }},
  scales: {{
    x: {{ ticks:{{ color:'#475569', font:{{ family:'JetBrains Mono',size:9 }}, maxTicksLimit:12, maxRotation:0 }}, grid:{{ color:'rgba(30,41,59,0.5)' }} }},
    y: {{ ticks:{{ color:'#475569', font:{{ family:'JetBrains Mono',size:9 }} }}, grid:{{ color:'rgba(30,41,59,0.5)' }} }}
  }}
}};

function makeLabels(signals) {{ return signals.map(s => s.datetime.substring(5)); }}

function sigColor(sig) {{
  if (sig.includes('BUY')) return '#22c55e';
  if (sig.includes('SELL')) return '#ef4444';
  return '#f59e0b';
}}

function renderPortfolioChart(signals, color) {{
  const ctx = document.getElementById('chartPortfolio').getContext('2d');
  let datasets;
  if (STRATEGY === 'v3') {{
    datasets = [
      {{ label:'V1 All-in/out', data:signals.map(s=>s.portfolio_v1), borderColor:'#3b82f6', backgroundColor:'#3b82f618', fill:false, borderWidth:2, pointRadius:0, tension:0.3 }},
      {{ label:'V2 Graduated', data:signals.map(s=>s.portfolio_v2), borderColor:'#22c55e', backgroundColor:'#22c55e18', fill:false, borderWidth:2, pointRadius:0, tension:0.3 }},
      {{ label:'Buy & Hold', data:signals.map(s=>s.hold_value), borderColor:'#64748b', borderDash:[4,4], borderWidth:1.5, pointRadius:0, fill:false, tension:0.3 }},
    ];
  }} else {{
    datasets = [
      {{ label:'ML Portfolio', data:signals.map(s=>s.portfolio_value), borderColor:color.main, backgroundColor:color.main+'18', fill:true, borderWidth:2, pointRadius:0, tension:0.3 }},
      {{ label:'Buy & Hold', data:signals.map(s=>s.hold_value), borderColor:'#64748b', borderDash:[4,4], borderWidth:1.5, pointRadius:0, fill:false, tension:0.3 }},
    ];
  }}
  charts.portfolio = new Chart(ctx, {{
    type:'line', data:{{ labels:makeLabels(signals), datasets }},
    options:{{ ...chartDefaults, interaction:{{ intersect:false, mode:'index' }} }}
  }});
}}

function renderConfidenceChart(signals, color) {{
  const ctx = document.getElementById('chartConf').getContext('2d');
  const colors = signals.map(s => sigColor(s.signal));
  charts.conf = new Chart(ctx, {{
    type:'bar',
    data:{{ labels:makeLabels(signals), datasets:[{{ label:'Confidence %', data:signals.map(s=>s.confidence), backgroundColor:colors.map(c=>c+'60'), borderColor:colors, borderWidth:1 }}] }},
    options:{{ ...chartDefaults, plugins:{{ ...chartDefaults.plugins, legend:{{ display:false }} }}, scales:{{ ...chartDefaults.scales, y:{{ ...chartDefaults.scales.y, min:40, max:100 }} }} }}
  }});
}}

function renderDistributionChart(signals) {{
  const counts = {{}};
  signals.forEach(s => {{ counts[s.signal] = (counts[s.signal]||0) + 1; }});
  const labels = Object.keys(counts);
  const data = Object.values(counts);
  const bgColors = labels.map(l => sigColor(l) + '40');
  const bdColors = labels.map(l => sigColor(l));
  const ctx = document.getElementById('chartDist').getContext('2d');
  charts.dist = new Chart(ctx, {{
    type:'doughnut',
    data:{{ labels, datasets:[{{ data, backgroundColor:bgColors, borderColor:bdColors, borderWidth:2 }}] }},
    options:{{ responsive:true, maintainAspectRatio:false, plugins:{{ legend:{{ position:'right', labels:{{ color:'#94a3b8', font:{{ family:'JetBrains Mono',size:11 }}, padding:12 }} }} }}, cutout:'65%' }}
  }});
}}

function renderPriceChart(signals, color) {{
  const ctx = document.getElementById('chartPrice').getContext('2d');
  const buyPts = signals.map(s => s.signal.includes('BUY') ? s.close : null);
  const sellPts = signals.map(s => s.signal.includes('SELL') ? s.close : null);
  const datasets = [
    {{ label:'Close', data:signals.map(s=>s.close), borderColor:color.main, borderWidth:1.5, pointRadius:0, tension:0.2, fill:false }},
    {{ label:'BUY', data:buyPts, borderColor:'transparent', backgroundColor:'#22c55e', pointRadius:4, pointStyle:'triangle', showLine:false }},
    {{ label:'SELL', data:sellPts, borderColor:'transparent', backgroundColor:'#ef4444', pointRadius:4, pointStyle:'rect', pointRotation:45, showLine:false }},
  ];
  charts.price = new Chart(ctx, {{
    type:'line', data:{{ labels:makeLabels(signals), datasets }},
    options:{{ ...chartDefaults, interaction:{{ intersect:false, mode:'index' }} }}
  }});
}}

function renderRSIChart(signals, color) {{
  const ctx = document.getElementById('chartRSI').getContext('2d');
  charts.rsi = new Chart(ctx, {{
    type:'line',
    data:{{ labels:makeLabels(signals), datasets:[{{ label:'RSI 14h', data:signals.map(s=>s.rsi), borderColor:color.main, borderWidth:1.5, pointRadius:0, tension:0.3, fill:false }}] }},
    options:{{ ...chartDefaults, plugins:{{ ...chartDefaults.plugins, legend:{{ display:false }} }}, scales:{{ ...chartDefaults.scales, y:{{ ...chartDefaults.scales.y, min:20, max:80 }} }} }},
    plugins:[{{
      id:'rsiBands',
      beforeDraw(chart) {{
        const {{ ctx, chartArea:{{top,bottom,left,right}}, scales:{{y}} }} = chart;
        const y70=y.getPixelForValue(70), y30=y.getPixelForValue(30);
        ctx.save();
        ctx.fillStyle='rgba(239,68,68,0.06)'; ctx.fillRect(left,top,right-left,y70-top);
        ctx.fillStyle='rgba(34,197,94,0.06)'; ctx.fillRect(left,y30,right-left,bottom-y30);
        ctx.strokeStyle='#ef444440'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
        ctx.beginPath(); ctx.moveTo(left,y70); ctx.lineTo(right,y70); ctx.stroke();
        ctx.strokeStyle='#22c55e40';
        ctx.beginPath(); ctx.moveTo(left,y30); ctx.lineTo(right,y30); ctx.stroke();
        ctx.restore();
      }}
    }}]
  }});
}}

function renderBBChart(signals, color) {{
  const ctx = document.getElementById('chartBB').getContext('2d');
  charts.bb = new Chart(ctx, {{
    type:'line',
    data:{{ labels:makeLabels(signals), datasets:[{{ label:'BB Position', data:signals.map(s=>s.bb_position), borderColor:'#f59e0b', borderWidth:1.5, pointRadius:0, tension:0.3, fill:false }}] }},
    options:{{ ...chartDefaults, plugins:{{ ...chartDefaults.plugins, legend:{{ display:false }} }} }},
    plugins:[{{
      id:'bbBands',
      beforeDraw(chart) {{
        const {{ ctx, chartArea:{{top,bottom,left,right}}, scales:{{y}} }} = chart;
        ctx.save(); ctx.strokeStyle='#64748b40'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
        [0,0.5,1].forEach(v => {{
          const yy=y.getPixelForValue(v);
          ctx.beginPath(); ctx.moveTo(left,yy); ctx.lineTo(right,yy); ctx.stroke();
        }});
        ctx.restore();
      }}
    }}]
  }});
}}

function renderSignalTable(signals) {{
  const container = document.getElementById('tabContent');
  const recent = [...signals].reverse().slice(0,60);
  let rows = recent.map(s => {{
    const cls = sigCssClass(s.signal);
    const actClass = s.actual==='UP' ? 'actual-up' : s.actual==='DOWN' ? 'actual-down' : '';
    const isCorrect = (s.signal.includes('BUY') && s.actual==='UP') || (s.signal.includes('SELL') && s.actual==='DOWN');
    const isWrong = s.actual && !isCorrect && s.signal!=='HOLD';
    const rowClass = isCorrect ? 'correct' : isWrong ? 'wrong' : '';
    const allocCol = STRATEGY==='v2' ? '<td style="text-align:right">' + (s.allocation!==undefined ? (s.allocation*100).toFixed(0)+'%' : '-') + '</td>' : '';
    return '<tr class="'+rowClass+'">'+
      '<td>'+s.datetime+'</td>'+
      '<td style="text-align:right">'+s.close.toLocaleString('en',{{minimumFractionDigits:2}})+'</td>'+
      '<td class="'+cls+'">'+s.signal+'</td>'+
      '<td style="text-align:right">'+s.confidence+'%</td>'+
      '<td style="text-align:center">'+s.buy_votes+'/'+s.total_votes+'</td>'+
      '<td style="text-align:right">'+s.rsi+'</td>'+
      '<td class="'+actClass+'" style="text-align:center">'+(s.actual||'-')+'</td>'+
      allocCol+
      '<td style="text-align:right">$'+s.portfolio_value.toFixed(2)+'</td>'+
      '</tr>';
  }}).join('');

  const allocHeader = STRATEGY==='v2' ? '<th style="text-align:right">Alloc</th>' : '';

  container.innerHTML =
    '<div class="signal-table-wrap"><h3>RECENT SIGNALS -- '+activeAsset+' (last 60)</h3><table><thead><tr>' +
    '<th>Datetime</th><th style="text-align:right">Close</th><th>Signal</th><th style="text-align:right">Conf</th><th style="text-align:center">Votes</th><th style="text-align:right">RSI</th><th style="text-align:center">Actual</th>' +
    allocHeader +
    '<th style="text-align:right">Portfolio</th></tr></thead><tbody>'+rows+'</tbody></table></div>';
}}

function renderAll() {{ renderSummaryCards(); renderTabs(); renderContent(); }}
renderAll();
</script>
</body>
</html>'''

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  Dashboard saved to {html_file} (open in browser)")
    return html_file
def run_mode_a(assets_list, strategy='v1'):
    """Full review: update -> diagnostic -> best models -> signals -> chart."""
    strat_label = {'v1': 'V1 (3-tier)', 'v2': 'V2 (5-tier graduated)', 'v3': 'V3 (V1 vs V2 comparison)'}[strategy]
    print("\n" + "=" * 60)
    print(f"  MODE A: FULL HOURLY REVIEW  [{strat_label}]")
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

        signals = generate_signals(asset_name, model_names, window, REPLAY_HOURS, strategy=strategy)
        if strategy == 'v3':
            signals = simulate_portfolio_v3(signals)
        elif strategy == 'v2':
            signals = simulate_portfolio_v2(signals)
        else:
            signals = simulate_portfolio(signals)
        all_signals[asset_name] = signals

        if signals:
            latest = signals[-1]
            alloc_str = f" | alloc={latest.get('allocation', 'N/A')}" if strategy in ('v2', 'v3') else ''
            print(f"\n  >> {asset_name} LATEST: {latest['signal']} ({latest['confidence']:.0f}%) "
                  f"| price={latest['close']:,.2f}{alloc_str}")

    # Step 4: Export
    suffix = {'v1': '', 'v2': '_v2', 'v3': '_v3'}[strategy]
    export_chart_data(all_signals, f'output/charts/hourly_chart_data{suffix}.json')
    export_html_dashboard(all_signals, strategy=strategy)

    print("\n" + "=" * 60)
    print(f"  MODE A (HOURLY) COMPLETE  [{strat_label}]")
    print("=" * 60)
def run_mode_b(assets_list, strategy='v1'):
    """Quick run: update -> read hourly_best_models.csv -> signals -> chart."""
    strat_label = {'v1': 'V1 (3-tier)', 'v2': 'V2 (5-tier graduated)', 'v3': 'V3 (V1 vs V2 comparison)'}[strategy]
    print("\n" + "=" * 60)
    print(f"  MODE B: QUICK HOURLY RUN  [{strat_label}]")
    print("=" * 60)

    if not os.path.exists('data/hourly_best_models.csv'):
        print("\nERROR: hourly_best_models.csv not found!")
        print("Please run Mode A first to generate best model configurations.")
        return

    df_best = pd.read_csv('data/hourly_best_models.csv')
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

        signals = generate_signals(asset_name, model_names, window, REPLAY_HOURS, strategy=strategy)
        if strategy == 'v3':
            signals = simulate_portfolio_v3(signals)
        elif strategy == 'v2':
            signals = simulate_portfolio_v2(signals)
        else:
            signals = simulate_portfolio(signals)
        all_signals[asset_name] = signals

        if signals:
            latest = signals[-1]
            alloc_str = f" | alloc={latest.get('allocation', 'N/A')}" if strategy in ('v2', 'v3') else ''
            print(f"\n  >> {asset_name} LATEST: {latest['signal']} ({latest['confidence']:.0f}%) "
                  f"| price={latest['close']:,.2f}{alloc_str}")

    # Step 3: Export
    suffix = {'v1': '', 'v2': '_v2', 'v3': '_v3'}[strategy]
    export_chart_data(all_signals, f'output/charts/hourly_chart_data{suffix}.json')
    export_html_dashboard(all_signals, strategy=strategy)

    print("\n" + "=" * 60)
    print(f"  MODE B (HOURLY) COMPLETE  [{strat_label}]")
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

    # Strategy selection
    print("\nChoose strategy:")
    print("  1. V1 -- 3-tier (BUY / HOLD / SELL) -- all-in / all-out")
    print("  2. V2 -- 5-tier (STRONG BUY / BUY / HOLD / SELL / STRONG SELL)")
    print(f"          Graduated sizing: STRONG=100%, Normal=50%, HOLD=stay")
    print(f"          Confidence threshold for STRONG: {CONFIDENCE_THRESHOLD}%")
    print("  3. V3 -- Comparison (V1 vs V2 vs Buy&Hold on same chart)")
    strat_choice = input("Enter choice (1-3): ").strip()
    strategy = 'v2' if strat_choice == '2' else 'v3' if strat_choice == '3' else 'v1'

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

    strat_label = {'v1': 'V1 (3-tier)', 'v2': 'V2 (5-tier graduated)', 'v3': 'V3 (V1 vs V2 comparison)'}[strategy]
    print(f"\nAssets: {', '.join(assets_list)}")
    print(f"Mode: {'A (Full Review)' if mode == 'A' else 'B (Quick Run)'}")
    print(f"Strategy: {strat_label}")

    if mode == 'A':
        run_mode_a(assets_list, strategy=strategy)
    else:
        run_mode_b(assets_list, strategy=strategy)

    suffix = {'v1': '', 'v2': '_v2', 'v3': '_v3'}[strategy]
    print(f"\nDone!")
    print(f"  Data: hourly_chart_data{suffix}.json")
    print(f"  Dashboard: hourly_dashboard{suffix}.html (open in browser)")


if __name__ == '__main__':
    main()
