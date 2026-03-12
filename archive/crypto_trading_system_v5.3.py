"""
Crypto Hourly Trading System — V5.3
============================================================
ML trading system for BTC, ETH, XRP, DOGE with 4h and 8h horizons.
125 features -> walk-forward ML -> BUY/SELL/HOLD signals.

V5.3 changes vs V5:
  - Feature analysis LGBM lightened: n_estimators=100, max_depth=4 (was 200/6)
    Same as diagnostic models — feature analysis only needs relative ranking.
  - LOKY_MAX_CPU_COUNT capped at N_JOBS_PARALLEL (not os.cpu_count())
    Prevents joblib from spawning more workers than intended.
  - OMP/MKL/OpenBLAS thread limits set to 1 per worker
    Prevents numpy/scipy hidden multithreading inside parallel workers.

Modes:
  B. Quick run (saved models)    D. Full pipeline (feature analysis -> diagnostic)
  E. Iterative refinement        5/6/7. Quick BTC/ETH/XRP

CLI Usage (skip all menus):
  python crypto_trading_system_v5.3.py B BTC 4,8h
  python crypto_trading_system_v5.3.py D ETH 8h 1y
  python crypto_trading_system_v5.3.py D BTC,ETH 4,8h 2y

Outputs:
  charts/{ASSET}_backtest.png
  models/crypto_hourly_best_models.csv
  models/crypto_hourly_chart_data.json
"""

import sys
import os

# ============================================================
# SUPPRESS ALL WARNINGS (must be before other imports)
# Must be set before ANY sklearn/joblib imports so child processes inherit it
# ============================================================
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
# Prevent hidden multithreading inside parallel workers (numpy, scipy, BLAS)
# Each joblib worker should be single-threaded; parallelism is at the joblib level.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
# Suppress sklearn parallel warnings specifically
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')

import time
import json
import contextlib


@contextlib.contextmanager
def _suppress_stderr():
    """Suppress stderr at file descriptor level (works with child processes on Windows)."""
    try:
        old_stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stderr_fd)
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
# Cap loky worker pool to configured parallelism (not raw CPU count)
os.environ['LOKY_MAX_CPU_COUNT'] = str(N_JOBS_PARALLEL)


def _kill_orphan_workers():
    """Kill any orphaned python/loky workers from previous interrupted runs.
    On Windows, Ctrl+C kills the parent but loky child processes survive,
    silently eating CPU and slowing down subsequent runs."""
    import subprocess
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ['wmic', 'process', 'where', "name='python.exe'", 'get', 'processid,commandline'],
            capture_output=True, text=True, timeout=10
        )
        killed = 0
        for line in result.stdout.splitlines():
            if 'loky' in line.lower():
                parts = line.strip().split()
                if parts:
                    try:
                        pid = int(parts[-1])
                        if pid != my_pid:
                            os.kill(pid, 9)
                            killed += 1
                    except (ValueError, OSError):
                        pass
        if killed:
            print(f"  [Cleanup] Killed {killed} orphaned worker(s) from previous runs")
    except Exception:
        pass


# Matplotlib (non-interactive backend for server/headless)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch

# Windows UTF-8 fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# CANDIDATE FEATURE SETS (from crypto_feature_analysis.py V2)
# ============================================================

FEATURE_SET_A = [
    'hour_cos', 'logret_240h', 'vol_ratio_12_48', 'logret_72h',
    'rsi_14h', 'logret_120h', 'stoch_k_14h', 'sma20_to_sma50h',
    'volatility_12h', 'xa_dax_relstr5d', 'xa_sp500_relstr5d',
    'spread_24h_4h', 'atr_pct_14h', 'volume_ratio_h', 'volatility_48h',
    'price_to_sma100h', 'logret_4h', 'spread_120h_12h',
]

FEATURE_SET_B = [
    'rsi_14h', 'hour_cos', 'hour_sin', 'sma20_to_sma50h', 'logret_72h',
    'm_sp500_chg5d', 'm_dxy_vol5d', 'xa_dax_corr30d', 'xa_dax_relstr5d',
    'fg_chg5d', 'm_us10y_chg10d', 'm_eurusd_vol5d', 'm_eurusd_chg5d',
    'vix_spike',
]

ACTIVE_FEATURE_SET = 'A'


# ============================================================
# ASSET CONFIGURATION
# ============================================================
ASSETS = {
    'BTC':   {'source': 'binance', 'ticker': 'BTC/USDT',  'file': 'data/btc_hourly_data.csv',  'start': '2017-08-01T00:00:00Z'},
    'ETH':   {'source': 'binance', 'ticker': 'ETH/USDT',  'file': 'data/eth_hourly_data.csv',  'start': '2017-08-01T00:00:00Z'},
    'XRP':   {'source': 'binance', 'ticker': 'XRP/USDT',  'file': 'data/xrp_hourly_data.csv',  'start': '2018-05-01T00:00:00Z'},
    'DOGE':  {'source': 'binance', 'ticker': 'DOGE/USDT', 'file': 'data/doge_hourly_data.csv', 'start': '2019-07-01T00:00:00Z'},
    'SMI':   {'source': 'yfinance', 'ticker': '^SSMI',    'file': 'data/smi_hourly_data.csv',  'start': None},
    'DAX':   {'source': 'yfinance', 'ticker': '^GDAXI',   'file': 'data/dax_hourly_data.csv',  'start': None},
    'CAC40': {'source': 'yfinance', 'ticker': '^FCHI',    'file': 'data/cac40_hourly_data.csv', 'start': None},
}

PREDICTION_HORIZON = 4            # default horizon (legacy)
AVAILABLE_HORIZONS = [4, 8]       # 4h and 8h models

# Create output folders
for _d in ['data', 'data/macro_data', 'charts', 'models', 'config']:
    os.makedirs(_d, exist_ok=True)
TRADING_FEE = 0.0009  # 0.09% Revolut X taker fee (applied on BUY and SELL)
MIN_CONFIDENCE = 75   # Minimum confidence % for strategy signals
REPLAY_HOURS = 200
REPLAY_HOURS_F = 400   # Mode F strategy selection — longer window for more trades
DIAG_STEP = 72
DIAG_WINDOWS = [48, 72, 100, 150, 200]  # 300/500 removed: slow and rarely win


# ============================================================
# FEATURE SET HELPERS
# ============================================================
def _get_active_features():
    if ACTIVE_FEATURE_SET == 'B':
        return list(FEATURE_SET_B)
    return list(FEATURE_SET_A)


def _get_set_label():
    if ACTIVE_FEATURE_SET == 'B':
        return f"Set B (KEEP 14 consensus, {len(FEATURE_SET_B)} features)"
    return f"Set A (Top 18 LGBM importance, {len(FEATURE_SET_A)} features)"


def _build_features(df_raw, asset_name='BTC', feature_override=None, horizon=PREDICTION_HORIZON):
    df_full, all_cols = build_all_features(df_raw, asset_name=asset_name, horizon=horizon)
    selected = feature_override if feature_override is not None else _get_active_features()
    missing = [f for f in selected if f not in all_cols]
    if missing:
        print(f"    WARNING: features not found (macro_data/ missing?): {missing}")
        selected = [f for f in selected if f in all_cols]
    if not selected:
        print(f"    ERROR: no valid features! Falling back to all {len(all_cols)} columns.")
        selected = all_cols
    return df_full, selected


# ============================================================
# DATA DOWNLOAD: BINANCE (ccxt)
# ============================================================
def download_binance(asset_name, update_only=True):
    import ccxt
    config = ASSETS[asset_name]
    filepath = config['file']
    exchange = ccxt.binance()
    symbol = config['ticker']
    timeframe = '1h'
    limit = 1000

    if os.path.exists(filepath) and update_only:
        df_existing = pd.read_csv(filepath)
        df_existing['datetime'] = pd.to_datetime(df_existing['datetime'])
        last_ts = int(df_existing['timestamp'].max()) + 1
        print(f"  Updating {asset_name} from {df_existing['datetime'].iloc[-1]}...")

        all_new = []
        since = last_ts
        while True:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not candles:
                break
            for c in candles:
                all_new.append({
                    'timestamp': c[0],
                    'datetime': pd.Timestamp(c[0], unit='ms'),
                    'open': c[1], 'high': c[2], 'low': c[3],
                    'close': c[4], 'volume': c[5],
                })
            since = candles[-1][0] + 1
            if len(candles) < limit:
                break
            time.sleep(0.1)

        if all_new:
            df_new = pd.DataFrame(all_new)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
            df_combined.to_csv(filepath, index=False)
            print(f"    {asset_name}: {len(df_new)} new candles -> {len(df_combined)} total")
        else:
            print(f"    {asset_name}: already up to date ({len(df_existing)} candles)")
            df_combined = df_existing
        return df_combined
    else:
        print(f"  Downloading {asset_name} full history from Binance...")
        since_str = config.get('start', '2020-01-01T00:00:00Z')
        since = exchange.parse8601(since_str)
        all_candles = []
        while True:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not candles:
                break
            for c in candles:
                all_candles.append({
                    'timestamp': c[0],
                    'datetime': pd.Timestamp(c[0], unit='ms'),
                    'open': c[1], 'high': c[2], 'low': c[3],
                    'close': c[4], 'volume': c[5],
                })
            since = candles[-1][0] + 1
            if len(candles) < limit:
                break
            time.sleep(0.1)
        df = pd.DataFrame(all_candles)
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
        df.to_csv(filepath, index=False)
        print(f"    {asset_name}: {len(df)} candles downloaded")
        return df


# ============================================================
# DATA DOWNLOAD: YFINANCE
# ============================================================
def _normalize_yf_chunk(chunk):
    df = chunk.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower().strip() for c in df.columns]
    if df.index.name and 'date' in df.index.name.lower():
        df = df.reset_index()
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'datetime'})
    if 'datetime' not in df.columns:
        df['datetime'] = df.index
        df = df.reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_localize(None)
    needed = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    for col in needed:
        if col not in df.columns:
            df[col] = 0
    return df[needed].copy()


def download_yfinance(asset_name, update_only=True):
    import yfinance as yf
    config = ASSETS[asset_name]
    filepath = config['file']
    ticker = config['ticker']

    if os.path.exists(filepath) and update_only:
        df_existing = pd.read_csv(filepath)
        df_existing['datetime'] = pd.to_datetime(df_existing['datetime'])
        print(f"  Updating {asset_name} from {df_existing['datetime'].iloc[-1]}...")
        chunk = yf.download(ticker, period='5d', interval='1h', progress=False)
        if chunk is not None and len(chunk) > 0:
            df_new = _normalize_yf_chunk(chunk)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset='datetime').sort_values('datetime').reset_index(drop=True)
            df_combined.to_csv(filepath, index=False)
            print(f"    {asset_name}: {len(df_new)} new candles -> {len(df_combined)} total")
            return df_combined
        else:
            print(f"    {asset_name}: no new data")
            return df_existing
    else:
        print(f"  Downloading {asset_name} hourly data (last 2 years)...")
        all_chunks = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=729)
        chunk_days = 59
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_days), end_date)
            try:
                chunk = yf.download(ticker, start=current_start.strftime('%Y-%m-%d'),
                    end=current_end.strftime('%Y-%m-%d'), interval='1h', progress=False)
                if chunk is not None and len(chunk) > 0:
                    all_chunks.append(_normalize_yf_chunk(chunk))
            except Exception as e:
                print(f"    Warning: chunk {current_start.date()} -> {current_end.date()}: {e}")
            current_start = current_end
            time.sleep(0.2)
        if not all_chunks:
            print(f"    {asset_name}: no data downloaded!")
            return None
        df = pd.concat(all_chunks, ignore_index=True)
        df = df.drop_duplicates(subset='datetime').sort_values('datetime').reset_index(drop=True)
        df.to_csv(filepath, index=False)
        print(f"    {asset_name}: {len(df)} hourly candles downloaded")
        return df


def download_asset(asset_name, update_only=True):
    config = ASSETS[asset_name]
    if config['source'] == 'binance':
        return download_binance(asset_name, update_only)
    else:
        return download_yfinance(asset_name, update_only)


def update_all_data(assets_list=None):
    if assets_list is None:
        assets_list = list(ASSETS.keys())
    print("\n" + "=" * 60)
    print("  UPDATING HOURLY DATA")
    print("=" * 60)
    for asset_name in assets_list:
        try:
            download_asset(asset_name, update_only=True)
        except Exception as e:
            print(f"  ERROR updating {asset_name}: {e}")


def load_data(asset_name):
    config = ASSETS[asset_name]
    filepath = config['file']
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found. Run update first.")
        return None
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close']).reset_index(drop=True)
    return df


# ============================================================
# HOURLY FEATURE ENGINEERING (ALL 36 technical features)
# ============================================================
def build_hourly_features(df_hourly, horizon=PREDICTION_HORIZON):
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

    # VVR: Volume-to-Volatility Ratio (distinguishes real moves from noise)
    vol_12 = df['close'].pct_change().rolling(12).std()
    vol_12 = vol_12.replace(0, np.nan)
    df['vvr_12h'] = df['volume_ratio_h'] / vol_12
    df['vvr_12h'] = df['vvr_12h'].fillna(1.0)
    df['vvr_12h'] = df['vvr_12h'].clip(0, 20)  # cap outliers

    hour = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    dow = df['datetime'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # ---- DERIVATIVES ----
    # First derivative (velocity) -- already captured by logret_1h, but explicit:
    df['price_velocity_1h'] = df['close'].diff() / df['close'].shift(1)     # pct change
    df['price_velocity_4h'] = df['close'].diff(4) / df['close'].shift(4)

    # Second derivative (acceleration) -- change in the rate of change
    df['price_accel_1h'] = df['logret_1h'].diff()          # d^2price/dt^2 (1h resolution)
    df['price_accel_4h'] = df['logret_4h'].diff(4)         # d^2price/dt^2 (4h resolution)
    df['price_accel_12h'] = df['logret_12h'].diff(12)      # d^2price/dt^2 (12h resolution)
    df['price_accel_24h'] = df['logret_24h'].diff(24)      # d^2price/dt^2 (24h resolution)

    # Jerk (third derivative) -- change in acceleration
    df['price_jerk_1h'] = df['price_accel_1h'].diff()      # d^3price/dt^3

    future_return = df['close'].shift(-horizon) / df['close'] - 1
    rolling_median = future_return.rolling(200, min_periods=50).median().shift(horizon)
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
        'volume_ratio_h', 'vvr_12h',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'price_velocity_1h', 'price_velocity_4h',
        'price_accel_1h', 'price_accel_4h', 'price_accel_12h', 'price_accel_24h',
        'price_jerk_1h',
    ]

    keep_cols = ['datetime', 'close', 'high', 'low', 'volume'] + feature_cols + ['label']
    df = df[keep_cols].copy()

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
# V2: MACRO & SENTIMENT FEATURES
# ============================================================
MACRO_DIR = 'data/macro_data'
DATA_DIR = 'data'
CHARTS_DIR = 'charts'
MODELS_DIR = 'models'
CONFIG_DIR = 'config'
_macro_cache = {}


def _load_macro_csv(filename):
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


def build_all_features(df_hourly, asset_name='BTC', horizon=PREDICTION_HORIZON):
    df, base_cols = build_hourly_features(df_hourly, horizon=horizon)
    all_cols = list(base_cols)
    added = 0

    if not os.path.exists(MACRO_DIR):
        print(f"    macro_data/ not found -- technical features only")
        return df, all_cols

    df['_merge_date'] = pd.to_datetime(df['datetime']).dt.normalize()

    macro_df = _load_macro_csv('macro_daily.csv')
    if macro_df is not None:
        macro_feats = _compute_macro_features(macro_df, prefix='m_')
        vix_feats = _compute_vix_regime(macro_df)
        macro_all = pd.concat([macro_feats, vix_feats], axis=1)
        macro_all['_merge_date'] = macro_all.index.normalize()
        df = df.merge(macro_all, on='_merge_date', how='left')
        new_cols = [c for c in macro_all.columns if c != '_merge_date']
        all_cols.extend(new_cols)
        added += len(new_cols)

    fg_df = _load_macro_csv('fear_greed.csv')
    if fg_df is not None:
        fg_feats = _compute_fear_greed_features(fg_df)
        fg_feats['_merge_date'] = fg_feats.index.normalize()
        df = df.merge(fg_feats, on='_merge_date', how='left')
        new_cols = [c for c in fg_feats.columns if c != '_merge_date']
        all_cols.extend(new_cols)
        added += len(new_cols)

    cross_df = _load_macro_csv('cross_asset.csv')
    if cross_df is not None:
        asset_map = {
            'BTC': 'BTC_USD', 'ETH': 'ETH_USD',
            'XRP': 'BTC_USD', 'DOGE': 'BTC_USD',
            'SMI': 'DAX', 'DAX': 'DAX', 'CAC40': 'DAX',
        }
        target_col = asset_map.get(asset_name, 'BTC_USD')
        xa_feats = _compute_cross_asset_features(cross_df, target_col, prefix='xa_')
        if len(xa_feats.columns) > 0:
            xa_feats['_merge_date'] = xa_feats.index.normalize()
            df = df.merge(xa_feats, on='_merge_date', how='left')
            new_cols = [c for c in xa_feats.columns if c != '_merge_date']
            all_cols.extend(new_cols)
            added += len(new_cols)

    df = df.drop(columns=['_merge_date'], errors='ignore')
    if '_merge_date' in all_cols:
        all_cols.remove('_merge_date')
    all_cols = list(dict.fromkeys(all_cols))
    all_cols = [c for c in all_cols if c in df.columns]
    macro_cols = [c for c in all_cols if c not in base_cols]
    if macro_cols:
        df[macro_cols] = df[macro_cols].ffill()

    print(f"    All features: {len(base_cols)} base + {added} macro/sentiment/cross-asset = {len(all_cols)} total")
    return df, all_cols


# ============================================================
# MODELS
# ============================================================
ALL_MODELS = get_all_models()


# ============================================================
# FEATURE ANALYSIS (integrated from crypto_feature_analysis.py)
# ============================================================
ANALYSIS_WINDOW = 500
ANALYSIS_STEP = 72


def _classify_feature(feat):
    """Classify a feature into its category."""
    if feat.startswith('m_'):
        return 'MACRO'
    elif feat.startswith('xa_'):
        return 'CROSS-ASSET'
    elif feat.startswith('fg_') or feat == 'vix_spike':
        return 'SENTIMENT'
    else:
        return 'BASE'


def _quick_score(df_features, feature_cols, window=ANALYSIS_WINDOW, step=ANALYSIS_STEP, device=None):
    """Fast walk-forward test with LGBM only.
    Returns (accuracy, alpha, n_tests).
    Alpha = strategy return - buy & hold return (same period, with 0.09% fees).
    device: override LGBM device ('cpu' for parallel safety, None = use default)."""
    from lightgbm import LGBMClassifier

    lgbm_device = device if device else LGBM_DEVICE
    n = len(df_features)
    min_start = window + 50
    if n < min_start + 30:
        return 0, 0, 0

    correct = 0
    total = 0

    # Portfolio simulation alongside accuracy
    cash     = 1.0
    in_pos   = False
    entry_px = 0.0
    start_px = float(df_features.iloc[min_start]['close'])

    for i in range(min_start, n, step):
        train    = df_features.iloc[max(0, i - window):i]
        test_row = df_features.iloc[i:i+1]
        X_train  = train[feature_cols]
        y_train  = train['label'].values
        X_test   = test_row[feature_cols]
        y_true   = test_row['label'].values[0]
        price    = float(test_row['close'].values[0])

        if len(np.unique(y_train)) < 2:
            continue

        scaler    = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train),
                                 columns=feature_cols, index=X_train.index)
        X_test_s  = pd.DataFrame(scaler.transform(X_test),
                                 columns=feature_cols, index=X_test.index)

        model = LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            class_weight='balanced', verbose=-1, random_state=42,
            device=lgbm_device
        )
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)[0]

        if pred == y_true:
            correct += 1
        total += 1

        # Portfolio: BUY on pred=1, SELL on pred=0
        if pred == 1 and not in_pos:
            in_pos   = True
            entry_px = price * (1 + TRADING_FEE)
        elif pred == 0 and in_pos:
            cash   *= (price * (1 - TRADING_FEE)) / entry_px
            in_pos  = False

    # Close open position at end
    if in_pos and total > 0:
        last_px = float(df_features.iloc[-1]['close'])
        cash   *= (last_px * (1 - TRADING_FEE)) / entry_px

    last_px   = float(df_features.iloc[-1]['close'])
    strat_ret = (cash - 1.0) * 100
    bh_ret    = (last_px / start_px - 1) * 100
    alpha     = round(strat_ret - bh_ret, 2)
    accuracy  = correct / total * 100 if total > 0 else 0
    return accuracy, alpha, total


def _quick_accuracy(df_features, feature_cols, window=ANALYSIS_WINDOW, step=ANALYSIS_STEP, device=None):
    """Backward-compatible wrapper -- returns (accuracy, n_tests). Used by Mode E."""
    acc, _, n = _quick_score(df_features, feature_cols, window=window, step=step, device=device)
    return acc, n


def _test_lgbm_importance(df_features, feature_cols):
    """Train LGBM and extract feature importance."""
    from lightgbm import LGBMClassifier

    print("\n  [1/5] LGBM Feature Importance (gain-based)")
    n = len(df_features)
    train = df_features.iloc[:int(n * 0.7)]
    X = train[feature_cols]
    y = train['label'].values

    scaler = StandardScaler()
    X_s = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    model = LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        class_weight='balanced', verbose=-1, random_state=42,
        device=LGBM_DEVICE
    )
    model.fit(X_s, y)

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance['pct'] = importance['importance'] / importance['importance'].sum() * 100
    importance['cumulative_pct'] = importance['pct'].cumsum()

    for _, row in importance.head(20).iterrows():
        bar = '#' * int(row['pct'] * 2)
        ftype = _classify_feature(row['feature'])
        print(f"    {row['feature']:30s} {row['pct']:5.1f}% [{ftype:>10s}] {bar}")

    low_count = len(importance[importance['pct'] < 1.0])
    print(f"    ... {low_count} features below 1% importance")
    return importance


def _perm_one_feature(df_features, feature_cols, feat, baseline_acc, baseline_alpha):
    """Helper for parallel permutation test. Returns (feat, acc_drop, alpha_drop)."""
    import os, warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    df_shuffled = df_features.copy()
    df_shuffled[feat] = np.random.permutation(df_shuffled[feat].values)
    shuffled_acc, shuffled_alpha, _ = _quick_score(df_shuffled, feature_cols, device='cpu')
    return feat, baseline_acc - shuffled_acc, baseline_alpha - shuffled_alpha


def _test_permutation_importance(df_features, feature_cols):
    """Shuffle each feature and measure accuracy + alpha drop. Parallelized."""
    print("\n  [2/5] Permutation Importance (parallel)")
    baseline_acc, baseline_alpha, n_tests = _quick_score(df_features, feature_cols)
    print(f"    Baseline: {baseline_acc:.1f}% acc | {baseline_alpha:+.1f}% alpha (n={n_tests})")

    n_workers = min(N_JOBS_PARALLEL, len(feature_cols))
    print(f"    Testing {len(feature_cols)} features ({n_workers} workers)...")
    t0 = time.time()
    with _suppress_stderr():
        perm_results = Parallel(n_jobs=n_workers, verbose=0)(
            delayed(_perm_one_feature)(df_features, feature_cols, feat, baseline_acc, baseline_alpha)
            for feat in feature_cols
        )
    print(f"    Done in {(time.time() - t0)/60:.1f} min")

    results = []
    for feat, acc_drop, alpha_drop in perm_results:
        results.append({'feature': feat, 'acc_drop': acc_drop, 'alpha_drop': alpha_drop})
        print(f"    {feat:30s} acc_drop: {acc_drop:+5.1f}%  alpha_drop: {alpha_drop:+6.1f}%")

    return pd.DataFrame(results).sort_values('acc_drop', ascending=False)


def _ablation_one_feature(df_features, feature_cols, feat, baseline_acc, baseline_alpha):
    """Helper for parallel ablation test. Returns (feat, acc, acc_change, alpha_change)."""
    import os, warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    reduced = [f for f in feature_cols if f != feat]
    acc, alpha, _ = _quick_score(df_features, reduced, device='cpu')
    return feat, acc, acc - baseline_acc, alpha - baseline_alpha


def _test_ablation(df_features, feature_cols):
    """Drop each feature one at a time and measure accuracy + alpha. Parallelized."""
    print("\n  [3/5] Ablation Test (parallel, drop one at a time)")
    baseline_acc, baseline_alpha, _ = _quick_score(df_features, feature_cols)
    print(f"    Baseline ({len(feature_cols)} features): {baseline_acc:.1f}% acc | {baseline_alpha:+.1f}% alpha")

    n_workers = min(N_JOBS_PARALLEL, len(feature_cols))
    print(f"    Testing {len(feature_cols)} features ({n_workers} workers)...")
    t0 = time.time()
    with _suppress_stderr():
        ablation_results = Parallel(n_jobs=n_workers, verbose=0)(
            delayed(_ablation_one_feature)(df_features, feature_cols, feat, baseline_acc, baseline_alpha)
            for feat in feature_cols
        )
    print(f"    Done in {(time.time() - t0)/60:.1f} min")

    results = []
    for feat, acc, acc_change, alpha_change in ablation_results:
        results.append({'dropped': feat, 'accuracy': acc,
                        'change': acc_change, 'alpha_change': alpha_change})
        baseline_combined = baseline_acc * (1 + max(baseline_alpha, 0) / 100)
        drop_combined = acc * (1 + max(baseline_alpha + alpha_change, 0) / 100)
        marker = ' ** IMPROVES' if drop_combined > baseline_combined else ''
        print(f"    Drop {feat:30s} -> {acc:5.1f}% ({acc_change:+5.1f}%)  "
              f"alpha_chg: {alpha_change:+6.1f}%{marker}")

    return pd.DataFrame(results).sort_values('change', ascending=False)


def _reduced_one_set(df_features, ranked, n_feat):
    """Helper for parallel reduced set test. Returns (n_feat, acc, alpha, combined_score)."""
    import os, warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    top_n = ranked[:n_feat]
    acc, alpha, _ = _quick_score(df_features, top_n, device='cpu')
    combined = acc * (1 + max(alpha, 0) / 100)
    return n_feat, acc, alpha, combined


def _test_reduced_sets(df_features, feature_cols, importance_df):
    """Test accuracy with top-N features. Parallelized."""
    print("\n  [4/5] Reduced Feature Sets (parallel, top-N by importance)")
    ranked = importance_df['feature'].tolist()

    test_sizes = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50]
    test_sizes = [n for n in test_sizes if n < len(feature_cols)]
    test_sizes.append(len(feature_cols))

    n_workers = min(N_JOBS_PARALLEL, len(test_sizes))
    print(f"    Testing {len(test_sizes)} set sizes ({n_workers} workers)...")
    t0 = time.time()
    with _suppress_stderr():
        reduced_results = Parallel(n_jobs=n_workers, verbose=0)(
            delayed(_reduced_one_set)(df_features, ranked, n_feat)
            for n_feat in test_sizes
        )
    print(f"    Done in {(time.time() - t0)/60:.1f} min")

    results = []
    for n_feat, acc, alpha, combined in sorted(reduced_results):
        results.append({'n_features': n_feat, 'accuracy': acc,
                        'alpha': alpha, 'combined_score': combined})
        bar = '#' * int(acc * 0.5)
        print(f"    Top {n_feat:3d} features: {acc:5.1f}% acc | {alpha:+6.1f}% alpha | "
              f"score={combined:.1f}  {bar}")

    df_results = pd.DataFrame(results)
    best_row = df_results.loc[df_results['combined_score'].idxmax()]
    print(f"\n    OPTIMAL: Top {int(best_row['n_features'])} -> "
          f"{best_row['accuracy']:.1f}% acc | {best_row['alpha']:+.1f}% alpha | "
          f"score={best_row['combined_score']:.1f}")
    return df_results


def _score_features(feature_cols, importance_df, ablation_df, permutation_df):
    """Score features across all tests, return optimal list."""
    print("\n  [5/5] Scoring & Selection")
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

    # Permutation -- accuracy drop + alpha drop
    if permutation_df is not None:
        for _, row in permutation_df.iterrows():
            f = row['feature']
            # Accuracy signal
            if row['acc_drop'] > 0.5:
                scores[f] += 2
            elif row['acc_drop'] > 0:
                scores[f] += 1
            else:
                scores[f] -= 1
            # Alpha signal (shuffling this feature kills alpha = it matters)
            alpha_drop = row.get('alpha_drop', 0)
            if alpha_drop > 10:
                scores[f] += 2
            elif alpha_drop > 3:
                scores[f] += 1
            elif alpha_drop < -3:
                scores[f] -= 1

    # Ablation -- accuracy change + alpha change
    if ablation_df is not None:
        for _, row in ablation_df.iterrows():
            f = row['dropped']
            # Accuracy signal
            if row['change'] > 0.3:
                scores[f] -= 3
            elif row['change'] > 0:
                scores[f] -= 1
            # Alpha signal (dropping this feature improves alpha = it was hurting)
            alpha_change = row.get('alpha_change', 0)
            if alpha_change > 5:
                scores[f] -= 2
            elif alpha_change > 2:
                scores[f] -= 1
            elif alpha_change < -5:
                scores[f] += 1

    score_df = pd.DataFrame([
        {'feature': f, 'score': s, 'category': _classify_feature(f)}
        for f, s in scores.items()
    ]).sort_values('score', ascending=False)

    keep = score_df[score_df['score'] >= 1]['feature'].tolist()
    maybe = score_df[(score_df['score'] >= -1) & (score_df['score'] < 1)]['feature'].tolist()
    drop = score_df[score_df['score'] < -1]['feature'].tolist()

    print(f"    KEEP:  {len(keep)} features (score >= 1)")
    print(f"    MAYBE: {len(maybe)} features (score -1 to 0)")
    print(f"    DROP:  {len(drop)} features (score < -1)")

    # Print keep list
    for f in keep:
        ftype = _classify_feature(f)
        print(f"      + {f} (score: {scores[f]}) [{ftype}]")

    return score_df, keep, maybe, drop


def run_feature_analysis(asset_name, df_features, all_feature_cols):
    """
    Run full 5-test feature analysis on one asset.
    Returns the optimal feature list.
    """
    print(f"\n{'='*60}")
    print(f"  FEATURE ANALYSIS: {asset_name} ({len(all_feature_cols)} features)")
    print(f"{'='*60}")

    t0 = time.time()

    # 1. LGBM importance
    importance_df = _test_lgbm_importance(df_features, all_feature_cols)

    # 2. Permutation importance
    permutation_df = _test_permutation_importance(df_features, all_feature_cols)

    # 3. Ablation
    ablation_df = _test_ablation(df_features, all_feature_cols)

    # 4. Reduced sets
    reduced_df = _test_reduced_sets(df_features, all_feature_cols, importance_df)

    # 5. Score and select
    score_df, keep, maybe, drop = _score_features(
        all_feature_cols, importance_df, ablation_df, permutation_df)

    elapsed = time.time() - t0
    print(f"\n  Feature analysis completed in {elapsed/60:.1f} minutes")

    # Determine optimal set: KEEP features + test with/without MAYBE
    optimal_features = list(keep)

    # Quick test: KEEP only vs KEEP + MAYBE -- use combined_score = acc x (1 + alpha/100)
    if maybe:
        acc_keep,  alpha_keep,  _ = _quick_score(df_features, keep)
        acc_all,   alpha_all,   _ = _quick_score(df_features, keep + maybe)
        score_keep = acc_keep * (1 + max(alpha_keep, 0) / 100)
        score_all  = acc_all  * (1 + max(alpha_all,  0) / 100)
        print(f"\n  KEEP only       ({len(keep):3d} feat): {acc_keep:.1f}% acc | "
              f"{alpha_keep:+.1f}% alpha | score={score_keep:.1f}")
        print(f"  KEEP + MAYBE    ({len(keep)+len(maybe):3d} feat): {acc_all:.1f}% acc | "
              f"{alpha_all:+.1f}% alpha | score={score_all:.1f}")
        if score_all > score_keep + 1.0:
            optimal_features = keep + maybe
            print(f"  >>> Using KEEP + MAYBE ({len(optimal_features)} features)")
        else:
            print(f"  >>> Using KEEP only ({len(optimal_features)} features)")
    else:
        print(f"\n  >>> Using KEEP ({len(optimal_features)} features)")

    # Also check best reduced set from test 4 -- compare by combined_score
    if reduced_df is not None and len(reduced_df) > 0:
        best_n_row     = reduced_df.loc[reduced_df['combined_score'].idxmax()]
        best_n         = int(best_n_row['n_features'])
        best_n_score   = best_n_row['combined_score']
        best_n_acc     = best_n_row['accuracy']
        best_n_alpha   = best_n_row['alpha']
        ranked         = importance_df['feature'].tolist()
        top_n_features = ranked[:best_n]

        opt_acc, opt_alpha, _ = _quick_score(df_features, optimal_features)
        opt_score = opt_acc * (1 + max(opt_alpha, 0) / 100)
        print(f"  Scored optimal  ({len(optimal_features):3d} feat): {opt_acc:.1f}% acc | "
              f"{opt_alpha:+.1f}% alpha | score={opt_score:.1f}")
        print(f"  Top-{best_n} by LGBM ({best_n:3d} feat): {best_n_acc:.1f}% acc | "
              f"{best_n_alpha:+.1f}% alpha | score={best_n_score:.1f}")

        if best_n_score > opt_score + 2.0:
            optimal_features = top_n_features
            print(f"  >>> Switching to Top-{best_n} (score +{best_n_score - opt_score:.1f})")

    # Save analysis results
    score_df.to_csv(f'{MODELS_DIR}/crypto_feature_analysis_{asset_name.lower()}_auto.csv', index=False)

    print(f"\n  OPTIMAL FEATURES ({len(optimal_features)}):")
    for f in optimal_features:
        print(f"    {f} [{_classify_feature(f)}]")

    return optimal_features


# ============================================================
# SIGNAL GENERATION
# ============================================================

# ============================================================
# IMPROVEMENT 2: BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================
def bootstrap_ci(signals, n_bootstrap=500, confidence=0.95):
    """
    Compute bootstrap confidence interval on directional accuracy.
    Runs on the existing signals list -- zero extra compute cost.

    Returns (acc_pct, ci_low_pct, ci_high_pct, n) or None if insufficient data.

    Usage:
        result = bootstrap_ci(signals)
        if result:
            acc, lo, hi, n = result
            print(f"Accuracy: {acc:.1f}% [95% CI: {lo:.1f}%-{hi:.1f}%] (n={n})")
    """
    if not signals or len(signals) < 10:
        return None

    # Build array of 1 (correct) / 0 (wrong) for each signal that has a known outcome
    outcomes = []
    for s in signals:
        if 'correct' in s:
            outcomes.append(1 if s['correct'] else 0)
        elif 'signal' in s and 'actual_direction' in s:
            pred = 1 if s['signal'] == 'BUY' else 0
            actual = s['actual_direction']
            outcomes.append(1 if pred == actual else 0)

    if len(outcomes) < 10:
        return None

    arr = np.array(outcomes, dtype=float)
    rng = np.random.RandomState(42)
    n = len(arr)
    boot_accs = []
    for _ in range(n_bootstrap):
        sample = arr[rng.randint(0, n, size=n)]
        boot_accs.append(sample.mean())

    boot_accs = np.array(boot_accs)
    alpha = 1.0 - confidence
    lo = np.percentile(boot_accs, alpha / 2 * 100)
    hi = np.percentile(boot_accs, (1 - alpha / 2) * 100)
    acc = arr.mean()

    return acc * 100, lo * 100, hi * 100, n


def _print_bootstrap_ci(signals, label=''):
    """Print bootstrap CI if enough data, otherwise skip silently."""
    result = bootstrap_ci(signals)
    if result:
        acc, lo, hi, n = result
        tag = f'  [{label}] ' if label else '  '
        print(f"{tag}Bootstrap accuracy: {acc:.1f}% [95% CI: {lo:.1f}%\u2013{hi:.1f}%] (n={n})")


def generate_signals(asset_name, model_names, window_size, replay_hours=REPLAY_HOURS,
                     feature_override=None, horizon=PREDICTION_HORIZON):
    set_label = _get_set_label() if feature_override is None else f"custom ({len(feature_override)} features)"
    print(f"\n  Generating {horizon}h-ahead signals for {asset_name} "
          f"(models={'+'.join(model_names)}, window={window_size}h, "
          f"replay={replay_hours}h, {set_label})...")

    df_raw = load_data(asset_name)
    if df_raw is None:
        return []

    df_features, feature_cols = _build_features(df_raw, asset_name, feature_override=feature_override, horizon=horizon)
    n = len(df_features)
    start_idx = max(window_size + 50, n - replay_hours)

    signals = []
    count = 0

    for i in range(start_idx, n):
        row = df_features.iloc[i]
        dt_str = row['datetime'].strftime('%Y-%m-%d %H:%M')

        train_start = max(0, i - window_size)
        train = df_features.iloc[train_start:i]
        X_train = train[feature_cols]
        y_train = train['label'].values
        X_test = df_features.iloc[i:i+1][feature_cols]

        if len(np.unique(y_train)) < 2:
            continue
        if X_train.isnull().any().any() or X_test.isnull().any().any():
            continue

        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
        X_test_s  = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

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

        buy_votes = sum(votes)
        total_votes = len(votes)
        buy_ratio = buy_votes / total_votes

        if buy_ratio > 0.5:
            signal = 'BUY'
        elif buy_ratio == 0:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        avg_proba = np.mean(probas)
        if signal == 'SELL':
            confidence = (1 - avg_proba) * 100
        else:
            confidence = avg_proba * 100

        actual = None
        if i + horizon < n:
            future_close = df_features.iloc[i + horizon]['close']
            actual_return = (future_close / row['close'] - 1) * 100
            actual = 'UP' if actual_return > 0 else 'DOWN'

        signals.append({
            'datetime': dt_str,
            'close': float(row['close']),
            'signal': signal,
            'confidence': round(float(confidence), 1),
            'buy_votes': int(buy_votes),
            'total_votes': int(total_votes),
            'rsi': round(float(row.get('rsi_14h', 0)), 1),
            'bb_position': round(float(row.get('bb_position_20h', 0)), 3),
            'volume_ratio': round(float(row.get('volume_ratio_h', 1)), 2),
            'hourly_change': round(float(row.get('logret_1h', 0) * 100), 3),
            'intraday_range': round(float(row.get('intraday_range', 0) * 100), 3),
            'spread_120h_8h': round(float(row.get('spread_120h_8h', 0) * 100), 2),
            'actual': actual,
        })

        count += 1
        if count % 50 == 0:
            print(f"    [{count}] {dt_str}: {signal} ({confidence:.0f}%) "
                  f"| price=${row['close']:,.2f}")

    print(f"  Generated {len(signals)} hourly signals for {asset_name}")
    return signals


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================
def simulate_portfolio(signals, initial=1000):
    if not signals:
        return signals
    cash = initial
    btc_held = 0
    position = 'cash'
    start_price = signals[0]['close']

    for sig in signals:
        price = sig['close']
        hold_value = initial * (price / start_price)

        if sig['signal'] == 'BUY' and position == 'cash':
            # BUY: spend cash, pay fee, get BTC
            btc_held = cash * (1 - TRADING_FEE) / price
            cash = 0
            position = 'invested'
        elif sig['signal'] == 'SELL' and position == 'invested':
            # SELL: sell BTC, pay fee, get cash
            cash = btc_held * price * (1 - TRADING_FEE)
            btc_held = 0
            position = 'cash'

        if position == 'invested':
            current_value = btc_held * price
        else:
            current_value = cash

        sig['portfolio_value'] = round(current_value, 2)
        sig['hold_value'] = round(hold_value, 2)
    return signals


# ============================================================
# BACKTEST CHART (matplotlib)
# ============================================================
def generate_backtest_chart(asset_name, signals, model_info=None):
    """
    Generate a multi-panel PNG chart for the asset backtest.
    Saves to {asset}_backtest.png and opens it.

    Panel 1: Price + BUY/SELL markers
    Panel 2: Portfolio ($1000) vs Buy & Hold
    Panel 3: Stats summary bar
    """
    if not signals or len(signals) < 5:
        print(f"  Not enough signals to chart for {asset_name}")
        return None

    # Prepare data
    dates = pd.to_datetime([s['datetime'] for s in signals])
    prices = [s['close'] for s in signals]
    portfolio = [s.get('portfolio_value', 1000) for s in signals]
    hold = [s.get('hold_value', 1000) for s in signals]
    sigs = [s['signal'] for s in signals]

    buy_dates  = [d for d, s in zip(dates, sigs) if s == 'BUY']
    buy_prices = [p for p, s in zip(prices, sigs) if s == 'BUY']
    sell_dates  = [d for d, s in zip(dates, sigs) if s == 'SELL']
    sell_prices = [p for p, s in zip(prices, sigs) if s == 'SELL']

    # Stats
    n_buy  = sum(1 for s in sigs if s == 'BUY')
    n_sell = sum(1 for s in sigs if s == 'SELL')
    n_hold = sum(1 for s in sigs if s == 'HOLD')
    total  = len(sigs)

    # Accuracy
    correct = sum(1 for s in signals if s.get('actual') is not None and (
        (s['signal'] == 'BUY' and s['actual'] == 'UP') or
        (s['signal'] == 'SELL' and s['actual'] == 'DOWN')
    ))
    with_actual = sum(1 for s in signals if s.get('actual') is not None and s['signal'] in ('BUY', 'SELL'))
    accuracy = (correct / with_actual * 100) if with_actual > 0 else 0

    strategy_return = (portfolio[-1] / portfolio[0] - 1) * 100
    hold_return = (hold[-1] / hold[0] - 1) * 100
    alpha = strategy_return - hold_return

    latest = signals[-1]

    # Model info string + feature set label
    if model_info:
        model_str = f"{model_info['best_combo']} | w={model_info['best_window']}h | {model_info['accuracy']:.1f}% diag"
        fs = model_info.get('feature_set', '')
        n_feat = model_info.get('n_features', '')
        if fs in ('D', 'E2', 'E3') and n_feat:
            set_label = f"Set {fs} ({n_feat} features)"
        elif fs == 'B':
            set_label = f"Set B ({len(FEATURE_SET_B)} features)"
        elif fs:
            set_label = f"Set {fs}"
        else:
            set_label = _get_set_label()
    else:
        model_str = ""
        set_label = _get_set_label()

    # --- PLOT ---
    fig = plt.figure(figsize=(16, 10), facecolor='#0b1120')
    fig.subplots_adjust(hspace=0.08, top=0.88, bottom=0.06, left=0.07, right=0.97)

    # Colors
    c_bg     = '#0b1120'
    c_card   = '#111b2e'
    c_buy    = '#3b82f6'  # blue for BUY
    c_sell   = '#ef4444'  # red for SELL
    c_price  = '#94a3b8'  # gray for price line
    c_gold   = '#eab308'  # yellow for HOLD
    c_text   = '#e0e6f0'
    c_muted  = '#6b7a94'
    c_grid   = '#1e2d4a'

    # TITLE BAR
    fig.text(0.07, 0.95, f'{asset_name}', fontsize=28, fontweight='bold', color=c_text,
             fontfamily='monospace')
    fig.text(0.07, 0.91, f'V3 Backtest  |  {set_label}  |  {model_str}',
             fontsize=11, color=c_muted, fontfamily='monospace')

    # Latest signal badge
    sig_color = c_buy if latest['signal'] == 'BUY' else c_sell if latest['signal'] == 'SELL' else c_gold
    fig.text(0.78, 0.945, f"LATEST: {latest['signal']}  {latest['confidence']:.0f}%",
             fontsize=16, fontweight='bold', color=sig_color, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=sig_color + '20', edgecolor=sig_color, linewidth=1.5))
    fig.text(0.78, 0.91, f"${latest['close']:,.2f}  |  RSI {latest['rsi']}  |  {latest['datetime']}",
             fontsize=9, color=c_muted, fontfamily='monospace')

    # Panel 1: Price + signals
    ax1 = fig.add_subplot(3, 1, (1, 2))
    ax1.set_facecolor(c_bg)
    ax1.plot(dates, prices, color=c_price, linewidth=1.2, alpha=0.9, label='Price')
    ax1.scatter(buy_dates, buy_prices, color=c_buy, marker='^', s=60, zorder=5,
                label=f'BUY ({n_buy})', alpha=0.85, edgecolors='white', linewidth=0.3)
    ax1.scatter(sell_dates, sell_prices, color=c_sell, marker='v', s=60, zorder=5,
                label=f'SELL ({n_sell})', alpha=0.85, edgecolors='white', linewidth=0.3)
    ax1.set_ylabel('Price ($)', color=c_muted, fontsize=10)
    ax1.tick_params(colors=c_muted, labelsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // (24 * 8))))
    ax1.grid(True, alpha=0.15, color=c_grid)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(c_grid)
    ax1.spines['bottom'].set_color(c_grid)
    ax1.legend(loc='upper left', fontsize=9, facecolor=c_card, edgecolor=c_grid,
               labelcolor=c_text, framealpha=0.9)
    ax1.set_xticklabels([])

    # Panel 2: Portfolio vs Hold
    ax2 = fig.add_subplot(3, 1, 3)
    ax2.set_facecolor(c_bg)
    ax2.plot(dates, portfolio, color=c_buy, linewidth=2, label=f'Strategy (${portfolio[-1]:,.0f})')
    ax2.plot(dates, hold, color=c_muted, linewidth=1.5, linestyle='--', label=f'Buy & Hold (${hold[-1]:,.0f})')
    ax2.axhline(y=1000, color=c_grid, linewidth=0.8, linestyle=':')
    ax2.fill_between(dates, portfolio, hold, where=[p > h for p, h in zip(portfolio, hold)],
                     alpha=0.08, color=c_buy)
    ax2.fill_between(dates, portfolio, hold, where=[p <= h for p, h in zip(portfolio, hold)],
                     alpha=0.08, color=c_sell)
    ax2.set_ylabel('Portfolio ($)', color=c_muted, fontsize=10)
    ax2.set_xlabel('', fontsize=10, color=c_muted)
    ax2.tick_params(colors=c_muted, labelsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // (24 * 8))))
    ax2.grid(True, alpha=0.15, color=c_grid)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(c_grid)
    ax2.spines['bottom'].set_color(c_grid)
    ax2.legend(loc='upper left', fontsize=9, facecolor=c_card, edgecolor=c_grid,
               labelcolor=c_text, framealpha=0.9)

    # Stats text in top-right of panel 2
    stats_text = (
        f"Strategy: {'+' if strategy_return > 0 else ''}{strategy_return:.1f}%\n"
        f"Buy&Hold: {'+' if hold_return > 0 else ''}{hold_return:.1f}%\n"
        f"Alpha: {'+' if alpha > 0 else ''}{alpha:.1f}%\n"
        f"Accuracy: {accuracy:.1f}% ({correct}/{with_actual})\n"
        f"Signals: {n_buy}B / {n_sell}S / {n_hold}H"
    )
    ax2.text(0.98, 0.95, stats_text, transform=ax2.transAxes, fontsize=9,
             fontfamily='monospace', color=c_text, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=c_card, edgecolor=c_grid, alpha=0.9))

    # Save
    os.makedirs(CHARTS_DIR, exist_ok=True)
    filename = f'{CHARTS_DIR}/{asset_name}_backtest.png'
    fig.savefig(filename, dpi=150, facecolor=c_bg, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart saved: {filename}")

    # Try to open (Windows)
    if sys.platform == 'win32':
        try:
            os.startfile(filename)
        except Exception:
            pass

    return filename


# ============================================================
# DIAGNOSTIC (CPU parallel)
# ============================================================
def _eval_one_config(features_np, labels_np, closes_np, combo, window, n, step, model_factories, pred_horizon=4):
    import os, warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    min_start = window + 50
    if n < min_start + 50:
        return None
    correct = 0
    total = 0

    # Portfolio simulation: all-in when BUY, all-out when SELL
    portfolio = 1.0       # normalized starting capital
    in_position = False
    entry_price = 0
    trades = 0
    wins = 0
    peak = 1.0
    max_dd = 0.0
    total_gain = 0.0      # sum of winning trade returns
    total_loss = 0.0      # sum of losing trade returns
    trade_returns = []    # kept for potential future use

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

        # Portfolio logic
        price = closes_np[i]
        if ensemble_pred == 1 and not in_position:
            # BUY -- pay fee
            in_position = True
            entry_price = price * (1 + TRADING_FEE)  # effective entry = price + fee
        elif ensemble_pred == 0 and in_position:
            # SELL -- pay fee
            sell_price = price * (1 - TRADING_FEE)  # effective exit = price - fee
            trade_return = (sell_price - entry_price) / entry_price
            portfolio *= (1 + trade_return)
            trades += 1
            if trade_return > 0:
                wins += 1
                total_gain += trade_return
            else:
                total_loss += trade_return
            trade_returns.append(trade_return)
            in_position = False

        # Max drawdown tracking
        current_val = portfolio * (price / entry_price) if in_position else portfolio
        if current_val > peak:
            peak = current_val
        dd = (peak - current_val) / peak
        if dd > max_dd:
            max_dd = dd

    # Close open position at end (with fee)
    if in_position and total > 0:
        last_price = closes_np[n - 1]
        sell_price = last_price * (1 - TRADING_FEE)
        trade_return = (sell_price - entry_price) / entry_price
        portfolio *= (1 + trade_return)
        trade_returns.append(trade_return)
        trades += 1
        if trade_return > 0:
            wins += 1

    if total == 0:
        return None

    accuracy = correct / total
    cum_return = (portfolio - 1.0) * 100  # percentage return
    win_rate = (wins / trades * 100) if trades > 0 else 0
    avg_gain = (total_gain / wins * 100) if wins > 0 else 0
    avg_loss = (total_loss / (trades - wins) * 100) if (trades - wins) > 0 else 0

    # V5 scoring: acc x (1 + return/100)
    # Directly optimises for accuracy AND return together.
    # Calmar/Sharpe removed -- they penalise active trading and drawdowns,
    # biasing toward low-trade-count configs. This formula rewards configs
    # that are both right often AND make the most money.
    combined_score = accuracy * (1 + max(cum_return, 0) / 100)

    return ('+'.join(combo), window, accuracy, total, cum_return, win_rate,
            trades, avg_gain, avg_loss, max_dd * 100, combined_score)


DIAG_MODELS = get_diagnostic_models()


def run_diagnostic_for_asset(asset_name, df_features, feature_cols):
    """Run diagnostic and return best config + all results."""
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
    print(f"  Running... (no output until complete)")

    n = len(df_features)
    features_np = df_features[feature_cols].values.astype(np.float64)
    labels_np = df_features['label'].values.astype(np.int32)
    closes_np = df_features['close'].values.astype(np.float64)

    t_diag = time.time()
    with _suppress_stderr():
        all_results = Parallel(n_jobs=N_JOBS_PARALLEL, verbose=0)(
            delayed(_eval_one_config)(
                features_np, labels_np, closes_np, combo, window, n, DIAG_STEP, DIAG_MODELS
            )
            for combo, window in all_configs
        )
    print(f"  Diagnostic completed in {(time.time() - t_diag)/60:.1f} minutes")

    best_score = 0
    best_config = None
    sorted_results = []

    for result in all_results:
        if result is None:
            continue
        combo_name, window, acc, n_total, cum_ret, win_rate, trades, avg_gain, avg_loss, max_dd, combined_score = result
        sorted_results.append(result)
        if combined_score > best_score:
            best_score = combined_score
            best_config = {
                'coin': asset_name,
                'best_window': window,
                'best_combo': combo_name,
                'accuracy': round(acc * 100, 2),
                'models': combo_name,
                'return_pct': round(cum_ret, 2),
                'win_rate': round(win_rate, 1),
                'trades': trades,
                'combined_score': round(combined_score, 4),
            }
        print(f"    w={window:4d}h | {combo_name:20s} | acc={acc*100:5.1f}% "
              f"ret={cum_ret:+6.1f}% win={win_rate:4.0f}% score={combined_score:.3f} (n={n_total})"
              f"{'  <-- BEST' if combined_score == best_score else ''}")

    # ========================================================
    # CLEAR BEST MODEL RECOMMENDATION
    # ========================================================
    if best_config:
        # Top 5 by combined score
        sorted_results.sort(key=lambda x: -x[10])  # index 10 = combined_score
        top5 = sorted_results[:5]

        print()
        print("  " + "=" * 76)
        print(f"  |{'':4s}BEST MODEL for {asset_name:6s}{'':48s}|")
        print("  " + "=" * 76)
        print(f"  |{'':4s}Models:   {best_config['best_combo']:60s}|")
        w_str = f"{best_config['best_window']}h"
        print(f"  |{'':4s}Window:   {w_str:60s}|")
        acc_str = f"{best_config['accuracy']:.1f}%"
        print(f"  |{'':4s}Accuracy: {acc_str:60s}|")
        ret_str = f"{best_config['return_pct']:+.1f}% (after 0.09% fees)"
        print(f"  |{'':4s}Return:   {ret_str:60s}|")
        wr_str = f"{best_config['win_rate']:.0f}% ({best_config['trades']} trades)"
        print(f"  |{'':4s}Win Rate: {wr_str:60s}|")
        sc_str = f"{best_config['combined_score']:.4f}  (acc x (1 + return/100))"
        print(f"  |{''!s:4s}Score:    {sc_str:60s}|")
        print("  " + "-" * 76)
        print(f"  |{''!s:4s}{'Rank':5s}{'Combo':22s}{'Window':8s}{'Acc':7s}{'Return':9s}{'Score':8s}  |")
        print("  " + "-" * 76)
        for rank, result in enumerate(top5, 1):
            combo_name, window, acc, n_total, cum_ret, win_rate, trades, avg_gain, avg_loss, max_dd, combined_score = result
            marker = " <--" if rank == 1 else ""
            print(f"  |{''!s:4s}{rank:<5d}{combo_name:22s}{window:5d}h  {acc*100:5.1f}%"
                  f"  {cum_ret:+6.1f}%  {combined_score:.3f}{marker:>4s} |")
        print("  " + "=" * 76)
        print(f"\n  >>> USE: models={best_config['best_combo']}, window={best_config['best_window']}h")
        print(f"  >>> Score = acc x (1 + return/100)")
        print()

    return best_config


# ============================================================
# CHART DATA EXPORT
# ============================================================
def generate_strategy_html(asset_name, signals_4h, signals_8h, strategy='both_agree'):
    """
    Generate interactive HTML charts (Plotly) with 4 panels:
    1. Price + 4h signals (blue=BUY, red=SELL)
    2. Price + 8h signals
    3. Price + combined strategy signals
    4. Portfolio equity ($1000 start) vs buy & hold
    Zoom synced across all panels. Colorblind-friendly.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)

    sig4_map = {s['datetime']: s for s in (signals_4h or [])}
    sig8_map = {s['datetime']: s for s in (signals_8h or [])}
    all_times = sorted(set(list(sig4_map.keys()) + list(sig8_map.keys())))

    if not all_times:
        print("  No signals to chart.")
        return

    merged = []
    for dt in all_times:
        s4 = sig4_map.get(dt)
        s8 = sig8_map.get(dt)
        price = (s4 or s8)['close']
        sig4 = s4['signal'] if s4 else 'HOLD'
        conf4 = s4['confidence'] if s4 else 50
        sig8 = s8['signal'] if s8 else 'HOLD'
        conf8 = s8['confidence'] if s8 else 50

        # Combined signal based on strategy
        if sig4 == 'SELL' or sig8 == 'SELL':
            combined = 'SELL'
        elif strategy == 'both_agree':
            if sig4 == 'BUY' and sig8 == 'BUY' and conf4 >= MIN_CONFIDENCE and conf8 >= MIN_CONFIDENCE:
                combined = 'BUY'
            else:
                combined = 'HOLD'
        else:  # either
            if (sig4 == 'BUY' and conf4 >= MIN_CONFIDENCE) or (sig8 == 'BUY' and conf8 >= MIN_CONFIDENCE):
                combined = 'BUY'
            else:
                combined = 'HOLD'

        merged.append({
            'datetime': dt, 'close': price,
            'sig4': sig4, 'conf4': conf4,
            'sig8': sig8, 'conf8': conf8,
            'combined': combined,
        })

    # Overall stats
    o_cash, o_held, o_in, o_entry, o_trades, o_wins = 1000.0, 0.0, False, 0, 0, 0
    o_start_px = merged[0]['close'] if merged else 1.0
    for m in merged:
        price = m['close']
        if m['combined'] == 'BUY' and not o_in:
            o_held = o_cash * (1 - TRADING_FEE) / price; o_cash = 0; o_in = True; o_entry = price; o_trades += 1
        elif m['combined'] == 'SELL' and o_in:
            o_cash = o_held * price * (1 - TRADING_FEE)
            if price > o_entry: o_wins += 1
            o_held = 0; o_in = False
    # Close open position at last price for overall stats
    if o_in and merged:
        last_px = merged[-1]['close']
        o_cash = o_held * last_px * (1 - TRADING_FEE)
    o_strat_ret = (o_cash / 1000.0 - 1) * 100
    o_bh_ret = (merged[-1]['close'] / o_start_px - 1) * 100 if merged else 0
    o_alpha = o_strat_ret - o_bh_ret

    for label, hours in [('1month', 720), ('1week', 168)]:
        data = merged[-hours:] if len(merged) >= hours else merged
        if not data:
            continue

        # Recalculate portfolio from start of this period at $1000
        p_cash, p_held, p_in, p_entry = 1000.0, 0.0, False, 0
        p_start = data[0]['close']
        p_trades, p_wins = 0, 0
        for d in data:
            price = d['close']
            d['buy_hold'] = 1000 * (price / p_start)
            if d['combined'] == 'BUY' and not p_in:
                p_held = p_cash * (1 - TRADING_FEE) / price; p_cash = 0; p_in = True; p_entry = price; p_trades += 1
            elif d['combined'] == 'SELL' and p_in:
                p_cash = p_held * price * (1 - TRADING_FEE)
                if price > p_entry: p_wins += 1
                p_held = 0; p_in = False
            d['portfolio'] = p_cash + p_held * price if p_in else p_cash

        dates = [d['datetime'] for d in data]
        prices = [d['close'] for d in data]
        portfolios = [round(d['portfolio'], 2) for d in data]
        buy_holds = [round(d['buy_hold'], 2) for d in data]

        def _markers(data, sig_key, conf_key):
            bx, by, bt, sx, sy, st = [], [], [], [], [], []
            for d in data:
                if d[sig_key] == 'BUY':
                    bx.append(d['datetime']); by.append(d['close']); bt.append(f"BUY {d[conf_key]:.0f}%")
                elif d[sig_key] == 'SELL':
                    sx.append(d['datetime']); sy.append(d['close']); st.append(f"SELL {d[conf_key]:.0f}%")
            return bx, by, bt, sx, sy, st

        b4x, b4y, b4t, s4x, s4y, s4t = _markers(data, 'sig4', 'conf4')
        b8x, b8y, b8t, s8x, s8y, s8t = _markers(data, 'sig8', 'conf8')
        cbx, cby, _, csx, csy, _ = _markers(data, 'combined', 'conf4')
        cbt = ['STRATEGY BUY'] * len(cbx)
        cst = ['STRATEGY SELL'] * len(csx)

        strat_ret = (portfolios[-1] / 1000 - 1) * 100
        bh_ret = (buy_holds[-1] / 1000 - 1) * 100
        alpha = strat_ret - bh_ret
        win_rate = (p_wins / p_trades * 100) if p_trades > 0 else 0
        price_fmt = ',.2f' if prices[0] >= 100 else ',.4f'

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{asset_name} Strategy -- {label}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0b0f19; color: #c8cdd5; font-family: 'Segoe UI', -apple-system, sans-serif; }}
  .header {{ padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1a1f2e; }}
  .header h1 {{ font-size: 22px; color: #fff; font-weight: 600; }}
  .header .sub {{ color: #6b7280; font-size: 12px; margin-top: 2px; }}
  .stats {{ display: flex; gap: 24px; padding: 12px 24px; border-bottom: 1px solid #1a1f2e; }}
  .stat {{ text-align: center; min-width: 70px; }}
  .stat .val {{ font-size: 20px; font-weight: 700; }}
  .stat .lbl {{ font-size: 10px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }}
  .blu {{ color: #3b82f6; }}
  .red {{ color: #ef4444; }}
  .chart {{ width: 100%; }}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>{asset_name} -- {label.replace('1', '1 ').title()}</h1>
    <div class="sub">Strategy: {strategy} | Min conf: {MIN_CONFIDENCE}% | 0.09% fees</div>
  </div>
  <div style="text-align:right">
    <div class="sub">${prices[-1]:{price_fmt}} | {dates[-1]}</div>
  </div>
</div>
<div class="stats">
  <div class="stat"><div class="val {'blu' if strat_ret > 0 else 'red'}">{strat_ret:+.1f}%</div><div class="lbl">Strategy</div></div>
  <div class="stat"><div class="val {'blu' if bh_ret > 0 else 'red'}">{bh_ret:+.1f}%</div><div class="lbl">Buy & Hold</div></div>
  <div class="stat"><div class="val {'blu' if alpha > 0 else 'red'}">{alpha:+.1f}%</div><div class="lbl">Alpha</div></div>
  <div class="stat"><div class="val">{p_trades}</div><div class="lbl">Trades</div></div>
  <div class="stat"><div class="val">{win_rate:.0f}%</div><div class="lbl">Win Rate</div></div>
</div>
<div id="c4h" class="chart"></div>
<div id="c8h" class="chart"></div>
<div id="cComb" class="chart"></div>
<div id="cPort" class="chart"></div>
<script>
var bg='#0f1318',paper='#0b0f19',grid='#161b26',fc='#6b7280',blue='#3b82f6',red='#ef4444',gray='#94a3b8';
var dates={json.dumps(dates)};
var prices={json.dumps(prices)};
var port={json.dumps(portfolios)};
var bh={json.dumps(buy_holds)};

function ml(t,h){{return{{title:{{text:t,font:{{color:'#94a3b8',size:13}},x:0.01}},paper_bgcolor:paper,plot_bgcolor:bg,font:{{color:fc,size:10}},xaxis:{{gridcolor:grid,tickfont:{{size:9}}}},yaxis:{{gridcolor:grid,tickformat:'{price_fmt}',side:'right'}},legend:{{x:0,y:1.12,orientation:'h',font:{{size:10}},bgcolor:'rgba(0,0,0,0)'}},height:h,margin:{{l:10,r:60,t:40,b:30}},hovermode:'x unified'}}}}
var pl={{x:dates,y:prices,type:'scatter',mode:'lines',name:'Price',line:{{color:gray,width:1.2}}}};

Plotly.newPlot('c4h',[pl,
  {{x:{json.dumps(b4x)},y:{json.dumps(b4y)},mode:'markers',name:'4h BUY',text:{json.dumps(b4t)},marker:{{color:blue,symbol:'triangle-up',size:8}}}},
  {{x:{json.dumps(s4x)},y:{json.dumps(s4y)},mode:'markers',name:'4h SELL',text:{json.dumps(s4t)},marker:{{color:red,symbol:'triangle-down',size:8}}}}
],ml('4h Model',240),{{responsive:true}});

Plotly.newPlot('c8h',[pl,
  {{x:{json.dumps(b8x)},y:{json.dumps(b8y)},mode:'markers',name:'8h BUY',text:{json.dumps(b8t)},marker:{{color:blue,symbol:'triangle-up',size:8}}}},
  {{x:{json.dumps(s8x)},y:{json.dumps(s8y)},mode:'markers',name:'8h SELL',text:{json.dumps(s8t)},marker:{{color:red,symbol:'triangle-down',size:8}}}}
],ml('8h Model',240),{{responsive:true}});

Plotly.newPlot('cComb',[pl,
  {{x:{json.dumps(cbx)},y:{json.dumps(cby)},mode:'markers',name:'BUY',text:{json.dumps(cbt)},marker:{{color:blue,symbol:'triangle-up',size:10,line:{{width:1,color:'#fff'}}}}}},
  {{x:{json.dumps(csx)},y:{json.dumps(csy)},mode:'markers',name:'SELL',text:{json.dumps(cst)},marker:{{color:red,symbol:'triangle-down',size:10,line:{{width:1,color:'#fff'}}}}}}
],ml('Combined Strategy ({strategy})',240),{{responsive:true}});

var pl2=ml('Portfolio ($1,000 start)',240);pl2.yaxis.tickformat=',.0f';
Plotly.newPlot('cPort',[
  {{x:dates,y:port,type:'scatter',mode:'lines',name:'Strategy ($'+port[port.length-1].toLocaleString()+')',line:{{color:blue,width:2}},fill:'tozeroy',fillcolor:'rgba(59,130,246,0.05)'}},
  {{x:dates,y:bh,type:'scatter',mode:'lines',name:'Buy&Hold ($'+bh[bh.length-1].toLocaleString()+')',line:{{color:'#6b7280',width:1.5,dash:'dash'}}}},
  {{x:[dates[0],dates[dates.length-1]],y:[1000,1000],type:'scatter',mode:'lines',name:'$1k',line:{{color:'#333',width:1,dash:'dot'}},showlegend:false}}
],pl2,{{responsive:true}});

['c4h','c8h','cComb','cPort'].forEach(function(id){{
  document.getElementById(id).on('plotly_relayout',function(ed){{
    if(ed['xaxis.range[0]']&&ed['xaxis.range[1]']){{
      var r=[ed['xaxis.range[0]'],ed['xaxis.range[1]']];
      ['c4h','c8h','cComb','cPort'].forEach(function(o){{if(o!==id)Plotly.relayout(o,{{'xaxis.range':r}})}});
    }}
    if(ed['xaxis.autorange']){{
      ['c4h','c8h','cComb','cPort'].forEach(function(o){{if(o!==id)Plotly.relayout(o,{{'xaxis.autorange':true}})}});
    }}
  }});
}});
</script>
</body>
</html>"""

        filename = f'{CHARTS_DIR}/{asset_name}_strategy_{label}.html'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"  Interactive chart: {filename}")

    print(f"  Strategy: {strategy} | Trades: {o_trades} | Win rate: {(o_wins/o_trades*100) if o_trades else 0:.0f}% | Alpha: {o_alpha:+.1f}%")


def generate_signal_table_html(asset_name, signals_4h, signals_8h, strategy='both_agree'):
    """Generate interactive HTML table: price, price+1h, delta, signals, strategy, correct."""
    os.makedirs(CHARTS_DIR, exist_ok=True)

    sig4_map = {s['datetime']: s for s in (signals_4h or [])}
    sig8_map = {s['datetime']: s for s in (signals_8h or [])}
    all_times = sorted(set(list(sig4_map.keys()) + list(sig8_map.keys())))
    if not all_times:
        return
    all_times = all_times[-168:]

    rows = []
    for i, dt in enumerate(all_times):
        s4 = sig4_map.get(dt)
        s8 = sig8_map.get(dt)
        price = (s4 or s8)['close']

        # Price +1h = actual price at next timestamp
        if i + 1 < len(all_times):
            next_dt = all_times[i + 1]
            next_s = sig4_map.get(next_dt) or sig8_map.get(next_dt)
            price_next = next_s['close'] if next_s else price
        else:
            price_next = price

        delta = ((price_next - price) / price * 100) if price > 0 else 0

        sig4 = s4['signal'] if s4 else 'N/A'
        conf4 = s4['confidence'] if s4 else 0
        sig8 = s8['signal'] if s8 else 'N/A'
        conf8 = s8['confidence'] if s8 else 0

        # Combined strategy
        if sig4 == 'SELL' or sig8 == 'SELL':
            combined = 'SELL'
        elif strategy == 'both_agree':
            if sig4 == 'BUY' and sig8 == 'BUY' and conf4 >= MIN_CONFIDENCE and conf8 >= MIN_CONFIDENCE:
                combined = 'BUY'
            else:
                combined = 'HOLD'
        else:
            if (sig4 == 'BUY' and conf4 >= MIN_CONFIDENCE) or (sig8 == 'BUY' and conf8 >= MIN_CONFIDENCE):
                combined = 'BUY'
            else:
                combined = 'HOLD'

        # Was the strategy correct? BUY and price went up, SELL and price went down
        if i >= len(all_times) - 1:
            correct = '...'
        elif combined == 'BUY':
            correct = '_' if delta > 0 else '_'
        elif combined == 'SELL':
            correct = '_' if delta <= 0 else '_'
        else:
            correct = '--'

        rows.append({
            'dt': dt, 'price': price, 'price_next': price_next,
            'delta': delta,
            'sig4': sig4, 'conf4': conf4,
            'sig8': sig8, 'conf8': conf8,
            'combined': combined, 'correct': correct,
        })

    # Stats
    actionable = [r for r in rows[:-1] if r['combined'] in ('BUY', 'SELL')]
    correct_count = sum(1 for r in actionable if r['correct'] == '_')
    accuracy = (correct_count / len(actionable) * 100) if actionable else 0
    buy_calls = [r for r in actionable if r['combined'] == 'BUY']
    sell_calls = [r for r in actionable if r['combined'] == 'SELL']
    correct_buys = sum(1 for r in buy_calls if r['correct'] == '_')
    correct_sells = sum(1 for r in sell_calls if r['correct'] == '_')

    price_fmt = ',.2f' if rows[0]['price'] >= 100 else ',.4f'

    # Build table rows
    table_rows = ""
    for r in rows:
        def _sc(sig):
            if sig == 'BUY': return 'buy'
            if sig == 'SELL': return 'sell'
            return 'hold'
        dc = 'buy' if r['delta'] > 0.01 else 'sell' if r['delta'] < -0.01 else 'hold'
        cc = 'right' if r['correct'] == '_' else 'wrong' if r['correct'] == '_' else ''

        table_rows += f"""<tr>
  <td>{r['dt']}</td>
  <td>${r['price']:{price_fmt}}</td>
  <td>${r['price_next']:{price_fmt}}</td>
  <td class="{dc}">{r['delta']:+.2f}%</td>
  <td class="{_sc(r['combined'])}"><b>{r['combined']}</b></td>
  <td class="{cc}">{r['correct']}</td>
  <td class="{_sc(r['sig4'])}">{r['sig4']}</td>
  <td>{r['conf4']:.0f}%</td>
  <td class="{_sc(r['sig8'])}">{r['sig8']}</td>
  <td>{r['conf8']:.0f}%</td>
</tr>
"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{asset_name} Signal Table -- Past Week</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0b0f19; color: #c8cdd5; font-family: 'Segoe UI', monospace; font-size: 13px; }}
  .header {{ padding: 16px 24px; border-bottom: 1px solid #1a1f2e; display: flex; justify-content: space-between; }}
  .header h1 {{ color: #fff; font-size: 20px; }}
  .header .sub {{ color: #6b7280; font-size: 12px; }}
  .stats {{ padding: 10px 24px; display: flex; gap: 30px; border-bottom: 1px solid #1a1f2e; }}
  .stat .val {{ font-size: 18px; font-weight: 700; }}
  .stat .lbl {{ font-size: 10px; color: #6b7280; text-transform: uppercase; }}
  .blu {{ color: #3b82f6; }}
  .red {{ color: #ef4444; }}
  .tbl {{ width: 100%; border-collapse: collapse; }}
  .tbl th {{ position: sticky; top: 0; background: #141820; padding: 8px 10px; text-align: left;
    font-size: 11px; text-transform: uppercase; color: #6b7280; border-bottom: 2px solid #1a1f2e; cursor: pointer; }}
  .tbl th:hover {{ color: #fff; }}
  .tbl td {{ padding: 6px 10px; border-bottom: 1px solid #111620; }}
  .tbl tr:hover {{ background: #141820; }}
  .buy {{ color: #3b82f6; }}
  .sell {{ color: #ef4444; }}
  .hold {{ color: #eab308; }}
  .right {{ color: #3b82f6; font-weight: bold; font-size: 16px; }}
  .wrong {{ color: #ef4444; font-weight: bold; font-size: 16px; }}
  .wrap {{ max-height: calc(100vh - 120px); overflow-y: auto; }}
  .filter {{ padding: 8px 24px; display: flex; gap: 10px; border-bottom: 1px solid #1a1f2e; }}
  .filter button {{ background: #1a1f2e; color: #c8cdd5; border: 1px solid #2a2f3e; padding: 4px 12px;
    border-radius: 4px; cursor: pointer; font-size: 12px; }}
  .filter button:hover, .filter button.active {{ background: #3b82f6; color: #fff; border-color: #3b82f6; }}
  .legend {{ padding: 6px 24px; font-size: 11px; color: #6b7280; border-bottom: 1px solid #1a1f2e; }}
  .legend span {{ margin-right: 16px; }}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>{asset_name} -- Signal Table (Past Week)</h1>
    <div class="sub">Strategy: {strategy} | Min confidence: {MIN_CONFIDENCE}% | _ = (Price+1h _ Price) / Price</div>
  </div>
  <div style="text-align:right">
    <div class="sub">{rows[-1]['dt']}</div>
  </div>
</div>
<div class="stats">
  <div class="stat"><div class="val">{len(actionable)}</div><div class="lbl">Signals</div></div>
  <div class="stat"><div class="val {'blu' if accuracy > 60 else 'red'}">{accuracy:.0f}%</div><div class="lbl">Accuracy</div></div>
  <div class="stat"><div class="val">{len(buy_calls)}</div><div class="lbl">BUY</div></div>
  <div class="stat"><div class="val">{correct_buys}/{len(buy_calls)}</div><div class="lbl">BUY _</div></div>
  <div class="stat"><div class="val">{len(sell_calls)}</div><div class="lbl">SELL</div></div>
  <div class="stat"><div class="val">{correct_sells}/{len(sell_calls)}</div><div class="lbl">SELL _</div></div>
</div>
<div class="legend">
  <span class="buy">_ BUY</span>
  <span class="sell">_ SELL</span>
  <span class="hold">_ HOLD</span>
  <span>_ = price moved in predicted direction next hour</span>
</div>
<div class="filter">
  <button class="active" onclick="filterRows('all')">All</button>
  <button onclick="filterRows('BUY')">BUY only</button>
  <button onclick="filterRows('SELL')">SELL only</button>
  <button onclick="filterRows('action')">BUY + SELL</button>
</div>
<div class="wrap">
<table class="tbl" id="signalTable">
<thead>
<tr>
  <th onclick="sortTable(0)">Time</th>
  <th onclick="sortTable(1)">Price</th>
  <th onclick="sortTable(2)">Price +1h</th>
  <th onclick="sortTable(3)">_ 1h</th>
  <th onclick="sortTable(4)">Strategy</th>
  <th onclick="sortTable(5)">Correct?</th>
  <th onclick="sortTable(6)">4h Signal</th>
  <th onclick="sortTable(7)">4h Conf</th>
  <th onclick="sortTable(8)">8h Signal</th>
  <th onclick="sortTable(9)">8h Conf</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>
<script>
function filterRows(type) {{
  document.querySelectorAll('.filter button').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.tbl tbody tr').forEach(row => {{
    var strat = row.cells[4].textContent.trim();
    if (type === 'all') row.style.display = '';
    else if (type === 'action') row.style.display = (strat === 'BUY' || strat === 'SELL') ? '' : 'none';
    else row.style.display = strat === type ? '' : 'none';
  }});
}}
var sortDir = {{}};
function sortTable(col) {{
  var table = document.getElementById('signalTable');
  var rows = Array.from(table.tBodies[0].rows);
  sortDir[col] = !sortDir[col];
  rows.sort(function(a, b) {{
    var va = a.cells[col].textContent.trim();
    var vb = b.cells[col].textContent.trim();
    var na = parseFloat(va.replace(/[$,%+]/g, ''));
    var nb = parseFloat(vb.replace(/[$,%+]/g, ''));
    if (!isNaN(na) && !isNaN(nb)) return sortDir[col] ? na - nb : nb - na;
    return sortDir[col] ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => table.tBodies[0].appendChild(r));
}}
</script>
</body>
</html>"""

    filename = f'{CHARTS_DIR}/{asset_name}_signal_table.html'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Signal table: {filename}")


# ============================================================
# CHART DATA EXPORT
# ============================================================
def export_chart_data(all_signals, output_file=f'{MODELS_DIR}/crypto_hourly_chart_data.json'):
    chart_data = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'type': 'hourly',
        'prediction_horizon': f'{PREDICTION_HORIZON}h',
        'feature_set': ACTIVE_FEATURE_SET,
        'assets': {}
    }
    for asset_name, signals in all_signals.items():
        chart_data['assets'][asset_name] = signals
    with open(output_file, 'w') as f:
        json.dump(chart_data, f, indent=2)
    print(f"\nChart data saved to {output_file}")
    return output_file


def run_mode_b(assets_list, horizon_filter=None, skip_data_update=False):
    print("\n" + "=" * 60)
    print("  MODE B: QUICK HOURLY RUN (saved best models)")
    print("=" * 60)

    if not os.path.exists(f'{MODELS_DIR}/crypto_hourly_best_models.csv'):
        print("\nERROR: crypto_hourly_best_models.csv not found!")
        print("Please run Mode D first to find best models.")
        return

    df_best = pd.read_csv(f'{MODELS_DIR}/crypto_hourly_best_models.csv')
    if 'horizon' not in df_best.columns:
        df_best['horizon'] = 4  # legacy = 4h

    # Filter by horizon if specified
    if horizon_filter is not None:
        df_best = df_best[df_best['horizon'] == horizon_filter].reset_index(drop=True)
        if df_best.empty:
            print(f"\nERROR: No {horizon_filter}h models found in CSV. Run Mode D with {horizon_filter}h first.")
            return

    print("\nLoaded best models:")
    for _, row in df_best.iterrows():
        fs = row.get('feature_set', '?')
        h = int(row.get('horizon', 4))
        print(f"  {row['coin']:6s} -> {row['best_combo']:20s} | w={row['best_window']:4d}h | {row['accuracy']:.1f}% | Set {fs} | {h}h")

    available_in_csv = set(df_best['coin'].values)
    assets_to_run = [a for a in assets_list if a in available_in_csv]
    missing = [a for a in assets_list if a not in available_in_csv]
    if missing:
        print(f"\nWARNING: No best model for: {', '.join(missing)}")
        print("Run Mode D first for these assets.")

    if not assets_to_run:
        print("No assets to process.")
        return

    # Refresh macro & sentiment data before generating signals
    if not skip_data_update:
        try:
            import download_macro_data
            download_macro_data.main()
        except ImportError:
            print("  WARNING: download_macro_data.py not found -- macro features may be stale.")
        except Exception as e:
            print(f"  WARNING: Macro data update failed: {e}")

        update_all_data(assets_to_run)

    print("\n" + "=" * 60)
    print("  GENERATING SIGNALS & BACKTEST CHARTS")
    print("=" * 60)

    all_signals = {}
    for asset_name in assets_to_run:
        asset_rows = df_best[df_best['coin'] == asset_name]
        for _, row in asset_rows.iterrows():
            model_names = row['models'].split('+')
            window = int(row['best_window'])
            fs = row.get('feature_set', 'A')
            h = int(row.get('horizon', 4))

            # Mode D saves optimal features in the CSV
            if fs in ('D', 'E2', 'E3') and 'optimal_features' in row and pd.notna(row.get('optimal_features', '')):
                feature_override = row['optimal_features'].split(',')
            elif fs == 'B':
                feature_override = list(FEATURE_SET_B)
            else:
                feature_override = list(FEATURE_SET_A)

            model_info = {
                'best_combo': row['best_combo'],
                'best_window': window,
                'accuracy': row['accuracy'],
                'feature_set': fs,
                'horizon': h,
                'n_features': int(row['n_features']) if 'n_features' in row.index and pd.notna(row.get('n_features')) else '',
            }

            signals = generate_signals(asset_name, model_names, window, REPLAY_HOURS,
                                       feature_override=feature_override, horizon=h)
            signals = simulate_portfolio(signals)
            _print_bootstrap_ci(signals, label=f"{asset_name} {h}h")

            label = f"{asset_name}_{h}h"
            all_signals[label] = signals

            chart_suffix = f"_{h}h" if h != 4 else ""
            generate_backtest_chart(f"{asset_name}{chart_suffix}", signals, model_info=model_info)

            if signals:
                latest = signals[-1]
                print(f"\n  >> {asset_name} ({h}h): {latest['signal']} ({latest['confidence']:.0f}%) "
                      f"| price=${latest['close']:,.2f} | Set {fs}")

    export_chart_data(all_signals)

    # Generate interactive HTML charts (1-month + 1-week) for each asset
    # Only when both 4h and 8h are available (or whichever are present)
    if horizon_filter is None:
        print("\n" + "=" * 60)
        print("  GENERATING INTERACTIVE STRATEGY CHARTS")
        print("=" * 60)

        for asset_name in assets_to_run:
            try:
                with open(f'{CONFIG_DIR}/trading_config.json') as _f:
                    _tcfg = json.load(_f)
                strategy = _tcfg.get(asset_name, {}).get('strategy', 'both_agree')
            except Exception:
                strategy = 'both_agree'

            asset_rows = df_best[df_best['coin'] == asset_name]
            signals_4h, signals_8h = None, None

            for _, row in asset_rows.iterrows():
                h = int(row.get('horizon', 4))
                model_names = row['models'].split('+')
                window = int(row['best_window'])
                fs = row.get('feature_set', 'A')
                opt = row.get('optimal_features', '')

                if fs in ('D', 'E2', 'E3') and pd.notna(opt) and str(opt).strip() and str(opt).strip() != 'nan':
                    feature_override = [f.strip() for f in str(opt).split(',') if f.strip() and f.strip() != 'nan']
                elif fs == 'B':
                    feature_override = list(FEATURE_SET_B)
                else:
                    feature_override = list(FEATURE_SET_A)

                print(f"  Generating {h}h signals (720h) for {asset_name}...")
                sigs = generate_signals(asset_name, model_names, window, 720,
                                        feature_override=feature_override, horizon=h)
                sigs = simulate_portfolio(sigs)
                if h == 4:
                    signals_4h = sigs
                else:
                    signals_8h = sigs

            if signals_4h or signals_8h:
                generate_strategy_html(asset_name, signals_4h, signals_8h, strategy=strategy)
                generate_signal_table_html(asset_name, signals_4h, signals_8h, strategy=strategy)

    print("\n" + "=" * 60)
    print("  MODE B COMPLETE")
    print("=" * 60)



# ============================================================
# IMPROVEMENT 3: PERMUTATION SIGNIFICANCE TEST (--permtest flag)
# ============================================================
def _run_permutation_test(asset_name, df, feature_cols, best_config, n_perm=200, horizon=4):
    """
    Shuffle labels N times and re-evaluate the best config each time.
    Builds a null distribution to compute p-value.

    p < 0.05 -> significant edge (strong)
    p < 0.10 -> marginal edge (acceptable)
    p > 0.10 -> warn: may be noise
    p > 0.20 -> hard warn: results consistent with random chance

    Uses LGBM only (fastest model) for speed. ~1-2 min per asset on desktop.
    """
    import warnings
    warnings.filterwarnings('ignore')

    print(f"\n  Permutation significance test ({n_perm} permutations)...")
    print(f"  Asset={asset_name}  window={best_config['best_window']}h  model=LGBM")
    print(f"  Expected time: ~{n_perm * 0.5 / 60:.0f}-{n_perm * 1.0 / 60:.0f} min")

    window    = best_config['best_window']
    feat_list = best_config['optimal_features'].split(',') if ',' in str(best_config.get('optimal_features', '')) else feature_cols
    feat_list = [f for f in feat_list if f in df.columns]
    if not feat_list:
        print("  WARNING: No valid features found -- skipping permutation test")
        return None

    features_np = df[feat_list].values
    closes_np   = df['close'].values if 'close' in df.columns else np.ones(len(df))
    n           = len(df)
    step        = max(DIAG_STEP, 24)

    # Build label from horizon
    label_col = f'label_{horizon}h'
    if label_col not in df.columns:
        label_col = 'label_4h' if 'label_4h' in df.columns else df.columns[-1]
    labels_np = df[label_col].values

    # Quick scorer: LGBM only, returns accuracy
    try:
        from lightgbm import LGBMClassifier
        def _lgbm(): return LGBMClassifier(n_estimators=50, learning_rate=0.1,
                                            num_leaves=15, random_state=42, verbose=-1)
        model_factories = {'LGBM': _lgbm}
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        def _gb(): return GradientBoostingClassifier(n_estimators=50, random_state=42)
        model_factories = {'GB': _gb}

    def _quick_acc(lbl):
        result = _eval_one_config(features_np, lbl, closes_np,
                                  ('LGBM',) if 'LGBM' in model_factories else ('GB',),
                                  window, n, step, model_factories, pred_horizon=horizon)
        return result[2] if result else None  # index 2 = accuracy

    # Real accuracy
    real_acc = _quick_acc(labels_np)
    if real_acc is None:
        print("  WARNING: Could not evaluate real config -- skipping")
        return None

    # Null distribution
    rng = np.random.RandomState(42)
    null_accs = []
    t0 = __import__('time').time()
    for p in range(n_perm):
        shuffled = rng.permutation(labels_np)
        acc = _quick_acc(shuffled)
        if acc is not None:
            null_accs.append(acc)
        if (p + 1) % 50 == 0:
            elapsed = __import__('time').time() - t0
            print(f"    {p+1}/{n_perm} done ({elapsed:.0f}s)...")

    if not null_accs:
        print("  WARNING: No valid null samples -- skipping")
        return None

    p_value = np.mean([a >= real_acc for a in null_accs])
    null_mean = np.mean(null_accs) * 100
    null_std  = np.std(null_accs) * 100

    print(f"\n  -- Permutation Test Result ------------------")
    print(f"  Real accuracy:  {real_acc*100:.1f}%")
    print(f"  Null mean_std:  {null_mean:.1f}% _ {null_std:.1f}%")
    print(f"  p-value:        {p_value:.3f}  ({n_perm} permutations)")

    if p_value < 0.05:
        verdict = "_ SIGNIFICANT  -- strong evidence of real edge (p<0.05)"
    elif p_value < 0.10:
        verdict = "_ MARGINAL     -- borderline edge, monitor closely (p<0.10)"
    elif p_value < 0.20:
        verdict = "_  WEAK        -- results may be noise (p<0.20)"
    else:
        verdict = "_ NOT SIGNIFICANT -- results consistent with random chance (p>0.20)"
    print(f"  Verdict:        {verdict}")
    print(f"  --------------------------------------------")

    return p_value


def run_mode_d(assets_list, diag_years=1, horizon=PREDICTION_HORIZON, permtest=False):
    """
    Complete pipeline from scratch:
    1. Build all ~124 features
    2. Run 5-test feature analysis to find optimal subset
    3. Run 75-config diagnostic with optimal features
    4. Save best models
    5. Generate signals + backtest charts
    """
    t_mode_start = time.time()

    # Kill any orphaned loky workers from previous interrupted runs
    _kill_orphan_workers()

    print("\n" + "=" * 60)
    print(f"  MODE D: FULL PIPELINE -- {horizon}h HORIZON")
    print(f"  Starts from ALL features, finds optimal subset per asset")
    print("=" * 60)

    # Download fresh macro & sentiment data before anything else
    print("\n  Updating macro & sentiment data...")
    t0 = time.time()
    try:
        import download_macro_data
        download_macro_data.main()
    except ImportError:
        print("  WARNING: download_macro_data.py not found -- macro features may be stale.")
    except Exception as e:
        print(f"  WARNING: Macro data update failed: {e}")
    print(f"  [Macro update: {(time.time()-t0)/60:.1f} min]")

    t0 = time.time()
    update_all_data(assets_list)
    print(f"  [Data update: {(time.time()-t0)/60:.1f} min]")

    diag_hours = diag_years * 365 * 24
    best_models = []

    for asset_name in assets_list:
        t_asset = time.time()
        print(f"\n{'='*60}")
        print(f"  FULL PIPELINE: {asset_name} ({horizon}h horizon)")
        print(f"{'='*60}")

        df_raw = load_data(asset_name)
        if df_raw is None:
            continue

        # Step 1: Build ALL features
        print(f"\n  Building all features (horizon={horizon}h)...")
        t0 = time.time()
        df_full, all_cols = build_all_features(df_raw, asset_name=asset_name, horizon=horizon)
        print(f"  [Feature build: {(time.time()-t0)/60:.1f} min]")

        total_rows = len(df_full)
        if total_rows > diag_hours:
            df_full = df_full.tail(diag_hours).reset_index(drop=True)
            print(f"  Trimmed: {total_rows:,} -> {len(df_full):,} rows (last {diag_years}y)")

        # Drop NaN for analysis
        df_clean = df_full.dropna(subset=all_cols + ['label']).reset_index(drop=True)
        print(f"  Clean data: {len(df_clean):,} rows, {len(all_cols)} features")

        if len(df_clean) < 500:
            print(f"  Not enough data ({len(df_clean)} rows). Need 500+. Skipping.")
            continue

        # Step 2: Feature analysis (5 tests -> optimal subset)
        t0 = time.time()
        optimal_features = run_feature_analysis(asset_name, df_clean, all_cols)
        print(f"  [Feature analysis total: {(time.time()-t0)/60:.1f} min]")

        if not optimal_features or len(optimal_features) < 3:
            print(f"  Feature analysis produced too few features ({len(optimal_features or [])}). Skipping.")
            continue

        # Step 3: Run diagnostic with optimal features
        print(f"\n{'='*60}")
        print(f"  DIAGNOSTIC: {asset_name} ({len(optimal_features)} optimal features)")
        print(f"{'='*60}")

        df_diag = df_clean.dropna(subset=optimal_features + ['label']).reset_index(drop=True)
        print(f"  {len(df_diag):,} rows, {len(optimal_features)} features")

        t0 = time.time()
        best_config = run_diagnostic_for_asset(asset_name, df_diag, optimal_features)
        print(f"  [Diagnostic: {(time.time()-t0)/60:.1f} min]")

        if best_config:
            best_config['feature_set'] = 'D'
            best_config['n_features'] = len(optimal_features)
            best_config['optimal_features'] = ','.join(optimal_features)
            best_config['horizon'] = horizon
            best_models.append(best_config)

            # -- Improvement 3: Permutation test (only if --permtest flag passed) --
            if permtest:
                t0 = time.time()
                _run_permutation_test(asset_name, df_diag, optimal_features,
                                      best_config, n_perm=200, horizon=horizon)
                print(f"  [Permutation test: {(time.time()-t0)/60:.1f} min]")

        print(f"  [{asset_name} total: {(time.time()-t_asset)/60:.1f} min]")

    if not best_models:
        print("\nNo results. Aborting.")
        return

    # Save best models (merge with existing horizons)
    csv_path = f'{MODELS_DIR}/crypto_hourly_best_models.csv'
    df_best = pd.DataFrame(best_models)
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        if 'horizon' not in df_existing.columns:
            df_existing['horizon'] = 4
        for m in best_models:
            mask = (df_existing['coin'] == m['coin']) & (df_existing['horizon'] == horizon)
            df_existing = df_existing[~mask]
        df_best = pd.concat([df_existing, df_best], ignore_index=True)
    df_best.to_csv(csv_path, index=False)
    df_best.to_csv(f'{MODELS_DIR}/crypto_hourly_best_models_mode_d.csv', index=False)

    print(f"\n{'='*60}")
    print(f"  BEST MODELS SAVED -- {horizon}h HORIZON")
    print(f"{'='*60}")
    for row in best_models:
        print(f"  {row['coin']:6s} -> {row['best_combo']:20s} | w={row['best_window']:4d}h | "
              f"{row['accuracy']:.1f}% | {row['n_features']} features | {horizon}h")
    print(f"{'='*60}")

    # Step 4: Generate signals + charts
    print("\n" + "=" * 60)
    print("  GENERATING SIGNALS & BACKTEST CHARTS")
    print("=" * 60)

    t0 = time.time()
    all_signals = {}
    for config in best_models:
        asset_name = config['coin']
        model_names = config['models'].split('+')
        window = config['best_window']
        feature_override = config['optimal_features'].split(',')

        signals = generate_signals(asset_name, model_names, window, REPLAY_HOURS,
                                   feature_override=feature_override, horizon=horizon)
        signals = simulate_portfolio(signals)
        _print_bootstrap_ci(signals, label=f"{asset_name} {horizon}h")
        all_signals[asset_name] = signals

        generate_backtest_chart(asset_name, signals, model_info=config)

        if signals:
            latest = signals[-1]
            print(f"\n  >> {asset_name} ({horizon}h): {latest['signal']} ({latest['confidence']:.0f}%) "
                  f"| price=${latest['close']:,.2f}")

    export_chart_data(all_signals)
    print(f"  [Signal generation: {(time.time()-t0)/60:.1f} min]")

    elapsed_total = time.time() - t_mode_start
    print("\n" + "=" * 60)
    print(f"  MODE D COMPLETE (full pipeline) -- {elapsed_total/60:.1f} min total")
    print("=" * 60)


# ============================================================
# MODE E: ITERATIVE REFINEMENT (2nd/3rd pass on Mode D results)
# ============================================================
def _load_mode_d_config(asset_name, horizon):
    """Load the Mode D (or previous E) result for a given asset/horizon."""
    csv_path = f'{MODELS_DIR}/crypto_hourly_best_models.csv'
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if 'horizon' not in df.columns:
        df['horizon'] = 4
    mask = (df['coin'] == asset_name) & (df['horizon'] == horizon)
    match = df[mask]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        'coin': row['coin'],
        'models': row['models'],
        'best_combo': row['best_combo'],
        'best_window': int(row['best_window']),
        'accuracy': row['accuracy'],
        'feature_set': row.get('feature_set', 'D'),
        'optimal_features': str(row.get('optimal_features', '')),
        'horizon': int(row.get('horizon', 4)),
    }


def _run_iteration_2(asset_name, df_features, prev_config, all_cols, horizon):
    """
    Iteration 2: Refine features + finer window grid.
    1. Leave-one-out: drop each feature, keep only those that help
    2. Add-back: try each dropped feature, add if it helps
    3. Expanded window grid around winner (step=25h)
    4. Finer DIAG_STEP=24
    """
    prev_features = prev_config['optimal_features'].split(',')
    prev_features = [f for f in prev_features if f in all_cols]
    prev_window = prev_config['best_window']
    prev_acc = prev_config['accuracy']

    print(f"\n  Starting from: {len(prev_features)} features, w={prev_window}h, {prev_acc:.1f}%")

    # --- Step 1: Leave-One-Out Refinement ---
    print(f"\n  [ITER2 1/3] Leave-One-Out Refinement (finer step=24)...")
    baseline_acc, _ = _quick_accuracy(df_features, prev_features, window=prev_window, step=24)
    print(f"    Baseline: {baseline_acc:.1f}% ({len(prev_features)} features)")

    features_to_drop = []
    for feat in prev_features:
        reduced = [f for f in prev_features if f != feat]
        acc, _ = _quick_accuracy(df_features, reduced, window=prev_window, step=24)
        change = acc - baseline_acc
        if change > 0.3:
            features_to_drop.append(feat)
            print(f"    Drop {feat:30s} -> {acc:5.1f}% ({change:+.1f}%) ** REMOVING")
        elif change < -0.5:
            print(f"    Drop {feat:30s} -> {acc:5.1f}% ({change:+.1f}%) ** ESSENTIAL")

    refined_features = [f for f in prev_features if f not in features_to_drop]
    if features_to_drop:
        new_acc, _ = _quick_accuracy(df_features, refined_features, window=prev_window, step=24)
        print(f"\n    Removed {len(features_to_drop)}: {', '.join(features_to_drop)}")
        print(f"    {len(prev_features)} -> {len(refined_features)} features | {baseline_acc:.1f}% -> {new_acc:.1f}%")
        if new_acc < baseline_acc - 0.5:
            print(f"    Refinement hurt accuracy -- reverting.")
            refined_features = list(prev_features)
    else:
        print(f"    No features to drop.")

    # --- Step 2: Add-Back Test ---
    print(f"\n  [ITER2 2/3] Add-Back Test (try adding dropped features)...")
    dropped_features = [f for f in all_cols if f not in refined_features]
    current_acc, _ = _quick_accuracy(df_features, refined_features, window=prev_window, step=24)
    added = []

    # Only test features that were in the top 50% by LGBM importance (speed optimization)
    from lightgbm import LGBMClassifier
    model_imp = LGBMClassifier(n_estimators=200, max_depth=6, verbose=-1,
                                device=LGBM_DEVICE, random_state=42)
    df_clean = df_features.dropna(subset=all_cols + ['label'])
    X = df_clean[all_cols].tail(5000)
    y = df_clean['label'].tail(5000)
    if len(np.unique(y)) >= 2:
        model_imp.fit(X, y)
        imp = pd.Series(model_imp.feature_importances_, index=all_cols)
        imp = imp.sort_values(ascending=False)
        candidates = [f for f in imp.index[:len(all_cols)//2] if f not in refined_features]
    else:
        candidates = dropped_features[:30]

    print(f"    Testing {len(candidates)} candidate features...")
    for feat in candidates:
        test_set = refined_features + [feat]
        test_df = df_features.dropna(subset=test_set + ['label'])
        if len(test_df) < 500:
            continue
        acc, _ = _quick_accuracy(test_df, test_set, window=prev_window, step=24)
        if acc > current_acc + 0.5:
            delta = acc - current_acc
            added.append(feat)
            refined_features.append(feat)
            current_acc = acc
            print(f"    Add {feat:30s} -> {acc:5.1f}% (+{delta:.1f}%) ** ADDING")

    if added:
        print(f"\n    Added {len(added)} features: {', '.join(added)}")
    else:
        print(f"    No features improved accuracy.")

    print(f"\n    Final feature count: {len(refined_features)}")

    # --- Step 3: Expanded Window Grid ---
    print(f"\n  [ITER2 3/3] Expanded Window Search (around w={prev_window}h)...")

    # Build fine grid around winner: _100h in steps of 25h
    fine_windows = sorted(set(
        [max(48, prev_window + offset) for offset in range(-100, 125, 25)]
    ))
    print(f"    Testing windows: {fine_windows}")

    # Run diagnostic with refined features + fine windows
    # Temporarily override DIAG_WINDOWS and DIAG_STEP
    global DIAG_WINDOWS, DIAG_STEP
    orig_windows = DIAG_WINDOWS
    orig_step = DIAG_STEP
    DIAG_WINDOWS = fine_windows
    DIAG_STEP = 24

    df_diag = df_features.dropna(subset=refined_features + ['label']).reset_index(drop=True)
    print(f"    {len(df_diag):,} rows, {len(refined_features)} features, step=24")

    best_config = run_diagnostic_for_asset(asset_name, df_diag, refined_features)

    DIAG_WINDOWS = orig_windows
    DIAG_STEP = orig_step

    if best_config:
        best_config['feature_set'] = 'E2'
        best_config['n_features'] = len(refined_features)
        best_config['optimal_features'] = ','.join(refined_features)
        best_config['horizon'] = horizon
        best_config['iteration'] = 2
        print(f"\n  ITERATION 2 RESULT: {best_config['best_combo']} | w={best_config['best_window']}h | "
              f"{best_config['accuracy']:.1f}% | {len(refined_features)} features")
        print(f"  vs PREVIOUS:        {prev_config['best_combo']} | w={prev_config['best_window']}h | "
              f"{prev_config['accuracy']:.1f}%")
        improvement = best_config['accuracy'] - prev_acc
        print(f"  Improvement: {improvement:+.1f}%")

    return best_config, refined_features


def _run_iteration_3(asset_name, df_features, prev_config, all_cols, horizon):
    """
    Iteration 3: Ultra-fine tuning.
    1. Very fine window grid (_40h, step=10h) around winner
    2. DIAG_STEP=12 (most granular)
    3. Feature interaction test: products of top-5 features
    """
    prev_features = prev_config['optimal_features'].split(',')
    prev_features = [f for f in prev_features if f in all_cols]
    prev_window = prev_config['best_window']
    prev_acc = prev_config['accuracy']

    print(f"\n  Starting from: {len(prev_features)} features, w={prev_window}h, {prev_acc:.1f}%")

    # --- Step 1: Feature Interaction Test ---
    print(f"\n  [ITER3 1/2] Feature Interaction Test (top-5 pairs)...")

    # Get top 5 most important features
    from lightgbm import LGBMClassifier
    df_clean = df_features.dropna(subset=prev_features + ['label']).reset_index(drop=True)
    model_imp = LGBMClassifier(n_estimators=200, max_depth=6, verbose=-1,
                                device=LGBM_DEVICE, random_state=42)
    X = df_clean[prev_features].tail(5000)
    y = df_clean['label'].tail(5000)
    if len(np.unique(y)) < 2:
        top5 = prev_features[:5]
    else:
        model_imp.fit(X, y)
        imp = pd.Series(model_imp.feature_importances_, index=prev_features).sort_values(ascending=False)
        top5 = list(imp.index[:5])
    print(f"    Top 5: {top5}")

    # Test interactions (product of pairs)
    current_acc, _ = _quick_accuracy(df_clean, prev_features, window=prev_window, step=24)
    interaction_features = []
    tested_features = list(prev_features)

    for i in range(len(top5)):
        for j in range(i + 1, len(top5)):
            f1, f2 = top5[i], top5[j]
            int_name = f"{f1}_x_{f2}"
            df_features[int_name] = df_features[f1] * df_features[f2]

            test_set = tested_features + [int_name]
            test_df = df_features.dropna(subset=test_set + ['label'])
            if len(test_df) < 500:
                continue
            acc, _ = _quick_accuracy(test_df, test_set, window=prev_window, step=24)
            if acc > current_acc + 0.3:
                interaction_features.append(int_name)
                tested_features.append(int_name)
                current_acc = acc
                print(f"    {int_name:40s} -> {acc:5.1f}% ** ADDING")
            else:
                print(f"    {int_name:40s} -> {acc:5.1f}%")
                # Clean up unused interaction
                if int_name in df_features.columns:
                    df_features.drop(columns=[int_name], inplace=True)

    if interaction_features:
        print(f"\n    Added {len(interaction_features)} interactions: {', '.join(interaction_features)}")
    else:
        print(f"    No interactions improved accuracy.")

    refined_features = list(tested_features)

    # --- Step 2: Ultra-Fine Window Grid ---
    print(f"\n  [ITER3 2/2] Ultra-Fine Window Search (around w={prev_window}h, step=10)...")

    fine_windows = sorted(set(
        [max(48, prev_window + offset) for offset in range(-40, 50, 10)]
    ))
    print(f"    Testing windows: {fine_windows}")

    global DIAG_WINDOWS, DIAG_STEP
    orig_windows = DIAG_WINDOWS
    orig_step = DIAG_STEP
    DIAG_WINDOWS = fine_windows
    DIAG_STEP = 12

    df_diag = df_features.dropna(subset=refined_features + ['label']).reset_index(drop=True)
    print(f"    {len(df_diag):,} rows, {len(refined_features)} features, step=12")

    best_config = run_diagnostic_for_asset(asset_name, df_diag, refined_features)

    DIAG_WINDOWS = orig_windows
    DIAG_STEP = orig_step

    if best_config:
        best_config['feature_set'] = 'E3'
        best_config['n_features'] = len(refined_features)
        best_config['optimal_features'] = ','.join(refined_features)
        best_config['horizon'] = horizon
        best_config['iteration'] = 3
        print(f"\n  ITERATION 3 RESULT: {best_config['best_combo']} | w={best_config['best_window']}h | "
              f"{best_config['accuracy']:.1f}% | {len(refined_features)} features")
        improvement = best_config['accuracy'] - prev_acc
        print(f"  Improvement over iter 2: {improvement:+.1f}%")

    return best_config, refined_features


def run_mode_e(assets_list, diag_years=1, horizon=PREDICTION_HORIZON, iterations='2'):
    """
    Mode E: Iterative refinement of Mode D results.
    iteration='2'  -> run 2nd pass only
    iteration='23' -> run 2nd + 3rd pass
    """
    do_iter3 = '3' in iterations

    # Kill any orphaned loky workers from previous interrupted runs
    _kill_orphan_workers()

    print("\n" + "=" * 60)
    print(f"  MODE E: ITERATIVE REFINEMENT -- {horizon}h HORIZON")
    print(f"  Iterations: {'2nd + 3rd pass' if do_iter3 else '2nd pass only'}")
    print("=" * 60)

    # Download fresh macro & sentiment data before anything else
    print("\n  Updating macro & sentiment data...")
    try:
        import download_macro_data
        download_macro_data.main()
    except ImportError:
        print("  WARNING: download_macro_data.py not found -- macro features may be stale.")
    except Exception as e:
        print(f"  WARNING: Macro data update failed: {e}")

    update_all_data(assets_list)
    diag_hours = diag_years * 365 * 24
    final_models = []

    for asset_name in assets_list:
        print(f"\n{'='*60}")
        print(f"  REFINING: {asset_name} ({horizon}h)")
        print(f"{'='*60}")

        # Load previous Mode D (or E) result
        prev_config = _load_mode_d_config(asset_name, horizon)
        if prev_config is None:
            print(f"  ERROR: No existing model for {asset_name} {horizon}h.")
            print(f"  Run Mode D first!")
            continue

        prev_features = prev_config['optimal_features']
        if not prev_features or prev_features == 'nan':
            print(f"  ERROR: No optimal features saved for {asset_name}.")
            print(f"  Run Mode D first.")
            continue

        print(f"  Previous: {prev_config['best_combo']} | w={prev_config['best_window']}h | "
              f"{prev_config['accuracy']:.1f}% | {len(prev_features.split(','))} features")

        # Build features
        df_raw = load_data(asset_name)
        if df_raw is None:
            continue

        print(f"\n  Building all features (horizon={horizon}h)...")
        df_full, all_cols = build_all_features(df_raw, asset_name=asset_name, horizon=horizon)

        total_rows = len(df_full)
        if total_rows > diag_hours:
            df_full = df_full.tail(diag_hours).reset_index(drop=True)
            print(f"  Trimmed: {total_rows:,} -> {len(df_full):,} rows (last {diag_years}y)")

        df_clean = df_full.dropna(subset=all_cols + ['label']).reset_index(drop=True)
        print(f"  Clean data: {len(df_clean):,} rows")

        if len(df_clean) < 500:
            print(f"  Not enough data. Skipping.")
            continue

        # --- ITERATION 2 ---
        print(f"\n{'='*60}")
        print(f"  ITERATION 2: FEATURE REFINEMENT + FINER GRID")
        print(f"{'='*60}")

        iter2_config, iter2_features = _run_iteration_2(
            asset_name, df_clean, prev_config, all_cols, horizon)

        if iter2_config is None:
            print("  Iteration 2 failed. Keeping previous config.")
            final_models.append(prev_config)
            continue

        # --- ITERATION 3 (optional) ---
        if do_iter3:
            print(f"\n{'='*60}")
            print(f"  ITERATION 3: ULTRA-FINE TUNING")
            print(f"{'='*60}")

            iter3_config, iter3_features = _run_iteration_3(
                asset_name, df_full, iter2_config, all_cols + [c for c in df_full.columns if '_x_' in c], horizon)

            if iter3_config and iter3_config['accuracy'] > iter2_config['accuracy']:
                final_config = iter3_config
                print(f"\n  >>> Using iteration 3 result (better by "
                      f"{iter3_config['accuracy'] - iter2_config['accuracy']:+.1f}%)")
            else:
                final_config = iter2_config
                print(f"\n  >>> Keeping iteration 2 result (iter 3 didn't improve)")
        else:
            final_config = iter2_config

        final_models.append(final_config)

        # Summary
        original_acc = prev_config['accuracy']
        final_acc = final_config['accuracy']
        print(f"\n  {'='*50}")
        print(f"  REFINEMENT SUMMARY: {asset_name} ({horizon}h)")
        print(f"  {'='*50}")
        print(f"  Before: {prev_config['best_combo']} | w={prev_config['best_window']}h | "
              f"{original_acc:.1f}% | {len(prev_config['optimal_features'].split(','))} features")
        print(f"  After:  {final_config['best_combo']} | w={final_config['best_window']}h | "
              f"{final_acc:.1f}% | {final_config['n_features']} features")
        print(f"  Total improvement: {final_acc - original_acc:+.1f}%")
        print(f"  {'='*50}")

    if not final_models:
        print("\nNo results. Aborting.")
        return

    # Save (merge with existing)
    csv_path = f'{MODELS_DIR}/crypto_hourly_best_models.csv'
    df_best = pd.DataFrame(final_models)
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        if 'horizon' not in df_existing.columns:
            df_existing['horizon'] = 4
        for m in final_models:
            mask = (df_existing['coin'] == m['coin']) & (df_existing['horizon'] == horizon)
            df_existing = df_existing[~mask]
        df_best = pd.concat([df_existing, df_best], ignore_index=True)
    df_best.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"  MODE E COMPLETE -- RESULTS SAVED")
    print(f"{'='*60}")
    for m in final_models:
        fs = m.get('feature_set', 'E')
        print(f"  {m['coin']:6s} -> {m['best_combo']:20s} | w={m['best_window']:4d}h | "
              f"{m['accuracy']:.1f}% | {m.get('n_features', '?')} features | {fs} | {horizon}h")

    # Generate signals + charts
    print("\n" + "=" * 60)
    print("  GENERATING SIGNALS & BACKTEST CHARTS")
    print("=" * 60)

    all_signals = {}
    for config in final_models:
        asset_name = config['coin']
        model_names = config['models'].split('+')
        window = config['best_window']
        feature_override = config['optimal_features'].split(',')

        signals = generate_signals(asset_name, model_names, window, REPLAY_HOURS,
                                   feature_override=feature_override, horizon=horizon)
        signals = simulate_portfolio(signals)
        _print_bootstrap_ci(signals, label=f"{asset_name} {horizon}h")
        all_signals[asset_name] = signals

        generate_backtest_chart(asset_name, signals, model_info=config)

        if signals:
            latest = signals[-1]
            print(f"\n  >> {asset_name} ({horizon}h): {latest['signal']} ({latest['confidence']:.0f}%) "
                  f"| price=${latest['close']:,.2f}")

    export_chart_data(all_signals)

    print("\n" + "=" * 60)
    print("  MODE E COMPLETE")
    print("=" * 60)


# ============================================================
# MODE F: STRATEGY COMPARISON (both_agree / either_agree / 4h / 8h)
# ============================================================
def run_strategy_comparison(assets_list, horizons=None):
    """
    Backtest all combination strategies for each asset using saved model configs:
      - both_agree  : trade only when 4h AND 8h agree
      - either_agree: trade when either 4h OR 8h signals
      - 4h_only     : use 4h model alone
      - 8h_only     : use 8h model alone
    Scores with acc x (1 + return/100). Updates trading_config.json with best strategy.
    Requires Mode D to have been run first for both 4h and 8h.
    """
    csv_path = f'{MODELS_DIR}/crypto_hourly_best_models.csv'
    if not os.path.exists(csv_path):
        print("  ERROR: No saved models found. Run Mode D first.")
        return

    df_models = pd.read_csv(csv_path)

    print("\n" + "=" * 60)
    print("  MODE F: STRATEGY COMPARISON")
    print("=" * 60)

    # Load trading config for updates
    tcfg_path = f'{CONFIG_DIR}/trading_config.json'
    try:
        with open(tcfg_path) as f:
            trading_config = json.load(f)
    except Exception:
        trading_config = {}

    for asset in assets_list:
        print(f"\n{'='*60}")
        print(f"  {asset}: Strategy Comparison")
        print(f"{'='*60}")

        # Load configs for 4h and 8h
        cfg4 = df_models[(df_models['coin'] == asset) & (df_models['horizon'] == 4)]
        cfg8 = df_models[(df_models['coin'] == asset) & (df_models['horizon'] == 8)]

        has_4h = len(cfg4) > 0
        has_8h = len(cfg8) > 0

        if not has_4h and not has_8h:
            print(f"  No saved models for {asset}. Run Mode D first.")
            continue

        # Generate signals for available horizons
        signals_4h, signals_8h = None, None

        if has_4h:
            row4 = cfg4.iloc[0]
            feats4 = row4['optimal_features'].split(',') if pd.notna(row4.get('optimal_features', '')) else None
            signals_4h = generate_signals(asset, row4['models'].split('+'),
                                          int(row4['best_window']), REPLAY_HOURS_F,
                                          feature_override=feats4, horizon=4)
            signals_4h = simulate_portfolio(signals_4h)

        if has_8h:
            row8 = cfg8.iloc[0]
            feats8 = row8['optimal_features'].split(',') if pd.notna(row8.get('optimal_features', '')) else None
            signals_8h = generate_signals(asset, row8['models'].split('+'),
                                          int(row8['best_window']), REPLAY_HOURS_F,
                                          feature_override=feats8, horizon=8)
            signals_8h = simulate_portfolio(signals_8h)

        # Build merged timeline
        sig4_map = {s['datetime']: s for s in (signals_4h or [])}
        sig8_map = {s['datetime']: s for s in (signals_8h or [])}
        all_times = sorted(set(list(sig4_map.keys()) + list(sig8_map.keys())))

        if not all_times:
            print(f"  No signals generated for {asset}.")
            continue

        # Define strategies to test
        strategies = []
        if has_4h and has_8h:
            strategies = ['both_agree', 'either_agree', '4h_only', '8h_only']
        elif has_4h:
            strategies = ['4h_only']
        elif has_8h:
            strategies = ['8h_only']

        results = []
        for strat in strategies:
            cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
            trades, wins, correct = 0, 0, 0
            total_candles = 0

            for dt in all_times:
                s4 = sig4_map.get(dt)
                s8 = sig8_map.get(dt)
                price = (s4 or s8)['close']
                sig4 = s4['signal'] if s4 else 'HOLD'
                conf4 = s4['confidence'] if s4 else 50
                sig8 = s8['signal'] if s8 else 'HOLD'
                conf8 = s8['confidence'] if s8 else 50

                # Determine combined signal based on strategy
                if strat == 'both_agree':
                    if sig4 == 'SELL' or sig8 == 'SELL':
                        signal = 'SELL'
                    elif sig4 == 'BUY' and sig8 == 'BUY' and conf4 >= MIN_CONFIDENCE and conf8 >= MIN_CONFIDENCE:
                        signal = 'BUY'
                    else:
                        signal = 'HOLD'
                elif strat == 'either_agree':
                    if sig4 == 'SELL' or sig8 == 'SELL':
                        signal = 'SELL'
                    elif (sig4 == 'BUY' and conf4 >= MIN_CONFIDENCE) or (sig8 == 'BUY' and conf8 >= MIN_CONFIDENCE):
                        signal = 'BUY'
                    else:
                        signal = 'HOLD'
                elif strat == '4h_only':
                    signal = sig4 if conf4 >= MIN_CONFIDENCE or sig4 == 'SELL' else 'HOLD'
                else:  # 8h_only
                    signal = sig8 if conf8 >= MIN_CONFIDENCE or sig8 == 'SELL' else 'HOLD'

                # Simulate trade
                if signal == 'BUY' and not in_pos:
                    held = cash * (1 - TRADING_FEE) / price
                    cash = 0
                    in_pos = True
                    entry_px = price
                    trades += 1
                elif signal == 'SELL' and in_pos:
                    cash = held * price * (1 - TRADING_FEE)
                    if price > entry_px:
                        wins += 1
                    held = 0
                    in_pos = False

                # Accuracy: did signal match direction?
                # Use 4h signal if available, else 8h
                ref_sig = sig4 if s4 else sig8
                if ref_sig != 'HOLD':
                    total_candles += 1
                    # Correct if signal aligns with strategy signal
                    if ref_sig == signal:
                        correct += 1

            # Close open position
            if in_pos and all_times:
                last_px = (sig4_map.get(all_times[-1]) or sig8_map.get(all_times[-1]))['close']
                cash = held * last_px * (1 - TRADING_FEE)

            cum_ret = (cash / 1000.0 - 1) * 100
            win_rate = (wins / trades * 100) if trades > 0 else 0
            # Use underlying model accuracy as accuracy signal
            # (weighted average of 4h/8h model accuracies if both available)
            if has_4h and has_8h:
                acc4 = cfg4.iloc[0].get('accuracy', 65) / 100
                acc8 = cfg8.iloc[0].get('accuracy', 65) / 100
                base_acc = (acc4 + acc8) / 2
            elif has_4h:
                base_acc = cfg4.iloc[0].get('accuracy', 65) / 100
            else:
                base_acc = cfg8.iloc[0].get('accuracy', 65) / 100

            score = base_acc * (1 + max(cum_ret, 0) / 100)
            results.append((strat, cum_ret, win_rate, trades, score))

        # Sort by score
        results.sort(key=lambda x: -x[4])
        best_strat = results[0][0]

        print(f"\n  {'Strategy':<16} {'Return':>8} {'WinRate':>8} {'Trades':>7} {'Score':>8}")
        print(f"  {'-'*52}")
        for strat, ret, wr, tr, sc in results:
            marker = " <-- BEST" if strat == best_strat else ""
            print(f"  {strat:<16} {ret:>+7.1f}% {wr:>7.0f}% {tr:>7d} {sc:>8.3f}{marker}")

        print(f"\n  >>> BEST STRATEGY for {asset}: {best_strat}")

        # -- Confidence threshold sweep --
        # Replay the best strategy's signals with different thresholds
        # No retraining — just filter the existing signals
        CONF_THRESHOLDS = [60, 65, 70, 75, 80, 85, 90]
        print(f"\n  Confidence threshold sweep (strategy={best_strat}):")
        print(f"  {'Threshold':>10} {'Return':>8} {'WinRate':>8} {'Trades':>7} {'Score':>8}")
        print(f"  {'-'*48}")

        best_conf_score = -999
        best_conf = MIN_CONFIDENCE  # fallback to global default

        for thresh in CONF_THRESHOLDS:
            cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
            trades_c, wins_c = 0, 0

            for dt in all_times:
                s4 = sig4_map.get(dt)
                s8 = sig8_map.get(dt)
                price = (s4 or s8)['close']
                sig4 = s4['signal'] if s4 else 'HOLD'
                conf4 = s4['confidence'] if s4 else 50
                sig8 = s8['signal'] if s8 else 'HOLD'
                conf8 = s8['confidence'] if s8 else 50

                # Apply best strategy with this threshold
                if best_strat == 'both_agree':
                    if sig4 == 'SELL' or sig8 == 'SELL':
                        signal = 'SELL'
                    elif sig4 == 'BUY' and sig8 == 'BUY' and conf4 >= thresh and conf8 >= thresh:
                        signal = 'BUY'
                    else:
                        signal = 'HOLD'
                elif best_strat == 'either_agree':
                    if sig4 == 'SELL' or sig8 == 'SELL':
                        signal = 'SELL'
                    elif (sig4 == 'BUY' and conf4 >= thresh) or (sig8 == 'BUY' and conf8 >= thresh):
                        signal = 'BUY'
                    else:
                        signal = 'HOLD'
                elif best_strat == '4h_only':
                    signal = sig4 if conf4 >= thresh or sig4 == 'SELL' else 'HOLD'
                else:  # 8h_only
                    signal = sig8 if conf8 >= thresh or sig8 == 'SELL' else 'HOLD'

                if signal == 'BUY' and not in_pos:
                    held = cash * (1 - TRADING_FEE) / price
                    cash = 0; in_pos = True; entry_px = price; trades_c += 1
                elif signal == 'SELL' and in_pos:
                    cash = held * price * (1 - TRADING_FEE)
                    if price > entry_px: wins_c += 1
                    held = 0; in_pos = False

            if in_pos and all_times:
                last_px = (sig4_map.get(all_times[-1]) or sig8_map.get(all_times[-1]))['close']
                cash = held * last_px * (1 - TRADING_FEE)

            cum_ret_c = (cash / 1000.0 - 1) * 100
            win_rate_c = (wins_c / trades_c * 100) if trades_c > 0 else 0
            score_c = base_acc * (1 + max(cum_ret_c, 0) / 100)
            marker = ""
            if score_c > best_conf_score:
                best_conf_score = score_c
                best_conf = thresh
                marker = " <-- BEST"
            print(f"  {thresh:>9}% {cum_ret_c:>+7.1f}% {win_rate_c:>7.0f}% {trades_c:>7d} {score_c:>8.3f}{marker}")

        print(f"\n  >>> BEST THRESHOLD for {asset}: {best_conf}%")

        # Update trading config with both strategy and threshold
        if asset not in trading_config:
            trading_config[asset] = {}
        trading_config[asset]['strategy'] = best_strat
        trading_config[asset]['min_confidence'] = best_conf
        print(f"  >>> Updated trading_config.json: {asset} -> strategy={best_strat}, min_confidence={best_conf}%")

    # Save updated trading config
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(tcfg_path, 'w') as f:
        json.dump(trading_config, f, indent=2)

    print("\n" + "=" * 60)
    print("  STRATEGY COMPARISON COMPLETE")
    print("=" * 60)


# ============================================================
# MAIN MENU
# ============================================================
def _run_quick_asset(asset):
    """Quick Mode B for a single asset, both horizons, with combined summary + interactive charts."""
    print("=" * 60)
    print(f"  Quick {asset}: Mode B, 4h+8h")
    print("=" * 60)

    csv_path = f'{MODELS_DIR}/crypto_hourly_best_models.csv'
    if not os.path.exists(csv_path):
        print("  ERROR: No models found!")
        return

    df_best = pd.read_csv(csv_path)
    if 'horizon' not in df_best.columns:
        df_best['horizon'] = 4

    # Run Mode B for PNG charts (200h)
    # Do one shared macro + data download up front, then skip inside each run_mode_b call
    print("\n  Refreshing macro & price data...")
    try:
        import download_macro_data
        download_macro_data.main()
    except ImportError:
        print("  WARNING: download_macro_data.py not found -- macro features may be stale.")
    except Exception as e:
        print(f"  WARNING: Macro data update failed: {e}")
    update_all_data([asset])

    results = {}
    for h in [4, 8]:
        row = df_best[(df_best['coin'] == asset) & (df_best['horizon'] == h)]
        if row.empty:
            print(f"\n  No {h}h model for {asset}")
            continue

        print(f"\n{'#'*60}")
        print(f"  RUNNING {h}h HORIZON")
        print(f"{'#'*60}")
        run_mode_b([asset], horizon_filter=h, skip_data_update=True)

        # Capture latest signal from chart data
        json_path = f'{MODELS_DIR}/crypto_hourly_chart_data.json'
        if os.path.exists(json_path):
            with open(json_path) as f:
                chart_data = json.load(f)
            key = f"{asset}_{h}h"
            asset_signals = chart_data.get('assets', chart_data)  # support both old and new format
            if key in asset_signals and asset_signals[key]:
                last = asset_signals[key][-1]
                r = row.iloc[0]
                results[h] = {
                    'signal': last.get('signal', '?'),
                    'confidence': last.get('confidence', 0),
                    'price': last.get('close', 0),
                    'model': r['best_combo'],
                    'window': int(r['best_window']),
                    'accuracy': r['accuracy'],
                    'feature_set': r.get('feature_set', '?'),
                }

    # Generate interactive HTML charts (720h for both horizons)
    print(f"\n{'#'*60}")
    print(f"  GENERATING INTERACTIVE STRATEGY CHARTS")
    print(f"{'#'*60}")

    # Determine strategy
    # Read strategy from trading_config.json, fall back to both_agree
    try:
        with open(f'{CONFIG_DIR}/trading_config.json') as _f:
            _tcfg = json.load(_f)
        strategy = _tcfg.get(asset, {}).get('strategy', 'both_agree')
    except Exception:
        strategy = 'both_agree'

    signals_4h = None
    signals_8h = None
    for h in [4, 8]:
        row = df_best[(df_best['coin'] == asset) & (df_best['horizon'] == h)]
        if row.empty:
            continue
        r = row.iloc[0]
        model_names = r['models'].split('+')
        window = int(r['best_window'])
        fs = r.get('feature_set', 'A')
        opt = r.get('optimal_features', '')

        if fs in ('D', 'E2', 'E3') and pd.notna(opt) and str(opt).strip() and str(opt).strip() != 'nan':
            feature_override = [f.strip() for f in str(opt).split(',') if f.strip() and f.strip() != 'nan']
        elif fs == 'B':
            feature_override = list(FEATURE_SET_B)
        else:
            feature_override = list(FEATURE_SET_A)

        print(f"  Generating {h}h signals (720h)...")
        sigs = generate_signals(asset, model_names, window, 720,
                                feature_override=feature_override, horizon=h)
        sigs = simulate_portfolio(sigs)
        if h == 4:
            signals_4h = sigs
        else:
            signals_8h = sigs

    if signals_4h or signals_8h:
        generate_strategy_html(asset, signals_4h, signals_8h, strategy=strategy)
        generate_signal_table_html(asset, signals_4h, signals_8h, strategy=strategy)

    # Combined summary
    if results:
        print(f"\n{'='*60}")
        print(f"  {asset} COMBINED SUMMARY")
        print(f"{'='*60}")
        for h, r in sorted(results.items()):
            emoji = '_' if r['signal'] == 'BUY' else '_' if r['signal'] == 'SELL' else '_'
            print(f"  {emoji} {h}h: {r['signal']} ({r['confidence']:.0f}%) | {r['model']} | w={r['window']}h | {r['accuracy']:.1f}% diag")

        if len(results) == 2:
            s4 = results.get(4, {}).get('signal', 'HOLD')
            s8 = results.get(8, {}).get('signal', 'HOLD')
            c4 = results.get(4, {}).get('confidence', 0)
            c8 = results.get(8, {}).get('confidence', 0)

            if s4 == 'SELL' or s8 == 'SELL':
                combined = 'SELL'
                reason = 'at least one model says SELL'
            elif s4 == 'BUY' and s8 == 'BUY' and c4 >= MIN_CONFIDENCE and c8 >= MIN_CONFIDENCE:
                combined = 'BUY (both agree)'
                reason = f'4h+8h both BUY with {c4:.0f}%/{c8:.0f}%'
            elif (s4 == 'BUY' or s8 == 'BUY') and strategy == 'either':
                which = '4h' if s4 == 'BUY' else '8h'
                conf = c4 if s4 == 'BUY' else c8
                if conf >= MIN_CONFIDENCE:
                    combined = f'BUY (either -- {which})'
                    reason = f'{which} says BUY with {conf:.0f}%'
                else:
                    combined = 'HOLD'
                    reason = f'{which} says BUY but confidence {conf:.0f}% < {MIN_CONFIDENCE}%'
            elif s4 == 'BUY' or s8 == 'BUY':
                # strategy == both_agree but only one side agrees
                which = '4h' if s4 == 'BUY' else '8h'
                combined = 'HOLD'
                reason = f'{which} says BUY but both_agree strategy requires both'
            else:
                combined = 'HOLD'
                reason = 'neither model says BUY'

            price = results.get(4, results.get(8, {})).get('price', 0)
            print(f"\n  >>> COMBINED [{strategy}]: {combined}")
            print(f"  >>> Reason: {reason}")
            print(f"  >>> Price: ${price:,.4f}" if price < 100 else f"  >>> Price: ${price:,.2f}")
        print(f"{'='*60}")

    print("\nDone!")


def main():

    has_macro = os.path.exists(MACRO_DIR)

    # ================================================================
    # CLI SHORTCUT: python crypto_trading_system.py D BTC 8h 1y
    # Supports: MODE [ASSETS] [HORIZON] [YEARS]
    # Examples:
    #   python crypto_trading_system.py B BTC 8h
    #   python crypto_trading_system.py D BTC,ETH 4h 1y
    #   python crypto_trading_system.py D BTC 8h 2y
    #   python crypto_trading_system.py B              (all assets, 4h default)
    #   python crypto_trading_system.py D BTC 1y --permtest  (add permutation significance test, ~30min extra)
    # ================================================================
    cli_args    = [a for a in sys.argv[1:] if not a.startswith('--')]
    flag_permtest = '--permtest' in sys.argv  # e.g. python crypto_trading_system.py D BTC 1y --permtest
    if cli_args and cli_args[0].upper() in ('B', 'D', 'E', 'F', '5', '6', '7'):
        mode = cli_args[0].upper()

        # Shortcuts 5/6/7 from CLI
        if mode in ('5', '6', '7'):
            shortcut_map = {'5': 'BTC', '6': 'ETH', '7': 'XRP'}
            _run_quick_asset(shortcut_map[mode])
            return

        # Parse assets (default: all)
        if len(cli_args) >= 2 and not cli_args[1].endswith('h') and not cli_args[1].endswith('y'):
            assets_list = [a.strip().upper() for a in cli_args[1].split(',') if a.strip().upper() in ASSETS]
            if not assets_list:
                assets_list = list(ASSETS.keys())
        else:
            assets_list = list(ASSETS.keys())

        # Parse horizon (default: 4,8h for Mode B, 4h for others)
        horizons = [4, 8] if mode == 'B' else [4]
        for a in cli_args:
            if a.lower().endswith('h') and a[:-1].replace(',', '').isdigit():
                horizons = [int(h) for h in a[:-1].split(',')]

        # Parse years (default: 1y for Mode D, 2y otherwise)
        diag_years = 1
        for a in cli_args:
            if a.lower().endswith('y') and a[:-1].isdigit():
                diag_years = int(a[:-1])

        e_iterations = '2'

        print("=" * 60)
        print(f"  CLI: Mode {mode} | {','.join(assets_list)} | {','.join(str(h)+'h' for h in horizons)} | {diag_years}y")
        print("=" * 60)

    else:

        print("=" * 60)
        print("  CRYPTO HOURLY ML TRADING SYSTEM -- V5")
        print("  Crypto: BTC, ETH, XRP, DOGE")
        print("  Indices: SMI, DAX, CAC40")
        print(f"  Prediction: {', '.join(str(h)+'h' for h in AVAILABLE_HORIZONS)} horizons available")
        print(f"  Macro data: {'FOUND' if has_macro else 'NOT FOUND -- run download_macro_data.py'}")
        print("=" * 60)

        print("\nChoose mode:")
        print("  B. Quick run (saved models + signals + chart)")
        print("  D. FULL PIPELINE (feature analysis -> diagnostic -> signals)")
        print("  E. ITERATIVE REFINEMENT (2nd/3rd pass on Mode D)")
        print("  F. STRATEGY COMPARISON (both_agree / either_agree / 4h / 8h)")
        print("  ---")
        print("  5. Quick BTC (Mode B, both 4h+8h)")
        print("  6. Quick ETH (Mode B, both 4h+8h)")
        print("  7. Quick XRP (Mode B, both 4h+8h)")
        mode = input("\nEnter B/D/E/F or 5-7: ").strip().upper()

        # Shortcuts 5/6/7
        if mode in ('5', '6', '7'):
            shortcut_map = {'5': 'BTC', '6': 'ETH', '7': 'XRP'}
            _run_quick_asset(shortcut_map[mode])
            return

        if mode not in ('B', 'D', 'E', 'F'):
            print("Invalid choice. Defaulting to B.")
            mode = 'B'

        # Mode E iteration choice
        e_iterations = '2'
        if mode == 'E':
            print("\nRefinement iterations:")
            print("  1. 2nd pass only (feature refinement + finer grid, ~1-2h)")
            print("  2. 2nd + 3rd pass (+ interactions + ultra-fine grid, ~3-4h)")
            iter_choice = input("Enter 1 or 2 [1]: ").strip()
            if iter_choice == '2':
                e_iterations = '23'
            else:
                e_iterations = '2'

        print("\nWhich assets?")
        print("  1. All (crypto + indices)")
        print("  2. Crypto only (BTC, ETH, XRP, DOGE)")
        print("  3. Indices only (SMI, DAX, CAC40)")
        print("  4. Choose specific")
        choice = input("Enter choice (1-4): ").strip()

        if choice == '2':
            assets_list = ['BTC', 'ETH', 'XRP', 'DOGE']
        elif choice == '3':
            assets_list = ['SMI', 'DAX', 'CAC40']
        elif choice == '4':
            print(f"Available: {', '.join(ASSETS.keys())}")
            selected = input("Enter comma-separated names: ").strip().upper()
            assets_list = [a.strip() for a in selected.split(',') if a.strip() in ASSETS]
        else:
            assets_list = list(ASSETS.keys())

        mode_labels = {'B': 'Quick Run',
                       'D': 'Full Pipeline', 'E': 'Iterative Refinement',
                       'F': 'Strategy Comparison'}
        print(f"\nAssets: {', '.join(assets_list)}")
        print(f"Mode: {mode} ({mode_labels.get(mode, mode)})")

        # Horizon selection
        print("\nPrediction horizon:")
        print("  1. 4 hours ahead (default)")
        print("  2. 8 hours ahead")
        print("  3. Both (4h + 8h)")
        h_choice = input("Enter choice (1-3) [1]: ").strip()
        if h_choice == '2':
            horizons = [8]
        elif h_choice == '3':
            horizons = [4, 8]
        else:
            horizons = [4]
        print(f"Horizon(s): {', '.join(str(h)+'h' for h in horizons)}")

        diag_years = 1
        if mode in ('D', 'E'):
            print("\nDiagnostic data range:")
            print("  1. Last 1 year   (recommended -- recent market behaviour)")
            print("  2. Last 2 years  (cross-regime test)")
            print("  3. Last 4 years  (slowest)")
            range_choice = input("Enter choice (1-3) [1]: ").strip()
            if range_choice == '2':
                diag_years = 2
            elif range_choice == '3':
                diag_years = 4
            else:
                diag_years = 1
            print(f"Diagnostic range: last {diag_years} year{'s' if diag_years > 1 else ''}")

    # Mode B doesn't loop per horizon -- it handles all horizons in one call
    if mode == 'B':
        run_mode_b(assets_list, horizon_filter=horizons[0] if len(horizons) == 1 else None)
    else:
        for h in horizons:
            if len(horizons) > 1:
                print(f"\n{'#'*60}")
                print(f"  RUNNING {h}h HORIZON")
                print(f"{'#'*60}")
            if mode == 'D':
                run_mode_d(assets_list, diag_years=diag_years, horizon=h, permtest=flag_permtest)
            elif mode == 'E':
                run_mode_e(assets_list, diag_years=diag_years, horizon=h, iterations=e_iterations)
            elif mode == 'F':
                pass  # handled below

        # After Mode D with both horizons: run strategy comparison
        if mode == 'D' and len(horizons) == 2:
            run_strategy_comparison(assets_list, horizons)
        elif mode == 'F':
            run_strategy_comparison(assets_list, horizons)

    print("\nDone!")


if __name__ == '__main__':
    main()
