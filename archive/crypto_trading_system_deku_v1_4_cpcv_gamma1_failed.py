"""
Crypto Hourly Trading System — DEKU V1.4
============================================================
ML trading system for BTC, ETH, XRP, DOGE, SOL, LINK, ADA, AVAX, DOT
with 4h and 8h horizons. 130 features -> walk-forward ML -> BUY/SELL/HOLD signals.

DEKU V1.4 — Fixed gamma=1.0, data_months hyperparameter, adaptive CPCV:
  1. RandomSampler always (no TPE) — pure random search
  2. gamma=1.0 fixed (no decay) — removed from search space
  3. data_months as Optuna param: [1, 2, 3, 4, 6] months of training history
  4. Adaptive CPCV: 6 groups for data_months >= 4, 4 groups otherwise
  5. 5 base models (RF, GB, XGB, LR, LGBM) = 26 ensemble combos
  6. LGBM importance ranking for feature selection

  Search space: combo × window × data_months × n_features
  APF scoring: Adjusted PF = raw_PF / buyhold_PF

Modes:
  B. Quick run (saved models)    D. Optuna optimization
  DF. D then F                   F. Strategy comparison
  H. Horizon search              HF. H then F
  5/6/7. Quick BTC/ETH/XRP

CLI Usage:
  python crypto_trading_system_deku_v1_4.py D BTC 4,8h
  python crypto_trading_system_deku_v1_4.py D BTC 4,8h --trials 150
  python crypto_trading_system_deku_v1_4.py DF BTC,ETH 4,8h
  python crypto_trading_system_deku_v1_4.py B BTC

Outputs:
  charts/{ASSET}_backtest.png
  models/crypto_deku_v14_best_models.csv
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
# NOTE: These are REMOVED before diagnostic phase (RF/GB/LR need BLAS threads).
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
from sklearn.utils.parallel import Parallel, delayed
try:
    import optuna
except ImportError:
    optuna = None  # only needed for Mode D/H optimization, not signal generation
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # graceful fallback if xgboost not installed
from hardware_config import (
    MACHINE, N_JOBS_PARALLEL, LGBM_DEVICE,
    get_cpu_models, get_gpu_models, get_all_models, get_diagnostic_models,
)
# Cap loky worker pool to configured parallelism (not raw CPU count)
os.environ['LOKY_MAX_CPU_COUNT'] = str(N_JOBS_PARALLEL)

# Lower process priority so the live trader always gets CPU first
# BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
except Exception:
    try:
        os.nice(10)  # Unix fallback
    except Exception:
        pass


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
            # Kill loky workers and resource trackers (but not ourselves)
            if 'loky' in line.lower():
                # Extract PID (last number on the line)
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
        pass  # non-critical — don't fail if cleanup fails


# ── Resume / Checkpoint helpers ──────────────────────────────────────────────
RESUME_DIR = 'models/.resume_hourly'

def _resume_path(asset, horizon, step):
    return os.path.join(RESUME_DIR, f'{asset}_{horizon}_{step}.json')

def _save_checkpoint(asset, horizon, step, data):
    """Save intermediate result so --resume can skip this step."""
    os.makedirs(RESUME_DIR, exist_ok=True)
    with open(_resume_path(asset, horizon, step), 'w') as f:
        json.dump(data, f)

def _load_checkpoint(asset, horizon, step):
    """Load a checkpoint if it exists. Returns None otherwise."""
    path = _resume_path(asset, horizon, step)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def _clear_checkpoints(asset, horizon):
    """Remove all checkpoints for an (asset, horizon) after successful completion."""
    for step in ('features', 'diagnostic', 'diag_partial'):
        path = _resume_path(asset, horizon, step)
        if os.path.exists(path):
            os.remove(path)


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
    'SOL':   {'source': 'binance', 'ticker': 'SOL/USDT',  'file': 'data/sol_hourly_data.csv',  'start': '2020-08-01T00:00:00Z'},
    'LINK':  {'source': 'binance', 'ticker': 'LINK/USDT', 'file': 'data/link_hourly_data.csv', 'start': '2019-01-01T00:00:00Z'},
    'ADA':   {'source': 'binance', 'ticker': 'ADA/USDT',  'file': 'data/ada_hourly_data.csv',  'start': '2018-04-01T00:00:00Z'},
    'AVAX':  {'source': 'binance', 'ticker': 'AVAX/USDT', 'file': 'data/avax_hourly_data.csv', 'start': '2021-09-01T00:00:00Z'},
    'DOT':   {'source': 'binance', 'ticker': 'DOT/USDT',  'file': 'data/dot_hourly_data.csv',  'start': '2020-08-01T00:00:00Z'},
    'SMI':   {'source': 'yfinance', 'ticker': '^SSMI',    'file': 'data/smi_hourly_data.csv',  'start': None},
    'DAX':   {'source': 'yfinance', 'ticker': '^GDAXI',   'file': 'data/dax_hourly_data.csv',  'start': None},
    'CAC40': {'source': 'yfinance', 'ticker': '^FCHI',    'file': 'data/cac40_hourly_data.csv', 'start': None},
}

DEKU_VERSION = '1.4'              # Deku V1.4: Fixed gamma=1.0, data_months hyperparameter, adaptive CPCV

HORIZON_SHORT = 4                 # short horizon (parametric — change here to switch)
HORIZON_LONG = 8                  # long horizon (parametric — change here to switch)
AVAILABLE_HORIZONS = [HORIZON_SHORT, HORIZON_LONG]
PREDICTION_HORIZON = HORIZON_SHORT  # default horizon

# Create output folders
for _d in ['data', 'data/macro_data', 'charts', 'models', 'config']:
    os.makedirs(_d, exist_ok=True)
TRADING_FEE_BASE = 0.0009  # 0.09% Revolut X taker fee (applied on BUY and SELL)
SLIPPAGE = 0.0002          # 0.02% estimated slippage (market impact, spread)
TRADING_FEE = TRADING_FEE_BASE + SLIPPAGE  # 0.11% total cost per trade
MIN_CONFIDENCE = 75   # Minimum confidence % for strategy signals
REPLAY_HOURS = 200
REPLAY_HOURS_F = 400   # Mode F strategy selection — longer window for more trades
DIAG_STEP = 36
DIAG_WINDOWS = [24, 36, 48, 72, 100, 150, 200]  # 24/36 added: winners hit floor at 48
MIN_COMBO_SIZE = 2   # minimum number of models in ensemble — solos removed (overfit, poor calibration)
DEFAULT_GAMMA = 1.0  # no decay fallback — per-model gamma read from CSV
EMBARGO_CANDLES = 4  # gap between train/test in walk-forward to prevent label overlap leakage

# Per-horizon feature ranges — short tighter to avoid overfitting, long keeps full range
N_FEATURES_RANGE = {
    HORIZON_SHORT: (4, 40),   # short horizon: narrower (overfitting observed with 80+ features)
    HORIZON_LONG:  (4, 80),   # long horizon: full range (kept 75 features and performed well)
}
N_FEATURES_RANGE_DEFAULT = (4, 80)  # fallback for unknown horizons

# Mode H: horizon search range
HORIZON_SEARCH = [3, 4, 5, 6, 7, 8]
MODE_H_DEFAULT_TRIALS = 200  # extra budget for horizon dimension

# CPCV: Combinatorial Purged Cross-Validation (López de Prado)
# Adaptive groups based on data_months — see _get_cpcv_params()
CPCV_PURGE_CANDLES = 4   # purge rows at train/test boundary (same as EMBARGO_CANDLES)
PBO_MAX = 0.50       # reject configs with PBO > 50% (probability of backtest overfitting)


def _get_cpcv_params(data_months):
    """Adaptive CPCV groups based on data size.
    Fewer groups for shorter data to keep folds statistically meaningful."""
    if data_months >= 4:
        return 6, 2  # 6 groups, k=2 → C(6,2) = 15 paths
    else:
        return 4, 2  # 4 groups, k=2 → C(4,2) = 6 paths

DATA_MONTHS_OPTIONS = [1, 2, 3, 4, 5, 6]  # months of training history for Optuna to search

# Optuna scoring metric — set via --metric flag (default: apf)
OPTUNA_METRIC = 'apf'
VALID_METRICS = {'apf', 'rawpf', 'calmar', 'return', 'rpf_sqrt'}

# Label mode: 'fee_aware' = label=1 when return > 2×fee (no lookahead bias)
LABEL_MODE = 'fee_aware'



def _compute_optuna_score(result):
    """Compute Optuna objective score from eval result tuple based on selected metric.
    Result tuple: (combo, window, acc, total, cum_return, win_rate,
                   trades, total_gain, total_loss, max_dd%, adjusted_pf, raw_pf, bh_pf)"""
    cum_return = result[4]
    trades = result[6]
    max_dd = result[9]  # already in %
    adjusted_pf = result[10]
    raw_pf = result[11]

    if OPTUNA_METRIC == 'apf':
        return adjusted_pf
    elif OPTUNA_METRIC == 'rawpf':
        return raw_pf
    elif OPTUNA_METRIC == 'calmar':
        return cum_return / max_dd if max_dd > 0 else cum_return
    elif OPTUNA_METRIC == 'return':
        return cum_return
    elif OPTUNA_METRIC == 'rpf_sqrt':
        import math
        return raw_pf * math.sqrt(trades) if trades > 0 else 0.0
    else:
        return adjusted_pf


def get_decay_weights(n_samples, gamma):
    """Exponential decay: newest sample=1, oldest=gamma^(n-1).
    Returns None when gamma >= 1.0 (no decay, zero overhead)."""
    if gamma is None or gamma >= 1.0:
        return None
    ages = np.arange(n_samples - 1, -1, -1)
    return gamma ** ages


def load_funding_rate(asset_name='BTC'):
    """Load funding rate from derivatives CSV. Returns Series indexed by datetime, or None."""
    fname = f'derivatives_{asset_name.lower()}.csv'
    fpath = os.path.join(MACRO_DIR, fname)
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath, parse_dates=['datetime'], index_col='datetime')
        if 'funding_rate' in df.columns:
            return df['funding_rate'].dropna()
    except Exception:
        pass
    return None


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
def build_hourly_features(df_hourly, horizon=PREDICTION_HORIZON, verbose=True):
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

    # ---- Garman-Klass volatility (1980): uses OHLC, 7.4x more efficient than close-to-close ----
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])
    gk_single = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    df['gk_volatility_14h'] = gk_single.rolling(14).mean().apply(np.sqrt)
    df['gk_volatility_48h'] = gk_single.rolling(48).mean().apply(np.sqrt)

    # ---- ADX — trend strength (Wilder 1978) ----
    _tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    plus_dm = (df['high'] - df['high'].shift(1)).clip(lower=0)
    minus_dm = (df['low'].shift(1) - df['low']).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr_14 = _tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-10))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx_14h'] = dx.rolling(14).mean()
    df['plus_di_14h'] = plus_di
    df['minus_di_14h'] = minus_di

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

    if LABEL_MODE == 'fee_aware':
        # Fee-aware: label=1 only when future return exceeds round-trip cost
        df['label'] = (future_return > 2 * TRADING_FEE).astype(int)
    else:
        # Rolling median on PAST realized returns (no lookahead)
        past_return = df['close'] / df['close'].shift(horizon) - 1
        rolling_median = past_return.rolling(200, min_periods=50).median()
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
        'gk_volatility_14h', 'gk_volatility_48h',
        'adx_14h', 'plus_di_14h', 'minus_di_14h',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'price_velocity_1h', 'price_velocity_4h',
        'price_accel_1h', 'price_accel_4h', 'price_accel_12h', 'price_accel_24h',
        'price_jerk_1h',
    ]

    # Persist forward return for return-weighted sampling (not a feature, underscore prefix)
    df['_forward_return'] = future_return

    keep_cols = ['datetime', 'close', 'high', 'low', 'volume'] + feature_cols + ['label', '_forward_return']
    df = df[keep_cols].copy()

    nan_counts = df[feature_cols + ['label']].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if verbose and len(nan_cols) > 0:
        print(f"    Rows before dropna: {len(df)}")

    df = df.dropna().reset_index(drop=True)
    if verbose:
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
# Test harnesses set this env var to redirect model output away from production CSV
MODELS_CSV_OVERRIDE = os.environ.get('MODELS_CSV_OVERRIDE', '')


def _get_models_csv_path():
    """Return the models CSV path for V1.4."""
    if MODELS_CSV_OVERRIDE:
        return MODELS_CSV_OVERRIDE
    if OPTUNA_METRIC != 'apf':
        return f'{MODELS_DIR}/crypto_deku_v14_best_models_{OPTUNA_METRIC}.csv'
    return f'{MODELS_DIR}/crypto_deku_v14_best_models.csv'


def _backup_models_csv():
    """Create a timestamped backup of production CSV before writing (failsafe against contamination)."""
    src = f'{MODELS_DIR}/crypto_deku_best_models.csv'
    if os.path.exists(src) and not MODELS_CSV_OVERRIDE:
        import shutil
        bak = f'{MODELS_DIR}/crypto_deku_best_models_backup.csv'
        shutil.copy2(src, bak)
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


def build_all_features(df_hourly, asset_name='BTC', horizon=PREDICTION_HORIZON, verbose=True):
    df, base_cols = build_hourly_features(df_hourly, horizon=horizon, verbose=verbose)
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

    # Load funding rate for regime gate (not a feature — underscore prefix)
    funding_series = load_funding_rate(asset_name)
    if funding_series is not None:
        fr_df = funding_series.to_frame('_funding_rate')
        fr_df['_merge_dt'] = fr_df.index.floor('h')
        df['_merge_dt'] = pd.to_datetime(df['datetime']).dt.floor('h')
        df = df.merge(fr_df[['_merge_dt', '_funding_rate']], on='_merge_dt', how='left')
        df = df.drop(columns=['_merge_dt'], errors='ignore')
        df['_funding_rate'] = df['_funding_rate'].ffill()
        n_funding = df['_funding_rate'].notna().sum()
        if verbose and n_funding > 0:
            print(f"    Funding rate: {n_funding} rows loaded (regime gate)")
    else:
        df['_funding_rate'] = np.nan

    if verbose:
        print(f"    All features: {len(base_cols)} base + {added} macro/sentiment/cross-asset = {len(all_cols)} total")
    return df, all_cols


# ============================================================
# MODELS — Deku adds XGBoost (5 base models, 26 ensemble combos)
# ============================================================
def _get_deku_models():
    """Production models with XGBoost added."""
    from lightgbm import LGBMClassifier
    return {
        'RF':   lambda: RandomForestClassifier(n_estimators=300, max_depth=4, class_weight='balanced', random_state=42, n_jobs=1),
        'GB':   lambda: GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=42),
        'XGB':  lambda: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42,
                                       tree_method='hist', verbosity=0, n_jobs=1),
        'LR':   lambda: LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'LGBM': lambda: LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                        class_weight='balanced', verbose=-1, random_state=42, device='gpu'),
    }

def _get_deku_diagnostic_models():
    """Lightweight models for diagnostic/Optuna search (100 estimators)."""
    from lightgbm import LGBMClassifier
    return {
        'RF':   lambda: RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced', random_state=42, n_jobs=1),
        'GB':   lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        'XGB':  lambda: XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42,
                                       tree_method='hist', verbosity=0, n_jobs=1),
        'LR':   lambda: LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'LGBM': lambda: LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                        class_weight='balanced', verbose=-1, random_state=42, device='gpu'),
    }

ALL_MODELS = _get_deku_models()


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


def _quick_score(df_features, feature_cols, window=ANALYSIS_WINDOW, step=ANALYSIS_STEP, device=None, gamma=1.0):
    """Fast walk-forward test with LGBM only.
    Returns (accuracy, alpha, n_tests, adjusted_pf).
    Alpha = strategy return - buy & hold return (same period, with 0.09% fees).
    Adjusted PF = raw_PF / buyhold_PF (measures skill vs passive holding).
    device: override LGBM device ('cpu' for parallel safety, None = use default)."""
    from lightgbm import LGBMClassifier

    lgbm_device = device if device else LGBM_DEVICE
    n = len(df_features)
    min_start = window + 50
    if n < min_start + 30:
        return 0, 0, 0, 0

    correct = 0
    total = 0

    # Portfolio simulation alongside accuracy
    cash       = 1.0
    in_pos     = False
    entry_px   = 0.0
    start_px   = float(df_features.iloc[min_start]['close'])
    total_gain = 0.0
    total_loss = 0.0
    trades     = 0

    # Buy-and-hold PF tracking: record each candle return at test points
    bh_gains = 0.0
    bh_losses = 0.0
    prev_price = None

    for i in range(min_start, n, step):
        train    = df_features.iloc[max(0, i - window):i]
        test_row = df_features.iloc[i:i+1]
        X_train  = train[feature_cols]
        y_train  = train['label'].values
        X_test   = test_row[feature_cols]
        y_true   = test_row['label'].values[0]
        price    = float(test_row['close'].values[0])

        # Buy-and-hold: track return from previous test point
        if prev_price is not None:
            bh_ret_step = (price - prev_price) / prev_price
            if bh_ret_step > 0:
                bh_gains += bh_ret_step
            else:
                bh_losses += bh_ret_step
        prev_price = price

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
        sw = get_decay_weights(len(y_train), gamma)
        model.fit(X_train_s, y_train, sample_weight=sw)
        pred = model.predict(X_test_s)[0]

        if pred == y_true:
            correct += 1
        total += 1

        # Portfolio: BUY on pred=1, SELL on pred=0
        if pred == 1 and not in_pos:
            in_pos   = True
            entry_px = price * (1 + TRADING_FEE)
        elif pred == 0 and in_pos:
            sell_px = price * (1 - TRADING_FEE)
            trade_ret = (sell_px - entry_px) / entry_px
            cash *= (1 + trade_ret)
            trades += 1
            if trade_ret > 0:
                total_gain += trade_ret
            else:
                total_loss += trade_ret
            in_pos  = False

    # Close open position at end
    if in_pos and total > 0:
        last_px = float(df_features.iloc[-1]['close'])
        sell_px = last_px * (1 - TRADING_FEE)
        trade_ret = (sell_px - entry_px) / entry_px
        cash *= (1 + trade_ret)
        trades += 1
        if trade_ret > 0:
            total_gain += trade_ret
        else:
            total_loss += trade_ret

    last_px   = float(df_features.iloc[-1]['close'])
    strat_ret = (cash - 1.0) * 100
    bh_ret    = (last_px / start_px - 1) * 100
    alpha     = round(strat_ret - bh_ret, 2)
    accuracy  = correct / total * 100 if total > 0 else 0

    # CASCA V1.4: Adjusted Profit Factor = raw_PF / buyhold_PF
    if trades < 3:
        raw_pf = 0.0
    elif total_loss == 0:
        raw_pf = min(total_gain * 100, 20.0)
    else:
        raw_pf = min(total_gain / abs(total_loss), 20.0)

    # Buy-and-hold PF over same test points
    if bh_losses == 0:
        bh_pf = max(bh_gains * 100, 1.0)  # floor at 1.0 to avoid division issues
    else:
        bh_pf = max(bh_gains / abs(bh_losses), 0.01)  # floor at 0.01

    adjusted_pf = raw_pf / bh_pf if raw_pf > 0 else 0.0

    return accuracy, alpha, total, adjusted_pf


def _quick_accuracy(df_features, feature_cols, window=ANALYSIS_WINDOW, step=ANALYSIS_STEP, device=None, gamma=1.0):
    """Backward-compatible wrapper -- returns (accuracy, n_tests). Used by Mode E."""
    acc, _, n, _ = _quick_score(df_features, feature_cols, window=window, step=step, device=device, gamma=gamma)
    return acc, n


def _test_lgbm_importance(df_features, feature_cols, gamma=1.0):
    """Train LGBM and extract feature importance."""
    from lightgbm import LGBMClassifier

    print(f"\n  [1/5] LGBM Feature Importance (gain-based)  [{datetime.now().strftime('%H:%M:%S')}]")
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
    sw = get_decay_weights(len(y), gamma)
    model.fit(X_s, y, sample_weight=sw)

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


def _perm_one_feature(df_features, feature_cols, feat, baseline_acc, baseline_alpha, baseline_pf, gamma=1.0):
    """Helper for parallel permutation test. Returns (feat, acc_drop, alpha_drop, pf_drop)."""
    import os, warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    df_shuffled = df_features.copy()
    df_shuffled[feat] = np.random.permutation(df_shuffled[feat].values)
    shuffled_acc, shuffled_alpha, _, shuffled_pf = _quick_score(df_shuffled, feature_cols, device='cpu', gamma=gamma)
    return feat, baseline_acc - shuffled_acc, baseline_alpha - shuffled_alpha, baseline_pf - shuffled_pf


def _test_permutation_importance(df_features, feature_cols, gamma=1.0):
    """Shuffle each feature and measure accuracy + alpha + profit factor drop. Parallelized."""
    print(f"\n  [2/5] Permutation Importance (parallel)  [{datetime.now().strftime('%H:%M:%S')}]")
    baseline_acc, baseline_alpha, n_tests, baseline_pf = _quick_score(df_features, feature_cols, gamma=gamma)
    print(f"    Baseline: {baseline_acc:.1f}% acc | {baseline_alpha:+.1f}% alpha | APF={baseline_pf:.2f} (n={n_tests})")

    n_workers = min(N_JOBS_PARALLEL, len(feature_cols))
    print(f"    Testing {len(feature_cols)} features ({n_workers} workers)...")
    t0 = time.time()
    with _suppress_stderr():
        perm_results = Parallel(n_jobs=n_workers, verbose=0)(
            delayed(_perm_one_feature)(df_features, feature_cols, feat, baseline_acc, baseline_alpha, baseline_pf, gamma=gamma)
            for feat in feature_cols
        )
    print(f"    Done in {(time.time() - t0)/60:.1f} min")

    results = []
    for feat, acc_drop, alpha_drop, pf_drop in perm_results:
        results.append({'feature': feat, 'acc_drop': acc_drop, 'alpha_drop': alpha_drop, 'pf_drop': pf_drop})
        print(f"    {feat:30s} pf_drop: {pf_drop:+5.2f}  alpha_drop: {alpha_drop:+6.1f}%")

    return pd.DataFrame(results).sort_values('pf_drop', ascending=False)


def _ablation_one_feature(df_features, feature_cols, feat, baseline_acc, baseline_alpha, baseline_pf, gamma=1.0):
    """Helper for parallel ablation test. Returns (feat, acc, acc_change, alpha_change, pf_change)."""
    import os, warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    reduced = [f for f in feature_cols if f != feat]
    acc, alpha, _, pf = _quick_score(df_features, reduced, device='cpu', gamma=gamma)
    return feat, acc, acc - baseline_acc, alpha - baseline_alpha, pf - baseline_pf


def _test_ablation(df_features, feature_cols, gamma=1.0):
    """Drop each feature one at a time and measure profit factor + alpha. Parallelized."""
    print(f"\n  [3/5] Ablation Test (parallel, drop one at a time)  [{datetime.now().strftime('%H:%M:%S')}]")
    baseline_acc, baseline_alpha, _, baseline_pf = _quick_score(df_features, feature_cols, gamma=gamma)
    print(f"    Baseline ({len(feature_cols)} features): APF={baseline_pf:.2f} | {baseline_alpha:+.1f}% alpha | {baseline_acc:.1f}% acc")

    n_workers = min(N_JOBS_PARALLEL, len(feature_cols))
    print(f"    Testing {len(feature_cols)} features ({n_workers} workers)...")
    t0 = time.time()
    with _suppress_stderr():
        ablation_results = Parallel(n_jobs=n_workers, verbose=0)(
            delayed(_ablation_one_feature)(df_features, feature_cols, feat, baseline_acc, baseline_alpha, baseline_pf, gamma=gamma)
            for feat in feature_cols
        )
    print(f"    Done in {(time.time() - t0)/60:.1f} min")

    results = []
    for feat, acc, acc_change, alpha_change, pf_change in ablation_results:
        results.append({'dropped': feat, 'accuracy': acc,
                        'change': acc_change, 'alpha_change': alpha_change, 'pf_change': pf_change})
        # CASCA: compare by profit factor change
        marker = ' ** IMPROVES' if pf_change > 0 else ''
        print(f"    Drop {feat:30s} -> PF_chg: {pf_change:+5.2f}  "
              f"alpha_chg: {alpha_change:+6.1f}%{marker}")

    return pd.DataFrame(results).sort_values('change', ascending=False)


def _reduced_one_set(df_features, ranked, n_feat, gamma=1.0):
    """Helper for parallel reduced set test. Returns (n_feat, acc, alpha, combined_score).
    CASCA: uses profit factor as score for feature selection."""
    import os, warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    top_n = ranked[:n_feat]
    acc, alpha, _, pf = _quick_score(df_features, top_n, device='cpu', gamma=gamma)
    combined = pf  # CASCA V1.4: rank feature sets by adjusted profit factor
    return n_feat, acc, alpha, combined


def _test_reduced_sets(df_features, feature_cols, importance_df, gamma=1.0):
    """Test accuracy with top-N features. Parallelized."""
    print(f"\n  [4/5] Reduced Feature Sets (parallel, top-N by importance)  [{datetime.now().strftime('%H:%M:%S')}]")
    ranked = importance_df['feature'].tolist()

    test_sizes = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50]
    test_sizes = [n for n in test_sizes if n < len(feature_cols)]
    test_sizes.append(len(feature_cols))

    n_workers = min(N_JOBS_PARALLEL, len(test_sizes))
    print(f"    Testing {len(test_sizes)} set sizes ({n_workers} workers)...")
    t0 = time.time()
    with _suppress_stderr():
        reduced_results = Parallel(n_jobs=n_workers, verbose=0)(
            delayed(_reduced_one_set)(df_features, ranked, n_feat, gamma=gamma)
            for n_feat in test_sizes
        )
    print(f"    Done in {(time.time() - t0)/60:.1f} min")

    results = []
    for n_feat, acc, alpha, combined in sorted(reduced_results):
        results.append({'n_features': n_feat, 'accuracy': acc,
                        'alpha': alpha, 'combined_score': combined})
        bar = '#' * int(combined * 10)  # CASCA V1.4: bar based on adjusted PF
        print(f"    Top {n_feat:3d} features: APF={combined:.2f} | {alpha:+6.1f}% alpha | "
              f"{acc:5.1f}% acc  {bar}")

    df_results = pd.DataFrame(results)
    best_row = df_results.loc[df_results['combined_score'].idxmax()]
    print(f"\n    OPTIMAL: Top {int(best_row['n_features'])} -> "
          f"APF={best_row['combined_score']:.2f} | {best_row['alpha']:+.1f}% alpha | "
          f"{best_row['accuracy']:.1f}% acc")
    return df_results


def _score_features(feature_cols, importance_df, ablation_df, permutation_df):
    """Score features across all tests, return optimal list."""
    print(f"\n  [5/5] Scoring & Selection  [{datetime.now().strftime('%H:%M:%S')}]")
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

    # Permutation -- CASCA: rank by alpha_drop only (accuracy ignored)
    # alpha_drop > 0 means shuffling this feature HURT returns = feature matters
    if permutation_df is not None:
        for _, row in permutation_df.iterrows():
            f = row['feature']
            # CASCA: pf_drop > 0 means shuffling this feature HURT profit factor = feature matters
            pf_drop = row.get('pf_drop', 0)
            if pf_drop > 1.0:
                scores[f] += 3
            elif pf_drop > 0.5:
                scores[f] += 2
            elif pf_drop > 0.1:
                scores[f] += 1
            elif pf_drop < -0.5:
                scores[f] -= 2
            elif pf_drop < -0.1:
                scores[f] -= 1

    # Ablation -- CASCA: rank by pf_change only (accuracy ignored)
    # pf_change > 0 means dropping this feature IMPROVED profit factor = feature was hurting
    if ablation_df is not None:
        for _, row in ablation_df.iterrows():
            f = row['dropped']
            pf_change = row.get('pf_change', 0)
            if pf_change > 1.0:
                scores[f] -= 3
            elif pf_change > 0.5:
                scores[f] -= 2
            elif pf_change > 0.1:
                scores[f] -= 1
            elif pf_change < -0.5:
                scores[f] += 2
            elif pf_change < -0.1:
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


def run_feature_analysis(asset_name, df_features, all_feature_cols, gamma=1.0):
    """
    Run full 5-test feature analysis on one asset.
    Returns the optimal feature list.
    """
    print(f"\n{'='*60}")
    print(f"  FEATURE ANALYSIS: {asset_name} ({len(all_feature_cols)} features)")
    print(f"{'='*60}")

    t0 = time.time()

    # 1. LGBM importance
    importance_df = _test_lgbm_importance(df_features, all_feature_cols, gamma=gamma)

    # 2. Permutation importance
    permutation_df = _test_permutation_importance(df_features, all_feature_cols, gamma=gamma)

    # 3. Ablation
    ablation_df = _test_ablation(df_features, all_feature_cols, gamma=gamma)

    # 4. Reduced sets
    reduced_df = _test_reduced_sets(df_features, all_feature_cols, importance_df, gamma=gamma)

    # 5. Score and select
    score_df, keep, maybe, drop = _score_features(
        all_feature_cols, importance_df, ablation_df, permutation_df)

    elapsed = time.time() - t0
    print(f"\n  Feature analysis completed in {elapsed/60:.1f} minutes")

    # Determine optimal set: KEEP features + test with/without MAYBE
    optimal_features = list(keep)

    # Quick test: KEEP only vs KEEP + MAYBE -- CASCA: compare by profit factor
    if maybe:
        acc_keep,  alpha_keep,  _, pf_keep = _quick_score(df_features, keep, gamma=gamma)
        acc_all,   alpha_all,   _, pf_all  = _quick_score(df_features, keep + maybe, gamma=gamma)
        print(f"\n  KEEP only       ({len(keep):3d} feat): APF={pf_keep:.2f} | "
              f"{alpha_keep:+.1f}% alpha | {acc_keep:.1f}% acc")
        print(f"  KEEP + MAYBE    ({len(keep)+len(maybe):3d} feat): APF={pf_all:.2f} | "
              f"{alpha_all:+.1f}% alpha | {acc_all:.1f}% acc")
        if pf_all > pf_keep + 0.1:
            optimal_features = keep + maybe
            print(f"  >>> Using KEEP + MAYBE ({len(optimal_features)} features)")
        else:
            print(f"  >>> Using KEEP only ({len(optimal_features)} features)")
    else:
        print(f"\n  >>> Using KEEP ({len(optimal_features)} features)")

    # Also check best reduced set from test 4 -- compare by profit factor
    if reduced_df is not None and len(reduced_df) > 0:
        best_n_row     = reduced_df.loc[reduced_df['combined_score'].idxmax()]
        best_n         = int(best_n_row['n_features'])
        best_n_score   = best_n_row['combined_score']  # profit factor
        best_n_acc     = best_n_row['accuracy']
        best_n_alpha   = best_n_row['alpha']
        ranked         = importance_df['feature'].tolist()
        top_n_features = ranked[:best_n]

        opt_acc, opt_alpha, _, opt_pf = _quick_score(df_features, optimal_features, gamma=gamma)
        print(f"  Scored optimal  ({len(optimal_features):3d} feat): APF={opt_pf:.2f} | "
              f"{opt_alpha:+.1f}% alpha | {opt_acc:.1f}% acc")
        print(f"  Top-{best_n} by LGBM ({best_n:3d} feat): APF={best_n_score:.2f} | "
              f"{best_n_alpha:+.1f}% alpha | {best_n_acc:.1f}% acc")

        if best_n_score > opt_pf + 0.2:
            optimal_features = top_n_features
            print(f"  >>> Switching to Top-{best_n} (PF +{best_n_score - opt_pf:.2f})")

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
                     feature_override=None, horizon=PREDICTION_HORIZON, gamma=1.0):
    warnings.filterwarnings('ignore')
    set_label = _get_set_label() if feature_override is None else f"custom ({len(feature_override)} features)"
    gamma_str = f", gamma={gamma}" if gamma < 1.0 else ""
    print(f"\n  Generating {horizon}h-ahead signals for {asset_name} "
          f"(models={'+'.join(model_names)}, window={window_size}h, "
          f"replay={replay_hours}h, {set_label}{gamma_str})...")

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

        sw = get_decay_weights(len(y_train), gamma)
        votes = []
        probas = []

        for model_name in model_names:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model = ALL_MODELS[model_name]()
                    model.fit(X_train_s, y_train, sample_weight=sw)
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
def _eval_one_config(features_np, labels_np, closes_np, combo, window, n, step, model_factories, pred_horizon=4, gamma=1.0):
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

    # Buy-and-hold PF tracking: record return at each test point
    bh_gains = 0.0
    bh_losses = 0.0
    prev_price = None

    for i in range(min_start, n, step):
        train_start = max(0, i - window)
        train_end = max(train_start, i - EMBARGO_CANDLES)  # embargo: gap between train and test
        X_train = features_np[train_start:train_end]
        y_train = labels_np[train_start:train_end]
        X_test  = features_np[i:i+1]
        y_true  = labels_np[i]

        # Buy-and-hold: track return from previous test point
        price = closes_np[i]
        if prev_price is not None:
            bh_ret_step = (price - prev_price) / prev_price
            if bh_ret_step > 0:
                bh_gains += bh_ret_step
            else:
                bh_losses += bh_ret_step
        prev_price = price

        if len(np.unique(y_train)) < 2:
            continue
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            continue
        mean = X_train.mean(axis=0)
        std  = X_train.std(axis=0)
        std[std == 0] = 1.0
        X_train_s = (X_train - mean) / std
        X_test_s  = (X_test - mean) / std
        sw = get_decay_weights(len(y_train), gamma)
        votes = []
        for model_name in combo:
            try:
                model = model_factories[model_name]()
                model.fit(X_train_s, y_train, sample_weight=sw)
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

    # CASCA V1.4: Adjusted Profit Factor = raw_PF / buyhold_PF
    # Measures genuine model skill by normalizing against market regime.
    # APF > 1.0 means model adds alpha over buy-and-hold.
    if trades < 3:
        raw_pf = 0.0
    elif total_loss == 0:
        raw_pf = min(total_gain * 100, 20.0)  # cap when no losses
    else:
        raw_pf = min(total_gain / abs(total_loss), 20.0)  # cap at 20.0

    # Buy-and-hold PF over same walk-forward test points
    if bh_losses == 0:
        bh_pf = max(bh_gains * 100, 1.0)  # floor at 1.0 to avoid division issues
    else:
        bh_pf = max(bh_gains / abs(bh_losses), 0.01)  # floor at 0.01

    adjusted_pf = raw_pf / bh_pf if raw_pf > 0 else 0.0

    # Return tuple: index 10 = adjusted_pf (combined_score), 11 = raw_pf, 12 = bh_pf
    return ('+'.join(combo), window, accuracy, total, cum_return, win_rate,
            trades, avg_gain, avg_loss, max_dd * 100, adjusted_pf, raw_pf, bh_pf)


DIAG_MODELS = _get_deku_diagnostic_models()


def run_diagnostic_for_asset(asset_name, df_features, feature_cols, gamma=1.0, resume=False, horizon=None):
    """Run diagnostic and return best config + all results.
    Kills the loky worker pool and removes BLAS limits before running,
    so RF/GB/LR get full BLAS parallelism in fresh workers.
    With resume=True, loads partial results from checkpoint and skips completed windows."""
    from joblib.externals.loky import get_reusable_executor

    # === KEY V5.4 OPTIMIZATION ===
    # Feature analysis ran with OMP=1 (good for LGBM-only workers).
    # Diagnostic uses RF/GB/LR which need BLAS threads.
    # 1. Kill existing loky pool (workers have OMP=1 baked in from import)
    # 2. Remove BLAS limits from parent env
    # 3. New workers will spawn fresh WITHOUT OMP=1
    get_reusable_executor().shutdown(wait=True)
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.pop(var, None)
    print(f"  [BLAS limits removed, worker pool reset for diagnostic]")

    combos = []
    model_names = list(ALL_MODELS.keys())
    for r in range(MIN_COMBO_SIZE, len(model_names) + 1):
        for combo in combinations(model_names, r):
            combos.append(list(combo))

    # Load partial results from checkpoint if resuming
    completed_windows = set()
    all_results = []
    if resume and horizon is not None:
        partial = _load_checkpoint(asset_name, horizon, 'diag_partial')
        if partial:
            all_results = partial.get('results', [])
            completed_windows = set(partial.get('completed_windows', []))
            # Convert stored results back to tuples
            all_results = [tuple(r) for r in all_results]
            print(f"  RESUME: loaded {len(all_results)} results from {len(completed_windows)} completed windows")

    # Build config batches per window (for progress + resume)
    window_batches = []
    for window in DIAG_WINDOWS:
        if window in completed_windows:
            continue
        batch = [(combo, window) for combo in combos]
        window_batches.append((window, batch))

    total_configs = len(DIAG_WINDOWS) * len(combos)
    done_configs = len(all_results)

    if not window_batches:
        print(f"  All {total_configs} configs already completed (resume)")
    else:
        remaining = sum(len(b) for _, b in window_batches)
        print(f"  {total_configs} configs total, {done_configs} done, {remaining} remaining ({N_JOBS_PARALLEL} workers, step={DIAG_STEP})")

    n = len(df_features)
    features_np = df_features[feature_cols].values.astype(np.float64)
    labels_np = df_features['label'].values.astype(np.int32)
    closes_np = df_features['close'].values.astype(np.float64)

    t_diag = time.time()
    best_score = max((r[10] for r in all_results), default=0)
    best_label = ''

    for batch_idx, (window, batch_configs) in enumerate(window_batches):
        t_batch = time.time()
        with _suppress_stderr():
            batch_results = Parallel(n_jobs=N_JOBS_PARALLEL, verbose=0)(
                delayed(_eval_one_config)(
                    features_np, labels_np, closes_np, combo, window, n, DIAG_STEP, DIAG_MODELS,
                    gamma=gamma
                )
                for combo, window in batch_configs
            )

        # Collect results from this batch
        batch_valid = 0
        for result in batch_results:
            if result is not None:
                all_results.append(result)
                batch_valid += 1
                if result[10] > best_score:
                    best_score = result[10]
                    best_label = f"{result[0]} w={result[1]}"

        done_configs += len(batch_configs)
        pct = done_configs / total_configs * 100
        elapsed = (time.time() - t_diag) / 60
        batch_time = (time.time() - t_batch) / 60

        # ETA calculation
        if done_configs < total_configs:
            rate = elapsed / done_configs if done_configs > 0 else 0
            eta = rate * (total_configs - done_configs)
            eta_str = f"ETA {eta:.0f}m"
        else:
            eta_str = "done"

        best_str = f"best APF={best_score:.2f} ({best_label})" if best_label else f"best APF={best_score:.2f}"
        print(f"  [{done_configs}/{total_configs}] {pct:.0f}% | w={window}h ({batch_valid} valid, {batch_time:.1f}m) | {best_str} | {eta_str}")

        # Save partial checkpoint after each window batch
        completed_windows.add(window)
        if horizon is not None:
            _save_checkpoint(asset_name, horizon, 'diag_partial', {
                'results': [list(r) for r in all_results],
                'completed_windows': list(completed_windows),
            })

    print(f"  Diagnostic completed in {(time.time() - t_diag)/60:.1f} minutes")

    # Restore BLAS limits for any subsequent feature analysis (e.g. next horizon)
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'
    get_reusable_executor().shutdown(wait=True)  # kill diagnostic workers too

    best_score = 0
    best_config = None
    sorted_results = []

    for result in all_results:
        if result is None:
            continue
        combo_name, window, acc, n_total, cum_ret, win_rate, trades, avg_gain, avg_loss, max_dd, adjusted_pf, raw_pf, bh_pf = result
        sorted_results.append(result)
        if adjusted_pf > best_score or (adjusted_pf == best_score and best_config and cum_ret > best_config.get('return_pct', -999)):
            best_score = adjusted_pf
            best_config = {
                'coin': asset_name,
                'best_window': window,
                'best_combo': combo_name,
                'accuracy': round(acc * 100, 2),
                'models': combo_name,
                'return_pct': round(cum_ret, 2),
                'win_rate': round(win_rate, 1),
                'trades': trades,
                'combined_score': round(adjusted_pf, 4),
                'raw_pf': round(raw_pf, 4),
                'bh_pf': round(bh_pf, 4),
            }
        print(f"    w={window:4d}h | {combo_name:20s} | acc={acc*100:5.1f}% "
              f"ret={cum_ret:+6.1f}% win={win_rate:4.0f}% "
              f"rawPF={raw_pf:.2f} bhPF={bh_pf:.2f} APF={adjusted_pf:.3f} (n={n_total})"
              f"{'  <-- BEST' if adjusted_pf == best_score else ''}")

    # ========================================================
    # CLEAR BEST MODEL RECOMMENDATION
    # ========================================================
    if best_config:
        # Top 5 by adjusted PF
        sorted_results.sort(key=lambda x: -x[10])  # index 10 = adjusted_pf
        top5 = sorted_results[:5]

        print()
        print("  " + "=" * 90)
        print(f"  |{'':4s}BEST MODEL for {asset_name:6s}{'':62s}|")
        print("  " + "=" * 90)
        print(f"  |{'':4s}Models:   {best_config['best_combo']:74s}|")
        w_str = f"{best_config['best_window']}h"
        print(f"  |{'':4s}Window:   {w_str:74s}|")
        acc_str = f"{best_config['accuracy']:.1f}%"
        print(f"  |{'':4s}Accuracy: {acc_str:74s}|")
        ret_str = f"{best_config['return_pct']:+.1f}% (after 0.09% fees)"
        print(f"  |{'':4s}Return:   {ret_str:74s}|")
        wr_str = f"{best_config['win_rate']:.0f}% ({best_config['trades']} trades)"
        print(f"  |{'':4s}Win Rate: {wr_str:74s}|")
        sc_str = f"APF={best_config['combined_score']:.4f}  (rawPF={best_config['raw_pf']:.2f} / bhPF={best_config['bh_pf']:.2f})"
        print(f"  |{''!s:4s}Score:    {sc_str:74s}|")
        print("  " + "-" * 90)
        print(f"  |{''!s:4s}{'Rank':5s}{'Combo':22s}{'Window':8s}{'Acc':7s}{'Return':9s}{'rawPF':7s}{'bhPF':7s}{'APF':7s}  |")
        print("  " + "-" * 90)
        for rank, result in enumerate(top5, 1):
            combo_name, window, acc, n_total, cum_ret, win_rate, trades, avg_gain, avg_loss, max_dd, adjusted_pf, raw_pf, bh_pf = result
            marker = " <--" if rank == 1 else ""
            print(f"  |{''!s:4s}{rank:<5d}{combo_name:22s}{window:5d}h  {acc*100:5.1f}%"
                  f"  {cum_ret:+6.1f}%  {raw_pf:5.2f}  {bh_pf:5.2f}  {adjusted_pf:5.3f}{marker:>4s} |")
        print("  " + "=" * 90)
        print(f"\n  >>> USE: models={best_config['best_combo']}, window={best_config['best_window']}h")
        print(f"  >>> Score = Adjusted PF (raw_PF / buyhold_PF) — APF > 1.0 = model adds alpha")
        print()

    # ========================================================
    # CSV EXPORT: all diagnostic results for analysis
    # ========================================================
    if sorted_results:
        import csv
        csv_path = f'{MODELS_DIR}/diagnostic_results_deku_{asset_name}.csv'
        sorted_for_csv = sorted(sorted_results, key=lambda x: (-x[10], -x[4]))  # APF desc, then return desc
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['coin', 'window', 'models', 'accuracy', 'return_pct', 'win_rate',
                             'trades', 'avg_gain', 'avg_loss', 'max_dd', 'raw_pf', 'buyhold_pf', 'adjusted_pf'])
            for r in sorted_for_csv:
                combo_name, window, acc, n_total, cum_ret, win_rate, trades, avg_gain, avg_loss, max_dd, adjusted_pf, raw_pf, bh_pf = r
                writer.writerow([asset_name, window, combo_name, round(acc*100, 2), round(cum_ret, 2),
                                 round(win_rate, 1), trades, round(avg_gain, 2), round(avg_loss, 2),
                                 round(max_dd, 2), round(raw_pf, 4), round(bh_pf, 4), round(adjusted_pf, 4)])
        print(f"  Diagnostic results saved: {csv_path}")

    return best_config, sorted_results


# ============================================================
# CHART DATA EXPORT
# ============================================================
def generate_strategy_html(asset_name, signals_short, signals_long, strategy='both_agree'):
    """
    Generate interactive HTML charts (Plotly) with 4 panels:
    1. Price + short-horizon signals (blue=BUY, red=SELL)
    2. Price + long-horizon signals
    3. Price + combined strategy signals
    4. Portfolio equity ($1000 start) vs buy & hold
    Zoom synced across all panels. Colorblind-friendly.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    h_short, h_long = HORIZON_SHORT, HORIZON_LONG

    sig_s_map = {s['datetime']: s for s in (signals_short or [])}
    sig_l_map = {s['datetime']: s for s in (signals_long or [])}
    all_times = sorted(set(list(sig_s_map.keys()) + list(sig_l_map.keys())))

    if not all_times:
        print("  No signals to chart.")
        return

    merged = []
    for dt in all_times:
        ss = sig_s_map.get(dt)
        sl = sig_l_map.get(dt)
        price = (ss or sl)['close']
        sig_s = ss['signal'] if ss else 'HOLD'
        conf_s = ss['confidence'] if ss else 50
        sig_l = sl['signal'] if sl else 'HOLD'
        conf_l = sl['confidence'] if sl else 50

        # Combined signal based on strategy
        if sig_s == 'SELL' or sig_l == 'SELL':
            combined = 'SELL'
        elif strategy == 'both_agree':
            if sig_s == 'BUY' and sig_l == 'BUY' and conf_s >= MIN_CONFIDENCE and conf_l >= MIN_CONFIDENCE:
                combined = 'BUY'
            else:
                combined = 'HOLD'
        else:  # either
            if (sig_s == 'BUY' and conf_s >= MIN_CONFIDENCE) or (sig_l == 'BUY' and conf_l >= MIN_CONFIDENCE):
                combined = 'BUY'
            else:
                combined = 'HOLD'

        merged.append({
            'datetime': dt, 'close': price,
            'sig_s': sig_s, 'conf_s': conf_s,
            'sig_l': sig_l, 'conf_l': conf_l,
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

        b4x, b4y, b4t, s4x, s4y, s4t = _markers(data, 'sig_s', 'conf_s')
        b8x, b8y, b8t, s8x, s8y, s8t = _markers(data, 'sig_l', 'conf_l')
        cbx, cby, _, csx, csy, _ = _markers(data, 'combined', 'conf_s')
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
  {{x:{json.dumps(b4x)},y:{json.dumps(b4y)},mode:'markers',name:'{h_short}h BUY',text:{json.dumps(b4t)},marker:{{color:blue,symbol:'triangle-up',size:8}}}},
  {{x:{json.dumps(s4x)},y:{json.dumps(s4y)},mode:'markers',name:'{h_short}h SELL',text:{json.dumps(s4t)},marker:{{color:red,symbol:'triangle-down',size:8}}}}
],ml('{h_short}h Model',240),{{responsive:true}});

Plotly.newPlot('c8h',[pl,
  {{x:{json.dumps(b8x)},y:{json.dumps(b8y)},mode:'markers',name:'{h_long}h BUY',text:{json.dumps(b8t)},marker:{{color:blue,symbol:'triangle-up',size:8}}}},
  {{x:{json.dumps(s8x)},y:{json.dumps(s8y)},mode:'markers',name:'{h_long}h SELL',text:{json.dumps(s8t)},marker:{{color:red,symbol:'triangle-down',size:8}}}}
],ml('{h_long}h Model',240),{{responsive:true}});

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


def generate_signal_table_html(asset_name, signals_short, signals_long, strategy='both_agree'):
    """Generate interactive HTML table: price, price+1h, delta, signals, strategy, correct."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    h_short, h_long = HORIZON_SHORT, HORIZON_LONG

    sig_s_map = {s['datetime']: s for s in (signals_short or [])}
    sig_l_map = {s['datetime']: s for s in (signals_long or [])}
    all_times = sorted(set(list(sig_s_map.keys()) + list(sig_l_map.keys())))
    if not all_times:
        return
    all_times = all_times[-168:]

    rows = []
    for i, dt in enumerate(all_times):
        ss = sig_s_map.get(dt)
        sl = sig_l_map.get(dt)
        price = (ss or sl)['close']

        # Price +1h = actual price at next timestamp
        if i + 1 < len(all_times):
            next_dt = all_times[i + 1]
            next_s = sig_s_map.get(next_dt) or sig_l_map.get(next_dt)
            price_next = next_s['close'] if next_s else price
        else:
            price_next = price

        delta = ((price_next - price) / price * 100) if price > 0 else 0

        sig_s = ss['signal'] if ss else 'N/A'
        conf_s = ss['confidence'] if ss else 0
        sig_l = sl['signal'] if sl else 'N/A'
        conf_l = sl['confidence'] if sl else 0

        # Combined strategy
        if sig_s == 'SELL' or sig_l == 'SELL':
            combined = 'SELL'
        elif strategy == 'both_agree':
            if sig_s == 'BUY' and sig_l == 'BUY' and conf_s >= MIN_CONFIDENCE and conf_l >= MIN_CONFIDENCE:
                combined = 'BUY'
            else:
                combined = 'HOLD'
        else:
            if (sig_s == 'BUY' and conf_s >= MIN_CONFIDENCE) or (sig_l == 'BUY' and conf_l >= MIN_CONFIDENCE):
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
            'sig_s': sig_s, 'conf_s': conf_s,
            'sig_l': sig_l, 'conf_l': conf_l,
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
  <td class="{_sc(r['sig_s'])}">{r['sig_s']}</td>
  <td>{r['conf_s']:.0f}%</td>
  <td class="{_sc(r['sig_l'])}">{r['sig_l']}</td>
  <td>{r['conf_l']:.0f}%</td>
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
  <th onclick="sortTable(6)">{h_short}h Signal</th>
  <th onclick="sortTable(7)">{h_short}h Conf</th>
  <th onclick="sortTable(8)">{h_long}h Signal</th>
  <th onclick="sortTable(9)">{h_long}h Conf</th>
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

    if not os.path.exists(_get_models_csv_path()):
        print("\nERROR: crypto_casca_best_models.csv not found!")
        print("Please run Mode D first to find best models.")
        return

    df_best = pd.read_csv(_get_models_csv_path())
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

            row_gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0

            model_info = {
                'best_combo': row['best_combo'],
                'best_window': window,
                'accuracy': row['accuracy'],
                'feature_set': fs,
                'horizon': h,
                'n_features': int(row['n_features']) if 'n_features' in row.index and pd.notna(row.get('n_features')) else '',
            }

            signals = generate_signals(asset_name, model_names, window, REPLAY_HOURS,
                                       feature_override=feature_override, horizon=h, gamma=row_gamma)
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

                row_gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0
                print(f"  Generating {h}h signals (720h) for {asset_name}...")
                sigs = generate_signals(asset_name, model_names, window, 720,
                                        feature_override=feature_override, horizon=h, gamma=row_gamma)
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


# ============================================================
# DEKU: OPTUNA-BASED JOINT OPTIMIZATION
# Replaces CASCA's sequential feature analysis + grid diagnostic + gamma sweep
# with a single Optuna study that jointly optimizes all hyperparameters.
# ============================================================
DEKU_DEFAULT_TRIALS = 150
DEKU_PRUNING_WARMUP = 8  # minimum walk-forward steps before pruning kicks in


def _deku_eval_with_pruning(features_np, labels_np, closes_np, combo, window, n,
                             step, model_factories, gamma=1.0, trial=None,
                             horizon=PREDICTION_HORIZON):
    """Walk-forward evaluation with optional Optuna pruning.
    Same logic as _eval_one_config but reports intermediate scores for Hyperband.
    Runs in the main process (not joblib worker) so models can use all cores."""
    min_start = window + 50
    if n < min_start + 50:
        return None

    correct = 0
    total = 0
    portfolio = 1.0
    in_position = False
    entry_price = 0
    trades = 0
    wins = 0
    peak = 1.0
    max_dd = 0.0
    total_gain = 0.0
    total_loss = 0.0

    # Buy-and-hold PF tracking
    bh_gains = 0.0
    bh_losses = 0.0
    prev_price = None

    step_idx = 0

    for i in range(min_start, n, step):
        train_start = max(0, i - window)
        train_end = max(train_start, i - EMBARGO_CANDLES)  # embargo: gap between train and test
        X_train = features_np[train_start:train_end]
        y_train = labels_np[train_start:train_end]
        X_test = features_np[i:i+1]
        y_true = labels_np[i]

        price = closes_np[i]
        if prev_price is not None:
            bh_ret_step = (price - prev_price) / prev_price
            if bh_ret_step > 0:
                bh_gains += bh_ret_step
            else:
                bh_losses += bh_ret_step
        prev_price = price

        if len(np.unique(y_train)) < 2:
            step_idx += 1
            continue
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            step_idx += 1
            continue

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0
        X_train_s = (X_train - mean) / std
        X_test_s = (X_test - mean) / std
        sw = get_decay_weights(len(y_train), gamma)

        votes = []
        for model_name in combo:
            try:
                model = model_factories[model_name]()
                model.fit(X_train_s, y_train, sample_weight=sw)
                pred = model.predict(X_test_s)[0]
                votes.append(pred)
            except Exception:
                continue
        if not votes:
            step_idx += 1
            continue

        buy_ratio = sum(votes) / len(votes)
        ensemble_pred = 1 if buy_ratio > 0.5 else 0

        if ensemble_pred == y_true:
            correct += 1
        total += 1

        # Portfolio logic
        if ensemble_pred == 1 and not in_position:
            in_position = True
            entry_price = price * (1 + TRADING_FEE)
        elif ensemble_pred == 0 and in_position:
            sell_price = price * (1 - TRADING_FEE)
            trade_return = (sell_price - entry_price) / entry_price
            portfolio *= (1 + trade_return)
            trades += 1
            if trade_return > 0:
                wins += 1
                total_gain += trade_return
            else:
                total_loss += trade_return
            in_position = False

        # Max drawdown tracking
        current_val = portfolio * (price / entry_price) if in_position else portfolio
        if current_val > peak:
            peak = current_val
        dd = (peak - current_val) / peak
        if dd > max_dd:
            max_dd = dd

        step_idx += 1

        # Hyperband pruning: report intermediate APF after warmup
        if trial is not None and step_idx >= DEKU_PRUNING_WARMUP:
            # Use cumulative return as intermediate metric (APF needs trades)
            intermediate = (portfolio - 1.0) * 100
            trial.report(intermediate, step_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Close open position at end
    if in_position and total > 0:
        last_price = closes_np[n - 1]
        sell_price = last_price * (1 - TRADING_FEE)
        trade_return = (sell_price - entry_price) / entry_price
        portfolio *= (1 + trade_return)
        trades += 1
        if trade_return > 0:
            wins += 1
            total_gain += trade_return
        else:
            total_loss += trade_return

    if total == 0:
        return None

    accuracy = correct / total
    cum_return = (portfolio - 1.0) * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0

    # APF scoring
    if trades < 3:
        raw_pf = 0.0
    elif total_loss == 0:
        raw_pf = min(total_gain * 100, 20.0)
    else:
        raw_pf = min(total_gain / abs(total_loss), 20.0)

    if bh_losses == 0:
        bh_pf = max(bh_gains * 100, 1.0)
    else:
        bh_pf = max(bh_gains / abs(bh_losses), 0.01)

    adjusted_pf = raw_pf / bh_pf if raw_pf > 0 else 0.0

    return ('+'.join(combo), window, accuracy, total, cum_return, win_rate,
            trades, total_gain, total_loss, max_dd * 100, adjusted_pf, raw_pf, bh_pf)


def _build_cpcv_paths(n_total, n_groups=6, k_test=2, purge=CPCV_PURGE_CANDLES):
    """
    Build CPCV (Combinatorial Purged Cross-Validation) paths.

    Splits n_total rows into n_groups contiguous groups.
    For each C(n_groups, k_test) combination of test groups:
      - test = union of selected groups
      - train = remaining groups, with purge rows removed at boundaries

    Returns list of (train_indices, test_indices) tuples — one per path.
    C(6,2) = 15 paths.
    """
    group_size = n_total // n_groups
    group_boundaries = []
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else n_total
        group_boundaries.append((start, end))

    paths = []
    for test_combo in combinations(range(n_groups), k_test):
        test_set = set()
        for g in test_combo:
            s, e = group_boundaries[g]
            test_set.update(range(s, e))

        # Build train set: all rows NOT in test, with purge around test boundaries
        purge_set = set()
        for g in test_combo:
            s, e = group_boundaries[g]
            # Purge rows just before and after each test group
            for p in range(max(0, s - purge), s):
                purge_set.add(p)
            for p in range(e, min(n_total, e + purge)):
                purge_set.add(p)

        train_indices = sorted([i for i in range(n_total) if i not in test_set and i not in purge_set])
        test_indices = sorted(test_set)

        paths.append((np.array(train_indices), np.array(test_indices)))

    return paths


def _compute_pbo(path_returns):
    """
    Compute Probability of Backtest Overfitting (PBO).

    PBO = fraction of CPCV paths where the strategy lost money (return <= 0).
    Lower is better. PBO > 50% means more paths lose money than gain.
    """
    if not path_returns:
        return 1.0
    n_losing = sum(1 for r in path_returns if r <= 0)
    return n_losing / len(path_returns)


def _build_combo_list():
    """Build all ensemble combos from Deku's 5 base models (MIN_COMBO_SIZE=2)."""
    model_names = list(ALL_MODELS.keys())
    combos = []
    for r in range(MIN_COMBO_SIZE, len(model_names) + 1):
        for combo in combinations(model_names, r):
            combos.append('+'.join(combo))
    return combos


def run_mode_d_optuna(assets_list, horizon=PREDICTION_HORIZON, n_trials=DEKU_DEFAULT_TRIALS, resume=False):
    """
    DEKU V1.4 Mode D: RANDOM OPTUNA + ADAPTIVE CPCV.

    Pipeline:
    1. Download fresh data
    2. Build all features (no fixed cap — data_months is an Optuna param)
    3. LGBM importance ranking (~1 min)
    4. Optuna study (RandomSampler, fixed gamma=1.0, data_months as param)
    5. Adaptive CPCV validation with PBO filter on top candidates
    6. Save best model to CSV
    7. Calibration summary (top trials analysis)
    8. Generate signals + backtest charts
    """
    t_mode_start = time.time()
    _kill_orphan_workers()

    combo_options = _build_combo_list()

    print("\n" + "=" * 70)
    print(f"  DEKU V1.4 MODE D: RANDOM OPTUNA + ADAPTIVE CPCV -- {horizon}h HORIZON")
    print(f"  Models: {', '.join(ALL_MODELS.keys())} ({len(combo_options)} combos)")
    print(f"  Trials: {n_trials} (RandomSampler) | gamma=1.0 (fixed)")
    print(f"  data_months: {DATA_MONTHS_OPTIONS} | CPCV: adaptive (4 or 6 groups) | PBO ≤ {PBO_MAX*100:.0f}%")
    metric_label = f" | metric={OPTUNA_METRIC}" if OPTUNA_METRIC != 'apf' else ""
    print(f"  Search: combo × window × data_months × n_features{metric_label}")
    print("=" * 70)

    # Download fresh data
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

    best_models = []

    for asset_name in assets_list:
        t_asset = time.time()

        print(f"\n{'='*70}")
        print(f"  DEKU OPTIMIZATION: {asset_name} ({horizon}h)")
        print(f"{'='*70}")

        df_raw = load_data(asset_name)
        if df_raw is None:
            continue

        # Step 1: Build ALL features (no fixed cap — data_months is an Optuna param)
        max_data_hours = max(DATA_MONTHS_OPTIONS) * 720  # keep enough for largest data_months option
        print(f"\n  Building all features (horizon={horizon}h)...")
        t0 = time.time()
        df_full, all_cols = build_all_features(df_raw, asset_name=asset_name, horizon=horizon)
        print(f"  [Feature build: {(time.time()-t0)/60:.1f} min]")

        total_rows = len(df_full)
        if total_rows > max_data_hours:
            df_full = df_full.tail(max_data_hours).reset_index(drop=True)
            print(f"  Capped: {total_rows:,} -> {len(df_full):,} rows (last {max(DATA_MONTHS_OPTIONS)}mo)")

        df_clean = df_full.dropna(subset=all_cols + ['label']).reset_index(drop=True)
        print(f"  Clean data: {len(df_clean):,} rows, {len(all_cols)} features")

        if len(df_clean) < 500:
            print(f"  Not enough data ({len(df_clean)} rows). Need 500+. Skipping.")
            continue

        # Step 2: LGBM importance ranking (fast, ~1 min)
        # Replaces full 5-test feature analysis — Optuna handles n_features jointly
        print(f"\n  LGBM feature importance ranking...")
        t0 = time.time()
        importance_df = _test_lgbm_importance(df_clean, all_cols, gamma=1.0)
        ranked_features = importance_df['feature'].tolist()
        print(f"  [Ranking: {(time.time()-t0)/60:.1f} min] — {len(ranked_features)} features ranked")

        # Prepare data for Optuna — columns in rank order
        df_optuna = df_clean.dropna(subset=ranked_features + ['label']).reset_index(drop=True)
        features_np_all = df_optuna[ranked_features].values.astype(np.float64)
        labels_np_all = df_optuna['label'].values.astype(np.int32)
        closes_np_all = df_optuna['close'].values.astype(np.float64)
        n_total = len(df_optuna)

        # CPCV is built adaptively after Optuna (depends on winner's data_months)
        print(f"  CPCV: adaptive (4 or 6 groups based on winner's data_months) | Purge: {CPCV_PURGE_CANDLES} candles")

        # Optuna trains on data sliced per trial's data_months
        features_np = features_np_all
        labels_np = labels_np_all
        closes_np = closes_np_all
        n_full = len(features_np)

        print(f"  Full data: {n_full:,} rows | data_months slicing per trial | CPCV validation after")

        # Per-horizon feature range
        feat_min, feat_max = N_FEATURES_RANGE.get(horizon, N_FEATURES_RANGE_DEFAULT)
        min_n_features = feat_min
        max_n_features = min(len(ranked_features), feat_max)

        # Step 3: Optuna study (RandomSampler, fixed gamma=1.0, data_months as param)
        print(f"\n{'='*70}")
        print(f"  OPTUNA: {asset_name} {horizon}h")
        print(f"  Search space: {len(combo_options)} combos × {len(DIAG_WINDOWS)} windows × data_months{DATA_MONTHS_OPTIONS} × features[{min_n_features}-{max_n_features}]")
        print(f"  gamma=1.0 (fixed) | {n_trials} random trials")
        print(f"{'='*70}")

        # Suppress Optuna's default logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # RandomSampler always (no TPE)
        sampler = optuna.samplers.RandomSampler(seed=42)

        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=DEKU_PRUNING_WARMUP,
                max_resource=n_full // DIAG_STEP,
                reduction_factor=3,
            ),
            study_name=f'deku_v14_{asset_name}_{horizon}h',
        )

        model_factories = _get_deku_diagnostic_models()
        best_apf_so_far = 0.0
        trial_count = 0

        def objective(trial):
            nonlocal best_apf_so_far, trial_count
            trial_count += 1

            combo_name = trial.suggest_categorical('combo', combo_options)
            window = trial.suggest_categorical('window', DIAG_WINDOWS)
            data_months = trial.suggest_categorical('data_months', DATA_MONTHS_OPTIONS)
            n_feat = trial.suggest_int('n_features', min_n_features, max_n_features)
            gamma = 1.0  # fixed — no decay

            data_hours = data_months * 720  # approximate hours per month

            combo = combo_name.split('+')

            # Slice data to data_months worth of history
            if n_full > data_hours:
                trial_feat = features_np[-data_hours:, :n_feat]
                trial_labels = labels_np[-data_hours:]
                trial_closes = closes_np[-data_hours:]
                n = data_hours
            else:
                trial_feat = features_np[:, :n_feat]
                trial_labels = labels_np
                trial_closes = closes_np
                n = n_full

            result = _deku_eval_with_pruning(
                trial_feat, trial_labels, trial_closes, combo, window, n,
                DIAG_STEP, model_factories, gamma=gamma, trial=trial,
                horizon=horizon
            )

            if result is None:
                return 0.0

            trades = result[6]

            # Reject configs with too few trades — not statistically reliable
            MIN_TRADES = 8
            if trades < MIN_TRADES:
                return 0.0

            score = _compute_optuna_score(result)
            ret = result[4]

            if score > best_apf_so_far:
                best_apf_so_far = score
                print(f"  #{trial_count:3d} NEW BEST: {combo_name:22s} w={window:4d}h "
                      f"dm={data_months}mo f={n_feat:3d} | {OPTUNA_METRIC}={score:.3f} ret={ret:+.1f}% "
                      f"rawPF={result[11]:.2f} bhPF={result[12]:.2f} trades={trades}")
            elif trial_count % 20 == 0:
                print(f"  #{trial_count:3d} progress: {OPTUNA_METRIC}={score:.3f} | best so far: {best_apf_so_far:.3f}")

            return score

        t_optuna = time.time()

        print(f"\n  {n_trials} random trials...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        optuna_elapsed = (time.time() - t_optuna) / 60

        # Results summary
        best_trial = study.best_trial
        print(f"\n  {'='*70}")
        print(f"  OPTUNA RESULTS: {asset_name} {horizon}h ({optuna_elapsed:.1f} min)")
        print(f"  {'='*70}")
        print(f"  Best trial: #{best_trial.number}")
        print(f"  {OPTUNA_METRIC.upper():10s} {best_trial.value:.4f}")
        print(f"  Combo:      {best_trial.params['combo']}")
        print(f"  Window:     {best_trial.params['window']}h")
        print(f"  Data_months:{best_trial.params['data_months']}mo")
        print(f"  Gamma:      1.0 (fixed)")
        print(f"  N_features: {best_trial.params['n_features']}")

        # Stats
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"\n  Trials: {n_complete} completed, {n_pruned} pruned ({n_pruned/(n_complete+n_pruned)*100:.0f}% pruned)")

        # Top 10 trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value > 0]
        completed_trials.sort(key=lambda t: -t.value)
        print(f"\n  {'Rank':>4s}  {OPTUNA_METRIC.upper():>7s}  {'Combo':22s}  {'Window':>6s}  {'DM':>4s}  {'Feats':>5s}")
        print(f"  {'-'*60}")
        for i, t in enumerate(completed_trials[:10], 1):
            marker = " <-- BEST" if i == 1 else ""
            print(f"  {i:4d}  {t.value:7.3f}  {t.params['combo']:22s}  {t.params['window']:5d}h  "
                  f"{t.params['data_months']:3d}m  {t.params['n_features']:5d}{marker}")

        # Parameter importance (if enough trials)
        if n_complete >= 20:
            try:
                importances = optuna.importance.get_param_importances(study)
                print(f"\n  Parameter importance:")
                for param, imp in importances.items():
                    bar = '#' * int(imp * 40)
                    print(f"    {param:15s} {imp*100:5.1f}% {bar}")
            except Exception:
                pass

        # ── ADAPTIVE CPCV VALIDATION ──
        # Deduplicate by (combo, window, data_months) — keep best per unique config
        seen_configs = set()
        unique_candidates = []
        for t in completed_trials:
            key = (t.params['combo'], t.params['window'], t.params['data_months'])
            if key not in seen_configs:
                seen_configs.add(key)
                unique_candidates.append(t)

        # Select top 10 deduped + best per unexplored combo
        cand_top = unique_candidates[:10]
        combos_in_top = set(t.params['combo'] for t in cand_top)
        cand_extra = []
        seen_extra = set()
        for t in unique_candidates:
            c = t.params['combo']
            if c not in combos_in_top and c not in seen_extra:
                seen_extra.add(c)
                cand_extra.append(t)
        candidates = cand_top + cand_extra
        N_CANDIDATES = min(20, len(candidates))

        # Use winner's data_months for adaptive CPCV
        winner_data_months = study.best_trial.params['data_months']
        cpcv_n_groups, cpcv_k_test = _get_cpcv_params(winner_data_months)
        winner_data_hours = winner_data_months * 720

        # Cap CPCV data to winner's data_months
        if n_full > winner_data_hours:
            cpcv_features = features_np_all[-winner_data_hours:]
            cpcv_labels = labels_np_all[-winner_data_hours:]
            cpcv_closes = closes_np_all[-winner_data_hours:]
            n_cpcv = winner_data_hours
        else:
            cpcv_features = features_np_all
            cpcv_labels = labels_np_all
            cpcv_closes = closes_np_all
            n_cpcv = n_full

        cpcv_paths = _build_cpcv_paths(n_cpcv, cpcv_n_groups, cpcv_k_test, CPCV_PURGE_CANDLES)
        n_paths = len(cpcv_paths)

        print(f"\n  {'='*70}")
        print(f"  ADAPTIVE CPCV VALIDATION: {asset_name} {horizon}h ({N_CANDIDATES} candidates × {n_paths} paths)")
        print(f"  data_months={winner_data_months} → {cpcv_n_groups} groups, k={cpcv_k_test} → C({cpcv_n_groups},{cpcv_k_test})={n_paths} paths")
        print(f"  CPCV data: {n_cpcv:,} rows | PBO threshold: ≤ {PBO_MAX*100:.0f}% | Winner by: median return")
        print(f"  {'='*70}")

        cpcv_results = []
        min_test_rows = min(len(te) for _, te in cpcv_paths)
        if min_test_rows >= 100:
            CPCV_STEP = 12  # finer step for CPCV paths

            for ci, trial in enumerate(candidates):
                c_combo = trial.params['combo'].split('+')
                c_window = trial.params['window']
                c_gamma = 1.0  # fixed
                c_n_feat = trial.params['n_features']
                c_data_months = trial.params['data_months']

                # Cap candidate data to its own data_months
                c_data_hours = c_data_months * 720
                if n_full > c_data_hours:
                    c_feat_all = features_np_all[-c_data_hours:]
                    c_labels_all = labels_np_all[-c_data_hours:]
                    c_closes_all = closes_np_all[-c_data_hours:]
                    c_n_total = c_data_hours
                else:
                    c_feat_all = features_np_all
                    c_labels_all = labels_np_all
                    c_closes_all = closes_np_all
                    c_n_total = n_full

                # Build CPCV paths for this candidate's data_months
                c_cpcv_n_groups, c_cpcv_k_test = _get_cpcv_params(c_data_months)
                c_cpcv_paths = _build_cpcv_paths(c_n_total, c_cpcv_n_groups, c_cpcv_k_test, CPCV_PURGE_CANDLES)

                path_rets = []
                path_scores = []
                path_trades = []

                for pi, (train_idx, test_idx) in enumerate(c_cpcv_paths):
                    # Use test partition only — walk-forward within test set
                    p_feat = c_feat_all[test_idx][:, :c_n_feat]
                    p_labels = c_labels_all[test_idx]
                    p_closes = c_closes_all[test_idx]
                    n_p = len(test_idx)

                    p_result = _deku_eval_with_pruning(
                        p_feat, p_labels, p_closes,
                        c_combo, c_window, n_p,
                        CPCV_STEP, model_factories, gamma=c_gamma, trial=None,
                        horizon=horizon
                    )

                    if p_result:
                        path_rets.append(p_result[4])
                        path_scores.append(_compute_optuna_score(p_result))
                        path_trades.append(p_result[6])
                    else:
                        path_rets.append(0.0)
                        path_scores.append(0.0)
                        path_trades.append(0)

                median_ret = float(np.median(path_rets))
                median_score = float(np.median(path_scores))
                pbo = _compute_pbo(path_rets)
                total_trades = sum(path_trades)
                n_positive = sum(1 for r in path_rets if r > 0)
                c_n_paths = len(c_cpcv_paths)

                cpcv_results.append({
                    'trial': trial,
                    'median_ret': median_ret,
                    'median_score': median_score,
                    'pbo': pbo,
                    'total_trades': total_trades,
                    'n_positive': n_positive,
                    'n_paths': c_n_paths,
                    'path_rets': path_rets,
                    'is_score': trial.value,
                })

                if (ci + 1) % 5 == 0:
                    print(f"    Evaluated {ci+1}/{N_CANDIDATES} candidates...")

            # Sort by median return (descending)
            cpcv_results.sort(key=lambda x: -x['median_ret'])

            # Display results
            print(f"\n  {'Rank':>4s}  {'Med_Ret':>8s}  {'PBO':>5s}  {'Pos':>5s}  "
                  f"{'Tr':>4s}  {'IS_'+OPTUNA_METRIC.upper():>8s}  {'Combo':22s}  {'Win':>4s}  {'DM':>4s}  {'F':>3s}  {'Status':>8s}")
            print(f"  {'-'*110}")
            for i, r in enumerate(cpcv_results[:15], 1):
                t = r['trial']
                passed = r['pbo'] <= PBO_MAX
                status = "PASS" if passed else "REJECT"
                marker = " <-- BEST" if i == 1 and passed else ""
                icon = "✓" if passed else "✗"
                c_n_paths = r['n_paths']
                print(f"  {i:4d}  {r['median_ret']:+7.1f}%  {r['pbo']*100:4.0f}%  {r['n_positive']:3d}/{c_n_paths:2d}  "
                      f"{r['total_trades']:4d}  {r['is_score']:8.3f}  {t.params['combo']:22s}  {t.params['window']:3d}h  "
                      f"{t.params['data_months']:3d}m  {t.params['n_features']:3d}  {icon} {status}{marker}")
        else:
            print(f"  CPCV skipped: smallest test partition only {min_test_rows} rows (need 100+)")

        # Pick winner: best median return among PBO-passing candidates
        passing = [r for r in cpcv_results if r['pbo'] <= PBO_MAX and r['median_ret'] > 0]
        if passing:
            winner = passing[0]
            winner_trial = winner['trial']
            winner_cpcv = winner
        else:
            # Fallback: best in-sample if no CPCV candidate passes
            winner_trial = study.best_trial
            winner_cpcv = None
            if cpcv_results:
                print(f"\n  WARNING: No candidate passed PBO ≤ {PBO_MAX*100:.0f}%. Using best in-sample trial.")

        best_combo = winner_trial.params['combo'].split('+')
        best_window = winner_trial.params['window']
        best_gamma = 1.0  # fixed
        best_n_feat = winner_trial.params['n_features']
        best_data_months = winner_trial.params['data_months']
        best_features = ranked_features[:best_n_feat]

        # Re-run winner on its data slice for in-sample metrics
        best_data_hours = best_data_months * 720
        if n_full > best_data_hours:
            win_feat = features_np[-best_data_hours:, :best_n_feat]
            win_labels = labels_np[-best_data_hours:]
            win_closes = closes_np[-best_data_hours:]
            win_n = best_data_hours
        else:
            win_feat = features_np[:, :best_n_feat]
            win_labels = labels_np
            win_closes = closes_np
            win_n = n_full
        full_result = _deku_eval_with_pruning(
            win_feat, win_labels, win_closes, best_combo, best_window, win_n,
            DIAG_STEP, model_factories, gamma=best_gamma, trial=None,
            horizon=horizon
        )

        if full_result:
            combo_name, window, acc, n_total_r, cum_ret, win_rate, trades, _, _, max_dd, apf, raw_pf, bh_pf = full_result
            is_score = _compute_optuna_score(full_result)

            # CPCV metrics
            cpcv_med_ret = winner_cpcv['median_ret'] if winner_cpcv else 0
            cpcv_pbo = winner_cpcv['pbo'] if winner_cpcv else 1.0
            cpcv_n_pos = winner_cpcv['n_positive'] if winner_cpcv else 0
            cpcv_path_rets = winner_cpcv['path_rets'] if winner_cpcv else []

            best_config = {
                'coin': asset_name,
                'best_window': best_window,
                'best_combo': winner_trial.params['combo'],
                'accuracy': round(acc * 100, 2),
                'models': winner_trial.params['combo'],
                'return_pct': round(cpcv_med_ret if winner_cpcv else cum_ret, 2),
                'win_rate': round(win_rate, 1),
                'trades': trades,
                'combined_score': round(is_score, 4),
                'feature_set': 'D',
                'n_features': best_n_feat,
                'optimal_features': ','.join(best_features),
                'horizon': horizon,
                'gamma': 1.0,
                'data_months': best_data_months,
                'pbo': round(cpcv_pbo, 4),
                'cpcv_median_ret': round(cpcv_med_ret, 2),
                'cpcv_positive_paths': cpcv_n_pos,
            }
            best_models.append(best_config)

            pbo_icon = "✓" if cpcv_pbo <= PBO_MAX else "✗"
            cpcv_n_groups_w, cpcv_k_test_w = _get_cpcv_params(best_data_months)
            n_paths_w = winner_cpcv['n_paths'] if winner_cpcv and 'n_paths' in winner_cpcv else n_paths

            print(f"\n  {'='*70}")
            print(f"  WINNER (Optuna + CPCV): {asset_name} {horizon}h  {pbo_icon}")
            print(f"  Models: {winner_trial.params['combo']}  Window: {best_window}h  data_months: {best_data_months}mo  Features: {best_n_feat}")
            print(f"  gamma=1.0 (fixed) | CPCV: {cpcv_n_groups_w} groups, k={cpcv_k_test_w}")
            print(f"  In-sample:  {OPTUNA_METRIC}={is_score:.3f}  ret={cum_ret:+.1f}%  acc={acc*100:.1f}%  trades={trades}  rawPF={raw_pf:.2f}")
            if winner_cpcv:
                path_str = " / ".join(f"{r:+.1f}%" for r in cpcv_path_rets[:5])
                if len(cpcv_path_rets) > 5:
                    path_str += f" ... ({len(cpcv_path_rets)} total)"
                print(f"  CPCV:       median_ret={cpcv_med_ret:+.1f}%  PBO={cpcv_pbo*100:.0f}%  positive={cpcv_n_pos}/{n_paths_w} paths")
                print(f"  Path returns: [{path_str}]")
            print(f"  {'='*70}")

            # ── Calibration summary ──
            completed_trials = [t for t in study.trials if t.value is not None and t.value > 0]
            top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:20]

            if top_trials:
                from collections import Counter
                print(f"\n  {'═'*70}")
                print(f"  CALIBRATION SUMMARY: {asset_name} {horizon}h")
                print(f"  {'═'*70}")

                print(f"  Top 10 trials by {OPTUNA_METRIC}:")
                for rank, t in enumerate(top_trials[:10], 1):
                    p = t.params
                    print(f"    #{rank:2d}  combo={p['combo']:22s}  w={p['window']:4d}h  "
                          f"dm={p['data_months']}mo  feats={p['n_features']:3d}  "
                          f"{OPTUNA_METRIC}={t.value:.4f}")

                combo_counts = Counter(t.params['combo'] for t in top_trials)
                combo_str = ' | '.join(f"{c}: {n}x" for c, n in combo_counts.most_common())
                print(f"\n  Combo frequency (top {len(top_trials)} trials):")
                print(f"    {combo_str}")

                window_counts = Counter(t.params['window'] for t in top_trials)
                window_str = ' | '.join(f"{w}h: {n}x" for w, n in sorted(window_counts.most_common(), key=lambda x: -x[1]))
                print(f"\n  Window frequency (top {len(top_trials)} trials):")
                print(f"    {window_str}")

                dm_counts = Counter(t.params['data_months'] for t in top_trials)
                dm_str = ' | '.join(f"{d}mo: {n}x" for d, n in sorted(dm_counts.most_common(), key=lambda x: -x[1]))
                print(f"\n  Data_months frequency (top {len(top_trials)} trials):")
                print(f"    {dm_str}")

                feats = [t.params['n_features'] for t in top_trials]
                print(f"  Feature range (top {len(top_trials)} trials): "
                      f"{min(feats)} – {max(feats)} (median: {sorted(feats)[len(feats)//2]})")
                print(f"  {'═'*70}")

        print(f"  [{asset_name} total: {(time.time()-t_asset)/60:.1f} min]")

    if not best_models:
        print("\nNo results. Aborting.")
        return

    # Save best models (merge with existing horizons)
    _backup_models_csv()
    csv_path = _get_models_csv_path()
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
    print(f"\n  Best models saved: {csv_path}")

    # Generate signals + charts for best models
    for m in best_models:
        features_list = m['optimal_features'].split(',')
        model_names = m['models'].split('+')
        generate_signals(m['coin'], model_names, m['best_window'],
                         feature_override=features_list, horizon=horizon,
                         gamma=m.get('gamma', 1.0))

    elapsed = (time.time() - t_mode_start) / 60
    print(f"\n  Deku V1.4 Mode D complete: {elapsed:.1f} min total")


# ============================================================
# MODE H: HORIZON SEARCH (Optuna searches horizon alongside other params)
# ============================================================
def run_mode_h(assets_list, n_trials=MODE_H_DEFAULT_TRIALS, resume=False):
    """
    DEKU Mode H: Like Mode D but with horizon as an Optuna parameter (3-8h).
    Finds the optimal horizon for each asset. Saves top-2 distinct horizons
    so that Mode F can compare dual-horizon strategies.
    """
    t_mode_start = time.time()
    _kill_orphan_workers()

    combo_options = _build_combo_list()
    search_horizons = HORIZON_SEARCH

    print("\n" + "=" * 70)
    print(f"  DEKU V1.4 MODE H: HORIZON SEARCH ({min(search_horizons)}-{max(search_horizons)}h)")
    print(f"  Models: {', '.join(ALL_MODELS.keys())} ({len(combo_options)} combos)")
    print(f"  Trials: {n_trials} (RandomSampler + Hyperband pruner) | gamma=1.0 (fixed)")
    print(f"  data_months: {DATA_MONTHS_OPTIONS} | CPCV: adaptive (4 or 6 groups)")
    metric_label = f" | metric={OPTUNA_METRIC}" if OPTUNA_METRIC != 'apf' else ""
    print(f"  Search: combo × window × data_months × n_features × horizon[{min(search_horizons)}-{max(search_horizons)}]{metric_label}")
    print("=" * 70)

    # Download fresh data
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

    best_models = []

    for asset_name in assets_list:
        t_asset = time.time()

        print(f"\n{'='*70}")
        print(f"  DEKU HORIZON SEARCH: {asset_name} ({min(search_horizons)}-{max(search_horizons)}h)")
        print(f"{'='*70}")

        df_raw = load_data(asset_name)
        if df_raw is None:
            continue

        # Step 1: Build features with reference horizon (midpoint=5h for LGBM ranking)
        max_data_hours = max(DATA_MONTHS_OPTIONS) * 720
        ref_horizon = 5
        print(f"\n  Building all features (horizon={ref_horizon}h [reference])...")
        t0 = time.time()
        df_full, all_cols = build_all_features(df_raw, asset_name=asset_name, horizon=ref_horizon)
        print(f"  [Feature build: {(time.time()-t0)/60:.1f} min]")

        total_rows = len(df_full)
        if total_rows > max_data_hours:
            df_full = df_full.tail(max_data_hours).reset_index(drop=True)
            print(f"  Capped: {total_rows:,} -> {len(df_full):,} rows (last {max(DATA_MONTHS_OPTIONS)}mo)")

        # Pre-compute labels and forward returns for each searchable horizon
        for h in search_horizons:
            fwd_ret = df_full['close'].shift(-h) / df_full['close'] - 1
            if LABEL_MODE == 'fee_aware':
                df_full[f'label_{h}h'] = (fwd_ret > 2 * TRADING_FEE).astype(int)
            else:
                past_ret = df_full['close'] / df_full['close'].shift(h) - 1
                rolling_med = past_ret.rolling(200, min_periods=50).median()
                df_full[f'label_{h}h'] = (fwd_ret > rolling_med).astype(int)
            df_full[f'_fwd_ret_{h}h'] = fwd_ret

        # Drop NaN across all horizon labels
        label_cols = [f'label_{h}h' for h in search_horizons]
        df_clean = df_full.dropna(subset=all_cols + label_cols).reset_index(drop=True)
        print(f"  Clean data: {len(df_clean):,} rows, {len(all_cols)} features, {len(search_horizons)} horizons")

        if len(df_clean) < 500:
            print(f"  Not enough data ({len(df_clean)} rows). Need 500+. Skipping.")
            continue

        # Step 2: LGBM importance ranking (using reference horizon labels)
        # Use label_5h as the ranking target — feature importance is stable across horizons
        df_clean_rank = df_clean.copy()
        df_clean_rank['label'] = df_clean_rank['label_5h']
        print(f"\n  LGBM feature importance ranking (ref: {ref_horizon}h)...")
        t0 = time.time()
        importance_df = _test_lgbm_importance(df_clean_rank, all_cols, gamma=1.0)
        ranked_features = importance_df['feature'].tolist()
        print(f"  [Ranking: {(time.time()-t0)/60:.1f} min] — {len(ranked_features)} features ranked")

        # Prepare numpy arrays
        df_optuna = df_clean.dropna(subset=ranked_features + label_cols).reset_index(drop=True)
        features_np_all = df_optuna[ranked_features].values.astype(np.float64)
        closes_np_all = df_optuna['close'].values.astype(np.float64)

        # Per-horizon label arrays
        labels_np_by_h = {}
        fwd_ret_np_by_h = {}
        for h in search_horizons:
            labels_np_by_h[h] = df_optuna[f'label_{h}h'].values.astype(np.int32)
            fwd_ret_np_by_h[h] = df_optuna[f'_fwd_ret_{h}h'].values.astype(np.float64)

        n_total = len(df_optuna)

        # 3-fold rolling holdout setup
        fold_train_frac = 0.60
        fold_test_frac = 0.20
        fold_stride = 0.10
        holdout_folds = []
        for fi in range(N_HOLDOUT_FOLDS):
            train_s = int(n_total * fi * fold_stride)
            train_e = int(n_total * (fi * fold_stride + fold_train_frac))
            test_s = train_e
            test_e = int(n_total * (fi * fold_stride + fold_train_frac + fold_test_frac))
            test_e = min(test_e, n_total)
            holdout_folds.append((train_s, train_e, test_s, test_e))
            print(f"  Fold {fi+1}: train [{train_s}:{train_e}] ({train_e-train_s:,} rows) → test [{test_s}:{test_e}] ({test_e-test_s:,} rows)")

        # Optuna trains on fold 1
        f1_train_s, f1_train_e, _, _ = holdout_folds[0]
        features_np = features_np_all[f1_train_s:f1_train_e]
        closes_np = closes_np_all[f1_train_s:f1_train_e]
        # Per-horizon labels for fold 1
        labels_np_f1_by_h = {h: labels_np_by_h[h][f1_train_s:f1_train_e] for h in search_horizons}
        n = len(features_np)

        print(f"  Optuna trains on fold 1: {n:,} rows | {N_HOLDOUT_FOLDS}-fold holdout ranking after")

        # Step 3: Optuna study (RandomSampler, fixed gamma=1.0, data_months as param)
        print(f"\n{'='*70}")
        print(f"  OPTUNA STUDY: {asset_name} (horizon search {min(search_horizons)}-{max(search_horizons)}h)")
        print(f"  Search space: {len(combo_options)} combos × {len(DIAG_WINDOWS)} windows × data_months{DATA_MONTHS_OPTIONS} × features × {len(search_horizons)} horizons")
        print(f"  Trials: {n_trials} | Data: {n:,} train | {N_HOLDOUT_FOLDS}-fold holdout | gamma=1.0 (fixed)")
        print(f"{'='*70}")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.RandomSampler(seed=42)

        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=DEKU_PRUNING_WARMUP,
                max_resource=n // DIAG_STEP,
                reduction_factor=3,
            ),
            study_name=f'deku_v14_H_{asset_name}',
        )

        model_factories = _get_deku_diagnostic_models()
        best_apf_so_far = 0.0
        trial_count = 0

        def objective(trial):
            nonlocal best_apf_so_far, trial_count
            trial_count += 1

            h = trial.suggest_categorical('horizon', search_horizons)
            combo_name = trial.suggest_categorical('combo', combo_options)
            window = trial.suggest_categorical('window', DIAG_WINDOWS)
            data_months = trial.suggest_categorical('data_months', DATA_MONTHS_OPTIONS)
            gamma = 1.0  # fixed — no decay

            data_hours = data_months * 720

            # Per-horizon feature range
            feat_min, feat_max = N_FEATURES_RANGE.get(h, N_FEATURES_RANGE_DEFAULT)
            max_f = min(len(ranked_features), feat_max)
            n_feat = trial.suggest_int('n_features', feat_min, max_f)

            combo = combo_name.split('+')

            # Slice data to data_months worth of history
            if n > data_hours:
                trial_feat = features_np[-data_hours:, :n_feat]
                trial_labels = labels_np_f1_by_h[h][-data_hours:]
                trial_closes = closes_np[-data_hours:]
                trial_n = data_hours
            else:
                trial_feat = features_np[:, :n_feat]
                trial_labels = labels_np_f1_by_h[h]
                trial_closes = closes_np
                trial_n = n

            result = _deku_eval_with_pruning(
                trial_feat, trial_labels, trial_closes, combo, window, trial_n,
                DIAG_STEP, model_factories, gamma=gamma, trial=trial,
                horizon=h
            )

            if result is None:
                return 0.0

            trades = result[6]
            MIN_TRADES = 8
            if trades < MIN_TRADES:
                return 0.0

            score = _compute_optuna_score(result)
            ret = result[4]

            if score > best_apf_so_far:
                best_apf_so_far = score
                print(f"  #{trial_count:3d} NEW BEST: {combo_name:22s} w={window:4d}h "
                      f"dm={data_months}mo f={n_feat:3d} h={h}h | {OPTUNA_METRIC}={score:.3f} ret={ret:+.1f}% "
                      f"rawPF={result[11]:.2f} bhPF={result[12]:.2f} trades={trades}")
            elif trial_count % 20 == 0:
                print(f"  #{trial_count:3d} progress: {OPTUNA_METRIC}={score:.3f} | best so far: {best_apf_so_far:.3f}")

            return score

        t_optuna = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        optuna_elapsed = (time.time() - t_optuna) / 60

        # Results summary
        best_trial = study.best_trial
        print(f"\n  {'='*70}")
        print(f"  OPTUNA RESULTS: {asset_name} horizon search ({optuna_elapsed:.1f} min)")
        print(f"  {'='*70}")
        print(f"  Best trial: #{best_trial.number}")
        print(f"  {OPTUNA_METRIC.upper():12s} {best_trial.value:.4f}")
        print(f"  Horizon:    {best_trial.params['horizon']}h")
        print(f"  Combo:      {best_trial.params['combo']}")
        print(f"  Window:     {best_trial.params['window']}h")
        print(f"  Data_months:{best_trial.params['data_months']}mo")
        print(f"  Gamma:      1.0 (fixed)")
        print(f"  N_features: {best_trial.params['n_features']}")

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value > 0]
        completed_trials.sort(key=lambda t: -t.value)

        n_completed = len(completed_trials)
        n_pruned = len(study.trials) - n_completed
        print(f"\n  Trials: {n_completed} completed, {n_pruned} pruned ({100*n_pruned//len(study.trials) if study.trials else 0}% pruned)")

        print(f"\n  {'Rank':>4s}  {OPTUNA_METRIC.upper():>7s}  {'Combo':22s}  {'Window':>6s}  {'DM':>4s}  {'Feats':>5s}  {'H':>3s}")
        print(f"  {'-'*65}")
        for i, t in enumerate(completed_trials[:10], 1):
            marker = " <-- BEST" if i == 1 else ""
            print(f"  {i:4d}  {t.value:7.3f}  {t.params['combo']:22s}  {t.params['window']:5d}h  "
                  f"{t.params['data_months']:3d}m  {t.params['n_features']:5d}  {t.params['horizon']:2d}h{marker}")

        # Horizon distribution
        from collections import Counter
        h_counts = Counter(t.params['horizon'] for t in completed_trials)
        print(f"\n  Horizon distribution (completed trials):")
        for h in sorted(h_counts.keys()):
            bar = '#' * (h_counts[h] * 30 // max(h_counts.values()))
            print(f"    {h}h: {h_counts[h]:4d} {bar}")

        # Parameter importance
        try:
            importances = optuna.importance.get_param_importances(study)
            print(f"\n  Parameter importance:")
            for param, imp in importances.items():
                bar = '#' * int(imp * 40)
                print(f"    {param:20s} {imp*100:5.1f}% {bar}")
        except Exception:
            pass

        # ── 3-FOLD ROLLING HOLDOUT ──
        # Diversity-aware: best per (combo, window, horizon), then ensure combo diversity
        seen_configs = set()
        unique_candidates = []
        for t in completed_trials:
            key = (t.params['combo'], t.params['window'], t.params['horizon'])
            if key not in seen_configs:
                seen_configs.add(key)
                unique_candidates.append(t)

        phase1 = unique_candidates[:10]
        phase1_set = set(id(t) for t in phase1)
        combos_in_phase1 = set(t.params['combo'] for t in phase1)

        phase2 = []
        seen_combos_p2 = set()
        for t in unique_candidates:
            combo = t.params['combo']
            if combo not in combos_in_phase1 and combo not in seen_combos_p2:
                seen_combos_p2.add(combo)
                phase2.append(t)

        candidates = phase1 + phase2
        N_CANDIDATES = min(20, len(candidates))
        candidates = candidates[:N_CANDIDATES]

        print(f"\n  {'='*70}")
        print(f"  {N_HOLDOUT_FOLDS}-FOLD ROLLING HOLDOUT: {asset_name} (top {N_CANDIDATES} candidates)")
        print(f"  {'='*70}")

        holdout_results = []
        min_fold_rows = min(te - ts for _, _, ts, te in holdout_folds)
        if min_fold_rows >= 200:
            for ci, trial in enumerate(candidates):
                c_combo = trial.params['combo'].split('+')
                c_window = trial.params['window']
                c_gamma = 1.0  # fixed
                c_n_feat = trial.params['n_features']
                c_horizon = trial.params['horizon']
                fold_scores = []
                fold_rets = []
                fold_accs = []
                fold_trades = []
                fold_raw_pfs = []

                for fi, (tr_s, tr_e, te_s, te_e) in enumerate(holdout_folds):
                    fold_feat_test = features_np_all[te_s:te_e, :c_n_feat]
                    fold_labels_test = labels_np_by_h[c_horizon][te_s:te_e]
                    fold_closes_test = closes_np_all[te_s:te_e]
                    n_fold_test = te_e - te_s

                    HOLDOUT_STEP = 12
                    ho_result = _deku_eval_with_pruning(
                        fold_feat_test, fold_labels_test, fold_closes_test,
                        c_combo, c_window, n_fold_test,
                        HOLDOUT_STEP, model_factories, gamma=c_gamma, trial=None,
                        horizon=c_horizon
                    )

                    if ho_result:
                        fold_scores.append(_compute_optuna_score(ho_result))
                        fold_rets.append(ho_result[4])
                        fold_accs.append(ho_result[2])
                        fold_trades.append(ho_result[6])
                        fold_raw_pfs.append(ho_result[11])
                    else:
                        fold_scores.append(0.0)
                        fold_rets.append(0.0)
                        fold_accs.append(0.0)
                        fold_trades.append(0)
                        fold_raw_pfs.append(0.0)

                avg_score = np.mean(fold_scores)
                avg_ret = np.mean(fold_rets)
                avg_acc = np.mean(fold_accs)
                total_trades = sum(fold_trades)
                avg_raw_pf = np.mean(fold_raw_pfs)

                holdout_results.append((trial, avg_score, avg_ret, avg_acc, total_trades,
                                        avg_raw_pf, fold_scores, fold_rets))

            holdout_results.sort(key=lambda x: -x[1])

            print(f"\n  {'Rank':>4s}  {'AVG_'+OPTUNA_METRIC.upper():>8s}  {'AVG_Ret':>8s}  {'AVG_Acc':>7s}  {'Tr':>4s}  "
                  f"{'F1':>6s}  {'F2':>6s}  {'F3':>6s}  {'IS_'+OPTUNA_METRIC.upper():>8s}  {'Combo':22s}  {'Win':>4s}  {'DM':>4s}  {'F':>3s}  {'H':>3s}")
            print(f"  {'-'*132}")
            for i, (trial, avg_sc, avg_ret, avg_acc, tot_tr, avg_rpf, f_scores, f_rets) in enumerate(holdout_results[:10], 1):
                is_score = trial.value
                marker = " <-- BEST" if i == 1 else ""
                gen = "✓" if avg_ret > 0 and avg_acc > 0.55 else "~" if avg_ret > 0 else "✗"
                f_str = "  ".join(f"{s:+5.1f}" for s in f_rets)
                print(f"  {i:4d}  {avg_sc:8.3f}  {avg_ret:+7.1f}%  {avg_acc*100:6.1f}%  {tot_tr:4d}  "
                      f"{f_str}  {is_score:8.3f}  {trial.params['combo']:22s}  {trial.params['window']:3d}h  "
                      f"{trial.params['data_months']:3d}m  {trial.params['n_features']:3d}  {gen}  {trial.params['horizon']:2d}h{marker}")
        else:
            print(f"  Hold-out skipped: smallest fold only {min_fold_rows} rows (need 200+)")

        # Pick top-2 distinct horizons for dual-horizon strategy comparison
        winners_by_h = {}  # horizon -> (trial, holdout_result)
        source = holdout_results if (holdout_results and holdout_results[0][1] > 0) else [(t, t.value, 0, 0, 0, 0, [], []) for t in completed_trials]

        for entry in source:
            trial = entry[0]
            h = trial.params['horizon']
            if h not in winners_by_h:
                winners_by_h[h] = entry

        # Sort horizons by their best holdout score
        sorted_horizons = sorted(winners_by_h.keys(), key=lambda h: -winners_by_h[h][1])
        selected_horizons = sorted_horizons[:2]  # top 2
        selected_horizons.sort()  # short first

        print(f"\n  {'='*70}")
        print(f"  TOP HORIZONS for {asset_name}: {', '.join(str(h)+'h' for h in selected_horizons)}")
        print(f"  {'='*70}")

        # Save a model for each selected horizon
        for h in selected_horizons:
            entry = winners_by_h[h]
            winner_trial = entry[0]
            winner_ho = entry if holdout_results else None

            best_combo = winner_trial.params['combo'].split('+')
            best_window = winner_trial.params['window']
            best_gamma = 1.0  # fixed
            best_n_feat = winner_trial.params['n_features']
            best_data_months = winner_trial.params['data_months']
            best_features = ranked_features[:best_n_feat]

            # Re-run winner for in-sample metrics
            feat_np = features_np[:, :best_n_feat]
            labels_np_h = labels_np_f1_by_h[h]
            full_result = _deku_eval_with_pruning(
                feat_np, labels_np_h, closes_np, best_combo, best_window, n,
                DIAG_STEP, model_factories, gamma=best_gamma, trial=None,
                horizon=h
            )

            if full_result:
                combo_name, window, acc, n_total_r, cum_ret, win_rate, trades, _, _, max_dd, apf, raw_pf, bh_pf = full_result
                is_score = _compute_optuna_score(full_result)

                ho_score = winner_ho[1] if winner_ho else 0
                ho_ret = winner_ho[2] if winner_ho else 0
                ho_acc = winner_ho[3] if winner_ho else 0
                ho_trades = winner_ho[4] if winner_ho else 0
                ho_raw_pf = winner_ho[5] if winner_ho else 0
                ho_fold_rets = winner_ho[7] if winner_ho else []

                best_config = {
                    'coin': asset_name,
                    'best_window': best_window,
                    'best_combo': winner_trial.params['combo'],
                    'accuracy': round(acc * 100, 2),
                    'models': winner_trial.params['combo'],
                    'return_pct': round(ho_ret if winner_ho else cum_ret, 2),
                    'win_rate': round(win_rate, 1),
                    'trades': trades,
                    'combined_score': round(ho_score if winner_ho else apf, 4),
                    'feature_set': 'D',
                    'n_features': best_n_feat,
                    'optimal_features': ','.join(best_features),
                    'horizon': h,
                    'gamma': 1.0,
                    'data_months': best_data_months,
                }
                best_models.append(best_config)

                gen_icon = "✓" if ho_ret > 0 and ho_acc > 0.55 else "~" if ho_ret > 0 else "✗"
                print(f"\n  WINNER {h}h: {winner_trial.params['combo']}  w={best_window}h  dm={best_data_months}mo  f={best_n_feat}  {gen_icon}")
                print(f"    In-sample:  {OPTUNA_METRIC}={is_score:.3f}  ret={cum_ret:+.1f}%  acc={acc*100:.1f}%  trades={trades}")
                if winner_ho and len(ho_fold_rets) > 0:
                    fold_str = " / ".join(f"{r:+.1f}%" for r in ho_fold_rets)
                    print(f"    Holdout:    {OPTUNA_METRIC}={ho_score:.3f}  avg_ret={ho_ret:+.1f}%  [{fold_str}]")

        # ── Calibration summary ──
        completed_trials = [t for t in study.trials if t.value is not None and t.value > 0]
        top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:20]

        if top_trials:
            from collections import Counter
            print(f"\n  {'═'*70}")
            print(f"  CALIBRATION SUMMARY: {asset_name} horizon search (Mode H)")
            print(f"  {'═'*70}")

            print(f"  Top 10 trials by {OPTUNA_METRIC}:")
            for rank, t in enumerate(top_trials[:10], 1):
                p = t.params
                print(f"    #{rank:2d}  combo={p['combo']:22s}  w={p['window']:4d}h  "
                      f"dm={p['data_months']}mo  feats={p['n_features']:3d}  "
                      f"{OPTUNA_METRIC}={t.value:.4f}")

            combo_counts = Counter(t.params['combo'] for t in top_trials)
            combo_str = ' | '.join(f"{c}: {n}x" for c, n in combo_counts.most_common())
            print(f"\n  Combo frequency (top {len(top_trials)} trials):")
            print(f"    {combo_str}")

            window_counts = Counter(t.params['window'] for t in top_trials)
            window_str = ' | '.join(f"{w}h: {n}x" for w, n in sorted(window_counts.most_common(), key=lambda x: -x[1]))
            print(f"\n  Window frequency (top {len(top_trials)} trials):")
            print(f"    {window_str}")

            dm_counts = Counter(t.params['data_months'] for t in top_trials)
            dm_str = ' | '.join(f"{d}mo: {n}x" for d, n in sorted(dm_counts.most_common(), key=lambda x: -x[1]))
            print(f"\n  Data_months frequency (top {len(top_trials)} trials):")
            print(f"    {dm_str}")

            feats = [t.params['n_features'] for t in top_trials]
            print(f"  Feature range (top {len(top_trials)} trials): "
                  f"{min(feats)} – {max(feats)} (median: {sorted(feats)[len(feats)//2]})")
            print(f"  {'═'*70}")

        print(f"  [{asset_name} total: {(time.time()-t_asset)/60:.1f} min]")

    if not best_models:
        print("\nNo results. Aborting.")
        return best_models

    # Save best models (merge with existing — clear all horizons for each asset)
    _backup_models_csv()
    csv_path = _get_models_csv_path()
    df_best = pd.DataFrame(best_models)
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        if 'horizon' not in df_existing.columns:
            df_existing['horizon'] = 4
        for m in best_models:
            mask = (df_existing['coin'] == m['coin']) & (df_existing['horizon'] == m['horizon'])
            df_existing = df_existing[~mask]
        df_best = pd.concat([df_existing, df_best], ignore_index=True)
    df_best.to_csv(csv_path, index=False)
    print(f"\n  Best models saved: {csv_path}")

    # Generate signals + charts for best models
    for m in best_models:
        features_list = m['optimal_features'].split(',')
        model_names = m['models'].split('+')
        generate_signals(m['coin'], model_names, m['best_window'],
                         feature_override=features_list, horizon=m['horizon'],
                         gamma=m.get('gamma', 1.0))

    elapsed = (time.time() - t_mode_start) / 60
    print(f"\n  Mode H complete: {elapsed:.1f} min total")

    return best_models


# ============================================================
# LEGACY MODE D (grid-based, kept for comparison)
# ============================================================
def run_mode_d(assets_list, horizon=PREDICTION_HORIZON, permtest=False, resume=False):
    """
    Complete pipeline from scratch:
    1. Build all ~124 features (uses ALL available data, decay handles recency)
    2. Run 5-test feature analysis to find optimal subset
    3. Run 75-config diagnostic with optimal features
    4. Save best models
    5. Generate signals + backtest charts

    Gamma (decay) is read per asset+horizon from CSV. Defaults to 1.0 (no decay).
    With --resume: skips steps that have saved checkpoints from a previous interrupted run.
    """
    t_mode_start = time.time()

    # Kill any orphaned loky workers from previous interrupted runs
    _kill_orphan_workers()

    print("\n" + "=" * 60)
    print(f"  MODE D: FULL PIPELINE -- {horizon}h HORIZON (V5 Cacarot)")
    print(f"  Starts from ALL features, finds optimal subset per asset")
    print(f"  Uses ALL available data — decay weighting handles recency")
    if resume:
        print(f"  RESUME MODE: will skip completed steps (checkpoints in {RESUME_DIR}/)")
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

    best_models = []

    for asset_name in assets_list:
        t_asset = time.time()

        # Read gamma from CSV for this asset+horizon (default 1.0 if not set)
        existing_config = _load_mode_d_config(asset_name, horizon)
        gamma = existing_config.get('gamma', 1.0) if existing_config else 1.0

        print(f"\n{'='*60}")
        print(f"  FULL PIPELINE: {asset_name} ({horizon}h horizon, gamma={gamma})")
        print(f"{'='*60}")

        df_raw = load_data(asset_name)
        if df_raw is None:
            continue

        # Step 1: Build ALL features, cap at 6 months
        # Decay handles recency: gamma=0.999 → 6mo data at 1% weight, gamma=0.995 → 1mo at ~0%
        MAX_DIAG_HOURS = 6 * 30 * 24  # 4,320 hours
        print(f"\n  Building all features (horizon={horizon}h)...")
        t0 = time.time()
        df_full, all_cols = build_all_features(df_raw, asset_name=asset_name, horizon=horizon)
        print(f"  [Feature build: {(time.time()-t0)/60:.1f} min]")

        total_rows = len(df_full)
        if total_rows > MAX_DIAG_HOURS:
            df_full = df_full.tail(MAX_DIAG_HOURS).reset_index(drop=True)
            print(f"  Capped: {total_rows:,} -> {len(df_full):,} rows (last 6mo, decay handles recency)")

        # Drop NaN for analysis
        df_clean = df_full.dropna(subset=all_cols + ['label']).reset_index(drop=True)
        print(f"  Clean data: {len(df_clean):,} rows, {len(all_cols)} features")

        if len(df_clean) < 500:
            print(f"  Not enough data ({len(df_clean)} rows). Need 500+. Skipping.")
            continue

        # Step 2: Feature analysis (5 tests -> optimal subset)
        feat_ckpt = _load_checkpoint(asset_name, horizon, 'features') if resume else None
        if feat_ckpt:
            optimal_features = feat_ckpt['features']
            print(f"\n  RESUME: loaded {len(optimal_features)} optimal features from checkpoint")
        else:
            t0 = time.time()
            optimal_features = run_feature_analysis(asset_name, df_clean, all_cols, gamma=gamma)
            print(f"  [Feature analysis total: {(time.time()-t0)/60:.1f} min]")
            if optimal_features and len(optimal_features) >= 3:
                _save_checkpoint(asset_name, horizon, 'features', {'features': optimal_features})

        if not optimal_features or len(optimal_features) < 3:
            print(f"  Feature analysis produced too few features ({len(optimal_features or [])}). Skipping.")
            continue

        # Step 3: Run diagnostic with optimal features
        diag_ckpt = _load_checkpoint(asset_name, horizon, 'diagnostic') if resume else None
        if diag_ckpt:
            best_config = diag_ckpt
            print(f"\n  RESUME: loaded diagnostic result from checkpoint")
            print(f"  {best_config['coin']} -> {best_config['best_combo']} | w={best_config['best_window']} | {best_config['accuracy']:.1f}%")
        else:
            print(f"\n{'='*60}")
            print(f"  DIAGNOSTIC: {asset_name} ({len(optimal_features)} optimal features)  [{datetime.now().strftime('%H:%M:%S')}]")
            print(f"{'='*60}")

            df_diag = df_clean.dropna(subset=optimal_features + ['label']).reset_index(drop=True)
            print(f"  {len(df_diag):,} rows, {len(optimal_features)} features")

            t0 = time.time()
            best_config, _ = run_diagnostic_for_asset(asset_name, df_diag, optimal_features, gamma=gamma, resume=resume, horizon=horizon)
            print(f"  [Diagnostic: {(time.time()-t0)/60:.1f} min]")
            if best_config:
                _save_checkpoint(asset_name, horizon, 'diagnostic', best_config)

        if best_config:
            best_config['feature_set'] = 'D'
            best_config['n_features'] = len(optimal_features)
            best_config['optimal_features'] = ','.join(optimal_features)
            best_config['horizon'] = horizon
            best_config['gamma'] = gamma
            best_models.append(best_config)

            # -- Improvement 3: Permutation test (only if --permtest flag passed) --
            if permtest:
                t0 = time.time()
                _run_permutation_test(asset_name, df_diag, optimal_features,
                                      best_config, n_perm=200, horizon=horizon)
                print(f"  [Permutation test: {(time.time()-t0)/60:.1f} min]")

        print(f"  [{asset_name} total: {(time.time()-t_asset)/60:.1f} min]")

        # Clear checkpoints after successful completion of this (asset, horizon)
        if best_config:
            _clear_checkpoints(asset_name, horizon)

    if not best_models:
        print("\nNo results. Aborting.")
        return

    # Save best models (merge with existing horizons)
    _backup_models_csv()
    csv_path = _get_models_csv_path()
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
    if not MODELS_CSV_OVERRIDE:
        df_best.to_csv(f'{MODELS_DIR}/crypto_casca_best_models_mode_d.csv', index=False)

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
        cfg_gamma = config.get('gamma', 1.0)

        signals = generate_signals(asset_name, model_names, window, REPLAY_HOURS,
                                   feature_override=feature_override, horizon=horizon, gamma=cfg_gamma)
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
    """Load the Mode D (or previous E) result for a given asset/horizon.
    V1.4: Falls back to production CASCA CSV for gamma if own CSV has no entry."""
    csv_path = _get_models_csv_path()
    row = None
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'horizon' not in df.columns:
            df['horizon'] = 4
        mask = (df['coin'] == asset_name) & (df['horizon'] == horizon)
        match = df[mask]
        if not match.empty:
            row = match.iloc[0]

    # Fallback to production CSV for gamma (V1.4 may not have its own entries yet)
    if row is None:
        prod_csv = f'{MODELS_DIR}/crypto_casca_best_models.csv'
        if os.path.exists(prod_csv):
            df_prod = pd.read_csv(prod_csv)
            if 'horizon' not in df_prod.columns:
                df_prod['horizon'] = 4
            mask = (df_prod['coin'] == asset_name) & (df_prod['horizon'] == horizon)
            match = df_prod[mask]
            if not match.empty:
                row = match.iloc[0]
                print(f"  [V1.4] Using gamma={float(row.get('gamma', 1.0))} from production CASCA CSV")

    if row is None:
        return None

    return {
        'coin': row['coin'],
        'models': row['models'],
        'best_combo': row['best_combo'],
        'best_window': int(row['best_window']),
        'accuracy': row['accuracy'],
        'feature_set': row.get('feature_set', 'D'),
        'optimal_features': str(row.get('optimal_features', '')),
        'horizon': int(row.get('horizon', 4)),
        'training_period': str(row.get('training_period', '')),
        'gamma': float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0,
    }


def _run_iteration_2(asset_name, df_features, prev_config, all_cols, horizon, gamma=1.0):
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
    baseline_acc, _ = _quick_accuracy(df_features, prev_features, window=prev_window, step=24, gamma=gamma)
    print(f"    Baseline: {baseline_acc:.1f}% ({len(prev_features)} features)")

    features_to_drop = []
    for feat in prev_features:
        reduced = [f for f in prev_features if f != feat]
        acc, _ = _quick_accuracy(df_features, reduced, window=prev_window, step=24, gamma=gamma)
        change = acc - baseline_acc
        if change > 0.3:
            features_to_drop.append(feat)
            print(f"    Drop {feat:30s} -> {acc:5.1f}% ({change:+.1f}%) ** REMOVING")
        elif change < -0.5:
            print(f"    Drop {feat:30s} -> {acc:5.1f}% ({change:+.1f}%) ** ESSENTIAL")

    refined_features = [f for f in prev_features if f not in features_to_drop]
    if features_to_drop:
        new_acc, _ = _quick_accuracy(df_features, refined_features, window=prev_window, step=24, gamma=gamma)
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
    current_acc, _ = _quick_accuracy(df_features, refined_features, window=prev_window, step=24, gamma=gamma)
    added = []

    # Only test features that were in the top 50% by LGBM importance (speed optimization)
    from lightgbm import LGBMClassifier
    model_imp = LGBMClassifier(n_estimators=200, max_depth=6, verbose=-1,
                                device=LGBM_DEVICE, random_state=42)
    df_clean = df_features.dropna(subset=all_cols + ['label'])
    X = df_clean[all_cols].tail(5000)
    y = df_clean['label'].tail(5000)
    if len(np.unique(y)) >= 2:
        sw = get_decay_weights(len(y), gamma)
        model_imp.fit(X, y, sample_weight=sw)
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
        acc, _ = _quick_accuracy(test_df, test_set, window=prev_window, step=24, gamma=gamma)
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

    best_config, _ = run_diagnostic_for_asset(asset_name, df_diag, refined_features)

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


def _run_iteration_3(asset_name, df_features, prev_config, all_cols, horizon, gamma=1.0):
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
        sw = get_decay_weights(len(y), gamma)
        model_imp.fit(X, y, sample_weight=sw)
        imp = pd.Series(model_imp.feature_importances_, index=prev_features).sort_values(ascending=False)
        top5 = list(imp.index[:5])
    print(f"    Top 5: {top5}")

    # Test interactions (product of pairs)
    current_acc, _ = _quick_accuracy(df_clean, prev_features, window=prev_window, step=24, gamma=gamma)
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
            acc, _ = _quick_accuracy(test_df, test_set, window=prev_window, step=24, gamma=gamma)
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

    best_config, _ = run_diagnostic_for_asset(asset_name, df_diag, refined_features)

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


def run_mode_e(assets_list, horizon=PREDICTION_HORIZON, iterations='2'):
    """
    Mode E: Iterative refinement of Mode D results.
    iteration='2'  -> run 2nd pass only
    iteration='23' -> run 2nd + 3rd pass
    Gamma read per asset+horizon from CSV.
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
    final_models = []

    for asset_name in assets_list:
        # Load previous Mode D (or E) result
        prev_config = _load_mode_d_config(asset_name, horizon)
        if prev_config is None:
            print(f"  ERROR: No existing model for {asset_name} {horizon}h.")
            print(f"  Run Mode D first!")
            continue

        gamma = prev_config.get('gamma', 1.0)

        print(f"\n{'='*60}")
        print(f"  REFINING: {asset_name} ({horizon}h, gamma={gamma})")
        print(f"{'='*60}")

        prev_features = prev_config['optimal_features']
        if not prev_features or prev_features == 'nan':
            print(f"  ERROR: No optimal features saved for {asset_name}.")
            print(f"  Run Mode D first.")
            continue

        print(f"  Previous: {prev_config['best_combo']} | w={prev_config['best_window']}h | "
              f"{prev_config['accuracy']:.1f}% | {len(prev_features.split(','))} features")

        # Build features, cap at 1 year (decay handles recency)
        df_raw = load_data(asset_name)
        if df_raw is None:
            continue

        print(f"\n  Building all features (horizon={horizon}h)...")
        df_full, all_cols = build_all_features(df_raw, asset_name=asset_name, horizon=horizon)

        MAX_E_HOURS = 6 * 30 * 24  # 4,320 hours
        if len(df_full) > MAX_E_HOURS:
            print(f"  Capped: {len(df_full):,} -> {MAX_E_HOURS:,} rows (last 6mo)")
            df_full = df_full.tail(MAX_E_HOURS).reset_index(drop=True)

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
            asset_name, df_clean, prev_config, all_cols, horizon, gamma=gamma)

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
                asset_name, df_full, iter2_config, all_cols + [c for c in df_full.columns if '_x_' in c], horizon, gamma=gamma)

            if iter3_config and iter3_config['accuracy'] > iter2_config['accuracy']:
                final_config = iter3_config
                print(f"\n  >>> Using iteration 3 result (better by "
                      f"{iter3_config['accuracy'] - iter2_config['accuracy']:+.1f}%)")
            else:
                final_config = iter2_config
                print(f"\n  >>> Keeping iteration 2 result (iter 3 didn't improve)")
        else:
            final_config = iter2_config

        final_config['gamma'] = gamma
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
    _backup_models_csv()
    csv_path = _get_models_csv_path()
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
        cfg_gamma = config.get('gamma', 1.0)

        signals = generate_signals(asset_name, model_names, window, REPLAY_HOURS,
                                   feature_override=feature_override, horizon=horizon, gamma=cfg_gamma)
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
    CASCA: scores by return directly. Updates trading_config.json with best strategy.
    Requires Mode D to have been run first for both 4h and 8h.
    """
    csv_path = _get_models_csv_path()
    if not os.path.exists(csv_path):
        print("  ERROR: No saved models found. Run Mode D first.")
        return

    df_models = pd.read_csv(csv_path)

    print("\n" + "=" * 60)
    print("  MODE F: STRATEGY COMPARISON")
    print("=" * 60)

    # Load trading config for updates
    metric_suffix = f'_{OPTUNA_METRIC}' if OPTUNA_METRIC != 'apf' else ''
    tcfg_path = f'{CONFIG_DIR}/trading_config_deku{metric_suffix}.json'
    try:
        with open(tcfg_path) as f:
            trading_config = json.load(f)
    except Exception:
        trading_config = {}

    for asset in assets_list:
        print(f"\n{'='*60}")
        print(f"  {asset}: Strategy Comparison")
        print(f"{'='*60}")

        # Load configs for each horizon (use CLI horizons if provided, else defaults)
        active_horizons = sorted(horizons) if horizons else sorted(AVAILABLE_HORIZONS)
        h_short = active_horizons[0]
        h_long = active_horizons[-1] if len(active_horizons) > 1 else active_horizons[0]
        cfg_by_h = {}
        for h in active_horizons:
            cfg_h = df_models[(df_models['coin'] == asset) & (df_models['horizon'] == h)]
            if len(cfg_h) > 0:
                cfg_by_h[h] = cfg_h

        has_short = h_short in cfg_by_h
        has_long = h_long in cfg_by_h

        if not has_short and not has_long:
            print(f"  No saved models for {asset}. Run Mode D first.")
            continue

        # Generate signals for available horizons
        signals_by_h = {}

        with _suppress_stderr():
            for h in active_horizons:
                if h not in cfg_by_h:
                    continue
                row_h = cfg_by_h[h].iloc[0]
                feats_h = row_h['optimal_features'].split(',') if pd.notna(row_h.get('optimal_features', '')) else None
                gamma_h = float(row_h.get('gamma', 1.0)) if pd.notna(row_h.get('gamma', 1.0)) else 1.0
                sigs = generate_signals(asset, row_h['models'].split('+'),
                                        int(row_h['best_window']), REPLAY_HOURS_F,
                                        feature_override=feats_h, horizon=h, gamma=gamma_h)
                signals_by_h[h] = simulate_portfolio(sigs)

        # Build merged timeline
        sig_maps = {h: {s['datetime']: s for s in (signals_by_h.get(h) or [])} for h in active_horizons}
        all_dts = set()
        for sm in sig_maps.values():
            all_dts.update(sm.keys())
        all_times = sorted(all_dts)

        if not all_times:
            print(f"  No signals generated for {asset}.")
            continue

        # Hold-out: evaluate strategies on last 33% of signals
        n_signals = len(all_times)
        holdout_start = int(n_signals * 0.67)
        holdout_times = all_times[holdout_start:]
        print(f"  Signals: {n_signals} total, evaluating on last {len(holdout_times)} (hold-out 33%)")

        # Define strategies to test
        strategies = []
        if has_short and has_long:
            strategies = ['both_agree', 'either_agree', f'{h_short}h_only', f'{h_long}h_only']
        elif has_short:
            strategies = [f'{h_short}h_only']
        elif has_long:
            strategies = [f'{h_long}h_only']

        results = []
        for strat in strategies:
            cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
            trades, wins, correct = 0, 0, 0
            total_candles = 0

            for dt in holdout_times:
                s_short = sig_maps[h_short].get(dt) if has_short else None
                s_long = sig_maps[h_long].get(dt) if has_long else None
                price = (s_short or s_long)['close']
                sig_s = s_short['signal'] if s_short else 'HOLD'
                conf_s = s_short['confidence'] if s_short else 50
                sig_l = s_long['signal'] if s_long else 'HOLD'
                conf_l = s_long['confidence'] if s_long else 50

                # Determine combined signal based on strategy
                if strat == 'both_agree':
                    if sig_s == 'SELL' or sig_l == 'SELL':
                        signal = 'SELL'
                    elif sig_s == 'BUY' and sig_l == 'BUY' and conf_s >= MIN_CONFIDENCE and conf_l >= MIN_CONFIDENCE:
                        signal = 'BUY'
                    else:
                        signal = 'HOLD'
                elif strat == 'either_agree':
                    if sig_s == 'SELL' or sig_l == 'SELL':
                        signal = 'SELL'
                    elif (sig_s == 'BUY' and conf_s >= MIN_CONFIDENCE) or (sig_l == 'BUY' and conf_l >= MIN_CONFIDENCE):
                        signal = 'BUY'
                    else:
                        signal = 'HOLD'
                elif strat == f'{h_short}h_only':
                    signal = sig_s if conf_s >= MIN_CONFIDENCE or sig_s == 'SELL' else 'HOLD'
                else:  # Xh_only (long)
                    signal = sig_l if conf_l >= MIN_CONFIDENCE or sig_l == 'SELL' else 'HOLD'

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
                ref_sig = sig_s if s_short else sig_l
                if ref_sig != 'HOLD':
                    total_candles += 1
                    if ref_sig == signal:
                        correct += 1

            # Close open position
            if in_pos and holdout_times:
                last_px = None
                for sm in sig_maps.values():
                    if holdout_times[-1] in sm:
                        last_px = sm[holdout_times[-1]]['close']
                        break
                if last_px:
                    cash = held * last_px * (1 - TRADING_FEE)

            cum_ret = (cash / 1000.0 - 1) * 100
            win_rate = (wins / trades * 100) if trades > 0 else 0
            # Use underlying model accuracy (average of available horizons)
            accs = [cfg_by_h[h].iloc[0].get('accuracy', 65) / 100 for h in cfg_by_h]
            base_acc = sum(accs) / len(accs) if accs else 0.65

            score = cum_ret  # rank strategies by return directly
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

            for dt in holdout_times:
                s_short = sig_maps[h_short].get(dt) if has_short else None
                s_long = sig_maps[h_long].get(dt) if has_long else None
                price = (s_short or s_long)['close']
                sig_s = s_short['signal'] if s_short else 'HOLD'
                conf_s = s_short['confidence'] if s_short else 50
                sig_l = s_long['signal'] if s_long else 'HOLD'
                conf_l = s_long['confidence'] if s_long else 50

                # Apply best strategy with this threshold
                if best_strat == 'both_agree':
                    if sig_s == 'SELL' or sig_l == 'SELL':
                        signal = 'SELL'
                    elif sig_s == 'BUY' and sig_l == 'BUY' and conf_s >= thresh and conf_l >= thresh:
                        signal = 'BUY'
                    else:
                        signal = 'HOLD'
                elif best_strat == 'either_agree':
                    if sig_s == 'SELL' or sig_l == 'SELL':
                        signal = 'SELL'
                    elif (sig_s == 'BUY' and conf_s >= thresh) or (sig_l == 'BUY' and conf_l >= thresh):
                        signal = 'BUY'
                    else:
                        signal = 'HOLD'
                elif best_strat == f'{h_short}h_only':
                    signal = sig_s if conf_s >= thresh or sig_s == 'SELL' else 'HOLD'
                else:  # Xh_only (long)
                    signal = sig_l if conf_l >= thresh or sig_l == 'SELL' else 'HOLD'

                if signal == 'BUY' and not in_pos:
                    held = cash * (1 - TRADING_FEE) / price
                    cash = 0; in_pos = True; entry_px = price; trades_c += 1
                elif signal == 'SELL' and in_pos:
                    cash = held * price * (1 - TRADING_FEE)
                    if price > entry_px: wins_c += 1
                    held = 0; in_pos = False

            if in_pos and holdout_times:
                last_px = None
                for sm in sig_maps.values():
                    if holdout_times[-1] in sm:
                        last_px = sm[holdout_times[-1]]['close']
                        break
                if last_px:
                    cash = held * last_px * (1 - TRADING_FEE)

            cum_ret_c = (cash / 1000.0 - 1) * 100
            win_rate_c = (wins_c / trades_c * 100) if trades_c > 0 else 0
            score_c = cum_ret_c  # CASCA: rank confidence thresholds by return
            marker = ""
            if score_c > best_conf_score:
                best_conf_score = score_c
                best_conf = thresh
                marker = " <-- BEST"
            print(f"  {thresh:>9}% {cum_ret_c:>+7.1f}% {win_rate_c:>7.0f}% {trades_c:>7d} {score_c:>8.3f}{marker}")

        print(f"\n  >>> BEST THRESHOLD for {asset}: {best_conf}%")

        # Generate chart for best strategy + threshold
        chart_signals = []
        for dt in all_times:
            s_short = sig_maps[h_short].get(dt) if has_short else None
            s_long = sig_maps[h_long].get(dt) if has_long else None
            price = (s_short or s_long)['close']
            sig_s_v = s_short['signal'] if s_short else 'HOLD'
            conf_s_v = s_short['confidence'] if s_short else 50
            sig_l_v = s_long['signal'] if s_long else 'HOLD'
            conf_l_v = s_long['confidence'] if s_long else 50

            if best_strat == 'both_agree':
                if sig_s_v == 'SELL' or sig_l_v == 'SELL':
                    signal = 'SELL'
                elif sig_s_v == 'BUY' and sig_l_v == 'BUY' and conf_s_v >= best_conf and conf_l_v >= best_conf:
                    signal = 'BUY'
                else:
                    signal = 'HOLD'
            elif best_strat == 'either_agree':
                if sig_s_v == 'SELL' or sig_l_v == 'SELL':
                    signal = 'SELL'
                elif (sig_s_v == 'BUY' and conf_s_v >= best_conf) or (sig_l_v == 'BUY' and conf_l_v >= best_conf):
                    signal = 'BUY'
                else:
                    signal = 'HOLD'
            elif best_strat == f'{h_short}h_only':
                signal = sig_s_v if conf_s_v >= best_conf or sig_s_v == 'SELL' else 'HOLD'
            else:
                signal = sig_l_v if conf_l_v >= best_conf or sig_l_v == 'SELL' else 'HOLD'

            best_conf_val = max(conf_s_v, conf_l_v)
            rsi_val = (s_short or s_long).get('rsi', 50)
            chart_signals.append({
                'datetime': dt, 'close': price, 'signal': signal,
                'confidence': best_conf_val, 'rsi': rsi_val,
            })

        chart_signals = simulate_portfolio(chart_signals)
        first_cfg = cfg_by_h[h_short].iloc[0] if has_short else cfg_by_h[h_long].iloc[0]
        model_info = {'best_combo': f'{best_strat}@{best_conf}%', 'best_window': 'F',
                       'accuracy': base_acc * 100, 'gamma': first_cfg.get('gamma', 1.0)}
        generate_backtest_chart(asset, chart_signals, model_info=model_info)

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
    print(f"  Quick {asset}: Mode B, {HORIZON_SHORT}h+{HORIZON_LONG}h")
    print("=" * 60)

    csv_path = _get_models_csv_path()
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
    for h in AVAILABLE_HORIZONS:
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

    signals_by_h = {}
    for h in AVAILABLE_HORIZONS:
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

        row_gamma = float(r.get('gamma', 1.0)) if pd.notna(r.get('gamma', 1.0)) else 1.0
        print(f"  Generating {h}h signals (720h)...")
        sigs = generate_signals(asset, model_names, window, 720,
                                feature_override=feature_override, horizon=h, gamma=row_gamma)
        signals_by_h[h] = simulate_portfolio(sigs)

    signals_short = signals_by_h.get(HORIZON_SHORT)
    signals_long = signals_by_h.get(HORIZON_LONG)
    if signals_short or signals_long:
        generate_strategy_html(asset, signals_short, signals_long, strategy=strategy)
        generate_signal_table_html(asset, signals_short, signals_long, strategy=strategy)

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


# ============================================================
# MODE DAF: COMBINED D → A → F PIPELINE
# Efficient gamma optimization: runs feature analysis once (gamma-independent),
# then sweeps gammas only on the diagnostic (top 10 combos), then Mode F.
# Usage: python crypto_trading_system_casca_v1.4.py DAF BTC 4,8h
# ============================================================
DAF_GAMMAS = [0.999, 0.998, 0.997, 0.996, 0.995]
DAF_TOP_N = 10  # number of top (combo, window) pairs from D phase to sweep in A phase


def run_mode_daf(assets_list, horizons, resume=False):
    """Combined D→A→F pipeline.

    D phase: Full pipeline at gamma=1.0 — data download, feature analysis, diagnostic.
             Feature analysis is gamma-independent so it only runs once.
             Captures sorted_results (all 55 diagnostic configs ranked by APF).

    A phase: Takes top 10 (combo, window) pairs from D results.
             Sweeps DAF_GAMMAS × top 10 = 50 eval calls per horizon (vs 275 in full Mode A).
             Picks the best (combo, window, gamma) by APF.

    F phase: Runs strategy comparison with the winning models saved to CSV.
    """
    from joblib.externals.loky import get_reusable_executor

    total_horizons = len(horizons) * len(assets_list)
    t_total = time.time()

    print("\n" + "=" * 70)
    print(f"  MODE DAF: COMBINED D → A → F PIPELINE (V1.4)")
    print(f"  D: feature analysis + diagnostic (gamma=1.0)")
    print(f"  A: gamma sweep on top {DAF_TOP_N} combos × {len(DAF_GAMMAS)} gammas")
    print(f"  F: strategy comparison")
    print(f"  Assets: {', '.join(assets_list)} | Horizons: {', '.join(str(h)+'h' for h in horizons)}")
    print(f"  Gammas: {', '.join(str(g) for g in DAF_GAMMAS)}")
    print("=" * 70)

    # Download fresh macro & sentiment data once
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

    _kill_orphan_workers()

    best_models = []  # final winners (one per asset+horizon)

    for asset_name in assets_list:
        for horizon in horizons:
            t_ah = time.time()
            print(f"\n{'#'*70}")
            print(f"  DAF: {asset_name} {horizon}h")
            print(f"{'#'*70}")

            # ── D PHASE: feature analysis + diagnostic at gamma=1.0 ──
            print(f"\n  ── D PHASE: {asset_name} {horizon}h (gamma=1.0) ──")

            df_raw = load_data(asset_name)
            if df_raw is None:
                continue

            MAX_DIAG_HOURS = 6 * 30 * 24  # 4,320 hours
            print(f"\n  Building all features (horizon={horizon}h)...")
            t0 = time.time()
            df_full, all_cols = build_all_features(df_raw, asset_name=asset_name, horizon=horizon)
            print(f"  [Feature build: {(time.time()-t0)/60:.1f} min]")

            total_rows = len(df_full)
            if total_rows > MAX_DIAG_HOURS:
                df_full = df_full.tail(MAX_DIAG_HOURS).reset_index(drop=True)
                print(f"  Capped: {total_rows:,} -> {len(df_full):,} rows (last 6mo)")

            df_clean = df_full.dropna(subset=all_cols + ['label']).reset_index(drop=True)
            print(f"  Clean data: {len(df_clean):,} rows, {len(all_cols)} features")

            if len(df_clean) < 500:
                print(f"  Not enough data ({len(df_clean)} rows). Need 500+. Skipping.")
                continue

            # Feature analysis (gamma-independent — only runs once)
            feat_ckpt = _load_checkpoint(asset_name, horizon, 'features') if resume else None
            if feat_ckpt:
                optimal_features = feat_ckpt['features']
                print(f"\n  RESUME: loaded {len(optimal_features)} optimal features from checkpoint")
            else:
                t0 = time.time()
                optimal_features = run_feature_analysis(asset_name, df_clean, all_cols, gamma=1.0)
                print(f"  [Feature analysis total: {(time.time()-t0)/60:.1f} min]")
                if optimal_features and len(optimal_features) >= 3:
                    _save_checkpoint(asset_name, horizon, 'features', {'features': optimal_features})

            if not optimal_features or len(optimal_features) < 3:
                print(f"  Feature analysis produced too few features ({len(optimal_features or [])}). Skipping.")
                continue

            # Diagnostic at gamma=1.0
            df_diag = df_clean.dropna(subset=optimal_features + ['label']).reset_index(drop=True)
            print(f"\n{'='*60}")
            print(f"  D-PHASE DIAGNOSTIC: {asset_name} ({len(optimal_features)} features, gamma=1.0)  [{datetime.now().strftime('%H:%M:%S')}]")
            print(f"{'='*60}")
            print(f"  {len(df_diag):,} rows, {len(optimal_features)} features")

            t0 = time.time()
            d_best_config, d_sorted_results = run_diagnostic_for_asset(
                asset_name, df_diag, optimal_features, gamma=1.0, resume=resume, horizon=horizon
            )
            d_elapsed = (time.time()-t0)/60
            print(f"  [D-phase diagnostic: {d_elapsed:.1f} min]")

            if not d_sorted_results:
                print(f"  No diagnostic results. Skipping.")
                continue

            # ── A PHASE: gamma sweep on top N combos ──
            print(f"\n  ── A PHASE: gamma sweep on top {DAF_TOP_N} combos ──")

            # Sort by APF descending, take top N unique (combo, window) pairs
            d_sorted_results.sort(key=lambda x: (-x[10], -x[4]))  # APF desc, return desc
            top_configs = []
            seen = set()
            for r in d_sorted_results:
                key = (r[0], r[1])  # (combo_name, window)
                if key not in seen:
                    seen.add(key)
                    top_configs.append(key)
                if len(top_configs) >= DAF_TOP_N:
                    break

            print(f"  Top {len(top_configs)} combos from D phase:")
            for i, (combo_name, window) in enumerate(top_configs, 1):
                # Find original APF for display
                orig = next(r for r in d_sorted_results if r[0] == combo_name and r[1] == window)
                print(f"    {i:2d}. {combo_name:22s} w={window:4d}h  APF={orig[10]:.3f}  ret={orig[4]:+.1f}%")

            n_tests = len(DAF_GAMMAS) * len(top_configs)
            print(f"\n  Sweeping {len(DAF_GAMMAS)} gammas × {len(top_configs)} combos = {n_tests} tests")

            # Prepare numpy arrays for _eval_one_config
            features_np = df_diag[optimal_features].values.astype(np.float64)
            labels_np = df_diag['label'].values.astype(np.int32)
            closes_np = df_diag['close'].values.astype(np.float64)
            n = len(df_diag)

            # Reset BLAS limits for diagnostic workers
            get_reusable_executor().shutdown(wait=True)
            for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                os.environ.pop(var, None)

            # Build all (gamma, combo, window) tasks
            a_tasks = []
            for gamma in DAF_GAMMAS:
                for combo_name, window in top_configs:
                    combo_list = combo_name.split('+')
                    a_tasks.append((gamma, combo_list, window))

            t_a = time.time()
            with _suppress_stderr():
                a_results = Parallel(n_jobs=N_JOBS_PARALLEL, verbose=0)(
                    delayed(_eval_one_config)(
                        features_np, labels_np, closes_np, combo, window, n, DIAG_STEP, DIAG_MODELS,
                        gamma=gamma
                    )
                    for gamma, combo, window in a_tasks
                )

            # Restore BLAS limits
            for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                os.environ[var] = '1'
            get_reusable_executor().shutdown(wait=True)

            a_elapsed = (time.time()-t_a)/60
            print(f"  [A-phase sweep: {a_elapsed:.1f} min]")

            # Combine D (gamma=1.0) results + A (gamma sweep) results
            # Build a unified list: (gamma, combo_name, window, APF, return, raw_pf, bh_pf, accuracy, trades)
            all_candidates = []

            # Add D-phase results (gamma=1.0) for the top N combos
            for combo_name, window in top_configs:
                orig = next(r for r in d_sorted_results if r[0] == combo_name and r[1] == window)
                all_candidates.append((1.0, combo_name, window, orig[10], orig[4], orig[11], orig[12], orig[2], orig[6]))

            # Add A-phase results
            for (gamma, combo_list, window), result in zip(a_tasks, a_results):
                if result is None:
                    continue
                combo_name, r_window, acc, n_total, cum_ret, win_rate, trades, avg_gain, avg_loss, max_dd, adj_pf, raw_pf, bh_pf = result
                all_candidates.append((gamma, combo_name, r_window, adj_pf, cum_ret, raw_pf, bh_pf, acc, trades))

            # Sort by APF descending, then return descending
            all_candidates.sort(key=lambda x: (-x[3], -x[4]))

            # Display top 15
            print(f"\n  {'Rank':>4s}  {'Gamma':>6s}  {'Combo':22s}  {'Window':>6s}  {'APF':>7s}  {'Return':>8s}  {'rawPF':>6s}  {'bhPF':>6s}  {'Acc':>6s}  {'Trades':>6s}")
            print(f"  {'-'*88}")
            for i, (gamma, combo_name, window, apf, ret, raw_pf, bh_pf, acc, trades) in enumerate(all_candidates[:15], 1):
                marker = " <-- WINNER" if i == 1 else ""
                print(f"  {i:4d}  {gamma:6.3f}  {combo_name:22s}  {window:5d}h  {apf:7.3f}  {ret:+7.1f}%  {raw_pf:6.2f}  {bh_pf:6.2f}  {acc*100:5.1f}%  {trades:5d}{marker}")

            # Pick winner
            winner = all_candidates[0]
            w_gamma, w_combo, w_window, w_apf, w_ret, w_raw_pf, w_bh_pf, w_acc, w_trades = winner

            print(f"\n  {'='*70}")
            print(f"  WINNER: {asset_name} {horizon}h")
            print(f"  Models: {w_combo}  Window: {w_window}h  Gamma: {w_gamma}")
            print(f"  APF: {w_apf:.3f}  Return: {w_ret:+.1f}%  Accuracy: {w_acc*100:.1f}%")
            print(f"  rawPF: {w_raw_pf:.2f}  bhPF: {w_bh_pf:.2f}  Trades: {w_trades}")
            print(f"  {'='*70}")

            # Save winner to V1.4 models CSV
            best_config = {
                'coin': asset_name,
                'best_window': w_window,
                'best_combo': w_combo,
                'accuracy': round(w_acc * 100, 2),
                'models': w_combo,
                'return_pct': round(w_ret, 2),
                'win_rate': 0,  # not tracked in summary
                'trades': w_trades,
                'combined_score': round(w_apf, 4),
                'feature_set': 'D',
                'n_features': len(optimal_features),
                'optimal_features': ','.join(optimal_features),
                'horizon': horizon,
                'gamma': w_gamma,
            }
            best_models.append(best_config)

            # Save Mode A results CSV for reference
            import csv
            a_csv_path = MODE_A_RESULTS_CSV
            file_exists = os.path.exists(a_csv_path)
            with open(a_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['coin', 'horizon', 'gamma', 'models', 'window', 'accuracy',
                                     'return_pct', 'raw_pf', 'buyhold_pf', 'adjusted_pf', 'trades'])
                for gamma, combo_name, window, apf, ret, raw_pf, bh_pf, acc, trades in all_candidates:
                    writer.writerow([asset_name, horizon, gamma, combo_name, window,
                                     round(acc*100, 2), round(ret, 2), round(raw_pf, 4),
                                     round(bh_pf, 4), round(apf, 4), trades])
            print(f"  A-phase results saved: {a_csv_path}")

            elapsed_ah = (time.time()-t_ah)/60
            print(f"  [{asset_name} {horizon}h total: {elapsed_ah:.1f} min]")

    if not best_models:
        print("\nNo results. Aborting.")
        return

    # Save best models to V1.4 CSV (merge with existing)
    _backup_models_csv()
    csv_path = _get_models_csv_path()
    df_best = pd.DataFrame(best_models)
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        if 'horizon' not in df_existing.columns:
            df_existing['horizon'] = 4
        for m in best_models:
            mask = (df_existing['coin'] == m['coin']) & (df_existing['horizon'] == m['horizon'])
            df_existing = df_existing[~mask]
        df_best = pd.concat([df_existing, df_best], ignore_index=True)
    df_best.to_csv(csv_path, index=False)
    print(f"\n  Best models saved: {csv_path}")

    # ── F PHASE: strategy comparison ──
    if len(horizons) >= 2:
        print(f"\n  ── F PHASE: strategy comparison ──")
        run_strategy_comparison(assets_list, horizons)
    else:
        print(f"\n  F PHASE skipped (need both 4h+8h for strategy comparison)")

    elapsed_total = (time.time()-t_total)/60
    print(f"\n  MODE DAF complete: {elapsed_total:.1f} min total")


# ============================================================
# MODE A: GAMMA OPTIMIZATION
# Tests multiple gamma values per horizon, saves to isolated CSV.
# Usage: python crypto_trading_system_casca.py A BTC 4,8h
# ============================================================
MODE_A_GAMMAS = [0.999, 0.998, 0.997, 0.996, 0.995]
MODE_A_RESULTS_CSV = f'{MODELS_DIR}/testing_deku_a_results.csv'


def run_mode_a(assets_list, horizons, resume=False):
    """Run Mode D for each gamma × horizon combo, saving results to isolated CSV."""
    import shutil

    total_tests = len(MODE_A_GAMMAS) * len(horizons) * len(assets_list)
    print(f"\n{'='*70}")
    print(f"  MODE A: GAMMA OPTIMIZATION")
    print(f"  {len(MODE_A_GAMMAS)} gammas × {len(horizons)} horizons × {len(assets_list)} assets = {total_tests} tests")
    print(f"  Gammas: {', '.join(str(g) for g in MODE_A_GAMMAS)}")
    print(f"  Baseline (gamma=1.0) already in production CSV")
    print(f"  Results: {MODE_A_RESULTS_CSV}")
    print(f"{'='*70}")

    t_total = time.time()
    completed = 0

    for asset in assets_list:
        for horizon in horizons:
            for gamma in MODE_A_GAMMAS:
                completed += 1

                # Check if already done (resume support)
                if resume and _mode_a_is_completed(asset, gamma, horizon):
                    print(f"\n  [{completed}/{total_tests}] SKIP: {asset} {horizon}h gamma={gamma} (already done)")
                    continue

                print(f"\n{'#'*70}")
                print(f"  [{completed}/{total_tests}] GAMMA TEST: {asset} | {horizon}h | gamma={gamma}")
                print(f"{'#'*70}")

                t0 = time.time()

                # Use MODELS_CSV_OVERRIDE to redirect Mode D output to a temp CSV
                # so it doesn't overwrite production models
                temp_csv = f'{MODELS_DIR}/_casca_a_temp.csv'
                global MODELS_CSV_OVERRIDE
                old_override = MODELS_CSV_OVERRIDE
                MODELS_CSV_OVERRIDE = temp_csv

                # Seed the temp CSV with a row that has our forced gamma
                # so _load_mode_d_config picks it up
                seed = pd.DataFrame([{
                    'coin': asset, 'best_window': 100, 'best_combo': '',
                    'accuracy': 0, 'models': '', 'return_pct': 0,
                    'win_rate': 0, 'trades': 0, 'combined_score': 0,
                    'feature_set': 'D', 'n_features': 0, 'optimal_features': '',
                    'horizon': horizon, 'gamma': gamma
                }])
                seed.to_csv(temp_csv, index=False)

                # Override charts dir for isolation
                global CHARTS_DIR
                old_charts = CHARTS_DIR
                CHARTS_DIR = f'charts/casca_a_test'
                os.makedirs(CHARTS_DIR, exist_ok=True)

                try:
                    run_mode_d([asset], horizon=horizon)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    MODELS_CSV_OVERRIDE = old_override
                    CHARTS_DIR = old_charts
                    continue
                finally:
                    MODELS_CSV_OVERRIDE = old_override
                    CHARTS_DIR = old_charts

                elapsed = time.time() - t0

                # Read result from temp CSV
                if os.path.exists(temp_csv):
                    df_temp = pd.read_csv(temp_csv)
                    mask = (df_temp['coin'] == asset) & (df_temp['horizon'] == horizon)
                    matches = df_temp[mask]
                    # Filter out our seed row (accuracy > 0)
                    matches = matches[matches['accuracy'] > 0]

                    if not matches.empty:
                        row = matches.iloc[-1]
                        result = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'asset': asset,
                            'horizon': horizon,
                            'gamma': gamma,
                            'best_combo': row['best_combo'],
                            'best_window': int(row['best_window']),
                            'accuracy': float(row['accuracy']),
                            'return_pct': float(row.get('return_pct', 0)),
                            'win_rate': float(row.get('win_rate', 0)),
                            'trades': int(row.get('trades', 0)),
                            'adjusted_pf': float(row.get('combined_score', 0)),
                            'n_features': int(row.get('n_features', 0)),
                            'elapsed_min': round(elapsed / 60, 1),
                        }
                        _mode_a_save_result(result)
                        print(f"\n  RESULT: gamma={gamma} | {result['best_combo']} w={result['best_window']} | "
                              f"APF={result['adjusted_pf']:.3f} | ret={result['return_pct']:+.1f}% | "
                              f"acc={result['accuracy']:.1f}% | trades={result['trades']} | {result['elapsed_min']}min")
                    else:
                        print(f"  No result found for {asset} {horizon}h gamma={gamma}")

                    # Clean up temp file
                    try:
                        os.remove(temp_csv)
                    except OSError:
                        pass

    print(f"\n  Total time: {(time.time()-t_total)/60:.1f} min")
    _mode_a_print_summary(assets_list, horizons)


def _mode_a_save_result(result):
    """Append a result to the Mode A results CSV."""
    df_new = pd.DataFrame([result])
    if os.path.exists(MODE_A_RESULTS_CSV):
        df_existing = pd.read_csv(MODE_A_RESULTS_CSV)
        mask = ((df_existing['asset'] == result['asset']) &
                (df_existing['horizon'] == result['horizon']) &
                (df_existing['gamma'] == result['gamma']))
        df_existing = df_existing[~mask]
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(MODE_A_RESULTS_CSV, index=False)


def _mode_a_is_completed(asset, gamma, horizon):
    """Check if this combo already has a result."""
    if not os.path.exists(MODE_A_RESULTS_CSV):
        return False
    df = pd.read_csv(MODE_A_RESULTS_CSV)
    mask = ((df['asset'] == asset) &
            (df['horizon'] == horizon) &
            (df['gamma'] == gamma))
    return mask.any()


def _mode_a_print_summary(assets_list, horizons):
    """Print comparison table including baseline from production CSV."""
    if not os.path.exists(MODE_A_RESULTS_CSV):
        print("No Mode A results yet.")
        return

    df = pd.read_csv(MODE_A_RESULTS_CSV)

    # Load baseline (gamma=1.0) from production CSV
    prod_csv = f'{MODELS_DIR}/crypto_casca_best_models.csv'
    if os.path.exists(prod_csv):
        df_prod = pd.read_csv(prod_csv)
        for asset in assets_list:
            for h in horizons:
                mask = (df_prod['coin'] == asset) & (df_prod['horizon'] == h)
                row = df_prod[mask]
                if not row.empty:
                    r = row.iloc[0]
                    baseline = {
                        'timestamp': '-', 'asset': asset, 'horizon': h, 'gamma': 1.0,
                        'best_combo': r['best_combo'], 'best_window': int(r['best_window']),
                        'accuracy': float(r['accuracy']), 'return_pct': float(r.get('return_pct', 0)),
                        'win_rate': float(r.get('win_rate', 0)), 'trades': int(r.get('trades', 0)),
                        'adjusted_pf': float(r.get('combined_score', 0)),
                        'n_features': int(r.get('n_features', 0)), 'elapsed_min': 0,
                    }
                    df = pd.concat([df, pd.DataFrame([baseline])], ignore_index=True)

    for asset in assets_list:
        df_a = df[df['asset'] == asset]
        if df_a.empty:
            continue

        print(f"\n{'='*95}")
        print(f"  MODE A: GAMMA OPTIMIZATION — {asset}")
        print(f"{'='*95}")
        print(f"  {'Horizon':<8} {'Gamma':<8} {'Model':<18} {'Window':<8} {'APF':<8} {'Return%':<10} {'WinRate':<8} {'Trades':<8} {'Acc%':<8} {'Feats':<6}")
        print(f"  {'-'*91}")

        for h in horizons:
            h_df = df_a[df_a['horizon'] == h].sort_values('gamma', ascending=False)
            best_pf = h_df['adjusted_pf'].max()
            for _, row in h_df.iterrows():
                gamma_str = f"{row['gamma']:.3f}" if row['gamma'] < 1.0 else "1.000*"
                marker = " <-- BEST" if row['adjusted_pf'] == best_pf else ""
                print(f"  {h}h{'':<5} {gamma_str:<8} {row['best_combo']:<18} "
                      f"{int(row['best_window'])}h{'':<5} {row['adjusted_pf']:<8.3f} "
                      f"{row['return_pct']:<+10.1f} {row['win_rate']:<8.1f} "
                      f"{int(row['trades']):<8} {row['accuracy']:<8.1f} {int(row['n_features']):<6}{marker}")
            if h != horizons[-1]:
                print(f"  {'-'*91}")

        print(f"  {'='*91}")
        print(f"  * = baseline (gamma=1.0, from production CSV)")


def main():

    has_macro = os.path.exists(MACRO_DIR)

    # ================================================================
    # CLI: python crypto_trading_system_deku.py D BTC 4,8h
    # Supports: MODE [ASSETS] [HORIZON] [--trials N]
    # Examples:
    #   python crypto_trading_system_deku.py D BTC 4,8h
    #   python crypto_trading_system_deku.py D BTC 4,8h --trials 150
    #   python crypto_trading_system_deku.py DF BTC,ETH 4,8h
    #   python crypto_trading_system_deku.py B BTC 8h
    #   python crypto_trading_system_deku.py F BTC 4,8h
    # ================================================================
    # Filter out --flag and their values (--trials N, --metric X)
    _skip_next = False
    cli_args = []
    for a in sys.argv[1:]:
        if _skip_next:
            _skip_next = False
            continue
        if a.startswith('--'):
            if a in ('--trials', '--metric'):
                _skip_next = True  # skip the next arg (the value)
            continue
        cli_args.append(a)
    flag_resume = '--resume' in sys.argv

    # Parse --trials N
    n_trials = DEKU_DEFAULT_TRIALS
    for i, a in enumerate(sys.argv[1:], 1):
        if a == '--trials' and i < len(sys.argv) - 1:
            try:
                n_trials = int(sys.argv[i + 1])
            except ValueError:
                pass

    # Parse --metric NAME or --metric all
    global OPTUNA_METRIC
    run_all_metrics = False
    for i, a in enumerate(sys.argv[1:], 1):
        if a == '--metric' and i < len(sys.argv) - 1:
            m = sys.argv[i + 1].lower()
            if m == 'all':
                run_all_metrics = True
                print(f"  Scoring metric: ALL ({', '.join(sorted(VALID_METRICS))})")
            elif m in VALID_METRICS:
                OPTUNA_METRIC = m
                print(f"  Scoring metric: {OPTUNA_METRIC}")
            else:
                print(f"  Unknown metric '{m}'. Valid: all, {', '.join(sorted(VALID_METRICS))}")
                return

    if cli_args and cli_args[0].upper() in ('B', 'D', 'DF', 'F', 'H', 'HF', '5', '6', '7'):
        mode = cli_args[0].upper()

        # Shortcuts 5/6/7 from CLI
        if mode in ('5', '6', '7'):
            shortcut_map = {'5': 'BTC', '6': 'ETH', '7': 'XRP'}
            _run_quick_asset(shortcut_map[mode])
            return

        # Parse assets (default: all)
        if len(cli_args) >= 2 and not cli_args[1].endswith('h') and not cli_args[1].endswith('y') and not cli_args[1].endswith('m'):
            assets_list = [a.strip().upper() for a in cli_args[1].split(',') if a.strip().upper() in ASSETS]
            if not assets_list:
                assets_list = list(ASSETS.keys())
        else:
            assets_list = list(ASSETS.keys())

        # Parse horizon (default: both for Mode B and DF, short for others)
        # Mode H/HF ignores horizon args (searches 3-8h automatically)
        horizons = list(AVAILABLE_HORIZONS) if mode in ('B', 'DF') else [HORIZON_SHORT]
        for a in cli_args:
            if a.lower().endswith('h') and a[:-1].replace(',', '').isdigit():
                horizons = [int(h) for h in a[:-1].split(',')]

        # Use MODE_H_DEFAULT_TRIALS for H/HF modes
        if mode in ('H', 'HF') and n_trials == DEKU_DEFAULT_TRIALS:
            n_trials = MODE_H_DEFAULT_TRIALS

        trials_str = f" | {n_trials} trials" if mode in ('D', 'DF', 'H', 'HF') else ""
        h_str = f"search {min(HORIZON_SEARCH)}-{max(HORIZON_SEARCH)}h" if mode in ('H', 'HF') else ','.join(str(h)+'h' for h in horizons)
        print("=" * 60)
        print(f"  DEKU V1.4: Mode {mode} | {','.join(assets_list)} | {h_str}{trials_str}")
        print("=" * 60)

    else:

        print("=" * 60)
        print("  CRYPTO HOURLY ML TRADING SYSTEM -- DEKU V1.4")
        print(f"  Random Optuna + Adaptive CPCV | gamma=1.0 (fixed)")
        print(f"  5 models (RF+GB+XGB+LR+LGBM) | data_months: {DATA_MONTHS_OPTIONS}")
        print(f"  Prediction: {', '.join(str(h)+'h' for h in AVAILABLE_HORIZONS)} horizons available")
        print(f"  Macro data: {'FOUND' if has_macro else 'NOT FOUND -- run download_macro_data.py'}")
        print("=" * 60)

        print("\nChoose mode:")
        print("  B.  Quick run (saved models + signals + chart)")
        print("  D.  OPTUNA OPTIMIZATION (joint combo × window × data_months × features)")
        print("  DF. D then F (optimize + strategy comparison)")
        print(f"  F.  STRATEGY COMPARISON (both_agree / either_agree / {HORIZON_SHORT}h / {HORIZON_LONG}h)")
        print(f"  H.  HORIZON SEARCH (Optuna sweeps {min(HORIZON_SEARCH)}-{max(HORIZON_SEARCH)}h)")
        print(f"  HF. H then F (horizon search + strategy comparison)")
        print("  ---")
        print(f"  5. Quick BTC (Mode B, both {HORIZON_SHORT}h+{HORIZON_LONG}h)")
        print(f"  6. Quick ETH (Mode B, both {HORIZON_SHORT}h+{HORIZON_LONG}h)")
        print(f"  7. Quick XRP (Mode B, both {HORIZON_SHORT}h+{HORIZON_LONG}h)")
        mode = input("\nEnter B/D/DF/F/H/HF or 5-7: ").strip().upper()

        # Shortcuts 5/6/7
        if mode in ('5', '6', '7'):
            shortcut_map = {'5': 'BTC', '6': 'ETH', '7': 'XRP'}
            _run_quick_asset(shortcut_map[mode])
            return

        if mode not in ('B', 'D', 'DF', 'F', 'H', 'HF'):
            print("Invalid choice. Defaulting to B.")
            mode = 'B'

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

        print(f"\nAssets: {', '.join(assets_list)}")

        # Horizon selection
        print("\nPrediction horizon:")
        print(f"  1. {HORIZON_SHORT} hours ahead (default)")
        print(f"  2. {HORIZON_LONG} hours ahead")
        print(f"  3. Both ({HORIZON_SHORT}h + {HORIZON_LONG}h)")
        h_choice = input("Enter choice (1-3) [1]: ").strip()
        if h_choice == '2':
            horizons = [HORIZON_LONG]
        elif h_choice == '3':
            horizons = list(AVAILABLE_HORIZONS)
        else:
            horizons = [HORIZON_SHORT]
        print(f"Horizon(s): {', '.join(str(h)+'h' for h in horizons)}")

        if mode in ('D', 'DF'):
            try:
                trials_input = input(f"Number of Optuna trials [{DEKU_DEFAULT_TRIALS}]: ").strip()
                if trials_input:
                    n_trials = int(trials_input)
            except ValueError:
                pass

    # Execute mode
    if run_all_metrics and mode in ('D', 'DF'):
        # --metric all: run Mode DF for each metric, then compare
        all_results = {}  # metric -> {asset: {strategy, conf, return, win_rate, trades}}
        for metric in sorted(VALID_METRICS):
            OPTUNA_METRIC = metric
            print(f"\n{'#'*70}")
            print(f"  METRIC: {metric.upper()}")
            print(f"{'#'*70}")
            for h in horizons:
                if len(horizons) > 1:
                    print(f"\n{'#'*60}")
                    print(f"  RUNNING {h}h HORIZON")
                    print(f"{'#'*60}")
                run_mode_d_optuna(assets_list, horizon=h, n_trials=n_trials, resume=flag_resume)
            run_strategy_comparison(assets_list, horizons)

            # Read the trading config that Mode F just wrote
            metric_suffix = f'_{metric}' if metric != 'apf' else ''
            tcfg_path = f'{CONFIG_DIR}/trading_config_deku{metric_suffix}.json'
            csv_path = _get_models_csv_path()
            try:
                with open(tcfg_path) as f:
                    tcfg = json.load(f)
            except Exception:
                tcfg = {}
            try:
                df_m = pd.read_csv(csv_path)
            except Exception:
                df_m = pd.DataFrame()

            metric_res = {}
            for asset in assets_list:
                ac = tcfg.get(asset, {})
                strat = ac.get('strategy', '?')
                conf = ac.get('min_confidence', '?')
                # Get return/trades from the CSV
                ret_short = df_m[(df_m['coin'] == asset) & (df_m['horizon'] == HORIZON_SHORT)]['return_pct'].values
                ret_long = df_m[(df_m['coin'] == asset) & (df_m['horizon'] == HORIZON_LONG)]['return_pct'].values
                tr_short = df_m[(df_m['coin'] == asset) & (df_m['horizon'] == HORIZON_SHORT)]['trades'].values
                tr_long = df_m[(df_m['coin'] == asset) & (df_m['horizon'] == HORIZON_LONG)]['trades'].values
                metric_res[asset] = {
                    'strategy': strat, 'conf': conf,
                    'ret_short': float(ret_short[0]) if len(ret_short) > 0 else None,
                    'ret_long': float(ret_long[0]) if len(ret_long) > 0 else None,
                    'trades_short': int(tr_short[0]) if len(tr_short) > 0 else None,
                    'trades_long': int(tr_long[0]) if len(tr_long) > 0 else None,
                }
            all_results[metric] = metric_res

        # Print comparison table
        print(f"\n{'='*80}")
        print(f"  METRIC COMPARISON — ALL RESULTS")
        print(f"{'='*80}")
        for asset in assets_list:
            print(f"\n  {asset}:")
            print(f"  {'Metric':<10} {f'{HORIZON_SHORT}h Ret':>8} {f'{HORIZON_SHORT}h Tr':>6} {f'{HORIZON_LONG}h Ret':>8} {f'{HORIZON_LONG}h Tr':>6} {'Strategy':<16} {'Conf':>5}")
            print(f"  {'-'*62}")
            best_metric = None
            best_total = -999
            for metric in sorted(VALID_METRICS):
                r = all_results[metric].get(asset, {})
                r4 = r.get('ret_short')
                r8 = r.get('ret_long')
                t4 = r.get('trades_short')
                t8 = r.get('trades_long')
                strat = r.get('strategy', '?')
                conf = r.get('conf', '?')
                r4_s = f"{r4:+.1f}%" if r4 is not None else "   N/A"
                r8_s = f"{r8:+.1f}%" if r8 is not None else "   N/A"
                t4_s = f"{t4:>5d}" if t4 is not None else "  N/A"
                t8_s = f"{t8:>5d}" if t8 is not None else "  N/A"
                total = (r4 or 0) + (r8 or 0)
                marker = ""
                if total > best_total:
                    best_total = total
                    best_metric = metric
                print(f"  {metric:<10} {r4_s:>8} {t4_s:>6} {r8_s:>8} {t8_s:>6} {strat:<16} {conf:>5}")
            # Mark best after printing all
            print(f"  >>> BEST METRIC for {asset}: {best_metric.upper()} (combined return: {best_total:+.1f}%)")
    elif mode == 'B':
        run_mode_b(assets_list, horizon_filter=horizons[0] if len(horizons) == 1 else None)
    elif mode in ('D', 'DF'):
        # Per-asset pipeline: D (all horizons) then F for each asset before moving to the next
        for asset in assets_list:
            print(f"\n{'='*60}")
            print(f"  ASSET: {asset}")
            print(f"{'='*60}")
            for h in horizons:
                if len(horizons) > 1:
                    print(f"\n{'#'*60}")
                    print(f"  RUNNING {h}h HORIZON")
                    print(f"{'#'*60}")
                run_mode_d_optuna([asset], horizon=h, n_trials=n_trials, resume=flag_resume)

            # Run strategy comparison right after D for this asset
            if mode == 'DF' or len(horizons) == 2:
                run_strategy_comparison([asset], horizons)
    elif mode == 'F':
        run_strategy_comparison(assets_list, horizons)
    elif mode in ('H', 'HF'):
        # Per-asset: run Mode H (horizon search), then F if HF
        for asset in assets_list:
            print(f"\n{'='*60}")
            print(f"  ASSET: {asset}")
            print(f"{'='*60}")
            h_results = run_mode_h([asset], n_trials=n_trials, resume=flag_resume)

            if mode == 'HF' and h_results:
                # Discover which horizons Mode H saved for this asset
                found_horizons = sorted(set(m['horizon'] for m in h_results if m['coin'] == asset))
                if len(found_horizons) >= 2:
                    print(f"\n  Running Mode F with discovered horizons: {', '.join(str(h)+'h' for h in found_horizons)}")
                    run_strategy_comparison([asset], found_horizons)
                elif len(found_horizons) == 1:
                    print(f"\n  Only 1 horizon found ({found_horizons[0]}h) — running F as single-horizon")
                    run_strategy_comparison([asset], found_horizons)
                else:
                    print(f"\n  No horizons found for {asset} — skipping Mode F")

    print("\nDone!")


if __name__ == '__main__':
    main()
