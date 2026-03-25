"""
Doohan — Production ML Trading System
============================================================
ML trading system for BTC, ETH, XRP, DOGE, SOL, LINK, ADA, AVAX, DOT.
130 features -> walk-forward ML -> BUY/SELL/HOLD signals.
Variable horizon per asset (5h, 6h, 7h, 8h etc.)

Exhaustive grid search:
  Grid: 3 combos × 6 windows × 6 features × 3 gammas = 324 evals (~20 min)
     - Combos: RF+LGBM, XGB+LGBM, RF+XGB
     - Windows: 72, 100, 150, 200, 250, 300
     - Features: 10, 13, 17, 20, 25, 30
     - Gammas: 0.999, 0.997, 0.995

Modes:
  P.   PySR feature discovery (symbolic regression → new features)
  D.   Diagnose — grid optimization (combo × window × gamma × features)
  V.   Validate — backtest top 6 → refine top 3 → pick production model
  DV.  Diagnose + Validate (standard pipeline)
  S.   Strategy comparison (both_agree / either_agree / Xh_only)
  DS.  Diagnose + Strategy
  DVS. Full pipeline: Diagnose → Validate → Strategy
  H.   Horizon sweep (DV per horizon → compare → save best)

CLI Usage:
  python crypto_trading_system_doohan.py P BTC 6h
  python crypto_trading_system_doohan.py DV BTC 8h
  python crypto_trading_system_doohan.py H BTC 5,6,7,8h
  python crypto_trading_system_doohan.py D BTC,ETH 6,7h

Outputs:
  models/crypto_doohan_v1_7_1_best_models.csv       (top 6 candidates from Mode D)
  models/crypto_doohan_v1_7_1_production.csv         (best live performer from Mode V)
  config/trading_config_doohan.json                  (horizon + min_confidence per asset)
  logs/doohan_v171_*.log                             (auto-saved terminal output)

============================================================
TODO (V1.7.1 → production readiness):
============================================================
  1. REFINED-ONLY PRODUCTION SELECTION — D candidates consistently underperform
     refined versions in live backtests. Mode V should select the production model
     from refined configs only (not raw D candidates). D candidates are still used
     for diagnostics and as input to pick top 3 for refinement.

  2. COMPLETE HORIZON SWEEP — Run BTC 7h and 8h:
       python crypto_trading_system_doohan.py DV BTC 7h
       python crypto_trading_system_doohan.py DV BTC 8h
     Results so far: 4h FAILED, 5h marginal (+2.46%), 6h BEST (+3.47% at 80%, holds at 90%).

  3. PICK DOMINANT HORIZON — Compare all 4 horizons (5/6/7/8h) and select winner.
     If 6h wins: run Mode S for strategy selection, expand to all 9 assets.

  4. APPLY EMBARGO FIX TO DEKU — Deku production still uses fixed EMBARGO_CANDLES=4
     for all horizons. Should use train_end = i - horizon like V1.7.1.

  5. WIRE INTO LIVE TRADER — Once winner is validated:
       - Update crypto_live_trader_doohan.py to read V1.7.1 production CSV
       - Update crypto_revolut_doohan.py to point to V1.7.1 configs
       - Or promote V1.7.1 as new Doohan production version

============================================================
Differences: V1.7.1 vs Deku production vs Doohan V1.6 production
============================================================

  EMBARGO (most critical difference):
    V1.7.1:  train_end = i - horizon          (dynamic, scales with prediction horizon)
    V1.6:    train_end = i - EMBARGO_CANDLES   (fixed EMBARGO_CANDLES = 4, too small for 8h)
    Deku:    train_end = i - EMBARGO_CANDLES   (fixed EMBARGO_CANDLES = 4, too small for 8h)
    Impact:  Pre-embargo APFs were inflated 5-26×. Post-embargo realistic range is 1.0-3.0.
             V1.6 and Deku results are NOT comparable to V1.7.1 due to label overlap leakage.

  SEARCH METHOD:
    V1.7.1:  Exhaustive grid (324 evals) + 50-trial Optuna refine per top 3
    V1.6:    Exhaustive grid (432 evals) + 30-trial Optuna refine per top 3
    Deku:    Optuna TPE+Hyperband (150 trials, auto-extend to 200/250)

  COMBOS:
    V1.7.1:  3 combos — RF+LGBM, XGB+LGBM, RF+XGB (dead combos dropped)
    V1.6:    6 combos — RF+LGBM, XGB+LGBM, RF+XGB, RF+GB, RF+LR, GB+LR
    Deku:    26 combos — all pairs + triples + quads + quint of RF/GB/XGB/LR/LGBM

  WINDOWS:
    V1.7.1:  [72, 100, 150, 200, 250, 300]  (36/48 dropped — embargo eats too much)
    V1.6:    [72, 100, 150, 200]
    Deku:    [24, 36, 48, 72, 100, 150, 200]  (Optuna picks from these)

  REFINE:
    V1.7.1:  50 trials per config, ±20h window, ±0.020 gamma, ±5 features
    V1.6:    30 trials per config, ±20h window, ±0.020 gamma, ±5 features
    Deku:    N/A (Optuna does continuous search, no separate refine phase)

  BACKTEST:
    V1.7.1:  336h (2 weeks)
    V1.6:    168h (1 week)
    Deku:    200h (Mode D replay) / 400h (Mode S replay)

  SCORING:
    V1.7.1:  APF (Adjusted Profit Factor) = raw_PF / buyhold_PF
    V1.6:    APF (same formula)
    Deku:    APF default, supports --metric flag (apf/rawpf/calmar/return/rpf_sqrt)

  HOLDOUT:
    V1.7.1:  N/A (uses Mode V live backtest instead)
    V1.6:    N/A (uses Mode V live backtest instead)
    Deku:    3-fold rolling holdout with embargo=4, diversity-aware (top 10 + best per combo)

  FEATURES:
    V1.7.1:  Grid tests [10, 13, 17, 20, 25, 30] — LGBM importance ranking
    V1.6:    Grid tests [10, 13, 17, 20, 25, 30] — LGBM importance ranking
    Deku:    Optuna picks n_features from LGBM-ranked list (continuous range)

  GAMMAS:
    V1.7.1:  Grid tests [0.995, 0.997, 0.999]
    V1.6:    Grid tests [0.995, 0.997, 0.999]
    Deku:    Optuna continuous (0.994, 1.0)

  MODELS:
    V1.7.1:  RF, XGB, LGBM (GB/LR only in combos that include them — but none do)
    V1.6:    RF, GB, XGB, LR, LGBM (all 5 available in combos)
    Deku:    RF, GB, XGB, LR, LGBM (all 5, 26 ensemble combinations)

  WALK-FORWARD STEP:
    V1.7.1:  DIAG_STEP = 36 (same as Deku)
    V1.6:    DIAG_STEP = 36
    Deku:    DIAG_STEP = 36
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
from datetime import datetime as _dt_log


# ============================================================
# AUTO-LOGGING: mirror all stdout to timestamped log file
# ============================================================
class _TeeWriter:
    """Write to both terminal and log file simultaneously."""
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file
        self.encoding = getattr(terminal, 'encoding', 'utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    def isatty(self):
        return self.terminal.isatty()
    def reconfigure(self, **kwargs):
        if hasattr(self.terminal, 'reconfigure'):
            self.terminal.reconfigure(**kwargs)

_LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(_LOG_DIR, exist_ok=True)
_log_path = os.path.join(_LOG_DIR, f"doohan_v171_{_dt_log.now().strftime('%Y%m%d_%H%M%S')}.log")
_log_fh = open(_log_path, 'w', encoding='utf-8')
sys.stdout = _TeeWriter(sys.__stdout__, _log_fh)
sys.stderr = _TeeWriter(sys.__stderr__, _log_fh)


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
    optuna = None  # only needed for Mode D optimization, not signal generation
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
REPLAY_HOURS_S = 400   # Mode S strategy selection — longer window for more trades
DIAG_STEP = 36
DIAG_WINDOWS = [72, 100, 150, 200]  # V1.2: min 72h — 48h also underperforms in live backtests
MIN_COMBO_SIZE = 2   # minimum number of models in ensemble — solos removed (overfit, poor calibration)
DEFAULT_GAMMA = 1.0  # no decay fallback — per-model gamma read from CSV
EMBARGO_CANDLES_DEFAULT = 8  # gap between train/test — must be >= horizon to prevent label overlap leakage


# Per-horizon feature ranges — short tighter to avoid overfitting, long keeps full range
N_FEATURES_RANGE = {
    HORIZON_SHORT: (4, 40),   # short horizon: narrower (overfitting observed with 80+ features)
    HORIZON_LONG:  (4, 80),   # long horizon: full range (kept 75 features and performed well)
}
N_FEATURES_RANGE_DEFAULT = (4, 80)  # fallback for unknown horizons

# Doohan V1.4.1: Continuous search spaces
DOOHAN_GAMMA_MIN = 0.995
DOOHAN_GAMMA_MAX = 0.999
DOOHAN_FEAT_MIN = 10
DOOHAN_FEAT_MAX = 40

# 3-fold rolling holdout — train on ~60%, validate on ~20%, across 3 temporal folds
N_HOLDOUT_FOLDS = 3

# Optuna scoring metric — set via --metric flag (default: apf)
OPTUNA_METRIC = 'apf'
VALID_METRICS = {'apf', 'rawpf', 'calmar', 'return', 'rpf_sqrt'}

# Label mode: 'fee_aware' = label=1 when return > 2×fee (no lookahead bias)
LABEL_MODE = 'fee_aware'

# Mode V: live backtest validation of top 6 candidates
MODE_G_REPLAY_HOURS = 336       # 2 full weeks
MODE_G_CONF_THRESHOLDS = [65, 70, 75, 80, 85, 90]
MODE_G_PRIMARY_CONF = 80        # confidence threshold used to rank live performance
PRODUCTION_CSV = 'models/crypto_doohan_v1_7_1_production.csv'



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

    for period in [1, 2, 3, 4, 5, 6, 7, 8, 12, 24, 48, 72, 120, 240]:
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
        'logret_1h', 'logret_2h', 'logret_3h', 'logret_4h', 'logret_5h',
        'logret_6h', 'logret_7h', 'logret_8h', 'logret_12h', 'logret_24h',
        'logret_48h', 'logret_72h',
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
    """Return the models CSV path — respects MODELS_CSV_OVERRIDE and --metric for test isolation."""
    if MODELS_CSV_OVERRIDE:
        return MODELS_CSV_OVERRIDE
    if OPTUNA_METRIC != 'apf':
        return f'{MODELS_DIR}/crypto_doohan_v1_7_1_best_models_{OPTUNA_METRIC}.csv'
    return f'{MODELS_DIR}/crypto_doohan_v1_7_1_best_models.csv'


def _backup_models_csv():
    """Create a timestamped backup of production CSV before writing (failsafe against contamination)."""
    src = f'{MODELS_DIR}/crypto_doohan_v1_7_1_best_models.csv'
    if os.path.exists(src) and not MODELS_CSV_OVERRIDE:
        import shutil
        bak = f'{MODELS_DIR}/crypto_doohan_v1_7_1_best_models_backup.csv'
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

    # ── PySR symbolic regression features ──
    pysr_added = _compute_pysr_features(df, all_cols, asset_name, horizon, verbose)
    added += pysr_added

    if verbose:
        print(f"    All features: {len(base_cols)} base + {added} macro/sentiment/cross-asset/pysr = {len(all_cols)} total")
    return df, all_cols


def _compute_pysr_features(df, all_cols, asset_name, horizon, verbose=True):
    """Load PySR-discovered expressions from JSON and compute them as new features.
    Expressions are evaluated using the existing feature columns in df.
    Returns the number of PySR features successfully added."""
    import sympy

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    pysr_path = os.path.join(models_dir, f'pysr_{asset_name}_{horizon}h.json')

    if not os.path.exists(pysr_path):
        if verbose:
            print(f"    PySR: no expressions found ({pysr_path}) — skipping")
        return 0

    with open(pysr_path) as f:
        pysr_data = json.load(f)

    expressions = pysr_data.get('expressions', [])
    if not expressions:
        if verbose:
            print(f"    PySR: JSON loaded but no expressions — skipping")
        return 0

    n_added = 0
    for i, expr_info in enumerate(expressions):
        col_name = f'pysr_{i+1}'
        sympy_str = expr_info.get('sympy_format', expr_info.get('equation', ''))

        try:
            sym_expr = sympy.sympify(sympy_str)
            free_symbols = [str(s) for s in sym_expr.free_symbols]

            missing = [s for s in free_symbols if s not in df.columns]
            if missing:
                if verbose:
                    print(f"    PySR #{i+1}: SKIP — missing features: {missing}")
                continue

            sym_vars = list(sym_expr.free_symbols)
            func = sympy.lambdify(sym_vars, sym_expr, modules=['numpy'])

            args = [df[str(s)].values.astype(np.float64) for s in sym_vars]
            values = func(*args)

            values = np.where(np.isfinite(values), values, np.nan)

            df[col_name] = values
            all_cols.append(col_name)
            n_added += 1

            if verbose:
                complexity = expr_info.get('complexity', '?')
                score = expr_info.get('score', 0)
                print(f"    PySR #{i+1}: {col_name} = {sympy_str[:60]}{'...' if len(sympy_str)>60 else ''} "
                      f"(complexity={complexity}, score={score:.4f})")

        except Exception as e:
            if verbose:
                print(f"    PySR #{i+1}: SKIP — eval error: {e}")
            continue

    if verbose and n_added > 0:
        print(f"    PySR: {n_added} features added from {pysr_path}")

    return n_added


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


def _quick_score(df_features, feature_cols, window=ANALYSIS_WINDOW, step=ANALYSIS_STEP, device=None, gamma=1.0, horizon=EMBARGO_CANDLES_DEFAULT):
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
        train_start = max(0, i - window)
        train_end = max(train_start, i - horizon)  # embargo: prevent label overlap leakage
        train    = df_features.iloc[train_start:train_end]
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
        train_end = max(train_start, i - horizon)  # embargo: prevent label overlap leakage
        train = df_features.iloc[train_start:train_end]
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
            confidence = round((1 - avg_proba) * 100)
        else:
            confidence = round(avg_proba * 100)

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
        train_end = max(train_start, i - pred_horizon)  # embargo: gap >= horizon to prevent label overlap leakage
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
                    pred_horizon=horizon or PREDICTION_HORIZON, gamma=gamma
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
        csv_path = f'{MODELS_DIR}/diagnostic_results_doohan_{asset_name}.csv'
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


def run_mode_p(assets_list, horizons):
    """Mode P: PySR symbolic regression feature discovery."""
    print("\n" + "=" * 60)
    print("  MODE P: PySR FEATURE DISCOVERY")
    print("=" * 60)

    try:
        from pysr_discover_features import discover_features, save_results
    except ImportError:
        print("\n  ERROR: pysr_discover_features.py not found!")
        print("  Make sure it's in the same directory as this script.")
        return

    for asset in assets_list:
        for h in horizons:
            print(f"\n{'#'*60}")
            print(f"  {asset} {h}h")
            print(f"{'#'*60}")

            results = discover_features(asset, h)

            if results:
                df_raw = load_data(asset)
                _, all_cols = build_all_features(df_raw, asset_name=asset, horizon=h, verbose=False)
                save_results(asset, h, results, all_cols)
                print(f"\n  Done! Now run Mode DV to test:")
                print(f"  python crypto_trading_system_doohan.py DV {asset} {h}h")
            else:
                print(f"\n  No useful expressions found for {asset} {h}h. Try increasing --iterations.")

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

# Doohan V1.6: Exhaustive grid search
# 6 combos × 4 windows × 6 feature counts × 5 gammas = 720 evals
# LGBM dominates all combos it's in; XGB dominates RF/GB/LR.
# 6 distinct signal groups identified from prior testing.
GRID_COMBOS = [
    'RF+LGBM',    # LGBM without XGB (best group — XGB hurts LGBM)
    'XGB+LGBM',   # LGBM with XGB
    'RF+XGB',     # XGB without LGBM
    # RF+GB, RF+LR, GB+LR dropped — always fail (0 valid results across all V1.6/V1.7 tests)
]
GRID_WINDOWS = [72, 100, 150, 200, 250, 300]  # 36/48 dropped (embargo eats too much), 250/300 added
GRID_FEATURES = [10, 13, 17, 20, 25, 30]
GRID_GAMMAS = [0.999, 0.997, 0.995]

# Refine step: Optuna fine-tuning around top 3 live-validated configs
REFINE_TOP_N = 3                   # how many configs to refine from Mode V
REFINE_TRIALS = 50                 # Optuna trials per config (was 30 in V1.7)
REFINE_GAMMA_RANGE = 0.002         # +/- around grid winner's gamma (minimal — grid already covers 0.995-0.999)
REFINE_FEAT_RANGE = 5              # +/- around grid winner's features
REFINE_WINDOW_RANGE = 20           # +/- around grid winner's window


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
        train_end = max(train_start, i - horizon)  # embargo: gap >= horizon to prevent label overlap leakage
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


def _build_combo_list():
    """Build all ensemble combos from Deku's 5 base models (MIN_COMBO_SIZE=2)."""
    model_names = list(ALL_MODELS.keys())
    combos = []
    for r in range(MIN_COMBO_SIZE, len(model_names) + 1):
        for combo in combinations(model_names, r):
            combos.append('+'.join(combo))
    combo_options = [c for c in combos if 'LGBM' in c]
    return combo_options


def run_mode_d_optuna(assets_list, horizon=PREDICTION_HORIZON, n_trials=DEKU_DEFAULT_TRIALS, resume=False):
    """
    DOOHAN V1.6 Mode D: Exhaustive grid search.

    Pipeline:
    1. Download fresh data + LGBM importance ranking
    2. Exhaustive grid: 6 combos × 4 windows × 6 features × 5 gammas = 720 evals
    3. Save full grid to CSV for analysis
    4. 3-fold holdout ranking → save top 6 → Mode V backtest
    """
    t_mode_start = time.time()
    _kill_orphan_workers()

    n_grid = len(GRID_COMBOS) * len(GRID_WINDOWS) * len(GRID_FEATURES) * len(GRID_GAMMAS)
    print("\n" + "=" * 70)
    print(f"  DOOHAN V1.7.1 MODE D: EXHAUSTIVE GRID -- {horizon}h HORIZON")
    print(f"  {len(GRID_COMBOS)} combos × {len(GRID_WINDOWS)} windows × {len(GRID_FEATURES)} features × {len(GRID_GAMMAS)} gammas = {n_grid} evals")
    metric_label = f" | metric={OPTUNA_METRIC}" if OPTUNA_METRIC != 'apf' else ""
    print(f"  Scoring: {OPTUNA_METRIC}{metric_label}")
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
        print(f"  DOOHAN OPTIMIZATION: {asset_name} ({horizon}h)")
        print(f"{'='*70}")

        df_raw = load_data(asset_name)
        if df_raw is None:
            continue

        # Step 1: Build ALL features, cap at 6 months
        MAX_DIAG_HOURS = 6 * 30 * 24
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

        # 3-fold rolling holdout: each fold trains on ~60%, validates on ~20%
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

        # Optuna trains on fold 1's training partition
        f1_train_s, f1_train_e, _, _ = holdout_folds[0]
        features_np = features_np_all[f1_train_s:f1_train_e]
        labels_np = labels_np_all[f1_train_s:f1_train_e]
        closes_np = closes_np_all[f1_train_s:f1_train_e]
        n = len(features_np)

        print(f"  Optuna trains on fold 1: {n:,} rows | {N_HOLDOUT_FOLDS}-fold holdout ranking after")

        # Per-horizon feature range
        feat_min, feat_max = N_FEATURES_RANGE.get(horizon, N_FEATURES_RANGE_DEFAULT)
        min_n_features = feat_min
        max_n_features = min(len(ranked_features), feat_max)

        # ══════════════════════════════════════════════════════════════
        # Step 3: EXHAUSTIVE GRID SEARCH (V1.6)
        # 6 combos × 4 windows × 6 features × 5 gammas = 720 evals
        # ══════════════════════════════════════════════════════════════
        model_factories = _get_deku_diagnostic_models()

        n_grid = len(GRID_COMBOS) * len(GRID_WINDOWS) * len(GRID_FEATURES) * len(GRID_GAMMAS)
        print(f"\n{'='*70}")
        print(f"  EXHAUSTIVE GRID: {asset_name} {horizon}h — {n_grid} evals")
        print(f"  Combos:   {GRID_COMBOS}")
        print(f"  Windows:  {GRID_WINDOWS}")
        print(f"  Features: {GRID_FEATURES}")
        print(f"  Gammas:   {GRID_GAMMAS}")
        print(f"{'='*70}")

        class _FakeTrial:
            def __init__(self, params, value, source=''):
                self.params = params
                self.value = value
                self._user_attrs = {'sampler': source}
            @property
            def user_attrs(self):
                return self._user_attrs
            def set_user_attr(self, k, v):
                self._user_attrs[k] = v

        t_grid = time.time()
        grid_rows = []  # for CSV export
        all_candidates = []  # _FakeTrial objects for holdout
        best_apf = 0.0
        eval_count = 0

        for combo_name in GRID_COMBOS:
            combo = combo_name.split('+')
            print(f"\n  {'─'*60}")
            print(f"  {combo_name}")
            print(f"  {'─'*60}")

            for window in GRID_WINDOWS:
                for n_feat in GRID_FEATURES:
                    if n_feat > len(ranked_features):
                        continue
                    feat_np = features_np[:, :n_feat]

                    for gamma in GRID_GAMMAS:
                        eval_count += 1

                        result = _deku_eval_with_pruning(
                            feat_np, labels_np, closes_np, combo, window, n,
                            DIAG_STEP, model_factories, gamma=gamma, trial=None,
                            horizon=horizon
                        )

                        if result is None:
                            grid_rows.append({
                                'combo': combo_name, 'window': window,
                                'n_features': n_feat, 'gamma': gamma,
                                'apf': 0, 'return_pct': 0, 'trades': 0,
                                'win_rate': 0, 'accuracy': 0, 'raw_pf': 0,
                                'status': 'FAILED',
                            })
                            continue

                        trades = result[6]
                        score = _compute_optuna_score(result) if trades >= 3 else 0.0
                        ret = result[4]
                        acc = result[2]
                        win_rate = result[5]
                        raw_pf = result[11]

                        grid_rows.append({
                            'combo': combo_name, 'window': window,
                            'n_features': n_feat, 'gamma': gamma,
                            'apf': round(score, 4), 'return_pct': round(ret, 2),
                            'trades': trades, 'win_rate': round(win_rate, 2),
                            'accuracy': round(acc, 4), 'raw_pf': round(raw_pf, 4),
                            'status': 'OK' if trades >= 8 else f'LOW_TR({trades})',
                        })

                        if trades >= 8:
                            all_candidates.append(_FakeTrial(
                                {'combo': combo_name, 'window': window,
                                 'gamma': gamma, 'n_features': n_feat},
                                score, source='Grid'
                            ))

                        marker = ""
                        if score > best_apf and trades >= 8:
                            best_apf = score
                            marker = " <-- BEST"

                        # Progress: print every 60 evals + any new best
                        if marker or eval_count % 60 == 0:
                            elapsed = (time.time() - t_grid) / 60
                            print(f"    [{eval_count:3d}/{n_grid}] {combo_name:10s} w={window:3d} f={n_feat:2d} g={gamma:.3f} | "
                                  f"apf={score:.3f} ret={ret:+.1f}% tr={trades}  ({elapsed:.1f}min){marker}")

        grid_elapsed = (time.time() - t_grid) / 60

        # Save full grid to CSV
        grid_csv_path = os.path.join('models', f'crypto_doohan_v1_7_1_grid_{asset_name}_{horizon}h.csv')
        df_grid = pd.DataFrame(grid_rows)
        df_grid = df_grid.sort_values('apf', ascending=False).reset_index(drop=True)
        df_grid.to_csv(grid_csv_path, index=False)

        # Sort candidates
        all_candidates.sort(key=lambda t: -t.value)

        # Deduplicate
        seen_scores = set()
        completed_trials = []
        for t in all_candidates:
            score_key = (round(t.value, 3), t.params['combo'], t.params['window'],
                         round(t.params['gamma'], 4), t.params['n_features'])
            if score_key not in seen_scores:
                seen_scores.add(score_key)
                completed_trials.append(t)

        n_valid = sum(1 for r in grid_rows if r['status'] == 'OK')
        n_failed = sum(1 for r in grid_rows if r['status'] == 'FAILED')
        n_low_tr = n_grid - n_valid - n_failed
        print(f"\n  {'='*70}")
        print(f"  GRID RESULTS: {asset_name} {horizon}h ({grid_elapsed:.1f} min)")
        print(f"  {'='*70}")
        print(f"  Total evals: {n_grid} | Valid (≥8 trades): {n_valid} | Low trades: {n_low_tr} | Failed: {n_failed}")
        print(f"  Unique candidates: {len(completed_trials)}")
        print(f"  Grid CSV: {grid_csv_path}")

        if completed_trials:
            best_t = completed_trials[0]
            print(f"  Best overall: {best_t.params['combo']} w={best_t.params['window']}h "
                  f"g={best_t.params['gamma']:.4f} f={best_t.params['n_features']} "
                  f"{OPTUNA_METRIC}={best_t.value:.3f}")

        print(f"\n  {'Rank':>4s}  {OPTUNA_METRIC.upper():>7s}  {'Combo':22s}  {'Window':>6s}  {'Gamma':>7s}  {'Feats':>5s}  {'Source':>7s}")
        print(f"  {'-'*75}")
        for i, t in enumerate(completed_trials[:20], 1):
            marker = " <-- BEST" if i == 1 else ""
            src = t.user_attrs.get('sampler', '?')
            print(f"  {i:4d}  {t.value:7.3f}  {t.params['combo']:22s}  {t.params['window']:5d}h  "
                  f"{t.params['gamma']:7.4f}  {t.params['n_features']:5d}  {src:>7s}{marker}")

        # ── 3-FOLD ROLLING HOLDOUT: re-evaluate top candidates on unseen data ──
        candidates = completed_trials[:20]
        N_CANDIDATES = len(candidates)

        print(f"\n  {'='*70}")
        print(f"  {N_HOLDOUT_FOLDS}-FOLD ROLLING HOLDOUT: {asset_name} {horizon}h (top {N_CANDIDATES} candidates)")
        print(f"  {'='*70}")

        holdout_results = []
        min_fold_rows = min(te - ts for _, _, ts, te in holdout_folds)
        if min_fold_rows >= 200:
            for ci, trial in enumerate(candidates):
                c_combo = trial.params['combo'].split('+')
                c_window = trial.params['window']
                c_gamma = trial.params['gamma']
                c_n_feat = trial.params['n_features']

                fold_scores = []
                fold_rets = []
                fold_accs = []
                fold_trades = []
                fold_raw_pfs = []

                for fi, (tr_s, tr_e, te_s, te_e) in enumerate(holdout_folds):
                    fold_feat_test = features_np_all[te_s:te_e, :c_n_feat]
                    fold_labels_test = labels_np_all[te_s:te_e]
                    fold_closes_test = closes_np_all[te_s:te_e]
                    n_fold_test = te_e - te_s

                    HOLDOUT_STEP = 12
                    ho_result = _deku_eval_with_pruning(
                        fold_feat_test, fold_labels_test, fold_closes_test,
                        c_combo, c_window, n_fold_test,
                        HOLDOUT_STEP, model_factories, gamma=c_gamma, trial=None,
                        horizon=horizon
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
                  f"{'F1':>6s}  {'F2':>6s}  {'F3':>6s}  {'IS_'+OPTUNA_METRIC.upper():>8s}  {'Combo':22s}  {'Win':>4s}  {'Gamma':>7s}  {'F':>3s}")
            print(f"  {'-'*125}")
            for i, (trial, avg_sc, avg_ret, avg_acc, tot_tr, avg_rpf, f_scores, f_rets) in enumerate(holdout_results[:10], 1):
                is_score = trial.value
                marker = " <-- BEST" if i == 1 else ""
                gen = "✓" if avg_ret > 0 and avg_acc > 0.55 else "~" if avg_ret > 0 else "✗"
                f_str = "  ".join(f"{s:+5.1f}" for s in f_rets)
                print(f"  {i:4d}  {avg_sc:8.3f}  {avg_ret:+7.1f}%  {avg_acc*100:6.1f}%  {tot_tr:4d}  "
                      f"{f_str}  {is_score:8.3f}  {trial.params['combo']:22s}  {trial.params['window']:3d}h  "
                      f"{trial.params['gamma']:7.4f}  {trial.params['n_features']:3d}  {gen}{marker}")
        else:
            print(f"  Hold-out skipped: smallest fold only {min_fold_rows} rows (need 200+)")

        # Save top 6 holdout candidates with DIVERSITY constraint
        def _diversity_key(trial):
            g = trial.params['gamma']
            f = trial.params['n_features']
            w = trial.params['window']
            g_band = 0 if g < 0.996 else (1 if g < 0.998 else 2)
            f_band = 0 if f <= 15 else (1 if f <= 30 else 2)
            return (w, g_band, f_band)

        DOOHAN_SAVE_TOP_N = 6
        candidates_to_save = []
        seen_regions = set()
        if holdout_results:
            for ho_entry in holdout_results:
                if len(candidates_to_save) >= DOOHAN_SAVE_TOP_N:
                    break
                trial = ho_entry[0]
                region = _diversity_key(trial)
                if region in seen_regions and len(candidates_to_save) >= 1:
                    continue
                seen_regions.add(region)
                candidates_to_save.append(ho_entry)
            if len(candidates_to_save) < DOOHAN_SAVE_TOP_N:
                for ho_entry in holdout_results:
                    if len(candidates_to_save) >= DOOHAN_SAVE_TOP_N:
                        break
                    if ho_entry not in candidates_to_save:
                        candidates_to_save.append(ho_entry)

        if not candidates_to_save and completed_trials:
            for t in completed_trials[:DOOHAN_SAVE_TOP_N]:
                candidates_to_save.append((t, t.value, 0, 0, 0, 0, [], []))

        if not candidates_to_save:
            print(f"  No valid trials for {asset_name} {horizon}h. Skipping.")
            continue

        print(f"\n  {'='*70}")
        print(f"  TOP {len(candidates_to_save)} CANDIDATES: {asset_name} {horizon}h")
        print(f"  {'='*70}")

        for rank_i, ho_entry in enumerate(candidates_to_save, 1):
            trial, ho_score, ho_ret, ho_acc, ho_trades, ho_raw_pf, ho_fold_scores, ho_fold_rets = ho_entry
            c_combo = trial.params['combo']
            c_window = trial.params['window']
            c_gamma = trial.params['gamma']
            c_n_feat = trial.params['n_features']
            c_features = ranked_features[:c_n_feat]

            c_sampler = trial.user_attrs.get('sampler', '?')
            gen_icon = "✓" if ho_ret > 0 and ho_acc > 0.55 else "~" if ho_ret > 0 else "✗"
            fold_str = " / ".join(f"{r:+.1f}%" for r in ho_fold_rets) if ho_fold_rets else "n/a"
            marker = " <-- BEST" if rank_i == 1 else ""
            print(f"  #{rank_i}: {c_combo}  w={c_window}h  g={c_gamma:.4f}  f={c_n_feat}  "
                  f"ho_apf={ho_score:.3f}  ho_ret={ho_ret:+.1f}%  [{fold_str}]  {c_sampler}  {gen_icon}{marker}")

            best_config = {
                'coin': asset_name,
                'best_window': c_window,
                'best_combo': c_combo,
                'models': c_combo,
                'return_pct': round(ho_ret, 2),
                'combined_score': round(ho_score, 4),
                'feature_set': 'D',
                'n_features': c_n_feat,
                'optimal_features': ','.join(c_features),
                'horizon': horizon,
                'gamma': round(c_gamma, 4),
                'rank': rank_i,
                'sampler': c_sampler,
            }
            best_models.append(best_config)

        print(f"  {'='*70}")

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

    # Generate signals + charts for rank #1 models only
    for m in best_models:
        if m.get('rank', 1) != 1:
            continue
        features_list = m['optimal_features'].split(',')
        model_names = m['models'].split('+')
        generate_signals(m['coin'], model_names, m['best_window'],
                         feature_override=features_list, horizon=horizon,
                         gamma=m.get('gamma', 1.0))

    elapsed = (time.time() - t_mode_start) / 60
    print(f"\n  Mode D complete: {elapsed:.1f} min total")


# ============================================================
# MODE V: LIVE BACKTEST VALIDATION (top 6 candidates → pick best)
# ============================================================

def _simulate_with_threshold(signals, conf_threshold):
    """Simulate trades with a confidence threshold filter."""
    cash = 1000.0
    qty = 0
    position = 'cash'
    trades = 0
    trade_log = []
    entry_price = 0

    for sig in signals:
        price = sig['close']
        conf = sig['confidence']

        if conf < conf_threshold:
            continue

        if sig['signal'] == 'BUY' and position == 'cash':
            qty = cash * (1 - TRADING_FEE) / price
            entry_price = price
            cash = 0
            position = 'invested'
            trades += 1
        elif sig['signal'] == 'SELL' and position == 'invested':
            cash = qty * price * (1 - TRADING_FEE)
            pnl_pct = (price / entry_price - 1) * 100
            trade_log.append(pnl_pct)
            qty = 0
            position = 'cash'
            trades += 1

    final = cash if position == 'cash' else qty * signals[-1]['close']
    ret = (final / 1000 - 1) * 100
    winners = sum(1 for t in trade_log if t > 0)
    win_rate = (winners / len(trade_log) * 100) if trade_log else 0

    return {
        'return_pct': ret,
        'final_value': final,
        'trades': trades,
        'round_trips': len(trade_log),
        'win_rate': win_rate,
        'trade_returns': trade_log,
        'still_invested': position == 'invested',
    }


def run_mode_v(assets_list, horizons=None):
    """
    Mode V: Live backtest + Optuna refine + final comparison.
    1. Backtest top 6 from Mode D at conf 70/80/90%
    2. Pick top 3 live performers
    3. Optuna refine those 3 (gamma±0.020, features±5, window±20h, 30 trials)
    4. Backtest the 3 refined configs at conf 70/80/90%
    5. Show combined summary: D candidates vs refined
    6. Save best overall to production CSV
    """
    if horizons is None:
        horizons = list(AVAILABLE_HORIZONS)

    candidates_csv = _get_models_csv_path()
    if not os.path.exists(candidates_csv):
        print(f"  ERROR: {candidates_csv} not found. Run Mode D first.")
        return

    df_candidates = pd.read_csv(candidates_csv)

    print("=" * 80)
    print(f"  MODE V: LIVE BACKTEST + REFINE — {','.join(assets_list)} {','.join(str(h)+'h' for h in horizons)}")
    print(f"  Period: last {MODE_G_REPLAY_HOURS} hours (2 weeks), every hour")
    print(f"  Ranking by: conf>={MODE_G_PRIMARY_CONF}% return")
    print(f"  Pipeline: backtest top 6 → refine top 3 → backtest refined → pick best")
    print(f"  Candidates: {candidates_csv}")
    print(f"  Production: {PRODUCTION_CSV}")
    print("=" * 80)

    all_results = {}
    production_models = []

    for asset in assets_list:
        for horizon in horizons:
            # Check if candidates exist for this (asset, horizon)
            mask = (df_candidates['coin'] == asset) & (df_candidates['horizon'] == horizon)
            if mask.sum() == 0:
                print(f"\n  No candidates for {asset} {horizon}h — skipping")
                continue

            key = f"{asset}_{horizon}h"
            print(f"\n{'#' * 70}")
            print(f"  {asset} {horizon}h")
            print(f"{'#' * 70}")

            # LGBM feature ranking for this asset
            print(f"\n  Computing LGBM feature ranking for {asset}...")
            t0 = time.time()
            df_raw = load_data(asset)
            df_full, all_cols = build_all_features(df_raw, asset_name=asset, horizon=horizon)
            df_clean = df_full.dropna(subset=all_cols + ['label']).reset_index(drop=True)
            importance_df = _test_lgbm_importance(df_clean, all_cols, gamma=1.0)
            ranked_features = importance_df['feature'].tolist()
            print(f"  [Ranking done: {time.time()-t0:.1f}s] — {len(ranked_features)} features ranked")

            configs = {}

            # Doohan candidates from Mode D
            asset_candidates = df_candidates[mask].sort_values(
                'rank' if 'rank' in df_candidates.columns else 'combined_score',
                ascending='rank' in df_candidates.columns)

            for idx, row in asset_candidates.iterrows():
                rank = int(row.get('rank', idx + 1))
                ho_apf = float(row.get('combined_score', 0))
                label = f"D #{rank} (APF={ho_apf:.1f})"

                if pd.notna(row.get('optimal_features', None)) and row['optimal_features']:
                    features = row['optimal_features'].split(',')
                else:
                    features = ranked_features[:int(row['n_features'])]

                configs[label] = {
                    'combo': row['best_combo'],
                    'window': int(row['best_window']),
                    'gamma': float(row['gamma']),
                    'features': features,
                    'n_features': int(row['n_features']),
                    'source': 'doohan',
                    'rank': rank,
                    'ho_apf': ho_apf,
                    'csv_row': row.to_dict(),
                }

            # ── STEP 1: Backtest Mode D candidates ──
            print(f"\n{'='*70}")
            print(f"  STEP 1: BACKTEST MODE D CANDIDATES")
            print(f"{'='*70}")

            results = {}
            for label, cfg in configs.items():
                results[label] = _backtest_one_config(asset, horizon, label, cfg)

            # ── STEP 2: Pick top 3 for refine ──
            doohan_results = [(lbl, r) for lbl, r in results.items()
                              if r and r['cfg'].get('source') == 'doohan'
                              and f'conf_{MODE_G_PRIMARY_CONF}' in r]
            doohan_results.sort(key=lambda x: -x[1][f'conf_{MODE_G_PRIMARY_CONF}']['return_pct'])
            top3_for_refine = doohan_results[:REFINE_TOP_N]

            if top3_for_refine:
                print(f"\n{'='*70}")
                print(f"  STEP 2: OPTUNA REFINE — top {len(top3_for_refine)} live performers")
                print(f"  {REFINE_TRIALS} trials per config | gamma ±{REFINE_GAMMA_RANGE} | "
                      f"features ±{REFINE_FEAT_RANGE} | window ±{REFINE_WINDOW_RANGE}h")
                print(f"{'='*70}")

                for i, (lbl, r) in enumerate(top3_for_refine, 1):
                    cfg = r['cfg']
                    live_ret = r[f'conf_{MODE_G_PRIMARY_CONF}']['return_pct']
                    print(f"  #{i}: {cfg['combo']}  w={cfg['window']}h  g={cfg['gamma']:.4f}  "
                          f"f={cfg['n_features']}  live_ret={live_ret:+.2f}%")

                # ── Run Optuna refine ──
                refined_configs = _refine_top_configs(
                    asset, horizon, top3_for_refine, df_raw, df_clean, all_cols, ranked_features)

                # ── STEP 3: Backtest refined configs ──
                if refined_configs:
                    print(f"\n{'='*70}")
                    print(f"  STEP 3: BACKTEST REFINED CONFIGS")
                    print(f"{'='*70}")

                    for i, rcfg in enumerate(refined_configs, 1):
                        label = f"Refined #{i} (APF={rcfg['apf']:.1f})"
                        cfg = {
                            'combo': rcfg['combo'],
                            'window': rcfg['window'],
                            'gamma': rcfg['gamma'],
                            'features': rcfg['features'],
                            'n_features': rcfg['n_features'],
                            'source': 'refined',
                        }
                        results[label] = _backtest_one_config(asset, horizon, label, cfg)

            all_results[key] = results

            # ── Find overall best across ALL configs and confidence thresholds ──
            # All candidates (D + refined) compete on equal footing via return × win_rate scoring
            all_candidates = [(lbl, r) for lbl, r in results.items() if r]

            if all_candidates:
                # Find best (config, confidence) combination
                # Scoring: negative returns rank below all positives;
                # among positives, return × win_rate favors consistency
                best_label, best_r, best_conf = None, None, MODE_G_PRIMARY_CONF
                best_score = -999
                for lbl, r in all_candidates:
                    for conf in MODE_G_CONF_THRESHOLDS:
                        if f'conf_{conf}' not in r:
                            continue
                        sim = r[f'conf_{conf}']
                        # Require at least 5 trades for the confidence to be valid
                        if sim['trades'] < 5:
                            continue
                        ret = sim['return_pct']
                        wr = sim['win_rate'] / 100.0
                        score = ret * wr if ret > 0 else ret
                        if score > best_score:
                            best_score = score
                            best_label = lbl
                            best_r = r
                            best_conf = conf

                if best_r is None:
                    # Fallback to PRIMARY_CONF
                    for lbl, r in all_candidates:
                        if f'conf_{MODE_G_PRIMARY_CONF}' in r:
                            sim = r[f'conf_{MODE_G_PRIMARY_CONF}']
                            ret = sim['return_pct']
                            wr = sim['win_rate'] / 100.0
                            score = ret * wr if ret > 0 else ret
                            if score > best_score:
                                best_score = score
                                best_label = lbl
                                best_r = r
                                best_conf = MODE_G_PRIMARY_CONF

                if best_r:
                    best_sim = best_r[f'conf_{best_conf}']
                    best_cfg = best_r['cfg']

                    print(f"\n  {'='*70}")
                    print(f"  OVERALL BEST: {best_label}  →  {asset} {horizon}h")
                    print(f"  {best_cfg['combo']}  w={best_cfg['window']}h  g={best_cfg['gamma']:.4f}  f={best_cfg['n_features']}")
                    print(f"  Return (conf>={best_conf}%): {best_sim['return_pct']:+.2f}%  "
                          f"WR={best_sim['win_rate']:.0f}%  trades={best_sim['trades']}")
                    print(f"  {'='*70}")

                    # Build production row
                    if best_cfg.get('source') == 'doohan' and 'csv_row' in best_cfg:
                        prod_row = best_cfg['csv_row'].copy()
                    else:
                        # Refined config — build row from scratch
                        prod_row = {
                            'coin': asset,
                            'best_window': best_cfg['window'],
                            'best_combo': best_cfg['combo'],
                            'models': best_cfg['combo'],
                            'return_pct': round(best_sim['return_pct'], 2),
                            'accuracy': round(best_sim['win_rate'], 1),
                            'combined_score': round(best_score, 4),
                            'feature_set': 'D',
                            'n_features': best_cfg['n_features'],
                            'optimal_features': ','.join(best_cfg['features']),
                            'horizon': horizon,
                            'gamma': round(best_cfg['gamma'], 4),
                            'sampler': 'Refined',
                        }
                    production_models.append((prod_row, horizon, best_conf))

    # ── Combined Summary ──
    print(f"\n\n{'=' * 80}")
    print(f"  SUMMARY: MODE V — D + Refined — Last {MODE_G_REPLAY_HOURS//24} days (scored by return × win_rate)")
    print(f"{'=' * 80}")

    for key, results in all_results.items():
        if not results:
            continue
        first = next((r for r in results.values() if r), None)
        if first:
            print(f"\n  {key} — Buy & Hold: {first['buy_hold']:+.2f}%\n")

        header = (f"  {'Model':<25} | {'Combo':14s} | {'W':>4} | {'G':>6} | {'F':>3} | "
                  f"{'Conf':>4} | {'Return':>8} | {'Tr':>3} | {'RT':>3} | {'WR':>4}")
        print(header)
        print(f"  {'─' * len(header)}")

        for label, r in results.items():
            if not r:
                continue
            cfg = r['cfg']
            for conf in MODE_G_CONF_THRESHOLDS:
                if f'conf_{conf}' not in r:
                    continue
                sim = r[f'conf_{conf}']
                inv = '*' if sim['still_invested'] else ' '
                print(f"  {label:<25} | {cfg['combo']:14s} | {cfg['window']:>3}h | {cfg['gamma']:>.3f} | "
                      f"{cfg['n_features']:>3} | {conf:>3}% | {sim['return_pct']:>+7.2f}% | "
                      f"{sim['trades']:>3} | {sim['round_trips']:>3} | {sim['win_rate']:>3.0f}%{inv}")

    print(f"\n  * = still invested at end of period")

    # Save production models
    if production_models:
        prod_rows = []
        for prod_row, h, best_conf in production_models:
            row = prod_row.copy()
            if 'rank' in row:
                del row['rank']
            prod_rows.append(row)

        df_prod = pd.DataFrame(prod_rows)

        # Merge with existing production CSV (other assets/horizons)
        if os.path.exists(PRODUCTION_CSV):
            df_existing = pd.read_csv(PRODUCTION_CSV)
            for prod_row, h, best_conf in production_models:
                mask = (df_existing['coin'] == prod_row['coin']) & (df_existing['horizon'] == h)
                df_existing = df_existing[~mask]
            df_prod = pd.concat([df_existing, df_prod], ignore_index=True)

        df_prod.to_csv(PRODUCTION_CSV, index=False)
        print(f"\n  Production model saved: {PRODUCTION_CSV}")
        for prod_row, h, best_conf in production_models:
            print(f"    {prod_row['coin']} {h}h: {prod_row['best_combo']}  w={prod_row['best_window']}h  "
                  f"g={prod_row['gamma']}  f={prod_row['n_features']}  conf>={best_conf}%")

        # Write trading config (horizon + min_confidence per asset)
        tcfg_path = f'{CONFIG_DIR}/trading_config_doohan.json'
        try:
            with open(tcfg_path) as f:
                trading_config = json.load(f)
        except Exception:
            trading_config = {}

        for prod_row, h, best_conf in production_models:
            asset = prod_row['coin']
            if asset not in trading_config:
                trading_config[asset] = {
                    'max_position_usd': 0,
                    'symbol': f'{asset}-USD',
                    'enabled': False,
                }
            trading_config[asset]['horizon'] = h
            trading_config[asset]['min_confidence'] = best_conf

        with open(tcfg_path, 'w') as f:
            json.dump(trading_config, f, indent=2)
        print(f"  Trading config updated: {tcfg_path}")
        for prod_row, h, best_conf in production_models:
            print(f"    {prod_row['coin']}: horizon={h}h, min_confidence={best_conf}%")
    else:
        print(f"\n  No production models to save (no valid candidates found)")

    print(f"\n{'=' * 80}")
    return all_results


def _backtest_one_config(asset, horizon, label, cfg):
    """Backtest a single config at all confidence thresholds. Returns results dict."""
    models = cfg['combo'].split('+')
    features = cfg['features']
    window = cfg['window']
    gamma = cfg['gamma']

    print(f"\n{'─' * 70}")
    print(f"  {label}: {asset}")
    print(f"  {cfg['combo']}  w={window}h  g={gamma:.4f}  f={len(features)}")
    print(f"{'─' * 70}")

    signals = generate_signals(
        asset_name=asset,
        model_names=models,
        window_size=window,
        replay_hours=MODE_G_REPLAY_HOURS,
        feature_override=features,
        horizon=horizon,
        gamma=gamma,
    )

    if not signals:
        print(f"  [!] No signals generated")
        return None

    bh_ret = (signals[-1]['close'] / signals[0]['close'] - 1) * 100
    result = {'cfg': cfg, 'signals': len(signals), 'buy_hold': bh_ret}

    for conf in MODE_G_CONF_THRESHOLDS:
        sim = _simulate_with_threshold(signals, conf)
        result[f'conf_{conf}'] = sim
        print(f"  Conf>={conf}%: return={sim['return_pct']:+.2f}%, "
              f"trades={sim['trades']}, round_trips={sim['round_trips']}, "
              f"win_rate={sim['win_rate']:.0f}%"
              f"{' [still in]' if sim['still_invested'] else ''}")

    print(f"  Buy&Hold: {bh_ret:+.2f}%")
    return result


def _refine_top_configs(asset, horizon, top3_for_refine, df_raw, df_clean, all_cols, ranked_features):
    """Run Optuna refine on top 3 live-validated configs. Returns list of refined config dicts."""

    MAX_DIAG_HOURS = 6 * 30 * 24
    df_full_r, _ = build_all_features(df_raw, asset_name=asset, horizon=horizon)
    total_rows = len(df_full_r)
    if total_rows > MAX_DIAG_HOURS:
        df_full_r = df_full_r.tail(MAX_DIAG_HOURS).reset_index(drop=True)
    df_clean_r = df_full_r.dropna(subset=ranked_features + ['label']).reset_index(drop=True)

    # Prepare fold 1 data (same as Mode D)
    features_np_all = df_clean_r[ranked_features].values.astype(np.float64)
    labels_np_all = df_clean_r['label'].values.astype(np.int32)
    closes_np_all = df_clean_r['close'].values.astype(np.float64)
    n_total = len(df_clean_r)

    fold_train_frac = 0.60
    fold_stride = 0.10
    f1_train_s = 0
    f1_train_e = int(n_total * fold_train_frac)

    features_np = features_np_all[f1_train_s:f1_train_e]
    labels_np = labels_np_all[f1_train_s:f1_train_e]
    closes_np = closes_np_all[f1_train_s:f1_train_e]
    n = len(features_np)

    model_factories = _get_deku_diagnostic_models()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    all_refined = []
    t_refine = time.time()

    for cfg_idx, (lbl, r) in enumerate(top3_for_refine):
        cfg = r['cfg']
        combo_name = cfg['combo']
        combo = combo_name.split('+')
        base_window = cfg['window']
        base_gamma = cfg['gamma']
        base_feats = cfg['n_features']

        # Search ranges
        gamma_lo = max(base_gamma - REFINE_GAMMA_RANGE, 0.970)
        gamma_hi = min(base_gamma + REFINE_GAMMA_RANGE, 1.0)
        feat_lo = max(base_feats - REFINE_FEAT_RANGE, 5)
        feat_hi = min(base_feats + REFINE_FEAT_RANGE, len(ranked_features))
        win_lo = max(base_window - REFINE_WINDOW_RANGE, 24)
        win_hi = base_window + REFINE_WINDOW_RANGE

        print(f"\n  {'─'*60}")
        print(f"  Refining #{cfg_idx+1}: {combo_name}  w={base_window}h  g={base_gamma:.4f}  f={base_feats}")
        print(f"  Ranges: gamma[{gamma_lo:.3f}-{gamma_hi:.3f}] features[{feat_lo}-{feat_hi}] window[{win_lo}-{win_hi}]")
        print(f"  {'─'*60}")

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=f'doohan_v171_refine_{asset}_{horizon}h_{cfg_idx}',
        )

        best_refine_apf = 0.0
        r_count = 0

        def refine_objective(trial, _combo=combo, _win_lo=win_lo, _win_hi=win_hi,
                             _gamma_lo=gamma_lo, _gamma_hi=gamma_hi,
                             _feat_lo=feat_lo, _feat_hi=feat_hi):
            nonlocal best_refine_apf, r_count
            r_count += 1

            t_window = trial.suggest_int('window', _win_lo, _win_hi)
            t_gamma = trial.suggest_float('gamma', _gamma_lo, _gamma_hi)
            t_feats = trial.suggest_int('n_features', _feat_lo, _feat_hi)

            feat_np = features_np[:, :t_feats]
            result = _deku_eval_with_pruning(
                feat_np, labels_np, closes_np, _combo, t_window, n,
                DIAG_STEP, model_factories, gamma=t_gamma, trial=None,
                horizon=horizon
            )

            if result is None:
                return 0.0
            if result[6] < 8:
                return 0.0

            score = _compute_optuna_score(result)
            ret = result[4]

            if score > best_refine_apf:
                best_refine_apf = score
                print(f"    #{r_count:3d} NEW BEST: w={t_window} g={t_gamma:.4f} f={t_feats} | "
                      f"apf={score:.3f} ret={ret:+.1f}% trades={result[6]}")

            return score

        study.optimize(refine_objective, n_trials=REFINE_TRIALS, show_progress_bar=False)

        # Get best
        completed = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE and t.value > 0]
        if completed:
            best = max(completed, key=lambda t: t.value)
            print(f"    → Best: w={best.params['window']}h g={best.params['gamma']:.4f} "
                  f"f={best.params['n_features']} apf={best.value:.3f}")
            all_refined.append({
                'combo': combo_name,
                'window': best.params['window'],
                'gamma': best.params['gamma'],
                'n_features': best.params['n_features'],
                'apf': best.value,
                'features': ranked_features[:best.params['n_features']],
            })

    refine_elapsed = (time.time() - t_refine) / 60
    print(f"\n  [Refine: {refine_elapsed:.1f} min]")

    if all_refined:
        all_refined.sort(key=lambda x: -x['apf'])
        for i, r in enumerate(all_refined, 1):
            print(f"  Refined #{i}: {r['combo']}  w={r['window']}h  g={r['gamma']:.4f}  "
                  f"f={r['n_features']}  apf={r['apf']:.3f}")

    return all_refined


# ============================================================
# MODE S: STRATEGY COMPARISON (both_agree / either_agree / 4h / 8h)
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
    print("  MODE S: STRATEGY COMPARISON")
    print("=" * 60)

    # Load trading config for updates
    metric_suffix = f'_{OPTUNA_METRIC}' if OPTUNA_METRIC != 'apf' else ''
    tcfg_path = f'{CONFIG_DIR}/trading_config_doohan{metric_suffix}.json'
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
                                        int(row_h['best_window']), REPLAY_HOURS_S,
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
# MODE H: HORIZON SWEEP (D+G per horizon, compare, save best)
# ============================================================
def run_mode_h(assets_list, horizons, n_trials=None, resume=False, skip_d=False):
    """
    Mode H: Horizon sweep — runs D+G for each specified horizon, then compares
    across horizons to find the best one per asset.

    Usage:
        python crypto_trading_system_doohan.py H BTC 4,5,6,7,8h          # full D+G per horizon
        python crypto_trading_system_doohan.py H BTC,ETH 5,6,7,8h --skip # skip D where results exist

    Flow per asset:
        1. For each horizon: run Mode D (grid, skipped with --skip if results exist) → run Mode V (backtest + refine)
        2. Compare best config from each horizon
        3. Save overall winner to production CSV + trading config
    """
    if n_trials is None:
        n_trials = DEKU_DEFAULT_TRIALS

    if len(horizons) < 2:
        print("  ERROR: Mode H requires at least 2 horizons (e.g., 5,6,7,8h)")
        return

    print("\n" + "=" * 80)
    print(f"  MODE H: HORIZON SWEEP")
    print(f"  Assets: {', '.join(assets_list)}")
    print(f"  Horizons: {', '.join(str(h)+'h' for h in horizons)}")
    skip_label = " (--skip: reuse existing)" if skip_d else ""
    print(f"  Pipeline: D (grid{skip_label}) → G (backtest + refine) per horizon → compare → save best")
    print(f"  Trials: {n_trials} per horizon")
    print("=" * 80)

    t_total = time.time()

    for asset in assets_list:
        print(f"\n{'#' * 80}")
        print(f"  HORIZON SWEEP: {asset} — testing {', '.join(str(h)+'h' for h in horizons)}")
        print(f"{'#' * 80}")

        horizon_results = {}  # horizon -> {best_label, best_cfg, best_conf, best_return, g_results}

        for h in horizons:
            print(f"\n{'=' * 70}")
            print(f"  {asset} — HORIZON {h}h ({horizons.index(h)+1}/{len(horizons)})")
            print(f"{'=' * 70}")

            # Step 1: Run Mode D for this horizon
            skip_this = False
            if skip_d:
                csv_path = _get_models_csv_path()
                if os.path.exists(csv_path):
                    df_bm = pd.read_csv(csv_path)
                    skip_this = len(df_bm[(df_bm['coin'] == asset) & (df_bm['horizon'] == h)]) > 0

            if skip_this:
                print(f"\n  Mode D results already exist for {asset} {h}h — skipping D (--skip)")
            else:
                run_mode_d_optuna([asset], horizon=h, n_trials=n_trials, resume=resume)

            # Step 2: Run Mode V for this horizon
            g_results = run_mode_v([asset], [h])

            # Extract best result from Mode V
            key = f"{asset}_{h}h"
            if g_results and key in g_results:
                results = g_results[key]
                # Find best across all configs and confidence levels
                best_score = -999
                best_label = None
                best_cfg = None
                best_conf = MODE_G_PRIMARY_CONF

                # All candidates (D + refined) compete via return × win_rate scoring
                all_items = {lbl: r for lbl, r in results.items() if r}

                for lbl, r in all_items.items():
                    for conf in MODE_G_CONF_THRESHOLDS:
                        if f'conf_{conf}' not in r:
                            continue
                        sim = r[f'conf_{conf}']
                        if sim['trades'] < 5:
                            continue
                        ret = sim['return_pct']
                        wr = sim['win_rate'] / 100.0
                        score = ret * wr if ret > 0 else ret
                        if score > best_score:
                            best_score = score
                            best_label = lbl
                            best_cfg = r['cfg']
                            best_conf = conf

                if best_label:
                    best_sim = results[best_label][f'conf_{best_conf}']
                    horizon_results[h] = {
                        'label': best_label,
                        'cfg': best_cfg,
                        'conf': best_conf,
                        'return_pct': best_sim['return_pct'],
                        'trades': best_sim['trades'],
                        'win_rate': best_sim['win_rate'],
                        'score': best_score,
                        'buy_hold': results[best_label].get('buy_hold', 0),
                    }
                    print(f"\n  {asset} {h}h WINNER: {best_label}")
                    print(f"  {best_cfg['combo']}  w={best_cfg['window']}h  g={best_cfg['gamma']:.4f}  "
                          f"f={best_cfg['n_features']}  conf>={best_conf}%")
                    print(f"  Return: {best_sim['return_pct']:+.2f}%  trades={best_sim['trades']}  "
                          f"WR={best_sim['win_rate']:.0f}%  score={best_score:.2f}")

        # ── Cross-horizon comparison ──
        if not horizon_results:
            print(f"\n  No valid results for {asset} across any horizon")
            continue

        print(f"\n{'=' * 80}")
        print(f"  HORIZON COMPARISON: {asset}")
        print(f"{'=' * 80}")
        print(f"  {'H':>3} | {'Model':<25} | {'Combo':14s} | {'W':>4} | {'G':>6} | {'F':>3} | "
              f"{'Conf':>4} | {'Return':>8} | {'Tr':>3} | {'WR':>4} | {'B&H':>7}")
        print(f"  {'─' * 100}")

        for h in sorted(horizon_results.keys()):
            hr = horizon_results[h]
            cfg = hr['cfg']
            marker = ""
            print(f"  {h:>2}h | {hr['label']:<25} | {cfg['combo']:14s} | {cfg['window']:>3}h | "
                  f"{cfg['gamma']:>.3f} | {cfg['n_features']:>3} | {hr['conf']:>3}% | "
                  f"{hr['return_pct']:>+7.2f}% | {hr['trades']:>3} | {hr['win_rate']:>3.0f}% | "
                  f"{hr['buy_hold']:>+6.2f}%")

        # Pick overall best horizon (return × win_rate scoring)
        best_h = max(horizon_results.keys(), key=lambda h: horizon_results[h]['score'])
        winner = horizon_results[best_h]

        print(f"\n  >>> BEST HORIZON for {asset}: {best_h}h — {winner['return_pct']:+.2f}% "
              f"(conf>={winner['conf']}%, {winner['trades']} trades)")
        print(f"  >>> {winner['cfg']['combo']}  w={winner['cfg']['window']}h  "
              f"g={winner['cfg']['gamma']:.4f}  f={winner['cfg']['n_features']}")

        # Note: Mode V already saved the production CSV and trading config
        # for each horizon individually. The user can read the comparison above
        # and the best per-horizon models are already in production CSV.
        # Trading config will have the LAST horizon's config — update it to the best:
        tcfg_path = f'{CONFIG_DIR}/trading_config_doohan.json'
        try:
            with open(tcfg_path) as f:
                trading_config = json.load(f)
        except Exception:
            trading_config = {}

        if asset not in trading_config:
            trading_config[asset] = {
                'max_position_usd': 0,
                'symbol': f'{asset}-USD',
                'enabled': False,
            }
        trading_config[asset]['horizon'] = best_h
        trading_config[asset]['min_confidence'] = winner['conf']

        with open(tcfg_path, 'w') as f:
            json.dump(trading_config, f, indent=2)
        print(f"  >>> Trading config: {asset} → horizon={best_h}h, min_confidence={winner['conf']}%")

    elapsed = (time.time() - t_total) / 60
    print(f"\n  Mode H complete: {elapsed:.1f} min total")
    print(f"{'=' * 80}")


# ============================================================
# MAIN MENU
# ============================================================

def main():

    has_macro = os.path.exists(MACRO_DIR)

    # ================================================================
    # CLI: python crypto_trading_system_doohan.py D BTC 4,8h
    # ================================================================
    # Order-independent CLI parser
    # Any order works: MODE ASSETS HORIZONS, ASSETS MODE HORIZONS, etc.
    # Examples:
    #   python crypto_trading_system_doohan.py D BTC 4,8h
    #   python crypto_trading_system_doohan.py BTC D 4,8h --trials 150
    #   python crypto_trading_system_doohan.py H 5,6,7,8h BTC --skip
    #   python crypto_trading_system_doohan.py DF BTC,ETH 4,8h
    # ================================================================
    VALID_MODES = {'P', 'D', 'DS', 'DV', 'DVS', 'S', 'V', 'H'}

    # Parse flags first
    flag_resume = '--resume' in sys.argv
    flag_skip = '--skip' in sys.argv

    # Parse --trials N
    n_trials = DEKU_DEFAULT_TRIALS
    for i, a in enumerate(sys.argv[1:], 1):
        if a == '--trials' and i < len(sys.argv) - 1:
            try:
                n_trials = int(sys.argv[i + 1])
            except ValueError:
                pass

    # --help
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""
Usage: python crypto_trading_system_doohan.py [MODE] [ASSETS] [HORIZONS] [OPTIONS]

  Arguments are order-independent — MODE, ASSETS, HORIZONS can appear in any order.

Modes:
  P       PySR feature discovery (symbolic regression → models/pysr_*.json)
  D       Grid optimization (combo x window x gamma x features)
  V       Validate (top 6 from D → refine top 3 → pick best)
  DV      D then V
  S       Strategy comparison (both_agree / either_agree / Xh_only)
  DS      D then S
  DVS     D then V then S (full pipeline)
  H       Horizon sweep (D+V per horizon → compare → save best)

Assets:
  BTC,ETH,LINK,...   Comma-separated asset names (default: all)

Horizons:
  5,6,7,8h           Comma-separated horizons in hours (default: 4,8h)

Options:
  --trials N          Number of Optuna trials (default: 150)
  --metric NAME       Scoring metric: apf, rawpf, calmar, return, rpf_sqrt, all
  --skip              Mode H only: skip Mode D for horizons that already have results
  --resume            Resume interrupted Optuna study
  --help, -h          Show this help

Examples:
  python crypto_trading_system_doohan.py P BTC 6h                  # discover PySR features (~30-120 min)
  python crypto_trading_system_doohan.py H BTC 5,6,7,8h          # full horizon sweep
  python crypto_trading_system_doohan.py H BTC 5,6,7h --skip     # skip D, re-run V only
  python crypto_trading_system_doohan.py DV ETH 6h               # optimize + validate ETH 6h
  python crypto_trading_system_doohan.py D BTC,ETH 8h --trials 200
  python crypto_trading_system_doohan.py V BTC 6h                 # re-validate existing results
  python crypto_trading_system_doohan.py BTC D 8h                 # order doesn't matter
""")
        return

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

    # Classify positional args (order-independent)
    cli_args = [a for a in sys.argv[1:] if not a.startswith('--')]
    # Remove values that follow --trials or --metric
    skip_next = set()
    for i, a in enumerate(sys.argv[1:], 1):
        if a in ('--trials', '--metric') and i < len(sys.argv) - 1:
            skip_next.add(sys.argv[i + 1])
    cli_args = [a for a in cli_args if a not in skip_next]

    mode = None
    assets_list = None
    horizons = None

    for arg in cli_args:
        upper = arg.upper()
        # Check if it's a mode
        if upper in VALID_MODES and mode is None:
            mode = upper
        # Check if it's horizons (ends with h, contains digits)
        elif arg.lower().endswith('h') and arg[:-1].replace(',', '').isdigit():
            horizons = [int(h) for h in arg[:-1].split(',')]
        # Otherwise treat as asset list
        else:
            parsed = [a.strip().upper() for a in arg.split(',') if a.strip().upper() in ASSETS]
            if parsed:
                assets_list = parsed

    if mode and mode in VALID_MODES:
        # Defaults
        if assets_list is None:
            assets_list = list(ASSETS.keys())
        if horizons is None:
            horizons = list(AVAILABLE_HORIZONS) if mode in ('P', 'DS', 'DV', 'DVS') else [HORIZON_SHORT]

        trials_str = f" | {n_trials} trials" if mode in ('D', 'DS', 'DV', 'DVS', 'H') else ""
        skip_str = " | --skip" if flag_skip and mode == 'H' else ""
        h_str = ','.join(str(h)+'h' for h in horizons)
        print("=" * 60)
        print(f"  DOOHAN: Mode {mode} | {','.join(assets_list)} | {h_str}{trials_str}{skip_str}")
        print("=" * 60)

    else:

        print("=" * 60)
        print("  CRYPTO HOURLY ML TRADING SYSTEM -- DOOHAN")
        print("  Exhaustive grid + Optuna refine + live backtest validation")
        print(f"  Prediction: variable horizons (specify via CLI)")
        print(f"  Macro data: {'FOUND' if has_macro else 'NOT FOUND -- run download_macro_data.py'}")
        print("=" * 60)

        print("\nChoose mode:")
        print("  P.  PySR FEATURE DISCOVERY (symbolic regression → new features)")
        print("  D.  GRID OPTIMIZATION (combo × window × gamma × features)")
        print("  DV. D then V (grid + validate top 6 + refine top 3 + pick best)")
        print("  DVS. DV then S (full pipeline + strategy comparison)")
        print("  DS. D then S (optimize + strategy comparison)")
        print(f"  S.  STRATEGY COMPARISON (both_agree / either_agree / {HORIZON_SHORT}h / {HORIZON_LONG}h)")
        print("  V.  VALIDATE (top 6 candidates from Mode D, pick best)")
        print("  H.  HORIZON SWEEP (D+V per horizon, compare, save best)")
        mode = input("\nEnter P/D/DV/DVS/DS/S/V/H: ").strip().upper()


        if mode not in VALID_MODES:
            print("Invalid choice. Defaulting to DV.")
            mode = 'DV'

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

        if mode in ('D', 'DS', 'DV', 'DVS'):
            try:
                trials_input = input(f"Number of Optuna trials [{DEKU_DEFAULT_TRIALS}]: ").strip()
                if trials_input:
                    n_trials = int(trials_input)
            except ValueError:
                pass

    # Execute mode
    if run_all_metrics and mode in ('D', 'DS'):
        # --metric all: run Mode DS for each metric, then compare
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

            # Read the trading config that Mode S just wrote
            metric_suffix = f'_{metric}' if metric != 'apf' else ''
            tcfg_path = f'{CONFIG_DIR}/trading_config_doohan{metric_suffix}.json'
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
    elif mode == 'P':
        run_mode_p(assets_list, horizons)
    elif mode in ('D', 'DS', 'DV', 'DVS'):
        # Per-asset pipeline: D (all horizons) then V and/or S for each asset
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

            # Run validate and/or strategy comparison after D
            if mode in ('DVS', 'DV'):
                run_mode_v([asset], horizons)
                if mode == 'DVS':
                    run_strategy_comparison([asset], horizons)
            elif mode == 'DS' or (mode == 'D' and len(horizons) == 2):
                run_strategy_comparison([asset], horizons)
    elif mode == 'S':
        run_strategy_comparison(assets_list, horizons)
    elif mode == 'V':
        run_mode_v(assets_list, horizons)
    elif mode == 'H':
        run_mode_h(assets_list, horizons, n_trials=n_trials, resume=flag_resume, skip_d=flag_skip)

    print("\nDone!")


if __name__ == '__main__':
    main()
