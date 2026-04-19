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
  python crypto_trading_system_ed.py P BTC 6h
  python crypto_trading_system_ed.py DV BTC 8h
  python crypto_trading_system_ed.py H BTC 5,6,7,8h
  python crypto_trading_system_ed.py D BTC,ETH 6,7h

Outputs:
  models/crypto_ed_best_models.csv       (top 6 candidates from Mode D)
  models/crypto_ed_production.csv         (best live performer from Mode V)
  config/regime_config_ed.json                  (horizon + min_confidence per asset)
  logs/ed_v1_*.log                             (auto-saved terminal output)

============================================================
TODO (V1.7.1 → production readiness):
============================================================
  1. REFINED-ONLY PRODUCTION SELECTION — D candidates consistently underperform
     refined versions in live backtests. Mode V should select the production model
     from refined configs only (not raw D candidates). D candidates are still used
     for diagnostics and as input to pick top 3 for refinement.

  2. COMPLETE HORIZON SWEEP — Run BTC 7h and 8h:
       python crypto_trading_system_ed.py DV BTC 7h
       python crypto_trading_system_ed.py DV BTC 8h
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
import pickle
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
_log_path = os.path.join(_LOG_DIR, f"ed_v1_{_dt_log.now().strftime('%Y%m%d_%H%M%S')}.log")
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
    # Crypto universe pruned 2026-04-19: dropped DOGE/ADA/AVAX/DOT (weak priors, no
    # diversification edge vs ETH). Kept: ETH (prod), BTC (standby), SOL (testing),
    # LINK (standby), XRP (decorrelation candidate).
    'BTC':   {'source': 'binance', 'ticker': 'BTC/USDT',  'file': 'data/btc_hourly_data.csv',  'start': '2017-08-01T00:00:00Z'},
    'ETH':   {'source': 'binance', 'ticker': 'ETH/USDT',  'file': 'data/eth_hourly_data.csv',  'start': '2017-08-01T00:00:00Z'},
    'XRP':   {'source': 'binance', 'ticker': 'XRP/USDT',  'file': 'data/xrp_hourly_data.csv',  'start': '2018-05-01T00:00:00Z'},
    'SOL':   {'source': 'binance', 'ticker': 'SOL/USDT',  'file': 'data/sol_hourly_data.csv',  'start': '2020-08-01T00:00:00Z'},
    'LINK':  {'source': 'binance', 'ticker': 'LINK/USDT', 'file': 'data/link_hourly_data.csv', 'start': '2019-01-01T00:00:00Z'},
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
TRADING_FEE_BASE = 0.0009  # 0.09% Revolut X taker fee (worst-case, pure-taker reality)
SLIPPAGE = 0.0002          # 0.02% estimated slippage (market impact, spread)
TRADING_FEE = TRADING_FEE_BASE + SLIPPAGE  # 0.11% taker+slippage — used ONLY for LABEL generation (`2 * TRADING_FEE` break-even target)

# Backtest simulations use a more realistic blend: live trader is maker-first
# (~95% maker @ 0% fee, ~5% taker fallback @ 0.11%). Measured blend ≈ 1.6 bps/leg.
# 5 bps/leg = 3× safety margin for periods when maker success degrades.
# Applied uniformly across Mode D/V/R/S/T/G backtest sims so relative policy
# comparisons stay consistent. See feedback_backtest_fees_matter.md in memory.
BACKTEST_FEE_PER_LEG = 0.0005
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
MODE_G_REPLAY_HOURS = 1440      # default 2 months (was 336=2wks)
MODE_G_CONF_THRESHOLDS = [65, 70, 75, 80, 85, 90]
MODE_G_PRIMARY_CONF = 80        # confidence threshold used to rank live performance
PRODUCTION_CSV = 'models/crypto_ed_production.csv'
REGIME_CONFIG_PATH = 'config/regime_config_ed.json'


def _atomic_write_json(path, data):
    """Write JSON atomically: temp file + os.replace, so a crash mid-write can't corrupt the target."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


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
    _compute_pysr_features(df_full, all_cols, asset_name, horizon, verbose=False)
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
        return f'{MODELS_DIR}/crypto_ed_best_models_{OPTUNA_METRIC}.csv'
    return f'{MODELS_DIR}/crypto_ed_best_models.csv'


def _backup_models_csv():
    """Create a timestamped backup of production CSV before writing (failsafe against contamination)."""
    src = f'{MODELS_DIR}/crypto_ed_best_models.csv'
    if os.path.exists(src) and not MODELS_CSV_OVERRIDE:
        import shutil
        bak = f'{MODELS_DIR}/crypto_ed_best_models_backup.csv'
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
        idx = pd.to_datetime(df.index)
        df.index = idx.tz_localize(None) if idx.tz is None else idx.tz_convert(None)
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


def _compute_gdelt_features(gdelt_df, prefix='gp_'):
    """Compute geopolitical tension features from GDELT data.
    Input columns: iran_vol, iran_tone, geopolitical_vol, geopolitical_tone
    Output features: volume spikes, tone shifts, rolling z-scores."""
    features = pd.DataFrame(index=gdelt_df.index)
    for col in gdelt_df.columns:
        s = gdelt_df[col].astype(float)
        tag = f"{prefix}{col}"
        # Raw value
        features[f'{tag}'] = s
        # Z-score (rolling 168h = 1 week)
        roll_mean = s.rolling(168, min_periods=24).mean()
        roll_std = s.rolling(168, min_periods=24).std()
        features[f'{tag}_zscore'] = (s - roll_mean) / (roll_std + 1e-10)
        # Short-term change (spike detection)
        features[f'{tag}_chg4h'] = s.diff(4)
        features[f'{tag}_chg24h'] = s.diff(24)
        # Spike flag (> 2 std above mean)
        features[f'{tag}_spike'] = ((s - roll_mean) / (roll_std + 1e-10) > 2.0).astype(float)
    return features


def _compute_onchain_features(oc_df, prefix='oc_'):
    """Compute on-chain feature derivations.
    Input columns (daily): active_addresses, mvrv, fees_native, exchange_inflow,
    exchange_outflow, hashrate, tx_count, exchange_netflow, [sopr for BTC].
    Per column: chg1d, chg5d, chg10d, zscore30d, ratio_ma30d."""
    features = pd.DataFrame(index=oc_df.index)
    for col in oc_df.columns:
        s = oc_df[col].astype(float)
        tag = f"{prefix}{col}"
        roll_mean_30 = s.rolling(30, min_periods=10).mean()
        roll_std_30 = s.rolling(30, min_periods=10).std()
        features[f'{tag}_chg1d'] = s.pct_change(1) * 100
        features[f'{tag}_chg5d'] = s.pct_change(5) * 100
        features[f'{tag}_chg10d'] = s.pct_change(10) * 100
        features[f'{tag}_zscore30d'] = (s - roll_mean_30) / (roll_std_30 + 1e-10)
        features[f'{tag}_ratio_ma30d'] = s / (roll_mean_30 + 1e-10)
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

    # On-chain features (daily — merge on date with ffill)
    onchain_asset_map = {'BTC': 'btc', 'ETH': 'eth'}
    oc_key = onchain_asset_map.get(asset_name)
    if oc_key is not None:
        oc_df = _load_macro_csv(f'onchain_{oc_key}.csv')
        if oc_df is not None:
            oc_feats = _compute_onchain_features(oc_df, prefix='oc_')
            oc_feats['_merge_date'] = oc_feats.index.normalize()
            df = df.merge(oc_feats, on='_merge_date', how='left')
            new_cols = [c for c in oc_feats.columns if c != '_merge_date']
            all_cols.extend(new_cols)
            added += len(new_cols)

    # GDELT geopolitical features (hourly — merge on datetime, not date)
    gdelt_df = _load_macro_csv('gdelt_geopolitical.csv')
    if gdelt_df is not None:
        gp_feats = _compute_gdelt_features(gdelt_df, prefix='gp_')
        gp_feats['_merge_hour'] = gp_feats.index.floor('h')
        df['_merge_hour'] = pd.to_datetime(df['datetime']).dt.floor('h')
        df = df.merge(gp_feats, on='_merge_hour', how='left')
        df = df.drop(columns=['_merge_hour'], errors='ignore')
        new_cols = [c for c in gp_feats.columns if c != '_merge_hour']
        all_cols.extend(new_cols)
        added += len(new_cols)

    df = df.drop(columns=['_merge_date', '_merge_hour'], errors='ignore')
    if '_merge_date' in all_cols:
        all_cols.remove('_merge_date')
    if '_merge_hour' in all_cols:
        all_cols.remove('_merge_hour')
    all_cols = list(dict.fromkeys(all_cols))
    all_cols = [c for c in all_cols if c in df.columns]
    macro_cols = [c for c in all_cols if c not in base_cols]
    if macro_cols:
        df[macro_cols] = df[macro_cols].ffill()

    # Load funding rate — as BOTH regime gate (_funding_rate) AND feature (deriv_funding_rate)
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
            print(f"    Funding rate: {n_funding} rows loaded (regime gate + feature)")
        # Create feature versions (no underscore = enters feature matrix)
        df['deriv_funding_rate'] = df['_funding_rate']
        df['deriv_funding_chg1d'] = df['deriv_funding_rate'].diff(24) * 100  # 24h change
        df['deriv_funding_zscore'] = (df['deriv_funding_rate'] - df['deriv_funding_rate'].rolling(168, min_periods=24).mean()) / \
                                     (df['deriv_funding_rate'].rolling(168, min_periods=24).std() + 1e-10)
        for col in ['deriv_funding_rate', 'deriv_funding_chg1d', 'deriv_funding_zscore']:
            if col not in all_cols:
                all_cols.append(col)
    else:
        df['_funding_rate'] = np.nan

    # Load open interest from derivatives CSV
    deriv_file = os.path.join(MACRO_DIR, f'derivatives_{asset_name.lower()}.csv')
    if os.path.exists(deriv_file):
        try:
            deriv_df = pd.read_csv(deriv_file, parse_dates=['datetime'] if 'datetime' in pd.read_csv(deriv_file, nrows=0).columns else [0])
            if deriv_df.columns[0] != 'datetime':
                deriv_df = deriv_df.rename(columns={deriv_df.columns[0]: 'datetime'})
            deriv_df['datetime'] = pd.to_datetime(deriv_df['datetime'])
            if deriv_df['datetime'].dt.tz is not None:
                deriv_df['datetime'] = deriv_df['datetime'].dt.tz_localize(None)
            deriv_df['_merge_dt'] = deriv_df['datetime'].dt.floor('h')
            df['_merge_dt'] = pd.to_datetime(df['datetime']).dt.floor('h')

            oi_cols_added = 0
            if 'open_interest_usd' in deriv_df.columns:
                oi_merge = deriv_df[['_merge_dt', 'open_interest_usd']].drop_duplicates(subset='_merge_dt', keep='last')
                df = df.merge(oi_merge, on='_merge_dt', how='left')
                df['open_interest_usd'] = df['open_interest_usd'].ffill()
                # Create feature derivatives
                df['deriv_oi_chg1d'] = df['open_interest_usd'].pct_change(24) * 100
                df['deriv_oi_chg3d'] = df['open_interest_usd'].pct_change(72) * 100
                df['deriv_oi_zscore'] = (df['open_interest_usd'] - df['open_interest_usd'].rolling(168, min_periods=24).mean()) / \
                                        (df['open_interest_usd'].rolling(168, min_periods=24).std() + 1e-10)
                for col in ['deriv_oi_chg1d', 'deriv_oi_chg3d', 'deriv_oi_zscore']:
                    if col not in all_cols:
                        all_cols.append(col)
                        oi_cols_added += 1
                df = df.drop(columns=['open_interest_usd'], errors='ignore')
            df = df.drop(columns=['_merge_dt'], errors='ignore')
            if verbose and oi_cols_added > 0:
                print(f"    Open interest: {oi_cols_added} features added")
        except Exception as e:
            if verbose:
                print(f"    Open interest: failed to load ({e})")

    # Load stablecoin flows
    stable_file = os.path.join(MACRO_DIR, 'stablecoin_flows.csv')
    if os.path.exists(stable_file):
        try:
            stable_df = pd.read_csv(stable_file, parse_dates=[0], index_col=0)
            if stable_df.index.tz is not None:
                stable_df.index = stable_df.index.tz_localize(None)
            stable_df['_merge_date'] = stable_df.index.normalize()
            df['_merge_date'] = pd.to_datetime(df['datetime']).dt.normalize()

            stable_cols_added = 0
            if 'total_stable_mcap' in stable_df.columns:
                s = stable_df['total_stable_mcap']
                stable_feats = pd.DataFrame(index=stable_df.index)
                stable_feats['stable_mcap_chg1d'] = s.pct_change(1) * 100
                stable_feats['stable_mcap_chg7d'] = s.pct_change(7) * 100
                stable_feats['stable_mcap_zscore'] = (s - s.rolling(30, min_periods=7).mean()) / (s.rolling(30, min_periods=7).std() + 1e-10)
                stable_feats['_merge_date'] = stable_feats.index.normalize()

                for col in ['stable_mcap_chg1d', 'stable_mcap_chg7d', 'stable_mcap_zscore']:
                    merge_df = stable_feats[['_merge_date', col]].dropna()
                    df = df.merge(merge_df, on='_merge_date', how='left')
                    df[col] = df[col].ffill()
                    if col not in all_cols:
                        all_cols.append(col)
                        stable_cols_added += 1

            df = df.drop(columns=['_merge_date'], errors='ignore')
            if verbose and stable_cols_added > 0:
                print(f"    Stablecoin flows: {stable_cols_added} features added")
        except Exception as e:
            if verbose:
                print(f"    Stablecoin flows: failed to load ({e})")

    # Load orderbook imbalance snapshots
    ob_file = os.path.join(MACRO_DIR, 'orderbook_snapshots.csv')
    if os.path.exists(ob_file):
        try:
            ob_df = pd.read_csv(ob_file, parse_dates=['datetime'])
            ob_asset = ob_df[ob_df['asset'] == asset_name].copy()
            if len(ob_asset) > 10:
                ob_asset['_merge_dt'] = pd.to_datetime(ob_asset['datetime']).dt.floor('h')
                df['_merge_dt'] = pd.to_datetime(df['datetime']).dt.floor('h')
                ob_cols_added = 0
                for col in ['ob_imbalance', 'spread_bps']:
                    if col in ob_asset.columns:
                        merge_df = ob_asset[['_merge_dt', col]].drop_duplicates(subset='_merge_dt', keep='last')
                        df = df.merge(merge_df, on='_merge_dt', how='left')
                        df[col] = df[col].ffill()
                        if col not in all_cols:
                            all_cols.append(col)
                            ob_cols_added += 1
                df = df.drop(columns=['_merge_dt'], errors='ignore')
                if verbose and ob_cols_added > 0:
                    print(f"    Orderbook: {ob_cols_added} features added ({len(ob_asset)} snapshots)")
        except Exception as e:
            if verbose:
                print(f"    Orderbook: failed to load ({e})")

    # Load options IV skew snapshots
    iv_file = os.path.join(MACRO_DIR, 'options_iv_snapshot.csv')
    if os.path.exists(iv_file):
        try:
            iv_df = pd.read_csv(iv_file, parse_dates=['datetime'])
            iv_asset = iv_df[iv_df['asset'] == asset_name].copy()
            if len(iv_asset) > 10:
                iv_asset['_merge_dt'] = pd.to_datetime(iv_asset['datetime']).dt.floor('h')
                df['_merge_dt'] = pd.to_datetime(df['datetime']).dt.floor('h')
                iv_cols_added = 0
                for col in ['avg_iv', 'iv_skew']:
                    if col in iv_asset.columns:
                        merge_df = iv_asset[['_merge_dt', col]].drop_duplicates(subset='_merge_dt', keep='last')
                        df = df.merge(merge_df, on='_merge_dt', how='left')
                        df[col] = df[col].ffill()
                        if col not in all_cols:
                            all_cols.append(col)
                            iv_cols_added += 1
                df = df.drop(columns=['_merge_dt'], errors='ignore')
                if verbose and iv_cols_added > 0:
                    print(f"    Options IV: {iv_cols_added} features added ({len(iv_asset)} snapshots)")
        except Exception as e:
            if verbose:
                print(f"    Options IV: failed to load ({e})")

    # Exchange netflow shorter windows (1d, 3d) — from on-chain data
    if 'oc_exchange_netflow' in df.columns:
        df['oc_exchange_netflow_chg1d'] = df['oc_exchange_netflow'].pct_change(24) * 100
        df['oc_exchange_netflow_chg3d'] = df['oc_exchange_netflow'].pct_change(72) * 100
        for col in ['oc_exchange_netflow_chg1d', 'oc_exchange_netflow_chg3d']:
            if col not in all_cols:
                all_cols.append(col)

    if verbose:
        print(f"    All features: {len(base_cols)} base + {len(all_cols) - len(base_cols)} macro/deriv/sentiment/cross-asset = {len(all_cols)} total")
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

    # Build all expressions with their names and parsed forms
    parsed = []
    for i, expr_info in enumerate(expressions):
        col_name = f'pysr_{i+1}'
        sympy_str = expr_info.get('sympy_format', expr_info.get('equation', ''))
        try:
            sym_expr = sympy.sympify(sympy_str)
            free_symbols = [str(s) for s in sym_expr.free_symbols]
            parsed.append((i, col_name, sympy_str, sym_expr, free_symbols, expr_info))
        except Exception as e:
            if verbose:
                print(f"    PySR #{i+1}: SKIP — parse error: {e}")

    # Iteratively resolve: compute expressions whose dependencies are all available,
    # then retry remaining ones (handles chained pysr_N → pysr_M dependencies)
    n_added = 0
    remaining = list(parsed)
    max_passes = len(remaining) + 1
    for _pass in range(max_passes):
        if not remaining:
            break
        progress = False
        still_remaining = []
        for i, col_name, sympy_str, sym_expr, free_symbols, expr_info in remaining:
            missing = [s for s in free_symbols if s not in df.columns]
            if missing:
                still_remaining.append((i, col_name, sympy_str, sym_expr, free_symbols, expr_info))
                continue
            try:
                sym_vars = list(sym_expr.free_symbols)
                func = sympy.lambdify(sym_vars, sym_expr, modules=['numpy'])
                args = [df[str(s)].values.astype(np.float64) for s in sym_vars]
                values = func(*args)
                values = np.where(np.isfinite(values), values, np.nan)
                df[col_name] = values
                all_cols.append(col_name)
                n_added += 1
                progress = True
                if verbose:
                    complexity = expr_info.get('complexity', '?')
                    score = expr_info.get('score', 0)
                    print(f"    PySR #{i+1}: {col_name} = {sympy_str[:60]}{'...' if len(sympy_str)>60 else ''} "
                          f"(complexity={complexity}, score={score:.4f})")
            except Exception as e:
                if verbose:
                    print(f"    PySR #{i+1}: SKIP — eval error: {e}")
        remaining = still_remaining
        if not progress:
            break  # no more resolvable

    # Report any unresolved
    for i, col_name, sympy_str, sym_expr, free_symbols, expr_info in remaining:
        missing = [s for s in free_symbols if s not in df.columns]
        if verbose:
            print(f"    PySR #{i+1}: SKIP — unresolvable deps: {missing}")

    if verbose and n_added > 0:
        print(f"    PySR: {n_added} features added from {pysr_path}")

    return n_added


def _check_pysr_leakage(features, asset, horizon):
    """Check if PySR features in a config were discovered on clean (non-overlapping) data.

    Returns (is_clean, message). Used before writing to production CSV.
    """
    pysr_features = [f for f in features if f.startswith('pysr_')]
    if not pysr_features:
        return True, "No PySR features"

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    pysr_path = os.path.join(models_dir, f'pysr_{asset}_{horizon}h.json')

    if not os.path.exists(pysr_path):
        return False, f"PySR features used but no JSON found at {pysr_path}"

    with open(pysr_path) as f:
        pysr_data = json.load(f)

    method = pysr_data.get('discovery_method', 'unknown')
    if method == 'historical':
        return True, f"PySR clean (discovery_method=historical, {len(pysr_features)} features)"
    else:
        return False, (f"LEAKAGE: PySR discovery_method='{method}' — "
                       f"formulas may be fitted on same data window as Mode D. "
                       f"Re-run Mode P first to generate clean PySR.")


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
            sell_px = price * (1 - BACKTEST_FEE_PER_LEG)
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
        sell_px = last_px * (1 - BACKTEST_FEE_PER_LEG)
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
            btc_held = cash * (1 - BACKTEST_FEE_PER_LEG) / price
            cash = 0
            position = 'invested'
        elif sig['signal'] == 'SELL' and position == 'invested':
            # SELL: sell BTC, pay fee, get cash
            cash = btc_held * price * (1 - BACKTEST_FEE_PER_LEG)
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
            sell_price = price * (1 - BACKTEST_FEE_PER_LEG)  # effective exit = price - fee
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
        sell_price = last_price * (1 - BACKTEST_FEE_PER_LEG)
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
            o_held = o_cash * (1 - BACKTEST_FEE_PER_LEG) / price; o_cash = 0; o_in = True; o_entry = price; o_trades += 1
        elif m['combined'] == 'SELL' and o_in:
            o_cash = o_held * price * (1 - BACKTEST_FEE_PER_LEG)
            if price > o_entry: o_wins += 1
            o_held = 0; o_in = False
    # Close open position at last price for overall stats
    if o_in and merged:
        last_px = merged[-1]['close']
        o_cash = o_held * last_px * (1 - BACKTEST_FEE_PER_LEG)
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
                p_held = p_cash * (1 - BACKTEST_FEE_PER_LEG) / price; p_cash = 0; p_in = True; p_entry = price; p_trades += 1
            elif d['combined'] == 'SELL' and p_in:
                p_cash = p_held * price * (1 - BACKTEST_FEE_PER_LEG)
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

            results, pysr_rows = discover_features(asset, h)

            if results:
                df_raw = load_data(asset)
                _, all_cols = build_all_features(df_raw, asset_name=asset, horizon=h, verbose=False)
                save_results(asset, h, results, all_cols, pysr_rows=pysr_rows)
                print(f"\n  Done! Now run Mode DV to test:")
                print(f"  python crypto_trading_system_ed.py DV {asset} {h}h")
            else:
                print(f"\n  No useful expressions found for {asset} {h}h. Try increasing --iterations.")

    print("\n" + "=" * 60)
    print("  MODE P COMPLETE")
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
            sell_price = price * (1 - BACKTEST_FEE_PER_LEG)
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
        sell_price = last_price * (1 - BACKTEST_FEE_PER_LEG)
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


def run_mode_d_optuna(assets_list, horizon=PREDICTION_HORIZON, n_trials=DEKU_DEFAULT_TRIALS, resume=False, replay_hours=None):
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
    print(f"  ED V1.0 MODE D: EXHAUSTIVE GRID -- {horizon}h HORIZON")
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

        # Step 1: Build ALL features, cap at replay_hours or 2 months default
        MAX_DIAG_HOURS = replay_hours if replay_hours else 60 * 24
        period_label = f"{replay_hours}h" if replay_hours else "last 2mo"
        print(f"\n  Building all features (horizon={horizon}h, period={period_label})...")
        t0 = time.time()
        df_full, all_cols = build_all_features(df_raw, asset_name=asset_name, horizon=horizon)
        n_pysr = _compute_pysr_features(df_full, all_cols, asset_name, horizon)
        # Early leakage check: if PySR features loaded but not from clean historical window, strip them
        if n_pysr > 0:
            pysr_cols_loaded = [c for c in all_cols if c.startswith('pysr_')]
            is_clean, leak_msg = _check_pysr_leakage(pysr_cols_loaded, asset_name, horizon)
            if not is_clean:
                print(f"\n  *** LEAKAGE DETECTED (early check): {leak_msg}")
                print(f"  *** Stripping {len(pysr_cols_loaded)} PySR features from this run")
                print(f"  *** Fix: run Mode P first to generate clean PySR")
                for pc in pysr_cols_loaded:
                    all_cols.remove(pc)
                    if pc in df_full.columns:
                        df_full.drop(columns=[pc], inplace=True)
        print(f"  [Feature build: {(time.time()-t0)/60:.1f} min]")

        total_rows = len(df_full)
        if total_rows > MAX_DIAG_HOURS:
            df_full = df_full.tail(MAX_DIAG_HOURS).reset_index(drop=True)
            print(f"  Capped: {total_rows:,} -> {len(df_full):,} rows ({period_label})")

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
        grid_csv_path = os.path.join('models', f'crypto_ed_grid_{asset_name}_{horizon}h.csv')
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
            qty = cash * (1 - BACKTEST_FEE_PER_LEG) / price
            entry_price = price
            cash = 0
            position = 'invested'
            trades += 1
        elif sig['signal'] == 'SELL' and position == 'invested':
            cash = qty * price * (1 - BACKTEST_FEE_PER_LEG)
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


def run_mode_v(assets_list, horizons=None, replay_hours=None):
    """
    Mode V: Live backtest + Optuna refine + final comparison.
    1. Backtest top 6 from Mode D at conf 70/80/90%
    2. Pick top 3 live performers
    3. Optuna refine those 3 (gamma±0.020, features±5, window±20h, 30 trials)
    4. Backtest the 3 refined configs at conf 70/80/90%
    5. Show combined summary: D candidates vs refined
    6. Save best overall to production CSV

    replay_hours: override MODE_G_REPLAY_HOURS (default 1440h = 2 months).
    """
    if horizons is None:
        horizons = list(AVAILABLE_HORIZONS)
    _replay = replay_hours or MODE_G_REPLAY_HOURS

    candidates_csv = _get_models_csv_path()
    if not os.path.exists(candidates_csv):
        print(f"  ERROR: {candidates_csv} not found. Run Mode D first.")
        return

    df_candidates = pd.read_csv(candidates_csv)

    print("=" * 80)
    print(f"  MODE V: LIVE BACKTEST + REFINE — {','.join(assets_list)} {','.join(str(h)+'h' for h in horizons)}")
    print(f"  Period: last {_replay} hours ({_replay/168:.1f} weeks), every hour")
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
            n_pysr = _compute_pysr_features(df_full, all_cols, asset, horizon)
            # Early leakage check
            if n_pysr > 0:
                pysr_cols_loaded = [c for c in all_cols if c.startswith('pysr_')]
                is_clean, leak_msg = _check_pysr_leakage(pysr_cols_loaded, asset, horizon)
                if not is_clean:
                    print(f"\n  *** LEAKAGE DETECTED (early check): {leak_msg}")
                    print(f"  *** Stripping {len(pysr_cols_loaded)} PySR features from this run")
                    for pc in pysr_cols_loaded:
                        all_cols.remove(pc)
                        if pc in df_full.columns:
                            df_full.drop(columns=[pc], inplace=True)
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
                results[label] = _backtest_one_config(asset, horizon, label, cfg, replay_hours=_replay)

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
                        results[label] = _backtest_one_config(asset, horizon, label, cfg, replay_hours=_replay)

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
                        # Override with validated backtest results
                        prod_row['return_pct'] = round(best_sim['return_pct'], 2)
                        prod_row['accuracy'] = round(best_sim['win_rate'], 1)
                        prod_row['combined_score'] = round(best_score, 4)
                        prod_row['sampler'] = 'Grid'
                        if 'rank' in prod_row:
                            del prod_row['rank']
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
                    # Check for PySR leakage before allowing production write
                    features_to_check = best_cfg.get('features', [])
                    if isinstance(features_to_check, str):
                        features_to_check = features_to_check.split(',')
                    is_clean, leak_msg = _check_pysr_leakage(features_to_check, asset, horizon)
                    if not is_clean:
                        print(f"\n  *** LEAKAGE BLOCKED: {leak_msg}")
                        print(f"  *** Skipping production write for {asset} {horizon}h")
                        print(f"  *** Fix: run Mode P first, then re-run Mode DV")
                    else:
                        if any(f.startswith('pysr_') for f in features_to_check):
                            print(f"  PySR leakage check: {leak_msg}")
                        production_models.append((prod_row, horizon, best_conf))

    # ── Combined Summary ──
    print(f"\n\n{'=' * 80}")
    print(f"  SUMMARY: MODE V — D + Refined — Last {_replay//24} days (scored by return × win_rate)")
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
        tcfg_path = REGIME_CONFIG_PATH
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


def _backtest_one_config(asset, horizon, label, cfg, replay_hours=None):
    """Backtest a single config at all confidence thresholds. Returns results dict."""
    models = cfg['combo'].split('+')
    features = cfg['features']
    window = cfg['window']
    gamma = cfg['gamma']
    _replay = replay_hours or MODE_G_REPLAY_HOURS

    print(f"\n{'─' * 70}")
    print(f"  {label}: {asset}")
    print(f"  {cfg['combo']}  w={window}h  g={gamma:.4f}  f={len(features)}")
    print(f"{'─' * 70}")

    signals = generate_signals(
        asset_name=asset,
        model_names=models,
        window_size=window,
        replay_hours=_replay,
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
    df_full_r, all_cols_r = build_all_features(df_raw, asset_name=asset, horizon=horizon)
    n_pysr = _compute_pysr_features(df_full_r, all_cols_r, asset, horizon, verbose=False)
    if n_pysr > 0:
        pysr_cols_r = [c for c in all_cols_r if c.startswith('pysr_')]
        is_clean, _ = _check_pysr_leakage(pysr_cols_r, asset, horizon)
        if not is_clean:
            for pc in pysr_cols_r:
                all_cols_r.remove(pc)
                if pc in df_full_r.columns:
                    df_full_r.drop(columns=[pc], inplace=True)
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
            study_name=f'ed_v1_refine_{asset}_{horizon}h_{cfg_idx}',
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
# MODE S: REGIME CONFIDENCE OPTIMIZATION
# ============================================================
def run_mode_s(assets_list, horizons, args=None):
    """
    Ed Mode S — FULL JOINT SWEEP (V3 logic in production).

    Jointly sweeps detector × bull_h × bear_h × bull_conf × bear_conf.
    Discovers the global optimum across the entire search space instead of
    relying on Mode R's greedy sequential horizon selection.

    When called after Mode R (RS/HRS/DVRS), Mode R's horizon picks are
    informational only — Mode S re-discovers horizons as part of its sweep.

    Flow:
        1. Determine available horizons from production models (or CLI)
        2. Generate signals for ALL unique horizons
        3. Build regime indicators + detector dict (shared helper)
        4. For each detector × horizon_pair × bull_conf × bear_conf, simulate
        5. Pick best by return × win_rate scoring
        6. Write winning detector + horizons + confidences to regime_config_ed.json
    """
    replay = int(getattr(args, 'replay', 0)) or 1440
    top_n = int(getattr(args, 'top', 0)) or 15

    # Load regime config
    tcfg_path = REGIME_CONFIG_PATH
    try:
        with open(tcfg_path) as f:
            regime_config = json.load(f)
    except Exception as e:
        print(f"  ERROR: Cannot read {tcfg_path}: {e}")
        return

    CONF_LEVELS = [65, 70, 75, 80, 85, 90, 95]

    df_models = pd.read_csv(PRODUCTION_CSV)

    for asset in assets_list:
        # Determine horizons: use CLI horizons if provided, else from production models
        if horizons:
            available_h = sorted([h for h in horizons
                                  if len(df_models[(df_models['coin'] == asset) & (df_models['horizon'] == h)]) > 0])
        else:
            available_h = sorted(df_models[df_models['coin'] == asset]['horizon'].unique())

        if len(available_h) < 1:
            print(f"\n  {asset}: no production models found — skipping")
            continue

        # Build all horizon pairs (including same-horizon pairs)
        horizon_pairs = [(b, r) for b in available_h for r in available_h]
        n_det = 5  # sma24>sma100, sma168>sma480, price>sma72, vol_calm, tsmom_672h
        n_combos = n_det * len(horizon_pairs) * len(CONF_LEVELS) * len(CONF_LEVELS)

        print(f"\n{'='*80}")
        print(f"  MODE S: JOINT SWEEP (V3)")
        print(f"  Asset: {asset} | Replay: {replay}h | Horizons: {available_h}")
        print(f"  {n_det} detectors × {len(horizon_pairs)} h-pairs × "
              f"{len(CONF_LEVELS)}×{len(CONF_LEVELS)} conf = {n_combos:,} combos")
        print(f"{'='*80}")

        # ── Generate signals for ALL unique horizons ──
        signals_cache = {}
        sig_error = False
        for h in available_h:
            rows = df_models[(df_models['coin'] == asset) & (df_models['horizon'] == h)]
            if len(rows) == 0:
                print(f"  ERROR: No production model for {asset} {h}h")
                sig_error = True
                break
            row = rows.sort_values('combined_score', ascending=False).iloc[0]
            feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
            gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0

            print(f"\n  Generating {h}h signals ({row['models']} w={int(row['best_window'])}h)...")
            with _suppress_stderr():
                sigs = generate_signals(asset, row['models'].split('+'),
                                        int(row['best_window']), replay,
                                        feature_override=feats, horizon=h, gamma=gamma)
            result = {}
            for s in sigs:
                dt = s['datetime']
                if isinstance(dt, str):
                    dt = _dt_log.strptime(dt, '%Y-%m-%d %H:%M')
                    s['datetime'] = dt
                result[dt] = s
            signals_cache[h] = result
            print(f"    {h}h: {len(result)} signals ({sum(1 for s in result.values() if s['signal']=='BUY')} BUY)")

        if sig_error or len(signals_cache) < len(available_h):
            print(f"  Skipping {asset} — missing signals")
            continue

        all_dts = sorted(set().union(*[set(s.keys()) for s in signals_cache.values()]))

        # ── Build regime indicators + detectors (shared helper) ──
        print(f"\n  Building regime indicators...")
        ind, detectors = _build_regime_indicators_and_detectors(asset)

        print(f"\n  Sweeping {n_combos:,} combos...")

        # ── Joint sweep: detector × horizon_pair × bull_conf × bear_conf ──
        def _sim(det_fn, bull_h, bear_h, bull_conf, bear_conf):
            cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
            trades, wins = 0, 0
            bull_trades, bear_trades = 0, 0
            first_price, last_price = None, None

            for dt in all_dts:
                is_bull = det_fn(dt)
                h = bull_h if is_bull else bear_h
                conf = bull_conf if is_bull else bear_conf

                sigs = signals_cache.get(h)
                if sigs is None:
                    continue
                s = sigs.get(dt)
                if s is None:
                    for oh in signals_cache:
                        os_ = signals_cache[oh].get(dt)
                        if os_:
                            last_price = os_['close']
                            if first_price is None:
                                first_price = last_price
                            break
                    continue

                price = s['close']
                last_price = price
                if first_price is None:
                    first_price = price

                if s['signal'] == 'BUY' and s['confidence'] >= conf and not in_pos:
                    held = cash * (1 - BACKTEST_FEE_PER_LEG) / price
                    cash = 0
                    in_pos = True
                    entry_px = price
                    trades += 1
                    if is_bull:
                        bull_trades += 1
                    else:
                        bear_trades += 1
                elif s['signal'] == 'SELL' and in_pos:
                    cash = held * price * (1 - BACKTEST_FEE_PER_LEG)
                    if price > entry_px:
                        wins += 1
                    held = 0
                    in_pos = False

            if in_pos and last_price:
                cash = held * last_price * (1 - BACKTEST_FEE_PER_LEG)
                if last_price > entry_px:
                    wins += 1

            ret = (cash / 1000.0 - 1) * 100
            wr = (wins / trades * 100) if trades > 0 else 0
            bh = ((last_price / first_price - 1) * 100) if first_price and last_price else 0
            return ret, trades, wr, bh, bull_trades, bear_trades

        results = []
        for det_name, det_fn in detectors.items():
            bull_count = sum(1 for dt in all_dts if det_fn(dt))
            bull_pct = bull_count / len(all_dts) * 100 if all_dts else 0
            for (bull_h, bear_h) in horizon_pairs:
                for bc in CONF_LEVELS:
                    for rc in CONF_LEVELS:
                        ret, tr, wr, bh, bt, rt = _sim(det_fn, bull_h, bear_h, bc, rc)
                        score = ret * (wr / 100) if ret > 0 else ret
                        results.append({
                            'detector': det_name,
                            'bull_h': bull_h, 'bear_h': bear_h,
                            'bull_conf': bc, 'bear_conf': rc,
                            'return': ret, 'trades': tr, 'wr': wr, 'bh': bh,
                            'alpha': ret - bh, 'score': score,
                            'bull_trades': bt, 'bear_trades': rt,
                            'bull_pct': bull_pct,
                        })

        results.sort(key=lambda x: x['score'], reverse=True)

        if not results:
            print(f"  ERROR: No valid results for {asset} — skipping config update")
            continue

        # ── Results ──
        print(f"\n  {'='*120}")
        print(f"  TOP {top_n} JOINT SWEEP — {asset}")
        print(f"  {'='*120}")
        print(f"  {'#':>3}  {'Detector':>16}  {'BullH':>5}  {'BearH':>5}  {'BullC':>6}  {'BearC':>6}  "
              f"{'Return':>8}  {'Trades':>7}  {'WR':>5}  {'Alpha':>7}  {'Score':>7}  {'Bull%':>6}")
        print(f"  {'-'*3}  {'-'*16}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*7}  "
              f"{'-'*5}  {'-'*7}  {'-'*7}  {'-'*6}")

        for i, r in enumerate(results[:top_n]):
            print(f"  {i+1:>3}  {r['detector']:>16}  {str(r['bull_h'])+'h':>5}  {str(r['bear_h'])+'h':>5}  "
                  f"{r['bull_conf']:>5}%  {r['bear_conf']:>5}%  "
                  f"{r['return']:>+7.2f}%  {r['trades']:>7}  {r['wr']:>4.0f}%  {r['alpha']:>+6.2f}%  "
                  f"{r['score']:>7.2f}  {r['bull_pct']:>5.0f}%")

        winner = results[0]
        print(f"\n  {'='*120}")
        print(f"  WINNER: detector={winner['detector']}  bull={winner['bull_h']}h@{winner['bull_conf']}%  "
              f"bear={winner['bear_h']}h@{winner['bear_conf']}%")
        print(f"  Return: {winner['return']:+.2f}%  Trades: {winner['trades']}  WR: {winner['wr']:.0f}%  "
              f"Alpha: {winner['alpha']:+.2f}%")
        print(f"  {'='*120}")

        # ── Write winning detector + horizons + confidences to config ──
        if asset not in regime_config:
            regime_config[asset] = {'enabled': True, 'symbol': f'{asset}-USD',
                                    'bull': {}, 'bear': {}, 'use_maker_orders': True}
        regime_config[asset]['regime_detector'] = {'type': 'named', 'params': {'name': winner['detector']}}
        regime_config[asset]['bull']['horizon'] = winner['bull_h']
        regime_config[asset]['bull']['min_confidence'] = winner['bull_conf']
        regime_config[asset]['bear']['horizon'] = winner['bear_h']
        regime_config[asset]['bear']['min_confidence'] = winner['bear_conf']

        _atomic_write_json(tcfg_path, regime_config)
        print(f"\n  Config updated: {asset} detector={winner['detector']} "
              f"bull={winner['bull_h']}h@{winner['bull_conf']}% / bear={winner['bear_h']}h@{winner['bear_conf']}%")

    print(f"\n{'='*80}")
    print(f"  MODE S COMPLETE")
    print(f"{'='*80}")


# ============================================================
# MODE T: THRESHOLD SWEEP (hold-until-profitable optimization)
# ============================================================
def run_mode_t(assets_list, args=None):
    """
    Mode T — Sweep min_sell_pnl_pct × max_hold_hours × shield on/off per regime.

    For each (threshold, failsafe) pair, tests all 4 shield on/off combos
    (bull ON/OFF × bear ON/OFF) and picks the combo that maximises total
    return across the bull and bear horizons.

    Uses `BACKTEST_FEE_PER_LEG` (5 bps/leg realistic maker blend) and reads the
    current regime config for production horizons + confidences.

    Saves winner to regime_config_ed.json:
      - min_sell_pnl_pct, max_hold_hours at asset level (shared thresholds)
      - bull.hold_shield, bear.hold_shield per regime
    """
    replay = int(getattr(args, 'replay', 0)) or 1440

    tcfg_path = REGIME_CONFIG_PATH
    try:
        with open(tcfg_path) as f:
            regime_config = json.load(f)
    except Exception as e:
        print(f"  ERROR: Cannot read {tcfg_path}: {e}")
        return

    THRESHOLDS = [round(0.30 + i * 0.05, 2) for i in range(7)]  # 0.30 to 0.60
    FAILSAFE_HOURS = [8, 10, 12]

    df_models = pd.read_csv(PRODUCTION_CSV)

    for asset in assets_list:
        asset_cfg = regime_config.get(asset, {})
        bull_h = asset_cfg.get('bull', {}).get('horizon')
        bear_h = asset_cfg.get('bear', {}).get('horizon')
        bull_conf = asset_cfg.get('bull', {}).get('min_confidence', 85)
        bear_conf = asset_cfg.get('bear', {}).get('min_confidence', 65)

        if not bull_h or not bear_h:
            print(f"\n  {asset}: no regime config (run Mode S first) — skipping")
            continue

        test_horizons = sorted(set([bull_h, bear_h]))

        print(f"\n{'='*80}")
        print(f"  MODE T: THRESHOLD SWEEP (hold-until-profitable)")
        print(f"  Asset: {asset} | Replay: {replay}h ({replay/720:.1f} months)")
        print(f"  Production: bull={bull_h}h@{bull_conf}% | bear={bear_h}h@{bear_conf}%")
        print(f"  Thresholds: {THRESHOLDS} | Failsafe: {FAILSAFE_HOURS}h")
        print(f"  Fee: {BACKTEST_FEE_PER_LEG*1e4:.1f} bps/leg (realistic maker blend)")
        print(f"{'='*80}")

        # Generate signals for production horizons only
        signals_cache = {}
        for h in test_horizons:
            rows = df_models[(df_models['coin'] == asset) & (df_models['horizon'] == h)]
            if len(rows) == 0:
                print(f"  ERROR: No production model for {asset} {h}h")
                break
            row = rows.sort_values('combined_score', ascending=False).iloc[0]
            feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
            gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0

            print(f"\n  Generating {h}h signals ({row['models']} w={int(row['best_window'])}h)...")
            with _suppress_stderr():
                sigs = generate_signals(asset, row['models'].split('+'),
                                        int(row['best_window']), replay,
                                        feature_override=feats, horizon=h, gamma=gamma)
            signals_cache[h] = sigs
            print(f"    {h}h: {len(sigs)} candles")

        if len(signals_cache) < len(test_horizons):
            print(f"  Skipping {asset} — missing signals")
            continue

        # Quick-release config for the shield (per-asset, optional). Defaults match
        # the trader's defaults so the sim mirrors live behavior.
        # Quick-release default OFF — opt-in only (see 2026-04-19 evaluation)
        qr_cfg = asset_cfg.get('shield_quick_release', {})
        QR_ENABLED = bool(qr_cfg.get('enabled', False))
        QR_MIN_CONF = float(qr_cfg.get('min_sell_conf', 95))
        QR_MAX_HOURS = float(qr_cfg.get('max_hours', 3))

        # Simulate for each horizon at its production confidence.
        # Optionally applies a rally-cooldown gate (rally_cfg), so the shield
        # sweep can be conditioned on the gate from a prior iteration — closes
        # the T↔G coupling for iterative convergence.
        def _sim_horizon(sigs, conf, min_pnl, max_hold_h, rally_cfg=None):
            cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
            trades, trade_log, blocked, quick_rel = 0, [], 0, 0
            hold_since_entry = 0
            cd = 0
            rs_arr = rl_arr = None
            t_s = t_l = 0.0
            cd_h = 0
            if rally_cfg is not None:
                h_s, h_l, t_s, t_l, cd_h = rally_cfg
                closes = np.array([float(s['close']) for s in sigs])
                def _rr(h):
                    out = np.full(len(closes), np.nan)
                    if h < len(closes):
                        out[h:] = (closes[h:] / closes[:-h] - 1.0) * 100.0
                    return out
                rs_arr = _rr(h_s); rl_arr = _rr(h_l)

            for i, s in enumerate(sigs):
                price = s['close']
                if in_pos:
                    hold_since_entry += 1
                # Rally-cooldown trigger check
                if rally_cfg is not None and cd_h > 0:
                    rs = rs_arr[i] if not np.isnan(rs_arr[i]) else 0
                    rl = rl_arr[i] if not np.isnan(rl_arr[i]) else 0
                    if rs >= t_s or rl >= t_l:
                        cd = max(cd, cd_h)

                if s['signal'] == 'BUY' and s['confidence'] >= conf and not in_pos:
                    if cd > 0:
                        pass  # gate blocks BUY
                    else:
                        held = cash * (1 - BACKTEST_FEE_PER_LEG) / price
                        cash = 0
                        in_pos = True
                        entry_px = price
                        trades += 1
                        hold_since_entry = 0
                elif s['signal'] == 'SELL' and in_pos:
                    cur_pnl = (price / entry_px - 1) * 100
                    override_expired = hold_since_entry >= max_hold_h
                    # Quick-release: strong SELL within N cycles of entry bypasses shield
                    quick_release = (QR_ENABLED and min_pnl > 0
                                     and hold_since_entry <= QR_MAX_HOURS
                                     and float(s.get('confidence', 0)) >= QR_MIN_CONF)
                    if (min_pnl <= 0 or cur_pnl >= min_pnl or override_expired
                            or quick_release):
                        cash = held * price * (1 - BACKTEST_FEE_PER_LEG)
                        trade_log.append(cur_pnl)
                        held = 0
                        in_pos = False
                        trades += 1
                        hold_since_entry = 0
                        if quick_release:
                            quick_rel += 1
                    else:
                        blocked += 1

                if cd > 0:
                    cd -= 1

            final = cash if not in_pos else held * sigs[-1]['close']
            if in_pos:
                trade_log.append((sigs[-1]['close'] / entry_px - 1) * 100)
            ret = (final / 1000.0 - 1) * 100
            winners = sum(1 for t in trade_log if t > 0)
            wr = (winners / len(trade_log) * 100) if trade_log else 0
            return ret, len(trade_log), wr, blocked

        # Run sweep per horizon
        horizon_results = {}
        for h in test_horizons:
            conf = bull_conf if h == bull_h else bear_conf
            sigs = signals_cache[h]

            base_ret, base_tr, base_wr, _ = _sim_horizon(sigs, conf, 0, 999)
            print(f"\n  {asset} {h}h (conf>={conf}%) — Baseline: {base_ret:+.2f}% | {base_tr} trades | WR {base_wr:.0f}%")

            print(f"\n  {'Threshold':<12} {'Failsafe':>8} {'Return':>9} {'vs Base':>8} {'Trades':>7} {'WinRate':>8} {'Blocked':>8}")
            print(f"  {'─'*12} {'─'*8} {'─'*9} {'─'*8} {'─'*7} {'─'*8} {'─'*8}")

            best_score = base_ret * (base_wr / 100) if base_ret > 0 else base_ret
            best = None

            for t in THRESHOLDS:
                for fh in FAILSAFE_HOURS:
                    ret, tr, wr, bl = _sim_horizon(sigs, conf, t, fh)
                    delta = ret - base_ret
                    score = ret * (wr / 100) if ret > 0 else ret
                    marker = ' ★' if score > best_score else (' ✓' if delta > 0 else '')
                    if score > best_score:
                        best_score = score
                        best = (t, fh, ret, tr, wr)
                    print(f"  {t:.2f}%        {fh:>7}h {ret:>+8.2f}% {delta:>+7.2f}% {tr:>7} {wr:>7.0f}% {bl:>8}{marker}")

            if best:
                horizon_results[h] = best
                print(f"\n  ★ {h}h BEST: threshold={best[0]:.2f}%, failsafe={best[1]}h → {best[2]:+.2f}%")
            else:
                print(f"\n  No improvement over baseline for {h}h")

        # ---- Iterative joint sweep: (threshold, failsafe, bull_on, bear_on) x G ----
        # Bull and bear share (t, fh); shield on/off is chosen independently per regime.
        # Iterates T <-> G until config is stable (closes the coupling where T's
        # shield sweep previously ignored the gate that G subsequently adds).
        if bull_h == bear_h:
            print(f"\n  NOTE: bull_h == bear_h == {bull_h}h — per-regime shield split has no effect.")

        bull_sigs = signals_cache[bull_h]
        bear_sigs = signals_cache[bear_h]

        max_iter = int(getattr(args, 'max_iter', 0)) or 4

        def _get_rally_tuple(cfg, regime=None):
            """Rally-cooldown tuple for a specific regime, with asset-level fallback."""
            def _tuple(rc):
                if not rc or not rc.get('enabled'):
                    return None
                try:
                    return (int(rc['h_short']), int(rc['h_long']),
                            float(rc['t_short_pct']), float(rc['t_long_pct']),
                            int(rc['cd_hours']))
                except (KeyError, TypeError, ValueError):
                    return None
            if regime is not None:
                block = cfg.get(regime)
                if isinstance(block, dict):
                    t = _tuple(block.get('rally_cooldown'))
                    if t is not None:
                        return t
            return _tuple(cfg.get('rally_cooldown'))

        def _config_fingerprint(cfg):
            # Per-regime gate lives in bull.rally_cooldown / bear.rally_cooldown
            # Legacy asset-level rally_cooldown kept as fallback in the fingerprint.
            asset_rc = cfg.get('rally_cooldown') or {}
            bull_rc = cfg.get('bull', {}).get('rally_cooldown') or {}
            bear_rc = cfg.get('bear', {}).get('rally_cooldown') or {}
            def _rc_tuple(rc):
                return (rc.get('h_short'), rc.get('h_long'),
                        rc.get('t_short_pct'), rc.get('t_long_pct'),
                        rc.get('cd_hours'))
            return (
                cfg.get('min_sell_pnl_pct'),
                cfg.get('max_hold_hours'),
                cfg.get('bull', {}).get('hold_shield'),
                cfg.get('bear', {}).get('hold_shield'),
                _rc_tuple(asset_rc),
                _rc_tuple(bull_rc),
                _rc_tuple(bear_rc),
            )

        prev_fp = None
        converged = False
        for iteration in range(1, max_iter + 1):
            print(f"\n{'#'*80}")
            print(f"  T<->G ITERATION {iteration}/{max_iter} | {asset}")
            print(f"{'#'*80}")

            bull_rally_cfg = _get_rally_tuple(regime_config[asset], regime='bull')
            bear_rally_cfg = _get_rally_tuple(regime_config[asset], regime='bear')
            def _desc(cfg):
                return (f"rr{cfg[0]}h>={cfg[2]}% OR rr{cfg[1]}h>={cfg[3]}%, cd={cfg[4]}h"
                        if cfg else "none")
            print(f"  Using per-regime gates: bull={_desc(bull_rally_cfg)} | "
                  f"bear={_desc(bear_rally_cfg)}")

            # All-OFF baseline (with current gate applied per regime)
            base_bull, _, _, _ = _sim_horizon(bull_sigs, bull_conf, 0, 999, bull_rally_cfg)
            base_bear, _, _, _ = _sim_horizon(bear_sigs, bear_conf, 0, 999, bear_rally_cfg)
            base_total = base_bull + base_bear

            best_total = None
            best_quad = None
            rows_print = []
            for t in THRESHOLDS:
                for fh in FAILSAFE_HOURS:
                    for bull_on in (False, True):
                        for bear_on in (False, True):
                            bull_t = t if bull_on else 0
                            bear_t = t if bear_on else 0
                            b_ret, _, _, _ = _sim_horizon(bull_sigs, bull_conf, bull_t, fh, bull_rally_cfg)
                            r_ret, _, _, _ = _sim_horizon(bear_sigs, bear_conf, bear_t, fh, bear_rally_cfg)
                            total = b_ret + r_ret
                            rows_print.append((t, fh, bull_on, bear_on, b_ret, r_ret, total))
                            if best_total is None or total > best_total:
                                best_total = total
                                best_quad = (t, fh, bull_on, bear_on)

            rows_print.sort(key=lambda r: -r[6])
            print(f"  Top 3 shield combos: "
                  f"{' / '.join(f'thr={r[0]:.2f} fh={r[1]}h bull={r[2]} bear={r[3]} tot={r[6]:+.2f}%' for r in rows_print[:3])}")
            print(f"  Baseline (all-OFF, gate applied): "
                  f"bull={base_bull:+.2f}% bear={base_bear:+.2f}% total={base_total:+.2f}%")

            t_win, fh_win, bull_on, bear_on = best_quad
            if best_total > base_total:
                regime_config[asset]['min_sell_pnl_pct'] = t_win
                regime_config[asset]['max_hold_hours'] = fh_win
                regime_config[asset].setdefault('bull', {})['hold_shield'] = bool(bull_on)
                regime_config[asset].setdefault('bear', {})['hold_shield'] = bool(bear_on)
                regime_config[asset].pop('hold_shield', None)
                print(f"  T winner: min_sell_pnl={t_win:.2f}% max_hold={fh_win}h "
                      f"bull_shield={'ON' if bull_on else 'OFF'} bear_shield={'ON' if bear_on else 'OFF'} "
                      f"(total {best_total:+.2f}% vs {base_total:+.2f}%, delta {best_total-base_total:+.2f}%)")
            else:
                regime_config[asset]['min_sell_pnl_pct'] = 0
                regime_config[asset]['max_hold_hours'] = 10
                regime_config[asset].setdefault('bull', {})['hold_shield'] = False
                regime_config[asset].setdefault('bear', {})['hold_shield'] = False
                regime_config[asset].pop('hold_shield', None)
                print(f"  T: no shield improvement — disabling for this iteration")
            _atomic_write_json(tcfg_path, regime_config)

            # Chain G with the new shield — per-regime: sweep bull and bear gates
            # independently. Each sweep fires the gate only on its regime's bars,
            # simulating the other regime as gate-less. Winners written to
            # cfg[asset].bull.rally_cooldown and cfg[asset].bear.rally_cooldown.
            tagged = _merge_tagged_signals(asset, bull_sigs, bear_sigs, regime_config[asset])
            print(f"  Per-regime G sweep on {len(tagged)} tagged bars "
                  f"(bull={sum(1 for s in tagged if s['regime']=='bull')} "
                  f"bear={sum(1 for s in tagged if s['regime']=='bear')})...")
            # Clear any legacy asset-level rally_cooldown — regime blocks are now authoritative
            regime_config[asset].pop('rally_cooldown', None)
            _atomic_write_json(tcfg_path, regime_config)
            for r_filter in ('bull', 'bear'):
                print(f"  --- {r_filter.upper()} gate sweep ---")
                _sweep_rally_cooldown(asset, tagged, regime_config[asset],
                                      replay_h=replay, rank='recent',
                                      write_config=True, regime_filter=r_filter)
                with open(tcfg_path) as f:
                    regime_config = json.load(f)

            # Reload config after G's write (G uses atomic write and we've kept
            # regime_config as the in-memory mirror via setdefault — re-read to be safe)
            with open(tcfg_path) as f:
                regime_config = json.load(f)

            # Convergence check
            fp = _config_fingerprint(regime_config[asset])
            if prev_fp is not None and fp == prev_fp:
                converged = True
                print(f"\n  >>> Converged at iteration {iteration} (config unchanged).")
                break
            prev_fp = fp

        if not converged:
            print(f"\n  >>> Reached max_iter={max_iter} without convergence. "
                  f"Final config written.")

    print(f"\n{'='*80}")
    print(f"  MODE T COMPLETE")
    print(f"{'='*80}")


# ============================================================
# MODE H: HORIZON SWEEP (D+G per horizon, compare, save best)
# ============================================================
def run_mode_h(assets_list, horizons, n_trials=None, resume=False, skip_d=False):
    """
    Mode H: Horizon sweep — runs D+G for each specified horizon, then compares
    across horizons to find the best one per asset.

    Usage:
        python crypto_trading_system_ed.py H BTC 4,5,6,7,8h          # full D+G per horizon
        python crypto_trading_system_ed.py H BTC,ETH 5,6,7,8h --skip # skip D where results exist

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

        # ── Cross-horizon comparison (informational — Mode R picks the winner) ──
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
            print(f"  {h:>2}h | {hr['label']:<25} | {cfg['combo']:14s} | {cfg['window']:>3}h | "
                  f"{cfg['gamma']:>.3f} | {cfg['n_features']:>3} | {hr['conf']:>3}% | "
                  f"{hr['return_pct']:>+7.2f}% | {hr['trades']:>3} | {hr['win_rate']:>3.0f}% | "
                  f"{hr['buy_hold']:>+6.2f}%")

        best_h = max(horizon_results.keys(), key=lambda h: horizon_results[h]['score'])
        winner = horizon_results[best_h]
        print(f"\n  Best single horizon: {best_h}h — {winner['return_pct']:+.2f}% "
              f"(conf>={winner['conf']}%, {winner['trades']} trades)")
        print(f"  Note: Run Mode R to find best bull/bear horizon pair, then Mode S for confidence.")

    elapsed = (time.time() - t_total) / 60
    print(f"\n  Mode H complete: {elapsed:.1f} min total")
    print(f"{'=' * 80}")


# ============================================================
# MODE R: REGIME BACKTEST
# ============================================================

def _build_regime_indicators_and_detectors(asset):
    """Shared helper used by Mode R and Mode S.

    Builds the regime indicator dataframe (SMAs, deseasonalized vol per
    Andersen-Bollerslev, TSMOM per Liu & Tsyvinski) and the detector dict.
    Single source of truth — keeps R and S in sync."""
    df_raw = load_data(asset)
    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
    df_ind = df_raw.set_index('datetime').sort_index()

    for w in [24, 48, 72, 100, 168, 200, 240, 480]:
        df_ind[f'sma{w}'] = df_ind['close'].rolling(w).mean()

    # Deseasonalized realized volatility (Andersen-Bollerslev 1997/1998)
    df_ind['logret_1h'] = np.log(df_ind['close'] / df_ind['close'].shift(1))
    df_ind['abs_logret'] = df_ind['logret_1h'].abs()
    df_ind['hour'] = df_ind.index.hour
    df_ind['seasonal_factor'] = (
        df_ind.groupby('hour')['abs_logret']
              .transform(lambda s: s.rolling(30, min_periods=10).mean())
    )
    df_ind['abs_logret_deseason'] = df_ind['abs_logret'] / df_ind['seasonal_factor'].replace(0, np.nan)
    df_ind['vol_24h_deseason'] = df_ind['abs_logret_deseason'].rolling(24).std()
    df_ind['vol_24h_deseason_q70'] = df_ind['vol_24h_deseason'].rolling(720, min_periods=240).quantile(0.70)

    # Time-Series Momentum (Liu & Tsyvinski 2021 RFS crypto replication)
    df_ind['tsmom_672h'] = np.log(df_ind['close'] / df_ind['close'].shift(672))

    ind = df_ind.to_dict('index')

    def safe(dt, fn, default=True):
        if dt not in ind:
            return default
        try:
            return fn(ind[dt])
        except (KeyError, TypeError):
            return default

    detectors = {
        'sma24>sma100':   lambda dt: safe(dt, lambda r: r['sma24'] > r['sma100']),
        'sma168>sma480':  lambda dt: safe(dt, lambda r: r['sma168'] > r['sma480']),
        'price>sma72':    lambda dt: safe(dt, lambda r: r['close'] > r['sma72']),
        'vol_calm':       lambda dt: safe(dt, lambda r: r['vol_24h_deseason'] < r['vol_24h_deseason_q70']),
        'tsmom_672h':     lambda dt: safe(dt, lambda r: r['tsmom_672h'] > 0),
    }
    return ind, detectors


def _run_mode_r(assets, horizons, args):
    """Regime-switching backtest: test bull/bear horizon combos with regime detectors."""
    import json as _json
    from itertools import permutations as _perms

    asset = assets[0] if assets else 'BTC'
    replay = int(getattr(args, 'replay', 0)) or 1440  # default 2 months
    conf = int(getattr(args, 'conf', 0)) or 90
    top_n = int(getattr(args, 'top', 0)) or 200

    # Determine available horizons from production CSV
    df_models = pd.read_csv(PRODUCTION_CSV)
    available_h = sorted(df_models[df_models['coin'] == asset]['horizon'].unique())
    test_horizons = horizons if horizons else [h for h in [4,5,6,7,8,10,12] if h in available_h]

    print(f"\n{'='*80}")
    print(f"  MODE R: REGIME BACKTEST")
    print(f"  Asset: {asset} | Replay: {replay}h ({replay//720}mo) | Conf: >={conf}%")
    print(f"  Horizons: {test_horizons}")
    print(f"{'='*80}")

    # ── Step 1: Generate signals for all horizons ──
    print(f"\n  Step 1: Generating signals for {len(test_horizons)} horizons...")
    signals_cache = {}  # {horizon: {datetime: signal_dict}}

    for h in test_horizons:
        rows = df_models[(df_models['coin'] == asset) & (df_models['horizon'] == h)]
        if len(rows) == 0:
            print(f"    {h}h: no production model, skipping")
            continue
        row = rows.sort_values('combined_score', ascending=False).iloc[0]
        feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
        gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0

        print(f"\n    Generating {h}h signals ({row['models']} w={int(row['best_window'])}h)...")
        with _suppress_stderr():
            sigs = generate_signals(asset, row['models'].split('+'),
                                    int(row['best_window']), replay,
                                    feature_override=feats, horizon=h, gamma=gamma)
        result = {}
        for s in sigs:
            dt = s['datetime']
            if isinstance(dt, str):
                dt = _dt_log.strptime(dt, '%Y-%m-%d %H:%M')
                s['datetime'] = dt
            result[dt] = s
        signals_cache[h] = result
        print(f"    {h}h: {len(result)} signals ({sum(1 for s in result.values() if s['signal']=='BUY')} BUY)")

    if len(signals_cache) < 2:
        print("\n  ERROR: Need at least 2 horizons with production models")
        return

    all_dts = sorted(set().union(*[set(s.keys()) for s in signals_cache.values()]))
    print(f"\n  Total: {len(all_dts)} timestamps, {len(signals_cache)} horizons")

    # ── Step 2: Build regime indicators ──
    print(f"\n  Step 2: Building regime indicators...")
    ind, detectors = _build_regime_indicators_and_detectors(asset)

    # Also load PySR regime if available
    for bull_h in test_horizons:
        for bear_h in test_horizons:
            if bull_h >= bear_h:
                continue
            pysr_path = os.path.join(os.path.dirname(PRODUCTION_CSV),
                                     f'pysr_regime_{asset}_{bull_h}h_{bear_h}h.json')
            if os.path.exists(pysr_path):
                with open(pysr_path) as _f:
                    pysr_data = _json.load(_f)
                if pysr_data.get('expressions'):
                    expr = pysr_data['expressions'][0]
                    feat_names = pysr_data.get('feature_names', [])
                    det_name = f'pysr_{bull_h}h_{bear_h}h'
                    # Build a simple evaluator (won't add this to detectors for now — too complex)
                    print(f"    Found PySR regime: {det_name} ({expr.get('equation', '?')[:60]})")

    print(f"    {len(detectors)} regime detectors loaded")

    # ── Step 3: Simulate ──
    def _sim(horizon_picker):
        cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
        trades, wins = 0, 0
        first_price, last_price = None, None
        for dt in all_dts:
            h = horizon_picker(dt)
            sigs = signals_cache.get(h)
            if sigs is None:
                continue
            s = sigs.get(dt)
            if s is None:
                for oh in signals_cache:
                    os_ = signals_cache[oh].get(dt)
                    if os_:
                        last_price = os_['close']
                        if first_price is None:
                            first_price = last_price
                        break
                continue
            price = s['close']
            last_price = price
            if first_price is None:
                first_price = price
            if s['signal'] == 'BUY' and s['confidence'] >= conf and not in_pos:
                held = cash * (1 - BACKTEST_FEE_PER_LEG) / price
                cash = 0
                in_pos = True
                entry_px = price
                trades += 1
            elif s['signal'] == 'SELL' and in_pos:
                cash = held * price * (1 - BACKTEST_FEE_PER_LEG)
                if price > entry_px:
                    wins += 1
                held = 0
                in_pos = False
        if in_pos and last_price:
            cash = held * last_price * (1 - BACKTEST_FEE_PER_LEG)
            if last_price > entry_px:
                wins += 1
        ret = (cash / 1000.0 - 1) * 100
        wr = (wins / trades * 100) if trades > 0 else 0
        bh = ((last_price / first_price - 1) * 100) if first_price and last_price else 0
        return ret, trades, wr, bh

    print(f"\n  Step 3: Running simulations...")

    # Single-horizon baselines
    print(f"\n  {'='*100}")
    print(f"  SINGLE-HORIZON BASELINES (conf>={conf}%)")
    print(f"  {'='*100}")
    print(f"  {'Horizon':>8s}  {'Return':>8s}  {'Trades':>7s}  {'WR':>5s}  {'B&H':>7s}  {'Alpha':>7s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*7}")
    baseline_results = {}
    for h in sorted(signals_cache.keys()):
        ret, tr, wr, bh = _sim(lambda dt, _h=h: _h)
        alpha = ret - bh
        baseline_results[h] = (ret, tr, wr, bh, alpha)
        print(f"  {str(h)+'h':>8s}  {ret:>+7.2f}%  {tr:>7d}  {wr:>4.0f}%  {bh:>+6.2f}%  {alpha:>+6.2f}%")

    # Regime-switching combos
    results = []
    horizon_pairs = [(b, r) for b in sorted(signals_cache.keys())
                     for r in sorted(signals_cache.keys()) if b != r]

    for det_name, det_fn in detectors.items():
        for bull_h, bear_h in horizon_pairs:
            picker = lambda dt, _d=det_fn, _b=bull_h, _r=bear_h: _b if _d(dt) else _r
            ret, tr, wr, bh = _sim(picker)
            alpha = ret - bh

            # Count regime splits
            bull_count = sum(1 for dt in all_dts if det_fn(dt))
            bear_count = len(all_dts) - bull_count
            bull_pct = bull_count / len(all_dts) * 100

            results.append({
                'detector': det_name,
                'bull_h': bull_h,
                'bear_h': bear_h,
                'return': ret,
                'trades': tr,
                'wr': wr,
                'bh': bh,
                'alpha': alpha,
                'bull_pct': bull_pct,
            })

    # Sort by return
    results.sort(key=lambda x: x['return'], reverse=True)

    print(f"\n  {'='*100}")
    print(f"  TOP {top_n} REGIME-SWITCHING STRATEGIES")
    print(f"  {'='*100}")
    print(f"  {'#':>3s}  {'Detector':>22s}  {'Bull':>5s}  {'Bear':>5s}  {'Return':>8s}  {'Trades':>7s}  {'WR':>5s}  {'B&H':>7s}  {'Alpha':>7s}  {'Bull%':>6s}")
    print(f"  {'-'*3}  {'-'*22}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*6}")

    for i, r in enumerate(results[:top_n]):
        print(f"  {i+1:>3d}  {r['detector']:>22s}  {str(r['bull_h'])+'h':>5s}  {str(r['bear_h'])+'h':>5s}  "
              f"{r['return']:>+7.2f}%  {r['trades']:>7d}  {r['wr']:>4.0f}%  {r['bh']:>+6.2f}%  "
              f"{r['alpha']:>+6.2f}%  {r['bull_pct']:>5.0f}%")

    # Best vs best baseline
    best_baseline = max(baseline_results.items(), key=lambda x: x[1][0])
    best_regime = results[0] if results else None
    print(f"\n  {'='*100}")
    print(f"  VERDICT")
    print(f"  {'='*100}")
    print(f"  Best baseline: {best_baseline[0]}h = {best_baseline[1][0]:+.2f}% ({best_baseline[1][1]} trades, WR {best_baseline[1][2]:.0f}%)")
    if best_regime:
        print(f"  Best regime:   {best_regime['detector']} bull={best_regime['bull_h']}h bear={best_regime['bear_h']}h "
              f"= {best_regime['return']:+.2f}% ({best_regime['trades']} trades, WR {best_regime['wr']:.0f}%)")
        diff = best_regime['return'] - best_baseline[1][0]
        if diff > 0:
            print(f"  Regime switching WINS by {diff:+.2f}%")
        else:
            print(f"  Baseline WINS by {-diff:+.2f}%")
    print(f"  {'='*100}")

    # Return best regime for HRS pipeline (R → S)
    return {asset: best_regime} if best_regime else {}


def _apply_mode_r_to_config(r_results):
    """Write Mode R's winning horizons to regime_config_ed.json so Mode S picks them up."""
    if not r_results:
        return
    tcfg_path = REGIME_CONFIG_PATH
    try:
        with open(tcfg_path) as f:
            regime_config = json.load(f)
    except Exception:
        return

    for asset, best in r_results.items():
        if not best:
            continue
        bull_h = best['bull_h']
        bear_h = best['bear_h']
        if asset not in regime_config:
            continue
        old_bull = regime_config[asset].get('bull', {}).get('horizon')
        old_bear = regime_config[asset].get('bear', {}).get('horizon')
        if old_bull != bull_h or old_bear != bear_h:
            regime_config[asset]['bull']['horizon'] = bull_h
            regime_config[asset]['bear']['horizon'] = bear_h
            print(f"\n  Mode R → Config: {asset} horizons updated: bull={old_bull}h→{bull_h}h, bear={old_bear}h→{bear_h}h")

    _atomic_write_json(tcfg_path, regime_config)


# ============================================================
# MAIN MENU
# ============================================================

# ============================================================
# RALLY-COOLDOWN SWEEP — shared helper used by Mode G and Mode T
# ============================================================

def _build_detector_from_cfg(asset, asset_cfg):
    """Return a detector callable(dt) → bool from asset_cfg. bull=True.
    Also returns the shared indicator dict for convenience."""
    ind, detectors = _build_regime_indicators_and_detectors(asset)
    det_cfg = asset_cfg.get('regime_detector', {})
    det_type = det_cfg.get('type', 'named')
    params = det_cfg.get('params', {})
    if det_type == 'named':
        name = params.get('name', 'tsmom_672h')
        det = detectors.get(name)
        if det is None:
            raise ValueError(f"Unknown named detector: {name}")
        return det
    elif det_type == 'sma_cross':
        fast = int(params.get('fast', 24))
        slow = int(params.get('slow', 100))
        def _sma_det(dt, _ind=ind, _f=fast, _s=slow):
            if dt not in _ind:
                return True
            try:
                return _ind[dt][f'sma{_f}'] > _ind[dt][f'sma{_s}']
            except (KeyError, TypeError):
                return True
        return _sma_det
    raise ValueError(f"Unknown detector type: {det_type}")


def _merge_tagged_signals(asset, bull_sig_list, bear_sig_list, asset_cfg):
    """Merge bull+bear signal streams into one tagged list.
    Each bar tagged with regime ('bull'|'bear') via the configured detector.

    Important: detector lookup keys are NAIVE pandas Timestamps (matching how
    _build_regime_indicators_and_detectors keys its `ind` dict). If we used
    tz-aware keys every lookup would miss and default to bull.
    """
    detector = _build_detector_from_cfg(asset, asset_cfg)

    def _to_naive(s_dt):
        if isinstance(s_dt, str):
            s_dt = datetime.strptime(s_dt, '%Y-%m-%d %H:%M')
        ts = pd.Timestamp(s_dt)
        if ts.tzinfo is not None:
            ts = ts.tz_convert('UTC').tz_localize(None)
        return ts

    bull_map = {_to_naive(s['datetime']): s for s in bull_sig_list}
    bear_map = {_to_naive(s['datetime']): s for s in bear_sig_list}
    all_dts = sorted(set(bull_map.keys()) | set(bear_map.keys()))

    bull_h = int(asset_cfg.get('bull', {}).get('horizon', 6))
    bear_h = int(asset_cfg.get('bear', {}).get('horizon', 8))
    bull_conf = float(asset_cfg.get('bull', {}).get('min_confidence', 85))
    bear_conf = float(asset_cfg.get('bear', {}).get('min_confidence', 65))

    merged = []
    for dt in all_dts:
        is_bull = bool(detector(dt))
        regime = 'bull' if is_bull else 'bear'
        src = bull_map.get(dt) if is_bull else bear_map.get(dt)
        fallback = bear_map.get(dt) if is_bull else bull_map.get(dt)
        chosen = src if src is not None else fallback
        if chosen is None:
            continue
        merged.append({
            'datetime': dt, 'close': float(chosen['close']),
            'signal': chosen['signal'] if src is not None else 'HOLD',
            'confidence': float(chosen['confidence']) if src is not None else 0.0,
            'conf_threshold': bull_conf if is_bull else bear_conf,
            'regime': regime, 'horizon': bull_h if is_bull else bear_h,
        })
    return merged


def _sweep_rally_cooldown(asset, signals, asset_cfg, replay_h, rank='recent',
                          write_config=True, regime_filter='all'):
    """Core rally-cooldown gate sweep. Returns winner dict (or None if no STRICT).

    Shared by Mode G (standalone, cache-fed) and Mode T (chain, fresh-signal-fed).
    Writes winner to regime_config_ed.json if write_config=True:
      - regime_filter='all': writes to asset_cfg['rally_cooldown'] (legacy, single gate)
      - regime_filter='bull' or 'bear': writes to asset_cfg[regime_filter]['rally_cooldown']
        AND the gate only fires on that regime's bars during the sim.
    """
    if replay_h < 720:
        print(f"  Rally-cooldown sweep: --replay must be >= 720h (30d). Got {replay_h}")
        return None
    days = replay_h / 24.0
    half_days = days / 2.0
    rank_col = 'score_recent' if rank == 'recent' else 'score_dd_aware'

    HORIZONS = [8, 10, 12, 14, 16, 18, 20, 24, 30, 36]
    def thr_for(h):
        if h in (8, 10, 12):       return [round(2.0 + 0.5*i, 2) for i in range(9)]
        if h in (14, 16, 18, 20):  return [round(3.0 + 0.5*i, 2) for i in range(9)]
        if h in (24, 30, 36):      return [round(4.0 + 0.5*i, 2) for i in range(11)]
        raise ValueError(f"no thr grid for {h}")
    CD_GRID = [6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 48]
    FEE = BACKTEST_FEE_PER_LEG
    PLATEAU_THR = 0.7

    print("=" * 80)
    print(f"  RALLY-COOLDOWN SWEEP | {asset} | window={days:.0f}d "
          f"(halves {half_days:.0f}+{half_days:.0f}d) | rank={rank}")
    print("=" * 80)

    # Log policy
    _bull_sh = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    _bear_sh = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    _bull_c = asset_cfg.get('bull', {}).get('min_confidence', '?')
    _bear_c = asset_cfg.get('bear', {}).get('min_confidence', '?')
    _msp = asset_cfg.get('min_sell_pnl_pct', '?')
    _mh = asset_cfg.get('max_hold_hours', '?')
    print(f"  {asset} | policy: bull@{_bull_c}% shield={_bull_sh} | "
          f"bear@{_bear_c}% shield={_bear_sh} | shield_thr={_msp}% / {_mh}h")
    _tagged = sum(1 for s in signals if 'regime' in s)
    if _tagged < len(signals):
        print(f"  {asset} | WARNING: {len(signals)-_tagged}/{len(signals)} signals "
              f"missing 'regime' tag — will default to 'bull'.")

    def simulate(sigs, rr_dict, h_s, h_l, t_s, t_l, cd_h, asset_cfg):
        """Simulate rally-cooldown gate against a signal stream.

        Reads per-regime policy from asset_cfg (set by Mode T):
          - bull/bear.hold_shield: whether to defer SELL on unrealized loss
          - bull/bear.min_confidence: BUY confidence threshold for that regime
          - min_sell_pnl_pct: shared shield profit threshold (percent)
          - max_hold_hours: shared shield failsafe
          - shield_quick_release: {enabled, min_sell_conf, max_hours}
            Bypass shield on strong SELL signals shortly after entry.

        Each signal bar carries s['regime'] = 'bull' | 'bear' (tagged at cache
        build time). Policy is looked up at the current bar's regime."""
        bull_shield = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
        bear_shield = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
        bull_conf = float(asset_cfg.get('bull', {}).get('min_confidence', 75.0))
        bear_conf = float(asset_cfg.get('bear', {}).get('min_confidence', 65.0))
        min_sell_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.5))  # already percent
        max_hold = int(asset_cfg.get('max_hold_hours', 10))
        # Quick-release default OFF — opt-in only (see 2026-04-19 evaluation)
        qr_cfg = asset_cfg.get('shield_quick_release', {})
        qr_enabled = bool(qr_cfg.get('enabled', False))
        qr_min_conf = float(qr_cfg.get('min_sell_conf', 95))
        qr_max_hours = float(qr_cfg.get('max_hours', 3))

        cash = 1000.0; qty = 0.0; in_pos = False; entry = 0.0
        hold = 0; trades = 0; skipped = 0; cd = 0
        ec = [1000.0]
        n = len(sigs)
        rs_arr = rr_dict.get(h_s); rl_arr = rr_dict.get(h_l)
        for i in range(n):
            s = sigs[i]; price = s['close']
            regime = s.get('regime', 'bull')
            conf_thr = bull_conf if regime == 'bull' else bear_conf
            # Trigger check: only when this bar's regime matches the filter
            # (or filter='all' = every bar). Enables per-regime gate sweeps.
            if cd_h > 0 and rs_arr is not None and rl_arr is not None:
                if regime_filter == 'all' or regime_filter == regime:
                    rs = rs_arr[i]; rl = rl_arr[i]
                    if (rs == rs and rs >= t_s) or (rl == rl and rl >= t_l):
                        cd = max(cd, cd_h)
            ec.append(cash + qty * price if in_pos else cash)
            if s['signal'] == 'BUY' and s['confidence'] >= conf_thr and not in_pos:
                if cd > 0:
                    skipped += 1
                else:
                    fp = sigs[i+1]['close'] if i+1 < n else price
                    qty = cash * (1 - FEE) / fp
                    cash = 0.0; in_pos = True; entry = fp; hold = 0
            elif s['signal'] == 'SELL' and in_pos:
                fp = sigs[i+1]['close'] if i+1 < n else price
                cur = (fp / entry - 1.0) * 100.0
                # Shield uses the CURRENT bar's regime — matches live trader,
                # which re-evaluates regime on every tick.
                shield_on = bull_shield if regime == 'bull' else bear_shield
                shield_min = min_sell_pnl if shield_on else 0.0
                # Quick-release: strong SELL within N cycles of entry bypasses shield
                quick_release = (qr_enabled and shield_on
                                 and hold <= qr_max_hours
                                 and float(s.get('confidence', 0)) >= qr_min_conf)
                if cur >= shield_min or hold >= max_hold or quick_release:
                    cash = qty * fp * (1 - FEE)
                    trades += 1; in_pos = False; qty = 0.0; entry = 0.0; hold = 0
            if in_pos: hold += 1
            if cd > 0: cd -= 1
        if in_pos:
            cash = qty * sigs[-1]['close'] * (1 - FEE); trades += 1
        pnl = (cash / 1000.0 - 1.0) * 100.0
        arr = np.array(ec); peak = np.maximum.accumulate(arr); dd = (arr - peak) / peak
        mdd = float(dd.min()) * 100.0 if len(dd) else 0.0
        return dict(pnl_pct=pnl, dd_pct=mdd, trades=trades, skipped=skipped)

    # Normalize signal datetimes to UTC
    for s in signals:
        t = pd.Timestamp(s['datetime'])
        s['datetime'] = t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')

    end_t = signals[-1]['datetime']
    t_h1_lo = end_t - pd.Timedelta(days=half_days)
    t_h2_lo = end_t - pd.Timedelta(days=days)
    sigs_h1 = [s for s in signals if s['datetime'] >= t_h1_lo]
    sigs_h2 = [s for s in signals if t_h2_lo <= s['datetime'] < t_h1_lo]
    sigs_ref = [s for s in signals if s['datetime'] >= t_h2_lo]

    if len(sigs_h2) < 100:
        print(f"  {asset}: signal stream too short — H2 has only {len(sigs_h2)} signals. SKIPPING.")
        return None

    def build_rr(ws):
        df = pd.DataFrame([{'datetime': s['datetime'], 'close': s['close']} for s in ws])
        df = df.sort_values('datetime').reset_index(drop=True)
        return {h: ((df['close'] / df['close'].shift(h) - 1.0) * 100.0).values for h in HORIZONS}
    rr_h1 = build_rr(sigs_h1); rr_h2 = build_rr(sigs_h2); rr_ref = build_rr(sigs_ref)

    b_h1 = simulate(sigs_h1, {}, 8, 36, 9999, 9999, 0, asset_cfg)
    b_h2 = simulate(sigs_h2, {}, 8, 36, 9999, 9999, 0, asset_cfg)
    b_ref = simulate(sigs_ref, {}, 8, 36, 9999, 9999, 0, asset_cfg)
    print(f"  {asset} | sigs h1={len(sigs_h1)} h2={len(sigs_h2)} ref={len(sigs_ref)}")
    print(f"  {asset} | baselines V0  H1={b_h1['pnl_pct']:+.2f}%  "
          f"H2={b_h2['pnl_pct']:+.2f}%  REF={b_ref['pnl_pct']:+.2f}%")

    pairs = [(a, b) for i, a in enumerate(HORIZONS) for b in HORIZONS[i+1:]]
    total = sum(len(thr_for(a)) * len(thr_for(b)) for a, b in pairs) * len(CD_GRID)
    print(f"  {asset} | sweeping {total:,} configs ...")

    rows = []
    t0 = time.time()
    for h_s, h_l in pairs:
        for t_s in thr_for(h_s):
            for t_l in thr_for(h_l):
                for cd in CD_GRID:
                    r1 = simulate(sigs_h1, rr_h1, h_s, h_l, t_s, t_l, cd, asset_cfg)
                    r2 = simulate(sigs_h2, rr_h2, h_s, h_l, t_s, t_l, cd, asset_cfg)
                    rR = simulate(sigs_ref, rr_ref, h_s, h_l, t_s, t_l, cd, asset_cfg)
                    rows.append(dict(
                        h_short=h_s, h_long=h_l, t_short=t_s, t_long=t_l, cd=cd,
                        pnl_H1=r1['pnl_pct'], pnl_H2=r2['pnl_pct'], pnl_REF=rR['pnl_pct'],
                        dd_H1=r1['dd_pct'], dd_H2=r2['dd_pct'], dd_REF=rR['dd_pct'],
                        tr_H1=r1['trades'], tr_H2=r2['trades'], tr_REF=rR['trades'],
                        sk_H1=r1['skipped'], sk_H2=r2['skipped'], sk_REF=rR['skipped'],
                    ))
    print(f"  {asset} | sweep done in {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    df['beats_H1']  = df['pnl_H1']  > b_h1['pnl_pct']
    df['beats_H2']  = df['pnl_H2']  > b_h2['pnl_pct']
    df['beats_REF'] = df['pnl_REF'] > b_ref['pnl_pct']
    df['beats_3of3'] = df['beats_H1'] & df['beats_H2'] & df['beats_REF']
    df['avg_pnl_halves'] = (df['pnl_H1'] + df['pnl_H2']) / 2.0
    df['worst_dd'] = np.maximum(df['dd_H1'].abs(), df['dd_H2'].abs())
    df['score_dd_aware'] = df['avg_pnl_halves'] - 0.5 * df['worst_dd']
    df['score_recent'] = df['pnl_H1'] - 0.5 * df['dd_H1'].abs()

    h_idx = {h: i for i, h in enumerate(HORIZONS)}
    cd_idx = {c: i for i, c in enumerate(CD_GRID)}
    key_to_idx = {(int(r['h_short']), int(r['h_long']), float(r['t_short']),
                   float(r['t_long']), int(r['cd'])): i for i, r in df.iterrows()}
    beats3 = df['beats_3of3'].values

    def nb_idx(hs, hl, ts, tl, cd, dim, step):
        if dim == 'hs':
            i = h_idx[hs] + step
            if not (0 <= i < len(HORIZONS)): return None
            new = HORIZONS[i]
            if new >= hl: return None
            key = (new, hl, ts, tl, cd)
        elif dim == 'hl':
            i = h_idx[hl] + step
            if not (0 <= i < len(HORIZONS)): return None
            new = HORIZONS[i]
            if new <= hs or tl not in thr_for(new): return None
            key = (hs, new, ts, tl, cd)
        elif dim == 'ts':
            new = round(ts + 0.5 * step, 2)
            if new not in thr_for(hs): return None
            key = (hs, hl, new, tl, cd)
        elif dim == 'tl':
            new = round(tl + 0.5 * step, 2)
            if new not in thr_for(hl): return None
            key = (hs, hl, ts, new, cd)
        elif dim == 'cd':
            i = cd_idx[cd] + step
            if not (0 <= i < len(CD_GRID)): return None
            key = (hs, hl, ts, tl, CD_GRID[i])
        return key_to_idx.get(key)

    plateau = np.zeros(len(df))
    for i, r in df.iterrows():
        nbrs = []
        for dim in ('hs', 'hl', 'ts', 'tl', 'cd'):
            for step in (-1, 1):
                j = nb_idx(int(r['h_short']), int(r['h_long']), float(r['t_short']),
                           float(r['t_long']), int(r['cd']), dim, step)
                if j is not None: nbrs.append(j)
        plateau[i] = sum(beats3[j] for j in nbrs) / len(nbrs) if nbrs else 0.0
    df['plateau_score'] = plateau

    os.makedirs('output', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'output/rally_cd_{asset}_{ts}.csv'
    df.to_csv(out_path, index=False)
    print(f"  {asset} | wrote {out_path}")

    strict = df[df['beats_3of3'] & (df['plateau_score'] >= PLATEAU_THR)]
    strict = strict.sort_values(rank_col, ascending=False)
    print(f"  {asset} | beats_3of3={int(df['beats_3of3'].sum())}  "
          f"STRICT (3of3 + plateau>={PLATEAU_THR})={len(strict)}")
    if len(strict) == 0:
        print(f"  {asset} | NO STRICT WINNER — config not modified.")
        return None

    cols = ['h_short','h_long','t_short','t_long','cd',
            'pnl_H1','pnl_H2','pnl_REF','worst_dd','score_recent','score_dd_aware','plateau_score']
    print(f"  {asset} | top 5 STRICT:")
    print(strict[cols].head(5).to_string(index=False,
          formatters={c: '{:+.2f}'.format for c in cols
                      if c.startswith(('pnl_','score_','plateau','worst'))}))

    win = strict.iloc[0]
    print(f"\n  {asset} WINNER: rr{int(win['h_short'])}h>={win['t_short']}% "
          f"OR rr{int(win['h_long'])}h>={win['t_long']}%, cd={int(win['cd'])}h")

    winner_dict = {
        'enabled': True,
        'h_short': int(win['h_short']),
        'h_long': int(win['h_long']),
        't_short_pct': float(win['t_short']),
        't_long_pct': float(win['t_long']),
        'cd_hours': int(win['cd']),
    }

    if write_config:
        cfg = {}
        if os.path.exists(REGIME_CONFIG_PATH):
            with open(REGIME_CONFIG_PATH) as f:
                cfg = json.load(f)
        if asset not in cfg:
            cfg[asset] = {'enabled': False, 'symbol': f'{asset}-USD'}
        # Route the winner to the right spot based on regime_filter
        if regime_filter in ('bull', 'bear'):
            cfg[asset].setdefault(regime_filter, {})['rally_cooldown'] = winner_dict
            write_loc = f"{asset}.{regime_filter}.rally_cooldown"
        else:
            cfg[asset]['rally_cooldown'] = winner_dict
            write_loc = f"{asset}.rally_cooldown"
        _atomic_write_json(REGIME_CONFIG_PATH, cfg)
        print(f"  {asset} | wrote {write_loc} to {REGIME_CONFIG_PATH}")

    return winner_dict


# ============================================================
# MODE G — RALLY-COOLDOWN OR-GATE SWEEP (cache-fed standalone)
# ============================================================
def run_mode_g(assets_list, args):
    """Standalone Mode G: load cached tagged signals (pre-built by extend_caches_90d.py),
    call _sweep_rally_cooldown. Useful when models haven't changed and you want fast iteration.
    For fresh signals, prefer Mode T which auto-chains rally-cooldown after its own sweep."""
    replay_h = int(getattr(args, 'replay', 0)) or 1440
    rank = getattr(args, 'rank', 'recent')

    try:
        with open(REGIME_CONFIG_PATH) as f:
            regime_cfg_all = json.load(f)
    except Exception as e:
        print(f"  Mode G: cannot read {REGIME_CONFIG_PATH}: {e}")
        return

    for asset in assets_list:
        cache_path = None
        for n in (90, 60, 30):
            p = os.path.join('data', f'{asset.lower()}_sl_signals_{n}d.pkl')
            if os.path.exists(p):
                cache_path = p; break
        if cache_path is None:
            print(f"  {asset}: no signal cache (data/{asset.lower()}_sl_signals_*.pkl). "
                  f"Run extend_caches_90d.py for {asset}, or use Mode T for fresh signals. SKIPPING.")
            continue

        # Cache-freshness guard: standalone G is useful only when models haven't changed.
        # If the production CSV is newer than the cache, the signals are stale.
        try:
            cache_mtime = os.path.getmtime(cache_path)
            prod_mtime = os.path.getmtime(PRODUCTION_CSV)
            if prod_mtime > cache_mtime:
                age_min = (prod_mtime - cache_mtime) / 60.0
                print("=" * 80)
                print(f"  {asset}: STALE CACHE WARNING")
                print(f"  Signal cache: {cache_path} ({datetime.fromtimestamp(cache_mtime)})")
                print(f"  Production CSV: {PRODUCTION_CSV} ({datetime.fromtimestamp(prod_mtime)})")
                print(f"  Production is {age_min:.0f} minutes NEWER than the cache.")
                print(f"  The signals in the cache were generated by older models.")
                print("  Options:")
                print("    - Prefer Mode T (regenerates signals fresh, chains G automatically)")
                print("    - Or rebuild: python extend_caches_90d.py")
                print(f"  SKIPPING {asset} to avoid writing bad config.")
                print("=" * 80)
                continue
        except OSError:
            pass

        with open(cache_path, 'rb') as f:
            signals = pickle.load(f)

        asset_cfg = regime_cfg_all.get(asset, {})
        _sweep_rally_cooldown(asset, signals, asset_cfg, replay_h, rank=rank, write_config=True)


def main():

    has_macro = os.path.exists(MACRO_DIR)

    # ================================================================
    # CLI: python crypto_trading_system_ed.py D BTC 4,8h
    # ================================================================
    # Order-independent CLI parser
    # Any order works: MODE ASSETS HORIZONS, ASSETS MODE HORIZONS, etc.
    # Examples:
    #   python crypto_trading_system_ed.py D BTC 4,8h
    #   python crypto_trading_system_ed.py BTC D 4,8h --trials 150
    #   python crypto_trading_system_ed.py H 5,6,7,8h BTC --skip
    #   python crypto_trading_system_ed.py DF BTC,ETH 4,8h
    # ================================================================
    VALID_MODES = {'P', 'D', 'DS', 'DV', 'DVS', 'DVRS', 'S', 'V', 'H', 'HRS', 'HRST', 'R', 'RS', 'T', 'G'}

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

    # Parse --replay N (Mode R: replay hours, default 2880)
    flag_replay = 0
    for i, a in enumerate(sys.argv[1:], 1):
        if a == '--replay' and i < len(sys.argv) - 1:
            try:
                flag_replay = int(sys.argv[i + 1])
            except ValueError:
                pass

    # Parse --conf N (Mode R: confidence threshold, default 90)
    flag_conf = 0
    for i, a in enumerate(sys.argv[1:], 1):
        if a == '--conf' and i < len(sys.argv) - 1:
            try:
                flag_conf = int(sys.argv[i + 1])
            except ValueError:
                pass

    # Parse --top N (Mode R: top N results, default 15)
    flag_top = 0
    for i, a in enumerate(sys.argv[1:], 1):
        if a == '--top' and i < len(sys.argv) - 1:
            try:
                flag_top = int(sys.argv[i + 1])
            except ValueError:
                pass

    # Parse --rank recent|balanced (Mode G tiebreak), default 'recent'
    flag_rank = 'recent'
    for i, a in enumerate(sys.argv[1:], 1):
        if a == '--rank' and i < len(sys.argv) - 1:
            v = sys.argv[i + 1].lower()
            if v in ('recent', 'balanced'):
                flag_rank = v
            else:
                print(f"  Unknown --rank value '{v}'. Valid: recent, balanced")
                return

    # Parse --max-iter N (Mode T: T<->G convergence iterations, default 4)
    flag_max_iter = 0
    for i, a in enumerate(sys.argv[1:], 1):
        if a == '--max-iter' and i < len(sys.argv) - 1:
            try:
                flag_max_iter = int(sys.argv[i + 1])
            except ValueError:
                pass

    # --help
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""
Usage: python crypto_trading_system_ed.py [MODE] [ASSETS] [HORIZONS] [OPTIONS]

  Arguments are order-independent — MODE, ASSETS, HORIZONS can appear in any order.

Modes:
  P       PySR feature discovery (symbolic regression → models/pysr_*.json)
  D       Grid optimization (combo x window x gamma x features)
  V       Validate (top 6 from D → refine top 3 → pick best)
  DV      D then V
  S       Regime confidence optimization (sweep bull/bear confidence → write config)
  RS      R then S (find best regime pair → optimize confidence)
  H       Horizon sweep (D+V per horizon — produces models, no winner picking)
  T       Threshold sweep (hold-until-profitable: min_sell_pnl × max_hold_hours)
  HRS     Full Ed pipeline: H → R → S (all horizons → regime pair → confidence)
  HRST    Full Ed pipeline + threshold: H → R → S → T
  DVRS    Same as HRS for specified horizons
  R       Regime backtest (bull/bear horizon switching with regime detectors)
  G       Rally-cooldown gate sweep — cache-fed standalone (fast iteration)
          NOTE: Mode T chains this sweep automatically against fresh signals, so
          HRST is the canonical way to get shield + gate together. G standalone
          is kept for fast re-runs when models haven't changed.

Assets:
  BTC,ETH,LINK,...   Comma-separated asset names (default: all)

Horizons:
  5,6,7,8h           Comma-separated horizons in hours (default: 4,8h)

Options:
  --trials N          Number of Optuna trials (default: 150)
  --metric NAME       Scoring metric: apf, rawpf, calmar, return, rpf_sqrt, all
  --skip              Mode H only: skip Mode D for horizons that already have results
  --resume            Resume interrupted Optuna study
  --replay N          Mode D/V/R/S/T: data window in hours (default: 1440=2mo)
  --conf N            Mode R only: confidence threshold (default: 90)
  --top N             Mode R only: show top N results (default: 15)
  --rank MODE         Mode G tiebreak: recent (H1-focused, default) | balanced (H1+H2 avg)
  --max-iter N        Mode T: T<->G convergence iterations (default 4; 1=single-pass legacy)
  --help, -h          Show this help

Examples:
  python crypto_trading_system_ed.py HRS BTC 5,6,7,8h          # full Ed pipeline
  python crypto_trading_system_ed.py DVRS BTC 5,6,7,8h         # same as HRS
  python crypto_trading_system_ed.py RS BTC 5,6,7,8h            # regime pair + confidence (skip D/V)
  python crypto_trading_system_ed.py S BTC                       # optimize confidence only (uses current config)
  python crypto_trading_system_ed.py R BTC 5,6,7,8h --replay 2880 --conf 85 --top 20
  python crypto_trading_system_ed.py P BTC 6h                  # discover PySR features (~30-120 min)
  python crypto_trading_system_ed.py H BTC 5,6,7,8h          # horizon sweep (D+V only)
  python crypto_trading_system_ed.py H BTC 5,6,7h --skip     # skip D, re-run V only
  python crypto_trading_system_ed.py DV ETH 6h               # optimize + validate ETH 6h
  python crypto_trading_system_ed.py D BTC,ETH 8h --trials 200
  python crypto_trading_system_ed.py V BTC 6h                 # re-validate existing results
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
    # Remove values that follow --trials, --metric, --replay, --conf, --top
    skip_next = set()
    for i, a in enumerate(sys.argv[1:], 1):
        if a in ('--trials', '--metric', '--replay', '--conf', '--top', '--rank', '--max-iter') and i < len(sys.argv) - 1:
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
            if mode in ('H', 'HRS', 'DVRS', 'R', 'RS'):
                horizons = [5, 6, 7, 8]  # Ed default: test 4 horizons
            elif mode in ('P', 'DV', 'DVS'):
                horizons = list(AVAILABLE_HORIZONS)
            else:
                horizons = [HORIZON_SHORT]

        trials_str = f" | {n_trials} trials" if mode in ('D', 'DS', 'DV', 'DVS', 'DVRS', 'H', 'HRS') else ""
        skip_str = " | --skip" if flag_skip and mode in ('H', 'HRS', 'DVRS') else ""
        h_str = '' if mode == 'G' else ' | ' + ','.join(str(h)+'h' for h in horizons)
        print("=" * 60)
        print(f"  ED: Mode {mode} | {','.join(assets_list)}{h_str}{trials_str}{skip_str}")
        print("=" * 60)

    else:

        print("=" * 60)
        print("  CRYPTO HOURLY ML TRADING SYSTEM -- ED")
        print("  Exhaustive grid + Optuna refine + regime-switching validation")
        print(f"  Prediction: variable horizons (specify via CLI)")
        print(f"  Macro data: {'FOUND' if has_macro else 'NOT FOUND -- run download_macro_data.py'}")
        print("=" * 60)

        print("\nChoose mode:")
        print("  P.  PySR FEATURE DISCOVERY (symbolic regression → new features)")
        print("  D.  GRID OPTIMIZATION (combo × window × gamma × features)")
        print("  DV. D then V (grid + validate top 6 + refine top 3 + pick best)")
        print("  DVS. DV then S (optimize + validate + confidence)")
        print("  S.  REGIME CONFIDENCE OPTIMIZATION (sweep bull/bear confidence)")
        print("  RS. R then S (find best regime pair + optimize confidence)")
        print("  V.  VALIDATE (top 6 candidates from Mode D, pick best)")
        print("  H.  HORIZON SWEEP (D+V per horizon — produces models)")
        print("  HRS. Full Ed pipeline: H → R → S (all horizons → regime pair → confidence)")
        print("  R.  REGIME BACKTEST (bull/bear horizon switching with regime detectors)")
        print("  DVRS. Same as HRS for specified horizons")
        print("  G.  RALLY-COOLDOWN gate sweep — cache-fed standalone (T chains G automatically)")
        mode = input("\nEnter P/D/DV/DVS/S/RS/V/H/HRS/HRST/R/DVRS/G/T: ").strip().upper()


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

        if mode in ('D', 'DV', 'DVS', 'DVRS', 'HRS', 'HRST'):
            try:
                trials_input = input(f"Number of Optuna trials [{DEKU_DEFAULT_TRIALS}]: ").strip()
                if trials_input:
                    n_trials = int(trials_input)
            except ValueError:
                pass

    # Execute mode
    if run_all_metrics and mode == 'D':
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
            class _Args: pass
            _r_args = _Args()
            _r_args.replay = 2880
            _r_args.top = 15
            run_mode_s(assets_list, horizons, _r_args)

            # Read the trading config that Mode S just wrote
            metric_suffix = f'_{metric}' if metric != 'apf' else ''
            tcfg_path = f'{CONFIG_DIR}/regime_config_ed{metric_suffix}.json'
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
    elif mode == 'R':
        # ── Mode R: Regime backtest ──
        class _Args: pass
        _r_args = _Args()
        _r_args.replay = flag_replay
        _r_args.conf = flag_conf
        _r_args.top = flag_top
        _run_mode_r(assets_list, horizons, _r_args)
    elif mode == 'S':
        class _Args: pass
        _r_args = _Args()
        _r_args.replay = flag_replay
        _r_args.top = flag_top
        run_mode_s(assets_list, horizons, _r_args)
    elif mode == 'RS':
        class _Args: pass
        _r_args = _Args()
        _r_args.replay = flag_replay
        _r_args.conf = flag_conf
        _r_args.top = flag_top
        r_results = _run_mode_r(assets_list, horizons, _r_args)
        _apply_mode_r_to_config(r_results)
        run_mode_s(assets_list, horizons, _r_args)
    elif mode in ('HRS', 'DVRS', 'HRST'):
        run_mode_h(assets_list, horizons, n_trials=n_trials, resume=flag_resume, skip_d=flag_skip)
        class _Args: pass
        _r_args = _Args()
        _r_args.replay = flag_replay
        _r_args.conf = flag_conf
        _r_args.top = flag_top
        _r_args.max_iter = flag_max_iter
        r_results = _run_mode_r(assets_list, horizons, _r_args)
        _apply_mode_r_to_config(r_results)
        run_mode_s(assets_list, horizons, _r_args)
        if mode == 'HRST':
            run_mode_t(assets_list, _r_args)
    elif mode == 'G':
        class _Args: pass
        _r_args = _Args()
        _r_args.replay = flag_replay
        _r_args.rank = flag_rank
        run_mode_g(assets_list, _r_args)
    elif mode == 'T':
        class _Args: pass
        _r_args = _Args()
        _r_args.replay = flag_replay
        _r_args.max_iter = flag_max_iter
        run_mode_t(assets_list, _r_args)
    elif mode == 'P':
        run_mode_p(assets_list, horizons)
    elif mode in ('D', 'DV', 'DVS'):
        for asset in assets_list:
            print(f"\n{'='*60}")
            print(f"  ASSET: {asset}")
            print(f"{'='*60}")
            for h in horizons:
                if len(horizons) > 1:
                    print(f"\n{'#'*60}")
                    print(f"  RUNNING {h}h HORIZON")
                    print(f"{'#'*60}")
                run_mode_d_optuna([asset], horizon=h, n_trials=n_trials, resume=flag_resume, replay_hours=flag_replay or None)
            if mode in ('DVS', 'DV'):
                run_mode_v([asset], horizons, replay_hours=flag_replay or None)
                if mode == 'DVS':
                    class _Args: pass
                    _r_args = _Args()
                    _r_args.replay = flag_replay
                    _r_args.top = flag_top
                    run_mode_s([asset], horizons, _r_args)
    elif mode == 'V':
        run_mode_v(assets_list, horizons, replay_hours=flag_replay or None)
    elif mode == 'H':
        run_mode_h(assets_list, horizons, n_trials=n_trials, resume=flag_resume, skip_d=flag_skip)

    print("\nDone!")


if __name__ == '__main__':
    main()
