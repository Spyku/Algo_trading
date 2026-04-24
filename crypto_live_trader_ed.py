"""
Crypto Live Trader (Ed) — Signal Generation for Ed Models
=============================================================
Signal generation library for the Ed live trader (crypto_revolut_ed_v2.py).
Imports from crypto_trading_system_ed and reads models/crypto_ed_production.csv.

Not run directly — used as a module by crypto_revolut_ed_v2.py.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
import ssl
_ssl_ctx = ssl._create_unverified_context()
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo('Europe/Zurich')
except ImportError:
    LOCAL_TZ = None

def _to_local(dt):
    if dt is None or not hasattr(dt, 'strftime'):
        return dt
    try:
        if LOCAL_TZ:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(LOCAL_TZ)
        else:
            if dt.tzinfo is None:
                return dt + timedelta(hours=1)
            return dt.astimezone(timezone(timedelta(hours=1)))
    except Exception:
        return dt

os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')

from crypto_trading_system_ed import (
    ASSETS, FEATURE_SET_A, FEATURE_SET_B,
    PREDICTION_HORIZON, ALL_MODELS,
    HORIZON_SHORT, HORIZON_LONG, AVAILABLE_HORIZONS,
    download_asset, load_data, build_all_features,
    get_decay_weights, _compute_pysr_features,
)
from sklearn.preprocessing import StandardScaler

# ── Runtime error inbox + rate-limited alerts (Fix #5 — 2026-04-24) ────
# Two-sink alert system:
#   1. Telegram push (rate-limited so recurring failures don't spam)
#   2. output/ERRORS_INBOX.md append (append-only, survives every restart)
# CLAUDE.md instructs me to read the inbox when user asks about TODO, so
# operational errors are surfaced during any status review. User can clear
# the file manually after triage.
_TG_RATE_LIMIT_LT = {}  # key -> last_sent_epoch

ERRORS_INBOX_PATH = os.path.join(os.path.dirname(__file__), 'output', 'ERRORS_INBOX.md')

def _append_error_inbox(key, msg, severity='warn'):
    """Append a runtime error entry to ERRORS_INBOX.md. Never raises."""
    try:
        os.makedirs(os.path.dirname(ERRORS_INBOX_PATH), exist_ok=True)
        if not os.path.exists(ERRORS_INBOX_PATH):
            header = (
                "# Runtime Errors Inbox (Ed V2 trader)\n\n"
                "Appended automatically by the live trader and helpers. Rate-limited to\n"
                "1 entry per unique key per hour. Severity: info / warn / critical.\n\n"
                "**Triage:** review → fix upstream → clear file (or move old entries to archive).\n\n"
                "---\n\n"
            )
            with open(ERRORS_INBOX_PATH, 'w', encoding='utf-8') as f:
                f.write(header)
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        icon = {'info': 'ℹ', 'warn': '⚠', 'critical': '🚨'}.get(severity, '•')
        line = f"- `{ts}` {icon} **{severity.upper()}** `{key}` — {msg}\n"
        with open(ERRORS_INBOX_PATH, 'a', encoding='utf-8') as f:
            f.write(line)
    except Exception:
        pass  # inbox is best-effort; never fail a caller because of logging

def _rate_limited_telegram_lt(key, msg, cooldown_sec=3600, severity='warn'):
    """Send Telegram alert AND append to ERRORS_INBOX.md; skip if same `key`
    alerted within cooldown_sec. Never raises — safe in except blocks.

    severity: 'info' | 'warn' | 'critical' — controls inbox icon.
    """
    import time as _t
    now = _t.time()
    last = _TG_RATE_LIMIT_LT.get(key, 0)
    if now - last < cooldown_sec:
        return False
    _TG_RATE_LIMIT_LT[key] = now
    _append_error_inbox(key, msg, severity)
    try:
        send_telegram(msg)
        return True
    except Exception:
        return False


# ── Stdout tee: always write to logs/ed_runtime_*.log (Fix #5D — 2026-04-24) ──
# Redirect stdout+stderr so every `print(...)` survives, even when the
# trader is launched ad-hoc without tee_launcher.bat. Independent of
# external tee — belt-and-suspenders logging.
class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

_RUNTIME_LOG_FILE = None

def init_runtime_log():
    """Install a TeeStream so stdout/stderr also write to a dated log file.
    Idempotent — safe to call multiple times. Returns the log path."""
    global _RUNTIME_LOG_FILE
    if _RUNTIME_LOG_FILE is not None:
        return _RUNTIME_LOG_FILE.name
    try:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'ed_runtime_{ts}.log')
        _RUNTIME_LOG_FILE = open(log_path, 'a', encoding='utf-8', buffering=1)
        sys.stdout = _TeeStream(sys.stdout, _RUNTIME_LOG_FILE)
        sys.stderr = _TeeStream(sys.stderr, _RUNTIME_LOG_FILE)
        print(f"[runtime-log] tee installed → {log_path}")
        return log_path
    except Exception as e:
        print(f"[!] runtime-log init failed: {e}")
        return None


# ── Regime detection ──────────────────────────────────────────────────
REGIME_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config', 'regime_config_ed.json')
_regime_config_cache = {}
_regime_config_mtime = 0

def load_regime_config():
    """Load regime_config_ed.json with caching based on file mtime."""
    global _regime_config_cache, _regime_config_mtime
    try:
        mtime = os.path.getmtime(REGIME_CONFIG_FILE)
        if mtime != _regime_config_mtime:
            with open(REGIME_CONFIG_FILE, 'r', encoding='utf-8') as f:
                _regime_config_cache = json.load(f)
            _regime_config_mtime = mtime
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if not _regime_config_cache:
            print(f"  [!] Regime config error: {e}")
    return _regime_config_cache


def detect_regime(asset, df_prices):
    """Detect bull/bear regime for an asset using regime_config_ed.json.

    Args:
        asset: Asset name (e.g., 'BTC')
        df_prices: DataFrame with at least 'close', 'high', 'low', 'datetime' columns

    Returns:
        tuple: (regime_str, active_config) where regime_str is
               'bull' | 'bear' | 'error'  (Fix #2 2026-04-24: 'error' added)
               and active_config is {'horizon': int, 'min_confidence': int, 'max_position_usd': float}

        Callers MUST check regime_str=='error' and refuse to trade that cycle.
        Telegram alert has already been sent by the detector on 'error' return.
    """
    cfg = load_regime_config()
    asset_cfg = cfg.get(asset, {})

    if not asset_cfg or not asset_cfg.get('enabled'):
        # Fallback: return bull config as default
        bull = asset_cfg.get('bull', {'horizon': 6, 'min_confidence': 85, 'max_position_usd': 0})
        return 'bull', bull

    detector = asset_cfg.get('regime_detector', {})
    det_type = detector.get('type', 'fixed')
    params = detector.get('params', {})

    # Evaluate detector — may return None on unrecoverable error
    is_bull = _evaluate_detector(det_type, params, df_prices, asset)

    if is_bull is None:
        # Detector errored. Telegram already sent. Refuse to trade this cycle.
        return 'error', {}

    regime = 'bull' if is_bull else 'bear'
    active = asset_cfg.get(regime, asset_cfg.get('bull', {}))

    return regime, active


def _evaluate_detector(det_type, params, df, asset):
    """Evaluate a regime detector.

    Returns:
        True  = bull
        False = bear
        None  = UNRECOVERABLE error (unknown detector type / unhandled exception).
                Caller MUST refuse to trade. Rate-limited Telegram alert fired.

    Per-branch: insufficient data → silent bull default (cold-start tolerance).
    Unknown-name / exception / PySR-unevaluable → None + Telegram (Fix #2 2026-04-24).
    """
    try:
        if det_type == 'fixed':
            return params.get('regime', 'bull') == 'bull'

        elif det_type == 'sma_cross':
            fast = int(params.get('fast', 24))
            slow = int(params.get('slow', 100))
            if len(df) < slow + 10:
                return True  # cold-start: not enough data, silent bull default
            sma_fast = df['close'].rolling(fast).mean().iloc[-1]
            sma_slow = df['close'].rolling(slow).mean().iloc[-1]
            return sma_fast > sma_slow

        elif det_type == 'price_vs_sma':
            period = int(params.get('period', 100))
            if len(df) < period + 10:
                return True
            sma = df['close'].rolling(period).mean().iloc[-1]
            return df['close'].iloc[-1] > sma

        elif det_type == 'rsi':
            level = float(params.get('level', 50))
            delta = df['close'].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, float('nan'))
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) > level

        elif det_type == 'drawdown':
            threshold = float(params.get('threshold', -3))
            window = int(params.get('window', 48))
            if len(df) < window + 10:
                return True
            rolling_high = df['high'].rolling(window).max().iloc[-1]
            dd = (df['close'].iloc[-1] / rolling_high - 1) * 100
            return dd > threshold

        elif det_type == 'pysr':
            return _evaluate_pysr_detector(params, df, asset)

        elif det_type == 'named':
            # Mode S writes detector by name. Must match the dict in
            # crypto_trading_system_ed.py::_build_regime_indicators_and_detectors.
            return _evaluate_named_detector(params.get('name', ''), df)

        else:
            # Unknown detector type = config error, not graceful degradation.
            print(f"  [!!] Unknown regime detector type: {det_type} — REFUSING to trade")
            _rate_limited_telegram_lt(
                f'regime_unknown_type_{det_type}',
                f"🚨 {asset}: unknown regime detector type '{det_type}' — refusing to trade this cycle. Fix config/regime_config_ed.json.",
            )
            return None  # caller must refuse

    except Exception as e:
        print(f"  [!!] Regime detection exception ({det_type}): {e} — REFUSING to trade")
        _rate_limited_telegram_lt(
            f'regime_exception_{asset}_{det_type}',
            f"🚨 {asset}: regime detector '{det_type}' raised exception: {e} — refusing to trade this cycle.",
        )
        return None  # caller must refuse


def _evaluate_named_detector(name, df):
    """Evaluate a named detector. Must mirror the dict in
    crypto_trading_system_ed.py::_build_regime_indicators_and_detectors.
    Supported names: sma24>sma100, sma168>sma480, price>sma72, vol_calm, tsmom_672h.

    Return convention (Fix #2 2026-04-24):
        True  = bull
        False = bear
        None  = unknown name OR unhandled exception — caller MUST refuse to trade.
    Insufficient-data returns True (cold-start default).
    """
    import numpy as _np
    if not name:
        # Empty name = unconfigured detector = real error
        print(f"  [!!] Named detector called with empty name — REFUSING to trade")
        _rate_limited_telegram_lt(
            'regime_named_empty',
            "🚨 regime_detector.params.name is empty/unconfigured — refusing to trade. Check config/regime_config_ed.json.",
        )
        return None
    try:
        if name == 'sma24>sma100':
            if len(df) < 110: return True  # cold-start
            return df['close'].rolling(24).mean().iloc[-1] > df['close'].rolling(100).mean().iloc[-1]

        if name == 'sma168>sma480':
            if len(df) < 490: return True
            return df['close'].rolling(168).mean().iloc[-1] > df['close'].rolling(480).mean().iloc[-1]

        if name == 'price>sma72':
            if len(df) < 82: return True
            return df['close'].iloc[-1] > df['close'].rolling(72).mean().iloc[-1]

        if name == 'vol_calm':
            # Andersen-Bollerslev deseasonalized 24h vol vs its 30d 70th percentile
            if len(df) < 800: return True
            d = df.copy()
            if 'datetime' in d.columns:
                d['datetime'] = pd.to_datetime(d['datetime'])
                d = d.set_index('datetime').sort_index()
            d['logret_1h'] = _np.log(d['close'] / d['close'].shift(1))
            d['abs_logret'] = d['logret_1h'].abs()
            d['hour'] = d.index.hour
            d['seasonal_factor'] = d.groupby('hour')['abs_logret'].transform(
                lambda s: s.rolling(30, min_periods=10).mean()
            )
            d['abs_logret_deseason'] = d['abs_logret'] / d['seasonal_factor'].replace(0, _np.nan)
            vol = d['abs_logret_deseason'].rolling(24).std()
            q70 = vol.rolling(720, min_periods=240).quantile(0.70)
            return float(vol.iloc[-1]) < float(q70.iloc[-1])

        if name == 'tsmom_672h':
            if len(df) < 680: return True
            return _np.log(df['close'].iloc[-1] / df['close'].iloc[-672]) > 0

        # Unknown name = typo in config = real error
        print(f"  [!!] Unknown named detector: '{name}' — REFUSING to trade")
        _rate_limited_telegram_lt(
            f'regime_named_unknown_{name}',
            f"🚨 Unknown regime detector name '{name}' — refusing to trade. Valid: sma24>sma100, sma168>sma480, price>sma72, vol_calm, tsmom_672h.",
        )
        return None
    except Exception as e:
        print(f"  [!!] Named detector '{name}' exception: {e} — REFUSING to trade")
        _rate_limited_telegram_lt(
            f'regime_named_exc_{name}',
            f"🚨 Named detector '{name}' raised: {e} — refusing to trade this cycle.",
        )
        return None


def _evaluate_pysr_detector(params, df, asset):
    """Evaluate a PySR-discovered regime formula.

    Return convention (Fix #2 2026-04-24):
        True  = bull
        False = bear
        None  = missing JSON, unevaluable formula, NaN inputs, or exception
                → caller MUST refuse to trade. Rate-limited Telegram alert.
    """
    model_file = params.get('model_file', '')
    expr_index = int(params.get('expression_index', 0))
    threshold = float(params.get('threshold', 0.5))

    model_path = os.path.join(os.path.dirname(__file__), model_file)
    if not os.path.exists(model_path):
        print(f"  [!!] PySR regime model not found: {model_path} — REFUSING to trade")
        _rate_limited_telegram_lt(
            f'pysr_regime_missing_{model_file}',
            f"🚨 PySR regime model missing: {model_file} — refusing to trade. Run Mode P or fix config.",
        )
        return None

    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            pysr_data = json.load(f)

        expressions = pysr_data.get('expressions', [])
        if expr_index >= len(expressions):
            print(f"  [!!] PySR regime: expression_index {expr_index} out of range (have {len(expressions)}) — REFUSING")
            _rate_limited_telegram_lt(
                f'pysr_regime_idx_{model_file}',
                f"🚨 PySR regime: expression_index {expr_index} out of range for {model_file} (have {len(expressions)}) — refusing to trade.",
            )
            return None

        expr = expressions[expr_index]
        equation = expr.get('sympy_format', expr.get('equation', ''))
        feat_names = pysr_data.get('feature_names', [])

        # Build feature values from df
        from tools.pysr_discover_regime import build_regime_features
        features = build_regime_features(df)

        # Evaluate expression using sympy
        import sympy
        local_dict = {}
        bad_feats = []
        for fname in feat_names:
            if fname in features.columns:
                val = features[fname].iloc[-1]
                if np.isfinite(val):
                    local_dict[fname] = val
                else:
                    bad_feats.append(f"{fname}(NaN)")
            else:
                bad_feats.append(f"{fname}(missing)")
        if bad_feats:
            # Any missing/NaN feature = unreliable prediction = refuse
            print(f"  [!!] PySR regime {model_file}: {len(bad_feats)} unusable features {bad_feats[:5]} — REFUSING to trade")
            _rate_limited_telegram_lt(
                f'pysr_regime_feats_{model_file}',
                f"🚨 PySR regime {model_file}: {len(bad_feats)} NaN/missing features ({bad_feats[:3]}) — refusing to trade.",
            )
            return None

        result = float(sympy.sympify(equation).evalf(subs=local_dict))
        return result <= threshold  # <= threshold = bull, > threshold = bear

    except Exception as e:
        print(f"  [!!] PySR regime eval exception: {e} — REFUSING to trade")
        _rate_limited_telegram_lt(
            f'pysr_regime_exc_{model_file}',
            f"🚨 PySR regime {model_file} raised: {e} — refusing to trade.",
        )
        return None

# ============================================================
# STRATEGY CONFIG
# ============================================================
MIN_CONFIDENCE = 75  # Minimum confidence to act (%)

# ============================================================
# TELEGRAM CONFIG
# ============================================================
TELEGRAM_CONFIG = {
    'token': os.environ.get('TELEGRAM_TOKEN', ''),
    'chat_id': os.environ.get('TELEGRAM_CHAT_ID', ''),
}
TELEGRAM_CONFIG_FILE = 'config/telegram_config.json'
if os.path.exists(TELEGRAM_CONFIG_FILE):
    with open(TELEGRAM_CONFIG_FILE) as f:
        TELEGRAM_CONFIG.update(json.load(f))

def send_telegram(message, parse_mode='HTML'):
    token = TELEGRAM_CONFIG.get('token', '')
    chat_id = TELEGRAM_CONFIG.get('chat_id', '')
    if not token or not chat_id:
        print("  [!] Telegram not configured.")
        print(f"\n  Message:\n{message}")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({'chat_id': chat_id, 'text': message, 'parse_mode': parse_mode}).encode('utf-8')
    try:
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as resp:
            result = json.loads(resp.read().decode())
            if result.get('ok'):
                print("  Telegram sent")
                return True
            print(f"  [!] Telegram error: {result}")
            return False
    except Exception as e:
        print(f"  [!] Telegram error: {e}")
        return False

# ============================================================
# LOAD MODEL CONFIG
# ============================================================
MODELS_CSV = 'models/crypto_ed_production.csv'

def load_best_config(asset_name, horizon=None):
    if not os.path.exists(MODELS_CSV):
        return None
    df = pd.read_csv(MODELS_CSV)
    if 'horizon' not in df.columns:
        df['horizon'] = HORIZON_SHORT
    match = df[df['coin'] == asset_name]
    if horizon is not None:
        match = match[match['horizon'] == horizon]
    if match.empty:
        return None
    row = match.iloc[0]
    opt = row.get('optimal_features', '')
    if pd.isna(opt):
        opt = ''
    return {
        'coin': row['coin'], 'models': row['models'],
        'best_combo': row['best_combo'], 'best_window': int(row['best_window']),
        'accuracy': row['accuracy'], 'feature_set': row.get('feature_set', 'A'),
        'horizon': int(row.get('horizon', HORIZON_SHORT)), 'optimal_features': str(opt),
        'training_period': str(row['training_period']) if 'training_period' in row and pd.notna(row['training_period']) else '',
        'gamma': float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0,
        # Enhancement toggles (backward compatible — defaults to disabled)
        'return_weight': str(row.get('enh_return_weight', 'False')).lower() == 'true',
        'disagree_filter': str(row.get('enh_disagree_filter', 'False')).lower() == 'true',
        'disagree_threshold': float(row.get('enh_disagree_threshold', 0.75)) if pd.notna(row.get('enh_disagree_threshold', 0.75)) and str(row.get('enh_disagree_threshold', '')).strip() else 0.75,
        'funding_gate': str(row.get('enh_funding_gate', 'False')).lower() == 'true',
        'funding_threshold': float(row.get('enh_funding_threshold', 0.001)) if pd.notna(row.get('enh_funding_threshold', 0.001)) and str(row.get('enh_funding_threshold', '')).strip() else 0.001,
    }

# ============================================================
# GENERATE SINGLE SIGNAL
# ============================================================
def generate_live_signal(asset_name, config, df_raw=None, verbose=True, metrics_out=None):
    """Generate one signal for (asset_name, config.horizon).

    Fix #9 (2026-04-24): optional `metrics_out` dict — if provided, health
    metrics are written into it for per-cycle forensics. Keys:
      n_features_expected / n_features_available / n_features_nan_inference
      feature_set_a_fallback (bool) / inference_row_age_h
      feature_build_sec / model_fit_sec
    No behavior change when metrics_out is None.
    """
    import time as _tm
    _fb_start = _tm.time()

    model_names = config['models'].split('+')
    window = config['best_window']
    fs = config.get('feature_set', 'A')
    horizon = config.get('horizon', HORIZON_SHORT)
    opt_features = config.get('optimal_features', '')
    gamma = config.get('gamma', 1.0)

    if fs in ('D', 'E2', 'E3') and opt_features and opt_features.strip() and opt_features.strip() != 'nan':
        feature_list = [f.strip() for f in opt_features.split(',') if f.strip() and f.strip() != 'nan']
    elif fs == 'B':
        feature_list = list(FEATURE_SET_B)
    else:
        feature_list = list(FEATURE_SET_A)

    if metrics_out is not None:
        metrics_out['n_features_expected'] = len(feature_list)

    if df_raw is None:
        try:
            download_asset(asset_name, update_only=True)
        except Exception:
            return None
        df_raw = load_data(asset_name)
        if df_raw is None:
            return None

    df_full, all_cols = build_all_features(df_raw, asset_name=asset_name, horizon=horizon, verbose=verbose)
    _compute_pysr_features(df_full, all_cols, asset_name, horizon, verbose=False)
    if metrics_out is not None:
        metrics_out['feature_build_sec'] = round(_tm.time() - _fb_start, 2)

    feature_cols = [f for f in feature_list if f in all_cols]
    missing_feats = [f for f in feature_list if f not in all_cols]
    if metrics_out is not None:
        metrics_out['n_features_available'] = len(feature_cols)
        metrics_out['feature_set_a_fallback'] = False  # will flip to True if fallback hits
    if missing_feats:
        print(f"  [!] {asset_name} {horizon}h: {len(missing_feats)} configured features unavailable: {missing_feats[:5]}{'...' if len(missing_feats) > 5 else ''}")
        # Partial mismatch — warn but continue with the matching subset
        _rate_limited_telegram_lt(
            f'feat_partial_{asset_name}_{horizon}',
            f"⚠ {asset_name} {horizon}h: {len(missing_feats)}/{len(feature_list)} configured features missing from current build "
            f"({missing_feats[:3]}{'...' if len(missing_feats)>3 else ''}) — using remaining {len(feature_cols)} features.",
            severity='warn',
        )
    if not feature_cols:
        if metrics_out is not None:
            metrics_out['feature_set_a_fallback'] = True
        # Fix #5A (2026-04-24): FEATURE_SET_A fallback is CRITICAL.
        # The model was trained on a specific feature set; falling back to
        # FEATURE_SET_A means we're running a DIFFERENT model than the one
        # in backtest. Has to reach Telegram.
        print(f"  [!!] {asset_name} {horizon}h: NO configured features matched -- falling back to FEATURE_SET_A (DIFFERENT MODEL)")
        _rate_limited_telegram_lt(
            f'feat_fallback_{asset_name}_{horizon}',
            f"🚨 {asset_name} {horizon}h: ZERO configured features match build — falling back to FEATURE_SET_A. "
            f"The model is now running on features it was NOT trained with. Check crypto_ed_production.csv "
            f"optimal_features vs current build_all_features output.",
            severity='critical',
        )
        feature_cols = [f for f in FEATURE_SET_A if f in all_cols]

    # Fix #4 (2026-04-24): forward-fill THEN zero-fill (not zero-only).
    # Old behavior: fillna(0.0) — numerically wrong for log-returns where 0 is
    # a specific signal ("zero movement"), not "unknown." Forward-fill preserves
    # the last known value (stale but plausible) and zero-fill catches features
    # with no history at all.
    #
    # Additionally: rate-limited Telegram alert when any feature imputation
    # fires at the inference row. Without this, the trader silently ran in
    # "degraded features" mode — the exact pathology behind the 86%-pin bug.
    #
    # Drop only label-NaN (needed for training window). Don't drop on feature
    # NaN — that's what froze `i` on a 49h-stale row.
    df = df_full.dropna(subset=['label']).reset_index(drop=True)
    feat_na_tail = df[feature_cols].iloc[-1].isna()
    if metrics_out is not None:
        metrics_out['n_features_nan_inference'] = int(feat_na_tail.sum())
    if feat_na_tail.any():
        stale_feats = [f for f in feature_cols if pd.isna(df.iloc[-1][f])]
        print(f"  [!] {asset_name} {horizon}h: {len(stale_feats)} feature(s) NaN in latest row: {stale_feats[:5]}{'...' if len(stale_feats)>5 else ''} -- forward-fill + zero-fill")
        _rate_limited_telegram_lt(
            f'nan_impute_{asset_name}_{horizon}',
            f"⚠ {asset_name} {horizon}h: {len(stale_feats)} feature(s) NaN at inference "
            f"({stale_feats[:3]}{'...' if len(stale_feats)>3 else ''}) — using forward-fill + zero-fill. "
            f"Upstream data likely stale; check P1 downloads.",
        )
    # Forward-fill carries last known value through NaN tails (correct for log-
    # returns, ratios, indicators). fillna(0.0) is final safety for columns
    # that have NEVER had data (cold start).
    df[feature_cols] = df[feature_cols].ffill().fillna(0.0)
    n = len(df)
    if n < window + 100:
        return None

    i = n - 1
    row = df.iloc[i]

    # Staleness refusal: if the latest row used for inference is more than
    # horizon+2h behind the freshest bar in df_full, abort with a loud warning.
    # Prevents silent stale-signal bugs when upstream data sources are late.
    try:
        latest_raw_dt = pd.to_datetime(df_full.iloc[-1]['datetime'])
        latest_used_dt = pd.to_datetime(row['datetime'])
        lag_hours = (latest_raw_dt - latest_used_dt).total_seconds() / 3600.0
        if metrics_out is not None:
            metrics_out['inference_row_age_h'] = round(lag_hours, 2)
        if lag_hours > horizon + 2:
            print(f"  [!!] {asset_name} {horizon}h: inference row is {lag_hours:.1f}h stale (>horizon+2h) -- REFUSING to emit signal")
            return None
    except Exception:
        pass

    train_start = max(0, i - window)
    train = df.iloc[train_start:i]
    X_train = train[feature_cols]
    y_train = train['label'].values
    X_test = df.iloc[i:i+1][feature_cols]

    if len(np.unique(y_train)) < 2:
        return None

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    sw = get_decay_weights(len(y_train), gamma)
    _fit_start = _tm.time()
    votes, probas = [], []
    for model_name in model_names:
        try:
            model = ALL_MODELS[model_name]()
            model.fit(X_train_s, y_train, sample_weight=sw)
            votes.append(model.predict(X_test_s)[0])
            probas.append(model.predict_proba(X_test_s)[0][1])
        except Exception:
            continue
    if metrics_out is not None:
        metrics_out['model_fit_sec'] = round(_tm.time() - _fit_start, 2)

    if not votes:
        return None

    buy_votes = sum(votes)
    buy_ratio = buy_votes / len(votes)

    # Enhancement: ensemble disagreement filter
    enh_disagree = config.get('disagree_filter', False)
    enh_disagree_thresh = config.get('disagree_threshold', 0.75)
    if enh_disagree and max(buy_ratio, 1 - buy_ratio) < enh_disagree_thresh:
        signal = 'HOLD'
    elif buy_ratio > 0.5:
        signal = 'BUY'
    elif buy_ratio == 0:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    # Enhancement: funding rate regime gate
    enh_funding = config.get('funding_gate', False)
    enh_funding_thresh = config.get('funding_threshold', 0.001)
    if enh_funding and signal == 'BUY' and '_funding_rate' in df.columns:
        fr_val = df.iloc[i].get('_funding_rate', np.nan)
        if not np.isnan(fr_val) and fr_val > enh_funding_thresh:
            signal = 'HOLD'

    avg_proba = np.mean(probas)
    # Fix #3 (2026-04-24): store 2 decimals so pinned vs noise-around-same-int
    # is distinguishable. 85.3 >= 85 still holds so min_conf threshold unchanged.
    confidence = round(avg_proba * 100, 2) if signal != 'SELL' else round((1 - avg_proba) * 100, 2)

    # Use raw data (df_raw) for last 4h — df drops last `horizon` rows due to label shift
    last_4h = []
    src = df_raw if df_raw is not None and len(df_raw) >= 4 else df
    for j in range(max(0, len(src) - 4), len(src)):
        r = src.iloc[j]
        last_4h.append({
            'datetime': _to_local(r['datetime']).strftime('%H:%M') if hasattr(r['datetime'], 'strftime') else str(r['datetime']),
            'close': float(r['close']),
        })

    return {
        'asset': asset_name, 'signal': signal,
        'confidence': round(float(confidence), 1),
        'close': float(df_raw.iloc[-1]['close']) if df_raw is not None and len(df_raw) > 0 else float(row['close']),
        'buy_votes': int(buy_votes), 'total_votes': len(votes),
        'rsi': round(float(row.get('rsi_14h', 0)), 1),
        'datetime': _to_local(row['datetime']).strftime('%Y-%m-%d %H:%M') if hasattr(row['datetime'], 'strftime') else str(row['datetime']),
        'last_4h': last_4h,
        'model': config['best_combo'], 'window': config['best_window'],
        'feature_set': fs, 'diag_accuracy': config['accuracy'], 'horizon': horizon,
    }


def generate_regime_signal(asset, df_raw=None):
    """Generate signal using regime-aware horizon selection.

    1. Loads price data
    2. Detects regime (bull/bear)
    3. Picks horizon + confidence from regime config
    4. Generates signal at that horizon
    5. Returns signal with regime metadata

    Returns:
        dict with keys: signal, confidence, regime, active_horizon, active_config, price, etc.
    """
    from crypto_trading_system_ed import load_data

    if df_raw is None:
        df_raw = load_data(asset)
        if df_raw is None:
            return None

    # Detect regime
    regime, active_cfg = detect_regime(asset, df_raw)
    horizon = active_cfg.get('horizon', 6)
    min_conf = active_cfg.get('min_confidence', 85)

    # Load production model for this horizon
    config = load_best_config(asset, horizon=horizon)
    if config is None:
        print(f"  [!] No production model for {asset} {horizon}h (regime={regime})")
        return None

    # Generate signal
    result = generate_live_signal(asset, config=config, df_raw=df_raw)
    if result is None:
        return None

    # Attach regime metadata
    result['regime'] = regime
    result['active_horizon'] = horizon
    result['active_config'] = active_cfg

    return result

# ============================================================
# "BOTH AGREE" STRATEGY
# ============================================================
def compute_combined_signal(sig_short, sig_long, min_confidence=MIN_CONFIDENCE):
    h_s, h_l = HORIZON_SHORT, HORIZON_LONG
    if sig_short is None or sig_long is None:
        sig = sig_short or sig_long
        if sig and sig['confidence'] >= min_confidence:
            return sig['signal'], sig['confidence'], 'single_model'
        return 'HOLD', 50, 'missing_model'

    ss, cs = sig_short['signal'], sig_short['confidence']
    sl, cl = sig_long['signal'], sig_long['confidence']

    if ss == 'SELL' or sl == 'SELL':
        if ss == 'SELL' and sl == 'SELL':
            return 'SELL', max(cs, cl), 'both_sell'
        elif ss == 'SELL':
            return 'SELL', cs, f'{h_s}h_sell'
        else:
            return 'SELL', cl, f'{h_l}h_sell'

    if ss == 'BUY' and sl == 'BUY':
        if cs >= min_confidence and cl >= min_confidence:
            return 'BUY', (cs + cl) / 2, 'both_buy'
        low = f'{h_s}h' if cs < min_confidence else f'{h_l}h'
        return 'HOLD', min(cs, cl), f'low_conf_{low}'

    return 'HOLD', 50, 'disagree'


def run_once(asset_name, _stale_warning=False):
    print(f"\n{'='*60}")
    print(f"  LIVE SIGNAL [ED]: {asset_name}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    config_short = load_best_config(asset_name, horizon=HORIZON_SHORT)
    config_long = load_best_config(asset_name, horizon=HORIZON_LONG)
    if not config_short and not config_long:
        print("  ERROR: No Ed models found!")
        return None

    print(f"\n  Downloading latest {asset_name} data...")
    try:
        download_asset(asset_name, update_only=True)
    except Exception as e:
        print(f"  Download error: {e}")

    df_raw = load_data(asset_name)
    if df_raw is None:
        print("  ERROR: No data.")
        return None

    sig_short = sig_long = None
    if config_short:
        print(f"\n  --- {HORIZON_SHORT}h: {config_short['best_combo']} | w={config_short['best_window']}h ---")
        sig_short = generate_live_signal(asset_name, config_short, df_raw=df_raw)
        if sig_short:
            print(f"  {sig_short['signal']} ({sig_short['confidence']:.2f}%)")

    if config_long:
        print(f"\n  --- {HORIZON_LONG}h: {config_long['best_combo']} | w={config_long['best_window']}h ---")
        sig_long = generate_live_signal(asset_name, config_long, df_raw=df_raw)
        if sig_long:
            print(f"  {sig_long['signal']} ({sig_long['confidence']:.2f}%)")

    action, confidence, reason = compute_combined_signal(sig_short, sig_long)
    any_sig = sig_short or sig_long
    price = any_sig['close'] if any_sig else 0

    print(f"\n  {'='*50}")
    print(f"  >>> {action} ({confidence:.2f}%) -- {reason}")
    print(f"  {'='*50}")

    msg = _format_combined_message(asset_name, sig_short, sig_long, action, confidence, reason)
    if _stale_warning:
        msg += "\n\nData may be delayed"
    send_telegram(msg)

    return {'action': action, 'confidence': confidence, 'reason': reason,
            'price': price, 'sig_short': sig_short, 'sig_long': sig_long}


def _format_combined_message(asset_name, sig_short, sig_long, action, confidence, reason):
    any_sig = sig_short or sig_long
    price = any_sig['close'] if any_sig else 0
    price_str = f"${price:,.2f}" if price > 1000 else f"${price:.4f}"

    if action == 'BUY':
        action_line = f"<b>ACTION: BUY</b> ({confidence:.2f}%)"
    elif action == 'SELL':
        action_line = f"<b>ACTION: SELL</b> ({confidence:.2f}%)"
    else:
        action_line = f"<b>HOLD</b> -- no action"

    h_s, h_l = HORIZON_SHORT, HORIZON_LONG

    def _sig_str(sig, h):
        if not sig:
            return f"{h}h: N/A"
        return f"{h}h: {sig['signal']} ({sig['confidence']:.2f}%)"

    reason_map = {
        'both_buy': f'{h_s}h + {h_l}h both BUY', 'both_sell': f'{h_s}h + {h_l}h both SELL',
        f'{h_s}h_sell': f'{h_s}h triggered SELL', f'{h_l}h_sell': f'{h_l}h triggered SELL',
        'disagree': 'Models disagree', f'low_conf_{h_s}h': f'{h_s}h confidence too low',
        f'low_conf_{h_l}h': f'{h_l}h confidence too low', 'single_model': 'One model only',
        'missing_model': 'Model failed',
    }

    lines = [
        f"<b>[ED] {asset_name}</b>  {_sig_str(sig_short, h_s)} | {_sig_str(sig_long, h_l)}",
        "", action_line, f"Reason: {reason_map.get(reason, reason)}",
        "", f"<b>Price: {price_str}</b>",
    ]
    if any_sig:
        lines.append(f"RSI: {any_sig['rsi']}")
    if sig_short:
        lines.append(f"{h_s}h: {sig_short['model']} | w={sig_short['window']}h | {sig_short['diag_accuracy']:.1f}%")
    if sig_long:
        lines.append(f"{h_l}h: {sig_long['model']} | w={sig_long['window']}h | {sig_long['diag_accuracy']:.1f}%")
    lines.append(f"{any_sig['datetime'] if any_sig else ''}")
    return "\n".join(lines)


# ============================================================
# HOURLY LOOP
# ============================================================
def wait_for_fresh_candle(asset_name, expected_hour_utc, max_retries=30, retry_interval=10):
    for attempt in range(1, max_retries + 1):
        try:
            download_asset(asset_name, update_only=True)
        except Exception:
            time.sleep(retry_interval)
            continue
        df_raw = load_data(asset_name)
        if df_raw is None:
            time.sleep(retry_interval)
            continue
        latest_dt = pd.to_datetime(df_raw['datetime'].iloc[-1])
        if latest_dt.tzinfo is not None:
            latest_dt = latest_dt.tz_localize(None)
        if latest_dt.replace(minute=0, second=0, microsecond=0) >= expected_hour_utc:
            print(f"    Fresh candle (attempt {attempt})")
            return True
        print(f"    Attempt {attempt}/{max_retries}...")
        time.sleep(retry_interval)
    return False

def run_loop(asset_name):
    print(f"\n{'='*60}")
    print(f"  LIVE TRADER [ED]: {asset_name}")
    print(f"  Min confidence: {MIN_CONFIDENCE}%")
    print(f"{'='*60}")

    c_s = load_best_config(asset_name, horizon=HORIZON_SHORT)
    c_l = load_best_config(asset_name, horizon=HORIZON_LONG)
    if c_s: print(f"  {HORIZON_SHORT}h: {c_s['best_combo']} | {c_s['accuracy']:.1f}%")
    if c_l: print(f"  {HORIZON_LONG}h: {c_l['best_combo']} | {c_l['accuracy']:.1f}%")
    if not c_s and not c_l:
        print("  ERROR: No Ed models!")
        return

    send_telegram(f"<b>[ED] Live Trader Started</b>\n\nAsset: {asset_name}\nMin conf: {MIN_CONFIDENCE}%\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    cycle = 0
    while True:
        try:
            now = datetime.now()
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            wait_sec = (next_hour - now).total_seconds()
            if wait_sec > 5:
                print(f"\n  Waiting for {next_hour.strftime('%H:%M')} ({wait_sec/60:.0f} min)...")
                time.sleep(wait_sec)

            cycle += 1
            print(f"\n  --- Cycle {cycle} | {datetime.now().strftime('%H:%M:%S')} ---")

            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            expected = now_utc.replace(minute=0, second=0, microsecond=0)
            fresh = wait_for_fresh_candle(asset_name, expected)
            run_once(asset_name, _stale_warning=not fresh)

        except KeyboardInterrupt:
            print("\n  Stopped.")
            send_telegram(f"<b>[ED] Stopped</b>\nCycles: {cycle}")
            break
        except Exception as e:
            print(f"\n  ERROR: {e}")
            send_telegram(f"<b>[ED] Error</b>\n<code>{e}</code>")
            time.sleep(120)

# ============================================================
# TELEGRAM SETUP
# ============================================================
def setup_telegram():
    print("\n" + "=" * 60)
    print("  TELEGRAM SETUP")
    print("=" * 60)
    token = input("\n  Bot token: ").strip()
    if not token: return
    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx) as resp:
            data = json.loads(resp.read().decode())
        if data.get('ok') and data.get('result'):
            chat_id = str(data['result'][-1]['message']['chat']['id'])
            print(f"  Found chat_id: {chat_id}")
        else:
            chat_id = input("  Enter chat_id: ").strip()
    except Exception:
        chat_id = input("  Enter chat_id: ").strip()
    if not chat_id: return
    config = {'token': token, 'chat_id': chat_id}
    with open(TELEGRAM_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    TELEGRAM_CONFIG.update(config)
    send_telegram("<b>[ED] Telegram connected!</b>")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  CRYPTO LIVE TRADER [ED]")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    asset_name = 'BTC'
    loop_mode = False

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--asset' and i + 1 < len(args):
            asset_name = args[i + 1].upper()
        elif arg == '--loop':
            loop_mode = True
        elif arg == '--setup':
            setup_telegram()
            return

    if not os.path.exists(MODELS_CSV):
        print(f"\n  ERROR: No Ed models found! Run crypto_trading_system_ed.py Mode D first.")
        return

    if not TELEGRAM_CONFIG.get('token'):
        print("\n  Telegram not configured! Run --setup")
        if input("  Continue? (y/n): ").strip().lower() != 'y':
            return

    if len(args) == 0:
        print(f"\n  1. Run once\n  2. Loop (hourly)\n  3. Setup Telegram")
        ch = input("\n  Enter 1-3: ").strip()
        if ch == '3': setup_telegram(); return
        if ch == '2': loop_mode = True
        a = input(f"  Asset [{asset_name}]: ").strip().upper()
        if a in ['BTC','ETH','SOL','XRP','DOGE']: asset_name = a

    print(f"\n  Asset: {asset_name} | Min conf: {MIN_CONFIDENCE}%")

    if loop_mode:
        run_loop(asset_name)
    else:
        run_once(asset_name)

if __name__ == '__main__':
    main()
