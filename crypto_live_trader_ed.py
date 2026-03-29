"""
Crypto Live Trader (Doohan) — Signal Generation for Doohan Models
=============================================================
Same as crypto_live_trader.py but imports from crypto_trading_system_doohan
and reads models/crypto_doohan_v1_6_production.csv.

Usage:
  python crypto_live_trader_doohan.py                  # Run once (BTC default)
  python crypto_live_trader_doohan.py --loop           # Run every hour
  python crypto_live_trader_doohan.py --loop --asset BTC
  python crypto_live_trader_doohan.py --setup          # Setup Telegram
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
        tuple: (regime_str, active_config) where regime_str is 'bull' or 'bear'
               and active_config is {'horizon': int, 'min_confidence': int, 'max_position_usd': float}
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

    # Evaluate detector
    is_bull = _evaluate_detector(det_type, params, df_prices, asset)

    regime = 'bull' if is_bull else 'bear'
    active = asset_cfg.get(regime, asset_cfg.get('bull', {}))

    return regime, active


def _evaluate_detector(det_type, params, df, asset):
    """Evaluate a regime detector. Returns True for bull, False for bear."""
    try:
        if det_type == 'fixed':
            return params.get('regime', 'bull') == 'bull'

        elif det_type == 'sma_cross':
            fast = int(params.get('fast', 24))
            slow = int(params.get('slow', 100))
            if len(df) < slow + 10:
                return True  # not enough data, default bull
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

        else:
            print(f"  [!] Unknown regime detector type: {det_type}")
            return True  # default bull

    except Exception as e:
        print(f"  [!] Regime detection error: {e}")
        return True  # default bull


def _evaluate_pysr_detector(params, df, asset):
    """Evaluate a PySR-discovered regime formula."""
    model_file = params.get('model_file', '')
    expr_index = int(params.get('expression_index', 0))
    threshold = float(params.get('threshold', 0.5))

    model_path = os.path.join(os.path.dirname(__file__), model_file)
    if not os.path.exists(model_path):
        print(f"  [!] PySR regime model not found: {model_path}")
        return True

    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            pysr_data = json.load(f)

        expressions = pysr_data.get('expressions', [])
        if expr_index >= len(expressions):
            return True

        expr = expressions[expr_index]
        equation = expr.get('sympy_format', expr.get('equation', ''))
        feat_names = pysr_data.get('feature_names', [])

        # Build feature values from df
        from tools.pysr_discover_regime import build_regime_features
        features = build_regime_features(df)

        # Evaluate expression using sympy
        import sympy
        local_dict = {}
        for fname in feat_names:
            if fname in features.columns:
                val = features[fname].iloc[-1]
                if np.isfinite(val):
                    local_dict[fname] = val
                else:
                    local_dict[fname] = 0.0
            else:
                local_dict[fname] = 0.0

        result = float(sympy.sympify(equation).evalf(subs=local_dict))
        return result <= threshold  # <= threshold = bull, > threshold = bear

    except Exception as e:
        print(f"  [!] PySR regime eval error: {e}")
        return True

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
def generate_live_signal(asset_name, config, df_raw=None, verbose=True):
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
    feature_cols = [f for f in feature_list if f in all_cols]
    if not feature_cols:
        feature_cols = [f for f in FEATURE_SET_A if f in all_cols]

    df = df_full.dropna(subset=feature_cols + ['label']).reset_index(drop=True)
    n = len(df)
    if n < window + 100:
        return None

    i = n - 1
    row = df.iloc[i]
    train_start = max(0, i - window)
    train_end = max(train_start, i - horizon)  # embargo: match backtesting
    train = df.iloc[train_start:train_end]
    X_train = train[feature_cols]
    y_train = train['label'].values
    X_test = df.iloc[i:i+1][feature_cols]

    if len(np.unique(y_train)) < 2:
        return None

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    sw = get_decay_weights(len(y_train), gamma)
    votes, probas = [], []
    for model_name in model_names:
        try:
            model = ALL_MODELS[model_name]()
            model.fit(X_train_s, y_train, sample_weight=sw)
            votes.append(model.predict(X_test_s)[0])
            probas.append(model.predict_proba(X_test_s)[0][1])
        except Exception:
            continue

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
    confidence = round(avg_proba * 100) if signal != 'SELL' else round((1 - avg_proba) * 100)

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
    print(f"  LIVE SIGNAL [DOOHAN]: {asset_name}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    config_short = load_best_config(asset_name, horizon=HORIZON_SHORT)
    config_long = load_best_config(asset_name, horizon=HORIZON_LONG)
    if not config_short and not config_long:
        print("  ERROR: No Doohan models found!")
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
            print(f"  {sig_short['signal']} ({sig_short['confidence']:.0f}%)")

    if config_long:
        print(f"\n  --- {HORIZON_LONG}h: {config_long['best_combo']} | w={config_long['best_window']}h ---")
        sig_long = generate_live_signal(asset_name, config_long, df_raw=df_raw)
        if sig_long:
            print(f"  {sig_long['signal']} ({sig_long['confidence']:.0f}%)")

    action, confidence, reason = compute_combined_signal(sig_short, sig_long)
    any_sig = sig_short or sig_long
    price = any_sig['close'] if any_sig else 0

    print(f"\n  {'='*50}")
    print(f"  >>> {action} ({confidence:.0f}%) -- {reason}")
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
        action_line = f"<b>ACTION: BUY</b> ({confidence:.0f}%)"
    elif action == 'SELL':
        action_line = f"<b>ACTION: SELL</b> ({confidence:.0f}%)"
    else:
        action_line = f"<b>HOLD</b> -- no action"

    h_s, h_l = HORIZON_SHORT, HORIZON_LONG

    def _sig_str(sig, h):
        if not sig:
            return f"{h}h: N/A"
        return f"{h}h: {sig['signal']} ({sig['confidence']:.0f}%)"

    reason_map = {
        'both_buy': f'{h_s}h + {h_l}h both BUY', 'both_sell': f'{h_s}h + {h_l}h both SELL',
        f'{h_s}h_sell': f'{h_s}h triggered SELL', f'{h_l}h_sell': f'{h_l}h triggered SELL',
        'disagree': 'Models disagree', f'low_conf_{h_s}h': f'{h_s}h confidence too low',
        f'low_conf_{h_l}h': f'{h_l}h confidence too low', 'single_model': 'One model only',
        'missing_model': 'Model failed',
    }

    lines = [
        f"<b>[DOOHAN] {asset_name}</b>  {_sig_str(sig_short, h_s)} | {_sig_str(sig_long, h_l)}",
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
    print(f"  LIVE TRADER [DOOHAN]: {asset_name}")
    print(f"  Min confidence: {MIN_CONFIDENCE}%")
    print(f"{'='*60}")

    c_s = load_best_config(asset_name, horizon=HORIZON_SHORT)
    c_l = load_best_config(asset_name, horizon=HORIZON_LONG)
    if c_s: print(f"  {HORIZON_SHORT}h: {c_s['best_combo']} | {c_s['accuracy']:.1f}%")
    if c_l: print(f"  {HORIZON_LONG}h: {c_l['best_combo']} | {c_l['accuracy']:.1f}%")
    if not c_s and not c_l:
        print("  ERROR: No Doohan models!")
        return

    send_telegram(f"<b>[DOOHAN] Live Trader Started</b>\n\nAsset: {asset_name}\nMin conf: {MIN_CONFIDENCE}%\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

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
            send_telegram(f"<b>[DOOHAN] Stopped</b>\nCycles: {cycle}")
            break
        except Exception as e:
            print(f"\n  ERROR: {e}")
            send_telegram(f"<b>[DOOHAN] Error</b>\n<code>{e}</code>")
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
    send_telegram("<b>[DOOHAN] Telegram connected!</b>")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  CRYPTO LIVE TRADER [DOOHAN]")
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
        print(f"\n  ERROR: No Doohan models found! Run crypto_trading_system_doohan.py Mode D first.")
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
