"""
Crypto Live Trader — "Both Agree" Strategy + Telegram
=======================================================
Uses the proven "both agree" strategy from backtesting:
  - BUY only when 4h AND 8h models both say BUY
  - SELL when EITHER model says SELL
  - Minimum confidence: 75% on both models
  - Results: 91% win rate, +103% alpha over 1 month

Usage:
  python crypto_live_trader.py                  # Run once (BTC default)
  python crypto_live_trader.py --loop           # Run every hour
  python crypto_live_trader.py --loop --asset BTC
  python crypto_live_trader.py --setup          # Setup Telegram
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
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

from crypto_trading_system import (
    ASSETS, FEATURE_SET_A, FEATURE_SET_B,
    PREDICTION_HORIZON, ALL_MODELS,
    download_asset, load_data, build_all_features,
)
from sklearn.preprocessing import StandardScaler

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
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            if result.get('ok'):
                print("  ✓ Telegram sent")
                return True
            print(f"  [!] Telegram error: {result}")
            return False
    except Exception as e:
        print(f"  [!] Telegram error: {e}")
        return False

# ============================================================
# LOAD MODEL CONFIG
# ============================================================
def load_best_config(asset_name, horizon=None):
    csv_path = 'models/crypto_hourly_best_models.csv'
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if 'horizon' not in df.columns:
        df['horizon'] = 4
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
        'horizon': int(row.get('horizon', 4)), 'optimal_features': str(opt),
    }

# ============================================================
# GENERATE SINGLE SIGNAL
# ============================================================
def generate_live_signal(asset_name, config, df_raw=None):
    model_names = config['models'].split('+')
    window = config['best_window']
    fs = config.get('feature_set', 'A')
    horizon = config.get('horizon', 4)
    opt_features = config.get('optimal_features', '')

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

    df_full, all_cols = build_all_features(df_raw, asset_name=asset_name, horizon=horizon)
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
    train = df.iloc[train_start:i]
    X_train = train[feature_cols]
    y_train = train['label'].values
    X_test = df.iloc[i:i+1][feature_cols]

    if len(np.unique(y_train)) < 2:
        return None

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    votes, probas = [], []
    for model_name in model_names:
        try:
            model = ALL_MODELS[model_name]()
            model.fit(X_train_s, y_train)
            votes.append(model.predict(X_test_s)[0])
            probas.append(model.predict_proba(X_test_s)[0][1])
        except Exception:
            continue

    if not votes:
        return None

    buy_votes = sum(votes)
    buy_ratio = buy_votes / len(votes)
    if buy_ratio > 0.5:
        signal = 'BUY'
    elif buy_ratio == 0:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    avg_proba = np.mean(probas)
    confidence = avg_proba * 100 if signal != 'SELL' else (1 - avg_proba) * 100

    last_4h = []
    for j in range(max(0, n - 4), n):
        r = df.iloc[j]
        last_4h.append({
            'datetime': _to_local(r['datetime']).strftime('%H:%M') if hasattr(r['datetime'], 'strftime') else str(r['datetime']),
            'close': float(r['close']),
        })

    return {
        'asset': asset_name, 'signal': signal,
        'confidence': round(float(confidence), 1),
        'close': float(row['close']),
        'buy_votes': int(buy_votes), 'total_votes': len(votes),
        'rsi': round(float(row.get('rsi_14h', 0)), 1),
        'datetime': _to_local(row['datetime']).strftime('%Y-%m-%d %H:%M') if hasattr(row['datetime'], 'strftime') else str(row['datetime']),
        'last_4h': last_4h,
        'model': config['best_combo'], 'window': config['best_window'],
        'feature_set': fs, 'diag_accuracy': config['accuracy'], 'horizon': horizon,
    }

# ============================================================
# "BOTH AGREE" STRATEGY
# ============================================================
def compute_combined_signal(sig_4h, sig_8h, min_confidence=MIN_CONFIDENCE):
    """
    BUY: both 4h AND 8h say BUY with confidence >= min_confidence
    SELL: EITHER says SELL
    HOLD: everything else
    """
    if sig_4h is None or sig_8h is None:
        sig = sig_4h or sig_8h
        if sig and sig['confidence'] >= min_confidence:
            return sig['signal'], sig['confidence'], 'single_model'
        return 'HOLD', 50, 'missing_model'

    s4, c4 = sig_4h['signal'], sig_4h['confidence']
    s8, c8 = sig_8h['signal'], sig_8h['confidence']

    # SELL if either says SELL
    if s4 == 'SELL' or s8 == 'SELL':
        if s4 == 'SELL' and s8 == 'SELL':
            return 'SELL', max(c4, c8), 'both_sell'
        elif s4 == 'SELL':
            return 'SELL', c4, '4h_sell'
        else:
            return 'SELL', c8, '8h_sell'

    # BUY only if BOTH agree AND both above threshold
    if s4 == 'BUY' and s8 == 'BUY':
        if c4 >= min_confidence and c8 >= min_confidence:
            return 'BUY', (c4 + c8) / 2, 'both_buy'
        low = '4h' if c4 < min_confidence else '8h'
        return 'HOLD', min(c4, c8), f'low_conf_{low}'

    return 'HOLD', 50, 'disagree'


def run_once(asset_name, _stale_warning=False):
    """Generate signals from both models and apply 'both agree' strategy."""
    print(f"\n{'='*60}")
    print(f"  LIVE SIGNAL: {asset_name} (Both Agree Strategy)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    config_4h = load_best_config(asset_name, horizon=4)
    config_8h = load_best_config(asset_name, horizon=8)
    if not config_4h and not config_8h:
        print("  ERROR: No models found!")
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

    sig_4h = sig_8h = None
    if config_4h:
        print(f"\n  --- 4h: {config_4h['best_combo']} | w={config_4h['best_window']}h ---")
        sig_4h = generate_live_signal(asset_name, config_4h, df_raw=df_raw)
        if sig_4h:
            print(f"  {sig_4h['signal']} ({sig_4h['confidence']:.0f}%)")

    if config_8h:
        print(f"\n  --- 8h: {config_8h['best_combo']} | w={config_8h['best_window']}h ---")
        sig_8h = generate_live_signal(asset_name, config_8h, df_raw=df_raw)
        if sig_8h:
            print(f"  {sig_8h['signal']} ({sig_8h['confidence']:.0f}%)")

    action, confidence, reason = compute_combined_signal(sig_4h, sig_8h)
    any_sig = sig_4h or sig_8h
    price = any_sig['close'] if any_sig else 0

    print(f"\n  {'='*50}")
    print(f"  >>> {action} ({confidence:.0f}%) — {reason}")
    print(f"  {'='*50}")

    msg = _format_combined_message(asset_name, sig_4h, sig_8h, action, confidence, reason)
    if _stale_warning:
        msg += "\n\n⚠️ <i>Data may be delayed</i>"
    send_telegram(msg)

    return {'action': action, 'confidence': confidence, 'reason': reason,
            'price': price, 'sig_4h': sig_4h, 'sig_8h': sig_8h}


def _format_combined_message(asset_name, sig_4h, sig_8h, action, confidence, reason):
    any_sig = sig_4h or sig_8h
    price = any_sig['close'] if any_sig else 0
    price_str = f"${price:,.2f}" if price > 1000 else f"${price:.4f}"

    if action == 'BUY':
        action_line = f"✅ <b>ACTION: BUY</b> ({confidence:.0f}%)"
    elif action == 'SELL':
        action_line = f"🚨 <b>ACTION: SELL</b> ({confidence:.0f}%)"
    else:
        action_line = f"⏸ <b>HOLD</b> — no action"

    def _sig_str(sig, h):
        if not sig:
            return f"{h}h: N/A"
        e = '🟢' if sig['signal'] == 'BUY' else '🔴' if sig['signal'] == 'SELL' else '🟡'
        return f"{e} {h}h: {sig['signal']} ({sig['confidence']:.0f}%)"

    reason_map = {
        'both_buy': '4h + 8h both BUY', 'both_sell': '4h + 8h both SELL',
        '4h_sell': '4h triggered SELL', '8h_sell': '8h triggered SELL',
        'disagree': 'Models disagree', 'low_conf_4h': '4h confidence too low',
        'low_conf_8h': '8h confidence too low', 'single_model': 'One model only',
        'missing_model': 'Model failed',
    }

    lines = [
        f"<b>{asset_name}</b>  {_sig_str(sig_4h, 4)} | {_sig_str(sig_8h, 8)}",
        "", action_line, f"📋 Reason: {reason_map.get(reason, reason)}",
        "", f"💰 <b>Price: {price_str}</b>",
    ]
    if any_sig:
        lines.append(f"📈 RSI: {any_sig['rsi']}")
    if any_sig and any_sig.get('last_4h'):
        lines.extend(["", "📊 <b>Last 4 hours:</b>"])
        for i, h in enumerate(any_sig['last_4h']):
            hp_str = f"${h['close']:,.2f}" if h['close'] > 1000 else f"${h['close']:.4f}"
            marker = " ← now" if i == len(any_sig['last_4h']) - 1 else ""
            lines.append(f"  <code>{h['datetime']}  {hp_str}{marker}</code>")
    lines.append("")
    if sig_4h:
        lines.append(f"🤖 4h: {sig_4h['model']} | w={sig_4h['window']}h | {sig_4h['diag_accuracy']:.1f}%")
    if sig_8h:
        lines.append(f"🤖 8h: {sig_8h['model']} | w={sig_8h['window']}h | {sig_8h['diag_accuracy']:.1f}%")
    lines.append(f"⏰ {any_sig['datetime'] if any_sig else ''}")
    lines.append(f"📊 Min conf: {MIN_CONFIDENCE}%")
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
            print(f"    ✓ Fresh candle (attempt {attempt})")
            return True
        print(f"    Attempt {attempt}/{max_retries}...")
        time.sleep(retry_interval)
    return False

def run_loop(asset_name):
    print(f"\n{'='*60}")
    print(f"  LIVE TRADER: {asset_name} — Both Agree")
    print(f"  BUY when 4h+8h agree | SELL when either says SELL")
    print(f"  Min confidence: {MIN_CONFIDENCE}%")
    print(f"{'='*60}")

    c4 = load_best_config(asset_name, horizon=4)
    c8 = load_best_config(asset_name, horizon=8)
    if c4: print(f"  ✓ 4h: {c4['best_combo']} | {c4['accuracy']:.1f}%")
    if c8: print(f"  ✓ 8h: {c8['best_combo']} | {c8['accuracy']:.1f}%")
    if not c4 and not c8:
        print("  ERROR: No models!")
        return

    send_telegram(f"🚀 <b>Live Trader Started</b>\n\nAsset: {asset_name}\nStrategy: Both Agree\nMin conf: {MIN_CONFIDENCE}%\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

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
            send_telegram(f"🛑 <b>Stopped</b>\nCycles: {cycle}")
            break
        except Exception as e:
            print(f"\n  ERROR: {e}")
            send_telegram(f"⚠️ <b>Error</b>\n<code>{e}</code>")
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
        with urllib.request.urlopen(req, timeout=10) as resp:
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
    send_telegram("✅ <b>Telegram connected!</b>")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  CRYPTO LIVE TRADER — BOTH AGREE STRATEGY")
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

    if not os.path.exists('models/crypto_hourly_best_models.csv'):
        print("\n  ERROR: No models found!")
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

    print(f"\n  Asset: {asset_name} | Strategy: Both Agree | Min conf: {MIN_CONFIDENCE}%")

    if loop_mode:
        run_loop(asset_name)
    else:
        run_once(asset_name)

if __name__ == '__main__':
    main()
