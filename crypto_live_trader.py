"""
Crypto Live Trader — Telegram Notifications
=============================================
Lightweight live signal generator for a single crypto asset.
Downloads latest hourly data from Binance, generates ML signal,
sends Telegram notification with BUY/HOLD/SELL + last 4h prices.

Setup:
  1. Create Telegram bot: message @BotFather → /newbot → copy token
  2. Get chat_id: message your bot, then visit:
     https://api.telegram.org/bot<TOKEN>/getUpdates
  3. Set environment variables or edit TELEGRAM_CONFIG below:
     set TELEGRAM_TOKEN=your_token
     set TELEGRAM_CHAT_ID=your_chat_id

Usage:
  python crypto_live_trader.py                  # Run once (BTC default)
  python crypto_live_trader.py --asset ETH      # Run once for ETH
  python crypto_live_trader.py --loop           # Run every hour
  python crypto_live_trader.py --loop --asset SOL
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

# Local timezone detection
try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo('Europe/Zurich')
except ImportError:
    LOCAL_TZ = None


def _to_local(dt):
    """Convert a UTC datetime to local time (Europe/Zurich)."""
    if dt is None or not hasattr(dt, 'strftime'):
        return dt
    try:
        if LOCAL_TZ:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(LOCAL_TZ)
        else:
            # Fallback: assume UTC+1 (CET)
            if dt.tzinfo is None:
                return dt + timedelta(hours=1)
            return dt.astimezone(timezone(timedelta(hours=1)))
    except Exception:
        return dt

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')

# Import from main trading system
from crypto_trading_system import (
    ASSETS, FEATURE_SET_A, FEATURE_SET_B,
    PREDICTION_HORIZON, ALL_MODELS,
    download_asset, load_data, build_all_features,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# TELEGRAM CONFIG
# ============================================================
TELEGRAM_CONFIG = {
    'token': os.environ.get('TELEGRAM_TOKEN', ''),       # Your bot token
    'chat_id': os.environ.get('TELEGRAM_CHAT_ID', ''),   # Your chat ID
}

# Override with config file if it exists
TELEGRAM_CONFIG_FILE = 'telegram_config.json'
if os.path.exists(TELEGRAM_CONFIG_FILE):
    with open(TELEGRAM_CONFIG_FILE) as f:
        TELEGRAM_CONFIG.update(json.load(f))


# ============================================================
# TELEGRAM SENDER
# ============================================================
def send_telegram(message, parse_mode='HTML'):
    """Send a message via Telegram bot."""
    token = TELEGRAM_CONFIG.get('token', '')
    chat_id = TELEGRAM_CONFIG.get('chat_id', '')

    if not token or not chat_id:
        print("  [!] Telegram not configured. Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID.")
        print("  [!] Or create telegram_config.json with {\"token\": \"...\", \"chat_id\": \"...\"}")
        print(f"\n  Message that would have been sent:\n{message}")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        'chat_id': chat_id,
        'text': message,
        'parse_mode': parse_mode,
    }).encode('utf-8')

    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            if result.get('ok'):
                print("  ✓ Telegram sent")
                return True
            else:
                print(f"  [!] Telegram error: {result}")
                return False
    except urllib.error.URLError as e:
        print(f"  [!] Telegram send failed: {e}")
        return False
    except Exception as e:
        print(f"  [!] Telegram error: {e}")
        return False


# ============================================================
# LOAD BEST MODEL CONFIG
# ============================================================
def load_best_config(asset_name, horizon=None):
    """Load the best model config for an asset from CSV. Optionally filter by horizon."""
    csv_path = 'crypto_hourly_best_models.csv'
    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found!")
        print("  Run crypto_trading_system.py Mode A first to find best models.")
        return None

    df = pd.read_csv(csv_path)
    if 'horizon' not in df.columns:
        df['horizon'] = 4  # legacy = 4h

    match = df[df['coin'] == asset_name]
    if horizon is not None:
        match = match[match['horizon'] == horizon]

    if match.empty:
        h_label = f" ({horizon}h)" if horizon else ""
        print(f"  No saved model for {asset_name}{h_label}.")
        return None

    row = match.iloc[0]
    config = {
        'coin': row['coin'],
        'models': row['models'],
        'best_combo': row['best_combo'],
        'best_window': int(row['best_window']),
        'accuracy': row['accuracy'],
        'feature_set': row.get('feature_set', 'A'),
        'horizon': int(row.get('horizon', 4)),
        'optimal_features': row.get('optimal_features', ''),
    }
    return config


# ============================================================
# GENERATE SINGLE SIGNAL (latest candle only)
# ============================================================
def generate_live_signal(asset_name, config):
    """
    Download latest data, build features, generate signal for latest candle.
    Returns signal dict or None.
    """
    model_names = config['models'].split('+')
    window = config['best_window']
    fs = config.get('feature_set', 'A')
    horizon = config.get('horizon', 4)
    opt_features = config.get('optimal_features', '')

    # Determine feature list
    if fs in ('D', 'E2', 'E3') and opt_features and pd.notna(opt_features) and str(opt_features).strip():
        feature_list = str(opt_features).split(',')
    elif fs == 'B':
        feature_list = list(FEATURE_SET_B)
    else:
        feature_list = list(FEATURE_SET_A)

    # Download latest data
    print(f"\n  Downloading latest {asset_name} data...")
    try:
        download_asset(asset_name, update_only=True)
    except Exception as e:
        print(f"  ERROR downloading {asset_name}: {e}")
        return None

    # Load data
    df_raw = load_data(asset_name)
    if df_raw is None:
        return None

    # Build all features with correct horizon
    print(f"  Building features (horizon={horizon}h)...")
    df_full, all_cols = build_all_features(df_raw, asset_name=asset_name, horizon=horizon)

    # Filter to active feature set
    feature_cols = [f for f in feature_list if f in all_cols]
    missing = [f for f in feature_list if f not in all_cols]
    if missing:
        print(f"  WARNING: Missing features: {missing}")
    print(f"  Using: Set {fs} ({len(feature_cols)} features) | {horizon}h horizon")

    if not feature_cols:
        print("  ERROR: No valid features!")
        return None

    # Drop NaN
    df = df_full.dropna(subset=feature_cols + ['label']).reset_index(drop=True)
    n = len(df)

    if n < window + 100:
        print(f"  ERROR: Not enough data ({n} rows, need {window + 100}+)")
        return None

    # Get latest row
    i = n - 1
    row = df.iloc[i]

    # Train on last `window` hours
    train_start = max(0, i - window)
    train = df.iloc[train_start:i]
    X_train = train[feature_cols]
    y_train = train['label'].values
    X_test = df.iloc[i:i+1][feature_cols]

    if len(np.unique(y_train)) < 2:
        print("  ERROR: Training data has only one class")
        return None

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    # Predict
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
        except Exception as e:
            print(f"  WARNING: Model {model_name} failed: {e}")
            continue

    if not votes:
        print("  ERROR: All models failed")
        return None

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
    confidence = avg_proba * 100 if signal != 'SELL' else (1 - avg_proba) * 100

    # Get last 4 hours of prices
    last_4h = []
    for j in range(max(0, n - 4), n):
        r = df.iloc[j]
        last_4h.append({
            'datetime': _to_local(r['datetime']).strftime('%H:%M') if hasattr(r['datetime'], 'strftime') else str(r['datetime']),
            'close': float(r['close']),
        })

    result = {
        'asset': asset_name,
        'signal': signal,
        'confidence': round(float(confidence), 1),
        'close': float(row['close']),
        'buy_votes': int(buy_votes),
        'total_votes': int(total_votes),
        'rsi': round(float(row.get('rsi_14h', 0)), 1),
        'datetime': _to_local(row['datetime']).strftime('%Y-%m-%d %H:%M') if hasattr(row['datetime'], 'strftime') else str(row['datetime']),
        'last_4h': last_4h,
        'model': config['best_combo'],
        'window': config['best_window'],
        'feature_set': fs,
        'diag_accuracy': config['accuracy'],
        'horizon': horizon,
    }

    return result


# ============================================================
# FORMAT TELEGRAM MESSAGE
# ============================================================
def format_telegram_message(sig):
    """Format signal as a Telegram HTML message."""
    # Signal emoji
    if sig['signal'] == 'BUY':
        emoji = '🟢'
        action_text = 'BUY'
    elif sig['signal'] == 'SELL':
        emoji = '🔴'
        action_text = 'SELL'
    else:
        emoji = '🟡'
        action_text = 'HOLD'

    # Price formatting
    price = sig['close']
    if price > 1000:
        price_str = f"${price:,.2f}"
    elif price > 1:
        price_str = f"${price:.4f}"
    else:
        price_str = f"${price:.6f}"

    # Build message
    lines = [
        f"{emoji} <b>{sig['asset']} — {action_text} ({sig['confidence']:.0f}%)</b>",
        "",
        f"💰 <b>Price: {price_str}</b>",
        f"📈 RSI: {sig['rsi']}  |  Votes: {sig['buy_votes']}/{sig['total_votes']}",
        "",
        "📊 <b>Last 4 hours:</b>",
    ]

    for i, h in enumerate(sig['last_4h']):
        hp = h['close']
        if hp > 1000:
            hp_str = f"${hp:,.2f}"
        elif hp > 1:
            hp_str = f"${hp:.4f}"
        else:
            hp_str = f"${hp:.6f}"

        marker = " ← latest" if i == len(sig['last_4h']) - 1 else ""
        lines.append(f"  <code>{h['datetime']}  {hp_str}{marker}</code>")

    h = sig.get('horizon', 4)
    lines.extend([
        "",
        f"🤖 Model: {sig['model']} | w={sig['window']}h | Set {sig['feature_set']} | {h}h ahead | {sig['diag_accuracy']:.1f}% diag",
        f"⏰ {sig['datetime']}",
    ])

    return "\n".join(lines)


# ============================================================
# RUN ONCE
# ============================================================
def run_once(asset_name, _stale_warning=False):
    """Download latest data, generate signals for all available horizons, send Telegram."""
    print(f"\n{'='*60}")
    print(f"  LIVE SIGNAL: {asset_name}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Load configs for all available horizons
    configs = {}
    for h in [1, 4]:
        c = load_best_config(asset_name, horizon=h)
        if c is not None:
            configs[h] = c

    if not configs:
        # Fallback: try loading without horizon filter (legacy CSV)
        c = load_best_config(asset_name)
        if c:
            configs[c.get('horizon', 4)] = c

    if not configs:
        print("  No model configs found. Run Mode A first.")
        return None

    # Generate signal for each horizon
    signals = {}
    for h, config in sorted(configs.items()):
        print(f"\n  --- {h}h horizon ---")
        print(f"  Model: {config['best_combo']} | w={config['best_window']}h | "
              f"Set {config.get('feature_set', 'A')} | {config['accuracy']:.1f}%")

        sig = generate_live_signal(asset_name, config)
        if sig:
            signals[h] = sig
            print(f"  {sig['signal']} ({sig['confidence']:.0f}%) | ${sig['close']:,.2f}")
        else:
            print(f"  Failed to generate {h}h signal.")

    if not signals:
        send_telegram(f"⚠️ <b>{asset_name}</b> signal generation failed at {datetime.now().strftime('%H:%M')}")
        return None

    # Print summary
    print(f"\n  {'='*50}")
    for h, sig in sorted(signals.items()):
        print(f"  [{h}h] {sig['signal']} ({sig['confidence']:.0f}%) | ${sig['close']:,.2f}")
    print(f"  {'='*50}")

    # Build combined Telegram message
    if len(signals) > 1:
        msg = _format_multi_horizon_message(asset_name, signals)
    else:
        h, sig = next(iter(signals.items()))
        msg = format_telegram_message(sig)

    if _stale_warning:
        msg += "\n\n⚠️ <i>Data may be delayed (candle not confirmed)</i>"

    print(f"\n  Sending Telegram notification...")
    send_telegram(msg)

    # Return the shortest horizon signal (most actionable)
    return signals[min(signals.keys())]


def _format_multi_horizon_message(asset_name, signals):
    """Format combined Telegram message for multiple horizons."""
    # Use any signal for price/time info
    any_sig = next(iter(signals.values()))
    price = any_sig['close']
    if price > 1000:
        price_str = f"${price:,.2f}"
    elif price > 1:
        price_str = f"${price:.4f}"
    else:
        price_str = f"${price:.6f}"

    # Header with both signals
    signal_parts = []
    for h in sorted(signals.keys()):
        sig = signals[h]
        if sig['signal'] == 'BUY':
            emoji = '🟢'
        elif sig['signal'] == 'SELL':
            emoji = '🔴'
        else:
            emoji = '🟡'
        signal_parts.append(f"{emoji} {h}h: {sig['signal']} ({sig['confidence']:.0f}%)")

    lines = [
        f"<b>{asset_name}</b>  {' | '.join(signal_parts)}",
        "",
        f"💰 <b>Price: {price_str}</b>",
    ]

    # Add RSI from shortest horizon
    short = signals[min(signals.keys())]
    lines.append(f"📈 RSI: {short['rsi']}")

    # Last 4 hours
    lines.extend(["", "📊 <b>Last 4 hours:</b>"])
    for i, h_data in enumerate(short['last_4h']):
        hp = h_data['close']
        if hp > 1000:
            hp_str = f"${hp:,.2f}"
        elif hp > 1:
            hp_str = f"${hp:.4f}"
        else:
            hp_str = f"${hp:.6f}"
        marker = " ← latest" if i == len(short['last_4h']) - 1 else ""
        lines.append(f"  <code>{h_data['datetime']}  {hp_str}{marker}</code>")

    # Model info per horizon
    lines.append("")
    for h in sorted(signals.keys()):
        sig = signals[h]
        lines.append(f"🤖 {h}h: {sig['model']} | w={sig['window']}h | Set {sig['feature_set']} | {sig['diag_accuracy']:.1f}%")

    lines.append(f"⏰ {short['datetime']}")

    return "\n".join(lines)


# ============================================================
# HOURLY LOOP
# ============================================================
def wait_for_fresh_candle(asset_name, expected_hour_utc, max_retries=30, retry_interval=10):
    """
    Keep downloading data until the latest candle matches the expected hour.
    Retries every retry_interval seconds, up to max_retries times.
    Returns the signal dict once fresh data is confirmed, or None.
    """
    for attempt in range(1, max_retries + 1):
        # Download latest data
        try:
            download_asset(asset_name, update_only=True)
        except Exception as e:
            print(f"    Download error: {e}")
            time.sleep(retry_interval)
            continue

        df_raw = load_data(asset_name)
        if df_raw is None:
            time.sleep(retry_interval)
            continue

        # Check latest candle timestamp
        latest_dt = pd.to_datetime(df_raw['datetime'].iloc[-1])
        if latest_dt.tzinfo is not None:
            latest_dt = latest_dt.tz_localize(None)

        latest_hour_utc = latest_dt.replace(minute=0, second=0, microsecond=0)

        if latest_hour_utc >= expected_hour_utc:
            print(f"    ✓ Fresh candle confirmed (attempt {attempt}): {latest_dt}")
            return True
        else:
            print(f"    Attempt {attempt}/{max_retries}: latest={latest_dt}, "
                  f"waiting for {expected_hour_utc}... (retry in {retry_interval}s)")
            time.sleep(retry_interval)

    print(f"    ✗ Timed out after {max_retries} attempts. Using latest available data.")
    return False


def run_loop(asset_name, interval_minutes=60):
    """Run signal generation every hour, on the hour. Retries until fresh candle arrives."""
    print(f"\n{'='*60}")
    print(f"  LIVE TRADER LOOP: {asset_name}")
    print(f"  Triggers at :00, retries until fresh candle arrives")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}")

    # Show which horizons are available
    available_horizons = []
    for h in [1, 4]:
        c = load_best_config(asset_name, horizon=h)
        if c is not None:
            available_horizons.append(h)
            print(f"  ✓ {h}h model: {c['best_combo']} | w={c['best_window']}h | "
                  f"Set {c.get('feature_set', 'A')} | {c['accuracy']:.1f}%")
    if not available_horizons:
        # Fallback: legacy config
        c = load_best_config(asset_name)
        if c:
            available_horizons.append(c.get('horizon', 4))
            print(f"  ✓ {c.get('horizon', 4)}h model (legacy): {c['best_combo']}")
        else:
            print("  ERROR: No model configs found. Run Mode A/D first.")
            return

    horizon_str = ' + '.join(f"{h}h" for h in available_horizons)

    # Startup notification
    send_telegram(
        f"🚀 <b>Live Trader Started</b>\n\n"
        f"Asset: {asset_name}\n"
        f"Horizons: {horizon_str}\n"
        f"Interval: every hour (on the hour)\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    cycle = 0
    while True:
        try:
            # Wait until the top of the next hour
            now = datetime.now()
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            wait_seconds = (next_hour - now).total_seconds()

            if wait_seconds > 5:  # Don't wait if we're already at :00
                print(f"\n  Waiting for {next_hour.strftime('%H:%M:%S')} "
                      f"({wait_seconds/60:.0f} min)...")
                time.sleep(wait_seconds)

            cycle += 1
            print(f"\n  --- Cycle {cycle} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

            # Expected candle: current hour in UTC
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            expected_hour = now_utc.replace(minute=0, second=0, microsecond=0)

            # Wait for fresh candle (retry until it appears)
            print(f"  Waiting for {asset_name} candle @ {expected_hour} UTC...")
            fresh = wait_for_fresh_candle(asset_name, expected_hour,
                                          max_retries=30, retry_interval=10)

            if not fresh:
                print("  ⚠ Using stale data (candle not yet updated)")

            # Generate signals for ALL available horizons via run_once
            sig = run_once(asset_name, _stale_warning=not fresh)

            if sig is None:
                print("  Signal generation failed.")
                continue

        except KeyboardInterrupt:
            print("\n\n  Stopped by user.")
            send_telegram(f"🛑 <b>Live Trader Stopped</b>\nCycles: {cycle}")
            break
        except Exception as e:
            print(f"\n  ERROR in cycle {cycle}: {e}")
            send_telegram(f"⚠️ <b>Error in cycle {cycle}</b>\n<code>{e}</code>")
            # Wait 2 min before retrying
            time.sleep(120)


# ============================================================
# TELEGRAM SETUP HELPER
# ============================================================
def setup_telegram():
    """Interactive Telegram setup wizard."""
    print("\n" + "=" * 60)
    print("  TELEGRAM SETUP")
    print("=" * 60)
    print("\n  Step 1: Open Telegram, search for @BotFather")
    print("  Step 2: Send /newbot, follow prompts, copy the token")
    print("  Step 3: Open your new bot and send /start")
    print("  Step 4: Enter your token below\n")

    token = input("  Bot token: ").strip()
    if not token:
        print("  Cancelled.")
        return

    # Get chat_id
    print("\n  Fetching your chat_id...")
    print("  (Make sure you've sent /start to your bot first!)")
    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        if data.get('ok') and data.get('result'):
            chat_id = str(data['result'][-1]['message']['chat']['id'])
            print(f"  Found chat_id: {chat_id}")
        else:
            print("  Could not find chat_id. Send /start to your bot and try again.")
            chat_id = input("  Enter chat_id manually: ").strip()
    except Exception as e:
        print(f"  Error: {e}")
        chat_id = input("  Enter chat_id manually: ").strip()

    if not chat_id:
        print("  Cancelled.")
        return

    # Save config
    config = {'token': token, 'chat_id': chat_id}
    with open(TELEGRAM_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Saved to {TELEGRAM_CONFIG_FILE}")

    # Test
    print("  Sending test message...")
    TELEGRAM_CONFIG.update(config)
    ok = send_telegram("✅ <b>Telegram setup complete!</b>\nYour live trader is connected.")
    if ok:
        print("  Check your phone!")
    else:
        print("  Test failed. Check token and chat_id.")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  CRYPTO LIVE TRADER — TELEGRAM NOTIFICATIONS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Parse CLI args
    asset_name = 'BTC'  # default
    loop_mode = False
    setup_mode = False

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--asset' and i + 1 < len(args):
            asset_name = args[i + 1].upper()
        elif arg == '--loop':
            loop_mode = True
        elif arg == '--setup':
            setup_mode = True

    # Setup mode
    if setup_mode:
        setup_telegram()
        return

    # Validate asset
    crypto_assets = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']
    if asset_name not in crypto_assets:
        print(f"  Invalid asset: {asset_name}")
        print(f"  Available: {', '.join(crypto_assets)}")
        return

    # Check for best models
    if not os.path.exists('crypto_hourly_best_models.csv'):
        print("\n  ERROR: No best models found!")
        print("  Run: python crypto_trading_system.py → Mode A first")
        return

    # Check Telegram config
    token = TELEGRAM_CONFIG.get('token', '')
    chat_id = TELEGRAM_CONFIG.get('chat_id', '')
    if not token or not chat_id:
        print("\n  Telegram not configured!")
        print("  Run: python crypto_live_trader.py --setup")
        print("  Or set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID environment variables")
        resp = input("\n  Continue without Telegram? (y/n): ").strip().lower()
        if resp != 'y':
            return

    # Menu (if no CLI args)
    if len(args) == 0:
        print(f"\n  Asset: {asset_name}")
        print("\n  Choose mode:")
        print("  1. Run once (generate signal, send notification)")
        print("  2. Loop (run every hour, auto-aligned)")
        print("  3. Setup Telegram")
        choice = input("\n  Enter 1, 2, or 3: ").strip()

        if choice == '3':
            setup_telegram()
            return
        elif choice == '2':
            loop_mode = True

        # Asset selection
        print(f"\n  Available: {', '.join(crypto_assets)}")
        asset_input = input(f"  Asset [{asset_name}]: ").strip().upper()
        if asset_input and asset_input in crypto_assets:
            asset_name = asset_input

    print(f"\n  Asset: {asset_name}")
    print(f"  Mode: {'Loop (hourly)' if loop_mode else 'Single run'}")

    if loop_mode:
        run_loop(asset_name)
    else:
        run_once(asset_name)

    print("\nDone!")


if __name__ == '__main__':
    main()
