"""
crypto_auto_trader.py - Automated BTC Trading via Revolut X
=============================================================
Connects ML 4h signals to Revolut X API for automated trading.

Rules:
  - Max position: $1,200 USD in BTC (NEVER more)
  - BUY signal  → buy full position ($1,200) if not holding
  - SELL signal → sell entire BTC holding
  - HOLD signal → do nothing
  - No confidence threshold — acts on every signal
  - Runs every hour using 4h model

Usage:
  python crypto_auto_trader.py                  # Run once (check signal + trade)
  python crypto_auto_trader.py --loop           # Hourly loop
  python crypto_auto_trader.py --dry-run        # Show what would happen, don't trade
  python crypto_auto_trader.py --status         # Show current position + balance
"""

import base64, json, time, uuid, sys, os
import urllib.request, urllib.error
from pathlib import Path
from datetime import datetime, timedelta, timezone

try:
    from nacl.signing import SigningKey
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
except ImportError:
    print("pip install pynacl cryptography")
    sys.exit(1)

# Import signal generation from main system
from crypto_live_trader import (
    run_once as generate_signal,
    send_telegram,
    load_best_config,
)
try:
    from crypto_live_trader import send_telegram_photo
except ImportError:
    send_telegram_photo = None


# ============================================================
# CONFIG
# ============================================================
CONFIG_FILE = "config/revolut_x_config.json"
PRIVATE_KEY_PATH = "config/private.pem"
import json as _json
with open(CONFIG_FILE) as _f:
    API_KEY = _json.load(_f)['api_key']
BASE_URL = "https://revx.revolut.com/api/1.0"

MAX_POSITION_USD = 1200.0   # NEVER buy more than this
SYMBOL = "BTC-USD"
ASSET = "BTC"
HORIZON = 4                 # 4h model

# Trade log file
TRADE_LOG = "auto_trader_log.json"


# ============================================================
# REVOLUT X API
# ============================================================
_signing_key = None

def _get_signing_key():
    global _signing_key
    if _signing_key is None:
        pk = serialization.load_pem_private_key(
            Path(PRIVATE_KEY_PATH).read_bytes(),
            password=None, backend=default_backend()
        )
        raw = pk.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption()
        )
        _signing_key = SigningKey(raw)
    return _signing_key


def revx_api(method, path, query='', body=None):
    """Make authenticated Revolut X API request."""
    sk = _get_signing_key()
    body_str = json.dumps(body, separators=(',', ':')) if body else ''
    full_path = f"/api/1.0{path}"
    ts = str(int(time.time() * 1000))
    msg = f"{ts}{method}{full_path}{query}{body_str}".encode('utf-8')
    sig = base64.b64encode(sk.sign(msg).signature).decode()

    url = f"{BASE_URL}{path}"
    if query:
        url += f"?{query}"

    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'X-Revx-Api-Key': API_KEY,
        'X-Revx-Timestamp': ts,
        'X-Revx-Signature': sig,
    }
    data = body_str.encode('utf-8') if body_str else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())
    except Exception as e:
        return 0, {'error': str(e)}


def get_btc_balance():
    """Get current BTC balance."""
    status, data = revx_api('GET', '/balances')
    if status == 200:
        for b in data:
            if b['currency'] == 'BTC':
                return float(b['available']), float(b['total'])
    return 0.0, 0.0


def get_btc_price():
    """Get current BTC-USD price from orderbook."""
    status, ob = revx_api('GET', '/order-book/BTC-USD')
    if status == 200:
        asks = ob.get('data', {}).get('asks', [])
        bids = ob.get('data', {}).get('bids', [])
        best_ask = float(asks[0]['p']) if asks else 0
        best_bid = float(bids[0]['p']) if bids else 0
        return best_ask, best_bid
    return 0, 0


def get_position_value():
    """Get current BTC position value in USD."""
    btc_avail, btc_total = get_btc_balance()
    if btc_total <= 0:
        return 0.0, 0.0, 0.0
    ask, bid = get_btc_price()
    mid_price = (ask + bid) / 2 if ask and bid else 0
    value_usd = btc_total * mid_price
    return btc_total, mid_price, value_usd


def buy_btc(amount_usd):
    """Place market BUY order for specified USD amount."""
    # Safety check: NEVER exceed max position
    _, _, current_value = get_position_value()
    remaining = MAX_POSITION_USD - current_value
    if remaining < 10:
        print(f"  ✗ Already at max position (${current_value:.0f}/${MAX_POSITION_USD:.0f})")
        return None

    amount_usd = min(amount_usd, remaining)
    amount_str = f"{amount_usd:.2f}"

    order = {
        "client_order_id": str(uuid.uuid4()),
        "symbol": SYMBOL,
        "side": "BUY",
        "order_configuration": {
            "market": {
                "quote_size": amount_str
            }
        }
    }

    print(f"  Placing BUY order: ${amount_str} of {SYMBOL}")
    status, result = revx_api('POST', '/orders', body=order)

    if status == 200:
        print(f"  ✓ BUY order placed!")
        log_trade('BUY', amount_usd, result)
        return result
    else:
        print(f"  ✗ BUY failed: {result.get('message', result)}")
        return None


def sell_btc():
    """Sell entire BTC position."""
    btc_avail, btc_total = get_btc_balance()
    if btc_avail <= 0.00000001:
        print(f"  ✗ No BTC to sell (balance: {btc_avail})")
        return None

    # Use base_size to sell exact BTC amount
    btc_str = f"{btc_avail:.8f}"

    order = {
        "client_order_id": str(uuid.uuid4()),
        "symbol": SYMBOL,
        "side": "SELL",
        "order_configuration": {
            "market": {
                "base_size": btc_str
            }
        }
    }

    print(f"  Placing SELL order: {btc_str} BTC")
    status, result = revx_api('POST', '/orders', body=order)

    if status == 200:
        ask, bid = get_btc_price()
        approx_usd = btc_avail * bid
        print(f"  ✓ SELL order placed! (~${approx_usd:.2f})")
        log_trade('SELL', approx_usd, result)
        return result
    else:
        print(f"  ✗ SELL failed: {result.get('message', result)}")
        return None


# ============================================================
# TRADE LOG
# ============================================================
def log_trade(action, amount_usd, api_response):
    """Log trade to JSON file."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'amount_usd': round(amount_usd, 2),
        'response': api_response,
    }

    log = []
    if os.path.exists(TRADE_LOG):
        try:
            with open(TRADE_LOG, 'r') as f:
                log = json.load(f)
        except:
            pass

    log.append(entry)
    with open(TRADE_LOG, 'w') as f:
        json.dump(log, f, indent=2)


# ============================================================
# SIGNAL → TRADE LOGIC
# ============================================================
def execute_signal(signal_data, dry_run=False):
    """
    Take ML signal and execute trade.
    Returns: action taken ('BUY', 'SELL', 'HOLD', 'SKIP')
    """
    if signal_data is None:
        print("  ✗ No signal data")
        return 'ERROR'

    signal = signal_data.get('signal', 'HOLD')
    confidence = signal_data.get('confidence', 0)
    price = signal_data.get('close', 0)

    print(f"\n  Signal: {signal} ({confidence:.0f}%) | BTC: ${price:,.2f}")

    # Get current position
    btc_total, mid_price, position_usd = get_position_value()
    is_holding = position_usd > 10  # holding if position > $10

    print(f"  Position: {btc_total:.8f} BTC ≈ ${position_usd:.2f}")

    action = 'HOLD'

    if signal == 'BUY' and not is_holding:
        # Buy full position
        buy_amount = min(MAX_POSITION_USD, MAX_POSITION_USD - position_usd)
        if buy_amount < 10:
            print(f"  Already at max position. Holding.")
            action = 'HOLD'
        elif dry_run:
            print(f"  [DRY RUN] Would BUY ${buy_amount:.2f} of BTC")
            action = 'BUY (dry)'
        else:
            result = buy_btc(buy_amount)
            action = 'BUY' if result else 'ERROR'

    elif signal == 'BUY' and is_holding:
        print(f"  Already holding ${position_usd:.0f}. Keeping position.")
        action = 'HOLD'

    elif signal == 'SELL' and is_holding:
        if dry_run:
            print(f"  [DRY RUN] Would SELL {btc_total:.8f} BTC (≈${position_usd:.2f})")
            action = 'SELL (dry)'
        else:
            result = sell_btc()
            action = 'SELL' if result else 'ERROR'

    elif signal == 'SELL' and not is_holding:
        print(f"  No position to sell. Skipping.")
        action = 'SKIP'

    else:
        print(f"  HOLD — no action.")

    return action


# ============================================================
# TELEGRAM NOTIFICATION
# ============================================================
def send_trade_notification(signal_data, action, position_before, position_after):
    """Send Telegram notification about trade."""
    if signal_data is None:
        return

    signal = signal_data.get('signal', '?')
    confidence = signal_data.get('confidence', 0)
    price = signal_data.get('close', 0)

    if action in ('BUY', 'SELL'):
        emoji = '🟢' if action == 'BUY' else '🔴'
        msg = (
            f"{emoji} <b>AUTO-TRADE: {action}</b>\n\n"
            f"Signal: {signal} ({confidence:.0f}%)\n"
            f"BTC Price: ${price:,.2f}\n"
            f"Position before: ${position_before:.2f}\n"
            f"Position after: ${position_after:.2f}\n"
            f"Max allowed: ${MAX_POSITION_USD:.0f}\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
    else:
        msg = (
            f"🟡 <b>AUTO-TRADER: {action}</b>\n\n"
            f"Signal: {signal} ({confidence:.0f}%)\n"
            f"BTC: ${price:,.2f}\n"
            f"Position: ${position_after:.2f}\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

    send_telegram(msg)


# ============================================================
# MAIN FUNCTIONS
# ============================================================
def show_status():
    """Show current position and account status."""
    print(f"\n{'='*60}")
    print(f"  AUTO-TRADER STATUS")
    print(f"{'='*60}")

    # Balances
    status, balances = revx_api('GET', '/balances')
    if status == 200:
        print(f"\n  Balances:")
        for b in balances:
            if float(b['total']) > 0:
                print(f"    {b['currency']:6s}  available={b['available']:>15s}  total={b['total']:>15s}")

    # Position value
    btc_total, mid_price, position_usd = get_position_value()
    print(f"\n  BTC Position: {btc_total:.8f} BTC")
    print(f"  BTC Price: ${mid_price:,.2f}")
    print(f"  Position Value: ${position_usd:.2f} / ${MAX_POSITION_USD:.0f} max")
    print(f"  Status: {'HOLDING' if position_usd > 10 else 'FLAT (no position)'}")

    # Last trades
    if os.path.exists(TRADE_LOG):
        with open(TRADE_LOG, 'r') as f:
            log = json.load(f)
        if log:
            last = log[-1]
            print(f"\n  Last trade: {last['action']} ${last['amount_usd']:.2f} at {last['timestamp']}")
            print(f"  Total trades: {len(log)}")

    # Check model config
    config = load_best_config(ASSET, horizon=HORIZON)
    if config:
        print(f"\n  Model: {config['best_combo']} | w={config['best_window']}h | "
              f"{config['accuracy']:.1f}% | Set {config.get('feature_set', 'A')}")
    else:
        print(f"\n  ✗ No {HORIZON}h model found! Run Mode D first.")

    print(f"\n{'='*60}")


def run_once_trade(dry_run=False):
    """Generate signal and execute trade."""
    print(f"\n{'='*60}")
    print(f"  AUTO-TRADER: {ASSET} ({HORIZON}h model)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Max position: ${MAX_POSITION_USD:.0f} | {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'='*60}")

    # Check model exists
    config = load_best_config(ASSET, horizon=HORIZON)
    if not config:
        print(f"\n  ✗ No {HORIZON}h model for {ASSET}! Run Mode D first.")
        return

    # Get position before
    _, _, position_before = get_position_value()

    # Generate ML signal (4h only)
    print(f"\n  Generating {HORIZON}h signal...")
    signal_data = generate_signal(ASSET, horizon_filter=[HORIZON])

    if signal_data is None:
        print("  ✗ Signal generation failed!")
        send_telegram(f"⚠️ <b>Auto-trader</b>: signal generation failed at {datetime.now().strftime('%H:%M')}")
        return

    # Execute trade
    action = execute_signal(signal_data, dry_run=dry_run)

    # Get position after
    time.sleep(2)
    _, _, position_after = get_position_value()

    # Summary
    print(f"\n  Action: {action}")
    print(f"  Position: ${position_before:.2f} → ${position_after:.2f}")

    # Telegram notification
    if not dry_run:
        send_trade_notification(signal_data, action, position_before, position_after)


def run_loop_trade(dry_run=False):
    """Run auto-trader every hour."""
    print(f"\n{'='*60}")
    print(f"  AUTO-TRADER LOOP: {ASSET} ({HORIZON}h)")
    print(f"  Max position: ${MAX_POSITION_USD:.0f}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}")

    # Startup notification
    if not dry_run:
        _, _, pos_usd = get_position_value()
        send_telegram(
            f"🤖 <b>Auto-Trader Started</b>\n\n"
            f"Asset: {ASSET}\n"
            f"Model: {HORIZON}h\n"
            f"Max position: ${MAX_POSITION_USD:.0f}\n"
            f"Current position: ${pos_usd:.2f}\n"
            f"Mode: LIVE\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

    cycle = 0
    while True:
        try:
            # Wait until top of next hour
            now = datetime.now()
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            wait_seconds = (next_hour - now).total_seconds()

            if wait_seconds > 5:
                print(f"\n  Waiting for {next_hour.strftime('%H:%M')} ({wait_seconds/60:.0f} min)...")
                time.sleep(wait_seconds)

            cycle += 1
            print(f"\n  === Cycle {cycle} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

            run_once_trade(dry_run=dry_run)

        except KeyboardInterrupt:
            print(f"\n\n  Stopped by user after {cycle} cycles.")
            if not dry_run:
                send_telegram(f"🛑 <b>Auto-Trader Stopped</b>\nCycles: {cycle}")
            break
        except Exception as e:
            print(f"\n  ERROR in cycle {cycle}: {e}")
            if not dry_run:
                send_telegram(f"⚠️ <b>Auto-Trader Error</b>\nCycle {cycle}: <code>{e}</code>")
            time.sleep(120)


# ============================================================
# CLI
# ============================================================
def main():
    args = sys.argv[1:]
    dry_run = '--dry-run' in args

    if '--status' in args:
        show_status()
    elif '--loop' in args:
        run_loop_trade(dry_run=dry_run)
    else:
        run_once_trade(dry_run=dry_run)


if __name__ == '__main__':
    main()
