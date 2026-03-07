"""
Revolut X Auto-Trader — Real Order Execution
================================================
Integrates the "both agree" ML strategy with Revolut X API:
  - Auto-places MARKET orders on Revolut X when signal fires
  - BUY only when 4h+8h agree with >=75% confidence
  - SELL when either model says SELL
  - Tracks position state + trade history
  - Sends Telegram notifications with PnL

Requirements:
  pip install pynacl cryptography

Setup:
  1. Place your private.pem in the same folder
  2. Create revolut_x_config.json with your API key:
     {"api_key": "YOUR_API_KEY", "symbol": "BTC-USD"}
  3. Set up Telegram: python crypto_revolut_trader.py --setup-telegram

Usage:
  python crypto_revolut_trader.py                  # Interactive
  python crypto_revolut_trader.py --loop           # Auto-loop
  python crypto_revolut_trader.py --loop --max 500 # Max $500 per trade
  python crypto_revolut_trader.py --status         # Check position
  python crypto_revolut_trader.py --balance        # Check Revolut X balance
  python crypto_revolut_trader.py --dry-run        # Signals only, no orders
  python crypto_revolut_trader.py --reset          # Reset position to cash
"""

import os
import sys
import time
import json
import uuid
import base64
import urllib.request
import urllib.error
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')

from crypto_live_trader import (
    load_best_config, generate_live_signal, compute_combined_signal,
    send_telegram, wait_for_fresh_candle, setup_telegram,
    download_asset, load_data, TELEGRAM_CONFIG, MIN_CONFIDENCE,
    _to_local,
)


# ============================================================
# REVOLUT X API CLIENT
# ============================================================
REVX_CONFIG_FILE = 'config/revolut_x_config.json'
REVX_BASE_URL = 'https://revx.revolut.com/api/1.0'
PRIVATE_KEY_PATH = 'config/private.pem'

def _load_revx_config():
    """Load Revolut X API config."""
    if not os.path.exists(REVX_CONFIG_FILE):
        return {'api_key': '', 'symbol': 'BTC-USD'}
    with open(REVX_CONFIG_FILE) as f:
        return json.load(f)

def _load_signing_key():
    """Load Ed25519 private key for request signing."""
    from nacl.signing import SigningKey
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend

    if not Path(PRIVATE_KEY_PATH).exists():
        print(f"  ERROR: {PRIVATE_KEY_PATH} not found!")
        return None

    pem_data = Path(PRIVATE_KEY_PATH).read_bytes()
    pk = serialization.load_pem_private_key(pem_data, password=None, backend=default_backend())
    raw = pk.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption()
    )
    return SigningKey(raw)


def revx_api(method, path, query='', body=None):
    """Make authenticated request to Revolut X API."""
    config = _load_revx_config()
    sk = _load_signing_key()
    if not sk or not config.get('api_key'):
        return 0, {'error': 'API not configured'}

    body_str = json.dumps(body, separators=(',', ':')) if body else ''
    full_path = f"/api/1.0{path}"
    ts = str(int(time.time() * 1000))
    msg = f"{ts}{method}{full_path}{query}{body_str}".encode('utf-8')
    sig = base64.b64encode(sk.sign(msg).signature).decode()

    url = f"{REVX_BASE_URL}{path}"
    if query:
        url += f"?{query}"

    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'X-Revx-Api-Key': config['api_key'],
        'X-Revx-Timestamp': ts,
        'X-Revx-Signature': sig,
    }

    data = body_str.encode('utf-8') if body_str else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except:
            return e.code, {'error': str(e)}
    except Exception as e:
        return 0, {'error': str(e)}


def get_balances():
    """Get all Revolut X balances."""
    status, data = revx_api('GET', '/balances')
    if status == 200:
        return {b['currency']: {'available': float(b['available']), 'total': float(b['total'])} for b in data}
    return {}


def get_btc_price():
    """Get current BTC-USD price from orderbook."""
    config = _load_revx_config()
    symbol = config.get('symbol', 'BTC-USD')
    try:
        url = f"{REVX_BASE_URL}/public/order-book/{symbol}"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        best_ask = float(data['asks'][0][0]) if data.get('asks') else 0
        best_bid = float(data['bids'][0][0]) if data.get('bids') else 0
        return {'ask': best_ask, 'bid': best_bid, 'mid': (best_ask + best_bid) / 2}
    except:
        return None


def place_market_buy(quote_size_usd):
    """Place a market BUY order for a given USD amount."""
    config = _load_revx_config()
    symbol = config.get('symbol', 'BTC-USD')
    order_id = str(uuid.uuid4())

    body = {
        'client_order_id': order_id,
        'symbol': symbol,
        'side': 'BUY',
        'order_configuration': {
            'market': {
                'quote_size': str(round(quote_size_usd, 2))
            }
        }
    }

    print(f"  Placing MARKET BUY: ${quote_size_usd:.2f} of {symbol}...")
    status, data = revx_api('POST', '/orders', body=body)
    print(f"  Response: {status} {json.dumps(data)[:200]}")
    return status, data


def place_market_sell(base_size_btc):
    """Place a market SELL order for a given BTC amount."""
    config = _load_revx_config()
    symbol = config.get('symbol', 'BTC-USD')
    order_id = str(uuid.uuid4())

    body = {
        'client_order_id': order_id,
        'symbol': symbol,
        'side': 'SELL',
        'order_configuration': {
            'market': {
                'base_size': str(base_size_btc)
            }
        }
    }

    print(f"  Placing MARKET SELL: {base_size_btc} BTC...")
    status, data = revx_api('POST', '/orders', body=body)
    print(f"  Response: {status} {json.dumps(data)[:200]}")
    return status, data


# ============================================================
# POSITION STATE
# ============================================================
POSITION_FILE = 'config/revolut_position.json'

def load_position():
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE) as f:
            return json.load(f)
    return {
        'state': 'cash',
        'max_position_usd': 0,
        'entry_price': 0,
        'entry_time': '',
        'btc_amount': 0,
        'usd_invested': 0,
        'trades': [],
        'auto_trade': False,
    }

def save_position(pos):
    with open(POSITION_FILE, 'w') as f:
        json.dump(pos, f, indent=2)


# ============================================================
# TRADE EXECUTION
# ============================================================
def execute_buy(position, price, dry_run=False):
    """Execute a BUY on Revolut X or log for manual execution."""
    max_usd = position['max_position_usd']
    if max_usd <= 0:
        print("  ERROR: Max position not set!")
        return False

    if not dry_run and position.get('auto_trade'):
        # Check USD balance first
        balances = get_balances()
        usd_available = balances.get('USD', {}).get('available', 0)

        if usd_available < max_usd * 0.95:  # 5% margin
            msg = f"⚠️ Insufficient USD balance: ${usd_available:.2f} < ${max_usd:.2f}"
            print(f"  {msg}")
            send_telegram(msg)
            return False

        # Place order
        status, data = place_market_buy(max_usd)
        if status not in (200, 201):
            msg = f"⚠️ BUY order failed: {status} {data}"
            print(f"  {msg}")
            send_telegram(msg)
            return False

        order_data = data.get('data', data)
        filled_qty = float(order_data.get('filled_size', 0))
        avg_price = float(order_data.get('average_fill_price', price))

        position['btc_amount'] = filled_qty
        position['entry_price'] = avg_price
        print(f"  ✓ BUY executed: {filled_qty} BTC @ ${avg_price:,.2f}")
    else:
        position['btc_amount'] = max_usd / price
        position['entry_price'] = price

    position['state'] = 'invested'
    position['entry_time'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    position['usd_invested'] = max_usd
    position['trades'].append({
        'action': 'BUY', 'price': position['entry_price'],
        'time': position['entry_time'], 'usd': max_usd,
        'btc': position['btc_amount'],
        'auto': not dry_run and position.get('auto_trade', False),
    })
    save_position(position)
    return True


def execute_sell(position, price, dry_run=False):
    """Execute a SELL on Revolut X or log for manual execution."""
    btc_amount = position.get('btc_amount', 0)

    if not dry_run and position.get('auto_trade') and btc_amount > 0:
        # Sell all BTC
        status, data = place_market_sell(btc_amount)
        if status not in (200, 201):
            msg = f"⚠️ SELL order failed: {status} {data}"
            print(f"  {msg}")
            send_telegram(msg)
            return False

        order_data = data.get('data', data)
        avg_price = float(order_data.get('average_fill_price', price))
        print(f"  ✓ SELL executed: {btc_amount} BTC @ ${avg_price:,.2f}")
        price = avg_price

    pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100 if position['entry_price'] > 0 else 0
    pnl_usd = position['usd_invested'] * pnl_pct / 100

    position['trades'].append({
        'action': 'SELL', 'price': price,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'pnl_pct': round(pnl_pct, 2), 'pnl_usd': round(pnl_usd, 2),
        'auto': not dry_run and position.get('auto_trade', False),
    })

    position['state'] = 'cash'
    position['entry_price'] = 0
    position['entry_time'] = ''
    position['btc_amount'] = 0
    position['usd_invested'] = 0
    save_position(position)
    return True


# ============================================================
# MAIN SIGNAL + EXECUTION LOOP
# ============================================================
def run_once(asset_name, position, dry_run=False):
    """Generate signal, decide action, execute if auto_trade."""
    print(f"\n{'='*60}")
    mode = "DRY RUN" if dry_run else ("AUTO" if position.get('auto_trade') else "MANUAL")
    print(f"  REVOLUT X TRADER [{mode}]: {asset_name}")
    print(f"  Position: {position['state'].upper()} | Max: ${position['max_position_usd']:,.2f}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    config_4h = load_best_config(asset_name, horizon=4)
    config_8h = load_best_config(asset_name, horizon=8)
    if not config_4h and not config_8h:
        print("  ERROR: No models!")
        return None

    print(f"\n  Downloading {asset_name} data...")
    try:
        download_asset(asset_name, update_only=True)
    except Exception as e:
        print(f"  Download error: {e}")

    df_raw = load_data(asset_name)
    if df_raw is None:
        return None

    sig_4h = sig_8h = None
    if config_4h:
        sig_4h = generate_live_signal(asset_name, config_4h, df_raw=df_raw)
        if sig_4h:
            print(f"  4h: {sig_4h['signal']} ({sig_4h['confidence']:.0f}%)")
    if config_8h:
        sig_8h = generate_live_signal(asset_name, config_8h, df_raw=df_raw)
        if sig_8h:
            print(f"  8h: {sig_8h['signal']} ({sig_8h['confidence']:.0f}%)")

    action, confidence, reason = compute_combined_signal(sig_4h, sig_8h)
    any_sig = sig_4h or sig_8h
    price = any_sig['close'] if any_sig else 0

    print(f"\n  >>> {action} ({confidence:.0f}%) — {reason}")

    # Execute based on state
    executed = False
    pnl_msg = ""

    if action == 'BUY' and position['state'] == 'cash':
        if dry_run:
            print(f"  [DRY RUN] Would BUY ${position['max_position_usd']:,.2f} of BTC at ${price:,.2f}")
        else:
            executed = execute_buy(position, price, dry_run=False)
    elif action == 'SELL' and position['state'] == 'invested':
        pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
        pnl_usd = position['usd_invested'] * pnl_pct / 100
        pnl_msg = f"\n{'📈' if pnl_pct > 0 else '📉'} PnL: {pnl_pct:+.1f}% (${pnl_usd:+,.2f})"
        if dry_run:
            print(f"  [DRY RUN] Would SELL {position['btc_amount']:.6f} BTC at ${price:,.2f}{pnl_msg}")
        else:
            executed = execute_sell(position, price, dry_run=False)

    # Build Telegram message
    msg = _format_message(asset_name, sig_4h, sig_8h, action, confidence, reason,
                          price, position, executed, dry_run, pnl_msg)
    send_telegram(msg)

    return {'action': action, 'confidence': confidence, 'price': price, 'executed': executed}


def _format_message(asset_name, sig_4h, sig_8h, action, confidence, reason,
                    price, position, executed, dry_run, pnl_msg):
    """Format Telegram message."""
    price_str = f"${price:,.2f}"

    def _sig(sig, h):
        if not sig: return f"{h}h: N/A"
        e = '🟢' if sig['signal'] == 'BUY' else '🔴' if sig['signal'] == 'SELL' else '🟡'
        return f"{e} {h}h: {sig['signal']} ({sig['confidence']:.0f}%)"

    # Action line
    if action == 'BUY' and position['state'] == 'cash':
        if executed:
            action_line = f"✅ <b>BOUGHT ${position['max_position_usd']:,.0f} BTC</b>"
        elif dry_run:
            action_line = f"🔵 <b>[DRY] BUY ${position['max_position_usd']:,.0f}</b>"
        else:
            action_line = f"🟢 <b>BUY SIGNAL — Open Revolut X → Buy ${position['max_position_usd']:,.0f}</b>"
    elif action == 'SELL' and position['state'] == 'invested':
        if executed:
            action_line = f"🚨 <b>SOLD all BTC</b>{pnl_msg}"
        elif dry_run:
            action_line = f"🔵 <b>[DRY] SELL</b>{pnl_msg}"
        else:
            action_line = f"🔴 <b>SELL SIGNAL — Open Revolut X → Sell All</b>{pnl_msg}"
    elif action == 'BUY' and position['state'] == 'invested':
        cur_pnl = (price - position['entry_price']) / position['entry_price'] * 100 if position['entry_price'] else 0
        action_line = f"🟢 Already invested ({cur_pnl:+.1f}%)"
    elif action == 'SELL' and position['state'] == 'cash':
        action_line = f"🔴 SELL signal — already in cash"
    else:
        if position['state'] == 'invested':
            cur_pnl = (price - position['entry_price']) / position['entry_price'] * 100 if position['entry_price'] else 0
            action_line = f"⏸ <b>HOLD</b> position ({cur_pnl:+.1f}%)"
        else:
            action_line = f"⏸ <b>HOLD</b> — stay in cash"

    reason_map = {
        'both_buy': '4h+8h agree BUY', 'both_sell': '4h+8h agree SELL',
        '4h_sell': '4h says SELL', '8h_sell': '8h says SELL',
        'disagree': 'Models disagree', 'low_conf_4h': '4h conf too low',
        'low_conf_8h': '8h conf too low',
    }

    lines = [
        f"<b>{asset_name}</b>  {_sig(sig_4h, 4)} | {_sig(sig_8h, 8)}",
        "", action_line,
        f"📋 {reason_map.get(reason, reason)}",
        f"💰 Price: {price_str}",
    ]

    any_sig = sig_4h or sig_8h
    if any_sig:
        lines.append(f"📈 RSI: {any_sig['rsi']}")

    # Show open position details when invested
    if position['state'] == 'invested' and position['entry_price'] > 0:
        cur_pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
        cur_pnl_usd = position.get('usd_invested', 0) * cur_pnl_pct / 100
        cur_value = position.get('usd_invested', 0) + cur_pnl_usd
        pnl_emoji = '📈' if cur_pnl_pct > 0 else '📉'
        lines.extend([
            "",
            "📦 <b>Open Position:</b>",
            f"  Entry: ${position['entry_price']:,.2f} ({position.get('entry_time', '')})",
            f"  Invested: ${position.get('usd_invested', 0):,.2f}",
            f"  BTC: {position.get('btc_amount', 0):.6f}",
            f"  Value: ${cur_value:,.2f}",
            f"  {pnl_emoji} PnL: {cur_pnl_pct:+.1f}% (${cur_pnl_usd:+,.2f})",
        ])

    mode = "DRY" if dry_run else ("AUTO" if position.get('auto_trade') else "MANUAL")
    lines.append(f"\n🤖 Mode: {mode} | Max: ${position['max_position_usd']:,.0f}")

    if any_sig:
        lines.append(f"⏰ {any_sig['datetime']}")

    return "\n".join(lines)


# ============================================================
# LOOP
# ============================================================
# ============================================================
# TELEGRAM COMMAND LISTENER
# ============================================================
_last_update_id = 0

def check_telegram_commands():
    """Check for incoming Telegram commands. Returns command or None."""
    global _last_update_id
    token = TELEGRAM_CONFIG.get('token', '')
    if not token:
        return None

    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates?offset={_last_update_id + 1}&timeout=0"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        if not data.get('ok') or not data.get('result'):
            return None

        for update in data['result']:
            _last_update_id = update['update_id']
            msg = update.get('message', {})
            text = msg.get('text', '').strip().lower()
            if text in ('/stop', '/status', '/pause', '/resume', '/balance'):
                return text
    except Exception:
        pass
    return None


def _flush_old_updates():
    """Flush old Telegram updates so we only respond to new commands."""
    global _last_update_id
    token = TELEGRAM_CONFIG.get('token', '')
    if not token:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates?offset=-1"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        if data.get('ok') and data.get('result'):
            _last_update_id = data['result'][-1]['update_id']
    except Exception:
        pass


def _handle_status_command(asset_name, position, price=None):
    """Handle /status command — send position info via Telegram."""
    pos = load_position()
    if pos['state'] == 'invested' and pos['entry_price'] > 0:
        if price is None:
            px = get_btc_price()
            price = px['mid'] if px else pos['entry_price']
        pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
        pnl_usd = pos.get('usd_invested', 0) * pnl_pct / 100
        pnl_emoji = '📈' if pnl_pct > 0 else '📉'
        msg = (
            f"📦 <b>Position: INVESTED</b>\n\n"
            f"Entry: ${pos['entry_price']:,.2f}\n"
            f"Now: ${price:,.2f}\n"
            f"BTC: {pos.get('btc_amount', 0):.6f}\n"
            f"{pnl_emoji} PnL: {pnl_pct:+.1f}% (${pnl_usd:+,.2f})\n"
            f"Since: {pos.get('entry_time', '')}"
        )
    else:
        msg = f"💵 <b>Position: CASH</b>\nMax: ${pos['max_position_usd']:,.2f}\nWaiting for BUY signal."

    sells = [t for t in pos.get('trades', []) if t.get('action') == 'SELL']
    if sells:
        total = sum(t.get('pnl_usd', 0) for t in sells)
        wins = sum(1 for t in sells if t.get('pnl_usd', 0) > 0)
        msg += f"\n\n📊 History: {len(sells)} trades ({wins} wins)\nTotal PnL: ${total:+,.2f}"

    send_telegram(msg)


# ============================================================
# LOOP WITH REMOTE CONTROL
# ============================================================
def run_loop(asset_name, position, dry_run=False):
    print(f"\n{'='*60}")
    mode = "DRY RUN" if dry_run else ("AUTO" if position.get('auto_trade') else "MANUAL")
    print(f"  REVOLUT X TRADER [{mode}]: {asset_name}")
    print(f"  Max: ${position['max_position_usd']:,.2f} | State: {position['state'].upper()}")
    print(f"  Strategy: Both Agree | Min conf: {MIN_CONFIDENCE}%")
    print(f"  Telegram commands: /stop /status /pause /resume /balance")
    print(f"{'='*60}")

    c4 = load_best_config(asset_name, horizon=4)
    c8 = load_best_config(asset_name, horizon=8)
    if c4: print(f"  ✓ 4h: {c4['best_combo']} | {c4['accuracy']:.1f}%")
    if c8: print(f"  ✓ 8h: {c8['best_combo']} | {c8['accuracy']:.1f}%")

    # Flush old messages so we don't react to stale commands
    _flush_old_updates()

    send_telegram(
        f"🚀 <b>Revolut X Trader Started</b>\n\n"
        f"Asset: {asset_name}\nMode: {mode}\n"
        f"Max: ${position['max_position_usd']:,.2f}\n"
        f"State: {position['state'].upper()}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"📱 <b>Commands:</b>\n"
        f"/stop — Stop trader\n"
        f"/status — Show position\n"
        f"/pause — Pause trading\n"
        f"/resume — Resume trading\n"
        f"/balance — Check Revolut X balance"
    )

    cycle = 0
    paused = False
    while True:
        try:
            now = datetime.now()
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            wait_sec = (next_hour - now).total_seconds()

            if wait_sec > 5:
                state = f"IN (${position['entry_price']:,.0f})" if position['state'] == 'invested' else "CASH"
                pause_str = " [PAUSED]" if paused else ""
                print(f"\n  [{state}{pause_str}] Next: {next_hour.strftime('%H:%M')} ({wait_sec/60:.0f} min)...")

                # Sleep in 30s chunks, checking for Telegram commands between
                remaining = wait_sec
                while remaining > 0:
                    sleep_time = min(30, remaining)
                    time.sleep(sleep_time)
                    remaining -= sleep_time

                    # Check for commands
                    cmd = check_telegram_commands()
                    if cmd == '/stop':
                        print("\n  🛑 STOP command received via Telegram!")
                        pos = load_position()
                        sells = [t for t in pos.get('trades', []) if t.get('action') == 'SELL']
                        total_pnl = sum(t.get('pnl_usd', 0) for t in sells)
                        send_telegram(f"🛑 <b>Trader Stopped (via Telegram)</b>\nCycles: {cycle}\nPnL: ${total_pnl:+,.2f}")
                        return
                    elif cmd == '/status':
                        print("  📊 Status requested via Telegram")
                        _handle_status_command(asset_name, position)
                    elif cmd == '/pause':
                        paused = True
                        print("  ⏸ PAUSED via Telegram")
                        send_telegram("⏸ <b>Trader PAUSED</b>\nSignals will still show but no orders will be placed.\nSend /resume to continue.")
                    elif cmd == '/resume':
                        paused = False
                        print("  ▶️ RESUMED via Telegram")
                        send_telegram("▶️ <b>Trader RESUMED</b>\nAuto-trading active again.")
                    elif cmd == '/balance':
                        balances = get_balances()
                        bal_lines = []
                        for curr, b in sorted(balances.items()):
                            if b['total'] > 0:
                                bal_lines.append(f"  {curr}: {b['available']:.6f}")
                        send_telegram(f"💰 <b>Revolut X Balance</b>\n" + "\n".join(bal_lines) if bal_lines else "No funds")

            cycle += 1
            print(f"\n  --- Cycle {cycle} | {datetime.now().strftime('%H:%M:%S')} ---")

            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            expected = now_utc.replace(minute=0, second=0, microsecond=0)
            wait_for_fresh_candle(asset_name, expected)

            position = load_position()

            if paused:
                print("  ⏸ PAUSED — skipping execution")
                # Still generate signals for display, but force dry_run
                run_once(asset_name, position, dry_run=True)
            else:
                run_once(asset_name, position, dry_run=dry_run)

        except KeyboardInterrupt:
            print("\n  Stopped.")
            pos = load_position()
            sells = [t for t in pos.get('trades', []) if t.get('action') == 'SELL']
            total_pnl = sum(t.get('pnl_usd', 0) for t in sells)
            send_telegram(f"🛑 <b>Stopped</b>\nCycles: {cycle}\nTrades: {len(sells)}\nPnL: ${total_pnl:+,.2f}")
            break
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            send_telegram(f"⚠️ <b>Error</b>\n<code>{e}</code>")
            time.sleep(120)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  REVOLUT X CRYPTO TRADER")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    asset_name = 'BTC'
    loop_mode = False
    dry_run = False
    max_pos = None

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--asset' and i + 1 < len(args):
            asset_name = args[i + 1].upper()
        elif arg == '--loop':
            loop_mode = True
        elif arg == '--dry-run':
            dry_run = True
        elif arg == '--max' and i + 1 < len(args):
            max_pos = float(args[i + 1])
        elif arg == '--setup-telegram':
            setup_telegram()
            return
        elif arg == '--reset':
            if os.path.exists(POSITION_FILE):
                os.remove(POSITION_FILE)
            print("  Position reset.")
            return
        elif arg == '--balance':
            balances = get_balances()
            if balances:
                print("\n  Revolut X Balances:")
                for curr, b in sorted(balances.items()):
                    if b['total'] > 0:
                        print(f"    {curr:6s}  available={b['available']:>12.6f}  total={b['total']:>12.6f}")
            else:
                print("  ERROR: Could not fetch balances.")
            return
        elif arg == '--status':
            pos = load_position()
            print(f"\n  State: {pos['state'].upper()}")
            print(f"  Max: ${pos['max_position_usd']:,.2f}")
            print(f"  Auto-trade: {pos.get('auto_trade', False)}")
            if pos['state'] == 'invested':
                print(f"  Entry: ${pos['entry_price']:,.2f} at {pos['entry_time']}")
                print(f"  BTC: {pos['btc_amount']:.8f}")
                px = get_btc_price()
                if px:
                    cur_pnl = (px['mid'] - pos['entry_price']) / pos['entry_price'] * 100
                    print(f"  Current: ${px['mid']:,.2f} ({cur_pnl:+.1f}%)")
            sells = [t for t in pos.get('trades', []) if t.get('action') == 'SELL']
            if sells:
                total = sum(t.get('pnl_usd', 0) for t in sells)
                wins = sum(1 for t in sells if t.get('pnl_usd', 0) > 0)
                print(f"  Trades: {len(sells)} ({wins} wins) | PnL: ${total:+,.2f}")
            return

    # Check requirements
    if not os.path.exists('models/crypto_hourly_best_models.csv'):
        print("\n  ERROR: No models found!")
        return

    # Load position
    position = load_position()

    # Check Revolut X config
    revx_config = _load_revx_config()
    has_api = bool(revx_config.get('api_key')) and Path(PRIVATE_KEY_PATH).exists()

    # Interactive menu
    if len(args) == 0:
        print(f"\n  State: {position['state'].upper()}")
        if position['max_position_usd'] > 0:
            print(f"  Max position: ${position['max_position_usd']:,.2f}")
        print(f"  Revolut X API: {'✓ Connected' if has_api else '✗ Not configured'}")
        print(f"  Auto-trade: {position.get('auto_trade', False)}")

        print(f"\n  1. Run once (signal + execute)")
        print(f"  2. Loop (hourly)")
        print(f"  3. Dry run (signals only)")
        print(f"  4. Check balance")
        print(f"  5. View status / history")
        print(f"  6. {'Disable' if position.get('auto_trade') else 'Enable'} auto-trading")
        print(f"  7. Setup Telegram")
        print(f"  8. Reset position")
        ch = input("\n  Enter 1-8: ").strip()

        if ch == '7':
            setup_telegram()
            return
        elif ch == '4':
            balances = get_balances()
            for curr, b in sorted(balances.items()):
                if b['total'] > 0:
                    print(f"    {curr:6s}  {b['available']:>12.6f}")
            return
        elif ch == '5':
            for t in position.get('trades', [])[-10:]:
                if t['action'] == 'BUY':
                    auto = " [AUTO]" if t.get('auto') else ""
                    print(f"    🟢 BUY  ${t['price']:,.2f} | {t['time']}{auto}")
                else:
                    auto = " [AUTO]" if t.get('auto') else ""
                    print(f"    🔴 SELL ${t['price']:,.2f} | {t['time']} | {t.get('pnl_pct', 0):+.1f}%{auto}")
            return
        elif ch == '6':
            if not has_api:
                print("  ERROR: Revolut X API not configured!")
                print(f"  Create {REVX_CONFIG_FILE} with your API key")
                return
            position['auto_trade'] = not position.get('auto_trade', False)
            save_position(position)
            status = "ENABLED" if position['auto_trade'] else "DISABLED"
            print(f"  Auto-trading: {status}")
            return
        elif ch == '8':
            position = {'state': 'cash', 'max_position_usd': 0, 'entry_price': 0,
                        'entry_time': '', 'btc_amount': 0, 'usd_invested': 0,
                        'trades': [], 'auto_trade': False}
            save_position(position)
            print("  Position reset.")
            return
        elif ch == '3':
            dry_run = True
        elif ch == '2':
            loop_mode = True

        a = input(f"\n  Asset [{asset_name}]: ").strip().upper()
        if a in ['BTC', 'ETH', 'XRP', 'DOGE']:
            asset_name = a

    # Set max position
    if max_pos is not None:
        position['max_position_usd'] = max_pos
        save_position(position)
    elif position['max_position_usd'] <= 0:
        print(f"\n  ⚠️  MAX POSITION not set.")
        print(f"  This is the maximum USD amount per trade.\n")
        while True:
            try:
                val = float(input("  Enter MAX position in USD: $").strip())
                if val > 0:
                    position['max_position_usd'] = val
                    save_position(position)
                    print(f"  ✓ Max set to ${val:,.2f}")
                    break
            except ValueError:
                print("  Enter a number")

    mode = "DRY RUN" if dry_run else ("AUTO" if position.get('auto_trade') else "MANUAL")
    print(f"\n  Asset: {asset_name} | Mode: {mode}")
    print(f"  Max: ${position['max_position_usd']:,.2f} | State: {position['state'].upper()}")

    if loop_mode:
        run_loop(asset_name, position, dry_run=dry_run)
    else:
        run_once(asset_name, position, dry_run=dry_run)


if __name__ == '__main__':
    main()
