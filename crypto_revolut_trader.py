"""
Revolut X Multi-Asset Auto-Trader
====================================
Trades BTC + ETH (and more) simultaneously with per-asset strategies:
  - BTC: "both_agree" (BUY when 4h+8h agree, SELL when either says SELL)
  - ETH: "either" (BUY when either says BUY, SELL when either says SELL)
  - Per-asset max position in USD
  - Per-asset position tracking

Usage:
  python crypto_revolut_trader.py                  # Interactive
  python crypto_revolut_trader.py --loop           # Auto-loop all assets
  python crypto_revolut_trader.py --dry-run --loop # Signals only
  python crypto_revolut_trader.py --status         # All positions
  python crypto_revolut_trader.py --balance        # Revolut X balance
  python crypto_revolut_trader.py --setup-telegram
  python crypto_revolut_trader.py --reset          # Reset all positions
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
)


# ============================================================
# TRADING CONFIG (per-asset strategies + max positions)
# ============================================================
TRADING_CONFIG_FILE = 'config/trading_config.json'

DEFAULT_TRADING_CONFIG = {
    'BTC': {
        'strategy': 'both_agree',   # BUY when 4h+8h agree
        'max_position_usd': 0,
        'symbol': 'BTC-USD',
        'enabled': True,
    },
    'ETH': {
        'strategy': 'either',       # BUY when either says BUY
        'max_position_usd': 0,
        'symbol': 'ETH-USD',
        'enabled': True,
    },
}

def load_trading_config():
    if os.path.exists(TRADING_CONFIG_FILE):
        with open(TRADING_CONFIG_FILE) as f:
            return json.load(f)
    return dict(DEFAULT_TRADING_CONFIG)

def save_trading_config(config):
    os.makedirs('config', exist_ok=True)
    with open(TRADING_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


# ============================================================
# REVOLUT X API
# ============================================================
REVX_CONFIG_FILE = 'config/revolut_x_config.json'
REVX_BASE_URL = 'https://revx.revolut.com/api/1.0'
PRIVATE_KEY_PATH = 'config/private.pem'

def _load_revx_config():
    if not os.path.exists(REVX_CONFIG_FILE):
        return {'api_key': ''}
    with open(REVX_CONFIG_FILE) as f:
        return json.load(f)

def _load_signing_key():
    from nacl.signing import SigningKey
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    if not Path(PRIVATE_KEY_PATH).exists():
        return None
    pem_data = Path(PRIVATE_KEY_PATH).read_bytes()
    pk = serialization.load_pem_private_key(pem_data, password=None, backend=default_backend())
    raw = pk.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    return SigningKey(raw)

def revx_api(method, path, query='', body=None):
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
        'Accept': 'application/json', 'Content-Type': 'application/json',
        'X-Revx-Api-Key': config['api_key'],
        'X-Revx-Timestamp': ts, 'X-Revx-Signature': sig,
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
    status, data = revx_api('GET', '/balances')
    if status == 200:
        return {b['currency']: {'available': float(b['available']), 'total': float(b['total'])} for b in data}
    return {}

def get_asset_price(symbol):
    """Get price from Revolut X. Handles actual API response formats."""
    # Method 1: Public orderbook — response is {"data": {"asks": [{"p": "67582.90", ...}], "bids": [...]}}
    try:
        url = f"{REVX_BASE_URL}/public/order-book/{symbol}"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = json.loads(resp.read().decode())
        data = raw.get('data', raw)
        asks = data.get('asks', [])
        bids = data.get('bids', [])
        if asks and bids:
            # Fields can be dicts with 'p' key or simple [price, qty] arrays
            if isinstance(asks[0], dict):
                ask = float(asks[0].get('p', 0))
                bid = float(bids[0].get('p', 0))
            else:
                ask = float(asks[0][0])
                bid = float(bids[0][0])
            if ask > 0 and bid > 0:
                return (ask + bid) / 2
    except Exception:
        pass

    # Method 2: Authenticated tickers — can be list of dicts or list of strings
    try:
        status, raw = revx_api('GET', '/tickers')
        if status == 200:
            tickers = raw if isinstance(raw, list) else raw.get('data', raw) if isinstance(raw, dict) else []
            for t in tickers:
                if isinstance(t, dict) and t.get('symbol') == symbol:
                    bid = float(t.get('bid', 0))
                    ask = float(t.get('ask', 0))
                    last = float(t.get('last_price', t.get('p', 0)))
                    if bid > 0 and ask > 0:
                        return (bid + ask) / 2
                    if last > 0:
                        return last
    except Exception:
        pass

    # Method 3: Authenticated last trades — {"data": [{"p": "71191.32", ...}]}
    try:
        status, raw = revx_api('GET', f'/trades/private/{symbol}')
        if status == 200:
            trades = raw.get('data', raw) if isinstance(raw, dict) else raw
            if trades and isinstance(trades, list) and isinstance(trades[0], dict):
                return float(trades[0].get('p', 0))
    except Exception:
        pass

    return 0

def place_market_buy(symbol, quote_size_usd):
    body = {
        'client_order_id': str(uuid.uuid4()),
        'symbol': symbol, 'side': 'BUY',
        'order_configuration': {'market': {'quote_size': str(round(quote_size_usd, 2))}}
    }
    return revx_api('POST', '/orders', body=body)

def place_market_sell(symbol, base_size):
    body = {
        'client_order_id': str(uuid.uuid4()),
        'symbol': symbol, 'side': 'SELL',
        'order_configuration': {'market': {'base_size': str(base_size)}}
    }
    return revx_api('POST', '/orders', body=body)


# ============================================================
# PER-ASSET POSITION STATE
# ============================================================
def _position_file(asset):
    return f'config/position_{asset}.json'

def load_position(asset):
    path = _position_file(asset)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        'asset': asset, 'state': 'cash', 'entry_price': 0,
        'entry_time': '', 'base_amount': 0, 'usd_invested': 0,
        'trades': [], 'auto_trade': False,
    }

def save_position(asset, pos):
    os.makedirs('config', exist_ok=True)
    with open(_position_file(asset), 'w') as f:
        json.dump(pos, f, indent=2)


# ============================================================
# SYNC POSITIONS FROM EXCHANGE
# ============================================================
MIN_POSITION_USD = 5    # Below this = treat as zero (dust)
MIN_TRADE_USD = 300     # Minimum USD to execute a trade

def sync_positions(trading_cfg, notify=True):
    """
    Check actual Revolut X balances and sync local position files.
    - Always updates held amounts from exchange (every 5 min)
    - Detects manual buys/sells (state mismatches)
    - Notifies only on state changes (not on balance updates)
    """
    balances = get_balances()
    if not balances:
        return []

    changes = []
    for asset, cfg in trading_cfg.items():
        if not cfg.get('enabled'):
            continue

        symbol = cfg.get('symbol', f'{asset}-USD')
        pos = load_position(asset)

        # Get actual holdings from exchange
        actual_amount = balances.get(asset, {}).get('total', 0)
        price = get_asset_price(symbol)
        actual_usd = actual_amount * price if price > 0 else 0

        local_state = pos['state']
        updated = False

        if actual_usd > MIN_POSITION_USD and local_state == 'cash':
            # Detected manual BUY — update local state
            pos['state'] = 'invested'
            pos['base_amount'] = actual_amount
            pos['entry_price'] = price  # approximate
            pos['entry_time'] = datetime.now().strftime('%Y-%m-%d %H:%M') + ' (synced)'
            pos['usd_invested'] = actual_usd
            pos['trades'].append({
                'action': 'BUY', 'price': price,
                'time': pos['entry_time'], 'usd': actual_usd,
                'auto': False, 'synced': True,
            })
            updated = True
            changes.append(f"🔄 {asset}: Detected MANUAL BUY — {actual_amount:.6f} {asset} (≈${actual_usd:,.2f})")
            print(f"  🔄 SYNC: {asset} manual BUY detected — {actual_amount:.6f} @ ~${price:,.2f}")

        elif actual_usd <= MIN_POSITION_USD and local_state == 'invested':
            # Detected manual SELL — update local state
            pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * 100 if pos['entry_price'] > 0 else 0
            pnl_usd = pos.get('usd_invested', 0) * pnl_pct / 100
            pos['trades'].append({
                'action': 'SELL', 'price': price,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M') + ' (synced)',
                'pnl_pct': round(pnl_pct, 2), 'pnl_usd': round(pnl_usd, 2),
                'auto': False, 'synced': True,
            })
            pos['state'] = 'cash'
            pos['entry_price'] = 0
            pos['entry_time'] = ''
            pos['base_amount'] = 0
            pos['usd_invested'] = 0
            updated = True
            changes.append(f"🔄 {asset}: Detected MANUAL SELL (PnL: {pnl_pct:+.1f}%)")
            print(f"  🔄 SYNC: {asset} manual SELL detected — PnL: {pnl_pct:+.1f}%")

        elif local_state == 'invested' and actual_amount > 0:
            # Always refresh actual held amount from exchange (silent)
            pos['base_amount'] = actual_amount
            updated = True

        if updated:
            save_position(asset, pos)

    if changes and notify:
        msg = "🔄 <b>Position Sync</b>\n\n" + "\n".join(changes)
        send_telegram(msg)

    return changes


# ============================================================
# STRATEGY: COMPUTE SIGNAL PER ASSET
# ============================================================
def compute_asset_signal(sig_4h, sig_8h, strategy, min_conf=MIN_CONFIDENCE):
    """
    Apply per-asset strategy:
    - 'both_agree': BUY when both agree, SELL when either says SELL
    - 'either': BUY when either says BUY, SELL when either says SELL
    """
    if strategy == 'both_agree':
        return compute_combined_signal(sig_4h, sig_8h, min_conf)

    if sig_4h is None and sig_8h is None:
        return 'HOLD', 50, 'no_signal'

    s4 = sig_4h['signal'] if sig_4h else 'HOLD'
    c4 = sig_4h['confidence'] if sig_4h else 50
    s8 = sig_8h['signal'] if sig_8h else 'HOLD'
    c8 = sig_8h['confidence'] if sig_8h else 50

    # SELL if either says SELL (always)
    if s4 == 'SELL' or s8 == 'SELL':
        if s4 == 'SELL' and s8 == 'SELL':
            return 'SELL', max(c4, c8), 'both_sell'
        elif s4 == 'SELL':
            return 'SELL', c4, '4h_sell'
        else:
            return 'SELL', c8, '8h_sell'

    # 'either' strategy
    if s4 == 'BUY' and c4 >= min_conf:
        return 'BUY', c4, '4h_buy'
    if s8 == 'BUY' and c8 >= min_conf:
        return 'BUY', c8, '8h_buy'
    if s4 == 'BUY' or s8 == 'BUY':
        return 'HOLD', max(c4, c8), 'low_conf'

    return 'HOLD', 50, 'no_signal'


# ============================================================
# PROCESS ONE ASSET
# ============================================================
def process_asset(asset, trading_cfg, dry_run=False):
    """Generate signals and execute for one asset. Returns result dict."""
    position = load_position(asset)
    strategy = trading_cfg.get('strategy', 'both_agree')
    symbol = trading_cfg.get('symbol', f'{asset}-USD')
    max_usd = trading_cfg.get('max_position_usd', 0)

    config_4h = load_best_config(asset, horizon=4)
    config_8h = load_best_config(asset, horizon=8)
    if not config_4h and not config_8h:
        return None

    # Download data
    try:
        download_asset(asset, update_only=True)
    except Exception:
        pass

    df_raw = load_data(asset)
    if df_raw is None:
        return None

    # Generate signals
    sig_4h = sig_8h = None
    if config_4h:
        sig_4h = generate_live_signal(asset, config_4h, df_raw=df_raw)
    if config_8h:
        sig_8h = generate_live_signal(asset, config_8h, df_raw=df_raw)

    # Apply asset-specific strategy (use per-asset min_confidence if set, else global)
    min_conf = trading_cfg.get('min_confidence', MIN_CONFIDENCE)
    action, confidence, reason = compute_asset_signal(sig_4h, sig_8h, strategy, min_conf=min_conf)
    any_sig = sig_4h or sig_8h
    price = any_sig['close'] if any_sig else 0

    # Reload position (sync may have updated it)
    position = load_position(asset)

    # Log
    s4_str = f"{sig_4h['signal']}({sig_4h['confidence']:.0f}%)" if sig_4h else "N/A"
    s8_str = f"{sig_8h['signal']}({sig_8h['confidence']:.0f}%)" if sig_8h else "N/A"
    print(f"  {asset}: 4h={s4_str} | 8h={s8_str} → {action} ({confidence:.0f}%) [{reason}] | pos={position['state']}")

    # Execute
    executed = False
    pnl_msg = ""

    if action == 'BUY' and position['state'] == 'cash' and max_usd > 0:
        if not dry_run and position.get('auto_trade'):
            # Pre-trade balance check
            balances = get_balances()
            usd_avail = balances.get('USD', {}).get('available', 0)
            print(f"    Pre-trade USD: ${usd_avail:,.2f}")

            # Minimum $300 to trade, target full max, or all available if < max
            if usd_avail < MIN_TRADE_USD:
                send_telegram(f"⚠️ {asset} BUY skipped — ${usd_avail:.2f} < ${MIN_TRADE_USD} minimum")
            else:
                buy_amount = min(max_usd, usd_avail)
                if buy_amount < max_usd:
                    print(f"    Partial buy: ${buy_amount:,.2f} of ${max_usd:,.2f} (limited by balance)")
                status, data = place_market_buy(symbol, buy_amount)
                if status in (200, 201):
                    order = data.get('data', data)
                    position['base_amount'] = float(order.get('filled_size', buy_amount / price))
                    position['entry_price'] = float(order.get('average_fill_price', price))
                    position['usd_invested'] = buy_amount
                    executed = True

                    # Post-trade verification
                    time.sleep(2)
                    post_bal = get_balances()
                    crypto_held = post_bal.get(asset, {}).get('total', 0)
                    usd_left = post_bal.get('USD', {}).get('available', 0)
                    print(f"    Post-trade: {crypto_held:.6f} {asset} | ${usd_left:,.2f} USD")
                else:
                    send_telegram(f"⚠️ {asset} BUY failed: {status} {data}")

        if not executed and not dry_run and not position.get('auto_trade'):
            # Manual mode — just track
            position['base_amount'] = max_usd / price
            position['entry_price'] = price
            position['usd_invested'] = max_usd

        if executed or (not dry_run and not position.get('auto_trade')):
            position['state'] = 'invested'
            position['entry_time'] = datetime.now().strftime('%Y-%m-%d %H:%M')
            if not executed:
                position['usd_invested'] = max_usd
            position['trades'].append({
                'action': 'BUY', 'price': position['entry_price'],
                'time': position['entry_time'], 'usd': position['usd_invested'],
                'auto': executed,
            })
            save_position(asset, position)

    elif action == 'SELL' and position['state'] == 'invested':
        if not dry_run and position.get('auto_trade'):
            # Get ACTUAL holdings from exchange — sell everything
            balances = get_balances()
            actual_held = balances.get(asset, {}).get('available', 0)
            print(f"    Pre-trade: {actual_held:.6f} {asset} (selling ALL)")

            if actual_held > 0:
                status, data = place_market_sell(symbol, actual_held)
                if status in (200, 201):
                    order = data.get('data', data)
                    price = float(order.get('average_fill_price', price))
                    executed = True

                    # Post-trade verification
                    time.sleep(2)
                    post_bal = get_balances()
                    usd_now = post_bal.get('USD', {}).get('available', 0)
                    crypto_left = post_bal.get(asset, {}).get('total', 0)
                    print(f"    Post-trade: {crypto_left:.6f} {asset} | ${usd_now:,.2f} USD")
                else:
                    send_telegram(f"⚠️ {asset} SELL failed: {status} {data}")
            else:
                print(f"    ⚠ Nothing to sell (exchange balance = 0)")

        pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100 if position['entry_price'] > 0 else 0
        pnl_usd = position['usd_invested'] * pnl_pct / 100
        pnl_msg = f" | PnL: {pnl_pct:+.1f}% (${pnl_usd:+,.2f})"

        position['trades'].append({
            'action': 'SELL', 'price': price,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'pnl_pct': round(pnl_pct, 2), 'pnl_usd': round(pnl_usd, 2),
            'auto': executed,
        })
        position['state'] = 'cash'
        position['entry_price'] = 0
        position['entry_time'] = ''
        position['base_amount'] = 0
        position['usd_invested'] = 0
        save_position(asset, position)

    return {
        'asset': asset, 'action': action, 'confidence': confidence,
        'reason': reason, 'price': price, 'executed': executed,
        'pnl_msg': pnl_msg, 'sig_4h': sig_4h, 'sig_8h': sig_8h,
        'position': position, 'strategy': strategy,
    }


# ============================================================
# TELEGRAM MESSAGE (MULTI-ASSET)
# ============================================================
def format_multi_asset_telegram(results, dry_run=False, balances=None):
    """Format combined Telegram message for all assets."""
    if balances is None:
        balances = {}
    lines = []
    mode = "DRY" if dry_run else "LIVE"
    lines.append(f"📊 <b>Hourly Update [{mode}]</b>")
    lines.append(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    for r in results:
        if r is None:
            continue
        asset = r['asset']
        action = r['action']
        conf = r['confidence']
        price = r['price']
        position = r['position']
        strategy = r['strategy']

        # Get RSI + last 4 candles from data
        rsi = 0
        last_prices = []
        try:
            any_sig = r.get('sig_4h') or r.get('sig_8h')
            if any_sig:
                rsi = any_sig.get('rsi', 0)
                for h in any_sig.get('last_4h', []):
                    last_prices.append((h['datetime'], h['close']))
        except Exception:
            pass

        # Exchange holdings
        actual_held = balances.get(asset, {}).get('total', 0)
        actual_usd = actual_held * price if price > 0 and actual_held > 0 else 0

        # Signal emojis
        def _se(sig, h):
            if not sig: return f"{h}h:N/A"
            e = '🔵' if sig['signal'] == 'BUY' else '🔴' if sig['signal'] == 'SELL' else '🟡'
            return f"{e}{h}h:{sig['signal']}({sig['confidence']:.0f}%)"

        sig_line = f"{_se(r['sig_4h'], 4)} {_se(r['sig_8h'], 8)}"

        # Action
        if action == 'BUY' and position['state'] == 'invested':
            if r['executed']:
                act = f"✅ BOUGHT ${position.get('usd_invested', 0):,.0f}"
            else:
                act = f"🔵 BUY ${position.get('usd_invested', 0):,.0f}"
        elif action == 'SELL' and r['pnl_msg']:
            if r['executed']:
                act = f"🚨 SOLD{r['pnl_msg']}"
            else:
                act = f"🔴 SELL{r['pnl_msg']}"
        elif position['state'] == 'invested':
            cur_pnl = (price - position['entry_price']) / position['entry_price'] * 100 if position['entry_price'] > 0 else 0
            act = f"⏸ HOLD ({cur_pnl:+.1f}%)"
        else:
            act = f"⏸ HOLD (cash)"

        # Header with price + RSI + auto status
        price_str = f"${price:,.2f}" if price >= 100 else f"${price:,.4f}"
        auto_icon = "🔵" if position.get('auto_trade') else "🔴"
        lines.append(f"{auto_icon} <b>{asset}</b> {price_str} | RSI:{rsi:.0f} | [{strategy}]")

        # Last 4 prices
        if last_prices:
            p_parts = []
            for dt_str, p in last_prices:
                p_str = f"${p:,.0f}" if p >= 100 else f"${p:,.4f}"
                p_parts.append(f"{dt_str}:{p_str}")
            lines.append(f"  <code>{' '.join(p_parts)}</code>")

        # Signals + action
        lines.append(f"  {sig_line}")
        lines.append(f"  {act}")

        # Position details if invested
        if position['state'] == 'invested' and position['entry_price'] > 0:
            cur_pnl = (price - position['entry_price']) / position['entry_price'] * 100
            cur_usd = position.get('usd_invested', 0) * (1 + cur_pnl / 100)
            emoji = '📈' if cur_pnl > 0 else '📉'
            lines.append(f"  📦 ${position.get('usd_invested',0):,.0f} @ ${position['entry_price']:,.2f} → ${cur_usd:,.0f} ({emoji}{cur_pnl:+.1f}%)")

        # Exchange balance (always show if holding)
        if actual_held > 0 and actual_usd > 5:
            lines.append(f"  💼 Exchange: {actual_held:.6f} {asset} (≈${actual_usd:,.2f})")
        elif position['state'] == 'invested' and actual_held == 0:
            lines.append(f"  ⚠️ Tracker says invested but exchange shows 0!")

        lines.append("")

    # USD balance
    usd_avail = balances.get('USD', {}).get('available', 0)
    if usd_avail > 0 or balances:
        lines.append(f"💵 USD: ${usd_avail:,.2f}")

    return "\n".join(lines)


# ============================================================
# TELEGRAM COMMAND LISTENER
# ============================================================
_last_update_id = 0

def check_telegram_commands():
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
            text = update.get('message', {}).get('text', '').strip().lower()
            if text in ('/stop', '/status', '/pause', '/resume', '/balance', '/sync'):
                return text
    except Exception:
        pass
    return None

def _flush_old_updates():
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

def _handle_status_command():
    trading_cfg = load_trading_config()
    balances = get_balances()
    lines = [f"📊 <b>Status</b> — {datetime.now().strftime('%H:%M')}\n"]

    for asset, cfg in trading_cfg.items():
        if not cfg.get('enabled'):
            continue
        pos = load_position(asset)
        symbol = cfg.get('symbol', f'{asset}-USD')
        strategy = cfg.get('strategy', 'both_agree')

        # Get live price from exchange
        price = get_asset_price(symbol)
        actual_held = balances.get(asset, {}).get('total', 0)
        actual_usd = actual_held * price if price > 0 else 0

        # Get last 4 hours + RSI from data
        last_prices = []
        rsi = 0
        try:
            df_raw = load_data(asset)
            if df_raw is not None and len(df_raw) >= 4:
                recent = df_raw.tail(4)
                for _, row in recent.iterrows():
                    dt = row['datetime']
                    if hasattr(dt, 'strftime'):
                        dt_str = dt.strftime('%H:%M')
                    else:
                        dt_str = str(dt)[-5:]
                    last_prices.append((dt_str, float(row['close'])))

                # Quick RSI calc from last 15 candles
                if len(df_raw) >= 15:
                    closes = df_raw['close'].tail(15).values.astype(float)
                    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                    gains = [d for d in deltas if d > 0]
                    losses = [-d for d in deltas if d < 0]
                    avg_gain = sum(gains) / 14 if gains else 0.001
                    avg_loss = sum(losses) / 14 if losses else 0.001
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
        except Exception:
            pass

        # Header
        price_str = f"${price:,.2f}" if price >= 100 else f"${price:,.4f}"
        lines.append(f"<b>{asset}</b> {price_str} | RSI: {rsi:.0f} | [{strategy}]")

        # Last 4 prices
        if last_prices:
            price_line = "  "
            for dt_str, p in last_prices:
                p_str = f"${p:,.2f}" if p >= 100 else f"${p:,.4f}"
                price_line += f"{dt_str}:{p_str}  "
            lines.append(f"<code>{price_line.strip()}</code>")

        # Position info
        if pos['state'] == 'invested' and pos['entry_price'] > 0:
            cur_pnl = (price - pos['entry_price']) / pos['entry_price'] * 100 if price > 0 else 0
            cur_value = actual_usd if actual_usd > 0 else pos.get('usd_invested', 0) * (1 + cur_pnl / 100)
            emoji = '📈' if cur_pnl > 0 else '📉'
            lines.append(f"  {emoji} INVESTED ${pos.get('usd_invested',0):,.0f} → ${cur_value:,.0f} ({cur_pnl:+.1f}%)")
            lines.append(f"  Entry: ${pos['entry_price']:,.2f} | {pos.get('entry_time', '')}")
            if actual_held > 0:
                lines.append(f"  Held: {actual_held:.6f} {asset}")
        else:
            lines.append(f"  💵 CASH (max ${cfg.get('max_position_usd',0):,.0f})")
            if actual_held > 0 and actual_usd > 5:
                lines.append(f"  ⚠️ Exchange: {actual_held:.6f} {asset} (≈${actual_usd:,.2f})")

        # Trade history summary
        sells = [t for t in pos.get('trades', []) if t.get('action') == 'SELL']
        if sells:
            total = sum(t.get('pnl_usd', 0) for t in sells)
            wins = sum(1 for t in sells if t.get('pnl_usd', 0) > 0)
            lines.append(f"  📊 {len(sells)} trades ({wins}W) ${total:+,.2f}")
        lines.append("")

    usd_avail = balances.get('USD', {}).get('available', 0)
    lines.append(f"💵 USD: ${usd_avail:,.2f}")
    send_telegram("\n".join(lines))


# ============================================================
# MAIN LOOP
# ============================================================
def run_all_once(trading_cfg, dry_run=False):
    """Sync positions, process all enabled assets."""
    print(f"\n{'='*60}")
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"  [{mode}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Sync positions from exchange FIRST
    balances = {}
    if not dry_run:
        print("\n  Syncing positions from Revolut X...")
        sync_positions(trading_cfg)
        balances = get_balances()

    results = []
    for asset, cfg in trading_cfg.items():
        if not cfg.get('enabled'):
            continue
        if not load_best_config(asset, horizon=4) and not load_best_config(asset, horizon=8):
            continue
        print(f"\n  --- {asset} ({cfg['strategy']}) ---")
        r = process_asset(asset, cfg, dry_run=dry_run)
        results.append(r)

    # Send combined Telegram
    valid = [r for r in results if r is not None]
    if valid:
        msg = format_multi_asset_telegram(valid, dry_run=dry_run, balances=balances)
        send_telegram(msg)

    return results


def run_loop(trading_cfg, dry_run=False):
    mode = "DRY RUN" if dry_run else "LIVE"
    assets_str = ", ".join(a for a, c in trading_cfg.items() if c.get('enabled'))

    print(f"\n{'='*60}")
    print(f"  REVOLUT X MULTI-ASSET TRADER [{mode}]")
    print(f"  Assets: {assets_str}")
    for asset, cfg in trading_cfg.items():
        if cfg.get('enabled'):
            pos = load_position(asset)
            auto = "AUTO" if pos.get('auto_trade') else "MANUAL"
            print(f"  {asset}: {cfg['strategy']} | max=${cfg['max_position_usd']:,.0f} | {auto} | {pos['state'].upper()}")
    print(f"  Min confidence: {MIN_CONFIDENCE}% (global default — overridden per asset by Mode F)")
    print(f"  Telegram: /stop /status /pause /resume /balance /sync")
    print(f"  Position sync: every 5 min (detects manual trades)")
    print(f"{'='*60}")

    _flush_old_updates()

    # Initial sync — detect any existing positions on exchange
    print("\n  Initial position sync from Revolut X...")
    changes = sync_positions(trading_cfg, notify=True)
    if not changes:
        print("  Positions in sync ✓")

    # Build per-asset status lines for Telegram
    asset_lines = []
    for a, c in trading_cfg.items():
        if not c.get('enabled'):
            continue
        pos = load_position(a)
        auto = pos.get('auto_trade', False)
        icon = "🔵 ON" if auto else "🔴 OFF"
        asset_lines.append(f"  {a}: {c['strategy']} | ${c['max_position_usd']:,.0f} | {icon}")

    send_telegram(
        f"🚀 <b>Multi-Asset Trader Started</b>\n\n"
        + "\n".join(asset_lines) + "\n\n"
        f"BUY needs ≥{trading_cfg.get(next(iter(trading_cfg)), {}).get('''min_confidence''', MIN_CONFIDENCE) if trading_cfg else MIN_CONFIDENCE}% model confidence (per-asset)\n"
        f"SELL when either model says SELL\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"📱 /stop /status /pause /resume /balance /sync"
    )

    # Immediate first scan — don't wait for next hour
    print("\n  Running initial signal scan...")
    run_all_once(trading_cfg, dry_run=dry_run)

    cycle = 0
    paused = False
    while True:
        try:
            now = datetime.now()
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            wait_sec = (next_hour - now).total_seconds()

            if wait_sec > 5:
                # Build status string
                parts = []
                for a, c in trading_cfg.items():
                    if not c.get('enabled'): continue
                    p = load_position(a)
                    if p['state'] == 'invested':
                        parts.append(f"{a}:IN")
                    else:
                        parts.append(f"{a}:CASH")
                status = " | ".join(parts)
                pause_str = " [PAUSED]" if paused else ""
                print(f"\n  [{status}{pause_str}] Next: {next_hour.strftime('%H:%M')} ({wait_sec/60:.0f} min)")

                remaining = wait_sec
                last_sync = time.time()
                while remaining > 0:
                    sleep_chunk = min(30, remaining)
                    time.sleep(sleep_chunk)
                    remaining -= sleep_chunk

                    # Sync positions every 5 minutes
                    if time.time() - last_sync >= 300:
                        try:
                            changes = sync_positions(trading_cfg, notify=True)
                            if changes:
                                print(f"  🔄 Position sync: {len(changes)} changes")
                            last_sync = time.time()
                        except Exception as e:
                            print(f"  Sync error: {e}")

                    # Check Telegram commands
                    cmd = check_telegram_commands()
                    if cmd == '/stop':
                        print("\n  🛑 STOP via Telegram")
                        send_telegram("🛑 <b>Trader Stopped</b>")
                        return
                    elif cmd == '/status':
                        _handle_status_command()
                    elif cmd == '/pause':
                        paused = True
                        send_telegram("⏸ <b>PAUSED</b> — /resume to continue")
                    elif cmd == '/resume':
                        paused = False
                        send_telegram("▶️ <b>RESUMED</b>")
                    elif cmd == '/balance':
                        bal = get_balances()
                        bl = [f"  {c}: {b['available']:.6f}" for c, b in sorted(bal.items()) if b['total'] > 0]
                        send_telegram("💰 <b>Balance</b>\n" + "\n".join(bl))
                    elif cmd == '/sync':
                        sync_positions(trading_cfg, notify=True)
                        send_telegram("🔄 <b>Synced</b>")

            cycle += 1
            print(f"\n  --- Cycle {cycle} | {datetime.now().strftime('%H:%M:%S')} ---")

            # Wait for fresh candle (use first enabled asset)
            first_asset = next(a for a, c in trading_cfg.items() if c.get('enabled'))
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            expected = now_utc.replace(minute=0, second=0, microsecond=0)
            wait_for_fresh_candle(first_asset, expected)

            if paused:
                print("  ⏸ PAUSED — signals only")
                run_all_once(trading_cfg, dry_run=True)
            else:
                run_all_once(trading_cfg, dry_run=dry_run)

        except KeyboardInterrupt:
            print("\n  Stopped.")
            send_telegram("🛑 <b>Stopped</b>")
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
    print("  REVOLUT X MULTI-ASSET TRADER")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    loop_mode = False
    dry_run = False

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--loop': loop_mode = True
        elif arg == '--dry-run': dry_run = True
        elif arg == '--setup-telegram': setup_telegram(); return
        elif arg == '--balance':
            bal = get_balances()
            if bal:
                print("\n  Revolut X Balances:")
                for c, b in sorted(bal.items()):
                    if b['total'] > 0:
                        print(f"    {c:6s}  {b['available']:>14.6f}")
            return
        elif arg == '--status':
            trading_cfg = load_trading_config()
            for asset, cfg in trading_cfg.items():
                pos = load_position(asset)
                print(f"\n  {asset} [{cfg['strategy']}]:")
                print(f"    State: {pos['state'].upper()} | Max: ${cfg['max_position_usd']:,.2f}")
                print(f"    Auto-trade: {pos.get('auto_trade', False)}")
                if pos['state'] == 'invested':
                    print(f"    Entry: ${pos['entry_price']:,.2f} at {pos['entry_time']}")
                sells = [t for t in pos.get('trades', []) if t.get('action') == 'SELL']
                if sells:
                    total = sum(t.get('pnl_usd', 0) for t in sells)
                    print(f"    Trades: {len(sells)} | PnL: ${total:+,.2f}")
            return
        elif arg == '--reset':
            for a in ['BTC', 'ETH', 'XRP', 'DOGE']:
                p = _position_file(a)
                if os.path.exists(p): os.remove(p)
            if os.path.exists(TRADING_CONFIG_FILE): os.remove(TRADING_CONFIG_FILE)
            print("  All positions and config reset.")
            return

    # Load or create trading config
    trading_cfg = load_trading_config()

    if not os.path.exists('models/crypto_hourly_best_models.csv'):
        print("\n  ERROR: No models found! Run crypto_trading_system.py Mode D first.")
        return

    # Interactive menu
    if len(args) == 0:
        print("\n  Current config:")
        for asset, cfg in trading_cfg.items():
            pos = load_position(asset)
            has_model = load_best_config(asset, horizon=4) or load_best_config(asset, horizon=8)
            status = "✓" if has_model else "✗"
            auto = "AUTO" if pos.get('auto_trade') else "MANUAL"
            print(f"    {status} {asset}: {cfg['strategy']} | max=${cfg['max_position_usd']:,.0f} | {auto} | {pos['state'].upper()}")

        print(f"\n  1. Run once (all assets)")
        print(f"  2. Loop (hourly)")
        print(f"  3. Dry run (once)")
        print(f"  4. Configure assets (max positions, enable/disable)")
        print(f"  5. Toggle auto-trade per asset")
        print(f"  6. View status / history")
        print(f"  7. Check balance")
        print(f"  8. Setup Telegram")
        ch = input("\n  Enter 1-8: ").strip()

        if ch == '8': setup_telegram(); return
        elif ch == '7':
            bal = get_balances()
            for c, b in sorted(bal.items()):
                if b['total'] > 0: print(f"    {c}: {b['available']:.6f}")
            return
        elif ch == '6':
            for asset in trading_cfg:
                pos = load_position(asset)
                trades = pos.get('trades', [])[-5:]
                if trades:
                    print(f"\n  {asset} (last 5):")
                    for t in trades:
                        if t['action'] == 'BUY':
                            print(f"    🔵 BUY  ${t['price']:,.2f} | {t['time']}")
                        else:
                            print(f"    🔴 SELL ${t['price']:,.2f} | {t['time']} | {t.get('pnl_pct',0):+.1f}%")
            return
        elif ch == '5':
            for asset in trading_cfg:
                pos = load_position(asset)
                current = pos.get('auto_trade', False)
                resp = input(f"  {asset} auto-trade [{current}] → toggle? (y/n): ").strip().lower()
                if resp == 'y':
                    pos['auto_trade'] = not current
                    save_position(asset, pos)
                    print(f"    {asset}: {'ENABLED' if pos['auto_trade'] else 'DISABLED'}")
            return
        elif ch == '4':
            for asset in trading_cfg:
                cfg = trading_cfg[asset]
                print(f"\n  {asset}:")
                en = input(f"    Enabled [{cfg.get('enabled', True)}]? (y/n/skip): ").strip().lower()
                if en == 'y': cfg['enabled'] = True
                elif en == 'n': cfg['enabled'] = False

                strat = input(f"    Strategy [{cfg['strategy']}] (both_agree/either/skip): ").strip().lower()
                if strat in ('both_agree', 'either'): cfg['strategy'] = strat

                max_inp = input(f"    Max USD [{cfg['max_position_usd']:.0f}] (number/skip): ").strip()
                if max_inp:
                    try:
                        cfg['max_position_usd'] = float(max_inp)
                    except ValueError:
                        pass

            save_trading_config(trading_cfg)
            print("\n  Config saved!")
            return
        elif ch == '3': dry_run = True
        elif ch == '2': loop_mode = True

    # Ensure max positions are set
    needs_config = False
    for asset, cfg in trading_cfg.items():
        if cfg.get('enabled') and cfg['max_position_usd'] <= 0:
            has_model = load_best_config(asset, horizon=4) or load_best_config(asset, horizon=8)
            if has_model:
                needs_config = True
                break

    if needs_config:
        print(f"\n  ⚠️ Max positions not set. Configure now:")
        for asset, cfg in trading_cfg.items():
            if not cfg.get('enabled'): continue
            has_model = load_best_config(asset, horizon=4) or load_best_config(asset, horizon=8)
            if not has_model: continue
            if cfg['max_position_usd'] <= 0:
                while True:
                    try:
                        val = float(input(f"  {asset} max position USD: $").strip())
                        if val > 0:
                            cfg['max_position_usd'] = val
                            break
                    except ValueError:
                        pass
        save_trading_config(trading_cfg)

    if loop_mode:
        run_loop(trading_cfg, dry_run=dry_run)
    else:
        run_all_once(trading_cfg, dry_run=dry_run)


if __name__ == '__main__':
    main()
