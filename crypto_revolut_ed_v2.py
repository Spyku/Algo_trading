"""
Revolut X Multi-Asset Auto-Trader (Ed V2)
==========================================
Regime-aware auto-trader using Ed models (Optuna + XGBoost).
Reads from regime_config_ed.json for per-asset regime detection and horizon selection.
Supports maker (limit) orders with 0% fee alongside taker (market) orders.

Usage:
  python crypto_revolut_ed_v2.py                  # Interactive
  python crypto_revolut_ed_v2.py --loop           # Auto-loop all assets
  python crypto_revolut_ed_v2.py --dry-run --loop # Signals only
  python crypto_revolut_ed_v2.py --status         # All positions
  python crypto_revolut_ed_v2.py --balance        # Revolut X balance
  python crypto_revolut_ed_v2.py --setup-telegram
  python crypto_revolut_ed_v2.py --reset          # Reset all positions
"""

import os
import sys
import time
import math
import json
import uuid
import base64
import ssl
import urllib.request
import urllib.error
import pandas as pd
import numpy as np
import threading
from pathlib import Path
from datetime import datetime, timedelta, timezone

os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')

_ssl_ctx = ssl._create_unverified_context()

from crypto_live_trader_ed import (
    load_best_config, generate_live_signal, compute_combined_signal,
    send_telegram, wait_for_fresh_candle, setup_telegram,
    download_asset, load_data, TELEGRAM_CONFIG, MIN_CONFIDENCE,
    MODELS_CSV, HORIZON_SHORT, HORIZON_LONG, AVAILABLE_HORIZONS,
    generate_regime_signal, detect_regime, load_regime_config,
)


# ============================================================
# TRADING CONFIG (per-asset strategies + max positions)
# ============================================================
REGIME_CONFIG_FILE = 'config/regime_config_ed.json'

DEFAULT_TRADING_CONFIG = {
    'BTC': {
        'enabled': True,
        'symbol': 'BTC-USD',
        'regime_detector': {'type': 'sma_cross', 'params': {'fast': 48, 'slow': 200}},
        'bull': {'horizon': 7, 'min_confidence': 95, 'max_position_usd': 12000},
        'bear': {'horizon': 8, 'min_confidence': 90, 'max_position_usd': 6000},
        'take_profit_pct': 0,
    },
    'ETH': {
        'enabled': True,
        'symbol': 'ETH-USD',
        'regime_detector': {'type': 'sma_cross', 'params': {'fast': 48, 'slow': 200}},
        'bull': {'horizon': 7, 'min_confidence': 95, 'max_position_usd': 12000},
        'bear': {'horizon': 8, 'min_confidence': 90, 'max_position_usd': 6000},
        'take_profit_pct': 0,
    },
}

def load_trading_config():
    defaults = json.loads(json.dumps(DEFAULT_TRADING_CONFIG))
    if os.path.exists(REGIME_CONFIG_FILE):
        with open(REGIME_CONFIG_FILE) as f:
            file_cfg = json.load(f)
        # Merge file config over defaults
        for asset in file_cfg:
            if asset not in defaults:
                defaults[asset] = {
                    'enabled': True, 'symbol': f'{asset}-USD',
                    'regime_detector': {'type': 'sma_cross', 'params': {'fast': 48, 'slow': 200}},
                    'bull': {'horizon': 7, 'min_confidence': 95, 'max_position_usd': 12000},
                    'bear': {'horizon': 8, 'min_confidence': 90, 'max_position_usd': 6000},
                }
            defaults[asset].update(file_cfg[asset])
    return defaults

def save_trading_config(cfg):
    os.makedirs('config', exist_ok=True)
    with open(REGIME_CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)


# ============================================================
# REVOLUT X API
# ============================================================
REVX_CONFIG_FILE = 'config/revolut_x_config.json'
REVX_BASE_URL = 'https://revx.revolut.com/api/1.0'
PRIVATE_KEY_PATH = 'config/private.pem'

def _load_revx_config():
    # Env var takes precedence over JSON file
    api_key = os.environ.get('REVX_API_KEY', '')
    if api_key:
        return {'api_key': api_key}
    if not os.path.exists(REVX_CONFIG_FILE):
        return {'api_key': ''}
    with open(REVX_CONFIG_FILE) as f:
        return json.load(f)

def _load_signing_key():
    from nacl.signing import SigningKey
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    # Try env var first (base64-encoded PEM)
    pem_b64 = os.environ.get('REVX_PRIVATE_KEY_B64', '')
    if pem_b64:
        pem_data = base64.b64decode(pem_b64)
    elif Path(PRIVATE_KEY_PATH).exists():
        pem_data = Path(PRIVATE_KEY_PATH).read_bytes()
    else:
        return None
    pk = serialization.load_pem_private_key(pem_data, password=None, backend=default_backend())
    raw = pk.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    return SigningKey(raw)

_clock_offset_ms = 0  # Adjusted if local clock drifts from server

def _sync_clock_ntp():
    """Get clock offset vs NTP. Returns offset in ms (negative = local clock is ahead)."""
    import socket, struct
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(3)
        data = b'\x1b' + 47 * b'\0'
        client.sendto(data, ('pool.ntp.org', 123))
        data, _ = client.recvfrom(1024)
        ntp_time = struct.unpack('!12I', data)[10] - 2208988800
        offset = (ntp_time * 1000) - int(time.time() * 1000)
        return offset
    except Exception:
        return 0

def revx_api(method, path, query='', body=None):
    global _clock_offset_ms
    config = _load_revx_config()
    sk = _load_signing_key()
    if not sk or not config.get('api_key'):
        return 0, {'error': 'API not configured'}
    body_str = json.dumps(body, separators=(',', ':')) if body else ''
    full_path = f"/api/1.0{path}"
    ts = str(int(time.time() * 1000) + _clock_offset_ms)
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
        with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode())
        except:
            err_body = {'error': str(e)}
        # Auto-correct clock drift on 409 "timestamp in the future"
        if e.code == 409 and 'timestamp' in str(err_body):
            ntp_offset = _sync_clock_ntp()
            if ntp_offset != 0:
                _clock_offset_ms = ntp_offset
                print(f"    🕐 Clock drift detected: {_clock_offset_ms:+d}ms — corrected via NTP")
            else:
                _clock_offset_ms -= 2000
                print(f"    🕐 Clock drift: NTP unreachable, nudging offset to {_clock_offset_ms:+d}ms")
        return e.code, err_body
    except Exception as e:
        return 0, {'error': str(e)}

def get_balances(retries=3):
    for attempt in range(1, retries + 1):
        status, data = revx_api('GET', '/balances')
        if status == 200:
            result = {b['currency']: {'available': float(b['available']), 'total': float(b['total'])} for b in data}
            # Log any asset where available != total (funds locked by open order)
            for cur, bal in result.items():
                if bal['total'] > 0 and bal['available'] != bal['total']:
                    print(f"    ⚠ {cur} balance: available={bal['available']:.6f} != total={bal['total']:.6f} (funds locked?)")
            return result
        if attempt < retries:
            print(f"    ⚠ get_balances failed (status={status}, data={data}, attempt {attempt}/{retries}) — retrying in 3s...")
            time.sleep(3)
    print(f"    ❌ get_balances failed after {retries} attempts (last status={status}, data={data})")
    return None

def get_asset_price(symbol):
    """Get price from Revolut X. Handles actual API response formats."""
    # Method 1: Public orderbook — response is {"data": {"asks": [{"p": "67582.90", ...}], "bids": [...]}}
    try:
        url = f"{REVX_BASE_URL}/public/order-book/{symbol}"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx) as resp:
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
# LIMIT / MAKER ORDER SUPPORT
# ============================================================
def place_limit_buy(symbol, quote_size_usd, price):
    """Place a post-only limit buy order (maker, 0% fee)."""
    # Calculate base_size from quote_size and price
    base_size = quote_size_usd / price
    body = {
        'client_order_id': str(uuid.uuid4()),
        'symbol': symbol, 'side': 'BUY',
        'order_configuration': {
            'limit': {
                'base_size': f'{base_size:.8f}',
                'price': f'{price:.2f}',
                'execution_instructions': ['post_only'],
            }
        }
    }
    return revx_api('POST', '/orders', body=body)


def place_limit_sell(symbol, base_size, price):
    """Place a post-only limit sell order (maker, 0% fee)."""
    body = {
        'client_order_id': str(uuid.uuid4()),
        'symbol': symbol, 'side': 'SELL',
        'order_configuration': {
            'limit': {
                'base_size': str(base_size),
                'price': f'{price:.2f}',
                'execution_instructions': ['post_only'],
            }
        }
    }
    return revx_api('POST', '/orders', body=body)


def get_order_status(venue_order_id):
    """Check status of an order."""
    return revx_api('GET', f'/orders/{venue_order_id}')


def cancel_order(venue_order_id):
    """Cancel an active order."""
    return revx_api('DELETE', f'/orders/{venue_order_id}')


def cancel_all_open_orders(symbol=None):
    """Cancel all open orders, optionally filtered by symbol. Returns count cancelled."""
    status, data = revx_api('GET', '/orders/active')
    if status != 200:
        print(f"    [!] Failed to list open orders (status={status})")
        return 0
    orders = data if isinstance(data, list) else data.get('data', data) if isinstance(data, dict) else []
    if not isinstance(orders, list):
        return 0
    cancelled = 0
    for o in orders:
        if not isinstance(o, dict):
            continue
        order_status = o.get('status', o.get('state', ''))
        if order_status not in ('open', 'pending', 'partially_filled', 'new'):
            continue
        if symbol and o.get('symbol', '') != symbol:
            continue
        oid = o.get('venue_order_id', o.get('id', ''))
        if oid:
            cancel_order(oid)
            print(f"    Cancelled order {oid} ({o.get('side', '?')} {o.get('symbol', '?')})")
            cancelled += 1
    return cancelled


def get_best_bid_ask(symbol):
    """Get best bid and ask from order book."""
    try:
        url = f"{REVX_BASE_URL}/public/order-book/{symbol}"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx) as resp:
            raw = json.loads(resp.read().decode())
        data = raw.get('data', raw)
        asks = data.get('asks', [])
        bids = data.get('bids', [])
        if asks and bids:
            if isinstance(asks[0], dict):
                ask = float(asks[0].get('p', 0))
                bid = float(bids[0].get('p', 0))
            else:
                ask = float(asks[0][0])
                bid = float(bids[0][0])
            return bid, ask
    except Exception:
        pass
    return 0, 0


def _execute_maker_order(symbol, size, side, maker_window=120, check_interval=3):
    """Try maker order, market fallback after maker_window.

    Buy at bid+0.01 — top of bid queue, filled by incoming market sells.
    Sell: starts at ask-0.01, steps down toward bid+0.01 each retry (20 attempts).
    Progressively undercuts to maximise fill chance while saving fees.
    Returns: (status, order_data) same format as place_market_buy/sell
    """
    is_buy = (side == 'buy')
    place_limit = place_limit_buy if is_buy else place_limit_sell
    place_market = place_market_buy if is_buy else place_market_sell
    side_label = 'buy' if is_buy else 'sell'
    max_attempts = maker_window // check_interval

    # Cancel any stale open orders for this symbol to free locked funds
    stale = cancel_all_open_orders(symbol)
    if stale > 0:
        print(f"    Cleared {stale} stale order(s) for {symbol}")
        time.sleep(1)

    elapsed = 0
    attempt = 0

    while elapsed < maker_window:
        attempt += 1
        bid, ask = get_best_bid_ask(symbol)
        if bid <= 0 or ask <= 0:
            print(f"    [!] Cannot get price for {symbol}, going market")
            return place_market(symbol, size)

        spread_bps = (ask - bid) / bid * 10000
        if is_buy:
            limit_price = round(bid + 0.01, 2)
            if limit_price >= ask:
                limit_price = bid
        else:
            # BUG 2 fix: SELL post_only requires price STRICTLY above best bid.
            # Slide from ask-0.01 down to bid+0.02 (2 cents above bid for tick-race safety).
            top = ask - 0.01
            bottom = bid + 0.02
            if top < bottom:
                top = bottom
            if top <= bid:
                print(f"    Maker {side_label}: spread too tight ({spread_bps:.1f}bps), going market")
                return place_market(symbol, size)
            progress = min((attempt - 1) / max(max_attempts - 1, 1), 1.0)
            limit_price = round(top - progress * (top - bottom), 2)
            if limit_price <= bid:
                limit_price = round(bid + 0.02, 2)

        print(f"    Maker {side_label} #{attempt}: {symbol} at ${limit_price:,.2f} bid=${bid:,.2f} ask=${ask:,.2f} spread={spread_bps:.1f}bps [{elapsed}s/{maker_window}s]")

        status, order = place_limit(symbol, size, limit_price)
        if status != 200 or not order:
            err_detail = order if isinstance(order, dict) else {}
            err_msg = str(err_detail).lower()
            # BUG 2 fix: post_only rejection means price crossed (race condition).
            # Retry the loop instead of falling through to market — we still have time.
            if 'post only' in err_msg or 'post_only' in err_msg:
                print(f"    [!] post_only rejected (price crossed), retrying with fresh quote")
                cancel_all_open_orders(symbol)
                time.sleep(1)
                elapsed += 1
                continue
            if 'already been placed' in err_msg or 'already_placed' in err_msg:
                print(f"    [!] Duplicate order detected, cancelling all and retrying")
                cancel_all_open_orders(symbol)
                time.sleep(2)
                elapsed += 2
                continue
            print(f"    [!] Limit order failed (status={status}, error={err_detail}), going market")
            cancel_all_open_orders(symbol)
            time.sleep(1)
            return place_market(symbol, size)

        data = order.get('data', order)
        order_id = data.get('venue_order_id', data.get('id', ''))
        if not order_id:
            print(f"    [!] No order ID returned, going market")
            return place_market(symbol, size)

        time.sleep(check_interval)
        elapsed += check_interval

        s, o = get_order_status(order_id)
        if s == 200 and o:
            od = o.get('data', o)
            order_status = od.get('status', od.get('state', ''))
            if order_status == 'filled':
                avg = float(od.get('average_fill_price', limit_price))
                print(f"    Maker {side_label} FILLED at ${avg:,.2f} (0% fee)")
                return s, od
            elif order_status == 'partially_filled':
                filled_qty = float(od.get('filled_quantity', 0))
                total_qty = float(od.get('quantity', 1))
                pct = filled_qty / total_qty * 100 if total_qty > 0 else 0
                print(f"    Partially filled: {pct:.0f}% — waiting...")
                # Give partial fills extra time
                time.sleep(check_interval)
                elapsed += check_interval
                s2, o2 = get_order_status(order_id)
                if s2 == 200 and o2:
                    od2 = o2.get('data', o2)
                    if od2.get('status', od2.get('state', '')) == 'filled':
                        avg = float(od2.get('average_fill_price', limit_price))
                        print(f"    Maker {side_label} FILLED at ${avg:,.2f} (0% fee)")
                        return s2, od2

        # Cancel and re-price — verify cancel completed before placing new order
        cs, _ = cancel_order(order_id)
        time.sleep(1)
        elapsed += 1
        vs, vo = get_order_status(order_id)
        if vs == 200 and vo:
            vd = vo.get('data', vo)
            vstatus = vd.get('status', vd.get('state', ''))
            if vstatus == 'filled':
                avg = float(vd.get('average_fill_price', limit_price))
                print(f"    Maker {side_label} FILLED at ${avg:,.2f} (0% fee) (filled during cancel)")
                return vs, vd
            if vstatus not in ('cancelled', 'expired', 'rejected'):
                print(f"    Cancel pending (status={vstatus}), force-cancelling all")
                cancel_all_open_orders(symbol)
                time.sleep(1)
                elapsed += 1

        # After partial fill, recalculate size from actual available balance
        if is_buy:
            bal = get_balances()
            if bal:
                usd_avail = bal.get('USD', {}).get('available', 0)
                if usd_avail < MIN_TRADE_USD:
                    print(f"    Remaining balance ${usd_avail:,.2f} < ${MIN_TRADE_USD} minimum, going market for residual")
                    return place_market(symbol, usd_avail)
                if usd_avail < size:
                    print(f"    Balance updated after partial fill: ${size:,.2f} → ${usd_avail:,.2f}")
                    size = math.floor(usd_avail * 100) / 100 - 0.01
        else:
            bal = get_balances()
            if bal:
                asset_name = symbol.split('-')[0]
                crypto_avail = bal.get(asset_name, {}).get('available', 0)
                if crypto_avail < size and crypto_avail > 0:
                    print(f"    Balance updated after partial fill: {size:.6f} → {crypto_avail:.6f}")
                    size = crypto_avail

    print(f"    Not filled after {maker_window}s, going MARKET")
    cancel_all_open_orders(symbol)
    time.sleep(1)
    return place_market(symbol, size)


def execute_maker_buy(symbol, quote_size_usd):
    return _execute_maker_order(symbol, quote_size_usd, 'buy')


def execute_maker_sell(symbol, base_size):
    return _execute_maker_order(symbol, base_size, 'sell')


# ============================================================
# PER-ASSET POSITION STATE
# ============================================================
def _position_file(asset):
    return f'config/position_ed_v2_{asset}.json'

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
    if balances is None:
        print("  ⚠ Position sync skipped — balance API unreachable")
        return []
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

            # Detect locked funds (stale open orders) and cancel them
            available_amount = balances.get(asset, {}).get('available', 0)
            if available_amount <= 0 < actual_amount:
                print(f"  ⚠ SYNC: {asset} funds locked (available=0, total={actual_amount:.6f}) — cancelling stale orders")
                cancelled = cancel_all_open_orders(cfg.get('symbol', f'{asset}-USD'))
                if cancelled > 0:
                    send_telegram(f"🧹 {asset}: cancelled {cancelled} stale order(s) — funds unlocked")
                    changes.append(f"🧹 {asset}: cancelled {cancelled} stale order(s)")

        if updated:
            save_position(asset, pos)

    if changes and notify:
        msg = "🔄 <b>Position Sync</b>\n\n" + "\n".join(changes)
        send_telegram(msg)

    return changes


# ============================================================
# STRATEGY: COMPUTE SIGNAL PER ASSET
# ============================================================
def compute_asset_signal(sigs_by_horizon, strategy, min_conf=MIN_CONFIDENCE):
    """
    Apply per-asset strategy using signals from any available horizons.
    sigs_by_horizon: dict {h: signal_dict} for each horizon (any subset)
    Returns (action, confidence, reason).

    Strategies: both_agree, either_agree, Xh_only (any horizon), any_agree
    """
    if not sigs_by_horizon or all(v is None for v in sigs_by_horizon.values()):
        return 'HOLD', 50, 'no_signal'

    def _s(h): return sigs_by_horizon.get(h, {}).get('signal', 'HOLD') if sigs_by_horizon.get(h) else 'HOLD'
    def _c(h): return sigs_by_horizon.get(h, {}).get('confidence', 50) if sigs_by_horizon.get(h) else 50

    # --- single horizon (Xh_only for any X) ---
    if strategy.endswith('h_only'):
        h = int(strategy.split('h')[0])
        s, c = _s(h), _c(h)
        if s == 'SELL': return 'SELL', c, f'{h}h_sell'
        if s == 'BUY' and c >= min_conf: return 'BUY', c, f'{h}h_buy'
        return 'HOLD', c, 'low_conf' if s == 'BUY' else 'no_signal'

    # --- multi horizon: determine relevant horizons ---
    if strategy in ('both_agree', 'either_agree'):
        horizons = list(AVAILABLE_HORIZONS)
    elif strategy == 'any_agree':
        horizons = [h for h in sorted(sigs_by_horizon.keys()) if sigs_by_horizon.get(h)]
    else:
        horizons = list(AVAILABLE_HORIZONS)  # default fallback

    available = [h for h in horizons if sigs_by_horizon.get(h)]
    if not available:
        return 'HOLD', 50, 'no_signal'

    # SELL: any relevant horizon says SELL
    selling = [h for h in available if _s(h) == 'SELL']
    if selling:
        reason = '+'.join(f'{h}h' for h in selling) + '_sell'
        return 'SELL', max(_c(h) for h in selling), reason

    # BUY logic
    if strategy == 'both_agree':
        # AND: all available relevant horizons must BUY with conf >= min_conf
        if all(_s(h) == 'BUY' and _c(h) >= min_conf for h in available):
            reason = '+'.join(f'{h}h' for h in available) + '_buy'
            return 'BUY', min(_c(h) for h in available), reason
    elif strategy in ('either_agree', 'any_agree'):
        for h in available:
            if _s(h) == 'BUY' and _c(h) >= min_conf:
                return 'BUY', _c(h), f'{h}h_buy'

    # Check if any are low-conf BUY
    if any(_s(h) == 'BUY' for h in available):
        return 'HOLD', max(_c(h) for h in available), 'low_conf'

    return 'HOLD', 50, 'no_signal'


# ============================================================
# TAKE-PROFIT CHECK
# ============================================================
def _check_take_profit(asset, trading_cfg, dry_run=False):
    """Check if take-profit should trigger for an invested position.
    Called every 5 minutes from the sync loop.
    Returns True if TP was triggered and position was sold.
    """
    cfg = trading_cfg.get(asset, {})
    if not cfg.get('enabled'):
        return False

    tp_pct = cfg.get('take_profit_pct', 0)
    if not tp_pct or tp_pct <= 0:
        return False

    position = load_position(asset)
    if position['state'] != 'invested' or not position.get('entry_price'):
        return False

    symbol = cfg.get('symbol', f'{asset}-USD')

    # Get current price
    try:
        from crypto_live_trader_ed import download_asset
        df = download_asset(asset)
        if df is None or len(df) == 0:
            return False
        current_price = float(df.iloc[-1]['close'])
    except Exception as e:
        print(f"  [!] TP price check error for {asset}: {e}")
        return False

    pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100

    if pnl_pct >= tp_pct:
        print(f"\n  *** TAKE PROFIT TRIGGERED: {asset} at ${current_price:,.2f} "
              f"(+{pnl_pct:.2f}% >= +{tp_pct}% target) ***")

        if dry_run:
            print(f"  [DRY RUN] Would sell {asset} at ${current_price:,.2f}")
            send_telegram(
                f"<b>TP TRIGGERED (dry run)</b>\n"
                f"{asset}: ${current_price:,.2f} (+{pnl_pct:.2f}%)\n"
                f"Entry: ${position['entry_price']:,.2f}\n"
                f"Target: +{tp_pct}%"
            )
            return False

        # Execute sell
        actual_held = position.get('base_amount', 0)
        if actual_held <= 0:
            return False

        try:
            use_maker = cfg.get('use_maker_orders', False)
            if use_maker:
                status, order = execute_maker_sell(symbol, actual_held)
            else:
                status, order = place_market_sell(symbol, actual_held)
        except Exception as e:
            print(f"  [!] TP sell failed: {e}")
            send_telegram(f"<b>TP sell failed:</b> {asset}\n{e}")
            return False

        if status == 200 and order:
            sell_price = float(order.get('average_fill_price', current_price))
            entry_price_saved = position['entry_price']
            pnl_pct_actual = (sell_price - entry_price_saved) / entry_price_saved * 100
            pnl_usd = position.get('usd_invested', 0) * pnl_pct_actual / 100

            # Record trade
            position['trades'].append({
                'action': 'SELL',
                'price': sell_price,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'pnl_pct': round(pnl_pct_actual, 2),
                'pnl_usd': round(pnl_usd, 2),
                'auto': True,
                'reason': f'TP_{tp_pct}%',
            })
            position['state'] = 'cash'
            position['entry_price'] = 0
            position['entry_time'] = ''
            position['base_amount'] = 0
            position['usd_invested'] = 0
            save_position(asset, position)

            emoji = '+' if pnl_usd >= 0 else ''
            send_telegram(
                f"<b>TAKE PROFIT {asset}</b>\n"
                f"Sold at ${sell_price:,.2f} (+{pnl_pct_actual:.2f}%)\n"
                f"PnL: ${emoji}{pnl_usd:.2f}\n"
                f"Entry: ${entry_price_saved:,.2f}\n"
                f"Target was: +{tp_pct}%"
            )
            print(f"  TP SOLD {asset} at ${sell_price:,.2f} | PnL: ${pnl_usd:+.2f} ({pnl_pct_actual:+.2f}%)")
            return True
        else:
            print(f"  [!] TP sell order failed: status={status}")
            return False

    return False


# ============================================================
# PROCESS ONE ASSET
# ============================================================
def process_asset(asset, trading_cfg, dry_run=False):
    """Generate signals and execute for one asset. Returns result dict."""
    position = load_position(asset)
    symbol = trading_cfg.get('symbol', f'{asset}-USD')
    max_usd = trading_cfg.get('max_position_usd', 0)

    # Download data first — needed for regime detection
    try:
        download_asset(asset, update_only=True)
    except Exception:
        pass

    df_raw = load_data(asset)
    if df_raw is None:
        return None

    # Detect regime — determines horizon, confidence, and max_position
    regime_label, regime_cfg = detect_regime(asset, df_raw)
    regime_horizon = regime_cfg.get('horizon')
    regime_min_conf = regime_cfg.get('min_confidence')
    if regime_cfg.get('max_position_usd'):
        max_usd = regime_cfg['max_position_usd']

    # Load model config for the regime-selected horizon only
    sigs_by_horizon = {}
    if regime_horizon:
        cfg_r = load_best_config(asset, horizon=regime_horizon)
        if cfg_r:
            sigs_by_horizon[regime_horizon] = cfg_r
    else:
        # Fallback: load all available horizons
        for h in list(AVAILABLE_HORIZONS):
            cfg = load_best_config(asset, horizon=h)
            if cfg:
                sigs_by_horizon[h] = cfg

    if not sigs_by_horizon:
        return None
    # Generate live signals for each available horizon
    first = True
    for h in list(sigs_by_horizon.keys()):
        cfg = sigs_by_horizon[h]
        sig = generate_live_signal(asset, cfg, df_raw=df_raw, verbose=first)
        sigs_by_horizon[h] = sig  # replace config with actual signal
        first = False

    # Apply strategy — Ed regime always uses single horizon (Xh_only)
    min_conf = regime_min_conf if regime_min_conf else trading_cfg.get('min_confidence', MIN_CONFIDENCE)
    if regime_horizon:
        strategy = f'{regime_horizon}h_only'
    else:
        # Fallback: use first available horizon
        first_h = sorted(sigs_by_horizon.keys())[0]
        strategy = f'{first_h}h_only'
    action, confidence, reason = compute_asset_signal(sigs_by_horizon, strategy, min_conf=min_conf)
    any_sig = next((s for s in sigs_by_horizon.values() if s), None)
    price = any_sig['close'] if any_sig else 0

    # Reload position (sync may have updated it)
    position = load_position(asset)

    # Log - show all available horizons
    sig_strs = []
    for h in sorted(sigs_by_horizon.keys()):
        s = sigs_by_horizon[h]
        sig_strs.append(f"{h}h={s['signal']}({s['confidence']:.0f}%)" if s else f"{h}h=N/A")
    print(f"  {asset}: {' | '.join(sig_strs)} → {action} ({confidence:.0f}%) [{regime_label.upper()}] [{reason}] | pos={position['state']}")

    # Log signal for /chart command
    _log_signal(asset, price, sigs_by_horizon, action, confidence)

    # Expose sig_short, sig_long for Telegram message
    sig_short = sigs_by_horizon.get(HORIZON_SHORT)
    sig_long = sigs_by_horizon.get(HORIZON_LONG)

    # Execute
    executed = False
    pnl_msg = ""
    hold_override_active = False

    if action == 'BUY' and position['state'] == 'cash' and max_usd > 0:
        if not dry_run and position.get('auto_trade'):
            # Pre-trade balance check
            balances = get_balances()
            if balances is None:
                print(f"    ❌ Cannot buy {asset} — balance API unreachable")
                send_telegram(f"❌ {asset} BUY aborted: balance API failed")
                return {
                    'asset': asset, 'action': 'BUY', 'confidence': confidence,
                    'reason': 'api_failure', 'price': price, 'executed': False,
                    'pnl_msg': '', 'sigs_by_horizon': sigs_by_horizon,
                    'sig_short': sig_short, 'sig_long': sig_long,
                    'position': position, 'strategy': strategy,
                    'min_confidence': min_conf,
                    'gamma': '', 'regime': regime_label,
                    'horizon': regime_horizon or '',
                }
            usd_avail = balances.get('USD', {}).get('available', 0)
            print(f"    Pre-trade USD: ${usd_avail:,.2f}")

            # Minimum $300 to trade, target full max, or all available if < max
            if usd_avail < MIN_TRADE_USD:
                send_telegram(f"⚠️ {asset} BUY skipped — ${usd_avail:.2f} < ${MIN_TRADE_USD} minimum")
            else:
                buy_amount = min(max_usd, usd_avail)
                # BUG 1 fix: Revolut rejects maker orders where amount > balance even by $0.01
                # (rounding in qty*price). Floor to cents and subtract $0.01 safety margin.
                buy_amount = math.floor(buy_amount * 100) / 100 - 0.01
                if buy_amount < max_usd:
                    print(f"    Partial buy: ${buy_amount:,.2f} of ${max_usd:,.2f} (limited by balance)")
                use_maker = trading_cfg.get('use_maker_orders', False)
                if use_maker:
                    status, data = execute_maker_buy(symbol, buy_amount)
                else:
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
        # Hold-until-profitable: block sell if unrealized P&L below threshold
        min_sell_pnl = trading_cfg.get('min_sell_pnl_pct', 0)
        max_hold_h = trading_cfg.get('max_hold_hours', 10)
        if min_sell_pnl > 0 and position.get('entry_price', 0) > 0:
            cur_pnl = (price - position['entry_price']) / position['entry_price'] * 100
            hours_held = 0
            if position.get('entry_time'):
                try:
                    entry_dt = datetime.strptime(position['entry_time'].replace(' (synced)', ''), '%Y-%m-%d %H:%M')
                    hours_held = (datetime.now() - entry_dt).total_seconds() / 3600
                except Exception:
                    pass
            if cur_pnl < min_sell_pnl and hours_held < max_hold_h:
                print(f"    🛡 HOLD override: PnL {cur_pnl:+.2f}% < {min_sell_pnl:+.2f}% (held {hours_held:.0f}h / {max_hold_h}h max)")
                send_telegram(f"🛡 {asset} SELL blocked: PnL {cur_pnl:+.2f}% vs {min_sell_pnl}% target (held {hours_held:.0f}h / {max_hold_h}h)")
                action = 'HOLD'
                hold_override_active = True
            elif cur_pnl < min_sell_pnl and hours_held >= max_hold_h:
                print(f"    ⚠ Failsafe: held {hours_held:.0f}h >= {max_hold_h}h, selling at {cur_pnl:+.2f}%")
                send_telegram(f"⚠️ {asset} failsafe sell: held {hours_held:.0f}h, PnL {cur_pnl:+.2f}%")

    if action == 'SELL' and position['state'] == 'invested':
        if not dry_run and position.get('auto_trade'):
            # Get ACTUAL holdings from exchange — sell everything
            balances = get_balances()
            if balances is None:
                print(f"    ❌ Cannot sell {asset} — balance API unreachable, keeping position")
                send_telegram(f"❌ {asset} SELL aborted: balance API failed (position kept as invested)")
                return {
                    'asset': asset, 'action': 'SELL', 'confidence': confidence,
                    'reason': 'api_failure', 'price': price, 'executed': False,
                    'pnl_msg': '', 'sigs_by_horizon': sigs_by_horizon,
                    'sig_short': sig_short, 'sig_long': sig_long,
                    'position': position, 'strategy': strategy,
                    'min_confidence': min_conf,
                    'gamma': '', 'regime': regime_label,
                    'horizon': regime_horizon or '',
                }
            actual_held = balances.get(asset, {}).get('available', 0)
            total_held = balances.get(asset, {}).get('total', 0)
            print(f"    Pre-trade: {actual_held:.6f} {asset} available, {total_held:.6f} total (selling ALL)")

            # If available=0 but total>0, funds are locked by an open order — cancel it first
            if actual_held <= 0 < total_held:
                print(f"    ⚠ {asset} balance locked (available=0, total={total_held:.6f}) — cancelling open orders...")
                send_telegram(f"⚠️ {asset} balance locked — cancelling open orders before sell")
                cancelled = cancel_all_open_orders(symbol)
                if cancelled > 0:
                    time.sleep(2)
                    balances = get_balances()
                    if balances is None:
                        print(f"    ❌ Cannot re-check balance after cancel — aborting sell")
                        send_telegram(f"❌ {asset} SELL aborted: balance API failed after order cancel")
                        return {
                            'asset': asset, 'action': 'SELL', 'confidence': confidence,
                            'reason': 'api_failure', 'price': price, 'executed': False,
                            'pnl_msg': '', 'sigs_by_horizon': sigs_by_horizon,
                            'sig_short': sig_short, 'sig_long': sig_long,
                            'position': position, 'strategy': strategy,
                            'min_confidence': min_conf,
                            'gamma': '', 'regime': regime_label,
                            'horizon': regime_horizon or '',
                        }
                    actual_held = balances.get(asset, {}).get('available', 0)
                    print(f"    After cancel: {actual_held:.6f} {asset} available")

            if actual_held > 0:
                use_maker = trading_cfg.get('use_maker_orders', False)
                if use_maker:
                    status, data = execute_maker_sell(symbol, actual_held)
                else:
                    status, data = place_market_sell(symbol, actual_held)
                if status in (200, 201):
                    order = data.get('data', data)
                    price = float(order.get('average_fill_price', price))
                    executed = True

                    # Post-trade verification
                    time.sleep(2)
                    post_bal = get_balances()
                    if post_bal:
                        usd_now = post_bal.get('USD', {}).get('available', 0)
                        crypto_left = post_bal.get(asset, {}).get('total', 0)
                        print(f"    Post-trade: {crypto_left:.6f} {asset} | ${usd_now:,.2f} USD")
                else:
                    send_telegram(f"⚠️ {asset} SELL failed: {status} {data}")
            else:
                print(f"    ⚠ Nothing to sell (exchange balance = 0)")
                send_telegram(f"⚠️ {asset} SELL skipped: exchange balance = 0 (position kept as invested)")

        if not dry_run and position.get('auto_trade') and not executed:
            # Auto-trade sell failed or nothing to sell — keep position as invested
            pass
        else:
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

    # Get gamma from the primary horizon model
    gamma_val = ''
    gamma_horizons = [regime_horizon] if regime_horizon else [HORIZON_LONG, HORIZON_SHORT]
    for h in gamma_horizons:
        cfg_g = load_best_config(asset, horizon=h)
        if cfg_g:
            g = cfg_g.get('gamma', 1.0)
            if g and g < 1.0:
                gamma_val = str(g)
            break

    return {
        'asset': asset, 'action': action, 'confidence': confidence,
        'reason': reason, 'price': price, 'executed': executed,
        'pnl_msg': pnl_msg, 'sigs_by_horizon': sigs_by_horizon,
        'sig_short': sig_short, 'sig_long': sig_long,
        'position': position, 'strategy': strategy,
        'min_confidence': min_conf,
        'gamma': gamma_val, 'regime': regime_label,
        'horizon': regime_horizon or '',
        '_hold_override_active': hold_override_active,
    }


# ============================================================
# TELEGRAM MESSAGE (MULTI-ASSET)
# ============================================================
def format_multi_asset_telegram(results, dry_run=False, balances=None, trading_cfg=None):
    """Format combined Telegram message for all assets."""
    if balances is None:
        balances = {}
    lines = []
    mode = "DRY" if dry_run else "LIVE"
    lines.append(f"📊 <b>Hourly Update [{mode}]</b>")
    lines.append(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("/help")
    lines.append("")

    for r in results:
        if r is None:
            continue
        asset = r['asset']
        action = r['action']
        conf = r['confidence']
        price = r['price']
        position = r['position']

        # Get RSI + last 4 candles from data
        rsi = 0
        last_prices = []
        try:
            sigs_h = r.get('sigs_by_horizon', {})
            any_sig = next((s for s in sigs_h.values() if s), None) or r.get('sig_short') or r.get('sig_long')
            if any_sig:
                rsi = any_sig.get('rsi', 0)
                for h in any_sig.get('last_4h', []):
                    last_prices.append((h['datetime'], h['close']))
        except Exception:
            pass

        # Exchange holdings
        actual_held = balances.get(asset, {}).get('total', 0)
        actual_usd = actual_held * price if price > 0 and actual_held > 0 else 0

        # Signal emojis — show actual horizons from sigs_by_horizon
        def _se(sig, h):
            if not sig: return f"{h}h:N/A"
            e = '🔵' if sig['signal'] == 'BUY' else '🔴' if sig['signal'] == 'SELL' else '🟡'
            return f"{e}{h}h:{sig['signal']}({sig['confidence']:.0f}%)"

        sigs_h = r.get('sigs_by_horizon', {})
        if sigs_h:
            sig_line = ' '.join(_se(sigs_h.get(h), h) for h in sorted(sigs_h.keys()))
        else:
            sig_line = f"{_se(r['sig_short'], HORIZON_SHORT)} {_se(r['sig_long'], HORIZON_LONG)}"

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
            # Check if hold override is active (sell was blocked this cycle)
            _min_pnl = r.get('_hold_override_active', False)
            if _min_pnl:
                act = f"🛡 HOLD (sell blocked, {cur_pnl:+.1f}%)"
            else:
                act = f"⏸ HOLD ({cur_pnl:+.1f}%)"
        else:
            act = f"⏸ HOLD (cash)"

        # Header with price + RSI + auto status
        price_str = f"${price:,.2f}" if price >= 100 else f"${price:,.4f}"
        auto_icon = "🔵" if position.get('auto_trade') else "🔴"
        gamma_str = r.get('gamma', '')
        gamma_suffix = f"|γ{gamma_str}" if gamma_str else ""
        mc = r.get('min_confidence', '')
        conf_str = f" | conf:{mc}%" if mc else ""
        regime_tag = r.get('regime', 'bull').upper()
        horizon_tag = r.get('horizon', '?')
        lines.append(f"{auto_icon} <b>{asset}</b> {price_str} | RSI:{rsi:.0f} | [{regime_tag}|{horizon_tag}h{gamma_suffix}]{conf_str}")

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

            # Show hold override status when active
            _asset_cfg = (trading_cfg or {}).get(asset, {})
            _cfg_min_pnl = _asset_cfg.get('min_sell_pnl_pct', 0)
            if _cfg_min_pnl > 0 and cur_pnl < _cfg_min_pnl:
                _max_h = _asset_cfg.get('max_hold_hours', 10)
                _held_h = 0
                if position.get('entry_time'):
                    try:
                        _edt = datetime.strptime(position['entry_time'].replace(' (synced)', ''), '%Y-%m-%d %H:%M')
                        _held_h = (datetime.now() - _edt).total_seconds() / 3600
                    except Exception:
                        pass
                lines.append(f"  🛡 Hold override: need {_cfg_min_pnl}% (at {cur_pnl:+.1f}%) | {_held_h:.0f}h / {_max_h}h")

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

def _answer_callback_query(callback_query_id):
    """Acknowledge inline button press (removes loading spinner)."""
    token = TELEGRAM_CONFIG.get('token', '')
    if not token:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
        payload = json.dumps({'callback_query_id': callback_query_id}).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=5, context=_ssl_ctx)
    except Exception:
        pass


def check_telegram_commands():
    global _last_update_id
    token = TELEGRAM_CONFIG.get('token', '')
    if not token:
        return None
    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates?offset={_last_update_id + 1}&timeout=0"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=5, context=_ssl_ctx) as resp:
            data = json.loads(resp.read().decode())
        if not data.get('ok') or not data.get('result'):
            return None
        last_text = None
        for update in data['result']:
            _last_update_id = update['update_id']
            # Handle text messages
            text = update.get('message', {}).get('text', '').strip()
            if text:
                last_text = text
            # Handle inline button presses (callback queries)
            cb = update.get('callback_query')
            if cb:
                _answer_callback_query(cb['id'])
                cb_data = cb.get('data', '').strip()
                if cb_data:
                    last_text = cb_data
        return last_text
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
        with urllib.request.urlopen(req, timeout=5, context=_ssl_ctx) as resp:
            data = json.loads(resp.read().decode())
        if data.get('ok') and data.get('result'):
            _last_update_id = data['result'][-1]['update_id']
    except Exception:
        pass

def _handle_status_command(with_charts=False):
    trading_cfg = load_trading_config()
    balances = get_balances()
    lines = [f"📊 <b>Status</b> — {datetime.now().strftime('%H:%M')}\n"]

    enabled_assets = []
    for asset, cfg in trading_cfg.items():
        if not cfg.get('enabled'):
            continue
        enabled_assets.append(asset)
        pos = load_position(asset)
        symbol = cfg.get('symbol', f'{asset}-USD')
        # Regime info for display
        bull_cfg = cfg.get('bull', {})
        bear_cfg = cfg.get('bear', {})
        regime_label = f"bull={bull_cfg.get('horizon','?')}h|bear={bear_cfg.get('horizon','?')}h"

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
        lines.append(f"<b>{asset}</b> {price_str} | RSI: {rsi:.0f} | [{regime_label}]")

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
            bull_max = cfg.get('bull', {}).get('max_position_usd', 0)
            bear_max = cfg.get('bear', {}).get('max_position_usd', 0)
            lines.append(f"  💵 CASH (bull max ${bull_max:,.0f} | bear max ${bear_max:,.0f})")
            if actual_held > 0 and actual_usd > 5:
                lines.append(f"  ⚠️ Exchange: {actual_held:.6f} {asset} (≈${actual_usd:,.2f})")

        # Trade history summary
        sells = [t for t in pos.get('trades', []) if t.get('action') == 'SELL']
        if sells:
            total = sum(t.get('pnl_usd', 0) for t in sells)
            wins = sum(1 for t in sells if t.get('pnl_usd', 0) > 0)
            lines.append(f"  📊 {len(sells)} trades ({wins}W) ${total:+,.2f}")
        lines.append("")

    # Balance summary
    usd_avail = balances.get('USD', {}).get('available', 0)
    lines.append(f"💵 USD: ${usd_avail:,.2f}")
    other = [(c, b) for c, b in sorted(balances.items()) if c != 'USD' and b['total'] > 0 and c not in enabled_assets]
    if other:
        for c, b in other:
            lines.append(f"  {c}: {b['available']:.6f}")

    send_telegram_with_buttons("\n".join(lines), MAIN_BUTTONS)

    # Auto-send charts
    if with_charts:
        for asset in enabled_assets:
            _generate_and_send_chart(asset)


# ============================================================
# BACKGROUND TELEGRAM COMMAND THREAD
# ============================================================
_stop_event = threading.Event()
_rerun_event = threading.Event()
_paused_lock = threading.Lock()
_paused_flag = [False]  # mutable container for thread-safe access

# Ed uses regime-based horizon selection — no strategy picker needed

# ---- Simple command handlers ----

def _handle_config_command():
    """Show current trading config for all assets with setup button."""
    trading_cfg = load_trading_config()
    lines = ["⚙️ <b>Config</b>\n"]
    for asset, cfg in trading_cfg.items():
        enabled = "🔵" if cfg.get('enabled') else "🔴"
        pos = load_position(asset)
        auto = "AUTO" if pos.get('auto_trade') else "MANUAL"
        det = cfg.get('regime_detector', {})
        if det.get('type') == 'sma_cross':
            det_label = f"sma_cross({det.get('params',{}).get('fast','')}/{det.get('params',{}).get('slow','')})"
        elif det.get('type') == 'named':
            det_label = det.get('params', {}).get('name', 'named')
        else:
            det_label = det.get('type', '?')
        bull_cfg = cfg.get('bull', {})
        bear_cfg = cfg.get('bear', {})
        tp_pct = cfg.get('take_profit_pct', 0)
        tp_str = f"TP={tp_pct}%" if tp_pct > 0 else "TP=OFF"
        maker_str = "MAKER" if cfg.get('use_maker_orders') else "TAKER"
        lines.append(
            f"{enabled} <b>{asset}</b> | {det_label} | "
            f"bull={bull_cfg.get('horizon','?')}h@{bull_cfg.get('min_confidence','?')}% | "
            f"bear={bear_cfg.get('horizon','?')}h@{bear_cfg.get('min_confidence','?')}% | {auto} | {tp_str} | {maker_str}"
        )
    buttons = [[('⚙️ Edit Config', '/setup')]]
    send_telegram_with_buttons("\n".join(lines), buttons)

def _handle_regime_command():
    """Show current regime per enabled asset."""
    trading_cfg = load_trading_config()
    lines = ["🌡 <b>Regime Status</b>\n"]
    for asset, cfg in trading_cfg.items():
        if not cfg.get('enabled'):
            continue
        try:
            df_raw = load_data(asset)
            if df_raw is not None:
                regime, active_cfg = detect_regime(asset, df_raw)
                h = active_cfg.get('horizon', '?')
                mc = active_cfg.get('min_confidence', '?')
                mp = active_cfg.get('max_position_usd', 0)
                icon = "🟢" if regime == 'bull' else "🔴"
                lines.append(
                    f"{icon} <b>{asset}</b>: {regime.upper()} | "
                    f"horizon={h}h | conf>={mc}% | max=${mp:,.0f}"
                )
            else:
                lines.append(f"⚪ <b>{asset}</b>: no data")
        except Exception as e:
            lines.append(f"⚪ <b>{asset}</b>: error ({e})")
    send_telegram("\n".join(lines))

def _handle_summary_command():
    """Full summary: status + balance + charts for all enabled assets."""
    _handle_status_command(with_charts=True)


def _handle_help_command():
    """Send list of available commands."""
    send_telegram_with_buttons(
        "📱 <b>Commands</b>\n\n"
        "/status — Positions, prices, P&L + balance\n"
        "/summary — Status + balance + charts\n"
        "/chart — 48h candlestick charts (all active)\n"
        "/chart BTC — 48h chart for single asset\n"
        "/conf — Show config\n"
        "/regime — Show current regime per asset\n"
        "/setup — Edit config (inline buttons)\n"
        "/sync — Sync positions from exchange\n"
        "/optimize BTC — Re-run Mode D optimization\n"
        "/pause — Pause trading (signals only)\n"
        "/resume — Resume trading\n"
        "/stop — Stop the trader",
        MAIN_BUTTONS
    )

# ---- /optimize: background Mode D re-optimization ----
import subprocess
_optimize_proc = None  # subprocess.Popen or None
_optimize_lock = threading.Lock()

def _handle_optimize_command(msg):
    """Launch Mode D optimization as a background subprocess."""
    global _optimize_proc
    parts = msg.strip().split()
    # Parse: /optimize BTC  or  /optimize BTC,ETH  or  /optimize (defaults to BTC)
    assets = parts[1].upper() if len(parts) > 1 else 'BTC'

    with _optimize_lock:
        if _optimize_proc is not None and _optimize_proc.poll() is None:
            send_telegram(f"⏳ Optimization already running (PID {_optimize_proc.pid}). /optstatus to check.")
            return

        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'crypto_trading_system_ed.py')

        # Build horizon arg from trading config (per-asset configured horizons)
        trading_config = load_trading_config()
        asset_list = [a.strip() for a in assets.split(',')]
        horizons_set = set()
        for a in asset_list:
            h = trading_config.get(a, {}).get('horizon')
            if h:
                horizons_set.add(h)
        if not horizons_set:
            horizons_set = set(AVAILABLE_HORIZONS)
        h_arg = ','.join(str(h) for h in sorted(horizons_set)) + 'h'

        cmd = [sys.executable, script, 'DV', assets, h_arg]

        # Run at below-normal priority on Windows
        creation_flags = 0
        if sys.platform == 'win32':
            creation_flags = 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS

        _optimize_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        print(f"  OPTIMIZE: launched Mode DV for {assets} {h_arg} (PID {_optimize_proc.pid})")
        send_telegram(f"🚀 <b>Mode DV started</b>\n{assets} {h_arg} (PID {_optimize_proc.pid})\nTrader stays live. Models hot-reload when done.")

        # Monitor in background thread
        def _monitor():
            global _optimize_proc
            proc = _optimize_proc
            proc.wait()
            rc = proc.returncode
            if rc == 0:
                print(f"  OPTIMIZE: completed successfully")
                send_telegram("✅ <b>Mode D complete</b>\nNew models saved. Hot-reload will pick them up.")
            else:
                # Grab last few lines of output for error context
                out = proc.stdout.read().decode('utf-8', errors='replace') if proc.stdout else ''
                tail = '\n'.join(out.strip().splitlines()[-5:])
                print(f"  OPTIMIZE: failed (exit code {rc})")
                send_telegram(f"❌ <b>Mode D failed</b> (exit {rc})\n<pre>{tail}</pre>")
            with _optimize_lock:
                _optimize_proc = None

        threading.Thread(target=_monitor, daemon=True).start()

def _handle_optstatus_command():
    """Check if optimization is running."""
    with _optimize_lock:
        if _optimize_proc is not None and _optimize_proc.poll() is None:
            send_telegram(f"⏳ Optimization running (PID {_optimize_proc.pid})")
        else:
            send_telegram("No optimization running.")

# ---- Signal logging for /chart ----
SIGNAL_LOG_DIR = 'config'
SIGNAL_LOG_FILE = os.path.join(SIGNAL_LOG_DIR, 'signal_log.csv')

def _log_signal(asset, price, sigs_by_horizon, action, confidence):
    """Append one row per signal to signal_log.csv for chart rendering."""
    import csv
    fieldnames = ['timestamp', 'asset', 'price', 'action', 'confidence',
                  f'sig_{HORIZON_SHORT}h', f'conf_{HORIZON_SHORT}h', f'sig_{HORIZON_LONG}h', f'conf_{HORIZON_LONG}h']
    file_exists = os.path.exists(SIGNAL_LOG_FILE) and os.path.getsize(SIGNAL_LOG_FILE) > 0
    try:
        sig_s = sigs_by_horizon.get(HORIZON_SHORT)
        sig_l = sigs_by_horizon.get(HORIZON_LONG)
        with open(SIGNAL_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'asset': asset,
                'price': f'{price:.2f}',
                'action': action,
                'confidence': f'{confidence:.1f}',
                f'sig_{HORIZON_SHORT}h': sig_s['signal'] if sig_s else '',
                f'conf_{HORIZON_SHORT}h': f"{sig_s['confidence']:.1f}" if sig_s else '',
                f'sig_{HORIZON_LONG}h': sig_l['signal'] if sig_l else '',
                f'conf_{HORIZON_LONG}h': f"{sig_l['confidence']:.1f}" if sig_l else '',
            })
    except Exception as e:
        print(f"  [!] Signal log error: {e}")


def send_telegram_photo(photo_path, caption=''):
    """Send a photo to Telegram."""
    token = TELEGRAM_CONFIG.get('token', '')
    chat_id = TELEGRAM_CONFIG.get('chat_id', '')
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        import io
        boundary = '----PythonBoundary'
        body = io.BytesIO()

        # chat_id field
        body.write(f'--{boundary}\r\n'.encode())
        body.write(f'Content-Disposition: form-data; name="chat_id"\r\n\r\n{chat_id}\r\n'.encode())

        # caption field
        if caption:
            body.write(f'--{boundary}\r\n'.encode())
            body.write(f'Content-Disposition: form-data; name="caption"\r\n\r\n{caption}\r\n'.encode())

        # photo file
        with open(photo_path, 'rb') as img:
            img_data = img.read()
        body.write(f'--{boundary}\r\n'.encode())
        body.write(f'Content-Disposition: form-data; name="photo"; filename="chart.png"\r\n'.encode())
        body.write(f'Content-Type: image/png\r\n\r\n'.encode())
        body.write(img_data)
        body.write(f'\r\n--{boundary}--\r\n'.encode())

        req = urllib.request.Request(
            url, data=body.getvalue(),
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
        )
        with urllib.request.urlopen(req, timeout=30, context=_ssl_ctx) as resp:
            result = json.loads(resp.read().decode())
            if result.get('ok'):
                print("  ✓ Telegram photo sent")
                return True
            print(f"  [!] Telegram photo error: {result}")
    except Exception as e:
        print(f"  [!] Telegram photo error: {e}")
    return False


def send_telegram_with_buttons(message, buttons, parse_mode='HTML'):
    """Send Telegram message with inline keyboard buttons.
    buttons: list of rows, each row is a list of (text, callback_data) tuples.
    Example: [[('📊 Charts', '/chart'), ('💰 Status', '/status')], [('⏸ Pause', '/pause')]]
    """
    token = TELEGRAM_CONFIG.get('token', '')
    chat_id = TELEGRAM_CONFIG.get('chat_id', '')
    if not token or not chat_id:
        return send_telegram(message, parse_mode)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    keyboard = {
        'inline_keyboard': [
            [{'text': text, 'callback_data': cb} for text, cb in row]
            for row in buttons
        ]
    }
    payload = json.dumps({
        'chat_id': chat_id, 'text': message,
        'parse_mode': parse_mode, 'reply_markup': keyboard
    }).encode('utf-8')
    try:
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as resp:
            result = json.loads(resp.read().decode())
            if result.get('ok'):
                print("  Telegram sent (with buttons)")
                return True
            print(f"  [!] Telegram error: {result}")
            return False
    except Exception as e:
        print(f"  [!] Telegram error: {e}")
        return False


# Standard button layouts
MAIN_BUTTONS = [
    [('📊 Status', '/status'), ('📈 Charts', '/chart')],
    [('⚙️ Config', '/conf'), ('🔄 Sync', '/sync')],
]


def _handle_chart_command(msg, trading_cfg):
    """Generate and send a 24h chart with price + model predictions.
    /chart       → all enabled assets (separate charts)
    /chart BTC   → single asset
    """
    parts = msg.split()
    if len(parts) >= 2:
        asset = parts[1].upper()
        if asset not in trading_cfg:
            send_telegram(f"Unknown asset: {asset}. Use: {', '.join(trading_cfg.keys())}")
            return
        assets_to_chart = [asset]
    else:
        # No asset specified → all enabled
        assets_to_chart = [a for a, c in trading_cfg.items() if c.get('enabled')]
        if not assets_to_chart:
            send_telegram("No enabled assets to chart.")
            return

    for asset in assets_to_chart:
        _generate_and_send_chart(asset)


def _generate_and_send_chart(asset):
    """Generate and send a 48h candlestick chart with signal overlays."""
    try:
        df_raw = load_data(asset)
        if df_raw is None:
            send_telegram(f"No data for {asset}")
            return
    except Exception as e:
        send_telegram(f"Error loading {asset} data: {e}")
        return

    df_price = df_raw.tail(48).copy()
    if len(df_price) < 4:
        send_telegram(f"Not enough data for {asset}")
        return

    # Load signal log (last 49h to catch signals at boundaries)
    signals = []
    if os.path.exists(SIGNAL_LOG_FILE):
        try:
            import csv
            with open(SIGNAL_LOG_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=49)).strftime('%Y-%m-%d %H:%M:%S')
                for row in reader:
                    if row['asset'] == asset and row['timestamp'] >= cutoff:
                        signals.append(row)
        except Exception:
            pass

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.lines import Line2D

        # Colors
        c_bg     = '#0e1117'
        c_card   = '#141922'
        c_buy    = '#3b82f6'
        c_sell   = '#ef4444'
        c_green  = '#3b82f6'   # blue — user is colorblind
        c_red    = '#ef4444'
        c_gold   = '#eab308'
        c_text   = '#e2e8f0'
        c_muted  = '#64748b'
        c_grid   = '#1e293b'
        c_wick   = '#475569'

        times = pd.to_datetime(df_price['datetime'])
        opens = df_price['open'].values
        highs = df_price['high'].values
        lows = df_price['low'].values
        closes = df_price['close'].values

        fig, ax = plt.subplots(figsize=(12, 6), facecolor=c_bg)
        ax.set_facecolor(c_card)

        # Candlesticks
        bar_width = timedelta(minutes=40)
        for i in range(len(times)):
            t = times.iloc[i]
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            color = c_green if c >= o else c_red
            # Wick
            ax.plot([t, t], [l, h], color=c_wick, linewidth=0.8, zorder=2)
            # Body
            body_bottom = min(o, c)
            body_height = abs(c - o)
            if body_height < (highs.max() - lows.min()) * 0.001:
                body_height = (highs.max() - lows.min()) * 0.001  # doji minimum
            ax.bar(t, body_height, bottom=body_bottom, width=bar_width,
                   color=color, edgecolor=color, linewidth=0.5, zorder=3, alpha=0.85)

        # Only show signals where action CHANGED (transitions, not repeated hourly)
        price_range = highs.max() - lows.min()
        offset_y = price_range * 0.03

        # Filter to transitions only (action changed from previous signal)
        transition_signals = []
        prev_action = None
        for sig in signals:
            action = sig['action']
            if action == 'HOLD':
                prev_action = action
                continue
            if action != prev_action:
                transition_signals.append(sig)
            prev_action = action

        for sig in transition_signals:
            try:
                sig_time = pd.to_datetime(sig['timestamp'])
                action = sig['action']

                # Find nearest candle
                time_diffs = abs(times - sig_time)
                nearest_idx = time_diffs.argmin()
                snap_time = times.iloc[nearest_idx]
                snap_price = closes[nearest_idx]

                # Validate prediction: did price move in predicted direction?
                future_closes = closes[nearest_idx + 1:nearest_idx + 5]
                if len(future_closes) >= 4:
                    future_price = future_closes[-1]
                    correct = (future_price > snap_price) if action == 'BUY' else (future_price < snap_price)
                else:
                    correct = None

                color = c_green if correct else c_red if correct is not None else c_gold
                if action == 'BUY':
                    marker, y_pos = '^', lows[nearest_idx] - offset_y
                else:
                    marker, y_pos = 'v', highs[nearest_idx] + offset_y

                ax.scatter(snap_time, y_pos, color=color, marker=marker,
                           s=200, zorder=6, edgecolors='white', linewidth=0.8)
            except Exception:
                continue

        # Current price line
        last_price = closes[-1]
        ax.axhline(y=last_price, color=c_muted, linewidth=0.6, linestyle='--', alpha=0.5, zorder=1)
        ax.annotate(f'${last_price:,.2f}', xy=(times.iloc[-1], last_price),
                    fontsize=8, color=c_text, fontweight='bold',
                    xytext=(10, 0), textcoords='offset points', va='center')

        # 24h change
        price_24h_ago = closes[0] if len(closes) >= 24 else closes[0]
        pct_change = (last_price / price_24h_ago - 1) * 100
        change_color = c_green if pct_change >= 0 else c_red

        ax.set_title(f'{asset}/USD  ${last_price:,.2f}  ({pct_change:+.1f}%)',
                     fontsize=14, fontweight='bold', color=c_text, loc='left')
        ax.text(0.99, 1.02, '48h', transform=ax.transAxes, fontsize=10,
                color=c_muted, ha='right', va='bottom')

        # X axis: show date + time
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.tick_params(axis='x', colors=c_muted, labelsize=8)
        ax.tick_params(axis='y', colors=c_muted, labelsize=9)
        # Smart y-axis: decimal places for low-price assets
        if price_range < 1:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.4f}'))
        elif price_range < 10:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.2f}'))
        elif price_range < 100:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.1f}'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax.grid(True, alpha=0.15, color=c_grid, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Y padding
        ax.set_ylim(lows.min() - price_range * 0.08, highs.max() + price_range * 0.08)

        # Legend — only show if there are signals
        if transition_signals:
            legend_elements = [
                Line2D([0], [0], marker='^', color='w', markerfacecolor=c_green,
                       markersize=9, label='BUY ✓', linestyle='None'),
                Line2D([0], [0], marker='v', color='w', markerfacecolor=c_red,
                       markersize=9, label='SELL ✗', linestyle='None'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=c_gold,
                       markersize=8, label='Pending', linestyle='None'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
                      facecolor=c_card, edgecolor=c_grid, labelcolor=c_text,
                      framealpha=0.9, ncol=3, columnspacing=1.0,
                      bbox_to_anchor=(1.0, -0.08))

        plt.tight_layout()
        chart_path = os.path.join('charts', f'{asset}_telegram_24h.png')
        os.makedirs('charts', exist_ok=True)
        fig.savefig(chart_path, dpi=150, facecolor=c_bg, bbox_inches='tight')
        plt.close(fig)

        n_transitions = len(transition_signals)
        caption = f"{asset}/USD 48h | {n_transitions} signals | {pct_change:+.1f}%"
        send_telegram_photo(chart_path, caption=caption)
        print(f"  Chart sent for {asset}")

    except Exception as e:
        send_telegram(f"Chart error: {e}")
        print(f"  [!] Chart error: {e}")


# ---- Interactive setup wizard (inline buttons) ----
_setup_state = {'active': False}

def _setup_send_asset_picker(cfg):
    """Show asset picker with inline buttons."""
    buttons = []
    row = []
    for asset in cfg:
        enabled = cfg[asset].get('enabled', False)
        icon = '🔵' if enabled else '🔴'
        row.append((f"{icon} {asset}", f"/cfg_{asset}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([('✅ Save & Close', '/cfg_save')])
    send_telegram_with_buttons("⚙️ <b>Config</b> — Pick an asset to configure:", buttons)

def _setup_send_menu(asset, cfg):
    """Show settings menu for an asset with inline buttons."""
    enabled = cfg.get('enabled', False)
    pos = load_position(asset)
    auto = pos.get('auto_trade', False)
    det = cfg.get('regime_detector', {})
    if det.get('type') == 'sma_cross':
        det_label = f"sma_cross({det.get('params',{}).get('fast','')}/{det.get('params',{}).get('slow','')})"
    elif det.get('type') == 'named':
        det_label = det.get('params', {}).get('name', 'named')
    else:
        det_label = det.get('type', '?')
    bull_cfg = cfg.get('bull', {})
    bear_cfg = cfg.get('bear', {})

    text = (
        f"⚙️ <b>{asset} Settings</b>\n\n"
        f"Enabled: {'🔵 ON' if enabled else '🔴 OFF'}\n"
        f"Detector: {det_label}\n"
        f"Bull: {bull_cfg.get('horizon','?')}h | conf>={bull_cfg.get('min_confidence','?')}% | max=${bull_cfg.get('max_position_usd',0):,.0f}\n"
        f"Bear: {bear_cfg.get('horizon','?')}h | conf>={bear_cfg.get('min_confidence','?')}% | max=${bear_cfg.get('max_position_usd',0):,.0f}\n"
        f"Auto-trade: {'🔵 ON' if auto else '🔴 OFF'}\n"
        f"Take-profit: {'🔵 ON (' + str(cfg.get('take_profit_pct', 0)) + '%)' if cfg.get('take_profit_pct', 0) > 0 else '🔴 OFF'}\n"
        f"Maker orders: {'🔵 ON (0% fee)' if cfg.get('use_maker_orders') else '🔴 OFF (market)'}"
    )
    tp_label = f"TP: ON ({cfg.get('take_profit_pct', 0)}%)" if cfg.get('take_profit_pct', 0) > 0 else "TP: OFF"
    maker_label = 'Maker: ON' if cfg.get('use_maker_orders') else 'Maker: OFF'
    buttons = [
        [('🔀 Toggle ON/OFF', f'/cfg_{asset}_toggle'),
         ('🔀 Auto-trade', f'/cfg_{asset}_auto')],
        [('📊 Bull Confidence', f'/cfg_{asset}_bull_conf'),
         ('📊 Bear Confidence', f'/cfg_{asset}_bear_conf')],
        [('💰 Bull Max Pos', f'/cfg_{asset}_bull_max'),
         ('💰 Bear Max Pos', f'/cfg_{asset}_bear_max')],
        [(tp_label, f'/cfg_{asset}_tp'),
         (maker_label, f'/cfg_{asset}_maker')],
        [('⬅️ Back', '/cfg_back')],
    ]
    send_telegram_with_buttons(text, buttons)

def _setup_send_strategy_picker(asset, cfg):
    """Strategy is controlled by regime detector — no manual picker."""
    send_telegram(f"📐 <b>{asset}</b>: Strategy is controlled by the regime detector. Use /conf to see current regime settings.")
    _setup_send_menu(asset, cfg)

def _setup_send_confidence_picker(asset, cfg, regime='bull'):
    """Show confidence picker for bull or bear regime."""
    regime_cfg = cfg.get(regime, {})
    current = regime_cfg.get('min_confidence', MIN_CONFIDENCE)
    values = [60, 65, 70, 75, 80, 85, 90, 95]
    buttons = []
    row = []
    for v in values:
        label = f"{'✅ ' if v == current else ''}{v}%"
        row.append((label, f'/cfg_{asset}_{regime}_confv_{v}'))
        if len(row) == 4:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([('⬅️ Back', f'/cfg_{asset}')])
    send_telegram_with_buttons(f"📊 Min confidence for <b>{asset}</b> ({regime.upper()}):", buttons)

def _setup_send_max_picker(asset, cfg, regime='bull'):
    """Show max position picker for bull or bear regime."""
    regime_cfg = cfg.get(regime, {})
    current = regime_cfg.get('max_position_usd', 0)
    values = [0, 1000, 3000, 6000, 10000, 12000]
    buttons = []
    row = []
    for v in values:
        label = f"{'✅ ' if v == current else ''}${v:,}"
        row.append((label, f'/cfg_{asset}_{regime}_maxv_{v}'))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    # Custom input button + current value display
    custom_label = f"✏️ Custom (now ${current:,})" if current not in values else "✏️ Custom"
    buttons.append([(custom_label, f'/cfg_{asset}_{regime}_maxcustom')])
    buttons.append([('⬅️ Back', f'/cfg_{asset}')])
    send_telegram_with_buttons(f"💰 Max position for <b>{asset}</b> ({regime.upper()}):", buttons)

def _setup_start(trading_cfg):
    """Begin interactive setup — show asset picker."""
    global _setup_state
    cfg_copy = json.loads(json.dumps(trading_cfg))
    _setup_state = {
        'active': True,
        'cfg': cfg_copy,
    }
    _setup_send_asset_picker(cfg_copy)

def _setup_finish(trading_cfg):
    """Save config and trigger signal re-run."""
    global _setup_state
    cfg = _setup_state['cfg']
    save_trading_config(cfg)
    for asset in cfg:
        trading_cfg[asset] = cfg[asset]
    _setup_state = {'active': False}
    send_telegram("✅ <b>Config saved!</b>\nRe-running signals...")
    _send_startup_telegram(trading_cfg)
    print("  Config updated via Telegram /setup")
    _rerun_event.set()

def _setup_handle(text, trading_cfg):
    """Process button callbacks during interactive setup."""
    global _setup_state
    text_l = text.strip()

    if text_l == '/cancel':
        _setup_state = {'active': False}
        send_telegram("❌ Setup cancelled")
        return

    if not _setup_state.get('active'):
        return

    cfg = _setup_state['cfg']

    # Handle custom max amount input (user typed a number)
    awaiting_asset = _setup_state.get('awaiting_custom_max')
    awaiting_regime = _setup_state.get('awaiting_custom_max_regime', 'bull')
    if awaiting_asset and awaiting_asset in cfg:
        try:
            val = int(text_l.strip().replace('$', '').replace(',', ''))
            if 0 <= val <= 100000:
                cfg[awaiting_asset].setdefault(awaiting_regime, {})['max_position_usd'] = val
                _setup_state.pop('awaiting_custom_max', None)
                _setup_state.pop('awaiting_custom_max_regime', None)
                send_telegram(f"✅ {awaiting_asset} {awaiting_regime} max position → ${val:,}")
                _setup_send_menu(awaiting_asset, cfg[awaiting_asset])
                return
        except ValueError:
            # Not a number — clear awaiting state and continue normal routing
            _setup_state.pop('awaiting_custom_max', None)
            _setup_state.pop('awaiting_custom_max_regime', None)

    # /cfg_save — save and close
    if text_l == '/cfg_save':
        _setup_finish(trading_cfg)
        return

    # /cfg_back — back to asset picker
    if text_l == '/cfg_back':
        _setup_send_asset_picker(cfg)
        return

    # /cfg_{ASSET} — show asset menu
    for asset in cfg:
        if text_l == f'/cfg_{asset}':
            _setup_send_menu(asset, cfg[asset])
            return

        # /cfg_{ASSET}_toggle — toggle enabled
        if text_l == f'/cfg_{asset}_toggle':
            cfg[asset]['enabled'] = not cfg[asset].get('enabled', False)
            st = "🔵 ON" if cfg[asset]['enabled'] else "🔴 OFF"
            send_telegram(f"✅ {asset} → {st}")
            _setup_send_menu(asset, cfg[asset])
            return

        # /cfg_{ASSET}_auto — toggle auto-trade
        if text_l == f'/cfg_{asset}_auto':
            pos = load_position(asset)
            pos['auto_trade'] = not pos.get('auto_trade', False)
            save_position(asset, pos)
            st = "🔵 ON" if pos['auto_trade'] else "🔴 OFF"
            send_telegram(f"✅ {asset} auto-trade → {st}")
            _setup_send_menu(asset, cfg[asset])
            return

        # /cfg_{ASSET}_tp — toggle take-profit
        if text_l == f'/cfg_{asset}_tp':
            current_tp = cfg[asset].get('take_profit_pct', 0)
            if current_tp > 0:
                cfg[asset]['take_profit_pct'] = 0
                send_telegram(f"TP OFF for {asset}")
            else:
                cfg[asset]['take_profit_pct'] = 1.0
                send_telegram(f"TP ON for {asset} (1%)")
            _setup_send_menu(asset, cfg[asset])
            return

        # /cfg_{ASSET}_maker — toggle maker orders
        if text_l == f'/cfg_{asset}_maker':
            current = cfg[asset].get('use_maker_orders', False)
            cfg[asset]['use_maker_orders'] = not current
            status = 'ON (0% fee)' if not current else 'OFF (market)'
            send_telegram(f"Maker orders {status} for {asset}")
            _setup_send_menu(asset, cfg[asset])
            return

        # /cfg_{ASSET}_bull_conf — show bull confidence picker
        if text_l == f'/cfg_{asset}_bull_conf':
            _setup_send_confidence_picker(asset, cfg[asset], regime='bull')
            return

        # /cfg_{ASSET}_bear_conf — show bear confidence picker
        if text_l == f'/cfg_{asset}_bear_conf':
            _setup_send_confidence_picker(asset, cfg[asset], regime='bear')
            return

        # /cfg_{ASSET}_{regime}_confv_{value} — set bull/bear confidence
        for regime in ('bull', 'bear'):
            if text_l.startswith(f'/cfg_{asset}_{regime}_confv_'):
                try:
                    val = int(text_l.split('_')[-1])
                    cfg[asset].setdefault(regime, {})['min_confidence'] = val
                    send_telegram(f"✅ {asset} {regime} confidence → {val}%")
                    _setup_send_menu(asset, cfg[asset])
                except ValueError:
                    pass
                return

        # /cfg_{ASSET}_bull_max — show bull max position picker
        if text_l == f'/cfg_{asset}_bull_max':
            _setup_send_max_picker(asset, cfg[asset], regime='bull')
            return

        # /cfg_{ASSET}_bear_max — show bear max position picker
        if text_l == f'/cfg_{asset}_bear_max':
            _setup_send_max_picker(asset, cfg[asset], regime='bear')
            return

        # /cfg_{ASSET}_{regime}_maxv_{value} — set bull/bear max position
        for regime in ('bull', 'bear'):
            if text_l.startswith(f'/cfg_{asset}_{regime}_maxv_'):
                try:
                    val = int(text_l.split('_')[-1])
                    cfg[asset].setdefault(regime, {})['max_position_usd'] = val
                    send_telegram(f"✅ {asset} {regime} max position → ${val:,}")
                    _setup_send_menu(asset, cfg[asset])
                except ValueError:
                    pass
                return

        # /cfg_{ASSET}_{regime}_maxcustom — prompt for custom amount
        for regime in ('bull', 'bear'):
            if text_l == f'/cfg_{asset}_{regime}_maxcustom':
                _setup_state['awaiting_custom_max'] = asset
                _setup_state['awaiting_custom_max_regime'] = regime
                send_telegram(f"💰 Type the max position amount in USD for <b>{asset}</b> ({regime.upper()}) (e.g. 2500):")
                return

# ---- Command loop ----

def _telegram_command_loop(trading_cfg):
    """Background thread: polls Telegram commands every 5 seconds."""
    while not _stop_event.is_set():
        try:
            msg = check_telegram_commands()
            if not msg:
                pass
            elif _setup_state.get('active'):
                # During setup: /stop and /cancel bypass, everything else goes to wizard
                if msg.lower() == '/stop':
                    _setup_state['active'] = False
                    print("\n  STOP via Telegram")
                    send_telegram("🛑 <b>Trader Stopped</b>")
                    _stop_event.set()
                else:
                    _setup_handle(msg, trading_cfg)
            else:
                # Normal mode: only process commands
                cmd = msg.lower()
                if cmd == '/stop':
                    print("\n  STOP via Telegram")
                    send_telegram("🛑 <b>Trader Stopped</b>")
                    _stop_event.set()
                elif cmd == '/status':
                    _handle_status_command()
                elif cmd == '/summary':
                    _handle_summary_command()
                elif cmd == '/pause':
                    with _paused_lock:
                        _paused_flag[0] = True
                    send_telegram("⏸ <b>PAUSED</b> — /resume to continue")
                    print("  PAUSED via Telegram")
                elif cmd == '/resume':
                    with _paused_lock:
                        _paused_flag[0] = False
                    send_telegram("▶️ <b>RESUMED</b>")
                    print("  RESUMED via Telegram")
                elif cmd == '/sync':
                    sync_positions(trading_cfg, notify=True)
                    send_telegram("🔄 <b>Synced</b>")
                elif cmd == '/conf':
                    _handle_config_command()
                elif cmd == '/regime':
                    _handle_regime_command()
                elif cmd.startswith('/chart'):
                    _handle_chart_command(msg, trading_cfg)
                elif cmd == '/setup' or cmd.startswith('/cfg_'):
                    if not _setup_state.get('active'):
                        _setup_start(trading_cfg)
                    if cmd.startswith('/cfg_'):
                        _setup_handle(msg, trading_cfg)
                elif cmd.startswith('/optimize'):
                    _handle_optimize_command(msg)
                elif cmd == '/optstatus':
                    _handle_optstatus_command()
                elif cmd == '/help':
                    _handle_help_command()
        except Exception:
            pass
        _stop_event.wait(5)  # sleep 5s but wake immediately on stop

def _start_telegram_thread(trading_cfg):
    """Start background Telegram command listener. Returns the thread."""
    _stop_event.clear()
    _paused_flag[0] = False
    t = threading.Thread(target=_telegram_command_loop, args=(trading_cfg,), daemon=True)
    t.start()
    return t

def _is_paused():
    with _paused_lock:
        return _paused_flag[0]

def _set_paused(val):
    with _paused_lock:
        _paused_flag[0] = val


# ============================================================
# MAIN LOOP
# ============================================================
def _reload_trading_config(trading_cfg):
    """Hot-reload regime_config_ed.json into existing dict. Returns True if anything changed."""
    fresh = load_trading_config()
    changed = False
    for asset in list(fresh.keys()):
        if asset not in trading_cfg:
            trading_cfg[asset] = fresh[asset]
            changed = True
        else:
            for key in fresh[asset]:
                if trading_cfg[asset].get(key) != fresh[asset][key]:
                    old_val = trading_cfg[asset].get(key)
                    trading_cfg[asset][key] = fresh[asset][key]
                    print(f"  CONFIG RELOAD: {asset}.{key}: {old_val} -> {fresh[asset][key]}")
                    changed = True
    return changed


def _get_models_fingerprint(trading_cfg):
    """Build fingerprint of best_models CSV for all enabled assets. Returns dict {(asset,h): summary}."""
    fp = {}
    for asset, cfg in trading_cfg.items():
        if not cfg.get('enabled'):
            continue
        for h in AVAILABLE_HORIZONS:
            model_cfg = load_best_config(asset, horizon=h)
            if model_cfg:
                fp[(asset, h)] = f"{model_cfg['best_combo']}|w{model_cfg['best_window']}|{model_cfg['accuracy']:.2f}|{model_cfg.get('feature_set','A')}|γ{model_cfg.get('gamma',1.0)}"
            else:
                fp[(asset, h)] = None
    return fp


def run_all_once(trading_cfg, dry_run=False):
    """Sync positions, process all enabled assets."""
    # Hot-reload trading config before each cycle
    _reload_trading_config(trading_cfg)

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
        _cfg_h = cfg.get('horizon')
        if _cfg_h:
            if not load_best_config(asset, horizon=_cfg_h):
                continue
        else:
            if not load_best_config(asset, horizon=HORIZON_SHORT) and not load_best_config(asset, horizon=HORIZON_LONG):
                continue
        bull_h = cfg.get('bull', {}).get('horizon', '?')
        bear_h = cfg.get('bear', {}).get('horizon', '?')
        print(f"\n  --- {asset} (bull={bull_h}h / bear={bear_h}h) ---")
        r = process_asset(asset, cfg, dry_run=dry_run)
        results.append(r)

    # Refresh balances AFTER trades so Telegram shows current exchange state
    if not dry_run:
        balances = get_balances()

    # Send combined Telegram with inline buttons
    valid = [r for r in results if r is not None]
    if valid:
        msg = format_multi_asset_telegram(valid, dry_run=dry_run, balances=balances, trading_cfg=trading_cfg)
        send_telegram_with_buttons(msg, MAIN_BUTTONS)

    return results


def _send_startup_telegram(trading_cfg):
    """Send the startup/config summary banner to Telegram."""
    asset_lines = []
    for a, c in trading_cfg.items():
        if not c.get('enabled'):
            continue
        pos = load_position(a)
        auto = pos.get('auto_trade', False)
        icon = "🔵 ON" if auto else "🔴 OFF"
        bull_cfg = c.get('bull', {})
        bear_cfg = c.get('bear', {})
        det = c.get('regime_detector', {})
        if det.get('type') == 'sma_cross':
            det_label = f"sma_cross({det.get('params',{}).get('fast','')}/{det.get('params',{}).get('slow','')})"
        elif det.get('type') == 'named':
            det_label = det.get('params', {}).get('name', 'named')
        else:
            det_label = det.get('type', '?')
        tp_pct = c.get('take_profit_pct', 0)
        tp_str = f"TP: {tp_pct}%" if tp_pct > 0 else "TP: OFF"
        maker_str = "MAKER" if c.get('use_maker_orders') else "TAKER"
        asset_lines.append(
            f"  {a}: {det_label}\n"
            f"    BULL: {bull_cfg.get('horizon','?')}h @ {bull_cfg.get('min_confidence','?')}% | ${bull_cfg.get('max_position_usd', c.get('max_position_usd', 0)):,.0f}\n"
            f"    BEAR: {bear_cfg.get('horizon','?')}h @ {bear_cfg.get('min_confidence','?')}% | ${bear_cfg.get('max_position_usd', 0):,.0f}\n"
            f"    {icon} | {tp_str} | {maker_str}"
        )

    send_telegram(
        f"🚀 <b>Ed V2 Trader Started</b>\n\n"
        + "\n".join(asset_lines) + "\n\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"📱 /help for all commands | /regime for current state"
    )


def run_loop(trading_cfg, dry_run=False):
    mode = "DRY RUN" if dry_run else "LIVE"
    assets_str = ", ".join(a for a, c in trading_cfg.items() if c.get('enabled'))

    print(f"\n{'='*60}")
    print(f"  REVOLUT X MULTI-ASSET TRADER [ED V2 {mode}]")
    print(f"  Assets: {assets_str}")
    for asset, cfg in trading_cfg.items():
        if cfg.get('enabled'):
            pos = load_position(asset)
            auto = "AUTO" if pos.get('auto_trade') else "MANUAL"
            bull_cfg = cfg.get('bull', {})
            bear_cfg = cfg.get('bear', {})
            det = cfg.get('regime_detector', {})
            if det.get('type') == 'sma_cross':
                det_label = f"sma_cross({det.get('params',{}).get('fast','')}/{det.get('params',{}).get('slow','')})"
            elif det.get('type') == 'named':
                det_label = det.get('params', {}).get('name', 'named')
            else:
                det_label = det.get('type', '?')
            tp_pct = cfg.get('take_profit_pct', 0)
            tp_str = f"TP={tp_pct}%" if tp_pct > 0 else "TP=OFF"
            maker_str = "MAKER" if cfg.get('use_maker_orders') else "TAKER"
            print(f"  {asset}: {det_label} | bull={bull_cfg.get('horizon','?')}h@{bull_cfg.get('min_confidence','?')}% "
                  f"| bear={bear_cfg.get('horizon','?')}h@{bear_cfg.get('min_confidence','?')}% | {auto} | {tp_str} | {maker_str} | {pos['state'].upper()}")
    print(f"  Telegram: /help /status /conf /setup /balance /sync /pause /resume /stop /regime")
    print(f"  Hot-reload: every 5 min (config + models + positions)")
    print(f"{'='*60}")

    _flush_old_updates()

    # Sync clock with NTP to prevent "timestamp in the future" errors
    global _clock_offset_ms
    offset = _sync_clock_ntp()
    if abs(offset) > 500:
        _clock_offset_ms = offset
        print(f"  Clock drift: {offset:+d}ms — corrected for API calls")
    else:
        print(f"  Clock sync OK ({offset:+d}ms)")

    # Cancel any stale open orders from previous runs
    stale = cancel_all_open_orders()
    if stale > 0:
        print(f"  Cancelled {stale} stale open order(s) from previous run")
        send_telegram(f"🧹 Startup: cancelled {stale} stale open order(s)")

    # Initial sync — detect any existing positions on exchange
    print("\n  Initial position sync from Revolut X...")
    changes = sync_positions(trading_cfg, notify=True)
    if not changes:
        print("  Positions in sync ✓")

    _send_startup_telegram(trading_cfg)

    # Start background Telegram command listener
    print("  Telegram commands: background thread (responsive anytime)")
    _start_telegram_thread(trading_cfg)

    # Immediate first scan — don't wait for next hour
    print("\n  Running initial signal scan...")
    run_all_once(trading_cfg, dry_run=dry_run)

    cycle = 0
    while not _stop_event.is_set():
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
                pause_str = " [PAUSED]" if _is_paused() else ""
                print(f"\n  [{status}{pause_str}] Next: {next_hour.strftime('%H:%M')} ({wait_sec/60:.0f} min)")

                remaining = wait_sec
                last_sync = time.time()
                last_models_fp = _get_models_fingerprint(trading_cfg)
                while remaining > 0 and not _stop_event.is_set():
                    sleep_chunk = min(30, remaining)
                    _stop_event.wait(sleep_chunk)  # interruptible sleep
                    remaining -= sleep_chunk

                    # Break early if /setup triggered a re-run
                    if _rerun_event.is_set():
                        _rerun_event.clear()
                        _reload_trading_config(trading_cfg)
                        print("  Config changed via /setup — re-running signals")
                        run_all_once(trading_cfg, dry_run=dry_run)
                        last_models_fp = _get_models_fingerprint(trading_cfg)

                    # Every 5 minutes: sync positions + check config + check models
                    if time.time() - last_sync >= 300:
                        try:
                            # Position sync
                            changes = sync_positions(trading_cfg, notify=True)
                            if changes:
                                print(f"  Position sync: {len(changes)} changes")

                            # Check take-profit for invested positions
                            for asset, cfg in trading_cfg.items():
                                if cfg.get('enabled') and cfg.get('take_profit_pct', 0) > 0:
                                    if _check_take_profit(asset, trading_cfg, dry_run=dry_run):
                                        print(f"  Take profit triggered for {asset} — re-running signals")

                            # Hot-reload trading config
                            if _reload_trading_config(trading_cfg):
                                print(f"  Config reloaded at :{datetime.now().minute:02d}")

                            # Check if best_models CSV changed
                            new_fp = _get_models_fingerprint(trading_cfg)
                            if new_fp != last_models_fp:
                                for key in set(list(new_fp.keys()) + list(last_models_fp.keys())):
                                    old_v = last_models_fp.get(key)
                                    new_v = new_fp.get(key)
                                    if old_v != new_v:
                                        asset, h = key
                                        if old_v is None:
                                            print(f"  MODEL UPDATE: {asset} {h}h — NEW: {new_v}")
                                        elif new_v is None:
                                            print(f"  MODEL UPDATE: {asset} {h}h — REMOVED")
                                        else:
                                            print(f"  MODEL UPDATE: {asset} {h}h — {old_v} -> {new_v}")
                                last_models_fp = new_fp
                                print("  Models updated — re-running signals")
                                send_telegram("🔄 <b>Models updated</b> — re-running signals")
                                run_all_once(trading_cfg, dry_run=dry_run)
                                last_models_fp = _get_models_fingerprint(trading_cfg)

                            # Periodic NTP clock sync
                            ntp_off = _sync_clock_ntp()
                            if abs(ntp_off) > 500:
                                _clock_offset_ms = ntp_off
                                print(f"    🕐 Periodic clock sync: {ntp_off:+d}ms")

                            last_sync = time.time()
                        except Exception as e:
                            print(f"  Sync error: {e}")

            if _stop_event.is_set():
                break

            cycle += 1
            print(f"\n  --- Cycle {cycle} | {datetime.now().strftime('%H:%M:%S')} ---")

            # Wait for fresh candle (use first enabled asset)
            first_asset = next(a for a, c in trading_cfg.items() if c.get('enabled'))
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            expected = now_utc.replace(minute=0, second=0, microsecond=0)
            wait_for_fresh_candle(first_asset, expected)

            if _is_paused():
                print("  PAUSED — signals only")
                run_all_once(trading_cfg, dry_run=True)
            else:
                run_all_once(trading_cfg, dry_run=dry_run)

        except KeyboardInterrupt:
            print("\n  Stopped.")
            _stop_event.set()
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
    print("  REVOLUT X MULTI-ASSET TRADER [ED V2]")
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
                bull_c = cfg.get('bull', {})
                bear_c = cfg.get('bear', {})
                print(f"\n  {asset} [bull={bull_c.get('horizon','?')}h|bear={bear_c.get('horizon','?')}h]:")
                print(f"    State: {pos['state'].upper()} | Bull max: ${bull_c.get('max_position_usd',0):,.0f} | Bear max: ${bear_c.get('max_position_usd',0):,.0f}")
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
            if os.path.exists(REGIME_CONFIG_FILE): os.remove(REGIME_CONFIG_FILE)
            print("  All positions and config reset.")
            return

    # Load or create trading config
    trading_cfg = load_trading_config()

    if not os.path.exists(MODELS_CSV):
        print("\n  ERROR: No Ed models found! Run crypto_trading_system_ed.py Mode D first.")
        return

    # Interactive menu
    if len(args) == 0:
        print("\n  Current config:")
        print(f"  {'Asset':<6} {'Bull':>10} {'Bear':>10} {'Enabled':>8} {'Auto':>7} {'Position':>10}")
        print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*7} {'-'*10}")
        for asset, cfg in trading_cfg.items():
            pos = load_position(asset)
            bull_cfg = cfg.get('bull', {})
            bear_cfg = cfg.get('bear', {})
            bull_str = f"{bull_cfg.get('horizon','?')}h@{bull_cfg.get('min_confidence','?')}%"
            bear_str = f"{bear_cfg.get('horizon','?')}h@{bear_cfg.get('min_confidence','?')}%"
            enabled_str = "Yes" if cfg.get('enabled', True) else "No"
            auto_str    = "Yes" if pos.get('auto_trade') else "No"
            pos_str     = pos['state'].upper()
            model_mark  = "" if not cfg.get('enabled') else ""
            print(f"  {asset:<6} {bull_str:>10} {bear_str:>10} {enabled_str:>8} {auto_str:>7} {pos_str:>10}{model_mark}")

        print(f"\n  1. Start loop (hourly)")
        print(f"  2. Run once")
        print(f"  3. Dry run (once)")
        print(f"  4. Configure assets")
        print(f"  5. View trade history")
        print(f"  6. Check balance")
        print(f"  7. Setup Telegram")
        ch = input("\n  Enter 1-7: ").strip()

        if ch == '7': setup_telegram(); return
        elif ch == '6':
            bal = get_balances()
            for c, b in sorted(bal.items()):
                if b['total'] > 0: print(f"    {c}: {b['available']:.6f}")
            return
        elif ch == '5':
            for asset in trading_cfg:
                pos = load_position(asset)
                trades = pos.get('trades', [])[-5:]
                if trades:
                    print(f"\n  {asset} (last 5):")
                    for t in trades:
                        if t['action'] == 'BUY':
                            print(f"    BUY  ${t['price']:,.2f} | {t['time']}")
                        else:
                            print(f"    SELL ${t['price']:,.2f} | {t['time']} | {t.get('pnl_pct',0):+.1f}%")
            return
        elif ch == '4':
            for asset in trading_cfg:
                cfg = trading_cfg[asset]
                pos = load_position(asset)
                bull_c = cfg.get('bull', {})
                bear_c = cfg.get('bear', {})
                print(f"\n  --- {asset} ---")

                en = input(f"    Enabled (y/n) [{('y' if cfg.get('enabled', True) else 'n')}]: ").strip().lower()
                if en == 'y': cfg['enabled'] = True
                elif en == 'n': cfg['enabled'] = False

                auto_cur = 'y' if pos.get('auto_trade') else 'n'
                auto = input(f"    Auto-trade (y/n) [{auto_cur}]: ").strip().lower()
                if auto == 'y':
                    pos['auto_trade'] = True
                    save_position(asset, pos)
                elif auto == 'n':
                    pos['auto_trade'] = False
                    save_position(asset, pos)

                bull_conf_inp = input(f"    Bull min confidence [{bull_c.get('min_confidence', 95)}]: ").strip()
                if bull_conf_inp:
                    try:
                        cfg.setdefault('bull', {})['min_confidence'] = int(bull_conf_inp)
                    except ValueError:
                        pass

                bear_conf_inp = input(f"    Bear min confidence [{bear_c.get('min_confidence', 90)}]: ").strip()
                if bear_conf_inp:
                    try:
                        cfg.setdefault('bear', {})['min_confidence'] = int(bear_conf_inp)
                    except ValueError:
                        pass

                bull_max_inp = input(f"    Bull max USD [{bull_c.get('max_position_usd', 0):.0f}]: ").strip()
                if bull_max_inp:
                    try:
                        cfg.setdefault('bull', {})['max_position_usd'] = float(bull_max_inp)
                    except ValueError:
                        pass

                bear_max_inp = input(f"    Bear max USD [{bear_c.get('max_position_usd', 0):.0f}]: ").strip()
                if bear_max_inp:
                    try:
                        cfg.setdefault('bear', {})['max_position_usd'] = float(bear_max_inp)
                    except ValueError:
                        pass

                tp_input = input(f"    Take profit % (0=off, current={cfg.get('take_profit_pct', 0)}): ").strip()
                if tp_input:
                    try:
                        tp_val = float(tp_input)
                        cfg['take_profit_pct'] = tp_val if tp_val > 0 else 0
                    except ValueError:
                        pass

                maker_input = input(f"    Maker orders (y/n, current={'y' if cfg.get('use_maker_orders') else 'n'}): ").strip().lower()
                if maker_input == 'y': cfg['use_maker_orders'] = True
                elif maker_input == 'n': cfg['use_maker_orders'] = False

            save_trading_config(trading_cfg)
            print("\n  Config saved.")
            return
        elif ch == '3': dry_run = True
        elif ch == '2': pass  # run once below
        elif ch == '1': loop_mode = True

    # Check max positions — Ed reads them from bull/bear in regime config, not top-level
    for asset, cfg in trading_cfg.items():
        if not cfg.get('enabled'):
            continue
        bull_max = cfg.get('bull', {}).get('max_position_usd', 0)
        bear_max = cfg.get('bear', {}).get('max_position_usd', 0)
        top_max = cfg.get('max_position_usd', 0)
        if bull_max <= 0 and bear_max <= 0 and top_max <= 0:
            print(f"  ⚠️ {asset}: No max_position_usd set in bull/bear config. Trades will be skipped.")
            print(f"     Edit config/regime_config_ed.json to set max_position_usd.")

    if loop_mode:
        run_loop(trading_cfg, dry_run=dry_run)
    else:
        run_all_once(trading_cfg, dry_run=dry_run)


if __name__ == '__main__':
    main()
