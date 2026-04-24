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
# PER-CYCLE METRICS (Fix #9 — 2026-04-24)
# Append one row per (asset, cycle) to output/cycle_metrics.csv.
# Provides forensics for "why was cycle X slow?" / "when did pinning start?" /
# "which feature went NaN on 04-23?" — answerable with pd.read_csv now.
# ============================================================
CYCLE_METRICS_CSV = os.path.join(os.path.dirname(__file__), 'output', 'cycle_metrics.csv')

CYCLE_METRICS_COLUMNS = [
    'timestamp',           # cycle start ISO
    'asset',
    'regime',              # bull / bear / error
    'signal',              # BUY / SELL / HOLD
    'confidence',          # 0-100, 2 decimals
    'horizon',             # active horizon (e.g. 5)
    'position_state',      # cash / invested
    'price',
    # Feature health (from generate_live_signal)
    'n_features_expected',       # len(optimal_features)
    'n_features_available',      # intersect(optimal_features, build)
    'n_features_nan_inference',  # NaN count on the inference row
    'feature_set_a_fallback',    # bool — model fell back to FEATURE_SET_A
    # Staleness
    'inference_row_age_h',       # latest_raw_dt - inference_row_dt, in hours
    # Timings
    'p1_duration_sec',
    'p2_duration_sec',
    'feature_build_sec',
    'model_fit_sec',
    'total_cycle_sec',
]


def _append_cycle_metrics(row_dict):
    """Append a per-cycle per-asset row to cycle_metrics.csv. Never raises.
    Keys missing from row_dict become empty cells."""
    import csv
    try:
        os.makedirs(os.path.dirname(CYCLE_METRICS_CSV), exist_ok=True)
        file_exists = os.path.exists(CYCLE_METRICS_CSV)
        with open(CYCLE_METRICS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CYCLE_METRICS_COLUMNS, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow({c: row_dict.get(c, '') for c in CYCLE_METRICS_COLUMNS})
    except Exception as _e:
        print(f"  [!] cycle_metrics write failed: {_e}")


# ============================================================
# PER-ASSET TRADE MUTATION LOCK (Fix N1 — 2026-04-24)
# Prevents the main-loop auto-cycle and the Telegram command thread from
# BOTH initiating a trade on the same asset simultaneously. Each call site
# try-acquires non-blocking; if busy, bails out with a user-facing message
# rather than waiting 30-180s on a maker-order completion.
# ============================================================
_asset_trade_locks = {}                # {asset: threading.Lock}
_asset_trade_locks_guard = threading.Lock()

def _get_asset_trade_lock(asset):
    """Return (creating if needed) a per-asset lock. Thread-safe lazy init."""
    with _asset_trade_locks_guard:
        lk = _asset_trade_locks.get(asset)
        if lk is None:
            lk = threading.Lock()
            _asset_trade_locks[asset] = lk
        return lk


# ============================================================
# RATE-LIMITED TELEGRAM ALERTS (Fix #1 — 2026-04-24)
# Prevents per-cycle Telegram spam when a source is persistently failing.
# Keyed so different (source, context) pairs alert independently.
# ============================================================
_TELEGRAM_RATE_LIMIT = {}  # key -> last_sent_epoch

def _rate_limited_telegram(key, msg, cooldown_sec=3600):
    """Send Telegram alert; skip if same `key` alerted within cooldown_sec.
    Returns True if sent, False if suppressed by cooldown.
    Never raises — safe for use in except blocks."""
    import time as _t
    now = _t.time()
    last = _TELEGRAM_RATE_LIMIT.get(key, 0)
    if now - last < cooldown_sec:
        return False
    _TELEGRAM_RATE_LIMIT[key] = now
    try:
        send_telegram(msg)
        return True
    except Exception:
        return False


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
    """Atomic write — trader hot-reloads regime_config_ed.json on mtime change;
    a direct `open('w') + json.dump` creates a race window where readers could
    see an empty or partial file. tempfile + os.replace is atomic on modern
    filesystems."""
    os.makedirs('config', exist_ok=True)
    tmp = REGIME_CONFIG_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, REGIME_CONFIG_FILE)


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


def _execute_maker_order(symbol, size, side, maker_window=180, check_interval=10):
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

_position_lock = threading.Lock()


def _now_utc_iso():
    """Current time as UTC ISO 8601 with Z suffix — what new entry_time writes use."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _parse_entry_time_utc(s):
    """Parse entry_time string to a UTC-aware datetime.

    Accepts both the new format ('2026-04-16T01:00:00Z' / '+00:00') and the
    legacy naive-local format ('2026-04-16 01:00' optionally with ' (synced)').
    Legacy strings are interpreted as local time and converted to UTC. Returns
    None if the string is empty or unparseable.
    """
    if not s:
        return None
    s = s.replace(' (synced)', '').strip()
    if not s:
        return None
    try:
        if 'T' in s or 'Z' in s or '+' in s[10:]:
            return datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(timezone.utc)
        naive_local = datetime.strptime(s, '%Y-%m-%d %H:%M')
        return naive_local.astimezone().astimezone(timezone.utc)
    except Exception:
        return None


def _format_entry_time_local(s):
    """Format an entry_time string for human display in local time."""
    dt = _parse_entry_time_utc(s)
    if dt is None:
        return s or ''
    return dt.astimezone().strftime('%Y-%m-%d %H:%M')

def load_position(asset):
    path = _position_file(asset)
    with _position_lock:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [!] Corrupted position file {path}: {e}. Falling back to cash state.")
    return {
        'asset': asset, 'state': 'cash', 'entry_price': 0,
        'entry_time': '', 'base_amount': 0, 'usd_invested': 0,
        'trades': [], 'auto_trade': False,
    }

def save_position(asset, pos):
    os.makedirs('config', exist_ok=True)
    path = _position_file(asset)
    tmp = path + '.tmp'
    with _position_lock:
        with open(tmp, 'w') as f:
            json.dump(pos, f, indent=2)
        os.replace(tmp, path)


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
            # Detected manual BUY — update local state.
            # Fix N2 (2026-04-24): entry_price is current market mid, NOT the
            # actual fill price. Sync runs every 5 min so the drift can be up
            # to ±1% depending on volatility. Acceptable per user decision
            # 2026-04-24 (option D — don't call Revolut X trades API).
            # The (synced) tag on entry_time flags the trade as approximate;
            # consumers of PnL history must tolerate ±0.5-1% error on these.
            pos['state'] = 'invested'
            pos['base_amount'] = actual_amount
            pos['entry_price'] = price  # approximate — see Fix N2 note above
            pos['entry_time'] = _now_utc_iso() + ' (synced)'
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
            # Detected manual SELL — update local state.
            # Fix N2 (2026-04-24): sell `price` is current mid at sync time,
            # NOT the actual sell fill price. Both sides of this PnL calc
            # carry ±1% noise: entry_price if the original BUY was also a
            # sync, and `price` here for the SELL. So reported PnL on a
            # synced-buy + synced-sell could be off by ±2% cumulatively.
            # Acceptable per user 2026-04-24. The (synced) tag in the trade
            # log flags the trade; treat its PnL numbers as approximate.
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
# RALLY-COOLDOWN GATE (V7: rr8>=3% OR rr36>=5.5% -> 30h block on BUYs)
# ============================================================
def _rally_cfg_for_regime(trading_cfg, regime_label):
    """Return the rally-cooldown config dict for the active regime, with fallback.
    Priority: trading_cfg[regime_label]['rally_cooldown'] → trading_cfg['rally_cooldown'] → None."""
    block = trading_cfg.get(regime_label) if isinstance(trading_cfg.get(regime_label), dict) else None
    if block:
        rc = block.get('rally_cooldown')
        if rc is not None:
            return rc
    return trading_cfg.get('rally_cooldown')


def _gate_on_for_regime(cfg_asset, regime_label):
    """True if the rally-cooldown gate is enabled for the given regime.
    Checks per-regime block first, falls back to asset-level."""
    block = cfg_asset.get(regime_label) if isinstance(cfg_asset.get(regime_label), dict) else None
    if block and isinstance(block.get('rally_cooldown'), dict):
        return bool(block['rally_cooldown'].get('enabled'))
    return bool((cfg_asset.get('rally_cooldown') or {}).get('enabled'))


def _set_gate_enabled(cfg_asset, regime_label, enabled):
    """Set enabled flag for the gate at a given regime.
    regime_label = 'bull' | 'bear' | None (None = asset-level + all regimes)."""
    default_rc = {'h_short': 8, 'h_long': 36, 't_short_pct': 3.0, 't_long_pct': 5.5, 'cd_hours': 30}
    if regime_label in ('bull', 'bear'):
        block = cfg_asset.setdefault(regime_label, {})
        rc = block.setdefault('rally_cooldown', dict(default_rc))
        rc['enabled'] = enabled
    else:
        # Asset-level + propagate to per-regime if those blocks exist
        rc = cfg_asset.setdefault('rally_cooldown', dict(default_rc))
        rc['enabled'] = enabled
        for r in ('bull', 'bear'):
            block = cfg_asset.get(r)
            if isinstance(block, dict) and isinstance(block.get('rally_cooldown'), dict):
                block['rally_cooldown']['enabled'] = enabled


def _update_rally_cooldown(asset, df_raw, position, rc_cfg):
    """Trigger detection — must run EVERY tick, regardless of position state.
    Scans the last cd_hours bars so rallies that fired while the engine was
    offline are still captured (implied cooldown-until adjusted for elapsed time).
    Returns (rs_pct, rl_pct, fresh_trigger_bool) for logging, or None if disabled."""
    if not rc_cfg or not rc_cfg.get('enabled'):
        return None
    h_s = int(rc_cfg['h_short']); h_l = int(rc_cfg['h_long'])
    t_s = float(rc_cfg['t_short_pct']); t_l = float(rc_cfg['t_long_pct'])
    cd_h = int(rc_cfg['cd_hours'])
    if df_raw is None or len(df_raw) < h_l + 1:
        return None
    closes = df_raw['close'].values
    # Bar timestamps for anchoring cooldown to actual candle time (not wall-clock)
    # Each row's datetime is the bar's OPEN time; trigger is observable once the bar closes.
    dt_series = pd.to_datetime(df_raw['datetime'])
    if dt_series.dt.tz is None:
        dt_series = dt_series.dt.tz_localize('UTC')
    else:
        dt_series = dt_series.dt.tz_convert('UTC')
    bar_times = dt_series.values  # numpy array of tz-aware Timestamps
    now = datetime.now(timezone.utc)

    # Current-window values (returned for logging)
    rs = (closes[-1] / closes[-1 - h_s] - 1.0) * 100.0
    rl = (closes[-1] / closes[-1 - h_l] - 1.0) * 100.0

    # Catch-up scan: check each of the last cd_h bars. If a past bar would have
    # triggered, compute the implied cooldown-until from the BAR'S OWN timestamp
    # (trigger_time + cd_h) so we don't over-extend cooldown when `now` drifts
    # past the last closed candle. Keep the latest still-active one.
    max_lookback = min(cd_h, len(closes) - h_l - 1)
    best_until = None
    for bars_ago in range(max_lookback + 1):
        end_idx = -1 - bars_ago
        rs_k = (closes[end_idx] / closes[end_idx - h_s] - 1.0) * 100.0
        rl_k = (closes[end_idx] / closes[end_idx - h_l] - 1.0) * 100.0
        if rs_k >= t_s or rl_k >= t_l:
            # Use the bar's own open timestamp as the trigger anchor so cooldown is
            # deterministic regardless of when `now` lands within the current hour.
            # Matches the intuitive mental model: "rally at 14:00 + 16h cd = 06:00 next day".
            trigger_time = pd.Timestamp(bar_times[end_idx]).to_pydatetime()
            if trigger_time.tzinfo is None:
                trigger_time = trigger_time.replace(tzinfo=timezone.utc)
            implied_until = trigger_time + timedelta(hours=cd_h)
            if implied_until > now and (best_until is None or implied_until > best_until):
                best_until = implied_until

    fired = False
    if best_until is not None:
        cur_until_str = position.get('rally_cooldown_until', '')
        cur_until = None
        if cur_until_str:
            try:
                cur_until = datetime.fromisoformat(cur_until_str.replace('Z', '+00:00'))
            except Exception:
                cur_until = None
        if cur_until is None or best_until > cur_until:
            position['rally_cooldown_until'] = best_until.strftime('%Y-%m-%dT%H:%M:%SZ')
            fired = cur_until is None or best_until > cur_until + timedelta(minutes=30)
    return (rs, rl, fired)


def _is_rally_cooldown_active(position):
    """Returns (active, hours_left, until_str)."""
    until_str = position.get('rally_cooldown_until', '')
    if not until_str:
        return False, 0.0, ''
    try:
        until = datetime.fromisoformat(until_str.replace('Z', '+00:00'))
    except Exception:
        return False, 0.0, until_str
    now = datetime.now(timezone.utc)
    if now >= until:
        return False, 0.0, until_str
    return True, (until - now).total_seconds() / 3600.0, until_str


def _shield_on_for_regime(trading_cfg, regime_label):
    """Return bool: hold-shield ON for the active regime.
    Priority: regime block's hold_shield → asset-level hold_shield → default True."""
    block = trading_cfg.get(regime_label, {}) if isinstance(trading_cfg.get(regime_label), dict) else {}
    if 'hold_shield' in block:
        return bool(block['hold_shield'])
    return bool(trading_cfg.get('hold_shield', True))


# ============================================================
# PROCESS ONE ASSET
# ============================================================
def process_asset(asset, trading_cfg, dry_run=False, cycle_metrics=None):
    """Generate signals and execute for one asset. Returns result dict.
    Fix #9 (2026-04-24): cycle_metrics optional dict — if provided, we
    populate one row per horizon with health/timing info."""
    position = load_position(asset)
    symbol = trading_cfg.get('symbol', f'{asset}-USD')
    max_usd = trading_cfg.get('max_position_usd', 0)

    # Fix #1 (2026-04-24): removed redundant download_asset call. The P1 block
    # in run_all_once already refreshed this asset's OHLCV at cycle start.
    # The old silent `except Exception: pass` had hidden every failure here.
    df_raw = load_data(asset)
    if df_raw is None:
        _rate_limited_telegram(
            f'load_data_{asset}',
            f'🚨 {asset}: load_data returned None — CSV missing or unreadable. Check P1 download logs.',
        )
        return None

    # Detect regime — determines horizon, confidence, and max_position
    # Fix #2 (2026-04-24): detect_regime now returns 'error' sentinel when the
    # detector fails structurally (unknown name, exception, PySR NaN inputs).
    # Refuse to trade this cycle; Telegram alert already fired by detector.
    # Cold-start insufficient-data still falls through to 'bull' (graceful).
    regime_label, regime_cfg = detect_regime(asset, df_raw)
    if regime_label == 'error':
        print(f"  [!!] {asset}: regime detector returned ERROR — skipping cycle")
        return None
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
    # Fix #9: pass metrics_out dict to capture feature-health + timings
    first = True
    per_horizon_metrics = {}  # {horizon: {health fields}}
    for h in list(sigs_by_horizon.keys()):
        cfg = sigs_by_horizon[h]
        m_out = {}
        sig = generate_live_signal(asset, cfg, df_raw=df_raw, verbose=first, metrics_out=m_out)
        sigs_by_horizon[h] = sig
        per_horizon_metrics[h] = m_out
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

    # Rally-cooldown trigger detection — runs EVERY tick (regardless of position state),
    # so a rally that fires while invested still extends the cooldown for the next BUY.
    # Per-regime: use the current regime's gate params (falls back to asset-level for legacy).
    rc_cfg = _rally_cfg_for_regime(trading_cfg, regime_label)
    rc_update = _update_rally_cooldown(asset, df_raw, position, rc_cfg)
    if rc_update is not None:
        rs_pct, rl_pct, rc_fresh = rc_update
        rc_active, rc_hours_left, _ = _is_rally_cooldown_active(position)
        h_s = int(rc_cfg['h_short']); h_l = int(rc_cfg['h_long'])
        rc_tag = f"[{regime_label}] rr{h_s}={rs_pct:+.2f}% rr{h_l}={rl_pct:+.2f}%"
        if rc_fresh:
            rc_tag += f" | TRIGGER -> cd active {rc_hours_left:.1f}h"
            send_telegram(f"⏸ {asset} rally trigger fired: {rc_tag}")
        elif rc_active:
            rc_tag += f" | cd {rc_hours_left:.1f}h left"
        print(f"  {asset} rally-cd: {rc_tag}")
        save_position(asset, position)

    # Log - show all available horizons
    sig_strs = []
    for h in sorted(sigs_by_horizon.keys()):
        s = sigs_by_horizon[h]
        sig_strs.append(f"{h}h={s['signal']}({s['confidence']:.2f}%)" if s else f"{h}h=N/A")
    print(f"  {asset}: {' | '.join(sig_strs)} → {action} ({confidence:.2f}%) [{regime_label.upper()}] [{reason}] | pos={position['state']}")

    # Log signal for /chart command
    _log_signal(asset, price, sigs_by_horizon, action, confidence)

    # Expose sig_short, sig_long for Telegram message
    sig_short = sigs_by_horizon.get(HORIZON_SHORT)
    sig_long = sigs_by_horizon.get(HORIZON_LONG)

    # Execute
    executed = False
    pnl_msg = ""
    hold_override_active = False
    disaster_brake = False

    # Disaster brake: force SELL on severe unrealized loss, bypassing shield.
    # Acts as free downside insurance — only fires in rare catastrophic moves.
    # Configured per-asset via `disaster_brake_pct` (default 0 = OFF).
    if position['state'] == 'invested' and position.get('entry_price', 0) > 0:
        brake_pct = float(trading_cfg.get('disaster_brake_pct', 0))
        if brake_pct > 0:
            cur_pnl = (price - position['entry_price']) / position['entry_price'] * 100
            if cur_pnl <= -brake_pct:
                print(f"    ⚠️ DISASTER BRAKE: PnL {cur_pnl:+.2f}% ≤ -{brake_pct}% — forcing SELL")
                send_telegram(f"⚠️ {asset} disaster brake triggered: PnL {cur_pnl:+.2f}% ≤ -{brake_pct}%")
                action = 'SELL'
                disaster_brake = True

    # Fix N1 (2026-04-24): acquire the per-asset trade lock BEFORE any auto
    # trade execution. Non-blocking — if the Telegram thread is mid-manual-
    # trade for this asset, skip this cycle's auto trade rather than double-
    # trading. Released in finally after the trade blocks.
    _asset_lock = _get_asset_trade_lock(asset)
    _trade_lock_held = False
    if action in ('BUY', 'SELL') and not dry_run and position.get('auto_trade'):
        if _asset_lock.acquire(blocking=False):
            _trade_lock_held = True
        else:
            print(f"    ⚠ {asset}: manual trade in progress — auto {action} SKIPPED this cycle")
            _rate_limited_telegram(
                f'auto_skip_manual_trade_{asset}',
                f"⚠ {asset}: manual trade active via Telegram — auto-{action} skipped this cycle.",
            )
            action = 'HOLD'

    if action == 'BUY' and position['state'] == 'cash' and max_usd > 0:
        rc_active, rc_hours_left, _ = _is_rally_cooldown_active(position)
        if rc_active:
            msg = f'rally_cooldown {rc_hours_left:.1f}h left'
            print(f"    ⏸ {asset} BUY skipped — {msg}")
            send_telegram(f"⏸ {asset} BUY skipped — {msg}")
            action = 'HOLD'

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
                    position['usd_invested'] = position['base_amount'] * position['entry_price']
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
            position['entry_time'] = _now_utc_iso()
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
        # Shield is per-regime; falls back to asset-level hold_shield for legacy config
        # Disaster brake (if triggered above) bypasses shield entirely.
        shield_on = (not disaster_brake) and _shield_on_for_regime(trading_cfg, regime_label)
        min_sell_pnl = trading_cfg.get('min_sell_pnl_pct', 0)
        max_hold_h = trading_cfg.get('max_hold_hours', 10)
        if shield_on and min_sell_pnl > 0 and position.get('entry_price', 0) > 0:
            cur_pnl = (price - position['entry_price']) / position['entry_price'] * 100
            hours_held = 0
            entry_dt_utc = _parse_entry_time_utc(position.get('entry_time'))
            if entry_dt_utc is not None:
                hours_held = (datetime.now(timezone.utc) - entry_dt_utc).total_seconds() / 3600
            # Quick-release: if the model flips SELL with very high confidence shortly
            # after entry, the BUY was likely a false positive — trust the signal.
            # Guards the 2026-04-19 failure mode where shield held through a 12h trend
            # reversal while the model screamed SELL @ 97-99% for 9 consecutive cycles.
            # Shield quick-release: default OFF (empirically neutral in backtests —
            # see 2026-04-19 test_qr_windows.py). Opt-in by setting
            # shield_quick_release.enabled=true in regime_config_ed.json.
            qr_cfg = trading_cfg.get('shield_quick_release', {})
            qr_enabled = bool(qr_cfg.get('enabled', False))
            qr_min_conf = float(qr_cfg.get('min_sell_conf', 95))
            qr_max_hours = float(qr_cfg.get('max_hours', 3))
            quick_release = (qr_enabled and hours_held <= qr_max_hours
                             and confidence >= qr_min_conf)
            if quick_release:
                print(f"    🛡→ QUICK-RELEASE: SELL @ {confidence:.2f}% confidence within "
                      f"{hours_held:.1f}h of entry (thr {qr_min_conf}%/{qr_max_hours}h) — trusting model")
                send_telegram(f"🛡→ {asset} shield released: strong SELL ({confidence:.2f}%) "
                              f"at {hours_held:.1f}h / PnL {cur_pnl:+.2f}%")
            elif cur_pnl < min_sell_pnl and hours_held < max_hold_h:
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

    # Fix N1: release per-asset trade lock now that trade execution is done
    if _trade_lock_held:
        try:
            _asset_lock.release()
        except Exception:
            pass

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

    # Fix #9 (2026-04-24): emit one cycle_metrics row per active horizon
    if cycle_metrics is not None:
        for h, h_metrics in per_horizon_metrics.items():
            row = {
                'timestamp': cycle_metrics.get('timestamp', ''),
                'asset': asset,
                'regime': regime_label,
                'signal': (sigs_by_horizon.get(h) or {}).get('signal', ''),
                'confidence': (sigs_by_horizon.get(h) or {}).get('confidence', ''),
                'horizon': h,
                'position_state': position.get('state', ''),
                'price': round(price, 2) if price else '',
                'n_features_expected': h_metrics.get('n_features_expected', ''),
                'n_features_available': h_metrics.get('n_features_available', ''),
                'n_features_nan_inference': h_metrics.get('n_features_nan_inference', ''),
                'feature_set_a_fallback': h_metrics.get('feature_set_a_fallback', ''),
                'inference_row_age_h': h_metrics.get('inference_row_age_h', ''),
                'feature_build_sec': h_metrics.get('feature_build_sec', ''),
                'model_fit_sec': h_metrics.get('model_fit_sec', ''),
                # P1/P2/total populated at cycle level, merged at flush time
            }
            cycle_metrics.setdefault('rows', []).append(row)

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
            return f"{e}{h}h:{sig['signal']}({sig['confidence']:.2f}%)"

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
                _edt_utc = _parse_entry_time_utc(position.get('entry_time'))
                if _edt_utc is not None:
                    _held_h = (datetime.now(timezone.utc) - _edt_utc).total_seconds() / 3600
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

        # Get live price from exchange
        price = get_asset_price(symbol)
        actual_held = balances.get(asset, {}).get('total', 0)
        actual_usd = actual_held * price if price > 0 else 0

        # Get last 4 hours + RSI + live regime from data
        last_prices = []
        rsi = 0
        regime_label = f"bull={cfg.get('bull',{}).get('horizon','?')}h|bear={cfg.get('bear',{}).get('horizon','?')}h"
        try:
            df_raw = load_data(asset)
            if df_raw is not None:
                try:
                    regime, active_cfg = detect_regime(asset, df_raw)
                    if regime == 'error':
                        # Fix #4 (2026-04-24): 'error' sentinel should display as a
                        # warning, not fall through to the bear icon.
                        regime_label = f"⚠️ DETECTOR ERROR (refusing trades)"
                    else:
                        icon = '🔵' if regime == 'bull' else '🔴'
                        regime_label = f"{icon} {regime.upper()} {active_cfg.get('horizon','?')}h@{active_cfg.get('min_confidence','?')}%"
                except Exception:
                    pass
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
            lines.append(f"  Entry: ${pos['entry_price']:,.2f} | {_format_entry_time_local(pos.get('entry_time', ''))}")
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

    send_telegram_with_buttons("\n".join(lines), _main_buttons())

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


def _handle_help_command():
    """Send list of available commands."""
    send_telegram_with_buttons(
        "📱 <b>Commands</b>\n\n"
        "/status — Positions, P&L, balance, regime\n"
        "/chart [ASSET] [HORIZON] — e.g. /chart ETH 7d\n"
        "/setup — View & edit config\n"
        "/sync — Force-sync positions\n"
        "/gate [ASSET on|off|clear] — V7 rally-cooldown gate\n"
        "/pause /resume — Pause or resume signals\n"
        "/stop — Stop the trader\n\n"
        "💡 Buy/Sell/Shield via main buttons. Typed form accepts [ASSET] [USD] args.",
        _main_buttons()
    )


def _handle_manual_buy_command(msg, trading_cfg):
    """Manual maker buy via Telegram. Fetches fresh price, executes maker order, updates position.
    /buy           → first enabled asset, full max_position_usd
    /buy ETH       → buy ETH with its max_position_usd
    /buy ETH 500   → buy ETH with $500
    """
    print(f"\n  [/buy] command received: {msg!r}", flush=True)
    try:
        _manual_buy_impl(msg, trading_cfg)
    except Exception as e:
        import traceback
        print(f"  [/buy] ERROR: {e}", flush=True)
        traceback.print_exc()
        try:
            send_telegram(f"❌ /buy crashed: {type(e).__name__}: {e}")
        except Exception:
            pass


def _manual_buy_impl(msg, trading_cfg):
    parts = msg.split()
    enabled = [a for a, c in trading_cfg.items() if c.get('enabled')]
    if not enabled:
        send_telegram("❌ No enabled asset")
        return
    asset = parts[1].upper() if len(parts) >= 2 else enabled[0]
    amount_usd = None
    if len(parts) >= 3:
        try:
            amount_usd = float(parts[2])
        except ValueError:
            send_telegram(f"❌ Bad amount: {parts[2]}")
            return
    if asset not in trading_cfg or not trading_cfg[asset].get('enabled'):
        send_telegram(f"❌ {asset} not enabled")
        return

    # Fix N1 (2026-04-24): acquire per-asset trade lock so auto-cycle can't
    # start its own trade on this asset while we run. Non-blocking + fail-fast
    # so user gets a 'busy' reply within seconds, not 180s.
    _asset_lock = _get_asset_trade_lock(asset)
    if not _asset_lock.acquire(blocking=False):
        send_telegram(f"⏳ {asset}: auto-cycle trade in progress — retry in ~30s")
        return

    try:
        _manual_buy_impl_locked(msg, trading_cfg, asset, amount_usd)
    finally:
        try:
            _asset_lock.release()
        except Exception:
            pass


def _manual_buy_impl_locked(msg, trading_cfg, asset, amount_usd):
    """Trade execution body — runs inside the per-asset trade lock."""
    cfg = trading_cfg[asset]
    symbol = cfg.get('symbol', f'{asset}-USD')
    pos = load_position(asset)
    if pos['state'] == 'invested':
        send_telegram(f"⚠️ {asset} already invested — /sell first")
        return

    bid, ask = get_best_bid_ask(symbol)
    if bid <= 0 or ask <= 0:
        send_telegram(f"❌ {asset} price unreachable")
        return
    spread_bps = (ask - bid) / bid * 10000

    balances = get_balances()
    if balances is None:
        send_telegram("❌ Balance API failed")
        return
    usd_avail = balances.get('USD', {}).get('available', 0)
    max_usd = cfg.get('bull', {}).get('max_position_usd', 0) or cfg.get('max_position_usd', 0)
    if amount_usd is None:
        amount_usd = min(max_usd, usd_avail) if max_usd else usd_avail
    amount_usd = min(amount_usd, usd_avail)
    amount_usd = math.floor(amount_usd * 100) / 100 - 0.01
    if amount_usd < MIN_TRADE_USD:
        send_telegram(f"❌ ${amount_usd:.2f} below ${MIN_TRADE_USD} min (USD avail ${usd_avail:,.2f})")
        return

    send_telegram(
        f"🔵 <b>{asset} MAKER BUY</b>\n"
        f"Price: bid ${bid:,.2f} / ask ${ask:,.2f} ({spread_bps:.1f}bps)\n"
        f"Amount: ${amount_usd:,.2f}\n"
        f"Executing..."
    )

    status, data = execute_maker_buy(symbol, amount_usd)
    if status in (200, 201):
        order = data.get('data', data)
        mid = (bid + ask) / 2
        fill_price = float(order.get('average_fill_price', mid))
        filled_size = float(order.get('filled_size', amount_usd / fill_price if fill_price > 0 else 0))

        pos['state'] = 'invested'
        pos['base_amount'] = filled_size
        pos['entry_price'] = fill_price
        pos['usd_invested'] = filled_size * fill_price
        pos['entry_time'] = _now_utc_iso()
        pos['trades'].append({
            'action': 'BUY', 'price': fill_price,
            'time': pos['entry_time'], 'usd': pos['usd_invested'], 'auto': False, 'manual': True,
        })
        save_position(asset, pos)

        time.sleep(2)
        pb = get_balances()
        crypto = pb.get(asset, {}).get('total', filled_size) if pb else filled_size
        usd_left = pb.get('USD', {}).get('available', 0) if pb else 0
        send_telegram(
            f"✅ <b>{asset} BUY filled</b>\n"
            f"Fill: ${fill_price:,.2f}\n"
            f"Size: {crypto:.6f} {asset}\n"
            f"USD left: ${usd_left:,.2f}"
        )
    else:
        send_telegram(f"❌ {asset} BUY failed: {status} {data}")


def _handle_manual_sell_command(msg, trading_cfg):
    """Manual maker sell via Telegram. Sells ALL available holdings via maker order.
    /sell          → first enabled asset
    /sell ETH      → sell ETH
    """
    print(f"\n  [/sell] command received: {msg!r}", flush=True)
    try:
        _manual_sell_impl(msg, trading_cfg)
    except Exception as e:
        import traceback
        print(f"  [/sell] ERROR: {e}", flush=True)
        traceback.print_exc()
        try:
            send_telegram(f"❌ /sell crashed: {type(e).__name__}: {e}")
        except Exception:
            pass


def _manual_sell_impl(msg, trading_cfg):
    parts = msg.split()
    enabled = [a for a, c in trading_cfg.items() if c.get('enabled')]
    if not enabled:
        send_telegram("❌ No enabled asset")
        return
    asset = parts[1].upper() if len(parts) >= 2 else enabled[0]
    if asset not in trading_cfg or not trading_cfg[asset].get('enabled'):
        send_telegram(f"❌ {asset} not enabled")
        return

    # Fix N1 (2026-04-24): per-asset trade lock (same pattern as /buy)
    _asset_lock = _get_asset_trade_lock(asset)
    if not _asset_lock.acquire(blocking=False):
        send_telegram(f"⏳ {asset}: auto-cycle trade in progress — retry in ~30s")
        return

    try:
        _manual_sell_impl_locked(msg, trading_cfg, asset)
    finally:
        try:
            _asset_lock.release()
        except Exception:
            pass


def _manual_sell_impl_locked(msg, trading_cfg, asset):
    """Trade execution body — runs inside the per-asset trade lock."""
    cfg = trading_cfg[asset]
    symbol = cfg.get('symbol', f'{asset}-USD')
    pos = load_position(asset)

    bid, ask = get_best_bid_ask(symbol)
    if bid <= 0 or ask <= 0:
        send_telegram(f"❌ {asset} price unreachable")
        return
    spread_bps = (ask - bid) / bid * 10000

    balances = get_balances()
    if balances is None:
        send_telegram("❌ Balance API failed")
        return
    actual_held = balances.get(asset, {}).get('available', 0)
    total_held = balances.get(asset, {}).get('total', 0)

    if actual_held <= 0 < total_held:
        send_telegram(f"⚠️ {asset} locked — cancelling open orders")
        cancel_all_open_orders(symbol)
        time.sleep(2)
        balances = get_balances()
        if balances is None:
            send_telegram("❌ Balance API failed after cancel")
            return
        actual_held = balances.get(asset, {}).get('available', 0)

    if actual_held <= 0:
        send_telegram(f"❌ {asset} balance = 0")
        return

    entry_price = pos.get('entry_price', 0) if pos['state'] == 'invested' else 0
    est_proceeds = actual_held * bid
    pnl_txt = ""
    if entry_price > 0:
        est_pnl = (bid - entry_price) / entry_price * 100
        pnl_txt = f"\nEntry: ${entry_price:,.2f} | PnL est: {est_pnl:+.2f}%"

    send_telegram(
        f"🔴 <b>{asset} MAKER SELL</b>\n"
        f"Price: bid ${bid:,.2f} / ask ${ask:,.2f} ({spread_bps:.1f}bps)\n"
        f"Size: {actual_held:.6f} {asset} (~${est_proceeds:,.2f}){pnl_txt}\n"
        f"Executing..."
    )

    status, data = execute_maker_sell(symbol, actual_held)
    if status in (200, 201):
        order = data.get('data', data)
        mid = (bid + ask) / 2
        fill_price = float(order.get('average_fill_price', mid))

        pnl_pct = 0
        pnl_usd = 0
        if entry_price > 0:
            pnl_pct = (fill_price - entry_price) / entry_price * 100
            pnl_usd = pos.get('usd_invested', 0) * pnl_pct / 100

        pos['trades'].append({
            'action': 'SELL', 'price': fill_price,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'pnl_pct': round(pnl_pct, 2), 'pnl_usd': round(pnl_usd, 2),
            'auto': False, 'manual': True,
        })
        pos['state'] = 'cash'
        pos['entry_price'] = 0
        pos['entry_time'] = ''
        pos['base_amount'] = 0
        pos['usd_invested'] = 0
        save_position(asset, pos)

        time.sleep(2)
        pb = get_balances()
        usd_now = pb.get('USD', {}).get('available', 0) if pb else 0
        pnl_line = f"\nPnL: {pnl_pct:+.2f}% (${pnl_usd:+,.2f})" if entry_price > 0 else ""
        send_telegram(
            f"✅ <b>{asset} SELL filled</b>\n"
            f"Fill: ${fill_price:,.2f}{pnl_line}\n"
            f"USD: ${usd_now:,.2f}"
        )
    else:
        send_telegram(f"❌ {asset} SELL failed: {status} {data}")


def _handle_hold_shield_command(msg, trading_cfg):
    """Toggle Hold Shield per regime (bull/bear).
    /hold                 → show current state (both regimes)
    /hold on | off        → set both regimes
    /hold bull on|off     → set bull only
    /hold bear on|off     → set bear only
    /hold ETH ...         → target specific asset (else first enabled)
    /hold ETH bull on     → asset + regime + state
    """
    print(f"\n  [/hold] command received: {msg!r}", flush=True)
    try:
        parts = msg.split()[1:]  # drop '/hold'
        enabled = [a for a, c in trading_cfg.items() if c.get('enabled')]
        if not enabled:
            send_telegram("❌ No enabled asset")
            return

        # Parse asset (optional first token if it's a known asset)
        asset = None
        if parts and parts[0].upper() in trading_cfg:
            asset = parts.pop(0).upper()
        if asset is None:
            asset = enabled[0]
        if asset not in trading_cfg:
            send_telegram(f"❌ {asset} not configured")
            return

        # Parse regime + state
        regime = None
        state_tok = None
        for tok in parts:
            low = tok.lower()
            if low in ('bull', 'bear'):
                regime = low
            elif low in ('on', 'off'):
                state_tok = low

        cfg = trading_cfg[asset]
        for r in ('bull', 'bear'):
            if r not in cfg or not isinstance(cfg[r], dict):
                cfg[r] = {} if not isinstance(cfg.get(r), dict) else cfg[r]

        def _current_state():
            return (_shield_on_for_regime(cfg, 'bull'), _shield_on_for_regime(cfg, 'bear'))

        # No state specified → show current (tap the Bull/Bear buttons to toggle)
        if state_tok is None and regime is None and not parts:
            b_on, r_on = _current_state()
            min_pnl = cfg.get('min_sell_pnl_pct', 0)
            max_h = cfg.get('max_hold_hours', 0)
            send_telegram_with_buttons(
                f"🛡 <b>{asset} Hold Shield</b>\n"
                f"Bull: {'ON' if b_on else 'OFF'} | Bear: {'ON' if r_on else 'OFF'}\n"
                f"Threshold: PnL ≥ {min_pnl}% or held ≥ {max_h}h\n\n"
                f"Tap the Bull/Bear button below to toggle that regime.",
                _main_buttons(trading_cfg)
            )
            return

        # No state → toggle target(s)
        if state_tok is None:
            targets = [regime] if regime else ['bull', 'bear']
            for r in targets:
                cfg[r]['hold_shield'] = not _shield_on_for_regime(cfg, r)
        else:
            new_val = (state_tok == 'on')
            targets = [regime] if regime else ['bull', 'bear']
            for r in targets:
                cfg[r]['hold_shield'] = new_val

        # Remove stale asset-level hold_shield so regime values are authoritative
        if 'hold_shield' in cfg:
            del cfg['hold_shield']

        save_trading_config(trading_cfg)
        b_on, r_on = _current_state()
        min_pnl = cfg.get('min_sell_pnl_pct', 0)
        max_h = cfg.get('max_hold_hours', 0)
        send_telegram_with_buttons(
            f"🛡 <b>{asset} Hold Shield updated</b>\n"
            f"Bull: {'ON' if b_on else 'OFF'} | Bear: {'ON' if r_on else 'OFF'}\n"
            f"Thresholds: ≥{min_pnl}% or {max_h}h",
            _main_buttons(trading_cfg)
        )
    except Exception as e:
        import traceback
        print(f"  [/hold] ERROR: {e}", flush=True)
        traceback.print_exc()
        try:
            send_telegram(f"❌ /hold crashed: {type(e).__name__}: {e}")
        except Exception:
            pass


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
def _main_buttons(trading_cfg=None):
    """Build main keyboard. Shield + rally-cooldown gate each have two per-regime toggles.
    Tapping "🛡 Bull: ON" sends "/hold bull"; "🚦 Gate Bull: ON" sends "/gate <ASSET> bull off" etc."""
    shield_bull = shield_bear = gate_bull = gate_bear = False
    active_asset = None
    try:
        cfg = trading_cfg if trading_cfg is not None else load_trading_config()
        for a, c in cfg.items():
            if c.get('enabled'):
                has_threshold = c.get('min_sell_pnl_pct', 0) > 0
                shield_bull = _shield_on_for_regime(c, 'bull') and has_threshold
                shield_bear = _shield_on_for_regime(c, 'bear') and has_threshold
                gate_bull = _gate_on_for_regime(c, 'bull')
                gate_bear = _gate_on_for_regime(c, 'bear')
                active_asset = a
                break
    except Exception:
        pass
    shield_bull_label = f"🛡 Bull: {'ON' if shield_bull else 'OFF'}"
    shield_bear_label = f"🛡 Bear: {'ON' if shield_bear else 'OFF'}"
    gate_bull_label = f"🚦 Gate Bull: {'ON' if gate_bull else 'OFF'}"
    gate_bear_label = f"🚦 Gate Bear: {'ON' if gate_bear else 'OFF'}"
    # When we know the active asset, direct the button to it; else /gate shows status menu
    gate_bull_cmd = f"/gate {active_asset} bull {'off' if gate_bull else 'on'}" if active_asset else "/gate"
    gate_bear_cmd = f"/gate {active_asset} bear {'off' if gate_bear else 'on'}" if active_asset else "/gate"
    return [
        [('📊 Status', '/status'), ('📈 Charts', '/chart')],
        [('🔵 Buy', '/buy'), ('🔴 Sell', '/sell')],
        [(shield_bull_label, '/hold bull'), (shield_bear_label, '/hold bear')],
        [(gate_bull_label, gate_bull_cmd), (gate_bear_label, gate_bear_cmd)],
        [('⚙️ Setup', '/setup')],
    ]


# Legacy static fallback (kept for code paths that import it)
MAIN_BUTTONS = [
    [('📊 Status', '/status'), ('📈 Charts', '/chart')],
    [('🔵 Buy', '/buy'), ('🔴 Sell', '/sell')],
    [('🛡 Shield', '/hold'), ('⚙️ Setup', '/setup')],
]


def _parse_horizon_arg(token):
    """Parse horizon like '12h', '48h', '3d', '7d' → hours int. Returns None if invalid."""
    import re
    m = re.fullmatch(r'(\d+)([hd])', token.lower())
    if not m:
        return None
    n = int(m.group(1))
    hours = n if m.group(2) == 'h' else n * 24
    if hours < 6 or hours > 30 * 24:
        return None
    return hours


def _handle_chart_command(msg, trading_cfg):
    """Generate and send a chart with price + model predictions.
    /chart              → 48h, all enabled assets
    /chart BTC          → 48h, BTC only
    /chart 12h          → 12h, all enabled
    /chart BTC 7d       → 7d, BTC
    Horizon accepts Nh (6..720) or Nd (1..30). Args are order-flexible.
    """
    parts = msg.split()[1:]
    asset_arg = None
    hours = 48
    for p in parts:
        h = _parse_horizon_arg(p)
        if h is not None:
            hours = h
            continue
        a = p.upper()
        if a in trading_cfg:
            asset_arg = a
        else:
            send_telegram(f"Unknown arg: {p}. Use asset ({', '.join(trading_cfg.keys())}) or horizon (e.g. 12h, 7d).")
            return

    if asset_arg:
        assets_to_chart = [asset_arg]
    else:
        assets_to_chart = [a for a, c in trading_cfg.items() if c.get('enabled')]
        if not assets_to_chart:
            send_telegram("No enabled assets to chart.")
            return

    for asset in assets_to_chart:
        _generate_and_send_chart(asset, hours=hours)


def _generate_and_send_chart(asset, hours=48):
    """Generate and send a candlestick chart with signal overlays.
    hours: display window (6..720). One hourly candle per hour.
    """
    try:
        df_raw = load_data(asset)
        if df_raw is None:
            send_telegram(f"No data for {asset}")
            return
    except Exception as e:
        send_telegram(f"Error loading {asset} data: {e}")
        return

    df_price = df_raw.tail(hours).copy()
    if len(df_price) < 4:
        send_telegram(f"Not enough data for {asset}")
        return

    # Load signal log (slightly wider window to catch boundary signals)
    signals = []
    if os.path.exists(SIGNAL_LOG_FILE):
        try:
            import csv
            with open(SIGNAL_LOG_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours + 1)).strftime('%Y-%m-%d %H:%M:%S')
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
        from matplotlib.patches import Patch

        # Palette — candles use neutral up/down; signal markers use distinctive shapes.
        c_bg      = '#0e1117'
        c_card    = '#141922'
        c_up      = '#3b82f6'   # up candle — blue (colorblind-safe)
        c_down    = '#ef4444'   # down candle — red
        c_buy     = '#06b6d4'   # BUY marker — cyan (distinct from up candle blue)
        c_sell    = '#f97316'   # SELL marker — orange (distinct from down candle red)
        c_ok      = '#22c55e'   # ✓ correct
        c_bad     = '#94a3b8'   # ✗ wrong (muted, not alarming)
        c_pend    = '#eab308'   # ⏳ pending
        c_text    = '#e2e8f0'
        c_muted   = '#64748b'
        c_grid    = '#1e293b'
        c_wick    = '#475569'

        times = pd.to_datetime(df_price['datetime'])
        opens = df_price['open'].values
        highs = df_price['high'].values
        lows = df_price['low'].values
        closes = df_price['close'].values

        # Figure size scales slightly with horizon
        fig_w = 12 if hours <= 72 else 14 if hours <= 7 * 24 else 16
        fig, ax = plt.subplots(figsize=(fig_w, 6), facecolor=c_bg)
        ax.set_facecolor(c_card)

        # Candlesticks — bar width scaled to horizon density
        bar_minutes = max(8, min(40, int(2400 / hours)))
        bar_width = timedelta(minutes=bar_minutes)
        price_range = highs.max() - lows.min()
        doji_min = price_range * 0.001

        for i in range(len(times)):
            t = times.iloc[i]
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            color = c_up if c >= o else c_down
            ax.plot([t, t], [l, h], color=c_wick, linewidth=0.8, zorder=2)
            body_bottom = min(o, c)
            body_height = max(abs(c - o), doji_min)
            ax.bar(t, body_height, bottom=body_bottom, width=bar_width,
                   color=color, edgecolor=color, linewidth=0.5, zorder=3, alpha=0.85)

        # Signals: keep only transitions (action changed from previous non-HOLD)
        transition_signals = []
        prev_action = None
        for sig in signals:
            action = sig['action']
            if action == 'HOLD':
                continue
            if action != prev_action:
                transition_signals.append(sig)
            prev_action = action

        offset_y = price_range * 0.04
        badge_pad = price_range * 0.015
        has_ok = has_bad = has_pend = False

        for sig in transition_signals:
            try:
                sig_time = pd.to_datetime(sig['timestamp'])
                action = sig['action']
                time_diffs = abs(times - sig_time)
                nearest_idx = time_diffs.argmin()
                snap_time = times.iloc[nearest_idx]
                snap_price = closes[nearest_idx]

                # Correctness: within next 4 candles, did price move ≥0.3% in predicted direction?
                future = closes[nearest_idx + 1:nearest_idx + 5]
                if len(future) >= 2:
                    if action == 'BUY':
                        correct = future.max() >= snap_price * 1.003
                    else:
                        correct = future.min() <= snap_price * 0.997
                    badge = '✓' if correct else '✗'
                    badge_color = c_ok if correct else c_bad
                    if correct:
                        has_ok = True
                    else:
                        has_bad = True
                else:
                    badge = '⏳'
                    badge_color = c_pend
                    has_pend = True

                if action == 'BUY':
                    marker = '^'
                    marker_color = c_buy
                    y_marker = lows[nearest_idx] - offset_y
                    y_badge = y_marker - badge_pad
                    va = 'top'
                else:
                    marker = 'v'
                    marker_color = c_sell
                    y_marker = highs[nearest_idx] + offset_y
                    y_badge = y_marker + badge_pad
                    va = 'bottom'

                ax.scatter(snap_time, y_marker, color=marker_color, marker=marker,
                           s=180, zorder=6, edgecolors='white', linewidth=0.9)
                ax.annotate(badge, xy=(snap_time, y_badge),
                            ha='center', va=va, fontsize=11, fontweight='bold',
                            color=badge_color, zorder=7)
            except Exception:
                continue

        # Current price line
        last_price = closes[-1]
        ax.axhline(y=last_price, color=c_muted, linewidth=0.6, linestyle='--', alpha=0.5, zorder=1)
        ax.annotate(f'${last_price:,.2f}', xy=(times.iloc[-1], last_price),
                    fontsize=9, color=c_text, fontweight='bold',
                    xytext=(10, 0), textcoords='offset points', va='center')

        # Change over window
        first_price = closes[0]
        pct_change = (last_price / first_price - 1) * 100

        # Human label for horizon
        if hours % 24 == 0 and hours >= 24:
            h_label = f'{hours // 24}d'
        else:
            h_label = f'{hours}h'

        ax.set_title(f'{asset}/USD  ${last_price:,.2f}  ({pct_change:+.1f}%)',
                     fontsize=14, fontweight='bold', color=c_text, loc='left')
        ax.text(0.99, 1.02, h_label, transform=ax.transAxes, fontsize=10,
                color=c_muted, ha='right', va='bottom')

        # X axis formatting scales with horizon
        if hours <= 24:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, hours // 8)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif hours <= 72:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
        elif hours <= 7 * 24:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        else:
            step = max(1, (hours // 24) // 8)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=step))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))

        ax.tick_params(axis='x', colors=c_muted, labelsize=8)
        ax.tick_params(axis='y', colors=c_muted, labelsize=9)

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

        ax.set_ylim(lows.min() - price_range * 0.10, highs.max() + price_range * 0.10)

        # Legend INSIDE the chart — always visible, explains markers + badges
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor=c_buy,
                   markeredgecolor='white', markersize=10, label='BUY', linestyle='None'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor=c_sell,
                   markeredgecolor='white', markersize=10, label='SELL', linestyle='None'),
        ]
        if has_ok:
            legend_elements.append(Patch(facecolor='none', edgecolor='none', label='✓ correct (+0.3% in 4h)'))
        if has_bad:
            legend_elements.append(Patch(facecolor='none', edgecolor='none', label='✗ wrong'))
        if has_pend:
            legend_elements.append(Patch(facecolor='none', edgecolor='none', label='⏳ pending'))

        leg = ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
                        facecolor=c_card, edgecolor=c_grid, labelcolor=c_text,
                        framealpha=0.92, ncol=1, handletextpad=0.6)
        leg.set_zorder(10)

        chart_path = os.path.join('charts', f'{asset}_telegram_{hours}h.png')
        os.makedirs('charts', exist_ok=True)
        fig.savefig(chart_path, dpi=150, facecolor=c_bg)
        plt.close(fig)

        n_transitions = len(transition_signals)
        caption = f"{asset}/USD {h_label} | {n_transitions} signals | {pct_change:+.1f}%"
        send_telegram_photo(chart_path, caption=caption)
        print(f"  Chart sent for {asset} ({h_label})")

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
    send_telegram(f"📐 <b>{asset}</b>: Strategy is controlled by the regime detector. Use /setup to see current regime settings.")
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

# ---- Rally-cooldown gate (V7) command ----

def _handle_gate_command(msg, trading_cfg):
    """Toggle V7 rally-cooldown gate per asset (and optionally per regime), or clear an active window.
    Forms:
      /gate                              -> status + buttons
      /gate on | off                     -> toggle on all enabled assets (all regimes)
      /gate ETH on | off                 -> per-asset (all regimes)
      /gate ETH bull on | off            -> per-asset, bull regime only
      /gate ETH bear on | off            -> per-asset, bear regime only
      /gate ETH clear                    -> wipe active cooldown timer only
    """
    parts = msg.split()
    enabled = [a for a, c in trading_cfg.items() if c.get('enabled')]

    def _show_status():
        lines = ["🚦 <b>Rally-Cooldown Gate (V7)</b>", ""]
        rows = []
        for a in enabled:
            cfg_a = trading_cfg[a]
            bull_on = _gate_on_for_regime(cfg_a, 'bull')
            bear_on = _gate_on_for_regime(cfg_a, 'bear')
            pos = load_position(a)
            until_str = pos.get('rally_cooldown_until', '')
            cd_left = ''
            cd_active = False
            if until_str:
                try:
                    until = datetime.fromisoformat(until_str.replace('Z', '+00:00'))
                    now = datetime.now(timezone.utc)
                    if until > now:
                        cd_active = True
                        cd_left = f" | cd {(until - now).total_seconds()/3600:.1f}h left"
                except Exception:
                    pass
            lines.append(f"<b>{a}</b>: bull {'🔵 ON' if bull_on else '🔴 OFF'} / bear {'🔵 ON' if bear_on else '🔴 OFF'}{cd_left}")
            bull_rc = (cfg_a.get('bull') or {}).get('rally_cooldown') or cfg_a.get('rally_cooldown') or {}
            bear_rc = (cfg_a.get('bear') or {}).get('rally_cooldown') or cfg_a.get('rally_cooldown') or {}
            if bull_rc:
                lines.append(f"   bull: rr{bull_rc.get('h_short')}h≥{bull_rc.get('t_short_pct')}% OR rr{bull_rc.get('h_long')}h≥{bull_rc.get('t_long_pct')}% → {bull_rc.get('cd_hours')}h")
            if bear_rc:
                lines.append(f"   bear: rr{bear_rc.get('h_short')}h≥{bear_rc.get('t_short_pct')}% OR rr{bear_rc.get('h_long')}h≥{bear_rc.get('t_long_pct')}% → {bear_rc.get('cd_hours')}h")
            row = [
                (f"{a} bull {'OFF' if bull_on else 'ON'}", f"/gate {a} bull {'off' if bull_on else 'on'}"),
                (f"{a} bear {'OFF' if bear_on else 'ON'}", f"/gate {a} bear {'off' if bear_on else 'on'}"),
            ]
            if cd_active:
                row.append((f"Clear cd", f"/gate {a} clear"))
            rows.append(row)
        send_telegram_with_buttons("\n".join(lines), rows)

    if len(parts) == 1:
        _show_status(); return

    # /gate on | off  → all enabled assets, all regimes
    if len(parts) == 2 and parts[1].lower() in ('on', 'off'):
        on = parts[1].lower() == 'on'
        for a in enabled:
            _set_gate_enabled(trading_cfg[a], None, on)
        save_trading_config(trading_cfg)
        send_telegram(f"🚦 Gate {'ON' if on else 'OFF'} for: {', '.join(enabled)}")
        _show_status(); return

    # /gate ASSET [regime] on|off|clear
    if len(parts) >= 3:
        asset = parts[1].upper()
        if asset not in trading_cfg:
            send_telegram(f"❌ {asset} not configured"); return

        # Form: /gate ASSET bull|bear on|off
        if len(parts) == 4 and parts[2].lower() in ('bull', 'bear') and parts[3].lower() in ('on', 'off'):
            regime = parts[2].lower()
            on = parts[3].lower() == 'on'
            _set_gate_enabled(trading_cfg[asset], regime, on)
            save_trading_config(trading_cfg)
            send_telegram(f"🚦 {asset} {regime} gate {'ON' if on else 'OFF'}")
            _show_status(); return

        # Form: /gate ASSET on|off|clear
        action = parts[2].lower()
        if action in ('on', 'off'):
            _set_gate_enabled(trading_cfg[asset], None, action == 'on')
            save_trading_config(trading_cfg)
            send_telegram(f"🚦 {asset} gate {action.upper()} (both regimes)")
            _show_status(); return
        if action == 'clear':
            pos = load_position(asset)
            if pos.get('rally_cooldown_until'):
                pos['rally_cooldown_until'] = ''
                save_position(asset, pos)
                send_telegram(f"🧹 {asset} cooldown window cleared (one-time override)")
            else:
                send_telegram(f"ℹ️ {asset} has no active cooldown")
            _show_status(); return

    send_telegram("Usage: /gate | /gate on|off | /gate ETH on|off|clear | /gate ETH bull|bear on|off")


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
                elif cmd == '/buy' or cmd.startswith('/buy '):
                    _handle_manual_buy_command(msg, trading_cfg)
                elif cmd == '/sell' or cmd.startswith('/sell '):
                    _handle_manual_sell_command(msg, trading_cfg)
                elif cmd == '/hold' or cmd.startswith('/hold '):
                    _handle_hold_shield_command(msg, trading_cfg)
                elif cmd.startswith('/chart'):
                    _handle_chart_command(msg, trading_cfg)
                elif cmd == '/setup' or cmd.startswith('/cfg_'):
                    if not _setup_state.get('active'):
                        _setup_start(trading_cfg)
                    if cmd.startswith('/cfg_'):
                        _setup_handle(msg, trading_cfg)
                elif cmd == '/gate' or cmd.startswith('/gate '):
                    _handle_gate_command(msg, trading_cfg)
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
# Fix #6 (2026-04-24): track whether startup preflight has passed.
# Hot-reload preflight only runs AFTER startup is complete so we don't
# double-run at initialization. Set by main() after startup preflight.
_TRADER_PREFLIGHT_PASSED = False


def _validate_config_or_revert(trading_cfg, snapshot):
    """Fix #6 (2026-04-24): re-run preflight after hot-reload. On failure,
    revert trading_cfg in-place to the pre-reload snapshot + Telegram alert.
    Returns True if config passed, False if reverted."""
    if not _TRADER_PREFLIGHT_PASSED:
        # Still in startup; let main()'s startup-preflight handle validation.
        return True

    # Regenerate manifest first (the active prod CSV may have changed too).
    try:
        import subprocess as _sp
        _sp.check_call(
            [sys.executable, os.path.join(os.path.dirname(__file__), 'tools', 'generate_feature_manifest.py')],
            stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
        )
    except Exception as _e:
        print(f"  [preflight-hotreload] manifest regen failed: {_e}")

    # Run preflight
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
        from trader_preflight import preflight as _preflight
        report = _preflight(verbose=False)
    except Exception as _e:
        print(f"  [preflight-hotreload] could not run ({_e}) — accepting hot-reload defensively.")
        return True

    if report.get('ok'):
        print(f"  [preflight-hotreload] ✓ PASSED ({len(report.get('checks', []))} checks)")
        return True

    # FAIL — revert in-place so trading_cfg resumes pre-reload state.
    failures = report.get('failures', [])
    print(f"  [!!] [preflight-hotreload] FAILED — reverting config to pre-reload snapshot")
    for msg in failures[:5]:
        print(f"       ✗ {msg}")

    # Revert: clear trading_cfg then repopulate from snapshot (preserve object identity
    # because many callers hold a reference to the dict).
    trading_cfg.clear()
    for k, v in snapshot.items():
        trading_cfg[k] = v

    # Telegram alert
    failures_short = '\n'.join(f'  • {m}' for m in failures[:3])
    _rate_limited_telegram(
        'hotreload_preflight_failed',
        f"🚨 Hot-reload preflight FAILED — reverted to previous config.\n{failures_short}\n\n"
        f"Fix regime_config_ed.json / crypto_ed_production.csv then re-save to trigger retry.",
    )
    return False


def _reload_trading_config(trading_cfg):
    """Hot-reload regime_config_ed.json into existing dict. Returns True if anything changed.
    Fix #6 (2026-04-24): after applying changes, re-run preflight. On failure, revert
    in-place to the pre-reload snapshot and send Telegram alert."""
    import copy as _copy
    # Snapshot BEFORE merge so we can revert if preflight fails
    snapshot = _copy.deepcopy(trading_cfg)

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

    if changed:
        # Validate — revert on failure. No-op if trader hasn't finished startup.
        _validate_config_or_revert(trading_cfg, snapshot)
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
    import time as _tm_cycle
    _cycle_t0 = _tm_cycle.time()
    cycle_metrics = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'rows': [],                      # per-asset-per-horizon rows populated by process_asset
        'p1_duration_sec': None,
        'p2_duration_sec': None,
    }

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

    # Two-priority data-dependency downloads (2026-04-23):
    #   P1 (BLOCKING, runs before signal) — sources every enabled (asset, horizon)
    #     model needs right now. Derived from feature_manifest.json +
    #     feature_sources.json so it auto-adapts when config changes.
    #   P2 (BACKGROUND, runs after Telegram is sent) — all remaining defensive
    #     sources, so they're fresh when you enable a new asset/horizon.
    # Per-source freshness thresholds prevent wasted downloads; a source is
    # only actually pulled if it's stale.
    from crypto_live_trader_ed import download_asset as _dl_asset
    import time as _time
    import json as _json

    _DATA_DEP_ASSETS = ('BTC', 'ETH', 'XRP', 'SOL', 'LINK')

    def _file_is_fresh(relpath, max_age_sec):
        """Fix #7 (2026-04-24): content-aware freshness using last-row datetime.
        Looks for a 'datetime' / 'date' / 'timestamp' column and checks its
        most recent value against max_age. Falls back to mtime on any failure.
        Catches partial downloads that bumped mtime but didn't advance data."""
        fp = os.path.join(os.path.dirname(__file__), relpath)
        if not os.path.exists(fp):
            return False
        try:
            import pandas as _pd
            df = _pd.read_csv(fp, usecols=lambda c: c in ('datetime', 'date', 'timestamp'))
            if len(df) == 0:
                return False
            if 'datetime' in df.columns:
                last = _pd.to_datetime(df['datetime'].iloc[-1])
            elif 'date' in df.columns:
                last = _pd.to_datetime(df['date'].iloc[-1])
            elif 'timestamp' in df.columns:
                last = _pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')
            else:
                # Snapshot ring files with no time column — use mtime
                return (_time.time() - os.path.getmtime(fp)) < max_age_sec
            if last.tzinfo is None:
                last = last.tz_localize('UTC')
            age_sec = (_pd.Timestamp.now(tz='UTC') - last).total_seconds()
            return age_sec < max_age_sec
        except Exception:
            return (_time.time() - os.path.getmtime(fp)) < max_age_sec

    # --- Source registry: every downloadable source, with freshness + download fn ---
    try:
        from download_macro_data import (
            download_yfinance_data, download_fear_greed, download_cross_asset,
            download_derivatives_data, download_onchain_data, download_stablecoin_flows,
        )
        _macro_ok = True
    except ImportError as _e:
        print(f"  [!] data-dep macro module import failed: {_e}")
        _macro_ok = False

    _SOURCE_REGISTRY = {
        # OHLCV (hourly). update_only=True = 1 cheap API check; nearly free.
        **{f'ohlcv_{a.lower()}': {
               'file': f'data/{a.lower()}_hourly_data.csv',
               'max_age_sec': 2 * 3600,
               'fn': (lambda _a=a: _dl_asset(_a, update_only=True)),
           } for a in _DATA_DEP_ASSETS},
    }
    if _macro_ok:
        # Derivatives per-asset (Binance perp hourly)
        for _a in _DATA_DEP_ASSETS:
            _SOURCE_REGISTRY[f'derivatives_{_a.lower()}'] = {
                'file': f'data/macro_data/derivatives_{_a.lower()}.csv',
                'max_age_sec': 2 * 3600,
                'fn': (lambda _aa=_a: download_derivatives_data(assets=[_aa])),
            }
        _SOURCE_REGISTRY.update({
            'macro_daily':      {'file': 'data/macro_data/macro_daily.csv',      'max_age_sec':  6 * 3600, 'fn': download_yfinance_data},
            'fear_greed':       {'file': 'data/macro_data/fear_greed.csv',       'max_age_sec':  6 * 3600, 'fn': download_fear_greed},
            'cross_asset':      {'file': 'data/macro_data/cross_asset.csv',      'max_age_sec':  6 * 3600, 'fn': download_cross_asset},
            'stablecoin_flows': {'file': 'data/macro_data/stablecoin_flows.csv', 'max_age_sec': 12 * 3600, 'fn': download_stablecoin_flows},
        })
        # On-chain per-asset (CoinMetrics daily; SOL skipped — 403 on free tier)
        for _oc in ('btc', 'eth', 'xrp', 'link'):
            _SOURCE_REGISTRY[f'onchain_{_oc}'] = {
                'file': f'data/macro_data/onchain_{_oc}.csv',
                'max_age_sec': 6 * 3600,
                'fn': (lambda _o=_oc: download_onchain_data(asset=_o)),
            }

    # --- Map each prod-model feature name to its source key(s) ---
    def _feature_to_source_keys(feat_name, primary_asset):
        a = primary_asset.lower()
        if feat_name.startswith('pysr_'):
            return set()  # PySR itself = json file, handled separately; its inputs resolve via recursion in the manifest
        if feat_name.startswith('xa_btc_lag'):   return {'ohlcv_btc'}
        if feat_name.startswith('xa_eth_lag'):   return {'ohlcv_eth'}
        if feat_name.startswith('xa_btc_usd') or feat_name.startswith('xa_eth_usd'):
            return {'ohlcv_btc', 'ohlcv_eth'}
        if any(feat_name.startswith(p) for p in ('xa_sp500', 'xa_nasdaq', 'xa_dax', 'xa_cac', 'xa_smi')):
            return {'macro_daily', 'cross_asset'}
        if any(feat_name.startswith(p) for p in ('m_vix', 'm_dxy', 'm_sp500', 'm_nasdaq', 'm_gold', 'm_oil', 'm_eurusd', 'm_us10y', 'm_usdjpy')):
            return {'macro_daily'}
        if feat_name.startswith('fg_'):          return {'fear_greed'}
        if feat_name.startswith('deriv_'):       return {f'derivatives_{a}'}
        if feat_name.startswith('oc_'):          return {f'onchain_{a}'}
        if feat_name.startswith('stable_mcap_'): return {'stablecoin_flows'}
        # Technical (logret, volatility, sma, adx, rsi, spread, hour_*, etc.)
        return {f'ohlcv_{a}'}

    # --- Compute P1 set: sources every enabled (asset, horizon) needs ---
    def _compute_p1_sources():
        p1 = set()
        manifest_path = os.path.join(os.path.dirname(__file__), 'config', 'feature_manifest.json')
        if not os.path.exists(manifest_path):
            # No manifest -> be conservative: P1 = enabled asset OHLCV + their BTC lead-lag + their derivatives
            for a, c in trading_cfg.items():
                if not c.get('enabled'):
                    continue
                p1.add(f'ohlcv_{a.lower()}')
                p1.add('ohlcv_btc')  # cross-asset lead-lag
                p1.add(f'derivatives_{a.lower()}')
                p1.add('macro_daily')
                p1.add('fear_greed')
            return p1
        try:
            manifest = _json.load(open(manifest_path))
        except Exception:
            return p1
        for a, c in trading_cfg.items():
            if not c.get('enabled'):
                continue
            p1.add(f'ohlcv_{a.lower()}')  # primary always required
            for h_key in ('bull', 'bear'):
                h = c.get(h_key, {}).get('horizon')
                if h is None:
                    continue
                info = manifest.get('assets', {}).get(a, {}).get(str(h))
                if not info:
                    continue
                for feat in info.get('union', []):
                    p1 |= _feature_to_source_keys(feat, a)
        return {k for k in p1 if k in _SOURCE_REGISTRY}

    def _refresh_sources(keys, label):
        """Iterate source keys; for each, skip if fresh else download. Never raises.
        Fix #1 (2026-04-24): failures now emit rate-limited Telegram alerts
        (once/hour per source+label) on top of the console log. Prevents silent
        download outages from persisting unnoticed."""
        stale = [k for k in keys if not _file_is_fresh(_SOURCE_REGISTRY[k]['file'], _SOURCE_REGISTRY[k]['max_age_sec'])]
        if not stale:
            print(f"  [{label}] all fresh ({len(keys)} sources) — skip")
            return
        print(f"  [{label}] refreshing {len(stale)}/{len(keys)} stale sources: {stale}")
        for k in stale:
            try:
                _SOURCE_REGISTRY[k]['fn']()
            except Exception as _e:
                print(f"  [{label}] [!] {k}: {_e}")
                _rate_limited_telegram(
                    f'datadep_{label}_{k}',
                    f"🚨 [{label}] {k} download failed: {_e}",
                )

    # --- PHASE 1: P1 required sources (BLOCKING before signal) ---
    p1_keys = _compute_p1_sources()
    p2_keys = set(_SOURCE_REGISTRY.keys()) - p1_keys
    print(f"\n  Data-dep priority split: P1={sorted(p1_keys)} | P2={sorted(p2_keys)}")
    _p1_t0 = _tm_cycle.time()
    _refresh_sources(p1_keys, 'P1 REQUIRED')
    cycle_metrics['p1_duration_sec'] = round(_tm_cycle.time() - _p1_t0, 2)

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
        # Fix #9: thread cycle_metrics dict through; process_asset appends per-horizon rows
        r = process_asset(asset, cfg, dry_run=dry_run, cycle_metrics=cycle_metrics)
        results.append(r)

    # Refresh balances AFTER trades so Telegram shows current exchange state
    if not dry_run:
        balances = get_balances()

    # Send combined Telegram with inline buttons
    valid = [r for r in results if r is not None]
    if valid:
        msg = format_multi_asset_telegram(valid, dry_run=dry_run, balances=balances, trading_cfg=trading_cfg)
        send_telegram_with_buttons(msg, _main_buttons(trading_cfg))

    # --- PHASE 2: P2 deferred sources (AFTER signal is sent to Telegram) ---
    # Keeps the defensive data (XRP/SOL/LINK OHLCV, their derivatives,
    # on-chain for non-primary assets, etc.) fresh for the next config flip
    # without delaying the current cycle's signal emission.
    _p2_t0 = _tm_cycle.time()
    _refresh_sources(p2_keys, 'P2 DEFERRED')
    cycle_metrics['p2_duration_sec'] = round(_tm_cycle.time() - _p2_t0, 2)

    # Accumulate hourly snapshots for future features (silent, non-blocking)
    try:
        from download_macro_data import download_orderbook_snapshot, download_options_iv_skew
        enabled_assets = [a for a, c in trading_cfg.items() if c.get('enabled')]
        download_orderbook_snapshot(assets=enabled_assets)
        download_options_iv_skew()
    except Exception:
        pass  # never block trading on snapshot failure

    # Fix #9: flush cycle metrics to CSV
    _total_sec = round(_tm_cycle.time() - _cycle_t0, 2)
    for row in cycle_metrics.get('rows', []):
        row['p1_duration_sec'] = cycle_metrics['p1_duration_sec']
        row['p2_duration_sec'] = cycle_metrics['p2_duration_sec']
        row['total_cycle_sec'] = _total_sec
        _append_cycle_metrics(row)

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
        f"📱 /help for all commands | /status for current state"
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
    print(f"  Telegram: /help /status /chart /setup /sync /pause /resume /stop (+ buy/sell/hold via buttons)")
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
    # Fix #5D (2026-04-24): install stdout tee BEFORE any other print so every
    # print from startup onwards is captured to a log file. Works with or
    # without external tee_launcher.bat.
    try:
        from crypto_live_trader_ed import init_runtime_log
        init_runtime_log()
    except Exception as _e:
        print(f"[!] runtime log init failed: {_e}")

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
                    print(f"    Entry: ${pos['entry_price']:,.2f} at {_format_entry_time_local(pos['entry_time'])}")
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

    # Pre-flight integrity check: validate that every required feature for
    # every enabled (asset, horizon) is present in the current build AND all
    # upstream data sources meet their freshness SLAs. Refuse to start if not.
    # Allow --skip-preflight for emergency overrides; warn in Telegram.
    skip_preflight = '--skip-preflight' in sys.argv
    if not skip_preflight and not dry_run:
        # Refresh the manifest first (cheap: just rebuilds from prod CSV)
        try:
            import subprocess as _sp
            _sp.check_call(
                [sys.executable, os.path.join(os.path.dirname(__file__), 'tools', 'generate_feature_manifest.py')],
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
            )
        except Exception as _e:
            print(f"  [preflight] manifest regen skipped: {_e}")
        # Run preflight
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
            from trader_preflight import preflight as _preflight
            report = _preflight(verbose=True)
            if not report['ok']:
                print()
                print("=" * 60)
                print("  TRADER STARTUP BLOCKED — pre-flight failed")
                print("=" * 60)
                print("  Fix the source freshness issues above, OR run with")
                print("  --skip-preflight to bypass (NOT recommended).")
                try:
                    failures = '\n'.join(f'  • {m}' for m in report['failures'][:5])
                    send_telegram(f"🚨 Trader startup BLOCKED — pre-flight failed:\n{failures}")
                except Exception:
                    pass
                sys.exit(2)
            # Fix #6: mark startup complete so hot-reloads will re-run preflight.
            global _TRADER_PREFLIGHT_PASSED
            _TRADER_PREFLIGHT_PASSED = True
        except SystemExit:
            raise
        except Exception as _e:
            print(f"  [preflight] could not run (proceeding cautiously): {_e}")
    elif skip_preflight:
        print("  [preflight] --skip-preflight set: SKIPPING integrity check")
        # Even when skipped, flag startup complete so hot-reloads validate.
        _TRADER_PREFLIGHT_PASSED = True

    if loop_mode:
        run_loop(trading_cfg, dry_run=dry_run)
    else:
        run_all_once(trading_cfg, dry_run=dry_run)


if __name__ == '__main__':
    main()
