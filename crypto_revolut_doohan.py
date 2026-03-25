"""
Revolut X Multi-Asset Auto-Trader (Doohan)
==========================================
Same as crypto_revolut_trader.py but uses Doohan models (Optuna + XGBoost).
Reads from crypto_doohan_best_models.csv and trading_config_doohan.json.

Usage:
  python crypto_revolut_doohan.py                  # Interactive
  python crypto_revolut_doohan.py --loop           # Auto-loop all assets
  python crypto_revolut_doohan.py --dry-run --loop # Signals only
  python crypto_revolut_doohan.py --status         # All positions
  python crypto_revolut_doohan.py --balance        # Revolut X balance
  python crypto_revolut_doohan.py --setup-telegram
  python crypto_revolut_doohan.py --reset          # Reset all positions
"""

import os
import sys
import time
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

from crypto_live_trader_doohan import (
    load_best_config, generate_live_signal, compute_combined_signal,
    send_telegram, wait_for_fresh_candle, setup_telegram,
    download_asset, load_data, TELEGRAM_CONFIG, MIN_CONFIDENCE,
    MODELS_CSV, HORIZON_SHORT, HORIZON_LONG, AVAILABLE_HORIZONS,
)


# ============================================================
# TRADING CONFIG (per-asset strategies + max positions)
# ============================================================
TRADING_CONFIG_FILE = 'config/trading_config_doohan.json'

DEFAULT_TRADING_CONFIG = {
    'BTC': {
        'strategy': 'both_agree',   # BUY when both horizons agree
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
    defaults = json.loads(json.dumps(DEFAULT_TRADING_CONFIG))
    if os.path.exists(TRADING_CONFIG_FILE):
        with open(TRADING_CONFIG_FILE) as f:
            file_cfg = json.load(f)
        # Merge file config over defaults (Mode F only writes strategy + min_confidence)
        for asset in file_cfg:
            if asset not in defaults:
                defaults[asset] = {'strategy': 'both_agree', 'max_position_usd': 0, 'symbol': f'{asset}-USD', 'enabled': True}
            defaults[asset].update(file_cfg[asset])
    return defaults

def save_trading_config(cfg):
    os.makedirs('config', exist_ok=True)
    with open(TRADING_CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)

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
        with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as resp:
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
# PROCESS ONE ASSET
# ============================================================
def process_asset(asset, trading_cfg, dry_run=False):
    """Generate signals and execute for one asset. Returns result dict."""
    position = load_position(asset)
    strategy = trading_cfg.get('strategy', 'both_agree')
    symbol = trading_cfg.get('symbol', f'{asset}-USD')
    max_usd = trading_cfg.get('max_position_usd', 0)

    # Load configs — use configured horizon if set, otherwise all available
    sigs_by_horizon = {}
    any_config = False
    configured_horizon = trading_cfg.get('horizon')
    horizons_to_load = [configured_horizon] if configured_horizon else list(AVAILABLE_HORIZONS)
    for h in horizons_to_load:
        cfg = load_best_config(asset, horizon=h)
        if cfg:
            any_config = True
            sigs_by_horizon[h] = cfg  # store config temporarily

    if not any_config:
        return None

    # Download data once
    try:
        download_asset(asset, update_only=True)
    except Exception:
        pass

    df_raw = load_data(asset)
    if df_raw is None:
        return None

    # Generate live signals for each available horizon
    first = True
    for h in list(sigs_by_horizon.keys()):
        cfg = sigs_by_horizon[h]
        sig = generate_live_signal(asset, cfg, df_raw=df_raw, verbose=first)
        sigs_by_horizon[h] = sig  # replace config with actual signal
        first = False

    # Apply asset-specific strategy — single configured horizon uses Xh_only
    min_conf = trading_cfg.get('min_confidence', MIN_CONFIDENCE)
    if configured_horizon and len(sigs_by_horizon) == 1:
        strategy = f'{configured_horizon}h_only'
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
    print(f"  {asset}: {' | '.join(sig_strs)} → {action} ({confidence:.0f}%) [{reason}] | pos={position['state']}")

    # Log signal for /chart command
    _log_signal(asset, price, sigs_by_horizon, action, confidence)

    # Expose sig_short, sig_long for Telegram message
    sig_short = sigs_by_horizon.get(HORIZON_SHORT)
    sig_long = sigs_by_horizon.get(HORIZON_LONG)

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

    # Get gamma from the primary horizon model
    gamma_val = ''
    gamma_horizons = [configured_horizon] if configured_horizon else [HORIZON_LONG, HORIZON_SHORT]
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
        'min_confidence': trading_cfg.get('min_confidence', MIN_CONFIDENCE),
        'gamma': gamma_val,
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
        lines.append(f"{auto_icon} <b>{asset}</b> {price_str} | RSI:{rsi:.0f} | [{strategy}{gamma_suffix}]{conf_str}")

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

STRATEGIES = ['both_agree', 'either_agree', f'{HORIZON_SHORT}h_only', f'{HORIZON_LONG}h_only']

# ---- Simple command handlers ----

def _handle_config_command():
    """Show current trading config for all assets with setup button."""
    trading_cfg = load_trading_config()
    lines = ["⚙️ <b>Config</b>\n"]
    for asset, cfg in trading_cfg.items():
        enabled = "🔵" if cfg.get('enabled') else "🔴"
        pos = load_position(asset)
        auto = "AUTO" if pos.get('auto_trade') else "MANUAL"
        lines.append(
            f"{enabled} <b>{asset}</b> | {cfg.get('strategy', '?')} | "
            f"{cfg.get('min_confidence', MIN_CONFIDENCE)}% | "
            f"${cfg.get('max_position_usd', 0):,.0f} | {auto}"
        )
    buttons = [[('⚙️ Edit Config', '/setup')]]
    send_telegram_with_buttons("\n".join(lines), buttons)

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

        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'crypto_trading_system_doohan.py')

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

        cmd = [sys.executable, script, 'DG', assets, h_arg]

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
        print(f"  OPTIMIZE: launched Mode DG for {assets} {h_arg} (PID {_optimize_proc.pid})")
        send_telegram(f"🚀 <b>Mode DG started</b>\n{assets} {h_arg} (PID {_optimize_proc.pid})\nTrader stays live. Models hot-reload when done.")

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
    strategy = cfg.get('strategy', '?')
    confidence = cfg.get('min_confidence', MIN_CONFIDENCE)
    max_pos = cfg.get('max_position_usd', 0)

    text = (
        f"⚙️ <b>{asset} Settings</b>\n\n"
        f"Enabled: {'🔵 ON' if enabled else '🔴 OFF'}\n"
        f"Strategy: {strategy}\n"
        f"Confidence: {confidence}%\n"
        f"Max position: ${max_pos:,.0f}\n"
        f"Auto-trade: {'🔵 ON' if auto else '🔴 OFF'}"
    )
    buttons = [
        [('🔀 Toggle ON/OFF', f'/cfg_{asset}_toggle'),
         ('🔀 Auto-trade', f'/cfg_{asset}_auto')],
        [('📐 Strategy', f'/cfg_{asset}_strategy')],
        [('📊 Confidence', f'/cfg_{asset}_conf'),
         ('💰 Max Position', f'/cfg_{asset}_max')],
        [('⬅️ Back', '/cfg_back')],
    ]
    send_telegram_with_buttons(text, buttons)

def _setup_send_strategy_picker(asset, cfg):
    """Show strategy picker with inline buttons."""
    current = cfg.get('strategy', '')
    buttons = []
    for s in STRATEGIES:
        label = f"{'✅ ' if s == current else ''}{s}"
        buttons.append([(label, f'/cfg_{asset}_strat_{s}')])
    buttons.append([('⬅️ Back', f'/cfg_{asset}')])
    send_telegram_with_buttons(f"📐 Strategy for <b>{asset}</b>:", buttons)

def _setup_send_confidence_picker(asset, cfg):
    """Show confidence picker with inline buttons."""
    current = cfg.get('min_confidence', MIN_CONFIDENCE)
    values = [60, 65, 70, 75, 80, 85, 90]
    buttons = []
    row = []
    for v in values:
        label = f"{'✅ ' if v == current else ''}{v}%"
        row.append((label, f'/cfg_{asset}_confv_{v}'))
        if len(row) == 4:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([('⬅️ Back', f'/cfg_{asset}')])
    send_telegram_with_buttons(f"📊 Min confidence for <b>{asset}</b>:", buttons)

def _setup_send_max_picker(asset, cfg):
    """Show max position picker with inline buttons."""
    current = cfg.get('max_position_usd', 0)
    values = [0, 1000, 5000, 10000]
    buttons = []
    row = []
    for v in values:
        label = f"{'✅ ' if v == current else ''}${v:,}"
        row.append((label, f'/cfg_{asset}_maxv_{v}'))
    buttons.append(row)
    # Custom input button + current value display
    custom_label = f"✏️ Custom (now ${current:,})" if current not in values else "✏️ Custom"
    buttons.append([(custom_label, f'/cfg_{asset}_maxcustom')])
    buttons.append([('⬅️ Back', f'/cfg_{asset}')])
    send_telegram_with_buttons(f"💰 Max position for <b>{asset}</b>:", buttons)

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
    if awaiting_asset and awaiting_asset in cfg:
        try:
            val = int(text_l.strip().replace('$', '').replace(',', ''))
            if 0 <= val <= 100000:
                cfg[awaiting_asset]['max_position_usd'] = val
                _setup_state.pop('awaiting_custom_max', None)
                send_telegram(f"✅ {awaiting_asset} max position → ${val:,}")
                _setup_send_menu(awaiting_asset, cfg[awaiting_asset])
                return
        except ValueError:
            # Not a number — clear awaiting state and continue normal routing
            _setup_state.pop('awaiting_custom_max', None)

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

        # /cfg_{ASSET}_strategy — show strategy picker
        if text_l == f'/cfg_{asset}_strategy':
            _setup_send_strategy_picker(asset, cfg[asset])
            return

        # /cfg_{ASSET}_strat_{strategy} — set strategy
        for s in STRATEGIES:
            if text_l == f'/cfg_{asset}_strat_{s}':
                cfg[asset]['strategy'] = s
                send_telegram(f"✅ {asset} strategy → {s}")
                _setup_send_menu(asset, cfg[asset])
                return

        # /cfg_{ASSET}_conf — show confidence picker
        if text_l == f'/cfg_{asset}_conf':
            _setup_send_confidence_picker(asset, cfg[asset])
            return

        # /cfg_{ASSET}_confv_{value} — set confidence
        if text_l.startswith(f'/cfg_{asset}_confv_'):
            try:
                val = int(text_l.split('_')[-1])
                cfg[asset]['min_confidence'] = val
                send_telegram(f"✅ {asset} confidence → {val}%")
                _setup_send_menu(asset, cfg[asset])
            except ValueError:
                pass
            return

        # /cfg_{ASSET}_max — show max position picker
        if text_l == f'/cfg_{asset}_max':
            _setup_send_max_picker(asset, cfg[asset])
            return

        # /cfg_{ASSET}_maxv_{value} — set max position
        if text_l.startswith(f'/cfg_{asset}_maxv_'):
            try:
                val = int(text_l.split('_')[-1])
                cfg[asset]['max_position_usd'] = val
                send_telegram(f"✅ {asset} max position → ${val:,}")
                _setup_send_menu(asset, cfg[asset])
            except ValueError:
                pass
            return

        # /cfg_{ASSET}_maxcustom — prompt for custom amount
        if text_l == f'/cfg_{asset}_maxcustom':
            _setup_state['awaiting_custom_max'] = asset
            send_telegram(f"💰 Type the max position amount in USD for <b>{asset}</b> (e.g. 2500):")
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
    """Hot-reload trading_config.json into existing dict. Returns True if anything changed."""
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
        print(f"\n  --- {asset} ({cfg.get('strategy', '?')}) ---")
        r = process_asset(asset, cfg, dry_run=dry_run)
        results.append(r)

    # Refresh balances AFTER trades so Telegram shows current exchange state
    if not dry_run:
        balances = get_balances()

    # Send combined Telegram with inline buttons
    valid = [r for r in results if r is not None]
    if valid:
        msg = format_multi_asset_telegram(valid, dry_run=dry_run, balances=balances)
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
        asset_lines.append(f"  {a}: {c.get('strategy','?')} | ${c.get('max_position_usd',0):,.0f} | {icon}")

    conf_parts_str = ', '.join(
        f"{a}={c.get('min_confidence', MIN_CONFIDENCE)}%"
        for a, c in trading_cfg.items() if c.get('enabled')
    )

    send_telegram(
        f"🚀 <b>Multi-Asset Trader Started</b>\n\n"
        + "\n".join(asset_lines) + "\n\n"
        f"BUY needs >= {conf_parts_str}\n"
        f"SELL when either model says SELL\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"📱 /help for all commands"
    )


def run_loop(trading_cfg, dry_run=False):
    mode = "DRY RUN" if dry_run else "LIVE"
    assets_str = ", ".join(a for a, c in trading_cfg.items() if c.get('enabled'))

    print(f"\n{'='*60}")
    print(f"  REVOLUT X MULTI-ASSET TRADER [DOOHAN {mode}]")
    print(f"  Assets: {assets_str}")
    for asset, cfg in trading_cfg.items():
        if cfg.get('enabled'):
            pos = load_position(asset)
            auto = "AUTO" if pos.get('auto_trade') else "MANUAL"
            print(f"  {asset}: {cfg.get('strategy','?')} | max=${cfg.get('max_position_usd',0):,.0f} | {auto} | {pos['state'].upper()}")
    conf_parts = [f"{a}={c.get('min_confidence', MIN_CONFIDENCE)}%" for a, c in trading_cfg.items() if c.get('enabled')]
    print(f"  Min confidence: {', '.join(conf_parts)}")
    print(f"  Telegram: /help /status /conf /setup /balance /sync /pause /resume /stop")
    print(f"  Hot-reload: every 5 min (config + models + positions)")
    print(f"{'='*60}")

    _flush_old_updates()

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
    print("  REVOLUT X MULTI-ASSET TRADER [DOOHAN]")
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
                print(f"\n  {asset} [{cfg.get('strategy','?')}]:")
                print(f"    State: {pos['state'].upper()} | Max: ${cfg.get('max_position_usd',0):,.2f}")
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

    if not os.path.exists(MODELS_CSV):
        print("\n  ERROR: No Doohan models found! Run crypto_trading_system_doohan.py Mode D first.")
        return

    # Interactive menu
    if len(args) == 0:
        print("\n  Current config:")
        print(f"  {'Asset':<6} {'Strategy':<14} {'Max USD':>10} {'Enabled':>8} {'Auto':>7} {'Position':>10}")
        print(f"  {'-'*6} {'-'*14} {'-'*10} {'-'*8} {'-'*7} {'-'*10}")
        for asset, cfg in trading_cfg.items():
            pos = load_position(asset)
            _cfg_h = cfg.get('horizon')
            has_model = load_best_config(asset, horizon=_cfg_h) if _cfg_h else (load_best_config(asset, horizon=HORIZON_SHORT) or load_best_config(asset, horizon=HORIZON_LONG))
            enabled_str = "Yes" if cfg.get('enabled', True) else "No"
            auto_str    = "Yes" if pos.get('auto_trade') else "No"
            pos_str     = pos['state'].upper()
            model_mark  = "" if has_model else " ✗"
            print(f"  {asset:<6} {cfg.get('strategy','?'):<14} ${cfg.get('max_position_usd',0):>9,.0f} {enabled_str:>8} {auto_str:>7} {pos_str:>10}{model_mark}")

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
            VALID_STRATEGIES = ('both_agree', 'either_agree', f'{HORIZON_SHORT}h_only', f'{HORIZON_LONG}h_only', 'any_agree')
            for asset in trading_cfg:
                cfg = trading_cfg[asset]
                pos = load_position(asset)
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

                max_inp = input(f"    Max USD [{cfg.get('max_position_usd', 0):.0f}]: ").strip()
                if max_inp:
                    try:
                        cfg['max_position_usd'] = float(max_inp)
                    except ValueError:
                        pass

                strat = input(f"    Strategy (Enter for default '{cfg['strategy']}'): ").strip().lower()
                if strat in VALID_STRATEGIES: cfg['strategy'] = strat
                elif strat: print(f"    Unknown strategy '{strat}' — keeping [{cfg['strategy']}]")

            save_trading_config(trading_cfg)
            print("\n  Config saved.")
            return
        elif ch == '3': dry_run = True
        elif ch == '2': pass  # run once below
        elif ch == '1': loop_mode = True

    # Ensure max positions are set
    needs_config = False
    for asset, cfg in trading_cfg.items():
        if cfg.get('enabled') and cfg.get('max_position_usd', 0) <= 0:
            _cfg_h = cfg.get('horizon')
            has_model = load_best_config(asset, horizon=_cfg_h) if _cfg_h else (load_best_config(asset, horizon=HORIZON_SHORT) or load_best_config(asset, horizon=HORIZON_LONG))
            if has_model:
                needs_config = True
                break

    if needs_config:
        print(f"\n  ⚠️ Max positions not set. Configure now:")
        for asset, cfg in trading_cfg.items():
            if not cfg.get('enabled'): continue
            _cfg_h = cfg.get('horizon')
            has_model = load_best_config(asset, horizon=_cfg_h) if _cfg_h else (load_best_config(asset, horizon=HORIZON_SHORT) or load_best_config(asset, horizon=HORIZON_LONG))
            if not has_model: continue
            if cfg.get('max_position_usd', 0) <= 0:
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
