"""
IB Automatic Hourly Trader
============================
Fully automatic trading system for European Index CFDs via Interactive Brokers.
Connects to TWS/IB Gateway, runs the hourly ML model, and places orders.

Instruments:
  SMI   -> IBCH20 (Swiss 20 CFD)
  DAX   -> IBDE40 (Germany 40 CFD)
  CAC40 -> IBFR40 (France 40 CFD)

Risk Controls:
  - Max position size (% of net liquidation)
  - Stop-loss per trade (% from entry)
  - Max daily loss limit (% of starting equity)
  - Max open positions limit
  - Market hours check (only trades during exchange hours)

Requirements:
  pip install ib_insync
  TWS or IB Gateway running with API enabled on port 7497 (paper)

Usage:
  python ib_auto_trader.py              # Run once (check signals, execute)
  python ib_auto_trader.py --loop       # Run continuously every hour
  python ib_auto_trader.py --status     # Show positions and P&L
  python ib_auto_trader.py --close-all  # Close all positions
"""

import sys
import os

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Python 3.14 fix: always create event loop before importing ib_insync
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import time
import json
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# --- IB Connection ---
IB_HOST = '127.0.0.1'
IB_PORT = 4002          # 4002 = IB Gateway paper, 4001 = IB Gateway live
IB_CLIENT_ID = 10       # Unique client ID for this bot

# --- Instrument Mapping ---
# Maps our asset names to IB CFD contracts
INSTRUMENTS = {
    'SMI': {
        'ib_symbol': 'IBCH20',
        'exchange': 'SMART',
        'currency': 'CHF',
        'data_file': 'smi_hourly_data.csv',
        'min_order_size': 1,       # Minimum CFD units
        'market_open_utc': 7,      # SIX opens ~07:00 UTC
        'market_close_utc': 16,    # SIX closes ~16:00 UTC
    },
    'DAX': {
        'ib_symbol': 'IBDE40',
        'exchange': 'SMART',
        'currency': 'EUR',
        'data_file': 'dax_hourly_data.csv',
        'min_order_size': 1,
        'market_open_utc': 7,      # XETRA opens ~07:00 UTC
        'market_close_utc': 16,    # XETRA closes ~16:00 UTC
    },
    'CAC40': {
        'ib_symbol': 'IBFR40',
        'exchange': 'SMART',
        'currency': 'EUR',
        'data_file': 'cac40_hourly_data.csv',
        'min_order_size': 1,
        'market_open_utc': 7,      # Euronext opens ~07:00 UTC
        'market_close_utc': 16,    # Euronext closes ~16:30 UTC
    },
}

# --- Risk Controls ---
MAX_POSITION_PCT = 20.0       # Max % of portfolio per position
STOP_LOSS_PCT = 2.0           # Stop-loss: close if down X% from entry
MAX_DAILY_LOSS_PCT = 5.0      # Max daily loss: stop trading if hit
MAX_OPEN_POSITIONS = 3        # Max concurrent open positions
COOLDOWN_AFTER_STOP = 2       # Hours to wait after a stop-loss triggers

# --- Trading Parameters ---
ORDER_TYPE = 'MKT'            # Market orders (use 'LMT' for limit)
SIGNAL_MIN_CONFIDENCE = 55.0  # Only trade signals above this confidence

# --- File Paths ---
TRADE_LOG_FILE = 'ib_trade_log.csv'
STATE_FILE = 'ib_trader_state.json'
LOG_FILE = 'ib_auto_trader.log'


# ============================================================
# LOGGING SETUP
# ============================================================
def setup_logging():
    """Configure logging to file and console."""
    logger = logging.getLogger('IBTrader')
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

log = setup_logging()


# ============================================================
# STATE MANAGEMENT (persist between runs)
# ============================================================
def load_state():
    """Load trader state from JSON file."""
    default_state = {
        'positions': {},          # {asset: {side, entry_price, size, entry_time}}
        'daily_pnl': 0.0,
        'daily_start_equity': 0.0,
        'last_trade_date': None,
        'cooldowns': {},          # {asset: cooldown_until_iso}
        'trade_count_today': 0,
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            # Merge with defaults for any missing keys
            for k, v in default_state.items():
                if k not in state:
                    state[k] = v
            return state
        except Exception as e:
            log.warning(f"Could not load state: {e}. Using defaults.")
    return default_state


def save_state(state):
    """Save trader state to JSON file."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


# ============================================================
# TRADE LOGGING
# ============================================================
def log_trade(action, asset, price, size, signal, confidence, reason=''):
    """Append trade to CSV log."""
    trade = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'action': action,
        'asset': asset,
        'ib_symbol': INSTRUMENTS[asset]['ib_symbol'],
        'price': round(price, 2),
        'size': size,
        'signal': signal,
        'confidence': confidence,
        'reason': reason,
    }
    df_new = pd.DataFrame([trade])
    if os.path.exists(TRADE_LOG_FILE):
        df_new.to_csv(TRADE_LOG_FILE, mode='a', header=False, index=False)
    else:
        df_new.to_csv(TRADE_LOG_FILE, index=False)
    log.info(f"TRADE LOGGED: {action} {size} {asset} @ {price:.2f} | {reason}")


# ============================================================
# IB CONNECTION
# ============================================================
class IBConnection:
    """Wrapper for ib_insync connection to Interactive Brokers."""

    def __init__(self):
        self.ib = None
        self.connected = False

    def connect(self):
        """Connect to TWS/IB Gateway."""
        try:
            from ib_insync import IB
            self.ib = IB()
            self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
            self.connected = True
            # Request delayed data as fallback (free, no subscription needed)
            self.ib.reqMarketDataType(3)
            log.info(f"Connected to IB on {IB_HOST}:{IB_PORT} "
                     f"(client={IB_CLIENT_ID}, market data: delayed)")

            # Log account info
            account_values = self.ib.accountSummary()
            for av in account_values:
                if av.tag == 'NetLiquidation':
                    log.info(f"Account: {av.account} | "
                             f"Net Liquidation: {av.currency} {float(av.value):,.2f}")
            return True

        except Exception as e:
            log.error(f"Failed to connect to IB: {e}")
            log.error("Make sure TWS or IB Gateway is running with API enabled.")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from IB."""
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            log.info("Disconnected from IB")

    def get_net_liquidation(self):
        """Get account net liquidation value."""
        if not self.connected:
            return 0
        account_values = self.ib.accountSummary()
        for av in account_values:
            if av.tag == 'NetLiquidation' and av.currency == 'BASE':
                return float(av.value)
        # Fallback: try without currency filter
        for av in account_values:
            if av.tag == 'NetLiquidation':
                return float(av.value)
        return 0

    def get_positions(self):
        """Get current positions as dict."""
        if not self.connected:
            return {}
        positions = {}
        for pos in self.ib.positions():
            symbol = pos.contract.symbol
            positions[symbol] = {
                'size': pos.position,
                'avg_cost': pos.avgCost,
                'contract': pos.contract,
            }
        return positions

    def create_cfd_contract(self, asset_name):
        """Create an IB CFD contract for the given asset."""
        from ib_insync import CFD
        inst = INSTRUMENTS[asset_name]
        contract = CFD(
            symbol=inst['ib_symbol'],
            exchange=inst['exchange'],
            currency=inst['currency'],
        )
        # Qualify the contract (resolve details)
        qualified = self.ib.qualifyContracts(contract)
        if qualified:
            return qualified[0]
        else:
            log.error(f"Could not qualify contract for {asset_name} "
                      f"({inst['ib_symbol']})")
            return contract

    def get_market_price(self, contract):
        """Get current market price for a contract."""
        ticker = self.ib.reqMktData(contract, '', False, False)
        self.ib.sleep(2)  # Wait for data
        price = ticker.marketPrice()
        self.ib.cancelMktData(contract)

        if np.isnan(price):
            # Try last price
            price = ticker.last
        if np.isnan(price):
            price = ticker.close
        return float(price) if not np.isnan(price) else None

    def place_market_order(self, contract, action, quantity):
        """
        Place a market order.
        action: 'BUY' or 'SELL'
        quantity: number of CFD units (positive)
        Returns order and trade objects.
        """
        from ib_insync import MarketOrder
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)  # Wait for acknowledgement
        log.info(f"Order placed: {action} {quantity} {contract.symbol} "
                 f"(status: {trade.orderStatus.status})")
        return trade

    def place_stop_order(self, contract, action, quantity, stop_price):
        """Place a stop-loss order."""
        from ib_insync import StopOrder
        order = StopOrder(action, quantity, stop_price)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)
        log.info(f"Stop order placed: {action} {quantity} {contract.symbol} "
                 f"@ {stop_price:.2f} (status: {trade.orderStatus.status})")
        return trade

    def cancel_all_orders(self):
        """Cancel all open orders."""
        if not self.connected:
            return
        open_orders = self.ib.openOrders()
        for order in open_orders:
            self.ib.cancelOrder(order)
        log.info(f"Cancelled {len(open_orders)} open orders")


# ============================================================
# SIGNAL GENERATION (minimal — generates signal for current hour)
# ============================================================
def get_current_signal(asset_name):
    """
    Generate the current hourly signal for an asset.
    Uses the hourly trading system's feature engineering.
    Reads hourly_best_models.csv for model config.
    Returns: (signal, confidence, price) or (None, None, None)
    """
    # Import feature engineering from hourly system
    # We inline it here to avoid circular imports
    from hourly_trading_system import (
        load_data, build_hourly_features, ALL_MODELS, PREDICTION_HORIZON
    )
    from sklearn.preprocessing import StandardScaler

    # Load best model config
    if not os.path.exists('hourly_best_models.csv'):
        log.error("hourly_best_models.csv not found! Run diagnostic first.")
        return None, None, None

    df_best = pd.read_csv('hourly_best_models.csv')
    asset_row = df_best[df_best['coin'] == asset_name]
    if len(asset_row) == 0:
        log.warning(f"No model config for {asset_name} in hourly_best_models.csv")
        return None, None, None

    model_names = asset_row.iloc[0]['models'].split('+')
    window_size = int(asset_row.iloc[0]['best_window'])

    # Load and prepare data
    df_raw = load_data(asset_name)
    if df_raw is None:
        return None, None, None

    df_features, feature_cols = build_hourly_features(df_raw)
    if len(df_features) < window_size + 50:
        log.warning(f"{asset_name}: not enough data for window={window_size}")
        return None, None, None

    # Use last row as test, preceding window as train
    n = len(df_features)
    i = n - 1
    row = df_features.iloc[i]

    train_start = max(0, i - window_size)
    train = df_features.iloc[train_start:i]
    X_train = train[feature_cols]
    y_train = train['label'].values
    X_test = df_features.iloc[i:i+1][feature_cols]

    if len(np.unique(y_train)) < 2:
        log.warning(f"{asset_name}: insufficient label variety in training data")
        return None, None, None

    # Scale
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train),
                             columns=feature_cols, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test),
                            columns=feature_cols, index=X_test.index)

    # Predict with ensemble
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
            log.warning(f"{asset_name}/{model_name}: prediction error: {e}")

    if not votes:
        return None, None, None

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
    confidence = (1 - avg_proba) * 100 if signal == 'SELL' else avg_proba * 100

    price = float(row['close'])
    dt = row['datetime']

    log.info(f"{asset_name} @ {dt} | Signal: {signal} ({confidence:.1f}%) | "
             f"Price: {price:,.2f} | Models: {'+'.join(model_names)} "
             f"({buy_votes}/{total_votes} buy)")

    return signal, round(confidence, 1), price


# ============================================================
# RISK MANAGER
# ============================================================
class RiskManager:
    """Enforces all risk controls before allowing trades."""

    def __init__(self, ib_conn, state):
        self.ib = ib_conn
        self.state = state

    def check_market_hours(self, asset_name):
        """Check if market is open for this asset."""
        inst = INSTRUMENTS[asset_name]
        now_utc = datetime.utcnow()
        hour = now_utc.hour
        weekday = now_utc.weekday()  # 0=Mon, 6=Sun

        if weekday >= 5:  # Weekend
            log.info(f"{asset_name}: Market closed (weekend)")
            return False

        if hour < inst['market_open_utc'] or hour >= inst['market_close_utc']:
            log.info(f"{asset_name}: Market closed "
                     f"(hours: {inst['market_open_utc']}-{inst['market_close_utc']} UTC, "
                     f"current: {hour} UTC)")
            return False

        return True

    def check_daily_loss_limit(self):
        """Check if daily loss limit has been hit."""
        nlv = self.ib.get_net_liquidation()
        if nlv <= 0:
            return True  # Can't check, allow

        start_equity = self.state.get('daily_start_equity', nlv)

        # Reset daily tracking if new day
        today = datetime.now().strftime('%Y-%m-%d')
        if self.state.get('last_trade_date') != today:
            self.state['daily_start_equity'] = nlv
            self.state['daily_pnl'] = 0.0
            self.state['trade_count_today'] = 0
            self.state['last_trade_date'] = today
            start_equity = nlv

        if start_equity > 0:
            daily_loss_pct = ((nlv - start_equity) / start_equity) * 100
            if daily_loss_pct < -MAX_DAILY_LOSS_PCT:
                log.warning(f"DAILY LOSS LIMIT HIT: {daily_loss_pct:.1f}% "
                            f"(limit: -{MAX_DAILY_LOSS_PCT}%)")
                return False

        return True

    def check_max_positions(self):
        """Check if max open positions limit reached."""
        open_count = len(self.state.get('positions', {}))
        if open_count >= MAX_OPEN_POSITIONS:
            log.info(f"Max open positions reached ({open_count}/{MAX_OPEN_POSITIONS})")
            return False
        return True

    def check_cooldown(self, asset_name):
        """Check if asset is in cooldown after stop-loss."""
        cooldowns = self.state.get('cooldowns', {})
        if asset_name in cooldowns:
            cooldown_until = datetime.fromisoformat(cooldowns[asset_name])
            if datetime.now() < cooldown_until:
                remaining = (cooldown_until - datetime.now()).total_seconds() / 3600
                log.info(f"{asset_name}: In cooldown for {remaining:.1f} more hours")
                return False
            else:
                # Cooldown expired
                del cooldowns[asset_name]
                self.state['cooldowns'] = cooldowns
        return True

    def calculate_position_size(self, asset_name, price):
        """
        Calculate position size based on max position % of portfolio.
        Returns number of CFD units.
        """
        nlv = self.ib.get_net_liquidation()
        if nlv <= 0 or price <= 0:
            return 0

        max_value = nlv * (MAX_POSITION_PCT / 100.0)
        size = int(max_value / price)

        inst = INSTRUMENTS[asset_name]
        if size < inst['min_order_size']:
            log.info(f"{asset_name}: Calculated size {size} < minimum "
                     f"{inst['min_order_size']}. Skipping.")
            return 0

        return size

    def calculate_stop_price(self, entry_price, side):
        """Calculate stop-loss price."""
        if side == 'BUY':
            return round(entry_price * (1 - STOP_LOSS_PCT / 100), 2)
        else:  # SELL / short
            return round(entry_price * (1 + STOP_LOSS_PCT / 100), 2)

    def can_trade(self, asset_name):
        """Run all risk checks. Returns True if trading is allowed."""
        checks = [
            ('Market hours', self.check_market_hours(asset_name)),
            ('Daily loss limit', self.check_daily_loss_limit()),
            ('Max positions', self.check_max_positions()),
            ('Cooldown', self.check_cooldown(asset_name)),
        ]

        all_passed = True
        for name, passed in checks:
            if not passed:
                log.info(f"Risk check FAILED: {name} for {asset_name}")
                all_passed = False

        return all_passed

    def can_close(self, asset_name):
        """Check if we can close a position (fewer restrictions)."""
        # Always allow closing during market hours
        return self.check_market_hours(asset_name)


# ============================================================
# TRADE EXECUTION ENGINE
# ============================================================
class TradeExecutor:
    """Executes trades based on signals with risk controls."""

    def __init__(self, ib_conn, risk_mgr, state):
        self.ib = ib_conn
        self.risk = risk_mgr
        self.state = state

    def process_signal(self, asset_name, signal, confidence, price):
        """
        Process a signal for an asset. Decides whether to open/close/hold.
        Returns action taken.
        """
        log.info(f"\nProcessing {asset_name}: signal={signal}, "
                 f"confidence={confidence}%, price={price:,.2f}")

        # Check minimum confidence
        if confidence < SIGNAL_MIN_CONFIDENCE and signal != 'HOLD':
            log.info(f"Confidence {confidence}% < minimum {SIGNAL_MIN_CONFIDENCE}%. "
                     f"Treating as HOLD.")
            signal = 'HOLD'

        has_position = asset_name in self.state.get('positions', {})

        if signal == 'BUY':
            if has_position:
                pos = self.state['positions'][asset_name]
                if pos['side'] == 'LONG':
                    log.info(f"{asset_name}: Already LONG. Holding.")
                    self._check_stop_loss(asset_name, price)
                    return 'HOLD_LONG'
                else:
                    # Close short, open long
                    self._close_position(asset_name, price, 'Signal flipped to BUY')
                    return self._open_position(asset_name, 'BUY', price, confidence)
            else:
                # Open new long
                return self._open_position(asset_name, 'BUY', price, confidence)

        elif signal == 'SELL':
            if has_position:
                pos = self.state['positions'][asset_name]
                if pos['side'] == 'LONG':
                    # Close long
                    self._close_position(asset_name, price, 'SELL signal')
                    return 'CLOSED_LONG'
                else:
                    log.info(f"{asset_name}: Already in SELL/cash. Holding.")
                    return 'HOLD_CASH'
            else:
                log.info(f"{asset_name}: No position, SELL signal. Staying cash.")
                return 'STAY_CASH'

        else:  # HOLD
            if has_position:
                self._check_stop_loss(asset_name, price)
                return 'HOLD_POSITION'
            return 'HOLD_CASH'

    def _open_position(self, asset_name, side, price, confidence):
        """Open a new position."""
        # Risk checks
        if not self.risk.can_trade(asset_name):
            log.info(f"{asset_name}: Risk checks failed. Not opening position.")
            return 'RISK_BLOCKED'

        # Calculate size
        size = self.risk.calculate_position_size(asset_name, price)
        if size <= 0:
            log.info(f"{asset_name}: Position size too small. Skipping.")
            return 'SIZE_TOO_SMALL'

        # Create contract and place order
        contract = self.ib.create_cfd_contract(asset_name)
        action = 'BUY' if side == 'BUY' else 'SELL'

        log.info(f"OPENING {action} {size} x {INSTRUMENTS[asset_name]['ib_symbol']} "
                 f"@ ~{price:,.2f}")

        try:
            trade = self.ib.place_market_order(contract, action, size)

            # Wait for fill
            self.ib.ib.sleep(3)

            # Get fill price
            fill_price = price  # Default to signal price
            if trade.fills:
                fill_price = trade.fills[0].execution.price

            # Place stop-loss
            stop_price = self.risk.calculate_stop_price(fill_price, side)
            stop_action = 'SELL' if side == 'BUY' else 'BUY'
            self.ib.place_stop_order(contract, stop_action, size, stop_price)

            # Record position
            if 'positions' not in self.state:
                self.state['positions'] = {}
            self.state['positions'][asset_name] = {
                'side': 'LONG' if side == 'BUY' else 'SHORT',
                'entry_price': fill_price,
                'size': size,
                'entry_time': datetime.now().isoformat(),
                'stop_price': stop_price,
            }
            self.state['trade_count_today'] = self.state.get('trade_count_today', 0) + 1
            save_state(self.state)

            # Log trade
            log_trade(action, asset_name, fill_price, size, side, confidence,
                      f'Open {side} | stop={stop_price:.2f}')

            log.info(f"OPENED: {side} {size} x {asset_name} @ {fill_price:,.2f} "
                     f"| Stop: {stop_price:,.2f}")
            return f'OPENED_{side}'

        except Exception as e:
            log.error(f"ORDER FAILED for {asset_name}: {e}")
            return 'ORDER_FAILED'

    def _close_position(self, asset_name, price, reason):
        """Close an existing position."""
        if asset_name not in self.state.get('positions', {}):
            return

        pos = self.state['positions'][asset_name]
        side = pos['side']
        size = pos['size']
        entry_price = pos['entry_price']

        if not self.risk.can_close(asset_name):
            log.info(f"{asset_name}: Cannot close (market closed)")
            return

        contract = self.ib.create_cfd_contract(asset_name)
        # Close: sell if long, buy if short
        action = 'SELL' if side == 'LONG' else 'BUY'

        log.info(f"CLOSING {side} {size} x {asset_name} | Reason: {reason}")

        try:
            # Cancel any existing stop orders for this contract
            self.ib.ib.sleep(0.5)
            for order_trade in self.ib.ib.openTrades():
                if order_trade.contract.symbol == INSTRUMENTS[asset_name]['ib_symbol']:
                    self.ib.ib.cancelOrder(order_trade.order)

            trade = self.ib.place_market_order(contract, action, size)
            self.ib.ib.sleep(3)

            fill_price = price
            if trade.fills:
                fill_price = trade.fills[0].execution.price

            # Calculate P&L
            if side == 'LONG':
                pnl_pct = ((fill_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - fill_price) / entry_price) * 100

            # Remove position from state
            del self.state['positions'][asset_name]
            save_state(self.state)

            log_trade(action, asset_name, fill_price, size, 'CLOSE', 0,
                      f'Close {side} | PnL: {pnl_pct:+.2f}% | {reason}')

            log.info(f"CLOSED: {side} {size} x {asset_name} @ {fill_price:,.2f} "
                     f"| PnL: {pnl_pct:+.2f}% | {reason}")

        except Exception as e:
            log.error(f"CLOSE FAILED for {asset_name}: {e}")

    def _check_stop_loss(self, asset_name, current_price):
        """Check if stop-loss should trigger (backup to IB's stop order)."""
        if asset_name not in self.state.get('positions', {}):
            return

        pos = self.state['positions'][asset_name]
        entry_price = pos['entry_price']
        side = pos['side']

        if side == 'LONG':
            loss_pct = ((current_price - entry_price) / entry_price) * 100
            if loss_pct < -STOP_LOSS_PCT:
                log.warning(f"STOP-LOSS TRIGGERED: {asset_name} LONG "
                            f"(loss={loss_pct:.1f}%, limit=-{STOP_LOSS_PCT}%)")
                self._close_position(asset_name, current_price,
                                     f'Stop-loss ({loss_pct:.1f}%)')
                # Set cooldown
                self.state['cooldowns'][asset_name] = (
                    datetime.now() + timedelta(hours=COOLDOWN_AFTER_STOP)
                ).isoformat()
                save_state(self.state)


# ============================================================
# MAIN TRADING LOOP
# ============================================================
def run_trading_cycle(ib_conn=None):
    """
    Run one trading cycle:
    1. Update data for all indices
    2. Generate signals
    3. Execute trades
    """
    log.info("=" * 60)
    log.info("  HOURLY TRADING CYCLE")
    log.info(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # Connect to IB if not already
    own_connection = False
    if ib_conn is None:
        ib_conn = IBConnection()
        if not ib_conn.connect():
            return False
        own_connection = True

    # Load state
    state = load_state()

    # Initialize risk manager and executor
    risk_mgr = RiskManager(ib_conn, state)
    executor = TradeExecutor(ib_conn, risk_mgr, state)

    # Step 1: Update data
    log.info("\n--- Updating data ---")
    try:
        from hourly_trading_system import update_all_data
        update_all_data(list(INSTRUMENTS.keys()))
    except Exception as e:
        log.error(f"Data update failed: {e}")
        if own_connection:
            ib_conn.disconnect()
        return False

    # Step 2: Generate signals and execute
    log.info("\n--- Generating signals & executing ---")
    results = {}

    for asset_name in INSTRUMENTS:
        try:
            signal, confidence, price = get_current_signal(asset_name)
            if signal is None:
                log.warning(f"{asset_name}: No signal generated")
                results[asset_name] = 'NO_SIGNAL'
                continue

            action = executor.process_signal(asset_name, signal, confidence, price)
            results[asset_name] = action

        except Exception as e:
            log.error(f"{asset_name}: Error in trading cycle: {e}")
            results[asset_name] = f'ERROR: {e}'

    # Summary
    log.info("\n--- Cycle Summary ---")
    for asset, result in results.items():
        log.info(f"  {asset:6s}: {result}")

    # Show positions
    positions = state.get('positions', {})
    if positions:
        log.info("\n--- Open Positions ---")
        for asset, pos in positions.items():
            log.info(f"  {asset:6s}: {pos['side']} {pos['size']}x "
                     f"@ {pos['entry_price']:,.2f} "
                     f"(stop: {pos['stop_price']:,.2f})")

    nlv = ib_conn.get_net_liquidation()
    log.info(f"\n  Net Liquidation: {nlv:,.2f}")

    save_state(state)

    if own_connection:
        ib_conn.disconnect()

    return True


def run_continuous_loop():
    """Run trading cycle every hour during market hours."""
    log.info("=" * 60)
    log.info("  STARTING CONTINUOUS HOURLY TRADING")
    log.info(f"  Press Ctrl+C to stop")
    log.info("=" * 60)

    ib_conn = IBConnection()
    if not ib_conn.connect():
        return

    try:
        while True:
            now = datetime.utcnow()

            # Only run during rough market hours (7-17 UTC on weekdays)
            if now.weekday() < 5 and 6 <= now.hour <= 17:
                try:
                    # Reconnect if needed
                    if not ib_conn.connected:
                        ib_conn.connect()
                    run_trading_cycle(ib_conn)
                except Exception as e:
                    log.error(f"Trading cycle error: {e}")
                    # Try to reconnect
                    try:
                        ib_conn.disconnect()
                    except:
                        pass
                    ib_conn = IBConnection()
                    ib_conn.connect()
            else:
                log.info(f"Outside market hours (UTC {now.hour}:00, "
                         f"{'weekend' if now.weekday() >= 5 else 'weekday'}). "
                         f"Sleeping...")

            # Sleep until next hour
            now = datetime.now()
            next_hour = (now + timedelta(hours=1)).replace(
                minute=5, second=0, microsecond=0
            )
            sleep_seconds = (next_hour - now).total_seconds()
            log.info(f"Next cycle: {next_hour.strftime('%H:%M:%S')} "
                     f"(sleeping {sleep_seconds/60:.0f} min)")
            time.sleep(max(sleep_seconds, 60))

    except KeyboardInterrupt:
        log.info("\nStopping... (Ctrl+C)")
    finally:
        ib_conn.disconnect()
        log.info("Trader stopped.")


def show_status():
    """Show current positions and P&L."""
    print("=" * 60)
    print("  IB TRADER STATUS")
    print("=" * 60)

    state = load_state()

    # Positions
    positions = state.get('positions', {})
    if positions:
        print(f"\n  Open Positions ({len(positions)}):")
        for asset, pos in positions.items():
            print(f"    {asset:6s}: {pos['side']} {pos['size']}x "
                  f"@ {pos['entry_price']:,.2f} | "
                  f"Stop: {pos['stop_price']:,.2f} | "
                  f"Since: {pos['entry_time'][:16]}")
    else:
        print("\n  No open positions.")

    # Cooldowns
    cooldowns = state.get('cooldowns', {})
    if cooldowns:
        print(f"\n  Cooldowns:")
        for asset, until in cooldowns.items():
            print(f"    {asset}: until {until[:16]}")

    # Today's stats
    print(f"\n  Daily Stats:")
    print(f"    Trades today: {state.get('trade_count_today', 0)}")
    print(f"    Daily P&L: {state.get('daily_pnl', 0):.2f}")
    print(f"    Start equity: {state.get('daily_start_equity', 0):,.2f}")

    # Trade log
    if os.path.exists(TRADE_LOG_FILE):
        df = pd.read_csv(TRADE_LOG_FILE)
        print(f"\n  Trade Log ({len(df)} total trades):")
        if len(df) > 0:
            for _, row in df.tail(10).iterrows():
                print(f"    {row['timestamp']} | {row['action']:4s} "
                      f"{row['size']}x {row['asset']} @ {row['price']:,.2f} "
                      f"| {row['reason']}")

    # Try to connect for live info
    try:
        ib_conn = IBConnection()
        if ib_conn.connect():
            nlv = ib_conn.get_net_liquidation()
            print(f"\n  Live Net Liquidation: {nlv:,.2f}")
            ib_positions = ib_conn.get_positions()
            if ib_positions:
                print(f"\n  IB Positions:")
                for sym, pos in ib_positions.items():
                    print(f"    {sym}: {pos['size']} units @ avg {pos['avg_cost']:,.2f}")
            ib_conn.disconnect()
    except:
        print("\n  (Could not connect to IB for live data)")


def close_all_positions():
    """Emergency close all positions."""
    log.info("=" * 60)
    log.info("  CLOSING ALL POSITIONS")
    log.info("=" * 60)

    state = load_state()
    positions = state.get('positions', {})

    if not positions:
        log.info("No open positions to close.")
        return

    ib_conn = IBConnection()
    if not ib_conn.connect():
        return

    risk_mgr = RiskManager(ib_conn, state)
    executor = TradeExecutor(ib_conn, risk_mgr, state)

    for asset_name in list(positions.keys()):
        try:
            pos = positions[asset_name]
            contract = ib_conn.create_cfd_contract(asset_name)
            price = ib_conn.get_market_price(contract)
            if price:
                executor._close_position(asset_name, price, 'Manual close-all')
            else:
                log.error(f"{asset_name}: Could not get market price for close")
        except Exception as e:
            log.error(f"{asset_name}: Close failed: {e}")

    # Cancel all remaining orders
    ib_conn.cancel_all_orders()
    ib_conn.disconnect()

    log.info("All positions closed.")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='IB Automatic Hourly Trader')
    parser.add_argument('--loop', action='store_true',
                        help='Run continuously every hour')
    parser.add_argument('--status', action='store_true',
                        help='Show positions and P&L')
    parser.add_argument('--close-all', action='store_true',
                        help='Close all positions')
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.close_all:
        close_all_positions()
    elif args.loop:
        run_continuous_loop()
    else:
        # Single run
        run_trading_cycle()


if __name__ == '__main__':
    main()
