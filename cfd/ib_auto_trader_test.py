"""
IB Auto Trader TEST — S&P 500 CFD (IBUS500)
=============================================
TEST version for overnight trading while DAX is closed.
Trades the S&P 500 CFD on Interactive Brokers.

Instrument:
  S&P 500 -> IBUS500 (US 500 CFD, USD)

Trading hours:
  Nearly 24h: Sunday 23:00 UTC to Friday 22:00 UTC
  (brief daily break ~22:00-23:00 UTC)

Uses same Broly 1.2 ML pipeline with self-contained features.
Separate state/logs from DAX trader (no conflict).

Usage:
  python ib_auto_trader_test.py              # Run once
  python ib_auto_trader_test.py --loop       # Continuous
  python ib_auto_trader_test.py --status     # Show position
  python ib_auto_trader_test.py --close-all  # Emergency close
"""

import sys
import os

# Allow imports from parent directory (engine/) and use engine/ as working dir
_ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, _ENGINE_DIR)
os.chdir(_ENGINE_DIR)

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
IB_CLIENT_ID = 20       # Different from DAX trader (clientId=10)

# --- Instrument Mapping ---
# S&P 500 CFD — trades nearly 24h
INSTRUMENTS = {
    'SP500': {
        'ib_symbol': 'IBUS500',
        'exchange': 'SMART',
        'currency': 'USD',
        'data_file': 'data/indices/sp500_hourly_data.csv',
        'min_order_size': 1,
        'market_open_utc': 23,     # Sunday 23:00 UTC (opens)
        'market_close_utc': 22,    # Friday 22:00 UTC (closes)
    },
}

# --- Risk Controls ---
MAX_BUDGET_EUR = 10_000.0     # Fixed max margin budget (EUR)
CFD_MARGIN_PCT = 5.0          # CFD margin requirement (%)
STOP_LOSS_PCT = 2.0           # Stop-loss: close if down X% from entry
MAX_DAILY_LOSS_EUR = 2_000.0  # Max daily loss in EUR (stop trading if hit)
MAX_OPEN_POSITIONS = 1        # Max concurrent open positions
COOLDOWN_AFTER_STOP = 2       # Hours to wait after a stop-loss triggers

# --- Trading Parameters ---
ORDER_TYPE = 'MKT'            # Market orders
SIGNAL_MIN_CONFIDENCE = 55.0  # Only trade signals above this confidence

# --- File Paths (separate from DAX trader) ---
TRADE_LOG_FILE = 'data/ib_trade_log_test.csv'
STATE_FILE = 'data/ib_trader_state_test.json'
DASHBOARD_FILE = 'output/dashboards/ib_live_data_test.json'
LOG_FILE = 'ib_auto_trader_test.log'


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
# DATA UPDATE (from Interactive Brokers)
# ============================================================
def update_market_data(ib_conn):
    """
    Fetch latest hourly bars from IB for S&P 500.
    Creates the CSV if it doesn't exist (bootstrap mode).
    """
    asset_name = 'SP500'
    data_file = INSTRUMENTS[asset_name]['data_file']

    if not ib_conn.connected:
        log.error("Cannot update data: not connected to IB")
        return False

    contract = ib_conn.create_cfd_contract(asset_name)

    # If CSV doesn't exist, bootstrap with 30 days
    if not os.path.exists(data_file):
        log.info(f"SP500: No CSV found — bootstrapping with 30 days of hourly data")
        duration = '30 D'
    else:
        existing = pd.read_csv(data_file)
        existing['datetime'] = pd.to_datetime(existing['datetime'])
        last_dt = existing['datetime'].max()
        hours_behind = (datetime.now() - last_dt).total_seconds() / 3600
        if hours_behind < 2:
            log.info(f"SP500: Already up to date (last: {last_dt})")
            return True
        days_needed = max(1, int(hours_behind / 24) + 2)
        duration = f'{min(days_needed, 30)} D'
        log.info(f"SP500: {hours_behind:.0f}h behind, fetching {duration}")

    try:
        bars = ib_conn.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting='1 hour',
            whatToShow='MIDPOINT',
            useRTH=False,          # Include extended hours (SP500 trades ~24h)
            formatDate=1,
        )
        ib_conn.ib.sleep(2)

        if not bars:
            log.warning("SP500: No historical bars received from IB")
            return False

        # Convert to DataFrame
        rows = []
        for bar in bars:
            rows.append({
                'datetime': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume if bar.volume > 0 else 0,
            })
        df_new = pd.DataFrame(rows)
        df_new['datetime'] = pd.to_datetime(df_new['datetime']).dt.tz_localize(None)

        if os.path.exists(data_file):
            existing = pd.read_csv(data_file)
            existing['datetime'] = pd.to_datetime(existing['datetime'])
            last_dt = existing['datetime'].max()

            new_rows = df_new[df_new['datetime'] > last_dt]
            if len(new_rows) > 0:
                combined = pd.concat([existing, new_rows], ignore_index=True)
                combined = combined.drop_duplicates(subset='datetime').sort_values('datetime')
                combined.to_csv(data_file, index=False)
                log.info(f"SP500: Added {len(new_rows)} new rows from IB "
                         f"(total: {len(combined)}, last: {combined['datetime'].max()})")
            else:
                log.info(f"SP500: No new rows after {last_dt}")
        else:
            # Bootstrap: create the CSV from scratch
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            df_new = df_new.drop_duplicates(subset='datetime').sort_values('datetime')
            df_new.to_csv(data_file, index=False)
            log.info(f"SP500: Bootstrapped {len(df_new)} rows → {data_file}")

        return True

    except Exception as e:
        log.warning(f"SP500: IB historical data error: {e}")
        return False


def wait_for_new_candle(ib_conn):
    """
    Wait until a new hourly candle is available.
    Polls IB every 5 seconds. Logs status every 2 minutes.
    Never times out — runs until new data arrives or Ctrl+C.
    """
    data_file = INSTRUMENTS['SP500']['data_file']

    if not os.path.exists(data_file):
        return False

    existing = pd.read_csv(data_file)
    existing['datetime'] = pd.to_datetime(existing['datetime'])
    last_dt = existing['datetime'].max()
    log.info(f"Last candle: {last_dt} — polling for next bar...")

    contract = ib_conn.create_cfd_contract('SP500')
    waited = 0

    while True:
        try:
            bars = ib_conn.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='3600 S',
                barSizeSetting='1 hour',
                whatToShow='MIDPOINT',
                useRTH=False,
                formatDate=1,
            )
            ib_conn.ib.sleep(1)

            if bars:
                latest_bar_dt = pd.to_datetime(bars[-1].date).tz_localize(None)
                if latest_bar_dt > last_dt:
                    log.info(f"New candle detected: {latest_bar_dt}")
                    return True

        except Exception as e:
            if waited % 120 == 0:
                log.warning(f"Poll error: {e}")

        time.sleep(5)
        waited += 5
        if waited % 120 == 0:
            log.info(f"Waiting for new candle... ({waited//60}min, last: {last_dt})")


# ============================================================
# SIGNAL GENERATION (Broly 1.2 — V2 features)
# ============================================================
def load_setup_config():
    """
    Self-contained config for S&P 500 TEST.
    Uses the same core features but doesn't need daily_setup.py.
    Falls back to base features only (no features_v2 dependency).
    """
    config = {
        'optimal_features': [
            'logret_1h', 'logret_2h', 'logret_3h', 'logret_6h',
            'logret_12h', 'logret_24h', 'logret_48h', 'logret_72h',
            'sma20_to_sma50h', 'rsi_14h', 'bb_position_20h',
            'zscore_50h', 'atr_pct_14h', 'volatility_48h',
            'hour_sin', 'dow_sin', 'dow_cos',
        ],
        'prediction_horizon': 4,
        'n_features': 17,
        'generated': 'self-contained TEST',
        'best_models_df': pd.DataFrame([{
            'coin': 'SP500',
            'best_window': 200,
            'best_combo': 'RF',
            'models': 'RF',
            'accuracy': 0.0,  # Unknown — test mode
        }]),
    }
    return config


def build_hourly_features(df_raw, prediction_horizon=4):
    """Build base hourly features (same as generate_signals.py)."""
    df = df_raw.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['close'] = df['close'].ffill()

    for h in [1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 120, 240]:
        df[f'logret_{h}h'] = np.log(df['close'] / df['close'].shift(h))
    for fast, slow in [(4, 24), (8, 48), (8, 120), (24, 120)]:
        sma_f = df['close'].rolling(fast).mean()
        sma_s = df['close'].rolling(slow).mean()
        df[f'spread_{slow}h_{fast}h'] = (sma_f - sma_s) / sma_s

    for w in [20, 50, 100, 200]:
        df[f'sma{w}h'] = df['close'].rolling(w).mean()
    df['price_to_sma20h']  = df['close'] / df['sma20h'] - 1
    df['price_to_sma50h']  = df['close'] / df['sma50h'] - 1
    df['price_to_sma100h'] = df['close'] / df['sma100h'] - 1
    df['sma20_to_sma50h']  = df['sma20h'] / df['sma50h'] - 1

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14h'] = 100 - (100 / (1 + rs))

    low14  = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['stoch_k_14h'] = 100 * (df['close'] - low14) / (high14 - low14)

    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position_20h'] = (df['close'] - (bb_mid - 2 * bb_std)) / (4 * bb_std)

    roll_mean = df['close'].rolling(50).mean()
    roll_std  = df['close'].rolling(50).std()
    df['zscore_50h'] = (df['close'] - roll_mean) / roll_std

    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    df['atr_pct_14h'] = tr.rolling(14).mean() / df['close']
    df['intraday_range'] = (df['high'] - df['low']) / df['close']
    df['volatility_12h'] = df['logret_1h'].rolling(12).std()
    df['volatility_48h'] = df['logret_1h'].rolling(48).std()
    df['vol_ratio_12_48'] = df['volatility_12h'] / df['volatility_48h']

    if df['volume'].sum() == 0 or df['volume'].isna().all():
        df['volume_ratio_h'] = 1.0
    else:
        df['volume'] = df['volume'].replace(0, np.nan).ffill().bfill()
        vol_sma = df['volume'].rolling(20).mean()
        df['volume_ratio_h'] = df['volume'] / vol_sma
        df['volume_ratio_h'] = df['volume_ratio_h'].fillna(1.0)

    hour = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    dow = df['datetime'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    future_return = df['close'].shift(-prediction_horizon) / df['close'] - 1
    rolling_median = future_return.rolling(200, min_periods=50).median().shift(prediction_horizon)
    df['label'] = (future_return > rolling_median).astype(int)

    return df


def get_current_signal(asset_name, config=None):
    """
    Self-contained signal generation for S&P 500 TEST.
    Uses base features + RF model. No external dependencies.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    if config is None:
        config = load_setup_config()

    df_best = config['best_models_df']
    features = config['optimal_features']
    pred_h = config['prediction_horizon']

    asset_row = df_best[df_best['coin'] == asset_name]
    if len(asset_row) == 0:
        log.warning(f"No model config for {asset_name}")
        return None, None, None

    window_size = int(asset_row.iloc[0]['best_window'])

    # Load data
    data_file = INSTRUMENTS[asset_name]['data_file']
    if not os.path.exists(data_file):
        log.error(f"Data file not found: {data_file}")
        return None, None, None

    df_raw = pd.read_csv(data_file)
    log.info(f"{asset_name}: Loaded {len(df_raw)} rows from {data_file}")

    # Build features (self-contained, base only)
    df_v2 = build_hourly_features(df_raw, pred_h)

    use_cols = [c for c in features if c in df_v2.columns]
    if len(use_cols) < 5:
        log.error(f"{asset_name}: Only {len(use_cols)} features available")
        return None, None, None

    df_features = df_v2.dropna(subset=use_cols + ['label']).reset_index(drop=True)

    if len(df_features) < window_size + 50:
        log.warning(f"{asset_name}: Not enough data ({len(df_features)}) "
                    f"for window={window_size}")
        return None, None, None

    # Walk-forward: train on window, predict current row
    n = len(df_features)
    i = n - 1
    row = df_features.iloc[i]

    train_start = max(0, i - window_size)
    train = df_features.iloc[train_start:i]
    X_train = train[use_cols]
    y_train = train['label'].values
    X_test = df_features.iloc[i:i+1][use_cols]

    if len(np.unique(y_train)) < 2:
        log.warning(f"{asset_name}: Insufficient label variety")
        return None, None, None

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train),
                             columns=use_cols, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test),
                            columns=use_cols, index=X_test.index)

    # Train RF model
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)[0]
    proba = model.predict_proba(X_test_s)[0][1]

    if pred == 1:
        signal = 'BUY'
        confidence = proba * 100
    else:
        signal = 'SELL'
        confidence = (1 - proba) * 100

    price = float(row['close'])
    dt = row['datetime']

    log.info(f"{asset_name} @ {dt} | Signal: {signal} ({confidence:.1f}%) | "
             f"Price: {price:,.2f} | Model: RF | Features: {len(use_cols)}")

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
        """Check if market is open for S&P 500 CFD.
        IBUS500 trades ~23h/day: Sun 23:00 to Fri 22:00 UTC
        Brief daily break around 22:00-23:00 UTC.
        """
        now_utc = datetime.utcnow()
        hour = now_utc.hour
        weekday = now_utc.weekday()  # 0=Mon, 6=Sun

        # Saturday: always closed
        if weekday == 5:
            log.info(f"{asset_name}: Market closed (Saturday)")
            return False

        # Sunday: only open after 23:00 UTC
        if weekday == 6 and hour < 23:
            log.info(f"{asset_name}: Market closed (Sunday, opens 23:00 UTC)")
            return False

        # Friday: closes at 22:00 UTC
        if weekday == 4 and hour >= 22:
            log.info(f"{asset_name}: Market closed (Friday after 22:00 UTC)")
            return False

        # Daily maintenance break ~22:00-23:00 UTC (Mon-Thu)
        if weekday < 4 and hour == 22:
            log.info(f"{asset_name}: Daily maintenance break (22:00-23:00 UTC)")
            return False

        return True

    def check_daily_loss_limit(self):
        """Check if daily loss limit has been hit (fixed EUR amount)."""
        nlv = self.ib.get_net_liquidation()
        if nlv <= 0:
            return True  # Can't check, allow

        # Reset daily tracking if new day
        today = datetime.now().strftime('%Y-%m-%d')
        if self.state.get('last_trade_date') != today:
            self.state['daily_start_equity'] = nlv
            self.state['daily_pnl'] = 0.0
            self.state['trade_count_today'] = 0
            self.state['last_trade_date'] = today

        start_equity = self.state.get('daily_start_equity', nlv)
        daily_loss = start_equity - nlv  # Positive = loss

        if daily_loss > MAX_DAILY_LOSS_EUR:
            log.warning(f"DAILY LOSS LIMIT HIT: -{daily_loss:,.0f} EUR "
                        f"(limit: -{MAX_DAILY_LOSS_EUR:,.0f} EUR). "
                        f"No more trades today.")
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
        Fixed: always 1 unit max.
        Logs margin info for reference.
        """
        if price <= 0:
            return 0

        margin_per_unit = price * (CFD_MARGIN_PCT / 100.0)
        size = 1  # ALWAYS 1 unit

        notional = price
        log.info(f"{asset_name}: Size=1 unit | "
                 f"Margin={margin_per_unit:,.0f} EUR | "
                 f"Notional={notional:,.0f} EUR | "
                 f"Max loss at {STOP_LOSS_PCT}% stop={notional * STOP_LOSS_PCT / 100:,.0f} EUR")

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
# DASHBOARD EXPORT (writes JSON for live HTML dashboard)
# ============================================================
def export_dashboard_data(state, last_signal=None, last_confidence=None,
                          last_price=None, last_action=None):
    """
    Export current state to JSON for the live HTML dashboard.
    Called after each trading cycle.
    """
    try:
        dashboard = {
            'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'budget': MAX_BUDGET_EUR,
            'margin_pct': CFD_MARGIN_PCT,
            'stop_loss_pct': STOP_LOSS_PCT,
            'daily_loss_limit': MAX_DAILY_LOSS_EUR,
        }

        # Current position
        positions = state.get('positions', {})
        if 'SP500' in positions:
            pos = positions['SP500']
            pnl_eur = 0
            pnl_pct = 0
            if last_price and pos['entry_price'] > 0:
                if pos['side'] == 'LONG':
                    pnl_eur = (last_price - pos['entry_price']) * pos['size']
                    pnl_pct = (last_price / pos['entry_price'] - 1) * 100
                else:
                    pnl_eur = (pos['entry_price'] - last_price) * pos['size']
                    pnl_pct = (pos['entry_price'] / last_price - 1) * 100

            dashboard['position'] = {
                'active': True,
                'side': pos['side'],
                'size': pos['size'],
                'entry_price': pos['entry_price'],
                'stop_price': pos['stop_price'],
                'entry_time': pos['entry_time'],
                'current_price': last_price or 0,
                'pnl_eur': round(pnl_eur, 2),
                'pnl_pct': round(pnl_pct, 3),
            }
        else:
            dashboard['position'] = {'active': False}

        # Last signal
        dashboard['last_signal'] = {
            'signal': last_signal or 'N/A',
            'confidence': last_confidence or 0,
            'price': last_price or 0,
            'action': last_action or 'N/A',
        }

        # Daily stats
        dashboard['daily'] = {
            'trades_today': state.get('trade_count_today', 0),
            'daily_pnl': state.get('daily_pnl', 0),
            'start_equity': state.get('daily_start_equity', 0),
        }

        # Trade history (last 50 trades)
        trades = []
        if os.path.exists(TRADE_LOG_FILE):
            try:
                df = pd.read_csv(TRADE_LOG_FILE)
                for _, row in df.tail(50).iterrows():
                    trades.append({
                        'timestamp': str(row['timestamp']),
                        'action': str(row['action']),
                        'asset': str(row['asset']),
                        'price': float(row['price']),
                        'size': int(row['size']),
                        'confidence': float(row.get('confidence', 0)),
                        'reason': str(row.get('reason', '')),
                    })
            except Exception:
                pass
        dashboard['trades'] = trades

        # Price history with signals (last 200 hours from CSV)
        data_file = INSTRUMENTS['SP500']['data_file']
        price_data = []
        if os.path.exists(data_file):
            try:
                df = pd.read_csv(data_file)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.tail(200)
                for _, row in df.iterrows():
                    price_data.append({
                        'dt': row['datetime'].strftime('%Y-%m-%d %H:%M'),
                        'o': round(float(row['open']), 2),
                        'h': round(float(row['high']), 2),
                        'l': round(float(row['low']), 2),
                        'c': round(float(row['close']), 2),
                    })
            except Exception:
                pass
        dashboard['prices'] = price_data

        # Mark trade entries/exits on the price chart
        trade_markers = []
        if os.path.exists(TRADE_LOG_FILE):
            try:
                df = pd.read_csv(TRADE_LOG_FILE)
                for _, row in df.iterrows():
                    trade_markers.append({
                        'dt': str(row['timestamp'])[:16],
                        'action': str(row['action']),
                        'price': float(row['price']),
                        'reason': str(row.get('reason', '')),
                    })
            except Exception:
                pass
        dashboard['trade_markers'] = trade_markers

        # Ensure output directory exists
        os.makedirs(os.path.dirname(DASHBOARD_FILE), exist_ok=True)
        with open(DASHBOARD_FILE, 'w') as f:
            json.dump(dashboard, f, indent=2)
        log.info(f"Dashboard data exported to {DASHBOARD_FILE}")

    except Exception as e:
        log.warning(f"Dashboard export failed: {e}")


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
    log.info("  SP500 HOURLY TRADING CYCLE")
    log.info(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Budget: {MAX_BUDGET_EUR:,.0f} EUR | Margin: {CFD_MARGIN_PCT}%")
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

    # Step 1: Update SP500 data from IB
    log.info("\n--- Updating SP500 data from IB ---")
    try:
        update_market_data(ib_conn)
    except Exception as e:
        log.error(f"Data update failed: {e}")
        log.info("Continuing with existing data...")

    # Step 2: Load Broly config
    config = load_setup_config()
    if config is None:
        log.error("Cannot load setup config. Run daily_setup.py first!")
        if own_connection:
            ib_conn.disconnect()
        return False

    log.info(f"Config: {config['n_features']} V2 features, "
             f"{config['prediction_horizon']}h prediction")

    # Step 3: Generate signals and execute
    log.info("\n--- Generating signals & executing ---")
    results = {}
    last_signal = last_confidence = last_price = last_action = None

    for asset_name in INSTRUMENTS:
        try:
            signal, confidence, price = get_current_signal(asset_name, config)
            if signal is None:
                log.warning(f"{asset_name}: No signal generated")
                results[asset_name] = 'NO_SIGNAL'
                continue

            action = executor.process_signal(asset_name, signal, confidence, price)
            results[asset_name] = action

            # Capture for dashboard
            last_signal = signal
            last_confidence = confidence
            last_price = price
            last_action = action

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

    # Export dashboard data for live HTML
    export_dashboard_data(state, last_signal, last_confidence,
                          last_price, last_action)

    if own_connection:
        ib_conn.disconnect()

    return True


def run_continuous_loop():
    """
    Run trading cycle every time a new hourly candle is available.
    SP500 CFD trades ~23h/day: Sun 23:00 to Fri 22:00 UTC.
    """
    log.info("=" * 60)
    log.info("  STARTING CONTINUOUS SP500 TEST TRADING")
    log.info(f"  Press Ctrl+C to stop")
    log.info("=" * 60)

    ib_conn = IBConnection()
    if not ib_conn.connect():
        return

    try:
        while True:
            now = datetime.utcnow()
            weekday = now.weekday()
            hour = now.hour

            # Check if market is open
            is_open = True
            reason = ''

            if weekday == 5:  # Saturday
                is_open = False
                reason = 'Saturday — market closed'
            elif weekday == 6 and hour < 23:  # Sunday before open
                is_open = False
                reason = f'Sunday {hour}:00 UTC — opens 23:00'
            elif weekday == 4 and hour >= 22:  # Friday after close
                is_open = False
                reason = 'Friday post-close'
            elif weekday < 4 and hour == 22:  # Daily break
                is_open = False
                reason = 'Daily maintenance (22-23 UTC)'

            if is_open:
                try:
                    if not ib_conn.connected:
                        ib_conn.connect()

                    # Bootstrap: if CSV doesn't exist, download 30 days first
                    data_file = INSTRUMENTS['SP500']['data_file']
                    if not os.path.exists(data_file):
                        log.info("No SP500 data — bootstrapping from IB...")
                        update_market_data(ib_conn)
                        if not os.path.exists(data_file):
                            continue

                    log.info("\n--- Waiting for next hourly candle from IB ---")
                    candle_ready = wait_for_new_candle(ib_conn)

                    if candle_ready:
                        run_trading_cycle(ib_conn)

                except Exception as e:
                    log.error(f"Trading cycle error: {e}")
                    try:
                        ib_conn.disconnect()
                    except:
                        pass
                    ib_conn = IBConnection()
                    ib_conn.connect()
            else:
                log.info(f"Market closed ({reason}). Sleeping 10 min...")
                time.sleep(600)

    except KeyboardInterrupt:
        log.info("\nStopping... (Ctrl+C)")
    finally:
        ib_conn.disconnect()
        log.info("Trader stopped.")


def show_status():
    """Show current positions and P&L."""
    print("=" * 60)
    print("  IB SP500 TEST STATUS")
    print("=" * 60)

    print(f"\n  Budget: {MAX_BUDGET_EUR:,.0f} EUR | "
          f"Margin: {CFD_MARGIN_PCT}% | "
          f"Stop-loss: {STOP_LOSS_PCT}% | "
          f"Daily loss limit: {MAX_DAILY_LOSS_EUR:,.0f} EUR")

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
