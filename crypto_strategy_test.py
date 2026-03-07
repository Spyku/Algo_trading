"""
crypto_strategy_test.py - BTC Strategy Backtester
===================================================
Tests different signal strategies and hold durations:
  1. Which signal to follow: 4h only, 8h only, or both agreeing
  2. After BUY, sell after N hours (2,3,4,5,6,7,8) vs following model SELL

Runs over past week and past month.

Usage:
    python crypto_strategy_test.py
"""

import sys
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from crypto_trading_system import (
    load_data, build_all_features, download_asset, update_all_data,
    ALL_MODELS, TRADING_FEE, REPLAY_HOURS,
    _build_features,
)
from hardware_config import N_JOBS_PARALLEL, LGBM_DEVICE
try:
    from hardware_config import MACHINE
except ImportError:
    MACHINE = 'UNKNOWN'
from sklearn.preprocessing import StandardScaler


def load_model_config(asset_name, horizon):
    """Load model config from CSV for a given horizon."""
    csv_path = 'models/crypto_hourly_best_models.csv'
    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found!")
        return None
    df = pd.read_csv(csv_path)
    if 'horizon' not in df.columns:
        df['horizon'] = 4
    match = df[(df['coin'] == asset_name) & (df['horizon'] == horizon)]
    if match.empty:
        print(f"  No saved {horizon}h model for {asset_name}")
        return None
    row = match.iloc[0]
    return {
        'models': row['best_combo'].split('+'),
        'window': int(row['best_window']),
        'accuracy': row['accuracy'],
        'feature_set': row.get('feature_set', 'A'),
        'optimal_features': '' if pd.isna(row.get('optimal_features', '')) else str(row.get('optimal_features', '')),
    }


def generate_raw_signals(asset_name, config, horizon, n_hours):
    """Generate hourly signals with actual outcomes for backtesting."""
    model_names = config['models']
    window = config['window']
    opt_features = config['optimal_features']

    # Determine features
    if opt_features and pd.notna(opt_features) and opt_features.strip() and opt_features.strip() != 'nan':
        feature_override = [f.strip() for f in opt_features.split(',') if f.strip() and f.strip() != 'nan']
    else:
        feature_override = None

    df_features, feature_cols = _build_features(
        load_data(asset_name), asset_name, 
        feature_override=feature_override, horizon=horizon
    )
    n = len(df_features)
    start_idx = max(window + 50, n - n_hours)

    signals = []
    for i in range(start_idx, n):
        row = df_features.iloc[i]

        train_start = max(0, i - window)
        train = df_features.iloc[train_start:i]
        X_train = train[feature_cols]
        y_train = train['label'].values
        X_test = df_features.iloc[i:i+1][feature_cols]

        if len(np.unique(y_train)) < 2:
            continue
        if X_train.isnull().any().any() or X_test.isnull().any().any():
            continue

        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
        X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

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
            except Exception:
                continue

        if not votes:
            continue

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

        # Actual price at future horizons
        future_prices = {}
        for fh in [2, 3, 4, 5, 6, 7, 8]:
            if i + fh < n:
                future_prices[fh] = float(df_features.iloc[i + fh]['close'])

        signals.append({
            'idx': i,
            'datetime': row['datetime'],
            'close': float(row['close']),
            'signal': signal,
            'confidence': round(float(confidence), 1),
            'future_prices': future_prices,
        })

    return signals


def simulate_strategy(signals_4h, signals_8h, strategy, hold_hours, period_hours):
    """
    Simulate a trading strategy.
    
    strategy: '4h', '8h', 'both_agree', 'either'
    hold_hours: 'model' (follow model SELL) or int (sell after N hours)
    period_hours: use only last N hours of signals
    """
    # Align signals by datetime
    sig_4h_map = {str(s['datetime']): s for s in signals_4h}
    sig_8h_map = {str(s['datetime']): s for s in signals_8h}

    # Get common datetimes, sorted
    if strategy == '4h':
        all_dts = sorted(sig_4h_map.keys())
    elif strategy == '8h':
        all_dts = sorted(sig_8h_map.keys())
    else:
        common = set(sig_4h_map.keys()) & set(sig_8h_map.keys())
        all_dts = sorted(common)

    # Trim to period
    if len(all_dts) > period_hours:
        all_dts = all_dts[-period_hours:]

    if not all_dts:
        return None

    cash = 1000.0
    position = 'cash'
    buy_price = 0
    buy_time_idx = 0
    trades = []
    hours_held = 0
    start_price = None
    correct = 0
    total_predictions = 0

    for t_idx, dt in enumerate(all_dts):
        s4 = sig_4h_map.get(dt)
        s8 = sig_8h_map.get(dt)

        # Get current price
        if strategy == '4h' and s4:
            price = s4['close']
        elif strategy == '8h' and s8:
            price = s8['close']
        elif s4:
            price = s4['close']
        elif s8:
            price = s8['close']
        else:
            continue

        if start_price is None:
            start_price = price

        # Determine signal based on strategy
        if strategy == '4h':
            sig = s4['signal'] if s4 else 'HOLD'
            conf = s4['confidence'] if s4 else 50
        elif strategy == '8h':
            sig = s8['signal'] if s8 else 'HOLD'
            conf = s8['confidence'] if s8 else 50
        elif strategy == 'both_agree':
            if s4 and s8 and s4['signal'] == s8['signal'] and s4['signal'] != 'HOLD':
                sig = s4['signal']
                conf = (s4['confidence'] + s8['confidence']) / 2
            else:
                sig = 'HOLD'
                conf = 50
        elif strategy == 'either':
            # BUY if either says BUY, SELL if either says SELL
            if (s4 and s4['signal'] == 'BUY') or (s8 and s8['signal'] == 'BUY'):
                sig = 'BUY'
                conf = max(s4['confidence'] if s4 and s4['signal'] == 'BUY' else 0,
                          s8['confidence'] if s8 and s8['signal'] == 'BUY' else 0)
            elif (s4 and s4['signal'] == 'SELL') or (s8 and s8['signal'] == 'SELL'):
                sig = 'SELL'
                conf = max(s4['confidence'] if s4 and s4['signal'] == 'SELL' else 0,
                          s8['confidence'] if s8 and s8['signal'] == 'SELL' else 0)
            else:
                sig = 'HOLD'
                conf = 50

        # Handle timed sell
        should_sell_timed = False
        if position == 'invested' and isinstance(hold_hours, int):
            hours_held += 1
            if hours_held >= hold_hours:
                should_sell_timed = True

        # Execute trades
        if sig == 'BUY' and position == 'cash':
            btc_held = cash * (1 - TRADING_FEE) / price
            cash = 0
            position = 'invested'
            buy_price = price
            hours_held = 0
        elif position == 'invested':
            sell_now = False
            if should_sell_timed:
                sell_now = True
            elif hold_hours == 'model' and sig == 'SELL':
                sell_now = True

            if sell_now:
                cash = btc_held * price * (1 - TRADING_FEE)
                pnl = (price - buy_price) / buy_price * 100
                trades.append({
                    'buy_price': buy_price,
                    'sell_price': price,
                    'pnl_pct': pnl,
                    'hours_held': hours_held,
                })
                if pnl > 0:
                    correct += 1
                total_predictions += 1
                position = 'cash'
                btc_held = 0

    # Close open position at last price
    last_price = None
    for dt in reversed(all_dts):
        s = sig_4h_map.get(dt) or sig_8h_map.get(dt)
        if s:
            last_price = s['close']
            break

    if position == 'invested' and last_price:
        cash = btc_held * last_price * (1 - TRADING_FEE)
        pnl = (last_price - buy_price) / buy_price * 100
        trades.append({'buy_price': buy_price, 'sell_price': last_price,
                       'pnl_pct': pnl, 'hours_held': hours_held})
        if pnl > 0:
            correct += 1
        total_predictions += 1

    final_value = cash
    total_return = (final_value - 1000) / 1000 * 100
    buy_hold_return = (last_price - start_price) / start_price * 100 if start_price and last_price else 0
    alpha = total_return - buy_hold_return
    win_rate = (correct / total_predictions * 100) if total_predictions > 0 else 0
    avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
    avg_hold = np.mean([t['hours_held'] for t in trades]) if trades else 0

    return {
        'return': round(total_return, 2),
        'buy_hold': round(buy_hold_return, 2),
        'alpha': round(alpha, 2),
        'trades': len(trades),
        'win_rate': round(win_rate, 1),
        'avg_pnl': round(avg_pnl, 2),
        'avg_hold_hours': round(avg_hold, 1),
        'final_value': round(final_value, 2),
    }


def main():
    asset = 'BTC'
    # Parse --asset from CLI
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--asset' and i + 1 < len(args):
            asset = args[i + 1].upper()

    print("=" * 70)
    print(f"  STRATEGY BACKTESTER: {asset}")
    print(f"  Machine: {MACHINE} | Workers: {N_JOBS_PARALLEL}")
    print("=" * 70)

    # Load model configs
    config_4h = load_model_config(asset, 4)
    config_8h = load_model_config(asset, 8)

    if not config_4h:
        print("  ERROR: No 4h model found!")
        return
    if not config_8h:
        print("  ERROR: No 8h model found!")
        return

    print(f"\n  4h model: {'+'.join(config_4h['models'])} | w={config_4h['window']}h | {config_4h['accuracy']}%")
    print(f"  8h model: {'+'.join(config_8h['models'])} | w={config_8h['window']}h | {config_8h['accuracy']}%")

    # Update data
    print("\n  Updating data...")
    update_all_data([asset])

    # Generate signals for the past month + buffer
    n_hours = 900  # ~month + buffer for warm-up
    
    print(f"\n  Generating 4h signals ({n_hours} hours)...")
    t0 = time.time()
    signals_4h = generate_raw_signals(asset, config_4h, horizon=4, n_hours=n_hours)
    print(f"  Done: {len(signals_4h)} signals in {(time.time()-t0)/60:.1f} min")

    print(f"\n  Generating 8h signals ({n_hours} hours)...")
    t0 = time.time()
    signals_8h = generate_raw_signals(asset, config_8h, horizon=8, n_hours=n_hours)
    print(f"  Done: {len(signals_8h)} signals in {(time.time()-t0)/60:.1f} min")

    # ================================================================
    # TEST 1: Which signal to follow (4h, 8h, both agree, either)
    # ================================================================
    strategies = ['4h', '8h', 'both_agree', 'either']
    periods = [('Week', 168), ('Month', 720)]

    print(f"\n{'='*70}")
    print(f"  TEST 1: WHICH SIGNAL TO FOLLOW (using model SELL to exit)")
    print(f"{'='*70}")

    for period_name, period_hours in periods:
        print(f"\n  --- Past {period_name} ({period_hours}h) ---")
        print(f"  {'Strategy':<15s} {'Return':>8s} {'B&H':>8s} {'Alpha':>8s} {'Trades':>7s} {'WinRate':>8s} {'AvgPnL':>8s}")
        print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")

        for strat in strategies:
            result = simulate_strategy(signals_4h, signals_8h, strat, 'model', period_hours)
            if result:
                print(f"  {strat:<15s} {result['return']:+7.1f}% {result['buy_hold']:+7.1f}% "
                      f"{result['alpha']:+7.1f}% {result['trades']:>6d} {result['win_rate']:>7.1f}% "
                      f"{result['avg_pnl']:+7.2f}%")
            else:
                print(f"  {strat:<15s}  (no data)")

    # ================================================================
    # TEST 2: Optimal hold duration after BUY
    # ================================================================
    hold_durations = [2, 3, 4, 5, 6, 7, 8, 'model']

    print(f"\n{'='*70}")
    print(f"  TEST 2: HOW LONG TO HOLD AFTER BUY")
    print(f"{'='*70}")

    for strat in strategies:
        for period_name, period_hours in periods:
            print(f"\n  --- {strat.upper()} signal | Past {period_name} ({period_hours}h) ---")
            print(f"  {'Hold':<10s} {'Return':>8s} {'B&H':>8s} {'Alpha':>8s} {'Trades':>7s} {'WinRate':>8s} {'AvgPnL':>8s} {'AvgHold':>8s}")
            print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")

            for hold in hold_durations:
                result = simulate_strategy(signals_4h, signals_8h, strat, hold, period_hours)
                if result:
                    hold_label = f"{hold}h" if isinstance(hold, int) else "Model"
                    print(f"  {hold_label:<10s} {result['return']:+7.1f}% {result['buy_hold']:+7.1f}% "
                          f"{result['alpha']:+7.1f}% {result['trades']:>6d} {result['win_rate']:>7.1f}% "
                          f"{result['avg_pnl']:+7.2f}% {result['avg_hold_hours']:>7.1f}h")

    # ================================================================
    # TEST 3: Confidence filter (only trade when confidence > threshold)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  TEST 3: CONFIDENCE FILTER (8h signal, model exit)")
    print(f"{'='*70}")

    for period_name, period_hours in periods:
        print(f"\n  --- Past {period_name} ({period_hours}h) ---")
        print(f"  {'MinConf':>8s} {'Return':>8s} {'Alpha':>8s} {'Trades':>7s} {'WinRate':>8s}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")

        for min_conf in [50, 60, 70, 75, 80, 85]:
            # Filter signals by confidence
            filtered_8h = [s for s in signals_8h if s['confidence'] >= min_conf or s['signal'] == 'HOLD']
            # For signals below threshold, convert to HOLD
            adjusted_8h = []
            for s in signals_8h:
                s_copy = dict(s)
                if s_copy['confidence'] < min_conf and s_copy['signal'] in ('BUY', 'SELL'):
                    s_copy['signal'] = 'HOLD'
                adjusted_8h.append(s_copy)

            result = simulate_strategy(signals_4h, adjusted_8h, '8h', 'model', period_hours)
            if result:
                print(f"  {min_conf:>7d}% {result['return']:+7.1f}% "
                      f"{result['alpha']:+7.1f}% {result['trades']:>6d} {result['win_rate']:>7.1f}%")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY & RECOMMENDATIONS")
    print(f"{'='*70}")
    print(f"\n  Check the tables above to find:")
    print(f"  1. Best signal source (4h, 8h, or combined)")
    print(f"  2. Optimal hold duration after BUY")
    print(f"  3. Whether confidence filtering improves results")
    print(f"\n  The best strategy is the one with highest ALPHA")
    print(f"  (outperformance vs buy & hold)")
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
