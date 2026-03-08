"""
crypto_correlation_analysis.py — BTC vs ETH Signal Correlation
================================================================
Analyzes whether BTC and ETH signals overlap (redundant) or diverge
(diversification opportunity). Tests:
  1. Signal agreement rate (how often both say BUY/SELL at same time)
  2. Entry/exit timing overlap
  3. Combined portfolio performance (BTC+ETH vs BTC-only vs ETH-only)
  4. Return correlation

Usage:
  python crypto_correlation_analysis.py
"""

import os, sys, time
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime

from crypto_trading_system import (
    load_data, build_all_features, download_asset, update_all_data,
    ALL_MODELS, TRADING_FEE, _build_features,
)
from crypto_live_trader import (
    load_best_config, compute_combined_signal,
)
from crypto_revolut_trader import compute_asset_signal
from hardware_config import N_JOBS_PARALLEL
try:
    from hardware_config import MACHINE
except ImportError:
    MACHINE = 'UNKNOWN'
from sklearn.preprocessing import StandardScaler


def generate_signals_for_asset(asset, horizon, n_hours=900):
    """Generate raw signals for one asset+horizon."""
    config = load_best_config(asset, horizon=horizon)
    if not config:
        return []

    model_names = config['models'].split('+')
    window = config['best_window']
    opt = config.get('optimal_features', '')
    if opt and pd.notna(opt) and str(opt).strip() and str(opt).strip() != 'nan':
        feature_override = [f.strip() for f in str(opt).split(',') if f.strip() and f.strip() != 'nan']
    else:
        feature_override = None

    df_features, feature_cols = _build_features(
        load_data(asset), asset, feature_override=feature_override, horizon=horizon
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

        if len(np.unique(y_train)) < 2 or X_train.isnull().any().any() or X_test.isnull().any().any():
            continue

        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
        X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

        votes, probas = [], []
        for mn in model_names:
            try:
                model = ALL_MODELS[mn]()
                model.fit(X_train_s, y_train)
                votes.append(model.predict(X_test_s)[0])
                probas.append(model.predict_proba(X_test_s)[0][1])
            except Exception:
                continue

        if not votes:
            continue

        buy_ratio = sum(votes) / len(votes)
        if buy_ratio > 0.5:
            signal = 'BUY'
        elif buy_ratio == 0:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        avg_proba = np.mean(probas)
        confidence = avg_proba * 100 if signal != 'SELL' else (1 - avg_proba) * 100

        signals.append({
            'datetime': str(row['datetime']),
            'close': float(row['close']),
            'signal': signal,
            'confidence': round(float(confidence), 1),
        })

    return signals


def compute_strategy_signals(signals_4h, signals_8h, strategy):
    """Apply strategy to get combined signal per hour."""
    map_4h = {s['datetime']: s for s in signals_4h}
    map_8h = {s['datetime']: s for s in signals_8h}
    all_dts = sorted(set(map_4h.keys()) | set(map_8h.keys()))

    combined = []
    for dt in all_dts:
        s4 = map_4h.get(dt)
        s8 = map_8h.get(dt)
        action, conf, reason = compute_asset_signal(s4, s8, strategy)
        price = (s4 or s8 or {}).get('close', 0)
        combined.append({
            'datetime': dt, 'signal': action, 'confidence': conf,
            'close': price, 'reason': reason,
        })
    return combined


def simulate_portfolio(signals, initial=1000):
    """Simple BUY/SELL portfolio sim with fees."""
    cash = initial
    held = 0
    position = 'cash'
    start_price = signals[0]['close'] if signals else 0

    values = []
    for s in signals:
        price = s['close']
        if s['signal'] == 'BUY' and position == 'cash':
            held = cash * (1 - TRADING_FEE) / price
            cash = 0
            position = 'invested'
        elif s['signal'] == 'SELL' and position == 'invested':
            cash = held * price * (1 - TRADING_FEE)
            held = 0
            position = 'cash'

        val = held * price if position == 'invested' else cash
        bh = initial * price / start_price if start_price else initial
        values.append({'datetime': s['datetime'], 'value': val, 'bh': bh, 'signal': s['signal']})

    return values


def main():
    print("=" * 70)
    print("  BTC vs ETH SIGNAL CORRELATION ANALYSIS")
    print(f"  {MACHINE} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Update data
    print("\n  Updating data...")
    update_all_data(['BTC', 'ETH'])

    # Generate signals for both assets (all 4 models)
    n_hours = 720  # 1 month

    print(f"\n  Generating BTC signals ({n_hours}h)...")
    t0 = time.time()
    btc_4h = generate_signals_for_asset('BTC', 4, n_hours)
    btc_8h = generate_signals_for_asset('BTC', 8, n_hours)
    print(f"  BTC: {len(btc_4h)} 4h + {len(btc_8h)} 8h signals ({(time.time()-t0)/60:.1f} min)")

    print(f"\n  Generating ETH signals ({n_hours}h)...")
    t0 = time.time()
    eth_4h = generate_signals_for_asset('ETH', 4, n_hours)
    eth_8h = generate_signals_for_asset('ETH', 8, n_hours)
    print(f"  ETH: {len(eth_4h)} 4h + {len(eth_8h)} 8h signals ({(time.time()-t0)/60:.1f} min)")

    # Apply per-asset strategies
    print("\n  Applying strategies...")
    btc_combined = compute_strategy_signals(btc_4h, btc_8h, 'both_agree')
    eth_combined = compute_strategy_signals(eth_4h, eth_8h, 'either')

    # ================================================================
    # 1. RAW SIGNAL AGREEMENT
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  1. RAW SIGNAL AGREEMENT (per hour)")
    print(f"{'='*70}")

    btc_map = {s['datetime']: s['signal'] for s in btc_combined}
    eth_map = {s['datetime']: s['signal'] for s in eth_combined}
    common_dts = sorted(set(btc_map.keys()) & set(eth_map.keys()))

    agree = 0
    both_buy = 0
    both_sell = 0
    both_hold = 0
    btc_buy_eth_not = 0
    eth_buy_btc_not = 0
    btc_sell_eth_not = 0
    eth_sell_btc_not = 0

    for dt in common_dts:
        b = btc_map[dt]
        e = eth_map[dt]
        if b == e:
            agree += 1
            if b == 'BUY': both_buy += 1
            elif b == 'SELL': both_sell += 1
            else: both_hold += 1
        else:
            if b == 'BUY' and e != 'BUY': btc_buy_eth_not += 1
            if e == 'BUY' and b != 'BUY': eth_buy_btc_not += 1
            if b == 'SELL' and e != 'SELL': btc_sell_eth_not += 1
            if e == 'SELL' and b != 'SELL': eth_sell_btc_not += 1

    total = len(common_dts)
    print(f"\n  Total common hours: {total}")
    print(f"\n  Agreement: {agree}/{total} ({agree/total*100:.1f}%)")
    print(f"    Both BUY:  {both_buy} ({both_buy/total*100:.1f}%)")
    print(f"    Both SELL: {both_sell} ({both_sell/total*100:.1f}%)")
    print(f"    Both HOLD: {both_hold} ({both_hold/total*100:.1f}%)")
    print(f"\n  Disagreement: {total-agree}/{total} ({(total-agree)/total*100:.1f}%)")
    print(f"    BTC BUY, ETH not:  {btc_buy_eth_not}")
    print(f"    ETH BUY, BTC not:  {eth_buy_btc_not}")
    print(f"    BTC SELL, ETH not: {btc_sell_eth_not}")
    print(f"    ETH SELL, BTC not: {eth_sell_btc_not}")

    # ================================================================
    # 2. TRADE OVERLAP ANALYSIS
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  2. TRADE TIMING OVERLAP")
    print(f"{'='*70}")

    # Find BUY periods for each
    def get_invested_periods(signals):
        periods = []
        in_trade = False
        entry_dt = None
        for s in signals:
            if s['signal'] == 'BUY' and not in_trade:
                in_trade = True
                entry_dt = s['datetime']
            elif s['signal'] == 'SELL' and in_trade:
                periods.append((entry_dt, s['datetime']))
                in_trade = False
        return periods

    btc_periods = get_invested_periods(btc_combined)
    eth_periods = get_invested_periods(eth_combined)

    print(f"\n  BTC trades: {len(btc_periods)}")
    print(f"  ETH trades: {len(eth_periods)}")

    # Check overlap
    btc_invested_hours = set()
    for start, end in btc_periods:
        for s in btc_combined:
            if start <= s['datetime'] <= end:
                btc_invested_hours.add(s['datetime'])

    eth_invested_hours = set()
    for start, end in eth_periods:
        for s in eth_combined:
            if start <= s['datetime'] <= end:
                eth_invested_hours.add(s['datetime'])

    overlap = btc_invested_hours & eth_invested_hours
    btc_only = btc_invested_hours - eth_invested_hours
    eth_only = eth_invested_hours - btc_invested_hours
    either = btc_invested_hours | eth_invested_hours

    print(f"\n  Hours invested:")
    print(f"    BTC only:      {len(btc_only):4d} hours")
    print(f"    ETH only:      {len(eth_only):4d} hours")
    print(f"    Both at once:  {len(overlap):4d} hours")
    print(f"    Either:        {len(either):4d} hours")
    print(f"    Neither:       {total - len(either):4d} hours")

    if len(either) > 0:
        overlap_pct = len(overlap) / len(either) * 100
        print(f"\n  Overlap rate: {overlap_pct:.1f}% (of invested time)")
        if overlap_pct > 70:
            print(f"  ⚠️  HIGH OVERLAP — signals mostly redundant")
        elif overlap_pct > 40:
            print(f"  📊 MODERATE OVERLAP — some diversification")
        else:
            print(f"  ✅ LOW OVERLAP — good diversification!")

    # ================================================================
    # 3. PORTFOLIO COMPARISON
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  3. PORTFOLIO COMPARISON ($1000 each)")
    print(f"{'='*70}")

    btc_portfolio = simulate_portfolio(btc_combined)
    eth_portfolio = simulate_portfolio(eth_combined)

    if btc_portfolio and eth_portfolio:
        btc_ret = (btc_portfolio[-1]['value'] - 1000) / 1000 * 100
        eth_ret = (eth_portfolio[-1]['value'] - 1000) / 1000 * 100
        btc_bh = (btc_portfolio[-1]['bh'] - 1000) / 1000 * 100
        eth_bh = (eth_portfolio[-1]['bh'] - 1000) / 1000 * 100

        # Combined: $500 BTC + $500 ETH
        btc_half = simulate_portfolio(btc_combined, initial=500)
        eth_half = simulate_portfolio(eth_combined, initial=500)

        # Merge by datetime for combined portfolio
        btc_val_map = {v['datetime']: v['value'] for v in btc_half}
        eth_val_map = {v['datetime']: v['value'] for v in eth_half}
        common = sorted(set(btc_val_map.keys()) & set(eth_val_map.keys()))

        if common:
            combined_final = btc_val_map[common[-1]] + eth_val_map[common[-1]]
            combined_ret = (combined_final - 1000) / 1000 * 100

            # Combined buy & hold
            btc_bh_map = {v['datetime']: v['bh'] for v in btc_half}
            eth_bh_map = {v['datetime']: v['bh'] for v in eth_half}
            combined_bh = btc_bh_map[common[-1]] + eth_bh_map[common[-1]]
            combined_bh_ret = (combined_bh - 1000) / 1000 * 100

            print(f"\n  {'Portfolio':<25s} {'Return':>8s} {'B&H':>8s} {'Alpha':>8s}")
            print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
            print(f"  {'BTC only ($1000)':<25s} {btc_ret:+7.1f}% {btc_bh:+7.1f}% {btc_ret-btc_bh:+7.1f}%")
            print(f"  {'ETH only ($1000)':<25s} {eth_ret:+7.1f}% {eth_bh:+7.1f}% {eth_ret-eth_bh:+7.1f}%")
            print(f"  {'50/50 BTC+ETH ($1000)':<25s} {combined_ret:+7.1f}% {combined_bh_ret:+7.1f}% {combined_ret-combined_bh_ret:+7.1f}%")

    # ================================================================
    # 4. HOURLY RETURN CORRELATION
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  4. PRICE RETURN CORRELATION")
    print(f"{'='*70}")

    btc_prices = {s['datetime']: s['close'] for s in btc_combined}
    eth_prices = {s['datetime']: s['close'] for s in eth_combined}
    common_price_dts = sorted(set(btc_prices.keys()) & set(eth_prices.keys()))

    if len(common_price_dts) > 10:
        btc_rets = []
        eth_rets = []
        for i in range(1, len(common_price_dts)):
            dt_prev = common_price_dts[i-1]
            dt_curr = common_price_dts[i]
            btc_r = (btc_prices[dt_curr] - btc_prices[dt_prev]) / btc_prices[dt_prev]
            eth_r = (eth_prices[dt_curr] - eth_prices[dt_prev]) / eth_prices[dt_prev]
            btc_rets.append(btc_r)
            eth_rets.append(eth_r)

        corr = np.corrcoef(btc_rets, eth_rets)[0, 1]
        print(f"\n  Hourly return correlation: {corr:.3f}")

        if corr > 0.8:
            print(f"  ⚠️  VERY HIGH — prices move together, limited diversification")
        elif corr > 0.6:
            print(f"  📊 HIGH — prices often move together")
        elif corr > 0.4:
            print(f"  ✅ MODERATE — some independent movement")
        else:
            print(f"  ✅ LOW — good diversification potential")

        # Check if signals are less correlated than prices
        btc_sig_num = [1 if btc_map.get(dt) == 'BUY' else -1 if btc_map.get(dt) == 'SELL' else 0 for dt in common_dts]
        eth_sig_num = [1 if eth_map.get(dt) == 'BUY' else -1 if eth_map.get(dt) == 'SELL' else 0 for dt in common_dts]

        if len(btc_sig_num) > 10:
            sig_corr = np.corrcoef(btc_sig_num, eth_sig_num)[0, 1]
            print(f"  Signal correlation: {sig_corr:.3f}")

            if sig_corr < corr:
                print(f"  ✅ Signals less correlated than prices ({sig_corr:.2f} < {corr:.2f})")
                print(f"     → ML models find asset-specific patterns!")
            else:
                print(f"  ⚠️  Signals more correlated than prices")

    # ================================================================
    # 5. WHEN ONE IS INVESTED AND OTHER IS NOT
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  5. UNIQUE OPPORTUNITIES")
    print(f"{'='*70}")

    # Count hours where only ETH is invested (BTC in cash)
    eth_unique_returns = []
    for dt in eth_only:
        if dt in eth_prices:
            idx = common_price_dts.index(dt) if dt in common_price_dts else -1
            if idx > 0:
                prev_dt = common_price_dts[idx - 1]
                if prev_dt in eth_prices:
                    ret = (eth_prices[dt] - eth_prices[prev_dt]) / eth_prices[prev_dt] * 100
                    eth_unique_returns.append(ret)

    btc_unique_returns = []
    for dt in btc_only:
        if dt in btc_prices:
            idx = common_price_dts.index(dt) if dt in common_price_dts else -1
            if idx > 0:
                prev_dt = common_price_dts[idx - 1]
                if prev_dt in btc_prices:
                    ret = (btc_prices[dt] - btc_prices[prev_dt]) / btc_prices[prev_dt] * 100
                    btc_unique_returns.append(ret)

    if eth_unique_returns:
        print(f"\n  ETH-only invested hours: {len(eth_only)}")
        print(f"    Avg hourly return: {np.mean(eth_unique_returns):+.3f}%")
        print(f"    Total unique return: {sum(eth_unique_returns):+.2f}%")

    if btc_unique_returns:
        print(f"\n  BTC-only invested hours: {len(btc_only)}")
        print(f"    Avg hourly return: {np.mean(btc_unique_returns):+.3f}%")
        print(f"    Total unique return: {sum(btc_unique_returns):+.2f}%")

    # ================================================================
    # VERDICT
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    if len(either) > 0:
        overlap_pct = len(overlap) / len(either) * 100
    else:
        overlap_pct = 0

    print(f"\n  Price correlation: {corr:.2f}" if 'corr' in dir() else "")
    print(f"  Signal overlap: {overlap_pct:.0f}%")
    print(f"  Unique ETH opportunities: {len(eth_only)} hours")
    print(f"  Unique BTC opportunities: {len(btc_only)} hours")

    if overlap_pct < 50 and len(eth_only) > 50:
        print(f"\n  ✅ TRADE BOTH — significant diversification value")
        print(f"     ETH catches opportunities BTC misses")
    elif overlap_pct < 70:
        print(f"\n  📊 MODERATE VALUE — some diversification, trade both with smaller ETH size")
    else:
        print(f"\n  ⚠️  LIMITED VALUE — signals too correlated, consider trading only the stronger model")

    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
