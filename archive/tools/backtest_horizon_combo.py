"""
Backtest horizon combo strategies since Wednesday 2026-03-25.
Compares single-horizon vs multi-horizon combos for BTC, ETH, SOL.
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime

from crypto_trading_system_doohan import (
    generate_signals, simulate_portfolio, load_data,
    TRADING_FEE, MIN_CONFIDENCE, _suppress_stderr
)

# --- Config ---
ASSETS = ['BTC', 'ETH', 'SOL']
START_DATE = datetime(2026, 3, 25, 0, 0)  # Wednesday
REPLAY_HOURS = 120  # ~5 days of data to cover since Wednesday

# Load production models
PROD_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'models', 'crypto_doohan_v1_7_1_production.csv')
df_models = pd.read_csv(PROD_CSV)

def get_signals_for_horizon(asset, horizon):
    """Generate signals for a given asset+horizon using production config."""
    rows = df_models[(df_models['coin'] == asset) & (df_models['horizon'] == horizon)]
    if len(rows) == 0:
        return {}
    row = rows.iloc[0]
    feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
    gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0
    models = row['models'].split('+')
    window = int(row['best_window'])

    sigs = generate_signals(asset, models, window, REPLAY_HOURS,
                            feature_override=feats, horizon=horizon, gamma=gamma)
    sigs = simulate_portfolio(sigs)
    # Index by datetime
    result = {}
    for s in sigs:
        dt = s['datetime']
        if isinstance(dt, str):
            dt = datetime.strptime(dt, '%Y-%m-%d %H:%M')
            s['datetime'] = dt
        if dt >= START_DATE:
            result[dt] = s
    return result


def simulate_strategy(sig_maps, horizons, strategy, conf_threshold):
    """Simulate a strategy over the merged timeline."""
    all_dts = set()
    for sm in sig_maps.values():
        all_dts.update(sm.keys())
    all_times = sorted(dt for dt in all_dts if dt >= START_DATE)

    if not all_times:
        return None

    h_short = min(horizons)
    h_long = max(horizons)
    has_short = h_short in sig_maps and len(sig_maps[h_short]) > 0
    has_long = h_long in sig_maps and len(sig_maps[h_long]) > 0

    cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
    trades, wins = 0, 0
    trade_log = []

    for dt in all_times:
        s_short = sig_maps.get(h_short, {}).get(dt)
        s_long = sig_maps.get(h_long, {}).get(dt)

        if s_short is None and s_long is None:
            continue

        price = (s_short or s_long)['close']
        sig_s = s_short['signal'] if s_short else 'HOLD'
        conf_s = s_short['confidence'] if s_short else 50
        sig_l = s_long['signal'] if s_long else 'HOLD'
        conf_l = s_long['confidence'] if s_long else 50

        # Determine signal based on strategy
        if strategy == 'both_agree':
            if sig_s == 'SELL' or sig_l == 'SELL':
                signal = 'SELL'
            elif sig_s == 'BUY' and sig_l == 'BUY' and conf_s >= conf_threshold and conf_l >= conf_threshold:
                signal = 'BUY'
            else:
                signal = 'HOLD'
        elif strategy == 'either_agree':
            if sig_s == 'SELL' or sig_l == 'SELL':
                signal = 'SELL'
            elif (sig_s == 'BUY' and conf_s >= conf_threshold) or (sig_l == 'BUY' and conf_l >= conf_threshold):
                signal = 'BUY'
            else:
                signal = 'HOLD'
        elif 'only' in strategy:
            h = int(strategy.replace('h_only', ''))
            sm = sig_maps.get(h, {})
            s = sm.get(dt)
            if s is None:
                signal = 'HOLD'
            else:
                sig = s['signal']
                conf = s['confidence']
                signal = sig if conf >= conf_threshold or sig == 'SELL' else 'HOLD'
        else:
            signal = 'HOLD'

        # Execute
        if signal == 'BUY' and not in_pos:
            held = cash * (1 - TRADING_FEE) / price
            cash = 0
            in_pos = True
            entry_px = price
            trades += 1
            trade_log.append((dt.strftime('%m-%d %H:%M'), 'BUY', f'${entry_px:.2f}'))
        elif signal == 'SELL' and in_pos:
            cash = held * price * (1 - TRADING_FEE)
            if price > entry_px:
                wins += 1
            trade_log.append((dt.strftime('%m-%d %H:%M'), 'SELL', f'${price:.2f}', f'{"WIN" if price > entry_px else "LOSS"}'))
            held = 0
            in_pos = False

    # Close open position at end
    if in_pos and all_times:
        last_dt = all_times[-1]
        last_px = None
        for sm in sig_maps.values():
            if last_dt in sm:
                last_px = sm[last_dt]['close']
                break
        if last_px:
            cash = held * last_px * (1 - TRADING_FEE)
            if last_px > entry_px:
                wins += 1
            trade_log.append((last_dt.strftime('%m-%d %H:%M'), 'CLOSE', f'${last_px:.2f}', f'{"WIN" if last_px > entry_px else "LOSS"}'))
            trades_display = trades  # don't count forced close as trade
        in_pos = False

    ret = (cash / 1000.0 - 1) * 100
    wr = (wins / trades * 100) if trades > 0 else 0
    fees_paid = trades * 2 * TRADING_FEE * 100  # approx % lost to fees (round trips)
    return {
        'return': ret, 'trades': trades, 'win_rate': wr,
        'fees_approx': fees_paid, 'trade_log': trade_log
    }


def main():
    for asset in ASSETS:
        print(f"\n{'='*70}")
        print(f"  {asset}: Horizon Combo Backtest (since Wed 2026-03-25)")
        print(f"{'='*70}")

        # Get available horizons from production CSV
        asset_horizons = sorted(df_models[df_models['coin'] == asset]['horizon'].unique())
        print(f"  Available horizons: {asset_horizons}")

        # Generate signals for each horizon
        sig_maps = {}
        with _suppress_stderr():
            for h in asset_horizons:
                print(f"  Generating {h}h signals...")
                sig_maps[h] = get_signals_for_horizon(asset, h)
                print(f"    -> {len(sig_maps[h])} signals since Wednesday")

        # Test strategies
        # Single horizons
        strategies = []
        for h in asset_horizons:
            strategies.append((f'{h}h_only', [h, h]))

        # All 2-horizon combos
        for i, h1 in enumerate(asset_horizons):
            for h2 in asset_horizons[i+1:]:
                strategies.append((f'both_{h1}+{h2}h', [h1, h2]))
                strategies.append((f'either_{h1}+{h2}h', [h1, h2]))

        # Test at multiple confidence levels
        for conf in [70, 80, 90]:
            print(f"\n  --- Confidence >= {conf}% ---")
            print(f"  {'Strategy':<22} {'Return':>8} {'WR':>6} {'Trades':>7} {'Fees~':>7}")
            print(f"  {'-'*55}")

            results = []
            for strat_name, h_pair in strategies:
                h_short, h_long = min(h_pair), max(h_pair)
                local_maps = {h: sig_maps.get(h, {}) for h in [h_short, h_long]}

                if 'both_' in strat_name and not strat_name.startswith('both_agree'):
                    strat_type = 'both_agree'
                elif 'either_' in strat_name:
                    strat_type = 'either_agree'
                else:
                    strat_type = strat_name  # Xh_only

                res = simulate_strategy(local_maps, [h_short, h_long], strat_type, conf)
                if res:
                    results.append((strat_name, res))
                    marker = ''
                    print(f"  {strat_name:<22} {res['return']:>+7.2f}% {res['win_rate']:>5.0f}% {res['trades']:>7d} {res['fees_approx']:>6.2f}%{marker}")

            # Sort and show best
            if results:
                results.sort(key=lambda x: -x[1]['return'])
                best_name, best_res = results[0]
                print(f"\n  >>> BEST: {best_name} @ conf>={conf}% -> {best_res['return']:+.2f}%, {best_res['trades']} trades")
                if best_res['trade_log']:
                    print(f"  Trade log:")
                    for t in best_res['trade_log']:
                        print(f"    {' | '.join(str(x) for x in t)}")

        print()


if __name__ == '__main__':
    main()
