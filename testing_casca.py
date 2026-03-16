"""
TESTING: Casca — Multi-Timeframe Fusion Test
============================================================
Combines V5 Cacarot (1h candles) and V15 Cacarot (15-min candles)
into a single decision engine. Tests all cross-timeframe strategies.

4 independent signals per decision point:
  - V5  4h  (hourly candle, 4-hour horizon)
  - V5  8h  (hourly candle, 8-hour horizon)
  - V15 60' (15-min candle, 4-candle horizon = 60 min)
  - V15 120'(15-min candle, 8-candle horizon = 120 min)

Evaluation cadences:
  - Hourly:  all 4 signals checked once per hour
  - 15-min:  V15 signals fresh every 15 min, V5 carried forward

Usage:
  python testing_casca.py              # BTC (default)
  python testing_casca.py ETH          # Test ETH
  python testing_casca.py --15min      # Evaluate on 15-min cadence (default: hourly)

Results: models/testing_casca_results.csv
Charts:  charts/casca_test/
"""

import sys
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import json

# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_ASSET = 'BTC'
V15_GAMMA = 0.997  # gamma for V15 signals (pending V15.1 test results)
REPLAY_HOURS = 400  # enough overlap for meaningful comparison
RESULTS_CSV = 'models/testing_casca_results.csv'
CHARTS_DIR = 'charts/casca_test'

# Trading costs (same as production)
TRADING_FEE = 0.0011
MIN_CONFIDENCE = 75  # default, swept in threshold phase

# ============================================================
# Import both systems
# ============================================================
import importlib
v5 = importlib.import_module('crypto_trading_system')
v15 = importlib.import_module('crypto_trading_system_v15')


# ============================================================
# STRATEGY DEFINITIONS
# ============================================================
# Each strategy is a function(sig_v5_4h, conf_v5_4h, sig_v5_8h, conf_v5_8h,
#                              sig_v15_4c, conf_v15_4c, sig_v15_8c, conf_v15_8c,
#                              min_conf) -> signal
# Returns 'BUY', 'SELL', or 'HOLD'

def _any_sell(*sigs):
    """True if any signal is SELL."""
    return any(s == 'SELL' for s in sigs)


STRATEGIES = {}

def _register(name):
    def decorator(fn):
        STRATEGIES[name] = fn
        return fn
    return decorator


# --- V5-only baselines ---
@_register('v5_both_agree')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    if s54 == 'SELL' or s58 == 'SELL': return 'SELL'
    if s54 == 'BUY' and s58 == 'BUY' and c54 >= mc and c58 >= mc: return 'BUY'
    return 'HOLD'

@_register('v5_either_agree')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    if s54 == 'SELL' or s58 == 'SELL': return 'SELL'
    if (s54 == 'BUY' and c54 >= mc) or (s58 == 'BUY' and c58 >= mc): return 'BUY'
    return 'HOLD'

@_register('v5_4h_only')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    if s54 == 'SELL': return 'SELL'
    if s54 == 'BUY' and c54 >= mc: return 'BUY'
    return 'HOLD'

@_register('v5_8h_only')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    if s58 == 'SELL': return 'SELL'
    if s58 == 'BUY' and c58 >= mc: return 'BUY'
    return 'HOLD'

# --- V15-only baselines ---
@_register('v15_60m_only')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    if s15_4 == 'SELL': return 'SELL'
    if s15_4 == 'BUY' and c15_4 >= mc: return 'BUY'
    return 'HOLD'

@_register('v15_120m_only')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    if s15_8 == 'SELL': return 'SELL'
    if s15_8 == 'BUY' and c15_8 >= mc: return 'BUY'
    return 'HOLD'

@_register('v15_both_agree')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    if s15_4 == 'SELL' or s15_8 == 'SELL': return 'SELL'
    if s15_4 == 'BUY' and s15_8 == 'BUY' and c15_4 >= mc and c15_8 >= mc: return 'BUY'
    return 'HOLD'

@_register('v15_either_agree')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    if s15_4 == 'SELL' or s15_8 == 'SELL': return 'SELL'
    if (s15_4 == 'BUY' and c15_4 >= mc) or (s15_8 == 'BUY' and c15_8 >= mc): return 'BUY'
    return 'HOLD'

# --- Cross-timeframe strategies ---
@_register('all_4_agree')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    """All 4 signals must say BUY. Any SELL triggers SELL."""
    if _any_sell(s54, s58, s15_4, s15_8): return 'SELL'
    if (s54 == 'BUY' and s58 == 'BUY' and s15_4 == 'BUY' and s15_8 == 'BUY'
            and c54 >= mc and c58 >= mc and c15_4 >= mc and c15_8 >= mc):
        return 'BUY'
    return 'HOLD'

@_register('any_4_buy')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    """Any of 4 signals says BUY. Any SELL triggers SELL."""
    if _any_sell(s54, s58, s15_4, s15_8): return 'SELL'
    if ((s54 == 'BUY' and c54 >= mc) or (s58 == 'BUY' and c58 >= mc) or
            (s15_4 == 'BUY' and c15_4 >= mc) or (s15_8 == 'BUY' and c15_8 >= mc)):
        return 'BUY'
    return 'HOLD'

@_register('majority_3of4')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    """3 out of 4 must say BUY. Any SELL triggers SELL."""
    if _any_sell(s54, s58, s15_4, s15_8): return 'SELL'
    buys = sum([
        s54 == 'BUY' and c54 >= mc,
        s58 == 'BUY' and c58 >= mc,
        s15_4 == 'BUY' and c15_4 >= mc,
        s15_8 == 'BUY' and c15_8 >= mc,
    ])
    if buys >= 3: return 'BUY'
    return 'HOLD'

@_register('v5_8h_and_v15_120m')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    """Both long horizons agree (V5 8h + V15 120min)."""
    if s58 == 'SELL' or s15_8 == 'SELL': return 'SELL'
    if s58 == 'BUY' and s15_8 == 'BUY' and c58 >= mc and c15_8 >= mc: return 'BUY'
    return 'HOLD'

@_register('v5_8h_or_v15_120m')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    """Either long horizon (V5 8h OR V15 120min)."""
    if s58 == 'SELL' or s15_8 == 'SELL': return 'SELL'
    if (s58 == 'BUY' and c58 >= mc) or (s15_8 == 'BUY' and c15_8 >= mc): return 'BUY'
    return 'HOLD'

@_register('v5_4h_and_v15_60m')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    """Both short horizons agree (V5 4h + V15 60min)."""
    if s54 == 'SELL' or s15_4 == 'SELL': return 'SELL'
    if s54 == 'BUY' and s15_4 == 'BUY' and c54 >= mc and c15_4 >= mc: return 'BUY'
    return 'HOLD'

@_register('v5_8h_confirmed_v15')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    """V5 8h makes the call, V15 must confirm (either V15 horizon)."""
    if s58 == 'SELL': return 'SELL'
    if (s58 == 'BUY' and c58 >= mc and
            ((s15_4 == 'BUY' and c15_4 >= mc) or (s15_8 == 'BUY' and c15_8 >= mc))):
        return 'BUY'
    return 'HOLD'

@_register('v15_fast_entry_v5_exit')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    """V15 triggers BUY (fast entry), V5 triggers SELL (slow exit)."""
    if s54 == 'SELL' or s58 == 'SELL': return 'SELL'
    if (s15_4 == 'BUY' and c15_4 >= mc) or (s15_8 == 'BUY' and c15_8 >= mc): return 'BUY'
    return 'HOLD'

@_register('v5_long_v15_short_agree')
def _(s54, c54, s58, c58, s15_4, c15_4, s15_8, c15_8, mc):
    """V5 8h (trend) + V15 60min (timing) must agree."""
    if s58 == 'SELL' or s15_4 == 'SELL': return 'SELL'
    if s58 == 'BUY' and s15_4 == 'BUY' and c58 >= mc and c15_4 >= mc: return 'BUY'
    return 'HOLD'


# ============================================================
# SIGNAL GENERATION
# ============================================================
def _load_v5_signals(asset):
    """Generate V5 Cacarot signals for 4h and 8h."""
    csv_path = v5._get_models_csv_path()
    if not os.path.exists(csv_path):
        print("  ERROR: No V5 models CSV found.")
        return None, None

    df = pd.read_csv(csv_path)
    signals_4h, signals_8h = None, None

    for horizon in [4, 8]:
        cfg = df[(df['coin'] == asset) & (df['horizon'] == horizon)]
        if cfg.empty:
            print(f"  WARNING: No V5 {horizon}h model for {asset}")
            continue

        row = cfg.iloc[0]
        feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
        gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0

        print(f"  Generating V5 {horizon}h signals (gamma={gamma})...")
        sigs = v5.generate_signals(asset, row['models'].split('+'),
                                   int(row['best_window']), REPLAY_HOURS,
                                   feature_override=feats, horizon=horizon, gamma=gamma)
        if horizon == 4:
            signals_4h = sigs
        else:
            signals_8h = sigs

    return signals_4h, signals_8h


def _load_v15_signals(asset):
    """Generate V15 Cacarot signals for 4-candle (60') and 8-candle (120')."""
    csv_path = f'{v15.MODELS_DIR}/crypto_15m_best_models.csv'
    if not os.path.exists(csv_path):
        print("  ERROR: No V15 models CSV found. Run V15 Mode D first.")
        return None, None

    df = pd.read_csv(csv_path)
    signals_4c, signals_8c = None, None

    # V15 replay: convert hours to 15-min candles
    replay_candles = REPLAY_HOURS * 4  # 400h = 1600 candles

    for horizon in [4, 8]:
        cfg = df[(df['coin'] == asset) & (df['horizon'] == horizon)]
        if cfg.empty:
            print(f"  WARNING: No V15 horizon-{horizon} model for {asset}")
            continue

        row = cfg.iloc[0]
        feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
        gamma = float(row.get('gamma', V15_GAMMA)) if pd.notna(row.get('gamma', V15_GAMMA)) else V15_GAMMA

        label = v15._horizon_label(horizon)
        print(f"  Generating V15 {label} signals (gamma={gamma})...")
        sigs = v15.generate_signals(asset, row['models'].split('+'),
                                    int(row['best_window']), replay_candles,
                                    feature_override=feats, horizon=horizon, gamma=gamma)
        if horizon == 4:
            signals_4c = sigs
        else:
            signals_8c = sigs

    return signals_4c, signals_8c


# ============================================================
# MERGE SIGNALS ON COMMON TIMELINE
# ============================================================
def _merge_signals_hourly(v5_4h, v5_8h, v15_4c, v15_8c):
    """
    Merge all 4 signal streams on hourly timestamps.
    V15 signals are snapped to the nearest hour (use the :00 candle).
    """
    # Build lookup maps
    map_v5_4h = {s['datetime']: s for s in (v5_4h or [])}
    map_v5_8h = {s['datetime']: s for s in (v5_8h or [])}

    # V15 signals: keep only the :00 candles (top of hour)
    map_v15_4c = {}
    for s in (v15_4c or []):
        dt = pd.Timestamp(s['datetime'])
        if dt.minute == 0:
            map_v15_4c[s['datetime']] = s

    map_v15_8c = {}
    for s in (v15_8c or []):
        dt = pd.Timestamp(s['datetime'])
        if dt.minute == 0:
            map_v15_8c[s['datetime']] = s

    # Common hourly timestamps (where at least V5 has data)
    v5_times = set(list(map_v5_4h.keys()) + list(map_v5_8h.keys()))
    all_times = sorted(v5_times)

    merged = []
    for dt in all_times:
        s54 = map_v5_4h.get(dt)
        s58 = map_v5_8h.get(dt)
        s15_4 = map_v15_4c.get(dt)
        s15_8 = map_v15_8c.get(dt)

        price = (s54 or s58 or s15_4 or s15_8)['close']

        merged.append({
            'datetime': dt,
            'close': price,
            'v5_4h_sig': s54['signal'] if s54 else 'HOLD',
            'v5_4h_conf': s54['confidence'] if s54 else 50,
            'v5_8h_sig': s58['signal'] if s58 else 'HOLD',
            'v5_8h_conf': s58['confidence'] if s58 else 50,
            'v15_4c_sig': s15_4['signal'] if s15_4 else 'HOLD',
            'v15_4c_conf': s15_4['confidence'] if s15_4 else 50,
            'v15_8c_sig': s15_8['signal'] if s15_8 else 'HOLD',
            'v15_8c_conf': s15_8['confidence'] if s15_8 else 50,
        })

    return merged


def _merge_signals_15min(v5_4h, v5_8h, v15_4c, v15_8c):
    """
    Merge all 4 signal streams on 15-min timestamps.
    V5 signals are carried forward (latest hourly signal applies until next hour).
    """
    map_v5_4h = {s['datetime']: s for s in (v5_4h or [])}
    map_v5_8h = {s['datetime']: s for s in (v5_8h or [])}
    map_v15_4c = {s['datetime']: s for s in (v15_4c or [])}
    map_v15_8c = {s['datetime']: s for s in (v15_8c or [])}

    # Use V15 timestamps as the cadence
    v15_times = sorted(set(list(map_v15_4c.keys()) + list(map_v15_8c.keys())))

    if not v15_times:
        return []

    # Carry-forward state for V5 signals
    last_v5_4h = {'signal': 'HOLD', 'confidence': 50}
    last_v5_8h = {'signal': 'HOLD', 'confidence': 50}

    merged = []
    for dt in v15_times:
        # Update V5 signals if we have a new hourly reading
        if dt in map_v5_4h:
            last_v5_4h = map_v5_4h[dt]
        if dt in map_v5_8h:
            last_v5_8h = map_v5_8h[dt]

        s15_4 = map_v15_4c.get(dt)
        s15_8 = map_v15_8c.get(dt)

        price = (s15_4 or s15_8)['close']

        merged.append({
            'datetime': dt,
            'close': price,
            'v5_4h_sig': last_v5_4h['signal'] if isinstance(last_v5_4h, dict) and 'signal' in last_v5_4h else last_v5_4h.get('signal', 'HOLD'),
            'v5_4h_conf': last_v5_4h['confidence'] if isinstance(last_v5_4h, dict) and 'confidence' in last_v5_4h else 50,
            'v5_8h_sig': last_v5_8h['signal'] if isinstance(last_v5_8h, dict) and 'signal' in last_v5_8h else last_v5_8h.get('signal', 'HOLD'),
            'v5_8h_conf': last_v5_8h['confidence'] if isinstance(last_v5_8h, dict) and 'confidence' in last_v5_8h else 50,
            'v15_4c_sig': s15_4['signal'] if s15_4 else 'HOLD',
            'v15_4c_conf': s15_4['confidence'] if s15_4 else 50,
            'v15_8c_sig': s15_8['signal'] if s15_8 else 'HOLD',
            'v15_8c_conf': s15_8['confidence'] if s15_8 else 50,
        })

    return merged


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest_strategy(merged_signals, strategy_name, min_conf):
    """Run a single strategy over merged signals. Returns result dict."""
    strat_fn = STRATEGIES[strategy_name]
    cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
    trades, wins = 0, 0

    for m in merged_signals:
        signal = strat_fn(
            m['v5_4h_sig'], m['v5_4h_conf'],
            m['v5_8h_sig'], m['v5_8h_conf'],
            m['v15_4c_sig'], m['v15_4c_conf'],
            m['v15_8c_sig'], m['v15_8c_conf'],
            min_conf
        )

        if signal == 'BUY' and not in_pos:
            held = cash * (1 - TRADING_FEE) / m['close']
            cash = 0
            in_pos = True
            entry_px = m['close']
            trades += 1
        elif signal == 'SELL' and in_pos:
            cash = held * m['close'] * (1 - TRADING_FEE)
            if m['close'] > entry_px:
                wins += 1
            held = 0
            in_pos = False

    # Close open position at end
    if in_pos and merged_signals:
        cash = held * merged_signals[-1]['close'] * (1 - TRADING_FEE)

    cum_ret = (cash / 1000.0 - 1) * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0
    # Buy & hold
    if merged_signals:
        bh_ret = (merged_signals[-1]['close'] / merged_signals[0]['close'] - 1) * 100
    else:
        bh_ret = 0

    return {
        'strategy': strategy_name,
        'min_conf': min_conf,
        'return_pct': round(cum_ret, 2),
        'trades': trades,
        'win_rate': round(win_rate, 1),
        'buy_hold_pct': round(bh_ret, 2),
        'alpha': round(cum_ret - bh_ret, 2),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    asset = DEFAULT_ASSET
    cadence = 'hourly'

    for arg in sys.argv[1:]:
        if arg.upper() in ('BTC', 'ETH', 'XRP', 'DOGE'):
            asset = arg.upper()
        elif arg == '--15min':
            cadence = '15min'

    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("=" * 70)
    print("  CASCA: Multi-Timeframe Fusion Test")
    print(f"  Asset: {asset}")
    print(f"  Cadence: {cadence}")
    print(f"  V15 gamma: {V15_GAMMA}")
    print(f"  Replay: {REPLAY_HOURS}h")
    print(f"  Strategies: {len(STRATEGIES)}")
    print("=" * 70)

    t0 = time.time()

    # Step 1: Generate all signals
    print("\n--- Generating V5 Cacarot signals ---")
    v5_4h, v5_8h = _load_v5_signals(asset)

    print("\n--- Generating V15 Cacarot signals ---")
    v15_4c, v15_8c = _load_v15_signals(asset)

    # Check what we have
    n_v5_4h = len(v5_4h) if v5_4h else 0
    n_v5_8h = len(v5_8h) if v5_8h else 0
    n_v15_4c = len(v15_4c) if v15_4c else 0
    n_v15_8c = len(v15_8c) if v15_8c else 0

    print(f"\n  Signal counts: V5 4h={n_v5_4h}, V5 8h={n_v5_8h}, "
          f"V15 60'={n_v15_4c}, V15 120'={n_v15_8c}")

    if n_v5_4h == 0 and n_v5_8h == 0:
        print("  ERROR: No V5 signals. Run V5 Mode D first.")
        return
    if n_v15_4c == 0 and n_v15_8c == 0:
        print("  ERROR: No V15 signals. Run V15 Mode D first.")
        return

    # Step 2: Merge signals
    print(f"\n--- Merging signals ({cadence} cadence) ---")
    if cadence == '15min':
        merged = _merge_signals_15min(v5_4h, v5_8h, v15_4c, v15_8c)
    else:
        merged = _merge_signals_hourly(v5_4h, v5_8h, v15_4c, v15_8c)

    print(f"  Merged timeline: {len(merged)} data points")

    if len(merged) < 10:
        print("  ERROR: Not enough overlapping data points.")
        return

    # Step 3: Test all strategies with confidence sweep
    print(f"\n--- Testing {len(STRATEGIES)} strategies ---")
    confidence_levels = [55, 60, 65, 70, 75, 80, 85, 90]

    all_results = []
    for strat_name in STRATEGIES:
        best_result = None
        best_return = -999

        for mc in confidence_levels:
            result = backtest_strategy(merged, strat_name, mc)
            result['cadence'] = cadence
            result['asset'] = asset
            result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M')
            result['n_datapoints'] = len(merged)

            if result['return_pct'] > best_return:
                best_return = result['return_pct']
                best_result = result

        all_results.append(best_result)

    # Sort by return
    all_results.sort(key=lambda x: x['return_pct'], reverse=True)

    # Step 4: Print results table
    print(f"\n{'='*90}")
    print(f"  CASCA RESULTS — {asset} ({cadence} cadence, {len(merged)} points)")
    print(f"{'='*90}")
    print(f"  {'Strategy':<28} {'Conf%':<7} {'Return%':<10} {'Trades':<8} {'WinRate':<9} {'B&H%':<9} {'Alpha':<8}")
    print("-" * 90)

    for r in all_results:
        marker = ' ***' if r['return_pct'] == all_results[0]['return_pct'] else ''
        print(f"  {r['strategy']:<28} {r['min_conf']:<7} {r['return_pct']:<+10.1f} "
              f"{r['trades']:<8} {r['win_rate']:<9.1f} {r['buy_hold_pct']:<+9.1f} "
              f"{r['alpha']:<+8.1f}{marker}")

    print(f"{'='*90}")

    # Highlight
    best = all_results[0]
    print(f"\n  BEST: {best['strategy']} @{best['min_conf']}% → "
          f"{best['return_pct']:+.1f}% return, {best['trades']} trades, "
          f"alpha={best['alpha']:+.1f}% vs B&H")

    # Is fusion better than V5-only?
    v5_baselines = [r for r in all_results if r['strategy'].startswith('v5_')]
    if v5_baselines:
        best_v5 = max(v5_baselines, key=lambda x: x['return_pct'])
        cross_strats = [r for r in all_results if not r['strategy'].startswith('v5_') and not r['strategy'].startswith('v15_')]
        if cross_strats:
            best_cross = max(cross_strats, key=lambda x: x['return_pct'])
            delta = best_cross['return_pct'] - best_v5['return_pct']
            print(f"\n  V5-only best: {best_v5['strategy']} → {best_v5['return_pct']:+.1f}%")
            print(f"  Cross-TF best: {best_cross['strategy']} → {best_cross['return_pct']:+.1f}%")
            print(f"  Fusion advantage: {delta:+.1f}% {'(BETTER)' if delta > 0 else '(WORSE)'}")

    # Step 5: Save results
    df_results = pd.DataFrame(all_results)
    if os.path.exists(RESULTS_CSV):
        df_existing = pd.read_csv(RESULTS_CSV)
        # Remove old results for same asset+cadence
        mask = (df_existing['asset'] == asset) & (df_existing['cadence'] == cadence)
        df_existing = df_existing[~mask]
        df_results = pd.concat([df_existing, df_results], ignore_index=True)
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"\n  Results saved to {RESULTS_CSV}")

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
