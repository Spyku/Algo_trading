"""
Testing Deku 1.3 — Horizon Sweep
============================================================
Tests all horizons 1h through 8h with Deku 1.2.1's validation framework
(3-fold rolling holdout, embargo, fee-aware labels), then compares
all pair combinations via strategy comparison.

Goal: determine if the legacy 4h+8h pairing is optimal, or if a
different horizon pair generalizes better under proper validation.

Results saved to: models/testing_deku_1_3_horizon_results.csv
Strategy results:  models/testing_deku_1_3_strategy_results.csv

CLI Usage:
  python testing_deku_1_3_horizons.py BTC           # All 8 horizons + strategies
  python testing_deku_1_3_horizons.py BTC --trials 100  # Custom trial count
  python testing_deku_1_3_horizons.py BTC,ETH       # Multiple assets
  python testing_deku_1_3_horizons.py BTC --horizons 1,2,3,4,5,6,7,8  # Explicit
  python testing_deku_1_3_horizons.py BTC --horizons 3,4,5,6  # Subset

Does NOT touch any production files. Fully isolated.
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations

# ── Import from Deku production ──
# We reuse all core functions but write to isolated output files
import crypto_trading_system_deku as deku

# ── Isolated output paths ──
MODELS_DIR = 'models'
RESULTS_CSV = f'{MODELS_DIR}/testing_deku_1_3_horizon_results.csv'
STRATEGY_CSV = f'{MODELS_DIR}/testing_deku_1_3_strategy_results.csv'
BEST_MODELS_CSV = f'{MODELS_DIR}/testing_deku_1_3_best_models.csv'

# Override Deku 1.2.1's CSV path so Mode D writes to our isolated file
_orig_get_csv = deku._get_models_csv_path
def _isolated_csv_path():
    return BEST_MODELS_CSV
deku._get_models_csv_path = _isolated_csv_path

# Also override backup to use our file
_orig_backup = deku._backup_models_csv
def _isolated_backup():
    src = BEST_MODELS_CSV
    if os.path.exists(src):
        bak = src.replace('.csv', '_backup.csv')
        import shutil
        shutil.copy2(src, bak)
deku._backup_models_csv = _isolated_backup


def run_horizon_sweep(assets_list, horizons=None, n_trials=100):
    """Run Mode D (Optuna) for each horizon, collect results."""
    if horizons is None:
        horizons = list(range(1, 9))  # 1h through 8h

    print("=" * 70)
    print(f"  DEKU 1.3 HORIZON SWEEP")
    print(f"  Assets: {', '.join(assets_list)}")
    print(f"  Horizons: {', '.join(str(h)+'h' for h in horizons)}")
    print(f"  Trials: {n_trials} per horizon")
    print(f"  Validation: 3-fold rolling holdout + 4-candle embargo")
    print(f"  Features: 4h range [4-40], 8h+ range [4-80]")
    print("=" * 70)

    all_results = []
    t_start = time.time()

    for h in horizons:
        print(f"\n{'#'*70}")
        print(f"  HORIZON {h}h — starting [{datetime.now().strftime('%H:%M:%S')}]")
        print(f"{'#'*70}")

        # Update the feature range for this horizon
        # 1-4h: tighter range (more prone to overfitting)
        # 5-8h: wider range (longer horizon = more signal)
        if h <= 4:
            deku.N_FEATURES_RANGE[h] = (4, 40)
        else:
            deku.N_FEATURES_RANGE[h] = (4, 80)

        t_h = time.time()

        try:
            deku.run_mode_d_optuna(assets_list, horizon=h, n_trials=n_trials)
        except Exception as e:
            print(f"  ERROR on {h}h: {e}")
            continue

        elapsed_h = (time.time() - t_h) / 60
        print(f"\n  Horizon {h}h completed in {elapsed_h:.1f} min")

        # Read results from our isolated CSV
        if os.path.exists(BEST_MODELS_CSV):
            df = pd.read_csv(BEST_MODELS_CSV)
            for asset in assets_list:
                row = df[(df['coin'] == asset) & (df['horizon'] == h)]
                if len(row) > 0:
                    r = row.iloc[0]
                    all_results.append({
                        'asset': asset,
                        'horizon': h,
                        'models': r.get('models', ''),
                        'window': r.get('best_window', 0),
                        'gamma': r.get('gamma', 1.0),
                        'n_features': r.get('n_features', 0),
                        'accuracy': r.get('accuracy', 0),
                        'return_pct': r.get('return_pct', 0),
                        'combined_score': r.get('combined_score', 0),
                        'trades': r.get('trades', 0),
                        'elapsed_min': round(elapsed_h, 1),
                    })

    # Save horizon results
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(RESULTS_CSV, index=False)
        print(f"\n\n{'='*70}")
        print(f"  HORIZON SWEEP RESULTS")
        print(f"{'='*70}")
        print(f"\n  {'H':>3s}  {'Score':>7s}  {'Return':>8s}  {'Acc':>6s}  {'Tr':>4s}  {'F':>3s}  {'Win':>5s}  {'Gamma':>7s}  {'Models'}")
        print(f"  {'-'*80}")
        for _, r in df_results.sort_values(['asset', 'combined_score'], ascending=[True, False]).iterrows():
            print(f"  {r['horizon']:2d}h  {r['combined_score']:7.3f}  {r['return_pct']:+7.1f}%  {r['accuracy']:5.1f}%  {r['trades']:4.0f}  "
                  f"{r['n_features']:3.0f}  {r['window']:4.0f}h  {r['gamma']:.4f}  {r['models']}")

        print(f"\n  Results saved: {RESULTS_CSV}")

    total_elapsed = (time.time() - t_start) / 60
    print(f"\n  Total sweep time: {total_elapsed:.1f} min")

    return all_results


def run_strategy_comparison(assets_list, horizons=None):
    """
    Compare all horizon pair strategies using saved models from the sweep.
    Tests: Xh_only (each horizon solo) + both_agree + either_agree for every pair.
    """
    if horizons is None:
        horizons = list(range(1, 9))

    if not os.path.exists(BEST_MODELS_CSV):
        print("  ERROR: No saved models. Run horizon sweep first.")
        return

    df_models = pd.read_csv(BEST_MODELS_CSV)

    print(f"\n\n{'='*70}")
    print(f"  DEKU 1.3 STRATEGY COMPARISON — ALL HORIZON PAIRS")
    print(f"{'='*70}")

    all_strategy_results = []

    for asset in assets_list:
        print(f"\n{'='*60}")
        print(f"  {asset}: Strategy Comparison")
        print(f"{'='*60}")

        # Find which horizons have models for this asset
        available = sorted(df_models[df_models['coin'] == asset]['horizon'].unique())
        if len(available) == 0:
            print(f"  No models for {asset}. Skipping.")
            continue

        print(f"  Available horizons: {', '.join(str(h)+'h' for h in available)}")

        # Generate signals for all available horizons
        signals_by_h = {}
        with deku._suppress_stderr():
            for h in available:
                cfg = df_models[(df_models['coin'] == asset) & (df_models['horizon'] == h)]
                if len(cfg) == 0:
                    continue
                row = cfg.iloc[0]
                feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
                gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0
                try:
                    sigs = deku.generate_signals(asset, row['models'].split('+'),
                                                  int(row['best_window']), deku.REPLAY_HOURS_F,
                                                  feature_override=feats, horizon=h, gamma=gamma)
                    sigs = deku.simulate_portfolio(sigs)
                    if sigs:
                        signals_by_h[h] = {s['datetime']: s for s in sigs}
                        print(f"    {h}h: {len(sigs)} signals generated")
                except Exception as e:
                    print(f"    {h}h: ERROR — {e}")

        if not signals_by_h:
            print(f"  No signals generated for {asset}.")
            continue

        # Build strategies: solo for each horizon + pairs (both_agree, either_agree)
        strategies_to_test = []

        # Solo strategies
        for h in signals_by_h:
            strategies_to_test.append(('solo', h, None))

        # Pair strategies
        h_list = sorted(signals_by_h.keys())
        for h_short, h_long in combinations(h_list, 2):
            strategies_to_test.append(('both_agree', h_short, h_long))
            strategies_to_test.append(('either_agree', h_short, h_long))

        # Build merged timeline across all horizons
        all_dts = set()
        for sig_map in signals_by_h.values():
            all_dts.update(sig_map.keys())
        all_times = sorted(all_dts)

        if not all_times:
            continue

        # Hold-out: last 33% (matches 1.2.1 Mode F)
        holdout_start = int(len(all_times) * 0.67)
        holdout_times = all_times[holdout_start:]
        print(f"  Signals: {len(all_times)} total, evaluating on last {len(holdout_times)} (hold-out 33%)")

        results = []
        for strat_type, h1, h2 in strategies_to_test:
            if strat_type == 'solo':
                strat_name = f'{h1}h_only'
            else:
                strat_name = f'{strat_type}_{h1}h+{h2}h'

            sig_map_1 = signals_by_h.get(h1, {})
            sig_map_2 = signals_by_h.get(h2, {}) if h2 else {}

            cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
            trades, wins = 0, 0

            for dt in holdout_times:
                s1 = sig_map_1.get(dt)
                s2 = sig_map_2.get(dt) if h2 else None
                price_ref = s1 or s2
                if price_ref is None:
                    continue
                price = price_ref['close']

                sig1 = s1['signal'] if s1 else 'HOLD'
                conf1 = s1['confidence'] if s1 else 50
                sig2 = s2['signal'] if s2 else 'HOLD'
                conf2 = s2['confidence'] if s2 else 50

                if strat_type == 'solo':
                    signal = sig1 if conf1 >= deku.MIN_CONFIDENCE or sig1 == 'SELL' else 'HOLD'
                elif strat_type == 'both_agree':
                    if sig1 == 'SELL' or sig2 == 'SELL':
                        signal = 'SELL'
                    elif sig1 == 'BUY' and sig2 == 'BUY' and conf1 >= deku.MIN_CONFIDENCE and conf2 >= deku.MIN_CONFIDENCE:
                        signal = 'BUY'
                    else:
                        signal = 'HOLD'
                elif strat_type == 'either_agree':
                    if sig1 == 'SELL' or sig2 == 'SELL':
                        signal = 'SELL'
                    elif (sig1 == 'BUY' and conf1 >= deku.MIN_CONFIDENCE) or (sig2 == 'BUY' and conf2 >= deku.MIN_CONFIDENCE):
                        signal = 'BUY'
                    else:
                        signal = 'HOLD'

                if signal == 'BUY' and not in_pos:
                    held = cash * (1 - deku.TRADING_FEE) / price
                    cash = 0; in_pos = True; entry_px = price; trades += 1
                elif signal == 'SELL' and in_pos:
                    cash = held * price * (1 - deku.TRADING_FEE)
                    if price > entry_px: wins += 1
                    held = 0; in_pos = False

            # Close open position
            if in_pos and holdout_times:
                last_dt = holdout_times[-1]
                last_price = None
                for sig_map in signals_by_h.values():
                    if last_dt in sig_map:
                        last_price = sig_map[last_dt]['close']
                        break
                if last_price:
                    cash = held * last_price * (1 - deku.TRADING_FEE)

            cum_ret = (cash / 1000.0 - 1) * 100
            win_rate = (wins / trades * 100) if trades > 0 else 0

            results.append({
                'asset': asset,
                'strategy': strat_name,
                'type': strat_type,
                'h1': h1,
                'h2': h2 if h2 else None,
                'return_pct': round(cum_ret, 2),
                'win_rate': round(win_rate, 1),
                'trades': trades,
            })

        # Sort by return
        results.sort(key=lambda x: -x['return_pct'])

        print(f"\n  {'Rank':>4s}  {'Strategy':<25s}  {'Return':>8s}  {'WinRate':>8s}  {'Trades':>7s}")
        print(f"  {'-'*60}")
        for i, r in enumerate(results[:20], 1):
            marker = " <-- BEST" if i == 1 else ""
            print(f"  {i:4d}  {r['strategy']:<25s}  {r['return_pct']:>+7.1f}%  {r['win_rate']:>7.0f}%  {r['trades']:>7d}{marker}")

        all_strategy_results.extend(results)

    # Save strategy results
    if all_strategy_results:
        df_strat = pd.DataFrame(all_strategy_results)
        df_strat.to_csv(STRATEGY_CSV, index=False)
        print(f"\n  Strategy results saved: {STRATEGY_CSV}")

        # Print summary: best strategy per asset
        print(f"\n{'='*70}")
        print(f"  RECOMMENDATIONS")
        print(f"{'='*70}")
        for asset in assets_list:
            asset_results = [r for r in all_strategy_results if r['asset'] == asset]
            if asset_results:
                best = max(asset_results, key=lambda x: x['return_pct'])
                print(f"  {asset}: {best['strategy']} → {best['return_pct']:+.1f}% ({best['trades']} trades, {best['win_rate']:.0f}% win rate)")

                # Also show best solo
                best_solo = max([r for r in asset_results if r['type'] == 'solo'], key=lambda x: x['return_pct'], default=None)
                if best_solo and best_solo['strategy'] != best['strategy']:
                    print(f"         Best solo: {best_solo['strategy']} → {best_solo['return_pct']:+.1f}%")

    return all_strategy_results


# ============================================================
# CLI
# ============================================================
if __name__ == '__main__':
    args = sys.argv[1:]

    # Parse assets
    assets_list = ['BTC']
    horizons = list(range(1, 9))
    n_trials = 100

    for a in args:
        if a.startswith('--'):
            continue
        if a.upper() in deku.ASSETS:
            assets_list = [a.upper()]
        elif ',' in a and all(x.upper() in deku.ASSETS for x in a.split(',')):
            assets_list = [x.upper() for x in a.split(',')]

    # Parse --horizons
    for i, a in enumerate(args):
        if a == '--horizons' and i + 1 < len(args):
            horizons = [int(h) for h in args[i+1].split(',')]
        elif a == '--trials' and i + 1 < len(args):
            n_trials = int(args[i+1])

    print("=" * 70)
    print("  DEKU 1.3 — HORIZON SWEEP TEST HARNESS")
    print(f"  Assets: {', '.join(assets_list)}")
    print(f"  Horizons: {', '.join(str(h)+'h' for h in horizons)} ({len(horizons)} total)")
    print(f"  Trials: {n_trials} per horizon")
    print(f"  Output: {RESULTS_CSV}")
    print(f"           {STRATEGY_CSV}")
    print("=" * 70)

    # Phase 1: Run Mode D for each horizon
    results = run_horizon_sweep(assets_list, horizons, n_trials)

    # Phase 2: Strategy comparison across all pairs
    if results:
        run_strategy_comparison(assets_list, horizons)

    print(f"\n{'='*70}")
    print(f"  DEKU 1.3 COMPLETE")
    print(f"{'='*70}")
