"""
Combined Strategy Optimizer (v2 — Fast)
========================================
Finds the optimal strategy + confidence threshold using saved Mode D configs.

WHAT CHANGED vs v1:
  - Steps 1+2 (re-running 105 configs each) are REMOVED.
    Mode D already found the best combo+window per horizon — no need to redo it.
  - Signal generation for all (asset × horizon) pairs runs in PARALLEL.
  - Step 3 (strategy × confidence combos) is instant math — 14 combos, not 1,400.
  - Supports multiple assets in one run.

Usage:
  python mock_strategy_optimizer.py                        # BTC + ETH
  python mock_strategy_optimizer.py --asset BTC
  python mock_strategy_optimizer.py --asset BTC,ETH,XRP
  python mock_strategy_optimizer.py --asset BTC --hours 720

What it does:
  1. Load saved Mode D best configs from crypto_hourly_best_models.csv
  2. Generate 4h + 8h signals for each asset IN PARALLEL (joblib)
  3. For each (strategy × confidence):
       - Apply combined strategy to pre-generated signals
       - Simulate portfolio with 0.09% fees
       - Record alpha vs buy & hold
  4. Show results per asset, ranked by alpha

Total combos per asset: 2 strategies × 7 confidence levels = 14 (instant)
Runtime: ~same as generating signals for 1 asset × 2 horizons sequentially,
         but all assets run in parallel — so 3 assets = same time as 1.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

from crypto_trading_system import (
    ASSETS, FEATURE_SET_A, FEATURE_SET_B, TRADING_FEE, MIN_CONFIDENCE,
    MODELS_DIR,
    download_asset, generate_signals, simulate_portfolio,
)

try:
    from hardware_config import N_JOBS_PARALLEL, LGBM_DEVICE
    MACHINE = 'DESKTOP' if N_JOBS_PARALLEL > 20 else 'LAPTOP'
except Exception:
    N_JOBS_PARALLEL = 4
    MACHINE = 'UNKNOWN'


# ============================================================
# CONFIG LOADER
# ============================================================
def load_model_config(asset, horizon):
    """Load saved Mode D config for (asset, horizon) from CSV."""
    csv_path = f'{MODELS_DIR}/crypto_hourly_best_models.csv'
    if not os.path.exists(csv_path):
        print(f'  ERROR: {csv_path} not found. Run Mode D first.')
        return None

    df = pd.read_csv(csv_path)
    if 'horizon' not in df.columns:
        df['horizon'] = 4  # legacy

    match = df[(df['coin'] == asset) & (df['horizon'] == horizon)]
    if match.empty:
        print(f'  WARNING: No {horizon}h config found for {asset}. Run Mode D first.')
        return None

    row = match.iloc[0]
    fs = row.get('feature_set', 'A')
    opt = str(row.get('optimal_features', '')).strip()

    if fs in ('D', 'E2', 'E3') and opt and opt != 'nan':
        features = [f.strip() for f in opt.split(',') if f.strip() and f.strip() != 'nan']
    elif fs == 'B':
        features = list(FEATURE_SET_B)
    else:
        features = list(FEATURE_SET_A)

    return {
        'asset':    asset,
        'horizon':  horizon,
        'models':   str(row['models']),
        'window':   int(row['best_window']),
        'features': features,
        'feature_set': fs,
        'accuracy': float(row['accuracy']),
    }


# ============================================================
# PARALLEL SIGNAL GENERATION WORKER
# ============================================================
def _generate_worker(cfg, n_signals):
    """
    Worker function called in a thread.
    Returns (asset, horizon, signals) or (asset, horizon, None) on failure.
    """
    asset   = cfg['asset']
    horizon = cfg['horizon']
    label   = f'{asset} {horizon}h'
    try:
        model_names = cfg['models'].split('+')
        signals = generate_signals(
            asset, model_names, cfg['window'], n_signals,
            feature_override=cfg['features'], horizon=horizon,
        )
        signals = simulate_portfolio(signals)
        print(f'  ✓ {label}: {len(signals)} signals generated '
              f'({cfg["models"]} w={cfg["window"]}h fs={cfg["feature_set"]})')
        return asset, horizon, signals
    except Exception as e:
        print(f'  ✗ {label}: FAILED — {e}')
        return asset, horizon, None


# ============================================================
# STRATEGY SIMULATION (pure math — instant)
# ============================================================
def simulate_strategy(signals_4h, signals_8h, strategy, min_conf, period_hours):
    """
    Apply combined strategy to pre-generated signals.
    Returns dict with return, buy_hold, alpha, trades, win_rate, avg_pnl.
    """
    sig4 = {s['datetime']: s for s in signals_4h}
    sig8 = {s['datetime']: s for s in signals_8h}
    common = sorted(set(sig4.keys()) & set(sig8.keys()))

    if len(common) > period_hours:
        common = common[-period_hours:]
    if not common:
        return None

    cash      = 1000.0
    held      = 0.0
    in_pos    = False
    entry_px  = 0.0
    trades    = 0
    wins      = 0
    start_px  = sig4[common[0]]['close']

    for dt in common:
        s4    = sig4[dt]
        s8    = sig8[dt]
        price = s4['close']
        sig4v = s4['signal'];  conf4 = s4['confidence']
        sig8v = s8['signal'];  conf8 = s8['confidence']

        # --- Combined signal ---
        if sig4v == 'SELL' or sig8v == 'SELL':
            combined = 'SELL'
        elif strategy == 'both_agree':
            combined = 'BUY' if (sig4v == 'BUY' and sig8v == 'BUY'
                                 and conf4 >= min_conf and conf8 >= min_conf) else 'HOLD'
        else:  # either
            combined = 'BUY' if ((sig4v == 'BUY' and conf4 >= min_conf)
                                 or (sig8v == 'BUY' and conf8 >= min_conf)) else 'HOLD'

        # --- Execute ---
        if combined == 'BUY' and not in_pos:
            held     = cash * (1 - TRADING_FEE) / price
            cash     = 0.0
            in_pos   = True
            entry_px = price
            trades  += 1
        elif combined == 'SELL' and in_pos:
            cash  = held * price * (1 - TRADING_FEE)
            wins += 1 if price > entry_px else 0
            held  = 0.0
            in_pos = False

    # Close open position
    last_px = sig4[common[-1]]['close']
    if in_pos:
        cash  = held * last_px * (1 - TRADING_FEE)
        wins += 1 if last_px > entry_px else 0
        trades += 1

    strat_ret = (cash / 1000 - 1) * 100
    bh_ret    = (last_px / start_px - 1) * 100
    alpha     = strat_ret - bh_ret
    win_rate  = (wins / trades * 100) if trades > 0 else 0
    avg_pnl   = strat_ret / trades if trades > 0 else 0

    return {
        'return':    round(strat_ret, 2),
        'buy_hold':  round(bh_ret, 2),
        'alpha':     round(alpha, 2),
        'trades':    trades,
        'win_rate':  round(win_rate, 1),
        'avg_pnl':   round(avg_pnl, 2),
    }


# ============================================================
# PER-ASSET RESULTS DISPLAY
# ============================================================
def show_results(asset, cfg4, cfg8, results, period_hours):
    strategies = ['both_agree', 'either']
    confs      = [65, 70, 75, 80, 85, 90, 95]

    print(f"\n{'='*72}")
    print(f"  {asset} RESULTS  —  last {period_hours}h ({period_hours//24}d)")
    print(f"  4h: {cfg4['models']} w={cfg4['window']}h ({cfg4['accuracy']:.1f}% diag)")
    print(f"  8h: {cfg8['models']} w={cfg8['window']}h ({cfg8['accuracy']:.1f}% diag)")
    print(f"{'='*72}")

    # Build full table
    rows = []
    for strat in strategies:
        for conf in confs:
            key = (strat, conf)
            if key in results:
                r = results[key]
                rows.append({
                    'strategy': strat,
                    'min_conf': conf,
                    **r,
                })

    if not rows:
        print('  No results.')
        return

    rows.sort(key=lambda x: x['alpha'], reverse=True)

    print(f"\n  {'Rank':>4}  {'Strategy':>12}  {'Conf':>5}  {'Return':>8}  "
          f"{'B&H':>8}  {'Alpha':>8}  {'Trades':>6}  {'WinRate':>7}  {'AvgPnL':>7}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}")

    for rank, r in enumerate(rows, 1):
        marker = ' <<<' if rank == 1 else ''
        print(f"  {rank:>4}  {r['strategy']:>12}  {r['min_conf']:>4}%  "
              f"  {r['return']:>+6.1f}%  {r['buy_hold']:>+6.1f}%  {r['alpha']:>+6.1f}%  "
              f"  {r['trades']:>5}  {r['win_rate']:>6.1f}%  {r['avg_pnl']:>+6.2f}%{marker}")

    best = rows[0]

    # Best per strategy
    print(f"\n  BEST PER STRATEGY:")
    for strat in strategies:
        strat_rows = [r for r in rows if r['strategy'] == strat]
        if strat_rows:
            b = strat_rows[0]
            print(f"    {strat:>12}:  conf={b['min_conf']}%  "
                  f"alpha={b['alpha']:+.1f}%  trades={b['trades']}  "
                  f"win={b['win_rate']:.1f}%")

    # Compare vs current production config
    try:
        cfg_path = 'config/trading_config.json'
        if os.path.exists(cfg_path):
            import json
            with open(cfg_path) as f:
                tcfg = json.load(f)
            prod_strategy = tcfg.get(asset, {}).get('strategy', 'both_agree')
            prod_conf     = MIN_CONFIDENCE  # 75%
            prod_key      = (prod_strategy, prod_conf)
            prod_match    = results.get(prod_key)

            print(f"\n  VS PRODUCTION  ({prod_strategy} @ {prod_conf}%):")
            if prod_match:
                improvement = best['alpha'] - prod_match['alpha']
                print(f"    Current:  alpha={prod_match['alpha']:+.1f}%  "
                      f"trades={prod_match['trades']}  win={prod_match['win_rate']:.1f}%")
                print(f"    Best:     alpha={best['alpha']:+.1f}%  "
                      f"trades={best['trades']}  win={best['win_rate']:.1f}%  "
                      f"[{best['strategy']} @ {best['min_conf']}%]")
                tag = '>>> IMPROVEMENT' if improvement > 0 else '>>> NO IMPROVEMENT'
                print(f"    {tag}: {improvement:+.1f}% alpha")
            else:
                print(f"    Production config ({prod_strategy} @ {prod_conf}%) not in results.")
    except Exception:
        pass

    print(f"\n  RECOMMENDATION FOR {asset}:")
    print(f"    strategy  = '{best['strategy']}'")
    print(f"    min_conf  = {best['min_conf']}%")
    print(f"    alpha     = {best['alpha']:+.1f}%  |  "
          f"trades = {best['trades']}  |  win_rate = {best['win_rate']:.1f}%")


# ============================================================
# MAIN
# ============================================================
def main():
    # --- Parse args ---
    assets       = ['BTC', 'ETH']
    period_hours = 720

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--asset' and i + 1 < len(args):
            assets = [a.strip().upper() for a in args[i + 1].split(',')]
        elif arg == '--hours' and i + 1 < len(args):
            period_hours = int(args[i + 1])

    print('=' * 72)
    print(f'  COMBINED STRATEGY OPTIMIZER v2  —  Fast')
    print(f'  Assets:  {", ".join(assets)}')
    print(f'  Machine: {MACHINE}  |  Workers: {N_JOBS_PARALLEL}')
    print(f'  Period:  last {period_hours}h ({period_hours // 24}d)')
    print(f'  Combos:  2 strategies × 7 confidence levels = 14 per asset (instant)')
    print('=' * 72)

    # --- Load configs ---
    print('\n  Loading Mode D configs from CSV...')
    configs = []
    for asset in assets:
        for h in [4, 8]:
            cfg = load_model_config(asset, h)
            if cfg:
                configs.append(cfg)
                print(f'    {asset} {h}h: {cfg["models"]} w={cfg["window"]}h '
                      f'fs={cfg["feature_set"]} ({len(cfg["features"])} features)')

    if not configs:
        print('\n  ERROR: No configs loaded. Run Mode D first.')
        return

    # --- Update data ---
    print('\n  Updating market data...')
    for asset in assets:
        try:
            download_asset(asset, update_only=True)
        except Exception as e:
            print(f'    WARNING: Could not update {asset}: {e}')

    # --- Generate signals in PARALLEL ---
    # Each (asset, horizon) pair runs in its own thread
    n_workers = min(len(configs), N_JOBS_PARALLEL)
    print(f'\n  Generating signals ({len(configs)} tasks, {n_workers} parallel workers)...')
    t0 = time.time()

    # signals_store[(asset, horizon)] = signals list
    signals_store = {}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_generate_worker, cfg, period_hours): cfg
            for cfg in configs
        }
        for future in as_completed(futures):
            asset, horizon, signals = future.result()
            if signals:
                signals_store[(asset, horizon)] = signals

    elapsed = time.time() - t0
    print(f'\n  Signal generation done in {elapsed/60:.1f} min  '
          f'({len(signals_store)}/{len(configs)} successful)')

    if not signals_store:
        print('\n  ERROR: No signals generated.')
        return

    # --- Step 3: Test strategy combinations (instant math) ---
    strategies = ['both_agree', 'either']
    min_confs  = [65, 70, 75, 80, 85, 90, 95]

    print(f'\n  Running {len(strategies) * len(min_confs)} strategy combos per asset...')

    for asset in assets:
        sigs4 = signals_store.get((asset, 4))
        sigs8 = signals_store.get((asset, 8))
        cfg4  = next((c for c in configs if c['asset'] == asset and c['horizon'] == 4), None)
        cfg8  = next((c for c in configs if c['asset'] == asset and c['horizon'] == 8), None)

        if not sigs4 or not sigs8:
            missing = '4h' if not sigs4 else '8h'
            print(f'\n  SKIP {asset}: missing {missing} signals.')
            continue

        results = {}
        for strategy in strategies:
            for conf in min_confs:
                r = simulate_strategy(sigs4, sigs8, strategy, conf, period_hours)
                if r:
                    results[(strategy, conf)] = r

        show_results(asset, cfg4, cfg8, results, period_hours)

    print(f'\n{"="*72}')
    print(f'  DONE  —  This is a MOCK test, no production files were changed.')
    print(f'{"="*72}')


if __name__ == '__main__':
    main()
