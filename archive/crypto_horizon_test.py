"""
crypto_horizon_test.py - Test Alternative Prediction Horizons
==============================================================
Tests 2h and 8h (or any custom) prediction horizons using the full
Mode D pipeline (feature analysis + diagnostic).

This is a TEST SCRIPT — results are shown but NOT saved to the main
CSV unless you explicitly choose to. Does not modify crypto_trading_system.py.

Usage:
    python crypto_horizon_test.py                          # default: BTC, horizons 2,8
    python crypto_horizon_test.py --horizons 2,3,6,8,12    # custom horizons
    python crypto_horizon_test.py --asset ETH              # other assets
    python crypto_horizon_test.py --skip-analysis          # reuse features if available
    python crypto_horizon_test.py --years 2                # 2 years of data
"""

import argparse
import sys
import os
import time

# Suppress all warnings before any sklearn/joblib imports
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')
import numpy as np
import pandas as pd
from itertools import combinations
from joblib import Parallel, delayed

# Import from main system
from crypto_trading_system import (
    load_data, build_all_features, download_asset, update_all_data,
    run_feature_analysis, run_diagnostic_for_asset,
    ASSETS, TRADING_FEE, REPLAY_HOURS, DIAG_WINDOWS, DIAG_STEP,
    _eval_one_config, _suppress_stderr,
)
from hardware_config import N_JOBS_PARALLEL, get_all_models, LGBM_DEVICE
try:
    from hardware_config import MACHINE
except ImportError:
    MACHINE = 'UNKNOWN'


def run_horizon_test(asset_name, horizon, years=1, skip_analysis=False):
    """Run full Mode D pipeline for a specific prediction horizon."""
    print(f"\n{'='*60}")
    print(f"  HORIZON TEST: {asset_name} — {horizon}h ahead")
    print(f"{'='*60}")

    t_total = time.time()

    # --- Build features with this horizon ---
    print(f"\n  Building features (horizon={horizon}h)...")
    df_raw = load_data(asset_name)
    if df_raw is None:
        print("  ERROR: Could not load data.")
        return None

    df_full, all_cols = build_all_features(df_raw, asset_name=asset_name, horizon=horizon)
    feature_cols = [c for c in all_cols if c not in ('datetime', 'close', 'high', 'low', 'volume', 'label')]

    # Trim to requested years
    n_rows = years * 8760
    if len(df_full) > n_rows:
        df_full = df_full.iloc[-n_rows:].reset_index(drop=True)
        print(f"  Trimmed to last {years}y: {len(df_full):,} rows")

    print(f"  Clean data: {len(df_full):,} rows, {len(feature_cols)} features")

    # --- Feature analysis ---
    if not skip_analysis:
        optimal_features = run_feature_analysis(asset_name, df_full, feature_cols)
    else:
        # Try loading from saved CSV
        csv_path = f'models/horizon_test_{asset_name}_{horizon}h_features.csv'
        if os.path.exists(csv_path):
            saved = pd.read_csv(csv_path)
            optimal_features = saved['feature'].tolist()
            print(f"  Loaded {len(optimal_features)} features from {csv_path}")
        else:
            print(f"  No saved features for {horizon}h, running analysis...")
            optimal_features = run_feature_analysis(asset_name, df_full, feature_cols)

    if not optimal_features:
        print("  ERROR: Feature analysis returned no features!")
        return None

    print(f"\n  Feature analysis: {(time.time() - t_total)/60:.1f} min | {len(optimal_features)} features selected")

    # --- Diagnostic ---
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC: {asset_name} ({len(optimal_features)} features, {horizon}h horizon)")
    print(f"{'='*60}")
    print(f"  {len(df_full):,} rows, {len(optimal_features)} features")

    best_config = run_diagnostic_for_asset(asset_name, df_full, optimal_features)

    if best_config:
        best_config['horizon'] = horizon
        best_config['n_features'] = len(optimal_features)
        best_config['optimal_features'] = ','.join(optimal_features)

    total_time = (time.time() - t_total) / 60
    print(f"\n  {horizon}h horizon total: {total_time:.1f} min")

    return best_config


def main():
    parser = argparse.ArgumentParser(description='Test alternative prediction horizons')
    parser.add_argument('--asset', default='BTC', help='Asset to test (default: BTC)')
    parser.add_argument('--horizons', default='2,8', help='Comma-separated horizons to test (default: 2,8)')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip feature analysis if saved')
    parser.add_argument('--years', type=int, default=1, help='Years of data (default: 1)')
    args = parser.parse_args()

    asset_name = args.asset.upper()
    horizons = [int(h) for h in args.horizons.split(',')]

    print("=" * 60)
    print(f"  HORIZON TEST: {asset_name}")
    print(f"  Horizons: {', '.join(str(h)+'h' for h in horizons)}")
    print(f"  Machine: {MACHINE} | Workers: {N_JOBS_PARALLEL} | LGBM: {LGBM_DEVICE}")
    print(f"  Data: last {args.years} year(s)")
    print("=" * 60)

    # Update data once
    print("\n  Updating data...")
    update_all_data([asset_name])

    # Run each horizon
    results = {}
    for horizon in horizons:
        config = run_horizon_test(asset_name, horizon, years=args.years,
                                  skip_analysis=args.skip_analysis)
        if config:
            results[horizon] = config

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  HORIZON COMPARISON: {asset_name}")
    print(f"{'='*60}")

    if not results:
        print("  No results! All horizons failed.")
        return

    # Include known baselines for comparison
    print(f"\n  {'Horizon':<10s} {'Models':<22s} {'Window':<8s} {'Acc':>6s} {'Return':>9s} {'WinRate':>8s} {'Trades':>7s} {'Score':>8s}")
    print(f"  {'-'*10} {'-'*22} {'-'*8} {'-'*6} {'-'*9} {'-'*8} {'-'*7} {'-'*8}")

    # Print known baselines
    print(f"  {'1h (ref)':<10s} {'—':<22s} {'—':<8s} {'~52%':>6s} {'neg':>9s} {'—':>8s} {'—':>7s} {'~0.50':>8s}")
    print(f"  {'4h (ref)':<10s} {'RF+LR+LGBM':<22s} {'100h':<8s} {'72.5%':>6s} {'+59.6%':>9s} {'68.6%':>8s} {'35':>7s} {'1.157':>8s}")

    for h in sorted(results.keys()):
        c = results[h]
        print(f"  {str(h)+'h':<10s} {c['best_combo']:<22s} {str(c['best_window'])+'h':<8s} "
              f"{c['accuracy']:5.1f}% {c['return_pct']:+8.1f}% {c['win_rate']:7.0f}% "
              f"{c['trades']:6d}  {c['combined_score']:.3f}")

    # Verdict
    print()
    for h in sorted(results.keys()):
        c = results[h]
        if c['accuracy'] >= 65:
            verdict = "PROMISING — worth exploring"
        elif c['accuracy'] >= 60:
            verdict = "MARGINAL — needs more testing"
        elif c['accuracy'] >= 55:
            verdict = "WEAK — probably not tradeable"
        else:
            verdict = "NO EDGE — skip"
        print(f"  {h}h: {verdict} ({c['accuracy']:.1f}%)")

    # Save option
    print(f"\n  Save results? (does NOT overwrite main CSV)")
    for h in sorted(results.keys()):
        c = results[h]
        resp = input(f"  Save {h}h model to main CSV? (y/n) [n]: ").strip().lower()
        if resp == 'y':
            _save_to_csv(asset_name, c, h)
            print(f"    Saved {h}h model!")

    print(f"\n{'='*60}")
    print(f"  HORIZON TEST COMPLETE")
    print(f"{'='*60}")


def _save_to_csv(asset_name, config, horizon):
    """Save a horizon result to the main best models CSV."""
    csv_path = 'models/crypto_hourly_best_models.csv'

    row = {
        'coin': asset_name,
        'best_window': config['best_window'],
        'best_combo': config['best_combo'],
        'accuracy': config['accuracy'],
        'models': config['best_combo'],
        'return_pct': config['return_pct'],
        'win_rate': config['win_rate'],
        'trades': config['trades'],
        'combined_score': config['combined_score'],
        'feature_set': 'D',
        'horizon': horizon,
        'n_features': config.get('n_features', ''),
        'optimal_features': config.get('optimal_features', ''),
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'horizon' not in df.columns:
            df['horizon'] = 4
        # Remove existing entry for this asset+horizon
        df = df[~((df['coin'] == asset_name) & (df['horizon'] == horizon))]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(csv_path, index=False)
    # Also save to mode_d backup
    df.to_csv('models/crypto_hourly_best_models_mode_d.csv', index=False)


if __name__ == '__main__':
    main()
