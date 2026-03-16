"""
TESTING: V15.1 — Gamma Optimization for 15-Min System
============================================================
Tests 7 gamma values × 2 horizons (4h, 8h) to find optimal decay for V15.
Each gamma runs a full Mode D pipeline (feature selection + diagnostic).

Gamma values tested: 1.0 (no decay), 0.999, 0.998, 0.997, 0.996, 0.995, 0.994

Results saved to: models/testing_v15.1_results.csv
Charts saved to:  charts/v15.1_test/

Usage:
  python testing_v15.1.py                  # Run all 14 tests (7 gammas × 2 horizons)
  python testing_v15.1.py --resume         # Skip already-completed gamma/horizon combos
  python testing_v15.1.py --asset ETH      # Test ETH instead of BTC (default: BTC)

Based on crypto_trading_system_v15.py (V15 Cacarot).
"""

import sys
import os
import time
from datetime import datetime
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
GAMMAS_TO_TEST = [1.0, 0.999, 0.998, 0.997, 0.996, 0.995, 0.994]
HORIZONS = [4, 8]
DEFAULT_ASSET = 'BTC'

# Isolated output paths — use absolute paths (Google Drive virtual FS needs them)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(_SCRIPT_DIR, 'models', 'testing_v15.1_results.csv')
CHARTS_DIR_TEST = os.path.join(_SCRIPT_DIR, 'charts', 'v15.1_test')

# ============================================================
# Import V15 Cacarot — we'll call its functions directly
# ============================================================
# We need to import from crypto_trading_system_v15 but override some paths
import importlib
v15 = importlib.import_module('crypto_trading_system_v15')


def run_gamma_test(asset, gamma, horizon, resume=False):
    """Run a single Mode D pipeline for one gamma/horizon combo.

    Returns dict with results or None if failed.
    """
    print("\n" + "#" * 70)
    print(f"  GAMMA TEST: {asset} | gamma={gamma} | horizon={v15._horizon_label(horizon)}")
    print("#" * 70)

    t0 = time.time()

    # Override the gamma in the existing config (or create a fresh one)
    # We force gamma by temporarily modifying what _load_mode_d_config returns
    original_load = v15._load_mode_d_config

    def _patched_load(asset_name, h):
        """Return config with forced gamma."""
        result = original_load(asset_name, h)
        if result:
            result['gamma'] = gamma
        else:
            # No existing config — create minimal one so gamma is available
            # Mode D will still run from scratch (all features)
            result = {'gamma': gamma}
        return result

    # Patch the loader
    v15._load_mode_d_config = _patched_load

    # Override charts dir for isolation
    original_charts_dir = v15.CHARTS_DIR
    v15.CHARTS_DIR = CHARTS_DIR_TEST
    os.makedirs(CHARTS_DIR_TEST, exist_ok=True)

    try:
        # Run Mode D for this single asset + horizon
        v15.run_mode_d([asset], horizon=horizon)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        v15._load_mode_d_config = original_load
        v15.CHARTS_DIR = original_charts_dir
        return None
    finally:
        # Restore originals
        v15._load_mode_d_config = original_load
        v15.CHARTS_DIR = original_charts_dir

    elapsed = time.time() - t0

    # Read the result that was just saved to the V15 CSV
    csv_path = f'{v15.MODELS_DIR}/crypto_15m_best_models.csv'
    if not os.path.exists(csv_path):
        print("  No results CSV found!")
        return None

    df = pd.read_csv(csv_path)
    mask = (df['coin'] == asset) & (df['horizon'] == horizon)
    matches = df[mask]

    if matches.empty:
        print("  No matching result found in CSV!")
        return None

    row = matches.iloc[-1]  # Latest result

    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'asset': asset,
        'horizon': horizon,
        'gamma': gamma,
        'best_combo': row['best_combo'],
        'best_window': int(row['best_window']),
        'accuracy': float(row['accuracy']),
        'return_pct': float(row.get('return_pct', 0)),
        'win_rate': float(row.get('win_rate', 0)),
        'trades': int(row.get('trades', 0)),
        'combined_score': float(row.get('combined_score', 0)),
        'n_features': int(row.get('n_features', 0)),
        'elapsed_min': round(elapsed / 60, 1),
    }

    print(f"\n  RESULT: gamma={gamma} | {result['best_combo']} | "
          f"acc={result['accuracy']:.1f}% | ret={result['return_pct']:+.1f}% | "
          f"trades={result['trades']} | {result['elapsed_min']} min")

    return result


def save_result(result):
    """Append a single result to the test CSV."""
    os.makedirs(os.path.join(_SCRIPT_DIR, 'models'), exist_ok=True)
    df_new = pd.DataFrame([result])

    if os.path.exists(RESULTS_CSV):
        df_existing = pd.read_csv(RESULTS_CSV)
        # Remove previous result for same asset/horizon/gamma (re-run)
        mask = ((df_existing['asset'] == result['asset']) &
                (df_existing['horizon'] == result['horizon']) &
                (df_existing['gamma'] == result['gamma']))
        df_existing = df_existing[~mask]
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(RESULTS_CSV, index=False)


def is_completed(asset, gamma, horizon):
    """Check if this gamma/horizon combo already has a result."""
    if not os.path.exists(RESULTS_CSV):
        return False
    df = pd.read_csv(RESULTS_CSV)
    mask = ((df['asset'] == asset) &
            (df['horizon'] == horizon) &
            (df['gamma'] == gamma))
    return mask.any()


def print_summary(asset):
    """Print a comparison table of all results."""
    if not os.path.exists(RESULTS_CSV):
        print("No results yet.")
        return

    df = pd.read_csv(RESULTS_CSV)
    df = df[df['asset'] == asset].sort_values(['horizon', 'gamma'], ascending=[True, False])

    if df.empty:
        print(f"No results for {asset}.")
        return

    print("\n" + "=" * 90)
    print(f"  V15.1 GAMMA OPTIMIZATION RESULTS — {asset}")
    print("=" * 90)
    print(f"  {'Horizon':<10} {'Gamma':<8} {'Model':<20} {'Window':<8} {'Acc%':<8} {'Return%':<10} {'Trades':<8} {'Score':<8} {'Feats':<6} {'Time':<6}")
    print("-" * 90)

    for h in HORIZONS:
        h_df = df[df['horizon'] == h].sort_values('gamma', ascending=False)
        for _, row in h_df.iterrows():
            gamma_str = f"{row['gamma']:.3f}" if row['gamma'] < 1.0 else "1.000"
            print(f"  {v15._horizon_label(int(row['horizon'])):<10} {gamma_str:<8} "
                  f"{row['best_combo']:<20} {int(row['best_window']):<8} "
                  f"{row['accuracy']:<8.1f} {row['return_pct']:<+10.1f} "
                  f"{int(row['trades']):<8} {row['combined_score']:<8.3f} "
                  f"{int(row['n_features']):<6} {row['elapsed_min']:<.0f}m")
        if h != HORIZONS[-1]:
            print("-" * 90)

    print("=" * 90)

    # Highlight best per horizon (by return)
    print("\n  BEST BY RETURN:")
    for h in HORIZONS:
        h_df = df[df['horizon'] == h]
        if not h_df.empty:
            best = h_df.loc[h_df['return_pct'].idxmax()]
            print(f"    {v15._horizon_label(h)}: gamma={best['gamma']:.3f} → "
                  f"{best['return_pct']:+.1f}% return, {best['accuracy']:.1f}% acc, "
                  f"{best['best_combo']}")
    print()


def main():
    # Parse args
    resume = '--resume' in sys.argv
    asset = DEFAULT_ASSET

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--asset' and i < len(sys.argv) - 1:
            asset = sys.argv[i + 1].upper()

    # Also support: python testing_v15.1.py ETH
    for arg in sys.argv[1:]:
        if arg.upper() in ('BTC', 'ETH', 'XRP', 'DOGE'):
            asset = arg.upper()
            break

    total_tests = len(GAMMAS_TO_TEST) * len(HORIZONS)
    completed = 0
    skipped = 0

    print("=" * 70)
    print(f"  V15.1 GAMMA OPTIMIZATION TEST")
    print(f"  Asset: {asset}")
    print(f"  Gammas: {GAMMAS_TO_TEST}")
    print(f"  Horizons: {[v15._horizon_label(h) for h in HORIZONS]}")
    print(f"  Total tests: {total_tests}")
    print(f"  Resume: {resume}")
    print(f"  Results: {RESULTS_CSV}")
    print(f"  Charts: {CHARTS_DIR_TEST}/")
    print("=" * 70)

    t_start = time.time()

    for gamma in GAMMAS_TO_TEST:
        for horizon in HORIZONS:
            test_num = completed + skipped + 1

            if resume and is_completed(asset, gamma, horizon):
                print(f"\n  [{test_num}/{total_tests}] gamma={gamma}, {v15._horizon_label(horizon)} — SKIPPED (already done)")
                skipped += 1
                continue

            print(f"\n  [{test_num}/{total_tests}] Testing gamma={gamma}, {v15._horizon_label(horizon)}...")

            result = run_gamma_test(asset, gamma, horizon, resume=resume)

            if result:
                save_result(result)
                completed += 1
            else:
                print(f"  FAILED: gamma={gamma}, {v15._horizon_label(horizon)}")
                completed += 1  # Count as attempted

    elapsed_total = time.time() - t_start

    print(f"\n\n  All tests done! {completed} completed, {skipped} skipped. "
          f"Total time: {elapsed_total/60:.1f} min")

    # Print summary table
    print_summary(asset)


if __name__ == '__main__':
    main()
