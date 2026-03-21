"""
TESTING: V15.2 — DIAG_STEP Impact Test (4h step vs 24h step)
============================================================
Tests whether a finer walk-forward step (4h = 16 candles) improves
trade count and win rate compared to V15.1's 24h step.

Uses the best gammas from V15.1:
  - 4h horizon: gamma = 0.999
  - 8h horizon: gamma = 0.997

Only 2 tests total. Quick run on laptop.

Results saved to: models/testing_v15.2_results.csv
Charts saved to:  charts/v15.2_test/

Usage:
  python testing_v15.2.py                  # Run both tests
  python testing_v15.2.py --resume         # Skip already-completed tests
  python testing_v15.2.py --asset ETH      # Test ETH instead of BTC

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
# Best gammas from V15.1 results (by return)
GAMMA_4H = 0.999   # V15.1: +17.8%, 81.4% acc
GAMMA_8H = 0.997   # V15.1: +21.0%, 88.6% acc

TESTS = [
    {'horizon': 4, 'gamma': GAMMA_4H},
    {'horizon': 8, 'gamma': GAMMA_8H},
]

NEW_DIAG_STEP_HOURS = 4  # 4h step = 16 candles (vs V15.1's 24h = 96 candles)

DEFAULT_ASSET = 'BTC'

# Isolated output paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(_SCRIPT_DIR, 'models', 'testing_v15.2_results.csv')
CHARTS_DIR_TEST = os.path.join(_SCRIPT_DIR, 'charts', 'v15.2_test')

# ============================================================
# Import V15 Cacarot
# ============================================================
import importlib
v15 = importlib.import_module('crypto_trading_system_v15')

# Override DIAG_STEP to 4h (16 candles)
v15.DIAG_STEP = v15._hours_to_rows(NEW_DIAG_STEP_HOURS)
print(f"  DIAG_STEP overridden: {NEW_DIAG_STEP_HOURS}h = {v15.DIAG_STEP} candles "
      f"(was 24h = {v15._hours_to_rows(24)} candles)")


def run_test(asset, gamma, horizon):
    """Run a single Mode D pipeline for one gamma/horizon combo."""
    print("\n" + "#" * 70)
    print(f"  DIAG_STEP TEST: {asset} | gamma={gamma} | {v15._horizon_label(horizon)} | step={NEW_DIAG_STEP_HOURS}h")
    print("#" * 70)

    t0 = time.time()

    # Patch gamma loader
    original_load = v15._load_mode_d_config

    def _patched_load(asset_name, h):
        result = original_load(asset_name, h)
        if result:
            result['gamma'] = gamma
        else:
            result = {'gamma': gamma}
        return result

    v15._load_mode_d_config = _patched_load

    # Override charts dir for isolation
    original_charts_dir = v15.CHARTS_DIR
    v15.CHARTS_DIR = CHARTS_DIR_TEST
    os.makedirs(CHARTS_DIR_TEST, exist_ok=True)

    try:
        v15.run_mode_d([asset], horizon=horizon)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        v15._load_mode_d_config = original_load
        v15.CHARTS_DIR = original_charts_dir
        return None
    finally:
        v15._load_mode_d_config = original_load
        v15.CHARTS_DIR = original_charts_dir

    elapsed = time.time() - t0

    # Read result from V15 CSV
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

    row = matches.iloc[-1]

    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'asset': asset,
        'horizon': horizon,
        'gamma': gamma,
        'diag_step_hours': NEW_DIAG_STEP_HOURS,
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

    print(f"\n  RESULT: gamma={gamma} | step={NEW_DIAG_STEP_HOURS}h | {result['best_combo']} | "
          f"acc={result['accuracy']:.1f}% | ret={result['return_pct']:+.1f}% | "
          f"wr={result['win_rate']:.1f}% | trades={result['trades']} | {result['elapsed_min']:.0f} min")

    return result


def save_result(result):
    """Append a single result to the test CSV."""
    os.makedirs(os.path.join(_SCRIPT_DIR, 'models'), exist_ok=True)
    df_new = pd.DataFrame([result])

    if os.path.exists(RESULTS_CSV):
        df_existing = pd.read_csv(RESULTS_CSV)
        mask = ((df_existing['asset'] == result['asset']) &
                (df_existing['horizon'] == result['horizon']) &
                (df_existing['gamma'] == result['gamma']))
        df_existing = df_existing[~mask]
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(RESULTS_CSV, index=False)


def is_completed(asset, gamma, horizon):
    if not os.path.exists(RESULTS_CSV):
        return False
    df = pd.read_csv(RESULTS_CSV)
    mask = ((df['asset'] == asset) &
            (df['horizon'] == horizon) &
            (df['gamma'] == gamma))
    return mask.any()


def main():
    resume = '--resume' in sys.argv
    asset = DEFAULT_ASSET

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--asset' and i < len(sys.argv) - 1:
            asset = sys.argv[i + 1].upper()

    for arg in sys.argv[1:]:
        if arg.upper() in ('BTC', 'ETH', 'XRP', 'DOGE'):
            asset = arg.upper()
            break

    print("=" * 70)
    print(f"  V15.2 DIAG_STEP IMPACT TEST")
    print(f"  Asset: {asset}")
    print(f"  DIAG_STEP: {NEW_DIAG_STEP_HOURS}h ({v15.DIAG_STEP} candles)")
    print(f"  Tests: {len(TESTS)} (4h @ gamma={GAMMA_4H}, 8h @ gamma={GAMMA_8H})")
    print(f"  Resume: {resume}")
    print(f"  Results: {RESULTS_CSV}")
    print("=" * 70)

    t_start = time.time()
    completed = 0
    skipped = 0

    for test in TESTS:
        horizon = test['horizon']
        gamma = test['gamma']
        test_num = completed + skipped + 1

        if resume and is_completed(asset, gamma, horizon):
            print(f"\n  [{test_num}/{len(TESTS)}] {v15._horizon_label(horizon)} gamma={gamma} — SKIPPED (already done)")
            skipped += 1
            continue

        print(f"\n  [{test_num}/{len(TESTS)}] {v15._horizon_label(horizon)} gamma={gamma}...")

        result = run_test(asset, gamma, horizon)

        if result:
            save_result(result)
            completed += 1
        else:
            print(f"  FAILED: {v15._horizon_label(horizon)} gamma={gamma}")
            completed += 1

    elapsed_total = time.time() - t_start

    print(f"\n\n  All tests done! {completed} completed, {skipped} skipped. "
          f"Total time: {elapsed_total/60:.1f} min")

    # Print comparison with V15.1
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        df = df[df['asset'] == asset]

        v15_1_csv = os.path.join(_SCRIPT_DIR, 'models', 'testing_v15.1_results.csv')
        if os.path.exists(v15_1_csv):
            df_v151 = pd.read_csv(v15_1_csv)
            df_v151 = df_v151[df_v151['asset'] == asset]

            print("\n" + "=" * 95)
            print(f"  V15.2 vs V15.1 COMPARISON — {asset}")
            print("=" * 95)
            print(f"  {'Horizon':<10} {'Test':<8} {'Step':<8} {'Gamma':<8} {'Model':<20} {'Acc%':<8} {'Return%':<10} {'WR%':<8} {'Trades':<8}")
            print("-" * 95)

            for test in TESTS:
                h = test['horizon']
                g = test['gamma']
                label = v15._horizon_label(h)

                # V15.1 row (24h step)
                v151_row = df_v151[(df_v151['horizon'] == h) & (df_v151['gamma'] == g)]
                if not v151_row.empty:
                    r = v151_row.iloc[0]
                    print(f"  {label:<10} {'V15.1':<8} {'24h':<8} {g:<8.3f} {r['best_combo']:<20} "
                          f"{r['accuracy']:<8.1f} {r['return_pct']:<+10.1f} {r['win_rate']:<8.1f} {int(r['trades']):<8}")

                # V15.2 row (4h step)
                v152_row = df[(df['horizon'] == h) & (df['gamma'] == g)]
                if not v152_row.empty:
                    r = v152_row.iloc[0]
                    print(f"  {label:<10} {'V15.2':<8} {'4h':<8} {g:<8.3f} {r['best_combo']:<20} "
                          f"{r['accuracy']:<8.1f} {r['return_pct']:<+10.1f} {r['win_rate']:<8.1f} {int(r['trades']):<8}")

                if h != TESTS[-1]['horizon']:
                    print("-" * 95)

            print("=" * 95)


if __name__ == '__main__':
    main()
