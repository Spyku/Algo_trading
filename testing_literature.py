"""
testing_literature.py — A/B test each V5.5 enhancement individually
====================================================================
Runs Mode D (BTC, 4h + 8h, 1y) for:
  0. Baseline          (all enhancements OFF)
  1. on_chain_features
  2. derivatives_features
  3. triple_barrier_label
  4. slippage_model
  5. extended_diag_step
  6. gb_calibration
  7. purged_embargo

Each test toggles ONE enhancement ON while keeping all others OFF.
Results are appended to testing_literature.csv (never overwritten).
Full logs are saved per-run to testing_literature_logs/ folder.

Usage:
  python testing_literature.py              # Run all 8 tests
  python testing_literature.py 0            # Run only baseline
  python testing_literature.py 3 5          # Run only tests 3 and 5
  python testing_literature.py --resume     # Skip already-completed tests (checks CSV)
"""

import subprocess
import sys
import os
import csv
import time
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V55_SCRIPT = os.path.join(SCRIPT_DIR, 'crypto_trading_system_v5.5.py')
PYTHON_EXE = sys.executable  # Same Python that launched this script
CSV_PATH = os.path.join(SCRIPT_DIR, 'testing_literature.csv')
LOG_DIR = os.path.join(SCRIPT_DIR, 'testing_literature_logs')
MODELS_CSV = os.path.join(SCRIPT_DIR, 'models', 'crypto_hourly_best_models.csv')
HORIZONS = [4, 8]
DIAG_YEARS = 1
ASSET = 'BTC'

# Enhancement keys in the order they appear in V5.5
ENHANCEMENT_KEYS = [
    'on_chain_features',
    'derivatives_features',
    'triple_barrier_label',
    'slippage_model',
    'extended_diag_step',
    'gb_calibration',
    'purged_embargo',
]

# Test definitions: (test_name, dict of enhancements to enable)
TESTS = [
    ('baseline',              {}),
    ('on_chain_features',     {'on_chain_features': True}),
    ('derivatives_features',  {'derivatives_features': True}),
    ('triple_barrier_label',  {'triple_barrier_label': True}),
    ('slippage_model',        {'slippage_model': True}),
    ('extended_diag_step',    {'extended_diag_step': True}),
    ('gb_calibration',        {'gb_calibration': True}),
    ('purged_embargo',        {'purged_embargo': True}),
]


# ============================================================
# Helpers
# ============================================================
def build_env(enabled_enhancements):
    """Build environment variables for a test run.
    All enhancements start OFF; only the ones in enabled_enhancements are set to '1'."""
    env = os.environ.copy()
    for key in ENHANCEMENT_KEYS:
        env[f'ENH_{key.upper()}'] = '1' if enabled_enhancements.get(key, False) else '0'
    # Force unbuffered stdout so output streams in real time
    env['PYTHONUNBUFFERED'] = '1'
    return env


def read_best_model(horizon):
    """Read the best model result from crypto_hourly_best_models.csv for BTC + given horizon."""
    if not os.path.exists(MODELS_CSV):
        return None
    try:
        import pandas as pd
        df = pd.read_csv(MODELS_CSV)
        if 'horizon' not in df.columns:
            df['horizon'] = 4
        match = df[(df['coin'] == ASSET) & (df['horizon'] == horizon)]
        if match.empty:
            return None
        return match.iloc[-1].to_dict()
    except Exception as e:
        print(f"  WARNING: Could not read best model: {e}")
        return None


def append_result(test_name, horizon, result, enabled_enhancements, elapsed_min):
    """Append one result row to testing_literature.csv. Never overwrites."""
    fieldnames = [
        'timestamp', 'test_name', 'test_idx', 'horizon', 'coin',
        'best_combo', 'best_window', 'accuracy', 'return_pct',
        'win_rate', 'trades', 'combined_score', 'n_features',
        'elapsed_min', 'enhancements_on',
    ]
    file_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        # Which enhancements are ON for this test
        enh_on = ','.join(k for k, v in enabled_enhancements.items() if v) or 'none'
        test_idx = next((i for i, (name, _) in enumerate(TESTS) if name == test_name), -1)

        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_name': test_name,
            'test_idx': test_idx,
            'horizon': horizon,
            'coin': ASSET,
            'best_combo': result.get('best_combo', '') if result else 'FAILED',
            'best_window': result.get('best_window', '') if result else '',
            'accuracy': result.get('accuracy', '') if result else '',
            'return_pct': result.get('return_pct', '') if result else '',
            'win_rate': result.get('win_rate', '') if result else '',
            'trades': result.get('trades', '') if result else '',
            'combined_score': result.get('combined_score', '') if result else '',
            'n_features': result.get('n_features', '') if result else '',
            'elapsed_min': f'{elapsed_min:.1f}',
            'enhancements_on': enh_on,
        }
        writer.writerow(row)


def get_completed_tests():
    """Read CSV and return set of (test_name, horizon) pairs already completed."""
    completed = set()
    if not os.path.exists(CSV_PATH):
        return completed
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row['test_name'], int(row['horizon'])))
    except Exception:
        pass
    return completed


def run_mode_d(test_name, horizon, env, log_file):
    """Run V5.5 Mode D as a subprocess for one horizon. Stream output to both console and log."""
    cmd = [PYTHON_EXE, V55_SCRIPT, 'D', ASSET, f'{horizon}h', f'{DIAG_YEARS}y']
    print(f"\n  $ {' '.join(cmd)}")
    print(f"  ENV: {' '.join(f'{k}={v}' for k, v in sorted(env.items()) if k.startswith('ENH_'))}")

    process = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace',
        cwd=SCRIPT_DIR,
    )

    # Stream output line by line to both console and log file
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_file.write(line)
        log_file.flush()

    process.wait()
    return process.returncode


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # Parse CLI args
    resume_mode = '--resume' in sys.argv
    requested = []
    for arg in sys.argv[1:]:
        if arg == '--resume':
            continue
        if arg.isdigit():
            idx = int(arg)
            if 0 <= idx < len(TESTS):
                requested.append(idx)

    # If specific tests requested, only run those
    if requested:
        tests_to_run = [(i, TESTS[i]) for i in requested]
    else:
        tests_to_run = list(enumerate(TESTS))

    # If resume mode, skip already-completed tests
    if resume_mode:
        completed = get_completed_tests()
        tests_to_run = [
            (i, (name, enh)) for i, (name, enh) in tests_to_run
            if not all((name, h) in completed for h in HORIZONS)
        ]
        if not tests_to_run:
            print("All tests already completed. Nothing to do.")
            return

    total_tests = len(tests_to_run)
    total_runs = total_tests * len(HORIZONS)

    print("=" * 80)
    print(f"  TESTING LITERATURE — A/B Enhancement Tests")
    print(f"  Asset: {ASSET} | Horizons: {','.join(str(h)+'h' for h in HORIZONS)} | Period: {DIAG_YEARS}y")
    print(f"  Tests to run: {total_tests} configs x {len(HORIZONS)} horizons = {total_runs} Mode D runs")
    print(f"  Results: {CSV_PATH}")
    print(f"  Logs: {LOG_DIR}/")
    if resume_mode:
        print(f"  Resume mode: skipping already-completed tests")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    t_global = time.time()
    run_count = 0

    for test_idx, (test_name, enabled_enhancements) in tests_to_run:
        enh_str = ','.join(k for k, v in enabled_enhancements.items() if v) or 'none'

        print(f"\n{'#' * 80}")
        print(f"  TEST {test_idx}/{len(TESTS)-1}: {test_name}")
        print(f"  Enhancements ON: {enh_str}")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#' * 80}")

        env = build_env(enabled_enhancements)
        log_path = os.path.join(LOG_DIR, f'test_{test_idx}_{test_name}.log')

        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"\n{'='*80}\n")
            log_file.write(f"TEST {test_idx}: {test_name} | Enhancements: {enh_str}\n")
            log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"{'='*80}\n\n")

            for horizon in HORIZONS:
                # Skip if already completed in resume mode
                if resume_mode:
                    completed = get_completed_tests()
                    if (test_name, horizon) in completed:
                        print(f"\n  SKIP: {test_name} {horizon}h already completed")
                        continue

                run_count += 1
                print(f"\n  --- Run {run_count}/{total_runs}: {test_name} | {horizon}h ---")

                # Record CSV mtime before run to detect stale reads
                csv_mtime_before = os.path.getmtime(MODELS_CSV) if os.path.exists(MODELS_CSV) else 0

                t0 = time.time()
                returncode = run_mode_d(test_name, horizon, env, log_file)
                elapsed = (time.time() - t0) / 60

                if returncode != 0:
                    print(f"\n  ERROR: Mode D exited with code {returncode}")
                    log_file.write(f"\nERROR: exit code {returncode}\n")
                    append_result(test_name, horizon, None, enabled_enhancements, elapsed)
                    continue

                # Read result from best_models CSV — verify it was actually updated
                csv_mtime_after = os.path.getmtime(MODELS_CSV) if os.path.exists(MODELS_CSV) else 0
                if csv_mtime_after <= csv_mtime_before:
                    print(f"\n  WARNING: best_models CSV was NOT updated — Mode D produced no result")
                    append_result(test_name, horizon, None, enabled_enhancements, elapsed)
                    continue

                result = read_best_model(horizon)
                append_result(test_name, horizon, result, enabled_enhancements, elapsed)

                if result:
                    print(f"\n  >> RESULT: {test_name} {horizon}h | "
                          f"combo={result.get('best_combo', '?')} | "
                          f"w={result.get('best_window', '?')}h | "
                          f"acc={result.get('accuracy', '?')}% | "
                          f"ret={result.get('return_pct', '?')}% | "
                          f"score={result.get('combined_score', '?')} | "
                          f"{elapsed:.1f} min")
                else:
                    print(f"\n  WARNING: No result found for {test_name} {horizon}h")

                log_file.write(f"\nRESULT: {test_name} {horizon}h -> {result}\n")
                log_file.write(f"Elapsed: {elapsed:.1f} min\n\n")

    elapsed_total = (time.time() - t_global) / 60
    print(f"\n{'=' * 80}")
    print(f"  ALL TESTS COMPLETE")
    print(f"  Total time: {elapsed_total:.1f} min ({elapsed_total/60:.1f} hours)")
    print(f"  Results: {CSV_PATH}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    # Print summary table
    print_summary()


def print_summary():
    """Print a comparison table from testing_literature.csv."""
    if not os.path.exists(CSV_PATH):
        return

    print(f"\n{'=' * 100}")
    print(f"  SUMMARY TABLE")
    print(f"{'=' * 100}")
    print(f"  {'Test':<25s} {'Hz':>3s}  {'Combo':<20s} {'Win':>5s}  {'Acc':>6s}  {'Return':>8s}  {'Score':>8s}  {'Min':>6s}")
    print(f"  {'-'*25} {'---':>3s}  {'-'*20} {'-----':>5s}  {'------':>6s}  {'--------':>8s}  {'--------':>8s}  {'------':>6s}")

    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(f"  {row['test_name']:<25s} {row['horizon']:>3s}h "
                  f" {row.get('best_combo', ''):20s} "
                  f" w={row.get('best_window', ''):>4s} "
                  f" {row.get('accuracy', ''):>6s}% "
                  f" {row.get('return_pct', ''):>7s}% "
                  f" {row.get('combined_score', ''):>8s} "
                  f" {row.get('elapsed_min', ''):>5s}m")

    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
