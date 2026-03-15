"""
testing_literature_v2.py — A/B test 12 literature enhancements (V6)
====================================================================
Runs Mode D (BTC, 4h + 8h, 1y) for:
  0. Baseline           (all enhancements OFF)
  1. wavelet_denoising   — Denoise close price via DWT
  2. fractional_diff     — Fractionally differentiated price features
  3. hmm_regime          — Hidden Markov Model regime detection
  4. xgboost_model       — Add XGBoost to model pool
  5. sample_weighting    — Time-decay + uniqueness sample weights
  6. entropy_filter      — Shannon entropy signal filtering
  7. tri_state_labels    — 3-class labeling (BUY/SELL/NO-ACTION)
  8. stacking_ensemble   — Stacking meta-learner on model probabilities
  9. dynamic_feature_select — MI-based feature selection per step
 10. meta_labeling       — Second model filters primary signals
 11. adversarial_validation — Drop distribution-shift features per step
 12. kelly_sizing         — Fractional Kelly position sizing in backtest

Each test toggles ONE enhancement ON while keeping all others OFF.
Results are appended to testing_literature_v2.csv (never overwritten).
Per-run logs saved to testing_literature_v2_logs/.

Usage:
  python testing_literature_v2.py              # Run all 13 tests
  python testing_literature_v2.py 0            # Run only baseline
  python testing_literature_v2.py 3 5          # Run only tests 3 and 5
  python testing_literature_v2.py --resume     # Skip already-completed tests
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
V6_SCRIPT = os.path.join(SCRIPT_DIR, 'crypto_trading_system_v6.py')
PYTHON_EXE = sys.executable
CSV_PATH = os.path.join(SCRIPT_DIR, 'testing_literature_v2.csv')
LOG_DIR = os.path.join(SCRIPT_DIR, 'testing_literature_v2_logs')
MODELS_CSV = os.path.join(SCRIPT_DIR, 'models', 'crypto_hourly_best_models.csv')
HORIZONS = [4, 8]
DIAG_YEARS = 1
ASSET = 'BTC'

# Enhancement keys in V6
ENHANCEMENT_KEYS = [
    'wavelet_denoising',
    'fractional_diff',
    'hmm_regime',
    'xgboost_model',
    'sample_weighting',
    'entropy_filter',
    'tri_state_labels',
    'stacking_ensemble',
    'dynamic_feature_select',
    'meta_labeling',
    'adversarial_validation',
    'kelly_sizing',
]

# Test definitions: (test_name, dict of enhancements to enable)
TESTS = [
    ('baseline',                {}),
    ('wavelet_denoising',       {'wavelet_denoising': True}),
    ('fractional_diff',         {'fractional_diff': True}),
    ('hmm_regime',              {'hmm_regime': True}),
    ('xgboost_model',           {'xgboost_model': True}),
    ('sample_weighting',        {'sample_weighting': True}),
    ('entropy_filter',          {'entropy_filter': True}),
    ('tri_state_labels',        {'tri_state_labels': True}),
    ('stacking_ensemble',       {'stacking_ensemble': True}),
    ('dynamic_feature_select',  {'dynamic_feature_select': True}),
    ('meta_labeling',           {'meta_labeling': True}),
    ('adversarial_validation',  {'adversarial_validation': True}),
    ('kelly_sizing',            {'kelly_sizing': True}),
]


# ============================================================
# Helpers
# ============================================================
def build_env(enabled_enhancements):
    """Build environment variables for a test run."""
    env = os.environ.copy()
    for key in ENHANCEMENT_KEYS:
        env[f'ENH_{key.upper()}'] = '1' if enabled_enhancements.get(key, False) else '0'
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
    """Append one result row to CSV. Never overwrites."""
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
    """Run V6 Mode D as a subprocess for one horizon."""
    cmd = [PYTHON_EXE, V6_SCRIPT, 'D', ASSET, f'{horizon}h', f'{DIAG_YEARS}y']
    print(f"\n  $ {' '.join(cmd)}")
    print(f"  ENV: {' '.join(f'{k}={v}' for k, v in sorted(env.items()) if k.startswith('ENH_'))}")

    process = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace',
        cwd=SCRIPT_DIR,
    )

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

    if requested:
        tests_to_run = [(i, TESTS[i]) for i in requested]
    else:
        tests_to_run = list(enumerate(TESTS))

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
    print(f"  TESTING LITERATURE V2 — 12 Literature Enhancements")
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

        for horizon in HORIZONS:
            if resume_mode:
                completed = get_completed_tests()
                if (test_name, horizon) in completed:
                    print(f"\n  SKIP: {test_name} {horizon}h already completed")
                    continue

            run_count += 1
            print(f"\n  --- Run {run_count}/{total_runs}: {test_name} | {horizon}h ---")

            log_path = os.path.join(LOG_DIR, f'test_{test_idx}_{test_name}_{horizon}h.log')
            csv_mtime_before = os.path.getmtime(MODELS_CSV) if os.path.exists(MODELS_CSV) else 0

            t0 = time.time()
            with open(log_path, 'w', encoding='utf-8') as log_file:
                log_file.write(f"{'='*80}\n")
                log_file.write(f"TEST {test_idx}: {test_name} {horizon}h | Enhancements: {enh_str}\n")
                log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"{'='*80}\n\n")

                returncode = run_mode_d(test_name, horizon, env, log_file)
                elapsed = (time.time() - t0) / 60
                log_file.write(f"\nElapsed: {elapsed:.1f} min\n")

            if returncode != 0:
                print(f"\n  ERROR: Mode D exited with code {returncode}")
                append_result(test_name, horizon, None, enabled_enhancements, elapsed)
                continue

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

    elapsed_total = (time.time() - t_global) / 60
    print(f"\n{'=' * 80}")
    print(f"  ALL TESTS COMPLETE")
    print(f"  Total time: {elapsed_total:.1f} min ({elapsed_total/60:.1f} hours)")
    print(f"  Results: {CSV_PATH}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    print_summary()


def print_summary():
    """Print a comparison table from testing_literature_v2.csv."""
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
