"""
testing_feature_stability.py — Feature stability across assets & horizons
=========================================================================
Runs Mode D feature analysis (Step 4 only) for BTC and ETH on 4h and 8h,
then cross-references which features are consistently KEPT or DROPPED.

Goal: Identify features that are always useless (→ remove from pipeline)
      and features that are always useful (→ trust in production).

Produces:
  - testing_feature_stability.csv  (raw scores per feature per run)
  - testing_feature_stability_summary.csv  (cross-run stability report)
  - Console summary table

Usage:
  python testing_feature_stability.py              # Run all 4 configs
  python testing_feature_stability.py --summary    # Just print summary from existing CSV
  python testing_feature_stability.py --resume     # Skip already-completed runs
"""

import subprocess
import sys
import os
import csv
import time
import re
from datetime import datetime
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V5_SCRIPT = os.path.join(SCRIPT_DIR, 'crypto_trading_system.py')  # Use production V5
PYTHON_EXE = sys.executable
CSV_PATH = os.path.join(SCRIPT_DIR, 'testing_feature_stability.csv')
SUMMARY_CSV_PATH = os.path.join(SCRIPT_DIR, 'testing_feature_stability_summary.csv')
LOG_DIR = os.path.join(SCRIPT_DIR, 'testing_feature_stability_logs')
FEATURE_ANALYSIS_DIR = os.path.join(SCRIPT_DIR, 'models')

# Test matrix: all combinations of asset × horizon
TEST_MATRIX = [
    ('BTC', 4),
    ('BTC', 8),
    ('ETH', 4),
    ('ETH', 8),
]

DIAG_YEARS = 1  # 1y period for all runs


# ============================================================
# Helpers
# ============================================================
def parse_feature_analysis_csv(asset):
    """Read the feature analysis CSV produced by Mode D step 4.
    Returns dict: {feature_name: {'score': int, 'category': str, 'triage': str}}
    """
    # Mode D writes to models/crypto_feature_analysis_{asset}_auto.csv
    csv_path = os.path.join(FEATURE_ANALYSIS_DIR, f'crypto_feature_analysis_{asset.lower()}_auto.csv')
    if not os.path.exists(csv_path):
        print(f"  WARNING: Feature analysis CSV not found: {csv_path}")
        return None

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        result = {}
        for _, row in df.iterrows():
            feature = row.get('feature', row.get('name', ''))
            if not feature:
                continue
            result[feature] = {
                'score': int(row.get('score', 0)),
                'category': row.get('category', ''),
                'triage': row.get('triage', row.get('decision', '')),
            }
        return result
    except Exception as e:
        print(f"  WARNING: Could not parse {csv_path}: {e}")
        return None


def parse_optimal_features_from_log(log_path):
    """Fallback: parse optimal features from Mode D log output."""
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        # Look for "Optimal features (N): feat1, feat2, ..."
        match = re.search(r'Optimal features \((\d+)\):\s*(.+)', content)
        if match:
            features = [f.strip() for f in match.group(2).split(',')]
            return features
    except Exception:
        pass
    return None


def append_raw_result(asset, horizon, feature_data, elapsed_min):
    """Append raw feature scores to testing_feature_stability.csv."""
    fieldnames = ['timestamp', 'asset', 'horizon', 'feature', 'score',
                  'category', 'triage', 'elapsed_min']
    file_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0

    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for feature, data in sorted(feature_data.items()):
            writer.writerow({
                'timestamp': ts,
                'asset': asset,
                'horizon': horizon,
                'feature': feature,
                'score': data['score'],
                'category': data['category'],
                'triage': data['triage'],
                'elapsed_min': f'{elapsed_min:.1f}',
            })


def get_completed_runs():
    """Read CSV and return set of (asset, horizon) pairs already completed."""
    completed = set()
    if not os.path.exists(CSV_PATH):
        return completed
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row['asset'], int(row['horizon'])))
    except Exception:
        pass
    return completed


def run_mode_d(asset, horizon, log_file):
    """Run V5 Mode D for one asset+horizon. Returns exit code."""
    cmd = [PYTHON_EXE, V5_SCRIPT, 'D', asset, f'{horizon}h', f'{DIAG_YEARS}y']
    print(f"\n  $ {' '.join(cmd)}")

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['MODELS_CSV_OVERRIDE'] = os.path.join(SCRIPT_DIR, 'models', 'crypto_hourly_best_models_stability_test.csv')

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
# Summary / Analysis
# ============================================================
def build_summary():
    """Build cross-run stability report from raw CSV.

    For each feature, count how many runs it was KEEP/MAYBE/DROP.
    Classify stability:
      - ALWAYS_KEEP: KEEP in all 4 runs
      - ALWAYS_DROP: DROP in all 4 runs
      - MOSTLY_KEEP: KEEP in 3+ runs
      - MOSTLY_DROP: DROP in 3+ runs
      - UNSTABLE: mixed results
    """
    if not os.path.exists(CSV_PATH):
        print("No raw data found. Run the tests first.")
        return None

    import pandas as pd
    df = pd.read_csv(CSV_PATH)

    # Get unique runs
    runs = df.groupby(['asset', 'horizon']).size().reset_index()
    n_runs = len(runs)
    print(f"\n  Found data for {n_runs} runs: {list(zip(runs['asset'], runs['horizon']))}")

    # Pivot: feature → {(asset, horizon): triage}
    features = defaultdict(lambda: {'scores': {}, 'triages': {}, 'category': ''})
    for _, row in df.iterrows():
        key = (row['asset'], int(row['horizon']))
        feat = row['feature']
        features[feat]['scores'][key] = int(row['score'])
        features[feat]['triages'][key] = row['triage']
        features[feat]['category'] = row['category']

    # Build summary
    summary_rows = []
    for feat, data in sorted(features.items()):
        triages = data['triages']
        scores = data['scores']
        n_keep = sum(1 for t in triages.values() if t == 'KEEP')
        n_drop = sum(1 for t in triages.values() if t == 'DROP')
        n_maybe = sum(1 for t in triages.values() if t == 'MAYBE')
        n_present = len(triages)
        avg_score = sum(scores.values()) / len(scores) if scores else 0

        if n_keep == n_present and n_present == n_runs:
            stability = 'ALWAYS_KEEP'
        elif n_drop == n_present and n_present == n_runs:
            stability = 'ALWAYS_DROP'
        elif n_keep >= n_runs - 1 and n_present >= n_runs - 1:
            stability = 'MOSTLY_KEEP'
        elif n_drop >= n_runs - 1 and n_present >= n_runs - 1:
            stability = 'MOSTLY_DROP'
        else:
            stability = 'UNSTABLE'

        # Per-run detail
        run_details = {}
        for asset, h in TEST_MATRIX:
            key = (asset, h)
            run_details[f'{asset}_{h}h'] = triages.get(key, '-')
            run_details[f'{asset}_{h}h_score'] = scores.get(key, '-')

        summary_rows.append({
            'feature': feat,
            'category': data['category'],
            'stability': stability,
            'avg_score': round(avg_score, 1),
            'n_keep': n_keep,
            'n_drop': n_drop,
            'n_maybe': n_maybe,
            'n_present': n_present,
            **run_details,
        })

    # Sort: ALWAYS_DROP first (candidates for removal), then ALWAYS_KEEP, etc.
    stability_order = {'ALWAYS_DROP': 0, 'MOSTLY_DROP': 1, 'UNSTABLE': 2,
                       'MOSTLY_KEEP': 3, 'ALWAYS_KEEP': 4}
    summary_rows.sort(key=lambda r: (stability_order.get(r['stability'], 2), -r['avg_score']))

    # Write summary CSV
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(SUMMARY_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n  Summary written to: {SUMMARY_CSV_PATH}")

    return summary_rows


def print_summary(summary_rows=None):
    """Print stability summary table to console."""
    if summary_rows is None:
        summary_rows = build_summary()
    if not summary_rows:
        return

    # Count by stability
    counts = defaultdict(int)
    for row in summary_rows:
        counts[row['stability']] += 1

    print(f"\n{'=' * 110}")
    print(f"  FEATURE STABILITY SUMMARY")
    print(f"  {counts.get('ALWAYS_KEEP', 0)} always kept | "
          f"{counts.get('MOSTLY_KEEP', 0)} mostly kept | "
          f"{counts.get('UNSTABLE', 0)} unstable | "
          f"{counts.get('MOSTLY_DROP', 0)} mostly dropped | "
          f"{counts.get('ALWAYS_DROP', 0)} always dropped")
    print(f"{'=' * 110}")

    # Print by stability group
    for stability in ['ALWAYS_DROP', 'MOSTLY_DROP', 'UNSTABLE', 'MOSTLY_KEEP', 'ALWAYS_KEEP']:
        group = [r for r in summary_rows if r['stability'] == stability]
        if not group:
            continue

        print(f"\n  --- {stability} ({len(group)} features) ---")
        print(f"  {'Feature':<35s} {'Category':<12s} {'Avg':>5s}  "
              f"{'BTC4h':>6s} {'BTC8h':>6s} {'ETH4h':>6s} {'ETH8h':>6s}")
        print(f"  {'-'*35} {'-'*12} {'-----':>5s}  "
              f"{'------':>6s} {'------':>6s} {'------':>6s} {'------':>6s}")

        for row in group:
            btc4 = row.get('BTC_4h', '-')
            btc8 = row.get('BTC_8h', '-')
            eth4 = row.get('ETH_4h', '-')
            eth8 = row.get('ETH_8h', '-')
            print(f"  {row['feature']:<35s} {row['category']:<12s} {row['avg_score']:>5.1f}  "
                  f"{btc4:>6s} {btc8:>6s} {eth4:>6s} {eth8:>6s}")

    print(f"\n{'=' * 110}")

    # Actionable recommendations
    always_drop = [r for r in summary_rows if r['stability'] == 'ALWAYS_DROP']
    always_keep = [r for r in summary_rows if r['stability'] == 'ALWAYS_KEEP']

    if always_drop:
        print(f"\n  RECOMMENDATION: Remove these {len(always_drop)} features from the pipeline:")
        for r in always_drop:
            print(f"    - {r['feature']} ({r['category']}, avg score: {r['avg_score']:.1f})")

    if always_keep:
        print(f"\n  CORE FEATURES ({len(always_keep)} features always kept):")
        for r in always_keep:
            print(f"    + {r['feature']} ({r['category']}, avg score: {r['avg_score']:.1f})")

    print(f"\n{'=' * 110}")


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # --summary: just print from existing data
    if '--summary' in sys.argv:
        print_summary()
        return

    resume_mode = '--resume' in sys.argv

    # Determine which runs to execute
    if resume_mode:
        completed = get_completed_runs()
        runs_to_do = [(a, h) for a, h in TEST_MATRIX if (a, h) not in completed]
        if not runs_to_do:
            print("All runs already completed. Printing summary.")
            print_summary()
            return
    else:
        runs_to_do = list(TEST_MATRIX)

    total_runs = len(runs_to_do)

    print("=" * 80)
    print(f"  FEATURE STABILITY TEST")
    print(f"  Matrix: {', '.join(f'{a} {h}h' for a, h in runs_to_do)}")
    print(f"  Period: {DIAG_YEARS}y | Runs: {total_runs}")
    print(f"  Raw results: {CSV_PATH}")
    print(f"  Summary: {SUMMARY_CSV_PATH}")
    print(f"  Logs: {LOG_DIR}/")
    if resume_mode:
        print(f"  Resume mode: skipping already-completed runs")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    t_global = time.time()

    for run_idx, (asset, horizon) in enumerate(runs_to_do, 1):
        print(f"\n{'#' * 80}")
        print(f"  RUN {run_idx}/{total_runs}: {asset} {horizon}h")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#' * 80}")

        log_path = os.path.join(LOG_DIR, f'feature_stability_{asset}_{horizon}h.log')

        # Record feature analysis CSV mtime before run
        fa_csv = os.path.join(FEATURE_ANALYSIS_DIR, f'crypto_feature_analysis_{asset.lower()}_auto.csv')
        fa_mtime_before = os.path.getmtime(fa_csv) if os.path.exists(fa_csv) else 0

        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"{'='*80}\n")
            log_file.write(f"FEATURE STABILITY: {asset} {horizon}h | {DIAG_YEARS}y\n")
            log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"{'='*80}\n\n")

            t0 = time.time()
            returncode = run_mode_d(asset, horizon, log_file)
            elapsed = (time.time() - t0) / 60

            if returncode != 0:
                print(f"\n  ERROR: Mode D exited with code {returncode}")
                log_file.write(f"\nERROR: exit code {returncode}\n")
                continue

        # Check if feature analysis CSV was updated
        fa_mtime_after = os.path.getmtime(fa_csv) if os.path.exists(fa_csv) else 0
        if fa_mtime_after <= fa_mtime_before:
            print(f"\n  WARNING: Feature analysis CSV was NOT updated for {asset}")
            # Try to parse from log
            optimal = parse_optimal_features_from_log(log_path)
            if optimal:
                print(f"  Parsed {len(optimal)} optimal features from log (no scores available)")
            continue

        # Read feature scores
        feature_data = parse_feature_analysis_csv(asset)
        if feature_data:
            append_raw_result(asset, horizon, feature_data, elapsed)
            n_keep = sum(1 for d in feature_data.values() if d['triage'] == 'KEEP')
            n_drop = sum(1 for d in feature_data.values() if d['triage'] == 'DROP')
            n_maybe = sum(1 for d in feature_data.values() if d['triage'] == 'MAYBE')
            print(f"\n  >> {asset} {horizon}h: {len(feature_data)} features scored | "
                  f"KEEP={n_keep} MAYBE={n_maybe} DROP={n_drop} | {elapsed:.1f} min")
        else:
            print(f"\n  WARNING: No feature data for {asset} {horizon}h")

    elapsed_total = (time.time() - t_global) / 60
    print(f"\n{'=' * 80}")
    print(f"  ALL RUNS COMPLETE")
    print(f"  Total time: {elapsed_total:.1f} min ({elapsed_total/60:.1f} hours)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    # Build and print summary
    print_summary()


if __name__ == '__main__':
    main()
