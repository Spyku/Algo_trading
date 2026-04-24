"""
ab_matrix_runner.py — Factorial A/B test of Trim + Meta + Vol-scaled horizon.

Runs the full HRST pipeline across all combinations of:
  - Trim:  config/disabled_features.json enabled True/False (Mode F on/off)
  - Meta:  --meta-filter 0.45 vs none
Plus:
  - Vol-scaled horizon test (tools/test_vol_scaled_horizon.py --replay 1440)

ALL VARIANTS RUN ON THE SAME DATA SNAPSHOT. At startup the runner snapshots
all macro/OHLCV files into data/_ab_snapshot_<timestamp>/ and restores from
that snapshot before every HRST variant. HRST is invoked with --no-data-update
so downloads don't fire. Live trader's data files ARE restored intermittently
during matrix runs — this is acceptable because the live trader re-downloads
and re-appends on its next tick (idempotent).

All runs use --no-persist. Outputs tagged with distinct names. Results
consolidated into output/ab_matrix_results_<timestamp>.csv with full audit
columns: detector, horizons, confidences, shield, gates, features per horizon,
meta stats, etc.

Safe to run while live trader is active and while a position is open.

Usage:
  python tools/ab_matrix_runner.py
  python tools/ab_matrix_runner.py --skip-vol   # skip the vol-scaled test
  python tools/ab_matrix_runner.py --dry-run    # print plan, don't execute

Estimated runtime: 15-20h on desktop (4 HRST × 3-4h + vol test 30min).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

CONFIG_NOPROD = os.path.join(ENGINE_DIR, 'config', 'regime_config_ed_noprod.json')
CSV_NOPROD = os.path.join(ENGINE_DIR, 'models', 'crypto_ed_production_noprod.csv')
BEST_MODELS_NOPROD = os.path.join(ENGINE_DIR, 'models', 'crypto_ed_best_models_noprod.csv')
DISABLED_CFG = os.path.join(ENGINE_DIR, 'config', 'disabled_features.json')
LOG_DIR = os.path.join(ENGINE_DIR, 'logs')
OUTPUT_DIR = os.path.join(ENGINE_DIR, 'output')
DATA_DIR = os.path.join(ENGINE_DIR, 'data')
MACRO_DIR = os.path.join(DATA_DIR, 'macro_data')

# Files to snapshot/restore — anything HRST reads that could change across runs
SNAPSHOT_FILES = [
    'data/eth_hourly_data.csv',
    'data/btc_hourly_data.csv',
    'data/macro_data/macro_hourly.csv',
    'data/macro_data/macro_daily.csv',
    'data/macro_data/cross_asset.csv',
    'data/macro_data/fear_greed.csv',
    'data/macro_data/onchain_eth.csv',
    'data/macro_data/onchain_btc.csv',
    'data/macro_data/derivatives_eth.csv',
    'data/macro_data/derivatives_btc.csv',
    'data/macro_data/stablecoin_flows.csv',
    'data/macro_data/orderbook_snapshots.csv',
    'data/macro_data/options_iv_snapshot.csv',
]


def snapshot_data(snapshot_dir: str):
    """Copy current state of all data files to snapshot_dir."""
    os.makedirs(snapshot_dir, exist_ok=True)
    n_copied = 0
    for rel in SNAPSHOT_FILES:
        src = os.path.join(ENGINE_DIR, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(snapshot_dir, rel.replace('/', '__').replace('\\', '__'))
        shutil.copyfile(src, dst)
        n_copied += 1
    return n_copied


def restore_data(snapshot_dir: str):
    """Restore all data files from snapshot_dir back to live paths."""
    n_restored = 0
    for rel in SNAPSHOT_FILES:
        dst = os.path.join(ENGINE_DIR, rel)
        src = os.path.join(snapshot_dir, rel.replace('/', '__').replace('\\', '__'))
        if not os.path.exists(src):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        n_restored += 1
    return n_restored


def read_trim_enabled():
    """Read current enabled flag from disabled_features.json (read-only).
    Note: the matrix NEVER writes this file — trim state is passed to
    HRST via --trim-override instead, so the live trader isn't affected."""
    with open(DISABLED_CFG, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return bool(cfg.get('enabled', True))


def tag_outputs(tag: str):
    """Copy _noprod.* files with a descriptive tag suffix. Returns the tagged paths."""
    targets = [
        (CONFIG_NOPROD, CONFIG_NOPROD.replace('_noprod.json', f'_noprod_{tag}.json')),
        (CSV_NOPROD, CSV_NOPROD.replace('_noprod.csv', f'_noprod_{tag}.csv')),
        (BEST_MODELS_NOPROD, BEST_MODELS_NOPROD.replace('_noprod.csv', f'_noprod_{tag}.csv')),
    ]
    tagged = []
    for src, dst in targets:
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            tagged.append(dst)
    return tagged


def latest_log_after(since_epoch: float):
    """Return the newest ed_v1_*.log file created after since_epoch (HRST log)."""
    logs = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR)
            if f.startswith('ed_v1_') and f.endswith('.log')]
    eligible = [l for l in logs if os.path.getmtime(l) >= since_epoch]
    if not eligible:
        return None
    return max(eligible, key=os.path.getmtime)


def parse_hrst_log(log_path: str):
    """Extract strategy parameters + Mode T results from an HRST log."""
    if not log_path or not os.path.exists(log_path):
        return {}
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    out = {}

    m = re.search(r'WINNER:\s+detector=(\S+)\s+bull=(\d+)h@(\d+)%\s+bear=(\d+)h@(\d+)%', text)
    if m:
        out['detector'] = m.group(1)
        out['bull_h'] = int(m.group(2))
        out['bull_conf'] = int(m.group(3))
        out['bear_h'] = int(m.group(4))
        out['bear_conf'] = int(m.group(5))

    policies = re.findall(
        r'policy:\s+bull@(\d+)%\s+shield=(\w+)\s+\|\s+bear@(\d+)%\s+shield=(\w+)\s+\|\s+shield_thr=([\d.]+)%\s+/\s+(\d+)h',
        text)
    if policies:
        last = policies[-1]
        out['t_bull_conf'] = int(last[0])
        out['t_bull_shield'] = last[1] == 'True'
        out['t_bear_conf'] = int(last[2])
        out['t_bear_shield'] = last[3] == 'True'
        out['t_min_sell_pnl'] = float(last[4])
        out['t_max_hold'] = int(last[5])

    refs = re.findall(
        r'baselines V0\s+H1=([+-]?[\d.]+)%\s+H2=([+-]?[\d.]+)%\s+REF=([+-]?[\d.]+)%', text)
    if refs:
        out['t_ref_pct'] = float(refs[-1][2])
        out['t_h1_pct'] = float(refs[-1][0])
        out['t_h2_pct'] = float(refs[-1][1])

    # Collect ALL gate winners — last bull and last bear
    gate_matches = re.findall(
        r'(--- (?:BULL|BEAR) gate sweep ---.*?)(?:ETH WINNER: rr(\d+)h>=([\d.]+)%\s+OR\s+rr(\d+)h>=([\d.]+)%,\s+cd=(\d+)h|NO STRICT WINNER)',
        text, flags=re.DOTALL)
    bull_gate = None
    bear_gate = None
    for chunk, a, b, c, d, e in gate_matches:
        if not a:
            gate_str = 'OFF'
        else:
            gate_str = f"rr{a}>={b}% OR rr{c}>={d}% cd={e}h"
        if 'BULL' in chunk:
            bull_gate = gate_str
        elif 'BEAR' in chunk:
            bear_gate = gate_str
    if bull_gate is not None:
        out['bull_gate'] = bull_gate
    if bear_gate is not None:
        out['bear_gate'] = bear_gate

    conv = re.search(r'Converged at iteration (\d+)', text)
    if conv:
        out['t_converged_iter'] = int(conv.group(1))
    elif 'max_iter' in text:
        out['t_converged_iter'] = -1

    meta = re.search(
        r'\[meta-filter p>=([\d.]+)\]\s+kept=(\d+)\s+dropped=(\d+)\s+no_pred=(\d+)', text)
    if meta:
        out['meta_threshold'] = float(meta.group(1))
        out['meta_kept'] = int(meta.group(2))
        out['meta_dropped'] = int(meta.group(3))
        out['meta_no_pred'] = int(meta.group(4))

    # Mode S per-horizon best (pulled from ETH XXh WINNER lines)
    per_h = re.findall(r'ETH (\d+)h WINNER:\s+(\S+)\s+#(\d+)\s+\(APF=([\d.]+)\)', text)
    for h, src, rank, apf in per_h:
        out[f'h{h}_source'] = f"{src}#{rank}"
        out[f'h{h}_apf'] = float(apf)

    return out


def parse_production_csv(csv_path: str):
    """Extract per-horizon model config + feature list from a tagged production CSV."""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    try:
        import pandas as pd
    except ImportError:
        return {}
    df = pd.read_csv(csv_path)
    eth = df[df['coin'] == 'ETH']
    out = {}
    for _, r in eth.iterrows():
        h = int(r['horizon'])
        out[f'h{h}_combo'] = str(r['best_combo'])
        out[f'h{h}_window'] = int(r['best_window'])
        out[f'h{h}_gamma'] = float(r['gamma']) if 'gamma' in r and pd.notna(r['gamma']) else None
        out[f'h{h}_return'] = float(r.get('return_pct', 0))
        out[f'h{h}_accuracy'] = float(r.get('accuracy', 0))
        feats = str(r.get('optimal_features', '')).split(',')
        feats = [f.strip() for f in feats if f.strip() and f.strip() != 'nan']
        out[f'h{h}_n_features'] = len(feats)
        out[f'h{h}_features'] = '|'.join(feats)
        out[f'h{h}_logrets'] = sum(1 for f in feats if f.startswith('logret_'))
        out[f'h{h}_pysr'] = sum(1 for f in feats if f.startswith('pysr_'))
    return out


def run_hrst(label: str, trim_enabled: bool, meta_p, snapshot_dir: str, dry_run: bool,
             no_feature_floor: bool = False, optuna_seed=None):
    """Run a single HRST variant against the frozen data snapshot.
    Trim state is passed via --trim-override (in-memory only, the live
    trader's disabled_features.json is NEVER modified)."""
    print('=' * 80)
    print(f'  VARIANT: {label}  |  trim={trim_enabled}  |  meta={meta_p}  |  floor={"OFF" if no_feature_floor else "ON"}  |  seed={optuna_seed or "default(42)"}')
    print('=' * 80)

    cmd = [PY, 'crypto_trading_system_ed.py', 'HRST', 'ETH', '5,6,7,8h',
           '--replay', '1440', '--no-persist', '--no-data-update',
           '--trim-override', 'on' if trim_enabled else 'off']
    if meta_p is not None:
        cmd += ['--meta-filter', str(meta_p)]
    if no_feature_floor:
        cmd += ['--no-feature-floor']
    if optuna_seed is not None:
        cmd += ['--optuna-seed', str(optuna_seed)]

    if dry_run:
        print(f'  [DRY-RUN] restore snapshot; trim via --trim-override (no config file write)')
        print(f'  [DRY-RUN] would run: {" ".join(cmd)}')
        return {'variant': label, 'status': 'dry_run',
                'trim_enabled': trim_enabled, 'meta_threshold': meta_p,
                'feature_floor': not no_feature_floor,
                'optuna_seed': optuna_seed}

    # Restore data snapshot so all variants see the same bars
    n_restored = restore_data(snapshot_dir)
    print(f'  restored {n_restored} files from snapshot')
    print(f'  trim via CLI flag (--trim-override {"on" if trim_enabled else "off"}) — prod config untouched')
    print(f'  command: {" ".join(cmd)}')
    start = time.time()

    try:
        result = subprocess.run(cmd, cwd=ENGINE_DIR, capture_output=False)
        exit_code = result.returncode
    except Exception as e:
        exit_code = -1
        print(f'  ERROR: {e}')

    runtime_min = (time.time() - start) / 60
    print(f'  completed in {runtime_min:.1f} min (exit {exit_code})')

    log_path = latest_log_after(start)
    print(f'  log: {log_path}')

    parsed = parse_hrst_log(log_path) if log_path else {}
    tagged = tag_outputs(label)
    print(f'  tagged outputs: {[os.path.basename(t) for t in tagged]}')

    # Pull per-horizon model features from the tagged production CSV
    tagged_prod = next((t for t in tagged if 'crypto_ed_production_noprod_' in t), None)
    feature_details = parse_production_csv(tagged_prod) if tagged_prod else {}

    return {
        'variant': label,
        'trim_enabled': trim_enabled,
        'meta_threshold': meta_p,
        'feature_floor': not no_feature_floor,
        'optuna_seed': optuna_seed,
        'exit_code': exit_code,
        'runtime_min': round(runtime_min, 1),
        'log_path': log_path,
        'tagged_prod_csv': tagged_prod,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        **parsed,
        **feature_details,
    }


def run_vol_test(snapshot_dir: str, dry_run: bool):
    """Run the vol-scaled horizon test."""
    print('=' * 80)
    print('  VARIANT: vol_scaled_horizon')
    print('=' * 80)
    cmd = [PY, 'tools/test_vol_scaled_horizon.py', '--replay', '1440']
    if dry_run:
        print(f'  [DRY-RUN] would run: {" ".join(cmd)}')
        return {'variant': 'vol_scaled', 'status': 'dry_run'}
    if not dry_run:
        restore_data(snapshot_dir)
    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=ENGINE_DIR, capture_output=False)
        exit_code = result.returncode
    except Exception as e:
        exit_code = -1
        print(f'  ERROR: {e}')
    runtime_min = (time.time() - start) / 60
    return {
        'variant': 'vol_scaled',
        'exit_code': exit_code,
        'runtime_min': round(runtime_min, 1),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-vol', action='store_true', help='Skip the vol-scaled horizon test')
    ap.add_argument('--dry-run', action='store_true', help='Print plan without executing')
    ap.add_argument('--variants', default='all',
                    help='Comma-separated variant labels to run, or "all" (default), '
                         'or "focus" for the 3-variant floor/trim validation (A/B/C)')
    ap.add_argument('--seed', type=int, default=None,
                    help='Optuna TPESampler seed override (default 42). Set to e.g. 2026 '
                         'to re-test whether floor/trim effects replicate on a different seed.')
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_csv = os.path.join(OUTPUT_DIR, f'ab_matrix_results_{tag}.csv')
    snapshot_dir = os.path.join(DATA_DIR, f'_ab_snapshot_{tag}')

    print(f'\n{"#"*80}')
    print(f'#  FACTORIAL A/B MATRIX — ETH HRST')
    print(f'#  replay: 1440h (2mo, matches current market regime)')
    print(f'#  --no-persist: all runs are research-only, production untouched')
    print(f'#  --no-data-update + frozen snapshot: all variants on identical data')
    print(f'#  results CSV: {results_csv}')
    print(f'#  snapshot dir: {snapshot_dir}')
    print(f'#  dry-run: {args.dry_run}  |  skip-vol: {args.skip_vol}')
    print(f'{"#"*80}\n')

    # Freeze data at matrix start — all variants restore from here
    if not args.dry_run:
        n = snapshot_data(snapshot_dir)
        print(f'Snapshotted {n} data files to {snapshot_dir}\n')
    else:
        print(f'[DRY-RUN] would snapshot {len(SNAPSHOT_FILES)} data files\n')

    current_trim = read_trim_enabled()
    print(f'config/disabled_features.json state: enabled={current_trim} '
          f'(WILL NOT BE MODIFIED — trim is passed to each variant via --trim-override)\n')

    # (label, trim, meta_p, no_feature_floor)
    all_variants = [
        ('trimOFF_metaOFF',        False, None, False),
        ('trimON_metaOFF',         True,  None, False),
        ('trimOFF_metaON',         False, 0.45, False),
        ('trimON_metaON',          True,  0.45, False),
        ('trimON_metaON_floorOFF', True,  0.45, True),
    ]
    # Focused 3-variant validation for floor+trim+convergence replication
    focus_variants = [
        ('A_floorON_trimOFF',  False, None, False),
        ('B_floorON_trimON',   True,  None, False),
        ('C_floorOFF_trimOFF', False, None, True),
    ]
    if args.variants == 'focus':
        variants = focus_variants
    elif args.variants == 'all':
        variants = all_variants
    else:
        wanted = set(args.variants.split(','))
        pool = {v[0]: v for v in all_variants + focus_variants}
        variants = [pool[w] for w in wanted if w in pool]
        if not variants:
            print(f'ERROR: no variants matched {args.variants!r}')
            print(f'Available: {list(pool.keys())}')
            return

    print(f'Running {len(variants)} variant(s): {[v[0] for v in variants]}')
    if args.seed is not None:
        print(f'Optuna seed override: {args.seed} (default would be 42)')
    print()

    results = []
    try:
        for label, trim, meta, no_floor in variants:
            r = run_hrst(label, trim, meta, snapshot_dir, args.dry_run,
                         no_feature_floor=no_floor, optuna_seed=args.seed)
            results.append(r)
            _flush_csv(results_csv, results)

        if not args.skip_vol:
            r = run_vol_test(snapshot_dir, args.dry_run)
            results.append(r)
            _flush_csv(results_csv, results)
    finally:
        if not args.dry_run:
            # Verify disabled_features.json was NOT touched (sanity check)
            final_trim = read_trim_enabled()
            if final_trim != current_trim:
                print(f'\n!! WARNING: disabled_features.json changed from {current_trim} → {final_trim} during run')
                print(f'!! This should not happen with --trim-override. Check logs.')
            else:
                print(f'\nconfig/disabled_features.json integrity OK (enabled={final_trim})')
            # Live trader's data files were restored from snapshot before each variant;
            # live trader re-downloads on its own schedule (idempotent).
            print(f'Snapshot kept at: {snapshot_dir} (delete manually if not needed)')

    if not args.dry_run:
        _print_summary(results)
    print(f'\nFull audit CSV: {results_csv}')


def _flush_csv(path: str, results: list):
    import csv
    if not results:
        return
    keys = set()
    for r in results:
        keys.update(r.keys())
    # Stable column order: id fields first, then stats, then detailed features last
    priority = ['variant', 'trim_enabled', 'meta_threshold', 'feature_floor',
                'exit_code', 'runtime_min',
                'timestamp', 'detector', 'bull_h', 'bull_conf', 'bear_h', 'bear_conf',
                't_bull_shield', 't_bear_shield', 't_min_sell_pnl', 't_max_hold',
                't_ref_pct', 't_h1_pct', 't_h2_pct', 't_converged_iter',
                'bull_gate', 'bear_gate',
                'meta_kept', 'meta_dropped', 'meta_no_pred']
    ordered = [k for k in priority if k in keys] + sorted(k for k in keys if k not in priority)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in results:
            w.writerow(r)


def _print_summary(results: list):
    print('\n' + '=' * 120)
    print('  SUMMARY')
    print('=' * 120)
    cols = ['variant', 'detector', 'bull_h', 'bear_h', 't_bull_shield', 't_bear_shield',
            't_ref_pct', 't_converged_iter', 'meta_kept', 'meta_dropped', 'runtime_min']
    print('  ' + '  '.join(f'{c[:14]:>14s}' for c in cols))
    for r in results:
        row = [str(r.get(c, '—'))[:14] for c in cols]
        print('  ' + '  '.join(f'{v:>14s}' for v in row))
    print('=' * 120)


if __name__ == '__main__':
    main()
