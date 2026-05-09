"""
Re-run Mode T (+ chained Mode G) on AB Matrix variants 1-3 with the new
pinned min_sell=0.50% / max_hold=10h settings (commit b793402), compare
to their original matrix results, keep BOTH side-by-side.

Each variant has tagged _noprod files already:
  V1 trimOFF_metaOFF: config/regime_config_ed_noprod_trimOFF_metaOFF.json
                      models/crypto_ed_production_noprod_trimOFF_metaOFF.csv
  V2 trimON_metaOFF:  ...trimON_metaOFF...
  V3 trimOFF_metaON:  ...trimOFF_metaON...

For each variant this script:
  1. Restore data from a snapshot taken at script start (all variants see
     identical bars so comparison is apples-to-apples)
  2. Copy variant's config/CSV into the working _noprod.* slot
  3. Invoke `crypto_trading_system_ed.py T ETH <horizons> --replay 1440 --no-persist`
  4. Parse Mode T stdout for T winner + total return + converged iter
  5. Copy the tuned _noprod outputs aside as _retuneT_<variant>.*
  6. Append row to output/ab_retune_t_results_<timestamp>.csv

At the end, prints a side-by-side comparison of original AB matrix vs
new retune. Original matrix outputs (_noprod_<variant>.*) are left
untouched.

Safe to run on laptop concurrently with the AB matrix on desktop:
  - Uses --no-persist so live prod is never touched
  - Uses --no-data-update + snapshot restore so the live trader's
    download loop doesn't perturb what this script sees
  - Only writes to its own timestamped output CSV + retune-tagged files

Usage (from laptop):
  python tools/ab_matrix_retune_t.py
  python tools/ab_matrix_retune_t.py --variants V1,V3    # partial rerun
  python tools/ab_matrix_retune_t.py --replay 1440       # customize window
  python tools/ab_matrix_retune_t.py --dry-run           # print plan only
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Auto-detect venv Python (desktop vs laptop)
_PY_CANDIDATES = [
    r'C:\algo_trading\venv\Scripts\python.exe',              # desktop
    r'C:\Users\Alex\algo_trading\venv\Scripts\python.exe',   # laptop
]
PYTHON = next((p for p in _PY_CANDIDATES if os.path.exists(p)), sys.executable)

MAIN_SCRIPT = os.path.join(ENGINE_DIR, 'crypto_trading_system_ed.py')
CONFIG_DIR = os.path.join(ENGINE_DIR, 'config')
MODELS_DIR = os.path.join(ENGINE_DIR, 'models')
OUTPUT_DIR = os.path.join(ENGINE_DIR, 'output')
DATA_DIR = os.path.join(ENGINE_DIR, 'data')

# Working _noprod slot that Mode T writes to with --no-persist
WORKING_CONFIG = os.path.join(CONFIG_DIR, 'regime_config_ed_noprod.json')
WORKING_CSV = os.path.join(MODELS_DIR, 'crypto_ed_production_noprod.csv')

# Data files to snapshot/restore so variants see identical bars
DATA_FILES_TO_FREEZE = [
    'data/btc_hourly_data.csv',
    'data/eth_hourly_data.csv',
    'data/xrp_hourly_data.csv',
    'data/sol_hourly_data.csv',
    'data/link_hourly_data.csv',
    'data/macro_data/macro_daily.csv',
    'data/macro_data/cross_asset.csv',
    'data/macro_data/fear_greed.csv',
    'data/macro_data/derivatives_btc.csv',
    'data/macro_data/derivatives_eth.csv',
    'data/macro_data/onchain_btc.csv',
    'data/macro_data/onchain_eth.csv',
    'data/macro_data/stablecoin_flows.csv',
    'data/macro_data/orderbook_snapshots.csv',
    'data/macro_data/options_iv_snapshot.csv',
]

VARIANTS = [
    {
        'name': 'V1',
        'label': 'trimOFF_metaOFF',
        'config_tag': 'regime_config_ed_noprod_trimOFF_metaOFF.json',
        'csv_tag': 'crypto_ed_production_noprod_trimOFF_metaOFF.csv',
        'orig_detector': 'tsmom_672h',
        'orig_bull_h': 5, 'orig_bull_conf': 85,
        'orig_bear_h': 6, 'orig_bear_conf': 80,
        'orig_t_bull_shield': True, 'orig_t_bear_shield': False,
        'orig_t_min_sell': 0.60, 'orig_t_max_hold': 10,
        'orig_t_ref_pct': 52.64,
        'orig_converged': True,  # iter 3
    },
    {
        'name': 'V2',
        'label': 'trimON_metaOFF',
        'config_tag': 'regime_config_ed_noprod_trimON_metaOFF.json',
        'csv_tag': 'crypto_ed_production_noprod_trimON_metaOFF.csv',
        'orig_detector': 'price>sma72',
        'orig_bull_h': 5, 'orig_bull_conf': 90,
        'orig_bear_h': 5, 'orig_bear_conf': 65,
        'orig_t_bull_shield': False, 'orig_t_bear_shield': True,  # oscillated; last iter
        'orig_t_min_sell': 0.55, 'orig_t_max_hold': 10,
        'orig_t_ref_pct': 30.60,
        'orig_converged': False,  # max_iter hit (oscillated)
    },
    {
        'name': 'V3',
        'label': 'trimOFF_metaON',
        'config_tag': 'regime_config_ed_noprod_trimOFF_metaON.json',
        'csv_tag': 'crypto_ed_production_noprod_trimOFF_metaON.csv',
        'orig_detector': 'vol_calm',
        'orig_bull_h': 5, 'orig_bull_conf': 85,
        'orig_bear_h': 6, 'orig_bear_conf': 70,
        'orig_t_bull_shield': True, 'orig_t_bear_shield': False,
        'orig_t_min_sell': 0.60, 'orig_t_max_hold': 10,
        'orig_t_ref_pct': 62.40,
        'orig_converged': False,  # max_iter hit (max_hold jittered)
        'meta_filter': 0.45,       # V3 = trimOFF_metaON
    },
    {
        'name': 'V4',
        'label': 'trimON_metaON',
        'config_tag': 'regime_config_ed_noprod_trimON_metaON.json',
        'csv_tag': 'crypto_ed_production_noprod_trimON_metaON.csv',
        'orig_detector': 'sma24>sma100',
        'orig_bull_h': 5, 'orig_bull_conf': 90,
        'orig_bear_h': 5, 'orig_bear_conf': 65,
        'orig_t_bull_shield': True, 'orig_t_bear_shield': True,
        'orig_t_min_sell': 0.50, 'orig_t_max_hold': 10,
        'orig_t_ref_pct': 52.00,
        'orig_converged': True,   # iter 3
        'meta_filter': 0.45,      # V4 = trimON_metaON
    },
    {
        'name': 'V5',
        'label': 'trimON_metaON_floorOFF',
        'config_tag': 'regime_config_ed_noprod_trimON_metaON_floorOFF.json',
        'csv_tag': 'crypto_ed_production_noprod_trimON_metaON_floorOFF.csv',
        'orig_detector': 'price>sma72',
        'orig_bull_h': 6, 'orig_bull_conf': 70,
        'orig_bear_h': 5, 'orig_bear_conf': 70,
        'orig_t_bull_shield': True, 'orig_t_bear_shield': True,
        'orig_t_min_sell': 0.50, 'orig_t_max_hold': 10,
        'orig_t_ref_pct': 17.15,
        'orig_converged': False,  # max_iter hit
        'meta_filter': 0.45,      # V5 = trimON_metaON_floorOFF
    },
]


def freeze_data_snapshot(snapshot_dir):
    """Copy all data files to snapshot_dir. Returns list of copied paths."""
    os.makedirs(snapshot_dir, exist_ok=True)
    copied = []
    for rel in DATA_FILES_TO_FREEZE:
        src = os.path.join(ENGINE_DIR, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(snapshot_dir, os.path.basename(rel))
        shutil.copy2(src, dst)
        copied.append((src, dst))
    return copied


def restore_data_snapshot(copied):
    """Restore each file from its snapshot copy."""
    for src, snap in copied:
        shutil.copy2(snap, src)


def load_variant_into_working(variant):
    """Copy variant's tagged config + CSV into the working _noprod slot."""
    src_cfg = os.path.join(CONFIG_DIR, variant['config_tag'])
    src_csv = os.path.join(MODELS_DIR, variant['csv_tag'])
    if not os.path.exists(src_cfg):
        raise FileNotFoundError(f"Missing config for {variant['name']}: {src_cfg}")
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Missing CSV for {variant['name']}: {src_csv}")
    shutil.copy2(src_cfg, WORKING_CONFIG)
    shutil.copy2(src_csv, WORKING_CSV)


def extract_horizons(variant):
    """Return '<bull_h>,<bear_h>h' format for the T command."""
    bull_h = variant['orig_bull_h']
    bear_h = variant['orig_bear_h']
    # Dedupe
    hs = sorted({bull_h, bear_h})
    return ','.join(str(h) for h in hs) + 'h'


def parse_mode_t_output(stdout):
    """Pull final T winner + convergence flag from Mode T stdout.

    Expected patterns:
      'T winner: min_sell_pnl=0.50% max_hold=10h bull_shield=ON bear_shield=OFF (total +111.11% vs +93.14%, delta +4.86%)'
      'T<->G CONVERGED at iteration 3/6' or 'Max-iter reached'
      'Using per-regime gates: bull=rr14h>=6.0% OR rr20h>=5.5%, cd=10h | bear=rr8h>=3.0% OR rr12h>=2.0%, cd=16h'
    """
    result = {
        't_iterations': 0,
        't_converged': False,
        't_bull_shield': None,
        't_bear_shield': None,
        't_min_sell_pnl': None,
        't_max_hold_hours': None,
        't_best_total_pct': None,
        't_baseline_total_pct': None,
        't_bull_gate': None,
        't_bear_gate': None,
    }

    # Count iterations
    iter_matches = re.findall(r'T<->G ITERATION (\d+)/\d+', stdout)
    if iter_matches:
        result['t_iterations'] = max(int(m) for m in iter_matches)

    # Convergence
    if re.search(r'CONVERGED at iteration \d+', stdout):
        result['t_converged'] = True
    elif re.search(r'max.?iter.*reached|hit.*max.?iter', stdout, re.IGNORECASE):
        result['t_converged'] = False

    # Last T winner (grab last match)
    winners = re.findall(
        r'T winner: min_sell_pnl=([\d.]+)% max_hold=(\d+)h '
        r'bull_shield=(ON|OFF) bear_shield=(ON|OFF) '
        r'\(total ([+\-][\d.]+)% vs ([+\-][\d.]+)%',
        stdout,
    )
    if winners:
        last = winners[-1]
        result['t_min_sell_pnl'] = float(last[0])
        result['t_max_hold_hours'] = int(last[1])
        result['t_bull_shield'] = (last[2] == 'ON')
        result['t_bear_shield'] = (last[3] == 'ON')
        result['t_best_total_pct'] = float(last[4])
        result['t_baseline_total_pct'] = float(last[5])

    # Last gate config
    gate_line = None
    for m in re.finditer(r'Using per-regime gates: bull=(.+?) \| bear=(.+)', stdout):
        gate_line = m
    if gate_line:
        result['t_bull_gate'] = gate_line.group(1).strip()
        result['t_bear_gate'] = gate_line.group(2).strip()

    return result


def run_variant(variant, replay_hours, max_iter, dry_run=False):
    """Run Mode T for one variant. Returns parsed result dict."""
    horizons = extract_horizons(variant)
    cmd = [
        PYTHON, MAIN_SCRIPT, 'T', 'ETH', horizons,
        '--replay', str(replay_hours),
        '--no-persist',
        '--no-persist-keep',  # keep pre-staged variant files; do NOT reseed from live
        '--no-data-update',
        '--max-iter', str(max_iter),
    ]
    meta_p = variant.get('meta_filter')
    if meta_p is not None:
        cmd += ['--meta-filter', str(meta_p)]
    print(f'\n[{variant["name"]}] cmd: {" ".join(cmd)}')
    if dry_run:
        return {'dry_run': True}

    t_start = time.time()
    try:
        p = subprocess.run(
            cmd,
            cwd=ENGINE_DIR,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=3600,  # 1h timeout per variant
        )
        exit_code = p.returncode
        stdout = p.stdout or ''
        stderr = p.stderr or ''
    except subprocess.TimeoutExpired:
        exit_code = -1
        stdout = ''
        stderr = 'TIMEOUT after 1h'

    runtime = time.time() - t_start

    parsed = parse_mode_t_output(stdout)
    parsed['exit_code'] = exit_code
    parsed['runtime_sec'] = round(runtime, 1)
    parsed['stderr_tail'] = stderr[-500:] if stderr else ''

    # Save raw log for audit
    log_path = os.path.join(OUTPUT_DIR, f'ab_retune_t_{variant["name"]}_stdout.log')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(stdout)
    parsed['log_path'] = log_path

    # Save tuned _noprod outputs with retune tag (original variant files stay untouched)
    retune_cfg = os.path.join(CONFIG_DIR, f'regime_config_ed_retuneT_{variant["label"]}.json')
    retune_csv = os.path.join(MODELS_DIR, f'crypto_ed_production_retuneT_{variant["label"]}.csv')
    if os.path.exists(WORKING_CONFIG):
        shutil.copy2(WORKING_CONFIG, retune_cfg)
    if os.path.exists(WORKING_CSV):
        shutil.copy2(WORKING_CSV, retune_csv)
    parsed['retune_config'] = retune_cfg
    parsed['retune_csv'] = retune_csv

    return parsed


def write_comparison_csv(results, out_path):
    """Write side-by-side comparison of original vs retuned Mode T results."""
    fieldnames = [
        'variant', 'label', 'timestamp',
        # Original AB-matrix (pre b793402, 84-combo sweep)
        'orig_detector', 'orig_bull_h', 'orig_bull_conf',
        'orig_bear_h', 'orig_bear_conf',
        'orig_bull_shield', 'orig_bear_shield',
        'orig_min_sell_pct', 'orig_max_hold_h',
        'orig_converged', 'orig_t_ref_pct',
        # Retune (post b793402, 4-combo sweep pinned min_sell=0.5% / max_hold=10h)
        'retune_iterations', 'retune_converged',
        'retune_bull_shield', 'retune_bear_shield',
        'retune_min_sell_pct', 'retune_max_hold_h',
        'retune_best_total_pct', 'retune_baseline_total_pct',
        'retune_bull_gate', 'retune_bear_gate',
        'retune_exit_code', 'retune_runtime_sec',
        # Delta analysis
        'delta_best_vs_orig_tref_pp',
        'shield_flipped_bull', 'shield_flipped_bear',
        'log_path',
    ]
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        # results is a list of (variant_dict, result_dict) tuples
        # (dicts can't be dict keys — early bug from this file)
        for v, r in results:
            row = {
                'variant': v['name'],
                'label': v['label'],
                'timestamp': datetime.now().isoformat(),
                'orig_detector': v['orig_detector'],
                'orig_bull_h': v['orig_bull_h'], 'orig_bull_conf': v['orig_bull_conf'],
                'orig_bear_h': v['orig_bear_h'], 'orig_bear_conf': v['orig_bear_conf'],
                'orig_bull_shield': v['orig_t_bull_shield'],
                'orig_bear_shield': v['orig_t_bear_shield'],
                'orig_min_sell_pct': v['orig_t_min_sell'],
                'orig_max_hold_h': v['orig_t_max_hold'],
                'orig_converged': v['orig_converged'],
                'orig_t_ref_pct': v['orig_t_ref_pct'],
                'retune_iterations': r.get('t_iterations'),
                'retune_converged': r.get('t_converged'),
                'retune_bull_shield': r.get('t_bull_shield'),
                'retune_bear_shield': r.get('t_bear_shield'),
                'retune_min_sell_pct': r.get('t_min_sell_pnl'),
                'retune_max_hold_h': r.get('t_max_hold_hours'),
                'retune_best_total_pct': r.get('t_best_total_pct'),
                'retune_baseline_total_pct': r.get('t_baseline_total_pct'),
                'retune_bull_gate': r.get('t_bull_gate'),
                'retune_bear_gate': r.get('t_bear_gate'),
                'retune_exit_code': r.get('exit_code'),
                'retune_runtime_sec': r.get('runtime_sec'),
                'delta_best_vs_orig_tref_pp': (
                    round(r['t_best_total_pct'] - v['orig_t_ref_pct'], 2)
                    if r.get('t_best_total_pct') is not None else None
                ),
                'shield_flipped_bull': (
                    r.get('t_bull_shield') != v['orig_t_bull_shield']
                    if r.get('t_bull_shield') is not None else None
                ),
                'shield_flipped_bear': (
                    r.get('t_bear_shield') != v['orig_t_bear_shield']
                    if r.get('t_bear_shield') is not None else None
                ),
                'log_path': r.get('log_path', ''),
            }
            w.writerow(row)


def print_comparison_table(results):
    print('\n' + '=' * 100)
    print('  AB MATRIX RETUNE — ORIGINAL vs PINNED min_sell/max_hold')
    print('=' * 100)
    print(f"\n{'Variant':<6} {'Label':<18} {'ORIG converged':<16} {'RETUNE converged':<18} "
          f"{'ORIG t_ref':<12} {'RETUNE total':<14} {'Δ (pp)':<10} "
          f"{'shield flip bull/bear':<22}")
    print('-' * 100)
    # results is a list of (variant_dict, result_dict) tuples
    for v, r in results:
        orig_conv = 'CONVERGED' if v['orig_converged'] else 'max_iter'
        retune_conv = 'CONVERGED' if r.get('t_converged') else f'iter {r.get("t_iterations","?")}'
        orig_tref = f'{v["orig_t_ref_pct"]:+.2f}%'
        ret_total = f'{r["t_best_total_pct"]:+.2f}%' if r.get('t_best_total_pct') is not None else 'n/a'
        delta = f'{r["t_best_total_pct"] - v["orig_t_ref_pct"]:+.2f}' if r.get('t_best_total_pct') is not None else 'n/a'
        fl_b = 'Y' if r.get('t_bull_shield') != v['orig_t_bull_shield'] else 'n'
        fl_be = 'Y' if r.get('t_bear_shield') != v['orig_t_bear_shield'] else 'n'
        print(f"{v['name']:<6} {v['label']:<18} {orig_conv:<16} {retune_conv:<18} "
              f"{orig_tref:<12} {ret_total:<14} {delta:<10} bull={fl_b} bear={fl_be}")
    print('=' * 100)
    print(f'\nPer-variant full stdout saved under output/ab_retune_t_<variant>_stdout.log')
    print(f'Per-variant retuned config+CSV saved as config/regime_config_ed_retuneT_<label>.json + models/crypto_ed_production_retuneT_<label>.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', default='V1,V2,V3',
                    help='Comma-separated variant names (default: V1,V2,V3)')
    ap.add_argument('--replay', type=int, default=1440,
                    help='Replay window in hours (default: 1440 = 60d, matches AB matrix)')
    ap.add_argument('--max-iter', type=int, default=6,
                    help='Mode T max iterations (default: 6)')
    ap.add_argument('--dry-run', action='store_true', help='Print plan, do not execute')
    args = ap.parse_args()

    wanted = {v.strip().upper() for v in args.variants.split(',') if v.strip()}
    variants_to_run = [v for v in VARIANTS if v['name'].upper() in wanted]
    if not variants_to_run:
        print(f'No variants match {args.variants}. Available: {[v["name"] for v in VARIANTS]}')
        sys.exit(1)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_dir = os.path.join(DATA_DIR, f'_retune_t_snap_{ts}')
    out_csv = os.path.join(OUTPUT_DIR, f'ab_retune_t_results_{ts}.csv')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'\n=== AB MATRIX RETUNE T ===')
    print(f'  Python:      {PYTHON}')
    print(f'  Engine dir:  {ENGINE_DIR}')
    print(f'  Variants:    {[v["name"] + " (" + v["label"] + ")" for v in variants_to_run]}')
    print(f'  Replay:      {args.replay}h')
    print(f'  Max iter:    {args.max_iter}')
    print(f'  Snapshot:    {snapshot_dir}')
    print(f'  Output CSV:  {out_csv}')
    print(f'  Dry run:     {args.dry_run}')

    # Back up live prod before messing with _noprod.* (variants only touch _noprod but be safe)
    live_cfg_bak = os.path.join(CONFIG_DIR, f'_retuneT_live_backup_{ts}.json')
    live_csv_bak = os.path.join(MODELS_DIR, f'_retuneT_live_backup_{ts}.csv')
    if os.path.exists(os.path.join(CONFIG_DIR, 'regime_config_ed.json')):
        shutil.copy2(os.path.join(CONFIG_DIR, 'regime_config_ed.json'), live_cfg_bak)
    if os.path.exists(os.path.join(MODELS_DIR, 'crypto_ed_production.csv')):
        shutil.copy2(os.path.join(MODELS_DIR, 'crypto_ed_production.csv'), live_csv_bak)

    # Snapshot the working _noprod slot too so it can be restored
    working_cfg_bak = WORKING_CONFIG + f'.preretune_{ts}'
    working_csv_bak = WORKING_CSV + f'.preretune_{ts}'
    if os.path.exists(WORKING_CONFIG):
        shutil.copy2(WORKING_CONFIG, working_cfg_bak)
    if os.path.exists(WORKING_CSV):
        shutil.copy2(WORKING_CSV, working_csv_bak)

    # Freeze data
    print('\n  Freezing data snapshot...')
    copied = freeze_data_snapshot(snapshot_dir)
    print(f'  Snapshotted {len(copied)} data files')

    results = {}
    try:
        for v in variants_to_run:
            print(f'\n\n{"#" * 80}\n#  {v["name"]} ({v["label"]})\n{"#" * 80}')
            restore_data_snapshot(copied)
            try:
                load_variant_into_working(v)
            except FileNotFoundError as e:
                print(f'  SKIP — {e}')
                results[tuple(v.items())] = {'error': str(e)}
                continue
            parsed = run_variant(v, args.replay, args.max_iter, dry_run=args.dry_run)
            results[v['name']] = parsed
    finally:
        # Always restore working _noprod + clean up
        if os.path.exists(working_cfg_bak):
            shutil.copy2(working_cfg_bak, WORKING_CONFIG)
            os.remove(working_cfg_bak)
        if os.path.exists(working_csv_bak):
            shutil.copy2(working_csv_bak, WORKING_CSV)
            os.remove(working_csv_bak)
        print(f'\n  Restored original _noprod files.')

    if not args.dry_run:
        # List of (variant_dict, result_dict) tuples — dicts aren't hashable so can't be dict keys
        pairs = [(v, results.get(v['name'], {})) for v in variants_to_run]
        write_comparison_csv(pairs, out_csv)
        print(f'\n  Comparison CSV: {out_csv}')

        # Printable summary table
        print_comparison_table(pairs)

        print(f'\n  Live prod backup (safety): {live_cfg_bak}')
        print(f'  Data snapshot kept:        {snapshot_dir}  (delete after analysis)')


if __name__ == '__main__':
    main()
