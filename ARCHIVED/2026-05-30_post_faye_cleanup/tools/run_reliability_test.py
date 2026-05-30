"""tools/run_reliability_test.py — 5-variant within-engine reliability harness.

Hypothesis
----------
Three root causes make crypto Mode D/V results unreliable:
  #1 Measurement noise dominates signal     (σ=5.82pp on label-noise alone)
  #2 n_features hard cap traps optimizer    (V2: all 5 modes pick identical 5/6/7h)
  #3 Scoring rule rewards luck              (return × WR is jumpy at small N)

This harness measures whether targeted fixes for #1, #2, #3 reduce σ
and produce a strictly-better winner vs the current production engine.

Scope (Phase 1)
---------------
ETH 8h only. V2's 5/6/7h came out bit-identical across all 5 importance
methods, so 8h is the highest-information horizon. If a variant wins 8h,
Phase 2 extends it to 5,6,7h. If no variant wins 8h, the others won't move.

The 5 variants
--------------
  A baseline                  current engine, no patchers          (control)
  B multi_seed                K=5 median scoring (#1)              tests root cause #1
  C no_feature_cap            drop n_features cap to 150 (#4)      tests root cause #2
  D multi_seed_plus_no_cap    B + C combined                       tests #1+#2 stack
  E full_fix                  B + C + rpf_sqrt DSR-proxy (#3)      tests the full stack

For each variant: Mode DV ETH 8h --replay 1440 → extract Mode V winner →
run stability test against that winner (--replay 336 for speed) → record σ.

Data snapshot isolation (trader-coexistence)
---------------------------------------------
  At campaign start: snapshot data/ -> data/_reliability_snapshot_<CID>/
  Each subprocess imports _idea_patchers.v2_data_snapshot which redirects
  pd.read_csv. Trader continues using live data/ unaffected.

State + resumability
--------------------
  output/run_reliability_test_state.json: per-variant rc + elapsed + paths.
  Crashes don't lose work. Failed (rc!=0) variants stay incomplete.

Output
------
  output/run_reliability_test_<TS>.csv     per-variant comparison
  output/run_reliability_test_<TS>_summary.txt  verdict + recommendation
  models/crypto_ed_production_reliability_<VAR>.csv  per-variant winners
  logs/run_reliability_test_<TS>_<VAR>.log  per-variant Mode DV console
  logs/run_reliability_test_<TS>_<VAR>_stability.log per-variant stability test

Usage
-----
  python tools/run_reliability_test.py                # run all 5 variants
  python tools/run_reliability_test.py --status       # show progress
  python tools/run_reliability_test.py --reset        # wipe state + snapshot
  python tools/run_reliability_test.py --variants A,B # subset
  python tools/run_reliability_test.py --report-only  # rebuild report

Verdict logic (set in advance, not post-hoc)
--------------------------------------------
  σ drops below 2pp on Variant B AND ≥1 variant strictly beats A by ≥3σ
    -> ship that variant as new prod engine
  σ drops but no strict winner
    -> ship denoised engine anyway (root cause #1 fixed), keep prod alpha
  σ does NOT drop on Variant B
    -> root cause #1 hypothesis wrong; pivot to execution-gap (~17pp gap)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent.parent
os.chdir(ENGINE)

VENV_PY = sys.executable
ASSET = 'ETH'
HORIZONS_PHASE1 = [8]  # 8h-only; V2 showed 5/6/7h are bump-locked across methods
REPLAY = 1440
STABILITY_REPLAY = 336   # 2 weeks — fast σ measurement (not the full 1440h)
STABILITY_TRIALS = 5     # 1 baseline + 5 noise + up to 5 permute = ~11 total

TS = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_DIR = ENGINE / 'logs'
OUT_DIR = ENGINE / 'output'
MODELS_DIR = ENGINE / 'models'
LOG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

ORCH_LOG = LOG_DIR / f'run_reliability_test_{TS}.log'
RESULTS_CSV = OUT_DIR / f'run_reliability_test_{TS}.csv'
SUMMARY_TXT = OUT_DIR / f'run_reliability_test_{TS}_summary.txt'

STATE_FILE = OUT_DIR / 'run_reliability_test_state.json'
LOCK_FILE = OUT_DIR / 'run_reliability_test.lock'

NOPROD_CSV = MODELS_DIR / 'crypto_ed_production_noprod.csv'
DATA_DIR = ENGINE / 'data'

# Marker files used for data-snapshot integrity verification.
DATA_FILES = [
    'data/eth_hourly_data.csv',
    'data/btc_hourly_data.csv',
    'data/macro_data/derivatives_eth.csv',
    'data/macro_data/fear_greed.csv',
]

# (name, patchers, env, description)
# Patchers are listed in REVERSE order of application (each appends a layer);
# v2_data_snapshot is always loaded first by the orchestrator.
VARIANTS = [
    ('A_baseline',           [],                                                                        {},                  'Reference: current production engine, no patches'),
    ('B_multi_seed',         ['_idea_patchers.reliability_multi_seed'],                                  {'RELIABILITY_K': '5'},  'K=5 multi-seed median Optuna scoring (tests #1: noise floor)'),
    ('C_no_feature_cap',     ['_idea_patchers.reliability_no_feature_cap'],                              {},                  'Drop n_features cap; expand GRID_FEATURES (tests #2: search shape)'),
    ('D_multi_seed_plus_cap', ['_idea_patchers.reliability_multi_seed',
                              '_idea_patchers.reliability_no_feature_cap'],                              {'RELIABILITY_K': '5'},  'B + C combined (tests #1+#2 stack)'),
    ('E_full_fix',           ['_idea_patchers.reliability_multi_seed',
                              '_idea_patchers.reliability_no_feature_cap',
                              '_idea_patchers.reliability_dsr_scoring'],                                 {'RELIABILITY_K': '5'},  'B + C + rpf_sqrt scoring (tests #1+#2+#3: full stack)'),
]


def log(msg: str):
    line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(ORCH_LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def data_freeze_marker(base_dir: Path | None = None) -> dict:
    out = {}
    for fname in DATA_FILES:
        if base_dir is not None:
            rel = fname[5:] if fname.startswith('data/') else fname
            p = base_dir / rel
        else:
            p = ENGINE / fname
        if not p.exists():
            out[fname] = 'missing'
            continue
        with open(p, 'rb') as f:
            h = hashlib.md5(f.read()).hexdigest()
        out[fname] = {
            'md5': h[:12],
            'mtime': datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds'),
            'size_kb': p.stat().st_size // 1024,
        }
    return out


def snapshot_dir_path(campaign_id: str) -> Path:
    return DATA_DIR / f'_reliability_snapshot_{campaign_id}'


def create_data_snapshot(campaign_id: str) -> Path:
    snap = snapshot_dir_path(campaign_id)
    if snap.exists():
        all_present = all((snap / fname[5:]).exists() for fname in DATA_FILES if 'data/' in fname)
        if all_present:
            log(f'  Snapshot already exists at {snap.name} — reusing')
            return snap

    log(f'  Creating data snapshot at {snap.name}/')
    snap.mkdir(exist_ok=True)
    n_files, n_bytes = 0, 0
    for root, dirs, files in os.walk(DATA_DIR):
        dirs[:] = [d for d in dirs if not (d.startswith('_v2_snapshot_') or d.startswith('_reliability_snapshot_'))]
        rel_root = Path(root).resolve().relative_to(DATA_DIR.resolve())
        if str(rel_root).startswith('_v2_snapshot_') or str(rel_root).startswith('_reliability_snapshot_'):
            continue
        dest_root = snap / rel_root
        dest_root.mkdir(parents=True, exist_ok=True)
        for fname in files:
            src = Path(root) / fname
            dst = dest_root / fname
            shutil.copy2(src, dst)
            n_files += 1
            n_bytes += src.stat().st_size
    log(f'  Snapshot complete: {n_files} files, {n_bytes/(1024*1024):.1f} MB')
    return snap


def freeze_diff(prev: dict, cur: dict) -> list[str]:
    drift = []
    for fname in DATA_FILES:
        a = prev.get(fname)
        b = cur.get(fname)
        if not isinstance(a, dict) or not isinstance(b, dict):
            if a != b:
                drift.append(f'{fname}: {a!r} -> {b!r}')
            continue
        if a.get('md5') != b.get('md5'):
            drift.append(f'{fname}: md5 {a.get("md5")} -> {b.get("md5")}')
    return drift


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {'started_at': None, 'data_freeze': None, 'completed': {}}
    try:
        return json.loads(STATE_FILE.read_text(encoding='utf-8'))
    except Exception:
        return {'started_at': None, 'data_freeze': None, 'completed': {}}


def save_state(state: dict):
    tmp = STATE_FILE.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(state, indent=2), encoding='utf-8')
    tmp.replace(STATE_FILE)


def acquire_lock() -> bool:
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text())
            try:
                os.kill(pid, 0)
                return False
            except (OSError, ProcessLookupError):
                pass
        except Exception:
            pass
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def release_lock():
    if LOCK_FILE.exists():
        try:
            LOCK_FILE.unlink()
        except Exception:
            pass


def run_mode_dv(name: str, patchers: list[str], env_overrides: dict, snapshot_path: Path) -> dict:
    """Run Mode DV ETH 8h with the variant's patchers applied.
    Returns dict with rc, elapsed_s, log path, snapshot path."""
    log(f'  >> RUNNING variant={name}  patchers={patchers}')
    mode_log = LOG_DIR / f'run_reliability_test_{TS}_{name}.log'

    py_init = (
        f"import sys, os; "
        f"sys.path.insert(0, r'{ENGINE}'); "
        f"os.chdir(r'{ENGINE}'); "
    )
    # ALWAYS load snapshot patcher first so pd.read_csv is redirected before
    # the engine begins reading data.
    py_init += "import _idea_patchers.v2_data_snapshot; "
    for patcher in patchers:
        py_init += f"import {patcher}; "
    py_init += "import crypto_trading_system_ed as e; e.main()"

    horizons_arg = ','.join(f'{h}h' for h in HORIZONS_PHASE1)
    # ENGINE CLI PARSER BUG WORKAROUND: 'ETH'.lower().endswith('h') is True, so
    # the engine's positional-arg parser at line ~7347 enters the horizon
    # detection branch, fails the isdigit check, consumes the elif WITHOUT
    # setting horizons or assets, and assets_list defaults to ALL 9 assets.
    # Appending a trailing comma forces the parser to fall through to the
    # ELSE branch which correctly identifies ETH as an asset. Same trick used
    # in crypto_trading_system_ed_robust.py for the same reason.
    asset_arg = f'{ASSET},' if ASSET.lower().endswith('h') else ASSET
    cmd = [
        VENV_PY, '-c',
        f"import sys; "
        f"sys.argv=['crypto_trading_system_ed.py', 'DV', '{asset_arg}', '{horizons_arg}', "
        f"'--replay', '{REPLAY}', '--no-persist', '--no-data-update', "
        f"'--grid-tag', 'REL_{name.upper()}']; "
        + py_init
    ]

    env = os.environ.copy()
    env.update(env_overrides)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['V2_DATA_SNAPSHOT'] = str(snapshot_path)  # reuse the V2 patcher's env var

    t0 = time.time()
    with open(mode_log, 'w', encoding='utf-8') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ENGINE).returncode
    elapsed = time.time() - t0

    snapshot = MODELS_DIR / f'crypto_ed_production_reliability_{name}.csv'
    snapshot_ok = False
    if rc == 0 and NOPROD_CSV.exists():
        shutil.copy2(NOPROD_CSV, snapshot)
        snapshot_ok = True

    log(f'  >> DONE variant={name} rc={rc} elapsed={elapsed/60:.1f}min snapshot={"ok" if snapshot_ok else "MISSING"}')
    return {
        'name': name, 'rc': rc, 'elapsed_s': elapsed,
        'log': str(mode_log),
        'snapshot': str(snapshot) if snapshot_ok else None,
        'env': env_overrides,
        'patchers': patchers,
    }


def run_stability_test(name: str, prod_csv: str, snapshot_path: Path) -> dict:
    """Run feature_stability_test.py against the variant's winner CSV.
    Returns dict with sigma, mean_ret, baseline_ret, n_trials_valid, elapsed_s."""
    log(f'  >> STABILITY-TEST variant={name}  csv={Path(prod_csv).name}')
    stab_log = LOG_DIR / f'run_reliability_test_{TS}_{name}_stability.log'

    cmd = [
        VENV_PY, '-u', str(ENGINE / 'tools' / 'feature_stability_test.py'),
        '--csv', prod_csv, '--asset', ASSET, '--horizon', str(HORIZONS_PHASE1[0]),
        '--trials', str(STABILITY_TRIALS),
    ]

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['V2_DATA_SNAPSHOT'] = str(snapshot_path)
    # Override stability test's REPLAY_HOURS via env (faster σ measurement).
    # The test reads REPLAY_HOURS from module global; we inject it via a side
    # channel — the runner code interpolates {replay} at import. To override,
    # set STAB_REPLAY env var and tests pick it up if supported.
    env['STAB_REPLAY'] = str(STABILITY_REPLAY)

    t0 = time.time()
    with open(stab_log, 'w', encoding='utf-8') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ENGINE).returncode
    elapsed = time.time() - t0

    # Parse the stability log for the σ result.
    sigma = float('nan')
    mean_ret = float('nan')
    baseline_ret = float('nan')
    n_valid = 0
    verdict = '?'
    if stab_log.exists():
        text = stab_log.read_text(encoding='utf-8', errors='replace')
        import re
        m = re.search(r'Returns:\s+mean=([+\-\d\.]+)%\s+σ=([\d\.]+)pp', text)
        if m:
            mean_ret = float(m.group(1))
            sigma = float(m.group(2))
        m = re.search(r'Trials with valid results:\s+(\d+)/', text)
        if m:
            n_valid = int(m.group(1))
        m = re.search(r'Baseline \(no perturbation\):\s+return=([+\-\d\.]+)%', text)
        if m:
            baseline_ret = float(m.group(1))
        if 'VERDICT: STABLE' in text:
            verdict = 'STABLE'
        elif 'VERDICT: MARGINAL' in text:
            verdict = 'MARGINAL'
        elif 'VERDICT: UNSTABLE' in text:
            verdict = 'UNSTABLE'

    log(f'  >> STABILITY-DONE variant={name} rc={rc} σ={sigma:.2f}pp mean={mean_ret:+.2f}% '
        f'baseline={baseline_ret:+.2f}% n_valid={n_valid} verdict={verdict} '
        f'elapsed={elapsed/60:.1f}min')
    return {
        'rc': rc, 'elapsed_s': elapsed, 'log': str(stab_log),
        'sigma_pp': sigma, 'mean_ret_pct': mean_ret,
        'baseline_ret_pct': baseline_ret,
        'n_trials_valid': n_valid, 'verdict': verdict,
    }


def parse_winner(prod_csv_path: str) -> dict | None:
    """Extract the ETH 8h Mode V winner from a production CSV."""
    if not prod_csv_path or not Path(prod_csv_path).exists():
        return None
    df = pd.read_csv(prod_csv_path)
    rows = df[(df['coin'] == ASSET) & (df['horizon'] == HORIZONS_PHASE1[0])]
    if not len(rows):
        return None
    winner = rows.nlargest(1, 'combined_score').iloc[0]
    return {
        'combined_score': float(winner['combined_score']),
        'return_pct': float(winner['return_pct']),
        'accuracy': float(winner['accuracy']) if 'accuracy' in winner else float('nan'),
        'combo': winner['best_combo'],
        'window': int(winner['best_window']),
        'n_features': int(winner['n_features']),
        'gamma': float(winner['gamma']),
        'features': winner['optimal_features'],
    }


def build_report(results: list[dict]):
    rows = []
    baseline_winner = None
    baseline_sigma = None
    for r in results:
        winner = parse_winner(r.get('snapshot'))
        stab = r.get('stability', {})
        row = {
            'variant': r['name'],
            'rc_mode_dv': r['rc'],
            'elapsed_dv_min': round(r['elapsed_s'] / 60, 1),
            'rc_stability': stab.get('rc', None),
            'elapsed_stab_min': round(stab.get('elapsed_s', 0) / 60, 1),
            'winner_cs': winner['combined_score'] if winner else float('nan'),
            'winner_ret': winner['return_pct'] if winner else float('nan'),
            'winner_combo': winner['combo'] if winner else '',
            'winner_window': winner['window'] if winner else 0,
            'winner_n_feat': winner['n_features'] if winner else 0,
            'winner_gamma': winner['gamma'] if winner else 0.0,
            'sigma_pp': stab.get('sigma_pp', float('nan')),
            'stab_mean_ret': stab.get('mean_ret_pct', float('nan')),
            'stab_baseline_ret': stab.get('baseline_ret_pct', float('nan')),
            'n_trials_valid': stab.get('n_trials_valid', 0),
            'stab_verdict': stab.get('verdict', '?'),
        }
        if r['name'] == 'A_baseline':
            baseline_winner = winner
            baseline_sigma = stab.get('sigma_pp', float('nan'))
        rows.append(row)

    # Compute deltas vs baseline.
    # IMPORTANT: comparison metric is `return_pct` (realized backtest return),
    # NOT `combined_score`. Variant E uses OPTUNA_METRIC='rpf_sqrt' which puts
    # its combined_score on a different scale (raw_pf*sqrt(trades) ~ 10-30
    # vs APF ~ 1-50). return_pct is scale-invariant and is the actual outcome
    # we care about. combined_score is kept in the row for context only.
    if baseline_winner is not None:
        b_ret = baseline_winner['return_pct']
        b_cs = baseline_winner['combined_score']
        for row in rows:
            if not pd.isna(row['winner_ret']):
                row['ret_d_vs_A'] = row['winner_ret'] - b_ret
            else:
                row['ret_d_vs_A'] = float('nan')
            if not pd.isna(row['winner_cs']):
                row['cs_d_vs_A'] = row['winner_cs'] - b_cs
            else:
                row['cs_d_vs_A'] = float('nan')
            if not pd.isna(row['sigma_pp']) and not pd.isna(baseline_sigma):
                row['sigma_d_vs_A'] = row['sigma_pp'] - baseline_sigma
            else:
                row['sigma_d_vs_A'] = float('nan')

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS_CSV, index=False)
    log(f'CSV: {RESULTS_CSV}')

    # Build text summary
    lines = []
    lines.append('=' * 110)
    lines.append(f'  RELIABILITY TEST — verdict ({TS})')
    lines.append(f'  Asset: {ASSET}  Horizon (Phase 1): {HORIZONS_PHASE1}  Replay: {REPLAY}h')
    lines.append(f'  Stability test replay: {STABILITY_REPLAY}h  trials: {STABILITY_TRIALS}')
    lines.append('=' * 110)
    lines.append('')
    hdr = f'  {"variant":<26}  {"ret%":>7}  {"Δret":>7}  {"feat":>4}  {"gamma":>6}  {"σpp":>5}  {"Δσ":>6}  {"cs(opt)":>8}  {"verdict":>9}  {"dv min":>6}  {"stab min":>8}'
    lines.append(hdr)
    lines.append('-' * len(hdr))
    for row in rows:
        ret = f'{row["winner_ret"]:+7.2f}' if not pd.isna(row['winner_ret']) else '  n/a  '
        d_ret = f'{row.get("ret_d_vs_A", float("nan")):+7.2f}' if not pd.isna(row.get('ret_d_vs_A', float('nan'))) else '  n/a  '
        feat = f'{row["winner_n_feat"]:4d}' if row['winner_n_feat'] else '   ?'
        g = f'{row["winner_gamma"]:.4f}' if row['winner_gamma'] else ' n/a '
        s = f'{row["sigma_pp"]:5.2f}' if not pd.isna(row['sigma_pp']) else ' n/a '
        d_s = f'{row.get("sigma_d_vs_A", float("nan")):+6.2f}' if not pd.isna(row.get('sigma_d_vs_A', float('nan'))) else ' n/a  '
        cs = f'{row["winner_cs"]:+8.3f}' if not pd.isna(row['winner_cs']) else '   n/a  '
        v = row['stab_verdict'] or '?'
        lines.append(f'  {row["variant"]:<26}  {ret}  {d_ret}  {feat}  {g}  {s}  {d_s}  {cs}  {v:>9}  {row["elapsed_dv_min"]:>6.1f}  {row["elapsed_stab_min"]:>8.1f}')

    lines.append('')
    lines.append('Verdict reading')
    lines.append('---------------')
    if baseline_sigma is not None and not pd.isna(baseline_sigma):
        # Check if any variant beat baseline + strict win
        # Comparison metric is `return_pct` (scale-invariant across Optuna
        # objectives). cs_d_vs_A is misleading for Variant E (rpf_sqrt scale).
        winners = []
        for row in rows:
            if row['variant'] == 'A_baseline':
                continue
            if pd.isna(row['sigma_pp']) or pd.isna(row.get('ret_d_vs_A', float('nan'))):
                continue
            if row['sigma_pp'] < 2.0 and row['ret_d_vs_A'] > 3 * baseline_sigma:
                winners.append((row['variant'], row['ret_d_vs_A'], row['sigma_pp']))

        if winners:
            lines.append(f'  CLEAR WINNER: {len(winners)} variant(s) drop σ < 2pp AND beat baseline return by ≥ 3·σ_A ({3*baseline_sigma:.1f}pp):')
            for name, d_ret, s in winners:
                lines.append(f'    {name}  Δret={d_ret:+.2f}pp  σ={s:.2f}pp')
            lines.append('  Action: promote winning variant\'s patchers to a production engine fork')
            lines.append('         then run Phase 2 (expand to 5,6,7h) and Phase 3 (full HRST).')
        else:
            # Did σ drop on the multi-seed variant?
            b_row = next((r for r in rows if r['variant'] == 'B_multi_seed'), None)
            if b_row and not pd.isna(b_row['sigma_pp']) and b_row['sigma_pp'] < baseline_sigma * 0.5:
                lines.append(f'  σ DROPPED ({baseline_sigma:.1f} → {b_row["sigma_pp"]:.1f}pp on Variant B) but no strict winner.')
                lines.append('  Action: ship denoised engine anyway (root cause #1 fixed), retain current alpha.')
                lines.append('         Pivot remaining work to execution-gap research (~17pp gap to live).')
            else:
                lines.append(f'  σ did NOT drop materially (baseline σ={baseline_sigma:.2f}pp).')
                lines.append('  Action: root cause #1 hypothesis was wrong. Pivot to execution-gap work')
                lines.append('         OR re-examine the patcher implementations.')
    else:
        lines.append('  Baseline σ not parseable; cannot compute verdict programmatically.')
        lines.append('  Inspect logs manually.')

    lines.append('')
    summary = '\n'.join(lines)
    print(summary)
    SUMMARY_TXT.write_text(summary, encoding='utf-8')
    log(f'SUMMARY: {SUMMARY_TXT}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variants', help='comma-separated subset (e.g. A_baseline,B_multi_seed)')
    p.add_argument('--skip', help='comma-separated variants to skip')
    p.add_argument('--reset', action='store_true', help='wipe state + snapshot + per-variant CSVs')
    p.add_argument('--report-only', action='store_true', help='rebuild report from existing snapshots')
    p.add_argument('--status', action='store_true', help='show progress then exit')
    p.add_argument('--skip-stability', action='store_true', help='skip stability test per variant (faster)')
    args = p.parse_args()

    if args.reset:
        prev = load_state()
        prev_cid = prev.get('campaign_id')
        if prev_cid:
            prev_snap = snapshot_dir_path(prev_cid)
            if prev_snap.exists():
                shutil.rmtree(prev_snap)
                print(f'wiped snapshot {prev_snap}')
        if STATE_FILE.exists():
            STATE_FILE.unlink()
            print(f'wiped {STATE_FILE}')
        for name, *_ in VARIANTS:
            snap = MODELS_DIR / f'crypto_ed_production_reliability_{name}.csv'
            if snap.exists():
                snap.unlink()
                print(f'wiped {snap}')
        return

    if args.status:
        state = load_state()
        completed = state.get('completed', {})
        for name, *_ in VARIANTS:
            snap = MODELS_DIR / f'crypto_ed_production_reliability_{name}.csv'
            status = '✓' if (name in completed and snap.exists() and completed[name].get('rc') == 0) else '·'
            stab_done = '✓' if (name in completed and 'stability' in completed[name] and completed[name]['stability'].get('sigma_pp') is not None) else '·'
            print(f'  DV={status}  STAB={stab_done}  {name}')
        return

    if not acquire_lock():
        print(f'Another instance is running (PID file {LOCK_FILE}). Aborting.')
        print(f'If you are sure no instance is running, delete {LOCK_FILE} and retry.')
        sys.exit(2)

    try:
        log('=' * 80)
        log(f'RUN RELIABILITY TEST — start {datetime.now()}')
        log(f'Orchestrator log: {ORCH_LOG}')
        log(f'Variants: {[v[0] for v in VARIANTS]}')
        log(f'Phase 1 horizon: {HORIZONS_PHASE1}  replay: {REPLAY}h')
        log(f'Stability test replay: {STABILITY_REPLAY}h  trials: {STABILITY_TRIALS}')
        log('=' * 80)

        state = load_state()
        if state.get('started_at') is None:
            state['started_at'] = datetime.now().isoformat(timespec='seconds')
            campaign_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            state['campaign_id'] = campaign_id
            log(f'NEW CAMPAIGN id={campaign_id}')
            log('Taking data snapshot for trader-coexistence...')
            snap_path = create_data_snapshot(campaign_id)
            initial_freeze = data_freeze_marker(base_dir=snap_path)
            state['data_freeze'] = initial_freeze
            state['snapshot_dir'] = str(snap_path)
            log(f'Initial data freeze markers (from snapshot {snap_path.name}):')
            for fname, meta in initial_freeze.items():
                log(f'  {fname}: {meta}')
            save_state(state)
        else:
            campaign_id = state.get('campaign_id')
            if not campaign_id:
                log('[ERROR] State file has no campaign_id. Use --reset to start fresh.')
                sys.exit(3)
            log(f'RESUMING — campaign id={campaign_id} started {state["started_at"]}')
            snap_path = snapshot_dir_path(campaign_id)
            if not snap_path.exists():
                log(f'[WARN] Snapshot dir missing at {snap_path} — recreating')
                snap_path = create_data_snapshot(campaign_id)
            cur_freeze = data_freeze_marker(base_dir=snap_path)
            drift = freeze_diff(state.get('data_freeze') or {}, cur_freeze)
            if drift:
                log('[ERROR] SNAPSHOT INTEGRITY DRIFT — snapshot files changed since campaign started')
                for d in drift:
                    log(f'  {d}')
                log('Use --reset to start fresh.')
                sys.exit(3)
            log(f'Snapshot integrity OK ({snap_path.name})')

        # Filter variants
        if args.variants:
            wanted = {v.strip() for v in args.variants.split(',')}
            variants = [v for v in VARIANTS if v[0] in wanted]
        else:
            variants = list(VARIANTS)
        if args.skip:
            skipped = {v.strip() for v in args.skip.split(',')}
            variants = [v for v in variants if v[0] not in skipped]

        completed = state.get('completed', {})

        if args.report_only:
            log('--report-only: reading existing snapshots')
            results = []
            for name, patchers, env, desc in VARIANTS:
                snap = MODELS_DIR / f'crypto_ed_production_reliability_{name}.csv'
                if snap.exists():
                    rec = completed.get(name, {})
                    results.append({
                        'name': name, 'rc': 0,
                        'elapsed_s': rec.get('elapsed_s', 0),
                        'log': rec.get('log', ''),
                        'snapshot': str(snap),
                        'env': env, 'patchers': patchers,
                        'stability': rec.get('stability', {}),
                    })
            build_report(results)
            return

        # Determine which variants need to run
        pending = []
        skipped_resume = []
        for v in variants:
            name = v[0]
            snap = MODELS_DIR / f'crypto_ed_production_reliability_{name}.csv'
            rec = completed.get(name, {})
            dv_done = (rec.get('rc') == 0 and snap.exists())
            stab_done = (rec.get('stability', {}).get('sigma_pp') is not None) if not args.skip_stability else True
            if dv_done and stab_done:
                skipped_resume.append(name)
            else:
                if not dv_done and name in completed:
                    log(f'  [WARN] state says {name} done but snapshot missing or rc!=0 — re-running')
                pending.append(v)

        if skipped_resume:
            log(f'AUTO-RESUME: skipping {len(skipped_resume)} fully-completed variants: {skipped_resume}')
        log(f'Variants to run: {[v[0] for v in pending]}')

        results = []
        # Carry over already-completed
        for name, patchers, env, desc in VARIANTS:
            rec = completed.get(name, {})
            snap = MODELS_DIR / f'crypto_ed_production_reliability_{name}.csv'
            if rec.get('rc') == 0 and snap.exists():
                results.append({
                    'name': name, 'rc': 0,
                    'elapsed_s': rec.get('elapsed_s', 0),
                    'log': rec.get('log', ''),
                    'snapshot': str(snap),
                    'env': env, 'patchers': patchers,
                    'stability': rec.get('stability', {}),
                })

        for name, patchers, env, desc in pending:
            # Pre-flight: snapshot integrity
            cur_freeze = data_freeze_marker(base_dir=snap_path)
            drift = freeze_diff(state.get('data_freeze') or {}, cur_freeze)
            if drift:
                log('[ERROR] snapshot drift detected before variant — aborting')
                for d in drift:
                    log(f'  {d}')
                sys.exit(3)

            rec = completed.get(name, {})
            snap = MODELS_DIR / f'crypto_ed_production_reliability_{name}.csv'
            dv_done = (rec.get('rc') == 0 and snap.exists())

            # 1. Mode DV (unless already done)
            if dv_done:
                dv_result = {
                    'name': name, 'rc': 0,
                    'elapsed_s': rec.get('elapsed_s', 0),
                    'log': rec.get('log', ''),
                    'snapshot': str(snap),
                    'env': env, 'patchers': patchers,
                }
                log(f'  Mode DV already done for {name} — skipping to stability test')
            else:
                dv_result = run_mode_dv(name, patchers, env, snap_path)
                rec.update({
                    'rc': dv_result['rc'],
                    'elapsed_s': dv_result['elapsed_s'],
                    'log': dv_result['log'],
                    'completed_at': datetime.now().isoformat(timespec='seconds'),
                    'env': env, 'patchers': patchers,
                })
                completed[name] = rec
                state['updated_at'] = datetime.now().isoformat(timespec='seconds')
                save_state(state)

                if dv_result['rc'] != 0:
                    log(f'  [SKIP STABILITY] Mode DV failed (rc={dv_result["rc"]}) — no stability test')
                    results.append({**dv_result, 'stability': {}})
                    continue

            # 2. Stability test on the winner (unless skipped)
            if args.skip_stability:
                dv_result['stability'] = {}
            else:
                stab_result = run_stability_test(name, dv_result['snapshot'], snap_path)
                dv_result['stability'] = stab_result
                rec['stability'] = stab_result
                completed[name] = rec
                state['updated_at'] = datetime.now().isoformat(timespec='seconds')
                save_state(state)

            results.append(dv_result)

        # Report
        build_report(results)

    finally:
        release_lock()


if __name__ == '__main__':
    main()
