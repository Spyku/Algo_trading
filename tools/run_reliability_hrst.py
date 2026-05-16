"""tools/run_reliability_hrst.py — HRST validation of B/C reliability variants.

Phase 2/3 follow-up to Phase 1 (tools/run_reliability_test.py). For ONE variant
(B_multi_seed OR C_no_feature_cap) per invocation, runs the full HRST pipeline
on ETH 5,6,7,8h --replay 1440 with the variant's patchers applied.

Goal
----
Phase 1 (2026-05-15) showed B and C each beat A_baseline by +5pp on the Mode V
combined_score metric (8h only). Per CLAUDE.md verdict logic, that triggers:
  "promote winning variant's patchers → Phase 2 (5,6,7h) → Phase 3 (full HRST)"

This script collapses Phase 2 + Phase 3 into a single HRST run — HRST already
sweeps Mode D/V at 5,6,7,8h internally, then chains Mode S regime selection and
Mode T threshold sweep. The final Mode T total is directly comparable to the
May 6 production HRST baseline (+76.77%).

Trader-coexistence
------------------
- Snapshot at data/_reliability_hrst_snapshot_<machine>_<CID>/ — pd.read_csv
  redirected via _idea_patchers.v2_data_snapshot. Trader stays active.
- --no-persist: production CSV/regime config NEVER touched. All writes go to
  *_noprod.csv / *_noprod.json.
- Per-machine state/lock — Desktop and Laptop can run two variants in parallel
  without colliding on shared files (state/lock keyed by --machine).

Usage
-----
  # Two terminals, run in parallel:
  python tools/run_reliability_hrst.py --variant B_multi_seed --machine desktop
  python tools/run_reliability_hrst.py --variant C_no_feature_cap --machine laptop

  # Status / control:
  python tools/run_reliability_hrst.py --machine desktop --status
  python tools/run_reliability_hrst.py --machine desktop --report-only
  python tools/run_reliability_hrst.py --machine desktop --reset

Verdict logic (set in advance, not post-hoc)
--------------------------------------------
  Mode T total ≥ +81.77% (≥ production +76.77% + 5pp)   → SHIP
  Mode T total within ±5pp of production                → MARGINAL
  Mode T total < +71.77% (< production − 5pp)           → DEAD

Outputs
-------
  output/run_reliability_hrst_<machine>_<TS>_<variant>.csv     summary row
  output/run_reliability_hrst_<machine>_<TS>_<variant>.txt     verdict + parsed metrics
  logs/run_reliability_hrst_<machine>_<TS>_<variant>.log       engine subprocess console
  models/crypto_ed_grid_ETH_<h>h_REL_HRST_<VARIANT>.csv         per-horizon grids
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent
os.chdir(ENGINE)

VENV_PY = sys.executable
ASSET = 'ETH'
HORIZONS = [5, 6, 7, 8]
REPLAY = 1440

# Variant definitions: name → (patchers, env_overrides, description)
VARIANTS = {
    'A_baseline': (
        [],
        {},
        'Reference: current production engine, no patchers',
    ),
    'B_multi_seed': (
        ['_idea_patchers.reliability_multi_seed'],
        {'RELIABILITY_K': '5'},
        'K=5 multi-seed median Optuna scoring',
    ),
    'C_no_feature_cap': (
        ['_idea_patchers.reliability_no_feature_cap'],
        {},
        'Drop n_features cap; expand GRID_FEATURES',
    ),
    'F_optimized': (
        # Stack order matters: v2_data_snapshot is loaded by run_hrst() first.
        # Then multi_seed (K=5) base, then the 3 Mode V architectural fixes:
        #   - optuna_objective_align: OPTUNA_METRIC -> ret_wr (matches winner selector)
        #   - bo_exploration:         TPESampler -> CmaEsSampler (more exploration)
        #   - expand_grid:            GRID_WINDOWS/GRID_FEATURES broader for diversity
        [
            '_idea_patchers.reliability_multi_seed',
            '_idea_patchers.reliability_optuna_objective_align',
            '_idea_patchers.reliability_bo_exploration',
            '_idea_patchers.reliability_expand_grid',
        ],
        {'RELIABILITY_K': '5'},
        'B + objective alignment + BO exploration + expanded grid (Mode V architectural fix)',
    ),
}

# Production HRST baseline (2026-05-06 promotion, Mode T total)
PRODUCTION_HRST_REF = 76.77
SHIP_THRESHOLD_PP = 5.0

LOG_DIR = ENGINE / 'logs'
OUT_DIR = ENGINE / 'output'
MODELS_DIR = ENGINE / 'models'
DATA_DIR = ENGINE / 'data'
LOG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# Marker files used for snapshot integrity verification.
DATA_FILES = [
    'data/eth_hourly_data.csv',
    'data/btc_hourly_data.csv',
    'data/macro_data/derivatives_eth.csv',
    'data/macro_data/fear_greed.csv',
]

NOPROD_CSV = MODELS_DIR / 'crypto_ed_production_noprod.csv'
NOPROD_CFG = ENGINE / 'config' / 'regime_config_ed_noprod.json'


def state_file(machine: str) -> Path:
    return OUT_DIR / f'run_reliability_hrst_{machine}_state.json'


def lock_file(machine: str) -> Path:
    return OUT_DIR / f'run_reliability_hrst_{machine}.lock'


def snapshot_dir_for(machine: str, campaign_id: str) -> Path:
    return DATA_DIR / f'_reliability_hrst_snapshot_{machine}_{campaign_id}'


def log_path(machine: str, ts: str, variant: str) -> Path:
    return LOG_DIR / f'run_reliability_hrst_{machine}_{ts}_{variant}.log'


def orch_log_path(machine: str, ts: str) -> Path:
    return LOG_DIR / f'run_reliability_hrst_{machine}_{ts}_orch.log'


def make_logger(orch_log: Path):
    def log(msg: str):
        line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(orch_log, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    return log


# ------------------------- snapshot management -------------------------

def data_freeze_marker(base_dir: Path | None = None) -> dict:
    out: dict = {}
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


def create_data_snapshot(snap: Path, log) -> Path:
    if snap.exists():
        all_present = all((snap / fname[5:]).exists() for fname in DATA_FILES)
        if all_present:
            log(f'  Snapshot already exists at {snap.name} — reusing')
            return snap

    log(f'  Creating data snapshot at {snap.name}/')
    snap.mkdir(exist_ok=True)
    n_files, n_bytes = 0, 0
    for root, dirs, files in os.walk(DATA_DIR):
        # Skip nested snapshot dirs (V2's, Phase 1's, and our own).
        dirs[:] = [d for d in dirs
                   if not (d.startswith('_v2_snapshot_')
                           or d.startswith('_reliability_snapshot_')
                           or d.startswith('_reliability_hrst_snapshot_'))]
        rel_root = Path(root).resolve().relative_to(DATA_DIR.resolve())
        s = str(rel_root)
        if (s.startswith('_v2_snapshot_')
                or s.startswith('_reliability_snapshot_')
                or s.startswith('_reliability_hrst_snapshot_')):
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


# ------------------------- state + lock -------------------------

def load_state(machine: str) -> dict:
    sf = state_file(machine)
    if not sf.exists():
        return {'started_at': None, 'campaign_id': None, 'data_freeze': None,
                'snapshot_dir': None, 'variant': None, 'completed': False, 'result': None}
    try:
        return json.loads(sf.read_text(encoding='utf-8'))
    except Exception:
        return {'started_at': None, 'campaign_id': None, 'data_freeze': None,
                'snapshot_dir': None, 'variant': None, 'completed': False, 'result': None}


def save_state(machine: str, state: dict):
    sf = state_file(machine)
    tmp = sf.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(state, indent=2), encoding='utf-8')
    tmp.replace(sf)


def acquire_lock(machine: str) -> bool:
    lf = lock_file(machine)
    if lf.exists():
        try:
            pid = int(lf.read_text())
            try:
                os.kill(pid, 0)
                return False
            except (OSError, ProcessLookupError):
                pass
        except Exception:
            pass
    lf.write_text(str(os.getpid()))
    return True


def release_lock(machine: str):
    lf = lock_file(machine)
    if lf.exists():
        try:
            lf.unlink()
        except Exception:
            pass


# ------------------------- HRST runner -------------------------

def run_hrst(variant: str, patchers: list[str], env_overrides: dict,
             snapshot_path: Path, machine: str, ts: str, log) -> dict:
    """Run HRST ETH 5,6,7,8h with the variant's patchers applied. Returns dict
    with rc, elapsed_s, log path."""
    log(f'  >> RUNNING HRST variant={variant}  patchers={patchers}')
    engine_log = log_path(machine, ts, variant)

    py_init = (
        f"import sys, os; "
        f"sys.path.insert(0, r'{ENGINE}'); "
        f"os.chdir(r'{ENGINE}'); "
    )
    # Snapshot patcher first so pd.read_csv is redirected before engine reads data.
    py_init += "import _idea_patchers.v2_data_snapshot; "
    for patcher in patchers:
        py_init += f"import {patcher}; "
    py_init += "import crypto_trading_system_ed as e; e.main()"

    horizons_arg = ','.join(f'{h}h' for h in HORIZONS)
    # ENGINE CLI PARSER BUG: 'ETH'.lower().endswith('h') is True → trailing comma
    # forces parser to treat as comma-separated assets. Same workaround as Phase 1.
    asset_arg = f'{ASSET},' if ASSET.lower().endswith('h') else ASSET

    grid_tag = f'REL_HRST_{variant.upper()}'
    # Auto-skip Mode D if all per-horizon grid CSVs for this variant already exist
    # (e.g. previous run crashed during Mode V). The engine's --skip flag reuses
    # existing grids instead of re-running the ~4h Mode D phase.
    skip_args: list[str] = []
    if all((MODELS_DIR / f'crypto_ed_grid_ETH_{h}h_{grid_tag}.csv').exists() for h in HORIZONS):
        skip_args = ["'--skip'"]
        log(f'  All Mode D grids for {variant} already on disk — passing --skip to engine')
    cmd = [
        VENV_PY, '-c',
        f"import sys; "
        f"sys.argv=['crypto_trading_system_ed.py', 'HRST', '{asset_arg}', '{horizons_arg}', "
        f"'--replay', '{REPLAY}', '--no-persist', '--no-data-update', "
        f"'--grid-tag', '{grid_tag}'" + (', ' + ', '.join(skip_args) if skip_args else '') + "]; "
        + py_init
    ]

    env = os.environ.copy()
    env.update(env_overrides)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['V2_DATA_SNAPSHOT'] = str(snapshot_path)

    t0 = time.time()
    with open(engine_log, 'w', encoding='utf-8') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                            cwd=ENGINE).returncode
    elapsed = time.time() - t0

    log(f'  >> DONE variant={variant} rc={rc} elapsed={elapsed/60:.1f}min')
    return {
        'variant': variant, 'rc': rc, 'elapsed_s': elapsed,
        'log': str(engine_log), 'patchers': patchers, 'env': env_overrides,
    }


# ------------------------- log parsing -------------------------

# T winner: min_sell_pnl=X.XX% max_hold=Yh bull_shield=ON/OFF bear_shield=ON/OFF (total +XX.XX% vs +YY.YY%, delta +Z.ZZ%)
T_WINNER_RE = re.compile(
    r'T winner: min_sell_pnl=(?P<pnl>[\d.]+)% max_hold=(?P<hold>\d+)h '
    r'bull_shield=(?P<bull>ON|OFF) bear_shield=(?P<bear>ON|OFF) '
    r'\(total (?P<total>[+\-][\d.]+)% vs (?P<base>[+\-][\d.]+)%, '
    r'delta (?P<delta>[+\-][\d.]+)%\)'
)

# FALLBACK: when Mode T finds "no shield improvement", the engine doesn't emit
# the "T winner:" line. Instead the iteration's effective total is in:
#   "Baseline (all-OFF, gate applied): regime-switched=+XX.XX% [N trades, WR YY%]"
# This is the Mode T total when shields are disabled (both False) — gates applied.
# B's 6h converged to this state at +89.41% — but the original parser reported
# no_t_winner because no "T winner:" line. Fix: parse this line as fallback.
BASELINE_GATED_RE = re.compile(
    r'Baseline \(all-OFF, gate applied\): regime-switched=(?P<total>[+\-][\d.]+)% '
    r'\[(?P<trades>\d+) trades, WR (?P<wr>\d+)%\]'
)

# >>> Converged at iteration N
CONVERGED_RE = re.compile(r'>>> Converged.* at iteration (?P<iter>\d+)')

# >>> Reached max_iter=N without convergence
MAXITER_RE = re.compile(r'>>> Reached max_iter=(?P<iter>\d+) without convergence')

# >>> 2-CYCLE DETECTED at iteration N
CYCLE_RE = re.compile(r'>>> 2-CYCLE DETECTED at iteration (?P<iter>\d+)')


def parse_hrst_log(log_path: Path) -> dict:
    """Parse engine log for Mode T final winner + convergence status."""
    last_t = None
    last_baseline_gated = None  # fallback for shield-disabled converged runs
    converged = None
    cycle = None
    if not log_path.exists():
        return {'mode_t_total': None, 'status': 'log_missing'}
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = T_WINNER_RE.search(line)
            if m:
                last_t = m.groupdict()
            m = BASELINE_GATED_RE.search(line)
            if m:
                last_baseline_gated = m.groupdict()
            m = CONVERGED_RE.search(line)
            if m:
                converged = int(m.group('iter'))
            m = MAXITER_RE.search(line)
            if m:
                converged = -int(m.group('iter'))  # negative = max-iter hit
            m = CYCLE_RE.search(line)
            if m:
                cycle = int(m.group('iter'))

    # Prefer explicit T winner line. Fall back to last baseline-gated line
    # (shield-disabled converged case — B's situation).
    if last_t is None and last_baseline_gated is None:
        return {'mode_t_total': None, 'status': 'no_t_winner'}

    if cycle is not None:
        status = f'2cycle_at_iter_{cycle}'
    elif converged is None:
        status = 'no_convergence_marker'
    elif converged < 0:
        status = f'maxiter_at_{-converged}'
    else:
        status = f'converged_at_{converged}'

    if last_t is not None:
        return {
            'mode_t_total': float(last_t['total']),
            'mode_t_base': float(last_t['base']),
            'mode_t_delta': float(last_t['delta']),
            'min_sell_pnl': float(last_t['pnl']),
            'max_hold': int(last_t['hold']),
            'bull_shield': last_t['bull'],
            'bear_shield': last_t['bear'],
            'status': status,
        }
    # Shield-disabled fallback: shields off, no explicit shield-tier winner
    return {
        'mode_t_total': float(last_baseline_gated['total']),
        'mode_t_base': float(last_baseline_gated['total']),  # baseline = total when shield disabled
        'mode_t_delta': 0.0,
        'min_sell_pnl': None,    # not applicable when shield is OFF
        'max_hold': None,
        'bull_shield': 'OFF',
        'bear_shield': 'OFF',
        'status': status + '_shield_disabled',
    }


# ------------------------- verdict + report -------------------------

def verdict_for(mode_t_total: float | None) -> str:
    if mode_t_total is None:
        return 'NO_RESULT'
    delta = mode_t_total - PRODUCTION_HRST_REF
    if delta >= SHIP_THRESHOLD_PP:
        return 'SHIP'
    if delta <= -SHIP_THRESHOLD_PP:
        return 'DEAD'
    return 'MARGINAL'


def write_summary(machine: str, ts: str, variant: str, result: dict, parsed: dict, log):
    summary_txt = OUT_DIR / f'run_reliability_hrst_{machine}_{ts}_{variant}.txt'
    summary_csv = OUT_DIR / f'run_reliability_hrst_{machine}_{ts}_{variant}.csv'

    mt = parsed.get('mode_t_total')
    delta = (mt - PRODUCTION_HRST_REF) if mt is not None else None
    v = verdict_for(mt)

    elapsed_min = result.get('elapsed_s', 0) / 60.0

    lines = [
        '=' * 90,
        f'  RELIABILITY HRST — {variant} on {machine}',
        f'  Timestamp: {ts}   Asset: {ASSET}   Horizons: {HORIZONS}   Replay: {REPLAY}h',
        '=' * 90,
        '',
        f'  Engine subprocess rc:    {result.get("rc")}',
        f'  Engine elapsed:          {elapsed_min:.1f} min',
        f'  Mode T status:           {parsed.get("status")}',
        '',
        f'  Mode T total return:     {f"{mt:+.2f}%" if mt is not None else "N/A"}',
        f'  Mode T baseline (all-OFF): {parsed.get("mode_t_base", "N/A")}'
        + (f'{parsed["mode_t_base"]:+.2f}%'.rjust(0) if isinstance(parsed.get('mode_t_base'), float) else ''),
        f'  Mode T delta over baseline: '
        + (f'{parsed["mode_t_delta"]:+.2f}pp' if isinstance(parsed.get('mode_t_delta'), float) else 'N/A'),
        '',
        f'  Production HRST baseline (2026-05-06): +{PRODUCTION_HRST_REF:.2f}%',
        f'  Δ vs production:         {f"{delta:+.2f}pp" if delta is not None else "N/A"}',
        f'  Ship threshold:          ±{SHIP_THRESHOLD_PP:.1f}pp',
        '',
        f'  Verdict:                 {v}',
        '',
        '  Mode T config (final):',
        f'    min_sell_pnl_pct:      {parsed.get("min_sell_pnl", "N/A")}',
        f'    max_hold_hours:        {parsed.get("max_hold", "N/A")}',
        f'    bull_shield:           {parsed.get("bull_shield", "N/A")}',
        f'    bear_shield:           {parsed.get("bear_shield", "N/A")}',
        '',
        '  Decision rule (set in advance):',
        f'    Δ ≥ +{SHIP_THRESHOLD_PP:.1f}pp  → SHIP (promote {variant} patchers to production engine)',
        f'    |Δ| < {SHIP_THRESHOLD_PP:.1f}pp → MARGINAL (no production change; consider Phase 2 followup)',
        f'    Δ ≤ −{SHIP_THRESHOLD_PP:.1f}pp → DEAD (variant does not generalize beyond Phase 1 8h win)',
        '',
        '  Engine log: ' + result.get('log', '(missing)'),
        '=' * 90,
    ]
    summary_txt.write_text('\n'.join(lines), encoding='utf-8')

    # CSV row
    headers = [
        'machine', 'variant', 'ts', 'rc', 'elapsed_min',
        'mode_t_total', 'mode_t_base', 'mode_t_delta',
        'production_ref', 'delta_vs_production', 'verdict',
        'min_sell_pnl', 'max_hold', 'bull_shield', 'bear_shield', 'mode_t_status',
    ]
    row = [
        machine, variant, ts, result.get('rc'), f'{elapsed_min:.1f}',
        f'{mt:.4f}' if mt is not None else '',
        f'{parsed.get("mode_t_base"):.4f}' if isinstance(parsed.get('mode_t_base'), float) else '',
        f'{parsed.get("mode_t_delta"):.4f}' if isinstance(parsed.get('mode_t_delta'), float) else '',
        f'{PRODUCTION_HRST_REF:.4f}',
        f'{delta:.4f}' if delta is not None else '',
        v,
        parsed.get('min_sell_pnl', ''), parsed.get('max_hold', ''),
        parsed.get('bull_shield', ''), parsed.get('bear_shield', ''),
        parsed.get('status', ''),
    ]
    summary_csv.write_text(','.join(str(x) for x in headers) + '\n'
                            + ','.join(str(x) for x in row) + '\n',
                            encoding='utf-8')

    log(f'CSV:     {summary_csv}')
    log(f'SUMMARY: {summary_txt}')


# ------------------------- entry points -------------------------

def cmd_status(machine: str):
    state = load_state(machine)
    if state.get('campaign_id') is None:
        print(f'[{machine}] no campaign in progress')
        return
    print(f'[{machine}] campaign_id={state["campaign_id"]}  variant={state.get("variant")}')
    print(f'         started_at={state.get("started_at")}  completed={state.get("completed")}')
    print(f'         snapshot_dir={state.get("snapshot_dir")}')
    if state.get('result'):
        r = state['result']
        print(f'         rc={r.get("rc")}  elapsed_min={r.get("elapsed_s", 0)/60:.1f}')
        print(f'         engine log: {r.get("log")}')


def cmd_reset(machine: str, log_to_stderr=True):
    state = load_state(machine)
    snap = state.get('snapshot_dir')
    if snap:
        p = Path(snap)
        if p.exists():
            if log_to_stderr:
                print(f'[{machine}] removing snapshot {p}')
            shutil.rmtree(p, ignore_errors=True)
    sf = state_file(machine)
    if sf.exists():
        sf.unlink()
    lf = lock_file(machine)
    if lf.exists():
        lf.unlink()
    if log_to_stderr:
        print(f'[{machine}] state + lock cleared')


def cmd_report_only(machine: str):
    """Rebuild summary from the existing engine log without rerunning."""
    state = load_state(machine)
    if not state.get('result') or not state.get('variant'):
        print(f'[{machine}] no completed run to report. State: {state.get("campaign_id")}')
        return
    variant = state['variant']
    ts = state['campaign_id']
    result = state['result']
    parsed = parse_hrst_log(Path(result['log']))
    orch_log = orch_log_path(machine, ts)

    def _log(m):
        print(m, flush=True)
        try:
            with open(orch_log, 'a', encoding='utf-8') as f:
                f.write(f'[{datetime.now().strftime("%H:%M:%S")}] {m}\n')
        except Exception:
            pass

    write_summary(machine, ts, variant, result, parsed, _log)


def cmd_run(variant: str, machine: str, reuse_snapshot: str | None = None):
    if not acquire_lock(machine):
        existing = lock_file(machine).read_text().strip()
        print(f'[{machine}] lock held by PID {existing}. Refusing to start.')
        sys.exit(1)

    try:
        patchers, env_overrides, _desc = VARIANTS[variant]

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        orch_log = orch_log_path(machine, ts)
        log = make_logger(orch_log)
        log('=' * 90)
        log(f'RUN RELIABILITY HRST  machine={machine}  variant={variant}  start={datetime.now().isoformat(timespec="seconds")}')
        log(f'Orchestrator log: {orch_log}')
        log(f'Asset={ASSET}  Horizons={HORIZONS}  Replay={REPLAY}h')
        log('=' * 90)

        # Snapshot — either reuse existing (--reuse-snapshot) or create fresh
        if reuse_snapshot is not None:
            snap = Path(reuse_snapshot).resolve()
            if not snap.exists():
                log(f'  [ERROR] --reuse-snapshot path does not exist: {snap}')
                sys.exit(2)
            # Verify all 4 marker files are present in the snapshot
            missing = [f for f in DATA_FILES if not (snap / f[5:]).exists()]
            if missing:
                log(f'  [ERROR] reused snapshot is missing marker files: {missing}')
                sys.exit(2)
            log(f'REUSING existing snapshot at {snap}')
            log(f'  All 4 marker files present.')
        else:
            log('Taking data snapshot for trader-coexistence...')
            snap = snapshot_dir_for(machine, ts)
            create_data_snapshot(snap, log)
        freeze = data_freeze_marker(snap)
        log('Initial data freeze markers:')
        for fname, m in freeze.items():
            log(f'  {fname}: {m}')

        state = {
            'started_at': datetime.now().isoformat(timespec='seconds'),
            'campaign_id': ts,
            'variant': variant,
            'machine': machine,
            'data_freeze': freeze,
            'snapshot_dir': str(snap),
            'completed': False,
            'result': None,
        }
        save_state(machine, state)

        # Run HRST
        result = run_hrst(variant, patchers, env_overrides, snap, machine, ts, log)
        state['result'] = result

        # Drift check
        post_freeze = data_freeze_marker(snap)
        drift = freeze_diff(freeze, post_freeze)
        if drift:
            log('  WARNING: snapshot integrity drift detected:')
            for d in drift:
                log(f'    {d}')
        else:
            log('  Snapshot integrity OK (no drift)')

        # Parse + report
        parsed = parse_hrst_log(Path(result['log']))
        log(f'Parse: mode_t_total={parsed.get("mode_t_total")}  status={parsed.get("status")}')
        write_summary(machine, ts, variant, result, parsed, log)

        state['completed'] = (result['rc'] == 0 and parsed.get('mode_t_total') is not None)
        state['parsed'] = parsed
        save_state(machine, state)

        v = verdict_for(parsed.get('mode_t_total'))
        log(f'VERDICT: {v}')
    finally:
        release_lock(machine)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=list(VARIANTS.keys()),
                    help='which reliability variant to test')
    ap.add_argument('--machine', required=True,
                    help='per-machine state/lock key (e.g. desktop, laptop)')
    ap.add_argument('--status', action='store_true', help='show state for this machine')
    ap.add_argument('--reset', action='store_true',
                    help='wipe state + snapshot + lock for this machine')
    ap.add_argument('--report-only', action='store_true',
                    help='rebuild summary from existing engine log without rerunning')
    ap.add_argument('--reuse-snapshot', default=None, metavar='PATH',
                    help='reuse an existing snapshot directory (skips fresh snapshot creation). '
                         'Use this to fairly compare a new variant against a prior run by '
                         'pointing at that run\'s snapshot dir. Example: '
                         '--reuse-snapshot data/_reliability_hrst_snapshot_desktop_20260515_154801')
    args = ap.parse_args()

    if args.status:
        cmd_status(args.machine)
        return
    if args.reset:
        cmd_reset(args.machine)
        return
    if args.report_only:
        cmd_report_only(args.machine)
        return

    if not args.variant:
        ap.error('--variant is required (unless --status / --reset / --report-only)')
    cmd_run(args.variant, args.machine, reuse_snapshot=args.reuse_snapshot)


if __name__ == '__main__':
    main()
