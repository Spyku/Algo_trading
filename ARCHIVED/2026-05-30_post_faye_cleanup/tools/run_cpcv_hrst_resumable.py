"""tools/run_cpcv_hrst_resumable.py — resumable HRST runner for the CPCV engine fork.

Tests Combinatorial Purged Cross-Validation (López de Prado AFML Ch 12) as a
research-only Mode D candidate filter. Production untouched (writes to
_cpcv-suffixed files via crypto_trading_system_ed_cpcv.py).

Why CPCV: the production engine selects Mode D's top-6 candidates by single
walk-forward + 3-fold rolling holdout. CPCV adds a 15-path validation pass
(C(6,2)=15 train/test combinations) that re-ranks by MEDIAN return across paths
+ computes PBO (Probability of Backtest Overfitting). Configs with PBO > 50%
are rejected as overfit.

Why we're testing it: Deku V1.4 tried CPCV in March 2026 with gamma=1.0 forced
("_cpcv_gamma1_failed.py"). The current engine has matured significantly
(184-feature universe, PySR, Optuna refine, parallel-merged Mode V/S/T). Plus
this fork uses Design Option A — preserves the candidate's natural gamma per
fold instead of forcing decay off. So both potential causes of V1.4's failure
are addressed.

Resumable HRST split into 11 phases (mirrors run_cvar_hrst_resumable.py):
  Phase 1.h  : DV (Mode D + V) per horizon h (h ∈ {5, 6, 7, 8})
  Phase 2.h  : single-horizon Mode V refine
  Phase 3    : Mode R (regime detector backtest)
  Phase 4    : Mode S (joint regime sweep)
  Phase 5    : Mode T (shield + gate sweep, chains G internally)

NOTE: Mode D's top-6 selection is now CPCV-validated INSIDE the engine fork.
The wrapper just runs the standard HRST chain — CPCV happens automatically
during each Mode D phase.

Run on Desktop:
  python tools/run_cpcv_hrst_resumable.py
  python tools/run_cpcv_hrst_resumable.py --reset       # wipe state, restart from Phase 1.5h
  python tools/run_cpcv_hrst_resumable.py --replay 2880 # 4-month run

Estimated runtime (Desktop):
  Mode D base (per horizon): ~7-13 min, median ~11 min (per CLAUDE.md)
  CPCV add-on per horizon: 6 candidates × 15 paths × ~5-15 sec = ~10-25 min
  Per-horizon DV total: ~30-60 min (vs ~30 min for standard DV)
  Mode R + S + T: ~80-120 min
  Total HRST: ~5-7h on Desktop (vs ~3-4h for standard HRST)

Outputs (production untouched):
  models/crypto_ed_grid_ETH_<h>h_cpcv.csv      Mode D grids (now with cpcv_* columns)
  models/crypto_ed_production_cpcv.csv         Mode V winners
  config/regime_config_ed_cpcv.json            Mode S/T regime config
  output/cpcv_hrst_state_<window>h.json        State file for resume
  logs/cpcv_hrst_<phase>_*.log                 Per-phase logs

Decision rule when complete:
  Compare Mode T REF in logs/cpcv_hrst_5.T_*.log to current production HRST
  Mode T REF (May 6 LIVE = +76.77%).
    >= +81.77% (>= +5pp)  → CPCV provides material lift; consider promotion
    within ±5pp           → CPCV adds robustness signal but not material alpha
                            (still useful: PBO scores tell you which configs are overfit)
    <= +71.77% (<= -5pp)  → CPCV-driven selection is worse; methodology DEAD
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(ENGINE, 'logs')
OUTPUT_DIR = os.path.join(ENGINE, 'output')
MODELS_DIR = os.path.join(ENGINE, 'models')

ASSET = 'ETH'
HORIZONS = [5, 6, 7, 8]
ENGINE_FORK = 'crypto_trading_system_ed_cpcv.py'
PYTHON = sys.executable

# Single-instance lock (race-safe via O_CREAT | O_EXCL).
LOCK_FILE = os.path.join(OUTPUT_DIR, 'cpcv_hrst.lock')


# ─────────────────────── single-instance lock ───────────────────────


def _is_pid_alive(pid: int) -> bool:
    if sys.platform == 'win32':
        try:
            out = subprocess.check_output(['tasklist', '/FI', f'PID eq {pid}'],
                                            stderr=subprocess.DEVNULL).decode('utf-8', errors='ignore')
            return str(pid) in out
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def acquire_lock():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, encoding='utf-8') as f:
                existing_pid = int(f.read().strip().split('\n')[0])
            if _is_pid_alive(existing_pid):
                print('=' * 100)
                print(f'  ❌ ANOTHER CPCV-HRST INSTANCE IS RUNNING (PID {existing_pid})')
                print(f'     Lock file: {LOCK_FILE}')
                print(f'     Kill + delete lock if needed:')
                print(f'       Stop-Process -Id {existing_pid} -Force; Remove-Item "{LOCK_FILE}"')
                print('=' * 100)
                sys.exit(2)
            print(f'  [lock] stale lock from dead PID {existing_pid} — reclaiming')
            try: os.remove(LOCK_FILE)
            except OSError: pass
        except (ValueError, OSError) as e:
            print(f'  [lock] could not parse existing lock ({e}) — reclaiming')
            try: os.remove(LOCK_FILE)
            except OSError: pass

    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as f:
            f.write(f'{os.getpid()}\n{datetime.now().isoformat()}\n')
    except FileExistsError:
        print('=' * 100)
        print('  ❌ DUPLICATE LAUNCH DETECTED — another wrapper grabbed the lock first')
        print('     This instance exits cleanly. The OTHER instance continues.')
        print('=' * 100)
        sys.exit(3)

    print(f'  [lock] acquired (PID {os.getpid()}, lock file: {LOCK_FILE})')

    def _cleanup():
        try:
            if os.path.exists(LOCK_FILE):
                with open(LOCK_FILE, encoding='utf-8') as f:
                    locked_pid = int(f.read().strip().split('\n')[0])
                if locked_pid == os.getpid():
                    os.remove(LOCK_FILE)
        except Exception:
            pass
    atexit.register(_cleanup)


# ─────────────────────── state machine ───────────────────────


def _state_path(replay: int) -> str:
    return os.path.join(OUTPUT_DIR, f'cpcv_hrst_state_{replay}h.json')


def load_state(replay: int) -> dict:
    p = _state_path(replay)
    if not os.path.exists(p):
        return {'replay': replay, 'started_at': datetime.now().isoformat(), 'done': []}
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def save_state(state: dict):
    p = _state_path(state['replay'])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    state['updated_at'] = datetime.now().isoformat()
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, p)


def is_done(state: dict, phase: str) -> bool:
    return phase in state.get('done', [])


def mark_done(state: dict, phase: str, meta: dict | None = None):
    if phase not in state['done']:
        state['done'].append(phase)
    state.setdefault('phase_meta', {})[phase] = (meta or {}) | {
        'completed_at': datetime.now().isoformat(),
    }
    save_state(state)


def grid_csv_exists_and_fresh(horizon: int, max_age_hours: int = 24) -> bool:
    """Defensive: accept Phase 1 as done if the cpcv-tagged grid CSV is fresh."""
    p = os.path.join(MODELS_DIR, f'crypto_ed_grid_{ASSET}_{horizon}h_cpcv.csv')
    if not os.path.exists(p) or os.path.getsize(p) < 100:
        return False
    age_h = (time.time() - os.path.getmtime(p)) / 3600
    return age_h < max_age_hours


def production_cpcv_has_horizon(horizon: int) -> bool:
    p = os.path.join(MODELS_DIR, 'crypto_ed_production_cpcv.csv')
    if not os.path.exists(p):
        return False
    try:
        import pandas as pd
        df = pd.read_csv(p)
        rows = df[(df['coin'] == ASSET) & (df['horizon'] == horizon)]
        return len(rows) > 0
    except Exception:
        return False


# ─────────────────────── phase runners ───────────────────────


def run_phase(phase: str, mode: str, horizon_arg: str, replay: int, log_path: str) -> int:
    cmd = [
        PYTHON, ENGINE_FORK, mode, ASSET, horizon_arg,
        '--replay', str(replay), '--no-persist', '--no-data-update',
    ]
    print(f'\n{"=" * 100}')
    print(f'  PHASE {phase} — {mode} {ASSET} {horizon_arg}  (replay={replay}h)')
    print(f'  Log: {log_path}')
    print(f'  Cmd: {" ".join(cmd)}')
    print(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"=" * 100}', flush=True)

    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as logf:
        logf.write(f'[cpcv-hrst] {datetime.now().isoformat()} starting {mode} {ASSET} {horizon_arg}\n')
        logf.flush()
        rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=ENGINE).returncode

    dur = (time.time() - t0) / 60
    print(f'  PHASE {phase} finished: rc={rc} duration={dur:.1f} min')
    if rc != 0:
        print(f'  [!] FAILED — last 30 lines of {log_path}:')
        try:
            with open(log_path, encoding='utf-8') as f:
                tail = f.readlines()[-30:]
            for ln in tail:
                print(f'    {ln.rstrip()}')
        except Exception:
            pass
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay', type=int, default=1440,
                    help='HRST replay window in hours (default 1440 = 2 months)')
    ap.add_argument('--reset', action='store_true',
                    help='Wipe state file and start from Phase 1.5h')
    args = ap.parse_args()

    if args.reset and os.path.exists(_state_path(args.replay)):
        os.remove(_state_path(args.replay))
        print(f'  [reset] removed state file for replay={args.replay}h')

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    acquire_lock()

    state = load_state(args.replay)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('=' * 100)
    print(f'  CPCV HRST (RESUMABLE) — ETH 5,6,7,8h, replay={args.replay}h')
    print(f'  State file: {_state_path(args.replay)}')
    print(f'  Already done: {state.get("done", [])}')
    print(f'  Engine fork: {ENGINE_FORK}  (CPCV-validated Mode D top-6, gamma preserved)')
    print('=' * 100)

    overall_t0 = time.time()

    # ─── PHASE 1: Mode D per horizon (with CPCV inside) ───
    for h in HORIZONS:
        phase = f'1.D_{h}h'
        if is_done(state, phase):
            print(f'  ✓ {phase} already done — skip')
            continue
        if grid_csv_exists_and_fresh(h):
            print(f'  ✓ {phase} grid CSV already fresh — accepting + marking done')
            mark_done(state, phase, {'auto_detected': True})
            continue
        log_path = os.path.join(LOGS_DIR, f'cpcv_hrst_{phase}_{ts}.log')
        rc = run_phase(phase, 'D', f'{h}h', args.replay, log_path)
        if rc != 0:
            print(f'\n  ❌ {phase} failed. Re-run script to retry.')
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})

    # ─── PHASE 2: Mode V per horizon ───
    for h in HORIZONS:
        phase = f'2.V_{h}h'
        if is_done(state, phase):
            print(f'  ✓ {phase} already done — skip')
            continue
        if production_cpcv_has_horizon(h):
            print(f'  ✓ {phase} production_cpcv.csv already has row for {h}h — accepting')
            mark_done(state, phase, {'auto_detected': True})
            continue
        log_path = os.path.join(LOGS_DIR, f'cpcv_hrst_{phase}_{ts}.log')
        rc = run_phase(phase, 'V', f'{h}h', args.replay, log_path)
        if rc != 0:
            print(f'\n  ❌ {phase} failed. Re-run script to retry.')
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})

    # ─── PHASE 3: Mode R ───
    phase = '3.R'
    if not is_done(state, phase):
        log_path = os.path.join(LOGS_DIR, f'cpcv_hrst_{phase}_{ts}.log')
        rc = run_phase(phase, 'R', f'{",".join(str(h) for h in HORIZONS)}h',
                       args.replay, log_path)
        if rc != 0:
            print(f'\n  ❌ {phase} failed.')
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})
    else:
        print(f'  ✓ {phase} already done — skip')

    # ─── PHASE 4: Mode S ───
    phase = '4.S'
    if not is_done(state, phase):
        log_path = os.path.join(LOGS_DIR, f'cpcv_hrst_{phase}_{ts}.log')
        rc = run_phase(phase, 'S', f'{",".join(str(h) for h in HORIZONS)}h',
                       args.replay, log_path)
        if rc != 0:
            print(f'\n  ❌ {phase} failed.')
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})
    else:
        print(f'  ✓ {phase} already done — skip')

    # ─── PHASE 5: Mode T ───
    phase = '5.T'
    if not is_done(state, phase):
        log_path = os.path.join(LOGS_DIR, f'cpcv_hrst_{phase}_{ts}.log')
        rc = run_phase(phase, 'T', f'{",".join(str(h) for h in HORIZONS)}h',
                       args.replay, log_path)
        if rc != 0:
            print(f'\n  ❌ {phase} failed.')
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})
    else:
        print(f'  ✓ {phase} already done — skip')

    # ─── DONE ───
    state['completed_at'] = datetime.now().isoformat()
    save_state(state)
    total_min = (time.time() - overall_t0) / 60
    print('\n' + '=' * 100)
    print(f'  ✅ CPCV HRST COMPLETE (this invocation: {total_min:.1f} min)')
    print('=' * 100)
    print('  Outputs:')
    print(f'    Grid CSVs:     models/crypto_ed_grid_ETH_<5,6,7,8>h_cpcv.csv  (with cpcv_* columns)')
    print(f'    Production:    models/crypto_ed_production_cpcv.csv')
    print(f'    Regime config: config/regime_config_ed_cpcv.json')
    print('  Per-phase logs:  logs/cpcv_hrst_<phase>_*.log')
    print('  Decision:')
    print('    Compare final Mode T REF in logs/cpcv_hrst_5.T_*.log to')
    print('    current production HRST Mode T REF (May 6 LIVE = +76.77%).')
    print('    >=+5pp → CPCV provides lift; consider promotion.')
    print('    ±5pp  → CPCV adds robustness signal (PBO) but not material alpha.')
    print('    <-5pp → CPCV-driven selection is worse; family DEAD.')
    print('  Diagnostic value (regardless of headline):')
    print('    Inspect cpcv_pbo column in grid CSVs. Configs with PBO>50% are overfit')
    print('    (>50% of paths lose money). Low-PBO configs survive multi-fold validation.')


if __name__ == '__main__':
    main()
