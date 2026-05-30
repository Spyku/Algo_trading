"""tools/run_cvar_hrst_resumable.py — resumable runner for the C05 CVaR HRST.

Why resumable: HRST = ~7-8h on Desktop. If laptop sleeps, network blip,
power glitch, or you need to /stop & restart, the current `python
crypto_trading_system_ed_cvar.py HRST ETH 5,6,7,8h --replay 1440 --no-persist`
restarts from scratch — losing hours of compute.

This wrapper splits HRST into discrete checkpointed phases:
  Phase 1.h  : Mode D for horizon h (h ∈ {5, 6, 7, 8}) — outputs grid CSV
  Phase 2.h  : Mode V for horizon h — outputs row in production_cvar.csv
  Phase 3    : Mode R (regime detector backtest)
  Phase 4    : Mode S (joint regime sweep) — writes detector + bull/bear to config_cvar.json
  Phase 5    : Mode T (shield + gate sweep, chains G internally)

Each phase runs as a separate engine subprocess. After each successful phase,
we mark it done in `output/cvar_hrst_state_<window>h.json`. On restart, we
skip phases already marked done.

Idempotency check (defensive):
  - For Mode D phases: also accept if the cvar-tagged grid CSV already exists
    AND has >0 rows AND mtime is recent enough (last 24h)
  - For Mode V phases: accept if production_cvar.csv has a row for that horizon
  - For Mode R/S/T: rely on state file alone (their outputs are harder to
    self-detect)

Run:
  python tools/run_cvar_hrst_resumable.py
  python tools/run_cvar_hrst_resumable.py --replay 2880   # 4-month run
  python tools/run_cvar_hrst_resumable.py --reset         # wipe state and restart from Phase 1.5h

Output state file: output/cvar_hrst_state_<window>h.json
Per-phase log: logs/cvar_hrst_<phase>_<ts>.log
Final HRST output:
  - models/crypto_ed_grid_ETH_<h>h_cvar.csv   (4 files, one per horizon)
  - models/crypto_ed_production_cvar.csv      (rows for 5,6,7,8h)
  - config/regime_config_ed_cvar.json         (final detector + bull + bear + shield + gate)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(ENGINE, 'logs')
OUTPUT_DIR = os.path.join(ENGINE, 'output')
MODELS_DIR = os.path.join(ENGINE, 'models')

ASSET = 'ETH'
HORIZONS = [5, 6, 7, 8]
ENGINE_FORK = 'crypto_trading_system_ed_cvar.py'
PYTHON = sys.executable


def _state_path(replay: int) -> str:
    return os.path.join(OUTPUT_DIR, f'cvar_hrst_state_{replay}h.json')


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
    p = os.path.join(MODELS_DIR, f'crypto_ed_grid_{ASSET}_{horizon}h_cvar.csv')
    if not os.path.exists(p) or os.path.getsize(p) < 100:
        return False
    age_h = (time.time() - os.path.getmtime(p)) / 3600
    return age_h < max_age_hours


def production_cvar_has_horizon(horizon: int) -> bool:
    p = os.path.join(MODELS_DIR, 'crypto_ed_production_cvar.csv')
    if not os.path.exists(p):
        return False
    try:
        import pandas as pd
        df = pd.read_csv(p)
        rows = df[(df['coin'] == ASSET) & (df['horizon'] == horizon)]
        return len(rows) > 0
    except Exception:
        return False


def run_phase(phase: str, mode: str, horizon_arg: str, replay: int, log_path: str) -> int:
    """Run a single engine phase as a subprocess. Returns subprocess return code.

    Streams to log_path. Output also tee'd to stdout (last 30 lines on completion)
    so the user sees progress in real time."""
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

    t_start = time.time()
    with open(log_path, 'w', encoding='utf-8') as logf:
        logf.write(f'[cvar-hrst] {datetime.now().isoformat()} starting {mode} {ASSET} {horizon_arg}\n')
        logf.flush()
        rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=ENGINE).returncode

    dur = (time.time() - t_start) / 60
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

    state = load_state(args.replay)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('=' * 100)
    print(f'  CVaR HRST (RESUMABLE) — ETH 5,6,7,8h, replay={args.replay}h')
    print(f'  State file: {_state_path(args.replay)}')
    print(f'  Already done: {state.get("done", [])}')
    print(f'  Engine fork: {ENGINE_FORK}  (OPTUNA_METRIC=cvar, CVAR_LAMBDA=1.0)')
    print('=' * 100)

    overall_t0 = time.time()

    # ─────────────── PHASE 1: Mode D per horizon ───────────────
    for h in HORIZONS:
        phase = f'1.D_{h}h'
        if is_done(state, phase):
            print(f'  ✓ {phase} already done — skip')
            continue
        # Defensive idempotency check: if the tagged grid CSV is fresh, accept.
        if grid_csv_exists_and_fresh(h):
            print(f'  ✓ {phase} grid CSV already fresh on disk — accepting + marking done')
            mark_done(state, phase, {'auto_detected': True})
            continue
        log_path = os.path.join(LOGS_DIR, f'cvar_hrst_{phase}_{ts}.log')
        rc = run_phase(phase, 'D', f'{h}h', args.replay, log_path)
        if rc != 0:
            print(f'\n  ❌ {phase} failed. Re-run this script to retry that phase only.')
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})

    # ─────────────── PHASE 2: Mode V per horizon ───────────────
    # NOTE: Mode V re-validates the existing D grid candidates and writes refined
    # configs to crypto_ed_production_cvar.csv. So Mode V depends on Mode D output.
    for h in HORIZONS:
        phase = f'2.V_{h}h'
        if is_done(state, phase):
            print(f'  ✓ {phase} already done — skip')
            continue
        if production_cvar_has_horizon(h):
            print(f'  ✓ {phase} production_cvar.csv already has row for {h}h — accepting')
            mark_done(state, phase, {'auto_detected': True})
            continue
        log_path = os.path.join(LOGS_DIR, f'cvar_hrst_{phase}_{ts}.log')
        rc = run_phase(phase, 'V', f'{h}h', args.replay, log_path)
        if rc != 0:
            print(f'\n  ❌ {phase} failed. Re-run this script to retry from this phase.')
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})

    # ─────────────── PHASE 3: Mode R (regime detector backtest) ───────────────
    phase = '3.R'
    if not is_done(state, phase):
        log_path = os.path.join(LOGS_DIR, f'cvar_hrst_{phase}_{ts}.log')
        rc = run_phase(phase, 'R', f'{",".join(str(h) for h in HORIZONS)}h',
                       args.replay, log_path)
        if rc != 0:
            print(f'\n  ❌ {phase} failed. Re-run this script to retry from this phase.')
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})
    else:
        print(f'  ✓ {phase} already done — skip')

    # ─────────────── PHASE 4: Mode S (joint regime sweep) ───────────────
    phase = '4.S'
    if not is_done(state, phase):
        log_path = os.path.join(LOGS_DIR, f'cvar_hrst_{phase}_{ts}.log')
        rc = run_phase(phase, 'S', f'{",".join(str(h) for h in HORIZONS)}h',
                       args.replay, log_path)
        if rc != 0:
            print(f'\n  ❌ {phase} failed. Re-run this script to retry from this phase.')
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})
    else:
        print(f'  ✓ {phase} already done — skip')

    # ─────────────── PHASE 5: Mode T (shield + gate sweep, chains G) ─────────
    phase = '5.T'
    if not is_done(state, phase):
        log_path = os.path.join(LOGS_DIR, f'cvar_hrst_{phase}_{ts}.log')
        rc = run_phase(phase, 'T', f'{",".join(str(h) for h in HORIZONS)}h',
                       args.replay, log_path)
        if rc != 0:
            print(f'\n  ❌ {phase} failed. Re-run this script to retry from this phase.')
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})
    else:
        print(f'  ✓ {phase} already done — skip')

    # ─────────────── DONE ───────────────
    state['completed_at'] = datetime.now().isoformat()
    save_state(state)
    total_min = (time.time() - overall_t0) / 60
    print('\n' + '=' * 100)
    print(f'  ✅ CVaR HRST COMPLETE (this invocation: {total_min:.1f} min)')
    print('=' * 100)
    print('  Outputs:')
    print(f'    Grid CSVs:     models/crypto_ed_grid_ETH_<5,6,7,8>h_cvar.csv')
    print(f'    Production:    models/crypto_ed_production_cvar.csv')
    print(f'    Regime config: config/regime_config_ed_cvar.json')
    print('  Per-phase logs:  logs/cvar_hrst_<phase>_*.log')
    print('  Decision:')
    print('    Compare final Mode T REF in logs/cvar_hrst_5.T_*.log to')
    print('    current production HRST Mode T REF (May 6: +76.77%).')
    print('    ≥+5pp → SHIP. ±5pp → null. <-5pp → CVaR family DEAD.')


if __name__ == '__main__':
    main()
