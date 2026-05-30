"""tools/run_c67_hrst_resumable.py — full HRST for C67 Connors RSI feature.

Tests if Connors RSI (Connors & Alvarez 2009) — a composite oscillator
(RSI(close,3) + RSI(streak,2) + roc_percentile) — adds material alpha at the
full HRST level. Mode D smoke test (May 10 V3 batch) showed +2.54 avg APF
delta with positive returns on 3/4 horizons (5h hurt return, 6h/7h/8h
helped). The 7h gain was particularly strong (+13.64pp return delta).

Hypothesis: if Mode S picks bull/bear from 6h/7h/8h (likely), the 5h damage
is irrelevant and the multi-horizon improvement should translate to a
material Mode T REF lift over the May 6 production baseline (+76.77%).

Architecture mirrors run_cpcv_hrst_resumable.py:
  - Standard engine + Connors RSI patcher injected per phase
  - Patcher redirects PRODUCTION_CSV, REGIME_CONFIG_PATH to _c67 paths
  - All outputs isolated from production via _c67 suffix

PHASES (resumable):
  1.D_5h, 1.D_6h, 1.D_7h, 1.D_8h  — Mode D per horizon with C67 feature
  2.V_5h, 2.V_6h, 2.V_7h, 2.V_8h  — Mode V refine per horizon
  3.R                              — Mode R regime detector backtest
  4.S                              — Mode S joint regime sweep
  5.T                              — Mode T shield + gate sweep

NOTE: Mode D grids already exist from V3 batch (May 10) but they were
written without engine-side path redirection. This run RE-EXECUTES Mode D
to ensure _c67 paths are populated cleanly. Skip Phase 1 manually via state
file edit if you trust the V3 batch grids.

Run on Desktop:
  python tools/run_c67_hrst_resumable.py
  python tools/run_c67_hrst_resumable.py --reset

ESTIMATED RUNTIME (Desktop):
  Phase 1 (Mode D × 4):      ~30-50 min
  Phase 2 (Mode V × 4):      ~2-3h
  Phase 3 (Mode R):          ~30 min
  Phase 4 (Mode S):          ~50 min
  Phase 5 (Mode T):          ~30-60 min
  Total:                     ~5-7h

OUTPUTS (production untouched):
  models/crypto_ed_grid_ETH_<h>h_c67.csv         Mode D grids
  models/crypto_ed_production_c67.csv            Mode V winners
  config/regime_config_ed_c67.json               Mode S/T regime config
  output/c67_hrst_state_<replay>h.json           State file
  logs/c67_hrst_<phase>_*.log                    Per-phase logs

DECISION RULE (after Phase 5.T):
  Compare final Mode T REF to +76.77% (May 6 LIVE production):
    >= +81.77% (>= +5pp)  → SHIP Connors RSI to production engine
    within ±5pp           → MARGINAL (within noise) — don't ship
    <= +71.77% (<= -5pp)  → DEAD — drop the feature
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(ENGINE, 'logs')
OUTPUT_DIR = os.path.join(ENGINE, 'output')
MODELS_DIR = os.path.join(ENGINE, 'models')
CONFIG_DIR = os.path.join(ENGINE, 'config')
PYTHON = sys.executable

ASSET = 'ETH'
HORIZONS = [5, 6, 7, 8]
ENGINE_SCRIPT = 'crypto_trading_system_ed.py'   # standard engine + patcher
PATCHER_MODULE = '_idea_patchers.connors_rsi_full'
LIVE_BASELINE_PCT = 76.77  # May 6 production HRST Mode T REF

# Single-instance lock
LOCK_FILE = os.path.join(OUTPUT_DIR, 'c67_hrst.lock')

# Files we seed from live production on first run
SEED_PAIRS = [
    (os.path.join(CONFIG_DIR, 'regime_config_ed.json'),
     os.path.join(CONFIG_DIR, 'regime_config_ed_c67.json')),
    (os.path.join(MODELS_DIR, 'crypto_ed_production.csv'),
     os.path.join(MODELS_DIR, 'crypto_ed_production_c67.csv')),
]


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
        os.kill(pid, 0); return True
    except (OSError, ProcessLookupError):
        return False


def acquire_lock():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, encoding='utf-8') as f:
                pid = int(f.read().strip().split('\n')[0])
            if _is_pid_alive(pid):
                print('=' * 100)
                print(f'  ❌ ANOTHER C67-HRST INSTANCE IS RUNNING (PID {pid})')
                print(f'     Kill + delete lock: Stop-Process -Id {pid} -Force; Remove-Item "{LOCK_FILE}"')
                print('=' * 100)
                sys.exit(2)
            try: os.remove(LOCK_FILE)
            except OSError: pass
        except Exception:
            try: os.remove(LOCK_FILE)
            except OSError: pass
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as f:
            f.write(f'{os.getpid()}\n{datetime.now().isoformat()}\n')
    except FileExistsError:
        print('=' * 100)
        print('  ❌ DUPLICATE LAUNCH — another wrapper grabbed the lock first')
        print('=' * 100)
        sys.exit(3)
    print(f'  [lock] acquired (PID {os.getpid()})')

    def _cleanup():
        try:
            if os.path.exists(LOCK_FILE):
                with open(LOCK_FILE, encoding='utf-8') as f:
                    locked = int(f.read().strip().split('\n')[0])
                if locked == os.getpid():
                    os.remove(LOCK_FILE)
        except Exception:
            pass
    atexit.register(_cleanup)


# ─────────────────────── state machine ───────────────────────


def _state_path(replay: int) -> str:
    return os.path.join(OUTPUT_DIR, f'c67_hrst_state_{replay}h.json')


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


# ─────────────────────── seeding ───────────────────────


def seed_c67_files():
    """Copy live production files to _c67 variants if missing."""
    for src, dst in SEED_PAIRS:
        if not os.path.exists(dst):
            if not os.path.exists(src):
                print(f'  [seed] SOURCE MISSING: {src} — cannot seed {dst}')
                continue
            shutil.copy2(src, dst)
            print(f'  [seed] {src} → {dst}')
        else:
            print(f'  [seed] {dst} already exists (skip)')


# ─────────────────────── phase runner ───────────────────────


def run_phase(phase: str, mode: str, horizon_arg: str, replay: int, log_path: str) -> int:
    """Run engine subprocess with C67 patcher injected. Patcher redirects
    PRODUCTION_CSV + REGIME_CONFIG_PATH to _c67 paths."""
    py_init = (
        f"import sys, os; sys.path.insert(0, r'{ENGINE}'); "
        f"os.chdir(r'{ENGINE}'); "
        f"import {PATCHER_MODULE}; "
        f"import crypto_trading_system_ed; crypto_trading_system_ed.main()"
    )
    cmd = [
        PYTHON, '-c',
        f"import sys; sys.argv = ['{ENGINE_SCRIPT}', '{mode}', "
        f"'{ASSET}', '{horizon_arg}', '--replay', '{replay}', "
        f"'--no-persist', '--no-data-update']; "
        + py_init,
    ]
    print(f'\n{"=" * 100}')
    print(f'  PHASE {phase}  |  C67 Connors RSI  |  {mode} {ASSET} {horizon_arg} replay={replay}h')
    print(f'  Log: {log_path}')
    print(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"=" * 100}', flush=True)

    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as logf:
        logf.write(f'[c67-hrst] {datetime.now().isoformat()} starting {mode} {ASSET} {horizon_arg}\n')
        logf.flush()
        rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            cwd=ENGINE).returncode
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


# ─────────────────────── main ───────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay', type=int, default=1440,
                    help='HRST replay window in hours (default 1440 = 2 months)')
    ap.add_argument('--reset', action='store_true',
                    help='Wipe state and restart from Phase 1.D_5h')
    args = ap.parse_args()

    if args.reset and os.path.exists(_state_path(args.replay)):
        os.remove(_state_path(args.replay))
        print(f'  [reset] removed state for replay={args.replay}h')

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    acquire_lock()

    print('=' * 100)
    print(f'  C67 CONNORS RSI HRST (RESUMABLE) — ETH 5,6,7,8h replay={args.replay}h')
    print(f'  Engine: {ENGINE_SCRIPT} + patcher {PATCHER_MODULE}')
    print(f'  Live baseline (May 6 prod): +{LIVE_BASELINE_PCT}%')
    print('=' * 100)

    print('\n  Seeding _c67 files from live production (if missing)...')
    seed_c67_files()

    state = load_state(args.replay)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f'\n  State: {_state_path(args.replay)}')
    print(f'  Done: {state.get("done", [])}')

    overall_t0 = time.time()

    # Phase 1: Mode D per horizon
    for h in HORIZONS:
        phase = f'1.D_{h}h'
        if is_done(state, phase):
            print(f'  ✓ {phase} already done — skip')
            continue
        log_path = os.path.join(LOGS_DIR, f'c67_hrst_{phase}_{ts}.log')
        rc = run_phase(phase, 'D', f'{h}h', args.replay, log_path)
        if rc != 0:
            print(f'\n  ❌ {phase} failed.')
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})

    # Phase 2: Mode V per horizon
    for h in HORIZONS:
        phase = f'2.V_{h}h'
        if is_done(state, phase):
            print(f'  ✓ {phase} already done — skip')
            continue
        log_path = os.path.join(LOGS_DIR, f'c67_hrst_{phase}_{ts}.log')
        rc = run_phase(phase, 'V', f'{h}h', args.replay, log_path)
        if rc != 0:
            print(f'\n  ❌ {phase} failed.')
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})

    # Phase 3: Mode R
    phase = '3.R'
    if not is_done(state, phase):
        log_path = os.path.join(LOGS_DIR, f'c67_hrst_{phase}_{ts}.log')
        rc = run_phase(phase, 'R', f'{",".join(str(h) for h in HORIZONS)}h',
                       args.replay, log_path)
        if rc != 0:
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})
    else:
        print(f'  ✓ {phase} already done — skip')

    # Phase 4: Mode S
    phase = '4.S'
    if not is_done(state, phase):
        log_path = os.path.join(LOGS_DIR, f'c67_hrst_{phase}_{ts}.log')
        rc = run_phase(phase, 'S', f'{",".join(str(h) for h in HORIZONS)}h',
                       args.replay, log_path)
        if rc != 0:
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})
    else:
        print(f'  ✓ {phase} already done — skip')

    # Phase 5: Mode T
    phase = '5.T'
    if not is_done(state, phase):
        log_path = os.path.join(LOGS_DIR, f'c67_hrst_{phase}_{ts}.log')
        rc = run_phase(phase, 'T', f'{",".join(str(h) for h in HORIZONS)}h',
                       args.replay, log_path)
        if rc != 0:
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})
    else:
        print(f'  ✓ {phase} already done — skip')

    state['completed_at'] = datetime.now().isoformat()
    save_state(state)
    total_min = (time.time() - overall_t0) / 60
    print('\n' + '=' * 100)
    print(f'  ✅ C67 HRST COMPLETE ({total_min:.1f} min this invocation)')
    print('=' * 100)
    print('  Outputs:')
    print(f'    Grid CSVs:     models/crypto_ed_grid_ETH_<5,6,7,8>h_c67.csv')
    print(f'    Production:    models/crypto_ed_production_c67.csv')
    print(f'    Regime config: config/regime_config_ed_c67.json')
    print('  Per-phase logs:  logs/c67_hrst_<phase>_*.log')
    print('  Decision:')
    print('    Compare final Mode T REF in logs/c67_hrst_5.T_*.log to')
    print(f'    LIVE production Mode T REF (+{LIVE_BASELINE_PCT}%).')
    print(f'    >= +{LIVE_BASELINE_PCT+5}% → SHIP Connors RSI feature to production')
    print(f'    ±5pp           → MARGINAL (within noise)')
    print(f'    <= +{LIVE_BASELINE_PCT-5}% → DEAD')


if __name__ == '__main__':
    main()
