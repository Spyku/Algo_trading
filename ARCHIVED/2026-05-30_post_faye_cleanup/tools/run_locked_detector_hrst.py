"""tools/run_locked_detector_hrst.py — Locked-detector HRST for scoring overlays.

PURPOSE: Test whether scoring overlays (CVaR/CDaR/PF/etc.) actually win at HRST
level when we eliminate Mode S regime sweep dilution. Per the 2026-05-10
re-analysis: the overlays ARE picking better per-horizon configs (e.g. CVaR 6h
Refined #1 +50.88%, CDaR 8h Refined #1 +67.03%). What kills them at full HRST
is Mode S landing on a different detector than the one those wins were
optimized under.

LOCKED-DETECTOR APPROACH:
  - Pin detector = current production detector (`tsmom_672h`)
  - Pin bull horizon = 6h, bear horizon = 8h (current production pairing)
  - Pin bull conf = 75%, bear conf = 65% (current production conf)
  - Skip Mode R (regime detector backtest)
  - Skip Mode S (joint regime sweep)
  - Run only: DV per horizon → Mode T (shield + gate sweep)

Engine forks supported:
  --engine cvar  → crypto_trading_system_ed_cvar.py    (CVaR scoring; default)
  --engine cdar  → crypto_trading_system_ed_cdar.py    (CDaR scoring)

NB: --engine apf intentionally omitted. The apf control is the LIVE production
HRST result (May 6: Mode T REF +76.77%). Any APF-control rerun would just
reproduce that ±5pp run-to-run variance.

PHASES (resumable, each runs as separate engine subprocess):
  1.DV_6h  : Mode DV ETH 6h --replay 1440  (writes h=6 row to production_<engine>.csv)
  1.DV_8h  : Mode DV ETH 8h --replay 1440  (writes h=8 row to production_<engine>.csv)
  2.lock   : Build locked regime_config_ed_<engine>.json (current detector + horizons,
             cleared shields + gates so Mode T sweeps them fresh)
  3.T      : Mode T ETH 6h,8h --replay 1440  (sweeps shield + gate on locked config)

State file: output/locked_hrst_state_<engine>.json
Per-phase log: logs/locked_hrst_<engine>_<phase>_<ts>.log

OUTPUTS:
  - models/crypto_ed_production_<engine>.csv   (h=6 + h=8 rows)
  - config/regime_config_ed_<engine>.json      (locked config + Mode T's shield+gate result)
  - logs/locked_hrst_<engine>_3.T_*.log        (Mode T REF in this log → compare to +76.77%)

DECISION RULE (after Phase 3 completes):
  - Mode T REF >= +81.77% (>=+5pp over May 6 prod)  → SHIP scoring change to live engine
  - Mode T REF in (+71.77%, +81.77%)                 → MARGINAL (run lambda sweep next)
  - Mode T REF <= +71.77% (<=-5pp)                   → DEAD even with locked detector;
                                                       overlay family conclusively closed

ESTIMATED RUNTIME (Desktop):
  Phase 1.DV_6h : ~25-40 min (D + V chain)
  Phase 1.DV_8h : ~25-40 min
  Phase 2.lock  : <1 min
  Phase 3.T     : ~30-60 min (shield + gate sweep, may iterate 2-3x)
  Total: ~1.5 - 2.5h

USAGE:
  python tools/run_locked_detector_hrst.py                  # default --engine cvar
  python tools/run_locked_detector_hrst.py --engine cdar
  python tools/run_locked_detector_hrst.py --reset           # wipe state, restart from 1.DV_6h
  python tools/run_locked_detector_hrst.py --replay 2880     # 4-month run
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

# Single-instance lock — prevents the duplicate-launch deadlock observed
# 2026-05-10 where the user's terminal accidentally launched the wrapper twice
# in 22-50ms intervals, causing two engines to fight on the same files.
LOCK_FILE = os.path.join(OUTPUT_DIR, 'locked_hrst.lock')


def _is_pid_alive(pid: int) -> bool:
    """Cross-platform best-effort liveness check."""
    if sys.platform == 'win32':
        try:
            out = subprocess.check_output(['tasklist', '/FI', f'PID eq {pid}'],
                                            stderr=subprocess.DEVNULL).decode('utf-8', errors='ignore')
            return str(pid) in out
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def acquire_lock():
    """Refuse to start if another instance is already running.

    Race-safe via os.O_CREAT | os.O_EXCL — if two wrappers launch in the same
    millisecond (observed 2026-05-10: PIDs differed by 22ms), only one will
    succeed in creating the lock; the other gets FileExistsError and exits.

    Stale locks (PID dead) are reclaimed automatically.
    Removes lock on normal exit via atexit.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Stale-lock check before atomic-create attempt
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, encoding='utf-8') as f:
                content = f.read().strip()
            existing_pid = int(content.split('\n')[0])
            if _is_pid_alive(existing_pid):
                print('=' * 100)
                print(f'  ❌ ANOTHER LOCKED-HRST INSTANCE IS ALREADY RUNNING (PID {existing_pid})')
                print(f'     This launch is being REFUSED to prevent duplicate-engine deadlock.')
                print(f'     Lock file: {LOCK_FILE}')
                print(f'     If the other instance is genuinely hung, kill it explicitly:')
                print(f'       Stop-Process -Id {existing_pid} -Force; Remove-Item "{LOCK_FILE}"')
                print('=' * 100)
                sys.exit(2)
            else:
                print(f'  [lock] stale lock from dead PID {existing_pid} — reclaiming')
                try:
                    os.remove(LOCK_FILE)
                except OSError:
                    pass
        except (ValueError, OSError) as e:
            print(f'  [lock] could not parse existing lock ({e}) — reclaiming')
            try:
                os.remove(LOCK_FILE)
            except OSError:
                pass

    # Atomic create — fails if another wrapper got here first within microseconds
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as f:
            f.write(f'{os.getpid()}\n{datetime.now().isoformat()}\n')
    except FileExistsError:
        print('=' * 100)
        print(f'  ❌ DUPLICATE LAUNCH DETECTED — another wrapper grabbed the lock first')
        print(f'     This is the second of (likely) two simultaneous launches.')
        print(f'     This instance is exiting cleanly. The OTHER instance continues normally.')
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

ASSET = 'ETH'
HORIZONS = [6, 8]  # locked to current production pairing (bull=6h, bear=8h)

# Current production config to lock against
LOCKED_DETECTOR = {'type': 'named', 'params': {'name': 'tsmom_672h'}}
LOCKED_BULL_H = 6
LOCKED_BEAR_H = 8
LOCKED_BULL_CONF = 75
LOCKED_BEAR_CONF = 65
LOCKED_MAX_POS_USD = 14300

ENGINES = {
    'cvar': {
        'fork': 'crypto_trading_system_ed_cvar.py',
        'prod_csv': os.path.join(MODELS_DIR, 'crypto_ed_production_cvar.csv'),
        'regime_json': os.path.join(CONFIG_DIR, 'regime_config_ed_cvar.json'),
    },
    'cdar': {
        'fork': 'crypto_trading_system_ed_cdar.py',
        'prod_csv': os.path.join(MODELS_DIR, 'crypto_ed_production_cdar.csv'),
        'regime_json': os.path.join(CONFIG_DIR, 'regime_config_ed_cdar.json'),
    },
}

LIVE_REGIME_JSON = os.path.join(CONFIG_DIR, 'regime_config_ed.json')
LIVE_PROD_CSV = os.path.join(MODELS_DIR, 'crypto_ed_production.csv')
LIVE_BASELINE_PCT = 76.77  # May 6 production HRST Mode T REF


# ─────────────────────── state machine ───────────────────────


def _state_path(engine: str, replay: int) -> str:
    return os.path.join(OUTPUT_DIR, f'locked_hrst_state_{engine}_{replay}h.json')


def load_state(engine: str, replay: int) -> dict:
    p = _state_path(engine, replay)
    if not os.path.exists(p):
        return {'engine': engine, 'replay': replay,
                'started_at': datetime.now().isoformat(), 'done': []}
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def save_state(state: dict):
    p = _state_path(state['engine'], state['replay'])
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


# ─────────────────────── phase runners ───────────────────────


def run_dv_phase(engine: str, fork: str, horizon: int, replay: int,
                 phase: str, log_path: str) -> int:
    """Run Mode DV for a single horizon. Engine writes per-horizon row to its
    own production_<engine>.csv (cvar/cdar) — production CSV untouched."""
    cmd = [
        PYTHON, fork, 'DV', ASSET, f'{horizon}h',
        '--replay', str(replay), '--no-persist', '--no-data-update',
    ]
    print(f'\n{"=" * 100}')
    print(f'  PHASE {phase}  |  engine={engine}  |  DV {ASSET} {horizon}h replay={replay}h')
    print(f'  Cmd: {" ".join(cmd)}')
    print(f'  Log: {log_path}')
    print(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"=" * 100}', flush=True)

    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as logf:
        logf.write(f'[locked-hrst] {datetime.now().isoformat()} starting DV {ASSET} {horizon}h\n')
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


def write_locked_config(engine: str, regime_json_path: str) -> bool:
    """Write a locked regime config:
     - Detector: locked to current production (tsmom_672h)
     - Horizons + confidences: locked to current production (6h@75% / 8h@65%)
     - Shields + gates: CLEARED so Mode T sweeps them fresh
     - All other ETH-block fields: copied from current production

    Returns True on success.
    """
    # Read current production config as starting point
    with open(LIVE_REGIME_JSON, encoding='utf-8') as f:
        live_cfg = json.load(f)
    if 'ETH' not in live_cfg:
        print('  [!] current production regime config has no ETH block — abort')
        return False

    eth_live = live_cfg['ETH']

    # Build locked ETH block — copy live, override locked fields, clear shield+gate
    eth_locked = dict(eth_live)
    eth_locked['regime_detector'] = LOCKED_DETECTOR
    if 'detector' in eth_locked:
        del eth_locked['detector']

    eth_locked['bull'] = {
        'horizon': LOCKED_BULL_H,
        'min_confidence': LOCKED_BULL_CONF,
        'max_position_usd': LOCKED_MAX_POS_USD,
        'hold_shield': False,                    # cleared — Mode T will sweep
        'rally_cooldown': {'enabled': False},     # cleared — Mode T will sweep
    }
    eth_locked['bear'] = {
        'horizon': LOCKED_BEAR_H,
        'min_confidence': LOCKED_BEAR_CONF,
        'max_position_usd': LOCKED_MAX_POS_USD,
        'hold_shield': False,
        'rally_cooldown': {'enabled': False},
    }
    eth_locked['min_sell_pnl_pct'] = 0.5
    eth_locked['max_hold_hours'] = 10
    eth_locked['enabled'] = True

    # Preserve any disabled-asset blocks from live
    locked_cfg = dict(live_cfg)
    locked_cfg['ETH'] = eth_locked

    # Backup existing engine-suffixed config (if present) before overwrite
    if os.path.exists(regime_json_path):
        backup = regime_json_path + f'.pre_locked_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        shutil.copy2(regime_json_path, backup)
        print(f'  Backed up existing engine config: {backup}')

    # Write locked config
    os.makedirs(os.path.dirname(regime_json_path), exist_ok=True)
    tmp = regime_json_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(locked_cfg, f, indent=2)
    os.replace(tmp, regime_json_path)
    print(f'  Locked config written: {regime_json_path}')
    print(f'    detector: {LOCKED_DETECTOR}')
    print(f'    bull: {LOCKED_BULL_H}h @ {LOCKED_BULL_CONF}% (shield cleared, gate cleared)')
    print(f'    bear: {LOCKED_BEAR_H}h @ {LOCKED_BEAR_CONF}% (shield cleared, gate cleared)')
    print(f'    Mode T will sweep shield + gate from this baseline')
    return True


def run_t_phase(engine: str, fork: str, replay: int,
                 phase: str, log_path: str) -> int:
    """Run Mode T (shield + gate sweep) on the locked config. Mode T reads
    regime_config_ed_<engine>.json + production_<engine>.csv, writes back
    optimized shield + gate."""
    horizons_arg = f'{",".join(str(h) for h in HORIZONS)}h'
    cmd = [
        PYTHON, fork, 'T', ASSET, horizons_arg,
        '--replay', str(replay), '--no-persist', '--no-data-update',
    ]
    print(f'\n{"=" * 100}')
    print(f'  PHASE {phase}  |  engine={engine}  |  T {ASSET} {horizons_arg} replay={replay}h')
    print(f'  Cmd: {" ".join(cmd)}')
    print(f'  Log: {log_path}')
    print(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"=" * 100}', flush=True)

    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as logf:
        logf.write(f'[locked-hrst] {datetime.now().isoformat()} starting T {ASSET}\n')
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


# ─────────────────────── result extraction + verdict ───────────────────────


def extract_mode_t_ref(t_log_path: str) -> dict:
    """Parse Mode T log to extract final Mode T REF + selected shield/gate."""
    if not os.path.exists(t_log_path):
        return {'error': 'log missing'}
    with open(t_log_path, encoding='utf-8') as f:
        lines = f.readlines()

    out = {
        't_ref_pct': None,
        'baseline_v0_ref_pct': None,
        'bull_shield': None,
        'bear_shield': None,
        'bull_gate': None,
        'bear_gate': None,
        'converged_iter': None,
        'gate_gain_pp': None,
    }

    # Look for the final iteration's "Baseline (all-OFF, gate applied): regime-switched=+X%"
    # and the last winning shield combo line, plus convergence marker.
    last_baseline = None
    last_shield = None
    last_bull_gate_winner = None
    last_bear_gate_winner = None
    converged = None
    last_v0 = None

    for i, ln in enumerate(lines):
        if 'baselines V0' in ln and 'REF=' in ln:
            try:
                ref_str = ln.split('REF=')[-1].strip().rstrip('%')
                last_v0 = float(ref_str)
            except Exception:
                pass
        if 'Baseline (all-OFF, gate applied)' in ln and 'regime-switched=' in ln:
            try:
                seg = ln.split('regime-switched=')[1].split('%')[0]
                last_baseline = float(seg.replace('+', '').strip())
            except Exception:
                pass
        if 'Top 3 shield combos' in ln and 'tot=' in ln:
            try:
                # Extract first "tot=+X%" from this line
                for tok in ln.split():
                    if tok.startswith('tot=+') or tok.startswith('tot=-'):
                        last_shield = float(tok.split('=')[1].rstrip('%'))
                        break
            except Exception:
                pass
        if 'WINNER: rr' in ln and 'cd=' in ln:
            # Bull winner line precedes 'wrote ETH.bull.rally_cooldown'
            txt = ln.split('WINNER:')[1].strip()
            if last_bull_gate_winner is None:
                last_bull_gate_winner = txt
            else:
                last_bear_gate_winner = txt
        if 'wrote ETH.bull.rally_cooldown' in ln:
            pass  # marker
        if 'wrote ETH.bear.rally_cooldown' in ln:
            # Reset bull/bear pair: next iteration will repopulate
            pass
        if 'Converged at iteration' in ln:
            try:
                converged = int(ln.split('iteration')[1].split()[0])
            except Exception:
                pass

    out['t_ref_pct'] = last_baseline if last_baseline is not None else last_shield
    out['baseline_v0_ref_pct'] = last_v0
    out['bull_gate'] = last_bull_gate_winner
    out['bear_gate'] = last_bear_gate_winner
    out['converged_iter'] = converged
    if last_v0 is not None and last_baseline is not None:
        out['gate_gain_pp'] = last_baseline - last_v0
    return out


def verdict(t_ref: float | None) -> str:
    if t_ref is None:
        return 'UNKNOWN'
    delta = t_ref - LIVE_BASELINE_PCT
    if delta >= 5:
        return f'SHIP  ({delta:+.2f}pp >= +5)'
    if delta > -5:
        return f'MARGINAL  ({delta:+.2f}pp within +/-5)'
    return f'DEAD  ({delta:+.2f}pp <= -5)'


# ─────────────────────── main ───────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=list(ENGINES.keys()), default='cvar',
                    help='Engine fork to test (default: cvar)')
    ap.add_argument('--replay', type=int, default=1440,
                    help='Replay window in hours (default 1440 = 2 months)')
    ap.add_argument('--reset', action='store_true',
                    help='Wipe state file and start from Phase 1.DV_6h')
    args = ap.parse_args()

    engine_cfg = ENGINES[args.engine]
    fork = engine_cfg['fork']
    prod_csv = engine_cfg['prod_csv']
    regime_json = engine_cfg['regime_json']

    state_path = _state_path(args.engine, args.replay)
    if args.reset and os.path.exists(state_path):
        os.remove(state_path)
        print(f'  [reset] removed state: {state_path}')

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Single-instance guard — refuse to start if another wrapper is alive.
    # Prevents the duplicate-launch deadlock observed 2026-05-10.
    acquire_lock()

    state = load_state(args.engine, args.replay)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('=' * 100)
    print(f'  LOCKED-DETECTOR HRST  |  engine={args.engine}  |  ETH 6h+8h replay={args.replay}h')
    print(f'  State: {state_path}')
    print(f'  Already done: {state.get("done", [])}')
    print(f'  Engine fork: {fork}')
    print(f'  Output prod CSV:  {prod_csv}')
    print(f'  Output regime cfg: {regime_json}')
    print(f'  Live baseline (May 6 prod Mode T REF): +{LIVE_BASELINE_PCT}%')
    print('=' * 100)

    overall_t0 = time.time()

    # ─── Phase 1.DV per horizon ───
    for h in HORIZONS:
        phase = f'1.DV_{h}h'
        if is_done(state, phase):
            print(f'  ✓ {phase} already done — skip')
            continue
        log_path = os.path.join(LOGS_DIR, f'locked_hrst_{args.engine}_{phase}_{ts}.log')
        rc = run_dv_phase(args.engine, fork, h, args.replay, phase, log_path)
        if rc != 0:
            print(f'\n  ❌ {phase} failed. Re-run script to retry.')
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})

    # ─── Phase 2.lock ───
    phase = '2.lock'
    if not is_done(state, phase):
        print(f'\n{"=" * 100}')
        print(f'  PHASE {phase}  |  engine={args.engine}  |  build locked regime config')
        print(f'{"=" * 100}')
        ok = write_locked_config(args.engine, regime_json)
        if not ok:
            print('\n  ❌ Phase 2.lock failed.')
            sys.exit(2)
        mark_done(state, phase, {'config_path': regime_json})
    else:
        print(f'  ✓ {phase} already done — skip')

    # ─── Phase 3.T ───
    phase = '3.T'
    if not is_done(state, phase):
        log_path = os.path.join(LOGS_DIR, f'locked_hrst_{args.engine}_{phase}_{ts}.log')
        rc = run_t_phase(args.engine, fork, args.replay, phase, log_path)
        if rc != 0:
            print(f'\n  ❌ {phase} failed. Re-run script to retry.')
            sys.exit(rc)
        mark_done(state, phase, {'log': log_path})
    else:
        print(f'  ✓ {phase} already done — skip')

    # ─── Verdict ───
    state['completed_at'] = datetime.now().isoformat()
    save_state(state)
    total_min = (time.time() - overall_t0) / 60

    # Find the most recent Mode T log to parse
    t_logs = sorted([f for f in os.listdir(LOGS_DIR)
                     if f.startswith(f'locked_hrst_{args.engine}_3.T_')])
    if t_logs:
        t_log_path = os.path.join(LOGS_DIR, t_logs[-1])
        result = extract_mode_t_ref(t_log_path)
    else:
        result = {'error': 'no Mode T log found'}

    print('\n' + '=' * 100)
    print(f'  ✅ LOCKED-DETECTOR HRST COMPLETE  ({total_min:.1f} min this invocation)')
    print('=' * 100)
    print(f'  Engine: {args.engine}')
    print(f'  Locked config: detector={LOCKED_DETECTOR["params"]["name"]}  '
          f'bull={LOCKED_BULL_H}h@{LOCKED_BULL_CONF}%  bear={LOCKED_BEAR_H}h@{LOCKED_BEAR_CONF}%')
    print()
    print(f'  Mode T result:')
    if 'error' in result:
        print(f'    {result["error"]}')
    else:
        print(f'    Mode T REF (gate applied): {result.get("t_ref_pct", "?")}%')
        print(f'    Baseline V0 (no gate):     {result.get("baseline_v0_ref_pct", "?")}%')
        print(f'    Gate gain:                 {result.get("gate_gain_pp", "?")}pp')
        print(f'    Bull gate:                 {result.get("bull_gate") or "(none)"}')
        print(f'    Bear gate:                 {result.get("bear_gate") or "(none)"}')
        print(f'    Converged iteration:       {result.get("converged_iter", "?")}')
    print()
    print(f'  COMPARISON vs live baseline (+{LIVE_BASELINE_PCT}% from May 6 production):')
    print(f'    {verdict(result.get("t_ref_pct"))}')
    print()
    print('  Decision rule:')
    print('    SHIP     >= +81.77% (>=+5pp over live)  → CVaR/CDaR scoring works at HRST')
    print('    MARGINAL within +/-5pp                  → run lambda sweep next (P3)')
    print('    DEAD     <= +71.77% (<=-5pp under live) → overlay family conclusively closed')
    print()
    print(f'  Per-phase logs: {os.path.join(LOGS_DIR, f"locked_hrst_{args.engine}_*_*.log")}')
    print(f'  Locked config:  {regime_json}')


if __name__ == '__main__':
    main()
