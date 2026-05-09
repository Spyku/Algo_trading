"""tools/test_c50_hrst.py — Full HRST validation of C50 (raw Profit Factor as
Mode V/Optuna scoring metric, replacing default APF).

Background:
  Overnight 2026-05-07 SUSPECT-retest run (Desktop, post-harness-fix) showed
  C50 PF objective scoring is the only candidate idea where ALL 4 horizons
  produced positive Mode D APF deltas:
      5h: +4.19, 6h: +2.09, 7h: +3.32, 8h: +3.90  (avg +3.38pp)
  Verdict was MARGINAL (below the +5pp PASS threshold) but consistent.

  This is the same family as C13 CDaR which won at Mode D level but FAILED
  at HRST level (Mode T total +71% vs live +86% on the May 4 run). Need to
  test whether C50 has the same fate or whether it actually improves Mode T.

What this script does:
  1. Writes the C50 patcher to _idea_patchers/c50_pf_objective.py
     (sets eng.OPTUNA_METRIC = 'rawpf' before HRST runs)
  2. Spawns crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440
     --no-persist --no-data-update with the patcher imported first
  3. Logs to logs/c50_hrst_<ts>.log
  4. After completion, prints a brief status (caller is expected to read the
     log + compare Mode T REF to current production HRST).

Decision rule:
  Mode T REF >= today's production HRST Mode T REF + 5pp
                  → SHIP (1-line scoring change in engine: OPTUNA_METRIC = 'rawpf')
  Within ±5pp     → null (scoring change is cosmetic)
  Worse by >5pp   → FAIL (confirm C50 DEAD at HRST level, file with C13 CDaR)

Runtime estimate: ~3-5h on laptop for HRST ETH 4-horizon --replay 1440 with
parallel V/S/T paths. Uses --no-persist so production untouched.

Run:
  python tools/test_c50_hrst.py
"""
from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)
sys.path.insert(0, os.path.join(ENGINE, 'tools'))

from test_14_ideas import write_patcher, LOGS_DIR, TS

ASSET = 'ETH'
HORIZONS = '5,6,7,8'
REPLAY = 1440
PYTHON = sys.executable

C50_PATCHER = '''
"""C50 raw profit factor as primary Optuna metric (instead of APF).

Sets eng.OPTUNA_METRIC = 'rawpf' BEFORE the engine's main() runs. The engine
already supports this branch in _compute_optuna_score (see crypto_trading_system_ed.py:534).
With OPTUNA_METRIC='rawpf', best_models_rawpf.csv path is used, but with
--no-persist the writes go to _noprod copies anyway.
"""
import crypto_trading_system_ed as eng

_orig = getattr(eng, 'OPTUNA_METRIC', 'apf')
eng.OPTUNA_METRIC = 'rawpf'
print(f'[C50] OPTUNA_METRIC: {_orig!r} -> {eng.OPTUNA_METRIC!r} (raw profit factor)')
'''


def main():
    print('=' * 100)
    print(f'  C50 HRST VALIDATION (PF objective scoring) — {datetime.now().isoformat()}')
    print(f'  Asset: {ASSET}  Horizons: {HORIZONS}  Replay: {REPLAY}h  --no-persist')
    print('=' * 100)

    write_patcher('c50_pf_objective', C50_PATCHER)
    print(f'  Patcher: _idea_patchers/c50_pf_objective.py')

    log_path = os.path.join(LOGS_DIR, f'c50_hrst_{TS}.log')
    print(f'  Log: {log_path}')

    py_init = (
        f"import sys, os; sys.path.insert(0, r'{ENGINE}'); "
        f"os.chdir(r'{ENGINE}'); "
        f"import _idea_patchers.c50_pf_objective; "
        f"import crypto_trading_system_ed; crypto_trading_system_ed.main()"
    )
    cmd = [
        PYTHON, '-c',
        (f"import sys; sys.argv = ['crypto_trading_system_ed.py', 'HRST', "
         f"'{ASSET}', '{HORIZONS}h', '--replay', '{REPLAY}', "
         f"'--no-persist', '--no-data-update']; ") + py_init,
    ]

    print(f'  cmd[2] preview: {cmd[2][:200]}...')
    print()
    print('  Launching HRST. This will run for ~3-5h. Live progress streams to the log.')
    print()

    with open(log_path, 'w', encoding='utf-8') as logf:
        logf.write(f"[c50-hrst] {datetime.now().isoformat()} starting HRST\n")
        logf.flush()
        rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            cwd=ENGINE).returncode

    print(f'  HRST finished. Return code: {rc}')
    print(f'  Log: {log_path}')
    print()
    print('  Next step: grep the log for "MODE T COMPLETE" and Mode T REF, then compare')
    print('  to today (2026-05-06 15:48) production HRST Mode T REF in')
    print('  logs/ed_v1_20260506_085146.log.')
    print('  Decision: ≥+5pp better → ship; within ±5pp → null; worse → DEAD.')


if __name__ == '__main__':
    main()
