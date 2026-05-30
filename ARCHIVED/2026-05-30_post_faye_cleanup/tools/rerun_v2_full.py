"""tools/rerun_v2_full.py — clean full re-run of v2 mode sweep + PBO/DSR scoring.

Chains in order:
  1. tools/test_engine_modes_v2.py --reset    (wipe stale state + snapshots)
  2. tools/test_engine_modes_v2.py            (run all 7 modes, ~37-40h Desktop)
  3. tools/score_modes_pbo_dsr.py             (PBO + DSR post-hoc scoring)

Prerequisites (already shipped 2026-05-14):
  - CLI parser fix in crypto_trading_system_ed.py:7347-7350 (5h,6h,7h,8h now parses)
  - interventional_shap.py patched with check_additivity=False
  - leaf_weight_l1_reg.py uses monkey-patched __init__ (preserves sklearn signature)
  - sympy installed in venv (pip install sympy if needed)

Usage:
  python tools/rerun_v2_full.py              fresh full rerun (default)
  python tools/rerun_v2_full.py --resume     skip --reset; resume from state
  python tools/rerun_v2_full.py --skip-score skip the PBO/DSR step
  python tools/rerun_v2_full.py --dry-run    print steps without executing

Honest caveat: even with the CLI fix making 5/6/7/8h all real horizons, trade
counts at 1440h replay stay 4-10 per candidate per horizon. Below the binomial
power threshold (~30 trades) — results are weakly informative, not ship-grade.
For ship decisions, use HRST/CPCV (tools/run_cpcv_hrst_resumable.py).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent
PY = sys.executable


def run_step(label: str, cmd: list[str], dry: bool) -> int:
    print('\n' + '=' * 80)
    print(f'>>> [{datetime.now().strftime("%H:%M:%S")}] {label}')
    print(f'    cmd: {" ".join(cmd)}')
    print('=' * 80, flush=True)
    if dry:
        print('(--dry-run: not executing)')
        return 0
    return subprocess.run(cmd, cwd=ENGINE).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--resume', action='store_true', help='skip --reset (resume from state)')
    ap.add_argument('--skip-score', action='store_true', help='skip PBO/DSR scoring step')
    ap.add_argument('--dry-run', action='store_true', help='print steps without executing')
    args = ap.parse_args()

    t0 = datetime.now()
    print(f'=== rerun_v2_full START {t0.isoformat(timespec="seconds")} ===')

    if not args.resume:
        rc = run_step('STEP 1/3: reset v2 state + snapshots',
                      [PY, 'tools/test_engine_modes_v2.py', '--reset'],
                      args.dry_run)
        if rc != 0:
            print(f'\n[FAIL] reset step exited rc={rc} — aborting')
            sys.exit(rc)
    else:
        print('\n[--resume] skipping reset')

    rc = run_step('STEP 2/3: run 7-mode sweep (4 horizons, ~37-40h)',
                  [PY, 'tools/test_engine_modes_v2.py'],
                  args.dry_run)
    if rc != 0:
        print(f'\n[FAIL] sweep step exited rc={rc} — aborting')
        print('State preserved. Re-launch with --resume to continue.')
        sys.exit(rc)

    if not args.skip_score:
        rc = run_step('STEP 3/3: PBO + DSR post-hoc scoring',
                      [PY, 'tools/score_modes_pbo_dsr.py'],
                      args.dry_run)
        if rc != 0:
            print(f'\n[WARN] scoring step exited rc={rc} (sweep itself succeeded)')
            sys.exit(rc)
    else:
        print('\n[--skip-score] skipping PBO/DSR step')

    t1 = datetime.now()
    elapsed = t1 - t0
    print(f'\n=== rerun_v2_full COMPLETE in {elapsed} ===')
    print('Outputs:')
    print('  output/test_engine_modes_v2_<TS>_summary.txt    comparison verdict')
    print('  output/score_modes_pbo_dsr_<TS>.txt             PBO/DSR ranking')


if __name__ == '__main__':
    main()
