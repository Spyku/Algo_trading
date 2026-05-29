"""
Smoke test for the MACRO_DIR absolute-path fix (2026-05-29 Fix #11).

Verifies that after the fix, a ProcessPoolExecutor worker — even when
launched from a wildly-different CWD — correctly resolves MACRO_DIR to
the absolute project-data path. This was the crash root cause that
killed v3 at 08:09 today: relative 'data/macro_data' resolved to whatever
CWD the worker happened to land in.

Pass criteria:
  - Worker reports MACRO_DIR is an absolute path
  - Worker reports os.path.exists(MACRO_DIR) == True
  - Both true even when worker's CWD is changed to C:\\Users
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor


def _worker_check_macro_dir():
    """Runs in worker subprocess — deliberately chdir to wrong CWD,
    then import engine and check MACRO_DIR resolves correctly anyway."""
    import os as _os
    import sys as _sys

    # Add project root to sys.path (workers don't inherit it)
    proj_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if proj_root not in _sys.path:
        _sys.path.insert(0, proj_root)

    # Deliberately change CWD to simulate the bug condition
    bad_cwd = 'C:/Users'
    _os.chdir(bad_cwd)

    # Now import engine — its module-level code runs in this worker
    import crypto_trading_system_ed as _E

    return {
        'worker_cwd': _os.getcwd(),
        'DATA_DIR': _E.DATA_DIR,
        'MACRO_DIR': _E.MACRO_DIR,
        'MACRO_is_abs': _os.path.isabs(_E.MACRO_DIR),
        'MACRO_exists': _os.path.exists(_E.MACRO_DIR),
        'DATA_is_abs': _os.path.isabs(_E.DATA_DIR),
        'DATA_exists': _os.path.exists(_E.DATA_DIR),
    }


def main():
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.getcwd() != proj_root:
        os.chdir(proj_root)
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    print('=' * 70)
    print('  MACRO_DIR fix smoke test (Fix #11, 2026-05-29)')
    print('=' * 70)

    print('\n[STEP 1] Main-process import — paths should already be absolute:')
    import crypto_trading_system_ed as E
    print(f'  DATA_DIR       = {E.DATA_DIR}')
    print(f'  MACRO_DIR      = {E.MACRO_DIR}')
    print(f'  DATA_DIR abs?  = {os.path.isabs(E.DATA_DIR)}')
    print(f'  MACRO_DIR abs? = {os.path.isabs(E.MACRO_DIR)}')
    print(f'  MACRO_DIR exists? = {os.path.exists(E.MACRO_DIR)}')

    assert os.path.isabs(E.DATA_DIR), 'DATA_DIR should be absolute after fix'
    assert os.path.isabs(E.MACRO_DIR), 'MACRO_DIR should be absolute after fix'
    assert os.path.exists(E.MACRO_DIR), 'MACRO_DIR should exist'
    print('  ✓ Main-process paths OK\n')

    print('[STEP 2] Worker with deliberately bad CWD (C:/Users):')
    print('  (This simulates the bug condition that crashed v3 at 08:09)')
    with ProcessPoolExecutor(max_workers=1) as pool:
        f = pool.submit(_worker_check_macro_dir)
        r = f.result(timeout=120)

    print(f'  worker CWD       = {r["worker_cwd"]} (intentionally wrong)')
    print(f'  worker DATA_DIR  = {r["DATA_DIR"]}')
    print(f'  worker MACRO_DIR = {r["MACRO_DIR"]}')
    print(f'  worker MACRO is abs?    = {r["MACRO_is_abs"]}')
    print(f'  worker MACRO exists?    = {r["MACRO_exists"]}')
    print(f'  worker DATA exists?     = {r["DATA_exists"]}')

    failures = []
    if not r['MACRO_is_abs']:
        failures.append('worker MACRO_DIR is NOT absolute — fix did not propagate to worker')
    if not r['MACRO_exists']:
        failures.append('worker MACRO_DIR does not exist from worker scope — relative path STILL broken')
    if not r['DATA_is_abs']:
        failures.append('worker DATA_DIR is NOT absolute')
    if not r['DATA_exists']:
        failures.append('worker DATA_DIR does not exist')

    print('')
    print('=' * 70)
    if failures:
        print('  SMOKE TEST: FAIL')
        for fl in failures:
            print(f'    - {fl}')
        print('=' * 70)
        sys.exit(1)

    print('  SMOKE TEST: PASS')
    print('  Worker with bad CWD still resolves MACRO_DIR / DATA_DIR correctly.')
    print('  The v3 08:09 crash class of bug is fixed.')
    print('=' * 70)


if __name__ == '__main__':
    main()
