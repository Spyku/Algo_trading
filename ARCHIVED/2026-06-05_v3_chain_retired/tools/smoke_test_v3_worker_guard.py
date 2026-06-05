"""
Smoke test for v3's main-process guard (commit 77dc87e).

Verifies the worker-guard fix without disturbing the running HRST:
  1. Direct import in this process -> _IS_MAIN_PROCESS should be True,
     dispatcher created, banners printed once.
  2. Import inside a real ProcessPoolExecutor worker -> _IS_MAIN_PROCESS
     should be False, _dispatcher should be None, NO banner spam.
  3. Capture the worker's stdout to confirm no [V3 PARALLEL_NEARLIVE]
     banner lines escape into the worker's output.

This is read-only — does NOT launch a Mode D / dispatch any configs.
Safe to run concurrently with the live HRST run.

Expected result on green:
  [MAIN]   _IS_MAIN_PROCESS=True, dispatcher created
  [WORKER] _IS_MAIN_PROCESS=False, _dispatcher is None
  [WORKER] no [V3 PARALLEL_NEARLIVE] banner in captured stdout
  SMOKE TEST: PASS
"""

import sys
import os
import io
import contextlib
from concurrent.futures import ProcessPoolExecutor


def _probe_worker_state():
    """Runs INSIDE a ProcessPoolExecutor worker (spawn-method on Windows).
    Captures all stdout/stderr during v3 import, then reports state."""
    import multiprocessing as mp
    import sys as _sys
    import os as _os
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    proc_name = mp.current_process().name

    # Worker process needs project root in sys.path to import v3
    # (Windows spawn-method gives the worker a fresh interpreter with sys.path
    # set from where ProcessPoolExecutor was launched, which is tools/.)
    proj_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))) \
        if '__file__' in dir() else _os.getcwd()
    if proj_root not in _sys.path:
        _sys.path.insert(0, proj_root)

    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        try:
            import crypto_trading_system_ed_g_narrow_d_parallel_nearlive_v3 as v3_in_worker
            is_main = v3_in_worker._IS_MAIN_PROCESS
            dispatcher = v3_in_worker._dispatcher
            stdout_text = captured_stdout.getvalue()
            stderr_text = captured_stderr.getvalue()
        except Exception as e:
            return {
                'error': f'{type(e).__name__}: {e}',
                'proc_name': proc_name,
            }

    # Detect banner leaks
    v3_banner_count = stdout_text.count('[V3 PARALLEL_NEARLIVE] Mode D outer ProcessPool patched')
    v3_max_workers_count = stdout_text.count('max_workers=')
    v3_csv_log_count = stdout_text.count('Full per-eval CSV')

    return {
        'proc_name': proc_name,
        'is_main': is_main,
        'dispatcher_is_none': dispatcher is None,
        'v3_banner_count': v3_banner_count,
        'v3_max_workers_count': v3_max_workers_count,
        'v3_csv_log_count': v3_csv_log_count,
        'stdout_chars': len(stdout_text),
        'stderr_chars': len(stderr_text),
    }


def main():
    # Tests run from project root: add it to sys.path so v3 module is importable
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if os.getcwd() != project_root:
        os.chdir(project_root)

    print('=' * 70)
    print('  V3 worker-guard smoke test (commit 77dc87e)')
    print('=' * 70)

    print('\n[STEP 1] Importing v3 in this (main) process — banners SHOULD print:')
    print('-' * 70)
    import crypto_trading_system_ed_g_narrow_d_parallel_nearlive_v3 as v3_main
    print('-' * 70)
    print(f'  v3._IS_MAIN_PROCESS   = {v3_main._IS_MAIN_PROCESS}')
    print(f'  v3._dispatcher        = {type(v3_main._dispatcher).__name__ if v3_main._dispatcher else None}')
    if v3_main._dispatcher is not None:
        print(f'  v3._dispatcher.max_workers = {v3_main._dispatcher.max_workers}')

    assert v3_main._IS_MAIN_PROCESS is True, 'EXPECTED _IS_MAIN_PROCESS=True in main process'
    assert v3_main._dispatcher is not None, 'EXPECTED _dispatcher created in main process'
    print('  ✓ Main-process state correct\n')

    print('[STEP 2] Spawning a ProcessPoolExecutor worker, importing v3 inside:')
    print('  (worker stdout is CAPTURED — banner spam would be detected here)')
    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_probe_worker_state)
        result = future.result(timeout=120)

    if 'error' in result:
        print(f'  ✗ WORKER ERROR: {result["error"]}')
        sys.exit(1)

    print(f'  worker process name      = {result["proc_name"]}')
    print(f'  worker _IS_MAIN_PROCESS  = {result["is_main"]}')
    print(f'  worker _dispatcher None? = {result["dispatcher_is_none"]}')
    print(f'  worker stdout chars      = {result["stdout_chars"]}')
    print(f'  worker stderr chars      = {result["stderr_chars"]}')
    print(f'  [V3] banner count        = {result["v3_banner_count"]}  (expected: 0)')
    print(f'  max_workers print count  = {result["v3_max_workers_count"]}  (expected: 0)')
    print(f'  CSV log print count      = {result["v3_csv_log_count"]}  (expected: 0)')

    # The hard assertions for "guard works"
    failures = []
    if result['is_main']:
        failures.append('worker reports _IS_MAIN_PROCESS=True — guard FAILED')
    if not result['dispatcher_is_none']:
        failures.append('worker has a dispatcher instance — guard FAILED')
    if result['v3_banner_count'] > 0:
        failures.append(f'worker printed [V3 PARALLEL_NEARLIVE] banner {result["v3_banner_count"]}× — guard FAILED on print suppression')
    if result['v3_max_workers_count'] > 0:
        failures.append(f'worker printed max_workers line {result["v3_max_workers_count"]}× — guard FAILED on print suppression')
    if result['v3_csv_log_count'] > 0:
        failures.append(f'worker printed Full per-eval CSV line {result["v3_csv_log_count"]}× — guard FAILED on print suppression')

    print('')
    print('=' * 70)
    if failures:
        print('  SMOKE TEST: FAIL')
        for f in failures:
            print(f'    - {f}')
        print('=' * 70)
        sys.exit(1)

    print('  SMOKE TEST: PASS')
    print('  v3 worker-guard fix verified — no v3 banner leakage in workers,')
    print('  no dispatcher created in workers, no main-process state confused.')
    print('=' * 70)


if __name__ == '__main__':
    main()
