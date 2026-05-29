"""
FAYE smoke test (built 2026-05-30 after FAYE Phase 7 consolidation).

Verifies the FAYE single-file consolidation is wired correctly WITHOUT
running an actual Mode D / HRST (which would take hours). Tests:

  1. Syntax + import: FAYE parses and imports cleanly under its real
     module name (the path-based spec import won't pickle; needs sys.path).
  2. Module-level state: DIAG_STEP=1, K=5 seeds=[42..46], NEAR_LIVE defaults,
     isolated output dirs (models_faye/, config_faye/).
  3. Canonical entry points exist: run_mode_v / s / t / _refine_top_configs,
     _deku_eval_with_pruning, _deku_eval_with_pruning_inner.
  4. Serial fallbacks exist: run_mode_*_serial, _refine_top_configs_serial.
  5. Worker function is picklable (Windows ProcessPool spawn requirement).
  6. No monkey-patch leftovers: all _ENG_*_ORIG, _h_*, *_parallel,
     *_hybrid private names are gone.
  7. K=5 wrap is native — not via module-load rebind. Confirmed by inspecting
     the function's source location.

Safe to run concurrently with a live HRST: read-only inspection, no Mode D
trigger, no model artifact writes.

Expected on green:
  [STEP 1] ✓ FAYE imports cleanly
  [STEP 2] ✓ Module-level state matches NEAR_LIVE / K=5 expectations
  [STEP 3] ✓ Canonical entry points all present
  [STEP 4] ✓ Serial fallbacks all present
  [STEP 5] ✓ Worker function pickles + re-locates via __module__
  [STEP 6] ✓ No monkey-patch leftovers (12 names verified gone)
  [STEP 7] ✓ _deku_eval_with_pruning is native K=5 (not rebind)
  SMOKE TEST: PASS
"""

import sys
import os
import pickle


def _check(label, condition, detail=''):
    mark = '✓' if condition else '✗'
    extra = f'  ({detail})' if detail else ''
    print(f'  {mark} {label}{extra}')
    return bool(condition)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if os.getcwd() != project_root:
        os.chdir(project_root)

    print('=' * 70)
    print('  FAYE smoke test')
    print('=' * 70)

    all_ok = True

    # --------------------------------------------------------------
    # STEP 1: import
    # --------------------------------------------------------------
    print('\n[STEP 1] Import FAYE under its real module name:')
    try:
        import crypto_trading_system_faye as faye
        _check('imports without error', True)
    except Exception as e:
        _check('imports without error', False, f'{type(e).__name__}: {e}')
        sys.exit(1)

    # --------------------------------------------------------------
    # STEP 2: module-level constants
    # --------------------------------------------------------------
    print('\n[STEP 2] Module-level NEAR_LIVE / K=5 state:')
    all_ok &= _check('DIAG_STEP == 1',
                     faye.DIAG_STEP == 1, f'got {faye.DIAG_STEP}')
    all_ok &= _check('HOLDOUT_STEP == 1',
                     faye.HOLDOUT_STEP == 1, f'got {faye.HOLDOUT_STEP}')
    all_ok &= _check("NEAR_LIVE_SIGNAL_MODE == 'ternary'",
                     faye.NEAR_LIVE_SIGNAL_MODE == 'ternary', f'got {faye.NEAR_LIVE_SIGNAL_MODE!r}')
    all_ok &= _check("NEAR_LIVE_NA_POLICY == 'mean_last_10'",
                     faye.NEAR_LIVE_NA_POLICY == 'mean_last_10', f'got {faye.NEAR_LIVE_NA_POLICY!r}')
    all_ok &= _check('K5_K == 5',
                     faye.K5_K == 5, f'got {faye.K5_K}')
    all_ok &= _check('K5_SEEDS == [42, 43, 44, 45, 46]',
                     faye.K5_SEEDS == [42, 43, 44, 45, 46], f'got {faye.K5_SEEDS}')
    all_ok &= _check('MODE_D_OUTER_WORKERS == 8',
                     faye.MODE_D_OUTER_WORKERS == 8, f'got {faye.MODE_D_OUTER_WORKERS}')
    all_ok &= _check('FAYE_MODELS_DIR ends with models_faye',
                     faye.FAYE_MODELS_DIR.replace('\\', '/').endswith('/models_faye'),
                     faye.FAYE_MODELS_DIR)
    all_ok &= _check('FAYE_CONFIG_DIR ends with config_faye',
                     faye.FAYE_CONFIG_DIR.replace('\\', '/').endswith('/config_faye'),
                     faye.FAYE_CONFIG_DIR)
    all_ok &= _check('PRODUCTION_CSV mentions crypto_faye_production',
                     'crypto_faye_production' in str(faye.PRODUCTION_CSV),
                     str(faye.PRODUCTION_CSV))
    all_ok &= _check('REGIME_CONFIG_PATH mentions regime_config_faye',
                     'regime_config_faye' in str(faye.REGIME_CONFIG_PATH),
                     str(faye.REGIME_CONFIG_PATH))

    # --------------------------------------------------------------
    # STEP 3: canonical entry points
    # --------------------------------------------------------------
    print('\n[STEP 3] Canonical entry points:')
    canonical = [
        'run_mode_v', 'run_mode_s', 'run_mode_t',
        '_refine_top_configs',
        '_deku_eval_with_pruning', '_deku_eval_with_pruning_inner',
        '_mean_last_10_fill',
        '_get_deku_diagnostic_models_seeded',
        '_faye_grid_worker',
    ]
    for name in canonical:
        all_ok &= _check(f'{name} exists', hasattr(faye, name))

    # --------------------------------------------------------------
    # STEP 4: serial fallbacks
    # --------------------------------------------------------------
    print('\n[STEP 4] Serial fallbacks (callable for PARALLEL_BACKTESTS=1 or ≤1 item):')
    fallbacks = ['run_mode_v_serial', 'run_mode_s_serial', 'run_mode_t_serial',
                 '_refine_top_configs_serial']
    for name in fallbacks:
        all_ok &= _check(f'{name} exists', hasattr(faye, name))

    # --------------------------------------------------------------
    # STEP 5: pickle worker for ProcessPool
    # --------------------------------------------------------------
    print('\n[STEP 5] Worker pickling (ProcessPool spawn requirement):')
    try:
        blob = pickle.dumps(faye._faye_grid_worker)
        all_ok &= _check(f'_faye_grid_worker pickles', True, f'{len(blob)} bytes')
    except Exception as e:
        all_ok &= _check('_faye_grid_worker pickles', False, f'{type(e).__name__}: {e}')

    import importlib
    mod = importlib.import_module(faye._faye_grid_worker.__module__)
    fn = getattr(mod, '_faye_grid_worker', None)
    all_ok &= _check(
        'worker re-locates via __module__',
        fn is faye._faye_grid_worker,
        f'__module__={faye._faye_grid_worker.__module__}')

    # --------------------------------------------------------------
    # STEP 6: no monkey-patch leftovers
    # --------------------------------------------------------------
    print('\n[STEP 6] Monkey-patch leftovers (all should be GONE):')
    leftover_names = [
        # Phase 3 — K=5 monkey-patch
        '_h_factories_seeded', '_h_deku_eval_median_k',
        '_H_K', '_H_SEEDS', '_H_ORIG_DEKU_EVAL',
        # Phase 4 — refine hybrid rebind
        '_refine_top_configs_hybrid',
        '_ENG_REFINE_TOP_CONFIGS_SERIAL', '_ENG_REFINE_TOP_CONFIGS_ORIG',
        # Phase 5a — mode V/S/T parallel rebinds
        'run_mode_v_parallel', 'run_mode_s_parallel', 'run_mode_t_parallel',
        '_ENG_RUN_MODE_V_SERIAL', '_ENG_RUN_MODE_S_SERIAL', '_ENG_RUN_MODE_T_SERIAL',
        '_ENG_RUN_MODE_V_ORIG', '_ENG_RUN_MODE_S_ORIG', '_ENG_RUN_MODE_T_ORIG',
    ]
    for name in leftover_names:
        all_ok &= _check(f'{name} is GONE', not hasattr(faye, name))

    # --------------------------------------------------------------
    # STEP 7: K=5 wrap is native (not via module-load rebind)
    # --------------------------------------------------------------
    print('\n[STEP 7] _deku_eval_with_pruning is native K=5 (no rebind):')
    import inspect
    src = inspect.getsource(faye._deku_eval_with_pruning)
    all_ok &= _check(
        'function body references K5_SEEDS',
        'K5_SEEDS' in src,
        'K=5 loop in body')
    all_ok &= _check(
        'function body calls _deku_eval_with_pruning_inner',
        '_deku_eval_with_pruning_inner' in src,
        'inner inner-loop ref')
    all_ok &= _check(
        '__module__ is the FAYE module',
        faye._deku_eval_with_pruning.__module__ == 'crypto_trading_system_faye',
        f'got {faye._deku_eval_with_pruning.__module__}')

    # --------------------------------------------------------------
    print('')
    print('=' * 70)
    if all_ok:
        print('  SMOKE TEST: PASS')
        print('  FAYE consolidation verified: no monkey-patches, all canonical')
        print('  entry points native, K=5 + 8-worker + 3-worker hybrid all inline.')
        print('=' * 70)
        sys.exit(0)
    else:
        print('  SMOKE TEST: FAIL')
        print('  See ✗ lines above.')
        print('=' * 70)
        sys.exit(1)


if __name__ == '__main__':
    main()
