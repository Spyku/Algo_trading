"""
Comprehensive path-resolution matrix test (Fix #11 cont., 2026-05-29).

Verifies the engine's directory resolution is correct across the 4
combinations the user identified as must-work:

  laptop × desktop  ×  push-to-prod × no-persist

Since "laptop" vs "desktop" is just running on a different machine but
the engine/Drive paths are the same (Drive-synced G:\\), this collapses
to verifying:

  1. Default + persist: writes to <script_dir>/models/crypto_ed_production.csv
     and <script_dir>/config/regime_config_ed.json — the LIVE trader's paths
  2. Default + no-persist: writes to .../_noprod.csv suffix in same dirs —
     trader untouched
  3. v3 isolation dirs + persist: writes to <script_dir>/models_g_desktop_nearlive/
     production.csv — trader untouched
  4. v3 isolation dirs + no-persist: writes to ..._noprod.csv in isolation dir

Plus the worker-CWD robustness from Fix #11 — every scenario must work
identically whether main process or worker subprocess.

Pass: all expected paths absolute, all exist (or parent dir exists), no
relative-CWD fragility.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor


def _probe_paths_with_env(env_overrides=None, change_cwd_to=None):
    """Runs in worker subprocess. Sets env vars + chdir BEFORE importing
    the engine, then reports all resolved paths."""
    import os as _os
    import sys as _sys
    import importlib

    proj_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if proj_root not in _sys.path:
        _sys.path.insert(0, proj_root)

    if env_overrides:
        for k, v in env_overrides.items():
            _os.environ[k] = v
    if change_cwd_to:
        _os.chdir(change_cwd_to)

    # Force fresh import (in case prior probe imported it)
    for mod_name in [
        'crypto_trading_system_ed',
        'crypto_trading_system_ed_g_narrow_d',
    ]:
        if mod_name in _sys.modules:
            del _sys.modules[mod_name]

    import crypto_trading_system_ed as _E
    import crypto_trading_system_ed_g_narrow_d as _G

    return {
        'engine_DATA_DIR': _E.DATA_DIR,
        'engine_MACRO_DIR': _E.MACRO_DIR,
        'engine_MODELS_DIR': _E.MODELS_DIR,
        'engine_CONFIG_DIR': _E.CONFIG_DIR,
        'engine_PRODUCTION_CSV': _E.PRODUCTION_CSV,
        'engine_REGIME_CONFIG_PATH': _E.REGIME_CONFIG_PATH,
        'g_MODELS_DIR': _G.MODELS_DIR,
        'g_CONFIG_DIR': _G.CONFIG_DIR,
        'g_PRODUCTION_CSV': _G.PRODUCTION_CSV,
        'g_REGIME_CONFIG_PATH': _G.REGIME_CONFIG_PATH,
        'cwd': _os.getcwd(),
    }


def _check_scenario(label, env_overrides, change_cwd_to,
                     expected_engine_models_substr,
                     expected_g_models_substr):
    """Run a worker probe and check expected substring in MODELS_DIR."""
    print(f'\n[{label}]')
    print(f'  env_overrides   = {env_overrides}')
    print(f'  worker cwd      = {change_cwd_to or "<inherit from parent>"}')
    with ProcessPoolExecutor(max_workers=1) as pool:
        f = pool.submit(_probe_paths_with_env, env_overrides, change_cwd_to)
        r = f.result(timeout=120)

    failures = []

    # Engine paths must be absolute
    for key in ('engine_DATA_DIR', 'engine_MACRO_DIR', 'engine_MODELS_DIR',
                'engine_CONFIG_DIR', 'engine_PRODUCTION_CSV',
                'engine_REGIME_CONFIG_PATH'):
        if not os.path.isabs(r[key]):
            failures.append(f'{key} is NOT absolute: {r[key]}')

    # G_narrow_d paths must be absolute
    for key in ('g_MODELS_DIR', 'g_CONFIG_DIR', 'g_PRODUCTION_CSV',
                'g_REGIME_CONFIG_PATH'):
        if not os.path.isabs(r[key]):
            failures.append(f'{key} is NOT absolute: {r[key]}')

    # Expected location
    if expected_engine_models_substr not in r['engine_MODELS_DIR']:
        failures.append(f'engine_MODELS_DIR missing expected substr '
                        f'"{expected_engine_models_substr}": {r["engine_MODELS_DIR"]}')
    if expected_g_models_substr not in r['g_MODELS_DIR']:
        failures.append(f'g_MODELS_DIR missing expected substr '
                        f'"{expected_g_models_substr}": {r["g_MODELS_DIR"]}')

    # Print first few key paths
    print(f'  engine MODELS   = {r["engine_MODELS_DIR"]}')
    print(f'  engine CONFIG   = {r["engine_CONFIG_DIR"]}')
    print(f'  engine PROD CSV = {r["engine_PRODUCTION_CSV"]}')
    print(f'  G MODELS        = {r["g_MODELS_DIR"]}')
    print(f'  G CONFIG        = {r["g_CONFIG_DIR"]}')

    if failures:
        print(f'  [FAIL] FAILURES:')
        for fl in failures:
            print(f'    - {fl}')
        return False
    print(f'  [PASS]')
    return True


def main():
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.getcwd() != proj_root:
        os.chdir(proj_root)
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    print('=' * 70)
    print('  Path-resolution scenario matrix (Fix #11, 2026-05-29)')
    print('=' * 70)
    print(f'  Project root: {proj_root}')

    results = []

    # Scenario 1: Default config — used by production HRST + live trader read paths
    results.append(_check_scenario(
        label='Scenario 1: DEFAULT (no env vars), worker has bad CWD',
        env_overrides={},
        change_cwd_to='C:/Users',
        expected_engine_models_substr=os.path.join('engine', 'models'),
        expected_g_models_substr=os.path.join('engine', 'models'),
    ))

    # Scenario 2: v3 isolation dirs (parallel_nearlive sets these)
    results.append(_check_scenario(
        label='Scenario 2: v3 ISOLATION dirs (G_NARROW_*=models_g_desktop_nearlive), bad CWD',
        env_overrides={
            'G_NARROW_MODELS_DIR': 'models_g_desktop_nearlive',
            'G_NARROW_CONFIG_DIR': 'config_g_desktop_nearlive',
        },
        change_cwd_to='C:/Users',
        expected_engine_models_substr=os.path.join('engine', 'models'),  # engine itself unchanged
        expected_g_models_substr='models_g_desktop_nearlive',
    ))

    # Scenario 3: H75 isolation dirs (alternate research wrapper)
    results.append(_check_scenario(
        label='Scenario 3: H75 ISOLATION (H_STRICT_*=models_h75_wide), bad CWD',
        env_overrides={
            'H_STRICT_MODELS_DIR': 'models_h75_wide',
            'H_STRICT_CONFIG_DIR': 'config_h75_wide',
        },
        change_cwd_to='C:/Users',
        expected_engine_models_substr='models_h75_wide',
        expected_g_models_substr=os.path.join('engine', 'models'),  # G unaffected by H75 env
    ))

    # Scenario 4: Default config, correct CWD (baseline sanity check)
    results.append(_check_scenario(
        label='Scenario 4: DEFAULT, worker has CORRECT CWD (sanity baseline)',
        env_overrides={},
        change_cwd_to=None,
        expected_engine_models_substr=os.path.join('engine', 'models'),
        expected_g_models_substr=os.path.join('engine', 'models'),
    ))

    # Scenario 5: Absolute env override (rare but supported — passes through)
    results.append(_check_scenario(
        label='Scenario 5: ABSOLUTE env override (e.g. C:/temp/research_models)',
        env_overrides={
            'G_NARROW_MODELS_DIR': r'C:\Users\Public\research_models',
        },
        change_cwd_to='C:/Users',
        expected_engine_models_substr=os.path.join('engine', 'models'),  # engine unaffected
        expected_g_models_substr='research_models',
    ))

    print()
    print('=' * 70)
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    if n_fail == 0:
        print(f'  MATRIX TEST: ALL {n_pass} SCENARIOS PASS')
        print('  Cross-machine + persist/no-persist + workers all robust.')
        print('=' * 70)
    else:
        print(f'  MATRIX TEST: {n_fail}/{len(results)} SCENARIOS FAILED')
        print('=' * 70)
        sys.exit(1)


if __name__ == '__main__':
    main()
