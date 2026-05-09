"""
C31 — Funding rate momentum / acceleration features.

Adds 3 new features on top of existing funding rate engine code:
  - deriv_funding_chg6h    (6-hour change of funding rate)
  - deriv_funding_chg72h   (72-hour change of funding rate)
  - deriv_funding_accel    (acceleration: diff of chg1d over 24h)

Engine already has: deriv_funding_rate, deriv_funding_chg1d, deriv_funding_zscore.
Hypothesis: shorter-window momentum + second-derivative captures funding-rate
inflections that a 24h-only change misses.

Method: monkey-patch crypto_trading_system_ed.build_all_features to inject the
3 new features AFTER the original returns. Run Mode D for ETH 5,6,7,8h with
--grid-tag IDEA31 --no-persist --no-data-update. Compare top-APF in tagged
grid CSVs vs untagged baselines.

Decision rule: avg APF delta >= +5pp -> PASS (escalate to HRST); >0 MARGINAL;
<= 0 FAIL.
"""
import os, sys, subprocess
from datetime import datetime

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)
sys.path.insert(0, os.path.join(ENGINE, 'tools'))

# Reuse harness helpers
from test_14_ideas import (write_patcher, run_mode_d, load_grid_csv,
                            load_baseline_grid, compare_grid_winners,
                            LOGS_DIR, MODELS_DIR, TS)

ASSET = 'ETH'
HORIZONS = [5, 6, 7, 8]
REPLAY = 1440
TAG = 'IDEA31'

PATCHER_CODE = '''
"""C31 funding rate momentum/acceleration patcher."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_all_features


def _patched_build(*args, **kwargs):
    res = _orig_build(*args, **kwargs)
    if not isinstance(res, tuple):
        return res
    df, cols = res
    if 'deriv_funding_rate' not in df.columns:
        print('[C31] deriv_funding_rate missing - patcher inactive')
        return df, cols

    fr = df['deriv_funding_rate']
    df['deriv_funding_chg6h'] = fr.diff(6) * 100.0
    df['deriv_funding_chg72h'] = fr.diff(72) * 100.0

    if 'deriv_funding_chg1d' in df.columns:
        df['deriv_funding_accel'] = df['deriv_funding_chg1d'].diff(24)
    else:
        df['deriv_funding_accel'] = fr.diff(24).diff(24) * 100.0

    added = ['deriv_funding_chg6h', 'deriv_funding_chg72h', 'deriv_funding_accel']
    for c in added:
        if c not in cols:
            cols.append(c)

    n_funding = fr.notna().sum()
    print(f'[C31] funding momentum patcher active: +3 features '
          f'(coverage rows={n_funding})')
    return df, cols


eng.build_all_features = _patched_build
print('[C31] build_all_features patched (+3 funding momentum features)')
'''


def main():
    print('=' * 100)
    print(f'  C31 — Funding rate momentum/acceleration features (ETH 5,6,7,8h, replay={REPLAY})')
    print('=' * 100)

    write_patcher('funding_momentum', PATCHER_CODE)
    print(f'  Patcher written: _idea_patchers/funding_momentum.py')

    log_path = os.path.join(LOGS_DIR, f'c31_funding_momentum_{TS}.log')
    print(f'  Log: {log_path}')
    print()

    deltas = []
    for h in HORIZONS:
        print(f'  >> Mode D ETH {h}h ...')
        rc = run_mode_d(ASSET, h, REPLAY, TAG, '_idea_patchers.funding_momentum',
                        log_path)
        if rc != 0:
            print(f'     ERROR rc={rc}')
            continue
        test_df, test_path = load_grid_csv(ASSET, h, TAG)
        base_df, base_path = load_baseline_grid(ASSET, h)
        if test_df is None:
            print(f'     ERROR no tagged grid: {test_path}')
            continue
        if base_df is None:
            print(f'     ERROR no baseline grid: {base_path}')
            continue
        tw, bw, delta = compare_grid_winners(test_df, base_df, sort_col='apf')
        if delta is None:
            print(f'     ERROR comparison failed')
            continue
        deltas.append((h, tw, bw, delta))
        print(f'     {h}h: test_apf={tw:.3f}  base_apf={bw:.3f}  delta={delta:+.3f}')

    print()
    print('=' * 100)
    if not deltas:
        print('  RESULT: NO valid deltas — likely all subprocesses errored. Check log.')
        return
    avg = sum(d[3] for d in deltas) / len(deltas)
    if avg >= 5:
        verdict = 'PASS — escalate to HRST validation'
    elif avg > 0:
        verdict = 'MARGINAL — within run-to-run noise; not actionable'
    else:
        verdict = 'FAIL — feature additions hurt'
    print(f'  C31 funding momentum: avg APF delta = {avg:+.3f} -> {verdict}')
    print()
    print('  Per-horizon:')
    for h, tw, bw, d in deltas:
        print(f'    {h}h: test={tw:.3f}  base={bw:.3f}  Δ={d:+.3f}')
    print('=' * 100)


if __name__ == '__main__':
    main()
