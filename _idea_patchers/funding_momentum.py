
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
