
"""C40 — Skewness + kurtosis features (3rd / 4th moments). Currently absent.
Computed on rolling logret window. Adds:
  ret_skew_24h, ret_skew_72h
  ret_kurt_24h, ret_kurt_72h
"""
import numpy as np
import pandas as pd
import crypto_trading_system_ed as eng

_orig_build = eng.build_all_features


def _compute_moments(df: pd.DataFrame) -> dict:
    if 'close' not in df.columns:
        return {}
    logret = np.log(df['close'].astype(float)).diff()
    out = {}
    out['ret_skew_24h'] = logret.rolling(24).skew().fillna(0).values
    out['ret_skew_72h'] = logret.rolling(72).skew().fillna(0).values
    out['ret_kurt_24h'] = logret.rolling(24).kurt().fillna(0).values
    out['ret_kurt_72h'] = logret.rolling(72).kurt().fillna(0).values
    return out


def _patched_build(*args, **kwargs):
    result = _orig_build(*args, **kwargs)
    if isinstance(result, tuple) and len(result) == 3:
        df, all_cols, lead_cols = result
    elif isinstance(result, tuple) and len(result) == 2:
        df, all_cols = result
        lead_cols = None
    else:
        return result
    add = _compute_moments(df)
    n_added = 0
    for name, vals in add.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            n_added += 1
    if n_added:
        print(f'[C40] skewness/kurtosis features added: +{n_added}')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    return df, all_cols


eng.build_all_features = _patched_build
print('[C40] build_all_features patched')
