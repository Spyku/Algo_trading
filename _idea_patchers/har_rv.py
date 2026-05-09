
"""HAR-RV feature injection patcher (Corsi 2009 J.Fin.Econometrics)."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_all_features


def _compute_har_rv(df: pd.DataFrame) -> dict:
    """Returns 3 columns: rv_1d, rv_5d, rv_22d (realized variance over 24, 120, 528 hourly bars)."""
    if 'close' not in df.columns:
        return {}
    logret = np.log(df['close']).diff()
    rv2 = logret ** 2
    out = {}
    for label, w in [('rv_1d', 24), ('rv_5d', 120), ('rv_22d', 528)]:
        out[f'har_{label}'] = rv2.rolling(w, min_periods=w // 2).sum()
    return out


def _patched_build(*args, **kwargs):
    res = _orig_build(*args, **kwargs)
    if isinstance(res, tuple):
        df, cols = res
    else:
        return res
    new = _compute_har_rv(df)
    for c, vals in new.items():
        df[c] = vals
        if c not in cols:
            cols.append(c)
    return df, cols


eng.build_all_features = _patched_build
print(f"[har_rv] build_all_features patched (+3 features)")
