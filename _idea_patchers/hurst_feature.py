
"""Hurst exponent feature patcher (Hurst 1951)."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_all_features


def _hurst_rs(series: np.ndarray) -> float:
    """R/S Hurst estimator. Series must be log-returns (zero-mean)."""
    n = len(series)
    if n < 50:
        return np.nan
    mean = series.mean()
    Y = series - mean
    Z = Y.cumsum()
    R = Z.max() - Z.min()
    S = series.std()
    if S == 0 or R <= 0:
        return np.nan
    return float(np.log(R / S) / np.log(n))


def _rolling_hurst(close: pd.Series, window: int = 168) -> pd.Series:
    logret = np.log(close).diff().fillna(0).values
    out = np.full(len(logret), np.nan)
    for i in range(window, len(logret)):
        out[i] = _hurst_rs(logret[i - window:i])
    return pd.Series(out, index=close.index, name='hurst_168h')


def _patched_build(*args, **kwargs):
    res = _orig_build(*args, **kwargs)
    if isinstance(res, tuple):
        df, cols = res
    else:
        return res
    if 'close' in df.columns:
        df['hurst_168h'] = _rolling_hurst(df['close'], window=168)
        if 'hurst_168h' not in cols:
            cols.append('hurst_168h')
    return df, cols


eng.build_all_features = _patched_build
print("[hurst_feature] build_all_features patched (+1 feature)")
