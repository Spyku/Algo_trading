
"""C35 wavelet multi-scale decomposition (Daubechies db4 levels 1-4)."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_all_features


def _wavelet_features(df: pd.DataFrame) -> dict:
    try:
        import pywt
    except ImportError:
        print('[C35] pywavelets not installed: pip install PyWavelets')
        return {}
    if 'close' not in df.columns:
        return {}
    logret = np.log(df['close']).diff().fillna(0).values
    out = {}
    window = 256
    n = len(logret)
    for level in range(1, 5):
        coeffs_series = np.full(n, np.nan)
        for i in range(window, n):
            try:
                segment = logret[i - window:i]
                coeffs = pywt.wavedec(segment, 'db4', level=4)
                # coeffs[0] = approx, coeffs[1..4] = details (level 4..1)
                detail_idx = 5 - level
                detail = coeffs[detail_idx]
                coeffs_series[i] = float(np.std(detail))
            except Exception:
                pass
        out[f'wavelet_d4_lvl{level}_std'] = pd.Series(coeffs_series, index=df.index)
    return out


def _patched_build(*args, **kwargs):
    res = _orig_build(*args, **kwargs)
    if not isinstance(res, tuple):
        return res
    df, cols = res
    new = _wavelet_features(df)
    for c, vals in new.items():
        df[c] = vals
        if c not in cols:
            cols.append(c)
    print(f'[C35] wavelet features added: +{len(new)} columns')
    return df, cols


eng.build_all_features = _patched_build
print('[C35] build_all_features patched (+wavelet decomposition)')
