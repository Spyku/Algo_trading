
"""C04 Variance Ratio (Lo & MacKinlay 1988) at q=12,24,48 as 3 feature columns.

VR(q) = Var(q-period sum of log-returns) / [q * Var(1-period log-return)]
  VR > 1  -> trending (positive autocorrelation)
  VR < 1  -> mean-reverting
  VR = 1  -> random walk

Computed on rolling 252h window (~10 days) for stable variance estimate.
"""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_all_features


def _vr_features(df: pd.DataFrame) -> dict:
    if 'close' not in df.columns:
        return {}
    logret = np.log(df['close'].astype(float)).diff().fillna(0).values
    n = len(logret)
    out = {}
    rolling_window = 252  # 10.5d at hourly cadence
    for q in [12, 24, 48]:
        vr = np.full(n, np.nan)
        # need at least rolling_window samples + q for the q-period aggregate
        min_i = rolling_window + q
        for i in range(min_i, n):
            seg = logret[i - rolling_window:i]
            var_1 = seg.var(ddof=1)
            if var_1 == 0 or not np.isfinite(var_1):
                continue
            # Non-overlapping q-period aggregates
            n_q = len(seg) // q
            if n_q < 2:
                continue
            q_rets = seg[:n_q * q].reshape(n_q, q).sum(axis=1)
            var_q = q_rets.var(ddof=1)
            if not np.isfinite(var_q):
                continue
            vr[i] = var_q / (q * var_1)
        out[f'vr_{q}'] = vr
    return out


def _patched_build(*args, **kwargs):
    result = _orig_build(*args, **kwargs)
    # build_all_features can return tuple of 2 or 3 — defensive unpack
    if isinstance(result, tuple) and len(result) == 3:
        df, all_cols, lead_cols = result
    elif isinstance(result, tuple) and len(result) == 2:
        df, all_cols = result
        lead_cols = None
    else:
        df = result
        all_cols, lead_cols = None, None
    vr = _vr_features(df)
    added = 0
    for name, vals in vr.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            added += 1
    if added:
        print(f'[C04] VR features added: +{added} columns')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    elif all_cols is not None:
        return df, all_cols
    return df


eng.build_all_features = _patched_build
print('[C04] build_all_features patched (+VR features at q=12,24,48)')
