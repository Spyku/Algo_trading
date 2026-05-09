
"""C47 vol-adjusted label. Replace binary (ret > 2×fee) with (ret/σ_h > 0.5)."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_hourly_features

VOL_ADJ_K = 0.5  # threshold on Sharpe-like ratio (ret_h / σ_h)


def _patched_hourly(*args, **kwargs):
    res = _orig_build(*args, **kwargs)
    if not isinstance(res, tuple):
        return res
    df, feature_cols = res
    if 'label' not in df.columns or '_forward_return' not in df.columns:
        return df, feature_cols
    # Recompute label = (forward_ret / forward_vol > k)
    fr = df['_forward_return']
    # Backward-looking realized vol (proxy for forward σ)
    if 'volatility_48h' in df.columns:
        sigma = df['volatility_48h']
    else:
        logret = np.log(df['close']).diff()
        sigma = logret.rolling(48, min_periods=12).std()
    sharpe_like = fr / (sigma + 1e-9)
    new_label = (sharpe_like > VOL_ADJ_K).astype(float)
    df['label'] = new_label.where(fr.notna(), np.nan)
    print(f'[C47] vol-adjusted label active: k={VOL_ADJ_K}, positives={int(new_label.sum())}/{len(new_label)}')
    return df, feature_cols


eng.build_hourly_features = _patched_hourly
print('[C47] build_hourly_features patched (vol-adjusted label)')
