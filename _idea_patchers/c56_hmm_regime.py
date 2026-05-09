
"""C56 HMM 2-state regime as feature columns (state probabilities)."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_all_features


def _hmm_state_features(df: pd.DataFrame) -> dict:
    """Fit 2-state Gaussian HMM on log-returns; return state-1 prob per row."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print('[C56] hmmlearn not installed: pip install hmmlearn')
        return {}
    if 'close' not in df.columns:
        return {}
    logret = np.log(df['close']).diff().fillna(0).values.reshape(-1, 1)
    n = len(logret)
    if n < 200:
        return {}
    out_state1 = np.full(n, 0.5)
    out_var = np.full(n, np.nan)
    fit_window = 720  # ~30d
    step = 24
    model = None
    for end in range(fit_window, n + 1, step):
        try:
            model = GaussianHMM(n_components=2, covariance_type='diag',
                                 n_iter=50, random_state=42)
            model.fit(logret[max(0, end - fit_window):end])
            # Determine which state is "high vol" by comparing covars
            covars = model.covars_.flatten()
            high_vol_state = int(np.argmax(covars))
            posterior = model.predict_proba(logret[:end])
            out_state1[max(0, end - step):end] = posterior[max(0, end - step):end, high_vol_state]
            out_var[max(0, end - step):end] = covars[high_vol_state]
        except Exception:
            pass
    return {
        'hmm_high_vol_prob': pd.Series(out_state1, index=df.index),
        'hmm_high_vol_var': pd.Series(out_var, index=df.index),
    }


def _patched_build(*args, **kwargs):
    res = _orig_build(*args, **kwargs)
    if not isinstance(res, tuple):
        return res
    df, cols = res
    new = _hmm_state_features(df)
    for c, vals in new.items():
        df[c] = vals
        if c not in cols:
            cols.append(c)
    print(f'[C56] HMM regime features added: +{len(new)} columns')
    return df, cols


eng.build_all_features = _patched_build
print('[C56] build_all_features patched (+HMM state probability features)')
