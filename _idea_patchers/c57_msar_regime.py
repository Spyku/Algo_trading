
"""C57 Markov-switching AR(1) — adds 2-state probability as feature."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_all_features


def _msar_features(df: pd.DataFrame) -> dict:
    try:
        from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
    except ImportError:
        print('[C57] statsmodels MS-AR not available')
        return {}
    if 'close' not in df.columns:
        return {}
    logret = np.log(df['close']).diff().dropna().values * 100  # %
    n_full = len(df)
    out_state1 = np.full(n_full, 0.5)
    fit_window = 720
    step = 48
    # Align: logret has n_full - 1 entries (diff dropped first)
    offset = n_full - len(logret)
    for end in range(fit_window, len(logret) + 1, step):
        try:
            seg = logret[max(0, end - fit_window):end]
            model = MarkovAutoregression(seg, k_regimes=2, order=1, switching_ar=True)
            res = model.fit(disp=False, maxiter=50)
            # smoothed prob of high-vol regime (assume regime 1 = higher variance)
            sp = res.smoothed_marginal_probabilities[1]
            offs = end - len(sp)
            slot_start = max(0, end - step) + offset
            slot_end = end + offset
            sp_slot = sp[max(0, len(sp) - step):]
            n_slot = min(len(sp_slot), slot_end - slot_start)
            if n_slot > 0:
                out_state1[slot_start:slot_start + n_slot] = sp_slot[-n_slot:]
        except Exception:
            pass
    return {'msar_state1_prob': pd.Series(out_state1, index=df.index)}


def _patched_build(*args, **kwargs):
    res = _orig_build(*args, **kwargs)
    if not isinstance(res, tuple):
        return res
    df, cols = res
    new = _msar_features(df)
    for c, vals in new.items():
        df[c] = vals
        if c not in cols:
            cols.append(c)
    print(f'[C57] MS-AR features added: +{len(new)} columns')
    return df, cols


eng.build_all_features = _patched_build
print('[C57] build_all_features patched (+MS-AR state probability)')
