
"""C44 quantile regression. Trains LGBM with objective='quantile', alpha=0.7
on forward returns, uses prediction as feature for the main classifier."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_all_features


def _quantile_feature(df: pd.DataFrame, feature_cols: list, alpha: float = 0.7) -> dict:
    """Walk-forward LGBM-quantile predictions on forward return."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return {}
    if '_forward_return' not in df.columns:
        return {}
    if not feature_cols:
        return {}
    X_all = df[feature_cols].copy()
    # Simple: rolling fit every 168h (1 week), predict next 168h
    n = len(df)
    fr = df['_forward_return'].values
    pred = np.full(n, np.nan)
    fit_window = 720
    step = 168
    cols_finite = [c for c in feature_cols if df[c].notna().mean() > 0.5]
    if not cols_finite:
        return {}
    X_all = df[cols_finite].fillna(0).values
    for end in range(fit_window, n, step):
        try:
            X_tr = X_all[max(0, end - fit_window):end]
            y_tr = fr[max(0, end - fit_window):end]
            mask = ~np.isnan(y_tr)
            if mask.sum() < 100:
                continue
            mdl = LGBMRegressor(
                objective='quantile', alpha=alpha,
                n_estimators=100, max_depth=4, learning_rate=0.05,
                verbose=-1, random_state=42, n_jobs=1,
            )
            mdl.fit(X_tr[mask], y_tr[mask])
            X_pred = X_all[end:min(n, end + step)]
            pred[end:min(n, end + step)] = mdl.predict(X_pred)
        except Exception:
            pass
    return {'qreg_q70_predict': pd.Series(pred, index=df.index)}


def _patched_build(*args, **kwargs):
    res = _orig_build(*args, **kwargs)
    if not isinstance(res, tuple):
        return res
    df, cols = res
    new = _quantile_feature(df, list(cols))
    for c, vals in new.items():
        df[c] = vals
        if c not in cols:
            cols.append(c)
    print(f'[C44] quantile-regressor feature added: +{len(new)} columns')
    return df, cols


eng.build_all_features = _patched_build
print('[C44] build_all_features patched (+LGBM quantile q=0.7 prediction)')
