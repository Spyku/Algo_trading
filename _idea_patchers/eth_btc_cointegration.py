
"""C34 — ETH-BTC cointegration residual (Engle-Granger). Rolling 168h regression
of log(ETH) on log(BTC); residual = log(ETH) - (alpha + beta*log(BTC)).
Distinct from existing xa_btc_lag1h/2h/3h (intraday lead-lag). Cointegration
captures equilibrium-deviation, not short-term lead-lag.
Adds 1 feature: xa_eth_btc_coint_resid_168h.
"""
import os
import numpy as np
import pandas as pd
import crypto_trading_system_ed as eng

_orig_build = eng.build_all_features


def _compute_coint_resid(df: pd.DataFrame) -> dict:
    if 'datetime' not in df.columns or 'close' not in df.columns:
        return {}
    btc_path = os.path.join(os.path.dirname(eng.__file__), 'data', 'btc_hourly_data.csv')
    if not os.path.exists(btc_path):
        return {}
    try:
        btc = pd.read_csv(btc_path)
        btc['datetime'] = pd.to_datetime(btc['datetime'])
        btc = btc.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
        df_dt = pd.to_datetime(df['datetime'])
        btc_close = btc['close'].reindex(df_dt, method='ffill').values.astype(float)
        eth_close = df['close'].astype(float).values
        n = len(df)
        log_eth = np.log(eth_close)
        log_btc = np.log(btc_close)
        window = 168
        resid = np.zeros(n)
        for i in range(window, n):
            x = log_btc[i - window:i]
            y = log_eth[i - window:i]
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            # OLS y = a + b*x
            x_mean = x.mean(); y_mean = y.mean()
            cov = ((x - x_mean) * (y - y_mean)).sum()
            var = ((x - x_mean) ** 2).sum()
            if var == 0:
                continue
            beta = cov / var
            alpha = y_mean - beta * x_mean
            resid[i] = log_eth[i] - (alpha + beta * log_btc[i])
        return {'xa_eth_btc_coint_resid_168h': resid}
    except Exception as e:
        print(f'[C34] FAILED: {e}')
        return {}


def _patched_build(*args, **kwargs):
    result = _orig_build(*args, **kwargs)
    if isinstance(result, tuple) and len(result) == 3:
        df, all_cols, lead_cols = result
    elif isinstance(result, tuple) and len(result) == 2:
        df, all_cols = result
        lead_cols = None
    else:
        return result
    add = _compute_coint_resid(df)
    n_added = 0
    for name, vals in add.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            n_added += 1
    if n_added:
        print(f'[C34] cointegration residual added: +{n_added}')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    return df, all_cols


eng.build_all_features = _patched_build
print('[C34] build_all_features patched')
