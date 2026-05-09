
"""C32 — Liquidation cascade event PROXY (no direct liquidation feed).

True liquidation data needs CoinGlass paid API. Proxy a cascade event from
existing signals: large negative 1h return AND 24h vol > rolling p90 AND OI
1h change < -3%. Adds 3 features:
  liq_cascade_event_h     binary (1 if cascade in current hour)
  liq_cascade_count_24h   count of cascade events in last 24h
  liq_cascade_dist_h      hours since most recent cascade (capped at 168)
"""
import os
import numpy as np
import pandas as pd
import crypto_trading_system_ed as eng

_orig_build = eng.build_all_features


def _compute_cascade(df: pd.DataFrame) -> dict:
    if 'close' not in df.columns:
        return {}
    out = {}
    n = len(df)
    logret_1h = np.log(df['close'].astype(float)).diff().fillna(0).values
    vol_24h = pd.Series(logret_1h).rolling(24).std().values
    vol_24h_p90 = pd.Series(vol_24h).rolling(168, min_periods=24).quantile(0.90).values
    # OI 1h change — pull from derivatives_eth.csv if available
    oi_chg_1h = np.zeros(n)
    deriv_path = os.path.join(os.path.dirname(eng.__file__), 'data', 'macro_data', 'derivatives_eth.csv')
    if os.path.exists(deriv_path):
        try:
            ddf = pd.read_csv(deriv_path)
            if 'datetime' in ddf.columns and 'open_interest' in ddf.columns:
                ddf['datetime'] = pd.to_datetime(ddf['datetime'])
                ddf = ddf.set_index('datetime').sort_index()
                ddf['oi_chg'] = ddf['open_interest'].pct_change().fillna(0)
                df_dt = pd.to_datetime(df['datetime']) if 'datetime' in df.columns else None
                if df_dt is not None:
                    aligned = ddf['oi_chg'].reindex(df_dt, method='ffill').values
                    oi_chg_1h = np.nan_to_num(aligned, nan=0.0)
        except Exception:
            pass
    cascade_thresh_ret = -0.015  # -1.5% 1h return
    cascade_thresh_oi = -0.03    # -3% 1h OI drop
    is_cascade = (
        (logret_1h < cascade_thresh_ret)
        & (vol_24h > vol_24h_p90)
        & (oi_chg_1h < cascade_thresh_oi)
    ).astype(np.float64)
    out['liq_cascade_event_h'] = is_cascade
    out['liq_cascade_count_24h'] = pd.Series(is_cascade).rolling(24, min_periods=1).sum().values
    # hours since most recent cascade, capped at 168
    dist = np.full(n, 168.0)
    last = -1
    for i in range(n):
        if is_cascade[i] > 0.5:
            last = i
        if last >= 0:
            dist[i] = min(168.0, float(i - last))
    out['liq_cascade_dist_h'] = dist
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
    add = _compute_cascade(df)
    n_added = 0
    for name, vals in add.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            n_added += 1
    if n_added:
        print(f'[C32] liquidation cascade features added: +{n_added}')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    return df, all_cols


eng.build_all_features = _patched_build
print('[C32] build_all_features patched')
