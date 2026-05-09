
"""C39 — Binance Futures long/short account ratio.
Reads data/macro_data/binance_long_short_ethusdt.csv (downloaded by runner
pre-flight). Adds:
  ls_ratio                   raw ratio
  ls_ratio_chg24h            24h change
  ls_ratio_zscore_168h       rolling z-score
"""
import os
import numpy as np
import pandas as pd
import crypto_trading_system_ed as eng

_orig_build = eng.build_all_features


def _compute_ls_ratio(df: pd.DataFrame) -> dict:
    if 'datetime' not in df.columns:
        return {}
    ls_path = os.path.join(os.path.dirname(eng.__file__), 'data', 'macro_data', 'binance_long_short_ethusdt.csv')
    if not os.path.exists(ls_path):
        return {}
    try:
        ls = pd.read_csv(ls_path)
        ls['datetime'] = pd.to_datetime(ls['datetime'])
        ls = ls.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
        ratio = ls['long_short_ratio'].astype(float)
        chg24 = ratio.pct_change(24).fillna(0)
        m = ratio.rolling(168, min_periods=24).mean()
        s = ratio.rolling(168, min_periods=24).std()
        z = ((ratio - m) / s.replace(0, np.nan)).fillna(0)
        df_dt = pd.to_datetime(df['datetime'])
        out = {
            'ls_ratio': ratio.reindex(df_dt, method='ffill').fillna(0).values,
            'ls_ratio_chg24h': chg24.reindex(df_dt, method='ffill').fillna(0).values,
            'ls_ratio_zscore_168h': z.reindex(df_dt, method='ffill').fillna(0).values,
        }
        return out
    except Exception as e:
        print(f'[C39] FAILED: {e}')
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
    add = _compute_ls_ratio(df)
    n_added = 0
    for name, vals in add.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            n_added += 1
    if n_added:
        print(f'[C39] long/short ratio features added: +{n_added}')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    return df, all_cols


eng.build_all_features = _patched_build
print('[C39] build_all_features patched')
