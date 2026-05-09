
"""C38 — Korea/Coinbase premium spread feature.
Reads data/macro_data/korea_premium.csv (downloaded by the runner pre-flight).
Daily resolution forward-filled to hourly. Adds 1 feature:
  korea_premium_pct_lagged   (last available daily premium aligned to hour)
"""
import os
import numpy as np
import pandas as pd
import crypto_trading_system_ed as eng

_orig_build = eng.build_all_features


def _compute_korea_premium(df: pd.DataFrame) -> dict:
    if 'datetime' not in df.columns:
        return {}
    kp_path = os.path.join(os.path.dirname(eng.__file__), 'data', 'macro_data', 'korea_premium.csv')
    if not os.path.exists(kp_path):
        return {}
    try:
        kp = pd.read_csv(kp_path)
        kp['date'] = pd.to_datetime(kp['date']).dt.date
        kp = kp.sort_values('date').drop_duplicates('date').set_index('date')
        df_dt = pd.to_datetime(df['datetime']).dt.date
        aligned = pd.Series(df_dt).map(kp['premium_pct']).fillna(method='ffill').fillna(0).values
        return {'korea_premium_pct_lagged': aligned}
    except Exception as e:
        print(f'[C38] FAILED: {e}')
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
    add = _compute_korea_premium(df)
    n_added = 0
    for name, vals in add.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            n_added += 1
    if n_added:
        print(f'[C38] korea premium features added: +{n_added}')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    return df, all_cols


eng.build_all_features = _patched_build
print('[C38] build_all_features patched')
