
"""C33 — Bid-ask spread compression z-score. Distinct from quarantined raw
spread_bps (which is in always_disabled_exact). Reads
data/macro_data/orderbook_snapshots.csv, computes hourly spread (bps), then
rolling 168h z-score. Adds 1 feature: ob_spread_compression_zscore.
"""
import os
import numpy as np
import pandas as pd
import crypto_trading_system_ed as eng

_orig_build = eng.build_all_features


def _compute_spread_z(df: pd.DataFrame) -> dict:
    if 'datetime' not in df.columns:
        return {}
    ob_path = os.path.join(os.path.dirname(eng.__file__), 'data', 'macro_data', 'orderbook_snapshots.csv')
    if not os.path.exists(ob_path):
        return {}
    try:
        ob = pd.read_csv(ob_path)
        if 'datetime' not in ob.columns or 'spread_bps' not in ob.columns:
            return {}
        ob['datetime'] = pd.to_datetime(ob['datetime'])
        # ETH only
        if 'symbol' in ob.columns:
            ob = ob[ob['symbol'].str.upper().str.startswith('ETH')]
        ob = ob.sort_values('datetime').drop_duplicates('datetime')
        ob = ob.set_index('datetime')
        spread = ob['spread_bps'].astype(float)
        # rolling z-score
        m = spread.rolling(168, min_periods=24).mean()
        s = spread.rolling(168, min_periods=24).std()
        z = (spread - m) / s.replace(0, np.nan)
        z = z.fillna(0)
        df_dt = pd.to_datetime(df['datetime'])
        aligned = z.reindex(df_dt, method='ffill').fillna(0).values
        return {'ob_spread_compression_zscore': aligned}
    except Exception as e:
        print(f'[C33] FAILED: {e}')
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
    add = _compute_spread_z(df)
    n_added = 0
    for name, vals in add.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            n_added += 1
    if n_added:
        print(f'[C33] spread compression z-score added: +{n_added}')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    return df, all_cols


eng.build_all_features = _patched_build
print('[C33] build_all_features patched')
