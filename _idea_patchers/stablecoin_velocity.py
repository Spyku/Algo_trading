
"""C37 — Stablecoin issuance velocity (1st difference of mcap).

Existing stable_mcap_chg1d / chg7d / zscore are LEVEL-based. Velocity is the
1st difference, captures issuance/redemption rate. CLAUDE.md notes the existing
stablecoin_flows.csv is already loaded. Adds:
  stable_mcap_velocity_1d   1st diff of mcap_total (1-day)
  stable_mcap_velocity_7d   1st diff of mcap_total (7-day rolling)
"""
import os
import numpy as np
import pandas as pd
import crypto_trading_system_ed as eng

_orig_build = eng.build_all_features


def _compute_velocity(df: pd.DataFrame) -> dict:
    if 'datetime' not in df.columns:
        return {}
    sf_path = os.path.join(os.path.dirname(eng.__file__), 'data', 'macro_data', 'stablecoin_flows.csv')
    if not os.path.exists(sf_path):
        return {}
    try:
        sf = pd.read_csv(sf_path)
        # column may be 'date' or 'datetime'
        date_col = 'date' if 'date' in sf.columns else ('datetime' if 'datetime' in sf.columns else None)
        if date_col is None:
            return {}
        sf['date'] = pd.to_datetime(sf[date_col]).dt.date
        # use total mcap (USDT + USDC if present)
        mcap_col = None
        for c in ['mcap_total', 'total_mcap', 'usdt_usdc_mcap']:
            if c in sf.columns:
                mcap_col = c; break
        if mcap_col is None:
            cands = [c for c in sf.columns if 'mcap' in c.lower()]
            if cands:
                sf['_mcap'] = sf[cands].sum(axis=1)
                mcap_col = '_mcap'
            else:
                return {}
        sf = sf.sort_values('date').drop_duplicates('date')
        sf['velocity_1d'] = sf[mcap_col].diff()
        sf['velocity_7d'] = sf[mcap_col].diff(7)
        sf = sf.set_index('date')
        df_dt = pd.to_datetime(df['datetime']).dt.date
        v1 = pd.Series(df_dt).map(sf['velocity_1d']).fillna(0).values
        v7 = pd.Series(df_dt).map(sf['velocity_7d']).fillna(0).values
        return {
            'stable_mcap_velocity_1d': v1,
            'stable_mcap_velocity_7d': v7,
        }
    except Exception as e:
        print(f'[C37] FAILED: {e}')
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
    add = _compute_velocity(df)
    n_added = 0
    for name, vals in add.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            n_added += 1
    if n_added:
        print(f'[C37] stablecoin velocity features added: +{n_added}')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    return df, all_cols


eng.build_all_features = _patched_build
print('[C37] build_all_features patched')
