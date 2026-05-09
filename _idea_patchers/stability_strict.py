
"""Stability filter v2 patcher (stricter)."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_rank = eng._test_lgbm_importance
RANK_THRESH = 50  # only drop if max_rank - min_rank > 50 (was 30)


def _patched_rank(df_clean, feature_cols, gamma=1.0):
    """Compute ranking on 3 sub-windows; drop unstable, then full ranking."""
    n = len(df_clean)
    if n < 600:
        return _orig_rank(df_clean, feature_cols, gamma=gamma)

    splits = [
        df_clean.iloc[:n // 2],            # oldest 50%
        df_clean.iloc[n // 4 : 3 * n // 4],  # middle 50%
        df_clean.iloc[n // 2:],            # newest 50%
    ]
    sub_ranks = []
    for sub in splits:
        try:
            r = _orig_rank(sub, feature_cols, gamma=gamma)
            r = r.reset_index(drop=True)
            r['rank'] = r.index
            sub_ranks.append(dict(zip(r['feature'], r['rank'])))
        except Exception:
            pass

    if len(sub_ranks) < 2:
        return _orig_rank(df_clean, feature_cols, gamma=gamma)

    full = _orig_rank(df_clean, feature_cols, gamma=gamma)
    stable = []
    for feat in full['feature']:
        ranks = [sr.get(feat, len(feature_cols)) for sr in sub_ranks]
        if max(ranks) - min(ranks) <= RANK_THRESH:
            stable.append(feat)
    if len(stable) >= 5:
        full = full[full['feature'].isin(stable)].reset_index(drop=True)
    return full


eng._test_lgbm_importance = _patched_rank
print(f"[stability_strict] thresh={RANK_THRESH}")
