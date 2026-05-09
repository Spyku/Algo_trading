
"""SHAP ranking patcher."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

_orig_rank = eng._test_lgbm_importance


def _patched_rank(df_clean, feature_cols, gamma=1.0):
    X = df_clean[feature_cols].values
    y = df_clean['label'].values
    if len(np.unique(y)) < 2:
        return _orig_rank(df_clean, feature_cols, gamma=gamma)
    model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                random_state=42, verbose=-1)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X[-min(500, len(X)):])
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    importance = np.abs(sv).mean(axis=0)
    return pd.DataFrame({'feature': feature_cols, 'importance': importance}).sort_values(
        'importance', ascending=False).reset_index(drop=True)


eng._test_lgbm_importance = _patched_rank
print("[shap_ranking] _test_lgbm_importance replaced with SHAP")
