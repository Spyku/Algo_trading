
"""Per-regime feature-set patcher.

Boosts technical features in LGBM ranking for short horizons (bull-bias),
boosts macro features for long horizons (bear-bias). Implemented as a
post-LGBM-ranking reorder.
"""
import crypto_trading_system_ed as eng

_orig_rank = eng._test_lgbm_importance

TECHNICAL_PREFIXES = ('logret_', 'sma', 'price_to_sma', 'spread_', 'volatility_',
                      'gk_volatility', 'rsi_', 'adx_', 'plus_di', 'minus_di',
                      'bb_position', 'stoch_k', 'price_accel', 'atr_', 'zscore_',
                      'vol_ratio', 'intraday_range', 'volume_ratio', 'hour_',
                      'dow_')

MACRO_PREFIXES = ('m_', 'xa_', 'fg_', 'oc_', 'deriv_', 'pysr_')


def _patched_rank(df_clean, feature_cols, gamma=1.0, horizon=None):
    """Post-rank reorder by horizon-conditional category boost."""
    rank_df = _orig_rank(df_clean, feature_cols, gamma=gamma)
    h = horizon or _CURRENT_HORIZON.get('h', 6)

    if h <= 6:  # bull-bias: boost technical
        boost = TECHNICAL_PREFIXES
        nerf = MACRO_PREFIXES
    else:       # bear-bias: boost macro
        boost = MACRO_PREFIXES
        nerf = TECHNICAL_PREFIXES

    def _category_score(feat: str) -> float:
        if any(feat.startswith(p) for p in boost):
            return 1.0
        if any(feat.startswith(p) for p in nerf):
            return -0.5
        return 0.0

    rank_df = rank_df.copy()
    rank_df['_cat'] = rank_df['feature'].apply(_category_score)
    rank_df['importance'] = rank_df['importance'] * (1 + 0.3 * rank_df['_cat'])
    rank_df = rank_df.drop(columns=['_cat']).sort_values('importance', ascending=False).reset_index(drop=True)
    return rank_df


_CURRENT_HORIZON = {'h': 6}
eng._test_lgbm_importance = _patched_rank
print("[per_regime_features] _test_lgbm_importance patched")
