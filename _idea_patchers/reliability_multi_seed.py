"""Reliability technique #1 — K-trial multi-seed median scoring.

Monkey-patches the engine's inner walk-forward evaluator so each Mode D /
Mode V Optuna trial reports the MEDIAN cum_return across K runs with
different RNG seeds. Standard error of the median drops by ~sqrt(K), so
the optimizer climbs a denoised surface and is less likely to land on a
lucky bump.

Why it matters
--------------
tools/feature_stability_test.py (2026-05-14) measured σ=5.82pp on the
production 8h ETH config under label-noise perturbations. The baseline
single-shot Mode D score is ~2σ BELOW the noise mean — i.e. production
is on the unlucky side of its own noise distribution. With K=5 median
scoring, σ on each Optuna trial drops to ~σ/√5 ≈ 2.6pp; with K=10 it
drops to ~1.8pp. At that point a true +5pp signal is distinguishable
from noise.

Implementation
--------------
Replaces `_deku_eval_with_pruning` with a wrapper that:
  1. Builds K seeded model_factories via _get_deku_diagnostic_models(seed=s)
  2. Runs the original `_deku_eval_with_pruning` once per seed
  3. Returns the result-tuple whose `cum_return` is the median across runs

Internal tuple consistency is preserved: each field (trades, win_rate,
max_dd, raw_pf, adjusted_pf) is taken from the same real backtest run,
not field-averaged.

Env vars
--------
  RELIABILITY_K  number of seeds (default 5)
"""
import os
import crypto_trading_system_ed as eng

K = int(os.environ.get('RELIABILITY_K', '5'))
SEEDS = list(range(42, 42 + K))

_ORIG_DEKU_EVAL = eng._deku_eval_with_pruning


def _factories_seeded(seed: int) -> dict:
    """Mirror _get_deku_diagnostic_models() but with random_state=seed."""
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    def _rf():
        return RandomForestClassifier(
            n_estimators=100, max_depth=4, class_weight='balanced',
            random_state=seed, n_jobs=1,
        )

    def _gb():
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=seed,
        )

    def _xgb():
        return XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            random_state=seed, tree_method='hist', verbosity=0, n_jobs=1,
        )

    def _lr():
        return LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=seed,
        )

    def _lgbm():
        return LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            class_weight='balanced', verbose=-1, random_state=seed,
            device='gpu',
        )

    return {'RF': _rf, 'GB': _gb, 'XGB': _xgb, 'LR': _lr, 'LGBM': _lgbm}


def _deku_eval_median_k(features_np, labels_np, closes_np, combo, window, n,
                         step, model_factories, gamma=1.0, trial=None,
                         horizon=None):
    """Drop-in replacement that runs the inner walk-forward K times with
    different LGBM/RF/XGB/GB/LR seeds. Returns the result whose
    cum_return (tuple index 4) is the median across the K trials."""
    if horizon is None:
        horizon = eng.PREDICTION_HORIZON

    results = []
    for seed in SEEDS:
        factories = _factories_seeded(seed)
        r = _ORIG_DEKU_EVAL(
            features_np, labels_np, closes_np, combo, window, n,
            step, factories, gamma=gamma, trial=None, horizon=horizon,
        )
        if r is not None:
            results.append(r)

    if not results:
        return None

    results.sort(key=lambda rr: rr[4])
    return results[len(results) // 2]


eng._deku_eval_with_pruning = _deku_eval_median_k
print(f'[RELIABILITY_MULTI_SEED] _deku_eval_with_pruning patched (K={K} seeds={SEEDS})')
