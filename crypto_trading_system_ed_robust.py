"""ROBUST scoring variant of crypto_trading_system_ed.

Patches the engine's inner walk-forward evaluator to run K=5 backtests with
different LGBM/RF/XGB/GB/LR seeds, then return the result whose `cum_return`
is the MEDIAN across the K runs. Other tuple fields (trades, win_rate, max_dd,
adjusted_pf, raw_pf) are taken from that same median run so the tuple stays
internally consistent.

Why
---
tools/feature_stability_test.py (2026-05-14) showed σ=5.82pp on label noise
alone for the production 8h ETH config. The D/V optimizer picks the highest
SINGLE-shot score across the 72-eval Mode D grid + 50-trial Optuna refine →
it inevitably lands on lucky local optima ("bumps"), not real peaks. Every
random perturbation in the stability test produced BETTER returns than the
baseline.

The K=5 median fix denoises the SELECTION step. Each candidate's score now
comes from the middle-quality run of 5 seeds rather than one (potentially
lucky/unlucky) seed. Standard error of the median is ~√K tighter than the
single shot, so a peak that survives across 5 seeds is much more likely real.

File mapping (none of these are read by the live trader):
  models/crypto_ed_best_models.csv → models/crypto_ed_best_models_robust.csv
  models/crypto_ed_production.csv  → models/crypto_ed_production_robust.csv
  config/regime_config_ed.json     → config/regime_config_ed_robust.json

Robust files are seeded from prod on first run.

Usage (same CLI as the real engine):
  python crypto_trading_system_ed_robust.py D ETH 8h --replay 1440

`--no-persist` is redundant here (this wrapper IS the no-production path) and
is stripped from argv to avoid a `*_robust_noprod.csv` triple suffix.

Optional env override of K:
  $env:ED_ROBUST_K = "3"   # PowerShell — K=3 instead of default 5
  ED_ROBUST_K=10 python ...  # bash

Runtime cost: K=5 means each Mode D eval runs 5× the inner work. ETH 8h
Mode D on 1440h replay: ~13 min single-seed → ~65 min K=5 on laptop. Mode V
parallel paths are NOT patched (loky workers won't see monkey-patches); only
Mode D's outer loop benefits. Plan: validate with Mode D first.

Validation:
  1. python crypto_trading_system_ed_robust.py D ETH 8h --replay 1440
  2. python tools/feature_stability_test.py  # against the new robust winner
     (you'll need to point the test's PROD_CFG at the new robust CSV row OR
      run a fresh stability test variant — see TODO).

Decision rule:
  σ drops below ~3pp → K=5 median is denoising effectively; expand to all
                       horizons + Mode V via full HRST.
  σ ~ 5pp unchanged → technique #1 alone is not enough; pivot to technique
                       #4 (drop n_features cap) or combine.
"""
import os
import shutil
import sys

_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))

_ROBUST_BEST_MODELS = 'models/crypto_ed_best_models_robust.csv'
_ROBUST_PRODUCTION  = 'models/crypto_ed_production_robust.csv'
_ROBUST_REGIME      = 'config/regime_config_ed_robust.json'

# MODELS_CSV_OVERRIDE is captured at engine import time. Must be set BEFORE
# the `import crypto_trading_system_ed` below.
os.environ['MODELS_CSV_OVERRIDE'] = _ROBUST_BEST_MODELS


def _seed_from_prod(real_rel, robust_rel):
    real = os.path.join(_ENGINE_DIR, real_rel)
    robust = os.path.join(_ENGINE_DIR, robust_rel)
    if os.path.exists(real) and not os.path.exists(robust):
        os.makedirs(os.path.dirname(robust), exist_ok=True)
        shutil.copy2(real, robust)
        print(f"  [robust] seeded {robust_rel} from {real_rel}")


_seed_from_prod('models/crypto_ed_best_models.csv', _ROBUST_BEST_MODELS)
_seed_from_prod('models/crypto_ed_production.csv',  _ROBUST_PRODUCTION)
_seed_from_prod('config/regime_config_ed.json',     _ROBUST_REGIME)

# Strip --no-persist — would otherwise add `_noprod` on top of `_robust`.
if '--no-persist' in sys.argv:
    sys.argv = [a for a in sys.argv if a != '--no-persist']
    print("  [robust] --no-persist is redundant in robust mode (stripped)")

# Workaround: the engine's CLI parser at line ~7347 in main() uses
# `arg.lower().endswith('h')` to detect horizon args, which erroneously
# matches asset names ending in 'h' (ETH). The matched arg then fails the
# `isdigit()` body check, the elif is consumed, and the asset never
# reaches the else branch — so ETH gets silently dropped and the engine
# defaults to all 9 assets. Append a trailing comma to any positional
# 'ETH' so endswith('h') is False, and the parser falls through to the
# asset branch which strips the empty token and recovers ['ETH'].
_FIXED = []
for tok in sys.argv:
    if tok.upper() == 'ETH':
        _FIXED.append('ETH,')
    else:
        _FIXED.append(tok)
if _FIXED != sys.argv:
    sys.argv = _FIXED
    print("  [robust] ETH-parse workaround applied: 'ETH' -> 'ETH,'")

# Auto-tag the grid CSV so the robust run writes
# `models/crypto_ed_grid_<asset>_<h>h_robust.csv` instead of clobbering the
# untagged grid CSV that production Mode V / H read. Skip if user explicitly
# passed --grid-tag.
if '--grid-tag' not in sys.argv:
    sys.argv += ['--grid-tag', 'robust']
    print("  [robust] auto-added --grid-tag robust")

import crypto_trading_system_ed as _ed

# Redirect path constants on the engine module (read by writers throughout).
_ed.PRODUCTION_CSV     = _ROBUST_PRODUCTION
_ed.REGIME_CONFIG_PATH = _ROBUST_REGIME

# ─────────────────────────────────────────────────────────────────────────────
# K-seed median scoring patch
# ─────────────────────────────────────────────────────────────────────────────

ROBUST_K_TRIALS = int(os.environ.get('ED_ROBUST_K', '5'))
ROBUST_SEEDS = list(range(42, 42 + ROBUST_K_TRIALS))

# Capture original BEFORE patching to avoid recursion.
_ORIG_DEKU_EVAL = _ed._deku_eval_with_pruning


def _factories_seeded(seed: int) -> dict:
    """Return model_factories matching engine's _get_deku_diagnostic_models()
    but with every classifier's random_state set to `seed`. Mirrors the
    100-estimator diagnostic config at engine line 1872."""
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
    """Drop-in replacement for _deku_eval_with_pruning that runs the inner
    walk-forward K times with different seeds and returns the median-cum_return
    result. The `model_factories` arg is accepted for signature compatibility
    but ignored — fresh seeded factories are built for each trial."""
    if horizon is None:
        horizon = _ed.PREDICTION_HORIZON

    results = []
    for seed in ROBUST_SEEDS:
        factories = _factories_seeded(seed)
        # Note: pass `trial=None` to every internal call so Optuna's pruner
        # doesn't see partial intermediate scores from K runs and prune
        # mid-trial on noisy partials. Pruning happens at the OUTER level
        # against the median.
        r = _ORIG_DEKU_EVAL(
            features_np, labels_np, closes_np, combo, window, n,
            step, factories, gamma=gamma, trial=None, horizon=horizon,
        )
        if r is not None:
            results.append(r)

    if not results:
        return None

    # Return the result with median cum_return (tuple index 4).
    # Returning a single element (not field-averaged) preserves internal
    # consistency: trades / win_rate / max_dd / pf all come from one real
    # backtest run.
    results.sort(key=lambda rr: rr[4])
    return results[len(results) // 2]


# Install the patch on the engine module so every call site sees the wrapper.
_ed._deku_eval_with_pruning = _deku_eval_median_k


if __name__ == '__main__':
    print("=" * 76)
    print(f"  ROBUST MODE — K={ROBUST_K_TRIALS} multi-seed median scoring")
    print(f"  Patches _deku_eval_with_pruning to median across {ROBUST_K_TRIALS} seeds.")
    print(f"  Seeds: {ROBUST_SEEDS}")
    print(f"  Writes to *_robust.* files — trader does NOT read these")
    print(f"  best_models: {_ROBUST_BEST_MODELS}")
    print(f"  production:  {_ROBUST_PRODUCTION}")
    print(f"  regime:      {_ROBUST_REGIME}")
    print("=" * 76)
    _ed.main()
