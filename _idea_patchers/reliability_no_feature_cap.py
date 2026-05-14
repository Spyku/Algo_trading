"""Reliability technique #4 — drop the n_features hard cap.

Why it matters
--------------
Engine constants pin the feature-selection search to a narrow band:

  GRID_FEATURES = [10, 13, 17, 25]                  (Mode D grid)
  N_FEATURES_RANGE = {SHORT: (4, 40), LONG: (4, 80)} (Mode V refine)

LGBM's internal regularization (reg_alpha, min_child_samples, num_leaves)
already does feature-pruning at split time. Hard-capping the input set
at 25-80 features throws away genuine information AND locks the model
into a fragile subset — because among the ~50 features that *could*
matter, which ones get included depends on which seed LGBM draws.

tools/feature_stability_test.py result (2026-05-14): production ETH 8h
selects 17 features, only 3 of which are load-bearing (permuting them
kills signal). Permuting the other 14 features either does nothing or
*improves* returns by +13-17pp. That's the bump-hopping mechanism:
LGBM's "top-17 importance" pick is determined by seed luck inside a
noise band, not by which features genuinely matter.

V2's partial result (2026-05-14): all 5 alt-importance methods produce
bit-identical Mode V winners at 5h/6h/7h. The bottleneck is not the
ranking method — it's the cap-induced low-dim trap.

Implementation
--------------
Expands Mode D's grid sweep to include larger feature counts:
  GRID_FEATURES: [10, 13, 17, 25] -> [10, 17, 25, 40, 60, 80]
Expands Mode V refine's per-horizon range upper bound:
  N_FEATURES_RANGE: {h: (4, 80)} -> {h: (4, 150)}
  N_FEATURES_RANGE_DEFAULT: (4, 80) -> (4, 150)

Note: this multiplies Mode D's grid eval count from 2*3*4*3=72 to
2*3*6*3=108 (50% more). Combined with multi-seed K=5, that's substantial
runtime cost — budget for it in the orchestrator.
"""
import crypto_trading_system_ed as eng

ORIG_GRID_FEATURES = list(eng.GRID_FEATURES)
ORIG_N_FEATURES_RANGE = dict(eng.N_FEATURES_RANGE)
ORIG_DEFAULT = eng.N_FEATURES_RANGE_DEFAULT

NEW_GRID_FEATURES = [10, 17, 25, 40, 60, 80]
NEW_UPPER = 150  # well above the typical 184-feature universe

eng.GRID_FEATURES = NEW_GRID_FEATURES
eng.N_FEATURES_RANGE = {h: (lo, NEW_UPPER) for h, (lo, _hi) in ORIG_N_FEATURES_RANGE.items()}
eng.N_FEATURES_RANGE_DEFAULT = (ORIG_DEFAULT[0], NEW_UPPER)

print(f'[RELIABILITY_NO_FEATURE_CAP] GRID_FEATURES: {ORIG_GRID_FEATURES} -> {eng.GRID_FEATURES}')
print(f'[RELIABILITY_NO_FEATURE_CAP] N_FEATURES_RANGE upper -> {NEW_UPPER} (per horizon + default)')
