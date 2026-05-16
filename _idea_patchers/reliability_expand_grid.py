"""Reliability fixes #2 + #3 — broaden Mode D's grid for diversity and coverage.

Why this matters
----------------
Two related defects in current Mode D + Mode V flow:

  #2 Cluster diversity: the engine's nested `_diversity_key` enforces
     diversity across (window, gamma_band, feature_band) but NOT across
     model family (combo). At 6h, all 6 top-6 D candidates were either
     XGB+LGBM 72h OR RF+LGBM 150h — never a third family.

  #3 Successive Halving / broader exploration: only top 6 D candidates
     feed Step 1. If the optimal config is rank 7-15 by APF, it's never
     evaluated at conf-sweep level.

Direct fix to the nested `_diversity_key` and `DOOHAN_SAVE_TOP_N` is hard
because they're local inside `run_mode_d`. Indirect fix: **broaden the
GRID itself** so that:
  - More grid candidates produce more diverse top-6 naturally
  - More grid points span more (combo × window × features) basins
  - Optuna refine has more raw material to work with

Trade-off: 2× more grid evals = ~2× Mode D compute.

Implementation
--------------
Module-level globals `GRID_COMBOS`, `GRID_WINDOWS`, `GRID_FEATURES`,
`GRID_GAMMAS` are patched. The engine reads these at Mode D dispatch time.

Combo: keep [RF+LGBM, XGB+LGBM]. RF+XGB has 0/20 wins history per the
engine comments at line 3984 — don't add without evidence.
Windows: 3 → 5 (re-add 200, 250 — previously trimmed for "0 wins" but
that was on single-seed scoring; K=5 multi-seed may favor longer windows).
Features: 4 → 5 (re-add 40 from no_feature_cap territory).
Gammas: 3 → 3 (keep — all 3 picked at meaningful rates).

Grid eval count: 2 × 5 × 5 × 3 = 150 (vs current 72 = 2.08× more).

With K=5 multi-seed multiplier: 150 × 5 = 750 internal backtests per
horizon (vs current 360 = same 2× multiplier).

Cost: ~+50 min Mode D total across 4 horizons. Acceptable inside the
~7h HRST budget.

Benefit: top-N D candidates (the 6 fed into Step 1) span more (combo,
window, features) clusters; Refine has more diverse seeds.
"""
import crypto_trading_system_ed as eng

# Capture originals for the print
_ORIG_COMBOS = list(eng.GRID_COMBOS)
_ORIG_WINDOWS = list(eng.GRID_WINDOWS)
_ORIG_FEATURES = list(eng.GRID_FEATURES)
_ORIG_GAMMAS = list(eng.GRID_GAMMAS)

# Patched (broader) grid
NEW_COMBOS = ['RF+LGBM', 'XGB+LGBM']   # unchanged — RF+XGB has 0/20 hist wins
NEW_WINDOWS = [72, 100, 150, 200, 250]  # was [72, 100, 150]; re-add 200, 250
NEW_FEATURES = [10, 17, 25, 40, 60]     # was [10, 13, 17, 25]; broaden upper
NEW_GAMMAS = [0.999, 0.997, 0.995]      # unchanged

eng.GRID_COMBOS = NEW_COMBOS
eng.GRID_WINDOWS = NEW_WINDOWS
eng.GRID_FEATURES = NEW_FEATURES
eng.GRID_GAMMAS = NEW_GAMMAS

old_evals = len(_ORIG_COMBOS) * len(_ORIG_WINDOWS) * len(_ORIG_FEATURES) * len(_ORIG_GAMMAS)
new_evals = len(NEW_COMBOS) * len(NEW_WINDOWS) * len(NEW_FEATURES) * len(NEW_GAMMAS)

print(f"[RELIABILITY_EXPAND_GRID] GRID_COMBOS:   {_ORIG_COMBOS} -> {NEW_COMBOS}")
print(f"[RELIABILITY_EXPAND_GRID] GRID_WINDOWS:  {_ORIG_WINDOWS} -> {NEW_WINDOWS}")
print(f"[RELIABILITY_EXPAND_GRID] GRID_FEATURES: {_ORIG_FEATURES} -> {NEW_FEATURES}")
print(f"[RELIABILITY_EXPAND_GRID] GRID_GAMMAS:   {_ORIG_GAMMAS} -> {NEW_GAMMAS}")
print(f"[RELIABILITY_EXPAND_GRID] eval count: {old_evals} -> {new_evals} (+{(new_evals/old_evals-1)*100:.0f}%)")
