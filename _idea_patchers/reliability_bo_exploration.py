"""Reliability fix #4 — swap Optuna's TPE sampler for an exploration-friendly one.

Why this matters
----------------
Optuna's default TPE (Tree-structured Parzen Estimator) is biased toward
exploitation around already-good regions. In B's 6h run, Optuna started from
D #6 (RF+LGBM 150h γ=0.999 f=13) and refined to RF+LGBM 137h γ=0.998 f=13 —
a tiny step. It barely explored. Result: the seed point (D #6, un-refined)
beat all 3 refined candidates.

The literature (Snoek 2012 "Practical Bayesian Optimization"; Hutter 2011
"SMAC") suggests exploration-friendly acquisition functions or samplers
help escape local basins.

Fix
---
Replace `TPESampler` with `CmaEsSampler` (Covariance Matrix Adaptation
Evolution Strategy). CMA-ES is well-suited for continuous + low-dim search
spaces with multimodal landscapes. It explores aggressively in the first
few generations, then narrows around promising regions.

Implementation: monkey-patch the engine's TPESampler reference so the
existing `_refine_top_configs` code instantiates CmaEsSampler instead.

Note: CMA-ES has a `restart_strategy='ipop'` option that explicitly
restarts with diversified population if stuck — directly addresses our
"local basin" problem.

Cost: ~+10-30% Step 2 runtime (Optuna CMA-ES is slower than TPE per trial).
Precision: +3-7pp expected from broader basin exploration.
"""
import optuna
import crypto_trading_system_ed as eng

_ORIG_TPESAMPLER = optuna.samplers.TPESampler


class _PatchedSampler(optuna.samplers.CmaEsSampler):
    """CmaEsSampler that accepts TPESampler's seed kwarg for drop-in compatibility."""
    def __init__(self, seed=None, **kwargs):
        # CmaEsSampler accepts 'seed' directly; this wrapper just normalizes signatures.
        super().__init__(seed=seed, restart_strategy='ipop', **kwargs)


# Monkey-patch optuna.samplers.TPESampler so the existing code at engine
# lines 5193 and 8521 picks up CMA-ES instead. This is a global patch but
# the F_optimized variant is isolated — outside this run, the engine is
# unmodified.
optuna.samplers.TPESampler = _PatchedSampler

# Also patch eng's reference if cached
try:
    eng.optuna.samplers.TPESampler = _PatchedSampler
except AttributeError:
    pass

print(f"[RELIABILITY_BO_EXPLORATION] Optuna TPESampler -> CmaEsSampler(restart='ipop'). "
      f"Optuna refine will explore more aggressively across the search space.")
