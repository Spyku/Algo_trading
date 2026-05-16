"""Reliability fix #1 — align Optuna objective with the winner-selection metric.

Why this matters
----------------
The engine has two different metrics in Mode V:
  1. **Optuna objective** (in `_refine_top_configs`): uses `_compute_optuna_score(result)`
     which returns `adjusted_pf` (APF) when `OPTUNA_METRIC='apf'` (default).
  2. **Winner selector** (in `run_mode_v_parallel` line 7958):
     `score = ret * wr if ret > 0 else ret` at best conf.

These two metrics DISAGREE. B's 6h run is the smoking gun:
  - Refining #1 started from D #6 (RF+LGBM 150h γ=0.999 f=13) with ret=+59.39%, WR=78%
    → score×WR = 46.32 at conf=65%
  - Optuna then explored neighborhood and produced Refined #1 with APF=26.1 (high!)
    but score×WR = 25.71 (lower!)
  - Optuna optimized APF, found a "high-APF" point — but at the cost of score×WR.
  - When the winner selector picked the best by score×WR, D #6 (the un-refined seed)
    beat all 3 Refined candidates.

Fix
---
Add a new `OPTUNA_METRIC = 'ret_wr'` option that computes `cum_return * win_rate/100`,
matching the winner selector. Then set the engine's `OPTUNA_METRIC = 'ret_wr'` so
Mode D + Mode V refine both optimize the same thing the winner selector uses.

Implementation: monkey-patch `_compute_optuna_score` to recognize 'ret_wr', then
flip the metric.

Cost: 0 minutes. Just changes WHICH config Optuna converges to.
"""
import crypto_trading_system_ed as eng

_ORIG_COMPUTE_OPTUNA_SCORE = eng._compute_optuna_score


def _patched_compute_optuna_score(result):
    """Adds 'ret_wr' metric: cum_return * win_rate/100 (matches winner selector)."""
    if eng.OPTUNA_METRIC == 'ret_wr':
        cum_return = result[4]   # cum_return %
        win_rate = result[5]     # win_rate %
        if cum_return > 0:
            return cum_return * (win_rate / 100.0)
        else:
            return cum_return    # keep negative returns unscaled
    return _ORIG_COMPUTE_OPTUNA_SCORE(result)


eng._compute_optuna_score = _patched_compute_optuna_score
eng.OPTUNA_METRIC = 'ret_wr'

print(f"[RELIABILITY_OPTUNA_OBJECTIVE_ALIGN] OPTUNA_METRIC -> 'ret_wr' "
      f"(cum_return * win_rate/100). Mode D grid + Mode V refine now optimize the "
      f"same metric the winner selector uses.")
