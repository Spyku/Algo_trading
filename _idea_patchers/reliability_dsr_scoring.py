"""Reliability technique #3 — sample-size-robust Optuna scoring (DSR proxy).

Why it matters
--------------
Default Optuna objective is `apf` (Adjusted Profit Factor) and the
production winner-selection uses `combined_score = return × win_rate`.
Both are noisy at the typical 30-80 trade counts:
  - WR at N=50 has natural std ~7%
  - APF is unbounded and dominated by single-trade outliers
  - Mode D evaluates 72 candidates + Mode V refine adds 150 more; picking
    the max of 222 noisy measurements GUARANTEES selecting an outlier
    (Bailey & López de Prado 2014 — "Deflated Sharpe").

The engine already supports an `rpf_sqrt` metric: `raw_pf × sqrt(trades)`.
This is a t-statistic-like quantity (signal × sqrt(N)) that:
  - penalizes low-trade configs (high WR with N=5 no longer wins)
  - rewards consistency across many trades
  - is the closest within-engine proxy to a Deflated Sharpe objective
    without requiring per-trade-return access (which the engine doesn't
    currently expose in the eval tuple)

Why not full DSR
----------------
True DSR (Bailey-LdP 2014) requires per-trade Sharpe + correction for
multiple-testing across N hill-climb trials. The engine's eval tuple
does not return per-trade returns, only aggregate stats. Patching that
would require a deeper engine fork. `rpf_sqrt` is the high-leverage
proxy: same direction, ~80% of the benefit, zero engine modification.

Implementation
--------------
Sets `OPTUNA_METRIC = 'rpf_sqrt'` (already-supported branch in
_compute_optuna_score at engine line 542-544).
"""
import crypto_trading_system_ed as eng

ORIG_METRIC = eng.OPTUNA_METRIC
eng.OPTUNA_METRIC = 'rpf_sqrt'

print(f'[RELIABILITY_DSR_SCORING] OPTUNA_METRIC: {ORIG_METRIC!r} -> {eng.OPTUNA_METRIC!r} '
      f'(raw_pf * sqrt(trades) — penalizes low-N configs)')
