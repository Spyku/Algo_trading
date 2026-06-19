---
name: data-source
description: >
  Use when adding or changing a data source / feature feed — a new macro/on-chain/
  derivatives/orderbook CSV, a new download, or changing a merge. The F4 data-integrity
  gates + the "Adding a New Data Source" checklist. Wrong merge key = backtest
  clairvoyance, invisible until live divergence.
---

# F4 — Data-source integrity (Adding a New Data Source)

*Mirrors CLAUDE.md "## Critical Rules" F4 — gates 16–17 + the 9-item "Adding a New Data Source" checklist.*

Canonical builder: `crypto_trading_system_faye.py::build_all_features`.

1. **Cadence → merge key.** Daily source (one row/`date`) → merge on `_merge_date` (inherits the publish lag). Hourly/intraday snapshot → merge on `_merge_dt` (floored hour, NO lag). Wrong key = wrong lag: daily-on-`_merge_dt` leaks the future; hourly-on-`_merge_date` over-lags. **Sub-hourly forks: hourly-cadence sources still floor to the hour; daily lag is in DAYS and broadcasts to all intraday bars.**
2. **Availability lag (daily only).** Daily feeds publish AFTER the day they describe. `_merge_date` already applies `DAILY_MERGE_LAG_DAYS=1`. Slower feeds get a deeper key (on-chain `ONCHAIN_MERGE_LAG_DAYS=2`, CoinMetrics lands ~midday D+1). Test: a 07:00 decision sees only days fully published by 07:00.
3. **Preserve history.** Any re-pull + write goes through `_dedup_preserve_history` / `_merge_preserve_history` (`keep='first'` historical, `keep='last'` current). Bare `to_csv` re-introduces the drift bug (upstream revisions overwrite originally-observed values).
4. **Cache mtime** — module-level caches re-read on mtime advance (see /inference-change gate 4).
5. **Sparse history — quarantine <60 days** in `always_disabled_exact`; schedule a re-enable A/B at start+60d. LGBM handles NaN natively — never force a sparse column into `dropna()`.
6. **PySR inheritance** is automatic (computed after the merge). To have PySR USE a new feature, re-run Mode P.
7. **Builder + live consistency** — the merge lives in BOTH `faye` (modeling) and `crypto_trading_system_ed.py` (live inference); keep identical or know the deliberate gap.
8. **VERIFY:** `python tools/audit_feature_lag.py` (sub-hourly fork: `tools/audit_feature_lag_fujiwara.py --candle 15|30`) → require **0 violations** before promotion.
9. **No hardcoded horizons/slots** if it feeds a signal-log slot (address by slot).
