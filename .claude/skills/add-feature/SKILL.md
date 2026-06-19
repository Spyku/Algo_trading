---
name: add-feature
description: >
  Use when ADDING or CHANGING a computed feature (a new technical indicator, ratio,
  transform, or PySR input) derived from data already in the pipeline — distinct from
  /data-source (a new raw feed). Covers the dual-builder mirror, PySR re-discovery, lag
  safety, and gated validation. Most features are already dead — prior-art check first.
---

# Adding / changing a computed feature

*Cross-cutting — F4 (builder/lag/consistency) + F6 (gated validation, PySR-ablation). Sibling of /data-source (which is for a new RAW feed).*

Pipeline touch points: `build_all_features` ([crypto_trading_system_faye.py:1732](../../../crypto_trading_system_faye.py#L1732)), technical block (~line 1329), `_compute_pysr_features` ([:2345](../../../crypto_trading_system_faye.py#L2345)).

1. **Prior-art check FIRST (it's probably already dead).** 127/206 features (62%) are never selected; faster-window variants of live features are DEAD (3h/4h/5h/6h all cost −22 to −27.5pp gated, even when importance-ranked #2–#4). Grep the Engine Reference Card feature grades in CLAUDE.md + ARCHIVED_LOG.md (C-scoreboard) for the idea before building. (See /validate-research step 0.)
2. **If it needs a NEW raw data source** → that's `/data-source` first (cadence, merge key, lag, audit). This skill is for features computed from data already merged.
3. **Add the computation in `build_all_features`** — and **mirror it in BOTH** `crypto_trading_system_faye.py` (modeling) AND `crypto_trading_system_ed.py` (live inference). Diverging builders = live ≠ backtest. Know any deliberate gap.
4. **Lag safety.** A feature derived from already-lagged daily/on-chain columns inherits the lag for free (computed after the merge). Don't reach back to an un-lagged source. If unsure, run `tools/audit_feature_lag.py` (or `_fujiwara` for sub-hourly) → 0 violations.
5. **NaN / warmup.** Long-window rolls add head-NaN; sparse inputs stay NaN (LGBM handles it — never force into `dropna()`). Check you didn't inflate `dropna` row loss. Unique name, no collision.
6. **To let PySR USE it, re-run Mode P** (PySR is computed after the merge; a new column is ignored until re-discovered). Then retrain (regen PySR ⇒ HRS/DV — see /promote-check gate 3).
7. **VALIDATE through the gated sim, NEVER importance.** Importance ≠ performance — judge only via the real gated backtest (`tools/bt_*.py`, Mode V/HRS, or the shadow). See /validate-research.
8. **Removing a feature is NOT a prefix-disable** — PySR embeds raw features; re-run Mode P with it excluded (Trim A vs Trim B). See /validate-research gate 4.
9. **Promote** only via /promote-check (flat, leakage, atomic), and only if the gated sim earns it. Don't auto-launch the long confirming run.
