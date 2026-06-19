---
name: inference-change
description: >
  Use when changing the LIVE inference / signal-generation path — inference-row
  selection, embargo, NaN/ffill policy, feature build at serve time, train-window
  edge, or anything in crypto_live_trader_ed.py / compute_signal_core. The F2 gates;
  breaking these produces SILENTLY wrong live signals.
---

# F2 — Live inference correctness

*Mirrors CLAUDE.md "## Critical Rules" F2 — gates 6–11; gates 7–8 below are the actionable demoted-reference items (Set-D feature_override, Windows SSL).*

1. **No embargo in live.** Live training window is `[train_start, n-1]` (all labelled data). The `train_end = i - horizon` purge is backtest/CV ONLY — live has no future to leak, so embargoing live just discards the freshest, most-predictive rows. Per Lopez de Prado. [[feedback-no-live-embargo]]
2. **Mirror changes to the shadow.** Any change to the live inference path MUST be mirrored in `crypto_live_shadow.py` (two separate codepaths). Grep `USE_CLOSED_BAR_FOR_INFERENCE` to find the mirror block in both. The shadow is the primary live-correctness tripwire; an un-mirrored change silently tanks the match rate (proven Jun-2026, 100%→48%).
3. **Validate on GPU.** Run sanity / engine-vs-trader parity on GPU (`LGBM_DEVICE='gpu'`, mirrors the live trader). `--cpu-lgbm` is a no-GPU fallback only (~3-5pt parity gap).
4. **Caches must be mtime-aware.** Any module-level cache holding file content stores `(mtime, df)` and re-reads when mtime advances. Banned: `if fn in cache: return cache[fn]` with no mtime compare (froze macro features at startup — TODO 0527).
5. **Address signals by SLOT, not horizon.** `signal_log.csv` is regime-anchored: slot 1 = bull, slot 2 = bear. Use `h_1/sig_1`, `h_2/sig_2`, never `HORIZON_SHORT/LONG`.
6. **Refuse the cycle on a structural fault.** Return None / skip on `regime=='error'`, `n_train < window+100`, or any non-sparse feature breaching its staleness SLA — NEVER fall back to a default config/horizon (the "86%-pin" bug). Mirror these guards in the shadow (gate 2).

## Calling-convention gotchas (when writing/changing a caller)
7. **Set-D `feature_override`.** Any caller of `generate_signals()` for a feature_set-D model MUST pass `feature_override=config['optimal_features'].split(',')`, else it silently falls back to the default feature set (wrong features, no error).
8. **Windows SSL.** Any NEW live file that makes HTTP calls must apply `ssl._create_unverified_context()` (as the existing live files + downloaders do) — Windows cert-store friction otherwise.
