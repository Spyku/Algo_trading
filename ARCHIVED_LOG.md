# ARCHIVED_LOG.md — Audit trail of tested/shelved decisions

This file preserves the full historical decision trail of the engine: tested/shelved/promoted research items, the canonical idea scoreboard (C01-C87), MERGED TOPICS (closed research arcs), and per-day batch closures. Split out of TODO.md on 2026-05-19 to keep TODO.md focused on active work only.

**This file is reference material — append-only.** Don't edit closed entries; if a new piece of evidence overturns a verdict, add a new entry that references the old one rather than rewriting history.

**Cross-references:**
- [CLAUDE.md](CLAUDE.md) — stable system reference (engine, machine setup, commands, critical rules)
- [TODO.md](TODO.md) — active work only

---

### ✅ CLOSED 2026-06-13 — Label-threshold (fee-aware) recalibration: WASH across all arms → KEEP 0.22%. [C87]

**Hypothesis (owner):** maker execution now costs ~0 (realized round-trip on recent live ETH ≈ 0%), so the fee-aware label's `2×TRADING_FEE = 0.22%` "worth-it" bar may be too conservative — skipping small moves that are actually profitable. Lower it to capture them?

**Method:** FAYE Mode D ETH 5,6,7,8h `--replay 960 --no-persist`, 3 arms (`--label-threshold` 0.0022 / 0.0010 / 0.0005), all pinned to the **frozen snapshot** `data/_v2_snapshot_ablation_20260613_112045` (zero drift — every arm reads identical data; satisfies the owner's "same data, clear answer" requirement). Robust metric: top-5-median `return_pct` among ≥20-trade configs per horizon (NOT noisy top-1-APF). Tool: `tools/test_label_threshold_ab.py`.

| H | BASE 0.0022 | LT0010 0.0010 | LT0005 0.0005 |
|---|---|---|---|
| 5h | 4.55% | +2.38pp | +3.98pp |
| 6h | 6.81% | −2.83pp | −1.89pp |
| 7h | 3.01% | +0.22pp | +0.97pp |
| 8h | 8.16% | −1.03pp | −3.68pp |
| **Agg** | — | **−0.31pp** | **−0.15pp** |

**Verdict: WASH — KEEP `2×TRADING_FEE = 0.22%`.** Both lower thresholds are mildly *negative* in aggregate with mixed per-horizon signs (5h/7h like a lower bar, 6h/8h dislike it) = noise within the ±3pp refine band. Lowering the label does what's expected mechanically (more trades: 5h 30→32-34, 6h 26→27-37) but the extra sub-0.22% moves net to ~zero — not reliably profitable even at a 0.05% bar. **Decisive: don't touch the label threshold.** Dovetails with the same-day `conf_sweep` (`tools/conf_sweep.py`): the directional edge lives in the **inference confidence gate** (65/65 = the empirical optimum on the live models), NOT in relabeling small moves at train time. Extends the "every directional lever fails the gated sim" pattern from features/model/decision to the **label** axis. Catalogued **C87**.

### 🔇 CLOSED 2026-06-12 — Feature-scout batch C61/C62/C67/C75: NO SIGNAL (screen-noise) + v3_lit harness fixes

Tested the 4 highest-prior **queued** features (from the 2026-06-11 feature-scout) via `tools/test_v3_lit_batch_C60_to_C82.py` (Mode D ETH 5,6,7,8h `--replay 1440 --no-persist`, top-APF delta vs baseline). **All effectively NO SIGNAL — none promoted.**
- **C61 vol-of-vol → MOOT/CLOSED.** `vol_of_vol_8h` + `vol_of_vol_24h` are **already native engine features** ([crypto_trading_system_ed.py:1018](crypto_trading_system_ed.py#L1018)); the patcher crashed on the duplicate. The C60-C82 queue mislabeled it untested — it's been live all along.
- **C62 DXY-accel / C67 Connors-RSI / C75 stablecoin-supply-ratio → SCREEN-NOISE, not promoted.** Valid recompute vs a freshly-rebuilt baseline: top-1 APF Δ C62 +4.16 / C67 +8.12 / C75 +6.51 — BUT every "PASS" rode on ONE horizon's overfit APF outlier (C62 8h=**28.6**, C67 6h=**31.3** vs realistic 1-3) with other horizons NEGATIVE; C75 collapsed to **+1.10** on the robust top-5; per-horizon deltas swung **+20 to −5**. No clean, consistent signal — consistent with the feature-add family being exhausted (0 real PASS).
- **HARNESS BUGS found + fixed (the first run's "4/4 PASS" was a false positive):** (1) `refresh_baselines()` ran all horizons as ONE cmd → engine only ran the LAST (8h) → 5/6/7h baselines stale → patched "beat" a stale baseline (fixed: per-horizon loop); (2) patchers crashed on native-duplicate feature names (fixed: template dedups `feature_cols`). **LESSON: the top-1-APF screen is structurally too noisy for feature screening** (MIN_TRADES=8 lets a few-trade overfit config inflate APF to ~30 and dominate top-1) → filter configs to ≥~20 trades and/or score top-5-*median*, and ALWAYS confirm any non-DEAD via the full regime+conf-gate+shield gated sim. Noted in the harness header.
- This run also surfaced (separately) the **live-config drift incident** (config silently moved to bull/bear 8h gates-off on June-8) — see the FAYE LIVE row in [TODO.md](TODO.md).

### 🔬 CLOSED 2026-06-10 — Strategy-class diversification: 3 orthogonal strategies tested, 1 promoted to paper

After exhausting every lever on the **directional** ETH model (features, model tuning, decision layer, target/label — each fails the gated sim), pivoted to **orthogonal strategy classes** that don't depend on directional prediction. Thorough crypto-quant landscape research → 3 classes backtested:
- **Basis / funding carry = WINNER** — delta-neutral (long spot + short perp, harvest 8h funding). ETH Sharpe 15-24 frictionless (~4.8 net per lit), APY +6.17%, maxDD 1.97%, funding +81% of periods, **uncorrelated** to the directional sleeve. → promoted to **paper-trading** on Desktop (`tools/funding_carry_eth.py`, FREE Binance public data, live since 2026-06-10 21:39 UTC).
- **Stat-arb ETH-BTC = DEAD** — −100%, Sharpe −3 to −6; the pair trends (dominance regimes) more than it mean-reverts (spread ADF p≈0.05).
- **Cross-sectional momentum = WEAK / SHELVED** — best Sharpe 0.73 @14d, but 6-coin universe too thin (lit uses ~50); revive on a ~30-50 coin universe.

Also closed: **trend-scanning label (LdP) = DEAD** (−8.64pp gated, same family as C20). Catalogued below as **C83** (carry, active-paper) / **C84** (stat-arb, dead) / **C85** (xsec momentum, shelved) / **C86** (trend-scanning label, dead). Full narrative: [TODO.md](TODO.md) "Logged 2026-06-10". The carry-sleeve note for CLAUDE.md stays deferred until the paper run validates net-of-frictions.

### 🔥 CLOSED 2026-06-05 — Shadow-monitor closed-bar bug, GPU device decision, data-revision confirmation, 1-4h DV leak analysis, BTC display fix

A multi-thread session (2026-06-04 → 06-05). Live trading was correct throughout; the findings were about *monitors*, *device*, and *short-horizon backtests*.

**1. Bear-config doc drift (corrected).** Live ran `bull 6h@65 / bear 5h@80` (bear manually swapped from `8h@65` on 2026-06-02 23:09; backtest drivers `f580d36`/`af1c0d5`) but TODO.md/CLAUDE.md still documented `bear 8h@65`. User confirmed **6h/5h is the intended live config**; LIVE STATE corrected in both files. Caveat recorded: the bear-swap backtests (`bt_new_vs_old_bear`, `bt_am1_vs_B_regime`) ran on the **un-lagged ED engine** (pre-`a7cb7c9`) → 5h>8h ranking is leak-suspect and should be re-confirmed on the lagged engine.

**2. GPU LightGBM is cross-machine deterministic → sanity device decision = GPU.** `tools/check_proba_repro.py` on the Laptop (LAPEX, RTX 3070 Ti) reproduced the Desktop (RTX 4080) reference **bit-for-bit**: `RF=0.51463784 LGBM=0.05573897` (run1==run2==desktop). Top branch of the TODO0604 decision tree. Consequence: **drop `--cpu-lgbm` from the sanity** — GPU matches the live trader's device *and* is portable across machines. RF-equal also proves `data/` is identically Drive-synced. Decision recorded in [TODO0604.md](TODO0604.md).

**3. Training-window data revision CONFIRMED (was a hypothesis, now measured).** For the 3 stubborn sanity mismatches (06-04 05:00/08:00/11:00): the 15 inference-row features are **byte-identical** live-vs-rebuilt-now (0/15 drift), the model is provably deterministic, yet the probas differ → by elimination the **training rows changed** (deriv/on-chain backfill of the recent window). Control bar (14:00) confirmed: its proba *also* shifted but stayed the same side of the threshold → matched. **Key nuance:** settling makes recompute *stable*, not equal-to-live — a revision that crossed a threshold is a **permanent** reproduction mismatch, not a transient one. So 100% sanity on live-data replay is **not** achievable for revision-flipped hours; only point-in-time data snapshots would close it. The 3 also **bounce run-to-run** under `--cpu-lgbm` (CPU LGBM nondeterminism), which GPU removes.

**4. SHADOW-monitor closed-bar bug (real regression, fixed).** Shadow match (`config/shadow_signal_diff.csv`) was a clean **100% every day May 26 → Jun 2, then crashed: Jun 3 = 63%, Jun 4 = 48%**. NOT a model change (config/CSV untouched since Jun 2, no auto-promote). Root cause: commit `28645bd` **fix #2 (infer on last CLOSED bar, not forming candle)** updated the live path (`crypto_live_trader_ed.py:706-727`) but **not** the shadow recomputation (`crypto_live_shadow.py`), whose `i = len(df)-1` kept inferring on the **forming** bar. Proven decisively, zero-compute: **23/23 mismatch rows had the shadow inferring exactly +1h ahead of live** (forming vs closed). **Fixed** `crypto_live_shadow.py:220` to mirror the live closed-bar step-back. The bug is a *forming-bar* artifact → cannot be reproduced offline (all historical bars are closed) → the live number only recovers on **trader restart**. Live trading was correct (closed-bar) the whole time; only the monitor was stale.

**5. Engine-vs-trader parity: 90% (CPU) → 96.7% (GPU).** Re-ran `validate_core_against_signal_log.py --samples 30 --recent-only` on GPU: **29/30 = 96.7%, avg conf delta −0.93** (was −5.71 on CPU that morning). The ~5pt systematic bias vanished — direct payoff of the GPU device decision (#2). The lone DIFF (06-05 13:00, live SELL / core HOLD) is a recent unsettled-hour boundary case (the #3 effect); 0 BUY↔SELL flips.

**6. 1-4h DV leak analysis (and why 1h ≠ "more embargo leak").** User ran Mode P (PySR) + `DV ETH 1,2,3,4h --replay 1440` on FAYE. Backtest WR gradient: **1h 92-98% / 2h 88% / 3h 85% / 4h 76% / 5-8h (real) 74-83%** — monotonic inflation as horizon shrinks. User correctly challenged "1h leaks": **embargo=horizon protects all horizons equally** against label-overlap leakage. The real drivers of short-horizon inflation are (a) signal-to-noise (1h return ≈ noise + 0.22% fee → 98% WR is overfit/selection, not edge), (b) un-tradeable microstructure the flat maker-fee under-charges, (c) horizon-sensitive *feature-timing* leak (a 1-bar misalignment = 100% of a 1h label, 12% of an 8h). 5-8h overstate too (84% backtest vs ~75% live) — just less. **Verdict: 4h worth pursuing (76% WR in-band; engine card already calls 4h asset-conditional); 2-3h borderline; 1h inflated. All gated on an embargo-sensitivity sweep** (`FAYE_EMBARGO_OVERRIDE=4/8`) + forward shadow week before any trust.

**7. Trader BTC-display bug (fixed) + untracked-BTC flag.** `/status` "Balance summary" listed every non-USD coin held (`crypto_revolut_ed_v2.py:2547`) → showed phantom `BTC: 0.078…`. **Fixed** — non-enabled coins removed from the display (line 2544). But `get_balances()` only echoes the exchange, and the BTC position file has a **malformed April SELL (price=0, pnl=0, timestamped before its own buy)** that set `state: cash` without zeroing `base_amount=0.0785` → **~0.078 BTC (~$5k) may be genuinely untracked on Revolut X**. Open: run `python crypto_revolut_ed_v2.py --balance` to confirm real-vs-phantom before zeroing the position file.

**8. Tooling built:** `tools/compare_prod_vs_4mo.py` — backtests Production (2mo: bull 6h@65/bear 5h@80) vs 4mo-HRST (`ed_v1_20260604_075223`: bull 5h@70/bear 5h@65) over 720h + 168h through the FAYE near-live engine. Purpose: decide whether 2-month or 4-month HRST windows produce the better model. Running on Desktop.

**Code changes:** `crypto_live_shadow.py` (closed-bar mirror of fix #2), `crypto_revolut_ed_v2.py` (hide non-enabled coins in `/status`), `tools/compare_prod_vs_4mo.py` (new). Doc: LIVE STATE bear=5h@80 in TODO.md+CLAUDE.md, device decision in TODO0604.md.

---

### 🔥 CLOSED 2026-05-30 (afternoon) — FAYE thorough line-by-line audit: 12 bugs found and fixed

**Trigger**: First FAYE launch hung at "[FAYE] dispatching 60 configs ..." with zero `[FAYE done N/60]` after 30 min. User pushed for a thorough audit ("check VERY carefully the D mode setup", "I SAID A THOROUGH AUDIT TO BE SURE IT MATCHES"). I had previously delegated to Explore agents that reported "NO NEW BUGS FOUND" — but the agents compared FAYE to intermediate layers (g_narrow_d, engine.py) instead of to post-patch runtime state (parallel_nearlive's final monkey-patches). Doing the line-by-line walk myself caught what the agents missed.

**12 bugs found in FAYE's consolidation of the v3 chain**:

| # | Bug | Severity | Commit | Symptom |
|---|---|---|---|---|
| 1 | `GRID_WINDOWS=[72,100,150]` (engine base) instead of `[72,100,150,200,250]` (g_narrow_d narrow_nearlive) | CRITICAL | `23b54b3` | Optuna refine never reaches w=281/293 basin where live winners sit; FAYE would produce uncompetitive winners every time |
| 2 | Warning suppression missing Layers 1 (`-W ignore` re-exec) + 4 (`warnings.warn=no-op` monkey-patch). Had only Layers 2 (PYTHONWARNINGS) + 3 (filterwarnings) | MEDIUM | `2aea8be` | Thousands of `UserWarning: sklearn.utils.parallel.delayed` warnings flooded stdout from inside ProcessPool workers, log noise |
| 3 | `NEAR_LIVE_MODE` env var not set at module init — FAYE has defaults baked in but any importer checking the env var sees None | DEFENSIVE | `f1b49f4` | None observed; bug is latent |
| 4 | `_get_deku_diagnostic_models_seeded` hardcoded `device='gpu'` (inherited from g_narrow_d pre-patch); parallel_nearlive's `_device_aware_factories_seeded` reads `G_PARALLEL_LGBM_DEVICE` (default 'cpu') | **CRITICAL** | `1cee360` | **The launch hang**: 8 ProcessPool workers × K=5 inner ThreadPool = 40 concurrent LGBM-GPU calls on one RTX 4080 queue → workers blocked indefinitely, never produced output |
| 5 | Refine worker did not broadcast `lgbm_device` arg to K=5 factories via `os.environ['G_PARALLEL_LGBM_DEVICE']` | MEDIUM | `3027255` | The dispatcher's hybrid GPU/CPU device routing was inert — workers all ran on CPU regardless of dispatcher choice |
| 6 | Refine dispatcher: 2 workers (GPU + CPU) with dynamic-3rd vs parallel_nearlive's 3 workers all-CPU when K>1 | CRITICAL | `3027255` | GPU worker + K=5 inner threads would crash CUDA single-context. parallel_nearlive comment: "5 threads × LGBM-GPU per worker crashes the GPU's single CUDA context." Refine would have crashed ~4h into the run |
| 7 | `_deku_eval_with_pruning` (K=5 wrap) ran K=5 seeds in a SERIAL for-loop. parallel_nearlive's `_parallel_deku_eval_median_k` uses ThreadPoolExecutor(max_workers=K=5) | CRITICAL | `a65d618` | 5x slowdown in Mode V (refine + holdout). Per-trial time: ~3.3 min serial K=5 vs ~3.3 min parallel = wait no, K=5 serial means 5× the per-trial work serially. ETA: 60h serial vs 17h parallel |
| 8 | `models_faye/` directory not auto-created — per-eval CSV flush failed with "Cannot save file into a non-existent directory" | MEDIUM | `ae056b7` | First FAYE launch's per-eval CSV (`mode_d_full_*.csv`) couldn't be written. Grid logging silently broken. v3 didn't hit this because `models_g_desktop_nearlive/` was a long-standing snapshot dir |
| 9 | `_get_deku_diagnostic_models_seeded` did not set LGBM `num_threads` — defaulted to physical core count (24). 8 workers × K=5 threads × 24 LGBM threads = 960 concurrent OS threads on 24 cores = catastrophic oversubscription | **CRITICAL** | `ae056b7` | 12 min/eval observed (vs v3's 5-7 min/eval). After fix (`num_threads=1` via `G_PARALLEL_LGBM_THREADS` env, matching parallel_nearlive line 309): 5.7 min/eval confirmed on user's relaunch |
| 10 | `N_FEATURES_RANGE` = (4,40)/(4,80) (engine pre-patch) instead of (4,100)/(4,100) (g_narrow_d, propagated by parallel_nearlive line 164-165). Comment in parallel_nearlive explicitly states: "the cap was creating the B-7h tied-APF trap" | MEDIUM | `d259efe` | Optuna refine's n_features upper bound capped too low, preventing exploration of 80-100 feature subsets |
| 11 | Regime config not seeded from live — parallel_nearlive lines 184-209 seeds isolated CONFIG_DIR/regime_config_ed.json from `config/regime_config_ed.json`. Mode R/S/T read+rewrite this file; without a seed, first HRST crashes with FileNotFoundError | MEDIUM | `d259efe` | Mode H only doesn't hit this. Future HRST would have crashed at Mode R |
| 12 | **HOLD branch in `_deku_eval_with_pruning_inner` diverges from step6_nearlive**: FAYE incremented `total` on HOLD (inflated accuracy denominator) and didn't tick drawdown (under-counted max_dd). step6_nearlive lines 4193-4201 does NEITHER | **CORRECTNESS** | `5382740` | FAYE accuracy artificially LOWER (1/N per HOLD) and max_dd artificially LOWER. Numbers diverge from v3 for same model on same data |

**Pattern of audit failure** (worth recording): I dispatched audits asking "does FAYE match v3-chain original?" Multiple Explore agents reported "NO NEW BUGS FOUND" — but they compared FAYE to **intermediate layers** (`g_narrow_d._g_factories_seeded` has `device='gpu'` → matches FAYE → marked OK) instead of to **runtime-post-patches** state (where parallel_nearlive overrides with `_device_aware_factories_seeded` that uses CPU + num_threads=1).

The 4 monkey-patch layers in v3 (`v3 → parallel_nearlive → step6_nearlive → g_narrow_d → engine`) meant the "right" reference for FAYE was the RUNTIME STATE after parallel_nearlive applied its patches, not any single source file. Audit prompts that didn't explicitly enforce this got it wrong.

**Items verified byte-identical** between FAYE and the appropriate post-patch reference (after line-by-line `diff` runs):
- `generate_signals` ✓ matches g_narrow_d
- `_backtest_one_config_worker` ✓ matches g_narrow_d
- `_run_parallel_backtests` ✓ matches g_narrow_d
- `_predict_signal_calls_for_horizons` ✓ matches g_narrow_d
- `_signal_gen_worker` ✓ matches g_narrow_d
- `_build_signals_cache_parallel` ✓ matches g_narrow_d
- `_key_for_call` ✓ matches g_narrow_d
- `_finish_mode_v` ✓ matches g_narrow_d
- `_compute_optuna_score` ✓ matches engine
- `_diversity_key` ✓ matches engine (intentionally NOT g_narrow_d's; H_STRICT_FAMILY's (combo, w) is what v3 runs because v3 → parallel_nearlive → ENGINE.main(), not g_narrow_d)
- `run_mode_v` ✓ matches g_narrow_d's `run_mode_v_parallel` (only func name differs)
- `run_mode_s` ✓ matches g_narrow_d's `run_mode_s_parallel` (func name + fallback name renamed)
- `run_mode_t` ✓ matches g_narrow_d's `run_mode_t_parallel` (func name + fallback name renamed)
- `_deku_eval_with_pruning_inner` ✓ matches step6_nearlive's `_deku_eval_with_pruning` (after Bug #12 fix)

**Items intentionally different** (FAYE design choices, not bugs):
- `MODELS_DIR`, `CONFIG_DIR`, `PRODUCTION_CSV`, `REGIME_CONFIG_PATH` — FAYE uses `FAYE_*` prefix (full isolation from Ed live paths)
- `DIAG_STEP=1`, `HOLDOUT_STEP=1` — NEAR_LIVE defaults baked in vs Ed's step=36 (functionally equivalent because step6_nearlive's NEAR_LIVE_MODE override sets step_to_use=1 in v3 runtime)
- `NEAR_LIVE_*` constants baked into `_deku_eval_with_pruning_inner` default args vs env-var override in step6_nearlive — same effect when env=1 (the v3 default)
- 8-worker Mode D dispatcher inlined natively in `run_mode_d_optuna` vs v3's `inspect.currentframe()` frame-capture monkey-patch — same behavior, cleaner code
- `pool.shutdown(wait=True)` replaced with `_hard_shutdown_pool` (SIGTERM + `taskkill /F /T`) to prevent the v3 orphan-worker leak observed Mode D 5h refine→6h transition

**Net effect**: After all 12 fixes, FAYE produces results that are functionally identical to v3 (modulo intentional FAYE-isolation paths). The HOLD-branch correctness bug (#12) was the last remaining divergence in the actual numbers.

**Verification**: User's FAYE H 6h,7h,8h relaunch after all fixes — first `[FAYE done 1/60]` appeared 5.7 min after dispatch (matches v3's measured 5-7 min/eval on Desktop). Mode V refine + 7h + 8h horizons still to come; ETA ~05:30 May 31.

**Files**: 12 fix commits 2026-05-30 09:35-12:50 — `23b54b3`, `2aea8be`, `f1b49f4`, `1cee360`, `3027255`, `a65d618`, `ae056b7`, `d259efe`, `5382740`.

---

### ✅ CLOSED 2026-05-30 — FAYE single-file consolidation shipped + post-FAYE archive cleanup

**Outcome**: Built `crypto_trading_system_faye.py` — a 9100-line single-file consolidation of the Ed v3 architecture (`crypto_trading_system_ed_g_narrow_d_parallel_nearlive_v3.py` + its 3 monkey-patch parents). **ZERO monkey-patches, every previously-patched feature first-class native code.** Engine-vs-trader parity test on 30 most recent ETH hours showed **0 real BUY↔SELL flips** out of 30 — the original "engine says X, trader says Y on the same hour" bug is gone.

**Architecture: the 4-layer monkey-patch chain → one file**:

The Ed v3 production stack was:
```
v3 → ENGINE._get_deku_diagnostic_models hook + ENGINE._deku_eval_with_pruning routing dispatcher
   → parallel_nearlive → _parallel_deku_eval_median_k K=5 wrap
   → step6_nearlive → _H_ORIG_DEKU_EVAL near-live semantics
   → g_narrow_d → _G_ORIG_DEKU_EVAL K=5 seeded factories
   → ed (base)
```

FAYE collapses all 4 layers. Each previously-patched feature is now defined at its natural location with no module-load rebinding.

**7-phase rollout, each phase its own commit + smoke check**:

| Phase | Commit | What | What it replaces |
|---|---|---|---|
| P1 | `8c122ef` | FAYE identity: `FAYE_MODELS_DIR`, `FAYE_CONFIG_DIR`, `crypto_faye_production.csv`, `regime_config_faye.json` | Ed dir/file names (isolated from prod) |
| P2 | `0883ea4` | `_mean_last_10_fill` helper + NEAR_LIVE defaults in `_deku_eval_with_pruning` (`step=1`, `signal_mode='ternary'`, `na_policy='mean_last_10'`, `embargo=horizon`, `return_probas=True`) | `crypto_signal_core_nearlive.py` helper + env-var gating |
| P3 | `ff63d30` | Native K=5 multi-seed median ensemble: bare loop renamed `_deku_eval_with_pruning_inner`, new K=5 wrap is canonical `_deku_eval_with_pruning`; seeds=[42..46]; pruning only seed 1 (median validity) | g_narrow_d's `reliability_multi_seed.py` patch + module-load rebind |
| P4 | `c878832` | 3-worker hybrid GPU+CPU refine inlined: hybrid is canonical `_refine_top_configs`, serial is `_refine_top_configs_serial` fallback for ≤1 config | parallel_nearlive's `_refine_top_configs_hybrid` + `_ENG_REFINE_TOP_CONFIGS_ORIG` indirection |
| P5a | `d7f7744` | `run_mode_v/s/t_parallel` → `run_mode_v/s/t` canonical; serial as `*_serial` fallback; `_ENG_RUN_MODE_*_ORIG` chain deleted | parallel_nearlive's module-load rebind |
| P5b | `bb0c6fe` | Native 8-worker Mode D grid dispatcher: builds 60 configs upfront, ProcessPool dispatch with K=5 ThreadPool inside each worker, per-eval CSV `mode_d_full_*.csv` with K=5 seed-by-seed breakdown, `freeze_support()` added | v3's `_capture_state_then_get_models` + `_v3_routed_deku_eval` + `_ParallelGridDispatcher` `inspect.currentframe()` hack |
| P6 | `694d85f` | `tools/smoke_test_faye.py` — 38-check verification (import, state, canonical entry points, serial fallbacks, worker pickling, no leftover monkey-patch names, K=5 wrap is native) | (new — no equivalent before) |
| P7 | `4ab34d5` | Module docstring "ARCHITECTURE: monkey-patch genealogy → native consolidation" block, RELIABILITY_K env note, smoke test pointer | (new — no equivalent before) |

**Engine-vs-trader parity test result** (`tools/validate_core_against_signal_log.py --samples 30 --recent-only --cpu-lgbm`):
- 30/30 hours evaluated, 0 errors, 0 crashes
- **22/30 direct MATCH (73.3%)**, 8/30 DIFF (26.7%), avg confidence delta −1.98 (near-zero bias)
- **Pattern analysis: 0/8 real BUY↔SELL flips** — all 8 DIFFs are HOLD-threshold boundary cases where one side's confidence happened to be above the trader's `min_confidence` (95% for bear) and the other below
  - 5/8: live=HOLD (trader below 95% threshold) vs core=BUY/SELL (raw model direction)
  - 3/8: core=HOLD (engine probability < 50%) vs live=BUY/SELL (trader hit threshold)
- **Verdict: the engine and trader codepaths agree on direction every single time both produce one.** The original "bug between live trader and crypto trading" — engine computing one signal, trader computing a different one for the same hour — is gone.

**Post-FAYE archive cleanup** (`ARCHIVED/2026-05-30_post_faye_cleanup/`, commit `1fce7f8`):

Now that FAYE supersedes the variant scripts and their tools clusters, archived in three tiers:

- **Tier A — engine root variants** (8 files): `crypto_trading_system_ed_cdar.py`, `_cvar.py`, `_cpcv.py`, `_robust.py`, `_h_strict_family.py`, `_noprod.py`, `_pre_macro_cache_fix_20260527_112231.py`, `_launch_h_strict_family.bat`. Each one had only doc/history references; the bits that survived (K=5, dedup by `(combo, w)`) are now native in FAYE.

- **Tier B — variant-driving tools** (14 files): the test runners and HRST drivers that only existed to fire the Tier A variants. CDaR idea-test cluster (`test_c04_to_c08_runner.py`, `test_desktop_5ideas_runner.py`, `test_c05_c06_only.py`, `test_c32_to_c40_batch.py`), CVaR+CPCV cluster (`run_cvar_hrst_resumable.py`, `run_cpcv_hrst_resumable.py`, `run_c67_hrst_resumable.py`, `rerun_v2_full.py`), robust+reliability cluster (`run_reliability_test.py`, `run_reliability_hrst.py`), step6+h_strict_family cluster (`test_step6_regression.py`, `compare_h_b_prod_30d.py`, `smoke_test_path_resolution_matrix.py`), and the orphan `run_locked_detector_hrst.py`.

- **Tier C — old model/config snapshot dirs** (14 dirs, ~440KB): `models_g_desktop/`, `_0524/`, `_0524_h75/`, `_0524_live_baseline/`, `_0525/`, `models_h75/`, `models_h75_wide_laptop/` + matching `config_*` siblings. Pre-`_nearlive` artifacts from the H75 / G_desktop / 0524 / 0525 runs.

- Plus `CLAUDE_NEW.md` (stale May-19 draft of CLAUDE.md, never promoted) and `TODO_TEST.md` (tiny stale scratchpad) → `docs/`.

**Engine root impact**: `.py` files at root 38 → 29; root dirs 35 → 21. `ARCHIVED/2026-05-30_post_faye_cleanup/INDEX.md` documents what was moved and where.

**NOT archived** (still load-bearing): `crypto_trading_system_ed.py`, `_g_narrow_d.py`, `_g_narrow_d_parallel_nearlive.py`, `_g_narrow_d_parallel_nearlive_v3.py`, `_step6.py`, `_step6_nearlive.py`, `crypto_signal_core.py`, `crypto_signal_core_nearlive.py` — actively imported by the v3 chain that is still running on Desktop. Will be safe to archive AFTER FAYE replaces v3 in production.

**Files**: FAYE consolidation commits `8c122ef` → `4ab34d5` (8 commits), archive cleanup `1fce7f8`. Parity test output `output/core_validation_20260530_015454.csv`. Smoke test `tools/smoke_test_faye.py` (38 ✓).

**Next**: when current v3 HRST on Desktop finishes (~18:00 May 30) and trader is flat, swap v3's winners into Ed paths via Copy-Item (per the docstring recipe in FAYE), restart trader. **No FAYE promotion to production until v3 HRST done + trader flat + final smoke check.**

---

### 🗄 ARCHIVED 2026-05-28/29 — v3 fork shipped + 4 superseded engine forks moved to `ARCHIVED/2026-05-28_v3_cleanup/`

**Triggered by**: night-of 2026-05-28 dev session debugging the running NEAR_LIVE_MODE HRST (parallel_nearlive v2, started 2026-05-27 23:46). Multiple iterations of warning suppression + grid trim + phase-aware threading (then reverted as a measured regression) — resulted in the **v3 fork** committed `f688e0e` and the **archive cleanup** committed `b2dbe7a`.

**v3 fork — `crypto_trading_system_ed_g_narrow_d_parallel_nearlive_v3.py`** (509 lines): builds on parallel_nearlive v2 with TRUE Mode D outer-loop parallelization via **dispatcher pattern** (no 545-line engine function replication). Monkey-patches `ENGINE._get_deku_diagnostic_models` to capture grid-setup locals via `inspect.currentframe().f_back`, then replaces `ENGINE._deku_eval_with_pruning` with a routing dispatcher: on first Mode D call it enumerates all 60 configs and submits them to `ProcessPoolExecutor(max_workers=8)`; each subsequent call blocks on its specific Future. Engine's serial for-loop still runs but every call hits a parallel-pre-dispatched cache. **8 outer × K=5 inner = 40 concurrent LGBM fits** (was 5). Worker function re-imports `parallel_nearlive` to inherit its patches; K=5 ThreadPool inside each worker UNCHANGED for quality preservation. Prints EVERY eval (not just NEW BEST). Writes full per-eval CSV `mode_d_full_{asset}_{horizon}_{ts}.csv` with K=5 seed-by-seed metrics. Expected ~5-6× Mode D speedup, ~3-4× full HRST. Other calls (Mode V step 1/2/3, holdout folds) fall back to parallel_nearlive's K=5 wrap unchanged.

**Archived files (4 .py forks → `ARCHIVED/2026-05-28_v3_cleanup/`)**:
- `crypto_trading_system_ed_pre_H75_20260518.py` (May 18, only doc refs — TODO.md rollback ladder)
- `crypto_trading_system_ed_pre_cli_fix_20260518.py` (May 18, 0 refs)
- `crypto_trading_system_ed_h75_wide.py` (May 26, 0 refs — superseded by parallel_nearlive)
- `crypto_trading_system_ed_g_narrow_d_parallel.py` (May 25, only doc refs in README/CLAUDE/TODO — superseded by parallel_nearlive then v3)

Files stay in git history; archive just declutters the working directory. Verified before move: no `.py` imports, no `.bat` launchers, no active tools reference any of the four. Isolation dirs (`models_h75/`, `config_h75/`, `models_h75_wide_laptop/`, `config_h75_wide_laptop/`, `models_g_desktop_0525/`, `config_g_desktop_0525/`) stay in place per user choice to preserve post-run CSVs for potential analysis.

**Companion deliverable — `tools/merge_pysr_old_new.py`** (committed `c753f46`): tool to keep OLD PySR (April 21) AND NEW PySR (May 28 Laptop run) as candidate features in one merged JSON per horizon. Slots `pysr_1..5` = OLD (preserves existing 5h/6h production models' references — zero inference drift); slots `pysr_6..10` = NEW (additional candidates). LGBM gain-based ranking + Optuna's `n_features` parameter then decides which to use during new training. Correlated OLD/NEW pairs split the gain budget; only one usually survives the top-N cut. Backs up OLD JSONs as `*_pre_20260528_old_only.json` before overwrite for one-rename rollback.

**Files**: v3 fork commit `f688e0e`, archive commit `b2dbe7a`, merge tool commit `c753f46`.

---

### ✅ CLOSED 2026-05-27 — TODO 0525 (G_narrow_d HRST with extended grid + V2 top-10 + Optuna win_hi=350)

**Outcome**: Desktop run 2026-05-25 22:38 → 2026-05-26 08:44 (~10h). Mode T REF +83.85% on May 22 data lost to current LIVE +91.01% by 7pp. Hypothesis "extend the window grid to `[72,100,150,200,250,300,350]` + bump V2 top-N to 10 + raise Optuna `win_hi` to 350 will unlock the high-window basin that LIVE's w=281/293 sits in" was **REJECTED** — the 0-trade holdout filter still ate the w=250/300 candidates regardless of grid expansion. The grid changes worked as designed but the holdout filter was the binding constraint, not the grid coverage.

**Spinoff**: spawned TODO 0526 architecture analysis (which itself was superseded by TODO 0527's `_macro_cache` root cause). Mode V top-N=10 + zero-holdout skip code changes from this run shipped in production engine commit `27a695f` (kept).

**Files**: log `logs/ed_v1_20260525_223822.log`, output trade tables retained in models/_grid_extended/.

---

### ✅ CLOSED 2026-05-25 — TODO 0524 (Top-5 HRST clean rerun on fixed parallel fork)

**Outcome**: Desktop 2026-05-24 22:53 → 2026-05-25 06:39 (Mode H+R+S), Mode T reruns 12:35 + 13:17. Mode T REF +80.56% vs current LIVE's +91.01% on same May 22 data — **no promotion**. The narrow grid `[72, 100, 150]` couldn't seed Optuna refine to reach LIVE's w=281/293 basin.

**Spinoff**: spawned TODO 0525 (extended grid) which also failed for a different reason (holdout filter). Both runs together established that the LIVE config sits in a region the current research grid can't naturally produce.

**Spinoff #2 — actually shipped**: validated the parallel-refine speedup from TODO 0522 produces clean output once the grid bug was fixed. **~8× refine speedup retained as production capability**. This is the only durable win from the 0522 → 0524 → 0525 arc.

**Files**: snapshot `data/_reliability_hrst_snapshot_desktop_20260524_2253` retained, log `logs/ed_v1_20260524_225309.log`.

---

### ✅ CLOSED 2026-05-24 — TODO 0522 (Parallel refine speedup G_narrow_d_parallel fork + long-horizon G test)

**Outcome (2-stage)**:
- **Stage 1** (Laptop 2026-05-22 00:26): identity-preserving parallel refine fork against the canonical g_narrow_d on a fixed snapshot — **PASSED**. Outputs bit-identical to serial refine within numerical tolerance. ~8× speedup measured. Fork kept as `crypto_trading_system_ed_g_narrow_d_parallel.py`.
- **Stage 2** (Laptop 2026-05-22 01:39 → 18:09): full HRST on 9,10,11,12h with the parallel fork — **VERDICT INVALIDATED** by a grid configuration bug discovered later (the fork's grid passing into refine differed silently from the canonical fork). Numbers from this run shouldn't be trusted; reruns under TODO 0524 with bug-fixed parallel fork supersede.

**Net**: parallel refine speedup itself is real and shipped. The long-horizon question (9-12h) became moot when TODO 0525's analysis showed no horizon outside 5h/8h was going to beat LIVE on the current data with the current model setup. Idea family for 9-12h horizons closed.

**Files**: Stage 1 log `logs/g_parallel_stage1_20260522_0026.log`, Stage 2 log `logs/g_parallel_stage2_20260522_0139.log` (numbers void), bug-fix commit `f88e4dc`.

---

### ✅ CLOSED 2026-05-27 — TODO 0527 (`_macro_cache` mtime bug — root cause of LIVE-vs-BACKTEST gap)

**Outcome**: Diagnosed + fixed + audited in one session. The dominant cause of the 50.9% live WR vs ~85% backtest WR gap was a process-lifetime in-memory cache in the live trader (`crypto_trading_system_ed.py:1077` `_macro_cache`) that was set once at trader startup and never refreshed. All macro/cross-asset/sentiment/onchain features were frozen at their startup-day values, so time-shifted features like `m_vix_chg1d`, `m_sp500_chg1d`, `fg_chg5d`, `oc_mvrv_chg1d`, `xa_dax_relstr5d` collapsed to ~0 once the cache aged beyond a day (the formula `(today − yesterday)/yesterday` becomes `(startup − startup)/startup = 0` when both lookups ffill to the same row).

**Why this had the largest impact**: per CLAUDE.md Engine Reference Card these are some of the highest-importance features (40–67% selection rate). With them dead, the model was effectively running on price + derivatives features only — a fraction of the feature space it was trained on.

**The fix** (6 lines added to `_load_macro_csv`):
```python
mtime = os.path.getmtime(path)
cached = _macro_cache.get(filename)
if cached is not None and cached[0] == mtime:
    return cached[1]
# else re-read + cache as (mtime, df)
_macro_cache[filename] = (mtime, df)
```

**Validation performed in same session**:
- Same-process test (`tools/validate_core_same_process.py`) → 100% match (core math = live math)
- Shadow mode in-process after fix → 100% match (10/10 comparisons, max conf delta 0.04pp = pure rounding)
- PIT validator with oldest-wins archeology merging → 66.7% (ceiling, limited by retrospective inability to reproduce trader's stale cache state at historical signal_dts)

**Companion fixes shipped same session** (audit findings, all production-deployed):

1. **`download_macro_data.py` — `download_fear_greed`**: was directly overwriting `fear_greed.csv` each call (no merge); now uses `_merge_preserve_history`. Without this, any upstream API revision to a past Fear & Greed reading would silently propagate into historical data.

2. **`crypto_revolut_ed_v2.py` — signal_log schema rename**: `sig_4h`/`conf_4h`/`sig_8h`/`conf_8h` → `h_1/sig_1/conf_1/h_2/sig_2/conf_2` (regime-anchored: slot 1 = bull-regime model output, slot 2 = bear). The old hardcoded `sig_4h` column name mismatched the production bull horizon (5h under H75/G_narrow) and was permanently empty for 79/136 recent rows. Migrated existing 2125-row CSV in place; backup at `config/signal_log_backup_pre_rename_20260527_092500.csv`.

3. **`crypto_revolut_ed_v2.py` — 3 latent HORIZON_SHORT/LONG bugs**: same anti-pattern as the signal_log issue but in (a) Telegram message exposure `sig_short/sig_long`, (b) asset preflight horizon check, (c) gamma lookup fallback. All three were silently returning None or wrong horizons because they assumed bull=4h but actual production has bull=5h. Fixed to use sorted-order from actually-computed horizons or the asset's actual bull/bear horizons from config.

**Audit performed across multiple angles**:
- sig_1/sig_2: code review + migration cell-by-cell verification + live post-restart writes + 6-case unit test → 0 mismatches
- data drift fix: all 6 active call sites verified using safe helpers; `_dedup_preserve_history` + `_merge_preserve_history` functional tests pass
- cache fix: code review + all 4 callers + 6 edge cases (cache hit, mtime advance, file vanish with/without cache) → all pass; verified `_regime_config_cache` already correctly mtime-aware; no other broken caches in trader path

**Closes TODO 0526** ("LIVE vs BACKTEST divergence — 4 semantic code-path differences"). The 4 divergences were directionally real (embargo / NaN policy / step size / signal mode) but second-order; the dominant cause was the cache bug above. Shadow mode in-process now demonstrates that with identical inputs core == live at 100% match — the remaining historical PIT gap reflects unrecoverable trader cache state, not core math errors.

**Monitoring**:
- Shadow mode continues running on Desktop — any drop below ~99% match flags a new bug
- Live WR over next 3-5 days expected to trend toward backtest WR
- Hard gate: if live WR remains <65% after 4 weeks, investigate execution layer (slippage, partial fills, maker order behavior under stress)

**Files touched this session**:
- `crypto_trading_system_ed.py` (lines 1077-1110, mtime-aware cache)
- `crypto_revolut_ed_v2.py` (lines 1693-1701, 2134, 3237-3300, 4574 — 4 HORIZON_SHORT cleanups)
- `download_macro_data.py` (lines 653-672, fear_greed merge)
- `config/signal_log.csv` (migrated in place)
- `tools/validate_core_point_in_time.py` (NEW, ~750 LOC PIT validator with oldest-wins archeology merge)
- `tools/drive_archeology.py` (NEW earlier, used to download Drive version history for 21 files)

**Pre-fix snapshots preserved**:
- `crypto_trading_system_ed_pre_macro_cache_fix_20260527_112231.py`
- `config/signal_log_backup_pre_rename_20260527_092500.csv`

**Cost**: 1 working session. **Benefit**: closes the single biggest known reliability gap between backtest and live; unlocks shadow mode as continuous live correctness gate.

---

### ✅ CLOSED 2026-05-21 — TODO 0519 (G_narrow_d relaunch on Desktop)

**Outcome**: Run completed 2026-05-21 10:28 with Mode T REF +89.14%. Initially marked "G_narrow_d shelved — no promotion regret signal" because aggregate Mode T REF was ≈ B baseline (+89.41%) within ±0.3pp. **Decision reversed 2026-05-21 21:56**: user promoted G_narrow to live anyway. The per-horizon Mode V winners told a different story than Mode T REF: G's 5h winner `RF+LGBM w=281 γ=0.998 f=12 ret=+72.16% WR=84%` materially beat H75-fresh's 5h `XGB+LGBM w=166 γ=0.997 f=15 ret=+53.76% WR=71%` (+18pp at the same conf level). With ETH bull primarily fired by the 5h model under `sma24>sma100` detector, that 18pp 5h delta translates to live trading even though the multi-horizon Mode T aggregate evens out. As of 2026-05-24 G_narrow models still live (see TODO.md LIVE STATE).

**Run details**: Desktop, 2026-05-20 11:05 → 2026-05-21 10:28 (wall ~23h 22m, log `logs/ed_v1_20260520_110556.log`). Full 4-horizon HRST (5,6,7,8h) + R + S + T. Mode T converged at iteration 2 unchanged, baseline H1=+25.30% / H2=+51.02% / REF=+89.14% vs B's +89.41%. No STRICT rally-cooldown winner. Snapshot used: `data/_reliability_hrst_snapshot_desktop_20260515_154801`.

**Spinoff insight**: per-phase timing exposed refine as 76% of HRST wall (17.7h of 23.4h). K=5 multi-seed runs sequentially per trial, and `_g_factories_seeded` hardcodes `device='gpu'` which breaks the hybrid GPU+CPU dispatcher. → spawned [TODO 0522 (parallel speedup fork)](#-closed-2026-05-24--todo-0522-parallel-refine-speedup--long-horizon-g-test-stage-1-passed-stage-2-invalid).

**Original launch query (for reference / re-run pattern)**:

```powershell
# ============================================================================
# TODO 0519 — G_narrow_d relaunch with safeguards (Desktop, Drive engine)
# ============================================================================
cd G:\engine

# 1. Keep Desktop awake for the full ~16h run
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0

# 2. Defensive backup of row 41 (G's 5h Mode V winner)
copy models_g_desktop\crypto_ed_production_noprod.csv models_g_desktop\crypto_ed_production_noprod_5h_only_pre_0519.csv

# 3. Env vars (snapshot + K=5 + isolated output dirs)
$env:V2_DATA_SNAPSHOT = "data\_reliability_hrst_snapshot_desktop_20260515_154801"
$env:RELIABILITY_K = "5"
$env:G_NARROW_MODELS_DIR = "models_g_desktop"
$env:G_NARROW_CONFIG_DIR = "config_g_desktop"

# 4. Launch Mode H for 6h, 7h, 8h only
$ts = Get-Date -Format "yyyyMMdd_HHmm"
$logfile = "logs\g_relaunch_0519_$ts.log"
python crypto_trading_system_ed_g_narrow_d.py H "ETH," 6h,7h,8h --skip --replay 1440 --no-persist --no-data-update --grid-tag G_NARROW_D *>&1 | Tee-Object -FilePath $logfile

# 5. After Mode H finishes — chain R, S, T sequentially (same PS session, env vars still set)
python crypto_trading_system_ed_g_narrow_d.py R "ETH," 5,6,7,8h --replay 1440 --no-persist --no-data-update --grid-tag G_NARROW_D *>&1 | Tee-Object -FilePath "logs\g_relaunch_0519_R_$(Get-Date -Format yyyyMMdd_HHmm).log"
python crypto_trading_system_ed_g_narrow_d.py S "ETH," 5,6,7,8h --replay 1440 --no-persist --no-data-update --grid-tag G_NARROW_D *>&1 | Tee-Object -FilePath "logs\g_relaunch_0519_S_$(Get-Date -Format yyyyMMdd_HHmm).log"
python crypto_trading_system_ed_g_narrow_d.py T "ETH," --replay 1440 --no-persist --no-data-update --grid-tag G_NARROW_D *>&1 | Tee-Object -FilePath "logs\g_relaunch_0519_T_$(Get-Date -Format yyyyMMdd_HHmm).log"
```

**Safeguards used** (kept for future research-fork launches):
- `Tee-Object *>&1` — captures stdout+stderr so crashes are diagnosable
- `powercfg standby-timeout 0` — kills Modern Standby (the failure mode that killed 1805D the day before)
- `"ETH,"` trailing comma — older fork's `endswith('h')` parser bug; without comma → all 9 assets load
- `--no-data-update` — snapshot env var only redirects READS; without this flag, downloads write to LIVE
- `--no-persist` — production CSV untouched (writes to `_noprod.csv`)
- `--skip` — 6h/7h Mode D skipped (grids exist), only 8h Mode D runs fresh
- `--grid-tag G_NARROW_D` — without it engine uses untagged grids (production B's)
- Both env vars `G_NARROW_MODELS_DIR` + `G_NARROW_CONFIG_DIR` set — confirms `[G_NARROW_D_ISO]` banner

**ETA actuals**:
| Phase | Estimated | Actual |
|---|---|---|
| Mode D × 1 (8h fresh) | ~12 min | ~11 min |
| Mode V × 3 horizons @ K=5 × 75 trials wider-Optuna | ~15h | ~17.7h (the 76%-of-wall figure that spawned TODO 0522) |
| Mode R + S + T+G | ~1.5h | ~1h 40min |
| **Total** | ~17h | **23h 22m** |

**Replaced**: failed 1805D (DIED on Desktop 2026-05-19 08-10 am with no diagnostic log — terminal-only output, no Tee). Lesson learned the hard way: always `Tee-Object *>&1` for long runs.

---

### ⚠️ CLOSED 2026-05-24 — TODO 0522 (Parallel refine speedup + long-horizon G test): Stage 1 PASSED, Stage 2 INVALID

**Outcome**: Stage 1 (parallel-fork correctness on 1 horizon) PASSED on Laptop 2026-05-22 ~00:26 — 3/3 configs refined under the 3-worker CPU-only dispatcher + parallel K=5 fan-out, refined configs sane and consistent with sequential baseline. Stage 2 (HRST 9,10,11,12h on Laptop, started 2026-05-22 01:39, completed ~18:09) wrote a full result set with Mode T REF +52.89% on 4-horizon basket — **but the verdict is INVALID** because the parallel fork was running the wrong hyperparameter grid.

**Root cause (discovered 2026-05-24)**: The fork imported `g_narrow_d` (for K=5 + module-level patches) and `crypto_trading_system_ed` (the H75 engine), then called `ENGINE.main()` at the bottom. `ENGINE.main()` reads `GRID_COMBOS/WINDOWS/FEATURES/GAMMAS/N_FEATURES_RANGE/MODELS_DIR/CONFIG_DIR/...` from ENGINE's own namespace — NOT from G's. So Stage 2 actually ran:
- Engine grid `RF+LGBM, XGB+LGBM, RF+XGB × [72,100,150] × [10,13,17,25] × [0.999,0.997,0.995]` (12 feature×gamma combos)

…instead of G's narrow grid `RF+LGBM, XGB+LGBM × [72,100,150] × [10,15,20] × [0.999,0.996]` (6 feature×gamma combos). It also used `N_FEATURES_RANGE = (4,40)/(4,80)` instead of G's `(4,100)`, and silently ignored the `G_NARROW_MODELS_DIR`/`G_NARROW_CONFIG_DIR` isolation env vars.

A wider grid samples ~2× more starting points for refine → biased toward higher max-APF candidates. So Stage 2's 9-12h numbers (11h +47% / 81% WR, 12h +30-58% / 75% WR, 9h +18% / 67% WR, 10h +34% / 82% WR) overstate clean-grid performance. The "9-12h is 11pp weaker than 5-8h" Stage 2 conclusion rests on tainted data and is invalidated.

**Fix (landed 2026-05-24 19:06)**: FIX #0 added to `crypto_trading_system_ed_g_narrow_d_parallel.py` — patches all the constants over to `ENGINE.*` from `G.*` at module load. Two follow-on fixes also landed before successful launch: `os.makedirs(_d, exist_ok=True)` loop (engine `to_csv` assumed dir exists), and module-level LGBM warnings filter (PowerShell `*>&1` wrapped them as `NativeCommandError`).

**What survives from Stage 1**: The 3 parallel patches themselves (device routing, parallel K=5 via `ThreadPoolExecutor`, 3-worker refine dispatcher) are identity-preserving and validated by Stage 1 smoke. They're the speedup core; the FIX #0 issue was orthogonal — a separate pre-existing namespace bug. If TODO 0524 confirms 5h reproduces the May 20-21 direct g_narrow_d.py baseline (RF+LGBM w≈281 γ≈0.998 f≈12 ret≈72%), the parallel patches can be ported into `crypto_trading_system_ed.py`'s H_STRICT_FAMILY K=5 block (lines 8831-8909) to speed up all future HRSTs by ~13h.

**Superseded by**: TODO 0524 (top-5 HRST 5,6,8,11,12h, launched Desktop 2026-05-24 20:27, reuses the same May 22 Laptop snapshot for direct A/B isolation of the bug-fix delta).

**Files preserved**: `logs/parallel_smoke_20260522_0026.log` (Stage 1 PASS), `logs/parallel_hrst_long_laptop_20260522_0139.log` (Stage 2 INVALID — kept for audit / before-after comparison with TODO 0524).

---

### 🟢 PROMOTED 2026-05-20 ~09:04 CEST — H75-fresh LIVE (config + production CSV swapped; engine unchanged)

**Source**: Laptop H75 HRST fresh-data run. Started 2026-05-18 23:38 (right after the H75-snapshot promotion the same evening), completed 2026-05-20 03:03 — total wall-clock ~28h on Laptop. Mode T REF +76.91%, converged at iteration 2. Detector unchanged from prior live (`sma24>sma100`), but per-horizon Mode V winners + regime split + Mode T gates all migrated.

**Decision sequence (chronological)**:
1. **2026-05-19 17:05 UTC**: prior H75-snapshot live fired BUY ETH @ $2,121.13 (bear regime, 8h@65% gate triggered at 65.30% confidence). First H75-driven live trade.
2. **2026-05-20 02:00 CEST**: SELL fired at $2,109.79, PnL −0.53% / −$75.02 / 9h hold. Trader went flat.
3. **2026-05-20 03:03 CEST**: Laptop HRST fresh-data Mode T completed, wrote final config to `regime_config_ed_noprod.json` + final production CSV to `crypto_ed_production_noprod.csv`.
4. **2026-05-20 09:04 CEST**: User explicitly invoked "promote to live". Pre-flight check confirmed trader state=cash (CLAUDE.md rule 19 satisfied). Backup files created. Live config + production CSV swapped from noprod files. **Engine file NOT swapped** — both prior H75-snapshot and new H75-fresh use the same `crypto_trading_system_ed.py` (H_STRICT_FAMILY merge from 2026-05-18). Promotion is config + per-horizon-models only.

**Per-horizon Mode V winners now live** (in `models/crypto_ed_production.csv`):
- 4h: legacy (not used at bull/bear levels)
- **5h: XGB+LGBM w=166 γ=0.997 15f conf=80%** → Mode V isolated return +53.76% / 141 trades / 71% WR (legacy under current symmetric 6h/6h regime split)
- **6h: RF+LGBM w=150 γ=0.999 10f conf=65%** → Mode V isolated return **+72.84% / 151 trades / 85% WR** ← **bull horizon (also used for bear in symmetric split)**
- **7h: RF+LGBM w=115 γ=0.9963 13f conf=75%** → Mode V isolated return +19.21% / 94 trades / 74% WR (legacy)
- **8h: XGB+LGBM w=158 γ=0.997 14f conf=80%** → Mode V isolated return +46.90% / 120 trades / 80% WR (legacy)
- 10h, 12h: legacy

**Live regime config now**:
- Detector: `sma24>sma100` (unchanged)
- Bull: 6h@65% (uses 6h winner model), gate rr8h≥2.0% OR rr12h≥2.0% cd=8h
- Bear: 6h@65% (uses same 6h winner), gate rr12h≥2.0% OR rr36h≥6.5% cd=6h
- Shields OFF (both), min_sell_pnl=0%, max_hold=10h, max_position_usd=$14,300

**Symmetric regime** is a structural change. Prior live had asymmetric horizon split (bull=5h/bear=8h on H75-snapshot; bull=6h/bear=8h further back). The fresh-data Mode S sweep picked 6h/6h as the joint winner with return +73.00% / 85 trades / WR 86%. Whether symmetric configs survive live OOS is an open question — all prior winning configs had asymmetric splits.

**Backups for one-command rollback** (already tested via copy-and-verify on promotion):
- `config/regime_config_ed_pre_H75fresh_20260520.json` (was the H75-snapshot config: bull=5h@75%, bear=8h@65%, gates rr8/14/cd=6 bull / rr10-12/cd=8 bear)
- `models/crypto_ed_production_pre_H75fresh_20260520.csv` (was the H75-snapshot per-horizon models)
- (Two-levels-back) `config/regime_config_ed_pre_H75_20260518.json` + `models/crypto_ed_production_pre_H75_20260518.csv` from 2026-05-18 promotion still preserved — see "PROMOTED 2026-05-18 ~22:02 CEST" entry below for context.

**Rollback (one level — to H75-snapshot, hot-reloads within 5 min)**:
```powershell
copy config\regime_config_ed_pre_H75fresh_20260520.json    config\regime_config_ed.json
copy models\crypto_ed_production_pre_H75fresh_20260520.csv models\crypto_ed_production.csv
```

**Monitoring criteria (CLAUDE.md R4) — rollback triggers over next 1-2 weeks**:
- Cumulative realized alpha < +5% after first 10 trades (sim Mode T REF was +76.91% over 60d)
- MaxDD exceeds −10% on new config alone (sim didn't expose per-trade DD; will monitor live)
- First 10 trades WR < 50% (Mode S sim was 86% WR on 85 trades over 60d)
- Trade count vastly different from sim 85/60d (~42/30d, ~1.4/day) — symmetric 6h/6h may produce more frequent signals than asymmetric 5h/8h split

**Caveats noted at promotion time**:
- **Mode T REF gap is within noise.** Prior H75-snapshot Mode T REF was approximately +76-77% (couldn't find exact number for one-to-one comparison); new H75-fresh is +76.91%. Promotion was a near-zero-EV swap on Mode T metric alone. User overrode my recommendation against promotion.
- **Per-horizon combos flipped 3 of 4 times** vs prior live (6h: XGB→RF, 7h: XGB→RF, 8h: RF→XGB). Strict-(combo,w) dedup is doing its job (picking different combos in different runs) but it's not converging to a stable winner across runs — Mode V refine surface remains bumpy despite K=5 multi-seed denoising.
- **Symmetric regime untested live**. First production config with bull-horizon == bear-horizon. Watch for regime-transition anomalies.
- **OOS window reset.** Prior H75-snapshot accumulated 1 closed trade in 1.5 days (−0.53%, PnL too small for any judgment). That data point now belongs to the H75-snapshot audit trail, not the new window.
- Engine 5-10pp run-to-run nondeterminism (GPU LGBM + joblib worker order) — accepted per CLAUDE.md.

**User override note**: my pre-promotion analysis (in conversation 2026-05-20 ~05:00) explicitly recommended AGAINST this swap, reasoning that (a) Mode T REF gap was within noise, (b) resetting the OOS monitoring window costs real data, (c) the G_narrow_d defensive check (TODO 0519) hadn't completed yet. User chose to promote anyway. Decision is logged here for audit-trail integrity — not a criticism, just preserved for future verdict-comparison.

**Architectural change**: NONE this round. Same engine, same K=5 / REFINE_TRIALS=75 / strict-(combo,w) dedup as H75-snapshot. Only config + per-horizon-models changed.

**Open follow-ups (carried to TODO.md)**:
- New H75-fresh OOS monitoring window (~2026-06-03), reset to 0/10 trades closed.
- TODO 0519 G_narrow_d relaunch tonight on Desktop — defensive check still valuable; if G > H75-fresh by >+5pp on snapshot data, that flags possible promotion regret.
- Verify trader hot-reload picks up new config within ~5 min of 09:04 swap (next signal at 10:00 UTC should fire on new model).

---

### 🚫 IDEA QUEUE drop-list (closed/shipped IDEA QUEUE items — quick lookup, 2026-05-19 audit)

Curated subset of the canonical scoreboard (C01-C86): only the IDs that lived briefly on the active IDEA QUEUE in TODO.md before being closed. Quick lookup so future audits don't re-add already-closed items. For the full per-CID scoreboard with evidence and revival conditions, see "CANONICAL IDEAS SCOREBOARD" below.

**Trigger for an item to land here**:
- Tested on fixed harness → DEAD / MARGINAL below +5pp ship threshold
- Tested → SHIPPED (in production already, no further test needed)
- STUB / blocked on architectural prerequisite (can't smoke-test)
- Documented SHELVED with revival condition (revival is its own future TODO if conditions are met)

**Re-adding to TODO.md requires evidence** the original verdict is invalidated (engine change since test, new hypothesis with different mechanism). Don't re-add as a "fresh start" — the verdict stands until overturned.

#### Tier 1 — SUSPECT retest batch (6 ideas, all DEAD on fixed harness)

Tested via `test_desktop_5ideas_runner.py` on 2026-05-07 (smoke Mode D delta vs baseline). Decision rule: ≥+5pp avg APF Δ to ship. **None cleared.**

| CID | Idea | Verdict | Avg Δ APF |
|---|---|---|---|
| **C35** | Wavelet multi-scale decomposition | MARGINAL | +0.227 |
| **C42** | CatBoost as 4th ensemble model | FAIL | −1.062 |
| **C43** | Stacking meta-learner (LR over RF+LGBM+XGB) | MARGINAL | +1.401 |
| **C44** | Quantile regression target (LGBM `objective='quantile', alpha=0.7`) | FAIL | −0.814 |
| **C47** | Vol-adjusted label (`ret_h / σ_h > threshold`) | MARGINAL | +0.547 |
| **C57** | Markov-switching AR detector | FAIL | −1.574 |

Full closure entry: "Closed 2026-05-07 — 5ideas runner Tier 1 retest batch" (below). Underlying log: [logs/desktop_5ideas_summary_20260507_002108.txt](logs/desktop_5ideas_summary_20260507_002108.txt).

#### Tier 2 — SUSPECT Tier C (5 ideas, all DEAD on 2026-05-10 batch)

7-idea batch via `tools/test_idea_batch_C03_C12_C23_C29_C31_C48.py` on 2026-05-10. Fixed harness + clean baseline. **Net: 0 PASS, 1 MARGINAL+, 6 FAIL.**

| CID | Idea | Verdict | Avg Δ APF |
|---|---|---|---|
| **C03** | SHAP feature ranking | FAIL | −2.91 |
| **C12** | Stability filter @ thr=50 | FAIL | −4.66 |
| **C23** | Per-regime feature set | FAIL | −1.86 |
| **C29a** | HAR-RV (Corsi 2009) | MARGINAL+ | +0.63 (5h-only win; not actionable globally) |
| **C29b** | Hurst exponent | FAIL | −1.89 |
| **C31** | Funding momentum / acceleration | FAIL | −1.90 |

Note on C29: 5h +6.66 isolated win mirrors the C05/C06 per-horizon-vs-aggregator pattern — could be revivable via per-horizon-only application, but not as a global feature add.

Full closure entry: "Closed 2026-05-10 (overnight mid-effort batch)" below.

#### Tier 3 — Untested clean items closed (4 ideas)

| CID | Idea | Outcome | Date |
|---|---|---|---|
| **C16-narrow** | Disaster brake at −5% retest | **SHIPPED MARGINAL+** (+1.96pp / 1 fire on 90d). User keeps disabled per CLAUDE.md preference — decision made, dormant. | 2026-05-09→10 |
| **C48** | Sharpe-aware label | FAIL Δ−2.92 (5h −9.98 catastrophic) | 2026-05-10 batch |
| **C52** | ATR-based vol-adapted trailing stop | DEAD — best 0pp (no fires); all firing variants lost (−3.78 to −13.39pp). 12 sweep configs. | 2026-05-09→10 |
| **C53** | Volume-spike exit trigger | DEAD — all 12 sweep variants −0.74 to −2.49pp | 2026-05-09→10 |

Full closure entry: "Closed 2026-05-09 → 2026-05-10 (overlay-tier batch)" below.

#### Tier 3 — STUB-blocked (3 ideas, can't smoke-test as-is)

| CID | Idea | Blocker |
|---|---|---|
| **C45** | Conformal prediction calibrated intervals | Architectural change required — engine has no per-trade interval representation. Distinct from C18 Platt (DEAD). |
| **C46** | Bayesian hyperparameter optimization (BoTorch GP) | Engine extension required — replace Optuna TPE sampler with GP. Non-trivial integration. |
| **C49** | Multi-class label (strong/weak buy/sell + hold) | Architectural label change — needs Mode V update + threshold mapping. |

These cannot be tested via standard Mode D smoke; require engine work first. If a real architectural change is in scope someday, these become un-blocked.

---

### ❌ Closed 2026-05-19 — TODO 1805D G_narrow_d Desktop relaunch CRASHED, no diagnostic log

**Search anchor**: `TODO 1805D`

**Status**: ❌ **DIED on Desktop between 08:00 and 10:00 on 2026-05-19.** Launched ~04:56 (per noprod CSV mtime in `models_g_desktop/`), wrote 5h Mode V winner to row 41 of `models_g_desktop/crypto_ed_production_noprod.csv` (ETH XGB+LGBM w=279 γ=0.9979 11f +58.87% — G's signature wide-Optuna pattern), then continued silently into 6h Mode V. **No diagnostic log was captured** — the engine was launched directly with `python ...` and stdout went to the Desktop terminal, not a log file. User saw "a lot of purple lines" — likely LightGBM/sklearn benign warnings amplified by K=5 multi-seed, NOT necessarily fatal signal.

**Lesson learned (now applied to TODO 0519 relaunch)**: always launch long HRSTs with `Tee-Object *>&1 | -FilePath` to capture stdout+stderr to disk. Otherwise crash diagnostics are lost when the terminal closes.

**Top crash suspects (in order of probability)**:
1. GPU VRAM OOM under K=5 + wider Optuna (~40%)
2. Modern Standby / accidental sleep (~25%)
3. Terminal closed accidentally (~10%)
4. Python OOM under joblib forks (~10%)
5. Other (~15%)

**Useful artifacts preserved** (used by TODO 0519 relaunch):
- Row 41 in `models_g_desktop/crypto_ed_production_noprod.csv` (G's 5h winner — survives subsequent Mode V writes via coin+horizon filter per CLAUDE.md rule 2)
- `models_g_desktop/crypto_ed_grid_ETH_5h_G_NARROW_D.csv`, `_6h_`, `_7h_` (mtime 2026-05-17 from prior Laptop runs — preserved, used with `--skip` on relaunch)
- `models_g_desktop/crypto_ed_best_models.csv` (updated 05:19 — G's working state at death)

**Relaunch**: tracked under TODO 0519 in TODO.md with mandatory output-capture safeguard so the next crash (if any) produces a diagnosable log.

The original full launch sequence + isolation setup + cross-contamination matrix that lived in this TODO is preserved in git history (`git log -p TODO.md`). TODO 0519 supersedes it with a streamlined 6h+7h+8h-only relaunch saving ~5h vs full HRST.

---

### 🟢 PROMOTED 2026-05-18 ~22:02 CEST — H75 LIVE (engine + config + production CSV all swapped)

**TODO 1706 outcome: SHIP.** H_strict_family HRST completed on Desktop midday 2026-05-18. The 30-day calendar-aligned backtest comparing H75 vs B_multi_seed (May 15 archive) vs LIVE_prod (May 6 promotion) ranked H75 #1 on every metric: +20.85% return / +32.01pp alpha vs B&H / 85% WR / −3.72% MaxDD / +0.77% avg win. Output: [output/compare_h_b_prod_30d_20260518_201636.csv](output/compare_h_b_prod_30d_20260518_201636.csv).

**Decision sequence (chronological):**
1. ETH SELL fired organically at 2026-05-18 22:00 UTC at $2,125.92 (PnL −0.18% / −$25.54). Position went to cash.
2. Engine file swap executed at 22:26:57 — `crypto_trading_system_ed_h_strict_family.py` → `crypto_trading_system_ed.py`. K=5 multi-seed + REFINE_TRIALS=75 + strict (combo, w) dedup now ACTIVE for any future Mode D/V/HRST.
3. Live config + production CSV swap executed at 22:02:27 once trader was flat. New live: `sma24>sma100` / bull=5h@75% / bear=8h@65% / shields OFF / per-regime gates (rr8h≥2.0 OR rr14h≥6.0 cd=6h bull; rr10h≥5.5 OR rr12h≥2.0 cd=8h bear) / min_sell_pnl=0% / max_hold=10h.
4. Trader hot-reloaded the new config within 5 min, kept running. Engine smoke test (`python -c "import crypto_trading_system_ed"`) confirmed all 3 H_STRICT_FAMILY banners fire on import → engine swap end-to-end verified.

**Per-horizon model winners now live** (in `models/crypto_ed_production.csv`):
- 4h: XGB+LGBM w=170 γ=0.9989 11f (legacy, unused at bull/bear levels)
- **5h: XGB+LGBM w=100 γ=0.9993 7f** ← bull horizon
- 6h: XGB+LGBM w=106 γ=0.9986 18f (legacy)
- 7h: XGB+LGBM w=100 γ=0.9990 20f (legacy)
- **8h: RF+LGBM w=162 γ=0.9954 6f** ← bear horizon
- 10h: RF+XGB w=166 γ=0.9958 9f (legacy)
- 12h: XGB+LGBM w=300 γ=0.9990 17f (legacy)

**Backups for one-command rollback:**
- `config/regime_config_ed_pre_H75_20260518.json` (was tsmom_672h / 6h@75% / 8h@65% / gates rr18-30 cd=36 bull / gates rr24-30 cd=14 bear / min_sell_pnl=0.5)
- `models/crypto_ed_production_pre_H75_20260518.csv` (was May 6 LIVE: 6h XGB+LGBM w=150, 8h RF+LGBM w=150)
- `crypto_trading_system_ed_pre_H75_20260518.py` (pre-swap engine code, 412KB)

**Rollback (instant, hot-reloads within 5 min):**
```powershell
copy config\regime_config_ed_pre_H75_20260518.json    config\regime_config_ed.json
copy models\crypto_ed_production_pre_H75_20260518.csv models\crypto_ed_production.csv
# Optional — only if engine-layer rollback also wanted:
copy crypto_trading_system_ed_pre_H75_20260518.py     crypto_trading_system_ed.py
```

**Monitoring criteria (CLAUDE.md R4) — rollback triggers over next 1-2 weeks:**
- Cumulative realized alpha drops >15pp vs sim baseline (H75 sim was +20.85% on 30d backtest; if live <+5% after first 10 trades → rollback)
- Max DD exceeds −10% (sim −3.72%)
- First 10 trades WR < 50% (sim 85%)
- Trade count vastly different from sim 47 trades/30d (H75's tighter cd=6h bull gate means BUYs fire much more often than under old cd=36h — expect 1.5-2× the historic LIVE trade frequency)

**Caveats noted at promotion time:**
- 30-day backtest is **fully in-sample** for H75 (HRST ran today against the same 30-day window we backtested on). Expected ~3-7pp in-sample bias. True OOS performance likely lower.
- Sim-vs-live execution gap measured at ~7pp on the prior LIVE config (live realized +4.52% vs sim +11.46%). Same gap likely applies uniformly → expected live 30d return on H75 ≈ +8-14%.
- Engine 5-10pp run-to-run nondeterminism (GPU LGBM + joblib worker order, accepted per CLAUDE.md).
- G_narrow_d (the third comparison variant per TODO 1705) was not completed — laptop crashed mid-HRST 2026-05-17. User intends to relaunch G later for full 3-way verdict.

**Architectural change now live in production engine** (applies to all future Mode D/V/HRST runs):
- `_diversity_key` returns `(combo, w)` — strict 1-per-(model_family, window) cluster
- `REFINE_TRIALS = 75` (up from 50)
- K=5 multi-seed denoising wraps `_deku_eval_with_pruning` (default `RELIABILITY_K=5`)
- Snapshot redirect + output isolation patchers present but env-var-gated (defaults to production paths)
- Future HRST runtime: ~20h (vs ~6-9h pre-H75)

**Open follow-ups (carried to TODO.md):**
- Relaunch G_narrow_d on Laptop for the 3-way verdict (now tracked under TODO 1805D in TODO.md after Laptop crashes; Desktop run in flight).
- Watch for first H75-driven BUY signal (5h@75% bull or 8h@65% bear with new gates).
- After ~1-2 weeks of live H75 performance, audit realized return vs sim and decide whether to keep, rollback, or rerun comparison (tracked under `H75-OOS-MONITOR` in TODO.md).

---

### ✅ TODO 1805 — H75 HRST on fresh data (Desktop, 2026-05-18 late evening) — PROMOTED 2026-05-19

**Search anchor**: `TODO 1805`

**Status (2026-05-19, updated by user)**: ✅ **FINISHED + PROMOTED to production**. Run completed on Desktop. Winner promoted to live (replaces the May-15 snapshot H75 with the fresh-data H75). Laptop sanity-check now in progress to confirm engine + config round-trip works correctly post-promotion.

**Original problem (now resolved)**: An earlier attempt 2026-05-18 22:36 crashed silently — log `ed_v1_20260518_223644.log` shows Mode D 5h aborted with "Capped: 76,348 -> 168 rows. Not enough data. Skipping. Aborting. Done!" because the CLI arg `--1440` wasn't normalized to `--replay 1440` (default `period=168h` was used). User relaunched with correct syntax and the run completed successfully.

**Why this existed**: H75 was promoted to live earlier tonight (~22:02 CEST) based on an HRST that ran against the `_reliability_hrst_snapshot_desktop_20260515_154801/` snapshot (data frozen May 15). User wanted a fresh-data re-run of the same architecture (H75 engine = strict (combo,w) dedup + REFINE_TRIALS=75 + K=default multi-seed) to validate that the May 15 snapshot didn't bias the per-horizon Mode V winners — and to get a more current OOS read before the first H75-driven live trades fire.

**Launch command used**:

```powershell
python .\crypto_trading_system_ed.py ETH 5,6,7,8h HRST --1440 --nopersist
```

CLI normalization handled `--1440 → --replay 1440` and `--nopersist → --no-persist` (engine lines 7015-7026). No trailing comma needed on `ETH` since the parser was merged with the asset-first ordering fix.

**Key differences vs the 2026-05-18 promotion HRST**:
- **Fresh data**: NO `--no-data-update` → engine downloaded macro + onchain + derivatives + OHLCV at startup (~5-10 min), wrote to live `data/macro_data/*.csv`.
- **No env-var overrides**: `V2_DATA_SNAPSHOT` not set → live data path used. `RELIABILITY_K` not set → default K.
- **`--no-persist` ON**: writes went to `crypto_ed_production_noprod.csv` + `regime_config_ed_noprod.json`. Live trader untouched during run.

**Outcome**: Fresh-data winner promoted to live, replacing the May-15-snapshot config. Now monitored under `H75-OOS-MONITOR` in TODO.md.

---

### ✅ TODO 1805B — Migrated from CLAUDE.md: 5 pending research items — CLOSED 2026-05-19 (all already resolved in canonical scoreboard)

**Search anchor**: `TODO 1805B`

**Status (2026-05-19, updated by user)**: ❌ **CLOSED — all 5 items either already tested+shelved or auto-resolved**. The 2026-05-18 migration from CLAUDE.md surfaced research items that looked open in the Engine Reference Card but had already been resolved elsewhere in TODO.md's canonical scoreboard (CIDs C01-C82). Audit results below.

#### Cross-reference audit vs canonical scoreboard

| Item | CID | Canonical verdict | Date | Verdict still valid? |
|---|---|---|---|---|
| 1. Vol-scaled horizons 4mo | **C01** | SHELVED — `vol_2band` +2.99pp on 4mo (below +5pp ship), window-shopping confirmed | 2026-05-09 | ✅ Yes. H75 detector change (`sma24>sma100`) doesn't revive vol-scaled — different family. |
| 2. ETH derivatives Mode D rerun | (no CID, feature track) | Auto-resolved — features now in H75's 184-feature pool | 2026-05-18 | ✅ Yes. Verify in H75 winner feature lists if `deriv_funding_chg1d` / `deriv_oi_*` appear. |
| 3. Per-regime feature set | **C23** | DEAD — Δ−1.18 (bias bull→tech / bear→macro hurts) | 2026-05-04 | ✅ Yes. |
| 4. Multi-horizon emergency exit | **C02** | DEAD — all 16 variants lose vs aligned-window baseline (best −3.30pp, worst −68.71pp) | 2026-05-04 | ✅ Yes. Shield + max_hold already handle crashes. |
| 5. On-chain correlation-crowded-out | (priority low) | Untested in depth; `oc_mvrv_chg1d` Grade 4 alone is already valuable | 2026-04-19 | ✅ Yes. Marked low priority; no action needed. |

**Net**: 4 items confirmed dead/shelved, 1 item auto-resolved by H75. No remaining research to perform from this migration.

**Lesson**: when migrating items from one tracking surface to another, cross-check the canonical CID scoreboard FIRST to avoid revisiting already-closed work. This migration didn't do that and the work above repeats verdicts already recorded under their CIDs.

The original verbose item-by-item migration content was preserved in TODO.md during 2026-05-19's first cleanup pass but is omitted here for brevity — the canonical verdicts in the table above are authoritative. Full original notes live in git history (`git log -p TODO.md`).

---

### ✅ TODO 1805C — Relaunch G_narrow_d full HRST on Laptop (2026-05-18) — CLOSED 2026-05-19 (CRASHED, REPLACED BY 1805D)

**Search anchor**: `TODO 1805C`

**Status (2026-05-19, updated by user)**: ❌ **CRASHED on Laptop** — second crash mid-HRST consistent with the 2026-05-17 failure pattern (Modern Standby / GPU OOM / Drive sync conflict — exact root cause not pursued). **Replaced by relaunch on Desktop this morning** (now tracked under TODO 1805D in TODO.md, which is the surviving G run). This TODO is closed; no further Laptop relaunch planned.

**Why this existed**: G_narrow_d was supposed to finish on Laptop per TODO 1705 (2026-05-17) but the Laptop crashed mid-HRST. The H75 promotion entry's "Open follow-ups" explicitly listed "Relaunch G_narrow_d on Laptop for the 3-way verdict (TODO 1705 — diagnose laptop crash first)." After H75 was live and the TODO 1805 H75-on-fresh-data was occupying Desktop, the Laptop was the right machine to relaunch G on in parallel. Both Laptop attempts crashed. Switched to Desktop sequential under TODO 1805D.

**Goal at the time**: complete the 3-way verdict table from TODO 1706 — `B (production, +89.41% Mode T) vs G_narrow_d (?) vs H75 (?)`. Now carried by 1805D.

The full launch command + critical guards + startup banner checklist + per-horizon comparison table that were originally documented in this entry remain valid for the Desktop relaunch (TODO 1805D in TODO.md) — they have not been duplicated here to avoid drift. If launching G on Laptop again in the future, refer to git history (`git log -p TODO.md`) for the full pre-launch checklist.

---

### 🟢 TODO 1706 — H_strict_family full HRST @ 75 trials on Desktop (2026-05-17 evening) — COMPLETED, SEE PROMOTED 2026-05-18

**Search anchor**: `TODO 1706`

**Outcome**: SHIP. See the PROMOTED 2026-05-18 entry above. H75 went live 22:02 CEST.

**Why this existed**: TODO 1705 launched G_narrow_d full HRST on Laptop concurrently. Both runs shared the same Drive-synced engine directory, which meant they'd compete for writes to `models/crypto_ed_production_noprod.csv`, `config/regime_config_ed_noprod.json`, and `models/crypto_ed_best_models.csv` — corrupting both runs' Mode T results and risking production-CSV pollution. This TODO set up H_strict_family @ 75 trials (per 2026-05-17's REFINE_TRIALS bump) with **fully isolated output directories** so G and H could run concurrently without infecting each other or production.

**Isolation mechanism (added 2026-05-17 to engine fork)**: `crypto_trading_system_ed_h_strict_family.py` reads `H_STRICT_MODELS_DIR` and `H_STRICT_CONFIG_DIR` env vars and redirects ALL writes. Default behavior preserved when env vars unset. Confirmed by startup banner: `[H_STRICT_FAMILY_ISO] output dirs redirected: models=models_h75 config=config_h75`.

**Launch command used**:
```powershell
$env:V2_DATA_SNAPSHOT = "data\_reliability_hrst_snapshot_desktop_20260515_154801"
$env:RELIABILITY_K = "5"
$env:H_STRICT_MODELS_DIR = "models_h75"
$env:H_STRICT_CONFIG_DIR = "config_h75"
python crypto_trading_system_ed_h_strict_family.py HRST "ETH," 5h,6h,7h,8h --skip --replay 1440 --no-persist --no-data-update --grid-tag H_STRICT_FAMILY
```

**Critical guards verified live**: trailing comma on `"ETH,"` (older fork's CLI parser bug), `--no-data-update` (snapshot redirect only affects reads), `--no-persist` (writes go to `*_noprod.*` in isolated dirs), `--skip` (skip Mode D for all 4 horizons since grids existed pre-launch), both env vars set (otherwise H writes to default `models/`/`config/`), `REFINE_TRIALS=75` verified via startup banner.

**Result**: H75 won the 3-way 30-day backtest vs B_multi_seed and LIVE_prod by every metric. See PROMOTED 2026-05-18 entry for full per-horizon winners + monitoring criteria.

The full pre-launch checklist + startup-banner verification flow used by this TODO has been re-applied to 1805D (Desktop G isolated run) — refer to TODO.md's 1805D entry for the current working version, or git history for this TODO's full step-by-step.

---

### 🔍 TODO 1705B — Pre-launch audit of G_narrow_d + H_strict_family (2026-05-17 evening)

**Search anchor**: `TODO 1705B`

**Purpose**: Verification record of both engine forks before launching G HRST (TODO 1705) on Laptop and H75 HRST (TODO 1706) on Desktop concurrently. Confirms architectural parameters match specs, output isolation is wired correctly, and no shared-file collisions exist between the two runs.

#### G_narrow_d audit (file: [crypto_trading_system_ed_g_narrow_d.py](crypto_trading_system_ed_g_narrow_d.py))

| Item | Expected | Actual | Line |
|---|---|---|---|
| GRID_COMBOS | [RF+LGBM, XGB+LGBM] | matches | 4044 |
| GRID_WINDOWS | [72, 100, 150] | matches | 4052 |
| GRID_FEATURES | [10, 15, 20] (wider spacing) | `[10, 15, 20]` | 4056 |
| GRID_GAMMAS | [0.999, 0.996] (wider spacing) | `[0.999, 0.996]` | 4058 |
| V1 total configs | 36 (2×3×3×2) | math checks out | — |
| REFINE_TRIALS | 75 | `75` | 4063 |
| N_FEATURES_RANGE_DEFAULT | (4, 100) — cap dropped | `(4, 100)` | 498 |
| `_diversity_key` | combo-aware | `(combo, window, g_band, f_band)` + force ≥1 per combo | 4652-4659 |
| V3 Optuna ranges | absolute (not seed-relative) | `window [50,300] features [4,60] gamma [0.995,1.0]` | 5263-5268 |
| K=5 multi-seed | via RELIABILITY_K env var | inlined patcher at module end | — |
| Snapshot redirect | via V2_DATA_SNAPSHOT env var | inlined patcher at module end | — |
| Output paths | default `models/` and `config/` (intentional) | hardcoded | 528, 1067 |

**Status**: ✅ PASS — all G_narrow_d parameters match the architectural spec.

#### H_strict_family audit (file: [crypto_trading_system_ed_h_strict_family.py](crypto_trading_system_ed_h_strict_family.py))

| Item | Expected | Actual | Line |
|---|---|---|---|
| GRID identical to B | combos, windows, features, gammas all match B | matches | 4032-4047 |
| V1 total configs | 72 (B's full grid) | math checks out | — |
| REFINE_TRIALS | 75 (bumped 2026-05-17 from 50) | `75` | 4051 |
| N_FEATURES_RANGE_DEFAULT | (4, 80) — B's cap kept | `(4, 80)` | 486 |
| `_diversity_key` | strict `(combo, window)` — max 6 clusters | `return (combo, w)` | 4646-4649 |
| V3 Optuna ranges | seed-relative narrow (B's) | unchanged from B | 5234-5236 |
| K=5 + snapshot + dedup banner | inlined patchers at module end | dedup banner at line 8907 | — |
| Output isolation block | env-var driven | reads `H_STRICT_MODELS_DIR` + `H_STRICT_CONFIG_DIR` | 294-298 |
| RESUME_DIR | uses H75_MODELS_DIR | `f'{H75_MODELS_DIR}/.resume_hourly'` | 302 |
| PRODUCTION_CSV | uses H75_MODELS_DIR | `f'{H75_MODELS_DIR}/crypto_ed_production.csv'` | 516 |
| REGIME_CONFIG_PATH | uses H75_CONFIG_DIR | `f'{H75_CONFIG_DIR}/regime_config_ed.json'` | 517 |
| MODELS_DIR (used elsewhere) | aliases H75_MODELS_DIR | `MODELS_DIR = H75_MODELS_DIR` | 1055 |
| CONFIG_DIR | aliases H75_CONFIG_DIR | `CONFIG_DIR = H75_CONFIG_DIR` | 1056 |
| All `{MODELS_DIR}/...` derived writes propagate | best_models, best_models_backup, chart_data, diagnostic_results, feature_analysis, grid CSVs | use MODELS_DIR variable | 1078, 1079, 1084, 1087, 2405, 3195, 3653, 4524 |
| `--no-persist` derivation | `_noprod` versions inherit isolated dirs | line 7200-7201 uses PRODUCTION_CSV.replace + REGIME_CONFIG_PATH.replace | 7200-7201 |

**Status**: ✅ PASS — H75 isolation wired end-to-end.

#### ⚠️ Single remaining hardcoded path (dormant)

`crypto_trading_system_ed_h_strict_family.py:7270`:
```python
PRODUCTION_CSV = f"models/crypto_ed_production_lt{pct:g}.csv"
```
Only triggers with `--label-threshold X` CLI flag. NOT used in TODO 1706 launch (which uses HRST mode without --label-threshold). **Dormant — safe for tonight's launch.**

**Future fix (defensive consistency, low priority)**: env-var this path too, in case a future `--label-threshold` run is launched without re-staging the isolation. Tracked here, not blocking.

#### Cross-contamination matrix (G default paths + H isolated paths)

| Shared file | G_narrow_d writes to | H75 writes to (env vars set) | Collision? |
|---|---|---|---|
| Grid CSVs | `models/grid_ETH_<h>h_G_NARROW_D.csv` | `models_h75/grid_ETH_<h>h_H_STRICT_FAMILY.csv` | ✅ none |
| `crypto_ed_production_noprod.csv` | `models/` | `models_h75/` | ✅ none |
| `crypto_ed_best_models.csv` | `models/` | `models_h75/` | ✅ none |
| `crypto_ed_best_models_backup.csv` | `models/` | `models_h75/` | ✅ none |
| `regime_config_ed_noprod.json` | `config/` | `config_h75/` | ✅ none |
| Live trader `models/crypto_ed_production.csv` | never (--no-persist) | never (--no-persist) | ✅ trader safe |
| RESUME_DIR | `models/.resume_hourly/` | `models_h75/.resume_hourly/` | ✅ none |
| Logs `logs/ed_v1_*.log` | timestamped name | timestamped name | ✅ unique names |
| Snapshot dir `data/_reliability_hrst_snapshot_*` | read-only | read-only | ✅ read-only |

**Status**: ✅ PASS — zero shared writes between G (Laptop, default paths) and H75 (Desktop, isolated paths). Live trader CSV is never touched by either run (`--no-persist`).

#### Desktop process state at audit time (2026-05-17 ~19:00)

4 python.exe processes running, all are the live trader chain:
- `tee_launcher → crypto_revolut_ed_v2.py --loop` (PID 17720 → 18768 → 13232 → 26344)
- Trader memory: ~397 MB
- **NO in-flight H 8h V process** — earlier 18:37 launch already terminated (log [ed_v1_20260517_183730.log](logs/ed_v1_20260517_183730.log) ends with `KeyboardInterrupt` at `ProcessPoolExecutor.shutdown`)

**Implication for TODO 1706**: Step 0 ("kill in-flight H 8h@50") is unnecessary — already dead. Proceed directly to Step 1 setup + Step 2 launch.

#### Audit verdict

| Aspect | Status |
|---|---|
| G_narrow_d architecture matches spec | ✅ PASS |
| H_strict_family architecture matches spec | ✅ PASS |
| H75 output isolation wired correctly | ✅ PASS |
| No shared writes between G + H concurrent runs | ✅ PASS |
| Live trader insulated from both | ✅ PASS (--no-persist for both) |
| In-flight processes need cleanup before H75 launch | ✅ Already clean |

**Cleared for concurrent launch**: G HRST (Laptop, TODO 1705) + H75 HRST (Desktop, TODO 1706). Verify by checking startup banners listed in each TODO's "Startup banners to confirm" section.

---

### 🟢 TODO 1705 — Finish G_narrow_d full HRST on Laptop (2026-05-17 17:05)

**Search anchor**: `TODO 1705`

**Why this exists**: G_narrow_d's Desktop HRST today (started 02:28, log [ed_v1_20260517_022856.log](logs/ed_v1_20260517_022856.log)) completed 5h Mode V (+74.54% RF+LGBM 281h) and 6h Mode V (+46.40% RF+LGBM 137h) but was killed mid-7h-refine. 8h Mode D never ran. Mode R/S/T never reached. The 5h/6h winners were ALSO wiped from `crypto_ed_production_noprod.csv` when H started the V 8h-only run at 18:37 (the `--no-persist` startup seed copied production B values back over G's writes). The only surviving record of G's 5h/6h winners is the LOG file. Need full HRST resume to produce a real Mode T verdict for G_narrow_d vs B (+89.41%) and vs H_strict.

**Why on Laptop**: Desktop is occupied (H_strict V 8h in flight, finishing ~21:45). Trader also runs on Desktop with priority arrangement (no contention there — but launching a 3rd heavy run on Desktop adds nothing vs Laptop). Laptop is free.

#### Step 1 — Snapshot existing G artifacts (run BEFORE the launch)

```powershell
cd "G:\Autres ordinateurs\My laptop\engine"
$ts = Get-Date -Format "yyyyMMdd_HHmm"
$dest = "models\_archive_g_narrow_d_$ts"
New-Item -ItemType Directory -Path $dest -Force | Out-Null
Copy-Item models\crypto_ed_grid_ETH_*_G_NARROW_D.csv $dest
Copy-Item logs\ed_v1_20260517_022856.log $dest
Copy-Item logs\ed_v1_20260516_144648.log $dest
```

#### Step 2 — Launch G full HRST resume

```powershell
$env:V2_DATA_SNAPSHOT = "data\_reliability_hrst_snapshot_desktop_20260515_154801"
$env:RELIABILITY_K = "5"
python crypto_trading_system_ed_g_narrow_d.py HRST "ETH," 5h,6h,7h,8h --skip --replay 1440 --no-persist --no-data-update --grid-tag G_NARROW_D
```

**Critical guards** (all live-tested gotchas):
- **`"ETH,"` with TRAILING COMMA, double-quoted** — without comma, `ETH` matches `endswith('h')` horizon parser → ALL 9 assets get loaded. Confirm `ASSET: ETH` at startup, not `ASSET: BTC`.
- **`--no-data-update` REQUIRED** — without it the engine spends ~10 min downloading macro/onchain/derivatives AND writes to LIVE `data/macro_data/*.csv`. Snapshot redirect only affects READS.
- **`--no-persist` REQUIRED** — without it, production files (and the live trader's reads) get overwritten with G's research configs.
- **`--skip`** — skips Mode D for horizons with existing grid CSVs (5h/6h/7h all done from prior runs). Without `--skip`, Mode D re-runs for all 4 (~40 min wasted).

#### Startup banners to confirm

- `[G_NARROW_D_SNAPSHOT] pd.read_csv redirected: data/<file> -> _reliability_hrst_snapshot_desktop_20260515_154801/<file>`
- `[G_NARROW_D_K5] _deku_eval_with_pruning patched (K=5 seeds=[42, 43, 44, 45, 46])`
- `ED: Mode HRST | ETH | 5h,6h,7h,8h | 150 trials | --skip`
- Per horizon: either `[--skip] Found existing grid CSV — skipping Mode D` (5h/6h/7h) or `EXHAUSTIVE GRID: ETH 8h — 36 evals` (8h, the only new D)
- After Mode H: `MODE R`, then `MODE S`, then `MODE T`, then `MODE G` (T chains G automatically)

#### ETA (Laptop)

- Mode D 8h: ~12 min
- Mode V 4 horizons re-run at K=5 + 75 trials + wider-Optuna: ~20h (G's per-horizon Mode V measured at ~5h on Desktop; Laptop ~30% slower → ~6.5h each × 4 = ~26h worst case)
- Mode R + S + T+G: ~1.5h
- **Total: ~22-27h. Done midday-to-evening 2026-05-18.**

#### Step 3 — After HRST completes, snapshot final results

```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmm"
$dest = "models\_archive_g_narrow_d_COMPLETE_$ts"
New-Item -ItemType Directory -Path $dest -Force | Out-Null
Copy-Item models\crypto_ed_grid_ETH_*_G_NARROW_D.csv $dest
Copy-Item models\crypto_ed_production_noprod.csv "$dest\production_FINAL.csv"
Copy-Item models\crypto_ed_best_models.csv "$dest\best_models_FINAL.csv"
Copy-Item config\regime_config_ed_noprod.json "$dest\regime_config_FINAL.json"
Copy-Item logs\ed_v1_$(Get-Date -Format yyyyMMdd)_*.log $dest
```

#### Success criteria → next step

| Mode T total | Decision |
|---|---|
| G > B's +89.41% | G_narrow_d is a real production candidate. Compare against H_strict's HRST (TODO 1706 once H finishes V 8h@50, then re-run H full HRST at 75 trials per today's bump). Ship the higher of G/H when trader flat. |
| G < B's +89.41% | G's wider-Optuna architecture didn't beat production despite finding stronger per-horizon Mode V winners. Likely Mode S regime collapse (same pattern that killed F_optimized). SHELVE G_narrow_d. |
| G ≈ B (within ±5pp) | Inconclusive on G alone; full H@75 HRST becomes the decisive test. |

#### Comparison context (filled per-horizon table — fill Mode T row when HRST finishes)

| Horizon | B (production) | G_narrow_d | H_strict @ 50 | H_strict @ 75 (pending) |
|---|---|---|---|---|
| 5h Mode V | +59.05% XGB+LGBM 150h | **+74.54%** RF+LGBM 281h | not run | not run |
| 6h Mode V | +68.47% XGB+LGBM 150h | +46.40% RF+LGBM 137h | not run | not run |
| 7h Mode V | +11.21% XGB+LGBM 72h (lottery) | +29.94% XGB+LGBM 100h | +27.43% XGB+LGBM 100h | not run |
| 8h Mode V | +56.54% RF+LGBM 150h | not run | in-flight | not run |
| **Mode T total** | **+89.41%** (proven) | pending TODO 1705 | n/a (no HRST) | n/a (no HRST yet) |

---

### 🟢 P0 — ACTIVE — G_narrow_d 7h-only diagnostic (engine fork, 2026-05-16)

## 🚀 LAUNCH COMMAND (Laptop) — run this now

```powershell
# In a Laptop PowerShell terminal, from the engine directory:
$env:V2_DATA_SNAPSHOT = "data\_reliability_hrst_snapshot_desktop_20260515_154801"
$env:RELIABILITY_K = "5"
python crypto_trading_system_ed_g_narrow_d.py D V "ETH," 7h --replay 1440 --no-persist --no-data-update --grid-tag G_NARROW_D
```

**CRITICAL** — two non-obvious requirements (both hit live during the 2026-05-16 launch and re-fixed):

1. **`"ETH,"` with TRAILING COMMA, double-quoted.** The engine's CLI parser at line 7347 treats `'eth'.endswith('h')` as a horizon match → eats ETH → asset_list defaults to ALL 9 ASSETS (BTC,ETH,XRP,SOL,LINK,BNB,SMI,DAX,CAC40). Trailing comma forces the asset-list branch. **If you see `ASSET: BTC` instead of `ASSET: ETH` after launch, kill it — the comma is missing.**

2. **`--no-data-update` is REQUIRED.** Without it the engine spends ~10 min downloading fresh macro/onchain/derivatives data, which (a) wastes time, (b) writes to LIVE `data/macro_data/*.csv` (V2_DATA_SNAPSHOT only redirects READS, not writes), and (c) creates trader-contamination risk. **If you see `MACRO & SENTIMENT DATA DOWNLOAD` lines, kill it — the flag is missing.**

**Pre-flight check**: `ls $env:V2_DATA_SNAPSHOT` should list ~47 CSV files. If empty, the Drive snapshot didn't sync to the laptop's working dir — fix before launching.

**Look for these prints at startup (post-fix)**:
- `[G_NARROW_D_SNAPSHOT] pd.read_csv redirected: data/<file> -> _reliability_hrst_snapshot_desktop_20260515_154801/<file>` ← snapshot redirect ON
- `[G_NARROW_D_K5] _deku_eval_with_pruning patched (K=5 seeds=[42, 43, 44, 45, 46])` ← K=5 active
- `[--no-data-update] Skipping macro + OHLCV downloads. Using whatever is currently on disk.` ← downloads OFF
- `ED: Mode D | ETH | 7h | 150 trials` ← ETH-only (NOT `BTC,ETH,XRP,SOL,LINK,BNB,SMI,DAX,CAC40`)
- `EXHAUSTIVE GRID: ETH 7h — 36 evals` ← G_narrow_d's narrow-spacing grid (36, not 72)

If `V2_DATA_SNAPSHOT not set — reading from live data/` appears, the snapshot env didn't load — abort and re-export.

**ETA**: ~2-3h on Laptop. Trader can stay active — snapshot redirect isolates reads.

#### Motivation

B's 7h Mode V winner was D #5 XGB+LGBM 72h g=0.997 f=10 conf=90% = +11.21% / 59 trades / WR 66% — a noise-spike lottery, not a real signal. Root cause: 9 candidates tied at APF=14.42 in Mode D's small held-out window (2-6 trades per config), all XGB+LGBM 72h. Top 6 selection deduped on (window, gamma_band, feature_band) but NOT combo, so 6 refine seeds were all XGB+LGBM 72h. RF+LGBM 150h (the family that wins 5h/6h/8h) never reached refine.

User intuition (2026-05-16): "Less choice in V1, more leeway in V2/V3". Wide spacing in Mode D grid → V3 Optuna refine fills the gaps between V1 seeds with continuous search instead of redundantly testing tight neighbors.

#### Fork file

[crypto_trading_system_ed_g_narrow_d.py](crypto_trading_system_ed_g_narrow_d.py) — 8866-line full file copy of production engine with the changes below baked in. Production engine **untouched**.

#### Changes vs B

| Change | B (current prod) | G_narrow_d |
|---|---|---|
| GRID_FEATURES | `[10, 13, 17, 25]` (gaps 3/4/8) | `[10, 15, 20]` (gaps of 5 — wide spacing) |
| GRID_GAMMAS | `[0.999, 0.997, 0.995]` (gaps 0.002) | `[0.999, 0.996]` (gap 0.003 — wide spacing) |
| GRID_WINDOWS | `[72, 100, 150]` | `[72, 100, 150]` (unchanged — windows don't suffer ranking noise) |
| GRID_COMBOS | both | both (unchanged) |
| **V1 total configs** | **72** | **36** |
| N_FEATURES_RANGE upper | 40/80 (hard cap) | 100 (cap dropped) |
| REFINE_TRIALS | 50 | 75 |
| `_diversity_key` | `(window, g_band, f_band)` | `(combo, window, g_band, f_band)` |
| Top-6 selection | APF-rank + region dedup | **Force ≥1 of each combo** in Pass A, then APF dedup |
| V3 Optuna ranges | seed ±20 window, ±5 features, ±0.002 gamma (narrow) | **Absolute**: window `[50, 300]`, features `[4, 60]`, gamma `[0.995, 1.0]` |
| K=5 multi-seed | applied via patcher | **inlined** at module end |

#### Hypothesis being tested

If narrow-spaced V1 + force combo diversity + wide V3 Optuna refine works, the 7h Mode V winner should be a meaningfully better config than B's lottery-spike. Specifically:
- RF+LGBM family should appear in the V3 refine pool
- 7h OVERALL BEST should have positive return at conf ≤ 80%
- ≥80 trades at chosen conf, WR ≥ 75%

Stretch: beat F_optimized's 7h winner (+28.70% conf=90% 77 trades WR=84% — F found this via wider grid but lost 5h/6h/8h to do it).

#### Success criteria → next step

| Result | Decision |
|---|---|
| ≥1 primary signal hits (RF in refine pool, positive at conf≤80%, ≥80 trades, WR≥75%) | **Run full 4-horizon HRST G_narrow_d** on Desktop (~6-8h). If Mode T > B's +89.41%, build production fork and promote when trader flat. |
| All primary signals miss | 7h is genuinely a weak horizon for ETH at this period. Architecture isn't masking alpha. SHELVE G_narrow_d. B (+89.41%) remains the unchallenged winner; ship when trader flat. |

#### Output files

- `models/crypto_ed_grid_ETH_7h_G_NARROW_D.csv` — V1 grid CSV (36 configs)
- `models/crypto_ed_production_noprod.csv` — V_final pick (overwritten; safe with `--no-persist`)
- `logs/ed_v1_<TS>.log` — engine log with V1/V2/V3 output

#### Compare against B (the bar to beat)

B's 7h Mode V SUMMARY winner: **D #5 XGB+LGBM 72h g=0.997 f=10 conf=90% = +11.21% / 59 trades / WR 66%**.

If G_narrow_d's 7h OVERALL BEST is in the same conf=90% lottery range with <80 trades and <75% WR, the architectural change didn't help.

---

### 🟢 P0 — ACTIVE — H_strict_family 7h-only diagnostic (engine fork, 2026-05-16)

## 🚀 LAUNCH COMMAND (Laptop) — run AFTER G_narrow_d Mode V completes

```powershell
# In a Laptop PowerShell terminal, from the engine directory:
$env:V2_DATA_SNAPSHOT = "data\_reliability_hrst_snapshot_desktop_20260515_154801"
$env:RELIABILITY_K = "5"
python crypto_trading_system_ed_h_strict_family.py DV "ETH," 7h --replay 1440 --no-persist --no-data-update --grid-tag H_STRICT_FAMILY
```

**CRITICAL** — same 2 launch-bug guards as G_narrow_d (both already caused live aborts on 2026-05-16):
- **`"ETH,"` with trailing comma + double-quotes.** If you see `ASSET: BTC` at startup → kill, comma missing.
- **`--no-data-update`** REQUIRED. If you see `MACRO & SENTIMENT DATA DOWNLOAD` → kill, flag missing.
- **DO NOT launch while G_narrow_d Mode V is still running on the same laptop** — both will halve via CPU/GPU contention. Wait for G's background task `b48239rqv` to complete first (notification will fire).

**Expected startup banners (post-fix)**:
- `[H_STRICT_FAMILY_SNAPSHOT] pd.read_csv redirected: data/<file> -> _reliability_hrst_snapshot_desktop_20260515_154801/<file>`
- `[H_STRICT_FAMILY_K5] _deku_eval_with_pruning patched (K=5 seeds=[42, 43, 44, 45, 46])`
- `[H_STRICT_FAMILY] _diversity_key changed to (combo, w) — 1 V2 slot per (model_family, window) cluster`
- `ED: Mode D | ETH | 7h | 150 trials` ← ETH only
- `EXHAUSTIVE GRID: ETH 7h — 72 evals` ← B's full grid (NOT G_narrow_d's narrow 36)

**ETA**: ~3-4h (Mode D ~25-30 min like G_narrow_d, Mode V ~2.5-3h). `DV` is one token = runs D AND V chained.

#### Motivation

B's 7h failure forensic (2026-05-16) showed all 6 V2 D-candidate slots filled by **XGB+LGBM 72h variants** (different gammas + features within same combo+window). The engine's `_diversity_key(window, gamma_band, feature_band)` doesn't include combo, so gamma/feature variants of the SAME model family all count as distinct clusters. RF+LGBM 100h (grid rank 10, APF=3.96) and other RF families were filtered out — never reached refine.

User intuition (2026-05-16): "Test the single best of each family — don't waste V2 slots on near-duplicates of one family."

#### Engine fork (file)

[crypto_trading_system_ed_h_strict_family.py](crypto_trading_system_ed_h_strict_family.py) — 8,894-line full file copy of production engine with **three inlined changes** (production engine untouched).

#### Changes vs B

| Aspect | B (current production engine) | H_strict_family |
|---|---|---|
| GRID_COMBOS | [RF+LGBM, XGB+LGBM] | **unchanged** (same B grid) |
| GRID_WINDOWS | [72, 100, 150] | **unchanged** |
| GRID_FEATURES | [10, 13, 17, 25] | **unchanged** (B's spacing, NOT G_narrow_d's narrow [10,15,20]) |
| GRID_GAMMAS | [0.999, 0.997, 0.995] | **unchanged** |
| V1 total configs | 72 | **72** (same as B; G_narrow_d uses 36) |
| `_diversity_key` | `(window, g_band, f_band)` | **`(combo, window)`** — strict, max 6 clusters total |
| V3 Optuna ranges | seed ±20 window, ±5 features, ±0.002 gamma | **unchanged** from B (not G_narrow_d's wide-absolute) |
| K=5 multi-seed | patcher | **inlined** at module end |
| Snapshot redirect | external patcher | **inlined** via `V2_DATA_SNAPSHOT` env var |

**Three changes only**: snapshot redirect + K=5 inlining + dedup. Clean A/B test vs B — any performance difference attributable to dedup change.

#### How H differs from G_narrow_d

| Aspect | G_narrow_d | H_strict_family |
|---|---|---|
| Grid spacing | narrow (36 configs, gaps of 5 features and 0.003 gamma) | B's full (72 configs, no spacing change) |
| `_diversity_key` | `(combo, window, g_band, f_band)` — combo-aware but still allows gamma/feature variants | **`(combo, window)`** — strictest, no variants within (combo, window) |
| V3 Optuna ranges | absolute [50,300] / [4,60] / [0.995,1.0] — wide search far from seeds | B's narrow seed-relative ranges |
| REFINE_TRIALS | 75 | 50 (B's default) |

**H tests dedup change in isolation**; G_narrow_d tests grid spacing + dedup + Optuna range jointly. Distinct experiments testing distinct hypotheses.

#### Hypothesis being tested

If `(combo, window)` dedup is the right architectural fix for B's 7h failure, H's V2 should have **3 RF+LGBM slots out of 6** (1 per RF window: 72h, 100h, 150h) — guaranteed by the dedup. V3 refine of those 3 RF candidates may surface a winner that doesn't exist in B's all-XGB-72h refine pool.

Predicted top 6 V2 inputs (proxy: B's grid-APF top-of-family per (combo, window) cluster):

| V2 slot | Family | Likely pick | grid APF | grid rank |
|---|---|---|---|---|
| 1 | XGB+LGBM 72h | XGB+LGBM 72h γ=0.999 f=13 | 14.42 | 1 |
| 2 | XGB+LGBM 100h | XGB+LGBM 100h γ=0.999 f=10 | 0.99 | 47 |
| 3 | XGB+LGBM 150h | XGB+LGBM 150h γ=0.999 f=25 | 3.04 | 17 |
| 4 | **RF+LGBM 72h** | **RF+LGBM 72h γ=0.997 f=25** | 2.98 | 19 |
| 5 | **RF+LGBM 100h** | **RF+LGBM 100h γ=0.999 f=17** | 3.96 | 10 |
| 6 | **RF+LGBM 150h** | **RF+LGBM 150h γ=0.999 f=25** | 3.64 | 13 |

(Holdout step may re-rank within family; this is grid-APF proxy.)

#### Success criteria → next step

| Result | Decision |
|---|---|
| ≥1 primary signal hit on H (RF+LGBM in V3 winner, positive at conf ≤ 80%, ≥80 trades, WR ≥ 75%) | **Run full 4-horizon HRST H_strict_family** on Desktop (~6-8h). If Mode T > B's +89.41%, build production fork and promote when trader flat. |
| All primary signals miss on H | Combined with G_narrow_d miss: 7h is genuinely a weak horizon for ETH. Architecture isn't masking alpha. SHELVE both experiments. B (+89.41%) remains the unchallenged winner; ship when trader flat. |
| H > G on 7h winner | Dedup criterion is the dominant lever (not grid spacing). Architectural conclusion: `(combo, w)` dedup should be promoted to production engine. |
| G > H on 7h winner | Grid spacing + wide Optuna refine matters more than dedup. Different architectural conclusion. |

#### Output files

- `models/crypto_ed_grid_ETH_7h_H_STRICT_FAMILY.csv` — V1 grid CSV (72 configs)
- `models/crypto_ed_production_noprod.csv` — V_final pick (overwritten by --no-persist; safe with trader)
- `logs/ed_v1_<TS>.log` — engine log with V1/V2/V3 output

#### Compare against B + G_narrow_d (the 3-way)

Once all 3 land, fill the table:

| Variant | 7h Mode V winner | conf | trades | WR | basis |
|---|---|---|---|---|---|
| **B** (yesterday) | XGB+LGBM 72h γ=0.997 f=10 | 90% | 59 | 66% | +11.21% (lottery) |
| **G_narrow_d** (in-flight `b48239rqv`) | TBD | TBD | TBD | TBD | TBD |
| **H_strict_family** (next) | TBD | TBD | TBD | TBD | TBD |

3 experiments testing 3 different architectural angles for the same 7h failure. The matrix tells us which lever (grid spacing, wide refine, dedup) is the dominant cause.

---

### 🔴 P0 — CLOSED 2026-05-16 — DEAD — Variant F_optimized = B + #1+#2+#3+#4

**Verdict**: Mode T total **+63.58%** — **−13.19pp vs production (+76.77%)**, **−25.83pp vs B (+89.41%)**. DEAD by harness threshold (≤ −5pp vs prod) and by the CLAUDE.md updated threshold (≤ +84.41% vs B).

**What costed F vs B**: F's per-horizon Mode V winners changed (metric alignment #1 + expanded grid #2/#3 + BO sampler #4 redirected the selection). F's 7h winner was genuinely better than B's (+28.70% vs +11.21%) — but 5h/6h/8h all lost 14-19pp each. **Crucially, Mode S converged to bull_h == bear_h == 5h** — the per-regime architecture collapsed. F's wider Optuna pulled all four horizons toward similar basins, eliminating the regime-differentiation alpha that B kept via its more APF-diverse per-horizon picks.

**Lesson**: optimizing per-horizon scoring locally (F's metric flip) breaks a non-local property (regime horizon spread) that the production architecture depends on. Consistent with the "don't filter the model" pattern. G_narrow_d (now active P0 above) tries a different architectural lever — wider V1 spacing + force combo diversity — to fix B's 7h failure WITHOUT the per-horizon optimization that broke Mode S.

**Files preserved**: [crypto_trading_system_ed.py:7347](crypto_trading_system_ed.py#L7347), [_idea_patchers/reliability_optuna_objective_align.py](_idea_patchers/reliability_optuna_objective_align.py), [_idea_patchers/reliability_expand_grid.py](_idea_patchers/reliability_expand_grid.py), [_idea_patchers/reliability_bo_exploration.py](_idea_patchers/reliability_bo_exploration.py). Do NOT delete — useful audit trail if a future variant revisits any of these levers.

---

(below: original F_optimized launch instructions, kept verbatim for context — DO NOT RE-RUN)



## 🚀 LAUNCH COMMAND (Desktop) — run this now

```powershell
# In a Desktop PowerShell terminal:
cd G:\engine          # or wherever your desktop engine is
git pull              # picks up commit 959865c (F_optimized + 3 new patchers + harness update)

# Launch F_optimized reusing B's snapshot for fair comparison (same data B saw today):
python tools/run_reliability_hrst.py --variant F_optimized --machine desktop --reuse-snapshot data/_reliability_hrst_snapshot_desktop_20260515_154801
```

**Why `--reuse-snapshot`**: we ran B today (2026-05-15) on the snapshot at `data/_reliability_hrst_snapshot_desktop_20260515_154801/` (56.9 MB, 4 marker files MD5-verified). F_optimized must run against the EXACT same data so the comparison is apples-to-apples. The harness verifies snapshot integrity before starting.

**ETA**: ~7h total (Mode H ~5.5h + Mode R 47min + Mode S 28min + Mode T+G 14min).

**Useful mid-run commands**:
```powershell
python tools/run_reliability_hrst.py --machine desktop --status
python tools/run_reliability_hrst.py --machine desktop --report-only   # re-parses log (fix shipped 2026-05-16)
python tools/run_reliability_hrst.py --machine desktop --reset         # nuclear reset
```

**OPTIONAL parallel A_baseline on Laptop** (clean 3-way A vs B vs F on the same snapshot):
```powershell
python tools/run_reliability_hrst.py --variant A_baseline --machine laptop --reuse-snapshot data/_reliability_hrst_snapshot_desktop_20260515_154801
```

---

**Why this is P0** (user explicit, 2026-05-16): Mode V's Step 1/2/3 pipeline has a structural defect that lets the actual best config get bypassed. B's 6h horizon is the smoking gun: D #6 (RF+LGBM 150h) WON the horizon at +59.39% return — but it was rank 6 by APF and was never sent to Refine. Refine ran on D #1/D #2/D #3 (XGB+LGBM 72h, completely different model family) and could not discover the winner because Optuna's TPE is biased toward exploitation around its starting points. **This is a search-space coverage problem; +5-15pp on the table per horizon when this happens.**

This entry must NOT be closed until the architectural fix is shipped and the F_optimized variant is either promoted to production or conclusively shelved.

#### Implementation status (2026-05-16, commit 959865c pushed to origin/main)

| Component | File | Status |
|---|---|---|
| Patcher #1: objective alignment (OPTUNA_METRIC=ret_wr) | [_idea_patchers/reliability_optuna_objective_align.py](_idea_patchers/reliability_optuna_objective_align.py) | ✅ shipped |
| Patcher #4: BO with CMA-ES + ipop restart | [_idea_patchers/reliability_bo_exploration.py](_idea_patchers/reliability_bo_exploration.py) | ✅ shipped |
| Patcher #2+#3: expanded grid (windows 3→5, features 4→5) | [_idea_patchers/reliability_expand_grid.py](_idea_patchers/reliability_expand_grid.py) | ✅ shipped |
| Harness: F_optimized + A_baseline variants | [tools/run_reliability_hrst.py](tools/run_reliability_hrst.py) | ✅ shipped |
| Harness: --reuse-snapshot flag | same | ✅ shipped |
| Harness: parser fix for shield-disabled "no_t_winner" bug | same | ✅ shipped — verified B's HRST log now correctly parses to +89.41% |

#### The 4 fixes, with measured time/precision trade-offs

| # | Fix | What changes | Time cost | Precision delta | Risk |
|---|---|---|---|---|---|
| **1** | **Metric alignment** | Step 2 picks Refine inputs by `score = ret × WR @ best conf` (from Step 1 data), NOT by Mode D APF | **0 min** | **+5-15pp** on horizons where APF≠score (6h proven case) | None — uses data already computed |
| **2** | **Cluster-aware diversity** | Step 2 picks 3 inputs spread across (combo × window-bucket) clusters, not 3 from the same basin | **0 min** | **+5-10pp** on multi-modal horizons | Minor on uni-modal horizons (wastes 1 of 3 refine slots) |
| **3** | **Successive Halving (Hyperband-style)** | Replace Step 1's "6 candidates × full 1440h backtest" with multi-budget tournament: 30 × 360h → 10 × 720h → 3 × 1440h. Same compute budget, ~5× more candidates explored. | **~+10-15% Step 1 (current 150 min → ~170 min)** in the budget-neutral config; ~+40% Step 1 in the more-thorough config. **The real risk** is short-replay can mis-rank: a config with 4 trades at 96h might be eliminated despite being best at 1440h. Particularly bad for high-conf configs where short replays give zero trades. Mitigation: candidates must "stay top-K" as budget grows. | **+5-15pp expected** OR **−5-15pp possible** if short-replay rank doesn't correlate with full-replay rank | Real precision risk — first thing to validate empirically |
| **4** | **BO with exploration (Optuna sampler swap)** | Replace TPE sampler with GP-based BO (`GPSampler` or `CmaEsSampler`) with higher exploration coefficient. Forces the refine to wander further from the seed point. | **+10-30%** Step 2 (53 min → ~60-70 min) | **+3-7pp** marginal | GP slow at high dim; mixed continuous+categorical works but borderline |

**Group A (free wins, no precision risk): #1 + #2.** Both have zero runtime cost and pure architectural gain. Address the 6h smoking gun directly.

**Group B (time/precision trade-offs): #3 + #4.** Real downsides. #3's short-replay risk needs an empirical check (does short-replay rank correlate with full-replay rank in our engine?). #4 is mostly dominated by #2 — pick one.

**Group C (precision at any cost — NOT in F_optimized): CPCV (López de Prado AFML Ch 12).** 3-5× total HRST time. Use ONCE before production promotion, not on F_optimized routine.

#### Why these specifically (literature anchor)

- **#1**: not really literature, system-design fix. Bailey & López de Prado (2014) "Deflated Sharpe Ratio" framing — use the same metric you select winners by.
- **#2**: Wagner et al. (2014) "Multi-Start Local Search"; Li & Talwalkar (2020) on diversity in hyperparameter search.
- **#3**: Karnin et al. (2013); Jamieson & Talwalkar (2016); Li et al. (2017) "Hyperband"; Falkner et al. (2018) "BOHB".
- **#4**: Snoek et al. (2012) "Practical Bayesian Optimization"; Hutter et al. (2011) "SMAC".

#### Implementation plan (in order)

1. **Snapshot reuse**: F_optimized must run against the EXACT data B saw, so we can fairly compare. Snapshot is preserved at `data/_reliability_hrst_snapshot_desktop_20260515_154801/` (56.9 MB, 4 marker files MD5-verified).
2. **Patchers** (new files under `_idea_patchers/`):
   - `reliability_metric_alignment.py` (#1, ~50 lines)
   - `reliability_cluster_diversity.py` (#2, ~80 lines)
   - `reliability_successive_halving.py` (#3, ~200-300 lines — heaviest)
   - `reliability_bo_exploration.py` (#4, ~20 lines)
3. **Harness extension**: `tools/run_reliability_hrst.py` gets a new variant `F_optimized` that loads `reliability_multi_seed` (base K=5 from B) + the 4 new patchers. Add `--reuse-snapshot SNAP_DIR` flag.
4. **Verdict parser fix**: the existing harness mis-parses Mode T result when shield is disabled (B's "no_t_winner" was actually +89.41%). Fix while we're touching the file.
5. **Smoke test** of each patcher individually (does it import? does it monkey-patch correctly?).
6. **Launch F_optimized HRST** on Desktop reusing B's snapshot: same data → fair comparison.

#### Expected runtime

| Component | B (current) | F_optimized (projected) |
|---|---|---|
| Mode H total | 272.5 min | ~310-340 min (+15% from #3 broader Step 1, +10-30% from #4 slower BO sampler) |
| Mode R | ~47 min | ~47 min (no change) |
| Mode S | ~28 min | ~28 min (no change) |
| Mode T+G | ~14 min | ~14 min (no change) |
| **Total HRST** | **361.5 min (6h 1.5min)** | **~400-440 min (6h 40min — 7h 20min)** |

Well within 40h budget. Adds ~30-80 min for a potential +10-25pp Mode T total improvement.

#### Verdict thresholds (set in advance)

| F_optimized Mode T total | Δ vs B (+89.41%) | Δ vs production (+76.77%) | Decision |
|---|---|---|---|
| ≥ +99.41% | ≥+10pp over B | ≥+22pp over prod | 🟢 **SHIP F_optimized** — promote ALL 4 fixes to production engine fork |
| +94.41% to +99.41% | +5-10pp over B | ≥+17pp over prod | 🟢 **SHIP** — meaningful gain over B |
| +84.41% to +94.41% | ±5pp around B | +7-17pp over prod | 🟡 **MARGINAL** — F_optimized doesn't beat B materially; ship B alone if not already |
| ≤ +84.41% | ≤−5pp vs B | ≤+7pp over prod | 🔴 **DEAD** — the fixes hurt; investigate which one(s) backfired (run F variants without each fix to isolate) |

#### Comparison table at end (3-way, against same snapshot)

| Variant | Patchers | Mode T total | Δ vs A | Δ vs B | Strict win on every horizon? | Verdict |
|---|---|---|---|---|---|---|
| A_baseline (HRST not yet run, may need to be added) | none | TBD | — | — | reference | reference |
| **B_multi_seed** (already run today 2026-05-15) | reliability_multi_seed K=5 | **+89.41%** | reference | reference | bull=6h@65% bear=8h@65% shield=OFF gates rr12/rr24 cd=8h + rr8/rr16 cd=8h | already SHIPS at +12.64pp over prod |
| **F_optimized** (this campaign) | reliability_multi_seed + reliability_metric_alignment + reliability_cluster_diversity + reliability_successive_halving + reliability_bo_exploration | TBD | TBD | TBD | TBD | depends on Mode T total |

Note: A_baseline HRST was NEVER run (Phase 1 only ran Mode DV for A, not full HRST). To make the 3-way comparison clean, we should also run A_baseline HRST against the same snapshot. Adds another ~6h Desktop time.

#### Required user actions

1. Once F_optimized harness is ready, launch on Desktop:
   ```
   python tools/run_reliability_hrst.py --variant F_optimized --machine desktop --reuse-snapshot data/_reliability_hrst_snapshot_desktop_20260515_154801
   ```
2. Optionally launch A_baseline HRST on Laptop in parallel (same snapshot, for clean 3-way):
   ```
   python tools/run_reliability_hrst.py --variant A_baseline --machine laptop --reuse-snapshot data/_reliability_hrst_snapshot_desktop_20260515_154801
   ```
3. After Desktop F_optimized finishes (~7h), read summary `.txt` files and decide per the verdict table.

---

### 🟢 P0 — CLOSED 2026-05-15 — HRST validation of B (multi_seed) and C (no_feature_cap), parallel across machines (2026-05-15)

**Motivation**: Phase 1 5-variant reliability test (finished 2026-05-15 08:28 — see "Closed 2026-05-15" entry below) showed **B_multi_seed +5.11pp** and **C_no_feature_cap +5.23pp** vs A_baseline on Mode V combined_score at ETH 8h. Both passed the pre-set "CLEAR WINNER" threshold (≥3·σ_A). Per CLAUDE.md verdict logic: promote winning variant's patchers → Phase 2 (5,6,7h) → Phase 3 (full HRST).

This step **collapses Phase 2+3 into a single HRST run per variant** (HRST already sweeps Mode DV at 5,6,7,8h internally before Mode S regime selection + Mode T threshold sweep + Mode G rally gates). Mode T total return is directly comparable to the May 6 production HRST baseline (+76.77%).

#### Current state (2026-05-15 14:50 CEST)

- **Desktop B_multi_seed — CRASHED at Mode V phase** after 4h 33min (campaign `20260515_101439`, rc=1). All 4 Mode D grid CSVs `crypto_ed_grid_ETH_<h>h_REL_HRST_B_MULTI_SEED.csv` ARE on disk (5h/6h/7h/8h written 10:38 through 14:44). The Mode V `_test_lgbm_importance → model.fit()` call raised `lightgbm.basic.LightGBMError: Out of Host Memory` at log line 2000. **Actual cause: GPU driver install on Desktop coincided with the crash — not a structural RAM issue from K=5 × `PARALLEL_BACKTESTS=6`.** Earlier framing (RAM peak hypothesis) was wrong. K=5 × 6 workers is fine on Desktop under normal conditions. **Retry pending user re-launch** (see "Retry command" below).
- **Laptop C_no_feature_cap — STILL RUNNING** (campaign `20260515_102358`). All 3 Mode D grids done by Drive sync evidence (5h@10:31, 6h@12:19, 7h@14:05); 8h grid pending. ETA finish ~17:30-19:00 CEST. Local working dir `C:\Users\Alex\algo_trading\engine\` so engine log stays on laptop — only state.json + grid CSVs sync to Drive.

#### Retry command — Desktop B_multi_seed (run AFTER current user gaming session)

PowerShell, one line, run from `G:\Autres ordinateurs\My laptop\engine`:

```powershell
Remove-Item "$env:TEMP\joblib_memmapping_folder_*" -Recurse -Force -ErrorAction SilentlyContinue ; python tools/run_reliability_hrst.py --variant B_multi_seed --machine desktop
```

What this does:
1. Clears stale joblib memmap folders from the crashed run (~hundreds of MB)
2. Launches the harness with the same args. The harness will auto-detect the 4 existing Mode D grid CSVs and pass `--skip` to the engine → jumps straight to Mode V refine → saves ~4.5h
3. `PARALLEL_BACKTESTS=6` (default, reverted 2026-05-15 once GPU-driver crash root cause was identified — earlier `=3` mitigation was a false alarm)
4. `RELIABILITY_K=5` still active via the harness env_overrides (full K=5 denoising preserved)

Look for `All Mode D grids for B_multi_seed already on disk — passing --skip to engine` in the orchestrator log within 5 seconds of launch — that confirms Mode D skip is engaged.

**Expected retry ETA**: ~3-5h (Mode V refine + Mode S + Mode T + Mode G only). Finish ~late evening 2026-05-15.

#### Original parallel launch commands (Laptop C still uses this — do NOT relaunch Laptop)

```
# On Desktop  (USE RETRY COMMAND ABOVE INSTEAD — this is the original that crashed)
python tools/run_reliability_hrst.py --variant B_multi_seed --machine desktop
```

```
# On Laptop (currently running, do not interrupt)
python tools/run_reliability_hrst.py --variant C_no_feature_cap --machine laptop
```

**What each invocation does** ([tools/run_reliability_hrst.py](tools/run_reliability_hrst.py), shipped 2026-05-15):
- Acquires per-machine lock at `output/run_reliability_hrst_{desktop|laptop}.lock`
- Snapshots `data/` → `data/_reliability_hrst_snapshot_{machine}_<CID>/` (~57 MB) — trader can stay active
- Runs `crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist` with the variant's patcher(s) loaded via `_idea_patchers.v2_data_snapshot` + the variant's reliability patcher
- Writes grid CSVs as `crypto_ed_grid_ETH_<h>h_REL_HRST_<VARIANT>.csv`; production CSV/regime config NEVER touched
- Parses the engine log for the final `T winner: ... (total +XX.XX% vs ..., delta +Z.ZZ%)` line
- Writes verdict summary at `output/run_reliability_hrst_{machine}_<TS>_<variant>.txt` + `.csv`

**ETA**:
- C_no_feature_cap (Laptop): **~7-9h** (1.5× standard HRST due to wider feature grid)
- B_multi_seed (Desktop): **~9-12h** (K=5 multiplier on Mode V refine — heaviest variant)
- Modern Standby caveat on Laptop: wall time can balloon to 24h+ if the lid closes; keep the laptop active or run on AC with sleep disabled

**Useful mid-run commands**:
- `python tools/run_reliability_hrst.py --machine desktop --status` (or `--machine laptop`)
- `python tools/run_reliability_hrst.py --machine {desktop|laptop} --report-only` — rebuild summary from existing log
- `python tools/run_reliability_hrst.py --machine {desktop|laptop} --reset` — wipe state + snapshot + lock

**Verdict logic (set in advance, not post-hoc)**:
- Mode T total ≥ **+81.77%** (≥ prod +76.77% + 5pp) → **SHIP**: promote that variant's patchers to production engine (build a `crypto_trading_system_ed_<variant>.py` fork or merge inline)
- Mode T total within **±5pp** of production → **MARGINAL**: 8h Phase 1 win didn't fully transfer to the regime-joint sweep; consider single-horizon application or shelve
- Mode T total ≤ **+71.77%** (≤ prod − 5pp) → **DEAD**: variant doesn't generalize beyond Phase 1 8h win; close the file

**Why this matters**: First clear "scoring overlay family" winners since CDaR/CVaR were tested in April. B and C win via *different* mechanisms (denoising vs wider feature search) and don't stack cleanly (D = B+C only +2.93pp, E = full stack only +0.19pp) — that asymmetry means running them separately tests them on their own merits rather than as a combined patch.

**Pre-flight checks**:
- Desktop venv has `xgboost` (installed 2026-05-14 22:30) — required for B_multi_seed patcher
- Laptop venv does NOT need xgboost — C_no_feature_cap doesn't use the multi-seed XGB factory
- Trader currently active on Desktop (ETH 6h+8h, restarted 2026-05-14 20:37) — snapshot isolation means HRST and trader can coexist; do not stop trader

**Required action after both finish**:
1. Read both summary `.txt` files; record Mode T totals + verdicts
2. If ≥1 SHIPs: build production fork — copy `crypto_trading_system_ed.py` → `crypto_trading_system_ed_<winner>.py`, inline the patcher's changes, run final HRST without `--no-persist` to write production CSV/regime config
3. If both SHIP: do them in sequence (no combined patch — Phase 1 showed they don't stack), keep the higher Δ
4. If both MARGINAL or DEAD: scoring-overlay family conclusively closed for this generation; pivot to execution-gap research (~17pp sim-vs-live alpha gap)

---

### 🔵 P0 — CLOSED 2026-05-15 — 5-variant reliability test (launched 2026-05-14 20:47:36, fix shipped 22:00 mid-flight)

**What it tests**: three root causes that make crypto Mode D/V results unreliable, identified 2026-05-14:
1. **Measurement noise dominates signal** — `tools/feature_stability_test.py` (shipped 2026-05-14 ~15:30) measured σ=5.82pp on label-noise alone for prod ETH 8h. Baseline is ~2σ BELOW the noise mean — production is on the unlucky side of its own noise distribution.
2. **n_features hard cap traps the optimizer** — V2's 5 completed modes ALL produced bit-identical Mode V winners at 5h/6h/7h. Only 8h showed variation. Adversarial mode beat baseline by +5.18pp avg, right AT the σ noise floor.
3. **Scoring rule rewards luck** — `return × WR` is jumpy at typical N=30-80 trade counts; no Deflated-Sharpe / PBO multiple-testing correction.

**5 variants × Mode DV ETH 8h + per-variant stability test (--replay 336)**:
| Tag | Patchers | Tests |
|---|---|---|
| `A_baseline` | (none) | reference |
| `B_multi_seed` | reliability_multi_seed (K=5 median scoring) | root cause #1 |
| `C_no_feature_cap` | reliability_no_feature_cap (GRID_FEATURES→[10,17,25,40,60,80], N_FEATURES_RANGE upper→150) | root cause #2 |
| `D_multi_seed_plus_cap` | B + C | #1+#2 stack |
| `E_full_fix` | B + C + reliability_dsr_scoring (OPTUNA_METRIC='rpf_sqrt') | full stack |

Phase 1 scope is **8h-only** because V2 proved 5/6/7h are bump-locked across all importance methods — testing them would burn 4× compute for known-noise results.

**Launch command** (resumable, V2-style snapshot isolation, trader stays active):
```
python tools/run_reliability_test.py
```

**Files (commit f2ca15a + fix 77e78cb)**:
- [tools/run_reliability_test.py](tools/run_reliability_test.py) — orchestrator (5 variants, snapshot, resumable state)
- [_idea_patchers/reliability_multi_seed.py](_idea_patchers/reliability_multi_seed.py) — K=5 median wrapper for `_deku_eval_with_pruning`
- [_idea_patchers/reliability_no_feature_cap.py](_idea_patchers/reliability_no_feature_cap.py) — drops n_features ceiling
- [_idea_patchers/reliability_dsr_scoring.py](_idea_patchers/reliability_dsr_scoring.py) — flips OPTUNA_METRIC to `rpf_sqrt`
- [crypto_trading_system_ed_robust.py](crypto_trading_system_ed_robust.py) — first-pass K=5 fork (audit-trail, superseded by harness)
- [tools/feature_stability_test.py](tools/feature_stability_test.py) — updated: `--csv/--asset/--horizon` args + `STAB_REPLAY` env

#### 🚨 ETH-parse bug discovered LIVE 2026-05-14 ~22:00 CEST

**Symptom**: 1h 15min into the Desktop run, BTC grid CSV `crypto_ed_grid_BTC_8h_REL_A_BASELINE.csv` was found alongside the expected ETH one. The harness was running ALL 9 ASSETS per variant (BTC, ETH, XRP, SOL, LINK, BNB, SMI, DAX, CAC40), not ETH-only.

**Root cause**: same engine CLI parser bug at [crypto_trading_system_ed.py:7347](crypto_trading_system_ed.py#L7347) that bit the laptop's `crypto_trading_system_ed_robust.py` test earlier — `'ETH'.lower().endswith('h')` is True, so the engine's positional-arg parser treats `'ETH'` as a horizon, fails the isdigit body, consumes the elif WITHOUT setting horizons or assets, and `assets_list` defaults to `list(ASSETS.keys())` = all 9. The robust wrapper had a workaround for this; the harness did NOT (my mistake — should have been ported during the harness build, smoke-tested with `--status` doesn't exercise subprocess argv construction).

**Impact prevented**: Variant A alone would have taken **3-5h** instead of ~1.5h. Full Phase 1 would have ballooned to **~50h** vs the planned ~9h — over the 40h budget.

**Fix (commit 77e78cb)**: in `run_mode_dv()`, append a trailing comma to `ASSET` when it ends in `'h'`. The engine parser then splits on comma, filters by ASSETS membership, and correctly identifies `['ETH']`.

**Required user action after the fix lands on the Desktop**:
1. Ctrl+C the running orchestrator (PID 18388 holds the lock at 22:00 CEST)
2. `cd` to desktop engine dir, `git pull` (picks up commit 77e78cb)
3. `del models\crypto_ed_grid_*_REL_A_BASELINE.csv` — optional, removes buggy partial artifacts
4. `python tools/run_reliability_test.py` — restart; state.json `completed: {}` so auto-resume kicks in cleanly from variant A

**Sanity check after restart**: first grid CSV written should be `crypto_ed_grid_ETH_8h_REL_A_BASELINE.csv` (ETH only). If `crypto_ed_grid_BTC_8h_REL_A_BASELINE.csv` appears, the fix didn't propagate — re-pull and try again.

#### Architecture (V2-style)
- Snapshot at `data/_reliability_snapshot_<CID>/` — pd.read_csv redirected via [_idea_patchers/v2_data_snapshot.py](_idea_patchers/v2_data_snapshot.py). Trader stays active.
- All writes to `*_reliability_<variant>.csv` or `*_noprod.csv` — production CSV/regime config never touched.
- Resumable: PID lock + state.json. Failed (rc!=0) variants NOT marked complete; re-run picks up where it left off.

#### Useful commands during the run
```
python tools/run_reliability_test.py --status        # show DV ✓ / STAB ✓ per variant
python tools/run_reliability_test.py --report-only   # rebuild verdict from existing snapshots
python tools/run_reliability_test.py --variants A_baseline,B_multi_seed  # subset
python tools/run_reliability_test.py --skip-stability  # Mode DV only (no σ measurement)
python tools/run_reliability_test.py --reset         # wipe state + snapshot + per-variant CSVs
```

Quick state probe (no Python needed):
```
cat output/run_reliability_test_state.json | findstr completed
```

#### Expected runtime on Desktop (post-fix, ETH-only)

Based on observed timing of Variant A's ETH 8h Mode D taking ~44 min on desktop (vs my ~30 min estimate):

| Variant | Mode DV | Stability | Per-variant | Cumulative ETA from 22:30 restart |
|---|---|---|---|---|
| A_baseline | ~75 min | ~25 min | ~100 min | **~00:10 CET** |
| B_multi_seed (K=5 ×~4.5 multiplier on Mode V refine) | ~250 min | ~25 min | ~275 min | **~04:45 CET** |
| C_no_feature_cap (no K=5, ~50% more grid evals) | ~110 min | ~25 min | ~135 min | **~07:00 CET** |
| D_multi_seed_plus_cap | ~300 min | ~25 min | ~325 min | **~12:25 CET tomorrow** |
| E_full_fix | ~300 min | ~25 min | ~325 min | **~17:50 CET tomorrow** |

Within 40h budget. The single most informative datapoint is **Variant B at ~04:45 CET** (does K=5 median actually drop σ?). Variants C/D/E refine the answer.

#### Verdict logic (set in advance, not post-hoc)
- **CLEAR WINNER**: ≥1 variant drops σ < 2pp AND beats baseline by ≥3·σ_A → promote winner's patchers to a production engine fork → Phase 2 (expand to 5,6,7h) → Phase 3 (full HRST validation).
- **σ DROPPED BUT NO STRICT WIN**: root cause #1 fix is real but combined effect doesn't beat baseline → ship denoised engine anyway (retain current alpha + lower variance), pivot to execution-gap research (~17pp gap to live).
- **σ DID NOT DROP**: root cause #1 hypothesis was wrong → pivot to execution-gap OR re-examine patcher implementations.

#### Today's research timeline (chronological, 2026-05-14)
1. **00:19** — V2 launched on desktop (different test — see CRASHED entry below)
2. **11:35** — V2 crashed during resume attempt (joblib KeyboardInterrupt). 5/7 modes complete. Adversarial = +5.18pp avg, right at σ noise floor → motivated the σ measurement work.
3. **~15:30** — User shipped [tools/feature_stability_test.py](tools/feature_stability_test.py) (commit `55cc9b6`). Ran on laptop → **σ=5.82pp UNSTABLE verdict** on prod ETH 8h. Baseline +23.40% is ~2σ BELOW the +34.32% mean of all 11 perturbations.
4. **~17:00** — Built [crypto_trading_system_ed_robust.py](crypto_trading_system_ed_robust.py), a first-pass K=5 median fork. Mode D ETH 8h --replay 1440 on laptop (42 min). Winner: RF+LGBM w=150 g=0.997 f=13 (4 fewer features than prod; dropped `hour_cos` which permute-test flagged as a noise-bump feature, kept `pysr_1`/`pysr_2` which were load-bearing).
5. **~19:50** — Started stability test against the robust winner; killed at 17 min (projected 3h on full --replay 1440 — too slow as a per-variant check inside a 5-variant harness).
6. **~20:00** — Designed the 5-variant harness with --replay 336 stability test (~25 min instead of ~3h). Built `tools/run_reliability_test.py` + 3 reliability patchers. Commit `f2ca15a` push.
7. **20:47:36** — User launched harness on Desktop.
8. **~22:00** — ETH-parse bug discovered live (BTC grid CSV appeared under variant A). Fix shipped as commit `77e78cb`. User Ctrl+C + git pull + restart pending.
9. **23:33:47 (2026-05-14) → 08:28 (2026-05-15)** — Fresh campaign `20260514_233347` ran cleanly to completion after `feature_stability_test.py` was patched to accept `--csv/--asset/--horizon` args + read `STAB_REPLAY` env (commit not tagged; in-place edit). xgboost installed in Desktop venv.

#### Phase 1 final result (campaign `20260514_233347`)

| Variant | combined_score | Δcs vs A | return | feat | gamma | σ (336h replay) | verdict |
|---|---|---|---|---|---|---|---|
| **C_no_feature_cap** | **49.07** | **+5.23** | **+63.79%** | 10 | 0.997 | 1.50 | STABLE |
| **B_multi_seed** | **48.95** | **+5.11** | +56.78% | 13 | 0.9967 | **1.19** | STABLE |
| D_multi_seed_plus_cap | 46.77 | +2.93 | +54.56% | 13 | 0.9967 | 1.41 | STABLE |
| E_full_fix | 44.03 | +0.19 | +56.88% | 10 | 0.995 | 1.53 | STABLE |
| A_baseline | 43.84 | +0.00 | +59.05% | 10 | 0.999 | 1.35 | STABLE |

Per pre-set verdict logic (≥1 variant drops σ<2pp AND beats baseline by ≥3·σ_A=4.05pp): **B and C both qualify as CLEAR WINNERS**. Triggers Phase 2/3 HRST (see top P0).

**Surprises worth remembering**:
- All 5 variants STABLE (σ 1.19-1.53pp). The pre-Phase-1 σ=5.82pp UNSTABLE on old production cfg did NOT replicate on the new winners — likely an artifact of the different cfg+replay combination, not a true property of the engine.
- D (B+C combined) +2.93pp < either alone — patchers don't stack. K=5 wants w=167 13f, no_cap wants w=150 10f; combined optimizer can't satisfy both.
- E (B+C+rpf_sqrt scoring) +0.19pp — `rpf_sqrt` metric collapses the gain. Tells us replacing the scoring metric isn't a win on top of the other fixes.
- 4 of 5 permute trials returned `no_signals` in the stability test — destroying a top-importance feature (pysr_1/2/4, oc_mvrv_chg1d) drops confidence below threshold → zero trades. Reframe: this is a positive sign — the model genuinely depends on those features, not on a lucky bump.

**Files written (audit trail)**:
- [output/run_reliability_test_20260514_233347.csv](output/run_reliability_test_20260514_233347.csv) — variant comparison
- [output/run_reliability_test_20260514_233347_summary.txt](output/run_reliability_test_20260514_233347_summary.txt) — verdict + table
- [models/crypto_ed_production_reliability_*.csv](models/) — per-variant winners (5 files)
- [logs/run_reliability_test_20260514_233347*.log](logs/) — orchestrator + per-variant Mode DV + per-variant stability

#### V2 status
Superseded — see CRASHED entry directly below.

---

### 🔴 P0 — CRASHED + SUPERSEDED — Engine mode comparison test V2 (launched 2026-05-14 00:19, crashed 11:35)

**Status**: launched on Desktop with snapshot isolation (trader-coexistence mode). ETA ~37h, finishes ~Friday afternoon. Resumable.

**Launch / resume command (same)**:
```
python tools/test_engine_modes_v2.py
```

**What it does**: full Mode DV pipeline across 7 literature-grounded importance/regularization modes. Compares Mode V's `combined_score` per horizon (production selection metric). Trader stays active — V2 reads from a private data snapshot, completely isolated from live data writes.

**V2 modes tested**:
1. `baseline` — reference (current production engine)
2. `interventional_shap` — Janzing, Minorics, Bloebaum (2020) arXiv 1910.13413 — causally-correct TreeSHAP with background
3. `loco_importance` — Hooker, Mentch, Zhou (2019) arXiv 1905.03151 — drop-column importance on top-40 (2-stage tractability)
4. `purged_cv_split_count` — López de Prado AFML Ch 7 — 5-fold purged CV avg via Borda rank
5. `lasso_prefilter` — Tibshirani (1996) — L1 logistic pre-filter → LGBM (true feature pruning)
6. `leaf_weight_l1_reg` — LightGBM `reg_alpha=1.0` — leaf-weight L1 (NOT feature pruning — properly framed)
7. `adversarial` — Strobl, Boulesteix, Zeileis, Hothorn (2007) BMC Bioinformatics — null-feature diagnostic

**🛡 Trader-coexistence architecture (added 2026-05-14)**:
- At campaign start: V2 snapshots `data/` → `data/_v2_snapshot_<CID>/`
- Each mode subprocess loads [_idea_patchers/v2_data_snapshot.py](_idea_patchers/v2_data_snapshot.py) which monkey-patches `pd.read_csv` to redirect `data/*` reads → `data/_v2_snapshot_<CID>/*`
- Trader continues reading/writing live `data/` files unaffected
- V2 reads frozen snapshot; no data drift possible across modes
- Drift check repurposed: now verifies snapshot integrity (should never change since V2 owns it)
- **TRADER CAN STAY ACTIVE during the entire 37h test** — no collision, no `--no-persist` conflicts (production CSV untouched by V2)

**V2 over V1 improvements**:
- Snapshot isolation (above) — trader coexistence
- Failed runs (rc != 0) NOT marked complete (v1 bug — would skip on retry)
- Methodologically-correct mode labels (e.g., `leaf_weight_l1_reg` corrects v1's misnamed `l1_alpha` "native feature pruning" claim)
- Post-hoc PBO + DSR scoring stub at end (Bailey & López de Prado 2014, 2017)

**Useful commands** (run anytime during execution):
- `python tools/test_engine_modes_v2.py --status` — show which modes done (✓) / pending (·)
- `python tools/test_engine_modes_v2.py --report-only` — rebuild verdict from existing snapshots (partial OK)
- `python tools/test_engine_modes_v2.py --reset` — wipe state + per-mode CSVs + data snapshot dir, start fresh

**Verification of trader isolation** — first subprocess log (e.g., `logs/test_engine_modes_v2_<TS>_baseline.log`) should show:
```
[V2_SNAPSHOT] pd.read_csv redirected: data/<file> -> _v2_snapshot_<CID>/<file>
```
If you see `[V2_SNAPSHOT] V2_DATA_SNAPSHOT not set`, the env var didn't wire — isolation OFF, drift possible. Stop and `--reset`.

**Outputs to check after completion**:
- `output/test_engine_modes_v2_<TS>_summary.txt` — verdict + rankings
- `output/test_engine_modes_v2_<TS>.csv` — comparison table
- `models/crypto_ed_production_mode_v2_<NAME>.csv` — per-mode Mode V winners (preserved)

**Decision rule on completion**:
- ≥1 mode strictly better than baseline on all 4 horizons AND PBO<50% AND DSR significant → ship it (after HRST validation)
- No strict winner → conclusively closes within-engine importance/regularization fixes → pivot to **Lever 1 (per-horizon feature pools)** infrastructure work

**Why this matters**: today's research (2026-05-13) tested 4 within-engine fixes (NOCAP, PERM, HYBRID, LGBM-only). All traded problems — no strict improvement. This 7-mode V2 test is the literature-grounded conclusive sweep. After this, we know whether ANY within-engine importance/regularization method can improve, or whether per-horizon work is the only remaining path.

**Files (all shipped 2026-05-13/14)**:
- [tools/test_engine_modes_v2.py](tools/test_engine_modes_v2.py) — orchestrator (snapshot isolation added 2026-05-14)
- [_idea_patchers/v2_data_snapshot.py](_idea_patchers/v2_data_snapshot.py) — `pd.read_csv` redirect (NEW 2026-05-14)
- [_idea_patchers/interventional_shap.py](_idea_patchers/interventional_shap.py) — mode 2
- [_idea_patchers/loco_importance.py](_idea_patchers/loco_importance.py) — mode 3
- [_idea_patchers/purged_cv_split_count.py](_idea_patchers/purged_cv_split_count.py) — mode 4
- [_idea_patchers/lasso_prefilter.py](_idea_patchers/lasso_prefilter.py) — mode 5
- [_idea_patchers/leaf_weight_l1_reg.py](_idea_patchers/leaf_weight_l1_reg.py) — mode 6
- [_idea_patchers/adversarial.py](_idea_patchers/adversarial.py) — mode 7

**Cleanup post-completion**: snapshot dir `data/_v2_snapshot_<CID>/` is ~100-500MB. `--reset` removes it after results are confirmed; or `rm -rf data/_v2_snapshot_*` manually.

**V1 superseded**: [tools/test_engine_modes.py](tools/test_engine_modes.py) is kept on disk for reference but V2 is what should be launched. V1 had loose literature citations and a state-tracking bug.

---

### 🟡 P0b — RUN ON LAPTOP — Feature stability diagnostic (2026-05-14)

**Run on Laptop in PARALLEL with the V2 sweep on Desktop**:
```
python tools/feature_stability_test.py
```

ETA ~15-30 min (11 subprocesses, single-config backtest each — not full Mode D).

**What it answers**: a different, more fundamental question than V2. V2 asks "which feature-importance ranker picks the best Mode DV candidate?". P0b asks: **"is the D/V pipeline feature-stable at all, regardless of which ranker we use?"**

Methodological premise: in a well-behaved pipeline, adding an irrelevant feature should produce ~zero change in backtest return (model ignores noise). If σ(return) across feature perturbations is large, the pipeline is over-fitting to feature noise — meaning *every* feature-related decision (which ranker, which features to add, which to drop) is dominated by noise, not signal.

**Test design**:
- Fixed config: production 8h ETH winner (RF+LGBM, w=150, gamma=0.995, 17 features) from `models/crypto_ed_production.csv`
- 11 trials on same 1440h replay at conf=65 (bear regime threshold):
  - 1 baseline (no perturbation, sanity check)
  - 5 trials adding 1 random-noise feature (different seeds 42-46) — model should ignore
  - 5 trials permuting one existing feature (pysr_1, pysr_2, deriv_basis_chg1d, hour_cos, logret_8h) — destroys feature info
- Measures σ(return_pct) across all 11 trials

**Verdict thresholds** (rule-of-thumb):
- σ < 2pp → STABLE: pipeline is feature-robust; D/V methodology is sound
- 2 ≤ σ < 5pp → MARGINAL: moderate feature sensitivity; consider fixes A/B/C below
- σ ≥ 5pp → UNSTABLE: pipeline over-fits to feature noise; structural fix required before any feature-related decision can be trusted

**If UNSTABLE, structural fixes (in priority order)**:
- **Fix A**: Drop the n_features hard cap in [crypto_trading_system_ed.py:431-435](crypto_trading_system_ed.py#L431) (`N_FEATURES_RANGE = {4: (4, 40), 8: (4, 80)}`). The cap forces feature DISPLACEMENT — adding feature X kicks out feature Y. Without the cap, LGBM naturally ignores noise features.
- **Fix B**: Bootstrap-aggregate the importance ranking (the existing `purged_cv_split_count` patcher is half of this — does folds, not bootstraps). Stabilizes ranking at small samples.
- **Fix C**: Replace `combined_score` with a sample-size-robust metric (rolling Sharpe over 30-day windows instead of cumulative win-rate × return). Reduces Optuna's overfitting to thin-sample lottery results.

**Why this is parallelizable with V2**:
- Different machine (Laptop, not Desktop)
- Reads from live `data/` (not the V2 snapshot dir) — so it captures CURRENT data state, not 2026-05-14 frozen snapshot. This is intentional: stability is a property of the pipeline + data, and we want to know about TODAY's pipeline.
- Doesn't touch noprod.csv or any V2-relevant file
- Trader can stay active (this test only does in-process backtest, no engine main() invocation)

**Why both tests are useful (don't skip P0b in favor of just V2)**:
- V2 result tells us if any specific ranker wins. P0b result tells us whether ANY ranker decision matters at all.
- If P0b returns UNSTABLE, the V2 results are also untrustworthy regardless of which mode "wins" — the whole methodology is noise-amplifying.
- If P0b returns STABLE, V2's null result (likely) is meaningful: the methodology is sound, the engine is feature-immune, and no ranker change helps.
- P0b finishes in 15-30 min vs V2's 37h. It can inform whether V2 is worth waiting for or whether the rerun's verdict is moot.

**Optional flags**:
- `--conf 75` use bull threshold instead of bear (75 vs 65)
- `--trials 3` reduce noise+permute trials to 3 each (faster ~10 min)

**File**: [tools/feature_stability_test.py](tools/feature_stability_test.py) — single self-contained orchestrator (no separate patcher files needed; inline subprocess code applies perturbations to `_build_features` per trial).

---

### 🟡 P0c / P0d / P0e — Structural fixes CONDITIONAL on P0b verdict (2026-05-14)

Only act on these IF P0b returns MARGINAL (σ 2-5pp) or UNSTABLE (σ ≥ 5pp). If P0b returns STABLE (σ < 2pp), all three are unnecessary — close them.

**P0c — Fix A: Drop the n_features hard cap** *(highest impact, easiest to test)*
- Patch [crypto_trading_system_ed.py:431-435](crypto_trading_system_ed.py#L431) — currently `N_FEATURES_RANGE = {HORIZON_SHORT: (4, 40), HORIZON_LONG: (4, 80)}`. Either remove the cap entirely (let LGBM pick from all features) or widen to (4, 200) so the cap doesn't bind.
- Why: the hard cap forces feature DISPLACEMENT — adding feature X kicks out feature Y. Without cap, LGBM naturally weights real features higher and noise features get near-zero split-count.
- After patch: re-run `python tools/feature_stability_test.py`. If σ drops below 2pp, Fix A is the root cause — keep the patch, ship it.
- ETA: 5 min patch + 30 min re-test = ~45 min total.

**P0d — Fix B: Bootstrap-aggregate the importance ranking**
- The existing [_idea_patchers/purged_cv_split_count.py](_idea_patchers/purged_cv_split_count.py) does folds, not bootstraps. Extend it to do K=10 bootstrap resamples (with replacement) instead of (or in addition to) the 5 purged folds. Keep features in top-N rank on ≥60% of bootstraps. This stabilizes feature selection at small samples.
- Use as a Mode D patcher (same pattern as V2 modes); replaces the default `_test_lgbm_importance`.
- ETA: ~1h to write + 30 min stability re-test.
- Only worth doing if Fix A alone doesn't fix the instability.

**P0e — Fix C: Replace combined_score with sample-size-robust metric**
- Current `combined_score` is `return_pct × accuracy / some_normalization` — at 5-10 trades, both factors are highly variable. Replace with rolling Sharpe over 30-day windows (~20 windows in 1440h replay = much higher effective sample).
- Engine wide change: touches Mode D scoring, Mode V scoring, the production CSV column. ~half-day engineering.
- Only worth doing if Fixes A + B together don't stabilize.

**Decision chain post-P0b**:
1. P0b STABLE → close P0c/P0d/P0e. Methodology is fine. V2 verdict (whenever it finishes) is meaningful as-is.
2. P0b MARGINAL → do P0c, re-test. If still marginal, do P0d. Skip P0e unless needed.
3. P0b UNSTABLE → do P0c, re-test. If still unstable, P0d. If still unstable, P0e (substantial engineering — pause and re-evaluate first).

---


### 🔄 RE-EVALUATION 2026-05-10 — shelved ideas RE-OPENED

User pushback on the 2026-05-09 C05/C06 verdict triggered a thorough re-analysis. The "scoring-overlay family DEAD" pattern obscured a real mechanical issue: **Mode S regime sweep is the bottleneck, not the metric**. Per-horizon scoring wins (CVaR 6h Refined #1 +50.88%, CDaR 8h Refined #1 +67.03%) ARE real but get destroyed when Mode S picks a different detector than the one those wins were optimized under.

**Reclassified from DEAD/SHELVED → TWEAKABLE**:
- **C05 CVaR / C06 Sortino** — per-horizon wins material; HRST loss attributable to detector swap during Mode S
- **C13 CDaR** — same pattern; single-horizon-only follow-up was already noted in CLAUDE.md but never executed
- **C14 triple-barrier exit overlay** — 60d gain real (+10.48pp); 90d washed; needs vol-conditional gating
- **C11 VPIN entry filter** — best variant +3.83pp at hourly; literature uses 1-min, never tested at 5-min sub-loop
- **C16 disaster brake** — already actionable: just confirmed +1.96pp / 1 fire on 90d cache (ship now)

**Confirmed GENUINELY DEAD** (no parameter tweak helps, root cause is concept/architecture):
- C12 stability filter (LGBM bagging already does this internally)
- C18 calibration / C45 conformal (architectural mismatch with binary engine)
- C09/C10/C24/C02 exit/entry filters (tested with sweeps; model's exit timing IS the alpha)
- C17 fracdiff / C19 turbulence (concept inverted on ETH bull regime)
- C20 triple-barrier label (positive-class starvation)
- C21 asymmetric loss / C22 Kelly (architectural mismatch with high-conf binary engine)
- The 7 mid-effort retest ideas from 2026-05-10 batch (C03/C12/C23/C29ab/C31/C48) — confirmed DEAD on fixed harness with clean baseline. Net: 0 PASS, 1 MARGINAL+ (C29 HAR-RV +0.63 driven by single-horizon 5h win).

### 🚀 Priority queue (post re-evaluation, updated 2026-05-10 evening)

| # | Action | Effort | Expected payoff | Status |
|---|---|---|---|---|
| **P1** | **Enable C16 -5% disaster brake** in `regime_config_ed.json` (`ETH.disaster_brake_pct: 5`) | 5 min | +2pp / free downside insurance | ⏸ ON HOLD (won't fire in current regime; user paused 2026-05-10) |
| **P2** | **Locked-detector HRST runner** — Mode S restricted to current detector (`tsmom_672h`); only sweep horizons + confs + scoring. | ~6h Desktop | tested CVaR, expected +3-8pp | ❌ DEAD 2026-05-10 (-8.88pp; scoring-overlay family conclusively closed 5-for-5) |
| **P3** ⭐ | **CPCV HRST** — `python tools/run_cpcv_hrst_resumable.py`. Engine fork [crypto_trading_system_ed_cpcv.py](crypto_trading_system_ed_cpcv.py) implements Design Option A (adaptive gamma per fold, López de Prado AFML Ch 12). Re-validates Mode D top-6 candidates via 15-path CPCV with PBO scoring. Production untouched. | ~5-7h Desktop | (1) re-rank may improve Mode T REF; (2) PBO diagnostics show overfitting risk regardless of headline | ⭐ **ACTIVE — built 2026-05-10, ready to run on Desktop** |
| **P4** | **C14 vol-conditional triple-barrier** retest — only use barriers when realized vol > p70 | ~2.5h | +3-5pp if conditional logic right | Standalone overlay sim |
| **P5** | **C11 VPIN at 5-min cadence** — sub-loop in trader; literature framing | ~1 day coding | +3-7pp if literature transfers | Real engineering |
| **P6** | **C15 meta-labeling on SOL/BTC** — door open per CLAUDE.md; primary weak there | ~6h | secondary asset enablement |  |
| ~~P7~~ | ~~Mode S diagnostic~~ | n/a | n/a | ❌ Made obsolete 2026-05-10 by P2 result. Locked-detector test PROVED Mode S is NOT the bottleneck (CVaR DEAD even with detector locked). Skip. |

**P3 is the highest-stakes test of the day.** CPCV is the only research direction that could (a) revive promotion of an alternate-scoring engine, OR (b) provide PBO-based confidence intervals on current production configs. Either outcome is informative.

### 📚 Why P3 (CPCV) matters even if it doesn't ship

López de Prado's CPCV (AFML Ch 12) is the gold-standard backtesting methodology for ML financial models. The Deku V1.4 attempt in March 2026 ([archive/crypto_trading_system_deku_v1_4_cpcv_gamma1_failed.py](archive/crypto_trading_system_deku_v1_4_cpcv_gamma1_failed.py)) FAILED because it forced `gamma=1.0` to make CPCV work. Design Option A fixes this: each CPCV fold's walk-forward uses the candidate's NATURAL gamma, preserving recency bias within each fold.

Two outcomes possible:
- **Headline win**: CPCV-driven candidate selection produces a Mode T REF >= +5pp over current production (+76.77%). Promote.
- **Headline neutral, diagnostic win**: even if Mode T REF is unchanged, the new `cpcv_pbo` column in grid CSVs reveals which Mode D winners are overfit (PBO > 50%) vs robust (PBO < 30%). Useful intel for assessing the engine's stability.

**Run command**:
```
python tools/run_cpcv_hrst_resumable.py
```

Single-instance lock ensures no duplicate-launch deadlock. State file at `output/cpcv_hrst_state_1440h.json`. Outputs to `_cpcv` suffixed files only.

**Why this re-evaluation matters**: I had been pattern-matching to "this week's 0-PASS streak" and shelving anything that didn't pass cleanly. But looking at the underlying numbers, several "FAIL" verdicts had real positive signal at the per-horizon level that got destroyed by aggregation. The bottleneck isn't the feature pipeline or the scoring metric — it's the regime selection logic. P7 (Mode S diagnostic) is the most important unbuilt tool.

### ✅ Closed 2026-05-11 — P3 CPCV HRST = headline-neutral (matched current method)

**Memorialized retroactively 2026-05-19** after audit revealed the run completed but verdict was never captured in the closed sections. State file `output/cpcv_hrst_state_1440h.json` shows all 11 phases done (1.D × 4 horizons, 2.V × 4 horizons, 3.R, 4.S, 5.T) with last write 2026-05-11 20:44. User-confirmed verdict (2026-05-19): **CPCV-driven candidate selection matched current method — no re-rank, no Mode T REF improvement vs APF-Optuna baseline.**

**What this means**:
- **Headline win path** (Mode T REF ≥ +5pp over production) — DID NOT FIRE.
- **Diagnostic-only outcome** — PBO data in the `_cpcv` suffixed grid CSVs is available but didn't change which configs win on rolling-horizon scoring.
- López de Prado AFML Ch 12 CPCV methodology validates the engine's current 3-fold rolling holdout for THIS dataset. The gold-standard backtester says "your method is fine."

**Status**: kept in TODO.md as **low-priority diagnostic** — available for re-run when the user wants PBO-overfit-risk intel on a new candidate set (e.g. after a major engine change like H75), but not actively planned. Periodic re-run would catch if a future engine change introduces overfit configs that CPCV would flag but our current 3-fold holdout misses.

**Files preserved for future use**:
- [crypto_trading_system_ed_cpcv.py](crypto_trading_system_ed_cpcv.py) (422 KB engine fork, mtime 2026-05-10) — Design Option A (adaptive gamma per fold)
- [tools/run_cpcv_hrst_resumable.py](tools/run_cpcv_hrst_resumable.py) (resumable launcher with single-instance lock)
- `output/cpcv_hrst_state_1440h.json` — state file from the May 11 run
- `logs/cpcv_hrst_*.log` × 11 files — per-phase logs

**Re-run command** (when needed):
```
python tools/run_cpcv_hrst_resumable.py
```

### ✅ Closed 2026-05-10 (overnight mid-effort batch — fixed harness + clean baseline)

7-idea batch via `tools/test_idea_batch_C03_C12_C23_C29_C31_C48.py` ran 2026-05-10 00:14 → 04:31 (4h17m). Verdicts computed directly from grid CSVs (auto-summary parser hit a Unicode crash but Mode D writes were clean):

| CID | Idea | Avg APF Δ | Verdict | Notes |
|---|---|---|---|---|
| C03 | SHAP feature ranking | −2.91 | FAIL | 5h +2.45 isolated, others negative |
| C12 | Stability filter @ thr=50 | −4.66 | FAIL | Worst of batch; concept genuinely dead |
| C23 | Per-regime feature set | −1.86 | FAIL | 8h +5.22 single-horizon win, others net negative |
| C29a | HAR-RV (Corsi 2009) | **+0.63** | **MARGINAL+** | Only positive avg; 5h +6.66 drives it |
| C29b | Hurst exponent | −1.89 | FAIL | Same 5h +6.66; other horizons hurt |
| C31 | Funding momentum | −1.90 | FAIL |  |
| C48 | Sharpe-aware label | −2.92 | FAIL | 5h −9.98 catastrophic |

**Net: 0 PASS, 1 MARGINAL+, 6 FAIL.** Suspect-pool from broken-harness era now formally closed. C29 HAR-RV's 5h +6.66 isolated win mirrors the C05/C06 per-horizon-vs-aggregator pattern — could be revivable via per-horizon-only application (not actionable as global feature add).

### ✅ Closed 2026-05-09 → 2026-05-10 (overlay-tier batch)

3-idea overlay batch via `tools/test_idea_batch_C16_C36_C45_C46_C48_C49_C51_C52_C53.py` (~30 sec runtime):

| CID | Idea | Result | Verdict |
|---|---|---|---|
| **C16** | **Disaster brake -5%** | **+1.96pp / 1 fire on 90d** | **MARGINAL+ → ACTIONABLE (ship)** |
| C52 | ATR trailing stop (12 sweep configs) | best 0pp (no fires); all firing variants lost (-3.78 to -13.39pp) | DEAD |
| C53 | Volume-spike exit (12 sweep configs) | all variants -0.74 to -2.49pp | DEAD |

C36, C45, C46, C48, C49, C51 returned STUB with explicit blockers (API keys / architectural / engine extension).

### ✅ Closed 2026-05-08 (overnight C50 HRST + trader fixes)

- **C50 PF objective HRST → DEAD.** Mode T total +70.92% (shield off+ON, n=57 trades, WR 84%) vs May 6 production HRST Mode T +76.77% (n=91, WR 79%) = **−5.85pp**, just past the kill threshold. Notable nuance: rawpf scoring DID pick better raw per-horizon configs (V0 baseline +75.17% vs prod +53.66%, +21.5pp), but Mode G found 0 STRICT gate winners on either bull or bear sweep — no shield/gate alpha could be added on top. Same family/fate as C13 CDaR. Log: [logs/c50_hrst_20260507_092632.log](logs/c50_hrst_20260507_092632.log) (~24h elapsed wall-clock due to laptop Modern Standby; actual compute ~7h). **Pattern confirmed**: scoring-change ideas show local Mode D wins but lose globally at HRST level. Canonical scoreboard: C50 moves to DEAD section.

### ✅ Closed 2026-05-07 — 5ideas runner Tier 1 retest batch (6 ideas DEAD on fixed harness)

**Memorialized retroactively 2026-05-19** after audit revealed these verdicts existed on disk but were never logged into the closed sections. Without this entry, IDEA QUEUE audits kept marking these as "needs retest" when in fact the retest happened and all 6 lost.

`test_desktop_5ideas_runner.py` ran 2026-05-07 00:21 → 03:50 on Desktop with fixed harness. Decision rule: ≥+5pp avg APF Δ vs baseline → promote to HRST validation. **NONE cleared.** Output: [logs/desktop_5ideas_summary_20260507_002108.txt](logs/desktop_5ideas_summary_20260507_002108.txt).

| CID | Idea | Runner verdict | Avg Δ APF | Why dead |
|---|---|---|---|---|
| **C35** | Wavelet multi-scale decomposition | MARGINAL | +0.227 | 5h +2.38 isolated, all other horizons negative. Below +5pp ship. Tier S prior (BTC 8h +40pp on prior engine) did not survive current engine. |
| **C47** | Vol-adjusted label | MARGINAL | +0.547 | 6h +2.38 + 7h +2.15, but 5h −0.17 and 8h −2.17 cancel out. KEPT-on-V3-lit prior did not carry. |
| **C43** | Stacking meta-learner (LR over RF+LGBM+XGB) | MARGINAL | +1.401 | 6h +4.50 isolated win, 8h −1.62. Best of the batch but well below +5pp ship. Same per-horizon-vs-aggregator pattern as C29 HAR-RV. |
| **C44** | Quantile regression target | FAIL | −0.814 | All horizons negative or flat. Tail-aware loss didn't help on this engine. |
| **C42** | CatBoost as 4th ensemble model | FAIL | −1.062 | 7h +1.32 isolated, three others lose. Different regularization than XGB but signal redundant with LGBM. |
| **C57** | Markov-switching AR detector | FAIL | −1.574 | 7h −4.95 catastrophic. Same regime-detection-family fate as C56 HMM (DEAD Δ−0.93). |

**C50 PF** was on the same runner (verdict=MARGINAL avg_delta=+3.377) and graduated to HRST → DEAD per 2026-05-08 entry above. **C05/C06** ran on the same runner but crashed with `OSError(22)` device-not-found; later re-run 2026-05-09 evening as the post-hoc rerank (see "Closed 2026-05-09 morning" entry).

**Pattern reinforced**: feature-add + scoring-overlay + alternate-model + alternate-regime-detector families all fail at Mode D smoke level. The H75 production engine's alpha ceiling from this class of changes is at-or-near zero. Future research should target **execution-gap** (~17pp untouched alpha per 2026-05-09 retro at line 1060 above).

### ✅ Shipped 2026-05-08 (trader correctness + plumbing)

1. **M-29b cold-path fix** ([crypto_revolut_ed_v2.py:876-895](crypto_revolut_ed_v2.py#L876)). Found via the **$1.87 BUY at 11:30:12 UTC** — first cold-path firing since M-29 fix shipped 2026-05-02. Bug: in `_execute_maker_order`, the "remaining cash < MIN_TRADE_USD, target unmet → market for residual" branch fired without checking whether real progress was made. When `usd_avail` was transient-low (cancel-release race or locked funds) and `spent_so_far == 0`, the trader market-bought $1.87 of ETH on a $14000 target. Fix: added `spent_so_far >= original_size * 0.5` progress guard. Without progress, returns `refused_low_cash_no_progress` and lets the maker loop retry next iteration. New log marker: `refusing dust-buy (likely cancel-release race or locked funds)`. **Status of M-29 entry**: 🟡 → **🚨** (cold path fired in production; fix shipped; needs another partial-fill BUY to confirm fix in cold path).

2. **SLA fix — eliminates P1 spam loop** ([crypto_revolut_ed_v2.py:4341+](crypto_revolut_ed_v2.py#L4341)). The trader's `_SOURCE_REGISTRY` had 6h SLAs on daily-cadence sources (`macro_daily, fear_greed, cross_asset, onchain_*`). Combined with content-aware `_file_is_fresh()` (checks last-row datetime, ~T-1 for daily data ≈ always >24h old), SLA always failed → re-download every cycle → wasted ~30s/cycle in P1 BLOCKING phase. Aligned to `feature_sources.json` proper SLAs:
   - `macro_daily`: 6h → **96h** (Mon-holiday-Fri-close gap)
   - `fear_greed`: 6h → **36h** (daily ~midnight UTC)
   - `cross_asset`: 6h → **48h** (daily yfinance)
   - `stablecoin_flows`: 12h → **48h** (daily CoinGecko)
   - `onchain_*` (BTC/ETH/XRP/LINK): 6h → **60h** (CoinMetrics T-1 + catch-up)
   - `ohlcv_*` and `derivatives_*`: kept at 2h (correct for hourly data)

   Verified post-restart: P1 duration alternates between 0.11s (no refresh) and ~33s (legitimate 2h hourly refresh) instead of 32s every cycle. **~50% fewer downloads, much shorter average P1 BLOCKING phase.**

3. **`/reload` double-fire fix — M-16b dedup** ([crypto_revolut_ed_v2.py:_telegram_command_loop](crypto_revolut_ed_v2.py#L4077)). Symptom: clicking the 🔄 Reload Telegram button ran the full reload cycle TWICE. Root cause: when an inline button tap arrives as both `callback_query` AND `message.text` in the same Telegram poll batch (client/network quirk), the same command was dispatched twice. Fix: per-command 5-second dedup window. Repeat of the same command within 5s logs `[telegram] dedup: dropped repeat` and is dropped. `/buy` and `/sell` bypass dedup (they have their own ✅ Confirm inline flow). New log marker to grep for: `[telegram] dedup: dropped repeat`.

4. **Orphan cleanup**: `create_hourly_macro()` function and `data/macro_data/macro_hourly.csv` removed from `download_macro_data.py` and AB matrix snapshot lists. The hourly proxy was originally a DAX system feature; zero consumers in current codebase (verified via grep across engine + cfd/). The file had been silently 48h stale for at least the past few days because `download_yfinance_data()` only refreshed `macro_daily.csv` when stale, and `create_hourly_macro` was gated on that.

### ✅ Shipped 2026-05-09 morning (M-30 spread-aware BUY-slide v4)

- **M-30 v4 spread-aware BUY-side slide** ([crypto_revolut_ed_v2.py:678](crypto_revolut_ed_v2.py#L678)). Trigger: 2026-05-09 01:00 UTC ETH BUY incident. Target $13,999.99, only **26.2% filled in maker** ($3,674.80) across attempts #1-#5, then **22 consecutive `post_only` rejections** (#6-#27) silently ignored by the bot, then 73.8% MARKET fallback at $2,311.34, **$9.29 in taker fees**.

  **Root cause**: Option A slide (2026-04-26) interpolates linearly from `bid+0.01` to `ask-0.01` over 30 attempts. On a 16bps / $3.75-spread book, attempt #6 was already at `bid+$0.92` (progress=0.26) — far enough up the book that normal latency between get_quote and POST arrival made the venue ask drop to that price. `post_only` correctly rejected. The bot's placement loop catches `post_only` errors and retries with a fresh quote (line ~742) but doesn't break or count rejects, so it kept reposting blindly.

  **Fix**: `SLIDE_SPREAD_THRESHOLD_BPS = 5.0`. If `spread_bps > 5.0` (typical Revolut X case for ETH/BTC, 10-20 bps), stay flat at `bid+0.01` for the whole window. If spread ≤ 5 bps (low-liquidity hour with tight book — original Option A motivation), use original slide. Worst case = unchanged: timeout at maker_window → MARKET. No regression possible.

  **Simulation prediction**: Central case ($19/s fill rate at bid+0.01) → 55% maker, 45% MARKET, $5.67 fees, $2,309.84 avg. **Both better than actual outcome** ($9.29 fees, $2,311.34 avg). Pessimistic case ($9.5/s) ≈ break-even on fees, slightly better avg price. Trade-off: sacrifices half-spread upside on rare taker-SELL flow during wide-spread periods.

  **What the v4 patch deliberately does NOT include**: reject counter, Telegram alert, market bail-out. v4 prevents the rejection storm at source — if rejects still happen despite v4, the hypothesis is wrong and we need re-debug, not mask with a faster bail-out (which pushes toward taker, the very thing we're avoiding).

  **Monitoring**: first wide-spread BUY after deploy → tail log, confirm posted price stays at `bid+0.01` for full window. After 1-2 episodes, compute observed fill rate at bid+0.01, compare to simulation. If consistently below pessimistic ($9.5/s), consider re-introducing a delayed slide (start at attempt 10 instead of 1).

  Pre-restart audit: `tools/test_c04_to_c08_runner.py` now also references CPU LGBM forcing (M-29-related); these don't conflict with M-30. Trader restart required to pick up M-30.

### ✅ Closed 2026-05-09 morning (overnight + early-morning sweep)

- **C05/C06 CVaR/Sortino rerank → 🎯 STRONG PASS DIRECTION** but **contaminated baseline grid**. CVaR_lam1 / Sortino reranks pick materially better candidates than APF on every horizon: 5h +92pp ret (apf_rank=13), 6h +30pp (apf_rank=9), 7h +21pp (apf_rank=9), 8h +85pp ret AND −20pp DD (apf_rank=8). HOWEVER the baseline grid CSVs were last refreshed by C50 HRST (`OPTUNA_METRIC='rawpf'`), so the "top APF among rawpf-optimized configs" is not directly transferable to production. **MOST PROMISING RESEARCH FINDING IN WEEKS** — needs clean APF-grid refresh + re-run to confirm before HRST validation. Output: `output/c05_c06_rerank_20260508_223911.csv` + `output/c05_c06_per_trade_20260508_223911.csv`.

- **C07 CUSUM sampling smoke test → MARGINAL/DEAD**. Best CUSUM (h=1.0%) +2.4pp accuracy and ~+35pp simPnL vs hourly baseline, but both are catastrophic losses (best CUSUM = −65%, hourly = −100%). Below +5pp accuracy ship threshold. Architectural rewrite of trader polling loop NOT justified. Move to DEAD.

- **C01 vol-scaled horizons 4mo → SHELVED/NULL**. Original 2mo "winner" `vol_2band low→8h high→6h @90%` (+5.02pp on 2mo) FAILED to replicate at 4mo: +30.98% vs current baseline `tsmom_672h bull=6h bear=8h @90% +38.49%` = **−7.51pp**. Different best variant on 4mo (`vol_median below→8h above→6h @90%` +41.48%) only beats baseline by +2.99pp — below +5pp ship. Window-shopping, won't generalize. Vol-scaled horizons family closed.

- **C32-C40 batch (7 active + C36 STUB) → ZERO PASS, 1 FAIL, 6 MARGINAL**:
  - C32 liquidation_cascade: **FAIL** Δ−5.36 (catastrophic 7h:−11.6, 8h:−15.5 — adding noisy features hurts long horizons)
  - C33 spread_compression: MARGINAL Δ−1.93
  - C34 eth_btc_cointegration: MARGINAL Δ−3.96 (catastrophic 7h:−12.0, 8h:−9.7)
  - C36 news_sentiment: SKIPPED (CryptoPanic/Santiment APIs need keys)
  - C37 stablecoin_velocity: MARGINAL Δ−0.10 (best of batch — essentially noise-neutral)
  - C38 korea_premium: MARGINAL Δ−1.03
  - C39 long_short_ratio: MARGINAL Δ−0.78
  - C40 skew_kurt: MARGINAL Δ−1.82 (with 7h Mode D ERROR, others mild loss)

  Summary: [logs/c32_to_c40_summary_20260509_000845.txt](logs/c32_to_c40_summary_20260509_000845.txt). **Pattern confirmed: feature-add family is exhausted. Net 0 PASS over 11 ideas tested in last 24h** (C50 + C32-C40 batch + C07 + C01 + C05/C06).

### 🚀 Most actionable next step (post 2026-05-09 morning) — C05/C06 path

User running the sequence on Desktop (2026-05-09 evening). Three sequential steps with explicit decision points:

**Step 1 (~30-50 min) — Refresh contaminated baseline grid:**
```
python crypto_trading_system_ed.py D ETH 5,6,7,8h --replay 1440 --no-persist --no-data-update
```
Overwrites `models/crypto_ed_grid_ETH_<h>h.csv` with proper APF-Optuna-hill-climbed candidates (was overwritten by C50 HRST 2026-05-07 with rawpf-optimized candidates).

**Step 2 (~90 min) — Re-run C05/C06 on clean grid:**
```
python tools/test_c05_c06_only.py
```
Output: `output/c05_c06_rerank_<ts>.csv`. Compare to 2026-05-08 result. **Decision point**:
- Reranks STILL pick materially different + better candidates (5pp+ delta on multiple horizons) → CONFIRMED PASS DIRECTION → proceed to Step 3.
- Reranks pick same candidates as APF → 2026-05-08 result was contamination artifact → C05/C06 family closed.
- Reranks differ but with smaller deltas → real but weaker effect → judgment call on whether Step 3 is worth ~7-8h.

**Step 3 (~7-8h overnight, CONDITIONAL on Step 2 PASS):**
1. Build engine fork `crypto_trading_system_ed_cvar.py` mirroring `_cdar.py` pattern (~30-60 min coding):
   - Output paths: `crypto_ed_production_cvar.csv` + `regime_config_ed_cvar.json` (no prod overwrite)
   - Add `'cvar'` branch in `_compute_optuna_score`: `score = return - λ × |CVaR_5%|` (start λ=1.0)
   - Default `OPTUNA_METRIC = 'cvar'`
   - Patch 4 Mode V winner-selection sites (lines 4962, 4976, 6194, 7986, 7999 per existing `_cdar.py` precedent) to use CVaR-aware variant
   - Port per-trade-return tracking from C05/C06 work into `_simulate_with_threshold`
2. Run full HRST:
   ```
   python crypto_trading_system_ed_cvar.py HRST ETH 5,6,7,8h --replay 1440 --no-persist
   ```
3. Compare Mode T REF to current production (May 6 promotion +76.77%). **Decision**:
   - **≥ +81.77% (≥+5pp)** → SHIP. 1-line scoring change goes to live engine.
   - **±5pp** → null. Mark CVaR/Sortino MARGINAL.
   - **< +71.77% (<-5pp)** → DEAD. Same fate as C13 CDaR (-15pp HRST) and C50 PF (-5.85pp HRST).

**Pattern context**: same-sub-family ideas C13 + C50 both showed Mode D wins but lost at HRST. C05/C06 differs notably — its smoke test showed massively better DRAWDOWNS (8h: 47% → 27%), not just better returns. Lower-DD configs survive Mode T better because shield/gate has less work to do. Prior estimate: 30-40% HRST PASS, 60-70% null/fail. Worth doing because asymmetry is good (massive upside if ships, modest cost if doesn't).

This is the only research path with positive prior left. If steps 1-3 produce a real ship, it's ETH's first scoring-overlay win. If they produce a null on clean grid, scoring-overlay family conclusively closed — pivot to **execution-gap research (~17pp untouched alpha)**.

### 🟡 Queued for manual launch (2026-05-08 evening) — STATUS UPDATE

C01 + C07 + C32-C40 batch all completed by 2026-05-09 ~03:42. See verdicts above. Original "queued" table superseded by the closed entries above.

| Script | Machine | ETA | Family / Decision rule |
|---|---|---|---|
| `python tools/test_c01_vol_scaled_4mo.py` | Laptop | ~30-45 min | C01 — vol-scaled horizons 4mo confirmation. 2mo was +5.02pp over `tsmom_672h`. Decision: best vol-scaled variant ≥+5pp over current live baseline AND WR ≥ 60% → ship as detector. Otherwise null. Wraps `test_vol_scaled_horizon.py` with `REPLAY=2880`. |
| `python tools/test_c07_cusum_sampling.py` | Laptop | ~5-10 min | C07 — CUSUM event-based sampling (López de Prado AFML Ch 2) feasibility check. 4 thresholds (0.5/1.0/1.5/2.0%) on 2yr ETH hourly. Decision: best CUSUM beats hourly baseline by ≥+5pp accuracy AND ≥+10pp simulated PnL → architectural rewrite worth the days of work. Within ±5pp/±10pp → null. Worse → DEAD. Self-contained (no engine import). |
| `python tools/test_c32_to_c40_batch.py` | Desktop overnight | ~5-7h | **C32-C40 batch** (8 ideas, 7 active + C36 STUB). Sequential Mode D ETH 5,6,7,8h --replay 1440 --no-persist for each: C32 liquidation cascade proxy (no Coinglass needed — derives from existing OI/return/vol), C33 spread compression z-score, C34 ETH-BTC cointegration residual, **C36 STUB** (CryptoPanic + Santiment need API keys), C37 stablecoin issuance velocity, C38 Korea/Coinbase premium spread (pre-fetches Bithumb + Coinbase + yfinance KRW=X), C39 Binance long/short ratio (pre-fetches futures endpoint), C40 skew + kurt features. Per-idea verdict: avg APF Δ ≥+5pp PASS / ±5pp MARGINAL / ≤−5pp FAIL. Output: `logs/c32_to_c40_summary_<ts>.txt`. **Pattern caveat**: this week's harness pattern shows 0 PASS in 7 retests. Expected hit rate: 0 PASS, 1-2 MARGINAL, rest FAIL — worth running to definitively close C30s + C40s. |

C05/C06 currently running on laptop (background task, started 2026-05-08 22:39 via `tools/test_c05_c06_only.py`). ETA ~01:00 local. Tracks: 39/60 candidates done at 23:44, no errors; M-29 OSError(22) was GPU contention with concurrent C50 HRST and is now resolved by forcing CPU LGBM in `_backtest_with_per_trade_capture`.

### ✅ Closed 2026-05-06 evening (C04→C08 runner partial outcome)

| CID | Verdict | Evidence |
|---|---|---|
| **C04** Variance Ratio (q=12,24,48) | **DEAD** Δ−1.21 avg APF (FAIL) | [logs/c04_to_c08_summary_20260506_213526.txt](logs/c04_to_c08_summary_20260506_213526.txt) |
| **C56** HMM 2-state regime | **DEAD** Δ−0.93 avg APF (marginal FAIL, fixed-harness-validated) | [logs/c56_only_summary_20260506_195258.txt](logs/c56_only_summary_20260506_195258.txt) |
| **C08** Single-horizon CDaR | **DEAD** 5h −20.48pp / 6h −14.05pp vs APF prod (7h crashed, 8h killed by user 2026-05-07 to free machine) | [logs/c08_cdar_V_5h_20260506_213526.log](logs/c08_cdar_V_5h_20260506_213526.log), [c08_cdar_V_6h_20260506_213526.log](logs/c08_cdar_V_6h_20260506_213526.log) |
| **C07** CUSUM | **SKIPPED** (architectural, not a smoke test) | — |
| C05 / C06 | runner crashed `KeyError('models')` — bug fixed 2026-05-07; bundled into tonight's overnight | — |

### ✅ Shipped 2026-05-06 → 2026-05-07 (production engine + trader)

- **HRST ETH 5,6,7,8h --replay 1440 promoted to live** ([commit-equivalent: surgical ETH-only edit in `regime_config_ed.json` + 4 model-CSV rows; backup files `*.backup_20260506_pre_HRST_promote.*`])
  - detector `sma24>sma100` → `tsmom_672h`
  - bull 6h@65% shield-ON → 6h@75% shield-OFF + rally rr18≥5% OR rr30≥7.5% cd=36h
  - bear 5h@75% shield-ON → 8h@65% shield-OFF + rally rr24≥4% OR rr30≥6% cd=14h
  - 2-month replay return: +76.77% (Mode T final), +70.75% (Mode S baseline)
- **SELL maker pricing rewrite** ([crypto_revolut_ed_v2.py:698-708](crypto_revolut_ed_v2.py#L698)) — sit at `bid + max($0.02, 5%×spread)` instead of slide from `ask-0.01`. Empirically every recent SELL filled at attempts #17-19 of the old slide at near-floor prices anyway, while the bid drifted away during the wait.
- **Layer 5 OI pagination boundary fix** ([download_macro_data.py:834-859](download_macro_data.py#L834)) — Layer 1 (`fetch_with_fallback`) was working; the recurring HTTP-400-endTime spam was actually pagination running past Binance's ~30d 1h-OI history limit and killing the entire fetch. Now `retries=1` + try/except for the expected `HTTP 400 endTime` → graceful break, fresh ~30d OI preserved. Validated 2026-05-07 07:13 trader restart: cycle 1 shows "Open interest: 3 features added", no retry storm.
- Trader restart 2026-05-07 07:13 confirmed to load all three changes.

### 🟡 Tonight's overnight (Desktop, ~7-8h hands-off)

```
python tools/test_desktop_5ideas_runner.py
```

Renamed in spirit to "suspects runner". Removed C56/C04 (already retested, both DEAD). Added C05/C06 post-hoc rerank (runner bug fixed). Order:
1. C35 wavelet (~30 min)
2. C50 PF-as-objective (~28 min)
3. C47 vol-adjusted label (~28 min)
4. C44 quantile-regression feature (~32 min)
5. C57 Markov-switching AR (~50 min)
6. C42 CatBoost ensemble (~80 min)
7. C43 Stacking meta-learner (~150-240 min, heaviest last)
8. **C05 CVaR + C06 Sortino joint rerank** (~90 min, post-hoc on existing baseline grid)

Outputs:
- `logs/desktop_5ideas_summary_<ts>.txt` — patcher-loop verdicts (PASS/MARGINAL/FAIL per idea)
- `output/c05_c06_rerank_<ts>.csv` + `c05_c06_per_trade_<ts>.csv` — manual review (does CVaR/Sortino pick a different top-3 with lower tail-risk?)

After tonight, the canonical-scoreboard "SUSPECT" pool drops from 13 → 6 (C03 SHAP, C12 stability thr=50, C23 per-regime, C29 HAR-RV, C31 funding momentum still untested on fixed harness — Tier C, low priors).

### 🟦 Untested actionable (post-tonight)

1. **C01 vol-scaled 4mo** — `python tools/test_vol_scaled_horizon.py --replay 2880`, ~30 min. **Caveat**: 2mo +5.02pp gain was vs the now-defunct `tsmom_672h` baseline used pre-HRST-promote. With live now ON `tsmom_672h` + shield-off + new gates, the baseline has shifted; rerun is needed.
2. **C16 narrow** — disaster brake at −5%, ~30 min.
3. Tier C suspect retests (C03, C12, C23, C29, C31) — only if budget available; low priors.

### Out-of-band

- **M-29 partial-fill live test** — fires whenever next live BUY happens; no scheduling. Trader currently in cash; 6h conf 46.50% well below 75% threshold.
- **Validate new HRST config in live trading** — 2026-05-07 07:13 restart confirmed: detector tsmom_672h active, bull 6h@75% shield-OFF, bear 8h@65% shield-OFF, rally gates rr18/rr30 firing correctly. Watch for first BUY/SELL signal after restart.

---

### 🚨 CRITICAL HARNESS BUG DISCOVERED — 2026-05-05 morning

**Symptom**: overnight 9-idea Desktop runner (C56, C35, C04, C50, C47, C44, C57, C42, C43) finished 23:24 → 03:59 with all 9 reporting FAIL ~−5pp avg APF and **eerily identical per-horizon delta patterns** (5h ~−14, 6h ~−7-9, 7h ~−2.6 to −4.5, 8h ~+4.5-5).

**Root cause**: [tools/test_14_ideas.py](tools/test_14_ideas.py) `run_mode_d` line ~125:
```python
py_init += "import runpy; runpy.run_module('crypto_trading_system_ed', run_name='__main__')"
```
`runpy.run_module(name, run_name='__main__')` **re-executes the engine module from scratch in a new namespace**, discarding any monkey-patches the patcher applied to `sys.modules['crypto_trading_system_ed']`. The fresh re-execution defines `build_all_features` anew, and Mode D calls find that fresh definition, not the patched one.

**Confirmation**: every idea's patcher prints its "import-time" banner (`[CXX] build_all_features patched`) once per log, but the runtime markers (e.g. `[C56] HMM regime features added`, `[C04] VR features added`) appear **0 times** across all 36 horizon runs. The engine ran with stock features for all 9 ideas.

**Verdicts now SUSPECT (need retest with fixed harness)**:
1. **C03 SHAP** — was marked DEAD (Δ−4.78), retest needed
2. **C23 per_regime_features** — was marked DEAD (Δ−1.18), retest needed
3. **C12 stability_strict** — was marked DEAD (Δ−0.75), retest needed
4. **C29 HAR-RV + Hurst** — was marked DEAD (Δ−2.99 / −3.09), retest needed
5. **C31 funding momentum** — was marked DEAD (Δ−5.25 yesterday), retest needed
6. **C56 HMM regime** — overnight FAIL (Δ−5.34), retest needed [Tier S prior]
7. **C35 wavelet** — overnight FAIL (Δ−5.37), retest needed [Tier S prior]
8. **C04 Variance Ratio** — overnight FAIL (Δ−5.44), retest needed
9. **C50 PF objective** — overnight FAIL (Δ−4.95), retest needed
10. **C47 vol-adjusted label** — overnight FAIL (Δ−5.00), retest needed
11. **C44 quantile regression** — overnight FAIL (Δ−5.02), retest needed
12. **C57 MS-AR** — overnight FAIL (Δ−5.01), retest needed
13. **C42 CatBoost** — overnight FAIL (Δ−5.20), retest needed
14. **C43 Stacking** — overnight FAIL (Δ−5.24), retest needed

**Verdicts STILL RELIABLE** (didn't use the harness — standalone overlay sims or full HRST):
- C09, C10 (DEAD via [tools/test_c09_c10_overlay.py](tools/test_c09_c10_overlay.py))
- C02 (DEAD via [tools/test_c02_multi_horizon_exit.py](tools/test_c02_multi_horizon_exit.py))
- C13 CDaR (DEAD via two full HRST runs of cdar engine fork)
- All historical DEAD/SHELVED verdicts predating the harness (C17, C18, C19, C20, C21, C22, C24, C25, C26, C27, C28 — all confirmed via standalone scripts or full engine runs in March-April)

**Fix shipped 2026-05-05 morning** ([tools/test_14_ideas.py:125](tools/test_14_ideas.py#L125)):
- Replaced `runpy.run_module('crypto_trading_system_ed', run_name='__main__')` with `import crypto_trading_system_ed; crypto_trading_system_ed.main()`. Direct call preserves the patcher's monkey-patches because the module remains the same `sys.modules` entry.
- **Verified**: smoke-test ran C35 wavelet on ETH 5h → log now shows `[C35] wavelet features added: +4 columns` (3× per Mode D run vs 0× pre-fix). Engine ran with 184 features (baseline 180 + 4 wavelet) instead of dropping the patch.
- Sanity-checked patcher persistence: after patcher import, `eng.build_all_features.__name__ == '_patched_build'` (was `'build_all_features'` original pre-fix).

**Smoke-test result**: ETH 5h C35 wavelet, fixed harness, Δ = **−15.56pp** vs baseline. Patches now genuinely active; C35 at 5h is genuinely negative on this baseline. (Caveat below.)

**Caveat about untagged baselines**: the baseline grids `models/crypto_ed_grid_ETH_<h>h.csv` (mtime 2026-05-04 03:02-06:08) were written by the Desktop CDaR HRST run, which used **CDaR scoring** as Optuna metric. Those baselines happen to have unusually high APF (5h apf=30.4, 6h=23.7) because Optuna hill-climbed under CDaR scoring rather than APF. Comparing fresh APF-Optuna-selected configs against those baselines produces a consistent ~−5pp drift that has NOTHING to do with the patcher. Before re-running the 14 SUSPECT verdicts, **refresh untagged baselines first**:
```
python crypto_trading_system_ed.py D ETH 5,6,7,8h --replay 1440 --no-persist --no-data-update
```
**~30-50 min on Desktop** (Mode D × 4 horizons, sequential outer loop; per Runtime Reference: 7-13 min/horizon median 11). Overwrites the 4 untagged grid CSVs with clean APF-Optuna baselines.

**Re-run plan** (post-baseline-refresh, **~4-5h on Desktop** = 9 ideas × 30 min each per Runtime Reference, with 2x harness parallelism):
- Re-run the 9 overnight ideas on the fixed harness via `python tools/test_desktop_5ideas_runner.py`. Now patches will actually fire AND the comparison baseline will be apples-to-apples.
- 5 prior-day SUSPECT verdicts (C03, C12, C23, C29, C31) can be re-validated cheaply by re-running their original test scripts (~20-30 min each). C12 thr=30 was a STANDALONE run (not via this harness) so it stays valid; only C12 thr=50 was via the broken harness.

---

### 🔴 CDaR HRST RESOLVED ON BOTH MACHINES — DO NOT PROMOTE — 2026-05-04 morning

**Status**: BOTH laptop AND desktop ran CDaR HRST overnight in parallel. Both completed independently. Both produced Mode T totals 14-16pp UNDER current live's +86.19%. Cross-machine variance was only ~1.5pp — confirming the gap is signal, not run-to-run noise. **SHELVED with high confidence. Live config UNTOUCHED.**

**Cross-machine comparison (2 independent runs)**:

| Metric | Laptop ([log](logs/ed_v1_20260503_200545.log), 9h, finished 05:13) | Desktop ([log](logs/ed_v1_20260504_025746.log), 6h, finished 08:43) | Live (May 3) |
|---|---|---|---|
| Mode S detector | `sma24>sma100` | `sma24>sma100` | `sma24>sma100` ✅ all match |
| Mode S bull | 5h@75% | 5h@75% | 6h@65% |
| Mode S bear | 8h@65% | 8h@80% | 5h@75% |
| Mode S return | +66.25% | +65.70% | +77.06% |
| Mode S alpha | +59.27pp | +56.65pp | +60.37pp |
| Mode T total | **+71.22%** | **+69.70%** | **+86.19%** |
| Bull shield | ON | ON | ON ✅ |
| Bear shield | ON | OFF | ON |
| Iters to converge | 2 | 2 | 2 |
| STRICT gate winners | 0 | 0 | 0 |
| Per-horizon picks (8h) | Refined #1 APF=22.0 ret +67.03% | Refined #3 APF=25.3 | — |

**Cross-machine consistency**:
- Detector: identical (`sma24>sma100`)
- Bull horizon + conf: identical (5h@75%)
- Bear horizon: identical (8h)
- Mode T total spread: 1.52pp (well within run-to-run noise)
- Both pick bull shield ON; bear shield differs (laptop ON, desktop OFF) — but Mode T total nearly identical regardless

**Gap to live: laptop −14.97pp, desktop −16.49pp**. Both far past the +5pp promotion bar AND past the −5pp "noise band" threshold. With two independent runs producing the same direction and similar magnitude, the −15pp gap is robust to seed/machine noise.

**Per-horizon Mode V winners (CDaR scoring) — not in dispute**:

| H | Source | Combo | W | g | f | Return | WR | Trades |
|---|---|---|---|---|---|---|---|---|
| 5h | D #3 (APF=−0.6) | RF+LGBM | 72h | 0.999 | 10 | +28.27% | 87% | 30 |
| 6h | Refined #1 (APF=36.7) | XGB+LGBM | 95h | 1.000 | 11 | +50.88% | 76% | 76 |
| 7h | D #6 (APF=0.1) | RF+LGBM | 100h | 0.995 | 17 | +33.71% | 86% | 28 |
| 8h | Refined #1 (APF=22.0) | RF+LGBM | 158h | 0.999 | 8 | **+67.03%** | 72% | 64 |

**Why CDaR failed at the HRST level despite winning Mode D smoke test**:
1. Mode D smoke-test wins (5-8h all dominated APF) DID translate cleanly to per-horizon Mode V winners.
2. Mode S joint sweep picked `sma24>sma100 5h/8h` — **NOT** a regime that exploits the 8h Refined #1 +67% winner. The sweep prefers shorter bull (5h) over CDaR-favored long-horizon configs.
3. Result: the strongest CDaR-favored single-horizon result (8h +67.03%) gets diluted in the Mode S regime split.
4. Compounding: Mode R verdict said "Baseline WINS by +1.66%" at conf>=90% — different from current live config which lives on a regime split.

**Open question (NOT resolved yet)**: does CDaR scoring help at the SINGLE-HORIZON level even if Mode S/T washes out? The 8h Refined #1 +67.03% is the strongest CDaR-only result. **Possible follow-up**: keep current production engine but use CDaR scoring for individual-horizon model selection only. Not a priority unless other paths exhaust.

**Action**: SHELVE CDaR engine fork. Do NOT promote. Live config / production CSV untouched. The CDaR fork (`crypto_trading_system_ed_cdar.py` + `_cdar.csv` + `_cdar.json`) can stay on disk for the single-horizon-only follow-up, or be deleted.

---

### 📅 TODAY (2026-05-04) — END-OF-DAY STATE + REMAINING PLAN

**The canonical scoreboard is at section "📋 CANONICAL IDEAS SCOREBOARD" below — that's the single source of truth for which ideas are tested/dead/untested. This section just summarizes today's work and sets the next-action queue.**

**Today's work log (chronological)**:

| Time | Job | Machine | Status | Outcome |
|---|---|---|---|---|
| 05:13 | CDaR HRST ETH 5,6,7,8h --replay 1440 | Laptop | ✅ Done | **C13 SHELVED** — Mode T +71.22% vs live +86.19% (−14.97pp) |
| 08:43 | CDaR HRST ETH 5,6,7,8h --replay 1440 (parallel) | Desktop | ✅ Done | **C13 SHELVE CONFIRMED** — Mode T +69.70% (−16.49pp); cross-machine variance 1.52pp |
| (overnight) | test_14 batches 1-2 (IDEA 1,5,6,7,10 + 2,12,13,14) | Desktop | mixed | C23/C12-revisit/C29×2 FAIL; C03 needs `pip install shap`; C09/C10 ERROR (harness bug); C02/C04 STUB |
| 20:39 | C09+C10 harness bug fix + audit rerun | Laptop | ✅ Done | Bug fixed (cache is list, not dict); STUBs now produce skip counts |
| 21:30 | **C09+C10 full PnL overlay** ([tools/test_c09_c10_overlay.py](tools/test_c09_c10_overlay.py)) | Laptop | ✅ Done | **Both DEAD.** C09: best variant −1.22pp; C10: best −7.79pp. All 35 variants LOSE vs baseline +147.59% |

**Net result of today's work**: 6 ideas closed (C13 shelved, C09/C10/C23/C29-pair newly DEAD, C12-revisit confirmed-DEAD-with-stricter-thresholds). 0 wins. Live config UNTOUCHED.

**Tomorrow's priority queue** (numbered by canonical CID — see scoreboard for full context):

1. ⭐ **C01 vol_scaled_4mo** — ~30 min, only remaining promising candidate. 2-month proved +5.02pp; 4-month is the only blocker:
   ```
   python tools/test_vol_scaled_horizon.py --replay 2880
   ```
   Decision rule: ≥+5pp Mode T REF over `tsmom_672h` → config-only ship + instant rollback.

2. **C03 shap_ranking** — ~30 min, just needs install:
   ```
   pip install shap
   python tools/test_14_ideas.py --idea 10
   ```

3. **C02 multi_horizon_exit** — 1-2h, build the overlay-PnL sim (template = today's [tools/test_c09_c10_overlay.py](tools/test_c09_c10_overlay.py), reuse `simulate()` function). Harness STUB shows 33% co-incidence — high prior this is too frequent and will fail like C09/C10, but cheap to verify.

4. **C04 variance_ratio_detector** — 2-3h, register `vr_12` as named detector in `_build_regime_indicators_and_detectors`, run Mode S, compare vs `tsmom_672h`.

5. **C08 single-horizon CDaR variant** — 2h, open follow-up after C13 HRST fail. The 8h Refined #1 +67.03% was the strongest CDaR result; needs new infra to run scoring per-horizon without Mode S regime split.

6. **C05/C06 CVaR + Sortino objectives** — 3-5h each, blocked on extending grid CSV writer to include per-trade-return list.

7. **C07 CUSUM event-based sampling** — days, architectural rewrite of trader polling loop.

**Out-of-band priorities**:
- **M-29 partial-fill live test** (#2 priority below) — fires whenever the next live BUY happens; no scheduling.
- **Re-validate harness FAIL verdicts** — optional but informative. The `--grid-tag` patch (2026-05-03 evening) shipped; need to confirm IDEA-tagged baselines were measured against CLEAN untagged baselines, not contaminated ones. If contaminated, C23/C12-revisit/C29 FAIL verdicts could flip.

**Standing recommendation**: after running #1-2 (~1h total), if both fail, the 20-ideas roadmap is essentially exhausted. Pivot to **execution-gap work on the live trader** — ~17pp unaccounted alpha gap between sim and live is the biggest lever left. TCA logging, slippage audit, manual-vs-auto PnL decomposition.

---

### 📅 ORIGINAL TOMORROW (2026-05-04) — TEST PLAN (now SUPERSEDED — overnight results above)

Concrete schedule to run in the morning. Cross-machine parallelism assumed (Desktop + Laptop both available).

**🌅 Morning (08:00-09:00) — kick off the long jobs first**

**Desktop (heavy GPU/CPU run, ~6-9h unattended, avg 7h — see Runtime Reference):**
```bash
python crypto_trading_system_ed_cdar.py HRST ETH 5,6,7,8h --replay 1440 --no-persist
```
This is the CDaR validation HRST. Already #2 priority below this block. Launch FIRST, runs in background until ~15:00 (worst case 17:00). This is the make-or-break test for whether CDaR scoring becomes the new production engine.

**Laptop (light parallel runs, ~15 min total):**
```bash
python tools/test_14_ideas.py --idea 2,12,13
```
Runs the 3 cheapest audit STUBs (`multi_horizon_exit`, `vol_entry_filter`, `btc_momentum_gate`) in sequence. Each ~5 min, no Mode D needed — they audit existing 90d signal cache. Outputs partial-result CSVs you can inspect over coffee.

**🌞 Mid-morning (09:00-10:30) — Tier 1 fully-wired smoke tests on Laptop**

```bash
python tools/test_14_ideas.py --idea 1,6
```
- **#1 per_regime_features** (~30 min) — never tested; bias bull horizons toward technical features, bear toward macro
- **#6 har_rv** (~20 min) — Corsi 2009 HAR-RV realized variance feature

These run via Mode D + monkey-patch on Laptop while Desktop is busy with CDaR HRST. Each writes to `_TAGGED` paths; production untouched.

**🌤 Late morning (10:30-11:00) — Tier 1 highest-EV test on Laptop**

```bash
python tools/test_vol_scaled_horizon.py --replay 2880
```
**This is the single most important test of the day** — `vol_scaled_4mo` already proved +5.02pp on 2-month window. 4-month confirmation is all that blocks promotion. ~30 min on Laptop. If this passes (Mode T REF beats `tsmom_672h` baseline by ≥+5pp on 4mo), it replaces the current detector via config-only change with instant rollback.

**🍽 Lunch break (12:00-13:00) — let things finish, no commands**

**🌥 Early afternoon (13:00-14:00) — REVIEW**

Check status of all 6 jobs. Order of inspection:

1. **CDaR HRST result** (Desktop): grep for Mode T REF in latest `logs/ed_v1_*.log`. Compare to current production HRST (+86.19% from May 3 morning).
   - If Mode T REF ≥ +91pp (>+5pp over current) → **PROMOTE CDaR ENGINE TO LIVE** (see promotion procedure in #2 priority below)
   - If within ±5pp → keep current scoring, document inconclusive
   - If worse by >5pp → CDaR scoring hurt at HRST level despite winning at Mode D level. Investigate.

2. **vol_scaled_4mo result**: read top of `tools/test_vol_scaled_horizon.py` output for the Mode T REF of best variant vs `tsmom_672h` baseline. ≥+5pp → promote (config-only, see Tier 1 instructions above).

3. **Tier 1 harness results**: read `logs/test_14_summary_<ts>.txt` for #1 + #6 verdicts. Look for PASS, MARGINAL, or FAIL. PASS → escalate to full HRST validation; MARGINAL → document; FAIL → close out as DEAD in scoreboard.

4. **Tier 2 STUB outputs**: skim `output/idea_02_*.csv`, `output/idea_12_*.csv`, `output/idea_13_*.csv`. For each, look at the skip-rate / overlap stat. If the rate is in 5-15% range → worth spending the time to flesh out a full simulation. Outside that range → kill or shelve.

**🌆 Late afternoon (14:00+) — DECISIONS**

Based on the morning's results, exactly one of these branches:

- **CDaR PROMOTED**: do the promotion (see #2 priority below for backup + copy commands), restart trader, monitor first 10 trades. Stop further Tier-1 testing today.
- **vol_scaled_4mo PROMOTED**: config-only change, no engine swap. Do simultaneously with CDaR if both win.
- **#1 per_regime_features PASS**: launch full HRST with monkey-patch in place to confirm — Desktop free now since CDaR is done.
- **All FAIL/MARGINAL**: proceed to Tier 3 (#5, #7, #10) or revisit STUB blockers for Tier 2.

**🌙 Evening — record + plan day after**

Update CLAUDE.md scoreboard for whatever was tested. Each idea moves to PASS / FAIL / SHELVED with one-line summary + delta. Pick next-day test based on what's left.

**Time budget**: ~8h elapsed (08:00-16:00) but only ~2h hands-on (the rest is jobs running unattended). All tests are parallel-safe across the two machines.

**M-29 partial-fill live test** (separate, opportunistic — see priority #3 below): runs whenever the next live BUY happens. No scheduling needed.

---

### 📜 14-IDEAS HARNESS RESULTS LOG — historical, see canonical scoreboard for current status

**Run results 2026-05-03 → 2026-05-04** (kept here as audit trail; canonical CIDs are the single source of truth):

| H14# | CID | Idea | Verdict | Notes |
|---|---|---|---|---|
| 1 | C23 | per_regime_features | ❌ FAIL Δ−1.18 | DEAD. Bias bull→tech / bear→macro hurts. |
| 2 | C02 | multi_horizon_exit | 🟦 STUB 33% co-incident | UNTESTED (full PnL sim pending). Probably DEAD given high co-incidence. |
| 3 | C01 | vol_scaled_4mo | 🟡 not run via harness | UNTESTED at 4mo (2mo +5.02pp known). HIGHEST EV remaining. |
| 4 | C11 | vpin_5min variant | ⚪ not yet attempted | Base C11 SHELVED (best +3.83pp); 5-min variant is the revisit path. |
| 5 | C12 (revisit) | stability_strict (thr=50) | ❌ FAIL Δ−0.75 | Confirms C12 stability filter is structurally bad on this engine. |
| 6 | C29 | har_rv | ❌ FAIL Δ−2.99 | DEAD (standalone +2.37 was a fluke). |
| 7 | C29 | hurst_feature | ❌ FAIL Δ−3.09 | DEAD. |
| 8 | C05 | cvar_objective | 🟦 STUB blocked | Needs per-trade-return list in grid CSV. |
| 9 | C06 | sortino_objective | 🟦 STUB blocked | Same blocker as C05. |
| 10 | C03 | shap_ranking | 🟦 STUB | Just needs `pip install shap`. |
| 11 | C07 | cusum_sampling | 🟦 architectural | Trader polling loop rewrite. |
| 12 | C09 | vol_entry_filter | ❌ DEAD 2026-05-04 | All 10 variants LOSE; best −1.22pp. Verified via [tools/test_c09_c10_overlay.py](tools/test_c09_c10_overlay.py). |
| 13 | C10 | btc_momentum_gate | ❌ DEAD 2026-05-04 | All 25 variants LOSE; best −7.79pp. Same script. |
| 14 | C04 | variance_ratio_detector | 🟦 STUB 30.6% bull | UNTESTED at full Mode S level. |

**Caveat**: harness batch ran after the engine `--grid-tag` fix (2026-05-03 evening). It's still worth confirming the IDEA-tagged grids were compared against CLEAN untagged baselines, not contaminated ones from the 2026-05-03 failed run. If contaminated, C23/C12-revisit/C29 FAIL verdicts could in principle flip. Verification command: `D ETH 5,6,7,8h --replay 1440 --no-persist --no-data-update` to refresh untagged baselines, then rerun.

Full harness reference (commands, idea list, decision rules) preserved below for re-running any single idea:

**Run any idea**:
```bash
python tools/test_14_ideas.py --list                       # show all 14
python tools/test_14_ideas.py --idea 1                     # by number
python tools/test_14_ideas.py --idea har_rv                # by name
python tools/test_14_ideas.py --idea 1,5,6,7               # subset
python tools/test_14_ideas.py --all                        # all sequentially (~5-7h)
python tools/test_14_ideas.py --idea 6 --quick             # --replay 720 for faster smoke
```

**The 14 ideas** (5 fully wired, 9 STUB with explicit blockers):

| # | Name | Category | Status | What it tests |
|---|---|---|---|---|
| 1 | per_regime_features | feature-set | **FULL** | Bias bull horizons toward technical features, bear toward macro |
| 2 | multi_horizon_exit | filter-overlay | STUB | Force exit when 5h AND 8h both flip SELL within 1h (#14 of original 20) |
| 3 | vol_scaled_4mo | detector | STUB | 4mo confirmation of `vol_2band` horizon picker (#12 of original 20) |
| 4 | vpin_5min | filter-overlay | STUB | Revive #3 VPIN filter at 5-min cadence (original was hourly) |
| 5 | stability_strict | feature-set | **FULL** | Revive #5 stability filter with threshold 50 instead of 30 |
| 6 | har_rv | feature-set | **FULL** | HAR-RV realized variance feature (Corsi 2009) |
| 7 | hurst_feature | feature-set | **FULL** | Rolling Hurst exponent as feature (Hurst 1951) |
| 8 | cvar_objective | scoring-change | STUB | CVaR_5% on returns (Rockafellar & Uryasev 2000) — sibling of #6 CDaR |
| 9 | sortino_objective | scoring-change | STUB | Sortino ratio scoring (downside-only vol) |
| 10 | shap_ranking | feature-set | **FULL** | SHAP value ranking instead of LGBM split-count importance |
| 11 | cusum_sampling | architectural | STUB | Event-based sampling (LdP AFML Ch 2) — needs trader rewrite |
| 12 | vol_entry_filter | filter-overlay | STUB | Skip BUYs at high realized vol (>p90 vs 30d) |
| 13 | btc_momentum_gate | filter-overlay | STUB | Allow ETH BUY only when BTC 24h momentum > 0 |
| 14 | variance_ratio_detector | detector | STUB | VR(q) regime detector (Lo & MacKinlay 1988) |

**Outputs**:
- `output/test_14_<idea>_<ts>.csv` — per-idea result CSV
- `logs/test_14_<idea>_<ts>.log` — per-idea console log
- `logs/test_14_summary_<ts>.txt` — aggregated verdict table

**Recommended execution order** (after CDaR HRST validates or fails):
1. Run the 5 FULL ideas first (~1.5-2h total): `python tools/test_14_ideas.py --idea 1,5,6,7,10`
2. Run the cheap audit STUBs (~5 min each): `python tools/test_14_ideas.py --idea 2,12,13,14` — these print what's missing + write partial-result CSVs you can inspect.
3. Decide whether to flesh out STUBs based on partial-result hints.

**🎯 MOST IMPORTANT IDEAS to test (ranked by expected value × testability)**:

**Tier 1 — Run these first (high prior probability of working, fully testable today)**:

1. **#3 vol_scaled_4mo** ⭐ HIGHEST EXPECTED VALUE — already proved +5.02pp on 2-month window in 2026-04-19 test. Promotion-blocked ONLY by missing 4-month confirmation. If 4mo confirms, this replaces `tsmom_672h` detector immediately.
   ```bash
   python tools/test_vol_scaled_horizon.py --replay 2880
   ```
   ETA ~30 min. Decision: if 4mo top variant beats `tsmom` baseline by ≥+5pp on Mode T REF → promote (config-only change, instant rollback).

2. **#1 per_regime_features** — never tested, low effort, real chance of helping. Bias bull horizons (5h/6h) toward technical features, bear horizons (7h/8h) toward macro features. Asymmetry already happens organically; question is whether forcing it beats organic.
   ```bash
   python tools/test_14_ideas.py --idea 1
   ```
   ETA ~30 min. Decision: smoke-test win ≥+5pp avg APF delta across horizons → run validation HRST.

3. **#6 har_rv** — better realized-variance estimator (Corsi 2009 HAR-RV). Vol features (`volatility_48h`, `gk_volatility_48h`) are already Grade 4 winners; a better vol estimator could lift them.
   ```bash
   python tools/test_14_ideas.py --idea 6
   ```
   ETA ~20 min. Decision: same +5pp threshold.

**Tier 2 — Cheap audit STUBs (run while Tier 1 is going on a different machine)**:

4. **#13 btc_momentum_gate** — skip ETH BUYs when BTC 24h momentum < 0. Cross-sectional confirmation. ~5 min audit using existing 90d cache.
   ```bash
   python tools/test_14_ideas.py --idea 13
   ```
   Output: `output/idea_13_btc_momentum_gate_<ts>.csv`. If many BUYs would be skipped during clear bear-leading periods → flesh out simulation.

5. **#12 vol_entry_filter** — skip BUYs when realized vol > 30d p90. ~5 min audit.
   ```bash
   python tools/test_14_ideas.py --idea 12
   ```
   Output: `output/idea_12_vol_filter_<ts>.csv`. Inspect skip count vs total BUYs; if many → simulate strategy delta.

6. **#2 multi_horizon_exit** — force exit when 5h AND 8h both flip SELL within 1h. Audit-only (cache exists).
   ```bash
   python tools/test_14_ideas.py --idea 2
   ```
   Output: `output/idea_02_multi_horizon_exit_<ts>.csv`. Check overlap rate; if 5-15% of bars → worth simulating, if <2% or >30% → not actionable.

**Tier 3 — Lower priority (run only if Tier 1 + 2 are exhausted or look promising)**:
- **#5 stability_strict**, **#7 hurst_feature**, **#10 shap_ranking**: feature-set tweaks; testable end-to-end via the harness.
- **#14 variance_ratio_detector**: writes regime label CSV; needs detector registration + Mode S to fully test.

**Tier 4 — Don't bother until something else works** (architectural / require new plumbing):
- #4 vpin_5min, #8 cvar_objective, #9 sortino_objective, #11 cusum_sampling

**Decision rule across all tiers**: Mode D smoke test win ≥+5pp average APF delta vs untagged baseline → promote to full HRST validation. Anything below +5pp is HRST run-to-run noise (5-10pp).

**Cross-machine parallelism opportunity**: Tier 1 #1 and #3 can run simultaneously (Tier 1 #3 on Desktop while Tier 1 #1 on Laptop, or vice versa) — both are independent monkey-patches writing to differently-tagged outputs. Don't run two harness invocations on the same machine concurrently — Mode D's parallel workers would oversubscribe the GPU/CPU.

**STUB blockers** (for follow-up work if any STUB looks promising):
- #2 multi_horizon_exit: needs `data/eth_per_horizon_signals_90d.pkl`
- #3 vol_scaled_4mo: shells out to existing `tools/test_vol_scaled_horizon.py --replay 2880`
- #4 vpin_5min: needs 1m OHLCV cache (run `tools/download_1m_data.py` first)
- #8/#9 cvar/sortino: need per-trade-return list in grid CSV (engine writes summary stats only — extend grid CSV writer)
- #11 cusum_sampling: architectural rewrite of trader polling loop, not a smoke test
- #12 vol_entry_filter: needs `data/eth_sl_signals_90d.pkl` (already exists)
- #13 btc_momentum_gate: same cache + `data/btc_hourly_data.csv`
- #14 variance_ratio_detector: writes regime label CSV; full test requires registering as named detector and running Mode S

---

### 🐛 HARNESS BUG FOUND + ENGINE FIX SHIPPED (2026-05-03 22:13 CEST)

**First harness run** — `python tools/test_14_ideas.py --idea 1` — completed Mode D for ETH 5,6,7,8h cleanly (31.4 min total, all 4 horizons produced grid winners), but the harness reported `status=ERROR / No grid winners found` because the harness expected per-idea-tagged grid CSVs (`crypto_ed_grid_ETH_5h_IDEA01.csv`) and the engine wrote untagged paths (`crypto_ed_grid_ETH_5h.csv`).

**Root cause**: harness passed `--grid-tag IDEA01` to the engine via `subprocess.run`, but the engine had **zero references** to `grid-tag` / `grid_tag` / `GRID_TAG`. Flag was silently ignored, untagged path was used, harness's `load_grid_csv(asset, h, 'IDEA01')` couldn't find the file.

**Compounding**: the patched (per_regime_features-monkey-patched) Mode D run **overwrote** the untagged baseline grids that `load_baseline_grid()` reads for comparison. Untagged grids on disk are now contaminated.

**Engine fix shipped** (`crypto_trading_system_ed.py`):
- Line 401: added `GRID_TAG_SUFFIX = None` module global
- Line 4462-4465: write path now uses `_{GRID_TAG_SUFFIX}` suffix when set
- Line 7091+: added `--grid-tag NAME` parser (alphanumeric/underscore validation)
- Line 7309: added `--grid-tag` to `skip_next` set so positional-arg parser ignores its value

Verified flag end-to-end: `[--grid-tag IDEA01] Mode D grid CSV will be written to crypto_ed_grid_<asset>_<h>h_IDEA01.csv` + module global propagates to write site.

**Recovery plan** (do these in order, ~60 min total):

1. **Refresh untagged baselines** (~25-30 min):
   ```
   python crypto_trading_system_ed.py D ETH 5,6,7,8h --replay 1440 --no-persist --no-data-update
   ```
   Overwrites contaminated `crypto_ed_grid_ETH_<h>h.csv` files with clean (un-patched) data.

2. **Re-run idea #1 with the fix** (~30 min):
   ```
   python tools/test_14_ideas.py --idea 1
   ```
   Now writes to `_IDEA01.csv` (tagged), compares against clean untagged baselines from step 1.

**Lessons for future ideas tested via this harness**:
- Always run a clean `D ETH ...h --replay 1440 --no-persist --no-data-update` first to refresh untagged baselines, especially after any patcher run that touched `_test_lgbm_importance` or `generate_signals`.
- Harness writes IDEA-tagged CSVs that are independent of the untagged baseline. So baseline contamination is per-asset/per-horizon — only the touched (asset, horizon, replay) combos need refreshing.
- All 5 FULL ideas in the harness (#1, #5, #6, #7, #10) use the same `compare_grid_winners(test_df, base_df)` pattern → same fix benefits all of them.

---

### 📋 CANONICAL IDEAS SCOREBOARD (built 2026-05-04 — single source of truth)

**Why this exists**: previously two parallel numberings — the "20-ideas roadmap" (built 2026-05-03) and the "14-ideas harness" (built 2026-05-03 evening) — created repeated confusion ("which idea is #6? CDaR or har_rv?"). This canonical list deduplicates, renumbers, and orders by status: **Untested → Partial → Error → Shelved → Dead**. The 20-roadmap and 14-harness numberings are kept as columns for backward reference; **always cite canonical CID# going forward**, with optional `(R20=#x H14=#y)` annotation when needed.

**Total**: 29 unique ideas across both lists. The 20-roadmap is preserved verbatim as a separate section below for historical context; this canonical list supersedes it for status-tracking.

#### 🟦 UNTESTED (8) — no result yet, ranked by expected value

| CID | Idea | R20 | H14 | Effort | Notes |
|---|---|---|---|---|---|
| **C01** | Vol-scaled horizons 4-month validation | #12 | #3 | ~30 min | 2mo +5.02pp was over now-defunct `tsmom_672h` baseline AND within HRST run-to-run noise band. Real test is fresh 4mo on current live's `sma24>sma100`. `python tools/test_vol_scaled_horizon.py --replay 2880` |
| ~~C02~~ | ~~Multi-horizon emergency-exit ensemble~~ | #14 | #2 | — | **MOVED TO DEAD 2026-05-04 late evening** — all 16 variants LOSE, best −3.30pp. See DEAD section. |
| ~~C03~~ | ~~SHAP ranking~~ | — | #10 | — | **MOVED TO DEAD 2026-05-04 evening** — Δ−4.78 avg, see DEAD section. |
| ~~C04~~ | ~~Variance Ratio detector~~ | — | #14 | — | **MOVED TO DEAD 2026-05-06 evening** — Δ−1.21 avg APF on fixed harness. |
| **C05** | CVaR_5% objective scoring | — | #8 | tonight | Bug fixed 2026-05-07, bundled into tonight's `test_desktop_5ideas_runner.py` overnight. |
| **C06** | Sortino objective scoring | — | #9 | tonight | Same backtest as C05 (zero marginal cost). Joint rerank csv for manual review. |
| **C07** | CUSUM event-based sampling (LdP AFML Ch 2) | — | #11 | days | Architectural rewrite of trader polling loop. Not a smoke test. |
| ~~C08~~ | ~~Single-horizon-only CDaR variant~~ | — | — | — | **MOVED TO DEAD 2026-05-06 evening** — 5h −20.48pp / 6h −14.05pp vs APF prod. Same shape as C13 HRST fail (May-4); per-horizon variant doesn't escape the gap. |

#### 🟡 PARTIAL (1) — tested in limited scope, validation pending

| CID | Idea | R20 | H14 | What's done | What's missing |
|---|---|---|---|---|---|
| (already C01 above) | (vol-scaled 4mo subsumes the partial entry — listed once in UNTESTED for the 4mo run) | | | | |

(The original partial-list slot for #12 is folded into C01 above; running `--replay 2880` resolves both the "untested 4mo" and "partial 2mo→4mo" framings simultaneously.)

#### 🐛 ERROR / BLOCKED ON BUG — RESOLVED 2026-05-04

Both C09 and C10 unblocked. Harness bug fixed (the cache is a list-of-dicts, not a dict; `cache.get('signals', cache)` was failing on a list). Full PnL overlay then run via [tools/test_c09_c10_overlay.py](tools/test_c09_c10_overlay.py). **Both DEAD.** See moved entries in DEAD section below (CIDs preserved, status flipped).

#### ⚪ SHELVED (6) — tested + lost, but explicit revisit-with-conditions path exists

| CID | Idea | R20 | H14 | Result | Revisit condition |
|---|---|---|---|---|---|
| **C11** | VPIN entry filter | #3 | #4 | Best +3.83pp 60d (below +5pp ship) | 5-min sub-loop OR 90d/6mo with stress events OR combine with vol filter |
| **C12** | Feature stability filter (Carhart-style) | #5 | #5 | ⚠️ **SUSPECT (harness bug 2026-05-05)**. Original thr=30 was a standalone test (May 3) and may be valid; thr=50 (May 4) ran via the broken harness — needs retest. | Threshold>50 AND 180d training window |
| **C13** | CDaR objective scoring | #6 | — | Smoke-test win 2026-05-03; HRST fail 2026-05-04 on BOTH machines (laptop −14.97pp, desktop −16.49pp vs live) | Single-horizon-only variant = C08; OR lambda sweep on Desktop's HRST |
| **C14** | Triple-barrier method as exit overlay | #7 | — | Best `up=6σ lo=2σ vert=24h conf=90`: 30d +6.77pp / 60d +10.48pp / 90d +1.24pp — 60d gain didn't survive | Different vol regime |
| **C15** | Meta-labeling secondary classifier | #9 | — | Lost on strong primary (E vs A −2.12pp); helped on weak primary but still lost overall | Retest on SOL/BTC/XRP primaries (those assets currently shelved) |
| **C16** | SL/TP/trailing-stop variants | #16 | — | Baseline (no SL) won every dimension; one variant viable: −5 to −7% disaster brake (fires 0 times in 60d, dormant) | User keeps disaster brake disabled; revisit only on new vol regime |

#### 🔴 DEAD (13) — tested + closed, no revisit path

| CID | Idea | R20 | H14 | Why dead |
|---|---|---|---|---|
| **C17** | Fractional differentiation features | #1 | — | LGBM didn't pick any `fd_*` into top tier; ETH hourly may have low long-memory; literature was for daily/weekly equity |
| **C18** | Probability calibration (Platt + isotonic) | #2 | — | Architectural mismatch with binary all-in/all-out engine; Mode S empirically tunes threshold so calibration is cosmetic relabeling |
| **C19** | Turbulence Index kill-switch | #4 | — | Directional sign WRONG on ETH bull regime; high-turb days were FOLLOWED by good ETH outcomes (opposite of Kritzman & Li) |
| **C20** | Triple-barrier as TRAINING LABEL | #8 | — | BTC same-week head-to-head: lost on return (−28pp / −51pp), WR (−9pp / −10pp), accuracy (−16pp / −13pp) |
| **C21** | Asymmetric loss / class weights | #10 | — | All weights produced identical ~3 trades at 95%/85% conf threshold; high-conf threshold already filters aggressively |
| **C22** | Kelly criterion position sizing | #11 | — | Architectural mismatch; engine has no position-size gradient to exploit (all-in at 85-95% conf) |
| **C23** | Per-regime feature set | #13 | #1 | ⚠️ **SUSPECT (harness bug 2026-05-05)** — original Δ−1.18 reading is meaningless. Needs retest. |
| **C24** | 5-min emergency-exit price-action overlay | #15 | — | No clean combo (max precision 12.5% — 42 false positives per real crash in 60d); false alarms eat alpha |
| **C25** | Multi-timeframe fusion / TabPFN / CPCV | #17 | — | All variants dominated by single-timeframe LGBM; CPCV incompatible with temporal decay (gamma weighting) |
| **C26** | LSTM as ensemble partner | #18 | — | LSTM solo failed; LSTM+LGBM ≡ RF+LGBM (LSTM votes randomly). LGBM dominance confirmed |
| **C27** | PySR for regime labels | #19 | — | Best regime accuracy 58% — too weak. Hand-crafted detectors win the joint sweep. (PySR alive as FEATURE synthesis — pysr_1..5 Grade 4) |
| **C28** | GDELT geopolitical features | #20 | — | iran_vol_zscore ranked #9 once but selected into 0/33 production models. VIX + 1d equity changes capture macro fear faster |
| **C29** | HAR-RV + Hurst feature additions | — | #6, #7 | ⚠️ **SUSPECT (harness bug 2026-05-05)** — original Δ−2.99 / Δ−3.09 readings are meaningless. Both need retest. |
| **C09** | Vol entry filter (skip BUYs at high realized vol) | — | #12 | **Tested 2026-05-04 evening** ([tools/test_c09_c10_overlay.py](tools/test_c09_c10_overlay.py)). All 10 variants LOSE vs baseline +147.59%. Best (p95 / 720h, 1 skip): −1.22pp. Worst (p70 / 168h, 73 skips): −62.7pp. Model's BUY signal is good even at high realized vol — shield+max_hold already handle drawdown. Filtering BUYs by vol percentile costs alpha on every variant. |
| **C10** | BTC momentum gate (allow ETH BUY only when BTC 24h mom > 0) | — | #13 | **Tested 2026-05-04 evening** (same script). All 25 variants LOSE vs baseline. Best (6h / −0.5% threshold, 61 skips): −7.79pp. Audit-suggested 24h / 0.0% threshold: **−85.84pp catastrophic**. Confirms hypothesis: ETH BUYs during BTC negative momentum ARE the good entries (mean-reversion setups). Cross-sectional gating destroys the alpha. |
| **C03** | SHAP feature ranking | — | #10 | ⚠️ **SUSPECT (harness bug 2026-05-05)** — patcher import-time print fired but runtime patches never reached Mode D's `_test_lgbm_importance`. Δ−4.78 reading is meaningless. Needs retest with fixed harness. |
| **C31** | Funding rate momentum / acceleration features (chg6h, chg72h, accel) | — | — | ⚠️ **SUSPECT (harness bug 2026-05-05)** — same `runpy.run_module` issue. Δ−5.25 reading is meaningless. Needs retest. (Note: this script `test_c31_funding_momentum.py` reuses the harness's `run_mode_d` helper — same bug.) |
| **C02** | Multi-horizon emergency-exit ensemble (h5+h8 both SELL → force exit) | #14 | #2 | **Tested 2026-05-04 late evening** ([tools/test_c02_multi_horizon_exit.py](tools/test_c02_multi_horizon_exit.py)). All 16 variants LOSE vs aligned-window baseline +137.50%. Best (W=0h thr≥90, 2 fires): −3.30pp. Worst (W=2h thr=0, 86 fires): −68.71pp. Same family as C09/C10/C24 (rejected 5-min): the model's existing exit timing is already good — emergency exits sell winners 1-3h early during natural drawdowns. Shield + max_hold already handle crashes. |

#### Status summary (updated 2026-05-07 morning — post C04/C56/C08 retests)

- **Untested actionable** (from C01-C29): 3 (C01, C05, C06) — C04 closed DEAD, C07 architectural-skip, C08 closed DEAD
- **Bug-blocked**: 0
- **Shelved (revisit-conditional)**: 6 (C11-C16)
- **🟡 SUSPECT** (harness ran but patches didn't fire — needs retest): **11** (was 14)
  - Validated DEAD on fixed harness 2026-05-06: C56, C04
  - Tonight (overnight 2026-05-07 → 2026-05-08): C35, C57, C50, C47, C44, C42, C43 + C05/C06 rerank → expected to drop SUSPECT pool to 5 (C03, C12, C23, C29, C31 — Tier C low priors)
- **Dead (genuinely closed, reliable)**: C02, C04, C08, C09, C10, C13, C17, C18, C19, C20, C21, C22, C24, C25, C26, C27, C28, C56 = **18 ideas**
- **C60-C82 archive-recovered**: 23 candidates (all UNTESTED — clean)
- **Total canonical IDs catalogued**: 82

#### Pattern observed this week — REVISED with harness bug awareness

The "every feature-addition fails" claim is now WEAKER — at least 9 of those failures (C03, C23, C12, C29, C31, plus 4 overnight) had a broken harness. **The pattern may not hold once the harness is fixed.** The reliable failures (C09 vol-entry, C10 BTC momentum, C24 5-min PA exit, C02 multi-horizon exit) were ALL exit/entry filters tested via standalone overlay sims — those still genuinely failed because they correctly tested the filter logic against the full live trader policy stack.

**Revised top priorities post-fix**:
1. Fix the harness (~15 min)
2. Verify fix with C56 HMM (~35 min) — Tier S empirical positive prior
3. Re-run remaining 13 SUSPECT verdicts on fixed harness if C56 retest produces a different result than overnight

#### Working priority order (top untested) — REVISED 2026-05-05 morning

**STEP 0 — BLOCKING**: Fix harness bug in [tools/test_14_ideas.py](tools/test_14_ideas.py) — replace `runpy.run_module(...)` with direct `main()` call (~15 min). Without this, every harness-driven test produces meaningless deltas.

**STEP 1 — Validate fix** (~35 min): Re-run C56 HMM (single idea). Confirm `[C56] HMM regime features added` appears in log during build_all_features. Compare delta to overnight FAIL (-5.34) — if materially different, fix is working.

**STEP 2 — Re-run SUSPECT verdicts** (~6-8h on Desktop, only if C56 retest changes verdict):
- ⭐ Tier S retests: C56 HMM, C35 wavelets (positive prior on prior engine)
- Tier A retests: C42 CatBoost, C43 Stacking, C04 VR, C57 MS-AR, C50 PF, C44 Quantile, C47 vol-adj label
- Tier C retests (probably stay FAIL): C03 SHAP, C23 per-regime, C12 stability, C29 HAR-RV+Hurst, C31 funding momentum

**STEP 3 — Untested ideas with no harness dependency** (run anytime in parallel):
1. **C01** Vol-scaled 4mo — 30 min, config-only ship if pass — **never actually launched yet**
2. **C08** Single-horizon CDaR variant — 2h, open follow-up after C13 HRST fail
3. **C16 narrow** Disaster brake at −5% retest — 30 min, current max DD 5.29% so brake fires now

#### How to cite ideas going forward

✅ "C01 (vol-scaled 4mo)" — preferred, unambiguous
✅ "C13 (CDaR — R20#6 / no harness entry)" — when you need the legacy mapping
❌ "Idea #6" alone — ambiguous between R20 (CDaR) and H14 (har_rv)
❌ "the new ideas" / "the harness ideas" — drift-prone

---

### 📋 PROPOSED IDEAS C30-C59 (added 2026-05-04 evening)

30 new candidate ideas, vetted as applicable to the engine and not previously tested. Pattern caveat: every feature-addition idea this week failed; **structural-change ideas are the highest-prior candidates** (marked ⭐ below).

#### Feature engineering (C30-C41) — ⚠️ near-100% prior-FAIL given week's pattern

| CID | Idea | Status | Notes |
|---|---|---|---|
| ~~C30~~ | Order Flow Imbalance (delta volume) | rejected by user 2026-05-04 | Not applicable per user (binary all-in/all-out) |
| ~~C31~~ | Funding rate momentum / acceleration | **DEAD** | Tested 2026-05-04; Δ−5.25 avg APF |
| C32 | Liquidation cascade event features | proposed | Binance liquidation feed; not currently downloaded |
| C33 | Bid-ask spread compression z-score | proposed | Distinct from quarantined raw spread_bps |
| C34 | ETH-BTC cointegration residual | proposed (feature) | Engle-Granger residual, distinct from lead-lag. **Standalone pairs-trading-strategy variant tested 2026-06-10 → C84 DEAD** (−100%, pair trends not mean-reverts). |
| C35 | Wavelet multi-scale decomposition | proposed | `pywavelets`, 3-4 scale coefficients |
| C36 | News/social sentiment polarity (CryptoPanic + Santiment) | proposed | Polarity, not event volume (different from C28 GDELT) |
| C37 | Stablecoin issuance velocity | proposed | First difference of mcap (level was C28-class dead) |
| C38 | Korea / Coinbase premium spread | proposed | Bithumb public API |
| C39 | Long/short ratio (Binance traders) | proposed | Binance Futures API |
| C40 | Rolling skewness / kurtosis features | proposed | 3rd/4th moments, currently absent |
| C41 | Yield curve features (US10Y−US2Y, US10Y−US3M) | proposed | FRED data; current macro has only level series |

#### Model architecture (C42-C46) — ⭐ structural changes, higher prior

| CID | Idea | Status | Notes |
|---|---|---|---|
| ⭐ **C42** | CatBoost as 4th ensemble model | proposed | Different regularization than XGB; native categorical handling |
| ⭐ **C43** | Stacking meta-learner (LR over RF+LGBM+XGB) | proposed | Different from C15 meta-labeling (C43 changes primary signal) |
| **C44** | Quantile regression target | proposed | LGBM `objective='quantile', alpha=0.7` — tail-aware loss |
| **C45** | Conformal prediction calibrated intervals | proposed | Distinct from C18 Platt (intervals not probability mapping) |
| **C46** | Bayesian hyperparameter optimization (BoTorch GP) | proposed | Replace Optuna TPE sampler |

#### Label / objective (C47-C51)

| CID | Idea | Status | Notes |
|---|---|---|---|
| **C47** | Vol-adjusted label (`ret_h / σ_h > threshold`) | proposed | Distinct from C20 triple-barrier label |
| **C48** | Sharpe-aware label | proposed | `ret > thr AND vol_pre_entry < pX` |
| **C49** | Multi-class label (5 classes) | proposed (~1 day) | Strong/weak buy/sell + hold |
| **C50** | Profit factor as primary Mode V scoring | proposed | Replace `combined_score = ret × WR` |
| **C51** | Maximum Adverse Excursion (MAE) penalty | proposed (blocked) | Same blocker as C05/C06 (per-trade-return list in grid CSV) |

#### Exit / execution (C52-C55)

| CID | Idea | Status | Notes |
|---|---|---|---|
| C52 | ATR-based vol-adapted trailing stop | proposed | Distinct from C14 (static σ) and C16 (HWM-x%) |
| C53 | Volume-spike exit trigger | proposed | sell when vol_5m > 3 × 30d avg AND in profit |
| C54 | Time-decay sell threshold | proposed | min_sell_pnl decreases as hold_h grows |
| C55 | Liquidity-aware entry timing | proposed | delay BUY if last 5m vol < 30d p20 |

#### Regime detection (C56-C59) — ⭐ structural, higher prior

| CID | Idea | Status | Notes |
|---|---|---|---|
| ⭐ **C56** | HMM 2-state Gaussian regime detector | proposed | `hmmlearn`; unsupervised state-space model |
| ⭐ **C57** | Markov-switching AR detector | proposed | `statsmodels`; regime-dependent AR coefficients |
| **C58** | Yield-curve macro regime detector | proposed | depends on C41 features first |
| **C59** | Cluster-based regime (K-means on macro+vol) | proposed | unsupervised, multi-dimensional |

---

### 📋 ARCHIVE-RECOVERED IDEAS C60-C82 (added 2026-05-04 late evening)

Cross-checked against `archive/literature_v3_ideas.md` + `archive/testing_literature*.csv`. These were either:
- **KEPT** in the V3 lit (approved 2026-03 era for testing) but never executed; OR
- **PENDING** in the V3 lit (lower priority, never reviewed/executed); OR
- on the V3 list with prior tested-MIXED result on BTC + prior engine (different from current ETH + post-embargo-fix engine).

**Note on archive-tested ideas already in C01-C59**: 2 cross-list overlaps were tested in `archive/testing_literature_v2.csv` with MIXED results on BTC prior engine — these stay at their existing CIDs but flagged as "deserve current-engine retest":
- **C35 wavelets** — BTC 8h beat baseline 92%/52% (+40pp) but 4h crashed; on prior engine
- **C56 HMM regime** — BTC 8h beat baseline 98%/52% (+46pp) but 4h crashed; on prior engine

Old ideas already tested + DEAD on prior engine, NO retest needed: C17 fracdiff, C20 triple-label, C18 calibration (all already DEAD on current engine too).

#### KEPT in V3 lit, never executed — 9 ideas (C60-C68) — prior approval = mid-priority

| CID | Idea | V3 # | Effort | Notes |
|---|---|---|---|---|
| **C60** | US Market Hours Flag (binary NYSE 14:30-21:00 UTC) | #3 | ~1h | Calendar feature; handle DST |
| **C61** | Volatility of Volatility (std of vol_12h over 24h+48h) | #5 | ~1h | 2 features added |
| **C62** | DXY Acceleration (2nd derivative of DXY) | #6 | ~30 min | Uses existing m_dxy data |
| **C63** | KAMA Slope (Kaufman Adaptive MA) | #7 | ~1.5h | Must beat existing SMA features (sma20_to_sma50h Grade 4); else doesn't add value |
| **C64** | Ehlers Fisher Transform | #8 | ~1.5h | Must beat existing rsi_14h/stoch_k_14h/bb_position_20h |
| **C65** | Approximate Entropy (ApEn) | #9 | ~2h | Predictability via `nolds`; windows 48h+120h |
| **C66** | BTC Dominance ROC (CoinGecko) | #10 | ~2h | New macro feature; rate-of-change of BTC mcap dominance |
| **C67** | Connors RSI (RSI(3) + streak + percentile composite) | #12 | ~1h | Composite oscillator |
| **C68** | Anchored Expanding Window (methodology) | #13 | ~3-4h | Conflicts with rolling-window design — architectural |

#### PENDING in V3 lit, never executed — 14 ideas (C69-C82) — lower priority

| CID | Idea | V3 # | Effort | Notes |
|---|---|---|---|---|
| C69 | Chaikin Money Flow | #23 | ~1h | Volume-weighted accumulation/distribution |
| C70 | Lempel-Ziv Complexity | #24 | ~2h | Compression-based complexity of direction sequence |
| C71 | Fractal Dimension (Higuchi) | #25 | ~2h | Price path complexity |
| C72 | Temporal Cross-Validation (non-overlapping folds) | #28 | ~3h | Methodology; conflicts with current 3-fold rolling holdout |
| C73 | Dispersion Ratio (cross-sectional vol / avg vol) | #29 | ~1.5h | Cross-asset only meaningful with universe enabled |
| C74 | Equity Put/Call Ratio (CBOE) | #30 | ~2h | Contrarian sentiment from options market |
| C75 | Stablecoin Supply Ratio (BTC mcap / stable supply) | #34 | ~1h | Different angle from C37 (issuance velocity) |
| C76 | Active Address Momentum (30d/365d ratio) | #35 | ~2h | On-chain; CoinMetrics |
| C77 | Google Trends Momentum (ROC of "Bitcoin" search) | #36 | ~3h | New macro feature; trends API |
| C78 | Transfer Entropy (BTC→altcoin) | #37 | days | Computationally heavy; directional info flow |
| C79 | Multifractal Spectrum (MFDFA) | #38 | ~3h | Singularity spectrum width as regime indicator |
| C80 | Mutual Information Decay | #39 | ~2h | MI between returns at increasing lags |
| C81 | DCCA Detrended Cross-Correlation | #40 | ~3h | Scale-dependent correlation between assets |
| C82 | Hilbert Transform Dominant Cycle | #42 | ~3h | Adaptive oscillator parameter extraction |

### 📋 STRATEGY-CLASS DIVERSIFICATION C83-C86 (added 2026-06-10)

A **different axis** from C01-C82 (which are feature / model / label / detector tweaks to the **directional** ETH model). After every directional lever was exhausted (each fails the gated sim — the model's selectivity + high-conf gate + slow features IS the edge), the 2026-06-10 pivot was to test **orthogonal strategy CLASSES** that don't rely on directional prediction. Thorough crypto-quant landscape research → 3 classes backtested + 1 promoted to paper. Full narrative in [TODO.md](TODO.md) "Logged 2026-06-10". Tools: `tools/bt_basis_carry.py`, `tools/bt_statarb_eth_btc.py`, `tools/bt_xsec_momentum.py`, `tools/bt_trend_scanning_8h.py`, paper bot `tools/funding_carry_eth.py`.

| CID | Idea | Status | Evidence / revival condition |
|---|---|---|---|
| **C83** | **Basis / funding carry** (delta-neutral: long spot + short perp, harvest 8h funding) | 🟢 **ACTIVE — PAPER** (the winner) | ETH always-on (unleveraged, frictionless): **Sharpe 15.08, APY +6.17%, maxDD 1.97%, funding +81% of periods**; BTC Sharpe 23.8 / LINK 24.1 / XRP 13.5; SOL/BNB weak (BNB +funding only 18% → reverse carry). Lit net-Sharpe ~4.8 (frictionless inflates the headline) — still far above the directional Sharpe AND **uncorrelated**. Paper bot `tools/funding_carry_eth.py` live on Desktop since 2026-06-10 21:39 UTC (FREE Binance public data, restart-safe, single-instance lock). **NOT a scoreboard "close" — live status in [TODO.md](TODO.md).** NEXT: ~2-4wk forward record (ideally spanning a +funding stretch) → perp-venue decision → Phase-2 live build. |
| **C84** | **Stat-arb ETH-BTC pairs trading** (rolling cointegration hedge ratio + z-score entry/exit on the spread) | 🔴 **DEAD** | `tools/bt_statarb_eth_btc.py`: **−100%, Sharpe −3 to −6, WR 39-49%** across configs. Spread ADF p≈0.05 (borderline) — ETH/BTC **trends** (dominance regimes) more than it mean-reverts. Distinct angle from **C34** (cointegration residual as a *feature*, also negative Δ−3.96): both the feature-angle and the standalone-strategy-angle of ETH-BTC cointegration are dead on the current engine/data. (Crude per-bar sizing, but the sign is robust.) No revival path. |
| **C85** | **Cross-sectional momentum** (rank a coin universe by trailing return, long top / short bottom) | ⚪ **SHELVED** | `tools/bt_xsec_momentum.py`, 6-coin universe (ETH/BTC/SOL/LINK/XRP/BNB): best **Sharpe 0.73 @14d lookback** (matches the academic 2-4wk crypto-momentum finding), APY +34% but **maxDD 36-63%**; most configs negative. **Universe too thin** — the literature uses ~50 coins; 6 can't diversify the idiosyncratic risk. **Revival condition: rebuild on a ~30-50 coin universe** (needs hourly data for the wider set) before judging. |
| **C86** | **Trend-scanning label** (López de Prado AFML — label = sign of the most-significant forward t-stat trend, variable lookahead) | 🔴 **DEAD** | `tools/bt_trend_scanning_8h.py` (ETH 8h, gated 60d): **−8.64pp** vs the fee-aware label (more trades 89 vs 80, lower WR 58 vs 68). The permissive target dilutes the model's selectivity — same failure family as **C20** (triple-barrier as training label, DEAD): both replace the fee-aware label and both lose because selectivity (high-conf gate + fee-aware target) IS the edge. No revival path; the fee-aware label is confirmed well-matched. |
| **C87** | **Rally-cooldown revert-abort** (dynamic cd: cancel the fixed cooldown early if the rally that triggered it reverts to its origin price → re-enable BUYs, vs the blunt fixed 24h/14h window) | 🔴 **DEAD** | `tools/test_cooldown_revert.py` on the LIVE `sma48>sma100` setup (cached 5h/8h signals, faithful replica of `faye._sweep_rally_cooldown.simulate`): **V0 no-gate +22.91%** (87% WR) → **V1 fixed-cd gate +28.84%** (89% WR, blocks 23 chase-buys = +5.9pp, the gate earns its keep) → **V2 revert-abort +21.92%** — un-blocks only **2** of the 23 buys but those are *falling-knife* entries that erase the gate's entire edge, dropping it **below even no-gate**. Reverting-to-origin is NOT a bottom; the reversion usually continues, so the cooldown's conservatism (staying out the whole reversion, not just the peak) is the *point*. Separately, `cd=24h`(bull)/`14h`(bear) were Mode-T sweep winners over `CD_GRID=[6,8,…,48]` → shorter *fixed* cd was already tested + lost. Both ways of loosening the cd (dynamic-abort, shorter-fixed) lose. One window (recent 2mo); a looser give-back abort would un-block more knife-catches → worse, not tested. No revival path. |
| **C88** | **Psychological floors** (TA support: does price bounce off round-number levels and/or local pivot-low "floors"? feature-add idea) | 🔴 **DEAD** (Phase-1 gate failed) | Literature is REAL — Osler 2000/2003 (order-clustering: take-profit AT round numbers, stops just beyond), arXiv 2101.07410 (prior-bounce strengthens a level), crypto clustering Urquhart'17/Hu'19 — BUT it doesn't survive ETH's hourly clock. `tools/diag_psych_floors_eth.py` model-free Phase-1 event study (fwd 5h/8h return when price sits ≤0.5% above a floor vs baseline; **two windows: recent 365d AND all 8.8y/77k hrs**; split by regime + prior-bounce count). **Every measurable edge is ≤0:** (A) round numbers ON-FLOOR Δvbase NEGATIVE+significant — $100 −0.10%/−0.10% (t −4.3/−3.5), $500 bear −0.21%/−0.24% (t −3.0/−2.7): resting on a round level precedes a mild DROP, confirming Osler's own caveat that the reversal is **<1h intraday → spent by our 5-8h horizon**. (B) local pivot-low support (the *recommended* primary) ALL near-support Δ −0.05%/−0.07% (t −3.0/−3.4); prior-bounce stratification **INVERTS** the literature — the dominant 2+ bucket (92% of events) Δ −0.06%/−0.08% (t −3.2/−3.7), the lone positive flicker (prior-bounce-1 bull +0.08/+0.14%) is **sub-threshold** (|t|<2, Δ<+0.3%). **Mechanism:** "near a floor from above" is a selection-biased *downtrend* sample → at 5-8h the drift outweighs any micro-bounce; "tested support is weak support" (each touch consumes resting bids). **Resolution-wall family** — same root as the fast-spikes / intra-hour-TP / order-flow-nervosity DEADs (sub-hourly bounce-chasing already proven to lose). **Honest caveat:** a study on hourly CLOSES can't see an intra-hour wick-and-recover — but the engine has already proven sub-hourly capture loses, so the gate is final *for this engine*. **Revival path:** only if the engine ever moves to sub-hourly bars (Ein/Eli) AND sub-hourly capture is independently shown profitable (currently DEAD). No Phase-2 gated A/B run — Phase-1 gate (no class clears Δ>+0.3% with \|t\|>2 *positive* in any regime, on 2 windows) is the stop. |

#### Recommended priority order (FULL — combining C01-C82 untested + proposed)

1. **C01** vol_scaled_4mo — running on user's terminal
2. ⭐ **C56** HMM detector — different regime mechanism, ~3h. **Prior**: BTC 8h beat baseline +46pp on prior engine; deserves current-engine retest.
3. ⭐ **C35** wavelet decomposition — feature-add, ~1.5h. **Prior**: BTC 8h beat baseline +40pp on prior engine.
4. ⭐ **C42** CatBoost — different model family, ~5h
5. ⭐ **C43** Stacking meta-learner — different ensemble strategy, ~6h
6. **C50** PF objective — different scoring metric, ~4h
7. **C04** Variance Ratio detector — already on untested list, ~2-3h
8. **C61** Volatility of Volatility — KEPT on V3 lit, never executed, ~1h
9. **C66** BTC Dominance ROC — KEPT on V3 lit, never executed, ~2h
10. **C47** Vol-adjusted label — KEPT on V3 lit (#2), never executed, ~4h
11. **C40** Skewness/Kurtosis — KEPT on V3 lit (#11), never executed, ~1h
12. **C44** Quantile regression target — tail-aware loss, ~4.5h
13. **C57** Markov-switching AR — different from C56 HMM, ~7h
14. **C60** US Market Hours Flag — cheap calendar feature, ~1h
15. **C63** KAMA Slope — must beat existing SMA features, ~1.5h

Items 8-11 reflect the "prior approval" boost from the V3 lit. They're not just untested — they were specifically approved-for-testing in the V3 era and never got run.

---

### 📊 20-IDEAS ROADMAP SCOREBOARD (built 2026-05-03 — superseded by canonical list above; kept for reference)

CFA-grade research roadmap. 20 academic-literature ideas applicable to this engine's architecture (binary all-in/all-out, hourly cycle, regime-switching ETH-only, no model persistence, LGBM-dominant). Ranked by impact at scoreboard-build time.

**Test methodology** (for ideas tested in this session): cheap monkey-patched Mode D smoke test, `--no-persist --no-data-update --replay 1440`, write to `_TAGGED` paths. Promote to full HRST validation only if smoke wins by ≥+5pp.

**Status as of 2026-05-03 19:50 CEST:**
- **Tested in THIS session (today, 2026-05-03):** ideas #1, #2, #3, #4, #5, #6 (six ideas)
- **Tested HISTORICALLY (pre-session, per CLAUDE.md audit trail):** ideas #7, #11
- **UNTESTED:** ideas #8, #9, #10, #12, #13, #14, #15, #16, #17, #18, #19, #20 (twelve ideas)

| # | Idea (impact-ranked) | Citation | What it does (mechanism) | Test status | Result / Verdict |
|---|---|---|---|---|---|
| **1** | **Fractional differentiation features** | López de Prado *AFML* Ch 5; Hosking 1981 *Biometrika* | Memory-preserving differencing. Computes `fd_*` features at fractional `d ∈ {0.3, 0.4, 0.5}` on log-close + log-vol. Integer differencing (`logret`) makes series stationary but kills long-memory; frac-diff retains memory while passing ADF stationarity tests. Hypothesis: top-3 selection-frequency features (`price_to_sma100h`, `logret_120h`, `sma20_to_sma50h`) would benefit. | **TESTED today** ([tools/test_fracdiff_mode_d.py](tools/test_fracdiff_mode_d.py), Mode D smoke 5,6,7,8h, 14:15-15:30 CEST) | **DEAD on ETH hourly.** All 5 `fd_*` features computed cleanly (88-100% coverage). LGBM importance did NOT pick any frac-diff feature into top tier on any horizon; grid winners did NOT include them. Original hypothesis disproved. Possible reasons: ETH hourly already has low long-memory in 60d window (Hurst<0.5); `price_to_sma100h` ratio partially preserves memory; literature documents frac-diff for DAILY/WEEKLY equity series, not hourly crypto. **Do not retry on ETH hourly without changing d-range or window.** |
| **2** | **Probability calibration** (Platt + isotonic) | Platt 1999 *Adv.Large Margin Classifiers*; Niculescu-Mizil & Caruana 2005 *ICML* | Re-maps raw LGBM scores to calibrated probabilities. Two-step test: Step A audited Brier score on 20 walk-forward cycles (raw 0.2774 → Platt 0.2285 = 17.6% improvement, model overconfident at high scores). Step B injected calibrator into Mode D's `generate_signals` to test impact on returns. | **TESTED today** ([tools/test_calibration_audit.py](tools/test_calibration_audit.py) + [tools/test_calibration_mode_d.py](tools/test_calibration_mode_d.py), 14:44-16:45 CEST) | **DEAD architecturally.** Step B catastrophic: returns +22-25% (4 horizons) vs prod baselines +46-67%, deltas −23 to −44pp. Trade counts collapsed 30-70 → 3-6. **Root cause**: in binary all-in/all-out engine where Mode S empirically tunes the confidence threshold, calibration is COSMETIC RELABELING — same trades fire whether threshold is 65% on raw or 50% on calibrated. AND Platt on imbalanced data (30% positives) squeezes probs toward base rate, killing trade frequency. Same family as Kelly (#11). Revisit only if tiered position sizing or EV-gated entries are added. |
| **3** | **VPIN entry filter** (Volume-synchronized PIN) | Easley, López de Prado, O'Hara 2012 *RFS* | Estimates informed-trading toxicity in flow via volume buckets. Higher VPIN = more adverse selection. Test: skip BUYs when `VPIN > threshold`. Sweep over `lookback ∈ {30m, 1h, 2h, 4h}` × `threshold ∈ {0.3, 0.4, 0.5, 0.55}`. | **TESTED today** ([tools/test_vpin_filter.py](tools/test_vpin_filter.py), 15:47 CEST) | **SHELVED.** Best variant `lb=30m thr=0.5`: +83.22% vs baseline +79.39% = **+3.83pp** (15 BUYs skipped). Below +5pp ship threshold; within HRST run-to-run nondeterminism (5-10pp). Aggressive thresholds (0.3-0.4) skipped 130-160 BUYs and lost 20-70pp. VPIN distribution: p50=0.40-0.41, p90=0.50-0.55 — no fat right tail of toxic days in 60d window. Revisit with: (a) 5-min sub-loop at native cadence (literature uses 1-min), (b) 90d/6mo window with known stress events, (c) combine with realized-vol filter. |
| **4** | **Turbulence Index kill-switch** | Kritzman & Li 2010 *FAJ* | Mahalanobis distance over multi-asset cross-section (BTC+VIX+DXY+SP500+GOLD+US10Y, 6-dim). 252d rolling cov + 252d z-score. Halt trading when turb_z > threshold. | **TESTED today** ([tools/test_turbulence_killswitch.py](tools/test_turbulence_killswitch.py), 15:49 CEST) | **DEAD on ETH bull regime.** All firing variants HURT: `z≥1.0` → −9.96pp (skipped 17 BUYs), `z≥1.5` → −5.83pp (skipped 9). Higher thresholds didn't fire (only 1 BUY day in 60d had `turb_z ≥ 2.0`). **Killer finding**: high-turb days in 60d were FOLLOWED by GOOD ETH outcomes — opposite of Kritzman & Li's premise. Mechanism: in crypto bull regimes, cross-asset stress spikes (VIX up, equities down) coincide with crypto dip-buying setups; model BUYs during turbulence were RIGHT, not WRONG. **Directional sign is wrong.** May still be useful as a FEATURE (not kill-switch) in different macro phase. |
| **5** | **Feature stability filter** (Carhart-style robustness) | Carhart 1997 *J.Finance* (4-factor stability principle, applied here to feature ranking) | Pre-filter time-unstable features before LGBM uses them. Compute LGBM ranking on 3 overlapping sub-windows (oldest 50%, middle 50%, newest 50%); drop features where `max_rank − min_rank > 30` positions; return ranking on full data using only stable features. Hypothesis: HRST 5-10pp run-to-run variance comes from LGBM making feature choices on noise. | **TESTED today** ([tools/test_stability_filter_mode_d.py](tools/test_stability_filter_mode_d.py), 16:03 CEST) | **DEAD.** Mode D ETH 5,6,7,8h vs un-filtered baseline: APF deltas mixed (3/4 horizons slightly worse, 1 better, all within ±2). Return deltas mostly negative or noise (5h −5.6pp, 6h −0.9pp, 7h −4.9pp, 8h −0.2pp). Winning configs use SAME or MORE features (f=13-25 unchanged). **Why it died**: (a) LGBM bagging already does this internally via tree splits, (b) sub-window ranking itself is noisy at 50% data, (c) regime-dependent importance is REAL signal — feature important in bull-not-bear is regime-specific info, not noise. Revisit only with stricter threshold (>50) + longer window (180d). |
| **6** | **CDaR objective scoring** | Chekhlov, Uryasev, Zabarankin 2005 *Mathematical Finance* | Replace `combined_score = ret × WR` with `score = (ret × WR) − λ × max_dd_pct`. CDaR_α = expected drawdown given DD is in worst α%. Aligns Mode V scoring with R4 rollback rule's max-DD threshold (rollback fires on >10pp DD). Default `λ=1.0`. | **TESTED 2026-05-03 smoke + 2026-05-04 HRST** ([tools/test_cdar_audit_only.py](tools/test_cdar_audit_only.py) + [crypto_trading_system_ed_cdar.py](crypto_trading_system_ed_cdar.py)) | **SHELVED at HRST level.** Smoke test 2026-05-03: CDaR top-1 STRICTLY DOMINATED APF top-1 on every horizon (5h: ret +35 vs +12, 6h: ret +45 vs +31, 7h: ret +49 vs +21, 8h: ret +20 vs 0). Engine fork shipped 2026-05-03 evening. **HRST validation 2026-05-04 (laptop ~9h)**: Mode T total **+71.22% vs current live +86.19% = −14.97pp UNDER threshold.** Mode S WINNER `sma24>sma100 5h@75%/8h@65%` does not exploit CDaR's strongest single-horizon win (8h Refined #1 +67.03%); regime sweep prefers shorter bull (5h) over CDaR-favored long-horizon configs. Smoke test win didn't translate to HRST. **DO NOT PROMOTE.** Open follow-up: single-horizon-only CDaR scoring (no regime split) untested. |
| **7** | **Triple-barrier method as exit overlay** | López de Prado *AFML* Ch 3 | Vol-adaptive barriers (upper σ × lower σ × vertical horizon h) replace model SELL + shield + max_hold. Exit on whichever barrier is hit first. | **TESTED HISTORICALLY** (2026-04-26, [tools/test_t6_triple_barrier.py](tools/test_t6_triple_barrier.py)) — NOT a session test | **SHELVED.** Best config (`up=6σ lo=2σ vert=24h conf=90`): 30d +6.77pp, 60d +10.48pp, 90d +1.24pp, 47 trades. 60d gain didn't survive to 90d. Not promotable, but not catastrophically wrong. |
| **8** | **Triple-barrier method as TRAINING LABEL** | López de Prado *AFML* Ch 3 | Replace `label = (ret > 2×fee)` with triple-barrier label (1 if upper barrier hit first, 0 otherwise). Aligns label with risk-adjusted realized outcome rather than raw return threshold. | **TESTED HISTORICALLY** (2026-03-14 BTC, archived in `archive/testing_literature.csv`) — NOT a session test | **DEAD.** BTC same-day baseline head-to-head: baseline 4h +57.22% / 8h +74.03% vs triple_barrier_label 4h +29.22% / 8h +22.53%. Lost on return (−28pp / −51pp), WR (−9pp / −10pp), accuracy (−16pp / −13pp). Lower-return label collapses signal. Earlier "+29% standalone" framing was misleading — same-day baseline beat it by 30-50pp. |
| **9** | **Meta-labeling secondary classifier** | López de Prado *AFML* Ch 3 | Train second LGBM on whether to ACT on primary BUY signals. Primary picks direction, meta picks size/skip. Hypothesis: filter false BUYs without retraining primary. | **TESTED HISTORICALLY** (AB matrix variants D + E, 2026-04-26/27) — NOT a session test | **SHELVED — door open.** R3 RESOLVED: E (meta on STRONG base) lost −2.12pp vs A. D (meta on WEAK base) gained +6.26pp but lost overall to A by −15.14pp. Conclusion: meta only helps when primary is broken. Door open for SOL/BTC/XRP retests (those assets shelved themselves for other reasons). |
| **10** | **Asymmetric loss / class weights** (`scale_pos_weight`) | Standard ML | Penalize false BUYs more than false SELLs via training-time class weighting. Sweep weights ∈ {0.3, 0.5, 0.7, 1.0, 1.3, 1.5}. | **TESTED HISTORICALLY** (2026-04-19 ETH 6h+8h, [tools/test_asymmetric_loss.py](tools/test_asymmetric_loss.py)) — NOT a session test | **DEAD at high conf thresholds.** All weights produced identical 3 trades (100% WR) on 6h. On 8h: weight=0.7 gained +0.27pp over baseline — noise. The 95%/85% confidence threshold already filters aggressively, so penalizing FPs at training time changes nothing. |
| **11** | **Kelly criterion position sizing** | Kelly 1956 *BSTJ*; Thorp *Beat the Dealer* | Size positions by `f* = (p×b − q) / b` where `p` = win prob, `b` = win/loss ratio. Replaces fixed all-in. | **EVALUATED HISTORICALLY** (2026-04-19, no backtest run) — NOT a session test | **DEAD architecturally.** Same family as #2 calibration. Architectural mismatch: Kelly sizes by confidence GRADIENT (95%=big, 65%=small), but engine only enters at 85-95% conf with NO position-size gradient to exploit. Would only matter if confidence thresholds were lowered. |
| **12** | **Volatility-scaled horizons** | Standard regime adaptation literature | Pick horizon based on 24h realized vol percentile vs 30d window. High vol → shorter 6h horizon (faster signal); low vol → longer 8h horizon (more confirmation). | **TESTED HISTORICALLY on 2-month, 4-month UNTESTED** (2026-04-19, [tools/test_vol_scaled_horizon.py](tools/test_vol_scaled_horizon.py)) — NOT a session test | **🟡 PROMISING.** 2-month: `vol_2band low→8h high→6h @90%` = +33.82% / 46 trades / 65% WR — beats current tsmom regime by +5.02pp and beats every single-horizon baseline. **4-month validation pending**: `python tools/test_vol_scaled_horizon.py --replay 2880` (~30 min). |
| **13** | **Per-regime feature set** | Open question (no specific citation) | Bull regime uses more technical features, bear uses more macro. Asymmetry already happens organically (6h has 7 features mostly PySR+technical; 8h has 32 features kitchen-sink) but never as deliberate design. | **TESTED 2026-05-04** (overnight harness, [tools/test_14_ideas.py](tools/test_14_ideas.py) idea #1) | **DEAD.** Mode D smoke ETH 5,6,7,8h: avg APF Δ−1.18 vs untagged baseline. Per horizon: 5h Δ+2.21, 6h Δ−1.82, 7h Δ−5.11, 8h Δ+0.00. The 7h horizon (which the May 1 8h HRST liked as bull) regressed hardest. **Hypothesis disproved** — forcing the bull→tech / bear→macro split costs alpha vs letting LGBM importance-rank the kitchen sink. |
| **14** | **Multi-horizon emergency-exit ensemble** | New idea (no formal citation) | Force exit when 5h AND 8h both flip SELL within 1h. Per-horizon signal cache `data/eth_per_horizon_signals_90d.pkl` already exists. Distinct from rejected 5-min price-action triggers. | **UNTESTED in current form** (T1b entry-side ensemble was tested + shelved 2026-04-27, but exit-side variant untested) | **UNTESTED.** Effort: 1-2h. Cache already exists. |
| **15** | **5-minute emergency-exit price-action overlay** | New idea (no formal citation) | Skip out fast on 5m crash signature (e.g., `ret_5m ≤ -0.7%` AND `d2_5m ≤ -0.3`). Goal: catch crashes before hourly cycle triggers. | **TESTED HISTORICALLY** (35 variants 2026-04-27, [tools/forensic_today_crash.py](tools/forensic_today_crash.py)) — NOT a session test | **DEAD on ETH bull regime.** Best `armed_at_pnl≥X% + thr=-2.0` fired 0 times in 60d. `-1.0%` threshold lost −23.66pp on 60d. Forensic of Apr 27 ETH crash: NO clean combo (max precision 12.5% — 42 false positives per real crash in 60d). False alarms eat alpha; hourly shield+max_hold+model SELL already handle 90% of crashes. |
| **16** | **Stop-loss / take-profit / trailing-stop variants** | Standard risk management | Various: fixed -2%, -5%, -7% disaster brake, +1% TP, trailing-stop at HWM-x%, profit-lock at +0.3%. | **TESTED HISTORICALLY** (8+ variants, 2026-04-14, [tools/test_t5_batch.py](tools/test_t5_batch.py) and earlier) — NOT a session test | **DEAD overall, ONE viable.** Baseline (no SL) won every dimension: +1.11% PnL / −8.71% DD vs all variants. Profit-lock variants (D/E/F): −11% to −20% PnL (chops fat tail). Trailing HWM variants (G/H): catastrophic. **Only viable**: −5% to −7% disaster brake as free insurance (fires 0 times in 60d, dormant). User keeps disabled. |
| **17** | **Multi-timeframe fusion / TabPFN / CPCV** | Multiple | Stack hourly + 4h + daily timeframe predictions / use tabular transformers / use Combinatorial Purged CV (López de Prado AFML Ch 12). | **TESTED HISTORICALLY** (multiple tests 2025-2026) — NOT a session test | **DEAD.** All variants dominated by single-timeframe LGBM. CPCV incompatible with temporal decay (gamma weighting). 4h overfits (PBO=1.0). |
| **18** | **LSTM as ensemble partner** | Standard DL | Add LSTM votes alongside RF+LGBM. Hypothesis: sequence model catches patterns trees miss. | **TESTED HISTORICALLY** (2026-03-26 V1.8 test) — NOT a session test | **DEAD.** LSTM solo: 0 valid results (all failed). LSTM+LGBM ≡ RF+LGBM (LSTM votes randomly, partner carries all signal). LGBM dominance confirmed. |
| **19** | **PySR for regime labels** | Cranmer 2023 *PySR* | Use symbolic regression to discover bull/bear regime detector formula instead of hand-crafted (sma24>sma100 etc). | **TESTED HISTORICALLY** (2026-03-29 Mode P regime label test) — NOT a session test | **DEAD as label, ALIVE as feature.** Best PySR regime accuracy 58% — too weak. Hand-crafted detectors (sma24>sma100, tsmom_672h, vol_calm) win the joint sweep. Keep PySR for FEATURE synthesis (where pysr_1..5 are Grade 4, used in 21-42% of production models). |
| **20** | **GDELT geopolitical features** | GDELT DOC 2.0 API + custom features | 21 features (iran_vol, geopolitical_tone, geopolitical_chg24h, iran_zscore, etc.). 5-10 min download per Mode D run. | **TESTED HISTORICALLY** (2026-04-10 added, 2026-04-19 disabled) — NOT a session test | **DEAD.** iran_vol_zscore ranked #9 once (LGBM importance) but selected into 0/33 production models across ETH/BTC/SOL/LINK/XRP. VIX + equity 1d changes capture macro fear faster (market-priced). Disabled both download and pipeline 2026-04-19. Code kept commented for future use. |

### Scoreboard summary

**This session (2026-05-03):**
- 6 ideas tested: #1, #2, #3, #4, #5, #6
- Wins: 1 (#6 CDaR — promoted to engine fork, validation HRST pending)
- Dead: 3 (#1 fracdiff, #2 calibration, #4 turbulence)
- Shelved: 2 (#3 VPIN, #5 stability filter)
- Hit rate today: **1/6** for ideas that produced a real ship-worthy signal

**Overnight harness session (2026-05-03 → 2026-05-04):**
- 5 fully-wired harness ideas tested: harness #1 (=scoreboard #13 per_regime_features), harness #5 (stability_strict, sibling of scoreboard #5), harness #6 (har_rv), harness #7 (hurst_feature), harness #10 (shap_ranking)
- 4 STUBs ran: harness #2 (multi_horizon_exit = scoreboard #14), #12 (vol_entry_filter), #13 (btc_momentum_gate), #14 (variance_ratio_detector)
- 1 HRST ran: scoreboard #6 CDaR validation
- Results: ALL 4 fully-wired ideas FAILED (Δ−0.75 to −3.09 vs baseline). #10 needs `pip install shap`. CDaR HRST FAILED (Mode T −15pp under live). #12+#13 ERROR'd on harness bug.
- Newly DEAD this morning: scoreboard #6 CDaR (HRST failed despite Mode D smoke win), scoreboard #13 per_regime_features (harness #1 FAIL)
- Net: 0 wins out of 6 attempted today.

**Lifetime status across all 20 (updated 2026-05-04 morning):**
- ✅ Ship-worthy: 0 (CDaR was the candidate; HRST validation FAILED; SHELVED)
- 🟡 Promising: 1 (#12 vol-scaled horizons — 4mo validation still untested, highest EV remaining)
- 🟦 Open / never tested: 1 (#14 multi-horizon emergency-exit ensemble — cache exists, 1-2h work)
- ⚪ Shelved (boundary or contextually conditional): 5 (#3 VPIN, #5 stability filter, #6 CDaR, #7 triple-barrier exit, #9 meta-labeling)
- 🔴 Dead (catastrophic or architectural mismatch): 13 (#1 fracdiff, #2 calibration, #4 turbulence, #8 triple-barrier label, #10 asymmetric loss, #11 Kelly, #13 per-regime features ← NEW, #15 5-min emergency-exit, #16 SL/TP variants, #17 multi-tf/TabPFN/CPCV, #18 LSTM, #19 PySR-as-regime-label, #20 GDELT)

**Next candidates to run**:
- **#12** Vol-scaled horizons 4mo confirmation — only remaining promising test. ~30 min.
- **#14** Multi-horizon emergency-exit ensemble — never tested, cache exists, 1-2h.
- Then: pivot to live-trader execution-gap work (~17pp unaccounted alpha) or Tier 4 architectural ideas (CUSUM sampling).

---

### 🟢 PROMOTED 2026-05-06 ~15:48 CEST — ETH live config flipped (today's HRST winner)

**Source**: ETH HRST 5,6,7,8h --replay 1440 --no-persist ([logs/ed_v1_20260506_085146.log](logs/ed_v1_20260506_085146.log), started 08:51 → finished 14:24, 5.5h on Desktop). Mode T converged iter 3 with both bull+bear gates active.

**Pre-promotion (May 3 winner)** → **Post-promotion (today's winner)**:
| Field | May 3 (was live until 2026-05-06 15:48) | 2026-05-06 (LIVE NOW) |
|---|---|---|
| Detector | `sma24>sma100` | **`tsmom_672h`** ← reverted |
| Bull | 6h@65% shield ON | **6h@75% shield OFF** |
| Bear | 5h@75% shield ON | **8h@65% shield OFF** |
| Bull gate | disabled | **rr18h≥5.0% OR rr30h≥7.5% cd=36h** ← now active |
| Bear gate | rr30h≥9% OR rr36h≥9% cd=48h | **rr24h≥4.0% OR rr30h≥6.0% cd=14h** |
| Max position USD | $14k | $14k (unchanged) |
| Mode T REF (sim 60d) | +86.19% | (read from log around iter 3) |

**Backup for rollback** (created automatically before promotion):
- [config/regime_config_ed.backup_20260506_pre_HRST_promote.json](config/regime_config_ed.backup_20260506_pre_HRST_promote.json)

**Rollback (one command)** — restores pre-2026-05-06 (= May 3 winner) config:
```
copy config\regime_config_ed.backup_20260506_pre_HRST_promote.json config\regime_config_ed.json
```

**Material policy deltas vs prior live (May 3)**:
1. **Detector flipped back to `tsmom_672h`** (28-day TS-momentum) from `sma24>sma100`. The `tsmom` detector won this HRST's Mode S joint sweep — different from the `sma24>sma100` plateau that won May 3.
2. **Both shields turned OFF** (was: bull ON, bear ON). Mode T's shield+gate joint sweep landed on no-shield + active-gate combo.
3. **Both gates now active**. Previous live had bull gate disabled and bear gate at long cooldown; now both gates contribute.
4. **Bull conf 65% → 75%**, **bear conf 75% → 65%** — confidence asymmetry inverted (high bull, low bear) — opposite of May 3.
5. **Bear horizon 5h → 8h** — material change (5h was rejected in plateau analysis on multiple recent HRSTs).

**Monitoring criteria** (1-2 weeks):
- Realized alpha vs sim Mode T REF over first 10 trades. If ≥15pp underperformance, rollback per CLAUDE.md R4.
- Max drawdown vs prior live's historical −10.02% on 60d.
- Both shields OFF means crashes hit the model-SELL exit timing only (no min_sell_pnl floor, no max_hold cap). Watch for premature loss-cutting OR runaway-loss scenarios that the shield previously caught.
- Both gates active means BUY frequency may drop materially during recent-rally periods. If trade count collapses below sim Mode T trade count, gates are over-blocking.

---

#### Audit-trail entry — superseded May 3 promotion (kept for context)

**May 3 → May 6 live config** was: `sma24>sma100 bull=6h@65% shield=ON / bear=5h@75% shield=ON, bear gate rr30≥9% OR rr36≥9% cd=48h`. Source: ETH HRST 5,6,7,8h --replay 1440 ([logs/ed_v1_20260502_201318.log](logs/ed_v1_20260502_201318.log), completed 2026-05-03 03:07). Mode S WINNER `sma24>sma100 6h@65%/5h@75% → +77.06% / 70 trades / 84% WR / alpha +60.37%`. Mode T REF +86.19%. Live for 3 days (2026-05-03 09:49 → 2026-05-06 15:48). Rollback to this config: `copy config\regime_config_ed_pre_sma24sma100_20260503.json config\regime_config_ed.json` (backup retained).

---

### (Original 2026-05-03 promotion notes — historical, kept for reference)

**Old live (pre-May-3)**: `tsmom_672h bull=6h@85% shield=OFF / bear=5h@65% shield=ON, bear gate rr30≥9% OR rr36≥9% cd=48h`
**New live (May 3 → May 6)**: `sma24>sma100 bull=6h@65% shield=ON / bear=5h@75% shield=ON, bear gate UNCHANGED (rr30≥9% OR rr36≥9% cd=48h)`

**Source**: ETH HRST 5,6,7,8h --replay 1440 ([logs/ed_v1_20260502_201318.log](logs/ed_v1_20260502_201318.log), completed 2026-05-03 03:07). Mode S WINNER: `sma24>sma100 6h@65%/5h@75% → +77.06% / 70 trades / 84% WR / alpha +60.37%`. TOP 15 plateau: top 7 entries unanimous on `sma24>sma100 bull=6h / bear=5h` — rock-solid plateau. Mode T converged iter 2: `bull_shield=ON, bear_shield=ON → +86.19%` (+9.14pp shield gain). Mode G found no STRICT-passing rally-cooldown winner, so existing live bear gate was preserved. Per-horizon Mode V: 6h winner (+67.14% / 75% WR) and 5h winner (+52.77% / 66% WR) both refreshed in production CSV.

**Backups for rollback**:
- `config/regime_config_ed_pre_sma24sma100_20260503.json`
- `models/crypto_ed_production_pre_sma24sma100_20260503.csv`

**Rollback (one command)**:
```bash
copy config\regime_config_ed_pre_sma24sma100_20260503.json config\regime_config_ed.json
copy models\crypto_ed_production_pre_sma24sma100_20260503.csv models\crypto_ed_production.csv
```

**State at promotion**: ETH position = `cash` (no open trade to disrupt). Live trader hot-reloads `regime_config_ed.json` every 5 min, no restart required. New config will take effect on next regime-triggered BUY signal or hourly cycle.

**Material policy changes (vs prior live)**:
1. **Detector**: `tsmom_672h` (28-day TS-momentum) → `sma24>sma100` (1d > ~4d SMA crossover). More responsive to short-term regime shifts.
2. **Bull conf**: 85% → **65%** (much lower) — accepts more BUY signals; expect ~2× the BUY frequency.
3. **Bull shield**: OFF → **ON** — bull regime now also uses `min_sell_pnl=0.5% / max_hold=10h` shield (previously only bear had it).
4. **Bear conf**: 65% → 75% (slightly higher) — slightly more selective bear-rally entries.
5. **Bear gate unchanged** — `rr30h≥9% OR rr36h≥9% cd=48h` preserved (Mode G found no STRICT replacement that beat baseline).
6. **Bull gate stays disabled** (no change).

**Monitoring criteria** (1-2 weeks):
- Realized alpha vs sim baseline +86.19% over first 10 trades. If >15pp underperformance, rollback per CLAUDE.md R4.
- Max drawdown vs prior live's historical -10.02% on 60d. If exceeded, investigate.
- BUY frequency: expect ~2× more BUYs than prior live due to bull_conf 85→65. If trade count explodes >3× sim baseline, gate is too loose.

**Per-asset non-impact**: BTC/SOL/LINK/XRP/BNB blocks left untouched. BNB HRST also completed overnight (`tsmom_672h 5h@90%/6h@90% → Mode T +20.58%, +2.92pp gate gain`, alpha ~24% of ETH per-$ — below 50% threshold → SHELVED, no enablement).

---

### 🔴 #1 PRIORITY — CDaR VALIDATION COMPLETE — SHELVED 2026-05-04 morning

**Status**: HRST validation completed on laptop overnight (started 2026-05-03 20:05, finished 05:13). **Mode T total +71.22% vs current live +86.19% = −14.97pp BELOW promotion threshold.** Per the decision rule below, this falls in the "Worse by >5pp = CDaR scoring hurts at the full HRST level" branch. **DO NOT PROMOTE.** See "CDaR HRST RESOLVED" section at top for full breakdown.

**Why it failed at HRST level despite winning Mode D smoke test**: smoke-test wins translated to per-horizon Mode V winners cleanly (8h Refined #1 +67.03% was the standout), but Mode S joint sweep picked `sma24>sma100 5h/8h` — NOT a regime that exploits the CDaR-favored long-horizon configs. The strongest CDaR result (8h) gets diluted in Mode S's preference for shorter bull horizons.

**Original smoke-test signal (kept for audit)**:
- 5h: ret +35.16% vs +12.12% (Δ +23.04pp)
- 6h: ret +45.46% vs +31.26% (Δ +14.20pp)
- 7h: ret +49.25% vs +21.22% (Δ +28.03pp), max_dd −7.45pp
- 8h: ret +19.78% vs −0.05% (Δ +19.83pp), max_dd −5.37pp

**Open follow-up (NOT a current priority)**: does CDaR scoring help at the SINGLE-HORIZON level even if Mode S/T washes out? 8h Refined #1 +67.03% suggests yes for the 8h-only path, but no infrastructure currently exposes single-horizon CDaR scoring without going through Mode S.

**Cleanup options**:
- Keep `crypto_trading_system_ed_cdar.py` + `_cdar.csv` + `_cdar.json` on disk for the single-horizon-only follow-up (recommended — the work is done)
- Or delete entirely: `rm crypto_trading_system_ed_cdar.py models/crypto_ed_production_cdar.csv config/regime_config_ed_cdar.json`

**Live trader impact**: ZERO. The CDaR engine wrote only to `_cdar`-tagged paths throughout. Production CSV + regime config untouched.

---

### 📋 PREVIOUS #1 PRIORITY — Stability Filter + CDaR (smoke tests done; CDaR moved to fork above)

#### 🟦 Idea #5 — Feature Stability Filter — SHELVED (see "2026-05-03 shelved" below)

#### 🟨 Idea #6 — CDaR — PROMOTED to engine fork (see #1 above)

(kept here only as audit trail; primary action is the validation HRST above)

**What**: Ideas #5 (Feature stability filter) + #6 (CDaR objective) from the 20-ideas roadmap. Both target a different layer than the 4 dead ideas (frac-diff features, calibration, VPIN filter, turbulence kill-switch):
- **#5 attacks Mode D's feature selection** — pre-filter time-unstable features before LGBM uses them
- **#6 attacks Mode V's scoring** — pick configs that better match the R4 rollback rule's max-DD criterion

#### 🟦 Idea #5 — Feature Stability Filter (Carhart-style robustness)

**Hypothesis**: HRST run-to-run nondeterminism (5-10pp variance) is partly driven by individual features whose LGBM importance ranking moves dramatically across sub-periods of training data. If a feature ranks #3 in days 1-30 but #45 in days 30-60, LGBM is making feature-selection decisions on noise.

**Mechanism**: monkey-patch `_test_lgbm_importance` to compute ranking on 3 overlapping sub-windows (oldest 50%, middle 50%, newest 50%) and DROP features whose `max_rank − min_rank > THRESHOLD` (default 30 positions). Then return the standard ranking computed on full data using ONLY the stable features.

**Run on Desktop**:
```
python tools/test_stability_filter_mode_d.py
```

Optional override: `set STABILITY_RANK_THRESHOLD=20 && python tools/test_stability_filter_mode_d.py` (lower = stricter)

**ETA**: ~10-20 min (4 horizons × ~5 min Mode D, 2 concurrent).

**Outputs**:
- `models/crypto_ed_grid_ETH_<5,6,7,8>h_STABILITY.csv`
- `logs/stability_smoke_<h>h_<ts>.log`
- `logs/stability_smoke_summary_<ts>.txt`

#### 🟨 Idea #6 — CDaR Optuna Scoring (Chekhlov, Uryasev, Zabarankin 2005, *Math Finance*)

**Hypothesis**: Mode V's current scoring `combined_score = return × WR` (Critical Rule 7) doesn't penalize deep drawdowns. CLAUDE.md R4 rollback rule explicitly judges live performance on max-DD (>10pp threshold). There's an alignment gap — Mode V picks for return×WR, R4 judges on DD. CDaR_5% = expected DD given DD is in worst 5%. Adding it as a penalty in scoring closes the gap.

**Mechanism**: run Mode D with a thin monkey-patch (eval-function wrapper for diagnostic), then refit the top-15 grid candidates per horizon and compute drawdown series + CDaR_5% per candidate. Rerank under multiple λ values {0.5, 1.0, 2.0, 5.0} where `cdar_score = return_pct − λ × max_dd_pct`. Compare top-3 under each scheme to the current top-3.

**Run on Desktop**:
```
python tools/test_cdar_rescore.py
```

**ETA**: ~30-40 min (Mode D ~10-15 min + refit-top-15-per-horizon ~10-15 min for the audit pass).

**Outputs**:
- `models/crypto_ed_grid_ETH_<5,6,7,8>h_CDAR.csv` (Mode D outputs, tagged)
- `output/cdar_audit_ETH_<h>h_<ts>.csv` (per-config rerank under each λ)
- `logs/cdar_smoke_<h>h_<ts>.log`
- `logs/cdar_smoke_summary_<ts>.txt`

#### Decision rules

| Test | Win → next step | Marginal | Lose |
|---|---|---|---|
| #5 Stability | ≥2 horizons get higher APF AND fewer features in winning config | Same winners as no-filter baseline | APF significantly lower → filter dropped legit features |
| #6 CDaR | At least one λ value picks a different top-3 with similar returns AND materially lower max-DD | Top-3 identical across all λ → return + max-DD highly correlated | All λ pick configs with much lower returns → λ too aggressive at all settings |

#### Invariants
- All writes are `_STABILITY` / `_CDAR`-tagged in `models/`, or written to `output/` / `logs/` — production untouched
- `--no-persist` + `--no-data-update` prevent any state leakage
- Patchers self-clean on exit (try/finally)

#### Run order recommendation
- **#5 first** (~10-20 min, faster, simpler audit). If it doesn't help, run #6 (~30-40 min, more involved audit but pure-rescore so the value is clearer to interpret).
- **OR run both in parallel on different machines** — no file collision since outputs are differently-tagged.

**Citations**: Chekhlov, A., Uryasev, S., Zabarankin, M. (2005) "Drawdown Measure in Portfolio Optimization", *Mathematical Finance* 15(3). Carhart, M. (1997) "On Persistence in Mutual Fund Performance", *J. Finance* — 4-factor stability principle applied here to feature ranking.

---

### 🟡 #2 PRIORITY — M-29 partial-fill fix — HOT-PATH-CLEAN, COLD-PATH UNTESTED IN PROD (audit 2026-05-06)

**Status as of 2026-05-06**: code shipped 2026-05-02 evening (commit `d568a30`, pushed to `origin/main`). Trader running with fix since restart. **6 post-fix BUYs audited, 0 partial-fills observed → cold path (the recalc/cap logic) hasn't been exercised against real exchange behavior yet.**

**Audited BUYs since fix shipped (2026-05-02 → 2026-05-06)**:

| # | Time (UTC) | Target | Recorded `usd_invested` | Maker attempts | Partial-fill? |
|---|---|---|---|---|---|
| 1 | 2026-05-03 20:02:11 | $12,000 | $12,000.00 | 4 single-leg | No |
| 2 | 2026-05-04 22:01:07 | $12,700 | $12,700.00 | 1 single-leg | No |
| 3 | 2026-05-05 15:01:12 | $14,000 | $13,999.99 | 3 single-leg | No |
| 4 | 2026-05-05 23:37:30 | $14,000 | $14,000.00 | 4 single-leg | No |
| 5 | 2026-05-06 03:01:07 | $14,000 | $14,000.00 | 3 single-leg | No |
| 6 | 2026-05-06 15:00:33 | $14,000 | $13,999.99 | 1 single-leg | No |

All 6 hot-path-clean (recorded basis = target ± $0.01 safety margin, never over-target). No `[M-29 recalc]` / `[M-29 cross-check]` / `Target reached` markers fired in any post-fix log → recalc path silent.

**Pre-fix base rate of bug** (for context — 3/3 BUY partial-fills exhibited the bug):

| Date | Recorded usd_invested | Bug evidence |
|---|---|---|
| 2026-04-25T12:09:16Z | $5,148.47 (synced) | Cascading 2-stage partial: wallet drained $12,423 → $0.01; sync-recovery |
| 2026-04-28T08:02:00Z | $12,521.74 | **Over-target by $521.74** |
| 2026-05-02T14:03:57Z | $12,609.53 | **Over-target by $609.53** (canonical bug event) |

**Verdict**: fix appears to be working but cold path is statistically unconfirmed. Per the original protocol "If it works correctly across 3-5 BUYs (at least 1 with a partial fill): this entry moves to Closed." — we have 6 BUYs but **zero partial-fills since fix**. Three options:

1. **Accept hot-path correctness as sufficient** (9 unit tests in [tools/test_m29_partial_fill_bug.py](tools/test_m29_partial_fill_bug.py) cover the cold path including the May-2 exact scenario) → close M-29 as resolved.
2. **Keep priority open + wait** — partial-fills happen every 5-7 days in volatile periods historically, so cold-path observation is likely within 1-2 weeks of normal trading.
3. **Force the cold path** with a deliberately oversized `/buy` to provoke a partial fill — risky, real money against real exchange, NOT recommended.

**Recommendation**: option 2 (passive wait). When the next partial-fill BUY happens, audit the log for the `[M-29 recalc]` markers + verify recorded basis ≤ target.

Original test protocol preserved below for re-application when a partial-fill BUY happens:

**Test protocol for tomorrow's first BUY (manual `/buy` OR auto regime BUY — both exercise the same code path)**:

1. **Verify trader is on the new code**: in the engine dir, `git log -1 --oneline` should show `d568a30 M-29 fix...`. If trader was running before the restart, confirm restart picked up the fix.
2. **Let a BUY happen during normal use** — don't manufacture a test trade. The fix only fires on partial-filled BUYs, so it might take a few entries before it's exercised.
3. **Watch the trader log for new diagnostic lines**:
   - `[M-29 recalc] target=$X filled=$Y remaining=$Z cash=$C next_size=$N` — fires after every partial-fill cancel cycle
   - `[M-29 cross-check] wallet says spent $A, orders say $B (delta $C). Using max → smaller remaining.` — fires only if the two sources disagree by >$1 (interesting telemetry, harmless)
   - `Target reached: spent $X of $T — remaining $Y below $300 min trade. Stopping.` — clean stop without market-fallback (replaces the case where old code would have over-spent the residual)
4. **After the trade completes, verify in [config/position_ed_v2_ETH.json](config/position_ed_v2_ETH.json)**:
   - `usd_invested` should be ≤ target (within ~$5 tolerance for rounding)
   - **Compare to today's bug**: today was $12,609.53 = $609.53 over. Tomorrow's value should be $11,995-$12,005 range for `/buy 12000`.
5. **If it overshoots again**: capture the log + position file, then revert immediately:
   ```bash
   git revert d568a30 && git push
   ```
   Restart trader. Back to old behavior (with the bug, but known-tolerable).
6. **If it works correctly across 3-5 BUYs (at least 1 with a partial fill)**: this entry moves to Closed and the bug is officially fixed.

**Expected first-BUY behavior** when wallet ≥ target, no partials encountered: trader places maker, fills fully, no `[M-29 recalc]` log line ever fires (recalc only fires on cancel-after-partial). Position file shows `usd_invested ≈ target`. This is the hot path — fix should be invisible.

**Expected partial-fill behavior** (the one we care about): trader places maker, partial fill, cancels, `[M-29 recalc]` fires, places SMALLER next leg (capped at `remaining_target`), eventually fills target. Position file shows `usd_invested ≤ target`. **NEVER over-target.**

---

### Background: the bug + the fix shipped

Found 2026-05-02 from today's `/buy` event ([logs/ed_v2_20260429_230924.log](logs/ed_v2_20260429_230924.log) lines 9012-9027). **Real overspend: $609.53** on a $12,000 target → recorded position `usd_invested = $12,609.53` ([config/position_ed_v2_ETH.json](config/position_ed_v2_ETH.json) latest BUY entry timestamped `2026-05-02T14:03:57Z`).

**What actually happened (per position file + log)**:
1. Wallet cash before /buy: ~$12,609.55 (MORE than the $12,000 target — this is the catastrophic-scenario condition)
2. User issued `/buy` → default size = `max_position_usd = $12,000.00` (logged as `$11,999.99` after the $0.01 safety-margin floor)
3. Phase 1 partial fill: $2,026.12 actually filled (log displayed "Partially filled: 11%" but that was rounding — true partial was 16.9%)
4. After partial: wallet cash dropped to $10,583.43 (= $12,609.55 − $2,026.12)
5. **Buggy recalc** at [crypto_revolut_ed_v2.py:826-828](crypto_revolut_ed_v2.py#L826): `if usd_avail < size: size = math.floor(usd_avail * 100) / 100 - 0.01` → `size = $10,583.41` (full remaining cash) instead of `target − already_filled = $12,000 − $2,026.12 = $9,973.88`
6. Phase 2 fill: $10,583.41 at $2,308.06 = 4.586 ETH
7. **Total spent: $2,026.12 + $10,583.41 = $12,609.53** ($609.53 over target)

**Bug location**: [crypto_revolut_ed_v2.py:818-828](crypto_revolut_ed_v2.py#L818) inside `_execute_maker_order()`:
```python
if usd_avail < size:
    print(f"    Balance updated after partial fill: ${size:,.2f} → ${usd_avail:,.2f}")
    size = math.floor(usd_avail * 100) / 100 - 0.01
```
The print statement writes `(target_size → wallet_avail)` — the log line `$11,999.99 → $10,583.42` is NOT before/after wallet, it's `target_var → avail_var` from the same moment. Subtle source of confusion when reading logs.

**Why it matters more broadly**: the same code path applies to every BUY (auto + manual). Today's overspend was bounded because wallet only had ~$12,610. With a larger wallet (e.g., $20k cash, /buy $12k target):
- Phase 1 partial fill ~17% = $2,026
- After: `usd_avail = $17,974`, `size = $12,000`. Condition `$17,974 < $12,000` is False → size stays $12k
- Phase 2 fills $12,000 (the original target — second time!) → **total $14,026, $2,026 over target**
- If THAT phase 2 also partial-fills, the bug compounds

For SELL side ([crypto_revolut_ed_v2.py:830-836](crypto_revolut_ed_v2.py#L830)): same logic, same bug, but less catastrophic because we're selling everything we own (the original `size = base_amount` already equals total holdings, and `crypto_avail < size` only fires after partial fill removes some — then we sell what's left, which IS what we want).

**Fix shipped (`d568a30`, 2026-05-02 evening)**:
1. Track `total_filled_usd` (BUY) / `total_filled_qty` (SELL) by reading `filled_quantity × average_fill_price` from each cancelled order's status (Source A — immune to other-asset USD activity contaminating wallet)
2. Cross-check against wallet-delta from `baseline_avail` captured at function entry (Source B — immune to order-status read failures)
3. Use `max(spent_by_wallet, spent_by_orders)` → smaller `remaining_target` → never overspend
4. `next_size = min(usd_avail - 0.01, original_size - total_filled_so_far)`
5. Two early-stop branches: cash-below-min (market-fallback for residual or stop) and target-met-within-tolerance (stop, don't place sub-min orders)
6. Symmetric SELL-side patch (currently benign because trader always sells 100% holdings, but partial-sell scenarios would have hit same bug)

**Verified by 9 unit tests** in [tools/test_m29_partial_fill_bug.py](tools/test_m29_partial_fill_bug.py): today's exact bug ($12,610 wallet, $12k target, $2,026 partial → next leg correctly capped at $9,973.88 instead of buggy $10,584.42), $50k wallet catastrophic scenario, wallet < target, cash near minimum, target essentially met, baseline read failed, cross-check disagreement, first-iter no fill.

**Caller-compat verified**: all 4 callers (auto BUY/SELL + manual `/buy`/`/sell`) use balance-delta as primary basis source via M-02/M-03 ledger-delta logic. New early-stop return dict `{'status': 'filled_target_reached', 'spent_usd': X}` gracefully falls back to balance-delta in callers — `filled_quantity`/`average_fill_price` missing fields don't cause issues.

**Audit completed**: 12 risk dimensions checked (return-dict compat, baseline race, double-counting, partial→full path, MIN_TRADE_USD edge cases, SELL-side asymmetry, M-02/M-03 interaction, retry paths, syntax/compile, etc.). No regressions identified. See conversation log 2026-05-02 evening for full audit.

**Priority severity**: HIGH — silent money-correctness bug, same class as M-02/M-03 ledger-delta bugs from 2026-04-25 bundle. Live test required because Revolut's actual API behavior on partial fills can't be mocked perfectly.

---

### Currently running
- **C04→C08 runner** ([tools/test_c04_to_c08_runner.py](tools/test_c04_to_c08_runner.py)) — launched on Desktop 2026-05-06 evening. ~5h ETA. See "🎯 MOST IMPORTANT" section at top of ACTIVE TODO for full method/decision rules per CID.

**Stale (was running 2026-05-02, since superseded — preserved for audit)**:
- HRST ETH 5,6,7,8h --replay 1440 --no-persist — started 2026-05-02 18:00 ([logs/ed_v1_20260502_180022.log](logs/ed_v1_20260502_180022.log)). Validated parallel-wrapper bug fixes A+B. Result fed into 2026-05-03 promotion (since superseded by today's 2026-05-06 15:48 promotion).
- HRST BNB 4,5,6,7,8h --replay 1440 --no-persist — started 2026-05-02 20:28. Done; result not promoted (BNB stays disabled).

### Next big work — IN ORDER

**1. Test BNB**

BNB code-wired today (commit `51c6f11`); derivatives data downloaded; status:
- ✅ Hourly OHLCV download — `data/bnb_hourly_data.csv` (73,672 candles, 2017-12-01 → 2026-05-02)
- ✅ Mode P PySR — DONE 2026-05-02 20:24, all 5 JSONs written, anti-leakage check passed, ~1.5h on 3-worker laptop (validated parallel-P path of merged engine)
- ⏳ **Full HRST 4,5,6,7,8h --replay 1440 --no-persist — RUNNING NOW** (started 20:28). See "Currently running" above.
- 🔜 Pipeline-health check (per SOL/LINK pattern): horizons positive, alpha vs ETH live, correlation with ETH (target ≤ 0.70 for diversification)
- 🔜 Decision: BNB clears the same bar as ETH (≥50% per-$ alpha + low correlation + plateau-stable) → flip `enabled: true` + `max_position_usd > 0`. Otherwise shelve.

**2. Build 15-minute candle system (Ed15)**

Old Ein/Eli scripts archived (`archive/crypto_trading_system_ein.py`, `archive/crypto_trading_system_ed15.py`) — both stale, missing parallel wrapper + post-2026-04-25 audit fixes. Right approach: clone current production engine.
- `cp crypto_trading_system_ed.py crypto_trading_system_ed15.py`
- Adapt constants — horizons in candles (4-10 candles = 1h-2h30), grid windows in candles (12-120), `MAX_DIAG_HOURS` interpretation
- Separate file paths — `models/crypto_ed15_production.csv`, `models/crypto_ed15_best_models.csv`, `config/regime_config_ed15.json`, `models/pysr_{ASSET}_{N}p.json` (p = periods/candles)
- Reuse macro pipeline — `download_macro_data.py` unchanged; 15-min OHLCV (`data/{asset}_15m_data.csv`) already exists for BTC/ETH; need full download for BNB
- Clone noprod wrapper — `crypto_trading_system_ed15_noprod.py` mirroring current pattern
- Optional separate trader — `crypto_revolut_ed15_v2.py` if running hourly + 15-min concurrently. Otherwise extend trader to dispatch by config block
- `config/disabled_features_ed15.json` already exists from old Ein attempts — re-check Grade-1 list (may differ at 15-min resolution)
- Validation order — run BNB on Ed15 first as new-asset case, then ETH/BTC if it works

### Week-of audit (2026-04-25 → 2026-05-02) — comparable cross-window summary

All ETH HRSTs from the past 7 days, normalized to "last 2 months" via Mode T's H1/H2 split (per Critical Rule 17). The 4mo run is included via its H2 (last 2mo) baseline + top STRICT gate winner H2.

| Date | Window | Detector | Bull | Bear | Last-2mo baseline | Last-2mo T total | Gate Δ on last 2mo | Conv |
|---|---|---|---|---|---|---|---|---|
| Apr 25 03:56 | 2mo | sma24>sma100 | 6h@75% | 5h@75% | +66.92% | +68.02% | +1.1pp | iter2 |
| Apr 25 07:57 | 2mo | sma168>sma480 | 7h@85% | 5h@75% | +88.55% | (no shield) | 0pp | iter3 |
| **Apr 25 23:12 (a)** | **2mo** | **sma24>sma100** | **6h@80%** | **5h@65%** | **+68.72%** | **+76.99%** | **+8.3pp** ← best | **iter2** |
| Apr 25 23:12 (b) | 2mo | sma24>sma100 | 6h@80% | 5h@65% | +58.57% | +63.42% | +4.9pp | iter3 |
| Apr 26 14:11 | 2mo | price>sma72 | 6h@80% | 7h@85% | +65.75% | +72.09% | +6.3pp | iter2 |
| Apr 28 08:08 | 2mo | tsmom_672h | 5h@65% | 6h@65% | +79.47% | +79.47% | 0pp | iter2 |
| **Apr 30 04:42 (H2)** | **last 2mo of 4mo** | **vol_calm** | **6h@70%** | **5h@85%** | **+65.37%** | **~+65.34%** | **~0pp** | **iter3** |
| May 01 09:21 | 2mo (8h HRST) | tsmom_672h | 7h@85% | 16h@75% | +68.81% | +68.81% | 0pp | iter2 |
| May 02 10:02 | 2mo (XRP) | tsmom_672h | 4h@80% | 7h@65% | +28.07% | +40.84% | +12.8pp | iter3 |
| May 02 13:53-15:48 | — | — (BTC, crashed) | — | — | — | — | — | parallel-engine bug |
| May 02 18:00 | 2mo | RUNNING | — | — | — | — | — | post-bug-fix |
| May 02 20:28 | 2mo | RUNNING (BNB) | — | — | — | — | — | — |

**Real biggest gate-gain of the week**: `Apr 25 23:12 sma24>sma100 6h@80%/5h@65% +8.3pp` (and Apr 26 `price>sma72 6h/7h +6.3pp` close behind). Apr 30 `vol_calm` looked huge at +20.5pp on the full 4mo REF, but its H2 (last 2mo) gate gain was ~0pp — the +20pp came entirely from months 1-2 of the 4mo window. Detector winners across the week: sma24>sma100 ×4, tsmom_672h ×3, sma168>sma480 ×1, price>sma72 ×1, vol_calm ×1.

**Current live config (`tsmom_672h bull=6h@85% / bear=5h@65%`) doesn't appear in any 7-day log.** Closest is Apr 28 (`tsmom_672h 5h/6h` — bull/bear horizons SWAPPED + different confs). Either set by an older HRST (logs deleted in today's cleanup) or manual edit/merge. Worth a `git log -p config/regime_config_ed.json` trace.

### User decisions pending
- **Live config bull horizon: 6h → 7h?** — 8h HRST May 01 plateau-unanimous on bull=7h. Compare its last-2mo gate gain (~0pp) against alternatives before flipping. Marginal evidence in absolute terms.
- **Live config bear horizon: 5h → 16h or 12h?** — Material change (8h HRST plateau picked 16h ×7 / 12h ×5 / 8h ×2 over 5h). 0pp shield/gate gain on last 2mo. Plateau-stable but doesn't dominate. Validation needed before flipping.
- **Investigate Apr 25 23:12 `sma24>sma100 6h@80%/5h@65% +8.3pp` finding** — actual best gate-gain of the week. Reproduce on current data window if not too stale; consider promoting if alpha holds. ~3-4h.

### Standing / monitoring (passive)
- **MIX gate live perf** — 5+ days in (since 2026-04-27 ~20:30 CEST). Rollback criteria: realized alpha drops >15pp vs sim baseline over first 10 trades; max DD exceeds −10%; signals consistently blocked when forward 24h is positive. One-line rollback: `copy config\regime_config_ed_pre_hrst2_20260427_evening.json config\regime_config_ed.json`
- **`output/ERRORS_INBOX.md`** — currently empty; check on every TODO review

### Scheduled (calendar-driven)
- **2026-05-22** — `deriv_oi_*` re-enable A/B test (3 features, ~63d Binance OI history by then). Procedure G1 in research-queue history below.
- **2026-06-18** — Orderbook + IV re-enable A/B test (`ob_imbalance, spread_bps, avg_iv, iv_skew`, ~60d data). Procedure G2 below.

### Research queue — actually untested, lower priority than BNB + Ed15
1. Vol-scaled horizon 4mo validation — 2-month tested 2026-04-19 (`vol_2band low→8h high→6h @90% = +33.82% / +5.02pp over tsmom`). Only 4mo missing. `tools/test_vol_scaled_horizon.py --replay 2880`. ~30 min.
2. Multi-horizon ensemble emergency-exit (4th angle from 2026-04-27 forensic) — force exit when 5h AND 8h both flip SELL within 1h. Distinct from T1b entry-side test which was already shelved. Per-horizon cache exists. ~1-2h.
3. Per-regime feature set — bull more technical / bear more macro. Untested deliberately. Low-medium.
4. Execution-gap TCA logging — biggest live-perf lever (~17pp unaccounted). Trader code change. 2-4h.
5. Trace current live config origin — `git log -p config/regime_config_ed.json` to find when `tsmom_672h 6h@85%/5h@65%` was actually set. ~10 min.
6. Verify BTC HRST works after parallel bug fix — `HRST BTC 4,5h --replay 1440 --no-persist` after laptop runs finish. Bug-fix verification only (BTC stays disabled). ~1h.

**Closed 2026-05-02:**
- ~~Re-run Mode D ETH 6h, 8h with derivative features (clean)~~ — derivatives are now standard in the clean pipeline (funding rate + perp-spot basis active and selected by LGBM; deriv_oi_* still quarantined per G1 schedule until 2026-05-22). Today's running ETH HRST 5,6,7,8h is the clean re-run this TODO was asking for; no separate work needed.
- ~~Triple barrier as TRAINING LABEL~~ — re-discovered tested + shelved 2026-03-14 in `archive/testing_literature.csv`. BTC same-week head-to-head: baseline 4h +57.22% / 8h +74.03% vs triple_barrier_label 4h +29.22% / 8h +22.53%. Lost on return (−28pp / −51pp), WR (−9pp / −10pp), accuracy (−16pp / −13pp), and combined score on every dimension. The "+29% standalone result" framing in earlier CLAUDE.md was misleading — same-day baseline beat it by ~30-50pp. SHELVED.
- ~~Pairwise long-horizon validation `R ETH 5,6,7,8,12,16`~~ — already done. The 8h HRST 2026-05-01 ran Mode R on `[4, 5, 6, 7, 8, 9, 12, 16]` (superset). Best regime: `tsmom_672h bull=7h bear=6h +43.49%`. Mode S joint sweep on the same 8 horizons picked WINNER `tsmom_672h bull=7h@85% / bear=16h@75% → Mode T REF +68.81%`. Re-running Mode R on 5,6,7,8,12,16 (subset) wouldn't add information. What's actually pending = USER DECISION whether to promote bear=16h to live based on existing evidence.
- ~~vol_calm Apr 30 promotion candidate~~ — initially looked like the biggest result of the week (+20.5pp gate gain on REF). Re-checked via Mode T's H1/H2 split: H2 (last 2mo of 4mo window) baseline +65.37% / T total ~+65.34% / **0pp gate gain on the comparable last-2mo slice**. The +20pp delta was driven entirely by months 1-2 of the 4mo window. NOT a promotion candidate. Methodology lesson now in Critical Rule 17.
- ~~Parallel-engine bugs A (`UnboundLocalError` at `_generate_signals_cached`) + B (`BrokenProcessPool` in hybrid refine)~~ — patched 2026-05-02 (per user). Validated by today's BNB Mode P (parallel-P all 5 horizons clean) + currently-running Mode V parallel cache + hybrid refine on laptop ETH/BNB HRSTs. Reverification of BTC HRST scheduled (#6 in research queue).
- ~~BNB Mode P 5,6,7,8h~~ — DONE 2026-05-02 20:24, all 5 PySR JSONs written cleanly, all `discovery_method = "historical"` (anti-leakage check passed). Parallel-P speedup observed (~1.5h for 5 horizons on 3-worker laptop, vs ~5h sequential reference).
- ~~Mode P auto-OHLCV-download patch~~ — landed 2026-05-02 (8 lines added at [crypto_trading_system_ed.py:3828](crypto_trading_system_ed.py#L3828)). Mode P now auto-fires `update_all_data([asset])` if hourly CSV is missing — prevents the "no data found" failure mode for fresh assets. Verified on BNB.

### Engineering / UX
- Telegram `/help` lists 8 of 13 commands (missing /buy, /sell, /hold, /cfg_*). [crypto_revolut_ed_v2.py:_handle_help_command](crypto_revolut_ed_v2.py) ~line 2392 + BotFather /setcommands. ~15 min.
- Document SOL/BNB on-chain CoinMetrics 403 as permanent free-tier limitation (visible in every macro download). Update `download_macro_data.py` to log "SKIPPED (free-tier 403)" instead of "ERROR" so it stops flagging as a failure. ~10 min.

**Closed (engineering):**
- ~~Engine Reference Card update for 4h / long horizons / 7h canonical bull~~ — done 2026-05-02 in Tier 1 edits. §"Horizon status" rewritten with 8h HRST evidence; §"What doesn't work" 4h entry struck-through with REVISED note; §"What's untested" purged of already-tested entries.

### Today's shelved (added 2026-05-02)
- **XRP enable** — user judged HRST results bad despite +25.69pp alpha. HRST winner `bull=4h@80% / bear=7h@65%` written to live config but `enabled: false` stays.
- **BTC re-enable** — results not good enough; no urgency. The BTC HRST work (after parallel bug fix verification) is purely engineering, not a promotion candidate.

### 2026-05-03 shelved
- **Idea #1 (Fractional differentiation features, López de Prado AFML Ch 5; Hosking 1981)** — SHELVED. Mode D smoke test on ETH 5,6,7,8h with `--replay 1440 --no-persist --no-data-update` ran via [tools/test_fracdiff_mode_d.py](tools/test_fracdiff_mode_d.py). Verdict: NOISE. Five `fd_*` features added (`fd_logclose_d03/d04/d05`, `fd_logvol_d04/d05`) at d ∈ {0.3, 0.4, 0.5}. Coverage 88-100% per d value, all features computed cleanly (smoke-test of patcher mechanism passed pre-run). LGBM importance ranking did NOT pick fracdiff features into the top tier; grid winners did NOT include them. Original hypothesis (top-3 selection-frequency features `price_to_sma100h`, `logret_120h`, `sma20_to_sma50h` would be the use case for memory-preserving frac-diff) DISPROVED on ETH at this window. Patcher + smoke-test scripts in [tools/test_fracdiff_mode_d.py](tools/test_fracdiff_mode_d.py) kept for future revival (different d values, different assets, different windows). Possible reasons it died on this data: (a) ETH log-prices may already have low long-memory in the current 60d window (Hurst < 0.5 = mean-reverting); (b) `price_to_sma100h` ratio already partially preserves memory; (c) different d values needed (try d ∈ {0.1, 0.2, 0.6, 0.7, 0.8} next time); (d) frac-diff payoff is documented for daily/weekly equity series — may not transfer to hourly crypto. **Do not retry on ETH hourly without changing one of those four levers.**

- **Idea #5 (Feature Stability Filter, Carhart-style)** — SHELVED. Mode D smoke test on ETH 5,6,7,8h with `--replay 1440 --no-persist --no-data-update` ran via [tools/test_stability_filter_mode_d.py](tools/test_stability_filter_mode_d.py). Patched `_test_lgbm_importance` to compute ranking on 3 overlapping sub-windows (oldest 50%, middle 50%, newest 50%) and drop features where `max_rank − min_rank > 30`. Results vs un-filtered baseline (this morning's calibration-run grid CSVs, which are valid since calibration patcher only patched `generate_signals` and Mode D's grid doesn't call it):
  - **5h**: APF 14.609 vs 14.885 (Δ −0.28), ret +17.21% vs +22.80% (Δ −5.6pp)
  - **6h**: APF 14.885 vs 15.146 (Δ −0.26), ret +21.67% vs +22.60% (Δ −0.9pp)
  - **7h**: APF 13.173 vs 14.895 (Δ −1.72), ret +18.25% vs +23.14% (Δ −4.9pp)
  - **8h**: APF 15.155 vs 14.186 (Δ +0.97), ret +25.22% vs +25.43% (Δ −0.2pp)
  - APF deltas mixed (3/4 horizons slightly worse, 1 better). Return deltas mostly negative or noise. Winning configs use SAME or MORE features than baseline (f=13-25 unchanged) — filter didn't simplify selection.
  - **Why it died**: (a) LGBM is already robust to noisy features via bagging + tree splits; the filter does work LGBM already does internally. (b) Sub-window ranking is itself noisy at 50% data — "rank movement >30" can come from sampling noise, not true instability. (c) Regime-dependent importance is REAL signal — a feature important in bull but not bear is regime-specific information, not noise; filtering drops legitimate information. **Conditions to revisit**: stricter threshold (rank movement >50), longer training window (180d not 60d) to give more stable sub-windows, or ensemble across multiple THRESHOLD values.

- **Idea #3 (VPIN entry filter, Easley/López de Prado/O'Hara 2012, RFS)** — SHELVED. Smoke test ran via [tools/test_vpin_filter.py](tools/test_vpin_filter.py). Best variant: `vpin_lb=30m_thr=0.5` → +83.22% vs baseline +79.39% = **+3.83pp** (15 BUYs skipped). Below +5pp ship threshold; within HRST run-to-run nondeterminism (5-10pp). Aggressive thresholds (0.3-0.4) skipped 130-160 BUYs and lost 20-70pp — discrimination is too weak. VPIN distribution at BUY signal times: p50=0.40-0.41, p90=0.50-0.55 — no fat right tail of toxic-flow days in 60d window. Either (a) ETH was unusually orderly during this window, (b) hourly aggregation washes out the 5-min toxicity spikes the literature describes (original Easley-LdP-O'Hara worked at 1-min cadence), or (c) crypto markets too efficient at hourly scale to leave informed-flow info unpriced. **Conditions to revisit**: (1) add 5-min cadence sub-loop in trader to check VPIN at native frequency; (2) test on 90d/6mo window with known stress events; (3) combine with realized-vol filter (VPIN may only inform during vol spikes). Tagged outputs in `output/vpin_filter_*.csv`.

- **Idea #4 (Turbulence Index kill-switch, Kritzman & Li 2010, FAJ)** — SHELVED. Smoke test ran via [tools/test_turbulence_killswitch.py](tools/test_turbulence_killswitch.py). All firing variants HURT returns: z≥1.0 → −9.96pp (skipped 17 BUYs), z≥1.5 → −5.83pp (skipped 9). Higher thresholds didn't fire (z≥2: skipped=0; only 1 BUY day in 60d had turb_z ≥ 2.0). **Killer finding**: in the 60d window, days when cross-asset turbulence was high were FOLLOWED by good ETH outcomes — opposite of Kritzman & Li's premise. Mechanism: in crypto bull regimes, spikes in cross-asset stress (VIX up, equities down) often coincide with crypto dip-buying setups; the model's BUYs during turbulence were RIGHT, not WRONG. Even if you wanted to retry on a longer/different window, the directional sign is wrong for ETH in bull regime. Risk vector tested: BTC + VIX + DXY + SP500 + GOLD + US10Y (6-dim), 252d rolling cov + 252d z-score. **Do not retry on ETH bull-regime data. May still be useful as a FEATURE (not kill-switch) in a different macro phase.** Tagged outputs in `output/turbulence_killswitch_*.csv`.

- **Idea #2 (Probability calibration of LGBM scores, Platt 1999; Niculescu-Mizil & Caruana 2005)** — SHELVED. Two-step test ran via [tools/test_calibration_audit.py](tools/test_calibration_audit.py) (Step A) and [tools/test_calibration_mode_d.py](tools/test_calibration_mode_d.py) (Step B).
  - **Step A audit (20 walk-forward cycles, ETH 6h prod model)**: Brier score improved 17.6% (raw 0.2774 → Platt 0.2285; isotonic 0.2666). Per-bucket gaps noisy at N=20 (max 80.9pp on a single-sample bucket; meaningful buckets with N≥4 showed model overconfident at high scores: predicted 70%, actual 30-50%). Verdict: directionally consistent with miscalibration literature; small-sample but informative.
  - **Step B Mode D smoke test (ETH 5,6,7,8h, --replay 1440, calibration injected via monkey-patch of generate_signals)**: CATASTROPHIC return loss vs prod baseline. Calibrated returns were +22.60% to +25.43% (4 horizons) vs prod baselines +46.30% to +67.14% — deltas of −23.16pp to −44.54pp. Trade counts collapsed to 3-6 per horizon vs prod's typical 30-70. Calibration HURT.
  - **Root cause** (deeper than test design): in a binary all-in/all-out engine where Mode S empirically tunes the confidence threshold, calibration is a **pure cosmetic relabeling**. The same trades fire whether threshold is 65% on raw scores or 50% on calibrated scores — Mode S would just rediscover the optimal threshold. AND Platt scaling on imbalanced data (30% positives) squeezes probabilities toward the base rate, so most predictions get pushed into 30-50% range and never cross the BUY threshold → trade frequency collapse → exposure collapse → return collapse. **Same architectural family as Kelly sizing (rejected): binary all-in/all-out doesn't benefit from accurate probability gradient.**
  - **Conditions to revisit**: (a) tiered position sizing (breaks binary), (b) EV-gated entries, (c) weighted ensembling across asset models. Any of those make calibration useful. Patcher kept in [tools/test_calibration_mode_d.py](tools/test_calibration_mode_d.py).

---

## 📜 HISTORICAL TODO (preserved as audit trail of tested/shelved decisions)

The entries below are kept verbatim so future-you can revive any shelved item with full context — what was tried, what the result was, why it was rejected. Do not delete; mark inline as RESOLVED if you re-evaluate.

---


## 📦 MERGED TOPIC: 8-horizon HRST (2026-04-30 launch → 2026-05-02 resolution)

*This topic groups several banners that were previously written separately as the work progressed. Content kept verbatim, just reordered chronologically and merged under one header.*

---

**🌙 OVERNIGHT 2026-05-01 → CHECK 2026-05-02: 8-HORIZON HRST RESULTS.**

Launched 2026-05-01 ~00:22 CEST on laptop:
```
python -u crypto_trading_system_ed.py HRST ETH 4,5,6,7,8,9,12,16h --replay 1440 --no-persist
```

ETA ~9-10h on laptop. Backups before launch:
- `config/regime_config_ed_noprod_pre_8h_20260501.json`
- `models/crypto_ed_production_noprod_pre_8h_20260501.csv`

**Context:** 2026-04-30 evening Mode D screen on horizons 9-18h (1440 replay) showed:
- **Killed:** 10h, 11h, 13h, 14h, 17h (return < +12%, APF weak, or accuracy < 60%)
- **Marginal/test:** 9h (+21.72%), 12h (+23.89% / 100% WR), 16h (+21.31% / 80% WR), 18h (LOW_TR)
- **Caveat:** trade counts collapse to 3-5 per config at horizons ≥ 14h on 1440h replay — Mode D for long horizons is borderline noise.

The 8-horizon HRST tests whether ANY long horizon (4h, 9h, 12h, 16h) survives full Mode H + V + S + T pipeline. Includes 4h to retest CLAUDE.md's "4h structurally broken" verdict — recent Mode D showed 4h XGB+LGBM apf=9.31 ret=+11.1%, suggesting the post-engine-fix data may have changed the picture.

**Decision matrix when results land:**
- Mode S TOP 15 stays in {5,6,7,8} → confirm canonical sweep, kill 4/9/12/16 from default permanently
- 4h appears in TOP 15 → CLAUDE.md "4h broken" verdict is data-snapshot-specific, revise Engine Reference Card
- 9h / 12h / 16h appears in TOP 15 → long horizons have signal in current regime, expand sweep
- Mode T REF > current LIVE +70.31% AND family-stable plateau → consider promoting
- Mode T REF < LIVE OR no plateau → keep LIVE 2-det config (`tsmom_672h 6h@85%/5h@65%`), document long horizons as dead

**Liveness signal (per CLAUDE.md rule 16):** check `logs/ed_v1_*.log` mtime; updates every few seconds during all phases. Mode S phase prints `2 detectors × 64 h-pairs × 7×7 conf = 6,272 combos` (vs today's 1,568 with 4 horizons) — confirms parser accepted all 8 horizons.

---


---

**✅ RESOLVED 2026-05-02:** 8-horizon HRST completed (~2026-05-01 09:15 CEST, total ~9h on laptop). Single-horizon Mode V winners ALL positive (4h +41.23%, 5h +34.88%, 6h +44.07%, 7h +64.90%, 8h +27.92%, 9h +31.36%, 12h +33.27%, 16h +30.79%). Mode S TOP 15 plateau: ALL bull=7h; bear migrated to 16h (×7), 12h (×5), 8h (×2). Mode S WINNER: `tsmom_672h bull=7h@85% / bear=16h@75% → +64.16%`. Mode T converged iter 2 → +68.81% / 24 trades / WR 92% (gates active, both shields OFF). Comparable to claimed live +70.31%. **Verdict**: long horizons (12h, 16h) are NOT dead in current regime — they dominate the bear plateau. Original CLAUDE.md "kill 4/9/12/16 from default permanently" decision rule was overcautious. New canonical sweep should consider extending bear range. Validation pending via current 5,6,7,8h HRST + pairwise `R ETH 5,6,7,8,12,16` follow-up.


## 📦 MERGED TOPIC: Parallel V/S/T + Mode P infrastructure (2026-04-29 launched on Desktop)

*This topic groups several banners that were previously written separately as the work progressed. Content kept verbatim, just reordered chronologically and merged under one header.*

---

**🧪 PARALLEL WRAPPER (in test) — 2026-04-29 launched on Desktop after laptop validation.**

`crypto_trading_system_ed_parallel.py` is an experimental wrapper that monkey-patches the engine in-process to parallelize:
- `eng.run_mode_v` ← Mode V Step 1 (6 D-candidate backtests) + Step 3 (3 refined backtests) via loky workers
- `eng.run_mode_s` ← per-horizon `generate_signals` cache build via loky, then delegate to engine sweep with cache-injected `generate_signals`
- `eng.run_mode_t` ← same per-horizon cache build, then delegate to engine T↔G iterative loop
- `eng._refine_top_configs` ← hybrid GPU+CPU refine (1 GPU process + 1 CPU process concurrent, 3rd config dynamic to first-freed device)

**Production engine `crypto_trading_system_ed.py` is NOT modified by the wrapper.** Patches fire only when the wrapper is invoked as `__main__`. Engine originals captured at wrapper-import time via `_ENG_RUN_MODE_S/T/V_ORIG` and `_ENG_REFINE_TOP_CONFIGS_ORIG` to prevent recursion through the monkey-patches.

**Validation status (as of 2026-04-29 17:30):**
- ✅ Smoke test passed on laptop (`HRST ETH 5,6h --replay 336 --no-persist --no-data-update`, 2h 06min): all 4 patches confirmed firing exactly once each, no recursion
- ✅ V/S/T parallel paths exercised cleanly (Step 1 ~3 min × 2 horizons, Step 3 ~2.5 min × 2 horizons, Mode S/T cache 2.4 min each)
- ✅ Hybrid refine focused test passed (3 synthetic configs, n_trials=5 override, ProcessPoolExecutor + GPU/CPU dispatch + dynamic 3rd config + APF-descending sort + result-list aggregation)
- 🔄 **Currently running**: full HRST 2880 ETH 5,6,7,8h on Desktop, started 2026-04-29 17:33. Projected ~9-9.5h vs sequential ~21h baseline.

**Test command** (for reproducing or relaunching):
```
python crypto_trading_system_ed_parallel.py HRST ETH 5,6,7,8h --replay 2880 --no-persist
```

Optional test-only flag for fast hybrid-refine validation: `--refine-trials 5` (default = `eng.REFINE_TRIALS = 50`).

**Per-machine policy (in wrapper, not in `hardware_config.py`):**
- All machines: `PARALLEL_BACKTESTS = 6` workers, `PARALLEL_LGBM_DEVICE = 'cpu'` inside parallel sections
- Sequential code paths still use the engine's auto-detected `LGBM_DEVICE` (gpu on Desktop/Laptop, cpu on Yoga)
- LGBM on CPU inside parallel section avoids GPU queue serialization (6 concurrent GPU LGBMs would queue and lose most of the speedup)

**What is NOT optimized (left sequential):**
- Mode H per-horizon outer loop — would oversubscribe CPU since Mode D internally uses `N_JOBS_PARALLEL=14` workers
- Optuna refine `n_jobs > 1` inside a single study — TPE sampler quality drops with parallel trials; the hybrid GPU+CPU split sidesteps this by running 3 separate studies in parallel processes instead

**Promotion criteria (test → production):**
1. Current 2880 finishes cleanly with `Done!` marker — proves all 4 patches survive a real 4mo workload at full trial count (50 trials × 3 configs × 4 horizons)
2. Wall-clock ≤ 10h (≥50% reduction vs sequential 21h baseline)
3. Mode S/T parallel cache fires exactly once per call (no recursion regression)
4. No `Traceback` / `OSError` / `TerminatedWorkerError` in the run log

If all 4 pass: promote the wrapper changes into production engine in a follow-up PR. Specifically:
- Move `PARALLEL_BACKTESTS` and `PARALLEL_LGBM_DEVICE` constants into `hardware_config.py`
- Inline `_run_parallel_backtests` + `_backtest_one_config_worker` into `crypto_trading_system_ed.py` Mode V Step 1 + Step 3
- Inline `_predict_signal_calls_for_horizons` + `_build_signals_cache_parallel` + `_generate_signals_cached` + replace Mode S/T per-horizon loops with parallel cache build
- Inline `_refine_top_configs_hybrid` into engine's `_refine_top_configs`
- Drop the wrapper file once production has the parallel paths

**Liveness monitoring** (per machine-setup rule above): track `logs/ed_v1_<latest>.log` mtime (writes every few seconds during all HRST phases). Mode V parallel sections show `[parallel] dispatching N backtests across N workers` and `[parallel] N backtests completed in N.N min` markers; Mode S/T cache shows `MODE S/T (PARALLEL signal cache)` then `[parallel] N signals generated in N.N min`; hybrid refine shows `[refine-hybrid] dispatching 3 configs across 2 workers` + `[refine-hybrid] {DEV} freed → starting config #3`.

---


---

**🧪 PARALLEL MODE P (queued for testing) — `crypto_trading_system_ed_parallel_p.py`.**

Separate experimental wrapper for Mode P (PySR feature discovery). Built 2026-04-29 17:55 alongside the V/S/T/refine parallel work. Mode P is the slowest research mode in the engine — each (asset, horizon) runs 4-7 sequential PySR studies, each with PySR's own internal parallelism explicitly disabled (`procs=0`, `parallelism="serial"`) because `deterministic=True` requires it.

**Approach (Option B — keeps determinism):**
- Each individual PySR run keeps `deterministic=True` + fixed per-run seed (42, 59, 76, 93, ...) — bit-for-bit reproducible across re-runs of the same settings on the same machine
- The N outer PySR runs (different feature subsets, different seeds) are dispatched across multiple processes via `ProcessPoolExecutor`
- **Per-machine policy:** Desktop=4 workers, Laptop=3 workers, Yoga=2 workers (each PySR Julia process ≈ 2-3 GB RAM)
- Expected speedup: **~2× per (asset, horizon)** for a 4-run setup (3 concurrent on laptop → 4 runs finish in ~ceil(4/3)=2× one-run time)

**Status:** built + wire-up checks PASSED, NOT yet run on real PySR.

**Test command — must run on a DIFFERENT asset than what desktop's ETH 2880 is using, since Mode P writes `models/pysr_<ASSET>_<H>h.json` files that the live HRST will read mid-run:**
```
python crypto_trading_system_ed_parallel_p.py P BTC 5,6,7,8h
```

Or smaller smoke test:
```
python crypto_trading_system_ed_parallel_p.py P BTC 5h
```

**Why BTC:** while ETH 2880 is running on desktop, ETH 8h Mode V (last horizon) will eventually read `models/pysr_ETH_8h.json`. If the laptop's Mode P writes a fresh ETH 8h PySR JSON between desktop's 7h and 8h Mode V starts, desktop will load the new file → inconsistent run (5h/6h/7h used old PySR features, 8h uses new ones). Running Mode P on BTC writes `pysr_BTC_*.json` — desktop never touches those, no contention.

**Validation criteria (test → would-promote):**
1. All N PySR studies dispatched complete cleanly (`[pysr-parallel] {LABEL} → N expressions` per dispatched run)
2. No `Traceback` / Julia worker death / serialization error
3. Wall-clock per (asset, horizon) ≤ 60% of sequential reference (i.e. ≥ 1.7× speedup on 3-worker laptop)
4. Output `pysr_BTC_*.json` files have `discovery_method = "historical"` (passes leakage check)
5. Same-seed re-run reproduces same expressions (determinism preserved)

If all pass: pattern is identical to the V/S/T/refine wrapper — eventually inline `discover_features_parallel` into `pysr_discover_features.py` directly.

**Liveness markers** to grep for during run: `[pysr-parallel] dispatching N PySR runs across M workers`, `[pysr-parallel] {LABEL} → N expressions`, `ALL RUNS COMPLETE: N candidate expressions in N.N min (parallel)`.

---


## 📦 MERGED TOPIC: 5m emergency-exit overlay (2026-04-27 morning 35-variant sweep → followup forensic of today's crash)

*This topic groups several banners that were previously written separately as the work progressed. Content kept verbatim, just reordered chronologically and merged under one header.*

---

**🚨 HIGH PRIORITY — 2026-04-27 morning EMERGENCY-EXIT 5m OVERLAY — TESTED + REJECTED.**

User asked for emergency exit triggered on 5-min price action to bypass shield during sharp crashes (motivated by 2026-04-27 03:45→05:45 UTC ETH crash $2400→$2319, -3.37% in 120min). Phase 1: identified 6 crashes ≥3%/120min in last 30d (incl. today's). Phase 2: built [tools/test_emergency_exit_5m.py](tools/test_emergency_exit_5m.py) overlay test — 35 variants of (threshold × lookback × cooldown × regime × bypass-mode × armed-at-profit).

**Verdict: NO variant beats baseline on any window.** The 60d baseline (+49.34% / 50tr / 70%WR) is already capturing crashes well via shield+max_hold+model SELL. Every emergency-exit variant either fires too often (false alarms eat alpha) or too rarely (no evidence of insurance value).

**Top results (sorted by 60d delta):**

| Rank | Config | d_30d | d_60d | d_90d | Fires 60d |
|---|---|---|---|---|---|
| 1 (cleanest) | **G. armed_at_pnl≥X% + thr=-2.0** (X∈{1..5}) | 0.00 | **0.00** | 0.00 | **0** |
| 2 | A. thr=-2.0% all bypass=always | -1.66 | -3.57 | -6.87 | 3 |
| 3 | G. thr=-1.5% armed_at_pnl≥X | 0.00 | -6.02 | -5.52 | 1 |
| **The one we expected to win:** A. thr=-1.0% | -4.71 | **-23.66** | -26.05 | 16 |
| Worst | A. thr=-0.7% | -8.51 | -38.14 | -41.05 | 43 |

**Why -1% loses so badly:** 16 fires in 60d, only 6 are real crashes; 10 false alarms at ordinary noise that mean-reverts within 30-60 min. Each false fire = ~0.5-1.5% lost alpha + double fees + 120min re-entry lockout that blocks the rebound BUY signal.

**What works as "free insurance":** `G.thr=-2.0% armed_at_pnl≥3%` — fired 0 times in 60d (no evidence either way). Theoretically correct design (only protect rally gains, not create losses) but unproven. Worth shipping with telemetry to collect real-world data over 1-3 months.

**Output CSV:** `output/emerg_exit_5m_20260427_091139.csv`. Crash list: `output/5m_crashes_20260427_090457.csv`. Indicator scores: `output/5m_indicators_20260427_090457.csv`.

**Crash list ≥3%/120min in last 30d (6 events incl. today):**
- 2026-03-29 21:35→22:45 (70min): $2007→$1939 (-3.41%)
- 2026-04-02 00:55→02:45 (110min): $2159→$2065 (-4.37%)
- 2026-04-08 13:05→14:55 (110min): $2271→$2187 (-3.70%)
- 2026-04-12 01:00→02:05 (65min): $2289→$2207 (-3.56%)
- 2026-04-14 14:30→15:05 (35min): $2416→$2333 (-3.43%)
- **2026-04-27 03:45→05:45 (120min): $2400→$2319 (-3.37%) ← TODAY**

**Confirms standing rule from CLAUDE.md:** *"All stop-loss / take-profit / profit-lock / trailing-stop variants — 8+ variants tested; baseline (no SL) won every dimension."* Same conclusion holds at 5-min granularity. Hourly shield + max_hold + model SELL already handles 90% of crashes; adding a faster trigger costs more in alpha than it saves in protection.

**Untested 4th angle:** emergency exit triggered by MULTI-HORIZON ENSEMBLE SELL agreement (using per-horizon signals from T1b) instead of raw price action. Would catch crashes when 2+ horizons unanimously flip SELL within 1 hour. Not yet tested.

---


---

**🚨 HIGH PRIORITY — 2026-04-27 forensic of today's crash + reverse-engineering attempt — CONFIRMS REJECT.**

After initial T5-T8 + emergency-exit sweep + H/I rally-conditioned variants, user asked to forensically reverse-engineer: given today's specific 03:45→05:45 UTC ETH crash ($2400→$2319), find ANY combination of derivatives (5/10/15min) + losses (5/10/15min) + prior rally that would have caught it cleanly. Built [tools/forensic_today_crash.py](tools/forensic_today_crash.py).

**Conclusion: today's crash was NOT detectable at 5-min granularity with acceptable FP rate.**

Bar-by-bar of today's crash:
- 03:45 (peak $2398) through 04:55: ALL indicators near zero. Slow drift mode. No signature.
- **05:00** ($2376, ret_5m=-0.29%, d2_5m=-0.26): FIRST momentum signature. Already -0.94% from peak.
- **05:15** ($2345, ret_15m=-1.29%, d2=-0.45): Clearest signal. Already -2.3% from peak.
- 05:45 (trough $2321): -3.4% from peak.

**Combo search results** (must fire by 05:30 UTC, scan 60d for false positives):
- 784 combinations fire today
- **Cleanest**: `r72h≥1.5 + ret_15m≤-0.8 + d2_5m≤-0.3` — fires today at 05:15 at $2345, but **42 false positives in 60d** (precision 12.5%)
- **Zero false-positive combos: NONE**
- **≤3 false positives: NONE**

**Why no clean combo exists**: ETH has been in slow uptrend for weeks. `rally_24h≥2%`, `rally_48h≥2%`, `rally_72h≥1.5%` are NOT rare events — they're the default state. Adding "preceded by rally" doesn't filter out false alarms because the rally context is permanent. The 5m drop signatures happen 70-100+ times in 60d; only 5-10 are real crashes. Math on best combo: save ~$120 once per 60d on real crash, lose ~$3000 to 42 false exits. Net catastrophic.

**Three converging lines of evidence say price-action emergency exit is a dead end for ETH in current regime:**
1. T5-T8 sweep (35 variants): best matches baseline at 0.00pp
2. H. rally-give-back winners (+1pp on 60d): caught blow-off tops, wouldn't have helped today
3. This forensic: no acceptable FP rate combo catches today

**Today's crash was news-driven** (Iran-related geopolitical context from yesterday/overnight) — by the time price cracked, the market had already digested the news. Price-derivative triggers can't anticipate this.

**Untested alternatives that might actually work:**
- **Multi-horizon SELL ensemble** as emergency trigger — fire if 3 of 4 model horizons (5h/6h/7h/8h, per T1b cache) all flip SELL within same hour. Catches model-recognized regime change, not lagging price action.
- **Manual override at high unrealized profit** — today user sold manually at +5.4%; the human supplied risk-aversion the algo can't infer from price.
- **Reduced position size in extended bull regimes** so a 3-5% give-back hurts less in absolute terms.

**Output CSVs**: `output/forensic_today_20260427_100214.csv`, `output/emerg_exit_5m_20260427_095120.csv`, `output/5m_crashes_20260427_090457.csv`.

---

**🌙 END-OF-DAY HANDOFF — 2026-04-27 ~23:00 CEST. User going to bed; resume tomorrow.**

### Currently active / running
- **MIX gate is LIVE in production** (promoted ~20:30 CEST tonight, trader restarted ~22:00 successfully). 4 layers of validation (Tier 1 OOS pass, cross-window 30d+60d, cumulative 30/60/90, disjoint 3×30d). Backups: `config/regime_config_ed_pre_hrst2_20260427_evening.json` + `models/crypto_ed_production_pre_hrst2_20260427_evening.csv`.
- **1440 HRST smoke test running on laptop** (~3-4h, started ~21:30ish CEST). Output goes to `_noprod` files. Validates today's engine bug fixes + grid trim end-to-end. Should finish overnight.
- **Live trader running on a different machine** (not this one — per CLAUDE.md rule 16). Status confirmed working with MIX gate after restart.

### Tomorrow's open decisions

**1. After 1440 HRST finishes** — decide:
- Review `_noprod` output → if winner looks sensible and infrastructure validated, decide whether to also launch 2880 HRST for true R5 validation
- 2880 with `--replay-v 1440` is now ~5-6h (not 12-15h, thanks to tonight's --replay-v fix) — much more palatable
- BUT: MIX is already live with 4 layers of validation, so R5 is now belt-and-braces, not strict requirement. Skipping is OK.

**2. Monitor MIX live performance** — passive, 1-2 weeks. Rollback criteria (per CLAUDE.md R4):
- Realized alpha drops >15pp vs sim baseline over first 10 trades
- Max DD exceeds -10% (worse than current PROD's historical -10.02% on 60d)
- Signals consistently blocked at moments where forward 24h is positive (gate too aggressive)

Rollback command (one-line, instant, no restart needed):
```powershell
copy config\regime_config_ed_pre_hrst2_20260427_evening.json config\regime_config_ed.json
```

**3. Expose available trader commands** — design decision pending:
- `/help` currently lists 8 commands but TRADER ACTUALLY HAS 13 — missing /buy, /sell, /hold, /cfg_*
- Options: (a) update /help text in engine code (~5 min), (b) BotFather /setcommands for native Telegram menu (~5 min user-side), (c) add /commands shortcut with buttons (~15 min), (d) document in CLAUDE.md
- Recommendation: A+B together (~15 min total)
- Existing /help defined in `crypto_revolut_ed_v2.py:_handle_help_command` (~line 2392)
- Inventory of all 13 commands captured in conversation; summary: /buy /sell /hold /cfg_ /chart /gate /help /pause /resume /setup /status /stop /sync

**4. T5b decision** — low priority:
- T5b winner (`bull_dd≥3% + bear_dd≥5% + bull_conf=90`) was +11.35pp on 60d standalone (against OLD PROD baseline)
- Likely also redundant with MIX (entry-side overlap, same pattern as T1b which dropped 87% of its alpha when measured against MIX-baseline)
- Re-test on MIX baseline before considering ship
- Test script template: copy structure from `tools/test_t1b_on_top_of_mix.py`

### Engine bug fixes shipped today (already persistent in CLAUDE.md history)
1. **Bug A (--replay propagation)**: HRST → Mode H → Mode D/V chain wasn't propagating CLI flag. Fixed at 5 sites.
2. **Bug B (MIN_GRID_TRADES dynamic)**: hardcoded `trades >= 8` killed all candidates on 5h horizon with small folds. Replaced with `max(4, n // 360)` dynamic threshold.
3. **Grid trim**: GRID_COMBOS 3→2 (dropped RF+XGB), GRID_WINDOWS 6→3 (dropped 200/250/300), GRID_FEATURES 6→4 (dropped 20/30) → 324→72 evals (-78% Mode D time). Backed by 20-winner empirical evidence — all kept-space configs match historical winners; all dropped configs had 0/20 wins.
4. **Optimizer bot aligned**: MODE_TIME_EST recalibrated (D=25→6, HRST=160→110, etc.), grid_total default updated.
5. **macro_daily SLA bumped 72h→96h**: Monday-morning yfinance lag was blocking hot-reload preflight. Fixed.
6. **/status enhanced** with shields + Bull G8 / Bear G8 lines for live config visibility.
7. **--replay-v flag**: decouples Mode V validation window from Mode D training window. 4mo HRST runtime drops from ~12-15h to ~5-6h with `--replay-v 1440`.

### Production state at end of day
- Live: MIX gate, bull_conf=80, bear_conf=85, both shields ON, bear gate unchanged (`rr30≥9.0% OR rr36≥9.0% cd=48h`), regime detector `price>sma72`, bull=6h bear=7h
- Backups in place for one-command rollback
- Engine + optimizer bot + trader all aligned

### What I'd start with tomorrow

1. Check the laptop's 1440 HRST log — did it finish? Did winners look reasonable?
2. Send `/status` in Telegram → confirm MIX gate still showing as live, no overnight regressions
3. If both clean → decide on 2880 R5 launch (now affordable at ~5-6h with --replay-v) OR move on to other work
4. If user wants to address Telegram commands discoverability → option A+B is the recommended path

---


## 📦 MERGED TOPIC: T1b ensemble vote (2026-04-27 morning discovery → evening shelved on MIX baseline)

*This topic groups several banners that were previously written separately as the work progressed. Content kept verbatim, just reordered chronologically and merged under one header.*

---

**🚨 HIGH PRIORITY — 2026-04-27 morning T1b TRUE multi-horizon ensemble vote — NEW TOP WINNER.**

Per-horizon signal cache `data/eth_per_horizon_signals_90d.pkl` was finally generated overnight (h=5,6,7,8 × 91d, ~2184 signals each). Built [tools/test_t1b_ensemble_vote.py](tools/test_t1b_ensemble_vote.py) — replaces the earlier T1b PROXY (conf-threshold sweep) with the real thing: at each bar, tally votes from horizons in `subset` requiring conf ≥ vote_thr; require k_buy votes for BUY, k_sell votes for SELL.

**Top winners — positive on all 3 windows (sorted by 60d delta vs same-base-conf baseline):**

| Rank | Config | d_30d | **d_60d** | d_90d | n_60d |
|---|---|---|---|---|---|
| 1 | **`58_only k_buy=1 k_sell=2 thr=85` base_conf=80** | +0.20 | **+19.84** | +16.38 | 20 |
| 2 | `567_only k_buy=1 k_sell=3 thr=90` base_conf=80 | +1.12 | +18.09 | +22.09 | 14 |
| 3 | `58_only k_buy=1 k_sell=2 thr=90` base_conf=80 | +5.58 | +6.60 | +18.39 | 18 |

**Top winner substantively (#1):** Use ONLY horizons 5h + 8h (drop the 6h/7h middle). Enter on ANY one BUY at conf ≥ 85%. EXIT requires BOTH 5h AND 8h to say SELL at conf ≥ 85% (2-of-2 confirmation). Asymmetric: easy entry, hard exit. **+19.84pp on 60d (66.45% strategy vs 46.61% baseline) — beats T5b winner's +11.35pp by +8.49pp.** Only 20 trades on 60d (vs 46 baseline = much more selective).

**Strong-but-thin winner (#2):** Use h=5,6,7. Enter on ANY one BUY at conf ≥ 90%. EXIT requires ALL 3 (5h+6h+7h) to say SELL at conf ≥ 90%. Most selective rule (n_60d=14 — borderline statistical thinness) but strongest 90d alpha (+22.09pp).

**Output CSV:** `output/t1b_ensemble_vote_20260427_085549.csv` (full sweep: 7 subsets × k_buy 1..N × k_sell 1..N × thr {70,80,85,90} × base_conf {80,90} × 3 windows).

**Implementation cost:** Higher than T5b — requires running ALL 4 horizon models (5h/6h/7h/8h) per cycle in live trader, not just the regime-anchor horizon. Currently the trader loads only `bull_h` (6h) and `bear_h` (7h) per `regime_config_ed.json`. Adding horizons 5h+8h: ~2× compute per cycle (2 extra model trainings per hour). Need new schema for "vote subset" + k_buy/k_sell + vote_thr.

**Caution:** 30d delta is +0.20 (barely positive). 60d/90d strong but recent 30d shows the regime-tilt risk — over-confirmation can miss recent shorter-cycle moves. Recommend 4mo HRST validation (`--replay 2880`) before promotion.

**Comparison to tonight's earlier T5b winner:**
- T1b ensemble (NEW): 60d +19.84pp / 20 trades / requires multi-model live infra
- T5b entry filter: 60d +11.35pp / 33 trades / config-only + ~30 lines code
- Trade-off: T1b is +75% better alpha but harder to ship and needs thicker validation.

---


---

**✅ T1b ENSEMBLE VOTE — SHELVED 2026-04-27 evening (after MIX promotion).**

T1b winner from 2026-04-26 sweep showed +19.84pp on 60d standalone. Re-tested on top of MIX-active baseline ([tools/test_t1b_on_top_of_mix.py](tools/test_t1b_on_top_of_mix.py)):

| Metric | Original (vs old PROD) | On top of MIX |
|---|---|---|
| 60d delta | +19.84pp | **+2.42pp** (87% gain evaporated) |
| 30d delta | (positive) | **-15.24pp** (HURTS recent month) |
| OOS held-out 30d | not tested | **+0.81pp** (essentially noise) |

**Mechanism redundancy**: T1b ensemble (k_sell=2 multi-horizon SELL agreement) was supposed to be exit-side, MIX is entry-side — should be orthogonal. But in practice, the trades T1b "saves" by holding through noise = the same trades MIX previously prevented from being entered. Once MIX cleans entries, exit-side noise drops and T1b has less to filter.

**Implementation cost** = 2× compute per cycle (4 horizon models vs 2), new schema, code changes in trader + live_trader. **Cost > marginal gain.**

**Methodology lesson**: most "winners" measured against weak baselines lose value when measured against a stronger baseline. T5b (also entry-side anti-overheat) likely shows the same collapse — should be retested on MIX baseline before considering. Don't promote anything from yesterday's sweep without re-measuring against current PROD (which is now MIX, not old gate).

**Output CSV**: `output/t1b_on_mix_20260427_224212.csv` (full sweep, all configs × 4 windows).

---


## 📦 MERGED TOPIC: MIX gate (2026-04-27 candidate → promoted same evening)

*This topic groups several banners that were previously written separately as the work progressed. Content kept verbatim, just reordered chronologically and merged under one header.*

---

**🚨 HIGH PRIORITY — 2026-04-27 afternoon: BULL RALLY-COOLDOWN GATE UPGRADE CANDIDATE — Tier 1 OOS PASSED, decision pending.**

After exhausting emergency-exit and short-term price-action options (see other 2026-04-27 entries below), pivoted to ENTRY-SIDE optimization. Built and ran a multi-stage cross-window-robust analysis on Mode G's rally-cooldown sweep, with extended HORIZONS list. Tier 1 OOS validation passed cleanly. **Awaiting promotion decision.**

### CANDIDATE FOR PROMOTION: bull rally-cooldown gate parameter change

**Current production:** `bull.rally_cooldown` = `rr20h>=4.0% OR rr24h>=4.5%, cd=12h`

**Proposed (MIX winner):** `bull.rally_cooldown` = **`rr12h>=2.5% OR rr20h>=4.0%, cd=24h`**

Diff: h_short 20→12, h_long 24→20, t_short 4.0→2.5, t_long 4.5→4.0, **cd_hours 12→24 (doubled)**.

Bear gate, shields, conf thresholds, max_hold, min_sell_pnl: **all unchanged from current PROD** (no other knob touched).

### Methodology used (be precise about what was/wasn't done)

| Step | What | Status today |
|---|---|---|
| HRS / HRST (regime detector + bull/bear horizons + confs + shields) | Full retune | **NOT redone** — used existing AB matrix Variant A (2-month optimum from 2026-04-26 promotion) |
| T (shield + min_sell_pnl + max_hold) | Sweep | **NOT redone** — kept current PROD values (bull/bear shield ON, 0.5%, 10h) |
| G (rally cooldown gate) | Sweep with **HORIZONS extended to include 48h** | **REDONE on 1m AND 2m**, then aggregated by cross-window STRICT intersection |
| Cross-window aggregation | 1m∩2m STRICT-passing intersection (9,113 configs out of 62,388 each) | New methodology added today |
| Tier 1 OOS validation | 3rd month (oldest 30d of 90d cache, not used in G optimization) | **DONE — PASSED** |

### Tier 1 held-out OOS validation result (this is the key promotion-justifier)

Tested on 2026-01-18 → 2026-02-17 (the FIRST 30 days of the 90d cache, not used in either 1m or 2m optimization):

| Setup | Held-out return | Δ vs no-gate | Δ vs current PROD | Verdict |
|---|---|---|---|---|
| NO GATE (baseline) | -5.84% | — | -0.89pp | reference |
| CURRENT PROD (`rr20≥4 OR rr24≥4.5 cd=12h`) | -4.95% | +0.89pp | reference | reference |
| **MIX (`rr12≥2.5 OR rr20≥4 cd=24h`)** | **+0.90%** | **+6.74pp** | **+5.85pp** | **PASS ✅** |
| 60d-opt with 48h (`rr24≥5 OR rr48≥6.5 cd=24h`) | -6.80% | -0.96pp | -1.85pp | FAIL ❌ |

**Critical observation:** B&H on this period was -40.82% (bear/correction phase). MIX gate not only didn't lose money during a bear regime — it ended slightly positive. This is a stress test the gate passed. Setup 60d-opt FAILED the held-out test, which validated the original concern that the 48h extension was overfit to recent regime.

### Performance summary across 3 windows (in-sample 1m & 2m + OOS held-out 3rd month)

| Window | Current PROD | MIX gate | Δ vs PROD | Status |
|---|---|---|---|---|
| Held-out 30d (2026-01-18→02-17) — OOS | -4.95% | **+0.90%** | **+5.85pp** | **OOS PASS** ✅ |
| 30d (used in opt) | +22.06% | +31.79% | +9.73pp | in-sample |
| 60d (used in opt) | +47.76% | +64.51% | +16.75pp | in-sample |
| **Average across 3 windows** | **+21.62%** | **+32.40%** | **+10.78pp** | — |

Drawdown also improves: 30d from -4.75% to -1.99% (-58%); 60d from -10.02% to -5.16% (-49%).

Win rate jumps from 73%/71% to 83%/79% across 30d/60d. Trade count drops 30-40% (more selective entries).

### Overfitting analysis — what the methodology DID and DID NOT eliminate

✅ **Filtered out:**
- Single-period flukes (each window's STRICT requires H1+H2+REF check)
- Pure-luck winners (cross-window must align 6 sub-period checks across 30d and 60d)
- 48h-extension overfit (60d-opt winner with 48h FAILED OOS, MIX winner without 48h PASSED)
- The cross-window MIX winner happens to also be the 30d-only winner — strong robustness coincidence

❌ **Not filtered out:**
- HRS not refreshed (gate optimized on stale model signals from 2026-04-26 cache)
- T not jointly re-optimized with new G
- No 4-month replay (CLAUDE.md rule R5 — gold standard for promotion)
- 1m, 2m, 3rd-month all from same continuous 90d cache (not truly independent regimes)

### Implementation plan if promoted (config-only change)

```bash
# Backup
cp config/regime_config_ed.json config/regime_config_ed_pre_mix_20260427.json

# Edit config: regime_config_ed.json -> ETH.bull.rally_cooldown:
#   h_short:      20 -> 12
#   h_long:       24 -> 20
#   t_short_pct: 4.0 -> 2.5
#   t_long_pct:  4.5 -> 4.0
#   cd_hours:     12 -> 24

# Hot-reloads in 5 min, no restart needed.
```

**Rollback:** `copy config\regime_config_ed_pre_mix_20260427.json config\regime_config_ed.json`. One-command, instant.

### Recommended path before flipping live

Two options the user is deciding between:

**(a) Ship now + queue HRST validation in parallel.** Config-only change, instant rollback. If 1-2 weeks of live perf is bad, rollback. Meanwhile run 4mo HRST to confirm.

**(b) Hold and run 4mo HRST first.** `python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 2880 --no-persist`. ~6-8h desktop. Decide based on whether 4mo winner matches MIX. Adds ~1 day before promotion but completes R5.

### Test artifacts created today (full audit trail)

All read-only, none touched production:

- [tools/sim_mode_g_with_48h.py](tools/sim_mode_g_with_48h.py) — extended Mode G with 48h, PROD-only vs PROD+48h comparison on 60d
- [tools/sim_mode_g_30_60_robust.py](tools/sim_mode_g_30_60_robust.py) — extended Mode G run on BOTH 30d and 60d, intersection of STRICT winners
- [tools/sweep_rally_48h_thresholds.py](tools/sweep_rally_48h_thresholds.py) — finer sweep of rally_48h thresholds 3.0-5.0% (manual approach)
- [tools/compare_shield_vs_rally_block.py](tools/compare_shield_vs_rally_block.py) — apples-to-apples shield-on/off vs rally-block matrix
- [tools/summary_table_30_60.py](tools/summary_table_30_60.py) — first clean comparison table (later superseded)
- [tools/final_comparison_table.py](tools/final_comparison_table.py) — full comparison incl shield states + gate strings
- [tools/tier1_held_out_test.py](tools/tier1_held_out_test.py) — OOS validation on first 30d of cache

Output CSVs:
- `output/sim_mode_g_robust_<ts>.csv` — 9,113 cross-window-robust winners
- `output/sweep_rally_48h_<ts>.csv` — fine 48h threshold grid
- `output/cmp_shield_vs_block_<ts>.csv` — shield/block matrix
- `output/summary_30_60_<ts>.csv` and `output/final_comparison_<ts>.csv` — clean comparison tables
- (Tier 1 prints to console; results captured above.)

### CRITICAL DECISION POINT

User has not yet flipped the switch. To promote, follow the implementation plan above. Pre-flight backup tag has not been created yet.

---


---

**✅ MIX GATE PROMOTED TO LIVE — 2026-04-27 ~20:30 CEST.**

User shipped MIX rally-cooldown gate to live production via config-only change to `config/regime_config_ed.json` ETH.bull.rally_cooldown:

```diff
- h_short: 20  → h_short: 12
- h_long:  24  → h_long:  20
- t_short_pct: 4.0  → t_short_pct: 2.5
- t_long_pct:  4.5  → t_long_pct:  4.0
- cd_hours: 12  → cd_hours: 24
```

**No other production knob changed.** Bull conf 80% (unchanged), bear conf 85% (unchanged), shields ON for both (unchanged), bear rally_cooldown unchanged (`rr30≥9.0% OR rr36≥9.0% cd=48h`), regime detector `price>sma72` (unchanged), min_sell_pnl=0.5%, max_hold=10h (unchanged).

**Position state at promotion**: `state=cash`, `auto_trade=true`, `rally_cooldown_until=""` (no active cooldown — clean slate for new gate).

**Live trader hot-reloads `regime_config_ed.json` every 5 min** — no restart required. Within next cycle the new gate is active.

**Rollback (one command, instant)**:
```powershell
copy config\regime_config_ed_pre_hrst2_20260427_evening.json config\regime_config_ed.json
```

Backup also kept for production CSV (`models/crypto_ed_production_pre_hrst2_20260427_evening.csv`) though MIX promotion did NOT touch the CSV (model selection unchanged).

### Evidence base for promotion (4 independent layers)

1. **Tier 1 OOS held-out 30d** (FIRST 30d of 90d cache, never used in optimization): MIX +0.90% vs PROD -4.95% during a -40.82% B&H period → **+5.85pp** ← only methodologically clean comparison
2. **Cross-window 30d+60d STRICT intersection** (MIX discovery methodology): same gate emerged
3. **Cumulative 90d+60d+30d STRICT intersection** (tonight's analysis via `tools/tg_window_decomposition.py`): same MIX gate emerged
4. **Disjoint 3×30d STRICT intersection**: close cousin emerged (rr12+rr14+cd=24, structurally identical family)

In-sample numbers (less rigorous but consistent):
- 30d in-sample: +9.73pp over PROD
- 60d in-sample: +15.17pp over PROD
- 90d in-sample: +25.56pp over PROD
- Drawdown reduced ~50% on both 30d and 60d

### Monitor + rollback criteria

**Monitor for 1-2 weeks** of live performance. Rollback if:
- Cumulative realized alpha drops >15pp vs sim baseline over first 10 trades (matches CLAUDE.md rule R4 standing policy)
- Max drawdown exceeds historical -10.02% (current PROD's 60d max)
- Live signals consistently blocked at moments where forward 24h is positive (gate too aggressive)

If rollback triggers: one `copy` command, hot-reloads in 5 min.

### Standing R5 caveat

Per CLAUDE.md rule R5: HRST validation gate normally requires 4-month replay confirmation before live promotion. **R5 NOT formally completed for this promotion.** User decided Tier 1 OOS + 3 cross-window confirmations were sufficient evidence given:
- Config-only change (no model/code modification)
- Instant rollback available
- All 4 evidence layers point to same gate family
- Full HRST 4mo would have taken 12-15h on laptop (declared not worth wait)

The 1440 HRST currently running is engineering smoke-test (validates the bug fixes + grid trim end-to-end), NOT R5 validation. The 2880 HRST queued for later WOULD be R5; if it picks something materially different from MIX, revisit.

---


## 📦 MERGED TOPIC: 4mo HRST validation arc (2026-04-27 launched 16:45 → evening realisation that --replay scales Mode V too)

*This topic groups several banners that were previously written separately as the work progressed. Content kept verbatim, just reordered chronologically and merged under one header.*

---

**🟡 RUNNING NOW — 2026-04-27 ~16:45 CEST — 4MO HRST VALIDATION (Tier 2 R5) on LAPTOP.**

Job launched on laptop ~16:45 CEST 2026-04-27.

**Command:** `& "C:\Users\Alex\algo_trading\venv\Scripts\python.exe" .\crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 2880 --no-persist`

**Working dir:** `C:\Users\Alex\algo_trading\engine` (laptop local copy)

**ETA:** ~6-8h → finishes between **22:45 → 00:45** tonight.

**Purpose:** Tier 2 R5 validation of the MIX rally-cooldown gate candidate (`rr12≥2.5 OR rr20≥4.0 cd=24h`). Tier 1 OOS already PASSED earlier today (held-out 30d test, +6.74pp vs no-gate, +5.85pp vs current PROD). This is the gold-standard 4-month HRST check before promotion.

**Output files when done:**
- `C:\Users\Alex\algo_trading\engine\config\regime_config_ed_noprod.json`
- `C:\Users\Alex\algo_trading\engine\models\crypto_ed_production_noprod.csv`
- `C:\Users\Alex\algo_trading\engine\logs\ed_v1_<timestamp>.log`

**When the user checks back, run this to compare 4mo winner to MIX:**
```powershell
Get-Content .\config\regime_config_ed_noprod.json | ConvertFrom-Json | Select-Object -ExpandProperty ETH | Select-Object -ExpandProperty bull | Select-Object -ExpandProperty rally_cooldown
```

**Compare to MIX winner:**
| Field | MIX (today's pick) | 4mo HRST result | Match? |
|---|---|---|---|
| `h_short` | **12** | ? | |
| `h_long` | **20** | ? | |
| `t_short_pct` | **2.5** | ? | |
| `t_long_pct` | **4.0** | ? | |
| `cd_hours` | **24** | ? | |

**Decision tree on completion:**
- 4mo bull gate matches MIX (within ±1 step on each param) → strong R5 confirmation, ship MIX with HIGH confidence
- 4mo bull gate is significantly different (e.g., different h_short/h_long pair, threshold deltas >2 steps, cd doubled or halved) → window-sensitivity; either ship MIX with MEDIUM confidence (config-revertible) OR shelve and wait for live data
- 4mo HRS picks a different regime detector or bull/bear horizons → bigger structural change; consult before any promotion
- Mode T converged + writes successfully → check it didn't hit `max_iter` (oscillation flag)

**Note:** 4mo HRST won't include 48h in Mode G's search by default (production HORIZONS still maxes at 36h). MIX winner is `rr12≥2.5 OR rr20≥4.0 cd=24h` — uses no 48h, so 4mo can find this same config natively if signal is real.

**Aliveness monitoring (if concern arises):**
```powershell
Get-ChildItem .\logs\ed_v1_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Select-Object Name, LastWriteTime
```

If log mtime < 5 min → alive. >30 min stale + no `python.exe` in tasklist → likely dead. Mode R silence up to 2h is normal (per CLAUDE.md rule 16) — don't declare dead during Mode R phase.

**Power management:** Laptop set with `powercfg /requestsoverride PROCESS python.exe SYSTEM AWAYMODE EXECUTION DISPLAY` to prevent Modern Standby from killing the job overnight.

---


---

**🚨 OPEN DECISION — 2026-04-27 evening: 4MO HRST RUNNING SLOW because `--replay` also scales Mode V (not just Mode D).**

**Diagnosis**: today's bug-fix cascade revealed an unintended consequence:
- Bug A fix (--replay propagation through HRST chain) → now correctly uses 2880 rows everywhere
- BUT: Mode V's STEP 1 ("backtest top 6 D candidates") runs the FULL 2880-hour replay through live signal generator (one model retrain per hour). Cost = ~20-30 min per candidate × 6 candidates × 4 horizons = **~10-12h just for Mode V**. Plus Mode S (3,920 combos × 2880 signals) and Mode T+G iteration also doubled.
- Realistic 4mo HRST runtime: **~12-15 hours** (not the 4-6h I originally estimated; Mode D trim only helped that one phase).

**Options to choose from when you read this:**

(a) **Kill + restart at `--replay 1440`** (~3-4h total). Same data scope as AB matrix; refreshes models + gate sweep without 4mo R5 grade.
```powershell
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist
```

(b) **Let it finish overnight** (~10-13h remaining at 19:00 CEST = finishes ~05:00-08:00 next morning). Get true 4mo R5 validation. Battery + cooling + Modern Standby risk.

(c) **Ship MIX based on Tier 1 OOS alone** (zero hours, instant rollback). Tier 1 already passed cleanly (+5.85pp vs PROD on truly held-out 30d). 4mo HRST was a "bonus" R5 confirmation, not strictly required for promotion.

**Recommended: (a) kill + restart at `--replay 1440`.** The marginal value of 4mo over 2mo is small (~+10-15h cost for slightly better OOS evidence on a window we'd already understand). Faster turnaround, lower risk.

**Engine bug-fixes from today (already shipped, persistent):**
- Bug A: `--replay` propagation through HRST → Mode H → D/V (4 sites added `replay_hours=` plumbing)
- Bug B: `MIN_GRID_TRADES = max(4, n // 360)` dynamic threshold (was hardcoded 8, killed all Mode D candidates on 5h-7h)
- Trim: GRID_COMBOS=2 (dropped RF+XGB), GRID_WINDOWS=3 (dropped 200/250/300), GRID_FEATURES=4 (dropped 20/30) → 324→72 evals (-78%)
- Optimizer bot aligned: MODE_TIME_EST recalibrated, grid_total default updated

**Engine bug B follow-up (KNOWN, NOT FIXED):** `--replay` cost scaling in Mode V/S/T+G is a feature, not a bug — these phases legitimately need the full replay window for accurate live backtest. If you want a permanent fix, would need separate `--replay-d` and `--replay-v` flags or hardcode a Mode V cap. Defer to user.

---

**🚨 HIGH PRIORITY — 2026-04-26 evening test sweep results (T5-T8). Persist across logoff.**

Tonight ran 4 standalone batch harnesses against 90d cached signal stream
(`data/eth_sl_signals_90d.pkl`). All read-only. None touched production.
Canonical evaluation window: **60d**. Results also reported on 30d / 90d.

**Baselines (ETH, prod config):**
- 30d: +22.03% / 26 trades / 58% WR
- 60d: +55.68% / 46 trades / 59% WR
- 90d: +47.17% / 68 trades / 56% WR

**Test files written:**
- [tools/test_t5_batch.py](tools/test_t5_batch.py) — 10 ideas: T5a asym sell-conf uplift, T5b per-regime dd, T5c trailing peak/retain, T5d per-regime min_sell_pnl, T5e rally-momentum exit, T5f days-down entry, T5g conf-weighted max_hold, T5h shield auto-off at profit, T5i vol-pctile entry gate, T5j sell-conf decay
- [tools/test_t6_triple_barrier.py](tools/test_t6_triple_barrier.py) — vol-adaptive triple barrier (upper σ × lower σ × vertical h) replacing model SELL+shield+max_hold
- [tools/test_t7_meta_proxy.py](tools/test_t7_meta_proxy.py) — cheap meta-labeling proxy (logistic regression on 7 meta features, walk-forward train_n × threshold sweep)
- [tools/test_t8_gdelt_overlay.py](tools/test_t8_gdelt_overlay.py) — GDELT geopolitical entry-overlay filters (geo_vol pctile, iran_tone, geo_tone_chg24h, only-improving-iran-24h)

**Output CSVs (timestamped):**
- `output/t5_batch_20260426_223330.csv`
- `output/t6_triple_barrier_20260426_223805.csv`
- `output/t7_meta_proxy_20260426_224605.csv`
- `output/t8_gdelt_overlay_20260426_225122.csv`

**Winners — positive on all 3 windows (30d/60d/90d):**

| Rank | Config | d_30d | **d_60d** | d_90d | n_60d |
|---|---|---|---|---|---|
| 1 | **T5b: bull_conf=90 + bull_dd≥3% + bear_dd≥5%** | +13.07 | **+11.35** | +26.12 | 33 |
| 2 | T5b: bull_conf=90 + bull_dd≥2% + bear_dd≥5% | +12.01 | +10.04 | +24.15 | 34 |
| 3 | T6: triple barrier up=6σ lo=2σ vert=24h conf=90 | +6.77 | +10.48 | +1.24 | 47 |
| 4 | T5b: bull_conf=90 + bull_dd≥3% + bear_dd≥3% | +12.67 | +6.49 | +18.04 | 34 |
| 5 | T5b: bull_conf=90 + bull_dd≥2% + bear_dd≥3% | +11.61 | +5.22 | +16.16 | 35 |
| 6 | T5c: trailing peak=3% retain=70% | +3.25 | +4.99 | +5.27 | 48 |
| 7 | T5j: bull_conf=90 + sell_conf_decay@h≥4 = -15pp | +5.72 | +3.92 | +16.64 | 45 |
| 8 | T5b: bull_conf=90 + bull_dd≥2% + bear_dd≥2% | +10.49 | +3.87 | +15.06 | 35 |

**Top winner substantively:** "Wait for ETH ≥3% off 7d high before BUY in bull regime, ≥5% off in bear regime, only enter at ≥90% confidence." Pure mean-reversion entry filter on top of model signal — no exit-side change, no shield change.

**Dead ideas tonight (NULL or NEGATIVE on 60d):**
- T5a sell-conf uplift (asymmetric exit) — NEGATIVE
- T5d per-regime min_sell_pnl — all variants ≤0 on 60d
- T5e rally-momentum exit override — all NEGATIVE (force-selling on momentum is wrong direction)
- T5f days-down entry — disastrous (−36 to −55pp)
- T5g conf-weighted max_hold extension — all NEGATIVE
- T5h shield auto-off at high profit — 0.00 (never fires in 60d sample; shield rarely binds when profit is high)
- T5i vol-percentile entry gate — small/null
- **T7 meta-labeling proxy** — every config NEGATIVE on 60d (best −10.76pp). Cheap proxy doesn't replicate literature claims; would need full LGBM-based meta from `crypto_trading_system_meta.py`
- **T8 GDELT overlay** — every config 0 to −26pp. GDELT data only covers 21% of recent signals (feed died 2026-04-19); overlay shows zero current value

**Untested combo worth running next:** T5b winner (#1) + T5c trailing peak=3% retain=70% — orthogonal mechanisms (entry filter vs exit lock), addresses today's "fear of give-back" concern directly.

**Promotion gate (per standing rule R5):** before live deployment, validate top T5b config on 4mo HRST replay (`--replay 2880`) for structural-consistency check. ~6-8h desktop runtime.

**Implementation cost:** T5b winner = config-only change (no code) — `bull.min_confidence: 80→90`, plus new keys `bull.dd_from_7d_high_min_pct: 3.0` and `bear.dd_from_7d_high_min_pct: 5.0` (these need ~30 lines of code added to live trader to compute dd_from_7d_high at each tick and gate BUYs). T5c trailing lock = ~40 lines, peak tracking + force-SELL bypass.

**Still unexplored (architectural, larger scope):**
- Triple-barrier as LABEL (not exit overlay) for retraining — new Mode D, ~3-6h
- Real meta-labeling with full LGBM + 25-feature set via `crypto_trading_system_meta.py` HRST run, 30-60 min — earlier biased run showed +23pp; clean re-run needed
- GDELT re-download + Mode F re-rank with current importance — small cost, but T8 overlay shows existing data has no signal in current regime

---

**🛎️ FIRST: check `output/ERRORS_INBOX.md` for runtime errors**

The live trader appends runtime warnings/errors to `output/ERRORS_INBOX.md`
(rate-limited to 1 entry per unique key per hour). When the user asks
"what's on my TODO" or similar status review, READ THIS FILE FIRST and
summarize any recent entries inline with the TODO review. Each entry has
severity (ℹ info / ⚠ warn / 🚨 critical). Critical entries should be
surfaced prominently — they usually mean the trader is running degraded
(FEATURE_SET_A fallback, regime detector errored, upstream data missing).

After review, user will either:
- Tell me to investigate/fix → find the root cause in code + logs
- Tell me to clear/ignore → delete the relevant lines from the inbox file

Inbox location: `output/ERRORS_INBOX.md`. Starts empty — only populates
when the trader hits an alert path. Absence of file = clean slate.

Stdout is also always mirrored to `logs/ed_runtime_*.log` per-launch
(Fix #5D 2026-04-24), so console-only warnings survive restart. Grep
there for post-incident forensics.

---

**✅ TRADER AUDIT FIXES — all deferred items resolved (2026-04-26):**

Bundles 1+2+3 (commits `8766d05` → `1124447`) shipped 14 distinct money-correctness bugs. Bundle audit findings deferred at the time were re-evaluated 2026-04-26 and resolved as follows:

- ✅ **M-10** — Orphan in-flight trade reconciliation shipped in commit `1124447` via `/trades/private/{symbol}` query in sync_positions.
- ✅ **N-15** — cycle_metrics now records a row on early-return paths (load_data_failed, regime_error, no_models_loaded). New `skip_reason` column; CSV auto-rotates to `cycle_metrics.v1.csv` on schema change.
- ❌ **M-20** — OBSOLETE (verified 2026-04-26). Manifest regen subprocess only fires when `regime_config_ed.json` actually changes (`_reload_trading_config` dict-equality gate). Original "every 5 min" framing was wrong.
- ❌ **N-06** — OBSOLETE (verified 2026-04-26). `_atomic_write_json` callers are all sequential within `crypto_trading_system_ed.py`; no threading. PID-suffixed tmp paths handle cross-process safely.

**Falsely flagged or auto-mitigated (kept for audit history):** M-05 (auto-mitigated by M-02/M-03), M-08 (ledger-delta captures), M-09 (unreachable + ledger-delta), M-11/M-12/M-14 (defensive only), M-18 (`/pause` not gating sync is intentional).

Rollback anchor: `git tag pre-trader-bundle-20260425` (commit `8766d05`).

---

**✅ LIVE TRADER DATA-UPDATE SANITY CHECK — fully shipped (originally 2026-04-23 86%-pinned bug response):**

Original incident: ETH 5h@86% stuck for 7+ hours because `xa_btc_lag2h` was NaN (BTC OHLCV stale 49h since BTC was disabled/not downloaded). `dropna(subset=feature_cols+['label'])` killed every recent row, `i=n-1` pointed at a 49h-old row, model retrained on the same frozen bar → identical 86% every hour. Prior 7h@99%/31h pinned bug was same class (likely `oc_mvrv_chg1d` stale).

All five hard rules now enforced in code:

1. ✅ **Refuse to predict on stale data** — M-01 in `crypto_live_trader_ed.py`. Computes `lag_hours` after dropna, returns None + Telegram alert if >2h.
2. ✅ **Don't drop whole rows on single-feature NaN** — M-01b decoupled `df_train` (label-NaN dropna) from `df` (all rows kept for inference). `keep_label_nan_tail=True` flag on `build_all_features`.
3. ✅ **BTC-in-ETH feature pipeline (structurally safe)** — `leaders_for['ETH']` is still `['BTC']` at `crypto_trading_system_ed.py:1281`, but M-25 cadence-aware staleness check makes the original failure mode impossible: trader refuses to predict if any non-sparse feature is >2h stale. Lead-lag is real signal per literature; removing it would lose alpha for no safety benefit. Rule's intent satisfied without the literal removal.
4. ✅ **Pre-inference data-freshness gate** — M-25 cadence-aware: `oc_*` 60h, `fg_*` 36h, hourly defaults 2h. Per-feature SLA in `config/feature_sources.json`. Refuses cycle on stale.
5. ✅ **Sparse-tail audit** — `tools/audit_features.py:134-169` scans last 48 bars of every prod-CSV feature, flags NaN count + last-valid lag, lists models using each.

---


## 📦 MERGED TOPIC: AB matrix evolution (2026-04-22 planning → 2026-04-25 4-variant → 2026-04-27 ABCDE final)

*This topic groups several banners that were previously written separately as the work progressed. Content kept verbatim, just reordered chronologically and merged under one header.*

---

**🧪 [UPMOST ACTION — overnight desktop] AB MATRIX: trim × meta × vol (2026-04-22 evening).**

Orchestrator: `tools/ab_matrix_runner.py`. Runs the full factorial on a SINGLE MACHINE with a FROZEN DATA SNAPSHOT so every variant sees identical bars. All runs use `--no-persist` + new `--no-data-update` flag. Writes consolidated audit CSV `output/ab_matrix_results_<timestamp>.csv` with per-horizon feature lists, confidences, shields, gates, detector, meta stats — everything needed to audit each variant's strategy.

**Data freezing:** at matrix start the runner snapshots 13 data files (ETH/BTC OHLCV, macro_hourly/daily, cross_asset, fear_greed, onchain_eth/btc, derivatives_eth/btc, stablecoin, orderbook, options_iv) into `data/_ab_snapshot_<timestamp>/`. Before every variant, the runner restores from snapshot. HRST is invoked with `--no-data-update` so the usual download step is skipped. Live trader keeps updating the real files independently — snapshot is read-only from the matrix's perspective.

**Matrix (4 HRST + 1 vol test = ~15-20h):**

| Variant | Trim (Mode F) | Meta filter | Purpose |
|---|---|---|---|
| trimOFF_metaOFF | disabled | none | Pre-trim baseline (original Ed) |
| trimON_metaOFF | enabled | none | Current prod behavior (Mode F only) |
| trimOFF_metaON | disabled | p=0.45 | Meta-only effect |
| trimON_metaON | enabled | p=0.45 | Combined — the proposed future prod |
| vol_scaled | — | — | `test_vol_scaled_horizon --replay 1440` — orthogonal detector test |

**ONE command to launch the whole matrix (desktop):**
```bash
python tools/ab_matrix_runner.py
```

Optional flags:
- `--dry-run` print the plan without executing
- `--skip-vol` skip the vol-scaled horizon test

**What the orchestrator does per variant:**
1. Restores all 13 data files from the frozen snapshot (identical data)
2. Flips `config/disabled_features.json.enabled` to trim True/False
3. Runs HRST with `--no-persist --no-data-update` + optional `--meta-filter 0.45`
4. Parses the HRST log for: Mode S winner (detector + bull/bear horizon + confidence), Mode T policy (shield per regime, min_sell_pnl, max_hold), Mode T REF baseline (H1/H2/REF returns), bull+bear gate winners, convergence iteration, meta filter stats (kept/dropped/no_pred counts)
5. Parses the tagged `crypto_ed_production_noprod_<label>.csv` to pull per-horizon: best_combo, best_window, gamma, return, accuracy, n_features, full feature list, logret count, pysr count
6. Tags all 3 `_noprod.*` files (config + production + best_models) with the variant label
7. Appends row to consolidated CSV (incremental so interruptions don't lose data)
8. Restores original trim state at end (via try/finally)

**Audit columns in the CSV (per variant row):**
- Identity: variant, trim_enabled, meta_threshold, exit_code, runtime_min, timestamp, log_path, tagged_prod_csv
- Mode S winner: detector, bull_h, bull_conf, bear_h, bear_conf
- Mode T policy: t_bull_shield, t_bear_shield, t_bull_conf, t_bear_conf, t_min_sell_pnl, t_max_hold
- Mode T returns: t_ref_pct (60d without gate), t_h1_pct (recent half), t_h2_pct (older half), t_converged_iter
- Gates: bull_gate, bear_gate (formula strings or "OFF")
- Meta: meta_kept, meta_dropped, meta_no_pred, meta_threshold
- Per-horizon detail (for h=5,6,7,8): h{h}_source (e.g. "Refined#1"), h{h}_apf, h{h}_combo, h{h}_window, h{h}_gamma, h{h}_return, h{h}_accuracy, h{h}_n_features, h{h}_features (pipe-separated list), h{h}_logrets (count), h{h}_pysr (count)

**When finished, compare the CSV:**
- Order variants by `t_ref_pct` (Mode T's unshielded/ungated 60d return)
- Check if trim helped (trimON > trimOFF with same meta)
- Check if meta helped (metaON > metaOFF with same trim)
- Check for interaction (does trim+meta combine linearly or is there synergy?)
- Look at `meta_dropped` to see how aggressive the filter is in each variant

**Decision tree:**
- Best variant's Mode T REF > current laptop 2mo+trim baseline (+86.79%) → **promote it** (when trade closes)
- Best variant close to current laptop result, simpler (e.g., trim-only works as well as trim+meta) → **ship the simpler one**
- No variant beats laptop result by >10pp → **stick with laptop config** from yesterday (already tagged `_2mo_backup`)
- Vol test winner differs from HRST's tsmom → **schedule detector A/B separately**

~~**G. Feature-family floor A/B**~~ — **FOLDED INTO AB MATRIX 2026-04-22 evening.** Matrix variant #5 (`trimON_metaON_floorOFF`) vs variant #4 (`trimON_metaON` with floor ON) isolates the floor effect in a single overnight run. No separate G workflow needed.

**Infra shipped 2026-04-22 (still applies — default for all future HRST unless `--no-feature-floor`):**
- New constants in `crypto_trading_system_ed.py`: `FEATURE_FLOOR_ENABLED=True`, `FEATURE_FLOOR_MIN_LOGRET=2`, `FEATURE_FLOOR_MIN_PYSR=1`
- Helper `_feature_floor_indices(ranked_features, n_feat)` — picks column indices guaranteeing ≥2 logret + ≥1 pysr in every selected subset. Promotes essentials from beyond position N, evicts lowest-ranked non-essentials in the top-N slice. Exactly N features out, no overflow. Graceful no-op if ranked list lacks essentials.
- Wired into 4 sites: Mode D grid + refine + top-candidates CSV + refined-candidates persist
- CLI flag `--no-feature-floor` disables the floor for A/B comparison (now used by matrix variant #5)
- Unit tests: 3/3 pass (already-compliant no-op, vol-only promotion, no-essentials graceful)
- Motivation: ETH 5h/6h prod + laptop trim configs both ended up with **0 logret + 0 pysr** in their final feature sets (trend-blind volatility-only models). Root cause: Mode D's Optuna refine climbs APF on the 1440h window and lands at sparse vol-only local optima.

**PRIORITY 2 — Asset enablement decisions:**

5. ~~**SOL HRST**~~ — **DONE 2026-04-19.** Config written: `sma168>sma480` | bull=8h@90% shield=ON / bear=8h@65% shield=OFF | 0.55%/12h | no gates. Pipeline health 6/7 positive, 3 horizons at +19-23% range. **Decision pending user: small test allocation ($2-4k) or shelve.** Bottleneck: best single-horizon +23% vs ETH's +52%, so per-$ alpha likely ~40-50% of ETH — borderline on the "≥50% threshold" rule.

6. ~~**LINK HRST**~~ — **DONE + SHELVED 2026-04-20.** Config: `vol_calm` | 7h@95% shield=OFF / 6h@95% shield=ON | 0.6%/12h | no gates. Pipeline weak: 5/8 horizons NEGATIVE, `beats_3of3=0` on gate sweep (49k configs, zero beat baseline). Model can't find reliable signal in LINK's data. `LINK.enabled: false` stays.

7. **XRP HRST** (only remaining untested) — Launching Mode P first (files from 2026-03-26 are pre-deep-PySR). Correlation with ETH ~0.50-0.70 → real diversification if it works. Priors were +9.99% on 1mo (Mode H 2026-03-26). Command: `python crypto_trading_system_ed.py P XRP 5,6,7,8h; if ($?) { python crypto_trading_system_ed.py HRST XRP 5,6,7,8h --replay 1440 }`. Optional — after the 3 shelved results (BTC/SOL-borderline/LINK-weak), diversification case for more crypto assets is thin. Expect XRP likely similar.

**PRIORITY 3 — Research:**

7. **Orderbook imbalance + IV skew accumulation** — Hourly snapshots now wired into Ed trader (`crypto_revolut_ed_v2.py`). Need ~2 weeks of data before testing as features. Currently ~36 rows each (as of 2026-04-21).

8. **Eli HRS BTC** — 30-minute candle test. Separate research track.

9. **Ein results review** — 15-minute candle BTC results from earlier laptop run. Separate research track.

10. **Grade-4 on-chain expansion after newborn cool-down** — `oc_mvrv_chg1d` (Grade 3-4 on BTC/ETH) is the only on-chain metric earning its keep. After basis + lead-lag newborns prove in/out, re-audit and consider disabling more macro derivatives (esp. `m_oil_*`, `m_eurusd_*`, `m_usdjpy_*` 5d/10d/zscore variants).


---

**🧪 PRIOR LAUNCH — AB MATRIX 4-VARIANT FOCUS (2026-04-24 22:40 CEST → completed 2026-04-25 12:38 CEST, partial — D never ran due to system freeze):**

Launched AFTER the third-pass audit shipped (file-handle leak + div-by-zero guards, commit `eaf80e9`) and AFTER today's 22:07 HRST promoted a clean winner (`sma168>sma480 bull=7h@65% bear=6h@85%` — already live). This matrix runs on the first fully-clean data snapshot: 1432/1440 rows, label-tail fix, sparse-feature quarantine, div-by-zero guards, atomic writes — none of the prior 3 weeks' results were trained on this data quality.

Command: `python tools/ab_matrix_runner.py --variants focus --skip-vol` (seed 42 default, matches today's live HRST for B-variant replication check).

| Variant | Floor | Trim | Meta | Purpose |
|---|---|---|---|---|
| A_floorON_trimOFF | ON | OFF | — | Floor alone on full 184-feature universe — does floor still matter without trim? |
| B_floorON_trimON | ON | ON | — | **Replicates today's 22:07 live HRST.** Sanity check that matrix subprocess = direct invocation (same seed + data). |
| C_floorOFF_trimOFF | OFF | OFF | — | Raw universe, no guarantees. Tests whether floor's feature-family floor is doing real work. |
| D_floorON_trimON_metaON | ON | ON | p≥0.45 | **R3 meta-labeling retest** — added 2026-04-24 22:40 (commit `23b73c1`). B↔D isolates meta contribution on clean primary. |

Runtime: 4 × ~4h = ~16h laptop. Results in `output/ab_matrix_results_<timestamp>.csv` + tagged `_noprod_{A,B,C,D}.*` files.

**Decision rules when matrix finishes (2026-04-25 afternoon):**
- **B replicates today's HRST** (detector + horizons + confs match within seed noise): sanity check passes, matrix infrastructure trustworthy.
- **Best (A/B/C)** alpha > today's live HRST by ≥5pp on Mode T REF: promote that variant instead.
- **D > B by ≥5pp**: meta filter is shippable behind a config flag (resolves R3).
- **D within ±5pp of B**: shelve meta permanently — was dead-end masquerading as signal on biased data.
- All within noise: keep today's live HRST running, revisit after next week's live performance.

**Earlier (superseded) matrix results — kept for historical record only:**
The 2026-04-22 17:32 matrix promoted V1 (`tsmom_672h 5h@85%/6h@80%` +122.59% Mode T). The 2026-04-24 07:40 seed-2026 relaunch promoted intermediate variants. Both were trained on poisoned data (672 rows + label-tail NaN + div-by-zero features). **Do not reference those numbers as baselines** — they're not comparable to this matrix's clean output.

---


---

**✅ ABCDE MATRIX FULL RESULTS — completed 2026-04-27 01:18 CEST. R3 (meta-labeling) RESOLVED — SHELVED.**

All 5 variants ran to completion. Output CSV: `output/ab_matrix_results_20260426_151744.csv` (last variant E timestamp).

**Final ranking by Mode T REF (canonical 60d return metric):**

| Rank | Variant | t_ref % | Detector | Bull | Bear | Trim | Floor | Meta | Bull Gate | Bear Gate | Conv iter | Runtime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **A_floorON_trimOFF** ← LIVE since 2026-04-26 14:22 | **+66.79%** | `price>sma72` | 6h@80% shield ON | 7h@85% shield ON | OFF | ON | — | `rr20≥4.0 OR rr24≥4.5 cd=12h` | `rr30≥9.0 OR rr36≥9.0 cd=48h` | 2 | 447 min |
| 2 | E_floorON_trimOFF_metaON | +64.67% | `sma168>sma480` | 8h@75% shield ON | 5h@65% shield OFF | OFF | ON | p≥0.45 | OFF | OFF | 2 | 601 min |
| 3 | C_floorOFF_trimOFF | +62.68% | `sma24>sma100` | 6h@80% shield ON | 5h@65% shield OFF | OFF | OFF | — | OFF | OFF | 2 | 237 min |
| 4 | D_floorON_trimON_metaON | +51.65% | `vol_calm` | 5h@75% shield ON | 6h@65% shield OFF | ON | ON | p≥0.45 | rr12≥3.5 OR rr20≥3.0 cd=18h | rr30≥9.0 OR rr36≥8.5 cd=10h | 3 | 451 min |
| 5 | B_floorON_trimON | +45.39% | `sma24>sma100` | 6h@80% shield ON | 5h@65% shield OFF | ON | ON | — | rr20≥5.5 OR rr30≥6.5 cd=30h | rr8≥6.0 OR rr10≥6.0 cd=6h | 3 | 452 min |

**Verdicts (knob isolation):**

- **Trim = OFF preserved.** A vs B (same floor, only trim differs): A=+66.79 vs B=+45.39 → trim costs **−21.40pp**. Decision unchanged from 2026-04-26.
- **Floor = ON preserved.** A vs C (same trim, only floor differs): A=+66.79 vs C=+62.68 → floor adds **+4.11pp**. Decision unchanged.
- **Meta filter SHELVED (R3 RESOLVED).** Two ways to read meta:
  - On no-trim: E (meta+no-trim+floor) = +64.67% vs A (no-meta+no-trim+floor) = +66.79% → **meta costs −2.12pp on the strong baseline.** Below the +5pp shipping threshold.
  - On trim: D (meta+trim+floor) = +51.65% vs B (no-meta+trim+floor) = +45.39% → **meta adds +6.26pp on a weaker baseline.** But that combined config still loses to A by −15.14pp.
  - **Conclusion: meta only helps when something else is broken (trim hurting). With the right primary config, meta adds nothing.** Permanently shelve for ETH unless future runs on different data show clear benefit.

**Production decision: NO PROMOTION CHANGE.** Variant A (already live since 2026-04-26 14:22 CEST) remains the winner across the full 5-variant matrix. The most-deferred test (D, then E) confirmed meta-labeling does not justify a config change.

**Closes 2026-04-26 deferred items:**
- ✅ R3 meta-labeling decision: **SHELVED** (E/A delta = -2.12pp, below +5pp ship threshold)
- ✅ Meta-aware variants (D, E) both run to completion — no infra gaps remain
- ✅ A/B/C/D/E full factorial complete on clean data snapshot
- Matrix infra trustworthy (A's reported t_ref +66.79% matches A's live behavior post-promotion)

---

**🔄 RETEST PRIORITIES AFTER MATRIX FINISHES (ETH-first, clean data post-dropna-fix 2026-04-24):**

All decisions from 2026-03-20 onwards were made on biased data (dropna eating ~half the window due to sparse-feature NaN). ETH-focused retests come first; secondary assets (SOL/LINK/BTC) lower priority — fix ETH robust FIRST.

**R1. [HIGH — ETH core] Promote clean matrix winner.**
When matrix finishes (~17:00 today), compare A/B/C variants' Mode T finals. Whichever converged + has top alpha + passes 4 promotion gates:
```powershell
# Backup current V4
copy config\regime_config_ed.json config\regime_config_ed_v4_pre_clean.json
copy models\crypto_ed_production.csv models\crypto_ed_production_v4_pre_clean.csv
# Promote the matrix winner (replace {LABEL} with actual winner: A_floorON_trimOFF, B_floorON_trimON, or C_floorOFF_trimOFF)
copy config\regime_config_ed_noprod_{LABEL}.json config\regime_config_ed.json
copy models\crypto_ed_production_noprod_{LABEL}.csv models\crypto_ed_production.csv
# Clear cooldown
# Edit config/position_ed_v2_ETH.json, set "rally_cooldown_until": ""
```
**Decision tree:** best variant's alpha > V4's live performance-adjusted expectation → promote; within noise → keep V4, re-evaluate next week.

**R2. [HIGH — ETH validation] 4-month HRST on clean winner.**
Structural-consistency check — does the matrix winner hold on a longer window?
```bash
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 2880 --no-persist
```
~6-8h desktop runtime. Tag outputs `_4mo_clean`. Compare Mode S winner to the 2mo clean winner:
- Same detector + horizons → strong structural confirmation → ship with confidence
- Different winner → window-sensitivity still present; use 2mo for current market regime (per standing rule R7)
Prior biased 4mo run gave `tsmom_672h 6h/7h @ +155.91pp` — that number is inflated. Clean 4mo will be more honest.

**R3. [HIGH — ETH, conditional] Meta-labeling retest on clean signals.**
Only if matrix's variant C (floor OFF) or B (trim+meta) shows meta as a contender. Prior standalone meta harness showed +23.21pp on raw signals at p=0.60 and strategy-aware +15.76pp — **both biased**. Retest with the NEW clean primary model (from R1) as the base:
```bash
python crypto_trading_system_meta.py ETH 5 --replay 1440
python crypto_trading_system_meta.py ETH 6 --replay 1440
python crypto_trading_system_meta.py ETH 7 --replay 1440
python crypto_trading_system_meta.py ETH 8 --replay 1440
python tools/test_meta_strategy_impact.py ETH --replay 1440
```
~30-60min total. Decision: ≥+10pp on clean data at any threshold → ship behind trader flag; <+5pp → shelve meta permanently.

**R4. [HIGH — ETH label] Label-threshold 1% retest on clean data.**
Previous test (−47pp delta) was run on biased data. Direction was clear but absolute numbers untrustworthy. Quick confirm on clean data:
```bash
python crypto_trading_system_ed.py DV ETH 5h --replay 1440 --label-threshold 0.01
```
~45min. Expected: still negative delta (direction robust). If suddenly POSITIVE on clean data, reopen the label-threshold question.

**R5. [HIGH — ETH gate] Bull rally-gate retest on clean data.**
Prior evidence (bull gate hurts: live 30d −7.48pp, OOS 60d −1.31pp, FULL 90d −3.32pp) was on biased data. The consistency across 3 windows suggests the direction is real, but retest:
- Check V4's current bull-gate (whichever the matrix winner picks) against a disabled-gate variant
- Manual test: after promoting R1's winner, run 1 week with bull gate ON, compare to baseline simulation
- Or: compare matrix A's bull gate output vs A with `/gate ETH bull off` post-promotion

**R6. [MEDIUM — ETH gate] Drop-gate sweep retest.**
Prior rejection (101/126 OOS rank = overfit). Direction robust to bias. Only retest if meta + 4mo + bull-gate all favor complex gate structure.
```bash
python tools/backtest_drop_gate.py  # already exists
```
Low priority — run only if time.

---

**🟡 LOW PRIORITY — SECONDARY ASSET RETESTS (only after ETH is robust):**

Don't touch these until ETH promotion + 4mo validation + meta decision are done. Running ETH-first preserves compute for ETH-specific tuning.

**R7. [LOW] SOL HRST retest.**
Prior result (2026-04-21): `sma168>sma480 5h@65%/8h@70%` +42.30%/+40.37pp alpha (biased). Retest:
```bash
python crypto_trading_system_ed.py HRST SOL 5,6,7,8h --replay 1440 --no-persist
```
~3-4h desktop. Decision: if clean alpha ≥ 50% of ETH's clean alpha → commit $2-3k. Otherwise shelve.

**R8. [LOW] LINK HRST retest.**
Shelved 2026-04-20: "5/8 horizons NEGATIVE, beats_3of3=0". Could be entirely a dropna artifact. Same command pattern as R7, replace SOL with LINK. If LINK now converges with decent plateau → unshelve. If still weak → permanently shelve.

**R9. [LOW] BTC HRST retest.**
Last done 2026-04-20 with +36.15%/+23.89pp alpha. Biased. Retest for reliability check — BTC is the simplest diversification target but wasn't enabled. Same command with BTC.

**R10. [LOW] XRP HRST (never tested on clean data).**
Only remaining untested asset. Mode P first if JSONs are stale:
```bash
python crypto_trading_system_ed.py P XRP 5,6,7,8h
python crypto_trading_system_ed.py HRST XRP 5,6,7,8h --replay 1440 --no-persist
```

---

**🟢 STILL TRUSTED (no retest needed — pre-bug-era or bias-independent):**

- Stop-loss / take-profit / trailing-stop variants (tested in earlier era, before sparse-feature pipeline)
- LSTM ensemble (pre-bug era)
- V1.7.2 regularization (pre-bug era)
- 4h horizon rejection (structural embargo issue, not bias)
- GDELT disable (feature selection evidence, not bias-dependent)
- PySR discovery leakage check (uses dedicated historical window, excludes Mode D's replay period)

`python tools/ab_matrix_runner.py` launched. 5 HRST variants + 1 vol test, all on a frozen data snapshot with `--no-persist` + `--no-data-update`. Safe to run alongside live trader; position stays open.

| # | Variant | Trim | Meta | Floor |
|---|---|---|---|---|
| 1 | trimOFF_metaOFF | OFF | OFF | ON |
| 2 | trimON_metaOFF | ON | OFF | ON |
| 3 | trimOFF_metaON | OFF | p=0.45 | ON |
| 4 | trimON_metaON | ON | p=0.45 | ON |
| 5 | trimON_metaON_floorOFF | ON | p=0.45 | OFF |
| 6 | vol_scaled | — | — | — |

**Outputs:**
- Full audit CSV: `output/ab_matrix_results_<timestamp>.csv` — detector, bull/bear horizons + confidences, shields, gates, min_sell_pnl, max_hold, Mode T REF/H1/H2 returns, meta stats, per-horizon combo/window/gamma/features/logret_count/pysr_count
- Tagged `_noprod_<label>.{json,csv}` per variant in `config/` and `models/` — can be promoted directly if decision says so
- Log per variant: newest `logs/ed_v1_*.log` around each run's start time
- Data snapshot preserved: `data/_ab_snapshot_<timestamp>/` (delete if not needed after analysis)

**Progress monitoring:** tail the latest `logs/ed_v1_*.log`; check CSV has grown after each variant (~2.5-3.5h apart).

**Decision dimensions when it finishes:**
- **Trim effect**: #2 vs #1 (trim only), #4 vs #3 (trim with meta active)
- **Meta effect**: #3 vs #1 (meta only), #4 vs #2 (meta with trim active)
- **Floor effect**: #5 vs #4 (same config, floor off vs on) — tiebreaker on shipped change
- **Vol-scaled detector**: #6 vs whichever primary variant wins
- **Winner**: highest `t_ref_pct` that also passes promotion gates (Mode T converged, top-15 plateau, trend features present)

---

~~**🚨 BLOCKING — POST-RALLY PROMOTION WORKFLOW**~~ — **RESOLVED 2026-04-22 late night:** user sold manually at ~$2,388 (+3.2% realized on the 2026-04-21 trade) and immediately promoted **Variant #1** (not laptop) to production. See "Closed 2026-04-22 late night" below for details.

Historical context (kept for audit):

**R0. [BLOCKING] Realized PnL logging + trade postmortem.**
When the SELL fires, record: entry price, exit price, exit reason (model/shield/max_hold), realized %, hold duration. Store in `output/live_trades_2026-04-21.md` or similar for the rally-retrospective decomposition. Expected exit somewhere between +2% and +5% depending on when prod's 7h model flips.

**R1. [BLOCKING] Compare realized vs laptop's hypothetical on this trade.**
Laptop model had SELL signal at 03:00Z ($2,373, +2.59%), stronger at 05:00Z / 06:00Z ($2,364 / $2,396, +2.2%/+3.6%). Actual prod exit price tells us which approach won on THIS trade. Single-trade is noise but informative.

**R2. [HIGH] Promote laptop config to production — cash-state switchover.**
```powershell
# Backup current prod
copy config\regime_config_ed.json config\regime_config_ed_2mo_pre_laptop.json
copy models\crypto_ed_production.csv models\crypto_ed_production_2mo_pre_laptop.csv

# Promote laptop
copy config\regime_config_ed_noprod.json config\regime_config_ed.json
copy models\crypto_ed_production_noprod.csv models\crypto_ed_production.csv

# Clear cooldown timer — fresh gate state
# Open config/position_ed_v2_ETH.json, set "rally_cooldown_until": ""
```
Trader hot-reloads within 5 min. Monitors:
- alpha tracking sim +68.37pp over first 10+ trades
- bull gate (`rr14h≥6.0 OR rr20h≥5.5 cd=10h`) — fires rarely, shouldn't block normal BUYs
- bear shield (now ON) — holds bear-regime trades through initial dip

**R3. [DECIDE DURING R2] Option C variant — bull gate OFF?**
Laptop's bull gate fires only on 6%+ rallies (rare). But evidence across multiple tests (live 30d, OOS 60d, FULL 90d) shows bull gates generally hurt or break even. Consider immediately:
```
/gate ETH bull off
```
Keeps laptop's other settings (shield flip, bear gate, detector) but removes even the rare-firing bull gate. If this test runs parallel to R2 mentally, can flip back with `/gate ETH bull on` if we see harm.

**R4. [HIGH] Rollback safety net — if laptop underperforms in live.**
Threshold: if cumulative realized alpha drops >15pp vs sim baseline over the first 10 trades, rollback:
```powershell
copy config\regime_config_ed_2mo_pre_laptop.json config\regime_config_ed.json
copy models\crypto_ed_production_2mo_pre_laptop.csv models\crypto_ed_production.csv
```
Also stops the bull-shield=OFF experiment mid-regime if it's clearly failing.

**R5. [MEDIUM] 4-month replay confirmation HRST for laptop config.**
Once laptop is in prod and running, validate on a longer window:
```bash
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 2880 --no-persist
```
~3h. Compare Mode S winner to current laptop config. If detector/horizons shift again on 4mo → ensemble-of-windows question reopens. Also serves as the "accept or reject" tiebreaker on the 2mo vs 4mo window-sensitivity question still open from 2026-04-21.

**R7. [STANDING POLICY] HRST re-sweep cadence.**
Evidence from 2026-04-21/22: three ETH HRST runs within 9h gave 3 completely different winners (detector, horizons, shield, gates all flipped). Running HRST daily chases 5-10pp nondeterminism noise as if it were signal.

**Standing rules going forward:**
- **Scheduled HRST**: every 2-4 weeks (material training-window shift)
- **Mandatory HRST**: always after a Mode P (PySR) run — else silent feature drift (rule 14)
- **Event-driven HRST**: after material regime shift, after new feature block added, after live >15pp underperformance over 10+ trades
- **Research HRST**: always `--no-persist`

**Promotion gate (must pass ALL four before writing to prod):**
1. Mode T converged (not `max_iter` hit)
2. Top-15 plateau: ≥10/15 configs share detector + bull horizon
3. Alpha > current-prod live alpha + nondeterminism margin (~10pp)
4. Structural consistency: if 4mo and 2mo disagree on detector, neither is promoted without a tiebreaker run

By these rules, today's 2mo Desktop HRST (hit max_iter, was scattered in top-15) **should not have been promoted**. The laptop 2mo+trim result passes all four gates — it's what should go live post-rally (R2).

**R6. [MEDIUM] SOL enablement decision with today's strong HRS result.**
SOL HRS 2026-04-21 15:23: `sma168>sma480 5h@65%/8h@70%` → **+42.30%/+40.37pp alpha**, 82% of ETH's alpha (up from borderline 40-50% yesterday). Clearly clears the ≥50% threshold. Consider $2-3k initial allocation. Watch correlation with ETH — if SOL live tracks ETH >70%, scale back.

---

**PRIORITY 1 — Open strategy tests (state at 2026-04-21 evening):**

~~**A. Large-upside label test — ETH 5h DV with `label = ret ≥ 1%`**~~ — **DONE + REJECTED (2026-04-21 00:37).** Ran `DV ETH 5h --replay 1440 --label-threshold 0.01` on Desktop overnight. Winner: RF+LGBM w=250 g=0.999 n_features=10 → **+11.87% return, 72% accuracy**. Current prod ETH 5h baseline: **+59.11% return, 65.9% accuracy**. Net **delta −47.24pp return** despite +6.1pp accuracy gain. Classic tighter-label regression: positive-class starvation reduces trainable signal; model becomes more selective at the cost of volume. Output preserved at `models/crypto_ed_production_lt1.csv`. Don't try `--label-threshold 0.005` — the direction is clearly wrong. **Label stays at `2×fee = 0.22%`.** Takeaway: the small-wins-are-noise intuition is right at the TRADE level but the MODEL extracts edge from volume, not individual trade contribution. Real execution gap (~17pp after bull-gate fix) is execution friction, not label choice.

~~**B. Feature-trim validation HRST**~~ — **DONE 2026-04-22 03:03** (log `ed_v1_20260421_212008.log`, laptop, 6h runtime). Winner: **`price>sma72` bull=6h@75% / bear=5h@75%, shield bull=OFF/bear=ON, min_sell_pnl=0.50%, max_hold=10h**, converged Mode T iter 4. Bull gate `rr14h≥6.0%/rr20h≥5.5% cd=10h` (rare firing), bear gate `rr8h≥3.0%/rr12h≥2.0% cd=16h`. 60d sim: **+86.79% return / +68.37pp alpha / 118 trades / 75% WR**. Top-15 UNANIMOUS on detector/horizons — tightest plateau of any ETH HRST this week. Trim verdict: **trim helped (+18.96pp vs no-trim 2mo prod)**. Feature matrix: 131 total (60 disabled, vs 191 without trim). Promotion workflow → see R2 above.

~~**C. Window-sensitivity 3-way comparison**~~ — **RESOLVED 2026-04-22.** Desktop 2mo = `sma168>sma480 7h/5h @ +49.41pp`, Desktop 4mo = `tsmom_672h 6h/7h @ +155.91pp`, Laptop 2mo+trim = `price>sma72 6h/5h @ +68.37pp`. All three pick DIFFERENT detectors + different horizon combinations — HRST is genuinely window-sensitive. User's regime-call: 2mo matches current market; 4mo includes a bear phase that inflated its alpha. Laptop's 2mo result is the tiebreaker winner (trim-validated, plateau-unanimous, Mode T converged, structurally similar to 4mo on shield config). 4mo kept backed up (`config/regime_config_ed_4mo_backup.json`) as insurance.

**D. Vol-scaled horizon 4-month validation — ~30 min, laptop free.**
```bash
python tools/test_vol_scaled_horizon.py --replay 2880
```
Prior: vol_2band (low→8h, high→6h) beat tsmom regime by +5.02pp on 2-month. Need 4mo confirmation before replacing tsmom_672h detector. Note: current live detector is now `sma168>sma480` (prod) but next promotion will switch to `price>sma72` (laptop) — this vol-scaled test operates orthogonally to whichever detector we're on.

~~**E. Meta-labeling — full HRST A/B test**~~ — **SUPERSEDED by AB MATRIX (running 2026-04-22 evening).** Standalone `--meta-filter` flag still works for one-off tests but the full strategy comparison is now covered by the matrix's variants #3/#4 vs #1/#2. Earlier E2/E3 concurrent runs on 2026-04-22 collided (both writing to `_noprod.*` across machines); that attempt's logs preserved for historical reference but results contaminated. Matrix runs on single desktop with data snapshot — clean comparison.

**Infra shipped 2026-04-22:**
- `--meta-filter P` CLI flag added to `crypto_trading_system_ed.py` (auto-enforces `--no-persist`)
- `_apply_meta_filter_to_signals()` hook at the end of `generate_signals()` — walk-forward meta per horizon, downgrades BUYs with `meta_prob < P` to HOLD
- Reuses `crypto_trading_system_meta.build_meta_dataset` + `walk_forward_meta_train` (lazy import, no circular)
- Meta predictions carried on signals as `s['meta_prob']` + `s['meta_filtered']=True` for downstream inspection

**E1. Backup current `_noprod.*` state before any runs (desktop):**
```powershell
copy config\regime_config_ed_noprod.json config\regime_config_ed_noprod_pre_meta.json
copy models\crypto_ed_production_noprod.csv models\crypto_ed_production_noprod_pre_meta.csv
```

**E2. Run A — BASELINE HRST (no meta), ~3-4h on desktop:**
```bash
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist
```
After completion, tag the outputs so Run B doesn't overwrite them:
```powershell
copy config\regime_config_ed_noprod.json config\regime_config_ed_noprod_NOMETA.json
copy models\crypto_ed_production_noprod.csv models\crypto_ed_production_noprod_NOMETA.csv
```

**E3. Run B — HRST WITH META at p=0.45, ~3-4h on desktop:**
```bash
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist --meta-filter 0.45
```
Tag outputs:
```powershell
copy config\regime_config_ed_noprod.json config\regime_config_ed_noprod_META045.json
copy models\crypto_ed_production_noprod.csv models\crypto_ed_production_noprod_META045.csv
```

**E4. Compare** — Mode S winner alpha + Mode T final return + per-regime decomposition in each run's stdout, plus `combined_score` / `return_pct` columns in the two tagged production CSVs.

**E5. Decision tree:**
- Meta run's **Mode S alpha ≥ baseline + 10pp** → **ship meta** (budget 2-3h to integrate into `crypto_revolut_ed_v2.py` behind a config flag + nightly meta model refresh)
- Within **±5pp** → null; strategy already captures meta's gains — **shelve** meta, save the integration complexity
- Meta run **worse than baseline** → meta hurts optimized strategy — **shelve** and note in memory

**E6. [if ship]** Promotion gate — run the same HRST with `--meta-filter 0.45` on 4mo replay as structural-consistency check. Same decision tree. Only then touch the live trader.

Estimated total runtime: **6-8h desktop, overnight**. Chain with `;` in PowerShell if you want unattended:
```powershell
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist; copy config\regime_config_ed_noprod.json config\regime_config_ed_noprod_NOMETA.json; copy models\crypto_ed_production_noprod.csv models\crypto_ed_production_noprod_NOMETA.csv; python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist --meta-filter 0.45; copy config\regime_config_ed_noprod.json config\regime_config_ed_noprod_META045.json; copy models\crypto_ed_production_noprod.csv models\crypto_ed_production_noprod_META045.csv
```

**F. Execution-gap investigation (no lit ref, but highest-EV lever).** The 30d gap: simulated baseline +27.11% vs live +2.54% = **~24.5pp unaccounted**. ~7.5pp was bull-gate harm (now disabled 2026-04-20). Remaining ~17pp: slippage, partial fills, manual interventions, timing, clock drift. Next: TCA logging (`implementation_shortfall = fill_price - arrival_mid` per trade), manual-vs-auto PnL decomposition, latency audit on order placement.

**G. Scheduled sparse-feature re-enable tests (do NOT auto-enable — test first).**

7 features disabled 2026-04-24 via `config/disabled_features.json` because short history was collapsing `dropna()` (only 101 clean rows out of 1440 before disable; 1432 after). Data pipelines untouched — features keep accumulating in the background. Tests scheduled for when each group has enough history for a clean 60-day replay.

**G1. 2026-05-22 — `deriv_oi_*` re-enable test:**
- Features: `deriv_oi_chg1d`, `deriv_oi_chg3d`, `deriv_oi_zscore`
- History started 2026-03-20 (Binance public OI retention ~30d). By 2026-05-22 there will be ~63 days + 72h warmup buffer for `deriv_oi_chg3d`.
- **Procedure**:
  1. `copy config\disabled_features.json config\disabled_features_pre_deriv_oi_20260522.json` (backup)
  2. Baseline: `python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist` (keep disabled list as-is). Tag outputs with `_deriv_oi_OFF`.
  3. Temporarily remove 3 `deriv_oi_*` entries from `disabled_features.json`.
  4. Test: `python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist`. Tag outputs with `_deriv_oi_ON`.
  5. Restore original `disabled_features.json` from backup.
  6. **Decision tree:** Mode T alpha ON ≥ OFF + 5pp → re-enable permanently. Within ±5pp → leave disabled. Worse → leave disabled.

**G2. 2026-06-18 — Orderbook + IV re-enable test:**
- Features: `ob_imbalance`, `spread_bps`, `avg_iv`, `iv_skew`
- History started 2026-04-19 when live Ed trader began writing `orderbook_snapshots.csv` + `options_iv_snapshot.csv`. By 2026-06-18 there will be ~60 days of coverage.
- **Procedure**: same as G1 — backup, baseline run `_ob_iv_OFF`, enable-test run `_ob_iv_ON`, restore backup, decide.

**Why test-first (not auto-enable):**
- Matrix already showed adding features can hurt (trim=ON was worse than OFF despite feature availability). Same discipline for these.
- Re-enable should be event-driven (evidence of improvement), not calendar-driven.
- These tests are cheap — ~6-8h per one-off run, no code changes needed.
- **NO automatic trigger on the date** — user decides when to run the test based on priorities.

### Closed 2026-04-25 late-afternoon (post-bundle hotfixes + bundle 3)

After bundle 1+2 shipped, user attempted to bring trader back online and surfaced **two more real-world bugs my earlier fixes had missed/introduced**. Re-audit then surfaced 3 more verified-real findings, all shipped same evening. Total of 6 additional commits on top of bundles 1+2.

**Hotfixes (post-bundle 2):**

- **`5ba0af7` HOTFIX M-01b** — `crypto_trading_system_ed.py` + `crypto_live_trader_ed.py`. The morning's M-01 fix was incomplete: `build_hourly_features` already did `dropna(subset=core_cols + ['label'])` BEFORE returning df_full. So even though I patched the live trader to keep the label-NaN tail, df_full's last row was already `freshest_raw_bar − horizon_hours` — wall-clock check saw 7h lag and refused every signal. Fix: added `keep_label_nan_tail=True` parameter to `build_all_features` / `build_hourly_features`, threaded through to live trader's call site. Engine training callers leave default False. Verified: ETH df_full last bar moved from 04:00 UTC (7.34h, REFUSE) to 11:00 UTC (0.34h, PASS). 7 rows recovered.

- **`cbc4ad3` HOTFIX N-01 + N-03 + N-05** — `crypto_revolut_ed_v2.py`. (N-01 CRITICAL): `process_asset` leaked the per-asset trade lock on three balance-API-failure early-return paths (lines 1324, 1471, 1501). One API hiccup → ETH locked out forever, blocking Telegram handlers deadlock. Pre-existing from Fix N1 (commit `5a7c834`), surfaced by re-audit. Released lock at each early-return. (N-03 HIGH latent): if all signals refused, `price=0` + `get_best_bid_ask=(0,0)` → live_mid=0 → cur_pnl=−100% → brake force-sell on phantom data. Currently dormant since brake is disabled, but trap if re-enabled. Added `live_mid > 0` guard. (N-05 HIGH): M-07 Telegram handlers used blocking `with _lk:` — would freeze Telegram for 5min during a maker order, or forever after N-01 lock leak. Replaced with `acquire(timeout=2.0)`.

- **`3b38190` N-02 MIN_TRADE_USD lowered to 299.9** — `crypto_revolut_ed_v2.py:746`. BUG 1's $0.01 safety margin shrunk `/buy ETH 300` to 299.99, which then failed `< 300` minimum check with a confusing "$299.99 below $300 min" message. Lowered threshold to 299.9 so exact-$300 inputs survive the safety margin. Anything below $299.99 still rejected.

**Bundle 3 (re-audit follow-ups):**

- **`d49cc22` N-04 + N-07 + N-08** — `crypto_revolut_ed_v2.py` + `crypto_live_trader_ed.py`. (N-04): `load_trading_config()` no longer merges DEFAULT_TRADING_CONFIG under file contents — earlier merge meant deleting a key from JSON had no runtime effect (defaults re-injected). Now JSON is single source of truth; defaults seed cold-start only. (N-07 Option A): per-feature staleness check in `generate_live_signal` — for each non-sparse feature_col, compute last-valid bar age vs latest_raw_dt; refuse if any >2h. Catches the M-01b-class case where an upstream-dependent feature source (e.g. `xa_btc_lag*` when BTC OHLCV is stale) silently ffill'd into the inference. Sparse-by-design features (orderbook, IV, deriv_oi, stablecoin, whale) are skipped. (N-08): added comment documenting why training-window features get ffill'd alongside the inference row — by design from Fix #4; LGBM handles native NaN but RF/XGB partner needs imputation; acceptable noise.

- **`1124447` M-10 orphan-trade reconciliation** — `crypto_revolut_ed_v2.py`. New helpers `get_recent_private_trades` and `_reconcile_orphan_trade` query Revolut's `/trades/private/{symbol}` endpoint to recover actual fill price/timestamp when sync_positions detects a manual BUY/SELL. Match by side + quantity (within 0.1% relative). Falls back to mid-price (Fix N2 behavior) only when no recent trade matches. Trade records get `reconciled: True` marker (vs `synced: True`) so PnL consumers can distinguish exact-basis trades from approximate.

**Disaster brake disabled (user decision, 2026-04-25):** doc previously claimed `disaster_brake_pct: 5` for ETH but the JSON had no such key, making the brake dormant. User confirmed to keep it disabled. The doc snippet now correctly shows `disaster_brake_pct: 0` with a note that M-04+N-03 fixes are in place if re-enabled later.

**Bundle 3 deferred (verified clean or low-impact):**
- **N-06** (LOW) — `_atomic_write_json` in engine has no in-process lock. Latent only; sequential code paths today.
- **N-15** (LOW) — `cycle_metrics` not flushed on `process_asset` early-return paths. Observability gap, not correctness.
- **M-20** (LOW) — Hot-reload regenerates feature manifest via subprocess every 5 min (3-10s blocking). User wants explanation before deciding.

**Status by end of day:** 16 commits since `pre-trader-bundle-20260425` tag, 14 distinct bugs fixed, trader running cleanly with full instrumentation. Position file shows post-fix BUY at 22:01 last night; M-02/M-03 ledger-delta basis ready for next BUY. M-01 staleness chain working (lag_hours ~0). All maker order knobs configurable per asset.

### Closed 2026-04-25 midday (trader audit bundle 2 — 4 commits, ~50 lines)

Bundle 2 cleared all real-bug findings from the bundle-1 audit deferral list. Commits land on top of bundle 1 (`pre-trader-bundle-20260425` tag still works as the nuclear-rollback anchor — it predates both bundles). Each bundle-2 commit is independently revertable.

- **`a8bea84` M-04** — Disaster brake now reads live exchange mid via `get_best_bid_ask(symbol)`, fallback to last-closed-candle close only if price API fails. Fix is dormant in current config: `disaster_brake_pct` is unset in `regime_config_ed.json`. Doc still claims `disaster_brake_pct: 5` for ETH (config drift); user decision pending.
- **`881da46` M-17** — Shield-block Telegram message rate-limited via `_rate_limited_telegram` keyed `shield_block_{asset}` (1h cooldown). One alert per shield event instead of one per cycle.
- **`f758f46` M-06 + M-07** — `sync_positions` and Telegram settings handlers (`/cfg_{ASSET}_auto`, `/gate ASSET clear`) now acquire the per-asset trade lock around their load→modify→save sequences. Eliminates the race window that was producing duplicate `(synced)` trade records and clobbered toggle state.
- **`a441d09` M-19 + M-15** — Hot-reload does per-asset wholesale replace (not per-key merge) so deleted keys propagate; diff lines logged. Main-loop error sleep `time.sleep(120)` → `_stop_event.wait(120)` so /stop is responsive after exceptions.

**Bundle 2 deferred:** M-10 (orphan reconciliation, ~30 lines, requires new API endpoint), M-20 (manifest regen mtime cache, polish).

### Closed 2026-04-25 morning (trader audit bundle 1 — 4 commits, ~70 lines)

Triggered by user observing $9.85 phantom basis on the 2026-04-24 22:01 BUY (recorded $12008.85 vs actual exchange charge $11999). Spawned a thorough audit; surfaced 20 findings; verified each against source. Bundle 1 covers the 6 with money or restart-blocker impact.

**Rollback anchor:** `git tag pre-trader-bundle-20260425` on commit `8766d05` (last known-good before bundle 1). Nuclear revert: `git reset --hard pre-trader-bundle-20260425 && git push --force-with-lease`. Per-commit revert: `git revert <sha>` for any of the four below — they touch different code regions.

**Commits (in chronological order on `main`):**

- **`69177a2` Patches A + B** — `crypto_revolut_ed_v2.py`. Patch A: removed `max_attempts = maker_window // check_interval` recompute inside the partial-fill branch (the SELL slide was jumping backwards on boost — denominator grew, progress fraction shrank, price re-quoted upward). Patch B: added `_config_write_lock = threading.Lock()` around `save_trading_config()` writes — PID-suffixed tmp protected cross-process collisions but two threads in same process (Telegram thread + main loop) shared PID and could half-write `regime_config_ed.json`.

- **`ef542e9` M-01 staleness/horizon refusal blocker** — `crypto_live_trader_ed.py:617-655`. CRITICAL: Fix #1 (label-tail NaN) + Fix #5 (staleness threshold = 2h) silently conflicted. After `df = df_full.dropna(subset=['label'])`, the inference row was `horizon` hours behind `df_full.iloc[-1]`. `lag_hours = horizon`, threshold `> 2` → REFUSE every cycle for any horizon ≥ 3h. Trader still ran only because the in-memory process loaded pre-fix code; on next restart it would have stopped trading entirely. Fixed by decoupling: `df_train = df_full.dropna(['label'])` for training, `df = df_full` (label may be NaN at tail) for inference. lag_hours now ~0.

- **`47c0dae` M-02 + M-03 ledger-delta basis** — `crypto_revolut_ed_v2.py`. Two coupled bugs corrupted recorded basis on every trade: (M-02) `order.get('filled_size')` always returned None — Revolut API returns `filled_quantity`. Trader fell back to `buy_amount/candle_close`, then multiplied by actual `average_fill_price` → inconsistent ratio gave drift up to ~$10 on a $12k trade. (M-03) Multi-leg maker fills (partial → cancel → re-price → next leg fills) only returned the LAST leg's `od`; first leg's filled portion was real on exchange but invisible to position recorder. Both fixed simultaneously by computing basis from balance delta around the order on all 4 trade execution paths (auto BUY, auto SELL, manual /buy, manual /sell). API fields kept as fallback with correct field names. SELL pnl_usd now also computed from `delta_usd_recv - usd_invested` when both ledger deltas are present.

- **`68acd30` M-13 + M-16** — `crypto_revolut_ed_v2.py`. M-13: `save_position` PID-suffixed tmp path (parity with `save_trading_config`); protects against accidental dual-trader-instance collision. M-16: `check_telegram_commands()` returned only the LAST message in each poll batch — earlier messages had their `update_id` consumed (gone forever) but were silently dropped. Concrete failure: user clicks Buy then types /sell during a market drop; both arrive in same 5s poll batch; only one survives. Now returns a list of all pending messages in arrival order; `_telegram_command_loop` iterates and dispatches each through new `_dispatch_telegram_message` helper with per-message try/except.

**Verified empirically:**
- Today's recorded basis $12008.85 = `(buy_amount/price) × actual_fill_price` (the M-02 fallback formula), $9.85 above actual exchange charge $11999.
- Today's recorded PnL of +$6.84 was within $0.05 of actual (~$6.79) by accident — two basis errors cancelled almost exactly through the percentage math.
- The PnL cancellation will NOT hold for trades where bid-mid spread is wider or where multi-leg fills happen. Bundle 1 prevents this going forward.

**8 findings deferred to bundle 2** — see "DEFERRED TRADER AUDIT FIXES" in the TODO section above for M-04, M-06, M-07, M-10, M-15, M-17, M-19, M-20.

### Closed 2026-04-24 evening (7-point safety audit — all critical paths hardened)

Deep audit of live trader + backtest engine flagged 7 silent-failure / logical-bug classes. All fixed the same evening (2-3h of work). Each fix is minimal and tested.

**Audit findings + fixes (severity order):**

1. **Label mislabeling at tail** (CRITICAL) — [crypto_trading_system_ed.py:767-782](crypto_trading_system_ed.py#L767)
   - `future_return.shift(-horizon)` is NaN for the last `horizon` rows. `(NaN > threshold).astype(int)` coerced to 0, silently marking those rows as negative class.
   - Over 1440-row window with horizon=8 → 8 rows poisoned as false SELL examples. Gamma-weighting gives recent rows highest weight = disproportionate training bias toward SELL-at-peak.
   - **Fix**: cast to `.astype(float)` then `.where(future_return.notna(), np.nan)`. Downstream dropna removes the NaN tail cleanly. Verified: built df ends exactly `horizon` hours before raw data end.

2. **Mode T `cd_hours` convergence tolerance too loose** (HIGH) — [crypto_trading_system_ed.py:5396](crypto_trading_system_ed.py#L5396)
   - `TOL_GATE_CD = 6` → cd=10h vs cd=16h considered "converged." But 60% longer cooldown is structurally different behavior.
   - **Fix**: tightened to `TOL_GATE_CD = 2` (matches `TOL_HOLD` semantics). Convergence gate now catches oscillating cooldowns that previously shipped silently.

3. **FEATURE_SET_A silent fallback when features missing** (HIGH) — [crypto_live_trader_ed.py:570](crypto_live_trader_ed.py#L570)
   - If a prod model's `optimal_features` didn't match the current build, trader silently fell back to `FEATURE_SET_A` — a 30-feature default the model was NEVER trained with. Decision boundary broke silently.
   - **Fix**: refuse to trade (`return None`) + critical Telegram alert when ZERO features match. Also added severe-partial threshold: coverage < 50% → refuse with Telegram alert. Verified current V4 prod models all at 100% coverage.

4. **Regime detector 'error' sentinel silently defaulting to bull** (HIGH) — [crypto_live_trader_ed.py:762](crypto_live_trader_ed.py#L762), [crypto_revolut_ed_v2.py:1723](crypto_revolut_ed_v2.py#L1723)
   - Main `process_asset` trader path already handled 'error' (from earlier Fix #2). But the legacy `generate_regime_signal` defaulted to `horizon=6` + empty cfg and traded anyway. The `/status` Telegram handler displayed 'error' as 🔴 bear.
   - **Fix**: both legacy paths now refuse; `/status` displays `⚠️ DETECTOR ERROR (refusing trades)`.

5. **Staleness threshold too loose** (MEDIUM) — [crypto_live_trader_ed.py:649](crypto_live_trader_ed.py#L649)
   - Old check: `if lag_hours > horizon + 2: refuse`. At horizon=8h, allowed 10h of staleness. No wall-clock check at all — missed the case where `df_full` itself was stale.
   - **Fix**: two checks, both with fixed 2h threshold:
     - Internal gap (inference row vs freshest bar in `df_full`) > 2h → refuse + Telegram
     - Wall-clock staleness (freshest bar vs `datetime.now(UTC)`) > 2h → refuse + Telegram
   - Hardened tz-awareness guard so naive vs aware comparisons don't raise. Replaced the `except: pass` wrapper with `except: refuse`.

6. **Non-atomic writes to live config/prod files** (MEDIUM) — multiple sites
   - `with open(path, 'w'): json.dump(...)` creates a truncated file mid-write. Race window (~10-100ms) where live trader's hot-reload could read partial JSON → `JSONDecodeError`.
   - **Fix**: added `_atomic_write_csv()` helper (mirrors existing `_atomic_write_json`). Patched 3 sites:
     - `crypto_trading_system_ed.py:4616` — `crypto_ed_production.csv` write
     - `crypto_trading_system_ed.py:4641` — `regime_config_ed.json` write
     - `crypto_revolut_ed_v2.py:185` — `save_trading_config()` now tempfile + `os.replace`
   - All writes atomic via `os.replace()` (atomic on modern NTFS/POSIX). Readers see OLD or NEW, never half-written.

7. **Silent model-fit exception swallowing** (MEDIUM) — 4 sites
   - `except Exception: continue` in signal-generation model.fit/predict loops hid GPU OOM, scaling errors, all-NaN features. Outer loop showed "no votes" with zero diagnostic.
   - **Fix**: added `_log_fit_exception(context, exc)` helper with session-scoped dedup set — first occurrence of each `(context, ExceptionClass, short_msg)` triple prints once, subsequent repeats stay silent. No log spam, but failure modes surface.
   - Patched: `generate_signals` (ed.py:2324), `_quick_score` (ed.py:2720), `_deku_eval_with_pruning` (ed.py:3699), `generate_live_signal` (live_trader.py:715). Live trader imports helper lazily from main module.

**Audit false positives (flagged by auditor, verified not bugs):**

- **`rs == rs` NaN guard at rally_cooldown simulator**: auditor claimed tautology, but `float('nan') == float('nan')` returns False in Python/NumPy, so `rs == rs` is the standard idiomatic NaN check. Behavior is correct.
- **Signal cache reuse across T↔G iterations**: signals are shield/gate-independent (shield applied AFTER signal generation), so caching across iterations is actually correct.

**Not fixed (intentional):**

- **`except Exception: pass` in Windows priority / orphan worker cleanup**: non-critical setup steps, intentional tolerance.
- **Config cold-start fallbacks to `{}`**: expected behavior when file doesn't exist yet.
- **`os.startfile()` GUI open**: cosmetic, platform-specific.
- **Funding rate load falling back to None**: documented semantic.

**Deployment safety:** all fixes tested with syntax parse + sample invocations. Live trader impact: zero functional change on healthy path. New behavior only kicks in when something is genuinely broken (refuse-to-trade + Telegram alert), which is what we want.

**Re-audit (2026-04-24 late evening) — 2 follow-up bugs from the fixes themselves:**

- **Atomic write tmp-path collision (CRITICAL)** — both `_atomic_write_json` and `_atomic_write_csv` used `path + '.tmp'`. Two concurrent writers (e.g., parallel HRST subprocesses) would collide on the same tmp file; one's content could overwrite the other's before `os.replace`. **Fix**: tmp path now includes PID (`path + f'.{os.getpid()}.tmp'`). Also applied to `crypto_revolut_ed_v2.py:save_trading_config`.

- **Partial data download flag (HIGH)** — `_DATA_DOWNLOADED_THIS_SESSION = True` was set after the download block regardless of whether macro and OHLCV actually succeeded. If one failed silently, subsequent horizons would skip the retry and run on partial/stale data. **Fix**: track `macro_ok` + `ohlcv_ok` flags; only cache-flag when BOTH succeed. Partial success prints a diagnostic and next horizon retries.

Re-audit also verified: all 7 original fixes clean, no regressions, label float/NaN cascade correct, Mode T max_iter still honored under tighter tolerance, feature-refuse paths cleanly return None, regime error propagation complete, staleness check handles all tz permutations, model-fit logging thread-safe under GIL.

### Closed 2026-04-24 (sparse-feature quarantine + dropna warning + AB matrix relaunched)

**Discovery:** all prior HRST runs (V1, V4, first AB matrix) trained on only **672 clean rows** out of a 1440-row (60d) window because `deriv_oi_*` (3 cols, ~50% NaN) and `ob_imbalance, spread_bps, avg_iv, iv_skew` (4 cols, ~93% NaN) had short history — dropna cascaded and wiped half the window. Models were effectively trained on the most recent 28 days, not 60. V1 and V4 promotion decisions were made on biased data.

**Fix — two-section `config/disabled_features.json`:**
- `disabled_exact` (65 entries): Mode F Grade-1 features — toggleable via `enabled` flag and `--trim-override`
- `always_disabled_exact` (7 entries): structurally-broken features (short history) — **applied REGARDLESS of enabled flag**
  - `deriv_oi_chg1d, deriv_oi_chg3d, deriv_oi_zscore` (Binance OI history started 2026-03-20)
  - `ob_imbalance, spread_bps` (live trader started writing 2026-04-19)
  - `avg_iv, iv_skew` (live trader Deribit feed started 2026-04-19)
- `_load_disabled_features()` returns `(exact, prefixes, enabled, always)` — 4-tuple now
- `_apply_feature_disable()` always strips `always_disabled_exact`; strips `disabled_exact+prefixes` only when `enabled=True`
- Data pipelines untouched — these features keep being downloaded, just excluded from LGBM/PySR inputs
- Result: clean rows went from 672 → **1432/1440 (99.4%)** in Mode D window

**Fix — DATA LOSS WARNING in Mode D ([crypto_trading_system_ed.py:3793](crypto_trading_system_ed.py#L3793)):**
When clean rows < 80% of window after dropna, prints bang-boxed warning with:
- Row count + % of window retained
- Top 8 NaN offenders by raw count
- Suggestion to add them to `disabled_features.json`
- Explicit note that model results are biased toward whatever regime spans the surviving rows
- Fires per horizon (4× per HRST if the issue persists)

**Live trader impact:** zero. `disabled_features.json` structure extended but backward-compatible. Live trader's hot-reload now sees 72 total disabled (65 Mode F + 7 always). Effective pool unchanged from yesterday. V4 prod models don't reference any of the 7 always-disabled features — verified via `optimal_features` scan.

**Matrix relaunched 2026-04-24 07:40 CEST** — `python tools/ab_matrix_runner.py --variants focus --seed 2026`:
- 3 variants: A (floorON_trimOFF), B (floorON_trimON), C (floorOFF_trimOFF)
- Optuna seed 2026 (vs default 42) — tests whether effects are seed-robust
- Now running on 1432/1440 clean rows per variant instead of 672
- ETA ~14:00-17:00 2026-04-24

**Known TODO scheduled re-enable tests:** 2026-05-22 for deriv_oi_*; 2026-06-18 for orderbook/IV. See TODO G1/G2. **NOT auto-enabled** — user runs A/B test first and decides.

### Closed 2026-04-22 late night (V1 promoted + rally-cd bug fix + matrix launched)

- **Variant #1 of AB matrix promoted to production at 23:24 CEST.** User sold 2026-04-21 trade manually at ~$2,388 (+3.2% realized, ~33h hold) and immediately promoted V1 without waiting for variants #2-#5. Promoted config:
  - Detector: **tsmom_672h** (different from both the earlier prod `price>sma72` and the 4mo backup `tsmom_672h` bull=6h/bear=7h)
  - Bull: **5h @ 85% conf, shield=ON**, gate `rr12≥2.5% OR rr18≥4.0% cd=16h`
  - Bear: **6h @ 80% conf, shield=OFF**, gate `rr16≥7.0% OR rr30≥4.5% cd=14h`
  - `min_sell_pnl=0.30%, max_hold=12h`
  - Mode T iteration 3 converged; per-regime decomposition: bull +46.23%, bear +70.22%
  - 60d sim total: **+122.59%** vs B&H +20.95% = **+101.64pp alpha** (strongest ETH result of the week)
  - Passes all 4 promotion gates (converged, top-15 plateau unanimous, alpha > baseline, will validate on 4mo in R5)
  - Backup of prior (laptop 2mo+trim) saved to `config/regime_config_ed_prev_prod_20260422.json` + `models/crypto_ed_production_prev_prod_20260422.csv`
  - Per-horizon feature quality (floor ON worked as designed):
    - 5h: 14 features, 2 logret, 1 pysr (return +80.92%, acc 73.5%)
    - 6h: 8 features, 2 logret, 1 pysr (return +76.87%, acc 62.3%)
    - 7h: 30 features, 4 logret, 5 pysr (return +25.35%)
    - 8h: 10 features, 1 logret, 4 pysr (return +24.19%)
  - **Caveat:** remaining matrix variants (#2-#5) could still beat V1 and warrant another promotion cycle tomorrow. Watch the full matrix CSV when it finishes ~14-15h 2026-04-23.

- **Rally-cooldown formula fixed (`crypto_revolut_ed_v2.py:929`).** Previous formula `implied_until = now - bars_ago + cd_h` used wall-clock time instead of actual bar timestamps. Result: cooldown end drifted by `(now - last_closed_bar_time)`, typically 1-59 min but up to 2h+ when data was stale. Symptom after V1 promotion: catch-up scan correctly detected the 14:00 UTC trigger bar (rr18=+5.16%≥4.0%) but computed cooldown_until = 08:24 UTC tomorrow instead of expected 06:00 UTC. Fix: use `trigger_time = df_raw['datetime'][end_idx]` (tz-localized UTC) as the anchor so `implied_until = bar_open_time + cd_h`. Deterministic regardless of wall clock. Cleared stale `rally_cooldown_until` in `config/position_ed_v2_ETH.json` before restart. After fix: cooldown_until = 2026-04-23T06:00:00Z (14:00 UTC + 16h).

- **AB matrix orchestrator launched on desktop at 17:32 CEST** — `tools/ab_matrix_runner.py`. Full factorial: 5 HRST variants (trimOFF/ON × metaOFF/ON × floorON, plus trimON+metaON+floorOFF tiebreaker) + vol-scaled horizon test. Variant #1 finished in 4h 2min (longer than predicted 2.5h). Variants #2-#5 + vol test still running overnight. Audit CSV: `output/ab_matrix_results_<timestamp>.csv` with per-horizon features, shields, gates, Mode T REF returns, meta stats.

- **New CLI flags shipped**:
  - `--no-feature-floor` disables the feature-family floor for A/B comparison
  - `--no-data-update` skips macro + OHLCV downloads at HRST start (used by matrix runner so all variants see identical data snapshot)
  - `--meta-filter P` (from earlier today) walk-forward secondary LGBM; BUYs with meta_prob < P become HOLD

### Closed 2026-04-21 late evening (meta scaffold + consistency tool + 4mo Desktop HRST)

- **Desktop 4-month ETH HRST complete** (log `ed_v1_20260421_171708.log`, ~17:17 → ~20:15, ran `--no-persist` → wrote to `_noprod.*`). Winner: **`tsmom_672h` bull=6h@90% / bear=7h@95%, shield bull=OFF/bear=ON, min_sell_pnl=0.40%, max_hold=8h**, converged Mode T iter 3, bull gate `rr12h≥2.0%/rr30h≥6.0% cd=24h`, bear gate `rr24h≥6.5%/rr30h≥5.5% cd=20h`. 60d sim return +133.64% / alpha +155.91pp / 132 trades / 69% WR. Plateau robust: top-15 Mode S all agree on bull=6h/bear=7h with detector split tsmom (12) vs price>sma72 (3). **Zero parameter overlap with 2-month Desktop winner** — every single parameter differs. Regime-context analysis: 4-month window includes a bear rally phase that isn't representative of today's market, so 2-month is the more applicable configuration despite lower alpha. **Backed up to `config/regime_config_ed_4mo_backup.json` + `models/crypto_ed_production_4mo_backup.csv`** before laptop's 2mo-trim run overwrites `_noprod.*`.

- **Meta-labeling scaffold shipped (`crypto_trading_system_meta.py`).** Standalone research tool, no prod impact. Training pipeline: reads primary model config from `crypto_ed_production.csv`, generates primary signals via `generate_signals()`, builds meta-labels (label=1 iff forward_return(horizon) > 2×fee), walk-forward LGBM training with embargo, evaluates at multiple probability thresholds. Implements López de Prado Ch. 3 meta-labeling. CLI: `python crypto_trading_system_meta.py <asset> <horizon> [--replay N] [--p-thresholds A,B,C] [--sizing]`. Runtime ~15-20 min on 1440h replay (primary signal gen dominates). Output: per-trade CSV at `output/meta_<asset>_<h>h_<timestamp>.csv`. Run not yet executed — queued for Desktop.

- **PySR/HRST consistency tool (`tools/check_pysr_consistency.py`).** 5 checks: (1) PySR JSONs present + metadata; (2) Ordering — JSONs pre-date latest HRST; (3) Production CSV rows' `optimal_features` reference pysr_*; (4) Leakage guard passed in latest log; (5) Functional — every formula parses, every referenced feature exists in current build. CLI: `python tools/check_pysr_consistency.py [--asset ETH] [--horizons 5,6,7,8]`. Exit code 0 = clean, non-zero = at least one [FAIL]. Ran on 2026-04-21 20:30 → **all 5 checks pass for ETH 5/6/7/8h**. Use as pre-flight before promoting any HRST result. Runtime ~5 seconds.

- **ETH HRS winner (overnight, log `ed_v1_20260420_232302.log`, 23:23 → early 2026-04-21)** — superseded by Desktop 12:07 HRST. Noted as stale because it ran before the morning Mode P finished, so used yesterday's PySR JSONs. Weak winner (+11.34pp alpha) attributable to pre-regen PySR, not bad data.

### Closed 2026-04-21 (Mode F shipped + overnight HRS chain + Desktop ETH HRST)

- **Overnight HRS chain (ETH/BTC/SOL/LINK, laptop)** — started 2026-04-20 23:23, SOL+LINK still running at 18:00. Results so far:
  - ETH HRS 23:23 (before morning P finished): `sma168>sma480 6h@90% / 8h@90%` → +28.50%, +11.34pp alpha, 58 trades, 64% WR. **Lower quality than yesterday** because it used stale PySR JSONs.
  - BTC HRS 04:18 → 12:30: `sma24>sma100 bull=8h@85% / bear=7h@65%` → **+36.15%, +23.89pp alpha, 84 trades, 70% WR**. Top-15 plateau unanimous on horizon pair. **+21.26pp alpha improvement vs 2026-04-19 HRST** (that one ran before the 2026-04-19 18:00 feature-additions commit, so it lacked derivatives/stablecoin/orderbook/IV).
  - SOL + LINK HRS pending.

- **Desktop ETH Mode P (03:00-07:59)** — regenerated all 4 PySR JSONs (5h/6h/7h/8h) against current feature set (incl. perp-spot basis + BTC lead-lag added 2026-04-20). Opens door for clean HRST with fresh PySR.

- **Desktop ETH HRST (12:07 → ~15:44, ~3h 37min)** using fresh morning PySR + new features. **Strongest ETH result of the week:**
  - Mode S winner: `sma168>sma480 bull=7h@70% / bear=5h@75%` → **+68.17%, +49.41pp alpha, 115 trades, 70% WR**.
  - Top-15 unanimous on detector, 12/15 bull=7h, very tight plateau.
  - Mode T final (iter 4, reached max_iter without convergence): `min_sell_pnl=0.40% max_hold=10h bull_shield=ON bear_shield=OFF`, total **+98.33%** (baseline +87.19% all-OFF shields).
  - Bull gate: `rr18h≥3.5% OR rr36h≥4.5%, cd=30h` — Mode T re-enabled bull gate (different structure from harmful 8h/12h one; plateau=1.00 across 21,812 STRICT configs). Option C effectively undone but the structurally different gate may be legitimate.
  - Bear gate: `rr14h≥3.5% OR rr24h≥5.5%, cd=24h` (plateau=0.70, borderline).
  - Per-regime contribution decomposition: bear delivers +59.27% on 37% of bars vs bull +27.92% on 63% of bars — bear gate still the main gate-gain driver.
  - **Non-convergence flag + run-to-run variance + Mode T bull-gate swapping from disabled→enabled within 24h = 4-month `--replay 2880` recommended before enabling live.**

- **Mode F shipped (Feature Trim).** Single-letter mode in `crypto_trading_system_ed.py` + button `🧹 Feature Trim (F)` in optimizer bot main menu. CLI entry points:
  - `python crypto_trading_system_ed.py F` — audit prod CSV + PySR formulas, populate `config/disabled_features.json` with Grade 1 (zero selection + zero PySR ref), spare newborns, write.
  - `F --restore` — empty the disabled list.
  - `F --include-newborns` — aggressive; disable newborns too.
  - `F BTC` — use BTC's feature universe (adds SOPR variants to the dead list).
  - Bot flow short-circuits to confirm: Mode F is universal (scans all prod models), no asset/horizon pickers needed.
  - Initial run disabled **65 features** (BTC universe) or **62 features** (ETH universe): 44 on-chain, 13 macro, 5 technical, 3 sentiment, 2 cross-asset. Universe shrinks BTC 193→128, ETH 191→129.
  - **22 newborn features intentionally spared** (`NEWBORN_FEATURES` set in `crypto_trading_system_ed.py`): derivatives-as-feature, stablecoin, orderbook, IV, basis, BTC/ETH intraday lead-lag. Maintainer note: prune from this set after 2-3 HRST cycles confirm newborns as dead.

- **Feature audit tool (`tools/audit_features.py`)** — grades every feature 1-5 (5=≥60% selection, 1=0 selection + 0 PySR refs). Outputs per-category breakdown, orphans list (features used by prod models but missing from current build — these silently fall back to FEATURE_SET_A), and CSV export. Grade-resurrection rule: any feature appearing in ANY PySR formula across any asset auto-bumps from Grade 1 → Grade 2, protecting it from disable.

- **Feature disable mechanism (`config/disabled_features.json` + `_apply_feature_disable()`)** — loaded at end of `build_all_features()` via module-level mtime-cached reader. Strips names from `all_cols` (the LGBM input list) but keeps columns in df (so PySR formulas can still evaluate against disabled features). Verified: 0 pysr_* output features in disabled list, 0 PySR input features in disabled list (by construction via Grade resurrection).

- **`--no-persist` CLI flag (+ Telegram toggle)** — `python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --no-persist` seeds `PRODUCTION_CSV` and `REGIME_CONFIG_PATH` to `*_noprod.*` copies at startup; all writes go there. Safe to run alongside live trader. In optimizer bot: confirm screen now has `🧪 Switch to NO-PROD` toggle; job label tags as `[NO-PROD]`.

- **Optimizer bot main menu: HRS → HRST.** Main 🔧 Full Re-tune button now launches HRST (chains Mode T for shield + gate sweep). HRS still available under Advanced ▸.

- **Rally-cooldown gate verdict (completed 2026-04-20 late evening, fleshed out 2026-04-21):**
  - **Bull rally-gate disabled.** Confirmed harmful across all three test windows: live 30d −7.48pp, OOS 60d −1.31pp, FULL 90d −3.32pp. Previous structure (rr8h≥2.5% OR rr12h≥2.0%, cd=8h) was parameterization artifact (12h branch dominated). Disabled in config 2026-04-20.
  - **Bear rally-gate kept.** Consistently helpful: OOS 60d +4.11pp, FULL 90d +5.59pp. Params `rr10h≥4.0% OR rr16h≥5.5%, cd=36h`.
  - **Drop-gate sweep (alternative design) rejected.** Tested on 90d simulated signals: 30d-live winner ranks 101/126 on OOS 60d, delta −13.50pp. Top-10 OOS ∩ top-10 IS = 0 rules in common. Textbook overfitting. Tool: `tools/backtest_drop_gate.py`.
  - **Deflated Sharpe audit** (`tools/deflated_sharpe.py`, Bailey-López de Prado 2014): live 30d DSR = 0.000 after correcting for 3920-config Mode S sweep. Live return indistinguishable from lucky-config draw. Theoretical Mode S winner at SR=2.5 passes DSR cleanly — skill exists on simulated signals; live execution is the leak.

- **Per-regime Telegram `🚦 Gate` buttons (trader)** — main keyboard now has row 4: `🚦 Gate Bull: ON/OFF` / `🚦 Gate Bear: ON/OFF`. Tap sends `/gate ETH <regime> on|off`. Also fixed silent schema bug: previously `/gate ETH off` only set asset-level `rally_cooldown.enabled` but left per-regime blocks untouched. Now `_set_gate_enabled()` helper propagates to both levels.

- **`tools/audit_features.py --asset X` + per-asset usage matrix** — saved `models/feature_usage_by_asset.csv` showing feature selection count per (feature × asset). Findings: 9 universal features picked by all 5 assets (price_to_sma100h, logret_120h, hour_cos, sma20_to_sma50h, logret_240h, xa_dax_relstr5d, m_sp500_chg1d, xa_sp500_relstr5d, adx_14h). `pysr_1..5` each appear in every asset. Only 1 exclusive feature per asset (XRP: `m_vix_chg5d`). LINK uses more exclusive macro features than others, consistent with its weak model diagnosis.

### Closed 2026-04-20 evening (literature review + infrastructure sprint)

**Triggered by full 2024-2025 crypto algo-trading literature review (12 areas covered).**
Cross-referenced against tested/rejected items in this doc. Actioned all untested items
that were "lit-endorsed" and feasible tonight.

- ~~**Deflated Sharpe Ratio audit (Bailey & López de Prado 2014)**~~ — `tools/deflated_sharpe.py` shipped. Computes Deflated SR + PSR + E[SR_max] under null given N-trial multi-testing. **First audit result: live ETH 30d DSR = 0.000** (23 trades, per-trade SR=0.057, after correcting for N=3920 Mode S sweep). Live return indistinguishable from a lucky-config draw. Theoretical Mode S winner (simulated SR≈2.5) passes DSR cleanly — skill exists on simulated signals; live execution is the leak.
- ~~**Perp-spot basis feature (Test 2)**~~ — Historical perp hourly klines now downloaded via Binance `/fapi/v1/klines`, stored as `perp_close` in `derivatives_<asset>.csv` alongside funding + OI. `build_all_features()` computes 3 new features: `deriv_basis = (perp-spot)/spot`, `deriv_basis_chg1d` (24h change), `deriv_basis_zscore` (168h rolling). Feature verified on ETH (191 features total vs 185 previously). All 5 assets have perp klines downloaded (BTC/ETH/XRP/SOL/LINK).
- ~~**Cross-asset BTC→ETH intraday lead-lag features (Test 5)**~~ — Literature-backed (BTC leads ETH 5-30min; ETH leads alts). At 1h resolution we add 3 lagged BTC return features (`xa_btc_lag1h/2h/3h`) for ETH/XRP/SOL/LINK, plus ETH lags for XRP/SOL/LINK. Matches the existing daily `xa_*_relstr5d` pattern but at intraday granularity.
- ~~**XRP full pipeline enablement**~~ — Previously XRP couldn't run Mode P (missing 8640 rows after sparse-feature dropna). Fix trio: (1) added `sparse_prefixes = ('gp_', 'deriv_oi_', 'ob_', 'avg_iv', 'iv_skew', 'stable_mcap_', 'whale_')` to `pysr_discover_features.py` NaN→0 fill block (mirrors main `ed.py`); (2) extended `onchain_asset_map` to include `'XRP': 'xrp'`; (3) extended derivatives download loop in `download_macro_data.py` to `['BTC', 'ETH', 'XRP', 'SOL', 'LINK']`. Post-fix: XRP data pulled (hourly +610 candles, derivatives 37697 rows, on-chain 1571 days with 4 metrics — active_addresses, mvrv, fees_native, tx_count). **XRP HRST completed**: winner `sma168>sma480` 5h@85% / 7h@90%, +19.96%, +14.70pp alpha on 60d. H1 was flat / H2 carried the alpha — concerning. Bull gate persisted from iter 1 tiebreak but 0 STRICT winners in iter 2/3 (spurious).
- ~~**SOL partial pipeline refresh**~~ — Hourly data refreshed (+17 candles), derivatives downloaded (37697 rows + perp klines). On-chain BLOCKED: CoinMetrics community API returns HTTP 403 for SOL (free tier excludes). Yesterday's SOL HRST was run on stale feature set (177 features, pre-2026-04-19 18:00 commit). Re-run queued for tonight's overnight chain.
- ~~**LINK full pipeline enablement**~~ — Hourly data refreshed (+19 candles), derivatives (37697 rows), on-chain (1570 days with 3 metrics — active_addresses, mvrv, tx_count). Yesterday's LINK HRST was stale. Re-run queued for tonight.
- ~~**Rally-cooldown gate empirical validation — bull OFF, bear KEPT ON**~~ — `tools/backtest_drop_gate.py` shipped (90d OOS test on cached signals). **Bull gate (rr8h≥2.5% OR rr12h≥2.0%, cd=8h) confirmed harmful**: −7.48pp on live 30d, −1.31pp OOS 60d, −3.32pp FULL 90d. **Disabled** via `ETH.bull.rally_cooldown.enabled: false` + cleared active cooldown timer. **Bear gate (rr10h≥4.0% OR rr16h≥5.5%, cd=36h) confirmed helpful**: +4.11pp OOS 60d, +5.59pp FULL 90d. Kept ON. **Drop-gate sweep overfits**: 30d-live winner (−2%/9h/24h) ranks 101/126 on OOS 60d (−13.50pp). Verdict: don't ship static gates without multi-window OOS validation.
- ~~**Live-vs-simulated gap discovered**~~ — Simulated 30d IS baseline (no gate, current policy): **+27.11%** vs actual live 30d: **+2.54%**. Gap **~24.5pp**. Decomposition: ~7.5pp bull rally-gate drag (now eliminated), ~17pp unaccounted execution friction (slippage, partial fills, manual interventions, clock drift, possibly other). Biggest lever for tomorrow isn't more features — it's closing the execution gap.
- ~~**Per-regime rally-cooldown Telegram UX**~~ — Added 4th button row: `🚦 Gate Bull: ON/OFF` + `🚦 Gate Bear: ON/OFF` (mirrors shield's per-regime pattern). New CLI subcommands: `/gate ETH bull on|off` + `/gate ETH bear on|off`. `/gate` status view now shows per-regime state + thresholds. **Fixed silent schema bug**: previously `/gate ETH off` only set asset-level `rally_cooldown.enabled` but left per-regime blocks untouched — trader was still reading `bull.rally_cooldown.enabled: true`. Now `_set_gate_enabled()` propagates to both levels. Helper `_gate_on_for_regime(cfg_asset, regime)` added.
- ~~**`--label-threshold` CLI flag on ed.py**~~ — Module-level `LABEL_THRESHOLD_PCT` override at `crypto_trading_system_ed.py:312`. When set via `--label-threshold 0.01`, swaps label from `fee_aware` (ret > 2×fee) to `ret > X`. Output auto-redirected to `models/crypto_ed_production_lt<pct>.csv` so prod untouched. Shipped for Desktop test A (large-upside label). `--help` updated.
- ~~**PySR sparse-feature fix for XRP/SOL/LINK**~~ — Extended GDELT-only NaN→0 block in `pysr_discover_features.py` to cover all sparse prefixes. Without this, stablecoin (1yr) + OI (30d) features killed the dropna down to 7970 / 984 rows. Post-fix XRP had full 12-month window.
- ~~**`extend_caches_90d.py` detector fix**~~ — Was hardcoded to `tsmom_672h`; now reads `regime_detector.params.name` from config. Current ETH config uses `sma168>sma480`.

### Closed today (2026-04-19)

- ~~**Asset universe prune**~~ — dropped DOGE / ADA / AVAX / DOT (4 assets). Config, ASSETS dicts in 4 Python files, optimizer bot list, and 15 stale production-CSV rows all cleaned. Kept 5: ETH (active), BTC / SOL / LINK / XRP (standby/testing).
- ~~**Strip "Doohan" labels from Ed code**~~ — removed obsolete "Doohan" references from user-facing labels / docstrings / prints in `crypto_trading_system_ed.py` (module header, "ED OPTIMIZATION" print, Mode D docstring), `crypto_live_trader_ed.py` ([DOOHAN] → [ED], error messages), `crypto_trading_system_ein.py` / `_eli.py` / `_ed_v3.py` (CSV output filenames: `diagnostic_results_doohan_*` → `diagnostic_results_{ed|ein|eli|ed_v3}_*`). Internal variable names (`DOOHAN_GAMMA_MIN` etc.) and dict keys (`'source': 'doohan'`) left — implementation details, not labels; would require broader refactor.
- ~~**SOPR NaN fix (misdiagnosis)**~~ — Re-audit of BTC HRST log showed SOPR is NOT filtered. The two "Columns:" log lines I initially read were for DIFFERENT files (BTC `onchain_btc.csv` with SOPR, ETH `onchain_eth.csv` without). SOPR is loaded and reaches LGBM importance ranking; just ranks below 1% importance. No fix needed.
- ~~**Fee consistency audit**~~ — Grep verified: all `0.0011`/`0.0022` hardcoded fees exist only in `archive/`. Active files use `BACKTEST_FEE_PER_LEG = 0.0005` consistently. `TRADING_FEE_BASE = 0.0009` remains only as pure-taker documentation / label semantic.
- ~~**Telegram optimizer bot — menu buttons**~~ — Added `G - Rally-cd (cache)` to advanced menu. Relabeled `T - Threshold+G` and `HRST - Full+T+G` to make the chaining explicit.
- ~~**Signal nondeterminism audit**~~ — All models, Optuna, numpy use `seed=42`. Two likely sources identified: (a) LGBM `device='gpu'` (CUDA floating-point reduction not reproducible); (b) `joblib.Parallel(n_jobs=...)` at 4 sites (task completion order nondeterministic). Repro protocol documented: pin commit, freeze data, set LGBM `device='cpu'` + `n_jobs=1`. Not a code change; 5-10pp run-to-run variance accepted as speedup tradeoff. Run HRS 2-3× when a winner looks marginal.
- ~~**ETH on-chain feature audit**~~ — MVRV Grade 4, exchange_netflow Grade 3, 38 others weak (<1% LGBM importance, not crowded-out)
- ~~**Iterative T ↔ G convergence**~~ — `--max-iter 4` shipped; ETH/BTC both converge in 2-3 iterations
- ~~**Per-regime gate (code + ETH config)**~~ — bull ungated (no plateau winner), bear `rr8h≥2.5% OR rr30h≥5.0% cd=18h`
- ~~**Bear horizon test (ETH)**~~ — 8h best on 60d by +10pp+ over 5/6/7h
- ~~**Disaster brake (code + ETH config)**~~ — `disaster_brake_pct: 5`, fires 0 times in 60d (dormant insurance)
- ~~**BTC HRST**~~ — completed; winner `vol_calm` detector + bull=8h/bear=5h + gate rr20h/rr24h cd=48h
- ~~**BTC enablement decision**~~ — KEEP DISABLED (ETH makes ~3× BTC's return per $; correlation too high for diversification; pipeline half-failed with 6h/7h "no valid trials"; SOPR loaded but below 1% LGBM importance)
- ~~**Shield quick-release evaluation**~~ — 0 fires on 60d at 95%/4h, kept in config as armed insurance; defaults flipped to opt-in only
- ~~**Mode G cleanup**~~ — removed HRSTG/HRSG/DVRSG/RSG; added cache-freshness warning
- ~~**Shield UX rework**~~ — per-regime `🛡 Bull:` `🛡 Bear:` buttons replace cramped `Shield B/B`
- ~~**`BACKTEST_FEE_PER_LEG = 0.0005` refactor**~~ — single source of truth for sim fees across 24 active files

### Completed (2026-04-19)

- **BTC HRST complete** — `HRST BTC 5,6,7,8h --replay 1440` ran successfully after last night's Mode P. Winner: detector `vol_calm` (Andersen-Bollerslev deseasonalized vol), bull=8h@95% shield=ON, bear=5h@80% shield=ON (both shields ON — asymmetric from ETH), min_sell_pnl=0.35%, max_hold=12h, gate rr20h≥3.0% OR rr24h≥4.5% cd=48h (conservative, long cooldown). BTC still `enabled: false` pending decision. BTC bull=8h vs bear=5h is INVERSE of ETH (6h bull / 8h bear).
- **`BACKTEST_FEE_PER_LEG = 0.0005` constant shipped** — single source of truth for sim fees across 24 active files. Live trader is maker-first (~95% maker/~5% taker blend ≈ 1.6 bps/leg measured; 5 bps adds 3× safety margin). Replaced Mode G's hardcoded `FEE = 0.0011` and Mode T's implicit `0% fee`. Both were inconsistent — Mode G's 11 bps made gates look free (fee-drag credit) while Mode T's 0% made them look like winners on the wrong baseline. At realistic 5 bps, single gate was a loser; per-regime gates recover. Standalone backtest scripts (backtest_rally_cooldown*, backtest_sl_variants, compare_*, audit_v6*, etc.) all updated to 0.0005. Scripts that imported TRADING_FEE from ed.py now alias BACKTEST_FEE_PER_LEG for the fee. Label generation (`2 * TRADING_FEE`) untouched at 0.22% break-even — labels represent pessimistic-fee training targets, not sim cost.
- **Per-regime rally-cooldown gate (code shipped + bear config active)** — Schema: `bull.rally_cooldown` / `bear.rally_cooldown` per regime, asset-level `rally_cooldown` kept as legacy fallback. Trader helper `_rally_cfg_for_regime(trading_cfg, regime_label)` in `crypto_revolut_ed_v2.py` reads regime-scoped with fallback. Mode T/G's `_sweep_rally_cooldown` accepts `regime_filter='all'|'bull'|'bear'` — when filtered, gate only fires on that regime's bars AND writes to regime block. Mode T's chain runs bull+bear sweeps independently. **2026-04-19 ETH sweep result**: bear gate = rr8h≥2.5% OR rr30h≥5.0% cd=18h (written). **Bull gate left ABSENT** — 445 configs beat baseline but 0 passed plateau filter (robust region missing in current uptrend regime). Not a bug; rally-cooldown is mean-reversion logic that doesn't help in trend regimes. Standalone test proved +18pp gain from per-regime vs single on 60d maker sim. Unit tests: `test_per_regime_gate_trader.py` 10/10 pass.
- **Iterative T ↔ G convergence** — Mode T wraps the shield + gate sweep in a `max_iter=4` loop. Each pass: snapshot config, sweep shield with current gate applied (via `_sim_horizon(rally_cfg=...)` — new param), pick shield, sweep gate with new shield, pick gate, write. Check fingerprint vs prior pass; break on match. ETH and BTC both converged in 2-3 iterations — proves the coupling is tight but not drift-prone on current data. New CLI flag `--max-iter N` (default 4; pass 1 for single-pass legacy behavior).
- **Mode T chains per-regime G — fee fix flipped the verdict** — Before fee refactor, Mode G's 0.11% taker fee made the rally gate look like a +16pp winner vs no-gate; Mode T's 0% fee made per-regime look like +18pp. After refactor to 5 bps/leg: per-regime gate still wins vs single on 60d; bull's best configs fail plateau robustness filter (don't ship); bear gate refined to `rr8h≥2.5%/rr30h≥5.0% cd=18h` (was 16h) reflecting the slightly higher fee penalty on over-blocking.
- **Disaster brake** (TODO 6) — `disaster_brake_pct` config key (ETH=5% currently). Trader force-sells on unrealized PnL ≤ −brake_pct, bypassing shield. `test_disaster_brake.py` — 13/13 pass. Fires 0 times on 60d backtest (max historical DD was −2.59% << 5% threshold). Dormant insurance against rare catastrophic moves. Originally suggested 7% (validated in 2026-04-14 backtest); user set 5% — slightly more aggressive but still safely above historical worst-case.
- **Shield quick-release (evaluated, removed)** — Added early morning as response to today's painful trade; implemented in trader + Mode T/G + 12/12 unit tests. Then tested empirically: on 60d maker-sim, 95%/4h and 95%/3h both fired **0 times** (identical to OFF). 90%/5h fired once and cost −0.36pp. **Verdict: 95%/4h is theater**, not insurance. Removed from config; defaults in trader flipped to `enabled=False` so missing config = off (not silently on). Code kept for future opt-in via config. User kept 95%/4h back in config as "armed insurance" for events rare enough to not appear in 60d.
- **Shield UX** — Main button split from `🛡 Shield B/B: ON/OFF` into two independent toggles: `🛡 Bull: ON` / `🛡 Bear: OFF` on row 3 of 4; Setup alone on row 4. Tap sends `/hold bull` or `/hold bear`. `/hold` bare shows state + threshold + hint. Unit tests `test_shield_ux.py` 13/13 pass.
- **Mode G cleanup** — Dropped `HRSTG`, `HRSG`, `DVRSG`, `RSG` from `VALID_MODES` (redundant since T chains G). Added cache-freshness warning to standalone Mode G. Purged stale `output/mode_g_*.csv`, kept new `output/rally_cd_*.csv`.
- **Telegram optimizer bot** — Added `G` / `HRST` / `DVRS` to MODES dict + MODE_TIME_EST + REPLAY_MODES. Labels updated (T="Threshold + chain G", HRST="Full + Threshold (incl. G)").
- **On-chain feature audit (ETH)** — 2026-04-18 HRS was the first run after 2026-04-17 wiring. Audit: `oc_mvrv_chg1d` Grade 4 (66.7%, 4/6 ETH rows), `oc_exchange_netflow_chg5d` Grade 3 (1/6, 8h only). Other 38 derivatives below threshold. Reference Card updated.
- **Production report script** — `report_production.py` generates comprehensive 30d + 60d stats for the current asset config: strategy return vs B&H, alpha, trade count, win rate, avg win/loss, expectancy/trade, profit factor, Sharpe-like, best/worst trade, max drawdown, time in market, brake fires. Current ETH (60d, 5 bps fee): +61.56% strategy vs +19.42% B&H = +42.14pp alpha, 75% WR, 8.7 profit factor, −5.29% MDD.
- **Shield quick-release added to trader + Mode T + Mode G sim** — New config block `shield_quick_release: {enabled, min_sell_conf, max_hours}` (defaults true/95/3h). When shield is ON and model flips SELL at ≥min_sell_conf within max_hours of entry, bypass shield. Standalone `test_shield_variants.py` + `test_window_sweep.py` run on 60d cache. Verdict: default 95%/3h fires 0 times in 60d backtest (conservative), does not impact winner. Today's failure mode (entry 21:55, first 97% SELL at 02:00 = 4h) wouldn't trigger with defaults — 90%/5h would have caught it.
- **Shield variant comparison (standalone)** — `test_shield_variants.py` compared 8 variants (no shield, current, QR variants, persistence variants) on 60d cache. All persistence rules (E/F/G/H) LOST 3-5pp vs current shield. QR variants tied or lost marginally. Conclusion: current shield is optimal on 60d historical; today's loss was shield's "insurance premium" for a systematic edge. **Today's failure is NOT in the cache** (cache ends 2026-04-18 12:00, BUY was 21:55) — backtest couldn't evaluate fixes against the specific event.
- **Mode G cleanup (removed chain-mode cruft)** — Dropped `HRSTG`, `HRSG`, `DVRSG`, `RSG` from `VALID_MODES` (redundant since T chains G internally). Help text updated. Interactive menu prompt updated. Cache-freshness guard added to standalone Mode G: prints warning + skips asset if `crypto_ed_production.csv` is newer than `eth_sl_signals_*.pkl`. Purged 5 stale `output/mode_g_*.csv` files, kept 3 new-format `output/rally_cd_*.csv`.

### Completed (2026-04-18)

- **PySR drift detection + full refresh (HRSTG chain)** — Discovered ETH 6h/8h production rows (trained 2026-04-10) used OLD pre-Apr-11 PySR formulas, but `_compute_pysr_features()` reads the CURRENT JSON at inference → silent feature drift on both LIVE horizons. Fix: `P ETH 5,7h` to deepen the stale ones with current PySR code (commits `0a3ba33`/`df40043`/`cdfca63` from 2026-04-11: maxsize 15→25, iterations 40→100, multi-run + island isolation), then full `HRS ETH 5,6,7,8h --replay 1440` to retrain against current PySR. All 4 horizons now internally consistent. Post-HRS winners: 5h (+57.86%), 6h (+63.16%), 7h (+6.80%, weak but not live), 8h (+21.32%). Mode S picked detector=tsmom_672h, bull=6h@95%, bear=8h@80%.
- **Per-regime hold-shield (new capability)** — Shield ON/OFF now splits by regime. Schema: `bull.hold_shield` / `bear.hold_shield` (per-regime), shared `min_sell_pnl_pct` / `max_hold_hours` at asset level. Trader `crypto_revolut_ed_v2.py` reads via new `_shield_on_for_regime(trading_cfg, regime_label)` helper with legacy fallback to asset-level `hold_shield`. `/hold` Telegram command extended: `/hold`, `/hold on|off`, `/hold bull on|off`, `/hold bear on|off`. Button label shows `🛡 Shield B/B: ON/OFF`. Unit tests in `test_per_regime_shield.py` (12/12 pass).
- **Mode T redesigned — per-regime shield sweep** — Now sweeps threshold × failsafe × bull_on × bear_on (4 on/off combos), picks the quadruple maximizing bull+bear total return. Writes `bull.hold_shield`, `bear.hold_shield`, shared `min_sell_pnl_pct`, `max_hold_hours`. Current ETH winner: `bull_shield=ON, bear_shield=OFF, min_sell_pnl=0.60%, max_hold=12h` → +115.50% vs all-OFF +92.05% (delta +23.45pp). All top-8 combos had bull=ON/bear=OFF — signal unambiguous.
- **Mode T chains rally-cooldown (T→G integration)** — After writing shield config, T now merges its fresh bull/bear signals via `_merge_tagged_signals()` (regime tag per bar via configured detector) and calls the shared `_sweep_rally_cooldown()` helper. No stale-cache issue: T and G share one signal stream. T total runtime ≈ 3-5 min. Rally-cooldown winner today: `rr10h ≥ 2.5% OR rr30h ≥ 6.0%, cd=20h` (H1=+31.01%, H2=+22.22%, REF=+60.12%, worst_dd=+4.55%, plateau=1.00). Mode G standalone kept for cache-fed fast iteration.
- **Mode G simulate() per-regime policy awareness** — Previously hardcoded `MIN_SELL_PNL_PCT=0.005, MAX_HOLD_HOURS=10` + cache's `conf_threshold`. Now reads `bull/bear.min_confidence`, `bull/bear.hold_shield`, shared `min_sell_pnl_pct`, `max_hold_hours` from `regime_config_ed.json`. Each bar's policy keyed by `s['regime']`. Reflects true live-trader behavior.
- **Mode G `--rank recent|balanced` flag** — Default `recent` (H1-focused tiebreak: `pnl_H1 − 0.5 × |dd_H1|`). `balanced` uses prior behavior (`avg_pnl_halves − 0.5 × worst_dd`). CLI + chain-mode wiring (G, RSG, HRSG, DVRSG, HRSTG).
- **_merge_tagged_signals bug fix** — First integrated T→G run tagged 100% of bars as bull. Root cause: detector dict keys are naive `pd.Timestamp`; merge was normalizing to UTC-aware before lookup → every `dt not in ind` → default `True`. Fixed by using naive Timestamps for detector lookup (matches `extend_caches_90d.py` pattern). Post-fix: 1442 bars tagged bull=888 / bear=554 — proper ~60/40 split.
- **extend_caches_90d.py modernized** — Was pulling bear row from archived `crypto_doohan_v1_6_production.csv`. Now reads current `crypto_ed_production.csv` for both regimes, horizons + conf thresholds sourced from `regime_config_ed.json`. Cache rebuilt: 2185 hourly signals, fresh from HRS-retrained models.

### Completed (2026-04-17)

- **Engine Reference Card** — Built from live audit of `crypto_ed_production.csv` (48 models) + history. Feature grades (5-1) across technical/macro/cross-asset/sentiment/PySR/on-chain/derivatives. See "Engine Reference Card" section above.
- **ETH Mode D `--replay 1440` (late evening)** — New prod row winners written: ETH 5h (RF+LGBM, APF higher than pre-fix), consistent with new PySR. Exact numbers superseded by 2026-04-18 HRS rerun.

### Completed (2026-04-16)
- **11 audit-pass bugs fixed**:
  1. **(HIGH) Position file race** — `load_position()` / `save_position()` in `crypto_revolut_ed_v2.py` now share a `_position_lock` and write atomically (temp + `os.replace`). Telegram thread + main loop were both touching the same JSON without sync. Adds `JSONDecodeError` fallback to cash state.
  2. **(HIGH) `regime_config_ed.json` non-atomic write** — Added `_atomic_write_json()` helper in `crypto_trading_system_ed.py`; replaced all 5 raw `json.dump` writes (Modes S, T×2, R, G). Mid-write crash or chained-mode race no longer corrupts prod config.
  3. **(HIGH) `usd_invested` mismatch on partial fill** — `crypto_revolut_ed_v2.py` lines 1020 (auto-buy) and 1603 (manual `/buy`) now compute `usd_invested = filled_size * fill_price` from actual order data instead of the requested amount. PnL math no longer overstates basis on partial fills.
  4. **(MED) Optimizer bot blocking stdout** — `_run_job()` in `crypto_optimizer_bot.py` now reads subprocess stdout via a daemon thread feeding a `queue.Queue`, polling with a 1s timeout. `/cancel` reaches the worker even if the optimizer stalls or goes silent. Heartbeat progress edits keep the Telegram message alive during quiet phases.
  5. **(MED) Confirm-button double-click** — `opt_confirm` in `crypto_optimizer_bot.py` checks `_menu_state['step'] == 'confirm'` and nullifies the step before `_enqueue_job()`. Fast double-tap can no longer queue duplicate jobs.
  6. **(MED) Hold-shield naive-local time** — `crypto_revolut_ed_v2.py` now stores `entry_time` as UTC ISO 8601 (`2026-04-16T01:00:00Z`) via `_now_utc_iso()`. Hours-held math uses `datetime.now(timezone.utc) - _parse_entry_time_utc(...)`. Parser still accepts the legacy naive-local format for the existing ETH position. Display sites use `_format_entry_time_local()` to convert ISO back to readable local time. DST flips no longer skew the shield by 1h.
  7. **(LOW) Dead `'DVS'` mode** — Removed from `REPLAY_MODES` set in `crypto_optimizer_bot.py`. No button emitted it; no `MODES` entry existed.
  8. **(LOW) Mode P horizon menu dead branch** — Collapsed the misleading `if/else` where both arms called `_show_horizon_menu()`. Comment promised a skip that was never implemented.
  9. **(LOW) `SV3` / `BLOWOFF` dead branches in `_run_job()`** — Removed two unreachable mode branches plus the now-unused `SCRIPT_PATH_V3` constant.
  10. **(LOW) Silent feature drop in `generate_live_signal()`** — `crypto_live_trader_ed.py` now prints a warning when configured features are missing from `all_cols` (partial-match case), and a separate warning when zero match and we fall back to `FEATURE_SET_A`. Production model/feature drift is no longer silent.
  11. **(LOW) PySR regime detector silent 0.0 substitution** — `_evaluate_pysr_detector()` in `crypto_live_trader_ed.py` now collects missing/NaN feature names, logs them, and returns `True` (bull) instead of evaluating the formula with substituted zeros. Bull is the safer default since bull horizons run higher confidence thresholds.
- **`/optimize` + `/optstatus` removed from trader bot** — Handlers were still wired despite being removed from `/help`. Dropped the whole block in `crypto_revolut_ed_v2.py` plus its dispatcher branches and the `subprocess` import (only used by that handler). Optimizer is launched from its own bot.
- **Bull/bear + gate ON icons recolored blue** — `/status` bull icon 🟢→🔵 at line 1404, `/gate` ON state 🟢→🔵 at line 2565. Matches the BUY=🔵 / SELL=🔴 convention used everywhere else.
- **V7 rally-cooldown gate in production** — BUY gate wired into `crypto_revolut_ed_v2.py`. Winner from `audit_v6_v3.py` 49,716-config grid: block BUYs for 30h when `rr_8h ≥ 3%` OR `rr_36h ≥ 5.5%`. Top by `score_dd_aware` (12.40) in a `plateau_score=1.00` ridge — every ±1-step neighbor also passed STRICT (beats V0 on H1 AND H2 AND 60d). Sweep perf: H1 +10.42% / H2 +18.01% / 60d +31.84% / worst DD −3.63%. Params in `regime_config_ed.json → ETH.rally_cooldown`. State `rally_cooldown_until` persisted in `position_ed_v2_ETH.json` (survives restart).
- **Gate trigger-detection bug fix** — Original wiring ran trigger check only inside the BUY-while-cash branch → gate never fired while invested. Split into `_update_rally_cooldown()` (every tick, unconditional) + `_is_rally_cooldown_active()` (BUY-time check). Matches backtest semantics.
- **Mode G added** — New mode in `crypto_trading_system_ed.py`: runs the rally-cooldown sweep programmatically and writes `regime_config_ed.json → rally_cooldown` for each enabled asset. STRICT pick rule first (beats_3of3 AND plateau≥0.7, tiebreak by `score_dd_aware`). Chained variants: G, RSG, HRSG, DVRSG, HRSTG.
- **Default replay window → 2 months (1440h)** — `crypto_trading_system_ed.py` Modes D/V/R/S/T now default to 1440h when `--replay` omitted. Matches user's typical entry.
- **`/gate` Telegram command** — Per-asset rally-cooldown toggle in trader bot. `/gate` (status + buttons), `/gate on|off`, `/gate ETH on|off|clear`. The `clear` action wipes `rally_cooldown_until` as a one-time override. Added to `/help`.
- **Optimizer bot menu simplified** — `_show_mode_menu()` replaced with 3-profile front door: 🔧 Full Re-tune (HRS) / 🔄 Regime Refresh (RS) / ⚡ Model Refresh (DV) / 🔬 PySR. Full 12-mode grid moved behind `Advanced ▸` button with `◂ Back` return. Pattern: letters alone are debug tools; chains are real workflows.
- **Chain-order verified (one-off)** — `compare_chain_order.py` 6-fold rolling walk-forward (30d train / 10d test / 10d step): Path B (joint model+gate search) beat Path A (G-last, current) by mean +0.49% / median +0.35% PnL across folds — noise-level, 3 wins / 1 loss / 2 ties. Verdict: keep G-last in production, ordering immaterial. Single 60/30 split earlier showed +7.92% for B but was one fold of cherry-picked data.
- **Adaptive gate tested and rejected** — `compare_gate_adaptive.py` tested "lift cooldown early if price reverts to pre-rally level". On 30d: `V7b lift ≤ 36h-base` never fired (0 early lifts over 30d — reversion-to-base condition not met). On 90d: adaptive variants all LOSE vs fixed 30h (−8.8% to −23.7% PnL). Post-rally reversion is partial; the extra cooldown hours past reversion are where bad BUYs cluster. Keep fixed 30h.

### Completed (2026-04-15)
- **Maker window tuned for fill reliability** — Extended `_execute_maker_order` defaults from 120s/3s → 180s/10s after observing SELL fill asymmetry (BUYs filled in 1–29 reprices, SELLs took 23–31 with 2 MARKET fallbacks over past week). Slower reprice reduces cancel/repost churn that was costing queue priority on the SELL side. Symmetric change: BUY also benefits (fewer churn cancels at `bid+0.01`).
- **HRS ETH 6,7,8h TODO closed** — Mode D 2026-04-10 already validated GDELT (iran_vol_zscore ranked #9 on 8h). Prod is on the 4mo OOS winner `sma168>sma480 6h@90%/8h@90%` (+96.27%). Strictly-HRS rerun skipped: 1440h has 11pp nondeterminism (TODO #1) and would not improve on current prod config.

### Completed (2026-04-14)
- **Stop-loss / profit-lock backtest (ETH prod, 30d, 5-min res)** — **Verdict: keep prod as-is (no SL).** Ran 8 variants via `backtest_sl_variants.py`. Baseline A (no SL) won every dimension: +1.11% PnL / −8.71% DD. Disaster brakes −7% (B) and −10% (C) never fired in window → identical to baseline (free but unused). Profit-lock variants (D/E/F) and trailing HWM (G/H) all catastrophic: −11% to −20% PnL. Mechanism: scalping +0.15–0.28% gains on 31–49 "wins" chops big winners; losing setups still hit Shield-failsafe at −2% to −3%, so full loss price paid with no winner upside. F (tightest lock, +0.3%/+0.15%) worst at −20%. Artifacts: `backtest_sl_variants.py`, `backtest_sl_variants_summary.csv`, `backtest_sl_variants_trades.csv`. Revolut X API supports `tpsl`/`conditional` server-side SL via extending `place_*` helpers in `crypto_live_trader_ed.py` (`order_configuration: { conditional: {...} }`) but no longer pursued — user decided not to add insurance-only SL.
- **Hold Shield toggle** — `/hold` Telegram command + dynamic "🛡 Shield: ON/OFF" main button. Toggles `hold_shield` flag per-asset; persists to `regime_config_ed.json`. Shield gates SELL signals by blocking when `hold_shield=True AND PnL < min_sell_pnl_pct AND held < max_hold_hours`. Test suite `test_hold_shield.py` (8 sections, all pass). Commits eb5ea9f, 5ae895b.
- **Chart overhaul** — `/chart` now accepts horizon arg (6h..30d): `/chart`, `/chart ETH`, `/chart 12h`, `/chart ETH 7d`. Unambiguous markers (cyan ▲ BUY / orange ▼ SELL, correctness as ✓/✗/⏳ badge instead of color), legend always visible inside chart, horizon-scaled candles/axis/figure size. Correctness: ≥0.3% move in predicted direction within 4 candles.
- **Manual `/buy` `/sell` maker commands** — Telegram-triggered maker orders with fresh bid/ask quote, position update, full instrumentation (`print(..., flush=True)` + try/except + Telegram error popup). Silent 30-min death on 2026-04-14 morning traced to Windows stdout block-buffering — fixed by flush=True. Commits f522d1b, 8938f86.
- **Telegram HTML fix** — Replaced `<` with `vs` in hold override message (was breaking Telegram parse_mode=HTML with HTTP 400). Commit f440316.

### Completed (2026-04-10)
- **GDELT geopolitical features added** — `download_macro_data.py` now fetches GDELT DOC 2.0 data (iran_vol, iran_tone, geopolitical_tone) with rate-limit handling. `_compute_gdelt_features()` generates 15 features (raw, zscore, chg4h, chg24h, spike) per GDELT column. Wired into `build_all_features()` via hourly merge. Feature count: 51 base + 101 macro/sentiment/cross-asset/geopolitical = 152 total. GDELT features ranked #9-12 in LGBM importance on 6h/8h horizons (iran_vol_zscore #9/157 on 8h).
- **V3 joint sweep ported to production Mode S** — Replaced sequential R→S (greedy: R locks horizons at ≥90% conf → S sweeps conf only, 245 combos) with full joint sweep (detector × bull_h × bear_h × bull_conf × bear_conf, 3,920 combos with 4 horizons). Discovers global optimum across all dimensions simultaneously. Source: `crypto_trading_system_ed_v3.py`.
- **Mode D `--replay` parameter** — Replaces hardcoded `MAX_DIAG_HOURS = 6*30*24`. Available in CLI and Telegram bot (D added to `REPLAY_MODES`).
- **Telegram bot menu cleanup** — Removed BLOWOFF and SV3 buttons (noise reduction). Updated S label to "Joint Sweep (V3)", RS to "Regime + Joint Sweep". Added DVRS button. Time estimates updated (S: 60min, RS: 90min, HRS: 150min).
- **4 critical bugs fixed** — (1) GDELT CSV timezone: strip tz before save + `tz_convert(None)` on load for tz-aware data. (2) `_merge_hour` leaked into `all_cols` after GDELT merge. (3) Empty results IndexError in Mode S (`winner = results[0]` with no guard). (4) `_load_macro_csv()` crash on tz-aware CSV.
- **Iran ceasefire confirmed as cause of bad late results** — Apr 8 2026 Iran/Hormuz ceasefire caused +8% ETH rally, pure geopolitical event not model deficiency. Motivated adding GDELT features.
- **Mode D ETH 6,7,8h --replay 1440 run** — GDELT features validated. 152 features total. iran_vol_zscore ranked in top-12 importance.

### Completed (2026-04-09)
- **SV3 ETH 1440h** (`optimizer_20260408_155122.log`, finished 8 Apr 22:51 via Telegram). Winner: `vol_calm 7h@80%/6h@80%` → +57.68% / 63 trades / 75% WR / alpha +46.90%. Written to `regime_config_ed_v3.json` (research only — no prod impact).
- **RS ETH 1440h rerun** (`ed_v1_20260409_160324.log`). Winner: `sma168>sma480 7h@80%/6h@85%` → +52.36% / 62 trades / 71% WR / alpha +49.25%. Used to confirm RS↔V3 structural parity and to surface nondeterminism (see TODO #2).
- **Momentum-decay / blowoff filter sweep ETH 2880h** (`optimizer_20260409_024812.log`). Baseline +10.32%; best filter `A_6h ≥5% force_sell` → +10.91% (+0.58pp). Not actionable.
- **Prod confirmed** — `regime_config_ed.json` ETH = `sma168>sma480 6h@90%/8h@90%` from 4mo OOS sweep (`rs_eth_oos_4mo_20260408_065746.log`, +96.27%). Defensible: longer window damps the seed jitter that plagues 1440h sweeps.

### Completed (2026-04-08 — soir)
- **Ed V3 (research)** — `crypto_trading_system_ed_v3.py` Mode S full joint H-sweep: 5 detectors × 8 horizon pairs `(6,6)(6,7)(6,8)(7,7)(7,8)(8,8)(7,6)(8,7)` × 49 conf combos = 1,960 evals/asset. Writes to `regime_config_ed_v3.json` (zero prod impact).
- **Telegram optimizer bot** — Added `SV3` button (S V3 Joint H-Sweep) + `Help` button in mode menu.
- **Telegram trader** — Hourly update shows real detector name (e.g. `sma168>sma480`) instead of `named`; added `/help` line after date.

### Completed (2026-04-08)
- **BUG 1 fix — maker buy balance rounding** — Floor `buy_amount` to cents minus $0.01 safety margin before passing to maker buy (Revolut rejects when qty×price > balance by even $0.01).
- **BUG 2 fix — maker sell post_only race** — SELL price floor raised from `bid+0.01` to `bid+0.02` with second guard after rounding; on `post_only` rejection, retry loop with fresh quote instead of immediate market fallback. Confirmed via 2 Revolut rejection emails (08:51:05 / 08:51:12).
- **Maker window extended 60s → 120s** (40 attempts).
- **BUG 3 — `Unknown regime detector type: named`** — already fixed in `crypto_live_trader_ed.py` (commit fac33a4); needs trader restart to load.
- **6h vs 7h decision** — Mode V `--replay 4320` complete: 7h winner D #1 XGB+LGBM w=300 γ=0.999 f=25 @70% → +24.73%, 66 trades, 64% WR vs 6h Refined #1 +23.17%, 78% WR. Stayed on 7h (higher raw return).
- **ETH live trader verified** — `/regime` confirms named-detector branch firing, hot-reload working.
- **ETH config cleanup** — ETH block clean; disabled assets left as-is.

### Completed (2026-04-07)
- **R→S detector handoff fix (Option C)** — Extracted shared `_build_regime_indicators_and_detectors()` helper so Mode R + Mode S use the same indicator/detector dict. Mode S rewritten as joint sweep over all 5 detectors × 49 conf combos = 245 evals. Winner written to config as `{type: "named", params: {name: <detector>}}`.
- **Named-detector branch wired into live trader** — `crypto_live_trader_ed.py` `_evaluate_detector` now dispatches `type=='named'` to `_evaluate_named_detector` with implementations for all 5 detectors (incl. Andersen-Bollerslev deseasonalized `vol_calm`).
- **ETH RS rerun with fixed pipeline** — `sma168>sma480` 7h@75% / 8h@85% → +60.72%, 66 trades, 65% WR, alpha +49.98% (`ed_v1_20260407_160841.log`). Config updated and live.
- **Mode V `--replay` argument support** — `run_mode_v` / `_backtest_one_config` / CLI dispatcher now accept `replay_hours`. Telegram optimizer bot's `REPLAY_MODES` extended to include V/DV/DVS/S so the menu prompts for replay before launching.
- **Detector set trimmed 14 → 5** — Removed all RSI (5), drawdown (4), `macd>0`, and 9 redundant SMA/momentum variants. Final: `sma24>sma100`, `sma168>sma480`, `price>sma72`, `vol_calm`, `tsmom_672h`.
- **Detector set trimmed 5 → 2 on 2026-04-30** — `ENABLED_DETECTORS = {'tsmom_672h', 'sma24>sma100'}` constant added in [crypto_trading_system_ed_engine.py](crypto_trading_system_ed_engine.py). Cross-run analysis of 35 1440-window HRSTs (Apr 18-30) showed only `tsmom_672h` (66% TOP-15 presence, 9 wins) and `sma24>sma100` (69% presence, 8 wins) consistently appear in TOP 15. The other 3 (`price>sma72`, `sma168>sma480`, `vol_calm`) were single-run wonders or weak (≤54% presence). Mode S joint sweep drops 3,920→1,568 combos (60% less multiple-testing). All 5 detector lambdas still defined in `_build_regime_indicators_and_detectors` so the live trader can evaluate any named detector found in `regime_config_ed.json`; only the SEARCH FILTER is restricted. To re-enable for quarterly detector-rediscovery: edit `ENABLED_DETECTORS` to include the wider set, run HRST, revert.
- **Literature-grounded detectors added** — `vol_calm` (Andersen-Bollerslev deseasonalized intraday vol), `tsmom_672h` (Liu & Tsyvinski 2021 RFS). SMA windows extended to 168/240/480h.
- **Mode R top_n default → 200** (`crypto_trading_system_ed.py:4551`).
- **ETH-only trading** — BTC disabled and position sold (45% WR, avg loss > avg win over 1-month backtest). Full $12k → ETH.
- **ETH HRS 2-month** — bull=6h@90% (RF+XGB, +7.4%, 87.5% WR), bear=7h@75% (XGB+LGBM, +1.9%, 63.6% WR).
- **ETH position doubled** — $6k → $12k per trade both regimes.

### Completed (2026-04-05)
- **Maker order penny-improvement fix** — `bid+0.01` for both buy and sell with `post_only`. Stale orders cancelled before each maker attempt. Maker window reduced to 60s.

### Completed (2026-04-03)
- **Ed V2 release** — 6 critical bug fixes in `crypto_revolut_ed_v2.py`: clock drift (NTP sync + 409 auto-correct), `get_balances()` silent failures, ghost sells, locked funds blind sync, stale order cleanup, maker order pricing.

### Completed (2026-03-29)
- **BTC Ed regime fully optimized** — sma48>sma200, bull=7h@95%, bear=8h@90% → +50.35% over 4 months (78 trades, 69% WR). Ed live on BTC.
- **Mode S implemented** — Regime confidence sweep (7×7=49 combos). HRS/DVRS/RS combo modes. Both_agree removed from Ed. Mode H no longer picks winner (R does).
- **PySR regime discovery — FAILED** — Tested forward48, sma48_200, forward72 labels. Best accuracy 58% (too weak). sma48>sma200 hand-crafted detector confirmed as winner.
- **Ed Telegram display fixed** — Shows bull/bear horizons + confidence instead of old strategy/both_agree.
- **Ed V1.0 release** — Regime-switching system. Dynamic bull/bear horizon selection via `regime_config_ed.json` (not hardcoded). Mode R regime backtest (16 detectors × all horizon pairs). PySR regime discovery script. Separate production CSV. Runs alongside Doohan. Telegram `/regime` command.
- **All 9 assets Mode H complete** — DOGE (5h), ADA (7h), AVAX (all negative), DOT (7h) completed. Every asset now has production models.
- **SOL disabled from Doohan** — Turned off from live trading.
- **Trailing stop analysis** — Tested 0.25-1.0% trailing stops + profit targets + regime filter on 336h replay. Baseline signal exits beat all variants for both BTC (+$826) and ETH (+$207). Model signal quality is the edge.
- **Dynamic confidence analysis** — Tested raising min_confidence in bearish regimes. All variants lost money vs baseline — blocked winning contrarian trades.
- **Start scripts with tee logging** — `start_trader.bat`, `start_optimizer.bat`, `start_ed.bat` with Unicode-safe tee, auto-restart, venv auto-detect.
- **Optimizer bot Unicode fix** — PySR Julia output crashed the bot with charmap encoding error. Fixed with ASCII fallback.

### Completed (2026-03-27)
- **LINK Mode H horizon sweep** — 5 horizons (4h-8h). Winner: 8h RF+LGBM +7.77%, 14 trades, 86% WR, conf>=90%. Config updated.
- **BTC Mode H re-run with PySR** — 2 horizons (5h/6h). Winner: 6h XGB+LGBM +8.78%, 26 trades, 85% WR, conf>=70%. Config updated.
- **Bat file venv auto-detect** — `start_trader.bat` and `start_optimizer.bat` now auto-detect Desktop vs Laptop venv path. Previously hardcoded to Desktop path, broke on Laptop.

### Completed (2026-03-26)
- **V1.8 LSTM test — FAILED** — Tested LSTM as classifier in grid: LSTM solo (0 valid results, all failed), LSTM+LGBM (identical to RF+LGBM), LSTM+XGB (identical to RF+XGB). LSTM votes randomly, adds nothing. Confirms LGBM dominance — partner model is irrelevant. All ML improvement ideas now tested and resolved.
- **XRP Mode H horizon sweep** — 4 horizons (5h-8h). Winner: 8h XGB+LGBM +9.99%, 18 trades, 78% WR, conf>=80%. Config updated.
- **SOL Mode H re-run (full 4h-8h with PySR)** — 5 horizons. Winner: 8h RF+XGB +22.43%, 32 trades, 69% WR, conf>=75%.
- **PySR discovery for all assets** — Mode P completed for BTC, ETH, SOL, XRP, LINK, DOGE, ADA, AVAX, DOT (all horizons 4h-8h). Historical window method.

### Completed (2026-03-25)
- **Telegram optimizer bot** — `crypto_optimizer_bot.py`. Remote triggering of Mode D/V/H/P/S via inline keyboard menus. Sequential job queue, subprocess execution with unbuffered real-time progress output, below-normal Windows priority. Separate bot token to avoid conflicts with trader bot.
- **PySR leakage fix** — Initial PySR results were inflated (BTC 6h: +9.27% with leaky PySR vs +3.74% baseline, ETH 7h: +23.32%). Root cause: PySR formulas fitted on same 6-month window Mode D evaluates on. Fix: PySR discovery now uses historical window (months 12→6 ago), zero overlap with Mode D's last 6 months. Anti-leakage checks added in Mode D/V/Refine — strips PySR features early if JSON lacks `discovery_method == "historical"`. Mode V blocks production writes for leaky configs.
- **PySR promoted to production** — `_compute_pysr_features()` merged into `crypto_trading_system_doohan.py`. Loads `models/pysr_{ASSET}_{H}h.json` if exists, safe fallback if not. Clean PySR results pending re-run with historical window.
- **logret_5h and logret_7h added** — Fills gaps in short-term momentum for 5h/7h horizon models. 132 total features.
- **ETH Mode H horizon sweep** — 3 horizons (5h/6h/7h) with PySR. Winner: 7h RF+LGBM +23.58%, 16 trades, 62% WR, conf>=90%. Config updated to 7h/90%.
- **SOL Mode H horizon sweep** — Initial run 5h/6h/7h. Re-run 2026-03-26 with full 4h-8h + PySR (see above).
- **BTC 4h/8h with embargo fix** — 4h confirmed overfit (negative post-embargo). 8h not viable. BTC production stays at 6h.
- **V1.7.2 — Regularization** — Tested on BTC 6h. Minimal reg won (ra=0, rl=0.1, cs=0.9, ss=0.5). V1.7.1 baseline (+3.75/+3.47/+3.74 at 70/80/90%) more consistent than V1.7.2 (+3.63/+0.11/+4.64). **Verdict: wash, not adopted.**
- **PySR installed on both machines** — Laptop + Desktop done (2026-03-25). Julia 1.11.9 backend compiled.
- **Telegram balance bug fixed** — Exchange balances were fetched before trade loop but displayed after, causing false "Tracker says invested but exchange shows 0" warnings.
- **Scoring formula changed** — `return × (win_rate/100)` for positives, raw return for negatives. Favors consistency over raw return. Refined-only filter removed — D and refined compete equally.
- **Telegram horizon fix** — Signal line now shows actual configured horizons (e.g., `6h=BUY(78%)`) instead of hardcoded 4h/8h N/A.

### Completed (2026-03-24)
- **Doohan V1.7.1 promoted to production** — Renamed to `crypto_trading_system_doohan.py`. Embargo-fixed grid + Optuna refine. Variable horizon per asset. BTC 6h winner: XGB+LGBM w=252h g=0.994 f=9, +3.75% at conf>=70%.
- **Deku archived** — All Deku files moved to `archive/`. Deku replaced by Doohan as production system.
- **Refined-only production selection** — Initially Mode V only picked from refined configs (confirmed on BTC 5h/6h/7h). Later replaced by `return × win_rate` scoring with all candidates competing equally (2026-03-25).
- **Variable horizon support** — Trading config stores `horizon` per asset. Trader reads it and uses `Xh_only`. Mode H sweeps multiple horizons.
- **New Mode H** — Horizon sweep: D+V per horizon → cross-horizon comparison → saves best to trading config. `--skip` flag to reuse existing D results.
- **Order-independent CLI** — Arguments can appear in any order. `--help` shows full usage.
- **Price fix ported** — Live trader now reports current market price (`df_raw.iloc[-1]['close']`) instead of label-shifted historical price.
- **Dead code cleanup** — ~1,435 lines removed (legacy modes A/E/DAF, old Mode H, legacy Mode D).
- **Root folder cleanup** — 28 files archived. Root now has 7 Python files + configs.
- **Mode V confidence thresholds expanded** — Tests 65/70/75/80/85/90% (was 70/80/90).

### Dropped
- ~~Deku~~ — Replaced by Doohan V1.7.1. Embargo fix revealed Deku APFs were inflated.
- ~~Doohan V1.1–V1.6~~ — Superseded by V1.7.1 embargo fix. V1.6 grid approach preserved but with proper embargo.
- ~~CASCA~~ — Superseded by Deku, then by Doohan.
- ~~4h horizons~~ — Confirmed overfit (all D candidates negative post-embargo). 5h+ horizons are viable.
- ~~CPCV as validation~~ — Incompatible with temporal decay (gamma). 4h overfits (PBO=1.0), 8h is real edge.
- ~~Deku enhancements~~ — 11 features + 2 toggles all FAILED (-62.2%).
- ~~Multi-timeframe fusion~~ — Cross-TF fusion worse than single-timeframe.
- ~~V1.7.2 Regularization~~ — Wash. Minimal reg won but inconsistent across confidence levels. V1.7.1 unregularized baseline kept.
- ~~TabPFN / tabular transformers~~ — Tested and failed.
- ~~V1.8 LSTM~~ — LSTM solo: 0 valid results (all failed). LSTM+LGBM/XGB: identical to RF+LGBM/XGB (LSTM votes randomly, partner model carries all signal). Confirms LGBM dominance.

---

## 2026-07-01 — TODO cleanup (dashboard reset to active-only)

Full pre-cleanup TODO preserved verbatim in `TODO_pre_cleanup_20260701.md` + git history. The ~30 closed/stale/superseded dashboard rows below were moved out of TODO.md; one-line verdicts here, full detail in the backup + the commits cited.

**Big arcs (2026-06):**
- **P0-0629 backtest≠live — DONE.** Root cause = the backtest (`generate_signals`) and live (`generate_live_signal`) built different TRAINING WINDOWS on identical data (~1-2 edge rows; the top-γ row flips ~25% of signals → 75% backtest↔live agreement). NOT execution/lag/data-revision/device (each ruled out by test). Fixes (6 commits): `30c270e` atomic writes, `3bd053e` cross_asset clobber + new [4] DATA INTEGRITY sanity check, `08ddb27` incremental deriv download (P1 latency 33s→4s), `2460937` FAYE_FAITHFUL_WINDOW (75→99%), `da57e72` F2 leak fix (last train row → i-horizon, trader+shadow), `a2f553e` leakage-free DEFAULT. Leakage-free HRST → promoted ETH `sma168>sma480` bull 6h@80/bear 4h@65 (2026-06-30 00:32). See CLAUDE.md "Backtest-vs-Live Fidelity playbook".
- **P0-0629b data clobber — RESOLVED.** orderbook (1350→34) + options_iv (2358→48) clobbered ~06-28 (except-writes-fresh on a read-race). Fixed systemically via `_atomic_to_csv` shrink-guard (refuses replacing a >100-row file with <50% its size; `tools/test_shrink_guard.py` 6/6). Impact ~zero (affected feats quarantined/marginal). ⚠️ orderbook/IV HISTORY RESTORE still pending → tracked in the live G1/G2 TODO row.

**2026-07-01 session:**
- **[5] backtest-vs-live recent-only — DONE** (`358a415`). Filters the sanity [5] check to post-promotion hours (boundary = prod-CSV mtime, UTC epoch); `--all-hours` + WARMING-UP guard. Killed the config-straddle false FAIL (4h 59.5% → 39/39 = 100% PASS, 161 pre-boundary hrs excluded). [5] drives the daily verdict, so it was flipping the whole sanity to ATTENTION.
- **Mode-V refine-dispatch deadlock — FIXED 0614, VERIFIED 0701.** loky `get_reusable_executor().shutdown` before the fresh refine ProcessPool ([faye.py:10282](crypto_trading_system_faye.py#L10282)) + `_refine_top_configs_serial` fallback. Standalone `V ETH 5h` confirmed no-hang (ProcessPool dispatched + `multiprocessing.spawn` workers refining live @19:45, loky coexisting). TODO row was stale (fix landed 06-14, never marked). Op-gotcha saved: faye's `os.execv` re-exec detaches stdout under a bg shell → set `_FAYE_WARNINGS_BAKED=1`.
- **6h-bull Grid-vs-Refined — answered, NOT a bug.** HRST log: refine ran 3/3 configs on all 5 horizons (120-205 min); 6h/7h/8h legitimately picked Grid over *higher-APF* refined configs (6h refined APF=2.969 lost to Grid D#2) because selection = `return×WR` while refine optimizes `APF` — they diverge on long horizons. CLAUDE.md "refined-only selection" claim corrected. Open design Q: align the refine objective with the selection metric.
- **P0-0613 logging code-change — DONE (verified 0701, was a stale "READY" row).** The top-level `import re` crash-guard + engine→`logs/hrst/` routing + no-empty-turds landed in BOTH `crypto_trading_system_faye.py` ([:305](crypto_trading_system_faye.py#L305)/[:311](crypto_trading_system_faye.py#L311)) and `crypto_trading_system_ed.py` ([:142](crypto_trading_system_ed.py#L142)/[:147](crypto_trading_system_ed.py#L147)) sometime after 06-13; the TODO row was never updated (`logs/hrst/` has 102 files). Cosmetic leftover only (trader still flat-tees to `logs/ed_runtime_*.log` not `logs/trader/`; ~1275 old flat `*.log` unmigrated; `logs/misc/` unused) → demoted to a P4 housekeeping row.
- **ed.py RETIRED → `ARCHIVED/2026-07-01_ed_retired/` (2026-07-01).** `crypto_trading_system_ed.py` (Ed V1.0 engine) superseded by faye — one engine now (user: "I don't want 2 engines, it's misleading + it's corrections"). Serving migrated to faye 2026-06-05; the **last live/operational users repointed to faye today**: shadow feature-build (`c734642` — Rule-23 mirror gap closed, smoke signal-neutral: 196 cols identical, live 4h feats bit-identical), optimizer `SCRIPT_PATH` (+`_FAYE_WARNINGS_BAKED=1` in the spawn env so faye's stdout stays piped; jobs now write `models_faye/` staging → manual promote), meta `import`, validate_core (dropped `--engine ed`). faye's own `--help`/usage de-ed'd (22 CLI refs → faye + 2 now-moot daily-lag "mirror into ed" comments). **Verified safe:** `import crypto_trading_system_ed` → ModuleNotFoundError; shadow + meta + sanity import/run clean without ed; `sanity --quick` = 🔵 CLEAN (shadow 48/48 100%, snapshot 725/725). ~48 research `tools/` + `_idea_patchers` still `import` ed → ImportError until repointed to faye (the dir README documents the revive-by-repoint). **Closes the daily-lag arc:** faye is now the SOLE live builder (already lagged), so the "mirror lag into ed at promotion" deferral is moot ([[project_daily_data_lag_fix]] updated).
- **G1/G2 sparse-feature re-enable A/B — was already DONE 2026-06-22 (found stale-QUEUED in TODO, closed 2026-07-01).** The "sparse-feature HRSTC" tested the 7 `always_disabled_exact` feats; verdict (per `config/disabled_features.json` `_note`): **`deriv_oi_chg1d` (3/8 selection) + `ob_imbalance` (2/8) EARNED their spot → un-quarantined** (`deriv_oi_chg1d` is in the live 6h/4h models); **`avg_iv`, `iv_skew`, `spread_bps`, `deriv_oi_chg3d`, `deriv_oi_zscore` (0/8) → re-quarantined** (the current 5-feature `always_disabled_exact`). Ran on the FULL orderbook/IV history (pre the 06-28 clobber) → valid. The 5 could in principle be re-tested under the leakage-free engine, but 0/8 is decisive — low value. The TODO "QUEUED" row (0620) was never updated after the run. Orderbook/IV history restore is now decoupled (standalone data-recovery for a future HRST that might re-select the enabled `ob_imbalance`).
- **Vol-targeted position sizing — SHELVED 2026-07-01 (fresh bull-window test confirms it's not a return-adder on spot; user's "binary better" prior validated).** Gated A/B `tools/bt_vol_target.py --days 60 --end-offset 60` on the 2026-03→05 **+16.1% bull** (live sma168>sma480 6h/4h): **BINARY all-in/out +42.20%** vs the only DEPLOYABLE spot rows **VT W=24h cap1.0 +38.15% (−4.05pp, loses 2/3 sub-period chunks) / VT W=48h cap1.0 +41.76% (−0.44pp, +0.09 Sharpe = a wash)**. Binary beats-or-ties vol-targeting on every spot config in the bull — confirms the 2026-06-11 finding (the overall +3.15pp on the mixed 2mo window was entirely downside-driven; VT ties/gives up in up-markets). The leverage rows (cap1.5/2.0) add +5–15pp but need a margin venue (Revolut X has none on the directional book) → informational only. **VERDICT:** on a no-leverage return-maxing spot book, VT only trades a hair of return for a hair of Sharpe — not worth deploying. It IS a legit DD-smoother. **Revival conditions:** (a) a leverage/margin venue appears (cap>1 added +5–15pp here, +13–21pp historically), OR (b) you specifically want a smoother equity curve (DD −3.8pp in mixed windows) over max return. NOT Kelly (C22) — different axis (risk vs confidence), so this shelving is independent.
- **Feature-family full-removal — CLOSED 2026-07-01 (removal direction tested multiple ways, all BAD; user-confirmed).** The TODO "P2 open experiment" overstated it. Removing/restructuring feature families is consistently negative in the record: **ALL_external removal −6.07pp** (2026-06-13 ablation — macro/onchain/stablecoin are useful *context*); **multi-view feature split −45pp** (2026-06-10 — one model + full universe + LGBM importance beats partitioning FAST/SLOW; "don't split features across models"); **per-regime feature set C23 DEAD −1.86**. The only technically-untested variant was the CLEAN single-family Trim-B (re-run PySR without the family, then HRST) — but with ALL_external −6.07pp + families being useful context + cross-asset relstr being Grade-4, the prior is overwhelmingly negative → not worth an 8h HRST. The "+7.55pp individual family" result was Trim-A (redundant raw duplicates, PySR kept the signal) — never a real removal. **Lesson reinforced: one model + full feature universe + LGBM importance-selection is the local optimum; don't remove / split / partition families.**
- **C01 vol-scaled horizons — RETESTED leakage-free 2026-07-02, STAYS SHELVED (reconfirmed, now −19pp below live).** First idea from the post-leak-fix audit (user's hypothesis: the 06-29 leak fix is horizon-asymmetric so a horizon-switcher might revive). Ran `tools/test_vol_scaled_horizon.py` — **rewritten** ed→faye (leakage-free) + re-baselined from the defunct tsmom to the CURRENT live `sma168>sma480` 6h@80/4h@65; 2mo/1440h, 5 horizons, 21 vol-variants × conf sweep {65,80,85}, 3-chunk sub-period guard. **Result: LIVE +62.88% (84% WR); best vol-scaled variant (volMed hi>4h lo>6h @65) +43.88% = −18.99pp and loses live in the last sub-period chunk.** NO variant clears +5pp-over-live with sub-period consistency. **Why:** live's TREND detector + tuned per-regime conf beats VOL-based horizon selection; and long horizons (7h-only −1.3%, 8h-only −5.5%) LOSE in this bear window (ETH −33%), so vol-scaling toward 8h in calm periods HURT. ⚠️ Bear-dominated 2mo — a bull window *could* differ, but −19pp vs the HRST-optimized baseline makes a flip very unlikely. **Confirms the leak fix does NOT revive C01** — the audit lens holds (the leak was symmetric; a strong current baseline dominates). The revival condition ("different vol regime") is not met on this run; not worth the 4mo/2880 confirm (verdict rule only escalates on a promising 2mo). Tool committed (was broken — imported archived ed + used the tsmom baseline).

**2026-06 closed items:**
- **SHADOW DOWN 2.5d (0618) — DONE** (`b24e91a`). `_TeeStream` lacked `reconfigure` → shadow `import_failed` 06-15→18 (silent). Fixed + sanity now escalates monitor-down. Trader signals verified correct throughout (snapshot replay 368/368 = 100%).
- **Mode C "Choice" (0617) — DONE, verdict HOLD** (`b24e91a`). Native cross-window+hysteresis promotion gate; 2880h/9-window: challenger (tsmom_168h bull8h/bear5h w300) won only 5/9 (<6/9 bar) → HOLD. `MIN_ROBUST_WINDOWS=5` guard added.
- **Trader UX (0616) — DONE** (`b24e91a`). `/gate off` wipes the timer, `/gate clear` stamps `rally_cd_cleared_at`, `/help` shows live setup.
- **Gate-simplicity study (0616) — DONE.** Double-condition rally gate wasn't earning its keep: on 04-16→06-15 one leg (bear rr8≥2.5) carried the whole +7.67pp; engine now sweeps SINGLE-window gates only ([faye.py:7951], `h_long=h_short`/`t_long=9999` sentinel). Superseded by the 06-30 promote (single-form gates live).
- **GB/LGBM combo + hyperparam sweep (0609) — REJECTED.** GB+LGBM lost the gated engine to RF+LGBM (+43.5% vs +51.2%); LGBM-reg tune reversed. Removed from GRID_COMBOS. Screen rewarded conviction-erasing regularization ([[feedback_screen_vs_gated_engine]]).
- **Sanity made deterministic (0611) — DONE.** snapshot-replay (202/202) drives the verdict; parity demoted to informational (data-revision artifacts).
- **C61/C62/C67/C75 feature A/B (0611) — CLOSED, no signal.** Every "PASS" rode one horizon's overfit APF outlier; per-horizon deltas swing +20 to −5. Feature-add family exhausted. C61 moot (vol_of_vol_8h/24h already native).

**2026-06-05 / stale-config watches:**
- **Embargo-sensitivity 1-4h (0605) — DEAD.** Forward dry-run (`tools/dryrun_1_4h.py`, 5d): 1h −1.99%, 2h −0.54%, 3h +0.98% (noise), 4h −1.05% — the high DV WR was leak/overfit. Whole 1-4h band dead; production 5-8h unaffected.
- **2mo-vs-4mo HRST decision (0605) — resolved:** 2mo (`--replay 1440`) is the standard (the leakage-free HRST default).
- **May-31 FAYE-config watches** (live WR/P&L monitor + engine-vs-trader parity verification) — **superseded** by the 06-30 promote + shadow (100%) / snapshot-replay (368/368) validation.
- **SESSION 0620 (ed.py T bull-double fix) — moot:** the 06-30 promote replaced the `sma48>sma100` config entirely; the HRSTC half folds into the G1/G2 TODO row.
- **Re-run HRST on refactored engine (Step 6) — moot:** FAYE unified the backtest+live paths and is live.
