## ⚡ TODO — Active work only

**Companion file**: [ARCHIVED_LOG.md](ARCHIVED_LOG.md) — historical audit trail, canonical scoreboard C01-C82, MERGED TOPICS, IDEA QUEUE drop-list (closed/shipped/STUB items with verdicts).

## 📊 At-a-glance — active TODO dashboard (2026-05-31 — FAYE PROMOTED TO PRODUCTION)

| Pri | Item | Where | Status |
|---|---|---|---|
| 📌 | **FAYE LIVE** (promoted 2026-05-31 14:22 CEST) — `tsmom_672h` detector / bull 6h@65% / bear 8h@65% / shields OFF / rally cooldowns ON (custom per-regime params) / min_sell_pnl_pct=0 / max_hold_hours=10 / maker orders ON. Models: ETH 6h/7h/8h RF+LGBM from FAYE H. Rollback in `archive/g_narrow_live_20260531_142202/`. | **Desktop** (always) | 🟢 running |
| 🔥 **P0** | **Live WR/P&L monitor on new FAYE config** — first 1-3 days = sanity window, 2-4 weeks = real validation | **Desktop** (passive) | 🕐 STARTED 2026-05-31 14:22. Watch: signal cycle at next hourly tick using bull=6h+bear=8h (was 5h+8h); WR tracking close to Mode V Step 3 predictions (79-83%); total return tracking the +55%/+37%/+46% scale on similar period. **Rollback trigger**: live WR <60% over 7+ trades or persistent negative trades week 1 → revert to G_narrow archive. |
| ✅ DONE 2026-05-31 | **FAYE promoted to production** (was P0 BLOCKED) | — | ✅ Promoted 14:22 CEST. Archive: `archive/g_narrow_live_20260531_142202/`. Trader hot-reloaded and picked up new config. Gates (a)+(c) deferred to post-promotion validation; gate (b) hard A/B remains queued in P3 as nice-to-have not blocker. |
| ✅ DONE 2026-05-31 | **FAYE RST ETH 6h,7h,8h --replay 1440** | — | ✅ DONE 14:14 — Mode R picked tsmom_672h × 6h/8h regime pair (REF +72.75% / +76.05% / B&H +61.88%). Mode S optimized confs at 65%/65%. Mode T converged shields=OFF + rally cooldowns + min_sell_pnl=0 + max_hold=10h. Total RST wall ~80 min. |
| ✅ DONE 2026-05-31 | **FAYE H ETH 6h,7h,8h --replay 1440** | — | ✅ DONE ~14:30 — Winners: 6h RF+LGBM w=150 γ=0.996 f=15 **+55.30%** WR=79.3% (Grid); 7h RF+LGBM w=151 γ=0.9992 f=15 **+37.54%** WR=82.4% (Refined); 8h RF+LGBM w=155 γ=0.9997 f=11 **+46.37%** WR=83.0% (Refined). 5 bugs found+fixed mid-run (#15-18 + sleep-guard). |
| 🔥 **P0** | **Shadow mode continuous match-rate check** — primary live correctness gate | **Desktop** (passive observation) | 🕐 IN PROGRESS — every 1-2 days run the match-rate query; any drop below ~99% = NEW bug to investigate. |
| ✅ CLOSED 2026-05-30 | **Counterfactual backtest on wider window** — superseded by engine-vs-trader parity test | — | ✅ CLOSED — original goal was statistical proof of macro_cache fix's economic impact. Yesterday's parity test (`tools/validate_core_against_signal_log.py --samples 30 --recent-only`) gave a more direct answer: 22/30 direct MATCH, **0/8 real BUY↔SELL flips**, all DIFFs HOLD-threshold boundary cases. Engine and trader codepaths agree on direction whenever both produce one. Wider counterfactual would only re-measure what the parity test already confirmed. Live WR/P&L monitor continues to validate economic impact in real time. |
| 🔥 **P1** | **Embargo A/B test** (`tools/embargo_ab_test.py --mode=both`) | **Laptop** (~2.5h) | 📅 NEXT — quantifies contribution of embargo (divergence #1 of the original TODO 0526 "4 semantic differences") to backtest-vs-live gap. Now actionable as scoping for Step 6 refactor. |
| 🔥 **P1** | **TODO 0519B-G1** — `deriv_oi_*` re-enable A/B test | **Desktop** (~6h, off-hours) | 📅 PENDING — Desktop is free between trader cycles. Newly relevant post-cache-fix: with fresh derivatives features now flowing, this could swing differently than pre-fix expectation. |
| 📋 **P2** | **Step 6 engine refactor** (`docs/STEP_6_ENGINE_REFACTOR.md`) — make backtest call `compute_signal_core()` so HRST results predict live | **Laptop** (~1.5 calendar days + 12h compute) | 📋 DESIGNED — implementation pending. After this, backtest WR projection will be realistic (likely 65-75%, not 85%). Required before next promotion. |
| 📋 **P2** | **Re-run HRST on refactored engine** to get realistic backtest WR | **Desktop** (one HRST, ~7h) | ⏸ BLOCKED — depends on Step 6 done |
| 📋 **P2** | **TODO 0519B-G2** — orderbook + IV re-enable A/B test | **Desktop** (~6h) | 📋 SCHEDULED ~2026-06-18; depends on G1 outcome |
| 📋 **P2** | **Verify feature importances stable** after cache fix — re-run Mode V importance ranking, compare to pre-fix | **Laptop** (~30 min) | 📋 OPTIONAL — sanity check that the same features still rank high once they actually vary across time |
| 🚀 P3 | **Continuous macro archeology** — capture daily snapshots so future PIT validation has clean coverage | **Desktop** (cron, 5 min/day) | 📅 NEW — set up nightly `python tools/drive_archeology.py --preset all` so the next time we need PIT, drift is bounded |
| 🚀 P3 **AFTER FAYE IN PROD** | **Counterfactual: ffill vs mean_last_10 on trader's actual May 1-28 hours** — measure exact signal-flip count, not estimate | **Desktop or Laptop** (~1-2h dev + ~15-30 min run) | 📅 DEFERRED 2026-05-29 — analytical estimate was 5-15 hours of 723 (1-3%) would emit different action under mean_last_10; net economic impact estimated ±0.5-1.5pp/month (in noise range). Exact count needs full counterfactual: bypass engine's auto-ffill, build features WITH NaN intact for each hour, call `compute_signal_core` with both `na_policy='ffill'` and `na_policy='mean_last_10'`, run trader's actual model, diff predictions. Existing `tools/counterfactual_backtest.py` is the framework but is built for cache-bug testing, not fill-policy testing — needs adaptation. Outputs: exact flip count, signal-distance histogram, per-action breakdown (BUY/HOLD/SELL transitions). Only run AFTER FAYE in production so we have validated mean_last_10 behavior to compare against. |
| 🚀 P3 | **P4** — C14 vol-conditional triple-barrier retest | Laptop (~2.5h) | open |
| 🚀 P3 | **P5** — C11 VPIN at 5-min cadence | Laptop (~1 day eng) | open |
| 🚀 P3 | **P6** — C15 meta-labeling SOL/BTC | blocked (assets shelved, ~6h) | open |
| 🚀 P3 | **IDEA QUEUE Tier A** — Untested clean (C13-narrow, C54, C55, C58, C59) | research backlog | 5 items open |
| 🚀 P3 | **IDEA QUEUE Tier B** — V3-lit C60-C82 (16 of 23 patcher-ready) | research backlog | 23 items open |
| 📋 **P2** | **Validate FAYE bug #16 perf patch HARD A/B** — soft validation DONE 2026-05-31 (chunks find variance, max-aggregation works), hard A/B still pending | **Desktop or Laptop** (~3-4h for one horizon) | 📅 PENDING — recipe: `$env:FAYE_REFINE_TRIAL_SPLIT=1; $env:FAYE_REFINE_EARLYSTOP_PATIENCE=0; python crypto_trading_system_faye.py V ETH 6h --replay 1440`. Compare Step 3 winner to today's H output (6h: w=150 RF+LGBM g=0.996 f=15 +55.30%). Acceptance: within ±2pp return + same combo/window region. Run when Desktop is idle (after RST done). **Promotion blocked on this passing.** Optimization further (queue items): re-dispatch unused workers on a TPE-shared study via RDB storage so chunks share TPE history (recovers sample efficiency); make WORKERS auto-scale by `cpu_count()` not hardcoded sweet-spot for 28-core. |
| 🚀 P3 | **Verify trade count for 6h winner at best_conf** — the +55.3% return at WR=79.3% might be from very few trades at a high conf threshold (small-sample cherry-pick from Step 3's 6-conf scan [65/70/75/80/85/90]) | **Desktop or Laptop** (~5-15 min) | 📅 NEXT — re-run single backtest of the production winner cfg, expose per-conf trade counts. If best_conf was 85% or 90% with only 5-10 trades, +55.3% is statistically weak. Easier alternative: search the H run's terminal scrollback for "OVERALL BEST: ... → ETH 6h" line — it prints conf and trade count there. |
| 🚀 P3 | **Investigate 8h Mode D survivor count** — only 2 candidates survived 3-fold rolling holdout vs 10 for 6h. **NOT A BUG**, identified as data-driven: 8h labels noisier → models less confident → 0-trade holdout filter prunes most candidates | **Desktop or Laptop** (~30 min) | 📅 LOW — could (a) lower holdout conf threshold for harder horizons, or (b) loosen 0-trade filter to "0 trades only if ALL 3 folds 0". Either is a behavior change that needs care. Not blocking. |
| ⚪ P4 | **TODO 0519C** — CPCV HRST diagnostic | trigger-based re-run | available, no plan |
| ⚪ P4 | **Kalshi** — prediction-market data integration | needs API key + impl | backlog |

### Recently CLOSED (2026-05-31)

| Item | Status |
|---|---|
| **Bug #15** — refine ignored `--replay` (4320h hardcoded `MAX_DIAG_HOURS`) | ✅ FIXED commit `7fad9bf` — threaded `replay_hours` through `_refine_top_configs` + `_refine_top_configs_serial` + 3 call sites. Validated end-to-end on DV ETH 7h --replay 528. Was inherited from v3 chain (parallel_nearlive.py:480). |
| **Bug #16** — refine perf: trial chunking + 6-worker pool + early-stop callback | ✅ SHIPPED commits `9f286f8` (opt-in) + `3b2426b` (defaults ON). Env-controlled: `FAYE_REFINE_TRIAL_SPLIT` (default 3), `FAYE_REFINE_WORKERS` (default 6), `FAYE_REFINE_EARLYSTOP_PATIENCE` (default 15). Soft validation done (chunks find variance, max-aggregation captures it). **Hard A/B still pending → P2 above.** |
| **Bug #17** — refine worker pool halved when n_cfgs<3 | ✅ FIXED commit `7ce0320` — formula `trial_split * max(1, n_cfgs-1)` gave 3 instead of 6 when only 2 candidates passed Mode D's holdout filter (8h case). Changed to `min(trial_split * n_cfgs, 6)`. My initial "Windows kernel handle leak" hypothesis was wrong — actual root cause was a one-line arithmetic bug. |
| **Bug #18** — early-stop callback dormant in chunked mode | ✅ FIXED commit `e16853e` — hardcoded `n_complete < 20` warm-up combined with chunked 25-trial budget meant earliest trigger was trial 35, exceeding chunk size. Callback NEVER fired in production chunked runs (bug #16's claimed "30-50% savings on convergence" never materialized). Now scales: `warm_up = max(5, n_trials//4)`, `patience = min(env, n_trials//3)`. Will activate on next H run. |
| **Sleep-guard** — prevent Windows system sleep during H/HRST runs | ✅ SHIPPED commit `b5cbe23` — `SetThreadExecutionState(ES_CONTINUOUS \| ES_SYSTEM_REQUIRED)` at `__main__` startup; atexit restore. Display sleep still allowed. No-op on non-Windows. Triggered by 2026-05-31 overnight ~18min loss when Desktop slept mid-run. |
| **Banner UX** — refine output shows `[chunk X/Y, seed=Z]` when chunking active | ✅ SHIPPED commit `8875934` — disambiguates 3× repeated "Refining #1" / "Refining #2" output lines that confused real-time tailing. Suffix only when n_chunks>1; legacy output unchanged. |
| **FAYE H ETH 6h,7h,8h --replay 1440** — first production H run on FAYE | ✅ DONE 2026-05-31 — winners: 6h +55.3% Grid, 7h +37.54% Refined, 8h +46.37% Refined. All RF+LGBM, window 150-155, gamma 0.996-0.9997. Strong cross-horizon coherence. Bugs #15/16/17/18 all discovered+fixed during this run; #17 baked into 8h's slower wall time but didn't affect correctness. Production CSV: `models_faye/crypto_faye_production.csv`. |
| **Investigation: Step 1 → Step 3 gap on 6h (+0.7% → +55.3%)** | ✅ NOT A BUG — Step 1 ranks at PRIMARY_CONF=80% only; Step 3 scans 6 conf levels [65/70/75/80/85/90] and picks best return×WR with ≥5 trades min. Different metrics → different winners. Statistical robustness still requires verifying trade count at best_conf (P3 above). |
| **Investigation: 8h Mode D only 2 candidates vs 6h's 10** | ✅ NOT A BUG — 3-fold rolling holdout filter at line 5592 (`if ho_entry[4] == 0: continue`) drops candidates with 0 trades in holdout folds. 8h labels noisier → models less confident → more 0-trade results → fewer survivors. Possible loosening as P3 above. |

### Recently CLOSED (2026-05-30)

| Item | Status |
|---|---|
| **v3 HRST on Desktop** (ETH 5h,6h,7h,8h --replay 1440) | ✅ DONE 2026-05-30 — completed all 4 horizons. 5h winner ETH RF+LGBM w=200 γ=0.999 10f Grid +49.56%. **NOT promoted** — superseded by FAYE H run on 2026-05-31 with bugs #15-18 fixed. |
| **FAYE single-file consolidation** (7 phases, commits `8c122ef` → `4ab34d5`) | ✅ DONE 2026-05-30 — `crypto_trading_system_faye.py` (~9100 lines) collapses the Ed v3 4-layer monkey-patch chain into one file with ZERO monkey-patches. Native K=5 + 8-worker Mode D + 3-worker hybrid refine + NEAR_LIVE defaults + isolated `models_faye/`+`config_faye/` outputs. Smoke test `tools/smoke_test_faye.py` 38/38 ✓. CLI identical to v3. Full writeup in ARCHIVED_LOG.md. **Not yet in production** — first FAYE HRST to validate equivalence with v3 still pending. |
| **Engine-vs-trader parity test** on G_narrow LIVE config (30 recent ETH hours) | ✅ DONE 2026-05-30 — `tools/validate_core_against_signal_log.py --samples 30 --recent-only --cpu-lgbm`. 30/30 evaluated, 0 errors, 22/30 direct MATCH (73.3%), **0/8 real BUY↔SELL flips** — all DIFFs are HOLD-threshold boundary cases (5 live=HOLD because conf<95% bear threshold; 3 core=HOLD because probability<50%). Engine and trader codepaths agree on direction every time both produce one. **The original "bug between live trader and crypto trading" is gone.** Output: `output/core_validation_20260530_015454.csv`. |
| **Post-FAYE archive cleanup** (`ARCHIVED/2026-05-30_post_faye_cleanup/`, commit `1fce7f8`) | ✅ DONE 2026-05-30 — Tier A (8 variant scripts: `_cdar`, `_cvar`, `_cpcv`, `_robust`, `_h_strict_family`, `_noprod`, `_pre_macro_cache_fix`, `_launch_h_strict_family.bat`) + Tier B (14 variant-driving tools: CDaR/CVaR/CPCV/robust/h_strict_family clusters) + Tier C (14 old `models_g_desktop_*/` + `config_g_desktop_*/` snapshot dirs ~440KB) + docs (`CLAUDE_NEW.md`, `TODO_TEST.md`). Root file count 38→29, root dir count 35→21. Per-item grep-checked first per memory rule; NOT archived = anything still imported by v3 chain. Smoke test still PASS after move. Restoration recipe in `ARCHIVED/2026-05-30_post_faye_cleanup/INDEX.md`. |
| **NEAR_LIVE_MODE HRST (v3 fork research run, 2026-05-27 → 2026-05-30)** | ✅ DONE 2026-05-30 — superseded by FAYE consolidation. The v3 run on Desktop completed 5h and is in 6h (Mode V Step 2 refine, ETA full HRST done ~17:30 May 30). When done + trader flat, copy v3's winners into `models/`+`config/` (Critical Rule 19). 5h winner: ETH RF+LGBM w=200 γ=0.999 10f Grid +49.56%. |

### Recently CLOSED (2026-05-27)

| Item | Status |
|---|---|
| **TODO 0527** — `_macro_cache` mtime bug | ✅ DONE — root cause of live-vs-backtest gap. Cache fix shipped; shadow mode 100% match. See ARCHIVED_LOG.md for full writeup. |
| **TODO 0526** — LIVE vs BACKTEST divergence investigation | ✅ CLOSED — superseded by TODO 0527 root-cause discovery. "4 semantic divergences" framing was directionally right but cache was the dominant cause. Step 6 refactor still pending to address residual backtest-vs-live semantic gap. |
| **Audit of sig_1/sig_2 + data drift fix + cache fix** | ✅ DONE — 3 audits across multiple angles each. Found + fixed 4 additional latent HORIZON_SHORT/LONG bugs (sig_short Telegram, asset preflight, gamma fallback, _log_signal edge case) + 1 missing safe-merge call site (`download_fear_greed`). |
| **Counterfactual backtest tool** (`tools/counterfactual_backtest.py`) | ✅ NEW — runs hourly inference with fresh data via oldest-wins archeology + simulates trades. 5-day result on May 22-27: +0.50pp return, 2× per-trade edge, smaller drawdowns vs broken-cache actual (4 vs 7 trades, sample too small to be definitive). |
| **CLAUDE.md stale-reference cleanup + ETH legacy-key strip** | ✅ DONE — CLAUDE.md now reflects G_narrow live state; regime_config_ed.json ETH block stripped of inert top-level `horizon: 8` + `min_confidence: 85` legacy keys (no behavior change). |
| **TODO 0525** — G_narrow_d HRST with extended grid (May 25-26) | ✅ DONE — REF +83.85% lost to LIVE +91.01% by 7pp. Triggered TODO 0526 architecture analysis which led to TODO 0527 discovery. Underlying hypothesis (extended grid unlocks high-window basin) rejected. |
| **TODO 0524** — Top-5 HRST clean rerun (May 24-25) | ✅ DONE — REF +80.56% lost to LIVE +91.01%. Parallel fork validated (~8× refine speedup retained). |
| **TODO 0522** — Parallel refine speedup fork | ✅ DONE — Stage 1 passed; Stage 2 verdict invalidated by grid bug, bug-fixed, superseded by TODO 0524. |
| **TODO 0519** — G_narrow_d relaunch on Desktop | ✅ DONE — REF +89.14%, no STRICT winner but per-horizon V winners drove G_narrow promote 2026-05-21. |

### Machine allocation summary

| Machine | Active load | What runs here next |
|---|---|---|
| **Desktop** | Trader (live) + v3 HRST (started 2026-05-29 19:10, currently 6h Mode V Step 2 refine, ETA done ~17:30 May 30) | When v3 done: promote winners + restart trader; then first FAYE HRST as the next validation run |
| **Laptop** | Currently idle (this is where FAYE was built) | Wider counterfactual backtest; embargo A/B test; idea-queue research. Optionally first FAYE HRST here if Desktop is busy with trader-only after promo. |

**Honest top-of-mind (2026-05-30 ~02:00)**: FAYE single-file consolidation shipped today — `crypto_trading_system_faye.py` collapses the v3 4-layer monkey-patch chain into native code, smoke-tested (38/38 ✓). Engine-vs-trader parity test on 30 recent ETH hours showed **0 real BUY↔SELL flips** — the major signal-divergence bug is gone, codepath-wise. The v3 HRST on Desktop is still running (5h done, 6h in Mode V Step 2 refine, ETA ~17:30 May 30). **Next 12 hours: v3 HRST finishes → promote winners to `models/`+`config/` (Critical Rule 19: trader flat) → restart trader.** **Next 1-2 days**: first FAYE HRST on same May data to validate it produces equivalent winners to v3. If equivalent ±2pp, FAYE replaces v3 as canonical engine path. **No FAYE-to-prod promotion until that validates.** The realistic backtest WR target after FAYE is in production is still ~65-75% (per Step 6 thesis) — anything higher is the old binary-step backtest math, not live-equivalent.

---

**Layout (priority-ordered, top → bottom)**:
- 📌 LIVE STATE (always visible — current production config + rollback)
- 🔥 **P1** — Act this week (in-flight + imminent)
- 📋 **P2** — Scheduled next month
- 🚀 **P3** — Research backlog (when capacity allows)
- ⚪ **P4** — Low priority / Diagnostics / Backlog

---

## 📌 LIVE STATE — G_narrow models on H75 engine (promoted 2026-05-21 21:56 CEST)

**Engine** (since 2026-05-18 H75; macro_cache mtime fix patched 2026-05-27 11:22): `crypto_trading_system_ed.py` — H_STRICT_FAMILY merge (K=5 multi-seed + REFINE_TRIALS=75 + strict `(combo, w)` dedup) + mtime-aware `_load_macro_csv` (lines 1077-1110). Pre-fix snapshot: `crypto_trading_system_ed_pre_macro_cache_fix_20260527_112231.py`.

**Models + regime config** (swapped 2026-05-21 21:56 from H75-fresh to G_narrow_d's HRST output; G_fresh "promote" on 2026-05-22 19:51 was content-identical and effectively no-op for ETH 5h/8h):
- Detector: `sma24>sma100` (unchanged across all 3 promotions)
- Bull = **5h@65%** RF+LGBM w=281 γ=0.9981 12f (G_narrow_d May 20-21 Desktop refine winner)
- Bear = **8h@65%** RF+LGBM w=293 γ=0.9990 16f (G_narrow_d May 20-21 Desktop refine winner)
- Shields OFF (both regimes)
- **Rally cooldown ON** (both regimes — REVERTED 2026-05-27 13:18 to Mode T optimal recommendation, after May 23 22:21 manual-OFF intermezzo). Bull: rr8h≥2.0 OR rr14h≥6.0 cd=6h. Bear: rr10h≥5.5 OR rr12h≥2.0 cd=8h.
- min_sell_pnl=0%, max_hold=10h, max_position_usd=$14,300

**Asset universe**: ETH live; BTC/SOL/LINK/BNB `enabled: false`; XRP removed from trader data pipeline 2026-05-23 (silent-crash mitigation).

**Rollback ladder (one-command each, hot-reloads within 5 min):**

```powershell
# One level back — to G_fresh / H75-fresh promote state (live 2026-05-20 09:04 → 2026-05-21 21:56)
# Note: pre_G_narrow snapshot captures H75-fresh state exactly
copy config\regime_config_ed_pre_G_narrow_20260521.json    config\regime_config_ed.json
copy models\crypto_ed_production_pre_G_narrow_20260521.csv models\crypto_ed_production.csv

# Two levels back — to H75-snapshot (live 2026-05-18 22:02 → 2026-05-20 09:04)
copy config\regime_config_ed_pre_H75fresh_20260520.json    config\regime_config_ed.json
copy models\crypto_ed_production_pre_H75fresh_20260520.csv models\crypto_ed_production.csv

# Three levels back — to pre-H75 baseline (live before 2026-05-18)
copy config\regime_config_ed_pre_H75_20260518.json    config\regime_config_ed.json
copy models\crypto_ed_production_pre_H75_20260518.csv models\crypto_ed_production.csv
# Optional engine-layer rollback (only if reverting to the pre-H_STRICT_FAMILY engine):
# Note: pre-H75 snapshot archived 2026-05-28 — fetch from ARCHIVED/2026-05-28_v3_cleanup/
copy ARCHIVED\2026-05-28_v3_cleanup\crypto_trading_system_ed_pre_H75_20260518.py     crypto_trading_system_ed.py
```

**Promotion source**: Desktop G_narrow_d HRST run 2026-05-20 11:05 → 2026-05-21 10:28 (wall ~23h 22m, log `logs/ed_v1_20260520_110556.log`). Mode T REF +89.14% (converged iter 2, no STRICT rally-cooldown winner).

**Promotion timeline**:
1. 2026-05-18 22:02 — H75 promoted (snapshot: `pre_H75_20260518`)
2. 2026-05-20 09:04 — H75-fresh promoted (snapshot: `pre_H75fresh_20260520`)
3. **2026-05-21 21:56 — G_narrow promoted (current)** (snapshot: `pre_G_narrow_20260521`)
4. 2026-05-22 19:51 — G_fresh promoted (content-identical ETH rows; snapshot: `pre_G_fresh_20260522`)
5. 2026-05-23 22:21 — manual: rally_cooldown enabled → disabled
6. **2026-05-27 11:22 — engine: macro_cache mtime fix patched** (TODO 0527 root cause; snapshot: `crypto_trading_system_ed_pre_macro_cache_fix_20260527_112231.py`)
7. **2026-05-27 13:18 — manual: rally_cooldown disabled → enabled** (reverted to Mode T optimal; backup: `regime_config_ed_pre_rc_reenable_20260527_131848.json`)

Full promotion events in ARCHIVED_LOG.md.

---

# 🔥 P0 — Step 6 fork ready to test on Desktop (action items)

**Search anchor**: `STEP6-DESKTOP-RUN`

**Status**: Step 6a + 6b + 6c shipped in research fork `crypto_trading_system_ed_step6.py` (commit `50a63ab`, 2026-05-27). Smoke regression (1 grid point) confirmed BIT-IDENTICAL vs production engine. Wider regression + LIVE_EQUIVALENT_MODE diagnostic pending — should run on Desktop (faster).

**What Step 6 is**: refactor of `_deku_eval_with_pruning` to delegate to `crypto_signal_core.compute_signal_core()`. With default parameters, output is bit-identical to current engine (regression-safe). New params (`embargo`, `na_policy`, `signal_mode`, `return_probas`, `eval_step`) let callers opt into live-trader semantics. `LIVE_EQUIVALENT_MODE=1` env var overrides all 5 to live conventions.

**Why it matters**: lets HRST runs produce a REALISTIC backtest projection that should predict live performance. Current backtest math (`_deku_eval_with_pruning` original) uses different semantics from live trader (binary signals, step=36, embargo=horizon, NaN skip) → it overstates live WR. Step 6c diagnostic mode tells us by how much.

## Desktop run instructions

### Step 1 — Confirm Drive sync (~1 min)

Fresh PowerShell on Desktop, venv activated:

```powershell
cd G:\engine
Test-Path crypto_trading_system_ed_step6.py
Test-Path tools\test_step6_regression.py
```

Both should be True.

### Step 2 — Wider regression test (~10 min, builds confidence beyond the 1-point smoke)

```powershell
# Clear any leftover env vars
$env:V2_DATA_SNAPSHOT = ""
$env:LIVE_EQUIVALENT_MODE = ""

# Capture baseline with the ORIGINAL engine (12 grid points)
python tools\test_step6_regression.py --engine ed --save baseline_wide

# Compare fork against that baseline
python tools\test_step6_regression.py --engine step6 --compare baseline_wide
```

Expected output: `[OK] BIT-IDENTICAL` on all 12 grid points.

If any point diverges, paste output to assistant for debug. If all 12 match, Phase 6a/6b verified production-grade.

### Step 3 — LIVE_EQUIVALENT_MODE diagnostic (~10 min)

The payoff — see what the model's realistic backtest WR looks like with live-trader semantics:

```powershell
# Enable LIVE_EQUIVALENT_MODE — overrides all 4 backtest semantics to live
$env:LIVE_EQUIVALENT_MODE = "1"

# Run the same 12-point grid; the fork's _deku_eval_with_pruning will now
# use embargo=1, na_policy='ffill', signal_mode='ternary', eval_step=1
python tools\test_step6_regression.py --engine step6 --save liveeq_diag

# Disable so subsequent runs are normal
$env:LIVE_EQUIVALENT_MODE = ""
```

Paste per-grid-point cum_return / accuracy / trades back to assistant. Comparing against `baseline_wide`:
- LIVE_EQUIVALENT cum_return **dramatically lower** than baseline → confirms backtest math was overoptimistic, Step 6 thesis validated
- LIVE_EQUIVALENT cum_return **similar** → either (a) backtest math wasn't the dominant issue or (b) small per-feature drift was already dominating

### Step 4 — Full HRST in LIVE_EQUIVALENT_MODE (optional, ~7h, the real test)

If Steps 2+3 look good, run a full HRST with the fork and live-equivalent mode to get the REAL Mode T REF projection:

```powershell
$env:LIVE_EQUIVALENT_MODE = "1"
$env:V2_DATA_SNAPSHOT = "data\_reliability_hrst_snapshot_desktop_20260515_154801"

python crypto_trading_system_ed_step6.py HRST ETH 5,8h --replay 1440 --no-persist --no-data-update
```

This gives a Mode T REF number that should approximate what live performance actually delivers. Probably much lower than the current overoptimistic ~89%. That number becomes the **realistic target** — any future HRST should be evaluated against it instead of the current backtest WR.

### Decision after Steps 2+3 (+ optionally 4)

| Step 2 result | Step 3 result | What it means | Next action |
|---|---|---|---|
| All 12 BIT-IDENTICAL | LIVE_EQUIVALENT much lower than baseline | Refactor correct + backtest was overoptimistic | Promote fork to production engine. Future HRST runs use LIVE_EQUIVALENT_MODE. |
| All 12 BIT-IDENTICAL | LIVE_EQUIVALENT ≈ baseline | Refactor correct + backtest math wasn't the dominant gap source | Keep fork as research tool only. Look elsewhere for model improvements. |
| Any DIFF in Step 2 | — | Refactor broke bit-identical | Paste DIFF output to assistant; debug before doing Step 3 or 4. |

---

# 🔥 P0 — Live monitoring (passive, Desktop)

## ⚡ G_narrow LIVE OOS monitoring — RESTARTED 2026-05-27 post-fixes

**Search anchor**: `G_NARROW-OOS-MONITOR`

The original OOS window (2026-05-21 21:56 → ~2026-06-04) is effectively a different system now after today's fixes:
- `_macro_cache` mtime fix at 11:22 (TODO 0527 root cause)
- `_log_signal` schema rename to sig_1/sig_2 + 3 latent HORIZON_SHORT bugs fixed
- `download_fear_greed` safe-merge fix
- `rally_cooldown.enabled` reverted to True (Mode T optimal) at 13:18

**Pre-fix closed trades under broken-cache G_narrow** (now historical context, not directly informative of forward performance):

| # | Open | Close | Entry | Exit | PnL |
|---|---|---|---|---|---|
| 1 | 2026-05-22 18:22Z (manual BUY) | 2026-05-23 22:00 CEST (auto SELL) | $2120.96 | $2075.26 | −2.15% / −$298.81 |
| 2 | 2026-05-24 18:02Z (auto BUY) | 2026-05-24 22:00 CEST (auto SELL) | $2100.60 | $2098.32 | −0.11% / −$14.73 |
| 3 | 2026-05-27 (today, post-cache-fix) | 2026-05-27 same day | — | — | LOSS (user noted "felt healthier" qualitatively despite outcome) |

**Forward monitoring window** (starts 2026-05-27 13:18, the moment all fixes are in place):
- 3-5 days: first qualitative read on trader behavior
- 10 trades (~7-14 days): first WR/return signal vs sample-size noise
- 30+ trades (~3-4 weeks): statistically meaningful estimate

**Rollback triggers under the now-fixed setup** (any one fires → discuss with user):
- Cumulative realized alpha < +5% after first 10 trades
- MaxDD exceeds −10% in any 14-day window
- First 10 trades WR < 50%
- Trade count vastly different from counterfactual prediction (counterfactual showed ≈4 trades / 5 days = 0.8/day; if live is dramatically higher or lower, investigate)

**Decision tree after 2-4 weeks**:
| Outcome | Action |
|---|---|
| Holds OOS (≥0 triggers, WR trending above broken-cache 50.9%) | Continue. Document cache-fix impact as confirmed. |
| Underperforms (>1 trigger OR alpha < +5% after 10 trades) | Investigate: shadow log first (any DIFFs?), then execution layer (slippage, partial fills) |
| Borderline | Watch another 1-2 weeks; don't act on small samples |

---

## 🔥 Shadow mode continuous match-rate check (Desktop)

**Tripwire**: any drop below ~99% match in `config/shadow_signal_diff.csv` = NEW bug to investigate. Currently at 10/10 (100%) since 2026-05-27 fixes.

**Periodic query** (every 1-2 days):
```powershell
Import-Csv config\shadow_signal_diff.csv | Group-Object match | Select Count,Name
```

When the broader gate from Step 6 (engine↔trader shared core) is in place, shadow mode can be retired. Until then it's the primary live-correctness gate.

---

# 🔥 P1 — Re-evaluate disqualified items under LIVE_EQUIVALENT_MODE (CONDITIONAL on Step 6 verdict)

**Search anchor**: `STEP6-REEVAL`

**Status**: 📅 PLANNED, conditional on Step 6 LIVE_EQUIVALENT_MODE results showing a meaningful gap vs current backtest baseline. If gap is >10pp, prior "DEAD" / "SHELVED" verdicts in the ideas scoreboard may have been methodology artifacts and worth re-testing.

**Why this exists**: every "DEAD"/"SHELVED" verdict in CLAUDE.md / ARCHIVED_LOG.md was reached using the SAME `_deku_eval_with_pruning` math that Step 6 is exposing as overoptimistic. The 4 semantic divergences (embargo, NaN policy, step size, signal mode) hurt different candidates asymmetrically — meaning relative rankings under broken backtest may not predict relative rankings under live.

**Decision gate**: re-test only if Step 6's LIVE_EQUIVALENT_MODE shows cum_return materially below baseline. If LIVE_EQUIVALENT ≈ baseline, prior verdicts hold and this whole block is moot.

## Priority list (ordered by recovery potential under LIVE_EQUIVALENT_MODE)

### Priority 1 — NaN-sensitive (sparse-history features quarantined by 'skip' policy)

These were filtered out because backtest's `na_policy='skip'` dropped any row with NaN in their column. Under `na_policy='ffill'` (live), they'd contribute.

| Item | Reason originally quarantined | Re-test action |
|---|---|---|
| `deriv_oi_*` family | Sparse OI data (30-day history only) → most training rows had NaN → skip removed them | TODO 0519B-G1 already queued; now reframed under Step 6 lens |
| Orderbook (`ob_imbalance`, `spread_bps`) | Hourly snapshots, gaps | Part of TODO 0519B-G2 |
| IV (`avg_iv`, `iv_skew`) | Sparse data | Part of TODO 0519B-G2 |
| Stablecoin mcap (3 features) | Currently Grade 1 (all importance <1%) | Re-test with ffill |

### Priority 2 — Step-size sensitive (hourly-cadence ideas)

Backtest evaluates every 36h; live every 1h. Anything responsive to short-term changes was undersampled.

| Item | Re-test action |
|---|---|
| C11 VPIN at 5-min cadence | Re-test with `eval_step=1` |

### Priority 3 — Embargo-sensitive (short-window logrets)

Backtest uses `embargo=horizon` (5-8h gap before test); live uses 0. Recent-momentum features lose their freshest data.

| Item | Re-test action |
|---|---|
| `logret_2h`, `logret_5h`, `logret_8h` (currently Grade 1) | Re-test with `embargo=0` |
| Any `chg1d` feature | Re-test with `embargo=0` |

### Priority 4 — Signal-mode sensitive (HOLD-aware strategies)

Backtest forces BUY-or-not on every step; live ternary allows HOLD. High-confidence-only strategies were penalized.

| Item | Re-test action |
|---|---|
| C14 triple-barrier overlay | Re-test with `signal_mode='ternary'` |
| C15 meta-labeling | Re-test with `signal_mode='ternary'` |
| Asymmetric loss (`scale_pos_weight`) | Re-test with `signal_mode='ternary'` |
| C56/C57 regime detectors | Re-test with `signal_mode='ternary'` |

### Priority 5 — HYPERPARAMETER RE-GRID (the biggest recovery surface)

**This is bigger than the idea/feature re-tests.** The current production winners — `RF+LGBM w=281 γ=0.9981 12f` (5h) and `RF+LGBM w=293 γ=0.9990 16f` (8h) — were selected by the same broken `_deku_eval_with_pruning` math. EVERY hyperparameter chosen (combo, window, gamma, feature count) was scored under backtest semantics that don't predict live performance.

If LIVE_EQUIVALENT_MODE shows a meaningful gap, the "best config" rankings can shift entirely. Different combos may win. Different windows. Different gammas. Different feature counts.

| Hyperparameter | Current production grid | Expanded re-grid (live-equivalent) | Recovery hypothesis |
|---|---|---|---|
| **Model combos** | 3 viable: RF+LGBM, XGB+LGBM, RF+XGB | Add back: **LR**, **GB**, **single-model LGBM**, **RF+GB**, **RF+LR**, **GB+LR** | Solo models may work in ternary HOLD mode (less overfitting risk); LR may benefit from ffill giving more usable rows |
| **Windows** | [72, 100, 150, 200, 250, 300] | [30, 50, 72, 100, 150, 200, 250, 300, 350, 400, 500, 720] | Shorter windows benefit from `eval_step=1` (more recent regime weighting); longer windows benefit from less embargo cutting away signal |
| **Gammas** | [0.995, 0.997, 0.999] | [0.99, 0.995, 0.997, 0.998, 0.999, 0.9995, 1.0] | Different time-decay weights under live semantics; `gamma=1.0` (no decay) might suddenly be viable when step=1 makes recent data more informative |
| **Feature counts** | [5, 10, 15, 20, 25, 30] + Optuna range [4, 40-80] | [3, 5, 8, 10, 15, 20, 25, 30, 40, 60, 100] | With ffill, sparse features become viable → more features can pass selection; with ternary HOLD, fewer features may suffice |
| **MIN_COMBO_SIZE** | 2 (solo removed) | 1 (solo allowed) | Solo LGBM or solo XGB may dominate when the ternary HOLD semantics provide their own "abstain" risk filter |

**Compute estimate**: current grid = 324 configs × 1 horizon ≈ 30 min on Desktop. Expanded grid = ~5,000 configs × 2 horizons ≈ 5-7h on Desktop. Plus Mode V refine for top candidates: +3-4h.

**Plan** (conditional on Step 6 showing meaningful gap):
1. Expand `GRID_COMBOS`, `GRID_WINDOWS`, `GRID_GAMMAS`, `GRID_FEATURES`, `MIN_COMBO_SIZE` in fork
2. Run Mode D ETH 5h + 8h with `LIVE_EQUIVALENT_MODE=1` on expanded grid
3. Compare top-10 winners under live-equivalent vs current production
4. If a meaningfully better config emerges → promote (after appropriate validation)
5. If current production is still in the top-10 under live-equivalent → it was selected correctly despite the broken backtest math; no change

This is **the highest-value single experiment** after Step 6 validates.

### Priority 6 — Dead model combos (subset of P5 — broken out for clarity)

| Item | Re-test action |
|---|---|
| GB+LR, RF+GB, RF+LR combos (dropped from `GRID_COMBOS`) | Covered by P5 expanded re-grid |

### Priority 7 — Disabled feature families

| Item | Re-test action |
|---|---|
| GDELT 21 features (disabled 2026-04-19, 0/33 selection) | Re-enable temporarily, run Mode V live-equivalent |

### Priority 8 — Full IDEA QUEUE Tier A/B sweep

Only if Step 6 shows >15pp gap AND P5 expanded re-grid produces meaningfully different winners — then ALL prior verdicts are suspect and a sweep is justified.

## How re-testing will work

Once Step 6 verdict is in, if gap is meaningful:

1. **Build `tools/re_evaluate_dead_ideas.py`** — takes a config + identifier list, runs each under both `backtest` mode (legacy) and `LIVE_EQUIVALENT_MODE`, outputs side-by-side delta table.
2. **Stage candidates from Priority 1 first** — already-queued G1/G2 work becomes the proof-of-concept for the methodology.
3. **If P1 produces a flipped verdict** (item revives), expand to P2-P4.
4. **Update scoreboard** in ARCHIVED_LOG.md with new verdicts. Old verdicts kept as audit trail, new entries reference them.

**Budget cap**: stop after spending 3 days of compute on re-testing. If nothing flips, prior verdicts hold; if many flip, the engine has a fundamentally different ranking under live semantics and a full HRST re-run + retraining is warranted.

---

# 🔥 P1 — Act this week (Laptop preferred unless noted)

## 📝 Wider counterfactual backtest (Laptop, ~30 min, RUNNING NOW)

**Search anchor**: `COUNTERFACTUAL-WIDE`

Running as of 2026-05-27 afternoon via `python tools/afternoon_run.py`. Output expected at `output/afternoon_summary_<ts>.md`. Will produce bootstrap CIs on:
- WR / compound return / avg per trade (both conditions)
- Return delta (counterfactual − actual)
- P(counterfactual > actual)
- Weekly breakdown

Decision gate on completion:
- P(counterfactual > actual) > 80% → cache-fix value confirmed; ride forward live data with confidence
- 50% < P < 80% → direction looks right; wait for forward live
- P < 50% → unexpected; re-examine execution layer or another bug

---

## 🔥 Embargo A/B test (Laptop, ~2.5h)

**Search anchor**: `TODO-0526-EMBARGO-AB`

**Command**: `python tools/embargo_ab_test.py --mode=both`

Quantifies the contribution of embargo (divergence #1 of the original TODO 0526 "4 semantic differences") to the backtest-vs-live gap. Informs Step 6 refactor scoping. Not blocking anything in production. Run when Laptop has 2.5h of capacity.

---

## 🔥 TODO 0519B-G1 — `deriv_oi_*` re-enable A/B test (Desktop off-hours, ~6h)

**Search anchor**: `TODO 0519B-G1`

**Status**: pending; deferred multiple times. Newly relevant after the macro_cache fix because `deriv_oi_*` features (when re-enabled) will now actually vary across trader cycles instead of staying frozen at startup values. The decision criterion may swing differently than the pre-fix expectation.

**Procedure**: A/B compare Mode V refine output with `deriv_oi_*` features in vs out of the disabled-feature quarantine list. Detailed steps in `archive/disabled_features_pre_g1_<DATE>.json` backup procedure.

**Don't run while a P1 Laptop job is going** — wait for capacity.

---

# 📝 P2 — Pending design work (Laptop, ~1.5 calendar days + 12h compute)

## 📖 Step 6 engine refactor

**Search anchor**: `STEP-6-REFACTOR`

**Design doc**: [docs/STEP_6_ENGINE_REFACTOR.md](docs/STEP_6_ENGINE_REFACTOR.md)

**Goal**: make Mode V / Mode T backtest call the same `compute_signal_core()` that the live trader uses (after TODO 0527 fixes). After this, HRST results predict live performance. Expected outcome: backtest WR projections drop from ~85% (overoptimistic) to a realistic ~65-75%, but with the property that live performance should approach that realistic projection.

**Required before next promotion** — promoting on the current overoptimistic backtest will keep producing live-vs-backtest gaps.

**4 phases**:
- 6a: regression-safe refactor (bit-identical Mode D output)
- 6b: expose embargo/NaN-policy/signal-mode as explicit parameters
- 6c: live-equivalent diagnostic mode in the engine
- 6d: cross-validate against shadow data

---

## ⏸ Re-run HRST on refactored engine (Desktop, ~7h, BLOCKED on Step 6)

Once Step 6 ships, re-run the canonical HRST so the recommended config reflects realistic live expectations. Validation: Mode T REF should match counterfactual backtest within ±5pp.

---

# 📋 P2 — Scheduled (next month)

## 📅 TODO 0519B-G2 — orderbook + IV re-enable A/B test (2026-06-18, ~30 days)

**Search anchor**: `TODO 0519B-G2`

**Features**: `ob_imbalance`, `spread_bps`, `avg_iv`, `iv_skew`. ~60d coverage from 2026-04-19 live trader snapshot writes.

**Procedure**: identical to G1 with these substitutions:
- Backup filename: `disabled_features_pre_ob_iv_20260618.json`
- Log filenames: `g2_ob_iv_OFF_*.log` / `g2_ob_iv_ON_*.log`
- 4 features to remove from `always_disabled_exact` (not 3)

Full G2 details: [ARCHIVED_LOG.md:3553](ARCHIVED_LOG.md).

**Don't auto-run on the date** — decide based on G1 outcome. If G1 ships → prior on G2 up. If G1 fails (matches "feature-add family exhausted" pattern) → consider skipping G2.

---

# 🚀 P3 — Research backlog (when capacity allows)

## P-Queue items (from 2026-05-10 priority list — kept open, lower than P1/P2)

### P4 — C14 vol-conditional triple-barrier retest (~2.5h overlay sim)

**What**: Only apply triple-barrier exit when realized vol > p70. Original C14 SHELVED 2026-04-26 (+10.48pp 60d but +1.24pp 90d — gain didn't survive). Hypothesis: barriers help only in high-vol regimes; current 60d-test averaged across vol regimes diluted the win.
**How**: Standalone overlay sim script reusing `data/eth_per_horizon_signals_90d.pkl` + realized-vol percentile filter on entries.
**Decision**: if vol-conditional gain ≥ +5pp on 90d → engine integration. Within ±5pp → SHELVE final. Worse → kill the angle.

### P5 — C11 VPIN at 5-min cadence (~1 day engineering)

**What**: Move VPIN entry filter from hourly to 5-min sub-loop in trader. Original C11 SHELVED 2026-05-03 (+3.83pp on 60d, below +5pp ship). Literature uses 1-min cadence — hourly was too slow.
**How**: Real engineering — needs 5-min OHLCV download in `download_macro_data.py`, sub-loop in `crypto_revolut_ed_v2.py`, threshold sweep.
**Decision**: not actionable until someone has a clear day for engineering. Lower priority than P4 because higher effort × similar expected payoff.

### P6 — C15 meta-labeling on SOL/BTC (~6h, blocked on assets shelved)

**What**: Retest meta-labeling on SOL/BTC primaries (current production is ETH-only). Original C15 SHELVED 2026-04-27 (lost on strong ETH primary by −2.12pp). Door explicitly left open for weaker-primary assets per CLAUDE.md.
**Blocker**: SOL/BTC/XRP/LINK all `enabled: false` in `regime_config_ed.json` for diversification/correlation reasons. Re-test requires re-enabling at least one asset.
**Decision**: deprioritized — only relevant if (a) ETH live performance forces asset diversification, OR (b) cross-asset thesis revives.

---

## 🚀 IDEA QUEUE — Tier A: Untested clean (5 items)

| CID | Idea | Effort |
|---|---|---|
| **C13-narrow** | Single-horizon CDaR variant (no regime split) | ~2h |
| **C54** | Time-decay sell threshold | ~1h |
| **C55** | Liquidity-aware entry timing | ~1.5h |
| **C58** | Yield-curve macro regime detector (depends on C41) | ~2h |
| **C59** | K-means cluster regime (multi-dim macro+vol) | ~3h |

C13-narrow has positive prior (C13's 8h Refined #1 +67.03% was strongest CDaR result). C54/C55 are execution-side, distinct from feature-add family. C58/C59 are regime-detection — prior LOW (C56 HMM DEAD Δ−0.93, C57 MS-AR FAIL Δ−1.574).

## 🚀 IDEA QUEUE — Tier B: V3-lit archive-recovered (23 ideas, 16 with ready patchers)

Pulled into the scoreboard 2026-05-04 from `archive/literature_v3_ideas.md`. All 23 still untested.

**Patcher-ready (16)** — already exist in `_idea_patchers/C*_v3lit.py`, launchable via existing harness:
C60, C61, C62, C63, C64, C65, C67, C69, C70, C71, C73, C75, C79, C80, C81, C82

**Patcher-missing (7)** — need writing first:
C66, C68, C72, C74, C76, C77, C78

**Top 5 cheap patcher-ready picks** (by effort × V3 priority):

| CID | Idea | Effort | V3 # |
|---|---|---|---|
| **C62** | DXY Acceleration (2nd derivative of DXY) | ~30 min | #6 |
| **C60** | US Market Hours Flag (binary NYSE 14:30-21:00 UTC) | ~1h | #3 |
| **C61** | Volatility of Volatility | ~1h | #5 |
| **C63** | KAMA Slope (Kaufman Adaptive MA) | ~1.5h | #7 |
| **C64** | Ehlers Fisher Transform | ~1.5h | #8 |

Full C60-C82 list: [ARCHIVED_LOG.md "ARCHIVE-RECOVERED IDEAS C60-C82"](ARCHIVED_LOG.md).

**Pattern caveat**: feature-add family has consistently failed (C32-C40 batch 0 PASS / 1 FAIL / 6 MARGINAL; C03/C12/C23/C29b/C31/C35/C42/C44/C47/C56/C57 all DEAD or marginal). C60-C82 are mostly feature-adds — prior LOW. Allocate ≤1h per first attempt; if smoke shows MARGINAL like the others, the family ceiling is real.

## 🚀 IDEA QUEUE — What was dropped (see [ARCHIVED_LOG.md "IDEA QUEUE drop-list"](ARCHIVED_LOG.md))

18 ideas that lived briefly on the IDEA QUEUE have been closed with verdicts. Curated drop-list lives in ARCHIVED_LOG.md as a quick-lookup section. One-line summary:
- **6 Tier 1 ideas** (C35, C42, C43, C44, C47, C57) — DEAD on 2026-05-07 fixed-harness retest
- **5 Tier 2 ideas** (C03, C12, C23, C29, C31) — DEAD on 2026-05-10 batch
- **4 Tier 3 ideas** (C16-narrow shipped, C48/C52/C53 DEAD)
- **3 STUB-blocked** (C45/C46/C49 — architectural prerequisites)

For revival, check the verdict + re-add to TODO.md only if evidence overturns the closure.

---

# ⚪ P4 — Low priority / Diagnostics / Backlog

## ⚪ TODO 0519C — CPCV HRST diagnostic (trigger-based re-run)

**Search anchor**: `TODO 0519C`

**Status**: ⚪ LOW PRIORITY. Tested 2026-05-11 → matched current method (no Mode T re-rank, no headline win). Kept because the PBO diagnostic remains useful intel — periodic re-runs would catch if a future engine change introduces overfit configs that current 3-fold rolling holdout misses.

**Trigger to re-run** (any of):
- Major engine architecture change (like H75 → H_STRICT_FAMILY merge 2026-05-18) — re-run on new top-6 candidates to verify the new arch isn't producing overfit configs
- Suspicious Mode T win on a new HRST (>+15pp over current production) — use PBO as overfit sanity check before promoting
- Quarterly hygiene check (~2026-08 next)

**Run command** (resumable, single-instance lock):
```powershell
python tools/run_cpcv_hrst_resumable.py
```

ETA ~5-7h Desktop. Engine fork `crypto_trading_system_ed_cpcv.py` + launcher already in place.

Full closure: [ARCHIVED_LOG.md "Closed 2026-05-11 — P3 CPCV HRST"](ARCHIVED_LOG.md).

## ⚪ Kalshi prediction-market integration (backlog — needs API key + impl)

**Source**: [download_macro_data.py:1501](download_macro_data.py#L1501) — `# TODO: implement when API key available`

**What**: download crypto-related prediction market data from Kalshi (https://kalshi.com/). Currently a stub: function exists, exits early if no `KALSHI_API_KEY` env var or `config/kalshi_config.json`. Implementation never written.

**Why low priority**:
- Requires user to register for Kalshi API access
- New macro feature — same family as GDELT (DEAD), stablecoin mcap (DEAD), C32-C40 batch (mostly MARGINAL/FAIL)
- Feature-add ceiling on this engine is at-or-near zero per the 2026-05-09 retro
- Not actionable until both (a) API key obtained, (b) someone has time to write `download_kalshi_data()`

**Trigger to action**: only if user obtains API key AND has a specific hypothesis about prediction-market data beating VIX/equity-1d-change for macro fear signal.

---

**Honest expectation across the backlog**: per the 2026-05-09 batch retro + 2026-05-19 audit, the H75 production engine is at-or-near its alpha ceiling from feature/scoring tweaks. **8 of 11 originally-queued ideas DEAD on fixed harness**. Future meaningful gains likely come from **execution-gap research** (~17pp untouched alpha per ARCHIVED_LOG.md:1060). C54/C55 in Tier A and P5 VPIN-5min are the only execution-side candidates currently scoped.
