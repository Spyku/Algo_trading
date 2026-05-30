# FAYE Research Log — 2026-05-30

**Purpose**: comprehensive log of today's FAYE consolidation, bug hunt, and audit session. Built so you can pick this up on Laptop without re-reading the whole conversation.

---

## STATUS SNAPSHOT (as of writing)

| Item | State |
|---|---|
| FAYE module | `crypto_trading_system_faye.py` — 14 bugs fixed today (all pushed to main) |
| Currently running on Desktop | `python crypto_trading_system_faye.py H ETH 7h --replay 528` |
| Run phase | Mode V Step 2 (Optuna refine, 3 workers parallel) |
| Refine status | 3 workers alive, CPU growing, asymmetric (PID 17836 fastest, 61400 ~2.7× slower) |
| ETA | 30-60 min remaining for refine; then Mode V Step 3 (~15 min); H 7h done in ~1h |
| Trader | OFF (no live signal generation since May 29 17:09) |
| Live production config | UNCHANGED — G_narrow on H75 engine (May 21 promotion still active) |
| v3 chain on Desktop | KILLED earlier today (had orphan-worker bug); not relaunched |

---

## WHAT IS FAYE & WHY

**FAYE** = single-file consolidation of the Ed v3 architecture.

v3 production stack was a 4-layer monkey-patch chain:
```
v3 → patches → parallel_nearlive → patches → step6_nearlive → patches → g_narrow_d → patches → ed
```

Each layer patched the next via `os.execv` warning sentinels, `ENGINE.<func> = new_func` rebinds, `inspect.currentframe()` state capture, and `_H_ORIG_DEKU_EVAL` chains. The result worked but was fragile (orphan workers, hard to audit, monkey-patches in 4 files).

**FAYE consolidates everything into one ~9700-line file with ZERO monkey-patches.** Every previously-patched feature is first-class native code at its natural location.

Shipped 2026-05-29 in 7 phases (commits `8c122ef` → `4ab34d5`):
- P1 — Identity rebrand (paths, banners, isolation)
- P2 — `_mean_last_10_fill` + NEAR_LIVE defaults inlined
- P3 — K=5 multi-seed median ensemble native (no `_H_ORIG_DEKU_EVAL` chain)
- P4 — 3-worker hybrid GPU+CPU refine inlined
- P5a — `run_mode_v/s/t_parallel` → canonical names
- P5b — 8-worker Mode D dispatcher inlined (replaces `_v3_routed_deku_eval` + `_ParallelGridDispatcher`)
- P6 — `tools/smoke_test_faye.py` (38-check verification)
- P7 — Architecture documentation block

After Phase 7 the smoke test passed, but **the smoke test only verified structure (function existence, monkey-patch removal), not runtime behavior.** That's how today's bugs survived.

---

## THE 14 BUGS FOUND TODAY (2026-05-30)

All commits pushed to `main` branch, repo `Spyku/Algo_trading`.

### Tier 1: Caused FAYE to fail or hang outright

| # | Commit | Bug | Symptom |
|---|---|---|---|
| 1 | `23b54b3` | `GRID_WINDOWS=[72,100,150]` instead of g_narrow_d's `[72,100,150,200,250]` | Optuna refine never reaches w=281/293 basin where live winners sit — FAYE produces uncompetitive winners every time |
| 4 | `1cee360` | `_get_deku_diagnostic_models_seeded` hardcoded `device='gpu'` instead of reading `G_PARALLEL_LGBM_DEVICE` env (default 'cpu') | 8 ProcessPool workers × K=5 inner = 40 concurrent LGBM-GPU calls on one RTX 4080 queue → workers deadlocked, never produced output. THIS WAS THE INITIAL HANG. |
| 6 | `3027255` | Refine dispatcher: 2 workers (GPU + CPU) with dynamic-3rd vs v3's 3 workers all-CPU when K>1 | GPU worker + K=5 inner threads would crash CUDA single-context (~4h into Mode V) |
| 7 | `a65d618` | `_deku_eval_with_pruning` (K=5 wrap) ran K=5 seeds in SERIAL for-loop vs `_parallel_deku_eval_median_k`'s ThreadPoolExecutor | **5× slowdown in Mode V** (refine + holdout) |
| 9 | `ae056b7` | `_get_deku_diagnostic_models_seeded` did not set LGBM `num_threads` — defaulted to physical core count (24) | 8 workers × K=5 × 24 = 960 concurrent OS threads on 24 cores = catastrophic oversubscription → 12 min/eval observed (vs v3's 5-7 min/eval) |
| 14 | `8776f38` | `_alive(pid)` used `os.kill(pid, 0)` on Windows. Python's `os.kill` on Windows maps signal 0 to `CTRL_C_EVENT` (Windows defines `CTRL_C_EVENT = 0`) → sends Ctrl+C to entire console process group | Each polling iteration of `_hard_shutdown_pool` fired Ctrl+C at the parent process → `KeyboardInterrupt` raised in `_time.sleep(0.5)` → script died at end of Mode D. **MY OWN DEFENSIVE CODE KILLED THE RUN.** |

### Tier 2: Caused incorrect output or partial failure

| # | Commit | Bug | Impact |
|---|---|---|---|
| 8 | `ae056b7` | `models_faye/` directory not auto-created on import | Per-eval CSV flush failed silently with `Cannot save file into a non-existent directory` |
| 10 | `d259efe` | `N_FEATURES_RANGE = (4,40)/(4,80)` (engine pre-patch) instead of g_narrow_d's `(4,100)/(4,100)` | Optuna refine's n_features upper bound capped too low, blocking exploration of 80-100 feature subsets. v3 comment: *"the cap was creating the B-7h tied-APF trap"* |
| 12 | `5382740` | **HOLD branch diverged from step6_nearlive**: FAYE incremented `total` on HOLD (inflated accuracy denominator) and didn't tick drawdown (under-counted max_dd). step6_nearlive lines 4193-4201 does NEITHER | FAYE accuracy artificially LOWER (1/N per HOLD) and max_dd artificially LOWER. **Numbers diverge from v3 for same model on same data.** |

### Tier 3: Defensive / UX / Quality

| # | Commit | Bug | Impact |
|---|---|---|---|
| 2 | `2aea8be` | Warning suppression had only Layers 2+3, missing Layers 1 (`-W ignore` re-exec) + 4 (`warnings.warn=no-op`) | Thousands of `UserWarning: sklearn.utils.parallel.delayed` warnings flooded stdout from workers |
| 3 | `f1b49f4` | `NEAR_LIVE_MODE` env var not set at module init — FAYE has defaults baked in but any importer checking env var saw None | Defensive — no observed symptom |
| 5 | `3027255` | Refine worker did not broadcast `lgbm_device` arg to K=5 factories via `os.environ['G_PARALLEL_LGBM_DEVICE']` | Hybrid GPU/CPU device routing was inert |
| 11 | `d259efe` | Regime config not seeded from live — parallel_nearlive lines 184-209 seeds isolated CONFIG_DIR | Mode H doesn't hit this. Future HRST would have crashed at Mode R with FileNotFoundError |
| 13 | `28227af` | `_hard_shutdown_pool` had 25s silent wait after Mode D end (no log line). User Ctrl+C'd thinking it was hung | Lost ~23 min of Mode D compute. Now prints `[<label>] phase done, waiting up to 25s for N worker(s) to exit cleanly...` then `all workers exited cleanly` |

### Plus: replay-too-short warning (commit `e9790c4`)

When `--replay` is too small for the grid windows (n < smallest_window + 100), Mode D produces all FAILED. Added a CRITICAL warning to console at fold-1 partitioning step. Note: engine has its own 500-row safeguard upstream that fires earlier — minimum viable `--replay` ≈ 528.

---

## AUDIT METHODOLOGY — WHAT WENT WRONG TODAY

I delegated to Explore agents asking "does FAYE match the v3-chain original?" Multiple agents reported "NO NEW BUGS FOUND" — but they compared FAYE to INTERMEDIATE LAYERS instead of the RUNTIME STATE.

Example: bug #4 (GPU device hardcoded):
- `g_narrow_d._g_factories_seeded` has `device='gpu'` hardcoded
- FAYE's `_get_deku_diagnostic_models_seeded` also has `device='gpu'` hardcoded
- Audit agent: "FAYE matches g_narrow_d" ✓ → marked OK
- **WRONG REFERENCE**: in v3 runtime, `parallel_nearlive` patches `G._g_factories_seeded = _device_aware_factories_seeded` which uses `os.environ.get('G_PARALLEL_LGBM_DEVICE', 'cpu')`. The "right" reference for FAYE is the POST-PATCH state, not any individual source file.

**Lesson for future audits**: when reviewing a consolidation of a monkey-patch chain, the reference is the RUNTIME STATE after all patches apply, not any single layer. Audits must trace patches to the final layer that "wins" at module load.

### Tools that worked

- `diff <(sed -n 'A,Bp' file1) <(sed -n 'X,Yp' file2)` — line-by-line for function-level comparison
- Direct read of parallel_nearlive end-to-end (~670 lines) + cataloging every `ENGINE.<x> = ...` patch + verifying FAYE has the post-patch value
- Comparing `_PNL_IS_MAIN_PROCESS` style guards across all monkey-patch parents

### Functions verified byte-identical (post line-by-line diff)
- `generate_signals` ✓
- `_backtest_one_config_worker` ✓
- `_run_parallel_backtests` ✓
- `_predict_signal_calls_for_horizons` ✓
- `_signal_gen_worker` ✓
- `_build_signals_cache_parallel` ✓
- `_key_for_call` ✓
- `_finish_mode_v` ✓
- `_compute_optuna_score` ✓
- `_diversity_key` ✓ (matches engine post-H_STRICT_FAMILY, intentionally NOT g_narrow_d's)
- `run_mode_v` ✓ (matches parallel_nearlive's `run_mode_v_parallel`)
- `run_mode_s` ✓ (matches `run_mode_s_parallel`)
- `run_mode_t` ✓ (matches `run_mode_t_parallel`)
- `_deku_eval_with_pruning_inner` ✓ (matches step6_nearlive's `_deku_eval_with_pruning` after bug #12 fix)
- `_mean_last_10_fill` ✓
- `run_mode_p` ✓
- `_sweep_rally_cooldown` ✓ (only `regime_config_ed.json` → `regime_config_faye.json` diff, intentional)
- `run_mode_g` ✓ (1-line diff: FAYE has 'RST' in VALID_MODES, others don't)

### NOT audited end-to-end yet
- Full Mode R / Mode T iteration logic (only verified module-level differences)
- PySR mode interactions
- HRST chain transitions between modes

---

## ENGINE-vs-TRADER PARITY (FROM YESTERDAY)

Yesterday I ran `tools/validate_core_against_signal_log.py --samples 30 --recent-only --cpu-lgbm` comparing `compute_signal_core()` against `signal_log.csv` entries:

- 30/30 ETH hours evaluated, 0 errors, 0 crashes
- **22/30 direct MATCH (73.3%)**, 8/30 DIFF (26.7%)
- Avg confidence delta: −1.98 (near-zero bias)
- **0/8 real BUY↔SELL flips** — all 8 DIFFs were HOLD-threshold boundary cases:
  - 5/8: live=HOLD (trader's `min_confidence` 95% for bear filtered it out) vs core=BUY/SELL (raw model direction)
  - 3/8: core=HOLD (engine probability < 50%) vs live=BUY/SELL (trader hit threshold)

**Verdict**: the engine and trader codepaths agree on direction every single time both produce one. The original "bug between live trader and crypto trading" — engine computing one signal, trader computing a different one for the same hour — is gone.

This validates that `compute_signal_core()` behaves identically in both backtest and live contexts (post macro_cache mtime fix from 2026-05-27).

Output: `output/core_validation_20260530_015454.csv`.

---

## TEST PLAN — CURRENT EXECUTION

### Step 1: H 7h single horizon (running now)
```
python crypto_trading_system_faye.py H ETH 7h --replay 528
```
Expected output: 1 row added to `models_faye/crypto_faye_production.csv` with `coin=ETH, horizon=7`.

### Step 2: Copy 7h row → fake 8h row (after H completes)
Python recipe (small script I'll write when you ask):
```python
import pandas as pd
df = pd.read_csv('models_faye/crypto_faye_production.csv')
row_7h = df[(df['coin']=='ETH') & (df['horizon']==7)].iloc[0].to_dict()
row_8h = row_7h.copy()
row_8h['horizon'] = 8
row_8h['gamma'] = row_7h['gamma'] - 0.0005  # small tweak so not byte-identical
df_new = pd.concat([df, pd.DataFrame([row_8h])], ignore_index=True)
df_new.to_csv('models_faye/crypto_faye_production.csv', index=False)
```

### Step 3: RST with the synthetic 2-horizon production CSV
```
python crypto_trading_system_faye.py RST ETH 7h,8h --replay 528
```
Tests Mode R + Mode S + Mode T (signal cache + regime sweep + T↔G convergence). No Mode D/V re-runs. Estimated ~40-60 min.

**Why this test matters**: Mode H already validates D + V end-to-end. RST exercises the OTHER half of the pipeline (regime detection, strategy selection, rally-cooldown sweep). Together they validate every mode except Mode P (PySR feature discovery, which is offline and rarely re-run).

If both H and RST complete cleanly, FAYE is end-to-end validated and you can launch full HRST at `--replay 1440` with confidence.

---

## COMMANDS CHEAT SHEET

```powershell
# Kill all python processes (e.g., before relaunch)
taskkill /F /IM python.exe /T 2>$null

# Verify nothing left
Get-Process python* -ErrorAction SilentlyContinue

# Check worker CPU growth (run twice with 30 sec between)
Get-Process python* | Select-Object Id, CPU, WorkingSet64 | Sort-Object CPU -Descending | Format-Table -AutoSize

# Smoke test FAYE structure (38 checks, ~2 sec, read-only)
python tools/smoke_test_faye.py

# Engine-vs-trader parity test (uses LIVE config from models/crypto_ed_production.csv)
python tools/validate_core_against_signal_log.py --samples 30 --recent-only --cpu-lgbm

# Launch FAYE Mode H (one or more horizons)
python crypto_trading_system_faye.py H ETH 7h --replay 528          # single horizon, fast
python crypto_trading_system_faye.py H ETH 7h,8h --replay 1440       # both horizons, full

# Launch FAYE RST chain (R + S + T, no D/V re-train)
python crypto_trading_system_faye.py RST ETH 7h,8h --replay 528

# Launch full HRST (D + V + R + S + T per horizon)
python crypto_trading_system_faye.py HRST ETH 5h,6h,7h,8h --replay 1440

# Short refine override (for fast iteration tests)
python crypto_trading_system_faye.py H ETH 7h --replay 528 --refine-trials 5

# --no-persist (research mode, writes to *_noprod.csv files inside models_faye/)
python crypto_trading_system_faye.py H ETH 7h --replay 528 --no-persist
```

---

## KNOWN ISSUES & PENDING WORK

### Issue: Refine workers capture stdout silently
**Symptom**: 1-2h of console silence during Mode V Step 2 (Optuna refine).

**Cause**: `_refine_one_config_worker` (FAYE line 9285-9286) redirects worker stdout to `io.StringIO()` buffer. Parent only prints captured stdout when each worker's future returns via `done.result()`. Until the slowest of 3 workers finishes, parent shows nothing.

**Workaround**: monitor `Get-Process python*` for CPU growth. If 3 workers' CPU keeps climbing, work is happening. If flat → hung.

**Fix candidate (future)**: stream stdout via a multiprocessing.Queue or write captured output to a per-worker temp file that parent tails.

### Issue: Worker speed asymmetry (2.7× spread observed today)
**Symptom**: 3 refine workers complete at very different speeds (PID 17836 at 4h CPU vs 61400 at 1.5h CPU after same wall time).

**Possible causes** (all unverified):
- Different (combo, window, n_features) configs assigned per worker. Bigger window → more walk-forward iterations.
- Optuna TPE sampler converges faster on simpler configs
- GIL contention specific to certain LGBM operations
- Memory layout / cache effects

**Research idea**: instrument workers to log per-trial time, then analyze whether the spread tracks config complexity.

### Issue: `--replay` < 528 fails with "Need 500+" engine safeguard
**Engine has a 500-clean-row minimum check before Mode D starts.** My replay-warning code (commit `e9790c4`) checks fold-1 against grid windows but never gets to run because engine aborts upstream.

**Minimum viable `--replay`**: ~528 (~22 days, accounting for ~5% dropna loss to clear 500 clean rows).

### Issue: 8-worker Mode D dispatcher's grid CSV path
Output goes to `models_faye/mode_d_full_<asset>_<H>h_<ts>.csv` (timestamped each run). Older CSVs accumulate. Currently no cleanup logic. **Not a bug, just FYI**: large research runs may want to occasionally clean old timestamped CSVs.

### Issue: NEAR_LIVE_MODE env override behavior
step6_nearlive's `_deku_eval_with_pruning` reads `os.environ.get('NEAR_LIVE_MODE')` and overrides function args if `'1'`. FAYE's `_deku_eval_with_pruning_inner` does NOT — uses default args (which are NEAR_LIVE constants).

**Effect**: cannot turn OFF NEAR_LIVE in FAYE via env var. Would need to call inner with explicit `signal_mode='binary'`, `na_policy='ffill'` etc. **Functionally equivalent for v3-matching runs (always NEAR_LIVE=1)** but limits testing flexibility.

### Issue: PARALLEL_K_WORKERS env override not supported
parallel_nearlive supports `PARALLEL_K_WORKERS` env var to override K=5. FAYE only reads `RELIABILITY_K`. Minor feature gap; user always wants K=5 in practice.

### Issue: Worker `_refine_one_config_worker` does not set G_PARALLEL_LGBM_THREADS in env
Parallel_nearlive line 452-453 sets it; FAYE just sets G_PARALLEL_LGBM_DEVICE. The seeded factory defaults to '1' anyway so this is functionally equivalent in current usage.

---

## CRITICAL PITFALLS — DO NOT REPEAT

### 1. `os.kill(pid, 0)` is NOT a liveness probe on Windows
Python's `os.kill` on Windows maps signal 0 to `CTRL_C_EVENT = 0` → sends Ctrl+C to console process group. **Use this on Windows and you'll kill your own parent process.** Use `OpenProcess`+`GetExitCodeProcess` instead (FAYE's `_alive` function now does this on `os.name == 'nt'`).

### 2. `pool.shutdown(wait=True)` can hang indefinitely
Workers blocked in native LGBM C code ignore Python's shutdown signals. The polite shutdown never returns. **Use `_hard_shutdown_pool`** (FAYE has it) that snapshots PIDs, polls liveness, SIGTERMs, then `taskkill /F /T`.

### 3. Trust CPU growth, not CPU percent
"43% CPU" stable for 10 min could mean steady work OR could mean only 1-2 of 3 workers active. Always sample CPU time counter across multiple snapshots and compute Δ. Process is hung only when Δ = 0.

### 4. ProcessPool workers capture stdout
Refine workers redirect stdout to local buffer. Console will be silent for 1-2h. This is NOT a hang — verify via Get-Process. Add UX progress logging if you want visibility.

### 5. `--replay` minimum is 528 (engine 500-row safeguard)
Smaller values abort cleanly with "Not enough data" but you still need to relaunch.

### 6. Smoke tests verify structure, not behavior
`tools/smoke_test_faye.py` checks that all canonical entry points exist and monkey-patch names are absent. It does NOT verify that the inlined functions produce the same OUTPUT as the v3 chain. Behavioral verification requires running actual workloads and comparing results.

### 7. Audit reference is RUNTIME state, not source files
When auditing FAYE (a consolidation), the question is "does FAYE behave like v3 at RUNTIME (after all monkey-patches apply)?", not "does FAYE match any single source file?" g_narrow_d's `_g_factories_seeded` and parallel_nearlive's `_device_aware_factories_seeded` are different — at runtime, the parallel_nearlive version wins (because it patches g_narrow_d's). FAYE must match the WINNING patch's behavior.

### 8. Trader is OFF, do not relaunch it during FAYE testing
Live trader's last signal was May 29 17:09. Position state = `cash`. **DO NOT start the trader while FAYE is running**, especially while Mode V refine is active — that's 15+ concurrent LGBM threads, and the trader hourly cycle would compete for CPU + may write to `models/crypto_ed_production.csv` while we're testing FAYE's own outputs in `models_faye/`.

---

## PRODUCTION STATE (UNCHANGED FROM YESTERDAY)

| Asset | Horizon | Config | Source | Notes |
|---|---|---|---|---|
| ETH | 5h | RF+LGBM w=281 γ=0.9981 12f +72.16% Refined | G_narrow May 21 | **Inflated by buggy backtest (`step=36`, binary signal, `na_policy='skip'`)**. NEAR_LIVE honest estimate would be lower. |
| ETH | 6h | RF+LGBM w=137 γ=0.9973 18f +47.19% Refined | G_narrow May 21 | Same caveat |
| ETH | 7h | RF+LGBM w=144 γ=0.9975 8f +29.24% Refined | G_narrow May 21 | Same caveat |
| ETH | 8h | RF+LGBM w=293 γ=0.9990 16f +45.87% Refined | G_narrow May 21 | Same caveat |

**Why not promote v3's 5h winner (RF+LGBM w=200 g=0.999 10f +49.56% Grid):**
- Lower headline return (+49.56% vs +72.16%) but **honest** (NEAR_LIVE backtest matches what trader actually does)
- Should be promoted per user's decision yesterday, but `tools/promote_v3_5h_winner.py` was never run
- v3's 5h winner is sitting in `models_g_desktop_nearlive/crypto_ed_production.csv`

**When FAYE H 7h finishes today**: comparable. Both v3 5h and FAYE 7h are NEAR_LIVE honest estimates. Pick whichever has higher Mode V Step 3 return for promotion.

---

## RESEARCH IDEAS FOR LAPTOP

(Roughly ordered by value)

### 1. Profile refine worker speed asymmetry
**Question**: why does worker 61400 run 2.7× slower than 17836 in the same refine batch?

**Approach**:
- Add per-trial logging to `_refine_one_config_worker` (write to per-worker temp file)
- Run H ETH 7h several times with `--refine-trials 5` (fast)
- Compare per-trial time across workers and configs
- Hypothesis to test: time ∝ window × n_features × walk-forward-iterations

**Output**: a CSV of (worker_pid, trial_idx, config_hash, time_sec) across multiple runs.

### 2. Reproducibility check across runs
**Question**: do two FAYE H runs on the same data + same code produce the same winner?

**Approach**:
- Run FAYE H ETH 7h --replay 528 twice back-to-back
- Compare `models_faye/crypto_faye_production.csv` rows
- Compare `models_faye/mode_d_full_ETH_7h_<ts>.csv` per-eval rows
- If different: identify the source of nondeterminism (Optuna seed? K=5 thread interleaving? sklearn random_state?)

**Why this matters**: if FAYE is nondeterministic, A/B comparisons across experiments are unreliable.

### 3. NEAR_LIVE counterfactual: ffill vs mean_last_10 on actual trader hours
**Already in TODO.md as P3** — bypass engine's auto-ffill, build features WITH NaN intact for each hour, call `compute_signal_core` with both `na_policy='ffill'` and `na_policy='mean_last_10'`, run trader's actual model, diff predictions.

**Expected**: 1-3% of signals differ between policies. Real economic impact estimated ±0.5-1.5pp/month.

### 4. Per-eval CSV mining
`models_faye/mode_d_full_ETH_*.csv` has K=5 seed-by-seed breakdown. Research:
- Is there a config (window, n_features, gamma) where the K=5 seeds disagree significantly?
- Median vs mean across seeds — does the choice matter?
- Variance across seeds vs config "complexity"

### 5. Optuna pruning impact
**Current**: K=5 wrap passes `trial=None` to all 5 seeds, disabling Optuna's Hyperband pruning. v3 does the same.

**Question**: would pruning the first seed (and using its intermediate APF) speed up refine without quality loss?

**Approach**: A/B with pruning ON vs OFF on a few short refine runs.

### 6. FAYE-vs-v3 numerical diff on the same data
Run FAYE H 7h --replay 1440 and v3 H 7h --replay 1440 on the SAME data snapshot. Compare:
- Mode D grid CSVs
- Mode V Step 1 backtest returns
- Mode V Step 2 refine winners
- Final production CSV row

**Expected**: numerically identical (or within seed-jitter floor of 1-2pp). If significantly different, FAYE has more bugs we haven't found.

**Setup**: `V2_DATA_SNAPSHOT` env var locks the data path. Use `data/_reliability_hrst_snapshot_desktop_<ts>` for reproducibility.

### 7. Investigate 8-worker Mode D ProcessPool spawn time
Windows multiprocessing.spawn takes 5-15 sec per worker. 8 workers = 40-120 sec just to spawn. Profile this — is it dominated by FAYE module import time (which now does os.execv re-exec)?

### 8. Smoke test extensions
`tools/smoke_test_faye.py` currently has 38 checks for structure. Add behavioral checks:
- Run `_deku_eval_with_pruning_inner` on synthetic data, verify return shape
- Run `_mean_last_10_fill` on known input, verify output matches expected
- Run `_get_deku_diagnostic_models_seeded(42)` twice, verify same factory dict

---

## OPEN QUESTIONS

- **Why does worker 61400 use only 238MB RAM vs 365MB for others?** Different config → fewer features → smaller model? Or different Optuna trial history size? Worth investigating during research #1.
- **Does FAYE's `_hard_shutdown_pool` produce different shutdown timings on Laptop vs Desktop?** 25s polling window is fixed; if Laptop workers take longer to respond, the SIGTERM phase fires more often. Not necessarily a bug but worth measuring.
- **The "Need 500+" check** — where exactly does it live in FAYE? I haven't traced it. Worth finding so we know the precise minimum.
- **Mode T iterative T↔G convergence** — typically 2 iterations, sometimes 3. Is this still consistent in FAYE post-consolidation, or did inlining change anything?

---

## DIAGNOSTIC FINDING — Refine ignores `--replay` (2026-05-30, 21:30)

After adding 3-layer diagnostic logging to refine workers (commits `a9cce99` + `df1b5e3`), running `DV ETH 7h --replay 528 --refine-trials 5` revealed a surprising but **NOT FAYE-introduced** behavior:

**n_size = 2578** observed in refine, despite `--replay 528`.

### Root cause

`_refine_top_configs` at line 9681 hardcodes:
```python
MAX_DIAG_HOURS = 6 * 30 * 24  # = 4320h (6 months) — ignores --replay
```

Then it takes `df_full.tail(MAX_DIAG_HOURS)` and trains on the first 60% → `int(4320 × 0.60) = 2592`, minus dropna/sparse-filter loss → **2578**.

### NOT a FAYE bug

`crypto_trading_system_ed_g_narrow_d_parallel_nearlive.py:480` has the **exact same** `MAX_DIAG_HOURS = 6 * 30 * 24` hardcoded. FAYE faithfully replicates v3 chain behavior. This is **documented v3 design**, not a FAYE consolidation regression.

### Why it surprised us

Test mental model assumed `--replay 528` would shorten everything including refine. It does NOT. `--replay 528` affects:
- Mode D screening: yes (uses `replay_hours or 60*24`)
- Mode V Step 1 backtest: yes (uses `MODE_G_REPLAY_HOURS` or `--replay`)
- **Mode V Step 2 refine: NO** (always 4320h hardcoded)
- **Mode V Step 3 final backtest: NO** (also 4320h hardcoded at line 6127 in `_backtest_one_config`)

### Implications for fast testing

`--refine-trials 5` reduces trial count but each trial still runs walk-forward over 2578 iters × ~400ms = ~17 min wall. K=5 fan-out is parallel (verified — seeds complete within 5s of each other), so the per-trial wall time is dominated by ONE seed's wall × shared CPU contention.

For **future fast tests**, options are:
1. Accept ~80 min refine even on small `--replay` (current behavior)
2. Add a `--refine-replay N` override that lowers `MAX_DIAG_HOURS` for testing only (deviation from v3, easy to add at line 9681)
3. Skip refine entirely with a `--skip-refine` flag (no Mode V Step 2)

User decision yet to be made — for now, accept the long refine and let validation complete.

### K=5 parallelism CONFIRMED working

Diagnostic logs show 5 seeds completing trials within 5 seconds of each other:
```
[21:14:08] cfg=0 seed=42 expected_iters=2383 elapsed=951.3s
[21:14:10] cfg=0 seed=43 expected_iters=2383 elapsed=953.1s
[21:14:10] cfg=0 seed=44 expected_iters=2383 elapsed=953.5s
[21:14:13] cfg=0 seed=45 expected_iters=2383 elapsed=955.9s
[21:14:13] cfg=0 seed=46 expected_iters=2383 elapsed=956.0s
```
Per-iter cost ≈ 400ms (K=5 threads × OMP/BLAS lock serialization) — matches expectations for K=5 fan-out on a shared process.

---

## CONTACT POINTS / REFERENCES

- **Repo**: https://github.com/Spyku/Algo_trading (branch: main)
- **FAYE module**: `crypto_trading_system_faye.py` (~9700 lines)
- **Smoke test**: `tools/smoke_test_faye.py`
- **Promotion script (one-time, manual)**: `tools/promote_v3_5h_winner.py`
- **Parity test**: `tools/validate_core_against_signal_log.py`
- **Audit log**: `ARCHIVED_LOG.md` — search for "FAYE thorough line-by-line audit"
- **Engine refs**:
  - Base engine: `crypto_trading_system_ed.py`
  - g_narrow_d (narrow grid override): `crypto_trading_system_ed_g_narrow_d.py`
  - parallel_nearlive (parallel + nearlive patches): `crypto_trading_system_ed_g_narrow_d_parallel_nearlive.py`
  - step6_nearlive (NEAR_LIVE inner): `crypto_trading_system_ed_step6_nearlive.py`
  - v3 (8-worker dispatcher): `crypto_trading_system_ed_g_narrow_d_parallel_nearlive_v3.py`
  - signal_core_nearlive (compute_signal_core with mean_last_10): `crypto_signal_core_nearlive.py`

- **Today's fix commit chain** (in order):
  1. `23b54b3` faye: fix grid override
  2. `2aea8be` faye: restore 4-layer warning suppression
  3. `f1b49f4` faye: NEAR_LIVE_MODE env var
  4. `1cee360` faye: seeded LGBM factory device=cpu default
  5. `3027255` faye: refine dispatcher 3-worker all-CPU
  6. `a65d618` faye: K=5 wrap parallel ThreadPoolExecutor
  7. `ae056b7` faye: LGBM num_threads=1 + auto-mkdir models_faye/
  8. `d259efe` faye: N_FEATURES_RANGE (4,100) + regime config seeding
  9. `5382740` faye: HOLD branch matches step6_nearlive
  10. `e9790c4` faye: replay-too-short warning
  11. `28227af` faye: announce ProcessPool shutdown progress
  12. `8776f38` faye: _alive(pid) Windows safe via OpenProcess+GetExitCodeProcess

---

## REMINDERS / TASKS WAITING

- [ ] Promote v3 5h winner OR FAYE 7h winner once H finishes (whichever is higher honest backtest)
- [ ] Run RST ETH 7h,8h --replay 528 after H done + 8h row copied
- [ ] If RST passes, launch full HRST at --replay 1440 for production candidates
- [ ] Restart trader when promotion is ready and trader is verified flat
- [ ] Add streaming stdout to refine workers (UX improvement — separate task)
- [ ] Trace the "Need 500+" check location in FAYE
- [ ] Profile worker speed asymmetry (research idea #1)
- [ ] Consider Mode R / Mode T full line-by-line audit (not done today)

---

*Generated 2026-05-30 during the FAYE H ETH 7h --replay 528 run on Desktop. Refine phase active, ~30-60 min remaining at write time. Document is current as of commit `8776f38`.*
