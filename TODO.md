## ⚡ TODO — Active work only

**Companion file**: [ARCHIVED_LOG.md](ARCHIVED_LOG.md) — historical audit trail, canonical scoreboard C01-C82, MERGED TOPICS, IDEA QUEUE drop-list (closed/shipped/STUB items with verdicts).

## 📊 At-a-glance — active TODO dashboard (2026-05-27)

| Pri | Item | When | Status |
|---|---|---|---|
| 📌 | **G_narrow LIVE** (CSV+config since 2026-05-21 21:56) running on **H75 engine** (K=5 + 75 trial refine) — `sma24>sma100` / bull 5h@65% / bear 8h@65% | active since 2026-05-21 21:56 (G_narrow promote); engine unchanged from 2026-05-18 H75 | running |
| 🚨 P0 | **TODO 0526 — LIVE vs BACKTEST divergence investigation** | Verdict 2026-05-26 23:15; shadow mode activated 2026-05-27 00:02 on Desktop | 🕐 **MONITORING — shadow data accumulating**. Root cause = semantic code divergence (4 specific algorithmic differences between live + backtest, code similarity 8.2%). First shadow row matched (live HOLD 39.7% vs core HOLD 39.66%). Periodic check + ~7-14 day analysis pending — see Step 4 of TODO 0526 plan. |
| ✅ | **TODO 0525** — G_narrow_d HRST with extended grid + V2 top-10 + Optuna win_hi=350 | Desktop 2026-05-25 22:38 → 2026-05-26 08:44 (~10h) | DONE — Mode T REF +83.85% on May 22 data, lost to LIVE +91.01% by 7pp. Hypothesis "extend grid unlocks high-window basin" REJECTED — 0-trade holdout filter eats w=250/300 candidates regardless. **Architecture analysis spawned TODO 0526.** |
| ✅ | **TODO 0524** — Top-5 HRST (5,6,8,11,12h) clean rerun on fixed parallel fork | Desktop 2026-05-24 22:53 → 2026-05-25 06:39 (Mode H+R+S); Mode T reruns 2026-05-25 12:35 + 13:17 | DONE — Mode T REF +80.56% vs LIVE's +91.01% on same May 22 data. No promotion. Parallel fork validated (~8× refine speedup = real shipped win). Root cause of weaker REF: narrow grid [72,100,150] couldn't seed Optuna refine to reach LIVE's w=281/293 basin. → spawned TODO 0525 |
| ✅ | **TODO 0522** — Parallel refine speedup (G_narrow_d_parallel fork) + long-horizon G test (9-12h) | Stage 1 Laptop 2026-05-22 00:26; Stage 2 Laptop 2026-05-22 01:39 → 18:09 | ⚠️ Stage 2 verdict INVALID (grid bug). Stage 1 PASSED. Parallel fork retained + bug-fixed; superseded by TODO 0524 |
| 🔥 P1 | **OOS monitoring** — first 10 trades audit on live G_narrow config | window ~2026-06-04 (14 days from 2026-05-21 promote) | 0/10 closed; trader running |
| ✅ | **TODO 0519** — G_narrow_d relaunch on Desktop | completed 2026-05-20 → 2026-05-21 | DONE — Mode T REF +89.14%, converged iter 2, no STRICT winner |
| 🔥 P1 | **TODO 0519B-G1** — `deriv_oi_*` re-enable A/B test | Originally scheduled 2026-05-22; slipped — Desktop booked by TODO 0524 until ~2026-05-25 morning | 📅 PENDING — procedure ready, awaiting Desktop free |
| 📋 P2 | **TODO 0519B-G2** — orderbook + IV re-enable A/B test | 2026-06-18 (~30 days) | 📋 SCHEDULED — depends on G1 outcome |
| 🚀 P3 | **P4** — C14 vol-conditional triple-barrier retest | when capacity (~2.5h) | open |
| 🚀 P3 | **P5** — C11 VPIN at 5-min cadence | when capacity (~1 day eng) | open |
| 🚀 P3 | **P6** — C15 meta-labeling SOL/BTC | blocked (assets shelved, ~6h) | open |
| 🚀 P3 | **IDEA QUEUE Tier A** — Untested clean (C13-narrow, C54, C55, C58, C59) | research backlog | 5 items open |
| 🚀 P3 | **IDEA QUEUE Tier B** — V3-lit C60-C82 (16 of 23 patcher-ready) | research backlog | 23 items open |
| ⚪ P4 | **TODO 0519C** — CPCV HRST diagnostic | trigger-based re-run | available, no plan |
| ⚪ P4 | **Kalshi** — prediction-market data integration | needs API key + impl | backlog |

**Honest top-of-mind**: G_narrow live since 2026-05-21 21:56 (CSV + regime config), H75 engine unchanged. TODO 0524 closed — Mode T head-to-head on May 22 data showed LIVE +91.01% beats new +80.56% (gap from narrow grid missing LIVE's w=281/293 basin). TODO 0525 (in flight on Desktop, ~9-9.5h) tests whether grid [72,100,150,200,250,300,350] + V2 top-10 + Optuna win_hi=350 fixes that. Validation gate: does ETH 5h refine reach w≈281 and 8h reach w≈293. If yes, REF should land near LIVE's +91% on May 22 data. Everything else is wait-or-research.

---

**Layout (priority-ordered, top → bottom)**:
- 📌 LIVE STATE (always visible — current production config + rollback)
- 🔥 **P1** — Act this week (in-flight + imminent)
- 📋 **P2** — Scheduled next month
- 🚀 **P3** — Research backlog (when capacity allows)
- ⚪ **P4** — Low priority / Diagnostics / Backlog

---

## 📌 LIVE STATE — G_narrow models on H75 engine (promoted 2026-05-21 21:56 CEST)

**Engine** (unchanged since 2026-05-18 H75 promotion): `crypto_trading_system_ed.py` — H_STRICT_FAMILY merge (K=5 multi-seed + REFINE_TRIALS=75 + strict `(combo, w)` dedup).

**Models + regime config** (swapped 2026-05-21 21:56 from H75-fresh to G_narrow_d's HRST output; G_fresh "promote" on 2026-05-22 19:51 was content-identical and effectively no-op for ETH 5h/8h):
- Detector: `sma24>sma100` (unchanged across all 3 promotions)
- Bull = **5h@65%** RF+LGBM w=281 γ=0.9981 12f (G_narrow_d May 20-21 Desktop refine winner)
- Bear = **8h@65%** RF+LGBM w=293 γ=0.9990 16f (G_narrow_d May 20-21 Desktop refine winner)
- Shields OFF (both regimes)
- **Rally cooldown OFF** (both regimes — manually toggled 2026-05-23 22:21 from `enabled: true`)
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
copy crypto_trading_system_ed_pre_H75_20260518.py     crypto_trading_system_ed.py
```

**Promotion source**: Desktop G_narrow_d HRST run 2026-05-20 11:05 → 2026-05-21 10:28 (wall ~23h 22m, log `logs/ed_v1_20260520_110556.log`). Mode T REF +89.14% (converged iter 2, no STRICT rally-cooldown winner).

**Promotion timeline**:
1. 2026-05-18 22:02 — H75 promoted (snapshot: `pre_H75_20260518`)
2. 2026-05-20 09:04 — H75-fresh promoted (snapshot: `pre_H75fresh_20260520`)
3. **2026-05-21 21:56 — G_narrow promoted (current)** (snapshot: `pre_G_narrow_20260521`)
4. 2026-05-22 19:51 — G_fresh promoted (content-identical ETH rows; snapshot: `pre_G_fresh_20260522`)
5. 2026-05-23 22:21 — manual: rally_cooldown enabled → disabled

Full promotion events in ARCHIVED_LOG.md.

---

# 🚨 P0 — TODO 0526 — LIVE vs BACKTEST divergence investigation

**Search anchor**: `TODO 0526`

**Status**: 🕐 **MONITORING — shadow accumulating since 2026-05-27 00:02 on Desktop**. Root cause identified 2026-05-26 23:15. Steps 1-3 done; Step 4 in progress (data accumulation + periodic match-rate checks); Steps 5-7 planned. **Halt all promotion decisions until Step 4 verdict in ~7-14 days.**

## 🎯 VERDICT (2026-05-26 23:15 — from `output/diagnostics_20260526_230310_fast/`)

**Bug is SEMANTIC CODE DIVERGENCE, not data drift.** The live trader's `generate_live_signal()` and the backtest's `_deku_eval_with_pruning()` are **fundamentally different algorithms** that produce different predictions on identical data. 4 specific divergences identified in code:

| # | Difference | Live trader behavior | Backtest behavior |
|---|---|---|---|
| **1** | **Embargo** ([B3 confirmed]) | Trains on `[H-window, H-1]` — no gap | Trains on `[H-window, H-horizon]` — 5h gap for 5h model |
| **2** | **NaN handling** (NEW finding) | `df.ffill().fillna(0.0)` — imputes stale features | Skips entire evaluation step if ANY NaN in X_train or X_test |
| **3** | **Step size** (NEW finding) | Evaluates every hour | `_deku_eval_with_pruning` walks forward by `DIAG_STEP=36` — evaluates every 36 hours |
| **4** | **Signal semantics** ([B2 confirmed]) | Ternary BUY/SELL/HOLD with proba-based confidence | Binary BUY/not-BUY only; no HOLD; no probabilities |

**Code similarity**: live vs backtest signal generation = **0.082 (8.2%) — 0.019 (1.9%) line-level identity**. They are not the same algorithm.

**This means**: every backtest result you've ever produced (Mode V return, Mode T REF, the +91.01% LIVE inference benchmark) was computed by a DIFFERENT algorithm than the one actually trading live. **You have been optimizing a metric that does not predict live performance**. The 68% signal disagreement is not a bug — it's the expected outcome of two different functions.

### Data drift verdict (partial — not the main cause)
| File | Drift found | Material for LIVE 5h? |
|---|---|---|
| `eth_hourly_data.csv` | ✓ none across 76,668 rows | n/a |
| `derivatives_eth.csv` | 239 cells (mostly NaN-recovery for `deriv_oi_*` quarantined features) | 1 material cell: perp_close 2026-05-21 23:00 drifted 0.11% |
| `onchain_eth.csv` | ✓ none across 1,601 rows | n/a (oc_mvrv stable) |
| `macro_daily.csv` | 4 cells, all 2026-05-21 (OIL −1.4%, GOLD −0.04%, DXY/NASDAQ float noise) | none of these in LIVE 5h features |
| `cross_asset.csv` | 4 cells, all 2026-05-21 (DAX −0.5%, BTC −0.11%, ETH/NASDAQ noise) | none in LIVE 5h features |
| `stablecoin_flows.csv` | ✓ none | n/a |
| `fear_greed.csv` | ✓ none | n/a |

Data drift IS real but explains <10% of the gap. Code-path divergence explains the rest.

## What's now in progress (started 2026-05-26 23:15)

**Refactor target**: extract a shared signal-generation core into `crypto_signal_core.py` (NEW FILE). Both live trader and backtest will eventually call this same core, with explicit parameters for embargo / NaN policy / signal semantics. Once both paths share the core, backtest results become a faithful simulation of live behavior.

**Plan** (1-2 days of careful engineering):
1. ✅ Verdict documented (2026-05-26 23:15)
2. ✅ DONE 2026-05-26 23:30 — [crypto_signal_core.py](crypto_signal_core.py) shared core written + 5/5 smoke tests pass
3. ✅ DONE 2026-05-26 23:45 — Shadow mode infrastructure shipped:
   - **[crypto_live_shadow.py](crypto_live_shadow.py)** — wrapper that replicates live trader data prep + calls `compute_signal_core()` + logs to `config/shadow_signal_diff.csv`. Never crashes live trader (all errors swallowed).
   - **Minimal patch to [crypto_revolut_ed_v2.py:1620-1627](crypto_revolut_ed_v2.py#L1620)** — 8-line `try` block gated on `SHADOW_MODE=1` env var. When OFF (default), zero overhead. When ON, shadow runs after every `generate_live_signal()` call.
   - **Activation**: `$env:SHADOW_MODE = "1"` then restart trader (`start_ed_v2.bat`). Deactivation: clear env var + restart.
   - ✅ ACTIVATED 2026-05-27 00:02 on Desktop. First row written 00:02:27: ETH 5h live=HOLD(39.70%), core=HOLD(39.66%), match=True, conf_delta=-0.04, n_features=12, n_train=281 (matches LIVE config exactly). Trader running with shadow enabled.
4. 🕐 IN PROGRESS — Accumulate ≥7 days of shadow data, then analyze.
   - **Target window**: ~2026-06-03 (7 days from activation) for first formal analysis. Continue accumulating to ~2026-06-10 (14 days) for stable verdict.
   - **Pending periodic checks** (every 1-2 days, takes <1 min):
     ```powershell
     Import-Csv config\shadow_signal_diff.csv | Group-Object match | Select Count,Name
     Import-Csv config\shadow_signal_diff.csv | Group-Object shadow_error | Select Count,Name
     ```
     First confirms match rate (≥95% means core faithfully replicates live; <95% means shadow's data prep has bugs vs `generate_live_signal()`). Second confirms no silent error rate.
   - **Pending divergence analysis** (run once enough data accumulated):
     ```powershell
     Import-Csv config\shadow_signal_diff.csv | Where-Object { $_.match -eq "False" } | Format-Table timestamp_utc,horizon,live_signal,core_signal,conf_delta -AutoSize
     ```
     Each match=False row is a concrete divergence case to debug. Look for patterns: specific hours, specific regime states, specific feature values.
   - **Decision gate before Step 5**: match rate must be ≥95% AND shadow_error rate must be <1%. If either fails, debug `crypto_live_shadow.py`'s prep replication before switching live to call core.
5. ⏸ PLANNED — Switch live trader to call `compute_signal_core()` directly. Remove inline math from `generate_live_signal()` — keep only data prep, staleness checks, result formatting.
6. ⏸ PLANNED — Refactor `_deku_eval_with_pruning` and `_simulate_with_threshold` in the backtest engine to use the same core with explicit `embargo`, `na_policy`, `signal_mode` parameters exposing the legitimately-different policy choices.
7. ⏸ PLANNED — Re-run HRST on refactored engine. Expected outcome: lower headline Mode T REF (because backtest now does what live does), but predictive of actual live performance.

## ACTIVATION COMMANDS (Shadow Mode — Step 3 of plan)

**👀 READ THIS ON DESKTOP** — the trader runs on Desktop (`G:\engine\`), even though the source files may have been edited on Laptop. Drive sync mirrors everything between machines.

### Why Desktop (NOT Laptop)

| Factor | Desktop | Laptop |
|---|---|---|
| CLAUDE.md convention | "Desktop (primary)" | secondary |
| 24/7 uptime track record | yes | sleep/power-management risk |
| The trader runs OK there now | yes | not currently running |
| Hardware | i7-14700KF, 32GB, RTX 4080 | weaker, mobile |

**🚨 Do NOT start the trader on Laptop while it's running on Desktop** — two traders on the same Revolut X account = race condition + duplicate Telegram bots responding to the same commands + potentially conflicting BUY/SELL orders.

### Step-by-step (run ALL on Desktop, in order)

#### Step 1 — Wait for Drive sync (~1-2 min after Laptop edits)

On Desktop, verify the 3 new/changed files synced from Laptop:

```powershell
cd G:\engine
Test-Path crypto_signal_core.py
Test-Path crypto_live_shadow.py
Get-Content crypto_revolut_ed_v2.py | Select-String -Pattern "shadow_compare"
```

All three must succeed:
- `Test-Path crypto_signal_core.py` → `True`
- `Test-Path crypto_live_shadow.py` → `True`
- Last command shows 2 matches (one for the import, one for the call site)

If files missing after 2-3 min: right-click Google Drive tray icon → "Sync now", or wait longer.

#### Step 2 — Stop the running trader on Desktop

**Preferred — via Telegram bot**:
```
/stop
```
Wait for the stop confirmation message. The trader finishes its current cycle cleanly.

**Alternative — kill the Python process** (if Telegram isn't responsive):
```powershell
Get-Process python | Where-Object { $_.MainWindowTitle -like '*ed_v2*' -or $_.Path -like '*revolut_ed_v2*' }
# Confirm it's the right process, then:
Stop-Process -Id <PID> -Force
```

#### Step 3 — Verify trader state (optional but recommended)

```powershell
Get-Content config\position_ed_v2_ETH.json | Select-String -Pattern '"state"'
```

- `"state": "cash"` → safe to restart immediately
- `"state": "invested"` → restart still safe (shadow is read-only, doesn't change trade decisions). Trader resumes its position on startup. Small risk of one missed cycle during restart.

#### Step 4 — **Open a FRESH PowerShell window** (critical)

Avoids leftover env vars from any prior diagnostic runs (V2_DATA_SNAPSHOT etc.).

In the fresh window:
```powershell
cd G:\engine

# Sanity: verify no leftover redirect env vars
echo "V2_DATA_SNAPSHOT=$env:V2_DATA_SNAPSHOT"
echo "H_STRICT_MODELS_DIR=$env:H_STRICT_MODELS_DIR"
echo "H75_WIDE_MODELS_DIR=$env:H75_WIDE_MODELS_DIR"
# All three should print "=" with nothing after — i.e. blank values
```

If any of those env vars are non-blank, open a different fresh PowerShell window.

#### Step 5 — Enable shadow mode and launch

```powershell
$env:SHADOW_MODE = "1"
echo "SHADOW_MODE=$env:SHADOW_MODE"   # should print: SHADOW_MODE=1
.\start_ed_v2.bat
```

(The `.\` prefix is REQUIRED — PowerShell doesn't search the current directory by default.)

Watch the trader start. Expected:
- Pre-flight check passes (`✓` next to each data source — NO `✗`)
- Banner: `==== REVOLUT X MULTI-ASSET TRADER [ED V2] ====`
- Within a few minutes, first cycle begins

If pre-flight FAILS with data staleness errors AND mentions snapshot paths → you didn't get a fresh window; env vars leaked. Go back to Step 4.

#### Step 6 — Verify shadow logging is active (after first signal cycle, max 1h)

```powershell
Test-Path config\shadow_signal_diff.csv
Get-Content config\shadow_signal_diff.csv | Select-Object -First 5
```

Expected: header row plus 1-2 data rows per cycle (one per ETH horizon). Columns:
```
timestamp_utc, asset, horizon, live_signal, live_confidence,
core_signal, core_confidence, match, conf_delta,
shadow_error, shadow_elapsed_ms, n_features, n_train, inference_row_dt
```

If the file exists but `core_signal` is mostly `None` and `shadow_error` shows entries → my replicated data prep in `crypto_live_shadow.py` has a bug. Open `config/shadow_signal_diff_errors.log` for tracebacks.

### Monitor from Laptop (without moving the trader)

You don't need the trader running on Laptop to monitor it:

```
Telegram → /status     # what trader is doing right now
Telegram → /regime     # current bull/bear state
Telegram → /balance    # current position
```

The shadow log appears on Laptop via Drive sync within ~1 min of being written on Desktop:
```powershell
# ON LAPTOP, after Desktop's first shadow cycle:
Get-Content C:\Users\Alex\algo_trading\engine\config\shadow_signal_diff.csv | Select-Object -First 5
```

### To disable shadow mode later

Close that PowerShell window on Desktop (session-only env var disappears) and restart trader without `SHADOW_MODE=1`. Or in the current window:
```powershell
$env:SHADOW_MODE = ""
# restart trader via Telegram /stop + .\start_ed_v2.bat
```

### Safety guarantees

- ✅ Shadow path is wrapped in `try/except: pass` — any error inside shadow code is swallowed, live trader continues normally
- ✅ Default behavior (no env var) = zero overhead, zero code path change
- ✅ Shadow is read-only — never modifies positions, regime config, or production CSV
- ✅ Cannot break trading by enabling shadow mode

### Disk cost

~14 columns × 80 bytes/row × 24 cycles/day × 4 horizons = ~107 KB/day = 39 MB/year. Negligible.

**Halts in place during refactor**:
- ❌ No promotion of any new config to LIVE
- ❌ No further HRST runs that would be used for promotion (research-only OK)
- ✅ Live trader continues on current G_narrow config

## Overnight runs status (after verdict)

The overnight diagnostic battery is no longer needed to find the bug — it's been found. Optional follow-ups:

- **Cancel laptop run** — not needed
- **Cancel desktop run** — not needed
- **Embargo A/B (`tools/embargo_ab_test.py --mode=both`)**: still worth running because it quantifies how much of the 4 divergences difference #1 alone contributes. ~2.5h, low value but cheap.

## The problem (proven facts)

| Metric | Live trader (56 days) | Backtest expectations (Mode V) | Gap |
|---|---|---|---|
| ETH 5h Win Rate | **50.9%** | **85%** | **−34pp** |
| ETH 5h Per-trade return | +0.124% | ~+0.85% | **−7×** |
| Sampled signal agreement | n/a | n/a | **40% match, 60% disagree** (10 samples) |
| Win distribution | 55% of wins are <0.5%, biggest +4.99% | Backtest wins include +57.8%, +63.0%, +68.5% | Fat tail amputated |

## What's already ruled out

- ❌ Maker order fees (verified 0% fee fills in trader log)
- ❌ PySR feature drift (JSONs dated April 21, predate production CSV)
- ❌ Recent config promotion (50% WR holds across 30+ days under 3 different configs)

## All unproven hypotheses (25 items, organized)

### Tier 1 — Data divergence (most probable, code-confirmed cause exists)
- **A1**: `macro_daily.csv` rows overwritten by yfinance via `keep='last'` ([download_macro_data.py:384](download_macro_data.py#L384))
- **A2**: `derivatives_eth.csv` (funding/OI/perp) FULL re-pulls overwrite ALL history ([download_macro_data.py:925-975](download_macro_data.py#L925))
- **A3**: `onchain_eth.csv` CoinMetrics revisions overwrite past daily values
- **A4**: `cross_asset.csv` revisions
- **A5**: `stablecoin_flows.csv` revisions via `keep='last'` ([download_macro_data.py:1252](download_macro_data.py#L1252))
- **A6**: Hourly price `eth_hourly_data.csv` backfill of past hours
- **A7**: Sparse features (orderbook, IV) — supposed to be append-only

### Tier 2 — Code-level divergence
- **B1**: `crypto_live_trader_ed.py` feature pipeline ≠ `crypto_trading_system_ed.py` backtest pipeline
- **B2**: Live signal generation function ≠ backtest's `_deku_eval_with_pruning`
- **B3**: Embargo: live trains `[H-281, H-1]` (no embargo per CLAUDE.md rule #9), backtest trains `[H-286, H-5]` (with embargo)
- **B4**: K=5 median (backtest) vs single seed (live unknown)
- **B5**: Random state at inference may differ
- **B6**: PySR feature compute path may differ

### Tier 3 — Execution-side
- **C1**: Backtest exits at fixed horizon; live exits on first SELL signal flip
- **C2**: `max_hold = 10h` violated live (observed 32.9h hold May 15-17)
- **C3**: `min_sell_pnl = 0%` caps wins early in live
- **C4**: Slippage modeling difference
- **C5**: Maker order partial fills timing

### Tier 4 — Regime / confidence / timing
- **D1**: Regime detector inconsistency between sources
- **D2**: Live trader regime cache staleness
- **E1**: Ensemble probability aggregation difference
- **E2**: Confidence threshold application difference
- **F1**: Backtest assumes signal at H:00, live at H:00:30+
- **F2**: Maker fill timing asymmetry vs hour close
- **G1**: Live trader model caching across cycles
- **G2**: Per-cycle retraining marginal differences

## The fix that's safe to apply NOW (no testing needed)

Patch the dedup pattern in `download_macro_data.py` from `keep='last'` → `keep='first'` for any rows older than the current hour. Preserves point-in-time integrity going forward (even if some currently-broken data is already drifted, future writes are safe).

Affected lines: 384, 1252, 1342, 1401. Each needs a "current-hour exception" so in-flight rows can still update.

For derivatives FULL re-pull (lines 925-975): switch to incremental tail-update only (last 24-48h read from API; older rows preserved from existing CSV).

**Status**: not yet applied — patch design ready, awaiting diagnostic results to confirm this is THE bug.

## Deliverables (committed 2026-05-26)

| Script | Purpose | Runtime | Output |
|---|---|---|---|
| **[tools/diagnostic_battery.py](tools/diagnostic_battery.py)** | Main overnight battery — runs 9 fast phases + scenario-specific deep phases | ~12h per machine | `output/diagnostics_<ts>_<scenario>/` |
| **[tools/embargo_ab_test.py](tools/embargo_ab_test.py)** | Standalone Phase 10 — embargo A/B via monkey-patched engine fork (auto-generated, auto-deleted) | ~2.5h | `output/embargo_ab_<ts>/` |

Both scripts:
- Use `--no-persist` + isolated `H_STRICT_MODELS_DIR` / `H_STRICT_CONFIG_DIR` env vars → live production CSV and regime config UNTOUCHED
- Wrap every phase in try/except → single failure doesn't abort the battery
- Write markdown report at the end so the next morning is a 2-minute read

## Definitiveness expectation: ~65-70%

Honest breakdown:
- **~50%** chance the `data_drift_check` phase produces a smoking gun (>5% cells changed in macro/derivatives/onchain) → fix is the dedup patch below → bug solved
- **~10-15%** added by Phase 10 embargo A/B definitively quantifying B3
- **~5-10%** added by Phase 8 code-path diff revealing clear semantic divergence
- **~30-35% residual risk**: multiple causes contributing, bug in something not enumerated, or feature-level divergence whose mechanism we can't pinpoint without ground-truth live feature logs (Phase 9 sets up that observability for NEXT debug cycle)

## Phase inventory (all 12 phases)

### Fast phases (~45 min, run on BOTH scenarios)

| Phase | Function | What it tests |
|---|---|---|
| 0a | `snapshot_current_state` | Backup all data CSVs to `data/_diagnostic_snapshots/snap_<ts>/` for future 7-day revision check |
| 0b | `static_code_analysis` | Grep 4 code files for 14 pattern categories (seed, cache, embargo, K=5, dedup, exit gates, signal entry points) |
| 0c | `trade_analytics` | Full live PnL breakdown — all-time / 30d / 14d / 7d, win-loss distribution, biggest wins/losses |
| 0d | `signal_log_analysis` | Parse signal_log.csv — action distribution per month, confidence stats by action |
| 1 | `data_drift_check` | **TIER 1 SMOKING-GUN TEST**: diff May 22 snapshot vs current data row-by-row for 7 CSVs (price, derivatives, onchain, macro, cross-asset, stablecoins, fear-greed). Reports per-column cell diffs + example changed dates |
| 2 | `existing_log_diff` | Parse [logs/ed_v1_20260525_204735.log](logs/ed_v1_20260525_204735.log) (May 25 head-to-head) — diff backtest signals against signal_log at scale |
| 3 | `exit_policy_replay` | Simulate 5 exit policies on live signal stream since 2026-05-21: live-actual, no-profit-cap, force-horizon-5h, stop-loss-2pct, hold-to-max. Compare total PnL |
| **8** | `code_path_diff` | **Phase 8** — AST-extract 6 functions (`generate_live_signal`, `detect_regime`, `generate_signals`, `_deku_eval_with_pruning`, `_h_deku_eval_median_k`, `_generate_signals_cached`) to standalone .py files, then difflib unified diffs with similarity ratios |
| **9** | `write_instrumentation_patch` | **Phase 9** — Writes `feature_logging_hook.py` + `INSTRUCTIONS.md` for manual application to live trader. Enables feature-level analysis of NEXT divergence (does NOT help retrospectively) |

### Laptop deep phases (focus: data drift, Tier 1)

| Phase | Function | What it tests |
|---|---|---|
| L1 | `backtest_signal_regen_k5` | Subprocess: `python crypto_trading_system_ed.py T ETH --replay 1440 --no-persist --no-data-update` with `V2_DATA_SNAPSHOT=data/_reliability_hrst_snapshot_laptop_20260522_0139` and `RELIABILITY_K=5`. Parses 1440 signals × 2 horizons. Diffs every signal against `config/signal_log.csv`. Outputs `signal_diff_k5.csv` |
| L2 | `mode_v_sensitivity_laptop` | Subprocess: `DV ETH 5h --replay 1440 --no-persist --no-data-update` on snapshot. Shows what Mode V picks today vs what LIVE has. Skip with `--skip-slow` |

### Desktop deep phases (focus: code/seed sensitivity)

| Phase | Function | What it tests |
|---|---|---|
| D1 | `backtest_signal_regen_k1` | Same as L1 but `RELIABILITY_K=1`. If K=1 matches signal_log better than K=5 → K=5 is the bug source |
| D2 | `backtest_signal_regen_current_k5` | Same as L1 but on CURRENT data (no snapshot). Diff vs L1's snapshot result quantifies data drift between May 22 and today |
| D3 | `mode_v_sensitivity_desktop` | Cross-machine duplicate of L2 |

### Sequential Phase 10 (run AFTER diagnostic battery, either machine)

| Script | What it tests |
|---|---|
| `tools/embargo_ab_test.py --mode=both` | (i) Mode T baseline (embargo=horizon, current behavior), (ii) Mode T with auto-generated fork `crypto_trading_system_ed_embargo_zero.py` that monkey-patches `_h_deku_eval_median_k` to force embargo=1, (iii) signal-by-signal diff. Verdict: minimal (≥95% match) / modest (80-95%) / major (<80%) embargo effect |

---

## TERMINAL COMMANDS — exact, copy-paste ready

### Step 1: smoke test on either machine (5-10 min, recommended first)

```powershell
cd C:\Users\Alex\algo_trading\engine
git pull
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null
python tools/diagnostic_battery.py --scenario=fast
```

**Verify before launching overnight runs**: open `output/diagnostics_<ts>_fast/report.md` and confirm at least these phases succeeded: `snapshot_current_state`, `trade_analytics`, `data_drift_check`, `code_path_diff`. If any of those error out, fix the issue before committing 12h of compute.

### Step 2: LAPTOP overnight (~12h)

```powershell
cd C:\Users\Alex\algo_trading\engine
git pull
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null
$env:PYTHONWARNINGS = "ignore:X does not have valid feature names:UserWarning"
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
python tools/diagnostic_battery.py --scenario=laptop
```

Tee log + foreground (recommended so you can `tail -f`):

```powershell
python tools/diagnostic_battery.py --scenario=laptop *>&1 | Tee-Object -FilePath "logs\diag_laptop_$(Get-Date -Format yyyyMMdd_HHmm).log"
```

### Step 3: DESKTOP overnight (~12h)

```powershell
cd G:\engine
git pull
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null
$env:PYTHONWARNINGS = "ignore:X does not have valid feature names:UserWarning"
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
python tools/diagnostic_battery.py --scenario=desktop
```

Tee log variant:

```powershell
python tools/diagnostic_battery.py --scenario=desktop *>&1 | Tee-Object -FilePath "logs\diag_desktop_$(Get-Date -Format yyyyMMdd_HHmm).log"
```

### Step 4: Embargo A/B (~2.5h, EITHER machine, AFTER Steps 2+3 complete)

```powershell
cd C:\Users\Alex\algo_trading\engine    # or G:\engine on desktop
git pull
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null
$env:PYTHONWARNINGS = "ignore:X does not have valid feature names:UserWarning"
python tools/embargo_ab_test.py --mode=both
```

Optional — run baseline only first as sanity check (40-70 min):

```powershell
python tools/embargo_ab_test.py --mode=baseline
# Inspect output/embargo_ab_<ts>/baseline_signals.csv — should look sane
python tools/embargo_ab_test.py --mode=no_embargo
```

### Skip slow phases (testing the script itself without committing 12h)

```powershell
python tools/diagnostic_battery.py --scenario=laptop --skip-slow
python tools/diagnostic_battery.py --scenario=desktop --skip-slow
```

---

## The fix that's safe to apply NOW (no testing needed)

Patch the dedup pattern in `download_macro_data.py` from `keep='last'` → `keep='first'` for any rows older than the current hour. Preserves point-in-time integrity going forward (even if some currently-broken data is already drifted, future writes are safe).

Affected lines: 384, 1252, 1342, 1401. Each needs a "current-hour exception" so in-flight rows can still update.

For derivatives FULL re-pull (lines 925-975): switch to incremental tail-update only (last 24-48h read from API; older rows preserved from existing CSV).

**Status**: NOT yet applied — patch design ready, awaiting diagnostic results to confirm this is THE bug.

## Decision tree (apply after both diagnostic batteries complete tomorrow morning)

| Result | Conclusion | Action |
|---|---|---|
| `data_drift_check` shows >5% cells changed in macro/derivatives/onchain files | A1-A5 confirmed (data drift is at least PART of cause) | Apply dedup patch (see above); 30-day stability gate before any new promotion |
| `data_drift_check` shows <1% drift | A1-A6 RULED OUT | Bug is in code/embargo/seed — look at Phase 8, 10, K=1 results |
| L1 vs D2 (snapshot K=5 vs current K=5) signals differ | Data drift between May 22 and now confirmed at signal level | Strengthens A1-A6 verdict |
| D1 vs L1 (K=1 vs K=5) signals differ >10% | K=5 multi-seed materially changes predictions | If live uses K=1, switching backtest to K=1 may close gap |
| L1 / D1 signals diverge from signal_log >40% (current gap) | Confirms divergence at scale | Look at Phase 8 code diff for mechanism |
| Phase 8 `code_path_diff` similarity ratio < 0.5 between live and backtest signal gen | Major code divergence | Manual review of extracted .py files; B1/B2 confirmed |
| Phase 10 `embargo_ab_test` match rate < 80% | Embargo asymmetry is major contributor (B3) | Consider (a) embargo=0 backtest as promotion gate, (b) document embargo gap |
| `exit_policy_replay` shows alternative policy gains >5pp on real live signals | C1/C3 confirmed; live exit policy is suboptimal | Patch live trader exit logic to match better-performing policy |
| Multiple of above | Multiple causes — rank by magnitude, fix largest first | |
| **None of above conclusive** | Bug is in something not enumerated; need Phase 9 hook applied + 30-day data | Apply Phase 9 instrumentation patch (`feature_logging_hook.py`) for next investigation |

## Halts in place until diagnostic completes

- ❌ No promotion of any new config to LIVE (even if it backtests better)
- ❌ No further HRST runs on the assumption that Mode T REF predicts live performance
- ✅ Live trader continues running on current G_narrow config (it's profitable at +7% / 56 days)

## What gets produced overnight

After both 12h batteries + the 2.5h embargo A/B complete:

```
output/
  diagnostics_<ts>_laptop/
    report.md                    ← READ THIS FIRST
    run.log                      progress log (tail -f friendly)
    data_diff_<csvname>.csv      cell-level differences per data file
    signal_diff_k5.csv           L1 result: backtest K=5 vs live log per hour
    code_extracts/               Phase 8 extracted .py files + difflib diffs
    instrumentation/             Phase 9 patch + INSTRUCTIONS.md
    mode_t_subprocess_k5.log     subprocess stdout
    mode_v_subprocess.log        deep-phase subprocess stdout
    all_results.json             machine-readable
  diagnostics_<ts>_desktop/
    report.md                    ← READ THIS SECOND
    signal_diff_k1.csv           D1 result: K=1 vs live log
    signal_diff_k5.csv           D2 result: current data K=5 vs live log
    [same structure]
  embargo_ab_<ts>/
    report.md                    ← READ THIS THIRD
    signal_diff.csv              baseline vs no_embargo per hour
    baseline_signals.csv
    no_embargo_signals.csv
```

Total user time tomorrow morning to digest: ~30-60 minutes of report reading.

---

# 🔥 P1 — Act this week

## ⚡ G_narrow LIVE OOS monitoring — active, ~14 days

**Search anchor**: `G_NARROW-OOS-MONITOR`

**Current state (2026-05-24 23:30)**: Trader currently **invested** (BUY 2026-05-24 21:03Z @ $2094.11). 2 closed round-trips under G_narrow so far — both losses. Window started 2026-05-21 21:56 → ends ~2026-06-04 (14 days).

**Closed trades under G_narrow** (since 2026-05-21 21:56 promote):

| # | Open | Close | Entry | Exit | PnL |
|---|---|---|---|---|---|
| 1 | 2026-05-22 18:22Z (manual BUY) | 2026-05-23 22:00 CEST (auto SELL) | $2120.96 | $2075.26 | **−2.15% / −$298.81** |
| 2 | 2026-05-24 18:02Z (auto BUY) | 2026-05-24 22:00 CEST (auto SELL) | $2100.60 | $2098.32 | **−0.11% / −$14.73** |

Cumulative realized **−2.26%** / 2 trades / WR **0%**. Note the first trade was a manual BUY (user-initiated), so it's only partially attributable to G_narrow signal quality. Trade #2 was fully automated and still lost.

**Rollback triggers under G_narrow** (any one fires → discuss with user):
- Cumulative realized alpha < +5% after first 10 trades (currently −2.26% on 2 — too early)
- MaxDD exceeds −10% on G_narrow alone (sim Mode T REF was +89.14% on 4-horizon, no per-trade-DD sim)
- First 10 trades WR < 50% (currently 0% on 2 — flag if it persists; sim showed 84% WR on 5h, 90% on 8h)
- Trade count vastly different from sim (sim showed ~146 5h trades / 60d ≈ 2.4/day; G_narrow live is 2 auto-trades over ~3 days ≈ 0.67/day — running below sim pace)

**Decision tree after 1-2 weeks (or 10 closed trades, whichever first)**:
| Outcome | Action |
|---|---|
| G_narrow holds OOS (≥+44% of sim alpha = +39% realized, no triggers) | Continue. Document the +5pp upgrade vs H75-fresh as real. |
| G_narrow underperforms (>2 triggers OR alpha < +5% after 10 trades) | **Rollback to H75-fresh** (one-command, see LIVE STATE rollback ladder L1). If H75-fresh ALSO underperforms (would mean broader issue), L2 to H75-snapshot. |
| Borderline (1 trigger fired, including current WR 0% on 2) | Watch another 1-2 weeks; don't act on small samples |
| TODO 0524 produces clean basket Mode T REF > +95% AND 11h/12h have promise | Don't auto-act — live OOS > sim. But schedule a discussion. |

**Asymmetric-regime context**: G_narrow restored the asymmetric 5h/8h split (bull faster, bear slower for more confirmation in volatile regime). This matches the CLAUDE.md "longer horizon in bear" pattern that held under H75-snapshot too. The 2-day H75-fresh symmetric experiment (6h/6h) was abandoned by the May 21 G_narrow promote.

---

## ✅ TODO 0519 — G_narrow_d relaunch on Desktop (CLOSED)

**Status**: Run completed 2026-05-21 10:28 with Mode T REF +89.14%. Initially marked "shelved" but **subsequently promoted to live 2026-05-21 21:56** based on per-horizon win comparison (G's 5h winner ret +72.16% beat H75-fresh's 5h ret +53.76% even though aggregate Mode T REF was equivalent). G_narrow models still live as of today.

Full launch query, safeguards, banners, ETA, success table, and verdict moved to [ARCHIVED_LOG.md "TODO 0519 — G_narrow_d relaunch (CLOSED 2026-05-21)"](ARCHIVED_LOG.md).

Key spinoff: this run's per-phase timing exposed refine as 76% of HRST wall (17.7h of 23.4h) → spawned TODO 0522 (parallel speedup fork).

---

## 🔥 TODO 0525 — G_narrow_d HRST with extended grid + V2 top-10 + Optuna win_hi=350 (Desktop)

**Search anchor**: `TODO 0525`

**Status**: 🟢 **RUNNING on Desktop** — launched 2026-05-25 [time]. Output dir: `models_g_desktop_0525/` + `config_g_desktop_0525/`. ETA: ~9-9.5h.

**Hypothesis**: the May 24 narrow-grid rerun (TODO 0524) produced REF +80.56% vs LIVE's +91.01% on the SAME May 22 data because Mode V Optuna refine couldn't reach the high-window basin where LIVE's winners live. LIVE has 5h @ w=281 and 8h @ w=293 (both Refined). The narrow grid [72,100,150] seeded Optuna's TPE in a low-window basin; max refined window reached was w=168. With seeds at 250/300/350 + Optuna win_hi=350 + V2 widened to top-10, refine should anchor in the [220-350] region and produce winners comparable to LIVE.

**Three coordinated tweaks** (commit `e3c450c`, [crypto_trading_system_ed_g_narrow_d.py](crypto_trading_system_ed_g_narrow_d.py)):
1. `GRID_WINDOWS [72,100,150] → [72,100,150,200,250,300,350]` (line 4065)
2. `DOOHAN_SAVE_TOP_N 6 → 10` (line 4674) — V2 funnel widened proportional to grid (top-7% of 84 evals was too strict)
3. `win_hi 300 → 350` in serial refine (line 5281) AND K=5 parallel worker (line 8607)

**Validation gates** (run-level diagnostics, before any promotion talk):
| Test | Pass criterion | Fail = |
|---|---|---|
| ETH 5h refined #1 window | w ≥ 240 (LIVE basin) | seed-lottery fix didn't take — refine still anchored low |
| ETH 8h refined #1 window | w ≥ 250 (LIVE basin) | same as above |
| Mode V Step 1 backtest count | ≥ 8 of top-10 attempted | V2 expansion broke something |
| Mode V Step 2 refine count | 3/3 configs refined | parallel fork still healthy |

**Promotion gate** (same as TODO 0524):
| Mode T REF | Action |
|---|---|
| ≥ +95% on May 22 data | better than LIVE (+91.01%); discuss promotion (trader-flat-first per [[production-swap-when-flat]]) |
| +80% to +95% | comparable to LIVE — document but don't auto-promote |
| < +80% | grid extension didn't help; close arc, reconsider architecture |

**Setup notes**:
- Same May 22 snapshot as TODO 0524 (`data/_reliability_hrst_snapshot_laptop_20260522_0139`)
- Same horizons (5h, 6h, 8h, 11h, 12h) for direct A/B vs TODO 0524 numbers
- Production CSV / live regime config untouched (`--no-persist` + isolated `_0525` dirs)
- After HRST completes, run Mode T head-to-head against LIVE (same protocol as 2026-05-25 ~20:47 baseline test) to land the apples-to-apples REF

---

## ✅ TODO 0524 — Top-horizons clean rerun on fixed parallel fork (Desktop)

**Search anchor**: `TODO 0524`

**Status**: ✅ **DONE 2026-05-25 ~13:33**. Launched 2026-05-24 20:27 CEST after 3 failed pre-fix attempts. Mode H+R+S completed 2026-05-25 06:39 (~7.8h wall). Mode T failed once (regime config FileNotFoundError, fork patch added), failed once (live-seed overwrote Mode S verdict, fork patch added), succeeded 2026-05-25 13:33. Log: `logs/ed_v1_20260524_225312.log` (HRST), `logs/ed_v1_20260525_131704.log` (final Mode T).

**Outcome**: Mode T REF **+80.56%** on May 22 data. Apples-to-apples head-to-head test 2026-05-25 ~20:47 showed LIVE config (sma24>sma100 bull=5h@65% w=281 / bear=8h@65% w=293) on the SAME May 22 data → REF **+91.01%**. **LIVE wins by +10.45pp.** No promotion.

**Root cause analysis** (production CSV diff, [models_g_desktop_0524/crypto_ed_production_noprod.csv](models_g_desktop_0524/crypto_ed_production_noprod.csv) vs [models/crypto_ed_production.csv](models/crypto_ed_production.csv)):
- 4 of 5 ETH horizons in the new run shipped Mode D `Grid` candidates (w=150 RF+LGBM γ=0.999 f=10) instead of refined ones
- The 75-trial Optuna refine landed at w=168 / 132 / 95 for 5h and w=136 / 118 / 84 for 8h — never approached LIVE's w=281 / w=293
- TPE sampler is anchored on top-6 Mode D seeds; with all seeds in [72,100,150], refine stayed in that basin even though search range was nominally [50, 300]
- **Architecture flaw exposed**: G_narrow_d's narrow grid is data-state sensitive. May 15 data (production refine source) → seeds happened to surface a high-window basin. May 22 data → seeds clustered low. Same engine, different snapshot, different winner basin.

**What shipped from this run**:
1. **Parallel fork validated at HRST scale (~8× refine speedup, real shipped win)** — independent of the strategy verdict
2. Fork patches added: FIX #0 (grid + dirs + N_FEATURES_RANGE propagation), mkdir auto-create, sklearn warning filter, regime-config seed/preserve logic
3. New engine mode `RST` added to dispatcher (commit `d71a1be`) — for follow-up runs that don't need the H phase
4. Engine banner fix: Mode T/G/F no longer prints misleading `| 4h` (commit `59b1c1c`)
5. Apples-to-apples comparison protocol established — copy live config/models into isolated dir, switch env vars, run Mode T on the new snapshot

**Spinoff**: TODO 0525 (above) tests whether extending GRID_WINDOWS + V2 top-N + Optuna win_hi fixes the basin-coverage problem.

### Original plan and procedure (preserved for history)

Launched 2026-05-24 20:27 CEST after 3 failed pre-fix attempts. Replaces the 2205 Stage 2 9-12h run whose verdict was invalidated by the parallel-fork grid bug fixed 2026-05-24.

### Bug correction history (all landed before successful 20:27 launch)

The fork needed three rounds of fixes between 19:06 (first patch) and 20:27 (successful launch):

1. **19:06 — FIX #0 (grid + N_FEATURES_RANGE + output dirs)** — original bug found in user conversation. Engine grid `[10,13,17,25]`/`[0.999,0.997,0.995]` replaced by G's narrow `[10,15,20]`/`[0.999,0.996]`. `N_FEATURES_RANGE` 40/80 cap → 100. `G_NARROW_MODELS_DIR`/`G_NARROW_CONFIG_DIR` env vars actually wired.
2. **19:44 launch FAILED — `Cannot save file into a non-existent directory` at engine line 4736** — engine's `to_csv` calls assume `MODELS_DIR` exists. Fix landed: `os.makedirs(_d, exist_ok=True)` loop after the path redirect ([crypto_trading_system_ed_g_narrow_d_parallel.py:126-127](crypto_trading_system_ed_g_narrow_d_parallel.py#L126-L127)).
3. **20:14 launch FAILED — PowerShell `*>&1` wrapped LGBM feature-names warnings as `NativeCommandError`, breaking the redirect tee** — added module-level `warnings.filterwarnings("ignore", message="X does not have valid feature names...")` so every spawned worker inherits the filter on Windows re-import ([crypto_trading_system_ed_g_narrow_d_parallel.py:60-71](crypto_trading_system_ed_g_narrow_d_parallel.py#L60-L71)).
4. **20:27 launch SUCCEEDED** — all banners visible, Mode D progressing across 5 horizons in parallel.

**Why these 5 horizons**: The 5/6/7/8h May 20-21 G_narrow_d run + the (tainted) 9/10/11/12h May 22 parallel run together expose a "top 5" — drop the bottom 3 (7h weakest clean, 9h weakest tainted, 10h middling) and rerun the survivors on the now-fixed fork. Apples-to-apples on the SAME May 22 snapshot the Stage 2 9-12h run used → direct A/B test of the bug-fix isolation.

| Horizon | Source of inclusion | Current data point |
|---|---|---|
| 5h | live bull, +72% / 84% WR (clean) | RF+LGBM w=281 γ=0.998 f=12 conf=70% |
| 6h | clean +47% / 79% WR | RF+LGBM w=137 γ=0.997 f=18 |
| 8h | live bear, +46% / 90% WR (clean) | RF+LGBM w=293 γ=0.999 f=16 |
| 11h | tainted +47% / 81% WR — best long, NEEDS clean rerun | XGB+LGBM w=157 γ=0.997 f=18 conf=80% (suspect) |
| 12h | tainted +30-58% / 75% WR — high variance, NEEDS clean rerun | RF+LGBM w=142 γ=0.998 f=8 conf=85% (suspect) |

Skipped: 7h (weakest clean, +29%/72% WR), 9h (weakest tainted, +18%/67% WR), 10h (middling, +34%/82% WR).

### Parallel-fork bug fix (2026-05-24, applied to [crypto_trading_system_ed_g_narrow_d_parallel.py:73-101](crypto_trading_system_ed_g_narrow_d_parallel.py#L73-L101))

Pre-fix the fork imported G_narrow_d for K=5 patches but called `ENGINE.main()`, which read GRID/N_FEATURES/output-dir constants from ENGINE's namespace — not G's. Three categories patched in:

- **GRID propagation**: `ENGINE.GRID_COMBOS/WINDOWS/FEATURES/GAMMAS ← G.*` — fork now searches `RF+LGBM, XGB+LGBM × [72,100,150] × [10,15,20] × [0.999,0.996]` as intended (was `3 combos × [10,13,17,25] × [0.999,0.997,0.995]`).
- **N_FEATURES_RANGE**: `(4,40)/(4,80) → (4,100)` per G's "drop hard cap" design.
- **Output routing**: `MODELS_DIR/CONFIG_DIR/PRODUCTION_CSV/REGIME_CONFIG_PATH/RESUME_DIR ← G.*` — `G_NARROW_MODELS_DIR` / `G_NARROW_CONFIG_DIR` env vars now actually route writes (were silently ignored).

Banner prints all 5 resolved values + dirs on launch — verify in first 30 sec.

### One-shot launch query (copy-paste on Desktop, with `(venv)` activated)

```powershell
# ============================================================================
# TODO 0524 — Top-5 HRST on FIXED parallel fork (Desktop overnight)
# Reuses May 22 Laptop snapshot for direct A/B against tainted Stage 2 result.
# Production CSVs / configs / engine untouched (--no-persist).
# ============================================================================

cd G:\engine
git pull

# UTF-8 console — without this the engine's em-dashes / box-drawing chars
# render as cp1252 mojibake (— → ÔÇö, ─ → ÔöÇ) in console AND in the Tee log.
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

# Suppress sklearn LGBM "X does not have valid feature names" warning across
# ALL python processes including ProcessPool refine workers. The Python-level
# warnings.filterwarnings() in the fork only covers the main process; Mode V
# refine spawns workers that re-import sklearn fresh, so the filter has to
# live in an env var to propagate. Predictions are byte-identical with or
# without column names — sklearn is just complaining.
$env:PYTHONWARNINGS = "ignore:X does not have valid feature names:UserWarning"

# Keep Desktop awake for the full ~10-12h run
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0

# Use the SAME snapshot Stage 2 ran on, so 11h/12h tainted vs clean is a
# pure bug-fix delta (no data-state confound).
$snap = "data\_reliability_hrst_snapshot_laptop_20260522_0139"
if (-not (Test-Path $snap)) { Write-Host "SNAPSHOT MISSING — abort" -ForegroundColor Red; exit 1 }

$env:V2_DATA_SNAPSHOT = $snap
$env:RELIABILITY_K = "5"
$env:G_NARROW_MODELS_DIR = "models_g_desktop_0524"   # isolated, doesn't touch live or G_desktop
$env:G_NARROW_CONFIG_DIR = "config_g_desktop_0524"

$ts = Get-Date -Format "yyyyMMdd_HHmm"
$logfile = "logs\parallel_hrst_0524_desktop_$ts.log"

python crypto_trading_system_ed_g_narrow_d_parallel.py HRST "ETH," 5h,6h,8h,11h,12h --replay 1440 --no-persist --no-data-update --grid-tag G_NARROW_0524 *>&1 | Tee-Object -FilePath $logfile
```

### Startup banners to verify (first 30 sec — abort if any missing)

1. `[G_NARROW_D_SNAPSHOT] pd.read_csv redirected: data/<file> -> _reliability_hrst_snapshot_laptop_20260522_0139/<file>`
2. `[G_NARROW_D_PARALLEL] FIX #0 applied: ENGINE.GRID_* + N_FEATURES_RANGE + output dirs <- G`
3. `  combos=['RF+LGBM', 'XGB+LGBM']`
4. `  features=[10, 15, 20]`
5. `  gammas=[0.999, 0.996]`
6. `  n_features_range={...: (4, 100), ...: (4, 100)}`
7. `  models_dir=models_g_desktop_0524`
8. `  config_dir=config_g_desktop_0524`
9. `[G_NARROW_D_K5] _deku_eval_with_pruning patched (K=5 seeds=[42, 43, 44, 45, 46])`
10. `[G_NARROW_D_PARALLEL] patches applied (machine=DESKTOP, N_JOBS_PARALLEL=26):` + 3 numbered patch lines
11. `Refine: 3-worker outer × 5-thread K=5 inner`
12. `--no-persist ACTIVE`
13. `[refine-3worker] dispatching 3 configs across 3 workers (devices=['cpu', 'cpu', 'cpu'], K_parallel=5)`

### ETA (HRST 5,6,8,11,12h on parallel fork, Desktop)

| Phase | Time |
|---|---|
| Mode D × 5 horizons (no `--skip`, fresh grids) | ~55-60 min (5 × ~11 min) |
| Mode V × 5 horizons (parallel-fork refine) | ~8h (5 × ~95 min) |
| Mode R (regime backtest, C(5,2)=10 horizon pairs) | ~40 min |
| Mode S (joint sweep) | ~25 min |
| Mode T + G (rally-cooldown convergence, ~2 iters) | ~30-60 min |
| **Total** | **~10-12h** |

### Pass criteria

**(A) Bug-fix validation** (5h is the diagnostic):
| 5h refined winner matches May 20-21 baseline (RF+LGBM w≈281 γ≈0.998 f≈12) within ±10pp Mode V return | ✅ → fix confirmed |
| 5h diverges by >10pp | ❌ → patches missed something; abort downstream interpretation |

**(B) Long-horizon clean verdict** (11h/12h vs tainted Stage 2):
| 11h clean Mode V return ≥ 40% / WR ≥ 75% | promising — explore mixed regime (bull 5h, bear 11h) in Mode S |
| 11h clean Mode V return drops below 30% | tainted result was wider-grid inflation; close long-horizon arc |
| 12h clean Mode V return ≥ 40% | viable bear alternative; consider Mode S sweep including 12h |

**(C) Promotion gate** (joint Mode T REF):
| Mode T REF ≥ +95% on the new 5-horizon basket | better than live; discuss promotion (trader-flat-first per [[feedback-flat-before-promotion]]) |
| +80% to +95% | comparable to current G_narrow live (+89.14% on 4-horizon); document but don't auto-promote |
| < +80% | the 5-horizon basket loses to current 4-horizon live; close arc |

### Rollback / abort

`--no-persist` means live config is bulletproof — even if the run crashes or produces garbage, only `models_g_desktop_0524/` and `config_g_desktop_0524/` files are written. Live trader continues on current G_narrow models. To kill the run: `Ctrl+C` in the Desktop terminal.

---

## 🔥 TODO 0522 — Parallel refine speedup (G_narrow_d_parallel fork)

**Search anchor**: `TODO 0522`

**Status**: 🟢 STAGE 1 PASSED on Laptop 2026-05-22 ~00:26 CEST. **STAGE 2 PLANNED — launching on Desktop tonight for overnight run.**

**Why**: G_narrow_d rerun (TODO 0519) consumed 23h 22m wall, of which 17.7h was Mode V refine. Three identity-preserving fixes can compress refine 3-5× without changing model selection — same K=5 seeds, same Optuna trial sequence, same refined configs:
1. **Device routing** — `_g_factories_seeded` reads `G_PARALLEL_LGBM_DEVICE` env var (set per worker) instead of hardcoding GPU.
2. **Parallel K=5** — the `for seed in _G_SEEDS:` loop becomes `ThreadPoolExecutor(5)`. Same seeds, same median selection.
3. **3rd refine worker** — `max_workers=2 → 3`. All 3 configs run truly in parallel.

**Note**: H75 inlined K=5 into the production engine (`crypto_trading_system_ed.py` lines 8831-8909). The same speedup applies to ANY post-H75 HRST, not just G_narrow_d.

**Standalone fork**: [crypto_trading_system_ed_g_narrow_d_parallel.py](crypto_trading_system_ed_g_narrow_d_parallel.py) — imports g_narrow_d, applies the 3 monkey-patches, delegates to `ENGINE.main()`. Production CSVs / configs / engine source untouched. Workers re-import the fork on spawn (Windows), so patches propagate.

### Stage 1 results — Laptop 2026-05-22 ~00:26 (log `logs/parallel_smoke_20260522_0026.log`)

| Run | Outcome |
|---|---|
| First attempt 00:12 | ⚠️ 2 of 3 refine workers crashed with `BrokenProcessPool`. Cause: K=5 thread fan-out + LGBM `device='gpu'` → 5 concurrent CUDA contexts on a single GPU hard-crash. Only the CPU-assigned worker survived. |
| **Second attempt 00:26 (with CPU-only-when-K>1 fix)** | ✅ **3/3 configs refined**. Step 1: 6.8 min, Refine: 3.9 min, Step 3: 3.5 min, total ~14 min. All banners visible. Refined configs sane: `XGB+LGBM w=114h g=0.9958 f=14 apf=1.522`, `RF+LGBM w=104h g=0.9956 f=21 apf=1.503`, `RF+LGBM w=104h g=0.9975 f=21 apf=1.315`. |

**Fix applied in fork** ([crypto_trading_system_ed_g_narrow_d_parallel.py](crypto_trading_system_ed_g_narrow_d_parallel.py)): when `_PARALLEL_K > 1`, force all refine workers to CPU (`initial_devices = ['cpu'] * n_workers`). Override via `$env:G_FORCE_CPU_REFINE = "0"` if testing the GPU path on a machine where multi-context GPU works. Also added sequential retry-in-main-process for any worker that dies (poisoned-pool recovery).

**Speedup extrapolation from Stage 1**: 3.9 min refine for 5 trials × K=5 × 3 configs = 75 seed-evals. Linear scaling to full 75 trials × K=5 × 3 = 1125 seed-evals (15× more work) → **~58 min projected refine for 7h Mode V at full scale** vs **223 min** May 20-21 baseline = **3.8× speedup**. Full HRST projection: 23.4h → 10-11h, saves ~12-13h.

### Stage 2 launch — LAPTOP overnight 2026-05-22 (parallel-fork validation + long-horizon G_narrow_d research)

**Why combine**: tests 9-12h horizons (10h/12h lightly tested, 11h NEVER tested, 9h "borderline" per CLAUDE.md) AND validates the parallel fork at 4-horizon HRST scale. Same wall-time we'd burn anyway. Uses a **fresh snapshot of current live data (taken on Laptop just before launch)** so the data state is frozen and reproducible. Mode T REF comparable to G_narrow_d 5-8h's +89.14% and H75 LIVE's +89.41%, modulo the data-state delta (G_narrow_d 5-8h read the May 15 snapshot; this run reads ~May 22 data).

**Isolation from Desktop run** (live trader + any research process Desktop has going): output dirs are tagged `_laptop`, snapshot tagged `_laptop`, grid tag tagged `_LAPTOP`. Drive-sync will propagate the files to Desktop's view but no Desktop process should read those `_laptop`-tagged paths. Production CSV (`models/crypto_ed_production.csv`) and live regime config (`config/regime_config_ed.json`) are completely untouched.

```powershell
# ============================================================================
# TODO 0522 Stage 2 — HRST 9,10,11,12h on parallel fork (LAPTOP overnight)
# Fully isolated from Desktop: _laptop output dirs + _laptop snapshot + _LAPTOP grid tag
# ============================================================================

# 0. cd to Laptop's engine folder (adjust path if your local engine is elsewhere)
cd C:\Users\Alex\algo_trading\engine

# 1. Pull latest commit (parallel fork + this TODO + the launch query)
git pull

# 2. Keep Laptop awake for the full ~9-11h run
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0

# 3. Create a fresh snapshot of current live data (Laptop, just before launch).
#    Tagged "_laptop" so Desktop won't confuse it with its own snapshots.
$ts = Get-Date -Format "yyyyMMdd_HHmm"
$snap = "data\_reliability_hrst_snapshot_laptop_$ts"
New-Item -ItemType Directory -Path $snap -Force | Out-Null
New-Item -ItemType Directory -Path "$snap\macro_data" -Force | Out-Null
New-Item -ItemType Directory -Path "$snap\indices" -Force -ErrorAction SilentlyContinue | Out-Null
Get-ChildItem data\*.csv,data\*.pkl,data\*.json -ErrorAction SilentlyContinue | Copy-Item -Destination $snap -Force
Get-ChildItem data\macro_data\*.csv -ErrorAction SilentlyContinue | Copy-Item -Destination "$snap\macro_data" -Force
if (Test-Path data\indices) { Copy-Item data\indices\* "$snap\indices" -Force -ErrorAction SilentlyContinue }
Write-Host "Snapshot created on Laptop: $snap" -ForegroundColor Green
Get-ChildItem $snap | Format-Table Name,Length,LastWriteTime

# 4. Env vars — Laptop-isolated everything
$env:V2_DATA_SNAPSHOT = $snap
$env:RELIABILITY_K = "5"
$env:G_NARROW_MODELS_DIR = "models_g_laptop"      # ← isolated from Desktop's models_g_desktop\
$env:G_NARROW_CONFIG_DIR = "config_g_laptop"      # ← isolated from Desktop's config_g_desktop\

# 5. Launch — full HRST on 9,10,11,12h, fresh grids (no --skip), parallel refine.
#    --grid-tag G_NARROW_LONG_LAPTOP so the 9-12h grid CSVs are also isolated.
$logfile = "logs\parallel_hrst_long_laptop_$ts.log"
python crypto_trading_system_ed_g_narrow_d_parallel.py HRST "ETH," 9h,10h,11h,12h --replay 1440 --no-persist --no-data-update --grid-tag G_NARROW_LONG_LAPTOP *>&1 | Tee-Object -FilePath $logfile
```

**Verify snapshot is good** (the `Get-ChildItem $snap | Format-Table` line shows you the contents). Should include `eth_hourly_data.csv` dated today, all 6 `derivatives_*.csv`, `cross_asset.csv`, `macro_daily.csv`, etc. If anything missing → don't launch, debug the snapshot first.

**Verify isolation banner** (first 30 sec after launch):
```
[G_NARROW_D_PARALLEL] patches applied (machine=LAPTOP, N_JOBS_PARALLEL=14):
  resolved budget:   3 outer × 5 inner × 1 LGBM-thread = 15 OS threads (cap=14)
[G_NARROW_D_ISO] output dirs redirected: models=models_g_laptop config=config_g_laptop
```
If `machine=DESKTOP` or `models=models_g_desktop` appears → Ctrl+C, you're on the wrong machine OR env vars didn't take. Desktop's run is at risk of corruption — relaunch with fresh PowerShell session.

**Startup banners to verify (first 30 sec — abort if any missing)**:
1. `[G_NARROW_D_SNAPSHOT] pd.read_csv redirected: data/<file> -> _reliability_hrst_snapshot_desktop_20260515_154801/<file>`
2. `[G_NARROW_D_K5] _deku_eval_with_pruning patched (K=5 seeds=[42, 43, 44, 45, 46])`
3. `[H_STRICT_FAMILY_K5] _deku_eval_with_pruning patched (K=5 seeds=[42, 43, 44, 45, 46])`
4. `[G_NARROW_D_PARALLEL] patches applied:` followed by 3 `#1/#2/#3` lines
5. `CRYPTO TRADING SYSTEM ED — DESKTOP [G_NARROW_D_PARALLEL]`
6. `Refine: 3-worker outer × 5-thread K=5 inner`
7. `--no-persist ACTIVE` block
8. `ED: Mode V | ETH | 7h`

**Then while it runs, look for**:
- `[refine-3worker] dispatching 3 configs across 3 workers (devices=['cpu', 'cpu', 'cpu'], K_parallel=5)` ← critical, confirms CPU-only mode
- 3 separate `Refining #1 / #2 / #3 (LGBM=cpu): ...` blocks (interleaved is fine; output is captured-and-flushed per-worker)
- Final `[Refine: X.X min] (3-worker parallel + K=5 thread fan-out, 3/3 configs refined)` — `3/3` is the key

### ETA (HRST 9-12h on parallel fork, Desktop)

| Phase | Time | Notes |
|---|---|---|
| Mode D × 4 horizons (no existing grids, can't `--skip`) | ~50 min | 4 × ~12 min each |
| Mode V × 4 horizons (parallel-fork refine) | ~6-7h | 4 × ~95 min each (vs baseline ~250 min each = ~17h non-parallel) |
| Mode R (regime backtest, C(4,2)=6 horizon pairs) | ~30 min | |
| Mode S (joint sweep) | ~25 min | |
| Mode T + G (rally-cooldown convergence, usually 2 iters) | ~30-60 min | |
| **Total** | **~8-10h** | vs ~26-28h on non-parallel G_narrow_d |

### Pass criteria — TWO things to check

**(A) Parallel fork validation** (regardless of strategy verdict):
| Metric | Pass | Action |
|---|---|---|
| Each Mode V refine `[Refine: <90 min] ... 3/3 configs refined` | ✅ | Parallel fork works at HRST scale — promote patches to prod engine candidate |
| Any Mode V refine `<3/3 configs refined` | ❌ | Sequential-retry fallback fired or workers died — review log, do not promote |

**(B) Long-horizon strategy verdict** (this run's research goal):
| Mode T REF (full window) | vs G_narrow_d 5-8h +89.14% / H75 +89.41% | Verdict |
|---|---|---|
| ≥ +95% | beats short horizons by ≥5pp | **Very promising** — explore mixed 5-8 + 9-12 regime configs |
| +80% to +95% | comparable to short horizons (±5pp) | **Promising** — long horizons hold their own; worth a follow-up with fresh data |
| +50% to +80% | weaker than short horizons by 5-40pp | **Marginal** — confirms 5-8h is the sweet spot, but document the alpha source |
| < +50% | significantly weaker | Long horizons not viable in current regime — close arc, archive learning |

### Stage 3 — only if (A) passes AND (B) shows promise

If Mode T REF ≥ +80% AND parallel fork validated: follow up with a **mixed regime test** — bull from 5-8h (using H75 LIVE config) + bear from the best 9-12h pair from this run. Asymmetric regime configs have never been tested with mixed horizon families.

If only (A) passes (parallel fork works) but (B) shows long horizons are weak: **port the 3 patches to the prod engine `crypto_trading_system_ed.py`** as the standalone parallel speedup (independent of long-horizon outcome). Estimated production HRST: 23h → ~10h.

### Promotion criterion → engine

If Stage 3 passes with refined configs matching G_narrow_d shape (= identity confirmed), the SAME 3 patches can be ported to `crypto_trading_system_ed.py` directly (replace the H_STRICT_FAMILY K=5 block at lines 8831-8909 with the parallel version + 3-worker dispatcher). **Estimated saving on production HRST: ~13-14h per full run (23h → ~10h).** This is independent of G_narrow_d's strategy outcome — it's pure parallelization.

### Open questions to revisit post-Stage 2

- Does `G_PARALLEL_LGBM_THREADS=2` on Desktop help or hurt? (3 outer × 5 inner × 2 threads = 30 threads on 26 cores; some oversubscription, but LGBM may use threads efficiently)
- Is the GPU path salvageable on Desktop's RTX 4080 (which has more VRAM than Laptop's)? Smoke test showed Laptop GPU can't handle 5-concurrent-CUDA — Desktop might.
- Should we add an option to drop K=5 → K=3 when running benchmarks (proper variance check needs full K=5, but quick scans could use K=3 for 1.67× speedup)?

---

## 📅 TODO 0519B-G1 — deriv_oi_* re-enable A/B test (slipped from 2026-05-22; pending Desktop free)

**Search anchor**: `TODO 0519B-G1`

**Status**: 📅 PENDING. Originally scheduled for 2026-05-22 but Desktop was occupied: 2026-05-22 Stage 2 of TODO 0522 (Laptop ran HRST 9-12h, but Desktop also had separate work in progress), then 2026-05-24 TODO 0524 launched and is still running through ~2026-05-25 morning. Procedure below is ready to launch the moment Desktop frees up.

**Features**: `deriv_oi_chg1d`, `deriv_oi_chg3d`, `deriv_oi_zscore`. **Data source switched to Bybit 2026-05-24** — Binance OI API only returns ~30d and the daily download doesn't accumulate; Bybit public OI endpoint paginates back **208 days** at hourly resolution for ETHUSDT (5,000 rows, 2025-10-28 → 2026-05-24). One-shot backfill script: [tools/_bybit_oi_backfill_eth.py](tools/_bybit_oi_backfill_eth.py). Source-switch caveat: Bybit ETH OI is ~36% of Binance magnitude (Bybit is #2-3 perp exchange) but level corr 0.82 and engine features are relative (`chg1d`/`chg3d`/`zscore`) so scale washes out. Importance rankings from April-May on Binance OI may not transfer cleanly to Bybit.

**Workflow gotcha**: The live trader's data-refresh cycle (via `download_macro_data.py`) calls Binance OI and overwrites the column with a 30d window — wiping the Bybit backfill. Policy is to **re-run the backfill manually whenever needed** (script is idempotent, takes ~10 sec):
```powershell
python tools/_bybit_oi_backfill_eth.py            # idempotent: re-run anytime
python tools/_bybit_oi_backfill_eth.py --dry-run  # check coverage without saving
```
**Before launching G1 OFF baseline AND before launching G1 ON test**: run the backfill once to confirm 208d coverage is in place. If `--dry-run` reports `non-NaN OI rows = 721 (30d)`, the trader clobbered it — re-run without `--dry-run`. If it reports `5,000 (208d)`, you're good.

If G1 verdict is ≥ +5pp ON over OFF, the next step is integrating Bybit into `download_macro_data.py` (add a Bybit fetcher alongside or replacing the Binance one). If G1 fails, leave both the one-shot script and the backup CSV in place but don't modify the live download flow.

**Pre-flight**:
```powershell
cd G:\engine
copy config\disabled_features.json config\disabled_features_pre_deriv_oi_20260522.json
Select-String -Path config\disabled_features.json -Pattern "deriv_oi"   # expect 3 matches
```

**Step 1 — Baseline (deriv_oi OFF, as currently live)** ~7-9h:
```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmm"
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist *>&1 | Tee-Object -FilePath "logs\g1_deriv_oi_OFF_$ts.log"
copy models\crypto_ed_production_noprod.csv models\crypto_ed_production_noprod_deriv_oi_OFF.csv
copy config\regime_config_ed_noprod.json config\regime_config_ed_noprod_deriv_oi_OFF.json
```

**Step 2 — Remove deriv_oi_* from `config/disabled_features.json`** (manual edit, 3 lines: `deriv_oi_chg1d`, `deriv_oi_chg3d`, `deriv_oi_zscore`).

**Step 3 — Test (deriv_oi ON)** ~7-9h:
```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmm"
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist *>&1 | Tee-Object -FilePath "logs\g1_deriv_oi_ON_$ts.log"
copy models\crypto_ed_production_noprod.csv models\crypto_ed_production_noprod_deriv_oi_ON.csv
copy config\regime_config_ed_noprod.json config\regime_config_ed_noprod_deriv_oi_ON.json
```

**Step 4 — Restore baseline disabled list** (until verdict decided):
```powershell
copy config\disabled_features_pre_deriv_oi_20260522.json config\disabled_features.json
```

**Decision tree**: ON ≥ OFF + 5pp → re-enable permanently (keep removed). Within ±5pp → leave disabled. ON < OFF - 5pp → leave disabled (confirms quarantine).

**ETA**: ~16-18h total. Coordinate with H75 LIVE (uses `--no-persist`, trader safe; but Desktop fully booked).

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
