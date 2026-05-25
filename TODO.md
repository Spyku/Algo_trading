## ⚡ TODO — Active work only

**Companion file**: [ARCHIVED_LOG.md](ARCHIVED_LOG.md) — historical audit trail, canonical scoreboard C01-C82, MERGED TOPICS, IDEA QUEUE drop-list (closed/shipped/STUB items with verdicts).

## 📊 At-a-glance — active TODO dashboard (2026-05-25)

| Pri | Item | When | Status |
|---|---|---|---|
| 📌 | **G_narrow LIVE** (CSV+config since 2026-05-21 21:56) running on **H75 engine** (K=5 + 75 trial refine) — `sma24>sma100` / bull 5h@65% / bear 8h@65% | active since 2026-05-21 21:56 (G_narrow promote); engine unchanged from 2026-05-18 H75 | running |
| 🔥 P1 | **TODO 0525** — G_narrow_d HRST with extended grid + V2 top-10 + Optuna win_hi=350 | Desktop launched 2026-05-25 [time] — output dir `models_g_desktop_0525/` | 🟢 RUNNING — testing whether the seed-lottery fix lets refine reach LIVE-grade w=281/293 basins |
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
