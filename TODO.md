## ⚡ TODO — Active work only

**Companion file**: [ARCHIVED_LOG.md](ARCHIVED_LOG.md) — historical audit trail, canonical scoreboard C01-C82, MERGED TOPICS, IDEA QUEUE drop-list (closed/shipped/STUB items with verdicts).

## 📊 At-a-glance — active TODO dashboard (2026-05-22)

| Pri | Item | When | Status |
|---|---|---|---|
| 📌 | **H75-fresh LIVE** — `sma24>sma100` / 6h@65% bull / 6h@65% bear (symmetric) | active since 2026-05-20 09:04 | running |
| 🔥 P1 | **H75-fresh LIVE OOS monitoring** — first 10 trades audit | window ~2026-06-03 (14 days) | 0/10 trades closed; flat |
| 🔥 P1 | **TODO 0524** — Top-5 HRST (5,6,8,11,12h) clean rerun on fixed parallel fork | Desktop overnight 2026-05-24 | 📅 PLANNED — parallel-fork grid bug fixed today, A/B vs tainted Stage 2 |
| 🔥 P1 | **TODO 2205** — Parallel refine speedup (G_narrow_d_parallel fork) + long-horizon G test (9-12h) | Combined Stage 2 on **LAPTOP** overnight 2026-05-22 (isolated from Desktop) | ⚠️ Stage 2 verdict INVALID — fork had grid bug (fixed 2026-05-24, see TODO 0524) |
| ✅ | **TODO 0519** — G_narrow_d relaunch on Desktop | completed 2026-05-20 → 2026-05-21 | DONE — Mode T REF +89.14%, converged iter 2, no STRICT winner |
| 🔥 P1 | **TODO 0519B-G1** — `deriv_oi_*` re-enable A/B test | Fri 2026-05-22 (today) | 📅 PLANNED — procedure ready |
| 📋 P2 | **TODO 0519B-G2** — orderbook + IV re-enable A/B test | 2026-06-18 (~30 days) | 📋 SCHEDULED — depends on G1 outcome |
| 🚀 P3 | **P4** — C14 vol-conditional triple-barrier retest | when capacity (~2.5h) | open |
| 🚀 P3 | **P5** — C11 VPIN at 5-min cadence | when capacity (~1 day eng) | open |
| 🚀 P3 | **P6** — C15 meta-labeling SOL/BTC | blocked (assets shelved, ~6h) | open |
| 🚀 P3 | **IDEA QUEUE Tier A** — Untested clean (C13-narrow, C54, C55, C58, C59) | research backlog | 5 items open |
| 🚀 P3 | **IDEA QUEUE Tier B** — V3-lit C60-C82 (16 of 23 patcher-ready) | research backlog | 23 items open |
| ⚪ P4 | **TODO 0519C** — CPCV HRST diagnostic | trigger-based re-run | available, no plan |
| ⚪ P4 | **Kalshi** — prediction-market data integration | needs API key + impl | backlog |

**Honest top-of-mind**: H75 is live, monitoring is passive. The two scheduled compute jobs (0519 tonight + 0519B-G1 Friday) are the only "act this week" items. Everything below P2 is wait-or-research.

---

**Layout (priority-ordered, top → bottom)**:
- 📌 LIVE STATE (always visible — current production config + rollback)
- 🔥 **P1** — Act this week (in-flight + imminent)
- 📋 **P2** — Scheduled next month
- 🚀 **P3** — Research backlog (when capacity allows)
- ⚪ **P4** — Low priority / Diagnostics / Backlog

---

## 📌 LIVE STATE — H75-fresh (promoted 2026-05-20 09:04 CEST)

Engine: `crypto_trading_system_ed.py` is the H_STRICT_FAMILY merge (K=5 multi-seed + REFINE_TRIALS=75 + strict `(combo, w)` dedup) — **unchanged from 2026-05-18 H75 promotion**, only config + production CSV swapped.

Detector: `sma24>sma100` ✅ (unchanged from prior H75). Bull = **6h@65%** RF+LGBM w=150 γ=0.999 10f. Bear = **6h@65%** (same model). **Symmetric regime** — both sides use the 6h winner. Shields OFF. Gates: rr8h≥2.0% OR rr12h≥2.0% cd=8h (bull); rr12h≥2.0% OR rr36h≥6.5% cd=6h (bear). min_sell_pnl=0%. max_hold=10h.

**Rollback ladder (one-command each, hot-reloads within 5 min):**

```powershell
# One level back — to H75-snapshot (live 2026-05-18 → 2026-05-20)
copy config\regime_config_ed_pre_H75fresh_20260520.json    config\regime_config_ed.json
copy models\crypto_ed_production_pre_H75fresh_20260520.csv models\crypto_ed_production.csv

# Two levels back — to pre-H75 (live before 2026-05-18)
copy config\regime_config_ed_pre_H75_20260518.json    config\regime_config_ed.json
copy models\crypto_ed_production_pre_H75_20260518.csv models\crypto_ed_production.csv
# Optional engine-layer rollback (only if reverting to the pre-H_STRICT_FAMILY engine):
copy crypto_trading_system_ed_pre_H75_20260518.py     crypto_trading_system_ed.py
```

Promotion source: Laptop H75 HRST fresh-data run (started 2026-05-18 23:38, completed 2026-05-20 03:03). Mode T REF +76.91% (converged iter 2).

Full promotion event in [ARCHIVED_LOG.md "PROMOTED 2026-05-20 ~09:04 CEST"](ARCHIVED_LOG.md). Prior H75 (snapshot) promotion event also preserved in archive.

---

# 🔥 P1 — Act this week

## ⚡ H75-fresh LIVE OOS monitoring — passive, ~1-2 weeks

**Search anchor**: `H75-OOS-MONITOR`

**Current state (2026-05-20 09:04)**: 🆕 **OOS window reset.** New live config (6h@65% symmetric) just promoted. Trader flat (last SELL fired at 02:00 CEST on prior config, PnL −0.53% / −$75.02). **0 closed round-trips** under new config. Window ends ~2026-06-03 (14 days).

**One closed trade under prior config (H75-snapshot, 2026-05-18 → 2026-05-20)** for historical reference:
- BUY 2026-05-19 17:05 UTC @ $2,121.13 (bear 8h@65% gate fired @ 65.30% conf)
- SELL 2026-05-20 02:00 CEST @ $2,109.79 — **PnL −0.53% / −$75.02 / 9h hold**
- That's 1 trade / WR 0% on prior config. Insufficient for any judgment; data now belongs to the snapshot-H75 audit trail.

**Rollback triggers under new config** (any one fires → discuss with user):
- Cumulative realized alpha < +5% after first 10 trades
- MaxDD exceeds −10% on new config alone (sim Mode T REF was +76.91%, no per-trade-DD sim available yet)
- First 10 trades WR < 50% (Laptop fresh Mode S showed 86% WR / 85 trades on 60d sim)
- Trade count vastly different from sim 85/60d (= 42/30d) — expect ~1.4 trades/day under symmetric 6h/6h

**Decision tree after 1-2 weeks**:
| Outcome | Action |
|---|---|
| H75-fresh holds OOS (≥+50% of sim alpha, no triggers) | Close TODO 0519 verdict, SHELVE G_narrow_d |
| H75-fresh underperforms (>2 triggers OR alpha<5%) | **Rollback to H75-snapshot** (one-command, see LIVE STATE rollback ladder). If that ALSO underperforms, two-level back to pre-H75 |
| Borderline (1 trigger fired) | Watch another 1-2 weeks |
| 0519 G shows G > H75-fresh by >+5pp on snapshot data | Don't auto-act; live OOS > sim |

**Symmetric-regime caveat**: this is the first live config with bull-horizon == bear-horizon (both 6h). Prior live configs all had asymmetric splits. Watch for unexpected behavior at regime-transition cycles — model decisions should be smooth but the "regime switch saves alpha" thesis hasn't been tested with symmetric configs.

---

## 📅 TODO 0519 — G_narrow_d relaunch on Desktop tonight (with safeguards)

**Search anchor**: `TODO 0519`

**Status**: 📅 PLANNED for tonight. Replaces failed 1805D (DIED Desktop 2026-05-19 08-10 am with no diagnostic log — terminal-only output). User saw "purple lines" (likely K=5 LGBM warnings, not crash signal). Step 1 saves ~5h vs full HRST by preserving G's 5h Mode V winner already in `models_g_desktop/crypto_ed_production_noprod.csv` row 41 (ETH XGB+LGBM w=279 γ=0.9979 11f +58.87% — G's signature wide-Optuna pattern).

### One-shot launch query (copy-paste on Desktop tonight)

```powershell
# ============================================================================
# TODO 0519 — G_narrow_d relaunch with safeguards (run on Desktop, Drive engine)
# ============================================================================
cd G:\engine

# 1. Keep Desktop awake for the full ~16h run
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0

# 2. Defensive backup of row 41 (G's 5h Mode V winner)
copy models_g_desktop\crypto_ed_production_noprod.csv models_g_desktop\crypto_ed_production_noprod_5h_only_pre_0519.csv

# 3. Set env vars (snapshot + K=5 + isolated output dirs)
$env:V2_DATA_SNAPSHOT = "data\_reliability_hrst_snapshot_desktop_20260515_154801"
$env:RELIABILITY_K = "5"
$env:G_NARROW_MODELS_DIR = "models_g_desktop"
$env:G_NARROW_CONFIG_DIR = "config_g_desktop"

# 4. Launch Mode H for 6h, 7h, 8h only — captures stdout AND stderr to disk
$ts = Get-Date -Format "yyyyMMdd_HHmm"
$logfile = "logs\g_relaunch_0519_$ts.log"
python crypto_trading_system_ed_g_narrow_d.py H "ETH," 6h,7h,8h --skip --replay 1440 --no-persist --no-data-update --grid-tag G_NARROW_D *>&1 | Tee-Object -FilePath $logfile
```

**Critical safeguards** (non-negotiable):
- `Tee-Object *>&1` — captures stdout+stderr to disk so any crash IS diagnosable this time
- `powercfg standby-timeout 0` — kills Modern Standby (suspect #2 from this morning)
- `"ETH,"` trailing comma — older fork's `endswith('h')` parser bug; without comma → all 9 assets load
- `--no-data-update` — snapshot env var only redirects READS; without this flag, downloads write to LIVE
- `--no-persist` — production CSV untouched
- `--skip` — 6h/7h Mode D skipped (grids exist), only 8h Mode D runs (~12 min)
- `--grid-tag G_NARROW_D` — without it engine uses untagged grids (production B's)
- Both env vars `G_NARROW_MODELS_DIR` + `G_NARROW_CONFIG_DIR` set — confirm `[G_NARROW_D_ISO]` banner

### Startup banners to verify (first 30 sec)

1. `[G_NARROW_D_SNAPSHOT] pd.read_csv redirected: ...`
2. `[G_NARROW_D_ISO] output dirs redirected: models=models_g_desktop config=config_g_desktop`
3. `[G_NARROW_D_K5] _deku_eval_with_pruning patched (K=5 seeds=[42, 43, 44, 45, 46])`
4. `--no-persist ACTIVE` block listing `models_g_desktop/crypto_ed_production_noprod.csv`
5. `ED: Mode H | ETH | 6h,7h,8h`
6. `Mode D results already exist for ETH 6h — skipping D (--skip)` (6h/7h) + `EXHAUSTIVE GRID: ETH 8h — 36 evals` (only 8h)

**Abort if any banner missing.** Ctrl+C, fix env vars, relaunch.

### ETA

| Phase | Time |
|---|---|
| Mode D × 1 (8h fresh) | ~12 min |
| Mode V × 3 horizons @ K=5 × 75 trials wider-Optuna | ~15h |
| **Step 1 total** | **~15-16h** |
| Step 2: R + S + T+G (separate launches) | ~1.5h |
| **Grand total** | **~17-17.5h** |

Done midday-late afternoon 2026-05-20.

### Step 2 — After Mode H finishes (~1.5h sequential)

```powershell
# Same PowerShell session (env vars still set)

# Verify 5h row 41 is still in noprod CSV
Get-Content models_g_desktop\crypto_ed_production_noprod.csv | Select-String -Pattern "^ETH,.*,5,0\."
# Expected: ETH,279,XGB+LGBM,...,5,0.9979,Refined
# If MISSING: copy models_g_desktop\crypto_ed_production_noprod_5h_only_pre_0519.csv models_g_desktop\crypto_ed_production_noprod.csv

# Mode R then S then T (T chains Mode G automatically)
python crypto_trading_system_ed_g_narrow_d.py R "ETH," 5,6,7,8h --replay 1440 --no-persist --no-data-update --grid-tag G_NARROW_D *>&1 | Tee-Object -FilePath "logs\g_relaunch_0519_R_$(Get-Date -Format yyyyMMdd_HHmm).log"
python crypto_trading_system_ed_g_narrow_d.py S "ETH," 5,6,7,8h --replay 1440 --no-persist --no-data-update --grid-tag G_NARROW_D *>&1 | Tee-Object -FilePath "logs\g_relaunch_0519_S_$(Get-Date -Format yyyyMMdd_HHmm).log"
python crypto_trading_system_ed_g_narrow_d.py T "ETH," --replay 1440 --no-persist --no-data-update --grid-tag G_NARROW_D *>&1 | Tee-Object -FilePath "logs\g_relaunch_0519_T_$(Get-Date -Format yyyyMMdd_HHmm).log"
```

### If it crashes again

The `g_relaunch_0519_*.log` file will have the crash trace this time. Diagnose:
- CUDA / GPU OOM in log → force CPU LGBM (`hardware_config.py` device='cpu'), ETA 25-30h
- MemoryError / Python OOM → reduce K=5 to K=3 (`$env:RELIABILITY_K = "3"`)
- Modern Standby (log cuts mid-line, no error) → powercfg fix should have prevented this
- Hard exit → terminal got closed; reconsider screen-locked PS

### Success criteria → 3-way verdict

| Variant | Mode T total | vs H75 | Decision |
|---|---|---|---|
| B (pre-H75 production) | +89.41% | — | baseline |
| H75 (current live) | from promotion HRST log | — | live |
| G_narrow_d (this run) | TBD | TBD | this TODO |

| Outcome | Action |
|---|---|
| G > H75 by >+5pp | Possible promotion regret. Discuss rollback (trader-flat-first per [[feedback-production-swap-when-flat]]) |
| G ≈ H75 (±5pp) | Both valid; keep H75 live (simpler code) |
| G < H75 by >5pp | Confirms H75 promotion was correct; SHELVE G_narrow_d |
| Crashes with diagnosable log | Implement specific fix; relaunch as TODO 0520 |

**Rerun verdict (2026-05-20 11:05 → 2026-05-21 10:28, wall ~23h 22m, log `logs/ed_v1_20260520_110556.log`):** completed full 4-horizon HRST + R + S + T. Mode T converged at iteration 2 unchanged, baseline H1=+25.30% / H2=+51.02% / **REF=+89.14%** vs B's +89.41% — within ±0.3pp of B baseline, so G ≈ B per the success table. **G_narrow_d shelved** — no promotion regret signal. The bigger win from this run came from the per-phase timing data: **refine is 76% of HRST wall (17.7h of 23.4h)**, K=5 multi-seed runs the 5 seeds sequentially per trial, and `_g_factories_seeded` hardcodes `device='gpu'` which breaks the hybrid GPU+CPU dispatcher. → spawned TODO 2205 (below).

---

## 🔥 TODO 0524 — Top-horizons clean rerun on fixed parallel fork (Desktop)

**Search anchor**: `TODO 0524`

**Status**: 📅 PLANNED for Desktop overnight. Replaces the 2205 Stage 2 9-12h run whose verdict is invalidated by the parallel-fork grid bug fixed 2026-05-24.

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

## 🔥 TODO 2205 — Parallel refine speedup (G_narrow_d_parallel fork)

**Search anchor**: `TODO 2205`

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
# TODO 2205 Stage 2 — HRST 9,10,11,12h on parallel fork (LAPTOP overnight)
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

## 📅 TODO 0519B-G1 — deriv_oi_* re-enable A/B test (Fri 2026-05-22 — today)

**Search anchor**: `TODO 0519B-G1`

**Features**: `deriv_oi_chg1d`, `deriv_oi_chg3d`, `deriv_oi_zscore`. ~63d Binance OI history + 72h warmup buffer.

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
