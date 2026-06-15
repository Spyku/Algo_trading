# Algo Trading Engine

Automated ML trading system for **crypto** (ETH live; BTC, SOL, LINK, XRP standby) and **index CFDs** (DAX, S&P 500 — separate system, see [cfd/](cfd/)). Generates hourly BUY/SELL/HOLD signals using ensemble ML models with walk-forward validation, temporal decay sample weighting, and embargo-corrected labels. Executes trades on Revolut X via Ed25519-signed API.

**Production system:** Ed V2 — regime-switching trading (bull/bear detection with dynamic horizon selection), maker-order pricing at `bid+0.01` with `post_only` for 0% Revolut X fees.

---

## Current production state

> ⚠️ **SUPERSEDED — current live (2026-06-16):** detector **`sma48>sma100`** (swapped from `tsmom_672h`, which was too slow on sharp reversals — missed the 06-15 rally), **bull 8h@90% shield ON / bear 5h@65% shield OFF**, rally-cd ON, min_sell 0, maker ON. `ENABLED_DETECTORS` expanded 2→6 after a 45-detector dedup audit. See [TODO.md](TODO.md) `📌 LIVE STATE` (single source of truth) for full config + rollback. **Everything below this line is historical** (the engine moved to **FAYE** on 2026-05-31; live inference now runs `compute_signal_core()` in `crypto_trading_system_faye.py`).

### Historical snapshot (2026-05-27, pre-FAYE)

Live config: **G_narrow models on H75 engine** with **macro-cache fix** (TODO 0527). The H75 H_STRICT_FAMILY merge engine has been unchanged since 2026-05-18; the per-horizon models + regime config were swapped to G_narrow_d's output on 2026-05-21 21:56; the `_load_macro_csv` mtime fix was patched 2026-05-27 11:22.

- **Engine code**: `crypto_trading_system_ed.py` — K=5 multi-seed denoising, REFINE_TRIALS=75, strict `(combo, w)` diversity dedup. **2026-05-27 patch**: `_load_macro_csv` now mtime-aware (lines 1077-1110) — was caching macro/cross-asset/sentiment/onchain data at trader startup and never refreshing, freezing 11+ high-importance time-shifted features. Root cause of the live-vs-backtest WR gap (50.9% live vs ~85% backtest).
- **Trader's models + regime config** (since 2026-05-21 21:56): G_narrow_d's HRST output
- **Asset**: ETH only (BTC/SOL/LINK/XRP/BNB `enabled: false`; XRP removed from data pipeline 2026-05-23)
- **Detector**: `sma24>sma100`. Bull = **5h@65%** RF+LGBM w=281 γ=0.9981 12f. Bear = **8h@65%** RF+LGBM w=293 γ=0.9990 16f
- **Rally cooldown OFF** (manually toggled 2026-05-23 22:21). **Shields OFF**, min_sell_pnl=0%, max_hold=10h
- **Signal log schema** (renamed 2026-05-27): `h_1/sig_1/conf_1` (bull) + `h_2/sig_2/conf_2` (bear) — replaces the old `sig_4h/sig_8h` which broke when actual model horizon differed from hardcoded constants
- **Validation**: same-process test + shadow mode in-process at 100% match (core math = live math, max conf delta 0.04pp). PIT validator using oldest-wins archeology merge at 66.7% (ceiling — limited by retrospective inability to reproduce pre-fix cache state)
- **Monitoring**: live WR over next 3-5 days expected to trend toward backtest WR; shadow mode running continuously as correctness gate

For the current live OOS monitoring, scheduled runs, and research backlog → see [TODO.md](TODO.md). For closed research arcs and audit trail → see [ARCHIVED_LOG.md](ARCHIVED_LOG.md).

---

## File hygiene — where to find what

| File | Purpose | Updated |
|---|---|---|
| **README.md** (this file) | Project overview, install, commands — evergreen | Rarely |
| **[CLAUDE.md](CLAUDE.md)** | Stable engine reference: feature grades, runtime measurements, architecture, **18 critical rules** | When stable reference changes |
| **[TODO.md](TODO.md)** | Active work, priority queue, idea backlog, live state snapshot | Daily during active research |
| **[ARCHIVED_LOG.md](ARCHIVED_LOG.md)** | Historical audit trail, canonical scoreboard (C01-C82), MERGED TOPICS, IDEA QUEUE drop-list | When work closes |

Rule: time-sensitive items belong in TODO.md while active, then ARCHIVED_LOG.md when closed. README is evergreen.

---

## Quick Start

```powershell
# Activate venv first (required)
# Desktop:  C:\algo_trading\venv\Scripts\activate.bat
# Laptop:   C:\Users\Alex\algo_trading\venv\Scripts\activate.bat

# === Live trading (Ed V2 — production) ===
start_ed_v2.bat                                     # Auto-restart wrapper (recommended)
python crypto_revolut_ed_v2.py --loop               # Direct (no auto-restart)
python crypto_revolut_ed_v2.py --dry-run --loop     # Signals only, no trades
python crypto_revolut_ed_v2.py --status             # Show positions

# === Optimizer bot (remote optimization via Telegram) ===
start_optimizer.bat                                 # Auto-restart wrapper
python crypto_optimizer_bot.py                      # Direct

# === Most common HRST (full pipeline: H → R → S → T with rally-cooldown chain) ===
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist

# CLI normalizes the shortcuts:
#   --1440 → --replay 1440
#   --nopersist → --no-persist
```

### Engine modes cheat-sheet

```powershell
# Arguments are order-independent: MODE, ASSETS, HORIZONS can appear in any order.
python crypto_trading_system_ed.py P ETH 6h               # Mode P — PySR feature discovery
python crypto_trading_system_ed.py D ETH 6h               # Mode D — exhaustive grid only
python crypto_trading_system_ed.py V ETH 6h               # Mode V — validate + Optuna refine
python crypto_trading_system_ed.py DV ETH 6h              # Mode DV — D then V
python crypto_trading_system_ed.py H ETH 5,6,7,8h         # Mode H — horizon sweep (D+V per horizon)
python crypto_trading_system_ed.py R ETH 5,6,7,8h         # Mode R — regime backtest (all detectors)
python crypto_trading_system_ed.py HRS ETH 5,6,7,8h       # Pipeline: H → R → S
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h      # Pipeline: H → R → S → T (chains G)
python crypto_trading_system_ed.py F                      # Mode F — feature trim audit
python crypto_trading_system_ed.py --help                 # Full usage
```

### Research flags (preserve production)

```powershell
# --no-persist redirects ALL writes to *_noprod.* files. Safe to run alongside live trader.
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist

# --no-data-update skips macro/onchain/derivatives download — useful when iterating
# on a fixed snapshot (combine with V2_DATA_SNAPSHOT env var for read-redirection).
```

### Telegram commands (Ed V2 trader)

`/stop` `/status` `/pause` `/resume` `/balance` `/sync` `/conf` `/config` `/setup` `/help` `/chart BTC` `/regime` `/gate [ASSET on|off|clear]`

---

## Setup

### Hardware

One shared engine folder synced via Google Drive — both machines use the same code, data, and models. Only the venv is local.

| Machine | Engine path | Venv | CPU | GPU |
|---|---|---|---|---|
| Desktop (primary) | `G:\engine\` | `C:\algo_trading\venv\` | i7-14700KF | RTX 4080 |
| Laptop | `G:\Autres ordinateurs\My laptop\engine\` | `C:\Users\Alex\algo_trading\venv\` | 16 cores | RTX 3070 Ti |
| Yoga (lightweight) | `G:\Autres ordinateurs\Yoga\engine\` *(path TBC)* | *TBD* | CPU-only | none — falls back to CPU LGBM |

`hardware_config.py` auto-detects Desktop (26 workers) / Laptop (14 workers) / Yoga (2 workers, CPU-only). LGBM uses GPU on Desktop+Laptop, CPU on Yoga.

### Installation

```powershell
# Desktop
python -m venv C:\algo_trading\venv
C:\algo_trading\venv\Scripts\activate.bat
pip install -r G:\engine\requirements.txt

# Laptop
python -m venv C:\Users\Alex\algo_trading\venv
C:\Users\Alex\algo_trading\venv\Scripts\activate.bat
pip install -r "G:\Autres ordinateurs\My laptop\engine\requirements.txt"
```

### Key dependencies

`pandas`, `numpy`, `scikit-learn`, `lightgbm` (GPU), `xgboost`, `optuna`, `ccxt`, `yfinance`, `pynacl`, `cryptography`, `matplotlib`, `joblib`, `pysr`

### Config (not in git — `config/` is `.gitignore`d)

```
config/regime_config_ed.json       # Per-asset regime detector + bull/bear params
config/revolut_x_config.json       # Revolut X API key
config/private.pem                 # Ed25519 signing key
config/telegram_config.json        # Trader bot token
config/telegram_optimizer_config.json  # Optimizer bot token (separate)
config/disabled_features.json      # Sparse-history feature quarantine
```

---

## Architecture

```
Production file chain:

crypto_trading_system_ed.py            (Single-file engine — Modes P/D/V/H/S/R/HRS/HRST/T/G/F)
  └── hardware_config.py

crypto_revolut_ed_v2.py                (Auto-trader — reads regime_config_ed.json)
  └── crypto_live_trader_ed.py         (Regime-aware signal generation)
        └── crypto_trading_system_ed.py

crypto_optimizer_bot.py                (Telegram bot — spawns engine as subprocess)
  └── crypto_trading_system_ed.py

Index CFDs (separate, see cfd/):
cfd/ib_auto_trader.py                  (DAX CFD trader — Broly 1.2)
cfd/ib_auto_trader_test.py             (S&P 500 CFD overnight)
```

### Core concepts (one-line summaries — full detail in CLAUDE.md)

| Concept | Detail |
|---|---|
| **Regime switching** | Bull/bear detector picks model+horizon+confidence per cycle |
| **Variable horizons** | Each regime has its own optimal prediction horizon (typically 5-8h) |
| **No model persistence** | Retrains from scratch every prediction — no `.pkl` files |
| **Temporal decay** | Sample weights `w_i = gamma^age`; gamma=0.995 → half-life ~6 days |
| **6-month data cap** | `MAX_DIAG_HOURS=4320` — prevents stale data dilution |
| **Fee-aware labels** | `label=1` iff future return > 2× trading fee (0.22% with slippage) |
| **Embargo** | `EMBARGO_CANDLES=horizon` for backtesting only — **never in live signals** |
| **Walk-forward** | Train on last `window` hours → predict next candle → step forward |
| **K=5 multi-seed** | Each Optuna trial averaged over 5 random seeds (denoise) |
| **PySR symbolic** | Offline feature discovery on months 12→6 ago window (anti-leakage) |

---

## Two-Machine Setup — Liveness Rule

The engine folder is shared via Google Drive. **Long-running jobs can run on either Desktop OR Laptop and write to the same shared files.** A `python.exe` on the OTHER machine is invisible to your local `tasklist` — but its file writes show up here within seconds.

**Decision rule for "is the job alive?":** check **file mtimes**, not local process list.

| File | Updates during | Cadence |
|---|---|---|
| `logs/ed_v1_<latest>.log` | All HRST phases | Every few seconds — primary signal |
| `models/crypto_ed_production_noprod.csv` | Mode V/H/S/T writes | Every Mode T iter (~10-20 min) |
| `config/regime_config_ed_noprod.json` | Mode S/T writes | Same |
| `models/crypto_ed_grid_ETH_<h>h.csv` | Mode D | Once per horizon |

- Newest mtime **within last 2 min** → alive on some machine, do not relaunch.
- Newest mtime **>10 min stale** + no local python → likely dead, but **ASK before declaring it**.
- Between 2-10 min: ambiguous (Mode T sometimes hangs on heavy iter) — wait, do not relaunch.

**Mode R exception**: Mode R is in-memory only and can sit silent for **30-120 min** on a long replay. If the previous activity was Mode D grid writes all done and Mode T hasn't written yet, the job is most likely in Mode R — wait, do not relaunch.

**Always ask "is it running on the desktop/laptop?" before assuming a job is stopped.** Full rule: CLAUDE.md rule 16.

---

## Critical safety rules (excerpt — full list in [CLAUDE.md](CLAUDE.md#critical-rules))

1. **Never modify `crypto_trading_system_ed.py` without testing first.** It's production. The live trader imports from it.
2. **CSV merge logic**: writes to production CSV filter by BOTH coin AND horizon — preserves other rows.
3. **NEVER add embargo in live signal generation.** Embargo is for backtesting only (per Lopez de Prado).
4. **Leakage check before any production promotion.** PySR `discovery_method` must equal `"historical"`.
5. **Verify trader is flat (state=cash) before any config/prod CSV swap.** User can override per-promotion.
6. **Clock drift correction uses NTP, not echo-back.** 409 responses echo back STALE request timestamp.
7. **Two-machine setup → don't declare a job dead from local process list alone.** Use file mtimes; ask first.
8. **Default to bare `python tools/<script>.py [args]` in user-facing commands.** Venv is already activated.

Full 18-rule list with rationale: [CLAUDE.md "Critical Rules"](CLAUDE.md).

---

## Common workflows

### Run a research HRST (no production impact)

```powershell
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist
# Writes to crypto_ed_production_noprod.csv + regime_config_ed_noprod.json
# Compare against current live; promote only with --no-persist files OFF and trader flat
```

### Promote a research result to live (when trader is flat)

```powershell
# 1. Backup current live
copy config\regime_config_ed.json    config\regime_config_ed_pre_<TAG>.json
copy models\crypto_ed_production.csv models\crypto_ed_production_pre_<TAG>.csv

# 2. Copy research result over live (after vetting)
copy models\crypto_ed_production_noprod.csv models\crypto_ed_production.csv
copy config\regime_config_ed_noprod.json    config\regime_config_ed.json

# 3. Trader hot-reloads within 5 min — no restart needed
```

### Rollback live to previous config

```powershell
copy config\regime_config_ed_pre_<TAG>.json    config\regime_config_ed.json
copy models\crypto_ed_production_pre_<TAG>.csv models\crypto_ed_production.csv
```

### Update macro / onchain / derivatives data

```powershell
python download_macro_data.py        # Refresh VIX, DXY, S&P500, Fear&Greed, on-chain, derivatives
```

---

## Version history

| Date | Milestone |
|---|---|
| **2026-06-09** | **GB+LGBM trialed & rejected; removed from FAYE grid.** A step=36 model-combo screen ranked GB+LGBM #1, but the real-engine 2mo regime backtest (`tools/bt_lgbm_tune_8h.py`) had it LOSE to the live RF+LGBM (+43.5% vs +51.2%) and the LGBM-regularization "win" reversed under the live conf gate — regularization erased the model's conviction (`tools/diag_lgbm_proba_spread.py`: proba IQR 0.84→0.61, held long through the drawdown). Removed from FAYE `GRID_COMBOS`; added a default-safe `LGBM_MIN_CHILD`/`LGBM_REG_LAMBDA` env hook for real-engine param tests. |
| **2026-05-31** | **FAYE promoted to production.** ETH bull/bear swapped to FAYE H/RST winners (later converged to **8h/8h** — bull 8h@65 shield-off / bear 8h@70 shield-on, `tsmom_672h`; see [TODO.md](TODO.md) LIVE STATE). First live config generated by `crypto_trading_system_faye.py`. |
| **2026-05-30** | **FAYE single-file consolidation shipped + post-FAYE archive cleanup.** Built `crypto_trading_system_faye.py` (~9100 lines, 8 commits `8c122ef` → `4ab34d5`) — collapses Ed v3's 4-layer monkey-patch chain (v3 → parallel_nearlive → step6_nearlive → g_narrow_d → ed) into one file with **ZERO monkey-patches, every previously-patched feature first-class native code**. 7 phases shipped: P1 isolated FAYE paths (`models_faye/`, `config_faye/`, `crypto_faye_production.csv`, `regime_config_faye.json`); P2 inlined `_mean_last_10_fill` + NEAR_LIVE defaults (`step=1`, `signal_mode='ternary'`, `na_policy='mean_last_10'`, `embargo=horizon`); P3 native K=5 multi-seed median ensemble (replaces `reliability_multi_seed.py` patch); P4 3-worker hybrid GPU+CPU refine canonical; P5a `run_mode_v/s/t_parallel`→canonical; P5b 8-worker Mode D `ProcessPool` grid dispatcher inlined (replaces v3's `inspect.currentframe()` hack); P6 `tools/smoke_test_faye.py` 38-check verification (passes); P7 architecture docstring. **Parity test**: 30/30 ETH hours evaluated, 0 errors, 73.3% direct MATCH, **0 real BUY↔SELL flips** (all 8 DIFFs are HOLD-threshold boundary cases). The original "bug between live trader and crypto trading engine" is gone. Same session: archived 36 items into `ARCHIVED/2026-05-30_post_faye_cleanup/` (commit `1fce7f8`) — 8 variant scripts (`_cdar`, `_cvar`, `_cpcv`, `_robust`, `_h_strict_family`, `_noprod`, `_pre_macro_cache_fix`, `_launch_h_strict_family.bat`), 14 variant-driving tools, 14 old `models_g_desktop_*/` + `config_g_desktop_*/` snapshot dirs, plus `CLAUDE_NEW.md` + `TODO_TEST.md`. Root file count 38→29, root dir count 35→21. NOT archived (still load-bearing): the v3 chain (`_g_narrow_d*`, `_step6*`, `signal_core*`) — that's the next archival round once FAYE replaces v3 in production. |
| **2026-05-28/29** | **v3 fork shipped + archive cleanup.** New `crypto_trading_system_ed_g_narrow_d_parallel_nearlive_v3.py` (509 lines, commit `f688e0e`) adds TRUE Mode D outer-loop parallelization via dispatcher pattern — `ProcessPoolExecutor(max_workers=8)` instead of serial grid, K=5 ThreadPool inside each worker preserved (40 concurrent LGBM fits vs 5 in v2). Engine's 545-line `run_mode_d_optuna` left untouched; v3 monkey-patches `_get_deku_diagnostic_models` to capture grid-setup state via `inspect.currentframe().f_back`, then routes `_deku_eval_with_pruning` calls through a dispatcher cache. Prints EVERY eval + writes full per-eval CSV with K=5 seed-by-seed metrics. Same session: archived 4 superseded engine forks (`pre_H75_20260518.py`, `pre_cli_fix_20260518.py`, `h75_wide.py`, `g_narrow_d_parallel.py`) → `ARCHIVED/2026-05-28_v3_cleanup/`. Plus `tools/merge_pysr_old_new.py` (commit `c753f46`) to keep both OLD (April) and NEW (May 28 Laptop) PySR features as candidates: OLD in slots `pysr_1..5`, NEW in `pysr_6..10`, LGBM gain decides via top-N feature selection. |
| **2026-05-24** | **Parallel fork bug fixes landed; TODO 0524 launched.** `crypto_trading_system_ed_g_narrow_d_parallel.py` had a namespace bug — called `ENGINE.main()` which read `GRID_*`/`N_FEATURES_RANGE`/output-dir constants from ENGINE instead of G_narrow_d. Stage 2 9-12h Laptop run from 2026-05-22 invalidated (was running engine's wider grid, not G's narrow). FIX #0 patches all constants over. Two follow-on fixes: `os.makedirs` for output dirs, LGBM warnings filter for PowerShell tee. Top-5 HRST (5,6,8,11,12h) launched Desktop 20:27 to A/B against the tainted Stage 2 result on the same May 22 snapshot. XRP also removed from trader data pipeline (silent crash during derivatives download). |
| **2026-05-21** | **G_narrow models promoted to live.** ETH bull/bear models swapped from H75-fresh (May 20 09:04) to G_narrow_d's HRST output — bull 5h@65% RF+LGBM, bear 8h@65% RF+LGBM. Engine code unchanged (still H75 H_STRICT_FAMILY). Live config switched from symmetric (6h/6h) back to asymmetric horizon split. |
| **2026-05-20** | **H75-fresh promoted.** Laptop H75 HRST fresh-data rerun completed 03:03 CEST after 28h wall. Mode T REF +76.91%. Bull 6h@65% / bear 6h@65% symmetric (first live config with bull-horizon == bear-horizon). |
| **2026-05-18** | **H75 promoted live.** H_STRICT_FAMILY merge: K=5 multi-seed denoising + REFINE_TRIALS=75 + strict `(combo, w)` dedup. Detector switched `tsmom_672h` → `sma24>sma100`. Bull horizon 6h → 5h@75%; bear 8h@65% kept. CLAUDE.md / TODO.md / ARCHIVED_LOG.md three-file split same week. |
| **2026-05-15** | **5-variant reliability test passed.** B_multi_seed (+5.11pp) and C_no_feature_cap (+5.23pp) over A_baseline on Mode V combined_score at ETH 8h. Both passed the pre-set CLEAR WINNER threshold — feeds into H75 build the next week. |
| **2026-05-11** | **CPCV HRST = neutral.** López de Prado AFML Ch 12 Combinatorial Purged CV matched current method (no Mode T re-rank). Kept available as diagnostic for trigger-based re-runs. |
| **2026-05-10** | **Suspect-pool batch closed.** 7 ideas (C03 SHAP, C12 stability, C23 per-regime, C29 HAR-RV+Hurst, C31 funding momentum, C48 Sharpe label) retested on fixed harness — net 0 PASS, 1 MARGINAL+, 6 FAIL. |
| **2026-05-07** | **5ideas runner Tier 1 batch closed.** C35 wavelet, C42 CatBoost, C43 stacking, C44 quantile, C47 vol-adj label, C57 MS-AR — all DEAD/MARGINAL below +5pp ship threshold. |
| **2026-05-05** | **Critical harness bug found + fixed.** `runpy.run_module` was bypassing patcher monkey-patches; replaced with direct `main()` call. Several prior-suspect verdicts re-run on fixed harness. |
| **2026-05-04** | Canonical scoreboard built — C01-C82 single source of truth. Replaces parallel 20-roadmap + 14-harness numberings. |
| **2026-05-02** | **Engine merge.** Single-file `crypto_trading_system_ed.py`: parallel V/S/T paths + Mode P parallel-P + wrapper all merged. Pre-merge snapshots archived. Ein/Eli 15-min/30-min variants archived. |
| **2026-04-27** | MIX gate promoted (bull rally-cooldown). T1b multi-horizon ensemble + 5m emergency-exit overlay both tested and shelved same evening. |
| **2026-04-19** | Asset universe pruned to 5 (DOGE/ADA/AVAX/DOT dropped — weak priors, no diversification). GDELT features disabled (0/33 production selection). ETH derivatives features (funding + OI) added. |
| **2026-04-18** | PySR refresh (maxsize 15→25, iterations 40→100, multi-run + islands). HRSTG full refresh; per-regime hold-shield split shipped. |
| **2026-04-17** | On-chain loader wired (CoinMetrics MVRV + exchange netflow). |
| **2026-04-16** | V7 rally-cooldown gate in production (ETH). Mode G optimizer + `/gate` Telegram command. |
| **2026-04-13** | NTP clock-drift fix (replaces broken echo-back). Maker order bug fixes (partial fill, cancel verification, locked funds). |
| **2026-04-07** | Detector trim (14→5). Mode S rewritten as joint sweep. ETH-only, $12k allocation. |
| **2026-04-03** | Ed V2 release — maker orders (0% fee), 6 critical bug fixes. |
| **2026-03-29** | **Ed V1.0 release.** Regime-switching, Mode R regime backtest, PySR regime discovery. |
| **2026-03-24** | Doohan V1.7.1 promoted. Variable horizon support. (Archived 2026-04.) |
| **2026-03-23** | LGBM dominance proven. CPCV first attempt dropped (gamma incompatibility). |
| **2026-03-20** | Deku promoted. 3-fold holdout validation. |
| **2026-03-18** | Deku release — Optuna TPE+Hyperband, XGBoost, APF scoring. |
| **2026-03-15** | CASCA release — profit factor scoring. Temporal decay. |
| **2026-02-22** | Broly 1.2 — IB auto-trader for DAX CFDs. |
| **2026-02-21** | Initial commit. |

### Evolution

Ed V2 H75 (current) ← Ed V2 B_multi_seed (May 6) ← Ed V2 (Apr 3) ← Ed V1.0 (Mar 29) ← Doohan V1.7.1 ← Doohan V1.6 ← Deku ← CASCA ← V5 ← earlier. All legacy systems in [archive/](archive/).

---

## Owner

Alex — Lausanne, Switzerland (CET/CEST timezone). Repo: https://github.com/Spyku/Algo_trading.
