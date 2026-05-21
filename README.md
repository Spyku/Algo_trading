# Algo Trading Engine

Automated ML trading system for **crypto** (ETH live; BTC, SOL, LINK, XRP standby) and **index CFDs** (DAX, S&P 500 — separate system, see [cfd/](cfd/)). Generates hourly BUY/SELL/HOLD signals using ensemble ML models with walk-forward validation, temporal decay sample weighting, and embargo-corrected labels. Executes trades on Revolut X via Ed25519-signed API.

**Production system:** Ed V2 — regime-switching trading (bull/bear detection with dynamic horizon selection), maker-order pricing at `bid+0.01` with `post_only` for 0% Revolut X fees.

---

## Current production state (snapshot 2026-05-19)

- **Live since 2026-05-18 22:02 CEST** — H75 (H_STRICT_FAMILY engine merge)
- **Engine**: `crypto_trading_system_ed.py` with K=5 multi-seed denoising, REFINE_TRIALS=75, strict `(combo, w)` diversity dedup
- **Asset**: ETH only (BTC/SOL/LINK/XRP `enabled: false` — diversification thin)
- **Detector**: `sma24>sma100`. Bull = 5h@75% XGB+LGBM w=100 γ=0.9993. Bear = 8h@65% RF+LGBM w=162 γ=0.9954
- **Gates**: rr8h≥2.0 OR rr14h≥6.0 cd=6h (bull); rr10h≥5.5 OR rr12h≥2.0 cd=8h (bear)
- **Shields OFF**, min_sell_pnl=0%, max_hold=10h

For the current live OOS monitoring, scheduled runs, and research backlog → see [TODO.md](TODO.md).

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
