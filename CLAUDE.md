# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Automated ML trading system for crypto (BTC, ETH, XRP). Generates hourly BUY/SELL/HOLD signals using dual-horizon (4h + 8h) ensemble ML models with walk-forward validation and temporal decay sample weighting. Executes trades on Revolut X via Ed25519-signed API.

**Owner:** Alex, Lausanne, Switzerland (CET/CEST timezone)

---

## Machine Setup

**One shared engine folder** synced via Google Drive — both machines use the same code, data, and models. Only the venv is local per machine.

| Machine | Engine Path | Venv | Python |
|---------|-------------|------|--------|
| Desktop (primary) | `G:\engine\` (Google Drive synced) | `C:\algo_trading\venv\` | `C:\algo_trading\venv\Scripts\python.exe` |
| Laptop | `G:\Autres ordinateurs\My laptop\engine\` (Google Drive synced) | `C:\Users\Alex\algo_trading\venv\` | `C:\Users\Alex\algo_trading\venv\Scripts\python.exe` |

- **Desktop:** i7-14700KF, RTX 4080, 32GB — used for long Mode D runs
- **Laptop:** 16 cores, RTX 3070 Ti
- **GitHub:** https://github.com/Spyku/Algo_trading
- **Push:** `git_push.bat` from `G:\engine\`
- **OS:** Windows 11, Python 3.14 venv (NOT conda)
- **GPU:** LGBM uses GPU (`device='gpu'`), configured in `hardware_config.py`

### Install / Venv Setup

Each machine needs its own venv with all dependencies. The engine folder is shared but venvs are local.

```bash
# Desktop
C:\algo_trading\venv\Scripts\activate.bat
pip install -r G:\engine\requirements.txt

# Laptop
C:\Users\Alex\algo_trading\venv\Scripts\activate.bat
pip install -r "G:\Autres ordinateurs\My laptop\engine\requirements.txt"
```

**Always use the venv Python** — never system Python. When running from Claude Code, use the full venv path:
- Desktop: `"C:/algo_trading/venv/Scripts/python.exe"`
- Laptop: `"C:/Users/Alex/algo_trading/venv/Scripts/python.exe"`

---

## Commands

```bash
# === CASCA (production — profit factor scoring) ===
python crypto_trading_system_casca.py A BTC 4,8h     # Mode A — gamma optimization (6 gammas × horizons)
python crypto_trading_system_casca.py A BTC 4,8h --resume  # Resume interrupted gamma sweep
python crypto_trading_system_casca.py B BTC 4,8h     # Mode B — signals from saved models
python crypto_trading_system_casca.py D BTC 4,8h     # Mode D — full pipeline with PF scoring
python crypto_trading_system_casca.py F BTC 4,8h     # Mode F — strategy comparison (ranked by return)
python crypto_trading_system_casca.py DF BTC,ETH 4,8h  # Mode DF — D then F

# === CASCA V1.1 (experimental — fee-aware labels) ===
python crypto_trading_system_casca_v1.1.py D BTC 4,8h  # PF scoring + label=1 when return > 2×fee

# === DEKU (Optuna + XGBoost — Bayesian optimization) ===
python crypto_trading_system_deku.py D BTC 4,8h          # Deku hourly — Optuna joint optimization
python crypto_trading_system_deku.py D BTC 4,8h --trials 150  # Custom trial count
python crypto_trading_system_deku.py DF BTC,ETH 4,8h     # Deku hourly — D then F
python crypto_trading_system_deku_15m.py D BTC 4,8h      # Deku V15 — 15-min candles (s4=60', s8=120')
python crypto_trading_system_deku_15m.py DF BTC,ETH 4,8h # Deku V15 — D then F

# === DEKU FUSION (multi-timeframe backtest) ===
python testing_deku_fusion.py BTC              # Fusion backtest — hourly cadence
python testing_deku_fusion.py BTC --15min      # Fusion backtest — 15-min cadence
python testing_deku_fusion.py BTC --compare    # Deku V15 vs CASCA V15 comparison only

# Auto-trader
python crypto_revolut_trader.py --loop            # Live trading loop
python crypto_revolut_trader.py --dry-run --loop  # Signals only, no trades
python crypto_revolut_trader.py --status          # Show positions
python crypto_revolut_trader.py --balance         # Revolut X balance
```

**Telegram commands while trader is running:** `/stop` `/status` `/pause` `/resume` `/balance` `/sync` `/conf` `/chart BTC`

---

## Architecture

### Production File Chain

```
crypto_trading_system_casca.py  (CASCA production — Modes B/D/E/F/DF, profit factor scoring)
  └── hardware_config.py  (machine-specific model configs, n_jobs, GPU settings)

crypto_revolut_trader.py  (auto-trader — reads trading_config.json)
  └── crypto_live_trader.py  (signal generation library — NOT run directly)
        └── crypto_trading_system_casca.py  (imports ASSETS, features, models, download/load/build)
```

### Key Concepts

- **Dual horizons:** 4h and 8h — completely independent models per horizon (different features, window, combo). Never mix them.
- **No model persistence:** System retrains from scratch every prediction. No .pkl files. Intentional design.
- **Temporal decay (Cacarot):** Exponential sample weighting `w_i = gamma^(age_in_hours)`. Per-model gamma stored in best_models.csv. Newest sample weight=1, oldest=gamma^(n-1). gamma=0.995 → half-life ~6 days, gamma=0.999 → ~29 days. gamma >= 1.0 disables decay (zero overhead).
- **6-month data cap:** Mode D and E cap training data at 4,320 hours (6 months). Not configurable — hardcoded as `MAX_DIAG_HOURS`.
- **Labels:** Relative to 168h rolling median return, not absolute price direction. `label = 1` means return > median.
- **Walk-forward validation:** Train on last `window` hours → predict next candle → step forward 72h. No future leakage.
- **Scoring (CASCA — production):** Profit Factor `gross_profit / |gross_loss|` (capped at 5.0, min 3 trades). Feature selection also uses profit factor throughout (permutation, ablation, reduced sets). Mode F ranks by return directly. Replaces V5's broken `acc × (1 + ret/100)` formula.
- **Scoring (Deku — Optuna):** APF (Adjusted Profit Factor) = `raw_PF / buyhold_PF`. Normalizes model scores against market regime — PF=3 in a bull run isn't the same as PF=3 in a bear market. Optuna maximizes APF during Bayesian search.
- **Fee-aware labels (CASCA V1.1 — experimental):** `label = 1` when `future_return > 2 × TRADING_FEE` (0.22%). Every correct BUY is inherently profitable after fees. Replaces rolling median labels.
- **MIN_COMBO_SIZE=2:** Solo models removed from diagnostic (15→11 combos). Prevents overconfidence from uncalibrated single-model predictions (e.g., GB giving 100% confidence).
- **Models (CASCA):** RF, GB, LR, LGBM — 11 ensemble combinations tested (pairs + triples + quad). Solo models excluded.
- **Models (Deku):** RF, GB, XGB, LR, LGBM — 26 ensemble combinations (pairs + triples + quads + quint). XGBoost added as 5th model.
- **Features:** 49 technical + 81 macro/sentiment/cross-asset = 130 total.
- **Optuna TPE + Hyperband (Deku):** Bayesian joint optimization of (combo, window, gamma, n_features) via Tree-structured Parzen Estimator. Hyperband pruning kills bad trials early (~60% pruned). Default 100 trials. Replaces CASCA's sequential grid search (75 configs).
- **LGBM importance ranking (Deku):** Features ranked by LGBM gain importance (~5 sec) instead of 5-test analysis (~10 min). Optuna picks n_features from the ranked list.
- **Per-model features (V1.3.1 experimental):** Each model gets its own optimized feature set via independent 5-test analysis. LR gets clean linear features, trees get their own. Union feature matrix loaded, each model slices its columns. Feature analysis results cached to JSON (`models/feature_analysis_v1.3.1_{ASSET}_{H}h.json`) — rerun diagnostic without redoing ~80 min analysis. Cache invalidates automatically on gamma change.

### Data Flow

```
data/{asset}_hourly_data.csv           <- price data (Binance via ccxt)
data/{asset}_15m_data.csv              <- 15-min price data (Binance via ccxt)
data/macro_data/*.csv                  <- VIX, DXY, S&P500, Fear&Greed, etc. (yfinance)
models/crypto_casca_best_models.csv    <- CASCA production: best model per (asset, horizon) — profit factor scored
models/crypto_casca_v1.1_best_models.csv <- CASCA V1.1 experimental: fee-aware labels
models/crypto_casca_v1.3.1_best_models.csv <- CASCA V1.3.1 experimental: per-model features
models/crypto_deku_best_models.csv     <- Deku hourly: best model per (asset, horizon) — APF scored
models/crypto_deku_15m_best_models.csv <- Deku V15: 15-min candle models — APF scored
models/feature_analysis_v1.3.1_{ASSET}_{H}h.json <- cached per-model feature sets (reusable across runs)
models/testing_deku_fusion_results.csv <- Deku fusion backtest results (1h + 15' combined)
config/trading_config.json             <- per-asset strategy + min_confidence (written by Mode F)
config/trading_config_deku.json        <- Deku hourly trading config (written by Deku Mode F)
config/trading_config_deku_15m.json    <- Deku V15 trading config (written by Deku V15 Mode F)
config/position_{ASSET}.json           <- position tracking
config/revolut_x_config.json           <- Revolut X API key
config/private.pem                     <- Ed25519 signing key
config/telegram_config.json            <- Telegram bot token
```

**Config files are NOT in git.** `config/` is in `.gitignore`. Never push credentials.

---

## Key Constants

```python
TRADING_FEE_BASE = 0.0009   # 0.09% Revolut X taker fee (0% maker)
SLIPPAGE = 0.0002           # 0.02% estimated slippage (market impact, spread)
TRADING_FEE = 0.0011        # total cost per trade (fee + slippage) — applied on BUY and SELL
MIN_CONFIDENCE = 75         # global fallback only — overridden per asset by Mode F
MAX_DIAG_HOURS = 4320       # 6 months data cap for Mode D and E
REPLAY_HOURS = 200          # Modes B, D signal replay
REPLAY_HOURS_F = 400        # Mode F — longer window for more trades in strategy selection
DIAG_STEP = 72
DIAG_WINDOWS = [48, 72, 100, 150, 200]       # horizons 5-8h
DIAG_WINDOWS_SHORT = [24, 48, 72, 100, 150]  # horizons 1-4h
```

---

## Version Files

| File | Status | Notes |
|------|--------|-------|
| `crypto_trading_system_casca.py` | **CASCA Production** | Profit factor scoring. Feature selection by PF (permutation, ablation, reduced sets). Model ranking by PF (cap 5.0, min 3 trades). Mode F by return. Live trader imports from it. Writes to `models/crypto_casca_best_models.csv`. |
| `crypto_trading_system_casca_v1.1.py` | CASCA V1.1 Experimental | CASCA + fee-aware labels (`return > 2×fee`). Writes to `models/crypto_casca_v1.1_best_models.csv`. |
| `crypto_trading_system_casca_v1.3.1.py` | CASCA V1.3.1 DEAD END | Per-model feature analysis — worse than CASCA. Feature union defeats pruning (130 features kept). Shared consensus is better. |
| `crypto_trading_system_v6.py` | V6 Experimental | 12 literature enhancements behind `ENHANCEMENTS` flags (env var override). NOT production. |
| `archive/testing_literature.py` | Test harness | A/B tests V5.5 enhancements — COMPLETE, archived. Results in `archive/testing_literature.csv` |
| `testing_literature_v2.py` | Test harness | A/B tests 12 V6 enhancements (Mode D BTC 4,8h 1y) |
| `archive/testing_cacarot_v1.5.py` | Archived | Tests 15 V5.6-rejected features WITH decay. Promoted to production → archived. |
| `archive/testing_feature_stability.py` | Archived | Feature stability test across BTC+ETH × 4h+8h |
| `archive/crypto_trading_system_v5_cacarot.py` | Archived | V5 Cacarot. Old scoring: `acc × (1 + ret/100)`. Replaced by CASCA. |
| `archive/crypto_trading_system_v5.8.py` | Archived | Early Cacarot prototype. |
| `crypto_revolut_trader.py` | Live | Multi-asset live trader + `/conf` `/chart` commands |
| `crypto_live_trader.py` | Live | Signal generation core — NOT run directly |
| `hardware_config.py` | Active | Auto-detects Desktop (26 workers) / Laptop (14 workers) |
| `crypto_trading_system_v15.py` | V15 Cacarot | 15-min candles, horizons 15'–120', temporal decay (gamma), 4320-row data cap (~45 days). |
| `crypto_trading_system_v30.py` | V30 Cacarot | 30-min candles, horizons 30'–240', temporal decay (gamma), 4320-row data cap (~3 months). |
| `testing_v15.1.py` | Test harness | V15 gamma optimization: 7 gammas (1.0–0.994) × 2 horizons (4,8). Isolated output: `models/testing_v15.1_results.csv` |
| `testing_v30.1.py` | Test harness | V30 gamma optimization: 7 gammas (1.0–0.994) × 2 horizons (4,8). Isolated output: `models/testing_v30.1_results.csv` |
| `testing_casca.py` | Test harness | Multi-timeframe fusion: V5 (1h) + V15 (15') signals combined. 16 strategies, confidence sweep. `models/testing_casca_results.csv` |
| `crypto_trading_system_deku.py` | **Deku Hourly** | Optuna TPE+Hyperband Bayesian optimization. 5 models (RF, GB, XGB, LR, LGBM), 26 combos. APF scoring. LGBM importance ranking (~5s vs ~10min). Continuous gamma [0.995–1.0]. Writes to `models/crypto_deku_best_models.csv`. |
| `crypto_trading_system_deku_15m.py` | **Deku V15** | Deku with 15-min candles. s4=60', s8=120'. 4320-candle cap (~45 days). Scaled constants via `_hours_to_rows()`. Writes to `models/crypto_deku_15m_best_models.csv`. |
| `crypto_trading_system_casca_v1.4.py` | CASCA V1.4 Experimental | CASCA baseline used for Deku comparison. Feature selection with LGBM importance ranking. Writes to `models/crypto_casca_v1.4_best_models.csv`. |
| `testing_deku_fusion.py` | Test harness | Multi-timeframe fusion: Deku hourly + Deku V15 combined. 16 strategies (1h-only, 15'-only, cross-TF), confidence sweep, max drawdown tracking. `models/testing_deku_fusion_results.csv` |
| `cfd/ib_auto_trader.py` | Live | DAX CFD trader (Broly 1.2) |
| `cfd/ib_auto_trader_test.py` | Live | S&P 500 CFD overnight trader |

---

## Strategies

| Strategy | Logic |
|----------|-------|
| `both_agree` | BUY when 4h AND 8h agree; SELL when either says SELL |
| `either_agree` | BUY when either says BUY; SELL when either says SELL |
| `4h_only` / `8h_only` | Single horizon only |
| `8h_and_1h`, `8h_and_2h` | V5.2 short+long combos |
| `any_agree` | BUY when any horizon says BUY |

**Mode F writes the best strategy and min_confidence directly to `trading_config.json`. The trader picks this up automatically on startup — no manual override needed.**

---

## Current Best Models

| Asset | Horizon | Models | Window | Accuracy | Return | PF Score | Features | Gamma | Status |
|-------|---------|--------|--------|----------|--------|----------|----------|-------|--------|
| BTC | 4h | RF+GB | 48h | 72.9% | +27.2% | 5.00 | 34 | 1.0 | CASCA 6mo |
| BTC | 8h | GB+LGBM | 150h | 72.4% | +25.4% | 2.00 | 8 | 1.0 | CASCA 6mo |
| ETH | 4h | RF+LR | 48h | 83.1% | +18.1% | — | 40 | 0.999 | Cacarot 6mo |
| ETH | 8h | RF | 200h | 91.2% | +35.6% | — | 90 | 0.995 | Cacarot 6mo |
| XRP | 4h | RF+GB+LGBM | 100h | 71.7% | +202.3% | — | 15 | 1.0 | V5 2y |
| XRP | 8h | RF | 100h | 82.5% | +139.2% | — | 50 | 1.0 | V5 2y |
| DOGE | 4h | RF+LR | 72h | 78.1% | — | — | 30 | 1.0 | V5 2y |
| DOGE | 8h | LR | 100h | 74.0% | — | — | 79 | 1.0 | V5 2y WARNING |

### Deku Hourly (BTC, Optuna)

| Asset | Horizon | Models | Window | Accuracy | Return | APF Score | Features | Gamma | Status |
|-------|---------|--------|--------|----------|--------|-----------|----------|-------|--------|
| BTC | 4h | RF+XGB+LGBM | 48h | — | +54.2% | 5.902 | — | 0.9951 | Deku 6mo |
| BTC | 8h | RF+XGB+LR+LGBM | 48h | — | +36.2% | 4.543 | — | 0.9995 | Deku 6mo |

**WARNING — DOGE 8h LR outputs 99-100% confidence on most signals.** Needs `CalibratedClassifierCV`. Do not enable DOGE for live trading until fixed.

## Current Trading Config

```json
{
  "BTC": { "strategy": "4h_only", "min_confidence": 80, "max_position_usd": 6000, "enabled": true },
  "ETH": { "strategy": "4h_only", "min_confidence": 60, "max_position_usd": 6000, "enabled": false },
  "XRP": { "strategy": "either_agree", "min_confidence": 75, "max_position_usd": 0, "enabled": false },
  "DOGE": { "strategy": "8h_only", "min_confidence": 90, "max_position_usd": 0, "enabled": false }
}
```

**BTC live trading at $6k (CASCA DF 2026-03-17, `4h_only` @80%). ETH DF running. XRP, DOGE disabled.**

---

## Critical Rules

1. **Never modify `crypto_trading_system_casca.py` without testing first.** It is CASCA production. The live trader imports from it by exact filename.
2. **CSV merge logic:** When saving to `crypto_casca_best_models.csv`, always filter by BOTH coin AND horizon.
3. **`generate_signals()` needs `feature_override`:** When feature_set is D, caller must pass `feature_override=config['optimal_features'].split(',')`.
4. **All timestamps in live trader use Europe/Zurich** via `_to_local()` helper.
5. **`trading_config.json` has `min_confidence` per asset** — set by Mode F. Global `MIN_CONFIDENCE=75` is only a fallback.
6. **SSL fix on Windows:** `ssl._create_unverified_context()` applied in both `crypto_revolut_trader.py` and `crypto_live_trader.py`.

---

## Pending Work

### Active
1. **Deku V15 BTC running** — `python crypto_trading_system_deku_15m.py D BTC 4,8h` — Optuna optimization for 15-min candles
2. **Deku fusion backtest** — After Deku V15 completes: `python testing_deku_fusion.py BTC` and `--15min` — validate Optuna > grid search for 15', find best 1h+15' strategies
3. **Apply Mode A gamma winners** — Re-run Mode D with gamma=0.996 (4h) and 0.997 (8h) to get production models with optimal decay
4. **Test CASCA V1.1** — `python crypto_trading_system_casca_v1.1.py D BTC 4,8h` — compare fee-aware labels vs rolling median
5. **Re-enable ETH in trading_config.json** — After ETH DF completes, set `enabled: true`

### After Active
6. **Deku DF ETH** — `python crypto_trading_system_deku.py DF ETH 4,8h` — Optuna optimization for ETH
7. **V15.1 gamma optimization** — `python testing_v15.1.py --resume` — re-run with MIN_COMBO_SIZE=2 on desktop
8. **V30.1 gamma optimization** — `python testing_v30.1.py` — 7 gammas × 2 horizons
9. **Fee-aware labeling test** — Current: `return > 168h median` (profit-blind). Alternative: `label = 1` only when `return > 2 × TRADING_FEE` (0.22%). Only needs 2 tests (winning gamma × 2 horizons), not full 14-test sweep.
10. **Dynamic data cap** — Replace hardcoded `MAX_DIAG_ROWS`/`MAX_DIAG_HOURS` with `calc_data_cap(gamma)` formula. TODO for later.

### Desktop TODO
11. **V6 A/B tests** — `python testing_literature_v2.py --resume` — RUNNING. Laptop hangs on 8h diagnostic (loky deadlock with 14 workers), must run on desktop.

### Lower Priority
12. **XRP CASCA** — `python crypto_trading_system_casca.py DF XRP 4,8h`
13. **Weekly F runs** — re-run CASCA `F BTC 4,8h` and `F ETH 4,8h` weekly
14. **Windows auto-start** — CryptoTrader scheduled task registered, needs reboot test

### Completed
- **Deku hourly release** — DONE (2026-03-18). Optuna TPE+Hyperband Bayesian optimization replaces CASCA grid search. 5 models (RF, GB, XGB, LR, LGBM), 26 combos. APF scoring (PF/buyhold_PF). LGBM importance ranking (~5s). Continuous gamma [0.995–1.0]. BTC 4h: RF+XGB+LGBM w=48h g=0.9951, APF=5.902, +54.2% (vs CASCA +27.2%). BTC 8h: RF+XGB+LR+LGBM w=48h g=0.9995, APF=4.543, +36.2%.
- **Deku V15 created** — DONE (2026-03-18). Deku with 15-min candles. s4=60', s8=120'. 4320-candle cap (~45 days). Constants scaled via `_hours_to_rows()`. Running BTC Mode D.
- **Deku fusion test harness** — DONE (2026-03-18). `testing_deku_fusion.py` — 16 strategies (1h-only, 15'-only, cross-TF), two cadences, confidence sweep, max drawdown tracking. `--compare` flag for Deku V15 vs CASCA V15.
- **CASCA V1.4 created** — DONE (2026-03-18). CASCA baseline for Deku comparison with LGBM importance ranking.
- **CASCA V1.3.1 DEAD END** — DONE (2026-03-17). Per-model feature analysis. BTC 4h: 55.9% acc, +31.5%, PF=2.19 (vs CASCA 72.9%, +27.2%, PF=5.0). BTC 8h: +15.1% (vs CASCA +25.4%). Problem: feature union of per-model sets = no pruning (130/118 features kept). Shared feature selection in CASCA acts as regularizer — forcing model consensus is better than per-model optimization.
- **Mode A gamma optimization BTC** — DONE (2026-03-17). 6 gammas × 2 horizons. Winners: 4h gamma=0.996 (LGBM w=200, +55.8%, PF=5.0) and 8h gamma=0.997 (RF+LGBM w=100, +50.0%, PF=3.88). Both ~2× baseline return. Results in `models/testing_casca_a_results.csv`.
- **CASCA DF ETH** — DONE (2026-03-17).
- **CASCA DF BTC** — DONE (2026-03-17). BTC 4h: RF+GB w=48, 72.9% acc, +27.2%, PF=5.00. BTC 8h: GB+LGBM w=150, 72.4% acc, +25.4%, PF=2.00. Mode F: `4h_only` @80%. No more solo models (MIN_COMBO_SIZE=2 fixes overconfidence).
- **CASCA V1.3.1 created** — DONE (2026-03-17). Per-model feature analysis: each of RF/GB/LR/LGBM gets its own 5-test feature optimization. Union matrix with per-model column slicing in diagnostic and signal generation. Feature analysis cached to `models/feature_analysis_v1.3.1_{ASSET}_{H}h.json` — skip ~80 min on re-runs (auto-invalidates on gamma change). File: `crypto_trading_system_casca_v1.3.1.py`. Isolated outputs: `crypto_casca_v1.3.1_best_models.csv`, `trading_config_v1.3.1.json`.
- **V30 CASCA port** — DONE (2026-03-17). All PF scoring changes ported to V30: `_quick_score` with PF, feature analysis by PF, diagnostic scoring by PF, MIN_COMBO_SIZE=2, Mode A, `_get_models_csv_path()`.
- **MIN_COMBO_SIZE=2 in CASCA** — DONE (2026-03-17). Solo models (RF, GB, LR, LGBM alone) removed from diagnostic. 15→11 combos. Fixes production overconfidence issue where solo GB gave 100% confidence.
- **V5 Cacarot archived** — DONE (2026-03-16). Archived to `archive/crypto_trading_system_v5_cacarot.py`. Also archived: `crypto_trading_system_v5.8.py`, `testing_cacarot_v1.5.py`, `testing_feature_stability.py`. CASCA is now production.
- **CASCA V1.1 created** — DONE (2026-03-16). Fee-aware labels (`return > 2×TRADING_FEE`). File: `crypto_trading_system_casca_v1.1.py`.
- **Live trader connected to CASCA** — DONE (2026-03-16). `crypto_live_trader.py` imports from `crypto_trading_system_casca` and reads `crypto_casca_best_models.csv`.
- **CASCA scoring model created** — DONE (2026-03-16). `crypto_trading_system_casca.py` — profit factor scoring replaces `acc × (1 + ret/100)`. Feature selection by PF (permutation, ablation, reduced sets). Model ranking by PF (cap 5.0, min 3 trades). Mode F by return. Isolated output: `models/crypto_casca_best_models.csv`.
- **V15 Mode D fresh run** — DONE (2026-03-16). BTC s4: RF+GB+LR w=144, 72.1% acc, +17.3%. s8: LGBM w=288, 88.1% acc, +12.1%. Mode F: `both_agree` @61%, +62.6% return, 93% win rate.
- **V5+V15 fusion backtest** — DONE (2026-03-16). 7 strategies tested over 7.75 days. Best: V15 override (+17.6%) > V5 alone (+13.9%) > V15 alone (+12.1%) > B&H (+7.9%). See `tools/backtest_v5_v15.py`.
- **Bug fix: test harnesses overwriting production config** — DONE (2026-03-16). `testing_cacarot_v1.5.py` and `crypto_trading_system_v5.8.py` Mode F wrote to `trading_config.json` instead of isolated test configs. Fixed to write to `trading_config_v1.5_test.json` and `trading_config_v5.8_test.json` respectively. This bug had overwritten BTC config from `either_agree @90%` to `4h_only @80%`.
- **Bug fix: MODELS_DIR forward reference** — DONE (2026-03-16). `crypto_trading_system_v15.py` and `crypto_trading_system_v30.py` used `MODELS_DIR` at line 128 before it was defined at line 525. Added early definition.
- **V15/V30 Cacarot release** — DONE (2026-03-16). Temporal decay (gamma), 4320-row data cap, DF mode added to both V15 and V30.
- **Live trader restarted** — DONE (2026-03-16). Running with BTC (either_agree @90%) and ETH (4h_only @60%) configs.
- **Cacarot release** — DONE (2026-03-16). Temporal decay (gamma per asset+horizon), 6mo data cap, Mode F charts.
- **BTC Cacarot Mode DF** — DONE (2026-03-16). BTC: `either_agree` @90%, return +42.1%.
- **ETH Cacarot Mode DF** — DONE (2026-03-16). ETH: `4h_only` @60%, return +62.8%.
- **Re-run Mode DF for BTC** — DONE (2026-03-15). Fresh models in best_models.csv.
- **Re-run Mode F for ETH** — DONE (2026-03-15). ETH: `8h_only`, min_confidence=85%.
- **Restart crypto_revolut_trader** — DONE (2026-03-15). Running with fresh models + 5-min hot-reload for config+models+positions.
- **Laptop venv install** — DONE. PyWavelets + xgboost installed.
- **V5.5 A/B tests (testing_literature.py)** — DONE, archived. 8 tests × BTC 4h+8h 1y. Results:
  - **slippage_model** — WINNER: 8h return +98.4% vs baseline +74.0%, consistent improvement → promoted to production
  - **extended_diag_step** — 8h accuracy 85.6% (best) but fewer trades (22 vs 32)
  - **on_chain_features** — 8h return +106.1% but accuracy dropped 5 points (73.3% vs 78.3%)
  - **gb_calibration** — neutral, slight regression
  - **triple_barrier_label** — worse on all metrics (63.9%/65.0% accuracy)
  - **purged_embargo** — much worse (58.7%/62.8% accuracy)
  - **derivatives_features** — only 2 trades, meaningless
  - Baseline: 4h 80.0% +57.2%, 8h 78.3% +74.0%
- **V5.5 promotion** — slippage model (TRADING_FEE 0.0009 → 0.0011) applied to V5, V15, V30
- **DF combined mode** — `python crypto_trading_system.py DF BTC,ETH 4,8h 1y` runs Mode D then F
- **Telegram /conf + /chart** — added to crypto_revolut_trader.py
- **V6 created** — 12 literature enhancements (wavelet, fracdiff, GMM regime, XGBoost, sample weighting, entropy filter, tri-state labels, stacking, dynamic feature select, meta-labeling, adversarial validation, Kelly sizing)
- **Process priority** — Mode D/F runs at BELOW_NORMAL priority on Windows so trader always gets CPU
- **V5.6 archived** — 20 literature features tested (Garman-Klass, ADX, Hurst, MFI, skewness, kurtosis, etc.). Only ADX + Garman-Klass proved useful → added directly to V5 production. V5.6 file archived.
