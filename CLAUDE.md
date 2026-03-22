# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Automated ML trading system for crypto (BTC, ETH, XRP, DOGE, SOL, LINK, ADA, AVAX, DOT). Generates hourly BUY/SELL/HOLD signals using dual-horizon (4h + 8h) ensemble ML models with walk-forward validation and temporal decay sample weighting. Executes trades on Revolut X via Ed25519-signed API. Horizons are parametric via `HORIZON_SHORT` and `HORIZON_LONG` constants in `crypto_trading_system_deku.py`.

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
# === DEKU (production — Optuna + XGBoost — Bayesian optimization) ===
python crypto_trading_system_deku.py D BTC 4,8h          # Mode D — Optuna joint optimization
python crypto_trading_system_deku.py D BTC 4,8h --trials 150  # Custom trial count
python crypto_trading_system_deku.py DF BTC,ETH 4,8h     # Mode DF — D then F
python crypto_trading_system_deku.py DF BTC 4,8h --metric calmar  # Test alternative scoring metric
python crypto_trading_system_deku.py DF BTC 4,8h --metric all    # Run all 5 metrics and compare
python crypto_trading_system_deku_15m.py D BTC 4,8h      # Deku V15 — 15-min candles
python crypto_trading_system_deku_15m.py DF BTC,ETH 4,8h # Deku V15 — D then F

# === CASCA (standby — profit factor scoring) ===
python crypto_trading_system_casca.py D BTC 4,8h     # Mode D — full pipeline with PF scoring
python crypto_trading_system_casca.py A BTC 4,8h     # Mode A — gamma optimization (6 gammas × horizons)
python crypto_trading_system_casca.py DF BTC,ETH 4,8h  # Mode DF — D then F

# === Auto-trader (Deku — production) ===
python crypto_revolut_deku.py --loop              # Live trading loop
python crypto_revolut_deku.py --dry-run --loop    # Signals only, no trades
python crypto_revolut_deku.py --status            # Show positions

# === Auto-trader (CASCA — standby) ===
python crypto_revolut_trader.py --loop            # Live trading loop
python crypto_revolut_trader.py --dry-run --loop  # Signals only, no trades
```

**Telegram commands (Deku trader):** `/stop` `/status` `/pause` `/resume` `/balance` `/sync` `/conf` `/config` `/setup` `/help` `/chart BTC` `/optimize BTC` `/optstatus`

**Telegram commands (CASCA trader):** `/stop` `/status` `/pause` `/resume` `/balance` `/sync` `/conf` `/chart BTC`

---

## Architecture

### Production File Chain

```
crypto_trading_system_deku.py  (Deku 1.3 production — Modes B/D/F/DF, Optuna + APF scoring)
  └── hardware_config.py  (machine-specific model configs, n_jobs, GPU settings)

crypto_revolut_deku.py  (Deku auto-trader — reads trading_config_deku.json)
  └── crypto_live_trader_deku.py  (signal generation library — NOT run directly)
        └── crypto_trading_system_deku.py  (imports ASSETS, features, models, download/load/build)
```

CASCA and all legacy systems (V15/V30 Cacarot, V6) archived — Deku strictly superior.

### Key Concepts

- **Dual horizons:** 4h and 8h (parametric via `HORIZON_SHORT=4`, `HORIZON_LONG=8` in deku.py) — completely independent models per horizon (different features, window, combo). Never mix them. To change horizons, update the two constants and re-run Mode DF.
- **No model persistence:** System retrains from scratch every prediction. No .pkl files. Intentional design.
- **Temporal decay (Cacarot):** Exponential sample weighting `w_i = gamma^(age_in_hours)`. Per-model gamma stored in best_models.csv. Newest sample weight=1, oldest=gamma^(n-1). gamma=0.995 → half-life ~6 days, gamma=0.999 → ~29 days. gamma >= 1.0 disables decay (zero overhead).
- **6-month data cap:** Mode D caps training data at 4,320 hours (6 months). Hardcoded as `MAX_DIAG_HOURS`.
- **Labels (fee-aware):** `label = 1` when future return > 2×TRADING_FEE (0.22%). Fallback mode uses 200h rolling median. Set by `LABEL_MODE = 'fee_aware'`.
- **Walk-forward validation:** Train on last `window` hours → predict next candle → step forward. DIAG_STEP=36 (Deku) or 72 (CASCA). No future leakage.
- **Scoring (Deku — production):** Default APF (Adjusted Profit Factor) = `raw_PF / buyhold_PF`. Normalizes against market regime. Supports `--metric` flag: `apf`, `rawpf`, `calmar`, `return`, `rpf_sqrt`. Use `--metric all` to compare.
- **Scoring (CASCA — standby):** Profit Factor `gross_profit / |gross_loss|` (capped at 5.0, min 3 trades). Feature selection also uses PF. Mode F ranks by return.
- **3-fold rolling holdout (Deku):** Train on fold 1 (2,592 rows), re-rank winners by out-of-sample performance across 3 folds with embargo=4. Reduces overfitting.
- **Auto-extend trials (Deku):** If best APF < 1.7 after 150 trials, extend to 200, then 250.
- **MIN_COMBO_SIZE=2:** Solo models removed from diagnostic. Prevents overconfidence from uncalibrated single-model predictions.
- **MIN_TRADES=8 (Deku):** Optuna objective returns 0 for trials with <8 trades. Prevents statistically unreliable configs from winning.
- **Models (Deku):** RF, GB, XGB, LR, LGBM — 26 ensemble combinations (pairs + triples + quads + quint).
- **Models (CASCA):** RF, GB, LR, LGBM — 11 ensemble combinations (pairs + triples + quad).
- **Features:** 49 technical + 81 macro/sentiment/cross-asset = 130 total.
- **LGBM importance ranking (Deku):** Features ranked by LGBM gain importance (~5 sec) instead of CASCA's 5-test analysis (~10 min). Optuna picks n_features from the ranked list.
- **Model hot-reload:** Trader checks best_models CSV every 5 minutes. All updates accepted immediately (no regression guard).

### Data Flow

```
data/{asset}_hourly_data.csv           <- price data (Binance via ccxt)
data/{asset}_15m_data.csv              <- 15-min price data (Binance via ccxt)
data/macro_data/*.csv                  <- VIX, DXY, S&P500, Fear&Greed, etc. (yfinance)
models/crypto_deku_best_models.csv     <- Deku production: best model per (asset, horizon) — APF scored
models/crypto_casca_best_models.csv    <- CASCA standby: best model per (asset, horizon) — PF scored
models/crypto_deku_best_models_{metric}.csv <- Deku metric-specific models (rawpf, calmar, return, rpf_sqrt)
models/crypto_deku_15m_best_models.csv <- Deku V15: 15-min candle models — APF scored
config/trading_config_deku.json        <- Deku per-asset strategy + min_confidence (written by Mode F)
config/trading_config.json             <- CASCA per-asset strategy + min_confidence (written by Mode F)
config/position_{ASSET}.json           <- position tracking
config/revolut_x_config.json           <- Revolut X API key
config/private.pem                     <- Ed25519 signing key
config/telegram_config.json            <- Telegram bot token
```

**Config files are NOT in git.** `config/` is in `.gitignore`. Never push credentials.

---

## Key Constants

```python
# Shared
TRADING_FEE_BASE = 0.0009   # 0.09% Revolut X taker fee (0% maker)
SLIPPAGE = 0.0002           # 0.02% estimated slippage (market impact, spread)
TRADING_FEE = 0.0011        # total cost per trade (fee + slippage) — applied on BUY and SELL
MIN_CONFIDENCE = 75         # global fallback only — overridden per asset by Mode F
MAX_DIAG_HOURS = 4320       # 6 months data cap for Mode D
REPLAY_HOURS = 200          # Modes B, D signal replay
REPLAY_HOURS_F = 400        # Mode F — longer window for more trades in strategy selection

# Deku
HORIZON_SHORT = 4                                  # Deku short horizon (parametric)
HORIZON_LONG = 8                                   # Deku long horizon (parametric)
AVAILABLE_HORIZONS = [4, 8]                        # Deku production horizons
DIAG_STEP = 36                                     # Deku walk-forward step (doubled eval points)
DIAG_WINDOWS = [24, 36, 48, 72, 100, 150, 200]    # Deku search space (7 windows)
GAMMA_RANGE = (0.994, 1.0)                         # Deku continuous gamma (hardcoded in Optuna suggest_float)
MIN_TRADES = 8                                     # Deku minimum trades filter
DEKU_DEFAULT_TRIALS = 150                          # Deku Optuna trial count
APF_EXTEND_THRESH = 1.7                            # Auto-extend trials if best APF below this
OPTUNA_METRIC = 'apf'                              # Deku scoring (apf|rawpf|calmar|return|rpf_sqrt)

# CASCA
DIAG_STEP = 72                                     # CASCA walk-forward step
DIAG_WINDOWS = [48, 72, 100, 150, 200]             # CASCA windows
DIAG_WINDOWS_SHORT = [24, 48, 72, 100, 150]        # CASCA only — horizons 1-4h
```

---

## Version Files

| File | Status | Notes |
|------|--------|-------|
| `crypto_trading_system_deku.py` | **Deku Production** | Optuna TPE+Hyperband. 5 models (RF, GB, XGB, LR, LGBM), 26 combos. 3-fold holdout. Auto-extend trials. `--metric` flag. Writes to `models/crypto_deku_best_models.csv`. |
| `crypto_trading_system_deku_v1_5.py` | **Testing** | V1.5: Dynamic data cap (99% gamma weight), 3 holdout variants (current/A/B). Tests non-overlapping folds. |
| `crypto_trading_system_deku_v1_4_cpcv_gamma1_failed.py` | **Archived** | V1.4: Gamma=1.0 + CPCV. Failed — killed returns and trade count. |
| `crypto_trading_system_deku_v1_3_2.py` | **Testing** | V1.3.2: Narrowed A/B comparison. LR+LGBM combos only (8), gamma 0.995–0.998, features 5–30. |
| `crypto_trading_system_deku_v1_3_1.py` | **Testing** | V1.3.1: A/B/C mode comparison. CPCV calibration. Found 4h overfits (PBO=1.0), 8h is real edge. |
| `crypto_trading_system_deku_15m.py` | **Deku V15** | Deku with 15-min candles. s4=60', s8=120'. 4320-candle cap (~45 days). |
| `crypto_revolut_deku.py` | **Live** | Deku auto-trader + Telegram inline buttons + `/optimize` `/optstatus` |
| `crypto_live_trader_deku.py` | **Live** | Deku signal generation library — NOT run directly |
| `hardware_config.py` | Active | Auto-detects Desktop (26 workers) / Laptop (14 workers) |
| `cfd/ib_auto_trader.py` | Live | DAX CFD trader (Broly 1.2) |
| `cfd/ib_auto_trader_test.py` | Live | S&P 500 CFD overnight trader |

---

## Strategies

| Strategy | Logic |
|----------|-------|
| `both_agree` | BUY when 4h AND 8h agree; SELL when either says SELL |
| `either_agree` | BUY when either says BUY; SELL when either says SELL |
| `4h_only` / `8h_only` | Single horizon only |

**Mode F writes the best strategy and min_confidence to `trading_config_deku.json` (Deku) or `trading_config.json` (CASCA). The trader picks this up automatically on startup.**

---

## Current Best Models (Deku Production)

| Asset | Horizon | Models | Window | Return | APF | Features | Gamma | Trades |
|-------|---------|--------|--------|--------|-----|----------|-------|--------|
| BTC | 4h | GB+XGB+LR | 150h | +1.9% | 1.82 | 4 | 0.9962 | 16 |
| BTC | 8h | **XGB+LR+LGBM** | 200h | +11.3% | 1.47 | 13 | 0.9956 | 22 |
| ETH | 4h | RF+GB+XGB+LGBM | 36h | +4.6% | 4.16 | 5 | 0.9944 | 14 |
| ETH | 8h | RF+GB+XGB+LGBM | 48h | +6.5% | 2.41 | 21 | 0.9957 | 14 |
| XRP | 4h | RF+XGB+LR+LGBM | 36h | +0.5% | 2.49 | 34 | 0.9941 | 8 |
| XRP | 8h | GB+LR+LGBM | 150h | +7.6% | 2.29 | 73 | 0.9971 | 16 |
| DOGE | 4h | LR+LGBM | 150h | +3.2% | 2.15 | 27 | 0.9962 | 14 |
| DOGE | 8h | GB+LR+LGBM | 150h | +6.4% | 2.22 | 73 | 0.9971 | 14 |
| SOL | 4h | RF+XGB+LR+LGBM | 72h | -0.7% | 1.98 | 13 | 0.9972 | 12 |
| SOL | 8h | RF+GB+XGB | 48h | +5.4% | 2.18 | 17 | 0.9971 | 17 |
| LINK | 4h | XGB+LGBM | 150h | +5.3% | 2.15 | 6 | 0.9972 | 17 |
| LINK | 8h | **RF+GB+LR+LGBM** | 200h | +23.5% | 1.71 | 20 | 0.9963 | 31 |
| ADA | 4h | RF+GB+XGB+LR+LGBM | 150h | +3.6% | 1.78 | 24 | 0.9967 | 14 |
| ADA | 8h | RF+GB+XGB+LGBM | 24h | +21.3% | 7.94 | 26 | 0.9996 | 8 |
| AVAX | 4h | RF+XGB+LR+LGBM | 100h | +6.5% | 1.97 | 20 | 1.0 | 11 |
| AVAX | 8h | XGB+LR | 150h | +19.6% | 4.03 | 52 | 0.9944 | 18 |
| DOT | 4h | RF+XGB+LGBM | 100h | +12.5% | 2.83 | 27 | 0.9954 | 16 |
| DOT | 8h | XGB+LR | 150h | +9.3% | 8.47 | 41 | 0.999 | 17 |

BTC 8h and LINK 8h updated with CPCV-validated configs from V1.3.1 (2026-03-22). Other assets unchanged.

## Current Trading Config (Deku)

```json
{
  "BTC": { "strategy": "8h_only", "min_confidence": 70, "max_position_usd": 1000, "enabled": true },
  "ETH": { "strategy": "8h_only", "min_confidence": 70, "max_position_usd": 1000, "enabled": false },
  "XRP": { "strategy": "either_agree", "min_confidence": 60, "enabled": false },
  "DOGE": { "strategy": "8h_only", "min_confidence": 75, "enabled": false },
  "SOL": { "strategy": "4h_only", "min_confidence": 75, "enabled": false },
  "LINK": { "strategy": "8h_only", "min_confidence": 75, "max_position_usd": 1000, "enabled": true },
  "ADA": { "strategy": "4h_only", "min_confidence": 75, "enabled": false },
  "AVAX": { "strategy": "4h_only", "min_confidence": 60, "enabled": false },
  "DOT": { "strategy": "8h_only", "min_confidence": 80, "enabled": false }
}
```

---

## Critical Rules

1. **Never modify `crypto_trading_system_deku.py` without testing first.** It is Deku production. The live trader imports from it by exact filename.
2. **CSV merge logic:** When saving to `crypto_deku_best_models.csv`, always filter by BOTH coin AND horizon.
3. **`generate_signals()` needs `feature_override`:** When feature_set is D, caller must pass `feature_override=config['optimal_features'].split(',')`.
4. **All timestamps in live trader use Europe/Zurich** via `_to_local()` helper.
5. **`trading_config_deku.json` has `min_confidence` per asset** — set by Mode F. Global `MIN_CONFIDENCE=75` is only a fallback.
6. **SSL fix on Windows:** `ssl._create_unverified_context()` applied in both `crypto_revolut_deku.py` and `crypto_live_trader_deku.py` (and CASCA equivalents).

---

## Pending Work

### Active
1. **V1.5 — Dynamic data cap + holdout comparison.** Tests 3 holdout strategies with gamma-aware data sizing. Dynamic cap = `log(0.01)/log(gamma)` hours (gamma=0.996 → 48 days, gamma=0.999 → 6 months). BTC 8h only.
   - `python crypto_trading_system_deku_v1_5.py D BTC 8h --holdout all --trials 150`
   - Holdout modes: `current` (overlapping, production baseline), `A` (non-overlapping sequential), `B` (expanding window)

### Completed (Recent — continued)
- **CPCV investigation complete** — DONE (2026-03-22). V1.3.1 (gamma search + CPCV) and V1.4 (gamma=1.0 + CPCV) both tested. 1-week backtest proved gamma models outperform (BTC +7.4%, LINK +12.1% vs gamma=1.0 +4.4%, LINK 0 signals). CPCV dropped as validation method. Key finding: 4h overfits everywhere (PBO=1.0), 8h is where real alpha lives.
- **Telegram UX overhaul** — DONE (2026-03-22). `/chart` now shows 48h candlestick chart with signal transitions only (no hourly repeats), blue/red color scheme for colorblind accessibility. `/setup` replaced with inline button navigation (asset picker, toggle, strategy, confidence, max position with $0/$1K/$5K/$10K presets + custom). All green indicators changed to blue. Fixed last 4h prices bug (now reads from df_raw).
- **V1.3.1 CPCV A/B/C test** — DONE (2026-03-22). All 9 runs complete. Key finding: LR+LGBM core required for PBO ≤ 0.33. 4h overfits everywhere. Mode C weakest, dropped.
- **Enhancement code stripped** — DONE (2026-03-22). Removed from production deku.py, V1.3.1, V1.3.2.
- **Weekly F re-runs** — DONE (2026-03-21). Re-ran Deku F for all active assets.

### Dropped
- ~~Enhancement A/B test~~ — Both enhancement sets dropped. Original 11 features + 2 toggles failed (-62.2%). Lighter `--enhancements` (3 Optuna toggles) also dropped.
- ~~Fold weighting~~ — Gamma already handles recency at sample level. Literature doesn't support fold weighting; CPCV is the proper next step.
- ~~Mode A gamma optimization~~ — Replaced by Optuna continuous gamma search
- ~~V15/V30 gamma optimization~~ — Multi-timeframe fusion gave bad results
- ~~Multi-timeframe fusion~~ — Cross-TF fusion worse than single-timeframe (tested 2026-03-19)
- ~~V1.4 CPCV gamma=1.0~~ — Gamma=1.0 made CPCV valid but killed returns (+4.4% vs +7.4% BTC) and LINK generated 0 signals. Gamma is genuinely needed. Archived.
- ~~CPCV as production gatekeeper~~ — CPCV with gamma is "sloppy" (temporal weights leak across folds). Without gamma, returns collapse. Key finding preserved: 4h models overfit (PBO=1.0), 8h is reliable.
- ~~Deku enhancements~~ — 11 features + 2 toggles tested and all FAILED (2026-03-20). Wavelet overfitting, tristate broke confidence. All code deleted.
- ~~V6 A/B tests~~ — Superseded by Deku enhancement testing
- ~~3h/7h horizons~~ — Performed well in bullish market but poorly in bearish (2026-03-20). Reverted to 4h/8h.

### Completed (Recent)
- **Multi-asset DF test** — DONE (2026-03-21). All 9 assets optimized. LINK 8h best (+16.4%, APF 5.31), ADA 8h (+21.3%), AVAX 8h (+19.6%). BTC and LINK activated for live trading.
- **Multi-asset expansion** — DONE (2026-03-21). Added SOL, LINK, ADA, AVAX, DOT to ASSETS dict. Per-asset DF pipeline (D+F per asset before next).
- **Diversity-aware holdout** — DONE (2026-03-21). Top 10 by APF + best per unexplored combo → up to 20 candidates. Ensures every combo gets a fair holdout shot.
- **Default trials bumped to 150** — DONE (2026-03-21). Was 100. Auto-extend now 150→200→250.
- **Optional enhancements** — DONE (2026-03-21). `--enhancements` flag adds 3 Optuna toggles: return-weighted sampling, ensemble disagreement filter, funding rate gate. BTC test: enhancements had ~5% parameter importance.
- **Horizons reverted to 4h/8h** — DONE (2026-03-20). 3h/7h tested but underperformed 4h/8h in bearish market (+5.6% vs +9.4% Mode F). Constants changed back to HORIZON_SHORT=4, HORIZON_LONG=8.
- **Deku 1.2.1 promoted to production** — DONE (2026-03-20). Fee-aware labels, 4-candle embargo, per-horizon feature ranges, 3-fold rolling holdout, candidate dedup, holdout step=12, Mode F on holdout.
- **Deku promoted to production** — DONE (2026-03-20). Deku beats CASCA: BTC +124.6% vs +52.6%, ETH +156.9% vs +53.7%. CASCA on standby.
- **Deku enhancements tested and FAILED** — DONE (2026-03-20). 11 features + 2 Optuna toggles. Result: -62.2% vs baseline. All code deleted.
- **Auto-extend trials** — DONE (2026-03-20). APF_EXTEND_THRESH=1.7. 100→150→200 trials if APF too low.
- **DXY ticker fix** — DONE (2026-03-20). `DX=F` delisted → `DX-Y.NYB` in `download_macro_data.py`.
- **Mode F horizon bug fix** — DONE (2026-03-20). Mode F was using hardcoded HORIZON_SHORT/HORIZON_LONG instead of CLI-specified horizons. Fixed with `active_horizons`.
- **Deku DF ETH** — DONE (2026-03-19). ETH 4h: +79.8%, APF=5.9. ETH 8h: +77.1%, APF=6.8.
- **Deku DIAG_STEP=36** — DONE (2026-03-19). Doubled eval points. BTC 8h: +77.4%, 29 trades.
- **Deku --metric flag** — DONE (2026-03-19). 5 scoring metrics + `--metric all`.
- **Deku live trader created** — DONE (2026-03-19). `crypto_revolut_deku.py` + `crypto_live_trader_deku.py`.
- **Deku hourly release** — DONE (2026-03-18). Optuna TPE+Hyperband, XGBoost, APF scoring.
