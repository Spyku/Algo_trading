# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Automated ML trading system for crypto (BTC, ETH, XRP, DOGE, SOL, LINK, ADA, AVAX, DOT). Generates hourly BUY/SELL/HOLD signals using ensemble ML models with walk-forward validation, temporal decay sample weighting, and embargo-corrected labels. Variable horizon per asset (5h, 6h, 7h, 8h — optimized via Mode H). Executes trades on Revolut X via Ed25519-signed API.

**Production system:** Doohan V1.7.1 (`crypto_trading_system_doohan.py`). Deku and all prior versions archived (2026-03-24).

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
# Arguments are order-independent: MODE, ASSETS, HORIZONS can appear in any order.
# Run --help for full usage.

# === Doohan (production) — exhaustive grid + Optuna refine + embargo fix ===
python crypto_trading_system_doohan.py H BTC 5,6,7,8h           # Mode H — full horizon sweep (D+G per horizon)
python crypto_trading_system_doohan.py H BTC 5,6,7h --skip      # Mode H — skip D where results exist, re-run G only
python crypto_trading_system_doohan.py DG BTC 6h                 # Mode DG — grid + backtest for single horizon
python crypto_trading_system_doohan.py D BTC 6h                  # Mode D — grid optimization only
python crypto_trading_system_doohan.py D BTC 6h --trials 200     # Custom trial count
python crypto_trading_system_doohan.py G BTC 6h                  # Mode G — re-backtest existing D results
python crypto_trading_system_doohan.py F BTC 5,8h                # Mode F — strategy comparison (multi-horizon)
python crypto_trading_system_doohan.py DGF BTC 6h                # Full pipeline: grid → backtest → strategy
python crypto_trading_system_doohan.py --help                    # Show all modes, options, examples

# === Auto-trader (Doohan — production) ===
python crypto_revolut_doohan.py --loop              # Live trading loop
python crypto_revolut_doohan.py --dry-run --loop    # Signals only, no trades
python crypto_revolut_doohan.py --status            # Show positions
```

**Telegram commands (Doohan trader):** `/stop` `/status` `/pause` `/resume` `/balance` `/sync` `/conf` `/config` `/setup` `/help` `/chart BTC` `/optimize BTC` `/optstatus`

---

## Architecture

### Production File Chain

```
crypto_trading_system_doohan.py  (Doohan V1.7.1 — Modes B/D/G/H/F, grid + refine + embargo)
  └── hardware_config.py  (machine-specific model configs, n_jobs, GPU settings)
crypto_revolut_doohan.py  (auto-trader — reads trading_config_doohan.json)
  └── crypto_live_trader_doohan.py  (signal generation library)
        └── crypto_trading_system_doohan.py  (imports ASSETS, features, models)
```

Deku, CASCA, Doohan V1.1-V1.7, and all legacy systems archived (2026-03-24).

### Key Concepts

- **Variable horizons:** Each asset has its own optimal prediction horizon (5h, 6h, 7h, 8h), stored in `trading_config_doohan.json`. Mode H sweeps horizons to find the best per asset. The trader reads the configured horizon and uses `Xh_only` strategy.
- **Label overlap embargo (V1.7+):** Adjacent rows share overlapping future windows used for labeling. Fix: `EMBARGO_CANDLES = horizon`, so `train_end = i - horizon`. Without this, training data leaks label information. Pre-embargo APFs were inflated 5-26×; post-embargo realistic range is 1.0-3.0.
- **No model persistence:** System retrains from scratch every prediction. No .pkl files. Intentional design.
- **Temporal decay:** Exponential sample weighting `w_i = gamma^(age_in_hours)`. Per-model gamma stored in production CSV. Newest sample weight=1, oldest=gamma^(n-1). gamma=0.995 → half-life ~6 days, gamma=0.999 → ~29 days.
- **6-month data cap:** Mode D caps training data at 4,320 hours (6 months). Hardcoded as `MAX_DIAG_HOURS`.
- **Labels (fee-aware):** `label = 1` when future return > 2×TRADING_FEE (0.22%). Set by `LABEL_MODE = 'fee_aware'`.
- **Walk-forward validation:** Train on last `window` hours → predict next candle → step forward. DIAG_STEP=36. No future leakage.
- **Scoring:** APF (Adjusted Profit Factor) = `raw_PF / buyhold_PF`. Normalizes against market regime.
- **Exhaustive grid (Mode D):** 3 combos × 6 windows × 6 features × 3 gammas = 324 evals per horizon. Saves top 6 candidates.
- **3-fold rolling holdout:** Train on fold 1 (60%), re-rank winners by out-of-sample performance across 3 folds with embargo. Reduces overfitting.
- **Refined-only production selection:** Mode G backtests D candidates to pick top 3 for Optuna refine (50 trials each), then selects production model from refined configs only. D candidates consistently underperform refined versions.
- **MIN_COMBO_SIZE=2:** Solo models removed. Prevents overconfidence from uncalibrated single-model predictions.
- **MIN_TRADES=8:** Optuna objective returns 0 for trials with <8 trades.
- **Models:** RF, GB, XGB, LR, LGBM — 3 viable combos: XGB+LGBM, RF+LGBM, RF+XGB (dead combos RF+GB, RF+LR, GB+LR dropped).
- **Features:** 51 technical + 81 macro/sentiment/cross-asset = 132 total. LGBM importance ranking (~5 sec). logret_5h and logret_7h added (2026-03-25) to fill gaps for 5h/7h horizon models.
- **Model hot-reload:** Trader checks production CSV every 5 minutes.

### Data Flow

```
data/{asset}_hourly_data.csv           <- price data (Binance via ccxt)
data/macro_data/*.csv                  <- VIX, DXY, S&P500, Fear&Greed, etc. (yfinance)
models/crypto_doohan_v1_7_1_best_models.csv    <- top 6 candidates per (asset, horizon) from Mode D
models/crypto_doohan_v1_7_1_production.csv     <- refined production model (written by Mode G)
models/crypto_doohan_v1_7_1_grid_{ASSET}_{H}h.csv <- full grid results (324 evals)
config/trading_config_doohan.json      <- per-asset: horizon, min_confidence, strategy, max_position, enabled
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
MIN_CONFIDENCE = 75         # global fallback only — overridden per asset by Mode G/H
MAX_DIAG_HOURS = 4320       # 6 months data cap for Mode D
REPLAY_HOURS = 200          # Modes B, D signal replay
REPLAY_HOURS_F = 400        # Mode F — longer window for more trades in strategy selection

# Doohan V1.7.1 (production)
GRID_COMBOS = ['RF+LGBM', 'XGB+LGBM', 'RF+XGB']  # 3 viable combos
GRID_WINDOWS = [72, 100, 150, 200, 250, 300]       # 250h sweet spot for most assets
GRID_FEATURES = [5, 10, 15, 20, 25, 30]            # feature counts to test
GRID_GAMMAS = [0.995, 0.997, 0.999]                # gamma values
DIAG_STEP = 36                                      # walk-forward step
DEKU_DEFAULT_TRIALS = 150                           # Optuna trial count
REFINE_TRIALS = 50                                  # Optuna refine trials per config
REFINE_TOP_N = 3                                    # top N D candidates to refine
MODE_G_REPLAY_HOURS = 336                           # 2-week backtest window
MODE_G_CONF_THRESHOLDS = [65, 70, 75, 80, 85, 90]  # confidence thresholds to test
EMBARGO_CANDLES = horizon                           # label overlap fix (dynamic per horizon)
```

---

## Version Files

| File | Status | Notes |
|------|--------|-------|
| `crypto_trading_system_doohan.py` | **Production** | Doohan V1.7.1: Embargo-fixed grid (3×6×6×3=324 evals) + 50-trial Optuna refine. Modes B/D/G/H/F. Variable horizons. Refined-only production selection. Order-independent CLI with `--help`. |
| `crypto_revolut_doohan.py` | **Live** | Auto-trader — reads `trading_config_doohan.json` (horizon per asset) + `crypto_doohan_v1_7_1_production.csv` |
| `crypto_live_trader_doohan.py` | **Live** | Signal generation library — NOT run directly. Reports current market price (not label-shifted). |
| `hardware_config.py` | Active | Auto-detects Desktop (26 workers) / Laptop (14 workers) |
| `download_macro_data.py` | Active | Downloads VIX, DXY, S&P500, NASDAQ, Fear&Greed, etc. |
| `crypto_trading_system_doohan_v1_7_3.py` | Testing | PySR symbolic regression features. Adds discovered formulas as new features. Safe fallback if no PySR JSON exists. |
| `pysr_discover_features.py` | Active | Offline PySR discovery script. Runs symbolic regression on training data, outputs `models/pysr_{ASSET}_{H}h.json`. |
| `crypto_trading_system_doohan_v1_7_2.py` | **Dropped** | Regularization test. Wash — not adopted. |
| `cfd/ib_auto_trader.py` | Live | DAX CFD trader (Broly 1.2) |
| `cfd/ib_auto_trader_test.py` | Live | S&P 500 CFD overnight trader |

All Deku files, Doohan V1.1-V1.7, CASCA, backtests, and testing scripts moved to `archive/` (2026-03-24).

---

## Strategies

| Strategy | Logic |
|----------|-------|
| `Xh_only` | Single horizon (e.g., `6h_only`) — used when `horizon` is set in trading config |
| `both_agree` | BUY when both horizons agree; SELL when either says SELL |
| `either_agree` | BUY when either says BUY; SELL when either says SELL |

**Mode G/H writes the best `horizon` and `min_confidence` to `trading_config_doohan.json`. The trader reads the configured horizon and automatically uses `Xh_only` strategy.**

---

## Current Best Models (Doohan Production)

| Asset | Horizon | Models | Window | Gamma | Features | Return (2wk) | Best Conf | Status |
|-------|---------|--------|--------|-------|----------|--------------|-----------|--------|
| **BTC** | **6h** | XGB+LGBM | 252h | 0.9936 | 9 | +3.75% | 70% | Refined |
| **BTC** | **5h** | RF+LGBM | 202h | 0.9955 | 15 | 0.0% | — | Refined |
| **BTC** | **7h** | RF+LGBM | 251h | 0.9954 | 25 | 0.0% | — | Refined |
| **ETH** | **8h** | XGB+LGBM | 200h | 0.999 | 13 | +5.93% | — | Grid |
| **ETH** | **6h** | RF+XGB | 76h | 0.9997 | 23 | 0.0% | — | Refined |

## Current Trading Config (Doohan)

```json
{
  "BTC": { "horizon": 6, "min_confidence": 90, "max_position_usd": 10000, "enabled": true },
  "ETH": { "horizon": 6, "min_confidence": 90, "max_position_usd": 2000, "enabled": true },
  "SOL": { "min_confidence": 90, "enabled": false },
  "LINK": { "min_confidence": 65, "enabled": false }
}
```

---

## Critical Rules

1. **Never modify `crypto_trading_system_doohan.py` without testing first.** It is production. The live trader imports from it by exact filename.
2. **CSV merge logic:** When saving to production CSV, always filter by BOTH coin AND horizon.
3. **`generate_signals()` needs `feature_override`:** When feature_set is D, caller must pass `feature_override=config['optimal_features'].split(',')`.
4. **All timestamps in live trader use Europe/Zurich** via `_to_local()` helper.
5. **`trading_config_doohan.json` has `horizon` and `min_confidence` per asset** — set by Mode G/H. Global `MIN_CONFIDENCE=75` is only a fallback.
6. **SSL fix on Windows:** `ssl._create_unverified_context()` applied in both `crypto_revolut_doohan.py` and `crypto_live_trader_doohan.py`.
7. **Refined-only production:** Mode G only selects from refined configs for production. D candidates are still backtested (needed to rank top 3 for refine input).

---

## Pending Work

### Active
1. **SOL Mode H horizon sweep** — Running on Desktop. `python crypto_trading_system_doohan.py H SOL 5,6,7,8h`
2. **Expand to other assets** — Run Mode H for LINK, XRP, DOGE, ADA, AVAX, DOT after SOL.
3. **V1.7.3 — PySR symbolic regression** (in progress). PySR discovery done for BTC 6h (`models/pysr_BTC_6h.json`). Currently testing `python crypto_trading_system_doohan_v1_7_3.py DG BTC 6h`. Compare against V1.7.1 baseline (+3.75% at 70%).
   - **Install PySR on Desktop:** `C:\algo_trading\venv\Scripts\pip.exe install pysr sympy` (TODO)

### To Test (ML improvements)
1. **LSTM regime embedding** — Low priority. Train LSTM daily → 2-3 dim embedding → feed to trees. Requires .pkl persistence.

### Completed (2026-03-25)
- **ETH Mode H horizon sweep** — All 4 horizons (5h/6h/7h/8h) grid complete. 5h/6h/7h refined. 8h best grid result (+5.93%). Config set to 6h/90%.
- **BTC 4h/8h with embargo fix** — 4h confirmed overfit (negative post-embargo). 8h not viable. BTC production stays at 6h.
- **V1.7.2 — Regularization** — Tested on BTC 6h. Minimal reg won (ra=0, rl=0.1, cs=0.9, ss=0.5). V1.7.1 baseline (+3.75/+3.47/+3.74 at 70/80/90%) more consistent than V1.7.2 (+3.63/+0.11/+4.64). **Verdict: wash, not adopted.**
- **PySR installed on Laptop** — `pip install pysr sympy` done (2026-03-25). Julia 1.11.9 backend compiled.

### Completed (2026-03-24)
- **Doohan V1.7.1 promoted to production** — Renamed to `crypto_trading_system_doohan.py`. Embargo-fixed grid + Optuna refine. Variable horizon per asset. BTC 6h winner: XGB+LGBM w=252h g=0.994 f=9, +3.75% at conf>=70%.
- **Deku archived** — All Deku files moved to `archive/`. Deku replaced by Doohan as production system.
- **Refined-only production selection** — Mode G now only picks from refined configs. D candidates consistently underperform (confirmed on BTC 5h/6h/7h).
- **Variable horizon support** — Trading config stores `horizon` per asset. Trader reads it and uses `Xh_only`. Mode H sweeps multiple horizons.
- **New Mode H** — Horizon sweep: D+G per horizon → cross-horizon comparison → saves best to trading config. `--skip` flag to reuse existing D results.
- **Order-independent CLI** — Arguments can appear in any order. `--help` shows full usage.
- **Price fix ported** — Live trader now reports current market price (`df_raw.iloc[-1]['close']`) instead of label-shifted historical price.
- **Dead code cleanup** — ~1,435 lines removed (legacy modes A/E/DAF, old Mode H, legacy Mode D).
- **Root folder cleanup** — 28 files archived. Root now has 7 Python files + configs.
- **Mode G confidence thresholds expanded** — Tests 65/70/75/80/85/90% (was 70/80/90).

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
