# Algo Trading Engine

Automated ML trading system for **crypto** (BTC, ETH, XRP, DOGE, SOL, LINK, ADA, AVAX, DOT) and **index CFDs** (DAX, S&P 500). Generates hourly BUY/SELL/HOLD signals using ensemble ML models with walk-forward validation, temporal decay sample weighting, and embargo-corrected labels. Variable horizon per asset (5h, 6h, 7h, 8h). Executes trades on Revolut X via Ed25519-signed API.

**Production systems:**
- **Doohan V1.7.1** — Fixed-horizon trading with PySR symbolic regression features (stable)
- **Ed V1.0** — Regime-switching trading: bull/bear detection with dynamic horizon selection (production)

**Testing:**
- **Ein V1.0** — 15-minute candles, horizons 4-10 candles (1h-2h30), grid windows 12h-120h
- **Eli V1.0** — 30-minute candles, horizons 4-10 candles (2h-5h), grid windows 12h-120h

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Features (132)](#features-132)
- [Strategies](#strategies)
- [Auto-Trader](#auto-trader)
- [Optimizer Bot](#optimizer-bot)
- [Current Models](#current-models)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Key Constants](#key-constants)
- [Version History](#version-history)

---

## Quick Start

```bash
# Activate venv first (required)
# Desktop:  C:\algo_trading\venv\Scripts\activate.bat
# Laptop:   C:\Users\Alex\algo_trading\venv\Scripts\activate.bat

# === Model optimization ===
python crypto_trading_system_doohan.py DV BTC 6h                 # Grid + validate for single horizon
python crypto_trading_system_doohan.py H BTC 5,6,7,8h            # Full horizon sweep
python crypto_trading_system_doohan.py P BTC 6h                  # PySR feature discovery

# === Live trading (Doohan -- production) ===
start_trader.bat                                    # Auto-restart wrapper
python crypto_revolut_doohan.py --loop              # Direct (no auto-restart)
python crypto_revolut_doohan.py --dry-run --loop    # Signals only, no trades
python crypto_revolut_doohan.py --status            # Show positions

# === Live trading (Ed -- regime-switching) ===
start_ed.bat                                        # Auto-restart wrapper
python crypto_revolut_ed.py --loop                  # Direct
python crypto_revolut_ed.py --dry-run --loop        # Signals only, no trades

# === Ed regime backtest ===
python crypto_trading_system_ed.py R BTC 5,6,7,8h              # Test all regime detectors
python crypto_trading_system_ed.py R BTC --replay 2880         # 4-month backtest
python tools/pysr_discover_regime.py --bull 6 --bear 8         # PySR regime formula discovery

# === Optimizer bot (remote optimization via Telegram) ===
start_optimizer.bat                                 # Auto-restart wrapper
python crypto_optimizer_bot.py                      # Direct
```

### All Optimization Modes

```bash
# Arguments are order-independent: MODE, ASSETS, HORIZONS can appear in any order.
python crypto_trading_system_doohan.py P BTC 6h                  # Mode P -- PySR feature discovery (~30-120 min)
python crypto_trading_system_doohan.py H BTC 5,6,7,8h            # Mode H -- full horizon sweep (D+V per horizon)
python crypto_trading_system_doohan.py H BTC 5,6,7h --skip       # Mode H -- skip D where results exist, re-run V only
python crypto_trading_system_doohan.py DV BTC 6h                 # Mode DV -- grid + validate for single horizon
python crypto_trading_system_doohan.py D BTC 6h                  # Mode D -- grid optimization only
python crypto_trading_system_doohan.py D BTC 6h --trials 200     # Custom trial count
python crypto_trading_system_doohan.py V BTC 6h                  # Mode V -- re-validate existing D results
python crypto_trading_system_doohan.py S BTC 5,8h                # Mode S -- strategy comparison (multi-horizon)
python crypto_trading_system_doohan.py DVS BTC 6h                # Full pipeline: grid -> validate -> strategy
python crypto_trading_system_doohan.py --help                    # Show all modes, options, examples
```

---

## Architecture

### Production File Chains

```
# Doohan (fixed horizons -- stable production)
crypto_trading_system_doohan.py  (V1.7.1 -- Modes P/D/V/H/S)
  +-- hardware_config.py
crypto_revolut_doohan.py  (auto-trader -- reads trading_config_doohan.json)
  +-- crypto_live_trader_doohan.py  (signal generation)
        +-- crypto_trading_system_doohan.py

# Ed (regime-switching -- dynamic horizons)
crypto_trading_system_ed.py  (V1.0 -- Modes P/D/V/H/S/R/HRS)
  +-- hardware_config.py
crypto_revolut_ed.py  (auto-trader -- reads regime_config_ed.json)
  +-- crypto_live_trader_ed.py  (regime-aware signal generation)
        +-- crypto_trading_system_ed.py

# Shared
crypto_optimizer_bot.py  (Telegram bot for remote optimization)
  +-- crypto_trading_system_doohan.py  (spawned as subprocess)
```

### Index CFDs (separate system)

```
cfd/ib_auto_trader.py       (DAX CFD trader -- Broly 1.2)
cfd/ib_auto_trader_test.py  (S&P 500 CFD overnight)
cfd/broly.py                (enhancement layer -- regime detection)
```

---

## How It Works

### Core Concepts

| Concept | Detail |
|---------|--------|
| **Variable horizons** | Each asset has its own optimal prediction horizon (5h, 6h, 7h, 8h), found by Mode H and stored in `trading_config_doohan.json` |
| **No model persistence** | Retrains from scratch every prediction. No .pkl files. Always uses latest market data |
| **Temporal decay** | Exponential sample weighting `w_i = gamma^(age)`. gamma=0.995 -> half-life ~6 days, gamma=0.999 -> ~29 days |
| **6-month data cap** | Training capped at 4,320 hours. Prevents stale data from diluting recent patterns |
| **Fee-aware labels** | `label = 1` when future return > 2x TRADING_FEE (0.22%). Only labels profitable moves as positive |
| **Label overlap embargo** | Adjacent rows share overlapping future windows. Fix: `EMBARGO_CANDLES = horizon`. **Backtesting only** — never in live signal generation. In live, all labels use known past prices (no leakage). Per Lopez de Prado. |
| **Walk-forward validation** | Train on last `window` hours -> predict next candle -> step forward (DIAG_STEP=36). No future leakage |
| **Ensemble voting** | Majority vote across 2-model combo. Confidence = average probability across models |
| **PySR symbolic regression** | Offline discovery of mathematical expressions from historical data. Anti-leakage: formulas discovered on months 12->6 ago only, never overlapping Mode D's evaluation window |

### ML Pipeline (Mode D)

1. **Data download** -- Hourly candles (Binance via ccxt) + macro data (yfinance)
2. **Feature engineering** -- 132 features (technical + macro + sentiment + cross-asset + PySR symbolic)
3. **Feature ranking** -- LGBM gain importance ranks all features (~5 sec)
4. **Exhaustive grid search** -- 3 combos x 6 windows x 6 features x 3 gammas = 324 evals
5. **3-fold rolling holdout** -- Re-rank winners by out-of-sample performance with embargo
6. **Walk-forward evaluation** -- Train -> predict -> step forward. Score by APF
7. **Save** -- Top 6 candidates to `models/crypto_doohan_v1_7_1_best_models.csv`

### Mode V (Validate + Refine)

1. **Backtest** D candidates across 6 confidence thresholds (65-90%)
2. **Select top 3** for Optuna refinement (50 trials each)
3. **Production selection** from refined configs only using `return x (win_rate/100)` scoring
4. **Save** winner to `models/crypto_doohan_v1_7_1_production.csv` and update `trading_config_doohan.json`

### Modes

| Mode | Purpose | When to Use |
|------|---------|-------------|
| **P** | PySR symbolic feature discovery | Before D/V to add symbolic features (~30-120 min) |
| **D** | Exhaustive grid optimization | After market regime change, or periodically |
| **V** | Validate + Optuna refine -> writes production model | After Mode D |
| **DV** | D then V in one command | Standard workflow |
| **H** | Horizon sweep (D+V per horizon -> compare -> best) | Find optimal horizon per asset |
| **S** | Strategy comparison (multi-horizon) | Compare strategies across horizons |
| **DVS** | Full pipeline: D -> V -> S | Complete optimization |

---

## Features (132)

| Category | Count | Examples |
|----------|-------|---------|
| **Technical** | 51 | Log returns (1-240h), RSI, Bollinger Bands, ATR, ADX/DI, Garman-Klass vol, volatility ratios, Stochastic, spread ratios, SMA ratios, hour sin/cos |
| **Macro** | 40 | VIX (level, zscore, regime), DXY, S&P500/Nasdaq changes (1/5/10d), US10Y, EUR/USD, USD/JPY, Oil, Gold volatility |
| **Sentiment** | 25 | Fear & Greed Index (value, zscore, changes, MA, extreme flags) |
| **Cross-asset** | 16 | BTC/ETH/DAX/Nasdaq/S&P500 rolling correlation (10/30d), relative strength (5d) |
| **PySR symbolic** | variable | Auto-loaded from `models/pysr_{ASSET}_{H}h.json` if available; safe fallback if not |

---

## Strategies

| Strategy | BUY Condition | SELL Condition |
|----------|--------------|----------------|
| `Xh_only` | Single horizon says BUY (e.g., `6h_only`) | Horizon says SELL |
| `both_agree` | Both horizons agree BUY (>= min_confidence) | Either says SELL |
| `either_agree` | Either horizon says BUY | Either says SELL |

Mode V/H writes the best `horizon` and `min_confidence` to `trading_config_doohan.json`. The trader reads the configured horizon and uses `Xh_only` strategy.

---

## Auto-Trader

### Startup

Run `start_trader.bat` (recommended) -- auto-restarts if the bot crashes. Or run directly: `python crypto_revolut_doohan.py --loop`

### Loop Cycle

1. **Startup** -- Sync positions with Revolut X -> Telegram notification -> immediate scan
2. **Every hour** -- Download data -> generate signals -> apply strategy -> execute trades -> Telegram
3. **Every 5 min** -- Position sync (detects manual trades), model + config hot-reload
4. **Every 5 sec** -- Poll Telegram for commands

### Telegram Commands

| Command | Action |
|---------|--------|
| `/status` | Current positions and PnL |
| `/balance` | Exchange account balance |
| `/stop` | Graceful shutdown |
| `/pause` / `/resume` | Pause/resume trading |
| `/sync` | Force position sync |
| `/conf` / `/config` | Show current config |
| `/setup` | Inline button config editor |
| `/chart BTC` | Generate and send chart |
| `/help` | List all commands |

### Authentication

Asymmetric Ed25519 signature: `timestamp + method + path + body -> base64 signature`.

---

## Optimizer Bot

Separate Telegram bot for remote model optimization. Run `start_optimizer.bat` (auto-restart) or `python crypto_optimizer_bot.py`.

Inline keyboard menus to select mode/assets/horizons. Sequential job queue with subprocess execution and real-time progress output. Uses separate bot token from the trader.

### Telegram Commands

| Command | Action |
|---------|--------|
| `/optimize` | Start optimization flow (Mode -> Assets -> Horizons -> Confirm) |
| `/queue` | Show running/pending/completed jobs |
| `/cancel` | Cancel current job or menu flow |
| `/status` | Show current production models |
| `/results` | Show last results for an asset |
| `/help` | List commands |
| `/stop` | Stop the bot |

---

## Regime Backtest

`tools/backtest_regime_master.py` tests whether dynamically switching between horizons based on market regime outperforms a fixed single-horizon strategy. For example, using the 6h model in bull markets (SMA24 > SMA100) and the 8h model in bear markets.

```bash
python tools/backtest_regime_master.py                         # 2-month default, all horizons
python tools/backtest_regime_master.py --months 4              # 4-month backtest
python tools/backtest_regime_master.py --horizons 6,8          # only test 6h and 8h (fast)
python tools/backtest_regime_master.py --bull 6 --bear 8       # fix pair, compare regimes only
python tools/backtest_regime_master.py --regimes sma,rsi       # filter regime families
python tools/backtest_regime_master.py --no-combos             # single-horizon baselines only
python tools/backtest_regime_master.py --asset ETH             # test other assets
```

---

## Current Models

### Doohan V1.7.1 Production

| Asset | Horizon | Models | Window | Gamma | Features | Min Conf | Status |
|-------|---------|--------|--------|-------|----------|----------|--------|
| **BTC** | **6h** | XGB+LGBM | 209h | 0.9938 | 31 | 85% | Refined+PySR |
| **BTC** | **5h** | RF+XGB | 159h | 0.9957 | 20 | -- | Refined+PySR |
| **ETH** | **7h** | RF+LGBM | 247h | 0.9997 | 8 | 90% | Refined+PySR |
| **ETH** | **6h** | XGB+LGBM | 117h | 0.9944 | 11 | -- | Refined+PySR |
| **ETH** | **8h** | RF+XGB | 250h | 0.999 | 10 | -- | Grid+PySR |
| **XRP** | **8h** | XGB+LGBM | 150h | 0.995 | 17 | 80% | Grid+PySR |
| **SOL** | **6h** | RF+XGB | 100h | 0.995 | 17 | 90% | Grid+PySR |
| **LINK** | **7h** | XGB+LGBM | 311h | 0.9974 | 8 | 90% | Refined+PySR |

### Trading Config

```
BTC:  6h_only      @85%  ($12,000 max)  enabled
ETH:  both_agree   @90%  ($2,000 max)   enabled
XRP:  both_agree   @80%                  disabled
SOL:  both_agree   @90%                  disabled
LINK: both_agree   @90%                  disabled
DOGE: both_agree   @80%                  disabled
ADA:  both_agree   @80%                  disabled
AVAX: both_agree   @90%                  disabled
DOT:  both_agree   @80%                  disabled
```

---

## Project Structure

```
engine/
+-- crypto_trading_system_doohan.py    # Doohan V1.7.1 -- Modes P/D/V/H/S
+-- crypto_revolut_doohan.py           # Doohan auto-trader (reads trading_config_doohan.json)
+-- crypto_live_trader_doohan.py       # Doohan signal generation (not run directly)
+-- crypto_trading_system_ed.py        # Ed V1.0 -- Modes P/D/V/H/S/R/HRS
+-- crypto_revolut_ed.py               # Ed auto-trader (reads regime_config_ed.json)
+-- crypto_live_trader_ed.py           # Ed regime-aware signal generation
+-- crypto_optimizer_bot.py            # Telegram bot for remote optimization
+-- hardware_config.py                 # Auto-detect Desktop/Laptop config
+-- download_macro_data.py             # Macro/sentiment/cross-asset downloader
+-- pysr_discover_features.py          # Offline PySR discovery (historical window)
+-- start_trader.bat                   # Doohan trader launcher with auto-restart
+-- start_ed.bat                       # Ed trader launcher with auto-restart
+-- start_optimizer.bat                # Optimizer bot launcher with auto-restart
+-- git_push.bat                       # Git push helper
|
+-- cfd/
|   +-- ib_auto_trader.py             # DAX CFD trader (Broly 1.2)
|   +-- ib_auto_trader_test.py        # S&P 500 CFD overnight
|   +-- broly.py                      # Enhancement layer (regime detection)
|
+-- tools/
|   +-- check_balance.py              # Exchange balance
|   +-- check_trades.py               # Trade history
|   +-- debug_price.py                # API price diagnostic
|   +-- revolut_x_test.py             # API connectivity test
|   +-- detect_hardware.py            # Hardware detection -> config
|   +-- buy_btc.py                    # Manual BTC purchase
|   +-- backtest_regime_master.py      # Regime detector backtest (21 detectors x horizon pairs)
|   +-- pysr_discover_regime.py       # PySR regime formula discovery
|   +-- backtest_v5_v15.py            # Legacy backtest comparison
|   +-- ib_test_connection.py         # IB broker connectivity test
|
+-- data/
|   +-- {asset}_hourly_data.csv       # Hourly OHLCV (Binance)
|   +-- macro_data/                   # VIX, DXY, S&P500, Fear&Greed, etc.
|   +-- indices/                      # DAX, S&P500, SMI, CAC40 OHLCV
|
+-- models/
|   +-- crypto_doohan_v1_7_1_production.csv     # Doohan production model (written by Mode V)
|   +-- crypto_doohan_v1_7_1_best_models.csv    # Top 6 candidates per (asset, horizon)
|   +-- crypto_doohan_v1_7_1_grid_*.csv         # Full grid results (324 evals)
|   +-- crypto_ed_production.csv                # Ed production model
|   +-- pysr_{ASSET}_{H}h.json                  # PySR symbolic expressions
|   +-- pysr_{ASSET}_{H}h_report.txt            # PySR discovery reports
|
+-- config/                           # NOT in git -- credentials + state
|   +-- trading_config_doohan.json    # Doohan: per-asset horizon, min_confidence, strategy
|   +-- regime_config_ed.json         # Ed: per-asset detector, bull/bear horizon+confidence+position
|   +-- revolut_x_config.json         # Exchange API key
|   +-- private.pem                   # Ed25519 signing key
|   +-- telegram_config.json          # Bot token (trader)
|   +-- telegram_optimizer_config.json # Bot token (optimizer -- separate)
|   +-- position_{ASSET}.json         # Doohan position tracking
|   +-- position_ed_{ASSET}.json      # Ed position tracking (separate)
|   +-- signal_log.csv                # Signal history (for /chart)
|
+-- logs/                             # Auto-generated by start_trader.bat / optimization runs
+-- charts/                           # Backtest PNGs + interactive HTML
+-- archive/                          # All legacy versions (Deku, CASCA, V1.1-V1.7, etc.)
+-- CLAUDE.md                         # Claude Code instructions
+-- README.md
```

---

## Setup

### Hardware

One shared engine folder synced via Google Drive. Only the venv is local per machine.

| Machine | Engine Path | Venv | CPU | GPU |
|---------|-------------|------|-----|-----|
| Desktop (primary) | `G:\Autres ordinateurs\My laptop\engine\` | `C:\algo_trading\venv\` | i7-14700KF | RTX 4080 |
| Laptop | `G:\Autres ordinateurs\My laptop\engine\` | `C:\Users\Alex\algo_trading\venv\` | 16 cores | RTX 3070 Ti |

`hardware_config.py` auto-detects Desktop (26 workers) vs Laptop (14 workers). LGBM uses GPU (`device='gpu'`).

### Installation

```powershell
# Desktop
python -m venv C:\algo_trading\venv
C:\algo_trading\venv\Scripts\activate.bat
pip install -r "G:\Autres ordinateurs\My laptop\engine\requirements.txt"

# Laptop
python -m venv C:\Users\Alex\algo_trading\venv
C:\Users\Alex\algo_trading\venv\Scripts\activate.bat
pip install -r "G:\Autres ordinateurs\My laptop\engine\requirements.txt"
```

### Key Dependencies

`pandas`, `numpy`, `scikit-learn`, `lightgbm` (GPU), `xgboost`, `optuna`, `ccxt`, `yfinance`, `pynacl`, `cryptography`, `matplotlib`, `joblib`, `pysr`

---

## Key Constants

```python
# Trading costs
TRADING_FEE_BASE = 0.0009       # 0.09% Revolut X taker fee (0% maker)
SLIPPAGE = 0.0002               # 0.02% estimated slippage
TRADING_FEE = 0.0011            # total cost per trade (fee + slippage)

# Grid search (Mode D)
GRID_COMBOS = ['RF+LGBM', 'XGB+LGBM', 'RF+XGB']  # 3 viable combos
GRID_WINDOWS = [72, 100, 150, 200, 250, 300]       # window sizes
GRID_FEATURES = [5, 10, 15, 20, 25, 30]            # feature counts
GRID_GAMMAS = [0.995, 0.997, 0.999]                # temporal decay values
# Total: 3 x 6 x 6 x 3 = 324 evaluations per horizon

# Walk-forward
DIAG_STEP = 36                  # walk-forward step size
MAX_DIAG_HOURS = 4320           # 6-month data cap
EMBARGO_CANDLES = horizon       # label overlap fix (dynamic per horizon)

# Optuna refinement
DEKU_DEFAULT_TRIALS = 150       # Optuna trial count
REFINE_TRIALS = 50              # Optuna refine trials per config
REFINE_TOP_N = 3                # top N D candidates to refine

# Validation
MODE_G_REPLAY_HOURS = 336       # 2-week backtest window
MODE_G_CONF_THRESHOLDS = [65, 70, 75, 80, 85, 90]
MIN_CONFIDENCE = 75             # global fallback (overridden per asset by Mode V/H)
MIN_COMBO_SIZE = 2              # no solo models
MIN_TRADES = 8                  # reject unreliable configs
```

---

## Pending Work

### Running
1. **ETH RS 1-month on Desktop** -- `RS ETH 6,7,8h --replay 720` to compare 1mo vs 2mo results

### TODO
1. **Eli HRS BTC** -- `python crypto_trading_system_eli.py HRS BTC 4,5,6,7,8,9,10` -- 30-minute candle test
2. **Ein results review** -- Check Ein (15min) BTC results from laptop run
3. **Consider ETH-only trading** -- ETH consistently outperforms BTC in backtests. Evaluate dropping BTC.

### Completed (2026-03-31)
- **ETH RS 2-month** -- Mode R: sma24>sma100 bull=6h bear=8h +59.58% (74% WR). Mode S: 6h@85%/8h@65% → +70.01% (67% WR). R→S pipeline fix confirmed working.
- **HRS R→S pipeline fix** -- Mode R now writes winning horizons to config before Mode S runs. Previously S ignored R's findings and used pre-set horizons.
- **LINK HRS complete** -- Weak results. Best: 5h +3.97%. All other horizons negative. Not worth trading.
- **Take-profit analysis** -- No TP is best. TP=1% boosts WR (67→72%) but cuts return nearly in half (+81→+49%). Model SELL signals capture bigger moves.
- **No-embargo-in-live confirmed** -- Per Lopez de Prado. Embargo for backtesting only. Reverted live embargo change.

### Completed (2026-03-29)
- **Ed V1.0 tested and bug-fixed** -- Fixed: max_position_usd always 0 (BUY never executed), shared position files with Doohan, unnecessary model loading, stale variable refs. All signal generation tests pass.
- **Ed V1.0 release** -- Regime-switching system: bull/bear detection with dynamic horizon/confidence. Mode R backtest. External regime config (not hardcoded). Runs alongside Doohan.
- **Regime backtest tool** -- `tools/backtest_regime_master.py`: tests all horizon pairs × regime detectors with monthly breakdown. Supports --months, --horizons, --bull/--bear, --regimes, --asset flags.
- **All 9 assets Mode H complete** -- BTC, ETH, SOL, LINK, XRP, DOGE, ADA, AVAX, DOT all have production models. AVAX negative (skip).
- **SOL disabled** -- Turned off from Doohan live trading
- **Trailing stop / regime filter analysis** -- Tested on 2-week replay. Baseline signal exits beat all trailing stop variants. Model signal quality is the edge.

---

## Version History

| Date | Milestone |
|------|-----------|
| **2026-03-29** | **Ed V1.0 release.** Regime-switching system: dynamic bull/bear horizon selection via external config. Mode R regime backtest. PySR regime discovery script. Runs alongside Doohan. SOL disabled from Doohan. Trailing stop analysis completed (baseline wins). |
| **2026-03-27** | BTC -7.5% selloff. Full Mode H re-run for BTC with crash data. All 9 assets Mode H complete. AVAX all horizons negative. |
| **2026-03-26** | V1.8 LSTM test -- FAILED. LSTM solo: 0 valid results. LSTM combos identical to RF combos. Not adopted. BTC Mode P re-run. LINK Mode P + H started. SOL Mode H completed (8h winner, +22.43%). |
| **2026-03-25** | Telegram optimizer bot. PySR leakage fix (historical window). PySR promoted to production. ETH + SOL Mode H. BTC embargo-verified (4h overfit, 6h production). V1.7.2 regularization test -- wash, not adopted. |
| **2026-03-24** | Doohan V1.7.1 promoted to production. Deku archived. Variable horizon support. New Mode H. Order-independent CLI. Dead code cleanup (~1,435 lines). Root folder cleanup (28 files archived). |
| **2026-03-23** | LGBM dominance proven -- 26 combos reduced to 6 signal-distinct groups. Doohan V1.3 multi-seed validated. CPCV investigation concluded (dropped). |
| **2026-03-22** | Telegram UX overhaul (candlestick charts, inline buttons). CPCV tested and dropped. |
| **2026-03-21** | Multi-asset expansion -- all 9 assets optimized. BTC + LINK activated for live trading. |
| **2026-03-20** | Deku promoted to production. 3-fold holdout validation. |
| **2026-03-18** | Deku release -- Optuna TPE+Hyperband, XGBoost, APF scoring, LGBM importance. |
| **2026-03-15** | CASCA release -- profit factor scoring. Temporal decay (Cacarot). |
| **2026-03-10** | V5 production. Mode F strategy selection. |
| **2026-03-04** | Dual horizon (4h+8h). Derivative features. |
| **2026-02-22** | Broly 1.2 -- IB auto-trader for DAX CFDs. |
| **2026-02-21** | Initial commit. |

### Evolution

Ed V1.0 (regime) + Doohan V1.7.1 (fixed) > Doohan V1.6 > Deku > CASCA > V5 > V4 > V3. All legacy systems archived.
