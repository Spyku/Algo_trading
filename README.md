# Algo Trading Engine

Automated ML-powered trading system for **crypto** (BTC, ETH, XRP) and **index CFDs** (DAX, S&P 500). Two independent pipelines:

1. **Crypto Pipeline** — Hourly BUY/SELL/HOLD signals via dual-horizon (4h + 8h) ensemble ML with walk-forward validation. Executes on **Revolut X** via Ed25519-signed API.
2. **Index CFD Pipeline** — Hourly signals via Broly 1.2 ML model. Executes on **Interactive Brokers** via ib_insync API.

**Owner:** Alex, Lausanne, Switzerland (CET/CEST)
**GitHub:** https://github.com/Spyku/Algo_trading

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Crypto Trading System (V5)](#crypto-trading-system-v5)
  - [Modes of Operation](#modes-of-operation)
  - [ML Pipeline](#ml-pipeline)
  - [Features](#features-125-total)
  - [Scoring Formula](#v5-scoring-formula)
  - [Strategies](#strategies)
  - [Auto-Trader (Revolut X)](#revolut-x-auto-trader)
- [Index CFD System (IB)](#index-cfd-system-interactive-brokers)
- [All Files Reference](#all-files-reference)
- [Folder Structure](#folder-structure)
- [Hardware Setup](#hardware-setup)
- [Installation](#installation)
- [Current Best Models](#current-best-models)
- [Version History](#version-history)
- [Development Timeline](#development-timeline)

---

## Quick Start

```bash
# Activate venv (required before any python command)
# Desktop: C:\algo_trading\venv\Scripts\activate.bat
# Laptop:  C:\Users\Alex\algo_trading\venv\Scripts\activate.bat

# === CRYPTO SIGNALS ===
python crypto_trading_system.py 5              # Quick BTC (both horizons)
python crypto_trading_system.py 6              # Quick ETH
python crypto_trading_system.py 7              # Quick XRP
python crypto_trading_system.py B BTC 4,8h     # Mode B — signals from saved models
python crypto_trading_system.py D BTC 4,8h 2y  # Mode D — full pipeline (~90 min/horizon)
python crypto_trading_system.py F BTC          # Mode F — strategy comparison

# === CRYPTO AUTO-TRADER ===
python crypto_revolut_trader.py --loop         # Live trading loop
python crypto_revolut_trader.py --dry-run --loop  # Signals only, no trades
python crypto_revolut_trader.py --status       # Show positions
python crypto_revolut_trader.py --balance      # Revolut X balance

# === INDEX CFD TRADER (IB) ===
python ib_auto_trader.py --loop                # DAX continuous trading
python ib_auto_trader.py --status              # Show IB positions
python ib_auto_trader_test.py --loop           # S&P 500 overnight trading

# === SETUP ===
powershell -ExecutionPolicy Bypass -File setup_algo_trading.ps1
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CRYPTO PIPELINE                              │
│                                                                     │
│  crypto_trading_system.py  (V5 — 3,519 lines)                      │
│    ├── Modes B/D/E/F + shortcuts 5/6/7                              │
│    ├── 125 features (44 technical + 81 macro/sentiment/cross-asset) │
│    ├── 4 ML models × 15 combinations × 5 windows                   │
│    ├── Walk-forward validation (no future leakage)                  │
│    └── hardware_config.py (machine-specific GPU/CPU settings)       │
│                                                                     │
│  crypto_live_trader.py  (506 lines — signal generation library)     │
│    ├── Imports models/features from crypto_trading_system.py        │
│    ├── Trains on latest window → predicts next candle               │
│    └── Telegram notifications (HTML formatted)                      │
│                                                                     │
│  crypto_revolut_trader.py  (1,109 lines — multi-asset auto-trader) │
│    ├── Ed25519-signed Revolut X API (buy/sell/balance)              │
│    ├── Per-asset strategies from trading_config.json                │
│    ├── Position sync (detects manual trades on exchange)            │
│    ├── Telegram commands (/stop /status /pause /resume /balance)    │
│    └── Hourly loop with 5-min position sync                        │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                     INDEX CFD PIPELINE                               │
│                                                                     │
│  ib_auto_trader.py  (1,444 lines — DAX CFD trader)                 │
│    ├── Broly 1.2 ML model with V2 features                         │
│    ├── Interactive Brokers via ib_insync                             │
│    ├── Risk manager (daily loss limit, cooldown, stop-loss)         │
│    └── Live HTML dashboard export                                   │
│                                                                     │
│  ib_auto_trader_test.py  (1,419 lines — S&P 500 CFD overnight)    │
│    └── Same architecture, different asset + trading hours            │
│                                                                     │
│  broly.py  (~53KB — enhancement layer)                              │
│    ├── Market regime detection (BULL/BEAR/SIDEWAYS)                 │
│    ├── Graduated 5-tier signals (STRONG_BUY → STRONG_SELL)          │
│    └── Discord + Telegram alerts                                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Key design principle:** The crypto and IB systems are **completely independent** — different models, different assets, different brokers, different configs. The only shared concept is the feature engineering approach.

---

## Crypto Trading System (V5)

### Core Concepts

| Concept | Detail |
|---------|--------|
| **Dual horizons** | 4h and 8h — completely independent models per horizon (different features, window, combo). Never mixed during training. |
| **No model persistence** | System retrains from scratch every prediction. No .pkl files. Ensures fresh models with current market regime. |
| **Labels** | Relative to 168h rolling median return, not absolute price direction. `label = 1` means return > median. Drift-robust. |
| **Walk-forward** | Train on last `window` hours → predict next candle → step forward 72h. No future leakage. |
| **Ensemble voting** | Majority vote across model combo (>50% BUY = BUY signal). Confidence = avg probability. |

### Modes of Operation

| Mode | Purpose | Time (desktop) | When to Run |
|------|---------|----------------|-------------|
| **B** | Quick signals from saved models | ~2 min | Daily |
| **D** | Full pipeline: 125 features → 5-test analysis → optimal subset → diagnostic grid search → signals | ~90 min per horizon | Quarterly or after market regime change |
| **E** | Iterative refinement (leave-one-out, add-back, finer window grid) | ~1–4h | After Mode D |
| **F** | Strategy comparison + confidence sweep → updates trading_config.json | ~seconds | After Mode D |
| **5/6/7** | Shortcuts: Quick BTC/ETH/XRP (Mode B, both horizons) | ~5 min | Daily |

#### Mode D Full Pipeline (per asset, per horizon)

1. **Macro update** — Download VIX, DXY, S&P500, Nasdaq, Gold, US10Y, EUR/USD, USD/JPY, Oil, Fear&Greed (~0.1 min)
2. **Data update** — Download latest hourly candles from Binance via ccxt (~0.1 min)
3. **Feature build** — Compute all 125 features (technical + macro + sentiment + cross-asset)
4. **Feature analysis** — 5 parallel tests:
   - LGBM gain importance (top features by information gain)
   - Permutation importance (shuffle-and-measure accuracy + alpha drop)
   - Ablation (drop-one-at-a-time impact)
   - Reduced sets (top-N feature subsets)
   - Consensus scoring → keep/maybe/drop bins → optimal subset
5. **Diagnostic** — Grid search: 15 model combos × 5 windows = 75 configs, parallel walk-forward evaluation (~16–18 min)
6. **Signal generation** — Generate signals with best config, bootstrap CI, portfolio simulation (~2 min)
7. **Charts** — Matplotlib backtest PNG + Plotly interactive HTML (4-panel strategy charts + signal table)
8. **Save** — Best model to `crypto_hourly_best_models.csv`, features to analysis CSV

### ML Pipeline

**Models available:** Random Forest (RF), Gradient Boosting (GB), Logistic Regression (LR), LightGBM (LGBM with GPU)

**All 15 combinations tested:**
- 4 solo: RF, GB, LR, LGBM
- 6 pairs: RF+GB, RF+LR, RF+LGBM, GB+LR, GB+LGBM, LR+LGBM
- 4 triples: RF+GB+LR, RF+GB+LGBM, RF+LR+LGBM, GB+LR+LGBM
- 1 quad: RF+GB+LR+LGBM

**Walk-forward validation:**
```
Train: [i-window : i]  →  Predict: row i  →  Step: i += 72h
Min start = window + 50 (warm-up period)
```

**Portfolio simulation:** Entry at `price × (1 + 0.0009)`, exit at `price × (1 - 0.0009)`. Tracks drawdown, win rate, trades, cumulative return.

### Features (125 total)

| Category | Count | Examples |
|----------|-------|---------|
| **Technical (base)** | 44 | Log returns (1–240h), RSI, Bollinger Bands, ATR, volatility ratios, momentum derivatives, Stochastic, spread ratios, SMA ratios, VVR, hour sin/cos |
| **Macro** | 40 | VIX (level, zscore, regime, spike, rising), DXY, S&P500 changes (1/5/10d), US10Y, EUR/USD, USD/JPY, Oil, Gold volatility (5/20d) |
| **Sentiment** | 25 | Fear & Greed Index (value, zscore, changes, MA, extreme flags) |
| **Cross-asset** | 16 | BTC/ETH/DAX/Nasdaq/S&P500 rolling correlation (10/30d), relative strength (5d) |

### V5 Scoring Formula

```python
# Diagnostic and Mode F:
combined_score = accuracy * (1 + max(cum_return, 0) / 100)

# Feature analysis:
combined = acc * (1 + max(alpha, 0) / 100)
```

Directly rewards being right AND making money. Replaced V4's `0.45×Calmar + 0.35×Sharpe + 0.20×Accuracy` which biased toward low-trade configs.

### Strategies

Set per asset by Mode F, stored in `config/trading_config.json`:

| Strategy | BUY Condition | SELL Condition |
|----------|--------------|----------------|
| `both_agree` | 4h AND 8h both say BUY (≥ min_confidence) | Either says SELL |
| `either_agree` | Either 4h or 8h says BUY (≥ min_confidence) | Either says SELL |
| `4h_only` | 4h says BUY | 4h says SELL |
| `8h_only` | 8h says BUY | 8h says SELL |

**BTC Strategy Comparison (V5, 200h replay):**

| Strategy | Return | Win Rate | Trades | Score |
|----------|--------|----------|--------|-------|
| **either_agree** | **+20.9%** | **90%** | **20** | **0.996** |
| 8h_only | +18.5% | 83% | 12 | 0.977 |
| both_agree | +16.4% | 100% | 13 | 0.959 |
| 4h_only | +15.1% | 88% | 16 | 0.948 |

### Revolut X Auto-Trader

**Authentication:** Ed25519 asymmetric signature (timestamp + method + path + body → base64 signature)

**Hourly loop:**
1. **Startup:** Sync positions with exchange → Telegram notification → immediate scan
2. **Every hour:** Download data → generate 4h+8h signals → apply per-asset strategy → execute trades → Telegram
3. **Every 5 min:** Position sync (detects manual trades on exchange)
4. **Every 30 sec:** Poll Telegram for commands

**Telegram commands:** `/stop` `/status` `/pause` `/resume` `/balance` `/sync`

**Position management:**
- State machine: `cash ↔ invested` per asset
- MIN_TRADE_USD = $300, MIN_POSITION_USD = $5 (dust threshold)
- Manual trade detection via exchange balance sync
- PnL tracking with entry price, entry time, USD invested

**Interactive configuration menu:**
1. Start loop (hourly) — main mode
2. Run once
3. Dry run (once)
4. Configure assets (strategy, max USD, auto-trade, enabled)
5. View trade history
6. Check balance
7. Setup Telegram

---

## Index CFD System (Interactive Brokers)

Completely independent from the crypto system. Uses Broly 1.2 ML model.

### Assets

| Asset | File | IB Symbol | Trading Hours (UTC) |
|-------|------|-----------|-------------------|
| DAX | `ib_auto_trader.py` | IBDE40 | Mon–Fri 07:00–16:00 |
| S&P 500 | `ib_auto_trader_test.py` | IBUS500 | Sun 23:00 – Fri 22:00 |

### Architecture

| Class | Purpose |
|-------|---------|
| `IBConnection` | IB API wrapper (connect, positions, orders, market data) |
| `RiskManager` | Pre-trade validation (market hours, daily loss limit, max positions, cooldown) |
| `TradeExecutor` | Signal → order execution (open/close positions, stop-loss management) |

### Risk Controls

- **Daily loss limit:** 2,000 EUR — stop trading if breached
- **Max positions:** 1 per asset
- **Stop-loss:** 2% below entry (automatic IB stop order)
- **Cooldown:** 2h lockout after stop-loss trigger
- **Max margin budget:** 10,000 EUR
- **CFD margin:** 5%

### Connection

- **Host:** 127.0.0.1 (localhost)
- **Port:** 4002 (paper) / 4001 (live)
- **Client IDs:** 10 (DAX), 20 (S&P 500)
- **Library:** `ib_insync`

### Data

Market data fetched directly from IB (`reqHistoricalData`), no yfinance during trading. 44 base technical features (same set as crypto technical features). Stored in `data/indices/{asset}_hourly_data.csv`.

---

## All Files Reference

### Production Files

| File | Lines | Purpose |
|------|-------|---------|
| `crypto_trading_system.py` | 3,519 | V5 Production — Modes B/D/E/F, all ML logic |
| `crypto_revolut_trader.py` | 1,109 | Multi-asset Revolut X auto-trader |
| `crypto_live_trader.py` | 506 | Signal generation library (imported by trader) |
| `hardware_config.py` | 42 | Machine-specific model configs, n_jobs, GPU |
| `ib_auto_trader.py` | 1,444 | DAX CFD auto-trader (IB) |
| `ib_auto_trader_test.py` | 1,419 | S&P 500 CFD overnight trader (IB) |
| `broly.py` | ~53KB | Enhancement layer: regime detection, graduated signals |

### Utility Scripts

| File | Lines | Purpose |
|------|-------|---------|
| `buy_btc.py` | 94 | Manual BTC purchase via Revolut X ($100 market order) |
| `check_balance.py` | 33 | Query Revolut X account balances |
| `check_trades.py` | 86 | Inspect Revolut X trade history, fills, fees |
| `debug_price.py` | 58 | Test 4 price fetch methods from Revolut X API |
| `revolut_x_test.py` | 114 | Comprehensive API endpoint connectivity test |
| `ib_test_connection.py` | 160 | IB Gateway connection diagnostic |
| `download_macro_data.py` | 264 | Download macro/sentiment/cross-asset data (VIX, F&G, etc.) |
| `detect_hardware.py` | 276 | Auto-detect CPU/GPU/RAM → generate hardware_config.py |

### Analysis & Testing Scripts

| File | Lines | Purpose |
|------|-------|---------|
| `crypto_correlation_analysis.py` | 449 | BTC vs ETH signal correlation & diversification analysis |
| `crypto_strategy_test.py` | 438 | Backtest signal strategies & hold durations |
| `crypto_horizon_test.py` | 233 | Test alternative horizons (2h, 3h, 6h, 8h, 12h) |
| `crypto_trading_threshold_system.py` | 150+ | Backtest confidence thresholds (0–90%) |
| `mock_strategy_optimizer.py` | 397 | Fast strategy + confidence combo optimizer |
| `mock_crypto_trading_system.py` | ~2,000 | Phase 1 validation on synthetic data (holdout, CI, permutation, Calmar/Sharpe) |
| `mock_crypto_trading_system_validation.py` | ~1,000 | Phase 2 validation (edge cases, convergence, integration) |
| `mode_d_improvements.py` | 583 | Documentation of 10 alpha-scoring patches |
| `apply_mode_d_improvements.py` | 573 | Auto-patcher to apply patches to production |

### Version Archives

| File | Version | Status |
|------|---------|--------|
| `crypto_trading_system.py` | V5 | Production |
| `crypto_trading_system_v5.py` | V5 | Versioned backup |
| `crypto_trading_system_v5.1.py` | V5.1 | Experimental — alpha scoring patches |
| `crypto_trading_system_v5.2.py` | V5.2 | Experimental — extended enhancements |
| `crypto_trading_system_v4.py` | V4 | Reference — Calmar/Sharpe scoring |
| `crypto_trading_system_v3_old.py` | V3 | Archive — original production |

### Setup & Migration

| File | Lines | Purpose |
|------|-------|---------|
| `setup_algo_trading.ps1` | 228 | Fresh install PowerShell script (Python, venv, GPU, deps) |
| `migrate_folders.py` | 112 | One-time folder restructure (data/, charts/, models/, config/) |

---

## Folder Structure

```
engine/
│
├── ===== PYTHON FILES (root) =====
├── crypto_trading_system.py           # V5 PRODUCTION — Modes B/D/E/F
├── crypto_revolut_trader.py           # Multi-asset Revolut X auto-trader
├── crypto_live_trader.py              # Signal generation library (NOT run directly)
├── hardware_config.py                 # Machine-specific config
├── ib_auto_trader.py                  # DAX CFD trader (Interactive Brokers)
├── ib_auto_trader_test.py             # S&P 500 CFD overnight trader
├── broly.py                           # Enhancement layer (regime detection)
├── buy_btc.py                         # Manual BTC purchase
├── check_balance.py                   # Revolut X balance query
├── check_trades.py                    # Trade history inspection
├── debug_price.py                     # API price fetch diagnostic
├── revolut_x_test.py                  # API connectivity test
├── ib_test_connection.py              # IB Gateway diagnostic
├── download_macro_data.py             # Macro data downloader
├── detect_hardware.py                 # Hardware detection → config generation
├── crypto_correlation_analysis.py     # BTC/ETH correlation analysis
├── crypto_strategy_test.py            # Strategy backtester
├── crypto_horizon_test.py             # Alternative horizon tester
├── crypto_trading_threshold_system.py # Confidence threshold backtester
├── mock_strategy_optimizer.py         # Fast strategy optimizer
├── mock_crypto_trading_system.py      # Phase 1 mock validation
├── mock_crypto_trading_system_validation.py  # Phase 2 mock validation
├── mode_d_improvements.py             # Alpha-scoring patch documentation
├── apply_mode_d_improvements.py       # Auto-patcher
├── crypto_trading_system_v5.py        # V5 backup
├── crypto_trading_system_v5.1.py      # V5.1 experimental
├── crypto_trading_system_v5.2.py      # V5.2 experimental
├── crypto_trading_system_v4.py        # V4 reference
├── crypto_trading_system_v3_old.py    # V3 archive
├── crypto_auto_trader.py              # Legacy BTC-only auto-trader (493 lines)
├── migrate_folders.py                 # One-time folder migration
├── setup_algo_trading.ps1             # Fresh install script
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── btc_hourly_data.csv            # BTC OHLCV (Binance via ccxt)
│   ├── eth_hourly_data.csv            # ETH OHLCV
│   ├── xrp_hourly_data.csv            # XRP OHLCV
│   ├── doge_hourly_data.csv           # DOGE OHLCV
│   ├── hourly_best_models.csv         # IB system best models
│   ├── setup_config.json              # IB system setup config
│   ├── ib_trader_state.json           # DAX position state
│   ├── ib_trader_state_test.json      # S&P position state
│   ├── ib_trade_log.csv               # IB trade log (all assets)
│   ├── indices/
│   │   ├── dax_hourly_data.csv        # DAX OHLCV (IB)
│   │   ├── sp500_hourly_data.csv      # S&P 500 OHLCV (IB)
│   │   ├── smi_hourly_data.csv        # SMI OHLCV
│   │   └── cac40_hourly_data.csv      # CAC 40 OHLCV
│   └── macro_data/
│       ├── macro_daily.csv            # VIX, DXY, S&P500, Nasdaq, Gold, US10Y, etc.
│       ├── fear_greed.csv             # Crypto Fear & Greed Index
│       ├── cross_asset.csv            # BTC, ETH, Nasdaq, S&P, DAX daily prices
│       ├── macro_hourly.csv           # Daily macro forward-filled to hourly
│       └── download_macro_data.py     # Downloader script
│
├── models/
│   ├── crypto_hourly_best_models.csv  # CENTRAL CONFIG: best model per (asset, horizon)
│   ├── crypto_feature_analysis_*.csv  # Feature scores per asset
│   ├── crypto_feature_set_comparison.csv
│   └── crypto_hourly_chart_data.json  # Signal export for charting
│
├── charts/
│   ├── {ASSET}_backtest.png           # Matplotlib 3-panel backtest chart
│   ├── {ASSET}_strategy_1week.html    # Plotly 4-panel interactive (1 week)
│   ├── {ASSET}_strategy_1month.html   # Plotly 4-panel interactive (1 month)
│   └── {ASSET}_signal_table.html      # Sortable signal table
│
├── config/
│   ├── trading_config.json            # Per-asset strategy + max USD + min_confidence
│   ├── telegram_config.json           # Bot token + chat_id
│   ├── revolut_x_config.json          # Revolut X API key
│   ├── private.pem                    # Ed25519 signing key
│   ├── position_BTC.json              # BTC position state (cash/invested)
│   ├── position_ETH.json              # ETH position state
│   └── revolut_position.json          # Legacy position file
│
├── output/
│   └── dashboards/
│       ├── hourly_dashboard.html      # Crypto dashboard
│       ├── ib_live_data.json          # DAX live dashboard data
│       └── ib_live_data_test.json     # S&P live dashboard data
│
├── CLAUDE.md                          # Claude Code instructions
└── README.md                          # This file
```

---

## Hardware Setup

### Desktop (Primary — ~2x faster than laptop)
```
CPU:     Intel i7-14700KF (20P/28L cores, 26 workers for parallel jobs)
GPU:     NVIDIA RTX 4080 16GB, CUDA 13.1
RAM:     32 GB
Path:    C:\algo_trading\engine
Venv:    C:\algo_trading\venv
```

### Laptop
```
CPU:     16 logical cores
GPU:     NVIDIA RTX 3070 Ti Laptop
Path:    C:\Users\Alex\algo_trading\engine
Venv:    C:\Users\Alex\algo_trading\venv
```

### Shared Config
```
OS:       Windows 11, Python 3.13+ venv (NOT conda)
LGBM:     GPU-enabled (device='gpu'), configured per machine in hardware_config.py
Broker:   Revolut X (0.09% taker fee) for crypto, Interactive Brokers for CFDs
Timezone: Europe/Zurich (CET/CEST)
```

---

## Installation

```powershell
# Run the setup script (creates venv, installs deps, detects GPU)
powershell -ExecutionPolicy Bypass -File setup_algo_trading.ps1

# Or manually:
python -m venv C:\algo_trading\venv
C:\algo_trading\venv\Scripts\activate.bat
pip install -r requirements.txt
python detect_hardware.py    # Generates hardware_config.py
```

### Dependencies

```
pandas>=2.0          # DataFrames, time series
numpy>=1.24          # Numerical computing
scikit-learn>=1.3    # RF, GB, LR models
lightgbm>=4.0        # GPU-accelerated gradient boosting
joblib               # Parallel processing
matplotlib           # Backtest chart PNGs
ccxt>=4.0            # Binance API (crypto data)
yfinance             # Yahoo Finance (macro + indices)
pynacl               # Ed25519 signing (Revolut X auth)
cryptography         # PEM key loading
ib_insync            # Interactive Brokers API (optional, for IB trader)
```

---

## Current Best Models

### Crypto (V5)

| Asset | Horizon | Models | Window | Accuracy | Return | Score | Features | Status |
|-------|---------|--------|--------|----------|--------|-------|----------|--------|
| BTC | 4h | RF+GB+LR | 100h | 80.2% | +125.0% | 1.804 | 125 (all) | V5 2y |
| BTC | 8h | RF+GB | 150h | 84.7% | +319.4% | 3.550 | 15 (optimal) | V5 2y |
| ETH | 4h | RF+LR | 100h | 78.3% | — | — | custom | V4 1y |
| ETH | 8h | RF+LR | 100h | 75.0% | — | — | custom | V4 1y |
| XRP | 4h | GB | 100h | 69.2% | — | — | custom | V4 1y |
| XRP | 8h | RF+LR | 100h | 80.8% | — | — | custom | V4 1y |

**BTC 8h Optimal Features (15):**
`logret_120h, xa_dax_relstr5d, price_to_sma100h, logret_240h, xa_sp500_relstr5d, vol_ratio_12_48, hour_cos, volatility_48h, atr_pct_14h, xa_nasdaq_relstr5d, sma20_to_sma50h, hour_sin, logret_72h, spread_24h_4h, m_nasdaq_chg1d`

### Index CFDs (Broly 1.2)

| Asset | Window | Model | Accuracy |
|-------|--------|-------|----------|
| DAX | 100h | RF | 82.08% |

### Trading Config (config/trading_config.json)

```json
{
  "BTC": {
    "strategy": "either_agree",
    "min_confidence": 75,
    "symbol": "BTC-USD",
    "max_position_usd": 10000
  },
  "ETH": {
    "strategy": "either",
    "symbol": "ETH-USD",
    "max_position_usd": 1000
  }
}
```

---

## Key Constants

```python
# Crypto system
TRADING_FEE = 0.0009                # 0.09% per trade (Revolut X taker fee)
MIN_CONFIDENCE = 75                 # Global fallback — overridden per asset by Mode F
AVAILABLE_HORIZONS = [4, 8]
REPLAY_HOURS = 200                  # Mode B signal replay window
DIAG_STEP = 72                      # Walk-forward step size
DIAG_WINDOWS = [48, 72, 100, 150, 200]  # 5 windows × 15 combos = 75 configs

# IB system
MAX_MARGIN_BUDGET = 10000           # EUR
DAILY_LOSS_LIMIT = 2000             # EUR
STOP_LOSS_PCT = 0.02                # 2%
COOLDOWN_HOURS = 2                  # After stop-loss trigger
SIGNAL_MIN_CONFIDENCE = 55          # %

# Directories
DATA_DIR = 'data'
CHARTS_DIR = 'charts'
MODELS_DIR = 'models'
CONFIG_DIR = 'config'
MACRO_DIR = 'data/macro_data'
```

---

## Version History

| Version | File | Scoring | Key Changes |
|---------|------|---------|-------------|
| **V5** | `crypto_trading_system.py` | `acc × (1 + ret/100)` | Production. Mode F, confidence sweep, per-step timers, lightweight diagnostics |
| **V5.2** | `crypto_trading_system_v5.2.py` | Experimental | Extended enhancements |
| **V5.1** | `crypto_trading_system_v5.1.py` | Experimental | Alpha scoring patches integrated |
| **V4** | `crypto_trading_system_v4.py` | `0.45×Calmar + 0.35×Sharpe + 0.20×Acc` | Bootstrap CI, Calmar/Sharpe, --permtest. Superseded — biased toward low-trade configs |
| **V3** | `crypto_trading_system_v3_old.py` | Heuristic | Original production. Archived |

### What V5 Adds Over V4

1. **New scoring formula** — `accuracy × (1 + max(return, 0) / 100)` replaces Calmar/Sharpe. Directly rewards being right AND making money.
2. **Mode F** — Backtests all 4 strategies + confidence threshold sweep (60–90%). Auto-writes best config to `trading_config.json`.
3. **Per-step timers** — Each Mode D pipeline stage prints elapsed time.
4. **Lightweight diagnostics** — `get_diagnostic_models()` uses n_estimators=100 / RF n_jobs=1. Diagnostics run in ~16 min instead of 40–60 min.
5. **DIAG_WINDOWS reduced** — [48, 72, 100, 150, 200] (was 7 windows). 75 configs total (was 105).

---

## Development Timeline

| Date | Commit | Milestone |
|------|--------|-----------|
| **2026-02-21** | `513b531` | Initial commit — basic structure, .gitignore |
| **2026-02-21** | `034a70a` | V2 strategy + auto dashboard export |
| **2026-02-21** | `ef80639` | V3 comparison strategy (V1 vs V2 vs Buy&Hold) |
| **2026-02-21** | `7451b89` | Fix IB port, delayed market data |
| **2026-02-22** | `bf89d55` | Crypto diagnostic pipeline, leveraged backtest |
| **2026-02-22** | `27509d3` | **Broly 1.2** — modular pipeline + IB auto-trader with V2 features |
| **2026-03-02** | — | Added macro features (VIX, DXY, S&P500). Created Modes A/B/C |
| **2026-03-03** | — | Mode D (full feature analysis), Telegram live trader |
| **2026-03-04** | `30d4688` | **v2.0** — Mode E, dual horizon (4h+8h), threshold system, derivative features |
| **2026-03-07** | — | 8h horizon BTC 80.3%. "Both Agree" strategy. Revolut X trader |
| **2026-03-08** | `bdf1d94` | Remove obsolete Modes A/C |
| **2026-03-08** | `6288c4f` | Apply Mode D alpha improvements |
| **2026-03-08** | `e64760f` | Mock tests for 4 improvements (holdout, bootstrap, permutation, Calmar/Sharpe) |
| **2026-03-08** | `ca40498` | **V4** — bootstrap CI, Calmar/Sharpe scoring, --permtest flag |
| **2026-03-08** | `2055302` | Bug fixes (chart_data key, overall alpha) |
| **2026-03-09** | `e974245` | Fix trader infinite loop when ETH max_position=0 |
| **2026-03-09** | `755e646` | Fix SSL certificate error on Windows (Telegram + API) |
| **2026-03-09** | `5dfba30` | V4: live progress tracking for diagnostic/permutation/ablation |
| **2026-03-10** | `23bc9f2` | **V5 Production** — acc×(1+ret/100) scoring, Mode F, BTC 2y: 80.2%/84.7% |
| **2026-03-10** | `289c5d4` | Update trading system |

### Session History (Detailed)

1. **Mar 2**: Added V2 macro features. Created crypto_trading_system.py with Modes A/B/C. BTC 4h: Set A 76.5% vs Set B 75.7%.
2. **Mar 3**: Mode D (full feature analysis pipeline), Telegram live trader, README.
3. **Mar 4**: Confidence threshold backtester, derivative features, live trader timing fix. Dual horizon (1h+4h → 4h+8h). Fixed critical run_loop bug. Mode E (iterative refinement). VVR feature, profit-weighted scoring.
4. **Mar 7**: 8h horizon — BTC 8h 80.3%. "Both Agree" strategy: +97.2% alpha, 90% win rate. Revolut X trader built.
5. **Mar 7-8**: ETH/XRP analysis. SOL removed (poor results). Per-asset strategies. Revolut X API fixes. 4-panel Plotly charts, signal table HTML. Weighted strategy tested & rejected.
6. **Mar 8**: V4 created — bootstrap CI, Calmar/Sharpe scoring, --permtest. Mock validation (Phase 1 + Phase 2) on synthetic data. hardware_config lightened. DIAG_WINDOWS reduced. BTC 4h V4: GB+LR w=200h 63% +48.7%.
7. **Mar 9**: V5 created — scoring replaced with acc×(1+ret/100). Mode F + confidence sweep. Per-step timers. SSL fix. Trader bugfixes.
8. **Mar 10**: V5 Mode D BTC 4,8h 2y complete. **4h: RF+GB+LR w=100h 80.2% +125%. 8h: RF+GB w=150h 84.7% +319%.** Mode F: either_agree + 75%. V5 promoted to production. V3 archived.

---

## File Dependencies

```
crypto_trading_system.py  (V5 — production)
  ├── imports: hardware_config.py
  ├── reads: data/macro_data/*.csv
  ├── reads/writes: data/{asset}_hourly_data.csv
  ├── reads/writes: models/crypto_hourly_best_models.csv
  ├── reads/writes: config/trading_config.json  (Mode F)
  └── writes: charts/*.html, charts/*.png, models/*.json

crypto_live_trader.py  (signal generation)
  ├── imports: crypto_trading_system.py  (ASSETS, features, models, download/load/build)
  ├── reads: models/crypto_hourly_best_models.csv
  ├── reads: config/telegram_config.json
  └── sends: Telegram API

crypto_revolut_trader.py  (auto-trader)
  ├── imports: crypto_live_trader.py  (signals, Telegram, data)
  ├── reads: config/trading_config.json  (strategy + min_confidence per asset)
  ├── reads: config/revolut_x_config.json + config/private.pem
  ├── reads/writes: config/position_*.json
  └── sends: Revolut X API + Telegram

ib_auto_trader.py  (DAX CFD)
  ├── connects: Interactive Brokers (localhost:4002)
  ├── reads: data/setup_config.json + data/hourly_best_models.csv
  ├── reads/writes: data/indices/dax_hourly_data.csv
  ├── reads/writes: data/ib_trader_state.json
  ├── writes: data/ib_trade_log.csv
  └── writes: output/dashboards/ib_live_data.json
```

---

## Pending Actions

- [ ] Run V5 Mode D ETH 4,8h 2y on laptop
- [ ] Run V5 Mode D XRP 4,8h 2y on laptop
- [ ] Run Mode F for ETH and XRP after V5 runs
- [ ] Update best models table once ETH/XRP 2y runs complete
- [x] V5 scoring (acc×(1+return/100))
- [x] Mode F (strategy comparison + confidence sweep)
- [x] Per-step timers in Mode D
- [x] hardware_config diagnostic models lightened
- [x] DIAG_WINDOWS reduced to [48,72,100,150,200]
- [x] BTC V5 2y run — 4h: 80.2% +125%, 8h: 84.7% +319%
- [x] V5 promoted to production, V3 archived
- [x] IB auto-trader for DAX + S&P 500 CFDs
- [x] Broly 1.2 enhancement layer
- [x] Revolut X multi-asset auto-trader with Telegram control
- [x] Mock validation framework (Phase 1 + Phase 2)
- [x] Crypto correlation analysis (BTC vs ETH)

---

*Last updated: March 11, 2026 — V5 production with IB integration documented. ~12,000+ lines of Python across 30 files.*
