# Algo Trading Engine

Automated ML-powered trading system for **crypto** (BTC, ETH, XRP) and **index CFDs** (DAX, S&P 500). Two independent pipelines:

1. **Crypto Pipeline** — Hourly BUY/SELL/HOLD signals via dual-horizon (4h + 8h) ensemble ML with walk-forward validation. Executes via exchange API.
2. **Sub-Hourly Crypto (V15/V30)** — 15-min and 30-min candle variants for higher trade frequency. Same ML pipeline, separate data/models/configs.
3. **Index CFD Pipeline** — Hourly signals via Broly 1.2 ML model. Executes via broker API.

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
  - [Auto-Trader](#crypto-auto-trader)
- [Sub-Hourly Crypto (V15/V30)](#sub-hourly-crypto-systems-v15--v30)
- [Index CFD System](#index-cfd-system)
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
# Run: venv\Scripts\activate.bat

# === CRYPTO SIGNALS ===
python crypto_trading_system.py 5              # Quick BTC (both horizons)
python crypto_trading_system.py 6              # Quick ETH
python crypto_trading_system.py 7              # Quick XRP
python crypto_trading_system.py B BTC 4,8h     # Mode B — signals from saved models
python crypto_trading_system.py D BTC 4,8h 1y  # Mode D — full pipeline (~90 min/horizon)
python crypto_trading_system.py F BTC          # Mode F — strategy comparison

# === CRYPTO SUB-HOURLY (V15 / V30) ===
python crypto_trading_system_v15.py D BTC 4,8h 1y   # 15-min candles (h4=60', h8=120')
python crypto_trading_system_v30.py D BTC 4,8h 1y   # 30-min candles (h4=2h, h8=4h)
python crypto_trading_system_v15.py F BTC 4,8h       # Strategy optimizer (15-min)
python crypto_trading_system_v30.py F BTC 4,8h       # Strategy optimizer (30-min)

# === CRYPTO AUTO-TRADER ===
python crypto_revolut_trader.py --loop         # Live trading loop
python crypto_revolut_trader.py --dry-run --loop  # Signals only, no trades
python crypto_revolut_trader.py --status       # Show positions
python crypto_revolut_trader.py --balance      # Exchange balance

# === INDEX CFD TRADER ===
python cfd/ib_auto_trader.py --loop            # DAX continuous trading
python cfd/ib_auto_trader.py --status          # Show positions
python cfd/ib_auto_trader_test.py --loop       # S&P 500 overnight trading

# === UTILITIES ===
python tools/check_balance.py                  # Exchange balance
python tools/check_trades.py                   # Trade history
python tools/detect_hardware.py                # Detect CPU/GPU → hardware_config.py
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CRYPTO PIPELINE                              │
│                                                                     │
│  crypto_trading_system.py  (V5.4 Production — 3,519 lines)          │
│    ├── Modes B/D/E/F + shortcuts 5/6/7                              │
│    ├── 125 features (44 technical + 81 macro/sentiment/cross-asset) │
│    ├── 4 ML models × 15 combinations × 5 windows                   │
│    ├── Walk-forward validation (no future leakage)                  │
│    └── hardware_config.py (machine-specific GPU/CPU settings)       │
│                                                                     │
│  crypto_trading_system_v6.py  (V6 Experimental)                     │
│    ├── 12 ENHANCEMENTS flags (toggle on/off, env var override)      │
│    ├── Wavelet denoising, fractional diff, GMM regime detection     │
│    ├── XGBoost, sample weighting, entropy filter, tri-state labels  │
│    ├── Stacking ensemble, dynamic feature select, meta-labeling     │
│    ├── Adversarial validation, Kelly sizing                         │
│    └── testing_literature_v2.py (A/B test harness)                  │
│                                                                     │
│  crypto_trading_system_v15.py  (V15 — 15-min candles)               │
│    ├── Horizons 1-8 = 15'–120' (more trades, shorter horizons)     │
│    ├── Separate data/models/config (*_15m_*)                        │
│    └── Max 1 year rolling data, ~35K rows/year                     │
│                                                                     │
│  crypto_trading_system_v30.py  (V30 — 30-min candles)               │
│    ├── Horizons 1-8 = 30'–240' (2× hourly frequency)              │
│    ├── Separate data/models/config (*_30m_*)                        │
│    └── Max 1 year rolling data, ~17.5K rows/year                   │
│                                                                     │
│  crypto_live_trader.py  (506 lines — signal generation library)     │
│    ├── Imports models/features from crypto_trading_system.py        │
│    ├── Trains on latest window → predicts next candle               │
│    └── Telegram notifications (HTML formatted)                      │
│                                                                     │
│  crypto_revolut_trader.py  (1,109 lines — multi-asset auto-trader) │
│    ├── Signed exchange API (buy/sell/balance)                       │
│    ├── Per-asset strategies from trading_config.json                │
│    ├── Position sync (detects manual trades on exchange)            │
│    ├── Telegram commands (/stop /status /pause /resume /balance)    │
│    └── Hourly loop with 5-min position sync                        │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                     INDEX CFD PIPELINE (cfd/)                        │
│                                                                     │
│  cfd/ib_auto_trader.py  (1,444 lines — DAX CFD trader)             │
│    ├── Broly 1.2 ML model with V2 features                         │
│    ├── Broker API via ib_insync                                     │
│    ├── Risk manager (daily loss limit, cooldown, stop-loss)         │
│    └── Live HTML dashboard export                                   │
│                                                                     │
│  cfd/ib_auto_trader_test.py  (1,419 lines — S&P 500 CFD overnight)│
│    └── Same architecture, different asset + trading hours            │
│                                                                     │
│  cfd/broly.py  (~53KB — enhancement layer)                          │
│    ├── Market regime detection (BULL/BEAR/SIDEWAYS)                 │
│    ├── Graduated 5-tier signals (STRONG_BUY → STRONG_SELL)          │
│    └── Discord + Telegram alerts                                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Key design principle:** The crypto and index CFD systems are **completely independent** — different models, different assets, different brokers, different configs. The only shared concept is the feature engineering approach.

---

## Crypto Trading System (V5.4 Production)

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

**Portfolio simulation:** Entry at `price × (1 + fee)`, exit at `price × (1 - fee)`. Tracks drawdown, win rate, trades, cumulative return.

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

### Crypto Auto-Trader

**Authentication:** Asymmetric signature (timestamp + method + path + body → base64 signature)

**Hourly loop:**
1. **Startup:** Sync positions with exchange → Telegram notification → immediate scan
2. **Every hour:** Download data → generate 4h+8h signals → apply per-asset strategy → execute trades → Telegram
3. **Every 5 min:** Position sync (detects manual trades on exchange)
4. **Every 30 sec:** Poll Telegram for commands

**Telegram commands:** `/stop` `/status` `/pause` `/resume` `/balance` `/sync` `/conf` `/chart BTC`

**Position management:**
- State machine: `cash ↔ invested` per asset
- Configurable min trade size and dust threshold
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

## Sub-Hourly Crypto Systems (V15 / V30)

The hourly system produces good signals but few trades. V15 and V30 use shorter candles for higher trade frequency while keeping the same ML pipeline.

| System | Candle | Horizons (candles 1-8) | Data/year | File |
|--------|--------|----------------------|-----------|------|
| **V15** | 15 min | 15', 30', 45', 60', 75', 90', 105', 120' | ~35K rows | `crypto_trading_system_v15.py` |
| **V30** | 30 min | 30', 60', 90', 120', 150', 180', 210', 240' | ~17.5K rows | `crypto_trading_system_v30.py` |

**Key differences from hourly system:**
- All rolling/shift/diff periods scaled via `_hours_to_rows()` (e.g., RSI 14h = 56 candles on V15)
- Extra features: `logret_1c..4c` (candle-scale returns), `minute_sin/cos` (intra-hour timing)
- Data capped to 1 rolling year (Binance 15m/30m)
- Crypto assets only (BTC, ETH, XRP, DOGE) — no yfinance indices
- Separate output files: `data/*_15m_data.csv`, `models/crypto_15m_best_models.csv`, `config/trading_config_15m.json`, `charts/*_15m_backtest.png`

**Usage:**
```bash
python crypto_trading_system_v15.py D BTC 4,8h 1y   # horizon 4 = 60min, 8 = 120min
python crypto_trading_system_v30.py D BTC 4,8h 1y   # horizon 4 = 2h, 8 = 4h
python crypto_trading_system_v15.py F BTC 4,8h       # strategy optimizer
python crypto_trading_system_v15.py G BTC             # horizon pair test
```

---

## Index CFD System

Completely independent from the crypto system. Uses Broly 1.2 ML model.

### Assets

| Asset | File | Trading Hours (UTC) |
|-------|------|-------------------|
| DAX | `ib_auto_trader.py` | Mon–Fri 07:00–16:00 |
| S&P 500 | `ib_auto_trader_test.py` | Sun 23:00 – Fri 22:00 |

### Architecture

| Class | Purpose |
|-------|---------|
| `IBConnection` | Broker API wrapper (connect, positions, orders, market data) |
| `RiskManager` | Pre-trade validation (market hours, daily loss limit, max positions, cooldown) |
| `TradeExecutor` | Signal → order execution (open/close positions, stop-loss management) |

### Risk Controls

- **Daily loss limit:** 2,000 EUR — stop trading if breached
- **Max positions:** 1 per asset
- **Stop-loss:** 2% below entry (automatic stop order)
- **Cooldown:** 2h lockout after stop-loss trigger
- **Max margin budget:** configurable
- **CFD margin:** 5%

### Data

Market data fetched directly from broker API, no yfinance during trading. 44 base technical features (same set as crypto technical features). Stored in `data/indices/{asset}_hourly_data.csv`.

---

## All Files Reference

### Production Files

| File | Lines | Purpose |
|------|-------|---------|
| `crypto_trading_system.py` | 3,519 | V5.4 Production — Modes B/D/E/F, all ML logic |
| `crypto_revolut_trader.py` | 1,109 | Multi-asset crypto auto-trader |
| `crypto_live_trader.py` | 506 | Signal generation library (imported by trader) |
| `crypto_trading_system_v15.py` | ~3,600 | V15 — 15-min candle system (Modes B/D/E/F/G) |
| `crypto_trading_system_v30.py` | ~3,600 | V30 — 30-min candle system (Modes B/D/E/F/G) |
| `hardware_config.py` | 42 | Auto-detects Desktop/Laptop, model configs, n_jobs, GPU |
| `cfd/ib_auto_trader.py` | 1,444 | DAX CFD auto-trader |
| `cfd/ib_auto_trader_test.py` | 1,419 | S&P 500 CFD overnight trader |
| `cfd/broly.py` | ~53KB | Enhancement layer: regime detection, graduated signals |

### Utility Scripts

| File | Lines | Purpose |
|------|-------|---------|
| `tools/buy_btc.py` | 94 | Manual BTC purchase ($100 market order) |
| `tools/check_balance.py` | 33 | Query exchange account balances |
| `tools/check_trades.py` | 86 | Inspect trade history, fills, fees |
| `tools/debug_price.py` | 58 | Test price fetch methods from exchange API |
| `tools/revolut_x_test.py` | 114 | Comprehensive API endpoint connectivity test |
| `tools/ib_test_connection.py` | 160 | Broker connection diagnostic |
| `tools/detect_hardware.py` | 276 | Auto-detect CPU/GPU/RAM → generate hardware_config.py |
| `download_macro_data.py` | ~350 | Download macro/sentiment/cross-asset + on-chain + derivatives data |
| `archive/testing_literature.py` | ~320 | A/B test harness for V5.5 enhancements — COMPLETE, archived. Results in `archive/testing_literature.csv` |
| `testing_literature_v2.py` | ~335 | A/B test harness for V6 (12 literature enhancements) — Mode D BTC 4,8h 1y |
| `testing_feature_stability.py` | ~300 | Feature stability test — cross-references KEEP/DROP across BTC+ETH × 4h+8h |

### Archived Scripts (in `archive/`)

| File | Purpose |
|------|---------|
| `crypto_correlation_analysis.py` | BTC vs ETH signal correlation & diversification analysis |
| `crypto_strategy_test.py` | Backtest signal strategies & hold durations |
| `crypto_horizon_test.py` | Test alternative horizons (2h, 3h, 6h, 8h, 12h) |
| `crypto_trading_threshold_system.py` | Backtest confidence thresholds (0–90%) |
| `mock_strategy_optimizer.py` | Fast strategy + confidence combo optimizer |
| `mock_crypto_trading_system.py` | Phase 1 validation on synthetic data |
| `mock_crypto_trading_system_validation.py` | Phase 2 validation (edge cases, convergence, integration) |
| `mode_d_improvements.py` | Documentation of 10 alpha-scoring patches |
| `apply_mode_d_improvements.py` | Auto-patcher to apply patches to production |
| `setup_algo_trading.ps1` | Fresh install PowerShell script (Python, venv, GPU, deps) |
| `migrate_folders.py` | One-time folder restructure (data/, charts/, models/, config/) |
| `testing_literature.py` | V5.5 A/B test harness — COMPLETE. Results: only slippage_model won |
| `testing_literature.csv` | V5.5 A/B test results (8 tests × BTC 4h+8h 1y) |
| `testing_literature_logs/` | V5.5 test run logs |
| `crypto_trading_system_v5.6.py` | V5.6 — 20 literature features tested. Only ADX+GK useful → promoted to V5 |

### Version Archives

| File | Version | Status |
|------|---------|--------|
| `crypto_trading_system.py` | V5.4+ | Production — phase-specific BLAS, loky pool reset, orphan cleanup, slippage model, DF mode |
| `crypto_trading_system_v6.py` | V6 | Experimental — 12 literature enhancements behind ENHANCEMENTS flags (env var override) |
| `archive/crypto_trading_system_v5.5.py` | V5.5 | Archived — 7 literature enhancements tested, only slippage_model promoted |
| `crypto_trading_system_v15.py` | V15 | Sub-hourly — 15-min candles, horizons 15'–120', 1y max |
| `crypto_trading_system_v30.py` | V30 | Sub-hourly — 30-min candles, horizons 30'–240', 1y max |
| `archive/crypto_trading_system_v5.4.py` | V5.4 | Archived — same as production (standalone copy) |
| `archive/crypto_trading_system_v5.3.py` | V5.3 | Archived — thread/worker fixes |
| `archive/crypto_trading_system_v5.2.py` | V5.2 | Archived — all 8 horizons + Mode G (no BLAS fixes) |
| `archive/crypto_trading_system_v5.1.py` | V5.1 | Archived — alpha scoring patches |
| `archive/crypto_trading_system_v5.py` | V5 | Archived — versioned backup |
| `archive/crypto_trading_system_v4.py` | V4 | Archived — Calmar/Sharpe scoring |
| `archive/crypto_trading_system_v3_old.py` | V3 | Archived — original production |

---

## Folder Structure

```
engine/
│
├── ===== PYTHON FILES (root) =====
├── crypto_trading_system.py           # V5.4 PRODUCTION — Modes B/D/E/F
├── # (V5.5 archived — slippage_model promoted to production)
├── crypto_trading_system_v15.py       # V15 — 15-min candles (15'–120' horizons)
├── crypto_trading_system_v30.py       # V30 — 30-min candles (30'–240' horizons)
├── crypto_revolut_trader.py           # Multi-asset crypto auto-trader
├── crypto_live_trader.py              # Signal generation library (NOT run directly)
├── hardware_config.py                 # Auto-detects Desktop/Laptop config
├── download_macro_data.py             # Macro data downloader (macro + on-chain + derivatives)
├── crypto_trading_system_v6.py        # V6 — 12 literature enhancements (experimental)
├── # testing_literature.py → archived (V5.5 A/B tests COMPLETE)
├── testing_literature_v2.py           # A/B test harness for V6 (12 enhancements)
├── testing_feature_stability.py      # Feature stability test (BTC+ETH × 4h+8h)
├── requirements.txt                   # Python dependencies
│
├── cfd/                               # Index CFD trading (separate pipeline)
│   ├── ib_auto_trader.py              # DAX CFD trader (Broly 1.2)
│   ├── ib_auto_trader_test.py         # S&P 500 CFD overnight trader
│   └── broly.py                       # Enhancement layer (regime detection)
│
├── tools/                             # Utilities & diagnostics
│   ├── buy_btc.py                     # Manual BTC purchase
│   ├── check_balance.py               # Exchange balance query
│   ├── check_trades.py                # Trade history inspection
│   ├── debug_price.py                 # API price fetch diagnostic
│   ├── revolut_x_test.py              # API connectivity test
│   ├── ib_test_connection.py          # IB Gateway diagnostic
│   └── detect_hardware.py             # Hardware detection → config generation
│
├── archive/                           # Archived / superseded files
│   ├── crypto_trading_system_v5.4.py  # V5.4 standalone copy
│   ├── crypto_trading_system_v5.5.py  # V5.5 literature enhancements (tested, slippage promoted)
│   ├── crypto_trading_system_v5.3.py  # V5.3 thread/worker fixes
│   ├── crypto_trading_system_v5.2.py  # V5.2 all 8 horizons + Mode G
│   ├── crypto_trading_system_v5.1.py  # V5.1 alpha scoring patches
│   ├── crypto_trading_system_v5.py    # V5 backup
│   ├── crypto_trading_system_v4.py    # V4 reference (Calmar/Sharpe)
│   ├── crypto_trading_system_v3_old.py# V3 original production
│   ├── crypto_auto_trader.py          # Legacy BTC-only auto-trader
│   ├── crypto_correlation_analysis.py # BTC/ETH correlation analysis
│   ├── crypto_strategy_test.py        # Strategy backtester
│   ├── crypto_horizon_test.py         # Alternative horizon tester
│   ├── crypto_trading_threshold_system.py
│   ├── mock_strategy_optimizer.py
│   ├── mock_crypto_trading_system.py
│   ├── mock_crypto_trading_system_validation.py
│   ├── mode_d_improvements.py
│   ├── apply_mode_d_improvements.py
│   ├── migrate_folders.py
│   └── setup_algo_trading.ps1
│
├── data/
│   ├── btc_hourly_data.csv            # BTC OHLCV (Binance via ccxt)
│   ├── eth_hourly_data.csv            # ETH OHLCV
│   ├── xrp_hourly_data.csv            # XRP OHLCV
│   ├── doge_hourly_data.csv           # DOGE OHLCV
│   ├── btc_15m_data.csv              # BTC 15-min OHLCV (V15)
│   ├── eth_15m_data.csv              # ETH 15-min OHLCV (V15)
│   ├── btc_30m_data.csv              # BTC 30-min OHLCV (V30)
│   ├── eth_30m_data.csv              # ETH 30-min OHLCV (V30)
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
│       ├── onchain_btc.csv            # On-chain: active addresses, hash rate, MVRV, SOPR, etc.
│       ├── derivatives_btc.csv        # Derivatives: funding rate + open interest (hourly)
│       └── download_macro_data.py     # Downloader script
│
├── models/
│   ├── crypto_hourly_best_models.csv  # CENTRAL CONFIG: best model per (asset, horizon)
│   ├── crypto_15m_best_models.csv    # V15 best models
│   ├── crypto_30m_best_models.csv    # V30 best models
│   ├── crypto_feature_analysis_*.csv  # Feature scores per asset
│   ├── crypto_feature_set_comparison.csv
│   ├── crypto_hourly_chart_data.json  # Signal export for charting
│   ├── crypto_15m_chart_data.json    # V15 chart data
│   └── crypto_30m_chart_data.json    # V30 chart data
│
├── charts/
│   ├── {ASSET}_backtest.png           # Matplotlib 3-panel backtest chart
│   ├── {ASSET}_strategy_1week.html    # Plotly 4-panel interactive (1 week)
│   ├── {ASSET}_strategy_1month.html   # Plotly 4-panel interactive (1 month)
│   └── {ASSET}_signal_table.html      # Sortable signal table
│
├── config/
│   ├── trading_config.json            # Per-asset strategy + max USD + min_confidence
│   ├── trading_config_15m.json       # V15 trading config (written by V15 Mode F)
│   ├── trading_config_30m.json       # V30 trading config (written by V30 Mode F)
│   ├── telegram_config.json           # Bot token + chat_id
│   ├── exchange_config.json            # Exchange API key
│   ├── private.pem                    # Signing key
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

**One shared engine folder** synced via Google Drive — both machines run the same code. Only the venv is local per machine. `hardware_config.py` auto-detects Desktop (26 workers) vs Laptop (14 workers) at import time via `os.cpu_count()`.

| Machine | Engine Path | Venv | CPU | GPU |
|---------|-------------|------|-----|-----|
| Desktop | `G:\engine\` | `C:\algo_trading\venv\` | i7-14700KF (28 cores) | RTX 4080 |
| Laptop | `G:\Autres ordinateurs\My laptop\engine\` | `C:\Users\Alex\algo_trading\venv\` | 16 cores | RTX 3070 Ti |

```
OS:       Windows 11, Python 3.14 venv (NOT conda)
LGBM:     GPU-enabled (device='gpu'), configured per machine in hardware_config.py
```

---

## Installation

**Each machine needs its own venv** — the engine folder is shared via Google Drive but venvs are local.

```powershell
# === DESKTOP ===
python -m venv C:\algo_trading\venv
C:\algo_trading\venv\Scripts\activate.bat
pip install -r G:\engine\requirements.txt
python tools/detect_hardware.py    # Generates hardware_config.py

# === LAPTOP ===
python -m venv C:\Users\Alex\algo_trading\venv
C:\Users\Alex\algo_trading\venv\Scripts\activate.bat
pip install -r "G:\Autres ordinateurs\My laptop\engine\requirements.txt"
python tools/detect_hardware.py
```

**Verify all deps are in the venv** (run after activating venv):
```bash
python -c "import pandas, numpy, sklearn, lightgbm, ccxt, yfinance, pywt, xgboost; print('All OK')"
```

### Dependencies

```
# Core
pandas>=2.0          # DataFrames, time series
numpy>=1.24          # Numerical computing
scikit-learn>=1.3    # RF, GB, LR, GaussianMixture models
lightgbm>=4.0        # GPU-accelerated gradient boosting
joblib               # Parallel processing
matplotlib           # Backtest chart PNGs
ccxt>=4.0            # Binance API (crypto data)
yfinance             # Yahoo Finance (macro + indices)
pynacl               # Asymmetric signing (exchange auth)
cryptography         # PEM key loading
ib_insync            # Broker API (optional, for CFD trader)

# V6 literature enhancements
PyWavelets           # Wavelet denoising (DWT)
xgboost              # XGBoost model (added to model pool)
# fracdiff & hmmlearn: manual implementations in V6 (no C build / Python 3.14 support)
```

---

## Current Best Models

### Crypto (V5)

| Asset | Horizon | Models | Window | Accuracy | Return | Score | Features | Status |
|-------|---------|--------|--------|----------|--------|-------|----------|--------|
| BTC | 4h | RF+GB+LR | 100h | 80.2% | +125.0% | 1.804 | 125 (all) | V5 2y |
| BTC | 8h | RF+GB | 150h | 84.7% | +319.4% | 3.550 | 15 (optimal) | V5 2y |
| ETH | 4h | RF+LGBM | 100h | 68.6% | +505% | 4.154 | custom | V5 2y |
| ETH | 8h | GB | 48h | 79.3% | +616% | 5.681 | custom | V5 2y ⚠️ WARNING |
| XRP | 4h | GB | 100h | 69.2% | — | — | custom | V4 1y — needs V5 run |
| XRP | 8h | RF+LR | 100h | 80.8% | — | — | custom | V4 1y — needs V5 run |

**⚠️ ETH 8h WARNING:** GB model outputs 100% confidence on every signal — overfit on 48h window. Do not increase ETH `max_position_usd` until `CalibratedClassifierCV` fix is applied.

**BTC 8h Optimal Features (15):**
`logret_120h, xa_dax_relstr5d, price_to_sma100h, logret_240h, xa_sp500_relstr5d, vol_ratio_12_48, hour_cos, volatility_48h, atr_pct_14h, xa_nasdaq_relstr5d, sma20_to_sma50h, hour_sin, logret_72h, spread_24h_4h, m_nasdaq_chg1d`

### Index CFDs (Broly 1.2)

| Asset | Window | Model | Accuracy |
|-------|--------|-------|----------|
| DAX | 100h | RF | 82.08% |

### Trading Config (config/trading_config.json)

Per-asset strategy, min_confidence, symbol, and max_position_usd are configured in `config/trading_config.json`. See Mode F for auto-generation.

---

## Key Constants

```python
# Crypto system
TRADING_FEE_BASE = 0.0009           # 0.09% exchange taker fee
SLIPPAGE = 0.0002                   # 0.02% estimated slippage
TRADING_FEE = 0.0011                # total cost per trade (fee + slippage)
MIN_CONFIDENCE = 75                 # Global fallback — overridden per asset by Mode F
AVAILABLE_HORIZONS = [4, 8]
REPLAY_HOURS = 200                  # Mode B signal replay window
DIAG_STEP = 72                      # Walk-forward step size
DIAG_WINDOWS = [48, 72, 100, 150, 200]  # 5 windows × 15 combos = 75 configs

# CFD system
MAX_MARGIN_BUDGET = 10000           # configurable
DAILY_LOSS_LIMIT = 2000             # configurable
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
| **V5.4** | `crypto_trading_system.py` | `acc × (1 + ret/100)` | **Production.** Phase-specific BLAS threading, loky pool reset, orphan cleanup, Mode F |
| **V5.5** | `crypto_trading_system_v5.5.py` | `acc × (1 + ret/100)` | Experimental. 7 literature enhancements behind ENHANCEMENTS flags (on-chain, derivatives, triple-barrier, slippage, extended diag, GB calibration, purged embargo). A/B testable via `testing_literature.py` |
| **V15** | `crypto_trading_system_v15.py` | `acc × (1 + ret/100)` | Sub-hourly. 15-min candles, 8 horizons (15'–120'), 1y max, separate data/models/config |
| **V30** | `crypto_trading_system_v30.py` | `acc × (1 + ret/100)` | Sub-hourly. 30-min candles, 8 horizons (30'–240'), 1y max, separate data/models/config |
| **V5.3** | `archive/crypto_trading_system_v5.3.py` | Archived | LGBM lightened, LOKY_MAX_CPU_COUNT capped, OMP/MKL/OpenBLAS thread limits |
| **V5.2** | `archive/crypto_trading_system_v5.2.py` | Archived | All 8 horizons + Mode G (no BLAS fixes) |
| **V5.1** | `archive/crypto_trading_system_v5.1.py` | Archived | Alpha scoring patches integrated |
| **V4** | `archive/crypto_trading_system_v4.py` | `0.45×Calmar + 0.35×Sharpe + 0.20×Acc` | Bootstrap CI, Calmar/Sharpe, --permtest. Superseded — biased toward low-trade configs |
| **V3** | `archive/crypto_trading_system_v3_old.py` | Heuristic | Original production. Archived |

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
| **2026-03-07** | — | 8h horizon BTC 80.3%. "Both Agree" strategy. Crypto auto-trader |
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
| **2026-03-11** | `7b681c2` | **V5.3** — thread/worker fixes; V5.4 added; archive/ cleanup; ETH V5 2y results |

### Session History (Detailed)

1. **Mar 2**: Added V2 macro features. Created crypto_trading_system.py with Modes A/B/C. BTC 4h: Set A 76.5% vs Set B 75.7%.
2. **Mar 3**: Mode D (full feature analysis pipeline), Telegram live trader, README.
3. **Mar 4**: Confidence threshold backtester, derivative features, live trader timing fix. Dual horizon (1h+4h → 4h+8h). Fixed critical run_loop bug. Mode E (iterative refinement). VVR feature, profit-weighted scoring.
4. **Mar 7**: 8h horizon — BTC 8h 80.3%. "Both Agree" strategy: +97.2% alpha, 90% win rate. Crypto auto-trader built.
5. **Mar 7-8**: ETH/XRP analysis. SOL removed (poor results). Per-asset strategies. Exchange API fixes. 4-panel Plotly charts, signal table HTML. Weighted strategy tested & rejected.
6. **Mar 8**: V4 created — bootstrap CI, Calmar/Sharpe scoring, --permtest. Mock validation (Phase 1 + Phase 2) on synthetic data. hardware_config lightened. DIAG_WINDOWS reduced. BTC 4h V4: GB+LR w=200h 63% +48.7%.
7. **Mar 9**: V5 created — scoring replaced with acc×(1+ret/100). Mode F + confidence sweep. Per-step timers. SSL fix. Trader bugfixes.
8. **Mar 10**: V5 Mode D BTC 4,8h 2y complete. **4h: RF+GB+LR w=100h 80.2% +125%. 8h: RF+GB w=150h 84.7% +319%.** Mode F: either_agree + 75%. V5 promoted to production. V3 archived.

---

## File Dependencies

```
crypto_trading_system.py  (V5.4 — production)
  ├── imports: hardware_config.py
  ├── reads: data/macro_data/*.csv
  ├── reads/writes: data/{asset}_hourly_data.csv
  ├── reads/writes: models/crypto_hourly_best_models.csv
  ├── reads/writes: config/trading_config.json  (Mode F)
  └── writes: charts/*.html, charts/*.png, models/*.json

crypto_trading_system_v15.py  (V15 — 15-min candles)
  ├── imports: hardware_config.py
  ├── reads: data/macro_data/*.csv
  ├── reads/writes: data/{asset}_15m_data.csv
  ├── reads/writes: models/crypto_15m_best_models.csv
  ├── reads/writes: config/trading_config_15m.json  (Mode F)
  └── writes: charts/*_15m_*.html, charts/*_15m_backtest.png

crypto_trading_system_v30.py  (V30 — 30-min candles)
  ├── imports: hardware_config.py
  ├── reads: data/macro_data/*.csv
  ├── reads/writes: data/{asset}_30m_data.csv
  ├── reads/writes: models/crypto_30m_best_models.csv
  ├── reads/writes: config/trading_config_30m.json  (Mode F)
  └── writes: charts/*_30m_*.html, charts/*_30m_backtest.png

crypto_live_trader.py  (signal generation)
  ├── imports: crypto_trading_system.py  (ASSETS, features, models, download/load/build)
  ├── reads: models/crypto_hourly_best_models.csv
  ├── reads: config/telegram_config.json
  └── sends: Telegram API

crypto_revolut_trader.py  (auto-trader)
  ├── imports: crypto_live_trader.py  (signals, Telegram, data)
  ├── reads: config/trading_config.json  (strategy + min_confidence per asset)
  ├── reads: config/exchange_config.json + config/private.pem
  ├── reads/writes: config/position_*.json
  └── sends: Exchange API + Telegram

cfd/ib_auto_trader.py  (DAX CFD)
  ├── imports: hardware_config.py (via sys.path to engine/)
  ├── connects: Broker API (localhost)
  ├── reads: data/setup_config.json + data/hourly_best_models.csv
  ├── reads/writes: data/indices/dax_hourly_data.csv
  ├── reads/writes: data/ib_trader_state.json
  ├── writes: data/ib_trade_log.csv
  └── writes: output/dashboards/ib_live_data.json
```

---

## Enhancement Testing History

### V5.5 Enhancements (testing_literature.py) — COMPLETE, archived

Tested 7 V5.5 enhancements individually via Mode D BTC 4h+8h 1y. Baseline: 4h 80.0% +57.2%, 8h 78.3% +74.0%.

| Enhancement | 4h Acc | 4h Return | 8h Acc | 8h Return | Verdict |
|-------------|--------|-----------|--------|-----------|---------|
| **slippage_model** | 77.5% | +57.1% | 79.2% | +98.4% | **WINNER — promoted to production** |
| extended_diag_step | 67.8% | +73.7% | 85.6% | +65.9% | Best 8h accuracy, fewer trades |
| on_chain_features | 70.0% | +60.6% | 73.3% | +106.1% | High return but accuracy drops |
| gb_calibration | 78.3% | +53.9% | 75.8% | +78.8% | Neutral |
| triple_barrier_label | 63.9% | +29.2% | 65.0% | +22.5% | Worse |
| purged_embargo | 62.8% | +12.1% | 58.7% | +17.9% | Much worse |
| derivatives_features | 85.7% | +6.8% | 83.3% | +12.1% | Only 2 trades, meaningless |

### V5.6 Feature Additions — COMPLETE, archived

Tested 20 new OHLCV-derived features (Garman-Klass vol, ADX, Parkinson vol, Rogers-Satchell, Hurst exponent, MFI, realized skewness/kurtosis, etc.). Only **ADX** (adx_14h, plus_di_14h, minus_di_14h) and **Garman-Klass volatility** (gk_volatility_14h, gk_volatility_48h) proved useful → added directly to V5 production. Production now has 49 base technical features (was 44).

### V6 Enhancements (testing_literature_v2.py) — IN PROGRESS

Testing 12 V6 literature enhancements individually via Mode D BTC 4h+8h 1y:

| # | Enhancement | Description | Status |
|---|-------------|-------------|--------|
| 0 | baseline | All enhancements OFF | 4h done, 8h running |
| 1 | wavelet_denoising | Denoise close price via DWT | pending |
| 2 | fractional_diff | Fractionally differentiated price features | pending |
| 3 | hmm_regime | Hidden Markov Model regime detection | pending |
| 4 | xgboost_model | Add XGBoost to model pool | pending |
| 5 | sample_weighting | Time-decay + uniqueness sample weights | pending |
| 6 | entropy_filter | Shannon entropy signal filtering | pending |
| 7 | tri_state_labels | 3-class labeling (BUY/SELL/NO-ACTION) | pending |
| 8 | stacking_ensemble | Stacking meta-learner on model probabilities | pending |
| 9 | dynamic_feature_select | MI-based feature selection per step | pending |
| 10 | meta_labeling | Second model filters primary signals | pending |
| 11 | adversarial_validation | Drop distribution-shift features per step | pending |
| 12 | kelly_sizing | Fractional Kelly position sizing in backtest | pending |

---

## Pending Actions

- [x] **V5.5 A/B tests** — COMPLETE, archived. Only slippage_model won → promoted. See [Enhancement Testing History](#enhancement-testing-history).
- [x] **V5.6 feature tests** — COMPLETE, archived. Only ADX + Garman-Klass useful → added to production (49 base features).

### Desktop TODO
- [ ] **V6 A/B tests** — `python testing_literature_v2.py --resume` — RUNNING. Laptop hangs (loky deadlock), must run on desktop.

### Laptop TODO (can run in parallel)
- [ ] **XRP V5** — `python crypto_trading_system.py D XRP 4,8h 1y` then `F XRP 4,8h`
- [ ] **V15 first run** — `python crypto_trading_system_v15.py D BTC 4,8h 1y`
- [ ] **V30 first run** — `python crypto_trading_system_v30.py D BTC 4,8h 1y`

### Either machine
- [ ] **Feature stability test** — `python testing_feature_stability.py` — BTC+ETH × 4h+8h
- [ ] **Weekly F runs** — re-run `F BTC 4,8h` and `F ETH 4,8h` weekly
- [x] **Re-run Mode DF for BTC** — DONE (2026-03-15). Fresh models in best_models.csv.
- [x] **Re-run Mode F for ETH** — DONE (2026-03-15). ETH: `8h_only`, min_confidence=85%.
- [x] **Restart crypto_revolut_trader** — DONE (2026-03-15). 5-min hot-reload for config+models+positions.
- [x] **Laptop venv install** — DONE. PyWavelets + xgboost installed.
- [x] V5 scoring (acc×(1+return/100))
- [x] Mode F (strategy comparison + confidence sweep)
- [x] Per-step timers in Mode D
- [x] hardware_config diagnostic models lightened
- [x] DIAG_WINDOWS reduced to [48,72,100,150,200]
- [x] BTC V5 2y run — 4h: 80.2% +125%, 8h: 84.7% +319%
- [x] ETH V5 2y run — 4h: RF+LGBM 68.6% +505%, 8h: GB 79.3% +616% (WARNING: overfit)
- [x] V5 promoted to production, V3 archived
- [x] Old/archived files moved to archive/
- [x] V5.3 and V5.4 experimental versions added
- [x] IB auto-trader for DAX + S&P 500 CFDs
- [x] Broly 1.2 enhancement layer
- [x] Multi-asset crypto auto-trader with Telegram control
- [x] Mock validation framework (Phase 1 + Phase 2)
- [x] Crypto correlation analysis (BTC vs ETH)

---

*Last updated: March 15, 2026 — V6 created (12 literature enhancements). DF combined mode added. /conf + /chart Telegram commands. Dual-machine install docs. fracdiff+hmmlearn replaced with manual implementations (Python 3.14 compat).*
