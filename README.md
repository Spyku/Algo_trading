# Algo Trading Engine

Automated ML-powered trading system for **crypto** (BTC, ETH, XRP) and **index CFDs** (DAX, S&P 500). Two independent pipelines:

1. **Crypto Pipeline** — Hourly BUY/SELL/HOLD signals via dual-horizon (4h + 8h) ensemble ML with walk-forward validation. Executes via exchange API.
2. **Sub-Hourly Crypto (V15/V30)** — 15-min and 30-min candle variants for higher trade frequency. Same ML pipeline, separate data/models/configs.
3. **Index CFD Pipeline** — Hourly signals via Broly 1.2 ML model. Executes via broker API.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Crypto Trading System (CASCA)](#crypto-trading-system-casca-production)
  - [Modes of Operation](#modes-of-operation)
  - [ML Pipeline](#ml-pipeline)
  - [Features](#features-125-total)
  - [Scoring Formulas](#scoring-formulas) (CASCA vs V5)
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

# === CASCA (profit factor scoring — use this) ===
python crypto_trading_system_casca.py D BTC 4,8h    # CASCA Mode D — full pipeline with PF scoring
python crypto_trading_system_casca.py F BTC 4,8h    # CASCA Mode F — strategy comparison (by return)
python crypto_trading_system_casca.py DF BTC,ETH 4,8h  # CASCA D then F

python crypto_trading_system_casca.py A BTC 4,8h    # CASCA Mode A — gamma optimization (6 gammas × horizons)
python crypto_trading_system_casca.py B BTC 4,8h    # CASCA Mode B — signals from saved models

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

# === CRYPTO SUB-HOURLY (V15 / V30 — CASCA grid search) ===
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
│  crypto_trading_system_casca.py  (CASCA — Profit Factor Scoring)     │
│    ├── Scoring: Profit Factor (gross_profit/|gross_loss|, cap 5.0)  │
│    ├── Feature selection by PF (permutation, ablation, reduced sets)│
│    ├── Mode F: strategies ranked by return, not accuracy             │
│    ├── Output: models/crypto_casca_best_models.csv                   │
│    └── Same pipeline as V5 — only scoring logic changed              │
│                                                                     │
│    ├── Mode A: gamma optimization (6 gammas × horizons)              │
│    ├── Temporal decay (gamma per model) + 6-month data cap          │
│    ├── 130 features (49 technical + 81 macro/sentiment/cross-asset) │
│    ├── 4 ML models × 15 combinations × 5 windows                   │
│    ├── Walk-forward validation (no future leakage)                  │
│    └── hardware_config.py (machine-specific GPU/CPU settings)       │
│                                                                     │
│  crypto_trading_system_deku.py  (Deku — Optuna Bayesian Optimization)│
│    ├── Optuna TPE + Hyperband pruning (100 trials default)          │
│    ├── Joint optimization: (combo, window, gamma, n_features)       │
│    ├── 5 models: RF, GB, XGB, LR, LGBM — 26 combos                │
│    ├── APF scoring: raw_PF / buyhold_PF (regime-normalized)        │
│    ├── LGBM importance ranking (~5s vs ~10min feature analysis)     │
│    ├── Continuous gamma [0.995–1.0] (vs CASCA discrete grid)       │
│    └── Output: models/crypto_deku_best_models.csv                   │
│                                                                     │
│  crypto_trading_system_deku_15m.py  (Deku V15 — 15-min candles)     │
│    ├── Same Optuna pipeline, 15-min candle scaling                  │
│    ├── s4=60', s8=120' horizons, 4320-candle cap (~45 days)        │
│    └── Output: models/crypto_deku_15m_best_models.csv               │
│                                                                     │
│  crypto_trading_system_v6.py  (V6 Experimental)                     │
│    ├── 12 ENHANCEMENTS flags (toggle on/off, env var override)      │
│    ├── Wavelet denoising, fractional diff, GMM regime detection     │
│    ├── XGBoost, sample weighting, entropy filter, tri-state labels  │
│    ├── Stacking ensemble, dynamic feature select, meta-labeling     │
│    ├── Adversarial validation, Kelly sizing                         │
│    └── testing_literature_v2.py (A/B test harness)                  │
│                                                                     │
│  crypto_trading_system_v15.py  (V15 Cacarot — 15-min candles)       │
│    ├── Horizons 1-8 = 15'–120' (more trades, shorter horizons)     │
│    ├── Temporal decay (gamma) + 4320-row data cap (~45 days)       │
│    ├── Separate data/models/config (*_15m_*)                        │
│    └── Max 1 year rolling data, ~35K rows/year                     │
│                                                                     │
│  crypto_trading_system_v30.py  (V30 Cacarot — 30-min candles)       │
│    ├── Horizons 1-8 = 30'–240' (2× hourly frequency)              │
│    ├── Temporal decay (gamma) + 4320-row data cap (~3 months)      │
│    ├── Separate data/models/config (*_30m_*)                        │
│    └── Max 1 year rolling data, ~17.5K rows/year                   │
│                                                                     │
│  crypto_live_trader.py  (506 lines — signal generation library)     │
│    ├── Imports models/features from crypto_trading_system_casca.py   │
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

## Crypto Trading System (CASCA Production)

### Core Concepts

| Concept | Detail |
|---------|--------|
| **Dual horizons** | 4h and 8h — completely independent models per horizon (different features, window, combo). Never mixed during training. |
| **No model persistence** | System retrains from scratch every prediction. No .pkl files. Ensures fresh models with current market regime. |
| **Temporal decay (Cacarot)** | Exponential sample weighting `w_i = gamma^(age)`. Per-model gamma stored in best_models.csv. Newest sample weight=1, oldest=gamma^(n-1). gamma >= 1.0 disables decay. |
| **6-month data cap** | Mode D/E cap training data at 4,320 hours. Prevents ancient data from diluting recent patterns. |
| **Labels** | Relative to 168h rolling median return, not absolute price direction. `label = 1` means return > median. Drift-robust. |
| **Walk-forward** | Train on last `window` hours → predict next candle → step forward 72h. No future leakage. |
| **Ensemble voting** | Majority vote across model combo (>50% BUY = BUY signal). Confidence = avg probability. |

### Modes of Operation

| Mode | Purpose | Time (desktop) | When to Run |
|------|---------|----------------|-------------|
| **A** | Gamma optimization — test 6 decay values (0.999–0.994) per horizon, ranked by PF | ~9h per horizon | After Mode D, to tune temporal decay |
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
   - Permutation importance (shuffle-and-measure impact — ranked by profit factor)
   - Ablation (drop-one-at-a-time — ranked by PF change)
   - Reduced sets (top-N feature subsets — ranked by PF)
   - Consensus scoring → keep/maybe/drop bins → optimal subset
5. **Diagnostic** — Grid search: 15 model combos × 5 windows = 75 configs, parallel walk-forward evaluation (~16–18 min). Ranked by Profit Factor.
6. **Signal generation** — Generate signals with best config, bootstrap CI, portfolio simulation (~2 min)
7. **Charts** — Matplotlib backtest PNG + Plotly interactive HTML (4-panel strategy charts + signal table)
8. **Save** — Best model to `crypto_casca_best_models.csv`, features to analysis CSV

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

### Scoring Formulas

#### CASCA Scoring (current — `crypto_trading_system_casca.py`)

```python
# Model selection (diagnostic):
if trades < 3:
    combined_score = 0.0
elif total_loss == 0:
    combined_score = min(total_gain * 100, 5.0)
else:
    combined_score = min(total_gain / abs(total_loss), 5.0)  # Profit Factor

# Feature selection (permutation, ablation, reduced sets):
# _quick_score() returns profit factor — features ranked by PF contribution

# Mode F strategy + confidence sweep:
score = cum_return  # strategies ranked by return directly
```

Profit Factor directly measures profitability. PF > 1.0 = profitable, PF > 2.0 = strong. Cap at 5.0 prevents infinity when 0 losses. Min 3 trades filters noise.

#### V5 Cacarot Scoring (old — `crypto_trading_system.py`)

```python
combined_score = accuracy * (1 + max(cum_return, 0) / 100)
```

**Known broken:** picks money-losing models because accuracy dominates and negative returns are clamped to zero. Example: V5 picked RF+GB+LR+LGBM w=96 (+14.2% return, 84% acc) over RF w=48 (+22.9% return, 73% acc). Replaced V4's `0.45×Calmar + 0.35×Sharpe + 0.20×Accuracy` which biased toward low-trade configs.

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

The hourly system produces good signals but few trades. V15 and V30 use shorter candles for higher trade frequency while keeping the same ML pipeline. Available in both CASCA (grid search) and Deku (Optuna) variants.

| System | Candle | Horizons | Data cap | Optimizer | File |
|--------|--------|----------|----------|-----------|------|
| **Deku V15** | 15 min | s4=60', s8=120' | 4320 rows (~45 days) | Optuna TPE+Hyperband | `crypto_trading_system_deku_15m.py` |
| **V15 Cacarot** | 15 min | 15'–120' | 4320 rows (~45 days) | Grid search | `crypto_trading_system_v15.py` |
| **V30 Cacarot** | 30 min | 30'–240' | 4320 rows (~3 months) | Grid search | `crypto_trading_system_v30.py` |

**Key differences from hourly system:**
- All rolling/shift/diff periods scaled via `_hours_to_rows()` (e.g., RSI 14h = 56 candles on V15)
- Extra features: `logret_1c..4c` (candle-scale returns), `minute_sin/cos` (intra-hour timing)
- Data cap: 4,320 rows (same computational cost as V5's 6 months, but shorter wall-clock span due to faster candles)
- Temporal decay: same `w_i = gamma^(age_in_candles)` — gamma stored per model in CSV
- Crypto assets only (BTC, ETH, XRP, DOGE) — no yfinance indices
- Separate output files: `data/*_15m_data.csv`, `models/crypto_15m_best_models.csv`, `config/trading_config_15m.json`, `charts/*_15m_backtest.png`

**Usage:**
```bash
python crypto_trading_system_v15.py D BTC 4,8h       # horizon 4 = 60min, 8 = 120min
python crypto_trading_system_v30.py D BTC 4,8h       # horizon 4 = 2h, 8 = 4h
python crypto_trading_system_v15.py DF BTC 4,8h      # Mode D then F in one command
python crypto_trading_system_v15.py F BTC 4,8h       # strategy optimizer
python crypto_trading_system_v15.py G BTC             # horizon pair test
```

### Gamma Optimization (V15.1 / V30.1)

Dedicated test harnesses to find the optimal decay rate for each sub-hourly system. Tests 7 gamma values (1.0, 0.999, 0.998, 0.997, 0.996, 0.995, 0.994) × 2 horizons (4, 8) = 14 full Mode D runs each.

```bash
python testing_v15.1.py              # V15 gamma optimization (14 tests)
python testing_v15.1.py --resume     # Skip completed combos
python testing_v30.1.py              # V30 gamma optimization (14 tests)
python testing_v30.1.py --resume     # Skip completed combos
```

Results: `models/testing_v15.1_results.csv`, `models/testing_v30.1_results.csv`

### Casca — Multi-Timeframe Fusion Test

Combines V5 Cacarot (1h candles, 4h+8h horizons) and V15 (15-min candles, 60'+120' horizons) into a single decision engine. Tests 16 cross-timeframe strategies to see if fusion beats any single timeframe.

**Strategy types:**
- V5-only baselines (4): `v5_both_agree`, `v5_either_agree`, `v5_4h_only`, `v5_8h_only`
- V15-only baselines (4): `v15_60m_only`, `v15_120m_only`, `v15_both_agree`, `v15_either_agree`
- Cross-timeframe fusion (8): `all_4_agree`, `any_4_buy`, `majority_3of4`, `v5_8h_and_v15_120m`, `v5_8h_confirmed_v15`, `v15_fast_entry_v5_exit`, etc.

**Two evaluation cadences:** hourly (V15 snapped to :00) and 15-min (V5 carried forward). Confidence sweep across [55–90%].

```bash
python testing_casca.py              # Hourly evaluation (default)
python testing_casca.py --15min      # 15-min evaluation cadence
python testing_casca.py ETH          # Test ETH
```

Results: `models/testing_casca_results.csv`

### Deku — Multi-Timeframe Fusion Test

Combines Deku hourly (1h candles, 4h+8h horizons) and Deku V15 (15-min candles, s4+s8 horizons) into a single decision engine. Tests 16 cross-timeframe strategies to find optimal signal mixing.

**Strategy types:**
- 1h-only (4): `1h_both_agree`, `1h_either_agree`, `1h_4h_only`, `1h_8h_only`
- 15'-only (4): `15m_s4_only`, `15m_s8_only`, `15m_both_agree`, `15m_either_agree`
- Cross-TF (8): `all_4_agree`, `any_4_buy`, `majority_3of4`, `1h_8h_and_15m_s8`, `1h_8h_or_15m_s8`, `1h_4h_and_15m_s4`, `1h_8h_confirmed_15m`, `15m_fast_entry_1h_exit`

**Two evaluation cadences:** hourly (15' signals snapped to :00) and 15-min (1h signals carried forward). Confidence sweep [55–90%]. Max drawdown tracking.

```bash
python testing_deku_fusion.py BTC              # Hourly evaluation cadence
python testing_deku_fusion.py BTC --15min      # 15-min evaluation cadence
python testing_deku_fusion.py BTC --compare    # Deku V15 vs CASCA V15 side-by-side
```

Results: `models/testing_deku_fusion_results.csv`

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
| `crypto_trading_system_casca.py` | ~3,900 | **CASCA Production** — Modes A/B/D/E/F/DF. Profit factor scoring. MIN_COMBO_SIZE=2 (no solo models). Mode A gamma optimization. Live trader imports from this. Output: `crypto_casca_best_models.csv` |
| `crypto_trading_system_casca_v1.3.1.py` | ~3,900 | **CASCA V1.3.1 Experimental** — Per-model feature analysis: each model (RF, GB, LR, LGBM) gets its own optimal feature set. Feature analysis cached to JSON for reuse. Output: `crypto_casca_v1.3.1_best_models.csv` |
| `crypto_trading_system_casca_v1.4.py` | ~3,900 | **CASCA V1.4 Experimental** — CASCA baseline for Deku comparison. LGBM importance ranking. Output: `crypto_casca_v1.4_best_models.csv` |
| `crypto_trading_system_deku.py` | ~4,900 | **Deku Hourly** — Optuna TPE+Hyperband. 5 models (RF, GB, XGB, LR, LGBM), 26 combos. APF scoring. Output: `crypto_deku_best_models.csv` |
| `crypto_trading_system_deku_15m.py` | ~4,900 | **Deku V15** — Deku with 15-min candles, s4=60'/s8=120', 4320-candle cap. Output: `crypto_deku_15m_best_models.csv` |
| `crypto_revolut_trader.py` | 1,109 | Multi-asset crypto auto-trader |
| `crypto_live_trader.py` | 506 | Signal generation library (imported by trader) |
| `crypto_trading_system_v15.py` | ~3,600 | V15 Cacarot — 15-min candles, temporal decay, 4320-row cap (Modes B/D/E/F/G/DF) |
| `crypto_trading_system_v30.py` | ~3,600 | V30 Cacarot — 30-min candles, temporal decay, 4320-row cap (Modes B/D/E/F/G/DF) |
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
| `testing_v15.1.py` | ~275 | V15 gamma optimization — 7 gammas × 2 horizons, isolated results |
| `testing_v30.1.py` | ~275 | V30 gamma optimization — 7 gammas × 2 horizons, isolated results |
| `testing_casca.py` | ~600 | Multi-timeframe fusion — V5 + V15 combined, 16 strategies, confidence sweep |
| `testing_deku_fusion.py` | ~400 | Deku multi-timeframe fusion — Deku hourly + Deku V15, 16 strategies, confidence sweep, max drawdown |
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
| `crypto_trading_system_deku.py` | Deku | **Hourly** — Optuna TPE+Hyperband. 5 models (RF, GB, XGB, LR, LGBM), 26 combos. APF scoring. Continuous gamma. |
| `crypto_trading_system_deku_15m.py` | Deku V15 | **Sub-hourly** — Deku with 15-min candles, s4=60'/s8=120', 4320-candle cap. |
| `crypto_trading_system_casca.py` | CASCA | **Production** — Profit Factor scoring. Modes A/B/D/E/F/DF. Mode A gamma optimization. Live trader imports from this. |
| `archive/crypto_trading_system_v5_cacarot.py` | V5.4+ | Archived — old scoring (acc×ret). Superseded by CASCA. |
| `crypto_trading_system_v6.py` | V6 | Experimental — 12 literature enhancements behind ENHANCEMENTS flags (env var override) |
| `archive/crypto_trading_system_v5.5.py` | V5.5 | Archived — 7 literature enhancements tested, only slippage_model promoted |
| `crypto_trading_system_v15.py` | V15 Cacarot | Sub-hourly — 15-min candles, temporal decay + 4320-row cap |
| `crypto_trading_system_v30.py` | V30 Cacarot | Sub-hourly — 30-min candles, temporal decay + 4320-row cap |
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
├── crypto_trading_system_casca.py     # CASCA Production — PF scoring, Modes A/B/D/E/F/DF (live trader imports)
├── crypto_trading_system_v15.py       # V15 Cacarot — 15-min candles (temporal decay + 4320-row cap)
├── crypto_trading_system_v30.py       # V30 Cacarot — 30-min candles (temporal decay + 4320-row cap)
├── crypto_revolut_trader.py           # Multi-asset crypto auto-trader
├── crypto_live_trader.py              # Signal generation library (NOT run directly)
├── hardware_config.py                 # Auto-detects Desktop/Laptop config
├── download_macro_data.py             # Macro data downloader (macro + on-chain + derivatives)
├── crypto_trading_system_deku.py      # Deku hourly — Optuna + XGBoost, APF scoring
├── crypto_trading_system_deku_15m.py  # Deku V15 — 15-min candles, Optuna optimization
├── crypto_trading_system_casca_v1.4.py # CASCA V1.4 — baseline for Deku comparison
├── crypto_trading_system_v6.py        # V6 — 12 literature enhancements (experimental)
├── # testing_literature.py → archived (V5.5 A/B tests COMPLETE)
├── testing_literature_v2.py           # A/B test harness for V6 (12 enhancements)
├── testing_v15.1.py                  # V15 gamma optimization (7 gammas × 2 horizons)
├── testing_v30.1.py                  # V30 gamma optimization (7 gammas × 2 horizons)
├── testing_casca.py                  # Multi-timeframe fusion test (V5 + V15, 16 strategies)
├── testing_deku_fusion.py            # Deku fusion test (1h + 15', 16 strategies)
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
│   ├── crypto_hourly_best_models.csv  # V5 Cacarot: best model per (asset, horizon)
│   ├── crypto_casca_best_models.csv   # CASCA: best model per (asset, horizon) — PF scored
│   ├── crypto_deku_best_models.csv   # Deku hourly: best model per (asset, horizon) — APF scored
│   ├── crypto_deku_15m_best_models.csv # Deku V15: 15-min candle models — APF scored
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

### Crypto (CASCA Production)

| Asset | Horizon | Models | Window | Accuracy | Return | PF Score | Features | Gamma | Status |
|-------|---------|--------|--------|----------|--------|----------|----------|-------|--------|
| BTC | 4h | RF+GB | 48h | 72.9% | +27.2% | 5.00 | 34 | 1.0 | CASCA 6mo |
| BTC | 8h | GB+LGBM | 150h | 72.4% | +25.4% | 2.00 | 8 | 1.0 | CASCA 6mo |
| ETH | 4h | — | — | — | — | — | — | — | CASCA DF running |
| ETH | 8h | — | — | — | — | — | — | — | CASCA DF running |
| XRP | 4h | RF+GB+LGBM | 100h | 71.7% | +202.3% | — | 15 | 1.0 | V5 2y |
| XRP | 8h | RF | 100h | 82.5% | +139.2% | — | 50 | 1.0 | V5 2y |

**BTC trading config:** `4h_only` @80% confidence, $6k max position. Mode F selected 2026-03-17.

### Crypto (Deku — Optuna)

| Asset | Horizon | Models | Window | Return | APF Score | Gamma | Status |
|-------|---------|--------|--------|--------|-----------|-------|--------|
| BTC | 4h | RF+XGB+LGBM | 48h | +54.2% | 5.902 | 0.9951 | Deku 6mo |
| BTC | 8h | RF+XGB+LR+LGBM | 48h | +36.2% | 4.543 | 0.9995 | Deku 6mo |

### Mode A Gamma Optimization Results (BTC, 2026-03-17)

Best gamma per horizon (tested 0.999–0.994, baseline gamma=1.0):

**BTC 4h — Winner: gamma=0.996** (baseline +27.2% → +55.8%, 2× improvement)

| Gamma | Combo | Window | Acc | Return | Win% | PF |
|-------|-------|--------|-----|--------|------|----|
| 0.999 | LR+LGBM | 200h | 68.4% | +24.8% | 66.7% | 1.80 |
| 0.998 | RF+LGBM | 200h | 73.7% | +45.2% | 70.6% | 3.80 |
| 0.997 | LGBM | 200h | 68.4% | +35.7% | 76.5% | 2.75 |
| **0.996** | **LGBM** | **200h** | **75.4%** | **+55.8%** | **81.2%** | **5.00** |
| 0.995 | RF+GB | 200h | 73.7% | +50.5% | 75.0% | 5.00 |
| 0.994 | LR+LGBM | 200h | 66.7% | +35.1% | 78.6% | 5.00 |

**BTC 8h — Winner: gamma=0.997** (baseline +25.4% → +50.0%, 2× improvement)

| Gamma | Combo | Window | Acc | Return | Win% | PF |
|-------|-------|--------|-----|--------|------|----|
| 0.999 | RF+GB+LGBM | 200h | 89.5% | +34.7% | 75.0% | 4.48 |
| 0.998 | GB | 200h | 87.7% | +34.7% | 75.0% | 4.48 |
| **0.997** | **RF+LGBM** | **100h** | **87.9%** | **+50.0%** | **73.3%** | **3.88** |
| 0.996 | RF+GB | 100h | 86.2% | +43.6% | 63.6% | 5.00 |
| 0.995 | RF+LR | 100h | 84.5% | +34.7% | 62.5% | 2.63 |
| 0.994 | RF+LGBM | 100h | 87.9% | +37.6% | 66.7% | 3.11 |

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
| **Deku** | `crypto_trading_system_deku.py` | APF (PF/buyhold) | Optuna TPE+Hyperband Bayesian optimization. 5 models (RF, GB, XGB, LR, LGBM), 26 combos. Joint search over (combo, window, gamma, n_features). LGBM importance ranking. BTC 4h: +54.2% APF=5.9 vs CASCA +27.2%. |
| **Deku V15** | `crypto_trading_system_deku_15m.py` | APF (PF/buyhold) | Deku with 15-min candles. s4=60', s8=120'. 4320-candle cap (~45 days). |
| **CASCA** | `crypto_trading_system_casca.py` | Profit Factor | **Production.** PF model selection (cap 5.0, min 3 trades). Feature selection by PF. MIN_COMBO_SIZE=2 (no solo models). Mode A gamma optimization. Mode F by return. Live trader imports from this. |
| **CASCA V1.3.1** | `crypto_trading_system_casca_v1.3.1.py` | Profit Factor | **Experimental.** Per-model feature analysis: 5-test pipeline run independently for RF, GB, LR, LGBM. Each model gets its own optimal features. Union matrix with per-model column slicing. Feature analysis cached to `models/feature_analysis_v1.3.1_{ASSET}_{H}h.json` — skip ~80 min on re-runs. |
| **V5 Cacarot** | `archive/crypto_trading_system_v5_cacarot.py` | `acc × (1 + ret/100)` | **Archived.** Temporal decay, 6mo data cap, DF mode, slippage model. Superseded by CASCA. |
| **V5.5** | `crypto_trading_system_v5.5.py` | `acc × (1 + ret/100)` | Experimental. 7 literature enhancements behind ENHANCEMENTS flags (on-chain, derivatives, triple-barrier, slippage, extended diag, GB calibration, purged embargo). A/B testable via `testing_literature.py` |
| **V15 Cacarot** | `crypto_trading_system_v15.py` | `acc × (1 + ret/100)` | Sub-hourly. 15-min candles, 8 horizons (15'–120'), temporal decay, 4320-row cap |
| **V30 Cacarot** | `crypto_trading_system_v30.py` | `acc × (1 + ret/100)` | Sub-hourly. 30-min candles, 8 horizons (30'–240'), temporal decay, 4320-row cap |
| **V5.3** | `archive/crypto_trading_system_v5.3.py` | Archived | LGBM lightened, LOKY_MAX_CPU_COUNT capped, OMP/MKL/OpenBLAS thread limits |
| **V5.2** | `archive/crypto_trading_system_v5.2.py` | Archived | All 8 horizons + Mode G (no BLAS fixes) |
| **V5.1** | `archive/crypto_trading_system_v5.1.py` | Archived | Alpha scoring patches integrated |
| **V4** | `archive/crypto_trading_system_v4.py` | `0.45×Calmar + 0.35×Sharpe + 0.20×Acc` | Bootstrap CI, Calmar/Sharpe, --permtest. Superseded — biased toward low-trade configs |
| **V3** | `archive/crypto_trading_system_v3_old.py` | Heuristic | Original production. Archived |

### What Deku Adds Over CASCA

1. **Optuna Bayesian optimization** — TPE + Hyperband jointly optimizes (combo, window, gamma, n_features) in ~100 trials. CASCA grid-searches 75 fixed configs. Deku finds better hyperparameter combinations in similar time.
2. **XGBoost (5th model)** — 26 ensemble combinations (vs CASCA's 11 from 4 models). More diversity, better ensembles.
3. **APF scoring** — `raw_PF / buyhold_PF` normalizes against market regime. A PF of 3 in a bull run is less impressive than PF=3 in a bear market.
4. **LGBM importance ranking** — Features ranked by LGBM gain importance (~5 sec) instead of 5-test analysis (~10 min). Optuna picks n_features from the ranked list.
5. **Continuous gamma** — Optuna searches [0.995–1.0] continuously instead of testing 6 discrete values with Mode A.

### What CASCA Adds Over V5

1. **Profit Factor scoring** — `gross_profit / |gross_loss|` replaces `acc × (1 + ret/100)`. V5 picked money-losing models because accuracy dominated. CASCA picks models that actually make money.
2. **Consistent scoring** — Profit factor used everywhere: diagnostic model ranking, feature selection (permutation, ablation, reduced sets), and Mode F strategies ranked by return.
3. **Minimum trade filter** — Models with < 3 trades score 0 (too few to be meaningful).

### What V5 Adds Over V4

1. **New scoring formula** — `accuracy × (1 + max(return, 0) / 100)` replaces Calmar/Sharpe. Better than V4 but still broken — superseded by CASCA.
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
| **2026-03-15–16** | `6530d60` | **CASCA Production** — PF scoring replaces acc×ret. V5 archived. Mode A gamma optimization. V15/V30 Cacarot. |
| **2026-03-17** | — | **CASCA DF BTC** — RF+GB 4h PF=5.00, GB+LGBM 8h PF=2.00. V1.3.1 (per-model features). V30 CASCA port. MIN_COMBO_SIZE=2. |
| **2026-03-18** | — | **Deku release** — Optuna TPE+Hyperband replaces CASCA grid search. XGBoost (5th model), APF scoring, LGBM importance ranking, continuous gamma. BTC 4h: +54.2% APF=5.9 (2× CASCA). Deku V15 (15-min candles). Deku fusion test harness. |

### Session History (Detailed)

1. **Mar 2**: Added V2 macro features. Created crypto_trading_system.py with Modes A/B/C. BTC 4h: Set A 76.5% vs Set B 75.7%.
2. **Mar 3**: Mode D (full feature analysis pipeline), Telegram live trader, README.
3. **Mar 4**: Confidence threshold backtester, derivative features, live trader timing fix. Dual horizon (1h+4h → 4h+8h). Fixed critical run_loop bug. Mode E (iterative refinement). VVR feature, profit-weighted scoring.
4. **Mar 7**: 8h horizon — BTC 8h 80.3%. "Both Agree" strategy: +97.2% alpha, 90% win rate. Crypto auto-trader built.
5. **Mar 7-8**: ETH/XRP analysis. SOL removed (poor results). Per-asset strategies. Exchange API fixes. 4-panel Plotly charts, signal table HTML. Weighted strategy tested & rejected.
6. **Mar 8**: V4 created — bootstrap CI, Calmar/Sharpe scoring, --permtest. Mock validation (Phase 1 + Phase 2) on synthetic data. hardware_config lightened. DIAG_WINDOWS reduced. BTC 4h V4: GB+LR w=200h 63% +48.7%.
7. **Mar 9**: V5 created — scoring replaced with acc×(1+ret/100). Mode F + confidence sweep. Per-step timers. SSL fix. Trader bugfixes.
8. **Mar 10**: V5 Mode D BTC 4,8h 2y complete. **4h: RF+GB+LR w=100h 80.2% +125%. 8h: RF+GB w=150h 84.7% +319%.** Mode F: either_agree + 75%. V5 promoted to production. V3 archived.
9. **Mar 15-16**: Cacarot release (temporal decay + 6mo data cap). CASCA created (profit factor scoring). V5 archived. V15/V30 Cacarot+DF. CASCA V1.1 (fee-aware labels). Live trader connected to CASCA.
10. **Mar 17**: CASCA DF BTC complete (RF+GB 4h PF=5.00, GB+LGBM 8h PF=2.00, `4h_only` @80%). MIN_COMBO_SIZE=2 fixes solo-model overconfidence. V30 CASCA port (PF scoring + Mode A). CASCA V1.3.1 created (per-model feature analysis). ETH DF running.
11. **Mar 18**: **Deku release** — Optuna TPE+Hyperband Bayesian optimization replaces CASCA sequential grid search. XGBoost added as 5th model (26 combos). APF scoring (PF/buyhold). LGBM importance ranking (~5s vs ~10min). Continuous gamma [0.995–1.0]. BTC 4h: RF+XGB+LGBM +54.2% APF=5.9 (vs CASCA +27.2%). Deku V15 (15-min candles, s4=60'/s8=120'). CASCA V1.4 baseline. Deku fusion test harness (16 strategies, 2 cadences).

---

## File Dependencies

```
crypto_trading_system_casca.py  (CASCA production — PF scoring, Modes A/B/D/E/F/DF)
  ├── imports: hardware_config.py
  ├── reads: data/macro_data/*.csv
  ├── reads/writes: data/{asset}_hourly_data.csv
  ├── reads/writes: models/crypto_casca_best_models.csv
  ├── reads/writes: config/trading_config.json  (Mode F)
  ├── writes: charts/*.html, charts/*.png, models/*.json
  └── writes: models/testing_casca_a_results.csv  (Mode A gamma results)

crypto_trading_system_casca_v1.3.1.py  (CASCA V1.3.1 experimental — per-model features)
  ├── imports: hardware_config.py
  ├── reads/writes: models/crypto_casca_v1.3.1_best_models.csv
  ├── reads/writes: config/trading_config_v1.3.1.json
  ├── reads/writes: models/feature_analysis_v1.3.1_{ASSET}_{H}h.json  (feature cache)
  └── writes: models/testing_casca_v1.3.1_a_results.csv  (Mode A)

crypto_trading_system_v15.py  (V15 Cacarot — 15-min candles)
  ├── imports: hardware_config.py
  ├── reads: data/macro_data/*.csv
  ├── reads/writes: data/{asset}_15m_data.csv
  ├── reads/writes: models/crypto_15m_best_models.csv  (includes gamma per model)
  ├── reads/writes: config/trading_config_15m.json  (Mode F)
  └── writes: charts/*_15m_*.html, charts/*_15m_backtest.png

crypto_trading_system_v30.py  (V30 Cacarot — 30-min candles)
  ├── imports: hardware_config.py
  ├── reads: data/macro_data/*.csv
  ├── reads/writes: data/{asset}_30m_data.csv
  ├── reads/writes: models/crypto_30m_best_models.csv  (includes gamma per model)
  ├── reads/writes: config/trading_config_30m.json  (Mode F)
  └── writes: charts/*_30m_*.html, charts/*_30m_backtest.png

testing_v15.1.py  (V15 gamma optimization)
  ├── imports: crypto_trading_system_v15.py  (patches _load_mode_d_config)
  └── writes: models/testing_v15.1_results.csv, charts/v15.1_test/

testing_v30.1.py  (V30 gamma optimization)
  ├── imports: crypto_trading_system_v30.py  (patches _load_mode_d_config)
  └── writes: models/testing_v30.1_results.csv, charts/v30.1_test/

crypto_trading_system_deku.py  (Deku hourly — Optuna optimization)
  ├── imports: hardware_config.py
  ├── reads: data/macro_data/*.csv
  ├── reads/writes: data/{asset}_hourly_data.csv
  ├── reads/writes: models/crypto_deku_best_models.csv
  ├── reads/writes: config/trading_config_deku.json  (Mode F)
  └── writes: charts/*.html, charts/*.png

crypto_trading_system_deku_15m.py  (Deku V15 — 15-min candles)
  ├── imports: hardware_config.py
  ├── reads/writes: data/{asset}_15m_data.csv
  ├── reads/writes: models/crypto_deku_15m_best_models.csv
  ├── reads/writes: config/trading_config_deku_15m.json  (Mode F)
  └── writes: charts/*_15m_*.html, charts/*_15m_backtest.png

testing_deku_fusion.py  (Deku multi-timeframe fusion)
  ├── imports: crypto_trading_system_deku.py  (Deku hourly signals)
  ├── imports: crypto_trading_system_deku_15m.py  (Deku V15 signals)
  └── writes: models/testing_deku_fusion_results.csv

testing_casca.py  (multi-timeframe fusion)
  ├── imports: crypto_trading_system_casca.py  (CASCA signals)
  ├── imports: crypto_trading_system_v15.py  (V15 signals)
  └── writes: models/testing_casca_results.csv, charts/casca_test/

crypto_live_trader.py  (signal generation)
  ├── imports: crypto_trading_system_casca.py  (ASSETS, features, models, download/load/build)
  ├── reads: models/crypto_casca_best_models.csv
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

### Active

| # | Task | Command | Status |
|---|------|---------|--------|
| 1 | **Deku V15 BTC** | `python crypto_trading_system_deku_15m.py D BTC 4,8h` | RUNNING (2026-03-18) |
| 2 | **Deku fusion backtest** | `python testing_deku_fusion.py BTC` + `--15min` | After Deku V15 completes |
| 3 | **Apply Mode A gamma winners** | Re-run Mode D with gamma=0.996 (4h), 0.997 (8h) | Ready |
| 4 | **Test CASCA V1.1** | `python crypto_trading_system_casca_v1.1.py D BTC 4,8h` | Fee-aware labels |
| 5 | **Re-enable ETH** | Set `enabled: true` in trading_config.json | After ETH DF completes |

### After Active

| # | Task | Command | Status |
|---|------|---------|--------|
| 6 | **Deku DF ETH** | `python crypto_trading_system_deku.py DF ETH 4,8h` | Optuna for ETH |
| 7 | **V15.1 gamma optimization** | `python testing_v15.1.py --resume` | Re-run with MIN_COMBO_SIZE=2 |
| 8 | **V30.1 gamma optimization** | `python testing_v30.1.py` | 7 gammas × 2 horizons |
| 9 | **Fee-aware labels** | `label = 1` when return > 2×TRADING_FEE (0.22%) | 2 tests only |
| 10 | **Dynamic data cap** | `calc_data_cap(gamma)` formula | TODO |

### Other TODO

- [ ] **V6 A/B tests** — `python testing_literature_v2.py --resume` — RUNNING on desktop
- [ ] **XRP CASCA** — `python crypto_trading_system_casca.py DF XRP 4,8h`
- [ ] **Weekly F runs** — re-run CASCA `F BTC 4,8h` and `F ETH 4,8h` weekly
- [ ] **Windows auto-start** — CryptoTrader scheduled task registered, needs reboot test

### Completed

- [x] **Deku hourly release** — DONE (2026-03-18). Optuna+Hyperband, XGBoost, APF scoring, LGBM importance. BTC 4h: +54.2% APF=5.9 (vs CASCA +27.2%).
- [x] **Deku V15 created** — DONE (2026-03-18). 15-min candles, s4=60'/s8=120', 4320-candle cap.
- [x] **Deku fusion test harness** — DONE (2026-03-18). `testing_deku_fusion.py` — 16 strategies, 2 cadences, confidence sweep.
- [x] **CASCA V1.4 created** — DONE (2026-03-18). CASCA baseline for Deku comparison.
- [x] **CASCA DF BTC** — DONE (2026-03-17). 4h: RF+GB w=48, 72.9% acc, +27.2%, PF=5.00. 8h: GB+LGBM w=150, 72.4% acc, +25.4%, PF=2.00. Mode F: `4h_only` @80%.
- [x] **CASCA V1.3.1 created** — DONE (2026-03-17). Per-model feature analysis (RF/GB/LR/LGBM each optimized independently). Union matrix with per-model slicing. Feature analysis cached to JSON for reuse across runs (auto-invalidates on gamma change).
- [x] **V30 CASCA port** — DONE (2026-03-17). PF scoring, MIN_COMBO_SIZE=2, Mode A, `_get_models_csv_path()`.
- [x] **MIN_COMBO_SIZE=2** — DONE (2026-03-17). Solo models removed from CASCA diagnostic. Fixes overconfidence from uncalibrated solo predictions.
- [x] **CASCA scoring model** — DONE (2026-03-16). `crypto_trading_system_casca.py` — profit factor replaces acc×ret. Feature selection by PF. Isolated output: `models/crypto_casca_best_models.csv`.
- [x] **V15 Mode D + Mode F** — DONE (2026-03-16). BTC s8: LGBM 88.1% acc. Mode F: `both_agree` @61%, +62.6% return.
- [x] **V5+V15 fusion backtest** — DONE (2026-03-16). V15 override (+17.6%) beats V5 (+13.9%), V15 (+12.1%), B&H (+7.9%).
- [x] **Bug fixes** — DONE (2026-03-16). Test harnesses overwriting production config; MODELS_DIR forward reference in V15/V30.
- [x] **V15/V30 Cacarot release** — DONE (2026-03-16). Temporal decay + 4320-row cap + DF mode.
- [x] **Cacarot release (V5)** — DONE (2026-03-16). Temporal decay, 6mo data cap, Mode F charts.
- [x] **BTC + ETH Cacarot DF** — DONE (2026-03-16). BTC: `either_agree` @90%. ETH: `4h_only` @60%.
- [x] **V5.5 A/B tests** — COMPLETE, archived. Only slippage_model won → promoted.
- [x] **V5.6 feature tests** — COMPLETE, archived. Only ADX + Garman-Klass useful → added to production.
- [x] V5 scoring, Mode F, DF mode, per-step timers, hardware_config lightened
- [x] BTC V5 2y run — 4h: 80.2% +125%, 8h: 84.7% +319%
- [x] ETH V5 2y run — 4h: RF+LGBM 68.6% +505%, 8h: GB 79.3% +616%
- [x] IB auto-trader for DAX + S&P 500 CFDs, Broly 1.2
- [x] Multi-asset crypto auto-trader with Telegram control

---

*Last updated: March 18, 2026 — Deku release: Optuna TPE+Hyperband replaces CASCA grid search, XGBoost (5th model), APF scoring, BTC 4h +54.2% (2× CASCA). Deku V15 (15-min candles). Deku fusion test harness. CASCA V1.4 baseline.*
