# Algo Trading Engine

Automated ML trading system for **crypto** (BTC, ETH, XRP, DOGE, SOL, LINK, ADA, AVAX, DOT) and **index CFDs** (DAX, S&P 500). Generates hourly BUY/SELL/HOLD signals using dual-horizon ensemble ML with walk-forward validation and temporal decay sample weighting. Executes trades via exchange API.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Trading Systems](#trading-systems)
- [Features (130)](#features-130)
- [Strategies](#strategies)
- [Auto-Trader](#auto-trader)
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

# Train models (Mode D) then pick best strategy (Mode F)
python crypto_trading_system_deku.py DF BTC,ETH 4,8h       # per-asset: D+F for BTC, then D+F for ETH

# Start live trading
python crypto_revolut_deku.py --loop

# Dry run (signals only, no trades)
python crypto_revolut_deku.py --dry-run --loop

# Check positions / balance
python crypto_revolut_deku.py --status
python crypto_revolut_deku.py --balance
```

### Other Commands

```bash
# Deku with custom trials or scoring metric
python crypto_trading_system_deku.py D BTC 4,8h --trials 150
python crypto_trading_system_deku.py DF BTC 4,8h --metric calmar
python crypto_trading_system_deku.py DF BTC 4,8h --metric all     # compare 5 metrics

# CASCA (standby system — profit factor scoring)
python crypto_trading_system_casca.py DF BTC,ETH 4,8h
python crypto_trading_system_casca.py A BTC 4,8h                  # gamma optimization

# Sub-hourly (15-min / 30-min candles)
python crypto_trading_system_deku_15m.py DF BTC,ETH 4,8h
python crypto_trading_system_v15.py D BTC 4,8h
python crypto_trading_system_v30.py D BTC 4,8h

# Index CFD trader
python cfd/ib_auto_trader.py --loop          # DAX
python cfd/ib_auto_trader_test.py --loop     # S&P 500
```

---

## Architecture

```
DEKU (production)                           CASCA (standby)
─────────────────                           ───────────────
crypto_trading_system_deku.py               crypto_trading_system_casca.py
  ↓ imports                                   ↓ imports
crypto_live_trader_deku.py                  crypto_live_trader.py
  ↓ imports                                   ↓ imports
crypto_revolut_deku.py                      crypto_revolut_trader.py
  ↓ trades via                                ↓ trades via
Exchange API (Ed25519 signed)               Exchange API (Ed25519 signed)
```

**Deku** uses Optuna Bayesian optimization (TPE + Hyperband). **CASCA** uses grid search with profit factor scoring. Both share the same ML pipeline, features, and walk-forward validation. The systems are independent — separate models, configs, and traders.

**Index CFDs** (in `cfd/`) are completely separate: different models, assets, broker, and config.

---

## How It Works

### Core Concepts

| Concept | Detail |
|---------|--------|
| **Dual horizons** | 4h and 8h — independent models per horizon (different features, window, combo). Never mixed. |
| **No model persistence** | Retrains from scratch every prediction. No .pkl files. Always uses latest market data. |
| **Temporal decay** | Exponential sample weighting `w_i = gamma^(age)`. Recent data matters more. gamma=0.996 → half-life ~7 days. |
| **6-month data cap** | Training capped at 4,320 hours. Prevents stale data from diluting recent patterns. |
| **Fee-aware labels** | `label = 1` when future return > 2×TRADING_FEE (0.22%). Only labels profitable moves as positive. Fallback: 200h rolling median. |
| **Walk-forward validation** | Train on `window` hours → predict next candle → step forward. No future leakage. |
| **Ensemble voting** | Majority vote across model combo. Confidence = average probability across models. |

### ML Pipeline (Mode D)

1. **Data download** — Hourly candles (Binance via ccxt) + macro data (yfinance)
2. **Feature engineering** — 130 features (technical + macro + sentiment + cross-asset)
3. **Feature ranking** — LGBM gain importance ranks all features (~5 sec)
4. **Optuna optimization** — Joint search over (model combo, window, gamma, n_features) using TPE + Hyperband pruning. Default 150 trials, auto-extends to 200/250 if APF < 1.7
5. **Walk-forward evaluation** — Train → predict → step forward (DIAG_STEP=36h). Score by APF
6. **Signal generation** — Best config generates signals with bootstrap confidence intervals
7. **Charts** — Backtest PNG + interactive HTML strategy charts
8. **Save** — Best model written to `models/crypto_deku_best_models.csv`

### Modes

| Mode | Purpose | When to Use |
|------|---------|-------------|
| **D** | Full optimization pipeline | After market regime change, or periodically |
| **F** | Strategy comparison + confidence sweep → writes `trading_config.json` | After Mode D |
| **DF** | D then F in one command | Standard workflow |
| **B** | Quick signals from saved models | Daily check |
| **A** | Gamma optimization (6 values × horizons) | CASCA only — tune temporal decay |
| **E** | Iterative refinement | CASCA only — after Mode D |

---

## Trading Systems

### Deku (Production)

Optuna Bayesian optimization with TPE + Hyperband pruning.

| Parameter | Value |
|-----------|-------|
| **Models** | RF, GB, XGB, LR, LGBM — 26 ensemble combos (pairs + triples + quads + quint) |
| **Optimizer** | Optuna TPE, Hyperband pruning (~60% trials pruned) |
| **Search space** | 7 windows [24–200], gamma [0.994–1.0], n_features [4–80] |
| **Scoring** | APF = raw_PF / buyhold_PF (regime-normalized). Alternatives via `--metric` flag |
| **Walk-forward step** | 36h (doubled eval points vs CASCA) |
| **Min trades** | 8 (reject unreliable configs) |
| **3-fold holdout** | Diversity-aware: top 10 + best per unexplored combo → up to 20 candidates |
| **Auto-extend** | If best APF < 1.7 after 150 trials → extend to 200 → 250 |
| **Enhancements** | Optional `--enhancements` flag: return weighting, disagreement filter, funding gate (Optuna toggles) |
| **DF mode** | Per-asset pipeline: D (all horizons) + F for each asset before moving to next |
| **Output** | `models/crypto_deku_best_models.csv`, `config/trading_config_deku.json` |

**Scoring metrics** (`--metric` flag): `apf` (default), `rawpf`, `calmar`, `return`, `rpf_sqrt`. Use `--metric all` to compare all 5.

### CASCA (Standby)

Grid search with profit factor scoring. Can be reactivated without code changes.

| Parameter | Value |
|-----------|-------|
| **Models** | RF, GB, LR, LGBM — 11 ensemble combos |
| **Optimizer** | Grid search: 11 combos × 5 windows = 55 configs |
| **Feature selection** | 5-test analysis: LGBM importance, permutation, ablation, reduced sets, consensus |
| **Scoring** | Profit Factor = gross_profit / \|gross_loss\| (cap 5.0, min 3 trades) |
| **Walk-forward step** | 72h |
| **Output** | `models/crypto_casca_best_models.csv`, `config/trading_config.json` |

### Sub-Hourly (V15 / V30)

Same ML pipeline with shorter candles for higher trade frequency.

| System | Candle | Horizons | Data Cap | File |
|--------|--------|----------|----------|------|
| Deku V15 | 15 min | s4=60', s8=120' | ~45 days | `crypto_trading_system_deku_15m.py` |
| V15 Cacarot | 15 min | 15'–120' | ~45 days | `crypto_trading_system_v15.py` |
| V30 Cacarot | 30 min | 30'–240' | ~3 months | `crypto_trading_system_v30.py` |

### Index CFDs

Independent system using Broly 1.2 ML model via broker API.

| Asset | File | Hours (UTC) |
|-------|------|-------------|
| DAX | `cfd/ib_auto_trader.py` | Mon–Fri 07:00–16:00 |
| S&P 500 | `cfd/ib_auto_trader_test.py` | Sun 23:00 – Fri 22:00 |

Risk controls: 2% stop-loss, €2,000 daily loss limit, 2h cooldown after stop-loss.

---

## Features (130)

| Category | Count | Examples |
|----------|-------|---------|
| **Technical** | 49 | Log returns (1–240h), RSI, Bollinger Bands, ATR, ADX/DI, Garman-Klass vol, volatility ratios, Stochastic, spread ratios, SMA ratios, hour sin/cos |
| **Macro** | 40 | VIX (level, zscore, regime), DXY, S&P500/Nasdaq changes (1/5/10d), US10Y, EUR/USD, USD/JPY, Oil, Gold volatility |
| **Sentiment** | 25 | Fear & Greed Index (value, zscore, changes, MA, extreme flags) |
| **Cross-asset** | 16 | BTC/ETH/DAX/Nasdaq/S&P500 rolling correlation (10/30d), relative strength (5d) |

---

## Strategies

Set per asset by Mode F, stored in trading config:

| Strategy | BUY Condition | SELL Condition |
|----------|--------------|----------------|
| `both_agree` | 4h AND 8h both say BUY (≥ min_confidence) | Either says SELL |
| `either_agree` | Either 4h or 8h says BUY | Either says SELL |
| `4h_only` | 4h says BUY | 4h says SELL |
| `8h_only` | 8h says BUY | 8h says SELL |

Mode F backtests all strategies with a confidence sweep (60–90%) and writes the winner to `trading_config_deku.json`.

---

## Auto-Trader

### Loop Cycle

1. **Startup** — Sync positions with exchange → Telegram notification → immediate scan
2. **Every hour** — Download data → generate 4h+8h signals → apply strategy → execute trades → Telegram
3. **Every 5 min** — Position sync (detects manual trades), model hot-reload from CSV
4. **Every 5 sec** — Poll Telegram for commands

### Telegram Commands

| Command | Action |
|---------|--------|
| `/status` | Current positions and PnL |
| `/balance` | Exchange account balance |
| `/stop` | Graceful shutdown |
| `/pause` / `/resume` | Pause/resume trading |
| `/sync` | Force position sync |
| `/conf` / `/config` | Show current config |
| `/setup` | Inline button config editor (asset picker, toggle, strategy, confidence, max position) |
| `/help` | List all commands |
| `/chart BTC` | Generate and send chart |
| `/optimize BTC` | Launch Mode D in background (Deku only) |
| `/optstatus` | Check optimization progress (Deku only) |

### Authentication

Asymmetric Ed25519 signature: `timestamp + method + path + body → base64 signature`.

---

## Current Models

### Deku (Production — 3-fold holdout validation)

| Asset | Horizon | Models | Window | Return | APF | Features | Gamma | Trades |
|-------|---------|--------|--------|--------|-----|----------|-------|--------|
| BTC | 4h | GB+XGB+LR | 150h | +1.9% | 1.82 | 4 | 0.9962 | 16 |
| BTC | 8h | XGB+LR+LGBM | 200h | +11.3% | 1.47 | 13 | 0.9956 | 22 |
| ETH | 4h | RF+GB+XGB+LGBM | 36h | +4.6% | 4.16 | 5 | 0.9944 | 14 |
| ETH | 8h | RF+GB+XGB+LGBM | 48h | +6.5% | 2.41 | 21 | 0.9957 | 14 |
| XRP | 4h | RF+XGB+LR+LGBM | 36h | +0.5% | 2.49 | 34 | 0.9941 | 8 |
| XRP | 8h | GB+LR+LGBM | 150h | +7.6% | 2.29 | 73 | 0.9971 | 16 |
| DOGE | 4h | LR+LGBM | 150h | +3.2% | 2.15 | 27 | 0.9962 | 14 |
| DOGE | 8h | GB+LR+LGBM | 150h | +6.4% | 2.22 | 73 | 0.9971 | 14 |
| SOL | 4h | RF+XGB+LR+LGBM | 72h | -0.7% | 1.98 | 13 | 0.9972 | 12 |
| SOL | 8h | RF+GB+XGB | 48h | +5.4% | 2.18 | 17 | 0.9971 | 17 |
| LINK | 4h | XGB+LGBM | 150h | +5.3% | 2.15 | 6 | 0.9972 | 17 |
| LINK | 8h | RF+GB+LR+LGBM | 200h | +23.5% | 1.71 | 20 | 0.9963 | 31 |
| ADA | 4h | RF+GB+XGB+LR+LGBM | 150h | +3.6% | 1.78 | 24 | 0.9967 | 14 |
| ADA | 8h | RF+GB+XGB+LGBM | 24h | +21.3% | 7.94 | 26 | 0.9996 | 8 |
| AVAX | 4h | RF+XGB+LR+LGBM | 100h | +6.5% | 1.97 | 20 | 1.0 | 11 |
| AVAX | 8h | XGB+LR | 150h | +19.6% | 4.03 | 52 | 0.9944 | 18 |
| DOT | 4h | RF+XGB+LGBM | 100h | +12.5% | 2.83 | 27 | 0.9954 | 16 |
| DOT | 8h | XGB+LR | 150h | +9.3% | 8.47 | 41 | 0.999 | 17 |

### Trading Config

```
BTC:  8h_only      @70%  ($1,000 max)  enabled
ETH:  8h_only      @70%  ($1,000 max)  disabled
XRP:  either_agree @60%                 disabled
DOGE: 8h_only      @75%                 disabled
SOL:  4h_only      @75%                 disabled
LINK: 8h_only      @75%  ($1,000 max)  enabled
ADA:  4h_only      @75%                 disabled
AVAX: 4h_only      @60%                 disabled
DOT:  8h_only      @80%                 disabled
```

---

## Project Structure

```
engine/
├── crypto_trading_system_deku.py        # Deku production — Optuna, Modes D/F/DF/B
├── crypto_trading_system_deku_15m.py    # Deku V15 — 15-min candles
├── crypto_trading_system_casca.py       # CASCA standby — PF scoring, Modes A/B/D/E/F/DF
├── crypto_trading_system_v15.py         # V15 Cacarot — 15-min, grid search
├── crypto_trading_system_v30.py         # V30 Cacarot — 30-min, grid search
├── crypto_trading_system_v6.py          # V5.4 legacy — enhancement test bed
│
├── crypto_revolut_deku.py               # Deku live trader
├── crypto_revolut_trader.py             # CASCA live trader
├── crypto_live_trader_deku.py           # Deku signal generation (library)
├── crypto_live_trader.py                # CASCA signal generation (library)
│
├── hardware_config.py                   # Auto-detect Desktop/Laptop config
├── download_macro_data.py               # Macro/sentiment/cross-asset downloader
│
├── testing_deku_fusion.py               # Deku 1h+15' fusion test (17 strategies)
├── testing_casca.py                     # CASCA 1h+15' fusion test (16 strategies)
├── testing_literature_v2.py             # V6 A/B test harness
├── testing_v15.1.py                     # V15 gamma optimization
├── testing_v30.1.py                     # V30 gamma optimization
│
├── cfd/
│   ├── ib_auto_trader.py                # DAX CFD trader (Broly 1.2)
│   ├── ib_auto_trader_test.py           # S&P 500 CFD overnight
│   └── broly.py                         # Enhancement layer (regime detection)
│
├── tools/
│   ├── check_balance.py                 # Exchange balance
│   ├── check_trades.py                  # Trade history
│   ├── debug_price.py                   # API price diagnostic
│   ├── revolut_x_test.py               # API connectivity test
│   ├── detect_hardware.py               # Hardware detection → config
│   ├── buy_btc.py                       # Manual BTC purchase
│   ├── backtest_v5_v15.py              # V5 vs V15 backtest comparison
│   └── ib_test_connection.py            # IB broker connectivity test
│
├── data/
│   ├── {asset}_hourly_data.csv          # Hourly OHLCV (Binance)
│   ├── {asset}_15m_data.csv             # 15-min OHLCV
│   ├── {asset}_30m_data.csv             # 30-min OHLCV
│   ├── macro_data/                      # VIX, DXY, S&P500, Fear&Greed, etc.
│   └── indices/                         # DAX, S&P500, SMI, CAC40 OHLCV
│
├── models/
│   ├── crypto_deku_best_models.csv      # Deku production models
│   ├── crypto_casca_best_models.csv     # CASCA standby models
│   ├── crypto_deku_15m_best_models.csv  # Deku V15 models
│   └── crypto_feature_analysis_*.csv    # Feature scores
│
├── config/                              # NOT in git — credentials + state
│   ├── trading_config_deku.json         # Deku per-asset strategy + confidence
│   ├── trading_config.json              # CASCA per-asset strategy
│   ├── revolut_x_config.json            # Exchange API key
│   ├── private.pem                      # Ed25519 signing key
│   ├── telegram_config.json             # Bot token
│   └── position_*.json                  # Position state per asset
│
├── charts/                              # Backtest PNGs + interactive HTML
├── archive/                             # Superseded versions (V3–V5.6)
├── CLAUDE.md                            # Claude Code instructions
└── README.md
```

---

## Setup

### Hardware

One shared engine folder synced via Google Drive. Only the venv is local per machine.

| Machine | Engine Path | Venv | CPU | GPU |
|---------|-------------|------|-----|-----|
| Desktop (primary) | `G:\engine\` | `C:\algo_trading\venv\` | i7-14700KF | RTX 4080 |
| Laptop | `G:\Autres ordinateurs\My laptop\engine\` | `C:\Users\Alex\algo_trading\venv\` | 16 cores | RTX 3070 Ti |

`hardware_config.py` auto-detects Desktop (26 workers) vs Laptop (14 workers).

### Installation

```powershell
# Desktop
python -m venv C:\algo_trading\venv
C:\algo_trading\venv\Scripts\activate.bat
pip install -r G:\engine\requirements.txt
python tools/detect_hardware.py

# Laptop
python -m venv C:\Users\Alex\algo_trading\venv
C:\Users\Alex\algo_trading\venv\Scripts\activate.bat
pip install -r "G:\Autres ordinateurs\My laptop\engine\requirements.txt"
python tools/detect_hardware.py
```

### Key Dependencies

`pandas`, `numpy`, `scikit-learn`, `lightgbm` (GPU), `xgboost`, `optuna`, `ccxt`, `yfinance`, `pynacl`, `cryptography`, `matplotlib`, `joblib`, `PyWavelets`

---

## Key Constants

```python
# Trading costs
TRADING_FEE_BASE = 0.0009       # 0.09% taker fee
SLIPPAGE = 0.0002               # 0.02% estimated
TRADING_FEE = 0.0011            # total per trade

# Horizons
HORIZON_SHORT = 4               # short prediction horizon
HORIZON_LONG = 8                # long prediction horizon

# Deku
DIAG_STEP = 36                  # walk-forward step (2× CASCA)
DIAG_WINDOWS = [24, 36, 48, 72, 100, 150, 200]
GAMMA_RANGE = [0.994, 1.0]      # continuous gamma search
MIN_TRADES = 8                  # reject low-trade trials
DEKU_DEFAULT_TRIALS = 150       # auto-extends to 200 then 250 if APF < 1.7
APF_EXTEND_THRESH = 1.7         # extension threshold

# CASCA
DIAG_STEP = 72                  # walk-forward step
DIAG_WINDOWS = [48, 72, 100, 150, 200]

# Shared
MIN_CONFIDENCE = 75             # fallback — overridden per asset by Mode F
MAX_DIAG_HOURS = 4320           # 6-month data cap
REPLAY_HOURS = 200              # Mode B/D signal replay
REPLAY_HOURS_F = 400            # Mode F — longer for more trades
```

---

## Pending Work

### Active
1. **V1.5 — Dynamic data cap + holdout comparison.** Tests 3 holdout strategies with gamma-aware data sizing. Dynamic cap = `log(0.01)/log(gamma)` hours. BTC 8h only.
   - `python crypto_trading_system_deku_v1_5.py D BTC 8h --holdout all --trials 150`
   - Holdout modes: `current` (overlapping baseline), `A` (non-overlapping sequential), `B` (expanding window)

### CPCV Investigation Summary (2026-03-22) — Completed

CPCV tested via V1.3.1 (gamma search) and V1.4 (gamma=1.0). Key findings:
- **4h models overfit everywhere** — PBO=1.0 across all assets. 4h returns are noise.
- **8h models are reliable** — PBO 0.0–0.33. Real alpha confirmed by 1-week live backtest.
- **Gamma is essential** — V1.4 (gamma=1.0) killed returns: BTC +4.4% vs +7.4%, LINK 0 signals. Gamma genuinely helps recency-weight crypto data.
- **CPCV dropped as validation** — With gamma, CPCV folds leak temporal info. Without gamma, returns collapse. 3-fold holdout kept as production validation.
- **LR+LGBM core required** — All CPCV-passing configs contain both. LR anchors linearly, LGBM captures nonlinear interactions.

### Completed
- **CPCV investigation** — DONE (2026-03-22). V1.3.1 + V1.4 tested. Gamma=1.0 failed (BTC +4.4% vs +7.4%, LINK 0 signals). CPCV dropped. Finding: 4h overfits (PBO=1.0), 8h reliable, LR+LGBM core required.
- **V1.4 gamma=1.0 + CPCV** — FAILED (2026-03-22). Archived as `_cpcv_gamma1_failed`. Gamma is essential for crypto recency.
- **1-week live backtest** — DONE (2026-03-22). Compared V1.3 Prod vs V1.3.1-A vs V1.4 on BTC+LINK 8h. Gamma models outperform: BTC +7.4%, LINK +12.1% in -4.6% market.
- **Telegram UX overhaul** — DONE (2026-03-22). Candlestick charts (48h, signal transitions only, blue/red colorblind scheme). Inline button config editor replaces text `/setup`. Max position presets ($0/$1K/$5K/$10K + custom). All green indicators changed to blue. Fixed last 4h prices bug.
- **V1.3.1 CPCV A/B/C test** — DONE (2026-03-22). All 9 runs. Mode C dropped (consistently weakest).
- **Enhancement A/B test** — Dropped (2026-03-22). Both enhancement sets failed. Code stripped from production deku.py.
- **Weekly F re-runs** — DONE (2026-03-21). Re-ran Deku F for all active assets.

---

## Version History

| Date | Milestone |
|------|-----------|
| **2026-03-22** | **CPCV investigation concluded.** V1.3.1 CPCV + V1.4 gamma=1.0 tested. CPCV dropped — gamma essential, 4h unreliable. V1.5 started (dynamic data cap + holdout comparison). Telegram UX overhaul (candlestick charts, inline buttons, colorblind blue scheme). |
| **2026-03-21** | **Multi-asset expansion.** Added SOL, LINK, ADA, AVAX, DOT. All 9 assets optimized. Per-asset DF pipeline. 150 default trials. Diversity-aware holdout. Optional enhancements. BTC+LINK activated for live trading. Top: LINK 8h +16.4%, ADA 8h +21.3%, AVAX 8h +19.6%. |
| **2026-03-20** | **Deku promoted to production.** 3-fold holdout validation. Enhancements tested and rejected. Auto-extend trials. |
| **2026-03-19** | Deku tuning: DIAG_STEP=36, expanded search space, `--metric` flag. Deku trader + `/optimize` Telegram. ETH DF: 4h +79.8%, 8h +77.1%. |
| **2026-03-18** | **Deku release.** Optuna TPE+Hyperband, XGBoost, APF scoring, LGBM importance. BTC 4h: +54.2% (2× CASCA). Deku V15 + fusion test. |
| **2026-03-17** | CASCA DF BTC (RF+GB 4h PF=5.0). MIN_COMBO_SIZE=2. V1.3.1 (per-model features — dead end). |
| **2026-03-15–16** | **CASCA release.** PF scoring replaces acc×ret. V5 archived. V15/V30 Cacarot. Temporal decay (Cacarot). |
| **2026-03-10** | **V5 production.** acc×(1+ret/100) scoring. Mode F. BTC 2y: 4h 80.2%, 8h 84.7%. |
| **2026-03-08** | **V4.** Bootstrap CI, Calmar/Sharpe scoring. Mock validation. |
| **2026-03-07** | 8h horizon BTC 80.3%. "Both Agree" strategy. Crypto auto-trader. |
| **2026-03-04** | Dual horizon (4h+8h). Mode E. Derivative features. |
| **2026-03-02** | Macro features (VIX, DXY, S&P500). Modes A/B/C. |
| **2026-02-22** | **Broly 1.2.** IB auto-trader for DAX CFDs. |
| **2026-02-21** | Initial commit. |

### Evolution: Deku > CASCA > V5 > V4 > V3

- **Deku** — Optuna Bayesian optimization, XGBoost (5th model), APF scoring, continuous gamma, LGBM importance ranking. Finds better configs in similar time.
- **CASCA** — Profit factor scoring. Fixed V5's broken formula but limited by grid search.
- **V5** — `acc × (1 + ret/100)` scoring. Picked money-losing models because accuracy dominated.
- **V4** — Calmar/Sharpe scoring. Biased toward low-trade configs.
