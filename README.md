# Crypto Hourly ML Trading System — v3

Automated machine learning trading system for crypto (BTC, ETH, XRP). Generates hourly BUY/SELL/HOLD signals using dual-horizon (4h + 8h) ensemble ML models with walk-forward validation. Executes trades automatically on Revolut X via Ed25519-signed API.

**Active assets:** BTC, ETH, XRP (SOL removed — poor results)
**Horizons:** 4h (short-term) and 8h (medium-term) — separate models, features, and windows per horizon
**Strategies:** BTC = "both_agree" | ETH = "either" | XRP = pending test

---

## FOR NEW CLAUDE CHAT — START HERE

If you are Claude reading this in a new conversation, this README is your complete reference. Read it fully before making any code changes.

### What to Upload for a New Chat

Alex should upload these files when starting a new conversation:
1. `README.md` (this file — the master reference)
2. `crypto_trading_system.py` (main system with shortcuts 5/6/7, ~2800+ lines)
3. `crypto_revolut_trader.py` (auto-trader with Revolut X integration)
4. `crypto_strategy_test.py` (strategy backtester — tests 1-3)
5. `crypto_horizon_test.py` (horizon model tester)
6. `mock_strategy_optimizer.py` (mock combined optimizer — DO NOT deploy to production)
7. `models/crypto_hourly_best_models.csv` (current best models)
8. Optionally: any backtest chart PNGs, HTML charts, or signal tables for analysis

### Key Rules for Editing This Codebase

1. **Horizons are 4h and 8h** (not 1h + 2h). AVAILABLE_HORIZONS = [4, 8]
2. **CSV merge logic is critical.** When saving to crypto_hourly_best_models.csv, always filter by BOTH coin AND horizon. Wrong filter = overwrite the other horizon's config
3. **feature_set check must include D.** All current models use Set D with custom optimal_features
4. **generate_signals() needs feature_override.** When feature_set is D, the caller must pass `feature_override=config['optimal_features'].split(',')`. Without this, it defaults to Set A
5. **Live trader imports from crypto_trading_system.py.** The filename must be exact
6. **Labels are relative to rolling median, not absolute.** `label = 1` means "return in next H hours is above the 168h rolling median return", not just "price goes up"
7. **No model persistence.** The system retrains from scratch every prediction. No .pkl files. This is intentional
8. **Each horizon = completely independent model.** Different features, model combo, window, labels. Never mix them
9. **All timestamps in live trader use Europe/Zurich** via `_to_local()` helper
10. **Folder structure is v3.** Data in `data/`, models in `models/`, charts in `charts/`, config in `config/`. Root has .py files only

---

## Alex's Setup — Two Machines

### Desktop (Primary — ~2x faster than laptop)
```
CPU:     Intel i7-14700KF (20P/28L cores)
GPU:     NVIDIA RTX 4080 16GB, CUDA 13.1
RAM:     32 GB
Path:    C:\algo_trading
Venv:    C:\algo_trading\venv
Activate: .\venv\Scripts\Activate.ps1
```

### Laptop
```
CPU:     16 logical cores
GPU:     NVIDIA RTX 3070 Ti Laptop
Path:    C:\Users\Alex\algo_trading\Algo_trading
Venv:    C:\Users\Alex\algo_trading\venv
Activate: .\venv\Scripts\Activate.ps1
```

### Shared Config
```
OS:       Windows 11
Python:   venv (NOT conda)
LGBM:     GPU-enabled (device='gpu')
Broker:   Revolut X (0.09% taker fee) for crypto
User:     Alex, Lausanne, Switzerland (CET/CEST timezone)
```

### Key Python Packages
numpy 2.4.2, pandas 3.0.1, scipy 1.17.0, scikit-learn 1.8.0, lightgbm 4.6.0 (GPU), ccxt, yfinance, plotly, pynacl, cryptography, joblib, matplotlib

---

## Complete Folder Structure (v3)

```
Algo_trading/
│
├── ===== CORE SYSTEM (root .py files) =====
├── crypto_trading_system.py           # Main system (~2800+ lines) — Modes A/B/C/D/E + shortcuts 5/6/7
├── crypto_revolut_trader.py           # Multi-asset Revolut X auto-trader with position management
├── crypto_live_trader.py              # Telegram-only live trader (Both Agree strategy)
├── crypto_strategy_test.py            # Strategy backtester (tests 1-3)
├── crypto_horizon_test.py             # Horizon model tester
├── mock_strategy_optimizer.py         # Mock combined optimizer (DO NOT deploy to production)
├── migrate_folders.py                 # One-time v2→v3 migration script
│
├── ===== INFRASTRUCTURE =====
├── hardware_config.py                 # AUTO-GENERATED — DO NOT EDIT
├── detect_hardware.py                 # Run once per machine
├── download_macro_data.py             # Downloads VIX, DXY, S&P500, Gold, Fear&Greed → data/macro_data/
│
├── ===== DATA (auto-downloaded) =====
├── data/
│   ├── btc_hourly_data.csv            # BTC/USDT hourly OHLCV (~75K candles since 2017)
│   ├── eth_hourly_data.csv            # ETH/USDT hourly OHLCV
│   ├── xrp_hourly_data.csv            # XRP/USDT hourly OHLCV
│   └── macro_data/
│       ├── macro_daily.csv            # VIX, DXY, S&P500, Nasdaq, Gold, US10Y, EUR/USD, USD/JPY, Oil
│       ├── fear_greed.csv             # Crypto Fear & Greed Index
│       ├── cross_asset.csv            # DAX, SMI, CAC40 daily
│       └── macro_hourly.csv           # Hourly interpolations (auto-generated)
│
├── ===== MODELS =====
├── models/
│   ├── crypto_hourly_best_models.csv  # ** CENTRAL CONFIG ** Best model per (asset, horizon)
│   ├── crypto_feature_analysis_*.csv  # Feature analysis scores per asset
│   └── crypto_hourly_chart_data.json  # Signal export for dashboards
│
├── ===== CHARTS (generated outputs) =====
├── charts/
│   ├── {ASSET}_strategy_1week.html    # Interactive 4-panel Plotly chart (168h)
│   ├── {ASSET}_strategy_1month.html   # Interactive 4-panel Plotly chart (720h)
│   ├── {ASSET}_signal_table.html      # Signal table with correctness tracking
│   └── {ASSET}_backtest.png           # Static backtest chart (matplotlib)
│
├── ===== CONFIG =====
├── config/
│   ├── telegram_config.json           # Telegram bot token + chat_id
│   ├── revolut_x_config.json          # Revolut X API key ID + key name
│   ├── private.pem                    # Ed25519 private key for API signing
│   ├── trading_config.json            # Per-asset strategy + max USD
│   ├── position_BTC.json              # Current BTC position state
│   ├── position_ETH.json              # Current ETH position state
│   └── position_XRP.json              # Current XRP position state
│
├── ===== INDICES (separate, not actively developed) =====
├── broly.py                           # Regime detection for indices (DAX)
├── ib_auto_trader.py                  # Interactive Brokers auto-trader
└── ib_auto_trader_test.py             # IB trader test suite
```

---

## Key Constants

```python
TRADING_FEE = 0.0009                # 0.09% per trade (Revolut X taker fee)
MIN_CONFIDENCE = 75                 # Model must be ≥75% sure to trigger BUY
MIN_TRADE_USD = 300                 # Won't execute trades below $300
MIN_POSITION_USD = 5                # Threshold for "has position"
AVAILABLE_HORIZONS = [4, 8]         # 4h and 8h models
DATA_DIR = 'data'
CHARTS_DIR = 'charts'
MODELS_DIR = 'models'
CONFIG_DIR = 'config'
```

---

## Current Best Models (models/crypto_hourly_best_models.csv)

| Asset | Horizon | Models | Window | Feature Set | Accuracy | Features |
|-------|---------|--------|--------|-------------|----------|----------|
| BTC | 4h | RF+LR+LGBM | 100h | D | 72.5% | 18 features |
| BTC | 8h | RF+GB+LGBM | 300h | D | 86.3% | 20 features |
| ETH | 4h | RF+LR | 100h | D | 78.3% | custom |
| ETH | 8h | RF+LR | 100h | D | 75.0% | custom |
| XRP | 4h | GB | 100h | D | 69.2% | custom |
| XRP | 8h | RF+LR | 100h | D | 80.8% | custom |

### BTC 8h Feature Set (20 features)
logret_240h, logret_120h, logret_72h, hour_cos, atr_pct_14h, xa_sp500_relstr5d, vol_ratio_12_48, volatility_48h, xa_dax_relstr5d, price_to_sma100h, price_accel_24h, sma20_to_sma50h, hour_sin, spread_120h_8h, volatility_12h, logret_48h, spread_240h_24h, m_nasdaq_chg1d, price_accel_12h, xa_nasdaq_relstr5d

---

## Strategy Results — Confirmed Optimal

| Asset | Strategy | Alpha | Win Rate | Trades/Month |
|-------|----------|-------|----------|--------------|
| BTC | both_agree | +97.2% | 90.0% | 40 |
| ETH | either | +107.8% | 78.9% | 57 |
| XRP | pending test | — | — | — |

**Weighted strategy:** Tested and REJECTED — both_agree/either already optimal.

### Strategy Rules
- **both_agree**: Both 4h AND 8h must agree on BUY/SELL to trigger
- **either**: If EITHER 4h OR 8h signals BUY/SELL, trigger (more trades, lower precision)

### Trading Rules
- **BUY**: Uses full max_position_usd, or all available USD if less
- **SELL**: Sells ALL held on exchange
- **Already invested**: BUY is ignored (no top-up)
- **MIN_CONFIDENCE = 75%**: Model must be ≥75% confident to trigger BUY

---

## Trading Config (config/trading_config.json)

```json
{
  "BTC": { "strategy": "both_agree", "symbol": "BTC-USD", "max_position_usd": 10000 },
  "ETH": { "strategy": "either", "symbol": "ETH-USD", "max_position_usd": 1000 }
}
```

---

## Interactive Charts — 4-Panel Layout

Generated by shortcuts 5/6/7, these are interactive Plotly HTML charts with zoom-synced panels:

**Panel 1:** Price + 4h signals (blue triangles = BUY, red triangles = SELL)
**Panel 2:** Price + 8h signals
**Panel 3:** Price + combined strategy signals (both_agree or either)
**Panel 4:** Portfolio ($1,000 start) vs Buy & Hold

Color scheme (colorblind-friendly):
- 🔵 Blue = BUY
- 🔴 Red = SELL
- 🟡 Yellow = HOLD

Files generated per asset:
- `charts/{ASSET}_strategy_1week.html` (168h window)
- `charts/{ASSET}_strategy_1month.html` (720h window)

### Signal Table (charts/{ASSET}_signal_table.html)

| Time | Price | Price+1h | Δ1h | Strategy | Correct? | 4h Signal | 4h Conf | 8h Signal | 8h Conf |
|------|-------|----------|-----|----------|----------|-----------|---------|-----------|---------|

Green row = correct prediction, Red row = wrong. Shows per-hour signals with both horizons and combined strategy.

---

## CLI Shortcuts

```powershell
python crypto_trading_system.py 5    # Quick BTC (charts + signal table)
python crypto_trading_system.py 6    # Quick ETH
python crypto_trading_system.py 7    # Quick XRP
```

Each shortcut generates: 4-panel 1-week HTML, 4-panel 1-month HTML, signal table HTML, and static backtest PNG.

---

## Revolut X Auto-Trader (crypto_revolut_trader.py)

### API Details
- Base URL: `https://revx.revolut.com/api/1.0`
- Auth: Ed25519 signing via pynacl + cryptography
- Price fetch: 3 fallbacks — public orderbook (`data.asks[0].p`) → authenticated tickers → last trades

### Behavior
1. **Startup**: Sync positions from exchange → send startup Telegram → immediate full scan
2. **Every hour**: Sync → download candle → generate 4h+8h signals → apply strategy → execute trade → Telegram
3. **Between hours (every 5 min)**: Check Telegram commands + sync balances

### Hourly Telegram Message Format
Per asset: 🔵/🔴 indicator, price, RSI, last 4 candle sparkline, 4h+8h signals with confidence, PnL if holding, exchange balance summary.

In Telegram: 🔵 before asset name = auto-trade ON, 🔴 = auto-trade OFF

### Telegram Commands
`/stop` `/status` `/pause` `/resume` `/balance` `/sync`

---

## Revolut X Account Status (as of Mar 8, 2026)

- USD: $12,697.53
- BTC: 0.001403 (~$124)
- ETH: 0.505 (~$1,160)

---

## Modes Quick Reference

| Mode | Purpose | Time | When to Run |
|------|---------|------|-------------|
| **A** | Tests Set A vs Set B, picks winner | ~3-6h | Monthly (4h only) |
| **B** | Quick signals from saved models | ~2 min | Daily |
| **C** | Set A vs Set B comparison only | ~3-6h | Ad hoc |
| **D** | Full pipeline: 124 features → analysis → optimal subset → diagnostic | ~3-8h | Quarterly |
| **E** | Iterative refinement of Mode D | ~1-4h | After Mode D |

---

## System Architecture

### Features (~124 total)
43 Technical + ~80 Macro (VIX, DXY, S&P500, Nasdaq, Gold, US10Y, Fear&Greed, cross-asset correlations)

### Labels
```python
median_return = df['logret_Xh'].rolling(168).median()
label = 1 if logret_Xh > median_return else 0
```
Adaptive threshold — not just "price goes up".

### Models
RF (RandomForest), GB (GradientBoosting), LR (LogisticRegression), LGBM (LightGBM GPU)
All 15 combinations tested: 4 solo + 6 pairs + 4 triples + 1 quad.

### Walk-Forward Validation
No future leakage. Train on last `window` hours → predict next candle → step forward. Retrains every hour in live mode.

### Scoring
`combined_score = accuracy × profit_factor` — must be both accurate AND profitable after 0.09% fees.

---

## Mock Strategy Optimizer (mock_strategy_optimizer.py)

**Purpose:** Tests 1,400 combinations to find optimal model+strategy+confidence config per asset. Does NOT modify production files.

**Process:**
1. Step 1: Test 105 4h configs (7 windows × 15 model combos) → pick top 10
2. Step 2: Test 105 8h configs (same grid) → pick top 10
3. Step 3: Test top 10 × top 10 × 2 strategies × 7 confidence levels (65-95%) = 1,400 combos

**Run:** `python mock_strategy_optimizer.py --asset BTC`

### Current Optimizer Run — BTC on Laptop (Mar 8, 2026)

**STEP 1 (4h) — COMPLETE** (105/105 in 633.9 min ≈ 10.6 hours)

Top 10 4h results:

| Rank | Models | Window | Return | Alpha |
|------|--------|--------|--------|-------|
| 1 | RF+GB+LR | 48h | +90.3% | +83.1% |
| 2 | RF+GB | 48h | +87.1% | +80.0% |
| 3 | RF+GB | 150h | +83.7% | +78.9% |
| 4 | RF+LR | 48h | +84.5% | +77.3% |
| 5 | RF+GB | 100h | +77.8% | +75.4% |
| 6 | RF+GB+LR+LGBM | 48h | +80.7% | +74.8% |
| 7 | RF+LGBM | 100h | +75.5% | +73.2% |
| 8 | RF | 48h | +79.0% | +71.9% |
| 9 | RF+LGBM | 150h | +76.5% | +71.7% |
| 10 | GB+LR | 48h | +77.9% | +70.7% |

Key findings: Short windows (48h) dominate. RF+GB combos in top 5. Current production config (RF+LR+LGBM, w=100h) ranks ~6-7 range.

**STEP 2 (8h) — IN PROGRESS** (~14/105 done, ~5+ hours remaining)
Currently testing 48h window configs. 8h uses 20 features (vs 18 for 4h).

**STEP 3 (combined strategy) — PENDING** (instant math once Step 2 done)

**Note:** The sklearn `parallel.delayed` warnings in console output are HARMLESS — not errors.

---

## Pending Actions / TODO

1. **WAITING:** Laptop mock optimizer to finish (Step 2 8h → Step 3 instant)
2. **TODO:** Run mock optimizer on desktop too: `python mock_strategy_optimizer.py --asset BTC` and `--asset ETH`
3. **TODO:** Run XRP strategy test on desktop: `python crypto_strategy_test.py --asset XRP`
4. **TODO:** Deploy v3 files to desktop
5. **TODO:** Copy `models/crypto_hourly_best_models.csv` to laptop (XRP models are on desktop only)
6. **DECISION:** Whether to kill laptop run and re-run on desktop (~2x faster) vs let it finish

---

## Session History

1. **Mar 2**: Added V2 macro features (VIX, DXY, yields, Fear&Greed, cross-asset) to crypto system
2. **Mar 2**: Created crypto_trading_system.py with Mode A/B/C, Set A vs Set B comparison
3. **Mar 2**: BTC 4h results: Set A 76.5% vs Set B 75.7%. Charts, Mode C saving fix
4. **Mar 3**: Added Mode D (full feature analysis pipeline), Telegram live trader, initial README
5. **Mar 4**: Added confidence threshold backtester, derivative features, live trader timing fix
6. **Mar 4**: Dual horizon support (1h + 4h). Discovered 1h was run with Mode A (55% failure)
7. **Mar 4**: Fixed critical run_loop bug. Full end-to-end flow verification
8. **Mar 4**: Added Mode E (iterative refinement 2nd/3rd pass)
9. **Mar 5**: VVR feature, profit-weighted scoring, small-window test, sklearn warning fix
10. **Mar 7**: 8h horizon testing — BTC 8h: 80.3% (+26.0% alpha). "Both Agree" strategy: +97.2% alpha, 90% win rate. Migrated folder structure to v3. Revolut X auto-trader created
11. **Mar 7-8**: ETH analysis (both_agree +107.8% alpha), XRP analysis (4h 69.2%, 8h 80.8%). SOL removed. Per-asset strategies deployed. Revolut X price fetch fix (3-fallback), balance sync, $300 min trade
12. **Mar 8**: 4-panel interactive Plotly charts (zoom-synced, colorblind-friendly), signal table HTML, weighted strategy tested & rejected, mock optimizer created and running on laptop
13. **Mar 8 (latest)**: Mock optimizer Step 1 (4h) COMPLETE — RF+GB+LR w=48h tops at +83.1% alpha. Step 2 (8h) in progress ~14/105 done

---

## Recent Claude Conversation Summary (Mar 8, 2026)

This section captures the key exchanges from the most recent sessions so the next Claude conversation doesn't start from scratch.

### Chart System Evolution (Mar 8)

**Before:** Static matplotlib PNG backtest charts with price + signals + portfolio.

**After:** Interactive 4-panel Plotly HTML charts with:
- Panel 1: Price candles + 4h BUY/SELL signals (blue/red triangles)
- Panel 2: Price candles + 8h BUY/SELL signals
- Panel 3: Price candles + combined strategy signals
- Panel 4: Portfolio value ($1,000 start) vs Buy & Hold
- All 4 panels are zoom-synced (zoom one = zoom all)
- Color scheme: Blue=BUY, Red=SELL, Yellow=HOLD (colorblind-friendly)

**Signal Table:** New HTML table showing hourly signals:
- Columns: Time | Price | Price+1h | Δ1h(%) | Strategy | Correct? | 4h Signal | 4h Conf | 8h Signal | 8h Conf
- Green rows = correct prediction, Red = wrong
- Shows combined strategy decision alongside individual horizon signals

### Weighted Strategy — Tested and Rejected

Alex asked about weighting horizons differently. We tested a confidence-weighted approach (e.g., weight 4h by its confidence, 8h by its confidence, combine). Results showed NO improvement over simple both_agree/either strategies. The simpler binary strategies already capture the signal well.

### Mock Combined Optimizer

Created `mock_strategy_optimizer.py` to systematically find the best model + strategy + confidence combination. Tests 1,400 combos (top 10 4h × top 10 8h × 2 strategies × 7 confidence levels). Running on laptop for BTC — Step 1 done, Step 2 in progress.

Key insight from Step 1 (4h): **w=48h with RF+GB combos** massively outperform current production config. The production config (RF+LR+LGBM, w=100h) is good but not optimal according to optimizer.

### Revolut X Fixes (Mar 7-8)

Price fetch was failing because API response format changed. Fixed with 3-fallback approach:
1. Public orderbook: `data.asks[0].p` (most reliable)
2. Authenticated tickers endpoint
3. Last trades endpoint

Balance sync: Trader now syncs actual exchange balances on startup and every cycle, tracking USD + crypto held. Position files in `config/position_{ASSET}.json` track entry price and quantity.

### Per-Asset Strategy Deployment

After testing strategies per asset:
- BTC: "both_agree" is optimal (fewer but higher quality trades, 90% win rate)
- ETH: "either" is optimal (more trades, still strong alpha at 78.9% win rate)
- XRP: Strategy test still pending

These are stored in `config/trading_config.json` and the Revolut trader reads them per-asset.

### Files Changed Since v2

| File | Status | Key Changes |
|------|--------|-------------|
| crypto_trading_system.py | Major update | Shortcuts 5/6/7, 4-panel Plotly charts, signal table, v3 folder paths |
| crypto_revolut_trader.py | New file | Multi-asset auto-trader, Revolut X API, position management |
| crypto_live_trader.py | Updated | Both Agree strategy, v3 paths |
| crypto_strategy_test.py | New file | Strategy backtester (both_agree, either, weighted) |
| crypto_horizon_test.py | Updated | v3 model paths |
| mock_strategy_optimizer.py | New file | Mock combined optimizer (non-production) |
| migrate_folders.py | New file | One-time v2→v3 migration |
| README.md | This file | Complete rewrite for v3 |

---

## Quick Start — New Claude Session Checklist

When starting a new Claude chat:

1. Upload: README.md + crypto_trading_system.py + crypto_revolut_trader.py + mock_strategy_optimizer.py + models/crypto_hourly_best_models.csv
2. Tell Claude: "Read the README first"
3. If mock optimizer finished: share results for Step 2 (8h) and Step 3 (combined)
4. If deploying to desktop: remind about different paths (C:\algo_trading vs C:\Users\Alex\algo_trading\Algo_trading)
5. Check pending actions list above for what needs doing

### Expected Next Steps (for next Claude session)

1. Review mock optimizer final results (if finished): best 4h config, best 8h config, best strategy+confidence combo
2. Decide whether to update production models in crypto_hourly_best_models.csv with optimizer findings
3. Run optimizer for ETH and XRP too
4. Run XRP strategy test
5. Deploy all v3 files to desktop
6. Start live trading with optimized configs
7. Consider: add more assets? Adjust confidence thresholds? Refine chart display?

---

## ASSETS Dictionary

```python
ASSETS = {
    'BTC':   {'source': 'binance', 'ticker': 'BTC/USDT',  'file': 'data/btc_hourly_data.csv',  'start': '2017-08-01'},
    'ETH':   {'source': 'binance', 'ticker': 'ETH/USDT',  'file': 'data/eth_hourly_data.csv',  'start': '2017-08-01'},
    'XRP':   {'source': 'binance', 'ticker': 'XRP/USDT',  'file': 'data/xrp_hourly_data.csv',  'start': '2018-05-01'},
}
```

Crypto = Binance via ccxt (free, no API key needed). All hourly OHLCV.

---

## File Dependencies

```
crypto_trading_system.py
  ├── imports: hardware_config.py
  ├── reads: data/macro_data/*.csv
  ├── reads/writes: data/{asset}_hourly_data.csv
  ├── reads/writes: models/crypto_hourly_best_models.csv
  └── writes: charts/*.html, charts/*.png, models/*.json

crypto_revolut_trader.py
  ├── imports: crypto_trading_system.py
  ├── reads: models/crypto_hourly_best_models.csv
  ├── reads: config/trading_config.json, config/revolut_x_config.json, config/private.pem
  ├── reads/writes: config/position_*.json
  └── sends: Revolut X API + Telegram

crypto_strategy_test.py
  ├── imports: crypto_trading_system.py
  └── reads: models/crypto_hourly_best_models.csv

mock_strategy_optimizer.py
  ├── imports: crypto_trading_system.py
  └── reads: models/crypto_hourly_best_models.csv (read-only, never writes)
```

---

*Last updated: March 8, 2026 — v3 with dual horizon, Revolut X auto-trading, interactive charts, mock optimizer*
