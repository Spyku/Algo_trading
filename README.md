# Crypto Hourly ML Trading System — v3 / v4

Automated machine learning trading system for crypto (BTC, ETH, XRP). Generates hourly BUY/SELL/HOLD signals using dual-horizon (4h + 8h) ensemble ML models with walk-forward validation. Executes trades automatically on Revolut X via Ed25519-signed API.

**Active assets:** BTC, ETH, XRP (SOL removed — poor results)
**Horizons:** 4h (short-term) and 8h (medium-term) — separate models, features, and windows per horizon
**Strategies:** BTC = "both_agree" | ETH = "either" | XRP = pending test

---

## Version Status

| File | Version | Status | Purpose |
|------|---------|--------|---------|
| `crypto_trading_system.py` | V3 | ✅ Production | Live trading — DO NOT modify without testing |
| `crypto_trading_system_v4.py` | V4 | 🧪 Experimental | Improved scoring + CI + permutation test |

### What V4 adds (not in V3)
1. **Bootstrap CI** — Every Mode B/D/E run now prints `Accuracy: 72.5% [95% CI: 66.3%–78.4%] (n=196)`. Zero extra compute.
2. **Calmar/Sharpe scoring** — Replaces heuristic `accuracy × profit_factor^1.5 × dd_penalty` with industry-standard `0.45×Calmar + 0.35×Sharpe + 0.20×Accuracy`. Top-5 table shows Calmar and Sharpe columns.
3. **Permutation significance test** (`--permtest` flag, off by default) — Shuffles labels 200× to compute p-value. Proves the model found real edge, not noise. Adds ~30 min per asset, only runs when explicitly requested.

### Running V4
```powershell
# Identical to V3 — drop-in replacement
python crypto_trading_system_v4.py 5
python crypto_trading_system_v4.py D BTC 4h 1y

# V4-only: permutation test (adds ~30 min per asset)
python crypto_trading_system_v4.py D BTC 4h 1y --permtest
```

### V4 Validation Status
All Phase 2 mock tests passed before porting:

| Test | Result | Key number |
|------|--------|------------|
| V4 — Calmar/Sharpe edge cases | 11/11 PASS | No crashes on zero trades, flat price, extreme returns |
| V2 — Bootstrap CI convergence | 4/4 PASS | CI compresses 2.76× as n grows (near-perfect 1/√n) |
| V1 — Holdout stability | 3/4 PASS, 1 WARN | Gap std=6.1% — WARN expected on synthetic random-walk data |
| V3 — Permutation anchors | 3/3 PASS | Noise p=0.980, signal p=0.000 |
| V5 — Full pipeline dry-run | 6/6 PASS | Gap=+1.4%, p=0.000, CI [61.9%–71.2%] on 396 predictions |

---

## FOR NEW CLAUDE CHAT — START HERE

If you are Claude reading this in a new conversation, this README is your complete reference. Read it fully before making any code changes.

### What to Upload for a New Chat

Alex should upload these files when starting a new conversation:
1. `README.md` (this file — the master reference)
2. `crypto_trading_system.py` (V3 production system, ~3300 lines)
3. `crypto_trading_system_v4.py` (V4 experimental with improvements, ~3200 lines)
4. `crypto_revolut_trader.py` (auto-trader with Revolut X integration)
5. `mock_strategy_optimizer.py` (mock combined optimizer — DO NOT deploy to production)
6. `models/crypto_hourly_best_models.csv` (current best models)
7. Optionally: `mock_crypto_trading_system.py` and `mock_crypto_trading_system_validation.py` if continuing V4 development

### Key Rules for Editing This Codebase

1. **Never modify `crypto_trading_system.py` without testing in V4 first.** V3 is production. V4 is the test bed.
2. **Horizons are 4h and 8h** (not 1h + 2h). AVAILABLE_HORIZONS = [4, 8]
3. **CSV merge logic is critical.** When saving to crypto_hourly_best_models.csv, always filter by BOTH coin AND horizon. Wrong filter = overwrite the other horizon's config
4. **feature_set check must include D.** All current models use Set D with custom optimal_features
5. **generate_signals() needs feature_override.** When feature_set is D, the caller must pass `feature_override=config['optimal_features'].split(',')`. Without this, it defaults to Set A
6. **Live trader imports from crypto_trading_system.py.** The filename must be exact — it imports V3, not V4
7. **Labels are relative to rolling median, not absolute.** `label = 1` means "return in next H hours is above the 168h rolling median return", not just "price goes up"
8. **No model persistence.** The system retrains from scratch every prediction. No .pkl files. This is intentional
9. **Each horizon = completely independent model.** Different features, model combo, window, labels. Never mix them
10. **All timestamps in live trader use Europe/Zurich** via `_to_local()` helper
11. **Folder structure is v3.** Data in `data/`, models in `models/`, charts in `charts/`, config in `config/`. Root has .py files only

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
├── crypto_trading_system.py           # V3 PRODUCTION — Modes B/D/E + shortcuts 5/6/7
├── crypto_trading_system_v4.py        # V4 EXPERIMENTAL — bootstrap CI, Calmar/Sharpe, --permtest
├── crypto_revolut_trader.py           # Multi-asset Revolut X auto-trader with position management
├── crypto_live_trader.py              # Required dependency of revolut trader — NOT run directly
├── mock_strategy_optimizer.py         # Mock combined optimizer (DO NOT deploy to production)
├── mock_crypto_trading_system.py      # Phase 1 mock tests for V4 improvements
├── mock_crypto_trading_system_validation.py  # Phase 2 validation tests for V4 improvements
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
```

---

## Key Constants

```python
TRADING_FEE = 0.0009                # 0.09% per trade (Revolut X taker fee)
MIN_CONFIDENCE = 75                 # Model must be ≥75% sure to trigger BUY
AVAILABLE_HORIZONS = [4, 8]         # 4h and 8h models
REPLAY_HOURS = 200                  # Hours of signals to generate in Mode B
DIAG_STEP = 72                      # Step size for walk-forward diagnostic
DATA_DIR = 'data'
CHARTS_DIR = 'charts'
MODELS_DIR = 'models'
CONFIG_DIR = 'config'
MACRO_DIR = 'data/macro_data'
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

## Modes Quick Reference

| Mode | Purpose | Time | When to Run |
|------|---------|------|-------------|
| **B** | Quick signals from saved models | ~2 min | Daily |
| **D** | Full pipeline: 124 features → analysis → optimal subset → diagnostic | ~3-8h | Quarterly |
| **E** | Iterative refinement of Mode D | ~1-4h | After Mode D |
| **5/6/7** | Shortcuts: Quick BTC/ETH/XRP (Mode B, both horizons) | ~5 min | Daily |

Modes A and C have been removed (superseded by Mode D).

---

## CLI Shortcuts

```powershell
# V3 (production)
python crypto_trading_system.py 5              # Quick BTC
python crypto_trading_system.py 6              # Quick ETH
python crypto_trading_system.py 7              # Quick XRP
python crypto_trading_system.py B BTC 4,8h    # Mode B
python crypto_trading_system.py D BTC 4h 1y   # Mode D

# V4 (experimental — same CLI + optional --permtest)
python crypto_trading_system_v4.py 5
python crypto_trading_system_v4.py D BTC 4h 1y --permtest
```

---

## Interactive Charts — 4-Panel Layout

Generated by shortcuts 5/6/7, these are interactive Plotly HTML charts with zoom-synced panels:

**Panel 1:** Price + 4h signals (blue triangles = BUY, red triangles = SELL)
**Panel 2:** Price + 8h signals
**Panel 3:** Price + combined strategy signals (both_agree or either)
**Panel 4:** Portfolio ($1,000 start) vs Buy & Hold

Color scheme (colorblind-friendly): 🔵 Blue = BUY | 🔴 Red = SELL | 🟡 Yellow = HOLD

Files generated per asset:
- `charts/{ASSET}_strategy_1week.html` (168h window)
- `charts/{ASSET}_strategy_1month.html` (720h window)
- `charts/{ASSET}_signal_table.html` (per-hour signal table with correctness)

---

## Revolut X Auto-Trader (crypto_revolut_trader.py)

### API Details
- Base URL: `https://revx.revolut.com/api/1.0`
- Auth: Ed25519 signing via pynacl + cryptography
- Price fetch: 3 fallbacks — public orderbook → authenticated tickers → last trades

### Behavior
1. **Startup**: Sync positions from exchange → send startup Telegram → immediate full scan
2. **Every hour**: Sync → download candle → generate 4h+8h signals → apply strategy → execute trade → Telegram
3. **Between hours (every 5 min)**: Check Telegram commands + sync balances

### Telegram Commands
`/stop` `/status` `/pause` `/resume` `/balance` `/sync`

---

## Revolut X Account Status (as of Mar 8, 2026)

- USD: $12,697.53
- BTC: 0.001403 (~$124)
- ETH: 0.505 (~$1,160)

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

### V3 Scoring
```
combined_score = accuracy × profit_factor^1.5 × drawdown_penalty
```

### V4 Scoring (Calmar/Sharpe)
```
combined_score = 0.45 × Calmar + 0.35 × Sharpe + 0.20 × accuracy_signal
```
Calmar = annualised return / max drawdown (clipped to [-5, 10])
Sharpe = per-trade mean/std × √n_trades (clipped to [-3, 5])

---

## Mock Strategy Optimizer (mock_strategy_optimizer.py)

**Purpose:** Tests 1,400 combinations to find optimal model+strategy+confidence config per asset. Does NOT modify production files.

**Run:** `python mock_strategy_optimizer.py --asset BTC`

### BTC Optimizer Results (Mar 8, 2026)

**STEP 1 (4h) — COMPLETE** (105/105 in ~10.6 hours)

Top 4h results: RF+GB+LR w=48h at +83.1% alpha. Short windows (48h) dominate. Current production config (RF+LR+LGBM, w=100h) is solid but not the top performer.

**STEP 2 (8h) — PENDING** (~14/105 done when last checked)

**STEP 3 (combined strategy) — PENDING** (instant once Step 2 done)

---

## Pending Actions / TODO

1. **WAITING:** Laptop mock optimizer to finish (Step 2 8h → Step 3 instant)
2. **TODO:** Run Mode D for XRP: `python crypto_trading_system.py D XRP 4,8h 1y`
3. **TODO:** Update `models/crypto_hourly_best_models.csv` with optimizer results once complete
4. **TODO:** Run V4 on real BTC data to compare Calmar/Sharpe scores vs old heuristic
5. **DECISION PENDING:** Keep or remove DOGE, SMI, DAX, CAC40 from the system
6. **DECISION PENDING:** Promote V4 to production once real-data validation done

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
13. **Mar 8**: Mock optimizer Step 1 (4h) COMPLETE — RF+GB+LR w=48h tops at +83.1% alpha. Step 2 (8h) in progress
14. **Mar 8 (latest)**: Full code review — 2 new bugs fixed (combined summary key lookup, overall alpha leak). Mock test framework built and validated for 4 improvements. V4 created with Improvements 2+3+4. V3 production file kept clean and untouched.

### V4 Development Summary (Mar 8, Session 6-7)

**10 bugs total fixed across sessions 1-6** (8 previous + 2 new):
- Bug 1-new: `_run_quick_asset` combined summary silently failed — `chart_data[key]` → `chart_data.get('assets', chart_data)[key]`
- Bug 2-new: `generate_strategy_html` overall alpha showed last-loop value — now computed from full period portfolio sim

**4 improvements designed, validated in mock, ported to V4:**
- Improvement 1 (holdout): Decided NOT to port — excluding last 720h from training is wrong for time-sensitive assets. Reframed as "regime alignment check" for future work
- Improvement 2 (bootstrap CI): Ported ✅ — zero compute cost, prints CI on every Mode B/D/E run
- Improvement 3 (permutation test): Ported ✅ — optional `--permtest` flag, off by default (~30 min per asset)
- Improvement 4 (Calmar/Sharpe): Ported ✅ — replaces heuristic scoring, adds Calmar/Sharpe columns to top-5 table

---

## File Dependencies

```
crypto_trading_system.py  (V3 — production)
  ├── imports: hardware_config.py
  ├── reads: data/macro_data/*.csv
  ├── reads/writes: data/{asset}_hourly_data.csv
  ├── reads/writes: models/crypto_hourly_best_models.csv
  └── writes: charts/*.html, charts/*.png, models/*.json

crypto_trading_system_v4.py  (V4 — experimental, same dependencies as V3)

crypto_revolut_trader.py
  ├── imports: crypto_trading_system.py  ← always V3, never V4
  ├── reads: models/crypto_hourly_best_models.csv
  ├── reads: config/trading_config.json, config/revolut_x_config.json, config/private.pem
  ├── reads/writes: config/position_*.json
  └── sends: Revolut X API + Telegram

mock_strategy_optimizer.py
  ├── imports: crypto_trading_system.py
  └── reads: models/crypto_hourly_best_models.csv (read-only, never writes)
```

---

## Quick Start — New Claude Session Checklist

When starting a new Claude chat:

1. Upload: README.md + crypto_trading_system.py + crypto_trading_system_v4.py + crypto_revolut_trader.py + models/crypto_hourly_best_models.csv
2. Tell Claude: "Read the README first"
3. If mock optimizer finished: share results for Step 2 (8h) and Step 3 (combined)
4. If deploying to desktop: remind about different paths (`C:\algo_trading` vs `C:\Users\Alex\algo_trading\Algo_trading`)
5. Check pending actions list above for what needs doing

---

*Last updated: March 8, 2026 — V3 production stable, V4 experimental with bootstrap CI + Calmar/Sharpe + --permtest*
