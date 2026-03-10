# Crypto Hourly ML Trading System — v5

Automated machine learning trading system for crypto (BTC, ETH, XRP). Generates hourly BUY/SELL/HOLD signals using dual-horizon (4h + 8h) ensemble ML models with walk-forward validation. Executes trades automatically on Revolut X via Ed25519-signed API.

**Active assets:** BTC, ETH, XRP (SOL removed — poor results)
**Horizons:** 4h (short-term) and 8h (medium-term) — separate models, features, and windows per horizon
**Strategies:** Set per asset by Mode F (strategy comparison + confidence sweep)

---

## Version Status

| File | Version | Status | Purpose |
|------|---------|--------|---------|
| `crypto_trading_system.py` | V5 | ✅ Production | Live trading — DO NOT modify without testing |
| `crypto_trading_system_v4.py` | V4 | 📦 Reference | Calmar/Sharpe scoring (superseded) |
| `crypto_trading_system_v3_old.py` | V3 | 📦 Archive | Original production (superseded) |

### What V5 adds over V4
1. **New scoring formula** — Replaces `0.45×Calmar + 0.35×Sharpe + 0.20×Accuracy` with `accuracy × (1 + max(return, 0) / 100)`. Directly rewards being right AND making money. Calmar/Sharpe biased toward low-trade configs.
2. **Mode F — Strategy comparison** — Backtests all 4 strategies (both_agree / either_agree / 4h_only / 8h_only) + confidence threshold sweep (60–90%). Auto-writes best strategy and threshold to `trading_config.json`.
3. **Per-step timers in Mode D** — Each pipeline stage prints elapsed time.
4. **hardware_config.py lightened** — `get_diagnostic_models()` uses n_estimators=100 / RF n_jobs=1. Diagnostics now run in ~16 min instead of 40–60 min.
5. **DIAG_WINDOWS reduced** — [48, 72, 100, 150, 200] (was 7 windows). 75 configs total (was 105).

### V5 Scoring
```python
# Diagnostic and Mode F:
combined_score = accuracy * (1 + max(cum_return, 0) / 100)

# Feature analysis (unchanged from V4):
combined = acc * (1 + max(alpha, 0) / 100)
```

---

## FOR NEW CLAUDE CHAT — START HERE

If you are Claude reading this in a new conversation, this README is your complete reference. Read it fully before making any code changes.

### What to Upload for a New Chat

Alex should upload these files when starting a new conversation:
1. `README.md` (this file — the master reference)
2. `crypto_trading_system.py` (V5 production system)
3. `crypto_trading_system_v4.py` (V4 reference — Calmar/Sharpe, kept for comparison)
4. `crypto_revolut_trader.py` (auto-trader — reads min_confidence from trading_config.json)
5. `hardware_config.py`
6. `models/crypto_hourly_best_models.csv` (current best models)

### Key Rules for Editing This Codebase

1. **Never modify `crypto_trading_system.py` without testing first.** It is V5 production.
2. **Horizons are 4h and 8h** (not 1h + 2h). AVAILABLE_HORIZONS = [4, 8]
3. **CSV merge logic is critical.** When saving to crypto_hourly_best_models.csv, always filter by BOTH coin AND horizon.
4. **generate_signals() needs feature_override.** When feature_set is D, the caller must pass `feature_override=config['optimal_features'].split(',')`.
5. **Live trader imports from crypto_trading_system.py.** The filename must be exact.
6. **Labels are relative to rolling median, not absolute.** `label = 1` means return in next H hours is above the 168h rolling median return.
7. **No model persistence.** The system retrains from scratch every prediction. No .pkl files. This is intentional.
8. **Each horizon = completely independent model.** Different features, model combo, window, labels. Never mix them.
9. **All timestamps in live trader use Europe/Zurich** via `_to_local()` helper.
10. **Folder structure is v3.** Data in `data/`, models in `models/`, charts in `charts/`, config in `config/`. Root has .py files only.
11. **trading_config.json now has min_confidence per asset** — set by Mode F. Trader reads this. Global MIN_CONFIDENCE=75 is only a fallback.

---

## Alex's Setup — Two Machines

### Desktop (Primary — ~2x faster than laptop)
```
CPU:     Intel i7-14700KF (20P/28L cores, 26 workers for parallel jobs)
GPU:     NVIDIA RTX 4080 16GB, CUDA 13.1
RAM:     32 GB
Path:    C:\algo_trading\engine
Venv:    C:\algo_trading\venv
Activate: C:\algo_trading\venv\Scripts\activate.bat
```

### Laptop
```
CPU:     16 logical cores
GPU:     NVIDIA RTX 3070 Ti Laptop
Path:    C:\Users\Alex\algo_trading\engine
Venv:    C:\Users\Alex\algo_trading\venv
Activate: C:\Users\Alex\algo_trading\venv\Scripts\activate.bat
```

### Shared Config
```
OS:       Windows 11
Python:   venv (NOT conda)
LGBM:     GPU-enabled (device='gpu')
Broker:   Revolut X (0.09% taker fee) for crypto
User:     Alex, Lausanne, Switzerland (CET/CEST timezone)
GitHub:   https://github.com/Spyku/Algo_trading
```

---

## Complete Folder Structure

```
Algo_trading/
│
├── ===== CORE SYSTEM (root .py files) =====
├── crypto_trading_system.py           # V5 PRODUCTION — Modes B/D/E/F + shortcuts 5/6/7
├── crypto_trading_system_v4.py        # V4 REFERENCE — Calmar/Sharpe scoring (superseded)
├── crypto_trading_system_v3_old.py    # V3 ARCHIVE — original production (superseded)
├── crypto_revolut_trader.py           # Multi-asset Revolut X auto-trader
├── crypto_live_trader.py              # Required dependency of revolut trader — NOT run directly
├── hardware_config.py                 # Machine config — lightweight diagnostic models
│
├── ===== DATA (auto-downloaded) =====
├── data/
│   ├── btc_hourly_data.csv
│   ├── eth_hourly_data.csv
│   ├── xrp_hourly_data.csv
│   └── macro_data/
│       ├── macro_daily.csv
│       ├── fear_greed.csv
│       ├── cross_asset.csv
│       └── macro_hourly.csv
│
├── ===== MODELS =====
├── models/
│   ├── crypto_hourly_best_models.csv  # ** CENTRAL CONFIG ** Best model per (asset, horizon)
│   ├── crypto_feature_analysis_*.csv
│   └── crypto_hourly_chart_data.json
│
├── ===== CHARTS =====
├── charts/
│   ├── {ASSET}_strategy_1week.html
│   ├── {ASSET}_strategy_1month.html
│   ├── {ASSET}_signal_table.html
│   └── {ASSET}_backtest.png
│
├── ===== CONFIG =====
├── config/
│   ├── telegram_config.json
│   ├── revolut_x_config.json
│   ├── private.pem
│   ├── trading_config.json            # Per-asset strategy + max USD + min_confidence (set by Mode F)
│   ├── position_BTC.json
│   ├── position_ETH.json
│   └── position_XRP.json
```

---

## Key Constants

```python
TRADING_FEE = 0.0009                # 0.09% per trade (Revolut X taker fee)
MIN_CONFIDENCE = 75                 # Global fallback — overridden per asset by Mode F
AVAILABLE_HORIZONS = [4, 8]
REPLAY_HOURS = 200
DIAG_STEP = 72
DIAG_WINDOWS = [48, 72, 100, 150, 200]  # 5 windows × 15 combos = 75 configs
DATA_DIR = 'data'
CHARTS_DIR = 'charts'
MODELS_DIR = 'models'
CONFIG_DIR = 'config'
MACRO_DIR = 'data/macro_data'
```

---

## Modes Quick Reference

| Mode | Purpose | Time (desktop) | When to Run |
|------|---------|----------------|-------------|
| **B** | Quick signals from saved models | ~2 min | Daily |
| **D** | Full pipeline: 125 features → analysis → optimal subset → diagnostic → signals | ~90 min per horizon | Quarterly or after market regime change |
| **E** | Iterative refinement after Mode D | ~1–4h | After Mode D |
| **F** | Strategy comparison + confidence sweep → updates trading_config.json | ~seconds | After Mode D (auto), or standalone |
| **5/6/7** | Shortcuts: Quick BTC/ETH/XRP (Mode B, both horizons) | ~5 min | Daily |

### Mode D Timing (desktop, 2y data)
| Step | 4h | 8h |
|------|----|----|
| Macro update | 0.1 min | 0.1 min |
| Data update | 0.1 min | 0.1 min |
| Feature build | 0.0 min | 0.0 min |
| Feature analysis | ~70 min | ~96 min |
| Diagnostic | ~16 min | ~18 min |
| Signal generation | ~2 min | ~1 min |
| **Total** | **~88 min** | **~115 min** |

---

## Current Best Models (V5, 2y)

| Asset | Horizon | Models | Window | Acc | Return | Score | Features | Run |
|-------|---------|--------|--------|-----|--------|-------|----------|-----|
| BTC | 4h | RF+GB+LR | 100h | 80.2% | +125.0% | 1.804 | 125 | V5 2y ✅ |
| BTC | 8h | RF+GB | 150h | 84.7% | +319.4% | 3.550 | 15 | V5 2y ✅ |
| ETH | 4h | RF+LR | 100h | 78.3% | — | — | custom | V4 1y ⚠️ |
| ETH | 8h | RF+LR | 100h | 75.0% | — | — | custom | V4 1y ⚠️ |
| XRP | 4h | GB | 100h | 69.2% | — | — | custom | V4 1y ⚠️ |
| XRP | 8h | RF+LR | 100h | 80.8% | — | — | custom | V4 1y ⚠️ |

⚠️ ETH and XRP still use V4 1y Calmar runs — need V5 2y re-run on laptop.

### BTC 8h Optimal Features (15)
`logret_120h, xa_dax_relstr5d, price_to_sma100h, logret_240h, xa_sp500_relstr5d, vol_ratio_12_48, hour_cos, volatility_48h, atr_pct_14h, xa_nasdaq_relstr5d, sma20_to_sma50h, hour_sin, logret_72h, spread_24h_4h, m_nasdaq_chg1d`

### BTC 4h Optimal Features (125 — all features)
Full feature set used. No smaller subset scored higher on 2y walk-forward.

---

## Trading Config (config/trading_config.json)

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

`min_confidence` is written automatically by Mode F. The trader reads it per asset. Global `MIN_CONFIDENCE=75` is only a fallback if the field is missing.

---

## Strategy Comparison Results (BTC V5, 200h replay)

| Strategy | Return | Win Rate | Trades | Score |
|----------|--------|----------|--------|-------|
| **either_agree** | **+20.9%** | **90%** | **20** | **0.996** |
| 8h_only | +18.5% | 83% | 12 | 0.977 |
| both_agree | +16.4% | 100% | 13 | 0.959 |
| 4h_only | +15.1% | 88% | 16 | 0.948 |

### Confidence Sweep (either_agree)

| Threshold | Return | Win Rate | Trades | Score |
|-----------|--------|----------|--------|-------|
| 60% | +18.5% | 87% | 23 | 0.977 |
| 65% | +18.5% | 87% | 23 | 0.977 |
| 70% | +19.4% | 87% | 23 | 0.984 |
| **75%** | **+20.9%** | **90%** | **20** | **0.996** ← BEST |
| 80% | +20.8% | 90% | 20 | 0.995 |
| 85% | +14.9% | 91% | 11 | 0.947 |
| 90% | +7.1% | 100% | 6 | 0.882 |

---

## CLI Reference

```powershell
# Production (V5)
python crypto_trading_system.py 5                    # Quick BTC
python crypto_trading_system.py 6                    # Quick ETH
python crypto_trading_system.py 7                    # Quick XRP
python crypto_trading_system.py B BTC 4,8h           # Mode B — signals
python crypto_trading_system.py D BTC 4,8h 2y        # Mode D — full pipeline
python crypto_trading_system.py F BTC                # Mode F — strategy comparison

# Auto-trader
python crypto_revolut_trader.py
```

---

## Revolut X Auto-Trader (crypto_revolut_trader.py)

### Behavior
1. **Startup**: Sync positions → Telegram → immediate scan
2. **Every hour**: Sync → download → generate 4h+8h signals → apply strategy → execute → Telegram
3. **Between hours (every 5 min)**: Telegram commands + balance sync

### Telegram Commands
`/stop` `/status` `/pause` `/resume` `/balance` `/sync`

### Confidence Handling
- Trader reads `min_confidence` from `trading_config.json` per asset
- Falls back to global `MIN_CONFIDENCE=75` if not set
- Mode F sets this automatically — no manual editing needed

---

## System Architecture

### Features (125 total)
44 Technical base + 81 Macro/sentiment/cross-asset (VIX, DXY, S&P500, Nasdaq, Gold, US10Y, EUR/USD, USD/JPY, Oil, Fear&Greed, ETH/BTC/DAX correlations)

### Labels
```python
median_return = df['logret_Xh'].rolling(168).median()
label = 1 if logret_Xh > median_return else 0
```
Adaptive threshold — not just "price goes up".

### Models
RF, GB, LR, LGBM — all 15 combinations tested (4 solo + 6 pairs + 4 triples + 1 quad).

### Walk-Forward Validation
No future leakage. Train on last `window` hours → predict next candle → step forward `DIAG_STEP=72h`.

---

## Pending Actions

1. **TODO:** Run V5 Mode D ETH 4,8h 2y on laptop
2. **TODO:** Run V5 Mode D XRP 4,8h 2y on laptop
3. **TODO:** Run Mode F for ETH and XRP after their V5 runs
4. **TODO:** Push V5 + updated hardware_config.py + README to GitHub (`git push`)
5. **TODO:** Pull on laptop before running ETH/XRP (`git pull`)
6. **TODO:** Update best models table once ETH/XRP 2y runs complete
7. **COMPLETED:** V5 scoring (acc×(1+return/100)) ✅
8. **COMPLETED:** Mode F (strategy comparison + confidence sweep) ✅
9. **COMPLETED:** Per-step timers in Mode D ✅
10. **COMPLETED:** hardware_config diagnostic models lightened ✅
11. **COMPLETED:** DIAG_WINDOWS reduced to [48,72,100,150,200] ✅
12. **COMPLETED:** BTC V5 2y run — 4h: RF+GB+LR w=100h 80.2% +125%, 8h: RF+GB w=150h 84.7% +319% ✅
13. **COMPLETED:** V5 promoted to production, V3 archived as v3_old ✅

---

## Session History

1. **Mar 2**: Added V2 macro features. Created crypto_trading_system.py with Modes A/B/C.
2. **Mar 2**: BTC 4h: Set A 76.5% vs Set B 75.7%. Charts, Mode C saving fix.
3. **Mar 3**: Mode D (full feature analysis pipeline), Telegram live trader, README.
4. **Mar 4**: Confidence threshold backtester, derivative features, live trader timing fix.
5. **Mar 4**: Dual horizon (1h+4h → 4h+8h). Fixed critical run_loop bug.
6. **Mar 4**: Mode E (iterative refinement). VVR feature, profit-weighted scoring.
7. **Mar 7**: 8h horizon — BTC 8h 80.3%. "Both Agree" +97.2% alpha, 90% win rate. Revolut X trader.
8. **Mar 7-8**: ETH/XRP analysis. SOL removed. Per-asset strategies. Revolut X fixes.
9. **Mar 8**: 4-panel Plotly charts, signal table HTML. Weighted strategy tested & rejected.
10. **Mar 8**: V4 created with bootstrap CI + Calmar/Sharpe + --permtest.
11. **Mar 9 (Sessions 8-9)**: hardware_config lightened. DIAG_WINDOWS reduced. BTC 4h V4: GB+LR w=200h 63% +48.7%.
12. **Mar 9 (Sessions 10-11)**: V5 created. Scoring replaced. Mode F + confidence sweep. Timers added.
13. **Mar 9 (Session 12)**: BTC 8h V4: RF+GB+LR+LGBM w=200h 74% +67.6%. Mode F on V4: either_agree + 65%.
14. **Mar 10 (Session 13)**: V5 Mode D BTC 4,8h 2y. 4h: RF+GB+LR w=100h 80.2% +125%. 8h: RF+GB w=150h 84.7% +319%. Mode F: either_agree + 75%. V5 → production. V3 → archive.

---

## File Dependencies

```
crypto_trading_system.py  (V5 — production)
  ├── imports: hardware_config.py
  ├── reads: data/macro_data/*.csv
  ├── reads/writes: data/{asset}_hourly_data.csv
  ├── reads/writes: models/crypto_hourly_best_models.csv
  ├── reads/writes: config/trading_config.json  (Mode F writes strategy + min_confidence)
  └── writes: charts/*.html, charts/*.png, models/*.json

crypto_revolut_trader.py
  ├── imports: crypto_trading_system.py  ← always production filename
  ├── reads: models/crypto_hourly_best_models.csv
  ├── reads: config/trading_config.json  (strategy + min_confidence per asset)
  ├── reads: config/revolut_x_config.json, config/private.pem
  ├── reads/writes: config/position_*.json
  └── sends: Revolut X API + Telegram
```

---

*Last updated: March 10, 2026 — V5 production. BTC fully validated on 2y data. ETH/XRP pending V5 re-run on laptop.*
