# Crypto ML Trading System v3

Multi-asset ML trading system for BTC, ETH, XRP, DOGE. Hourly BUY/SELL/HOLD signals using ensemble models with walk-forward validation. Auto-trades via Revolut X API with Telegram notifications and remote control.

## Models (March 7, 2026)

| Asset | 4h Model | 4h Acc | 8h Model | 8h Acc | Strategy |
|-------|----------|--------|----------|--------|----------|
| BTC | RF+LR+LGBM w=100h | 72.5% | RF+GB+LGBM w=300h | 86.3% | both_agree |
| ETH | RF+LR w=100h | 78.3% | RF+LR w=100h | 75.0% | either |
| XRP | GB w=100h | 69.2% | RF+LR w=100h | 80.8% | either |

**BTC "both agree":** 91.2% win rate, +103% alpha over 1 month
**ETH "either":** 75.4% win rate, +132% alpha over 1 month

---

## Quick Start

```powershell
# Quick signals + interactive charts
python crypto_trading_system.py 5    # BTC
python crypto_trading_system.py 6    # ETH
python crypto_trading_system.py 7    # XRP

# CLI (no menus)
python crypto_trading_system.py B BTC,ETH 4,8h
python crypto_trading_system.py D XRP 8h 1y

# Live trading
python crypto_revolut_trader.py --loop
python crypto_revolut_trader.py --status
python crypto_revolut_trader.py --balance

# Strategy backtest
python crypto_strategy_test.py --asset BTC
```

---

## For New Claude Chat

Upload: README.md, crypto_trading_system.py, crypto_live_trader.py, crypto_revolut_trader.py, models/crypto_hourly_best_models.csv

### Key Rules
1. Horizons: 4h and 8h only (1h/2h = noise)
2. No SOL (removed — poor results)
3. CSV filter by BOTH coin AND horizon when saving
4. feature_set D/E2/E3 uses optimal_features column; A/B uses hardcoded sets
5. NaN check on optimal_features: `pd.isna()` and `'nan'` string
6. Revolut X trades USD pairs (BTC-USD, not USDT)
7. Desktop: C:\algo_trading | Laptop: C:\Users\Alex\algo_trading\Algo_trading
8. Per-asset strategies: BTC=both_agree, ETH=either, XRP=either
9. Each horizon = independent model (features, window, combo)
10. No model persistence — retrains every hour from scratch

### Architecture
```
Binance hourly data -> 125 features (technical + macro + sentiment)
  -> Walk-forward ML (RF/GB/LR/LGBM ensembles)
  -> 4h + 8h signals -> Per-asset strategy filter
  -> Revolut X API order + Telegram notification
```

---

## Folder Structure

```
Algo_trading/
|-- *.py                    # All scripts in root
|-- data/                   # Hourly CSVs + macro_data/
|-- charts/                 # PNG backtests + interactive HTML strategy charts
|-- models/                 # best_models CSV, feature analysis, chart data JSON
|-- config/                 # telegram, revolut_x, private.pem, positions
```

---

## Files

### Core
- `crypto_trading_system.py` — Main system. Modes A/B/C/D/E + shortcuts 5/6/7. Interactive strategy charts.
- `crypto_live_trader.py` — "Both Agree" signal engine + Telegram
- `crypto_revolut_trader.py` — Multi-asset Revolut X auto-trader. Per-asset strategies/positions/max USD.

### Testing
- `crypto_strategy_test.py` — Tests 4h/8h/both_agree/either + hold durations + confidence filters
- `crypto_horizon_test.py` — Discovers optimal horizons for new assets

### Infrastructure
- `detect_hardware.py` — Generates hardware_config.py (GPU, cores)
- `download_macro_data.py` — VIX, DXY, S&P500, Fear&Greed -> data/macro_data/
- `migrate_folders.py` — One-time folder restructure (v2 -> v3)

---

## Interactive Charts (v3 new)

Shortcuts 5/6/7 generate zoomable Plotly HTML charts:
- `charts/{ASSET}_strategy_1month.html` — 30-day backtest
- `charts/{ASSET}_strategy_1week.html` — 7-day backtest

Shows: price + 4h signals (triangles) + 8h signals (diamonds) + combined strategy (stars) + portfolio equity vs buy & hold. Header with alpha, trades, win rate.

---

## Revolut X Auto-Trader

Assets: BTC-USD, ETH-USD, XRP-USD, DOGE-USD
Modes: AUTO (real orders) | MANUAL (Telegram alerts) | DRY RUN

Per-asset config in `config/trading_config.json`:
- BTC: both_agree, max $X
- ETH: either, max $Y
- XRP: either, max $Z

Telegram remote: /stop /status /pause /resume /balance

---

## Machines

| | Desktop | Laptop |
|--|---------|--------|
| Path | C:\algo_trading | C:\Users\Alex\algo_trading\Algo_trading |
| CPU | i7-14700KF (28 threads) | 16 cores |
| GPU | RTX 4080 16GB | RTX 3070 Ti |
| Workers | 26 | 14 |

---

## Setup New Machine

```powershell
python -m venv venv && .\venv\Scripts\activate
pip install pandas numpy scikit-learn lightgbm joblib matplotlib ccxt pynacl cryptography
python detect_hardware.py
python migrate_folders.py --go    # if copying from existing machine
python download_macro_data.py
python crypto_trading_system.py D BTC 4,8h 1y
python crypto_live_trader.py --setup
python crypto_revolut_trader.py --balance
```

---

## History

- v1 (Mar 2-3): Single asset, Mode A/B/C/D, Set A/B, Telegram
- v2 (Mar 4-5): Mode E, dual horizon, VVR, profit scoring, threshold system
- **v3 (Mar 5-7): 8h model discovery, "Both Agree" strategy, multi-asset (BTC+ETH+XRP), Revolut X auto-trading, interactive Plotly charts, folder restructure, SOL removed, per-asset strategies, Telegram remote commands, CLI shortcuts**
