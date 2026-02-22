# Broly 1.2 — Algorithmic Trading System

ML-powered hourly trading system for European index CFDs (SMI, DAX, CAC40) with Interactive Brokers integration.

## Architecture

```
algo_trading/
├── daily_setup.py          # Morning: data update + 75-config diagnostic
├── generate_signals.py     # Backtest dashboard (1:1, 1:5, 1:10 leverage)
├── ib_auto_trader.py       # Live trading via Interactive Brokers
├── ib_test_connection.py   # IB connection test utility
├── features_v2.py          # V2 feature engineering (momentum/mean-rev/cross-asset)
├── hardware_config.py      # GPU/CPU model definitions (machine-specific)
├── broly.py                # Core Broly system
├── data/
│   ├── indices/            # SMI, DAX, CAC40 hourly OHLCV CSVs
│   ├── crypto/             # BTC, ETH, SOL, etc.
│   ├── setup_config.json   # Generated: optimal features + model config
│   └── hourly_best_models.csv  # Generated: best model per asset
├── output/
│   ├── dashboards/         # Generated HTML dashboards
│   ├── charts/             # Chart data JSON
│   ├── backtests/          # Backtest results
│   └── diagnostics/        # Diagnostic outputs
└── docs/
```

## Daily Workflow

```
1. Morning (once, ~40 min):
   python daily_setup.py
   → Downloads/updates hourly data
   → Runs 75-config diagnostic (5 windows × 15 model combos × all assets)
   → Exports data/setup_config.json + data/hourly_best_models.csv

2. Backtest (on demand, ~2 min):
   python generate_signals.py
   → Walk-forward backtest: last month with last-week marker
   → Equity curves at 1:1, 1:5, 1:10 leverage (all on same chart)
   → Interactive HTML dashboard with zoom + legend toggle

3. Live trading (continuous):
   python ib_test_connection.py           # Verify IB setup
   python ib_auto_trader.py               # Single cycle
   python ib_auto_trader.py --loop        # Hourly loop during market hours
   python ib_auto_trader.py --status      # Check positions
   python ib_auto_trader.py --close-all   # Emergency close
```

## Model Pipeline

- **15 optimal V2 features** selected from 50+ candidates via forward selection
- **Walk-forward training**: retrain ensemble every hour on sliding window
- **5-tier signals**: STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
- **Ensemble voting**: best model combo per asset (LGBM, XGBoost, RF, etc.)
- **76.1% accuracy** on validation (V2 features vs 71.8% V1)

## Risk Controls (IB Auto Trader)

- Max position: 20% of net liquidation per trade
- Stop-loss: 2% per position
- Daily loss limit: 5% of starting equity
- Max 3 concurrent positions
- 2h cooldown after stop-loss
- Market hours enforcement (07:00–16:00 UTC)

## Requirements

```
pip install pandas numpy scikit-learn lightgbm xgboost yfinance ib_insync
```

GPU acceleration (optional): CUDA + LightGBM GPU build.
