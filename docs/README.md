# Algo Trading System

ML-based trading system using walk-forward ensemble models (Random Forest, Gradient Boosting, LightGBM, Logistic Regression) for European indices and crypto.

## Live Dashboards

- **[Hourly Dashboard (V1)](hourly_dashboard.html)** — 3-tier signals (BUY / HOLD / SELL)
- **[Hourly Dashboard (V2)](hourly_dashboard_v2.html)** — 5-tier graduated (STRONG BUY / BUY / HOLD / SELL / STRONG SELL)

## Systems

| File | Description |
|------|-------------|
| `hourly_trading_system.py` | Hourly system: SMI/DAX/CAC40 indices, 4h prediction horizon |
| `crypto_trading_system.py` | Daily system: BTC/ETH/SOL/XRP/DOGE + SMI/DAX/CAC40 |
| `ib_auto_trader.py` | Interactive Brokers auto-trader for index CFDs |
| `hardware_config.py` | Auto-detects DESKTOP vs LAPTOP, configures GPU/CPU/parallelism |
| `model_diagnostic.py` | Standalone diagnostic tool for daily system |
| `feature_analysis.py` | Hourly feature importance/selection |
| `feature_analysis_daily.py` | Daily feature importance/selection |

## Architecture

- **Walk-forward validation**: no lookahead bias, models retrained at each step
- **Ensemble**: majority vote across RF, GB, LGBM (+ LR in some combos)
- **Diagnostic**: tests 75 configurations (15 model combos x 5 training windows) in parallel
- **Hardware auto-detection**: GPU (LightGBM) for signal generation, CPU-only parallel for diagnostics
- **Two strategies**: V1 (all-in/all-out) and V2 (graduated position sizing 50%/100%)

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/algo_trading.git
cd algo_trading

# Create venv
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install numpy pandas scipy scikit-learn lightgbm ccxt yfinance ib_insync nest_asyncio joblib matplotlib

# Run hourly system
python hourly_trading_system.py

# Run daily crypto system
python crypto_trading_system.py
```

## Requirements

- Python 3.12+ (tested on 3.14)
- NVIDIA GPU recommended (LightGBM GPU acceleration)
- Interactive Brokers TWS/Gateway for live trading
