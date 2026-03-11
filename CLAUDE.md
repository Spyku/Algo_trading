# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Automated ML trading system for crypto (BTC, ETH, XRP). Generates hourly BUY/SELL/HOLD signals using dual-horizon (4h + 8h) ensemble ML models with walk-forward validation. Executes trades on Revolut X via Ed25519-signed API.

**Owner:** Alex, Lausanne, Switzerland (CET/CEST timezone)

---

## Machine Setup

| Machine | Engine Path | Venv |
|---------|-------------|------|
| Desktop (primary) | `G:\engine\` (Google Drive synced) | `C:\algo_trading\venv\Scripts\activate.bat` |
| Laptop | `C:\Users\Alex\algo_trading\engine\` | `C:\Users\Alex\algo_trading\venv\Scripts\activate.bat` |

- **Desktop:** i7-14700KF, RTX 4080, 32GB — used for long Mode D runs
- **Laptop:** 16 cores, RTX 3070 Ti
- **GitHub:** https://github.com/Spyku/Algo_trading
- **Push:** `git_push.bat` from `G:\engine\`
- **OS:** Windows 11, Python 3.13+ venv (NOT conda)
- **GPU:** LGBM uses GPU (`device='gpu'`), configured in `hardware_config.py`

---

## Commands

```bash
# Quick signals (Mode B shortcuts)
python crypto_trading_system.py 5              # Quick BTC (both horizons)
python crypto_trading_system.py 6              # Quick ETH
python crypto_trading_system.py 7              # Quick XRP

# Full pipeline
python crypto_trading_system.py B BTC 4,8h     # Mode B — signals from saved models
python crypto_trading_system.py D BTC 4,8h 2y  # Mode D — full pipeline (~90 min/horizon)
python crypto_trading_system.py F BTC 4,8h     # Mode F — strategy comparison (400h window)
python crypto_trading_system_v5.2.py D BTC 1,2,3,4,5,6,7,8h 2y  # V5.2 all horizons
python crypto_trading_system_v5.2.py G BTC     # V5.2 Mode G — horizon pair test (168h)

# Auto-trader
python crypto_revolut_trader.py --loop            # Live trading loop
python crypto_revolut_trader.py --dry-run --loop  # Signals only, no trades
python crypto_revolut_trader.py --status          # Show positions
python crypto_revolut_trader.py --balance         # Revolut X balance
```

**Telegram commands while trader is running:** `/stop` `/status` `/pause` `/resume` `/balance` `/sync`

---

## Architecture

### Production File Chain

```
crypto_trading_system.py  (V5 production — Modes B/D/E/F)
  └── hardware_config.py  (machine-specific model configs, n_jobs, GPU settings)

crypto_revolut_trader.py  (auto-trader — reads trading_config.json)
  └── crypto_live_trader.py  (signal generation library — NOT run directly)
        └── crypto_trading_system.py  (imports ASSETS, features, models, download/load/build)
```

### Key Concepts

- **Dual horizons:** 4h and 8h — completely independent models per horizon (different features, window, combo). Never mix them.
- **No model persistence:** System retrains from scratch every prediction. No .pkl files. Intentional design.
- **Labels:** Relative to 168h rolling median return, not absolute price direction. `label = 1` means return > median.
- **Walk-forward validation:** Train on last `window` hours → predict next candle → step forward 72h. No future leakage.
- **Scoring (V5):** `accuracy × (1 + max(cum_return, 0) / 100)` — rewards being right AND making money.
- **Models:** RF, GB, LR, LGBM — all 15 combinations tested (solo + pairs + triples + quad).
- **Features:** 44 technical + 81 macro/sentiment/cross-asset = 125 total.

### Data Flow

```
data/{asset}_hourly_data.csv           <- price data (Binance via ccxt)
data/macro_data/*.csv                  <- VIX, DXY, S&P500, Fear&Greed, etc. (yfinance)
models/crypto_hourly_best_models.csv   <- CENTRAL CONFIG: best model per (asset, horizon)
config/trading_config.json             <- per-asset strategy + min_confidence (written by Mode F)
config/position_{ASSET}.json           <- position tracking
config/revolut_x_config.json           <- Revolut X API key
config/private.pem                     <- Ed25519 signing key
config/telegram_config.json            <- Telegram bot token
```

**Config files are NOT in git.** `config/` is in `.gitignore`. Never push credentials.

---

## Key Constants

```python
TRADING_FEE = 0.0009        # 0.09% Revolut X taker fee (0% maker) — applied on BUY and SELL
MIN_CONFIDENCE = 75         # global fallback only — overridden per asset by Mode F
REPLAY_HOURS = 200          # Modes B, D signal replay
REPLAY_HOURS_F = 400        # Mode F — longer window for more trades in strategy selection
REPLAY_HOURS_G = 168        # Mode G — last week only
DIAG_STEP = 72
DIAG_WINDOWS = [48, 72, 100, 150, 200]       # horizons 5-8h
DIAG_WINDOWS_SHORT = [24, 48, 72, 100, 150]  # horizons 1-4h
```

---

## Version Files

| File | Status | Notes |
|------|--------|-------|
| `crypto_trading_system.py` | V5 Production | DO NOT break — live trader imports from it |
| `crypto_trading_system_v5.2.py` | V5.2 Experimental | All 8 horizons (1-8h) + Mode G |
| `crypto_revolut_trader.py` | Live | Multi-asset live trader |
| `crypto_live_trader.py` | Live | Signal generation core — NOT run directly |
| `hardware_config.py` | Active | Machine-specific config |
| `crypto_trading_system_v4.py` | Reference | Calmar/Sharpe scoring (superseded) |
| `crypto_trading_system_v3_old.py` | Archive | Original production |

---

## Strategies

| Strategy | Logic |
|----------|-------|
| `both_agree` | BUY when 4h AND 8h agree; SELL when either says SELL |
| `either_agree` | BUY when either says BUY; SELL when either says SELL |
| `4h_only` / `8h_only` | Single horizon only |
| `8h_and_1h`, `8h_and_2h` | V5.2 short+long combos |
| `any_agree` | BUY when any horizon says BUY |

**Mode F writes the best strategy and min_confidence directly to `trading_config.json`. The trader picks this up automatically on startup — no manual override needed.**

---

## Current Best Models

| Asset | Horizon | Models | Window | Accuracy | Return | Score | Status |
|-------|---------|--------|--------|----------|--------|-------|--------|
| BTC | 4h | RF+GB+LR | 100h | 80.2% | +125% | 1.804 | V5 2y |
| BTC | 8h | RF+GB | 150h | 84.7% | +319% | 3.550 | V5 2y |
| ETH | 4h | RF+LGBM | 100h | 68.6% | +505% | 4.154 | V5 2y |
| ETH | 8h | GB | 48h | 79.3% | +616% | 5.681 | V5 2y WARNING |
| XRP | 4h | GB | 100h | 69.2% | — | — | V4 1y only — needs V5 run |
| XRP | 8h | RF+LR | 100h | 80.8% | — | — | V4 1y only — needs V5 run |

**WARNING — ETH 8h GB outputs 100% confidence on every signal** — overfit on 48h window.
Fix: add `CalibratedClassifierCV` wrapper in Mode D for GB models. Do not increase ETH `max_position_usd` until fixed.

## Current Trading Config

```json
{
  "BTC": { "strategy": "either_agree", "min_confidence": 80, "symbol": "BTC-USD", "max_position_usd": 10000 },
  "ETH": { "strategy": "8h_only",      "min_confidence": 60, "symbol": "ETH-USD", "max_position_usd": 1000 }
}
```

**BTC is live trading. ETH is configured but not recommended until GB calibration is fixed.**

---

## Critical Rules

1. **Never modify `crypto_trading_system.py` without testing first.** It is V5 production. The live trader imports from it by exact filename.
2. **CSV merge logic:** When saving to `crypto_hourly_best_models.csv`, always filter by BOTH coin AND horizon.
3. **`generate_signals()` needs `feature_override`:** When feature_set is D, caller must pass `feature_override=config['optimal_features'].split(',')`.
4. **All timestamps in live trader use Europe/Zurich** via `_to_local()` helper.
5. **`trading_config.json` has `min_confidence` per asset** — set by Mode F. Global `MIN_CONFIDENCE=75` is only a fallback.
6. **SSL fix on Windows:** `ssl._create_unverified_context()` applied in both `crypto_revolut_trader.py` and `crypto_live_trader.py`.

---

## Pending Work

1. **XRP V5** — run `python crypto_trading_system.py D XRP 4,8h 2y` on laptop, then Mode F
2. **ETH GB calibration** — add `CalibratedClassifierCV` to Mode D for GB model
3. **V5.2 Mode D BTC** — all-8-horizons run hung on 1h diagnostic (worker deadlock suspected). Needs root cause investigation before retry.
4. **Mode G** — can only run after all 8 horizons complete for V5.2
5. **Weekly F runs** — re-run `F BTC 4,8h` and `F ETH 4,8h` weekly to refresh strategy
