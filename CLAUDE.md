# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Automated ML trading system for crypto (BTC, ETH, XRP). Generates hourly BUY/SELL/HOLD signals using dual-horizon (4h + 8h) ensemble ML models with walk-forward validation. Executes trades on Revolut X via Ed25519-signed API.

**Owner:** Alex, Lausanne, Switzerland (CET/CEST timezone)

---

## Machine Setup

**One shared engine folder** synced via Google Drive — both machines use the same code, data, and models. Only the venv is local per machine.

| Machine | Engine Path | Venv | Python |
|---------|-------------|------|--------|
| Desktop (primary) | `G:\engine\` (Google Drive synced) | `C:\algo_trading\venv\` | `C:\algo_trading\venv\Scripts\python.exe` |
| Laptop | `G:\Autres ordinateurs\My laptop\engine\` (Google Drive synced) | `C:\Users\Alex\algo_trading\venv\` | `C:\Users\Alex\algo_trading\venv\Scripts\python.exe` |

- **Desktop:** i7-14700KF, RTX 4080, 32GB — used for long Mode D runs
- **Laptop:** 16 cores, RTX 3070 Ti
- **GitHub:** https://github.com/Spyku/Algo_trading
- **Push:** `git_push.bat` from `G:\engine\`
- **OS:** Windows 11, Python 3.14 venv (NOT conda)
- **GPU:** LGBM uses GPU (`device='gpu'`), configured in `hardware_config.py`

### Install / Venv Setup

Each machine needs its own venv with all dependencies. The engine folder is shared but venvs are local.

```bash
# Desktop
C:\algo_trading\venv\Scripts\activate.bat
pip install -r G:\engine\requirements.txt

# Laptop
C:\Users\Alex\algo_trading\venv\Scripts\activate.bat
pip install -r "G:\Autres ordinateurs\My laptop\engine\requirements.txt"
```

**Always use the venv Python** — never system Python. When running from Claude Code, use the full venv path:
- Desktop: `"C:/algo_trading/venv/Scripts/python.exe"`
- Laptop: `"C:/Users/Alex/algo_trading/venv/Scripts/python.exe"`

---

## Commands

```bash
# Quick signals (Mode B shortcuts)
python crypto_trading_system.py 5              # Quick BTC (both horizons)
python crypto_trading_system.py 6              # Quick ETH
python crypto_trading_system.py 7              # Quick XRP

# Full pipeline
python crypto_trading_system.py B BTC 4,8h     # Mode B — signals from saved models
python crypto_trading_system.py D BTC 4,8h 1y  # Mode D — full pipeline (~90 min/horizon)
python crypto_trading_system.py F BTC 4,8h     # Mode F — strategy comparison (400h window)
python crypto_trading_system_v5.2.py D BTC 1,2,3,4,5,6,7,8h 1y  # V5.2 all horizons
python crypto_trading_system_v5.2.py G BTC     # V5.2 Mode G — horizon pair test (168h)

# Auto-trader
python crypto_revolut_trader.py --loop            # Live trading loop
python crypto_revolut_trader.py --dry-run --loop  # Signals only, no trades
python crypto_revolut_trader.py --status          # Show positions
python crypto_revolut_trader.py --balance         # Revolut X balance
```

**Telegram commands while trader is running:** `/stop` `/status` `/pause` `/resume` `/balance` `/sync` `/conf` `/chart BTC`

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
TRADING_FEE_BASE = 0.0009   # 0.09% Revolut X taker fee (0% maker)
SLIPPAGE = 0.0002           # 0.02% estimated slippage (market impact, spread)
TRADING_FEE = 0.0011        # total cost per trade (fee + slippage) — applied on BUY and SELL
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
| `crypto_trading_system.py` | V5 Production | DO NOT break — live trader imports from it. Includes DF mode + slippage model. |
| `crypto_trading_system_v6.py` | V6 Experimental | 12 literature enhancements behind `ENHANCEMENTS` flags (env var override). NOT production. |
| `testing_literature.py` | Test harness | A/B tests each V5.5 enhancement (Mode D BTC 4,8h 1y) — COMPLETE |
| `testing_literature_v2.py` | Test harness | A/B tests 12 V6 enhancements (Mode D BTC 4,8h 1y) |
| `testing_feature_stability.py` | Test harness | Feature stability test across BTC+ETH × 4h+8h |
| `crypto_revolut_trader.py` | Live | Multi-asset live trader + `/conf` `/chart` commands |
| `crypto_live_trader.py` | Live | Signal generation core — NOT run directly |
| `hardware_config.py` | Active | Auto-detects Desktop (26 workers) / Laptop (14 workers) |
| `crypto_trading_system_v15.py` | V15 | 15-min candles, horizons 15'–120', 1y max |
| `crypto_trading_system_v30.py` | V30 | 30-min candles, horizons 30'–240', 1y max |
| `cfd/ib_auto_trader.py` | Live | DAX CFD trader (Broly 1.2) |
| `cfd/ib_auto_trader_test.py` | Live | S&P 500 CFD overnight trader |

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

| Asset | Horizon | Models | Window | Accuracy | Features | Status |
|-------|---------|--------|--------|----------|----------|--------|
| BTC | 4h | RF+GB+LR | 100h | 80.2% | 31 | V5 2y |
| BTC | 8h | RF+GB | 150h | 84.7% | 20 | V5 2y |
| ETH | 4h | RF | 100h | 71.7% | 57 | V5 2y |
| ETH | 8h | RF+GB | 100h | 76.7% | 78 | V5 2y |
| XRP | 4h | RF+GB+LGBM | 100h | 71.7% | — | V5 |
| XRP | 8h | RF | 100h | 82.5% | — | V5 |
| DOGE | 4h | RF+LR | 72h | 78.1% | 30 | V5 |
| DOGE | 8h | LR | 100h | 74.0% | 79 | V5 WARNING |

**WARNING — DOGE 8h LR outputs 99-100% confidence on most signals.** Needs `CalibratedClassifierCV`. Do not enable DOGE for live trading until fixed.

## Current Trading Config

```json
{
  "BTC": { "strategy": "either_agree", "min_confidence": 66, "max_position_usd": 6000, "enabled": true },
  "ETH": { "strategy": "either_agree", "min_confidence": 75, "max_position_usd": 6000, "enabled": true },
  "XRP": { "strategy": "either_agree", "min_confidence": 75, "max_position_usd": 0, "enabled": false },
  "DOGE": { "strategy": "8h_only", "min_confidence": 90, "max_position_usd": 0, "enabled": false }
}
```

**BTC + ETH live trading at $6k each. XRP disabled (no improvement to portfolio). DOGE disabled (calibration issue).**

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

1. **V6 A/B tests** — `python testing_literature_v2.py`: 12 new literature enhancements + baseline, Mode D BTC 4,8h 1y. Results in `testing_literature_v2.csv`
2. **Re-run Mode DF for BTC** — `python crypto_trading_system.py DF BTC 4,8h 1y` (best_models CSV contaminated from V5.5 tests)
3. **Re-run Mode F for ETH** — `python crypto_trading_system.py F ETH 4,8h` to recalibrate confidence with slippage model
4. **Feature stability test** — `python testing_feature_stability.py`: BTC+ETH × 4h+8h
5. **XRP V5** — run `python crypto_trading_system.py D XRP 4,8h 1y` on laptop, then Mode F
6. **Weekly F runs** — re-run `F BTC 4,8h` and `F ETH 4,8h` weekly to refresh strategy
7. **Windows auto-start** — CryptoTrader scheduled task registered, needs reboot test

### Completed
- **V5.5 A/B tests** — DONE. Only slippage_model won → promoted to production. V5.5 archived.
- **V5.5 promotion** — slippage model (TRADING_FEE 0.0009 → 0.0011) applied to V5, V15, V30
- **DF combined mode** — `python crypto_trading_system.py DF BTC,ETH 4,8h 1y` runs Mode D then F
- **Telegram /conf + /chart** — added to crypto_revolut_trader.py
- **V6 created** — 12 literature enhancements (wavelet, fracdiff, GMM regime, XGBoost, sample weighting, entropy filter, tri-state labels, stacking, dynamic feature select, meta-labeling, adversarial validation, Kelly sizing)
- **Process priority** — Mode D/F runs at BELOW_NORMAL priority on Windows so trader always gets CPU
