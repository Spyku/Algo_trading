# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Automated ML trading system for crypto (BTC, ETH, XRP). Generates hourly BUY/SELL/HOLD signals using dual-horizon (4h + 8h) ensemble ML models with walk-forward validation and temporal decay sample weighting. Executes trades on Revolut X via Ed25519-signed API.

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
python crypto_trading_system.py D BTC 4,8h     # Mode D — full pipeline (6mo data cap)
python crypto_trading_system.py F BTC 4,8h     # Mode F — strategy comparison (400h window, generates chart)
python crypto_trading_system.py DF BTC,ETH 4,8h  # Mode DF — D then F in one command

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
crypto_trading_system.py  (V5 Cacarot production — Modes B/D/E/F/DF)
  └── hardware_config.py  (machine-specific model configs, n_jobs, GPU settings)

crypto_revolut_trader.py  (auto-trader — reads trading_config.json)
  └── crypto_live_trader.py  (signal generation library — NOT run directly)
        └── crypto_trading_system.py  (imports ASSETS, features, models, download/load/build)
```

### Key Concepts

- **Dual horizons:** 4h and 8h — completely independent models per horizon (different features, window, combo). Never mix them.
- **No model persistence:** System retrains from scratch every prediction. No .pkl files. Intentional design.
- **Temporal decay (Cacarot):** Exponential sample weighting `w_i = gamma^(age_in_hours)`. Per-model gamma stored in best_models.csv. Newest sample weight=1, oldest=gamma^(n-1). gamma=0.995 → half-life ~6 days, gamma=0.999 → ~29 days. gamma >= 1.0 disables decay (zero overhead).
- **6-month data cap:** Mode D and E cap training data at 4,320 hours (6 months). Not configurable — hardcoded as `MAX_DIAG_HOURS`.
- **Labels:** Relative to 168h rolling median return, not absolute price direction. `label = 1` means return > median.
- **Walk-forward validation:** Train on last `window` hours → predict next candle → step forward 72h. No future leakage.
- **Scoring (V5):** `accuracy × (1 + max(cum_return, 0) / 100)` — rewards being right AND making money. Judge by return, not accuracy.
- **Models:** RF, GB, LR, LGBM — all 15 combinations tested (solo + pairs + triples + quad).
- **Features:** 49 technical + 81 macro/sentiment/cross-asset = 130 total.

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
MAX_DIAG_HOURS = 4320       # 6 months data cap for Mode D and E
REPLAY_HOURS = 200          # Modes B, D signal replay
REPLAY_HOURS_F = 400        # Mode F — longer window for more trades in strategy selection
DIAG_STEP = 72
DIAG_WINDOWS = [48, 72, 100, 150, 200]       # horizons 5-8h
DIAG_WINDOWS_SHORT = [24, 48, 72, 100, 150]  # horizons 1-4h
```

---

## Version Files

| File | Status | Notes |
|------|--------|-------|
| `crypto_trading_system.py` | V5 Cacarot Production | DO NOT break — live trader imports from it. Includes DF mode, slippage model, temporal decay (gamma), 6mo data cap. |
| `crypto_trading_system_v6.py` | V6 Experimental | 12 literature enhancements behind `ENHANCEMENTS` flags (env var override). NOT production. |
| `archive/testing_literature.py` | Test harness | A/B tests V5.5 enhancements — COMPLETE, archived. Results in `archive/testing_literature.csv` |
| `testing_literature_v2.py` | Test harness | A/B tests 12 V6 enhancements (Mode D BTC 4,8h 1y) |
| `testing_cacarot_v1.5.py` | Test harness | Tests 15 V5.6-rejected features WITH decay. Isolated output: `models/testing_cacarot_v1.5_results.csv`, `charts/v1.5_test/` |
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

| Asset | Horizon | Models | Window | Accuracy | Return | Features | Gamma | Status |
|-------|---------|--------|--------|----------|--------|----------|-------|--------|
| BTC | 4h | RF+LGBM | 48h | 71.2% | +43.1% | 20 | 0.995 | Cacarot 6mo |
| BTC | 8h | LGBM | 200h | 80.7% | +27.1% | 111 | 0.996 | Cacarot 6mo |
| ETH | 4h | RF+LR | 48h | 83.1% | +18.1% | 40 | 0.999 | Cacarot 6mo |
| ETH | 8h | RF | 200h | 91.2% | +35.6% | 90 | 0.995 | Cacarot 6mo |
| XRP | 4h | RF+GB+LGBM | 100h | 71.7% | +202.3% | 15 | 1.0 | V5 2y |
| XRP | 8h | RF | 100h | 82.5% | +139.2% | 50 | 1.0 | V5 2y |
| DOGE | 4h | RF+LR | 72h | 78.1% | — | 30 | 1.0 | V5 2y |
| DOGE | 8h | LR | 100h | 74.0% | — | 79 | 1.0 | V5 2y WARNING |

**WARNING — DOGE 8h LR outputs 99-100% confidence on most signals.** Needs `CalibratedClassifierCV`. Do not enable DOGE for live trading until fixed.

## Current Trading Config

```json
{
  "BTC": { "strategy": "either_agree", "min_confidence": 90, "max_position_usd": 6000, "enabled": true },
  "ETH": { "strategy": "4h_only", "min_confidence": 60, "max_position_usd": 6000, "enabled": true },
  "XRP": { "strategy": "either_agree", "min_confidence": 75, "max_position_usd": 0, "enabled": false },
  "DOGE": { "strategy": "8h_only", "min_confidence": 90, "max_position_usd": 0, "enabled": false }
}
```

**BTC + ETH live trading at $6k each (Cacarot configs from Mode F 2026-03-16). XRP disabled. DOGE disabled (calibration issue).**

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

### Desktop TODO
1. **V6 A/B tests** — `python testing_literature_v2.py --resume` — RUNNING. Laptop hangs on 8h diagnostic (loky deadlock with 14 workers), must run on desktop.

### Laptop TODO (can run in parallel)
2. **V1.5 feature test** — `python testing_cacarot_v1.5.py D BTC 4,8h` then F — tests 15 V5.6-rejected features with decay
3. **XRP Cacarot** — `python crypto_trading_system.py D XRP 4,8h` then Mode F
4. **V15 first run** — `python crypto_trading_system_v15.py D BTC 4,8h`
5. **V30 first run** — `python crypto_trading_system_v30.py D BTC 4,8h`

### Either machine
6. **Feature stability test** — `python testing_feature_stability.py`
7. **Weekly F runs** — re-run `F BTC 4,8h` and `F ETH 4,8h` weekly
8. **Windows auto-start** — CryptoTrader scheduled task registered, needs reboot test
9. **Restart live trader** — pick up new BTC (either_agree @90%) and ETH (4h_only @60%) configs

### Completed
- **Cacarot release** — DONE (2026-03-16). Temporal decay (gamma per asset+horizon), 6mo data cap, Mode F charts.
- **BTC Cacarot Mode DF** — DONE (2026-03-16). BTC: `either_agree` @90%, return +42.1%.
- **ETH Cacarot Mode DF** — DONE (2026-03-16). ETH: `4h_only` @60%, return +62.8%.
- **Re-run Mode DF for BTC** — DONE (2026-03-15). Fresh models in best_models.csv.
- **Re-run Mode F for ETH** — DONE (2026-03-15). ETH: `8h_only`, min_confidence=85%.
- **Restart crypto_revolut_trader** — DONE (2026-03-15). Running with fresh models + 5-min hot-reload for config+models+positions.
- **Laptop venv install** — DONE. PyWavelets + xgboost installed.
- **V5.5 A/B tests (testing_literature.py)** — DONE, archived. 8 tests × BTC 4h+8h 1y. Results:
  - **slippage_model** — WINNER: 8h return +98.4% vs baseline +74.0%, consistent improvement → promoted to production
  - **extended_diag_step** — 8h accuracy 85.6% (best) but fewer trades (22 vs 32)
  - **on_chain_features** — 8h return +106.1% but accuracy dropped 5 points (73.3% vs 78.3%)
  - **gb_calibration** — neutral, slight regression
  - **triple_barrier_label** — worse on all metrics (63.9%/65.0% accuracy)
  - **purged_embargo** — much worse (58.7%/62.8% accuracy)
  - **derivatives_features** — only 2 trades, meaningless
  - Baseline: 4h 80.0% +57.2%, 8h 78.3% +74.0%
- **V5.5 promotion** — slippage model (TRADING_FEE 0.0009 → 0.0011) applied to V5, V15, V30
- **DF combined mode** — `python crypto_trading_system.py DF BTC,ETH 4,8h 1y` runs Mode D then F
- **Telegram /conf + /chart** — added to crypto_revolut_trader.py
- **V6 created** — 12 literature enhancements (wavelet, fracdiff, GMM regime, XGBoost, sample weighting, entropy filter, tri-state labels, stacking, dynamic feature select, meta-labeling, adversarial validation, Kelly sizing)
- **Process priority** — Mode D/F runs at BELOW_NORMAL priority on Windows so trader always gets CPU
- **V5.6 archived** — 20 literature features tested (Garman-Klass, ADX, Hurst, MFI, skewness, kurtosis, etc.). Only ADX + Garman-Klass proved useful → added directly to V5 production. V5.6 file archived.
