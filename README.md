# Algo Trading Engine

Automated ML trading system for **crypto** (BTC, ETH, XRP, DOGE, SOL, LINK, ADA, AVAX, DOT) and **index CFDs** (DAX, S&P 500). Generates BUY/SELL/HOLD signals using ensemble ML models with walk-forward validation, temporal decay sample weighting, and embargo-corrected labels. Executes trades on Revolut X via Ed25519-signed API.

**Production:**
- **Ed V2** — Regime-switching trading: bull/bear detection with dynamic horizon selection, maker-order pricing (0% fee). Currently ETH-only.

**Testing (no live trader):**
- **Ein V1.0** — 15-minute candles, horizons 4-10 candles (1h-2h30)
- **Eli V1.0** — 30-minute candles, horizons 4-10 candles (2h-5h)

**Archived:** Doohan V1.7.1 (fixed-horizon, retired 2026-04), Deku, CASCA, V5 and earlier — see [archive/](archive/).

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Features (132)](#features-132)
- [Auto-Trader (Ed V2)](#auto-trader-ed-v2)
- [Optimizer Bot](#optimizer-bot)
- [Current Models](#current-models)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Key Constants](#key-constants)
- [Pending Work](#pending-work)
- [Version History](#version-history)

---

## Quick Start

```bash
# Activate venv first (required)
# Desktop:  C:\algo_trading\venv\Scripts\activate.bat
# Laptop:   C:\Users\Alex\algo_trading\venv\Scripts\activate.bat

# === Live trading (Ed V2 — production) ===
start_ed_v2.bat                                     # Auto-restart wrapper (recommended)
python crypto_revolut_ed_v2.py --loop               # Direct (no auto-restart)
python crypto_revolut_ed_v2.py --dry-run --loop     # Signals only, no trades
python crypto_revolut_ed_v2.py --status             # Show positions

# === Ed regime optimization ===
python crypto_trading_system_ed.py R BTC 5,6,7,8h              # Test all regime detectors
python crypto_trading_system_ed.py R BTC --replay 2880         # 4-month backtest
python crypto_trading_system_ed.py HRS ETH 5,6,7,8h --replay 1440   # Full pipeline (2-month)
python tools/pysr_discover_regime.py --bull 6 --bear 8         # PySR regime formula discovery

# === Optimizer bot (remote optimization via Telegram) ===
start_optimizer.bat                                 # Auto-restart wrapper
python crypto_optimizer_bot.py                      # Direct

# === Testing variants ===
python crypto_trading_system_ein.py HRS BTC 4,5,6,7,8,9,10    # 15-min candle test
python crypto_trading_system_eli.py HRS BTC 4,5,6,7,8,9,10    # 30-min candle test

# === Backtests / utilities ===
python backtest_full_month.py                       # 1-month backtest
python test_btc_horizons.py                         # BTC horizon sanity check
python test_btc_accuracy.py                         # BTC accuracy check
python sell_btc_now.py                              # Manual BTC liquidation
```

### Optimization Modes (Ed)

```bash
# Arguments are order-independent: MODE, ASSETS, HORIZONS can appear in any order.
python crypto_trading_system_ed.py P BTC 6h                # Mode P — PySR feature discovery
python crypto_trading_system_ed.py D BTC 6h                # Mode D — grid only
python crypto_trading_system_ed.py V BTC 6h                # Mode V — validate + Optuna refine
python crypto_trading_system_ed.py DV BTC 6h               # Mode DV — D then V
python crypto_trading_system_ed.py H BTC 5,6,7,8h          # Mode H — horizon sweep
python crypto_trading_system_ed.py R BTC 5,6,7,8h          # Mode R — regime backtest
python crypto_trading_system_ed.py HRS BTC 5,6,7,8h        # Full pipeline: H → R → S
python crypto_trading_system_ed.py --help                  # Show all modes
```

---

## Architecture

### Production File Chain

```
crypto_trading_system_ed.py  (Ed V1.0 — Modes P/D/V/H/S/R/HRS)
  └── hardware_config.py
crypto_revolut_ed_v2.py  (Ed V2 auto-trader — reads regime_config_ed.json)
  └── crypto_live_trader_ed.py  (regime-aware signal generation)
        └── crypto_trading_system_ed.py

# Optimizer
crypto_optimizer_bot.py  (Telegram bot for remote optimization)
  └── crypto_trading_system_ed.py  (spawned as subprocess)
```

### Index CFDs (separate system)

```
cfd/ib_auto_trader.py       (DAX CFD trader — Broly 1.2)
cfd/ib_auto_trader_test.py  (S&P 500 CFD overnight)
cfd/broly.py                (enhancement layer — regime detection)
```

---

## How It Works

### Core Concepts

| Concept | Detail |
|---------|--------|
| **Regime switching** | Bull/bear detector picks which model (horizon + confidence + position size) to use. Per-asset detector stored in `regime_config_ed.json` |
| **Variable horizons** | Each regime has its own optimal prediction horizon (5h, 6h, 7h, 8h), found by Mode H/HRS |
| **No model persistence** | Retrains from scratch every prediction. No .pkl files. Always uses latest market data |
| **Temporal decay** | Exponential sample weighting `w_i = gamma^(age)`. gamma=0.995 → half-life ~6 days, gamma=0.999 → ~29 days |
| **6-month data cap** | Training capped at 4,320 hours. Prevents stale data from diluting recent patterns |
| **Fee-aware labels** | `label = 1` when future return > 2× TRADING_FEE (0.22%) |
| **Label overlap embargo** | Adjacent rows share overlapping future windows. Fix: `EMBARGO_CANDLES = horizon`. **Backtesting only** — never in live signal generation. Per Lopez de Prado |
| **Walk-forward validation** | Train on last `window` hours → predict next candle → step forward (DIAG_STEP=36). No future leakage |
| **Ensemble voting** | Majority vote across 2-model combo. Confidence = average probability across models |
| **PySR symbolic regression** | Offline discovery of mathematical expressions. Anti-leakage: formulas discovered on months 12→6 ago only, never overlapping Mode D's evaluation window |

### ML Pipeline (Mode D)

1. **Data download** — Hourly candles (Binance via ccxt) + macro data (yfinance)
2. **Feature engineering** — 132 features (technical + macro + sentiment + cross-asset + PySR symbolic)
3. **Feature ranking** — LGBM gain importance ranks all features (~5 sec)
4. **Exhaustive grid search** — 3 combos × 6 windows × 6 features × 3 gammas = 324 evals
5. **3-fold rolling holdout** — Re-rank winners by out-of-sample performance with embargo
6. **Walk-forward evaluation** — Train → predict → step forward. Score by APF
7. **Save** — Top 6 candidates to `models/crypto_ed_best_models.csv`

### Mode V (Validate + Refine)

1. **Backtest** D candidates across 6 confidence thresholds (65-90%)
2. **Select top 3** for Optuna refinement (50 trials each)
3. **Production selection** using `return × (win_rate/100)` scoring
4. **Save** winner to `models/crypto_ed_production.csv`

### Mode R / HRS (Regime Selection)

Tests whether dynamically switching between horizons based on market regime outperforms a fixed single-horizon strategy. For example, using the 6h model in bull markets (SMA24 > SMA100) and the 8h model in bear markets. HRS = full pipeline (H → R → S) writing the winning bull/bear pair to `regime_config_ed.json`.

### Modes

| Mode | Purpose | When to Use |
|------|---------|-------------|
| **P** | PySR symbolic feature discovery | Before D/V to add symbolic features (~30-120 min) |
| **D** | Exhaustive grid optimization | After market regime change, or periodically |
| **V** | Validate + Optuna refine → writes production model | After Mode D |
| **DV** | D then V in one command | Standard workflow |
| **H** | Horizon sweep (D+V per horizon → compare → best) | Find optimal horizon per asset |
| **R** | Regime backtest (all detectors × horizon pairs) | Find best bull/bear regime split |
| **S** | Strategy / confidence sweep | Tune per-regime confidence |
| **HRS** | Full pipeline: H → R → S | Complete optimization, writes regime config |

---

## Features (132)

| Category | Count | Examples |
|----------|-------|---------|
| **Technical** | 51 | Log returns (1-240h), RSI, Bollinger Bands, ATR, ADX/DI, Garman-Klass vol, volatility ratios, Stochastic, spread ratios, SMA ratios, hour sin/cos |
| **Macro** | 40 | VIX (level, zscore, regime), DXY, S&P500/Nasdaq changes (1/5/10d), US10Y, EUR/USD, USD/JPY, Oil, Gold volatility |
| **Sentiment** | 25 | Fear & Greed Index (value, zscore, changes, MA, extreme flags) |
| **Cross-asset** | 16 | BTC/ETH/DAX/Nasdaq/S&P500 rolling correlation (10/30d), relative strength (5d) |
| **PySR symbolic** | variable | Auto-loaded from `models/pysr_{ASSET}_{H}h.json` if available; safe fallback if not |

---

## Auto-Trader (Ed V2)

### Startup

Run `start_ed_v2.bat` (recommended) — auto-restarts if the bot crashes. Or run directly: `python crypto_revolut_ed_v2.py --loop`.

### Loop Cycle

1. **Startup** — NTP clock sync → cancel stale orders → sync positions with Revolut X → Telegram notification → immediate scan
2. **Every hour** — Download data → detect regime → generate signals → apply per-regime config → execute trades → Telegram
3. **Every 5 min** — Position sync (detects manual trades, locked funds, ghost sells), model + config hot-reload
4. **Every 5 sec** — Poll Telegram for commands

### Maker Order Strategy (V2)

- Place limit orders at `bid + 0.01` (penny improvement) with `post_only` for guaranteed 0% maker fee
- Re-price every 10s, market fallback after 60-120s
- Stale order cleanup before each maker attempt (prevents fund locking)
- Handles 409 clock-drift errors by auto-correcting from server timestamp

### Telegram Commands

| Command | Action |
|---------|--------|
| `/status` | Current positions and PnL |
| `/balance` | Exchange account balance |
| `/regime` | Show current bull/bear state per asset |
| `/stop` | Graceful shutdown |
| `/pause` / `/resume` | Pause/resume trading |
| `/sync` | Force position sync |
| `/conf` / `/config` | Show current config |
| `/setup` | Inline button config editor |
| `/chart BTC` | Generate and send chart |
| `/help` | List all commands |

### Authentication

Asymmetric Ed25519 signature: `timestamp + method + path + body → base64 signature`.

---

## Optimizer Bot

Separate Telegram bot for remote model optimization. Run `start_optimizer.bat` (auto-restart) or `python crypto_optimizer_bot.py`.

Inline keyboard menus to select mode/assets/horizons. Sequential job queue with subprocess execution and real-time progress output. Uses separate bot token from the trader.

### Telegram Commands

| Command | Action |
|---------|--------|
| `/optimize` | Start optimization flow (Mode → Assets → Horizons → Confirm) |
| `/queue` | Show running/pending/completed jobs |
| `/cancel` | Cancel current job or menu flow |
| `/status` | Show current production models |
| `/results` | Show last results for an asset |
| `/help` | List commands |
| `/stop` | Stop the bot |

---

## Current Models

### Ed V2 Production (Regime-Switching)

| Asset | Regime | Horizon | Models | Window | Gamma | Features | Min Conf | Return | WR |
|-------|--------|---------|--------|--------|-------|----------|----------|--------|----|
| **ETH** | Bull | 6h | RF+XGB | 287h | 0.9979 | 10 | 90% | +7.4% | 87.5% |
| **ETH** | Bear | 7h | XGB+LGBM | — | — | — | 75% | +1.9% | 63.6% |

### Trading Config (Ed V2 — active)

```
ETH:  bull=6h@90%, bear=7h@75%   ($12,000 max)  enabled
BTC:  disabled (sold 2026-04-06, underperforming)
XRP:  disabled
SOL:  disabled
LINK: disabled
DOGE: disabled
ADA:  disabled
AVAX: disabled
DOT:  disabled
```

---

## Project Structure

```
engine/
├── crypto_trading_system_ed.py        # Ed V1.0 — Modes P/D/V/H/S/R/HRS
├── crypto_revolut_ed_v2.py            # Ed V2 auto-trader (reads regime_config_ed.json)
├── crypto_live_trader_ed.py           # Ed regime-aware signal generation
├── crypto_trading_system_ein.py       # Ein V1.0 — 15-min candle testing
├── crypto_trading_system_eli.py       # Eli V1.0 — 30-min candle testing
├── crypto_optimizer_bot.py            # Telegram bot for remote optimization
├── hardware_config.py                 # Auto-detect Desktop/Laptop config
├── download_macro_data.py             # Macro/sentiment/cross-asset downloader
├── pysr_discover_features.py          # Offline PySR discovery (historical window)
├── backtest_full_month.py             # 1-month backtest harness
├── test_btc_accuracy.py               # BTC accuracy diagnostic
├── test_btc_horizons.py               # BTC horizon comparison
├── sell_btc_now.py                    # Manual BTC liquidation
├── start_ed_v2.bat                    # Ed V2 trader launcher with auto-restart
├── start_optimizer.bat                # Optimizer bot launcher with auto-restart
├── git_push.bat                       # Git push helper
│
├── cfd/
│   ├── ib_auto_trader.py             # DAX CFD trader (Broly 1.2)
│   ├── ib_auto_trader_test.py        # S&P 500 CFD overnight
│   └── broly.py                      # Enhancement layer (regime detection)
│
├── tools/
│   ├── check_balance.py              # Exchange balance
│   ├── check_trades.py               # Trade history
│   ├── debug_price.py                # API price diagnostic
│   ├── revolut_x_test.py             # API connectivity test
│   ├── detect_hardware.py            # Hardware detection → config
│   ├── buy_btc.py                    # Manual BTC purchase
│   ├── backtest_regime_master.py     # Regime detector backtest (21 detectors × horizon pairs)
│   ├── pysr_discover_regime.py       # PySR regime formula discovery
│   ├── ib_test_connection.py         # IB broker connectivity test
│   └── tee_launcher.py               # Logging tee for .bat launchers
│
├── data/
│   ├── {asset}_hourly_data.csv       # Hourly OHLCV (Binance)
│   ├── macro_data/                   # VIX, DXY, S&P500, Fear&Greed, etc.
│   └── indices/                      # DAX, S&P500, SMI, CAC40 OHLCV
│
├── models/
│   ├── crypto_ed_production.csv      # Ed production model
│   ├── crypto_ed_best_models.csv     # Top 6 candidates per (asset, horizon)
│   ├── pysr_{ASSET}_{H}h.json        # PySR symbolic expressions
│   └── pysr_{ASSET}_{H}h_report.txt  # PySR discovery reports
│
├── config/                           # NOT in git — credentials + state
│   ├── regime_config_ed.json         # Ed: per-asset detector, bull/bear horizon+confidence+position
│   ├── revolut_x_config.json         # Exchange API key
│   ├── private.pem                   # Ed25519 signing key
│   ├── telegram_config.json          # Bot token (trader)
│   ├── telegram_optimizer_config.json # Bot token (optimizer — separate)
│   ├── position_ed_{ASSET}.json      # Ed position tracking
│   └── signal_log.csv                # Signal history (for /chart)
│
├── logs/                             # Auto-generated by start scripts
├── charts/                           # Backtest PNGs + interactive HTML
├── archive/                          # All legacy versions (Doohan, Deku, CASCA, V1.1-V1.7, etc.)
├── CLAUDE.md                         # Claude Code instructions
└── README.md
```

---

## Setup

### Hardware

One shared engine folder synced via Google Drive. Only the venv is local per machine.

| Machine | Engine Path | Venv | CPU | GPU |
|---------|-------------|------|-----|-----|
| Desktop (primary) | `G:\Autres ordinateurs\My laptop\engine\` | `C:\algo_trading\venv\` | i7-14700KF | RTX 4080 |
| Laptop | `G:\Autres ordinateurs\My laptop\engine\` | `C:\Users\Alex\algo_trading\venv\` | 16 cores | RTX 3070 Ti |

`hardware_config.py` auto-detects Desktop (26 workers) vs Laptop (14 workers). LGBM uses GPU (`device='gpu'`).

### Installation

```powershell
# Desktop
python -m venv C:\algo_trading\venv
C:\algo_trading\venv\Scripts\activate.bat
pip install -r "G:\Autres ordinateurs\My laptop\engine\requirements.txt"

# Laptop
python -m venv C:\Users\Alex\algo_trading\venv
C:\Users\Alex\algo_trading\venv\Scripts\activate.bat
pip install -r "G:\Autres ordinateurs\My laptop\engine\requirements.txt"
```

### Key Dependencies

`pandas`, `numpy`, `scikit-learn`, `lightgbm` (GPU), `xgboost`, `optuna`, `ccxt`, `yfinance`, `pynacl`, `cryptography`, `matplotlib`, `joblib`, `pysr`

---

## Key Constants

```python
# Trading costs
TRADING_FEE_BASE = 0.0009       # 0.09% Revolut X taker fee (0% maker)
SLIPPAGE = 0.0002               # 0.02% estimated slippage
TRADING_FEE = 0.0011            # total cost per trade (fee + slippage)

# Grid search (Mode D)
GRID_COMBOS = ['RF+LGBM', 'XGB+LGBM', 'RF+XGB']  # 3 viable combos
GRID_WINDOWS = [72, 100, 150, 200, 250, 300]
GRID_FEATURES = [5, 10, 15, 20, 25, 30]
GRID_GAMMAS = [0.995, 0.997, 0.999]
# Total: 3 × 6 × 6 × 3 = 324 evaluations per horizon

# Walk-forward
DIAG_STEP = 36                  # walk-forward step size
MAX_DIAG_HOURS = 4320           # 6-month data cap
EMBARGO_CANDLES = horizon       # label overlap fix (dynamic per horizon)

# Optuna refinement
DEKU_DEFAULT_TRIALS = 150
REFINE_TRIALS = 50              # Optuna refine trials per config
REFINE_TOP_N = 3                # top N D candidates to refine

# Validation
MODE_G_REPLAY_HOURS = 336       # 2-week backtest window
MODE_G_CONF_THRESHOLDS = [65, 70, 75, 80, 85, 90]
MIN_CONFIDENCE = 75             # global fallback (overridden per regime)
MIN_COMBO_SIZE = 2              # no solo models
MIN_TRADES = 8                  # reject unreliable configs
```

---

## Regime Detection — Literature Review

Honest assessment of how Mode R/HRS detectors map to peer-reviewed crypto research. Compiled 2026-04-07.

### What dominates academic literature

The overwhelming majority of peer-reviewed work on crypto regime detection uses **statistical models** (HMM, MSGARCH), not technical-indicator rules.

**Hidden Markov Models — most cited approach:**
- **Giudici & Abu Hashish (2020)**, *Quality and Reliability Engineering International* — foundational paper. 3-state HMM (bull/stable/bear) on Bitcoin daily log-returns. Diagonal covariance beats full.
- **Koki, Leonardos & Piliouras (2022)**, *Research in International Business and Finance* — non-homogeneous HMM with Pólya–Gamma augmentation. **3-state outperforms 2-state** on BTC/ETH/XRP.
- **Pennoni et al. (2022)**, *Economic Notes* — multivariate HMM across BTC/ETH/XRP/LTC for cross-asset regime dependence.

**Markov-Switching GARCH:**
- **Ardia, Bluteau & Rüede (2019)**, *Finance Research Letters* — MSGARCH on Bitcoin daily; 2-regime beats single-regime for VaR.
- **Caporale & Zekokh (2019)**, *Research in International Business and Finance* — MSGARCH on BTC/ETH/LTC/XRP.
- **Tan et al. (TV-MSGARCH)** — time-varying transition probabilities driven by **trading volume + Google search trends**. Beats fixed-transition out-of-sample.
- **Katsiampa (2017, 2019)** — MS-GARCH on Bitcoin volatility, identifies persistent high-vol regimes.

**Technical-rule paper (closest to our approach):**
- **Hudson & Urquhart (2021)**, *International Review of Financial Analysis* — tested 15,000+ technical trading rules on BTC/ETH/LTC/XRP. **MA-based rules generate statistically significant returns even after data-snooping correction.** This is the only direct academic validation for SMA-style detectors on crypto.

### Inputs they actually use

| Input | Frequency | Used by |
|---|---|---|
| Log-returns (primary) | daily | Giudici 2020, Koki 2022, Pennoni 2022, all MSGARCH |
| Realized volatility | daily | Ardia 2019, Caporale 2019, Katsiampa |
| Trading volume | daily | Tan et al. (TV-MSGARCH) |
| Google search trends | weekly | Tan et al., Urquhart |
| Macro (DXY, S&P, VIX) | daily | Chen et al. (VECM-MS), Bayesian MCMC paper |

**Notable absences from peer-reviewed crypto regime literature:**
- ❌ Moving-average crossovers as *regime detectors* (used in practitioner blogs only — except as trading rules in Hudson/Urquhart)
- ❌ RSI as a regime classifier (used as ML feature, never as regime label)
- ❌ Drawdown thresholds (`dd72>-3%` style) — **zero papers** use this for crypto regimes
- ❌ **Hourly/sub-daily regime detection** — almost everything is daily

### Key empirical findings

1. **3 states beat 2 states** consistently — bull/sideways/bear, not just bull/bear (Koki 2022). Our binary setup is academically suboptimal.
2. **Regime persistence is real** at daily frequency (self-transition >0.95). At hourly frequency this breaks down — part of why hourly regime detection is uncharted.
3. **Volatility is the strongest regime signal**, not return direction. Bear state in HMMs is consistently the *high-vol* state, regardless of return sign. Most robust finding in the literature.
4. **Out-of-sample improvements are modest** — typically Sharpe +0.2 to +0.5 over buy-and-hold, not the +50%-return improvements seen in our 2-month replays. Large-margin Mode R wins are an overfitting warning.
5. **Daily timeframe dominates** — there is essentially no peer-reviewed literature on hourly crypto regime detection. Our hourly approach is uncharted academic territory.

### Mapping our detectors to literature

| Our detector | Literature support |
|---|---|
| `sma24>sma72`, `sma24>sma100`, `sma48>sma100`, `sma48>sma200` | **Validated** — Hudson/Urquhart 2021 on crypto, Faber 2007, Brock/Lakonishok/LeBaron 1992 |
| `price>sma72`, `price>sma100`, `price>sma240`, `price>sma480` | **Validated** — same papers |
| `sma24>sma168`, `sma72>sma240`, `sma168>sma480` | **Validated** — same family, calendar-calibrated for hourly crypto |
| `macd>0` | **Weak** — widely tested (Chong/Ng 2008) but rarely beats raw MA crossovers |
| `dd48>-2%`, `dd48>-3%`, `dd72>-3%`, `dd72>-5%` | **No academic support.** Drawdown is a *risk metric* in literature. Pagan/Sossounov 2003 uses peak-trough rules but on monthly data with 20% thresholds. |

### Sources

- [Giudici & Abu Hashish (2020) — HMM regime changes in cryptoasset markets](https://onlinelibrary.wiley.com/doi/abs/10.1002/qre.2673)
- [Pennoni et al. (2022) — HMM cryptocurrency log-returns dependencies](https://onlinelibrary.wiley.com/doi/abs/10.1111/ecno.12193)
- [Koki, Leonardos & Piliouras — Bayesian HMM for crypto predictability](https://www.sciencedirect.com/science/article/abs/pii/S0275531921001756)
- [Caporale & Zekokh — Markov-Switching GARCH for cryptocurrencies](https://www.sciencedirect.com/science/article/pii/S027553191830669X)
- [Markov and Hidden Markov Models for Bitcoin (2024–2026 evidence)](https://www.preprints.org/manuscript/202603.0831)
- [Bitcoin Cycle through Markov Regime-Switching Model](https://www.mdpi.com/2673-4591/74/1/12)
- [Bitcoin Price Regime Shifts: Bayesian MCMC + HMM](https://www.mdpi.com/2227-7390/13/10/1577)
- [Regime switching forecasting for cryptocurrencies — Springer Digital Finance](https://link.springer.com/article/10.1007/s42521-024-00123-2)
- [A Quantile Spillover Markov Switching Model for Crypto Volatility](https://www.mdpi.com/2227-7390/13/15/2382)
- [Bitcoin Regime Detection: K-Means + HMM](https://jdmdc.com/index.php/JDMDC/article/view/57)
- [Applications of HMMs in Detecting Regime Changes in Bitcoin Markets](https://journalajpas.com/index.php/AJPAS/article/view/781)

---

## Pending Work

### TODO

**HIGHEST PRIORITY:**

0a. **Rally-cooldown BUY gate — fine-tune V6 + pick winner + wire to prod** — Core hypothesis already validated: after a large ETH rally, price tends to revert within 24h; blocking BUY for a cooldown window improves PnL. V6 (rr10≥5 OR rr24≥7, cd=24h) beat V0 on all 3 windows (+3.58 / +13.55 / −11.48 vs +1.11 / +2.24 / −16.70); 77–79% of skips were followed by a 24h price drop.
   - **Scope:** fine-tune V6's OR-of-two-rolling-returns structure inside current crisis regime (not re-open mode choice). Workflow: (1) test if sweep finds anything beating V6, (2) promote only relevant horizons, (3) wire to prod, (4) demote if decays.
   - **Sweep design (`audit_v6_v2.py`):** structure FIXED to `block BUYs for C hours when rr(Hshort)≥Tshort OR rr(Hlong)≥Tlong`. Horizon pairs = all 15 (Hshort<Hlong) from {10,12,14,16,18,24}h. Horizon-scaled thresholds: 10/12h→{3,4,5,6}; 14/16h→{4,5,6,7}; 18h→{5,6,7,8}; 24h→{5,6,7,8,9}. Cooldowns={10,12,18,24}h.
   - **Windows:** two non-overlapping 30d halves (H1=days 0–30 back, H2=days 30–60 back) + 60d reference. No 90d (straddles regime change the detector should handle).
   - **Scoring:** PnL%. Winner must beat V0 on BOTH halves. Trades + skipped-BUYs reported for eyeball churn-check only, no hard filter.
   - **Next step:** run sweep → review winner + churn → wire into [crypto_revolut_ed_v2.py process_asset](G:/Autres ordinateurs/My laptop/engine/crypto_revolut_ed_v2.py) BUY path (compute rr(Hshort) and rr(Hlong) at top-of-hour; if either ≥ threshold, set cooldown; block BUY while cd>0, SELL unaffected; persist cd state to disk).
   - **Artifacts (prior):** `backtest_rally_cooldown_multi.py`, `audit_v6.py`, `rally_cooldown_summary_{30d,60d,90d}.csv`, `rally_cooldown_stability.csv`. **Caches:** `data/eth_5m_backtest_90d.csv` (26,497 rows), `data/eth_sl_signals_90d.pkl` (2,186 hourly signals).

0b. **Remove `/optimize` + `/optstatus` from trader bot** — Trader should not launch optimizations; that belongs to the optimizer bot. Already hidden from `/help`; next step is to delete `_handle_optimize_command`, `_handle_optstatus_command`, dispatcher branches, and related globals in `crypto_revolut_ed_v2.py`.
1. **Restart optimizer bot** to load SV3 + Help buttons (commits a900d98, c0e674d).
2. **Run SV3 ETH `--replay 2880`** — Ed V3 joint H-sweep test. If results beat current Mode S winner, push to prod.
3. **RS ETH `--replay 2880` OOS** — running on Yoga since 06:57, report results when complete.

**Other:**
4. **Eli HRS BTC** — `python crypto_trading_system_eli.py HRS BTC 4,5,6,7,8,9,10` — 30-minute candle test
5. **Ein results review** — Check Ein (15min) BTC results from laptop run

### Completed (2026-04-16)
- **V7 rally-cooldown gate in production** (ETH) + Mode G optimizer + `/gate` Telegram command. Block BUYs for 30h after `rr_8h ≥ 3%` OR `rr_36h ≥ 5.5%`. State persists in position file.
- **Optimizer bot menu simplified** — 3-profile front door (Full Re-tune / Regime Refresh / Model Refresh) + PySR + Advanced submenu.
- **`/optimize` removed from trader bot** — Optimizer lives in its own bot now.
- **Bull/gate icons recolored blue** — `/status` and `/gate` now use 🔵 (matches BUY=🔵 / SELL=🔴 convention).
- **11 audit-pass bugs fixed** — Position file race (lock + atomic write), `regime_config_ed.json` non-atomic write (5 sites), partial-fill `usd_invested` mismatch, optimizer-bot blocking stdout (queue-based reader), confirm-button double-click, hold-shield naive-local time (UTC ISO), `'DVS'`/`SV3`/`BLOWOFF` dead code, silent feature drop in `generate_live_signal`, PySR regime detector silent 0.0 substitution. See `CLAUDE.md` for line-by-line breakdown.
- **Adaptive cooldown tested and rejected** (`compare_gate_adaptive.py`) — lifting V7 cooldown early on price reversion costs −8.8% to −23.7% PnL on 90d. Keep fixed 30h.
- **Chain-order verified** (`compare_chain_order.py`) — G-last vs joint optimization is noise-level over 6-fold walk-forward. Keep G-last.

### Completed (2026-04-14)
- **Stop-loss / profit-lock backtest (ETH prod, 30d, 5m res)** — **Verdict: keep prod as-is.** Ran 8 variants via `backtest_sl_variants.py`. Baseline A (no SL) won every dimension: +1.11% PnL / −8.71% DD. Disaster −7%/−10% (B, C) never fired → identical to baseline. Profit-lock D/E/F and trailing G/H all catastrophic (−11% to −20%). Scalping tiny locked gains chops big winners; full loss price still paid on bad setups via failsafe.
- **Hold Shield toggle** — `/hold` Telegram command + dynamic "🛡 Shield: ON/OFF" button. Persists per-asset to `regime_config_ed.json`. Test suite `test_hold_shield.py` passes.
- **Chart overhaul** — `/chart` accepts horizon (`/chart`, `/chart ETH`, `/chart 12h`, `/chart ETH 7d`). Markers clarified (cyan ▲ BUY / orange ▼ SELL + ✓/✗/⏳ badge), legend inside chart, horizon-scaled axis.
- **Manual `/buy` `/sell` maker commands** — Fresh-quote maker orders via Telegram with full instrumentation. Silent 30-min log stall traced to Windows stdout buffering (fixed with `flush=True`).
- **Telegram HTML fix** — `<` → `vs` in hold override message (HTTP 400 fix).

### Completed (2026-04-08 — soir)
- **Ed V3 (research)** — `crypto_trading_system_ed_v3.py` Mode S full joint H-sweep: 5 detectors × 8 horizon pairs × 49 conf combos = 1,960 evals/asset. Writes to `regime_config_ed_v3.json` (zero prod impact).
- **Telegram optimizer bot** — Added `SV3` button + `Help` button in mode menu.
- **Telegram trader** — Hourly update shows real detector name (e.g. `sma168>sma480`) instead of `named`; added `/help` line after date.

### Completed (2026-04-08)
- **BUG 1 — maker buy balance rounding** — Floor `buy_amount` to cents minus $0.01 safety margin (Revolut rejects qty×price > balance by even $0.01). Was causing market fallback on buys.
- **BUG 2 — maker sell post_only race** — SELL floor raised `bid+0.01` → `bid+0.02` with second guard after rounding; on `post_only` rejection, retry loop with fresh quote instead of market fallback. Root cause confirmed via 2 Revolut rejection emails (08:51:05 / 08:51:12). 7 cancelled sells this morning.
- **Maker window 60s → 120s** (40 attempts at 3s).
- **BUG 3 — `Unknown regime detector type: named`** — code fix already in fac33a4; trader restart pending.
- **Mode V ETH 6h vs 7h `--replay 4320` complete** — 7h winner +24.73% / 64% WR vs 6h +23.17% / 78% WR. Kept 7h.
- **ETH live trader verified** — `/regime` confirms named-detector branch and hot-reload.
- **ETH regime config cleanup** — ETH block clean; legacy keys on disabled assets left as-is.

### Completed (2026-04-07)
- **R→S detector handoff fix (Option C — full joint sweep)** — Extracted shared `_build_regime_indicators_and_detectors()` helper so Mode R + Mode S use a single source of truth for indicators and detector dict. Mode S rewritten as joint sweep over all 5 detectors × 49 conf combos = 245 evals. Winner written to config as `{type: "named", params: {name: <detector>}}`.
- **Named-detector branch wired into live trader** — `crypto_live_trader_ed.py` `_evaluate_detector` now dispatches `type=='named'` to `_evaluate_named_detector` with implementations for all 5 trimmed detectors (incl. Andersen-Bollerslev deseasonalized `vol_calm`).
- **ETH RS rerun with fixed pipeline** — Winner: `sma168>sma480` 7h@75% / 8h@85% → +60.72%, 66 trades, 65% WR, alpha +49.98% (`ed_v1_20260407_160841.log`). Top-15 entries dominated by the 3 literature-grounded detectors. Config updated and live.
- **Mode V `--replay` argument support** — `run_mode_v` / `_backtest_one_config` / CLI dispatcher now accept `replay_hours`. Telegram optimizer bot's `REPLAY_MODES` extended to include V/DV/DVS/S so the menu prompts for replay before launching (fixes "can't specify replay from phone").
- **Detector set trimmed 14 → 5** — Removed all 5 RSI detectors, all 4 drawdown detectors, `macd>0`, and 9 redundant SMA/momentum variants. Final set: `sma24>sma100`, `sma168>sma480`, `price>sma72`, `vol_calm`, `tsmom_672h`. Selection driven by Mode R rankings across all/top50/top100 cuts; the 3 literature-grounded newcomers (vol_calm, tsmom_672h, sma168>sma480) all ranked top-3.
- **Added literature-grounded detectors** — `vol_calm` (Andersen-Bollerslev intraday deseasonalized vol regime), `tsmom_672h` (Liu & Tsyvinski 2021 RFS direct crypto replication, 4-week lookback). SMA windows extended to 168/240/480h for hourly crypto calibration.
- **Mode R top-N expanded to 200** — `top_n` default changed from 15 to 200 in `crypto_trading_system_ed.py:4551` to allow full ranking visibility.
- **ETH-only trading** — BTC disabled and position sold (underperforming: 45% WR, avg loss > avg win over 1-month backtest). Full $12k allocation moved to ETH.
- **ETH HRS 2-month complete** — `HRS ETH 5,6,7,8h --replay 1440`. Results: bull=6h@90% (RF+XGB, +7.4%, 87.5% WR), bear=7h@75% (XGB+LGBM, +1.9%, 63.6% WR). Config updated automatically.
- **ETH position doubled** — $6k → $12k per trade (both regimes).

### Completed (2026-04-05)
- **Maker order pricing fix** — mid-price strategy (from 2026-04-03) still never filled: buys sat between bid/ask, sells chased price down. Fix: penny-improvement at `bid+0.01` for both buy and sell. `post_only` ensures 0% maker fee. Stale orders cancelled before each maker attempt, error body now logged on limit failures, maker window reduced to 60s.

### Completed (2026-04-03)
- **Ed V2 trader critical bug fixes** — 6 bugs fixed in `crypto_revolut_ed_v2.py`:
  1. **Clock drift** — Windows clock ahead of Revolut server → all API calls rejected (409). Fix: NTP sync on startup + auto-correct on 409. Updated 2026-04-13: 409 echo-back method replaced with NTP-based correction (echo-back re-applies stale offset on worsening drift) + periodic NTP sync every 5 min.
  2. **`get_balances()` silent failures** — returned `{}` on any API error. Fix: 3 retries, returns `None` on failure, logs full error.
  3. **Ghost sells** — sell flipped position to cash even when exchange returned 0 balance. Fix: position only updates if sell confirmed executed.
  4. **Sync blind to locked funds** — sync checked `total`, sell checked `available`. Fix: sync now detects `available != total` and cancels stale orders.
  5. **No stale order cleanup** — orphaned limit orders from crashed processes locked funds. Fix: `cancel_all_open_orders()` on startup.
  6. **Maker orders placed at ask** — never filled on quiet markets. Fix: place at mid-price, re-price every 10s, market fallback after 120s.

### Completed (2026-03-31)
- **ETH RS 2-month** — Mode R: sma24>sma100 bull=6h bear=8h +59.58% (74% WR). Mode S: 6h@85%/8h@65% → +70.01% (67% WR).
- **HRS R→S pipeline fix** — Mode R now writes winning horizons to config before Mode S runs.
- **LINK HRS complete** — Weak. Best: 5h +3.97%. Not worth trading.
- **Take-profit analysis** — No TP is best. TP=1% boosts WR (67→72%) but cuts return nearly in half.
- **No-embargo-in-live confirmed** — Per Lopez de Prado. Embargo for backtesting only.

### Completed (2026-03-29)
- **Ed V1.0 release** — Regime-switching system: bull/bear detection with dynamic horizon/confidence. Mode R backtest. External regime config.
- **Regime backtest tool** — `tools/backtest_regime_master.py`: tests all horizon pairs × regime detectors with monthly breakdown.
- **All 9 assets Mode H complete** — BTC, ETH, SOL, LINK, XRP, DOGE, ADA, AVAX, DOT. AVAX negative (skip).
- **Trailing stop / regime filter analysis** — Baseline signal exits beat all variants. Model signal quality is the edge.

---

## Version History

| Date | Milestone |
|------|-----------|
| **2026-04-16** | V7 rally-cooldown gate in production (ETH). Mode G optimizer + `/gate` Telegram command. Optimizer-bot menu simplified to 3 profiles + Advanced. `/optimize` removed from trader bot. 11 audit-pass bugs fixed across position state, config writes, fill accounting, timezone math, optimizer concurrency, and silent ML-feature drift. |
| **2026-04-08** | ETH live trader verified on new named-detector config (`sma168>sma480` 7h@75% / 8h@85%). Mode V 6m replay running for 6h vs 7h reliability comparison. |
| **2026-04-07** | **Detector trim + R→S fix + named-detector wiring.** 14→5 detectors (literature-grounded `vol_calm`, `tsmom_672h`, `sma168>sma480` added). Mode S rewritten as Option C joint sweep (5×49=245). Mode V `--replay` arg added end-to-end. ETH RS rerun: `sma168>sma480` 7h/8h → +60.72%, 65% WR. ETH-only, BTC sold, $12k allocation. |
| **2026-04-13** | Clock drift fix: NTP-based correction replaces broken echo-back method, periodic NTP sync every 5 min. Maker order bug fixes (partial fill balance, cancel verification, duplicate orders, locked funds). Noprod wrapper for safe experimentation. |
| **2026-04-05** | Maker order penny-improvement fix (`bid+0.01`, post_only). |
| **2026-04-03** | Ed V2 release — 6 critical bug fixes (clock drift, ghost sells, locked funds, stale orders, maker pricing). |
| **2026-03-31** | ETH RS / HRS R→S pipeline fix. LINK HRS dropped (weak). Take-profit analysis (no TP wins). |
| **2026-03-29** | **Ed V1.0 release.** Regime-switching system. Mode R regime backtest. PySR regime discovery. SOL disabled. |
| **2026-03-27** | BTC -7.5% selloff. Full Mode H re-run for BTC. All 9 assets Mode H complete. AVAX negative. |
| **2026-03-26** | V1.8 LSTM test — FAILED. SOL Mode H (8h winner, +22.43%). |
| **2026-03-25** | Telegram optimizer bot. PySR leakage fix (historical window). PySR promoted to production. |
| **2026-03-24** | Doohan V1.7.1 promoted to production. Variable horizon support. New Mode H. (Now archived 2026-04.) |
| **2026-03-23** | LGBM dominance proven. CPCV dropped. |
| **2026-03-22** | Telegram UX overhaul (candlestick charts, inline buttons). |
| **2026-03-21** | Multi-asset expansion — all 9 assets optimized. |
| **2026-03-20** | Deku promoted to production. 3-fold holdout validation. |
| **2026-03-18** | Deku release — Optuna TPE+Hyperband, XGBoost, APF scoring. |
| **2026-03-15** | CASCA release — profit factor scoring. Temporal decay. |
| **2026-03-10** | V5 production. Mode F strategy selection. |
| **2026-03-04** | Dual horizon (4h+8h). Derivative features. |
| **2026-02-22** | Broly 1.2 — IB auto-trader for DAX CFDs. |
| **2026-02-21** | Initial commit. |

### Evolution

Ed V2 (regime + maker orders) → Ed V1.0 → Doohan V1.7.1 → Doohan V1.6 → Deku → CASCA → V5 → V4 → V3. All legacy systems archived.
