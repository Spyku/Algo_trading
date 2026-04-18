# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Automated ML trading system for crypto (BTC, ETH, XRP, DOGE, SOL, LINK, ADA, AVAX, DOT). Generates hourly BUY/SELL/HOLD signals using ensemble ML models with walk-forward validation, temporal decay sample weighting, and embargo-corrected labels. Variable horizon per asset (5h, 6h, 7h, 8h — optimized via Mode H). Executes trades on Revolut X via Ed25519-signed API.

**Production:**
- **Ed V2** — Regime-switching trading (`crypto_trading_system_ed.py` + `crypto_revolut_ed_v2.py`). Dynamic bull/bear horizon selection via external config (`config/regime_config_ed.json`). Maker-order pricing at `bid+0.01` with `post_only` for 0% fees. Mode R regime backtest. Currently ETH-only.

Doohan V1.7.1, Deku, CASCA and all prior versions archived (Doohan retired 2026-04, others 2026-03-24). See `archive/`.

**Owner:** Alex, Lausanne, Switzerland (CET/CEST timezone)

---

## Engine Reference Card (2026-04-17)

**Base knowledge for current models + future analysis.** Built from live audit of `crypto_ed_production.csv` (48 models) + CLAUDE.md / README history. Do not delete — update in place when new evidence arrives.

### Feature grades (5=best, 1=worst/dead)

Grades = frequency of feature selection across the 48 production models.

**Technical — winners (5):** `hour_cos` (73%), `price_to_sma100h` (73%), `logret_120h` (69%), `sma20_to_sma50h` (58%), `adx_14h` (50%).
**Technical — strong (4):** `vol_ratio_12_48`, `logret_240h`, `logret_72h`, `logret_24h`, `volatility_48h`, `gk_volatility_48h`, `volatility_12h`, `spread_120h_12h` (31-40%).
**Technical — useful (3):** `plus_di_14h`, `spread_240h_24h`, `atr_pct_14h`, `bb_position_20h`, `price_accel_24h`, `minus_di_14h`, `price_to_sma50h` (15-25%).
**Technical — marginal (2):** `zscore_50h`, `spread_24h_4h`, `hour_sin`, `stoch_k_14h`, short-window logret/spread variants (8-13%).
**Technical — dead (1):** `rsi_14h`, `logret_{2h,5h,8h,12h,48h}`, `dow_sin`, `intraday_range`, `volume_ratio_h`, `price_accel_4h`, `spread_48h_4h`, `price_to_sma20h` (≤6%).

**Macro — useful (4):** `m_nasdaq_chg1d` (46%), `m_sp500_chg1d` (40%), `m_vix_chg1d` (38%).
**Macro — marginal (3):** `m_dxy_chg1d` (15%).
**Macro — dead (1):** all 5d/10d/vol variants, `m_gold_*` (except 1d), `m_oil_*`, `m_eurusd_*`, `m_us10y_*`. Only 1-day equity/VIX changes earn their keep.

**Cross-asset — strong (4):** `xa_dax_relstr5d` (44%), `xa_eth_usd_relstr5d` (38%), `xa_sp500_relstr5d` (38%).
**Cross-asset — useful (3):** `xa_nasdaq_relstr5d` (29%), `xa_dax_corr10d` (27%), `xa_sp500_corr10d` (17%).
**Cross-asset — dead (1):** `xa_eth_usd_corr30d`, `xa_nasdaq_corr10d`, `xa_btc_usd_relstr5d`, `xa_eth_usd_corr10d` (≤8%). **Rule of thumb: relative-strength > correlation.**

**Sentiment — useful (3):** `fg_chg5d` (23%).
**Sentiment — marginal (2):** `fg_chg10d`, `fg_zscore`.
**Sentiment — dead (1):** `fg_value`, `fg_chg1d`, `fg_ma5d`. **GDELT (7+ features): DEAD — downloaded but never loaded in `build_all_features()`.**

**PySR (all earn a spot, 4):** `pysr_5` (42%), `pysr_4` (35%), `pysr_2` (33%), `pysr_3` (29%), `pysr_1` (21%). Genetic programming found real nonlinear combinations.

**On-chain — DEAD (1):** MVRV, SOPR, hashrate, active addresses, exchange netflow, fees, tx count — download skeleton exists in `download_macro_data.py`, no loader in `build_all_features()`. **Biggest unrealised opportunity.**

**Derivatives — DEAD as feature (1):** `_funding_rate` loaded with underscore to exclude from feature matrix. Only available as regime gate (not active). BTC-only source.

### Horizon status

Production CSV: 5h/6h/8h = 9 models each, 7h = 8, 10h = 4, 12h = 4, **4h = 2, 14h = 2**, 16h = 1.

- **5h, 6h, 7h, 8h** — core band, default sweep in Ed ([line 5428](crypto_trading_system_ed.py#L5428)). Current production: ETH bull=6h, bear=8h.
- **4h — broken.** All Mode D candidates negative post-embargo. Label overlap dominates — horizon too close to embargo window. The 2 rows in CSV are pre-embargo legacy. **Do not revive.**
- **10h, 12h — under-tested.** In default Mode R/S sweep ([line 4818](crypto_trading_system_ed.py#L4818)) but no dedicated head-to-head vs 5-8h. **Action: run `HRS BTC 5,6,7,8,10,12` on 2880h.**
- **14h — barely tested.** Not in default sweeps. Only 2 legacy rows. Worth including in BTC sweep since BTC trends on longer timescales than ETH.

### What works (in production, with evidence)

- **Ed V2 regime-switching (2026-04-07).** ETH bull=6h@85% / bear=8h@65%, HRS replay showed bull +7.4% @ 87.5% WR, bear +1.9% @ 63.6% WR.
- **V7 rally-cooldown BUY gate (2026-04-16).** `h_short=8, t_short=3%, h_long=36, t_long=5.5%, cd=30h`. Grid of 49,716 configs; winner: H1 +10.42% / H2 +18.01% / 60d +31.84% / worst DD −3.63%. Plateau ridge = robust.
- **Named detector `tsmom_672h`.** 28-day time-series momentum (Liu & Tsyvinski 2021). Winner of Mode S joint sweep.
- **Hold-shield (0.5% PnL / 10h max).** Prevents premature loss-cutting; 10h failsafe caps disaster paths at −2 to −3%.
- **Maker orders (bid+0.01, post_only, 180s/10s reprice).** 0% fees. Multiple bug fixes landed (penny-improvement, post-only race, buy-balance rounding).
- **Embargo fix (`train_end = i - horizon`).** Killed inflated pre-embargo APFs (5-26×) — realistic 1-3× post-fix.

### What doesn't work (tested and abandoned)

- **All stop-loss / take-profit / profit-lock / trailing-stop variants.** 8+ variants tested; baseline (no SL) won every dimension. Scalping sub-0.3% winners surrenders the fat tail; hold-shield already caps losses. **Only exception worth considering: −5% to −7% disaster brake as free insurance.**
- **Raising bear min-confidence.** Blocks contrarian bear-rally trades — every variant lost. Current bull=85%/bear=65% is correct.
- **LSTM as ensemble partner.** Voted randomly; RF+LGBM ≡ LSTM+LGBM. Single strong tree-based learner dominates.
- **V1.7.2 regularization.** Wash; signal/noise too low for fine regularization tuning.
- **PySR for regime labels (not features).** 58% accuracy — too weak. Keep PySR for feature synthesis only.
- **4h horizon.** Structurally broken post-embargo.
- **Adaptive rally-cooldown lift.** −8.8 to −23.7% vs fixed 30h on 90d.
- **Blow-off filters (RSI>70, %B>1, etc.).** Best filter +0.58pp — not actionable. V7 rally-cooldown works because `rr_8h ≥ 3%` catches the move earlier than distribution cutoffs.
- **BTC trading.** Disabled 2026-04-06 (45% WR, avg loss > avg win on 1m OOS). Needs re-evaluation.
- **Multi-timeframe fusion, TabPFN, CPCV, GB/RF solo, GB+LR.** All dominated in tests.

### Regime-conditional asymmetries (important for future models)

- **ETH horizon asymmetry:** bull=6h / bear=8h. Longer horizon in bear = more confirmation needed in volatile regime.
- **ETH confidence asymmetry:** bull=85% / bear=65%. Counterintuitive but battle-tested — bear's low-confidence signals are mean-reversion setups that *should* fire.
- **No per-regime SL/TP found useful.** All rules that override SELL timing lose.
- **No per-regime feature set found useful yet.** Open question: should bear use more macro, bull more technical? Not tested.

### Key lessons for future model design

1. **Model signal quality IS the risk edge.** Don't override SELL timing with price-based rules.
2. **Relative-strength > correlation** for cross-asset features.
3. **1-day macro changes > multi-day macro variants.** Longer windows smooth out the signal.
4. **Feature count should be dynamic (Set D) not static.** Per-horizon selection via LGBM importance beats fixed sets A or B.
5. **Ensemble dilutes a strong base model.** Don't stack weak models onto LGBM.
6. **Trigger early, don't wait for distribution cutoffs.** `rr_8h ≥ 3%` beats `RSI > 70`.
7. **Horizon must clear embargo window with margin.** 4h is at the edge and fails; 5h+ works.
8. **On-chain features are the biggest untapped source** — all coded, none loaded.

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
# Arguments are order-independent: MODE, ASSETS, HORIZONS can appear in any order.
# Run --help for full usage.

# === Ed (production) — regime-switching, dynamic bull/bear horizon selection ===
python crypto_trading_system_ed.py P BTC 6h                    # Mode P — PySR feature discovery (~30-120 min)
python crypto_trading_system_ed.py H BTC 5,6,7,8h              # Mode H — horizon sweep
python crypto_trading_system_ed.py DV BTC 6h                   # Mode DV — grid + validate
python crypto_trading_system_ed.py D BTC 6h                    # Mode D — grid optimization only
python crypto_trading_system_ed.py V BTC 6h                    # Mode V — re-validate existing D results
python crypto_trading_system_ed.py R BTC 5,6,7,8h              # Mode R — regime backtest (all detectors)
python crypto_trading_system_ed.py R BTC --replay 2880         # Mode R — 4-month replay
python crypto_trading_system_ed.py HRS ETH 5,6,7,8h --replay 1440  # Full pipeline (H→R→S), writes regime config
python crypto_trading_system_ed.py --help                       # Show all modes, options, examples

# === Auto-trader (Ed V2 — production) ===
start_ed_v2.bat                                     # Auto-restart launcher (recommended)
python crypto_revolut_ed_v2.py --loop               # Live trading loop
python crypto_revolut_ed_v2.py --dry-run --loop     # Signals only, no trades
python crypto_revolut_ed_v2.py --status             # Show positions

# === Optimizer bot (remote optimization via Telegram) ===
python crypto_optimizer_bot.py                      # Start optimizer bot (separate from trader bot)

python tools/pysr_discover_regime.py --bull 6 --bear 8          # PySR regime formula discovery

# === Ein (15-minute candles) — testing only, no trader ===
python crypto_trading_system_ein.py D BTC 6                     # Mode D — horizon 6 candles (= 1h30)
python crypto_trading_system_ein.py H BTC 4,5,6,7,8,9,10       # Horizon sweep (all candle horizons)
python crypto_trading_system_ein.py HRS BTC 4,5,6,7,8,9,10     # Full pipeline

# === Eli (30-minute candles) — testing only, no trader ===
python crypto_trading_system_eli.py D BTC 6                     # Mode D — horizon 6 candles (= 3h)
python crypto_trading_system_eli.py H BTC 4,5,6,7,8,9,10       # Horizon sweep
python crypto_trading_system_eli.py HRS BTC 4,5,6,7,8,9,10     # Full pipeline

# === Legacy regime backtest (pre-Ed) ===
python tools/backtest_regime_master.py                         # 2-month default, all horizons
python tools/backtest_regime_master.py --months 4              # 4-month backtest
python tools/backtest_regime_master.py --horizons 6,8          # only test 6h and 8h (fast)
python tools/backtest_regime_master.py --bull 6 --bear 8       # fix pair, compare regimes only
python tools/backtest_regime_master.py --regimes sma,rsi       # filter regime families
python tools/backtest_regime_master.py --no-combos             # single-horizon baselines only
python tools/backtest_regime_master.py --asset ETH             # test other assets
```

**Telegram commands (Ed V2 trader):** `/stop` `/status` `/pause` `/resume` `/balance` `/sync` `/conf` `/config` `/setup` `/help` `/chart BTC` `/regime` (show current bull/bear state per asset) `/gate [ASSET on\|off\|clear]` (V7 rally-cooldown gate)

**Telegram commands (Optimizer bot):** `/optimize` (interactive menu) `/queue` `/cancel` `/status` `/results` `/help` `/stop`

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

Doohan V1.7.1 retired 2026-04. Deku, CASCA, V1.1-V1.7 archived 2026-03-24. All legacy systems in `archive/`.

### Key Concepts

- **Regime switching:** Bull/bear detector picks which model (horizon + confidence + position size) to use per asset. Per-asset detector + bull/bear pair stored in `regime_config_ed.json`. Mode H/HRS sweeps horizons to find the best per regime.
- **Label overlap embargo (V1.7+):** Adjacent rows share overlapping future windows used for labeling. Fix: `EMBARGO_CANDLES = horizon`, so `train_end = i - horizon`. Without this, training data leaks label information. Pre-embargo APFs were inflated 5-26×; post-embargo realistic range is 1.0-3.0. **IMPORTANT: Embargo applies to backtesting/cross-validation ONLY, never to live signal generation.** In live trading, all labels use known past prices — no future leakage exists. Removing training rows wastes valid data. Per Lopez de Prado: purging/embargo apply to train/test splits for evaluation, not to live deployment.
- **No model persistence:** System retrains from scratch every prediction. No .pkl files. Intentional design.
- **Temporal decay:** Exponential sample weighting `w_i = gamma^(age_in_hours)`. Per-model gamma stored in production CSV. Newest sample weight=1, oldest=gamma^(n-1). gamma=0.995 → half-life ~6 days, gamma=0.999 → ~29 days.
- **6-month data cap:** Mode D caps training data at 4,320 hours (6 months). Hardcoded as `MAX_DIAG_HOURS`.
- **Labels (fee-aware):** `label = 1` when future return > 2×TRADING_FEE (0.22%). Set by `LABEL_MODE = 'fee_aware'`.
- **Walk-forward validation:** Train on last `window` hours → predict next candle → step forward. DIAG_STEP=36. No future leakage.
- **Scoring:** APF (Adjusted Profit Factor) = `raw_PF / buyhold_PF`. Normalizes against market regime.
- **Exhaustive grid (Mode D):** 3 combos × 6 windows × 6 features × 3 gammas = 324 evals per horizon. Saves top 6 candidates.
- **3-fold rolling holdout:** Train on fold 1 (60%), re-rank winners by out-of-sample performance across 3 folds with embargo. Reduces overfitting.
- **Refined-only production selection:** Mode V backtests D candidates to pick top 3 for Optuna refine (50 trials each), then selects production model from refined configs only. D candidates consistently underperform refined versions.
- **MIN_COMBO_SIZE=2:** Solo models removed. Prevents overconfidence from uncalibrated single-model predictions.
- **MIN_TRADES=8:** Optuna objective returns 0 for trials with <8 trades.
- **Models:** RF, GB, XGB, LR, LGBM — 3 viable combos: XGB+LGBM, RF+LGBM, RF+XGB (dead combos RF+GB, RF+LR, GB+LR dropped).
- **Features:** 51 technical + 81 macro/sentiment/cross-asset + PySR symbolic = 132+ total. LGBM importance ranking (~5 sec). PySR features auto-loaded from `models/pysr_{ASSET}_{H}h.json` if available; safe fallback if not.
- **PySR symbolic regression:** Mode P runs offline discovery (`pysr_discover_features.py`), saves expressions to JSON. Production loads them as computed columns. Anti-leakage: PySR formulas are discovered on months 12→6 ago only, never overlapping with Mode D's last-6-month evaluation window.
- **PySR anti-leakage checks:** `_check_pysr_leakage()` runs early in Mode D, V, and Refine. If PySR JSON lacks `discovery_method == "historical"`, all PySR features are stripped before the run starts. Mode V also blocks production CSV writes for leaky PySR configs.
- **Model hot-reload:** Trader checks production CSV every 5 minutes.
- **Regime-switching horizons:** `backtest_regime_master.py` tests whether switching between horizons based on market regime (e.g., SMA24>SMA100 = bull -> use 6h model, bear -> use 8h model) outperforms a fixed single-horizon strategy. Evaluates multiple regime detectors (SMA, RSI, volatility, etc.) and horizon pairs.

### Data Flow

```
data/{asset}_hourly_data.csv           <- price data (Binance via ccxt)
data/macro_data/*.csv                  <- VIX, DXY, S&P500, Fear&Greed, etc. (yfinance)
models/crypto_ed_best_models.csv       <- top 6 candidates per (asset, horizon) from Mode D
models/crypto_ed_production.csv        <- Ed production model (written by Mode V)
config/regime_config_ed.json           <- Ed: per-asset regime detector, bull/bear horizon+confidence+position
config/position_ed_{ASSET}.json        <- Ed position tracking
config/revolut_x_config.json           <- Revolut X API key
config/private.pem                     <- Ed25519 signing key
config/telegram_config.json            <- Telegram bot token (trader)
config/telegram_optimizer_config.json  <- Telegram bot token (optimizer bot — separate bot)
```

**Config files are NOT in git.** `config/` is in `.gitignore`. Never push credentials.

---

## Key Constants

```python
# Shared
TRADING_FEE_BASE = 0.0009   # 0.09% Revolut X taker fee (0% maker)
SLIPPAGE = 0.0002           # 0.02% estimated slippage (market impact, spread)
TRADING_FEE = 0.0011        # total cost per trade (fee + slippage) — applied on BUY and SELL
MIN_CONFIDENCE = 75         # global fallback only — overridden per asset by Mode V/H
MAX_DIAG_HOURS = 4320       # 6 months data cap for Mode D
REPLAY_HOURS = 200          # Modes B, D signal replay
REPLAY_HOURS_S = 400        # Mode S — longer window for more trades in strategy selection

# Ed (production) — same grid params
GRID_COMBOS = ['RF+LGBM', 'XGB+LGBM', 'RF+XGB']  # 3 viable combos
GRID_WINDOWS = [72, 100, 150, 200, 250, 300]       # 250h sweet spot for most assets
GRID_FEATURES = [5, 10, 15, 20, 25, 30]            # feature counts to test
GRID_GAMMAS = [0.995, 0.997, 0.999]                # gamma values
DIAG_STEP = 36                                      # walk-forward step
DEKU_DEFAULT_TRIALS = 150                           # Optuna trial count
REFINE_TRIALS = 50                                  # Optuna refine trials per config
REFINE_TOP_N = 3                                    # top N D candidates to refine
MODE_G_REPLAY_HOURS = 336                           # 2-week backtest window
MODE_G_CONF_THRESHOLDS = [65, 70, 75, 80, 85, 90]  # confidence thresholds to test
EMBARGO_CANDLES = horizon                           # label overlap fix (dynamic per horizon)
```

---

## Version Files

| File | Status | Notes |
|------|--------|-------|
| `crypto_trading_system_ed.py` | **Production** | Ed V1.0: Regime-switching (1h candles). Modes P/D/V/H/S/R/HRS. Embargo-fixed grid (3×6×6×3=324 evals) + 50-trial Optuna refine + PySR symbolic features. Reads/writes `crypto_ed_production.csv` and `regime_config_ed.json`. |
| `crypto_revolut_ed_v2.py` | **Live** | Ed V2 auto-trader — maker orders (0% fee) with market fallback. Penny-improvement pricing: buy/sell at `bid+0.01`. `post_only` ensures maker. Stale order cleanup, NTP clock sync, locked funds detection. Reads `regime_config_ed.json`. |
| `crypto_live_trader_ed.py` | **Live** | Ed signal generation — regime-aware. `detect_regime()` + `generate_regime_signal()`. Reports current market price (not label-shifted). |
| `start_ed_v2.bat` | **Live** | Launches Ed V2 trader with auto-restart + log tee. Auto-detects Desktop/Laptop venv. |
| `crypto_optimizer_bot.py` | **Live** | Telegram bot for remote optimization. Inline keyboard menus. Sequential job queue, subprocess execution. Separate bot token (`config/telegram_optimizer_config.json`). Below-normal priority. Spawns `crypto_trading_system_ed.py`. |
| `start_optimizer.bat` | **Live** | Launches optimizer bot with auto-restart + log tee. |
| `hardware_config.py` | Active | Auto-detects Desktop (26 workers) / Laptop (14 workers) |
| `download_macro_data.py` | Active | Downloads VIX, DXY, S&P500, NASDAQ, Fear&Greed, etc. |
| `pysr_discover_features.py` | Active | Offline PySR discovery. Historical window (months 12→6 ago) to avoid leakage with Mode D. Outputs `models/pysr_{ASSET}_{H}h.json` with `discovery_method: "historical"`. |
| `backtest_full_month.py` | Active | 1-month backtest harness. |
| `test_btc_horizons.py` | Active | BTC horizon comparison sanity check. |
| `test_btc_accuracy.py` | Active | BTC accuracy diagnostic. |
| `sell_btc_now.py` | Utility | Manual BTC liquidation script. |
| `crypto_trading_system_ein.py` | **Testing** | Ein V1.0: 15-minute candles. Horizons 4-10 candles (1h-2h30). Grid windows 12h-120h. No trader yet. |
| `crypto_trading_system_eli.py` | **Testing** | Eli V1.0: 30-minute candles. Horizons 4-10 candles (2h-5h). Grid windows 12h-120h. No trader yet. |
| `tools/pysr_discover_regime.py` | Active | PySR regime formula discovery. Historical window. Anti-leakage. |
| `tools/backtest_regime_master.py` | Active | Hand-crafted regime detector backtest. 21 detectors × all horizon pairs. |
| `cfd/ib_auto_trader.py` | Live | DAX CFD trader (Broly 1.2) |
| `cfd/ib_auto_trader_test.py` | Live | S&P 500 CFD overnight trader |
| `crypto_trading_system_doohan.py` (+ all `_doohan*`, `crypto_revolut_doohan.py`, `crypto_live_trader_doohan.py`, `start_trader.bat`, `start_ed.bat`, `crypto_revolut_ed.py`) | **Archived** | Doohan V1.7.1 retired 2026-04. Replaced by Ed V2 (regime-switching). |

All Doohan, Deku, CASCA, V1.1-V1.7, and legacy backtests in `archive/`.

---

## Strategies

| Strategy | Logic |
|----------|-------|
| `Xh_only` | Single horizon (e.g., `6h_only`) — used when `horizon` is set in trading config |
| `both_agree` | BUY when both horizons agree; SELL when either says SELL |
| `either_agree` | BUY when either says BUY; SELL when either says SELL |

**Mode HRS writes the best bull/bear horizon + confidence to `regime_config_ed.json`. The Ed V2 trader reads the configured regime detector and switches between bull/bear models per cycle.**

---

## Current Regime Config (Ed V2 — 2026-04-08)

```json
{
  "ETH": { "detector": "named:sma168>sma480", "bull": "7h@75%/$12k", "bear": "8h@85%/$12k", "enabled": true },
  "BTC": { "enabled": false, "note": "sold 2026-04-06, underperforming over 1-month backtest" },
  "XRP": { "enabled": false },
  "SOL": { "enabled": false },
  "LINK": { "enabled": false },
  "DOGE": { "enabled": false },
  "ADA": { "enabled": false },
  "AVAX": { "enabled": false },
  "DOT": { "enabled": false }
}
```

ETH HRS 2-month (2026-04-07) initially picked bull=6h@90% / bear=7h@75%. After R→S handoff fix + Option C joint detector sweep (2026-04-07 evening), ETH RS rerun selected detector `sma168>sma480` with bull=7h@75% / bear=8h@85% → Mode S +60.72%, 66 trades, 65% WR. 6h vs 7h reliability comparison still pending Mode V replay 4320h run.

## Ed Backtest Results (2026-03-31)

### BTC (4 months, B&H: -24.93%)
- **Winner:** sma48>sma200 bull=7h@95% bear=8h@90% → **+50.35%**, 78 trades, 69% WR
- Best baseline: 8h_only +21.55%

### ETH — 2-month RS (B&H: -23.93%)
- **Mode R winner:** sma24>sma100 bull=6h bear=8h → +59.58%, 38 trades, 74% WR
- **Mode S winner:** bull=6h@85% bear=8h@65% → **+70.01%**, 57 trades, 67% WR
- Best baseline: 8h_only +47.21%
- R→S pipeline fix confirmed: R updated config to 6h/8h before S ran

### ETH — 4-month RS (B&H: -33.30%)
- **Mode R winner:** rsi>45 bull=6h bear=7h → +89.82%, 91 trades, 68% WR
- Note: 4mo prefers 6h/7h while 2mo prefers 6h/8h. Bull=6h consistent across both.

### SOL (4 months)
- **Winner:** sma_cross bull=6h@95% bear=8h@90% → +31.60%, 79 trades, 61% WR

### LINK (4 months) — WEAK, not worth trading
- Best horizon: 5h +3.97% (only positive). All others negative.

---

## Critical Rules

1. **Never modify `crypto_trading_system_ed.py` without testing first.** It is production. The live trader imports from it by exact filename.
2. **CSV merge logic:** When saving to production CSV, always filter by BOTH coin AND horizon.
3. **`generate_signals()` needs `feature_override`:** When feature_set is D, caller must pass `feature_override=config['optimal_features'].split(',')`.
4. **All timestamps in live trader use Europe/Zurich** via `_to_local()` helper.
5. **`regime_config_ed.json` has `detector` + `bull`/`bear` blocks per asset** — set by Mode HRS. Each block holds horizon, min_confidence, max_position_usd. Global `MIN_CONFIDENCE=75` is only a fallback.
6. **SSL fix on Windows:** `ssl._create_unverified_context()` applied in `crypto_revolut_ed_v2.py` and `crypto_live_trader_ed.py`.
7. **Production scoring:** Mode V/H selects best model using `return × (win_rate/100)` for positive returns, raw return for negatives. All candidates (D + refined) compete equally. This favors consistent winners over high-return low-win-rate configs.
8. **Leakage check before any production promotion:** Before writing to production CSV or pushing code that affects live trading, verify no data leakage exists (PySR, feature engineering, or otherwise). For PySR: confirm `discovery_method == "historical"` in JSON metadata. Enforced in code by `_check_pysr_leakage()`.
9. **NEVER add embargo in live signal generation.** Embargo (`train_end = i - horizon`) is for backtesting/cross-validation ONLY. In `generate_live_signal()` the training window must be `df.iloc[train_start:i]` — all available data. Per Lopez de Prado: purging/embargo apply to evaluation splits, not live deployment.
10. **Ed V2 maker order rules:** place limits at `bid+0.01` with `post_only`, re-price every 10s, market fallback at 60-120s. Always cancel stale orders before placing new ones (prevents fund locking). NTP clock sync on startup + every 5 min + on 409 errors.
11. **Clock drift correction uses NTP, not echo-back.** The 409 error response echoes back the request timestamp, not the server's time. Using it for correction re-applies the same stale offset on worsening drift. Fix (2026-04-13): `_sync_clock_ntp()` on every 409 + periodic sync every 5 min in the main loop.
12. **Stale config cleanup:** ETH block in `regime_config_ed.json` is now clean (no top-level legacy keys). SOL/LINK/XRP/DOGE/ADA/AVAX/DOT still have legacy `strategy`/`horizon`/`min_confidence`/`max_position_usd` at top level but are all `enabled: false` — left as-is intentionally.

---

## Pending Work

### TODO

**HIGHEST PRIORITY:**

0a. ~~**Rally-cooldown BUY gate**~~ — **DONE 2026-04-16.** V7 `(h_short=8, h_long=36, t_short=3%, t_long=5.5%, cd=30h)` in production via `_update_rally_cooldown()` / `_is_rally_cooldown_active()` in `crypto_revolut_ed_v2.py`. See Completed 2026-04-16 section for full details. Mode G added for re-optimization.

0b. ~~**Remove `/optimize` and `/optstatus` from trader bot**~~ — **DONE 2026-04-16.** Deleted `_handle_optimize_command` + `_handle_optstatus_command` + `_optimize_proc`/`_optimize_lock` globals + dispatcher branches + unused `subprocess` import from `crypto_revolut_ed_v2.py`. Optimizer lives in its own bot.
1. **Investigate signal nondeterminism (demoted 2026-04-17)** — Original claim: three RS ETH 1440h reruns (7 Apr 15:24 / 16:08, 9 Apr 16:03) produced winners +49.98% / +60.72% / +52.36% from "same script + same window". Re-audit on 2026-04-17 weakened the evidence: run 3 had a real commit (`fac33a4` R->S handoff fix) between it and runs 1/2; Google Drive version history for `crypto_trading_system_ed.py` shows edits on 7 Apr around the runs (v25-27 timestamped late afternoon Europe time), so runs 1/2 may not have shared identical code either. Logs from those dates no longer exist (earliest log = 10 Apr). Not actionable until reproduced with a controlled back-to-back rerun on pinned code + fixed `--replay` end date. Close unless reproduced.
2. **SV3 ETH `--replay 2880` full grid** — Now lower priority since V3 joint sweep is in production Mode S. The 8 Apr SV3 run was 1440h with only 8 horizon pairs — full-grid still pending but may be redundant.

**Other:**
4. **Eli HRS BTC** — `python crypto_trading_system_eli.py HRS BTC 4,5,6,7,8,9,10` — 30-minute candle test
5. **Ein results review** — Check Ein (15min) BTC results from laptop run

### Completed (2026-04-16)
- **11 audit-pass bugs fixed**:
  1. **(HIGH) Position file race** — `load_position()` / `save_position()` in `crypto_revolut_ed_v2.py` now share a `_position_lock` and write atomically (temp + `os.replace`). Telegram thread + main loop were both touching the same JSON without sync. Adds `JSONDecodeError` fallback to cash state.
  2. **(HIGH) `regime_config_ed.json` non-atomic write** — Added `_atomic_write_json()` helper in `crypto_trading_system_ed.py`; replaced all 5 raw `json.dump` writes (Modes S, T×2, R, G). Mid-write crash or chained-mode race no longer corrupts prod config.
  3. **(HIGH) `usd_invested` mismatch on partial fill** — `crypto_revolut_ed_v2.py` lines 1020 (auto-buy) and 1603 (manual `/buy`) now compute `usd_invested = filled_size * fill_price` from actual order data instead of the requested amount. PnL math no longer overstates basis on partial fills.
  4. **(MED) Optimizer bot blocking stdout** — `_run_job()` in `crypto_optimizer_bot.py` now reads subprocess stdout via a daemon thread feeding a `queue.Queue`, polling with a 1s timeout. `/cancel` reaches the worker even if the optimizer stalls or goes silent. Heartbeat progress edits keep the Telegram message alive during quiet phases.
  5. **(MED) Confirm-button double-click** — `opt_confirm` in `crypto_optimizer_bot.py` checks `_menu_state['step'] == 'confirm'` and nullifies the step before `_enqueue_job()`. Fast double-tap can no longer queue duplicate jobs.
  6. **(MED) Hold-shield naive-local time** — `crypto_revolut_ed_v2.py` now stores `entry_time` as UTC ISO 8601 (`2026-04-16T01:00:00Z`) via `_now_utc_iso()`. Hours-held math uses `datetime.now(timezone.utc) - _parse_entry_time_utc(...)`. Parser still accepts the legacy naive-local format for the existing ETH position. Display sites use `_format_entry_time_local()` to convert ISO back to readable local time. DST flips no longer skew the shield by 1h.
  7. **(LOW) Dead `'DVS'` mode** — Removed from `REPLAY_MODES` set in `crypto_optimizer_bot.py`. No button emitted it; no `MODES` entry existed.
  8. **(LOW) Mode P horizon menu dead branch** — Collapsed the misleading `if/else` where both arms called `_show_horizon_menu()`. Comment promised a skip that was never implemented.
  9. **(LOW) `SV3` / `BLOWOFF` dead branches in `_run_job()`** — Removed two unreachable mode branches plus the now-unused `SCRIPT_PATH_V3` constant.
  10. **(LOW) Silent feature drop in `generate_live_signal()`** — `crypto_live_trader_ed.py` now prints a warning when configured features are missing from `all_cols` (partial-match case), and a separate warning when zero match and we fall back to `FEATURE_SET_A`. Production model/feature drift is no longer silent.
  11. **(LOW) PySR regime detector silent 0.0 substitution** — `_evaluate_pysr_detector()` in `crypto_live_trader_ed.py` now collects missing/NaN feature names, logs them, and returns `True` (bull) instead of evaluating the formula with substituted zeros. Bull is the safer default since bull horizons run higher confidence thresholds.
- **`/optimize` + `/optstatus` removed from trader bot** — Handlers were still wired despite being removed from `/help`. Dropped the whole block in `crypto_revolut_ed_v2.py` plus its dispatcher branches and the `subprocess` import (only used by that handler). Optimizer is launched from its own bot.
- **Bull/bear + gate ON icons recolored blue** — `/status` bull icon 🟢→🔵 at line 1404, `/gate` ON state 🟢→🔵 at line 2565. Matches the BUY=🔵 / SELL=🔴 convention used everywhere else.
- **V7 rally-cooldown gate in production** — BUY gate wired into `crypto_revolut_ed_v2.py`. Winner from `audit_v6_v3.py` 49,716-config grid: block BUYs for 30h when `rr_8h ≥ 3%` OR `rr_36h ≥ 5.5%`. Top by `score_dd_aware` (12.40) in a `plateau_score=1.00` ridge — every ±1-step neighbor also passed STRICT (beats V0 on H1 AND H2 AND 60d). Sweep perf: H1 +10.42% / H2 +18.01% / 60d +31.84% / worst DD −3.63%. Params in `regime_config_ed.json → ETH.rally_cooldown`. State `rally_cooldown_until` persisted in `position_ed_v2_ETH.json` (survives restart).
- **Gate trigger-detection bug fix** — Original wiring ran trigger check only inside the BUY-while-cash branch → gate never fired while invested. Split into `_update_rally_cooldown()` (every tick, unconditional) + `_is_rally_cooldown_active()` (BUY-time check). Matches backtest semantics.
- **Mode G added** — New mode in `crypto_trading_system_ed.py`: runs the rally-cooldown sweep programmatically and writes `regime_config_ed.json → rally_cooldown` for each enabled asset. STRICT pick rule first (beats_3of3 AND plateau≥0.7, tiebreak by `score_dd_aware`). Chained variants: G, RSG, HRSG, DVRSG, HRSTG.
- **Default replay window → 2 months (1440h)** — `crypto_trading_system_ed.py` Modes D/V/R/S/T now default to 1440h when `--replay` omitted. Matches user's typical entry.
- **`/gate` Telegram command** — Per-asset rally-cooldown toggle in trader bot. `/gate` (status + buttons), `/gate on|off`, `/gate ETH on|off|clear`. The `clear` action wipes `rally_cooldown_until` as a one-time override. Added to `/help`.
- **Optimizer bot menu simplified** — `_show_mode_menu()` replaced with 3-profile front door: 🔧 Full Re-tune (HRS) / 🔄 Regime Refresh (RS) / ⚡ Model Refresh (DV) / 🔬 PySR. Full 12-mode grid moved behind `Advanced ▸` button with `◂ Back` return. Pattern: letters alone are debug tools; chains are real workflows.
- **Chain-order verified (one-off)** — `compare_chain_order.py` 6-fold rolling walk-forward (30d train / 10d test / 10d step): Path B (joint model+gate search) beat Path A (G-last, current) by mean +0.49% / median +0.35% PnL across folds — noise-level, 3 wins / 1 loss / 2 ties. Verdict: keep G-last in production, ordering immaterial. Single 60/30 split earlier showed +7.92% for B but was one fold of cherry-picked data.
- **Adaptive gate tested and rejected** — `compare_gate_adaptive.py` tested "lift cooldown early if price reverts to pre-rally level". On 30d: `V7b lift ≤ 36h-base` never fired (0 early lifts over 30d — reversion-to-base condition not met). On 90d: adaptive variants all LOSE vs fixed 30h (−8.8% to −23.7% PnL). Post-rally reversion is partial; the extra cooldown hours past reversion are where bad BUYs cluster. Keep fixed 30h.

### Completed (2026-04-15)
- **Maker window tuned for fill reliability** — Extended `_execute_maker_order` defaults from 120s/3s → 180s/10s after observing SELL fill asymmetry (BUYs filled in 1–29 reprices, SELLs took 23–31 with 2 MARKET fallbacks over past week). Slower reprice reduces cancel/repost churn that was costing queue priority on the SELL side. Symmetric change: BUY also benefits (fewer churn cancels at `bid+0.01`).
- **HRS ETH 6,7,8h TODO closed** — Mode D 2026-04-10 already validated GDELT (iran_vol_zscore ranked #9 on 8h). Prod is on the 4mo OOS winner `sma168>sma480 6h@90%/8h@90%` (+96.27%). Strictly-HRS rerun skipped: 1440h has 11pp nondeterminism (TODO #1) and would not improve on current prod config.

### Completed (2026-04-14)
- **Stop-loss / profit-lock backtest (ETH prod, 30d, 5-min res)** — **Verdict: keep prod as-is (no SL).** Ran 8 variants via `backtest_sl_variants.py`. Baseline A (no SL) won every dimension: +1.11% PnL / −8.71% DD. Disaster brakes −7% (B) and −10% (C) never fired in window → identical to baseline (free but unused). Profit-lock variants (D/E/F) and trailing HWM (G/H) all catastrophic: −11% to −20% PnL. Mechanism: scalping +0.15–0.28% gains on 31–49 "wins" chops big winners; losing setups still hit Shield-failsafe at −2% to −3%, so full loss price paid with no winner upside. F (tightest lock, +0.3%/+0.15%) worst at −20%. Artifacts: `backtest_sl_variants.py`, `backtest_sl_variants_summary.csv`, `backtest_sl_variants_trades.csv`. Revolut X API supports `tpsl`/`conditional` server-side SL via extending `place_*` helpers in `crypto_live_trader_ed.py` (`order_configuration: { conditional: {...} }`) but no longer pursued — user decided not to add insurance-only SL.
- **Hold Shield toggle** — `/hold` Telegram command + dynamic "🛡 Shield: ON/OFF" main button. Toggles `hold_shield` flag per-asset; persists to `regime_config_ed.json`. Shield gates SELL signals by blocking when `hold_shield=True AND PnL < min_sell_pnl_pct AND held < max_hold_hours`. Test suite `test_hold_shield.py` (8 sections, all pass). Commits eb5ea9f, 5ae895b.
- **Chart overhaul** — `/chart` now accepts horizon arg (6h..30d): `/chart`, `/chart ETH`, `/chart 12h`, `/chart ETH 7d`. Unambiguous markers (cyan ▲ BUY / orange ▼ SELL, correctness as ✓/✗/⏳ badge instead of color), legend always visible inside chart, horizon-scaled candles/axis/figure size. Correctness: ≥0.3% move in predicted direction within 4 candles.
- **Manual `/buy` `/sell` maker commands** — Telegram-triggered maker orders with fresh bid/ask quote, position update, full instrumentation (`print(..., flush=True)` + try/except + Telegram error popup). Silent 30-min death on 2026-04-14 morning traced to Windows stdout block-buffering — fixed by flush=True. Commits f522d1b, 8938f86.
- **Telegram HTML fix** — Replaced `<` with `vs` in hold override message (was breaking Telegram parse_mode=HTML with HTTP 400). Commit f440316.

### Completed (2026-04-10)
- **GDELT geopolitical features added** — `download_macro_data.py` now fetches GDELT DOC 2.0 data (iran_vol, iran_tone, geopolitical_tone) with rate-limit handling. `_compute_gdelt_features()` generates 15 features (raw, zscore, chg4h, chg24h, spike) per GDELT column. Wired into `build_all_features()` via hourly merge. Feature count: 51 base + 101 macro/sentiment/cross-asset/geopolitical = 152 total. GDELT features ranked #9-12 in LGBM importance on 6h/8h horizons (iran_vol_zscore #9/157 on 8h).
- **V3 joint sweep ported to production Mode S** — Replaced sequential R→S (greedy: R locks horizons at ≥90% conf → S sweeps conf only, 245 combos) with full joint sweep (detector × bull_h × bear_h × bull_conf × bear_conf, 3,920 combos with 4 horizons). Discovers global optimum across all dimensions simultaneously. Source: `crypto_trading_system_ed_v3.py`.
- **Mode D `--replay` parameter** — Replaces hardcoded `MAX_DIAG_HOURS = 6*30*24`. Available in CLI and Telegram bot (D added to `REPLAY_MODES`).
- **Telegram bot menu cleanup** — Removed BLOWOFF and SV3 buttons (noise reduction). Updated S label to "Joint Sweep (V3)", RS to "Regime + Joint Sweep". Added DVRS button. Time estimates updated (S: 60min, RS: 90min, HRS: 150min).
- **4 critical bugs fixed** — (1) GDELT CSV timezone: strip tz before save + `tz_convert(None)` on load for tz-aware data. (2) `_merge_hour` leaked into `all_cols` after GDELT merge. (3) Empty results IndexError in Mode S (`winner = results[0]` with no guard). (4) `_load_macro_csv()` crash on tz-aware CSV.
- **Iran ceasefire confirmed as cause of bad late results** — Apr 8 2026 Iran/Hormuz ceasefire caused +8% ETH rally, pure geopolitical event not model deficiency. Motivated adding GDELT features.
- **Mode D ETH 6,7,8h --replay 1440 run** — GDELT features validated. 152 features total. iran_vol_zscore ranked in top-12 importance.

### Completed (2026-04-09)
- **SV3 ETH 1440h** (`optimizer_20260408_155122.log`, finished 8 Apr 22:51 via Telegram). Winner: `vol_calm 7h@80%/6h@80%` → +57.68% / 63 trades / 75% WR / alpha +46.90%. Written to `regime_config_ed_v3.json` (research only — no prod impact).
- **RS ETH 1440h rerun** (`ed_v1_20260409_160324.log`). Winner: `sma168>sma480 7h@80%/6h@85%` → +52.36% / 62 trades / 71% WR / alpha +49.25%. Used to confirm RS↔V3 structural parity and to surface nondeterminism (see TODO #2).
- **Momentum-decay / blowoff filter sweep ETH 2880h** (`optimizer_20260409_024812.log`). Baseline +10.32%; best filter `A_6h ≥5% force_sell` → +10.91% (+0.58pp). Not actionable.
- **Prod confirmed** — `regime_config_ed.json` ETH = `sma168>sma480 6h@90%/8h@90%` from 4mo OOS sweep (`rs_eth_oos_4mo_20260408_065746.log`, +96.27%). Defensible: longer window damps the seed jitter that plagues 1440h sweeps.

### Completed (2026-04-08 — soir)
- **Ed V3 (research)** — `crypto_trading_system_ed_v3.py` Mode S full joint H-sweep: 5 detectors × 8 horizon pairs `(6,6)(6,7)(6,8)(7,7)(7,8)(8,8)(7,6)(8,7)` × 49 conf combos = 1,960 evals/asset. Writes to `regime_config_ed_v3.json` (zero prod impact).
- **Telegram optimizer bot** — Added `SV3` button (S V3 Joint H-Sweep) + `Help` button in mode menu.
- **Telegram trader** — Hourly update shows real detector name (e.g. `sma168>sma480`) instead of `named`; added `/help` line after date.

### Completed (2026-04-08)
- **BUG 1 fix — maker buy balance rounding** — Floor `buy_amount` to cents minus $0.01 safety margin before passing to maker buy (Revolut rejects when qty×price > balance by even $0.01).
- **BUG 2 fix — maker sell post_only race** — SELL price floor raised from `bid+0.01` to `bid+0.02` with second guard after rounding; on `post_only` rejection, retry loop with fresh quote instead of immediate market fallback. Confirmed via 2 Revolut rejection emails (08:51:05 / 08:51:12).
- **Maker window extended 60s → 120s** (40 attempts).
- **BUG 3 — `Unknown regime detector type: named`** — already fixed in `crypto_live_trader_ed.py` (commit fac33a4); needs trader restart to load.
- **6h vs 7h decision** — Mode V `--replay 4320` complete: 7h winner D #1 XGB+LGBM w=300 γ=0.999 f=25 @70% → +24.73%, 66 trades, 64% WR vs 6h Refined #1 +23.17%, 78% WR. Stayed on 7h (higher raw return).
- **ETH live trader verified** — `/regime` confirms named-detector branch firing, hot-reload working.
- **ETH config cleanup** — ETH block clean; disabled assets left as-is.

### Completed (2026-04-07)
- **R→S detector handoff fix (Option C)** — Extracted shared `_build_regime_indicators_and_detectors()` helper so Mode R + Mode S use the same indicator/detector dict. Mode S rewritten as joint sweep over all 5 detectors × 49 conf combos = 245 evals. Winner written to config as `{type: "named", params: {name: <detector>}}`.
- **Named-detector branch wired into live trader** — `crypto_live_trader_ed.py` `_evaluate_detector` now dispatches `type=='named'` to `_evaluate_named_detector` with implementations for all 5 detectors (incl. Andersen-Bollerslev deseasonalized `vol_calm`).
- **ETH RS rerun with fixed pipeline** — `sma168>sma480` 7h@75% / 8h@85% → +60.72%, 66 trades, 65% WR, alpha +49.98% (`ed_v1_20260407_160841.log`). Config updated and live.
- **Mode V `--replay` argument support** — `run_mode_v` / `_backtest_one_config` / CLI dispatcher now accept `replay_hours`. Telegram optimizer bot's `REPLAY_MODES` extended to include V/DV/DVS/S so the menu prompts for replay before launching.
- **Detector set trimmed 14 → 5** — Removed all RSI (5), drawdown (4), `macd>0`, and 9 redundant SMA/momentum variants. Final: `sma24>sma100`, `sma168>sma480`, `price>sma72`, `vol_calm`, `tsmom_672h`.
- **Literature-grounded detectors added** — `vol_calm` (Andersen-Bollerslev deseasonalized intraday vol), `tsmom_672h` (Liu & Tsyvinski 2021 RFS). SMA windows extended to 168/240/480h.
- **Mode R top_n default → 200** (`crypto_trading_system_ed.py:4551`).
- **ETH-only trading** — BTC disabled and position sold (45% WR, avg loss > avg win over 1-month backtest). Full $12k → ETH.
- **ETH HRS 2-month** — bull=6h@90% (RF+XGB, +7.4%, 87.5% WR), bear=7h@75% (XGB+LGBM, +1.9%, 63.6% WR).
- **ETH position doubled** — $6k → $12k per trade both regimes.

### Completed (2026-04-05)
- **Maker order penny-improvement fix** — `bid+0.01` for both buy and sell with `post_only`. Stale orders cancelled before each maker attempt. Maker window reduced to 60s.

### Completed (2026-04-03)
- **Ed V2 release** — 6 critical bug fixes in `crypto_revolut_ed_v2.py`: clock drift (NTP sync + 409 auto-correct), `get_balances()` silent failures, ghost sells, locked funds blind sync, stale order cleanup, maker order pricing.

### Completed (2026-03-29)
- **BTC Ed regime fully optimized** — sma48>sma200, bull=7h@95%, bear=8h@90% → +50.35% over 4 months (78 trades, 69% WR). Ed live on BTC.
- **Mode S implemented** — Regime confidence sweep (7×7=49 combos). HRS/DVRS/RS combo modes. Both_agree removed from Ed. Mode H no longer picks winner (R does).
- **PySR regime discovery — FAILED** — Tested forward48, sma48_200, forward72 labels. Best accuracy 58% (too weak). sma48>sma200 hand-crafted detector confirmed as winner.
- **Ed Telegram display fixed** — Shows bull/bear horizons + confidence instead of old strategy/both_agree.
- **Ed V1.0 release** — Regime-switching system. Dynamic bull/bear horizon selection via `regime_config_ed.json` (not hardcoded). Mode R regime backtest (16 detectors × all horizon pairs). PySR regime discovery script. Separate production CSV. Runs alongside Doohan. Telegram `/regime` command.
- **All 9 assets Mode H complete** — DOGE (5h), ADA (7h), AVAX (all negative), DOT (7h) completed. Every asset now has production models.
- **SOL disabled from Doohan** — Turned off from live trading.
- **Trailing stop analysis** — Tested 0.25-1.0% trailing stops + profit targets + regime filter on 336h replay. Baseline signal exits beat all variants for both BTC (+$826) and ETH (+$207). Model signal quality is the edge.
- **Dynamic confidence analysis** — Tested raising min_confidence in bearish regimes. All variants lost money vs baseline — blocked winning contrarian trades.
- **Start scripts with tee logging** — `start_trader.bat`, `start_optimizer.bat`, `start_ed.bat` with Unicode-safe tee, auto-restart, venv auto-detect.
- **Optimizer bot Unicode fix** — PySR Julia output crashed the bot with charmap encoding error. Fixed with ASCII fallback.

### Completed (2026-03-27)
- **LINK Mode H horizon sweep** — 5 horizons (4h-8h). Winner: 8h RF+LGBM +7.77%, 14 trades, 86% WR, conf>=90%. Config updated.
- **BTC Mode H re-run with PySR** — 2 horizons (5h/6h). Winner: 6h XGB+LGBM +8.78%, 26 trades, 85% WR, conf>=70%. Config updated.
- **Bat file venv auto-detect** — `start_trader.bat` and `start_optimizer.bat` now auto-detect Desktop vs Laptop venv path. Previously hardcoded to Desktop path, broke on Laptop.

### Completed (2026-03-26)
- **V1.8 LSTM test — FAILED** — Tested LSTM as classifier in grid: LSTM solo (0 valid results, all failed), LSTM+LGBM (identical to RF+LGBM), LSTM+XGB (identical to RF+XGB). LSTM votes randomly, adds nothing. Confirms LGBM dominance — partner model is irrelevant. All ML improvement ideas now tested and resolved.
- **XRP Mode H horizon sweep** — 4 horizons (5h-8h). Winner: 8h XGB+LGBM +9.99%, 18 trades, 78% WR, conf>=80%. Config updated.
- **SOL Mode H re-run (full 4h-8h with PySR)** — 5 horizons. Winner: 8h RF+XGB +22.43%, 32 trades, 69% WR, conf>=75%.
- **PySR discovery for all assets** — Mode P completed for BTC, ETH, SOL, XRP, LINK, DOGE, ADA, AVAX, DOT (all horizons 4h-8h). Historical window method.

### Completed (2026-03-25)
- **Telegram optimizer bot** — `crypto_optimizer_bot.py`. Remote triggering of Mode D/V/H/P/S via inline keyboard menus. Sequential job queue, subprocess execution with unbuffered real-time progress output, below-normal Windows priority. Separate bot token to avoid conflicts with trader bot.
- **PySR leakage fix** — Initial PySR results were inflated (BTC 6h: +9.27% with leaky PySR vs +3.74% baseline, ETH 7h: +23.32%). Root cause: PySR formulas fitted on same 6-month window Mode D evaluates on. Fix: PySR discovery now uses historical window (months 12→6 ago), zero overlap with Mode D's last 6 months. Anti-leakage checks added in Mode D/V/Refine — strips PySR features early if JSON lacks `discovery_method == "historical"`. Mode V blocks production writes for leaky configs.
- **PySR promoted to production** — `_compute_pysr_features()` merged into `crypto_trading_system_doohan.py`. Loads `models/pysr_{ASSET}_{H}h.json` if exists, safe fallback if not. Clean PySR results pending re-run with historical window.
- **logret_5h and logret_7h added** — Fills gaps in short-term momentum for 5h/7h horizon models. 132 total features.
- **ETH Mode H horizon sweep** — 3 horizons (5h/6h/7h) with PySR. Winner: 7h RF+LGBM +23.58%, 16 trades, 62% WR, conf>=90%. Config updated to 7h/90%.
- **SOL Mode H horizon sweep** — Initial run 5h/6h/7h. Re-run 2026-03-26 with full 4h-8h + PySR (see above).
- **BTC 4h/8h with embargo fix** — 4h confirmed overfit (negative post-embargo). 8h not viable. BTC production stays at 6h.
- **V1.7.2 — Regularization** — Tested on BTC 6h. Minimal reg won (ra=0, rl=0.1, cs=0.9, ss=0.5). V1.7.1 baseline (+3.75/+3.47/+3.74 at 70/80/90%) more consistent than V1.7.2 (+3.63/+0.11/+4.64). **Verdict: wash, not adopted.**
- **PySR installed on both machines** — Laptop + Desktop done (2026-03-25). Julia 1.11.9 backend compiled.
- **Telegram balance bug fixed** — Exchange balances were fetched before trade loop but displayed after, causing false "Tracker says invested but exchange shows 0" warnings.
- **Scoring formula changed** — `return × (win_rate/100)` for positives, raw return for negatives. Favors consistency over raw return. Refined-only filter removed — D and refined compete equally.
- **Telegram horizon fix** — Signal line now shows actual configured horizons (e.g., `6h=BUY(78%)`) instead of hardcoded 4h/8h N/A.

### Completed (2026-03-24)
- **Doohan V1.7.1 promoted to production** — Renamed to `crypto_trading_system_doohan.py`. Embargo-fixed grid + Optuna refine. Variable horizon per asset. BTC 6h winner: XGB+LGBM w=252h g=0.994 f=9, +3.75% at conf>=70%.
- **Deku archived** — All Deku files moved to `archive/`. Deku replaced by Doohan as production system.
- **Refined-only production selection** — Initially Mode V only picked from refined configs (confirmed on BTC 5h/6h/7h). Later replaced by `return × win_rate` scoring with all candidates competing equally (2026-03-25).
- **Variable horizon support** — Trading config stores `horizon` per asset. Trader reads it and uses `Xh_only`. Mode H sweeps multiple horizons.
- **New Mode H** — Horizon sweep: D+V per horizon → cross-horizon comparison → saves best to trading config. `--skip` flag to reuse existing D results.
- **Order-independent CLI** — Arguments can appear in any order. `--help` shows full usage.
- **Price fix ported** — Live trader now reports current market price (`df_raw.iloc[-1]['close']`) instead of label-shifted historical price.
- **Dead code cleanup** — ~1,435 lines removed (legacy modes A/E/DAF, old Mode H, legacy Mode D).
- **Root folder cleanup** — 28 files archived. Root now has 7 Python files + configs.
- **Mode V confidence thresholds expanded** — Tests 65/70/75/80/85/90% (was 70/80/90).

### Dropped
- ~~Deku~~ — Replaced by Doohan V1.7.1. Embargo fix revealed Deku APFs were inflated.
- ~~Doohan V1.1–V1.6~~ — Superseded by V1.7.1 embargo fix. V1.6 grid approach preserved but with proper embargo.
- ~~CASCA~~ — Superseded by Deku, then by Doohan.
- ~~4h horizons~~ — Confirmed overfit (all D candidates negative post-embargo). 5h+ horizons are viable.
- ~~CPCV as validation~~ — Incompatible with temporal decay (gamma). 4h overfits (PBO=1.0), 8h is real edge.
- ~~Deku enhancements~~ — 11 features + 2 toggles all FAILED (-62.2%).
- ~~Multi-timeframe fusion~~ — Cross-TF fusion worse than single-timeframe.
- ~~V1.7.2 Regularization~~ — Wash. Minimal reg won but inconsistent across confidence levels. V1.7.1 unregularized baseline kept.
- ~~TabPFN / tabular transformers~~ — Tested and failed.
- ~~V1.8 LSTM~~ — LSTM solo: 0 valid results (all failed). LSTM+LGBM/XGB: identical to RF+LGBM/XGB (LSTM votes randomly, partner model carries all signal). Confirms LGBM dominance.
