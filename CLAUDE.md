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

**On-chain — MIXED (updated 2026-04-19 after first post-wiring HRS audit):**
- `oc_mvrv_chg1d` — **Grade 4 (66.7%, 4/6 ETH horizons).** Selected on 5h, 6h, 7h, 8h.
- `oc_exchange_netflow_chg5d` — **Grade 3 (16.7%, 8h only).**
- Other 38 derivatives (hashrate, active_addresses, tx_count, fees_native, exchange_inflow/outflow, plus other chg/zscore/ratio variants) — below selection threshold this run. Open question: weak or correlation-crowded-out (TODO 1a).
- BTC SOPR untested yet (pending BTC HRST — TODO 2).
- Loader wired 2026-04-17; first HRS post-wiring was 2026-04-18. Prior "DEAD" verdict was pre-wiring.

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
- **Asymmetric loss (scale_pos_weight).** Tested 2026-04-19 on ETH 6h+8h (`tools/test_asymmetric_loss.py`). Penalising false BUYs more heavily (weights 0.3-1.5) made zero difference because the 95%/85% confidence threshold already filters aggressively. On 6h: all weights produced identical 3 trades (100% WR). On 8h: weight=0.7 gained +0.27pp over baseline — noise. **Dead at high confidence thresholds.**
- **Signal staleness (consecutive BUY filter).** Tested 2026-04-19 on ETH 6h+8h (`tools/test_signal_staleness.py`). Tracked consecutive hours the model said BUY; skipped entry after 1/2/3/4/5/8 consecutive BUYs. Best: skip-after-3 on 6h gained +1.14pp (+27.34% vs +26.20% baseline, 61% vs 58% WR, 41 vs 43 trades). On 8h: skip-after-2 gained +0.92pp. BUY streaks average 5.3h with max 81h. **Not actionable — the improvement is noise-level.** Rally-cooldown gate serves the same anti-churn function better.
- **GDELT geopolitical features (21 features).** Downloaded from GDELT DOC 2.0 API (iran_vol, geopolitical_tone, etc.). iran_vol_zscore ranked #9 in one LGBM importance run (2026-04-10) but was **never selected into any of 33 production models** (0% selection rate across ETH/BTC/SOL/LINK/XRP). VIX and equity 1-day changes capture macro fear faster because they're market-priced. Download takes 5-10 min of rate-limited API calls per Mode D. **Disabled from both download and feature pipeline 2026-04-19.** Code kept commented for future use.
- **Kelly criterion position sizing.** Evaluated 2026-04-19. Not backtested — architectural mismatch with all-in/all-out binary position model. Kelly sizes by confidence gradient (95% = big, 65% = small), but system only enters at high confidence (85-95%) with no position-size gradient to exploit. Would only matter if confidence thresholds were lowered. **Not implemented.**
- **Stablecoin market cap features (3 features).** Downloaded 2026-04-19 from CoinGecko (USDT+USDC market cap, 1y daily). Features: `stable_mcap_chg1d`, `stable_mcap_chg7d`, `stable_mcap_zscore`. LGBM importance: all ranked below 1% on both ETH 6h and 8h. **Grade 1 — dead.** Data kept in `stablecoin_flows.csv` but features are effectively noise.

### What's promising (tested, pending decision)

- **Volatility-scaled horizons.** Tested 2026-04-19 on ETH 2-month replay (`tools/test_vol_scaled_horizon.py`). Instead of fixed regime-based horizon (tsmom bull=6h / bear=8h), dynamically pick horizon based on 24h realized vol percentile rank (vs 30-day window). **Best: `vol_2band low→8h high→6h @90%` = +33.82%, 46 trades, 65% WR — beats current tsmom regime (+28.80%) by +5.02pp and beats every single-horizon baseline.** Logic: high vol (>70th pctile) → shorter 6h horizon (faster signal); low vol → longer 8h (more confirmation). Hybrid tsmom+vol variants also tested but didn't beat the pure vol picker. **Next step: validate on 4-month window and consider replacing tsmom detector or combining.**
- **ETH derivatives as features (funding rate + open interest).** Added 2026-04-19 — extended Binance derivatives download to ETH (was BTC-only). LGBM importance ranking: `deriv_funding_chg1d` ranked **#4 on 6h (4.2%)** and **#2 on 8h (3.7%)**. `deriv_oi_chg3d` ranked **#5 on 6h (3.3%)**, `deriv_oi_chg1d` ranked **#4 on 8h (3.6%)**. These are top-tier — higher than established features like `adx_14h` and `price_to_sma100h`. Mode D grid produced 0 valid candidates initially due to OI NaN (30-day history only); fixed by excluding sparse features from `dropna()` (LGBM handles NaN natively). **Next step: re-run Mode D or HRST with the NaN fix to get actual production candidates.**

### What's untested (queued)

- **Triple barrier labeling.** Replace fixed-horizon fee-aware labels with volatility-adaptive triple barrier (upper profit target + lower stop + time limit). Tested in 2025 pre-embargo on BTC with old combos (RF+GB+LR) — result was +29% but incomparable to current pipeline. Needs fresh test with Ed's RF+LGBM ensemble, embargo, PySR. **Effort: medium (change label creation in `build_all_features()`, run Mode D).**
- **Meta-labeling.** Train a secondary model that predicts whether the primary model's BUY/SELL signal will be correct. Separates signal direction from signal quality. Never tested. Lopez de Prado Ch. 3. **Effort: medium-high (new training pipeline).**
- **Per-regime feature set.** Open question from engine reference: should bear use more macro features, bull more technical? The asymmetry already happens organically (6h model: 7 features, mostly PySR+technical; 8h model: 32 features, kitchen sink) but hasn't been deliberately tested as a design choice. **Effort: low-medium (code change to Mode D).**

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
8. **On-chain MVRV is valuable (Grade 4); other on-chain metrics are dead.** 38/40 on-chain derivatives never selected. Only `oc_mvrv_chg1d` (67% ETH) and `oc_exchange_netflow_chg5d` (3%, 8h only) earn their keep.
9. **ETH derivatives (funding rate + open interest) are top-tier features.** `deriv_funding_chg1d` ranked #2-4 across horizons (3.7-4.2% importance). `deriv_oi_chg1d/3d` ranked #4-5. Higher than many established technical features. BTC-only derivatives were tested and marked Grade 1; ETH derivatives are Grade 4+.
10. **Feature bloat costs little compute but adds NaN noise.** 127/206 features (62%) are never selected. Trimming to ~80 features would save ~10-15% compute and reduce dropna row loss from sparse features. GDELT (21 dead features + slow download) was the worst offender — disabled 2026-04-19.

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

## Current Regime Config (Ed V2 — 2026-04-19)

```json
{
  "ETH": {
    "detector": "named:tsmom_672h",
    "bull": "6h@95%/$12k, shield=ON, gate=NONE (no STRICT winner on 60d)",
    "bear": "8h@80%/$12k, shield=OFF, gate=rr8h≥2.5% OR rr30h≥5.0%, cd=18h",
    "shared": "min_sell_pnl=0.6%, max_hold=12h",
    "disaster_brake_pct": 5,
    "shield_quick_release": "95%/4h (armed; empirically inert on 60d)",
    "backtest_fee_per_leg": "0.0005 (5 bps, realistic maker blend)",
    "enabled": true
  },
  "BTC":  { "enabled": false, "note": "HRST done 2026-04-19; shelved — opportunity cost vs ETH" },
  "SOL":  { "enabled": false, "note": "HRST running overnight 2026-04-19 — review tomorrow" },
  "LINK": { "enabled": false, "note": "standby" },
  "XRP":  { "enabled": false, "note": "standby; decorrelation candidate (~0.60 corr w/ ETH)" }
}
```

**Asset universe pruned 2026-04-19.** Dropped DOGE / ADA / AVAX / DOT (weak priors, no diversification). Kept 5 (ETH active + BTC/SOL/LINK/XRP standby). Config blocks removed from `regime_config_ed.json`; ASSETS dicts pruned in `crypto_trading_system_ed.py / ed_v3.py / ein.py / eli.py`; ASSETS list pruned in `crypto_optimizer_bot.py`; 15 stale model rows removed from `crypto_ed_production.csv`. PySR JSONs + hourly data CSVs kept for later revive.

**2026-04-19 ETH live backtest (60d, 5 bps/leg fee):** +61.56% strategy vs +19.42% B&H, **+42.14pp alpha**. Win rate 75%, expectancy +1.34%/trade, profit factor 8.7, max drawdown −5.29%. Disaster brake (5%) and quick-release (95%/4h) both inert in backtest. Bull gate intentionally absent — STRICT filter rejected all 445 candidates that beat baseline (no plateau-robust region). Bear gate shipped from iterative T↔G convergence with realistic fee.

**2026-04-18 full refresh (HRSTG pipeline):** PySR deepened 2026-04-11 (`maxsize 15→25, iterations 40→100, multi-run + islands`) but 6h/8h production models were still trained against old PySR formulas → silent feature drift. Fixed: Mode P on 5h/7h, then HRS on all 4 horizons, then per-regime T (shield ON bull, OFF bear), then G (rally-cooldown). Per-regime shield split beat single-flag both-ON by +23.45pp (+115.50% vs +92.05% all-OFF baseline over 60d, 0% fee sim).

**Prior history:** ETH HRS 2-month (2026-04-07) initially picked bull=6h@90% / bear=7h@75%. After R→S handoff fix + Option C joint detector sweep, ETH RS rerun selected `sma168>sma480` bull=7h@75% / bear=8h@85% → Mode S +60.72%. 2026-04-18 HRSTG rerun switched detector to `tsmom_672h` and confidences to 95%/80%.

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
13. **Per-regime hold-shield (2026-04-18):** `hold_shield` now lives inside `bull` and `bear` blocks, not at asset level. Trader reads via `_shield_on_for_regime(trading_cfg, regime_label)` which falls back to asset-level `hold_shield` for legacy config. Shared `min_sell_pnl_pct` + `max_hold_hours` stay at asset level — per-regime split is only ON/OFF.
14. **Regenerating PySR requires retraining production models.** Mode P rewrites `models/pysr_*.json` but production CSV stores feature *names* only — inference re-evaluates whichever formula is in the JSON *now*. If you run P, follow with DV (or HRS) for the same horizon, else silent feature drift at inference. Leakage check (`_check_pysr_leakage`) catches leakage but NOT train/inference mismatch.
15. **Mode T chains rally-cooldown (2026-04-18):** After writing shield + thresholds, T merges its fresh bull/bear signals via `_merge_tagged_signals()` and calls the shared `_sweep_rally_cooldown()` helper. No stale cache issue, no need to run G separately. Mode G standalone kept as cache-fed fallback (fast iteration when models haven't changed).

---

## Pending Work

### TODO

**🛎️ FIRST: check `output/ERRORS_INBOX.md` for runtime errors**

The live trader appends runtime warnings/errors to `output/ERRORS_INBOX.md`
(rate-limited to 1 entry per unique key per hour). When the user asks
"what's on my TODO" or similar status review, READ THIS FILE FIRST and
summarize any recent entries inline with the TODO review. Each entry has
severity (ℹ info / ⚠ warn / 🚨 critical). Critical entries should be
surfaced prominently — they usually mean the trader is running degraded
(FEATURE_SET_A fallback, regime detector errored, upstream data missing).

After review, user will either:
- Tell me to investigate/fix → find the root cause in code + logs
- Tell me to clear/ignore → delete the relevant lines from the inbox file

Inbox location: `output/ERRORS_INBOX.md`. Starts empty — only populates
when the trader hits an alert path. Absence of file = clean slate.

Stdout is also always mirrored to `logs/ed_runtime_*.log` per-launch
(Fix #5D 2026-04-24), so console-only warnings survive restart. Grep
there for post-incident forensics.

---

**🚨🚨🚨 UPMOST IMPORTANT — LIVE TRADER DATA-UPDATE SANITY CHECK (2026-04-23 morning, root cause of 86%-pinned bug):**

**Root cause discovered this morning:** The 5h@86% stuck signal (7+ consecutive identical BUY signals) was caused by `xa_btc_lag2h` being NaN in all recent ETH rows because BTC OHLCV hadn't been downloaded in 49 hours (BTC is disabled/not traded → no downloader touches it → stale → merge yields NaN on recent rows). `generate_live_signal` at `crypto_live_trader_ed.py:379` does `df.dropna(subset=feature_cols + ['label'])` which kills every recent row, then `i = n - 1` points to a row from 49h ago. Model retrains each cycle but on the SAME frozen row → SAME 86% output every hour.

Same class of bug almost certainly caused the prior **7h@99% pinned for 31+ hours** (likely `oc_mvrv_chg1d` on-chain data was stale).

**Hard rules going forward — trader must enforce these at every cycle:**

1. **Refuse to predict on stale data.** In `generate_live_signal`, after dropna, compute `lag_hours = (df_full.iloc[-1]['datetime'] - df.iloc[-1]['datetime']).total_seconds() / 3600`. If `lag_hours > horizon + 2`, print an error, send Telegram alert, and **return None** (do NOT emit a signal). Never silently predict on a row >horizon+2h old.

2. **Don't drop whole rows on single-feature NaN.** Change `df = df_full.dropna(subset=feature_cols + ['label'])` → drop on `['label']` only. For feature NaN at inference, impute with 0 (matches PySR's NaN handling and LGBM's native NaN tolerance; RF needs the impute). This preserves all rows even when one data source is late.

3. **Remove BTC entirely from ETH feature pipeline.** User doesn't trade BTC — `leaders_for['ETH']` in `crypto_trading_system_ed.py:1190` should be `[]` not `['BTC']`. Strips the `xa_btc_lag1h/2h/3h` features. Also remove BTC from the trader's download loop so stale BTC data isn't silently masking this class of bug.

4. **Pre-inference data-freshness gate.** Before calling `build_all_features` in every cycle, the trader must verify each data source's mtime/last-row freshness:
   - Primary OHLCV: last row ≤ 1.5h old
   - Derivatives (`derivatives_eth.csv`): last row ≤ 2h old
   - On-chain (`onchain_eth.csv`): last row ≤ 30h old (daily data)
   - Stablecoin flows: last row ≤ 36h old
   - Orderbook snapshots: ≥ 20 rows in last 24h
   - Options IV: ≥ 20 rows in last 24h
   - Macro (`macro_hourly.csv`, `macro_daily.csv`): last row ≤ 24h old
   
   If any check fails, log the failing sources, send Telegram alert, try to refresh via `download_macro_data.py` / `download_asset`, re-check. If still stale, refuse to emit a signal this cycle.

5. **Audit every feature used by a prod model for sparse-tail risk.** Add to `tools/audit_features.py` a check that reports, per prod-CSV feature, the NaN count in the last 24 bars. Any feature with >0 NaN in tail is flagged for removal from new models (or requires a freshness guardian).

**Current blast radius:** the 5h model just underwent this bug live. The 7h model probably did too. Any newborn/sparse feature in a future prod model (derivatives, on-chain, orderbook, IV, basis, lead-lag) is a ticking bomb until (1)-(4) ship.

**Effort: ~2h to ship all four. Do this BEFORE any further trading.**

---

**🧪 CURRENTLY RUNNING — AB MATRIX RELAUNCH (2026-04-24 07:40 CEST → ETA ~17:00 2026-04-24):**

Previous matrix (2026-04-22 17:32) was **biased** — dropna was eating ~half the window due to 7 sparse features with short history. All V1/V4 promotion decisions made on 672-row training set instead of 1432. Full explanation in "Closed 2026-04-24" section below.

Relaunch: `python tools/ab_matrix_runner.py --variants focus --seed 2026`. Three variants (A_floorON_trimOFF, B_floorON_trimON, C_floorOFF_trimOFF), clean data (1432/1440 rows per variant), seed 2026 to cross-check effects aren't seed-specific.

**Earlier (superseded) matrix results — kept for historical record only:**
Variant #1 finished 2026-04-22 21:35 and was promoted live as V1 (then later replaced with V4 from that same matrix). Both V1 and V4 trained on the 672-row biased window — their Mode T totals (+122.59% and similar) are inflated by training narrowness. **Do not reference those numbers as baselines — they're not comparable to the clean relaunch.**

---

**🔄 RETEST PRIORITIES AFTER MATRIX FINISHES (ETH-first, clean data post-dropna-fix 2026-04-24):**

All decisions from 2026-03-20 onwards were made on biased data (dropna eating ~half the window due to sparse-feature NaN). ETH-focused retests come first; secondary assets (SOL/LINK/BTC) lower priority — fix ETH robust FIRST.

**R1. [HIGH — ETH core] Promote clean matrix winner.**
When matrix finishes (~17:00 today), compare A/B/C variants' Mode T finals. Whichever converged + has top alpha + passes 4 promotion gates:
```powershell
# Backup current V4
copy config\regime_config_ed.json config\regime_config_ed_v4_pre_clean.json
copy models\crypto_ed_production.csv models\crypto_ed_production_v4_pre_clean.csv
# Promote the matrix winner (replace {LABEL} with actual winner: A_floorON_trimOFF, B_floorON_trimON, or C_floorOFF_trimOFF)
copy config\regime_config_ed_noprod_{LABEL}.json config\regime_config_ed.json
copy models\crypto_ed_production_noprod_{LABEL}.csv models\crypto_ed_production.csv
# Clear cooldown
# Edit config/position_ed_v2_ETH.json, set "rally_cooldown_until": ""
```
**Decision tree:** best variant's alpha > V4's live performance-adjusted expectation → promote; within noise → keep V4, re-evaluate next week.

**R2. [HIGH — ETH validation] 4-month HRST on clean winner.**
Structural-consistency check — does the matrix winner hold on a longer window?
```bash
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 2880 --no-persist
```
~6-8h desktop runtime. Tag outputs `_4mo_clean`. Compare Mode S winner to the 2mo clean winner:
- Same detector + horizons → strong structural confirmation → ship with confidence
- Different winner → window-sensitivity still present; use 2mo for current market regime (per standing rule R7)
Prior biased 4mo run gave `tsmom_672h 6h/7h @ +155.91pp` — that number is inflated. Clean 4mo will be more honest.

**R3. [HIGH — ETH, conditional] Meta-labeling retest on clean signals.**
Only if matrix's variant C (floor OFF) or B (trim+meta) shows meta as a contender. Prior standalone meta harness showed +23.21pp on raw signals at p=0.60 and strategy-aware +15.76pp — **both biased**. Retest with the NEW clean primary model (from R1) as the base:
```bash
python crypto_trading_system_meta.py ETH 5 --replay 1440
python crypto_trading_system_meta.py ETH 6 --replay 1440
python crypto_trading_system_meta.py ETH 7 --replay 1440
python crypto_trading_system_meta.py ETH 8 --replay 1440
python tools/test_meta_strategy_impact.py ETH --replay 1440
```
~30-60min total. Decision: ≥+10pp on clean data at any threshold → ship behind trader flag; <+5pp → shelve meta permanently.

**R4. [HIGH — ETH label] Label-threshold 1% retest on clean data.**
Previous test (−47pp delta) was run on biased data. Direction was clear but absolute numbers untrustworthy. Quick confirm on clean data:
```bash
python crypto_trading_system_ed.py DV ETH 5h --replay 1440 --label-threshold 0.01
```
~45min. Expected: still negative delta (direction robust). If suddenly POSITIVE on clean data, reopen the label-threshold question.

**R5. [HIGH — ETH gate] Bull rally-gate retest on clean data.**
Prior evidence (bull gate hurts: live 30d −7.48pp, OOS 60d −1.31pp, FULL 90d −3.32pp) was on biased data. The consistency across 3 windows suggests the direction is real, but retest:
- Check V4's current bull-gate (whichever the matrix winner picks) against a disabled-gate variant
- Manual test: after promoting R1's winner, run 1 week with bull gate ON, compare to baseline simulation
- Or: compare matrix A's bull gate output vs A with `/gate ETH bull off` post-promotion

**R6. [MEDIUM — ETH gate] Drop-gate sweep retest.**
Prior rejection (101/126 OOS rank = overfit). Direction robust to bias. Only retest if meta + 4mo + bull-gate all favor complex gate structure.
```bash
python tools/backtest_drop_gate.py  # already exists
```
Low priority — run only if time.

---

**🟡 LOW PRIORITY — SECONDARY ASSET RETESTS (only after ETH is robust):**

Don't touch these until ETH promotion + 4mo validation + meta decision are done. Running ETH-first preserves compute for ETH-specific tuning.

**R7. [LOW] SOL HRST retest.**
Prior result (2026-04-21): `sma168>sma480 5h@65%/8h@70%` +42.30%/+40.37pp alpha (biased). Retest:
```bash
python crypto_trading_system_ed.py HRST SOL 5,6,7,8h --replay 1440 --no-persist
```
~3-4h desktop. Decision: if clean alpha ≥ 50% of ETH's clean alpha → commit $2-3k. Otherwise shelve.

**R8. [LOW] LINK HRST retest.**
Shelved 2026-04-20: "5/8 horizons NEGATIVE, beats_3of3=0". Could be entirely a dropna artifact. Same command pattern as R7, replace SOL with LINK. If LINK now converges with decent plateau → unshelve. If still weak → permanently shelve.

**R9. [LOW] BTC HRST retest.**
Last done 2026-04-20 with +36.15%/+23.89pp alpha. Biased. Retest for reliability check — BTC is the simplest diversification target but wasn't enabled. Same command with BTC.

**R10. [LOW] XRP HRST (never tested on clean data).**
Only remaining untested asset. Mode P first if JSONs are stale:
```bash
python crypto_trading_system_ed.py P XRP 5,6,7,8h
python crypto_trading_system_ed.py HRST XRP 5,6,7,8h --replay 1440 --no-persist
```

---

**🟢 STILL TRUSTED (no retest needed — pre-bug-era or bias-independent):**

- Stop-loss / take-profit / trailing-stop variants (tested in earlier era, before sparse-feature pipeline)
- LSTM ensemble (pre-bug era)
- V1.7.2 regularization (pre-bug era)
- 4h horizon rejection (structural embargo issue, not bias)
- GDELT disable (feature selection evidence, not bias-dependent)
- PySR discovery leakage check (uses dedicated historical window, excludes Mode D's replay period)

`python tools/ab_matrix_runner.py` launched. 5 HRST variants + 1 vol test, all on a frozen data snapshot with `--no-persist` + `--no-data-update`. Safe to run alongside live trader; position stays open.

| # | Variant | Trim | Meta | Floor |
|---|---|---|---|---|
| 1 | trimOFF_metaOFF | OFF | OFF | ON |
| 2 | trimON_metaOFF | ON | OFF | ON |
| 3 | trimOFF_metaON | OFF | p=0.45 | ON |
| 4 | trimON_metaON | ON | p=0.45 | ON |
| 5 | trimON_metaON_floorOFF | ON | p=0.45 | OFF |
| 6 | vol_scaled | — | — | — |

**Outputs:**
- Full audit CSV: `output/ab_matrix_results_<timestamp>.csv` — detector, bull/bear horizons + confidences, shields, gates, min_sell_pnl, max_hold, Mode T REF/H1/H2 returns, meta stats, per-horizon combo/window/gamma/features/logret_count/pysr_count
- Tagged `_noprod_<label>.{json,csv}` per variant in `config/` and `models/` — can be promoted directly if decision says so
- Log per variant: newest `logs/ed_v1_*.log` around each run's start time
- Data snapshot preserved: `data/_ab_snapshot_<timestamp>/` (delete if not needed after analysis)

**Progress monitoring:** tail the latest `logs/ed_v1_*.log`; check CSV has grown after each variant (~2.5-3.5h apart).

**Decision dimensions when it finishes:**
- **Trim effect**: #2 vs #1 (trim only), #4 vs #3 (trim with meta active)
- **Meta effect**: #3 vs #1 (meta only), #4 vs #2 (meta with trim active)
- **Floor effect**: #5 vs #4 (same config, floor off vs on) — tiebreaker on shipped change
- **Vol-scaled detector**: #6 vs whichever primary variant wins
- **Winner**: highest `t_ref_pct` that also passes promotion gates (Mode T converged, top-15 plateau, trend features present)

---

~~**🚨 BLOCKING — POST-RALLY PROMOTION WORKFLOW**~~ — **RESOLVED 2026-04-22 late night:** user sold manually at ~$2,388 (+3.2% realized on the 2026-04-21 trade) and immediately promoted **Variant #1** (not laptop) to production. See "Closed 2026-04-22 late night" below for details.

Historical context (kept for audit):

**R0. [BLOCKING] Realized PnL logging + trade postmortem.**
When the SELL fires, record: entry price, exit price, exit reason (model/shield/max_hold), realized %, hold duration. Store in `output/live_trades_2026-04-21.md` or similar for the rally-retrospective decomposition. Expected exit somewhere between +2% and +5% depending on when prod's 7h model flips.

**R1. [BLOCKING] Compare realized vs laptop's hypothetical on this trade.**
Laptop model had SELL signal at 03:00Z ($2,373, +2.59%), stronger at 05:00Z / 06:00Z ($2,364 / $2,396, +2.2%/+3.6%). Actual prod exit price tells us which approach won on THIS trade. Single-trade is noise but informative.

**R2. [HIGH] Promote laptop config to production — cash-state switchover.**
```powershell
# Backup current prod
copy config\regime_config_ed.json config\regime_config_ed_2mo_pre_laptop.json
copy models\crypto_ed_production.csv models\crypto_ed_production_2mo_pre_laptop.csv

# Promote laptop
copy config\regime_config_ed_noprod.json config\regime_config_ed.json
copy models\crypto_ed_production_noprod.csv models\crypto_ed_production.csv

# Clear cooldown timer — fresh gate state
# Open config/position_ed_v2_ETH.json, set "rally_cooldown_until": ""
```
Trader hot-reloads within 5 min. Monitors:
- alpha tracking sim +68.37pp over first 10+ trades
- bull gate (`rr14h≥6.0 OR rr20h≥5.5 cd=10h`) — fires rarely, shouldn't block normal BUYs
- bear shield (now ON) — holds bear-regime trades through initial dip

**R3. [DECIDE DURING R2] Option C variant — bull gate OFF?**
Laptop's bull gate fires only on 6%+ rallies (rare). But evidence across multiple tests (live 30d, OOS 60d, FULL 90d) shows bull gates generally hurt or break even. Consider immediately:
```
/gate ETH bull off
```
Keeps laptop's other settings (shield flip, bear gate, detector) but removes even the rare-firing bull gate. If this test runs parallel to R2 mentally, can flip back with `/gate ETH bull on` if we see harm.

**R4. [HIGH] Rollback safety net — if laptop underperforms in live.**
Threshold: if cumulative realized alpha drops >15pp vs sim baseline over the first 10 trades, rollback:
```powershell
copy config\regime_config_ed_2mo_pre_laptop.json config\regime_config_ed.json
copy models\crypto_ed_production_2mo_pre_laptop.csv models\crypto_ed_production.csv
```
Also stops the bull-shield=OFF experiment mid-regime if it's clearly failing.

**R5. [MEDIUM] 4-month replay confirmation HRST for laptop config.**
Once laptop is in prod and running, validate on a longer window:
```bash
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 2880 --no-persist
```
~3h. Compare Mode S winner to current laptop config. If detector/horizons shift again on 4mo → ensemble-of-windows question reopens. Also serves as the "accept or reject" tiebreaker on the 2mo vs 4mo window-sensitivity question still open from 2026-04-21.

**R7. [STANDING POLICY] HRST re-sweep cadence.**
Evidence from 2026-04-21/22: three ETH HRST runs within 9h gave 3 completely different winners (detector, horizons, shield, gates all flipped). Running HRST daily chases 5-10pp nondeterminism noise as if it were signal.

**Standing rules going forward:**
- **Scheduled HRST**: every 2-4 weeks (material training-window shift)
- **Mandatory HRST**: always after a Mode P (PySR) run — else silent feature drift (rule 14)
- **Event-driven HRST**: after material regime shift, after new feature block added, after live >15pp underperformance over 10+ trades
- **Research HRST**: always `--no-persist`

**Promotion gate (must pass ALL four before writing to prod):**
1. Mode T converged (not `max_iter` hit)
2. Top-15 plateau: ≥10/15 configs share detector + bull horizon
3. Alpha > current-prod live alpha + nondeterminism margin (~10pp)
4. Structural consistency: if 4mo and 2mo disagree on detector, neither is promoted without a tiebreaker run

By these rules, today's 2mo Desktop HRST (hit max_iter, was scattered in top-15) **should not have been promoted**. The laptop 2mo+trim result passes all four gates — it's what should go live post-rally (R2).

**R6. [MEDIUM] SOL enablement decision with today's strong HRS result.**
SOL HRS 2026-04-21 15:23: `sma168>sma480 5h@65%/8h@70%` → **+42.30%/+40.37pp alpha**, 82% of ETH's alpha (up from borderline 40-50% yesterday). Clearly clears the ≥50% threshold. Consider $2-3k initial allocation. Watch correlation with ETH — if SOL live tracks ETH >70%, scale back.

---

**PRIORITY 1 — Open strategy tests (state at 2026-04-21 evening):**

~~**A. Large-upside label test — ETH 5h DV with `label = ret ≥ 1%`**~~ — **DONE + REJECTED (2026-04-21 00:37).** Ran `DV ETH 5h --replay 1440 --label-threshold 0.01` on Desktop overnight. Winner: RF+LGBM w=250 g=0.999 n_features=10 → **+11.87% return, 72% accuracy**. Current prod ETH 5h baseline: **+59.11% return, 65.9% accuracy**. Net **delta −47.24pp return** despite +6.1pp accuracy gain. Classic tighter-label regression: positive-class starvation reduces trainable signal; model becomes more selective at the cost of volume. Output preserved at `models/crypto_ed_production_lt1.csv`. Don't try `--label-threshold 0.005` — the direction is clearly wrong. **Label stays at `2×fee = 0.22%`.** Takeaway: the small-wins-are-noise intuition is right at the TRADE level but the MODEL extracts edge from volume, not individual trade contribution. Real execution gap (~17pp after bull-gate fix) is execution friction, not label choice.

~~**B. Feature-trim validation HRST**~~ — **DONE 2026-04-22 03:03** (log `ed_v1_20260421_212008.log`, laptop, 6h runtime). Winner: **`price>sma72` bull=6h@75% / bear=5h@75%, shield bull=OFF/bear=ON, min_sell_pnl=0.50%, max_hold=10h**, converged Mode T iter 4. Bull gate `rr14h≥6.0%/rr20h≥5.5% cd=10h` (rare firing), bear gate `rr8h≥3.0%/rr12h≥2.0% cd=16h`. 60d sim: **+86.79% return / +68.37pp alpha / 118 trades / 75% WR**. Top-15 UNANIMOUS on detector/horizons — tightest plateau of any ETH HRST this week. Trim verdict: **trim helped (+18.96pp vs no-trim 2mo prod)**. Feature matrix: 131 total (60 disabled, vs 191 without trim). Promotion workflow → see R2 above.

~~**C. Window-sensitivity 3-way comparison**~~ — **RESOLVED 2026-04-22.** Desktop 2mo = `sma168>sma480 7h/5h @ +49.41pp`, Desktop 4mo = `tsmom_672h 6h/7h @ +155.91pp`, Laptop 2mo+trim = `price>sma72 6h/5h @ +68.37pp`. All three pick DIFFERENT detectors + different horizon combinations — HRST is genuinely window-sensitive. User's regime-call: 2mo matches current market; 4mo includes a bear phase that inflated its alpha. Laptop's 2mo result is the tiebreaker winner (trim-validated, plateau-unanimous, Mode T converged, structurally similar to 4mo on shield config). 4mo kept backed up (`config/regime_config_ed_4mo_backup.json`) as insurance.

**D. Vol-scaled horizon 4-month validation — ~30 min, laptop free.**
```bash
python tools/test_vol_scaled_horizon.py --replay 2880
```
Prior: vol_2band (low→8h, high→6h) beat tsmom regime by +5.02pp on 2-month. Need 4mo confirmation before replacing tsmom_672h detector. Note: current live detector is now `sma168>sma480` (prod) but next promotion will switch to `price>sma72` (laptop) — this vol-scaled test operates orthogonally to whichever detector we're on.

~~**E. Meta-labeling — full HRST A/B test**~~ — **SUPERSEDED by AB MATRIX (running 2026-04-22 evening).** Standalone `--meta-filter` flag still works for one-off tests but the full strategy comparison is now covered by the matrix's variants #3/#4 vs #1/#2. Earlier E2/E3 concurrent runs on 2026-04-22 collided (both writing to `_noprod.*` across machines); that attempt's logs preserved for historical reference but results contaminated. Matrix runs on single desktop with data snapshot — clean comparison.

**Infra shipped 2026-04-22:**
- `--meta-filter P` CLI flag added to `crypto_trading_system_ed.py` (auto-enforces `--no-persist`)
- `_apply_meta_filter_to_signals()` hook at the end of `generate_signals()` — walk-forward meta per horizon, downgrades BUYs with `meta_prob < P` to HOLD
- Reuses `crypto_trading_system_meta.build_meta_dataset` + `walk_forward_meta_train` (lazy import, no circular)
- Meta predictions carried on signals as `s['meta_prob']` + `s['meta_filtered']=True` for downstream inspection

**E1. Backup current `_noprod.*` state before any runs (desktop):**
```powershell
copy config\regime_config_ed_noprod.json config\regime_config_ed_noprod_pre_meta.json
copy models\crypto_ed_production_noprod.csv models\crypto_ed_production_noprod_pre_meta.csv
```

**E2. Run A — BASELINE HRST (no meta), ~3-4h on desktop:**
```bash
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist
```
After completion, tag the outputs so Run B doesn't overwrite them:
```powershell
copy config\regime_config_ed_noprod.json config\regime_config_ed_noprod_NOMETA.json
copy models\crypto_ed_production_noprod.csv models\crypto_ed_production_noprod_NOMETA.csv
```

**E3. Run B — HRST WITH META at p=0.45, ~3-4h on desktop:**
```bash
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist --meta-filter 0.45
```
Tag outputs:
```powershell
copy config\regime_config_ed_noprod.json config\regime_config_ed_noprod_META045.json
copy models\crypto_ed_production_noprod.csv models\crypto_ed_production_noprod_META045.csv
```

**E4. Compare** — Mode S winner alpha + Mode T final return + per-regime decomposition in each run's stdout, plus `combined_score` / `return_pct` columns in the two tagged production CSVs.

**E5. Decision tree:**
- Meta run's **Mode S alpha ≥ baseline + 10pp** → **ship meta** (budget 2-3h to integrate into `crypto_revolut_ed_v2.py` behind a config flag + nightly meta model refresh)
- Within **±5pp** → null; strategy already captures meta's gains — **shelve** meta, save the integration complexity
- Meta run **worse than baseline** → meta hurts optimized strategy — **shelve** and note in memory

**E6. [if ship]** Promotion gate — run the same HRST with `--meta-filter 0.45` on 4mo replay as structural-consistency check. Same decision tree. Only then touch the live trader.

Estimated total runtime: **6-8h desktop, overnight**. Chain with `;` in PowerShell if you want unattended:
```powershell
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist; copy config\regime_config_ed_noprod.json config\regime_config_ed_noprod_NOMETA.json; copy models\crypto_ed_production_noprod.csv models\crypto_ed_production_noprod_NOMETA.csv; python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist --meta-filter 0.45; copy config\regime_config_ed_noprod.json config\regime_config_ed_noprod_META045.json; copy models\crypto_ed_production_noprod.csv models\crypto_ed_production_noprod_META045.csv
```

**F. Execution-gap investigation (no lit ref, but highest-EV lever).** The 30d gap: simulated baseline +27.11% vs live +2.54% = **~24.5pp unaccounted**. ~7.5pp was bull-gate harm (now disabled 2026-04-20). Remaining ~17pp: slippage, partial fills, manual interventions, timing, clock drift. Next: TCA logging (`implementation_shortfall = fill_price - arrival_mid` per trade), manual-vs-auto PnL decomposition, latency audit on order placement.

**G. Scheduled sparse-feature re-enable tests (do NOT auto-enable — test first).**

7 features disabled 2026-04-24 via `config/disabled_features.json` because short history was collapsing `dropna()` (only 101 clean rows out of 1440 before disable; 1432 after). Data pipelines untouched — features keep accumulating in the background. Tests scheduled for when each group has enough history for a clean 60-day replay.

**G1. 2026-05-22 — `deriv_oi_*` re-enable test:**
- Features: `deriv_oi_chg1d`, `deriv_oi_chg3d`, `deriv_oi_zscore`
- History started 2026-03-20 (Binance public OI retention ~30d). By 2026-05-22 there will be ~63 days + 72h warmup buffer for `deriv_oi_chg3d`.
- **Procedure**:
  1. `copy config\disabled_features.json config\disabled_features_pre_deriv_oi_20260522.json` (backup)
  2. Baseline: `python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist` (keep disabled list as-is). Tag outputs with `_deriv_oi_OFF`.
  3. Temporarily remove 3 `deriv_oi_*` entries from `disabled_features.json`.
  4. Test: `python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist`. Tag outputs with `_deriv_oi_ON`.
  5. Restore original `disabled_features.json` from backup.
  6. **Decision tree:** Mode T alpha ON ≥ OFF + 5pp → re-enable permanently. Within ±5pp → leave disabled. Worse → leave disabled.

**G2. 2026-06-18 — Orderbook + IV re-enable test:**
- Features: `ob_imbalance`, `spread_bps`, `avg_iv`, `iv_skew`
- History started 2026-04-19 when live Ed trader began writing `orderbook_snapshots.csv` + `options_iv_snapshot.csv`. By 2026-06-18 there will be ~60 days of coverage.
- **Procedure**: same as G1 — backup, baseline run `_ob_iv_OFF`, enable-test run `_ob_iv_ON`, restore backup, decide.

**Why test-first (not auto-enable):**
- Matrix already showed adding features can hurt (trim=ON was worse than OFF despite feature availability). Same discipline for these.
- Re-enable should be event-driven (evidence of improvement), not calendar-driven.
- These tests are cheap — ~6-8h per one-off run, no code changes needed.
- **NO automatic trigger on the date** — user decides when to run the test based on priorities.

**🧪 [UPMOST ACTION — overnight desktop] AB MATRIX: trim × meta × vol (2026-04-22 evening).**

Orchestrator: `tools/ab_matrix_runner.py`. Runs the full factorial on a SINGLE MACHINE with a FROZEN DATA SNAPSHOT so every variant sees identical bars. All runs use `--no-persist` + new `--no-data-update` flag. Writes consolidated audit CSV `output/ab_matrix_results_<timestamp>.csv` with per-horizon feature lists, confidences, shields, gates, detector, meta stats — everything needed to audit each variant's strategy.

**Data freezing:** at matrix start the runner snapshots 13 data files (ETH/BTC OHLCV, macro_hourly/daily, cross_asset, fear_greed, onchain_eth/btc, derivatives_eth/btc, stablecoin, orderbook, options_iv) into `data/_ab_snapshot_<timestamp>/`. Before every variant, the runner restores from snapshot. HRST is invoked with `--no-data-update` so the usual download step is skipped. Live trader keeps updating the real files independently — snapshot is read-only from the matrix's perspective.

**Matrix (4 HRST + 1 vol test = ~15-20h):**

| Variant | Trim (Mode F) | Meta filter | Purpose |
|---|---|---|---|
| trimOFF_metaOFF | disabled | none | Pre-trim baseline (original Ed) |
| trimON_metaOFF | enabled | none | Current prod behavior (Mode F only) |
| trimOFF_metaON | disabled | p=0.45 | Meta-only effect |
| trimON_metaON | enabled | p=0.45 | Combined — the proposed future prod |
| vol_scaled | — | — | `test_vol_scaled_horizon --replay 1440` — orthogonal detector test |

**ONE command to launch the whole matrix (desktop):**
```bash
python tools/ab_matrix_runner.py
```

Optional flags:
- `--dry-run` print the plan without executing
- `--skip-vol` skip the vol-scaled horizon test

**What the orchestrator does per variant:**
1. Restores all 13 data files from the frozen snapshot (identical data)
2. Flips `config/disabled_features.json.enabled` to trim True/False
3. Runs HRST with `--no-persist --no-data-update` + optional `--meta-filter 0.45`
4. Parses the HRST log for: Mode S winner (detector + bull/bear horizon + confidence), Mode T policy (shield per regime, min_sell_pnl, max_hold), Mode T REF baseline (H1/H2/REF returns), bull+bear gate winners, convergence iteration, meta filter stats (kept/dropped/no_pred counts)
5. Parses the tagged `crypto_ed_production_noprod_<label>.csv` to pull per-horizon: best_combo, best_window, gamma, return, accuracy, n_features, full feature list, logret count, pysr count
6. Tags all 3 `_noprod.*` files (config + production + best_models) with the variant label
7. Appends row to consolidated CSV (incremental so interruptions don't lose data)
8. Restores original trim state at end (via try/finally)

**Audit columns in the CSV (per variant row):**
- Identity: variant, trim_enabled, meta_threshold, exit_code, runtime_min, timestamp, log_path, tagged_prod_csv
- Mode S winner: detector, bull_h, bull_conf, bear_h, bear_conf
- Mode T policy: t_bull_shield, t_bear_shield, t_bull_conf, t_bear_conf, t_min_sell_pnl, t_max_hold
- Mode T returns: t_ref_pct (60d without gate), t_h1_pct (recent half), t_h2_pct (older half), t_converged_iter
- Gates: bull_gate, bear_gate (formula strings or "OFF")
- Meta: meta_kept, meta_dropped, meta_no_pred, meta_threshold
- Per-horizon detail (for h=5,6,7,8): h{h}_source (e.g. "Refined#1"), h{h}_apf, h{h}_combo, h{h}_window, h{h}_gamma, h{h}_return, h{h}_accuracy, h{h}_n_features, h{h}_features (pipe-separated list), h{h}_logrets (count), h{h}_pysr (count)

**When finished, compare the CSV:**
- Order variants by `t_ref_pct` (Mode T's unshielded/ungated 60d return)
- Check if trim helped (trimON > trimOFF with same meta)
- Check if meta helped (metaON > metaOFF with same trim)
- Check for interaction (does trim+meta combine linearly or is there synergy?)
- Look at `meta_dropped` to see how aggressive the filter is in each variant

**Decision tree:**
- Best variant's Mode T REF > current laptop 2mo+trim baseline (+86.79%) → **promote it** (when trade closes)
- Best variant close to current laptop result, simpler (e.g., trim-only works as well as trim+meta) → **ship the simpler one**
- No variant beats laptop result by >10pp → **stick with laptop config** from yesterday (already tagged `_2mo_backup`)
- Vol test winner differs from HRST's tsmom → **schedule detector A/B separately**

~~**G. Feature-family floor A/B**~~ — **FOLDED INTO AB MATRIX 2026-04-22 evening.** Matrix variant #5 (`trimON_metaON_floorOFF`) vs variant #4 (`trimON_metaON` with floor ON) isolates the floor effect in a single overnight run. No separate G workflow needed.

**Infra shipped 2026-04-22 (still applies — default for all future HRST unless `--no-feature-floor`):**
- New constants in `crypto_trading_system_ed.py`: `FEATURE_FLOOR_ENABLED=True`, `FEATURE_FLOOR_MIN_LOGRET=2`, `FEATURE_FLOOR_MIN_PYSR=1`
- Helper `_feature_floor_indices(ranked_features, n_feat)` — picks column indices guaranteeing ≥2 logret + ≥1 pysr in every selected subset. Promotes essentials from beyond position N, evicts lowest-ranked non-essentials in the top-N slice. Exactly N features out, no overflow. Graceful no-op if ranked list lacks essentials.
- Wired into 4 sites: Mode D grid + refine + top-candidates CSV + refined-candidates persist
- CLI flag `--no-feature-floor` disables the floor for A/B comparison (now used by matrix variant #5)
- Unit tests: 3/3 pass (already-compliant no-op, vol-only promotion, no-essentials graceful)
- Motivation: ETH 5h/6h prod + laptop trim configs both ended up with **0 logret + 0 pysr** in their final feature sets (trend-blind volatility-only models). Root cause: Mode D's Optuna refine climbs APF on the 1440h window and lands at sparse vol-only local optima.

**PRIORITY 2 — Asset enablement decisions:**

5. ~~**SOL HRST**~~ — **DONE 2026-04-19.** Config written: `sma168>sma480` | bull=8h@90% shield=ON / bear=8h@65% shield=OFF | 0.55%/12h | no gates. Pipeline health 6/7 positive, 3 horizons at +19-23% range. **Decision pending user: small test allocation ($2-4k) or shelve.** Bottleneck: best single-horizon +23% vs ETH's +52%, so per-$ alpha likely ~40-50% of ETH — borderline on the "≥50% threshold" rule.

6. ~~**LINK HRST**~~ — **DONE + SHELVED 2026-04-20.** Config: `vol_calm` | 7h@95% shield=OFF / 6h@95% shield=ON | 0.6%/12h | no gates. Pipeline weak: 5/8 horizons NEGATIVE, `beats_3of3=0` on gate sweep (49k configs, zero beat baseline). Model can't find reliable signal in LINK's data. `LINK.enabled: false` stays.

7. **XRP HRST** (only remaining untested) — Launching Mode P first (files from 2026-03-26 are pre-deep-PySR). Correlation with ETH ~0.50-0.70 → real diversification if it works. Priors were +9.99% on 1mo (Mode H 2026-03-26). Command: `python crypto_trading_system_ed.py P XRP 5,6,7,8h; if ($?) { python crypto_trading_system_ed.py HRST XRP 5,6,7,8h --replay 1440 }`. Optional — after the 3 shelved results (BTC/SOL-borderline/LINK-weak), diversification case for more crypto assets is thin. Expect XRP likely similar.

**PRIORITY 3 — Research:**

7. **Orderbook imbalance + IV skew accumulation** — Hourly snapshots now wired into Ed trader (`crypto_revolut_ed_v2.py`). Need ~2 weeks of data before testing as features. Currently ~36 rows each (as of 2026-04-21).

8. **Eli HRS BTC** — 30-minute candle test. Separate research track.

9. **Ein results review** — 15-minute candle BTC results from earlier laptop run. Separate research track.

10. **Grade-4 on-chain expansion after newborn cool-down** — `oc_mvrv_chg1d` (Grade 3-4 on BTC/ETH) is the only on-chain metric earning its keep. After basis + lead-lag newborns prove in/out, re-audit and consider disabling more macro derivatives (esp. `m_oil_*`, `m_eurusd_*`, `m_usdjpy_*` 5d/10d/zscore variants).

### Closed 2026-04-24 (sparse-feature quarantine + dropna warning + AB matrix relaunched)

**Discovery:** all prior HRST runs (V1, V4, first AB matrix) trained on only **672 clean rows** out of a 1440-row (60d) window because `deriv_oi_*` (3 cols, ~50% NaN) and `ob_imbalance, spread_bps, avg_iv, iv_skew` (4 cols, ~93% NaN) had short history — dropna cascaded and wiped half the window. Models were effectively trained on the most recent 28 days, not 60. V1 and V4 promotion decisions were made on biased data.

**Fix — two-section `config/disabled_features.json`:**
- `disabled_exact` (65 entries): Mode F Grade-1 features — toggleable via `enabled` flag and `--trim-override`
- `always_disabled_exact` (7 entries): structurally-broken features (short history) — **applied REGARDLESS of enabled flag**
  - `deriv_oi_chg1d, deriv_oi_chg3d, deriv_oi_zscore` (Binance OI history started 2026-03-20)
  - `ob_imbalance, spread_bps` (live trader started writing 2026-04-19)
  - `avg_iv, iv_skew` (live trader Deribit feed started 2026-04-19)
- `_load_disabled_features()` returns `(exact, prefixes, enabled, always)` — 4-tuple now
- `_apply_feature_disable()` always strips `always_disabled_exact`; strips `disabled_exact+prefixes` only when `enabled=True`
- Data pipelines untouched — these features keep being downloaded, just excluded from LGBM/PySR inputs
- Result: clean rows went from 672 → **1432/1440 (99.4%)** in Mode D window

**Fix — DATA LOSS WARNING in Mode D ([crypto_trading_system_ed.py:3793](crypto_trading_system_ed.py#L3793)):**
When clean rows < 80% of window after dropna, prints bang-boxed warning with:
- Row count + % of window retained
- Top 8 NaN offenders by raw count
- Suggestion to add them to `disabled_features.json`
- Explicit note that model results are biased toward whatever regime spans the surviving rows
- Fires per horizon (4× per HRST if the issue persists)

**Live trader impact:** zero. `disabled_features.json` structure extended but backward-compatible. Live trader's hot-reload now sees 72 total disabled (65 Mode F + 7 always). Effective pool unchanged from yesterday. V4 prod models don't reference any of the 7 always-disabled features — verified via `optimal_features` scan.

**Matrix relaunched 2026-04-24 07:40 CEST** — `python tools/ab_matrix_runner.py --variants focus --seed 2026`:
- 3 variants: A (floorON_trimOFF), B (floorON_trimON), C (floorOFF_trimOFF)
- Optuna seed 2026 (vs default 42) — tests whether effects are seed-robust
- Now running on 1432/1440 clean rows per variant instead of 672
- ETA ~14:00-17:00 2026-04-24

**Known TODO scheduled re-enable tests:** 2026-05-22 for deriv_oi_*; 2026-06-18 for orderbook/IV. See TODO G1/G2. **NOT auto-enabled** — user runs A/B test first and decides.

### Closed 2026-04-22 late night (V1 promoted + rally-cd bug fix + matrix launched)

- **Variant #1 of AB matrix promoted to production at 23:24 CEST.** User sold 2026-04-21 trade manually at ~$2,388 (+3.2% realized, ~33h hold) and immediately promoted V1 without waiting for variants #2-#5. Promoted config:
  - Detector: **tsmom_672h** (different from both the earlier prod `price>sma72` and the 4mo backup `tsmom_672h` bull=6h/bear=7h)
  - Bull: **5h @ 85% conf, shield=ON**, gate `rr12≥2.5% OR rr18≥4.0% cd=16h`
  - Bear: **6h @ 80% conf, shield=OFF**, gate `rr16≥7.0% OR rr30≥4.5% cd=14h`
  - `min_sell_pnl=0.30%, max_hold=12h`
  - Mode T iteration 3 converged; per-regime decomposition: bull +46.23%, bear +70.22%
  - 60d sim total: **+122.59%** vs B&H +20.95% = **+101.64pp alpha** (strongest ETH result of the week)
  - Passes all 4 promotion gates (converged, top-15 plateau unanimous, alpha > baseline, will validate on 4mo in R5)
  - Backup of prior (laptop 2mo+trim) saved to `config/regime_config_ed_prev_prod_20260422.json` + `models/crypto_ed_production_prev_prod_20260422.csv`
  - Per-horizon feature quality (floor ON worked as designed):
    - 5h: 14 features, 2 logret, 1 pysr (return +80.92%, acc 73.5%)
    - 6h: 8 features, 2 logret, 1 pysr (return +76.87%, acc 62.3%)
    - 7h: 30 features, 4 logret, 5 pysr (return +25.35%)
    - 8h: 10 features, 1 logret, 4 pysr (return +24.19%)
  - **Caveat:** remaining matrix variants (#2-#5) could still beat V1 and warrant another promotion cycle tomorrow. Watch the full matrix CSV when it finishes ~14-15h 2026-04-23.

- **Rally-cooldown formula fixed (`crypto_revolut_ed_v2.py:929`).** Previous formula `implied_until = now - bars_ago + cd_h` used wall-clock time instead of actual bar timestamps. Result: cooldown end drifted by `(now - last_closed_bar_time)`, typically 1-59 min but up to 2h+ when data was stale. Symptom after V1 promotion: catch-up scan correctly detected the 14:00 UTC trigger bar (rr18=+5.16%≥4.0%) but computed cooldown_until = 08:24 UTC tomorrow instead of expected 06:00 UTC. Fix: use `trigger_time = df_raw['datetime'][end_idx]` (tz-localized UTC) as the anchor so `implied_until = bar_open_time + cd_h`. Deterministic regardless of wall clock. Cleared stale `rally_cooldown_until` in `config/position_ed_v2_ETH.json` before restart. After fix: cooldown_until = 2026-04-23T06:00:00Z (14:00 UTC + 16h).

- **AB matrix orchestrator launched on desktop at 17:32 CEST** — `tools/ab_matrix_runner.py`. Full factorial: 5 HRST variants (trimOFF/ON × metaOFF/ON × floorON, plus trimON+metaON+floorOFF tiebreaker) + vol-scaled horizon test. Variant #1 finished in 4h 2min (longer than predicted 2.5h). Variants #2-#5 + vol test still running overnight. Audit CSV: `output/ab_matrix_results_<timestamp>.csv` with per-horizon features, shields, gates, Mode T REF returns, meta stats.

- **New CLI flags shipped**:
  - `--no-feature-floor` disables the feature-family floor for A/B comparison
  - `--no-data-update` skips macro + OHLCV downloads at HRST start (used by matrix runner so all variants see identical data snapshot)
  - `--meta-filter P` (from earlier today) walk-forward secondary LGBM; BUYs with meta_prob < P become HOLD

### Closed 2026-04-21 late evening (meta scaffold + consistency tool + 4mo Desktop HRST)

- **Desktop 4-month ETH HRST complete** (log `ed_v1_20260421_171708.log`, ~17:17 → ~20:15, ran `--no-persist` → wrote to `_noprod.*`). Winner: **`tsmom_672h` bull=6h@90% / bear=7h@95%, shield bull=OFF/bear=ON, min_sell_pnl=0.40%, max_hold=8h**, converged Mode T iter 3, bull gate `rr12h≥2.0%/rr30h≥6.0% cd=24h`, bear gate `rr24h≥6.5%/rr30h≥5.5% cd=20h`. 60d sim return +133.64% / alpha +155.91pp / 132 trades / 69% WR. Plateau robust: top-15 Mode S all agree on bull=6h/bear=7h with detector split tsmom (12) vs price>sma72 (3). **Zero parameter overlap with 2-month Desktop winner** — every single parameter differs. Regime-context analysis: 4-month window includes a bear rally phase that isn't representative of today's market, so 2-month is the more applicable configuration despite lower alpha. **Backed up to `config/regime_config_ed_4mo_backup.json` + `models/crypto_ed_production_4mo_backup.csv`** before laptop's 2mo-trim run overwrites `_noprod.*`.

- **Meta-labeling scaffold shipped (`crypto_trading_system_meta.py`).** Standalone research tool, no prod impact. Training pipeline: reads primary model config from `crypto_ed_production.csv`, generates primary signals via `generate_signals()`, builds meta-labels (label=1 iff forward_return(horizon) > 2×fee), walk-forward LGBM training with embargo, evaluates at multiple probability thresholds. Implements López de Prado Ch. 3 meta-labeling. CLI: `python crypto_trading_system_meta.py <asset> <horizon> [--replay N] [--p-thresholds A,B,C] [--sizing]`. Runtime ~15-20 min on 1440h replay (primary signal gen dominates). Output: per-trade CSV at `output/meta_<asset>_<h>h_<timestamp>.csv`. Run not yet executed — queued for Desktop.

- **PySR/HRST consistency tool (`tools/check_pysr_consistency.py`).** 5 checks: (1) PySR JSONs present + metadata; (2) Ordering — JSONs pre-date latest HRST; (3) Production CSV rows' `optimal_features` reference pysr_*; (4) Leakage guard passed in latest log; (5) Functional — every formula parses, every referenced feature exists in current build. CLI: `python tools/check_pysr_consistency.py [--asset ETH] [--horizons 5,6,7,8]`. Exit code 0 = clean, non-zero = at least one [FAIL]. Ran on 2026-04-21 20:30 → **all 5 checks pass for ETH 5/6/7/8h**. Use as pre-flight before promoting any HRST result. Runtime ~5 seconds.

- **ETH HRS winner (overnight, log `ed_v1_20260420_232302.log`, 23:23 → early 2026-04-21)** — superseded by Desktop 12:07 HRST. Noted as stale because it ran before the morning Mode P finished, so used yesterday's PySR JSONs. Weak winner (+11.34pp alpha) attributable to pre-regen PySR, not bad data.

### Closed 2026-04-21 (Mode F shipped + overnight HRS chain + Desktop ETH HRST)

- **Overnight HRS chain (ETH/BTC/SOL/LINK, laptop)** — started 2026-04-20 23:23, SOL+LINK still running at 18:00. Results so far:
  - ETH HRS 23:23 (before morning P finished): `sma168>sma480 6h@90% / 8h@90%` → +28.50%, +11.34pp alpha, 58 trades, 64% WR. **Lower quality than yesterday** because it used stale PySR JSONs.
  - BTC HRS 04:18 → 12:30: `sma24>sma100 bull=8h@85% / bear=7h@65%` → **+36.15%, +23.89pp alpha, 84 trades, 70% WR**. Top-15 plateau unanimous on horizon pair. **+21.26pp alpha improvement vs 2026-04-19 HRST** (that one ran before the 2026-04-19 18:00 feature-additions commit, so it lacked derivatives/stablecoin/orderbook/IV).
  - SOL + LINK HRS pending.

- **Desktop ETH Mode P (03:00-07:59)** — regenerated all 4 PySR JSONs (5h/6h/7h/8h) against current feature set (incl. perp-spot basis + BTC lead-lag added 2026-04-20). Opens door for clean HRST with fresh PySR.

- **Desktop ETH HRST (12:07 → ~15:44, ~3h 37min)** using fresh morning PySR + new features. **Strongest ETH result of the week:**
  - Mode S winner: `sma168>sma480 bull=7h@70% / bear=5h@75%` → **+68.17%, +49.41pp alpha, 115 trades, 70% WR**.
  - Top-15 unanimous on detector, 12/15 bull=7h, very tight plateau.
  - Mode T final (iter 4, reached max_iter without convergence): `min_sell_pnl=0.40% max_hold=10h bull_shield=ON bear_shield=OFF`, total **+98.33%** (baseline +87.19% all-OFF shields).
  - Bull gate: `rr18h≥3.5% OR rr36h≥4.5%, cd=30h` — Mode T re-enabled bull gate (different structure from harmful 8h/12h one; plateau=1.00 across 21,812 STRICT configs). Option C effectively undone but the structurally different gate may be legitimate.
  - Bear gate: `rr14h≥3.5% OR rr24h≥5.5%, cd=24h` (plateau=0.70, borderline).
  - Per-regime contribution decomposition: bear delivers +59.27% on 37% of bars vs bull +27.92% on 63% of bars — bear gate still the main gate-gain driver.
  - **Non-convergence flag + run-to-run variance + Mode T bull-gate swapping from disabled→enabled within 24h = 4-month `--replay 2880` recommended before enabling live.**

- **Mode F shipped (Feature Trim).** Single-letter mode in `crypto_trading_system_ed.py` + button `🧹 Feature Trim (F)` in optimizer bot main menu. CLI entry points:
  - `python crypto_trading_system_ed.py F` — audit prod CSV + PySR formulas, populate `config/disabled_features.json` with Grade 1 (zero selection + zero PySR ref), spare newborns, write.
  - `F --restore` — empty the disabled list.
  - `F --include-newborns` — aggressive; disable newborns too.
  - `F BTC` — use BTC's feature universe (adds SOPR variants to the dead list).
  - Bot flow short-circuits to confirm: Mode F is universal (scans all prod models), no asset/horizon pickers needed.
  - Initial run disabled **65 features** (BTC universe) or **62 features** (ETH universe): 44 on-chain, 13 macro, 5 technical, 3 sentiment, 2 cross-asset. Universe shrinks BTC 193→128, ETH 191→129.
  - **22 newborn features intentionally spared** (`NEWBORN_FEATURES` set in `crypto_trading_system_ed.py`): derivatives-as-feature, stablecoin, orderbook, IV, basis, BTC/ETH intraday lead-lag. Maintainer note: prune from this set after 2-3 HRST cycles confirm newborns as dead.

- **Feature audit tool (`tools/audit_features.py`)** — grades every feature 1-5 (5=≥60% selection, 1=0 selection + 0 PySR refs). Outputs per-category breakdown, orphans list (features used by prod models but missing from current build — these silently fall back to FEATURE_SET_A), and CSV export. Grade-resurrection rule: any feature appearing in ANY PySR formula across any asset auto-bumps from Grade 1 → Grade 2, protecting it from disable.

- **Feature disable mechanism (`config/disabled_features.json` + `_apply_feature_disable()`)** — loaded at end of `build_all_features()` via module-level mtime-cached reader. Strips names from `all_cols` (the LGBM input list) but keeps columns in df (so PySR formulas can still evaluate against disabled features). Verified: 0 pysr_* output features in disabled list, 0 PySR input features in disabled list (by construction via Grade resurrection).

- **`--no-persist` CLI flag (+ Telegram toggle)** — `python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --no-persist` seeds `PRODUCTION_CSV` and `REGIME_CONFIG_PATH` to `*_noprod.*` copies at startup; all writes go there. Safe to run alongside live trader. In optimizer bot: confirm screen now has `🧪 Switch to NO-PROD` toggle; job label tags as `[NO-PROD]`.

- **Optimizer bot main menu: HRS → HRST.** Main 🔧 Full Re-tune button now launches HRST (chains Mode T for shield + gate sweep). HRS still available under Advanced ▸.

- **Rally-cooldown gate verdict (completed 2026-04-20 late evening, fleshed out 2026-04-21):**
  - **Bull rally-gate disabled.** Confirmed harmful across all three test windows: live 30d −7.48pp, OOS 60d −1.31pp, FULL 90d −3.32pp. Previous structure (rr8h≥2.5% OR rr12h≥2.0%, cd=8h) was parameterization artifact (12h branch dominated). Disabled in config 2026-04-20.
  - **Bear rally-gate kept.** Consistently helpful: OOS 60d +4.11pp, FULL 90d +5.59pp. Params `rr10h≥4.0% OR rr16h≥5.5%, cd=36h`.
  - **Drop-gate sweep (alternative design) rejected.** Tested on 90d simulated signals: 30d-live winner ranks 101/126 on OOS 60d, delta −13.50pp. Top-10 OOS ∩ top-10 IS = 0 rules in common. Textbook overfitting. Tool: `tools/backtest_drop_gate.py`.
  - **Deflated Sharpe audit** (`tools/deflated_sharpe.py`, Bailey-López de Prado 2014): live 30d DSR = 0.000 after correcting for 3920-config Mode S sweep. Live return indistinguishable from lucky-config draw. Theoretical Mode S winner at SR=2.5 passes DSR cleanly — skill exists on simulated signals; live execution is the leak.

- **Per-regime Telegram `🚦 Gate` buttons (trader)** — main keyboard now has row 4: `🚦 Gate Bull: ON/OFF` / `🚦 Gate Bear: ON/OFF`. Tap sends `/gate ETH <regime> on|off`. Also fixed silent schema bug: previously `/gate ETH off` only set asset-level `rally_cooldown.enabled` but left per-regime blocks untouched. Now `_set_gate_enabled()` helper propagates to both levels.

- **`tools/audit_features.py --asset X` + per-asset usage matrix** — saved `models/feature_usage_by_asset.csv` showing feature selection count per (feature × asset). Findings: 9 universal features picked by all 5 assets (price_to_sma100h, logret_120h, hour_cos, sma20_to_sma50h, logret_240h, xa_dax_relstr5d, m_sp500_chg1d, xa_sp500_relstr5d, adx_14h). `pysr_1..5` each appear in every asset. Only 1 exclusive feature per asset (XRP: `m_vix_chg5d`). LINK uses more exclusive macro features than others, consistent with its weak model diagnosis.

### Closed 2026-04-20 evening (literature review + infrastructure sprint)

**Triggered by full 2024-2025 crypto algo-trading literature review (12 areas covered).**
Cross-referenced against tested/rejected items in this doc. Actioned all untested items
that were "lit-endorsed" and feasible tonight.

- ~~**Deflated Sharpe Ratio audit (Bailey & López de Prado 2014)**~~ — `tools/deflated_sharpe.py` shipped. Computes Deflated SR + PSR + E[SR_max] under null given N-trial multi-testing. **First audit result: live ETH 30d DSR = 0.000** (23 trades, per-trade SR=0.057, after correcting for N=3920 Mode S sweep). Live return indistinguishable from a lucky-config draw. Theoretical Mode S winner (simulated SR≈2.5) passes DSR cleanly — skill exists on simulated signals; live execution is the leak.
- ~~**Perp-spot basis feature (Test 2)**~~ — Historical perp hourly klines now downloaded via Binance `/fapi/v1/klines`, stored as `perp_close` in `derivatives_<asset>.csv` alongside funding + OI. `build_all_features()` computes 3 new features: `deriv_basis = (perp-spot)/spot`, `deriv_basis_chg1d` (24h change), `deriv_basis_zscore` (168h rolling). Feature verified on ETH (191 features total vs 185 previously). All 5 assets have perp klines downloaded (BTC/ETH/XRP/SOL/LINK).
- ~~**Cross-asset BTC→ETH intraday lead-lag features (Test 5)**~~ — Literature-backed (BTC leads ETH 5-30min; ETH leads alts). At 1h resolution we add 3 lagged BTC return features (`xa_btc_lag1h/2h/3h`) for ETH/XRP/SOL/LINK, plus ETH lags for XRP/SOL/LINK. Matches the existing daily `xa_*_relstr5d` pattern but at intraday granularity.
- ~~**XRP full pipeline enablement**~~ — Previously XRP couldn't run Mode P (missing 8640 rows after sparse-feature dropna). Fix trio: (1) added `sparse_prefixes = ('gp_', 'deriv_oi_', 'ob_', 'avg_iv', 'iv_skew', 'stable_mcap_', 'whale_')` to `pysr_discover_features.py` NaN→0 fill block (mirrors main `ed.py`); (2) extended `onchain_asset_map` to include `'XRP': 'xrp'`; (3) extended derivatives download loop in `download_macro_data.py` to `['BTC', 'ETH', 'XRP', 'SOL', 'LINK']`. Post-fix: XRP data pulled (hourly +610 candles, derivatives 37697 rows, on-chain 1571 days with 4 metrics — active_addresses, mvrv, fees_native, tx_count). **XRP HRST completed**: winner `sma168>sma480` 5h@85% / 7h@90%, +19.96%, +14.70pp alpha on 60d. H1 was flat / H2 carried the alpha — concerning. Bull gate persisted from iter 1 tiebreak but 0 STRICT winners in iter 2/3 (spurious).
- ~~**SOL partial pipeline refresh**~~ — Hourly data refreshed (+17 candles), derivatives downloaded (37697 rows + perp klines). On-chain BLOCKED: CoinMetrics community API returns HTTP 403 for SOL (free tier excludes). Yesterday's SOL HRST was run on stale feature set (177 features, pre-2026-04-19 18:00 commit). Re-run queued for tonight's overnight chain.
- ~~**LINK full pipeline enablement**~~ — Hourly data refreshed (+19 candles), derivatives (37697 rows), on-chain (1570 days with 3 metrics — active_addresses, mvrv, tx_count). Yesterday's LINK HRST was stale. Re-run queued for tonight.
- ~~**Rally-cooldown gate empirical validation — bull OFF, bear KEPT ON**~~ — `tools/backtest_drop_gate.py` shipped (90d OOS test on cached signals). **Bull gate (rr8h≥2.5% OR rr12h≥2.0%, cd=8h) confirmed harmful**: −7.48pp on live 30d, −1.31pp OOS 60d, −3.32pp FULL 90d. **Disabled** via `ETH.bull.rally_cooldown.enabled: false` + cleared active cooldown timer. **Bear gate (rr10h≥4.0% OR rr16h≥5.5%, cd=36h) confirmed helpful**: +4.11pp OOS 60d, +5.59pp FULL 90d. Kept ON. **Drop-gate sweep overfits**: 30d-live winner (−2%/9h/24h) ranks 101/126 on OOS 60d (−13.50pp). Verdict: don't ship static gates without multi-window OOS validation.
- ~~**Live-vs-simulated gap discovered**~~ — Simulated 30d IS baseline (no gate, current policy): **+27.11%** vs actual live 30d: **+2.54%**. Gap **~24.5pp**. Decomposition: ~7.5pp bull rally-gate drag (now eliminated), ~17pp unaccounted execution friction (slippage, partial fills, manual interventions, clock drift, possibly other). Biggest lever for tomorrow isn't more features — it's closing the execution gap.
- ~~**Per-regime rally-cooldown Telegram UX**~~ — Added 4th button row: `🚦 Gate Bull: ON/OFF` + `🚦 Gate Bear: ON/OFF` (mirrors shield's per-regime pattern). New CLI subcommands: `/gate ETH bull on|off` + `/gate ETH bear on|off`. `/gate` status view now shows per-regime state + thresholds. **Fixed silent schema bug**: previously `/gate ETH off` only set asset-level `rally_cooldown.enabled` but left per-regime blocks untouched — trader was still reading `bull.rally_cooldown.enabled: true`. Now `_set_gate_enabled()` propagates to both levels. Helper `_gate_on_for_regime(cfg_asset, regime)` added.
- ~~**`--label-threshold` CLI flag on ed.py**~~ — Module-level `LABEL_THRESHOLD_PCT` override at `crypto_trading_system_ed.py:312`. When set via `--label-threshold 0.01`, swaps label from `fee_aware` (ret > 2×fee) to `ret > X`. Output auto-redirected to `models/crypto_ed_production_lt<pct>.csv` so prod untouched. Shipped for Desktop test A (large-upside label). `--help` updated.
- ~~**PySR sparse-feature fix for XRP/SOL/LINK**~~ — Extended GDELT-only NaN→0 block in `pysr_discover_features.py` to cover all sparse prefixes. Without this, stablecoin (1yr) + OI (30d) features killed the dropna down to 7970 / 984 rows. Post-fix XRP had full 12-month window.
- ~~**`extend_caches_90d.py` detector fix**~~ — Was hardcoded to `tsmom_672h`; now reads `regime_detector.params.name` from config. Current ETH config uses `sma168>sma480`.

### Closed today (2026-04-19)

- ~~**Asset universe prune**~~ — dropped DOGE / ADA / AVAX / DOT (4 assets). Config, ASSETS dicts in 4 Python files, optimizer bot list, and 15 stale production-CSV rows all cleaned. Kept 5: ETH (active), BTC / SOL / LINK / XRP (standby/testing).
- ~~**Strip "Doohan" labels from Ed code**~~ — removed obsolete "Doohan" references from user-facing labels / docstrings / prints in `crypto_trading_system_ed.py` (module header, "ED OPTIMIZATION" print, Mode D docstring), `crypto_live_trader_ed.py` ([DOOHAN] → [ED], error messages), `crypto_trading_system_ein.py` / `_eli.py` / `_ed_v3.py` (CSV output filenames: `diagnostic_results_doohan_*` → `diagnostic_results_{ed|ein|eli|ed_v3}_*`). Internal variable names (`DOOHAN_GAMMA_MIN` etc.) and dict keys (`'source': 'doohan'`) left — implementation details, not labels; would require broader refactor.
- ~~**SOPR NaN fix (misdiagnosis)**~~ — Re-audit of BTC HRST log showed SOPR is NOT filtered. The two "Columns:" log lines I initially read were for DIFFERENT files (BTC `onchain_btc.csv` with SOPR, ETH `onchain_eth.csv` without). SOPR is loaded and reaches LGBM importance ranking; just ranks below 1% importance. No fix needed.
- ~~**Fee consistency audit**~~ — Grep verified: all `0.0011`/`0.0022` hardcoded fees exist only in `archive/`. Active files use `BACKTEST_FEE_PER_LEG = 0.0005` consistently. `TRADING_FEE_BASE = 0.0009` remains only as pure-taker documentation / label semantic.
- ~~**Telegram optimizer bot — menu buttons**~~ — Added `G - Rally-cd (cache)` to advanced menu. Relabeled `T - Threshold+G` and `HRST - Full+T+G` to make the chaining explicit.
- ~~**Signal nondeterminism audit**~~ — All models, Optuna, numpy use `seed=42`. Two likely sources identified: (a) LGBM `device='gpu'` (CUDA floating-point reduction not reproducible); (b) `joblib.Parallel(n_jobs=...)` at 4 sites (task completion order nondeterministic). Repro protocol documented: pin commit, freeze data, set LGBM `device='cpu'` + `n_jobs=1`. Not a code change; 5-10pp run-to-run variance accepted as speedup tradeoff. Run HRS 2-3× when a winner looks marginal.
- ~~**ETH on-chain feature audit**~~ — MVRV Grade 4, exchange_netflow Grade 3, 38 others weak (<1% LGBM importance, not crowded-out)
- ~~**Iterative T ↔ G convergence**~~ — `--max-iter 4` shipped; ETH/BTC both converge in 2-3 iterations
- ~~**Per-regime gate (code + ETH config)**~~ — bull ungated (no plateau winner), bear `rr8h≥2.5% OR rr30h≥5.0% cd=18h`
- ~~**Bear horizon test (ETH)**~~ — 8h best on 60d by +10pp+ over 5/6/7h
- ~~**Disaster brake (code + ETH config)**~~ — `disaster_brake_pct: 5`, fires 0 times in 60d (dormant insurance)
- ~~**BTC HRST**~~ — completed; winner `vol_calm` detector + bull=8h/bear=5h + gate rr20h/rr24h cd=48h
- ~~**BTC enablement decision**~~ — KEEP DISABLED (ETH makes ~3× BTC's return per $; correlation too high for diversification; pipeline half-failed with 6h/7h "no valid trials"; SOPR loaded but below 1% LGBM importance)
- ~~**Shield quick-release evaluation**~~ — 0 fires on 60d at 95%/4h, kept in config as armed insurance; defaults flipped to opt-in only
- ~~**Mode G cleanup**~~ — removed HRSTG/HRSG/DVRSG/RSG; added cache-freshness warning
- ~~**Shield UX rework**~~ — per-regime `🛡 Bull:` `🛡 Bear:` buttons replace cramped `Shield B/B`
- ~~**`BACKTEST_FEE_PER_LEG = 0.0005` refactor**~~ — single source of truth for sim fees across 24 active files

### Completed (2026-04-19)

- **BTC HRST complete** — `HRST BTC 5,6,7,8h --replay 1440` ran successfully after last night's Mode P. Winner: detector `vol_calm` (Andersen-Bollerslev deseasonalized vol), bull=8h@95% shield=ON, bear=5h@80% shield=ON (both shields ON — asymmetric from ETH), min_sell_pnl=0.35%, max_hold=12h, gate rr20h≥3.0% OR rr24h≥4.5% cd=48h (conservative, long cooldown). BTC still `enabled: false` pending decision. BTC bull=8h vs bear=5h is INVERSE of ETH (6h bull / 8h bear).
- **`BACKTEST_FEE_PER_LEG = 0.0005` constant shipped** — single source of truth for sim fees across 24 active files. Live trader is maker-first (~95% maker/~5% taker blend ≈ 1.6 bps/leg measured; 5 bps adds 3× safety margin). Replaced Mode G's hardcoded `FEE = 0.0011` and Mode T's implicit `0% fee`. Both were inconsistent — Mode G's 11 bps made gates look free (fee-drag credit) while Mode T's 0% made them look like winners on the wrong baseline. At realistic 5 bps, single gate was a loser; per-regime gates recover. Standalone backtest scripts (backtest_rally_cooldown*, backtest_sl_variants, compare_*, audit_v6*, etc.) all updated to 0.0005. Scripts that imported TRADING_FEE from ed.py now alias BACKTEST_FEE_PER_LEG for the fee. Label generation (`2 * TRADING_FEE`) untouched at 0.22% break-even — labels represent pessimistic-fee training targets, not sim cost.
- **Per-regime rally-cooldown gate (code shipped + bear config active)** — Schema: `bull.rally_cooldown` / `bear.rally_cooldown` per regime, asset-level `rally_cooldown` kept as legacy fallback. Trader helper `_rally_cfg_for_regime(trading_cfg, regime_label)` in `crypto_revolut_ed_v2.py` reads regime-scoped with fallback. Mode T/G's `_sweep_rally_cooldown` accepts `regime_filter='all'|'bull'|'bear'` — when filtered, gate only fires on that regime's bars AND writes to regime block. Mode T's chain runs bull+bear sweeps independently. **2026-04-19 ETH sweep result**: bear gate = rr8h≥2.5% OR rr30h≥5.0% cd=18h (written). **Bull gate left ABSENT** — 445 configs beat baseline but 0 passed plateau filter (robust region missing in current uptrend regime). Not a bug; rally-cooldown is mean-reversion logic that doesn't help in trend regimes. Standalone test proved +18pp gain from per-regime vs single on 60d maker sim. Unit tests: `test_per_regime_gate_trader.py` 10/10 pass.
- **Iterative T ↔ G convergence** — Mode T wraps the shield + gate sweep in a `max_iter=4` loop. Each pass: snapshot config, sweep shield with current gate applied (via `_sim_horizon(rally_cfg=...)` — new param), pick shield, sweep gate with new shield, pick gate, write. Check fingerprint vs prior pass; break on match. ETH and BTC both converged in 2-3 iterations — proves the coupling is tight but not drift-prone on current data. New CLI flag `--max-iter N` (default 4; pass 1 for single-pass legacy behavior).
- **Mode T chains per-regime G — fee fix flipped the verdict** — Before fee refactor, Mode G's 0.11% taker fee made the rally gate look like a +16pp winner vs no-gate; Mode T's 0% fee made per-regime look like +18pp. After refactor to 5 bps/leg: per-regime gate still wins vs single on 60d; bull's best configs fail plateau robustness filter (don't ship); bear gate refined to `rr8h≥2.5%/rr30h≥5.0% cd=18h` (was 16h) reflecting the slightly higher fee penalty on over-blocking.
- **Disaster brake** (TODO 6) — `disaster_brake_pct` config key (ETH=5% currently). Trader force-sells on unrealized PnL ≤ −brake_pct, bypassing shield. `test_disaster_brake.py` — 13/13 pass. Fires 0 times on 60d backtest (max historical DD was −2.59% << 5% threshold). Dormant insurance against rare catastrophic moves. Originally suggested 7% (validated in 2026-04-14 backtest); user set 5% — slightly more aggressive but still safely above historical worst-case.
- **Shield quick-release (evaluated, removed)** — Added early morning as response to today's painful trade; implemented in trader + Mode T/G + 12/12 unit tests. Then tested empirically: on 60d maker-sim, 95%/4h and 95%/3h both fired **0 times** (identical to OFF). 90%/5h fired once and cost −0.36pp. **Verdict: 95%/4h is theater**, not insurance. Removed from config; defaults in trader flipped to `enabled=False` so missing config = off (not silently on). Code kept for future opt-in via config. User kept 95%/4h back in config as "armed insurance" for events rare enough to not appear in 60d.
- **Shield UX** — Main button split from `🛡 Shield B/B: ON/OFF` into two independent toggles: `🛡 Bull: ON` / `🛡 Bear: OFF` on row 3 of 4; Setup alone on row 4. Tap sends `/hold bull` or `/hold bear`. `/hold` bare shows state + threshold + hint. Unit tests `test_shield_ux.py` 13/13 pass.
- **Mode G cleanup** — Dropped `HRSTG`, `HRSG`, `DVRSG`, `RSG` from `VALID_MODES` (redundant since T chains G). Added cache-freshness warning to standalone Mode G. Purged stale `output/mode_g_*.csv`, kept new `output/rally_cd_*.csv`.
- **Telegram optimizer bot** — Added `G` / `HRST` / `DVRS` to MODES dict + MODE_TIME_EST + REPLAY_MODES. Labels updated (T="Threshold + chain G", HRST="Full + Threshold (incl. G)").
- **On-chain feature audit (ETH)** — 2026-04-18 HRS was the first run after 2026-04-17 wiring. Audit: `oc_mvrv_chg1d` Grade 4 (66.7%, 4/6 ETH rows), `oc_exchange_netflow_chg5d` Grade 3 (1/6, 8h only). Other 38 derivatives below threshold. Reference Card updated.
- **Production report script** — `report_production.py` generates comprehensive 30d + 60d stats for the current asset config: strategy return vs B&H, alpha, trade count, win rate, avg win/loss, expectancy/trade, profit factor, Sharpe-like, best/worst trade, max drawdown, time in market, brake fires. Current ETH (60d, 5 bps fee): +61.56% strategy vs +19.42% B&H = +42.14pp alpha, 75% WR, 8.7 profit factor, −5.29% MDD.
- **Shield quick-release added to trader + Mode T + Mode G sim** — New config block `shield_quick_release: {enabled, min_sell_conf, max_hours}` (defaults true/95/3h). When shield is ON and model flips SELL at ≥min_sell_conf within max_hours of entry, bypass shield. Standalone `test_shield_variants.py` + `test_window_sweep.py` run on 60d cache. Verdict: default 95%/3h fires 0 times in 60d backtest (conservative), does not impact winner. Today's failure mode (entry 21:55, first 97% SELL at 02:00 = 4h) wouldn't trigger with defaults — 90%/5h would have caught it.
- **Shield variant comparison (standalone)** — `test_shield_variants.py` compared 8 variants (no shield, current, QR variants, persistence variants) on 60d cache. All persistence rules (E/F/G/H) LOST 3-5pp vs current shield. QR variants tied or lost marginally. Conclusion: current shield is optimal on 60d historical; today's loss was shield's "insurance premium" for a systematic edge. **Today's failure is NOT in the cache** (cache ends 2026-04-18 12:00, BUY was 21:55) — backtest couldn't evaluate fixes against the specific event.
- **Mode G cleanup (removed chain-mode cruft)** — Dropped `HRSTG`, `HRSG`, `DVRSG`, `RSG` from `VALID_MODES` (redundant since T chains G internally). Help text updated. Interactive menu prompt updated. Cache-freshness guard added to standalone Mode G: prints warning + skips asset if `crypto_ed_production.csv` is newer than `eth_sl_signals_*.pkl`. Purged 5 stale `output/mode_g_*.csv` files, kept 3 new-format `output/rally_cd_*.csv`.

### Completed (2026-04-18)

- **PySR drift detection + full refresh (HRSTG chain)** — Discovered ETH 6h/8h production rows (trained 2026-04-10) used OLD pre-Apr-11 PySR formulas, but `_compute_pysr_features()` reads the CURRENT JSON at inference → silent feature drift on both LIVE horizons. Fix: `P ETH 5,7h` to deepen the stale ones with current PySR code (commits `0a3ba33`/`df40043`/`cdfca63` from 2026-04-11: maxsize 15→25, iterations 40→100, multi-run + island isolation), then full `HRS ETH 5,6,7,8h --replay 1440` to retrain against current PySR. All 4 horizons now internally consistent. Post-HRS winners: 5h (+57.86%), 6h (+63.16%), 7h (+6.80%, weak but not live), 8h (+21.32%). Mode S picked detector=tsmom_672h, bull=6h@95%, bear=8h@80%.
- **Per-regime hold-shield (new capability)** — Shield ON/OFF now splits by regime. Schema: `bull.hold_shield` / `bear.hold_shield` (per-regime), shared `min_sell_pnl_pct` / `max_hold_hours` at asset level. Trader `crypto_revolut_ed_v2.py` reads via new `_shield_on_for_regime(trading_cfg, regime_label)` helper with legacy fallback to asset-level `hold_shield`. `/hold` Telegram command extended: `/hold`, `/hold on|off`, `/hold bull on|off`, `/hold bear on|off`. Button label shows `🛡 Shield B/B: ON/OFF`. Unit tests in `test_per_regime_shield.py` (12/12 pass).
- **Mode T redesigned — per-regime shield sweep** — Now sweeps threshold × failsafe × bull_on × bear_on (4 on/off combos), picks the quadruple maximizing bull+bear total return. Writes `bull.hold_shield`, `bear.hold_shield`, shared `min_sell_pnl_pct`, `max_hold_hours`. Current ETH winner: `bull_shield=ON, bear_shield=OFF, min_sell_pnl=0.60%, max_hold=12h` → +115.50% vs all-OFF +92.05% (delta +23.45pp). All top-8 combos had bull=ON/bear=OFF — signal unambiguous.
- **Mode T chains rally-cooldown (T→G integration)** — After writing shield config, T now merges its fresh bull/bear signals via `_merge_tagged_signals()` (regime tag per bar via configured detector) and calls the shared `_sweep_rally_cooldown()` helper. No stale-cache issue: T and G share one signal stream. T total runtime ≈ 3-5 min. Rally-cooldown winner today: `rr10h ≥ 2.5% OR rr30h ≥ 6.0%, cd=20h` (H1=+31.01%, H2=+22.22%, REF=+60.12%, worst_dd=+4.55%, plateau=1.00). Mode G standalone kept for cache-fed fast iteration.
- **Mode G simulate() per-regime policy awareness** — Previously hardcoded `MIN_SELL_PNL_PCT=0.005, MAX_HOLD_HOURS=10` + cache's `conf_threshold`. Now reads `bull/bear.min_confidence`, `bull/bear.hold_shield`, shared `min_sell_pnl_pct`, `max_hold_hours` from `regime_config_ed.json`. Each bar's policy keyed by `s['regime']`. Reflects true live-trader behavior.
- **Mode G `--rank recent|balanced` flag** — Default `recent` (H1-focused tiebreak: `pnl_H1 − 0.5 × |dd_H1|`). `balanced` uses prior behavior (`avg_pnl_halves − 0.5 × worst_dd`). CLI + chain-mode wiring (G, RSG, HRSG, DVRSG, HRSTG).
- **_merge_tagged_signals bug fix** — First integrated T→G run tagged 100% of bars as bull. Root cause: detector dict keys are naive `pd.Timestamp`; merge was normalizing to UTC-aware before lookup → every `dt not in ind` → default `True`. Fixed by using naive Timestamps for detector lookup (matches `extend_caches_90d.py` pattern). Post-fix: 1442 bars tagged bull=888 / bear=554 — proper ~60/40 split.
- **extend_caches_90d.py modernized** — Was pulling bear row from archived `crypto_doohan_v1_6_production.csv`. Now reads current `crypto_ed_production.csv` for both regimes, horizons + conf thresholds sourced from `regime_config_ed.json`. Cache rebuilt: 2185 hourly signals, fresh from HRS-retrained models.

### Completed (2026-04-17)

- **Engine Reference Card** — Built from live audit of `crypto_ed_production.csv` (48 models) + history. Feature grades (5-1) across technical/macro/cross-asset/sentiment/PySR/on-chain/derivatives. See "Engine Reference Card" section above.
- **ETH Mode D `--replay 1440` (late evening)** — New prod row winners written: ETH 5h (RF+LGBM, APF higher than pre-fix), consistent with new PySR. Exact numbers superseded by 2026-04-18 HRS rerun.

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
