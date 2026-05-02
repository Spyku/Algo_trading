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

- **5h, 6h, 7h, 8h** — core band, default sweep in Ed ([line 5428](crypto_trading_system_ed.py#L5428)). Current production: ETH bull=6h@85%, bear=5h@65% (live; updated 2026-04-30 to 2-detector 6h/5h).
- **4h — REVIVED 2026-05-01 (was previously "structurally broken").** ETH 8h HRST 4-16h Mode V on 1440h replay produced ETH 4h winner +41.23% / 143 trades / 69% WR (refined RF+LGBM w=88h). XRP HRST 2026-05-02 picked bull=4h@80% as winner. The earlier "broken post-embargo" verdict was specific to the 2026-03-24 data snapshot and pre-feature-additions pipeline. 4h is asset-conditional, not universally dead — but it never wins the regime joint sweep over 5-8h on ETH (Mode S TOP 15 from 8h HRST: zero entries with bull=4h; 7h dominates).
- **9h, 12h, 16h — TESTED 2026-05-01 (8h HRST), VIABLE.** Single-horizon Mode V: 9h +31.36%/62%WR (borderline), 12h +33.27%/67%WR/42 trades (real), 16h +30.79%/77%WR/26 trades (high WR / thin). **Mode S TOP 15 plateau**: ALL 15 entries had bull=7h; bear migrated to 16h (×7), 12h (×5), 8h (×2). Bear 5/6/7 absent from TOP 15. Mode S WINNER: `tsmom_672h bull=7h@85% / bear=16h@75%` → Mode T REF +68.81% / 24 trades / WR 92% (gates active, both shields OFF) — competitive with claimed live ~+70%. **Promotion pending**: validate on a fresh window before flipping live bear from 5h to 16h.
- **10h — incidental rows in CSV** from earlier non-default sweeps; no recent dedicated test.
- **14h — barely tested.** Not in 8h HRST horizons (4,5,6,7,8,9,12,16); only 2 legacy CSV rows.

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
- ~~**4h horizon. Structurally broken post-embargo.**~~ **REVISED 2026-05-01.** 8h HRST showed ETH 4h Mode V winner +41.23% / 143 trades / 69% WR; XRP HRST picked 4h@80% as bull. Original "broken" verdict was tied to 2026-03-24 data + fewer features. 4h still loses the joint regime sweep on ETH (Mode S TOP 15 zero entries with bull=4h; 7h dominates), but it's no longer "do not revive" — keep in the default sweep set.
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

- **Per-regime feature set.** Open question from engine reference: should bear use more macro features, bull more technical? The asymmetry already happens organically (6h model: 7 features, mostly PySR+technical; 8h model: 32 features, kitchen sink) but hasn't been deliberately tested as a design choice. **Effort: low-medium (code change to Mode D).**
- **Multi-horizon ensemble emergency-exit** (4th angle from 2026-04-27 forensic). Force exit when 5h AND 8h both flip SELL within 1h. Per-horizon signal cache `data/eth_per_horizon_signals_90d.pkl` already exists. Distinct from rejected 5-min price-action triggers and from T1b entry-side ensemble (which was tested + shelved 2026-04-27). **Effort: 1-2h.**

**Recently moved out of "untested" (now tested + shelved):**
- ~~Meta-labeling~~ — tested 2026-04-27 in AB matrix variants D and E. R3 RESOLVED: E (meta on strong base) lost −2.12pp vs A; D (meta on weak base) gained +6.26pp but still lost to A overall. Door open for SOL/BTC/XRP retests but those assets are also shelved.
- ~~Triple barrier as exit overlay (T6)~~ — tested 2026-04-26: +10.48pp 60d / +1.24pp 90d / 47 trades. Not promoted (60d gain didn't survive to 90d).
- ~~Triple barrier as TRAINING LABEL~~ — tested 2026-03-14 on BTC (`archive/testing_literature.csv`). Direct same-day baseline comparison: baseline 4h +57.22% / 8h +74.03% vs triple_barrier_label 4h +29.22% / 8h +22.53%. Lost on every dimension (return −28pp / −51pp, WR −9pp / −10pp, accuracy −16pp / −13pp). Earlier framing of "+29% standalone, never tested with current pipeline" was misleading — the 2026-03-14 test had a same-day baseline that was 30-50pp better. SHELVED.

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
| `crypto_trading_system_ed.py` | **Production** | Single-file Ed V1.0 engine. **Merged 2026-05-02**: previously split into entry-point + `_engine.py` + `_parallel_p.py` (parallelism wrapper) + `_wrapper`; all merged back into one file. Modes P/D/V/H/S/R/HRS/HRST/T/G/F. Embargo-fixed grid (2×3×4×3=72 evals after trim), 50-trial Optuna refine, PySR symbolic features. Built-in parallel paths for Modes V/S/T (loky workers, LGBM cpu inside parallel sections, hybrid GPU+CPU refine dispatcher). Reads/writes `crypto_ed_production.csv` and `regime_config_ed.json`. Pre-merge snapshots in `archive/crypto_trading_system_ed_engine_pre_merge_20260502.py` + `_parallel_p_pre_merge_20260502.py` + `_wrapper_pre_merge_20260502.py`. |
| `crypto_trading_system_ed_noprod.py` | Active wrapper | Safety wrapper for research runs. Sets `MODELS_CSV_OVERRIDE` env var before importing the engine, monkey-patches `PRODUCTION_CSV` + `REGIME_CONFIG_PATH` to `*_noprod.*` paths, seeds the noprod files from prod on first run. Equivalent to engine's built-in `--no-persist` flag; kept as the file-based alternative. |
| `crypto_revolut_ed_v2.py` | **Live** | Ed V2 auto-trader — maker orders (0% fee) with market fallback. Penny-improvement pricing: buy/sell at `bid+0.01`. `post_only` ensures maker. Stale order cleanup, NTP clock sync, locked funds detection. Reads `regime_config_ed.json`. |
| `crypto_live_trader_ed.py` | **Live** | Ed signal generation — regime-aware. `detect_regime()` + `generate_regime_signal()`. Reports current market price (not label-shifted). |
| `crypto_trading_system_meta.py` | Research | Standalone meta-labeling research tool. Imported lazily by `crypto_trading_system_ed.py` when `--meta-filter P` flag is used. R3 RESOLVED: AB matrix variants D+E showed meta hurts on strong baselines (E vs A −2.12pp). Kept available for SOL/BTC/XRP retests. |
| `start_ed_v2.bat` | **Live** | Launches Ed V2 trader with auto-restart + log tee. Auto-detects Desktop/Laptop venv. |
| `crypto_optimizer_bot.py` | **Live** | Telegram bot for remote optimization. Inline keyboard menus. Sequential job queue, subprocess execution. Separate bot token (`config/telegram_optimizer_config.json`). Below-normal priority. Spawns `crypto_trading_system_ed.py`. |
| `start_optimizer.bat` | **Live** | Launches optimizer bot with auto-restart + log tee. |
| `hardware_config.py` | Active | Auto-detects Desktop (26 workers) / Laptop (14 workers) |
| `download_macro_data.py` | Active | Downloads VIX, DXY, S&P500, NASDAQ, Fear&Greed, on-chain (CoinMetrics), derivatives (Binance), orderbook+IV snapshots, etc. |
| `pysr_discover_features.py` | Active | Offline PySR discovery. Historical window (months 12→6 ago) to avoid leakage with Mode D. Outputs `models/pysr_{ASSET}_{H}h.json` with `discovery_method: "historical"`. Mode P delegates to this module. |
| `tools/pysr_discover_regime.py` | Active | PySR regime formula discovery (binary regime label). Historical window. Anti-leakage. |
| `tools/backtest_regime_master.py` | Active | Hand-crafted regime detector backtest. 21 detectors × all horizon pairs. |
| `cfd/ib_auto_trader.py` | Live | DAX CFD trader (Broly 1.2) |
| `cfd/ib_auto_trader_test.py` | Live | S&P 500 CFD overnight trader |
| `crypto_trading_system_ein.py`, `crypto_trading_system_eli.py`, `crypto_trading_system_ed15.py` | **Archived 2026-05-02** | 15-min and 30-min candle variants. All in `archive/`. Stale (missing parallel wrapper + post-2026-04-25 audit fixes). To revive 15-min testing → clone current `crypto_trading_system_ed.py` into `crypto_trading_system_ed15.py` (see Pending Work TODO). |
| `report_production.py`, `sell_btc_now.py`, `test_btc_horizons.py`, `test_btc_accuracy.py` | **Archived 2026-05-02** | Standalone helpers. Moved to `archive/` during folder cleanup. Revive if needed. |
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

## Current Regime Config (Ed V2 — 2026-04-26)

```json
{
  "ETH": {
    "detector": "named:price>sma72",
    "bull": "6h@80%/$12k, shield=ON, gate=rr20≥4.0% OR rr24≥4.5%, cd=12h",
    "bear": "7h@85%/$12k, shield=ON, gate=rr30≥9.0% OR rr36≥9.0%, cd=48h",
    "shared": "min_sell_pnl=0.5%, max_hold=10h",
    "disaster_brake_pct": 0,   // currently OFF (key absent from JSON; brake_pct=0 disables). M-04+N-03 fixes are in place — user opted to keep disabled 2026-04-25.
    "backtest_fee_per_leg": "0.0005 (5 bps, realistic maker blend)",
    "enabled": true
  },
  "BTC":  { "enabled": false, "note": "HRST done 2026-04-19; shelved — opportunity cost vs ETH" },
  "SOL":  { "enabled": false, "note": "HRST done 2026-04-19; shelved" },
  "LINK": { "enabled": false, "note": "standby" },
  "XRP":  { "enabled": false, "note": "standby; decorrelation candidate (~0.60 corr w/ ETH)" }
}
```

**Promoted 2026-04-26 14:22 from AB matrix variant A** (`A_floorON_trimOFF`) — 60d sim Mode T REF +66.79% (beats live HRST by ~+10pp). Mode T converged iter 2. Backups: `regime_config_ed_pre_A_20260426.json` + `crypto_ed_production_pre_A_20260426.csv`.

### Training-time feature pipeline knobs (production state 2026-04-26)

| Knob | State | Where | Effect |
|---|---|---|---|
| **Trim** (Grade-1 disable list) | **OFF** | [config/disabled_features.json:3](config/disabled_features.json#L3) `"enabled": false` | 65 Grade-1 features re-enter LGBM ranking pool (matrix B↔A showed trim costs −21.4pp; no runtime gain — features are computed regardless) |
| **Always-disabled** (sparse-history quarantine) | **ON** | `always_disabled_exact` block (7 features) | `deriv_oi_*`, `ob_imbalance`, `spread_bps`, `avg_iv`, `iv_skew` excluded regardless of trim flag — too sparse for clean training. Re-enable tests scheduled G1 2026-05-22 / G2 2026-06-18. |
| **Feature floor** (logret + pysr minimum) | **ON** | [crypto_trading_system_ed.py:337](crypto_trading_system_ed.py#L337) `FEATURE_FLOOR_ENABLED = True`, MIN_LOGRET=2, MIN_PYSR=1 | Every selected feature subset must include ≥2 logret + ≥1 pysr feature. Prevents Mode D's Optuna refine from landing on trend-blind vol-only local optima. CLI override: `--no-feature-floor`. |

**Asset universe pruned 2026-04-19.** Dropped DOGE / ADA / AVAX / DOT (weak priors, no diversification). Kept 5 (ETH active + BTC/SOL/LINK/XRP standby). Config blocks removed from `regime_config_ed.json`; ASSETS dicts pruned in `crypto_trading_system_ed.py / ed_v3.py / ein.py / eli.py`; ASSETS list pruned in `crypto_optimizer_bot.py`; 15 stale model rows removed from `crypto_ed_production.csv`. PySR JSONs + hourly data CSVs kept for later revive.

**2026-04-19 ETH live backtest (60d, 5 bps/leg fee):** +61.56% strategy vs +19.42% B&H, **+42.14pp alpha**. Win rate 75%, expectancy +1.34%/trade, profit factor 8.7, max drawdown −5.29%. Disaster brake (5%) and quick-release (95%/4h) both inert in backtest. Bull gate intentionally absent — STRICT filter rejected all 445 candidates that beat baseline (no plateau-robust region). Bear gate shipped from iterative T↔G convergence with realistic fee.

**2026-04-18 full refresh (HRSTG pipeline):** PySR deepened 2026-04-11 (`maxsize 15→25, iterations 40→100, multi-run + islands`) but 6h/8h production models were still trained against old PySR formulas → silent feature drift. Fixed: Mode P on 5h/7h, then HRS on all 4 horizons, then per-regime T (shield ON bull, OFF bear), then G (rally-cooldown). Per-regime shield split beat single-flag both-ON by +23.45pp (+115.50% vs +92.05% all-OFF baseline over 60d, 0% fee sim).

**Prior history:** ETH HRS 2-month (2026-04-07) initially picked bull=6h@90% / bear=7h@75%. After R→S handoff fix + Option C joint detector sweep, ETH RS rerun selected `sma168>sma480` bull=7h@75% / bear=8h@85% → Mode S +60.72%. 2026-04-18 HRSTG rerun switched detector to `tsmom_672h` and confidences to 95%/80%.

## Ed Backtest Results (2026-03-31) — historical, superseded

Initial Ed regime-switching backtests from the system's first week. These numbers are pre-embargo-fix-cascade, pre-AB-matrix, pre-feature-additions (no on-chain wired, no derivatives, no PySR refresh). **Subsumed by the Engine Reference Card above and the Closed sections below.** Kept here for audit-trail of the progression: BTC `sma48>sma200 7h@95%/8h@90% +50.35%`, ETH 2mo `6h@85%/8h@65% +70.01%`, ETH 4mo `rsi>45 6h/7h +89.82%`, SOL `sma_cross 6h@95%/8h@90% +31.60%`, LINK best 5h +3.97% (rejected).

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
12. **Asset universe (current):** ETH (active live, +BNB code-wired since 2026-05-02 commit `51c6f11` but PySR/HRST not yet run), BTC/SOL/LINK/XRP all `enabled: false` (HRST done, results not promoted). Asset universe was pruned 2026-04-19 — DOGE/ADA/AVAX/DOT removed from config + ASSETS dicts + production CSV. ETH block in `regime_config_ed.json` is clean (no top-level legacy keys); SOL/LINK/XRP still carry legacy `strategy`/`horizon`/`min_confidence`/`max_position_usd` at top level — harmless since `enabled: false`.
13. **Per-regime hold-shield (2026-04-18):** `hold_shield` now lives inside `bull` and `bear` blocks, not at asset level. Trader reads via `_shield_on_for_regime(trading_cfg, regime_label)` which falls back to asset-level `hold_shield` for legacy config. Shared `min_sell_pnl_pct` + `max_hold_hours` stay at asset level — per-regime split is only ON/OFF.
14. **Regenerating PySR requires retraining production models.** Mode P rewrites `models/pysr_*.json` but production CSV stores feature *names* only — inference re-evaluates whichever formula is in the JSON *now*. If you run P, follow with DV (or HRS) for the same horizon, else silent feature drift at inference. Leakage check (`_check_pysr_leakage`) catches leakage but NOT train/inference mismatch.
15. **Mode T chains rally-cooldown (2026-04-18):** After writing shield + thresholds, T merges its fresh bull/bear signals via `_merge_tagged_signals()` and calls the shared `_sweep_rally_cooldown()` helper. No stale cache issue, no need to run G separately. Mode G standalone kept as cache-fed fallback (fast iteration when models haven't changed).
16. **Two-machine setup → don't declare a job dead from local process list alone.** The engine folder is shared via Google Drive. Any `python.exe` running on the OTHER machine (desktop ↔ laptop) is invisible to `tasklist` / `Get-Process` on this one, but the files it writes show up here within seconds. **Liveness must be judged by file mtimes, not local processes.** Order of evidence:
    - `logs/ed_v1_<latest>.log` mtime — updates every few seconds during all HRST phases. **Primary signal.**
    - `models/crypto_ed_production_noprod.csv` + `config/regime_config_ed_noprod.json` mtimes — updated each Mode T iteration (~10-20 min).
    - `models/crypto_ed_grid_ETH_<h>h.csv` mtimes — written once per horizon during Mode D.
    - `output/ab_matrix_results_<tag>.csv` — appended after each variant completes.

    **Decision rule:** if newest matrix-related mtime is **within last 2 min → alive on some machine**. **>10 min stale + no local python → likely dead, but ASK the user before declaring it.** Between 2-10 min: ambiguous (Mode T on a heavy iter step) — wait, do not relaunch.

    **Mode R exception (silent for up to ~2h).** Mode R runs all regime detectors in-memory across the replay window and only flushes the winner once at the end. During Mode R, `noprod.csv` + `regime_config_ed_noprod.json` + `ed_v1_*.log` can all sit unchanged for 30-120 min on a long replay. **If the previous mtime burst was Mode D/V grid writes (`models/crypto_ed_grid_ETH_<h>h.csv`) all completed and Mode T hasn't started writing yet, you're probably in Mode R — don't declare dead.** Confirm by asking the user before relaunching.

    **Always ask "is it running on the desktop?" before saying it's stopped.** This applies to: AB matrix runs, HRST/Mode D/V/H/S/T/R, Mode P (PySR), backtests, any long-running job. Same applies in reverse — laptop jobs are invisible from the desktop. Past mistake: declared variant A dead because no python.exe on laptop, when it was running on desktop the whole time.

17. **When comparing returns across different replay windows, CONSIDER the window length.** Mode T REF / Mode S WINNER returns scale roughly with replay length — a 4mo (`--replay 2880`) run will tend to have roughly 2× the absolute return of a 2mo (`--replay 1440`) run on the same strategy because it has 2× the bars and 2× the trade opportunities. Always check `Replay: NNNNh` in the log's MODE R / MODE S header before drawing conclusions. Three ways to make a fair comparison:
    - **(BEST) Take the last 2mo (H2) of the 4mo run** — Mode T/G logs already print `H1=...% H2=...% REF=...%` where H1 = first half, H2 = second (most recent) half, REF = full window. Compare H2 of a 4mo run to the FULL T total of a 2mo run from a similar end date — that's apples-to-apples since both cover the same recent ~1440h period. Same trick works for top STRICT rally-cooldown candidates (the table prints `pnl_H1, pnl_H2, pnl_REF` per config). This is usually the cheapest, most accurate comparison and avoids re-running anything.
    - **Annualize** for a single common metric: `annualized_pct = return_pct × (365×24 / replay_h)`. e.g. `+148% on 2880h ≈ +452%/yr`; `+68% on 1440h ≈ +416%/yr`. Useful for quick gut-check but assumes return is roughly linear in time — only valid if the regime didn't shift mid-window.
    - **Re-run on a common window** — slow (3-9h on parallel engine) but definitive. Use only when (a) and (b) disagree or when the question is critical enough to warrant the wait.
    Past mistake (2026-05-02): I claimed Apr 30 `vol_calm +148.06%` was "the biggest result of the week" without checking that it ran on `--replay 2880` (4mo) vs the other ETH HRSTs on `--replay 1440` (2mo). The Apr 30 log prints `H1=+34.71% H2=+65.37% REF=+125.87%` baseline and the top STRICT gate winner had `pnl_H1=+42.62 pnl_H2=+65.34 pnl_REF=+144.60`. So vol_calm's H2 (= last 2mo, comparable to the 2mo runs) was ~+65% baseline with ~0pp gate gain — actually one of the WORSE 2mo results that week, not the best. The full-REF +20.5pp gate gain was driven entirely by months 1-2 of the window which the other runs never saw.

---

## Pending Work

### TODO

---

## ⚡ ACTIVE TODO — 2026-05-02 evening (CURRENT)

This is the freshest snapshot. All sections below this block (`---`) are preserved as historical audit trail of tested/shelved decisions — re-read them when reviving a shelved item or when you need to remember why something was rejected.

### 🔴 #1 PRIORITY (TOMORROW, 2026-05-03) — LIVE-TEST the M-29 partial-fill fix

**Status**: code shipped 2026-05-02 evening (commit `d568a30`, pushed to `origin/main`). Trader picks it up automatically on next restart via `start_ed_v2.bat`. **Cannot be unit-tested any further — needs a real `/buy` event against Revolut's actual API to validate end-to-end.**

**Test protocol for tomorrow's first BUY (manual `/buy` OR auto regime BUY — both exercise the same code path)**:

1. **Verify trader is on the new code**: in the engine dir, `git log -1 --oneline` should show `d568a30 M-29 fix...`. If trader was running before the restart, confirm restart picked up the fix.
2. **Let a BUY happen during normal use** — don't manufacture a test trade. The fix only fires on partial-filled BUYs, so it might take a few entries before it's exercised.
3. **Watch the trader log for new diagnostic lines**:
   - `[M-29 recalc] target=$X filled=$Y remaining=$Z cash=$C next_size=$N` — fires after every partial-fill cancel cycle
   - `[M-29 cross-check] wallet says spent $A, orders say $B (delta $C). Using max → smaller remaining.` — fires only if the two sources disagree by >$1 (interesting telemetry, harmless)
   - `Target reached: spent $X of $T — remaining $Y below $300 min trade. Stopping.` — clean stop without market-fallback (replaces the case where old code would have over-spent the residual)
4. **After the trade completes, verify in [config/position_ed_v2_ETH.json](config/position_ed_v2_ETH.json)**:
   - `usd_invested` should be ≤ target (within ~$5 tolerance for rounding)
   - **Compare to today's bug**: today was $12,609.53 = $609.53 over. Tomorrow's value should be $11,995-$12,005 range for `/buy 12000`.
5. **If it overshoots again**: capture the log + position file, then revert immediately:
   ```bash
   git revert d568a30 && git push
   ```
   Restart trader. Back to old behavior (with the bug, but known-tolerable).
6. **If it works correctly across 3-5 BUYs (at least 1 with a partial fill)**: this entry moves to Closed and the bug is officially fixed.

**Expected first-BUY behavior** when wallet ≥ target, no partials encountered: trader places maker, fills fully, no `[M-29 recalc]` log line ever fires (recalc only fires on cancel-after-partial). Position file shows `usd_invested ≈ target`. This is the hot path — fix should be invisible.

**Expected partial-fill behavior** (the one we care about): trader places maker, partial fill, cancels, `[M-29 recalc]` fires, places SMALLER next leg (capped at `remaining_target`), eventually fills target. Position file shows `usd_invested ≤ target`. **NEVER over-target.**

---

### Background: the bug + the fix shipped

Found 2026-05-02 from today's `/buy` event ([logs/ed_v2_20260429_230924.log](logs/ed_v2_20260429_230924.log) lines 9012-9027). **Real overspend: $609.53** on a $12,000 target → recorded position `usd_invested = $12,609.53` ([config/position_ed_v2_ETH.json](config/position_ed_v2_ETH.json) latest BUY entry timestamped `2026-05-02T14:03:57Z`).

**What actually happened (per position file + log)**:
1. Wallet cash before /buy: ~$12,609.55 (MORE than the $12,000 target — this is the catastrophic-scenario condition)
2. User issued `/buy` → default size = `max_position_usd = $12,000.00` (logged as `$11,999.99` after the $0.01 safety-margin floor)
3. Phase 1 partial fill: $2,026.12 actually filled (log displayed "Partially filled: 11%" but that was rounding — true partial was 16.9%)
4. After partial: wallet cash dropped to $10,583.43 (= $12,609.55 − $2,026.12)
5. **Buggy recalc** at [crypto_revolut_ed_v2.py:826-828](crypto_revolut_ed_v2.py#L826): `if usd_avail < size: size = math.floor(usd_avail * 100) / 100 - 0.01` → `size = $10,583.41` (full remaining cash) instead of `target − already_filled = $12,000 − $2,026.12 = $9,973.88`
6. Phase 2 fill: $10,583.41 at $2,308.06 = 4.586 ETH
7. **Total spent: $2,026.12 + $10,583.41 = $12,609.53** ($609.53 over target)

**Bug location**: [crypto_revolut_ed_v2.py:818-828](crypto_revolut_ed_v2.py#L818) inside `_execute_maker_order()`:
```python
if usd_avail < size:
    print(f"    Balance updated after partial fill: ${size:,.2f} → ${usd_avail:,.2f}")
    size = math.floor(usd_avail * 100) / 100 - 0.01
```
The print statement writes `(target_size → wallet_avail)` — the log line `$11,999.99 → $10,583.42` is NOT before/after wallet, it's `target_var → avail_var` from the same moment. Subtle source of confusion when reading logs.

**Why it matters more broadly**: the same code path applies to every BUY (auto + manual). Today's overspend was bounded because wallet only had ~$12,610. With a larger wallet (e.g., $20k cash, /buy $12k target):
- Phase 1 partial fill ~17% = $2,026
- After: `usd_avail = $17,974`, `size = $12,000`. Condition `$17,974 < $12,000` is False → size stays $12k
- Phase 2 fills $12,000 (the original target — second time!) → **total $14,026, $2,026 over target**
- If THAT phase 2 also partial-fills, the bug compounds

For SELL side ([crypto_revolut_ed_v2.py:830-836](crypto_revolut_ed_v2.py#L830)): same logic, same bug, but less catastrophic because we're selling everything we own (the original `size = base_amount` already equals total holdings, and `crypto_avail < size` only fires after partial fill removes some — then we sell what's left, which IS what we want).

**Fix shipped (`d568a30`, 2026-05-02 evening)**:
1. Track `total_filled_usd` (BUY) / `total_filled_qty` (SELL) by reading `filled_quantity × average_fill_price` from each cancelled order's status (Source A — immune to other-asset USD activity contaminating wallet)
2. Cross-check against wallet-delta from `baseline_avail` captured at function entry (Source B — immune to order-status read failures)
3. Use `max(spent_by_wallet, spent_by_orders)` → smaller `remaining_target` → never overspend
4. `next_size = min(usd_avail - 0.01, original_size - total_filled_so_far)`
5. Two early-stop branches: cash-below-min (market-fallback for residual or stop) and target-met-within-tolerance (stop, don't place sub-min orders)
6. Symmetric SELL-side patch (currently benign because trader always sells 100% holdings, but partial-sell scenarios would have hit same bug)

**Verified by 9 unit tests** in [tools/test_m29_partial_fill_bug.py](tools/test_m29_partial_fill_bug.py): today's exact bug ($12,610 wallet, $12k target, $2,026 partial → next leg correctly capped at $9,973.88 instead of buggy $10,584.42), $50k wallet catastrophic scenario, wallet < target, cash near minimum, target essentially met, baseline read failed, cross-check disagreement, first-iter no fill.

**Caller-compat verified**: all 4 callers (auto BUY/SELL + manual `/buy`/`/sell`) use balance-delta as primary basis source via M-02/M-03 ledger-delta logic. New early-stop return dict `{'status': 'filled_target_reached', 'spent_usd': X}` gracefully falls back to balance-delta in callers — `filled_quantity`/`average_fill_price` missing fields don't cause issues.

**Audit completed**: 12 risk dimensions checked (return-dict compat, baseline race, double-counting, partial→full path, MIN_TRADE_USD edge cases, SELL-side asymmetry, M-02/M-03 interaction, retry paths, syntax/compile, etc.). No regressions identified. See conversation log 2026-05-02 evening for full audit.

**Priority severity**: HIGH — silent money-correctness bug, same class as M-02/M-03 ledger-delta bugs from 2026-04-25 bundle. Live test required because Revolut's actual API behavior on partial fills can't be mocked perfectly.

---

### Currently running
- **HRST ETH 5,6,7,8h --replay 1440 --no-persist** — started 2026-05-02 18:00 ([logs/ed_v1_20260502_180022.log](logs/ed_v1_20260502_180022.log)). Validates today's parallel-wrapper bug fixes (A: `UnboundLocalError` at `_generate_signals_cached`; B: `BrokenProcessPool` in hybrid refine config #3). Compare winner to 8h HRST result (`bull=7h@85% / bear=16h@75% +68.81%`).
- **HRST BNB 4,5,6,7,8h --replay 1440 --no-persist** — started 2026-05-02 20:28 ([logs/ed_v1_20260502_202819.log](logs/ed_v1_20260502_202819.log)). First full pipeline for BNB. Mode P was completed earlier today (~1.5h, all 5 PySR JSONs written cleanly with `discovery_method = "historical"`). HRST will produce single-horizon Mode V winners, Mode S joint regime sweep, and Mode T shield+gate sweep. ETA ~3-4h on laptop. Output: `models/crypto_ed_production_noprod.csv` BNB rows + `config/regime_config_ed_noprod.json` BNB block.

### Next big work — IN ORDER

**1. Test BNB**

BNB code-wired today (commit `51c6f11`); derivatives data downloaded; status:
- ✅ Hourly OHLCV download — `data/bnb_hourly_data.csv` (73,672 candles, 2017-12-01 → 2026-05-02)
- ✅ Mode P PySR — DONE 2026-05-02 20:24, all 5 JSONs written, anti-leakage check passed, ~1.5h on 3-worker laptop (validated parallel-P path of merged engine)
- ⏳ **Full HRST 4,5,6,7,8h --replay 1440 --no-persist — RUNNING NOW** (started 20:28). See "Currently running" above.
- 🔜 Pipeline-health check (per SOL/LINK pattern): horizons positive, alpha vs ETH live, correlation with ETH (target ≤ 0.70 for diversification)
- 🔜 Decision: BNB clears the same bar as ETH (≥50% per-$ alpha + low correlation + plateau-stable) → flip `enabled: true` + `max_position_usd > 0`. Otherwise shelve.

**2. Build 15-minute candle system (Ed15)**

Old Ein/Eli scripts archived (`archive/crypto_trading_system_ein.py`, `archive/crypto_trading_system_ed15.py`) — both stale, missing parallel wrapper + post-2026-04-25 audit fixes. Right approach: clone current production engine.
- `cp crypto_trading_system_ed.py crypto_trading_system_ed15.py`
- Adapt constants — horizons in candles (4-10 candles = 1h-2h30), grid windows in candles (12-120), `MAX_DIAG_HOURS` interpretation
- Separate file paths — `models/crypto_ed15_production.csv`, `models/crypto_ed15_best_models.csv`, `config/regime_config_ed15.json`, `models/pysr_{ASSET}_{N}p.json` (p = periods/candles)
- Reuse macro pipeline — `download_macro_data.py` unchanged; 15-min OHLCV (`data/{asset}_15m_data.csv`) already exists for BTC/ETH; need full download for BNB
- Clone noprod wrapper — `crypto_trading_system_ed15_noprod.py` mirroring current pattern
- Optional separate trader — `crypto_revolut_ed15_v2.py` if running hourly + 15-min concurrently. Otherwise extend trader to dispatch by config block
- `config/disabled_features_ed15.json` already exists from old Ein attempts — re-check Grade-1 list (may differ at 15-min resolution)
- Validation order — run BNB on Ed15 first as new-asset case, then ETH/BTC if it works

### Week-of audit (2026-04-25 → 2026-05-02) — comparable cross-window summary

All ETH HRSTs from the past 7 days, normalized to "last 2 months" via Mode T's H1/H2 split (per Critical Rule 17). The 4mo run is included via its H2 (last 2mo) baseline + top STRICT gate winner H2.

| Date | Window | Detector | Bull | Bear | Last-2mo baseline | Last-2mo T total | Gate Δ on last 2mo | Conv |
|---|---|---|---|---|---|---|---|---|
| Apr 25 03:56 | 2mo | sma24>sma100 | 6h@75% | 5h@75% | +66.92% | +68.02% | +1.1pp | iter2 |
| Apr 25 07:57 | 2mo | sma168>sma480 | 7h@85% | 5h@75% | +88.55% | (no shield) | 0pp | iter3 |
| **Apr 25 23:12 (a)** | **2mo** | **sma24>sma100** | **6h@80%** | **5h@65%** | **+68.72%** | **+76.99%** | **+8.3pp** ← best | **iter2** |
| Apr 25 23:12 (b) | 2mo | sma24>sma100 | 6h@80% | 5h@65% | +58.57% | +63.42% | +4.9pp | iter3 |
| Apr 26 14:11 | 2mo | price>sma72 | 6h@80% | 7h@85% | +65.75% | +72.09% | +6.3pp | iter2 |
| Apr 28 08:08 | 2mo | tsmom_672h | 5h@65% | 6h@65% | +79.47% | +79.47% | 0pp | iter2 |
| **Apr 30 04:42 (H2)** | **last 2mo of 4mo** | **vol_calm** | **6h@70%** | **5h@85%** | **+65.37%** | **~+65.34%** | **~0pp** | **iter3** |
| May 01 09:21 | 2mo (8h HRST) | tsmom_672h | 7h@85% | 16h@75% | +68.81% | +68.81% | 0pp | iter2 |
| May 02 10:02 | 2mo (XRP) | tsmom_672h | 4h@80% | 7h@65% | +28.07% | +40.84% | +12.8pp | iter3 |
| May 02 13:53-15:48 | — | — (BTC, crashed) | — | — | — | — | — | parallel-engine bug |
| May 02 18:00 | 2mo | RUNNING | — | — | — | — | — | post-bug-fix |
| May 02 20:28 | 2mo | RUNNING (BNB) | — | — | — | — | — | — |

**Real biggest gate-gain of the week**: `Apr 25 23:12 sma24>sma100 6h@80%/5h@65% +8.3pp` (and Apr 26 `price>sma72 6h/7h +6.3pp` close behind). Apr 30 `vol_calm` looked huge at +20.5pp on the full 4mo REF, but its H2 (last 2mo) gate gain was ~0pp — the +20pp came entirely from months 1-2 of the 4mo window. Detector winners across the week: sma24>sma100 ×4, tsmom_672h ×3, sma168>sma480 ×1, price>sma72 ×1, vol_calm ×1.

**Current live config (`tsmom_672h bull=6h@85% / bear=5h@65%`) doesn't appear in any 7-day log.** Closest is Apr 28 (`tsmom_672h 5h/6h` — bull/bear horizons SWAPPED + different confs). Either set by an older HRST (logs deleted in today's cleanup) or manual edit/merge. Worth a `git log -p config/regime_config_ed.json` trace.

### User decisions pending
- **Live config bull horizon: 6h → 7h?** — 8h HRST May 01 plateau-unanimous on bull=7h. Compare its last-2mo gate gain (~0pp) against alternatives before flipping. Marginal evidence in absolute terms.
- **Live config bear horizon: 5h → 16h or 12h?** — Material change (8h HRST plateau picked 16h ×7 / 12h ×5 / 8h ×2 over 5h). 0pp shield/gate gain on last 2mo. Plateau-stable but doesn't dominate. Validation needed before flipping.
- **Investigate Apr 25 23:12 `sma24>sma100 6h@80%/5h@65% +8.3pp` finding** — actual best gate-gain of the week. Reproduce on current data window if not too stale; consider promoting if alpha holds. ~3-4h.

### Standing / monitoring (passive)
- **MIX gate live perf** — 5+ days in (since 2026-04-27 ~20:30 CEST). Rollback criteria: realized alpha drops >15pp vs sim baseline over first 10 trades; max DD exceeds −10%; signals consistently blocked when forward 24h is positive. One-line rollback: `copy config\regime_config_ed_pre_hrst2_20260427_evening.json config\regime_config_ed.json`
- **`output/ERRORS_INBOX.md`** — currently empty; check on every TODO review

### Scheduled (calendar-driven)
- **2026-05-22** — `deriv_oi_*` re-enable A/B test (3 features, ~63d Binance OI history by then). Procedure G1 in research-queue history below.
- **2026-06-18** — Orderbook + IV re-enable A/B test (`ob_imbalance, spread_bps, avg_iv, iv_skew`, ~60d data). Procedure G2 below.

### Research queue — actually untested, lower priority than BNB + Ed15
1. Vol-scaled horizon 4mo validation — 2-month tested 2026-04-19 (`vol_2band low→8h high→6h @90% = +33.82% / +5.02pp over tsmom`). Only 4mo missing. `tools/test_vol_scaled_horizon.py --replay 2880`. ~30 min.
2. Multi-horizon ensemble emergency-exit (4th angle from 2026-04-27 forensic) — force exit when 5h AND 8h both flip SELL within 1h. Distinct from T1b entry-side test which was already shelved. Per-horizon cache exists. ~1-2h.
3. Per-regime feature set — bull more technical / bear more macro. Untested deliberately. Low-medium.
4. Execution-gap TCA logging — biggest live-perf lever (~17pp unaccounted). Trader code change. 2-4h.
5. Trace current live config origin — `git log -p config/regime_config_ed.json` to find when `tsmom_672h 6h@85%/5h@65%` was actually set. ~10 min.
6. Verify BTC HRST works after parallel bug fix — `HRST BTC 4,5h --replay 1440 --no-persist` after laptop runs finish. Bug-fix verification only (BTC stays disabled). ~1h.

**Closed 2026-05-02:**
- ~~Re-run Mode D ETH 6h, 8h with derivative features (clean)~~ — derivatives are now standard in the clean pipeline (funding rate + perp-spot basis active and selected by LGBM; deriv_oi_* still quarantined per G1 schedule until 2026-05-22). Today's running ETH HRST 5,6,7,8h is the clean re-run this TODO was asking for; no separate work needed.
- ~~Triple barrier as TRAINING LABEL~~ — re-discovered tested + shelved 2026-03-14 in `archive/testing_literature.csv`. BTC same-week head-to-head: baseline 4h +57.22% / 8h +74.03% vs triple_barrier_label 4h +29.22% / 8h +22.53%. Lost on return (−28pp / −51pp), WR (−9pp / −10pp), accuracy (−16pp / −13pp), and combined score on every dimension. The "+29% standalone result" framing in earlier CLAUDE.md was misleading — same-day baseline beat it by ~30-50pp. SHELVED.
- ~~Pairwise long-horizon validation `R ETH 5,6,7,8,12,16`~~ — already done. The 8h HRST 2026-05-01 ran Mode R on `[4, 5, 6, 7, 8, 9, 12, 16]` (superset). Best regime: `tsmom_672h bull=7h bear=6h +43.49%`. Mode S joint sweep on the same 8 horizons picked WINNER `tsmom_672h bull=7h@85% / bear=16h@75% → Mode T REF +68.81%`. Re-running Mode R on 5,6,7,8,12,16 (subset) wouldn't add information. What's actually pending = USER DECISION whether to promote bear=16h to live based on existing evidence.
- ~~vol_calm Apr 30 promotion candidate~~ — initially looked like the biggest result of the week (+20.5pp gate gain on REF). Re-checked via Mode T's H1/H2 split: H2 (last 2mo of 4mo window) baseline +65.37% / T total ~+65.34% / **0pp gate gain on the comparable last-2mo slice**. The +20pp delta was driven entirely by months 1-2 of the 4mo window. NOT a promotion candidate. Methodology lesson now in Critical Rule 17.
- ~~Parallel-engine bugs A (`UnboundLocalError` at `_generate_signals_cached`) + B (`BrokenProcessPool` in hybrid refine)~~ — patched 2026-05-02 (per user). Validated by today's BNB Mode P (parallel-P all 5 horizons clean) + currently-running Mode V parallel cache + hybrid refine on laptop ETH/BNB HRSTs. Reverification of BTC HRST scheduled (#6 in research queue).
- ~~BNB Mode P 5,6,7,8h~~ — DONE 2026-05-02 20:24, all 5 PySR JSONs written cleanly, all `discovery_method = "historical"` (anti-leakage check passed). Parallel-P speedup observed (~1.5h for 5 horizons on 3-worker laptop, vs ~5h sequential reference).
- ~~Mode P auto-OHLCV-download patch~~ — landed 2026-05-02 (8 lines added at [crypto_trading_system_ed.py:3828](crypto_trading_system_ed.py#L3828)). Mode P now auto-fires `update_all_data([asset])` if hourly CSV is missing — prevents the "no data found" failure mode for fresh assets. Verified on BNB.

### Engineering / UX
- Telegram `/help` lists 8 of 13 commands (missing /buy, /sell, /hold, /cfg_*). [crypto_revolut_ed_v2.py:_handle_help_command](crypto_revolut_ed_v2.py) ~line 2392 + BotFather /setcommands. ~15 min.
- Document SOL/BNB on-chain CoinMetrics 403 as permanent free-tier limitation (visible in every macro download). Update `download_macro_data.py` to log "SKIPPED (free-tier 403)" instead of "ERROR" so it stops flagging as a failure. ~10 min.

**Closed (engineering):**
- ~~Engine Reference Card update for 4h / long horizons / 7h canonical bull~~ — done 2026-05-02 in Tier 1 edits. §"Horizon status" rewritten with 8h HRST evidence; §"What doesn't work" 4h entry struck-through with REVISED note; §"What's untested" purged of already-tested entries.

### Today's shelved (added 2026-05-02)
- **XRP enable** — user judged HRST results bad despite +25.69pp alpha. HRST winner `bull=4h@80% / bear=7h@65%` written to live config but `enabled: false` stays.
- **BTC re-enable** — results not good enough; no urgency. The BTC HRST work (after parallel bug fix verification) is purely engineering, not a promotion candidate.

---

## 📜 HISTORICAL TODO (preserved as audit trail of tested/shelved decisions)

The entries below are kept verbatim so future-you can revive any shelved item with full context — what was tried, what the result was, why it was rejected. Do not delete; mark inline as RESOLVED if you re-evaluate.

---


## 📦 MERGED TOPIC: 8-horizon HRST (2026-04-30 launch → 2026-05-02 resolution)

*This topic groups several banners that were previously written separately as the work progressed. Content kept verbatim, just reordered chronologically and merged under one header.*

---

**🌙 OVERNIGHT 2026-05-01 → CHECK 2026-05-02: 8-HORIZON HRST RESULTS.**

Launched 2026-05-01 ~00:22 CEST on laptop:
```
python -u crypto_trading_system_ed.py HRST ETH 4,5,6,7,8,9,12,16h --replay 1440 --no-persist
```

ETA ~9-10h on laptop. Backups before launch:
- `config/regime_config_ed_noprod_pre_8h_20260501.json`
- `models/crypto_ed_production_noprod_pre_8h_20260501.csv`

**Context:** 2026-04-30 evening Mode D screen on horizons 9-18h (1440 replay) showed:
- **Killed:** 10h, 11h, 13h, 14h, 17h (return < +12%, APF weak, or accuracy < 60%)
- **Marginal/test:** 9h (+21.72%), 12h (+23.89% / 100% WR), 16h (+21.31% / 80% WR), 18h (LOW_TR)
- **Caveat:** trade counts collapse to 3-5 per config at horizons ≥ 14h on 1440h replay — Mode D for long horizons is borderline noise.

The 8-horizon HRST tests whether ANY long horizon (4h, 9h, 12h, 16h) survives full Mode H + V + S + T pipeline. Includes 4h to retest CLAUDE.md's "4h structurally broken" verdict — recent Mode D showed 4h XGB+LGBM apf=9.31 ret=+11.1%, suggesting the post-engine-fix data may have changed the picture.

**Decision matrix when results land:**
- Mode S TOP 15 stays in {5,6,7,8} → confirm canonical sweep, kill 4/9/12/16 from default permanently
- 4h appears in TOP 15 → CLAUDE.md "4h broken" verdict is data-snapshot-specific, revise Engine Reference Card
- 9h / 12h / 16h appears in TOP 15 → long horizons have signal in current regime, expand sweep
- Mode T REF > current LIVE +70.31% AND family-stable plateau → consider promoting
- Mode T REF < LIVE OR no plateau → keep LIVE 2-det config (`tsmom_672h 6h@85%/5h@65%`), document long horizons as dead

**Liveness signal (per CLAUDE.md rule 16):** check `logs/ed_v1_*.log` mtime; updates every few seconds during all phases. Mode S phase prints `2 detectors × 64 h-pairs × 7×7 conf = 6,272 combos` (vs today's 1,568 with 4 horizons) — confirms parser accepted all 8 horizons.

---


---

**✅ RESOLVED 2026-05-02:** 8-horizon HRST completed (~2026-05-01 09:15 CEST, total ~9h on laptop). Single-horizon Mode V winners ALL positive (4h +41.23%, 5h +34.88%, 6h +44.07%, 7h +64.90%, 8h +27.92%, 9h +31.36%, 12h +33.27%, 16h +30.79%). Mode S TOP 15 plateau: ALL bull=7h; bear migrated to 16h (×7), 12h (×5), 8h (×2). Mode S WINNER: `tsmom_672h bull=7h@85% / bear=16h@75% → +64.16%`. Mode T converged iter 2 → +68.81% / 24 trades / WR 92% (gates active, both shields OFF). Comparable to claimed live +70.31%. **Verdict**: long horizons (12h, 16h) are NOT dead in current regime — they dominate the bear plateau. Original CLAUDE.md "kill 4/9/12/16 from default permanently" decision rule was overcautious. New canonical sweep should consider extending bear range. Validation pending via current 5,6,7,8h HRST + pairwise `R ETH 5,6,7,8,12,16` follow-up.


## 📦 MERGED TOPIC: Parallel V/S/T + Mode P infrastructure (2026-04-29 launched on Desktop)

*This topic groups several banners that were previously written separately as the work progressed. Content kept verbatim, just reordered chronologically and merged under one header.*

---

**🧪 PARALLEL WRAPPER (in test) — 2026-04-29 launched on Desktop after laptop validation.**

`crypto_trading_system_ed_parallel.py` is an experimental wrapper that monkey-patches the engine in-process to parallelize:
- `eng.run_mode_v` ← Mode V Step 1 (6 D-candidate backtests) + Step 3 (3 refined backtests) via loky workers
- `eng.run_mode_s` ← per-horizon `generate_signals` cache build via loky, then delegate to engine sweep with cache-injected `generate_signals`
- `eng.run_mode_t` ← same per-horizon cache build, then delegate to engine T↔G iterative loop
- `eng._refine_top_configs` ← hybrid GPU+CPU refine (1 GPU process + 1 CPU process concurrent, 3rd config dynamic to first-freed device)

**Production engine `crypto_trading_system_ed.py` is NOT modified by the wrapper.** Patches fire only when the wrapper is invoked as `__main__`. Engine originals captured at wrapper-import time via `_ENG_RUN_MODE_S/T/V_ORIG` and `_ENG_REFINE_TOP_CONFIGS_ORIG` to prevent recursion through the monkey-patches.

**Validation status (as of 2026-04-29 17:30):**
- ✅ Smoke test passed on laptop (`HRST ETH 5,6h --replay 336 --no-persist --no-data-update`, 2h 06min): all 4 patches confirmed firing exactly once each, no recursion
- ✅ V/S/T parallel paths exercised cleanly (Step 1 ~3 min × 2 horizons, Step 3 ~2.5 min × 2 horizons, Mode S/T cache 2.4 min each)
- ✅ Hybrid refine focused test passed (3 synthetic configs, n_trials=5 override, ProcessPoolExecutor + GPU/CPU dispatch + dynamic 3rd config + APF-descending sort + result-list aggregation)
- 🔄 **Currently running**: full HRST 2880 ETH 5,6,7,8h on Desktop, started 2026-04-29 17:33. Projected ~9-9.5h vs sequential ~21h baseline.

**Test command** (for reproducing or relaunching):
```
python crypto_trading_system_ed_parallel.py HRST ETH 5,6,7,8h --replay 2880 --no-persist
```

Optional test-only flag for fast hybrid-refine validation: `--refine-trials 5` (default = `eng.REFINE_TRIALS = 50`).

**Per-machine policy (in wrapper, not in `hardware_config.py`):**
- All machines: `PARALLEL_BACKTESTS = 6` workers, `PARALLEL_LGBM_DEVICE = 'cpu'` inside parallel sections
- Sequential code paths still use the engine's auto-detected `LGBM_DEVICE` (gpu on Desktop/Laptop, cpu on Yoga)
- LGBM on CPU inside parallel section avoids GPU queue serialization (6 concurrent GPU LGBMs would queue and lose most of the speedup)

**What is NOT optimized (left sequential):**
- Mode H per-horizon outer loop — would oversubscribe CPU since Mode D internally uses `N_JOBS_PARALLEL=14` workers
- Optuna refine `n_jobs > 1` inside a single study — TPE sampler quality drops with parallel trials; the hybrid GPU+CPU split sidesteps this by running 3 separate studies in parallel processes instead

**Promotion criteria (test → production):**
1. Current 2880 finishes cleanly with `Done!` marker — proves all 4 patches survive a real 4mo workload at full trial count (50 trials × 3 configs × 4 horizons)
2. Wall-clock ≤ 10h (≥50% reduction vs sequential 21h baseline)
3. Mode S/T parallel cache fires exactly once per call (no recursion regression)
4. No `Traceback` / `OSError` / `TerminatedWorkerError` in the run log

If all 4 pass: promote the wrapper changes into production engine in a follow-up PR. Specifically:
- Move `PARALLEL_BACKTESTS` and `PARALLEL_LGBM_DEVICE` constants into `hardware_config.py`
- Inline `_run_parallel_backtests` + `_backtest_one_config_worker` into `crypto_trading_system_ed.py` Mode V Step 1 + Step 3
- Inline `_predict_signal_calls_for_horizons` + `_build_signals_cache_parallel` + `_generate_signals_cached` + replace Mode S/T per-horizon loops with parallel cache build
- Inline `_refine_top_configs_hybrid` into engine's `_refine_top_configs`
- Drop the wrapper file once production has the parallel paths

**Liveness monitoring** (per machine-setup rule above): track `logs/ed_v1_<latest>.log` mtime (writes every few seconds during all HRST phases). Mode V parallel sections show `[parallel] dispatching N backtests across N workers` and `[parallel] N backtests completed in N.N min` markers; Mode S/T cache shows `MODE S/T (PARALLEL signal cache)` then `[parallel] N signals generated in N.N min`; hybrid refine shows `[refine-hybrid] dispatching 3 configs across 2 workers` + `[refine-hybrid] {DEV} freed → starting config #3`.

---


---

**🧪 PARALLEL MODE P (queued for testing) — `crypto_trading_system_ed_parallel_p.py`.**

Separate experimental wrapper for Mode P (PySR feature discovery). Built 2026-04-29 17:55 alongside the V/S/T/refine parallel work. Mode P is the slowest research mode in the engine — each (asset, horizon) runs 4-7 sequential PySR studies, each with PySR's own internal parallelism explicitly disabled (`procs=0`, `parallelism="serial"`) because `deterministic=True` requires it.

**Approach (Option B — keeps determinism):**
- Each individual PySR run keeps `deterministic=True` + fixed per-run seed (42, 59, 76, 93, ...) — bit-for-bit reproducible across re-runs of the same settings on the same machine
- The N outer PySR runs (different feature subsets, different seeds) are dispatched across multiple processes via `ProcessPoolExecutor`
- **Per-machine policy:** Desktop=4 workers, Laptop=3 workers, Yoga=2 workers (each PySR Julia process ≈ 2-3 GB RAM)
- Expected speedup: **~2× per (asset, horizon)** for a 4-run setup (3 concurrent on laptop → 4 runs finish in ~ceil(4/3)=2× one-run time)

**Status:** built + wire-up checks PASSED, NOT yet run on real PySR.

**Test command — must run on a DIFFERENT asset than what desktop's ETH 2880 is using, since Mode P writes `models/pysr_<ASSET>_<H>h.json` files that the live HRST will read mid-run:**
```
python crypto_trading_system_ed_parallel_p.py P BTC 5,6,7,8h
```

Or smaller smoke test:
```
python crypto_trading_system_ed_parallel_p.py P BTC 5h
```

**Why BTC:** while ETH 2880 is running on desktop, ETH 8h Mode V (last horizon) will eventually read `models/pysr_ETH_8h.json`. If the laptop's Mode P writes a fresh ETH 8h PySR JSON between desktop's 7h and 8h Mode V starts, desktop will load the new file → inconsistent run (5h/6h/7h used old PySR features, 8h uses new ones). Running Mode P on BTC writes `pysr_BTC_*.json` — desktop never touches those, no contention.

**Validation criteria (test → would-promote):**
1. All N PySR studies dispatched complete cleanly (`[pysr-parallel] {LABEL} → N expressions` per dispatched run)
2. No `Traceback` / Julia worker death / serialization error
3. Wall-clock per (asset, horizon) ≤ 60% of sequential reference (i.e. ≥ 1.7× speedup on 3-worker laptop)
4. Output `pysr_BTC_*.json` files have `discovery_method = "historical"` (passes leakage check)
5. Same-seed re-run reproduces same expressions (determinism preserved)

If all pass: pattern is identical to the V/S/T/refine wrapper — eventually inline `discover_features_parallel` into `pysr_discover_features.py` directly.

**Liveness markers** to grep for during run: `[pysr-parallel] dispatching N PySR runs across M workers`, `[pysr-parallel] {LABEL} → N expressions`, `ALL RUNS COMPLETE: N candidate expressions in N.N min (parallel)`.

---


## 📦 MERGED TOPIC: 5m emergency-exit overlay (2026-04-27 morning 35-variant sweep → followup forensic of today's crash)

*This topic groups several banners that were previously written separately as the work progressed. Content kept verbatim, just reordered chronologically and merged under one header.*

---

**🚨 HIGH PRIORITY — 2026-04-27 morning EMERGENCY-EXIT 5m OVERLAY — TESTED + REJECTED.**

User asked for emergency exit triggered on 5-min price action to bypass shield during sharp crashes (motivated by 2026-04-27 03:45→05:45 UTC ETH crash $2400→$2319, -3.37% in 120min). Phase 1: identified 6 crashes ≥3%/120min in last 30d (incl. today's). Phase 2: built [tools/test_emergency_exit_5m.py](tools/test_emergency_exit_5m.py) overlay test — 35 variants of (threshold × lookback × cooldown × regime × bypass-mode × armed-at-profit).

**Verdict: NO variant beats baseline on any window.** The 60d baseline (+49.34% / 50tr / 70%WR) is already capturing crashes well via shield+max_hold+model SELL. Every emergency-exit variant either fires too often (false alarms eat alpha) or too rarely (no evidence of insurance value).

**Top results (sorted by 60d delta):**

| Rank | Config | d_30d | d_60d | d_90d | Fires 60d |
|---|---|---|---|---|---|
| 1 (cleanest) | **G. armed_at_pnl≥X% + thr=-2.0** (X∈{1..5}) | 0.00 | **0.00** | 0.00 | **0** |
| 2 | A. thr=-2.0% all bypass=always | -1.66 | -3.57 | -6.87 | 3 |
| 3 | G. thr=-1.5% armed_at_pnl≥X | 0.00 | -6.02 | -5.52 | 1 |
| **The one we expected to win:** A. thr=-1.0% | -4.71 | **-23.66** | -26.05 | 16 |
| Worst | A. thr=-0.7% | -8.51 | -38.14 | -41.05 | 43 |

**Why -1% loses so badly:** 16 fires in 60d, only 6 are real crashes; 10 false alarms at ordinary noise that mean-reverts within 30-60 min. Each false fire = ~0.5-1.5% lost alpha + double fees + 120min re-entry lockout that blocks the rebound BUY signal.

**What works as "free insurance":** `G.thr=-2.0% armed_at_pnl≥3%` — fired 0 times in 60d (no evidence either way). Theoretically correct design (only protect rally gains, not create losses) but unproven. Worth shipping with telemetry to collect real-world data over 1-3 months.

**Output CSV:** `output/emerg_exit_5m_20260427_091139.csv`. Crash list: `output/5m_crashes_20260427_090457.csv`. Indicator scores: `output/5m_indicators_20260427_090457.csv`.

**Crash list ≥3%/120min in last 30d (6 events incl. today):**
- 2026-03-29 21:35→22:45 (70min): $2007→$1939 (-3.41%)
- 2026-04-02 00:55→02:45 (110min): $2159→$2065 (-4.37%)
- 2026-04-08 13:05→14:55 (110min): $2271→$2187 (-3.70%)
- 2026-04-12 01:00→02:05 (65min): $2289→$2207 (-3.56%)
- 2026-04-14 14:30→15:05 (35min): $2416→$2333 (-3.43%)
- **2026-04-27 03:45→05:45 (120min): $2400→$2319 (-3.37%) ← TODAY**

**Confirms standing rule from CLAUDE.md:** *"All stop-loss / take-profit / profit-lock / trailing-stop variants — 8+ variants tested; baseline (no SL) won every dimension."* Same conclusion holds at 5-min granularity. Hourly shield + max_hold + model SELL already handles 90% of crashes; adding a faster trigger costs more in alpha than it saves in protection.

**Untested 4th angle:** emergency exit triggered by MULTI-HORIZON ENSEMBLE SELL agreement (using per-horizon signals from T1b) instead of raw price action. Would catch crashes when 2+ horizons unanimously flip SELL within 1 hour. Not yet tested.

---


---

**🚨 HIGH PRIORITY — 2026-04-27 forensic of today's crash + reverse-engineering attempt — CONFIRMS REJECT.**

After initial T5-T8 + emergency-exit sweep + H/I rally-conditioned variants, user asked to forensically reverse-engineer: given today's specific 03:45→05:45 UTC ETH crash ($2400→$2319), find ANY combination of derivatives (5/10/15min) + losses (5/10/15min) + prior rally that would have caught it cleanly. Built [tools/forensic_today_crash.py](tools/forensic_today_crash.py).

**Conclusion: today's crash was NOT detectable at 5-min granularity with acceptable FP rate.**

Bar-by-bar of today's crash:
- 03:45 (peak $2398) through 04:55: ALL indicators near zero. Slow drift mode. No signature.
- **05:00** ($2376, ret_5m=-0.29%, d2_5m=-0.26): FIRST momentum signature. Already -0.94% from peak.
- **05:15** ($2345, ret_15m=-1.29%, d2=-0.45): Clearest signal. Already -2.3% from peak.
- 05:45 (trough $2321): -3.4% from peak.

**Combo search results** (must fire by 05:30 UTC, scan 60d for false positives):
- 784 combinations fire today
- **Cleanest**: `r72h≥1.5 + ret_15m≤-0.8 + d2_5m≤-0.3` — fires today at 05:15 at $2345, but **42 false positives in 60d** (precision 12.5%)
- **Zero false-positive combos: NONE**
- **≤3 false positives: NONE**

**Why no clean combo exists**: ETH has been in slow uptrend for weeks. `rally_24h≥2%`, `rally_48h≥2%`, `rally_72h≥1.5%` are NOT rare events — they're the default state. Adding "preceded by rally" doesn't filter out false alarms because the rally context is permanent. The 5m drop signatures happen 70-100+ times in 60d; only 5-10 are real crashes. Math on best combo: save ~$120 once per 60d on real crash, lose ~$3000 to 42 false exits. Net catastrophic.

**Three converging lines of evidence say price-action emergency exit is a dead end for ETH in current regime:**
1. T5-T8 sweep (35 variants): best matches baseline at 0.00pp
2. H. rally-give-back winners (+1pp on 60d): caught blow-off tops, wouldn't have helped today
3. This forensic: no acceptable FP rate combo catches today

**Today's crash was news-driven** (Iran-related geopolitical context from yesterday/overnight) — by the time price cracked, the market had already digested the news. Price-derivative triggers can't anticipate this.

**Untested alternatives that might actually work:**
- **Multi-horizon SELL ensemble** as emergency trigger — fire if 3 of 4 model horizons (5h/6h/7h/8h, per T1b cache) all flip SELL within same hour. Catches model-recognized regime change, not lagging price action.
- **Manual override at high unrealized profit** — today user sold manually at +5.4%; the human supplied risk-aversion the algo can't infer from price.
- **Reduced position size in extended bull regimes** so a 3-5% give-back hurts less in absolute terms.

**Output CSVs**: `output/forensic_today_20260427_100214.csv`, `output/emerg_exit_5m_20260427_095120.csv`, `output/5m_crashes_20260427_090457.csv`.

---

**🌙 END-OF-DAY HANDOFF — 2026-04-27 ~23:00 CEST. User going to bed; resume tomorrow.**

### Currently active / running
- **MIX gate is LIVE in production** (promoted ~20:30 CEST tonight, trader restarted ~22:00 successfully). 4 layers of validation (Tier 1 OOS pass, cross-window 30d+60d, cumulative 30/60/90, disjoint 3×30d). Backups: `config/regime_config_ed_pre_hrst2_20260427_evening.json` + `models/crypto_ed_production_pre_hrst2_20260427_evening.csv`.
- **1440 HRST smoke test running on laptop** (~3-4h, started ~21:30ish CEST). Output goes to `_noprod` files. Validates today's engine bug fixes + grid trim end-to-end. Should finish overnight.
- **Live trader running on a different machine** (not this one — per CLAUDE.md rule 16). Status confirmed working with MIX gate after restart.

### Tomorrow's open decisions

**1. After 1440 HRST finishes** — decide:
- Review `_noprod` output → if winner looks sensible and infrastructure validated, decide whether to also launch 2880 HRST for true R5 validation
- 2880 with `--replay-v 1440` is now ~5-6h (not 12-15h, thanks to tonight's --replay-v fix) — much more palatable
- BUT: MIX is already live with 4 layers of validation, so R5 is now belt-and-braces, not strict requirement. Skipping is OK.

**2. Monitor MIX live performance** — passive, 1-2 weeks. Rollback criteria (per CLAUDE.md R4):
- Realized alpha drops >15pp vs sim baseline over first 10 trades
- Max DD exceeds -10% (worse than current PROD's historical -10.02% on 60d)
- Signals consistently blocked at moments where forward 24h is positive (gate too aggressive)

Rollback command (one-line, instant, no restart needed):
```powershell
copy config\regime_config_ed_pre_hrst2_20260427_evening.json config\regime_config_ed.json
```

**3. Expose available trader commands** — design decision pending:
- `/help` currently lists 8 commands but TRADER ACTUALLY HAS 13 — missing /buy, /sell, /hold, /cfg_*
- Options: (a) update /help text in engine code (~5 min), (b) BotFather /setcommands for native Telegram menu (~5 min user-side), (c) add /commands shortcut with buttons (~15 min), (d) document in CLAUDE.md
- Recommendation: A+B together (~15 min total)
- Existing /help defined in `crypto_revolut_ed_v2.py:_handle_help_command` (~line 2392)
- Inventory of all 13 commands captured in conversation; summary: /buy /sell /hold /cfg_ /chart /gate /help /pause /resume /setup /status /stop /sync

**4. T5b decision** — low priority:
- T5b winner (`bull_dd≥3% + bear_dd≥5% + bull_conf=90`) was +11.35pp on 60d standalone (against OLD PROD baseline)
- Likely also redundant with MIX (entry-side overlap, same pattern as T1b which dropped 87% of its alpha when measured against MIX-baseline)
- Re-test on MIX baseline before considering ship
- Test script template: copy structure from `tools/test_t1b_on_top_of_mix.py`

### Engine bug fixes shipped today (already persistent in CLAUDE.md history)
1. **Bug A (--replay propagation)**: HRST → Mode H → Mode D/V chain wasn't propagating CLI flag. Fixed at 5 sites.
2. **Bug B (MIN_GRID_TRADES dynamic)**: hardcoded `trades >= 8` killed all candidates on 5h horizon with small folds. Replaced with `max(4, n // 360)` dynamic threshold.
3. **Grid trim**: GRID_COMBOS 3→2 (dropped RF+XGB), GRID_WINDOWS 6→3 (dropped 200/250/300), GRID_FEATURES 6→4 (dropped 20/30) → 324→72 evals (-78% Mode D time). Backed by 20-winner empirical evidence — all kept-space configs match historical winners; all dropped configs had 0/20 wins.
4. **Optimizer bot aligned**: MODE_TIME_EST recalibrated (D=25→6, HRST=160→110, etc.), grid_total default updated.
5. **macro_daily SLA bumped 72h→96h**: Monday-morning yfinance lag was blocking hot-reload preflight. Fixed.
6. **/status enhanced** with shields + Bull G8 / Bear G8 lines for live config visibility.
7. **--replay-v flag**: decouples Mode V validation window from Mode D training window. 4mo HRST runtime drops from ~12-15h to ~5-6h with `--replay-v 1440`.

### Production state at end of day
- Live: MIX gate, bull_conf=80, bear_conf=85, both shields ON, bear gate unchanged (`rr30≥9.0% OR rr36≥9.0% cd=48h`), regime detector `price>sma72`, bull=6h bear=7h
- Backups in place for one-command rollback
- Engine + optimizer bot + trader all aligned

### What I'd start with tomorrow

1. Check the laptop's 1440 HRST log — did it finish? Did winners look reasonable?
2. Send `/status` in Telegram → confirm MIX gate still showing as live, no overnight regressions
3. If both clean → decide on 2880 R5 launch (now affordable at ~5-6h with --replay-v) OR move on to other work
4. If user wants to address Telegram commands discoverability → option A+B is the recommended path

---


## 📦 MERGED TOPIC: T1b ensemble vote (2026-04-27 morning discovery → evening shelved on MIX baseline)

*This topic groups several banners that were previously written separately as the work progressed. Content kept verbatim, just reordered chronologically and merged under one header.*

---

**🚨 HIGH PRIORITY — 2026-04-27 morning T1b TRUE multi-horizon ensemble vote — NEW TOP WINNER.**

Per-horizon signal cache `data/eth_per_horizon_signals_90d.pkl` was finally generated overnight (h=5,6,7,8 × 91d, ~2184 signals each). Built [tools/test_t1b_ensemble_vote.py](tools/test_t1b_ensemble_vote.py) — replaces the earlier T1b PROXY (conf-threshold sweep) with the real thing: at each bar, tally votes from horizons in `subset` requiring conf ≥ vote_thr; require k_buy votes for BUY, k_sell votes for SELL.

**Top winners — positive on all 3 windows (sorted by 60d delta vs same-base-conf baseline):**

| Rank | Config | d_30d | **d_60d** | d_90d | n_60d |
|---|---|---|---|---|---|
| 1 | **`58_only k_buy=1 k_sell=2 thr=85` base_conf=80** | +0.20 | **+19.84** | +16.38 | 20 |
| 2 | `567_only k_buy=1 k_sell=3 thr=90` base_conf=80 | +1.12 | +18.09 | +22.09 | 14 |
| 3 | `58_only k_buy=1 k_sell=2 thr=90` base_conf=80 | +5.58 | +6.60 | +18.39 | 18 |

**Top winner substantively (#1):** Use ONLY horizons 5h + 8h (drop the 6h/7h middle). Enter on ANY one BUY at conf ≥ 85%. EXIT requires BOTH 5h AND 8h to say SELL at conf ≥ 85% (2-of-2 confirmation). Asymmetric: easy entry, hard exit. **+19.84pp on 60d (66.45% strategy vs 46.61% baseline) — beats T5b winner's +11.35pp by +8.49pp.** Only 20 trades on 60d (vs 46 baseline = much more selective).

**Strong-but-thin winner (#2):** Use h=5,6,7. Enter on ANY one BUY at conf ≥ 90%. EXIT requires ALL 3 (5h+6h+7h) to say SELL at conf ≥ 90%. Most selective rule (n_60d=14 — borderline statistical thinness) but strongest 90d alpha (+22.09pp).

**Output CSV:** `output/t1b_ensemble_vote_20260427_085549.csv` (full sweep: 7 subsets × k_buy 1..N × k_sell 1..N × thr {70,80,85,90} × base_conf {80,90} × 3 windows).

**Implementation cost:** Higher than T5b — requires running ALL 4 horizon models (5h/6h/7h/8h) per cycle in live trader, not just the regime-anchor horizon. Currently the trader loads only `bull_h` (6h) and `bear_h` (7h) per `regime_config_ed.json`. Adding horizons 5h+8h: ~2× compute per cycle (2 extra model trainings per hour). Need new schema for "vote subset" + k_buy/k_sell + vote_thr.

**Caution:** 30d delta is +0.20 (barely positive). 60d/90d strong but recent 30d shows the regime-tilt risk — over-confirmation can miss recent shorter-cycle moves. Recommend 4mo HRST validation (`--replay 2880`) before promotion.

**Comparison to tonight's earlier T5b winner:**
- T1b ensemble (NEW): 60d +19.84pp / 20 trades / requires multi-model live infra
- T5b entry filter: 60d +11.35pp / 33 trades / config-only + ~30 lines code
- Trade-off: T1b is +75% better alpha but harder to ship and needs thicker validation.

---


---

**✅ T1b ENSEMBLE VOTE — SHELVED 2026-04-27 evening (after MIX promotion).**

T1b winner from 2026-04-26 sweep showed +19.84pp on 60d standalone. Re-tested on top of MIX-active baseline ([tools/test_t1b_on_top_of_mix.py](tools/test_t1b_on_top_of_mix.py)):

| Metric | Original (vs old PROD) | On top of MIX |
|---|---|---|
| 60d delta | +19.84pp | **+2.42pp** (87% gain evaporated) |
| 30d delta | (positive) | **-15.24pp** (HURTS recent month) |
| OOS held-out 30d | not tested | **+0.81pp** (essentially noise) |

**Mechanism redundancy**: T1b ensemble (k_sell=2 multi-horizon SELL agreement) was supposed to be exit-side, MIX is entry-side — should be orthogonal. But in practice, the trades T1b "saves" by holding through noise = the same trades MIX previously prevented from being entered. Once MIX cleans entries, exit-side noise drops and T1b has less to filter.

**Implementation cost** = 2× compute per cycle (4 horizon models vs 2), new schema, code changes in trader + live_trader. **Cost > marginal gain.**

**Methodology lesson**: most "winners" measured against weak baselines lose value when measured against a stronger baseline. T5b (also entry-side anti-overheat) likely shows the same collapse — should be retested on MIX baseline before considering. Don't promote anything from yesterday's sweep without re-measuring against current PROD (which is now MIX, not old gate).

**Output CSV**: `output/t1b_on_mix_20260427_224212.csv` (full sweep, all configs × 4 windows).

---


## 📦 MERGED TOPIC: MIX gate (2026-04-27 candidate → promoted same evening)

*This topic groups several banners that were previously written separately as the work progressed. Content kept verbatim, just reordered chronologically and merged under one header.*

---

**🚨 HIGH PRIORITY — 2026-04-27 afternoon: BULL RALLY-COOLDOWN GATE UPGRADE CANDIDATE — Tier 1 OOS PASSED, decision pending.**

After exhausting emergency-exit and short-term price-action options (see other 2026-04-27 entries below), pivoted to ENTRY-SIDE optimization. Built and ran a multi-stage cross-window-robust analysis on Mode G's rally-cooldown sweep, with extended HORIZONS list. Tier 1 OOS validation passed cleanly. **Awaiting promotion decision.**

### CANDIDATE FOR PROMOTION: bull rally-cooldown gate parameter change

**Current production:** `bull.rally_cooldown` = `rr20h>=4.0% OR rr24h>=4.5%, cd=12h`

**Proposed (MIX winner):** `bull.rally_cooldown` = **`rr12h>=2.5% OR rr20h>=4.0%, cd=24h`**

Diff: h_short 20→12, h_long 24→20, t_short 4.0→2.5, t_long 4.5→4.0, **cd_hours 12→24 (doubled)**.

Bear gate, shields, conf thresholds, max_hold, min_sell_pnl: **all unchanged from current PROD** (no other knob touched).

### Methodology used (be precise about what was/wasn't done)

| Step | What | Status today |
|---|---|---|
| HRS / HRST (regime detector + bull/bear horizons + confs + shields) | Full retune | **NOT redone** — used existing AB matrix Variant A (2-month optimum from 2026-04-26 promotion) |
| T (shield + min_sell_pnl + max_hold) | Sweep | **NOT redone** — kept current PROD values (bull/bear shield ON, 0.5%, 10h) |
| G (rally cooldown gate) | Sweep with **HORIZONS extended to include 48h** | **REDONE on 1m AND 2m**, then aggregated by cross-window STRICT intersection |
| Cross-window aggregation | 1m∩2m STRICT-passing intersection (9,113 configs out of 62,388 each) | New methodology added today |
| Tier 1 OOS validation | 3rd month (oldest 30d of 90d cache, not used in G optimization) | **DONE — PASSED** |

### Tier 1 held-out OOS validation result (this is the key promotion-justifier)

Tested on 2026-01-18 → 2026-02-17 (the FIRST 30 days of the 90d cache, not used in either 1m or 2m optimization):

| Setup | Held-out return | Δ vs no-gate | Δ vs current PROD | Verdict |
|---|---|---|---|---|
| NO GATE (baseline) | -5.84% | — | -0.89pp | reference |
| CURRENT PROD (`rr20≥4 OR rr24≥4.5 cd=12h`) | -4.95% | +0.89pp | reference | reference |
| **MIX (`rr12≥2.5 OR rr20≥4 cd=24h`)** | **+0.90%** | **+6.74pp** | **+5.85pp** | **PASS ✅** |
| 60d-opt with 48h (`rr24≥5 OR rr48≥6.5 cd=24h`) | -6.80% | -0.96pp | -1.85pp | FAIL ❌ |

**Critical observation:** B&H on this period was -40.82% (bear/correction phase). MIX gate not only didn't lose money during a bear regime — it ended slightly positive. This is a stress test the gate passed. Setup 60d-opt FAILED the held-out test, which validated the original concern that the 48h extension was overfit to recent regime.

### Performance summary across 3 windows (in-sample 1m & 2m + OOS held-out 3rd month)

| Window | Current PROD | MIX gate | Δ vs PROD | Status |
|---|---|---|---|---|
| Held-out 30d (2026-01-18→02-17) — OOS | -4.95% | **+0.90%** | **+5.85pp** | **OOS PASS** ✅ |
| 30d (used in opt) | +22.06% | +31.79% | +9.73pp | in-sample |
| 60d (used in opt) | +47.76% | +64.51% | +16.75pp | in-sample |
| **Average across 3 windows** | **+21.62%** | **+32.40%** | **+10.78pp** | — |

Drawdown also improves: 30d from -4.75% to -1.99% (-58%); 60d from -10.02% to -5.16% (-49%).

Win rate jumps from 73%/71% to 83%/79% across 30d/60d. Trade count drops 30-40% (more selective entries).

### Overfitting analysis — what the methodology DID and DID NOT eliminate

✅ **Filtered out:**
- Single-period flukes (each window's STRICT requires H1+H2+REF check)
- Pure-luck winners (cross-window must align 6 sub-period checks across 30d and 60d)
- 48h-extension overfit (60d-opt winner with 48h FAILED OOS, MIX winner without 48h PASSED)
- The cross-window MIX winner happens to also be the 30d-only winner — strong robustness coincidence

❌ **Not filtered out:**
- HRS not refreshed (gate optimized on stale model signals from 2026-04-26 cache)
- T not jointly re-optimized with new G
- No 4-month replay (CLAUDE.md rule R5 — gold standard for promotion)
- 1m, 2m, 3rd-month all from same continuous 90d cache (not truly independent regimes)

### Implementation plan if promoted (config-only change)

```bash
# Backup
cp config/regime_config_ed.json config/regime_config_ed_pre_mix_20260427.json

# Edit config: regime_config_ed.json -> ETH.bull.rally_cooldown:
#   h_short:      20 -> 12
#   h_long:       24 -> 20
#   t_short_pct: 4.0 -> 2.5
#   t_long_pct:  4.5 -> 4.0
#   cd_hours:     12 -> 24

# Hot-reloads in 5 min, no restart needed.
```

**Rollback:** `copy config\regime_config_ed_pre_mix_20260427.json config\regime_config_ed.json`. One-command, instant.

### Recommended path before flipping live

Two options the user is deciding between:

**(a) Ship now + queue HRST validation in parallel.** Config-only change, instant rollback. If 1-2 weeks of live perf is bad, rollback. Meanwhile run 4mo HRST to confirm.

**(b) Hold and run 4mo HRST first.** `python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 2880 --no-persist`. ~6-8h desktop. Decide based on whether 4mo winner matches MIX. Adds ~1 day before promotion but completes R5.

### Test artifacts created today (full audit trail)

All read-only, none touched production:

- [tools/sim_mode_g_with_48h.py](tools/sim_mode_g_with_48h.py) — extended Mode G with 48h, PROD-only vs PROD+48h comparison on 60d
- [tools/sim_mode_g_30_60_robust.py](tools/sim_mode_g_30_60_robust.py) — extended Mode G run on BOTH 30d and 60d, intersection of STRICT winners
- [tools/sweep_rally_48h_thresholds.py](tools/sweep_rally_48h_thresholds.py) — finer sweep of rally_48h thresholds 3.0-5.0% (manual approach)
- [tools/compare_shield_vs_rally_block.py](tools/compare_shield_vs_rally_block.py) — apples-to-apples shield-on/off vs rally-block matrix
- [tools/summary_table_30_60.py](tools/summary_table_30_60.py) — first clean comparison table (later superseded)
- [tools/final_comparison_table.py](tools/final_comparison_table.py) — full comparison incl shield states + gate strings
- [tools/tier1_held_out_test.py](tools/tier1_held_out_test.py) — OOS validation on first 30d of cache

Output CSVs:
- `output/sim_mode_g_robust_<ts>.csv` — 9,113 cross-window-robust winners
- `output/sweep_rally_48h_<ts>.csv` — fine 48h threshold grid
- `output/cmp_shield_vs_block_<ts>.csv` — shield/block matrix
- `output/summary_30_60_<ts>.csv` and `output/final_comparison_<ts>.csv` — clean comparison tables
- (Tier 1 prints to console; results captured above.)

### CRITICAL DECISION POINT

User has not yet flipped the switch. To promote, follow the implementation plan above. Pre-flight backup tag has not been created yet.

---


---

**✅ MIX GATE PROMOTED TO LIVE — 2026-04-27 ~20:30 CEST.**

User shipped MIX rally-cooldown gate to live production via config-only change to `config/regime_config_ed.json` ETH.bull.rally_cooldown:

```diff
- h_short: 20  → h_short: 12
- h_long:  24  → h_long:  20
- t_short_pct: 4.0  → t_short_pct: 2.5
- t_long_pct:  4.5  → t_long_pct:  4.0
- cd_hours: 12  → cd_hours: 24
```

**No other production knob changed.** Bull conf 80% (unchanged), bear conf 85% (unchanged), shields ON for both (unchanged), bear rally_cooldown unchanged (`rr30≥9.0% OR rr36≥9.0% cd=48h`), regime detector `price>sma72` (unchanged), min_sell_pnl=0.5%, max_hold=10h (unchanged).

**Position state at promotion**: `state=cash`, `auto_trade=true`, `rally_cooldown_until=""` (no active cooldown — clean slate for new gate).

**Live trader hot-reloads `regime_config_ed.json` every 5 min** — no restart required. Within next cycle the new gate is active.

**Rollback (one command, instant)**:
```powershell
copy config\regime_config_ed_pre_hrst2_20260427_evening.json config\regime_config_ed.json
```

Backup also kept for production CSV (`models/crypto_ed_production_pre_hrst2_20260427_evening.csv`) though MIX promotion did NOT touch the CSV (model selection unchanged).

### Evidence base for promotion (4 independent layers)

1. **Tier 1 OOS held-out 30d** (FIRST 30d of 90d cache, never used in optimization): MIX +0.90% vs PROD -4.95% during a -40.82% B&H period → **+5.85pp** ← only methodologically clean comparison
2. **Cross-window 30d+60d STRICT intersection** (MIX discovery methodology): same gate emerged
3. **Cumulative 90d+60d+30d STRICT intersection** (tonight's analysis via `tools/tg_window_decomposition.py`): same MIX gate emerged
4. **Disjoint 3×30d STRICT intersection**: close cousin emerged (rr12+rr14+cd=24, structurally identical family)

In-sample numbers (less rigorous but consistent):
- 30d in-sample: +9.73pp over PROD
- 60d in-sample: +15.17pp over PROD
- 90d in-sample: +25.56pp over PROD
- Drawdown reduced ~50% on both 30d and 60d

### Monitor + rollback criteria

**Monitor for 1-2 weeks** of live performance. Rollback if:
- Cumulative realized alpha drops >15pp vs sim baseline over first 10 trades (matches CLAUDE.md rule R4 standing policy)
- Max drawdown exceeds historical -10.02% (current PROD's 60d max)
- Live signals consistently blocked at moments where forward 24h is positive (gate too aggressive)

If rollback triggers: one `copy` command, hot-reloads in 5 min.

### Standing R5 caveat

Per CLAUDE.md rule R5: HRST validation gate normally requires 4-month replay confirmation before live promotion. **R5 NOT formally completed for this promotion.** User decided Tier 1 OOS + 3 cross-window confirmations were sufficient evidence given:
- Config-only change (no model/code modification)
- Instant rollback available
- All 4 evidence layers point to same gate family
- Full HRST 4mo would have taken 12-15h on laptop (declared not worth wait)

The 1440 HRST currently running is engineering smoke-test (validates the bug fixes + grid trim end-to-end), NOT R5 validation. The 2880 HRST queued for later WOULD be R5; if it picks something materially different from MIX, revisit.

---


## 📦 MERGED TOPIC: 4mo HRST validation arc (2026-04-27 launched 16:45 → evening realisation that --replay scales Mode V too)

*This topic groups several banners that were previously written separately as the work progressed. Content kept verbatim, just reordered chronologically and merged under one header.*

---

**🟡 RUNNING NOW — 2026-04-27 ~16:45 CEST — 4MO HRST VALIDATION (Tier 2 R5) on LAPTOP.**

Job launched on laptop ~16:45 CEST 2026-04-27.

**Command:** `& "C:\Users\Alex\algo_trading\venv\Scripts\python.exe" .\crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 2880 --no-persist`

**Working dir:** `C:\Users\Alex\algo_trading\engine` (laptop local copy)

**ETA:** ~6-8h → finishes between **22:45 → 00:45** tonight.

**Purpose:** Tier 2 R5 validation of the MIX rally-cooldown gate candidate (`rr12≥2.5 OR rr20≥4.0 cd=24h`). Tier 1 OOS already PASSED earlier today (held-out 30d test, +6.74pp vs no-gate, +5.85pp vs current PROD). This is the gold-standard 4-month HRST check before promotion.

**Output files when done:**
- `C:\Users\Alex\algo_trading\engine\config\regime_config_ed_noprod.json`
- `C:\Users\Alex\algo_trading\engine\models\crypto_ed_production_noprod.csv`
- `C:\Users\Alex\algo_trading\engine\logs\ed_v1_<timestamp>.log`

**When the user checks back, run this to compare 4mo winner to MIX:**
```powershell
Get-Content .\config\regime_config_ed_noprod.json | ConvertFrom-Json | Select-Object -ExpandProperty ETH | Select-Object -ExpandProperty bull | Select-Object -ExpandProperty rally_cooldown
```

**Compare to MIX winner:**
| Field | MIX (today's pick) | 4mo HRST result | Match? |
|---|---|---|---|
| `h_short` | **12** | ? | |
| `h_long` | **20** | ? | |
| `t_short_pct` | **2.5** | ? | |
| `t_long_pct` | **4.0** | ? | |
| `cd_hours` | **24** | ? | |

**Decision tree on completion:**
- 4mo bull gate matches MIX (within ±1 step on each param) → strong R5 confirmation, ship MIX with HIGH confidence
- 4mo bull gate is significantly different (e.g., different h_short/h_long pair, threshold deltas >2 steps, cd doubled or halved) → window-sensitivity; either ship MIX with MEDIUM confidence (config-revertible) OR shelve and wait for live data
- 4mo HRS picks a different regime detector or bull/bear horizons → bigger structural change; consult before any promotion
- Mode T converged + writes successfully → check it didn't hit `max_iter` (oscillation flag)

**Note:** 4mo HRST won't include 48h in Mode G's search by default (production HORIZONS still maxes at 36h). MIX winner is `rr12≥2.5 OR rr20≥4.0 cd=24h` — uses no 48h, so 4mo can find this same config natively if signal is real.

**Aliveness monitoring (if concern arises):**
```powershell
Get-ChildItem .\logs\ed_v1_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Select-Object Name, LastWriteTime
```

If log mtime < 5 min → alive. >30 min stale + no `python.exe` in tasklist → likely dead. Mode R silence up to 2h is normal (per CLAUDE.md rule 16) — don't declare dead during Mode R phase.

**Power management:** Laptop set with `powercfg /requestsoverride PROCESS python.exe SYSTEM AWAYMODE EXECUTION DISPLAY` to prevent Modern Standby from killing the job overnight.

---


---

**🚨 OPEN DECISION — 2026-04-27 evening: 4MO HRST RUNNING SLOW because `--replay` also scales Mode V (not just Mode D).**

**Diagnosis**: today's bug-fix cascade revealed an unintended consequence:
- Bug A fix (--replay propagation through HRST chain) → now correctly uses 2880 rows everywhere
- BUT: Mode V's STEP 1 ("backtest top 6 D candidates") runs the FULL 2880-hour replay through live signal generator (one model retrain per hour). Cost = ~20-30 min per candidate × 6 candidates × 4 horizons = **~10-12h just for Mode V**. Plus Mode S (3,920 combos × 2880 signals) and Mode T+G iteration also doubled.
- Realistic 4mo HRST runtime: **~12-15 hours** (not the 4-6h I originally estimated; Mode D trim only helped that one phase).

**Options to choose from when you read this:**

(a) **Kill + restart at `--replay 1440`** (~3-4h total). Same data scope as AB matrix; refreshes models + gate sweep without 4mo R5 grade.
```powershell
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist
```

(b) **Let it finish overnight** (~10-13h remaining at 19:00 CEST = finishes ~05:00-08:00 next morning). Get true 4mo R5 validation. Battery + cooling + Modern Standby risk.

(c) **Ship MIX based on Tier 1 OOS alone** (zero hours, instant rollback). Tier 1 already passed cleanly (+5.85pp vs PROD on truly held-out 30d). 4mo HRST was a "bonus" R5 confirmation, not strictly required for promotion.

**Recommended: (a) kill + restart at `--replay 1440`.** The marginal value of 4mo over 2mo is small (~+10-15h cost for slightly better OOS evidence on a window we'd already understand). Faster turnaround, lower risk.

**Engine bug-fixes from today (already shipped, persistent):**
- Bug A: `--replay` propagation through HRST → Mode H → D/V (4 sites added `replay_hours=` plumbing)
- Bug B: `MIN_GRID_TRADES = max(4, n // 360)` dynamic threshold (was hardcoded 8, killed all Mode D candidates on 5h-7h)
- Trim: GRID_COMBOS=2 (dropped RF+XGB), GRID_WINDOWS=3 (dropped 200/250/300), GRID_FEATURES=4 (dropped 20/30) → 324→72 evals (-78%)
- Optimizer bot aligned: MODE_TIME_EST recalibrated, grid_total default updated

**Engine bug B follow-up (KNOWN, NOT FIXED):** `--replay` cost scaling in Mode V/S/T+G is a feature, not a bug — these phases legitimately need the full replay window for accurate live backtest. If you want a permanent fix, would need separate `--replay-d` and `--replay-v` flags or hardcode a Mode V cap. Defer to user.

---

**🚨 HIGH PRIORITY — 2026-04-26 evening test sweep results (T5-T8). Persist across logoff.**

Tonight ran 4 standalone batch harnesses against 90d cached signal stream
(`data/eth_sl_signals_90d.pkl`). All read-only. None touched production.
Canonical evaluation window: **60d**. Results also reported on 30d / 90d.

**Baselines (ETH, prod config):**
- 30d: +22.03% / 26 trades / 58% WR
- 60d: +55.68% / 46 trades / 59% WR
- 90d: +47.17% / 68 trades / 56% WR

**Test files written:**
- [tools/test_t5_batch.py](tools/test_t5_batch.py) — 10 ideas: T5a asym sell-conf uplift, T5b per-regime dd, T5c trailing peak/retain, T5d per-regime min_sell_pnl, T5e rally-momentum exit, T5f days-down entry, T5g conf-weighted max_hold, T5h shield auto-off at profit, T5i vol-pctile entry gate, T5j sell-conf decay
- [tools/test_t6_triple_barrier.py](tools/test_t6_triple_barrier.py) — vol-adaptive triple barrier (upper σ × lower σ × vertical h) replacing model SELL+shield+max_hold
- [tools/test_t7_meta_proxy.py](tools/test_t7_meta_proxy.py) — cheap meta-labeling proxy (logistic regression on 7 meta features, walk-forward train_n × threshold sweep)
- [tools/test_t8_gdelt_overlay.py](tools/test_t8_gdelt_overlay.py) — GDELT geopolitical entry-overlay filters (geo_vol pctile, iran_tone, geo_tone_chg24h, only-improving-iran-24h)

**Output CSVs (timestamped):**
- `output/t5_batch_20260426_223330.csv`
- `output/t6_triple_barrier_20260426_223805.csv`
- `output/t7_meta_proxy_20260426_224605.csv`
- `output/t8_gdelt_overlay_20260426_225122.csv`

**Winners — positive on all 3 windows (30d/60d/90d):**

| Rank | Config | d_30d | **d_60d** | d_90d | n_60d |
|---|---|---|---|---|---|
| 1 | **T5b: bull_conf=90 + bull_dd≥3% + bear_dd≥5%** | +13.07 | **+11.35** | +26.12 | 33 |
| 2 | T5b: bull_conf=90 + bull_dd≥2% + bear_dd≥5% | +12.01 | +10.04 | +24.15 | 34 |
| 3 | T6: triple barrier up=6σ lo=2σ vert=24h conf=90 | +6.77 | +10.48 | +1.24 | 47 |
| 4 | T5b: bull_conf=90 + bull_dd≥3% + bear_dd≥3% | +12.67 | +6.49 | +18.04 | 34 |
| 5 | T5b: bull_conf=90 + bull_dd≥2% + bear_dd≥3% | +11.61 | +5.22 | +16.16 | 35 |
| 6 | T5c: trailing peak=3% retain=70% | +3.25 | +4.99 | +5.27 | 48 |
| 7 | T5j: bull_conf=90 + sell_conf_decay@h≥4 = -15pp | +5.72 | +3.92 | +16.64 | 45 |
| 8 | T5b: bull_conf=90 + bull_dd≥2% + bear_dd≥2% | +10.49 | +3.87 | +15.06 | 35 |

**Top winner substantively:** "Wait for ETH ≥3% off 7d high before BUY in bull regime, ≥5% off in bear regime, only enter at ≥90% confidence." Pure mean-reversion entry filter on top of model signal — no exit-side change, no shield change.

**Dead ideas tonight (NULL or NEGATIVE on 60d):**
- T5a sell-conf uplift (asymmetric exit) — NEGATIVE
- T5d per-regime min_sell_pnl — all variants ≤0 on 60d
- T5e rally-momentum exit override — all NEGATIVE (force-selling on momentum is wrong direction)
- T5f days-down entry — disastrous (−36 to −55pp)
- T5g conf-weighted max_hold extension — all NEGATIVE
- T5h shield auto-off at high profit — 0.00 (never fires in 60d sample; shield rarely binds when profit is high)
- T5i vol-percentile entry gate — small/null
- **T7 meta-labeling proxy** — every config NEGATIVE on 60d (best −10.76pp). Cheap proxy doesn't replicate literature claims; would need full LGBM-based meta from `crypto_trading_system_meta.py`
- **T8 GDELT overlay** — every config 0 to −26pp. GDELT data only covers 21% of recent signals (feed died 2026-04-19); overlay shows zero current value

**Untested combo worth running next:** T5b winner (#1) + T5c trailing peak=3% retain=70% — orthogonal mechanisms (entry filter vs exit lock), addresses today's "fear of give-back" concern directly.

**Promotion gate (per standing rule R5):** before live deployment, validate top T5b config on 4mo HRST replay (`--replay 2880`) for structural-consistency check. ~6-8h desktop runtime.

**Implementation cost:** T5b winner = config-only change (no code) — `bull.min_confidence: 80→90`, plus new keys `bull.dd_from_7d_high_min_pct: 3.0` and `bear.dd_from_7d_high_min_pct: 5.0` (these need ~30 lines of code added to live trader to compute dd_from_7d_high at each tick and gate BUYs). T5c trailing lock = ~40 lines, peak tracking + force-SELL bypass.

**Still unexplored (architectural, larger scope):**
- Triple-barrier as LABEL (not exit overlay) for retraining — new Mode D, ~3-6h
- Real meta-labeling with full LGBM + 25-feature set via `crypto_trading_system_meta.py` HRST run, 30-60 min — earlier biased run showed +23pp; clean re-run needed
- GDELT re-download + Mode F re-rank with current importance — small cost, but T8 overlay shows existing data has no signal in current regime

---

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

**✅ TRADER AUDIT FIXES — all deferred items resolved (2026-04-26):**

Bundles 1+2+3 (commits `8766d05` → `1124447`) shipped 14 distinct money-correctness bugs. Bundle audit findings deferred at the time were re-evaluated 2026-04-26 and resolved as follows:

- ✅ **M-10** — Orphan in-flight trade reconciliation shipped in commit `1124447` via `/trades/private/{symbol}` query in sync_positions.
- ✅ **N-15** — cycle_metrics now records a row on early-return paths (load_data_failed, regime_error, no_models_loaded). New `skip_reason` column; CSV auto-rotates to `cycle_metrics.v1.csv` on schema change.
- ❌ **M-20** — OBSOLETE (verified 2026-04-26). Manifest regen subprocess only fires when `regime_config_ed.json` actually changes (`_reload_trading_config` dict-equality gate). Original "every 5 min" framing was wrong.
- ❌ **N-06** — OBSOLETE (verified 2026-04-26). `_atomic_write_json` callers are all sequential within `crypto_trading_system_ed.py`; no threading. PID-suffixed tmp paths handle cross-process safely.

**Falsely flagged or auto-mitigated (kept for audit history):** M-05 (auto-mitigated by M-02/M-03), M-08 (ledger-delta captures), M-09 (unreachable + ledger-delta), M-11/M-12/M-14 (defensive only), M-18 (`/pause` not gating sync is intentional).

Rollback anchor: `git tag pre-trader-bundle-20260425` (commit `8766d05`).

---

**✅ LIVE TRADER DATA-UPDATE SANITY CHECK — fully shipped (originally 2026-04-23 86%-pinned bug response):**

Original incident: ETH 5h@86% stuck for 7+ hours because `xa_btc_lag2h` was NaN (BTC OHLCV stale 49h since BTC was disabled/not downloaded). `dropna(subset=feature_cols+['label'])` killed every recent row, `i=n-1` pointed at a 49h-old row, model retrained on the same frozen bar → identical 86% every hour. Prior 7h@99%/31h pinned bug was same class (likely `oc_mvrv_chg1d` stale).

All five hard rules now enforced in code:

1. ✅ **Refuse to predict on stale data** — M-01 in `crypto_live_trader_ed.py`. Computes `lag_hours` after dropna, returns None + Telegram alert if >2h.
2. ✅ **Don't drop whole rows on single-feature NaN** — M-01b decoupled `df_train` (label-NaN dropna) from `df` (all rows kept for inference). `keep_label_nan_tail=True` flag on `build_all_features`.
3. ✅ **BTC-in-ETH feature pipeline (structurally safe)** — `leaders_for['ETH']` is still `['BTC']` at `crypto_trading_system_ed.py:1281`, but M-25 cadence-aware staleness check makes the original failure mode impossible: trader refuses to predict if any non-sparse feature is >2h stale. Lead-lag is real signal per literature; removing it would lose alpha for no safety benefit. Rule's intent satisfied without the literal removal.
4. ✅ **Pre-inference data-freshness gate** — M-25 cadence-aware: `oc_*` 60h, `fg_*` 36h, hourly defaults 2h. Per-feature SLA in `config/feature_sources.json`. Refuses cycle on stale.
5. ✅ **Sparse-tail audit** — `tools/audit_features.py:134-169` scans last 48 bars of every prod-CSV feature, flags NaN count + last-valid lag, lists models using each.

---


## 📦 MERGED TOPIC: AB matrix evolution (2026-04-22 planning → 2026-04-25 4-variant → 2026-04-27 ABCDE final)

*This topic groups several banners that were previously written separately as the work progressed. Content kept verbatim, just reordered chronologically and merged under one header.*

---

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


---

**🧪 PRIOR LAUNCH — AB MATRIX 4-VARIANT FOCUS (2026-04-24 22:40 CEST → completed 2026-04-25 12:38 CEST, partial — D never ran due to system freeze):**

Launched AFTER the third-pass audit shipped (file-handle leak + div-by-zero guards, commit `eaf80e9`) and AFTER today's 22:07 HRST promoted a clean winner (`sma168>sma480 bull=7h@65% bear=6h@85%` — already live). This matrix runs on the first fully-clean data snapshot: 1432/1440 rows, label-tail fix, sparse-feature quarantine, div-by-zero guards, atomic writes — none of the prior 3 weeks' results were trained on this data quality.

Command: `python tools/ab_matrix_runner.py --variants focus --skip-vol` (seed 42 default, matches today's live HRST for B-variant replication check).

| Variant | Floor | Trim | Meta | Purpose |
|---|---|---|---|---|
| A_floorON_trimOFF | ON | OFF | — | Floor alone on full 184-feature universe — does floor still matter without trim? |
| B_floorON_trimON | ON | ON | — | **Replicates today's 22:07 live HRST.** Sanity check that matrix subprocess = direct invocation (same seed + data). |
| C_floorOFF_trimOFF | OFF | OFF | — | Raw universe, no guarantees. Tests whether floor's feature-family floor is doing real work. |
| D_floorON_trimON_metaON | ON | ON | p≥0.45 | **R3 meta-labeling retest** — added 2026-04-24 22:40 (commit `23b73c1`). B↔D isolates meta contribution on clean primary. |

Runtime: 4 × ~4h = ~16h laptop. Results in `output/ab_matrix_results_<timestamp>.csv` + tagged `_noprod_{A,B,C,D}.*` files.

**Decision rules when matrix finishes (2026-04-25 afternoon):**
- **B replicates today's HRST** (detector + horizons + confs match within seed noise): sanity check passes, matrix infrastructure trustworthy.
- **Best (A/B/C)** alpha > today's live HRST by ≥5pp on Mode T REF: promote that variant instead.
- **D > B by ≥5pp**: meta filter is shippable behind a config flag (resolves R3).
- **D within ±5pp of B**: shelve meta permanently — was dead-end masquerading as signal on biased data.
- All within noise: keep today's live HRST running, revisit after next week's live performance.

**Earlier (superseded) matrix results — kept for historical record only:**
The 2026-04-22 17:32 matrix promoted V1 (`tsmom_672h 5h@85%/6h@80%` +122.59% Mode T). The 2026-04-24 07:40 seed-2026 relaunch promoted intermediate variants. Both were trained on poisoned data (672 rows + label-tail NaN + div-by-zero features). **Do not reference those numbers as baselines** — they're not comparable to this matrix's clean output.

---


---

**✅ ABCDE MATRIX FULL RESULTS — completed 2026-04-27 01:18 CEST. R3 (meta-labeling) RESOLVED — SHELVED.**

All 5 variants ran to completion. Output CSV: `output/ab_matrix_results_20260426_151744.csv` (last variant E timestamp).

**Final ranking by Mode T REF (canonical 60d return metric):**

| Rank | Variant | t_ref % | Detector | Bull | Bear | Trim | Floor | Meta | Bull Gate | Bear Gate | Conv iter | Runtime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **A_floorON_trimOFF** ← LIVE since 2026-04-26 14:22 | **+66.79%** | `price>sma72` | 6h@80% shield ON | 7h@85% shield ON | OFF | ON | — | `rr20≥4.0 OR rr24≥4.5 cd=12h` | `rr30≥9.0 OR rr36≥9.0 cd=48h` | 2 | 447 min |
| 2 | E_floorON_trimOFF_metaON | +64.67% | `sma168>sma480` | 8h@75% shield ON | 5h@65% shield OFF | OFF | ON | p≥0.45 | OFF | OFF | 2 | 601 min |
| 3 | C_floorOFF_trimOFF | +62.68% | `sma24>sma100` | 6h@80% shield ON | 5h@65% shield OFF | OFF | OFF | — | OFF | OFF | 2 | 237 min |
| 4 | D_floorON_trimON_metaON | +51.65% | `vol_calm` | 5h@75% shield ON | 6h@65% shield OFF | ON | ON | p≥0.45 | rr12≥3.5 OR rr20≥3.0 cd=18h | rr30≥9.0 OR rr36≥8.5 cd=10h | 3 | 451 min |
| 5 | B_floorON_trimON | +45.39% | `sma24>sma100` | 6h@80% shield ON | 5h@65% shield OFF | ON | ON | — | rr20≥5.5 OR rr30≥6.5 cd=30h | rr8≥6.0 OR rr10≥6.0 cd=6h | 3 | 452 min |

**Verdicts (knob isolation):**

- **Trim = OFF preserved.** A vs B (same floor, only trim differs): A=+66.79 vs B=+45.39 → trim costs **−21.40pp**. Decision unchanged from 2026-04-26.
- **Floor = ON preserved.** A vs C (same trim, only floor differs): A=+66.79 vs C=+62.68 → floor adds **+4.11pp**. Decision unchanged.
- **Meta filter SHELVED (R3 RESOLVED).** Two ways to read meta:
  - On no-trim: E (meta+no-trim+floor) = +64.67% vs A (no-meta+no-trim+floor) = +66.79% → **meta costs −2.12pp on the strong baseline.** Below the +5pp shipping threshold.
  - On trim: D (meta+trim+floor) = +51.65% vs B (no-meta+trim+floor) = +45.39% → **meta adds +6.26pp on a weaker baseline.** But that combined config still loses to A by −15.14pp.
  - **Conclusion: meta only helps when something else is broken (trim hurting). With the right primary config, meta adds nothing.** Permanently shelve for ETH unless future runs on different data show clear benefit.

**Production decision: NO PROMOTION CHANGE.** Variant A (already live since 2026-04-26 14:22 CEST) remains the winner across the full 5-variant matrix. The most-deferred test (D, then E) confirmed meta-labeling does not justify a config change.

**Closes 2026-04-26 deferred items:**
- ✅ R3 meta-labeling decision: **SHELVED** (E/A delta = -2.12pp, below +5pp ship threshold)
- ✅ Meta-aware variants (D, E) both run to completion — no infra gaps remain
- ✅ A/B/C/D/E full factorial complete on clean data snapshot
- Matrix infra trustworthy (A's reported t_ref +66.79% matches A's live behavior post-promotion)

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

### Closed 2026-04-25 late-afternoon (post-bundle hotfixes + bundle 3)

After bundle 1+2 shipped, user attempted to bring trader back online and surfaced **two more real-world bugs my earlier fixes had missed/introduced**. Re-audit then surfaced 3 more verified-real findings, all shipped same evening. Total of 6 additional commits on top of bundles 1+2.

**Hotfixes (post-bundle 2):**

- **`5ba0af7` HOTFIX M-01b** — `crypto_trading_system_ed.py` + `crypto_live_trader_ed.py`. The morning's M-01 fix was incomplete: `build_hourly_features` already did `dropna(subset=core_cols + ['label'])` BEFORE returning df_full. So even though I patched the live trader to keep the label-NaN tail, df_full's last row was already `freshest_raw_bar − horizon_hours` — wall-clock check saw 7h lag and refused every signal. Fix: added `keep_label_nan_tail=True` parameter to `build_all_features` / `build_hourly_features`, threaded through to live trader's call site. Engine training callers leave default False. Verified: ETH df_full last bar moved from 04:00 UTC (7.34h, REFUSE) to 11:00 UTC (0.34h, PASS). 7 rows recovered.

- **`cbc4ad3` HOTFIX N-01 + N-03 + N-05** — `crypto_revolut_ed_v2.py`. (N-01 CRITICAL): `process_asset` leaked the per-asset trade lock on three balance-API-failure early-return paths (lines 1324, 1471, 1501). One API hiccup → ETH locked out forever, blocking Telegram handlers deadlock. Pre-existing from Fix N1 (commit `5a7c834`), surfaced by re-audit. Released lock at each early-return. (N-03 HIGH latent): if all signals refused, `price=0` + `get_best_bid_ask=(0,0)` → live_mid=0 → cur_pnl=−100% → brake force-sell on phantom data. Currently dormant since brake is disabled, but trap if re-enabled. Added `live_mid > 0` guard. (N-05 HIGH): M-07 Telegram handlers used blocking `with _lk:` — would freeze Telegram for 5min during a maker order, or forever after N-01 lock leak. Replaced with `acquire(timeout=2.0)`.

- **`3b38190` N-02 MIN_TRADE_USD lowered to 299.9** — `crypto_revolut_ed_v2.py:746`. BUG 1's $0.01 safety margin shrunk `/buy ETH 300` to 299.99, which then failed `< 300` minimum check with a confusing "$299.99 below $300 min" message. Lowered threshold to 299.9 so exact-$300 inputs survive the safety margin. Anything below $299.99 still rejected.

**Bundle 3 (re-audit follow-ups):**

- **`d49cc22` N-04 + N-07 + N-08** — `crypto_revolut_ed_v2.py` + `crypto_live_trader_ed.py`. (N-04): `load_trading_config()` no longer merges DEFAULT_TRADING_CONFIG under file contents — earlier merge meant deleting a key from JSON had no runtime effect (defaults re-injected). Now JSON is single source of truth; defaults seed cold-start only. (N-07 Option A): per-feature staleness check in `generate_live_signal` — for each non-sparse feature_col, compute last-valid bar age vs latest_raw_dt; refuse if any >2h. Catches the M-01b-class case where an upstream-dependent feature source (e.g. `xa_btc_lag*` when BTC OHLCV is stale) silently ffill'd into the inference. Sparse-by-design features (orderbook, IV, deriv_oi, stablecoin, whale) are skipped. (N-08): added comment documenting why training-window features get ffill'd alongside the inference row — by design from Fix #4; LGBM handles native NaN but RF/XGB partner needs imputation; acceptable noise.

- **`1124447` M-10 orphan-trade reconciliation** — `crypto_revolut_ed_v2.py`. New helpers `get_recent_private_trades` and `_reconcile_orphan_trade` query Revolut's `/trades/private/{symbol}` endpoint to recover actual fill price/timestamp when sync_positions detects a manual BUY/SELL. Match by side + quantity (within 0.1% relative). Falls back to mid-price (Fix N2 behavior) only when no recent trade matches. Trade records get `reconciled: True` marker (vs `synced: True`) so PnL consumers can distinguish exact-basis trades from approximate.

**Disaster brake disabled (user decision, 2026-04-25):** doc previously claimed `disaster_brake_pct: 5` for ETH but the JSON had no such key, making the brake dormant. User confirmed to keep it disabled. The doc snippet now correctly shows `disaster_brake_pct: 0` with a note that M-04+N-03 fixes are in place if re-enabled later.

**Bundle 3 deferred (verified clean or low-impact):**
- **N-06** (LOW) — `_atomic_write_json` in engine has no in-process lock. Latent only; sequential code paths today.
- **N-15** (LOW) — `cycle_metrics` not flushed on `process_asset` early-return paths. Observability gap, not correctness.
- **M-20** (LOW) — Hot-reload regenerates feature manifest via subprocess every 5 min (3-10s blocking). User wants explanation before deciding.

**Status by end of day:** 16 commits since `pre-trader-bundle-20260425` tag, 14 distinct bugs fixed, trader running cleanly with full instrumentation. Position file shows post-fix BUY at 22:01 last night; M-02/M-03 ledger-delta basis ready for next BUY. M-01 staleness chain working (lag_hours ~0). All maker order knobs configurable per asset.

### Closed 2026-04-25 midday (trader audit bundle 2 — 4 commits, ~50 lines)

Bundle 2 cleared all real-bug findings from the bundle-1 audit deferral list. Commits land on top of bundle 1 (`pre-trader-bundle-20260425` tag still works as the nuclear-rollback anchor — it predates both bundles). Each bundle-2 commit is independently revertable.

- **`a8bea84` M-04** — Disaster brake now reads live exchange mid via `get_best_bid_ask(symbol)`, fallback to last-closed-candle close only if price API fails. Fix is dormant in current config: `disaster_brake_pct` is unset in `regime_config_ed.json`. Doc still claims `disaster_brake_pct: 5` for ETH (config drift); user decision pending.
- **`881da46` M-17** — Shield-block Telegram message rate-limited via `_rate_limited_telegram` keyed `shield_block_{asset}` (1h cooldown). One alert per shield event instead of one per cycle.
- **`f758f46` M-06 + M-07** — `sync_positions` and Telegram settings handlers (`/cfg_{ASSET}_auto`, `/gate ASSET clear`) now acquire the per-asset trade lock around their load→modify→save sequences. Eliminates the race window that was producing duplicate `(synced)` trade records and clobbered toggle state.
- **`a441d09` M-19 + M-15** — Hot-reload does per-asset wholesale replace (not per-key merge) so deleted keys propagate; diff lines logged. Main-loop error sleep `time.sleep(120)` → `_stop_event.wait(120)` so /stop is responsive after exceptions.

**Bundle 2 deferred:** M-10 (orphan reconciliation, ~30 lines, requires new API endpoint), M-20 (manifest regen mtime cache, polish).

### Closed 2026-04-25 morning (trader audit bundle 1 — 4 commits, ~70 lines)

Triggered by user observing $9.85 phantom basis on the 2026-04-24 22:01 BUY (recorded $12008.85 vs actual exchange charge $11999). Spawned a thorough audit; surfaced 20 findings; verified each against source. Bundle 1 covers the 6 with money or restart-blocker impact.

**Rollback anchor:** `git tag pre-trader-bundle-20260425` on commit `8766d05` (last known-good before bundle 1). Nuclear revert: `git reset --hard pre-trader-bundle-20260425 && git push --force-with-lease`. Per-commit revert: `git revert <sha>` for any of the four below — they touch different code regions.

**Commits (in chronological order on `main`):**

- **`69177a2` Patches A + B** — `crypto_revolut_ed_v2.py`. Patch A: removed `max_attempts = maker_window // check_interval` recompute inside the partial-fill branch (the SELL slide was jumping backwards on boost — denominator grew, progress fraction shrank, price re-quoted upward). Patch B: added `_config_write_lock = threading.Lock()` around `save_trading_config()` writes — PID-suffixed tmp protected cross-process collisions but two threads in same process (Telegram thread + main loop) shared PID and could half-write `regime_config_ed.json`.

- **`ef542e9` M-01 staleness/horizon refusal blocker** — `crypto_live_trader_ed.py:617-655`. CRITICAL: Fix #1 (label-tail NaN) + Fix #5 (staleness threshold = 2h) silently conflicted. After `df = df_full.dropna(subset=['label'])`, the inference row was `horizon` hours behind `df_full.iloc[-1]`. `lag_hours = horizon`, threshold `> 2` → REFUSE every cycle for any horizon ≥ 3h. Trader still ran only because the in-memory process loaded pre-fix code; on next restart it would have stopped trading entirely. Fixed by decoupling: `df_train = df_full.dropna(['label'])` for training, `df = df_full` (label may be NaN at tail) for inference. lag_hours now ~0.

- **`47c0dae` M-02 + M-03 ledger-delta basis** — `crypto_revolut_ed_v2.py`. Two coupled bugs corrupted recorded basis on every trade: (M-02) `order.get('filled_size')` always returned None — Revolut API returns `filled_quantity`. Trader fell back to `buy_amount/candle_close`, then multiplied by actual `average_fill_price` → inconsistent ratio gave drift up to ~$10 on a $12k trade. (M-03) Multi-leg maker fills (partial → cancel → re-price → next leg fills) only returned the LAST leg's `od`; first leg's filled portion was real on exchange but invisible to position recorder. Both fixed simultaneously by computing basis from balance delta around the order on all 4 trade execution paths (auto BUY, auto SELL, manual /buy, manual /sell). API fields kept as fallback with correct field names. SELL pnl_usd now also computed from `delta_usd_recv - usd_invested` when both ledger deltas are present.

- **`68acd30` M-13 + M-16** — `crypto_revolut_ed_v2.py`. M-13: `save_position` PID-suffixed tmp path (parity with `save_trading_config`); protects against accidental dual-trader-instance collision. M-16: `check_telegram_commands()` returned only the LAST message in each poll batch — earlier messages had their `update_id` consumed (gone forever) but were silently dropped. Concrete failure: user clicks Buy then types /sell during a market drop; both arrive in same 5s poll batch; only one survives. Now returns a list of all pending messages in arrival order; `_telegram_command_loop` iterates and dispatches each through new `_dispatch_telegram_message` helper with per-message try/except.

**Verified empirically:**
- Today's recorded basis $12008.85 = `(buy_amount/price) × actual_fill_price` (the M-02 fallback formula), $9.85 above actual exchange charge $11999.
- Today's recorded PnL of +$6.84 was within $0.05 of actual (~$6.79) by accident — two basis errors cancelled almost exactly through the percentage math.
- The PnL cancellation will NOT hold for trades where bid-mid spread is wider or where multi-leg fills happen. Bundle 1 prevents this going forward.

**8 findings deferred to bundle 2** — see "DEFERRED TRADER AUDIT FIXES" in the TODO section above for M-04, M-06, M-07, M-10, M-15, M-17, M-19, M-20.

### Closed 2026-04-24 evening (7-point safety audit — all critical paths hardened)

Deep audit of live trader + backtest engine flagged 7 silent-failure / logical-bug classes. All fixed the same evening (2-3h of work). Each fix is minimal and tested.

**Audit findings + fixes (severity order):**

1. **Label mislabeling at tail** (CRITICAL) — [crypto_trading_system_ed.py:767-782](crypto_trading_system_ed.py#L767)
   - `future_return.shift(-horizon)` is NaN for the last `horizon` rows. `(NaN > threshold).astype(int)` coerced to 0, silently marking those rows as negative class.
   - Over 1440-row window with horizon=8 → 8 rows poisoned as false SELL examples. Gamma-weighting gives recent rows highest weight = disproportionate training bias toward SELL-at-peak.
   - **Fix**: cast to `.astype(float)` then `.where(future_return.notna(), np.nan)`. Downstream dropna removes the NaN tail cleanly. Verified: built df ends exactly `horizon` hours before raw data end.

2. **Mode T `cd_hours` convergence tolerance too loose** (HIGH) — [crypto_trading_system_ed.py:5396](crypto_trading_system_ed.py#L5396)
   - `TOL_GATE_CD = 6` → cd=10h vs cd=16h considered "converged." But 60% longer cooldown is structurally different behavior.
   - **Fix**: tightened to `TOL_GATE_CD = 2` (matches `TOL_HOLD` semantics). Convergence gate now catches oscillating cooldowns that previously shipped silently.

3. **FEATURE_SET_A silent fallback when features missing** (HIGH) — [crypto_live_trader_ed.py:570](crypto_live_trader_ed.py#L570)
   - If a prod model's `optimal_features` didn't match the current build, trader silently fell back to `FEATURE_SET_A` — a 30-feature default the model was NEVER trained with. Decision boundary broke silently.
   - **Fix**: refuse to trade (`return None`) + critical Telegram alert when ZERO features match. Also added severe-partial threshold: coverage < 50% → refuse with Telegram alert. Verified current V4 prod models all at 100% coverage.

4. **Regime detector 'error' sentinel silently defaulting to bull** (HIGH) — [crypto_live_trader_ed.py:762](crypto_live_trader_ed.py#L762), [crypto_revolut_ed_v2.py:1723](crypto_revolut_ed_v2.py#L1723)
   - Main `process_asset` trader path already handled 'error' (from earlier Fix #2). But the legacy `generate_regime_signal` defaulted to `horizon=6` + empty cfg and traded anyway. The `/status` Telegram handler displayed 'error' as 🔴 bear.
   - **Fix**: both legacy paths now refuse; `/status` displays `⚠️ DETECTOR ERROR (refusing trades)`.

5. **Staleness threshold too loose** (MEDIUM) — [crypto_live_trader_ed.py:649](crypto_live_trader_ed.py#L649)
   - Old check: `if lag_hours > horizon + 2: refuse`. At horizon=8h, allowed 10h of staleness. No wall-clock check at all — missed the case where `df_full` itself was stale.
   - **Fix**: two checks, both with fixed 2h threshold:
     - Internal gap (inference row vs freshest bar in `df_full`) > 2h → refuse + Telegram
     - Wall-clock staleness (freshest bar vs `datetime.now(UTC)`) > 2h → refuse + Telegram
   - Hardened tz-awareness guard so naive vs aware comparisons don't raise. Replaced the `except: pass` wrapper with `except: refuse`.

6. **Non-atomic writes to live config/prod files** (MEDIUM) — multiple sites
   - `with open(path, 'w'): json.dump(...)` creates a truncated file mid-write. Race window (~10-100ms) where live trader's hot-reload could read partial JSON → `JSONDecodeError`.
   - **Fix**: added `_atomic_write_csv()` helper (mirrors existing `_atomic_write_json`). Patched 3 sites:
     - `crypto_trading_system_ed.py:4616` — `crypto_ed_production.csv` write
     - `crypto_trading_system_ed.py:4641` — `regime_config_ed.json` write
     - `crypto_revolut_ed_v2.py:185` — `save_trading_config()` now tempfile + `os.replace`
   - All writes atomic via `os.replace()` (atomic on modern NTFS/POSIX). Readers see OLD or NEW, never half-written.

7. **Silent model-fit exception swallowing** (MEDIUM) — 4 sites
   - `except Exception: continue` in signal-generation model.fit/predict loops hid GPU OOM, scaling errors, all-NaN features. Outer loop showed "no votes" with zero diagnostic.
   - **Fix**: added `_log_fit_exception(context, exc)` helper with session-scoped dedup set — first occurrence of each `(context, ExceptionClass, short_msg)` triple prints once, subsequent repeats stay silent. No log spam, but failure modes surface.
   - Patched: `generate_signals` (ed.py:2324), `_quick_score` (ed.py:2720), `_deku_eval_with_pruning` (ed.py:3699), `generate_live_signal` (live_trader.py:715). Live trader imports helper lazily from main module.

**Audit false positives (flagged by auditor, verified not bugs):**

- **`rs == rs` NaN guard at rally_cooldown simulator**: auditor claimed tautology, but `float('nan') == float('nan')` returns False in Python/NumPy, so `rs == rs` is the standard idiomatic NaN check. Behavior is correct.
- **Signal cache reuse across T↔G iterations**: signals are shield/gate-independent (shield applied AFTER signal generation), so caching across iterations is actually correct.

**Not fixed (intentional):**

- **`except Exception: pass` in Windows priority / orphan worker cleanup**: non-critical setup steps, intentional tolerance.
- **Config cold-start fallbacks to `{}`**: expected behavior when file doesn't exist yet.
- **`os.startfile()` GUI open**: cosmetic, platform-specific.
- **Funding rate load falling back to None**: documented semantic.

**Deployment safety:** all fixes tested with syntax parse + sample invocations. Live trader impact: zero functional change on healthy path. New behavior only kicks in when something is genuinely broken (refuse-to-trade + Telegram alert), which is what we want.

**Re-audit (2026-04-24 late evening) — 2 follow-up bugs from the fixes themselves:**

- **Atomic write tmp-path collision (CRITICAL)** — both `_atomic_write_json` and `_atomic_write_csv` used `path + '.tmp'`. Two concurrent writers (e.g., parallel HRST subprocesses) would collide on the same tmp file; one's content could overwrite the other's before `os.replace`. **Fix**: tmp path now includes PID (`path + f'.{os.getpid()}.tmp'`). Also applied to `crypto_revolut_ed_v2.py:save_trading_config`.

- **Partial data download flag (HIGH)** — `_DATA_DOWNLOADED_THIS_SESSION = True` was set after the download block regardless of whether macro and OHLCV actually succeeded. If one failed silently, subsequent horizons would skip the retry and run on partial/stale data. **Fix**: track `macro_ok` + `ohlcv_ok` flags; only cache-flag when BOTH succeed. Partial success prints a diagnostic and next horizon retries.

Re-audit also verified: all 7 original fixes clean, no regressions, label float/NaN cascade correct, Mode T max_iter still honored under tighter tolerance, feature-refuse paths cleanly return None, regime error propagation complete, staleness check handles all tz permutations, model-fit logging thread-safe under GIL.

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
- **Detector set trimmed 5 → 2 on 2026-04-30** — `ENABLED_DETECTORS = {'tsmom_672h', 'sma24>sma100'}` constant added in [crypto_trading_system_ed_engine.py](crypto_trading_system_ed_engine.py). Cross-run analysis of 35 1440-window HRSTs (Apr 18-30) showed only `tsmom_672h` (66% TOP-15 presence, 9 wins) and `sma24>sma100` (69% presence, 8 wins) consistently appear in TOP 15. The other 3 (`price>sma72`, `sma168>sma480`, `vol_calm`) were single-run wonders or weak (≤54% presence). Mode S joint sweep drops 3,920→1,568 combos (60% less multiple-testing). All 5 detector lambdas still defined in `_build_regime_indicators_and_detectors` so the live trader can evaluate any named detector found in `regime_config_ed.json`; only the SEARCH FILTER is restricted. To re-enable for quarterly detector-rediscovery: edit `ENABLED_DETECTORS` to include the wider set, run HRST, revert.
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
