# Algo Trading Engine

Automated ML trading system for **crypto** (BTC, ETH, XRP, DOGE, SOL, LINK, ADA, AVAX, DOT) and **index CFDs** (DAX, S&P 500). Generates BUY/SELL/HOLD signals using ensemble ML models with walk-forward validation, temporal decay sample weighting, and embargo-corrected labels. Executes trades on Revolut X via Ed25519-signed API.

**Production:**
- **Ed V2** — Regime-switching trading: bull/bear detection with dynamic horizon selection, maker-order pricing (0% fee). Currently ETH-only.

**Testing (no live trader):**
- **Ein V1.0** — 15-minute candles, horizons 4-10 candles (1h-2h30)
- **Eli V1.0** — 30-minute candles, horizons 4-10 candles (2h-5h)

**Archived:** Doohan V1.7.1 (fixed-horizon, retired 2026-04), Deku, CASCA, V5 and earlier — see [archive/](archive/).

---

## 🧪 Currently Running

**AB Matrix 4-variant focus (launched 2026-04-24 22:40 CEST → ETA ~14:00 2026-04-25, ~16h laptop):**

First matrix on methodologically-clean data — ran after today's three-pass audit shipped (label-tail fix, div-by-zero guards, sparse-feature quarantine, atomic writes, file-handle leak fix). Today's 22:07 live HRST (`sma168>sma480 bull=7h@65% bear=6h@85%`) is already promoted and live; this matrix validates the decision and tests meta-labeling on a clean primary.

```bash
python tools/ab_matrix_runner.py --variants focus --skip-vol
```

| Variant | Floor | Trim | Meta | Purpose |
|---|---|---|---|---|
| A_floorON_trimOFF | ON | OFF | — | Floor alone on full universe |
| B_floorON_trimON | ON | ON | — | Replicates today's live HRST (seed-robustness sanity check) |
| C_floorOFF_trimOFF | OFF | OFF | — | Raw universe, no floor guarantees |
| D_floorON_trimON_metaON | ON | ON | p≥0.45 | R3 meta-labeling retest — B↔D isolates meta contribution |

All variants run with `--no-persist --no-data-update --data-dir data_matrix_<ts>/` — live production files + live `data/` never touched. Results land in `output/ab_matrix_results_<timestamp>.csv` + tagged `config/regime_config_ed_noprod_{A,B,C,D}.json` / `models/crypto_ed_production_noprod_{A,B,C,D}.csv`. See [CLAUDE.md](CLAUDE.md#) "CURRENTLY RUNNING" section for decision rules.

---

## Engine Reference — Full Inventory (2026-04-17)

Compiled from 4 parallel audits across `crypto_trading_system_ed.py`, `crypto_revolut_ed_v2.py`, `crypto_live_trader_ed.py`, CLAUDE.md, README.md, `config/regime_config_ed.json`, and all `models/*.csv`.

### 1. ENGINE ARCHITECTURE — MODES

| Mode | What it does | Output |
|---|---|---|
| **P** | PySR symbolic-regression feature discovery. Runs genetic-program symbolic regression offline on the historical window (months 12→6 ago, anti-leakage). Production Mode D auto-loads the JSON as computed feature columns. ~30–120 min per (asset, horizon). | `models/pysr_{asset}_{h}h.json` |
| **D** | Grid optimization. Exhaustive sweep of 3 combos × 6 windows × 6 feature-counts × 3 gammas = 324 configs per horizon. Saves top 6 candidates. The expensive core step. | `models/crypto_hourly_best_models.csv` |
| **V** | Validate + Optuna refine. Live-backtests top 6 D candidates at conf 70/80/90%, picks top 3, runs 50-trial Optuna refine (gamma±0.020, features±5, window±20h), writes winner. **Always prefer V-refined over raw D.** | Production row in `crypto_ed_production.csv` |
| **H** | Horizon sweep — runs D+V for each horizon in the list, compares, picks the best per asset. Typical call: `H BTC 5,6,7,8h`. | Best horizon per asset |
| **R** | Regime-detector backtest — sweeps named detectors (SMA, RSI, vol, tsmom, etc.) × horizon pairs; greedy per-regime horizon selection. Feeds Mode S. | Winner detector+horizons |
| **S** | Strategy joint sweep — full detector × bull_h × bear_h × bull_conf × bear_conf search in one shot. Preferred over R's greedy search when compute allows. | Updates `regime_config_ed.json` |
| **T** | Hold-shield tuning — sweeps `min_sell_pnl_pct × max_hold_hours`. Uses 0% fee assumption (maker-first). | Config update |
| **G** | Rally-cooldown BUY gate — sweeps `(h_short, h_long, t_short, t_long, cd_hours)` over 49,716 configs. STRICT pick: beats_3of3 windows AND plateau_score ≥ 0.7. Current winner: `h_short=8, t_short=3%, h_long=36, t_long=5.5%, cd=30h`. | Config update |

**Chains in use:** DV, HRS, DVRS, HRSTG, RSG. Default `--replay = 1440h` (2 months) when omitted.

**Production dispatch path:** HRSTG = Horizon sweep → Regime → Strategy (detector+conf) → hold-shield Tune → rally-cooldown Gate.

### 2. FEATURES — WHAT'S COMPUTED AND WHAT WINS

130+ feature universe, ~100 computed live, 18–25 actually selected by LGBM importance for ETH production.

**2.1 Technical (51 total, all live)**

| Group | Count | Examples | Selected? |
|---|---|---|---|
| Log returns | 14 | `logret_{1..240}h` | `logret_120h` (75%), `logret_240h`, `logret_72h` |
| Spreads | 6 | `spread_24h_4h`, `spread_120h_12h` | occasional |
| MAs | 4 | `price_to_sma100h`, `sma20_to_sma50h` | `price_to_sma100h` (70%) |
| Momentum | 2 | `rsi_14h`, `stoch_k_14h` | occasional |
| Volatility | 6 + 2 GK | `volatility_{12,48}h`, `atr_pct_14h`, `gk_volatility` | `volatility_48h` (55%) |
| ADX trend | 3 | `adx_14h`, `plus/minus_di_14h` | `adx_14h` (60%) |
| Temporal | 4 | `hour_{sin,cos}`, `dow_{sin,cos}` | `hour_cos` (100% of top models) |
| Derivatives of price | 7 | velocity, accel, jerk | occasional |
| Bollinger/Z | 2 | `bb_position_20h`, `zscore_50h` | rare |
| Volume | 1 | `volume_ratio_h` | rare |

**2.2 Macro (72 total, ~40 live when CSVs present)**

Sources: VIX, DXY, SP500, NASDAQ, GOLD, US10Y, EURUSD, USDJPY, OIL → each with zscore + chg{1,5,10}d + vol{5,20}d, plus 6 VIX-regime flags.

- **Actually winning:** `m_sp500_chg1d` (70%), `m_nasdaq_chg1d` (55%), `m_dxy_chg*`, `m_us10y_chg*` (20–30%)
- **DXY note:** delisted March 2026, code uses `DX-Y.NYB` fallback

**2.3 Cross-asset (16 total, heavy hitters)**

- `xa_dax_relstr5d` (80% of top models) — BTC/ETH relative strength vs European equities
- `xa_sp500_relstr5d` (75%)
- Correlations (10d, 30d) weaker

**2.4 PySR-discovered (5 formulas per horizon, ETH only)**

- Stored per horizon in `models/pysr_ETH_{5,6}h.json`
- `pysr_2` and `pysr_5` selected ~50% of the time; others 20–30%
- Encode nonlinear interactions between momentum + cross-asset + seasonal terms
- Evaluated via SymPy in `_compute_pysr_features()` at `crypto_trading_system_ed.py:1060-1150`

**2.5 Sentiment (16 total, partial)**

- **Fear & Greed:** `fg_chg5d` selected 40% of top models. Works.
- **GDELT:** completely dead. Code exists in `_compute_gdelt_features()` but `data/macro_data/gdelt.csv` is never written or loaded. The HRST BTC crash this morning was mid-GDELT download — reactivating GDELT is a separate project.

**2.6 Fully dead feature groups (code exists, never wired)**

- **On-chain** — CoinMetrics (`active_addresses`, `MVRV`, `SOPR` BTC-only, `hashrate`, `tx_count`, exchange flows) downloaded by `download_macro_data.py`. **Wired into `build_all_features()` 2026-04-17.** First post-wiring HRS audit (2026-04-18): `oc_mvrv_chg1d` earned Grade 4 (4/6 ETH horizons), `oc_exchange_netflow_chg5d` Grade 3. Other 38 derivatives below selection — open question whether weak or correlation-crowded-out. SOPR pending BTC HRST.
- **Derivatives / funding rate** — loaded as `_funding_rate` (underscore = non-feature), used only as a regime gate candidate, never in training.
- **Feature Set B** — defined but `ACTIVE_FEATURE_SET='A'`; production actually uses Set D (dynamic, per-horizon selection).

**2.7 Feature grades 1–5 (from Topic 2 deep-dive)**

Grading basis: frequency across the 48 production models (from `crypto_ed_production.csv`). Grade 5 = picked in ≥50%, Grade 1 = never or <5%.

*Technical (price/volatility/momentum):*

| Feature | What it is | % of models | Grade |
|---|---|---|---|
| `hour_cos` | Cosine of hour-of-day (circadian cycle) | 73% | 5 |
| `price_to_sma100h` | Price relative to 100h SMA (trend strength) | 73% | 5 |
| `logret_120h` | 5-day log return | 69% | 5 |
| `sma20_to_sma50h` | Short-vs-medium MA crossover | 58% | 5 |
| `adx_14h` | Trend strength (ADX) | 50% | 5 |
| `vol_ratio_12_48` | 12h/48h realised-vol ratio | 40% | 4 |
| `logret_240h` | 10-day log return | 38% | 4 |
| `logret_72h` | 3-day log return | 35% | 4 |
| `logret_24h` | 1-day log return | 35% | 4 |
| `volatility_48h` / `gk_volatility_48h` | 48h realised vol (plain and Garman-Klass) | 31% each | 4 |
| `volatility_12h` | 12h realised vol | 31% | 4 |
| `spread_120h_12h` | MA spread, medium-vs-short | 31% | 4 |
| `plus_di_14h` | Upside directional index | 25% | 3 |
| `spread_240h_24h` | MA spread, long-vs-day | 21% | 3 |
| `atr_pct_14h` | 14h ATR % | 19% | 3 |
| `bb_position_20h` | Bollinger position | 19% | 3 |
| `price_accel_24h` | 24h acceleration | 17% | 3 |
| `minus_di_14h` | Downside DI | 17% | 3 |
| `price_to_sma50h` | Price vs 50h SMA | 15% | 3 |
| `zscore_50h` | Z-score on 50h window | 13% | 2 |
| `spread_24h_4h` | Very-short spread | 13% | 2 |
| `logret_6h` / `logret_8h` / `spread_120h_8h` | Short-window variants | 8–10% | 2 |
| `hour_sin`, `stoch_k_14h`, `gk_volatility_14h`, `logret_7h`, `price_accel_12h`, `spread_48h_12h` | Minor | 8% | 2 |
| `rsi_14h`, `logret_{2h,5h,8h,12h,48h}`, `dow_sin`, `intraday_range`, `volume_ratio_h`, `price_accel_4h`, `spread_48h_4h`, `price_to_sma20h` | Rare or never | ≤6% | 1 |

*Macro (external markets):*

| Feature | What it is | % | Grade |
|---|---|---|---|
| `m_nasdaq_chg1d` | NASDAQ 1-day change | 46% | 4 |
| `m_sp500_chg1d` | S&P 1-day change | 40% | 4 |
| `m_vix_chg1d` | VIX 1-day change | 38% | 4 |
| `m_dxy_chg1d` | DXY 1-day change | 15% | 3 |
| `m_gold_chg1d` | Gold 1-day change | 13% | 2 |
| `m_gold_chg10d`, `m_dxy_chg5d`, `m_gold_vol5d`, `m_oil_chg5d`, `m_sp500_chg5d`, `m_us10y_chg5d`, `m_vix_chg5d`, `m_eurusd_*`, `m_vix_vol5d` | Longer-window macro variants | ≤6% | 1 |

**Verdict on macro:** only the 1-day equity/VIX changes earn their keep. Everything longer-window or on FX/commodity sub-dimensions is basically dead weight.

*Cross-asset:*

| Feature | What it is | % | Grade |
|---|---|---|---|
| `xa_dax_relstr5d` | BTC/ETH 5-day relative strength vs DAX | 44% | 4 |
| `xa_eth_usd_relstr5d` | Relative strength vs ETH | 38% | 4 |
| `xa_sp500_relstr5d` | vs S&P | 38% | 4 |
| `xa_nasdaq_relstr5d` | vs NASDAQ | 29% | 3 |
| `xa_dax_corr10d` | 10-day correlation vs DAX | 27% | 3 |
| `xa_sp500_corr10d` | vs S&P | 17% | 3 |
| `xa_eth_usd_corr30d`, `xa_nasdaq_corr10d`, `xa_btc_usd_relstr5d`, `xa_eth_usd_corr10d` | Minor | ≤8% | 1–2 |

**Verdict:** relative-strength (5-day momentum differential) is the signal; plain correlations are weaker.

*Sentiment:*

| Feature | What it is | % | Grade |
|---|---|---|---|
| `fg_chg5d` | Fear & Greed 5-day change | 23% | 3 |
| `fg_chg10d` | 10-day change | 13% | 2 |
| `fg_zscore` | Z-score | 10% | 2 |
| `fg_value`, `fg_chg1d`, `fg_ma5d` | Raw, short-window, smoothed | ≤2% | 1 |
| GDELT (geopolitical tone/volume, 7+ features) | Never loaded into `build_all_features()` | 0% | 1 (DEAD) |

*PySR symbolic features:*

| Feature | % | Grade |
|---|---|---|
| `pysr_5` | 42% | 4 |
| `pysr_4` | 35% | 4 |
| `pysr_2` | 33% | 4 |
| `pysr_3` | 29% | 3 |
| `pysr_1` | 21% | 3 |

**Verdict:** all 5 PySR formulas earn their spot (none below 20%). Genetic programming found real nonlinear combinations.

*On-chain (MVRV, SOPR, hashrate, active addresses, netflow…):*

**Update 2026-04-19:** Loader landed 2026-04-17, first HRS post-wiring 2026-04-18. `oc_mvrv_chg1d` immediately promoted to **Grade 4 (66.7%, 4/6 ETH horizons)** — near the top of the feature table, just behind `hour_cos` (G5) and tied with `pysr_1/2` (G4). `oc_exchange_netflow_chg5d` Grade 3. Other 38 derivatives (hashrate, active_addresses, tx_count, fees) below selection threshold this run — open question whether genuinely weak or correlation-crowded-out (audit pending next Mode D run). BTC SOPR untested until BTC HRST runs.

*Derivatives:*

`funding_rate`: Grade 1 as a feature. Loaded as `_funding_rate` with the underscore deliberately excluding it from the feature matrix. Only used as an optional regime gate in `crypto_live_trader_ed.py`, and even that path isn't active. BTC-only data source; would need per-asset sourcing to generalise.

### 3. MACHINE LEARNING — WHAT TRAINS AND WHAT DOMINATES

**3.1 Current production ensemble combos (3)**

Per `crypto_trading_system_ed.py:3000-3005`:

- RF + LGBM
- XGB + LGBM
- RF + XGB

Binary vote; confidence = `buy_votes / total_votes × 100`.

**3.2 Models tested and dropped — why**

| Model | Status | Why dropped / not used |
|---|---|---|
| **LSTM** (solo + as partner) | Dropped | 0 valid solo results. `LSTM+LGBM ≡ RF+LGBM` — LSTM voted essentially randomly; partner carried all signal. Sequence modeling doesn't add over hand-crafted lags on hourly data. |
| **TabPFN / tabular transformers** | Dropped | Failed — no usable signal. Pretrained tabular transformers don't fit our feature distribution; training from scratch lacks data. |
| **RF+GB, RF+LR, GB+LR** (combos) | Dropped | 0 wins across V1.6–V1.7.1. Removed from `ALL_MODELS`. |
| **Gradient Boosting solo** | Not in `ALL_MODELS` | Dominated by LGBM (same family, LGBM faster + GPU). |
| **SVM** | Not in `ALL_MODELS` | Doesn't scale to 50k rows with 100+ features; kernel tuning unstable. |
| **LR solo** | Dropped post-embargo | Historically won `crypto_hourly_best_models.csv` pre-embargo — turned out to be leakage artifact. Post-embargo: 0 wins. |
| **V1.7.2 regularization sweep** | Wash | Signal/noise too low for fine reg tuning. |
| **Multi-timeframe fusion** | Dropped | Worse than single-TF. |
| **CPCV validation** | Incompatible | Conflicts with temporal-decay (gamma) weighting. |

**3.3 PySR role**

- **For features (P mode):** ACTIVE — per-horizon formulas on ETH, feeding into D/V grid as synthetic features. Working.
- **For regime detection (pysr detector type):** code exists (`crypto_live_trader_ed.py:219-250`) but not used in any asset's live config. Tested 2026-03-29: forward48, sma48_200, forward72 labels — best accuracy 58%, too weak. Hand-crafted `tsmom_672h` won over PySR-discovered regime.

**3.4 Named regime detectors (catalog)**

Defined at `crypto_trading_system_ed.py:4795-4801`. Five survivors after trimming from 16:

- `sma24>sma100`
- `sma168>sma480`
- `price>sma72`
- `vol_calm` (Andersen-Bollerslev deseasonalized 24h vol < 70th pct over 30d)
- `tsmom_672h` — time-series momentum 28d (Liu & Tsyvinski 2021) — **currently active on ETH**

**Dropped detector families:** 5 RSI variants, 4 drawdown variants, MACD>0, 9 redundant SMA/momentum variants. No academic support for RSI/drawdown as regime classifier in crypto literature.

**3.5 Should we test other ML models?**

- **Worth testing: CatBoost.** Only candidate genuinely likely to beat LGBM. Optimized for categorical features, often wins on structured data with minimal tuning. Cheap to drop into `ALL_MODELS` as a 4th combo partner (`RF+CatBoost`, `CatBoost+LGBM`).
- **Maybe worth revisiting: HistGradientBoosting (sklearn).** Similar spirit to LGBM but different split logic; could diversify the tree-based stack.
- **Not worth revisiting:** tabular transformers (FT-Transformer, SAINT) — same scaling problems as TabPFN on our ~50k-row datasets. ExtraTrees — variance-reduction version of RF; RF already in combos.
- **Definitively closed:** LSTM, TabPFN, LR solo, SVM, plain GB solo.

### 4. HORIZONS — 5 / 6 / 7 / 8 h

- Mode H sweeps all four, picks best per asset via `return × (win_rate/100)` score.
- **4h dropped** — all Mode D candidates negative post-embargo fix (CPCV PBO=1.0, overfit).
- **ETH production:** bull=6h@85%, bear=8h@65%. This asymmetry is intentional (see §6).
- **6h vs 7h tradeoff (2026-04-08, partially unresolved):** 7h D#1 XGB+LGBM +24.73% / 64% WR vs 6h Refined#1 +23.17% / 78% WR. Picked 7h initially for raw return; later Mode S joint sweep put bull back to 6h at higher confidence.
- Full 5,6,7,8h HRST BTC run was kicked off this morning but crashed at GDELT step — needs restart.

**4.1 Horizon distribution + deep-dive on 4h / 10h / 12h / 14h (from Topic 4)**

Production CSV distribution:

| Horizon | Model count | Share |
|---|---|---|
| 5h, 6h, 8h | 9 each | 56% combined |
| 7h | 8 | 17% |
| 4h | 2 | 4% |
| 10h | 4 | 8% |
| 12h | 4 | 8% |
| 14h | 2 | 4% |
| 16h | 1 | 2% |

Code reality:

- `crypto_trading_system_ed.py:5428` — Ed default sweep is `[5, 6, 7, 8]`. That's why HRST BTC this morning is testing those four.
- `crypto_trading_system_ed.py:4818` — Mode R/S default test horizons extend to `[4, 5, 6, 7, 8, 10, 12]` when not specified. So 10h and 12h get tested opportunistically; 14h only when explicitly requested.

Per-horizon verdict:

- **4h — UNRELIABLE.** Dropped historically: "All Mode D candidates negative post-embargo" (CLAUDE.md 2026-03-24). The embargo fix killed 4h because label overlap with horizon-4 is severe — there's simply not enough time between training end and evaluation to keep signal fresh. The 2 rows present in production CSV are legacy pre-embargo survivors. **Do not revive.**
- **10h — LIKELY USEFUL, UNDER-TESTED.** Has 4 models in CSV, appears in default Mode R/S sweeps. No dedicated experiment against 5–8h. Worth a pairwise test on a long replay (2880h+) before committing.
- **12h — SAME STATUS AS 10h.** 4 models in CSV, default-swept. Same recommendation: run a head-to-head.
- **14h — BARELY TESTED.** Only 2 models, not in default sweeps. No evidence either way. For the rally-cooldown `h_long` parameter (Mode G), 14 is a tested value — but as a prediction horizon it's essentially untried.

**Recommended experiment:** a dedicated `HRS BTC 5,6,7,8,10,12,14` run on a 2880h window to see if longer horizons dominate the 5–8h band on BTC specifically. BTC tends to trend on longer timescales than ETH — the current 5–8h concentration may be an artefact of ETH-focused tuning bleeding into BTC runs.

### 5. RISK MANAGEMENT & GATES

| Mechanism | State | Evidence | Verdict |
|---|---|---|---|
| Stop-Loss / Take-Profit | OFF | 8 variants tested 2026-04-14 (`backtest_sl_variants.py`). Baseline (no SL) won: +1.11% PnL / −8.71% DD. Profit-lock and trailing all −11% to −20%. Disaster −7%/−10% never fired. | Keep OFF. Signal edge > risk mitigation. |
| V7 Rally-Cooldown BUY Gate | ON | 2026-04-18 HRSTG rerun (integrated T→G): winner `rr10h ≥ 2.5% OR rr30h ≥ 6.0%, cd=20h`. H1 +31.01% / H2 +22.22% / 60d +60.12% / worst DD +4.55% / plateau 1.00. Ranked by `--rank recent` (H1-focused, default). | Keep ON. |
| Adaptive cooldown lift | Rejected | 0 triggers on 30d, −8.8% to −23.7% on 90d vs fixed 30h | Dropped. |
| Hold-shield (`min_sell_pnl` + `max_hold`) | **ON bull / OFF bear** (2026-04-18) | ETH: 0.6% / 12h shared, bull_shield=ON, bear_shield=OFF. Per-regime Mode T sweep (4 on/off combos) picked this config with +23.45pp over single-flag both-ON baseline. | Keep. Shield on bull rides dips; off in bear cuts fast. |
| Regime filter (bull/bear) | ON | Mode R/S validated `tsmom_672h` for ETH (2026-04-18 HRSTG confirmed) | Keep, regime-conditional params. |
| Min-confidence thresholds | ON | Per-regime: bull 95% / bear 80% (updated 2026-04-18 HRSTG). Global fallback 75. | Keep, per regime. |
| Dynamic confidence raises in bear | Rejected (2026-03-29) | All variants lost money — blocked winning contrarian trades | Dropped. |
| Maker orders (`bid+0.01`, `post_only`) | ON | 4 bugs fixed. 180s/10s window tuned 2026-04-15. | Keep. |
| NTP clock sync | ON | Fixed 2026-04-13 after echo-back bug. Startup + every 5min + on 409. | Keep. |
| Trailing stops | Rejected (2026-03-29) | Baseline signal exits beat all variants on 336h | Dropped. |
| Blow-off filter (6 families × 4 actions) | Experimental only | Best variant +0.58pp — "not actionable" | Not wired. |
| Momentum-decay signals (5 signals) | Experimental only | Commit `bfdd115` "replaces blowoff in bot" but not live | Not wired. |

### 6. REGIME-CONDITIONAL ASYMMETRY (critical for future work)

Current ETH asymmetry (2026-04-18 HRSTG):

| | Bull | Bear |
|---|---|---|
| Horizon | 6h (short) | 8h (longer) |
| Min confidence | 95% | 80% |
| Max position | $12k | $12k |
| Hold-shield | **ON** | **OFF** |

Pattern observed across backtests: bull reliably 6h; bear drifts 7h–8h. Rationale: bear needs longer horizon to capture reversals. Confidence thresholds tightened 2026-04-18 (was 85/65 → 95/80) by the current HRSTG Mode S joint sweep.

**Confidence threshold per regime — DO NOT unify.** 2026-03-29 test: raising bear confidence across board blocked winning contrarian trades. Per-regime thresholds are load-bearing — today's S picked *higher* bear confidence (80 vs 65) but it's still below bull, so the principle holds. If live performance degrades, revert bear to 65%.

**Hold-shield regime split (2026-04-18):** Per-regime Mode T sweep found bull=ON / bear=OFF beats single-flag both-ON by +23.45pp (+115.50% vs +92.05% over 60d). Mechanism: bull regime rewards riding through dips (shield defers SELL on small losses); bear regime punishes holding losers (bear rallies are mean-reversion setups, cut fast).

**Rally-cooldown regime-conditionality: partial.** Gate still operates globally on BUYs (one set of params), but Mode T→G integration now evaluates the gate against a per-regime-tagged signal stream, so the chosen gate reflects the combined bull+bear behavior under the active shield policy. Regime-split gate (separate params per regime) remains untested.

**6.1 What doesn't work — deeper (from Topic 6)**

Going beyond the "dropped" one-liners in Agent 4's output to surface the lessons:

*Trend-following risk rules fight the model.* Stop-loss / take-profit (all 8 variants), trailing stops (0.25–1%), profit-lock (+0.5/+0.22) — every one lost vs. the no-SL baseline:

- Profit-lock variants: −11% to −20% PnL on 30-day window. Root cause: scalping sub-0.3% winners by locking in micro-gains surrenders the fat tail of 2–5% winners that drive P&L. Meanwhile, hold-shield already caps losses at ~2–3% on disaster paths, so the "loss mitigation" benefit of SL is zero.
- Trailing stops: similar. Baseline signal exits beat all variants for both BTC (+$826) and ETH (+$207) on 336h.
- **Lesson for future:** the model's signal quality IS the risk edge. Any rule that overrides SELL timing will chop winners. Only exception: a disaster brake at −5% to −7% that never fires in normal operation — free insurance, but no backtest uplift since it never triggers.

*Regime confidence asymmetry cannot be inverted.* Raising `min_confidence` in bear was tested (2026-03-29). Every variant lost money because it blocked the contrarian bear-rally trades that are the bear book's only positive-expectancy setups. The current bull=85% / bear=65% split is counterintuitive but battle-tested — higher confidence in the calm regime, lower in the volatile regime (precisely because in bear the low-confidence signals contain more actionable mean-reversion).

*Ensemble size has sharply diminishing returns.*

- LSTM + LGBM = RF + LGBM (identical results). LSTM voted randomly; partner carried all the signal. **Lesson:** marginal model additions don't help once you have one strong tree-based learner in the stack.
- Multi-model ensembles (RF+GB+LGBM, RF+GB+LR+LGBM) rarely win in production CSVs — single LR or LGBM often wins outright. **Lesson:** ensemble-averaging dilutes a strong base model when the added models are weaker. The "committee of experts" intuition is wrong here.
- V1.7.2 regularization added `ra`/`rl` params; was a wash. **Lesson:** the signal-to-noise ratio at this timescale doesn't support fine regularization tuning — either it matters catastrophically (embargo) or it matters imperceptibly.

*PySR for regime labels (as opposed to features) fails.* Tested forward48, sma48_200, forward72 regime labels. Best accuracy 58% — too weak. PySR's strength is finding compact algebraic interactions of continuous inputs. Binary regime classification is not that problem. **Lesson:** keep PySR for feature synthesis (where it works: `pysr_2/4/5` at 33–42% selection); use explicit named detectors (`tsmom_672h`, `sma168>sma480`) for regime binary classification.

*4h horizon is structurally broken.* Post-embargo, nothing survives. Label overlap dominates. **Lesson:** horizon must exceed embargo window with enough margin for the prediction target to be genuinely future-unknown. 4h is at the edge and doesn't clear it.

*Adaptive rules beat fixed rules only when they have enough data.* Adaptive rally-cooldown lift (early removal on reversion) — tested on 90d: lost −8.8% to −23.7% vs fixed 30h. **Lesson:** added parameters need enough events to learn from. 90 days contained too few rally-then-reversion cases to estimate the reversion threshold reliably. A fixed-duration rule with one tuned parameter beat an adaptive rule with three.

*Blow-off filters didn't beat the no-filter baseline.* 6 filter families × 4 actions tested. Best improvement: +0.58pp (+10.32% baseline → +10.91% with filter). Not actionable. The V7 rally-cooldown gate, which does something similar (block BUYs after recent strength), did win — because its trigger (`rr_8h ≥ 3%`) is a softer, earlier, more probabilistic condition than RSI>70 or %B>1.0. **Lesson:** blow-off tops are hard to identify from price alone because distribution cutoffs (like RSI>70) fire after the move is already done. Momentum acceleration features (`rr_8h` + `rr_36h`) catch it earlier.

### 7. WHAT'S DROPPED / ARCHIVED (WHOLE VERSIONS)

- **BTC trading** — disabled 2026-04-06 (45% WR, avg loss > avg win on 1-month OOS). Position liquidated.
- **CASCA, Deku, Doohan V1.1–V1.7** — all archived when embargo fix (2026-03-24) revealed pre-embargo APFs were 5–26× inflated. V1.7.1 is the last surviving member.
- **4h horizon** — all negative post-embargo.
- **V1.7.2 regularization sweep** — wash, V1.7.1 unregularized more stable.
- **Multi-timeframe fusion** — worse than single-TF.
- **CPCV validation** — incompatible with temporal decay (gamma).
- **SV3 horizon sweep (partial)** — superseded by Mode S joint sweep in production.
- **`/optimize` + `/optstatus` in trader bot** — removed 2026-04-16; optimizer lives in its own bot.

### 8. DEAD / UNVERIFIED / WORTH REVISITING

| Item | State | Opportunity |
|---|---|---|
| On-chain features (MVRV, SOPR, hashrate, tx_count, exchange flows) | **Wired 2026-04-17.** MVRV Grade 4 (66.7%), netflow_5d Grade 3. Others below threshold. | Audit whether unselected ones are weak or crowded out (TODO 1a). SOPR untested until BTC HRST. |
| GDELT sentiment | Code exists, loader missing | Sourcing GDELT cleanly (rate limits, cache) is the work. |
| Funding rate as feature | Loaded, never trained | Experimental feature in next Mode D sweep. |
| Feature Set B (macro-heavy) | Defined, unused | Could outperform Set A in macro-driven regimes. |
| PySR for regime (not features) | Tested, rejected 58% acc | Probably not worth revisiting until new labels. |
| Other assets (BTC, SOL, ADA, XRP, DOGE, LINK, AVAX, DOT) | Production rows exist, disabled | Re-enable after BTC 45%-WR issue solved — may need per-asset PySR discovery. |
| Signal nondeterminism | Demoted 2026-04-17 | Only actionable if reproduced on pinned code. |

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
python crypto_trading_system_ed.py HRST BTC 5,6,7,8h       # Full pipeline + T (shield + gate)
python crypto_trading_system_ed.py F                       # Mode F — feature trim (audit + disable Grade-1)
python crypto_trading_system_ed.py F --restore             # Empty the disabled list
python crypto_trading_system_ed.py --help                  # Show all modes
```

**Research flags (preserve production):**
```bash
python crypto_trading_system_ed.py HRST ETH 5,6,7,8h --replay 1440 --no-persist
# --no-persist: redirect ALL writes (production CSV + regime config) to *_noprod.*
# Safe to run alongside live trader. Use for validation/sensitivity tests.

python crypto_trading_system_ed.py DV ETH 5h --label-threshold 0.01
# --label-threshold X: retrain with label=1 iff future_return > X (default 2×fee = 0.0022)
# Output redirected to models/crypto_ed_production_lt<pct>.csv.
```

**Mode T convergence (upgraded 2026-04-22):**
Mode T iterates shield↔gate optimization up to `--max-iter N` (default 6, was 4). Convergence semantics:
- **Strict**: consecutive iterations produce identical config → `Converged`
- **Tolerant**: numerics within plateau noise (min_sell_pnl ±0.10pp, max_hold ±2h, gate windows ±4h, thresholds ±0.5pp, cd ±6h) → `Converged-in-tolerance`
- **2-cycle**: iter N close-in-tolerance to iter N−2 but not N−1 → `2-CYCLE DETECTED` (structural instability, **should not be promoted live**)
- Otherwise → `Reached max_iter without convergence` (also unstable)

Promotion gate: only "Converged" or "Converged-in-tolerance" should be promoted to production.

**Verify Mode T logic:**
```bash
python tools/test_mode_t_convergence.py     # 8 unit tests + retrospective analysis
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

## Features (132 → 191 with 2026-04-19/20 additions)

| Category | Count | Examples |
|----------|-------|---------|
| **Technical** | 51 | Log returns (1-240h), RSI, Bollinger Bands, ATR, ADX/DI, Garman-Klass vol, volatility ratios, Stochastic, spread ratios, SMA ratios, hour sin/cos |
| **Macro** | 40 | VIX (level, zscore, regime), DXY, S&P500/Nasdaq changes (1/5/10d), US10Y, EUR/USD, USD/JPY, Oil, Gold volatility |
| **Sentiment** | 25 | Fear & Greed Index (value, zscore, changes, MA, extreme flags) |
| **Cross-asset (daily)** | 16 | BTC/ETH/DAX/Nasdaq/S&P500 rolling correlation (10/30d), relative strength (5d) |
| **Cross-asset (intraday)** ⏱ | 3-6 | BTC 1/2/3h lagged returns (all assets); ETH 1/2/3h lags (alts only). Added 2026-04-20. |
| **On-chain** | ~40 per asset | CoinMetrics: MVRV, active addresses, fees, tx count, hashrate, exchange flows. BTC+ETH+XRP+LINK; SOL blocked (CoinMetrics 403). |
| **Derivatives** | 9 | Funding rate + chg1d + zscore; OI chg1d/3d/zscore; perp-spot basis + chg1d + zscore. From Binance. |
| **Stablecoin** | 3 | USDT+USDC market cap chg1d/chg7d/zscore (CoinGecko). Mostly Grade 1 dead. |
| **Options IV** | 2 | ATM IV + 25-delta skew (Deribit). Grade 1 so far (recent addition). |
| **Orderbook** | 2 | Hourly snapshot: imbalance + spread_bps. ETH-only accumulation. |
| **PySR symbolic** | 5 per (asset, horizon) | Auto-loaded from `models/pysr_{ASSET}_{H}h.json`. Grade 4-5 when selected. |

**Feature trimming (`config/disabled_features.json`)** — added 2026-04-21. Any feature listed in `disabled_exact` or matching `disabled_prefixes` is stripped from the LGBM input matrix at the end of `build_all_features()`. Data download + column computation still run — only the model's input set is trimmed. Default list is populated via `python crypto_trading_system_ed.py F` (Mode F), which audits 33 production models + 49 PySR JSONs and disables features with 0 selections AND 0 PySR references (Grade 1). 22 "newborn" features (added 2026-04-19/20) are spared by default; drop them from `NEWBORN_FEATURES` in code after a fair HRST cycle. Revert via `F --restore`.

**Audit helpers:**
- `python tools/audit_features.py --export models/feature_audit.csv` — grade every feature 1-5 by selection rate + PySR references
- `python tools/check_pysr_consistency.py` — 5 checks verifying Mode P JSONs + HRST config + production CSV are consistent (no silent drift)

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

**Main `/optimize` menu** (since 2026-04-21):
- 🔧 Full Re-tune (HRST) — chains H → R → S → T
- 🔄 Regime Refresh (RS)
- ⚡ Model Refresh (DV)
- 🔬 PySR Discovery
- 🧹 Feature Trim (F) — audit + populate `disabled_features.json`; short-circuits past asset/horizon pickers
- Advanced ▸ — full mode grid (D, V, H, R, S, T, G, HRS, RS, DVRS, etc.)

**🧪 Switch to NO-PROD toggle** on the confirm screen: redirects the job's writes to `*_noprod.*` files so production stays untouched. Job labels show `[NO-PROD]` tag. Use for validation runs (e.g., `HRST ETH 5,6,7,8h --replay 1440 --no-persist`).

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

### PRIORITY 1 — Strategy Tests (2026-04-20)

1. **Vol-scaled horizons deeper validation** — `vol_2band (low→8h, high→6h)` beat tsmom regime by +5.02pp on 2-month ETH replay (+33.82% vs +28.80%, 65% WR). Picks horizon by 24h realized vol percentile rank: >70th pctile = high vol = use faster 6h, otherwise use 8h for more confirmation. Next steps: (a) validate on 4-month window, (b) test combining with tsmom detector, (c) decide whether to replace tsmom_672h. Script: `tools/test_vol_scaled_horizon.py`.

2. **Triple barrier label test** — Replace fee-aware labels (`future_return > 2×fee`) with volatility-adaptive triple barrier. The label duration adapts to volatility: calm market waits longer for a meaningful move, volatile market resolves faster. Tested in 2025 pre-embargo on BTC with old combos (RF+GB+LR, +29%) — never tested with Ed's current pipeline (RF+LGBM, embargo, PySR, on-chain). Effort: medium (change label creation, run Mode D ETH 6,8h).

3. **Meta-labeling test** — Train a secondary model that predicts whether the primary model's BUY/SELL will be correct. Primary says BUY → meta-model says "this BUY has 80% chance of winning" → skip low-quality signals. Separates direction from quality. Lopez de Prado Ch. 3. Effort: medium-high (new training pipeline). Never tested.

4. **Re-run Mode D ETH 6,8h with new derivative features** — Funding rate + OI ranked #2-5 in LGBM importance but Mode D grid failed initially (OI NaN caused dropna row loss: 1440→898 rows). Fixed: sparse features excluded from dropna (LGBM handles NaN natively). Need a clean re-run to produce actual production model candidates that include funding rate features.

### PRIORITY 2 — Running Jobs

5. **SOL HRST** — Launched 2026-04-19 evening on Desktop. Review criteria: pipeline health (4/4 horizons?), 60d alpha (need ≥20pp), ETH correlation, on-chain feature usage.

6. **LINK P+HRST** — Running on Laptop. PySR regen (March expressions don't include on-chain) then full pipeline.

### PRIORITY 3 — Research

7. **Feature trimming** — 127/206 features (62%) never selected across 33 production models. Only 24 are Grade 4+ (selected ≥30% of the time). GDELT already disabled (21 dead features + slow download). Remaining trim targets: 37 dead macro variants (keep only `_chg1d` for VIX/SP500/NASDAQ/DXY/Gold), 38 dead on-chain derivatives (keep MVRV + netflow only), 11 dead sentiment, 4 dead cross-asset corr30d. Saves ~10-15% compute per Mode D.

8. **Orderbook + IV accumulation** — Hourly snapshots now wired into Ed trader. Need ~2 weeks of data to evaluate as features. Binance doesn't provide historical orderbook depth; Deribit free API is snapshot-only. CryptoDataDownload has free DVOL history (ATM IV) as a stopgap.

9. **Eli HRS BTC** — 30-minute candle system test. 10. **Ein results review** — 15-minute candle system.

### Completed (2026-04-19)

- **11 new feature sources added** — ETH derivatives (funding rate + open interest from Binance, previously BTC-only), stablecoin market cap flows (CoinGecko USDT+USDC), orderbook imbalance snapshots (Binance depth), options IV skew snapshots (Deribit), exchange netflow 1d/3d windows, perp-spot basis check. Whale Alert + Kalshi stubs built (need API keys). Total features: 152 → 185 (after GDELT removal). Derivative features ranked #2-5 in LGBM importance on ETH; stablecoin features ranked below 1% (dead).
- **GDELT disabled** — 21 features, 0% selection rate across 33 models, 5-10 min rate-limited download per Mode D. Removed from both download pipeline and feature engineering. Code kept commented.
- **Freshness check for macro downloads** — `_is_fresh()` helper skips re-downloading if CSV modified within 1 hour. Saves 5-10 min per Mode D run. IV and orderbook snapshots always append (accumulate over time).
- **Sparse NaN handling fixed** — `dropna()` now only drops on core features, not sparse ones (OI, orderbook, IV, stablecoins). LGBM handles NaN natively via missing-value splits. Previously OI's 30-day-only history caused 1440→898 usable rows.
- **Hourly snapshot accumulation** — Ed trader (`crypto_revolut_ed_v2.py`) now calls `download_orderbook_snapshot()` and `download_options_iv_skew()` at the end of each hourly cycle. Silent on failure. Data accumulates in `orderbook_snapshots.csv` and `options_iv_snapshot.csv`.
- **Full feature audit** — 206 features counted and ranked across all 33 production models (ETH/BTC/SOL/LINK/XRP). 82 ever selected (40%), 127 never selected (62%). Grade 5 (≥50%): `price_to_sma100h` (70%), `hour_cos` (67%), `logret_120h` (64%), `m_sp500_chg1d` (52%). Per-asset top-10 documented. Full ranking available in conversation history.
- **5 strategy improvement ideas tested:**
  - **#3 Asymmetric loss** — DEAD. `scale_pos_weight` [0.3-1.5] made zero difference at 95%/85% confidence. High confidence threshold already filters false BUYs.
  - **#4 Signal staleness** — DEAD. Consecutive BUY filter (skip after 1-8 consecutive hours). Best: +1.14pp on 6h (skip-after-3). Not actionable; rally-cooldown serves the same anti-churn function better.
  - **#5 Vol-scaled horizons** — **PROMISING.** `vol_2band (low→8h, high→6h) @90%` = +33.82%, 46 trades, 65% WR. Beats tsmom_672h regime (+28.80%) by +5.02pp. Beats every single-horizon baseline. Needs validation on longer windows.
  - **Kelly criterion** — Not tested (architectural mismatch with binary all-in/all-out positions).
  - **#1 Triple barrier + #2 Meta-labeling** — Queued for 2026-04-20.

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
