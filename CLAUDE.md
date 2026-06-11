# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Automated ML trading system for crypto. **5 active assets**: ETH (live since 2026-05-18 H75 promotion), BTC/SOL/LINK/XRP (standby, `enabled: false` — HRST done, results not promoted). BNB code-wired 2026-05-02 but PySR/HRST not yet run. DOGE/ADA/AVAX/DOT pruned 2026-04-19 (weak priors, no diversification). Generates hourly BUY/SELL/HOLD signals using ensemble ML models with walk-forward validation, temporal decay sample weighting, and embargo-corrected labels. Variable horizon per asset (5h, 6h, 7h, 8h — optimized via Mode H).

**Production:**
- **Ed V2** — Regime-switching trading (`crypto_trading_system_ed.py` + `crypto_revolut_ed_v2.py`). Dynamic bull/bear horizon selection via external config (`config/regime_config_ed.json`). Maker-order pricing at `bid+0.01` with `post_only` for 0% fees. Mode R regime backtest. Currently ETH-only.

Doohan V1.7.1, Deku, CASCA and all prior versions archived (Doohan retired 2026-04, others 2026-03-24). See `archive/`.

**Owner:** Alex (CET/CEST timezone)

---

## Engine Reference Card (built 2026-04-17, kept current — H75 live state inline)

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

Production CSV (2026-05-18, post-H75 promote): 4h = 5, 5h = 6, 6h = 6, 7h = 6, 8h = 6, 10h = 4, 12h = 4, 14h = 2, 16h = 1. (Per-coin: ETH covers 4-12h; BTC covers 5-16h; LINK covers 4-14h; SOL/XRP/BNB cover the core 4-8h band.)

- **5h, 6h, 7h, 8h** — core band, default sweep in Ed ([line 5428](crypto_trading_system_ed.py#L5428)). **Current live (G_narrow models on H75 engine, promoted 2026-05-21 21:56 CEST): ETH detector=`sma24>sma100`, bull=5h@65% RF+LGBM w=281 γ=0.9981, bear=8h@65% RF+LGBM w=293 γ=0.9990**, shields OFF (both regimes), rally cooldown ON (both regimes — reverted 2026-05-27 13:18 to Mode T optimal recommendation after May 23 22:21 manual-OFF intermezzo): bull rr8h≥2.0 OR rr14h≥6.0 cd=6h; bear rr10h≥5.5 OR rr12h≥2.0 cd=8h, min_sell_pnl=0%, max_hold=10h. **Macro cache mtime fix patched 2026-05-27 11:22** (TODO 0527 — root cause of live-vs-backtest gap). Prior live (2026-05-18 → 2026-05-21): H75-fresh, bull=5h@75% XGB+LGBM w=100 γ=0.9993 / bear=8h@65% RF+LGBM w=162 γ=0.9954 on same detector, gates ENABLED. Earlier (2026-04-30 → 2026-05-18): bull=6h@85% / bear=5h@65% on `named:price>sma72`.
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

- **All stop-loss / take-profit / profit-lock / trailing-stop variants.** 8+ variants tested; baseline (no SL) won every dimension. Scalping sub-0.3% winners surrenders the fat tail; hold-shield already caps losses. **Only exception worth considering: −5% to −7% disaster brake as free insurance.** **Intra-hour TP also tested 2026-06-10** (`tools/bt_intrahour_tp_8h.py`, fills mid-candle when bar high wicks to target — TP's fairest shot at catching reverting spikes): every level loses monotonically (TP1.5% −21.3pp ... TP7% −1.3pp), avg-win collapses 1.24%→0.93% as it tightens, converges to baseline as it loosens. An intra-hour TP only ever subtracts (caps a winner), never adds. Closes the "hourly-resolution" caveat — sub-hourly TP loses too.
- **Raising bear min-confidence.** Blocks contrarian bear-rally trades — every variant lost. Current bull=85%/bear=65% is correct.
- **LSTM as ensemble partner.** Voted randomly; RF+LGBM ≡ LSTM+LGBM. Single strong tree-based learner dominates.
- **Multi-view feature split (different features per model).** Tested 2026-06-10 (`tools/bt_multiview_features_8h.py`, ETH 8h, 2mo). Partition universe into FAST (technical/pysr/deriv) vs SLOW (macro/xa/onchain/sentiment), LGBM-solo each, naive-average blend. Both gate conditions FAILED: SLOW anti-predictive standalone (AUC 0.435 < 0.5 — macro/onchain has no standalone signal on ETH 8h), and FAST/SLOW **error-corr 0.728** despite prediction-corr 0.155 (decorrelated predictions but same hard bars wrong → no complementary signal for a meta-learner to exploit). Sim: FULL +32.8% vs FAST −8.8% / SLOW −19.9% / BLEND −13.3%. Slow features are *conditioning variables, not predictors* — they only pay inside a model that also sees price context (prod's importance-selected 24 already mixes `stable_mcap_chg7d` with deriv/pysr/technical). **Extends "ensemble dilutes a strong base" from algorithms to features: one model + full universe + LGBM importance beats partitioning by ~45pp. Don't split features across models.**
- **Triples / quads / quintuple ensembles, and any LR combo.** Exhaustive 26-combo sweep 2026-06-10 (`tools/bt_all_combos_8h.py`, ETH 8h, 2mo, gated sim): **RF+LGBM wins outright +53.6%** (best at all 3 windows, 77% WR / 71 trades). No triple/quad/quintuple beats it — best non-pair RF+GB+XGB+LGBM +46.0%; quintuple +25.1%. LR poisons every combo it enters (all bottom-half; XGB+LR +8.3% worst). RF+LGBM adds ~21pp over LGBM-solo (+32.8%) — the one combination that earns its keep. Cross-checks: LGBM-solo +32.8% = multiview FULL (same path reproduced); GB+LGBM +45.9% < RF+LGBM (reproduces Jun 9 GB+LGBM rejection). **2 models is the sweet spot; RF+LGBM is the optimal combo — combination topology is not a lever.**
- **Sub-hourly resolution / "nervous" features for reactivity.** Diagnosed 2026-06-10 (`tools/diag_fast_spikes_resolution.py`, ETH 1-min, 60d, model-free). The hourly clock already reaches **76% of slow-move upside** (1-2h sustained moves, where the money is). Genuine sub-15-min spikes exist but are small (median 1.3%) and rare (big ≥2% fast spikes: 13 in 60d), usually mean-revert, and even 15m bars only reach 37% of the big ones — so sub-hourly wouldn't fix them and chasing them = scalping (already proven to lose). **Resolution is NOT the bottleneck; reactivity is not a data/feature problem.** Note: short-window technicals (logret_2-8h, price_accel_4h, rsi_14h) are already Grade-1 dead — the model ignores nervous features at hourly res by design. Open (not yet run): model entry-latency on sustained moves it DOES trade — the only reactivity angle at the right altitude.
- **Order-flow / intra-hour "nervosity" features (taker imbalance, intra-hour rvol/runup from 1m).** Tested 2026-06-10 (`tools/mock_nervosity_8h.py`, ETH 8h, Mar-May, forensic + mock). Phase 1: of 118 fwd8h≥2% up-moves the baseline caught 44 / missed 74 (37% capture). On CAUGHT moves taker-imbalance was bullish (+0.044, z+0.38); on MISSED moves it was neutral (−0.003, z+0.03 ≈ baseline) — **the misses had no order-flow signature to detect.** Phase 2: adding the 4 nv feats HURT the gated sim (+41.7% → +34.3%, −7.4pp), recovered only 4/74 misses, and LGBM weighted them ~12% (noise it mistook for signal → more, worse BUYs). **Order flow is not a lever; the 37% capture is the model correctly declining unpredictable moves (the misses are mostly right skips). eth_1m_data.csv has taker_buy_base_volume if ever revisited, but mechanism is dead.**
- **Entry-capture mechanisms for missed up-moves (lower gate / relaxed vote / proba threshold).** Forensic 2026-06-10 (`tools/forensic_misses_capture_8h.py`, ETH 8h, recent 2mo). Of 74 fwd8h≥2% up-moves, baseline caught 24 / missed 50. Miss decomposition: **35 WRONG** (both models predicted DOWN, med proba 0.187 — confidently wrong), 12 SPLIT (med 0.627), only **3 GATE-BLOCKED** (conf 63 vs gate 65). 70% of misses are moves the model bet against — uncapturable by any entry rule. Capture sweep: every method that recovers misses LOSES return (gate55 −9.5pp/+3, gate50 −10.5pp/+3, relaxed_vote −2.9pp/+3, proba≥0.55 −13.7pp/+13 but WR 77%→65%, 109 trades). **The misses are correct skips; the model's selectivity (77% WR) IS the edge. Reconfirms gate is not too tight (only 3 blocked, 2pts under).**
- **Faster (5h/4h) variants of the live technical features.** Mock-tested 2026-06-10 (`tools/mock_fast_features_8h.py`, ETH 8h, 2mo). Built causal 5h/4h analogues of the live 24 (kama_5, adx_5h, plus_di_5h, vol_of_vol_4h, vol_ratio_4_8, price_to_sma5h, spread_8h_4h + existing price_accel_4h/logret_5h); causality proven via truncation-invariance (0 leak). LGBM DETECTED 6/9 (adx_5h ranked **#4 of 33**, 4.0% — above prod-median 3.3%, ~tied with adx_14h), BUT the gated backtest CRATERED: **+53.6% → +31.6% (−22pp)**, WR 77→72, trades 71→82. **Textbook importance≠performance / over-fit-the-noise ([[feedback_feature_addition_should_not_harm]]): faster windows give LGBM spurious in-sample split gain that doesn't generalize.** Lesson: never judge a feature by importance — only the gated sim. Reconfirms reactivity is not a lever. **6h variant retested 2026-06-10 (`--fast-window 6`): same pattern, even worse — 6/9 detected (adx_6h #9) but −27.5pp (+53.6%→+26.1%). THREE windows tested (3h/4-5h/6h) → same outcome; it's the mechanism (overfit faster-window noise), not the window choice. 3h −23.6pp, 4/5h −22.0pp, 6h −27.5pp; all 6/9 detected. AIRTIGHT: 3h's price_to_sma3h ranked **#2 of 33 by importance** and STILL cost 24pp. Model's top importance stays on the SLOW features (price_accel_24h, price_to_sma100h, vol_of_vol_24h). Importance is NOT a safe guide — only the gated sim is.**
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
- **Volatility-scaled horizons (C01).** Originally promising on 2-month replay 2026-04-19: `vol_2band low→8h high→6h @90%` = +33.82% / 46 trades / 65% WR, +5.02pp over `tsmom_672h` baseline. **4-month confirmation FAILED to replicate.** Best 4mo variant `vol_2band low→8h high→6h @90%` = +30.98% vs current baseline `tsmom_672h bull=6h bear=8h @90% +38.49%` = **−7.51pp**. A different best variant on 4mo (`vol_median below→8h above→6h @90%` +41.48%) only beats baseline by +2.99pp — below the +5pp promotion threshold. Verdict: 2mo win was window-shopping, doesn't generalize. **Vol-scaled horizons family closed.** Additional caveat: the 2mo +5.02pp gain was measured against the now-defunct `tsmom_672h` baseline used pre-H75-promote; with live now on `sma24>sma100` (H75) the baseline has shifted further. SHELVED.

### What's promising (tested, pending decision)

- **ETH derivatives as features (funding rate + open interest).** Added 2026-04-19 — extended Binance derivatives download to ETH (was BTC-only). LGBM importance ranking: `deriv_funding_chg1d` ranked **#4 on 6h (4.2%)** and **#2 on 8h (3.7%)**. `deriv_oi_chg3d` ranked **#5 on 6h (3.3%)**, `deriv_oi_chg1d` ranked **#4 on 8h (3.6%)**. These are top-tier — higher than established features like `adx_14h` and `price_to_sma100h`. Mode D grid produced 0 valid candidates initially due to OI NaN (30-day history only); fixed by excluding sparse features from `dropna()` (LGBM handles NaN natively). **Next step: re-run Mode D or HRST with the NaN fix to get actual production candidates.**

### What's untested (queued)

- **Per-regime feature set.** Open question from engine reference: should bear use more macro features, bull more technical? The asymmetry already happens organically (6h model: 7 features, mostly PySR+technical; 8h model: 32 features, kitchen sink) but hasn't been deliberately tested as a design choice. **Effort: low-medium (code change to Mode D).**
- **Multi-horizon ensemble emergency-exit** (4th angle from 2026-04-27 forensic). Force exit when 5h AND 8h both flip SELL within 1h. Per-horizon signal cache `data/eth_per_horizon_signals_90d.pkl` already exists. Distinct from rejected 5-min price-action triggers and from T1b entry-side ensemble (which was tested + shelved 2026-04-27). **Effort: 1-2h.**

**Recently moved out of "untested" (now tested + shelved):**
- ~~Meta-labeling~~ — tested 2026-04-27 in AB matrix variants D and E. R3 RESOLVED: E (meta on strong base) lost −2.12pp vs A; D (meta on weak base) gained +6.26pp but still lost to A overall. Door open for SOL/BTC/XRP retests but those assets are also shelved.
- ~~Triple barrier as exit overlay (T6)~~ — tested 2026-04-26: +10.48pp 60d / +1.24pp 90d / 47 trades. Not promoted (60d gain didn't survive to 90d).
- ~~Triple barrier as TRAINING LABEL~~ — tested 2026-03-14 on BTC (`archive/testing_literature.csv`). Direct same-day baseline comparison: baseline 4h +57.22% / 8h +74.03% vs triple_barrier_label 4h +29.22% / 8h +22.53%. Lost on every dimension (return −28pp / −51pp, WR −9pp / −10pp, accuracy −16pp / −13pp). Earlier framing of "+29% standalone, never tested with current pipeline" was misleading — the 2026-03-14 test had a same-day baseline that was 30-50pp better. SHELVED.

### Regime-conditional asymmetries (important for future models)

- **ETH horizon asymmetry:** bull=5h / bear=8h (since H75 2026-05-18; was 6h/8h pre-H75). Longer horizon in bear = more confirmation needed in volatile regime.
- **ETH confidence asymmetry:** bull=75% / bear=65% (since H75 2026-05-18; was 85%/65% pre-H75). Counterintuitive but battle-tested — bear's low-confidence signals are mean-reversion setups that *should* fire. H75 lowered bull threshold 85%→75% because strict-(combo,w) dedup yielded a more conservative 5h winner that benefits from a lower confidence gate.
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

**One shared engine folder** synced via Google Drive — all three machines use the same code, data, and models. Only the venv is local per machine.

| Machine | Engine Path | Venv | Python |
|---------|-------------|------|--------|
| Desktop (primary) | `G:\engine\` (Google Drive synced) | `C:\algo_trading\venv\` | `C:\algo_trading\venv\Scripts\python.exe` |
| Laptop | `G:\Autres ordinateurs\My laptop\engine\` (Google Drive synced) | `C:\Users\Alex\algo_trading\venv\` | `C:\Users\Alex\algo_trading\venv\Scripts\python.exe` |
| Yoga | `G:\Autres ordinateurs\Yoga\engine\` (Google Drive synced) — *path TBC, user to confirm* | *TBD — user to fill in* | *TBD — user to fill in* |

- **Desktop:** i7-14700KF, RTX 4080, 32GB — used for long Mode D runs
- **Laptop:** 16 cores, RTX 3070 Ti
- **Yoga:** CPU-only (no GPU) — used for lightweight tests / PySR workers (2 workers per per-machine policy); engine auto-detects `LGBM_DEVICE=cpu` on Yoga, `gpu` on Desktop/Laptop
- **GitHub:** https://github.com/Spyku/Algo_trading
- **Push:** `git_push.bat` from `G:\engine\`
- **OS:** Windows 11, Python 3.14 venv (NOT conda)
- **GPU:** LGBM uses GPU on Desktop/Laptop (`device='gpu'`); Yoga falls back to CPU. Configured in `hardware_config.py`.

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
python crypto_trading_system_ed.py P BTC 6h                    # Mode P — PySR feature discovery (single horizon ~40 min sequential, ~15 min parallel-P; see Runtime Reference)
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
python tools/backtest_regime_master.py                          # 2-month default, all horizons
python tools/backtest_regime_master.py --months 4               # 4-month backtest
python tools/backtest_regime_master.py --horizons 6,8           # only test 6h and 8h (fast)
python tools/backtest_regime_master.py --bull 6 --bear 8        # fix pair, compare regimes only
python tools/backtest_regime_master.py --regimes sma,rsi        # filter regime families
python tools/backtest_regime_master.py --no-combos              # single-horizon baselines only
python tools/backtest_regime_master.py --asset ETH              # test other assets
```

**Telegram commands (Ed V2 trader):** `/stop` `/status` `/pause` `/resume` `/balance` `/sync` `/sanity` (on-demand live-correctness check — shadow + snapshot replay, runs in a background thread; `/sanity full` adds the ~15-min informational parity; replaces the old 🔄 Reload button on the main keyboard) `/conf` `/config` `/setup` `/help` `/chart BTC` `/regime` (show current bull/bear state per asset) `/gate [ASSET on\|off\|clear]` (V7 rally-cooldown gate)

**Telegram commands (Optimizer bot):** `/optimize` (interactive menu) `/queue` `/cancel` `/status` `/results` `/help` `/stop`

---

## Runtime Reference (measured, not estimated — 2026-05-06 audit)

All numbers below come from actual log timestamps on Desktop (post-2026-05-02 parallel-merged engine). When quoting an ETA in a TODO entry, **cross-reference this table** instead of guessing. If a phase isn't in this table, say "not measured" rather than estimating.

### Top-level chains (ETH, --replay 1440)

| Workload | Sample N | Range | Median |
|---|---|---|---|
| **HRST (4 horizons + R+S+T+G)** Desktop | 3 runs | **5h 45min – 9h 07min** | **~7h** |
| **HRST 4mo (--replay 2880)** Desktop | 1 run (run #1) | 7h 07min | n/a (single sample) |
| **Mode H only (4 horizons, --replay 1440)** | 3 runs | **258 – 428 min** | ~316 min |
| **R + S + T + G (after Mode H)** | 3 runs | **82 – 114 min** | ~93 min |

### Per-horizon (inside Mode H, --replay 1440)

| Phase | Sample N | Range | Median |
|---|---|---|---|
| **Mode D grid (72 evals, serial outer + threaded LGBM)** | ~18 | **6.9 – 13.2 min** | ~11 min |
| **Mode V refine (hybrid GPU+CPU)** | ~20 | **10.3 – 21.0 min** | ~14 min |
| Mode V Step 1 (6 D-candidates, parallel) | embedded | not separately exposed | — |
| Mode V Step 3 (3 refined, parallel) | embedded | not separately exposed | — |
| **Mode V total per horizon (D→Step1→refine→Step3→summary)** | inferred | **~50 – 95 min** | ~70 min |

### After Mode H

| Phase | Sample N | Range |
|---|---|---|
| Mode R regime backtest | implicit | ~30 min |
| **Mode S signal cache** (4 horizons parallel, 4 workers) | 3 | **22.6 – 33.8 min** |
| Mode S joint sweep (3920 combos) | 3 | ~3-5 min after cache |
| **Mode T signal cache** (2 horizons parallel, 2 workers) | 3 | **15.2 – 21.5 min** |
| Mode T↔G iterations | 5 runs | typically **2 iterations**, sometimes 3 |
| Rally-cooldown sweep (49,716 configs) | per sweep | **42 – 116 sec** |

### Mode P (PySR feature discovery)

| Workload | Sample N | Wall time |
|---|---|---|
| Mode P single-asset 5 horizons, sequential (Apr 25) | 1 | **3h 22min** (BTC, 202 min) |
| Mode P single-asset 5 horizons, parallel-P 3 workers (Laptop) | 1 | **~1h 30min** (BNB, per CLAUDE.md May 2) |
| Mode P single-asset, single horizon | not measured | (estimate ~40 min sequential, ~15 min parallel-P) |

### 14-ideas harness (single idea, ETH 5,6,7,8h --replay 1440)

| Sample | Wall time |
|---|---|
| har_rv (single-run, May 3) | 31 min |
| per_regime_features (chained) | 32 min |
| stability_strict (chained) | 31 min |
| har_rv (chained, May 4) | 30 min |
| hurst_feature (chained) | 30 min |
| shap_ranking (single-run) | 33 min |
| **Range** | **30 – 33 min** (very tight on Desktop) |

### Variance drivers

1. **Optuna refine convergence** dominates per-horizon variance (10–21 min per refine block).
2. **Mode T↔G iteration count** — 2 vs 3 iterations adds ~30 min to Mode T total.
3. **Modern Standby on Laptop** can pause work without crashing (logs go silent); Desktop unaffected (`SetThreadExecutionState(ES_AWAYMODE_REQUIRED)` keeps it awake).
4. **Concurrent /reload or other downloads** can compete for disk I/O during Mode S signal-cache build.

### What this table replaces

Estimates I previously quoted as "~6h", "~3.5h", "~30 min Mode D × 4", "~30-120 min Mode P single-horizon" — those were guesses. Use the measured ranges above. If an estimate isn't in the table, label it **"not measured"** rather than fabricate.

---

## Architecture

### Production File Chain

```
crypto_trading_system_ed.py  (Ed V1.0 — Modes P/D/V/H/S/R/HRS/HRST/T/G/F)
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
- **Refined-only production selection:** Mode V backtests D candidates to pick top 3 for Optuna refine (75 trials each since H75, was 50 before 2026-05-18), then selects production model from refined configs only. D candidates consistently underperform refined versions.
- **MIN_COMBO_SIZE=2:** Solo models removed. Prevents overconfidence from uncalibrated single-model predictions.
- **MIN_TRADES=8:** Optuna objective returns 0 for trials with <8 trades.
- **Models:** RF, GB, XGB, LR, LGBM — 3 viable combos: XGB+LGBM, RF+LGBM, RF+XGB (dead combos RF+GB, RF+LR, GB+LR dropped). **GB+LGBM** trialed in FAYE 2026-06-08/09 and **rejected**: won the ungated bear-context screen but LOST the real-engine 2mo regime backtest to RF+LGBM (+43.5% vs +51.2%) — regularization that "wins" an ungated screen erases the model's conviction (decisive 0/1 probas + cash calls) under the live conf gate, so it holds long through drawdowns. **Validate combo/hyperparam ideas through the real gated engine** (`tools/bt_lgbm_tune_8h.py` + `tools/diag_lgbm_proba_spread.py`), not the screen; LGBM hyperparams are injectable via the default-safe `LGBM_MIN_CHILD`/`LGBM_REG_LAMBDA`/`LGBM_LR`/`LGBM_MAX_DEPTH` env hook (`_lgbm_hyperparam_overrides`, unset = prod-identical).
- **Features:** 51 technical + 81 macro/sentiment/cross-asset + PySR symbolic = 132+ total. LGBM importance ranking (~5 sec). PySR features auto-loaded from `models/pysr_{ASSET}_{H}h.json` if available; safe fallback if not.
- **PySR symbolic regression:** Mode P runs offline discovery (`pysr_discover_features.py`), saves expressions to JSON. Production loads them as computed columns. Anti-leakage: PySR formulas are discovered on months 12→6 ago only, never overlapping with Mode D's last-6-month evaluation window.
- **PySR anti-leakage checks:** `_check_pysr_leakage()` runs early in Mode D, V, and Refine. If PySR JSON lacks `discovery_method == "historical"`, all PySR features are stripped before the run starts. Mode V also blocks production CSV writes for leaky PySR configs.
- **Model hot-reload:** Trader checks production CSV every 5 minutes.
- **K=5 multi-seed denoising (since H75 2026-05-18):** Every Optuna trial in Mode V refine is averaged over 5 random seeds (42, 43, 44, 45, 46) before being scored. Reduces single-seed run-to-run variance from ~5-10pp to ~2-3pp. Wraps `_deku_eval_with_pruning`; controlled by `RELIABILITY_K` env var (default 5). Cost: 5× per trial; offset by REFINE_TRIALS bump 50→75.
- **Strict `(combo, w)` diversity dedup (since H75 2026-05-18):** Mode V's `_diversity_key()` returns `(model_combo, window)` — at most 1 V2 refine slot per (combo, window) cluster. Pre-H75 allowed multiple seeds of the same (combo, window) pair into the top-6, which biased refine toward "lottery" configs. Strict dedup forces V2 to explore distinct (combo, window) buckets — yields more conservative/robust winners.
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
config/private.pem                     <- Ed***** signing key
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
REFINE_TRIALS = 75                                  # Optuna refine trials per config (bumped 50→75 with H75 promotion 2026-05-18)
REFINE_TOP_N = 3                                    # top N D candidates to refine
MODE_G_REPLAY_HOURS = 336                           # 2-week backtest window
MODE_G_CONF_THRESHOLDS = [65, 70, 75, 80, 85, 90]  # confidence thresholds to test
EMBARGO_CANDLES = horizon                           # label overlap fix (dynamic per horizon)
```

---

## Version Files

| File | Status | Notes |
|------|--------|-------|
| `crypto_trading_system_ed.py` | **Production** | Single-file Ed V1.0 engine. **Merged 2026-05-02**: previously split into entry-point + `_engine.py` + `_parallel_p.py` (parallelism wrapper) + `_wrapper`; all merged back into one file. **H75 promotion 2026-05-18 added K=5 multi-seed denoising + REFINE_TRIALS=75 + strict `(combo, w)` diversity dedup** (pre-H75 snapshot moved 2026-05-28 → `ARCHIVED/2026-05-28_v3_cleanup/crypto_trading_system_ed_pre_H75_20260518.py`). **macro_cache mtime fix 2026-05-27** at lines 1077-1110 (pre-fix snapshot 2026-05-30 → `ARCHIVED/2026-05-30_post_faye_cleanup/variants/crypto_trading_system_ed_pre_macro_cache_fix_20260527_112231.py`). Modes P/D/V/H/S/R/HRS/HRST/T/G/F/RST. Embargo-fixed grid (2×3×4×3=72 evals after trim), 75-trial Optuna refine, PySR symbolic features. Built-in parallel paths for Modes V/S/T (loky workers, LGBM cpu inside parallel sections, hybrid GPU+CPU refine dispatcher). Reads/writes `crypto_ed_production.csv` and `regime_config_ed.json`. Pre-merge snapshots in `archive/crypto_trading_system_ed_engine_pre_merge_20260502.py` + `_parallel_p_pre_merge_20260502.py` + `_wrapper_pre_merge_20260502.py`. **Successor: `crypto_trading_system_faye.py`** (built 2026-05-30, ZERO monkey-patches, see row below) — replaces this file + the entire v3 chain once promoted. |
| `crypto_trading_system_faye.py` | **Next-gen consolidation** (built 2026-05-30 — NOT YET IN PRODUCTION) | **Single-file 9100-line consolidation** of Ed v3 (`_g_narrow_d_parallel_nearlive_v3.py`) and its 3 monkey-patched parents (parallel_nearlive + step6_nearlive + g_narrow_d + ed). **ZERO monkey-patches** — every previously-patched feature is now first-class native code. NEAR_LIVE_MODE defaults built in (`step=1`, `signal_mode='ternary'`, `na_policy='mean_last_10'`, `embargo=horizon`, `return_probas=True`). K=5 multi-seed median ensemble native (seeds=[42..46], `K5_SEEDS` module const). 8-worker Mode D `ProcessPool` grid dispatcher native (`_faye_grid_worker` + 60-config upfront submit, per-eval CSV with K=5 seed-by-seed breakdown). 3-worker hybrid GPU+CPU refine native (`_refine_top_configs` canonical, `_refine_top_configs_serial` is ≤1-config fallback). Isolated outputs: `models_faye/crypto_faye_production.csv`, `config_faye/regime_config_faye.json`, `models_faye/mode_d_full_*.csv`. CLI identical to v3 (`python crypto_trading_system_faye.py HRST ETH 5,6,7,8h --replay 1440`). Same `RELIABILITY_K=5`, `MODE_D_OUTER_WORKERS=8`, `FAYE_MODELS_DIR`, `FAYE_CONFIG_DIR` env overrides. **Smoke test** (`tools/smoke_test_faye.py` — 38 ✓ on green): verifies module-level state, canonical entry points, worker pickling, and explicitly that all 17 old monkey-patch names (`_h_factories_seeded`, `_H_K`, `_H_SEEDS`, `_ENG_*_ORIG`, `*_parallel`, `*_hybrid`) are gone. **Engine-vs-trader parity test** (2026-05-30, `tools/validate_core_against_signal_log.py --samples 30 --recent-only --cpu-lgbm`): 30/30 hours evaluated, 0 errors, 22/30 direct MATCH (73.3%), **0 real BUY↔SELL flips** out of 8 DIFFs — all DIFFs are HOLD-threshold boundary cases. **Phases**: P1 paths `8c122ef`, P2 NEAR_LIVE defaults `0883ea4`, P3 native K=5 `ff63d30`, P4 3-worker refine `c878832`, P5a mode V/S/T rename `d7f7744`, P5b 8-worker dispatcher `bb0c6fe`, P6 smoke test `694d85f`, P7 docstring `4ab34d5`. **Promotion plan**: copy `models_faye/crypto_faye_production.csv` → `models/crypto_ed_production.csv` and `config_faye/regime_config_faye.json` → `config/regime_config_ed.json` AFTER FAYE HRST validates AND trader is flat (Critical Rule 19). Trader (`crypto_revolut_ed_v2.py`) is unchanged — picks up new winners on next hourly cycle via the existing `compute_signal_core()` codepath. |
| `crypto_trading_system_ed_noprod.py` | **Archived 2026-05-30** | Moved to `ARCHIVED/2026-05-30_post_faye_cleanup/variants/`. Was a safety wrapper for research runs (set `MODELS_CSV_OVERRIDE` env, monkey-patched `PRODUCTION_CSV` + `REGIME_CONFIG_PATH` to `*_noprod.*` paths). Superseded by engine's built-in `--no-persist` flag and FAYE's isolated `models_faye/` / `config_faye/` paths. |
| `crypto_trading_system_ed_cdar.py` | **Archived 2026-05-30** | Moved to `ARCHIVED/2026-05-30_post_faye_cleanup/variants/`. Was a CDaR-aware scoring research fork (Idea #6 from 20-ideas roadmap) — never promoted, only doc/history references survived. |
| `crypto_trading_system_ed_cvar.py`, `_cpcv.py`, `_robust.py`, `_h_strict_family.py` | **Archived 2026-05-30** | Moved to `ARCHIVED/2026-05-30_post_faye_cleanup/variants/`. CVaR / CPCV / robust / H_STRICT_FAMILY experimental forks. H_STRICT_FAMILY's K=5 + `(combo, w)` dedup bits survived as native code in the production engine since 2026-05-18 H75 promotion. |
| `crypto_trading_system_ed_g_narrow_d_parallel_nearlive.py` | **ARCHIVED 2026-06-05** (→ `ARCHIVED/2026-06-05_v3_chain_retired/`; was Research fork v2) | Builds on parallel.py (now archived) with NEAR_LIVE_MODE semantics for honest live-equivalent backtest. Auto-sets `NEAR_LIVE_MODE=1`, `G_NARROW_MODELS_DIR=models_g_desktop_nearlive`, `G_NARROW_CONFIG_DIR=config_g_desktop_nearlive`, plus 4-layer warning suppression (`-W ignore` flag + `PYTHONWARNINGS=ignore` + `warnings.filterwarnings('ignore')` + monkey-patch `warnings.warn=no-op`) via `os.execv` re-exec block. Imports `crypto_trading_system_ed_step6_nearlive` which adds `na_policy=mean_last_10`, `step=1`, `signal=ternary` (BUY/SELL/HOLD), `embargo=horizon` to the K=5 wrap. Patches `G._G_ORIG_DEKU_EVAL → step6_nearlive._H_ORIG_DEKU_EVAL` so K=5 ThreadPool calls go through near-live semantics. **Critical**: Mode D OUTER loop stays serial in this v2 — only K=5 inner is parallel (5 OS threads, ~11% CPU on 26-core Desktop with NEAR_LIVE step=1). Refine: 3-worker outer × K=5 inner = 15 threads, ~60% CPU. Output dirs isolated from production. |
| `crypto_trading_system_ed_g_narrow_d_parallel_nearlive_v3.py` | **ARCHIVED 2026-06-05** (→ `ARCHIVED/2026-06-05_v3_chain_retired/`; was Research fork v3, with `_g_narrow_d`, `_step6`, `_step6_nearlive`, `crypto_signal_core_nearlive`, `smoke_test_v3_worker_guard`) | Builds on v2 with TRUE Mode D outer-loop parallelization via **dispatcher pattern** (no 545-line engine function replication). Monkey-patches `ENGINE._get_deku_diagnostic_models` to capture grid-setup locals via `inspect.currentframe().f_back` (features_np, labels_np, closes_np, ranked_features, n, horizon, asset_name), then replaces `ENGINE._deku_eval_with_pruning` with a routing dispatcher. On first Mode D call: enumerates all 60 configs (in engine iteration order), submits to `ProcessPoolExecutor(max_workers=8)` via env-tunable `MODE_D_OUTER_WORKERS`. Each subsequent call blocks on its specific Future. Engine's serial for-loop still runs but every call hits a parallel pre-dispatched cache. **8 outer × K=5 inner ThreadPool = 40 concurrent LGBM fits** (was 5 in v2). Worker function pickles args (numpy arrays as bytes + dtype/shape), re-imports parallel_nearlive in worker process to inherit its patches, runs K=5 ThreadPool internally — quality preserved. Other call sites (Mode V step 1/2/3, holdout folds, refine) fall back to v2's K=5 wrap unchanged. Monitor thread prints EVERY eval (not just NEW BEST) + writes full per-eval CSV `models_g_desktop_nearlive/mode_d_full_{asset}_{horizon}_{ts}.csv` with K=5 seed-by-seed metrics. NO grid reorder, NO early-kill (both rejected by user — quality concerns). Expected ~5-6× Mode D speedup, ~3-4× full HRST. Commit `f688e0e`. |
| `crypto_trading_system_ed_g_narrow_d_parallel.py` | **ARCHIVED 2026-05-28** | Moved to `ARCHIVED/2026-05-28_v3_cleanup/`. Superseded by parallel_nearlive (v2) then v3 above. Was the original 2026-05-22 parallel benchmark fork with 4 fix categories landed 2026-05-24 (FIX #0 grid+output propagation, makedirs loop, LGBM warnings filter, ProcessPoolExecutor worker re-import). Kept for git history only. |
| `crypto_revolut_ed_v2.py` | **Live** | Ed V2 auto-trader — maker orders (0% fee) with market fallback. Penny-improvement pricing: buy/sell at `bid+0.01`. `post_only` ensures maker. Stale order cleanup, NTP clock sync, locked funds detection. Reads `regime_config_ed.json`. Auto-promote-when-flat staging (`8454d00`). **2026-06-05: `/status` balance summary no longer lists non-enabled coins (was showing phantom BTC).** |
| `crypto_live_trader_ed.py` | **Live** | Ed signal generation — regime-aware. `detect_regime()` + `generate_regime_signal()`. Reports current market price (not label-shifted). **Fix #2 (2026-06-03, `28645bd`): infers on the last CLOSED hourly bar, not the forming candle (`USE_CLOSED_BAR_FOR_INFERENCE`, lines 706-727).** Daily-merge lag mirrored onto serving path (`a7cb7c9`). |
| `crypto_live_shadow.py` | **Live (monitor)** | Real-time shadow check — runs `compute_signal_core()` in parallel with `generate_live_signal()` each cycle, logs both to `config/shadow_signal_diff.csv`. Primary live-correctness monitor (tripwire <99%). **Must mirror the live inference path — see Critical Rule 23.** Closed-bar step-back added 2026-06-05 (mirror of fix #2; line 220) after it silently fell out of sync Jun 3-4 (100%→48%). Shadow errors are swallowed — never affects trading. |
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

## Current Regime Config — see TODO.md `📌 LIVE STATE` for current

The current live config lives in [TODO.md](TODO.md) `📌 LIVE STATE` section. This block intentionally does NOT duplicate it because live config changes more often than this stable-reference file — keeping the source of truth in one place (TODO.md) prevents drift. **Always trust TODO.md over the quick summary below if they disagree.**

Quick summary of current live (FAYE models, promoted 2026-05-31 14:22 CEST; bear swapped 2026-06-02): detector `tsmom_672h`, **bull=6h@65%** RF+LGBM w=150 γ=0.996 15f, **bear=5h@80%** RF+LGBM w=150 γ=0.999 15f (changed 2026-06-02 from 8h@65%), shields OFF (both regimes), rally cooldown ON (both regimes: bull rr8h≥2.0 OR rr14h≥6.0 cd=6h; bear rr10h≥5.5 OR rr12h≥2.0 cd=8h), min_sell_pnl=0%, max_hold=10h, max_position_usd=$14,300. Live inference still runs through `compute_signal_core()` in `crypto_trading_system_ed.py` (macro_cache mtime fix 2026-05-27 + daily-merge lag mirrored to serving path 2026-06-04 commit `a7cb7c9`); FAYE generated the models only. Per-asset enabled state: ETH live, BTC/SOL/LINK `enabled: false`, BNB code-wired since 2026-05-02 (PySR/HRST not yet run), XRP removed from trader data pipeline 2026-05-23.

### Prior production states (historical audit trail)

- **2026-05-21 → present**: G_narrow models on H75 engine (current) — see [TODO.md](TODO.md) for full LIVE STATE block. Macro cache mtime fix patched 2026-05-27 11:22 (TODO 0527).
- **2026-05-18 → 2026-05-21**: H75-fresh — bull=5h@75% XGB+LGBM w=100 γ=0.9993 / bear=8h@65% RF+LGBM w=162 γ=0.9954 on `sma24>sma100`, gates ENABLED (bull rr8h≥2.0 OR rr14h≥6.0 cd=6h; bear rr10h≥5.5 OR rr12h≥2.0 cd=8h).
- **2026-04-30 → 2026-05-18**: bull=6h@85% / bear=5h@65% on `named:price>sma72`.
- **2026-04-26 → 2026-04-30**: AB matrix variant A (`A_floorON_trimOFF`) promoted from `regime_config_ed_pre_A_20260426.json` — bull=6h@80% / bear=7h@85% on `named:price>sma72`, shields ON, gates rr20≥4.0% OR rr24≥4.5% cd=12h (bull) / rr30≥9.0% OR rr36≥9.0% cd=48h (bear), min_sell_pnl=0.5%, max_hold=10h. 60d sim Mode T REF +66.79%.
- **2026-04-18 → 2026-04-26**: HRSTG refresh; bull=tsmom_672h 95% / bear 80%.
- **2026-04-07 → 2026-04-18**: `sma168>sma480` bull=7h@75% / bear=8h@85%. Mode S +60.72%.

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

18. **DEFAULT TO BARE `python tools/<script>.py [args]` FOR USER-FACING PYTHON COMMANDS.** The user runs in PowerShell 5.1 with the venv already activated (prompt shows `(venv) PS C:\...>`). Always emit the simplest form:
    ```
    python tools/<script>.py [args]
    ```
    Do NOT emit the absolute venv path (`"C:/algo_trading/venv/Scripts/python.exe" ...`) by default — the venv is activated, so `python` resolves to the venv's interpreter. Quoted paths cause PowerShell parse errors (PowerShell needs the `&` call operator before a quoted path, which adds friction the user has explicitly rejected). Only emit a venv-explicit path if the user requests it or if the venv is provably not active. Internal `Bash` tool calls I make for myself can use any form they need; this rule is about what I tell the user to type. **User explicitly requested this default 2026-05-03 — do not revert.**

19. **Verify trader is flat (`state=cash` in `position_ed_v2_<ASSET>.json`) before any config / production CSV swap.** Promoting new winners while the trader is mid-position can cause: (a) the trader's regime cache loads stale model parameters mid-trade; (b) the SELL hour's signal generation uses the new model but the BUY hour's `entry_price` was from the old model → bookkeeping inconsistency; (c) rollback requires re-flatting before reverting. **The user can override per-promotion** (e.g. if the new config is strictly more conservative and the trade is already losing). **Default rule**: verify flat first, ask user if not. This is enforced via the [[feedback-flat-before-promotion]] memory and was the operational rule used during the H75 promotion 2026-05-18 (waited for the 22:00 SELL to fire before the 22:02 config swap).

20. **`_load_macro_csv` MUST be mtime-aware (fixed 2026-05-27 — TODO 0527).** The cache stores `(mtime, df)` tuples — re-reads the file when its mtime advances. The pre-fix code stored just `df` and short-circuited unconditionally, freezing macro/cross-asset/sentiment/onchain features at trader-startup values for the entire process lifetime. With cache stale, chg1d/chg5d features collapsed to ~0 (both lookup days ffill to the same row), killing 11+ of the model's highest-importance features (`m_vix_chg1d`, `m_sp500_chg1d`, `oc_mvrv_chg1d`, `xa_dax_relstr5d`, etc.). **If you ever add another module-level cache that holds file content, it MUST include an mtime check.** Verified clean caches: `_regime_config_cache` ([crypto_live_trader_ed.py:155-170](crypto_live_trader_ed.py#L155)). Banned pattern: `if filename in cache: return cache[filename]` without mtime comparison.

21. **`signal_log.csv` schema is `h_1/sig_1/conf_1/h_2/sig_2/conf_2` (since 2026-05-27 — TODO 0527).** Slot 1 = bull-regime model output (lower horizon); slot 2 = bear-regime model output (higher horizon). The previous `sig_4h`/`sig_8h` schema was tied to hardcoded `HORIZON_SHORT=4`/`HORIZON_LONG=8` constants and broke whenever the actual model horizon differed (bull=5h under H75/G_narrow left sig_4h permanently empty for 79/136 recent rows). The slot assignment in `_log_signal()` is regime-anchored: bull cycle populates sig_1 only, bear cycle populates sig_2 only; multi-horizon fallback uses sort-order. **Always look up via the slot, not via `HORIZON_SHORT/LONG`.** Same anti-pattern was found and fixed in 3 other places: `sig_short/sig_long` Telegram exposure (line 1693), asset preflight check (line 4574), gamma fallback (line 2134). Pre-rename CSV preserved at `config/signal_log_backup_pre_rename_20260527_092500.csv`.

22. **`download_macro_data.py` overwrites MUST use `_dedup_preserve_history` or `_merge_preserve_history` (data drift fix 2026-05-26 + completed 2026-05-27).** Historical rows are preserved with `keep='first'`; current-period rows can update with `keep='last'`. Confirmed-safe call sites: macro_daily (`_yf_merge_with_existing`), cross_asset (`_yf_merge_with_existing`), onchain, derivatives, stablecoin, IV snapshot, orderbook, and (added 2026-05-27) fear_greed. **Any new file download path MUST go through one of these helpers** — bare `to_csv` after pulling full history would re-introduce the drift bug where upstream API revisions silently overwrite originally-observed values.

23. **Any change to the live inference path MUST be mirrored in the shadow monitor (`crypto_live_shadow.py`) — they are two separate codepaths (fixed 2026-06-05).** The shadow recomputation in `crypto_live_shadow.py` deliberately *replicates* `generate_live_signal()`'s data prep line-by-line so it can compare in real time. When `28645bd` fix #2 added the closed-bar step-back to the LIVE path (`crypto_live_trader_ed.py:706-727`, `USE_CLOSED_BAR_FOR_INFERENCE`) but NOT to the shadow, the shadow kept inferring on the *forming* bar while live used the *closed* bar → 1-hour offset → shadow match crashed 100%→48% (Jun 3). Proven 23/23: every mismatch had the shadow inferring exactly +1h ahead. **This bug is invisible offline** (historical bars are all closed; the forming-vs-closed split only exists live) → only caught by the real-time shadow + a trader restart. **Whenever you touch inference-row selection, embargo, NaN policy, feature ffill, or train-window edge in the live path, make the identical change in `crypto_live_shadow.py` (mirror of lines 706-727 currently lives at `crypto_live_shadow.py:220`).** The shadow is the primary live-correctness monitor; a stale shadow silently tanks the match rate without any trading impact (live trading uses `sig`, not the shadow).

24. **Sanity / engine-vs-trader validation MUST run on GPU to match the live trader, NOT `--cpu-lgbm` (decided 2026-06-05).** The live trader's LGBM factory is `device='gpu'`. GPU LightGBM was proven **cross-machine deterministic** (Laptop RTX 3070 Ti == Desktop RTX 4080, bit-identical `RF=0.51463784 LGBM=0.05573897` via `tools/check_proba_repro.py`), so GPU is both portable and faithful to live. Running the validator on GPU lifted engine-vs-trader parity from 90% (CPU) to **96.7%** and collapsed the systematic conf bias from −5.71 to −0.93. `--cpu-lgbm` is a no-GPU fallback only, with a documented ~3-5pt gap. `tools/sanity_check.py` still carries `--cpu-lgbm` (line ~60) — drop it where a GPU is present. The irreducible residual (~3%) is training-window data revision on the freshest hours (deriv/on-chain backfill), NOT a leak and NOT fixable by replay — only point-in-time data snapshots would close it.

---

## Adding a New Data Source — Checklist (built 2026-06-01)

**Before wiring any new feature/data source into the pipeline, work through this. Every item is a bug we actually hit** (the 2026-06-01 daily-availability-lag flip + the 0526/0527 data-drift/cache work). Canonical builder: `crypto_trading_system_faye.py::build_all_features`. A pointer comment sits at the merge block in code so you see this when you're there.

1. **Cadence — is it 1/day or 24/day?** Check the source's timestamp column.
   - **Daily** (one row per `date`) → merge on `df['_merge_date']` → inherits the availability lag (step 2).
   - **Hourly / intraday snapshot** (one row per `datetime`/hour) → merge on `_merge_dt` (floored hour) → **NO lag** (point-in-time).
   - Wrong key = wrong lag. Daily-on-`_merge_dt` leaks the future; hourly-on-`_merge_date` over-lags.

2. **Availability lag (daily sources only) — when does day D actually publish?** Daily external feeds publish *after* the day they describe; a same-date merge stamps day-D's value onto day-D's early hours that weren't live yet → backtest clairvoyance (the 2026-06-01 `oc_mvrv_chg1d` BUY→SELL flip).
   - Merging on `df['_merge_date']` already applies `DAILY_MERGE_LAG_DAYS = 1` — most daily families are covered.
   - If the source publishes *later* than ~start of the next day, give it a DEEPER lag via its OWN merge key (like on-chain's `ONCHAIN_MERGE_LAG_DAYS = 2`, because CoinMetrics lands ~midday D+1). **Test:** a 07:00 decision must only see days fully published by 07:00.
   - Trader-captured / real-time / hourly data needs NO lag — it's born at the decision moment (orderbook/IV).

3. **Overwrite safety (Critical Rule 22) — never clobber history.** Any download that re-pulls + writes MUST go through `_dedup_preserve_history` / `_merge_preserve_history` (`keep='first'` historical, `keep='last'` current period). Bare `to_csv` after a full-history pull re-introduces drift (upstream revisions silently overwrite originally-observed values).

4. **Cache mtime (Critical Rule 20) — if you cache file content, check mtime.** Any module-level cache holding file data MUST re-read when the file's mtime advances. Banned: `if fn in cache: return cache[fn]` with no mtime comparison (this froze macro features at trader startup — TODO 0527).

5. **Sparse history — quarantine until ≥60 days; know if it's backfillable.** New sources start thin. If <60 days, add the features to `always_disabled_exact` and schedule a re-enable A/B at `start_date + 60 days` (e.g. deriv_oi → 2026-06-25, orderbook/IV → 2026-06-18). LGBM handles NaN natively — never force a sparse column into `dropna()`. **Backfillable?** API daily feeds can fetch history (instant coverage); **forward-only** trader-captured snapshots (orderbook/IV) are ephemeral → coverage only grows ~24 rows/day and the 60-day wait is unavoidable.

6. **PySR inheritance — automatic, but only if the lag is at the merge.** PySR is computed AFTER the merge (`_compute_pysr_features` runs on already-merged columns), so a new daily feature's lag is inherited for free (a price-only PySR simply won't shift — fine). To have PySR actually USE the new feature, re-run Mode P (now builds via FAYE's lagged builder → writes `models_faye/pysr_*.json`, isolated from live `models/`).

7. **Builder + live consistency.** The merge lives in BOTH `crypto_trading_system_faye.py` (modeling) and `crypto_trading_system_ed.py` (live inference). Keep them identical — or know the deliberate gap (FAYE lagged; ed.py mirror deferred to promotion — see the `project_daily_data_lag_fix` memory). Diverging builders = live ≠ backtest.

8. **Verify — run the per-column lag audit before promotion.** `python tools/audit_feature_lag.py` builds with the lag on vs off and flags any column whose detected shift ≠ expected (hourly → 0h, daily → 24h, on-chain-class → 48h). Expect **0 violations**. A violation means the merge key is wrong.

9. **No hardcoded horizons/slots (Critical Rule 21).** If the source feeds a signal-log slot or horizon-keyed lookup, address by slot, never by `HORIZON_SHORT/LONG`.

---

## Pending Work

Active TODOs and in-flight runs live in [TODO.md](TODO.md). Historical audit trail (canonical ideas scoreboard C01-C82, MERGED TOPICS, closed/DEAD/SHELVED entries) lives in [ARCHIVED_LOG.md](ARCHIVED_LOG.md). Split out of CLAUDE.md on 2026-05-18 (CLAUDE.md/TODO.md) then 2026-05-19 (TODO.md/ARCHIVED_LOG.md) to keep this file focused on stable reference material (engine card, machine setup, commands, architecture, critical rules).

**🔔 Whenever the user asks to "open CLAUDE.md" / "show CLAUDE.md" / "review my CLAUDE" or any equivalent phrasing, automatically read [TODO.md](TODO.md) as well and surface the ⚡ ACTIVE block + any in-flight job status in the response. The user shouldn't have to ask for the TODO separately — opening CLAUDE.md is implicitly asking for the current state of work.** User explicitly requested this default 2026-05-18 — do not revert.

**Read TODO.md when:**
- Starting any work session — the "⚡ ACTIVE" block at the top is the freshest snapshot of in-flight runs and pending decisions.
- Launching a long-running job (HRST, Mode P, AB matrix) — add an entry under ACTIVE so future-you / next-session knows what's running.
- Promoting a config to live — record backup paths + rollback command so future-you can revert in one line.

**Read ARCHIVED_LOG.md when:**
- Considering a feature/idea — check the canonical scoreboard (C01-C82) to avoid retesting something already DEAD or revisiting a SHELVED idea without its revival conditions.
- Reviving a historical decision — preserved verdicts include revival conditions in SHELVED entries and root causes in DEAD entries.
- Auditing past decisions ("why did we reject X?", "what was the verdict on Y?").

**Update rules:**
- Active work changes (new HRST launched, config promoted, rollback executed) → TODO.md
- Closing a research arc → move the entry from TODO.md → ARCHIVED_LOG.md with the verdict + numerical evidence inline

This CLAUDE.md is now stable reference — don't write volatile state here; that's TODO.md's job. ARCHIVED_LOG.md is append-only — don't edit closed entries; add new entries that reference old ones rather than rewriting history.
