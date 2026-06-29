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

- **Rally-cooldown revert-abort / dynamic cd (2026-06-16).** Cancelling the fixed cooldown early when the triggering rally reverts to its origin price HURT — `tools/test_cooldown_revert.py` on live `sma48>sma100`: fixed-cd gate +28.84% (blocks 23 chase-buys, +5.9pp over no-gate +22.91%) → revert-abort +21.92% (un-blocks only 2, but they're *falling-knife* entries that erase the gate's edge, dropping *below* no-gate). Reverting-to-origin is not a bottom; the cd's conservatism is the point (stays out for the whole reversion). `cd=24h/14h` were Mode-T winners over `[6..48]`, so shorter fixed cd lost too. See ARCHIVED_LOG C87.
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

**⚠ Revised 2026-06-14 (single 4-horizon HRST, ETH, LGBM=cpu).** The pre-2026-05-06 numbers below them are STALE — they predate H75 (refine trials 50→75 + K=5 multi-seed = 5× cost/trial) and were GPU-assisted. On CPU with K=5/75-trials the refine is now ~5-10× longer. Use these:

| Phase | Sample N | Range | Median |
|---|---|---|---|
| **Mode D grid (60 evals, K=5 parallel)** | 4 (Jun-14) | **24.1 – 26.2 min** | ~25 min |
| Mode V Step 1 (backtest D candidates, parallel) | 4 (Jun-14) | **21.0 – 42.3 min** | ~33 min |
| **Mode V Optuna refine (75 trials × K=5 seeds, CPU)** | 4 (Jun-14) | **30 – 207 min** (207 was a loky pool-stall outlier; healthy 30–121) | ~115 min |
| Mode V Step 3 (refined backtest, parallel) | 4 (Jun-14) | **17.7 – 22.2 min** | ~21 min |
| **Mode V total per horizon (Step1→refine→Step3)** | 4 (Jun-14) | **~86 – 250 min (1.4–4.2h)** | ~170 min |

<details><summary>Stale pre-H75 numbers (2026-05-06 audit, GPU, 50 trials, no K=5)</summary>

| Mode D grid (72 evals) | ~18 | 6.9 – 13.2 min | ~11 min |
| Mode V refine (hybrid GPU+CPU) | ~20 | 10.3 – 21.0 min | ~14 min |
| Mode V total per horizon | inferred | ~50 – 95 min | ~70 min |
</details>

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
| `crypto_revolut_ed_v2.py` | **Live** | Ed V2 auto-trader — maker orders (0% fee) with market fallback. Penny-improvement pricing: buy/sell at `bid+0.01`. `post_only` ensures maker. Stale order cleanup, NTP clock sync, locked funds detection. Reads `regime_config_ed.json`. Auto-promote-when-flat staging (`8454d00`). **2026-06-05: `/status` balance summary no longer lists non-enabled coins (was showing phantom BTC).** **2026-06-25: `/setup` toggles now persist immediately** — `_setup_commit()` saves + syncs live `trading_cfg` after every toggle/setter (was snapshot-only until /cfg_save, so toggles "didn't stick"); confirmation+menu collapsed to one message (the "2 popups" report). Smoke `tools/smoke_test_setup_toggle.py`. |
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

Quick summary of current live (**promoted 2026-06-22, rally-cooldown gates DISABLED 2026-06-23** — verified 2026-06-24 against `config/regime_config_ed.json`): detector **`price>sma72`**, **bull=4h@65%** / **bear=4h@70%** (both RF+LGBM w250 γ0.996, feat-set D, 10 feats), **rally-cooldown gates OFF both regimes**, shields OFF, min_sell_pnl=0%, max_hold=10h, max_position_usd=$14,300, maker ON. **Why no gate on 4h:** a cooldown can only BLOCK buys, and 4h's edge is buying often + riding momentum, so every gate blocks net-winners — **0 / 2304 sweep configs beat no-gate** (best bull −1.3pp, every bear ≈ −25pp; reconfirmed by a fresh rerun 2026-06-24). This *reverses* the 5h/8h Idea-1 result (+7.7pp) where blocking chase-buys helped — a cooldown helps a slow horizon, hurts a fast/frequent one. **⚠️ WATCH:** 4h/4h is a historically-marginal horizon (Mode-C history only ~2mo, recent-window-favoring); the rally-free OOS comparison is the open residual — monitor live. **`ENABLED_DETECTORS`** = `tsmom_672h, sma168>sma480, sma48>sma100, tsmom_168h, price>sma72, vol_calm` (6; trader `_evaluate_named_detector` wired for all). Live inference runs through `compute_signal_core()` in `crypto_trading_system_faye.py`; regime eval in `crypto_live_trader_ed.py::_evaluate_named_detector`. Per-asset: ETH live, BTC/SOL/LINK `enabled: false`, BNB code-wired (PySR/HRST not run), XRP removed 2026-05-23. **Prior live (06-16→06-22, superseded):** `sma48>sma100` bull 8h@90 / bear 5h@65, single bear gate `rr8≥2.5 cd14`. **Full state + rollback: [TODO.md](TODO.md) `📌 LIVE STATE`.**

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

*Reorganized + code-audited 2026-06-14 (rule-by-rule with the owner): now 21 gates in 6 families (promotion-first), de-duplicated, verified against live code. 8 reference/state rules demoted (end of section); asset universe → TODO.md. Each family heads the queries it gates. (4 gates added 2026-06-14 from a code-grounded gap audit — F1.5 atomic FAYE promote, F2.11 refuse-on-fault, F4.17 data-source lag-audit, F6.21 gated-sim-not-screen.)*

> **Rule 0 — Answer from evidence, not intuition** *(prime directive, user-mandated 2026-06-14).* When the user asks a question — especially about live state, production config, data, dates, or results — **verify against the actual source before answering**: read the code (cite `file:line`), the production CSV / regime config, the logs, the data. Answer **with the numbers/values you found, not from memory or gut**; if you can't verify it, say "unverified" rather than assert. This was the session's #1 failure mode — the live-config saga, the "06-01" build date, the "contaminated" ablation verdict were all confident gut-claims the numbers later contradicted. The check is always cheaper than the correction. [[feedback-verify-with-numbers]]

### Query Pre-Flight — run on EVERY user message (no exceptions, user-mandated 2026-06-14)

For every request, before acting: (1) apply **Rule 0** (verify against the source, answer with numbers), then (2) classify the query and fire the matching family's gates — STOP if any fails:

| If the query… | Fire |
|---|---|
| writes to live config / models / prod CSV / live engine | **F1 Promotion gates** |
| changes live inference / signal generation | **F2 Inference correctness** |
| touches the live trader's config, orders, or display | **F3 Execution** |
| downloads / merges a data source | **F4 Data integrity** |
| judges if a job runs, compares backtests, or emits a user command | **F5 Process** |
| promotes / validates a feature, model, or hyperparam (gated sim vs screen) | **F6 Research validity** |

Universal: *verified with numbers not guts (Rule 0)? · flat-before-promote? · leakage clean? · isolated from production? · liveness by mtimes not processes? · screen or gated-truth?*

### F1 — Promotion gates (before any live config / model / CSV / engine change)

**Principle (was R1):** never modify the live engine without testing first. The live trader imports its engine by exact filename — currently **`crypto_trading_system_faye.py`** (serving migrated from `crypto_trading_system_ed.py` on 2026-06-05, [crypto_live_trader_ed.py:61](crypto_live_trader_ed.py#L61)); FAYE generates the models, the v2 trader consumes them.

1. **Trader FLAT before swap (was R19).** Verify `state=cash` in `config/position_ed_v2_<ASSET>.json` before any config / production-CSV swap — promoting mid-position corrupts entry/exit bookkeeping. User may override per-promotion (e.g. a strictly-more-conservative config on a losing trade). [[feedback-flat-before-promotion]]
2. **Leakage check before promotion (was R8).** Before writing production CSV, confirm no leakage — for PySR, `discovery_method == "historical"`. Enforced by `_check_pysr_leakage` ([crypto_trading_system_ed.py:2024](crypto_trading_system_ed.py#L2024)), which BLOCKS the write on a non-historical formula.
3. **Regen PySR ⇒ retrain (was R14).** Mode P rewrites `models/pysr_*.json`, but production stores feature *names* only — inference re-evaluates whichever formula is in the JSON *now*. After any Mode P, run DV/HRS for the same horizon or you get silent feature drift. (The leakage check does NOT catch train/inference mismatch.)
4. **CSV merge by coin AND horizon (was R2).** Upserting to production CSV masks on BOTH coin AND horizon before delete-then-append; filtering by coin alone wipes every horizon's row for that coin.
5. **Promote FAYE atomically + verify staged freshness (added 2026-06-14).** Copy `models_faye/crypto_faye_production.csv` AND `config_faye/regime_config_faye.json` into live `models/`+`config/` in ONE step, and verify both are unchanged (mtime) since the HRST that produced them — a later HRST overwriting the staged path yields a frankenstein (new config + old models). Auto-promote was disabled 2026-06-13 ([crypto_revolut_ed_v2.py:4374](crypto_revolut_ed_v2.py#L4374)) after exactly that fired in production. [[feedback-flat-before-promotion]]

### F2 — Live inference correctness (break these → silently wrong live signals)

6. **No embargo in live (was R9).** Live training window must be `[train_start, n-1]` (all labelled data). The dynamic `train_end = i - horizon` purge (`EMBARGO_CANDLES_DEFAULT`) is backtest/CV ONLY — it prevents label-overlap leakage in *evaluation*; live has no future to leak (every label uses known past prices), so embargoing live just discards the freshest, most-predictive rows. Per Lopez de Prado. [[feedback-no-live-embargo]]
7. **Mirror inference changes to the shadow (was R23).** Any change to the live inference path (inference-row selection, embargo, NaN policy, feature ffill, train-window edge) MUST be mirrored in `crypto_live_shadow.py` — two separate codepaths. Grep `USE_CLOSED_BAR_FOR_INFERENCE` to find the mirror block in both files (don't trust line numbers). The shadow is the primary live-correctness tripwire; an un-mirrored change silently tanks the match rate (proven Jun-2026, 100%→48%).
8. **Validate on GPU (was R24).** Run sanity / engine-vs-trader parity on GPU — the validator defaults to `device=LGBM_DEVICE='gpu'`, mirroring the live trader, and is proven cross-machine deterministic. `--cpu-lgbm` is a no-GPU fallback only (~3-5pt parity gap). The validator + `tools/sanity_check.py` already run GPU by default (no `--cpu-lgbm`).
9. **Caches must be mtime-aware (was R20).** Any module-level cache holding file content MUST store `(mtime, df)` and re-read when mtime advances. Verified clean: `_load_macro_csv` ([crypto_trading_system_ed.py:1204](crypto_trading_system_ed.py#L1204)), `_regime_config_cache` ([crypto_live_trader_ed.py:169](crypto_live_trader_ed.py#L169)). Banned: `if fn in cache: return cache[fn]` with no mtime compare (froze macro features at startup — TODO 0527).
10. **Address signals by slot, not horizon (was R21, gate remnant).** `signal_log.csv` + horizon-keyed lookups are regime-anchored: slot 1 = bull model, slot 2 = bear model. Address by SLOT (`h_1/sig_1`, `h_2/sig_2`), never by `HORIZON_SHORT/LONG` (which leaves slots empty when the live horizon differs).
11. **Refuse the cycle on a structural fault (added 2026-06-14).** The live trader MUST return None / skip the cycle on `regime=='error'`, `n_train < window+100`, or any non-sparse feature breaching its cadence-aware staleness SLA — NEVER fall back to a default config/horizon. These guards are the only thing between a structural fault and a real-money trade on garbage inputs (the "86%-pin" bug: the trader silently inferred on a forced bull-6h default). Live: [crypto_live_trader_ed.py:993](crypto_live_trader_ed.py#L993) (regime-error, Fix #4), `:716` (insufficient data), `:787-814` (staleness SLA — `oc_`/`m_`/`fg_`/`pysr_` 60h, hourly 2h, sparse skipped); mirror these in `crypto_live_shadow.py` (rule 7).

### F3 — Live trader config & execution

12. **Regime config shape (was R5, +R13).** `config/regime_config_ed.json` holds per-asset `regime_detector` + `bull`/`bear` blocks (each: horizon, min_confidence, max_position_usd, hold_shield, per-regime rally_cooldown). The per-regime block's `min_confidence` is the live gate ([crypto_live_trader_ed.py:997](crypto_live_trader_ed.py#L997)); module-level `MIN_CONFIDENCE=75` is only a fallback. Per-regime `hold_shield` lives inside the bull/bear blocks (`_shield_on_for_regime` falls back to asset-level for legacy config); shared `min_sell_pnl_pct` + `max_hold_hours` stay at asset level.
13. **Maker orders (was R10).** Orders are `post_only` and never cross the book: BUY rests at `bid+0.01`, SELL at `bid+max($0.02, 5%×spread)`; cancel stale orders first, reprice each cycle, market-fallback on timeout. Windows/intervals are per-asset config (`regime_config_ed.json <ASSET>.maker`: `window_secs`/`check_interval_secs`/`partial_boost_secs`, defaults 300/15/150).
14. **Clock drift via NTP (was R11).** Correct drift from an independent NTP query (`_sync_clock_ntp`) on every 409 + every 5 min — NOT from the 409 response's echoed timestamp (which re-applies the stale offset on worsening drift).
15. **Display timezone (was R4).** Telegram signal-display timestamps in `crypto_live_trader_ed.py` render in Europe/Zurich via `_to_local()` (2 sites); all order/internal logic runs in UTC and the main trader's other displays are system-local. Display-only — no trading-correctness impact.

### F4 — Data-source integrity (write-side)

16. **Preserve history on every download (was R22).** Any download that re-pulls + writes MUST go through `_dedup_preserve_history` / `_merge_preserve_history` (`keep='first'` historical, `keep='last'` current). Bare `to_csv` after a full-history pull re-introduces the drift bug (upstream revisions silently overwrite originally-observed values). See the Adding-a-New-Data-Source checklist.
17. **Data-source wiring: merge-key + publish-lag + 60-day quarantine, lag-audited (added 2026-06-14).** For any new/changed source: daily sources merge on `_merge_date` (inherits the publish lag — standard D-1, on-chain D-2), hourly/intraday on `_merge_dt` (no lag); sources with <60 days history go in `always_disabled_exact`; then run `python tools/audit_feature_lag.py` and require **exit 0 (zero lag violations)** before promotion. Wrong merge key = backtest clairvoyance (daily-on-`_merge_dt` leaks the future — the 2026-06-01 `oc_mvrv_chg1d` BUY→SELL flip), invisible in backtest until live divergence. Detail: the 9-item "Adding a New Data Source" checklist.

### F5 — Operating process (judgment, not code gates)

18. **Liveness by mtimes, two-machine (was R16).** The engine folder is Drive-shared, so a job on the other machine is invisible to local `tasklist`/`Get-Process`. Judge liveness by file mtimes — the run's log first, then the noprod / `models_faye` prod CSV + regime config, then grid CSVs. <2 min → alive; 2-10 min → wait; >10 min stale + no local python → likely dead but ASK first. **Mode R runs silent up to ~2h** (flushes once at end) — don't declare it dead. Always ask "is it running on the other machine?" before saying stopped. (Past miss: declared variant A dead off the laptop process list while it ran on the desktop.)
19. **Window-scale backtest comparisons (was R17).** Returns scale with replay length — a 4mo run ≈ 2× a 2mo run on the same strategy. To compare fairly, take the **most-recent half** of the longer run — the Mode T/S/G logs print this as **H1** (the engine defines **H1 = recent half, H2 = earlier half**, [crypto_trading_system_ed.py:6938](crypto_trading_system_ed.py#L6938)) — and compare to a short run over the same period; or annualize (`return × 365×24/replay_h`); or re-run on a common window. Always check the `Replay: NNNNh` header first. *(Corrected 2026-06-14: the old text had H1/H2 inverted.)*
20. **Bare commands for the user (was R18).** Emit user-facing Python as bare `python tools/<script>.py [args]` — the venv is already active. No absolute/quoted interpreter paths (PowerShell parse friction). User-requested 2026-05-03; do not revert.

### F6 — Research validity (how you measure, not what you promote)

21. **Validate through the gated sim, never an ungated screen (added 2026-06-14).** Never promote a feature, model combo, or hyperparam on LGBM importance / AUC / an ungated grid score — validate ONLY through the real gated backtest (Mode V/HRS, `tools/bt_*.py`, or the live shadow). Ungated screens are systematically optimistic by 7-27pp because they skip the live confidence gate: importance-ranked fast-window features (ranked #2-#4) cost −22 to −27.5pp gated (Engine Reference Card); GB+LGBM won the ungated bear screen but lost the real engine +43.5% vs +51.2% ([crypto_trading_system_ed.py:336](crypto_trading_system_ed.py#L336)); a 2-model RF+LGBM pair beats every triple/quad/quintuple in the gated sim while LR poisons every combo. **Importance ≠ performance — only the gated sim decides.**

22. **Backtest fills are zero-lag idealized; live execution isn't — a thin edge can be a fill-timing artifact (added 2026-06-29).** Every backtest (Mode V/HRS, `bt_*.py`, the lag sim) transacts **instantly at the inference-bar close**; the live trader executes 1+ cycles later at the real **bid**, which in fast moves diverges sharply — the 2026-06-26 ETH crash filled SELLs up to **3% below** the decision-bar close, yet **all were CLEAN maker fills at the live bid, 0% fee — NOT slippage**. So a backtest **overstates** returns by the decision-to-execution lag, worst for **frequent-trading horizons** (4h) and **volatile** regimes. **Before trusting a backtest edge, run the lag ladder** (re-fill the same signals at +0/+1/+2 bars): live 4h/4h backtested **+3.85% at +0 lag** but **−2.89% at +1 bar ≈ the live −3.78%**. An edge that survives only at +0 lag is not real; the honest number is the **live realized PnL (ledger)**, not the backtest. Full method + traps: the **Backtest-vs-Live Fidelity playbook** below.

### Demoted to reference (2026-06-14 — non-gating, kept for lookup)

- **Production scoring (was R7):** Mode V/H picks the production model by `score = return × (win_rate/100)` for positive returns, raw return for negatives (≥5-trade configs; 5 identical call sites). *Describes selection, gates nothing.*
- **Windows SSL (was R6):** `ssl._create_unverified_context()` is applied in the live files + downloaders — apply it in any new live HTTP-calling file on Windows. *Setup workaround.*
- **Set-D feature_override (was R3):** callers of `generate_signals()` for a feature_set-D model must pass `feature_override=config['optimal_features'].split(',')`, else it falls back to the default feature set (silent wrong features). *Internal calling convention — relevant only when writing a new caller.*
- **Mode T chains rally-cooldown (was R15):** Mode T merges bull/bear signals (`_merge_tagged_signals`) then auto-runs the rally-cooldown sweep (`_sweep_rally_cooldown`) — no need to run Mode G separately after T; standalone G is a cache-fed fast-iteration fallback. *Engine data-flow.* **No-winner = DISABLE, not no-op (fixed 2026-06-23, [faye.py:8048](crypto_trading_system_faye.py#L8048)):** when no single gate beats the no-gate baseline (STRICT 3of3 empty), the sweep now WRITES an explicit `enabled:false` single-form gate (`h_long=h_short`, `t_long=9999`) instead of leaving the prior config untouched. The old `return None` no-op let a stale `enabled:true` double survive every run and ride a verbatim `--promote` to live (the 2026-06-22 4h/4h double-gate incident — FAYE only ever *computes* single gates, so the double was pure stale residue). Regression: `tools/test_gate_disable_on_nowinner.py`. *Open follow-ups: `--promote` gate-validation, and a selection-ranking review (the binary STRICT cliff rejects gates that merely tie on a rally-less half).*
- **signal_log full schema (was R21):** `timestamp,asset,price,action,confidence,h_1,sig_1,conf_1,h_2,sig_2,conf_2`; chart-rendering only, write errors swallowed. *The gate remnant is rule 9.*
- **Asset universe (was R12) → TODO.md LIVE STATE:** per-asset enabled/standby state is volatile, not a rule. Invariant retained: **never trade an `enabled:false` asset**; the live roster is `regime_config_ed.json` enabled flags + TODO.md.

---

## Backtest-vs-Live Fidelity — diagnosis playbook (built 2026-06-29)

**When a backtest beats live and you need to know WHY** (the 2026-06-26 ETH 4h/4h case: sim **+2.40%** vs live realized **−3.78%**, same ~14 trades). Localize the gap with an isolation ladder — never guess. **Ground-truth artifacts:** `output/inference_snapshots.jsonl` (the frozen per-hour signal + the exact features the trader used — reproduces live signals 100%), the trader order log (`logs/trader/ed_runtime_*.log` — real bid/ask + fill price per order), `tools/sanity_check.py`.

**Isolation ladder (hold ONE variable per rung):**
1. **Bookkeeping** — `tools/validate_snapshot_replay.py`: recompute (signal, conf) from the trader's frozen inputs → must be 100% (was 650/650 — live signals are self-consistent given their inputs).
2. **Engine vs trader** — `tools/validate_refit_replay.py`: refit on the trader's frozen training matrix → isolates engine code vs the data it trained on.
3. **Data revision** — re-infer on CURRENT (revised) data, same hours → flips here (when 1+2 are clean) = the sim's **hindsight** (deriv_* features get rewritten every download; cross-asset can be stale).
4. **Feature attribution** — diff the frozen vs rebuilt feature vector on the flipped hours → names the culprit source.
5. **Execution lag** — the **lag ladder**: re-fill the same signals at +0/+1/+2/+3 bars. The decisive rung for thin, frequent-trading edges (gate 22).

**Traps that cost a full session (2026-06-29) — do NOT repeat them:**
- **Timezone.** The position-ledger BARE timestamps are **CEST (UTC+2)**; price bars + `inference_snapshots.logged_at` are **UTC**; ISO-`Z` ledger times are UTC. Comparing a bare ledger time to a UTC bar is **2h off** → invents phantom 1–2% "discrepancies". (This produced a WRONG "degraded data / out-of-range fills" conclusion. Convert to a common tz FIRST.)
- **Confounded benchmark.** Never compare a fill to the **bar close of the trade's own hour** — that close is up to 59 min in the future → conflates execution with next-hour drift. The theoretical fill price is the **decision/inference-bar close** (== `inference_snapshots.close`), nothing else.
- **Degraded intrabar OHLC.** `eth_hourly_data.csv` high/low ranges are collapsed (~**0.03%** vs **0.57%** baseline, since ~April) — **closes are fine** (signals OK) but **high/low are unreliable** → any backtest using intra-hour high/low (TP/SL overlays, intra-hour fills) is invalid on this data.
- **Maker fills are clean — don't blame "slippage".** The order log proves fills land at the real live bid, 0% fee, even the worst crash exit (`Maker sell #5 @1514.09 bid=1514.01 FILLED`). The gap is the **decision-to-execution price MOVE** (lag), which hourly bars cannot model — not order-execution inefficiency.

**Verdict (2026-06-26 4h/4h week) — ROOT CAUSE FOUND 2026-06-29:** the backtest (+3.85%) ≠ live (−3.78%) because **the backtest and the live trader are two different code paths that build different TRAINING WINDOWS** — `generate_signals` (faye, backtest) uses `[i-window, i-embargo]` (last row i-5, over-embargoed) while `generate_live_signal` ([crypto_live_trader_ed.py:611](crypto_live_trader_ed.py#L611), live) trains the last `window` labelled rows ending at i-3 (the live df_full carries the forming bar i+1, so dropna(label) leaves labels valid to i-horizon+1). **Same data (proven byte-identical), but ~1-2 different edge rows — and that 1 highest-gamma-weight row flips ~25% of signals** (live↔backtest signal agreement was **75%**). **NOT execution, NOT timing/lag, NOT data revision, NOT device** — all of those were measured and ruled out (fills clean ~1.7pp; inference+training features 0.000000 diff; engine refits the frozen matrix at 97.5%). **Fix:** `FAYE_FAITHFUL_WINDOW=1` makes `generate_signals` replicate live's edge → backtest↔live agreement **75% → 99%** ([faye.py:3106](crypto_trading_system_faye.py#L3106), additive, default-off, backtest-only). **Use it for any live-prediction backtest.** **LEAK FIXED 2026-06-29 (F2):** live (`generate_live_signal` [crypto_live_trader_ed.py:905](crypto_live_trader_ed.py#L905)) + shadow (`crypto_live_shadow.py`) now DROP the forming-bar-labelled row from `df_train` when the closed-bar stepback fires (`_leak_rows = (n-1)-i`), so the last training row is `i-horizon` (label uses only the closed inference bar) — leakage-free, matching `FAYE_FAITHFUL_WINDOW` (now edge-4 = `i-horizon+1`). NOT an embargo (the dropped row's label was computed from the incomplete forming bar). **Takes effect on trader restart; shadow validates live==core ≥99% post-restart.** A leakage-free HRST (`python crypto_trading_system_faye.py HRST ETH 4,5,6,7,8h --replay 1440` (leakage-free DEFAULT)) re-selects the config on a backtest that predicts the fixed live — the prior 4h/4h vs 8h/5h verdicts ran on the divergent path and should be re-run. **The earlier "execution-lag" verdict was WRONG** — execution lag is small (a 60s lag sweep cost ~0pp); the gap was the train-window edge.

**Where the live lag actually came from (root-caused 2026-06-29, FIXED):** the order was placed **~36s after the bar close**, and **~33s of that was the P1 blocking data download** — `derivatives_eth` re-paginated the **full 2022→now history every 2h** (perp klines = 40 pages; the retrain is only 0.6s, feature build ~2s — neither is the bottleneck). In the 06-26 crash the price moved ~2.4% during that download before the order existed. **Fix:** `download_derivatives_data` is now **incremental** (fetch from `last row − 3d` + `_merge_preserve_history`), validated byte-identical to a full pull at **4.3s vs 32s**. So P1 ≈ 7s now. This was NOT execution slippage (maker fills are clean at the live bid) and NOT a data bug — a slow download on the critical path. **Lesson for backtest fidelity: the sim assumes instant fill at the decision bar; live can't act until P1 finishes — keep the critical-path download fast.** Per-cycle phase timings are recorded in `output/cycle_metrics.csv` (`p1_duration_sec`, `feature_build_sec`, `model_fit_sec`).

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

3. **Overwrite safety (Critical Rule 22) — never clobber history.** Any download that re-pulls + writes MUST go through `_dedup_preserve_history` / `_merge_preserve_history` (`keep='first'` historical, `keep='last'` current period). Bare `to_csv` after a full-history pull re-introduces drift (upstream revisions silently overwrite originally-observed values). **Atomic write + read-guard (2026-06-24):** write snapshot/merge files via `_atomic_to_csv` (temp file + `os.replace`) so a concurrent reader / Drive-sync never catches the file mid-write at 0 bytes (the "No columns to parse from file" `EmptyDataError` race that broke a live cycle), and guard every existing-file read with `os.path.getsize>0` + `try/except` so an already-empty/corrupt file self-heals to a fresh write. The merge reads (fear_greed/on-chain/derivatives, + yfinance incremental) are wrapped in `try/except` so a bad read self-heals to a fresh pull. **2026-06-28 follow-up:** the original 2026-06-24 fix only made the writes atomic for the **3 forward-only snapshot saves** (IV/orderbook/whale) and left the **7 merge/download writes bare** (`macro :656`, `fear_greed :716`, `cross_asset :816`, `onchain :975`, `derivatives :1290`, `hourly :1426`, `stablecoin :1481`) — so those files still hit the 0-byte/partial-write race, which resurfaced as recurring "No columns to parse from file" + "time data does not match format" errors at hourly cycle boundaries (2026-06-28 02:01/07:01/10:01). All 7 now go through `_atomic_to_csv` — **10/10 writes in `download_macro_data.py` are atomic.** **Shrink-guard (2026-06-29):** `_atomic_to_csv` ([download_macro_data.py:67](download_macro_data.py#L67)) itself now REFUSES to replace a >100-row file with a df <50% its size (alerts critical; `allow_shrink=True` escape for genuine full-rewrites) — a single chokepoint that structurally blocks the data-clobber class (orderbook 1351→34 / cross_asset 1637→9) for EVERY data write incl. future ones, so a forgotten per-file read-guard can no longer silently lose history. Atomic prevents *torn* writes; the shrink-guard prevents *short-but-complete* ones. Test `tools/test_shrink_guard.py` (6/6). **2026-06-28 silent-clobber guard:** the same race also bit the READ side — `_yf_merge_with_existing` ([download_macro_data.py:524](download_macro_data.py#L524)) read `cross_asset.csv` empty mid-write (during a concurrent FAYE HRST) and **silently returned the tiny incremental slice over 1637 rows of history** (no alert), NaN-ing every `xa_*_corr*` feature → 8h PySR `pysr_1` (which divides by xa-corr) → NaN → all-rows-skip → 0 signals. The empty-existing branch now **retries-to-recover, then ALERTs critical** instead of silently clobbering; history restored via `download_cross_asset(full=True)`. **Daily guard:** `tools/sanity_check.py` now has a deterministic **[4] DATA INTEGRITY** check (`data_integrity_check`, `_INTEGRITY_REGISTRY`) that floors row-count + date-span for 6 critical CSVs and **drives the daily-sanity verdict** — a history collapse now FAILs the 08:30 daily + Telegrams. Add new long-history sources to `_INTEGRITY_REGISTRY`.

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
