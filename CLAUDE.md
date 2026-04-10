# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Automated ML trading system for crypto (BTC, ETH, XRP, DOGE, SOL, LINK, ADA, AVAX, DOT). Generates hourly BUY/SELL/HOLD signals using ensemble ML models with walk-forward validation, temporal decay sample weighting, and embargo-corrected labels. Variable horizon per asset (5h, 6h, 7h, 8h — optimized via Mode H). Executes trades on Revolut X via Ed25519-signed API.

**Production:**
- **Ed V2** — Regime-switching trading (`crypto_trading_system_ed.py` + `crypto_revolut_ed_v2.py`). Dynamic bull/bear horizon selection via external config (`config/regime_config_ed.json`). Maker-order pricing at `bid+0.01` with `post_only` for 0% fees. Mode R regime backtest. Currently ETH-only.

Doohan V1.7.1, Deku, CASCA and all prior versions archived (Doohan retired 2026-04, others 2026-03-24). See `archive/`.

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

**Telegram commands (Ed V2 trader):** `/stop` `/status` `/pause` `/resume` `/balance` `/sync` `/conf` `/config` `/setup` `/help` `/chart BTC` `/regime` (show current bull/bear state per asset)

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
10. **Ed V2 maker order rules:** place limits at `bid+0.01` with `post_only`, re-price every 10s, market fallback at 60-120s. Always cancel stale orders before placing new ones (prevents fund locking). NTP-sync clock on startup.
11. **Stale config cleanup:** ETH block in `regime_config_ed.json` is now clean (no top-level legacy keys). SOL/LINK/XRP/DOGE/ADA/AVAX/DOT still have legacy `strategy`/`horizon`/`min_confidence`/`max_position_usd` at top level but are all `enabled: false` — left as-is intentionally.

---

## Pending Work

### TODO

**HIGHEST PRIORITY:**
1. **Check HRS ETH 6,7,8h --replay 1440 results** — Running overnight 2026-04-10. First full pipeline run with GDELT geopolitical features + V3 joint sweep in production Mode S. Check logs tomorrow morning (2026-04-11). This tests the entire new stack: GDELT features (iran_vol, iran_tone, geopolitical_tone), V3 joint sweep Mode S (global optimization instead of sequential R→S), and --replay 1440 (2-month window). Compare with previous RS results (+52-61%) and check if GDELT features rank in importance.
2. **Investigate signal nondeterminism** — Three RS ETH 1440h reruns (7 Apr 15:24, 7 Apr 16:08, 9 Apr 16:03) produced winners +49.98% / +60.72% / +52.36% with shifting bear horizons (8h@90% → 8h@85% → 6h@85%). ~11pp swing from same script + same window. Pin seeds in XGB/LGBM/RF signal generators OR add a "run sweep N times and average" wrapper before trusting any single winner.
3. **SV3 ETH `--replay 2880` full grid** — Now lower priority since V3 joint sweep is in production Mode S. The 8 Apr SV3 run was 1440h with only 8 horizon pairs — full-grid still pending but may be redundant.

**Other:**
4. **Eli HRS BTC** — `python crypto_trading_system_eli.py HRS BTC 4,5,6,7,8,9,10` — 30-minute candle test
5. **Ein results review** — Check Ein (15min) BTC results from laptop run

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
