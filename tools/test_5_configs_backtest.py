"""tools/test_5_configs_backtest.py — counterfactual backtest of 5 historical
production configs on a common Apr 10 → May 6 15:48 window (~27d, 648h).

Tests:
  1. V1 era       - models/crypto_ed_production_pre_A_20260426.csv
                    config/regime_config_ed_pre_A_20260426.json
  2. Variant A    - models/crypto_ed_production_pre_2det_20260430.csv
                    config/regime_config_ed_pre_2det_20260430.json
  3. 2-detector   - models/crypto_ed_production_pre_sma24sma100_20260503.csv
                    config/regime_config_ed_pre_sma24sma100_20260503.json
  4. sma24>sma100 - models/crypto_ed_production.backup_20260506_pre_HRST_promote.csv
                    config/regime_config_ed.backup_20260506_pre_HRST_promote.json
  5. today (live) - models/crypto_ed_production.csv
                    config/regime_config_ed.json

Method:
  For each config:
    A. Parse regime config to get: detector name, bull/bear horizon+conf+shield+gate,
       shared min_sell_pnl + max_hold
    B. From prod CSV, get bull-horizon and bear-horizon model spec
       (combo, window, gamma, n_features)
    C. Walk-forward regenerate hourly signals for both horizons over Apr 10 →
       May 6 15:48. Retrain every K=6h (108 fits per horizon, model reused
       intra-K-block). Each fit: train on data[i-window : i-horizon] with
       gamma weights, evaluate features via LGBM importance + floor.
    D. Apply detector to tag each bar bull/bear
    E. Merge: pick bull-horizon vote in bull bars, bear-horizon vote in bear bars
    F. Apply confidence threshold (per regime)
    G. Apply shield (per regime, if enabled): block SELL if pnl < min_sell_pnl
       AND held < max_hold
    H. Apply max_hold failsafe: force SELL if held >= max_hold
    I. Apply rally-cooldown gate (per regime, if enabled): block BUYs for
       cd_hours after rolling-rr exceeds threshold
    J. Walk through hourly bars, simulate trades, capture per-trade returns +
       per-bar drawdown

Output:
  Per-config: return%, max_dd%, cdar_5%, n_trades, win_rate, avg/trade, $/day
  Then ablations on leader: shield ON/OFF × gate ON/OFF (4 variants)

Runtime: ~90 min on laptop (5 configs × 2 horizons × 108 retrains × ~5s/fit).
Compute scales linearly with TEST_WINDOW size and inversely with K (retrain step).

Run on laptop (Desktop is busy with C04→C08):
  python tools/test_5_configs_backtest.py

Outputs:
  output/test_5_configs_<ts>.csv    — per-config + ablation results
  output/test_5_configs_trades_<ts>.csv  — per-trade returns + entry/exit times
  logs/test_5_configs_<ts>.log      — console log
"""
from __future__ import annotations

import os
import sys
import json
import math
from datetime import datetime, timedelta

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)

import numpy as np
import pandas as pd

# Engine imports — guard import-time so the script gives a clear message if engine moved
from crypto_trading_system_ed import (
    load_data, _build_features, _test_lgbm_importance, ALL_MODELS,
    get_decay_weights, BACKTEST_FEE_PER_LEG, _feature_floor_indices,
    _build_regime_indicators_and_detectors,
)

ASSET = 'ETH'
TS = datetime.now().strftime('%Y%m%d_%H%M%S')
LOGS_DIR = os.path.join(ENGINE, 'logs')
OUT_DIR = os.path.join(ENGINE, 'output')
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Test window: Apr 10 00:00 UTC → May 6 15:48 UTC (today's HRST promotion time).
# All 5 configs evaluated on this same window so comparisons are apples-to-apples.
# CLI overrides: --start "YYYY-MM-DD HH:MM" --end "YYYY-MM-DD HH:MM" --tag SUFFIX
import argparse as _ap
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument('--start', type=str, default='2026-04-10 00:00:00')
_p.add_argument('--end',   type=str, default='2026-05-06 15:48:00')
_p.add_argument('--tag',   type=str, default='')
_args, _ = _p.parse_known_args()
TEST_START = pd.Timestamp(_args.start)
TEST_END   = pd.Timestamp(_args.end)
WINDOW_TAG = _args.tag if _args.tag else f"{TEST_START.strftime('%m%d')}_{TEST_END.strftime('%m%d')}"

# Walk-forward retrain step. K=6 means refit every 6 hours; predictions are
# still emitted every hour, using the most recent fitted model. K=1 is true
# live behavior but ~6× slower. K=6 is a reasonable compromise.
RETRAIN_STEP = 6

# 5 historical configs (label, regime_config_path, prod_csv_path)
CONFIGS = [
    ('V1 era',       'config/regime_config_ed_pre_A_20260426.json',
                     'models/crypto_ed_production_pre_A_20260426.csv'),
    ('Variant A',    'config/regime_config_ed_pre_2det_20260430.json',
                     'models/crypto_ed_production_pre_2det_20260430.csv'),
    ('2-detector',   'config/regime_config_ed_pre_sma24sma100_20260503.json',
                     'models/crypto_ed_production_pre_sma24sma100_20260503.csv'),
    ('sma24>sma100', 'config/regime_config_ed.backup_20260506_pre_HRST_promote.json',
                     'models/crypto_ed_production.backup_20260506_pre_HRST_promote.csv'),
    ('today live',   'config/regime_config_ed.json',
                     'models/crypto_ed_production.csv'),
]

LOG_PATH = os.path.join(LOGS_DIR, f'test_5_configs_{WINDOW_TAG}_{TS}.log')


def _log(msg: str = ''):
    print(msg)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


# =============================================================================
# Phase 1 — generate hourly signals for one (config, horizon) over test window
# =============================================================================
def generate_hourly_signals(asset: str, horizon: int, model_spec: dict,
                            test_start: pd.Timestamp, test_end: pd.Timestamp,
                            retrain_step: int = RETRAIN_STEP) -> pd.DataFrame:
    """Walk-forward generate hourly signals for a single horizon.

    model_spec = {'combo': ['XGB', 'LGBM'], 'window': 250, 'gamma': 0.997,
                  'n_features': 13}

    Returns DataFrame indexed by datetime with columns:
       prob, prediction (0 or 1), close
    Bars from test_start to test_end inclusive (hourly).
    """
    combo  = model_spec['combo']
    window = int(model_spec['window'])
    gamma  = float(model_spec['gamma'])
    nfeat  = int(model_spec['n_features'])

    df_raw = load_data(asset)
    if df_raw is None:
        raise RuntimeError(f'load_data({asset}) returned None')

    df_features, feature_cols = _build_features(df_raw, asset,
                                                feature_override=None,
                                                horizon=horizon)
    df_clean = df_features.dropna(subset=feature_cols + ['label']).reset_index(drop=True)
    importance_df = _test_lgbm_importance(df_clean, feature_cols, gamma=1.0)
    ranked = importance_df['feature'].tolist()
    df_op = df_clean.dropna(subset=ranked + ['label']).reset_index(drop=True)
    sel_idx = _feature_floor_indices(ranked, nfeat)
    feat_np = df_op[ranked].values.astype(np.float64)[:, sel_idx]
    labels_np = df_op['label'].values.astype(np.int32)
    closes_np = df_op['close'].values.astype(np.float64)
    times = pd.to_datetime(df_op['datetime']).values

    n_total = len(df_op)

    # Find indices that fall inside the test window
    times_pd = pd.to_datetime(times)
    test_mask = (times_pd >= test_start) & (times_pd <= test_end)
    if not test_mask.any():
        raise RuntimeError(f'No bars in test window for {asset} h={horizon}')
    test_indices = np.where(test_mask)[0]
    first_i = int(test_indices[0])
    last_i  = int(test_indices[-1])

    # Need at least `window + horizon` bars before first_i to start training
    if first_i < window + horizon + 50:
        raise RuntimeError(f'Insufficient pre-window data: first_i={first_i}, '
                           f'need {window + horizon + 50}')

    out_rows = []
    fitted_models = None
    fitted_norm = None
    last_fit_i = -10**9

    for i in range(first_i, last_i + 1):
        # Retrain every retrain_step bars (or at first iteration)
        if i - last_fit_i >= retrain_step or fitted_models is None:
            train_start = max(0, i - window)
            train_end = max(train_start, i - horizon)
            X_train = feat_np[train_start:train_end]
            y_train = labels_np[train_start:train_end]
            if (len(np.unique(y_train)) < 2 or
                np.isnan(X_train).any() or len(y_train) < 50):
                # Fall back: skip prediction this bar
                fitted_models = None
                last_fit_i = i
                continue
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0); std[std == 0] = 1.0
            X_tr = (X_train - mean) / std
            sw = get_decay_weights(len(y_train), gamma)
            ms = []
            for mn in combo:
                try:
                    m = ALL_MODELS[mn]()
                    m.fit(X_tr, y_train, sample_weight=sw)
                    ms.append(m)
                except Exception:
                    pass
            if not ms:
                fitted_models = None
                last_fit_i = i
                continue
            fitted_models = ms
            fitted_norm = (mean, std)
            last_fit_i = i

        if fitted_models is None:
            continue

        X_test = feat_np[i:i+1]
        if np.isnan(X_test).any():
            continue
        mean, std = fitted_norm
        X_te = (X_test - mean) / std

        votes = []
        for m in fitted_models:
            try:
                if hasattr(m, 'predict_proba'):
                    p = float(m.predict_proba(X_te)[0, 1])
                    votes.append(p)
                else:
                    votes.append(float(m.predict(X_te)[0]))
            except Exception:
                continue
        if not votes:
            continue
        prob = float(np.mean(votes))

        out_rows.append({
            'datetime': pd.Timestamp(times[i]),
            'close': float(closes_np[i]),
            'prob': prob,
        })

    out = pd.DataFrame(out_rows).set_index('datetime').sort_index()
    return out


# =============================================================================
# Phase 2 — regime tag + merge bull/bear signals
# =============================================================================
def get_regime_for_window(asset: str, detector_name: str,
                          test_start: pd.Timestamp, test_end: pd.Timestamp) -> pd.Series:
    """Return Series of bool (True=bull) indexed by hourly datetime."""
    ind, detectors = _build_regime_indicators_and_detectors(asset)
    if detector_name not in detectors:
        # Fallback to all-bull (mirror engine's default safety) — but we should
        # never hit this since legacy configs include the detector lambdas
        # in the engine's full _all_detectors map.
        # Try to access _all_detectors via a module-level fetch by building once.
        # Simpler: re-derive here by manually evaluating the detector on the indicator dict.
        from crypto_trading_system_ed import _build_regime_indicators_and_detectors as _b
        _ind, _det = _b(asset)
        # As a last resort, return all-bull
        idx = pd.date_range(test_start, test_end, freq='1h')
        return pd.Series(True, index=idx)

    fn = detectors[detector_name]
    idx = pd.date_range(test_start, test_end, freq='1h')
    out = pd.Series([fn(dt) for dt in idx], index=idx)
    return out


# =============================================================================
# Phase 3 — apply policy stack and simulate trades
# =============================================================================
def simulate_with_policy(bull_sig: pd.DataFrame, bear_sig: pd.DataFrame,
                         regime_bull: pd.Series, cfg: dict,
                         shield_override: dict = None,
                         gate_override: dict = None) -> dict:
    """Apply regime-aware policy stack and simulate trades.

    cfg is the parsed regime config block for ETH.
    shield_override: {'bull': True/False, 'bear': True/False} or None to use cfg
    gate_override: 'enabled' True/False or None

    Returns: {return_pct, max_dd_pct, cdar_5_pct, n_trades, win_rate,
              trades: [...], equity_curve: [...]}
    """
    bull_h = int(cfg['bull']['horizon'])
    bear_h = int(cfg['bear']['horizon'])
    bull_conf = float(cfg['bull']['min_confidence']) / 100.0
    bear_conf = float(cfg['bear']['min_confidence']) / 100.0
    min_sell_pnl = float(cfg.get('min_sell_pnl_pct', 0.5)) / 100.0
    max_hold_h = int(cfg.get('max_hold_hours', 10))

    if shield_override is not None:
        bull_shield = bool(shield_override.get('bull', cfg['bull'].get('hold_shield', False)))
        bear_shield = bool(shield_override.get('bear', cfg['bear'].get('hold_shield', False)))
    else:
        bull_shield = bool(cfg['bull'].get('hold_shield', False))
        bear_shield = bool(cfg['bear'].get('hold_shield', False))

    bull_rc = cfg['bull'].get('rally_cooldown', {})
    bear_rc = cfg['bear'].get('rally_cooldown', {})
    if gate_override is not None:
        gate_on = bool(gate_override)
        bull_gate_on = gate_on and bool(bull_rc.get('enabled', False))
        bear_gate_on = gate_on and bool(bear_rc.get('enabled', False))
    else:
        bull_gate_on = bool(bull_rc.get('enabled', False))
        bear_gate_on = bool(bear_rc.get('enabled', False))

    # Build joint timeline = union of bull and bear signal index, intersected with regime index
    idx = bull_sig.index.union(bear_sig.index).intersection(regime_bull.index)
    idx = pd.DatetimeIndex(sorted(set(idx)))
    if len(idx) == 0:
        return {'return_pct': 0.0, 'max_dd_pct': 0.0, 'cdar_5_pct': 0.0,
                'n_trades': 0, 'win_rate': 0.0, 'trades': [], 'equity_curve': []}

    # Pre-compute rally returns for gate (rolling lookback in % from window-low)
    # We use close from whichever signal frame has it (bull preferred)
    closes = bull_sig['close'].reindex(idx, method='ffill')
    if closes.isna().any():
        closes = closes.fillna(bear_sig['close'].reindex(idx, method='ffill'))

    def rolling_rr(close_series: pd.Series, hours: int) -> pd.Series:
        # rr at time t = (close[t] - min(close[t-hours+1 .. t])) / min(...)
        roll_min = close_series.rolling(hours, min_periods=1).min()
        return (close_series - roll_min) / roll_min

    bull_rr_short = rolling_rr(closes, int(bull_rc.get('h_short', 12))) if bull_gate_on else None
    bull_rr_long  = rolling_rr(closes, int(bull_rc.get('h_long', 30))) if bull_gate_on else None
    bear_rr_short = rolling_rr(closes, int(bear_rc.get('h_short', 12))) if bear_gate_on else None
    bear_rr_long  = rolling_rr(closes, int(bear_rc.get('h_long', 30))) if bear_gate_on else None

    bull_t_short = float(bull_rc.get('t_short_pct', 999)) / 100.0 if bull_gate_on else None
    bull_t_long  = float(bull_rc.get('t_long_pct', 999)) / 100.0 if bull_gate_on else None
    bull_cd_h    = int(bull_rc.get('cd_hours', 0)) if bull_gate_on else 0
    bear_t_short = float(bear_rc.get('t_short_pct', 999)) / 100.0 if bear_gate_on else None
    bear_t_long  = float(bear_rc.get('t_long_pct', 999)) / 100.0 if bear_gate_on else None
    bear_cd_h    = int(bear_rc.get('cd_hours', 0)) if bear_gate_on else 0

    # Walk through bars
    portfolio = 1.0
    in_pos = False
    entry_px = 0.0
    entry_t  = None
    cooldown_until = None  # datetime
    trades = []
    equity_curve = []  # list of (datetime, equity)
    peak = 1.0
    max_dd = 0.0
    dd_series = []

    for t in idx:
        is_bull = bool(regime_bull.loc[t]) if t in regime_bull.index else True
        # Pick horizon-specific signal
        sig_frame = bull_sig if is_bull else bear_sig
        conf_thr  = bull_conf if is_bull else bear_conf
        shield_on = bull_shield if is_bull else bear_shield
        gate_on   = bull_gate_on if is_bull else bear_gate_on
        rr_short  = bull_rr_short if is_bull else bear_rr_short
        rr_long   = bull_rr_long if is_bull else bear_rr_long
        t_short   = bull_t_short if is_bull else bear_t_short
        t_long    = bull_t_long if is_bull else bear_t_long
        cd_h      = bull_cd_h if is_bull else bear_cd_h

        if t not in sig_frame.index:
            # No signal this bar (model didn't fit) — carry equity, no action
            cur = portfolio * (closes.loc[t] / entry_px) if (in_pos and t in closes.index) else portfolio
            if cur > peak:
                peak = cur
            dd = (peak - cur) / peak
            if dd > max_dd:
                max_dd = dd
            dd_series.append(dd)
            equity_curve.append((t, cur))
            continue

        prob  = float(sig_frame.loc[t, 'prob'])
        price = float(sig_frame.loc[t, 'close'])

        # Update cooldown trigger (gate ON arms cooldown when rr exceeds threshold)
        if gate_on and rr_short is not None and t in rr_short.index:
            rs = float(rr_short.loc[t])
            rl = float(rr_long.loc[t]) if (rr_long is not None and t in rr_long.index) else 0.0
            if (not pd.isna(rs) and rs >= t_short) or (not pd.isna(rl) and rl >= t_long):
                cooldown_until = t + pd.Timedelta(hours=cd_h)

        # Determine model action: BUY if prob >= conf_thr else (no model BUY)
        # Engine policy is BUY=enter, SELL=exit, no signal = hold
        model_buy = prob >= conf_thr
        model_sell = prob < (1 - conf_thr)  # symmetric: low prob = SELL signal

        if not in_pos:
            # Currently in cash. Consider BUY.
            if model_buy:
                # Gate check
                if cooldown_until is not None and t < cooldown_until:
                    pass  # gate blocks BUY
                else:
                    entry_px = price * (1 + BACKTEST_FEE_PER_LEG)
                    entry_t = t
                    in_pos = True
        else:
            # Currently invested. Consider SELL.
            held_h = (t - entry_t).total_seconds() / 3600.0
            cur_pnl = (price - entry_px) / entry_px
            should_sell = False
            sell_reason = None
            # max_hold failsafe (always active)
            if held_h >= max_hold_h:
                should_sell = True
                sell_reason = 'max_hold'
            elif model_sell:
                if shield_on and cur_pnl < min_sell_pnl and held_h < max_hold_h:
                    pass  # shield blocks model SELL
                else:
                    should_sell = True
                    sell_reason = 'model'
            if should_sell:
                sell_px = price * (1 - BACKTEST_FEE_PER_LEG)
                r = (sell_px - entry_px) / entry_px
                portfolio *= (1 + r)
                trades.append({
                    'entry_time': entry_t,
                    'exit_time': t,
                    'held_h': round(held_h, 2),
                    'entry_px': entry_px,
                    'exit_px': sell_px,
                    'pnl_pct': r * 100,
                    'reason': sell_reason,
                    'regime': 'bull' if is_bull else 'bear',
                })
                in_pos = False
                entry_t = None
                entry_px = 0.0

        # Update equity + drawdown
        cur = portfolio * (price / entry_px) if in_pos else portfolio
        if cur > peak:
            peak = cur
        dd = (peak - cur) / peak
        if dd > max_dd:
            max_dd = dd
        dd_series.append(dd)
        equity_curve.append((t, cur))

    # Close any open position at the last bar
    if in_pos:
        last_close = float(closes.iloc[-1])
        sell_px = last_close * (1 - BACKTEST_FEE_PER_LEG)
        r = (sell_px - entry_px) / entry_px
        portfolio *= (1 + r)
        trades.append({
            'entry_time': entry_t,
            'exit_time': idx[-1],
            'held_h': (idx[-1] - entry_t).total_seconds() / 3600.0,
            'entry_px': entry_px,
            'exit_px': sell_px,
            'pnl_pct': r * 100,
            'reason': 'end_of_window',
            'regime': 'bull' if regime_bull.loc[idx[-1]] else 'bear',
        })

    if dd_series:
        dd_arr = np.array(dd_series)
        thresh = np.quantile(dd_arr, 0.95)
        worst = dd_arr[dd_arr >= thresh]
        cdar5 = float(worst.mean() * 100) if len(worst) > 0 else max_dd * 100
    else:
        cdar5 = max_dd * 100

    win_rate = (sum(1 for tr in trades if tr['pnl_pct'] > 0) / len(trades) * 100) if trades else 0.0

    return {
        'return_pct': (portfolio - 1) * 100,
        'max_dd_pct': max_dd * 100,
        'cdar_5_pct': cdar5,
        'n_trades': len(trades),
        'win_rate': win_rate,
        'trades': trades,
        'equity_curve': equity_curve,
    }


# =============================================================================
# Phase 4 — orchestration
# =============================================================================
def parse_config(cfg_path: str) -> dict:
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    eth = cfg.get('ETH', cfg)  # support both top-level and nested
    return eth


def get_model_spec(prod_csv_path: str, asset: str, horizon: int) -> dict:
    df = pd.read_csv(prod_csv_path)
    rows = df[(df['coin'] == asset) & (df['horizon'] == horizon)]
    if len(rows) == 0:
        raise RuntimeError(f'No row for {asset} h={horizon} in {prod_csv_path}')
    # Take first row (production CSV typically has unique (coin, horizon))
    r = rows.iloc[0]
    combo_str = str(r['models'])
    # Column is 'best_window' across all production CSVs (engine writes it that way).
    win_col = 'best_window' if 'best_window' in df.columns else 'window'
    return {
        'combo': combo_str.split('+'),
        'window': int(r[win_col]),
        'gamma': float(r['gamma']),
        'n_features': int(r['n_features']),
    }


def run_one_config(label: str, cfg_path: str, prod_csv_path: str) -> dict:
    """Generate signals + apply policy for one config; return result dict."""
    _log('=' * 100)
    _log(f'CONFIG: {label}')
    _log(f'  config: {cfg_path}')
    _log(f'  prod_csv: {prod_csv_path}')
    _log('=' * 100)
    cfg = parse_config(cfg_path)
    _log(f'  detector: {cfg.get("regime_detector", {}).get("params", {}).get("name", "?")}')
    _log(f'  bull: h={cfg["bull"]["horizon"]} conf={cfg["bull"]["min_confidence"]}% '
         f'shield={cfg["bull"].get("hold_shield", False)} '
         f'gate={cfg["bull"].get("rally_cooldown", {}).get("enabled", False)}')
    _log(f'  bear: h={cfg["bear"]["horizon"]} conf={cfg["bear"]["min_confidence"]}% '
         f'shield={cfg["bear"].get("hold_shield", False)} '
         f'gate={cfg["bear"].get("rally_cooldown", {}).get("enabled", False)}')
    _log(f'  shared: min_sell_pnl={cfg.get("min_sell_pnl_pct", "?")}% '
         f'max_hold={cfg.get("max_hold_hours", "?")}h')

    bull_h = int(cfg['bull']['horizon'])
    bear_h = int(cfg['bear']['horizon'])

    bull_spec = get_model_spec(prod_csv_path, ASSET, bull_h)
    bear_spec = get_model_spec(prod_csv_path, ASSET, bear_h) if bear_h != bull_h else bull_spec

    _log(f'  bull spec: {"+".join(bull_spec["combo"])} w={bull_spec["window"]} '
         f'g={bull_spec["gamma"]:.4f} f={bull_spec["n_features"]}')
    if bear_h != bull_h:
        _log(f'  bear spec: {"+".join(bear_spec["combo"])} w={bear_spec["window"]} '
             f'g={bear_spec["gamma"]:.4f} f={bear_spec["n_features"]}')

    _log(f'  >> generating bull signals (h={bull_h}) over {TEST_START} → {TEST_END} ...')
    t0 = datetime.now()
    bull_sig = generate_hourly_signals(ASSET, bull_h, bull_spec, TEST_START, TEST_END)
    _log(f'     done in {(datetime.now()-t0).total_seconds()/60:.1f} min, {len(bull_sig)} bars')

    if bear_h != bull_h:
        _log(f'  >> generating bear signals (h={bear_h}) ...')
        t0 = datetime.now()
        bear_sig = generate_hourly_signals(ASSET, bear_h, bear_spec, TEST_START, TEST_END)
        _log(f'     done in {(datetime.now()-t0).total_seconds()/60:.1f} min, {len(bear_sig)} bars')
    else:
        bear_sig = bull_sig

    detector_name = cfg.get('regime_detector', {}).get('params', {}).get('name', 'tsmom_672h')
    _log(f'  >> applying detector "{detector_name}" + regime merge + policy ...')
    regime_bull = get_regime_for_window(ASSET, detector_name, TEST_START, TEST_END)
    bull_pct = float(regime_bull.mean() * 100)
    _log(f'     regime split: {bull_pct:.1f}% bull / {100-bull_pct:.1f}% bear')

    result = simulate_with_policy(bull_sig, bear_sig, regime_bull, cfg)
    _log(f'  RESULT: ret={result["return_pct"]:+.2f}%  max_dd={result["max_dd_pct"]:.2f}%  '
         f'cdar5={result["cdar_5_pct"]:.2f}%  trades={result["n_trades"]}  WR={result["win_rate"]:.0f}%')
    _log('')

    return {
        'label': label,
        'cfg': cfg,
        'bull_sig': bull_sig,
        'bear_sig': bear_sig,
        'regime_bull': regime_bull,
        'detector': detector_name,
        'baseline': result,
    }


def run_ablations(cfg_data: dict) -> dict:
    """Run shield ON/OFF × gate ON/OFF on the leader config (cheap reapply)."""
    cfg = cfg_data['cfg']
    bull_sig = cfg_data['bull_sig']
    bear_sig = cfg_data['bear_sig']
    regime_bull = cfg_data['regime_bull']
    out = {}
    variants = [
        ('shield_OFF gate_OFF', {'bull': False, 'bear': False}, False),
        ('shield_OFF gate_ON ', {'bull': False, 'bear': False}, True),
        ('shield_ON  gate_OFF', {'bull': True,  'bear': True},  False),
        ('shield_ON  gate_ON ', {'bull': True,  'bear': True},  True),
    ]
    _log('=' * 100)
    _log(f'ABLATIONS on leader: {cfg_data["label"]}')
    _log('=' * 100)
    for name, sh, gt in variants:
        r = simulate_with_policy(bull_sig, bear_sig, regime_bull, cfg,
                                  shield_override=sh, gate_override=gt)
        out[name] = r
        _log(f'  {name}: ret={r["return_pct"]:+.2f}%  dd={r["max_dd_pct"]:.2f}%  '
             f'cdar5={r["cdar_5_pct"]:.2f}%  trades={r["n_trades"]}  WR={r["win_rate"]:.0f}%')
    _log('')
    return out


def main():
    _log('=' * 100)
    _log(f'5-CONFIG COUNTERFACTUAL BACKTEST — {datetime.now().isoformat()}')
    _log(f'Asset: {ASSET}  Window: {TEST_START} → {TEST_END}  '
         f'Retrain step: {RETRAIN_STEP}h')
    _log('=' * 100)
    _log('')

    all_data = []
    for label, cfg_path, prod_csv_path in CONFIGS:
        if not os.path.exists(cfg_path):
            _log(f'SKIP {label}: config missing at {cfg_path}')
            continue
        if not os.path.exists(prod_csv_path):
            _log(f'SKIP {label}: prod CSV missing at {prod_csv_path}')
            continue
        try:
            data = run_one_config(label, cfg_path, prod_csv_path)
            all_data.append(data)
        except Exception as e:
            _log(f'ERROR running {label}: {e!r}')
            import traceback
            _log(traceback.format_exc())

    if not all_data:
        _log('NO RESULTS — abort')
        return

    # Comparison table
    _log('=' * 100)
    _log('FINAL COMPARISON (all 5 configs on common Apr 10 → May 6 15:48 window)')
    _log('=' * 100)
    days = (TEST_END - TEST_START).total_seconds() / 86400.0
    rows = []
    for d in all_data:
        r = d['baseline']
        per_day = r['return_pct'] / days
        rows.append({
            'label': d['label'],
            'detector': d['detector'],
            'bull_h': d['cfg']['bull']['horizon'],
            'bear_h': d['cfg']['bear']['horizon'],
            'return_pct': r['return_pct'],
            'max_dd_pct': r['max_dd_pct'],
            'cdar_5_pct': r['cdar_5_pct'],
            'n_trades':   r['n_trades'],
            'win_rate':   r['win_rate'],
            'return_per_day': per_day,
            'risk_adjusted_pf': r['return_pct'] / max(r['max_dd_pct'], 0.01),
        })

    df = pd.DataFrame(rows).sort_values('return_pct', ascending=False)
    pd.set_option('display.width', 180)
    pd.set_option('display.max_columns', 20)
    _log(df.to_string(index=False))
    _log('')

    # Save comparison + all per-trade detail
    cmp_path = os.path.join(OUT_DIR, f'test_5_configs_{WINDOW_TAG}_{TS}.csv')
    df.to_csv(cmp_path, index=False)
    _log(f'  comparison CSV: {cmp_path}')

    trades_rows = []
    for d in all_data:
        for tr in d['baseline']['trades']:
            trades_rows.append({
                'config': d['label'],
                'detector': d['detector'],
                **tr,
            })
    if trades_rows:
        tr_path = os.path.join(OUT_DIR, f'test_5_configs_trades_{TS}.csv')
        pd.DataFrame(trades_rows).to_csv(tr_path, index=False)
        _log(f'  trades CSV:     {tr_path}')

    # Ablations on leader
    leader = all_data[0]  # before sort: original order — pick highest-return
    leader = max(all_data, key=lambda d: d['baseline']['return_pct'])
    _log('')
    _log(f'Leader by return: {leader["label"]}')
    abl = run_ablations(leader)
    abl_rows = []
    for name, r in abl.items():
        abl_rows.append({
            'leader': leader['label'],
            'variant': name,
            'return_pct': r['return_pct'],
            'max_dd_pct': r['max_dd_pct'],
            'cdar_5_pct': r['cdar_5_pct'],
            'n_trades':   r['n_trades'],
            'win_rate':   r['win_rate'],
        })
    abl_path = os.path.join(OUT_DIR, f'test_5_configs_ablations_{TS}.csv')
    pd.DataFrame(abl_rows).to_csv(abl_path, index=False)
    _log(f'  ablations CSV: {abl_path}')

    _log('')
    _log(f'DONE. Logs: {LOG_PATH}')


if __name__ == '__main__':
    main()
