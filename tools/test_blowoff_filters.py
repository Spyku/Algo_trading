"""
test_blowoff_filters.py — Backtest grid of "blow-off top" filters that block BUYs
when price has rallied too much.

Compares baseline (current ETH winner config) vs each filter combo.
Uses 3-fold rolling validation (train 60% / val 20% / stride 10%) like Mode H/V.
Caches generated signals to pickle so re-runs are instant.

Usage:
    python tools/test_blowoff_filters.py                # ETH, replay 2880, default grid
    python tools/test_blowoff_filters.py --asset ETH --replay 2880

Output:
    logs/blowoff_filter_sweep_<asset>_<timestamp>.csv  (full grid)
    Top 20 printed to console
"""
import os
import sys
import json
import pickle
import argparse
import csv
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

# Make engine importable
ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

from crypto_trading_system_ed import (  # noqa: E402
    generate_signals,
    _build_regime_indicators_and_detectors,
    _suppress_stderr,
    PRODUCTION_CSV,
    CONFIG_DIR,
    TRADING_FEE,
)


# ─────────────────────────── Filter definitions ───────────────────────────

def make_filter(kind, value):
    """Return a function f(dt, ind_row) -> True if BUY should be blocked.

    ind_row is the indicator row at dt (dict with sma72, sma168, close, pct_24h, pct_6h, rsi14, bb_pctb).
    """
    if kind == 'A_24h':
        return lambda r: r.get('pct_24h', 0) >= value
    if kind == 'A_6h':
        return lambda r: r.get('pct_6h', 0) >= value
    if kind == 'B1_rsi':
        return lambda r: r.get('rsi14', 0) >= value
    if kind == 'B2_sma72':
        return lambda r: r.get('close', 0) / r.get('sma72', 1) >= value if r.get('sma72') else False
    if kind == 'B3_sma168':
        return lambda r: r.get('close', 0) / r.get('sma168', 1) >= value if r.get('sma168') else False
    if kind == 'C_bb_pctb':
        return lambda r: r.get('bb_pctb', 0) >= value
    raise ValueError(f"Unknown filter kind: {kind}")


def combine_filters(*fns):
    return lambda r: any(fn(r) for fn in fns)


GRID = [
    ('A_24h',     [3, 4, 5, 6, 8, 10]),
    ('A_6h',      [2, 3, 4, 5]),
    ('B1_rsi',    [70, 75, 80, 85]),
    ('B2_sma72',  [1.03, 1.05, 1.07, 1.10]),
    ('B3_sma168', [1.05, 1.08, 1.12, 1.15]),
    ('C_bb_pctb', [0.80, 0.90, 1.00, 1.10]),
]

# Combo filter D — top of A + top of B
COMBOS = [
    ('D_A24_B2', [('A_24h', 4), ('B2_sma72', 1.05)]),
    ('D_A24_B1', [('A_24h', 5), ('B1_rsi', 80)]),
    ('D_A6_B2',  [('A_6h', 3),  ('B2_sma72', 1.05)]),
]

ACTIONS = ['block_buy', 'force_sell', 'delay_4h']


# ─────────────────────────── Indicator builder ───────────────────────────

def build_filter_indicators(asset):
    """Build per-hour dict of {dt: {close, sma72, sma168, pct_24h, pct_6h, rsi14, bb_pctb}}."""
    from crypto_trading_system_ed import load_data
    df = load_data(asset).copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()

    df['sma72'] = df['close'].rolling(72).mean()
    df['sma168'] = df['close'].rolling(168).mean()
    df['pct_24h'] = (df['close'] / df['close'].shift(24) - 1) * 100
    df['pct_6h'] = (df['close'] / df['close'].shift(6) - 1) * 100

    # RSI 14
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi14'] = 100 - 100 / (1 + rs)

    # Bollinger %B (20, 2)
    ma20 = df['close'].rolling(20).mean()
    sd20 = df['close'].rolling(20).std()
    upper = ma20 + 2 * sd20
    lower = ma20 - 2 * sd20
    df['bb_pctb'] = (df['close'] - lower) / (upper - lower).replace(0, np.nan)

    keep = ['close', 'sma72', 'sma168', 'pct_24h', 'pct_6h', 'rsi14', 'bb_pctb']
    return df[keep].to_dict('index')


# ─────────────────────────── Signal cache ───────────────────────────

def get_signals(asset, replay, force=False):
    """Generate or load cached signals for the current winner config (bull_h, bear_h)."""
    cache_path = os.path.join(ENGINE_DIR, 'logs', f'_blowoff_signals_{asset}_{replay}.pkl')
    if os.path.exists(cache_path) and not force:
        print(f"  Loading cached signals from {os.path.basename(cache_path)}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    with open(f'{CONFIG_DIR}/regime_config_ed.json') as f:
        cfg = json.load(f)
    bull_h = cfg[asset]['bull']['horizon']
    bear_h = cfg[asset]['bear']['horizon']
    bull_conf = cfg[asset]['bull']['min_confidence']
    bear_conf = cfg[asset]['bear']['min_confidence']
    det_name = cfg[asset]['regime_detector']['params']['name']

    print(f"  Config: detector={det_name} bull={bull_h}h@{bull_conf}% bear={bear_h}h@{bear_conf}%")

    df_models = pd.read_csv(PRODUCTION_CSV)
    signals_cache = {}
    for h in sorted(set([bull_h, bear_h])):
        rows = df_models[(df_models['coin'] == asset) & (df_models['horizon'] == h)]
        row = rows.sort_values('combined_score', ascending=False).iloc[0]
        feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
        gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0
        print(f"  Generating {h}h signals ({row['models']} w={int(row['best_window'])}h, replay={replay}h)...")
        with _suppress_stderr():
            sigs = generate_signals(asset, row['models'].split('+'),
                                    int(row['best_window']), replay,
                                    feature_override=feats, horizon=h, gamma=gamma)
        result = {}
        for s in sigs:
            dt = s['datetime']
            if isinstance(dt, str):
                dt = datetime.strptime(dt, '%Y-%m-%d %H:%M')
                s['datetime'] = dt
            result[dt] = s
        signals_cache[h] = result
        print(f"    {h}h: {len(result)} signals")

    bundle = {
        'signals': signals_cache,
        'bull_h': bull_h, 'bear_h': bear_h,
        'bull_conf': bull_conf, 'bear_conf': bear_conf,
        'det_name': det_name,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"  Cached → {os.path.basename(cache_path)}")
    return bundle


# ─────────────────────────── Simulator ───────────────────────────

def simulate(dts, signals_cache, det_fn, bull_h, bear_h, bull_conf, bear_conf,
             filter_fn, filter_ind, action='block_buy'):
    """Simulate a single strategy run on a slice of dts.

    filter_fn(ind_row) -> True if filter triggers.
    action ∈ {'block_buy', 'force_sell', 'delay_4h'}.
    Returns (return_pct, n_trades, win_rate, max_drawdown_pct).
    """
    cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
    trades, wins = 0, 0
    first_price, last_price = None, None
    delay_until = None  # for delay_4h
    equity_curve = []
    peak = 1000.0
    max_dd = 0.0

    for dt in dts:
        is_bull = det_fn(dt)
        h = bull_h if is_bull else bear_h
        conf = bull_conf if is_bull else bear_conf

        sigs = signals_cache.get(h)
        if sigs is None:
            continue
        s = sigs.get(dt)
        if s is None:
            continue

        price = s['close']
        last_price = price
        if first_price is None:
            first_price = price

        # equity tracking
        eq = cash + (held * price if in_pos else 0)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

        # filter check
        ind_row = filter_ind.get(dt, {})
        triggered = filter_fn(ind_row) if filter_fn else False

        sig = s['signal']
        cf = s['confidence']

        # Action: force_sell on trigger (overrides everything if in position)
        if triggered and action == 'force_sell' and in_pos:
            cash = held * price * (1 - TRADING_FEE)
            if price > entry_px:
                wins += 1
            held = 0
            in_pos = False
            continue

        if sig == 'BUY' and cf >= conf and not in_pos:
            if triggered:
                if action == 'block_buy':
                    continue
                if action == 'delay_4h':
                    delay_until = dt + pd.Timedelta(hours=4)
                    continue
                # force_sell already handled above; for BUY signal it just blocks (no pos to sell)
                if action == 'force_sell':
                    continue
            if delay_until is not None and dt < delay_until:
                continue
            delay_until = None
            held = cash * (1 - TRADING_FEE) / price
            cash = 0
            in_pos = True
            entry_px = price
            trades += 1
        elif sig == 'SELL' and in_pos:
            cash = held * price * (1 - TRADING_FEE)
            if price > entry_px:
                wins += 1
            held = 0
            in_pos = False

    if in_pos and last_price:
        cash = held * last_price * (1 - TRADING_FEE)
        if last_price > entry_px:
            wins += 1

    ret = (cash / 1000.0 - 1) * 100
    wr = (wins / trades * 100) if trades > 0 else 0
    return ret, trades, wr, max_dd


# ─────────────────────────── Main ───────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', default='ETH')
    ap.add_argument('--replay', type=int, default=2880)
    ap.add_argument('--force-regen', action='store_true')
    args = ap.parse_args()

    print(f"\n{'='*80}")
    print(f"  BLOW-OFF FILTER SWEEP — {args.asset} | replay {args.replay}h")
    print(f"{'='*80}\n")

    # 1. Get signals + config
    bundle = get_signals(args.asset, args.replay, force=args.force_regen)
    signals = bundle['signals']
    bull_h, bear_h = bundle['bull_h'], bundle['bear_h']
    bull_conf, bear_conf = bundle['bull_conf'], bundle['bear_conf']
    det_name = bundle['det_name']

    # 2. Build regime detector
    print(f"\n  Building regime detector + filter indicators...")
    _, detectors = _build_regime_indicators_and_detectors(args.asset)
    det_fn = detectors[det_name]
    filter_ind = build_filter_indicators(args.asset)

    all_dts = sorted(set().union(*[set(s.keys()) for s in signals.values()]))
    n = len(all_dts)
    print(f"  {n} timestamps total")

    # 3. Define 3-fold rolling holdout (train 60% / val 20% / stride 10%)
    folds = []
    for fi in range(3):
        train_s = int(n * fi * 0.10)
        train_e = int(n * (fi * 0.10 + 0.60))
        val_s = train_e
        val_e = int(n * (fi * 0.10 + 0.60 + 0.20))
        folds.append((val_s, val_e))
    print(f"  Folds (val windows): {folds}")

    # 4. Build filter list
    filter_specs = []  # list of (kind, value, fn)
    for kind, values in GRID:
        for v in values:
            filter_specs.append((kind, str(v), make_filter(kind, v)))
    for combo_name, parts in COMBOS:
        fns = [make_filter(k, v) for k, v in parts]
        param_str = '+'.join(f'{k}={v}' for k, v in parts)
        filter_specs.append((combo_name, param_str, combine_filters(*fns)))

    print(f"\n  {len(filter_specs)} filters × {len(ACTIONS)} actions × 3 folds = "
          f"{len(filter_specs)*len(ACTIONS)*3 + 3} sims")

    # 5. Baseline (no filter) per fold
    print(f"\n  Running baseline (no filter)...")
    baseline_per_fold = []
    for (vs, ve) in folds:
        dts_slice = all_dts[vs:ve]
        ret, tr, wr, dd = simulate(dts_slice, signals, det_fn, bull_h, bear_h,
                                   bull_conf, bear_conf, None, filter_ind)
        baseline_per_fold.append((ret, tr, wr, dd))
    base_mean_ret = np.mean([x[0] for x in baseline_per_fold])
    print(f"  Baseline mean return: {base_mean_ret:+.2f}%  per-fold: "
          f"{[f'{x[0]:+.1f}%' for x in baseline_per_fold]}")

    # 6. Sweep
    print(f"\n  Sweeping filters...")
    rows_out = []
    for (kind, val_str, fn) in filter_specs:
        for action in ACTIONS:
            fold_results = []
            for (vs, ve) in folds:
                dts_slice = all_dts[vs:ve]
                r = simulate(dts_slice, signals, det_fn, bull_h, bear_h,
                             bull_conf, bear_conf, fn, filter_ind, action=action)
                fold_results.append(r)
            rets = [r[0] for r in fold_results]
            trs = [r[1] for r in fold_results]
            wrs = [r[2] for r in fold_results]
            dds = [r[3] for r in fold_results]
            rows_out.append({
                'filter': kind,
                'params': val_str,
                'action': action,
                'fold1_ret': rets[0],
                'fold2_ret': rets[1],
                'fold3_ret': rets[2],
                'mean_ret': np.mean(rets),
                'std_ret': np.std(rets),
                'mean_trades': np.mean(trs),
                'mean_wr': np.mean(wrs),
                'mean_dd': np.mean(dds),
                'vs_baseline': np.mean(rets) - base_mean_ret,
            })

    # Add baseline row
    rows_out.append({
        'filter': 'BASELINE',
        'params': 'none',
        'action': 'none',
        'fold1_ret': baseline_per_fold[0][0],
        'fold2_ret': baseline_per_fold[1][0],
        'fold3_ret': baseline_per_fold[2][0],
        'mean_ret': base_mean_ret,
        'std_ret': np.std([x[0] for x in baseline_per_fold]),
        'mean_trades': np.mean([x[1] for x in baseline_per_fold]),
        'mean_wr': np.mean([x[2] for x in baseline_per_fold]),
        'mean_dd': np.mean([x[3] for x in baseline_per_fold]),
        'vs_baseline': 0.0,
    })

    rows_out.sort(key=lambda x: x['mean_ret'], reverse=True)

    # 7. Write CSV
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(ENGINE_DIR, 'logs', f'blowoff_filter_sweep_{args.asset}_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"\n  CSV written: {csv_path}")

    # 8. Top 20 print
    print(f"\n  {'='*110}")
    print(f"  TOP 20 — {args.asset} (baseline mean ret = {base_mean_ret:+.2f}%)")
    print(f"  {'='*110}")
    print(f"  {'#':>3}  {'Filter':>12}  {'Params':>22}  {'Action':>12}  "
          f"{'MeanRet':>9}  {'Std':>6}  {'Trades':>7}  {'WR':>5}  {'DD':>6}  {'vsBase':>8}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*22}  {'-'*12}  {'-'*9}  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*8}")
    for i, r in enumerate(rows_out[:20]):
        print(f"  {i+1:>3}  {r['filter']:>12}  {r['params']:>22}  {r['action']:>12}  "
              f"{r['mean_ret']:>+8.2f}%  {r['std_ret']:>5.1f}  {r['mean_trades']:>7.1f}  "
              f"{r['mean_wr']:>4.0f}%  {r['mean_dd']:>5.1f}%  {r['vs_baseline']:>+7.2f}%")
    print(f"  {'='*110}\n")


if __name__ == '__main__':
    main()
