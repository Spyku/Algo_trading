"""
test_momentum_decay.py — Smart blow-off detection via momentum decay.

Concept: instead of blocking BUYs by hard threshold, only act if (a) we've already
rallied X% on the day AND (b) momentum is decelerating (1st/2nd derivative signals).

Grid:
  rally_gate (pct_24h >= X%)  : 2, 3, 4, 5
  momentum_signal:
    M1_slope_flat   : slope_3h < slope_6h * 0.5
    M2_accel_neg    : discrete 2nd derivative < 0
    M3_below_peak6  : close < max(close[-6:]) * 0.995
    M4_below_peak3  : close < max(close[-3:]) * 0.998
    M5_roc_decay    : roc_3h < roc_6h < roc_12h

Actions: block_buy, force_sell, tighten_conf (require >=95%), wait_confirm (need pct_1h>0).

Uses 3-fold rolling validation (val 60-80, 70-90, 80-100% of replay).
Caches signals to pickle.

Usage:
    python tools/test_momentum_decay.py --asset ETH --replay 2880
"""
import os
import sys
import json
import pickle
import argparse
import csv
from datetime import datetime

import numpy as np
import pandas as pd

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


# ─────────────────────────── Indicator builder ───────────────────────────

def build_momentum_indicators(asset):
    """Build per-hour dict with derivative-based features.

    For each dt:
      close, pct_24h, pct_1h, pct_3h, pct_6h, pct_12h
      slope_3h, slope_6h          (avg pct return per hour over window)
      accel                       (discrete d²: c[t] - 2*c[t-3] + c[t-6])
      max_close_3h, max_close_6h  (rolling max for 'below_peak' check)
      roc_3h, roc_6h, roc_12h     (rate of change)
    """
    from crypto_trading_system_ed import load_data
    df = load_data(asset).copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()

    df['pct_1h']  = (df['close'] / df['close'].shift(1)  - 1) * 100
    df['pct_3h']  = (df['close'] / df['close'].shift(3)  - 1) * 100
    df['pct_6h']  = (df['close'] / df['close'].shift(6)  - 1) * 100
    df['pct_12h'] = (df['close'] / df['close'].shift(12) - 1) * 100
    df['pct_24h'] = (df['close'] / df['close'].shift(24) - 1) * 100

    # Slopes (% per hour)
    df['slope_3h'] = df['pct_3h'] / 3.0
    df['slope_6h'] = df['pct_6h'] / 6.0

    # Discrete second derivative on price (c[t] - 2*c[t-3] + c[t-6])
    df['accel'] = df['close'] - 2 * df['close'].shift(3) + df['close'].shift(6)

    # Rolling max for "below peak"
    df['max_close_3h'] = df['close'].rolling(3).max()
    df['max_close_6h'] = df['close'].rolling(6).max()

    # Rate of change (same as pct_*, kept named for clarity)
    df['roc_3h'] = df['pct_3h']
    df['roc_6h'] = df['pct_6h']
    df['roc_12h'] = df['pct_12h']

    keep = ['close', 'pct_1h', 'pct_3h', 'pct_6h', 'pct_12h', 'pct_24h',
            'slope_3h', 'slope_6h', 'accel',
            'max_close_3h', 'max_close_6h',
            'roc_3h', 'roc_6h', 'roc_12h']
    return df[keep].to_dict('index')


# ─────────────────────────── Momentum signals ───────────────────────────

def m1_slope_flat(r):
    s3, s6 = r.get('slope_3h'), r.get('slope_6h')
    if s3 is None or s6 is None or pd.isna(s3) or pd.isna(s6):
        return False
    return s3 < s6 * 0.5

def m2_accel_neg(r):
    a = r.get('accel')
    if a is None or pd.isna(a):
        return False
    return a < 0

def m3_below_peak6(r):
    c, mx = r.get('close'), r.get('max_close_6h')
    if c is None or mx is None or pd.isna(c) or pd.isna(mx) or mx == 0:
        return False
    return c < mx * 0.995

def m4_below_peak3(r):
    c, mx = r.get('close'), r.get('max_close_3h')
    if c is None or mx is None or pd.isna(c) or pd.isna(mx) or mx == 0:
        return False
    return c < mx * 0.998

def m5_roc_decay(r):
    r3, r6, r12 = r.get('roc_3h'), r.get('roc_6h'), r.get('roc_12h')
    if any(x is None or pd.isna(x) for x in (r3, r6, r12)):
        return False
    return r3 < r6 < r12

MOMENTUM_SIGNALS = {
    'M1_slope_flat':  m1_slope_flat,
    'M2_accel_neg':   m2_accel_neg,
    'M3_below_peak6': m3_below_peak6,
    'M4_below_peak3': m4_below_peak3,
    'M5_roc_decay':   m5_roc_decay,
}

GATES = [2, 3, 4, 5]   # pct_24h >= gate
ACTIONS = ['block_buy', 'force_sell', 'tighten_conf', 'wait_confirm']
TIGHTEN_CONF = 95


def make_filter(gate, momentum_fn):
    def fn(r):
        p24 = r.get('pct_24h')
        if p24 is None or pd.isna(p24):
            return False
        if p24 < gate:
            return False
        return momentum_fn(r)
    return fn


# ─────────────────────────── Signal cache ───────────────────────────

def get_signals(asset, replay, force=False):
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

    bundle = {
        'signals': signals_cache,
        'bull_h': bull_h, 'bear_h': bear_h,
        'bull_conf': bull_conf, 'bear_conf': bear_conf,
        'det_name': det_name,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(bundle, f)
    return bundle


# ─────────────────────────── Simulator ───────────────────────────

def simulate(dts, signals_cache, det_fn, bull_h, bear_h, bull_conf, bear_conf,
             filter_fn, filter_ind, action='block_buy'):
    cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
    trades, wins = 0, 0
    first_price, last_price = None, None
    peak = 1000.0
    max_dd = 0.0
    n_triggered_buy = 0  # diagnostic: how many BUY signals were intercepted

    for dt in dts:
        is_bull = det_fn(dt)
        h = bull_h if is_bull else bear_h
        conf_req = bull_conf if is_bull else bear_conf

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

        eq = cash + (held * price if in_pos else 0)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

        ind_row = filter_ind.get(dt, {})
        triggered = filter_fn(ind_row) if filter_fn else False

        sig = s['signal']
        cf = s['confidence']

        # force_sell on trigger (overrides everything if in position)
        if triggered and action == 'force_sell' and in_pos:
            cash = held * price * (1 - TRADING_FEE)
            if price > entry_px:
                wins += 1
            held = 0
            in_pos = False
            continue

        if sig == 'BUY' and not in_pos:
            local_conf_req = conf_req
            if triggered:
                if action == 'block_buy':
                    n_triggered_buy += 1
                    continue
                if action == 'wait_confirm':
                    p1h = ind_row.get('pct_1h')
                    if p1h is None or pd.isna(p1h) or p1h <= 0:
                        n_triggered_buy += 1
                        continue
                if action == 'tighten_conf':
                    local_conf_req = max(conf_req, TIGHTEN_CONF)
                if action == 'force_sell':
                    n_triggered_buy += 1
                    continue
            if cf >= local_conf_req:
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
    return ret, trades, wr, max_dd, n_triggered_buy


# ─────────────────────────── Main ───────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', default='ETH')
    ap.add_argument('--replay', type=int, default=2880)
    ap.add_argument('--force-regen', action='store_true')
    args = ap.parse_args()

    print(f"\n{'='*80}")
    print(f"  MOMENTUM DECAY FILTER SWEEP — {args.asset} | replay {args.replay}h")
    print(f"{'='*80}\n")

    bundle = get_signals(args.asset, args.replay, force=args.force_regen)
    signals = bundle['signals']
    bull_h, bear_h = bundle['bull_h'], bundle['bear_h']
    bull_conf, bear_conf = bundle['bull_conf'], bundle['bear_conf']
    det_name = bundle['det_name']

    print(f"\n  Building regime detector + momentum indicators...")
    _, detectors = _build_regime_indicators_and_detectors(args.asset)
    det_fn = detectors[det_name]
    filter_ind = build_momentum_indicators(args.asset)

    all_dts = sorted(set().union(*[set(s.keys()) for s in signals.values()]))
    n = len(all_dts)
    print(f"  {n} timestamps total")

    folds = []
    for fi in range(3):
        val_s = int(n * (fi * 0.10 + 0.60))
        val_e = int(n * (fi * 0.10 + 0.60 + 0.20))
        folds.append((val_s, val_e))
    print(f"  Folds (val windows): {folds}")

    # Build filter list
    filter_specs = []
    for gate in GATES:
        for m_name, m_fn in MOMENTUM_SIGNALS.items():
            label = f'gate{gate}_{m_name}'
            param_str = f'pct24>={gate}% AND {m_name}'
            filter_specs.append((label, param_str, make_filter(gate, m_fn)))

    print(f"\n  {len(filter_specs)} filters × {len(ACTIONS)} actions × 3 folds = "
          f"{len(filter_specs)*len(ACTIONS)*3 + 3} sims")

    # Baseline
    print(f"\n  Running baseline (no filter)...")
    baseline_per_fold = []
    for (vs, ve) in folds:
        dts_slice = all_dts[vs:ve]
        r = simulate(dts_slice, signals, det_fn, bull_h, bear_h,
                     bull_conf, bear_conf, None, filter_ind)
        baseline_per_fold.append(r)
    base_mean_ret = np.mean([x[0] for x in baseline_per_fold])
    print(f"  Baseline mean return: {base_mean_ret:+.2f}%  per-fold: "
          f"{[f'{x[0]:+.1f}%' for x in baseline_per_fold]}")

    # Sweep
    print(f"\n  Sweeping...")
    rows_out = []
    for (label, param_str, fn) in filter_specs:
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
            triggers = [r[4] for r in fold_results]
            rows_out.append({
                'filter': label,
                'params': param_str,
                'action': action,
                'fold1_ret': rets[0],
                'fold2_ret': rets[1],
                'fold3_ret': rets[2],
                'mean_ret': np.mean(rets),
                'std_ret': np.std(rets),
                'mean_trades': np.mean(trs),
                'mean_wr': np.mean(wrs),
                'mean_dd': np.mean(dds),
                'mean_triggers': np.mean(triggers),
                'vs_baseline': np.mean(rets) - base_mean_ret,
            })

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
        'mean_triggers': 0,
        'vs_baseline': 0.0,
    })

    rows_out.sort(key=lambda x: x['mean_ret'], reverse=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(ENGINE_DIR, 'logs', f'momentum_decay_sweep_{args.asset}_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"\n  CSV written: {csv_path}")

    print(f"\n  {'='*120}")
    print(f"  TOP 20 — {args.asset} (baseline mean ret = {base_mean_ret:+.2f}%)")
    print(f"  {'='*120}")
    print(f"  {'#':>3}  {'Filter':>22}  {'Action':>14}  {'MeanRet':>9}  {'Std':>6}  "
          f"{'Trades':>7}  {'WR':>5}  {'DD':>6}  {'Trig':>5}  {'vsBase':>8}")
    print(f"  {'-'*3}  {'-'*22}  {'-'*14}  {'-'*9}  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*8}")
    for i, r in enumerate(rows_out[:20]):
        print(f"  {i+1:>3}  {r['filter']:>22}  {r['action']:>14}  "
              f"{r['mean_ret']:>+8.2f}%  {r['std_ret']:>5.1f}  {r['mean_trades']:>7.1f}  "
              f"{r['mean_wr']:>4.0f}%  {r['mean_dd']:>5.1f}%  {r['mean_triggers']:>5.1f}  "
              f"{r['vs_baseline']:>+7.2f}%")
    print(f"  {'='*120}\n")


if __name__ == '__main__':
    main()
