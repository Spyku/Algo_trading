"""
rally_signal_precision.py — measure precision/recall of candidate "rally
precursor" features. For each feature, find every hour where it sat at or
below a percentile threshold and check whether a >=3% rally started within
the next N hours. Reports precision, recall, lift over base rate, and
combo signals (multiple features simultaneously oversold).
"""

import os
import sys
import argparse
from datetime import datetime, timezone, timedelta
from itertools import combinations

import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

from crypto_trading_system_ed import build_all_features, load_data

PRICE_PATH = os.path.join(ENGINE_DIR, 'data', 'eth_hourly_data.csv')

# The 9 candidate features identified in the rally_feature_scan
CANDIDATES = [
    'price_to_sma20h',
    'bb_position_20h',
    'zscore_50h',
    'rsi_14h',
    'price_velocity_4h',
    'logret_3h',
    'logret_4h',
    'logret_5h',
    'logret_8h',
    'price_to_sma50h',
    'stoch_k_14h',
]


def find_rallies(df, min_pct=3.0, max_hours=72, swing_window=4):
    highs = df['high'].values
    lows = df['low'].values
    times = df.index.to_list()
    n = len(df)
    candidates = []
    for i in range(n - 1):
        lo_l = max(0, i - swing_window)
        lo_r = min(n, i + swing_window + 1)
        if lows[i] != lows[lo_l:lo_r].min():
            continue
        end = min(n, i + max_hours + 1)
        wh = highs[i+1:end]
        if len(wh) == 0:
            continue
        peak_offset = int(np.argmax(wh))
        peak_idx = i + 1 + peak_offset
        peak_price = highs[peak_idx]
        start_price = lows[i]
        pct = 100 * (peak_price - start_price) / start_price
        if pct < min_pct:
            continue
        candidates.append({'start_idx': i, 'peak_idx': peak_idx,
                           'start_dt': times[i], 'pct': pct})
    candidates.sort(key=lambda r: r['pct'], reverse=True)
    chosen = []
    used = np.zeros(n, dtype=bool)
    for c in candidates:
        s, e = c['start_idx'], c['peak_idx']
        if used[s:e+1].any():
            continue
        chosen.append(c)
        used[s:e+1] = True
    chosen.sort(key=lambda r: r['start_dt'])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=30)
    ap.add_argument('--min-pct', type=float, default=3.0)
    ap.add_argument('--max-hours', type=int, default=72)
    ap.add_argument('--horizon', type=int, default=6)
    ap.add_argument('--lookahead', type=int, default=72,
                    help='hours after signal to check for rally start')
    ap.add_argument('--threshold', type=float, default=10.0,
                    help='percentile threshold (signal fires if feature <= this)')
    ap.add_argument('--debounce', type=int, default=12,
                    help='dedupe nearby signal fires (suppress within N hours)')
    args = ap.parse_args()

    print(f'\nLoading ETH data + building feature matrix (horizon={args.horizon}h)...')
    df_raw = load_data('ETH')
    df, all_cols = build_all_features(df_raw, asset_name='ETH', horizon=args.horizon,
                                       verbose=False, keep_label_nan_tail=True)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    dfw = df[df.index >= cutoff].copy()
    print(f'  Rows in last {args.days}d: {len(dfw)}')

    # Find rallies
    price_df = pd.read_csv(PRICE_PATH)
    price_df['datetime'] = pd.to_datetime(price_df['datetime'], utc=True)
    price_df = price_df.set_index('datetime').sort_index()
    pw = price_df[price_df.index >= cutoff]
    rallies = find_rallies(pw, args.min_pct, args.max_hours)
    print(f'  Rallies >= {args.min_pct}% detected: {len(rallies)}\n')

    # Mark rally-start hours and build a "rally within next lookahead" mask
    rally_start_times = set()
    for r in rallies:
        rally_start_times.add(r['start_dt'])

    n = len(dfw)
    # rally_within[i] = True if any rally starts in (t_i, t_i + lookahead]
    rally_within = np.zeros(n, dtype=bool)
    times = dfw.index.to_list()
    for r in rallies:
        # find first row index where t > r['start_dt'] - lookahead
        rally_dt = r['start_dt']
        for i in range(n):
            t = times[i]
            if t < rally_dt and (rally_dt - t).total_seconds() / 3600 <= args.lookahead:
                rally_within[i] = True
            elif t == rally_dt:
                rally_within[i] = True

    base_rate = rally_within.sum() / n
    print(f'  Base rate (random hour leads to rally within {args.lookahead}h): {100*base_rate:.1f}%\n')

    # Compute percentile rank of each feature for every row in window
    print(f'  Computing percentile thresholds for {len(CANDIDATES)} candidates...')
    feat_pct = {}
    for f in CANDIDATES:
        if f not in dfw.columns:
            print(f'    WARN: {f} not in feature matrix')
            continue
        vals = dfw[f].values
        # Use rank-based percentile
        ser = pd.Series(vals).rank(pct=True) * 100
        feat_pct[f] = ser.values

    print(f'\n{"="*120}')
    print(f'  SINGLE-FEATURE PRECISION (signal: feature <= {args.threshold}th pctile)')
    print(f'  Window: {args.days}d | Lookahead: {args.lookahead}h | Debounce: {args.debounce}h | Base rate: {100*base_rate:.1f}%')
    print(f'{"="*120}\n')

    results = []
    for f, pcts in feat_pct.items():
        signals = pcts <= args.threshold
        # Apply debounce: if signal fired within last `debounce` hours, suppress
        debounced = np.zeros(n, dtype=bool)
        last_fire = -10**9
        for i in range(n):
            if signals[i] and (i - last_fire) >= args.debounce:
                debounced[i] = True
                last_fire = i
        n_sig = int(debounced.sum())
        if n_sig == 0:
            continue
        n_hits = int(np.logical_and(debounced, rally_within).sum())
        n_miss = n_sig - n_hits
        precision = n_hits / n_sig if n_sig > 0 else 0
        # recall = of the rallies, how many had this feature signal in their lookback window
        recall_hits = 0
        for r in rallies:
            rally_dt = r['start_dt']
            for i in range(n):
                t = times[i]
                if t > rally_dt:
                    break
                lookback_h = (rally_dt - t).total_seconds() / 3600
                if 0 <= lookback_h <= args.lookahead and debounced[i]:
                    recall_hits += 1
                    break
        recall = recall_hits / len(rallies) if rallies else 0
        lift = precision / base_rate if base_rate > 0 else 0
        results.append({
            'feature': f,
            'n_signals': n_sig,
            'n_hits': n_hits,
            'n_miss': n_miss,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'lift_vs_baserate': round(lift, 2),
        })

    rdf = pd.DataFrame(results).sort_values('precision', ascending=False)
    print(rdf.to_string(index=False))

    # Combos: for each pair, signal fires when BOTH features are <= threshold
    print(f'\n{"="*120}')
    print(f'  PAIR COMBO PRECISION (signal: feature_A <= {args.threshold}th AND feature_B <= {args.threshold}th)')
    print(f'  Top 25 by precision (min 3 signals)')
    print(f'{"="*120}\n')

    combo_results = []
    feats = list(feat_pct.keys())
    for a, b in combinations(feats, 2):
        sig_a = feat_pct[a] <= args.threshold
        sig_b = feat_pct[b] <= args.threshold
        sig = np.logical_and(sig_a, sig_b)
        # debounce
        debounced = np.zeros(n, dtype=bool)
        last_fire = -10**9
        for i in range(n):
            if sig[i] and (i - last_fire) >= args.debounce:
                debounced[i] = True
                last_fire = i
        n_sig = int(debounced.sum())
        if n_sig < 3:
            continue
        n_hits = int(np.logical_and(debounced, rally_within).sum())
        precision = n_hits / n_sig
        recall_hits = 0
        for r in rallies:
            rally_dt = r['start_dt']
            for i in range(n):
                t = times[i]
                if t > rally_dt:
                    break
                lookback_h = (rally_dt - t).total_seconds() / 3600
                if 0 <= lookback_h <= args.lookahead and debounced[i]:
                    recall_hits += 1
                    break
        recall = recall_hits / len(rallies)
        combo_results.append({
            'feature_a': a, 'feature_b': b,
            'n_signals': n_sig, 'n_hits': n_hits,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'lift': round(precision / base_rate, 2) if base_rate > 0 else 0,
        })

    cdf = pd.DataFrame(combo_results).sort_values(['precision', 'n_hits'], ascending=False).head(25)
    print(cdf.to_string(index=False))

    # Triple combos
    print(f'\n{"="*120}')
    print(f'  TRIPLE COMBO PRECISION (signal: 3 features ALL <= {args.threshold}th)')
    print(f'  Top 15 by precision (min 3 signals)')
    print(f'{"="*120}\n')

    trip_results = []
    for a, b, c in combinations(feats, 3):
        sig = np.logical_and.reduce([feat_pct[a] <= args.threshold,
                                      feat_pct[b] <= args.threshold,
                                      feat_pct[c] <= args.threshold])
        debounced = np.zeros(n, dtype=bool)
        last_fire = -10**9
        for i in range(n):
            if sig[i] and (i - last_fire) >= args.debounce:
                debounced[i] = True
                last_fire = i
        n_sig = int(debounced.sum())
        if n_sig < 3:
            continue
        n_hits = int(np.logical_and(debounced, rally_within).sum())
        precision = n_hits / n_sig
        recall_hits = 0
        for r in rallies:
            rally_dt = r['start_dt']
            for i in range(n):
                t = times[i]
                if t > rally_dt:
                    break
                lookback_h = (rally_dt - t).total_seconds() / 3600
                if 0 <= lookback_h <= args.lookahead and debounced[i]:
                    recall_hits += 1
                    break
        recall = recall_hits / len(rallies)
        trip_results.append({
            'features': f'{a} & {b} & {c}',
            'n_signals': n_sig, 'n_hits': n_hits,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'lift': round(precision / base_rate, 2) if base_rate > 0 else 0,
        })

    tdf = pd.DataFrame(trip_results).sort_values(['precision', 'n_hits'], ascending=False).head(15)
    print(tdf.to_string(index=False))


if __name__ == '__main__':
    main()
