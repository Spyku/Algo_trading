"""
rally_feature_scan.py — for each ETH rally >=3% in the last N days, snapshot
the feature matrix at t-6h, t-3h, t-1h, t=0 and compute percentile rank vs
the full N-day distribution. Surface features that consistently sit at
extreme percentiles before rally starts.

Cross-references findings against the production bull-horizon model's
selected features (from crypto_ed_production.csv) to identify potentially
useful features that are NOT being used.
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

from crypto_trading_system_ed import build_all_features, load_data

PRICE_PATH = os.path.join(ENGINE_DIR, 'data', 'eth_hourly_data.csv')
PROD_CSV = os.path.join(ENGINE_DIR, 'models', 'crypto_ed_production.csv')


def find_rallies(df, min_pct=3.0, max_hours=72, swing_window=4):
    closes = df['close'].values
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
                           'start_dt': times[i], 'peak_dt': times[peak_idx],
                           'pct': pct})
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


def percentile_rank(value, distribution):
    """Return percentile rank (0-100) of value vs distribution (NaN-safe)."""
    arr = distribution[~np.isnan(distribution)]
    if len(arr) == 0 or np.isnan(value):
        return np.nan
    return 100 * (arr < value).sum() / len(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=30)
    ap.add_argument('--min-pct', type=float, default=3.0)
    ap.add_argument('--max-hours', type=int, default=72)
    ap.add_argument('--horizon', type=int, default=6)
    ap.add_argument('--lags', type=str, default='-6,-3,-1,0',
                    help='Hours before rally start to snapshot')
    ap.add_argument('--top-n', type=int, default=30,
                    help='Top features to display per ranking')
    args = ap.parse_args()

    lags = [int(x) for x in args.lags.split(',')]

    print(f'\nLoading ETH data + building feature matrix (horizon={args.horizon}h)...')
    df_raw = load_data('ETH')
    df, all_cols = build_all_features(df_raw, asset_name='ETH', horizon=args.horizon,
                                       verbose=False, keep_label_nan_tail=True)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()
    print(f'  Total feature columns: {len(all_cols)}')
    print(f'  Total rows: {len(df)}')

    # Restrict to last N days for rally detection AND distribution
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    dfw = df[df.index >= cutoff]
    print(f'  Rows in last {args.days}d: {len(dfw)}')

    # Detect rallies
    price_df = pd.read_csv(PRICE_PATH)
    price_df['datetime'] = pd.to_datetime(price_df['datetime'], utc=True)
    price_df = price_df.set_index('datetime').sort_index()
    price_w = price_df[price_df.index >= cutoff]
    rallies = find_rallies(price_w, args.min_pct, args.max_hours)
    print(f'  Rallies >= {args.min_pct}% detected: {len(rallies)}\n')

    if not rallies:
        print('No rallies in window.')
        return

    # Build percentile-rank table: (feature, lag) -> [pct_rank for each rally]
    feature_cols = [c for c in all_cols if c in dfw.columns]
    print(f'  Computing percentile ranks for {len(feature_cols)} features '
          f'across {len(rallies)} rallies x {len(lags)} lags...')

    # Snapshots: pct_ranks[feat][lag] = list of percentile ranks across rallies
    snapshots = {feat: {lag: [] for lag in lags} for feat in feature_cols}
    raw_values = {feat: {lag: [] for lag in lags} for feat in feature_cols}

    for r in rallies:
        for lag in lags:
            target_dt = r['start_dt'] + timedelta(hours=lag)
            # Find nearest available row at or before target
            available = dfw.index[dfw.index <= target_dt]
            if len(available) == 0:
                continue
            row_dt = available[-1]
            row = dfw.loc[row_dt]
            for feat in feature_cols:
                if feat not in row.index:
                    continue
                val = row[feat]
                if pd.isna(val):
                    continue
                dist = dfw[feat].values
                pr = percentile_rank(val, dist)
                snapshots[feat][lag].append(pr)
                raw_values[feat][lag].append(val)

    # Compute per-feature scoring: features at extreme percentiles
    # consistently across rallies. Score = mean abs(pct_rank - 50) * count
    # weighted by consistency (low std of pct_rank).
    rows = []
    for feat in feature_cols:
        for lag in lags:
            ranks = snapshots[feat][lag]
            if len(ranks) < max(3, len(rallies) - 2):
                continue  # need most rallies represented
            arr = np.array(ranks)
            extremity = np.mean(np.abs(arr - 50))  # 0-50, higher = more extreme
            consistency = 100 - np.std(arr)  # higher = more consistent across rallies
            n_high = (arr >= 80).sum()
            n_low = (arr <= 20).sum()
            mean_pr = arr.mean()
            score = extremity * (1 + consistency / 100)
            rows.append({
                'feature': feat,
                'lag': lag,
                'n_rallies': len(arr),
                'mean_pct_rank': round(mean_pr, 1),
                'std_pct_rank': round(arr.std(), 1),
                'n_high_80+': int(n_high),
                'n_low_20-': int(n_low),
                'extremity': round(extremity, 1),
                'score': round(score, 2),
            })

    if not rows:
        print('No features had enough non-NaN snapshots.')
        return

    sf = pd.DataFrame(rows)

    # Load production model bull-horizon features
    prod_features = set()
    if os.path.exists(PROD_CSV):
        prod_df = pd.read_csv(PROD_CSV)
        eth_prod = prod_df[(prod_df['coin'] == 'ETH') & (prod_df['horizon'] == args.horizon)]
        if len(eth_prod) > 0:
            opt_feats = eth_prod.iloc[0].get('optimal_features', '')
            if isinstance(opt_feats, str):
                prod_features = set(opt_feats.split(','))
    print(f'  Production bull ({args.horizon}h) model uses {len(prod_features)} features\n')

    sf['in_prod_model'] = sf['feature'].isin(prod_features)

    # Show top features at t=0 (rally start) by score
    print(f'\n{"="*120}')
    print(f'  TOP {args.top_n} FEATURES AT t=0 (rally start) BY EXTREMITY+CONSISTENCY')
    print(f'{"="*120}\n')
    t0 = sf[sf['lag'] == 0].sort_values('score', ascending=False).head(args.top_n)
    print(t0[['feature', 'mean_pct_rank', 'std_pct_rank', 'n_high_80+',
              'n_low_20-', 'extremity', 'score', 'in_prod_model']].to_string(index=False))

    print(f'\n{"="*120}')
    print(f'  TOP {args.top_n} FEATURES AT t=-6h (6h before rally) BY SCORE')
    print(f'{"="*120}\n')
    tm6 = sf[sf['lag'] == -6].sort_values('score', ascending=False).head(args.top_n)
    print(tm6[['feature', 'mean_pct_rank', 'std_pct_rank', 'n_high_80+',
               'n_low_20-', 'extremity', 'score', 'in_prod_model']].to_string(index=False))

    # Now: features that are EXTREME at ALL lags (consistent pre-rally pattern)
    print(f'\n{"="*120}')
    print(f'  FEATURES EXTREME AT ALL LAGS (-6h, -3h, -1h, 0)')
    print(f'  (mean_pct_rank in [0,20] OR [80,100] for ALL lags)')
    print(f'{"="*120}\n')
    by_feat = sf.pivot(index='feature', columns='lag', values='mean_pct_rank')
    extreme_at_all = by_feat[
        ((by_feat <= 20).all(axis=1)) | ((by_feat >= 80).all(axis=1))
    ]
    if len(extreme_at_all) == 0:
        print('  No features were extreme at all 4 lags.')
    else:
        extreme_at_all['in_prod'] = extreme_at_all.index.isin(prod_features)
        print(extreme_at_all.sort_values(0, ascending=False).to_string())

    # Features NOT in production model that scored high at t=0
    print(f'\n{"="*120}')
    print(f'  TOP UNUSED FEATURES (high score at t=0, NOT in production bull model)')
    print(f'{"="*120}\n')
    unused = t0[~t0['in_prod_model']].head(20)
    print(unused[['feature', 'mean_pct_rank', 'std_pct_rank', 'n_high_80+',
                  'n_low_20-', 'extremity', 'score']].to_string(index=False))

    # Save full table
    out_path = os.path.join(ENGINE_DIR, 'output',
                            f'rally_feature_scan_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.to_csv(out_path, index=False)
    print(f'\nFull table saved to: {out_path}')


if __name__ == '__main__':
    main()
