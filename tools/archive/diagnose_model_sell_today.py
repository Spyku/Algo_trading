"""
diagnose_model_sell_today.py — what is the bull (6h) production model
seeing right now that's making it say SELL with 90%+ confidence, and
what would the broader feature set say if asked?

For each hour 14:00-20:00 UTC today (when model was firing SELL):
  1. Snapshot the 10 features the production bull model actually uses
  2. Compute each feature's percentile rank vs 30d distribution
  3. Show which direction each feature is leaning (bearish high pct OR
     bullish low pct depending on the feature's typical interpretation)
  4. Snapshot the candidate "rally precursor" features identified earlier
     (rsi_14h, price_to_sma20h, bb_position_20h, zscore_50h, logret_3-8h)
  5. Compare to recent BUY-firing bars and SELL-firing bars to see
     what's different

Reads (read-only):
  config/regime_config_ed.json
  models/crypto_ed_production.csv
  data/eth_hourly_data.csv
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)

from crypto_trading_system_ed import build_all_features, load_data

REGIME_CFG = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
PROD_CSV = os.path.join(ENGINE, 'models', 'crypto_ed_production.csv')
CYCLE_CSV = os.path.join(ENGINE, 'output', 'cycle_metrics.csv')

# Time window of interest: today's SELL signals
TODAY = pd.Timestamp('2026-04-26', tz='UTC')


def percentile_rank(value, distribution):
    arr = distribution[~np.isnan(distribution)]
    if len(arr) == 0 or pd.isna(value):
        return np.nan
    return 100 * (arr < value).sum() / len(arr)


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    bull_h = cfg['ETH']['bull']['horizon']  # 6
    bear_h = cfg['ETH']['bear']['horizon']  # 7

    prod = pd.read_csv(PROD_CSV)
    bull_row = prod[(prod['coin'] == 'ETH') & (prod['horizon'] == bull_h)].iloc[0]
    bull_features = bull_row['optimal_features'].split(',')
    print(f"Bull (6h) model features ({len(bull_features)}):")
    for f in bull_features:
        print(f"  - {f}")

    print(f"\nBuilding feature matrix for ETH (horizon={bull_h}h)...")
    df_raw = load_data('ETH')
    df, all_cols = build_all_features(df_raw, asset_name='ETH', horizon=bull_h,
                                       verbose=False, keep_label_nan_tail=True)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()
    print(f"  total feature columns: {len(all_cols)}, total rows: {len(df)}")

    # Today's signal-firing hours (from cycle_metrics)
    cm = pd.read_csv(CYCLE_CSV)
    cm['timestamp'] = pd.to_datetime(cm['timestamp'], utc=True)
    today_eth = cm[(cm['asset'] == 'ETH') & (cm['timestamp'] >= TODAY)].copy()
    print(f"\nToday's cycles (ETH, post-SELL):")
    sell_hours = today_eth[today_eth['signal'] == 'SELL']
    print(sell_hours[['timestamp', 'signal', 'confidence', 'horizon']].to_string(index=False))

    # Snapshot each SELL hour: bull model features at that bar
    print("\n" + "=" * 130)
    print(f"  PRODUCTION BULL MODEL FEATURE SNAPSHOTS at SELL bars")
    print(f"  Each value's percentile rank vs last 30d (low pct = unusually low value)")
    print("=" * 130)

    # 30d window for percentile distribution
    cutoff_30d = TODAY - pd.Timedelta(days=30)
    df30 = df[df.index >= cutoff_30d]

    # Pick the most interesting cycle bars to analyze (15:00, 16:00, 17:00 = peak SELL conf)
    target_dts = [
        pd.Timestamp('2026-04-26 15:00', tz='UTC'),
        pd.Timestamp('2026-04-26 16:00', tz='UTC'),
        pd.Timestamp('2026-04-26 17:00', tz='UTC'),
        pd.Timestamp('2026-04-26 18:00', tz='UTC'),
    ]

    snap_rows = []
    for feat in bull_features:
        if feat not in df.columns:
            continue
        dist = df30[feat].dropna().values
        row = {'feature': feat}
        for t in target_dts:
            available = df.index[df.index <= t]
            if len(available) == 0:
                continue
            v = df.loc[available[-1], feat]
            pct = percentile_rank(v, dist)
            row[t.strftime('%H:%M')] = f"{v:+.4f} ({pct:.0f}p)"
        snap_rows.append(row)
    snap_df = pd.DataFrame(snap_rows)
    print(snap_df.to_string(index=False))

    # Same for the candidate RALLY-PRECURSOR features (from earlier scan)
    print("\n" + "=" * 130)
    print(f"  'RALLY PRECURSOR' FEATURES (NOT in production model — for comparison)")
    print(f"  These were found to correlate with rally starts. Low pct = oversold.")
    print("=" * 130)
    rally_feats = ['price_to_sma20h', 'bb_position_20h', 'zscore_50h', 'rsi_14h',
                   'price_velocity_4h', 'logret_3h', 'logret_4h', 'logret_5h',
                   'logret_8h', 'price_to_sma50h', 'stoch_k_14h']
    snap_rows = []
    for feat in rally_feats:
        if feat not in df.columns:
            continue
        dist = df30[feat].dropna().values
        row = {'feature': feat}
        for t in target_dts:
            available = df.index[df.index <= t]
            if len(available) == 0:
                continue
            v = df.loc[available[-1], feat]
            pct = percentile_rank(v, dist)
            row[t.strftime('%H:%M')] = f"{v:+.4f} ({pct:.0f}p)"
        snap_rows.append(row)
    snap_df = pd.DataFrame(snap_rows)
    print(snap_df.to_string(index=False))

    # Comparison: what did these same features look like at the recent
    # successful BUY moments (e.g., last bull rally start 04-26 06:00 area)?
    print("\n" + "=" * 130)
    print(f"  COMPARISON: feature values at 04-26 04:00 (today's local low — when a")
    print(f"  BUY would have been correct in retrospect) vs current 17:00 SELL signal")
    print("=" * 130)

    comp_dts = [
        pd.Timestamp('2026-04-26 04:00', tz='UTC'),  # local low
        pd.Timestamp('2026-04-26 11:00', tz='UTC'),  # SELL fire (model)
        pd.Timestamp('2026-04-26 16:00', tz='UTC'),  # peak conf SELL
    ]
    comp_rows = []
    all_check = bull_features + rally_feats
    for feat in all_check:
        if feat not in df.columns:
            continue
        dist = df30[feat].dropna().values
        row = {'feature': feat, 'in_bull_model': feat in bull_features}
        for t in comp_dts:
            available = df.index[df.index <= t]
            if len(available) == 0:
                continue
            v = df.loc[available[-1], feat]
            pct = percentile_rank(v, dist)
            row[t.strftime('%m-%d %H:%M')] = f"{v:+.4f} ({pct:.0f}p)"
        comp_rows.append(row)
    print(pd.DataFrame(comp_rows).to_string(index=False))

    # Also: the close prices for context
    print("\n" + "=" * 130)
    print("  ETH PRICE CONTEXT")
    print("=" * 130)
    for t in [pd.Timestamp(s, tz='UTC') for s in
              ['2026-04-26 00:00', '2026-04-26 04:00', '2026-04-26 11:00',
               '2026-04-26 14:00', '2026-04-26 16:00', '2026-04-26 18:00']]:
        available = df.index[df.index <= t]
        if len(available) == 0:
            continue
        c = df.loc[available[-1], 'close']
        print(f"  {t.strftime('%H:%M UTC')}: ${c:.2f}")


if __name__ == '__main__':
    main()
