"""
zoom_05_10.py — zoom on 05:10 UTC bar of today's crash. Find the cleanest
combination that fires AT-OR-BEFORE 05:10 (price $2363, -1.46% from peak).

Approach: instead of grid search, take the 05:10 bar's EXACT indicator
values, then check what subset of those would be "rare enough" historically
to be a clean trigger. Use AND of multiple simultaneous conditions.

Also try: combine multiple LARGE indicators that align ONLY when the move
is real (not when it's a single-bar flicker).

Reads: 5m data + fresh Binance.
Writes: output/zoom_05_10_<ts>.csv
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETH_5M_CSV = os.path.join(ENGINE, 'data', 'eth_5m_backtest_90d.csv')

TARGET_BAR = pd.Timestamp('2026-04-27 05:10:00', tz='UTC')
PEAK_BAR = pd.Timestamp('2026-04-27 03:45:00', tz='UTC')
TROUGH_BAR = pd.Timestamp('2026-04-27 05:45:00', tz='UTC')


def load_combined_5m():
    df_cache = pd.read_csv(ETH_5M_CSV)
    df_cache['datetime'] = pd.to_datetime(df_cache['datetime'], utc=True)
    df_cache = df_cache.set_index('datetime').sort_index()
    last_cache = df_cache.index[-1]
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    fresh_rows = []
    start_ms = int((last_cache + timedelta(minutes=5)).timestamp() * 1000)
    while True:
        url = (f'https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=5m'
               f'&startTime={start_ms}&limit=1000')
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=10) as r:
                batch = json.loads(r.read())
        except Exception:
            break
        if not batch:
            break
        for k in batch:
            fresh_rows.append({
                'datetime': pd.Timestamp(k[0], unit='ms', tz='UTC'),
                'open': float(k[1]), 'high': float(k[2]),
                'low': float(k[3]), 'close': float(k[4]),
                'volume': float(k[5]),
            })
        if len(batch) < 1000:
            break
        start_ms = batch[-1][0] + 5 * 60 * 1000
    if fresh_rows:
        df_fresh = pd.DataFrame(fresh_rows).set_index('datetime')
        df = pd.concat([df_cache, df_fresh])
        df = df[~df.index.duplicated(keep='last')].sort_index()
        return df
    return df_cache


def precompute_full(df):
    closes = df['close']
    volumes = df['volume']
    highs = df['high']
    out = pd.DataFrame(index=df.index)
    for win_min in [5, 10, 15, 20, 30]:
        b = win_min // 5
        out[f'ret_{win_min}m'] = (closes / closes.shift(b) - 1) * 100
        out[f'd2_{win_min}m'] = out[f'ret_{win_min}m'] - out[f'ret_{win_min}m'].shift(b)
    # Volume spike
    out['vol_ratio_30m'] = volumes / volumes.rolling(6).median()
    out['vol_ratio_60m'] = volumes / volumes.rolling(12).median()
    # Drawdown from rolling highs
    out['dd_60m'] = (closes / highs.rolling(12).max() - 1) * 100
    out['dd_2h'] = (closes / highs.rolling(24).max() - 1) * 100
    out['dd_4h'] = (closes / highs.rolling(48).max() - 1) * 100
    # Range expansion (true-range proxy)
    tr = (highs - df['low']) / closes * 100
    out['tr_pct'] = tr
    out['tr_pct_avg30m'] = tr.rolling(6).mean()
    out['tr_ratio'] = tr / out['tr_pct_avg30m']
    return out


def main():
    print("Loading 5m data...")
    df = load_combined_5m()
    ind = precompute_full(df)

    # Print 05:10 bar's exact values
    print(f"\n=== 05:10 UTC TARGET BAR (price $2363, -1.46% from peak $2398) ===")
    if TARGET_BAR not in df.index:
        print(f"  Bar not found in data!")
        return
    row = ind.loc[TARGET_BAR]
    px = df.loc[TARGET_BAR, 'close']
    vol = df.loc[TARGET_BAR, 'volume']
    print(f"  close: ${px:.2f}, volume: {vol:.2f}")
    for col in row.index:
        print(f"  {col:25s}: {row[col]:+.4f}")

    # Historical scan: find bars where multiple indicators align
    print(f"\n=== HISTORICAL ANALYSIS (last 60d, excluding known crashes) ===")
    hist_lo = pd.Timestamp('2026-02-26 00:00:00', tz='UTC')
    hist_hi = pd.Timestamp('2026-04-26 23:55:00', tz='UTC')
    hist_mask = np.array((df.index >= hist_lo) & (df.index <= hist_hi))

    known_crashes = [
        (pd.Timestamp('2026-03-29 21:35:00', tz='UTC'), pd.Timestamp('2026-03-29 22:45:00', tz='UTC')),
        (pd.Timestamp('2026-04-02 00:55:00', tz='UTC'), pd.Timestamp('2026-04-02 02:45:00', tz='UTC')),
        (pd.Timestamp('2026-04-08 13:05:00', tz='UTC'), pd.Timestamp('2026-04-08 14:55:00', tz='UTC')),
        (pd.Timestamp('2026-04-12 01:00:00', tz='UTC'), pd.Timestamp('2026-04-12 02:05:00', tz='UTC')),
        (pd.Timestamp('2026-04-14 14:30:00', tz='UTC'), pd.Timestamp('2026-04-14 15:05:00', tz='UTC')),
    ]
    legitimate_idx = set()
    for c_peak, c_trough in known_crashes:
        c_mask = (df.index >= c_peak) & (df.index <= c_trough)
        for j in np.where(c_mask)[0]:
            legitimate_idx.add(j)

    today_at_510 = np.array(df.index <= TARGET_BAR) & np.array(df.index >= TARGET_BAR - pd.Timedelta(minutes=5))

    # Try AND-combinations of strong simultaneous indicators
    cond_grid = {
        'ret_15m<=-0.8': (ind['ret_15m'] <= -0.8).fillna(False).values,
        'ret_15m<=-0.7': (ind['ret_15m'] <= -0.7).fillna(False).values,
        'ret_10m<=-0.5': (ind['ret_10m'] <= -0.5).fillna(False).values,
        'ret_10m<=-0.4': (ind['ret_10m'] <= -0.4).fillna(False).values,
        'd2_15m<=-0.5': (ind['d2_15m'] <= -0.5).fillna(False).values,
        'd2_15m<=-0.4': (ind['d2_15m'] <= -0.4).fillna(False).values,
        'd2_10m<=-0.2': (ind['d2_10m'] <= -0.2).fillna(False).values,
        'd2_5m<=0': (ind['d2_5m'] <= 0).fillna(False).values,
        'vol_ratio_30m>=2': (ind['vol_ratio_30m'] >= 2.0).fillna(False).values,
        'vol_ratio_30m>=3': (ind['vol_ratio_30m'] >= 3.0).fillna(False).values,
        'vol_ratio_60m>=1.5': (ind['vol_ratio_60m'] >= 1.5).fillna(False).values,
        'tr_ratio>=2': (ind['tr_ratio'] >= 2.0).fillna(False).values,
        'tr_ratio>=3': (ind['tr_ratio'] >= 3.0).fillna(False).values,
        'dd_60m<=-1': (ind['dd_60m'] <= -1.0).fillna(False).values,
        'dd_60m<=-1.5': (ind['dd_60m'] <= -1.5).fillna(False).values,
        'dd_2h<=-1.5': (ind['dd_2h'] <= -1.5).fillna(False).values,
        'dd_2h<=-2': (ind['dd_2h'] <= -2.0).fillna(False).values,
    }
    # Today fired? = condition true at 05:10
    target_idx = np.where(np.array(df.index == TARGET_BAR))[0][0]
    today_at_510_indicators = {k: bool(v[target_idx]) for k, v in cond_grid.items()}
    print(f"\n=== INDIVIDUAL CONDITIONS — at 05:10 ===")
    for k, fired in today_at_510_indicators.items():
        v = cond_grid[k]
        hist_fires = np.where(v & hist_mask)[0]
        fp = sum(1 for j in hist_fires if j not in legitimate_idx)
        tp = sum(1 for j in hist_fires if j in legitimate_idx)
        marker = "Y" if fired else "."
        print(f"  {marker} {k:25s}  fired_today={fired}  hist_60d: total={len(hist_fires):3d} TP={tp} FP={fp}")

    # Try all PAIRS of two (AND'd) where both fire today
    print(f"\n=== PAIRS (AND) — both must fire at 05:10 — sorted by FP ===")
    pair_rows = []
    keys = [k for k, v in today_at_510_indicators.items() if v]
    print(f"  {len(keys)} conditions fire today; testing {len(keys)*(len(keys)-1)//2} pairs")
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            combo = cond_grid[k1] & cond_grid[k2]
            hist_fires = np.where(combo & hist_mask)[0]
            fp = sum(1 for j in hist_fires if j not in legitimate_idx)
            tp = sum(1 for j in hist_fires if j in legitimate_idx)
            pair_rows.append({
                'cond_A': k1, 'cond_B': k2,
                'tp_60d': tp, 'fp_60d': fp,
                'total_fires_60d': len(hist_fires),
            })
    df_pairs = pd.DataFrame(pair_rows).sort_values(['fp_60d', 'total_fires_60d'])
    print(df_pairs.head(15).to_string(index=False))

    # Try TRIPLES (AND of 3) - where all 3 fire today
    print(f"\n=== TRIPLES (AND) — all 3 must fire at 05:10 ===")
    trip_rows = []
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys[i+1:], i+1):
            for k3 in keys[j+1:]:
                combo = cond_grid[k1] & cond_grid[k2] & cond_grid[k3]
                hist_fires = np.where(combo & hist_mask)[0]
                fp = sum(1 for jj in hist_fires if jj not in legitimate_idx)
                tp = sum(1 for jj in hist_fires if jj in legitimate_idx)
                trip_rows.append({
                    'cond_A': k1, 'cond_B': k2, 'cond_C': k3,
                    'tp_60d': tp, 'fp_60d': fp,
                    'total_fires_60d': len(hist_fires),
                })
    df_triples = pd.DataFrame(trip_rows).sort_values(['fp_60d', 'total_fires_60d'])
    print(f"\n  Top 25 triples by lowest FP_60d:")
    print(df_triples.head(25).to_string(index=False))

    print(f"\n  ZERO FP_60d triples (would only fire on real moves):")
    zero = df_triples[df_triples['fp_60d'] == 0]
    print(zero.head(30).to_string(index=False) if not zero.empty else "  None.")

    # Save
    out_dir = os.path.join(ENGINE, 'output')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    df_pairs.to_csv(os.path.join(out_dir, f'zoom_pairs_{ts}.csv'), index=False)
    df_triples.to_csv(os.path.join(out_dir, f'zoom_triples_{ts}.csv'), index=False)
    print(f"\nSaved.")


if __name__ == '__main__':
    main()
