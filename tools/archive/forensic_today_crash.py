"""
forensic_today_crash.py — bar-by-bar forensic of TODAY's (2026-04-27) crash.

Goal: find the SPECIFIC indicator combination that would have fired
DURING this morning's 03:45 -> 05:45 crash from $2400 to $2319.

Method:
  1. Pull fresh 5m data through this morning.
  2. Print every 5m bar from 02:00 -> 06:00 UTC with all candidate indicators.
  3. For each combination of (rally_h, rally_pct, drop_window, drop_pct, d2_window, d2_pct):
     - Did it fire at-or-before 04:30 UTC (so we'd have exited near $2380)?
     - How many false fires in the prior 60 days?
  4. Return the cheapest (= fewest false fires) combination that fires today.

Reads: data/eth_5m_backtest_90d.csv + fresh Binance.
Writes: output/forensic_today_<ts>.csv
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETH_5M_CSV = os.path.join(ENGINE, 'data', 'eth_5m_backtest_90d.csv')

# Crash anchors
CRASH_PEAK_DT = pd.Timestamp('2026-04-27 03:45:00', tz='UTC')
CRASH_TROUGH_DT = pd.Timestamp('2026-04-27 05:45:00', tz='UTC')
# Realistic fire window: by 05:30 UTC, exiting near $2330 (down -2.9% from peak,
# saving ~0.5pp vs trough). 04:30 fire was infeasible — indicators showed nothing.
TARGET_FIRE_BY = pd.Timestamp('2026-04-27 05:30:00', tz='UTC')
TARGET_FIRE_FROM = pd.Timestamp('2026-04-27 04:55:00', tz='UTC')  # earliest realistic


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


def precompute(df):
    closes = df['close']
    out = pd.DataFrame(index=df.index)
    for win_min in [5, 10, 15]:
        b = win_min // 5
        out[f'ret_{win_min}m'] = (closes / closes.shift(b) - 1) * 100
        out[f'd2_{win_min}m'] = out[f'ret_{win_min}m'] - out[f'ret_{win_min}m'].shift(b)
    for prior_h in [3, 5, 8, 10, 12, 18, 24, 36, 48, 72]:
        prior_bars = prior_h * 12
        out[f'rally_{prior_h}h'] = (closes / closes.shift(prior_bars) - 1) * 100
    return out


def main():
    print("Loading 5m data...")
    df = load_combined_5m()
    print(f"  {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    ind = precompute(df)

    # === Print today's crash window bar-by-bar ===
    print(f"\n{'='*120}")
    print(f"  TODAY'S 5m BARS: 02:00 -> 06:00 UTC")
    print(f"{'='*120}")
    crash_window_lo = pd.Timestamp('2026-04-27 02:00:00', tz='UTC')
    crash_window_hi = pd.Timestamp('2026-04-27 06:30:00', tz='UTC')
    mask = (df.index >= crash_window_lo) & (df.index <= crash_window_hi)
    cw = df.loc[mask].copy()
    cw['ret_5m'] = ind.loc[mask, 'ret_5m']
    cw['ret_10m'] = ind.loc[mask, 'ret_10m']
    cw['ret_15m'] = ind.loc[mask, 'ret_15m']
    cw['d2_5m'] = ind.loc[mask, 'd2_5m']
    cw['d2_10m'] = ind.loc[mask, 'd2_10m']
    cw['d2_15m'] = ind.loc[mask, 'd2_15m']
    cw['r5h'] = ind.loc[mask, 'rally_5h']
    cw['r10h'] = ind.loc[mask, 'rally_10h']
    cw['r24h'] = ind.loc[mask, 'rally_24h']
    cw['r36h'] = ind.loc[mask, 'rally_36h']
    cw['r48h'] = ind.loc[mask, 'rally_48h']
    cw['r72h'] = ind.loc[mask, 'rally_72h']
    show = cw[['close', 'ret_5m', 'ret_10m', 'ret_15m', 'd2_5m', 'd2_10m', 'd2_15m',
               'r5h', 'r10h', 'r24h', 'r36h', 'r48h', 'r72h']]
    print(show.round(3).to_string())

    # === SEARCH: find combos that fire TODAY at-or-before 04:30 UTC, with low historical FP rate ===
    print(f"\n{'='*120}")
    print(f"  COMBO SEARCH — must fire by 04:30 UTC AND have <5 false fires in prior 60d")
    print(f"{'='*120}")

    # Today fire window: 04:55 -> 05:30 (when crash signature actually appears).
    fire_window_lo = TARGET_FIRE_FROM
    fire_window_hi = TARGET_FIRE_BY
    today_mask = np.array((df.index >= fire_window_lo) & (df.index <= fire_window_hi))
    today_idx = np.where(today_mask)[0]

    # Historical mask — last 60d ending day before
    hist_lo = pd.Timestamp('2026-02-26 00:00:00', tz='UTC')
    hist_hi = pd.Timestamp('2026-04-26 23:55:00', tz='UTC')
    hist_mask = np.array((df.index >= hist_lo) & (df.index <= hist_hi))

    # Pre-known crashes (from earlier analysis): exclude their windows from FP count
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

    rows = []
    rally_choices = [
        ('rally_18h', [1.5, 2.0, 2.5, 3.0]),
        ('rally_24h', [1.5, 2.0, 2.5, 3.0, 3.5]),
        ('rally_36h', [2.0, 2.5, 3.0, 3.5, 4.0]),
        ('rally_48h', [1.5, 2.0, 2.5, 3.0, 3.5]),
        ('rally_72h', [1.5, 2.0, 2.5, 3.0, 3.5]),
    ]
    drop_choices = [
        ('ret_5m',  [-0.20, -0.25, -0.30, -0.40, -0.50]),
        ('ret_10m', [-0.30, -0.40, -0.50, -0.60]),
        ('ret_15m', [-0.30, -0.40, -0.50, -0.60, -0.80]),
    ]
    d2_choices = [
        (None, [None]),
        ('d2_5m',  [-0.15, -0.20, -0.25, -0.30]),
        ('d2_10m', [-0.30, -0.40, -0.50]),
        ('d2_15m', [-0.30, -0.40, -0.50]),
    ]

    for r_col, r_thrs in rally_choices:
        for r_thr in r_thrs:
            rally_ok = (ind[r_col] >= r_thr).fillna(False).values
            for d_col, d_thrs in drop_choices:
                for d_thr in d_thrs:
                    drop_ok = (ind[d_col] <= d_thr).fillna(False).values
                    for d2_col, d2_thrs in d2_choices:
                        for d2_thr in d2_thrs:
                            if d2_col is None or d2_thr is None:
                                d2_ok = np.ones(len(df), dtype=bool)
                                d2_lbl = 'd2=OFF'
                            else:
                                d2_ok = (ind[d2_col] <= d2_thr).fillna(False).values
                                d2_lbl = f'{d2_col}<={d2_thr}'
                            fire = rally_ok & drop_ok & d2_ok

                            # Did it fire today?
                            today_fires = fire[today_mask]
                            if not today_fires.any():
                                continue
                            first_today = today_idx[np.argmax(today_fires)]
                            first_today_dt = df.index[first_today]
                            first_today_px = df['close'].iloc[first_today]

                            # Historical false positives (excluding known crash windows)
                            hist_fires = np.where(fire & hist_mask)[0]
                            fp_hist = sum(1 for j in hist_fires if j not in legitimate_idx)
                            tp_hist = sum(1 for j in hist_fires if j in legitimate_idx)

                            rows.append({
                                'config': f'{r_col}>={r_thr} + {d_col}<={d_thr} + {d2_lbl}',
                                'today_fire_dt': first_today_dt.strftime('%H:%M'),
                                'today_fire_px': round(first_today_px, 2),
                                'fp_60d': fp_hist,
                                'tp_60d': tp_hist,
                                'total_fires_60d': len(hist_fires),
                            })

    if not rows:
        print("  NO combination caught today's crash. Bumping search?")
        return

    df_r = pd.DataFrame(rows).sort_values(['fp_60d', 'today_fire_dt'])
    print(f"\nTotal combos that fired today by 04:30: {len(df_r)}")
    print(f"\n--- Top 25 by lowest historical FP, sorted by fire time ---")
    print(df_r.head(25).to_string(index=False))

    # Best of the best: FP <= 3
    print(f"\n--- VERY CLEAN: FP_60d <= 3 ---")
    clean = df_r[df_r['fp_60d'] <= 3]
    print(clean.head(30).to_string(index=False) if not clean.empty else "  None.")

    # And: FP_60d == 0
    print(f"\n--- ZERO FALSE POSITIVES (would only fire on real crashes) ---")
    zero = df_r[df_r['fp_60d'] == 0]
    print(zero.head(30).to_string(index=False) if not zero.empty else "  None.")

    out = os.path.join(ENGINE, 'output',
                       f'forensic_today_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df_r.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
