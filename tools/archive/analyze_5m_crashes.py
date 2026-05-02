"""
analyze_5m_crashes.py — find the big ETH crashes in the last month using
5-minute candles + identify the earliest reliable INDICATOR that flags
them before the deepest part. Includes today's 2026-04-27 morning crash
by splicing fresh Binance 5m data on top of the cached 90d file.

Phase 1 (this script):
  - Define crash = peak-to-trough drawdown >= X% within rolling W minutes
  - List every crash in last 30 days (start time, depth, duration)
  - For each crash, scan candidate INDICATORS computed on the 5m bars at
    every t in [crash_start - 60min, crash_start]:
        - rolling 15m / 30m / 60m return
        - 5m close vs 15m EMA (fast/slow cross)
        - 5m volume spike (vs 30m median)
        - rolling realized vol over 30m vs 4h baseline
        - 5m derivative (price acceleration over 10m, 20m)
        - cumulative 5m drawdown from 60m high
  - Per indicator, find threshold (X) that FIRES at-or-before crash_start
    AND does not fire too often outside crashes (precision/recall).

Phase 2 (next script — only if Phase 1 finds something):
  - Backtest: emergency-exit overlay on current strategy. If indicator
    fires while in_pos, force-SELL bypassing shield. Measure delta vs
    baseline on 30/60/90d.

Reads: data/eth_5m_backtest_90d.csv + fresh Binance pull.
Writes: output/5m_crashes_<timestamp>.csv (crash list)
        output/5m_indicators_<timestamp>.csv (per-indicator scores)
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


def load_combined_5m():
    """Load cached + fresh Binance to bring through current time."""
    df_cache = pd.read_csv(ETH_5M_CSV)
    df_cache['datetime'] = pd.to_datetime(df_cache['datetime'], utc=True)
    df_cache = df_cache.set_index('datetime').sort_index()
    last_cache = df_cache.index[-1]
    print(f"  Cache last bar: {last_cache}")

    # Pull fresh from Binance
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
        except Exception as e:
            print(f"  Binance fetch failed ({e}); proceeding with cache only")
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
        print(f"  Fresh appended: {len(fresh_rows)} bars; combined last: {df.index[-1]}")
        return df
    return df_cache


def find_crashes(df, drop_pct, window_min, min_bars_apart=24):
    """Return list of crashes: peak-to-trough drop >= drop_pct within
    window_min minutes. Crashes within min_bars_apart of each other are
    merged into the deepest one."""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    times = df.index
    win_bars = window_min // 5  # 5-min bars

    raw = []
    for i in range(win_bars, len(df)):
        # Look at last win_bars bars + current
        peak = highs[i - win_bars:i + 1].max()
        peak_idx = i - win_bars + int(np.argmax(highs[i - win_bars:i + 1]))
        # trough = min low between peak and current
        trough = lows[peak_idx:i + 1].min()
        trough_idx = peak_idx + int(np.argmin(lows[peak_idx:i + 1]))
        if peak <= 0:
            continue
        drop = (trough / peak - 1) * 100
        if drop <= -abs(drop_pct):
            raw.append({
                'peak_idx': peak_idx,
                'trough_idx': trough_idx,
                'detect_idx': i,
                'peak_time': times[peak_idx],
                'trough_time': times[trough_idx],
                'detect_time': times[i],
                'peak_px': peak,
                'trough_px': trough,
                'drop_pct': drop,
                'duration_min': (trough_idx - peak_idx) * 5,
            })

    # Merge overlapping crashes — keep deepest within each merge cluster
    if not raw:
        return []
    raw.sort(key=lambda x: x['peak_idx'])
    merged = []
    cur = raw[0]
    for r in raw[1:]:
        if r['peak_idx'] - cur['trough_idx'] < min_bars_apart:
            if r['drop_pct'] < cur['drop_pct']:
                cur = r
        else:
            merged.append(cur)
            cur = r
    merged.append(cur)
    return merged


def precompute_indicators(df):
    """Compute candidate indicator series at every 5m bar."""
    closes = df['close']
    highs = df['high']
    lows = df['low']
    volumes = df['volume']

    out = pd.DataFrame(index=df.index)
    # Returns over rolling windows (in %, signed)
    for win_min in [10, 15, 20, 30, 45, 60, 90]:
        b = win_min // 5
        out[f'ret_{win_min}m'] = (closes / closes.shift(b) - 1) * 100
    # EMA cross (close vs 15m EMA = 3 bars)
    ema_fast = closes.ewm(span=3, adjust=False).mean()
    ema_slow = closes.ewm(span=12, adjust=False).mean()
    out['ema_cross_pct'] = (ema_fast / ema_slow - 1) * 100
    # Volume spike — current 5m vol vs 30m median
    vol_med30 = volumes.rolling(6).median()
    out['vol_ratio_30m'] = volumes / vol_med30
    # Realized vol over 30m vs 4h baseline
    rets_5m = closes.pct_change()
    rv_30m = rets_5m.rolling(6).std() * np.sqrt(6)  # 30m vol
    rv_4h = rets_5m.rolling(48).std() * np.sqrt(48)  # 4h vol
    out['rv_ratio_30m_vs_4h'] = rv_30m / rv_4h
    # Acceleration (2nd derivative): change in 10m return vs prior 10m
    ret_10m = (closes / closes.shift(2) - 1) * 100
    out['accel_10m'] = ret_10m - ret_10m.shift(2)
    # Drawdown from 60m high
    high_60m = highs.rolling(12).max()
    out['dd_from_60m_high'] = (closes / high_60m - 1) * 100
    # Drawdown from 2h high
    high_2h = highs.rolling(24).max()
    out['dd_from_2h_high'] = (closes / high_2h - 1) * 100

    return out


def evaluate_indicator(ind_series, threshold, direction, crashes, df,
                       lookback_min=60):
    """For each crash, determine: did indicator fire at-or-before peak+lookback?
    Direction: 'below' means fires when ind < threshold; 'above' = ind > threshold.
    Returns: (n_caught, n_pre_peak, n_at_peak, mean_lead_min)
    Also count false positives outside crashes."""
    closes = df['close'].values
    n_caught = 0
    n_pre_peak = 0  # fires before trough
    leads = []
    fired_idx_set = set()

    if direction == 'below':
        fired = ind_series < threshold
    else:
        fired = ind_series > threshold
    fired_idx = np.where(fired.values)[0]
    fired_idx_set = set(fired_idx.tolist())

    for c in crashes:
        # Crash window = [peak_idx, trough_idx]
        peak = c['peak_idx']
        trough = c['trough_idx']
        # Scan from peak to trough+lookback
        scan_lo = peak
        scan_hi = min(len(df) - 1, trough + lookback_min // 5)
        # First fire in [peak, trough] — that's "in time"
        first_fire = None
        for j in range(scan_lo, scan_hi + 1):
            if j in fired_idx_set:
                first_fire = j
                break
        if first_fire is not None:
            n_caught += 1
            if first_fire <= trough:
                n_pre_peak += 1
                leads.append((trough - first_fire) * 5)  # minutes before trough

    # False positives: bars where indicator fires but no crash in next 60m
    fp = 0
    crash_window_idx = set()
    for c in crashes:
        for j in range(max(0, c['peak_idx'] - 12), c['trough_idx'] + 12):
            crash_window_idx.add(j)
    for j in fired_idx:
        if j not in crash_window_idx:
            fp += 1

    mean_lead = np.mean(leads) if leads else 0
    return {
        'n_caught': n_caught,
        'n_pre_trough': n_pre_peak,
        'mean_lead_min_to_trough': round(mean_lead, 1),
        'false_positives': fp,
        'total_fires': len(fired_idx),
    }


def main():
    print("Loading 5m data...")
    df = load_combined_5m()
    print(f"  Loaded: {len(df)} 5m bars, {df.index[0]} -> {df.index[-1]}")

    # Filter to last 30 days
    cutoff = df.index[-1] - timedelta(days=30)
    df30 = df[df.index >= cutoff]
    print(f"  Last 30d: {len(df30)} bars")

    # Define crash thresholds
    print("\n=== CRASH DETECTION ===")
    print("Looking for peak->trough drops within 60-180 min windows")

    all_crashes = {}
    for drop_pct, win_min in [(2.0, 60), (3.0, 120), (4.0, 180), (5.0, 240)]:
        crashes = find_crashes(df30, drop_pct, win_min)
        all_crashes[(drop_pct, win_min)] = crashes
        print(f"  >={drop_pct}% in {win_min}min: {len(crashes)} events")
        for c in crashes:
            print(f"    {c['peak_time']} -> {c['trough_time']} "
                  f"({c['duration_min']}min): {c['peak_px']:.0f} -> "
                  f"{c['trough_px']:.0f} ({c['drop_pct']:+.2f}%)")

    # Use the >=3% in 120min set as canonical (matches "this morning" feel)
    canonical = all_crashes[(3.0, 120)]
    print(f"\n=== CANONICAL CRASH SET: {len(canonical)} events (>=3% in 120min) ===")

    if not canonical:
        print("  No crashes in window — exiting.")
        return

    # Precompute indicators
    print("\n=== INDICATOR PRECOMPUTE ===")
    ind = precompute_indicators(df30)
    print(f"  Computed {len(ind.columns)} indicator series")

    # Evaluate each indicator with multiple thresholds
    print("\n=== INDICATOR EVALUATION ===")
    results = []
    test_grid = {
        'ret_10m':  ([-0.5, -0.7, -1.0, -1.5, -2.0], 'below'),
        'ret_15m':  ([-0.7, -1.0, -1.5, -2.0, -2.5], 'below'),
        'ret_20m':  ([-1.0, -1.5, -2.0, -2.5, -3.0], 'below'),
        'ret_30m':  ([-1.5, -2.0, -2.5, -3.0], 'below'),
        'ret_45m':  ([-2.0, -2.5, -3.0, -3.5], 'below'),
        'ret_60m':  ([-2.5, -3.0, -3.5, -4.0], 'below'),
        'ret_90m':  ([-3.0, -4.0, -5.0], 'below'),
        'ema_cross_pct': ([-0.3, -0.5, -0.7, -1.0], 'below'),
        'vol_ratio_30m': ([2.0, 3.0, 5.0, 8.0], 'above'),
        'rv_ratio_30m_vs_4h': ([1.5, 2.0, 3.0, 4.0], 'above'),
        'accel_10m': ([-0.5, -0.8, -1.2, -1.8], 'below'),
        'dd_from_60m_high': ([-1.5, -2.0, -2.5, -3.0], 'below'),
        'dd_from_2h_high': ([-2.0, -2.5, -3.0, -4.0], 'below'),
    }

    for ind_name, (thresholds, direction) in test_grid.items():
        for thr in thresholds:
            r = evaluate_indicator(ind[ind_name], thr, direction, canonical, df30)
            r['indicator'] = ind_name
            r['threshold'] = thr
            r['direction'] = direction
            results.append(r)

    df_r = pd.DataFrame(results)
    df_r['recall'] = df_r['n_caught'] / max(1, len(canonical))
    df_r['precision'] = df_r['n_caught'] / df_r['total_fires'].replace(0, np.nan)
    df_r['fp_rate_per_day'] = df_r['false_positives'] / 30  # 30 days

    df_r = df_r.sort_values(['n_pre_trough', 'mean_lead_min_to_trough', 'precision'],
                            ascending=[False, False, False])

    # Display
    cols_show = ['indicator', 'threshold', 'direction', 'n_caught', 'n_pre_trough',
                 'mean_lead_min_to_trough', 'total_fires', 'false_positives',
                 'fp_rate_per_day', 'recall', 'precision']
    print(df_r[cols_show].to_string(index=False))

    # Save
    out_dir = os.path.join(ENGINE, 'output')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    crash_path = os.path.join(out_dir, f'5m_crashes_{ts}.csv')
    pd.DataFrame(canonical).to_csv(crash_path, index=False)
    ind_path = os.path.join(out_dir, f'5m_indicators_{ts}.csv')
    df_r.to_csv(ind_path, index=False)
    print(f"\nCrashes saved: {crash_path}")
    print(f"Indicator scores: {ind_path}")

    # Top picks: caught all crashes + lead time > 0 + low FP
    print(f"\n=== TOP CANDIDATES (caught >= {len(canonical)-1} crashes, FP/day < 5) ===")
    top = df_r[(df_r['n_pre_trough'] >= len(canonical) - 1) &
               (df_r['fp_rate_per_day'] < 5)]
    print(top[cols_show].to_string(index=False) if not top.empty else "  None passed.")


if __name__ == '__main__':
    main()
