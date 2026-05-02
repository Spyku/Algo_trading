"""
analyze_rallies_2mo.py — find every >=3% ETH rally in the last 60 days, then
assess the trader's behavior on each:
  - Was trader IN position at rally start? (caught it)
  - Did trader BUY during the rally? (entered late)
  - Did trader SELL before the peak? (left upside)
  - Did trader miss it entirely? (was OUT, never entered)

Rally definition: a swing-low -> swing-high move of >=3% within <=72h, where
swings use a 4h confirmation window. Dedupe overlapping rallies by keeping
the largest in each cluster.
"""

import json
import os
import argparse
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS_PATH = os.path.join(ENGINE_DIR, 'config', 'position_ed_v2_ETH.json')
PRICE_PATH = os.path.join(ENGINE_DIR, 'data', 'eth_hourly_data.csv')


def parse_trade_time(t):
    if not t:
        return None
    s = t.replace(' (synced)', '').replace(' (auto)', '').strip()
    try:
        if 'T' in s:
            dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        dt = datetime.strptime(s, '%Y-%m-%d %H:%M')
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def find_rallies(df, min_pct=3.0, max_hours=72, swing_window=4):
    """
    Walk through every hour. For each hour, look ahead up to max_hours and
    find the highest price. If (max - cur) / cur >= min_pct, record the rally
    candidate. Then dedupe: cluster overlapping rallies, keep the largest.
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    times = df.index.to_list()
    n = len(df)

    candidates = []
    for i in range(n - 1):
        # local trough check: low at i should be the lowest in [i-swing, i+swing]
        lo_l = max(0, i - swing_window)
        lo_r = min(n, i + swing_window + 1)
        if lows[i] != lows[lo_l:lo_r].min():
            continue
        # find peak in next max_hours
        end = min(n, i + max_hours + 1)
        window_highs = highs[i+1:end]
        if len(window_highs) == 0:
            continue
        peak_offset = int(np.argmax(window_highs))
        peak_idx = i + 1 + peak_offset
        peak_price = highs[peak_idx]
        start_price = lows[i]
        pct = 100 * (peak_price - start_price) / start_price
        if pct < min_pct:
            continue
        candidates.append({
            'start_idx': i, 'peak_idx': peak_idx,
            'start_dt': times[i], 'peak_dt': times[peak_idx],
            'start_px': start_price, 'peak_px': peak_price,
            'pct': pct,
            'duration_h': (times[peak_idx] - times[i]).total_seconds() / 3600,
        })

    # Dedupe: sort by pct desc, greedily pick largest, drop overlaps
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


def trader_state_at(trades_seq, dt):
    """Return ('cash'|'position', last_buy_price_or_None, last_buy_dt_or_None) at dt."""
    state = 'cash'
    last_buy_px = None
    last_buy_dt = None
    for tr in trades_seq:
        t = parse_trade_time(tr.get('time'))
        if t is None or t > dt:
            break
        if tr['action'] == 'BUY':
            state = 'position'
            last_buy_px = tr['price']
            last_buy_dt = t
        else:
            state = 'cash'
    return state, last_buy_px, last_buy_dt


def trades_during(trades_seq, t0, t1):
    out = []
    for tr in trades_seq:
        t = parse_trade_time(tr.get('time'))
        if t is None:
            continue
        if t0 <= t <= t1:
            out.append((t, tr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=60)
    ap.add_argument('--min-pct', type=float, default=3.0)
    ap.add_argument('--max-hours', type=int, default=72)
    args = ap.parse_args()

    with open(POS_PATH) as f:
        pos = json.load(f)
    trades = sorted(
        [tr for tr in pos.get('trades', []) if parse_trade_time(tr.get('time'))],
        key=lambda tr: parse_trade_time(tr['time'])
    )

    df = pd.read_csv(PRICE_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    dfw = df[df.index >= cutoff]

    rallies = find_rallies(dfw, min_pct=args.min_pct, max_hours=args.max_hours)

    print(f'\n{"="*120}')
    print(f'  ETH RALLIES >= {args.min_pct}% IN LAST {args.days}d (up to {args.max_hours}h duration)')
    print(f'{"="*120}\n')

    rows = []
    for r in rallies:
        state_at_start, last_buy_px, last_buy_dt = trader_state_at(trades, r['start_dt'])
        active_in_window = trades_during(trades, r['start_dt'], r['peak_dt'])
        # Determine outcome:
        # Compound all round-trips that occur (or are open) during rally window
        # to get the trader's REAL participation in the rally.
        capital_mult = 1.0
        leg_count = 0
        # Walk full trade sequence; for each completed BUY->SELL pair where
        # the SELL falls within the rally window, multiply by (sell/buy).
        cur_buy = None
        for tr in trades:
            t = parse_trade_time(tr.get('time'))
            if t is None:
                continue
            if tr['action'] == 'BUY':
                cur_buy = (t, tr['price'])
            elif tr['action'] == 'SELL' and cur_buy is not None:
                buy_t, buy_px = cur_buy
                # leg participates if the sell falls in window OR the buy falls in window
                if (r['start_dt'] <= t <= r['peak_dt']) or (r['start_dt'] <= buy_t <= r['peak_dt']):
                    capital_mult *= (tr['price'] / buy_px)
                    leg_count += 1
                cur_buy = None
        # If still holding open position at peak, count unrealized
        if cur_buy is not None:
            buy_t, buy_px = cur_buy
            if buy_t <= r['peak_dt']:
                capital_mult *= (r['peak_px'] / buy_px)
                leg_count += 1

        captured_pct = 100 * (capital_mult - 1)
        full_potential = r['pct']
        capture_ratio = captured_pct / full_potential if full_potential > 0 else 0

        if state_at_start == 'position':
            outcome = f'IN at start ({leg_count} leg(s) during rally)'
        elif leg_count > 0:
            buys_in = [tt for tt in active_in_window if tt[1]['action'] == 'BUY']
            if buys_in:
                buy_t, _ = buys_in[0]
                hrs_late = (buy_t - r['start_dt']).total_seconds() / 3600
                outcome = f'CASH at start, first BUY {hrs_late:.1f}h into rally ({leg_count} leg(s))'
            else:
                outcome = f'CASH at start, late entry ({leg_count} leg(s))'
        else:
            outcome = 'CASH throughout rally — MISSED ENTIRELY'

        rows.append({
            'start': r['start_dt'].strftime('%m-%d %H:%M'),
            'peak': r['peak_dt'].strftime('%m-%d %H:%M'),
            'dur_h': round(r['duration_h'], 1),
            'start_px': round(r['start_px'], 0),
            'peak_px': round(r['peak_px'], 0),
            'rally_%': round(r['pct'], 2),
            'state_at_start': state_at_start,
            'captured_%': round(captured_pct, 2),
            'capture_ratio': round(capture_ratio, 2),
            'outcome': outcome,
        })

    # Print sorted by date, then sorted by size
    print('--- chronological ---')
    for r in rows:
        print(f"{r['start']} -> {r['peak']} ({r['dur_h']:5.1f}h)  "
              f"${r['start_px']:.0f} -> ${r['peak_px']:.0f}  +{r['rally_%']:.2f}%  "
              f"[{r['state_at_start'].upper():8}]  cap={r['captured_%']:+.2f}% "
              f"({100*r['capture_ratio']:.0f}%)")
        print(f"           {r['outcome']}")
        print()

    print(f'\n{"="*120}')
    print(f'  SUMMARY — {len(rows)} rallies >={args.min_pct}% in last {args.days}d')
    print(f'{"="*120}')

    if not rows:
        return
    df_r = pd.DataFrame(rows)
    total_rally = df_r['rally_%'].sum()
    total_cap = df_r['captured_%'].sum()
    print(f'\n  Total rally upside available:   {total_rally:+.2f}%')
    print(f'  Total upside captured:          {total_cap:+.2f}%')
    print(f'  Capture rate:                   {100*total_cap/total_rally:.1f}%')
    print()
    print(f'  Rallies caught from start (IN):    {(df_r["state_at_start"]=="position").sum()}')
    print(f'  Rallies trader was OUT at start:   {(df_r["state_at_start"]=="cash").sum()}')
    print(f'  Rallies missed entirely:           {(df_r["captured_%"]==0).sum()}')
    print(f'  Rallies with capture ratio >=80%:  {(df_r["capture_ratio"]>=0.8).sum()}')
    print(f'  Rallies with capture ratio <50%:   {(df_r["capture_ratio"]<0.5).sum()}')

    print('\n  Top 5 LEAST-captured rallies (most painful):')
    worst = df_r.nsmallest(5, 'capture_ratio')
    for _, r in worst.iterrows():
        leaked = r['rally_%'] - r['captured_%']
        print(f"    {r['start']} +{r['rally_%']:.2f}% -> captured {r['captured_%']:+.2f}% "
              f"({100*r['capture_ratio']:.0f}%)  [{r['state_at_start']}]  leak={leaked:+.2f}%")

    print('\n  Top 5 BEST-captured rallies:')
    best = df_r.nlargest(5, 'captured_%')
    for _, r in best.iterrows():
        print(f"    {r['start']} +{r['rally_%']:.2f}% -> captured {r['captured_%']:+.2f}% "
              f"({100*r['capture_ratio']:.0f}%)  [{r['state_at_start']}]")


if __name__ == '__main__':
    main()
