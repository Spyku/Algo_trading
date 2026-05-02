"""
analyze_missed_rallies.py — for each completed trade (BUY -> SELL) in the last
N days, quantify post-exit rally that the trader missed. Also flag cash gaps
where the trader sat out meaningful rises.

Output: ranked table of misses + summary by likely cause.
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone, timedelta

import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS_PATH = os.path.join(ENGINE_DIR, 'config', 'position_ed_v2_ETH.json')
PRICE_PATH = os.path.join(ENGINE_DIR, 'data', 'eth_hourly_data.csv')


def parse_trade_time(t):
    """Trade times come in many forms; normalize to UTC datetime."""
    if not t:
        return None
    s = t.replace(' (synced)', '').replace(' (auto)', '').strip()
    try:
        if 'T' in s:
            dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        # naive local-format like "2026-04-26 11:03"
        dt = datetime.strptime(s, '%Y-%m-%d %H:%M')
        # legacy entries were stored as Europe/Zurich naive; treat as UTC
        # for analysis since the price CSV is UTC and ~2h skew won't change
        # the qualitative picture
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=30, help='look-back window')
    ap.add_argument('--lookahead', type=int, default=48,
                    help='hours after each SELL to scan for missed upside')
    args = ap.parse_args()

    with open(POS_PATH) as f:
        pos = json.load(f)
    trades = pos.get('trades', [])

    df = pd.read_csv(PRICE_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    pairs = []
    open_buy = None
    for tr in trades:
        dt = parse_trade_time(tr.get('time'))
        if dt is None:
            continue
        if tr['action'] == 'BUY':
            open_buy = (dt, tr)
        elif tr['action'] == 'SELL' and open_buy:
            buy_dt, buy_tr = open_buy
            if dt >= cutoff:
                pairs.append({
                    'buy_dt': buy_dt, 'buy_price': buy_tr['price'],
                    'sell_dt': dt, 'sell_price': tr['price'],
                    'pnl_pct': tr.get('pnl_pct', 0),
                    'auto': tr.get('auto', True),
                    'manual': tr.get('manual', False),
                })
            open_buy = None

    print(f'\n{"="*100}')
    print(f'  POST-EXIT RALLY ANALYSIS — last {args.days}d, {len(pairs)} closed trades')
    print(f'{"="*100}\n')

    # For each trade compute max upside within `lookahead` hours after SELL
    rows = []
    for p in pairs:
        # next BUY datetime if any (cap the lookahead at re-entry)
        next_buy_dt = None
        for tr in trades:
            t = parse_trade_time(tr.get('time'))
            if tr['action'] == 'BUY' and t and t > p['sell_dt']:
                next_buy_dt = t
                break

        end_window = p['sell_dt'] + timedelta(hours=args.lookahead)
        if next_buy_dt and next_buy_dt < end_window:
            end_window = next_buy_dt

        window = df[(df.index > p['sell_dt']) & (df.index <= end_window)]
        if window.empty:
            continue
        peak = window['high'].max()
        peak_dt = window['high'].idxmax()
        missed_pct = 100 * (peak - p['sell_price']) / p['sell_price']
        hours_to_peak = (peak_dt - p['sell_dt']).total_seconds() / 3600

        # also: what was the realized PnL of this trade vs what it would
        # have been if held until peak
        held_to_peak_pct = 100 * (peak - p['buy_price']) / p['buy_price']

        # categorize SELL trigger
        if p['manual']:
            trigger = 'MANUAL'
        elif p['pnl_pct'] < 0.5:
            trigger = 'SHIELD-RELEASED'  # shield blocked, then released near floor
        elif p['pnl_pct'] < 1.0:
            trigger = 'EARLY-MODEL-SELL'  # model fired right above shield floor
        else:
            trigger = 'MODEL-SELL'

        rows.append({
            'buy_dt': p['buy_dt'].strftime('%m-%d %H:%M'),
            'sell_dt': p['sell_dt'].strftime('%m-%d %H:%M'),
            'buy_px': round(p['buy_price'], 2),
            'sell_px': round(p['sell_price'], 2),
            'pnl_%': round(p['pnl_pct'], 2),
            'peak_px': round(peak, 2),
            'h_to_peak': round(hours_to_peak, 1),
            'missed_%': round(missed_pct, 2),
            'if_held_%': round(held_to_peak_pct, 2),
            'trigger': trigger,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print('No closed trades in window.')
        return

    # Show full table sorted by missed_%
    out_sorted = out.sort_values('missed_%', ascending=False)
    print(out_sorted.to_string(index=False))

    print(f'\n{"="*100}')
    print('  SUMMARY')
    print(f'{"="*100}')
    print(f'\nTotal closed trades:           {len(out)}')
    print(f'Total realized PnL:            {out["pnl_%"].sum():+.2f}%')
    print(f'Total "if held to peak" PnL:   {out["if_held_%"].sum():+.2f}%')
    print(f'Sum of missed upside:          {out["missed_%"].sum():+.2f}%')
    print(f'Avg missed per trade:          {out["missed_%"].mean():+.2f}%')
    print(f'Median missed per trade:       {out["missed_%"].median():+.2f}%')

    print(f'\nMisses >= 1% (rally left on table):')
    big = out_sorted[out_sorted['missed_%'] >= 1.0]
    print(f'  Count: {len(big)} / {len(out)} trades')
    print(f'  Sum:   {big["missed_%"].sum():+.2f}%')

    print(f'\nMisses >= 2% (significant rally missed):')
    huge = out_sorted[out_sorted['missed_%'] >= 2.0]
    print(f'  Count: {len(huge)} / {len(out)} trades')
    print(f'  Sum:   {huge["missed_%"].sum():+.2f}%')

    print(f'\nBy trigger type:')
    grp = out.groupby('trigger').agg(
        n=('missed_%', 'count'),
        avg_missed=('missed_%', 'mean'),
        sum_missed=('missed_%', 'sum'),
        avg_pnl=('pnl_%', 'mean'),
    ).round(2)
    print(grp.to_string())

    # Now: cash gaps. For each interval where trader was OUT of position,
    # did price rise meaningfully?
    print(f'\n\n{"="*100}')
    print(f'  CASH-GAP ANALYSIS — what trader missed while OUT')
    print(f'{"="*100}\n')

    gap_rows = []
    for i in range(len(pairs) - 1):
        sell_dt = pairs[i]['sell_dt']
        sell_px = pairs[i]['sell_price']
        next_buy_dt = pairs[i+1]['buy_dt']
        next_buy_px = pairs[i+1]['buy_price']
        gap_h = (next_buy_dt - sell_dt).total_seconds() / 3600
        if gap_h < 0.5:
            continue
        window = df[(df.index > sell_dt) & (df.index < next_buy_dt)]
        if window.empty:
            continue
        peak = window['high'].max()
        missed_in_gap = 100 * (peak - sell_px) / sell_px
        cost_of_re_entry = 100 * (next_buy_px - sell_px) / sell_px
        gap_rows.append({
            'sell_dt': sell_dt.strftime('%m-%d %H:%M'),
            'gap_h': round(gap_h, 1),
            'sell_px': round(sell_px, 2),
            'gap_peak_px': round(peak, 2),
            'next_buy_px': round(next_buy_px, 2),
            'gap_peak_%': round(missed_in_gap, 2),
            're_entry_drift_%': round(cost_of_re_entry, 2),
        })

    if gap_rows:
        gdf = pd.DataFrame(gap_rows).sort_values('gap_peak_%', ascending=False)
        print(gdf.to_string(index=False))
        print(f'\nTotal cash-gap upside left on table: {gdf["gap_peak_%"].sum():+.2f}%')
        print(f'Cash-gaps where price went +1%+ during gap: '
              f'{(gdf["gap_peak_%"] >= 1.0).sum()} / {len(gdf)}')
        print(f'Cash-gaps where re-entry was MORE expensive than exit: '
              f'{(gdf["re_entry_drift_%"] > 0).sum()} / {len(gdf)}')
    else:
        print('No cash gaps in window.')


if __name__ == '__main__':
    main()
