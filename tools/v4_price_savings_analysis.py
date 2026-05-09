"""tools/v4_price_savings_analysis.py

For each historical post_only rejection incident, compute:
  - Actual avg fill price + actual fees paid (from position file + log)
  - v4 hypothetical avg fill price + estimated fees
  - Savings: $ per incident, in price + fees combined

Uses 3 v4 outcome scenarios from POST_ONLY_FIX_SUMMARY.md:
  - Optimistic: 100% maker fills at bid+0.01 ($38/s flow)
  - Central:    55% maker, 45% taker fallback ($19/s flow) — most realistic
  - Pessimistic: 27.5% maker, 72.5% taker fallback ($9.5/s flow)

Maker = 0% fee, Taker = 0.09% Revolut taker fee.
"""
import json
import os
import re

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSITION_FILE = os.path.join(ENGINE, 'config', 'position_ed_v2_ETH.json')

# Per-incident summary, built from the Gmail rejection list + matched log data.
# bid_avg = average of observed bids across the rejection bursts
# ask_avg = average of observed asks
# spread_bps_avg = average spread (already > 5 for all BUY incidents)
INCIDENTS = [
    # (date, side, n_rejects, bid_avg, ask_avg, spread_bps_avg, position_idx_in_trades)
    {
        'date': '2026-04-30 12:01 UTC',
        'side': 'BUY', 'symbol': 'ETH',
        'n_rejects': 1,
        'bid_avg': 2258.73, 'ask_avg': 2261.89, 'spread_bps_avg': 14.0,
        'recorded_fill': 2261.43, 'recorded_usd': 12000.00,
        'pos_time_match': '2026-04-30T12:02:08Z',
    },
    {
        'date': '2026-04-30 23:01-02 UTC',
        'side': 'BUY', 'symbol': 'ETH',
        'n_rejects': 4,
        'bid_avg': 2253.86, 'ask_avg': 2259.35, 'spread_bps_avg': 24.4,  # incl one 48bps spike
        'recorded_fill': 2258.00, 'recorded_usd': 11999.99,
        'pos_time_match': '2026-04-30T23:05:45Z',
    },
    {
        'date': '2026-05-05 15:00 UTC',
        'side': 'BUY', 'symbol': 'ETH',
        'n_rejects': 1,
        'bid_avg': 2378.00, 'ask_avg': 2381.56, 'spread_bps_avg': 15.0,
        'recorded_fill': 2377.05, 'recorded_usd': 13999.99,
        'pos_time_match': '2026-05-05T15:01:12Z',
    },
    {
        'date': '2026-05-05 23:36 UTC',
        'side': 'BUY', 'symbol': 'ETH',
        'n_rejects': 1,
        'bid_avg': 2355.82, 'ask_avg': 2357.40, 'spread_bps_avg': 6.7,
        'recorded_fill': 2356.08, 'recorded_usd': 14000.00,
        'pos_time_match': '2026-05-05T23:37:30Z',
    },
    {
        'date': '2026-05-06 21:00 UTC',
        'side': 'SELL', 'symbol': 'ETH',
        'n_rejects': 2,
        'bid_avg': 2351.33, 'ask_avg': 2352.21, 'spread_bps_avg': 3.8,
        'recorded_fill': 2350.49, 'recorded_usd': None,  # SELL
        'pos_time_match': '2026-05-06 23:02',
    },
    {
        'date': '2026-05-08 23:02-05 UTC (May 9 01:02 LOCAL)',
        'side': 'BUY', 'symbol': 'ETH',
        'n_rejects': 10,  # at least 10 confirmed via Gmail thread; v4 doc says 22 total
        'bid_avg': 2309.85, 'ask_avg': 2312.84, 'spread_bps_avg': 13.0,
        'recorded_fill': 2311.34, 'recorded_usd': 13999.99,
        'pos_time_match': '2026-05-08T23:08:53Z',
    },
]

TAKER_FEE = 0.0009  # Revolut X taker fee 0.09%
MAKER_FEE = 0.0000  # 0% maker

SCENARIOS = {
    'Optimistic (100% maker @ bid+0.01)': {'maker_frac': 1.00},
    'Central (55% maker / 45% taker)':    {'maker_frac': 0.55},
    'Pessimistic (27.5% maker / 72.5% taker)': {'maker_frac': 0.275},
}


def v4_estimate(incident, scenario):
    """For a BUY: maker portion at bid+0.01 (≈bid), taker portion at ask + 0.09% fee."""
    bid = incident['bid_avg']
    ask = incident['ask_avg']
    f_maker = scenario['maker_frac']
    f_taker = 1 - f_maker
    # Avg execution price (BUY): pay bid for maker, pay ask for taker
    # Taker fee adds 0.09% to the effective cost
    avg_price = f_maker * (bid + 0.01) + f_taker * ask * (1 + TAKER_FEE)
    # Fee in $ for the incident at recorded_usd target:
    usd = incident['recorded_usd'] or 0
    fees = f_taker * usd * TAKER_FEE
    return avg_price, fees


def main():
    print('# v4 Price + Fee Savings Analysis')
    print()
    print('## Per-incident table (v4 = Central case, the most realistic estimate)')
    print()
    print('| # | Incident | Side | Spread | Actual fill | v4 Central est | Delta price/ETH | Delta fees | Total $ saved |')
    print('|---|---|---|---|---|---|---|---|---|')

    total_saved = 0.0
    for i, inc in enumerate(INCIDENTS, 1):
        if inc['side'] == 'SELL':
            print(f"| {i} | {inc['date']} | SELL | {inc['spread_bps_avg']:.1f}bps | "
                  f"${inc['recorded_fill']:.2f} | _v4 doesn't apply_ | — | — | $0 |")
            continue

        # BUY case
        if inc['recorded_usd'] is None:
            continue
        usd = inc['recorded_usd']
        actual_fill = inc['recorded_fill']
        actual_eth = usd / actual_fill

        # Actual fees: implied from (actual_fill - mid) × eth_qty
        # In reality, recorded fill INCLUDES taker fees (balance-delta basis).
        # Conservative estimate of actual taker fee: assume 75% taker (per v4 doc) on
        # the May 9 incident; for others assume scaled by spread.
        # Just use the v4 doc's $9.29 / $14k = 0.066% as the upper bound for May 9.

        v4_price, v4_fees = v4_estimate(inc, SCENARIOS['Central (55% maker / 45% taker)'])
        v4_eth = usd / v4_price

        delta_price_per_eth = actual_fill - v4_price  # positive = v4 cheaper
        delta_eth_extra = v4_eth - actual_eth          # positive = v4 buys more
        # Savings = (price_diff × eth) + (fee_diff)
        # Approximate: assume actual fees were 73% taker × usd × 0.0009 (May 9 baseline)
        actual_fee_est = 0.73 * usd * TAKER_FEE if 'May 9' in inc['date'] else 0.55 * usd * TAKER_FEE
        delta_fees = actual_fee_est - v4_fees
        total_saved_inc = delta_price_per_eth * actual_eth + delta_fees
        total_saved += total_saved_inc

        print(f"| {i} | {inc['date']} | BUY | {inc['spread_bps_avg']:.1f}bps | "
              f"${actual_fill:.2f} | ${v4_price:.2f} | ${delta_price_per_eth:+.2f} | "
              f"${delta_fees:+.2f} | ${total_saved_inc:+.2f} |")

    print()
    print(f'**Total dollar savings across BUY incidents (Central case): ${total_saved:.2f}**')
    print()

    # Three-scenario table for incidents (full sweep)
    print('## Per-incident: Full scenario sweep (Optimistic / Central / Pessimistic)')
    print()
    print('| Incident | Side | Spread | Actual fill | v4 Opt | v4 Cen | v4 Pes |')
    print('|---|---|---|---|---|---|---|')
    for inc in INCIDENTS:
        if inc['side'] == 'SELL':
            print(f"| {inc['date']} | SELL | {inc['spread_bps_avg']:.1f}bps | ${inc['recorded_fill']:.2f} | n/a | n/a | n/a |")
            continue
        opt_p, opt_f = v4_estimate(inc, SCENARIOS['Optimistic (100% maker @ bid+0.01)'])
        cen_p, cen_f = v4_estimate(inc, SCENARIOS['Central (55% maker / 45% taker)'])
        pes_p, pes_f = v4_estimate(inc, SCENARIOS['Pessimistic (27.5% maker / 72.5% taker)'])
        print(f"| {inc['date']} | BUY | {inc['spread_bps_avg']:.1f}bps | "
              f"${inc['recorded_fill']:.2f} | ${opt_p:.2f} | ${cen_p:.2f} | ${pes_p:.2f} |")

    print()
    print('## Methodology notes')
    print('- v4 maker fills at bid+0.01 (effectively the bid)')
    print('- v4 taker fills at ask + 0.09% Revolut taker fee')
    print('- Central case (55/45 maker/taker) is the most realistic per the v4 sim doc')
    print('- "Actual fees" estimate: 73% taker for the May 9 incident (from v4 doc), 55% for other')
    print('  wide-spread incidents (assumed similar slide → reject pattern). This is approximate;')
    print('  exact taker fraction would need per-incident log replay (filled_size on each leg).')
    print('- SELL incidents not included in savings (v4 only patches BUY side).')
    print('- 15 unmatched April 3-13 rejections excluded (logs predate 2026-04-23, not recoverable).')


if __name__ == '__main__':
    main()
