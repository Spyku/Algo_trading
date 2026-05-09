"""tools/analyze_rejections_v4_backtest.py

Cross-references Revolut post_only rejection events (from email) with the
ed_v2 trader logs to extract for each rejection:
  - exact timestamp
  - side (BUY/SELL)
  - posted limit price
  - bid / ask / spread_bps observed at that moment
  - slide attempt # (1..N)
  - what v4 (spread-aware slide) would have done

For BUY rejections at spread > 5 bps where the slide pushed price above bid+0.01:
  v4 would have stayed at bid+0.01 — no rejection (assuming ask hadn't dropped to
  bid+0.01 level in latency window, which is extremely rare).

For SELL rejections: v4 doesn't apply (only patches BUY side).

Outputs a markdown table to stdout.
"""
import os
import re
import glob
from datetime import datetime, timedelta, timezone

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(ENGINE, 'logs')

# Rejection list compiled from Gmail search (all 'Ordre ... refusé' from
# no-reply@revolut.com). Times are UTC. Each row is one individual rejection.
REJECTIONS = [
    # (utc_iso, side, symbol, qty, posted_price)
    ('2026-04-03T09:00:17Z', 'BUY',  'BTC-USD', 0.08959503, 66968.00),
    ('2026-04-03T09:00:29Z', 'BUY',  'BTC-USD', 0.08959496, 66968.05),
    ('2026-04-03T09:01:01Z', 'BUY',  'BTC-USD', 0.08957160, 66985.52),
    ('2026-04-03T19:00:17Z', 'BUY',  'BTC-USD', 0.08978696, 66824.85),
    ('2026-04-03T19:00:27Z', 'BUY',  'BTC-USD', 0.08978696, 66824.85),
    ('2026-04-04T11:00:17Z', 'BUY',  'BTC-USD', 0.08931581, 67177.36),
    ('2026-04-04T11:00:28Z', 'BUY',  'BTC-USD', 0.08931970, 67174.43),
    ('2026-04-04T11:00:39Z', 'BUY',  'BTC-USD', 0.08928868, 67197.77),
    ('2026-04-04T11:00:49Z', 'BUY',  'BTC-USD', 0.08929883, 67190.13),
    ('2026-04-04T11:01:00Z', 'BUY',  'BTC-USD', 0.08928460, 67200.84),
    ('2026-04-08T08:51:05Z', 'SELL', 'ETH-USD', 3.27813976, 2256.27),
    ('2026-04-08T08:51:12Z', 'SELL', 'ETH-USD', 3.27813976, 2256.25),
    ('2026-04-10T21:02:13Z', 'SELL', 'ETH-USD', 5.08084825, 2255.51),
    ('2026-04-12T04:07:53Z', 'SELL', 'ETH-USD', 5.23396099, 2210.00),
    ('2026-04-13T00:25:04Z', 'BUY',  'ETH-USD', 5.31112294, 2198.52),
    ('2026-04-30T12:01:34Z', 'BUY',  'ETH-USD', 5.31191597, 2259.07),
    ('2026-04-30T23:01:13Z', 'BUY',  'ETH-USD', 5.31954536, 2255.83),
    ('2026-04-30T23:01:42Z', 'BUY',  'ETH-USD', 5.32263630, 2254.52),
    ('2026-04-30T23:01:47Z', 'BUY',  'ETH-USD', 5.32209336, 2254.75),
    ('2026-04-30T23:02:03Z', 'BUY',  'ETH-USD', 5.32155052, 2254.98),
    ('2026-05-05T15:00:38Z', 'BUY',  'ETH-USD', 5.88680094, 2378.20),
    ('2026-05-05T23:36:57Z', 'BUY',  'ETH-USD', 5.94206903, 2356.08),
    ('2026-05-06T21:00:21Z', 'SELL', 'ETH-USD', 5.96719320, 2351.09),
    ('2026-05-06T21:00:38Z', 'SELL', 'ETH-USD', 5.96719320, 2351.70),
    ('2026-05-08T23:02:39Z', 'BUY',  'ETH-USD', 4.46738115, 2311.24),
    ('2026-05-08T23:02:56Z', 'BUY',  'ETH-USD', 4.46738115, 2311.24),
    ('2026-05-08T23:03:13Z', 'BUY',  'ETH-USD', 4.46668541, 2311.60),
    ('2026-05-08T23:03:30Z', 'BUY',  'ETH-USD', 4.46612512, 2311.89),
    ('2026-05-08T23:03:47Z', 'BUY',  'ETH-USD', 4.46728450, 2311.29),
    ('2026-05-08T23:04:04Z', 'BUY',  'ETH-USD', 4.46556496, 2312.18),
    ('2026-05-08T23:04:21Z', 'BUY',  'ETH-USD', 4.46544909, 2312.24),
    ('2026-05-08T23:04:38Z', 'BUY',  'ETH-USD', 4.46699460, 2311.44),
    ('2026-05-08T23:04:55Z', 'BUY',  'ETH-USD', 4.46494703, 2312.50),
    ('2026-05-08T23:05:11Z', 'BUY',  'ETH-USD', 4.46828979, 2310.77),
]


# Match lines like:
#   Maker buy #6: ETH-USD at $2,311.24 bid=$2,309.55 ask=$2,313.30 spread=16.2bps [60s/450s]
MAKER_LINE_RE = re.compile(
    r'Maker (buy|sell) #(\d+):\s+(\S+)\s+at \$([\d,]+\.\d+)\s+'
    r'bid=\$([\d,]+\.\d+)\s+ask=\$([\d,]+\.\d+)\s+spread=([\d.]+)bps\s+\[(\d+)s/(\d+)s\]'
)
# Wall-clock for a maker line is inferred from the surrounding [LIVE] header
# block timestamp + the elapsed counter inside the brackets.
LIVE_HEADER_RE = re.compile(r'\[LIVE\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')


def parse_log_for_maker_lines(log_path):
    """Yield dict per maker line found in the log.
    Uses the most recent [LIVE] header to anchor wall-clock time."""
    if not os.path.exists(log_path):
        return
    cur_live_dt = None
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m_live = LIVE_HEADER_RE.search(line)
            if m_live:
                # Trader logs LIVE timestamps in LOCAL CEST. Convert to UTC.
                local_dt = datetime.strptime(m_live.group(1), '%Y-%m-%d %H:%M:%S')
                # CEST is UTC+2 in May 2026 (post-DST 2026-03-29)
                cur_live_dt = local_dt - timedelta(hours=2)
                continue
            m = MAKER_LINE_RE.search(line)
            if m and cur_live_dt:
                side, attempt, sym, posted, bid, ask, spread, elapsed, window = m.groups()
                # Maker line wall-clock = LIVE header time + elapsed seconds
                wall_dt = cur_live_dt + timedelta(seconds=int(elapsed))
                yield {
                    'wall_dt_utc': wall_dt,
                    'side': side.upper(),
                    'attempt': int(attempt),
                    'symbol': sym,
                    'posted_price': float(posted.replace(',', '')),
                    'bid': float(bid.replace(',', '')),
                    'ask': float(ask.replace(',', '')),
                    'spread_bps': float(spread),
                    'elapsed_s': int(elapsed),
                    'window_s': int(window),
                }


def find_log_for_rejection(reject_dt_utc):
    """Return the ed_v2 log file most likely to contain this rejection."""
    candidates = sorted(glob.glob(os.path.join(LOGS_DIR, 'ed_v2_*.log')))
    # Filename includes a local-time timestamp: ed_v2_YYYYMMDD_HHMMSS.log
    # Pick the latest log whose start <= reject + 1 hour buffer (in local).
    best = None
    for f in candidates:
        name = os.path.basename(f)
        m = re.match(r'ed_v2_(\d{8})_(\d{6})\.log', name)
        if not m:
            continue
        try:
            log_start_local = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
            log_start_utc = log_start_local - timedelta(hours=2)
        except ValueError:
            continue
        if log_start_utc <= reject_dt_utc + timedelta(minutes=5):
            best = f
    return best


def match_rejection(reject_iso, posted_price, log_path, tolerance_sec=120):
    """Find the maker line in log_path closest to (reject time, posted price)."""
    target_dt = datetime.fromisoformat(reject_iso.replace('Z', '+00:00')).replace(tzinfo=None)
    best = None
    best_score = None
    for entry in parse_log_for_maker_lines(log_path):
        dt_diff = abs((entry['wall_dt_utc'] - target_dt).total_seconds())
        price_diff = abs(entry['posted_price'] - posted_price)
        if dt_diff > tolerance_sec:
            continue
        # Score = price match dominant (within $1) + time as tiebreak
        score = price_diff * 100 + dt_diff
        if best_score is None or score < best_score:
            best = entry
            best_score = score
    return best


def main():
    print('# v4 backtest: post_only rejection analysis')
    print(f'# Total rejections to analyze: {len(REJECTIONS)}')
    print()
    rows = []
    matched = 0
    unmatched = 0
    for reject_iso, side, sym, qty, posted in REJECTIONS:
        reject_dt = datetime.fromisoformat(reject_iso.replace('Z', '+00:00')).replace(tzinfo=None)
        log_path = find_log_for_rejection(reject_dt)
        log_name = os.path.basename(log_path) if log_path else 'NO_LOG'
        entry = match_rejection(reject_iso, posted, log_path) if log_path else None

        if entry:
            matched += 1
            spread = entry['spread_bps']
            bid = entry['bid']
            slide_above = entry['posted_price'] - (bid + 0.01)
            v4_would_post = bid + 0.01 if spread > 5.0 else None  # narrow spread keeps slide
            v4_helps = (
                side == 'BUY'
                and spread > 5.0
                and slide_above > 0.10  # posted clearly above bid+0.01
            )
            rows.append({
                'reject_iso': reject_iso,
                'side': side,
                'sym': sym,
                'posted': posted,
                'attempt': entry['attempt'],
                'bid': bid,
                'ask': entry['ask'],
                'spread_bps': spread,
                'elapsed_s': entry['elapsed_s'],
                'slide_above_bid_plus_1c': slide_above,
                'v4_would_post': v4_would_post,
                'v4_helps': v4_helps,
                'log': log_name,
            })
        else:
            unmatched += 1
            rows.append({
                'reject_iso': reject_iso,
                'side': side,
                'sym': sym,
                'posted': posted,
                'attempt': None,
                'log': log_name,
                'v4_helps': False,
            })

    # Print markdown table
    print(f'**Matched in logs: {matched}/{len(REJECTIONS)} | Unmatched: {unmatched}**')
    print()
    print('| # | UTC time | Side | Sym | Posted | Bid | Ask | Spread bps | Att# | Slide above bid+$0.01 | v4 post | v4 prevents? |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for i, r in enumerate(rows, 1):
        if r.get('attempt') is None:
            print(f"| {i} | {r['reject_iso']} | {r['side']} | {r['sym']} | ${r['posted']:.2f} | _no log match_ | | | | | | _N/A_ |")
        else:
            slide_str = f"+${r['slide_above_bid_plus_1c']:.2f}" if r['slide_above_bid_plus_1c'] is not None else 'n/a'
            v4_str = f"${r['v4_would_post']:.2f}" if r['v4_would_post'] else '(slide)'
            v4_help_icon = 'YES' if r['v4_helps'] else 'no'
            print(
                f"| {i} | {r['reject_iso']} | {r['side']} | {r['sym']} | "
                f"${r['posted']:.2f} | ${r['bid']:.2f} | ${r['ask']:.2f} | "
                f"{r['spread_bps']:.1f} | {r['attempt']} | {slide_str} | {v4_str} | {v4_help_icon} |"
            )

    print()
    print('## Summary')
    n_buy = sum(1 for r in rows if r['side'] == 'BUY')
    n_sell = sum(1 for r in rows if r['side'] == 'SELL')
    n_buy_matched = sum(1 for r in rows if r['side'] == 'BUY' and r.get('attempt') is not None)
    n_v4_helps = sum(1 for r in rows if r['v4_helps'])
    n_buy_wide = sum(1 for r in rows if r['side'] == 'BUY' and r.get('attempt') is not None and r.get('spread_bps', 0) > 5.0)
    print(f'- BUY rejections: {n_buy} (matched in log: {n_buy_matched})')
    print(f'- SELL rejections: {n_sell} (v4 does NOT touch SELL side — would still happen)')
    print(f'- BUY rejections in wide-spread regime (>5bps): {n_buy_wide}')
    print(f'- BUY rejections v4 would prevent: {n_v4_helps}')
    if n_buy > 0:
        print(f'- v4 prevention rate (of BUY): {100 * n_v4_helps / n_buy:.0f}%')


if __name__ == '__main__':
    main()
