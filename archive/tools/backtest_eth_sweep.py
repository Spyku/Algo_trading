"""
ETH backtest: sweep confidence thresholds + trailing stop combos.
Same signal log, but test what happens at 70/75/80/85/90% min_confidence.
"""
import csv
import os
from datetime import datetime, timedelta

TRADING_FEE = 0.0011
HORIZON = 7
POSITION_USD = 2000
REGIME_WINDOW = 6
REGIME_SELL_THRESHOLD = 4
ENGINE_DIR = os.path.dirname(os.path.dirname(__file__))


def load_signals(path, asset):
    signals = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['asset'] != asset:
                continue
            ts = datetime.strptime(row['timestamp'][:16], '%Y-%m-%d %H:%M')
            price = float(row['price'])
            action = row['action']
            conf = float(row['confidence'])
            hour_key = ts.replace(minute=0, second=0)
            if signals and signals[-1][0].replace(minute=0, second=0) == hour_key:
                continue
            signals.append((ts, price, action, conf))
    return signals


def load_ohlcv(asset):
    path = os.path.join(ENGINE_DIR, 'data', f'{asset}_hourly_data.csv')
    ohlcv = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = row['datetime'][:16]
            ohlcv[dt] = {
                'open': float(row['open']), 'high': float(row['high']),
                'low': float(row['low']), 'close': float(row['close']),
            }
    return ohlcv


def simulate(sigs, ohlcv, min_conf, trailing_stop_pct=None, profit_target_pct=None,
             use_regime_filter=False):
    trades = []
    position = None

    for i, (ts, price, action, conf) in enumerate(sigs):
        if position is None:
            if action == 'BUY' and conf >= min_conf:
                if use_regime_filter:
                    recent = sigs[max(0, i - REGIME_WINDOW):i]
                    sell_count = sum(1 for _, _, a, _ in recent if a == 'SELL')
                    if sell_count >= REGIME_SELL_THRESHOLD:
                        continue
                position = {'entry_time': ts, 'entry_price': price, 'peak': price}
        else:
            # Check TS/TP via OHLCV
            if trailing_stop_pct is not None or profit_target_pct is not None:
                prev_ts = sigs[i - 1][0] if i > 0 else position['entry_time']
                check_hour = prev_ts.replace(minute=0, second=0) + timedelta(hours=1)
                current_hour = ts.replace(minute=0, second=0)
                exited = False
                while check_hour <= current_hour:
                    h_str = check_hour.strftime('%Y-%m-%d %H:%M')
                    candle = ohlcv.get(h_str)
                    if candle:
                        if candle['high'] > position['peak']:
                            position['peak'] = candle['high']
                        if profit_target_pct is not None:
                            target = position['entry_price'] * (1 + profit_target_pct / 100)
                            if candle['high'] >= target:
                                _close(trades, position, check_hour, target, 'TP')
                                position = None; exited = True; break
                        if trailing_stop_pct is not None:
                            stop = position['peak'] * (1 - trailing_stop_pct / 100)
                            if candle['low'] <= stop:
                                _close(trades, position, check_hour, stop, 'TS')
                                position = None; exited = True; break
                    check_hour += timedelta(hours=1)
                if exited:
                    continue

            if position is not None and action == 'SELL':
                _close(trades, position, ts, price, 'signal')
                position = None

    if position:
        last_ts, last_price, _, _ = sigs[-1]
        _close(trades, position, last_ts, last_price, 'open')
        trades[-1]['open'] = True

    return trades


def _close(trades, pos, exit_time, exit_price, reason):
    ep = pos['entry_price']
    gross = (exit_price - ep) / ep * 100
    net = gross - (TRADING_FEE * 2 * 100)
    pnl = POSITION_USD * net / 100
    hold = (exit_time - pos['entry_time']).total_seconds() / 3600
    trades.append({
        'entry_time': pos['entry_time'], 'exit_time': exit_time,
        'entry_price': ep, 'exit_price': exit_price,
        'gross_pct': gross, 'net_pct': net, 'pnl_usd': pnl,
        'hold_hours': hold, 'exit_reason': reason, 'peak': pos['peak'],
    })


def fmt(name, trades):
    if not trades:
        print(f"    {name:40s} | No trades")
        return
    n = len(trades)
    pnl = sum(t['pnl_usd'] for t in trades)
    pct = sum(t['net_pct'] for t in trades)
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] < 0]
    wr = len(wins) / n * 100
    aw = sum(t['pnl_usd'] for t in wins) / len(wins) if wins else 0
    al = sum(t['pnl_usd'] for t in losses) / len(losses) if losses else 0
    ah = sum(t['hold_hours'] for t in trades) / n
    reasons = {}
    for t in trades:
        r = t.get('exit_reason', '?')
        reasons[r] = reasons.get(r, 0) + 1
    rs = ' '.join(f"{k}:{v}" for k, v in sorted(reasons.items()))
    print(f"    {name:40s} | {n:2d} tr | ${pnl:+7.0f} ({pct:+5.2f}%) | "
          f"WR {wr:3.0f}% | W ${aw:+.0f} L ${al:+.0f} | {ah:4.1f}h | {rs}")


def print_trades_detail(name, trades):
    if not trades:
        return
    print(f"\n    {name}:")
    print(f"    {'Entry':>12s}  {'Exit':>12s}  {'Hold':>5s}  {'Why':>6s}  "
          f"{'Entry$':>9s}  {'Peak$':>9s}  {'Exit$':>9s}  {'Net%':>7s}  {'PnL$':>8s}")
    print(f"    {'-'*12}  {'-'*12}  {'-'*5}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*8}")
    for t in trades:
        m = ' *' if t.get('open') else ''
        print(f"    {t['entry_time'].strftime('%m-%d %H:%M'):>12s}  "
              f"{t['exit_time'].strftime('%m-%d %H:%M'):>12s}  "
              f"{t['hold_hours']:.0f}h".rjust(5) + f"  "
              f"{t['exit_reason']:>6s}  "
              f"{t['entry_price']:>9.2f}  {t.get('peak', t['entry_price']):>9.2f}  "
              f"{t['exit_price']:>9.2f}  {t['net_pct']:>+6.2f}%  ${t['pnl_usd']:>+7.2f}{m}")


# ── Main ─────────────────────────────────────────────────────────
signal_log = os.path.join(ENGINE_DIR, 'config', 'signal_log.csv')
sigs = load_signals(signal_log, 'ETH')
ohlcv = load_ohlcv('ETH')

# Count signals by type
buys = sum(1 for _, _, a, c in sigs if a == 'BUY')
sells = sum(1 for _, _, a, c in sigs if a == 'SELL')
holds = sum(1 for _, _, a, c in sigs if a == 'HOLD')

print("=" * 130)
print(f"  ETH STRATEGY SWEEP (horizon={HORIZON}h, position=${POSITION_USD:,})")
print(f"  Signals: {len(sigs)} total ({buys} BUY, {sells} SELL, {holds} HOLD)")
print(f"  Period: {sigs[0][0].strftime('%Y-%m-%d')} to {sigs[-1][0].strftime('%Y-%m-%d')}")
print("=" * 130)

# Show BUY signal distribution by confidence
print(f"\n  BUY signals by confidence:")
for thresh in [60, 70, 75, 80, 85, 90, 95]:
    count = sum(1 for _, _, a, c in sigs if a == 'BUY' and c >= thresh)
    print(f"    >= {thresh}%: {count} BUY signals")

CONF_LEVELS = [70, 75, 80, 85, 90]
TS_LEVELS = [None, 0.3, 0.5, 0.7, 1.0]

for min_conf in CONF_LEVELS:
    buy_count = sum(1 for _, _, a, c in sigs if a == 'BUY' and c >= min_conf)
    print(f"\n{'='*130}")
    print(f"  min_confidence = {min_conf}% ({buy_count} qualifying BUY signals)")
    print(f"{'='*130}")

    # Signal-only baselines
    print(f"\n  --- Signal-only exits ---")
    fmt('Baseline', simulate(sigs, ohlcv, min_conf))
    fmt('B: Regime filter', simulate(sigs, ohlcv, min_conf, use_regime_filter=True))

    # Trailing stop sweep
    print(f"\n  --- Trailing stop ---")
    for ts_pct in [0.3, 0.5, 0.7, 1.0]:
        fmt(f'TS {ts_pct}%', simulate(sigs, ohlcv, min_conf, trailing_stop_pct=ts_pct))

    # TS + regime filter
    print(f"\n  --- TS + regime filter (B) ---")
    for ts_pct in [0.3, 0.5, 0.7, 1.0]:
        fmt(f'B + TS {ts_pct}%',
            simulate(sigs, ohlcv, min_conf, trailing_stop_pct=ts_pct, use_regime_filter=True))

    # TS + PT + regime
    print(f"\n  --- TS + PT + regime (B) ---")
    for ts_pct in [0.5, 0.7]:
        for pt_pct in [1.5, 2.0, 3.0]:
            fmt(f'B + TS {ts_pct}% + PT {pt_pct}%',
                simulate(sigs, ohlcv, min_conf, trailing_stop_pct=ts_pct,
                         profit_target_pct=pt_pct, use_regime_filter=True))

    # Print best trades detail
    best_pnl = -999999
    best_name = None
    best_trades = None
    for ts_pct in [None, 0.3, 0.5, 0.7, 1.0]:
        for regime in [False, True]:
            for pt_pct in [None, 1.5, 2.0, 3.0]:
                t = simulate(sigs, ohlcv, min_conf, trailing_stop_pct=ts_pct,
                             profit_target_pct=pt_pct, use_regime_filter=regime)
                if not t:
                    continue
                pnl = sum(x['pnl_usd'] for x in t)
                parts = []
                if regime: parts.append('B')
                if ts_pct: parts.append(f'TS {ts_pct}%')
                if pt_pct: parts.append(f'PT {pt_pct}%')
                name = ' + '.join(parts) if parts else 'Baseline'
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_name = name
                    best_trades = t

    if best_trades:
        print_trades_detail(f'BEST @ {min_conf}%: {best_name}', best_trades)

    # Also show baseline trades for context
    baseline_trades = simulate(sigs, ohlcv, min_conf)
    if baseline_trades:
        print_trades_detail(f'Baseline @ {min_conf}%', baseline_trades)

print(f"\n  TS=trailing stop | TP=profit target | signal=SELL signal | open=still held")
print("=" * 130)
