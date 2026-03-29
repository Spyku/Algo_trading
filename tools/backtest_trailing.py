"""
Backtest trailing stop + profit target + regime filter using actual hourly OHLCV.
Compares against baseline (signal-only exits).
"""
import csv
import os
from datetime import datetime

TRADING_FEE = 0.0011
CONFIGS = {
    'BTC': {'min_confidence': 85, 'horizon': 6, 'position_usd': 12000},
    'ETH': {'min_confidence': 90, 'horizon': 7, 'position_usd': 2000},
}
REGIME_WINDOW = 6
REGIME_SELL_THRESHOLD = 4

ENGINE_DIR = os.path.dirname(os.path.dirname(__file__))


def load_signals(path):
    signals = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            asset = row['asset']
            ts = datetime.strptime(row['timestamp'][:16], '%Y-%m-%d %H:%M')
            price = float(row['price'])
            action = row['action']
            conf = float(row['confidence'])
            if asset not in signals:
                signals[asset] = []
            hour_key = ts.replace(minute=0, second=0)
            if signals[asset] and signals[asset][-1][0].replace(minute=0, second=0) == hour_key:
                continue
            signals[asset].append((ts, price, action, conf))
    return signals


def load_ohlcv(asset):
    """Load hourly OHLCV, return dict: datetime_str -> (open, high, low, close)."""
    path = os.path.join(ENGINE_DIR, 'data', f'{asset}_hourly_data.csv')
    ohlcv = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = row['datetime'][:16]  # '2026-03-23 07:00'
            ohlcv[dt] = {
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
            }
    return ohlcv


def simulate(asset_signals, ohlcv, min_conf, horizon, position_usd,
             trailing_stop_pct=None, profit_target_pct=None, use_regime_filter=False):
    """
    Simulate trading.
    - trailing_stop_pct: exit if price drops X% from peak since entry (e.g. 0.5)
    - profit_target_pct: exit if price rises X% from entry (e.g. 1.5)
    - Both checked at hourly candle lows/highs between signals.
    """
    trades = []
    position = None  # (entry_time, entry_price, peak_price)

    for i, (ts, price, action, conf) in enumerate(asset_signals):
        hour_str = ts.strftime('%Y-%m-%d %H:00')[:16]

        if position is None:
            # ── Entry logic ──
            if action == 'BUY' and conf >= min_conf:
                if use_regime_filter:
                    recent = asset_signals[max(0, i - REGIME_WINDOW):i]
                    sell_count = sum(1 for _, _, a, _ in recent if a == 'SELL')
                    if sell_count >= REGIME_SELL_THRESHOLD:
                        continue
                position = {'entry_time': ts, 'entry_price': price, 'peak': price, 'exit_reason': None}
        else:
            # ── Check trailing stop / profit target using OHLCV between prev and current signal ──
            if trailing_stop_pct is not None or profit_target_pct is not None:
                # Walk through hourly candles from entry (or last checked) to current signal
                prev_ts = asset_signals[i - 1][0] if i > 0 else position['entry_time']
                check_hour = prev_ts.replace(minute=0, second=0)
                from datetime import timedelta
                check_hour += timedelta(hours=1)
                current_hour = ts.replace(minute=0, second=0)

                exited = False
                while check_hour <= current_hour:
                    h_str = check_hour.strftime('%Y-%m-%d %H:%M')
                    candle = ohlcv.get(h_str)
                    if candle:
                        # Update peak
                        if candle['high'] > position['peak']:
                            position['peak'] = candle['high']

                        # Check profit target (hit during this candle?)
                        if profit_target_pct is not None:
                            target_price = position['entry_price'] * (1 + profit_target_pct / 100)
                            if candle['high'] >= target_price:
                                exit_price = target_price
                                _close_trade(trades, position, check_hour, exit_price, position_usd, 'TP')
                                position = None
                                exited = True
                                break

                        # Check trailing stop (hit during this candle?)
                        if trailing_stop_pct is not None:
                            stop_price = position['peak'] * (1 - trailing_stop_pct / 100)
                            if candle['low'] <= stop_price:
                                exit_price = stop_price
                                _close_trade(trades, position, check_hour, exit_price, position_usd, 'TS')
                                position = None
                                exited = True
                                break

                    check_hour += timedelta(hours=1)

                if exited:
                    continue

            # ── Signal-based exit (SELL signal) ──
            if position is not None and action == 'SELL':
                _close_trade(trades, position, ts, price, position_usd, 'signal')
                position = None

    # Mark-to-market if still open
    if position:
        last_ts, last_price, _, _ = asset_signals[-1]
        _close_trade(trades, position, last_ts, last_price, position_usd, 'open')
        trades[-1]['open'] = True

    return trades


def _close_trade(trades, position, exit_time, exit_price, position_usd, reason):
    entry_price = position['entry_price']
    gross_pct = (exit_price - entry_price) / entry_price * 100
    net_pct = gross_pct - (TRADING_FEE * 2 * 100)
    pnl_usd = position_usd * net_pct / 100
    hold_hours = (exit_time - position['entry_time']).total_seconds() / 3600
    trades.append({
        'entry_time': position['entry_time'],
        'exit_time': exit_time,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'gross_pct': gross_pct,
        'net_pct': net_pct,
        'pnl_usd': pnl_usd,
        'hold_hours': hold_hours,
        'exit_reason': reason,
        'peak': position['peak'],
    })


def print_summary(name, trades, position_usd):
    if not trades:
        print(f"  {name:35s} | No trades")
        return
    n = len(trades)
    total_pnl = sum(t['pnl_usd'] for t in trades)
    total_pct = sum(t['net_pct'] for t in trades)
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] < 0]
    wr = len(wins) / n * 100
    avg_win = sum(t['pnl_usd'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl_usd'] for t in losses) / len(losses) if losses else 0
    avg_hold = sum(t['hold_hours'] for t in trades) / n
    reasons = {}
    for t in trades:
        r = t.get('exit_reason', '?')
        reasons[r] = reasons.get(r, 0) + 1
    reason_str = ' '.join(f"{k}:{v}" for k, v in sorted(reasons.items()))

    print(f"  {name:35s} | {n:2d} tr | ${total_pnl:+8.0f} ({total_pct:+5.2f}%) | "
          f"WR {wr:3.0f}% | W ${avg_win:+.0f} L ${avg_loss:+.0f} | "
          f"{avg_hold:4.1f}h | {reason_str}")


def print_trades_detail(name, trades):
    print(f"\n  {name}:")
    print(f"  {'Entry':>12s}  {'Exit':>12s}  {'Hold':>5s}  {'Reason':>6s}  {'Entry$':>10s}  "
          f"{'Peak$':>10s}  {'Exit$':>10s}  {'Net%':>7s}  {'PnL$':>9s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*9}")
    for t in trades:
        marker = ' *' if t.get('open') else ''
        print(f"  {t['entry_time'].strftime('%m-%d %H:%M'):>12s}  "
              f"{t['exit_time'].strftime('%m-%d %H:%M'):>12s}  "
              f"{t['hold_hours']:.0f}h".rjust(5) + f"  "
              f"{t.get('exit_reason','?'):>6s}  "
              f"{t['entry_price']:>10.2f}  "
              f"{t.get('peak', t['entry_price']):>10.2f}  "
              f"{t['exit_price']:>10.2f}  "
              f"{t['net_pct']:>+6.2f}%  "
              f"${t['pnl_usd']:>+8.2f}{marker}")


# ── Main ────────────────────────────────────────────────────────────────
signal_log = os.path.join(ENGINE_DIR, 'config', 'signal_log.csv')
signals = load_signals(signal_log)

# Trailing stop levels to test
TS_LEVELS = [0.3, 0.5, 0.7, 1.0]
# Profit target levels to test
PT_LEVELS = [1.0, 1.5, 2.0, 3.0]

print("=" * 130)
print("  TRAILING STOP + PROFIT TARGET BACKTEST")
print(f"  Trailing stops: {TS_LEVELS}%  |  Profit targets: {PT_LEVELS}%  |  Regime filter: B ({REGIME_SELL_THRESHOLD}/{REGIME_WINDOW} SELL)")
print("=" * 130)

for asset in ['BTC', 'ETH']:
    if asset not in signals:
        continue
    cfg = CONFIGS[asset]
    sigs = signals[asset]
    ohlcv = load_ohlcv(asset)

    print(f"\n{'='*130}")
    print(f"  {asset} (horizon={cfg['horizon']}h, min_conf={cfg['min_confidence']}%, position=${cfg['position_usd']:,})")
    print(f"{'='*130}")

    # ── Baseline ──
    print(f"\n  --- Signal-only exits ---")
    baseline = simulate(sigs, ohlcv, cfg['min_confidence'], cfg['horizon'], cfg['position_usd'])
    baseline_b = simulate(sigs, ohlcv, cfg['min_confidence'], cfg['horizon'], cfg['position_usd'],
                          use_regime_filter=True)
    print_summary('Baseline (signal exit)', baseline, cfg['position_usd'])
    print_summary('B: Regime + signal exit', baseline_b, cfg['position_usd'])

    # ── Trailing stop only ──
    print(f"\n  --- Trailing stop (exit on TS or SELL signal, whichever first) ---")
    for ts_pct in TS_LEVELS:
        t = simulate(sigs, ohlcv, cfg['min_confidence'], cfg['horizon'], cfg['position_usd'],
                     trailing_stop_pct=ts_pct)
        print_summary(f'TS {ts_pct}%', t, cfg['position_usd'])

    # ── Profit target only ──
    print(f"\n  --- Profit target (exit on TP or SELL signal, whichever first) ---")
    for pt_pct in PT_LEVELS:
        t = simulate(sigs, ohlcv, cfg['min_confidence'], cfg['horizon'], cfg['position_usd'],
                     profit_target_pct=pt_pct)
        print_summary(f'PT {pt_pct}%', t, cfg['position_usd'])

    # ── Combined: trailing stop + profit target ──
    print(f"\n  --- TS + PT combos ---")
    for ts_pct in [0.5, 0.7, 1.0]:
        for pt_pct in [1.5, 2.0, 3.0]:
            t = simulate(sigs, ohlcv, cfg['min_confidence'], cfg['horizon'], cfg['position_usd'],
                         trailing_stop_pct=ts_pct, profit_target_pct=pt_pct)
            print_summary(f'TS {ts_pct}% + PT {pt_pct}%', t, cfg['position_usd'])

    # ── Best combos with regime filter ──
    print(f"\n  --- Best combos + regime filter (B) ---")
    for ts_pct in [0.5, 0.7, 1.0]:
        for pt_pct in [1.5, 2.0, 3.0]:
            t = simulate(sigs, ohlcv, cfg['min_confidence'], cfg['horizon'], cfg['position_usd'],
                         trailing_stop_pct=ts_pct, profit_target_pct=pt_pct, use_regime_filter=True)
            print_summary(f'B + TS {ts_pct}% + PT {pt_pct}%', t, cfg['position_usd'])

    # ── Print detailed trades for best few ──
    print(f"\n  --- Detailed trades ---")
    print_trades_detail('Baseline', baseline)

    # Find best combo
    best_pnl = -999999
    best_cfg = None
    for ts_pct in [0.5, 0.7, 1.0]:
        for pt_pct in [1.5, 2.0, 3.0]:
            for regime in [False, True]:
                t = simulate(sigs, ohlcv, cfg['min_confidence'], cfg['horizon'], cfg['position_usd'],
                             trailing_stop_pct=ts_pct, profit_target_pct=pt_pct, use_regime_filter=regime)
                pnl = sum(x['pnl_usd'] for x in t)
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_cfg = (ts_pct, pt_pct, regime, t)

    if best_cfg:
        ts_pct, pt_pct, regime, t = best_cfg
        r_str = ' + B' if regime else ''
        print_trades_detail(f'Best: TS {ts_pct}% + PT {pt_pct}%{r_str}', t)

print(f"\n  TS=trailing stop exit | TP=profit target exit | signal=SELL signal exit | open=still held")
print(f"  * = still open at end of period")
print("=" * 130)
