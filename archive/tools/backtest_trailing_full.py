"""
Full 2-week trailing stop backtest for BTC + ETH.
Generates signals using the production model over 336h, then simulates
baseline vs trailing stop vs regime filter.
"""
import sys, os, csv, warnings
from datetime import datetime, timedelta

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)
os.chdir(ENGINE_DIR)

warnings.filterwarnings('ignore')

TRADING_FEE = 0.0011
REPLAY_HOURS = 336  # 2 weeks
REGIME_WINDOW = 6
REGIME_SELL_THRESHOLD = 4


def load_production_config(asset, horizon):
    """Read production CSV and return model config for (asset, horizon)."""
    csv_path = os.path.join(ENGINE_DIR, 'models', 'crypto_doohan_v1_7_1_production.csv')
    best = None
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['coin'] == asset and int(row['horizon']) == horizon:
                score = float(row.get('combined_score', 0))
                if best is None or score > best.get('_score', 0):
                    best = dict(row)
                    best['_score'] = score
    return best


def load_ohlcv(asset):
    path = os.path.join(ENGINE_DIR, 'data', f'{asset}_hourly_data.csv')
    ohlcv = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = row['datetime'][:16]
            ohlcv[dt] = {
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
            }
    return ohlcv


def simulate(signals, ohlcv, min_conf, position_usd,
             trailing_stop_pct=None, profit_target_pct=None, use_regime_filter=False):
    trades = []
    position = None

    for i, sig in enumerate(signals):
        ts = datetime.strptime(sig['datetime'], '%Y-%m-%d %H:%M')
        price = sig['close']
        action = sig['signal']
        conf = sig['confidence']

        if position is None:
            if action == 'BUY' and conf >= min_conf:
                if use_regime_filter:
                    recent = signals[max(0, i - REGIME_WINDOW):i]
                    sell_count = sum(1 for s in recent if s['signal'] == 'SELL')
                    if sell_count >= REGIME_SELL_THRESHOLD:
                        continue
                position = {'entry_time': ts, 'entry_price': price, 'peak': price}
        else:
            # Check TS/TP via OHLCV between signals
            if trailing_stop_pct is not None or profit_target_pct is not None:
                prev_ts = datetime.strptime(signals[i-1]['datetime'], '%Y-%m-%d %H:%M') if i > 0 else position['entry_time']
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
                                _close(trades, position, check_hour, target, position_usd, 'TP')
                                position = None; exited = True; break
                        if trailing_stop_pct is not None:
                            stop = position['peak'] * (1 - trailing_stop_pct / 100)
                            if candle['low'] <= stop:
                                _close(trades, position, check_hour, stop, position_usd, 'TS')
                                position = None; exited = True; break
                    check_hour += timedelta(hours=1)
                if exited:
                    continue

            if position is not None and action == 'SELL':
                _close(trades, position, ts, price, position_usd, 'signal')
                position = None

    if position:
        last = signals[-1]
        last_ts = datetime.strptime(last['datetime'], '%Y-%m-%d %H:%M')
        _close(trades, position, last_ts, last['close'], position_usd, 'open')
        trades[-1]['open'] = True

    return trades


def _close(trades, pos, exit_time, exit_price, position_usd, reason):
    ep = pos['entry_price']
    gross = (exit_price - ep) / ep * 100
    net = gross - (TRADING_FEE * 2 * 100)
    pnl = position_usd * net / 100
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
        reasons[t['exit_reason']] = reasons.get(t['exit_reason'], 0) + 1
    rs = ' '.join(f"{k}:{v}" for k, v in sorted(reasons.items()))
    print(f"    {name:40s} | {n:2d} tr | ${pnl:+8.0f} ({pct:+6.2f}%) | "
          f"WR {wr:3.0f}% | W ${aw:+.0f} L ${al:+.0f} | {ah:4.1f}h | {rs}")


def print_trades_detail(name, trades):
    if not trades:
        return
    print(f"\n    {name}:")
    print(f"    {'Entry':>16s}  {'Exit':>16s}  {'Hold':>5s}  {'Why':>6s}  "
          f"{'Entry$':>10s}  {'Peak$':>10s}  {'Exit$':>10s}  {'Net%':>7s}  {'PnL$':>9s}")
    print(f"    {'-'*16}  {'-'*16}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*9}")
    for t in trades:
        m = ' *' if t.get('open') else ''
        print(f"    {t['entry_time'].strftime('%Y-%m-%d %H:%M'):>16s}  "
              f"{t['exit_time'].strftime('%Y-%m-%d %H:%M'):>16s}  "
              f"{t['hold_hours']:.0f}h".rjust(5) + f"  "
              f"{t['exit_reason']:>6s}  "
              f"{t['entry_price']:>10.2f}  {t.get('peak', t['entry_price']):>10.2f}  "
              f"{t['exit_price']:>10.2f}  {t['net_pct']:>+6.2f}%  ${t['pnl_usd']:>+8.2f}{m}")


# ── Main ──────────────────────────────────────────────────────────────
from crypto_trading_system_doohan import generate_signals

ASSETS = {
    'BTC': {'horizon': 6, 'min_confidence': 85, 'position_usd': 12000},
    'ETH': {'horizon': 7, 'min_confidence': 90, 'position_usd': 2000},
}

print("=" * 135)
print(f"  FULL 2-WEEK TRAILING STOP BACKTEST ({REPLAY_HOURS}h replay)")
print("=" * 135)

for asset, cfg in ASSETS.items():
    horizon = cfg['horizon']
    min_conf = cfg['min_confidence']
    pos_usd = cfg['position_usd']

    # Load production model config
    prod = load_production_config(asset, horizon)
    if not prod:
        print(f"\n  {asset}: No production model for horizon {horizon}h — skipping")
        continue

    models = prod['models'].split('+')
    window = int(prod['best_window'])
    gamma = float(prod['gamma'])
    features = prod['optimal_features'].split(',')
    n_features = int(prod['n_features'])

    print(f"\n{'='*135}")
    print(f"  {asset} — {'+'.join(models)} w={window}h g={gamma} f={n_features} "
          f"horizon={horizon}h min_conf={min_conf}% position=${pos_usd:,}")
    print(f"{'='*135}")

    # Generate signals
    sigs = generate_signals(asset, models, window, replay_hours=REPLAY_HOURS,
                            feature_override=features, horizon=horizon, gamma=gamma)

    if not sigs:
        print(f"  No signals generated for {asset}")
        continue

    buys = sum(1 for s in sigs if s['signal'] == 'BUY')
    sells = sum(1 for s in sigs if s['signal'] == 'SELL')
    holds = sum(1 for s in sigs if s['signal'] == 'HOLD')
    print(f"  {len(sigs)} signals: {buys} BUY, {sells} SELL, {holds} HOLD")
    print(f"  Period: {sigs[0]['datetime']} to {sigs[-1]['datetime']}")

    ohlcv = load_ohlcv(asset)

    # ── Baselines ──
    print(f"\n  --- Baselines ---")
    baseline = simulate(sigs, ohlcv, min_conf, pos_usd)
    baseline_b = simulate(sigs, ohlcv, min_conf, pos_usd, use_regime_filter=True)
    fmt('Baseline (signal exit)', baseline)
    fmt('B: Regime filter', baseline_b)

    # ── Trailing stop sweep ──
    print(f"\n  --- Trailing stop only ---")
    for ts in [0.3, 0.5, 0.7, 1.0]:
        fmt(f'TS {ts}%', simulate(sigs, ohlcv, min_conf, pos_usd, trailing_stop_pct=ts))

    # ── TS + Regime ──
    print(f"\n  --- TS + Regime (B) ---")
    for ts in [0.3, 0.5, 0.7, 1.0]:
        fmt(f'B + TS {ts}%', simulate(sigs, ohlcv, min_conf, pos_usd,
            trailing_stop_pct=ts, use_regime_filter=True))

    # ── TS + PT + Regime ──
    print(f"\n  --- TS + PT + Regime (B) ---")
    for ts in [0.5, 0.7]:
        for pt in [1.5, 2.0, 3.0]:
            fmt(f'B + TS {ts}% + PT {pt}%', simulate(sigs, ohlcv, min_conf, pos_usd,
                trailing_stop_pct=ts, profit_target_pct=pt, use_regime_filter=True))

    # ── Detail: baseline + best ──
    print_trades_detail('Baseline', baseline)

    best_pnl = -999999
    best_name = best_trades = None
    for ts in [None, 0.3, 0.5, 0.7, 1.0]:
        for pt in [None, 1.5, 2.0, 3.0]:
            for regime in [False, True]:
                t = simulate(sigs, ohlcv, min_conf, pos_usd,
                             trailing_stop_pct=ts, profit_target_pct=pt, use_regime_filter=regime)
                if not t: continue
                pnl = sum(x['pnl_usd'] for x in t)
                parts = []
                if regime: parts.append('B')
                if ts: parts.append(f'TS {ts}%')
                if pt: parts.append(f'PT {pt}%')
                name = ' + '.join(parts) if parts else 'Baseline'
                if pnl > best_pnl:
                    best_pnl = pnl; best_name = name; best_trades = t

    if best_trades:
        print_trades_detail(f'BEST: {best_name}', best_trades)

print(f"\n  TS=trailing stop | TP=profit target | signal=SELL signal | open=still held")
print("=" * 135)
