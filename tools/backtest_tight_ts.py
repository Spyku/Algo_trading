"""
Test tight trailing stops (0.25%, 0.3%) on the full 2-week replay for BTC + ETH.
"""
import sys, os, csv, warnings
from datetime import datetime, timedelta

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)
os.chdir(ENGINE_DIR)
warnings.filterwarnings('ignore')

TRADING_FEE = 0.0011
REPLAY_HOURS = 336
REGIME_WINDOW = 6
REGIME_SELL_THRESHOLD = 4


def load_production_config(asset, horizon):
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
            ohlcv[dt] = {'high': float(row['high']), 'low': float(row['low']), 'close': float(row['close'])}
    return ohlcv


def simulate(signals, ohlcv, min_conf, position_usd,
             trailing_stop_pct=None, use_regime_filter=False):
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
                    if sum(1 for s in recent if s['signal'] == 'SELL') >= REGIME_SELL_THRESHOLD:
                        continue
                position = {'entry_time': ts, 'entry_price': price, 'peak': price}
        else:
            if trailing_stop_pct is not None:
                prev_ts = datetime.strptime(signals[i-1]['datetime'], '%Y-%m-%d %H:%M') if i > 0 else position['entry_time']
                check = prev_ts.replace(minute=0, second=0) + timedelta(hours=1)
                current = ts.replace(minute=0, second=0)
                exited = False
                while check <= current:
                    candle = ohlcv.get(check.strftime('%Y-%m-%d %H:%M'))
                    if candle:
                        if candle['high'] > position['peak']:
                            position['peak'] = candle['high']
                        stop = position['peak'] * (1 - trailing_stop_pct / 100)
                        if candle['low'] <= stop:
                            _close(trades, position, check, stop, position_usd, 'TS')
                            position = None; exited = True; break
                    check += timedelta(hours=1)
                if exited:
                    continue
            if position is not None and action == 'SELL':
                _close(trades, position, ts, price, position_usd, 'signal')
                position = None
    if position:
        last = signals[-1]
        _close(trades, position, datetime.strptime(last['datetime'], '%Y-%m-%d %H:%M'),
               last['close'], position_usd, 'open')
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


def detail(name, trades):
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


# ── Main ──
from crypto_trading_system_doohan import generate_signals

ASSETS = {
    'BTC': {'horizon': 6, 'min_confidence': 85, 'position_usd': 12000},
    'ETH': {'horizon': 7, 'min_confidence': 90, 'position_usd': 2000},
}

TS_LEVELS = [0.25, 0.3]

print("=" * 135)
print(f"  TIGHT TRAILING STOP TEST: {TS_LEVELS}%  ({REPLAY_HOURS}h replay)")
print("=" * 135)

for asset, cfg in ASSETS.items():
    prod = load_production_config(asset, cfg['horizon'])
    if not prod:
        continue
    models = prod['models'].split('+')
    window = int(prod['best_window'])
    gamma = float(prod['gamma'])
    features = prod['optimal_features'].split(',')

    print(f"\n{'='*135}")
    print(f"  {asset} — {'+'.join(models)} w={window}h g={gamma} f={len(features)} "
          f"horizon={cfg['horizon']}h conf={cfg['min_confidence']}% pos=${cfg['position_usd']:,}")
    print(f"{'='*135}")

    sigs = generate_signals(asset, models, window, replay_hours=REPLAY_HOURS,
                            feature_override=features, horizon=cfg['horizon'], gamma=gamma)
    if not sigs:
        continue

    buys = sum(1 for s in sigs if s['signal'] == 'BUY')
    sells = sum(1 for s in sigs if s['signal'] == 'SELL')
    print(f"  {len(sigs)} signals: {buys} BUY, {sells} SELL, {len(sigs)-buys-sells} HOLD\n")

    ohlcv = load_ohlcv(asset)
    min_conf = cfg['min_confidence']
    pos_usd = cfg['position_usd']

    baseline = simulate(sigs, ohlcv, min_conf, pos_usd)
    baseline_b = simulate(sigs, ohlcv, min_conf, pos_usd, use_regime_filter=True)

    print(f"  --- Summary ---")
    fmt('Baseline (signal exit)', baseline)
    fmt('B: Regime filter', baseline_b)
    for ts in TS_LEVELS:
        fmt(f'TS {ts}%', simulate(sigs, ohlcv, min_conf, pos_usd, trailing_stop_pct=ts))
    for ts in TS_LEVELS:
        fmt(f'B + TS {ts}%', simulate(sigs, ohlcv, min_conf, pos_usd,
            trailing_stop_pct=ts, use_regime_filter=True))

    # Detailed trades
    detail('Baseline', baseline)
    for ts in TS_LEVELS:
        t = simulate(sigs, ohlcv, min_conf, pos_usd, trailing_stop_pct=ts)
        detail(f'TS {ts}%', t)
    for ts in TS_LEVELS:
        t = simulate(sigs, ohlcv, min_conf, pos_usd, trailing_stop_pct=ts, use_regime_filter=True)
        detail(f'B + TS {ts}%', t)

print(f"\n  TS=trailing stop | signal=SELL signal | open=still held")
print("=" * 135)
