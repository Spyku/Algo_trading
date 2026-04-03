"""
Backtest dynamic confidence: raise min_confidence when recent signals are bearish.
Uses full 336h replay with production model signals.
"""
import sys, os, csv, warnings
from datetime import datetime, timedelta

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)
os.chdir(ENGINE_DIR)
warnings.filterwarnings('ignore')

TRADING_FEE = 0.0011
REPLAY_HOURS = 336


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
            ohlcv[row['datetime'][:16]] = {
                'high': float(row['high']), 'low': float(row['low']), 'close': float(row['close'])}
    return ohlcv


def get_buy_ratio(signals, idx, lookback):
    """BUY ratio over last `lookback` signals."""
    start = max(0, idx - lookback)
    window = signals[start:idx]
    if not window:
        return 0.5
    buys = sum(1 for s in window if s['signal'] == 'BUY')
    return buys / len(window)


def simulate(signals, ohlcv, base_conf, position_usd, dynamic_conf=None):
    """
    dynamic_conf: list of (buy_ratio_threshold, confidence) tuples, checked high-to-low.
    Example: [(0.40, 85), (0.20, 92), (0.0, 97)]
    Means: if BUY ratio > 40% use 85%, 20-40% use 92%, <20% use 97%.
    If None, use base_conf for all.
    """
    trades = []
    position = None

    for i, sig in enumerate(signals):
        ts = datetime.strptime(sig['datetime'], '%Y-%m-%d %H:%M')
        price = sig['close']
        action = sig['signal']
        conf = sig['confidence']

        if position is None:
            if action == 'BUY':
                # Determine effective min_confidence
                if dynamic_conf:
                    buy_ratio = get_buy_ratio(signals, i, dynamic_conf['lookback'])
                    eff_conf = base_conf
                    for thresh, new_conf in dynamic_conf['tiers']:
                        if buy_ratio >= thresh:
                            eff_conf = new_conf
                            break
                else:
                    eff_conf = base_conf
                    buy_ratio = None

                if conf >= eff_conf:
                    position = {
                        'entry_time': ts, 'entry_price': price, 'peak': price,
                        'eff_conf': eff_conf, 'buy_ratio': buy_ratio,
                    }
        else:
            if action == 'SELL':
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
        'hold_hours': hold, 'exit_reason': reason,
        'eff_conf': pos.get('eff_conf'), 'buy_ratio': pos.get('buy_ratio'),
    })


def fmt(name, trades):
    if not trades:
        print(f"    {name:50s} | No trades")
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
    # Count blocked (trades that baseline had but this didn't)
    print(f"    {name:50s} | {n:2d} tr | ${pnl:+8.0f} ({pct:+6.2f}%) | "
          f"WR {wr:3.0f}% | W ${aw:+.0f} L ${al:+.0f} | {ah:4.1f}h")


def detail(name, trades):
    if not trades:
        return
    print(f"\n    {name}:")
    print(f"    {'Entry':>16s}  {'Exit':>16s}  {'Hold':>5s}  "
          f"{'BuyR':>5s}  {'EffC':>5s}  {'Conf':>5s}  "
          f"{'Entry$':>10s}  {'Exit$':>10s}  {'Net%':>7s}  {'PnL$':>9s}")
    print(f"    {'-'*16}  {'-'*16}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*9}")
    for t in trades:
        m = ' *' if t.get('open') else ''
        br = f"{t['buy_ratio']:.0%}" if t.get('buy_ratio') is not None else '  -'
        ec = f"{t['eff_conf']:.0f}%" if t.get('eff_conf') is not None else '  -'
        # Reconstruct original confidence from entry
        print(f"    {t['entry_time'].strftime('%Y-%m-%d %H:%M'):>16s}  "
              f"{t['exit_time'].strftime('%Y-%m-%d %H:%M'):>16s}  "
              f"{t['hold_hours']:.0f}h".rjust(5) + f"  "
              f"{br:>5s}  {ec:>5s}  "
              f"{'':>5s}  "
              f"{t['entry_price']:>10.2f}  {t['exit_price']:>10.2f}  "
              f"{t['net_pct']:>+6.2f}%  ${t['pnl_usd']:>+8.2f}{m}")


# ── Main ──
from crypto_trading_system_doohan import generate_signals

ASSETS = {
    'BTC': {'horizon': 6, 'min_confidence': 85, 'position_usd': 12000},
    'ETH': {'horizon': 7, 'min_confidence': 90, 'position_usd': 2000},
}

# Dynamic confidence configs to test
# Format: {'lookback': N, 'tiers': [(ratio_threshold, conf), ...]} — checked high-to-low
DYNAMIC_CONFIGS = {
    # Lookback window variations
    '12h L12': {'lookback': 12, 'tiers': [(0.40, 85), (0.20, 92), (0.0, 97)]},
    '12h L12 soft': {'lookback': 12, 'tiers': [(0.40, 85), (0.20, 90), (0.0, 95)]},
    '12h L12 hard': {'lookback': 12, 'tiers': [(0.40, 85), (0.20, 95), (0.0, 99)]},
    '6h L6': {'lookback': 6, 'tiers': [(0.40, 85), (0.20, 92), (0.0, 97)]},
    '6h L6 soft': {'lookback': 6, 'tiers': [(0.40, 85), (0.20, 90), (0.0, 95)]},
    '24h L24': {'lookback': 24, 'tiers': [(0.40, 85), (0.20, 92), (0.0, 97)]},
    '24h L24 soft': {'lookback': 24, 'tiers': [(0.40, 85), (0.20, 90), (0.0, 95)]},
    # Different thresholds
    '12h 30/15': {'lookback': 12, 'tiers': [(0.30, 85), (0.15, 92), (0.0, 97)]},
    '12h 50/25': {'lookback': 12, 'tiers': [(0.50, 85), (0.25, 92), (0.0, 97)]},
    # 2-tier (simpler)
    '12h 2tier': {'lookback': 12, 'tiers': [(0.25, 85), (0.0, 95)]},
    '12h 2tier hard': {'lookback': 12, 'tiers': [(0.25, 85), (0.0, 97)]},
    '6h 2tier': {'lookback': 6, 'tiers': [(0.25, 85), (0.0, 95)]},
}

# ETH versions (base_conf=90)
DYNAMIC_CONFIGS_ETH = {
    '12h L12': {'lookback': 12, 'tiers': [(0.40, 90), (0.20, 95), (0.0, 99)]},
    '12h L12 soft': {'lookback': 12, 'tiers': [(0.40, 90), (0.20, 93), (0.0, 96)]},
    '6h L6': {'lookback': 6, 'tiers': [(0.40, 90), (0.20, 95), (0.0, 99)]},
    '24h L24': {'lookback': 24, 'tiers': [(0.40, 90), (0.20, 95), (0.0, 99)]},
    '12h 2tier': {'lookback': 12, 'tiers': [(0.25, 90), (0.0, 96)]},
    '6h 2tier': {'lookback': 6, 'tiers': [(0.25, 90), (0.0, 96)]},
}

print("=" * 135)
print(f"  DYNAMIC CONFIDENCE BACKTEST ({REPLAY_HOURS}h replay)")
print(f"  Raise min_confidence when recent BUY ratio is low (bearish regime)")
print("=" * 135)

for asset, cfg in ASSETS.items():
    prod = load_production_config(asset, cfg['horizon'])
    if not prod:
        continue
    models = prod['models'].split('+')
    window = int(prod['best_window'])
    gamma = float(prod['gamma'])
    features = prod['optimal_features'].split(',')
    base_conf = cfg['min_confidence']
    pos_usd = cfg['position_usd']

    print(f"\n{'='*135}")
    print(f"  {asset} — {'+'.join(models)} w={window}h g={gamma} f={len(features)} "
          f"horizon={cfg['horizon']}h base_conf={base_conf}% pos=${pos_usd:,}")
    print(f"{'='*135}")

    sigs = generate_signals(asset, models, window, replay_hours=REPLAY_HOURS,
                            feature_override=features, horizon=cfg['horizon'], gamma=gamma)
    if not sigs:
        continue

    buys = sum(1 for s in sigs if s['signal'] == 'BUY')
    sells = sum(1 for s in sigs if s['signal'] == 'SELL')
    print(f"  {len(sigs)} signals: {buys} BUY, {sells} SELL, {len(sigs)-buys-sells} HOLD\n")

    ohlcv = load_ohlcv(asset)
    dyn_configs = DYNAMIC_CONFIGS if asset == 'BTC' else DYNAMIC_CONFIGS_ETH

    # Baseline
    print(f"  --- Summary ---")
    baseline = simulate(sigs, ohlcv, base_conf, pos_usd)
    fmt(f'Baseline (fixed {base_conf}%)', baseline)
    print()

    # Dynamic configs
    results = []
    for name, dc in dyn_configs.items():
        t = simulate(sigs, ohlcv, base_conf, pos_usd, dynamic_conf=dc)
        fmt(f'Dynamic: {name}', t)
        results.append((name, dc, t, sum(x['pnl_usd'] for x in t) if t else -999999))

    # Find best
    results.sort(key=lambda x: x[3], reverse=True)
    best_name, best_dc, best_trades, best_pnl = results[0]

    # Detail for baseline and best
    detail('Baseline', baseline)
    detail(f'BEST: {best_name}', best_trades)

    # Show which trades were blocked
    if best_trades:
        baseline_entries = {t['entry_time'] for t in baseline}
        best_entries = {t['entry_time'] for t in best_trades}
        blocked = baseline_entries - best_entries
        if blocked:
            print(f"\n    Trades BLOCKED by dynamic conf ({best_name}):")
            for t in baseline:
                if t['entry_time'] in blocked:
                    print(f"      {t['entry_time'].strftime('%Y-%m-%d %H:%M')}  "
                          f"${t['pnl_usd']:>+8.2f}  {'(was a loss)' if t['pnl_usd'] < 0 else '(was a win)'}")

        new = best_entries - baseline_entries
        if new:
            print(f"\n    Trades ADDED by dynamic conf ({best_name}):")
            for t in best_trades:
                if t['entry_time'] in new:
                    print(f"      {t['entry_time'].strftime('%Y-%m-%d %H:%M')}  "
                          f"${t['pnl_usd']:>+8.2f}")

print("\n" + "=" * 135)
