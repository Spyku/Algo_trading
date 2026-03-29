"""
Backtest Strategy A (min hold) and B (regime filter) against baseline.
Uses the actual signal_log.csv to replay what would have happened.
"""
import csv
from datetime import datetime, timedelta

# ── Config ──────────────────────────────────────────────────────────────
TRADING_FEE = 0.0011  # 0.11% per trade (fee + slippage)
CONFIGS = {
    'BTC': {'min_confidence': 85, 'horizon': 6, 'position_usd': 12000},
    'ETH': {'min_confidence': 90, 'horizon': 7, 'position_usd': 2000},
}
MIN_HOLD_OFFSET = 1  # Strategy A: hold for (horizon - offset) hours
REGIME_WINDOW = 6     # Strategy B: look at last N hourly signals
REGIME_SELL_THRESHOLD = 4  # Strategy B: block BUY if >= this many SELLs in window


def load_signals(path):
    """Load signal_log.csv, keep only hourly signals (deduplicate)."""
    signals = {}  # {asset: [(datetime, price, action, confidence), ...]}
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

            # Deduplicate: keep first signal per (asset, hour)
            hour_key = ts.replace(minute=0, second=0)
            if signals[asset] and signals[asset][-1][0].replace(minute=0, second=0) == hour_key:
                continue
            signals[asset].append((ts, price, action, conf))
    return signals


def simulate(asset_signals, min_conf, horizon, position_usd, strategy_name,
             use_min_hold=False, use_regime_filter=False):
    """Simulate trading with given strategy."""
    trades = []
    position = None  # (entry_time, entry_price)
    min_hold_hours = horizon - MIN_HOLD_OFFSET

    for i, (ts, price, action, conf) in enumerate(asset_signals):
        if position is None:
            # ── Not invested: look for BUY ──
            if action == 'BUY' and conf >= min_conf:
                # Strategy B: regime filter
                if use_regime_filter:
                    recent = asset_signals[max(0, i - REGIME_WINDOW):i]
                    sell_count = sum(1 for _, _, a, _ in recent if a == 'SELL')
                    if sell_count >= REGIME_SELL_THRESHOLD:
                        continue  # skip this BUY — bearish regime

                position = (ts, price)
        else:
            # ── Invested: look for exit ──
            if action == 'SELL':
                # Strategy A: min hold period
                if use_min_hold:
                    hours_held = (ts - position[0]).total_seconds() / 3600
                    if hours_held < min_hold_hours:
                        continue  # too early to exit

                entry_price = position[1]
                gross_pct = (price - entry_price) / entry_price * 100
                net_pct = gross_pct - (TRADING_FEE * 2 * 100)  # fee on buy + sell
                pnl_usd = position_usd * net_pct / 100
                hold_hours = (ts - position[0]).total_seconds() / 3600

                trades.append({
                    'entry_time': position[0],
                    'exit_time': ts,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'gross_pct': gross_pct,
                    'net_pct': net_pct,
                    'pnl_usd': pnl_usd,
                    'hold_hours': hold_hours,
                })
                position = None

    # If still invested at end, mark-to-market with last signal price
    if position:
        last_ts, last_price, _, _ = asset_signals[-1]
        gross_pct = (last_price - position[1]) / position[1] * 100
        net_pct = gross_pct - (TRADING_FEE * 2 * 100)
        pnl_usd = position_usd * net_pct / 100
        hold_hours = (last_ts - position[0]).total_seconds() / 3600
        trades.append({
            'entry_time': position[0],
            'exit_time': last_ts,
            'entry_price': position[1],
            'exit_price': last_price,
            'gross_pct': gross_pct,
            'net_pct': net_pct,
            'pnl_usd': pnl_usd,
            'hold_hours': hold_hours,
            'open': True,
        })

    return trades


def print_results(asset, strategy_name, trades, position_usd):
    """Print formatted results for a strategy."""
    if not trades:
        print(f"  {strategy_name:20s} | No trades")
        return

    n = len(trades)
    total_pnl = sum(t['pnl_usd'] for t in trades)
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] < 0]
    wr = len(wins) / n * 100 if n else 0
    avg_win = sum(t['pnl_usd'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl_usd'] for t in losses) / len(losses) if losses else 0
    avg_hold = sum(t['hold_hours'] for t in trades) / n
    best = max(trades, key=lambda t: t['pnl_usd'])
    worst = min(trades, key=lambda t: t['pnl_usd'])
    total_pct = sum(t['net_pct'] for t in trades)

    open_marker = ''
    if trades and trades[-1].get('open'):
        open_marker = ' *'

    print(f"  {strategy_name:20s} | {n:2d} trades | PnL ${total_pnl:+8.2f} ({total_pct:+.2f}%) | "
          f"WR {wr:4.0f}% | avg win ${avg_win:+.0f} / loss ${avg_loss:+.0f} | "
          f"hold {avg_hold:.1f}h | best ${best['pnl_usd']:+.0f} worst ${worst['pnl_usd']:+.0f}{open_marker}")


def print_trades(strategy_name, trades):
    """Print individual trades."""
    print(f"\n  {strategy_name} trades:")
    print(f"  {'Entry':>19s}  {'Exit':>19s}  {'Hold':>5s}  {'Entry$':>10s}  {'Exit$':>10s}  {'Net%':>7s}  {'PnL$':>9s}")
    print(f"  {'-'*19}  {'-'*19}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*9}")
    for t in trades:
        entry = t['entry_time'].strftime('%m-%d %H:%M')
        exit_ = t['exit_time'].strftime('%m-%d %H:%M')
        hold = f"{t['hold_hours']:.0f}h"
        marker = ' *' if t.get('open') else ''
        print(f"  {entry:>19s}  {exit_:>19s}  {hold:>5s}  {t['entry_price']:>10.2f}  "
              f"{t['exit_price']:>10.2f}  {t['net_pct']:>+6.2f}%  ${t['pnl_usd']:>+8.2f}{marker}")


# ── Main ────────────────────────────────────────────────────────────────
import os
SIGNAL_LOG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'signal_log.csv')
signals = load_signals(SIGNAL_LOG)

print("=" * 120)
print("  STRATEGY BACKTEST: A (min hold) + B (regime filter)")
print(f"  Signal log: {sum(len(v) for v in signals.values())} signals, "
      f"{min(s[0] for sigs in signals.values() for s in sigs).strftime('%Y-%m-%d')} to "
      f"{max(s[0] for sigs in signals.values() for s in sigs).strftime('%Y-%m-%d')}")
print(f"  Fee: {TRADING_FEE*100:.2f}% per trade | A: min hold = horizon-{MIN_HOLD_OFFSET}h | "
      f"B: block BUY if {REGIME_SELL_THRESHOLD}+ of last {REGIME_WINDOW} signals = SELL")
print("=" * 120)

for asset in ['BTC', 'ETH']:
    if asset not in signals:
        continue
    cfg = CONFIGS[asset]
    sigs = signals[asset]
    print(f"\n  {asset} (horizon={cfg['horizon']}h, min_conf={cfg['min_confidence']}%, position=${cfg['position_usd']:,})")
    print(f"  {'-'*110}")

    baseline = simulate(sigs, cfg['min_confidence'], cfg['horizon'], cfg['position_usd'],
                        'Baseline', use_min_hold=False, use_regime_filter=False)
    strat_a = simulate(sigs, cfg['min_confidence'], cfg['horizon'], cfg['position_usd'],
                       'A: Min Hold', use_min_hold=True, use_regime_filter=False)
    strat_b = simulate(sigs, cfg['min_confidence'], cfg['horizon'], cfg['position_usd'],
                       'B: Regime Filter', use_min_hold=False, use_regime_filter=True)
    strat_ab = simulate(sigs, cfg['min_confidence'], cfg['horizon'], cfg['position_usd'],
                        'A+B: Combined', use_min_hold=True, use_regime_filter=True)

    print_results(asset, 'Baseline', baseline, cfg['position_usd'])
    print_results(asset, 'A: Min Hold', strat_a, cfg['position_usd'])
    print_results(asset, 'B: Regime Filter', strat_b, cfg['position_usd'])
    print_results(asset, 'A+B: Combined', strat_ab, cfg['position_usd'])

    # Print detailed trades for each
    for name, trades in [('Baseline', baseline), ('A: Min Hold', strat_a),
                          ('B: Regime Filter', strat_b), ('A+B: Combined', strat_ab)]:
        if trades:
            print_trades(name, trades)

print(f"\n  * = still open at end of period (mark-to-market)")
print("=" * 120)
