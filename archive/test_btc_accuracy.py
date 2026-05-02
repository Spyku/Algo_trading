"""Check actual accuracy of each BTC horizon over Apr 5-6."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from crypto_trading_system_ed import generate_signals
import csv

configs = []
with open(os.path.join(os.path.dirname(__file__), 'models', 'crypto_ed_production.csv')) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['coin'] == 'BTC':
            h = int(row['horizon'])
            if h in [5, 6, 7, 8, 10, 12, 14]:
                configs.append({
                    'horizon': h,
                    'models': row['models'].split('+'),
                    'window': int(row['best_window']),
                    'features': row['optimal_features'].split(','),
                    'gamma': float(row['gamma']),
                    'accuracy': row['accuracy'],
                    'return_pct': row['return_pct'],
                })

configs.sort(key=lambda x: x['horizon'])

print(f"{'Horizon':>7} | {'Backtest':>8} | {'Actual':>6} | {'BUY ok':>6} | {'SELL ok':>7} | {'BUY wrong':>9} | {'SELL wrong':>10} | {'HOLD':>4}")
print("-" * 85)

for cfg in configs:
    h = cfg['horizon']
    sigs = generate_signals('BTC', cfg['models'], cfg['window'], replay_hours=48,
                           feature_override=cfg['features'], horizon=h, gamma=cfg['gamma'])

    buy_right = buy_wrong = sell_right = sell_wrong = hold_count = no_actual = 0
    for s in sigs:
        if s['actual'] is None:
            no_actual += 1
            continue
        if s['signal'] == 'HOLD':
            hold_count += 1
            continue
        if s['signal'] == 'BUY':
            if s['actual'] == 'UP':
                buy_right += 1
            else:
                buy_wrong += 1
        elif s['signal'] == 'SELL':
            if s['actual'] == 'DOWN':
                sell_right += 1
            else:
                sell_wrong += 1

    total_decisive = buy_right + buy_wrong + sell_right + sell_wrong
    correct = buy_right + sell_right
    acc = (correct / total_decisive * 100) if total_decisive > 0 else 0

    print(f"{h:>5}h | {cfg['accuracy']:>7}% | {acc:>5.1f}% | {buy_right:>6} | {sell_right:>7} | {buy_wrong:>9} | {sell_wrong:>10} | {hold_count:>4}")
