"""Quick test: run all BTC horizons against Apr 5-6 data to compare signals."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from crypto_trading_system_ed import generate_signals, load_data
import csv

# Load BTC configs from production CSV
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

print("=" * 90)
print("BTC HORIZON COMPARISON - April 5-6, 2026")
print("=" * 90)

for cfg in configs:
    h = cfg['horizon']
    print(f"\n{'='*90}")
    print(f"HORIZON: {h}h | Models: {'+'.join(cfg['models'])} | Window: {cfg['window']} | "
          f"Backtest acc: {cfg['accuracy']}% | Return: {cfg['return_pct']}%")
    print("-" * 90)

    sigs = generate_signals(
        'BTC',
        cfg['models'],
        cfg['window'],
        replay_hours=48,  # ~2 days to cover Apr 5-6
        feature_override=cfg['features'],
        horizon=h,
        gamma=cfg['gamma'],
    )

    # Filter for Apr 5-6 only
    for s in sigs:
        if '2026-04-05' in s['datetime'] or '2026-04-06' in s['datetime']:
            actual_str = f" | actual={s['actual']}" if s['actual'] else ""
            print(f"  {s['datetime']} | ${s['close']:>10,.2f} | {s['signal']:>4} {s['confidence']:>5.0f}% "
                  f"| votes={s['buy_votes']}/{s['total_votes']}{actual_str}")

print(f"\n{'='*90}")
print("DONE")
