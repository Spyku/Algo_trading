"""Dump per-bar primary signals for the last 48h (research throwaway)."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from crypto_trading_system_ed import generate_signals

sigs = generate_signals(
    asset_name='ETH',
    model_names=['RF', 'LGBM'],
    window_size=141,
    replay_hours=48,
    feature_override=None,
    horizon=6,
    gamma=0.9958,
)
print()
print('=' * 70)
print(f'{len(sigs)} bars from last 48h:')
print('=' * 70)
for s in sigs:
    print(f"{s['datetime']}  {s['signal']:4s} ({s['confidence']:3.0f}%)  ${s['close']:,.2f}")
