"""Quick market sell of BTC position"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from crypto_revolut_ed_v2 import place_market_sell, get_balances, load_position, save_position

# Check current balance
print("Checking BTC balance...")
balances = get_balances()
if balances:
    btc_bal = balances.get('BTC', {})
    print(f"  BTC available: {btc_bal.get('available', 0)}")
    print(f"  BTC total: {btc_bal.get('total', 0)}")
    btc_amount = float(btc_bal.get('available', 0))
else:
    print("  Could not fetch balances, using position file amount")
    btc_amount = 0.07851438

if btc_amount <= 0:
    print("No BTC to sell!")
    sys.exit(1)

print(f"\nSelling {btc_amount} BTC via MARKET order (taker)...")
status, data = place_market_sell('BTC-USD', btc_amount)
print(f"  Status: {status}")
print(f"  Response: {data}")

if status == 200:
    print("\nSELL executed. Updating position file...")
    pos = load_position('BTC')
    pos['state'] = 'cash'
    pos['trades'].append({
        'action': 'SELL',
        'price': 0,  # will be updated from exchange
        'time': __import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M'),
        'pnl_pct': 0,
        'pnl_usd': 0,
        'auto': False,
        'note': 'manual market sell - disabling BTC'
    })
    save_position('BTC', pos)
    print("Position updated to CASH")
else:
    print(f"\nSELL FAILED! Status {status}")
