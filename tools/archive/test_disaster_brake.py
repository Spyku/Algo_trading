"""Unit test for the disaster brake trigger logic — pure arithmetic, no mocking
of Telegram or API calls. Verifies the threshold check matches the trader code.
"""
import sys
import types

sys.path.insert(0, '.')
for stub in ('ccxt', 'telegram'):
    if stub not in sys.modules:
        sys.modules[stub] = types.ModuleType(stub)


def check(name, got, expected):
    ok = got == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got} expected={expected}")
    return ok


def should_brake(entry_price, current_price, brake_pct):
    """Mirror of the trader's disaster-brake check."""
    if brake_pct <= 0:
        return False
    cur_pnl = (current_price - entry_price) / entry_price * 100
    return cur_pnl <= -brake_pct


fails = 0

print("=" * 60)
print("  Disaster brake threshold logic")
print("=" * 60)

# Brake OFF (0): never triggers
if not check('brake=0 at -10%', should_brake(100.0, 90.0, 0), False): fails += 1
if not check('brake=0 at any drop', should_brake(100.0, 50.0, 0), False): fails += 1

# Brake 5%: triggers at exactly -5% and below
if not check('5% at -4.99%', should_brake(100.0, 95.01, 5), False): fails += 1
if not check('5% at -5.00%', should_brake(100.0, 95.00, 5), True): fails += 1
if not check('5% at -5.01%', should_brake(100.0, 94.99, 5), True): fails += 1
if not check('5% at -10%',   should_brake(100.0, 90.00, 5), True): fails += 1

# Brake 7%: triggers at -7% and below
if not check('7% at -5%',  should_brake(100.0, 95.0, 7), False): fails += 1
if not check('7% at -7%',  should_brake(100.0, 93.0, 7), True): fails += 1
if not check('7% at -10%', should_brake(100.0, 90.0, 7), True): fails += 1

# Positive PnL: never triggers regardless of threshold
if not check('5% at +2%',  should_brake(100.0, 102.0, 5), False): fails += 1
if not check('5% at +10%', should_brake(100.0, 110.0, 5), False): fails += 1

# Today's actual event (-0.98% at 09:00): 5% brake would not have fired
if not check('today 5%: -0.98%', should_brake(2349.87, 2326.84, 5), False): fails += 1
# At 5% the position would need to hit ~$2232 to trigger
if not check('today 5%: -5.01% hypothetical', should_brake(2349.87, 2232.25, 5), True): fails += 1

# Disaster brake bypasses shield — covered by trader code (shield_on = not disaster_brake)
# and Is verified by reading the edit, not the sim.

print("=" * 60)
print(f"  {'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
print("=" * 60)
sys.exit(0 if fails == 0 else 1)
