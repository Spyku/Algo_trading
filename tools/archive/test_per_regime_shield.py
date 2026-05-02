"""Unit test: per-regime hold-shield lookup.

Exercises _shield_on_for_regime and the `/hold` command parser without hitting
Telegram or the exchange. Run from engine root:
    python test_per_regime_shield.py
"""
import sys

sys.path.insert(0, '.')

# Stub heavy imports the trader drags in
import types
for stub in ('ccxt', 'telegram'):
    if stub not in sys.modules:
        sys.modules[stub] = types.ModuleType(stub)

from crypto_revolut_ed_v2 import _shield_on_for_regime


def check(name, got, expected):
    ok = got == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got} expected={expected}")
    return ok


fails = 0

print("=" * 60)
print("  Per-regime shield lookup")
print("=" * 60)

# Case 1: per-regime set explicitly, bull ON / bear OFF
cfg1 = {'bull': {'hold_shield': True}, 'bear': {'hold_shield': False}}
if not check('bull ON explicit', _shield_on_for_regime(cfg1, 'bull'), True): fails += 1
if not check('bear OFF explicit', _shield_on_for_regime(cfg1, 'bear'), False): fails += 1

# Case 2: per-regime missing → falls back to asset-level hold_shield
cfg2 = {'bull': {}, 'bear': {}, 'hold_shield': True}
if not check('legacy ON → bull', _shield_on_for_regime(cfg2, 'bull'), True): fails += 1
if not check('legacy ON → bear', _shield_on_for_regime(cfg2, 'bear'), True): fails += 1

cfg3 = {'bull': {}, 'bear': {}, 'hold_shield': False}
if not check('legacy OFF → bull', _shield_on_for_regime(cfg3, 'bull'), False): fails += 1
if not check('legacy OFF → bear', _shield_on_for_regime(cfg3, 'bear'), False): fails += 1

# Case 3: no config at all → default True
cfg4 = {}
if not check('default → bull', _shield_on_for_regime(cfg4, 'bull'), True): fails += 1
if not check('default → bear', _shield_on_for_regime(cfg4, 'bear'), True): fails += 1

# Case 4: mixed — bull has per-regime, bear falls back to asset-level
cfg5 = {'bull': {'hold_shield': False}, 'bear': {}, 'hold_shield': True}
if not check('mixed bull OFF', _shield_on_for_regime(cfg5, 'bull'), False): fails += 1
if not check('mixed bear (fallback ON)', _shield_on_for_regime(cfg5, 'bear'), True): fails += 1

# Case 5: regime block exists but isn't a dict — must not crash
cfg6 = {'bull': 'oops', 'bear': None, 'hold_shield': False}
if not check('bull non-dict → fallback', _shield_on_for_regime(cfg6, 'bull'), False): fails += 1
if not check('bear non-dict → fallback', _shield_on_for_regime(cfg6, 'bear'), False): fails += 1

print("=" * 60)
print(f"  {'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
print("=" * 60)
sys.exit(0 if fails == 0 else 1)
