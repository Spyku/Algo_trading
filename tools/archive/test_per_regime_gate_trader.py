"""Unit test for the per-regime rally-cooldown lookup in the trader.
Verifies the _rally_cfg_for_regime helper's priority: regime-block → asset-level → None.
"""
import sys
import types

sys.path.insert(0, '.')
for stub in ('ccxt', 'telegram'):
    if stub not in sys.modules:
        sys.modules[stub] = types.ModuleType(stub)

from crypto_revolut_ed_v2 import _rally_cfg_for_regime


def check(name, got, expected):
    ok = got == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         got:      {got}")
        print(f"         expected: {expected}")
    return ok


fails = 0
print("=" * 60)
print("  Per-regime rally-cooldown lookup")
print("=" * 60)

# Case 1: new config, per-regime gates explicit
BULL_RC = {'enabled': True, 'h_short': 14, 'h_long': 36, 't_short_pct': 4.5, 't_long_pct': 4.0, 'cd_hours': 10}
BEAR_RC = {'enabled': True, 'h_short': 30, 'h_long': 36, 't_short_pct': 7.5, 't_long_pct': 4.5, 'cd_hours': 8}
cfg1 = {'bull': {'rally_cooldown': BULL_RC}, 'bear': {'rally_cooldown': BEAR_RC}}
if not check('per-regime: bull', _rally_cfg_for_regime(cfg1, 'bull'), BULL_RC): fails += 1
if not check('per-regime: bear', _rally_cfg_for_regime(cfg1, 'bear'), BEAR_RC): fails += 1

# Case 2: legacy asset-level, no per-regime → fallback applies to both
LEGACY_RC = {'enabled': True, 'h_short': 10, 'h_long': 30, 't_short_pct': 2.5, 't_long_pct': 6.0, 'cd_hours': 20}
cfg2 = {'rally_cooldown': LEGACY_RC, 'bull': {}, 'bear': {}}
if not check('legacy fallback: bull', _rally_cfg_for_regime(cfg2, 'bull'), LEGACY_RC): fails += 1
if not check('legacy fallback: bear', _rally_cfg_for_regime(cfg2, 'bear'), LEGACY_RC): fails += 1

# Case 3: mixed — bull has override, bear falls back to asset-level
cfg3 = {'rally_cooldown': LEGACY_RC, 'bull': {'rally_cooldown': BULL_RC}, 'bear': {}}
if not check('mixed: bull uses override', _rally_cfg_for_regime(cfg3, 'bull'), BULL_RC): fails += 1
if not check('mixed: bear uses legacy', _rally_cfg_for_regime(cfg3, 'bear'), LEGACY_RC): fails += 1

# Case 4: no rally_cooldown anywhere
cfg4 = {'bull': {}, 'bear': {}}
if not check('no rally_cd: bull', _rally_cfg_for_regime(cfg4, 'bull'), None): fails += 1
if not check('no rally_cd: bear', _rally_cfg_for_regime(cfg4, 'bear'), None): fails += 1

# Case 5: regime block is not a dict (defensive)
cfg5 = {'bull': 'oops', 'rally_cooldown': LEGACY_RC}
if not check('bull non-dict → fallback', _rally_cfg_for_regime(cfg5, 'bull'), LEGACY_RC): fails += 1

# Case 6: disabled gate in regime, disabled gate in asset
DISABLED_RC = {'enabled': False, 'h_short': 10, 'h_long': 30}
cfg6 = {'bull': {'rally_cooldown': DISABLED_RC}}
# Current helper returns the dict unchanged (trader checks .enabled downstream)
# So a disabled per-regime dict IS returned; caller must check .enabled.
if not check('disabled per-regime returns dict (caller checks enabled)',
             _rally_cfg_for_regime(cfg6, 'bull'), DISABLED_RC): fails += 1

print("=" * 60)
print(f"  {'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
print("=" * 60)
sys.exit(0 if fails == 0 else 1)
