"""Verify the new per-regime shield UX: button labels + /hold command parsing.
Does not hit Telegram. Run from engine root.
"""
import sys
import types

sys.path.insert(0, '.')
for stub in ('ccxt', 'telegram'):
    if stub not in sys.modules:
        sys.modules[stub] = types.ModuleType(stub)

from crypto_revolut_ed_v2 import _main_buttons, _shield_on_for_regime


def check(name, got, expected):
    ok = got == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         got:      {got}")
        print(f"         expected: {expected}")
    return ok


def find_btn(rows, label_prefix):
    """Find button row containing a label starting with prefix."""
    for row in rows:
        for lbl, cmd in row:
            if lbl.startswith(label_prefix):
                return lbl, cmd
    return None, None


fails = 0
print("=" * 60)
print("  Per-regime shield UX — button layout")
print("=" * 60)

# Case 1: bull ON, bear OFF (current ETH config)
cfg = {
    'ETH': {
        'enabled': True,
        'min_sell_pnl_pct': 0.6,
        'bull': {'hold_shield': True},
        'bear': {'hold_shield': False},
    }
}
rows = _main_buttons(cfg)
bull_lbl, bull_cmd = find_btn(rows, '🛡 Bull:')
bear_lbl, bear_cmd = find_btn(rows, '🛡 Bear:')

if not check('bull label (ON)', bull_lbl, '🛡 Bull: ON'): fails += 1
if not check('bull cmd', bull_cmd, '/hold bull'): fails += 1
if not check('bear label (OFF)', bear_lbl, '🛡 Bear: OFF'): fails += 1
if not check('bear cmd', bear_cmd, '/hold bear'): fails += 1

# Case 2: both OFF
cfg['ETH']['bull']['hold_shield'] = False
cfg['ETH']['bear']['hold_shield'] = False
rows = _main_buttons(cfg)
bull_lbl, _ = find_btn(rows, '🛡 Bull:')
bear_lbl, _ = find_btn(rows, '🛡 Bear:')
if not check('both OFF - bull', bull_lbl, '🛡 Bull: OFF'): fails += 1
if not check('both OFF - bear', bear_lbl, '🛡 Bear: OFF'): fails += 1

# Case 3: no min_sell_pnl_pct → shield effectively OFF even if flag true
cfg['ETH']['bull']['hold_shield'] = True
cfg['ETH']['bear']['hold_shield'] = True
cfg['ETH']['min_sell_pnl_pct'] = 0
rows = _main_buttons(cfg)
bull_lbl, _ = find_btn(rows, '🛡 Bull:')
bear_lbl, _ = find_btn(rows, '🛡 Bear:')
if not check('no threshold - bull shows OFF', bull_lbl, '🛡 Bull: OFF'): fails += 1
if not check('no threshold - bear shows OFF', bear_lbl, '🛡 Bear: OFF'): fails += 1

# Case 4: row layout — 4 rows, last row is Setup alone
cfg['ETH']['min_sell_pnl_pct'] = 0.6
rows = _main_buttons(cfg)
if not check('row count', len(rows), 4): fails += 1
if not check('last row is Setup alone', [b for b in rows[-1]], [('⚙️ Setup', '/setup')]): fails += 1

# Case 5: Row with Bull + Bear has 2 buttons side by side
shield_row = rows[2]
if not check('shield row has 2 buttons', len(shield_row), 2): fails += 1
if not check('shield row bull first', shield_row[0][0].startswith('🛡 Bull'), True): fails += 1
if not check('shield row bear second', shield_row[1][0].startswith('🛡 Bear'), True): fails += 1

print("=" * 60)
print(f"  {'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
print("=" * 60)
sys.exit(0 if fails == 0 else 1)
