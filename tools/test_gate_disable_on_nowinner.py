"""Regression test for the 2026-06-23 (a) fix: _sweep_rally_cooldown's no-winner
branch must DISABLE the gate (single-form), not leave a stale double in the config.

Reproduces the 06-22 incident in miniature: seed a config with a stale enabled
DOUBLE gate, feed a flat (no-rally) signal stream so no single gate can beat the
no-gate baseline -> NO STRICT WINNER -> the branch under test must overwrite the
stale double with enabled=False + single-form (h_long==h_short, t_long=9999).
"""
import os, sys, json, tempfile, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['FAYE_LIBRARY_MODE'] = '1'
import crypto_trading_system_faye as faye

tmpdir = tempfile.mkdtemp()
cfgpath = os.path.join(tmpdir, 'regime_test.json')
seed = {'ETH': {
    'enabled': True, 'symbol': 'ETH-USD',
    'bull': {'horizon': 4, 'min_confidence': 65, 'hold_shield': False},
    'bear': {'horizon': 4, 'min_confidence': 70, 'hold_shield': False,
             'rally_cooldown': {'enabled': True, 'h_short': 8, 'h_long': 14,
                                't_short_pct': 5.0, 't_long_pct': 3.0, 'cd_hours': 48}},
    'min_sell_pnl_pct': 0, 'max_hold_hours': 10}}
json.dump(seed, open(cfgpath, 'w'), indent=2)
faye.REGIME_CONFIG_PATH = cfgpath   # redirect the write target (test-only)

# Flat oscillating price: max bar move 0.05%, net ~0 -> no rally ever clears t_s>=2%.
base = dt.datetime(2026, 1, 1)
sigs, price = [], 1000.0
for i in range(740):
    price *= 1.0 + (0.0005 if i % 2 == 0 else -0.0005)
    sigs.append({'datetime': base + dt.timedelta(hours=i), 'close': price,
                 'signal': ('BUY' if i % 5 == 0 else ('SELL' if i % 5 == 2 else 'HOLD')),
                 'confidence': 80.0, 'regime': 'bear'})

print("seed bear gate (stale DOUBLE):", json.dumps(seed['ETH']['bear']['rally_cooldown']))
res = faye._sweep_rally_cooldown('ETH', sigs, seed['ETH'], replay_h=720,
                                 rank='recent', write_config=True, regime_filter='bear')
rc = json.load(open(cfgpath))['ETH']['bear']['rally_cooldown']
print("after  bear gate:", json.dumps(rc))
print("return:", res)

assert rc['enabled'] is False, f"FAIL: gate still enabled: {rc}"
assert rc['h_long'] == rc['h_short'], f"FAIL: not single-form (h_long!=h_short): {rc}"
assert float(rc['t_long_pct']) == 9999.0, f"FAIL: long leg not neutralised: {rc}"
assert res is not None and res.get('enabled') is False, f"FAIL: return contract: {res}"
print("\nPASS: stale enabled DOUBLE -> NO STRICT WINNER -> wrote DISABLED single-form gate")
