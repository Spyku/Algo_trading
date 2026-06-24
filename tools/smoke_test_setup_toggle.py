"""Smoke test for the 2026-06-25 /setup toggle fix in crypto_revolut_ed_v2.py.

Bug: /setup toggles edited the in-memory _setup_state snapshot only and persisted
to regime_config_ed.json solely on /cfg_save -> "didn't stick"; and each handler
sent a confirmation AND a menu = 2 popups.

Fix: _setup_commit() saves to disk + syncs live trading_cfg after each mutation,
and the redundant confirmation is dropped so each toggle sends ONE menu message.

This test redirects the config path to a temp file, stubs the Telegram senders to
counters, drives _setup_handle directly, and asserts: persisted to disk, live
in-memory synced, and exactly one menu / zero stray confirmations.
"""
import os, sys, json, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); os.chdir(HERE)
import crypto_revolut_ed_v2 as t

results = []
def check(name, cond):
    print(('PASS' if cond else 'FAIL'), '-', name); results.append(bool(cond))

# redirect config writes to a temp file (save_trading_config uses module global)
d = tempfile.mkdtemp()
t.REGIME_CONFIG_FILE = os.path.join(d, 'regime_test.json')

# stub senders to count messages; let _setup_send_menu run but mock its sender + load_position
plain_msgs, menu_calls = [], []
t.send_telegram = lambda *a, **k: plain_msgs.append(a[0] if a else '')
t.send_telegram_with_buttons = lambda *a, **k: None
t._setup_send_menu = lambda asset, cfg: menu_calls.append(asset)
t.load_position = lambda asset: {}

def reset_state():
    live = {
        'BTC': {'enabled': True, 'symbol': 'BTC-USD', 'bull': {'min_confidence': 85}, 'bear': {}},
        'ETH': {'enabled': True, 'symbol': 'ETH-USD', 'use_maker_orders': True, 'take_profit_pct': 0,
                'bull': {'min_confidence': 65}, 'bear': {'min_confidence': 70}},
    }
    t._setup_state = {'active': True, 'cfg': json.loads(json.dumps(live))}
    return json.loads(json.dumps(live))

def saved_cfg():
    return json.load(open(t.REGIME_CONFIG_FILE, encoding='utf-8'))

# ---------- 1) enabled toggle (the reported bug) ----------
trading_cfg = reset_state(); plain_msgs.clear(); menu_calls.clear()
t._setup_handle('/cfg_BTC_toggle', trading_cfg)   # BTC True -> False
check('enabled: PERSISTED to disk (BTC True->False)', saved_cfg()['BTC']['enabled'] is False)
check('enabled: live trading_cfg synced', trading_cfg['BTC']['enabled'] is False)
check('enabled: exactly ONE menu message', len(menu_calls) == 1)
check('enabled: NO stray confirmation popup (2-popup fix)', len(plain_msgs) == 0)

# and it STICKS without /cfg_save (the whole point)
trading_cfg = reset_state(); plain_msgs.clear(); menu_calls.clear()
t._setup_handle('/cfg_BTC_toggle', trading_cfg)
check('enabled: sticks WITHOUT /cfg_save', saved_cfg()['BTC']['enabled'] is False)

# ---------- 2) maker toggle ----------
trading_cfg = reset_state(); plain_msgs.clear(); menu_calls.clear()
t._setup_handle('/cfg_ETH_maker', trading_cfg)    # ETH maker True -> False
check('maker: persisted + synced', saved_cfg()['ETH']['use_maker_orders'] is False and trading_cfg['ETH']['use_maker_orders'] is False)
check('maker: one menu, no confirmation', len(menu_calls) == 1 and len(plain_msgs) == 0)

# ---------- 3) take-profit toggle ----------
trading_cfg = reset_state(); plain_msgs.clear(); menu_calls.clear()
t._setup_handle('/cfg_ETH_tp', trading_cfg)       # tp 0 -> 1.0
check('tp: persisted ON', saved_cfg()['ETH']['take_profit_pct'] == 1.0)
check('tp: one menu, no confirmation', len(menu_calls) == 1 and len(plain_msgs) == 0)

# ---------- 4) confidence value setter ----------
trading_cfg = reset_state(); plain_msgs.clear(); menu_calls.clear()
t._setup_handle('/cfg_ETH_bull_confv_80', trading_cfg)
check('conf: persisted (ETH bull 65->80)', saved_cfg()['ETH']['bull']['min_confidence'] == 80)
check('conf: synced + one menu', trading_cfg['ETH']['bull']['min_confidence'] == 80 and len(menu_calls) == 1)

print('\n' + ('ALL PASS (%d/%d)' % (sum(results), len(results)) if all(results)
              else 'FAILURES (%d/%d)' % (sum(results), len(results))))
sys.exit(0 if all(results) else 1)
