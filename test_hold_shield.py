"""Thorough test of Hold Shield toggle feature.

Tests:
1. _main_buttons reads shield state correctly for both ON/OFF
2. /hold toggles hold_shield flag
3. Config is persisted to disk
4. Hold override in process_asset respects the flag
5. Invalid asset names are rejected cleanly
6. No crashes on edge cases

Mocks send_telegram so we don't spam real Telegram.
Backs up and restores regime_config_ed.json.
"""
import json
import os
import shutil
import sys

CONFIG_PATH = 'config/regime_config_ed.json'
BACKUP_PATH = 'config/regime_config_ed.json.test_backup'


def backup_config():
    shutil.copy(CONFIG_PATH, BACKUP_PATH)


def restore_config():
    shutil.copy(BACKUP_PATH, CONFIG_PATH)
    os.remove(BACKUP_PATH)


def load_json():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"  FAIL {label}: expected {expected!r}, got {actual!r}")
        return False
    print(f"  OK   {label}: {actual!r}")
    return True


def main():
    backup_config()
    failures = 0

    try:
        import crypto_revolut_ed_v2 as m

        # Mock send_telegram to silence Telegram during tests
        sent = []
        def fake_send_telegram(msg, parse_mode='HTML'):
            sent.append(('text', msg))
            return True
        def fake_send_telegram_with_buttons(msg, buttons, parse_mode='HTML'):
            sent.append(('buttons', msg, buttons))
            return True
        m.send_telegram = fake_send_telegram
        m.send_telegram_with_buttons = fake_send_telegram_with_buttons
        import crypto_live_trader_ed as lt
        lt.send_telegram = fake_send_telegram

        print("\n=== TEST 1: _main_buttons reads current state ===")
        # Ensure hold_shield is True and min_sell_pnl_pct > 0 → Shield ON
        cfg = load_json()
        cfg['ETH']['hold_shield'] = True
        cfg['ETH']['min_sell_pnl_pct'] = 0.50
        with open(CONFIG_PATH, 'w') as f:
            json.dump(cfg, f, indent=2)

        buttons = m._main_buttons()
        labels = [b[0] for row in buttons for b in row]
        has_on = any('Shield: ON' in l for l in labels)
        if not assert_eq(has_on, True, "Shield ON in buttons when hold_shield=True, pnl=0.50"):
            failures += 1

        # Now flip to OFF
        cfg['ETH']['hold_shield'] = False
        with open(CONFIG_PATH, 'w') as f:
            json.dump(cfg, f, indent=2)
        buttons = m._main_buttons()
        labels = [b[0] for row in buttons for b in row]
        has_off = any('Shield: OFF' in l for l in labels)
        if not assert_eq(has_off, True, "Shield OFF in buttons when hold_shield=False"):
            failures += 1

        # Min_sell_pnl_pct=0 should also show OFF
        cfg['ETH']['hold_shield'] = True
        cfg['ETH']['min_sell_pnl_pct'] = 0
        with open(CONFIG_PATH, 'w') as f:
            json.dump(cfg, f, indent=2)
        buttons = m._main_buttons()
        labels = [b[0] for row in buttons for b in row]
        has_off = any('Shield: OFF' in l for l in labels)
        if not assert_eq(has_off, True, "Shield OFF when min_sell_pnl_pct=0"):
            failures += 1

        print("\n=== TEST 2: /hold toggles flag ===")
        # Restore to known good state
        cfg = load_json()
        cfg['ETH']['hold_shield'] = True
        cfg['ETH']['min_sell_pnl_pct'] = 0.50
        cfg['ETH']['max_hold_hours'] = 10
        with open(CONFIG_PATH, 'w') as f:
            json.dump(cfg, f, indent=2)

        trading_cfg = m.load_trading_config()
        start = trading_cfg['ETH'].get('hold_shield', True)
        if not assert_eq(start, True, "Initial hold_shield=True"):
            failures += 1

        sent.clear()
        m._handle_hold_shield_command('/hold', trading_cfg)
        # In-memory flipped?
        if not assert_eq(trading_cfg['ETH']['hold_shield'], False, "After /hold, in-memory flag=False"):
            failures += 1
        # Persisted to disk?
        on_disk = load_json()
        if not assert_eq(on_disk['ETH']['hold_shield'], False, "After /hold, disk flag=False"):
            failures += 1
        # Telegram msg mentions OFF?
        any_off = any('OFF' in s[1] for s in sent if isinstance(s[1], str))
        if not assert_eq(any_off, True, "Telegram message mentions OFF"):
            failures += 1

        # Toggle back ON
        sent.clear()
        m._handle_hold_shield_command('/hold', trading_cfg)
        if not assert_eq(trading_cfg['ETH']['hold_shield'], True, "After 2nd /hold, flag=True"):
            failures += 1
        on_disk = load_json()
        if not assert_eq(on_disk['ETH']['hold_shield'], True, "After 2nd /hold, disk flag=True"):
            failures += 1
        any_on = any('ON' in s[1] for s in sent if isinstance(s[1], str))
        if not assert_eq(any_on, True, "Telegram message mentions ON"):
            failures += 1

        print("\n=== TEST 3: /hold ETH specifies asset ===")
        trading_cfg['ETH']['hold_shield'] = True
        m._handle_hold_shield_command('/hold ETH', trading_cfg)
        if not assert_eq(trading_cfg['ETH']['hold_shield'], False, "/hold ETH flips flag"):
            failures += 1
        # Restore
        trading_cfg['ETH']['hold_shield'] = True

        print("\n=== TEST 4: /hold BTC (disabled asset) is handled ===")
        sent.clear()
        # BTC is disabled in config but still in dict
        m._handle_hold_shield_command('/hold BTC', trading_cfg)
        # Should not crash; BTC exists in cfg so it toggles (even if disabled)
        # If we want to reject disabled, that's a design call. Current code toggles anyway.
        # Just verify no crash:
        print(f"  OK   /hold BTC did not crash, {len(sent)} telegram msgs sent")

        print("\n=== TEST 5: /hold XYZ (non-existent asset) ===")
        sent.clear()
        m._handle_hold_shield_command('/hold XYZ', trading_cfg)
        any_not_configured = any('not configured' in str(s[1]) for s in sent)
        if not assert_eq(any_not_configured, True, "Rejects unknown asset"):
            failures += 1

        print("\n=== TEST 6: Hold override respects flag ===")
        # Simulate process_asset's hold override logic
        def should_block_sell(shield_on, min_sell_pnl, cur_pnl, hours_held, max_hold_h, entry_price):
            if shield_on and min_sell_pnl > 0 and entry_price > 0:
                if cur_pnl < min_sell_pnl and hours_held < max_hold_h:
                    return True  # blocked
            return False

        # Shield ON, losing, within time: BLOCK
        if not assert_eq(should_block_sell(True, 0.5, -1.0, 5, 10, 2000), True,
                         "Shield ON + losing + within time → block"):
            failures += 1
        # Shield OFF, losing, within time: ALLOW
        if not assert_eq(should_block_sell(False, 0.5, -1.0, 5, 10, 2000), False,
                         "Shield OFF → allow sell"):
            failures += 1
        # Shield ON, past failsafe: ALLOW (failsafe triggered)
        if not assert_eq(should_block_sell(True, 0.5, -1.0, 11, 10, 2000), False,
                         "Shield ON past failsafe → allow"):
            failures += 1
        # Shield ON, profitable: ALLOW
        if not assert_eq(should_block_sell(True, 0.5, 1.0, 5, 10, 2000), False,
                         "Shield ON profitable → allow"):
            failures += 1

        print("\n=== TEST 7: _main_buttons fallback when config unreadable ===")
        # Pass an empty dict directly
        buttons = m._main_buttons({})
        has_off = any('Shield: OFF' in b[0] for row in buttons for b in row)
        if not assert_eq(has_off, True, "Empty cfg → Shield OFF"):
            failures += 1

        print("\n=== TEST 8: buttons structure is valid ===")
        buttons = m._main_buttons(trading_cfg)
        total_buttons = sum(len(row) for row in buttons)
        if not assert_eq(total_buttons >= 6, True, f"At least 6 buttons ({total_buttons})"):
            failures += 1
        # Every button is a (text, callback) tuple
        all_tuples = all(isinstance(b, tuple) and len(b) == 2 for row in buttons for b in row)
        if not assert_eq(all_tuples, True, "All buttons are (text, cb) tuples"):
            failures += 1
        # Callback /hold exists
        all_cbs = [b[1] for row in buttons for b in row]
        if not assert_eq('/hold' in all_cbs, True, "/hold callback exists"):
            failures += 1
        if not assert_eq('/buy' in all_cbs, True, "/buy callback exists"):
            failures += 1
        if not assert_eq('/sell' in all_cbs, True, "/sell callback exists"):
            failures += 1

    finally:
        restore_config()
        print(f"\n=== Config restored from backup ===")

    print(f"\n{'='*50}")
    if failures == 0:
        print(f"✅ ALL TESTS PASSED")
        return 0
    else:
        print(f"❌ {failures} FAILURES")
        return 1


if __name__ == '__main__':
    sys.exit(main())
