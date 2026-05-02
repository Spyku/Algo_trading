"""Test M-29: partial-fill recalc must compute next_size = min(usd_avail, target - already_filled).

Standalone simulation of the recalc logic. Does NOT touch the live trader.
Verifies: (1) overspend bug is fixed for the today's-bug scenario, (2) edge cases.
"""
import math

MIN_TRADE_USD = 299.9


def recalc_buy_size(original_size, total_filled_usd, baseline_usd, usd_avail, current_size):
    """Mirrors the M-29 BUY-side recalc logic from _execute_maker_order.
    Returns: (action, new_size_or_residual, message)
      action ∈ {'continue', 'market_residual', 'stop_target_reached'}
    """
    spent_by_wallet = max(0.0, baseline_usd - usd_avail) if baseline_usd is not None else total_filled_usd
    spent_by_orders = total_filled_usd
    spent_so_far = max(spent_by_wallet, spent_by_orders)
    remaining_target = max(0.0, original_size - spent_so_far)

    if usd_avail < MIN_TRADE_USD:
        if remaining_target >= MIN_TRADE_USD:
            return ('market_residual', usd_avail, f'cash<min, target unmet, market for {usd_avail}')
        return ('stop_target_reached', spent_so_far, f'cash<min, target met (spent {spent_so_far})')

    if remaining_target < MIN_TRADE_USD:
        return ('stop_target_reached', spent_so_far, f'remaining<min, target met (spent {spent_so_far})')

    next_size = min(math.floor(usd_avail * 100) / 100 - 0.01, remaining_target)
    if abs(next_size - current_size) > 0.01:
        return ('continue', next_size, f'recalc: target={original_size} filled={spent_so_far} remaining={remaining_target} cash={usd_avail} → {next_size}')
    return ('continue', current_size, 'no change')


def test_today_bug_scenario():
    """Today's actual reproduction: target $12k, wallet $12,610.55, partial $2,026.12."""
    print("\n=== TEST 1: Today's manual /buy bug scenario ===")
    original_size = 12000.00
    baseline_usd = 12610.55  # actual wallet at start
    # After phase 1 partial fill of $2,026.12:
    total_filled_usd = 2026.12
    usd_avail = baseline_usd - total_filled_usd  # = 10,584.43
    current_size = 12000.00  # was at original

    action, new_size, msg = recalc_buy_size(original_size, total_filled_usd, baseline_usd, usd_avail, current_size)
    print(f"  baseline=${baseline_usd:,.2f}  filled_so_far=${total_filled_usd:,.2f}  cash_avail=${usd_avail:,.2f}")
    print(f"  → action={action}  new_size=${new_size:,.2f}")
    print(f"  → {msg}")

    expected_remaining = 12000.00 - 2026.12  # = 9,973.88
    assert action == 'continue', f"Expected 'continue', got {action}"
    assert abs(new_size - expected_remaining) < 0.05, f"Expected ~${expected_remaining:.2f}, got ${new_size:.2f}"
    # OLD BUG would have returned new_size = $10,584.42 (full available cash)
    # which leads to over-spend = $10,584.42 - $9,973.88 = $610.54
    print(f"  ✓ PASS: next leg correctly capped at remaining target ${new_size:,.2f}")
    print(f"  (old bug would have set next_size = ${usd_avail-0.01:,.2f}, over-spending by ${(usd_avail - 0.01) - expected_remaining:.2f})")


def test_wallet_equals_target():
    """Wallet exactly matches target — old code happened to be correct here."""
    print("\n=== TEST 2: Wallet == target (today's earlier scenario I incorrectly assumed) ===")
    original_size = 12000.00
    baseline_usd = 11999.99
    total_filled_usd = 1416.57
    usd_avail = baseline_usd - total_filled_usd  # = 10,583.42
    action, new_size, msg = recalc_buy_size(original_size, total_filled_usd, baseline_usd, usd_avail, 12000.00)
    print(f"  baseline=${baseline_usd:,.2f}  filled=${total_filled_usd:,.2f}  cash=${usd_avail:,.2f}")
    print(f"  → new_size=${new_size:,.2f}")
    expected_remaining = 12000.00 - 1416.57  # = 10,583.43
    # next_size = min(10,583.41, 10,583.43) = 10,583.41 (cash binds, slightly under target by $0.02)
    assert new_size <= expected_remaining + 0.05
    print(f"  ✓ PASS: capped at min(cash-0.01, remaining_target)")


def test_wallet_much_more_than_target():
    """Wallet $50k, /buy $12k. Catastrophic if old bug applied."""
    print("\n=== TEST 3: Wallet >> target ($50k cash, $12k target) ===")
    original_size = 12000.00
    baseline_usd = 50000.00
    # Scenario: phase 1 partial of $2k, then check what next leg should be
    total_filled_usd = 2000.00
    usd_avail = baseline_usd - total_filled_usd  # = 48,000
    action, new_size, msg = recalc_buy_size(original_size, total_filled_usd, baseline_usd, usd_avail, 12000.00)
    print(f"  baseline=${baseline_usd:,.2f}  filled=${total_filled_usd:,.2f}  cash=${usd_avail:,.2f}")
    print(f"  → new_size=${new_size:,.2f}")
    assert new_size <= 10000.05, f"Should cap at remaining=$10k, got ${new_size:.2f}"
    # OLD BUG: condition `usd_avail($48k) < size($12k)` is False → size stays $12k → over-spend $14k total
    print(f"  ✓ PASS: capped at remaining target $10k (old bug would have left size=$12k → total spend $14k = $2k over)")


def test_wallet_less_than_target():
    """Wallet $5k, /buy $12k target. Should spend wallet without overshooting."""
    print("\n=== TEST 4: Wallet < target ($5k cash, $12k target) ===")
    original_size = 12000.00
    baseline_usd = 5000.00
    # Phase 1 partial of $1k:
    total_filled_usd = 1000.00
    usd_avail = baseline_usd - total_filled_usd  # = 4,000
    action, new_size, msg = recalc_buy_size(original_size, total_filled_usd, baseline_usd, usd_avail, 5000.00)
    print(f"  baseline=${baseline_usd:,.2f}  filled=${total_filled_usd:,.2f}  cash=${usd_avail:,.2f}")
    print(f"  → new_size=${new_size:,.2f}")
    # Remaining_target = $11k. Next_size = min(cash-0.01=$3,999.99, $11k) = $3,999.99 ✓
    assert abs(new_size - 3999.99) < 0.05
    print(f"  ✓ PASS: spend remaining wallet, capped by cash (not target)")


def test_wallet_below_min_after_partial():
    """Cash drops below MIN_TRADE_USD after partial. Should market-fallback if target unmet."""
    print("\n=== TEST 5: Cash dropped below MIN_TRADE_USD ($250 left, target unmet) ===")
    original_size = 12000.00
    baseline_usd = 12000.00
    total_filled_usd = 11750.00
    usd_avail = baseline_usd - total_filled_usd  # = 250
    action, new_size, msg = recalc_buy_size(original_size, total_filled_usd, baseline_usd, usd_avail, 250.00)
    print(f"  cash=${usd_avail:,.2f}  remaining_target=${original_size - total_filled_usd:,.2f}")
    print(f"  → action={action}  residual=${new_size:,.2f}")
    # remaining = $250. Both conditions: usd_avail<min AND remaining<min → stop, target met
    assert action == 'stop_target_reached'
    print(f"  ✓ PASS: stop cleanly (residual below min trade size)")


def test_target_essentially_met():
    """Filled $11,800 of $12k target. Remaining $200 < MIN_TRADE_USD."""
    print("\n=== TEST 6: Target essentially met (remaining < MIN_TRADE_USD) ===")
    original_size = 12000.00
    baseline_usd = 12500.00  # had extra cash
    total_filled_usd = 11800.00
    usd_avail = baseline_usd - total_filled_usd  # = 700 (above min, but remaining target < min)
    action, new_size, msg = recalc_buy_size(original_size, total_filled_usd, baseline_usd, usd_avail, 12000.00)
    print(f"  cash=${usd_avail:,.2f}  filled=${total_filled_usd:,.2f}  remaining_target=${original_size - total_filled_usd:,.2f}")
    print(f"  → action={action}")
    assert action == 'stop_target_reached'
    print(f"  ✓ PASS: stop because remaining target < MIN_TRADE_USD")


def test_no_baseline():
    """get_balances() failed at function entry → baseline=None. Falls back to order accumulator."""
    print("\n=== TEST 7: baseline read failed (None) ===")
    original_size = 12000.00
    baseline_usd = None
    total_filled_usd = 2000.00  # accumulated from order status
    usd_avail = 10000.00  # current wallet read
    action, new_size, msg = recalc_buy_size(original_size, total_filled_usd, baseline_usd, usd_avail, 12000.00)
    print(f"  baseline=None  filled_by_orders=${total_filled_usd:,.2f}  cash=${usd_avail:,.2f}")
    print(f"  → new_size=${new_size:,.2f}")
    # spent_so_far = max(total_filled_usd, total_filled_usd) = $2000
    # remaining = $10,000
    # next_size = min($9,999.99, $10,000) = $9,999.99
    assert abs(new_size - 9999.99) < 0.05
    print(f"  ✓ PASS: uses order accumulator when baseline unavailable")


def test_cross_check_disagreement():
    """Order-status says spent=$2000, wallet says spent=$3000 (other-asset trade contaminated wallet).
    Should use the LARGER (more conservative) → smaller remaining → no over-spend."""
    print("\n=== TEST 8: Cross-check disagreement (other-asset USD activity) ===")
    original_size = 12000.00
    baseline_usd = 15000.00
    # Wallet says we spent $3,000 (but $1,000 was a different asset's BUY firing concurrently)
    usd_avail = 12000.00  # = 15000 - 3000
    total_filled_usd = 2000.00  # what our order status says we actually filled
    action, new_size, msg = recalc_buy_size(original_size, total_filled_usd, baseline_usd, usd_avail, 12000.00)
    print(f"  baseline=${baseline_usd:,.2f}  cash=${usd_avail:,.2f}  spent_by_wallet=${baseline_usd - usd_avail:,.2f}  spent_by_orders=${total_filled_usd:,.2f}")
    print(f"  → new_size=${new_size:,.2f}")
    # spent_so_far = max($3000, $2000) = $3000 (use conservative — assume we spent more)
    # remaining = $12k - $3k = $9000
    # next_size = min($11,999.99, $9000) = $9000
    assert abs(new_size - 9000) < 0.05
    print(f"  ✓ PASS: uses larger spent estimate → smaller next leg → no risk of over-spend")


def test_first_iter_no_fill():
    """First iteration, post_only rejected, no fill. baseline preserved, no accumulation."""
    print("\n=== TEST 9: First iteration, no fill yet ===")
    original_size = 12000.00
    baseline_usd = 12000.00
    total_filled_usd = 0.0
    usd_avail = 12000.00
    action, new_size, msg = recalc_buy_size(original_size, total_filled_usd, baseline_usd, usd_avail, 11999.99)
    print(f"  baseline=${baseline_usd:,.2f}  filled=$0  cash=${usd_avail:,.2f}")
    print(f"  → new_size=${new_size:,.2f}")
    # remaining = $12k. next_size = min($11,999.99, $12k) = $11,999.99. Same as current size.
    assert abs(new_size - 11999.99) < 0.02
    print(f"  ✓ PASS: no change when no fills happened")


if __name__ == '__main__':
    test_today_bug_scenario()
    test_wallet_equals_target()
    test_wallet_much_more_than_target()
    test_wallet_less_than_target()
    test_wallet_below_min_after_partial()
    test_target_essentially_met()
    test_no_baseline()
    test_cross_check_disagreement()
    test_first_iter_no_fill()
    print("\n=== ALL TESTS PASSED ===")
