"""Unit tests for Mode T convergence logic: strict, tolerant, 2-cycle, max_iter.

Extracts the convergence helpers out of crypto_trading_system_ed.py and runs them
against synthetic fingerprint sequences + retrospective Desktop 2mo data.

Run: python tools/test_mode_t_convergence.py
"""
import sys
import os

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

# Tolerance constants (mirror the values in crypto_trading_system_ed.py)
TOL_PNL = 0.10
TOL_HOLD = 2
TOL_GATE_H = 4
TOL_GATE_T = 0.5
TOL_GATE_CD = 6


def _rc_close(rc_a, rc_b):
    if rc_a is None and rc_b is None:
        return True
    def _is_null(rc):
        return rc is None or all(v is None for v in rc)
    if _is_null(rc_a) and _is_null(rc_b):
        return True
    if _is_null(rc_a) or _is_null(rc_b):
        return False
    try:
        hs_a, hl_a, ts_a, tl_a, cd_a = rc_a
        hs_b, hl_b, ts_b, tl_b, cd_b = rc_b
        if abs(int(hs_a) - int(hs_b)) > TOL_GATE_H:  return False
        if abs(int(hl_a) - int(hl_b)) > TOL_GATE_H:  return False
        if abs(float(ts_a) - float(ts_b)) > TOL_GATE_T: return False
        if abs(float(tl_a) - float(tl_b)) > TOL_GATE_T: return False
        if abs(int(cd_a) - int(cd_b)) > TOL_GATE_CD: return False
        return True
    except (TypeError, ValueError):
        return False


def _configs_close_enough(fp_now, fp_prev):
    if fp_now is None or fp_prev is None:
        return False
    try:
        pnl_n, hold_n, bs_n, bz_n, art_n, brt_n, zrt_n = fp_now
        pnl_p, hold_p, bs_p, bz_p, art_p, brt_p, zrt_p = fp_prev
    except (TypeError, ValueError):
        return False
    if bs_n != bs_p or bz_n != bz_p:
        return False
    try:
        if (pnl_n is None) != (pnl_p is None): return False
        if pnl_n is not None and abs(float(pnl_n) - float(pnl_p)) > TOL_PNL:
            return False
    except (TypeError, ValueError):
        return False
    try:
        if (hold_n is None) != (hold_p is None): return False
        if hold_n is not None and abs(int(hold_n) - int(hold_p)) > TOL_HOLD:
            return False
    except (TypeError, ValueError):
        return False
    if not _rc_close(art_n, art_p): return False
    if not _rc_close(brt_n, brt_p): return False
    if not _rc_close(zrt_n, zrt_p): return False
    return True


def simulate_convergence(fingerprints, max_iter=6):
    """Run the Mode T convergence algorithm against a sequence of pre-computed fingerprints.
    Returns dict with status + iteration it stopped at."""
    fp_history = []
    for iteration in range(1, min(max_iter, len(fingerprints)) + 1):
        fp = fingerprints[iteration - 1]
        # 1. Strict
        if fp_history and fp == fp_history[-1]:
            return {'status': 'converged', 'iter': iteration, 'final_fp': fp}
        # 2. Tolerant
        if fp_history and _configs_close_enough(fp, fp_history[-1]):
            return {'status': 'converged_in_tolerance', 'iter': iteration, 'final_fp': fp}
        # 3. 2-cycle
        if len(fp_history) >= 2 and _configs_close_enough(fp, fp_history[-2]):
            return {'status': 'cycle_detected', 'iter': iteration, 'final_fp': fp}
        fp_history.append(fp)
        if len(fp_history) > 2:
            fp_history = fp_history[-2:]
    return {'status': 'max_iter', 'iter': len(fingerprints), 'final_fp': fingerprints[-1]}


def make_fp(pnl, hold, bull_sh, bear_sh, bull_gate=None, bear_gate=None, asset_gate=None):
    """Build a fingerprint tuple matching _config_fingerprint()'s output."""
    return (pnl, hold, bull_sh, bear_sh, asset_gate or (None,)*5, bull_gate or (None,)*5, bear_gate or (None,)*5)


# ─── Tests ───────────────────────────────────────────────────────────────────
def run_tests():
    results = []
    def case(name, fps, expected):
        out = simulate_convergence(fps)
        ok = out['status'] == expected
        results.append((name, expected, out['status'], out['iter'], ok))
        return ok

    # A. Strict exact match at iter 2 (identical fingerprints)
    A = make_fp(0.30, 10, True, False,
                bull_gate=(18, 36, 3.5, 4.5, 30),
                bear_gate=(14, 24, 3.5, 5.5, 24))
    case('strict_exact_at_2', [A, A], 'converged')

    # B. Tolerant plateau at iter 2 (iter1 & iter2 close but not identical)
    B1 = make_fp(0.30, 10, True, False,
                 bull_gate=(18, 36, 3.5, 5.0, 30),
                 bear_gate=(14, 24, 3.5, 5.5, 24))
    B2 = make_fp(0.35, 10, True, False,
                 bull_gate=(18, 36, 3.5, 5.0, 30),
                 bear_gate=(14, 24, 3.5, 5.5, 24))
    case('tolerant_plateau_at_2', [B1, B2], 'converged_in_tolerance')

    # C. 2-cycle: A-B-A sequence
    C1 = make_fp(0.30, 8, True, False,
                 bull_gate=(18, 36, 3.5, 4.5, 30),
                 bear_gate=(10, 24, 6.0, 5.5, 24))
    C2 = make_fp(0.45, 12, True, False,
                 bull_gate=(18, 36, 6.0, 4.5, 36),
                 bear_gate=(16, 24, 4.0, 5.5, 24))
    case('cycle_A_B_A_at_3', [C1, C2, C1], 'cycle_detected')

    # D. Non-cyclic 3-sequence, all different — should reach max_iter
    D3 = make_fp(0.50, 14, True, False,
                 bull_gate=(22, 44, 7.0, 6.0, 42),
                 bear_gate=(20, 30, 5.0, 6.0, 30))
    case('non_cyclic_3_reaches_max_iter', [C1, C2, D3], 'max_iter')

    # E. Shield flip between iterations should NOT count as tolerant convergence
    E1 = make_fp(0.30, 10, True, False)
    E2 = make_fp(0.30, 10, False, True)  # shield flipped
    case('shield_flip_not_tolerant', [E1, E2], 'max_iter')

    # F. Large gate jump (t_short 3.5→6.0) should NOT count as tolerant
    F1 = make_fp(0.30, 10, True, False,
                 bull_gate=(18, 36, 3.5, 4.5, 30))
    F2 = make_fp(0.30, 10, True, False,
                 bull_gate=(18, 36, 6.0, 4.5, 30))
    case('large_gate_jump_not_tolerant', [F1, F2], 'max_iter')

    # G. Tolerant A-B-A' where A' is close to A (cycle caught at iter 3 via tolerance)
    G1 = make_fp(0.30, 8, True, False,
                 bull_gate=(18, 36, 3.5, 4.5, 30),
                 bear_gate=(10, 24, 6.0, 5.5, 24))
    G2 = make_fp(0.40, 12, True, False,
                 bull_gate=(20, 40, 6.5, 5.5, 40),
                 bear_gate=(14, 24, 4.0, 5.5, 30))
    G3 = make_fp(0.32, 8, True, False,   # G3 close to G1 (pnl diff 0.02, rest same)
                 bull_gate=(18, 36, 3.5, 4.5, 30),
                 bear_gate=(10, 24, 6.0, 5.5, 24))
    case('tolerant_cycle_A_B_Aprime', [G1, G2, G3], 'cycle_detected')

    # H. Max-iter stress: 6 all-different configs, no convergence
    H_fps = []
    for i in range(6):
        H_fps.append(make_fp(0.2 + 0.1*i, 6 + 2*i, True, False,
                             bull_gate=(10 + i*3, 20 + i*5, 2.0 + i*0.8, 3.0 + i*0.7, 8 + i*8)))
    case('all_different_6_iter_max', H_fps, 'max_iter')

    # ─── Print results ───
    print('='*80)
    print('  MODE T CONVERGENCE — UNIT TESTS')
    print('='*80)
    print(f'{"name":<40} {"expected":<25} {"actual":<25} {"iter":>4} {"ok":>3}')
    print('-'*100)
    passed = failed = 0
    for name, expected, actual, it, ok in results:
        tag = 'OK' if ok else 'FAIL'
        print(f'{name:<40} {expected:<25} {actual:<25} {it:>4} {tag:>3}')
        if ok: passed += 1
        else: failed += 1
    print('-'*100)
    print(f'{passed} passed, {failed} failed')
    return failed == 0


def retrospective_desktop_2mo():
    """Apply new convergence logic to Desktop's actual 2mo iteration history.
    Sourced from logs/ed_v1_20260421_120733.log (iter 1-4 T winners + gate winners).
    Expected behavior: detect cycle or stop early, instead of running all 4 iterations."""
    print()
    print('='*80)
    print('  RETROSPECTIVE: Desktop 2mo ETH HRST (would new logic have caught it?)')
    print('='*80)

    # From logs/ed_v1_20260421_120733.log iterations 1-4:
    fp1 = make_fp(0.35, 8, True, False,
                  bull_gate=(18, 36, 3.5, 4.5, 30),
                  bear_gate=(10, 24, 6.0, 5.5, 24))
    fp2 = make_fp(0.30, 10, True, False,
                  bull_gate=(18, 36, 6.0, 4.5, 36),
                  bear_gate=(16, 24, 4.0, 5.5, 24))
    fp3 = make_fp(0.30, 8, True, False,
                  bull_gate=(18, 36, 3.5, 5.0, 30),
                  bear_gate=(10, 24, 6.0, 5.5, 24))
    fp4 = make_fp(0.40, 10, True, False,
                  bull_gate=(18, 36, 3.5, 4.5, 30),
                  bear_gate=(14, 24, 3.5, 5.5, 24))

    # Check pairwise closeness
    print('Pairwise tolerance matrix:')
    fps = [('iter 1', fp1), ('iter 2', fp2), ('iter 3', fp3), ('iter 4', fp4)]
    for i, (na, a) in enumerate(fps):
        for nb, b in fps[i+1:]:
            print(f'  {na} ↔ {b[0] if isinstance(b, tuple) else nb}: '
                  f'close={_configs_close_enough(a, b)}')
    print()
    # Run the simulation
    out = simulate_convergence([fp1, fp2, fp3, fp4], max_iter=6)
    print(f'Verdict: {out["status"]} at iter {out["iter"]}')
    if out['status'] == 'max_iter':
        print('  → Desktop would STILL hit max_iter under new rules. No early stop.')
        print('  → This means Desktop\'s iterations were too randomly spread for cycle detection.')
    elif out['status'] == 'cycle_detected':
        print('  → Desktop would have stopped early with CYCLE flag.')
        print('  → Promotion gate #1 correctly rejects this config.')
    elif out['status'] == 'converged_in_tolerance':
        print('  → Desktop would have converged-in-tolerance.')
        print('  → Promotion gate #1 accepts this config.')


def retrospective_laptop_2mo_trim():
    """Apply new logic to laptop 2mo+trim iterations.
    Sourced from logs/ed_v1_20260421_212008.log."""
    print()
    print('='*80)
    print('  RETROSPECTIVE: Laptop 2mo+trim ETH HRST')
    print('='*80)

    fp1 = make_fp(0.45, 10, True, False,
                  bull_gate=(14, 30, 6.0, 5.5, 16),
                  bear_gate=(10, 16, 4.5, 5.5, 16))
    fp2 = make_fp(0.55, 12, False, True,
                  bull_gate=(14, 20, 6.0, 5.5, 10),
                  bear_gate=(8, 12, 3.0, 2.0, 16))
    fp3 = make_fp(0.50, 10, False, True,
                  bull_gate=(14, 20, 6.0, 5.5, 10),
                  bear_gate=(8, 12, 3.0, 2.0, 16))
    fp4 = make_fp(0.50, 10, False, True,
                  bull_gate=(14, 20, 6.0, 5.5, 10),
                  bear_gate=(8, 12, 3.0, 2.0, 16))

    out = simulate_convergence([fp1, fp2, fp3, fp4], max_iter=6)
    print(f'Verdict: {out["status"]} at iter {out["iter"]}')
    if out['status'] == 'converged':
        print('  → Laptop\'s iter 3 == iter 4 → STRICT convergence. Correct.')


if __name__ == '__main__':
    ok = run_tests()
    retrospective_desktop_2mo()
    retrospective_laptop_2mo_trim()
    sys.exit(0 if ok else 1)
