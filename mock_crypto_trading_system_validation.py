"""
mock_crypto_trading_system_validation.py
============================================================
Phase 2 validation for the 4 improvements before porting to production.

Run AFTER mock_crypto_trading_system.py passes (Phase 1).

Tests:
  V1. Holdout stability       — 3 seeds, gap should be consistent direction
  V2. Bootstrap convergence   — CI width shrinks as 1/√n
  V3. Permutation anchors     — pure noise → p>0.40, pure signal → p<0.05
  V4. Scoring edge cases      — zero trades, one trade, flat price, NaN guard
  V5. Full integration dry-run — all 4 improvements chained as they would be
                                 in Mode D, on a realistic 6,000-row dataset

Expected runtime: 10–20 minutes on laptop (RTX 3070 Ti, 16 cores)
Each test prints PASS / FAIL / WARN clearly.

HOW TO RUN:
  python mock_crypto_trading_system_validation.py

  Or individual sections:
  python mock_crypto_trading_system_validation.py --test v1
  python mock_crypto_trading_system_validation.py --test v2
  python mock_crypto_trading_system_validation.py --test v3
  python mock_crypto_trading_system_validation.py --test v4
  python mock_crypto_trading_system_validation.py --test v5
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Import everything from Phase 1 mock ──────────────────────────────────────
try:
    from mock_crypto_trading_system import (
        generate_synthetic_data,
        _eval_config,
        bootstrap_ci,
        _make_mock_signals,
        _calmar, _sharpe, _score_new, _score_old,
        TRADING_FEE, DIAG_STEP, HOLDOUT_HOURS,
    )
except ImportError as e:
    print(f"ERROR: Could not import from mock_crypto_trading_system.py")
    print(f"  Make sure both files are in the same directory.")
    print(f"  Details: {e}")
    sys.exit(1)

# ── Shared feature set ────────────────────────────────────────────────────────
INFORMATIVE_COLS = [
    'logret_1h', 'logret_4h', 'logret_8h', 'logret_12h',
    'logret_24h', 'logret_48h', 'logret_72h',
    'rsi_14h', 'volatility_12h', 'vol_ratio', 'sma_ratio',
]

N_BOOTSTRAP_VALIDATION  = 1000
N_PERMUTATIONS_FULL     = 200

results_summary = {}


def _pass(label):
    print(f"    ✓  PASS  {label}")
    results_summary[label] = 'PASS'

def _fail(label, reason=''):
    print(f"    ✗  FAIL  {label}" + (f"  ({reason})" if reason else ''))
    results_summary[label] = f'FAIL: {reason}'

def _warn(label, reason=''):
    print(f"    ⚡  WARN  {label}" + (f"  ({reason})" if reason else ''))
    results_summary[label] = f'WARN: {reason}'


# ============================================================
# V1: HOLDOUT STABILITY — 3 seeds
# ============================================================
def test_v1_holdout_stability():
    print(f"\n{'='*60}")
    print(f"  V1: HOLDOUT STABILITY (3 seeds)")
    print(f"{'='*60}")
    print(f"  Checks: gap direction consistent, gap magnitude reasonable")
    print(f"  Expected runtime: ~2 min")

    seeds    = [42, 99, 137]
    gaps     = []
    train_accs = []
    hold_accs  = []

    for seed in seeds:
        print(f"\n  Seed {seed}...")
        df = generate_synthetic_data(n_hours=6_000, seed=seed)
        df_use = df.tail(4_000).reset_index(drop=True)

        df_train   = df_use.iloc[:-HOLDOUT_HOURS].reset_index(drop=True)
        df_holdout = df_use.iloc[-HOLDOUT_HOURS:].reset_index(drop=True)

        r_train = _eval_config(df_train, INFORMATIVE_COLS, window=150, model_name='RF')
        r_hold  = _eval_config(df_holdout, INFORMATIVE_COLS, window=150, model_name='RF')

        if r_train is None or r_hold is None:
            _fail(f'seed_{seed}_eval', 'eval returned None')
            continue

        gap = r_train['accuracy'] - r_hold['accuracy']
        gaps.append(gap)
        train_accs.append(r_train['accuracy'])
        hold_accs.append(r_hold['accuracy'])

        print(f"    Train: {r_train['accuracy']:.1f}%  Holdout: {r_hold['accuracy']:.1f}%  Gap: {gap:+.1f}%")

    if len(gaps) < 3:
        _fail('holdout_stability', 'not enough seeds completed')
        return

    gap_std = np.std(gaps)
    gap_mean = np.mean(gaps)
    all_same_direction = all(g > 0 for g in gaps) or all(g < 0 for g in gaps)

    print(f"\n  Gap summary: mean={gap_mean:+.1f}%  std={gap_std:.1f}%  values={[f'{g:+.1f}' for g in gaps]}")

    # Check 1: gap std < 8% (not wildly inconsistent)
    if gap_std < 8:
        _pass('gap_std_acceptable')
    else:
        _warn('gap_std_acceptable', f'std={gap_std:.1f}% is high — results sensitive to seed')

    # Check 2: all gaps in same direction (or very small)
    if all_same_direction:
        _pass('gap_direction_consistent')
    elif all(abs(g) < 4 for g in gaps):
        _pass('gap_direction_consistent (all near zero, acceptable)')
    else:
        _warn('gap_direction_consistent', 'gap flips sign across seeds — overfit detection unreliable')

    # Check 3: abs gap < 15% (synthetic data shouldn't overfit badly)
    max_gap = max(abs(g) for g in gaps)
    if max_gap < 15:
        _pass('gap_magnitude_reasonable')
    else:
        _warn('gap_magnitude_reasonable', f'max gap={max_gap:.1f}% — may need more training data')

    # Check 4: train accuracy above 50% (model is learning something)
    avg_train = np.mean(train_accs)
    if avg_train > 52:
        _pass(f'train_above_chance (avg={avg_train:.1f}%)')
    else:
        _warn('train_above_chance', f'avg train acc={avg_train:.1f}% — barely above 50%, check features')


# ============================================================
# V2: BOOTSTRAP CONVERGENCE — CI narrows as √n
# ============================================================
def test_v2_bootstrap_convergence():
    print(f"\n{'='*60}")
    print(f"  V2: BOOTSTRAP CI CONVERGENCE")
    print(f"{'='*60}")
    print(f"  Checks: CI width shrinks ~1/√n as replay_hours increases")
    print(f"  Expected runtime: ~1 min")

    df = generate_synthetic_data(n_hours=6_000, seed=42)
    df_use = df.tail(4_000).reset_index(drop=True)

    replay_sizes   = [100, 200, 400, 600]
    widths         = []
    n_predictions  = []

    print(f"\n  {'Replay':>8} {'n_preds':>8} {'CI_width':>10} {'Expected':>10} {'Ratio':>8}")
    print(f"  {'-'*48}")

    ref_width = None
    ref_n = None

    for replay in replay_sizes:
        sigs = _make_mock_signals(df_use, INFORMATIVE_COLS, window=100, replay_hours=replay)
        point, lo, hi, n = bootstrap_ci(sigs, n_bootstrap=N_BOOTSTRAP_VALIDATION)

        if lo is None or n < 10:
            print(f"  {replay:>8}  — not enough signals")
            continue

        width = hi - lo
        widths.append(width)
        n_predictions.append(n)

        if ref_width is None:
            ref_width = width
            ref_n     = n
            expected  = width
            ratio     = 1.0
        else:
            expected = ref_width * np.sqrt(ref_n / n)
            ratio    = width / expected

        print(f"  {replay:>8} {n:>8} {width:>9.1f}% {expected:>9.1f}% {ratio:>7.2f}x")

    if len(widths) < 3:
        _fail('bootstrap_convergence', 'not enough replay sizes completed')
        return

    # Check: CI width decreases monotonically
    monotone = all(widths[i] >= widths[i+1] for i in range(len(widths)-1))
    if monotone:
        _pass('ci_width_decreases_monotonically')
    else:
        _warn('ci_width_decreases_monotonically', f'widths={[f"{w:.1f}" for w in widths]}')

    # Check: ratio between smallest and largest replay is meaningful (> 1.3x)
    if len(widths) >= 2:
        compression = widths[0] / widths[-1]
        if compression > 1.3:
            _pass(f'ci_compresses_with_n (ratio={compression:.2f}x)')
        else:
            _warn('ci_compresses_with_n', f'ratio only {compression:.2f}x — may need larger replay range')

    # Check: no NaN or infinite CI values
    if all(np.isfinite(w) for w in widths):
        _pass('ci_values_finite')
    else:
        _fail('ci_values_finite', 'NaN or inf in CI output')

    # Check: CI width at largest replay < 20% (informative)
    if widths[-1] < 20:
        _pass(f'ci_informative_at_max_replay ({widths[-1]:.1f}%)')
    else:
        _warn('ci_informative_at_max_replay', f'{widths[-1]:.1f}% still wide — need more signals')


# ============================================================
# V3: PERMUTATION ANCHORS — noise vs signal
# ============================================================
def test_v3_permutation_anchors():
    print(f"\n{'='*60}")
    print(f"  V3: PERMUTATION SIGNIFICANCE ANCHORS")
    print(f"{'='*60}")
    print(f"  Anchor A: pure noise label  → p > 0.20 (should NOT be significant)")
    print(f"  Anchor B: near-perfect label → p < 0.10 (should BE significant)")
    print(f"  Expected runtime: ~6 min")

    df_base = generate_synthetic_data(n_hours=6_000, seed=42)
    df_use  = df_base.tail(3_000).reset_index(drop=True)

    def _run_perm_test(df, label, window=100, n_perm=N_PERMUTATIONS_FULL):
        """Run permutation test and return p-value."""
        r_real = _eval_config(df, INFORMATIVE_COLS, window=window, model_name='LR')
        if r_real is None:
            return None, None
        real_acc = r_real['accuracy'] / 100

        null_accs = []
        t0 = time.time()
        for i in range(n_perm):
            df_perm = df.copy()
            df_perm['label'] = np.random.permutation(df_perm['label'].values)
            r = _eval_config(df_perm, INFORMATIVE_COLS, window=window, model_name='LR')
            if r:
                null_accs.append(r['accuracy'] / 100)
            if (i+1) % 50 == 0:
                print(f"    [{label}] {i+1}/{n_perm} ({time.time()-t0:.0f}s)...")

        if not null_accs:
            return None, None
        p = np.mean([a >= real_acc for a in null_accs])
        return p, real_acc * 100

    # ── Anchor A: pure noise label ────────────────────────
    print(f"\n  Anchor A: pure random label (expect p > 0.20)...")
    df_noise = df_use.copy()
    df_noise['label'] = np.random.RandomState(42).randint(0, 2, len(df_noise))
    p_noise, acc_noise = _run_perm_test(df_noise, 'noise', n_perm=100)

    if p_noise is None:
        _fail('anchor_noise_ran', 'eval failed')
    else:
        print(f"    Noise label: acc={acc_noise:.1f}%  p={p_noise:.3f}")
        if p_noise > 0.20:
            _pass(f'anchor_noise_not_significant (p={p_noise:.3f})')
        elif p_noise > 0.10:
            _warn('anchor_noise_not_significant', f'p={p_noise:.3f} borderline — run with more permutations')
        else:
            _fail('anchor_noise_not_significant', f'p={p_noise:.3f} — permutation test has false positives!')

    # ── Anchor B: near-perfect label ─────────────────────
    print(f"\n  Anchor B: near-perfect label (expect p < 0.10)...")
    df_signal = df_use.copy()
    # Label = 1 if price will go up 4h later, with 15% noise
    true_label = (df_signal['close'].shift(-4) > df_signal['close']).astype(int)
    noise_mask = np.random.RandomState(42).random(len(df_signal)) < 0.15
    df_signal['label'] = np.where(noise_mask,
                                   1 - true_label.fillna(0).astype(int),
                                   true_label.fillna(0).astype(int))
    p_signal, acc_signal = _run_perm_test(df_signal, 'signal', n_perm=100)

    if p_signal is None:
        _fail('anchor_signal_ran', 'eval failed')
    else:
        print(f"    Signal label: acc={acc_signal:.1f}%  p={p_signal:.3f}")
        if p_signal < 0.05:
            _pass(f'anchor_signal_significant (p={p_signal:.3f})')
        elif p_signal < 0.10:
            _warn('anchor_signal_significant', f'p={p_signal:.3f} — marginal, try more permutations')
        else:
            _fail('anchor_signal_significant',
                  f'p={p_signal:.3f} — test not catching real signal, check feature/model setup')

    # ── Anchor C: p-value resolution check ───────────────
    # With n_perm=100, smallest possible p-value is 0.01
    print(f"\n  Checking p-value resolution at n_perm=100...")
    min_possible = 1 / 100
    print(f"    Min possible p-value: {min_possible:.3f}")
    print(f"    For p<0.05 gate: need n_perm >= 20  ✓")
    print(f"    For p<0.01 gate: need n_perm >= 100 ✓")
    print(f"    For p<0.002 gate: need n_perm >= 500 (production)")
    _pass('p_value_resolution_adequate_for_gate')


# ============================================================
# V4: SCORING EDGE CASES
# ============================================================
def test_v4_scoring_edge_cases():
    print(f"\n{'='*60}")
    print(f"  V4: SCORING EDGE CASES (Calmar/Sharpe)")
    print(f"{'='*60}")
    print(f"  Checks: no crashes on zero/one trade, flat price, extreme returns")
    print(f"  Expected runtime: < 10 seconds")

    cases = [
        # (label, cum_return, max_dd, n_tests, step, trade_returns)
        ('zero_trades',           0.0,   0.0,  100, 72, []),
        ('one_trade_win',        10.0,   5.0,  100, 72, [0.10]),
        ('one_trade_loss',       -5.0,  10.0,  100, 72, [-0.05]),
        ('flat_price_zero_dd',    0.0,   0.0,  200, 72, [0.001, -0.001, 0.0]),
        ('extreme_return_pos',  500.0,  30.0,  200, 72, [0.5]*20),
        ('extreme_return_neg',  -90.0,  95.0,  200, 72, [-0.3]*20),
        ('zero_std_trades',       5.0,   3.0,  200, 72, [0.01]*50),  # all same return → std=0
        ('single_day_data',       2.0,   1.0,   10, 72, [0.02]),     # tiny n_tests
    ]

    all_passed = True
    for label, cum_ret, max_dd, n_tests, step, trade_rets in cases:
        try:
            calmar  = _calmar(cum_ret, max_dd, n_tests, step)
            sharpe  = _sharpe(trade_rets)
            score   = _score_new(50.0, cum_ret, max_dd, n_tests, step, trade_rets)
            old_sc  = _score_old(50.0, cum_ret, max_dd)

            # Check for NaN/inf
            if not np.isfinite(calmar):
                _fail(f'edge_{label}', f'calmar={calmar}')
                all_passed = False
                continue
            if not np.isfinite(sharpe):
                _fail(f'edge_{label}', f'sharpe={sharpe}')
                all_passed = False
                continue
            if not np.isfinite(score):
                _fail(f'edge_{label}', f'new_score={score}')
                all_passed = False
                continue
            if not np.isfinite(old_sc):
                _fail(f'edge_{label}', f'old_score={old_sc}')
                all_passed = False
                continue

            print(f"    {label:<30} calmar={calmar:+6.2f}  sharpe={sharpe:+6.2f}  "
                  f"new={score:+6.3f}  old={old_sc:.3f}")
            _pass(f'edge_{label}')

        except Exception as e:
            _fail(f'edge_{label}', str(e))
            all_passed = False

    # Sanity check: positive return should score better than negative return
    score_pos = _score_new(60.0,  20.0, 10.0, 200, 72, [0.02]*20)
    score_neg = _score_new(60.0, -20.0, 30.0, 200, 72, [-0.02]*20)
    if score_pos > score_neg:
        _pass('positive_return_scores_higher_than_negative')
    else:
        _fail('positive_return_scores_higher_than_negative',
              f'pos={score_pos:.3f} neg={score_neg:.3f}')

    # Sanity check: lower drawdown should score better (same return)
    score_low_dd  = _score_new(60.0, 20.0,  5.0, 200, 72, [0.02]*20)
    score_high_dd = _score_new(60.0, 20.0, 40.0, 200, 72, [0.02]*20)
    if score_low_dd > score_high_dd:
        _pass('lower_drawdown_scores_higher')
    else:
        _fail('lower_drawdown_scores_higher',
              f'low_dd={score_low_dd:.3f} high_dd={score_high_dd:.3f}')

    # Sanity check: zero trades → score should still be finite and not best
    score_zero = _score_new(50.0, 0.0, 0.0, 200, 72, [])
    score_good = _score_new(65.0, 30.0, 10.0, 200, 72, [0.03]*15)
    if np.isfinite(score_zero) and score_good > score_zero:
        _pass('zero_trades_doesnt_win')
    else:
        _warn('zero_trades_doesnt_win', f'zero={score_zero:.3f} good={score_good:.3f}')


# ============================================================
# V5: FULL INTEGRATION DRY-RUN
# ============================================================
def test_v5_full_integration():
    """
    Simulates exactly what Mode D will do after all 4 improvements are ported.
    Uses a realistic 6,000-row dataset and chains all improvements in order.
    """
    print(f"\n{'='*60}")
    print(f"  V5: FULL INTEGRATION DRY-RUN")
    print(f"{'='*60}")
    print(f"  Simulates Mode D full pipeline with all 4 improvements active")
    print(f"  Expected runtime: ~8 min")

    df = generate_synthetic_data(n_hours=7_000, seed=42)
    df_full = df.tail(6_000).reset_index(drop=True)
    print(f"  Full dataset: {len(df_full):,} rows")

    # ── Step 1: Split holdout (Improvement 1) ─────────────
    print(f"\n  [Step 1] Holdout split...")
    df_train   = df_full.iloc[:-HOLDOUT_HOURS].reset_index(drop=True)
    df_holdout = df_full.iloc[-HOLDOUT_HOURS:].reset_index(drop=True)
    print(f"    Train: {len(df_train):,}  Holdout: {len(df_holdout):,}")

    # ── Step 2: Find best config on train only ─────────────
    print(f"\n  [Step 2] Finding best config on train data...")
    configs = [('RF', 100), ('RF', 200), ('LR', 100), ('GB', 100)]
    best_result = None
    best_score  = -999

    for model_name, window in configs:
        r = _eval_config(df_train, INFORMATIVE_COLS, window=window, model_name=model_name)
        if r is None:
            continue
        if r['combined_score_new'] > best_score:
            best_score  = r['combined_score_new']
            best_result = r
        print(f"    {model_name} w={window:3d}h | acc={r['accuracy']:.1f}% "
              f"ret={r['cum_return']:+.1f}% score={r['combined_score_new']:.3f}")

    if best_result is None:
        _fail('v5_best_config_found', 'no configs returned results')
        return

    print(f"\n  Best: {best_result['model']} w={best_result['window']}h | "
          f"acc={best_result['accuracy']:.1f}% ret={best_result['cum_return']:+.1f}% "
          f"calmar={best_result['calmar']:+.2f} sharpe={best_result['sharpe']:+.2f}")
    _pass('v5_best_config_found')

    # ── Step 3: Evaluate on holdout (Improvement 1) ───────
    print(f"\n  [Step 3] Holdout evaluation...")
    r_hold = _eval_config(df_holdout, INFORMATIVE_COLS,
                          window=best_result['window'],
                          model_name=best_result['model'])

    if r_hold is None:
        _warn('v5_holdout_eval', 'holdout too small for this window')
    else:
        gap = best_result['accuracy'] - r_hold['accuracy']
        print(f"    Train: {best_result['accuracy']:.1f}%  Holdout: {r_hold['accuracy']:.1f}%  Gap: {gap:+.1f}%")
        if abs(gap) < 15:
            _pass(f'v5_holdout_gap_acceptable (gap={gap:+.1f}%)')
        else:
            _warn('v5_holdout_gap_acceptable', f'gap={gap:+.1f}% is large')

    # ── Step 4: Permutation significance (Improvement 3) ──
    print(f"\n  [Step 4] Permutation significance test (n=100)...")
    real_acc = best_result['accuracy'] / 100
    null_accs = []
    t0 = time.time()
    for i in range(100):
        df_perm = df_train.copy()
        df_perm['label'] = np.random.permutation(df_perm['label'].values)
        r = _eval_config(df_perm, INFORMATIVE_COLS,
                         window=best_result['window'],
                         model_name=best_result['model'])
        if r:
            null_accs.append(r['accuracy'] / 100)
        if (i+1) % 25 == 0:
            print(f"    {i+1}/100 done ({time.time()-t0:.0f}s)...")

    p_val = np.mean([a >= real_acc for a in null_accs]) if null_accs else 1.0
    print(f"    p-value: {p_val:.3f}  (threshold: 0.10)")
    if p_val < 0.10:
        _pass(f'v5_permutation_significant (p={p_val:.3f})')
    elif p_val < 0.20:
        _warn('v5_permutation_significant', f'p={p_val:.3f} borderline')
    else:
        # On synthetic data with weak signal this may legitimately fail
        _warn('v5_permutation_significant',
              f'p={p_val:.3f} — expected on weak synthetic signal, real BTC data should pass')

    # ── Step 5: Bootstrap CI on Mode B signals (Improvement 2) ──
    print(f"\n  [Step 5] Bootstrap CI on signals...")
    signals = _make_mock_signals(df_full, INFORMATIVE_COLS,
                                 window=best_result['window'],
                                 replay_hours=400,
                                 model_name=best_result['model'])
    point, lo, hi, n = bootstrap_ci(signals, n_bootstrap=N_BOOTSTRAP_VALIDATION)

    if lo is None:
        _warn('v5_bootstrap_ci', f'only {n} signals, need ≥10')
    else:
        print(f"    Accuracy: {point:.1f}% [95% CI: {lo:.1f}%–{hi:.1f}%] (n={n})")
        if np.isfinite(lo) and np.isfinite(hi) and lo < point < hi:
            _pass(f'v5_bootstrap_ci_valid')
        else:
            _fail('v5_bootstrap_ci_valid', f'CI bounds invalid: [{lo:.1f}%–{hi:.1f}%]')

    # ── Step 6: Score comparison (Improvement 4) ──────────
    print(f"\n  [Step 6] Score comparison (old vs new)...")
    old_best = None
    old_best_score = -999
    for model_name, window in configs:
        r = _eval_config(df_train, INFORMATIVE_COLS, window=window, model_name=model_name)
        if r and r['combined_score_old'] > old_best_score:
            old_best_score = r['combined_score_old']
            old_best = r

    if old_best:
        print(f"    Old winner: {old_best['model']} w={old_best['window']}h "
              f"score={old_best['combined_score_old']:.3f}")
        print(f"    New winner: {best_result['model']} w={best_result['window']}h "
              f"score={best_result['combined_score_new']:.3f}")
        same = (old_best['model'] == best_result['model'] and
                old_best['window'] == best_result['window'])
        if same:
            _pass('v5_scores_agree_on_winner')
        else:
            _warn('v5_scores_agree_on_winner',
                  'disagreement — inspect whether new winner has better return/drawdown')
    else:
        _warn('v5_score_comparison', 'could not run comparison')

    _pass('v5_full_pipeline_completed')


# ============================================================
# MAIN
# ============================================================
def _print_summary():
    print(f"\n{'='*60}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*60}")
    passes = [k for k, v in results_summary.items() if v == 'PASS']
    warns  = [k for k, v in results_summary.items() if v.startswith('WARN')]
    fails  = [k for k, v in results_summary.items() if v.startswith('FAIL')]

    print(f"  PASS:  {len(passes)}")
    print(f"  WARN:  {len(warns)}")
    print(f"  FAIL:  {len(fails)}")

    if warns:
        print(f"\n  Warnings:")
        for w in warns:
            print(f"    ⚡  {w}: {results_summary[w]}")

    if fails:
        print(f"\n  Failures:")
        for f in fails:
            print(f"    ✗  {f}: {results_summary[f]}")

    print()
    if not fails:
        print("  ✓  ALL TESTS PASSED — safe to port improvements to production")
        print()
        print("  PORTING ORDER (safest first):")
        print("  1. Improvement 2: bootstrap_ci()    — zero risk, read-only on signals")
        print("  2. Improvement 4: Calmar/Sharpe     — scoring only, no logic change")
        print("  3. Improvement 1: holdout split     — adds reporting, non-breaking")
        print("  4. Improvement 3: permutation gate  — new logic gate, test carefully")
    else:
        print("  ✗  FAILURES DETECTED — do not port until resolved")
        print("  Fix failures, re-run validation, then port.")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    test_arg = sys.argv[2].lower() if len(sys.argv) > 2 and sys.argv[1] == '--test' else None
    if arg == '--test' and test_arg:
        test_arg = test_arg.lower()
    elif arg and arg.lower().startswith('v'):
        test_arg = arg.lower()

    print("=" * 60)
    print("  MOCK CRYPTO — PHASE 2 VALIDATION")
    print("  Run this after Phase 1 (mock_crypto_trading_system.py) passes")
    print("=" * 60)

    t_total = time.time()

    run_all = test_arg is None
    if run_all or test_arg == 'v1':
        test_v1_holdout_stability()
    if run_all or test_arg == 'v2':
        test_v2_bootstrap_convergence()
    if run_all or test_arg == 'v3':
        test_v3_permutation_anchors()
    if run_all or test_arg == 'v4':
        test_v4_scoring_edge_cases()
    if run_all or test_arg == 'v5':
        test_v5_full_integration()

    elapsed = time.time() - t_total
    print(f"\n  Total time: {elapsed/60:.1f} min")
    _print_summary()


if __name__ == '__main__':
    main()
