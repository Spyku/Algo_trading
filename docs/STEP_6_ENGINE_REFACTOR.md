# Step 6 — Backtest Engine Refactor Design

**Created**: 2026-05-27 ~01:00 (sketch for tomorrow's work)
**Status**: 📋 DESIGN — no code changes yet
**Prerequisite**: ✅ `compute_signal_core()` validated 2026-05-27 00:51 via same-process test (100% match with live trader)

## Goal

Make backtest engine functions call `compute_signal_core()` so that:
- Backtest uses the **same algorithm** as live trader
- The 4 semantic differences (embargo, NaN policy, step size, signal mode) become **explicit parameters** at call site
- Backtest results in "live-equivalent mode" predict live performance
- Backtest results in "legacy mode" remain bit-identical to current engine (for back-compat with prior HRST records)

## Success criteria

| After phase | Test | Expected |
|---|---|---|
| 6a | Run Mode D on May 22 snapshot before vs after refactor | Bit-identical output CSV |
| 6b | Same as 6a — no semantic change | Bit-identical |
| 6c | Run Mode T REF in legacy mode vs live-equivalent mode | Numbers differ; document the gap |
| 6d | Re-run with shadow data validation | Live-equivalent Mode T REF ≈ actual live PnL over same window |

## Functions to refactor

### Primary — `_deku_eval_with_pruning` ([crypto_trading_system_ed.py:4057](crypto_trading_system_ed.py#L4057))

**Current behavior** (151 lines):
- Walk-forward loop with `step=DIAG_STEP=36`
- `train_end = i - horizon` (embargo = horizon)
- Skips rows with any NaN
- Binary signal (`ensemble_pred = 1 if buy_ratio > 0.5 else 0`)
- Inline portfolio simulation with TRADING_FEE / BACKTEST_FEE_PER_LEG
- Returns aggregate metrics: portfolio, accuracy, win_rate, APF

**Refactor approach**:
- Keep the walk-forward loop structure and portfolio sim (these are separate concerns from signal generation)
- **Replace** the inline train + standardize + predict + aggregate block (~30 lines) with a call to `compute_signal_core()`
- Expose embargo / na_policy / signal_mode / step / return_probas as parameters with defaults preserving current behavior

### Secondary — `_h_deku_eval_median_k` ([crypto_trading_system_ed.py:8898](crypto_trading_system_ed.py#L8898))

K=5 wrapper around `_deku_eval_with_pruning`. Calls 5 times with different seeds, takes median by cum_return.

**Refactor**: NONE needed at this layer — it wraps `_deku_eval_with_pruning`. When the inner function calls `compute_signal_core()`, the K=5 averaging still works at the outer layer.

### Tertiary — `generate_signals` ([crypto_trading_system_ed.py:2464](crypto_trading_system_ed.py#L2464))

Used to produce per-hour signal streams for Mode T cache. Currently has its own train+predict logic.

**Refactor approach**: Same pattern — replace inline math with `compute_signal_core()` call. Keep the loop and result formatting.

### Quaternary — `_simulate_with_threshold` (need to grep for exact line)

Used in Mode V Step 1 to backtest top-N D candidates. Generates signals + simulates trading with threshold gating.

**Refactor approach**: Signal generation portion replaced with `compute_signal_core()`. Threshold logic and PnL tracking stay inline.

## New parameters (added to `_deku_eval_with_pruning`)

All with defaults matching CURRENT behavior so Phase 6a is regression-safe:

```python
def _deku_eval_with_pruning(
    features_np, labels_np, closes_np, combo, window, n,
    step, model_factories, gamma=1.0, trial=None,
    horizon=PREDICTION_HORIZON,
    # NEW PARAMETERS — defaults preserve current behavior
    embargo=None,         # None = use horizon (current default)
    na_policy='skip',     # 'skip' (current) | 'ffill' (live-equivalent)
    signal_mode='binary', # 'binary' (current) | 'ternary' (live-equivalent)
    return_probas=False,  # False (current) | True (live-equivalent, slightly slower)
    eval_step=None,       # None = use `step` (i.e. DIAG_STEP=36); 1 = every hour
):
    if embargo is None:
        embargo = horizon
    step_to_use = eval_step if eval_step is not None else step
    # ... walk-forward loop using step_to_use, embargo
    #     each step calls compute_signal_core() with na_policy/return_probas/binary_signal
```

**Shortcut** — single flag to flip all four together:

```python
def _eval_live_equivalent(features_np, labels_np, closes_np, ...):
    """Call _deku_eval_with_pruning with all 4 policies set to live trader's
    conventions. Use for diagnostic HRST that should predict live performance."""
    return _deku_eval_with_pruning(
        ..., embargo=1, na_policy='ffill', signal_mode='ternary',
        return_probas=True, eval_step=1,
    )
```

## Migration phases

### Phase 6a — Internal refactor, zero behavior change (~4-6 hours)

1. Add `import crypto_signal_core as _csc` at top of engine
2. Replace the inline train + standardize + predict + aggregate block in `_deku_eval_with_pruning` with a call to `_csc.compute_signal_core(...)` using **embargo=horizon, na_policy='skip', signal_mode='binary', return_probas=False, eval_step=DIAG_STEP** — exactly preserving current behavior
3. The walk-forward loop, NaN-skip counter, portfolio sim, drawdown tracking, APF computation all stay as they are
4. **Regression test**: run `python crypto_trading_system_ed.py D ETH 5h --replay 1440 --no-persist --no-data-update` with `V2_DATA_SNAPSHOT` set to May 22 snapshot, BEFORE and AFTER the refactor. Compare `models/crypto_ed_grid_ETH_5h.csv` byte-for-byte. Must be identical.
5. If not identical, debug until it is. Most likely culprits: standardization ddof differences, float precision in mean/std, ordering of operations

### Phase 6b — Expose parameters with current defaults (~1 hour)

1. Add the 5 new parameters to `_deku_eval_with_pruning` signature with defaults preserving 6a behavior
2. Wire them through to the `_csc.compute_signal_core()` call
3. Add to `_h_deku_eval_median_k` signature (just pass-through)
4. **Regression test**: same as 6a — must remain bit-identical when called with default args

### Phase 6c — Add live-equivalent diagnostic mode (~1-2 hours)

1. Add `LIVE_EQUIVALENT_MODE` env var check at engine startup
2. When set to `1`: wrap `_deku_eval_with_pruning` to always use live-equivalent params
3. Print banner clearly when active
4. NO change to default behavior
5. **Test**: run Mode V on snapshot in both modes. Document the result delta in `docs/`.

### Phase 6d — Cross-validate against shadow data (~1 day calendar, 12h compute)

1. Once Desktop's shadow log has 24-48h of data, compute the live trader's actual signal stream
2. Run engine in `LIVE_EQUIVALENT_MODE=1` Mode T on same period
3. Compare signal-by-signal — should be near-100% match (modulo GPU non-determinism if Mode T uses different machine)
4. Run full HRST in live-equivalent mode
5. Mode T REF in live-equivalent mode should be SUBSTANTIALLY LOWER than current legacy mode (probably 30-50% instead of 91%)
6. This LOWER number is the **honest** expectation for live performance

## Code skeleton — the refactored `_deku_eval_with_pruning` body

```python
import crypto_signal_core as _csc

def _deku_eval_with_pruning(
    features_np, labels_np, closes_np, combo, window, n,
    step, model_factories, gamma=1.0, trial=None,
    horizon=PREDICTION_HORIZON,
    embargo=None, na_policy='skip', signal_mode='binary',
    return_probas=False, eval_step=None,
):
    if embargo is None:
        embargo = horizon
    if eval_step is None:
        eval_step = step

    min_start = window + 50
    if n < min_start + 50:
        return None

    # Portfolio + tracking state (unchanged)
    correct = 0; total = 0
    portfolio = 1.0
    in_position = False
    entry_price = 0
    trades = 0; wins = 0
    peak = 1.0; max_dd = 0.0
    total_gain = 0.0; total_loss = 0.0
    bh_gains = 0.0; bh_losses = 0.0
    prev_price = None
    step_idx = 0

    for i in range(min_start, n, eval_step):
        train_start = max(0, i - window)
        train_end = max(train_start, i - embargo)
        X_train = features_np[train_start:train_end]
        y_train = labels_np[train_start:train_end]
        X_test = features_np[i:i+1]
        y_true = labels_np[i]
        price = closes_np[i]

        # Buy-and-hold tracking (unchanged)
        if prev_price is not None:
            bh_ret_step = (price - prev_price) / prev_price
            if bh_ret_step > 0: bh_gains += bh_ret_step
            else: bh_losses += bh_ret_step
        prev_price = price

        # NEW: delegate signal generation to shared core
        result = _csc.compute_signal_core(
            X_train=X_train, y_train=y_train, X_test=X_test,
            model_factories=model_factories, gamma=gamma,
            na_policy=na_policy, return_probas=return_probas,
            binary_signal=(signal_mode == 'binary'),
        )

        if result is None or result.get('signal') is None:
            step_idx += 1
            continue

        ensemble_pred = result['ensemble_pred']

        if ensemble_pred == y_true:
            correct += 1
        total += 1

        # Portfolio logic (unchanged from current)
        if ensemble_pred == 1 and not in_position:
            in_position = True
            entry_price = price * (1 + TRADING_FEE)
        elif ensemble_pred == 0 and in_position:
            sell_price = price * (1 - BACKTEST_FEE_PER_LEG)
            trade_return = (sell_price - entry_price) / entry_price
            portfolio *= (1 + trade_return)
            trades += 1
            if trade_return > 0:
                wins += 1; total_gain += trade_return
            else:
                total_loss += trade_return
            in_position = False

        # Max DD tracking (unchanged)
        current_val = portfolio * (price / entry_price) if in_position else portfolio
        if current_val > peak: peak = current_val
        dd = (peak - current_val) / peak
        if dd > max_dd: max_dd = dd

        step_idx += 1

        # Hyperband pruning (unchanged)
        if trial is not None and step_idx >= DEKU_PRUNING_WARMUP:
            intermediate = (portfolio - 1.0) * 100
            trial.report(intermediate, step_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Close final position (unchanged)
    if in_position and total > 0:
        last_price = closes_np[n - 1]
        sell_price = last_price * (1 - BACKTEST_FEE_PER_LEG)
        trade_return = (sell_price - entry_price) / entry_price
        portfolio *= (1 + trade_return)
        trades += 1
        if trade_return > 0: wins += 1; total_gain += trade_return
        else: total_loss += trade_return

    if total == 0: return None

    # Result aggregation (unchanged)
    accuracy = correct / total
    cum_return = (portfolio - 1.0) * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0
    if trades < 3: raw_pf = 0.0
    elif total_loss == 0: raw_pf = min(total_gain * 100, 20.0)
    else: raw_pf = min(total_gain / abs(total_loss), 20.0)
    if bh_losses == 0: bh_pf = max(bh_gains * 100, 1.0)
    else: bh_pf = max(bh_gains / abs(bh_losses), 0.01)
    adjusted_pf = raw_pf / bh_pf if raw_pf > 0 else 0.0

    return ('+'.join(combo), window, accuracy, total, cum_return, win_rate,
            trades, total_gain, total_loss, max_dd * 100, adjusted_pf, raw_pf, bh_pf)
```

**Diff stats**:
- ~30 lines of inline math replaced with one `compute_signal_core()` call
- 5 new parameters added (all defaults preserve behavior)
- No change to portfolio logic, no change to return type

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Refactored function gives different results than current | Phase 6a regression test must pass before adding parameters in 6b |
| Float-precision drift in standardization (sklearn vs numpy) | Core already uses numpy (matches current engine) — verified |
| Slower because of function-call overhead | Negligible (one function call per hour, not per row) |
| Live-equivalent mode breaks during Mode V/T/G chain | Phase 6c only activates via opt-in env var; default unchanged |
| Some downstream code relies on binary signal semantics | Audit before 6c; portfolio logic in `_deku_eval_with_pruning` uses `ensemble_pred` (binary) regardless of `signal_mode` |
| Core's `compute_signal_core` doesn't accept `trial` for Optuna pruning | Pruning lives in the outer loop, not in core. Trial passing stays in engine. |

## Validation checklist before promotion

- [ ] Phase 6a: Mode D output CSV bit-identical vs current
- [ ] Phase 6b: Same with defaults
- [ ] Phase 6c: live-equivalent mode produces plausible (non-NaN, non-zero) results
- [ ] Phase 6d: live-equivalent Mode T REF within ±5pp of actual live PnL over same window
- [ ] No regression in HRST runtime (within ±10%)
- [ ] Shadow log on Desktop continues to show high match rate

## Things explicitly OUT OF SCOPE for Step 6

- Switching live trader to call core (that's Step 5 of the original plan, separate)
- Fixing the data drift (`keep='last'` → `keep='first'`) — separate concern, do anytime
- Changing model selection logic (Mode V refine ranking, etc.)
- Adding new horizons or features
- Performance optimization

## Estimated total effort

| Phase | Effort | Risk |
|---|---|---|
| 6a | 4-6h | Low (regression-safe by design) |
| 6b | 1h | Low |
| 6c | 1-2h | Low (opt-in only) |
| 6d | 1 day calendar (12h compute + 1h analysis) | Medium (need shadow data + HRST) |
| **Total** | **~1.5 calendar days + 12h compute** | |

## What gets unblocked after Step 6 completes

1. **Honest backtest numbers**: live-equivalent Mode T REF will be ~30-50% (vs legacy 91%). This is the real expected live performance.
2. **Trustworthy optimization**: HRST/Mode V/T in live-equivalent mode → if a new config improves the live-equivalent Mode T REF by +5pp, it'll actually improve live by ~+5pp.
3. **Decoupled backtest modes**: keep legacy mode for back-compat with historical HRST records; use live-equivalent for forward optimization.
4. **Step 7 ready**: can re-run HRST on refactored engine using live-equivalent mode to find configs that ACTUALLY beat current LIVE.

## Open questions for tomorrow

1. Should `LIVE_EQUIVALENT_MODE` be the new default eventually, or stay opt-in forever? (Recommend: opt-in for 30 days, then re-evaluate)
2. Do we also want a "PARTIAL_LIVE_EQUIVALENT" mode that flips e.g. only embargo (B3)? Useful for ablation studies.
3. Should `_simulate_with_threshold` get the same treatment in this step, or defer to Step 6.5?
4. What's the right place for the `LIVE_EQUIVALENT_MODE` banner — engine startup, or per-call?

## Recommended starting point tomorrow

Open `crypto_trading_system_ed.py` at line 4057. Read the current `_deku_eval_with_pruning` end-to-end (~5 min). Run the regression test once on current engine to capture baseline output:

```powershell
$env:V2_DATA_SNAPSHOT = "data\_reliability_hrst_snapshot_laptop_20260522_0139"
python crypto_trading_system_ed.py D ETH 5h --replay 1440 --no-persist --no-data-update
copy models\crypto_ed_grid_ETH_5h.csv output\baseline_grid_ETH_5h.csv
```

Then apply Phase 6a refactor, run again, and compare:
```powershell
python crypto_trading_system_ed.py D ETH 5h --replay 1440 --no-persist --no-data-update
fc /b output\baseline_grid_ETH_5h.csv models\crypto_ed_grid_ETH_5h.csv
```

If `fc /b` reports "no differences", Phase 6a is good — proceed to 6b.
If differences appear, diff the values column by column to find which math diverged.
