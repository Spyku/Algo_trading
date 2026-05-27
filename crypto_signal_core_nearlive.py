"""
crypto_signal_core_nearlive.py — TEST COPY of crypto_signal_core.py with `mean_last_10` NaN policy
====================================================================================================

Created 2026-05-27 evening as a non-production sandbox to validate a new
NaN-fill policy before considering it for the live shared core.

Differences from production `crypto_signal_core.py`:
  - New na_policy option: 'mean_last_10'
      For each NaN cell, replace with the mean of the 10 most-recent non-NaN
      values preceding it (column-wise). Falls back gracefully if fewer than
      10 values exist; uses 0.0 only when ZERO history is available.
  - Helper `_mean_last_10_fill(arr, context=None)` — column-wise imputer.
      `context` lets a 1-row test array reference its training history.

Existing na_policy options ('skip', 'ffill', 'zero') are preserved unchanged
so this file is a strict superset of the production core.

NOT IMPORTED BY:
  - the live trader (`crypto_revolut_ed_v2.py`)
  - the shadow path (`crypto_live_shadow.py`)
  - the production engine (`crypto_trading_system_ed.py`)
Only `crypto_trading_system_ed_step6_nearlive.py` (companion test fork) imports
this file. Touching this file CANNOT affect live trading.

PROMOTION PATH (if mean_last_10 proves useful):
  1. Run HRST with `NEAR_LIVE_MODE=1` using `_step6_nearlive` fork
  2. Compare Mode T REF to current live REF and to plain live-equivalent REF
  3. If mean_last_10 produces materially better realism, port the policy back
     into `crypto_signal_core.py` as an additional option (still opt-in)
  4. Eventually promote to the default for backtest if validated
"""

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Helpers — replicated locally to keep this file self-contained for unit tests.
# Eventually these can be re-imported from crypto_trading_system_ed once the
# circular-dependency direction is decided (probably: core has none; engine
# and live trader import core).
# ============================================================================
def get_decay_weights_local(n_samples: int, gamma: Optional[float]) -> Optional[np.ndarray]:
    """Exponential sample-weighting decay.
    Newest sample weight = 1.0, oldest = gamma**(n-1).
    Returns None when gamma is None or >= 1.0 (zero-overhead no-decay case).
    """
    if gamma is None or gamma >= 1.0:
        return None
    ages = np.arange(n_samples - 1, -1, -1)
    return gamma ** ages


# ============================================================================
# The shared core
# ============================================================================
def compute_signal_core(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    model_factories: Dict[str, Callable],
    gamma: float = 1.0,
    na_policy: str = 'ffill',
    return_probas: bool = True,
    binary_signal: bool = False,
    hold_threshold: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Shared signal-generation core (TEST VARIANT with mean_last_10 policy).

    See module docstring for differences from production core.

    Parameters
    ----------
    na_policy : 'ffill' | 'skip' | 'zero' | 'mean_last_10' (default 'ffill')
        - 'skip'         : refuse to evaluate if any NaN in X_train or X_test (backtest legacy)
        - 'ffill'        : forward-fill then zero-fill (live trader convention)
        - 'zero'         : nan_to_num zero-fill (legacy)
        - 'mean_last_10' : NEW — for each NaN cell, replace with mean of the
                           10 most-recent non-NaN values preceding it (column-wise).
                           Falls back to fewer values if <10 history available.
                           Falls back to 0.0 only when ZERO history exists.

    All other parameters: see crypto_signal_core.compute_signal_core docstring.
    """
    # -------- NaN policy --------
    if na_policy == 'skip':
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            return {'skipped_reason': 'na_skip', 'signal': None}
    elif na_policy == 'ffill':
        X_train = _ffill_then_zero(X_train)
        X_test = _ffill_then_zero(X_test)
    elif na_policy == 'zero':
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
    elif na_policy == 'mean_last_10':
        # New policy: fill NaN with rolling mean of last 10 non-NaN preceding values.
        # X_test fills reference X_train as context (X_test is a single test row).
        X_train = _mean_last_10_fill(X_train, context=None)
        X_test = _mean_last_10_fill(X_test, context=X_train)
    else:
        raise ValueError(f"Unknown na_policy: {na_policy!r}. "
                         f"Use 'ffill', 'skip', 'zero', or 'mean_last_10'.")

    # -------- Sanity: need ≥2 classes in training labels --------
    if len(np.unique(y_train)) < 2:
        return {'skipped_reason': 'single_class', 'signal': None}

    # -------- Standardization (manual, matches backtest's exact math) --------
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    X_train_s = (X_train - mean) / std
    X_test_s = (X_test - mean) / std

    # -------- Decay weights --------
    sample_weight = get_decay_weights_local(len(y_train), gamma)

    # -------- Model fitting + prediction --------
    votes: List[int] = []
    probas: List[float] = []
    fit_errors: List[str] = []

    for model_name, factory in model_factories.items():
        try:
            model = factory()
            if sample_weight is not None:
                model.fit(X_train_s, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train_s, y_train)
            pred = int(model.predict(X_test_s)[0])
            votes.append(pred)
            if return_probas:
                proba = float(model.predict_proba(X_test_s)[0][1])
                probas.append(proba)
        except Exception as e:
            fit_errors.append(f"{model_name}: {e!r}")
            continue

    if not votes:
        return {'skipped_reason': f'all_models_failed: {fit_errors}', 'signal': None}

    # -------- Aggregation --------
    buy_ratio = sum(votes) / len(votes)
    ensemble_pred = 1 if buy_ratio > 0.5 else 0
    avg_proba = float(np.mean(probas)) if probas else None

    # -------- Signal semantics --------
    if binary_signal:
        signal = 'BUY' if ensemble_pred == 1 else 'SELL'
    else:
        if buy_ratio > 0.5 + hold_threshold:
            signal = 'BUY'
        elif buy_ratio == 0:
            signal = 'SELL'
        else:
            signal = 'HOLD'

    # -------- Confidence --------
    if avg_proba is None:
        confidence = round(buy_ratio * 100, 2) if signal != 'SELL' else round((1 - buy_ratio) * 100, 2)
    else:
        confidence = round(avg_proba * 100, 2) if signal != 'SELL' else round((1 - avg_proba) * 100, 2)

    return {
        'votes': votes,
        'probas': probas if return_probas else None,
        'buy_ratio': buy_ratio,
        'avg_proba': avg_proba,
        'signal': signal,
        'confidence': confidence,
        'ensemble_pred': ensemble_pred,
        'n_train': len(y_train),
        'scaler_mean': mean,
        'scaler_std': std,
        'skipped_reason': None,
    }


# ============================================================================
# Helper: forward-fill then zero-fill a numpy 2D array (column-wise)
# ============================================================================
def _ffill_then_zero(arr: np.ndarray) -> np.ndarray:
    """Column-wise forward-fill then zero-fill of NaN values."""
    if not np.isnan(arr).any():
        return arr
    out = arr.copy().astype(float)
    n_rows, n_cols = out.shape
    for col in range(n_cols):
        last_valid = None
        for row in range(n_rows):
            v = out[row, col]
            if np.isnan(v):
                if last_valid is not None:
                    out[row, col] = last_valid
                else:
                    out[row, col] = 0.0
            else:
                last_valid = v
    return out


# ============================================================================
# Helper: mean-of-last-10 fill (NEW)
# ============================================================================
def _mean_last_10_fill(arr: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
    """Column-wise: for each NaN cell, replace with the mean of the 10
    most-recent non-NaN values preceding it.

    Parameters
    ----------
    arr : 2D numpy array (n_rows, n_cols)
        Array to impute. NaN cells in this array will be filled.
    context : Optional 2D numpy array (n_context_rows, n_cols)
        If provided, non-NaN values from `context` are used as a SEED history
        for filling NaN in early rows of `arr`. Useful when `arr` is a single
        test row that should reference its training history.

    Semantics
    ---------
    For each column independently:
      1. Build a running list of observed (non-NaN) values, seeded from `context`.
      2. Walk `arr` row by row. If a cell is NaN, fill it with the mean of the
         last min(10, len(history)) values in the running list. If history is
         empty, fill with 0.0.
      3. Imputed (filled) values are NOT added back to the history — keeps the
         imputation grounded in actually-observed data only.

    Edge cases
    ----------
    - No NaN in arr: returns arr unchanged (zero copy if possible).
    - All NaN in a column with empty history: fills entire column with 0.0.
    - context has NaN: those cells are silently skipped (only non-NaN values
      seed history).
    """
    if not np.isnan(arr).any():
        return arr
    out = arr.copy().astype(float)
    n_rows, n_cols = out.shape
    for col in range(n_cols):
        # Seed history from context (if provided)
        if context is not None and context.size > 0:
            ctx_col = context[:, col]
            history: List[float] = ctx_col[~np.isnan(ctx_col)].tolist()
        else:
            history = []
        for row in range(n_rows):
            v = out[row, col]
            if np.isnan(v):
                if history:
                    last_n = history[-10:]
                    out[row, col] = float(np.mean(last_n))
                else:
                    out[row, col] = 0.0
                # Imputed value NOT appended to history (keeps bound to observed data)
            else:
                history.append(float(v))
    return out


# ============================================================================
# Self-test / smoke test
# ============================================================================
def _smoke_test():
    """Smoke test. Run with: python crypto_signal_core_nearlive.py"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 5)
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = rng.randn(1, 5)

    factories = {
        'LR': lambda: LogisticRegression(max_iter=200, random_state=42),
        'RF': lambda: RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42),
    }

    print("=== Smoke test 1: clean data, ternary signal ===")
    r = compute_signal_core(X_train, y_train, X_test, factories, gamma=0.998,
                            na_policy='ffill', return_probas=True, binary_signal=False)
    print(f"  signal={r['signal']} conf={r['confidence']} votes={r['votes']}")
    assert r['signal'] in ('BUY', 'SELL', 'HOLD')

    print("=== Smoke test 2: same data, binary signal ===")
    r = compute_signal_core(X_train, y_train, X_test, factories, gamma=0.998,
                            na_policy='ffill', return_probas=False, binary_signal=True)
    print(f"  signal={r['signal']} conf={r['confidence']} ensemble_pred={r['ensemble_pred']}")
    assert r['signal'] in ('BUY', 'SELL')

    print("=== Smoke test 3: NaN tail with ffill policy ===")
    X_train_nan = X_train.copy()
    X_train_nan[-5:, 0] = np.nan
    r = compute_signal_core(X_train_nan, y_train, X_test, factories, gamma=0.998,
                            na_policy='ffill', return_probas=True)
    assert r['skipped_reason'] is None
    print(f"  ffill handled NaN: signal={r['signal']}")

    print("=== Smoke test 4: NaN tail with skip policy ===")
    r = compute_signal_core(X_train_nan, y_train, X_test, factories, gamma=0.998,
                            na_policy='skip', return_probas=True)
    assert r['signal'] is None and r['skipped_reason'] == 'na_skip'
    print(f"  skip refused NaN as expected: skipped_reason={r['skipped_reason']}")

    print("=== Smoke test 5: single-class labels ===")
    y_train_one = np.zeros(len(y_train), dtype=int)
    r = compute_signal_core(X_train, y_train_one, X_test, factories, gamma=1.0,
                            na_policy='ffill', return_probas=True)
    assert r['signal'] is None and r['skipped_reason'] == 'single_class'
    print(f"  single-class refused: skipped_reason={r['skipped_reason']}")

    print("=== Smoke test 6 (NEW): mean_last_10 policy on scattered NaN ===")
    X_train_scattered = X_train.copy()
    # Insert NaN in random scattered cells
    rng2 = np.random.RandomState(7)
    nan_mask = rng2.rand(*X_train_scattered.shape) < 0.05  # ~5% NaN
    X_train_scattered[nan_mask] = np.nan
    n_nans_before = int(np.isnan(X_train_scattered).sum())
    r = compute_signal_core(X_train_scattered, y_train, X_test, factories, gamma=0.998,
                            na_policy='mean_last_10', return_probas=True)
    assert r['skipped_reason'] is None
    print(f"  mean_last_10 handled {n_nans_before} scattered NaNs: signal={r['signal']}")

    print("=== Smoke test 7 (NEW): mean_last_10 helper correctness ===")
    # Test the helper directly with a known sequence
    test_arr = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [np.nan, 8.0],       # row 3 col 0 NaN: mean of last 3 (1,2,3) = 2.0
        [5.0, np.nan],       # row 4 col 1 NaN: mean of last 4 (2,4,6,8) = 5.0
    ])
    filled = _mean_last_10_fill(test_arr)
    assert abs(filled[3, 0] - 2.0) < 1e-9, f"expected 2.0, got {filled[3, 0]}"
    assert abs(filled[4, 1] - 5.0) < 1e-9, f"expected 5.0, got {filled[4, 1]}"
    print(f"  helper: filled[3,0]={filled[3,0]:.3f} (expected 2.0), filled[4,1]={filled[4,1]:.3f} (expected 5.0)")

    print("=== Smoke test 8 (NEW): mean_last_10 with context (test row references training) ===")
    train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    test = np.array([[np.nan, np.nan]])  # both columns NaN in test row
    filled_test = _mean_last_10_fill(test, context=train)
    # Both columns: mean of 1+2+3=6/3=2.0, mean of 10+20+30=60/3=20.0
    assert abs(filled_test[0, 0] - 2.0) < 1e-9
    assert abs(filled_test[0, 1] - 20.0) < 1e-9
    print(f"  with context: filled_test={filled_test.tolist()} (expected [[2.0, 20.0]])")

    print("\nAll smoke tests passed [OK]")


if __name__ == '__main__':
    _smoke_test()
