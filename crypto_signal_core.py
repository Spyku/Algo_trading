"""
crypto_signal_core.py — Shared signal-generation core for LIVE TRADER and BACKTEST
====================================================================================

Created 2026-05-26 23:30 for TODO 0526 verdict.

WHY THIS FILE EXISTS
--------------------
The fast diagnostic battery (output/diagnostics_20260526_230310_fast/) confirmed
that the live trader's `generate_live_signal()` and the backtest's
`_deku_eval_with_pruning()` are SEMANTICALLY DIFFERENT ALGORITHMS, not the
same function called from two places. Code similarity = 0.082-0.019.

That's why backtest WR 85% does not predict live WR 50%: they're computing
two different things on the same data.

This file extracts the SHARED MATH (train + standardize + predict + aggregate)
into ONE function that both callers will eventually use. Differences that are
LEGITIMATELY DIFFERENT between live and backtest (embargo, NaN policy, signal
semantics) become explicit PARAMETERS — visible at call site, not hidden in
two separate implementations.

DESIGN PRINCIPLE
----------------
"Caller decides POLICY (what data, what embargo, what NaN strategy).
 Core decides MATH (how to standardize, fit, aggregate, score)."

USAGE — Live trader will call:
    result = compute_signal_core(
        X_train=df_train.iloc[train_start:][feature_cols].values,  # caller picks rows
        y_train=df_train.iloc[train_start:]['label'].values,
        X_test=df.iloc[i:i+1][feature_cols].values,                # caller picks test row
        model_names=['RF', 'LGBM'],
        gamma=0.998,
        na_policy='ffill',                                          # live convention
        return_probas=True,                                         # live needs confidence
    )

USAGE — Backtest will call:
    result = compute_signal_core(
        X_train=features_np[train_start:i-horizon],                 # caller applies embargo
        y_train=labels_np[train_start:i-horizon],
        X_test=features_np[i:i+1],
        model_names=combo,
        gamma=gamma,
        na_policy='skip',                                            # backtest convention
        return_probas=False,                                         # backtest uses binary
    )

MIGRATION PLAN (do NOT touch live trader or engine yet — only this file is new)
-----------------------------------------------------------------------------
1. (DONE)   Write this file with shared core
2. (NEXT)   Add `--shadow-mode` flag to live trader:
            - Runs OLD generate_live_signal() AND new compute_signal_core() side by side
            - Logs (timestamp, asset, horizon, old_signal, new_signal, old_conf, new_conf,
                    feat_diff_summary) to config/shadow_signal_diff.csv
            - Trades on OLD path so live behavior is unchanged
3. (NEXT)   Wait 7-14 days. If shadow_signal_diff.csv shows <5% mismatch on
            real data with matched parameters, the core is validated.
4. (NEXT)   Switch live trader to call compute_signal_core() directly. Remove
            inline math from generate_live_signal() — keep only the data prep,
            staleness checks, and result formatting.
5. (NEXT)   Refactor backtest's _deku_eval_with_pruning to also call
            compute_signal_core(). Walk-forward loop stays in caller; only the
            inner train+predict+aggregate step is replaced.
6. (NEXT)   Re-run HRST on the refactored engine. Expected: lower headline
            numbers than before, but predictive of live performance.

REMEMBER (per CLAUDE.md rule #1)
--------------------------------
NEVER modify crypto_trading_system_ed.py without testing first. This file is
SAFE because nothing imports it yet. Live trader and engine are untouched.
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

    This is a verbatim copy of crypto_trading_system_ed.get_decay_weights to
    decouple the unit test of this file from the engine. Both should remain
    identical; if they diverge, the migration step #4-5 will surface it.
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
    """Shared signal-generation core.

    Trains each model in `model_factories` on (X_train, y_train) with optional
    exponential sample decay (`gamma`), predicts on X_test, aggregates the
    ensemble vote, and returns a structured result.

    All POLICY decisions live in parameters — the math is the same for live
    and backtest.

    Parameters
    ----------
    X_train, y_train : numpy arrays (n_samples × n_features) and (n_samples,)
        Training data. CALLER is responsible for applying embargo (i.e. backtest
        passes [train_start, i-horizon]; live passes [train_start, n-1]).
    X_test : numpy array (1, n_features)
        Single row to predict on.
    model_factories : dict of name -> callable returning a sklearn-style model
        e.g. {'RF': lambda: RandomForestClassifier(...), 'LGBM': lambda: LGBMClassifier(...)}.
        Each callable must return a fresh instance (so different seeds don't share state).
    gamma : float (default 1.0)
        Decay factor for sample weights. 1.0 = no decay. <1.0 = older samples downweighted.
    na_policy : 'ffill' | 'skip' | 'zero' (default 'ffill')
        Forward-fill + zero-fill NaN (live convention) vs skip evaluation if any NaN
        (backtest convention) vs replace NaN with zero (legacy).
    return_probas : bool (default True)
        Whether to compute and return ensemble probabilities. Live needs True for
        confidence reporting. Backtest can use False to save predict_proba calls.
    binary_signal : bool (default False)
        False = ternary BUY/SELL/HOLD (live convention).
        True = binary BUY/not-BUY = backtest's `ensemble_pred = 1 if buy_ratio > 0.5 else 0`.
    hold_threshold : float (default 0.0)
        Range around 0.5 for HOLD verdict (ternary mode). 0.0 means HOLD only when
        buy_ratio is strictly 0 (all-SELL) — matches live trader. Set higher (e.g. 0.1)
        to widen the HOLD band.

    Returns
    -------
    dict with keys:
        votes        : list of int (0 or 1) per model
        probas       : list of float (P(class=1)) per model, OR None if return_probas=False
        buy_ratio    : float
        avg_proba    : float or None
        signal       : 'BUY' / 'SELL' / 'HOLD' (or just 'BUY' / 'SELL' if binary_signal=True)
        confidence   : float in [0, 100]
        ensemble_pred: int 0 or 1 (binary version, always computed for compatibility)
        n_train      : int (rows actually used after na_policy)
        scaler_mean  : np.ndarray (for reproducibility / debug)
        scaler_std   : np.ndarray
        skipped_reason : str or None — if the function bailed early (e.g. 'all_nan', 'single_class')

    OR None if the function decided not to produce a signal (caller can interpret
    this as "skip this evaluation step" in backtest, or "refuse to trade" in live).
    """
    # -------- NaN policy --------
    if na_policy == 'skip':
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            return {'skipped_reason': 'na_skip', 'signal': None}
    elif na_policy == 'ffill':
        # Forward-fill ROW-wise (carry last known value forward through NaN tails).
        # This is the live trader's convention. Applied to BOTH train and test.
        X_train = _ffill_then_zero(X_train)
        X_test = _ffill_then_zero(X_test)
    elif na_policy == 'zero':
        # Legacy: just zero-fill (deprecated; produces wrong signals for log-return features).
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
    else:
        raise ValueError(f"Unknown na_policy: {na_policy!r}. Use 'ffill', 'skip', or 'zero'.")

    # -------- Sanity: need ≥2 classes in training labels --------
    if len(np.unique(y_train)) < 2:
        return {'skipped_reason': 'single_class', 'signal': None}

    # -------- Standardization (manual, matches backtest's exact math) --------
    # Using numpy directly (not sklearn) so live and backtest get IDENTICAL numbers.
    # sklearn StandardScaler subtleties around ddof / with_mean / with_std caused
    # tiny numerical drift in the old code path. Replicate the engine's existing
    # manual standardization here.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # avoid divide-by-zero for constant cols
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
        # Backtest convention — no HOLD
        signal = 'BUY' if ensemble_pred == 1 else 'SELL'
    else:
        # Live trader convention — ternary
        if buy_ratio > 0.5 + hold_threshold:
            signal = 'BUY'
        elif buy_ratio == 0:
            signal = 'SELL'
        else:
            signal = 'HOLD'

    # -------- Confidence (only meaningful if return_probas=True) --------
    if avg_proba is None:
        # No probas: use buy_ratio as proxy
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
    """Column-wise forward-fill then zero-fill of NaN values.
    Matches pandas .ffill().fillna(0.0) but stays in numpy for speed and to
    keep this file pandas-optional.
    """
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
# Self-test / smoke test
# ============================================================================
def _smoke_test():
    """Tiny smoke test using sklearn's DummyClassifier. Confirms the core runs
    end-to-end without engine dependencies. Run with: python crypto_signal_core.py
    """
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
    print(f"  signal={r['signal']} conf={r['confidence']} votes={r['votes']} probas={[round(p,3) for p in r['probas']]}")
    assert r['signal'] in ('BUY', 'SELL', 'HOLD')

    print("=== Smoke test 2: same data, binary signal ===")
    r = compute_signal_core(X_train, y_train, X_test, factories, gamma=0.998,
                            na_policy='ffill', return_probas=False, binary_signal=True)
    print(f"  signal={r['signal']} conf={r['confidence']} ensemble_pred={r['ensemble_pred']}")
    assert r['signal'] in ('BUY', 'SELL')

    print("=== Smoke test 3: NaN tail with ffill policy ===")
    X_train_nan = X_train.copy()
    X_train_nan[-5:, 0] = np.nan  # last 5 rows have NaN in col 0
    r = compute_signal_core(X_train_nan, y_train, X_test, factories, gamma=0.998,
                            na_policy='ffill', return_probas=True)
    assert r['skipped_reason'] is None
    print(f"  ffill handled NaN: signal={r['signal']} conf={r['confidence']}")

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

    print("\nAll smoke tests passed [OK]")


if __name__ == '__main__':
    _smoke_test()
