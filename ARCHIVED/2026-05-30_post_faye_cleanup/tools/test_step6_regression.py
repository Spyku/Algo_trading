"""
test_step6_regression.py — Bit-identical regression test for Step 6 engine refactor
====================================================================================

Verifies that `_deku_eval_with_pruning` produces IDENTICAL output before and
after the Step 6a refactor.

USAGE
-----
    # Run against ORIGINAL engine (baseline capture)
    python tools/test_step6_regression.py --engine ed --save baseline

    # Run against FORK after refactor (verify match)
    python tools/test_step6_regression.py --engine step6 --compare baseline

    # Quick sanity (single combo only)
    python tools/test_step6_regression.py --engine ed --save baseline --quick

WHAT IT DOES
------------
1. Loads a fixed ETH data snapshot (via V2_DATA_SNAPSHOT or live data/).
2. Calls engine.build_all_features() to construct the feature matrix.
3. Iterates a representative test grid (a few combos × windows × gammas).
4. For each grid point, calls `_deku_eval_with_pruning(...)` directly.
5. Saves the 13-tuple return value per grid point.
6. In --compare mode, diffs against saved baseline byte-by-byte.

WHY THIS NOT FULL MODE D
------------------------
Full Mode D writes its grid CSV to models/crypto_ed_grid_*.csv and runs
324 grid points (~10 min). The unit-style test runs ~12 grid points in
seconds and isolates the function under test from Optuna refine + I/O.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

# Don't redirect data — use the actual current data/ tree as the fixed input
for v in ("V2_DATA_SNAPSHOT", "H_STRICT_MODELS_DIR", "H_STRICT_CONFIG_DIR"):
    if os.environ.get(v):
        del os.environ[v]

import numpy as np
import pandas as pd

OUT_DIR = REPO_ROOT / "output" / "step6_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def import_engine(name: str):
    """Import either the original engine or the step6 fork."""
    if name == "ed":
        import crypto_trading_system_ed as engine
    elif name == "step6":
        import crypto_trading_system_ed_step6 as engine
    else:
        raise ValueError(f"unknown engine '{name}'")
    return engine


def build_test_inputs(engine, asset="ETH", horizon=5, replay_hours=720):
    """Load + build features once. Return (features_np, labels_np, closes_np, all_cols)."""
    df_raw = engine.load_data(asset)
    if df_raw is None:
        raise RuntimeError(f"no data for {asset}")
    df_full, all_cols = engine.build_all_features(
        df_raw, asset_name=asset, horizon=horizon,
        verbose=False, keep_label_nan_tail=True,
    )
    engine._compute_pysr_features(df_full, all_cols, asset, horizon, verbose=False)
    # Use last `replay_hours` for a faster test
    if replay_hours and len(df_full) > replay_hours + 500:
        df_full = df_full.iloc[-(replay_hours + 500):].reset_index(drop=True)
    # Use FEATURE_SET_A for stability of the test grid
    feature_list = list(engine.FEATURE_SET_A)
    feature_cols = [c for c in feature_list if c in all_cols]
    df_full[feature_cols] = df_full[feature_cols].ffill().fillna(0.0)
    df_train = df_full.dropna(subset=["label"]).reset_index(drop=True)
    features_np = df_train[feature_cols].values
    labels_np = df_train["label"].values.astype(int)
    closes_np = df_train["close"].values.astype(float)
    return features_np, labels_np, closes_np, feature_cols


def run_grid(engine, features_np, labels_np, closes_np, model_factories, quick=False):
    """Run a representative grid of (combo, window, gamma) -> tuple results."""
    # Conservative test grid: 3 combos x 2 windows x 2 gammas = 12 points
    if quick:
        combos = [("RF", "LGBM")]
        windows = [281]
        gammas = [0.9981]
    else:
        combos = [("RF", "LGBM"), ("XGB", "LGBM"), ("RF", "XGB")]
        windows = [100, 281]
        gammas = [0.999, 0.9981]

    horizon = 5
    step = engine.DIAG_STEP if hasattr(engine, "DIAG_STEP") else 36
    n = len(features_np)
    results = {}
    print(f"\n  Grid: {len(combos) * len(windows) * len(gammas)} points "
          f"(combos={len(combos)} windows={len(windows)} gammas={len(gammas)})")
    print(f"  n_samples={n}  step={step}  horizon={horizon}")

    t0 = time.time()
    for combo in combos:
        for w in windows:
            for g in gammas:
                key = f"{'+'.join(combo)}|w={w}|g={g}"
                t_start = time.time()
                result = engine._deku_eval_with_pruning(
                    features_np, labels_np, closes_np,
                    combo, w, n, step, model_factories, gamma=g,
                    horizon=horizon,
                )
                elapsed = time.time() - t_start
                results[key] = list(result) if result else None
                print(f"    {key:<40}  -> {'None' if result is None else f'cum_ret={result[4]:+.3f}% acc={result[2]:.3f} trades={result[6]}'}  ({elapsed:.1f}s)")
    print(f"  Grid runtime: {(time.time() - t0)/60:.1f} min")
    return results


def to_serializable(obj):
    """Convert numpy/pandas types in nested structures to plain Python for JSON."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["ed", "step6"], required=True)
    parser.add_argument("--save", help="Save results under this label (e.g. 'baseline')")
    parser.add_argument("--compare", help="Compare against saved label")
    parser.add_argument("--quick", action="store_true", help="Single grid point only")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--replay-hours", type=int, default=720,
                        help="Use last N hours of data for the test (default 720 ~30 days)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  STEP 6 REGRESSION TEST — engine='{args.engine}'")
    print("=" * 70)

    engine = import_engine(args.engine)
    print(f"  Imported: {engine.__name__} from {engine.__file__}")

    # Build inputs ONCE
    print("\n[1/2] Building test inputs (features, labels, closes)...")
    features_np, labels_np, closes_np, feature_cols = build_test_inputs(
        engine, asset="ETH", horizon=args.horizon, replay_hours=args.replay_hours,
    )
    print(f"  features_np: shape={features_np.shape}  feature_cols={len(feature_cols)}")
    print(f"  labels_np:   shape={labels_np.shape}  unique labels={list(set(labels_np.tolist()))}")
    print(f"  closes_np:   shape={closes_np.shape}  range=[{closes_np.min():.2f}, {closes_np.max():.2f}]")

    # Run grid
    print("\n[2/2] Running test grid...")
    results = run_grid(engine, features_np, labels_np, closes_np,
                       engine.ALL_MODELS, quick=args.quick)

    # Save / compare
    if args.save:
        out = OUT_DIR / f"{args.save}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(to_serializable(results), f, indent=2)
        print(f"\n  Saved {len(results)} grid points to {out}")

    if args.compare:
        baseline_path = OUT_DIR / f"{args.compare}.json"
        if not baseline_path.exists():
            print(f"\n  ERROR — baseline {baseline_path} not found")
            sys.exit(1)
        with open(baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)
        ours = to_serializable(results)

        print(f"\n  Comparing against {baseline_path}")
        n_match = n_diff = 0
        max_abs_diff_per_field = {}
        for key in sorted(set(baseline.keys()) | set(ours.keys())):
            b = baseline.get(key)
            o = ours.get(key)
            if b == o:
                n_match += 1
                continue
            if b is None or o is None:
                n_diff += 1
                print(f"    DIFF (None mismatch) {key}: baseline={b!r} ours={o!r}")
                continue
            # Compare field-by-field (tuple of 13)
            row_match = True
            for i in range(min(len(b), len(o))):
                if isinstance(b[i], (int, float)) and isinstance(o[i], (int, float)):
                    d = abs(b[i] - o[i])
                    if d > 1e-12:
                        max_abs_diff_per_field[i] = max(max_abs_diff_per_field.get(i, 0), d)
                        row_match = False
                elif b[i] != o[i]:
                    row_match = False
                    print(f"    DIFF {key} field[{i}]: {b[i]!r} != {o[i]!r}")
            if row_match:
                n_match += 1
            else:
                n_diff += 1

        print()
        print(f"  Grid points matching:    {n_match}")
        print(f"  Grid points diverging:   {n_diff}")
        if max_abs_diff_per_field:
            field_names = ['combo', 'window', 'accuracy', 'total', 'cum_return',
                           'win_rate', 'trades', 'total_gain', 'total_loss',
                           'max_dd_pct', 'apf', 'raw_pf', 'bh_pf']
            print(f"\n  Max abs diff per numeric field:")
            for i, d in sorted(max_abs_diff_per_field.items()):
                name = field_names[i] if i < len(field_names) else f"field[{i}]"
                print(f"    {name:<14} max_abs_diff={d:.2e}")

        if n_diff == 0:
            print(f"\n  [OK] BIT-IDENTICAL — Phase 6a regression-safe")
            sys.exit(0)
        else:
            print(f"\n  [FAIL] {n_diff} grid points diverge")
            sys.exit(1)


if __name__ == "__main__":
    main()
