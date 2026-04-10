"""
PySR Feature Discovery — Offline symbolic regression for Doohan
================================================================
Discovers compact mathematical formulas from HISTORICAL data (months 12→6 ago)
that capture nonlinear feature interactions. Uses a data window that does NOT
overlap with Mode D's last-6-month evaluation window, preventing leakage.

Usage:
  pip install pysr sympy          # first time only (auto-installs Julia)
  python pysr_discover_features.py BTC 6h
  python pysr_discover_features.py BTC 6h --top 5
  python pysr_discover_features.py BTC 6h --iterations 100

Output:
  models/pysr_BTC_6h.json         — discovered expressions + metadata
  models/pysr_BTC_6h_report.txt   — human-readable summary

Runtime: ~30-120 min depending on iterations and hardware.

Anti-leakage design:
  Mode D uses the LAST 6 months (4320h) for grid evaluation.
  PySR uses the PREVIOUS 6 months (months 12→6 ago) for formula discovery.
  Zero data overlap → PySR coefficients cannot inflate Mode D backtest results.
"""

import sys
import os
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Add engine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_trading_system_ed import (
    load_data, build_all_features, PREDICTION_HORIZON
)

MAX_DIAG_HOURS = 6 * 30 * 24  # 4320 hours = 6 months (same cap as Mode D)


def discover_features(asset, horizon, n_top=5, iterations=100, populations=30,
                      max_corr=0.7):
    """Run PySR symbolic regression to discover feature formulas.

    Uses the 6 months BEFORE Mode D's window to prevent data leakage.

    Args:
        asset: Asset name (e.g., 'BTC')
        horizon: Prediction horizon in hours
        n_top: Number of top expressions to save
        iterations: PySR iterations (more = better but slower)
        populations: Number of populations for genetic algorithm
        max_corr: Maximum pairwise correlation between kept expressions (0.7 = diverse)

    Returns:
        Tuple of (results_list, pysr_rows) where results_list contains dicts
        with 'equation', 'complexity', 'loss', 'score' keys
    """
    from pysr import PySRRegressor

    print(f"\n{'='*70}")
    print(f"  PySR FEATURE DISCOVERY: {asset} {horizon}h")
    print(f"  Iterations: {iterations} | Populations: {populations} | Top: {n_top}")
    print(f"{'='*70}")

    # Load and prepare data (same pipeline as Mode D)
    print(f"\n  Loading data for {asset}...")
    df_raw = load_data(asset)
    if df_raw is None:
        print(f"  ERROR: No data for {asset}")
        return [], 0

    df_full, all_cols = build_all_features(df_raw, asset_name=asset, horizon=horizon)

    # Exclude any existing pysr_* columns — discovery must use only base features
    pysr_cols = [c for c in all_cols if c.startswith('pysr_')]
    if pysr_cols:
        all_cols = [c for c in all_cols if not c.startswith('pysr_')]
        print(f"  Excluded {len(pysr_cols)} existing PySR columns from inputs")

    # Short-history features (GDELT: only ~3 months of data) would kill the historical
    # window via dropna. Fill NaN with 0 for these columns so PySR can still use the
    # full 12-month window — PySR will simply see 0 for periods before GDELT coverage.
    gdelt_cols = [c for c in all_cols if c.startswith('gp_')]
    if gdelt_cols:
        for gc in gdelt_cols:
            df_full[gc] = df_full[gc].fillna(0.0)
        print(f"  GDELT: {len(gdelt_cols)} columns filled NaN→0 (short history, ~3 months)")

    df_clean = df_full.dropna(subset=all_cols + ['label']).reset_index(drop=True)

    # Anti-leakage: use the 6 months BEFORE Mode D's window (months 12→6 ago)
    # Mode D uses tail(MAX_DIAG_HOURS) = last 6 months
    # PySR uses exactly the 6 months before that — no older data allowed
    total_needed = MAX_DIAG_HOURS * 2  # 12 months required
    if len(df_clean) < total_needed:
        print(f"  ERROR: Not enough data ({len(df_clean)} rows).")
        print(f"  Need at least {total_needed} rows (6 months for Mode D + 6 months for PySR).")
        return [], 0

    df_pysr = df_clean.iloc[-total_needed:-MAX_DIAG_HOURS].reset_index(drop=True)

    print(f"  Total data: {len(df_clean)} rows, {len(all_cols)} features")
    print(f"  PySR window: {len(df_pysr)} rows (historical, BEFORE Mode D's 6-month window)")
    print(f"  Mode D window: last {MAX_DIAG_HOURS} rows (excluded from PySR)")

    X = df_pysr[all_cols].values.astype(np.float32)
    y = df_pysr['label'].values.astype(np.float32)

    # Remove any remaining NaN/Inf
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"  Training set: {len(X)} rows (from {len(df_pysr)}-row historical window)")
    print(f"  Label distribution: {y.sum():.0f} BUY ({y.mean()*100:.1f}%) / {len(y)-y.sum():.0f} SELL")

    # Subsample if too large (PySR is O(n²) in some operations)
    max_samples = 3000
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        idx.sort()
        X = X[idx]
        y = y[idx]
        print(f"  Subsampled to {max_samples} rows for tractability")

    print(f"\n  Starting PySR (this may take 30-120 min)...")
    print(f"  First run will install Julia backend (~2 min one-time)...")
    t0 = time.time()

    model = PySRRegressor(
        niterations=iterations,
        populations=populations,
        population_size=50,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "abs", "sqrt", "tanh"],
        maxsize=25,               # max expression complexity (was 15)
        maxdepth=8,               # max tree depth (was 5)
        parsimony=0.002,          # penalty for complexity (lower = allows more complex formulas)
        ncycles_per_iteration=300,
        weight_optimize=0.001,    # constant optimization weight
        adaptive_parsimony_scaling=100.0,
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        random_state=42,
        deterministic=True,
        procs=0,                  # single process for reproducibility
        parallelism="serial",
        temp_equation_file=False,
        verbosity=1,
    )

    model.fit(X, y, variable_names=all_cols)
    elapsed = (time.time() - t0) / 60

    print(f"\n  PySR completed in {elapsed:.1f} min")

    # Extract best equations
    equations = model.equations_
    pysr_rows = len(X)

    if equations is None or len(equations) == 0:
        print("  No equations found!")
        return [], 0

    # Sort by score (higher = better accuracy-per-complexity)
    equations = equations.sort_values('score', ascending=False)

    results = []
    print(f"\n  {'='*70}")
    print(f"  TOP {min(n_top, len(equations))} DISCOVERED EXPRESSIONS")
    print(f"  {'='*70}")
    print(f"  {'#':>3} | {'Score':>8} | {'Loss':>10} | {'Complexity':>10} | Expression")
    print(f"  {'─'*80}")

    for i, (_, row) in enumerate(equations.head(n_top * 5).iterrows()):
        # Skip trivially simple expressions (just a single feature)
        if row['complexity'] <= 2:
            continue

        expr = str(row['equation'])
        result = {
            'equation': expr,
            'sympy_format': str(row['sympy_format']) if 'sympy_format' in row else expr,
            'complexity': int(row['complexity']),
            'loss': float(row['loss']),
            'score': float(row['score']),
            'lambda_format': str(row['lambda_format']) if 'lambda_format' in row else None,
        }
        results.append(result)
        print(f"  {len(results):>3} | {row['score']:>8.4f} | {row['loss']:>10.6f} | {row['complexity']:>10} | {expr}")

        # Collect extra candidates when dedup is active (many will be dropped)
        candidate_limit = n_top * 3 if max_corr < 1.0 else n_top
        if len(results) >= candidate_limit:
            break

    # Deduplicate: drop expressions too correlated with already-kept ones
    if len(results) > 1 and max_corr < 1.0:
        import sympy
        kept = [results[0]]
        kept_vals = []
        # Compute values for first expression
        try:
            sym0 = sympy.sympify(results[0].get('sympy_format', results[0]['equation']))
            sym_vars0 = list(sym0.free_symbols)
            func0 = sympy.lambdify(sym_vars0, sym0, modules=['numpy'])
            missing0 = [str(s) for s in sym_vars0 if str(s) not in all_cols]
            if not missing0:
                args0 = [X[:, all_cols.index(str(s))] for s in sym_vars0]
                v0 = func0(*args0)
                v0 = np.where(np.isfinite(v0), v0, 0.0)
                kept_vals.append(v0)
            else:
                kept_vals.append(None)
        except Exception:
            kept_vals.append(None)

        for r in results[1:]:
            try:
                sym_r = sympy.sympify(r.get('sympy_format', r['equation']))
                sym_vars_r = list(sym_r.free_symbols)
                func_r = sympy.lambdify(sym_vars_r, sym_r, modules=['numpy'])
                missing_r = [str(s) for s in sym_vars_r if str(s) not in all_cols]
                if missing_r:
                    kept.append(r)
                    kept_vals.append(None)
                    continue
                args_r = [X[:, all_cols.index(str(s))] for s in sym_vars_r]
                v_r = func_r(*args_r)
                v_r = np.where(np.isfinite(v_r), v_r, 0.0)
            except Exception:
                kept.append(r)
                kept_vals.append(None)
                continue

            # Check correlation against all kept expressions
            too_similar = False
            for kv in kept_vals:
                if kv is None:
                    continue
                if np.std(v_r) < 1e-10 or np.std(kv) < 1e-10:
                    continue
                corr = abs(np.corrcoef(v_r, kv)[0, 1])
                if corr > max_corr:
                    too_similar = True
                    break
            if too_similar:
                print(f"    SKIP (corr>{max_corr:.1f}): {r['equation'][:60]}")
            else:
                kept.append(r)
                kept_vals.append(v_r)

            if len(kept) >= n_top:
                break

        print(f"\n  After dedup: {len(kept)}/{len(results)} expressions kept (max_corr={max_corr})")
        results = kept

    print(f"\n  {len(results)} expressions saved")
    return results, pysr_rows


def save_results(asset, horizon, results, all_cols, pysr_rows=0):
    """Save discovered expressions to JSON and human-readable report."""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    json_path = os.path.join(models_dir, f'pysr_{asset}_{horizon}h.json')
    report_path = os.path.join(models_dir, f'pysr_{asset}_{horizon}h_report.txt')

    output = {
        'asset': asset,
        'horizon': horizon,
        'discovered_at': time.strftime('%Y-%m-%d %H:%M'),
        'discovery_method': 'historical',
        'pysr_data_rows': pysr_rows,
        'n_expressions': len(results),
        'feature_names': all_cols,
        'expressions': results,
    }

    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {json_path}")

    with open(report_path, 'w') as f:
        f.write(f"PySR Feature Discovery: {asset} {horizon}h\n")
        f.write(f"{'='*60}\n")
        f.write(f"Date: {output['discovered_at']}\n\n")
        for i, expr in enumerate(results, 1):
            f.write(f"#{i}: {expr['equation']}\n")
            f.write(f"    Score: {expr['score']:.4f}  Loss: {expr['loss']:.6f}  Complexity: {expr['complexity']}\n\n")
    print(f"  Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='PySR Feature Discovery for Doohan')
    parser.add_argument('asset', type=str, help='Asset name (e.g., BTC)')
    parser.add_argument('horizon', type=str, help='Horizon (e.g., 6h)')
    parser.add_argument('--top', type=int, default=5, help='Number of top expressions to save (default: 5)')
    parser.add_argument('--iterations', type=int, default=100, help='PySR iterations (default: 100, more=better but slower)')
    parser.add_argument('--populations', type=int, default=30, help='Number of populations (default: 30)')
    parser.add_argument('--max-corr', type=float, default=0.7, help='Max correlation between kept expressions (default: 0.7)')

    args = parser.parse_args()
    horizon = int(args.horizon.replace('h', ''))

    results, pysr_rows = discover_features(args.asset, horizon, n_top=args.top,
                                iterations=args.iterations, populations=args.populations,
                                max_corr=args.max_corr)

    if results:
        # Get feature names for saving
        df_raw = load_data(args.asset)
        _, all_cols = build_all_features(df_raw, asset_name=args.asset, horizon=horizon, verbose=False)
        save_results(args.asset, horizon, results, all_cols, pysr_rows=pysr_rows)
        print(f"\n  Done! Now run Mode DV to test these features:")
        print(f"  python crypto_trading_system_ed.py DV {args.asset} {horizon}h")
    else:
        print("\n  No useful expressions found. Try increasing --iterations.")


if __name__ == '__main__':
    main()
