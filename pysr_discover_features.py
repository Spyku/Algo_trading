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


def _prepare_data(asset, horizon):
    """Load data and prepare X, y, all_cols for PySR discovery.
    Returns (X, y, all_cols, pysr_rows) or (None, None, None, 0) on error."""

    print(f"\n  Loading data for {asset}...")
    df_raw = load_data(asset)
    if df_raw is None:
        print(f"  ERROR: No data for {asset}")
        return None, None, None, 0

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
    total_needed = MAX_DIAG_HOURS * 2  # 12 months required
    if len(df_clean) < total_needed:
        print(f"  ERROR: Not enough data ({len(df_clean)} rows).")
        print(f"  Need at least {total_needed} rows (6 months for Mode D + 6 months for PySR).")
        return None, None, None, 0

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

    return X, y, all_cols, len(X)


def _run_single_pysr(X, y, all_cols, feature_subset, seed, iterations, run_label):
    """Run a single PySR instance on a feature subset. Returns list of result dicts."""
    from pysr import PySRRegressor

    # Select feature subset columns
    col_indices = [all_cols.index(c) for c in feature_subset]
    X_sub = X[:, col_indices]

    print(f"\n  ── Run {run_label}: {len(feature_subset)} features, seed={seed} ──")
    t0 = time.time()

    model = PySRRegressor(
        niterations=iterations,
        populations=40,           # more islands for diversity
        population_size=100,      # larger populations
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "abs", "sqrt", "tanh"],
        maxsize=25,
        maxdepth=8,
        parsimony=0.002,
        ncycles_per_iteration=300,
        weight_optimize=0.001,
        adaptive_parsimony_scaling=100.0,
        fraction_replaced=0.0001,       # low migration → islands stay distinct
        fraction_replaced_hof=0.01,     # less hall-of-fame dominance
        tournament_selection_n=6,       # softer selection pressure
        tournament_selection_p=0.72,    # allow weaker novel expressions to survive
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        random_state=seed,
        deterministic=True,
        procs=0,
        parallelism="serial",
        temp_equation_file=False,
        verbosity=1,
    )

    model.fit(X_sub, y, variable_names=feature_subset)
    elapsed = (time.time() - t0) / 60
    print(f"  Run {run_label} completed in {elapsed:.1f} min")

    equations = model.equations_
    if equations is None or len(equations) == 0:
        return []

    equations = equations.sort_values('score', ascending=False)

    results = []
    for _, row in equations.head(10).iterrows():
        if row['complexity'] <= 2:
            continue
        results.append({
            'equation': str(row['equation']),
            'sympy_format': str(row['sympy_format']) if 'sympy_format' in row else str(row['equation']),
            'complexity': int(row['complexity']),
            'loss': float(row['loss']),
            'score': float(row['score']),
            'lambda_format': str(row['lambda_format']) if 'lambda_format' in row else None,
            'run': run_label,
        })

    print(f"  Run {run_label}: {len(results)} candidate expressions")
    return results


def _dedup_by_correlation(results, X, all_cols, n_top, max_corr):
    """Keep top n_top expressions with pairwise correlation < max_corr."""
    import sympy

    if len(results) <= 1:
        return results

    # Evaluate all expressions
    evaluated = []
    for r in results:
        try:
            sym_expr = sympy.sympify(r.get('sympy_format', r['equation']))
            sym_vars = list(sym_expr.free_symbols)
            missing = [str(s) for s in sym_vars if str(s) not in all_cols]
            if missing:
                evaluated.append((r, None))
                continue
            func = sympy.lambdify(sym_vars, sym_expr, modules=['numpy'])
            args = [X[:, all_cols.index(str(s))] for s in sym_vars]
            vals = func(*args)
            vals = np.where(np.isfinite(vals), vals, 0.0)
            evaluated.append((r, vals))
        except Exception:
            evaluated.append((r, None))

    # Sort by loss (lower = better fit)
    evaluated.sort(key=lambda x: x[0]['loss'])

    kept = []
    kept_vals = []
    for r, v in evaluated:
        if v is None:
            if len(kept) < n_top:
                kept.append(r)
                kept_vals.append(None)
            continue

        too_similar = False
        for kv in kept_vals:
            if kv is None:
                continue
            if np.std(v) < 1e-10 or np.std(kv) < 1e-10:
                continue
            corr = abs(np.corrcoef(v, kv)[0, 1])
            if corr > max_corr:
                too_similar = True
                break

        if too_similar:
            print(f"    SKIP (corr>{max_corr:.1f}): [{r.get('run','')}] {r['equation'][:60]}")
        else:
            kept.append(r)
            kept_vals.append(v)

        if len(kept) >= n_top:
            break

    return kept


# ── Feature group definitions for multi-run diversity ──
FEATURE_GROUPS = {
    'momentum': ['logret_1h', 'logret_2h', 'logret_3h', 'logret_4h', 'logret_5h',
                  'logret_6h', 'logret_7h', 'logret_8h', 'logret_12h', 'logret_24h',
                  'logret_48h', 'logret_72h', 'logret_120h', 'logret_240h',
                  'price_velocity_1h', 'price_velocity_4h', 'price_accel_1h',
                  'price_accel_4h', 'price_accel_12h', 'price_accel_24h', 'price_jerk_1h'],
    'volatility': ['volatility_12h', 'volatility_48h', 'vol_ratio_12_48', 'volume_ratio_h',
                   'vvr_12h', 'gk_volatility_14h', 'gk_volatility_48h', 'atr_pct_14h',
                   'intraday_range', 'adx_14h', 'plus_di_14h', 'minus_di_14h'],
    'mean_reversion': ['rsi_14h', 'stoch_k_14h', 'bb_position_20h', 'zscore_50h',
                       'price_to_sma20h', 'price_to_sma50h', 'price_to_sma100h',
                       'sma20_to_sma50h', 'spread_24h_4h', 'spread_48h_4h',
                       'spread_120h_8h', 'spread_240h_24h', 'spread_48h_12h', 'spread_120h_12h'],
    'macro': [],       # filled dynamically from m_* columns
    'cross_asset': [], # filled dynamically from xa_* columns
    'sentiment': [],   # filled dynamically from fg_*, vix_*, gp_* columns
    'temporal': ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'],
}


def discover_features(asset, horizon, n_top=5, iterations=100, populations=30,
                      max_corr=0.7, n_runs=4):
    """Run PySR symbolic regression with multiple diverse runs.

    Each run uses a different feature subset to force structurally different
    expressions. Results are pooled and deduplicated by output correlation.

    Args:
        asset: Asset name (e.g., 'BTC')
        horizon: Prediction horizon in hours
        n_top: Number of top expressions to save
        iterations: PySR iterations per run
        populations: Number of populations (ignored, uses 40 internally)
        max_corr: Maximum pairwise correlation between kept expressions
        n_runs: Number of independent PySR runs with different feature subsets

    Returns:
        Tuple of (results_list, pysr_rows)
    """
    print(f"\n{'='*70}")
    print(f"  PySR FEATURE DISCOVERY: {asset} {horizon}h")
    print(f"  {n_runs} diverse runs × {iterations} iterations | Top: {n_top} | max_corr: {max_corr}")
    print(f"{'='*70}")

    X, y, all_cols, pysr_rows = _prepare_data(asset, horizon)
    if X is None:
        return [], 0

    # Build feature subsets for each run — each run sees a different mix
    # Fill dynamic groups from actual columns
    macro_cols = [c for c in all_cols if c.startswith('m_')]
    xa_cols = [c for c in all_cols if c.startswith('xa_')]
    sent_cols = [c for c in all_cols if c.startswith(('fg_', 'vix_', 'gp_'))]

    # Filter group definitions to only include columns that exist
    groups = {}
    for name, cols in FEATURE_GROUPS.items():
        if name == 'macro':
            groups[name] = macro_cols
        elif name == 'cross_asset':
            groups[name] = xa_cols
        elif name == 'sentiment':
            groups[name] = sent_cols
        else:
            groups[name] = [c for c in cols if c in all_cols]

    # Define run configurations: each run gets 2-3 groups → different feature perspective
    run_configs = [
        ('A_mom+xa',     ['momentum', 'cross_asset', 'temporal']),
        ('B_vol+macro',  ['volatility', 'macro', 'temporal']),
        ('C_mr+sent',    ['mean_reversion', 'sentiment', 'temporal']),
        ('D_full_light', ['momentum', 'volatility', 'mean_reversion']),  # no external data
    ]

    # Add extra runs if requested
    extra_configs = [
        ('E_xa+sent',    ['cross_asset', 'sentiment', 'momentum']),
        ('F_macro+vol',  ['macro', 'volatility', 'mean_reversion']),
        ('G_all',        list(groups.keys())),  # full feature set as control
    ]
    while len(run_configs) < n_runs and extra_configs:
        run_configs.append(extra_configs.pop(0))

    run_configs = run_configs[:n_runs]

    # Execute runs
    all_results = []
    t0_total = time.time()

    for i, (label, group_names) in enumerate(run_configs):
        subset = []
        for gn in group_names:
            subset.extend(groups.get(gn, []))
        subset = list(dict.fromkeys(subset))  # dedupe preserving order

        if len(subset) < 5:
            print(f"\n  SKIP run {label}: only {len(subset)} features")
            continue

        seed = 42 + i * 17  # different seed per run
        run_results = _run_single_pysr(X, y, all_cols, subset, seed, iterations, label)
        all_results.extend(run_results)

    elapsed_total = (time.time() - t0_total) / 60
    print(f"\n  {'='*70}")
    print(f"  ALL RUNS COMPLETE: {len(all_results)} candidate expressions in {elapsed_total:.1f} min")
    print(f"  {'='*70}")

    if not all_results:
        print("  No expressions found!")
        return [], 0

    # Print all candidates before dedup
    print(f"\n  {'#':>3} | {'Run':<14} | {'Score':>8} | {'Loss':>10} | {'Cplx':>4} | Expression")
    print(f"  {'─'*90}")
    all_results.sort(key=lambda r: r['loss'])
    for i, r in enumerate(all_results[:30], 1):
        print(f"  {i:>3} | {r.get('run','?'):<14} | {r['score']:>8.4f} | {r['loss']:>10.6f} | {r['complexity']:>4} | {r['equation'][:50]}")

    # Dedup across all runs
    results = _dedup_by_correlation(all_results, X, all_cols, n_top, max_corr)

    print(f"\n  FINAL: {len(results)} diverse expressions kept (from {len(all_results)} candidates)")
    for i, r in enumerate(results, 1):
        print(f"    #{i} [{r.get('run','')}] loss={r['loss']:.4f}: {r['equation'][:70]}")

    # Remove 'run' key from results before saving
    for r in results:
        r.pop('run', None)

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
    parser.add_argument('--runs', type=int, default=4, help='Number of diverse PySR runs (default: 4, max: 7)')

    args = parser.parse_args()
    horizon = int(args.horizon.replace('h', ''))

    results, pysr_rows = discover_features(args.asset, horizon, n_top=args.top,
                                iterations=args.iterations, populations=args.populations,
                                max_corr=args.max_corr, n_runs=args.runs)

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
