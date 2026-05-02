"""
crypto_trading_system_ed_parallel_p.py
======================================

Experimental parallel-Mode-P wrapper. Mode P (PySR symbolic regression
feature discovery) is the slowest research mode in the engine — each
(asset, horizon) pair runs 4-7 sequential PySR studies, each with PySR's
own internal parallelism explicitly disabled (`procs=0`,
`parallelism="serial"`) because `deterministic=True` requires it.

This wrapper takes Option B:
  - Each individual PySR run stays deterministic (same seed + serial =
    bit-for-bit reproducible across re-runs)
  - The N outer PySR runs (different feature subsets, different seeds)
    are dispatched across multiple processes via ProcessPoolExecutor
  - On the laptop: 3 concurrent PySR Julia processes
  - On the desktop: 4 concurrent PySR Julia processes
  - Each worker holds its own Julia runtime + population history
    (~2-3 GB RAM each) — total ≤ ~10 GB on laptop with 3 workers,
    ≤ ~12 GB on desktop with 4 workers

Production `pysr_discover_features.py` and `crypto_trading_system_ed.py`
are NOT modified — this script monkey-patches `discover_features` only
in the `__main__` process, so importing this module from elsewhere has
zero side effects on the engine.

Usage:
    python crypto_trading_system_ed_parallel_p.py P ETH 5h
    python crypto_trading_system_ed_parallel_p.py P ETH 5,6,7,8h

All other CLI flags pass through to the engine unchanged. Recommended:
do NOT run while a Mode V / HRST is also running — Mode P spins up
multiple Julia processes that compete with loky workers for cores + RAM.
"""

from __future__ import annotations

import io
import os
import sys
import time
import contextlib

# Production modules — DO NOT mutate at import time. Patches only apply
# inside the `__main__` block at the bottom.
# Engine internals were renamed from `crypto_trading_system_ed` to
# `crypto_trading_system_ed_engine` on 2026-04-30 when the parallel
# wrapper was promoted to be the canonical `crypto_trading_system_ed`
# entry point. We need the engine module specifically (not the wrapper)
# because the Mode P patches below operate on engine internals.
import crypto_trading_system_ed_engine as eng
import pysr_discover_features as pysr_mod

# Capture the originals NOW so we always have a handle to the real
# implementations even after we monkey-patch them in __main__.
_PYSR_DISCOVER_FEATURES_ORIG = pysr_mod.discover_features
_PYSR_RUN_SINGLE_ORIG = pysr_mod._run_single_pysr


# ──────────────────────────────────────────────────────────────
# Per-machine policy
# ──────────────────────────────────────────────────────────────
# Each PySR worker spawns its own Julia runtime (~2-3 GB memory, 1 core
# of CPU because each PySR keeps `parallelism='serial'` for determinism).
# Number of concurrent workers = how many of the 4-7 outer runs execute
# at once. PySR cold-start is ~10-20s per worker (paid once when the
# pool spins up).
if eng.MACHINE == 'DESKTOP':
    PYSR_PARALLEL_RUNS = 4   # 26 cores, 32 GB — plenty of headroom
elif eng.MACHINE == 'YOGA':
    PYSR_PARALLEL_RUNS = 2   # 14 GB RAM — keep tight
else:  # LAPTOP
    PYSR_PARALLEL_RUNS = 3   # 16 GB RAM — 3 × ~2.5 GB Julia ≈ 7.5 GB peak


# ──────────────────────────────────────────────────────────────
# Worker — runs one PySR study in its own process
# ──────────────────────────────────────────────────────────────
def _pysr_run_worker(X, y, all_cols, feature_subset, seed,
                     iterations, run_label):
    """ProcessPool worker — runs ONE _run_single_pysr call in a fresh
    process. Each worker has its own Julia runtime; failures here don't
    affect siblings or parent. stdout captured per worker to keep parent
    log clean (parent prints captured output back in dispatch order on
    completion).

    stdin is closed in the worker because PySR's interactive progress
    dashboard (verbosity=1, progress=True default) reads from stdin to
    handle "Press 'q' and then <enter> to stop" — with multiple workers
    sharing the parent's stdin, that read deadlocks until the user
    presses Enter, which only unblocks one worker per keypress. Closing
    stdin here makes PySR's read return EOF immediately and skip the
    interactive prompt entirely.

    Returns (run_label, results_list_or_None, captured_stdout, error_str_or_None).
    """
    # Redirect OS-level fd 0 (stdin) to NUL/devnull BEFORE any PySR import.
    # Julia opens stdin via the OS file handle, bypassing Python's sys.stdin
    # entirely — closing sys.stdin alone is not enough. We must replace fd 0
    # so Julia's interactive prompt sees EOF and skips polling. This must
    # happen BEFORE _run_single_pysr's `from pysr import PySRRegressor`
    # because that import triggers Julia runtime initialization, and Julia
    # captures stdin at init time.
    try:
        _devnull_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(_devnull_fd, 0)
        os.close(_devnull_fd)
    except (OSError, AttributeError):
        pass  # best effort — don't crash worker if redirection fails

    buf = io.StringIO()
    saved_out, saved_err, saved_in = sys.stdout, sys.stderr, sys.stdin
    sys.stdout, sys.stderr = buf, buf
    sys.stdin = io.StringIO()  # belt + braces — Python-level too
    err_msg = None
    results = None
    try:
        # Re-import in worker (fresh process). Both engine and pysr_mod
        # are imported at module level in workers because cloudpickle
        # serializes by-reference; functions resolve their imports
        # against the worker's own globals.
        import pysr_discover_features as _pysr_mod
        # progress=False + verbosity=0: disable PySR's interactive dashboard
        # AND its OS-level stdin poll. Closing sys.stdin above only covers
        # Python's view of stdin — Julia opens stdin at the OS file-handle
        # level (bypassing sys.stdin), so verbosity=0 is required to kill
        # the "Press q + enter" prompt that polls between horizons.
        results = _pysr_mod._run_single_pysr(
            X, y, all_cols, feature_subset, seed, iterations, run_label,
            progress=False, verbosity=0,
        )
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err
        sys.stdin = saved_in
    return run_label, results, buf.getvalue(), err_msg


# ──────────────────────────────────────────────────────────────
# Parallel discover_features — drop-in replacement
# ──────────────────────────────────────────────────────────────
def discover_features_parallel(asset, horizon, n_top=5, iterations=100,
                               populations=30, max_corr=0.7, n_runs=4,
                               load_data_fn=None, build_features_fn=None,
                               horizon_suffix='h'):
    """Drop-in replacement for pysr_discover_features.discover_features.
    Same signature, same outputs (results list + pysr_rows count).
    Differs only in that the N outer PySR runs (4-7 by default) execute
    in parallel across PYSR_PARALLEL_RUNS processes instead of sequentially.

    Each PySR run stays internally deterministic (procs=0, serial,
    deterministic=True, fixed seed) — the speedup comes purely from
    overlapping different runs in parallel processes.

    Falls back to the engine's sequential discover_features if
    PYSR_PARALLEL_RUNS <= 1 or only 1 run is configured.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # ── Setup phase (mirrors the original function up to the run loop) ──
    print(f"\n{'='*70}")
    print(f"  PySR FEATURE DISCOVERY (PARALLEL): {asset} {horizon}{horizon_suffix}")
    print(f"  {n_runs} diverse runs × {iterations} iterations | Top: {n_top} | "
          f"max_corr: {max_corr}")
    print(f"  Workers: {PYSR_PARALLEL_RUNS} concurrent PySR processes "
          f"(deterministic, serial inside each)")
    print(f"{'='*70}")

    X, y, all_cols, pysr_rows = pysr_mod._prepare_data(
        asset, horizon,
        load_data_fn=load_data_fn,
        build_features_fn=build_features_fn
    )
    if X is None:
        return [], 0

    # Build dynamic feature groups (same logic as the engine)
    macro_cols = [c for c in all_cols if c.startswith('m_')]
    xa_cols = [c for c in all_cols if c.startswith('xa_')]
    sent_cols = [c for c in all_cols if c.startswith(('fg_', 'vix_', 'gp_'))]
    groups = {}
    for name, cols in pysr_mod.FEATURE_GROUPS.items():
        if name == 'macro':
            groups[name] = macro_cols
        elif name == 'cross_asset':
            groups[name] = xa_cols
        elif name == 'sentiment':
            groups[name] = sent_cols
        else:
            groups[name] = [c for c in cols if c in all_cols]

    run_configs = [
        ('A_mom+xa',     ['momentum', 'cross_asset', 'temporal']),
        ('B_vol+macro',  ['volatility', 'macro', 'temporal']),
        ('C_mr+sent',    ['mean_reversion', 'sentiment', 'temporal']),
        ('D_full_light', ['momentum', 'volatility', 'mean_reversion']),
    ]
    extra_configs = [
        ('E_xa+sent',    ['cross_asset', 'sentiment', 'momentum']),
        ('F_macro+vol',  ['macro', 'volatility', 'mean_reversion']),
        ('G_all',        list(groups.keys())),
    ]
    while len(run_configs) < n_runs and extra_configs:
        run_configs.append(extra_configs.pop(0))
    run_configs = run_configs[:n_runs]

    # Build (subset, seed, label) tuples for each run, skipping any with
    # too few features (matches the engine's sequential skip logic).
    tasks = []
    for i, (label, group_names) in enumerate(run_configs):
        subset = []
        for gn in group_names:
            subset.extend(groups.get(gn, []))
        subset = list(dict.fromkeys(subset))  # dedupe preserving order
        if len(subset) < 5:
            print(f"\n  SKIP run {label}: only {len(subset)} features")
            continue
        seed = 42 + i * 17
        tasks.append((label, subset, seed))

    if not tasks:
        print("  No valid run configs — nothing to dispatch")
        return [], pysr_rows

    # Sequential fallback (matches engine semantics exactly)
    if PYSR_PARALLEL_RUNS <= 1 or len(tasks) <= 1:
        print(f"  [pysr-parallel] only {len(tasks)} task(s) — falling back "
              f"to sequential via captured original")
        return _PYSR_DISCOVER_FEATURES_ORIG(
            asset, horizon, n_top=n_top, iterations=iterations,
            populations=populations, max_corr=max_corr, n_runs=n_runs,
            load_data_fn=load_data_fn,
            build_features_fn=build_features_fn,
            horizon_suffix=horizon_suffix,
        )

    n_workers = min(len(tasks), PYSR_PARALLEL_RUNS)
    print(f"\n  [pysr-parallel] dispatching {len(tasks)} PySR runs across "
          f"{n_workers} workers")
    print(f"  [pysr-parallel] cold-start: ~10-20s per worker for Julia init "
          f"(paid concurrently)")
    t0 = time.time()

    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        # Submit every task; pool dispatches them across workers as
        # workers free up. With max_workers=N and len(tasks)>N, the pool
        # naturally batches: first N run concurrently, then as each
        # finishes, the next pending task starts on that worker.
        futures = {}
        for label, subset, seed in tasks:
            f = pool.submit(_pysr_run_worker, X, y, all_cols, subset, seed,
                            iterations, label)
            futures[f] = label

        for f in as_completed(futures):
            label = futures[f]
            try:
                got_label, results, captured, err_msg = f.result()
                if captured:
                    sys.stdout.write(captured)
                    if not captured.endswith('\n'):
                        sys.stdout.write('\n')
                if err_msg:
                    print(f"  [pysr-parallel] {label} FAILED: {err_msg}")
                elif results:
                    all_results.extend(results)
                    print(f"  [pysr-parallel] {label} → {len(results)} expressions")
                else:
                    print(f"  [pysr-parallel] {label} → 0 expressions")
            except Exception as e:
                print(f"  [pysr-parallel] {label} CRASHED: "
                      f"{type(e).__name__}: {e}")

    elapsed = (time.time() - t0) / 60
    print(f"\n  {'='*70}")
    print(f"  ALL RUNS COMPLETE: {len(all_results)} candidate expressions in "
          f"{elapsed:.1f} min (parallel)")
    print(f"  {'='*70}")

    if not all_results:
        print("  No expressions found!")
        return [], pysr_rows

    # ── Same dedup + return as the engine's original ──
    print(f"\n  {'#':>3} | {'Run':<14} | {'Score':>8} | {'Loss':>10} | "
          f"{'Cplx':>4} | Expression")
    print(f"  {'─'*90}")
    all_results.sort(key=lambda r: r['loss'])
    for i, r in enumerate(all_results[:30], 1):
        print(f"  {i:>3} | {r.get('run','?'):<14} | {r['score']:>8.4f} | "
              f"{r['loss']:>10.6f} | {r['complexity']:>4} | "
              f"{r['equation'][:50]}")

    results = pysr_mod._dedup_by_correlation(all_results, X, all_cols, n_top, max_corr)

    print(f"\n  FINAL: {len(results)} diverse expressions kept (from "
          f"{len(all_results)} candidates)")
    for i, r in enumerate(results, 1):
        print(f"    #{i} [{r.get('run','')}] loss={r['loss']:.4f}: "
              f"{r['equation'][:70]}")

    for r in results:
        r.pop('run', None)

    return results, pysr_rows


# ──────────────────────────────────────────────────────────────
# Entry point — monkey-patch and call engine.main()
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 80)
    print("  EXPERIMENTAL: Parallel Mode P wrapper")
    print(f"  Machine: {eng.MACHINE}  |  Workers: {PYSR_PARALLEL_RUNS}  |  "
          f"deterministic=True (per-run seed reproducibility kept)")
    print("  Production engine + pysr_discover_features NOT modified —")
    print("  only this process patches:")
    print("    pysr_discover_features.discover_features ← discover_features_parallel")
    print("=" * 80)

    # Patch the source module. The engine's `from pysr_discover_features
    # import discover_features` inside run_mode_p resolves the name at
    # call time against the (now patched) source module attribute.
    pysr_mod.discover_features = discover_features_parallel

    # Hand off to engine main() — routes to run_mode_p which will call
    # the patched discover_features.
    eng.main()
