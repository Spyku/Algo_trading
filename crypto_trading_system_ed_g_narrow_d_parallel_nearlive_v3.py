"""
Ed - G_narrow_d PARALLEL NEAR-LIVE V3 fork (2026-05-28 night)
============================================================
Builds on crypto_trading_system_ed_g_narrow_d_parallel_nearlive.py with TRUE
Mode D outer-loop parallelization via ProcessPoolExecutor.

What changes vs parallel_nearlive (v2):
  - Mode D outer loop: SERIAL (1 eval at a time) -> 8 ProcessPool workers
    Each worker still runs K=5 ThreadPool inside (unchanged quality)
    Net: 8 outer x 5 inner = 40 concurrent LGBM fits (was 5)
    Expected speedup: ~5-6x on Mode D, ~3-4x on full HRST
  - Print EVERY eval (not just NEW BEST + every 60th)
  - Full per-phase CSV logging with K=5 seed-by-seed breakdown:
      mode_d_full_{asset}_{horizon}_{ts}.csv

What stays IDENTICAL to parallel_nearlive (preserves quality):
  - K=5 ThreadPool inside each worker, all 5 seeds always complete
  - LGBM on CPU, num_threads=1
  - NEAR_LIVE_MODE=1 default (step=1, mean_last_10, ternary, embargo=horizon)
  - Isolation dirs (models_g_desktop_nearlive/, config_g_desktop_nearlive/)
  - Warning suppression (4-layer defense)
  - Regime config preservation across runs
  - NO grid reorder (windows iterated in standard order)
  - NO early-kill (all 60 configs run to completion)

Architecture: DISPATCHER PATTERN
  We do NOT replace the engine's 545-line run_mode_d_optuna function.
  Instead we monkey-patch two hooks:

    1. ENGINE._get_deku_diagnostic_models()  - the engine calls this once,
       right before its grid loop. The patch captures local variables from
       the calling frame (features_np, labels_np, closes_np, ranked_features,
       n, horizon, asset_name) via inspect.currentframe().f_back.

    2. ENGINE._deku_eval_with_pruning()  - on the FIRST call from the grid
       loop, the dispatcher uses the captured state to enumerate all 60
       configs and dispatch them to ProcessPoolExecutor. Returns this call's
       config result. Subsequent calls hit cached Futures (instant).

  The engine's own grid for-loop still runs in serial-iter order, but every
  call now returns from a parallel-pre-dispatched cache. Total wall-clock
  determined by ceil(60 / 8) * per_eval_time = ~7.5 * 5 min = ~38 min/horizon
  (was ~150 min serial).

Launch (single bare command, no env var setup needed):
  python crypto_trading_system_ed_g_narrow_d_parallel_nearlive_v3.py HRST "ETH," 5h,6h,7h,8h --replay 1440 --no-persist --no-data-update

Env overrides:
  MODE_D_OUTER_WORKERS=8   - number of ProcessPool workers (default 8)
  NEAR_LIVE_MODE=0          - turn off near-live (inherited default: 1)
"""

import os
import sys

# ====================================================================
# BULLETPROOF WARNING SUPPRESSION + NEAR_LIVE_MODE + ISOLATION DIRS
# (re-exec block, same as parallel_nearlive)
# ====================================================================
_WARNINGS_SENTINEL = '_PARALLEL_NEARLIVE_V3_WARNINGS_BAKED'
if not os.environ.get(_WARNINGS_SENTINEL):
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ.setdefault('NEAR_LIVE_MODE', '1')
    os.environ.setdefault('G_NARROW_MODELS_DIR', 'models_g_desktop_nearlive')
    os.environ.setdefault('G_NARROW_CONFIG_DIR', 'config_g_desktop_nearlive')
    os.environ.setdefault('MODE_D_OUTER_WORKERS', '8')
    os.environ[_WARNINGS_SENTINEL] = '1'
    os.execv(sys.executable, [sys.executable, '-W', 'ignore'] + sys.argv)

import time
import warnings
warnings.filterwarnings("ignore")
warnings.warn = lambda *a, **k: None
warnings.warn_explicit = lambda *a, **k: None

import inspect
import threading
import traceback
from concurrent.futures import (
    ThreadPoolExecutor, ProcessPoolExecutor, as_completed as _as_completed,
)
from datetime import datetime

import numpy as np
import pandas as pd

# ====================================================================
# Import parallel_nearlive FIRST - inherits ALL its patches
#   - GRID propagation (combos, windows, features, gammas)
#   - N_FEATURES_RANGE propagation
#   - Output dir routing
#   - Regime config preservation
#   - G._G_ORIG_DEKU_EVAL = step6_nearlive._H_ORIG_DEKU_EVAL
#   - G._deku_eval_with_pruning = _parallel_deku_eval_median_k (K=5 wrap)
#   - ENGINE._deku_eval_with_pruning = _parallel_deku_eval_median_k
# ====================================================================
import crypto_trading_system_ed_g_narrow_d_parallel_nearlive as PNL  # noqa: E402
import crypto_trading_system_ed as ENGINE  # noqa: E402
import crypto_trading_system_ed_g_narrow_d as G  # noqa: E402
import crypto_trading_system_ed_step6_nearlive as ENGINE_NL  # noqa: E402


# ====================================================================
# V3 STATE CAPTURE
# ====================================================================
# The engine's run_mode_d_optuna() calls _get_deku_diagnostic_models()
# exactly once per (asset, horizon), right BEFORE the grid loop. We hook
# that call to snapshot the calling-frame locals: features_np, labels_np,
# closes_np, ranked_features, n, horizon, asset_name. These are what
# the dispatcher needs to enumerate + dispatch the 60 configs.
# ====================================================================
_captured_state = {'current': None}

_orig_get_models = ENGINE._get_deku_diagnostic_models


def _capture_state_then_get_models():
    """Capture engine grid-setup locals, then delegate to original."""
    try:
        frame = inspect.currentframe().f_back
        loc = frame.f_locals if frame is not None else {}
        # Engine variable names inside run_mode_d_optuna (verified by grep
        # against crypto_trading_system_ed.py lines 4304-4420)
        state = {
            'features_np': loc.get('features_np'),
            'labels_np': loc.get('labels_np'),
            'closes_np': loc.get('closes_np'),
            'ranked_features': loc.get('ranked_features'),
            'n': loc.get('n'),
            'horizon': loc.get('horizon'),
            'asset_name': loc.get('asset_name'),
        }
        # Only treat as a valid Mode D entry if all required vars present
        if all(state[k] is not None for k in
               ('features_np', 'labels_np', 'closes_np', 'ranked_features', 'n', 'horizon')):
            _captured_state['current'] = state
            _dispatcher.reset_for_new_horizon(state)
            print(f"\n  [V3] state captured: asset={state['asset_name']} horizon={state['horizon']}h "
                  f"n={state['n']} ranked_features={len(state['ranked_features'])}", flush=True)
    except Exception as e:
        # Don't break engine if capture fails - dispatcher falls back to serial
        print(f"  [V3] state capture warning: {e}", flush=True)
    return _orig_get_models()


ENGINE._get_deku_diagnostic_models = _capture_state_then_get_models


# ====================================================================
# WORKER FUNCTION (runs in ProcessPool child processes)
# ====================================================================
# IMPORTANT: This function must be importable from the v3 module path
# so ProcessPool can pickle the reference. Args must be picklable.
# Worker re-imports modules (fresh interpreter on spawn), then runs K=5
# ThreadPool internally - identical to parallel_nearlive's K=5 path.
# ====================================================================

def _v3_worker_eval_one_config(args):
    """ProcessPool worker - runs one (combo, window, n_feat, gamma) config with K=5."""
    (combo_name, window, n_feat, gamma,
     features_b, features_shape, features_dtype,
     labels_b, labels_shape, labels_dtype,
     closes_b, closes_shape, closes_dtype,
     ranked_features, n, horizon) = args

    # Re-import in worker process - inheriting parallel_nearlive's patches
    # (parallel_nearlive sets G._G_ORIG_DEKU_EVAL -> step6_nearlive inner,
    #  Importing it re-applies the patch in this worker)
    import numpy as _np
    import crypto_trading_system_ed_g_narrow_d_parallel_nearlive as _PNL  # noqa
    import crypto_trading_system_ed_g_narrow_d as _G
    import crypto_trading_system_ed as _ENGINE

    # Defense: ensure NEAR_LIVE_MODE is set in worker env (should be inherited)
    os.environ.setdefault('NEAR_LIVE_MODE', '1')

    # Reconstruct numpy arrays from bytes
    features_np = _np.frombuffer(features_b, dtype=features_dtype).reshape(features_shape)
    labels_np = _np.frombuffer(labels_b, dtype=labels_dtype).reshape(labels_shape)
    closes_np = _np.frombuffer(closes_b, dtype=closes_dtype).reshape(closes_shape)

    # Build feature slice (matches engine line 4471: _feature_floor_indices)
    combo = combo_name.split('+')
    sel_idx = _ENGINE._feature_floor_indices(ranked_features, n_feat)
    feat_np = features_np[:, sel_idx]

    # K=5 inner ThreadPool - same code shape as parallel_nearlive's
    # _parallel_deku_eval_median_k, but we ALSO capture each seed's individual
    # result so the CSV can store K=5 seed-by-seed metrics.
    def _run_one_seed(seed):
        try:
            factories = _G._g_factories_seeded(seed)
            return _G._G_ORIG_DEKU_EVAL(
                feat_np, labels_np, closes_np, combo, window, n,
                _ENGINE.DIAG_STEP, factories,
                gamma=gamma, trial=None, horizon=horizon,
            )
        except Exception:
            return None

    seed_results = [None] * len(_G._G_SEEDS)
    with ThreadPoolExecutor(max_workers=len(_G._G_SEEDS)) as pool:
        futures = {pool.submit(_run_one_seed, s): i for i, s in enumerate(_G._G_SEEDS)}
        for f in _as_completed(futures):
            i = futures[f]
            try:
                seed_results[i] = f.result()
            except Exception:
                seed_results[i] = None

    valid = [r for r in seed_results if r is not None]
    if not valid:
        return (combo_name, window, n_feat, gamma, None, seed_results)

    # Median by cum_return at result index 4 (matches parallel_nearlive)
    valid.sort(key=lambda rr: rr[4])
    median = valid[len(valid) // 2]
    return (combo_name, window, n_feat, gamma, median, seed_results)


# ====================================================================
# PARALLEL GRID DISPATCHER
# ====================================================================
class _ParallelGridDispatcher:
    """Singleton that pre-dispatches all 60 grid configs to ProcessPool.

    The engine's run_mode_d_optuna() iterates the 4-nested grid serially
    and calls _deku_eval_with_pruning() per config. We patch that call:
    on the first hit, dispatch ALL configs in parallel; on every hit,
    block on the specific Future for that config. Engine still loops
    serially but each call returns from cache instead of executing.
    """

    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.pool = None
        self.futures = {}      # key -> Future
        self.config_order = []
        self.n_grid = 0
        self.dispatched = False
        self.t_start = None
        self.lock = threading.Lock()
        self.eval_count = 0
        self.best_apf = 0.0
        self.csv_rows = []
        self.csv_path = None
        self.monitor_thread = None

    def reset_for_new_horizon(self, state):
        """Called by state-capture hook at start of each (asset, horizon)."""
        with self.lock:
            # Shut down previous pool (HRST iterates 4 horizons)
            if self.pool is not None:
                try:
                    self.pool.shutdown(wait=True, cancel_futures=True)
                except Exception:
                    pass
                self.pool = None
            self.futures = {}
            self.config_order = []
            self.n_grid = 0
            self.dispatched = False
            self.t_start = None
            self.eval_count = 0
            self.best_apf = 0.0
            self.csv_rows = []
            asset = state['asset_name']
            horizon = state['horizon']
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.csv_path = os.path.join(
                ENGINE.MODELS_DIR,
                f'mode_d_full_{asset}_{horizon}h_{ts}.csv',
            )

    @staticmethod
    def _config_key(combo_name, window, n_feat, gamma):
        return (combo_name, window, int(n_feat), round(float(gamma), 5))

    def _dispatch_all(self):
        state = _captured_state['current']
        if state is None:
            raise RuntimeError("v3 dispatcher: no state captured - grid setup hook failed")

        features_np = state['features_np']
        labels_np = state['labels_np']
        closes_np = state['closes_np']
        ranked_features = state['ranked_features']
        n = state['n']
        horizon = state['horizon']

        # Enumerate configs in the SAME order the engine iterates (lines 4461-4474)
        self.config_order = []
        for combo_name in ENGINE.GRID_COMBOS:
            for window in ENGINE.GRID_WINDOWS:
                for n_feat in ENGINE.GRID_FEATURES:
                    if n_feat > len(ranked_features):
                        continue
                    for gamma in ENGINE.GRID_GAMMAS:
                        self.config_order.append((combo_name, window, n_feat, gamma))
        self.n_grid = len(self.config_order)

        # Serialize arrays (single pickle, shared across workers' arg tuples)
        features_b = features_np.tobytes()
        features_shape = features_np.shape
        features_dtype = str(features_np.dtype)
        labels_b = labels_np.tobytes()
        labels_shape = labels_np.shape
        labels_dtype = str(labels_np.dtype)
        closes_b = closes_np.tobytes()
        closes_shape = closes_np.shape
        closes_dtype = str(closes_np.dtype)

        self.t_start = time.time()
        self.pool = ProcessPoolExecutor(max_workers=self.max_workers)
        for (combo_name, window, n_feat, gamma) in self.config_order:
            args = (
                combo_name, window, n_feat, gamma,
                features_b, features_shape, features_dtype,
                labels_b, labels_shape, labels_dtype,
                closes_b, closes_shape, closes_dtype,
                ranked_features, n, horizon,
            )
            key = self._config_key(combo_name, window, n_feat, gamma)
            f = self.pool.submit(_v3_worker_eval_one_config, args)
            self.futures[key] = f
        self.dispatched = True
        print(f"  [V3] dispatched {self.n_grid} configs across {self.max_workers} "
              f"ProcessPool workers (K=5 ThreadPool inside each)", flush=True)
        print(f"  [V3] mode_d full CSV: {self.csv_path}", flush=True)

        # Spawn monitor thread to print + write CSV as futures complete
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()

    def _monitor(self):
        """Print + log every completion (out-of-order)."""
        try:
            futs = list(self.futures.values())
            for f in _as_completed(futs):
                try:
                    res = f.result()
                except Exception as e:
                    print(f"  [V3] worker exception: {e}\n{traceback.format_exc()}",
                          flush=True)
                    continue
                combo_name, window, n_feat, gamma, median, seed_results = res

                with self.lock:
                    self.eval_count += 1
                    eval_n = self.eval_count
                elapsed = (time.time() - self.t_start) / 60.0

                if median is None:
                    apf = ret = trades = win_rate = accuracy = raw_pf = 0
                    status = 'FAILED'
                    print(f"    [{eval_n:3d}/{self.n_grid}] {combo_name:10s} "
                          f"w={window:3d} f={n_feat:2d} g={gamma:.3f} | "
                          f"FAILED                       ({elapsed:.1f}min)",
                          flush=True)
                else:
                    trades = median[6]
                    accuracy = median[2]
                    ret = median[4]
                    win_rate = median[5]
                    raw_pf = median[11]
                    apf = ENGINE._compute_optuna_score(median) if trades >= 2 else 0.0
                    status = 'OK' if trades >= 2 else f'LOW_TR({trades})'
                    marker = ""
                    if apf > self.best_apf and trades >= 2:
                        self.best_apf = apf
                        marker = " <-- BEST"
                    print(f"    [{eval_n:3d}/{self.n_grid}] {combo_name:10s} "
                          f"w={window:3d} f={n_feat:2d} g={gamma:.3f} | "
                          f"apf={apf:.3f} ret={ret:+.1f}% tr={trades}  "
                          f"({elapsed:.1f}min){marker}", flush=True)

                # Full CSV row with K=5 seed breakdown
                row = {
                    'eval_order': eval_n,
                    'combo': combo_name,
                    'window': window,
                    'n_features': n_feat,
                    'gamma': gamma,
                    'apf': round(float(apf), 4),
                    'return_pct': round(float(ret), 2),
                    'trades': int(trades),
                    'win_rate': round(float(win_rate), 2),
                    'accuracy': round(float(accuracy), 4),
                    'raw_pf': round(float(raw_pf), 4),
                    'status': status,
                    'elapsed_min': round(elapsed, 2),
                }
                for sidx, sr in enumerate(seed_results):
                    if sr is None:
                        row[f'seed{sidx+1}_apf'] = None
                        row[f'seed{sidx+1}_ret'] = None
                        row[f'seed{sidx+1}_tr'] = None
                    else:
                        seed_tr = sr[6]
                        seed_apf = ENGINE._compute_optuna_score(sr) if seed_tr >= 2 else 0.0
                        row[f'seed{sidx+1}_apf'] = round(float(seed_apf), 4)
                        row[f'seed{sidx+1}_ret'] = round(float(sr[4]), 2)
                        row[f'seed{sidx+1}_tr'] = int(seed_tr)
                with self.lock:
                    self.csv_rows.append(row)

                # Flush every 5 evals + final
                if eval_n % 5 == 0 or eval_n == self.n_grid:
                    self._flush_csv()
        except Exception as e:
            print(f"  [V3] monitor thread crashed: {e}\n{traceback.format_exc()}",
                  flush=True)

    def _flush_csv(self):
        try:
            with self.lock:
                rows_copy = list(self.csv_rows)
            df = pd.DataFrame(rows_copy)
            df.to_csv(self.csv_path, index=False)
        except Exception as e:
            print(f"  [V3] CSV flush error: {e}", flush=True)

    def lookup(self, combo_name, window, n_feat, gamma):
        """Engine calls this synchronously. We block on the right Future."""
        if not self.dispatched:
            self._dispatch_all()
        key = self._config_key(combo_name, window, n_feat, gamma)
        f = self.futures.get(key)
        if f is None:
            return None
        res = f.result()  # blocks until ready
        _, _, _, _, median, _seed_results = res
        return median


_dispatcher = _ParallelGridDispatcher(
    max_workers=int(os.environ.get('MODE_D_OUTER_WORKERS', '8'))
)


# ====================================================================
# PATCH ENGINE._deku_eval_with_pruning
# ====================================================================
# On Mode D grid calls: route through the dispatcher (parallel cache).
# On other calls (Mode V step 1/2/3, holdout folds): fall back to the
# parallel_nearlive K=5 wrap (which is identical to the original Mode V
# pipeline, already proven). We discriminate by checking whether the
# (combo, window, n_feat, gamma) tuple matches a pre-dispatched key.
# ====================================================================

def _v3_routed_deku_eval(features_np, labels_np, closes_np, combo, window, n,
                         step, model_factories, gamma=1.0, trial=None,
                         horizon=None):
    """Routes Mode D grid calls through the dispatcher, others through K=5 wrap."""
    n_feat = features_np.shape[1]
    combo_name = '+'.join(combo)
    key = _ParallelGridDispatcher._config_key(combo_name, window, n_feat, gamma)

    # If dispatcher has this config pre-dispatched: use it (Mode D)
    if _dispatcher.dispatched and key in _dispatcher.futures:
        return _dispatcher.lookup(combo_name, window, n_feat, gamma)

    # If not yet dispatched AND we have captured state AND the engine is
    # calling from the grid loop: first call triggers dispatch.
    # (Identified by trial=None + having captured state.)
    if (not _dispatcher.dispatched
            and trial is None
            and _captured_state['current'] is not None):
        # Verify this call's params match an expected grid config
        state = _captured_state['current']
        configs = []
        for cn in ENGINE.GRID_COMBOS:
            for w in ENGINE.GRID_WINDOWS:
                for nf in ENGINE.GRID_FEATURES:
                    if nf > len(state['ranked_features']):
                        continue
                    for g in ENGINE.GRID_GAMMAS:
                        configs.append(_ParallelGridDispatcher._config_key(cn, w, nf, g))
        if key in set(configs):
            _dispatcher._dispatch_all()
            return _dispatcher.lookup(combo_name, window, n_feat, gamma)

    # Not a Mode D grid call - delegate to parallel_nearlive's K=5 wrap
    # (Holdout folds, Mode V step 1, Optuna refine all go through here.)
    return PNL._parallel_deku_eval_median_k(
        features_np, labels_np, closes_np, combo, window, n, step,
        model_factories, gamma=gamma, trial=trial, horizon=horizon,
    )


# Apply v3 routing patch to both ENGINE and G namespaces
ENGINE._deku_eval_with_pruning = _v3_routed_deku_eval
G._deku_eval_with_pruning = _v3_routed_deku_eval

print(f"\n[V3 PARALLEL_NEARLIVE] Mode D outer ProcessPool patched", flush=True)
print(f"  max_workers={_dispatcher.max_workers} (env MODE_D_OUTER_WORKERS)", flush=True)
print(f"  K=5 ThreadPool inside each worker - quality preserved", flush=True)
print(f"  NO early-kill, NO grid reorder - identical algorithm to parallel_nearlive", flush=True)
print(f"  Full per-eval CSV + console logging enabled", flush=True)


# ====================================================================
# Entry point
# ====================================================================
if __name__ == '__main__':
    # Need this guard for Windows spawn-based multiprocessing
    import multiprocessing as _mp
    _mp.freeze_support()
    ENGINE.main()
