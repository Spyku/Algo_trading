"""
Ed — G_narrow_d PARALLEL test fork (2026-05-22)
============================================================
Standalone benchmark of three parallelization fixes on top of
crypto_trading_system_ed_g_narrow_d.py. DOES NOT MODIFY PRODUCTION.

Fixes applied (all identity-preserving on model selection):

  #1 Device routing:
     _g_factories_seeded() now uses the worker's assigned LGBM device
     (read from G_PARALLEL_LGBM_DEVICE env var) instead of hardcoded GPU.
     Removes GPU contention when hybrid dispatcher labels a worker CPU.

  #2 Parallel K=5 within trial:
     _g_deku_eval_median_k() runs the K=5 seeds in a ThreadPoolExecutor
     instead of a sequential for-loop. Same seeds, same results, same
     median selection — just 5 threads instead of 1.

  #3 3rd refine worker:
     _refine_top_configs_hybrid replaced by a 3-worker version
     (initial_devices=['gpu','cpu','cpu'], max_workers=3). All 3 configs
     refine in parallel instead of 2-then-dynamic-1.

Why these three are safe:
  - Same K=5 seeds in the same order, results sorted and median picked
    the same way.
  - Same Optuna study per config, same TPESampler seed, same trial
    sequence (since TPE sampler is deterministic given seed and prior
    trials, and we don't share state across configs here).
  - Same 3 D-top configs refined.
  - Only the wall-clock dispatch changes; the model selection identity
    is preserved.

Risk surface:
  - LGBM thread-safety with 5 concurrent instances per process (verified
    in benchmark — separate Booster objects, no shared state).
  - Resource oversubscription if outer×inner > available cores. Desktop
    has 26 cores; 3 outer × 5 inner = 15 threads. Fine.
  - GPU contention if device='gpu' isn't replaced cleanly by env-var
    routing in every refine context.

Benchmark recipe:
  Smoke test (5 min, validates correctness):
    python crypto_trading_system_ed_g_narrow_d_parallel.py V ETH 7h \
      --replay 200 --refine-trials 5 --no-persist --no-data-update \
      --grid-tag G_NARROW_D

  Real benchmark (1 horizon, ~45-70 min target):
    $env:V2_DATA_SNAPSHOT = "data/_reliability_hrst_snapshot_desktop_20260515_154801"
    python crypto_trading_system_ed_g_narrow_d_parallel.py V ETH 7h \
      --replay 1440 --no-persist --no-data-update --grid-tag G_NARROW_D

  Compare against the May 20-21 G_narrow_d baseline for 7h Mode V:
    Step 1: 21.4 min | Refine: 223.1 min | Step 3: 19.2 min | total ~4h 24min
  Target with fixes: Refine ~50-70 min, total ~1.5-2h.
"""
import os
import sys

# ════════════════════════════════════════════════════════════════════════
# BULLETPROOF WARNING SUPPRESSION — 4-layer defense (2026-05-27 late evening)
# ════════════════════════════════════════════════════════════════════════
# First attempt (per-message PYTHONWARNINGS filters) leaked: sklearn or joblib
# calls warnings.simplefilter('always') after PYTHONWARNINGS is applied, which
# wipes the matched filters and lets the same two warnings flood thousands of
# times (29MB log of just two warning messages).
# Four layers now, in order of robustness:
#   1. `-W ignore` command-line flag added to execv argv — read at interpreter
#      startup, highest priority.
#   2. `PYTHONWARNINGS=ignore` env var — catch-all, also read at startup.
#   3. `warnings.filterwarnings('ignore')` after import — Python-level catch-all.
#   4. Monkey-patch `warnings.warn = no-op` — neutralizes the warning emission
#      function itself. Bulletproof against simplefilter('always') resets, since
#      the warn function never reaches the filter chain.
# A sentinel env var prevents infinite re-exec.
_WARNINGS_SENTINEL = '_PARALLEL_NEARLIVE_WARNINGS_BAKED'
if not os.environ.get(_WARNINGS_SENTINEL):
    # Earlier attempt used per-message filters; they were silently overridden
    # by some sklearn/joblib code path that calls warnings.simplefilter('always'),
    # so the same two warnings flooded the log thousands of times. Catch-all
    # 'ignore' wins regardless: it matches every category and every message.
    os.environ['PYTHONWARNINGS'] = 'ignore'

    # Bake NEAR_LIVE_MODE=1 by default — this fork's whole purpose is the
    # near-live semantics (step=1, na_policy=mean_last_10, signal=ternary,
    # return_probas=True). User can still override via:
    #     $env:NEAR_LIVE_MODE = "0"
    # before launching, if testing the parallel infra without near-live.
    os.environ.setdefault('NEAR_LIVE_MODE', '1')

    # Bake isolation dirs so this fork's writes never overlap with production
    # 'models/' and 'config/' directories. --no-persist alone uses _noprod
    # suffixes inside production dirs; isolation dirs add a stronger guarantee
    # (no shared parent at all). User can override either via env var before launch.
    os.environ.setdefault('G_NARROW_MODELS_DIR', 'models_g_desktop_nearlive')
    os.environ.setdefault('G_NARROW_CONFIG_DIR', 'config_g_desktop_nearlive')

    os.environ[_WARNINGS_SENTINEL] = '1'
    # Re-exec with `-W ignore` BEFORE the script argv. The -W command-line flag
    # is equivalent to PYTHONWARNINGS but applied even earlier (parsed by the
    # interpreter command-line, not via env). Belt-and-suspenders for cases
    # where PYTHONWARNINGS is reset by spawned worker init code.
    os.execv(sys.executable, [sys.executable, '-W', 'ignore'] + sys.argv)

import time
import warnings

# Layer 3: Python-level catch-all filter (in case PYTHONWARNINGS/-W get reset).
warnings.filterwarnings("ignore")

# Layer 4: monkey-patch warnings.warn to no-op. Brutal but bulletproof against
# any library code that calls warnings.simplefilter('always') / resetwarnings()
# after our filters were applied. sklearn's validation.py:2691 and parallel.py:144
# both use `warnings.warn(...)` — patching the module-level function neutralizes
# both. Real errors continue to raise via exceptions, not via warnings.warn.
warnings.warn = lambda *a, **k: None
warnings.warn_explicit = lambda *a, **k: None

from concurrent.futures import ThreadPoolExecutor, as_completed as _thread_as_completed
from concurrent.futures import ProcessPoolExecutor as _PoolExec
from concurrent.futures import as_completed as _proc_as_completed

# ────────────────────────────────────────────────────────────────────────
# Import G_narrow_d FIRST — this triggers its module-level K=5 patch
# (replaces engine._deku_eval_with_pruning with _g_deku_eval_median_k).
# We will then patch over G's patches.
# ────────────────────────────────────────────────────────────────────────
import crypto_trading_system_ed_g_narrow_d as G
import crypto_trading_system_ed as ENGINE


# ════════════════════════════════════════════════════════════════════════
# FIX #0: grid-constant propagation (pre-existing bug, identified 2026-05-24)
# ════════════════════════════════════════════════════════════════════════
# This fork calls ENGINE.main() at the bottom, which reads GRID_* from the
# ENGINE module namespace — not from G. G_narrow_d defines narrower grids
# (RF+LGBM/XGB+LGBM × [10,15,20] × [0.999,0.996]) but those never reach
# ENGINE.main(). Without this patch the parallel fork runs the engine's
# WIDER grid (3 combos × [10,13,17,25] × [0.999,0.997,0.995]), producing
# different Mode D top-6 candidates, different refine starting points,
# and ultimately different winners than a plain g_narrow_d.py run.
#
# Propagate G's narrow grid into ENGINE so ENGINE.main() searches the
# same hyperparameter space g_narrow_d would have searched.
# ════════════════════════════════════════════════════════════════════════
ENGINE.GRID_COMBOS = G.GRID_COMBOS
ENGINE.GRID_WINDOWS = G.GRID_WINDOWS
ENGINE.GRID_FEATURES = G.GRID_FEATURES
ENGINE.GRID_GAMMAS = G.GRID_GAMMAS

# Mode V refine n_features search bounds (Optuna upper cap).
# Engine caps short-horizon at 40 and long at 80; G drops the cap to 100
# per g_narrow_d's explicit "drop hard cap" design (LGBM's internal reg
# prunes — the cap was creating the B-7h tied-APF trap).
ENGINE.N_FEATURES_RANGE = G.N_FEATURES_RANGE
ENGINE.N_FEATURES_RANGE_DEFAULT = G.N_FEATURES_RANGE_DEFAULT

# Output routing — propagate G's MODELS_DIR / CONFIG_DIR / derived paths so
# the G_NARROW_MODELS_DIR / G_NARROW_CONFIG_DIR env vars set by the user
# (e.g. for Laptop isolation per TODO 2205) actually take effect when
# ENGINE.main() writes results. Without this, ENGINE reads H_STRICT_*
# instead and the G_NARROW_* env vars are silently ignored.
ENGINE.MODELS_DIR = G.MODELS_DIR
ENGINE.CONFIG_DIR = G.CONFIG_DIR
ENGINE.PRODUCTION_CSV = G.PRODUCTION_CSV
ENGINE.REGIME_CONFIG_PATH = G.REGIME_CONFIG_PATH
ENGINE.RESUME_DIR = G.RESUME_DIR

# Ensure redirected output dirs exist before any to_csv / json.dump fires.
# Without this, the first write into a fresh G_NARROW_MODELS_DIR / _CONFIG_DIR
# crashes with "Cannot save file into a non-existent directory" (engine line 4736).
for _d in (G.MODELS_DIR, G.CONFIG_DIR, G.RESUME_DIR):
    os.makedirs(_d, exist_ok=True)

# Seed the isolated CONFIG_DIR with a regime config. Two scenarios:
#   1. Fresh dir (neither regime_config_ed.json nor _noprod.json exists)
#      → seed from live so Mode R/S has a template to update.
#   2. Follow-up command after a prior RS/HRST left results in _noprod.json
#      → promote _noprod.json back onto regime_config_ed.json. The engine's
#        --no-persist code unconditionally does shutil.copyfile(regime_config_ed.json,
#        _noprod.json) on every invocation, which would otherwise WIPE OUT
#        Mode R/S's writes when Mode T is launched as a separate command.
# Mode R's _apply_mode_r_to_config (engine line 6525) reads the source then
# rewrites it; without a seed, Mode S/T crash with FileNotFoundError.
import shutil as _shutil
_live_regime_cfg = os.path.join(ENGINE.H75_CONFIG_DIR, 'regime_config_ed.json')
_seed_regime_cfg = G.REGIME_CONFIG_PATH
_noprod_regime_cfg = _seed_regime_cfg.replace('.json', '_noprod.json')
if os.path.exists(_noprod_regime_cfg):
    _shutil.copyfile(_noprod_regime_cfg, _seed_regime_cfg)
    print(f'[G_NARROW_D_PARALLEL] preserved prior run state: {_noprod_regime_cfg} -> {_seed_regime_cfg}', flush=True)
elif not os.path.exists(_seed_regime_cfg):
    if os.path.exists(_live_regime_cfg):
        _shutil.copyfile(_live_regime_cfg, _seed_regime_cfg)
        print(f'[G_NARROW_D_PARALLEL] seeded regime config: {_live_regime_cfg} -> {_seed_regime_cfg}', flush=True)
    else:
        print(f'[G_NARROW_D_PARALLEL] WARN: live regime config not found at {_live_regime_cfg} — Mode R/S/T may fail', flush=True)

print(f'[G_NARROW_D_PARALLEL] FIX #0 applied: ENGINE.GRID_* + N_FEATURES_RANGE + output dirs <- G', flush=True)
print(f'  combos={ENGINE.GRID_COMBOS}', flush=True)
print(f'  windows={ENGINE.GRID_WINDOWS}', flush=True)
print(f'  features={ENGINE.GRID_FEATURES}', flush=True)
print(f'  gammas={ENGINE.GRID_GAMMAS}', flush=True)
print(f'  n_features_range={ENGINE.N_FEATURES_RANGE}', flush=True)
print(f'  models_dir={ENGINE.MODELS_DIR}', flush=True)
print(f'  config_dir={ENGINE.CONFIG_DIR}', flush=True)


# ════════════════════════════════════════════════════════════════════════
# NEAR-LIVE PATCH (2026-05-27 evening, TEST FORK)
# ════════════════════════════════════════════════════════════════════════
# Wire the step6_nearlive INNER `_deku_eval_with_pruning` (the one with
# compute_signal_core delegation + NEAR_LIVE_MODE handling) into the
# H_STRICT_FAMILY K=5 wrapper. The wrapper calls `ENGINE._H_ORIG_DEKU_EVAL`
# K=5 times with different seeds; replacing that reference makes every
# K=5 inner call delegate to `crypto_signal_core_nearlive.compute_signal_core()`
# and honor the NEAR_LIVE_MODE env var.
#
# IMPORTANT: step6_nearlive's module-level `_deku_eval_with_pruning` is
# ITSELF wrapped by H_STRICT_FAMILY's K=5 in that module (the step6 copy
# inherits the production engine's K=5 patch at line 8941). So we point at
# `step6_nearlive._H_ORIG_DEKU_EVAL` — the PRE-K=5 inner function — to
# avoid K=25 nested-wrap explosion (5 outer × 5 inner).
#
# When NEAR_LIVE_MODE is NOT set, step6_nearlive's inner function uses
# defaults that preserve original engine math, so this patch is regression-
# safe when the env var is off. With NEAR_LIVE_MODE=1, every Mode D eval
# and Mode V refine eval runs under the new semantics.
# ════════════════════════════════════════════════════════════════════════
import crypto_trading_system_ed_step6_nearlive as ENGINE_NL  # noqa: E402
# The parallel fork replaces `_deku_eval_with_pruning` with a custom
# `_parallel_deku_eval_median_k` (defined later in this file) that calls
# `G._G_ORIG_DEKU_EVAL` — NOT `ENGINE._H_ORIG_DEKU_EVAL`. So we MUST patch
# G._G_ORIG_DEKU_EVAL to get NEAR_LIVE_MODE into the actual eval path.
# We also patch ENGINE._H_ORIG_DEKU_EVAL for defense-in-depth in case any
# code path falls through to the ENGINE wrapper.
G._G_ORIG_DEKU_EVAL = ENGINE_NL._H_ORIG_DEKU_EVAL  # primary patch — what _parallel_deku_eval_median_k actually calls
ENGINE._H_ORIG_DEKU_EVAL = ENGINE_NL._H_ORIG_DEKU_EVAL  # defense-in-depth
print(f'[G_NARROW_D_PARALLEL_NEARLIVE] patched G._G_ORIG_DEKU_EVAL + ENGINE._H_ORIG_DEKU_EVAL -> step6_nearlive inner _deku_eval_with_pruning', flush=True)
print(f'  NEAR_LIVE_MODE env var honored inside step6_nearlive (na=mean_last_10, signal=ternary, step=1, embargo=horizon)', flush=True)
print(f'  Off-by-default — set $env:NEAR_LIVE_MODE = "1" to activate', flush=True)


# ════════════════════════════════════════════════════════════════════════
# Machine-aware auto-tuning (Desktop / Laptop / Yoga)
# ════════════════════════════════════════════════════════════════════════
# Single file, one set of patches, but knobs auto-scale per machine via
# hardware_config (same detection logic the engine itself uses). User can
# still override anything via env vars.
# ════════════════════════════════════════════════════════════════════════
try:
    from hardware_config import MACHINE, N_JOBS_PARALLEL
except ImportError:
    MACHINE = 'UNKNOWN'
    N_JOBS_PARALLEL = os.cpu_count() or 8


# ════════════════════════════════════════════════════════════════════════
# FIX #1: device-routed K=5 factories
# ════════════════════════════════════════════════════════════════════════
# The K=5 wrap creates LGBM with hardcoded device='gpu' regardless of
# which refine worker (GPU/CPU) is running. We replace it with a version
# that reads G_PARALLEL_LGBM_DEVICE from the env (set by the wrapped
# refine worker before optimize()) and falls back to 'cpu' (safer than
# silently going GPU under contention).
# ════════════════════════════════════════════════════════════════════════

def _device_aware_factories_seeded(seed):
    """Mirror G._g_factories_seeded with worker-routed LGBM device.

    Reads device from G_PARALLEL_LGBM_DEVICE env var. Set by wrapped
    refine worker (see _refine_one_with_device below). Defaults to 'cpu'
    if unset, which is the correct safe choice when parallelizing K=5.

    Also caps LGBM num_threads so 5 parallel seeds × outer workers don't
    oversubscribe the CPU.
    """
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    device = os.environ.get('G_PARALLEL_LGBM_DEVICE', 'cpu')
    # Phase-aware thread budget (2026-05-27 late evening).
    # G_PARALLEL_LGBM_DEVICE is set ONLY by refine workers (lines 411, 538).
    # Its absence means we're in Mode D's serial grid loop — only K=5 inner is
    # parallel, not 3-outer × 5-inner. So we can give each LGBM N_JOBS // 5
    # threads instead of N_JOBS // 15, and fully saturate the CPU.
    #
    #   Mode D   Desktop: 26 // 5  = 5 threads/LGBM × 5 K = 25 cores
    #   Refine   Desktop: 26 // 15 = 1 thread/LGBM × 15 = 15 cores
    #
    # Without this, Mode D under-utilizes (5 cores × 1 thread = 5 cores raw,
    # GIL-bound to ~3 effective → 11% on a 26-core machine, observed on the
    # 2026-05-27 NEAR_LIVE_MODE run).
    _in_refine = 'G_PARALLEL_LGBM_DEVICE' in os.environ
    if _in_refine:
        _phase_default_threads = _DEFAULT_LGBM_THREADS
    else:
        _phase_default_threads = max(1, N_JOBS_PARALLEL // _PARALLEL_K)
    lgbm_threads = int(os.environ.get(
        'G_PARALLEL_LGBM_THREADS', str(_phase_default_threads)))

    def _rf():
        return RandomForestClassifier(
            n_estimators=100, max_depth=4, class_weight='balanced',
            random_state=seed, n_jobs=1,
        )

    def _gb():
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=seed,
        )

    def _xgb():
        return XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            random_state=seed, tree_method='hist', verbosity=0, n_jobs=1,
        )

    def _lr():
        return LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=seed,
        )

    def _lgbm():
        return LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            class_weight='balanced', verbose=-1, random_state=seed,
            device=device, num_threads=lgbm_threads,
        )

    return {'RF': _rf, 'GB': _gb, 'XGB': _xgb, 'LR': _lr, 'LGBM': _lgbm}


# Apply patch on G module
G._g_factories_seeded = _device_aware_factories_seeded


# ════════════════════════════════════════════════════════════════════════
# FIX #2: parallel K=5 multi-seed via ThreadPoolExecutor
# ════════════════════════════════════════════════════════════════════════
# Replace G._g_deku_eval_median_k's sequential for-loop with a thread
# pool. Same seeds, same model factories, same median-by-cum_return
# selection. Each seed independently builds its own sklearn/LGBM models
# (no shared state) → safe to thread.
# ════════════════════════════════════════════════════════════════════════

_PARALLEL_K = int(os.environ.get('PARALLEL_K_WORKERS', str(G._G_K)))

# Auto-compute LGBM OpenMP threads per instance to avoid oversubscription.
# Total LGBM instances in flight during refine = n_outer_workers × _PARALLEL_K.
# With n_outer = 3 (REFINE_TOP_N) → 3 × 5 = 15 LGBM instances concurrent.
# Budget per instance: N_JOBS_PARALLEL / 15. Floor to 1 (LGBM needs ≥1).
#
#   Desktop  N_JOBS=26 → 26 // 15 = 1
#   Laptop   N_JOBS=14 → 14 // 15 = 0 → 1
#   Yoga     N_JOBS=12 → 12 // 15 = 0 → 1
#
# Everyone defaults to 1. Override only if you have a reason and know the
# math (e.g. machine with 64+ cores → could try threads=2 or higher).
_N_OUTER_REFINE_WORKERS = 3   # ENGINE.REFINE_TOP_N
_DEFAULT_LGBM_THREADS = max(
    1, N_JOBS_PARALLEL // (_N_OUTER_REFINE_WORKERS * _PARALLEL_K)
)


def _parallel_deku_eval_median_k(features_np, labels_np, closes_np, combo,
                                  window, n, step, model_factories,
                                  gamma=1.0, trial=None, horizon=None):
    """Parallel version of G._g_deku_eval_median_k.

    K seeds run concurrently via a ThreadPoolExecutor. Returns the same
    median (by cum_return at result index 4) as the sequential version.
    Note: model_factories argument is ignored on purpose, exactly like
    G's sequential version — each seed builds its own seeded factories
    via G._g_factories_seeded (now the device-aware version).
    """
    if horizon is None:
        horizon = G.PREDICTION_HORIZON

    def _run_one_seed(seed):
        factories = G._g_factories_seeded(seed)
        return G._G_ORIG_DEKU_EVAL(
            features_np, labels_np, closes_np, combo, window, n, step,
            factories, gamma=gamma, trial=None, horizon=horizon,
        )

    results = []
    with ThreadPoolExecutor(max_workers=_PARALLEL_K) as pool:
        futures = [pool.submit(_run_one_seed, s) for s in G._G_SEEDS]
        for f in _thread_as_completed(futures):
            try:
                r = f.result()
                if r is not None:
                    results.append(r)
            except Exception as e:
                # Match G's behavior — silently skip failed seeds, only
                # return None if NO seed succeeded.
                pass

    if not results:
        return None

    results.sort(key=lambda rr: rr[4])
    return results[len(results) // 2]


# Patch on both G and ENGINE namespaces so any name resolution path
# (inside refine workers re-importing engine, or inside G's own scope)
# picks up the parallel version.
G._deku_eval_with_pruning = _parallel_deku_eval_median_k
ENGINE._deku_eval_with_pruning = _parallel_deku_eval_median_k


# ════════════════════════════════════════════════════════════════════════
# FIX #3: 3-worker refine dispatcher + device env-var routing
# ════════════════════════════════════════════════════════════════════════
# Wrap engine._refine_one_config_worker so it sets G_PARALLEL_LGBM_DEVICE
# in the worker process's env before calling optimize(). This is the
# bridge that lets the K=5 factories pick up the correct device per
# worker (without modifying engine code).
#
# Then replace _refine_top_configs_hybrid with a 3-parallel-worker
# version (was: 2 workers + dynamic 3rd).
# ════════════════════════════════════════════════════════════════════════

_ORIG_REFINE_ONE = ENGINE._refine_one_config_worker


def _refine_one_with_device(cfg_idx, top_entry_pickle, asset, horizon,
                             ranked_features, features_np, labels_np,
                             closes_np, n_size, lgbm_device,
                             n_trials_override=None):
    """Wrapper that broadcasts lgbm_device to the K=5 factories via env.

    Runs inside the spawned worker process. The env var is per-process
    (each worker has its own), so this is naturally isolated.
    """
    os.environ['G_PARALLEL_LGBM_DEVICE'] = lgbm_device
    # Seed the thread budget for this worker's K=5 fan-out using the
    # parent process's machine-aware default (passed via env so spawned
    # workers inherit it). User can override via G_PARALLEL_LGBM_THREADS.
    if 'G_PARALLEL_LGBM_THREADS' not in os.environ:
        os.environ['G_PARALLEL_LGBM_THREADS'] = str(_DEFAULT_LGBM_THREADS)
    return _ORIG_REFINE_ONE(
        cfg_idx, top_entry_pickle, asset, horizon, ranked_features,
        features_np, labels_np, closes_np, n_size, lgbm_device,
        n_trials_override,
    )


def _refine_top_configs_3workers(asset, horizon, top3_for_refine, df_raw,
                                  df_clean, all_cols, ranked_features):
    """3-worker hybrid refine. Replaces _refine_top_configs_hybrid.

    Submits ALL 3 configs at once across 3 workers (1 GPU + 2 CPU)
    instead of the engine's 2-then-dynamic-1. Same setup phase, same
    workers, same Optuna studies.

    Falls back to engine's original implementation when len <= 1.
    """
    import numpy as _np

    if len(top3_for_refine) <= 1:
        return ENGINE._ENG_REFINE_TOP_CONFIGS_ORIG(
            asset, horizon, top3_for_refine, df_raw, df_clean, all_cols,
            ranked_features,
        )

    # ── Setup phase (identical to engine's _refine_top_configs_hybrid) ──
    MAX_DIAG_HOURS = 6 * 30 * 24
    df_full_r, all_cols_r = ENGINE.build_all_features(
        df_raw, asset_name=asset, horizon=horizon)
    n_pysr = ENGINE._compute_pysr_features(
        df_full_r, all_cols_r, asset, horizon, verbose=False)
    if n_pysr > 0:
        pysr_cols_r = [c for c in all_cols_r if c.startswith('pysr_')]
        is_clean, _ = ENGINE._check_pysr_leakage(pysr_cols_r, asset, horizon)
        if not is_clean:
            for pc in pysr_cols_r:
                all_cols_r.remove(pc)
                if pc in df_full_r.columns:
                    df_full_r.drop(columns=[pc], inplace=True)
    if len(df_full_r) > MAX_DIAG_HOURS:
        df_full_r = df_full_r.tail(MAX_DIAG_HOURS).reset_index(drop=True)
    df_full_r, all_cols_r, dropped_r = ENGINE._filter_sparse_features(
        df_full_r, all_cols_r)
    ranked_features_r = [c for c in ranked_features if c not in dropped_r]
    df_clean_r = df_full_r.dropna(
        subset=ranked_features_r + ['label']).reset_index(drop=True)

    features_np_all = df_clean_r[ranked_features_r].values.astype(_np.float64)
    labels_np_all = df_clean_r['label'].values.astype(_np.int32)
    closes_np_all = df_clean_r['close'].values.astype(_np.float64)
    n_total = len(df_clean_r)
    f1_train_e = int(n_total * 0.60)

    features_np = features_np_all[:f1_train_e]
    labels_np = labels_np_all[:f1_train_e]
    closes_np = closes_np_all[:f1_train_e]
    n_size = len(features_np)

    n_workers = min(3, len(top3_for_refine))
    # CPU-ONLY when K=5 fan-out is active: 5 threads × LGBM-GPU per worker
    # crashes the GPU's single CUDA context. With CPU we already get 5×
    # parallelism from the seed fan-out, so GPU isn't needed on the outer
    # worker. Override via G_FORCE_CPU_REFINE=0 if testing GPU path on
    # a machine where multi-context GPU works (e.g. Desktop RTX 4080).
    force_cpu = os.environ.get('G_FORCE_CPU_REFINE', '1') == '1'
    if force_cpu and _PARALLEL_K > 1:
        initial_devices = ['cpu'] * n_workers
    else:
        initial_devices = (['gpu'] + ['cpu'] * (n_workers - 1))[:n_workers]

    print(f"\n  [refine-3worker] dispatching {len(top3_for_refine)} configs "
          f"across {n_workers} workers (devices={initial_devices}, "
          f"K_parallel={_PARALLEL_K})", flush=True)

    t_refine = time.time()
    refined_list = [None] * len(top3_for_refine)

    pool = _PoolExec(max_workers=n_workers)
    futures = {}
    try:
        for cfg_idx in range(len(top3_for_refine)):
            dev = initial_devices[cfg_idx % n_workers]
            top_entry = top3_for_refine[cfg_idx]
            f = pool.submit(
                _refine_one_with_device, cfg_idx, top_entry, asset,
                horizon, ranked_features_r, features_np, labels_np,
                closes_np, n_size, dev, ENGINE._REFINE_TRIALS_OVERRIDE,
            )
            futures[f] = (cfg_idx, dev)

        while futures:
            done = next(_proc_as_completed(futures))
            cfg_idx, dev = futures.pop(done)
            try:
                got_idx, refined, captured, _dev_used = done.result()
                if captured:
                    sys.stdout.write(captured)
                    if not captured.endswith('\n'):
                        sys.stdout.write('\n')
                    sys.stdout.flush()
                refined_list[got_idx] = refined
            except Exception as e:
                # A worker died — pool is now poisoned. Log it, finish
                # the wait, then retry this config sequentially on the
                # main process (no pool, no thread fan-out) so we still
                # get a refined config back.
                print(f"  [refine-3worker] config #{cfg_idx+1} ({dev}) "
                      f"FAILED with {type(e).__name__}: {e}",
                      flush=True)
                refined_list[cfg_idx] = ('FAILED', cfg_idx, dev, top3_for_refine[cfg_idx])
    finally:
        pool.shutdown(wait=True)

    # Sequential retry for any failed configs (pool is poisoned after a
    # BrokenProcessPool, so we run these in-process on the main thread).
    failed_entries = [r for r in refined_list if isinstance(r, tuple) and r[0] == 'FAILED']
    if failed_entries:
        print(f"\n  [refine-3worker] retrying {len(failed_entries)} failed "
              f"config(s) sequentially in main process", flush=True)
        for tag, cfg_idx, dev, top_entry in failed_entries:
            try:
                os.environ['G_PARALLEL_LGBM_DEVICE'] = 'cpu'
                got_idx, refined, captured, _dev_used = _ORIG_REFINE_ONE(
                    cfg_idx, top_entry, asset, horizon, ranked_features_r,
                    features_np, labels_np, closes_np, n_size, 'cpu',
                    ENGINE._REFINE_TRIALS_OVERRIDE,
                )
                if captured:
                    sys.stdout.write(captured)
                    sys.stdout.flush()
                refined_list[got_idx] = refined
            except Exception as e:
                print(f"  [refine-3worker] sequential retry config #{cfg_idx+1} "
                      f"ALSO FAILED with {type(e).__name__}: {e}", flush=True)
                refined_list[cfg_idx] = None

    # Filter out None (truly failed) AND any leftover FAILED tuples that
    # didn't get a successful sequential retry.
    all_refined = [r for r in refined_list
                   if r is not None and not (isinstance(r, tuple) and r[0] == 'FAILED')]
    refine_elapsed = (time.time() - t_refine) / 60
    print(f"\n  [Refine: {refine_elapsed:.1f} min] "
          f"(3-worker parallel + K=5 thread fan-out, "
          f"{len(all_refined)}/{len(top3_for_refine)} configs refined)",
          flush=True)

    if all_refined:
        all_refined.sort(key=lambda x: -x['apf'])
        for i, r in enumerate(all_refined, 1):
            print(f"  Refined #{i}: {r['combo']}  w={r['window']}h  "
                  f"g={r['gamma']:.4f}  f={r['n_features']}  apf={r['apf']:.3f}")

    return all_refined


# Patch both the implementation and the public alias so run_mode_v's
# call to _refine_top_configs resolves to our 3-worker version.
ENGINE._refine_top_configs_hybrid = _refine_top_configs_3workers
ENGINE._refine_top_configs = _refine_top_configs_3workers


# ────────────────────────────────────────────────────────────────────────
# Banner — confirms patches landed for both parent and workers (workers
# print their own banner when they re-import this module).
# ────────────────────────────────────────────────────────────────────────
print(f'[G_NARROW_D_PARALLEL] patches applied (machine={MACHINE}, N_JOBS_PARALLEL={N_JOBS_PARALLEL}):', flush=True)
print(f'  #1 device routing: K=5 factories read G_PARALLEL_LGBM_DEVICE per worker (CPU default when K>1)', flush=True)
print(f'  #2 parallel K=5:   {_PARALLEL_K} seeds via ThreadPoolExecutor per trial', flush=True)
print(f'  #3 3-worker refine: configs submitted across max 3 workers', flush=True)
print(f'  resolved budget:   {_N_OUTER_REFINE_WORKERS} outer × {_PARALLEL_K} inner × {_DEFAULT_LGBM_THREADS} LGBM-thread '
      f'= {_N_OUTER_REFINE_WORKERS * _PARALLEL_K * _DEFAULT_LGBM_THREADS} OS threads (cap={N_JOBS_PARALLEL})', flush=True)


# ────────────────────────────────────────────────────────────────────────
# Entry point — mirrors g_narrow_d's __main__ block
# ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Parse --refine-trials override (same as g_narrow_d)
    if '--refine-trials' in sys.argv:
        idx = sys.argv.index('--refine-trials')
        try:
            n_trials = int(sys.argv[idx + 1])
            ENGINE._REFINE_TRIALS_OVERRIDE = n_trials
            G._REFINE_TRIALS_OVERRIDE = n_trials
            del sys.argv[idx:idx + 2]
            print(f"  --refine-trials override: {n_trials} "
                  f"(default REFINE_TRIALS={ENGINE.REFINE_TRIALS})")
        except (ValueError, IndexError):
            print("  --refine-trials must be an integer (e.g. --refine-trials 5)")
            sys.exit(2)

    print("=" * 80)
    print(f"  CRYPTO TRADING SYSTEM ED — {ENGINE.MACHINE} [G_NARROW_D_PARALLEL]")
    print(f"  Parallel dispatch: {ENGINE.PARALLEL_BACKTESTS} workers, "
          f"LGBM={ENGINE.PARALLEL_LGBM_DEVICE} for parallel sections")
    print(f"  Refine: 3-worker outer × {_PARALLEL_K}-thread K=5 inner")
    print("=" * 80)

    ENGINE.main()

    # Same loky shutdown / clean-exit dance as g_narrow_d, to suppress
    # the harmless resource_tracker shutdown noise.
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except Exception:
        pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        _devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull, 2)
        os._exit(0)
    except Exception:
        pass
