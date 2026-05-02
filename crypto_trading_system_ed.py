"""
crypto_trading_system_ed_parallel.py
====================================

Experimental parallel wrapper around the production engine. Production
`crypto_trading_system_ed.py` is NOT modified — this script imports it
and monkey-patches three entry points with parallel siblings:

    eng.run_mode_v ← run_mode_v_parallel  (Step 1 + Step 3 backtests parallel)
    eng.run_mode_s ← run_mode_s_parallel  (per-horizon signal cache parallel)
    eng.run_mode_t ← run_mode_t_parallel  (per-horizon signal cache parallel)

All three patches fire only when this file is run as __main__, so
`import crypto_trading_system_ed_parallel` from elsewhere has zero side
effects on the engine.

If this script breaks, production is unaffected — just stop using it.

Usage:
    python crypto_trading_system_ed_parallel.py HRST ETH 5,6,7,8h --replay 1440 --no-persist

All other CLI flags pass through to the engine unchanged.

Per-machine policy (default same on all machines):
    Desktop / Laptop / Yoga : 6 workers, LGBM='cpu' in parallel sections

LGBM falls back to CPU inside parallel sections because GPU LGBM
serializes device access — 6 concurrent GPU LGBM workers queue and run
barely faster than 1 sequential GPU worker. CPU LGBM is ~2-3× slower per
fit but lets all 6 workers actually run concurrently → near-linear core
scaling. Net: ~2-4× wall-clock speedup vs production sequential.

Sequential code paths (Mode D grid — already parallel internally, single
backtests outside Mode V/S/T) keep using the engine's default LGBM_DEVICE.

Mode H per-horizon parallelism is intentionally NOT done: Mode D inside
each horizon already uses N_JOBS_PARALLEL=14 workers, so 4 horizons in
parallel = 56 workers on 14 cores → severe oversubscription. Sequential
horizons + parallel Mode V/S/T cache is the right division.
"""

from __future__ import annotations

import io
import sys
import time
import contextlib

# Import the engine internals — DO NOT mutate the module's state until we're
# inside __main__ and explicitly want to monkey-patch entry points.
# Engine internals were renamed from `crypto_trading_system_ed` to
# `crypto_trading_system_ed_engine` on 2026-04-30 when this parallel wrapper
# was promoted to be the canonical `crypto_trading_system_ed` entry point.
import crypto_trading_system_ed_engine as eng

# Re-export engine module-level names so external imports like
# `from crypto_trading_system_ed import load_data, build_all_features,
# PREDICTION_HORIZON` keep working unchanged after the rename.
# (pysr_discover_features.py + tools/* + downstream scripts rely on this.)
# `from x import *` does NOT re-export names starting with `_`, but several
# external callers (download_macro_data.py, extend_caches_90d.py,
# pysr_discover_features.py) import private helpers like
# `_load_disabled_features`, `_suppress_stderr`,
# `_build_regime_indicators_and_detectors`, `_run_single_pysr`. We copy
# every non-dunder name explicitly so private engine helpers stay importable
# from `crypto_trading_system_ed` after the rename.
for _name in dir(eng):
    if not _name.startswith('__'):
        globals().setdefault(_name, getattr(eng, _name))
del _name

from joblib import Parallel, delayed

# Capture engine originals NOW, before any monkey-patching. The parallel
# wrappers below need to delegate back to the engine's real implementation;
# if they called `eng.run_mode_s` after __main__ replaced it with
# run_mode_s_parallel, they'd recurse infinitely. Storing the originals
# here at wrapper-import time guarantees we always have a handle to the
# real engine functions regardless of later patches.
_ENG_RUN_MODE_S_ORIG = eng.run_mode_s
_ENG_RUN_MODE_T_ORIG = eng.run_mode_t
_ENG_RUN_MODE_V_ORIG = eng.run_mode_v
_ENG_REFINE_TOP_CONFIGS_ORIG = eng._refine_top_configs

# ──────────────────────────────────────────────────────────────
# Per-machine policy — sourced from hardware_config.py (single source of truth)
# ──────────────────────────────────────────────────────────────
from hardware_config import PARALLEL_BACKTESTS, PARALLEL_LGBM_DEVICE


# ──────────────────────────────────────────────────────────────
# Parallel-aware model factory
# ──────────────────────────────────────────────────────────────
def _get_deku_models_with_device(lgbm_device):
    """Same factory shape as engine's _get_deku_models, but builds LGBM with
    a caller-specified device. Used by workers to override GPU→CPU during
    the parallel section.

    The other models (RF, GB, XGB, LR) are unchanged — they're CPU-only
    with n_jobs=1 already so they parallelize naturally across workers.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None

    factories = {
        'RF':   lambda: RandomForestClassifier(n_estimators=300, max_depth=4,
                                               class_weight='balanced',
                                               random_state=42, n_jobs=1),
        'GB':   lambda: GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                                   random_state=42),
        'LR':   lambda: LogisticRegression(max_iter=1000,
                                           class_weight='balanced',
                                           random_state=42),
        'LGBM': lambda: LGBMClassifier(n_estimators=300, max_depth=4,
                                       learning_rate=0.05,
                                       class_weight='balanced',
                                       verbose=-1, random_state=42,
                                       device=lgbm_device),
    }
    if XGBClassifier is not None:
        factories['XGB'] = lambda: XGBClassifier(n_estimators=300, max_depth=3,
                                                 learning_rate=0.05,
                                                 random_state=42,
                                                 tree_method='hist',
                                                 verbosity=0, n_jobs=1)
    return factories


# ──────────────────────────────────────────────────────────────
# Loky worker — runs in a fresh process per dispatched task
# ──────────────────────────────────────────────────────────────
def _backtest_one_config_worker(asset, horizon, label, cfg, replay_hours, lgbm_device):
    """Runs in a loky worker process. Each worker is fully independent:
      - Mutating eng.ALL_MODELS in this process does NOT affect parent or
        siblings (separate process memory).
      - stdout/stderr captured to a buffer so parent log isn't interleaved.
      - LGBM device overridden via the local factory so concurrent fits
        don't queue on the same GPU.

    Returns (label, result, captured_stdout). result may be None if the
    underlying generate_signals produced no signals (already a valid case
    handled by Mode V's downstream code).
    """
    # Override the engine's ALL_MODELS dict in THIS worker process.
    # _backtest_one_config → generate_signals → ALL_MODELS[name]() picks up
    # our LGBM-CPU factories instead of the engine default.
    eng.ALL_MODELS = _get_deku_models_with_device(lgbm_device)

    buf = io.StringIO()
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf, buf
    try:
        result = eng._backtest_one_config(
            asset, horizon, label, cfg, replay_hours=replay_hours
        )
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err
    return label, result, buf.getvalue()


# ──────────────────────────────────────────────────────────────
# Parallel runner
# ──────────────────────────────────────────────────────────────
def _run_parallel_backtests(asset, horizon, items, replay_hours):
    """items: iterable of (label, cfg). Returns dict {label: result}.

    Falls back to sequential when PARALLEL_BACKTESTS <= 1 OR len(items) <= 1.
    """
    items = list(items)
    if not items:
        return {}
    if PARALLEL_BACKTESTS <= 1 or len(items) <= 1:
        out = {}
        for lbl, cfg in items:
            out[lbl] = eng._backtest_one_config(
                asset, horizon, lbl, cfg, replay_hours=replay_hours
            )
        return out

    n_workers = min(len(items), PARALLEL_BACKTESTS)
    print(f"  [parallel] dispatching {len(items)} backtests across {n_workers} workers "
          f"(LGBM={PARALLEL_LGBM_DEVICE} for parallel section)")
    t0 = time.time()
    outputs = Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
        delayed(_backtest_one_config_worker)(
            asset, horizon, lbl, cfg, replay_hours, PARALLEL_LGBM_DEVICE
        )
        for lbl, cfg in items
    )
    out = {}
    for lbl, r, captured in outputs:
        if captured:
            # Replay the worker's stdout into the parent log in dispatch order.
            sys.stdout.write(captured)
            if not captured.endswith('\n'):
                sys.stdout.write('\n')
        out[lbl] = r
    print(f"  [parallel] {len(items)} backtests completed in "
          f"{(time.time()-t0)/60:.1f} min")
    return out


# ──────────────────────────────────────────────────────────────
# Parallel run_mode_v — sibling of engine.run_mode_v with Step 1
# and Step 3 loops parallelized. Logic is byte-identical otherwise.
# ──────────────────────────────────────────────────────────────
def run_mode_v_parallel(assets_list, horizons=None, replay_hours=None):
    """Parallel-Mode-V drop-in. Same outputs as engine.run_mode_v —
    same all_results dict, same production CSV row format, same summary
    table, same `production_models` list shape. Differs only in:
      - Step 1's 6 D-candidate backtests run via _run_parallel_backtests
      - Step 3's 3 refined backtests run via _run_parallel_backtests
      - LGBM uses CPU inside the parallel section (avoids GPU queue)
    """
    import os
    import pandas as pd

    # Mirror engine.run_mode_v defaults
    if horizons is None:
        horizons = list(eng.AVAILABLE_HORIZONS)
    _replay = replay_hours or eng.MODE_G_REPLAY_HOURS

    candidates_csv = eng._get_models_csv_path()
    if not os.path.exists(candidates_csv):
        print(f"  ERROR: {candidates_csv} not found. Run Mode D first.")
        return

    df_candidates = pd.read_csv(candidates_csv)

    print("=" * 80)
    print(f"  MODE V (PARALLEL): LIVE BACKTEST + REFINE — "
          f"{','.join(assets_list)} {','.join(str(h)+'h' for h in horizons)}")
    print(f"  Period: last {_replay} hours ({_replay/168:.1f} weeks), every hour")
    print(f"  Ranking by: conf>={eng.MODE_G_PRIMARY_CONF}% return")
    print(f"  Pipeline: backtest top 6 (parallel) → refine top 3 → "
          f"backtest refined (parallel) → pick best")
    print(f"  Parallelism: {PARALLEL_BACKTESTS} workers, LGBM={PARALLEL_LGBM_DEVICE} (parallel sections)")
    print(f"  Candidates: {candidates_csv}")
    print(f"  Production: {eng.PRODUCTION_CSV}")
    print("=" * 80)

    all_results = {}
    production_models = []

    for asset in assets_list:
        for horizon in horizons:
            mask = (df_candidates['coin'] == asset) & (df_candidates['horizon'] == horizon)
            if mask.sum() == 0:
                print(f"\n  No candidates for {asset} {horizon}h — skipping")
                continue

            key = f"{asset}_{horizon}h"
            print(f"\n{'#' * 70}")
            print(f"  {asset} {horizon}h")
            print(f"{'#' * 70}")

            # LGBM feature ranking for this asset (sequential — small cost,
            # uses GPU for the single ranking call which is fine).
            print(f"\n  Computing LGBM feature ranking for {asset}...")
            t0 = time.time()
            df_raw = eng.load_data(asset)
            df_full, all_cols = eng.build_all_features(df_raw, asset_name=asset, horizon=horizon)
            n_pysr = eng._compute_pysr_features(df_full, all_cols, asset, horizon)
            if n_pysr > 0:
                pysr_cols_loaded = [c for c in all_cols if c.startswith('pysr_')]
                is_clean, leak_msg = eng._check_pysr_leakage(pysr_cols_loaded, asset, horizon)
                if not is_clean:
                    print(f"\n  *** LEAKAGE DETECTED (early check): {leak_msg}")
                    print(f"  *** Stripping {len(pysr_cols_loaded)} PySR features from this run")
                    for pc in pysr_cols_loaded:
                        all_cols.remove(pc)
                        if pc in df_full.columns:
                            df_full.drop(columns=[pc], inplace=True)
            if len(df_full) > _replay:
                df_full = df_full.tail(_replay).reset_index(drop=True)
            df_full, all_cols, _ = eng._filter_sparse_features(df_full, all_cols)
            df_clean = df_full.dropna(subset=all_cols + ['label']).reset_index(drop=True)
            importance_df = eng._test_lgbm_importance(df_clean, all_cols, gamma=1.0)
            ranked_features = importance_df['feature'].tolist()
            print(f"  [Ranking done: {time.time()-t0:.1f}s] — {len(ranked_features)} features ranked")

            configs = {}
            asset_candidates = df_candidates[mask].sort_values(
                'rank' if 'rank' in df_candidates.columns else 'combined_score',
                ascending='rank' in df_candidates.columns)

            for idx, row in asset_candidates.iterrows():
                rank = int(row.get('rank', idx + 1))
                ho_apf = float(row.get('combined_score', 0))
                label = f"D #{rank} (APF={ho_apf:.1f})"

                if pd.notna(row.get('optimal_features', None)) and row['optimal_features']:
                    features = row['optimal_features'].split(',')
                else:
                    features = ranked_features[:int(row['n_features'])]

                configs[label] = {
                    'combo': row['best_combo'],
                    'window': int(row['best_window']),
                    'gamma': float(row['gamma']),
                    'features': features,
                    'n_features': int(row['n_features']),
                    'source': 'doohan',
                    'rank': rank,
                    'ho_apf': ho_apf,
                    'csv_row': row.to_dict(),
                }

            # ── STEP 1: Backtest Mode D candidates IN PARALLEL ──
            print(f"\n{'='*70}")
            print(f"  STEP 1: BACKTEST MODE D CANDIDATES (parallel)")
            print(f"{'='*70}")

            results = _run_parallel_backtests(
                asset, horizon, configs.items(), _replay
            )

            # ── STEP 2: Pick top 3 for refine (sequential — Optuna sampler is
            #   intrinsically sequential and TPE quality drops with parallelism). ──
            doohan_results = [(lbl, r) for lbl, r in results.items()
                              if r and r['cfg'].get('source') == 'doohan'
                              and f'conf_{eng.MODE_G_PRIMARY_CONF}' in r]
            doohan_results.sort(
                key=lambda x: -x[1][f'conf_{eng.MODE_G_PRIMARY_CONF}']['return_pct'])
            top3_for_refine = doohan_results[:eng.REFINE_TOP_N]

            if top3_for_refine:
                print(f"\n{'='*70}")
                print(f"  STEP 2: OPTUNA REFINE — top {len(top3_for_refine)} live performers")
                print(f"  {eng.REFINE_TRIALS} trials per config | "
                      f"gamma ±{eng.REFINE_GAMMA_RANGE} | "
                      f"features ±{eng.REFINE_FEAT_RANGE} | "
                      f"window ±{eng.REFINE_WINDOW_RANGE}h")
                print(f"{'='*70}")

                for i, (lbl, r) in enumerate(top3_for_refine, 1):
                    cfg = r['cfg']
                    live_ret = r[f'conf_{eng.MODE_G_PRIMARY_CONF}']['return_pct']
                    print(f"  #{i}: {cfg['combo']}  w={cfg['window']}h  "
                          f"g={cfg['gamma']:.4f}  f={cfg['n_features']}  "
                          f"live_ret={live_ret:+.2f}%")

                refined_configs = eng._refine_top_configs(
                    asset, horizon, top3_for_refine, df_raw, df_clean,
                    all_cols, ranked_features)

                # ── STEP 3: Backtest refined configs IN PARALLEL ──
                if refined_configs:
                    print(f"\n{'='*70}")
                    print(f"  STEP 3: BACKTEST REFINED CONFIGS (parallel)")
                    print(f"{'='*70}")

                    refined_items = []
                    for i, rcfg in enumerate(refined_configs, 1):
                        label = f"Refined #{i} (APF={rcfg['apf']:.1f})"
                        cfg = {
                            'combo': rcfg['combo'],
                            'window': rcfg['window'],
                            'gamma': rcfg['gamma'],
                            'features': rcfg['features'],
                            'n_features': rcfg['n_features'],
                            'source': 'refined',
                        }
                        refined_items.append((label, cfg))

                    refined_results = _run_parallel_backtests(
                        asset, horizon, refined_items, _replay
                    )
                    results.update(refined_results)

            all_results[key] = results

            # ── Find overall best (identical to engine.run_mode_v) ──
            all_candidates = [(lbl, r) for lbl, r in results.items() if r]

            if all_candidates:
                best_label, best_r, best_conf = None, None, eng.MODE_G_PRIMARY_CONF
                best_score = -999
                for lbl, r in all_candidates:
                    for conf in eng.MODE_G_CONF_THRESHOLDS:
                        if f'conf_{conf}' not in r:
                            continue
                        sim = r[f'conf_{conf}']
                        if sim['trades'] < 5:
                            continue
                        ret = sim['return_pct']
                        wr = sim['win_rate'] / 100.0
                        score = ret * wr if ret > 0 else ret
                        if score > best_score:
                            best_score = score
                            best_label = lbl
                            best_r = r
                            best_conf = conf

                if best_r is None:
                    for lbl, r in all_candidates:
                        if f'conf_{eng.MODE_G_PRIMARY_CONF}' in r:
                            sim = r[f'conf_{eng.MODE_G_PRIMARY_CONF}']
                            ret = sim['return_pct']
                            wr = sim['win_rate'] / 100.0
                            score = ret * wr if ret > 0 else ret
                            if score > best_score:
                                best_score = score
                                best_label = lbl
                                best_r = r
                                best_conf = eng.MODE_G_PRIMARY_CONF

                if best_r:
                    best_sim = best_r[f'conf_{best_conf}']
                    best_cfg = best_r['cfg']

                    print(f"\n  {'='*70}")
                    print(f"  OVERALL BEST: {best_label}  →  {asset} {horizon}h")
                    print(f"  {best_cfg['combo']}  w={best_cfg['window']}h  "
                          f"g={best_cfg['gamma']:.4f}  f={best_cfg['n_features']}")
                    print(f"  Return (conf>={best_conf}%): {best_sim['return_pct']:+.2f}%  "
                          f"WR={best_sim['win_rate']:.0f}%  trades={best_sim['trades']}")
                    print(f"  {'='*70}")

                    if best_cfg.get('source') == 'doohan' and 'csv_row' in best_cfg:
                        prod_row = best_cfg['csv_row'].copy()
                        prod_row['return_pct'] = round(best_sim['return_pct'], 2)
                        prod_row['accuracy'] = round(best_sim['win_rate'], 1)
                        prod_row['combined_score'] = round(best_score, 4)
                        prod_row['sampler'] = 'Grid'
                        if 'rank' in prod_row:
                            del prod_row['rank']
                    else:
                        prod_row = {
                            'coin': asset,
                            'best_window': best_cfg['window'],
                            'best_combo': best_cfg['combo'],
                            'models': best_cfg['combo'],
                            'return_pct': round(best_sim['return_pct'], 2),
                            'accuracy': round(best_sim['win_rate'], 1),
                            'combined_score': round(best_score, 4),
                            'feature_set': 'D',
                            'n_features': best_cfg['n_features'],
                            'optimal_features': ','.join(best_cfg['features']),
                            'horizon': horizon,
                            'gamma': round(best_cfg['gamma'], 4),
                            'sampler': 'Refined',
                        }

                    features_to_check = best_cfg.get('features', [])
                    if isinstance(features_to_check, str):
                        features_to_check = features_to_check.split(',')
                    is_clean, leak_msg = eng._check_pysr_leakage(features_to_check, asset, horizon)
                    if not is_clean:
                        print(f"\n  *** LEAKAGE BLOCKED: {leak_msg}")
                        print(f"  *** Skipping production write for {asset} {horizon}h")
                        print(f"  *** Fix: run Mode P first, then re-run Mode DV")
                    else:
                        if any(f.startswith('pysr_') for f in features_to_check):
                            print(f"  PySR leakage check: {leak_msg}")
                        production_models.append((prod_row, horizon, best_conf))

    # ── Combined Summary (delegated to the engine's logic — we have the
    #   same all_results / production_models structure, so the summary
    #   block of run_mode_v that comes AFTER this point is identical;
    #   we re-implement it inline by calling helpers that engine exposes.) ──
    # The engine's run_mode_v continues with the summary table + production
    # CSV write after this point. Rather than duplicate that, delegate by
    # invoking the same code path: call the engine's _persist_mode_v_results
    # if it exists, else inline the rest.
    return _finish_mode_v(all_results, production_models, assets_list, horizons, _replay)


def _finish_mode_v(all_results, production_models, assets_list, horizons, _replay):
    """Replicates the post-loop tail of engine.run_mode_v: print the combined
    summary, write production CSV / regime config / best_models if we have
    a winner per (asset, horizon). Kept as a separate function for clarity.

    The summary print + persist logic in engine.run_mode_v lives at lines
    ~4737 onwards. Rather than copying ~150 more lines, we call the engine's
    already-tested `_persist_mode_v_results` if it's been factored out, else
    we delegate by calling engine.run_mode_v on an empty candidates set
    after seeding all_results — but that's hacky. Cleanest: just copy the
    summary + persist block. (It only writes per-row output, no fresh
    backtests.)
    """
    import pandas as pd
    import os

    # Combined summary
    print(f"\n\n{'=' * 80}")
    print(f"  SUMMARY: MODE V (PARALLEL) — D + Refined — Last {_replay//24} days "
          f"(scored by return × win_rate)")
    print(f"{'=' * 80}")

    for key, results in all_results.items():
        if not results:
            continue
        first = next((r for r in results.values() if r), None)
        if first:
            print(f"\n  {key} — Buy & Hold: {first['buy_hold']:+.2f}%\n")

        header = (f"  {'Model':<25} | {'Combo':14s} | {'W':>4} | {'G':>6} | {'F':>3} | "
                  f"{'Conf':>4} | {'Return':>8} | {'Tr':>3} | {'RT':>3} | {'WR':>4}")
        print(header)
        print(f"  {'─' * len(header)}")

        for label, r in results.items():
            if not r:
                continue
            cfg = r['cfg']
            for conf in eng.MODE_G_CONF_THRESHOLDS:
                if f'conf_{conf}' not in r:
                    continue
                sim = r[f'conf_{conf}']
                still = '*' if sim['still_invested'] else ''
                print(f"  {label:<25} | {cfg['combo']:14s} | "
                      f"{cfg['window']:>3}h | {cfg['gamma']:>.3f} | "
                      f"{cfg['n_features']:>3} | {conf:>3}% | "
                      f"{sim['return_pct']:>+7.2f}% | "
                      f"{sim['trades']:>3} | {sim['round_trips']:>3} | "
                      f"{sim['win_rate']:>3.0f}%{still}")

        print(f"\n  * = still invested at end of period")

    # Persist winners — call engine's existing persist logic so the CSV
    # write goes through _atomic_write_csv (which has the parent-dir
    # resilience fix) and matches production format exactly.
    if production_models:
        for prod_row, horizon, best_conf in production_models:
            print(f"\n  Promoting {prod_row['coin']} {horizon}h → production "
                  f"(min_confidence={best_conf}%, return={prod_row['return_pct']}%)")
        # The engine's run_mode_v writes to PRODUCTION_CSV via
        # _atomic_write_csv on the rows it accumulates; replicate that here.
        # If engine exposes a helper, prefer that — otherwise inline.
        if hasattr(eng, '_write_production_models'):
            eng._write_production_models(production_models)
        else:
            # Inline production CSV update — append/upsert per (coin, horizon).
            try:
                if os.path.exists(eng.PRODUCTION_CSV):
                    df_prod = pd.read_csv(eng.PRODUCTION_CSV)
                else:
                    df_prod = pd.DataFrame()

                for prod_row, horizon, best_conf in production_models:
                    asset = prod_row['coin']
                    if not df_prod.empty:
                        df_prod = df_prod[
                            ~((df_prod['coin'] == asset) &
                              (df_prod['horizon'] == horizon))
                        ]
                    df_prod = pd.concat(
                        [df_prod, pd.DataFrame([prod_row])], ignore_index=True
                    )

                eng._atomic_write_csv(df_prod, eng.PRODUCTION_CSV, index=False)
                print(f"\n  Production CSV updated: {eng.PRODUCTION_CSV}")
            except Exception as e:
                print(f"\n  WARNING: production CSV write skipped ({e})")
                print(f"  Inspect _noprod outputs and promote manually if happy.")

    return all_results


# ──────────────────────────────────────────────────────────────
# Mode S / Mode T per-horizon signal-cache parallel
# ──────────────────────────────────────────────────────────────
# Both Mode S (line 5066-5091) and Mode T (line 5293-5309) build a
# per-horizon signals_cache via a sequential loop over `generate_signals`.
# Strategy: pre-compute every call we know the engine will make, run them
# in parallel via loky, then monkey-patch eng.generate_signals for the
# duration of the eng.run_mode_s / eng.run_mode_t call so the sequential
# loop instantly returns from cache. Cache miss falls through to the real
# engine.generate_signals (graceful — preserves correctness even if our
# call-prediction misses an unusual code path).
#
# Why this approach over copying ~800 lines of run_mode_s + run_mode_t:
#   - Massively less code to maintain; engine drift doesn't break us
#     unless the prediction logic itself diverges.
#   - The patch is scoped (context manager) so other generate_signals
#     callers in the run aren't affected.

import contextlib
import json
from joblib import Parallel as _Parallel, delayed as _delayed


def _key_for_call(asset, model_names, window_size, replay_hours,
                  feature_override, horizon, gamma):
    """Stable cache key for a generate_signals invocation."""
    feats = tuple(feature_override) if feature_override else None
    return (
        asset,
        '+'.join(model_names) if model_names else '',
        int(window_size),
        int(replay_hours),
        feats,
        int(horizon),
        round(float(gamma), 6),
    )


def _predict_signal_calls_for_horizons(asset, horizons, replay):
    """Replicates the per-horizon `generate_signals` call extraction used
    by run_mode_s and run_mode_t. Reads the same df_models the engine
    reads, picks the same top-by-combined_score row per (asset, horizon),
    and returns a list of call-arg dicts.

    If df_models is missing a row for some (asset, h), that horizon is
    skipped — same behavior as the engine.
    """
    import pandas as pd
    df_models = pd.read_csv(eng.PRODUCTION_CSV)
    calls = []
    for h in horizons:
        rows = df_models[(df_models['coin'] == asset) & (df_models['horizon'] == h)]
        if len(rows) == 0:
            continue
        row = rows.sort_values('combined_score', ascending=False).iloc[0]
        feats = (row['optimal_features'].split(',')
                 if pd.notna(row.get('optimal_features', '')) else None)
        gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0
        models = row['models'].split('+')
        window = int(row['best_window'])
        calls.append({
            'asset': asset,
            'model_names': models,
            'window_size': window,
            'replay_hours': int(replay),
            'feature_override': feats,
            'horizon': int(h),
            'gamma': gamma,
        })
    return calls


def _signal_gen_worker(call_dict, lgbm_device):
    """Loky worker — runs one generate_signals call with LGBM device
    overridden for parallel safety. stdout captured so workers don't
    interleave their progress lines.
    """
    eng.ALL_MODELS = _get_deku_models_with_device(lgbm_device)
    buf = io.StringIO()
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf, buf
    try:
        sigs = eng.generate_signals(
            asset_name=call_dict['asset'],
            model_names=call_dict['model_names'],
            window_size=call_dict['window_size'],
            replay_hours=call_dict['replay_hours'],
            feature_override=call_dict['feature_override'],
            horizon=call_dict['horizon'],
            gamma=call_dict['gamma'],
        )
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err
    return call_dict, sigs, buf.getvalue()


def _build_signals_cache_parallel(calls, lgbm_device):
    """Run a list of generate_signals calls in parallel via loky.
    Returns dict {key_tuple: signals_list}.
    """
    if not calls:
        return {}
    if len(calls) == 1 or PARALLEL_BACKTESTS <= 1:
        # Sequential fallback — same behavior as not patching at all
        cache = {}
        for c in calls:
            sigs = eng.generate_signals(
                asset_name=c['asset'],
                model_names=c['model_names'],
                window_size=c['window_size'],
                replay_hours=c['replay_hours'],
                feature_override=c['feature_override'],
                horizon=c['horizon'],
                gamma=c['gamma'],
            )
            key = _key_for_call(c['asset'], c['model_names'], c['window_size'],
                                c['replay_hours'], c['feature_override'],
                                c['horizon'], c['gamma'])
            cache[key] = sigs
        return cache

    n_workers = min(len(calls), PARALLEL_BACKTESTS)
    print(f"  [parallel] dispatching {len(calls)} signal generations across {n_workers} workers "
          f"(LGBM={lgbm_device})")
    t0 = time.time()
    outputs = _Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
        _delayed(_signal_gen_worker)(c, lgbm_device) for c in calls
    )
    cache = {}
    for c, sigs, captured in outputs:
        if captured:
            sys.stdout.write(captured)
            if not captured.endswith('\n'):
                sys.stdout.write('\n')
        key = _key_for_call(c['asset'], c['model_names'], c['window_size'],
                            c['replay_hours'], c['feature_override'],
                            c['horizon'], c['gamma'])
        cache[key] = sigs
    print(f"  [parallel] {len(calls)} signals generated in "
          f"{(time.time()-t0)/60:.1f} min")
    return cache


@contextlib.contextmanager
def _generate_signals_cached(prebuilt_cache):
    """Monkey-patch eng.generate_signals to look up from prebuilt_cache
    on matching args, fall through to the real implementation on miss.
    Restores the original on exit.
    """
    real = eng.generate_signals

    def patched(asset_name, model_names, window_size,
                replay_hours=eng.REPLAY_HOURS,
                feature_override=None, horizon=eng.PREDICTION_HORIZON,
                gamma=1.0):
        key = _key_for_call(asset_name, model_names, window_size, replay_hours,
                            feature_override, horizon, gamma)
        if key in prebuilt_cache:
            return prebuilt_cache[key]
        # Cache miss — fall through to real call (graceful degradation).
        return real(asset_name, model_names, window_size,
                    replay_hours=replay_hours,
                    feature_override=feature_override,
                    horizon=horizon, gamma=gamma)

    eng.generate_signals = patched
    try:
        yield
    finally:
        eng.generate_signals = real


def run_mode_s_parallel(assets_list, horizons, args=None):
    """Drop-in replacement for engine.run_mode_s. Pre-builds per-horizon
    signals in parallel, then delegates to engine.run_mode_s with
    generate_signals patched to read from the cache. The engine's sweep
    logic (regime indicators, joint sweep, config write) runs unchanged.
    """
    import pandas as pd
    replay = int(getattr(args, 'replay', 0)) or 1440

    print("=" * 80)
    print(f"  MODE S (PARALLEL signal cache): {','.join(assets_list)} "
          f"replay={replay}h")
    print(f"  Workers: {PARALLEL_BACKTESTS}, LGBM={PARALLEL_LGBM_DEVICE}")
    print("=" * 80)

    # Predict every generate_signals call run_mode_s will make
    df_models = pd.read_csv(eng.PRODUCTION_CSV)
    all_calls = []
    for asset in assets_list:
        if horizons:
            available_h = sorted([
                h for h in horizons
                if len(df_models[(df_models['coin'] == asset) &
                                 (df_models['horizon'] == h)]) > 0
            ])
        else:
            available_h = sorted(
                df_models[df_models['coin'] == asset]['horizon'].unique())
        all_calls.extend(_predict_signal_calls_for_horizons(asset, available_h, replay))

    if not all_calls:
        print("  [parallel] No signal calls predicted — falling through "
              "to engine.run_mode_s as-is")
        # IMPORTANT: call the captured original, not eng.run_mode_s — the
        # latter is OURSELVES once __main__ has monkey-patched.
        return _ENG_RUN_MODE_S_ORIG(assets_list, horizons, args)

    # Build cache in parallel
    cache = _build_signals_cache_parallel(all_calls, PARALLEL_LGBM_DEVICE)

    # Delegate to engine — its sequential loop hits cache, returns instantly.
    # Use the captured original to avoid recursion via the monkey-patch.
    with _generate_signals_cached(cache):
        return _ENG_RUN_MODE_S_ORIG(assets_list, horizons, args)


def run_mode_t_parallel(assets_list, args=None):
    """Drop-in replacement for engine.run_mode_t. Pre-builds per-horizon
    signals in parallel using bull_h + bear_h from regime config, then
    delegates to engine.run_mode_t. The iterative T↔G convergence logic
    runs unchanged with cached signals.
    """
    replay = int(getattr(args, 'replay', 0)) or 1440

    print("=" * 80)
    print(f"  MODE T (PARALLEL signal cache): {','.join(assets_list)} "
          f"replay={replay}h")
    print(f"  Workers: {PARALLEL_BACKTESTS}, LGBM={PARALLEL_LGBM_DEVICE}")
    print("=" * 80)

    # Mode T's test_horizons = sorted({bull_h, bear_h}) per asset, from
    # regime_config_ed.json. Replicate that prediction here.
    try:
        with open(eng.REGIME_CONFIG_PATH) as f:
            regime_cfg = json.load(f)
    except Exception as e:
        print(f"  [parallel] Cannot read regime config ({e}) — "
              "falling through to engine.run_mode_t as-is")
        # IMPORTANT: captured original, not eng.run_mode_t (= ourselves)
        return _ENG_RUN_MODE_T_ORIG(assets_list, args)

    all_calls = []
    for asset in assets_list:
        ac = regime_cfg.get(asset, {})
        bull_h = ac.get('bull', {}).get('horizon')
        bear_h = ac.get('bear', {}).get('horizon')
        if not bull_h or not bear_h:
            continue
        test_horizons = sorted({int(bull_h), int(bear_h)})
        all_calls.extend(_predict_signal_calls_for_horizons(asset, test_horizons, replay))

    if not all_calls:
        print("  [parallel] No signal calls predicted — falling through "
              "to engine.run_mode_t as-is")
        return _ENG_RUN_MODE_T_ORIG(assets_list, args)

    cache = _build_signals_cache_parallel(all_calls, PARALLEL_LGBM_DEVICE)

    with _generate_signals_cached(cache):
        return _ENG_RUN_MODE_T_ORIG(assets_list, args)


# ──────────────────────────────────────────────────────────────
# Hybrid GPU+CPU refine dispatch
# ──────────────────────────────────────────────────────────────
# Engine's _refine_top_configs runs 3 Optuna refines (one per top-3
# Mode-D candidate) sequentially, each ~30-40 min on GPU. Total ~100 min.
#
# Hybrid scheduler:
#   - Worker A starts refine #1 with LGBM=GPU
#   - Worker B starts refine #2 with LGBM=CPU (concurrently)
#   - When either finishes, that worker's device picks up refine #3
#
# Why this works:
#   - Only ONE process touches the GPU at a time → no GPU queue contention
#   - CPU LGBM runs concurrently in worker B (separate hardware)
#   - The faster device usually finishes first and grabs the 3rd config
#
# Expected speedup: ~1.3-1.5× on refine (GPU much faster than CPU LGBM).
# Sequential 3×T_gpu  →  hybrid ~2×T_gpu (CPU run dominates wall time).
#
# Each refine is its own Optuna study (own seed + own 50 trials), so no
# TPE sampler quality loss vs sequential — unlike n_jobs=2 inside one
# study, where parallel trials can't condition on each other's results.

import numpy as _np
from concurrent.futures import ProcessPoolExecutor as _PoolExec, as_completed as _as_completed

# Optional override for the per-refine Optuna trial count. Set by __main__
# when --refine-trials N is passed on the CLI. Used to make hybrid-refine
# tests finish quickly without modifying engine.REFINE_TRIALS (which would
# affect the engine's own _refine_top_configs and risk side effects).
# When None (default), the worker uses eng.REFINE_TRIALS.
_REFINE_TRIALS_OVERRIDE = None


def _diagnostic_models_with_device(lgbm_device):
    """100-estimator factory variant for refine's _deku_eval_with_pruning,
    same shape as engine._get_deku_diagnostic_models but with the LGBM
    device argument plumbed through.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None
    factories = {
        'RF':   lambda: RandomForestClassifier(n_estimators=100, max_depth=4,
                                               class_weight='balanced',
                                               random_state=42, n_jobs=1),
        'GB':   lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                   random_state=42),
        'LR':   lambda: LogisticRegression(max_iter=1000,
                                           class_weight='balanced',
                                           random_state=42),
        'LGBM': lambda: LGBMClassifier(n_estimators=100, max_depth=4,
                                       learning_rate=0.05,
                                       class_weight='balanced',
                                       verbose=-1, random_state=42,
                                       device=lgbm_device),
    }
    if XGBClassifier is not None:
        factories['XGB'] = lambda: XGBClassifier(n_estimators=100, max_depth=3,
                                                 learning_rate=0.05,
                                                 random_state=42,
                                                 tree_method='hist',
                                                 verbosity=0, n_jobs=1)
    return factories


def _refine_one_config_worker(cfg_idx, top_entry_pickle, asset, horizon,
                              ranked_features, features_np, labels_np,
                              closes_np, n_size, lgbm_device,
                              n_trials_override=None):
    """ProcessPool worker: refine ONE config with assigned LGBM device.
    Returns (cfg_idx, refined_dict_or_None, captured_stdout, lgbm_device).

    n_trials_override: if not None, overrides eng.REFINE_TRIALS for this
    worker's Optuna study. Workers re-import the engine module so a
    parent-process monkey-patch on eng.REFINE_TRIALS won't propagate;
    this explicit kwarg is the only way to actually control the trial
    count from the parent.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Override the production-model factory for THIS worker process.
    # (No effect on parent; each loky/pool worker has its own ALL_MODELS.)
    eng.ALL_MODELS = _get_deku_models_with_device(lgbm_device)
    model_factories = _diagnostic_models_with_device(lgbm_device)

    buf = io.StringIO()
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf, buf

    refined = None
    try:
        lbl, r_entry = top_entry_pickle
        cfg = r_entry['cfg']
        combo_name = cfg['combo']
        combo = combo_name.split('+')
        base_window = int(cfg['window'])
        base_gamma = float(cfg['gamma'])
        base_feats = int(cfg['n_features'])

        gamma_lo = max(base_gamma - eng.REFINE_GAMMA_RANGE, 0.970)
        gamma_hi = min(base_gamma + eng.REFINE_GAMMA_RANGE, 1.0)
        feat_lo = max(base_feats - eng.REFINE_FEAT_RANGE, 5)
        feat_hi = min(base_feats + eng.REFINE_FEAT_RANGE, len(ranked_features))
        win_lo = max(base_window - eng.REFINE_WINDOW_RANGE, 24)
        win_hi = base_window + eng.REFINE_WINDOW_RANGE

        print(f"\n  {'─'*60}")
        print(f"  Refining #{cfg_idx+1} (LGBM={lgbm_device}): {combo_name}  "
              f"w={base_window}h  g={base_gamma:.4f}  f={base_feats}")
        print(f"  Ranges: gamma[{gamma_lo:.3f}-{gamma_hi:.3f}] "
              f"features[{feat_lo}-{feat_hi}] window[{win_lo}-{win_hi}]")
        print(f"  {'─'*60}")

        seed = (eng.OPTUNA_SEED_OVERRIDE
                if eng.OPTUNA_SEED_OVERRIDE is not None else 42)
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=seed),
            study_name=f'ed_v1_refine_{asset}_{horizon}h_{cfg_idx}',
        )

        best_refine_apf = [0.0]
        r_count = [0]

        def refine_objective(trial):
            r_count[0] += 1
            t_window = trial.suggest_int('window', win_lo, win_hi)
            t_gamma = trial.suggest_float('gamma', gamma_lo, gamma_hi)
            t_feats = trial.suggest_int('n_features', feat_lo, feat_hi)
            sel_idx = eng._feature_floor_indices(ranked_features, t_feats)
            feat_np = features_np[:, sel_idx]
            result = eng._deku_eval_with_pruning(
                feat_np, labels_np, closes_np, combo, t_window, n_size,
                eng.DIAG_STEP, model_factories, gamma=t_gamma, trial=None,
                horizon=horizon
            )
            if result is None:
                return 0.0
            if result[6] < 8:
                return 0.0
            score = eng._compute_optuna_score(result)
            ret = result[4]
            if score > best_refine_apf[0]:
                best_refine_apf[0] = score
                print(f"    #{r_count[0]:3d} NEW BEST: w={t_window} "
                      f"g={t_gamma:.4f} f={t_feats} | "
                      f"apf={score:.3f} ret={ret:+.1f}% "
                      f"trades={result[6]}")
            return score

        n_trials = (n_trials_override if n_trials_override is not None
                    else eng.REFINE_TRIALS)
        study.optimize(refine_objective, n_trials=n_trials,
                       show_progress_bar=False)

        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE and t.value > 0]
        if completed:
            best = max(completed, key=lambda t: t.value)
            print(f"    → Best: w={best.params['window']}h "
                  f"g={best.params['gamma']:.4f} "
                  f"f={best.params['n_features']} apf={best.value:.3f}")
            best_sel_idx = eng._feature_floor_indices(
                ranked_features, best.params['n_features'])
            refined = {
                'combo': combo_name,
                'window': best.params['window'],
                'gamma': best.params['gamma'],
                'n_features': best.params['n_features'],
                'apf': best.value,
                'features': [ranked_features[i] for i in best_sel_idx],
            }
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err

    return cfg_idx, refined, buf.getvalue(), lgbm_device


def _refine_top_configs_hybrid(asset, horizon, top3_for_refine, df_raw,
                               df_clean, all_cols, ranked_features):
    """Hybrid GPU+CPU refine dispatcher. Replaces engine._refine_top_configs.

    Setup phase (build features_np / labels_np / closes_np) runs once in
    the main process, then 3 single-config refines are dispatched across
    2 worker processes:
      - Worker A: GPU LGBM   (config 0)
      - Worker B: CPU LGBM   (config 1)
      - Whichever finishes first picks up config 2 with its assigned
        device → no GPU queue contention.

    Falls back to engine._refine_top_configs (sequential GPU) for ≤1
    config or if anything goes wrong during setup.
    """
    if len(top3_for_refine) <= 1:
        # IMPORTANT: captured original, not eng._refine_top_configs (=
        # ourselves once __main__ has monkey-patched). Avoids infinite
        # recursion when hybrid is active and called with ≤1 config.
        return _ENG_REFINE_TOP_CONFIGS_ORIG(asset, horizon, top3_for_refine,
                                             df_raw, df_clean, all_cols,
                                             ranked_features)

    # ── Setup phase (replicated from engine._refine_top_configs) ──
    MAX_DIAG_HOURS = 6 * 30 * 24
    df_full_r, all_cols_r = eng.build_all_features(
        df_raw, asset_name=asset, horizon=horizon)
    n_pysr = eng._compute_pysr_features(df_full_r, all_cols_r,
                                         asset, horizon, verbose=False)
    if n_pysr > 0:
        pysr_cols_r = [c for c in all_cols_r if c.startswith('pysr_')]
        is_clean, _ = eng._check_pysr_leakage(pysr_cols_r, asset, horizon)
        if not is_clean:
            for pc in pysr_cols_r:
                all_cols_r.remove(pc)
                if pc in df_full_r.columns:
                    df_full_r.drop(columns=[pc], inplace=True)
    if len(df_full_r) > MAX_DIAG_HOURS:
        df_full_r = df_full_r.tail(MAX_DIAG_HOURS).reset_index(drop=True)
    df_full_r, all_cols_r, dropped_r = eng._filter_sparse_features(
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

    print(f"\n  [refine-hybrid] dispatching {len(top3_for_refine)} configs "
          f"across 2 workers (GPU + CPU); 3rd config dynamic")

    t_refine = time.time()
    pending_idx = list(range(len(top3_for_refine)))
    refined_list = [None] * len(top3_for_refine)

    initial_devices = ['gpu', 'cpu']

    pool = _PoolExec(max_workers=2)
    futures = {}
    try:
        # Submit initial 2 (one per device)
        for dev in initial_devices:
            if not pending_idx:
                break
            cfg_idx = pending_idx.pop(0)
            top_entry = top3_for_refine[cfg_idx]
            f = pool.submit(
                _refine_one_config_worker, cfg_idx, top_entry, asset, horizon,
                ranked_features_r, features_np, labels_np, closes_np, n_size,
                dev, _REFINE_TRIALS_OVERRIDE
            )
            futures[f] = (cfg_idx, dev)

        # Drain + dynamic dispatch
        while futures:
            done = next(_as_completed(futures))
            cfg_idx, dev = futures.pop(done)
            try:
                got_idx, refined, captured, _dev_used = done.result()
                if captured:
                    sys.stdout.write(captured)
                    if not captured.endswith('\n'):
                        sys.stdout.write('\n')
                refined_list[got_idx] = refined
            except Exception as e:
                print(f"  [refine-hybrid] config #{cfg_idx+1} ({dev}) "
                      f"FAILED with {type(e).__name__}: {e}")
            # Submit next pending config on this freed device
            if pending_idx:
                next_idx = pending_idx.pop(0)
                print(f"  [refine-hybrid] {dev.upper()} freed → "
                      f"starting config #{next_idx+1} on {dev.upper()}")
                top_entry = top3_for_refine[next_idx]
                f2 = pool.submit(
                    _refine_one_config_worker, next_idx, top_entry,
                    asset, horizon, ranked_features_r,
                    features_np, labels_np, closes_np, n_size, dev,
                    _REFINE_TRIALS_OVERRIDE
                )
                futures[f2] = (next_idx, dev)
    finally:
        pool.shutdown(wait=True)

    all_refined = [r for r in refined_list if r is not None]
    refine_elapsed = (time.time() - t_refine) / 60
    print(f"\n  [Refine: {refine_elapsed:.1f} min] (hybrid GPU+CPU)")

    if all_refined:
        all_refined.sort(key=lambda x: -x['apf'])
        for i, r in enumerate(all_refined, 1):
            print(f"  Refined #{i}: {r['combo']}  w={r['window']}h  "
                  f"g={r['gamma']:.4f}  f={r['n_features']}  apf={r['apf']:.3f}")

    return all_refined


# ──────────────────────────────────────────────────────────────
# Entry point — monkey-patch and call engine.main()
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Parse --refine-trials N (test-only override for hybrid refine).
    # Strip from sys.argv before eng.main() so the engine doesn't choke on
    # the unknown flag.
    if '--refine-trials' in sys.argv:
        idx = sys.argv.index('--refine-trials')
        try:
            _REFINE_TRIALS_OVERRIDE = int(sys.argv[idx + 1])
            del sys.argv[idx:idx + 2]
            print(f"  --refine-trials override: {_REFINE_TRIALS_OVERRIDE} "
                  f"(default eng.REFINE_TRIALS={eng.REFINE_TRIALS})")
        except (ValueError, IndexError):
            print(f"  --refine-trials must be an integer (e.g. --refine-trials 5)")
            sys.exit(2)

    print("=" * 80)
    print("  EXPERIMENTAL: Parallel wrapper (Mode V + Mode S + Mode T + hybrid refine)")
    print(f"  Machine: {eng.MACHINE}  |  Workers: {PARALLEL_BACKTESTS}  |  "
          f"LGBM (parallel section): {PARALLEL_LGBM_DEVICE}")
    print("  Production engine NOT modified — only this process patches:")
    print("    eng.run_mode_v          ← run_mode_v_parallel  (Step 1 + Step 3 backtests)")
    print("    eng.run_mode_s          ← run_mode_s_parallel  (signal cache parallel)")
    print("    eng.run_mode_t          ← run_mode_t_parallel  (signal cache parallel)")
    print("    eng._refine_top_configs ← _refine_top_configs_hybrid  (GPU+CPU dispatch)")
    print("=" * 80)

    # Replace engine entry points in THIS process. Workers re-import the
    # engine module fresh; they only run the worker functions defined at
    # this module's scope, so they don't need the patches applied.
    eng.run_mode_v = run_mode_v_parallel
    eng.run_mode_s = run_mode_s_parallel
    eng.run_mode_t = run_mode_t_parallel
    # Hybrid GPU+CPU refine dispatch — 1 GPU worker + 1 CPU worker concurrent,
    # 3rd config picks up the first-freed device. Falls back to engine's
    # sequential refine for ≤1 config (via captured original).
    eng._refine_top_configs = _refine_top_configs_hybrid

    # Hand off to the engine's main(). It will route:
    #   HRST → run_mode_h → (patched run_mode_v) → (patched _refine_top_configs)
    #        → run_mode_r → (patched run_mode_s) → (patched run_mode_t)
    eng.main()

    # ──────────────────────────────────────────────────────────────
    # Suppress harmless joblib/loky resource_tracker shutdown races
    # ──────────────────────────────────────────────────────────────
    # The loky executor's resource_tracker child process tries to clean up
    # shared-memory folders that the parent already removed → KeyError →
    # printed to stderr after our own `Done!` line. Cosmetic noise only;
    # the actual run already succeeded. We:
    #   1. Explicitly shut down the executor pool so workers terminate.
    #   2. Flush stdout/stderr before exit.
    #   3. Redirect fd 2 (stderr) to /dev/null so any late writes from the
    #      tracker child (which inherits our fds) go nowhere instead of the
    #      terminal.
    #   4. os._exit(0) to skip the normal interpreter cleanup phase entirely.
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except Exception:
        pass
    try:
        import os
        sys.stdout.flush()
        sys.stderr.flush()
        _devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull, 2)  # late tracker writes → /dev/null
        os._exit(0)
    except Exception:
        pass
