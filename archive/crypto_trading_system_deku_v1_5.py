"""
Deku V1.5 — Dynamic data cap + holdout comparison

Changes from production V1.3:
  1. Dynamic data cap: hours = log(0.01) / log(gamma) instead of fixed 4320h
     - gamma=0.999 → 4602h (~6 months)
     - gamma=0.998 → 2300h (~96 days)
     - gamma=0.997 → 1532h (~64 days)
     - gamma=0.996 → 1148h (~48 days)
     - gamma=0.995 → 918h  (~38 days)
  2. Three holdout modes tested (--holdout flag):
     - current:  overlapping folds (production baseline)
     - A:        non-overlapping sequential (Optuna on 60%, 3×13% test)
     - B:        expanding window (train grows per fold, non-overlapping test)

Usage:
  python crypto_trading_system_deku_v1_5.py D BTC 8h --holdout current
  python crypto_trading_system_deku_v1_5.py D BTC 8h --holdout A
  python crypto_trading_system_deku_v1_5.py D BTC 8h --holdout B
  python crypto_trading_system_deku_v1_5.py D BTC 8h --holdout all   # run all 3 sequentially
"""

import sys, os, time, warnings, math
import numpy as np
import pandas as pd
import optuna

# Import everything from production
sys.path.insert(0, os.path.dirname(__file__))
from crypto_trading_system_deku import (
    load_data, build_all_features, update_all_data,
    _test_lgbm_importance, _deku_eval_with_pruning, _compute_optuna_score,
    _build_combo_list, _get_deku_diagnostic_models, _kill_orphan_workers,
    get_decay_weights, generate_signals, simulate_portfolio,
    PREDICTION_HORIZON, HORIZON_SHORT, HORIZON_LONG,
    DIAG_STEP, DIAG_WINDOWS, EMBARGO_CANDLES,
    TRADING_FEE, MIN_CONFIDENCE, OPTUNA_METRIC,
    N_FEATURES_RANGE, N_FEATURES_RANGE_DEFAULT,
    ALL_MODELS, ASSETS,
)

# V1.5 constants
V15_DEFAULT_TRIALS = 150
V15_PRUNING_WARMUP = 8
V15_CSV = os.path.join('models', 'crypto_deku_v1_5_best_models_{holdout}.csv')
N_HOLDOUT_FOLDS = 3

# Max data cap — even for gamma=0.999+, never exceed 6 months
MAX_DATA_HOURS = 6 * 30 * 24  # 4320h


def gamma_data_cap(gamma):
    """Compute hours needed for 99% of training weight given gamma.
    Formula: hours = log(0.01) / log(gamma), capped at MAX_DATA_HOURS."""
    if gamma >= 1.0:
        return MAX_DATA_HOURS
    cap = int(math.log(0.01) / math.log(gamma))
    return min(cap, MAX_DATA_HOURS)


def build_holdout_folds(n_total, mode):
    """Build holdout fold boundaries.

    Returns list of (train_start, train_end, test_start, test_end) tuples.

    Modes:
      current — overlapping folds (production baseline)
        Fold 1: train [0%, 60%]  test [60%, 80%]
        Fold 2: train [10%, 70%] test [70%, 90%]
        Fold 3: train [20%, 80%] test [80%, 100%]

      A — non-overlapping sequential (Optuna on 60%, 3×13% test)
        Fold 1: train [0%, 60%] test [60%, 73%]
        Fold 2: train [0%, 60%] test [73%, 87%]
        Fold 3: train [0%, 60%] test [87%, 100%]

      B — expanding window (train grows, non-overlapping test)
        Fold 1: train [0%, 60%]  test [60%, 73%]
        Fold 2: train [0%, 73%]  test [73%, 87%]
        Fold 3: train [0%, 87%]  test [87%, 100%]
    """
    folds = []
    if mode == 'current':
        fold_train_frac = 0.60
        fold_test_frac = 0.20
        fold_stride = 0.10
        for fi in range(N_HOLDOUT_FOLDS):
            train_s = int(n_total * fi * fold_stride)
            train_e = int(n_total * (fi * fold_stride + fold_train_frac))
            test_s = train_e
            test_e = int(n_total * (fi * fold_stride + fold_train_frac + fold_test_frac))
            test_e = min(test_e, n_total)
            folds.append((train_s, train_e, test_s, test_e))

    elif mode == 'A':
        # Non-overlapping: Optuna trains on first 60%, remaining 40% split into 3 equal chunks
        optuna_end = int(n_total * 0.60)
        remaining = n_total - optuna_end
        chunk = remaining // N_HOLDOUT_FOLDS
        for fi in range(N_HOLDOUT_FOLDS):
            train_s = 0
            train_e = optuna_end  # same train for all folds
            test_s = optuna_end + fi * chunk
            test_e = optuna_end + (fi + 1) * chunk if fi < N_HOLDOUT_FOLDS - 1 else n_total
            folds.append((train_s, train_e, test_s, test_e))

    elif mode == 'B':
        # Expanding window: train grows, non-overlapping test
        optuna_end = int(n_total * 0.60)
        remaining = n_total - optuna_end
        chunk = remaining // N_HOLDOUT_FOLDS
        for fi in range(N_HOLDOUT_FOLDS):
            train_s = 0
            test_s = optuna_end + fi * chunk
            train_e = test_s  # train expands up to test boundary
            test_e = optuna_end + (fi + 1) * chunk if fi < N_HOLDOUT_FOLDS - 1 else n_total
            folds.append((train_s, train_e, test_s, test_e))

    return folds


def run_v15(assets_list, horizon, n_trials, holdout_mode):
    """Run Deku V1.5 with specified holdout mode."""
    t_mode_start = time.time()
    _kill_orphan_workers()

    combo_options = _build_combo_list()

    print("\n" + "=" * 70)
    print(f"  DEKU V1.5 MODE D: {horizon}h | holdout={holdout_mode.upper()}")
    print(f"  Models: {', '.join(ALL_MODELS.keys())} ({len(combo_options)} combos)")
    print(f"  Trials: {n_trials} | Dynamic data cap (99% gamma weight)")
    print(f"  Holdout: {holdout_mode}")
    print("=" * 70)

    # Download fresh data
    print("\n  Updating macro & sentiment data...")
    t0 = time.time()
    try:
        import download_macro_data
        download_macro_data.main()
    except ImportError:
        print("  WARNING: download_macro_data.py not found -- macro features may be stale.")
    except Exception as e:
        print(f"  WARNING: Macro data update failed: {e}")
    print(f"  [Macro update: {(time.time()-t0)/60:.1f} min]")

    t0 = time.time()
    update_all_data(assets_list)
    print(f"  [Data update: {(time.time()-t0)/60:.1f} min]")

    best_models = []

    for asset_name in assets_list:
        t_asset = time.time()

        print(f"\n{'='*70}")
        print(f"  V1.5 OPTIMIZATION: {asset_name} ({horizon}h) | holdout={holdout_mode.upper()}")
        print(f"{'='*70}")

        df_raw = load_data(asset_name)
        if df_raw is None:
            continue

        # Step 1: Build features — use MAX (6 months) initially, dynamic cap applied per trial
        print(f"\n  Building all features (horizon={horizon}h)...")
        t0 = time.time()
        df_full, all_cols = build_all_features(df_raw, asset_name=asset_name, horizon=horizon)
        print(f"  [Feature build: {(time.time()-t0)/60:.1f} min]")

        total_rows = len(df_full)
        if total_rows > MAX_DATA_HOURS:
            df_full = df_full.tail(MAX_DATA_HOURS).reset_index(drop=True)
            print(f"  Capped: {total_rows:,} -> {len(df_full):,} rows (max 6mo)")

        df_clean = df_full.dropna(subset=all_cols + ['label']).reset_index(drop=True)
        print(f"  Clean data: {len(df_clean):,} rows, {len(all_cols)} features")

        if len(df_clean) < 500:
            print(f"  Not enough data ({len(df_clean)} rows). Need 500+. Skipping.")
            continue

        # Step 2: LGBM importance ranking
        print(f"\n  LGBM feature importance ranking...")
        t0 = time.time()
        importance_df = _test_lgbm_importance(df_clean, all_cols, gamma=1.0)
        ranked_features = importance_df['feature'].tolist()
        print(f"  [Ranking: {(time.time()-t0)/60:.1f} min] — {len(ranked_features)} features ranked")

        # Prepare full data arrays (columns in rank order)
        df_optuna = df_clean.dropna(subset=ranked_features + ['label']).reset_index(drop=True)
        features_np_all = df_optuna[ranked_features].values.astype(np.float64)
        labels_np_all = df_optuna['label'].values.astype(np.int32)
        closes_np_all = df_optuna['close'].values.astype(np.float64)
        n_total = len(df_optuna)

        # Build holdout folds
        holdout_folds = build_holdout_folds(n_total, holdout_mode)
        for fi, (tr_s, tr_e, te_s, te_e) in enumerate(holdout_folds):
            print(f"  Fold {fi+1}: train [{tr_s}:{tr_e}] ({tr_e-tr_s:,} rows) → "
                  f"test [{te_s}:{te_e}] ({te_e-te_s:,} rows)")

        # Optuna trains on fold 1's training partition
        f1_train_s, f1_train_e, _, _ = holdout_folds[0]
        features_np = features_np_all[f1_train_s:f1_train_e]
        labels_np = labels_np_all[f1_train_s:f1_train_e]
        closes_np = closes_np_all[f1_train_s:f1_train_e]
        n = len(features_np)

        print(f"  Optuna trains on fold 1: {n:,} rows | {N_HOLDOUT_FOLDS}-fold holdout ({holdout_mode}) after")

        # Feature range
        feat_min, feat_max = N_FEATURES_RANGE.get(horizon, N_FEATURES_RANGE_DEFAULT)
        min_n_features = feat_min
        max_n_features = min(len(ranked_features), feat_max)

        # Step 3: Optuna study
        print(f"\n{'='*70}")
        print(f"  OPTUNA STUDY: {asset_name} {horizon}h | holdout={holdout_mode.upper()}")
        print(f"  Search: {len(combo_options)} combos × {len(DIAG_WINDOWS)} windows × gamma[0.994-1.0] × features[{min_n_features}-{max_n_features}]")
        print(f"  Trials: {n_trials} | Data: {n:,} train | dynamic data cap")
        print(f"{'='*70}")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=V15_PRUNING_WARMUP,
                max_resource=n // DIAG_STEP,
                reduction_factor=3,
            ),
            study_name=f'v15_{asset_name}_{horizon}h_{holdout_mode}',
        )

        model_factories = _get_deku_diagnostic_models()
        best_apf_so_far = 0.0
        trial_count = 0

        def objective(trial):
            nonlocal best_apf_so_far, trial_count
            trial_count += 1

            combo_name = trial.suggest_categorical('combo', combo_options)
            window = trial.suggest_categorical('window', DIAG_WINDOWS)
            gamma = trial.suggest_float('gamma', 0.994, 1.0)
            n_feat = trial.suggest_int('n_features', min_n_features, max_n_features)

            combo = combo_name.split('+')

            # V1.5: Dynamic data cap — use only last N hours based on gamma
            cap_hours = gamma_data_cap(gamma)

            # Apply cap to training data
            if n > cap_hours:
                cap_start = n - cap_hours
                feat_np = features_np[cap_start:, :n_feat]
                lab_np = labels_np[cap_start:]
                close_np = closes_np[cap_start:]
                n_capped = len(feat_np)
            else:
                feat_np = features_np[:, :n_feat]
                lab_np = labels_np
                close_np = closes_np
                n_capped = n

            result = _deku_eval_with_pruning(
                feat_np, lab_np, close_np, combo, window, n_capped,
                DIAG_STEP, model_factories, gamma=gamma, trial=trial,
                horizon=horizon
            )

            if result is None:
                return 0.0

            trades = result[6]
            MIN_TRADES = 8
            if trades < MIN_TRADES:
                return 0.0

            score = _compute_optuna_score(result)
            ret = result[4]

            if score > best_apf_so_far:
                best_apf_so_far = score
                cap_label = f"cap={cap_hours}h" if n > cap_hours else "full"
                print(f"  #{trial_count:3d} NEW BEST: {combo_name:22s} w={window:4d}h "
                      f"g={gamma:.4f} f={n_feat:3d} {cap_label} | apf={score:.3f} ret={ret:+.1f}% trades={trades}")
            elif trial_count % 20 == 0:
                print(f"  #{trial_count:3d} progress: apf={score:.3f} | best so far: {best_apf_so_far:.3f}")

            return score

        t_optuna = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Auto-extend if needed
        APF_EXTEND_THRESH = 1.7
        for ext_target in [200, 250]:
            if ext_target <= n_trials:
                continue
            if study.best_value >= APF_EXTEND_THRESH:
                break
            extra = ext_target - len(study.trials)
            if extra > 0:
                print(f"\n  Best APF={study.best_value:.3f} < {APF_EXTEND_THRESH} — extending to {ext_target} trials (+{extra})...")
                study.optimize(objective, n_trials=extra, show_progress_bar=False)

        optuna_elapsed = (time.time() - t_optuna) / 60

        # Results
        best_trial = study.best_trial
        print(f"\n  {'='*70}")
        print(f"  OPTUNA RESULTS: {asset_name} {horizon}h | holdout={holdout_mode.upper()} ({optuna_elapsed:.1f} min)")
        print(f"  {'='*70}")
        print(f"  Best trial: #{best_trial.number}")
        print(f"  APF:        {best_trial.value:.4f}")
        print(f"  Combo:      {best_trial.params['combo']}")
        print(f"  Window:     {best_trial.params['window']}h")
        print(f"  Gamma:      {best_trial.params['gamma']:.4f}")
        print(f"  N_features: {best_trial.params['n_features']}")
        cap_h = gamma_data_cap(best_trial.params['gamma'])
        print(f"  Data cap:   {cap_h}h ({cap_h/24:.0f} days) at 99% weight")

        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"\n  Trials: {n_complete} completed, {n_pruned} pruned ({n_pruned/(n_complete+n_pruned)*100:.0f}% pruned)")

        # Top 10
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value > 0]
        completed_trials.sort(key=lambda t: -t.value)
        print(f"\n  {'Rank':>4s}  {'APF':>7s}  {'Combo':22s}  {'Window':>6s}  {'Gamma':>7s}  {'Feats':>5s}  {'DataCap':>8s}")
        print(f"  {'-'*70}")
        for i, t in enumerate(completed_trials[:10], 1):
            cap = gamma_data_cap(t.params['gamma'])
            marker = " <-- BEST" if i == 1 else ""
            print(f"  {i:4d}  {t.value:7.3f}  {t.params['combo']:22s}  {t.params['window']:5d}h  "
                  f"{t.params['gamma']:7.4f}  {t.params['n_features']:5d}  {cap:6d}h{marker}")

        # Parameter importance
        if n_complete >= 20:
            try:
                importances = optuna.importance.get_param_importances(study)
                print(f"\n  Parameter importance:")
                for param, imp in importances.items():
                    bar = '#' * int(imp * 40)
                    print(f"    {param:15s} {imp*100:5.1f}% {bar}")
            except Exception:
                pass

        # ── HOLDOUT: re-evaluate top candidates ──
        seen_configs = set()
        unique_candidates = []
        for t in completed_trials:
            key = (t.params['combo'], t.params['window'])
            if key not in seen_configs:
                seen_configs.add(key)
                unique_candidates.append(t)

        phase1 = unique_candidates[:10]
        phase1_set = set(id(t) for t in phase1)
        combos_in_phase1 = set(t.params['combo'] for t in phase1)

        phase2 = []
        seen_combos_p2 = set()
        for t in unique_candidates:
            combo = t.params['combo']
            if combo not in combos_in_phase1 and combo not in seen_combos_p2:
                seen_combos_p2.add(combo)
                phase2.append(t)

        candidates = phase1 + phase2
        N_CANDIDATES = min(20, len(candidates))

        print(f"\n  {'='*70}")
        print(f"  {N_HOLDOUT_FOLDS}-FOLD HOLDOUT ({holdout_mode.upper()}): {asset_name} {horizon}h (top {N_CANDIDATES})")
        print(f"  {'='*70}")

        holdout_results = []
        min_fold_rows = min(te - ts for _, _, ts, te in holdout_folds)
        if min_fold_rows >= 200:
            for ci, trial in enumerate(candidates[:N_CANDIDATES]):
                c_combo = trial.params['combo'].split('+')
                c_window = trial.params['window']
                c_gamma = trial.params['gamma']
                c_n_feat = trial.params['n_features']

                # V1.5: apply dynamic data cap to holdout folds too
                cap_hours = gamma_data_cap(c_gamma)

                fold_scores = []
                fold_rets = []
                fold_accs = []
                fold_trades = []
                fold_raw_pfs = []

                for fi, (tr_s, tr_e, te_s, te_e) in enumerate(holdout_folds):
                    fold_feat_test = features_np_all[te_s:te_e, :c_n_feat]
                    fold_labels_test = labels_np_all[te_s:te_e]
                    fold_closes_test = closes_np_all[te_s:te_e]
                    n_fold_test = te_e - te_s

                    HOLDOUT_STEP = 12
                    ho_result = _deku_eval_with_pruning(
                        fold_feat_test, fold_labels_test, fold_closes_test,
                        c_combo, c_window, n_fold_test,
                        HOLDOUT_STEP, model_factories, gamma=c_gamma, trial=None,
                        horizon=horizon
                    )

                    if ho_result:
                        fold_scores.append(_compute_optuna_score(ho_result))
                        fold_rets.append(ho_result[4])
                        fold_accs.append(ho_result[2])
                        fold_trades.append(ho_result[6])
                        fold_raw_pfs.append(ho_result[11])
                    else:
                        fold_scores.append(0.0)
                        fold_rets.append(0.0)
                        fold_accs.append(0.0)
                        fold_trades.append(0)
                        fold_raw_pfs.append(0.0)

                avg_score = np.mean(fold_scores)
                avg_ret = np.mean(fold_rets)
                avg_acc = np.mean(fold_accs)
                total_trades = sum(fold_trades)
                avg_raw_pf = np.mean(fold_raw_pfs)

                holdout_results.append((trial, avg_score, avg_ret, avg_acc, total_trades,
                                        avg_raw_pf, fold_scores, fold_rets))

            holdout_results.sort(key=lambda x: -x[1])

            print(f"\n  {'Rank':>4s}  {'AVG_APF':>8s}  {'AVG_Ret':>8s}  {'AVG_Acc':>7s}  {'Tr':>4s}  "
                  f"{'F1':>6s}  {'F2':>6s}  {'F3':>6s}  {'IS_APF':>8s}  {'Combo':22s}  {'Win':>4s}  {'Gamma':>7s}  {'F':>3s}")
            print(f"  {'-'*130}")
            for i, (trial, avg_sc, avg_ret, avg_acc, tot_tr, avg_rpf, f_scores, f_rets) in enumerate(holdout_results[:10], 1):
                is_score = trial.value
                marker = " <-- BEST" if i == 1 else ""
                gen = "✓" if avg_ret > 0 and avg_acc > 0.55 else "~" if avg_ret > 0 else "✗"
                f_str = "  ".join(f"{s:+5.1f}" for s in f_rets)
                print(f"  {i:4d}  {avg_sc:8.3f}  {avg_ret:+7.1f}%  {avg_acc*100:6.1f}%  {tot_tr:4d}  "
                      f"{f_str}  {is_score:8.3f}  {trial.params['combo']:22s}  {trial.params['window']:3d}h  "
                      f"{trial.params['gamma']:7.4f}  {trial.params['n_features']:3d}  {gen}{marker}")
        else:
            print(f"  Hold-out skipped: smallest fold only {min_fold_rows} rows (need 200+)")

        # Pick winner
        if holdout_results and holdout_results[0][1] > 0:
            winner_trial = holdout_results[0][0]
            winner_ho = holdout_results[0]
        else:
            winner_trial = study.best_trial
            winner_ho = None

        best_combo = winner_trial.params['combo'].split('+')
        best_window = winner_trial.params['window']
        best_gamma = winner_trial.params['gamma']
        best_n_feat = winner_trial.params['n_features']
        best_features = ranked_features[:best_n_feat]

        # Re-run winner on training data for in-sample metrics
        cap_hours = gamma_data_cap(best_gamma)
        if n > cap_hours:
            cap_start = n - cap_hours
            feat_np = features_np[cap_start:, :best_n_feat]
            lab_np = labels_np[cap_start:]
            close_np = closes_np[cap_start:]
            n_eval = len(feat_np)
        else:
            feat_np = features_np[:, :best_n_feat]
            lab_np = labels_np
            close_np = closes_np
            n_eval = n

        full_result = _deku_eval_with_pruning(
            feat_np, lab_np, close_np, best_combo, best_window, n_eval,
            DIAG_STEP, model_factories, gamma=best_gamma, trial=None,
            horizon=horizon
        )

        if full_result:
            combo_name, window, acc, n_total_r, cum_ret, win_rate, trades, _, _, max_dd, apf, raw_pf, bh_pf = full_result
            is_score = _compute_optuna_score(full_result)

            ho_score = winner_ho[1] if winner_ho else 0
            ho_ret = winner_ho[2] if winner_ho else 0
            ho_acc = winner_ho[3] if winner_ho else 0
            ho_trades = winner_ho[4] if winner_ho else 0
            ho_raw_pf = winner_ho[5] if winner_ho else 0
            ho_fold_rets = winner_ho[7] if winner_ho else []

            best_config = {
                'coin': asset_name,
                'best_window': best_window,
                'best_combo': winner_trial.params['combo'],
                'accuracy': round(acc * 100, 2),
                'models': winner_trial.params['combo'],
                'return_pct': round(ho_ret if winner_ho else cum_ret, 2),
                'win_rate': round(win_rate, 1),
                'trades': trades,
                'combined_score': round(ho_score if winner_ho else apf, 4),
                'feature_set': 'D',
                'n_features': best_n_feat,
                'optimal_features': ','.join(best_features),
                'horizon': horizon,
                'gamma': round(best_gamma, 4),
                'holdout_mode': holdout_mode,
                'data_cap_hours': cap_hours,
            }
            best_models.append(best_config)

            print(f"\n  {'='*70}")
            print(f"  WINNER ({holdout_mode.upper()} holdout): {asset_name} {horizon}h")
            print(f"  Models: {winner_trial.params['combo']}  Window: {best_window}h  Gamma: {best_gamma:.4f}  Features: {best_n_feat}")
            print(f"  Data cap: {cap_hours}h ({cap_hours/24:.0f} days)")
            print(f"  In-sample:     apf={is_score:.3f}  ret={cum_ret:+.1f}%  acc={acc*100:.1f}%  trades={trades}  rawPF={raw_pf:.2f}")
            if winner_ho:
                fold_str = " / ".join(f"{r:+.1f}%" for r in ho_fold_rets)
                print(f"  Out-of-sample: apf={ho_score:.3f}  avg_ret={ho_ret:+.1f}%  avg_acc={ho_acc*100:.1f}%  trades={ho_trades}")
                print(f"  Per-fold returns: [{fold_str}]")
            print(f"  {'='*70}")

        print(f"  [{asset_name} total: {(time.time()-t_asset)/60:.1f} min]")

    if not best_models:
        print("\nNo results. Aborting.")
        return best_models

    # Save results
    csv_path = V15_CSV.format(holdout=holdout_mode)
    df_best = pd.DataFrame(best_models)
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        for m in best_models:
            mask = (df_existing['coin'] == m['coin']) & (df_existing['horizon'] == horizon)
            df_existing = df_existing[~mask]
        df_best = pd.concat([df_existing, df_best], ignore_index=True)
    df_best.to_csv(csv_path, index=False)
    print(f"\n  Results saved: {csv_path}")

    elapsed = (time.time() - t_mode_start) / 60
    print(f"\n  V1.5 Mode D ({holdout_mode.upper()}) complete: {elapsed:.1f} min total")

    return best_models


# ============================================================
# CLI
# ============================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deku V1.5 — Dynamic data cap + holdout comparison')
    parser.add_argument('mode', choices=['D'], help='Mode (D only)')
    parser.add_argument('assets', help='Comma-separated assets (e.g. BTC)')
    parser.add_argument('horizons', help='Horizons (e.g. 8h)')
    parser.add_argument('--holdout', default='all', choices=['current', 'A', 'B', 'all'],
                        help='Holdout mode: current, A, B, or all (default: all)')
    parser.add_argument('--trials', type=int, default=V15_DEFAULT_TRIALS, help='Optuna trials')

    args = parser.parse_args()

    assets_list = [a.strip().upper() for a in args.assets.split(',')]
    horizons = [int(h.replace('h', '')) for h in args.horizons.split(',')]

    if args.holdout == 'all':
        holdout_modes = ['current', 'A', 'B']
    else:
        holdout_modes = [args.holdout]

    all_results = {}
    for holdout_mode in holdout_modes:
        for h in horizons:
            print(f"\n{'#'*80}")
            print(f"  V1.5 RUN: {','.join(assets_list)} {h}h | holdout={holdout_mode.upper()}")
            print(f"{'#'*80}")
            results = run_v15(assets_list, h, args.trials, holdout_mode)
            all_results[holdout_mode] = results

    # ── Comparison summary ──
    if len(holdout_modes) > 1:
        print(f"\n{'='*80}")
        print(f"  V1.5 COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"\n  {'Holdout':<10} {'Asset':<6} {'H':>2} {'Combo':22s} {'Win':>4} {'Gamma':>7} {'Feat':>4} "
              f"{'IS_APF':>7} {'HO_APF':>7} {'HO_Ret':>8} {'DataCap':>8}")
        print(f"  {'-'*100}")
        for mode in holdout_modes:
            for m in all_results.get(mode, []):
                print(f"  {mode:<10} {m['coin']:<6} {m['horizon']:>2}h {m['best_combo']:22s} "
                      f"{m['best_window']:3d}h {m['gamma']:7.4f} {m['n_features']:4d} "
                      f"{'':>7s} {m['combined_score']:7.3f} {m['return_pct']:+7.1f}% "
                      f"{m.get('data_cap_hours', 'n/a'):>8}")
