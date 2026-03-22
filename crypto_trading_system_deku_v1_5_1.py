"""
Deku V1.5.1 — Discrete features & gamma (production holdout, fixed 6-month cap)

Changes from production V1.3:
  1. Discrete feature counts: [5, 10, 13, 17, 20, 25, 30, 40] instead of continuous [4, 80]
     - Prevents Optuna from maxing out features (V1.5 showed 58-76 features = overfitting)
     - More resolution at 10-20 range where production winners cluster (BTC 8h=13, SOL 8h=17)
  2. Discrete gamma: [0.995, 0.996, 0.997, 0.998, 0.999] instead of continuous [0.994, 1.0]
     - 3 decimal places, no 4-decimal noise
  3. Same holdout and data cap as production (overlapping 3-fold, fixed 6-month)

Usage:
  python crypto_trading_system_deku_v1_5_1.py D BTC 8h
  python crypto_trading_system_deku_v1_5_1.py D BTC 8h --trials 150
"""

import sys, os, time, warnings
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
    ALL_MODELS, ASSETS,
)

MAX_DIAG_HOURS = 6 * 30 * 24  # 4320h — same as production

# V1.5.1 constants
V151_DEFAULT_TRIALS = 150
V151_PRUNING_WARMUP = 8
V151_CSV = os.path.join('models', 'crypto_deku_v1_5_1_best_models.csv')
N_HOLDOUT_FOLDS = 3

# V1.5.1: Discrete search spaces
FEATURE_COUNTS = [10, 13, 17, 20, 25, 30, 40]
GAMMA_VALUES = [0.995, 0.996, 0.997, 0.998, 0.999]


def build_holdout_folds(n_total):
    """Build overlapping holdout folds (same as production).

    Fold 1: train [0%, 60%]  test [60%, 80%]
    Fold 2: train [10%, 70%] test [70%, 90%]
    Fold 3: train [20%, 80%] test [80%, 100%]
    """
    folds = []
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
    return folds


def run_v151(assets_list, horizon, n_trials):
    """Run Deku V1.5.1 with discrete features & gamma."""
    t_mode_start = time.time()
    _kill_orphan_workers()

    combo_options = _build_combo_list()

    gamma_str = ", ".join(str(g) for g in GAMMA_VALUES)
    feat_str = ", ".join(str(f) for f in FEATURE_COUNTS)

    print("\n" + "=" * 70)
    print(f"  DEKU V1.5.1 MODE D: {horizon}h")
    print(f"  Models: {', '.join(ALL_MODELS.keys())} ({len(combo_options)} combos)")
    print(f"  Trials: {n_trials} | Fixed 6-month data cap")
    print(f"  Gamma:    [{gamma_str}]")
    print(f"  Features: [{feat_str}]")
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
        print(f"  V1.5.1 OPTIMIZATION: {asset_name} ({horizon}h)")
        print(f"{'='*70}")

        df_raw = load_data(asset_name)
        if df_raw is None:
            continue

        # Step 1: Build features — fixed 6-month cap (same as production)
        print(f"\n  Building all features (horizon={horizon}h)...")
        t0 = time.time()
        df_full, all_cols = build_all_features(df_raw, asset_name=asset_name, horizon=horizon)
        print(f"  [Feature build: {(time.time()-t0)/60:.1f} min]")

        total_rows = len(df_full)
        if total_rows > MAX_DIAG_HOURS:
            df_full = df_full.tail(MAX_DIAG_HOURS).reset_index(drop=True)
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

        # V1.5.1: Filter feature counts to available features
        available_feat_counts = [f for f in FEATURE_COUNTS if f <= len(ranked_features)]
        if not available_feat_counts:
            available_feat_counts = [len(ranked_features)]
        print(f"  Feature choices: {available_feat_counts}")

        # Prepare full data arrays (columns in rank order)
        df_optuna = df_clean.dropna(subset=ranked_features + ['label']).reset_index(drop=True)
        features_np_all = df_optuna[ranked_features].values.astype(np.float64)
        labels_np_all = df_optuna['label'].values.astype(np.int32)
        closes_np_all = df_optuna['close'].values.astype(np.float64)
        n_total = len(df_optuna)

        # Build holdout folds (production-style overlapping)
        holdout_folds = build_holdout_folds(n_total)
        for fi, (tr_s, tr_e, te_s, te_e) in enumerate(holdout_folds):
            print(f"  Fold {fi+1}: train [{tr_s}:{tr_e}] ({tr_e-tr_s:,} rows) -> "
                  f"test [{te_s}:{te_e}] ({te_e-te_s:,} rows)")

        # Optuna trains on fold 1's training partition
        f1_train_s, f1_train_e, _, _ = holdout_folds[0]
        features_np = features_np_all[f1_train_s:f1_train_e]
        labels_np = labels_np_all[f1_train_s:f1_train_e]
        closes_np = closes_np_all[f1_train_s:f1_train_e]
        n = len(features_np)

        print(f"  Optuna trains on fold 1: {n:,} rows | {N_HOLDOUT_FOLDS}-fold holdout after")

        # Step 3: Optuna study
        print(f"\n{'='*70}")
        print(f"  OPTUNA STUDY: {asset_name} {horizon}h")
        print(f"  Search: {len(combo_options)} combos x {len(DIAG_WINDOWS)} windows x {len(GAMMA_VALUES)} gammas x {len(available_feat_counts)} feat counts")
        print(f"  Trials: {n_trials} | Data: {n:,} train")
        print(f"{'='*70}")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=V151_PRUNING_WARMUP,
                max_resource=n // DIAG_STEP,
                reduction_factor=3,
            ),
            study_name=f'v151_{asset_name}_{horizon}h',
        )

        model_factories = _get_deku_diagnostic_models()
        best_apf_so_far = 0.0
        trial_count = 0
        seen_params = set()  # V1.5.1: dedup — prune already-evaluated combos

        def objective(trial):
            nonlocal best_apf_so_far, trial_count
            trial_count += 1

            combo_name = trial.suggest_categorical('combo', combo_options)
            window = trial.suggest_categorical('window', DIAG_WINDOWS)
            # V1.5.1: Discrete gamma and feature counts
            gamma = trial.suggest_categorical('gamma', GAMMA_VALUES)
            n_feat = trial.suggest_categorical('n_features', available_feat_counts)

            # V1.5.1: Prune duplicates instantly — costs 0 compute,
            # and TPE sees pruned = bad → explores different combos
            param_key = (combo_name, window, gamma, n_feat)
            if param_key in seen_params:
                raise optuna.TrialPruned(f"Duplicate: {param_key}")
            seen_params.add(param_key)

            combo = combo_name.split('+')

            # Use all training data (no dynamic cap)
            feat_np = features_np[:, :n_feat]

            result = _deku_eval_with_pruning(
                feat_np, labels_np, closes_np, combo, window, n,
                DIAG_STEP, model_factories, gamma=gamma, trial=trial,
                horizon=horizon
            )

            if result is None:
                return 0.0

            trades = result[6]
            MIN_TRADES = 8
            if trades < MIN_TRADES:
                return 0.0

            # V1.5.1: Score by return (not APF) — directly optimizes for money
            ret = result[4]
            score = ret

            if score > best_apf_so_far:
                best_apf_so_far = score
                print(f"  #{trial_count:3d} NEW BEST: {combo_name:22s} w={window:4d}h "
                      f"g={gamma:.3f} f={n_feat:3d} | ret={ret:+.1f}% trades={trades}")
            elif trial_count % 20 == 0:
                print(f"  #{trial_count:3d} progress: ret={score:+.1f}% | best so far: {best_apf_so_far:+.1f}%")

            return score

        t_optuna = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Auto-extend if needed
        RET_EXTEND_THRESH = 3.0  # extend if best return < +3%
        for ext_target in [200, 250]:
            if ext_target <= n_trials:
                continue
            if study.best_value >= RET_EXTEND_THRESH:
                break
            extra = ext_target - len(study.trials)
            if extra > 0:
                print(f"\n  Best ret={study.best_value:+.1f}% < +{RET_EXTEND_THRESH}% — extending to {ext_target} trials (+{extra})...")
                study.optimize(objective, n_trials=extra, show_progress_bar=False)

        optuna_elapsed = (time.time() - t_optuna) / 60

        # Results
        best_trial = study.best_trial
        print(f"\n  {'='*70}")
        print(f"  OPTUNA RESULTS: {asset_name} {horizon}h ({optuna_elapsed:.1f} min)")
        print(f"  {'='*70}")
        print(f"  Best trial: #{best_trial.number}")
        print(f"  Return:     {best_trial.value:+.2f}%")
        print(f"  Combo:      {best_trial.params['combo']}")
        print(f"  Window:     {best_trial.params['window']}h")
        print(f"  Gamma:      {best_trial.params['gamma']:.3f}")
        print(f"  N_features: {best_trial.params['n_features']}")

        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"\n  Trials: {n_complete} completed, {n_pruned} pruned ({n_pruned/(n_complete+n_pruned)*100:.0f}% pruned)")

        # Top 10 — deduplicated by (score, window, gamma, n_features)
        # When different combos produce identical return, they made the same trades
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value > 0]
        completed_trials.sort(key=lambda t: -t.value)

        seen_scores = set()
        unique_top = []
        for t in completed_trials:
            score_key = (round(t.value, 3), t.params['window'], t.params['gamma'], t.params['n_features'])
            if score_key not in seen_scores:
                seen_scores.add(score_key)
                unique_top.append(t)

        print(f"\n  {'Rank':>4s}  {'Return':>8s}  {'Combo':22s}  {'Window':>6s}  {'Gamma':>7s}  {'Feats':>5s}")
        print(f"  {'-'*62}")
        for i, t in enumerate(unique_top[:10], 1):
            marker = " <-- BEST" if i == 1 else ""
            print(f"  {i:4d}  {t.value:+7.1f}%  {t.params['combo']:22s}  {t.params['window']:5d}h  "
                  f"{t.params['gamma']:7.3f}  {t.params['n_features']:5d}{marker}")

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
        # Dedup by (score, window, gamma, n_features) — same-score combos make identical trades
        seen_configs = set()
        unique_candidates = []
        for t in completed_trials:
            score_key = (round(t.value, 3), t.params['window'], t.params['gamma'], t.params['n_features'])
            if score_key not in seen_configs:
                seen_configs.add(score_key)
                unique_candidates.append(t)

        phase1 = unique_candidates[:10]
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
        print(f"  {N_HOLDOUT_FOLDS}-FOLD HOLDOUT: {asset_name} {horizon}h (top {N_CANDIDATES})")
        print(f"  {'='*70}")

        holdout_results = []
        min_fold_rows = min(te - ts for _, _, ts, te in holdout_folds)
        if min_fold_rows >= 200:
            for ci, trial in enumerate(candidates[:N_CANDIDATES]):
                c_combo = trial.params['combo'].split('+')
                c_window = trial.params['window']
                c_gamma = trial.params['gamma']
                c_n_feat = trial.params['n_features']

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

            # V1.5.1: Rank by AVG_Return (not AVG_APF) — directly picks highest profit
            holdout_results.sort(key=lambda x: -x[2])

            print(f"\n  {'Rank':>4s}  {'AVG_Ret':>8s}  {'AVG_Acc':>7s}  {'Tr':>4s}  "
                  f"{'F1':>6s}  {'F2':>6s}  {'F3':>6s}  {'IS_Ret':>8s}  {'Combo':22s}  {'Win':>4s}  {'Gamma':>7s}  {'F':>3s}")
            print(f"  {'-'*120}")
            for i, (trial, avg_sc, avg_ret, avg_acc, tot_tr, avg_rpf, f_scores, f_rets) in enumerate(holdout_results[:10], 1):
                is_ret = trial.value
                marker = " <-- BEST" if i == 1 else ""
                gen = "+" if avg_ret > 0 and avg_acc > 0.55 else "~" if avg_ret > 0 else "-"
                f_str = "  ".join(f"{s:+5.1f}" for s in f_rets)
                print(f"  {i:4d}  {avg_ret:+7.1f}%  {avg_acc*100:6.1f}%  {tot_tr:4d}  "
                      f"{f_str}  {is_ret:+7.1f}%  {trial.params['combo']:22s}  {trial.params['window']:3d}h  "
                      f"{trial.params['gamma']:7.3f}  {trial.params['n_features']:3d}  {gen}{marker}")
        else:
            print(f"  Hold-out skipped: smallest fold only {min_fold_rows} rows (need 200+)")

        # Pick winner (by avg_ret — index 2)
        if holdout_results and holdout_results[0][2] > 0:
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
        feat_np = features_np[:, :best_n_feat]

        full_result = _deku_eval_with_pruning(
            feat_np, labels_np, closes_np, best_combo, best_window, n,
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
            }
            best_models.append(best_config)

            print(f"\n  {'='*70}")
            print(f"  WINNER: {asset_name} {horizon}h")
            print(f"  Models: {winner_trial.params['combo']}  Window: {best_window}h  Gamma: {best_gamma:.3f}  Features: {best_n_feat}")
            print(f"  In-sample:     ret={cum_ret:+.1f}%  acc={acc*100:.1f}%  trades={trades}  rawPF={raw_pf:.2f}")
            if winner_ho:
                fold_str = " / ".join(f"{r:+.1f}%" for r in ho_fold_rets)
                print(f"  Out-of-sample: avg_ret={ho_ret:+.1f}%  avg_acc={ho_acc*100:.1f}%  trades={ho_trades}")
                print(f"  Per-fold returns: [{fold_str}]")
            print(f"  {'='*70}")

        print(f"  [{asset_name} total: {(time.time()-t_asset)/60:.1f} min]")

    if not best_models:
        print("\nNo results. Aborting.")
        return best_models

    # Save results
    df_best = pd.DataFrame(best_models)
    if os.path.exists(V151_CSV):
        df_existing = pd.read_csv(V151_CSV)
        for m in best_models:
            mask = (df_existing['coin'] == m['coin']) & (df_existing['horizon'] == horizon)
            df_existing = df_existing[~mask]
        df_best = pd.concat([df_existing, df_best], ignore_index=True)
    df_best.to_csv(V151_CSV, index=False)
    print(f"\n  Results saved: {V151_CSV}")

    elapsed = (time.time() - t_mode_start) / 60
    print(f"\n  V1.5.1 Mode D complete: {elapsed:.1f} min total")

    return best_models


# ============================================================
# CLI
# ============================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deku V1.5.1 — Discrete features & gamma')
    parser.add_argument('mode', choices=['D'], help='Mode (D only)')
    parser.add_argument('assets', help='Comma-separated assets (e.g. BTC)')
    parser.add_argument('horizons', help='Horizons (e.g. 8h)')
    parser.add_argument('--trials', type=int, default=V151_DEFAULT_TRIALS, help='Optuna trials')

    args = parser.parse_args()

    assets_list = [a.strip().upper() for a in args.assets.split(',')]
    horizons = [int(h.replace('h', '')) for h in args.horizons.split(',')]

    for h in horizons:
        print(f"\n{'#'*80}")
        print(f"  V1.5.1 RUN: {','.join(assets_list)} {h}h")
        print(f"{'#'*80}")
        run_v151(assets_list, h, args.trials)
