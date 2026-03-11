# =============================================================================
# MODE D IMPROVEMENTS — Two changes to crypto_trading_system.py
# =============================================================================
# CHANGE 1: _quick_accuracy → _quick_score (adds alpha to the walk-forward)
# CHANGE 2: All callers updated to use combined_score = accuracy × (1 + alpha/100)
# CHANGE 3: _score_features updated to score on alpha_drop + alpha_change
# CHANGE 4: run_mode_d downloads macro data before crypto candles
# =============================================================================
#
# HOW TO APPLY:
# Find each "FIND THIS" block in crypto_trading_system.py and replace with
# the corresponding "REPLACE WITH" block below.
# =============================================================================


# =============================================================================
# CHANGE 1 — Replace _quick_accuracy with _quick_score
# Returns (accuracy, alpha, n_tests) instead of (accuracy, n_tests)
# Adds a portfolio simulation loop alongside the existing accuracy loop.
# close prices come from df_features['close'] which is always present.
# =============================================================================

# ---------- FIND THIS ----------
"""
def _quick_accuracy(df_features, feature_cols, window=ANALYSIS_WINDOW, step=ANALYSIS_STEP, device=None):
    \"\"\"Fast walk-forward test with LGBM only. Returns accuracy.
    device: override LGBM device ('cpu' for parallel safety, None = use default).\"\"\"
    from lightgbm import LGBMClassifier

    lgbm_device = device if device else LGBM_DEVICE

    n = len(df_features)
    min_start = window + 50
    if n < min_start + 30:
        return 0, 0

    correct = 0
    total = 0

    for i in range(min_start, n, step):
        train = df_features.iloc[max(0, i - window):i]
        test_row = df_features.iloc[i:i+1]
        X_train = train[feature_cols]
        y_train = train['label'].values
        X_test = test_row[feature_cols]
        y_true = test_row['label'].values[0]

        if len(np.unique(y_train)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train),
                                 columns=feature_cols, index=X_train.index)
        X_test_s = pd.DataFrame(scaler.transform(X_test),
                                columns=feature_cols, index=X_test.index)

        model = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            class_weight='balanced', verbose=-1, random_state=42,
            device=lgbm_device
        )
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)[0]
        if pred == y_true:
            correct += 1
        total += 1

    return (correct / total * 100 if total > 0 else 0), total
"""

# ---------- REPLACE WITH ----------
"""
def _quick_score(df_features, feature_cols, window=ANALYSIS_WINDOW, step=ANALYSIS_STEP, device=None):
    \"\"\"Fast walk-forward test with LGBM only.
    Returns (accuracy, alpha, n_tests).
    Alpha = strategy return - buy & hold return (same period, with 0.09% fees).
    device: override LGBM device ('cpu' for parallel safety, None = use default).\"\"\"
    from lightgbm import LGBMClassifier

    lgbm_device = device if device else LGBM_DEVICE
    n = len(df_features)
    min_start = window + 50
    if n < min_start + 30:
        return 0, 0, 0

    correct = 0
    total = 0

    # Portfolio simulation (alongside accuracy)
    cash      = 1.0
    in_pos    = False
    entry_px  = 0.0
    start_px  = float(df_features.iloc[min_start]['close'])

    for i in range(min_start, n, step):
        train    = df_features.iloc[max(0, i - window):i]
        test_row = df_features.iloc[i:i+1]
        X_train  = train[feature_cols]
        y_train  = train['label'].values
        X_test   = test_row[feature_cols]
        y_true   = test_row['label'].values[0]
        price    = float(test_row['close'].values[0])

        if len(np.unique(y_train)) < 2:
            continue

        scaler    = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train),
                                 columns=feature_cols, index=X_train.index)
        X_test_s  = pd.DataFrame(scaler.transform(X_test),
                                 columns=feature_cols, index=X_test.index)

        model = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            class_weight='balanced', verbose=-1, random_state=42,
            device=lgbm_device
        )
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)[0]

        if pred == y_true:
            correct += 1
        total += 1

        # Portfolio: BUY on pred=1, SELL on pred=0
        if pred == 1 and not in_pos:
            in_pos   = True
            entry_px = price * (1 + TRADING_FEE)
        elif pred == 0 and in_pos:
            cash   *= (price * (1 - TRADING_FEE)) / entry_px
            in_pos  = False

    # Close open position at end
    if in_pos and total > 0:
        last_px = float(df_features.iloc[-1]['close'])
        cash   *= (last_px * (1 - TRADING_FEE)) / entry_px

    last_px    = float(df_features.iloc[-1]['close'])
    strat_ret  = (cash - 1.0) * 100
    bh_ret     = (last_px / start_px - 1) * 100
    alpha      = round(strat_ret - bh_ret, 2)
    accuracy   = correct / total * 100 if total > 0 else 0
    return accuracy, alpha, total


def _quick_accuracy(df_features, feature_cols, window=ANALYSIS_WINDOW, step=ANALYSIS_STEP, device=None):
    \"\"\"Backward-compatible wrapper — returns (accuracy, n_tests). Used by Mode E.\"\"\"\
    acc, _, n = _quick_score(df_features, feature_cols, window=window, step=step, device=device)
    return acc, n
"""


# =============================================================================
# CHANGE 2 — Update _perm_one_feature to return alpha_drop too
# =============================================================================

# ---------- FIND THIS ----------
"""
def _perm_one_feature(df_features, feature_cols, feat, baseline_acc):
    \"\"\"Helper for parallel permutation test.\"\"\"
    import os, warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    df_shuffled = df_features.copy()
    df_shuffled[feat] = np.random.permutation(df_shuffled[feat].values)
    shuffled_acc, _ = _quick_accuracy(df_shuffled, feature_cols, device='cpu')
    return feat, baseline_acc - shuffled_acc
"""

# ---------- REPLACE WITH ----------
"""
def _perm_one_feature(df_features, feature_cols, feat, baseline_acc, baseline_alpha):
    \"\"\"Helper for parallel permutation test. Returns (feat, acc_drop, alpha_drop).\"\"\"
    import os, warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    df_shuffled = df_features.copy()
    df_shuffled[feat] = np.random.permutation(df_shuffled[feat].values)
    shuffled_acc, shuffled_alpha, _ = _quick_score(df_shuffled, feature_cols, device='cpu')
    return feat, baseline_acc - shuffled_acc, baseline_alpha - shuffled_alpha
"""


# =============================================================================
# CHANGE 3 — Update _test_permutation_importance to use new worker + store alpha_drop
# =============================================================================

# ---------- FIND THIS ----------
"""
def _test_permutation_importance(df_features, feature_cols):
    \"\"\"Shuffle each feature and measure accuracy drop. Parallelized.\"\"\"
    print("\\n  [2/5] Permutation Importance (parallel)")
    baseline_acc, n_tests = _quick_accuracy(df_features, feature_cols)
    print(f"    Baseline: {baseline_acc:.1f}% (n={n_tests})")

    n_workers = min(N_JOBS_PARALLEL, len(feature_cols))
    print(f"    Testing {len(feature_cols)} features ({n_workers} workers)...")
    t0 = time.time()
    with _suppress_stderr():
        perm_results = Parallel(n_jobs=n_workers, verbose=0)(
            delayed(_perm_one_feature)(df_features, feature_cols, feat, baseline_acc)
            for feat in feature_cols
        )
    print(f"    Done in {(time.time() - t0)/60:.1f} min")

    results = []
    for feat, drop in perm_results:
        results.append({'feature': feat, 'acc_drop': drop})
        print(f"    {feat:30s} drop: {drop:+5.1f}%")

    return pd.DataFrame(results).sort_values('acc_drop', ascending=False)
"""

# ---------- REPLACE WITH ----------
"""
def _test_permutation_importance(df_features, feature_cols):
    \"\"\"Shuffle each feature and measure accuracy + alpha drop. Parallelized.\"\"\"
    print("\\n  [2/5] Permutation Importance (parallel)")
    baseline_acc, baseline_alpha, n_tests = _quick_score(df_features, feature_cols)
    print(f"    Baseline: {baseline_acc:.1f}% acc | {baseline_alpha:+.1f}% alpha (n={n_tests})")

    n_workers = min(N_JOBS_PARALLEL, len(feature_cols))
    print(f"    Testing {len(feature_cols)} features ({n_workers} workers)...")
    t0 = time.time()
    with _suppress_stderr():
        perm_results = Parallel(n_jobs=n_workers, verbose=0)(
            delayed(_perm_one_feature)(df_features, feature_cols, feat, baseline_acc, baseline_alpha)
            for feat in feature_cols
        )
    print(f"    Done in {(time.time() - t0)/60:.1f} min")

    results = []
    for feat, acc_drop, alpha_drop in perm_results:
        results.append({'feature': feat, 'acc_drop': acc_drop, 'alpha_drop': alpha_drop})
        print(f"    {feat:30s} acc_drop: {acc_drop:+5.1f}%  alpha_drop: {alpha_drop:+6.1f}%")

    return pd.DataFrame(results).sort_values('acc_drop', ascending=False)
"""


# =============================================================================
# CHANGE 4 — Update _ablation_one_feature to return alpha_change too
# =============================================================================

# ---------- FIND THIS ----------
"""
def _ablation_one_feature(df_features, feature_cols, feat, baseline_acc):
    \"\"\"Helper for parallel ablation test.\"\"\"
    import os, warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    reduced = [f for f in feature_cols if f != feat]
    acc, _ = _quick_accuracy(df_features, reduced, device='cpu')
    return feat, acc, acc - baseline_acc
"""

# ---------- REPLACE WITH ----------
"""
def _ablation_one_feature(df_features, feature_cols, feat, baseline_acc, baseline_alpha):
    \"\"\"Helper for parallel ablation test. Returns (feat, acc, acc_change, alpha_change).\"\"\"
    import os, warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    reduced = [f for f in feature_cols if f != feat]
    acc, alpha, _ = _quick_score(df_features, reduced, device='cpu')
    return feat, acc, acc - baseline_acc, alpha - baseline_alpha
"""


# =============================================================================
# CHANGE 5 — Update _test_ablation to use new worker + store alpha_change
# =============================================================================

# ---------- FIND THIS ----------
"""
def _test_ablation(df_features, feature_cols):
    \"\"\"Drop each feature one at a time and measure accuracy. Parallelized.\"\"\"
    print("\\n  [3/5] Ablation Test (parallel, drop one at a time)")
    baseline_acc, _ = _quick_accuracy(df_features, feature_cols)
    print(f"    Baseline ({len(feature_cols)} features): {baseline_acc:.1f}%")

    n_workers = min(N_JOBS_PARALLEL, len(feature_cols))
    print(f"    Testing {len(feature_cols)} features ({n_workers} workers)...")
    t0 = time.time()
    with _suppress_stderr():
        ablation_results = Parallel(n_jobs=n_workers, verbose=0)(
            delayed(_ablation_one_feature)(df_features, feature_cols, feat, baseline_acc)
            for feat in feature_cols
        )
    print(f"    Done in {(time.time() - t0)/60:.1f} min")

    results = []
    for feat, acc, change in ablation_results:
        results.append({'dropped': feat, 'accuracy': acc, 'change': change})
        marker = ' ** IMPROVES' if change > 0.3 else ''
        print(f"    Drop {feat:30s} -> {acc:5.1f}% ({change:+5.1f}%){marker}")

    return pd.DataFrame(results).sort_values('change', ascending=False)
"""

# ---------- REPLACE WITH ----------
"""
def _test_ablation(df_features, feature_cols):
    \"\"\"Drop each feature one at a time and measure accuracy + alpha. Parallelized.\"\"\"
    print("\\n  [3/5] Ablation Test (parallel, drop one at a time)")
    baseline_acc, baseline_alpha, _ = _quick_score(df_features, feature_cols)
    print(f"    Baseline ({len(feature_cols)} features): {baseline_acc:.1f}% acc | {baseline_alpha:+.1f}% alpha")

    n_workers = min(N_JOBS_PARALLEL, len(feature_cols))
    print(f"    Testing {len(feature_cols)} features ({n_workers} workers)...")
    t0 = time.time()
    with _suppress_stderr():
        ablation_results = Parallel(n_jobs=n_workers, verbose=0)(
            delayed(_ablation_one_feature)(df_features, feature_cols, feat, baseline_acc, baseline_alpha)
            for feat in feature_cols
        )
    print(f"    Done in {(time.time() - t0)/60:.1f} min")

    results = []
    for feat, acc, acc_change, alpha_change in ablation_results:
        results.append({'dropped': feat, 'accuracy': acc,
                        'change': acc_change, 'alpha_change': alpha_change})
        marker = ' ** IMPROVES' if acc_change > 0.3 or alpha_change > 3 else ''
        print(f"    Drop {feat:30s} -> {acc:5.1f}% ({acc_change:+5.1f}%)  "
              f"alpha_chg: {alpha_change:+6.1f}%{marker}")

    return pd.DataFrame(results).sort_values('change', ascending=False)
"""


# =============================================================================
# CHANGE 6 — Update _reduced_one_set to return alpha too
# =============================================================================

# ---------- FIND THIS ----------
"""
def _reduced_one_set(df_features, ranked, n_feat):
    \"\"\"Helper for parallel reduced set test.\"\"\"
    import os, warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    top_n = ranked[:n_feat]
    acc, _ = _quick_accuracy(df_features, top_n, device='cpu')
    return n_feat, acc
"""

# ---------- REPLACE WITH ----------
"""
def _reduced_one_set(df_features, ranked, n_feat):
    \"\"\"Helper for parallel reduced set test. Returns (n_feat, acc, alpha, combined_score).\"\"\"
    import os, warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    top_n = ranked[:n_feat]
    acc, alpha, _ = _quick_score(df_features, top_n, device='cpu')
    combined = acc * (1 + max(alpha, 0) / 100)
    return n_feat, acc, alpha, combined
"""


# =============================================================================
# CHANGE 7 — Update _test_reduced_sets to display + rank by combined score
# =============================================================================

# ---------- FIND THIS ----------
"""
    results = []
    for n_feat, acc in sorted(reduced_results):
        results.append({'n_features': n_feat, 'accuracy': acc})
        bar = '#' * int(acc * 0.5)
        print(f"    Top {n_feat:3d} features: {acc:5.1f}% {bar}")

    df_results = pd.DataFrame(results)
    best_row = df_results.loc[df_results['accuracy'].idxmax()]
    print(f"\\n    OPTIMAL: Top {int(best_row['n_features'])} -> {best_row['accuracy']:.1f}%")
    return df_results
"""

# ---------- REPLACE WITH ----------
"""
    results = []
    for n_feat, acc, alpha, combined in sorted(reduced_results):
        results.append({'n_features': n_feat, 'accuracy': acc,
                        'alpha': alpha, 'combined_score': combined})
        bar = '#' * int(acc * 0.5)
        print(f"    Top {n_feat:3d} features: {acc:5.1f}% acc | {alpha:+6.1f}% alpha | "
              f"score={combined:.1f}  {bar}")

    df_results = pd.DataFrame(results)
    best_row = df_results.loc[df_results['combined_score'].idxmax()]
    print(f"\\n    OPTIMAL: Top {int(best_row['n_features'])} -> "
          f"{best_row['accuracy']:.1f}% acc | {best_row['alpha']:+.1f}% alpha | "
          f"score={best_row['combined_score']:.1f}")
    return df_results
"""


# =============================================================================
# CHANGE 8 — Update _score_features to include alpha scoring
# =============================================================================

# ---------- FIND THIS ----------
"""
    # Permutation
    if permutation_df is not None:
        for _, row in permutation_df.iterrows():
            f = row['feature']
            if row['acc_drop'] > 0.5:
                scores[f] += 2
            elif row['acc_drop'] > 0:
                scores[f] += 1
            else:
                scores[f] -= 1

    # Ablation
    if ablation_df is not None:
        for _, row in ablation_df.iterrows():
            f = row['dropped']
            if row['change'] > 0.3:
                scores[f] -= 3
            elif row['change'] > 0:
                scores[f] -= 1
"""

# ---------- REPLACE WITH ----------
"""
    # Permutation — accuracy drop + alpha drop
    if permutation_df is not None:
        for _, row in permutation_df.iterrows():
            f = row['feature']
            # Accuracy signal
            if row['acc_drop'] > 0.5:
                scores[f] += 2
            elif row['acc_drop'] > 0:
                scores[f] += 1
            else:
                scores[f] -= 1
            # Alpha signal (shuffling this feature kills alpha = it matters)
            alpha_drop = row.get('alpha_drop', 0)
            if alpha_drop > 10:
                scores[f] += 2
            elif alpha_drop > 3:
                scores[f] += 1
            elif alpha_drop < -3:
                scores[f] -= 1

    # Ablation — accuracy change + alpha change
    if ablation_df is not None:
        for _, row in ablation_df.iterrows():
            f = row['dropped']
            # Accuracy signal
            if row['change'] > 0.3:
                scores[f] -= 3
            elif row['change'] > 0:
                scores[f] -= 1
            # Alpha signal (dropping this feature improves alpha = it was hurting)
            alpha_change = row.get('alpha_change', 0)
            if alpha_change > 5:
                scores[f] -= 2
            elif alpha_change > 2:
                scores[f] -= 1
            elif alpha_change < -5:
                scores[f] += 1
"""


# =============================================================================
# CHANGE 9 — Update run_feature_analysis final comparison to use combined_score
# =============================================================================

# ---------- FIND THIS ----------
"""
    # Quick test: KEEP only vs KEEP + MAYBE
    if maybe:
        acc_keep, _ = _quick_accuracy(df_features, keep)
        acc_all, _ = _quick_accuracy(df_features, keep + maybe)
        print(f"\\n  KEEP only ({len(keep)} features): {acc_keep:.1f}%")
        print(f"  KEEP + MAYBE ({len(keep) + len(maybe)} features): {acc_all:.1f}%")
        if acc_all > acc_keep + 0.5:
            optimal_features = keep + maybe
            print(f"  >>> Using KEEP + MAYBE ({len(optimal_features)} features)")
        else:
            print(f"  >>> Using KEEP only ({len(optimal_features)} features)")
    else:
        print(f"\\n  >>> Using KEEP ({len(optimal_features)} features)")

    # Also check best reduced set from test 4
    if reduced_df is not None and len(reduced_df) > 0:
        best_n_row = reduced_df.loc[reduced_df['accuracy'].idxmax()]
        best_n = int(best_n_row['n_features'])
        best_n_acc = best_n_row['accuracy']
        ranked = importance_df['feature'].tolist()
        top_n_features = ranked[:best_n]

        opt_acc, _ = _quick_accuracy(df_features, optimal_features)
        print(f"  Scored optimal ({len(optimal_features)}): {opt_acc:.1f}%")
        print(f"  Top-{best_n} by LGBM: {best_n_acc:.1f}%")

        if best_n_acc > opt_acc + 1.0:
            optimal_features = top_n_features
            print(f"  >>> Switching to Top-{best_n} by LGBM (better by {best_n_acc - opt_acc:.1f}%)")
"""

# ---------- REPLACE WITH ----------
"""
    # Quick test: KEEP only vs KEEP + MAYBE — use combined_score = acc × (1 + alpha/100)
    if maybe:
        acc_keep,  alpha_keep,  _ = _quick_score(df_features, keep)
        acc_all,   alpha_all,   _ = _quick_score(df_features, keep + maybe)
        score_keep = acc_keep  * (1 + max(alpha_keep, 0)  / 100)
        score_all  = acc_all   * (1 + max(alpha_all,  0)  / 100)
        print(f"\\n  KEEP only       ({len(keep):3d} feat): {acc_keep:.1f}% acc | "
              f"{alpha_keep:+.1f}% alpha | score={score_keep:.1f}")
        print(f"  KEEP + MAYBE    ({len(keep)+len(maybe):3d} feat): {acc_all:.1f}% acc | "
              f"{alpha_all:+.1f}% alpha | score={score_all:.1f}")
        if score_all > score_keep + 1.0:
            optimal_features = keep + maybe
            print(f"  >>> Using KEEP + MAYBE ({len(optimal_features)} features)")
        else:
            print(f"  >>> Using KEEP only ({len(optimal_features)} features)")
    else:
        print(f"\\n  >>> Using KEEP ({len(optimal_features)} features)")

    # Also check best reduced set from test 4 — compare by combined_score
    if reduced_df is not None and len(reduced_df) > 0:
        best_n_row     = reduced_df.loc[reduced_df['combined_score'].idxmax()]
        best_n         = int(best_n_row['n_features'])
        best_n_score   = best_n_row['combined_score']
        best_n_acc     = best_n_row['accuracy']
        best_n_alpha   = best_n_row['alpha']
        ranked         = importance_df['feature'].tolist()
        top_n_features = ranked[:best_n]

        opt_acc, opt_alpha, _ = _quick_score(df_features, optimal_features)
        opt_score = opt_acc * (1 + max(opt_alpha, 0) / 100)
        print(f"  Scored optimal  ({len(optimal_features):3d} feat): {opt_acc:.1f}% acc | "
              f"{opt_alpha:+.1f}% alpha | score={opt_score:.1f}")
        print(f"  Top-{best_n} by LGBM ({best_n:3d} feat): {best_n_acc:.1f}% acc | "
              f"{best_n_alpha:+.1f}% alpha | score={best_n_score:.1f}")

        if best_n_score > opt_score + 2.0:
            optimal_features = top_n_features
            print(f"  >>> Switching to Top-{best_n} (score +{best_n_score - opt_score:.1f})")
"""


# =============================================================================
# CHANGE 10 — run_mode_d: download macro data BEFORE crypto candles
# =============================================================================

# ---------- FIND THIS ----------
"""
    update_all_data(assets_list)
    diag_hours = diag_years * 365 * 24
    best_models = []
"""
# (This is inside run_mode_d — identify it by the surrounding context:
#  it comes right after the print("  Starts from ALL features...") block)

# ---------- REPLACE WITH ----------
"""
    # Download macro data first (VIX, DXY, S&P500, Fear&Greed, cross-asset)
    print("\\n  Updating macro & sentiment data...")
    try:
        import download_macro_data
        download_macro_data.main()
    except ImportError:
        print("  WARNING: download_macro_data.py not found — macro features may be stale.")
    except Exception as e:
        print(f"  WARNING: Macro data update failed: {e}")

    # Then download crypto candles
    update_all_data(assets_list)
    diag_hours = diag_years * 365 * 24
    best_models = []
"""
