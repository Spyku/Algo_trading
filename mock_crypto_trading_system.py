"""
mock_crypto_trading_system.py
============================================================
MOCK version of crypto_trading_system.py for testing 4 new improvements
before integrating them into production.

All improvements are self-contained and use SYNTHETIC data — no CSV files,
no Binance API, no macro downloads needed. Just run it.

IMPROVEMENTS TESTED:
  1. Holdout set      — reserve last 500h, report overfit gap
  2. Bootstrap CI     — 95% confidence interval on Mode B accuracy
  3. Permutation test — p-value: could this accuracy be luck?
  4. Calmar/Sharpe    — replace heuristic combined score

HOW TO RUN:
  python mock_crypto_trading_system.py

EXPECTED OUTPUT:
  Section headers for each improvement, clear PASS/FAIL indicators,
  and copy-paste-ready code snippets showing exactly what changes
  in the real system.

PORTING TO PRODUCTION:
  Each section is labelled with the exact function and line it replaces.
  Search for "REAL SYSTEM:" comments to find the integration points.
"""

import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

# ============================================================
# CONSTANTS (mirrors production)
# ============================================================
TRADING_FEE  = 0.0009
MIN_CONFIDENCE = 75
DIAG_STEP    = 72
HOLDOUT_HOURS = 500   # NEW: reserved for out-of-sample test
N_BOOTSTRAP  = 500    # NEW: bootstrap resamples for CI
N_PERMUTATIONS = 100  # NEW: permutation test iterations (use 500 in production)

# ============================================================
# SYNTHETIC DATA GENERATOR
# ============================================================
def generate_synthetic_data(n_hours=10_000, seed=42):
    """
    Generate realistic-ish hourly OHLCV + features + label.

    Price: geometric random walk with slight upward drift and vol clustering.
    Features: mix of informative (lag returns, RSI-like) and noise columns.
    Label: 1 if close[t+4] > close[t], 0 otherwise — slightly predictable
           from recent momentum to give the model something real to learn.
    """
    rng = np.random.default_rng(seed)

    # Price path
    dt = 1 / 8760
    drift = 0.15 * dt
    base_vol = 0.015
    vol = np.ones(n_hours) * base_vol

    # GARCH-lite: vol clustering
    for i in range(1, n_hours):
        shock = abs(rng.normal())
        vol[i] = 0.94 * vol[i-1] + 0.06 * base_vol * shock

    log_returns = rng.normal(drift, vol)
    prices = 40_000 * np.exp(np.cumsum(log_returns))

    df = pd.DataFrame({
        'datetime': pd.date_range('2022-01-01', periods=n_hours, freq='h'),
        'close': prices,
        'volume': rng.lognormal(10, 1, n_hours),
    })

    # Features: some informative, some noise
    for p in [1, 4, 8, 12, 24, 48, 72]:
        df[f'logret_{p}h'] = np.log(df['close'] / df['close'].shift(p))

    # RSI-like (informative)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi_14h'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

    # Volatility (informative)
    df['volatility_12h'] = df['logret_1h'].rolling(12).std()
    df['vol_ratio']      = df['logret_1h'].rolling(12).std() / \
                           df['logret_1h'].rolling(48).std()

    # SMA ratio (informative)
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['sma_ratio'] = df['sma20'] / df['sma50'] - 1

    # Pure noise features (should be dropped by feature analysis)
    for i in range(5):
        df[f'noise_{i}'] = rng.normal(0, 1, n_hours)

    # Label: 1 if price is higher 4h from now
    df['label'] = (df['close'].shift(-4) > df['close']).astype(int)

    df = df.dropna().reset_index(drop=True)
    print(f"  Synthetic data: {len(df):,} rows, price range "
          f"${df['close'].min():,.0f}–${df['close'].max():,.0f}")
    return df


FEATURE_COLS = [
    'logret_1h', 'logret_4h', 'logret_8h', 'logret_12h',
    'logret_24h', 'logret_48h', 'logret_72h',
    'rsi_14h', 'volatility_12h', 'vol_ratio', 'sma_ratio',
    'noise_0', 'noise_1', 'noise_2', 'noise_3', 'noise_4',
]


# ============================================================
# SIMPLIFIED WALK-FORWARD EVAL (mirrors _eval_one_config)
# ============================================================
def _eval_config(df, feature_cols, window=200, step=DIAG_STEP, model_name='RF'):
    """
    Walk-forward evaluation. Returns a result dict.
    Mirrors _eval_one_config in the real system.
    """
    models = {
        'RF':  lambda: RandomForestClassifier(n_estimators=50, max_depth=5,
                                               class_weight='balanced', random_state=42, n_jobs=1),
        'LR':  lambda: LogisticRegression(max_iter=200, class_weight='balanced',
                                           random_state=42, C=0.5),
        'GB':  lambda: GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
    }
    factory = models[model_name]

    feats_np  = df[feature_cols].values.astype(np.float64)
    labels_np = df['label'].values.astype(np.int32)
    closes_np = df['close'].values.astype(np.float64)
    n = len(df)
    min_start = window + 50
    if n < min_start + 50:
        return None

    correct = 0
    total   = 0
    portfolio   = 1.0
    in_position = False
    entry_price = 0.0
    trades = wins = 0
    peak   = 1.0
    max_dd = 0.0
    trade_returns = []   # NEW: collect per-trade returns for Sharpe

    for i in range(min_start, n, step):
        X_train = feats_np[max(0, i-window):i]
        y_train = labels_np[max(0, i-window):i]
        X_test  = feats_np[i:i+1]
        y_true  = labels_np[i]
        price   = closes_np[i]

        if len(np.unique(y_train)) < 2:
            continue
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            continue

        mean = X_train.mean(axis=0)
        std  = X_train.std(axis=0)
        std[std == 0] = 1.0
        X_train_s = (X_train - mean) / std
        X_test_s  = (X_test  - mean) / std

        m = factory()
        m.fit(X_train_s, y_train)
        pred = m.predict(X_test_s)[0]

        if pred == y_true:
            correct += 1
        total += 1

        if pred == 1 and not in_position:
            in_position = True
            entry_price = price * (1 + TRADING_FEE)
        elif pred == 0 and in_position:
            sell_price   = price * (1 - TRADING_FEE)
            trade_ret    = (sell_price - entry_price) / entry_price
            portfolio   *= (1 + trade_ret)
            trade_returns.append(trade_ret)
            trades += 1
            if trade_ret > 0:
                wins += 1
            in_position = False

        # Drawdown tracking
        cur = portfolio * (price / entry_price) if in_position else portfolio
        if cur > peak:
            peak = cur
        dd = (peak - cur) / peak
        if dd > max_dd:
            max_dd = dd

    # Close open position at end
    if in_position and total > 0:
        last_px    = closes_np[n - 1]
        sell_price = last_px * (1 - TRADING_FEE)
        trade_ret  = (sell_price - entry_price) / entry_price
        portfolio *= (1 + trade_ret)
        trade_returns.append(trade_ret)
        trades += 1
        if trade_ret > 0:
            wins += 1

    if total == 0:
        return None

    accuracy   = correct / total * 100
    cum_return = (portfolio - 1.0) * 100
    win_rate   = (wins / trades * 100) if trades > 0 else 0

    return {
        'model':       model_name,
        'window':      window,
        'accuracy':    accuracy,
        'cum_return':  cum_return,
        'win_rate':    win_rate,
        'trades':      trades,
        'max_dd':      max_dd * 100,
        'n_tests':     total,
        'trade_returns': trade_returns,
        # OLD combined score (kept for comparison)
        'combined_score_old': _score_old(accuracy, cum_return, max_dd),
        # NEW combined scores
        'calmar':      _calmar(cum_return, max_dd, total, step),
        'sharpe':      _sharpe(trade_returns),
        'combined_score_new': _score_new(accuracy, cum_return, max_dd, total, step, trade_returns),
    }


# ============================================================
# IMPROVEMENT 4: SCORING FUNCTIONS
# REAL SYSTEM: replace combined_score block in _eval_one_config
# ============================================================
def _score_old(accuracy, cum_return, max_dd):
    """
    CURRENT production formula.
    combined_score = accuracy × profit_factor^1.5 × drawdown_penalty
    """
    accuracy_frac = accuracy / 100
    profit_factor = 1 + cum_return / 100
    dd_penalty    = max(0.5, 1.0 - max(0, max_dd - 0.20) * 0.5)
    return accuracy_frac * (max(profit_factor, 0.01) ** 1.5) * dd_penalty


def _calmar(cum_return, max_dd, n_tests, step):
    """
    Calmar ratio: annualised return / max drawdown.
    Interpretable: 2.0 = earned 2× your max loss per year.

    REAL SYSTEM: add to _eval_one_config return tuple.
    """
    hours_tested  = n_tests * step
    n_years       = hours_tested / 8760
    ann_return    = ((1 + cum_return / 100) ** (1 / max(n_years, 0.1)) - 1) * 100
    if max_dd < 0.001:          # near-zero DD: treat as 1% to avoid divide-by-zero
        return ann_return / 1.0
    return ann_return / (max_dd * 100)


def _sharpe(trade_returns):
    """
    Sharpe on per-trade returns (annualised, assuming ~2 trades/week).
    Standard: >1.0 good, >2.0 excellent.

    REAL SYSTEM: collect trade_returns list in _eval_one_config loop,
                 compute here.
    """
    if len(trade_returns) < 3:
        return 0.0
    r   = np.array(trade_returns)
    mu  = r.mean()
    sig = r.std(ddof=1)
    if sig == 0:
        return 0.0
    # Assume ~2 trades per week = ~104 trades per year
    trades_per_year = 104
    return (mu / sig) * np.sqrt(trades_per_year)


def _score_new(accuracy, cum_return, max_dd, n_tests, step, trade_returns):
    """
    NEW combined score: blend Calmar + Sharpe + accuracy signal.

    Weights chosen to:
    - Ground in trading theory (Calmar, Sharpe)
    - Keep accuracy as a tiebreaker (prevents lucky single-trade models)
    - Map to roughly the same order-of-magnitude as the old score

    REAL SYSTEM: replace combined_score = accuracy * ... block
                 in _eval_one_config with this function.
    """
    calmar     = _calmar(cum_return, max_dd, n_tests, step)
    sharpe     = _sharpe(trade_returns)
    acc_signal = (accuracy / 100 - 0.5) * 2   # maps [50..100%] → [0..1]

    # Clip extreme values to prevent one metric dominating
    calmar_clipped = np.clip(calmar, -5, 10)
    sharpe_clipped = np.clip(sharpe, -3, 5)
    acc_clipped    = max(acc_signal, 0)

    return 0.45 * calmar_clipped + 0.35 * sharpe_clipped + 0.20 * acc_clipped


# ============================================================
# IMPROVEMENT 1: HOLDOUT SET
# REAL SYSTEM: add to run_mode_d(), after df_full is built,
#              before run_feature_analysis() is called
# ============================================================
def test_holdout(df, feature_cols, window=200, model_name='RF'):
    """
    Reserve last HOLDOUT_HOURS rows. Train+select on everything before.
    Evaluate best config on holdout. Report overfit gap.
    """
    print(f"\n{'='*60}")
    print(f"  IMPROVEMENT 1: HOLDOUT OUT-OF-SAMPLE TEST")
    print(f"{'='*60}")
    print(f"  Reserving last {HOLDOUT_HOURS}h as holdout (never seen during training)")

    n = len(df)
    df_train   = df.iloc[:-HOLDOUT_HOURS].reset_index(drop=True)
    df_holdout = df.iloc[-HOLDOUT_HOURS:].reset_index(drop=True)

    print(f"  Train:   {len(df_train):,} rows")
    print(f"  Holdout: {len(df_holdout):,} rows")

    # Evaluate on training window (mirrors what Mode D does today)
    print(f"\n  Evaluating on TRAIN window...")
    t0 = time.time()
    train_result = _eval_config(df_train, feature_cols, window=window, model_name=model_name)
    print(f"  Done in {time.time()-t0:.1f}s")

    if train_result is None:
        print("  Not enough data for train eval")
        return

    # Evaluate SAME config on holdout (no re-training, no re-selection)
    print(f"\n  Evaluating on HOLDOUT window...")
    t0 = time.time()
    holdout_result = _eval_config(df_holdout, feature_cols, window=window, model_name=model_name)
    print(f"  Done in {time.time()-t0:.1f}s")

    if holdout_result is None:
        print("  Not enough holdout data for evaluation (need >500h)")
        return

    # Report
    train_acc    = train_result['accuracy']
    holdout_acc  = holdout_result['accuracy']
    train_ret    = train_result['cum_return']
    holdout_ret  = holdout_result['cum_return']
    overfit_gap  = train_acc - holdout_acc

    print(f"\n  {'Metric':<20} {'Train':>10} {'Holdout':>10} {'Gap':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Accuracy':<20} {train_acc:>9.1f}% {holdout_acc:>9.1f}% {overfit_gap:>+9.1f}%")
    print(f"  {'Return':<20} {train_ret:>+9.1f}% {holdout_ret:>+9.1f}%")
    print(f"  {'Max DD':<20} {train_result['max_dd']:>9.1f}% {holdout_result['max_dd']:>9.1f}%")

    if overfit_gap > 8:
        verdict = "⚠  HIGH OVERFIT — increase window or reduce features"
    elif overfit_gap > 4:
        verdict = "⚡  MILD OVERFIT — monitor closely"
    elif overfit_gap < -3:
        verdict = "✓  HOLDOUT BETTER THAN TRAIN — regime shift, model generalises well"
    else:
        verdict = "✓  ACCEPTABLE — gap within noise range"

    print(f"\n  Overfit gap: {overfit_gap:+.1f}%  →  {verdict}")

    print(f"""
  ─── INTEGRATION NOTES ──────────────────────────────────
  REAL SYSTEM: in run_mode_d(), after df_full is built and
  before run_feature_analysis():

    HOLDOUT_HOURS = 500
    df_full_raw = df_full.copy()
    df_full     = df_full.iloc[:-HOLDOUT_HOURS]   # training only
    df_holdout  = df_full_raw.iloc[-HOLDOUT_HOURS:]

  After best_config is found, call _eval_one_config on
  df_holdout and print the overfit gap.
  Threshold: >5% gap = warn. >10% gap = reject model.
  ────────────────────────────────────────────────────────""")

    return {'train': train_result, 'holdout': holdout_result, 'gap': overfit_gap}


# ============================================================
# IMPROVEMENT 2: BOOTSTRAP CONFIDENCE INTERVAL
# REAL SYSTEM: call after generate_signals() in run_mode_b()
#              and in _run_quick_asset()
# ============================================================
def bootstrap_ci(signals, n_bootstrap=N_BOOTSTRAP, ci=0.95):
    """
    Bootstrap 95% CI on directional accuracy from a signals list.
    Works on the existing signals list from generate_signals() — zero extra compute.

    Returns: (point_estimate, lower_bound, upper_bound, n_predictions)
    """
    actionals = [
        (s['signal'], s['actual'])
        for s in signals
        if s.get('actual') and s['signal'] in ('BUY', 'SELL')
    ]
    n = len(actionals)
    if n < 10:
        return None, None, None, n

    # Point estimate
    correct = sum(
        1 for sig, act in actionals
        if (sig == 'BUY' and act == 'UP') or (sig == 'SELL' and act == 'DOWN')
    )
    point_est = correct / n * 100

    # Bootstrap
    actionals_arr = np.array(actionals)
    boot_accs = []
    for _ in range(n_bootstrap):
        idx     = np.random.choice(n, size=n, replace=True)
        sample  = actionals_arr[idx]
        correct_b = sum(
            1 for sig, act in sample
            if (sig == 'BUY' and act == 'UP') or (sig == 'SELL' and act == 'DOWN')
        )
        boot_accs.append(correct_b / n * 100)

    alpha = (1 - ci) / 2
    lo    = np.percentile(boot_accs, alpha * 100)
    hi    = np.percentile(boot_accs, (1 - alpha) * 100)
    return point_est, lo, hi, n


def _make_mock_signals(df, feature_cols, window=200, replay_hours=300, model_name='RF'):
    """Generate signals list in the same format as generate_signals() in production."""
    models = {
        'RF': lambda: RandomForestClassifier(n_estimators=50, max_depth=5,
                                              class_weight='balanced', random_state=42, n_jobs=1),
        'LR': lambda: LogisticRegression(max_iter=200, class_weight='balanced',
                                          random_state=42, C=0.5),
    }
    factory = models.get(model_name, models['RF'])
    n = len(df)
    start_idx = max(window + 50, n - replay_hours)
    signals = []

    for i in range(start_idx, n):
        train = df.iloc[max(0, i-window):i]
        X_train = train[feature_cols].values
        y_train = train['label'].values
        X_test  = df.iloc[i:i+1][feature_cols].values
        price   = float(df.iloc[i]['close'])

        if len(np.unique(y_train)) < 2:
            continue
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        m = factory()
        m.fit(X_train_s, y_train)
        pred  = m.predict(X_test_s)[0]
        proba = m.predict_proba(X_test_s)[0][1]

        signal     = 'BUY' if pred == 1 else 'SELL'
        confidence = proba * 100 if pred == 1 else (1 - proba) * 100

        actual = None
        if i + 4 < n:
            future = df.iloc[i + 4]['close']
            actual = 'UP' if future > price else 'DOWN'

        signals.append({
            'datetime':   df.iloc[i]['datetime'].strftime('%Y-%m-%d %H:%M'),
            'close':      price,
            'signal':     signal,
            'confidence': round(confidence, 1),
            'actual':     actual,
        })

    return signals


def test_bootstrap_ci(df, feature_cols):
    """Test bootstrap CI on mock signals."""
    print(f"\n{'='*60}")
    print(f"  IMPROVEMENT 2: BOOTSTRAP CONFIDENCE INTERVAL")
    print(f"{'='*60}")
    print(f"  Generating mock signals for CI test...")

    signals = _make_mock_signals(df, feature_cols, window=100, replay_hours=250)
    print(f"  Generated {len(signals)} signals")

    point, lo, hi, n = bootstrap_ci(signals, n_bootstrap=N_BOOTSTRAP)

    if point is None:
        print("  Not enough actionable signals for CI")
        return

    width = hi - lo
    print(f"\n  Accuracy point estimate: {point:.1f}%")
    print(f"  95% CI (n={n} predictions): [{lo:.1f}% – {hi:.1f}%]")
    print(f"  CI width: {width:.1f}%")

    if width > 20:
        reliability = "⚠  WIDE — not enough predictions, treat accuracy as rough estimate"
    elif width > 12:
        reliability = "⚡  MODERATE — directionally informative"
    else:
        reliability = "✓  NARROW — high confidence in accuracy estimate"

    print(f"  Reliability: {reliability}")

    # Show effect of sample size on CI width
    print(f"\n  CI width vs prediction count (reference table):")
    print(f"  {'n_predictions':<18} {'CI width (est)':<18} {'Status'}")
    print(f"  {'-'*55}")
    for n_ref in [30, 50, 100, 150, 200]:
        # Approximate CI width using normal approximation for proportion
        # at ~65% accuracy (typical)
        p = 0.65
        margin = 1.96 * np.sqrt(p * (1-p) / n_ref) * 100
        status = "✓" if margin * 2 < 12 else "⚡" if margin * 2 < 20 else "⚠"
        print(f"  {n_ref:<18} {margin*2:<17.1f}% {status}")

    print(f"""
  ─── INTEGRATION NOTES ──────────────────────────────────
  REAL SYSTEM: in run_mode_b() after generate_signals(),
  and in _run_quick_asset() for the combined summary:

    point, lo, hi, n = bootstrap_ci(signals)
    if lo is not None:
        print(f"  Accuracy: {{point:.1f}}% [95% CI: {{lo:.1f}}%–{{hi:.1f}}%] (n={{n}})")

  Zero extra compute — runs on the existing signals list.
  ────────────────────────────────────────────────────────""")

    return {'point': point, 'lo': lo, 'hi': hi, 'n': n}


# ============================================================
# IMPROVEMENT 3: PERMUTATION SIGNIFICANCE TEST
# REAL SYSTEM: call once after best_config is chosen in
#              run_mode_d() as a one-time validation gate
# ============================================================
def permutation_significance(df, feature_cols, window=200,
                              real_result=None, model_name='RF', n_perm=N_PERMUTATIONS):
    """
    Shuffle labels N times, run full eval each time.
    p-value = fraction of null runs that beat real accuracy.

    Real system usage: run after best_config is found.
    If p > 0.05, warn that the model may not be learning real signal.
    """
    print(f"\n{'='*60}")
    print(f"  IMPROVEMENT 3: PERMUTATION SIGNIFICANCE TEST")
    print(f"{'='*60}")

    if real_result is None:
        print("  Computing real accuracy first...")
        t0 = time.time()
        real_result = _eval_config(df, feature_cols, window=window, model_name=model_name)
        print(f"  Real eval done in {time.time()-t0:.1f}s")

    if real_result is None:
        print("  Could not compute real result")
        return

    real_acc = real_result['accuracy'] / 100
    real_ret = real_result['cum_return']
    print(f"  Real model: acc={real_acc*100:.1f}%  return={real_ret:+.1f}%")
    print(f"  Running {n_perm} permutations... (using every 10th for speed in mock)")

    # For speed in mock: use reduced permutations
    # In production you'd use all N_PERMUTATIONS=500
    n_perm_actual = min(n_perm, 50)
    null_accs = []
    null_rets  = []

    t0 = time.time()
    for i in range(n_perm_actual):
        df_perm = df.copy()
        df_perm['label'] = np.random.permutation(df_perm['label'].values)
        result = _eval_config(df_perm, feature_cols, window=window, model_name=model_name)
        if result:
            null_accs.append(result['accuracy'] / 100)
            null_rets.append(result['cum_return'])
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{n_perm_actual} done ({elapsed:.0f}s elapsed)...")

    elapsed = time.time() - t0
    print(f"  Permutation test done in {elapsed:.1f}s")

    # p-value: fraction of null runs beating real
    p_acc = np.mean([a >= real_acc for a in null_accs])
    p_ret = np.mean([r >= real_ret for r in null_rets])

    null_mean = np.mean(null_accs) * 100
    null_std  = np.std(null_accs)  * 100
    z_score   = (real_acc * 100 - null_mean) / max(null_std, 0.001)

    print(f"\n  {'Metric':<25} {'Real':>10} {'Null mean':>12} {'p-value':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Accuracy':<25} {real_acc*100:>9.1f}% {null_mean:>11.1f}% {p_acc:>10.3f}")
    print(f"  {'Return':<25} {real_ret:>+9.1f}% {'—':>12} {p_ret:>10.3f}")
    print(f"  {'Z-score (accuracy)':<25} {z_score:>+10.2f}")

    if p_acc < 0.01:
        verdict = "✓  HIGHLY SIGNIFICANT (p<0.01) — model is learning real signal"
    elif p_acc < 0.05:
        verdict = "✓  SIGNIFICANT (p<0.05) — model likely learning real signal"
    elif p_acc < 0.10:
        verdict = "⚡  MARGINAL (p<0.10) — borderline, increase data or features"
    else:
        verdict = "⚠  NOT SIGNIFICANT (p>{:.2f}) — could be luck, do not trade".format(p_acc)

    print(f"\n  Result: {verdict}")

    print(f"""
  ─── INTEGRATION NOTES ──────────────────────────────────
  REAL SYSTEM: in run_mode_d(), after best_config is found,
  add as a validation gate (optional, adds ~2 min per asset):

    if best_config:
        p_val, _ = permutation_significance(
            df_diag, optimal_features,
            window=best_config['best_window'],
            real_result=best_config,
            n_perm=500
        )
        best_config['p_value'] = round(p_val, 3)
        if p_val > 0.10:
            print(f"  ⚠ WARNING: p={{p_val:.3f}} — model may not be significant!")

  Save p_value to CSV so it's visible in Mode B output.
  Rule of thumb: p > 0.10 on final model → run Mode D again
  with different diag_years or after more data accumulates.
  ────────────────────────────────────────────────────────""")

    return {'p_acc': p_acc, 'p_ret': p_ret, 'z_score': z_score, 'null_accs': null_accs}


# ============================================================
# IMPROVEMENT 4: SCORING COMPARISON
# REAL SYSTEM: replace combined_score block in _eval_one_config
# ============================================================
def test_scoring_comparison(df, feature_cols):
    """
    Compare old heuristic score vs new Calmar/Sharpe score across configs.
    Shows where they agree and disagree — disagreements are the interesting cases.
    """
    print(f"\n{'='*60}")
    print(f"  IMPROVEMENT 4: CALMAR/SHARPE vs HEURISTIC SCORE")
    print(f"{'='*60}")

    configs = [
        ('RF',  100), ('RF',  200),
        ('LR',  100), ('LR',  200),
        ('GB',  100),
    ]

    results = []
    print(f"  Evaluating {len(configs)} configs...")
    for model_name, window in configs:
        t0 = time.time()
        r = _eval_config(df, feature_cols, window=window, model_name=model_name)
        elapsed = time.time() - t0
        if r:
            results.append(r)
            print(f"    {model_name} w={window:3d}h | acc={r['accuracy']:5.1f}% "
                  f"ret={r['cum_return']:+6.1f}% dd={r['max_dd']:5.1f}% "
                  f"old={r['combined_score_old']:.3f} "
                  f"new={r['combined_score_new']:.3f} "
                  f"calmar={r['calmar']:+.2f} sharpe={r['sharpe']:+.2f}  ({elapsed:.1f}s)")

    if not results:
        print("  No results")
        return

    # Rankings
    ranked_old = sorted(results, key=lambda x: -x['combined_score_old'])
    ranked_new = sorted(results, key=lambda x: -x['combined_score_new'])

    print(f"\n  OLD RANKING (heuristic: acc × profit_factor^1.5 × dd_penalty):")
    for rank, r in enumerate(ranked_old, 1):
        marker = " ← WINNER" if rank == 1 else ""
        print(f"    #{rank}  {r['model']} w={r['window']:3d}h | "
              f"acc={r['accuracy']:.1f}% ret={r['cum_return']:+.1f}% "
              f"score={r['combined_score_old']:.3f}{marker}")

    print(f"\n  NEW RANKING (Calmar/Sharpe blend):")
    for rank, r in enumerate(ranked_new, 1):
        marker = " ← WINNER" if rank == 1 else ""
        print(f"    #{rank}  {r['model']} w={r['window']:3d}h | "
              f"calmar={r['calmar']:+.2f} sharpe={r['sharpe']:+.2f} "
              f"score={r['combined_score_new']:.3f}{marker}")

    old_winner = ranked_old[0]
    new_winner = ranked_new[0]
    same = (old_winner['model'] == new_winner['model'] and
            old_winner['window'] == new_winner['window'])

    if same:
        print(f"\n  ✓  Both scores pick the same winner: "
              f"{old_winner['model']} w={old_winner['window']}h")
    else:
        print(f"\n  ⚡  Scores disagree on winner:")
        print(f"     Old: {old_winner['model']} w={old_winner['window']}h "
              f"(acc={old_winner['accuracy']:.1f}% ret={old_winner['cum_return']:+.1f}%)")
        print(f"     New: {new_winner['model']} w={new_winner['window']}h "
              f"(acc={new_winner['accuracy']:.1f}% ret={new_winner['cum_return']:+.1f}%)")
        print(f"     → Investigate: old score may be favouring return without "
              f"penalising drawdown risk enough")

    print(f"""
  ─── INTEGRATION NOTES ──────────────────────────────────
  REAL SYSTEM: in _eval_one_config(), replace the block:

    profit_factor = 1 + cum_return / 100
    dd_penalty = max(0.5, 1.0 - max(0, max_dd - 0.20) * 0.5)
    combined_score = accuracy * (max(profit_factor, 0.01) ** 1.5) * dd_penalty

  With:

    calmar  = _calmar(cum_return, max_dd * 100, total, step)
    sharpe  = _sharpe(trade_returns)   # need list collected in loop
    combined_score = _score_new(accuracy*100, cum_return, max_dd*100, total, step, trade_returns)

  Also add 'calmar' and 'sharpe' to the return tuple so they
  appear in the top-5 summary table in run_diagnostic_for_asset().

  NOTE: trade_returns must be collected as a list in the loop
  (one float per closed trade). The loop already tracks total_gain
  and total_loss — just replace with a list.
  ────────────────────────────────────────────────────────""")

    return results


# ============================================================
# MAIN: RUN ALL 4 IMPROVEMENTS
# ============================================================
def main():
    print("=" * 60)
    print("  MOCK CRYPTO TRADING SYSTEM — IMPROVEMENT TESTS")
    print("  Testing 4 new features on synthetic data")
    print("=" * 60)

    # Use only the non-noise features for a cleaner test
    # (noise features are included to verify feature scoring would catch them)
    informative_cols = [
        'logret_1h', 'logret_4h', 'logret_8h', 'logret_12h',
        'logret_24h', 'logret_48h', 'logret_72h',
        'rsi_14h', 'volatility_12h', 'vol_ratio', 'sma_ratio',
    ]

    print("\n  Generating synthetic data...")
    df = generate_synthetic_data(n_hours=4_000)

    # Use a smaller subset for speed in the mock
    df_small = df.tail(2000).reset_index(drop=True)
    print(f"  Using last {len(df_small):,} rows for tests")

    t_total = time.time()

    # ── Improvement 1: Holdout ─────────────────────────────
    holdout_result = test_holdout(df_small, informative_cols, window=100)

    # ── Improvement 2: Bootstrap CI ───────────────────────
    ci_result = test_bootstrap_ci(df_small, informative_cols)

    # ── Improvement 3: Permutation significance ───────────
    # Use the train result from holdout to avoid re-running eval
    real_res = holdout_result['train'] if holdout_result else None
    perm_result = permutation_significance(
        df_small.iloc[:-HOLDOUT_HOURS].reset_index(drop=True),
        informative_cols, window=100,
        real_result=real_res,
        n_perm=50    # reduced for mock speed; use 500 in production
    )

    # ── Improvement 4: Scoring comparison ─────────────────
    score_result = test_scoring_comparison(df_small, informative_cols)

    # ── Final summary ─────────────────────────────────────
    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  ALL TESTS COMPLETE  ({total_elapsed:.0f}s total)")
    print(f"{'='*60}")
    print(f"""
  INTEGRATION CHECKLIST (copy to real system one at a time):

  [ ] 1. HOLDOUT: add HOLDOUT_HOURS=500 split in run_mode_d()
         before run_feature_analysis(); print overfit gap after
         best_config is found. Gate: >10% gap = reject.

  [ ] 2. BOOTSTRAP CI: add bootstrap_ci() call after
         generate_signals() in run_mode_b() and _run_quick_asset().
         Print: "Accuracy: X% [95% CI: Y%–Z%] (n=N)"

  [ ] 3. PERMUTATION TEST: add permutation_significance() after
         best_config is found in run_mode_d(). Save p_value to CSV.
         Gate: p > 0.10 = warn; p > 0.20 = hard reject.

  [ ] 4. CALMAR/SHARPE: replace combined_score formula in
         _eval_one_config(). Collect trade_returns list in loop.
         Add calmar + sharpe to return tuple and top-5 table.
""")


if __name__ == '__main__':
    main()
