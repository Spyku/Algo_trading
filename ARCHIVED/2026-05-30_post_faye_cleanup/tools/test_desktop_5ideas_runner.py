"""
Desktop runner — tests 7 SUSPECT ideas + C05/C06 post-hoc rerank,
~7-8h total wall-clock.

C56 HMM and C04 Variance Ratio were already retested on the fixed harness
(2026-05-06 evening) — both FAIL, see logs/c56_only_summary_20260506_195258.txt
and logs/c04_to_c08_summary_20260506_213526.txt. Removed from this batch.

C05 CVaR_5% and C06 Sortino reranks were also attempted in the c04_to_c08
runner but crashed with KeyError('models') (bug fixed). Bundled here so a
single overnight launch covers all remaining inconclusive ideas.

Ideas (in execution order — Tier S/A first, by structural-change priority):
  C35  Wavelet multi-scale decomposition (Daubechies db4 levels 1-4) as features.
       PRIOR: BTC 8h beat baseline +40pp on prior engine (lit_v2 wavelet_denoising).
  C57  Markov-switching AR — added as state probability feature.
  C50  PF as primary scoring — flip OPTUNA_METRIC from 'apf' to 'rawpf'.
  C47  Vol-adjusted label — replace label = (ret > 2×fee) with (ret/σ_h > k).
  C44  Quantile regression — add LGBM-quantile-prediction (alpha=0.7) as feature.
  C42  CatBoost ensemble — replaces XGB+LGBM with CB+LGBM combo.
  C43  Stacking meta-learner — sklearn StackingClassifier replacing voting.
  C05  CVaR_5% objective — post-hoc rerank top-15 baseline-grid candidates.
  C06  Sortino objective — same backtest as C05 (zero marginal cost).

Each idea runs Mode D × 4 horizons (5,6,7,8) on ETH with --replay 1440
--no-persist --no-data-update --grid-tag <IDEA_TAG>. Compares top-APF in
tagged grid CSV vs untagged baseline. Verdict thresholds:
  delta >= +5pp avg -> PASS (escalate to HRST validation)
  delta > 0         -> MARGINAL
  delta <= 0        -> FAIL

Run on Desktop with the venv already active:
  python tools/test_desktop_5ideas_runner.py

Honest time breakdown (per-horizon → 4-horizon total):
  C35 wavelet ~7-9  min   →  ~30 min
  C57 MS-AR   ~10-15 min  →  ~50 min  (statsmodels regime fit is heavy)
  C50 PF      ~6-8  min   →  ~28 min
  C47 vol-adj ~6-8  min   →  ~28 min
  C44 quantile ~7-9 min   →  ~32 min
  C42 CatBoost ~15-25 min →  ~80 min
  C43 Stacking ~30-60 min → ~150-240 min
  C05/C06 joint rerank      ~90 min  (15 backtests × 4 horizons)
TOTAL: ~7-8 hours hands-off. C43 dominates.
"""
import os, sys
from datetime import datetime

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)
sys.path.insert(0, os.path.join(ENGINE, 'tools'))

from test_14_ideas import (write_patcher, run_mode_d, load_grid_csv,
                            load_baseline_grid, compare_grid_winners,
                            LOGS_DIR, MODELS_DIR, TS)

ASSET = 'ETH'
HORIZONS = [5, 6, 7, 8]
REPLAY = 1440

# =============================================================================
# C35 — Wavelet multi-scale decomposition
# =============================================================================
C35_PATCHER = '''
"""C35 wavelet multi-scale decomposition (Daubechies db4 levels 1-4)."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_all_features


def _wavelet_features(df: pd.DataFrame) -> dict:
    try:
        import pywt
    except ImportError:
        print('[C35] pywavelets not installed: pip install PyWavelets')
        return {}
    if 'close' not in df.columns:
        return {}
    logret = np.log(df['close']).diff().fillna(0).values
    out = {}
    window = 256
    n = len(logret)
    for level in range(1, 5):
        coeffs_series = np.full(n, np.nan)
        for i in range(window, n):
            try:
                segment = logret[i - window:i]
                coeffs = pywt.wavedec(segment, 'db4', level=4)
                # coeffs[0] = approx, coeffs[1..4] = details (level 4..1)
                detail_idx = 5 - level
                detail = coeffs[detail_idx]
                coeffs_series[i] = float(np.std(detail))
            except Exception:
                pass
        out[f'wavelet_d4_lvl{level}_std'] = pd.Series(coeffs_series, index=df.index)
    return out


def _patched_build(*args, **kwargs):
    res = _orig_build(*args, **kwargs)
    if not isinstance(res, tuple):
        return res
    df, cols = res
    new = _wavelet_features(df)
    for c, vals in new.items():
        df[c] = vals
        if c not in cols:
            cols.append(c)
    print(f'[C35] wavelet features added: +{len(new)} columns')
    return df, cols


eng.build_all_features = _patched_build
print('[C35] build_all_features patched (+wavelet decomposition)')
'''

# =============================================================================
# C50 — PF as primary scoring
# =============================================================================
C50_PATCHER = '''
"""C50 raw profit factor as primary Optuna metric (instead of APF)."""
import crypto_trading_system_ed as eng

_orig_metric = getattr(eng, 'OPTUNA_METRIC', 'apf')
eng.OPTUNA_METRIC = 'rawpf'
print(f'[C50] OPTUNA_METRIC: {_orig_metric!r} -> {eng.OPTUNA_METRIC!r} (raw profit factor)')
'''

# =============================================================================
# C42 — CatBoost ensemble
# =============================================================================
C42_PATCHER = '''
"""C42 CatBoost as 4th ensemble model. Adds CB+LGBM combo to grid."""
import crypto_trading_system_ed as eng

# Add CatBoost to model dispatch
try:
    from catboost import CatBoostClassifier
    _CB_AVAILABLE = True
except ImportError:
    print('[C42] catboost not installed: pip install catboost')
    _CB_AVAILABLE = False

if _CB_AVAILABLE:
    _orig_models = eng.ALL_MODELS

    def _make_cb():
        return CatBoostClassifier(
            iterations=300, depth=4, learning_rate=0.05,
            random_seed=42, verbose=0, thread_count=1,
            auto_class_weights='Balanced',
        )

    eng.ALL_MODELS = dict(_orig_models)
    eng.ALL_MODELS['CB'] = _make_cb

    # Patch DIAG_MODELS (used in Mode D grid via _get_deku_diagnostic_models)
    if hasattr(eng, 'DIAG_MODELS'):
        eng.DIAG_MODELS = dict(eng.DIAG_MODELS)
        eng.DIAG_MODELS['CB'] = lambda: CatBoostClassifier(
            iterations=100, depth=4, learning_rate=0.05,
            random_seed=42, verbose=0, thread_count=1,
            auto_class_weights='Balanced',
        )

    # Inject CB+LGBM into GRID_COMBOS (replace one or add)
    _orig_combos = list(eng.GRID_COMBOS)
    if 'CB+LGBM' not in _orig_combos:
        eng.GRID_COMBOS = ['RF+LGBM', 'CB+LGBM']  # cap at 2 for time budget
        print(f'[C42] GRID_COMBOS: {_orig_combos} -> {eng.GRID_COMBOS}')
    print('[C42] CatBoost added to ALL_MODELS and DIAG_MODELS')
'''

# =============================================================================
# C43 — Stacking meta-learner
# =============================================================================
C43_PATCHER = '''
"""C43 Stacking meta-learner. Adds STACK pseudo-combo using sklearn StackingClassifier."""
import crypto_trading_system_ed as eng

_orig_models = eng.ALL_MODELS

def _make_stack():
    from sklearn.ensemble import StackingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=4,
                                       class_weight='balanced', random_state=42, n_jobs=1)),
        ('lgbm', LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  class_weight='balanced', verbose=-1, random_state=42)),
    ]
    if XGBClassifier is not None:
        estimators.append(('xgb', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                                  random_state=42, tree_method='hist',
                                                  verbosity=0, n_jobs=1)))
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
        cv=3, n_jobs=1, passthrough=False,
    )

eng.ALL_MODELS = dict(_orig_models)
eng.ALL_MODELS['STACK'] = _make_stack
if hasattr(eng, 'DIAG_MODELS'):
    eng.DIAG_MODELS = dict(eng.DIAG_MODELS)
    eng.DIAG_MODELS['STACK'] = _make_stack

# Use STACK alone (it's already an ensemble — paired with itself or LGBM as second voter)
_orig_combos = list(eng.GRID_COMBOS)
eng.GRID_COMBOS = ['STACK+LGBM', 'STACK+RF']  # Stack votes with another base
print(f'[C43] GRID_COMBOS: {_orig_combos} -> {eng.GRID_COMBOS}')
print('[C43] StackingClassifier added to ALL_MODELS / DIAG_MODELS')
'''


# =============================================================================
# C57 — Markov-switching AR detector (state probability as feature)
# =============================================================================
C57_PATCHER = '''
"""C57 Markov-switching AR(1) — adds 2-state probability as feature."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_all_features


def _msar_features(df: pd.DataFrame) -> dict:
    try:
        from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
    except ImportError:
        print('[C57] statsmodels MS-AR not available')
        return {}
    if 'close' not in df.columns:
        return {}
    logret = np.log(df['close']).diff().dropna().values * 100  # %
    n_full = len(df)
    out_state1 = np.full(n_full, 0.5)
    fit_window = 720
    step = 48
    # Align: logret has n_full - 1 entries (diff dropped first)
    offset = n_full - len(logret)
    for end in range(fit_window, len(logret) + 1, step):
        try:
            seg = logret[max(0, end - fit_window):end]
            model = MarkovAutoregression(seg, k_regimes=2, order=1, switching_ar=True)
            res = model.fit(disp=False, maxiter=50)
            # smoothed prob of high-vol regime (assume regime 1 = higher variance)
            sp = res.smoothed_marginal_probabilities[1]
            offs = end - len(sp)
            slot_start = max(0, end - step) + offset
            slot_end = end + offset
            sp_slot = sp[max(0, len(sp) - step):]
            n_slot = min(len(sp_slot), slot_end - slot_start)
            if n_slot > 0:
                out_state1[slot_start:slot_start + n_slot] = sp_slot[-n_slot:]
        except Exception:
            pass
    return {'msar_state1_prob': pd.Series(out_state1, index=df.index)}


def _patched_build(*args, **kwargs):
    res = _orig_build(*args, **kwargs)
    if not isinstance(res, tuple):
        return res
    df, cols = res
    new = _msar_features(df)
    for c, vals in new.items():
        df[c] = vals
        if c not in cols:
            cols.append(c)
    print(f'[C57] MS-AR features added: +{len(new)} columns')
    return df, cols


eng.build_all_features = _patched_build
print('[C57] build_all_features patched (+MS-AR state probability)')
'''

# =============================================================================
# C47 — Vol-adjusted label (ret_h / σ_h > threshold)
# =============================================================================
C47_PATCHER = '''
"""C47 vol-adjusted label. Replace binary (ret > 2×fee) with (ret/σ_h > 0.5)."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_hourly_features

VOL_ADJ_K = 0.5  # threshold on Sharpe-like ratio (ret_h / σ_h)


def _patched_hourly(*args, **kwargs):
    res = _orig_build(*args, **kwargs)
    if not isinstance(res, tuple):
        return res
    df, feature_cols = res
    if 'label' not in df.columns or '_forward_return' not in df.columns:
        return df, feature_cols
    # Recompute label = (forward_ret / forward_vol > k)
    fr = df['_forward_return']
    # Backward-looking realized vol (proxy for forward σ)
    if 'volatility_48h' in df.columns:
        sigma = df['volatility_48h']
    else:
        logret = np.log(df['close']).diff()
        sigma = logret.rolling(48, min_periods=12).std()
    sharpe_like = fr / (sigma + 1e-9)
    new_label = (sharpe_like > VOL_ADJ_K).astype(float)
    df['label'] = new_label.where(fr.notna(), np.nan)
    print(f'[C47] vol-adjusted label active: k={VOL_ADJ_K}, positives={int(new_label.sum())}/{len(new_label)}')
    return df, feature_cols


eng.build_hourly_features = _patched_hourly
print('[C47] build_hourly_features patched (vol-adjusted label)')
'''

# =============================================================================
# C44 — Quantile regression target (LGBM quantile prediction as feature)
# =============================================================================
C44_PATCHER = '''
"""C44 quantile regression. Trains LGBM with objective='quantile', alpha=0.7
on forward returns, uses prediction as feature for the main classifier."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_all_features


def _quantile_feature(df: pd.DataFrame, feature_cols: list, alpha: float = 0.7) -> dict:
    """Walk-forward LGBM-quantile predictions on forward return."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return {}
    if '_forward_return' not in df.columns:
        return {}
    if not feature_cols:
        return {}
    X_all = df[feature_cols].copy()
    # Simple: rolling fit every 168h (1 week), predict next 168h
    n = len(df)
    fr = df['_forward_return'].values
    pred = np.full(n, np.nan)
    fit_window = 720
    step = 168
    cols_finite = [c for c in feature_cols if df[c].notna().mean() > 0.5]
    if not cols_finite:
        return {}
    X_all = df[cols_finite].fillna(0).values
    for end in range(fit_window, n, step):
        try:
            X_tr = X_all[max(0, end - fit_window):end]
            y_tr = fr[max(0, end - fit_window):end]
            mask = ~np.isnan(y_tr)
            if mask.sum() < 100:
                continue
            mdl = LGBMRegressor(
                objective='quantile', alpha=alpha,
                n_estimators=100, max_depth=4, learning_rate=0.05,
                verbose=-1, random_state=42, n_jobs=1,
            )
            mdl.fit(X_tr[mask], y_tr[mask])
            X_pred = X_all[end:min(n, end + step)]
            pred[end:min(n, end + step)] = mdl.predict(X_pred)
        except Exception:
            pass
    return {'qreg_q70_predict': pd.Series(pred, index=df.index)}


def _patched_build(*args, **kwargs):
    res = _orig_build(*args, **kwargs)
    if not isinstance(res, tuple):
        return res
    df, cols = res
    new = _quantile_feature(df, list(cols))
    for c, vals in new.items():
        df[c] = vals
        if c not in cols:
            cols.append(c)
    print(f'[C44] quantile-regressor feature added: +{len(new)} columns')
    return df, cols


eng.build_all_features = _patched_build
print('[C44] build_all_features patched (+LGBM quantile q=0.7 prediction)')
'''


def run_idea(name, tag, patcher_code, summary):
    """Write patcher, run Mode D × 4 horizons, return (idea_name, avg_delta, per_horizon_deltas)."""
    print('=' * 100)
    print(f'  RUNNING {name}  (tag={tag})')
    print('=' * 100)
    write_patcher(name, patcher_code)
    log_path = os.path.join(LOGS_DIR, f'{name}_{TS}.log')
    print(f'  Log: {log_path}')

    deltas = []
    for h in HORIZONS:
        print(f'  >> Mode D ETH {h}h ...')
        rc = run_mode_d(ASSET, h, REPLAY, tag, f'_idea_patchers.{name}', log_path)
        if rc != 0:
            print(f'     ERROR rc={rc}')
            continue
        test_df, _ = load_grid_csv(ASSET, h, tag)
        base_df, _ = load_baseline_grid(ASSET, h)
        if test_df is None or base_df is None:
            print(f'     ERROR loading grid CSVs')
            continue
        tw, bw, delta = compare_grid_winners(test_df, base_df, sort_col='apf')
        if delta is None:
            print(f'     ERROR comparison failed')
            continue
        deltas.append((h, tw, bw, delta))
        print(f'     {h}h: test_apf={tw:.3f}  base_apf={bw:.3f}  delta={delta:+.3f}')

    if not deltas:
        verdict = 'ERROR'
        avg = float('nan')
    else:
        avg = sum(d[3] for d in deltas) / len(deltas)
        if avg >= 5:
            verdict = 'PASS'
        elif avg > 0:
            verdict = 'MARGINAL'
        else:
            verdict = 'FAIL'

    summary.append({
        'name': name, 'tag': tag, 'avg_delta': avg, 'verdict': verdict,
        'per_horizon': deltas,
    })
    print(f'\n  >>> {name}: avg APF delta = {avg:+.3f} -> {verdict}\n')
    return summary


def _preflight():
    """Check optional dependencies. Print what's missing — patchers will skip
    cleanly but the corresponding idea won't have effect."""
    print('  Pre-flight dependency check:')
    import importlib
    deps = {
        'pywt': 'C35 wavelet decomposition (PyWavelets)',
        'statsmodels.tsa.regime_switching.markov_autoregression': 'C57 Markov-switching AR',
        'catboost': 'C42 CatBoost ensemble',
    }
    missing = []
    for pkg, idea in deps.items():
        try:
            importlib.import_module(pkg)
            print(f'    {pkg:<12} OK   ({idea})')
        except ImportError:
            print(f'    {pkg:<12} MISSING — install via `pip install {pkg}` for {idea} to fire')
            missing.append(pkg)
    if missing:
        print(f'\n  WARNING: {len(missing)} package(s) missing. Run:')
        print(f'    pip install {" ".join(missing)}')
        print(f'  ...then re-launch this script. (Or proceed and the corresponding ideas will no-op.)\n')
    else:
        print('    All optional deps present.\n')


def main():
    print('=' * 100)
    print(f'  DESKTOP SUSPECTS RUNNER — {datetime.now().isoformat()}')
    print(f'  Asset: {ASSET}  Horizons: {HORIZONS}  Replay: {REPLAY}h')
    print(f'  Ideas: C35 wavelets, C57 MS-AR, C50 PF, C47 vol-adj, C44 quantile, C42 CatBoost, C43 Stacking')
    print(f'  (C56 HMM and C04 VR already retested 2026-05-06 — both FAIL)')
    print('=' * 100)
    print()
    _preflight()

    summary = []
    # Order: Tier S (positive prior) → Tier A (structural) → heaviest last
    # Cheap-and-fast first so partial results are useful if you abort early
    run_idea('c35_wavelet', 'C35WV', C35_PATCHER, summary)           # Tier S, ~30 min
    run_idea('c50_pf_objective', 'C50PF', C50_PATCHER, summary)      # Tier A, ~28 min
    run_idea('c47_vol_adj_label', 'C47VL', C47_PATCHER, summary)     # Tier A, ~28 min
    run_idea('c44_quantile_reg', 'C44QR', C44_PATCHER, summary)      # Tier A, ~32 min
    run_idea('c57_msar_regime', 'C57MS', C57_PATCHER, summary)       # Tier A, ~50 min
    run_idea('c42_catboost', 'C42CB', C42_PATCHER, summary)          # Tier A, ~80 min
    run_idea('c43_stacking', 'C43ST', C43_PATCHER, summary)          # Tier A, ~150-240 min (heaviest, last)

    # C05 + C06 — post-hoc rerank of top-15 baseline-grid candidates with
    # per-trade-return capture. Shape is different from the patcher loop
    # (no Mode D run, just custom backtest + scoring). Imported so a single
    # launch covers all remaining inconclusive ideas.
    print('=' * 100)
    print('  C05 + C06 — CVaR_5% + Sortino post-hoc rerank')
    print('=' * 100)
    try:
        from test_c04_to_c08_runner import run_c05_c06
        c05_c06 = run_c05_c06()
        summary.append({
            'name': 'c05_c06_rerank', 'tag': 'C05CV/C06SR',
            'avg_delta': float('nan'),
            'verdict': c05_c06.get('verdict', 'ERROR'),
            'per_horizon': [],
        })
        if 'rerank_path' in c05_c06:
            print(f"  Rerank table: {c05_c06['rerank_path']}")
            print(f"  Per-trade audit: {c05_c06['audit_path']}")
    except Exception as e:
        print(f'  C05/C06 EXCEPTION: {e!r}')
        summary.append({
            'name': 'c05_c06_rerank', 'tag': 'C05CV/C06SR',
            'avg_delta': float('nan'), 'verdict': f'EXCEPTION: {e!r}',
            'per_horizon': [],
        })

    print('=' * 100)
    print('  FINAL SUMMARY')
    print('=' * 100)
    for r in summary:
        avg = r['avg_delta']
        avg_s = f'{avg:+.3f}' if avg == avg else 'n/a'  # NaN-safe
        print(f"  {r['name']:<25} verdict={r['verdict']:<25} avg_delta={avg_s}")
        for h, tw, bw, d in r['per_horizon']:
            print(f"    {h}h: test={tw:.3f}  base={bw:.3f}  delta={d:+.3f}")

    out_summary = os.path.join(LOGS_DIR, f'desktop_5ideas_summary_{TS}.txt')
    with open(out_summary, 'w', encoding='utf-8') as f:
        f.write(f'Desktop suspects runner {TS}\n')
        f.write(f'Asset={ASSET} horizons={HORIZONS} replay={REPLAY}h\n\n')
        for r in summary:
            avg = r['avg_delta']
            avg_s = f'{avg:+.3f}' if avg == avg else 'n/a'
            f.write(f"{r['name']}: verdict={r['verdict']} avg_delta={avg_s}\n")
            for h, tw, bw, d in r['per_horizon']:
                f.write(f"  {h}h: test={tw:.3f} base={bw:.3f} delta={d:+.3f}\n")
            f.write('\n')
    print(f'\nSummary file: {out_summary}')


if __name__ == '__main__':
    main()
