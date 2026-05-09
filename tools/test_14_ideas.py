"""test_14_ideas.py — Unified harness to test 14 candidate ideas individually.

Each idea is registered as a function returning a verdict dict. The harness can
run any single idea, a list, or all sequentially. All writes are tagged
`_IDEA_<NN>` so production is never touched.

Usage:
  python tools/test_14_ideas.py --list                      # print all 14 ideas
  python tools/test_14_ideas.py --idea 1                    # run idea #1
  python tools/test_14_ideas.py --idea har_rv               # run by name
  python tools/test_14_ideas.py --idea 1,3,5                # run a subset
  python tools/test_14_ideas.py --all                       # run all 14 sequentially
  python tools/test_14_ideas.py --quick                     # use --replay 720
  python tools/test_14_ideas.py --asset ETH --horizons 5,6,7,8

Outputs:
  output/test_14_<idea>_<ts>.csv     per-idea result CSV
  logs/test_14_<idea>_<ts>.log       per-idea console log
  logs/test_14_summary_<ts>.txt      aggregated verdict table

Verdict for each idea:
  PASS      smoke test wins by >= +5pp vs baseline on the relevant metric
  MARGINAL  delta in (0, +5pp) — within HRST run-to-run noise
  FAIL      delta <= 0
  ERROR     idea crashed (test infra or engine error)
  STUB      idea is registered but the implementation is too architectural
            for a smoke test in this harness — flagged as needing a dedicated
            tool. Verdict reflects what would need to be built.

Design:
  - Ideas in 5 categories: feature-injection, scoring-change, filter-overlay,
    detector, architectural. Each category has a corresponding mini-runner.
  - Most use the existing engine in --no-persist --no-data-update mode via a
    subprocess + monkey-patcher pattern (pioneered by test_fracdiff_mode_d.py).
  - Each idea is independently revertable: running idea #N never affects
    idea #M's environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ENGINE)
sys.path.insert(0, ENGINE)

TS = datetime.now().strftime('%Y%m%d_%H%M%S')
LOGS_DIR = os.path.join(ENGINE, 'logs')
OUT_DIR = os.path.join(ENGINE, 'output')
MODELS_DIR = os.path.join(ENGINE, 'models')
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# Registry
# =============================================================================

IDEAS: List[Dict[str, Any]] = []


def register(num: int, name: str, category: str, summary: str,
             effort: str, est_runtime: str):
    """Decorator: register a test function as idea #num."""
    def deco(fn):
        IDEAS.append({
            'num': num,
            'name': name,
            'category': category,
            'summary': summary,
            'effort': effort,
            'est_runtime': est_runtime,
            'fn': fn,
        })
        return fn
    return deco


def find_idea(selector: str) -> Optional[Dict[str, Any]]:
    """Resolve --idea selector (number or name) to a registered idea dict."""
    selector = selector.strip().lower()
    for idea in IDEAS:
        if str(idea['num']) == selector or idea['name'].lower() == selector:
            return idea
    return None


# =============================================================================
# Shared helpers (used by multiple ideas)
# =============================================================================

def write_patcher(name: str, code: str) -> str:
    """Write a temp patcher module + sitecustomize-style hook for subprocesses.
    Returns the absolute path of the patcher .py file."""
    path = os.path.join(ENGINE, '_idea_patchers', f'{name}.py')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    init_path = os.path.join(os.path.dirname(path), '__init__.py')
    if not os.path.exists(init_path):
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write('# auto-generated; do not edit\n')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(code)
    return path


def run_mode_d(asset: str, horizon: int, replay: int, tag: str,
               patcher_module: Optional[str], log_path: str,
               extra_args: Optional[List[str]] = None) -> int:
    """Spawn a Mode D subprocess. patcher_module = 'package.module' or None.
    Returns subprocess return code.

    HARNESS BUG FIX (2026-05-05): replaced `runpy.run_module(name,
    run_name='__main__')` with a direct call to `crypto_trading_system_ed.main()`.
    runpy.run_module re-executes the module from scratch in a NEW namespace,
    discarding the patcher's monkey-patches that had been applied to
    sys.modules['crypto_trading_system_ed']. Direct main() call preserves the
    patches because the module remains the same sys.modules entry. Verified by
    checking that patcher's RUNTIME prints (e.g. `[C56] HMM features added`)
    now appear during Mode D execution instead of 0 times pre-fix.
    """
    import subprocess

    extra_args = extra_args or []
    py_init = (
        f"import sys, os; sys.path.insert(0, r'{ENGINE}'); "
        f"os.chdir(r'{ENGINE}'); "
    )
    if patcher_module:
        py_init += f"import {patcher_module}; "
    # Import engine then call its main() — this preserves the patcher's
    # monkey-patches because we DON'T re-execute the engine module.
    py_init += "import crypto_trading_system_ed; crypto_trading_system_ed.main()"

    cmd = [
        sys.executable, '-c',
        f"import sys; sys.argv = ['crypto_trading_system_ed.py', 'D', "
        f"'{asset}', '{horizon}h', '--replay', '{replay}', "
        f"'--no-persist', '--no-data-update', '--grid-tag', '{tag}'"
        + ''.join([f", '{a}'" for a in extra_args]) + "]; "
        + py_init,
    ]
    with open(log_path, 'w', encoding='utf-8') as logf:
        logf.write(f"[idea-runner] {datetime.now().isoformat()} starting Mode D\n")
        logf.write(f"[idea-runner] cmd: {cmd[2][:200]}...\n")
        logf.flush()
        rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            cwd=ENGINE).returncode
    return rc


def load_grid_csv(asset: str, horizon: int, tag: str):
    """Load tagged grid CSV. Returns (df, path). df is None if missing."""
    import pandas as pd
    path = os.path.join(MODELS_DIR, f'crypto_ed_grid_{asset}_{horizon}h_{tag}.csv')
    if not os.path.exists(path):
        return None, path
    df = pd.read_csv(path)
    df_ok = df[df['status'] == 'OK'].copy()
    return df_ok, path


def load_baseline_grid(asset: str, horizon: int):
    """Try to find a recent UNTAGGED Mode D grid CSV to use as baseline."""
    import pandas as pd
    candidates = [
        os.path.join(MODELS_DIR, f'crypto_ed_grid_{asset}_{horizon}h.csv'),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df[df['status'] == 'OK'].copy(), path
    return None, None


def compare_grid_winners(test_df, base_df, sort_col='apf', top_n=1):
    """Return delta in mean-of-top-N on sort_col. Positive = test beats base."""
    if test_df is None or len(test_df) == 0:
        return None, None, None
    if base_df is None or len(base_df) == 0:
        return None, None, None
    tw = test_df.nlargest(top_n, sort_col)[sort_col].mean()
    bw = base_df.nlargest(top_n, sort_col)[sort_col].mean()
    return tw, bw, tw - bw


def write_summary_row(idea_num: int, name: str, status: str,
                      baseline: Any, test_val: Any, delta: Any, notes: str,
                      summary_path: str):
    line = (f"#{idea_num:02d} {name:<32}  status={status:<10} "
            f"baseline={baseline}  test={test_val}  delta={delta}  | {notes}")
    print(line)
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# =============================================================================
# IDEA #1 — Per-regime feature set (bull=technical / bear=macro)
# =============================================================================

@register(1, 'per_regime_features', 'feature-set',
          'Force bull horizons to use technical-heavy features, bear horizons to '
          'use macro-heavy features. Test if forced asymmetry beats organic.',
          'medium', '~30 min')
def idea_per_regime_features(asset: str, horizons: List[int], replay: int,
                              summary_path: str) -> Dict[str, Any]:
    """Monkey-patches Mode D's feature ranking to bias by regime.
    Bull horizons (5h, 6h): boost technical features in ranking.
    Bear horizons (7h, 8h): boost macro features in ranking.
    """
    patcher_code = '''
"""Per-regime feature-set patcher.

Boosts technical features in LGBM ranking for short horizons (bull-bias),
boosts macro features for long horizons (bear-bias). Implemented as a
post-LGBM-ranking reorder.
"""
import crypto_trading_system_ed as eng

_orig_rank = eng._test_lgbm_importance

TECHNICAL_PREFIXES = ('logret_', 'sma', 'price_to_sma', 'spread_', 'volatility_',
                      'gk_volatility', 'rsi_', 'adx_', 'plus_di', 'minus_di',
                      'bb_position', 'stoch_k', 'price_accel', 'atr_', 'zscore_',
                      'vol_ratio', 'intraday_range', 'volume_ratio', 'hour_',
                      'dow_')

MACRO_PREFIXES = ('m_', 'xa_', 'fg_', 'oc_', 'deriv_', 'pysr_')


def _patched_rank(df_clean, feature_cols, gamma=1.0, horizon=None):
    """Post-rank reorder by horizon-conditional category boost."""
    rank_df = _orig_rank(df_clean, feature_cols, gamma=gamma)
    h = horizon or _CURRENT_HORIZON.get('h', 6)

    if h <= 6:  # bull-bias: boost technical
        boost = TECHNICAL_PREFIXES
        nerf = MACRO_PREFIXES
    else:       # bear-bias: boost macro
        boost = MACRO_PREFIXES
        nerf = TECHNICAL_PREFIXES

    def _category_score(feat: str) -> float:
        if any(feat.startswith(p) for p in boost):
            return 1.0
        if any(feat.startswith(p) for p in nerf):
            return -0.5
        return 0.0

    rank_df = rank_df.copy()
    rank_df['_cat'] = rank_df['feature'].apply(_category_score)
    rank_df['importance'] = rank_df['importance'] * (1 + 0.3 * rank_df['_cat'])
    rank_df = rank_df.drop(columns=['_cat']).sort_values('importance', ascending=False).reset_index(drop=True)
    return rank_df


_CURRENT_HORIZON = {'h': 6}
eng._test_lgbm_importance = _patched_rank
print("[per_regime_features] _test_lgbm_importance patched")
'''
    write_patcher('per_regime_features', patcher_code)

    log_path = os.path.join(LOGS_DIR, f'idea_01_per_regime_{TS}.log')
    deltas = []
    for h in horizons:
        rc = run_mode_d(asset, h, replay, 'IDEA01', '_idea_patchers.per_regime_features',
                        log_path)
        if rc != 0:
            return {'status': 'ERROR', 'notes': f'Mode D rc={rc} on h={h}'}
        test_df, _ = load_grid_csv(asset, h, 'IDEA01')
        base_df, _ = load_baseline_grid(asset, h)
        tw, bw, delta = compare_grid_winners(test_df, base_df)
        if delta is not None:
            deltas.append((h, tw, bw, delta))

    if not deltas:
        return {'status': 'ERROR', 'notes': 'No grid winners found'}

    avg_delta = sum(d[3] for d in deltas) / len(deltas)
    status = 'PASS' if avg_delta >= 5 else ('MARGINAL' if avg_delta > 0 else 'FAIL')
    notes = '; '.join(f'h{h}: Δ{d:+.2f}' for h, tw, bw, d in deltas)
    return {'status': status, 'baseline': 'avg_apf', 'test': f'{avg_delta:+.2f}',
            'delta': f'{avg_delta:+.2f}', 'notes': notes,
            'output_csv': os.path.join(MODELS_DIR, f'crypto_ed_grid_{asset}_*h_IDEA01.csv')}


# =============================================================================
# IDEA #2 — Multi-horizon emergency-exit ensemble (5h+8h both flip SELL)
# =============================================================================

@register(2, 'multi_horizon_exit', 'filter-overlay',
          'Force exit when 5h AND 8h both flip SELL within 1h. Uses existing '
          'per-horizon signal cache. Distinct from rejected entry-side T1b.',
          'low', '~10 min')
def idea_multi_horizon_exit(asset: str, horizons: List[int], replay: int,
                             summary_path: str) -> Dict[str, Any]:
    """Audit-only: load existing 90d per-horizon signals, simulate exit overlay."""
    import pickle
    import pandas as pd

    cache_path = os.path.join(ENGINE, 'data', f'{asset.lower()}_per_horizon_signals_90d.pkl')
    if not os.path.exists(cache_path):
        return {'status': 'STUB', 'notes': f'Cache missing: {cache_path}. '
                f'Run tools/gen_per_horizon_signals.py first.'}

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    signals_5h = cache.get(5)
    signals_8h = cache.get(8)
    if signals_5h is None or signals_8h is None:
        return {'status': 'STUB', 'notes': 'Cache lacks 5h or 8h signals.'}

    df5 = pd.DataFrame(signals_5h)
    df8 = pd.DataFrame(signals_8h)
    if 'datetime' in df5.columns and 'datetime' in df8.columns:
        merged = pd.merge(df5[['datetime', 'signal']], df8[['datetime', 'signal']],
                          on='datetime', suffixes=('_5h', '_8h'))
    else:
        return {'status': 'ERROR', 'notes': 'Cache schema unexpected.'}

    n_total = len(merged)
    both_sell = ((merged['signal_5h'] == 'SELL') & (merged['signal_8h'] == 'SELL')).sum()
    pct = 100.0 * both_sell / max(n_total, 1)

    out_csv = os.path.join(OUT_DIR, f'idea_02_multi_horizon_exit_{TS}.csv')
    merged.to_csv(out_csv, index=False)

    notes = f'{both_sell}/{n_total} bars ({pct:.1f}%) had 5h+8h both SELL'
    status = 'STUB'
    return {'status': status, 'baseline': '-', 'test': f'{pct:.1f}%',
            'delta': '-', 'notes': notes + '. NEXT: simulate exit-overlay PnL '
            'vs current baseline on 60d cache.', 'output_csv': out_csv}


# =============================================================================
# IDEA #3 — Vol-scaled horizons 4-month confirmation
# =============================================================================

@register(3, 'vol_scaled_4mo', 'detector',
          'Confirm vol_2band horizon picker on 4-month replay. 2-month already '
          'showed +5.02pp; promotion-blocked only by missing 4mo confirm.',
          'low', '~30 min')
def idea_vol_scaled_4mo(asset: str, horizons: List[int], replay: int,
                         summary_path: str) -> Dict[str, Any]:
    """Delegates to existing tools/test_vol_scaled_horizon.py at --replay 2880."""
    import subprocess
    script = os.path.join(ENGINE, 'tools', 'test_vol_scaled_horizon.py')
    if not os.path.exists(script):
        return {'status': 'STUB', 'notes': f'Missing {script}'}

    log_path = os.path.join(LOGS_DIR, f'idea_03_vol_scaled_4mo_{TS}.log')
    with open(log_path, 'w', encoding='utf-8') as logf:
        rc = subprocess.run([sys.executable, script, '--replay', '2880'],
                            stdout=logf, stderr=subprocess.STDOUT, cwd=ENGINE).returncode
    if rc != 0:
        return {'status': 'ERROR', 'notes': f'rc={rc}, see {log_path}'}

    return {'status': 'STUB', 'baseline': 'tsmom_672h', 'test': 'vol_2band',
            'delta': '?', 'notes': f'Inspect log: {log_path}. Look for the '
            'top-line return delta vs tsmom_672h baseline.',
            'output_csv': log_path}


# =============================================================================
# IDEA #4 — VPIN at 5-min cadence (revival of #3 from original 20)
# =============================================================================

@register(4, 'vpin_5min', 'filter-overlay',
          'Revive VPIN filter at 5-min cadence (original test was hourly). '
          'Literature uses 1-min; sub-loop in trader is needed.',
          'medium', '~1h (needs 1m OHLCV)')
def idea_vpin_5min(asset: str, horizons: List[int], replay: int,
                    summary_path: str) -> Dict[str, Any]:
    """Requires 1m OHLCV cache. Skip if not present."""
    import pandas as pd
    cache_1m = os.path.join(ENGINE, 'data', f'{asset.lower()}_1m_data.csv')
    if not os.path.exists(cache_1m):
        return {'status': 'STUB', 'notes': f'1m cache missing: {cache_1m}. '
                f'Run tools/download_1m_data.py first to fetch ~90d of 1-min bars.'}

    return {'status': 'STUB', 'notes': '1m cache exists. NEXT: build VPIN '
            'estimator on 5-min volume buckets, audit existing 90d hourly '
            'signal cache, compute filter delta.',
            'output_csv': cache_1m}


# =============================================================================
# IDEA #5 — Stability filter with stricter threshold (revival of #5 from original 20)
# =============================================================================

@register(5, 'stability_strict', 'feature-set',
          'Revive feature stability filter with rank threshold 50 (vs 30 in '
          'first test) and 180d window (vs 60d). First test was DEAD; question '
          'is whether the threshold or the concept was wrong.',
          'low', '~20 min')
def idea_stability_strict(asset: str, horizons: List[int], replay: int,
                           summary_path: str) -> Dict[str, Any]:
    patcher_code = '''
"""Stability filter v2 patcher (stricter)."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_rank = eng._test_lgbm_importance
RANK_THRESH = 50  # only drop if max_rank - min_rank > 50 (was 30)


def _patched_rank(df_clean, feature_cols, gamma=1.0):
    """Compute ranking on 3 sub-windows; drop unstable, then full ranking."""
    n = len(df_clean)
    if n < 600:
        return _orig_rank(df_clean, feature_cols, gamma=gamma)

    splits = [
        df_clean.iloc[:n // 2],            # oldest 50%
        df_clean.iloc[n // 4 : 3 * n // 4],  # middle 50%
        df_clean.iloc[n // 2:],            # newest 50%
    ]
    sub_ranks = []
    for sub in splits:
        try:
            r = _orig_rank(sub, feature_cols, gamma=gamma)
            r = r.reset_index(drop=True)
            r['rank'] = r.index
            sub_ranks.append(dict(zip(r['feature'], r['rank'])))
        except Exception:
            pass

    if len(sub_ranks) < 2:
        return _orig_rank(df_clean, feature_cols, gamma=gamma)

    full = _orig_rank(df_clean, feature_cols, gamma=gamma)
    stable = []
    for feat in full['feature']:
        ranks = [sr.get(feat, len(feature_cols)) for sr in sub_ranks]
        if max(ranks) - min(ranks) <= RANK_THRESH:
            stable.append(feat)
    if len(stable) >= 5:
        full = full[full['feature'].isin(stable)].reset_index(drop=True)
    return full


eng._test_lgbm_importance = _patched_rank
print(f"[stability_strict] thresh={RANK_THRESH}")
'''
    write_patcher('stability_strict', patcher_code)
    log_path = os.path.join(LOGS_DIR, f'idea_05_stability_strict_{TS}.log')
    deltas = []
    for h in horizons:
        rc = run_mode_d(asset, h, replay, 'IDEA05', '_idea_patchers.stability_strict', log_path)
        if rc != 0:
            return {'status': 'ERROR', 'notes': f'rc={rc}'}
        test_df, _ = load_grid_csv(asset, h, 'IDEA05')
        base_df, _ = load_baseline_grid(asset, h)
        tw, bw, delta = compare_grid_winners(test_df, base_df)
        if delta is not None:
            deltas.append((h, tw, bw, delta))

    if not deltas:
        return {'status': 'ERROR', 'notes': 'No grid winners'}
    avg = sum(d[3] for d in deltas) / len(deltas)
    status = 'PASS' if avg >= 5 else ('MARGINAL' if avg > 0 else 'FAIL')
    notes = '; '.join(f'h{h}: Δ{d:+.2f}' for h, tw, bw, d in deltas)
    return {'status': status, 'baseline': 'avg_apf', 'test': f'{avg:+.2f}',
            'delta': f'{avg:+.2f}', 'notes': notes,
            'output_csv': os.path.join(MODELS_DIR, f'crypto_ed_grid_{asset}_*h_IDEA05.csv')}


# =============================================================================
# IDEA #6 — HAR-RV realized variance feature (Corsi 2009)
# =============================================================================

@register(6, 'har_rv', 'feature-set',
          'HAR-RV three-component realized variance forecaster as feature '
          '(1d, 5d, 22d components). Better vol estimator than 12h/48h windows.',
          'low', '~20 min')
def idea_har_rv(asset: str, horizons: List[int], replay: int,
                 summary_path: str) -> Dict[str, Any]:
    patcher_code = '''
"""HAR-RV feature injection patcher (Corsi 2009 J.Fin.Econometrics)."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_all_features


def _compute_har_rv(df: pd.DataFrame) -> dict:
    """Returns 3 columns: rv_1d, rv_5d, rv_22d (realized variance over 24, 120, 528 hourly bars)."""
    if 'close' not in df.columns:
        return {}
    logret = np.log(df['close']).diff()
    rv2 = logret ** 2
    out = {}
    for label, w in [('rv_1d', 24), ('rv_5d', 120), ('rv_22d', 528)]:
        out[f'har_{label}'] = rv2.rolling(w, min_periods=w // 2).sum()
    return out


def _patched_build(*args, **kwargs):
    res = _orig_build(*args, **kwargs)
    if isinstance(res, tuple):
        df, cols = res
    else:
        return res
    new = _compute_har_rv(df)
    for c, vals in new.items():
        df[c] = vals
        if c not in cols:
            cols.append(c)
    return df, cols


eng.build_all_features = _patched_build
print(f"[har_rv] build_all_features patched (+3 features)")
'''
    write_patcher('har_rv', patcher_code)
    log_path = os.path.join(LOGS_DIR, f'idea_06_har_rv_{TS}.log')
    deltas = []
    for h in horizons:
        rc = run_mode_d(asset, h, replay, 'IDEA06', '_idea_patchers.har_rv', log_path)
        if rc != 0:
            return {'status': 'ERROR', 'notes': f'rc={rc}'}
        test_df, _ = load_grid_csv(asset, h, 'IDEA06')
        base_df, _ = load_baseline_grid(asset, h)
        tw, bw, delta = compare_grid_winners(test_df, base_df)
        if delta is not None:
            deltas.append((h, tw, bw, delta))

    if not deltas:
        return {'status': 'ERROR', 'notes': 'No grid winners'}
    avg = sum(d[3] for d in deltas) / len(deltas)
    status = 'PASS' if avg >= 5 else ('MARGINAL' if avg > 0 else 'FAIL')
    notes = '; '.join(f'h{h}: Δ{d:+.2f}' for h, tw, bw, d in deltas)
    return {'status': status, 'baseline': 'avg_apf', 'test': f'{avg:+.2f}',
            'delta': f'{avg:+.2f}', 'notes': notes,
            'output_csv': os.path.join(MODELS_DIR, f'crypto_ed_grid_{asset}_*h_IDEA06.csv')}


# =============================================================================
# IDEA #7 — Hurst exponent feature
# =============================================================================

@register(7, 'hurst_feature', 'feature-set',
          'Rolling Hurst exponent (R/S analysis). H>0.5 trending, H<0.5 mean-reverting. '
          'Tested as ADDED FEATURE (not regime detector).',
          'low', '~20 min')
def idea_hurst_feature(asset: str, horizons: List[int], replay: int,
                        summary_path: str) -> Dict[str, Any]:
    patcher_code = '''
"""Hurst exponent feature patcher (Hurst 1951)."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_all_features


def _hurst_rs(series: np.ndarray) -> float:
    """R/S Hurst estimator. Series must be log-returns (zero-mean)."""
    n = len(series)
    if n < 50:
        return np.nan
    mean = series.mean()
    Y = series - mean
    Z = Y.cumsum()
    R = Z.max() - Z.min()
    S = series.std()
    if S == 0 or R <= 0:
        return np.nan
    return float(np.log(R / S) / np.log(n))


def _rolling_hurst(close: pd.Series, window: int = 168) -> pd.Series:
    logret = np.log(close).diff().fillna(0).values
    out = np.full(len(logret), np.nan)
    for i in range(window, len(logret)):
        out[i] = _hurst_rs(logret[i - window:i])
    return pd.Series(out, index=close.index, name='hurst_168h')


def _patched_build(*args, **kwargs):
    res = _orig_build(*args, **kwargs)
    if isinstance(res, tuple):
        df, cols = res
    else:
        return res
    if 'close' in df.columns:
        df['hurst_168h'] = _rolling_hurst(df['close'], window=168)
        if 'hurst_168h' not in cols:
            cols.append('hurst_168h')
    return df, cols


eng.build_all_features = _patched_build
print("[hurst_feature] build_all_features patched (+1 feature)")
'''
    write_patcher('hurst_feature', patcher_code)
    log_path = os.path.join(LOGS_DIR, f'idea_07_hurst_{TS}.log')
    deltas = []
    for h in horizons:
        rc = run_mode_d(asset, h, replay, 'IDEA07', '_idea_patchers.hurst_feature', log_path)
        if rc != 0:
            return {'status': 'ERROR', 'notes': f'rc={rc}'}
        test_df, _ = load_grid_csv(asset, h, 'IDEA07')
        base_df, _ = load_baseline_grid(asset, h)
        tw, bw, delta = compare_grid_winners(test_df, base_df)
        if delta is not None:
            deltas.append((h, tw, bw, delta))

    if not deltas:
        return {'status': 'ERROR', 'notes': 'No grid winners'}
    avg = sum(d[3] for d in deltas) / len(deltas)
    status = 'PASS' if avg >= 5 else ('MARGINAL' if avg > 0 else 'FAIL')
    return {'status': status, 'baseline': 'avg_apf', 'test': f'{avg:+.2f}',
            'delta': f'{avg:+.2f}',
            'notes': '; '.join(f'h{h}: Δ{d:+.2f}' for h, _, _, d in deltas),
            'output_csv': os.path.join(MODELS_DIR, f'crypto_ed_grid_{asset}_*h_IDEA07.csv')}


# =============================================================================
# IDEA #8 — CVaR objective scoring (sibling of CDaR)
# =============================================================================

@register(8, 'cvar_objective', 'scoring-change',
          'CVaR_5% on returns (worst 5% of trade returns). Penalizes tail of '
          'return distribution rather than drawdown distribution. '
          'Sibling of #6 CDaR (Rockafellar & Uryasev 2000).',
          'low', '~30 min')
def idea_cvar_objective(asset: str, horizons: List[int], replay: int,
                         summary_path: str) -> Dict[str, Any]:
    """Post-hoc rerank existing grid CSVs by CVaR-adjusted score.
    Requires per-trade return list — not always in current grid CSV.
    """
    return {'status': 'STUB',
            'notes': 'Requires per-trade-return list per config in grid CSV. '
            'Engine writes summary stats only. NEXT: extend grid CSV writer to '
            'persist per-trade returns, then post-hoc rank by ret − λ × CVaR_5%.'}


# =============================================================================
# IDEA #9 — Sortino ratio objective
# =============================================================================

@register(9, 'sortino_objective', 'scoring-change',
          'Sortino ratio (only downside vol in denominator) instead of '
          'return × WR. Aligns with asymmetric loss aversion.',
          'low', '~30 min')
def idea_sortino_objective(asset: str, horizons: List[int], replay: int,
                            summary_path: str) -> Dict[str, Any]:
    """Same dependency on per-trade returns as CVaR."""
    return {'status': 'STUB',
            'notes': 'Same dependency as #8: needs per-trade return list. '
            'Could be co-implemented with CVaR (#8) — both need same plumbing.'}


# =============================================================================
# IDEA #10 — SHAP-based feature ranking
# =============================================================================

@register(10, 'shap_ranking', 'feature-set',
          'Replace LGBM importance with SHAP value ranking. SHAP captures '
          'interaction effects that raw split-count importance misses.',
          'medium', '~30 min (needs shap install)')
def idea_shap_ranking(asset: str, horizons: List[int], replay: int,
                       summary_path: str) -> Dict[str, Any]:
    """Check if shap installed; if so, monkey-patch _test_lgbm_importance."""
    try:
        import shap  # noqa
    except ImportError:
        return {'status': 'STUB', 'notes': 'pip install shap; then rerun this idea.'}

    patcher_code = '''
"""SHAP ranking patcher."""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

_orig_rank = eng._test_lgbm_importance


def _patched_rank(df_clean, feature_cols, gamma=1.0):
    X = df_clean[feature_cols].values
    y = df_clean['label'].values
    if len(np.unique(y)) < 2:
        return _orig_rank(df_clean, feature_cols, gamma=gamma)
    model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                random_state=42, verbose=-1)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X[-min(500, len(X)):])
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    importance = np.abs(sv).mean(axis=0)
    return pd.DataFrame({'feature': feature_cols, 'importance': importance}).sort_values(
        'importance', ascending=False).reset_index(drop=True)


eng._test_lgbm_importance = _patched_rank
print("[shap_ranking] _test_lgbm_importance replaced with SHAP")
'''
    write_patcher('shap_ranking', patcher_code)
    log_path = os.path.join(LOGS_DIR, f'idea_10_shap_{TS}.log')
    deltas = []
    for h in horizons:
        rc = run_mode_d(asset, h, replay, 'IDEA10', '_idea_patchers.shap_ranking', log_path)
        if rc != 0:
            return {'status': 'ERROR', 'notes': f'rc={rc}'}
        test_df, _ = load_grid_csv(asset, h, 'IDEA10')
        base_df, _ = load_baseline_grid(asset, h)
        tw, bw, delta = compare_grid_winners(test_df, base_df)
        if delta is not None:
            deltas.append((h, tw, bw, delta))

    if not deltas:
        return {'status': 'ERROR', 'notes': 'No grid winners'}
    avg = sum(d[3] for d in deltas) / len(deltas)
    status = 'PASS' if avg >= 5 else ('MARGINAL' if avg > 0 else 'FAIL')
    return {'status': status, 'baseline': 'avg_apf', 'test': f'{avg:+.2f}',
            'delta': f'{avg:+.2f}',
            'notes': '; '.join(f'h{h}: Δ{d:+.2f}' for h, _, _, d in deltas),
            'output_csv': os.path.join(MODELS_DIR, f'crypto_ed_grid_{asset}_*h_IDEA10.csv')}


# =============================================================================
# IDEA #11 — CUSUM event-based sampling
# =============================================================================

@register(11, 'cusum_sampling', 'architectural',
          'CUSUM filter: trade only when cumulative log-return crosses threshold. '
          'Replaces hourly polling with event-driven sampling (LdP AFML Ch 2).',
          'high', 'architectural — STUB only')
def idea_cusum_sampling(asset: str, horizons: List[int], replay: int,
                          summary_path: str) -> Dict[str, Any]:
    """Architectural; would change trader's polling cadence."""
    return {'status': 'STUB',
            'notes': 'Architectural change: trader polls hourly, would need to '
            'switch to event-trigger. Not a Mode D smoke test. Build a separate '
            'CUSUM-replay simulator that reads the 90d signal cache and fires '
            'BUY/SELL only when |Σ logret since last event| > h_threshold. '
            'Compare CUSUM-PnL to baseline-hourly-PnL.'}


# =============================================================================
# IDEA #12 — Realized-vol entry filter
# =============================================================================

@register(12, 'vol_entry_filter', 'filter-overlay',
          'Skip BUYs when realized vol > 90th percentile (vs 30d window). '
          'Tail-vol periods often have wider stops + worse fills.',
          'low', '~5 min')
def idea_vol_entry_filter(asset: str, horizons: List[int], replay: int,
                            summary_path: str) -> Dict[str, Any]:
    """Audit existing 90d signal cache; simulate skip-on-high-vol overlay."""
    import pickle
    import numpy as np
    import pandas as pd

    cache_path = os.path.join(ENGINE, 'data', f'{asset.lower()}_sl_signals_90d.pkl')
    if not os.path.exists(cache_path):
        return {'status': 'STUB', 'notes': f'Cache missing: {cache_path}'}

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    sigs_data = cache.get('signals', cache) if isinstance(cache, dict) else cache
    sigs = pd.DataFrame(sigs_data)
    if 'datetime' not in sigs.columns or 'close' not in sigs.columns:
        return {'status': 'STUB', 'notes': 'Cache schema unexpected.'}

    sigs['logret'] = np.log(sigs['close']).diff()
    sigs['rv_24h'] = sigs['logret'].rolling(24).std()
    sigs['rv_pct90_30d'] = sigs['rv_24h'].rolling(30 * 24).quantile(0.9)
    sigs['high_vol'] = sigs['rv_24h'] > sigs['rv_pct90_30d']

    n_total_buys = (sigs['signal'] == 'BUY').sum()
    n_skipped = ((sigs['signal'] == 'BUY') & sigs['high_vol']).sum()

    out_csv = os.path.join(OUT_DIR, f'idea_12_vol_filter_{TS}.csv')
    sigs.to_csv(out_csv, index=False)

    notes = f'{n_skipped}/{n_total_buys} BUYs would be skipped (high_vol > p90)'
    return {'status': 'STUB', 'baseline': '-', 'test': f'{n_skipped} skips',
            'delta': '-', 'notes': notes + '. NEXT: simulate strategy with '
            'skipped BUYs vs baseline; compute return delta.',
            'output_csv': out_csv}


# =============================================================================
# IDEA #13 — Cross-sectional BTC momentum gate
# =============================================================================

@register(13, 'btc_momentum_gate', 'filter-overlay',
          'Allow ETH BUY only when BTC is in a positive 24h momentum regime. '
          'Cross-sectional confirmation: if BTC is dumping, skip ETH BUYs.',
          'low', '~5 min')
def idea_btc_momentum_gate(asset: str, horizons: List[int], replay: int,
                             summary_path: str) -> Dict[str, Any]:
    """Audit existing signal cache; check BTC 24h logret on each ETH BUY."""
    import pickle
    import numpy as np
    import pandas as pd

    eth_cache = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
    btc_csv = os.path.join(ENGINE, 'data', 'btc_hourly_data.csv')
    if not os.path.exists(eth_cache) or not os.path.exists(btc_csv):
        return {'status': 'STUB', 'notes': 'Need eth signal cache + btc OHLCV CSV.'}

    with open(eth_cache, 'rb') as f:
        cache = pickle.load(f)
    sigs_data = cache.get('signals', cache) if isinstance(cache, dict) else cache
    sigs = pd.DataFrame(sigs_data)
    btc = pd.read_csv(btc_csv)

    if 'datetime' not in sigs.columns or 'datetime' not in btc.columns:
        return {'status': 'STUB', 'notes': 'Schema mismatch.'}

    sigs['datetime'] = pd.to_datetime(sigs['datetime'])
    btc['datetime'] = pd.to_datetime(btc['datetime'])
    btc['btc_logret_24h'] = np.log(btc['close']).diff(24)
    merged = pd.merge(sigs, btc[['datetime', 'btc_logret_24h']],
                      on='datetime', how='left')

    n_total_buys = (merged['signal'] == 'BUY').sum()
    n_btc_negative = ((merged['signal'] == 'BUY') & (merged['btc_logret_24h'] < 0)).sum()

    out_csv = os.path.join(OUT_DIR, f'idea_13_btc_momentum_gate_{TS}.csv')
    merged.to_csv(out_csv, index=False)

    notes = f'{n_btc_negative}/{n_total_buys} BUYs had negative BTC 24h momentum'
    return {'status': 'STUB', 'baseline': '-', 'test': f'{n_btc_negative} skips',
            'delta': '-', 'notes': notes + '. NEXT: simulate strategy skipping '
            'these BUYs vs baseline; compute return delta.',
            'output_csv': out_csv}


# =============================================================================
# IDEA #14 — Variance Ratio test as regime detector (Lo & MacKinlay 1988)
# =============================================================================

@register(14, 'variance_ratio_detector', 'detector',
          'VR(q) = Var(q-period return) / [q × Var(1-period return)]. VR>1 trending, '
          'VR<1 mean-reverting. Use as alternative regime detector.',
          'medium', '~10 min')
def idea_variance_ratio_detector(asset: str, horizons: List[int], replay: int,
                                   summary_path: str) -> Dict[str, Any]:
    """Compute VR(q=12) on close prices; output as a regime label time-series."""
    import pickle
    import numpy as np
    import pandas as pd

    csv_path = os.path.join(ENGINE, 'data', f'{asset.lower()}_hourly_data.csv')
    if not os.path.exists(csv_path):
        return {'status': 'STUB', 'notes': f'Missing {csv_path}'}

    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    logret = np.log(df['close']).diff()

    Q = 12  # 12-hour aggregation
    win = 240  # 10-day rolling
    var1 = logret.rolling(win).var()
    retq = logret.rolling(Q).sum()
    varq = retq.rolling(win).var()
    vr = varq / (Q * var1)
    df['vr_12'] = vr
    df['regime_vr'] = np.where(df['vr_12'] > 1.0, 'bull', 'bear')

    n_bull = (df['regime_vr'] == 'bull').sum()
    n_bear = (df['regime_vr'] == 'bear').sum()
    pct_bull = 100.0 * n_bull / max(len(df), 1)

    out_csv = os.path.join(OUT_DIR, f'idea_14_variance_ratio_{TS}.csv')
    df[['datetime', 'close', 'vr_12', 'regime_vr']].to_csv(out_csv, index=False)

    notes = f'{n_bull}/{n_bull+n_bear} bull bars ({pct_bull:.1f}%)'
    return {'status': 'STUB', 'baseline': '-', 'test': f'{pct_bull:.1f}% bull',
            'delta': '-', 'notes': notes + '. NEXT: register `vr_12` as named '
            'detector in `_build_regime_indicators_and_detectors`, run Mode S '
            'with it added, compare alpha to current `tsmom_672h` baseline.',
            'output_csv': out_csv}


# =============================================================================
# Main runner
# =============================================================================

def run_one(idea: Dict[str, Any], asset: str, horizons: List[int], replay: int,
            summary_path: str) -> Dict[str, Any]:
    print('=' * 90)
    print(f"#{idea['num']:02d} {idea['name']:<30} [{idea['category']}] — {idea['summary']}")
    print(f"  effort={idea['effort']} est_runtime={idea['est_runtime']}")
    print('=' * 90)
    t0 = time.time()
    try:
        result = idea['fn'](asset, horizons, replay, summary_path)
    except Exception as e:
        result = {'status': 'ERROR', 'notes': f'{type(e).__name__}: {e}'}
    elapsed = time.time() - t0
    result['elapsed_min'] = f'{elapsed/60:.1f}'

    write_summary_row(
        idea['num'], idea['name'], result.get('status', '?'),
        result.get('baseline', '-'), result.get('test', '-'),
        result.get('delta', '-'),
        f"({elapsed/60:.1f}min) {result.get('notes', '')}",
        summary_path,
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Unified harness for 14 candidate ideas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--list', action='store_true', help='List all 14 ideas and exit')
    parser.add_argument('--idea', help='Run single idea (number or name) or comma-list')
    parser.add_argument('--all', action='store_true', help='Run all 14 sequentially')
    parser.add_argument('--asset', default='ETH')
    parser.add_argument('--horizons', default='5,6,7,8')
    parser.add_argument('--replay', type=int, default=1440)
    parser.add_argument('--quick', action='store_true',
                        help='Use --replay 720 for faster smoke')
    args = parser.parse_args()

    if args.list:
        print('=' * 100)
        print(f"  {'#':>3}  {'name':<30}  {'category':<18}  {'effort':<8}  {'est_runtime'}")
        print('=' * 100)
        for idea in IDEAS:
            print(f"  {idea['num']:>3}  {idea['name']:<30}  {idea['category']:<18}  "
                  f"{idea['effort']:<8}  {idea['est_runtime']}")
            print(f"        {idea['summary']}")
        print('=' * 100)
        return 0

    horizons = [int(h.strip()) for h in args.horizons.split(',') if h.strip()]
    replay = 720 if args.quick else args.replay

    if args.all:
        targets = list(IDEAS)
    elif args.idea:
        targets = []
        for sel in args.idea.split(','):
            idea = find_idea(sel)
            if idea is None:
                print(f"ERROR: idea '{sel}' not found. Use --list to see all 14.")
                return 1
            targets.append(idea)
    else:
        parser.print_help()
        return 1

    summary_path = os.path.join(LOGS_DIR, f'test_14_summary_{TS}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Harness run {TS}, asset={args.asset}, horizons={horizons}, replay={replay}\n")
        f.write(f"Targets: {[i['name'] for i in targets]}\n")
        f.write('=' * 100 + '\n')

    results = []
    for idea in targets:
        r = run_one(idea, args.asset, horizons, replay, summary_path)
        results.append((idea, r))

    print('\n' + '=' * 90)
    print('  FINAL VERDICTS')
    print('=' * 90)
    for idea, r in results:
        print(f"  #{idea['num']:02d} {idea['name']:<30}  {r.get('status', '?'):<10} "
              f"{r.get('notes', '')[:80]}")
    print(f"\nSummary file: {summary_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
