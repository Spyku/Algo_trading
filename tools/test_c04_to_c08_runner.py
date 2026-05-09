"""test_c04_to_c08_runner.py — Sequential Desktop runner for C04, C05, C06, C07, C08.

Run on Desktop with venv active:
  python tools/test_c04_to_c08_runner.py

Total runtime estimate: ~5-6h sequential. C07 is a no-op (architectural — not a
Mode D smoke test; documented in CLAUDE.md). C05 and C06 share a single backtest
pass (different metrics computed off the same per-trade-return series).

Per-idea scope and method:
  C04 Variance Ratio detector (Lo & MacKinlay 1988) — adds VR(q=12,24,48) as 3
      features via build_all_features patch. Mode D × 4 horizons (5,6,7,8).
      Tags grid CSVs with _C04VR. ~30 min.
  C05 CVaR_5% objective — post-hoc rerank: for each horizon, take top-15 grid
      candidates from clean untagged baseline, walk-forward backtest WITH
      per-trade-return capture, compute CVaR_5% on per-trade returns, rerank
      under λ × CVaR. ~1.5h.
  C06 Sortino objective — same backtest pass as C05; metric = mean / downside_std.
      Reranks under Sortino. Marginal cost ~0 (shares C05 backtest data).
  C07 CUSUM event-based sampling (LdP AFML Ch 2) — SKIP. Architectural rewrite
      of trader polling loop; canonical idea is event-driven re-sampling, not
      a Mode D feature-add. CLAUDE.md flags it as 'days' effort.
  C08 Single-horizon CDaR — subprocess `crypto_trading_system_ed_cdar.py V` per
      horizon (5,6,7,8) with --no-persist. Compares CDaR Mode V winner to
      current APF Mode V winner from production CSV. Tests whether
      single-horizon CDaR scoring beats APF when Mode S regime split is
      bypassed. ~2h (4 × ~30 min).

Prerequisites:
  - clean untagged baseline grids at models/crypto_ed_grid_ETH_<h>h.csv
    (refreshed by HRST today 14:24 → APF-Optuna clean baselines)
  - crypto_trading_system_ed_cdar.py exists at repo root (engine fork)
  - statsmodels installed in venv (transitive — used by other ideas)

Decision rules per idea:
  C04: avg APF Δ ≥ +5pp vs untagged baseline → PASS (escalate to HRST)
       0 < avg < +5  → MARGINAL
       avg ≤ 0       → FAIL
  C05/C06: rerank picks materially different top-3 vs APF rerank with comparable
       returns → SHIP-CANDIDATE (1-line scoring change)
       Same top-3 → null (return + tail risk highly correlated, scoring change cosmetic)
       Lower returns across all λ → FAIL
  C08: any single-horizon CDaR Mode V winner beats production APF winner by
       ≥ +5pp on net return AND ≥ +1pp on max_dd reduction → SHIP for that horizon
       (per-horizon ship is OK; not gated on Mode S regime sweep)

Outputs:
  tools/_idea_patchers/c04_variance_ratio.py — generated patcher
  models/crypto_ed_grid_ETH_<h>h_C04VR.csv   — C04 tagged grids
  logs/c04_variance_ratio_<ts>.log           — C04 per-horizon log
  output/c05_c06_per_trade_<asset>_<h>h_<ts>.csv — per-trade-return audit data
  output/c05_c06_rerank_<ts>.csv             — joint rerank table (CVaR + Sortino)
  logs/c08_cdar_V_<h>h_<ts>.log              — C08 cdar engine Mode V logs
  logs/c04_to_c08_summary_<ts>.txt           — final summary across all ideas
"""
from __future__ import annotations

import os
import sys
import subprocess
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
PYTHON = sys.executable
OUT_DIR = os.path.join(ENGINE, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

SUMMARY_PATH = os.path.join(LOGS_DIR, f'c04_to_c08_summary_{TS}.txt')


def _summary_write(line: str):
    print(line)
    with open(SUMMARY_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# =============================================================================
# C04 — Variance Ratio detector as 3 features (q=12,24,48)
# =============================================================================
C04_PATCHER = '''
"""C04 Variance Ratio (Lo & MacKinlay 1988) at q=12,24,48 as 3 feature columns.

VR(q) = Var(q-period sum of log-returns) / [q * Var(1-period log-return)]
  VR > 1  -> trending (positive autocorrelation)
  VR < 1  -> mean-reverting
  VR = 1  -> random walk

Computed on rolling 252h window (~10 days) for stable variance estimate.
"""
import crypto_trading_system_ed as eng
import numpy as np
import pandas as pd

_orig_build = eng.build_all_features


def _vr_features(df: pd.DataFrame) -> dict:
    if 'close' not in df.columns:
        return {}
    logret = np.log(df['close'].astype(float)).diff().fillna(0).values
    n = len(logret)
    out = {}
    rolling_window = 252  # 10.5d at hourly cadence
    for q in [12, 24, 48]:
        vr = np.full(n, np.nan)
        # need at least rolling_window samples + q for the q-period aggregate
        min_i = rolling_window + q
        for i in range(min_i, n):
            seg = logret[i - rolling_window:i]
            var_1 = seg.var(ddof=1)
            if var_1 == 0 or not np.isfinite(var_1):
                continue
            # Non-overlapping q-period aggregates
            n_q = len(seg) // q
            if n_q < 2:
                continue
            q_rets = seg[:n_q * q].reshape(n_q, q).sum(axis=1)
            var_q = q_rets.var(ddof=1)
            if not np.isfinite(var_q):
                continue
            vr[i] = var_q / (q * var_1)
        out[f'vr_{q}'] = vr
    return out


def _patched_build(*args, **kwargs):
    result = _orig_build(*args, **kwargs)
    # build_all_features can return tuple of 2 or 3 — defensive unpack
    if isinstance(result, tuple) and len(result) == 3:
        df, all_cols, lead_cols = result
    elif isinstance(result, tuple) and len(result) == 2:
        df, all_cols = result
        lead_cols = None
    else:
        df = result
        all_cols, lead_cols = None, None
    vr = _vr_features(df)
    added = 0
    for name, vals in vr.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            added += 1
    if added:
        print(f'[C04] VR features added: +{added} columns')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    elif all_cols is not None:
        return df, all_cols
    return df


eng.build_all_features = _patched_build
print('[C04] build_all_features patched (+VR features at q=12,24,48)')
'''


def run_c04() -> dict:
    """Run C04 Variance Ratio Mode D × 4 horizons. Returns verdict dict."""
    _summary_write('=' * 100)
    _summary_write(f'C04 — Variance Ratio detector (Lo & MacKinlay 1988)')
    _summary_write('=' * 100)
    write_patcher('c04_variance_ratio', C04_PATCHER)
    log_path = os.path.join(LOGS_DIR, f'c04_variance_ratio_{TS}.log')
    deltas = []
    for h in HORIZONS:
        _summary_write(f'  >> C04 Mode D ETH {h}h (replay={REPLAY}h) ...')
        rc = run_mode_d(ASSET, h, REPLAY, 'C04VR',
                        '_idea_patchers.c04_variance_ratio', log_path)
        if rc != 0:
            _summary_write(f'     ERROR rc={rc}')
            continue
        test_df, _ = load_grid_csv(ASSET, h, 'C04VR')
        base_df, _ = load_baseline_grid(ASSET, h)
        if test_df is None or base_df is None:
            _summary_write(f'     ERROR loading grid CSVs')
            continue
        tw, bw, delta = compare_grid_winners(test_df, base_df, sort_col='apf')
        if delta is None:
            _summary_write(f'     ERROR comparison failed')
            continue
        deltas.append((h, tw, bw, delta))
        _summary_write(f'     {h}h: test_apf={tw:.3f}  base_apf={bw:.3f}  delta={delta:+.3f}')

    if not deltas:
        verdict, avg = 'ERROR', float('nan')
    else:
        avg = sum(d[3] for d in deltas) / len(deltas)
        verdict = 'PASS' if avg >= 5 else 'MARGINAL' if avg > 0 else 'FAIL'
    _summary_write(f'  C04 verdict: avg APF delta = {avg:+.3f} -> {verdict}')
    _summary_write('')
    return {'name': 'C04', 'verdict': verdict, 'avg_delta': avg, 'per_horizon': deltas}


# =============================================================================
# C05 + C06 — CVaR / Sortino post-hoc rerank (shared backtest pass)
# =============================================================================
def _backtest_with_per_trade_capture(asset: str, horizon: int, combo: list,
                                      window: int, gamma: float, n_features: int):
    """Walk-forward backtest with per-trade-return capture.

    Returns dict: {return_pct, max_dd_pct, n_trades, trade_returns: list[float]}.
    Adapted from test_cdar_rescore.refit_and_compute_dd but additionally records
    the per-trade close-to-close return list (post-fee) for CVaR/Sortino math.

    2026-05-08 fix: forces CPU LGBM for the duration of the call. Previous
    OSError(22) "device doesn't exist" likely came from GPU contention with
    a concurrent HRST run. Walk-forward per-trade backtest is small enough
    that CPU LGBM is fast enough; safer than racing for the GPU.
    """
    import numpy as np
    import crypto_trading_system_ed as eng
    from crypto_trading_system_ed import (load_data, _build_features,
                                          _test_lgbm_importance,
                                          get_decay_weights, DIAG_STEP,
                                          BACKTEST_FEE_PER_LEG, TRADING_FEE,
                                          _feature_floor_indices)
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression

    # Local CPU-only model registry — bypasses the engine's GPU LGBM defaults.
    CPU_MODELS = {
        'RF':   lambda: RandomForestClassifier(n_estimators=300, max_depth=6,
                                                class_weight='balanced', random_state=42, n_jobs=1),
        'XGB':  lambda: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                       random_state=42, tree_method='hist', verbosity=0, n_jobs=1),
        'LR':   lambda: LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'LGBM': lambda: LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                        class_weight='balanced', verbose=-1, random_state=42, device='cpu'),
    }
    _orig_device = eng.LGBM_DEVICE
    eng.LGBM_DEVICE = 'cpu'  # force CPU for _test_lgbm_importance call below
    ALL_MODELS = CPU_MODELS

    df_raw = load_data(asset)
    if df_raw is None:
        return None
    df_features, feature_cols = _build_features(df_raw, asset,
                                                feature_override=None,
                                                horizon=horizon)
    df_clean = df_features.dropna(subset=feature_cols + ['label']).reset_index(drop=True)
    importance_df = _test_lgbm_importance(df_clean, feature_cols, gamma=1.0)
    ranked = importance_df['feature'].tolist()
    df_op = df_clean.dropna(subset=ranked + ['label']).reset_index(drop=True)
    sel_idx = _feature_floor_indices(ranked, n_features)
    feat_np = df_op[ranked].values.astype(np.float64)[:, sel_idx]
    labels_np = df_op['label'].values.astype(np.int32)
    closes_np = df_op['close'].values.astype(np.float64)
    n = len(df_op)
    train_n = int(0.6 * n)
    min_start = max(window + 50, train_n)

    portfolio = 1.0
    in_pos = False
    entry_px = 0.0
    trade_returns = []  # one entry per closed trade (post-fee fractional return)
    peak = 1.0
    max_dd = 0.0

    for i in range(min_start, n, DIAG_STEP):
        train_start = max(0, i - window)
        train_end = max(train_start, i - horizon)
        X_train = feat_np[train_start:train_end]
        y_train = labels_np[train_start:train_end]
        X_test = feat_np[i:i + 1]
        if len(np.unique(y_train)) < 2 or np.isnan(X_train).any() or np.isnan(X_test).any():
            continue
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0); std[std == 0] = 1.0
        X_tr = (X_train - mean) / std
        X_te = (X_test - mean) / std
        sw = get_decay_weights(len(y_train), gamma)
        votes = []
        for mn in combo:
            try:
                m = ALL_MODELS[mn]()
                m.fit(X_tr, y_train, sample_weight=sw)
                votes.append(m.predict(X_te)[0])
            except Exception:
                continue
        if not votes:
            continue
        pred = 1 if sum(votes) / len(votes) > 0.5 else 0
        price = closes_np[i]
        if pred == 1 and not in_pos:
            in_pos = True
            entry_px = price * (1 + TRADING_FEE)
        elif pred == 0 and in_pos:
            sell_px = price * (1 - BACKTEST_FEE_PER_LEG)
            r = (sell_px - entry_px) / entry_px
            trade_returns.append(r)
            portfolio *= (1 + r)
            in_pos = False
        cur = portfolio * (price / entry_px) if in_pos else portfolio
        if cur > peak:
            peak = cur
        dd = (peak - cur) / peak
        if dd > max_dd:
            max_dd = dd

    # Close any open position at the last bar
    if in_pos:
        last_px = closes_np[n - 1]
        sell_px = last_px * (1 - BACKTEST_FEE_PER_LEG)
        r = (sell_px - entry_px) / entry_px
        trade_returns.append(r)
        portfolio *= (1 + r)

    # Restore engine's LGBM_DEVICE so any subsequent caller (other tools, trader)
    # sees the original setting rather than the CPU override we set above.
    eng.LGBM_DEVICE = _orig_device

    return {
        'return_pct': (portfolio - 1.0) * 100,
        'max_dd_pct': max_dd * 100,
        'n_trades': len(trade_returns),
        'trade_returns': trade_returns,
    }


def _cvar_5pct(trade_returns: list) -> float:
    """CVaR_5% on per-trade returns: mean of worst 5% trades. Returns % (negative)."""
    import numpy as np
    if len(trade_returns) < 4:
        return float('nan')
    arr = np.array(trade_returns) * 100  # to %
    thresh = np.quantile(arr, 0.05)
    worst = arr[arr <= thresh]
    return float(worst.mean()) if len(worst) > 0 else float('nan')


def _sortino(trade_returns: list) -> float:
    """Sortino ratio: mean / downside_std. Returns ratio (unitless)."""
    import numpy as np
    if len(trade_returns) < 4:
        return float('nan')
    arr = np.array(trade_returns)
    downside = arr[arr < 0]
    if len(downside) == 0:
        return float('inf') if arr.mean() > 0 else 0.0
    ds_std = downside.std(ddof=1)
    if ds_std == 0:
        return float('inf') if arr.mean() > 0 else 0.0
    return float(arr.mean() / ds_std)


def run_c05_c06() -> dict:
    """Joint C05/C06 — shared backtest, two reranks."""
    import pandas as pd
    import traceback
    _summary_write('=' * 100)
    _summary_write('C05 + C06 — CVaR_5% + Sortino post-hoc rerank (shared backtest)')
    _summary_write('=' * 100)
    TOP_N = 15
    audit_rows = []
    rerank_rows = []
    for h in HORIZONS:
        _summary_write(f'  >> C05/C06 horizon {h}h: load top-{TOP_N} from baseline grid')
        base_df, base_path = load_baseline_grid(ASSET, h)
        if base_df is None:
            _summary_write(f'     SKIP {h}h: no baseline grid CSV')
            continue
        cand = base_df.nlargest(TOP_N, 'apf').reset_index(drop=True)
        for idx, row in cand.iterrows():
            # Baseline grid CSV has column 'combo' (e.g. "XGB+LGBM"), not 'models'.
            combo = row['combo'].split('+')
            window = int(row['window'])
            gamma = float(row['gamma'])
            n_feat = int(row['n_features'])
            _summary_write(f'     [{h}h #{idx+1:02d}/{TOP_N}] {"+".join(combo):<10} '
                           f'w={window} g={gamma:.3f} f={n_feat} ...')
            try:
                r = _backtest_with_per_trade_capture(ASSET, h, combo, window,
                                                      gamma, n_feat)
            except Exception as exc:
                _summary_write(f'        EXCEPTION on this candidate: {exc!r}')
                _summary_write(f'        Traceback:\n{traceback.format_exc()}')
                continue
            if r is None or r['n_trades'] == 0:
                _summary_write(f'        skipped (no data or 0 trades)')
                continue
            cvar = _cvar_5pct(r['trade_returns'])
            sortino = _sortino(r['trade_returns'])
            apf = float(row['apf'])
            rerank_rows.append({
                'horizon': h, 'rank_apf': idx + 1,
                'combo': '+'.join(combo), 'window': window,
                'gamma': gamma, 'n_features': n_feat,
                'apf': apf,
                'return_pct': r['return_pct'],
                'max_dd_pct': r['max_dd_pct'],
                'n_trades': r['n_trades'],
                'cvar_5pct': cvar,
                'sortino': sortino,
                'cvar_score_lam_1': r['return_pct'] - 1.0 * abs(cvar),
                'cvar_score_lam_2': r['return_pct'] - 2.0 * abs(cvar),
                'sortino_score': sortino,
            })
            for tr in r['trade_returns']:
                audit_rows.append({'horizon': h, 'rank_apf': idx + 1,
                                   'combo': '+'.join(combo),
                                   'trade_return_frac': tr})
    if not rerank_rows:
        _summary_write('  C05/C06 ERROR: no candidates produced trades')
        return {'name': 'C05_C06', 'verdict': 'ERROR'}

    rerank_df = pd.DataFrame(rerank_rows)
    audit_df = pd.DataFrame(audit_rows)
    rerank_path = os.path.join(OUT_DIR, f'c05_c06_rerank_{TS}.csv')
    audit_path = os.path.join(OUT_DIR, f'c05_c06_per_trade_{TS}.csv')
    rerank_df.to_csv(rerank_path, index=False)
    audit_df.to_csv(audit_path, index=False)
    _summary_write(f'  Wrote rerank table: {rerank_path}')
    _summary_write(f'  Wrote per-trade audit: {audit_path}')

    # Compare top-3 across scoring schemes per horizon
    _summary_write('')
    schemes = ['apf', 'cvar_score_lam_1', 'cvar_score_lam_2', 'sortino_score']
    for h in HORIZONS:
        sub = rerank_df[rerank_df['horizon'] == h]
        if len(sub) == 0:
            continue
        _summary_write(f'  --- horizon {h}h ---')
        for s in schemes:
            top3 = sub.nlargest(3, s)[['rank_apf', 'combo', 'return_pct',
                                        'max_dd_pct', 'cvar_5pct', 'sortino']]
            _summary_write(f'    {s:>20s} top-3:')
            for _, r in top3.iterrows():
                _summary_write(f'      apf_rank={r["rank_apf"]:<3d} {r["combo"]:<10s} '
                               f'ret={r["return_pct"]:+.2f}%  dd={r["max_dd_pct"]:.2f}%  '
                               f'cvar5={r["cvar_5pct"]:.2f}%  sortino={r["sortino"]:.3f}')
    _summary_write('')
    _summary_write('  C05/C06 decision (apply manually): does any rerank scheme pick a')
    _summary_write('  materially different top-3 with similar returns AND lower tail-risk?')
    _summary_write('  If YES across multiple horizons -> ship-candidate (1-line scoring change).')
    _summary_write('  If reranks identical to APF -> null (return + tail correlated).')
    _summary_write('')
    return {'name': 'C05_C06', 'verdict': 'PENDING_MANUAL_REVIEW',
            'rerank_path': rerank_path, 'audit_path': audit_path}


# =============================================================================
# C07 — CUSUM event-based sampling: SKIP
# =============================================================================
def run_c07() -> dict:
    _summary_write('=' * 100)
    _summary_write('C07 — CUSUM event-based sampling — SKIPPED')
    _summary_write('=' * 100)
    _summary_write('  Per CLAUDE.md canonical scoreboard: "architectural rewrite of trader')
    _summary_write('  polling loop. Not a smoke test."')
    _summary_write('  Estimated effort: days. Cannot be validated via Mode D Δ-APF.')
    _summary_write('  To revisit: write a separate trader-side prototype, not here.')
    _summary_write('')
    return {'name': 'C07', 'verdict': 'SKIPPED', 'reason': 'architectural'}


# =============================================================================
# C08 — Single-horizon CDaR (no Mode S regime split)
# =============================================================================
def run_c08() -> dict:
    """Run cdar engine Mode V per horizon, compare to APF Mode V production."""
    _summary_write('=' * 100)
    _summary_write('C08 — Single-horizon CDaR Mode V (no Mode S split)')
    _summary_write('=' * 100)
    cdar_engine = os.path.join(ENGINE, 'crypto_trading_system_ed_cdar.py')
    if not os.path.exists(cdar_engine):
        _summary_write(f'  ERROR: cdar engine not found at {cdar_engine}')
        return {'name': 'C08', 'verdict': 'ERROR', 'reason': 'engine_missing'}

    rows = []
    for h in HORIZONS:
        log_path = os.path.join(LOGS_DIR, f'c08_cdar_V_{h}h_{TS}.log')
        cmd = [PYTHON, cdar_engine, 'V', ASSET, f'{h}h',
               '--replay', str(REPLAY), '--no-persist', '--no-data-update']
        _summary_write(f'  >> C08 cdar V ETH {h}h ...')
        _summary_write(f'     cmd: {" ".join(cmd)}')
        _summary_write(f'     log: {log_path}')
        with open(log_path, 'w', encoding='utf-8') as lf:
            try:
                rc = subprocess.run(cmd, cwd=ENGINE, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=2 * 3600).returncode
            except subprocess.TimeoutExpired:
                _summary_write(f'     TIMEOUT after 2h')
                continue
        if rc != 0:
            _summary_write(f'     rc={rc} (non-zero)')
            continue
        # Parse the log for the Mode V winner line + return + max_dd
        winner = _parse_mode_v_winner(log_path)
        if winner:
            _summary_write(f'     CDaR V winner: {winner.get("source", "?")}  '
                           f'ret={winner.get("return_pct", float("nan")):+.2f}%  '
                           f'WR={winner.get("win_rate", float("nan")):.0f}%  '
                           f'trades={winner.get("trades", 0)}')
            winner['horizon'] = h
            rows.append(winner)
        else:
            _summary_write(f'     could not parse Mode V winner from log')
    if not rows:
        _summary_write('  C08: no parsable winners. Read logs manually.')
        return {'name': 'C08', 'verdict': 'ERROR', 'reason': 'no_winners'}
    _summary_write('')
    _summary_write('  C08 cross-horizon summary (CDaR Mode V picks):')
    for r in rows:
        _summary_write(f'    {r["horizon"]}h: ret={r.get("return_pct", float("nan")):+.2f}%  '
                       f'WR={r.get("win_rate", float("nan")):.0f}%  '
                       f'trades={r.get("trades", 0)}  source={r.get("source", "?")}')
    _summary_write('')
    _summary_write('  Compare each horizon return to current production APF Mode V winner.')
    _summary_write('  Per-horizon ship if CDaR ≥ APF + 5pp on return AND ≥ +1pp on DD reduction.')
    _summary_write('')
    return {'name': 'C08', 'verdict': 'PENDING_MANUAL_REVIEW', 'per_horizon': rows}


def _parse_mode_v_winner(log_path: str) -> dict:
    """Best-effort scrape of cdar engine Mode V winner from its stdout log."""
    import re
    if not os.path.exists(log_path):
        return {}
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    out = {}
    # Common engine print patterns; tolerate whitespace and variants.
    m = re.search(r'WINNER.*?:\s*(.+?)\n', text, flags=re.IGNORECASE)
    if m:
        out['source'] = m.group(1).strip()
    m = re.search(r'return[_\s]*pct\s*[:=]\s*([+-]?\d+\.?\d*)', text, flags=re.IGNORECASE)
    if m:
        out['return_pct'] = float(m.group(1))
    m = re.search(r'\bret\s*[:=]\s*([+-]?\d+\.?\d*)', text, flags=re.IGNORECASE)
    if m and 'return_pct' not in out:
        out['return_pct'] = float(m.group(1))
    m = re.search(r'win[_\s]*rate\s*[:=]\s*(\d+\.?\d*)', text, flags=re.IGNORECASE)
    if m:
        out['win_rate'] = float(m.group(1))
    m = re.search(r'\bWR\s*[:=]\s*(\d+\.?\d*)', text, flags=re.IGNORECASE)
    if m and 'win_rate' not in out:
        out['win_rate'] = float(m.group(1))
    m = re.search(r'(?:n_trades|trades)\s*[:=]\s*(\d+)', text, flags=re.IGNORECASE)
    if m:
        out['trades'] = int(m.group(1))
    return out


# =============================================================================
# Orchestrator
# =============================================================================
def main():
    _summary_write('=' * 100)
    _summary_write(f'C04→C08 RUNNER — {datetime.now().isoformat()}')
    _summary_write(f'Asset: {ASSET}  Horizons: {HORIZONS}  Replay: {REPLAY}h')
    _summary_write(f'Summary file: {SUMMARY_PATH}')
    _summary_write('=' * 100)
    _summary_write('')

    results = []
    for fn in (run_c04, run_c05_c06, run_c07, run_c08):
        try:
            r = fn()
            results.append(r)
        except Exception as exc:
            _summary_write(f'  EXCEPTION in {fn.__name__}: {exc!r}')
            results.append({'name': fn.__name__, 'verdict': 'EXCEPTION',
                            'error': repr(exc)})

    _summary_write('=' * 100)
    _summary_write('FINAL VERDICTS')
    _summary_write('=' * 100)
    for r in results:
        _summary_write(f'  {r.get("name", "?"):<10}  {r.get("verdict", "?")}'
                       + (f'  avg_delta={r["avg_delta"]:+.3f}'
                          if 'avg_delta' in r else ''))
    _summary_write('')
    _summary_write(f'Summary: {SUMMARY_PATH}')


if __name__ == '__main__':
    main()
