"""test_cdar_audit_only.py — Fix-up for test_cdar_rescore.py audit step that crashed.

Reads the existing _CDAR-tagged grid CSVs (Mode D already done) and runs the
refit + drawdown + CDaR audit pass with proper error handling.

Usage:
  python tools/test_cdar_audit_only.py
"""
from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ENGINE)
sys.path.insert(0, ENGINE)
warnings.filterwarnings('ignore')

ASSET = 'ETH'
HORIZONS = [5, 6, 7, 8]
REPLAY_HOURS = 1440
TS = datetime.now().strftime('%Y%m%d_%H%M%S')
TOP_N = 15
LAMBDAS = [0.5, 1.0, 2.0, 5.0]


_CACHED_FEATURES = {}  # cache (asset, horizon) -> (features_np, labels_np, closes_np, ranked)


def _prepare_data_cached(asset, horizon):
    """Cache feature build + ranking per (asset, horizon) — invariant across
    candidates within a horizon."""
    key = (asset, horizon)
    if key in _CACHED_FEATURES:
        return _CACHED_FEATURES[key]

    from crypto_trading_system_ed import (load_data, build_all_features,
                                          _compute_pysr_features,
                                          _test_lgbm_importance)

    df_raw = load_data(asset)
    if df_raw is None:
        raise RuntimeError(f"load_data({asset}) returned None")
    # M-32 fix (2026-05-09): _build_features returns FEATURE_SET_A (no pysr).
    # Direct build matches production Mode D and includes pysr_1..5.
    df_features, feature_cols = build_all_features(df_raw, asset_name=asset,
                                                   horizon=horizon, verbose=False)
    _compute_pysr_features(df_features, feature_cols, asset, horizon, verbose=False)
    df_clean = df_features.dropna(subset=feature_cols + ['label']).reset_index(drop=True)
    if len(df_clean) < 500:
        raise RuntimeError(f"only {len(df_clean)} clean rows")

    # Restrict to the LAST replay_hours = 1440 — matches what Mode D evaluates on
    if len(df_clean) > REPLAY_HOURS + 500:
        df_clean = df_clean.iloc[-(REPLAY_HOURS + 500):].reset_index(drop=True)
        # +500 buffer for max(window) + lookback for the warm-up

    importance_df = _test_lgbm_importance(df_clean, feature_cols, gamma=1.0)
    ranked = importance_df['feature'].tolist()
    df_op = df_clean.dropna(subset=ranked + ['label']).reset_index(drop=True)

    features_np_all = df_op[ranked].values.astype(np.float64)
    labels_np_all = df_op['label'].values.astype(np.int32)
    closes_np_all = df_op['close'].values.astype(np.float64)

    print(f"  [cache] {asset} {horizon}h: {len(df_op)} rows after replay-window trim "
          f"(last {REPLAY_HOURS} + 500 buffer)")
    _CACHED_FEATURES[key] = (features_np_all, labels_np_all, closes_np_all, ranked)
    return _CACHED_FEATURES[key]


def refit_and_compute_dd(asset: str, horizon: int, combo: list, window: int,
                          gamma: float, n_features: int):
    from crypto_trading_system_ed import (ALL_MODELS, get_decay_weights, DIAG_STEP,
                                          BACKTEST_FEE_PER_LEG, TRADING_FEE,
                                          _feature_floor_indices)

    features_np_all, labels_np_all, closes_np_all, ranked = _prepare_data_cached(asset, horizon)
    sel_idx = _feature_floor_indices(ranked, n_features)
    feat_np = features_np_all[:, sel_idx]
    n_total = len(features_np_all)

    # Walk forward with same step as engine (DIAG_STEP=36) over the last replay window
    min_start = window + 50
    portfolio = 1.0
    in_pos = False
    entry_px = 0
    trades = 0
    peak = 1.0
    max_dd = 0.0
    dd_series = []

    for i in range(min_start, n_total, DIAG_STEP):
        train_start = max(0, i - window)
        train_end = max(train_start, i - horizon)
        X_train = feat_np[train_start:train_end]
        y_train = labels_np_all[train_start:train_end]
        X_test = feat_np[i:i+1]
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

        price = closes_np_all[i]
        if pred == 1 and not in_pos:
            in_pos = True
            entry_px = price * (1 + TRADING_FEE)
        elif pred == 0 and in_pos:
            sell_px = price * (1 - BACKTEST_FEE_PER_LEG)
            portfolio *= (1 + (sell_px - entry_px) / entry_px)
            trades += 1
            in_pos = False
        cur = portfolio * (price / entry_px) if in_pos else portfolio
        if cur > peak:
            peak = cur
        dd = (peak - cur) / peak
        dd_series.append(dd)
        if dd > max_dd:
            max_dd = dd

    if in_pos:
        last_px = closes_np_all[n_total - 1]
        sell_px = last_px * (1 - BACKTEST_FEE_PER_LEG)
        portfolio *= (1 + (sell_px - entry_px) / entry_px)
        trades += 1

    ret_pct = (portfolio - 1.0) * 100
    max_dd_pct = max_dd * 100
    if dd_series:
        dd_arr = np.array(dd_series)
        thresh = np.quantile(dd_arr, 0.95)
        worst = dd_arr[dd_arr >= thresh]
        cdar_5pct = float(worst.mean() * 100) if len(worst) > 0 else max_dd_pct
    else:
        cdar_5pct = max_dd_pct
    return {'return_pct': ret_pct, 'max_dd_pct': max_dd_pct,
            'cdar_5pct': cdar_5pct, 'trades': trades}


def main():
    log_dir = os.path.join(ENGINE, 'logs')
    out_dir = os.path.join(ENGINE, 'output')
    os.makedirs(out_dir, exist_ok=True)
    summary_lines = []
    summary_lines.append('=' * 100)
    summary_lines.append(f"  CDaR AUDIT-ONLY (using existing _CDAR grid CSVs) — ETH {HORIZONS}h")
    summary_lines.append('=' * 100)

    for h in HORIZONS:
        grid_csv = os.path.join(ENGINE, 'models', f'crypto_ed_grid_{ASSET}_{h}h_CDAR.csv')
        if not os.path.exists(grid_csv):
            summary_lines.append(f"\n  --- {h}h --- MISSING grid CSV: {grid_csv}")
            continue

        df = pd.read_csv(grid_csv)
        valid = df[df['status'] == 'OK'].head(TOP_N).reset_index(drop=True).copy()
        if len(valid) == 0:
            summary_lines.append(f"\n  --- {h}h --- no OK candidates")
            continue

        # Pre-allocate columns so they exist even if all refits fail
        valid['max_dd_pct'] = float('nan')
        valid['cdar_5pct']  = float('nan')
        valid['sim_return_pct'] = float('nan')
        valid['sim_trades'] = 0

        print(f"\n[{h}h] Refitting top-{len(valid)} grid candidates...")
        n_ok = 0
        n_fail = 0
        for j in range(len(valid)):
            row = valid.iloc[j]
            try:
                m = refit_and_compute_dd(
                    ASSET, h, str(row['combo']).split('+'),
                    int(row['window']), float(row['gamma']), int(row['n_features']),
                )
                valid.loc[j, 'max_dd_pct']     = m['max_dd_pct']
                valid.loc[j, 'cdar_5pct']      = m['cdar_5pct']
                valid.loc[j, 'sim_return_pct'] = m['return_pct']
                valid.loc[j, 'sim_trades']     = m['trades']
                n_ok += 1
                print(f"  [{j+1:>2d}/{len(valid)}] {row['combo']:<10} w={int(row['window']):3d} "
                      f"g={row['gamma']} f={int(row['n_features']):2d} "
                      f"ret={m['return_pct']:+.2f}% max_dd={m['max_dd_pct']:.2f}% "
                      f"cdar5={m['cdar_5pct']:.2f}% tr={m['trades']}")
            except Exception as e:
                n_fail += 1
                print(f"  [{j+1:>2d}/{len(valid)}] FAILED: {type(e).__name__}: {e}")

        summary_lines.append(f"\n  --- {h}h ---")
        summary_lines.append(f"  Top-{TOP_N}: {n_ok} refits OK, {n_fail} failed")

        # Score under multiple objectives
        for lam in LAMBDAS:
            valid[f'cdar_score_lam{lam}'] = valid['sim_return_pct'] - lam * valid['max_dd_pct']

        # Top-3 under current scoring (apf in grid CSV)
        top_curr = valid.sort_values('apf', ascending=False).head(3)
        summary_lines.append(f"  Top-3 by current APF scoring:")
        for _, rr in top_curr.iterrows():
            summary_lines.append(
                f"    apf={rr['apf']:.3f} sim_ret={rr['sim_return_pct']:+.2f}% "
                f"max_dd={rr['max_dd_pct']:.2f}% cdar5={rr['cdar_5pct']:.2f}% "
                f"combo={rr['combo']} w={int(rr['window'])} g={rr['gamma']} f={int(rr['n_features'])}")

        for lam in LAMBDAS:
            score_col = f'cdar_score_lam{lam}'
            top_cdar = valid[valid[score_col].notna()].sort_values(score_col, ascending=False).head(3)
            summary_lines.append(f"  Top-3 by CDaR score (lambda={lam}):")
            for _, rr in top_cdar.iterrows():
                summary_lines.append(
                    f"    score={rr[score_col]:+.2f} sim_ret={rr['sim_return_pct']:+.2f}% "
                    f"max_dd={rr['max_dd_pct']:.2f}% cdar5={rr['cdar_5pct']:.2f}% "
                    f"combo={rr['combo']} w={int(rr['window'])} g={rr['gamma']} f={int(rr['n_features'])}")

        # Save audited frame
        out_csv = os.path.join(out_dir, f'cdar_audit_ETH_{h}h_{TS}.csv')
        valid.to_csv(out_csv, index=False)
        summary_lines.append(f"  Audit CSV: {out_csv}")

    summary_lines.append("\n" + '=' * 100)
    summary_lines.append("  VERDICT GUIDE:")
    summary_lines.append("  - CDaR top-3 INCLUDES configs not in current top-3 with comparable")
    summary_lines.append("    sim_returns AND lower max_dd -> CDaR scoring meaningfully reranks.")
    summary_lines.append("    Consider promoting (1-line scoring change in engine).")
    summary_lines.append("  - CDaR top-3 identical to current top-3 across all lambdas")
    summary_lines.append("    -> return and max_dd highly correlated; CDaR adds nothing. Shelve.")
    summary_lines.append('=' * 100)

    summary = '\n'.join(summary_lines)
    print('\n' + summary)

    summary_path = os.path.join(log_dir, f'cdar_audit_only_{TS}.txt')
    os.makedirs(log_dir, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\nSummary saved: {summary_path}")


if __name__ == '__main__':
    main()
