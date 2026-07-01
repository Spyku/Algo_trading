"""
crypto_trading_system_meta.py — Meta-labeling harness (López de Prado, Advances in Financial ML, Ch. 3).

STANDALONE research tool. Does NOT write to production CSV or regime config.
Safe to run alongside live trader.

Concept:
  Primary model = existing Ed pipeline (LGBM ensemble, produces BUY/SELL/HOLD + confidence).
  Secondary model = LGBM that predicts P(primary was correct) given features + primary_conf.
  Filter  at inference = execute primary's BUY only if secondary's P >= threshold.
  Sizing  at inference = scale position by P (optional, --sizing flag).

Label definition (secondary target):
  For each primary BUY at time t, label = 1 if forward_return(t → t+horizon) > 2×fee (0.22%),
  else 0. This matches the primary's own label semantic.

Validation:
  Walk-forward, step=36h, min train=200 rows, embargo=horizon (same as primary training).
  No look-ahead: secondary trains on rows [0, t-horizon), predicts rows [t, t+36).

Usage:
  python crypto_trading_system_meta.py ETH 5 --replay 1440
  python crypto_trading_system_meta.py ETH 5 --replay 1440 --p-threshold 0.55
  python crypto_trading_system_meta.py ETH 5 --replay 2880 --p-thresholds 0.45,0.50,0.55,0.60

Output:
  - Stdout: primary baseline vs meta-filtered comparison across multiple thresholds
  - output/meta_<ASSET>_<H>h_<timestamp>.csv — per-trade predictions
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ENGINE_DIR)

# Import primary pipeline bits from FAYE (the engine; ed retired 2026-07-01).
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')       # skip faye's main-mode + os.execv re-exec on import
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
os.environ.setdefault('FAYE_MODELS_DIR', 'models')
from crypto_trading_system_faye import (
    load_data,
    build_all_features,
    generate_signals,
    PRODUCTION_CSV,
    TRADING_FEE,
)

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm required. pip install lightgbm")
    sys.exit(1)


# -------- Primary config lookup --------

def load_primary_config(asset: str, horizon: int) -> dict:
    """Pick the best-scoring production row for this (asset, horizon)."""
    df = pd.read_csv(PRODUCTION_CSV)
    rows = df[(df['coin'] == asset) & (df['horizon'] == horizon)]
    if rows.empty:
        raise ValueError(f"No production model for {asset} {horizon}h in {PRODUCTION_CSV}")
    row = rows.sort_values('return_pct', ascending=False).iloc[0]
    feats = [f.strip() for f in str(row['optimal_features']).split(',') if f.strip()]
    return {
        'combo': row['best_combo'],
        'window': int(row['best_window']),
        'gamma': float(row['gamma']) if pd.notna(row.get('gamma', None)) else 1.0,
        'features': feats,
    }


# -------- Meta dataset builder --------

def build_meta_dataset(asset: str, horizon: int, replay_hours: int, primary_cfg: dict,
                       signals: list = None):
    """For each primary BUY signal in the replay window, build a meta-training row:
       X = features at signal time + primary_conf
       y = 1 if forward_return(horizon) > 2×fee else 0.

    signals: optional pre-computed signal list. If None, calls generate_signals() (slow).
    Returns (meta_df, feature_cols) where meta_df has columns:
      ['datetime', 'close', 'primary_conf', 'label'] + feature_cols
    """
    print(f"  Building meta dataset for {asset} {horizon}h (replay={replay_hours}h)...")

    # 1) Primary signals (reuse Ed's walk-forward generator — no leak)
    if signals is None:
        models = primary_cfg['combo'].split('+')
        warnings.filterwarnings('ignore')
        signals = generate_signals(
            asset_name=asset,
            model_names=models,
            window_size=primary_cfg['window'],
            replay_hours=replay_hours,
            feature_override=primary_cfg['features'],
            horizon=horizon,
            gamma=primary_cfg['gamma'],
        )
    buy_signals = [s for s in signals if s['signal'] == 'BUY']
    print(f"  Primary produced {len(signals)} signals total, {len(buy_signals)} BUYs")

    # 2) Feature matrix over the whole asset history (we only need lookup by datetime)
    df_raw = load_data(asset)
    df_feat, all_cols = build_all_features(df_raw, asset_name=asset, horizon=horizon, verbose=False)
    df_feat['datetime'] = pd.to_datetime(df_feat['datetime'])
    # Normalize signal datetimes to match
    for s in signals:
        s['datetime'] = pd.Timestamp(s['datetime'])
    df_feat = df_feat.set_index('datetime', drop=False)

    # 3) For each primary BUY, build a row: features + primary_conf + label
    records = []
    for s in buy_signals:
        dt = s['datetime']
        if dt not in df_feat.index:
            continue
        row_feat = df_feat.loc[dt]
        if isinstance(row_feat, pd.DataFrame):
            row_feat = row_feat.iloc[0]  # duplicate datetime? take first
        # Forward return over `horizon` bars
        i = df_feat.index.get_loc(dt)
        if isinstance(i, slice) or isinstance(i, np.ndarray):
            continue
        if i + horizon >= len(df_feat):
            continue
        fut_close = df_feat.iloc[i + horizon]['close']
        cur_close = row_feat['close']
        fwd_ret = fut_close / cur_close - 1.0
        label = 1 if fwd_ret > 2 * TRADING_FEE else 0

        rec = {
            'datetime': dt,
            'close': cur_close,
            'fwd_return': fwd_ret,
            'primary_conf': s['confidence'],
            'regime': s.get('regime', 'bull'),
            'horizon': s.get('horizon', horizon),
            'label': label,
        }
        for c in all_cols:
            rec[c] = row_feat.get(c, np.nan)
        records.append(rec)

    meta_df = pd.DataFrame(records).reset_index(drop=True)
    print(f"  Meta dataset: {len(meta_df)} rows (BUYs with valid forward window)")
    if len(meta_df) == 0:
        return meta_df, all_cols

    # LGBM handles NaN natively — don't drop rows just for sparse-feature gaps.
    # Only drop if primary_conf itself is missing (needed as secondary input).
    before = len(meta_df)
    meta_df = meta_df.dropna(subset=['primary_conf', 'fwd_return', 'label']).reset_index(drop=True)
    if before - len(meta_df) > 0:
        print(f"  Dropped {before - len(meta_df)} rows missing primary_conf/fwd_return/label")
    # Report sparse-feature coverage (informational)
    sparse_prefixes = ('deriv_oi_', 'ob_', 'avg_iv', 'iv_skew', 'stable_mcap_', 'whale_', 'deriv_basis')
    sparse_cols = [c for c in all_cols if c.startswith(sparse_prefixes)]
    if sparse_cols:
        frac_present = meta_df[sparse_cols].notna().mean().mean()
        print(f"  Sparse feature coverage: {100*frac_present:.0f}% of sparse cells populated (rest = NaN, LGBM handles)")
    print(f"  Label distribution: {meta_df['label'].sum()} / {len(meta_df)} positive "
          f"({100 * meta_df['label'].mean():.1f}% hit rate)")
    return meta_df, all_cols


# -------- Walk-forward secondary training --------

def walk_forward_meta_train(
    meta_df: pd.DataFrame,
    feature_cols: list,
    horizon: int,
    min_train: int = 40,
    step: int = 10,
) -> pd.DataFrame:
    """Train secondary LGBM in a walk-forward manner with embargo.
    Returns DataFrame with columns: datetime, primary_conf, label, meta_prob, fwd_return, regime."""
    print(f"  Walk-forward training: min_train={min_train}, step={step}, embargo={horizon}")
    preds = []

    X_cols = [c for c in feature_cols if c in meta_df.columns] + ['primary_conf']
    y_col = 'label'

    i = min_train
    fit_count = 0
    while i < len(meta_df):
        train_end = max(0, i - horizon)  # embargo
        train = meta_df.iloc[:train_end]
        test = meta_df.iloc[i:i + step]
        if len(train) < 50 or len(test) == 0:
            i += step
            continue
        y_train = train[y_col].values.astype(np.int32)
        if len(np.unique(y_train)) < 2:
            # single class → can't train; default to prior mean
            prior = y_train.mean() if len(y_train) else 0.5
            for _, r in test.iterrows():
                preds.append({
                    'datetime': r['datetime'], 'primary_conf': r['primary_conf'],
                    'regime': r['regime'], 'label': r['label'], 'fwd_return': r['fwd_return'],
                    'meta_prob': float(prior),
                })
            i += step
            continue

        X_train = train[X_cols].values.astype(np.float32)
        X_test = test[X_cols].values.astype(np.float32)
        # Classifier — GBDT with class balancing
        model = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            class_weight='balanced',
            verbose=-1,
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        fit_count += 1
        for j, row in enumerate(test.itertuples()):
            preds.append({
                'datetime': row.datetime, 'primary_conf': row.primary_conf,
                'regime': row.regime, 'label': row.label, 'fwd_return': row.fwd_return,
                'meta_prob': float(p[j]),
            })
        i += step

    preds_df = pd.DataFrame(preds)
    print(f"  Walk-forward fits: {fit_count}. Predictions: {len(preds_df)}")
    return preds_df


# -------- Evaluation --------

def simulate_trades(rows: pd.DataFrame, horizon: int, fee_per_leg: float = 0.0005):
    """Given a filtered set of BUY signals (rows), compute simple hold-to-horizon PnL.
    Each BUY → holds `horizon` bars → fees × 2.
    Returns (total_pnl_pct, n_trades, win_rate)."""
    if len(rows) == 0:
        return 0.0, 0, 0.0
    returns = rows['fwd_return'].values - 2 * fee_per_leg
    wins = int((returns > 0).sum())
    total = 100 * float(np.sum(returns))  # sum of %-returns (not compounded — approximate)
    wr = wins / len(rows)
    return total, len(rows), wr


def evaluate(preds_df: pd.DataFrame, horizon: int, thresholds: list, sizing: bool = False):
    """Compare primary baseline vs meta-filtered at each threshold."""
    print()
    print("=" * 90)
    print("  EVALUATION: primary-alone vs meta-filtered (hold-to-horizon, 5bps/leg fee)")
    print("=" * 90)
    base_pnl, base_n, base_wr = simulate_trades(preds_df, horizon)
    base_acc = preds_df['label'].mean() if len(preds_df) else 0.0
    print(f"  Primary baseline:")
    print(f"    BUYs taken: {base_n}")
    print(f"    Hit rate (label=1): {base_acc:.3f}")
    print(f"    Summed PnL: {base_pnl:+.2f}%  (avg/trade {base_pnl/max(base_n,1):+.3f}%)")
    print(f"    Win rate (fee-aware): {100*base_wr:.1f}%")
    print()
    print(f"  Meta-filter at thresholds (execute only if meta_prob >= p):")
    print(f"  {'p':>5} {'BUYs':>5} {'pct_kept':>9} {'hit_rate':>9} {'pnl_%':>9} {'avg_%':>8} {'wr':>7} {'delta_vs_baseline':>18}")
    for thr in thresholds:
        kept = preds_df[preds_df['meta_prob'] >= thr]
        if len(kept) == 0:
            print(f"  {thr:>5.2f} {0:>5}  (no trades pass filter)")
            continue
        p, n, wr = simulate_trades(kept, horizon)
        acc = kept['label'].mean()
        delta = p - base_pnl
        pct = 100 * n / max(base_n, 1)
        print(f"  {thr:>5.2f} {n:>5} {pct:>8.1f}% {acc:>9.3f} {p:>+8.2f}% {p/max(n,1):>+7.3f}% {100*wr:>6.1f}% {delta:>+18.2f}pp")

    # Sizing mode: scale position by meta_prob (0 if < min_p)
    if sizing:
        print()
        print("  Sizing mode (position ∝ meta_prob, gated by p_min=0.40):")
        p_min = 0.40
        kept = preds_df[preds_df['meta_prob'] >= p_min].copy()
        if len(kept):
            kept['sized_ret'] = (kept['fwd_return'] - 2 * 0.0005) * kept['meta_prob']
            sized_total = 100 * kept['sized_ret'].sum()
            sized_avg = 100 * kept['sized_ret'].mean()
            print(f"    {len(kept)} BUYs, total PnL {sized_total:+.2f}%, avg/trade {sized_avg:+.3f}%")


# -------- Main --------

def main():
    ap = argparse.ArgumentParser(description="Meta-labeling harness (non-production)")
    ap.add_argument('asset', help="Asset symbol (ETH, BTC, ...)")
    ap.add_argument('horizon', type=int, help="Primary horizon in hours (e.g. 5)")
    ap.add_argument('--replay', type=int, default=1440,
                    help="Backtest window in hours (1440=2mo, 2880=4mo)")
    ap.add_argument('--p-thresholds', default='0.40,0.45,0.50,0.55,0.60,0.65,0.70',
                    help='Comma-separated list of P thresholds to evaluate')
    ap.add_argument('--min-train', type=int, default=100,
                    help='Minimum training rows before first walk-forward fit')
    ap.add_argument('--step', type=int, default=24,
                    help='Walk-forward step size in meta rows')
    ap.add_argument('--sizing', action='store_true',
                    help='Also report position-sized variant (position ∝ meta_prob)')
    args = ap.parse_args()

    thresholds = [float(t) for t in args.p_thresholds.split(',')]

    print("=" * 90)
    print(f"  META-LABELING HARNESS — {args.asset} {args.horizon}h | replay={args.replay}h")
    print("=" * 90)
    primary_cfg = load_primary_config(args.asset, args.horizon)
    print(f"  Primary config: combo={primary_cfg['combo']} w={primary_cfg['window']}h "
          f"gamma={primary_cfg['gamma']} n_features={len(primary_cfg['features'])}")
    print()

    meta_df, feature_cols = build_meta_dataset(
        args.asset, args.horizon, args.replay, primary_cfg,
    )
    if len(meta_df) == 0:
        print("  ERROR: empty meta dataset — nothing to train on.")
        return

    preds = walk_forward_meta_train(
        meta_df, feature_cols, horizon=args.horizon,
        min_train=args.min_train, step=args.step,
    )
    if len(preds) == 0:
        print("  ERROR: no walk-forward predictions produced.")
        return

    evaluate(preds, args.horizon, thresholds, sizing=args.sizing)

    # Save per-trade CSV
    os.makedirs('output', exist_ok=True)
    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f"output/meta_{args.asset}_{args.horizon}h_{tag}.csv"
    preds.to_csv(out_path, index=False)
    print()
    print(f"  Saved per-trade predictions: {out_path}")
    print()
    print("  No production files were modified.")


if __name__ == '__main__':
    main()
