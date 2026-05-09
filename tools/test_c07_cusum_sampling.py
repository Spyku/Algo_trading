"""tools/test_c07_cusum_sampling.py — C07 CUSUM event-based sampling smoke test
(López de Prado, *Advances in Financial Machine Learning*, Ch 2).

Background:
  Currently the engine samples at fixed hourly intervals — every hour produces a
  training row regardless of whether anything informative happened. CLAUDE.md
  describes C07 as "architectural rewrite of trader polling loop. Not a smoke
  test." That's the FULL implementation. This script is the *research-grade
  smoke test* that decides whether the architectural lift is worth attempting:
  does an event-sampled training set produce a measurably better classifier
  than fixed-hourly sampling on the same data window?

Method:
  1. Load 2 years of ETH hourly OHLCV (already cached locally)
  2. Compute log returns
  3. Apply symmetric CUSUM filter at multiple thresholds h ∈ {0.5%, 1.0%, 1.5%, 2.0%}
       S_pos = max(0, S_pos + r_t); S_neg = min(0, S_neg + r_t)
       emit event when |S| > h, then reset
  4. Build a basic feature set (8 features: logret 24/72/120/240, vol 24/48,
     sma ratios) on every bar
  5. Build label (1 if forward 8h return > 2 × 0.0011 fee, else 0)
  6. For each CUSUM threshold:
     - Subset to event bars only
     - 60/40 walk-forward train/test split
     - Train LGBM-CPU on training event bars
     - Evaluate on test event bars: accuracy, precision (BUY trades only),
       and a simulated PnL where each predicted-1 is "buy at close, sell N hours
       later" using actual forward returns
  7. Compare to a fixed-hourly baseline: same features, label, split, model —
     just trained on every hourly bar. Down-sample test set to same N as
     CUSUM events at p=0.5 baseline for apples-to-apples final number.

Decision rule:
  Best CUSUM threshold's classifier shows ≥+5pp better test accuracy AND ≥+10pp
  better simulated PnL vs hourly baseline → architectural rewrite worth doing
  (commit ~days of work to refactor trader polling loop). Within ±5pp / ±10pp
  → null. Worse → confirmed DEAD, mark in scoreboard.

  This is a feasibility check only. Even if the smoke test passes, the live
  trader needs separate work to actually sample at events (currently it polls
  every hour and the engine's training loop assumes hourly cadence).

Run:
  python tools/test_c07_cusum_sampling.py

Runtime: ~5-10 min on laptop CPU (LGBM CPU-only, no full engine import).
Output: console table only; no production-state writes.
"""
from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)

ASSET = 'ETH'
HORIZON = 8       # forward-return horizon in hours (matches current bear horizon)
FEE = 0.0011      # 2× single-leg fee for label break-even
THRESHOLDS = [0.005, 0.010, 0.015, 0.020]
HISTORY_HOURS = 24 * 365 * 2  # 2 years (cap to whatever's actually in the CSV)


def cusum_events(returns: np.ndarray, h: float) -> np.ndarray:
    """Symmetric CUSUM filter (LdP AFML Ch 2). Returns boolean event mask same
    length as `returns`, True at each filter trigger (then resets)."""
    s_pos = 0.0
    s_neg = 0.0
    events = np.zeros(len(returns), dtype=bool)
    for i, r in enumerate(returns):
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_pos > h:
            events[i] = True
            s_pos = 0.0
        elif s_neg < -h:
            events[i] = True
            s_neg = 0.0
    return events


def build_features_and_label(df: pd.DataFrame, horizon: int) -> tuple:
    """Compute basic features + label on hourly OHLCV df. Returns (X, y, df_clean)."""
    df = df.copy().sort_values('datetime').reset_index(drop=True)
    df['logret'] = np.log(df['close']).diff()
    df['logret_24h'] = df['close'].pct_change(24)
    df['logret_72h'] = df['close'].pct_change(72)
    df['logret_120h'] = df['close'].pct_change(120)
    df['logret_240h'] = df['close'].pct_change(240)
    df['vol_24h'] = df['logret'].rolling(24).std()
    df['vol_48h'] = df['logret'].rolling(48).std()
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma100'] = df['close'].rolling(100).mean()
    df['sma_ratio'] = df['sma20'] / df['sma100']
    df['fwd_ret'] = df['close'].pct_change(horizon).shift(-horizon)
    df['label'] = (df['fwd_ret'] > FEE).astype(int)
    feat_cols = ['logret_24h', 'logret_72h', 'logret_120h', 'logret_240h',
                 'vol_24h', 'vol_48h', 'sma_ratio']
    df_clean = df.dropna(subset=feat_cols + ['label']).reset_index(drop=True)
    return df_clean[feat_cols].values, df_clean['label'].values, df_clean


def evaluate(X_train, y_train, X_test, y_test, fwd_ret_test):
    """Train LGBM-CPU; return dict with accuracy, n_buy_predicted, precision,
    avg_buy_return_pct, simulated_pnl_pct (cumulative product of (1+fwd_ret)
    over predicted-BUY rows, expressed as %)."""
    from lightgbm import LGBMClassifier
    if len(np.unique(y_train)) < 2:
        return None
    model = LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        class_weight='balanced', verbose=-1, random_state=42, device='cpu',
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = (preds == y_test).mean()
    buy_mask = preds == 1
    n_buy = int(buy_mask.sum())
    if n_buy == 0:
        return {'accuracy': accuracy, 'n_buy': 0, 'precision': float('nan'),
                'avg_buy_ret': float('nan'), 'sim_pnl_pct': 0.0}
    precision = (y_test[buy_mask] == 1).mean()
    avg_buy_ret = float(np.nanmean(fwd_ret_test[buy_mask])) * 100  # to %
    # Simulated PnL: compound (1 + fwd_ret − 2×fee) on every predicted-BUY row
    realised = fwd_ret_test[buy_mask] - 2 * FEE
    sim_pnl = float(np.prod(1 + realised) - 1.0) * 100
    return {
        'accuracy': float(accuracy),
        'n_buy': n_buy,
        'precision': float(precision),
        'avg_buy_ret': avg_buy_ret,
        'sim_pnl_pct': sim_pnl,
    }


def main():
    print('=' * 90)
    print('  C07 — CUSUM EVENT-BASED SAMPLING SMOKE TEST')
    print('  (López de Prado AFML Ch 2)')
    print('=' * 90)

    csv_path = os.path.join(ENGINE, f'data/{ASSET.lower()}_hourly_data.csv')
    print(f'  Loading {csv_path}')
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    if len(df) > HISTORY_HOURS:
        df = df.tail(HISTORY_HOURS).reset_index(drop=True)
    print(f'  Window: {df["datetime"].iloc[0]} -> {df["datetime"].iloc[-1]} ({len(df)} hourly bars)')
    print()

    X_all, y_all, df_clean = build_features_and_label(df, HORIZON)
    fwd_ret_all = df_clean['fwd_ret'].values
    n = len(df_clean)
    train_n = int(n * 0.6)
    print(f'  Built {n} clean rows. Walk-forward split: train={train_n} ({df_clean["datetime"].iloc[train_n-1]}), test={n - train_n}')
    print(f'  Label rate (test): {y_all[train_n:].mean()*100:.1f}% positives')
    print()

    # --- Baseline: every hourly bar ---
    print(f'  {"Strategy":<35} {"N test":>8} {"Acc":>7} {"#BUY":>6} {"Prec":>7} {"AvgRet":>8} {"SimPnL":>9}')
    print(f'  {"-"*84}')
    base = evaluate(X_all[:train_n], y_all[:train_n],
                    X_all[train_n:], y_all[train_n:],
                    fwd_ret_all[train_n:])
    print(f'  {"hourly (every bar) baseline":<35} {n-train_n:>8d} '
          f'{base["accuracy"]*100:>6.1f}% {base["n_buy"]:>6d} '
          f'{base["precision"]*100:>6.1f}% {base["avg_buy_ret"]:>+7.2f}% {base["sim_pnl_pct"]:>+8.2f}%')

    # --- CUSUM event sampling at multiple thresholds ---
    logret = df_clean['logret'].values
    for h in THRESHOLDS:
        events = cusum_events(logret, h)
        n_events = int(events.sum())
        if n_events < 100:
            print(f'  cusum h={h*100:.1f}% events={n_events:<5d}  SKIP — too few events')
            continue
        # Subset to event bars only
        X_ev = X_all[events]
        y_ev = y_all[events]
        fwd_ret_ev = fwd_ret_all[events]
        train_n_ev = int(len(X_ev) * 0.6)
        if train_n_ev < 50 or len(X_ev) - train_n_ev < 30:
            print(f'  cusum h={h*100:.1f}% events={n_events:<5d}  SKIP — split too small')
            continue
        result = evaluate(X_ev[:train_n_ev], y_ev[:train_n_ev],
                          X_ev[train_n_ev:], y_ev[train_n_ev:],
                          fwd_ret_ev[train_n_ev:])
        if result is None:
            print(f'  cusum h={h*100:.1f}% events={n_events:<5d}  SKIP — single-class train')
            continue
        rate_pct_per_year = n_events / (n / (24 * 365)) if n > 0 else 0
        print(f'  {f"cusum h={h*100:.1f}% events={n_events}":<35} '
              f'{len(X_ev)-train_n_ev:>8d} '
              f'{result["accuracy"]*100:>6.1f}% {result["n_buy"]:>6d} '
              f'{result["precision"]*100:>6.1f}% {result["avg_buy_ret"]:>+7.2f}% '
              f'{result["sim_pnl_pct"]:>+8.2f}%')

    print()
    print('=' * 90)
    print('  DECISION RULE')
    print('=' * 90)
    print('  Best CUSUM threshold beats hourly baseline by:')
    print('    >=+5pp accuracy AND >=+10pp simulated PnL -> architectural rewrite worth doing')
    print('    Within +/-5pp / +/-10pp -> null')
    print('    Worse -> CUSUM family DEAD, mark in scoreboard')
    print()
    print('  Caveats:')
    print('    1. SimPnL above is per-trade, not portfolio simulation — does not apply')
    print('       shield, max_hold, gates. It is a quick relative comparison only.')
    print('    2. Features used here are 7 basic price/vol — production engine uses 184.')
    print('       If CUSUM helps with this small set, it would likely help more with full set.')
    print('    3. CUSUM sampling reduces N. The accuracy comparison is fair (different test')
    print('       sets) but not strictly apples-to-apples on regime coverage. Confirmation')
    print('       run would need to subsample hourly baseline to same N for tighter comparison.')


if __name__ == '__main__':
    main()
