"""
PySR Regime Discovery — Discover optimal bull/bear switching formula
====================================================================
Uses symbolic regression on HISTORICAL data to find a formula that predicts
when to use a short horizon (bull) vs long horizon (bear).

Anti-leakage design:
  The regime backtest uses the last N months for evaluation.
  PySR uses the 6 months BEFORE that window for formula discovery.
  Zero data overlap.

  --months 4 (default):  PySR = months 10->4 ago,  backtest = months 4->0
  --months 2:            PySR = months 8->2 ago,    backtest = months 2->0
  --months 6:            PySR = months 12->6 ago,   backtest = months 6->0

Usage:
  python tools/pysr_discover_regime.py                          # BTC, bull=6h, bear=8h, 4-month gap
  python tools/pysr_discover_regime.py --asset ETH              # ETH
  python tools/pysr_discover_regime.py --bull 6 --bear 10       # custom horizon pair
  python tools/pysr_discover_regime.py --months 2               # 2-month backtest gap
  python tools/pysr_discover_regime.py --iterations 80          # longer PySR run

Output:
  models/pysr_regime_{ASSET}_{BULL}h_{BEAR}h.json    -- discovered formulas + metadata
  models/pysr_regime_{ASSET}_{BULL}h_{BEAR}h_report.txt
"""

import sys
import os
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

from crypto_trading_system_doohan import load_data, TRADING_FEE

PYSR_WINDOW_HOURS = 6 * 30 * 24  # 6 months = 4320 hours
MONTHS_TO_HOURS = {1: 720, 2: 1440, 3: 2160, 4: 2880, 6: 4320}


def parse_args():
    p = argparse.ArgumentParser(description='PySR regime formula discovery')
    p.add_argument('--asset', type=str, default='BTC')
    p.add_argument('--bull', type=int, default=6, help='Bull horizon (shorter)')
    p.add_argument('--bear', type=int, default=8, help='Bear horizon (longer)')
    p.add_argument('--months', type=int, default=4, choices=[1, 2, 3, 4, 6],
                   help='Backtest gap (PySR window is 6 months before this)')
    p.add_argument('--iterations', type=int, default=40)
    p.add_argument('--populations', type=int, default=30)
    p.add_argument('--top', type=int, default=5, help='Number of top expressions to save')
    return p.parse_args()


def build_regime_features(df):
    """Build technical indicators used as PySR inputs for regime detection."""
    out = pd.DataFrame(index=df.index)

    # SMAs
    for w in [24, 48, 72, 100, 200]:
        out[f'sma_ratio_{w}'] = df['close'] / df['close'].rolling(w).mean() - 1

    # SMA crosses (as ratios)
    for fast, slow in [(24, 72), (24, 100), (48, 100), (48, 200)]:
        sma_f = df['close'].rolling(fast).mean()
        sma_s = df['close'].rolling(slow).mean()
        out[f'sma{fast}_vs_{slow}'] = sma_f / sma_s - 1

    # RSI 14
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out['rsi14'] = (100 - (100 / (1 + rs))) / 100  # normalized 0-1

    # Drawdown from rolling highs
    for w in [24, 48, 72]:
        rolling_high = df['high'].rolling(w).max()
        out[f'drawdown_{w}h'] = df['close'] / rolling_high - 1

    # Bounce from rolling lows
    for w in [24, 48]:
        rolling_low = df['low'].rolling(w).min()
        out[f'bounce_{w}h'] = df['close'] / rolling_low - 1

    # Volatility
    out['volatility_24h'] = df['close'].pct_change().rolling(24).std()
    out['volatility_48h'] = df['close'].pct_change().rolling(48).std()
    out['vol_ratio'] = out['volatility_24h'] / out['volatility_48h'].replace(0, np.nan)

    # Momentum
    for h in [6, 12, 24, 48, 72]:
        out[f'return_{h}h'] = df['close'].pct_change(h)

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    out['macd_norm'] = (ema12 - ema26) / df['close']

    # ATR normalized
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    out['atr_14h'] = tr.rolling(14).mean() / df['close']

    # Volume ratio
    if 'volume' in df.columns:
        out['volume_ratio'] = df['volume'] / df['volume'].rolling(24).mean().replace(0, np.nan)

    # Hour of day (cyclical)
    if 'datetime' in df.columns:
        dt = pd.to_datetime(df['datetime'])
        out['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
        out['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)

    return out


def build_labels(df, bull_h, bear_h):
    """
    Label = 1 if the bear (longer) horizon captures a bigger directional move.
    Label = 0 if the bull (shorter) horizon is better (choppy market).

    Uses fee-adjusted forward returns to determine which horizon wins.
    """
    fee = TRADING_FEE * 2  # round-trip fee

    # Forward returns at each horizon
    ret_bull = df['close'].shift(-bull_h) / df['close'] - 1
    ret_bear = df['close'].shift(-bear_h) / df['close'] - 1

    # Profit after fees (only if positive, otherwise loss)
    profit_bull = ret_bull.abs() - fee
    profit_bear = ret_bear.abs() - fee

    # Label = 1 when bear horizon captures more profit
    # (i.e., the move is sustained/trending, longer horizon is better)
    labels = (profit_bear > profit_bull).astype(float)

    return labels


def main():
    args = parse_args()
    backtest_hours = MONTHS_TO_HOURS[args.months]
    total_needed = PYSR_WINDOW_HOURS + backtest_hours

    print(f"\n{'='*70}")
    print(f"  PySR REGIME DISCOVERY: {args.asset}")
    print(f"  Bull horizon: {args.bull}h | Bear horizon: {args.bear}h")
    print(f"  PySR window: 6 months ({PYSR_WINDOW_HOURS}h)")
    print(f"  Backtest gap: {args.months} months ({backtest_hours}h)")
    print(f"  Total data needed: {total_needed}h ({total_needed/720:.0f} months)")
    print(f"  Iterations: {args.iterations} | Populations: {args.populations}")
    print(f"{'='*70}")

    # ── Load data ──
    print(f"\n  Loading {args.asset} data...")
    df = load_data(args.asset)
    if df is None:
        print(f"  ERROR: No data for {args.asset}")
        return

    df = df.reset_index(drop=True)
    print(f"  Total rows: {len(df)}")

    if len(df) < total_needed + 500:
        print(f"  ERROR: Not enough data. Need {total_needed + 500}, have {len(df)}")
        return

    # ── Anti-leakage windowing ──
    # Backtest window = last backtest_hours rows
    # PySR window = the 6 months before that
    pysr_end = len(df) - backtest_hours
    pysr_start = pysr_end - PYSR_WINDOW_HOURS

    if pysr_start < 0:
        print(f"  ERROR: PySR window starts before data begins")
        return

    df_pysr = df.iloc[pysr_start:pysr_end].copy().reset_index(drop=True)

    # Date range
    dt_start = df_pysr.iloc[0]['datetime'] if 'datetime' in df_pysr.columns else 'unknown'
    dt_end = df_pysr.iloc[-1]['datetime'] if 'datetime' in df_pysr.columns else 'unknown'
    bt_start = df.iloc[pysr_end]['datetime'] if 'datetime' in df.columns else 'unknown'
    bt_end = df.iloc[-1]['datetime'] if 'datetime' in df.columns else 'unknown'

    print(f"\n  PySR window:    {dt_start} to {dt_end} ({len(df_pysr)} rows)")
    print(f"  Backtest window: {bt_start} to {bt_end} ({backtest_hours} rows)")
    print(f"  Gap: ZERO overlap")

    # ── Build features ──
    print(f"\n  Building regime features...")
    features = build_regime_features(df_pysr)
    feature_cols = [c for c in features.columns if features[c].notna().sum() > len(features) * 0.8]
    print(f"  Features: {len(feature_cols)}")

    # ── Build labels ──
    print(f"  Building labels (bull={args.bull}h vs bear={args.bear}h)...")
    labels = build_labels(df_pysr, args.bull, args.bear)

    # Combine and drop NaN
    combined = features[feature_cols].copy()
    combined['label'] = labels
    combined = combined.dropna().reset_index(drop=True)

    X = combined[feature_cols].values.astype(np.float32)
    y = combined['label'].values.astype(np.float32)

    # Remove Inf
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]

    bear_pct = y.mean() * 100
    print(f"  Training samples: {len(X)}")
    print(f"  Label distribution: {bear_pct:.1f}% bear (use {args.bear}h) / {100-bear_pct:.1f}% bull (use {args.bull}h)")

    # Subsample if too large
    max_samples = 3000
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        idx.sort()
        X = X[idx]
        y = y[idx]
        print(f"  Subsampled to {max_samples} rows")

    # ── Run PySR ──
    print(f"\n  Starting PySR (this may take 30-120 min)...")
    print(f"  First run will install Julia backend (~2 min one-time)...")

    from pysr import PySRRegressor

    t0 = time.time()
    model = PySRRegressor(
        niterations=args.iterations,
        populations=args.populations,
        population_size=50,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["abs", "sqrt", "tanh"],
        maxsize=15,
        maxdepth=5,
        parsimony=0.003,
        ncycles_per_iteration=300,
        weight_optimize=0.001,
        adaptive_parsimony_scaling=100.0,
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        random_state=42,
        deterministic=True,
        procs=0,
        parallelism="serial",
        temp_equation_file=False,
        verbosity=1,
    )

    model.fit(X, y, variable_names=feature_cols)
    elapsed = (time.time() - t0) / 60
    print(f"\n  PySR completed in {elapsed:.1f} min")

    # ── Extract results ──
    equations = model.equations_
    if equations is None or len(equations) == 0:
        print("  No equations found!")
        return

    # Sort by score (trade-off between complexity and accuracy)
    eq_sorted = equations.sort_values('score', ascending=False)
    top = eq_sorted.head(args.top)

    print(f"\n  {'='*70}")
    print(f"  TOP {args.top} DISCOVERED REGIME EXPRESSIONS")
    print(f"  Formula > 0.5 => bear (use {args.bear}h) | <= 0.5 => bull (use {args.bull}h)")
    print(f"  {'='*70}")
    print(f"  {'#':>4s} | {'Score':>8s} | {'Loss':>10s} | {'Complexity':>10s} | Expression")
    print(f"  {'-'*75}")

    results = []
    for i, (_, row) in enumerate(top.iterrows()):
        eq_str = str(row['equation'])
        print(f"  {i+1:>4d} | {row['score']:>8.4f} | {row['loss']:>10.6f} | {int(row['complexity']):>10d} | {eq_str}")
        results.append({
            'equation': eq_str,
            'sympy_format': str(row.get('sympy_format', eq_str)),
            'complexity': int(row['complexity']),
            'loss': float(row['loss']),
            'score': float(row['score']),
        })

    # ── Evaluate accuracy on training set ──
    print(f"\n  Training set accuracy:")
    for i, res in enumerate(results):
        try:
            pred = model.predict(X, index=eq_sorted.index[i])
            pred_labels = (pred > 0.5).astype(float)
            acc = (pred_labels == y).mean() * 100
            # Also compute: when it says bull, what % of time bull_h actually wins?
            bull_mask = pred_labels == 0
            bear_mask = pred_labels == 1
            bull_acc = (y[bull_mask] == 0).mean() * 100 if bull_mask.sum() > 0 else 0
            bear_acc = (y[bear_mask] == 1).mean() * 100 if bear_mask.sum() > 0 else 0
            bull_pct = bull_mask.mean() * 100

            res['accuracy'] = round(acc, 1)
            res['bull_accuracy'] = round(bull_acc, 1)
            res['bear_accuracy'] = round(bear_acc, 1)
            res['bull_pct'] = round(bull_pct, 1)

            print(f"    #{i+1}: {acc:.1f}% overall | "
                  f"bull({args.bull}h)={bull_acc:.0f}% correct ({bull_pct:.0f}% of time) | "
                  f"bear({args.bear}h)={bear_acc:.0f}% correct ({100-bull_pct:.0f}% of time)")
        except Exception as e:
            print(f"    #{i+1}: evaluation error: {e}")

    # ── Save results ──
    tag = f"{args.asset}_{args.bull}h_{args.bear}h"
    json_path = os.path.join(ENGINE_DIR, 'models', f'pysr_regime_{tag}.json')
    report_path = os.path.join(ENGINE_DIR, 'models', f'pysr_regime_{tag}_report.txt')

    output = {
        'asset': args.asset,
        'bull_horizon': args.bull,
        'bear_horizon': args.bear,
        'discovery_method': 'historical',
        'pysr_window': f'{dt_start} to {dt_end}',
        'backtest_gap_months': args.months,
        'backtest_window': f'{bt_start} to {bt_end}',
        'training_samples': len(X),
        'bear_label_pct': round(bear_pct, 1),
        'feature_names': feature_cols,
        'expressions': results,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    # Report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"PySR Regime Discovery: {args.asset} bull={args.bull}h bear={args.bear}h\n")
        f.write(f"PySR window: {dt_start} to {dt_end}\n")
        f.write(f"Backtest gap: {args.months} months ({bt_start} to {bt_end})\n")
        f.write(f"Samples: {len(X)} | Bear: {bear_pct:.1f}%\n\n")
        for i, res in enumerate(results):
            f.write(f"#{i+1} (score={res['score']:.4f}, loss={res['loss']:.6f}, "
                    f"complexity={res['complexity']})\n")
            f.write(f"  {res['equation']}\n")
            if 'accuracy' in res:
                f.write(f"  Accuracy: {res['accuracy']}% | "
                        f"Bull: {res['bull_accuracy']}% | Bear: {res['bear_accuracy']}%\n")
            f.write("\n")
    print(f"  Saved: {report_path}")

    print(f"\n  Usage: formula > 0.5 => use {args.bear}h (bear), <= 0.5 => use {args.bull}h (bull)")
    print(f"  Next: run backtest_regime_master.py --months {args.months} to compare against hand-crafted detectors")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
