"""
Test #3: Asymmetric Loss — penalise false BUYs more than false HOLDs.

Compares baseline (class_weight='balanced') vs scale_pos_weight variants
on ETH 6h and 8h using the same walk-forward framework as Mode D.

Usage: python tools/test_asymmetric_loss.py
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from crypto_trading_system_ed import (
    load_data, build_all_features, _compute_pysr_features,
    simulate_portfolio, TRADING_FEE, _suppress_stderr, BACKTEST_FEE_PER_LEG,
    get_decay_weights,
)
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

ASSET = 'ETH'
HORIZONS = [6, 8]
REPLAY = 1440
DIAG_STEP = 36

# scale_pos_weight: <1 = penalise false BUYs, >1 = penalise false SELLs, 1 = neutral
WEIGHTS_TO_TEST = [0.3, 0.5, 0.7, 1.0, 1.5]  # 1.0 = baseline (balanced only)


def run_walkforward(df, feature_cols, horizon, gamma, scale_weight, window):
    """Walk-forward evaluation with given scale_pos_weight."""
    n = len(df)
    start_idx = max(window + 50, n - REPLAY)
    signals = []

    for i in range(start_idx, n, DIAG_STEP):
        train_start = max(0, i - window)
        train_end = max(train_start, i - horizon)  # embargo
        train = df.iloc[train_start:train_end]
        X_train = train[feature_cols].values
        y_train = train['label'].values

        if len(np.unique(y_train)) < 2:
            continue
        if len(y_train) < 50:
            continue

        # Test on next DIAG_STEP candles
        test_end = min(i + DIAG_STEP, n)
        test = df.iloc[i:test_end]
        X_test = test[feature_cols].values

        sw = get_decay_weights(len(y_train), gamma)

        # LGBM with asymmetric weight
        lgbm = LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_weight,
            verbose=-1, random_state=42, device='gpu'
        )
        lgbm.fit(X_train, y_train, sample_weight=sw)

        # RF baseline (no scale_pos_weight param — use class_weight)
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=4, random_state=42, n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(X_train, y_train, sample_weight=sw)

        for j in range(len(test)):
            row = test.iloc[j]
            x = X_test[j:j+1]

            # Ensemble vote (RF + LGBM)
            p_lgbm = lgbm.predict_proba(x)[0]
            p_rf = rf.predict_proba(x)[0]
            avg_prob = (p_lgbm + p_rf) / 2
            pred = int(avg_prob[1] > 0.5)
            conf = max(avg_prob) * 100

            signal = 'BUY' if pred == 1 else 'SELL'
            signals.append({
                'datetime': row['datetime'],
                'signal': signal,
                'confidence': conf,
                'close': row['close'],
            })

    return simulate_portfolio(signals)


def simulate_trades(signals, conf_threshold=90):
    """Simple portfolio simulation."""
    cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
    trades, wins = 0, 0
    first_price = last_price = None

    for s in signals:
        price = s['close']
        sig = s['signal']
        conf = s['confidence']
        if first_price is None:
            first_price = price
        last_price = price

        if sig == 'BUY' and conf >= conf_threshold and not in_pos:
            held = cash * (1 - BACKTEST_FEE_PER_LEG) / price
            cash = 0
            in_pos = True
            entry_px = price
            trades += 1
        elif sig == 'SELL' and in_pos:
            cash = held * price * (1 - BACKTEST_FEE_PER_LEG)
            if price > entry_px:
                wins += 1
            held = 0
            in_pos = False

    if in_pos and last_price:
        cash = held * last_price * (1 - BACKTEST_FEE_PER_LEG)
        if last_price > entry_px:
            wins += 1

    ret = (cash / 1000.0 - 1) * 100
    wr = (wins / trades * 100) if trades > 0 else 0
    bh = (last_price / first_price - 1) * 100 if first_price and last_price else 0
    return ret, trades, wr, bh


def main():
    print("=" * 70)
    print("  TEST #3: ASYMMETRIC LOSS (scale_pos_weight)")
    print("  ETH 6h + 8h | 2-month replay | conf>=90%")
    print("=" * 70)

    # Load production model configs for window/gamma/features
    prod_csv = 'models/crypto_ed_production.csv'
    df_models = pd.read_csv(prod_csv)

    for h in HORIZONS:
        rows = df_models[(df_models['coin'] == ASSET) & (df_models['horizon'] == h)]
        if len(rows) == 0:
            print(f"\n  No production model for {ASSET} {h}h — skipping")
            continue
        row = rows.iloc[0]
        feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
        gamma = float(row.get('gamma', 1.0))
        window = int(row['best_window'])

        print(f"\n{'='*70}")
        print(f"  ETH {h}h | w={window} | g={gamma} | f={len(feats)} features")
        print(f"{'='*70}")

        # Build features
        df_raw = load_data(ASSET)
        with _suppress_stderr():
            df_full, all_cols = build_all_features(df_raw, asset_name=ASSET, horizon=h, verbose=False)
            _compute_pysr_features(df_full, all_cols, ASSET, h, verbose=False)

        feature_cols = [f for f in feats if f in all_cols]
        # Keep rows with core features
        sparse_prefixes = ('deriv_oi_', 'ob_', 'avg_iv', 'iv_skew', 'stable_mcap_', 'whale_')
        core_cols = [c for c in feature_cols if not any(c.startswith(p) for p in sparse_prefixes)]
        df = df_full.dropna(subset=core_cols + ['label']).reset_index(drop=True)

        print(f"  Clean rows: {len(df)}")
        print(f"\n  {'Weight':<10} {'Return':>8} {'Trades':>7} {'WR':>6} {'B&H':>8} {'Alpha':>8}")
        print(f"  {'-'*50}")

        for w in WEIGHTS_TO_TEST:
            t0 = time.time()
            sigs = run_walkforward(df, feature_cols, h, gamma, w, window)
            ret, tr, wr, bh = simulate_trades(sigs, conf_threshold=90)
            alpha = ret - bh
            elapsed = time.time() - t0
            marker = " <-- baseline" if w == 1.0 else ""
            print(f"  {w:<10.1f} {ret:>+7.2f}% {tr:>7d} {wr:>5.0f}% {bh:>+7.2f}% {alpha:>+7.2f}%  ({elapsed:.0f}s){marker}")


if __name__ == '__main__':
    main()
