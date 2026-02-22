"""
╔═══════════════════════════════════════════════════════════════╗
║         BROLY REGIME BACKTEST — INDICES vs CRYPTO             ║
║                                                               ║
║  Question: Does BULL/BEAR/SIDEWAYS regime detection           ║
║  actually improve ML accuracy for European indices?           ║
║  Or is it only useful for crypto?                             ║
║                                                               ║
║  Test:                                                        ║
║   For each asset (DAX, SMI, CAC40, BTC, ETH, SOL):           ║
║    A) UNIVERSAL model (all data)                              ║
║    B) REGIME-CONDITIONED model (separate per regime)          ║
║    C) REGIME-AS-FEATURE model (regime added as input)         ║
║                                                               ║
║  Walk-forward validation (no look-ahead)                      ║
║  Uses same feature pipeline as your existing systems          ║
║                                                               ║
║  Usage: conda activate algo_trading                           ║
║         python regime_backtest.py                             ║
╚═══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings, time, sys
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
from functools import partial

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# ============================================================
# CONFIG
# ============================================================

ASSETS = {
    # European indices (hourly → daily)
    'DAX':   {'ticker': '^GDAXI', 'type': 'index'},
    'SMI':   {'ticker': '^SSMI',  'type': 'index'},
    'CAC40': {'ticker': '^FCHI',  'type': 'index'},
    # Crypto (daily)
    'BTC':   {'ticker': 'BTC-USD', 'type': 'crypto'},
    'ETH':   {'ticker': 'ETH-USD', 'type': 'crypto'},
    'SOL':   {'ticker': 'SOL-USD', 'type': 'crypto'},
}

WINDOW = 200           # Walk-forward training window
PRED_HORIZON = 3       # Predict 3 days ahead
TEST_FRACTION = 0.35   # Use last 35% for testing
STEP = 3               # Every 3rd bar (speed up)

# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_macro() -> pd.DataFrame:
    """Download VIX + S&P500 for regime classification."""
    print("  Downloading macro data (VIX, S&P500)...")
    vix = yf.download('^VIX', period='5y', interval='1d', progress=False)
    sp = yf.download('^GSPC', period='5y', interval='1d', progress=False)

    # Handle MultiIndex columns
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = sp.columns.get_level_values(0)

    macro = pd.DataFrame(index=vix.index)
    macro['vix'] = vix['Close']
    macro['sp500'] = sp['Close'].reindex(vix.index, method='ffill')
    macro = macro.dropna()
    macro.index = macro.index.tz_localize(None) if macro.index.tz else macro.index

    print(f"    Macro data: {len(macro)} days ({macro.index[0].date()} → {macro.index[-1].date()})")
    return macro


def download_asset(name: str, config: dict) -> pd.DataFrame:
    """Download daily OHLCV for an asset."""
    print(f"  Downloading {name} ({config['ticker']})...")
    df = yf.download(config['ticker'], period='5y', interval='1d', progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    })

    # Drop any fully-NaN rows
    df = df.dropna(subset=['close'])
    print(f"    {name}: {len(df)} days")
    return df


# ============================================================
# REGIME LABELLING
# ============================================================

def label_regimes(macro: pd.DataFrame) -> pd.Series:
    """
    Label each day as BULL / BEAR / SIDEWAYS.

    BULL:     VIX < 20  AND  S&P > SMA200  AND  6m trend > +5%
    BEAR:     VIX > 28  AND  S&P < SMA200  AND  6m trend < -5%
    SIDEWAYS: Everything else
    """
    df = macro.copy()
    df['sp_sma200'] = df['sp500'].rolling(200, min_periods=50).mean()
    df['sp_vs_sma'] = (df['sp500'] / df['sp_sma200'] - 1) * 100
    df['sp_trend_6m'] = df['sp500'].pct_change(126) * 100

    regimes = []
    for _, row in df.iterrows():
        vix = row['vix']
        sp_vs = row['sp_vs_sma']
        trend = row['sp_trend_6m']

        bull = 0
        bear = 0

        # VIX
        if vix < 15: bull += 3
        elif vix < 20: bull += 2
        elif vix < 25: bull += 1
        elif vix < 30: bear += 1
        elif vix < 35: bear += 2
        else: bear += 3

        # S&P vs SMA200
        if not pd.isna(sp_vs):
            if sp_vs > 5: bull += 2
            elif sp_vs > 0: bull += 1
            elif sp_vs > -5: bear += 1
            else: bear += 2

        # 6m trend
        if not pd.isna(trend):
            if trend > 10: bull += 2
            elif trend > 5: bull += 1
            elif trend > -5: pass
            elif trend > -10: bear += 1
            else: bear += 2

        if bull >= bear + 3:
            regimes.append('BULL')
        elif bear >= bull + 3:
            regimes.append('BEAR')
        else:
            regimes.append('SIDEWAYS')

    return pd.Series(regimes, index=df.index)


# ============================================================
# FEATURE ENGINEERING (self-contained, matches your systems)
# ============================================================

def build_features(df: pd.DataFrame, asset_type: str) -> tuple:
    """
    Build features + labels from daily OHLCV.
    Returns (df_with_features, feature_cols).
    """
    df = df.copy()

    # --- Log Returns ---
    for p in [1, 2, 3, 5, 7, 10, 14, 20, 30, 50, 100]:
        df[f'logret_{p}d'] = np.log(df['close'] / df['close'].shift(p))

    # --- Spreads ---
    df['spread_10_2'] = df['logret_10d'] - df['logret_2d']
    df['spread_20_2'] = df['logret_20d'] - df['logret_2d']
    df['spread_30_10'] = df['logret_30d'] - df['logret_10d']
    df['spread_7_3'] = df['logret_7d'] - df['logret_3d']
    df['spread_100_10'] = df['logret_100d'] - df['logret_10d']

    # --- SMA ---
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['sma200'] = df['close'].rolling(200).mean()
    df['price_to_sma20'] = df['close'] / df['sma20'] - 1
    df['price_to_sma50'] = df['close'] / df['sma50'] - 1
    df['sma20_to_sma50'] = df['sma20'] / df['sma50'] - 1

    # --- RSI ---
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # --- Bollinger Band Position ---
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - (bb_mid - 2 * bb_std)) / (4 * bb_std)

    # --- Z-Score ---
    roll_mean = df['close'].rolling(30).mean()
    roll_std = df['close'].rolling(30).std()
    df['zscore_30d'] = (df['close'] - roll_mean) / roll_std

    # --- ATR ---
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    df['atr_pct'] = tr.rolling(14).mean() / df['close']

    # --- Volatility ---
    df['volatility_10d'] = df['logret_1d'].rolling(10).std()
    df['volatility_30d'] = df['logret_1d'].rolling(30).std()
    df['vol_ratio'] = df['volatility_10d'] / df['volatility_30d']

    # --- Volume ---
    if df['volume'].sum() == 0 or df['volume'].isna().all():
        df['volume_ratio'] = 1.0
    else:
        df['volume'] = df['volume'].replace(0, np.nan).ffill().bfill()
        vol_sma20 = df['volume'].rolling(20).mean()
        df['volume_ratio'] = (df['volume'] / vol_sma20).fillna(1.0)

    # --- Day of week ---
    dow = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # --- Labels ---
    future_return = df['close'].shift(-PRED_HORIZON) / df['close'] - 1
    rolling_med = future_return.rolling(90, min_periods=30).median().shift(PRED_HORIZON)
    df['label'] = (future_return > rolling_med).astype(float)

    feature_cols = [
        'logret_1d', 'logret_2d', 'logret_3d', 'logret_5d', 'logret_7d',
        'logret_10d', 'logret_14d', 'logret_20d', 'logret_30d',
        'spread_10_2', 'spread_20_2', 'spread_30_10', 'spread_7_3', 'spread_100_10',
        'price_to_sma20', 'price_to_sma50', 'sma20_to_sma50',
        'rsi_14', 'bb_position', 'zscore_30d', 'atr_pct',
        'volatility_10d', 'volatility_30d', 'vol_ratio',
        'volume_ratio', 'dow_sin', 'dow_cos',
    ]

    df = df.dropna(subset=feature_cols + ['label']).reset_index(drop=True)
    return df, feature_cols


# ============================================================
# WALK-FORWARD ENGINE
# ============================================================

def _make_models():
    """Create ensemble model factories."""
    factories = {
        'RF': lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=1
        ),
        'GB': lambda: GradientBoostingClassifier(
            n_estimators=150, max_depth=4, random_state=42
        ),
    }
    if HAS_LGBM:
        factories['LGBM'] = lambda: lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=42, verbose=-1, n_jobs=1
        )
    return factories


def walk_forward_accuracy(X, y, window, step, test_start, model_factories=None):
    """
    Walk-forward ensemble accuracy.
    Returns (accuracy, correct, total).
    """
    if model_factories is None:
        model_factories = _make_models()

    n = len(X)
    correct = 0
    total = 0

    for i in range(test_start, n, step):
        train_start = max(0, i - window)
        X_train = X[train_start:i]
        y_train = y[train_start:i]

        if len(X_train) < 30 or len(np.unique(y_train)) < 2:
            continue

        X_test = X[i:i+1]
        y_test = y[i]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        votes = []
        for name, factory in model_factories.items():
            try:
                model = factory()
                model.fit(X_train_s, y_train)
                votes.append(int(model.predict(X_test_s)[0]))
            except:
                pass

        if not votes:
            continue

        pred = 1 if sum(votes) > len(votes) / 2 else 0
        if pred == int(y_test):
            correct += 1
        total += 1

    acc = round(correct / total * 100, 2) if total > 0 else 0
    return acc, correct, total


# ============================================================
# REGIME BACKTEST FOR ONE ASSET
# ============================================================

def backtest_asset(name, config, macro, regime_labels):
    """
    Run 3 approaches for one asset:
      A) UNIVERSAL — trained on all data
      B) REGIME-CONDITIONED — separate model per current regime
      C) REGIME-AS-FEATURE — regime added as input features

    Returns dict of results.
    """
    print(f"\n{'='*60}")
    print(f"  BACKTESTING: {name} ({config['type'].upper()})")
    print(f"{'='*60}")

    # Download & build features
    df_raw = download_asset(name, config)
    if len(df_raw) < 300:
        print(f"  ⚠️  Insufficient data ({len(df_raw)} rows), skipping")
        return None

    df, feature_cols = build_features(df_raw, config['type'])
    print(f"  Features: {len(feature_cols)} | Rows: {len(df)}")

    # Map regimes to asset dates
    df_dates = df_raw.index[:len(df)] if len(df_raw) >= len(df) else df_raw.index
    # Build regime column by matching dates
    regime_map = regime_labels.to_dict()
    asset_regimes = []
    for idx in range(len(df)):
        # Find closest macro date
        if hasattr(df, 'index') and hasattr(df.index[idx], 'date'):
            d = df.index[idx]
        else:
            d = df_raw.index[idx] if idx < len(df_raw) else None

        if d is not None:
            # Try exact match first, then nearest
            if d in regime_map:
                asset_regimes.append(regime_map[d])
            else:
                # Find nearest date
                diffs = np.abs(regime_labels.index - d)
                nearest = regime_labels.index[diffs.argmin()]
                asset_regimes.append(regime_labels[nearest])
        else:
            asset_regimes.append('SIDEWAYS')

    df['regime'] = asset_regimes

    # Regime distribution
    regime_counts = df['regime'].value_counts()
    print(f"\n  Regime distribution:")
    for r in ['BULL', 'BEAR', 'SIDEWAYS']:
        c = regime_counts.get(r, 0)
        print(f"    {r:10s}: {c:5d} days ({c/len(df)*100:5.1f}%)")

    # Prepare data
    X = df[feature_cols].values
    y = df['label'].values
    n = len(X)
    test_start = int(n * (1 - TEST_FRACTION))

    print(f"\n  Train: {test_start} rows | Test: {n - test_start} rows")
    print(f"  Window: {WINDOW} | Step: {STEP}")

    factories = _make_models()

    # ── A) UNIVERSAL (all data, no regime info) ──
    print(f"\n  [A] UNIVERSAL model (no regime info)...")
    t0 = time.time()
    acc_universal, corr_u, tot_u = walk_forward_accuracy(
        X, y, WINDOW, STEP, test_start, factories
    )
    t_u = time.time() - t0
    print(f"      Accuracy: {acc_universal:.2f}% ({corr_u}/{tot_u}) [{t_u:.0f}s]")

    # ── B) REGIME-CONDITIONED (separate model per regime) ──
    print(f"\n  [B] REGIME-CONDITIONED (separate models per regime)...")
    t0 = time.time()
    regime_arr = np.array(asset_regimes)

    correct_rc = 0
    total_rc = 0

    for i in range(test_start, n, STEP):
        current_regime = regime_arr[i]

        # Train only on data from this regime within window
        train_start = max(0, i - WINDOW)
        regime_mask = regime_arr[train_start:i] == current_regime
        X_regime_train = X[train_start:i][regime_mask]
        y_regime_train = y[train_start:i][regime_mask]

        # Fallback: if too few regime-specific samples, use all data
        if len(X_regime_train) < 30 or len(np.unique(y_regime_train)) < 2:
            X_regime_train = X[train_start:i]
            y_regime_train = y[train_start:i]

        if len(X_regime_train) < 30 or len(np.unique(y_regime_train)) < 2:
            continue

        X_test = X[i:i+1]
        y_test = y[i]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_regime_train)
        X_test_s = scaler.transform(X_test)

        votes = []
        for fname, factory in factories.items():
            try:
                model = factory()
                model.fit(X_train_s, y_regime_train)
                votes.append(int(model.predict(X_test_s)[0]))
            except:
                pass

        if not votes:
            continue

        pred = 1 if sum(votes) > len(votes) / 2 else 0
        if pred == int(y_test):
            correct_rc += 1
        total_rc += 1

    acc_regime_cond = round(correct_rc / total_rc * 100, 2) if total_rc > 0 else 0
    t_rc = time.time() - t0
    print(f"      Accuracy: {acc_regime_cond:.2f}% ({correct_rc}/{total_rc}) [{t_rc:.0f}s]")

    # ── C) REGIME-AS-FEATURE (add regime as extra features) ──
    print(f"\n  [C] REGIME-AS-FEATURE (regime added as ML inputs)...")
    t0 = time.time()

    # Add regime one-hot + numeric score as features
    X_with_regime = np.column_stack([
        X,
        (regime_arr == 'BULL').astype(float),
        (regime_arr == 'BEAR').astype(float),
        (regime_arr == 'SIDEWAYS').astype(float),
    ])

    acc_regime_feat, corr_rf, tot_rf = walk_forward_accuracy(
        X_with_regime, y, WINDOW, STEP, test_start, factories
    )
    t_rf = time.time() - t0
    print(f"      Accuracy: {acc_regime_feat:.2f}% ({corr_rf}/{tot_rf}) [{t_rf:.0f}s]")

    # ── D) Per-regime accuracy breakdown (using universal model) ──
    print(f"\n  [D] ACCURACY BREAKDOWN BY REGIME (universal model):")
    regime_accs = {}
    for regime in ['BULL', 'BEAR', 'SIDEWAYS']:
        regime_indices = [i for i in range(test_start, n, STEP) if regime_arr[i] == regime]
        if len(regime_indices) < 5:
            regime_accs[regime] = None
            print(f"      {regime:10s}: too few samples ({len(regime_indices)})")
            continue

        r_correct = 0
        r_total = 0
        for i in regime_indices:
            train_start_r = max(0, i - WINDOW)
            X_train = X[train_start_r:i]
            y_train = y[train_start_r:i]

            if len(X_train) < 30 or len(np.unique(y_train)) < 2:
                continue

            X_test = X[i:i+1]
            y_test = y[i]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            votes = []
            for fname, factory in factories.items():
                try:
                    m = factory()
                    m.fit(X_train_s, y_train)
                    votes.append(int(m.predict(X_test_s)[0]))
                except:
                    pass

            if not votes:
                continue

            pred = 1 if sum(votes) > len(votes) / 2 else 0
            if pred == int(y_test):
                r_correct += 1
            r_total += 1

        r_acc = round(r_correct / r_total * 100, 2) if r_total > 0 else 0
        regime_accs[regime] = r_acc
        print(f"      {regime:10s}: {r_acc:.2f}% ({r_correct}/{r_total})")

    # ── Results ──
    result = {
        'asset': name,
        'type': config['type'],
        'rows': len(df),
        'test_rows': n - test_start,
        'universal_acc': acc_universal,
        'regime_conditioned_acc': acc_regime_cond,
        'regime_feature_acc': acc_regime_feat,
        'lift_conditioned': round(acc_regime_cond - acc_universal, 2),
        'lift_feature': round(acc_regime_feat - acc_universal, 2),
        'best_approach': 'CONDITIONED' if acc_regime_cond > max(acc_universal, acc_regime_feat)
                    else ('FEATURE' if acc_regime_feat > acc_universal else 'UNIVERSAL'),
        'regime_dist': dict(regime_counts),
        'regime_accs': regime_accs,
    }

    print(f"\n  ┌──────────────────────────────────────────────────┐")
    print(f"  │ RESULT: {name:6s}                                  │")
    print(f"  ├──────────────────────────────────────────────────┤")
    print(f"  │ [A] Universal:          {acc_universal:6.2f}%                 │")
    print(f"  │ [B] Regime-Conditioned: {acc_regime_cond:6.2f}%  ({result['lift_conditioned']:+.2f}%)     │")
    print(f"  │ [C] Regime-as-Feature:  {acc_regime_feat:6.2f}%  ({result['lift_feature']:+.2f}%)     │")
    print(f"  │ → Best: {result['best_approach']:15s}                      │")
    print(f"  └──────────────────────────────────────────────────┘")

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  ⚡ BROLY REGIME BACKTEST")
    print("  Indices vs Crypto — Does regime detection help?")
    print("=" * 60)

    t_start = time.time()

    # Step 1: Download macro data for regime classification
    macro = download_macro()
    regime_labels = label_regimes(macro)

    # Show overall regime distribution
    regime_counts = regime_labels.value_counts()
    print(f"\n  Overall regime distribution (last 5 years):")
    for r in ['BULL', 'BEAR', 'SIDEWAYS']:
        c = regime_counts.get(r, 0)
        print(f"    {r:10s}: {c:5d} days ({c/len(regime_labels)*100:.1f}%)")

    # Step 2: Backtest each asset
    results = []
    for name, config in ASSETS.items():
        try:
            result = backtest_asset(name, config, macro, regime_labels)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n  ⚠️  Error on {name}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\n  No results generated!")
        return

    # Step 3: Summary comparison
    print(f"\n\n{'='*75}")
    print(f"  ⚡ FINAL RESULTS: DOES REGIME DETECTION HELP?")
    print(f"{'='*75}")

    print(f"\n  {'Asset':<8} {'Type':<8} {'Universal':>10} {'Conditioned':>12} {'AsFeature':>10} {'Best Lift':>10} {'Best':>13}")
    print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*12} {'─'*10} {'─'*10} {'─'*13}")

    index_lifts = []
    crypto_lifts = []

    for r in results:
        best_lift = max(r['lift_conditioned'], r['lift_feature'])
        emoji = '✅' if best_lift > 0 else '❌'
        print(f"  {r['asset']:<8} {r['type']:<8} {r['universal_acc']:>9.2f}% "
              f"{r['regime_conditioned_acc']:>11.2f}% {r['regime_feature_acc']:>9.2f}% "
              f"{best_lift:>+9.2f}% {emoji} {r['best_approach']}")

        if r['type'] == 'index':
            index_lifts.append(best_lift)
        else:
            crypto_lifts.append(best_lift)

    # Average lift by asset type
    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    if index_lifts:
        avg_idx = np.mean(index_lifts)
        emoji_idx = '✅' if avg_idx > 0 else '❌'
        print(f"  │ INDICES avg regime lift:  {avg_idx:+.2f}%  {emoji_idx}                             │")
    if crypto_lifts:
        avg_cry = np.mean(crypto_lifts)
        emoji_cry = '✅' if avg_cry > 0 else '❌'
        print(f"  │ CRYPTO  avg regime lift:  {avg_cry:+.2f}%  {emoji_cry}                             │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    # Verdict
    print(f"\n  VERDICT:")
    if index_lifts and crypto_lifts:
        if np.mean(index_lifts) > np.mean(crypto_lifts):
            print(f"  → Regime detection helps INDICES MORE than crypto")
        elif np.mean(crypto_lifts) > np.mean(index_lifts):
            print(f"  → Regime detection helps CRYPTO MORE than indices")
        else:
            print(f"  → Regime detection helps BOTH equally")

        if np.mean(index_lifts) > 0.5:
            print(f"  → ✅ RECOMMEND adding regime to indices pipeline")
        elif np.mean(index_lifts) > 0:
            print(f"  → 🟡 MARGINAL benefit for indices — use as feature, not separate models")
        else:
            print(f"  → ❌ Regime detection does NOT help indices — skip it")

        if np.mean(crypto_lifts) > 0.5:
            print(f"  → ✅ RECOMMEND adding regime to crypto pipeline")
        elif np.mean(crypto_lifts) > 0:
            print(f"  → 🟡 MARGINAL benefit for crypto — use as feature, not separate models")
        else:
            print(f"  → ❌ Regime detection does NOT help crypto — skip it")

    # Per-regime accuracy breakdown
    print(f"\n  ACCURACY BY REGIME (universal model):")
    print(f"  {'Asset':<8} {'BULL':>10} {'BEAR':>10} {'SIDEWAYS':>10} {'Spread':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for r in results:
        accs = r.get('regime_accs', {})
        b = accs.get('BULL')
        br = accs.get('BEAR')
        s = accs.get('SIDEWAYS')
        vals = [x for x in [b, br, s] if x is not None]
        spread = max(vals) - min(vals) if len(vals) >= 2 else 0
        b_str = f"{b:9.1f}%" if b is not None else f"{'N/A':>10s}"
        br_str = f"{br:9.1f}%" if br is not None else f"{'N/A':>10s}"
        s_str = f"{s:9.1f}%" if s is not None else f"{'N/A':>10s}"
        print(f"  {r['asset']:<8} {b_str} {br_str} {s_str} {spread:>9.1f}%")

    print(f"\n  ↑ If spread is LARGE (>5%), regime-specific models make sense")
    print(f"    If spread is SMALL (<2%), universal model is fine")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('output/diagnostics/regime_backtest_results.csv', index=False)
    print(f"\n  Results saved to regime_backtest_results.csv")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} min")
    print(f"\n{'='*60}")
    print(f"  ⚡ BROLY REGIME BACKTEST COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
