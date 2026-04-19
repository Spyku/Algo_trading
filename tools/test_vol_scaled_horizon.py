"""
Test #5: Volatility-Scaled Horizons — instead of fixed 6h bull / 8h bear,
pick horizon dynamically based on current realized volatility.

High vol → shorter horizon (faster signal, less exposure)
Low vol → longer horizon (more confirmation needed)

Uses existing production models for 5h/6h/7h/8h, just changes which one
gets used per candle based on vol regime.

Usage: python tools/test_vol_scaled_horizon.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
from crypto_trading_system_ed import (
    generate_signals, simulate_portfolio, load_data,
    _suppress_stderr, BACKTEST_FEE_PER_LEG,
)

ASSET = 'ETH'
REPLAY = 1440
PROD_CSV = 'models/crypto_ed_production.csv'


def main():
    print("=" * 70)
    print("  TEST #5: VOLATILITY-SCALED HORIZONS")
    print(f"  {ASSET} | 2-month replay")
    print("=" * 70)

    df_models = pd.read_csv(PROD_CSV)

    # Generate signals for all horizons
    sigs = {}
    for h in [5, 6, 7, 8]:
        rows = df_models[(df_models['coin'] == ASSET) & (df_models['horizon'] == h)]
        if len(rows) == 0:
            print(f"  No model for {h}h — skipping")
            continue
        row = rows.iloc[0]
        feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
        gamma = float(row.get('gamma', 1.0))
        print(f"  Generating {h}h signals...")
        with _suppress_stderr():
            raw = generate_signals(ASSET, row['models'].split('+'),
                                   int(row['best_window']), REPLAY,
                                   feature_override=feats, horizon=h, gamma=gamma)
            raw = simulate_portfolio(raw)
        store = {}
        for s in raw:
            dt = s['datetime']
            if isinstance(dt, str):
                dt = datetime.strptime(dt, '%Y-%m-%d %H:%M')
                s['datetime'] = dt
            store[dt] = s
        sigs[h] = store
        print(f"    {len(store)} signals")

    # Build vol indicator
    df = load_data(ASSET)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df_idx = df.set_index('datetime').sort_index()
    df_idx['logret'] = np.log(df_idx['close'] / df_idx['close'].shift(1))
    df_idx['vol_24h'] = df_idx['logret'].rolling(24).std()
    df_idx['vol_48h'] = df_idx['logret'].rolling(48).std()
    # Percentile rank of current vol vs last 30 days
    df_idx['vol_pctile'] = df_idx['vol_24h'].rolling(720, min_periods=100).rank(pct=True)

    all_dts = sorted(set().union(*[s.keys() for s in sigs.values()]))

    def simulate(horizon_picker, conf_threshold, label):
        cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
        trades, wins = 0, 0
        first_price = last_price = None

        for dt in all_dts:
            if dt not in df_idx.index:
                continue
            h = horizon_picker(dt)
            s = sigs.get(h, {}).get(dt)
            if s is None:
                continue
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

    # Baselines
    print(f"\n  {'Strategy':<40} {'Return':>8} {'Trades':>7} {'WR':>6} {'B&H':>8}")
    print(f"  {'-'*72}")

    for h in [5, 6, 7, 8]:
        if h not in sigs:
            continue
        for conf in [85, 90]:
            ret, tr, wr, bh = simulate(lambda dt, hh=h: hh, conf, f"{h}h @{conf}%")
            print(f"  {f'{h}h only @{conf}%':<40} {ret:>+7.2f}% {tr:>7d} {wr:>5.0f}% {bh:>+7.2f}%")

    # Fixed regime (current prod)
    print(f"\n  --- Regime switching (current) ---")
    # tsmom_672h detector
    df_idx['tsmom'] = np.log(df_idx['close'] / df_idx['close'].shift(672))

    def current_regime(dt):
        if dt in df_idx.index and df_idx.loc[dt, 'tsmom'] > 0:
            return 6  # bull
        return 8  # bear

    ret, tr, wr, bh = simulate(current_regime, 90, "tsmom bull=6h/bear=8h @90%")
    print(f"  {'tsmom_672h bull=6h bear=8h @90%':<40} {ret:>+7.2f}% {tr:>7d} {wr:>5.0f}% {bh:>+7.2f}%")

    # Vol-scaled strategies
    print(f"\n  --- Volatility-scaled horizons ---")

    strategies = [
        ("vol_2band: low→8h high→5h", lambda dt:
            5 if dt in df_idx.index and df_idx.loc[dt, 'vol_pctile'] > 0.7 else 8),
        ("vol_2band: low→8h high→6h", lambda dt:
            6 if dt in df_idx.index and df_idx.loc[dt, 'vol_pctile'] > 0.7 else 8),
        ("vol_3band: low→8h mid→6h high→5h", lambda dt:
            5 if dt in df_idx.index and df_idx.loc[dt, 'vol_pctile'] > 0.8 else
            6 if dt in df_idx.index and df_idx.loc[dt, 'vol_pctile'] > 0.4 else 8),
        ("vol_3band: low→7h mid→6h high→5h", lambda dt:
            5 if dt in df_idx.index and df_idx.loc[dt, 'vol_pctile'] > 0.8 else
            6 if dt in df_idx.index and df_idx.loc[dt, 'vol_pctile'] > 0.4 else 7),
        ("vol_median: below→8h above→6h", lambda dt:
            6 if dt in df_idx.index and df_idx.loc[dt, 'vol_pctile'] > 0.5 else 8),
        ("vol_median: below→7h above→5h", lambda dt:
            5 if dt in df_idx.index and df_idx.loc[dt, 'vol_pctile'] > 0.5 else 7),
        # Combine tsmom regime + vol scaling
        ("tsmom+vol: bull→6h/5h bear→8h/7h", lambda dt:
            (5 if dt in df_idx.index and df_idx.loc[dt, 'vol_pctile'] > 0.7 else 6)
            if dt in df_idx.index and df_idx.loc[dt, 'tsmom'] > 0 else
            (7 if dt in df_idx.index and df_idx.loc[dt, 'vol_pctile'] > 0.7 else 8)),
        ("tsmom+vol: bull→5h/6h bear→7h/8h", lambda dt:
            (5 if dt in df_idx.index and df_idx.loc[dt, 'vol_pctile'] > 0.6 else 6)
            if dt in df_idx.index and df_idx.loc[dt, 'tsmom'] > 0 else
            (7 if dt in df_idx.index and df_idx.loc[dt, 'vol_pctile'] > 0.6 else 8)),
    ]

    for conf in [85, 90]:
        print(f"\n  conf >= {conf}%:")
        for label, picker in strategies:
            ret, tr, wr, bh = simulate(picker, conf, label)
            print(f"  {label:<40} {ret:>+7.2f}% {tr:>7d} {wr:>5.0f}% {bh:>+7.2f}%")


if __name__ == '__main__':
    main()
