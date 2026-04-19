"""
Test #4: Signal Staleness — skip BUY when model has said BUY for N consecutive hours.

Idea: A fresh BUY (first hour) is stronger than a stale BUY (5th consecutive hour).
Tests: baseline (no filter) vs skip after 2/3/4/5 consecutive BUYs.

Usage: python tools/test_signal_staleness.py
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
REPLAY = 1440  # 2 months
PROD_CSV = 'models/crypto_ed_production.csv'


def get_signals(horizon):
    df_models = pd.read_csv(PROD_CSV)
    rows = df_models[(df_models['coin'] == ASSET) & (df_models['horizon'] == horizon)]
    if len(rows) == 0:
        return []
    row = rows.iloc[0]
    feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
    gamma = float(row.get('gamma', 1.0))
    with _suppress_stderr():
        sigs = generate_signals(ASSET, row['models'].split('+'),
                                int(row['best_window']), REPLAY,
                                feature_override=feats, horizon=horizon, gamma=gamma)
        return simulate_portfolio(sigs)


def simulate_with_staleness(signals, conf_threshold, max_consecutive_buy):
    """Simulate with staleness filter. max_consecutive_buy=0 means no filter (baseline)."""
    cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
    trades, wins = 0, 0
    consecutive_buy = 0
    skipped_stale = 0
    first_price = last_price = None

    for s in signals:
        price = s['close']
        sig = s['signal']
        conf = s['confidence']
        if first_price is None:
            first_price = price
        last_price = price

        # Track consecutive BUY signals
        if sig == 'BUY':
            consecutive_buy += 1
        else:
            consecutive_buy = 0

        # Staleness filter: skip if BUY has been repeating too long
        if sig == 'BUY' and max_consecutive_buy > 0 and consecutive_buy > max_consecutive_buy:
            skipped_stale += 1
            continue

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
            consecutive_buy = 0  # reset on sell

    if in_pos and last_price:
        cash = held * last_price * (1 - BACKTEST_FEE_PER_LEG)
        if last_price > entry_px:
            wins += 1

    ret = (cash / 1000.0 - 1) * 100
    wr = (wins / trades * 100) if trades > 0 else 0
    bh = (last_price / first_price - 1) * 100 if first_price and last_price else 0
    return ret, trades, wr, bh, skipped_stale


def main():
    print("=" * 70)
    print("  TEST #4: SIGNAL STALENESS")
    print(f"  {ASSET} | 2-month replay | conf>=85% (bull) / conf>=80% (bear)")
    print("=" * 70)

    for h in [6, 8]:
        conf = 85 if h == 6 else 80  # match current prod config
        print(f"\n  Generating {h}h signals...")
        sigs = get_signals(h)
        print(f"  {len(sigs)} signals")

        # Count consecutive BUY streaks in the data
        streaks = []
        current_streak = 0
        for s in sigs:
            if s['signal'] == 'BUY':
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            streaks.append(current_streak)

        if streaks:
            print(f"  BUY streaks: max={max(streaks)}, avg={np.mean(streaks):.1f}, "
                  f"median={np.median(streaks):.0f}, >3h={sum(1 for s in streaks if s > 3)}, "
                  f">5h={sum(1 for s in streaks if s > 5)}")

        print(f"\n  ETH {h}h @ conf>={conf}%:")
        print(f"  {'Filter':<25} {'Return':>8} {'Trades':>7} {'WR':>6} {'Skipped':>8} {'B&H':>8}")
        print(f"  {'-'*65}")

        for max_consec in [0, 1, 2, 3, 4, 5, 8]:
            label = "No filter (baseline)" if max_consec == 0 else f"Skip after {max_consec} consec"
            ret, tr, wr, bh, skipped = simulate_with_staleness(sigs, conf, max_consec)
            marker = " <-- baseline" if max_consec == 0 else ""
            print(f"  {label:<25} {ret:>+7.2f}% {tr:>7d} {wr:>5.0f}% {skipped:>8d} {bh:>+7.2f}%{marker}")


if __name__ == '__main__':
    main()
