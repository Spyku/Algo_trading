"""
Compare all BTC horizons side by side over the same backtest window.
Shows return curve progression for each horizon.
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime

from crypto_trading_system_doohan import (
    generate_signals, simulate_portfolio, TRADING_FEE, _suppress_stderr
)

PROD_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'models', 'crypto_doohan_v1_7_1_production.csv')
df_models = pd.read_csv(PROD_CSV)

ASSET = 'BTC'
REPLAY = 200
CONF = 90
HORIZONS = [5, 6, 7, 8, 10, 12, 14]  # skip 16 (dead)

results = {}

for h in HORIZONS:
    rows = df_models[(df_models['coin'] == ASSET) & (df_models['horizon'] == h)]
    if len(rows) == 0:
        print(f"  No model for {h}h, skipping")
        continue
    row = rows.iloc[0]
    feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
    gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0

    with _suppress_stderr():
        sigs = generate_signals(ASSET, row['models'].split('+'),
                                int(row['best_window']), REPLAY,
                                feature_override=feats, horizon=h, gamma=gamma)
        sigs = simulate_portfolio(sigs)

    # Simulate with conf threshold
    cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
    trades, wins = 0, 0
    equity_curve = []
    trade_log = []
    first_price = None

    for s in sigs:
        dt = s['datetime']
        if isinstance(dt, str):
            dt = datetime.strptime(dt, '%Y-%m-%d %H:%M')
        price = s['close']
        sig = s['signal']
        conf = s['confidence']

        if first_price is None:
            first_price = price

        if sig == 'BUY' and conf >= CONF and not in_pos:
            held = cash * (1 - TRADING_FEE) / price
            cash = 0
            in_pos = True
            entry_px = price
            trades += 1
            trade_log.append(f"  {dt.strftime('%m-%d %H:%M')} BUY  @${price:.0f}")
        elif sig == 'SELL' and in_pos:
            cash = held * price * (1 - TRADING_FEE)
            pnl = "WIN" if price > entry_px else "LOSS"
            if price > entry_px:
                wins += 1
            trade_log.append(f"  {dt.strftime('%m-%d %H:%M')} SELL @${price:.0f} {pnl}")
            held = 0
            in_pos = False

        # Track equity
        eq = cash + held * price if in_pos else cash
        bh = 1000.0 * price / first_price
        equity_curve.append((dt, eq, bh))

    # Close open position
    if in_pos and sigs:
        last_price = sigs[-1]['close']
        cash = held * last_price * (1 - TRADING_FEE)
        pnl = "WIN" if last_price > entry_px else "LOSS"
        if last_price > entry_px:
            wins += 1
        trade_log.append(f"  (open) CLOSE @${last_price:.0f} {pnl}")

    final_eq = cash if not in_pos else cash
    ret = (final_eq / 1000.0 - 1) * 100
    wr = (wins / trades * 100) if trades > 0 else 0
    bh_ret = (equity_curve[-1][2] / 1000.0 - 1) * 100 if equity_curve else 0

    results[h] = {
        'return': ret, 'trades': trades, 'win_rate': wr,
        'bh': bh_ret, 'curve': equity_curve, 'trade_log': trade_log,
        'in_pos': in_pos
    }

# Print summary
print(f"\n{'='*70}")
print(f"  BTC: All Horizons Compared (conf>={CONF}%, replay={REPLAY}h)")
print(f"{'='*70}")
print(f"  {'Horizon':>8} {'Return':>8} {'B&H':>8} {'Alpha':>8} {'Trades':>7} {'WR':>6} {'Status':>10}")
print(f"  {'-'*62}")

for h in HORIZONS:
    if h not in results:
        continue
    r = results[h]
    status = "OPEN" if r['in_pos'] else "CASH"
    alpha = r['return'] - r['bh']
    print(f"  {h:>6}h {r['return']:>+7.2f}% {r['bh']:>+7.2f}% {alpha:>+7.2f}% {r['trades']:>7d} {r['win_rate']:>5.0f}% {status:>10}")

# Print trade logs for each
for h in HORIZONS:
    if h not in results:
        continue
    r = results[h]
    if r['trade_log']:
        print(f"\n  --- {h}h trades ---")
        for t in r['trade_log']:
            print(t)

# Print daily equity snapshot
print(f"\n  --- Daily Equity Snapshot (starting $1000) ---")
# Sample every 24h
all_curves = {h: results[h]['curve'] for h in HORIZONS if h in results}
ref_h = HORIZONS[0]
ref_curve = all_curves.get(ref_h, [])

header = f"  {'Date':>14}"
for h in HORIZONS:
    if h in results:
        header += f" {h:>6}h"
header += f" {'B&H':>8}"
print(header)
print(f"  {'-'*(16 + 8*len([h for h in HORIZONS if h in results]) + 10)}")

# Sample at 00:00 each day
seen_dates = set()
for i, (dt, eq, bh) in enumerate(ref_curve):
    date_key = dt.strftime('%m-%d')
    if dt.hour == 0 and date_key not in seen_dates:
        seen_dates.add(date_key)
        line = f"  {dt.strftime('%m-%d %H:%M'):>14}"
        for h in HORIZONS:
            if h in results:
                curve = results[h]['curve']
                if i < len(curve):
                    eq_h = curve[i][1]
                    ret_h = (eq_h / 1000.0 - 1) * 100
                    line += f" {ret_h:>+6.1f}%"
                else:
                    line += f" {'N/A':>7}"
        bh_ret = (bh / 1000.0 - 1) * 100
        line += f" {bh_ret:>+7.1f}%"
        print(line)

# Final line
if ref_curve:
    dt, eq, bh = ref_curve[-1]
    line = f"  {dt.strftime('%m-%d %H:%M'):>14}"
    for h in HORIZONS:
        if h in results:
            curve = results[h]['curve']
            eq_h = curve[-1][1]
            ret_h = (eq_h / 1000.0 - 1) * 100
            line += f" {ret_h:>+6.1f}%"
    bh_ret = (bh / 1000.0 - 1) * 100
    line += f" {bh_ret:>+7.1f}%"
    print(f"  {'-'*(16 + 8*len([h for h in HORIZONS if h in results]) + 10)}")
    print(line + "  <- FINAL")
