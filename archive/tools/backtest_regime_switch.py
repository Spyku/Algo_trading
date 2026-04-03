"""
Backtest adaptive horizon switching: 6h in bull, 8h in bear.
Tests different regime detection methods.
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime

from crypto_trading_system_doohan import (
    generate_signals, simulate_portfolio, load_data,
    TRADING_FEE, _suppress_stderr
)

PROD_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'models', 'crypto_doohan_v1_7_1_production.csv')
df_models = pd.read_csv(PROD_CSV)

ASSET = 'BTC'
REPLAY = 1440  # ~2 months (60 days * 24h)
CONF = 90

# --- Generate signals for both horizons ---
def get_signals(horizon):
    rows = df_models[(df_models['coin'] == ASSET) & (df_models['horizon'] == horizon)]
    row = rows.iloc[0]
    feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
    gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0
    with _suppress_stderr():
        sigs = generate_signals(ASSET, row['models'].split('+'),
                                int(row['best_window']), REPLAY,
                                feature_override=feats, horizon=horizon, gamma=gamma)
        sigs = simulate_portfolio(sigs)
    result = {}
    for s in sigs:
        dt = s['datetime']
        if isinstance(dt, str):
            dt = datetime.strptime(dt, '%Y-%m-%d %H:%M')
            s['datetime'] = dt
        result[dt] = s
    return result

print("Generating 6h signals...")
sigs_6h = get_signals(6)
print(f"  -> {len(sigs_6h)} signals")
print("Generating 8h signals...")
sigs_8h = get_signals(8)
print(f"  -> {len(sigs_8h)} signals")

# --- Load price data for regime indicators ---
df_raw = load_data(ASSET)
price_map = {}
for _, row in df_raw.iterrows():
    dt = pd.to_datetime(row['datetime'])
    price_map[dt] = row['close']

# Build merged timeline
all_dts = sorted(set(sigs_6h.keys()) | set(sigs_8h.keys()))

# --- Regime detectors ---
def get_sma(dt, period):
    """Get SMA at a given datetime."""
    idx = df_raw[df_raw['datetime'] <= dt.strftime('%Y-%m-%d %H:%M')].index
    if len(idx) < period:
        return None
    prices = df_raw.loc[idx[-period:], 'close'].values
    return np.mean(prices)

def get_logret(dt, period):
    """Get log return over period hours."""
    idx = df_raw[df_raw['datetime'] <= dt.strftime('%Y-%m-%d %H:%M')].index
    if len(idx) < period + 1:
        return 0
    p_now = df_raw.loc[idx[-1], 'close']
    p_prev = df_raw.loc[idx[-period - 1], 'close']
    return np.log(p_now / p_prev)

def get_volatility(dt, period):
    """Get rolling volatility."""
    idx = df_raw[df_raw['datetime'] <= dt.strftime('%Y-%m-%d %H:%M')].index
    if len(idx) < period + 1:
        return 0
    prices = df_raw.loc[idx[-period-1:], 'close'].values
    returns = np.diff(np.log(prices))
    return np.std(returns)

# Pre-compute regime indicators for all timestamps
print("\nPre-computing regime indicators...")
regime_cache = {}
for dt in all_dts:
    s = sigs_6h.get(dt) or sigs_8h.get(dt)
    price = s['close']
    regime_cache[dt] = {
        'price': price,
        'sma_24': get_sma(dt, 24),
        'sma_48': get_sma(dt, 48),
        'sma_72': get_sma(dt, 72),
        'sma_100': get_sma(dt, 100),
        'sma_200': get_sma(dt, 200),
        'logret_12': get_logret(dt, 12),
        'logret_24': get_logret(dt, 24),
        'logret_48': get_logret(dt, 48),
        'logret_72': get_logret(dt, 72),
        'vol_24': get_volatility(dt, 24),
        'vol_48': get_volatility(dt, 48),
    }
print(f"  Cached {len(regime_cache)} timestamps")

# Median vol for vol-based regimes
vol_48_values = [v['vol_48'] for v in regime_cache.values() if v['vol_48'] > 0]
vol_48_median = np.median(vol_48_values) if vol_48_values else 0.01

# --- Regime definitions ---
REGIMES = {
    'price>sma24':   lambda dt: regime_cache[dt]['price'] > regime_cache[dt]['sma_24'] if regime_cache[dt]['sma_24'] else True,
    'price>sma48':   lambda dt: regime_cache[dt]['price'] > regime_cache[dt]['sma_48'] if regime_cache[dt]['sma_48'] else True,
    'price>sma72':   lambda dt: regime_cache[dt]['price'] > regime_cache[dt]['sma_72'] if regime_cache[dt]['sma_72'] else True,
    'price>sma100':  lambda dt: regime_cache[dt]['price'] > regime_cache[dt]['sma_100'] if regime_cache[dt]['sma_100'] else True,
    'price>sma200':  lambda dt: regime_cache[dt]['price'] > regime_cache[dt]['sma_200'] if regime_cache[dt]['sma_200'] else True,
    'logret12>0':    lambda dt: regime_cache[dt]['logret_12'] > 0,
    'logret24>0':    lambda dt: regime_cache[dt]['logret_24'] > 0,
    'logret48>0':    lambda dt: regime_cache[dt]['logret_48'] > 0,
    'logret72>0':    lambda dt: regime_cache[dt]['logret_72'] > 0,
    'sma24>sma72':   lambda dt: (regime_cache[dt]['sma_24'] or 0) > (regime_cache[dt]['sma_72'] or 0),
    'sma24>sma100':  lambda dt: (regime_cache[dt]['sma_24'] or 0) > (regime_cache[dt]['sma_100'] or 0),
    'sma48>sma200':  lambda dt: (regime_cache[dt]['sma_48'] or 0) > (regime_cache[dt]['sma_200'] or 0),
    'low_vol':       lambda dt: regime_cache[dt]['vol_48'] < vol_48_median,
}


def simulate(horizon_picker, label=""):
    """Simulate trading with dynamic horizon selection."""
    cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
    trades, wins = 0, 0
    trade_log = []
    equity_curve = []
    first_price = None
    h_switches = 0
    last_h = None

    for dt in all_dts:
        h = horizon_picker(dt)
        if last_h is not None and h != last_h:
            h_switches += 1
        last_h = h

        sigs = sigs_6h if h == 6 else sigs_8h
        s = sigs.get(dt)
        if s is None:
            # Use other horizon's signal for price tracking
            s_other = (sigs_8h if h == 6 else sigs_6h).get(dt)
            if s_other:
                price = s_other['close']
                if first_price is None:
                    first_price = price
                eq = cash + held * price if in_pos else cash
                equity_curve.append((dt, eq, 1000.0 * price / first_price, h))
            continue

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
            trade_log.append((dt, 'BUY', price, h))
        elif sig == 'SELL' and in_pos:
            cash = held * price * (1 - TRADING_FEE)
            win = price > entry_px
            if win:
                wins += 1
            trade_log.append((dt, 'SELL', price, h, 'WIN' if win else 'LOSS'))
            held = 0
            in_pos = False

        eq = cash + held * price if in_pos else cash
        bh = 1000.0 * price / first_price
        equity_curve.append((dt, eq, bh, h))

    # Close open position
    if in_pos and all_dts:
        last_s = sigs_6h.get(all_dts[-1]) or sigs_8h.get(all_dts[-1])
        if last_s:
            last_price = last_s['close']
            cash = held * last_price * (1 - TRADING_FEE)
            if last_price > entry_px:
                wins += 1
            trade_log.append((all_dts[-1], 'CLOSE', last_price, last_h, 'WIN' if last_price > entry_px else 'LOSS'))

    final_eq = cash
    ret = (final_eq / 1000.0 - 1) * 100
    wr = (wins / trades * 100) if trades > 0 else 0
    bh_ret = (equity_curve[-1][2] / 1000.0 - 1) * 100 if equity_curve else 0

    return {
        'return': ret, 'trades': trades, 'win_rate': wr,
        'bh': bh_ret, 'trade_log': trade_log,
        'curve': equity_curve, 'switches': h_switches,
        'in_pos': in_pos
    }


# --- Run baselines ---
print("\n" + "="*70)
print(f"  BTC: Adaptive Horizon (6h bull / 8h bear) — conf>={CONF}%, replay={REPLAY}h")
print("="*70)

res_6h = simulate(lambda dt: 6, "6h_only")
res_8h = simulate(lambda dt: 8, "8h_only")

print(f"\n  BASELINES:")
print(f"  {'Strategy':<20} {'Return':>8} {'B&H':>8} {'Alpha':>8} {'Trades':>7} {'WR':>6} {'Switches':>9}")
print(f"  {'-'*68}")
print(f"  {'6h_only':<20} {res_6h['return']:>+7.2f}% {res_6h['bh']:>+7.2f}% {res_6h['return']-res_6h['bh']:>+7.2f}% {res_6h['trades']:>7d} {res_6h['win_rate']:>5.0f}% {0:>9d}")
print(f"  {'8h_only':<20} {res_8h['return']:>+7.2f}% {res_8h['bh']:>+7.2f}% {res_8h['return']-res_8h['bh']:>+7.2f}% {res_8h['trades']:>7d} {res_8h['win_rate']:>5.0f}% {0:>9d}")

# --- Run regime strategies ---
print(f"\n  ADAPTIVE (bull=6h, bear=8h):")
print(f"  {'Regime':<20} {'Return':>8} {'B&H':>8} {'Alpha':>8} {'Trades':>7} {'WR':>6} {'Switches':>9}")
print(f"  {'-'*68}")

results = []
for name, is_bull in REGIMES.items():
    res = simulate(lambda dt, f=is_bull: 6 if f(dt) else 8, name)
    alpha = res['return'] - res['bh']
    results.append((name, res))
    print(f"  {name:<20} {res['return']:>+7.2f}% {res['bh']:>+7.2f}% {alpha:>+7.2f}% {res['trades']:>7d} {res['win_rate']:>5.0f}% {res['switches']:>9d}")

# Sort by return
results.sort(key=lambda x: -x[1]['return'])
best_name, best_res = results[0]
print(f"\n  >>> BEST ADAPTIVE: {best_name} -> {best_res['return']:+.2f}%, {best_res['trades']} trades, {best_res['win_rate']:.0f}% WR")

# Show trade log for best
print(f"\n  --- Trade log ({best_name}) ---")
for t in best_res['trade_log']:
    dt, action = t[0], t[1]
    price, h = t[2], t[3]
    result = t[4] if len(t) > 4 else ''
    print(f"  {dt.strftime('%m-%d %H:%M')} {action:<5} @${price:>9.0f}  [{h}h] {result}")

# Show regime transitions for best
print(f"\n  --- Regime transitions ({best_name}) ---")
is_bull_fn = REGIMES[best_name]
prev_regime = None
for dt in all_dts:
    bull = is_bull_fn(dt)
    regime = "BULL(6h)" if bull else "BEAR(8h)"
    if regime != prev_regime:
        print(f"  {dt.strftime('%m-%d %H:%M')} -> {regime}")
        prev_regime = regime

# Daily equity for top 3 + baselines
print(f"\n  --- Daily Equity Snapshot ---")
top3 = results[:3]
header = f"  {'Date':>14} {'6h':>7} {'8h':>7}"
for name, _ in top3:
    header += f" {name[:12]:>13}"
header += f" {'B&H':>8}"
print(header)
print(f"  {'-'*(16 + 16 + 14*len(top3) + 10)}")

ref_curve = res_6h['curve']
seen = set()
for i, (dt, eq, bh, _) in enumerate(ref_curve):
    dk = dt.strftime('%m-%d')
    if dt.hour == 0 and dk not in seen:
        seen.add(dk)
        line = f"  {dt.strftime('%m-%d %H:%M'):>14}"
        line += f" {(eq/1000-1)*100:>+6.1f}%"
        # 8h
        if i < len(res_8h['curve']):
            line += f" {(res_8h['curve'][i][1]/1000-1)*100:>+6.1f}%"
        for name, res in top3:
            if i < len(res['curve']):
                line += f" {(res['curve'][i][1]/1000-1)*100:>+12.1f}%"
        line += f" {(bh/1000-1)*100:>+7.1f}%"
        print(line)

# Final
if ref_curve:
    dt, eq, bh, _ = ref_curve[-1]
    line = f"  {dt.strftime('%m-%d %H:%M'):>14}"
    line += f" {(eq/1000-1)*100:>+6.1f}%"
    if len(res_8h['curve']) > 0:
        line += f" {(res_8h['curve'][-1][1]/1000-1)*100:>+6.1f}%"
    for name, res in top3:
        if res['curve']:
            line += f" {(res['curve'][-1][1]/1000-1)*100:>+12.1f}%"
    line += f" {(bh/1000-1)*100:>+7.1f}%"
    print(f"  {'-'*(16 + 16 + 14*len(top3) + 10)}")
    print(line + "  <- FINAL")
