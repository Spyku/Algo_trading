"""
Test simple regime detectors for 6h/8h switching.
Focus on low-switch, robust indicators.
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
REPLAY = 1440  # 2 months
CONF = 90

# --- Generate signals ---
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

# --- Load price data ---
df_raw = load_data(ASSET)
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
df_raw = df_raw.set_index('datetime').sort_index()

# Pre-compute rolling indicators on full price series
df_raw['sma24'] = df_raw['close'].rolling(24).mean()
df_raw['sma48'] = df_raw['close'].rolling(48).mean()
df_raw['sma72'] = df_raw['close'].rolling(72).mean()
df_raw['sma100'] = df_raw['close'].rolling(100).mean()
df_raw['sma200'] = df_raw['close'].rolling(200).mean()
df_raw['rsi14'] = None  # compute below
df_raw['high_24'] = df_raw['high'].rolling(24).max()
df_raw['high_48'] = df_raw['high'].rolling(48).max()
df_raw['high_72'] = df_raw['high'].rolling(72).max()
df_raw['low_24'] = df_raw['low'].rolling(24).min()
df_raw['low_48'] = df_raw['low'].rolling(48).min()

# RSI
delta = df_raw['close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df_raw['rsi14'] = 100 - (100 / (1 + rs))

# Consecutive red/green candles
df_raw['is_green'] = (df_raw['close'] > df_raw['close'].shift(1)).astype(int)
df_raw['consec_green'] = 0
df_raw['consec_red'] = 0
cg, cr = 0, 0
for i in range(len(df_raw)):
    if df_raw['is_green'].iloc[i] == 1:
        cg += 1
        cr = 0
    else:
        cr += 1
        cg = 0
    df_raw.iloc[i, df_raw.columns.get_loc('consec_green')] = cg
    df_raw.iloc[i, df_raw.columns.get_loc('consec_red')] = cr

# Drawdown from N-hour high
df_raw['dd_24'] = (df_raw['close'] / df_raw['high_24'] - 1) * 100
df_raw['dd_48'] = (df_raw['close'] / df_raw['high_48'] - 1) * 100
df_raw['dd_72'] = (df_raw['close'] / df_raw['high_72'] - 1) * 100

# Distance from N-hour low (bounce)
df_raw['bounce_24'] = (df_raw['close'] / df_raw['low_24'] - 1) * 100
df_raw['bounce_48'] = (df_raw['close'] / df_raw['low_48'] - 1) * 100

# Build lookup
indicator_map = {}
for dt, row in df_raw.iterrows():
    indicator_map[dt] = row

all_dts = sorted(set(sigs_6h.keys()) | set(sigs_8h.keys()))
print(f"\nTimeline: {len(all_dts)} candles")

# --- Regime definitions ---
def make_regime(name, fn):
    return (name, fn)

REGIMES = [
    # Baselines from v1
    make_regime('sma24>sma100',    lambda dt: indicator_map[dt]['sma24'] > indicator_map[dt]['sma100'] if dt in indicator_map and pd.notna(indicator_map[dt]['sma100']) else True),

    # RSI-based
    make_regime('rsi14>50',        lambda dt: indicator_map[dt]['rsi14'] > 50 if dt in indicator_map and pd.notna(indicator_map[dt]['rsi14']) else True),
    make_regime('rsi14>45',        lambda dt: indicator_map[dt]['rsi14'] > 45 if dt in indicator_map and pd.notna(indicator_map[dt]['rsi14']) else True),
    make_regime('rsi14>40',        lambda dt: indicator_map[dt]['rsi14'] > 40 if dt in indicator_map and pd.notna(indicator_map[dt]['rsi14']) else True),

    # Drawdown from recent high
    make_regime('dd24>-2%',        lambda dt: indicator_map[dt]['dd_24'] > -2 if dt in indicator_map and pd.notna(indicator_map[dt]['dd_24']) else True),
    make_regime('dd48>-3%',        lambda dt: indicator_map[dt]['dd_48'] > -3 if dt in indicator_map and pd.notna(indicator_map[dt]['dd_48']) else True),
    make_regime('dd72>-4%',        lambda dt: indicator_map[dt]['dd_72'] > -4 if dt in indicator_map and pd.notna(indicator_map[dt]['dd_72']) else True),
    make_regime('dd48>-5%',        lambda dt: indicator_map[dt]['dd_48'] > -5 if dt in indicator_map and pd.notna(indicator_map[dt]['dd_48']) else True),

    # Bounce from recent low
    make_regime('bounce48>3%',     lambda dt: indicator_map[dt]['bounce_48'] > 3 if dt in indicator_map and pd.notna(indicator_map[dt]['bounce_48']) else True),
    make_regime('bounce48>2%',     lambda dt: indicator_map[dt]['bounce_48'] > 2 if dt in indicator_map and pd.notna(indicator_map[dt]['bounce_48']) else True),

    # Consecutive candles
    make_regime('not_3red',        lambda dt: indicator_map[dt]['consec_red'] < 3 if dt in indicator_map else True),
    make_regime('not_4red',        lambda dt: indicator_map[dt]['consec_red'] < 4 if dt in indicator_map else True),
    make_regime('not_5red',        lambda dt: indicator_map[dt]['consec_red'] < 5 if dt in indicator_map else True),

    # Price vs SMA (simple)
    make_regime('price>sma48',     lambda dt: indicator_map[dt]['close'] > indicator_map[dt]['sma48'] if dt in indicator_map and pd.notna(indicator_map[dt]['sma48']) else True),
    make_regime('price>sma72',     lambda dt: indicator_map[dt]['close'] > indicator_map[dt]['sma72'] if dt in indicator_map and pd.notna(indicator_map[dt]['sma72']) else True),

    # Combo: drawdown + RSI
    make_regime('dd48>-3%+rsi>45', lambda dt: (indicator_map[dt]['dd_48'] > -3 and indicator_map[dt]['rsi14'] > 45) if dt in indicator_map and pd.notna(indicator_map[dt]['dd_48']) and pd.notna(indicator_map[dt]['rsi14']) else True),
    make_regime('dd72>-4%+rsi>45', lambda dt: (indicator_map[dt]['dd_72'] > -4 and indicator_map[dt]['rsi14'] > 45) if dt in indicator_map and pd.notna(indicator_map[dt]['dd_72']) and pd.notna(indicator_map[dt]['rsi14']) else True),

    # Combo: SMA + drawdown
    make_regime('sma24>100+dd48>-3', lambda dt: (indicator_map[dt]['sma24'] > indicator_map[dt]['sma100'] and indicator_map[dt]['dd_48'] > -3) if dt in indicator_map and pd.notna(indicator_map[dt]['sma100']) and pd.notna(indicator_map[dt]['dd_48']) else True),
]


def simulate(horizon_picker):
    cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
    trades, wins = 0, 0
    h_switches = 0
    last_h = None
    bull_count, bear_count = 0, 0

    for dt in all_dts:
        h = horizon_picker(dt)
        if h == 6:
            bull_count += 1
        else:
            bear_count += 1
        if last_h is not None and h != last_h:
            h_switches += 1
        last_h = h

        sigs = sigs_6h if h == 6 else sigs_8h
        s = sigs.get(dt)
        if s is None:
            continue

        price = s['close']
        sig = s['signal']
        conf = s['confidence']

        if sig == 'BUY' and conf >= CONF and not in_pos:
            held = cash * (1 - TRADING_FEE) / price
            cash = 0
            in_pos = True
            entry_px = price
            trades += 1
        elif sig == 'SELL' and in_pos:
            cash = held * price * (1 - TRADING_FEE)
            if price > entry_px:
                wins += 1
            held = 0
            in_pos = False

    if in_pos and all_dts:
        last_s = sigs_6h.get(all_dts[-1]) or sigs_8h.get(all_dts[-1])
        if last_s:
            cash = held * last_s['close'] * (1 - TRADING_FEE)
            if last_s['close'] > entry_px:
                wins += 1

    ret = (cash / 1000.0 - 1) * 100
    wr = (wins / trades * 100) if trades > 0 else 0
    bull_pct = bull_count / max(1, bull_count + bear_count) * 100
    return ret, trades, wr, h_switches, bull_pct


# --- Run ---
print(f"\n{'='*80}")
print(f"  BTC: Regime Detectors (6h bull / 8h bear) — conf>={CONF}%, replay={REPLAY}h")
print(f"{'='*80}")

# Baselines
ret_6h, tr_6h, wr_6h, _, _ = simulate(lambda dt: 6)
ret_8h, tr_8h, wr_8h, _, _ = simulate(lambda dt: 8)

print(f"\n  BASELINES:")
print(f"  {'6h_only':<25} {ret_6h:>+7.2f}%  {tr_6h:>3d} trades  {wr_6h:>5.1f}% WR")
print(f"  {'8h_only':<25} {ret_8h:>+7.2f}%  {tr_8h:>3d} trades  {wr_8h:>5.1f}% WR")

print(f"\n  {'Regime':<25} {'Return':>8} {'Trades':>7} {'WR':>6} {'Switches':>9} {'%Bull':>7}")
print(f"  {'-'*68}")

results = []
for name, is_bull in REGIMES:
    ret, tr, wr, sw, bp = simulate(lambda dt, f=is_bull: 6 if f(dt) else 8)
    results.append((name, ret, tr, wr, sw, bp))
    print(f"  {name:<25} {ret:>+7.2f}% {tr:>7d} {wr:>5.1f}% {sw:>9d} {bp:>6.0f}%")

results.sort(key=lambda x: -x[1])
print(f"\n  TOP 5:")
for i, (name, ret, tr, wr, sw, bp) in enumerate(results[:5]):
    print(f"  {i+1}. {name:<25} {ret:>+7.2f}%  {tr} trades  {wr:.0f}% WR  {sw} switches  {bp:.0f}% bull")

print(f"\n  WORST 3:")
for name, ret, tr, wr, sw, bp in results[-3:]:
    print(f"     {name:<25} {ret:>+7.2f}%  {tr} trades  {wr:.0f}% WR  {sw} switches  {bp:.0f}% bull")
