"""
Backtest: 1% Take Profit — hourly close vs hourly high (proxy for 5-min check)
Generates equity curve chart comparing:
  1. No TP (signal exit only)
  2. TP checked at hourly CLOSE (conservative — misses intra-hour spikes)
  3. TP checked at hourly HIGH (equivalent to 5-min check — catches any intra-hour spike)
  4. Buy & Hold
"""
import sys, os, csv, warnings
from datetime import datetime, timedelta

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)
os.chdir(ENGINE_DIR)
warnings.filterwarnings('ignore')

import numpy as np

TRADING_FEE = 0.0005  # 5 bps/leg realistic maker blend (see ed.py BACKTEST_FEE_PER_LEG)
REPLAY_HOURS = 720  # 1 month
TP_PCT = 1.0

CONFIGS = {
    'BTC': {'horizon': 8, 'min_confidence': 90, 'position_usd': 12000},
}


def load_production_config(asset, horizon):
    csv_path = os.path.join(ENGINE_DIR, 'models', 'crypto_ed_production.csv')
    best = None
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['coin'] == asset and int(row['horizon']) == horizon:
                score = float(row.get('combined_score', 0))
                if best is None or score > best.get('_score', 0):
                    best = dict(row)
                    best['_score'] = score
    return best


def load_ohlcv(asset):
    path = os.path.join(ENGINE_DIR, 'data', f'{asset}_hourly_data.csv')
    ohlcv = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = row['datetime'][:16]
            ohlcv[dt] = {
                'open': float(row['open']), 'high': float(row['high']),
                'low': float(row['low']), 'close': float(row['close']),
            }
    return ohlcv


def simulate(signals, ohlcv, min_conf, tp_mode='none'):
    """
    tp_mode: 'none' (signal exit only), 'close' (check TP at hourly close), 'high' (check TP at hourly high)
    Returns: list of (datetime, equity) tuples for charting
    """
    cash = 1000.0
    held = 0.0
    in_pos = False
    entry_px = 0.0
    trades = 0
    wins = 0
    tp_hits = 0

    equity_curve = []

    for i, sig in enumerate(signals):
        ts = sig['datetime']
        if isinstance(ts, str):
            ts = datetime.strptime(ts, '%Y-%m-%d %H:%M')

        price = sig['close']
        h_str = ts.strftime('%Y-%m-%d %H:%M')
        candle = ohlcv.get(h_str, {})
        candle_high = candle.get('high', price)
        candle_close = candle.get('close', price)

        # Track equity
        eq = cash + held * price if in_pos else cash
        equity_curve.append((ts, eq))

        if in_pos:
            # Check TP
            tp_target = entry_px * (1 + TP_PCT / 100)
            tp_triggered = False

            if tp_mode == 'high' and candle_high >= tp_target:
                # TP hit during this candle (5-min equivalent)
                exit_price = tp_target
                tp_triggered = True
            elif tp_mode == 'close' and candle_close >= tp_target:
                # TP hit at candle close
                exit_price = candle_close
                tp_triggered = True

            if tp_triggered:
                gross = (exit_price - entry_px) / entry_px
                cash = held * exit_price * (1 - TRADING_FEE)
                held = 0
                in_pos = False
                trades += 1
                tp_hits += 1
                if gross > 0:
                    wins += 1
                continue

            # Signal exit
            if sig['signal'] == 'SELL':
                cash = held * price * (1 - TRADING_FEE)
                if price > entry_px:
                    wins += 1
                held = 0
                in_pos = False
                trades += 1

        else:
            # Entry
            if sig['signal'] == 'BUY' and sig['confidence'] >= min_conf:
                held = cash * (1 - TRADING_FEE) / price
                cash = 0
                in_pos = True
                entry_px = price
                trades += 1

    # Close open position at end
    if in_pos:
        last_price = signals[-1]['close']
        cash = held * last_price * (1 - TRADING_FEE)
        held = 0

    final_eq = cash
    ret = (final_eq / 1000 - 1) * 100
    wr = wins / max(1, trades // 2) * 100

    return equity_curve, ret, trades, wr, tp_hits


def make_bh_curve(signals, ohlcv):
    """Buy & hold equity curve."""
    first_price = signals[0]['close']
    curve = []
    for sig in signals:
        ts = sig['datetime']
        if isinstance(ts, str):
            ts = datetime.strptime(ts, '%Y-%m-%d %H:%M')
        price = sig['close']
        eq = 1000 * (price / first_price)
        curve.append((ts, eq))
    return curve


# ── Main ──
from crypto_trading_system_ed import generate_signals

asset = 'BTC'
cfg = CONFIGS[asset]

prod = load_production_config(asset, cfg['horizon'])
if not prod:
    print(f"No production model for {asset} {cfg['horizon']}h")
    sys.exit(1)

models = prod['models'].split('+')
window = int(prod['best_window'])
gamma = float(prod['gamma'])
features = prod['optimal_features'].split(',')

print(f"Generating {cfg['horizon']}h signals for {asset} ({REPLAY_HOURS}h = 1 month)...")
sigs = generate_signals(asset, models, window, replay_hours=REPLAY_HOURS,
                        feature_override=features, horizon=cfg['horizon'], gamma=gamma)
print(f"  {len(sigs)} signals generated")

ohlcv = load_ohlcv(asset)

# Run simulations
print(f"\nSimulating...")
curve_none, ret_none, tr_none, wr_none, _ = simulate(sigs, ohlcv, cfg['min_confidence'], tp_mode='none')
curve_close, ret_close, tr_close, wr_close, tp_close = simulate(sigs, ohlcv, cfg['min_confidence'], tp_mode='close')
curve_high, ret_high, tr_high, wr_high, tp_high = simulate(sigs, ohlcv, cfg['min_confidence'], tp_mode='high')
curve_bh = make_bh_curve(sigs, ohlcv)

print(f"\n  {'Strategy':<30s}  {'Return':>8s}  {'Trades':>7s}  {'WR':>5s}  {'TP hits':>8s}")
print(f"  {'-'*30}  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*8}")
print(f"  {'No TP (signal exit)':<30s}  {ret_none:>+7.2f}%  {tr_none:>7d}  {wr_none:>4.0f}%  {'N/A':>8s}")
print(f"  {'TP 1% (hourly CLOSE)':<30s}  {ret_close:>+7.2f}%  {tr_close:>7d}  {wr_close:>4.0f}%  {tp_close:>8d}")
print(f"  {'TP 1% (hourly HIGH = 5min)':<30s}  {ret_high:>+7.2f}%  {tr_high:>7d}  {wr_high:>4.0f}%  {tp_high:>8d}")
bh_ret = (curve_bh[-1][1] / 1000 - 1) * 100
print(f"  {'Buy & Hold':<30s}  {bh_ret:>+7.2f}%  {'N/A':>7s}  {'N/A':>5s}  {'N/A':>8s}")

# ── Generate chart ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(14, 7))

# Plot equity curves
dates_none = [d for d, _ in curve_none]
eq_none = [e for _, e in curve_none]
dates_close = [d for d, _ in curve_close]
eq_close = [e for _, e in curve_close]
dates_high = [d for d, _ in curve_high]
eq_high = [e for _, e in curve_high]
dates_bh = [d for d, _ in curve_bh]
eq_bh = [e for _, e in curve_bh]

ax.plot(dates_bh, eq_bh, color='gray', linewidth=1, alpha=0.6, label=f'Buy & Hold ({bh_ret:+.1f}%)')
ax.plot(dates_none, eq_none, color='blue', linewidth=1.5, label=f'No TP - signal exit ({ret_none:+.1f}%)')
ax.plot(dates_close, eq_close, color='orange', linewidth=1.5, label=f'TP 1% hourly CLOSE ({ret_close:+.1f}%, {tp_close} TP hits)')
ax.plot(dates_high, eq_high, color='green', linewidth=2, label=f'TP 1% hourly HIGH ~5min ({ret_high:+.1f}%, {tp_high} TP hits)')

ax.axhline(y=1000, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Equity ($)')
ax.set_title(f'BTC Take Profit 1% Comparison — Last Month ({REPLAY_HOURS}h)\n'
             f'{cfg["horizon"]}h horizon, conf>={cfg["min_confidence"]}%')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()

chart_path = os.path.join(ENGINE_DIR, 'charts', 'tp_comparison_1month.png')
os.makedirs(os.path.dirname(chart_path), exist_ok=True)
plt.savefig(chart_path, dpi=150)
print(f"\nChart saved: {chart_path}")
plt.close()
