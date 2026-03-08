"""
Crypto Confidence Threshold Backtest
======================================
Tests how confidence filtering affects real trading performance.
Uses your actual ML signals from crypto_trading_system.py.

Strategy: $5,000 all-in / all-out
  - BUY when signal=BUY AND confidence >= threshold
  - SELL when signal=SELL (any confidence)
  - HOLD otherwise

Thresholds tested:
  1. All signals (no filter)
  2. Confidence > 60%
  3. Confidence > 70%
  4. Confidence > 80%

Usage:
  python crypto_trading_threshold_system.py                # BTC default
  python crypto_trading_threshold_system.py --asset ETH    # other asset
  python crypto_trading_threshold_system.py --hours 500    # more history
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime

os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')

# Import from main system
from crypto_trading_system import (
    ASSETS, FEATURE_SET_A, FEATURE_SET_B,
    PREDICTION_HORIZON, ALL_MODELS, REPLAY_HOURS,
    download_asset, load_data, _build_features,
    generate_signals, update_all_data,
)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# CONFIG
# ============================================================
STARTING_CAPITAL = 5000
TRADING_FEE = 0.0009  # 0.09% Revolut X taker fee (applied on BUY and SELL)
THRESHOLDS = [0, 60, 70, 80]  # 0 = all signals


# ============================================================
# LOAD BEST MODEL CONFIG
# ============================================================
def load_best_config(asset_name):
    csv_path = 'crypto_hourly_best_models.csv'
    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found! Run Mode A first.")
        return None

    df = pd.read_csv(csv_path)
    match = df[df['coin'] == asset_name]
    if match.empty:
        print(f"  ERROR: No saved model for {asset_name}.")
        return None

    row = match.iloc[0]
    return {
        'coin': row['coin'],
        'models': row['models'],
        'best_combo': row['best_combo'],
        'best_window': int(row['best_window']),
        'accuracy': row['accuracy'],
        'feature_set': row.get('feature_set', 'A'),
        'optimal_features': row.get('optimal_features', ''),
        'horizon': int(row.get('horizon', 4)),
    }


# ============================================================
# THRESHOLD BACKTEST
# ============================================================
def run_threshold_backtest(signals, threshold, capital=STARTING_CAPITAL):
    """
    All-in / all-out backtest with confidence threshold.
    BUY only when signal=BUY and confidence >= threshold.
    SELL on any SELL signal.
    """
    cash = capital
    btc_held = 0
    in_position = False
    buy_price = 0
    buy_conf = 0

    trades = []       # completed BUY→SELL pairs
    open_trades = []   # BUY entries waiting for SELL
    equity_curve = []

    # Track accuracy (correct predictions)
    correct = 0
    total_predictions = 0

    for i, sig in enumerate(signals):
        price = sig['close']
        signal = sig['signal']
        conf = sig['confidence']
        actual = sig.get('actual')

        # Count prediction accuracy for signals above threshold
        if signal == 'BUY' and conf >= threshold and actual is not None:
            predicted_up = True
            actual_up = actual == 'UP'
            if predicted_up == actual_up:
                correct += 1
            total_predictions += 1
        elif signal == 'SELL' and actual is not None:
            predicted_up = False
            actual_up = actual == 'UP'
            if predicted_up == actual_up:
                correct += 1
            total_predictions += 1

        # Trading logic (with 0.09% fee on each BUY and SELL)
        if not in_position and signal == 'BUY' and conf >= threshold:
            btc_held = cash * (1 - TRADING_FEE) / price  # pay fee on BUY
            buy_price = price
            buy_conf = conf
            cash = 0
            in_position = True
            open_trades.append({
                'buy_idx': i,
                'buy_dt': sig['datetime'],
                'buy_price': price,
                'buy_conf': conf,
            })

        elif in_position and signal == 'SELL':
            cash = btc_held * price * (1 - TRADING_FEE)  # pay fee on SELL
            effective_sell = price * (1 - TRADING_FEE)
            effective_buy = buy_price * (1 + TRADING_FEE)
            pnl = (effective_sell - effective_buy) / effective_buy * 100
            trade = open_trades.pop() if open_trades else {}
            trade.update({
                'sell_idx': i,
                'sell_dt': sig['datetime'],
                'sell_price': price,
                'pnl_pct': round(pnl, 2),
                'profit': round(cash - (btc_held * buy_price), 2),
                'duration_h': i - trade.get('buy_idx', i),
            })
            trades.append(trade)
            btc_held = 0
            in_position = False

        # Equity
        current_value = btc_held * price if in_position else cash
        equity_curve.append({
            'idx': i,
            'datetime': sig['datetime'],
            'price': price,
            'value': round(current_value, 2),
            'in_position': in_position,
        })

    # Final value
    # Final value (apply sell fee if still in position)
    final_value = btc_held * signals[-1]['close'] * (1 - TRADING_FEE) if in_position else cash
    start_price = signals[0]['close']
    end_price = signals[-1]['close']
    buy_hold_value = capital * (end_price / start_price)
    buy_hold_return = (end_price / start_price - 1) * 100

    total_return = (final_value / capital - 1) * 100
    alpha = total_return - buy_hold_return

    # Trade stats
    n_trades = len(trades)
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
    avg_duration = np.mean([t['duration_h'] for t in trades]) if trades else 0
    max_trade = max([t['pnl_pct'] for t in trades]) if trades else 0
    min_trade = min([t['pnl_pct'] for t in trades]) if trades else 0

    # Prediction accuracy at this threshold
    pred_accuracy = correct / total_predictions * 100 if total_predictions > 0 else 0

    # Max drawdown
    peak = capital
    max_dd = 0
    for eq in equity_curve:
        if eq['value'] > peak:
            peak = eq['value']
        dd = (peak - eq['value']) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Buy signal count (how many BUY signals pass the threshold)
    buy_signals_total = sum(1 for s in signals if s['signal'] == 'BUY')
    buy_signals_filtered = sum(1 for s in signals if s['signal'] == 'BUY' and s['confidence'] >= threshold)

    return {
        'threshold': threshold,
        'label': 'All Signals' if threshold == 0 else f'>{threshold}%',
        'final_value': round(final_value, 2),
        'return_pct': round(total_return, 2),
        'buy_hold_value': round(buy_hold_value, 2),
        'buy_hold_return': round(buy_hold_return, 2),
        'alpha': round(alpha, 2),
        'n_trades': n_trades,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(win_rate, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'avg_duration_h': round(avg_duration, 1),
        'max_trade': round(max_trade, 2),
        'min_trade': round(min_trade, 2),
        'max_drawdown': round(max_dd, 2),
        'pred_accuracy': round(pred_accuracy, 1),
        'buy_signals_total': buy_signals_total,
        'buy_signals_filtered': buy_signals_filtered,
        'equity_curve': equity_curve,
        'trades': trades,
    }


# ============================================================
# CHART GENERATION (matplotlib)
# ============================================================
def generate_threshold_chart(asset_name, results, signals, config):
    """Generate multi-panel PNG comparing all threshold strategies."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  matplotlib not installed. Skipping chart.")
        return None

    colors = {0: '#6366f1', 60: '#06d6a0', 70: '#fbbf24', 80: '#ef476f'}
    bg_color = '#0b1120'
    text_color = '#e2e8f0'
    grid_color = '#1a2040'

    fig = plt.figure(figsize=(18, 22), facecolor=bg_color)

    # ---- PANEL 1: PRICE + SIGNALS BY CONFIDENCE ----
    ax1 = fig.add_axes([0.06, 0.76, 0.88, 0.18])
    ax1.set_facecolor(bg_color)

    dates = pd.to_datetime([s['datetime'] for s in signals])
    prices = [s['close'] for s in signals]
    ax1.plot(dates, prices, color='#8892b0', linewidth=1.2, alpha=0.9)

    # Color BUY markers by confidence band
    for s, d in zip(signals, dates):
        if s['signal'] == 'BUY':
            c = s['confidence']
            if c >= 80:
                color, size = '#ef476f', 55
            elif c >= 70:
                color, size = '#fbbf24', 45
            elif c >= 60:
                color, size = '#06d6a0', 35
            else:
                color, size = '#6366f1', 25
            ax1.scatter(d, s['close'], color=color, s=size, marker='^',
                       alpha=0.8, zorder=5, edgecolors='none')
        elif s['signal'] == 'SELL':
            ax1.scatter(d, s['close'], color='#ff6b6b', s=40, marker='v',
                       alpha=0.7, zorder=5, edgecolors='none')

    ax1.set_title(f'{asset_name} Price + Signals (colored by confidence)',
                  color=text_color, fontsize=13, fontweight='bold', pad=10)
    ax1.tick_params(colors=text_color, labelsize=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.grid(True, alpha=0.15, color=grid_color)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(grid_color)
    ax1.spines['bottom'].set_color(grid_color)

    # Legend
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#6366f1', markersize=7, label='BUY <60%', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#06d6a0', markersize=8, label='BUY 60-70%', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#fbbf24', markersize=9, label='BUY 70-80%', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#ef476f', markersize=10, label='BUY >80%', linestyle='None'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#ff6b6b', markersize=8, label='SELL', linestyle='None'),
    ]
    ax1.legend(handles=legend_items, loc='upper left', fontsize=8,
              facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)

    # ---- PANEL 2: EQUITY CURVES ----
    ax2 = fig.add_axes([0.06, 0.52, 0.88, 0.20])
    ax2.set_facecolor(bg_color)

    # Buy & hold
    start_price = prices[0]
    buy_hold = [STARTING_CAPITAL * (p / start_price) for p in prices]
    ax2.plot(dates, buy_hold, color='#4a5568', linewidth=1, linestyle='--',
             alpha=0.6, label='Buy & Hold')

    # Reference line
    ax2.axhline(y=STARTING_CAPITAL, color='#2d3748', linewidth=0.5, linestyle=':')

    for r in results:
        eq_values = [e['value'] for e in r['equity_curve']]
        eq_dates = pd.to_datetime([e['datetime'] for e in r['equity_curve']])
        ax2.plot(eq_dates, eq_values, color=colors[r['threshold']],
                linewidth=2, alpha=0.85, label=f"{r['label']} (${r['final_value']:,.0f})")

    ax2.set_title(f'Portfolio Value — ${STARTING_CAPITAL:,} Starting Capital',
                  color=text_color, fontsize=13, fontweight='bold', pad=10)
    ax2.tick_params(colors=text_color, labelsize=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax2.grid(True, alpha=0.15, color=grid_color)
    ax2.legend(loc='upper left', fontsize=8, facecolor=bg_color,
              edgecolor=grid_color, labelcolor=text_color)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(grid_color)
    ax2.spines['bottom'].set_color(grid_color)

    # ---- PANEL 3: CONFIDENCE DISTRIBUTION ----
    ax3 = fig.add_axes([0.06, 0.34, 0.40, 0.14])
    ax3.set_facecolor(bg_color)

    buy_confs = [s['confidence'] for s in signals if s['signal'] == 'BUY']
    bins = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    bin_colors = []
    for b in bins[:-1]:
        if b >= 80:
            bin_colors.append('#ef476f')
        elif b >= 70:
            bin_colors.append('#fbbf24')
        elif b >= 60:
            bin_colors.append('#06d6a0')
        else:
            bin_colors.append('#6366f1')

    counts, _, patches = ax3.hist(buy_confs, bins=bins, edgecolor='none', alpha=0.75)
    for patch, color in zip(patches, bin_colors):
        patch.set_facecolor(color)

    # Draw threshold lines
    for thresh in [60, 70, 80]:
        ax3.axvline(x=thresh, color=colors[thresh], linewidth=1.5, linestyle='--', alpha=0.6)

    ax3.set_title('BUY Signal Confidence Distribution',
                  color=text_color, fontsize=11, fontweight='bold', pad=8)
    ax3.set_xlabel('Confidence %', color=text_color, fontsize=9)
    ax3.set_ylabel('Count', color=text_color, fontsize=9)
    ax3.tick_params(colors=text_color, labelsize=8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color(grid_color)
    ax3.spines['bottom'].set_color(grid_color)

    # ---- PANEL 4: TRADE P&L BY THRESHOLD ----
    ax4 = fig.add_axes([0.54, 0.34, 0.40, 0.14])
    ax4.set_facecolor(bg_color)

    bar_width = 0.18
    x_pos = np.arange(len(results))

    for i, r in enumerate(results):
        color = colors[r['threshold']]
        if r['trades']:
            pnls = [t['pnl_pct'] for t in r['trades']]
            bar_colors = ['#06d6a0' if p > 0 else '#ef476f' for p in pnls]
            x = np.arange(len(pnls)) + i * 0.02
            ax4.bar(x * 1.5 + i * bar_width, pnls, width=bar_width,
                   color=color, alpha=0.7, edgecolor='none',
                   label=r['label'])

    ax4.axhline(y=0, color='#4a5568', linewidth=0.5)
    ax4.set_title('Per-Trade P&L by Threshold',
                  color=text_color, fontsize=11, fontweight='bold', pad=8)
    ax4.set_ylabel('P&L %', color=text_color, fontsize=9)
    ax4.tick_params(colors=text_color, labelsize=8)
    ax4.legend(loc='upper right', fontsize=7, facecolor=bg_color,
              edgecolor=grid_color, labelcolor=text_color)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_color(grid_color)
    ax4.spines['bottom'].set_color(grid_color)

    # ---- PANEL 5: RESULTS TABLE ----
    ax5 = fig.add_axes([0.06, 0.04, 0.88, 0.26])
    ax5.set_facecolor(bg_color)
    ax5.axis('off')

    # Header
    headers = ['Threshold', 'Final $', 'Return', 'Alpha', 'Trades',
               'Wins', 'Losses', 'Win Rate', 'Avg Win', 'Avg Loss',
               'Max DD', 'Best', 'Worst', 'Pred Acc']
    col_x = np.linspace(0.01, 0.95, len(headers))

    # Title
    ax5.text(0.5, 0.95, f'{asset_name} THRESHOLD COMPARISON — ${STARTING_CAPITAL:,} ALL-IN / ALL-OUT',
             transform=ax5.transAxes, fontsize=14, fontweight='bold',
             color=text_color, ha='center', va='top',
             fontfamily='monospace')

    # Model info
    info_text = f"Model: {config['best_combo']} | Window: {config['best_window']}h | " \
                f"Set {config.get('feature_set', 'A')} | Diag Accuracy: {config['accuracy']:.1f}%"
    ax5.text(0.5, 0.88, info_text,
             transform=ax5.transAxes, fontsize=9, color='#718096',
             ha='center', va='top', fontfamily='monospace')

    # Headers
    y_header = 0.78
    for j, h in enumerate(headers):
        ax5.text(col_x[j], y_header, h, transform=ax5.transAxes,
                fontsize=8, fontweight='bold', color='#a0aec0',
                ha='center', va='top', fontfamily='monospace')

    # Separator
    ax5.plot([0.01, 0.99], [y_header - 0.04, y_header - 0.04],
             transform=ax5.transAxes, color=grid_color, linewidth=0.5)

    # Data rows
    for i, r in enumerate(results):
        y = y_header - 0.09 - i * 0.12
        color = colors[r['threshold']]

        row_data = [
            r['label'],
            f"${r['final_value']:,.0f}",
            f"{r['return_pct']:+.1f}%",
            f"{r['alpha']:+.1f}%",
            str(r['n_trades']),
            str(r['wins']),
            str(r['losses']),
            f"{r['win_rate']:.0f}%",
            f"{r['avg_win']:+.1f}%",
            f"{r['avg_loss']:+.1f}%",
            f"{r['max_drawdown']:.1f}%",
            f"{r['max_trade']:+.1f}%",
            f"{r['min_trade']:+.1f}%",
            f"{r['pred_accuracy']:.1f}%",
        ]

        # Highlight best return
        best_return = max(res['return_pct'] for res in results)
        row_bg_alpha = 0.15 if r['return_pct'] == best_return else 0

        if row_bg_alpha > 0:
            from matplotlib.patches import FancyBboxPatch
            bbox = FancyBboxPatch((0.005, y - 0.035), 0.99, 0.10,
                                  transform=ax5.transAxes,
                                  boxstyle="round,pad=0.01",
                                  facecolor=color, alpha=0.08,
                                  edgecolor=color, linewidth=0.5)
            ax5.add_patch(bbox)

        for j, val in enumerate(row_data):
            fc = color if j == 0 else text_color
            if j == 2:  # return
                fc = '#06d6a0' if r['return_pct'] >= 0 else '#ef476f'
            elif j == 3:  # alpha
                fc = '#06d6a0' if r['alpha'] >= 0 else '#ef476f'
            elif j == 7:  # win rate
                fc = '#06d6a0' if r['win_rate'] >= 60 else '#fbbf24' if r['win_rate'] >= 50 else '#ef476f'

            fw = 'bold' if j in (0, 1, 2) else 'normal'
            ax5.text(col_x[j], y, val, transform=ax5.transAxes,
                    fontsize=9, fontweight=fw, color=fc,
                    ha='center', va='top', fontfamily='monospace')

        # Separator
        if i < len(results) - 1:
            ax5.plot([0.01, 0.99], [y - 0.05, y - 0.05],
                     transform=ax5.transAxes, color=grid_color, linewidth=0.3)

    # Winner banner
    best_result = max(results, key=lambda r: r['return_pct'])
    winner_text = f"WINNER: {best_result['label']} → {best_result['return_pct']:+.1f}% " \
                  f"(${best_result['final_value']:,.0f}) | " \
                  f"Win Rate {best_result['win_rate']:.0f}% | Alpha {best_result['alpha']:+.1f}%"
    winner_color = colors[best_result['threshold']]

    y_winner = 0.78 - 0.09 - len(results) * 0.12 - 0.04
    from matplotlib.patches import FancyBboxPatch
    bbox = FancyBboxPatch((0.1, y_winner - 0.02), 0.8, 0.06,
                          transform=ax5.transAxes,
                          boxstyle="round,pad=0.01",
                          facecolor=winner_color, alpha=0.15,
                          edgecolor=winner_color, linewidth=1)
    ax5.add_patch(bbox)
    ax5.text(0.5, y_winner + 0.01, winner_text,
             transform=ax5.transAxes, fontsize=11, fontweight='bold',
             color=winner_color, ha='center', va='center', fontfamily='monospace')

    # Save
    filename = f'{asset_name}_threshold_backtest.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight',
                facecolor=bg_color, edgecolor='none')
    plt.close()
    print(f"\n  Chart saved: {filename}")

    # Auto-open on Windows
    if sys.platform == 'win32':
        try:
            os.startfile(filename)
        except Exception:
            pass

    return filename


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  CRYPTO CONFIDENCE THRESHOLD BACKTEST")
    print(f"  Capital: ${STARTING_CAPITAL:,} | All-in / All-out")
    print(f"  Thresholds: All signals, >60%, >70%, >80%")
    print(f"  Trading fee: {TRADING_FEE*100:.2f}% per trade (Revolut X)")
    print("=" * 70)

    # Parse args
    asset_name = 'BTC'
    replay_hours = 500
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--asset' and i + 1 < len(args):
            asset_name = args[i + 1].upper()
        elif arg == '--hours' and i + 1 < len(args):
            replay_hours = int(args[i + 1])

    # Interactive if no args
    if len(args) == 0:
        print(f"\n  Available: BTC, ETH, SOL, XRP, DOGE")
        asset_input = input(f"  Asset [{asset_name}]: ").strip().upper()
        if asset_input and asset_input in ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']:
            asset_name = asset_input

        print(f"\n  How many hours of signals?")
        print(f"  1. 200 hours  (~8 days)")
        print(f"  2. 500 hours  (~21 days)")
        print(f"  3. 1000 hours (~42 days)")
        print(f"  4. 2000 hours (~83 days)")
        h_choice = input(f"  Choice [2]: ").strip()
        if h_choice == '1':
            replay_hours = 200
        elif h_choice == '3':
            replay_hours = 1000
        elif h_choice == '4':
            replay_hours = 2000
        else:
            replay_hours = 500

    print(f"\n  Asset: {asset_name}")
    print(f"  Signal window: {replay_hours} hours ({replay_hours/24:.0f} days)")

    # Load best model config
    config = load_best_config(asset_name)
    if config is None:
        return

    model_names = config['models'].split('+')
    window = config['best_window']
    fs = config.get('feature_set', 'A')
    opt_features = config.get('optimal_features', '')
    horizon = int(config.get('horizon', 4))

    if fs in ('D', 'E2', 'E3') and opt_features and pd.notna(opt_features):
        feature_override = opt_features.split(',')
    elif fs == 'B':
        feature_override = list(FEATURE_SET_B)
    else:
        feature_override = list(FEATURE_SET_A)

    print(f"  Model: {config['best_combo']} | Window: {window}h | "
          f"Set {fs} | Diag: {config['accuracy']:.1f}% | Horizon: {horizon}h")
    print(f"  Features: {len(feature_override)}")

    # Update data
    print(f"\n  Updating {asset_name} data...")
    update_all_data([asset_name])

    # Generate signals
    print(f"\n  Generating {replay_hours} hourly signals ({horizon}h horizon)...")
    t0 = time.time()
    signals = generate_signals(asset_name, model_names, window, replay_hours,
                                feature_override=feature_override, horizon=horizon)
    t_signals = time.time() - t0

    if not signals or len(signals) < 20:
        print(f"  ERROR: Only {len(signals)} signals generated. Need 20+.")
        return

    print(f"  Generated {len(signals)} signals in {t_signals:.0f}s")

    # Count signal distribution
    buy_count = sum(1 for s in signals if s['signal'] == 'BUY')
    sell_count = sum(1 for s in signals if s['signal'] == 'SELL')
    hold_count = sum(1 for s in signals if s['signal'] == 'HOLD')
    print(f"  Distribution: {buy_count} BUY | {sell_count} SELL | {hold_count} HOLD")

    # Confidence distribution
    buy_confs = [s['confidence'] for s in signals if s['signal'] == 'BUY']
    if buy_confs:
        print(f"  BUY confidence: min={min(buy_confs):.0f}% | "
              f"avg={np.mean(buy_confs):.0f}% | max={max(buy_confs):.0f}%")
        for thresh in [60, 70, 80]:
            above = sum(1 for c in buy_confs if c >= thresh)
            print(f"    >={thresh}%: {above} signals ({above/len(buy_confs)*100:.0f}% of BUY)")

    # Run backtests
    print(f"\n{'='*70}")
    print(f"  RUNNING THRESHOLD BACKTESTS")
    print(f"{'='*70}")

    results = []
    for threshold in THRESHOLDS:
        label = 'All Signals' if threshold == 0 else f'>{threshold}%'
        print(f"\n  --- {label} ---")
        r = run_threshold_backtest(signals, threshold, STARTING_CAPITAL)
        results.append(r)

        # Print summary
        ret_color = '+' if r['return_pct'] >= 0 else ''
        print(f"    Final:     ${r['final_value']:>9,.2f}  ({ret_color}{r['return_pct']:.1f}%)")
        print(f"    Buy&Hold:  ${r['buy_hold_value']:>9,.2f}  ({r['buy_hold_return']:+.1f}%)")
        print(f"    Alpha:     {r['alpha']:+.1f}%")
        print(f"    Trades:    {r['n_trades']} ({r['wins']}W / {r['losses']}L)")
        print(f"    Win Rate:  {r['win_rate']:.0f}%")
        print(f"    Avg Win:   {r['avg_win']:+.1f}% | Avg Loss: {r['avg_loss']:+.1f}%")
        print(f"    Max DD:    {r['max_drawdown']:.1f}%")
        print(f"    Pred Acc:  {r['pred_accuracy']:.1f}%")
        print(f"    Signals:   {r['buy_signals_filtered']}/{r['buy_signals_total']} BUY passed filter")

    # ---- COMPARISON TABLE ----
    print(f"\n{'='*70}")
    print(f"  THRESHOLD COMPARISON — {asset_name} — ${STARTING_CAPITAL:,}")
    print(f"{'='*70}")
    print(f"  {'Threshold':<14s} {'Final $':>10s} {'Return':>8s} {'Alpha':>8s} "
          f"{'Trades':>7s} {'Win%':>6s} {'AvgW':>7s} {'AvgL':>7s} {'MaxDD':>7s} {'PredAcc':>8s}")
    print(f"  {'-'*90}")

    for r in results:
        marker = ' <<<' if r['return_pct'] == max(res['return_pct'] for res in results) else ''
        print(f"  {r['label']:<14s} ${r['final_value']:>9,.0f} {r['return_pct']:>+7.1f}% "
              f"{r['alpha']:>+7.1f}% {r['n_trades']:>6d}  {r['win_rate']:>5.0f}% "
              f"{r['avg_win']:>+6.1f}% {r['avg_loss']:>+6.1f}% {r['max_drawdown']:>6.1f}% "
              f"{r['pred_accuracy']:>7.1f}%{marker}")

    # Winner
    best = max(results, key=lambda r: r['return_pct'])
    print(f"\n  {'='*70}")
    print(f"  WINNER: {best['label']}")
    print(f"  ${STARTING_CAPITAL:,} → ${best['final_value']:,.2f} ({best['return_pct']:+.1f}%)")
    print(f"  Alpha vs Buy&Hold: {best['alpha']:+.1f}%")
    print(f"  Win Rate: {best['win_rate']:.0f}% | Trades: {best['n_trades']}")
    print(f"  {'='*70}")

    # Trade log for winner
    if best['trades']:
        print(f"\n  TRADE LOG ({best['label']}):")
        print(f"  {'#':<4s} {'Buy Date':>16s} {'Buy $':>10s} {'Conf':>6s} "
              f"{'Sell Date':>16s} {'Sell $':>10s} {'P&L':>8s} {'Hrs':>5s}")
        print(f"  {'-'*80}")
        for j, t in enumerate(best['trades']):
            pnl_str = f"{t['pnl_pct']:+.1f}%"
            marker = ' W' if t['pnl_pct'] > 0 else ' L'
            print(f"  {j+1:<4d} {t['buy_dt']:>16s} ${t['buy_price']:>9,.0f} "
                  f"{t['buy_conf']:>5.0f}% {t['sell_dt']:>16s} "
                  f"${t['sell_price']:>9,.0f} {pnl_str:>7s}{marker} {t['duration_h']:>4d}h")

    # Generate chart
    print(f"\n  Generating chart...")
    generate_threshold_chart(asset_name, results, signals, config)

    # Save results CSV
    csv_rows = []
    for r in results:
        csv_rows.append({
            'asset': asset_name,
            'threshold': r['label'],
            'final_value': r['final_value'],
            'return_pct': r['return_pct'],
            'buy_hold_return': r['buy_hold_return'],
            'alpha': r['alpha'],
            'trades': r['n_trades'],
            'win_rate': r['win_rate'],
            'avg_win': r['avg_win'],
            'avg_loss': r['avg_loss'],
            'max_drawdown': r['max_drawdown'],
            'pred_accuracy': r['pred_accuracy'],
        })
    df_csv = pd.DataFrame(csv_rows)
    csv_name = f'{asset_name}_threshold_results.csv'
    df_csv.to_csv(csv_name, index=False)
    print(f"  Results saved: {csv_name}")

    print(f"\nDone!")


if __name__ == '__main__':
    main()
