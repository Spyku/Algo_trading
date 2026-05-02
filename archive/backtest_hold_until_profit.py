"""Backtest: hold-until-profitable strategy.

Compares baseline (sell immediately on SELL signal) vs holding when at a loss
and only selling once P&L >= min_sell_pnl_pct AND signal == SELL.

Usage:
  python -u backtest_hold_until_profit.py ETH 2880
  python -u backtest_hold_until_profit.py BTC 4320
"""
import os, sys, json
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from crypto_trading_system_ed import (
    generate_signals, BACKTEST_FEE_PER_LEG as TRADING_FEE, PRODUCTION_CSV, _suppress_stderr,
)  # imports the new realistic-blend backtest fee (5 bps/leg), aliased as TRADING_FEE for the local sim


def simulate_baseline(signals, conf_threshold):
    cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
    trades, trade_log = 0, []

    for s in signals:
        price = s['close']
        conf = s['confidence']

        if s['signal'] == 'BUY' and conf >= conf_threshold and not in_pos:
            held = cash * (1 - TRADING_FEE) / price
            cash = 0
            in_pos = True
            entry_px = price
            trades += 1
        elif s['signal'] == 'SELL' and in_pos:
            cash = held * price * (1 - TRADING_FEE)
            pnl_pct = (price / entry_px - 1) * 100
            trade_log.append(pnl_pct)
            held = 0
            in_pos = False
            trades += 1

    final = cash if not in_pos else held * signals[-1]['close'] * (1 - TRADING_FEE)
    if in_pos:
        trade_log.append((signals[-1]['close'] / entry_px - 1) * 100)
    return _stats(final, trades, trade_log, in_pos)


def simulate_hold_until_profit(signals, conf_threshold, min_sell_pnl, max_override_hours):
    cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
    trades, trade_log = 0, []
    blocked_sells = 0
    max_hold_hours = 0
    current_hold = 0

    for s in signals:
        price = s['close']
        conf = s['confidence']

        if in_pos:
            current_hold += 1

        if s['signal'] == 'BUY' and conf >= conf_threshold and not in_pos:
            held = cash * (1 - TRADING_FEE) / price
            cash = 0
            in_pos = True
            entry_px = price
            trades += 1
            current_hold = 0
        elif s['signal'] == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = current_hold >= max_override_hours
            if cur_pnl >= min_sell_pnl or override_expired:
                cash = held * price * (1 - TRADING_FEE)
                trade_log.append(cur_pnl)
                held = 0
                in_pos = False
                trades += 1
                max_hold_hours = max(max_hold_hours, current_hold)
                current_hold = 0
            else:
                blocked_sells += 1

    final = cash if not in_pos else held * signals[-1]['close'] * (1 - TRADING_FEE)
    if in_pos:
        trade_log.append((signals[-1]['close'] / entry_px - 1) * 100)
        max_hold_hours = max(max_hold_hours, current_hold)
    stats = _stats(final, trades, trade_log, in_pos)
    stats['blocked_sells'] = blocked_sells
    stats['max_hold_hours'] = max_hold_hours
    return stats


def _stats(final, trades, trade_log, still_invested):
    ret = (final / 1000.0 - 1) * 100
    winners = sum(1 for t in trade_log if t > 0)
    losers = sum(1 for t in trade_log if t <= 0)
    win_rate = (winners / len(trade_log) * 100) if trade_log else 0
    avg_win = sum(t for t in trade_log if t > 0) / max(winners, 1)
    avg_loss = sum(t for t in trade_log if t <= 0) / max(losers, 1)
    return {
        'return_pct': round(ret, 2),
        'final_value': round(final, 2),
        'trades': trades,
        'round_trips': len(trade_log),
        'win_rate': round(win_rate, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'still_invested': still_invested,
    }


def run_backtest(asset, replay_hours):
    df_models = pd.read_csv(PRODUCTION_CSV)
    available_h = sorted(df_models[df_models['coin'] == asset]['horizon'].unique())

    cfg_path = 'config/regime_config_ed.json'
    with open(cfg_path) as f:
        regime_cfg = json.load(f)
    asset_cfg = regime_cfg.get(asset, {})
    bull_h = asset_cfg.get('bull', {}).get('horizon', 7)
    bear_h = asset_cfg.get('bear', {}).get('horizon', 8)
    bull_conf = asset_cfg.get('bull', {}).get('min_confidence', 85)
    bear_conf = asset_cfg.get('bear', {}).get('min_confidence', 65)

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    failsafe_hours = [6, 8, 10, 12]

    print(f"\n{'='*80}")
    print(f"  BACKTEST: HOLD-UNTIL-PROFITABLE (fine sweep)")
    print(f"  Asset: {asset} | Replay: {replay_hours}h ({replay_hours/720:.1f} months)")
    print(f"  Thresholds: {thresholds}")
    print(f"  Failsafe hours: {failsafe_hours}")
    print(f"  Production: bull={bull_h}h@{bull_conf}% | bear={bear_h}h@{bear_conf}%")
    print(f"  Fee per trade: {TRADING_FEE*100:.3f}%")
    print(f"{'='*80}")

    target_horizons = [h for h in available_h if h in [6, 7, 8]]
    for h in target_horizons:
        rows = df_models[(df_models['coin'] == asset) & (df_models['horizon'] == h)]
        if len(rows) == 0:
            continue
        row = rows.sort_values('combined_score', ascending=False).iloc[0]
        feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
        gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0

        if h == bull_h:
            conf = bull_conf
        elif h == bear_h:
            conf = bear_conf
        else:
            conf = 85

        print(f"\n{'─'*80}")
        print(f"  {asset} {h}h | {row['models']} | w={int(row['best_window'])}h | γ={gamma} | conf>={conf}%")
        print(f"{'─'*80}")

        print(f"  Generating signals ({replay_hours}h)...", end='', flush=True)
        with _suppress_stderr():
            sigs = generate_signals(asset, row['models'].split('+'),
                                    int(row['best_window']), replay_hours,
                                    feature_override=feats, horizon=h, gamma=gamma)
        print(f" {len(sigs)} candles")

        if not sigs:
            print("  No signals generated, skipping")
            continue

        base = simulate_baseline(sigs, conf)
        print(f"\n  Baseline (current model): {base['return_pct']:+.2f}% | {base['round_trips']} trades | WR {base['win_rate']:.1f}%")

        print(f"\n  {'Threshold':<12} {'Failsafe':>8} {'Return':>9} {'vs Base':>8} {'Trades':>7} {'WinRate':>8} {'AvgWin':>8} {'AvgLoss':>8} {'Blocked':>8} {'MaxHold':>8}")
        print(f"  {'─'*12} {'─'*8} {'─'*9} {'─'*8} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

        best_ret = base['return_pct']
        best_combo = None

        for t in thresholds:
            for fh in failsafe_hours:
                r = simulate_hold_until_profit(sigs, conf, t, fh)
                delta = r['return_pct'] - base['return_pct']
                marker = ' ★' if r['return_pct'] > best_ret else (' ✓' if delta > 0 else '')
                if r['return_pct'] > best_ret:
                    best_ret = r['return_pct']
                    best_combo = (t, fh)
                print(f"  {t:.2f}%        {fh:>7}h {r['return_pct']:>+8.2f}% {delta:>+7.2f}% {r['round_trips']:>7} {r['win_rate']:>7.1f}% {r['avg_win']:>+7.2f}% {r['avg_loss']:>+7.2f}% {r['blocked_sells']:>8} {r['max_hold_hours']:>7}h{marker}")

        if best_combo:
            print(f"\n  ★ BEST: threshold={best_combo[0]:.2f}%, failsafe={best_combo[1]}h → {best_ret:+.2f}% (vs baseline {base['return_pct']:+.2f}%)")
        else:
            print(f"\n  No improvement over baseline ({base['return_pct']:+.2f}%)")


if __name__ == '__main__':
    asset = sys.argv[1] if len(sys.argv) > 1 else 'ETH'
    replay = int(sys.argv[2]) if len(sys.argv) > 2 else 2880
    run_backtest(asset, replay)
