"""test_tp_1m_overlay.py — Take-profit overlay backtest using 1-minute candles.

Goal: catch upward spikes the hourly model misses (e.g. a +1% pop over 5min
where the model wouldn't have flipped SELL until next hour, by which time
price reverted).

Method:
  1. Run baseline hourly strategy on existing signal cache (current promoted
     config: sma24>sma100, bull=6h@65% / bear=5h@75%, shields ON, bear gate
     active). Identify each completed BUY -> SELL trade.
  2. For each baseline trade, replay 1m candles between entry+1min and
     exit. At each minute, compute ret_lookback_min = close[t] / close[t-LB] - 1.
     If ret >= threshold AND price > entry (in profit), exit at this minute's
     close instead of the baseline exit.
  3. Sweep threshold x lookback x cooldown. Compare total return + per-trade
     metrics vs baseline.

NO production code touched. Read-only on cache + 1m CSV.

Usage: python tools/test_tp_1m_overlay.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ENGINE)

CACHE_PKL    = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
ONE_MIN_CSV  = os.path.join(ENGINE, 'data', 'eth_1m_data.csv')
REGIME_CFG   = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
OUT_CSV      = os.path.join(ENGINE, 'output',
                            f"tp_1m_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
TRADES_CSV   = os.path.join(ENGINE, 'output',
                            f"tp_1m_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

REPLAY_DAYS = 60
FEE_PER_LEG = 0.0005   # 5 bps maker blend (per CLAUDE.md BACKTEST_FEE_PER_LEG)

# Sweep grid (TP-only, as requested)
TP_THRESHOLDS_PCT = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]   # ret_5m must reach >= this %
LOOKBACK_MINUTES   = [3, 5, 10]                          # rolling window in minutes
# 18 variants total


def load_cfg():
    with open(REGIME_CFG) as f:
        cfg = json.load(f).get('ETH', {})
    return {
        'bull_h':    int(cfg.get('bull', {}).get('horizon', 6)),
        'bull_thr':  float(cfg.get('bull', {}).get('min_confidence', 65)),
        'bear_h':    int(cfg.get('bear', {}).get('horizon', 5)),
        'bear_thr':  float(cfg.get('bear', {}).get('min_confidence', 75)),
        'bull_shield': bool(cfg.get('bull', {}).get('hold_shield', True)),
        'bear_shield': bool(cfg.get('bear', {}).get('hold_shield', True)),
        'min_sell_pnl_pct': float(cfg.get('min_sell_pnl_pct', 0.5)),
        'max_hold_hours':   int(cfg.get('max_hold_hours', 10)),
        'bear_gate':        cfg.get('bear', {}).get('rally_cooldown', {}),
        'bull_gate':        cfg.get('bull', {}).get('rally_cooldown', {}),
    }


def load_signals():
    with open(CACHE_PKL, 'rb') as f:
        sigs = pickle.load(f)
    for s in sigs:
        s['datetime'] = pd.Timestamp(s['datetime'])
    sigs.sort(key=lambda s: s['datetime'])
    end = sigs[-1]['datetime']
    lo = end - pd.Timedelta(days=REPLAY_DAYS)
    return [s for s in sigs if s['datetime'] >= lo]


def load_1m():
    df = pd.read_csv(ONE_MIN_CSV)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    return df


def _gate_blocks_buy(gate_cfg, signal_dt, sig_history_idx, signals):
    """Apply rally_cooldown gate: block BUY if a recent rally trigger fires.
    Mirrors crypto_revolut_ed_v2._update_rally_cooldown logic.

    gate_cfg keys: enabled, h_short, h_long, t_short_pct, t_long_pct, cd_hours.
    Triggers when retX_short >= t_short_pct OR retX_long >= t_long_pct in the
    last cd_hours block window. We approximate by checking each historical
    bar within `cd_hours` lookback.
    """
    if not gate_cfg or not gate_cfg.get('enabled', False):
        return False
    h_short = int(gate_cfg.get('h_short', 30))
    h_long  = int(gate_cfg.get('h_long', 36))
    t_short = float(gate_cfg.get('t_short_pct', 9.0))
    t_long  = float(gate_cfg.get('t_long_pct', 9.0))
    cd_h    = int(gate_cfg.get('cd_hours', 48))

    # Walk backwards from signal_dt looking for a trigger bar within cd_h hours
    cutoff = signal_dt - pd.Timedelta(hours=cd_h)
    for j in range(sig_history_idx - 1, -1, -1):
        s = signals[j]
        bar_dt = s['datetime']
        if bar_dt < cutoff:
            return False
        bar_close = s['close']
        # rrX = (close[bar] / close[bar - X*1h]) - 1
        for k_back, thr in ((h_short, t_short), (h_long, t_long)):
            target_dt = bar_dt - pd.Timedelta(hours=k_back)
            ref = None
            for kk in range(j, -1, -1):
                if signals[kk]['datetime'] <= target_dt:
                    ref = signals[kk]['close']
                    break
            if ref is None:
                continue
            rr = (bar_close / ref - 1.0) * 100.0
            if rr >= thr:
                return True
    return False


def baseline_simulate(sigs, cfg):
    """Simulate the current promoted strategy. Returns list of completed trades
    with entry/exit datetime + price + regime at entry.

    Each trade: dict(entry_dt, entry_price, exit_dt, exit_price, regime_entry,
                     hold_hours, pnl_pct, exit_reason).
    """
    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry_dt = None
    entry_px = 0.0
    entry_regime = None
    hold = 0
    trades = []

    for i, s in enumerate(sigs):
        regime = s['regime']
        sig    = s['signal']
        sconf  = float(s.get('confidence', 0))
        price  = float(s['close'])
        thr    = cfg['bull_thr'] if regime == 'bull' else cfg['bear_thr']
        shield_on = cfg['bull_shield'] if regime == 'bull' else cfg['bear_shield']
        gate_cfg  = cfg['bull_gate']   if regime == 'bull' else cfg['bear_gate']

        if in_pos:
            hold += 1

        if sig == 'BUY' and sconf >= thr and not in_pos:
            # Apply gate
            if _gate_blocks_buy(gate_cfg, s['datetime'], i, sigs):
                continue
            qty = cash * (1 - FEE_PER_LEG) / price
            cash = 0.0
            in_pos = True
            entry_dt = s['datetime']
            entry_px = price
            entry_regime = regime
            hold = 0

        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1.0) * 100.0
            do_sell = False
            reason = None

            if not shield_on:
                do_sell = True; reason = 'shield_off'
            elif cur_pnl >= cfg['min_sell_pnl_pct']:
                do_sell = True; reason = 'pnl_target'
            elif hold >= cfg['max_hold_hours']:
                do_sell = True; reason = 'failsafe'

            if do_sell:
                cash = qty * price * (1 - FEE_PER_LEG)
                trades.append({
                    'entry_dt': entry_dt,
                    'entry_price': entry_px,
                    'exit_dt': s['datetime'],
                    'exit_price': price,
                    'regime_entry': entry_regime,
                    'hold_hours': hold,
                    'pnl_pct': cur_pnl,
                    'exit_reason': reason,
                })
                qty = 0.0
                in_pos = False

    if in_pos:
        last = sigs[-1]
        cash = qty * last['close'] * (1 - FEE_PER_LEG)
        trades.append({
            'entry_dt': entry_dt,
            'entry_price': entry_px,
            'exit_dt': last['datetime'],
            'exit_price': float(last['close']),
            'regime_entry': entry_regime,
            'hold_hours': hold,
            'pnl_pct': (last['close'] / entry_px - 1.0) * 100.0,
            'exit_reason': 'end_of_window',
        })

    total_ret = (cash / 1000.0 - 1.0) * 100.0
    return trades, total_ret


def tp_overlay_for_trade(trade, df_1m, threshold_pct, lookback_min):
    """For a single baseline trade, scan 1m candles forward from entry+lookback_min
    until exit_dt. Find first minute where ret_lookback_min >= threshold AND
    price > entry. Return (tp_exit_dt, tp_exit_price) or (None, None) if no TP.
    """
    entry_dt = pd.Timestamp(trade['entry_dt'])
    exit_dt  = pd.Timestamp(trade['exit_dt'])
    entry_px = trade['entry_price']

    # Slice 1m data: from entry to exit (exclusive of exit bar)
    sl = df_1m.loc[entry_dt:exit_dt]
    if len(sl) < lookback_min + 1:
        return None, None

    # We need lookback_min minutes of history before evaluating
    closes = sl['close'].values
    times = sl.index

    for j in range(lookback_min, len(closes)):
        ref = closes[j - lookback_min]
        cur = closes[j]
        if ref <= 0:
            continue
        ret_pct = (cur / ref - 1.0) * 100.0
        if ret_pct >= threshold_pct and cur > entry_px:
            return times[j], cur

    return None, None


def simulate_with_tp(baseline_trades, df_1m, threshold_pct, lookback_min):
    """Re-run a list of baseline trades with TP overlay. Compounds capital."""
    cash = 1000.0
    new_trades = []
    tp_fires = 0
    tp_better = 0
    tp_worse  = 0

    for t in baseline_trades:
        # Apply BUY at original entry price (regime + gate already passed)
        qty = cash * (1 - FEE_PER_LEG) / t['entry_price']
        # Check if TP fires before baseline exit
        tp_dt, tp_px = tp_overlay_for_trade(t, df_1m, threshold_pct, lookback_min)
        if tp_dt is not None:
            exit_px = tp_px
            exit_dt = tp_dt
            exit_reason = 'tp_overlay'
            tp_fires += 1
            if tp_px > t['exit_price']:
                tp_better += 1
            else:
                tp_worse += 1
        else:
            exit_px = t['exit_price']
            exit_dt = t['exit_dt']
            exit_reason = t['exit_reason']

        cash = qty * exit_px * (1 - FEE_PER_LEG)
        pnl = (exit_px / t['entry_price'] - 1.0) * 100.0 - 2 * FEE_PER_LEG * 100.0
        new_trades.append({
            'entry_dt': t['entry_dt'],
            'entry_price': t['entry_price'],
            'exit_dt': exit_dt,
            'exit_price': exit_px,
            'regime_entry': t['regime_entry'],
            'pnl_pct': pnl,
            'baseline_pnl_pct': t['pnl_pct'],
            'exit_reason': exit_reason,
        })

    total_ret = (cash / 1000.0 - 1.0) * 100.0
    return new_trades, total_ret, tp_fires, tp_better, tp_worse


def main():
    print("=" * 100)
    print(f"  ETH Take-Profit 1m Overlay Backtest — {REPLAY_DAYS}d, fee={FEE_PER_LEG*100:.2f}%/leg")
    print("=" * 100)
    cfg = load_cfg()
    print(f"  Live config: detector=sma24>sma100, "
          f"bull={cfg['bull_h']}h@{cfg['bull_thr']}% (shield={cfg['bull_shield']}), "
          f"bear={cfg['bear_h']}h@{cfg['bear_thr']}% (shield={cfg['bear_shield']})")
    print(f"               min_sell_pnl={cfg['min_sell_pnl_pct']}%, max_hold={cfg['max_hold_hours']}h")
    print(f"               bull_gate.enabled={cfg['bull_gate'].get('enabled', False)}, "
          f"bear_gate.enabled={cfg['bear_gate'].get('enabled', False)}")

    print("\n  Loading signal cache + 1m candles...")
    sigs = load_signals()
    df_1m = load_1m()
    print(f"  Hourly signals: {len(sigs)}  range {sigs[0]['datetime']} -> {sigs[-1]['datetime']}")
    print(f"  1m candles:     {len(df_1m):,}  range {df_1m.index.min()} -> {df_1m.index.max()}")

    # Trim to overlap window
    overlap_start = max(sigs[0]['datetime'], df_1m.index.min())
    overlap_end   = min(sigs[-1]['datetime'], df_1m.index.max())
    print(f"  Overlap window: {overlap_start} -> {overlap_end}")

    sigs = [s for s in sigs if overlap_start <= s['datetime'] <= overlap_end]
    df_1m = df_1m.loc[overlap_start:overlap_end]
    print(f"  Trimmed: {len(sigs)} hourly signals, {len(df_1m):,} 1m candles")

    print("\n  --- BASELINE simulation (no TP overlay) ---")
    base_trades, base_ret = baseline_simulate(sigs, cfg)
    base_wr = (sum(1 for t in base_trades if t['pnl_pct'] > 0) / len(base_trades) * 100) if base_trades else 0
    base_avg = np.mean([t['pnl_pct'] for t in base_trades]) if base_trades else 0
    print(f"  Baseline: return={base_ret:+.2f}%  trades={len(base_trades)}  "
          f"WR={base_wr:.0f}%  avg_pnl={base_avg:+.2f}%")
    if not base_trades:
        print("  ! No baseline trades — abort")
        return

    print("\n  --- TP OVERLAY sweep (18 variants) ---")
    print(f"  {'Variant':<22}{'Ret%':>9}{'d_vs_base':>11}{'Trades':>8}"
          f"{'WR%':>6}{'AvgPnL':>9}{'TP_fires':>10}{'better':>8}{'worse':>7}")
    print(f"  {'-'*22}{'-'*9}{'-'*11}{'-'*8}{'-'*6}{'-'*9}{'-'*10}{'-'*8}{'-'*7}")

    rows = []
    all_trades_rows = []
    for thr_pct in TP_THRESHOLDS_PCT:
        for lb_min in LOOKBACK_MINUTES:
            new_trades, new_ret, fires, better, worse = simulate_with_tp(
                base_trades, df_1m, thr_pct, lb_min)
            wr = (sum(1 for t in new_trades if t['pnl_pct'] > 0) / len(new_trades) * 100)
            avg = np.mean([t['pnl_pct'] for t in new_trades])
            delta = new_ret - base_ret
            name = f"thr={thr_pct}%_lb={lb_min}m"
            print(f"  {name:<22}{new_ret:>+8.2f}%{delta:>+10.2f}{len(new_trades):>8}"
                  f"{wr:>5.0f}%{avg:>+8.2f}%{fires:>10}{better:>8}{worse:>7}")
            rows.append({
                'variant': name,
                'threshold_pct': thr_pct,
                'lookback_min': lb_min,
                'return_pct': round(new_ret, 4),
                'delta_vs_base': round(delta, 4),
                'trades': len(new_trades),
                'win_rate': round(wr, 2),
                'avg_pnl': round(avg, 4),
                'tp_fires': fires,
                'tp_better_than_base': better,
                'tp_worse_than_base': worse,
            })
            for t in new_trades:
                t['variant'] = name
                all_trades_rows.append(t)

    # Add baseline as ranking reference
    rows.append({
        'variant': 'BASELINE',
        'threshold_pct': None, 'lookback_min': None,
        'return_pct': round(base_ret, 4), 'delta_vs_base': 0.0,
        'trades': len(base_trades),
        'win_rate': round(base_wr, 2),
        'avg_pnl': round(base_avg, 4),
        'tp_fires': 0, 'tp_better_than_base': 0, 'tp_worse_than_base': 0,
    })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n  Summary CSV: {OUT_CSV}")

    df_trades = pd.DataFrame(all_trades_rows)
    df_trades.to_csv(TRADES_CSV, index=False)
    print(f"  Trades CSV:  {TRADES_CSV}")

    # Top 5 by delta
    print("\n  --- TOP 5 by delta_vs_base ---")
    top = df_out[df_out['variant'] != 'BASELINE'].sort_values('delta_vs_base', ascending=False).head(5)
    for _, r in top.iterrows():
        print(f"    {r['variant']:<22}  ret={r['return_pct']:+.2f}%  "
              f"delta={r['delta_vs_base']:+.2f}pp  trades={r['trades']}  WR={r['win_rate']:.0f}%  "
              f"fires={r['tp_fires']} (better={r['tp_better_than_base']}, worse={r['tp_worse_than_base']})")
    print(f"\n    BASELINE             ret={base_ret:+.2f}%  trades={len(base_trades)}  WR={base_wr:.0f}%")


if __name__ == '__main__':
    main()
