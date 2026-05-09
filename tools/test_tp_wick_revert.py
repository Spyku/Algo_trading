"""test_tp_wick_revert.py — Wick-revert take-profit overlay using 1m candles.

Different mechanism from test_tp_1m_overlay.py: instead of exiting on rolling
ret_5min >= X (which fires on continuation rallies), this fires only when a
1m candle prints a bullish WICK >= X% and reverts within the same candle —
i.e., classic blow-off pin bar / failed thrust.

Wick definition:
  upper_wick_pct = (high - max(open, close)) / max(open, close) * 100

Revert condition (3 modes tested):
  - below_open      : close < open (red candle with big upper wick)
  - below_mid       : close < (open + high) / 2
  - body_in_lower_half : close - open <= 0  AND  close - low >= 0.5 * (high - low)
                        (body in lower 50% of the candle's range)

Exit price is the candle's close (since the wick has already failed by the
end of the minute).

Standalone: NO production files touched. Reads cache + 1m CSV.

Usage: python tools/test_tp_wick_revert.py
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ENGINE)

CACHE_PKL    = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
ONE_MIN_CSV  = os.path.join(ENGINE, 'data', 'eth_1m_data.csv')
REGIME_CFG   = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_CSV      = os.path.join(ENGINE, 'output', f"tp_wick_revert_{ts}.csv")
TRADES_CSV   = os.path.join(ENGINE, 'output', f"tp_wick_revert_trades_{ts}.csv")

REPLAY_DAYS = 60
FEE_PER_LEG = 0.0005

WICK_THRESHOLDS_PCT = [0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
REVERT_MODES = ['below_open', 'below_mid', 'body_in_lower_half']
# 18 variants


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
    if not gate_cfg or not gate_cfg.get('enabled', False):
        return False
    h_short = int(gate_cfg.get('h_short', 30))
    h_long  = int(gate_cfg.get('h_long', 36))
    t_short = float(gate_cfg.get('t_short_pct', 9.0))
    t_long  = float(gate_cfg.get('t_long_pct', 9.0))
    cd_h    = int(gate_cfg.get('cd_hours', 48))
    cutoff = signal_dt - pd.Timedelta(hours=cd_h)
    for j in range(sig_history_idx - 1, -1, -1):
        s = signals[j]
        bar_dt = s['datetime']
        if bar_dt < cutoff:
            return False
        bar_close = s['close']
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
            do_sell = False; reason = None
            if not shield_on:
                do_sell = True; reason = 'shield_off'
            elif cur_pnl >= cfg['min_sell_pnl_pct']:
                do_sell = True; reason = 'pnl_target'
            elif hold >= cfg['max_hold_hours']:
                do_sell = True; reason = 'failsafe'
            if do_sell:
                cash = qty * price * (1 - FEE_PER_LEG)
                trades.append({
                    'entry_dt': entry_dt, 'entry_price': entry_px,
                    'exit_dt': s['datetime'], 'exit_price': price,
                    'regime_entry': entry_regime,
                    'hold_hours': hold, 'pnl_pct': cur_pnl,
                    'exit_reason': reason,
                })
                qty = 0.0; in_pos = False
    if in_pos:
        last = sigs[-1]
        cash = qty * last['close'] * (1 - FEE_PER_LEG)
        trades.append({
            'entry_dt': entry_dt, 'entry_price': entry_px,
            'exit_dt': last['datetime'], 'exit_price': float(last['close']),
            'regime_entry': entry_regime,
            'hold_hours': 0, 'pnl_pct': (last['close']/entry_px - 1.0)*100,
            'exit_reason': 'end_of_window',
        })
    total_ret = (cash / 1000.0 - 1.0) * 100.0
    return trades, total_ret


def find_wick_revert_in_trade(trade, df_1m, wick_thr_pct, revert_mode):
    """Scan 1m candles in [entry_dt+1min, exit_dt). Find first one where:
      - upper_wick_pct >= wick_thr_pct
      - revert condition met (per mode)
      - close > entry_price (in profit)
    Return (exit_dt, exit_price) or (None, None).
    """
    entry_dt = pd.Timestamp(trade['entry_dt'])
    exit_dt  = pd.Timestamp(trade['exit_dt'])
    entry_px = trade['entry_price']
    sl = df_1m.loc[entry_dt:exit_dt]
    if len(sl) < 2:
        return None, None

    op = sl['open'].values
    hi = sl['high'].values
    lo = sl['low'].values
    cl = sl['close'].values
    times = sl.index

    for j in range(1, len(cl)):
        body_top = max(op[j], cl[j])
        if body_top <= 0:
            continue
        upper_wick_pct = (hi[j] - body_top) / body_top * 100.0
        if upper_wick_pct < wick_thr_pct:
            continue
        # Profit check: close after exit must beat entry
        if cl[j] <= entry_px:
            continue
        # Revert condition
        revert_ok = False
        if revert_mode == 'below_open':
            revert_ok = cl[j] < op[j]
        elif revert_mode == 'below_mid':
            mid = (op[j] + hi[j]) / 2.0
            revert_ok = cl[j] < mid
        elif revert_mode == 'body_in_lower_half':
            rng = hi[j] - lo[j]
            if rng > 0:
                # Body must be red AND positioned in lower half
                body_low = min(op[j], cl[j])
                # body's low should be in lower half: body_low - lo <= 0.5 * rng
                revert_ok = (cl[j] <= op[j]) and ((body_low - lo[j]) <= 0.5 * rng)
        if revert_ok:
            return times[j], cl[j]
    return None, None


def simulate_with_tp(baseline_trades, df_1m, wick_thr, revert_mode):
    cash = 1000.0
    new_trades = []
    fires = 0; better = 0; worse = 0
    for t in baseline_trades:
        qty = cash * (1 - FEE_PER_LEG) / t['entry_price']
        tp_dt, tp_px = find_wick_revert_in_trade(t, df_1m, wick_thr, revert_mode)
        if tp_dt is not None:
            exit_px = tp_px; exit_dt = tp_dt; exit_reason = 'wick_revert'
            fires += 1
            if tp_px > t['exit_price']:
                better += 1
            else:
                worse += 1
        else:
            exit_px = t['exit_price']; exit_dt = t['exit_dt']
            exit_reason = t['exit_reason']
        cash = qty * exit_px * (1 - FEE_PER_LEG)
        pnl = (exit_px / t['entry_price'] - 1.0) * 100.0 - 2 * FEE_PER_LEG * 100.0
        new_trades.append({
            'entry_dt': t['entry_dt'], 'entry_price': t['entry_price'],
            'exit_dt': exit_dt, 'exit_price': exit_px,
            'regime_entry': t['regime_entry'],
            'pnl_pct': pnl, 'baseline_pnl_pct': t['pnl_pct'],
            'exit_reason': exit_reason,
        })
    total_ret = (cash / 1000.0 - 1.0) * 100.0
    return new_trades, total_ret, fires, better, worse


def main():
    print("=" * 100)
    print(f"  ETH Wick-Revert TP Overlay Backtest -- {REPLAY_DAYS}d, fee={FEE_PER_LEG*100:.2f}%/leg")
    print("=" * 100)
    cfg = load_cfg()
    print(f"  Config: bull={cfg['bull_h']}h@{cfg['bull_thr']}% (shield={cfg['bull_shield']}), "
          f"bear={cfg['bear_h']}h@{cfg['bear_thr']}% (shield={cfg['bear_shield']})")
    print(f"          min_sell_pnl={cfg['min_sell_pnl_pct']}%, max_hold={cfg['max_hold_hours']}h")

    sigs = load_signals()
    df_1m = load_1m()
    overlap_start = max(sigs[0]['datetime'], df_1m.index.min())
    overlap_end   = min(sigs[-1]['datetime'], df_1m.index.max())
    sigs = [s for s in sigs if overlap_start <= s['datetime'] <= overlap_end]
    df_1m = df_1m.loc[overlap_start:overlap_end]
    print(f"  Window: {overlap_start} -> {overlap_end}")
    print(f"  Hourly signals: {len(sigs)}  |  1m candles: {len(df_1m):,}")

    # Pre-compute baseline
    print("\n  --- BASELINE simulation ---")
    base_trades, base_ret = baseline_simulate(sigs, cfg)
    base_wr  = (sum(1 for t in base_trades if t['pnl_pct'] > 0) / len(base_trades) * 100) if base_trades else 0
    base_avg = np.mean([t['pnl_pct'] for t in base_trades]) if base_trades else 0
    print(f"  Baseline: return={base_ret:+.2f}%  trades={len(base_trades)}  WR={base_wr:.0f}%  avg_pnl={base_avg:+.2f}%")
    if not base_trades:
        return

    # Diagnostic: how many wicks of each magnitude exist in baseline trade windows?
    print("\n  --- WICK PREVALENCE (in-trade 1m candles only) ---")
    in_trade_mask = pd.Series(False, index=df_1m.index)
    for t in base_trades:
        in_trade_mask.loc[pd.Timestamp(t['entry_dt']):pd.Timestamp(t['exit_dt'])] = True
    df_in = df_1m[in_trade_mask].copy()
    body_top = df_in[['open', 'close']].max(axis=1)
    df_in['wick_pct'] = (df_in['high'] - body_top) / body_top * 100.0
    for thr in WICK_THRESHOLDS_PCT:
        n_wicks = (df_in['wick_pct'] >= thr).sum()
        print(f"    wick >= {thr:.2f}%: {n_wicks:,} candles ({n_wicks / max(1,len(df_in)) * 100:.2f}%)")

    print("\n  --- WICK-REVERT TP sweep (15 variants) ---")
    print(f"  {'Variant':<32}{'Ret%':>9}{'d_vs_base':>11}{'Trades':>8}"
          f"{'WR%':>6}{'AvgPnL':>9}{'Fires':>8}{'better':>8}{'worse':>7}")
    print(f"  {'-'*32}{'-'*9}{'-'*11}{'-'*8}{'-'*6}{'-'*9}{'-'*8}{'-'*8}{'-'*7}")

    rows = []
    all_trades_rows = []
    for thr in WICK_THRESHOLDS_PCT:
        for mode in REVERT_MODES:
            new_trades, new_ret, fires, better, worse = simulate_with_tp(
                base_trades, df_1m, thr, mode)
            wr = (sum(1 for t in new_trades if t['pnl_pct'] > 0) / len(new_trades) * 100)
            avg = np.mean([t['pnl_pct'] for t in new_trades])
            delta = new_ret - base_ret
            name = f"wick={thr}%_{mode}"
            print(f"  {name:<32}{new_ret:>+8.2f}%{delta:>+10.2f}{len(new_trades):>8}"
                  f"{wr:>5.0f}%{avg:>+8.2f}%{fires:>8}{better:>8}{worse:>7}")
            rows.append({
                'variant': name, 'wick_threshold_pct': thr, 'revert_mode': mode,
                'return_pct': round(new_ret, 4), 'delta_vs_base': round(delta, 4),
                'trades': len(new_trades), 'win_rate': round(wr, 2),
                'avg_pnl': round(avg, 4),
                'tp_fires': fires, 'tp_better': better, 'tp_worse': worse,
            })
            for t in new_trades:
                t['variant'] = name
                all_trades_rows.append(t)

    rows.append({
        'variant': 'BASELINE', 'wick_threshold_pct': None, 'revert_mode': None,
        'return_pct': round(base_ret, 4), 'delta_vs_base': 0.0,
        'trades': len(base_trades), 'win_rate': round(base_wr, 2),
        'avg_pnl': round(base_avg, 4),
        'tp_fires': 0, 'tp_better': 0, 'tp_worse': 0,
    })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    pd.DataFrame(all_trades_rows).to_csv(TRADES_CSV, index=False)
    print(f"\n  Summary CSV: {OUT_CSV}")
    print(f"  Trades CSV:  {TRADES_CSV}")

    print("\n  --- TOP 5 by delta_vs_base ---")
    top = df_out[df_out['variant'] != 'BASELINE'].sort_values('delta_vs_base', ascending=False).head(5)
    for _, r in top.iterrows():
        print(f"    {r['variant']:<32}  ret={r['return_pct']:+.2f}%  "
              f"delta={r['delta_vs_base']:+.2f}pp  trades={r['trades']}  WR={r['win_rate']:.0f}%  "
              f"fires={r['tp_fires']} (better={r['tp_better']}, worse={r['tp_worse']})")
    print(f"\n    BASELINE                          ret={base_ret:+.2f}%  trades={len(base_trades)}  WR={base_wr:.0f}%")


if __name__ == '__main__':
    main()
