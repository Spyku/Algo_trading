"""3-way comparison: BUY-HOLD vs current models (V0, no gate) vs current models + V7 gate.

V7 gate winner from audit_v6_v3 sweep:
  block BUYs for 30h when rr_8h >= 3.0% OR rr_36h >= 5.5%

Reports last 7d and last 30d. Uses the same signal cache + simulator as v3.
"""
from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
SIG_90D    = os.path.join(ENGINE_DIR, 'data', 'eth_sl_signals_90d.pkl')

TRADING_FEE      = 0.0005  # 5 bps/leg realistic maker blend (see ed.py BACKTEST_FEE_PER_LEG)
MIN_SELL_PNL_PCT = 0.005
MAX_HOLD_HOURS   = 10

# V7 winner (from audit_v6_v3 sweep, all 4 scoring views agreed)
V7 = dict(h_short=8, h_long=36, t_short=3.0, t_long=5.5, cd_hours=30)


def load_signals():
    with open(SIG_90D, 'rb') as f:
        signals = pickle.load(f)
    for s in signals:
        t = pd.Timestamp(s['datetime'])
        s['datetime'] = t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    return signals


def build_hourly(signals, horizons):
    rows = [{'datetime': s['datetime'], 'close': s['close']} for s in signals]
    df = pd.DataFrame(rows).sort_values('datetime').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    for h in horizons:
        df[f'rr_{h}h'] = (df['close'] / df['close'].shift(h) - 1.0) * 100.0
    return df


def align_rr(signals, df, horizons):
    rr_map = df.set_index('datetime')
    out = {}
    for h in horizons:
        col = f'rr_{h}h'
        arr = np.empty(len(signals), dtype=float)
        for i, s in enumerate(signals):
            arr[i] = rr_map.at[s['datetime'], col] if s['datetime'] in rr_map.index else np.nan
        out[h] = arr
    return out


def simulate(signals, rr_arrs, h_short, h_long, t_short, t_long, cd_hours):
    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry_px = 0.0
    hold_hours = 0
    trades = 0
    skipped = 0
    cd = 0
    equity_curve = [1000.0]
    n = len(signals)
    rs_arr = rr_arrs[h_short] if h_short in rr_arrs else None
    rl_arr = rr_arrs[h_long]  if h_long  in rr_arrs else None

    for i in range(n):
        s = signals[i]
        price = s['close']
        sig = s['signal']; conf = s['confidence']; thr = s['conf_threshold']

        if cd_hours > 0 and rs_arr is not None and rl_arr is not None:
            rs = rs_arr[i]; rl = rl_arr[i]
            hit_s = (rs == rs) and rs >= t_short
            hit_l = (rl == rl) and rl >= t_long
            if hit_s or hit_l:
                cd = max(cd, cd_hours)

        equity = cash + qty * price if in_pos else cash
        equity_curve.append(equity)

        if sig == 'BUY' and conf >= thr and not in_pos:
            if cd > 0:
                skipped += 1
            else:
                fill_px = signals[i + 1]['close'] if i + 1 < n else price
                qty = cash * (1 - TRADING_FEE) / fill_px
                cash = 0.0
                in_pos = True
                entry_px = fill_px
                hold_hours = 0

        elif sig == 'SELL' and in_pos:
            fill_px = signals[i + 1]['close'] if i + 1 < n else price
            cur_pnl_pct = (fill_px / entry_px - 1.0) * 100.0
            shield_ok = cur_pnl_pct >= (MIN_SELL_PNL_PCT * 100.0)
            failsafe = hold_hours >= MAX_HOLD_HOURS
            if shield_ok or failsafe:
                cash = qty * fill_px * (1 - TRADING_FEE)
                trades += 1
                in_pos = False
                qty = 0.0
                entry_px = 0.0
                hold_hours = 0

        if in_pos:
            hold_hours += 1
        if cd > 0:
            cd -= 1

    if in_pos:
        final_px = signals[-1]['close']
        cash = qty * final_px * (1 - TRADING_FEE)
        trades += 1

    total_pnl = (cash / 1000.0 - 1.0) * 100.0
    ec = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / peak
    max_dd = float(dd.min()) * 100.0 if len(dd) else 0.0
    return dict(pnl_pct=total_pnl, dd_pct=max_dd, trades=trades, skipped=skipped)


def buy_and_hold(signals):
    """Open at first close, close at last close, two fees applied."""
    if len(signals) < 2:
        return dict(pnl_pct=0.0, dd_pct=0.0, trades=0, skipped=0)
    p0 = signals[0]['close']
    pN = signals[-1]['close']
    qty = 1000.0 * (1 - TRADING_FEE) / p0
    closes = np.array([s['close'] for s in signals], dtype=float)
    equity = qty * closes  # marked-to-market through hold (no fee until exit)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min()) * 100.0 if len(dd) else 0.0
    final_cash = qty * pN * (1 - TRADING_FEE)
    pnl = (final_cash / 1000.0 - 1.0) * 100.0
    return dict(pnl_pct=pnl, dd_pct=max_dd, trades=1, skipped=0)


def slice_tail(signals, df, days):
    end_t = df['datetime'].iloc[-1]
    start_t = end_t - pd.Timedelta(days=days)
    w_sigs = [s for s in signals if s['datetime'] >= start_t]
    w_df   = df[df['datetime'] >= start_t].reset_index(drop=True)
    return w_sigs, w_df


def main():
    print("=" * 80)
    print("  3-WAY COMPARISON: BUY-HOLD vs MODELS-ONLY vs MODELS+V7")
    print(f"  V7 = block BUY 30h when rr_{V7['h_short']}h >= {V7['t_short']}% "
          f"OR rr_{V7['h_long']}h >= {V7['t_long']}%")
    print("=" * 80)

    signals = load_signals()
    horizons = sorted({V7['h_short'], V7['h_long']})
    df = build_hourly(signals, horizons)
    print(f"  cache: {len(signals)} signals  "
          f"{df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    for label, days in [('LAST 7 DAYS', 7), ('LAST 30 DAYS', 30)]:
        w_sigs, w_df = slice_tail(signals, df, days)
        if not w_sigs:
            print(f"\n  {label}: empty slice"); continue
        rr_arrs = align_rr(w_sigs, w_df, horizons)

        bh   = buy_and_hold(w_sigs)
        v0   = simulate(w_sigs, {}, h_short=8, h_long=36,
                        t_short=9999, t_long=9999, cd_hours=0)  # gate disabled
        v7   = simulate(w_sigs, rr_arrs,
                        h_short=V7['h_short'], h_long=V7['h_long'],
                        t_short=V7['t_short'], t_long=V7['t_long'],
                        cd_hours=V7['cd_hours'])

        print(f"\n  === {label}  ({len(w_sigs)} hourly signals, "
              f"{w_df['datetime'].iloc[0]} -> {w_df['datetime'].iloc[-1]}) ===")
        print(f"  {'Strategy':<28}{'PnL%':>10}{'MaxDD%':>10}{'Trades':>9}{'Skipped':>9}")
        print("  " + "-" * 66)
        print(f"  {'1) Buy & Hold':<28}{bh['pnl_pct']:>+10.2f}{bh['dd_pct']:>+10.2f}"
              f"{bh['trades']:>9}{bh['skipped']:>9}")
        print(f"  {'2) Models, no gate (V0)':<28}{v0['pnl_pct']:>+10.2f}{v0['dd_pct']:>+10.2f}"
              f"{v0['trades']:>9}{v0['skipped']:>9}")
        print(f"  {'3) Models + V7 gate':<28}{v7['pnl_pct']:>+10.2f}{v7['dd_pct']:>+10.2f}"
              f"{v7['trades']:>9}{v7['skipped']:>9}")
        print(f"  {'V7 vs V0 (Δ)':<28}{v7['pnl_pct']-v0['pnl_pct']:>+10.2f}"
              f"{v7['dd_pct']-v0['dd_pct']:>+10.2f}"
              f"{v7['trades']-v0['trades']:>+9}{v7['skipped']-v0['skipped']:>+9}")
        print(f"  {'V7 vs B&H (Δ)':<28}{v7['pnl_pct']-bh['pnl_pct']:>+10.2f}"
              f"{v7['dd_pct']-bh['dd_pct']:>+10.2f}"
              f"{v7['trades']-bh['trades']:>+9}{'-':>9}")


if __name__ == '__main__':
    main()
