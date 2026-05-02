"""V6 fine-tuning sweep: pick best (Hshort, Hlong, Tshort, Tlong, cooldown).

Structure is FIXED to V6's winning form — OR gate on two rolling returns:
  block BUYs for C hours when rr(Hshort) >= Tshort OR rr(Hlong) >= Tlong.
We're not exploring modes (AND / GATE / EXIT) anymore. We're fine-tuning the
horizons / thresholds / cooldown length INSIDE the current crisis regime.

Windows: two non-overlapping 30-day halves from the tail of the cache
  (H1 = days 0-30 back from today, H2 = days 30-60 back). 60d combined reported
  as reference only. No 90d here — 90d straddles a regime change we want to
  let the detector handle, not the BUY gate.

Scoring: PnL%. Winner must beat the V0 (no-gate) baseline PnL on BOTH H1 AND H2.
Reported (not scored): trades + skipped-BUYs per window, for eyeballing churn.

Grid:
  horizon pairs : all 15 (Hshort < Hlong) from {10, 12, 14, 16, 18, 24}
  thresholds    : horizon-scaled
                  10h, 12h      -> {3, 4, 5, 6}
                  14h, 16h      -> {4, 5, 6, 7}
                  18h           -> {5, 6, 7, 8}
                  24h           -> {5, 6, 7, 8, 9}
  cooldowns     : {10, 12, 18, 24}

Writes audit_v6_v2_summary.csv with all configs + prints robust winners.
"""
from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ENGINE_DIR, 'data')
SIG_90D    = os.path.join(DATA_DIR, 'eth_sl_signals_90d.pkl')

TRADING_FEE      = 0.0005  # 5 bps/leg realistic maker blend (see ed.py BACKTEST_FEE_PER_LEG)
MIN_SELL_PNL_PCT = 0.005
MAX_HOLD_HOURS   = 10

HORIZONS = [10, 12, 14, 16, 18, 24]
THRESHOLDS_BY_H = {
    10: [3.0, 4.0, 5.0, 6.0],
    12: [3.0, 4.0, 5.0, 6.0],
    14: [4.0, 5.0, 6.0, 7.0],
    16: [4.0, 5.0, 6.0, 7.0],
    18: [5.0, 6.0, 7.0, 8.0],
    24: [5.0, 6.0, 7.0, 8.0, 9.0],
}
CD_GRID = [10, 12, 18, 24]


def load_signals():
    with open(SIG_90D, 'rb') as f:
        signals = pickle.load(f)
    for s in signals:
        t = pd.Timestamp(s['datetime'])
        s['datetime'] = t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    return signals


def build_hourly(signals):
    rows = [{'datetime': s['datetime'], 'close': s['close']} for s in signals]
    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    for h in HORIZONS:
        df[f'rr_{h}h'] = (df['close'] / df['close'].shift(h) - 1.0) * 100.0
    return df


def simulate(signals, df_hourly, h_short, h_long, t_short, t_long, cd_hours):
    rr_map = df_hourly.set_index('datetime')
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

    col_s = f'rr_{h_short}h'
    col_l = f'rr_{h_long}h'

    for i, s in enumerate(signals):
        dt = s['datetime']; price = s['close']
        sig = s['signal']; conf = s['confidence']; thr = s['conf_threshold']
        rr_row = rr_map.loc[dt] if dt in rr_map.index else None

        hit = False
        if rr_row is not None and cd_hours > 0:
            rs = rr_row[col_s]; rl = rr_row[col_l]
            hit_s = pd.notna(rs) and rs >= t_short
            hit_l = pd.notna(rl) and rl >= t_long
            hit = bool(hit_s or hit_l)
        if hit:
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
    return dict(trades=trades, pnl_pct=total_pnl, dd_pct=max_dd, skipped=skipped)


def baseline(signals, df_hourly):
    """V0: no gate at all — pass cd_hours=0 so 'hit' never fires."""
    return simulate(signals, df_hourly, h_short=10, h_long=24,
                    t_short=9999, t_long=9999, cd_hours=0)


def slice_window(signals, df_hourly, start_days_back, end_days_back):
    """Half-open window: (end_t - start_days_back) <= dt < (end_t - end_days_back).
    With start_days_back > end_days_back (older > newer)."""
    end_t = df_hourly['datetime'].iloc[-1]
    t_lo = end_t - pd.Timedelta(days=start_days_back)
    t_hi = end_t - pd.Timedelta(days=end_days_back)
    w_sigs = [s for s in signals if t_lo <= s['datetime'] < t_hi]
    w_df   = df_hourly[(df_hourly['datetime'] >= t_lo) &
                       (df_hourly['datetime'] <  t_hi)].reset_index(drop=True)
    return w_sigs, w_df


def slice_tail(signals, df_hourly, days):
    end_t = df_hourly['datetime'].iloc[-1]
    start_t = end_t - pd.Timedelta(days=days)
    w_sigs = [s for s in signals if s['datetime'] >= start_t]
    w_df   = df_hourly[df_hourly['datetime'] >= start_t].reset_index(drop=True)
    return w_sigs, w_df


def main():
    print("=" * 100)
    print("  V6 FINE-TUNING SWEEP (OR gate, horizons x thresholds x cooldown)")
    print("=" * 100)
    signals = load_signals()
    df = build_hourly(signals)
    print(f"  signals={len(signals)}  span: {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    # Windows: H1 = newest 30d, H2 = prior 30d, REF = newest 60d
    win_h1  = slice_window(signals, df, 30, 0)   # days 0-30 back
    win_h2  = slice_window(signals, df, 60, 30)  # days 30-60 back
    win_60  = slice_tail(signals, df, 60)
    windows = [('H1_30d', win_h1), ('H2_30d', win_h2), ('REF_60d', win_60)]

    for label, (ws, wd) in windows:
        if len(wd):
            print(f"  {label}: {len(ws)} signals  {wd['datetime'].iloc[0]} -> {wd['datetime'].iloc[-1]}")
        else:
            print(f"  {label}: EMPTY")

    # Per-window baseline
    base = {}
    for label, (ws, wd) in windows:
        b = baseline(ws, wd)
        base[label] = b
        print(f"  baseline {label}: PnL={b['pnl_pct']:+.2f}%  DD={b['dd_pct']:.2f}%  trades={b['trades']}")

    # Build horizon pairs (Hshort < Hlong)
    pairs = [(a, b) for i, a in enumerate(HORIZONS) for b in HORIZONS[i + 1:]]
    total_configs = sum(len(THRESHOLDS_BY_H[a]) * len(THRESHOLDS_BY_H[b]) for a, b in pairs) * len(CD_GRID)
    print(f"\n  Sweeping {len(pairs)} horizon pairs x thresholds x {len(CD_GRID)} cooldowns "
          f"= {total_configs} configs x {len(windows)} windows")

    rows = []
    for (h_s, h_l) in pairs:
        for t_s in THRESHOLDS_BY_H[h_s]:
            for t_l in THRESHOLDS_BY_H[h_l]:
                for cd in CD_GRID:
                    row = {'h_short': h_s, 'h_long': h_l,
                           't_short': t_s, 't_long': t_l, 'cd': cd}
                    beats = 0
                    for label, (ws, wd) in windows:
                        r = simulate(ws, wd, h_s, h_l, t_s, t_l, cd)
                        row[f'pnl_{label}'] = r['pnl_pct']
                        row[f'dd_{label}']  = r['dd_pct']
                        row[f'tr_{label}']  = r['trades']
                        row[f'sk_{label}']  = r['skipped']
                        if label in ('H1_30d', 'H2_30d') and r['pnl_pct'] > base[label]['pnl_pct']:
                            beats += 1
                    row['beats_both_halves'] = (beats == 2)
                    row['avg_pnl_halves'] = (row['pnl_H1_30d'] + row['pnl_H2_30d']) / 2.0
                    rows.append(row)

    out = pd.DataFrame(rows)
    out_path = os.path.join(ENGINE_DIR, 'audit_v6_v2_summary.csv')
    out.to_csv(out_path, index=False)
    print(f"\n  Wrote {out_path}  ({len(out)} configs)")

    # --- Robust winners: beat baseline on both halves ---
    print("\n" + "=" * 100)
    print("  ROBUST WINNERS — beat V0 baseline PnL on BOTH H1_30d and H2_30d")
    print("=" * 100)
    robust = out[out['beats_both_halves']].copy().sort_values('avg_pnl_halves', ascending=False)
    cols = ['h_short', 'h_long', 't_short', 't_long', 'cd',
            'pnl_H1_30d', 'pnl_H2_30d', 'avg_pnl_halves', 'pnl_REF_60d',
            'dd_H1_30d', 'dd_H2_30d', 'dd_REF_60d',
            'tr_H1_30d', 'tr_H2_30d', 'sk_H1_30d', 'sk_H2_30d']
    if len(robust) == 0:
        print("  NONE — no config beats baseline on both halves.")
    else:
        num_fmt = {c: '{:+.2f}'.format for c in cols
                   if c.startswith(('pnl_', 'dd_', 'avg_'))}
        print(f"  ({len(robust)} configs beat baseline on both halves — showing top 25)")
        print(robust[cols].head(25).to_string(index=False, formatters=num_fmt))

    # --- Top 15 by avg_pnl_halves overall ---
    print("\n" + "=" * 100)
    print("  TOP 15 BY avg_pnl_halves (regardless of beat-baseline)")
    print("=" * 100)
    top = out.sort_values('avg_pnl_halves', ascending=False).head(15)
    num_fmt = {c: '{:+.2f}'.format for c in cols
               if c.startswith(('pnl_', 'dd_', 'avg_'))}
    print(top[cols].to_string(index=False, formatters=num_fmt))

    # --- V6 reference row (Hshort=10, Hlong=24, Ts=5, Tl=7, cd=24) ---
    v6 = out[(out['h_short'] == 10) & (out['h_long'] == 24) &
             (out['t_short'] == 5.0) & (out['t_long'] == 7.0) & (out['cd'] == 24)]
    if len(v6):
        print("\n  V6 reference row (rr10>=5 OR rr24>=7, cd=24):")
        print(v6[cols].to_string(index=False, formatters=num_fmt))


if __name__ == '__main__':
    main()
