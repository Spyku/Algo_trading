"""Audit V6 before pushing to prod.

(1) List every skipped BUY + what price did 24h later — were the skips
    actually protective, or did we leave money on the table?
(2) Sensitivity: perturb thresholds +-1pp and +-6h cooldown, see if win-rate holds.
(3) Check for regime-boundary artifacts (rolling returns crossing bull<->bear).
"""
from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ENGINE_DIR, 'data')
SIG  = os.path.join(DATA, 'eth_sl_signals_90d.pkl')

TRADING_FEE = 0.0005  # 5 bps/leg realistic maker blend (see ed.py BACKTEST_FEE_PER_LEG)
MIN_SELL_PNL_PCT = 0.005
MAX_HOLD_HOURS = 10


def load():
    with open(SIG, 'rb') as f:
        signals = pickle.load(f)
    for s in signals:
        t = pd.Timestamp(s['datetime'])
        s['datetime'] = t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    return signals


def build_hourly(signals):
    df = pd.DataFrame([{
        'datetime': s['datetime'], 'close': s['close'],
        'signal': s['signal'], 'confidence': s['confidence'],
        'conf_threshold': s['conf_threshold'], 'regime': s.get('regime')
    } for s in signals]).sort_values('datetime').reset_index(drop=True)
    for h in (6, 10, 24, 48):
        df[f'rr_{h}h'] = (df['close'] / df['close'].shift(h) - 1.0) * 100.0
    return df


def v6_trigger(row):
    return ((pd.notna(row['rr_10h']) and row['rr_10h'] >= 5.0)
            or (pd.notna(row['rr_24h']) and row['rr_24h'] >= 7.0))


def simulate(signals, df_hourly, rr10_thr=5.0, rr24_thr=7.0, cool_hours=24):
    rr_map = df_hourly.set_index('datetime')
    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry_px = 0.0
    entry_time = None
    hold_hours = 0
    trades = []
    skipped = []
    cd = 0
    n = len(signals)
    for i, s in enumerate(signals):
        dt = s['datetime']; price = s['close']
        sig = s['signal']; conf = s['confidence']; thr = s['conf_threshold']
        rr = rr_map.loc[dt] if dt in rr_map.index else None

        if rr is not None:
            hit = ((pd.notna(rr['rr_10h']) and rr['rr_10h'] >= rr10_thr)
                   or (pd.notna(rr['rr_24h']) and rr['rr_24h'] >= rr24_thr))
            if hit:
                cd = max(cd, cool_hours)

        if sig == 'BUY' and conf >= thr and not in_pos:
            if cd > 0:
                # record the would-be entry
                fill_px = signals[i+1]['close'] if i+1 < n else price
                fill_dt = signals[i+1]['datetime'] if i+1 < n else dt
                # look 24h ahead to see if price moved up or down from skip point
                ahead_idx = i + 24
                ahead_px = signals[ahead_idx]['close'] if ahead_idx < n else np.nan
                skipped.append({
                    'skip_time': dt, 'skip_price': fill_px,
                    'price_24h_later': ahead_px,
                    'move_24h_pct': (ahead_px/fill_px - 1) * 100 if not np.isnan(ahead_px) else np.nan,
                    'rr_10h': rr['rr_10h'] if rr is not None else np.nan,
                    'rr_24h': rr['rr_24h'] if rr is not None else np.nan,
                })
            else:
                fill_px = signals[i+1]['close'] if i+1 < n else price
                fill_dt = signals[i+1]['datetime'] if i+1 < n else dt
                qty = cash * (1 - TRADING_FEE) / fill_px
                cash = 0.0; in_pos = True
                entry_px = fill_px; entry_time = fill_dt; hold_hours = 0

        elif sig == 'SELL' and in_pos:
            fill_px = signals[i+1]['close'] if i+1 < n else price
            fill_dt = signals[i+1]['datetime'] if i+1 < n else dt
            pnl_pct = (fill_px/entry_px - 1) * 100
            shield_ok = pnl_pct >= MIN_SELL_PNL_PCT * 100
            failsafe = hold_hours >= MAX_HOLD_HOURS
            if shield_ok or failsafe:
                cash = qty * fill_px * (1 - TRADING_FEE)
                trades.append({'entry_time': entry_time, 'exit_time': fill_dt,
                               'entry_px': entry_px, 'exit_px': fill_px,
                               'pnl_pct': pnl_pct,
                               'reason': 'failsafe' if (failsafe and not shield_ok) else 'shield'})
                in_pos = False; qty = 0.0; entry_px = 0.0; entry_time = None; hold_hours = 0

        if in_pos:
            hold_hours += 1
        if cd > 0:
            cd -= 1

    if in_pos:
        final_px = signals[-1]['close']
        cash = qty * final_px * (1 - TRADING_FEE)
        pnl_pct = (final_px/entry_px - 1) * 100
        trades.append({'entry_time': entry_time, 'exit_time': signals[-1]['datetime'],
                       'entry_px': entry_px, 'exit_px': final_px,
                       'pnl_pct': pnl_pct, 'reason': 'eow_flatten'})

    return (cash/1000.0 - 1) * 100, trades, skipped


def main():
    print("=" * 100)
    print("  V6 AUDIT")
    print("=" * 100)
    signals = load()
    df = build_hourly(signals)
    print(f"  signals: {len(signals)}  span: {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    for days, label in [(30, '30d'), (60, '60d'), (90, '90d')]:
        end_t = df['datetime'].iloc[-1]
        start_t = end_t - pd.Timedelta(days=days)
        w_sigs = [s for s in signals if s['datetime'] >= start_t]
        w_df   = df[df['datetime'] >= start_t].reset_index(drop=True)
        pnl, trades, skipped = simulate(w_sigs, w_df)
        print()
        print(f"  === {label}: V6 audit ===")
        print(f"  total_pnl={pnl:+.2f}%  trades={len(trades)}  skipped_buys={len(skipped)}")
        if skipped:
            sdf = pd.DataFrame(skipped)
            n = len(sdf)
            hurt = (sdf['move_24h_pct'] < 0).sum()   # skip was correct: price fell
            helped_us = (sdf['move_24h_pct'] > 0).sum()  # skip cost us: price rose
            avg_move = sdf['move_24h_pct'].mean()
            print(f"  of {n} skips: {hurt} price FELL in next 24h (skip saved us), "
                  f"{helped_us} price ROSE (skip cost us)")
            print(f"  avg price move 24h after skip: {avg_move:+.2f}%  "
                  f"(negative = skip was right on avg)")
            if label == '90d':
                print("\n  First 15 skips (90d):")
                print(sdf.head(15).to_string(index=False, formatters={
                    'skip_price': '{:.2f}'.format, 'price_24h_later': '{:.2f}'.format,
                    'move_24h_pct': '{:+.2f}'.format,
                    'rr_10h': '{:+.2f}'.format, 'rr_24h': '{:+.2f}'.format}))

    # -------- Sensitivity sweep --------
    print("\n" + "=" * 100)
    print("  SENSITIVITY: V6 with perturbed thresholds (90d window)")
    print("=" * 100)
    end_t = df['datetime'].iloc[-1]
    start_t = end_t - pd.Timedelta(days=90)
    w_sigs = [s for s in signals if s['datetime'] >= start_t]
    w_df   = df[df['datetime'] >= start_t].reset_index(drop=True)
    print(f"  {'rr10':>6}{'rr24':>6}{'cd':>6}{'PnL%':>10}{'Trades':>8}{'Skipped':>10}")
    for rr10 in (4.0, 5.0, 6.0):
        for rr24 in (6.0, 7.0, 8.0):
            for cd in (12, 24, 36):
                pnl, tr, sk = simulate(w_sigs, w_df, rr10_thr=rr10, rr24_thr=rr24, cool_hours=cd)
                marker = '  <-- V6' if (rr10, rr24, cd) == (5.0, 7.0, 24) else ''
                print(f"  {rr10:>6.1f}{rr24:>6.1f}{cd:>6d}{pnl:>+10.2f}{len(tr):>8d}{len(sk):>10d}{marker}")

    # Same sweep on 60d and 30d
    for days, label in [(60, '60d'), (30, '30d')]:
        end_t = df['datetime'].iloc[-1]
        start_t = end_t - pd.Timedelta(days=days)
        w_sigs = [s for s in signals if s['datetime'] >= start_t]
        w_df   = df[df['datetime'] >= start_t].reset_index(drop=True)
        print(f"\n  {'rr10':>6}{'rr24':>6}{'cd':>6}{'PnL%':>10}{'Trades':>8}{'Skipped':>10}   ({label})")
        for rr10 in (4.0, 5.0, 6.0):
            for rr24 in (6.0, 7.0, 8.0):
                for cd in (12, 24, 36):
                    pnl, tr, sk = simulate(w_sigs, w_df, rr10_thr=rr10, rr24_thr=rr24, cool_hours=cd)
                    marker = '  <-- V6' if (rr10, rr24, cd) == (5.0, 7.0, 24) else ''
                    print(f"  {rr10:>6.1f}{rr24:>6.1f}{cd:>6d}{pnl:>+10.2f}{len(tr):>8d}{len(sk):>10d}{marker}")


if __name__ == '__main__':
    main()
