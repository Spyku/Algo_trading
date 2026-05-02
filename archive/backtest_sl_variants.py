"""Backtest 8 stop-loss / profit-lock variants on ETH over cached 30-day window.

Reuses cached artifacts:
  - data/eth_5m_backtest.csv (5-min OHLC)
  - data/eth_sl_signals.pkl  (hourly merged signals, regime+conf tagged)

Does NOT re-download, re-infer, or touch production files.

Variants (entry = next-hour open after BUY w/ regime+conf gate):
  A  No SL (baseline)            — Shield (>=0.5%) or failsafe (>=10h)
  B  Disaster -7%                — A + hard SL at entry*0.93
  C  Disaster -10%               — A + hard SL at entry*0.90
  D  Profit-lock +0.5/+0.22      — arm at entry*1.005, floor at entry*1.0022
  E  D + Disaster -7%
  F  Profit-lock +0.3/+0.15      — arm at entry*1.003, floor at entry*1.0015
  G  Trailing 1% from HWM        — arm at entry*1.005, SL = HWM*0.99 (ratchet up)
  H  G + Disaster -7%

Interpretations (documented):
  * All SL checks run on 5-min LOW (for downside) and 5-min HIGH (for arming
    profit-lock / updating HWM). First trigger wins within a given 5m bar;
    when both disaster-SL and profit-lock floors are armed, we evaluate the
    disaster level first because -7% losses happen on the same bar as deep
    wicks. (In practice they're mutually exclusive — profit-lock requires
    price already +0.5% above entry, so a -7% wick on the same bar is
    essentially impossible; ordering doesn't matter.)
  * SL fill assumed at exact SL price (no slippage). Hourly shield/failsafe
    only evaluated at the top of the hour (matching prod).
  * Profit-lock and trailing SL can only SELL — they don't override a BUY
    signal. Re-entry can happen on the next BUY hour after exit.

Run:  python backtest_sl_variants.py
"""
from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ENGINE_DIR, 'data')

SIG_CACHE = os.path.join(DATA_DIR, 'eth_sl_signals.pkl')
FIVE_MIN  = os.path.join(DATA_DIR, 'eth_5m_backtest.csv')

TRADING_FEE        = 0.0005          # 5 bps/leg realistic maker blend (see ed.py BACKTEST_FEE_PER_LEG)
MIN_SELL_PNL_PCT   = 0.005           # 0.5%
MAX_HOLD_HOURS     = 10

OUT_SUMMARY = os.path.join(ENGINE_DIR, 'backtest_sl_variants_summary.csv')
OUT_TRADES  = os.path.join(ENGINE_DIR, 'backtest_sl_variants_trades.csv')


# ----------------------------------------------------------------
# Variant configs
# ----------------------------------------------------------------
# Each variant is a dict describing which exit mechanics are active.
#   disaster_pct:  None or positive float (e.g. 7.0 means SL at entry*0.93)
#   pl_arm:        None or ratio (e.g. 1.005 means arm when any HIGH >= entry*1.005)
#   pl_floor:      ratio once armed (e.g. 1.0022)
#   trailing_arm:  None or ratio to arm trailing SL
#   trailing_gap:  SL = running_max_high * (1 - trailing_gap), e.g. 0.01 = 1%
VARIANTS = [
    ('A', 'No SL (baseline)',        dict(disaster_pct=None, pl_arm=None,  pl_floor=None,   trailing_arm=None,  trailing_gap=None)),
    ('B', 'Disaster -7%',            dict(disaster_pct=7.0,  pl_arm=None,  pl_floor=None,   trailing_arm=None,  trailing_gap=None)),
    ('C', 'Disaster -10%',           dict(disaster_pct=10.0, pl_arm=None,  pl_floor=None,   trailing_arm=None,  trailing_gap=None)),
    ('D', 'Profit-lock +0.5/+0.22',  dict(disaster_pct=None, pl_arm=1.005, pl_floor=1.0022, trailing_arm=None,  trailing_gap=None)),
    ('E', 'D + Disaster -7%',        dict(disaster_pct=7.0,  pl_arm=1.005, pl_floor=1.0022, trailing_arm=None,  trailing_gap=None)),
    ('F', 'Profit-lock +0.3/+0.15',  dict(disaster_pct=None, pl_arm=1.003, pl_floor=1.0015, trailing_arm=None,  trailing_gap=None)),
    ('G', 'Trailing 1% from HWM',    dict(disaster_pct=None, pl_arm=None,  pl_floor=None,   trailing_arm=1.005, trailing_gap=0.01)),
    ('H', 'G + Disaster -7%',        dict(disaster_pct=7.0,  pl_arm=None,  pl_floor=None,   trailing_arm=1.005, trailing_gap=0.01)),
]


# ----------------------------------------------------------------
# Data load
# ----------------------------------------------------------------
def load_caches():
    if not os.path.exists(SIG_CACHE):
        print(f"ERROR: signal cache missing: {SIG_CACHE}")
        print("Aborting — re-inference disallowed per spec.")
        sys.exit(2)
    if not os.path.exists(FIVE_MIN):
        print(f"ERROR: 5m candle cache missing: {FIVE_MIN}")
        print("Aborting — re-download disallowed per spec.")
        sys.exit(2)
    with open(SIG_CACHE, 'rb') as f:
        signals = pickle.load(f)
    df5 = pd.read_csv(FIVE_MIN)
    df5['datetime'] = pd.to_datetime(df5['datetime'], utc=True)
    df5 = df5.set_index('datetime').sort_index()
    print(f"  loaded {len(signals)} signals, {len(df5)} 5m bars")
    return signals, df5


def five_min_slice(df5: pd.DataFrame, hour_start: pd.Timestamp) -> pd.DataFrame:
    if hour_start.tzinfo is None:
        hour_start = hour_start.tz_localize('UTC')
    hour_end = hour_start + pd.Timedelta(hours=1)
    return df5.loc[(df5.index >= hour_start) & (df5.index < hour_end)]


# ----------------------------------------------------------------
# Core simulation
# ----------------------------------------------------------------
def simulate(signals: list, df5: pd.DataFrame, cfg: dict) -> dict:
    """Simulate one variant. cfg keys defined at top of file."""
    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry_px = 0.0
    entry_time = None
    hold_hours = 0
    # per-trade state for within-hour SL logic
    pl_armed = False
    trail_armed = False
    running_max_high = 0.0
    current_sl_px = None        # active SL price (from pl_floor or trailing)

    trades = []
    sl_fires = 0
    shield_fires = 0
    failsafe_fires = 0
    equity_curve = [1000.0]

    n = len(signals)
    for i, s in enumerate(signals):
        dt = s['datetime']
        price = s['close']
        sig = s['signal']
        conf = s['confidence']
        thr = s['conf_threshold']

        equity = cash + qty * price if in_pos else cash
        equity_curve.append(equity)

        # ----- intra-hour SL evaluation on 5m bars -----
        if in_pos:
            bars = five_min_slice(df5, dt)
            exit_reason = None
            exit_px = None
            exit_ts = None

            disaster_px = (entry_px * (1 - cfg['disaster_pct'] / 100.0)
                           if cfg['disaster_pct'] is not None else None)

            for ts, row in bars.iterrows():
                bar_high = row['high']
                bar_low = row['low']

                # 1) disaster SL — check first (deep loss wicks)
                if disaster_px is not None and bar_low <= disaster_px:
                    exit_reason = 'sl_disaster'
                    exit_px = disaster_px
                    exit_ts = ts
                    break

                # 2) profit-lock static floor (D/E/F)
                if cfg['pl_arm'] is not None:
                    if not pl_armed and bar_high >= entry_px * cfg['pl_arm']:
                        pl_armed = True
                        current_sl_px = entry_px * cfg['pl_floor']
                    if pl_armed and current_sl_px is not None and bar_low <= current_sl_px:
                        exit_reason = 'sl_profit_lock'
                        exit_px = current_sl_px
                        exit_ts = ts
                        break

                # 3) trailing SL (G/H)
                if cfg['trailing_arm'] is not None:
                    if bar_high > running_max_high:
                        running_max_high = bar_high
                    if not trail_armed and running_max_high >= entry_px * cfg['trailing_arm']:
                        trail_armed = True
                    if trail_armed:
                        new_sl = running_max_high * (1 - cfg['trailing_gap'])
                        # ratchet up, never down
                        if current_sl_px is None or new_sl > current_sl_px:
                            current_sl_px = new_sl
                        if bar_low <= current_sl_px:
                            exit_reason = 'sl_trailing'
                            exit_px = current_sl_px
                            exit_ts = ts
                            break

            if exit_reason is not None:
                cash = qty * exit_px * (1 - TRADING_FEE)
                pnl_pct = (exit_px / entry_px - 1.0) * 100.0
                trades.append({
                    'entry_time': entry_time, 'entry_price': entry_px,
                    'exit_time': exit_ts, 'exit_price': exit_px,
                    'pnl_pct': pnl_pct, 'hold_hours': hold_hours,
                    'exit_reason': exit_reason,
                })
                sl_fires += 1
                in_pos = False
                qty = 0.0
                entry_px = 0.0
                entry_time = None
                hold_hours = 0
                pl_armed = False
                trail_armed = False
                running_max_high = 0.0
                current_sl_px = None
                # fall through — may re-buy this hour

        # ----- hourly decision -----
        if sig == 'BUY' and conf >= thr and not in_pos:
            if i + 1 < n:
                nxt = signals[i + 1]
                fill_dt = nxt['datetime']
                fill_px = nxt['close']
            else:
                fill_dt = dt
                fill_px = price
            qty = cash * (1 - TRADING_FEE) / fill_px
            cash = 0.0
            in_pos = True
            entry_px = fill_px
            entry_time = fill_dt
            hold_hours = 0
            pl_armed = False
            trail_armed = False
            running_max_high = fill_px        # HWM starts at entry
            current_sl_px = None

        elif sig == 'SELL' and in_pos:
            if i + 1 < n:
                nxt = signals[i + 1]
                fill_dt = nxt['datetime']
                fill_px = nxt['close']
            else:
                fill_dt = dt
                fill_px = price
            cur_pnl_pct = (fill_px / entry_px - 1.0) * 100.0
            shield_ok = cur_pnl_pct >= (MIN_SELL_PNL_PCT * 100.0)
            failsafe = hold_hours >= MAX_HOLD_HOURS
            if shield_ok or failsafe:
                cash = qty * fill_px * (1 - TRADING_FEE)
                reason = 'failsafe' if (failsafe and not shield_ok) else 'shield_release'
                if reason == 'failsafe':
                    failsafe_fires += 1
                else:
                    shield_fires += 1
                trades.append({
                    'entry_time': entry_time, 'entry_price': entry_px,
                    'exit_time': fill_dt, 'exit_price': fill_px,
                    'pnl_pct': cur_pnl_pct, 'hold_hours': hold_hours + 1,
                    'exit_reason': reason,
                })
                in_pos = False
                qty = 0.0
                entry_px = 0.0
                entry_time = None
                hold_hours = 0
                pl_armed = False
                trail_armed = False
                running_max_high = 0.0
                current_sl_px = None

        if in_pos:
            hold_hours += 1

    # close out open pos at end
    if in_pos:
        final_px = signals[-1]['close']
        cash = qty * final_px * (1 - TRADING_FEE)
        pnl_pct = (final_px / entry_px - 1.0) * 100.0
        trades.append({
            'entry_time': entry_time, 'entry_price': entry_px,
            'exit_time': signals[-1]['datetime'], 'exit_price': final_px,
            'pnl_pct': pnl_pct, 'hold_hours': hold_hours,
            'exit_reason': 'model_sell',
        })

    total_pnl = (cash / 1000.0 - 1.0) * 100.0
    wins = sum(1 for t in trades if t['pnl_pct'] > 0)
    ec = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / peak
    max_dd = float(dd.min()) * 100.0 if len(dd) else 0.0

    return {
        'trades': trades,
        'n_trades': len(trades),
        'wins': wins,
        'total_pnl_pct': total_pnl,
        'max_dd_pct': max_dd,
        'sl_fires': sl_fires,
        'shield_fires': shield_fires,
        'failsafe_fires': failsafe_fires,
    }


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    print("=" * 92)
    print("  ETH SL-variants backtest (30d cached window, 5m resolution)")
    print("=" * 92)
    signals, df5 = load_caches()
    print(f"  window: {signals[0]['datetime']}  ->  {signals[-1]['datetime']}")
    print()

    results = []
    all_trades_rows = []
    for letter, name, cfg in VARIANTS:
        r = simulate(signals, df5, cfg)
        r['variant'] = letter
        r['name'] = name
        results.append(r)
        for t in r['trades']:
            all_trades_rows.append({
                'variant': letter,
                'entry_time': t['entry_time'],
                'entry_price': t['entry_price'],
                'exit_time': t['exit_time'],
                'exit_price': t['exit_price'],
                'pnl_pct': t['pnl_pct'],
                'hold_hours': t['hold_hours'],
                'exit_reason': t['exit_reason'],
            })

    # ---- main summary ----
    hdr = (f"  {'Variant':<36}{'Trades':>8}{'Wins':>6}"
           f"{'Total PnL%':>12}{'Max DD%':>10}"
           f"{'SL fires':>10}{'Shield':>8}{'Failsafe':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    summary_rows = []
    for r in results:
        label = f"{r['variant']}  {r['name']}"
        print(f"  {label:<36}{r['n_trades']:>8}{r['wins']:>6}"
              f"{r['total_pnl_pct']:>+12.2f}{r['max_dd_pct']:>+10.2f}"
              f"{r['sl_fires']:>10}{r['shield_fires']:>8}{r['failsafe_fires']:>10}")
        summary_rows.append({
            'variant': r['variant'],
            'name': r['name'],
            'trades': r['n_trades'],
            'wins': r['wins'],
            'total_pnl_pct': round(r['total_pnl_pct'], 4),
            'max_dd_pct': round(r['max_dd_pct'], 4),
            'sl_fires': r['sl_fires'],
            'shield_fires': r['shield_fires'],
            'failsafe_fires': r['failsafe_fires'],
        })

    # ---- save ----
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    pd.DataFrame(all_trades_rows).to_csv(OUT_TRADES, index=False)
    print()
    print(f"  wrote {OUT_SUMMARY}")
    print(f"  wrote {OUT_TRADES}")

    # ---- verdict helpers ----
    best_pnl = max(results, key=lambda r: r['total_pnl_pct'])
    best_dd = max(results, key=lambda r: r['max_dd_pct'])       # max_dd is negative -> "max" = least negative
    def risk_adj(r):
        if r['max_dd_pct'] == 0:
            return float('inf') if r['total_pnl_pct'] > 0 else 0.0
        return r['total_pnl_pct'] / abs(r['max_dd_pct'])
    best_ra = max(results, key=risk_adj)

    print()
    print("  --- quick take ---")
    print(f"  best total PnL:      {best_pnl['variant']} {best_pnl['name']}  "
          f"({best_pnl['total_pnl_pct']:+.2f}%)")
    print(f"  best (shallowest) DD:{best_dd['variant']} {best_dd['name']}  "
          f"({best_dd['max_dd_pct']:+.2f}%)")
    print(f"  best risk-adjusted:  {best_ra['variant']} {best_ra['name']}  "
          f"(PnL/|DD|={risk_adj(best_ra):.3f})")


if __name__ == '__main__':
    main()
