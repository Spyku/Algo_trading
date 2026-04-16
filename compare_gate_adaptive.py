"""One-off: does lifting the V7 cooldown early, when price reverts to pre-rally level,
improve PnL vs the fixed 30h window?

Variants compared on the full 90d signal cache:
  V0    no gate
  V7    fixed 30h cooldown (production)
  V7a   lift early if close <= close[trigger_time - h_short]   (lenient; 8h-ago base)
  V7b   lift early if close <= close[trigger_time - h_long]    (strict; 36h-ago base)
  V7c   lift early if close <= min(base_short, base_long)       (must retrace further of the two)
  V7d   lift early on X% retrace from trigger-time peak         (X in {1.0, 2.0, 3.0})
"""
from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
SIG_90D    = os.path.join(ENGINE_DIR, 'data', 'eth_sl_signals_90d.pkl')

TRADING_FEE      = 0.0011
MIN_SELL_PNL_PCT = 0.5   # matches production
MAX_HOLD_HOURS   = 10

V7 = dict(h_short=8, h_long=36, t_short=3.0, t_long=5.5, cd_hours=30)


def load_signals():
    with open(SIG_90D, 'rb') as f:
        signals = pickle.load(f)
    for s in signals:
        t = pd.Timestamp(s['datetime'])
        s['datetime'] = t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    return signals


def simulate(signals, *, mode, retrace_pct=None):
    """mode ∈ {'none','fixed','adaptive_short','adaptive_long','adaptive_both','retrace'}"""
    h_s, h_l = V7['h_short'], V7['h_long']
    t_s, t_l = V7['t_short'], V7['t_long']
    cd_h     = V7['cd_hours']
    closes   = np.array([s['close'] for s in signals], dtype=float)
    n        = len(signals)

    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry_px = 0.0
    hold_hours = 0
    trades = 0
    skipped = 0

    # cooldown state
    cd_remaining = 0     # hours left in fixed window
    cd_base = None       # price to revert to in order to lift early (adaptive modes)
    cd_peak = None       # trigger-time peak (for retrace mode)
    lifted_early = 0
    expired_normally = 0
    cd_active_any = 0

    for i in range(n):
        price = closes[i]
        sig = signals[i]['signal']; conf = signals[i]['confidence']; thr = signals[i]['conf_threshold']

        # ----- trigger detection (runs every tick) -----
        if mode != 'none' and i >= h_l:
            rs = (closes[i] / closes[i - h_s] - 1.0) * 100.0
            rl = (closes[i] / closes[i - h_l] - 1.0) * 100.0
            hit_s = rs >= t_s
            hit_l = rl >= t_l
            if hit_s or hit_l:
                cd_remaining = max(cd_remaining, cd_h)
                base_s = closes[i - h_s]
                base_l = closes[i - h_l]
                if mode == 'adaptive_short':
                    cd_base = base_s if hit_s else base_l
                elif mode == 'adaptive_long':
                    cd_base = base_l if hit_l else base_s
                elif mode == 'adaptive_both':
                    cd_base = min(base_s, base_l)
                elif mode == 'retrace':
                    cd_peak = price
                else:  # fixed
                    cd_base = None
                    cd_peak = None

        # ----- early lift check -----
        if cd_remaining > 0:
            cd_active_any += 1
            if mode in ('adaptive_short', 'adaptive_long', 'adaptive_both') and cd_base is not None:
                if price <= cd_base:
                    cd_remaining = 0
                    cd_base = None
                    lifted_early += 1
            elif mode == 'retrace' and cd_peak is not None and retrace_pct is not None:
                if price <= cd_peak * (1.0 - retrace_pct / 100.0):
                    cd_remaining = 0
                    cd_peak = None
                    lifted_early += 1

        # ----- trading -----
        if sig == 'BUY' and conf >= thr and not in_pos:
            if cd_remaining > 0:
                skipped += 1
            else:
                fill_px = closes[i + 1] if i + 1 < n else price
                qty = cash * (1 - TRADING_FEE) / fill_px
                cash = 0.0
                in_pos = True
                entry_px = fill_px
                hold_hours = 0
        elif sig == 'SELL' and in_pos:
            fill_px = closes[i + 1] if i + 1 < n else price
            cur_pnl_pct = (fill_px / entry_px - 1.0) * 100.0
            if cur_pnl_pct >= MIN_SELL_PNL_PCT or hold_hours >= MAX_HOLD_HOURS:
                cash = qty * fill_px * (1 - TRADING_FEE)
                trades += 1
                in_pos = False
                qty = 0.0
                hold_hours = 0

        if in_pos:
            hold_hours += 1
        if cd_remaining > 0:
            cd_remaining -= 1
            if cd_remaining == 0:
                expired_normally += 1
                cd_base = None
                cd_peak = None

    if in_pos:
        cash = qty * closes[-1] * (1 - TRADING_FEE)
        trades += 1

    return dict(
        pnl=(cash / 1000.0 - 1.0) * 100.0,
        trades=trades, skipped=skipped,
        lifted_early=lifted_early, expired=expired_normally,
        cd_active_hours=cd_active_any,
    )


def slice_tail(signals, days):
    end_t = signals[-1]['datetime']
    start_t = end_t - pd.Timedelta(days=days)
    return [s for s in signals if s['datetime'] >= start_t]


def run_variants(signals, label_prefix=''):
    variants = [
        ('V0 (no gate)',              dict(mode='none')),
        ('V7 fixed 30h (prod)',       dict(mode='fixed')),
        ('V7a lift <=8h-base',        dict(mode='adaptive_short')),
        ('V7b lift <=36h-base',       dict(mode='adaptive_long')),
        ('V7c lift <=min(bases)',     dict(mode='adaptive_both')),
    ]
    print(f"\n  {'variant':<26}{'PnL%':>9}{'trades':>9}{'skipped':>9}"
          f"{'lift_early':>12}{'expired':>10}{'cd_hrs':>9}")
    print("  " + "-" * 84)
    base_pnl = None
    for label, kw in variants:
        r = simulate(signals, **kw)
        if 'prod' in label:
            base_pnl = r['pnl']
        print(f"  {label:<26}{r['pnl']:>+9.2f}{r['trades']:>9}{r['skipped']:>9}"
              f"{r['lifted_early']:>12}{r['expired']:>10}{r['cd_active_hours']:>9}")
    return base_pnl


def main():
    signals = load_signals()
    print("=" * 86)
    print("  ADAPTIVE GATE TEST -- lift V7 cooldown early if price reverts to pre-rally level")
    print(f"  cache: {len(signals)} sigs  {signals[0]['datetime']} -> {signals[-1]['datetime']}")
    print(f"  V7: rr{V7['h_short']}h>={V7['t_short']}% OR rr{V7['h_long']}h>={V7['t_long']}% -> {V7['cd_hours']}h cooldown")
    print(f"  'original level' = close at trigger_time - lookback (h_short=8h or h_long=36h)")
    print("=" * 86)

    for days in (30, 90):
        sub = slice_tail(signals, days)
        print(f"\n  === LAST {days} DAYS  ({len(sub)} sigs, "
              f"{sub[0]['datetime']} -> {sub[-1]['datetime']}) ===")
        run_variants(sub)


if __name__ == '__main__':
    main()
