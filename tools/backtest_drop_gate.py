"""
Drop-gate OOS backtest on 90d cached ETH signals.

Uses `data/eth_sl_signals_90d.pkl` (2185 hourly BUY/SELL signals over ~90 days)
to simulate trades under current production policy (shield ON, regime-conf thresholds,
max_hold) and applies a drop-gate rule (X% loss in Y hours -> block BUY for Z hours)
to measure lift over baseline.

Splits results into three windows for IN-SAMPLE vs OOS comparison:
  - Full 90d: 2026-01-17 to 2026-04-18 (all available)
  - OOS (~60d): 2026-01-17 to 2026-03-19 (before live-trade window started)
  - IS  (~30d): 2026-03-19 to 2026-04-18 (overlaps live-trade sample)

This lets us see whether the drop-gate we tuned on 23 live trades also helps
on the ~60d BEFORE those trades started.

Usage:
  python tools/backtest_drop_gate.py                    # sweep + heatmap
  python tools/backtest_drop_gate.py --x 2 --y 9 --z 24 # single rule
"""

import argparse
import pickle
import os
import sys
import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

PKL_PATH = os.path.join(ENGINE_DIR, 'data', 'eth_sl_signals_90d.pkl')

# Current ETH production policy (from config/regime_config_ed.json)
POLICY = {
    'bull_conf': 90.0,
    'bear_conf': 75.0,
    'bull_shield': True,
    'bear_shield': True,
    'min_sell_pnl_pct': 0.55,  # percent
    'max_hold_hours': 8,
    'fee_per_leg': 0.0005,  # 5 bps realistic maker blend
}


def load_signals():
    with open(PKL_PATH, 'rb') as f:
        return pickle.load(f)


def precompute_drops(sigs, Y_windows):
    """For each hour i, compute close[i]/close[i-Y] - 1 for each Y.
    Returns {Y: np.array of length len(sigs)} of percentage drops (negative = drop)."""
    closes = np.array([s['close'] for s in sigs])
    out = {}
    for Y in Y_windows:
        arr = np.full(len(closes), np.nan)
        arr[Y:] = (closes[Y:] / closes[:-Y] - 1.0) * 100.0
        out[Y] = arr
    return out


def simulate(sigs, drop_arr=None, X=None, Z=None, rally_arr=None, rX=None, rZ=None):
    """Simulate trades with current policy.
    drop_arr + X + Z: drop-gate (block BUY for Z hours after close dropped >=X% over Y hours).
    rally_arr + rX + rZ: rally-gate (like current prod - block BUY for rZ hours after +rX%).
    Both can be active simultaneously (independent cooldowns).
    Returns dict with pnl_pct, trades, blocked_drop, blocked_rally.
    """
    fee = POLICY['fee_per_leg']
    bull_conf = POLICY['bull_conf']
    bear_conf = POLICY['bear_conf']
    bull_shield = POLICY['bull_shield']
    bear_shield = POLICY['bear_shield']
    min_sell_pnl = POLICY['min_sell_pnl_pct']
    max_hold = POLICY['max_hold_hours']

    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry = 0.0
    hold = 0
    trades = 0
    blocked_drop = 0
    blocked_rally = 0

    drop_cd_left = 0
    rally_cd_left = 0

    for i, s in enumerate(sigs):
        price = s['close']
        regime = s.get('regime', 'bull')
        conf_thr = bull_conf if regime == 'bull' else bear_conf
        shield_on = (bull_shield if regime == 'bull' else bear_shield)
        shield_min = min_sell_pnl if shield_on else 0.0

        # Drop-gate trigger
        if drop_arr is not None and X is not None and Z is not None:
            d = drop_arr[i]
            if not np.isnan(d) and d <= -X:
                drop_cd_left = max(drop_cd_left, Z)

        # Rally-gate trigger (if used)
        if rally_arr is not None and rX is not None and rZ is not None:
            r = rally_arr[i]
            if not np.isnan(r) and r >= rX:
                rally_cd_left = max(rally_cd_left, rZ)

        # BUY logic
        if s['signal'] == 'BUY' and s.get('confidence', 0) >= conf_thr and not in_pos:
            if drop_cd_left > 0:
                blocked_drop += 1
            elif rally_cd_left > 0:
                blocked_rally += 1
            else:
                # Buy next bar open (use current price as proxy)
                fp = sigs[i + 1]['close'] if i + 1 < len(sigs) else price
                qty = cash * (1 - fee) / fp
                cash = 0.0
                in_pos = True
                entry = fp
                hold = 0
        elif s['signal'] == 'SELL' and in_pos:
            fp = sigs[i + 1]['close'] if i + 1 < len(sigs) else price
            cur_pnl_pct = (fp / entry - 1.0) * 100.0
            if cur_pnl_pct >= shield_min or hold >= max_hold:
                cash = qty * fp * (1 - fee)
                trades += 1
                in_pos = False
                qty = 0.0
                entry = 0.0
                hold = 0

        if in_pos:
            hold += 1
        if drop_cd_left > 0:
            drop_cd_left -= 1
        if rally_cd_left > 0:
            rally_cd_left -= 1

    # Close any open position at last price
    if in_pos:
        cash = qty * sigs[-1]['close'] * (1 - fee)
        trades += 1

    pnl_pct = (cash / 1000.0 - 1.0) * 100.0
    return {
        'pnl_pct': pnl_pct,
        'trades': trades,
        'blocked_drop': blocked_drop,
        'blocked_rally': blocked_rally,
    }


def split_signals(sigs, oos_days=60, is_days=30):
    """Return (oos_sigs, is_sigs, full_sigs).
    OOS = earliest oos_days; IS = latest is_days; full = all."""
    import pandas as pd
    dts = [pd.Timestamp(s['datetime']) for s in sigs]
    end = dts[-1]
    oos_cut = end - pd.Timedelta(days=is_days)
    oos_sigs = [s for s, t in zip(sigs, dts) if t < oos_cut]
    is_sigs = [s for s, t in zip(sigs, dts) if t >= oos_cut]
    return oos_sigs, is_sigs, sigs


def sweep_grid(sigs, X_grid, Y_grid, Z_grid, label=''):
    drops = precompute_drops(sigs, Y_grid)
    rows = []
    baseline = simulate(sigs)['pnl_pct']
    for X in X_grid:
        for Y in Y_grid:
            for Z in Z_grid:
                result = simulate(sigs, drop_arr=drops[Y], X=X, Z=Z)
                rows.append({
                    'X': X, 'Y': Y, 'Z': Z,
                    'pnl': result['pnl_pct'],
                    'delta': result['pnl_pct'] - baseline,
                    'trades': result['trades'],
                    'blocked': result['blocked_drop'],
                })
    r = pd.DataFrame(rows).sort_values('pnl', ascending=False)
    return r, baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x', type=float, help='Single rule: drop threshold (e.g. 2.0)')
    ap.add_argument('--y', type=int, help='Single rule: window hours (e.g. 9)')
    ap.add_argument('--z', type=int, help='Single rule: cooldown hours (e.g. 24)')
    args = ap.parse_args()

    sigs = load_signals()
    print(f'Loaded {len(sigs)} signals from {pd.Timestamp(sigs[0]["datetime"])} to {pd.Timestamp(sigs[-1]["datetime"])}')
    print(f'Policy: bull@{POLICY["bull_conf"]}% shield=ON, bear@{POLICY["bear_conf"]}% shield=ON, '
          f'min_sell_pnl={POLICY["min_sell_pnl_pct"]}%, max_hold={POLICY["max_hold_hours"]}h, '
          f'fee={POLICY["fee_per_leg"]*2*100:.2f}%/roundtrip')
    print()

    oos, ins, full = split_signals(sigs, oos_days=60, is_days=30)
    print(f'OOS window (~60d, pre-live-trade): {len(oos)} hours, {pd.Timestamp(oos[0]["datetime"])} -> {pd.Timestamp(oos[-1]["datetime"])}')
    print(f'IS window (~30d, live-trade overlap): {len(ins)} hours, {pd.Timestamp(ins[0]["datetime"])} -> {pd.Timestamp(ins[-1]["datetime"])}')
    print(f'FULL: {len(full)} hours')
    print()

    # Baselines per window
    bl_oos = simulate(oos)
    bl_is = simulate(ins)
    bl_full = simulate(full)
    print(f'Baselines (no gate):')
    print(f'  OOS  ~60d: {bl_oos["pnl_pct"]:+7.2f}%  ({bl_oos["trades"]} trades)')
    print(f'  IS   ~30d: {bl_is["pnl_pct"]:+7.2f}%  ({bl_is["trades"]} trades)')
    print(f'  FULL ~90d: {bl_full["pnl_pct"]:+7.2f}%  ({bl_full["trades"]} trades)')
    print()

    if args.x and args.y and args.z:
        # Single rule mode
        print(f'Single rule: -{args.x}% over {args.y}h -> block {args.z}h')
        for label, window, bl in [('OOS', oos, bl_oos), ('IS', ins, bl_is), ('FULL', full, bl_full)]:
            drops = precompute_drops(window, [args.y])
            r = simulate(window, drop_arr=drops[args.y], X=args.x, Z=args.z)
            delta = r['pnl_pct'] - bl['pnl_pct']
            print(f'  {label:4}: {r["pnl_pct"]:+7.2f}%  delta {delta:+.2f}pp  trades={r["trades"]}  blocked={r["blocked_drop"]}')
        return

    # Grid sweep mode — focused grid from the 30d live study
    X_grid = [1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
    Y_grid = [4, 6, 8, 9, 10, 12]
    Z_grid = [8, 12, 24]

    oos_r, _ = sweep_grid(oos, X_grid, Y_grid, Z_grid, 'OOS')
    is_r, _ = sweep_grid(ins, X_grid, Y_grid, Z_grid, 'IS')
    full_r, _ = sweep_grid(full, X_grid, Y_grid, Z_grid, 'FULL')

    # Top 10 for each window
    for label, r, bl in [('OOS', oos_r, bl_oos), ('IS', is_r, bl_is), ('FULL', full_r, bl_full)]:
        print(f'=== {label} top 10 rules (baseline {bl["pnl_pct"]:+.2f}%) ===')
        print(f'{"X%":>4} {"Yh":>3} {"Zh":>3} {"pnl":>7} {"delta":>7} {"trades":>6} {"blk":>4}')
        for _, row in r.head(10).iterrows():
            print(f'{row["X"]:>4.2f} {int(row["Y"]):>3} {int(row["Z"]):>3} {row["pnl"]:+7.2f} {row["delta"]:+7.2f} {int(row["trades"]):>6} {int(row["blocked"]):>4}')
        print()

    # Key question: does the 30d-live-trade winner (-2% / 9h / 24h) also win OOS?
    target = (2.0, 9, 24)
    print(f'=== Target rule -{target[0]}% / {target[1]}h / {target[2]}h across windows ===')
    for label, r, bl in [('OOS', oos_r, bl_oos), ('IS', is_r, bl_is), ('FULL', full_r, bl_full)]:
        sub = r[(r['X']==target[0]) & (r['Y']==target[1]) & (r['Z']==target[2])]
        if len(sub):
            row = sub.iloc[0]
            rank = r.reset_index(drop=True).index[(r.reset_index(drop=True)['X']==target[0]) &
                                                   (r.reset_index(drop=True)['Y']==target[1]) &
                                                   (r.reset_index(drop=True)['Z']==target[2])][0] + 1
            print(f'  {label:4}: pnl {row["pnl"]:+7.2f}%  delta {row["delta"]:+7.2f}pp  trades={int(row["trades"])}  blocked={int(row["blocked"])}  rank {rank}/{len(r)}')

    print()
    print('=== Cross-window consistency: top-10 intersection ===')
    top10_oos = set(zip(oos_r.head(10)['X'], oos_r.head(10)['Y'], oos_r.head(10)['Z']))
    top10_is = set(zip(is_r.head(10)['X'], is_r.head(10)['Y'], is_r.head(10)['Z']))
    overlap = top10_oos & top10_is
    print(f'  OOS top 10 INTERSECT IS top 10: {len(overlap)} rules in common')
    if overlap:
        print(f'  {"X%":>4} {"Yh":>3} {"Zh":>3} {"OOS_delta":>10} {"IS_delta":>10}')
        for (X, Y, Z) in sorted(overlap):
            oos_row = oos_r[(oos_r['X']==X) & (oos_r['Y']==Y) & (oos_r['Z']==Z)].iloc[0]
            is_row = is_r[(is_r['X']==X) & (is_r['Y']==Y) & (is_r['Z']==Z)].iloc[0]
            print(f'  {X:>4.2f} {int(Y):>3} {int(Z):>3} {oos_row["delta"]:+10.2f} {is_row["delta"]:+10.2f}')


if __name__ == '__main__':
    main()
