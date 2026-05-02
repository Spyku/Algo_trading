"""
tier1_held_out_test.py — TIER 1 validation. Test the MIX gate winner on
the FIRST 30 days of the 90d cache (data from days 60-90 ago) — which
was NOT used in either the 30d or 60d optimization runs.

Cache spans roughly 2026-01-17 to 2026-04-18.
- 30d window (used in opt): last 30d = ~2026-03-19 to 2026-04-18
- 60d window (used in opt): last 60d = ~2026-02-17 to 2026-04-18
- HELD-OUT slice: first ~30d = ~2026-01-17 to 2026-02-16 (not in either opt)

If MIX winner survives this true OOS check, it's real signal.
If it fails, cross-window filter wasn't enough.

Read-only.
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PKL = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
REGIME_CFG = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
FEE = 0.0005


def load_signals():
    with open(CACHE_PKL, 'rb') as f:
        sigs = pickle.load(f)
    for s in sigs:
        t = pd.Timestamp(s['datetime'])
        s['datetime'] = t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    sigs.sort(key=lambda s: s['datetime'])
    return sigs


def build_rr(sigs, horizons):
    df = pd.DataFrame([{'datetime': s['datetime'], 'close': s['close']} for s in sigs])
    df = df.sort_values('datetime').reset_index(drop=True)
    return {h: ((df['close'] / df['close'].shift(h) - 1.0) * 100.0).values for h in horizons}


def simulate(sigs, asset_cfg, gate=None):
    bull_shield = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    bull_conf = float(asset_cfg.get('bull', {}).get('min_confidence', 75.0))
    bear_conf = float(asset_cfg.get('bear', {}).get('min_confidence', 65.0))
    min_sell_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.5))
    max_hold = int(asset_cfg.get('max_hold_hours', 10))

    h_s = h_l = 0; t_s = t_l = 9999.0; cd_h = 0
    rs_arr = rl_arr = None
    if gate is not None:
        h_s = int(gate['h_short']); h_l = int(gate['h_long'])
        t_s = float(gate['t_short_pct']); t_l = float(gate['t_long_pct'])
        cd_h = int(gate['cd_hours'])
        rr_dict = build_rr(sigs, [h_s, h_l])
        rs_arr = rr_dict.get(h_s); rl_arr = rr_dict.get(h_l)

    cash = 1000.0; qty = 0.0; in_pos = False; entry = 0.0
    hold = 0; cd = 0
    trade_pnls = []
    ec = [1000.0]
    skipped = 0
    for i in range(len(sigs)):
        s = sigs[i]; price = s['close']
        regime = s.get('regime', 'bull')
        conf_thr = bull_conf if regime == 'bull' else bear_conf
        shield_on = bull_shield if regime == 'bull' else bear_shield
        if cd_h > 0 and rs_arr is not None and rl_arr is not None:
            rs = rs_arr[i]; rl = rl_arr[i]
            if (rs == rs and rs >= t_s) or (rl == rl and rl >= t_l):
                cd = max(cd, cd_h)
        ec.append(cash + qty * price if in_pos else cash)
        if s['signal'] == 'BUY' and s['confidence'] >= conf_thr and not in_pos:
            if cd > 0:
                skipped += 1
            else:
                qty = cash * (1 - FEE) / price
                cash = 0.0; in_pos = True; entry = price; hold = 0
        elif s['signal'] == 'SELL' and in_pos:
            cur = (price / entry - 1.0) * 100.0
            shield_min = min_sell_pnl if shield_on else 0.0
            if cur >= shield_min or hold >= max_hold:
                cash = qty * price * (1 - FEE)
                trade_pnls.append(cur)
                in_pos = False; qty = 0.0; entry = 0.0; hold = 0
        if in_pos: hold += 1
        if cd > 0: cd -= 1
    if in_pos:
        cash = qty * sigs[-1]['close'] * (1 - FEE)
        trade_pnls.append((sigs[-1]['close'] / entry - 1.0) * 100.0)

    pnl = (cash / 1000.0 - 1.0) * 100.0
    arr = np.array(ec); peak = np.maximum.accumulate(arr); dd = (arr - peak) / peak
    mdd = float(dd.min()) * 100.0 if len(dd) else 0.0
    n = len(trade_pnls)
    wins = sum(1 for p in trade_pnls if p > 0)
    return {
        'return_pct': round(pnl, 2),
        'n_trades': n,
        'win_rate': round(wins / n * 100, 1) if n else 0.0,
        'max_dd': round(mdd, 2),
        'skipped': skipped,
        'avg_pnl': round(float(np.mean(trade_pnls)), 2) if trade_pnls else 0.0,
    }


def buy_and_hold(sigs):
    p0 = float(sigs[0]['close']); p1 = float(sigs[-1]['close'])
    return ((p1 / p0) * (1 - FEE) * (1 - FEE) - 1) * 100


def main():
    print("Loading...")
    sigs_all = load_signals()
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    end_t = sigs_all[-1]['datetime']
    print(f"  Cache: {len(sigs_all)} signals from {sigs_all[0]['datetime']} to {end_t}")

    # Define windows
    # 30d (used in opt): last 30d
    # 60d (used in opt): last 60d
    # HELD-OUT: 90d ago to 60d ago = first 30d of the 90d cache
    held_out_lo = end_t - pd.Timedelta(days=90)
    held_out_hi = end_t - pd.Timedelta(days=60)
    sigs_held_out = [s for s in sigs_all if held_out_lo <= s['datetime'] < held_out_hi]
    print(f"\n  HELD-OUT slice: {sigs_held_out[0]['datetime']} to {sigs_held_out[-1]['datetime']}")
    print(f"  HELD-OUT signals: {len(sigs_held_out)}")
    print(f"  This data was NOT used in either 30d or 60d optimization.")

    # Define candidate gates
    current_prod_gate = {
        'h_short': 20, 'h_long': 24,
        't_short_pct': 4.0, 't_long_pct': 4.5,
        'cd_hours': 12,
    }
    setup_30d_gate = {  # also = MIX winner
        'h_short': 12, 'h_long': 20,
        't_short_pct': 2.5, 't_long_pct': 4.0,
        'cd_hours': 24,
    }
    setup_60d_gate = {
        'h_short': 24, 'h_long': 48,
        't_short_pct': 5.0, 't_long_pct': 6.5,
        'cd_hours': 24,
    }

    # B&H
    bh = buy_and_hold(sigs_held_out)
    print(f"\n  Buy & hold: {bh:+.2f}%")

    # Run each candidate on the held-out slice
    setups = [
        ('NO GATE (baseline)', None),
        ('CURRENT PROD (rr20>=4 OR rr24>=4.5 cd=12)', current_prod_gate),
        ('Setup 30d-opt / MIX (rr12>=2.5 OR rr20>=4 cd=24)', setup_30d_gate),
        ('Setup 60d-opt (rr24>=5 OR rr48>=6.5 cd=24)', setup_60d_gate),
    ]

    print(f"\n{'='*120}")
    print(f"  HELD-OUT 30d SLICE TEST RESULTS")
    print(f"{'='*120}")
    print(f"  {'Setup':<55} {'Return':>9} {'WR':>6} {'MaxDD':>7} {'Trades':>7} {'Skipped':>9}")
    print("  " + "-"*120)

    baseline = None
    for label, gate in setups:
        r = simulate(sigs_held_out, asset_cfg, gate=gate)
        if baseline is None:
            baseline = r['return_pct']
            delta_str = ""
        else:
            delta = r['return_pct'] - baseline
            delta_str = f"  ({'+' if delta >= 0 else ''}{delta:.2f}pp vs no-gate)"
        print(f"  {label:<55} {r['return_pct']:>+8.2f}% {r['win_rate']:>5.0f}% {r['max_dd']:>+6.2f}% {r['n_trades']:>7} {r['skipped']:>9}{delta_str}")

    # The verdict on MIX
    print(f"\n{'='*120}")
    print(f"  VERDICT")
    print(f"{'='*120}")
    no_gate_r = simulate(sigs_held_out, asset_cfg, gate=None)
    prod_r = simulate(sigs_held_out, asset_cfg, gate=current_prod_gate)
    mix_r = simulate(sigs_held_out, asset_cfg, gate=setup_30d_gate)
    s60_r = simulate(sigs_held_out, asset_cfg, gate=setup_60d_gate)

    delta_mix_vs_nogate = mix_r['return_pct'] - no_gate_r['return_pct']
    delta_mix_vs_prod = mix_r['return_pct'] - prod_r['return_pct']
    delta_60_vs_nogate = s60_r['return_pct'] - no_gate_r['return_pct']
    delta_60_vs_prod = s60_r['return_pct'] - prod_r['return_pct']

    print(f"\n  MIX gate vs NO gate:       {delta_mix_vs_nogate:+.2f}pp  ({'PASS' if delta_mix_vs_nogate > 0 else 'FAIL'})")
    print(f"  MIX gate vs CURRENT PROD:  {delta_mix_vs_prod:+.2f}pp  ({'PASS' if delta_mix_vs_prod > 0 else 'FAIL'})")
    print(f"  60d gate vs NO gate:       {delta_60_vs_nogate:+.2f}pp  ({'PASS' if delta_60_vs_nogate > 0 else 'FAIL'})")
    print(f"  60d gate vs CURRENT PROD:  {delta_60_vs_prod:+.2f}pp  ({'PASS' if delta_60_vs_prod > 0 else 'FAIL'})")

    print(f"\n  Tier 1 PRIMARY question: does MIX gate help on data NOT used to find it?")
    if delta_mix_vs_nogate > 0 and delta_mix_vs_prod > 0:
        print(f"  --> YES. MIX gate beats both baselines on held-out data. Real signal.")
    elif delta_mix_vs_nogate > 0:
        print(f"  --> PARTIAL. MIX gate beats no-gate but not current PROD on held-out.")
        print(f"      Suggests new gate has signal but isn't strictly better than current.")
    else:
        print(f"  --> NO. MIX gate does NOT help on held-out data. Cross-window filter was insufficient.")
        print(f"      DO NOT ship to production without further validation (Tier 2).")


if __name__ == '__main__':
    main()
