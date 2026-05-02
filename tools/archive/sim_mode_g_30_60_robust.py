"""
sim_mode_g_30_60_robust.py — run extended-HORIZONS Mode G on BOTH 30d and
60d, then find configs that pass STRICT on BOTH windows. Cross-window
robustness filter — heavy overfit reduction vs single-window winner.

Reads: cache + regime_config_ed.json. Writes nothing.
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

HORIZONS_EXTENDED = [8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 48]
CD_GRID = [6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 48]
PLATEAU_THR = 0.7


def thr_for(h):
    if h in (8, 10, 12):       return [round(2.0 + 0.5*i, 2) for i in range(9)]
    if h in (14, 16, 18, 20):  return [round(3.0 + 0.5*i, 2) for i in range(9)]
    if h in (24, 30, 36):      return [round(4.0 + 0.5*i, 2) for i in range(11)]
    if h == 48:                return [round(4.0 + 0.5*i, 2) for i in range(11)]
    raise ValueError(h)


def load_signals():
    with open(CACHE_PKL, 'rb') as f:
        sigs = pickle.load(f)
    for s in sigs:
        t = pd.Timestamp(s['datetime'])
        s['datetime'] = t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    sigs.sort(key=lambda s: s['datetime'])
    return sigs


def simulate(sigs, rr_dict, h_s, h_l, t_s, t_l, cd_h, asset_cfg):
    bull_shield = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    bull_conf = float(asset_cfg.get('bull', {}).get('min_confidence', 75.0))
    bear_conf = float(asset_cfg.get('bear', {}).get('min_confidence', 65.0))
    min_sell_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.5))
    max_hold = int(asset_cfg.get('max_hold_hours', 10))

    cash = 1000.0; qty = 0.0; in_pos = False; entry = 0.0
    hold = 0; trades = 0; skipped = 0; cd = 0
    ec = [1000.0]
    n = len(sigs)
    rs_arr = rr_dict.get(h_s); rl_arr = rr_dict.get(h_l)
    for i in range(n):
        s = sigs[i]; price = s['close']
        regime = s.get('regime', 'bull')
        conf_thr = bull_conf if regime == 'bull' else bear_conf
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
            shield_on = bull_shield if regime == 'bull' else bear_shield
            shield_min = min_sell_pnl if shield_on else 0.0
            if cur >= shield_min or hold >= max_hold:
                cash = qty * price * (1 - FEE)
                trades += 1; in_pos = False; qty = 0.0; entry = 0.0; hold = 0
        if in_pos: hold += 1
        if cd > 0: cd -= 1
    if in_pos:
        cash = qty * sigs[-1]['close'] * (1 - FEE); trades += 1
    pnl = (cash / 1000.0 - 1.0) * 100.0
    arr = np.array(ec); peak = np.maximum.accumulate(arr); dd = (arr - peak) / peak
    mdd = float(dd.min()) * 100.0 if len(dd) else 0.0
    return dict(pnl_pct=pnl, dd_pct=mdd, trades=trades, skipped=skipped)


def build_rr(sigs, horizons):
    df = pd.DataFrame([{'datetime': s['datetime'], 'close': s['close']} for s in sigs])
    df = df.sort_values('datetime').reset_index(drop=True)
    return {h: ((df['close'] / df['close'].shift(h) - 1.0) * 100.0).values for h in horizons}


def sweep(sigs, asset_cfg, replay_h, label):
    days = replay_h / 24.0
    half_days = days / 2.0
    end_t = sigs[-1]['datetime']
    t_h1_lo = end_t - pd.Timedelta(days=half_days)
    t_h2_lo = end_t - pd.Timedelta(days=days)
    sigs_h1 = [s for s in sigs if s['datetime'] >= t_h1_lo]
    sigs_h2 = [s for s in sigs if t_h2_lo <= s['datetime'] < t_h1_lo]
    sigs_ref = [s for s in sigs if s['datetime'] >= t_h2_lo]

    rr_h1 = build_rr(sigs_h1, HORIZONS_EXTENDED)
    rr_h2 = build_rr(sigs_h2, HORIZONS_EXTENDED)
    rr_ref = build_rr(sigs_ref, HORIZONS_EXTENDED)

    b_h1 = simulate(sigs_h1, {}, 8, 36, 9999, 9999, 0, asset_cfg)
    b_h2 = simulate(sigs_h2, {}, 8, 36, 9999, 9999, 0, asset_cfg)
    b_ref = simulate(sigs_ref, {}, 8, 36, 9999, 9999, 0, asset_cfg)
    print(f"\n[{label} window: {days:.0f}d / halves {half_days:.0f}+{half_days:.0f}d]")
    print(f"  baselines (no gate): H1={b_h1['pnl_pct']:+.2f}% H2={b_h2['pnl_pct']:+.2f}% REF={b_ref['pnl_pct']:+.2f}%")

    pairs = [(a, b) for i, a in enumerate(HORIZONS_EXTENDED) for b in HORIZONS_EXTENDED[i+1:]]
    results = []
    for h_s, h_l in pairs:
        for t_s in thr_for(h_s):
            for t_l in thr_for(h_l):
                for cd_h in CD_GRID:
                    r_h1 = simulate(sigs_h1, rr_h1, h_s, h_l, t_s, t_l, cd_h, asset_cfg)
                    r_h2 = simulate(sigs_h2, rr_h2, h_s, h_l, t_s, t_l, cd_h, asset_cfg)
                    r_ref = simulate(sigs_ref, rr_ref, h_s, h_l, t_s, t_l, cd_h, asset_cfg)
                    beats_h1 = r_h1['pnl_pct'] > b_h1['pnl_pct']
                    beats_h2 = r_h2['pnl_pct'] > b_h2['pnl_pct']
                    beats_ref = r_ref['pnl_pct'] > b_ref['pnl_pct']
                    beats_3of3 = beats_h1 and beats_h2 and beats_ref
                    results.append(dict(
                        h_s=h_s, h_l=h_l, t_s=t_s, t_l=t_l, cd_h=cd_h,
                        h1=r_h1['pnl_pct'], h2=r_h2['pnl_pct'], ref=r_ref['pnl_pct'],
                        worst_dd=min(r_h1['dd_pct'], r_h2['dd_pct'], r_ref['dd_pct']),
                        beats_3of3=beats_3of3,
                        skipped_ref=r_ref['skipped'],
                    ))

    df = pd.DataFrame(results)
    df['delta_h1'] = df['h1'] - b_h1['pnl_pct']
    df['delta_h2'] = df['h2'] - b_h2['pnl_pct']
    df['delta_ref'] = df['ref'] - b_ref['pnl_pct']
    df['score_dd_aware'] = df['delta_ref'] - 0.5 * abs(df['worst_dd'])
    df['baseline_ref'] = b_ref['pnl_pct']
    df['key'] = df.apply(lambda r: (int(r['h_s']), int(r['h_l']), float(r['t_s']),
                                     float(r['t_l']), int(r['cd_h'])), axis=1)
    print(f"  Total: {len(df)}, beats_3of3: {df['beats_3of3'].sum()}")
    return df, b_ref['pnl_pct']


def main():
    print("Loading...")
    sigs = load_signals()
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    df_60, base_60 = sweep(sigs, asset_cfg, 1440, "60d")
    df_30, base_30 = sweep(sigs, asset_cfg, 720, "30d")

    # Cross-window robustness check
    print("\n" + "=" * 110)
    print("  CROSS-WINDOW ROBUSTNESS — must pass STRICT on BOTH 30d AND 60d")
    print("=" * 110)

    strict_30_keys = set(df_30[df_30['beats_3of3']]['key'].tolist())
    strict_60_keys = set(df_60[df_60['beats_3of3']]['key'].tolist())
    intersect_keys = strict_30_keys & strict_60_keys
    print(f"  STRICT in 30d only: {len(strict_30_keys)}")
    print(f"  STRICT in 60d only: {len(strict_60_keys)}")
    print(f"  STRICT in BOTH: {len(intersect_keys)}")

    if not intersect_keys:
        print("  No config passes STRICT on both windows. Reporting top-30 by 60d that ALSO passes 30d delta_ref > 0:")
        df_60_strict = df_60[df_60['beats_3of3']].copy()
        keys_60_strict = df_60_strict['key'].tolist()
        # Look up 30d delta for each
        df_30_lookup = df_30.set_index('key')
        df_60_strict['delta_ref_30d'] = df_60_strict['key'].map(lambda k: df_30_lookup.loc[k, 'delta_ref'] if k in df_30_lookup.index else np.nan)
        pos30 = df_60_strict[df_60_strict['delta_ref_30d'] > 0].sort_values('score_dd_aware', ascending=False)
        print(pos30.head(20)[['h_s', 'h_l', 't_s', 't_l', 'cd_h', 'h1', 'h2', 'ref',
                               'delta_h1', 'delta_h2', 'delta_ref', 'delta_ref_30d',
                               'worst_dd', 'skipped_ref']].round(2).to_string(index=False))
        return

    # Build joint table for intersect
    rows = []
    # Build dict-based lookups to sidestep pandas tuple-index indexer
    df_30_dict = {row['key']: row for _, row in df_30.iterrows()}
    df_60_dict = {row['key']: row for _, row in df_60.iterrows()}
    for k in intersect_keys:
        r60 = df_60_dict[k]
        r30 = df_30_dict[k]
        rows.append({
            'h_s': k[0], 'h_l': k[1], 't_s': k[2], 't_l': k[3], 'cd_h': k[4],
            'ref_30d': r30['ref'], 'delta_30d': r30['delta_ref'],
            'ref_60d': r60['ref'], 'delta_60d': r60['delta_ref'],
            'worst_dd_30d': r30['worst_dd'], 'worst_dd_60d': r60['worst_dd'],
            'skipped_30d': r30['skipped_ref'], 'skipped_60d': r60['skipped_ref'],
            'has_48h': (k[0] == 48 or k[1] == 48),
        })
    joint = pd.DataFrame(rows)
    joint['avg_delta'] = (joint['delta_30d'] + joint['delta_60d']) / 2
    joint['min_delta'] = np.minimum(joint['delta_30d'], joint['delta_60d'])
    joint = joint.sort_values('avg_delta', ascending=False)

    print(f"\n  Top 25 cross-window robust configs (sorted by avg delta):")
    cols = ['h_s', 'h_l', 't_s', 't_l', 'cd_h',
            'ref_30d', 'delta_30d', 'ref_60d', 'delta_60d',
            'worst_dd_60d', 'skipped_30d', 'skipped_60d', 'has_48h']
    print(joint.head(25)[cols].round(2).to_string(index=False))

    # Now ranked by min_delta (most conservative — both windows must look good)
    joint_min = joint.sort_values('min_delta', ascending=False)
    print(f"\n  Top 15 by MIN delta (worst-case window):")
    print(joint_min.head(15)[cols + ['min_delta']].round(2).to_string(index=False))

    # 48h subset
    has_48h = joint[joint['has_48h']].sort_values('avg_delta', ascending=False)
    print(f"\n  Cross-window robust configs that include 48h ({len(has_48h)} total):")
    print(has_48h.head(15)[cols].round(2).to_string(index=False))

    # Compare best winner of each
    print(f"\n{'='*110}")
    print("  SUMMARY")
    print(f"{'='*110}")
    if len(joint) > 0:
        w = joint.iloc[0]
        print(f"  Best CROSS-WINDOW winner (avg delta): "
              f"rr{int(w['h_s'])}h>={w['t_s']}% OR rr{int(w['h_l'])}h>={w['t_l']}%, cd={int(w['cd_h'])}h")
        print(f"    30d: {w['ref_30d']:.2f}% (Δ{w['delta_30d']:+.2f}pp), "
              f"60d: {w['ref_60d']:.2f}% (Δ{w['delta_60d']:+.2f}pp)")
        print(f"    DD60d: {w['worst_dd_60d']:.2f}%, blocks 30d: {int(w['skipped_30d'])}, "
              f"60d: {int(w['skipped_60d'])}, has_48h: {bool(w['has_48h'])}")
        wm = joint_min.iloc[0]
        print(f"\n  Best CROSS-WINDOW winner (min delta — most conservative): "
              f"rr{int(wm['h_s'])}h>={wm['t_s']}% OR rr{int(wm['h_l'])}h>={wm['t_l']}%, cd={int(wm['cd_h'])}h")
        print(f"    30d: {wm['ref_30d']:.2f}% (Δ{wm['delta_30d']:+.2f}pp), "
              f"60d: {wm['ref_60d']:.2f}% (Δ{wm['delta_60d']:+.2f}pp)")
        print(f"    DD60d: {wm['worst_dd_60d']:.2f}%")

    # Save
    out_path = os.path.join(ENGINE, 'output',
                            f'sim_mode_g_robust_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    joint.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
