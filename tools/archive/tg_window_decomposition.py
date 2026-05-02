"""
tg_window_decomposition.py — run Mode G (rally-cooldown sweep) on TWO
slicing strategies of the 90d signal cache, then compare:

CUMULATIVE windows (overlapping, most-recent N days):
  - cum_30d = last 30 days
  - cum_60d = last 60 days
  - cum_90d = full 90 days

DISJOINT windows (non-overlapping, 3 distinct 30-day chunks):
  - disj_recent  = days 0-30  (= cum_30d)
  - disj_mid     = days 30-60
  - disj_old     = days 60-90

Question: does the 90d cumulative winner agree with the disjoint-passing
intersection? If yes → cumulative is honest. If no → 90d aggregate hides
regime variance and the disjoint-strict approach is more conservative.

Mode T sweep is also included (4 shield on/off combos × current PROD
threshold/failsafe). Standalone, read-only.
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

# Use Mode G's existing search space + the 48h extension we found useful
HORIZONS = [8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 48]
CD_GRID = [6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 48]


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


def slice_window(sigs, from_days_ago, to_days_ago):
    """Return signals where (now - to_days_ago) <= dt < (now - from_days_ago).
    from_days_ago < to_days_ago (e.g., 0, 30 = most recent 30 days).
    """
    end = sigs[-1]['datetime']
    lo = end - pd.Timedelta(days=to_days_ago)
    hi = end - pd.Timedelta(days=from_days_ago)
    return [s for s in sigs if lo <= s['datetime'] <= hi]


def build_rr(sigs, horizons):
    df = pd.DataFrame([{'datetime': s['datetime'], 'close': s['close']} for s in sigs])
    df = df.sort_values('datetime').reset_index(drop=True)
    return {h: ((df['close'] / df['close'].shift(h) - 1.0) * 100.0).values for h in horizons}


def simulate(sigs, rr_dict, h_s, h_l, t_s, t_l, cd_h, asset_cfg,
             bull_shield_override=None, bear_shield_override=None):
    bull_shield = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    if bull_shield_override is not None: bull_shield = bull_shield_override
    if bear_shield_override is not None: bear_shield = bear_shield_override
    bull_conf = float(asset_cfg.get('bull', {}).get('min_confidence', 75.0))
    bear_conf = float(asset_cfg.get('bear', {}).get('min_confidence', 65.0))
    min_sell_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.5))
    max_hold = int(asset_cfg.get('max_hold_hours', 10))

    cash = 1000.0; qty = 0.0; in_pos = False; entry = 0.0
    hold = 0; trades = 0; skipped = 0; cd = 0
    n = len(sigs)
    rs_arr = rr_dict.get(h_s); rl_arr = rr_dict.get(h_l)
    trade_pnls = []
    for i in range(n):
        s = sigs[i]; price = s['close']
        regime = s.get('regime', 'bull')
        conf_thr = bull_conf if regime == 'bull' else bear_conf
        shield_on = bull_shield if regime == 'bull' else bear_shield
        if cd_h > 0 and rs_arr is not None and rl_arr is not None:
            rs = rs_arr[i]; rl = rl_arr[i]
            if (rs == rs and rs >= t_s) or (rl == rl and rl >= t_l):
                cd = max(cd, cd_h)
        if s['signal'] == 'BUY' and s['confidence'] >= conf_thr and not in_pos:
            if cd > 0: skipped += 1
            else:
                qty = cash * (1 - FEE) / price
                cash = 0.0; in_pos = True; entry = price; hold = 0
        elif s['signal'] == 'SELL' and in_pos:
            cur = (price / entry - 1.0) * 100.0
            shield_min = min_sell_pnl if shield_on else 0.0
            if cur >= shield_min or hold >= max_hold:
                cash = qty * price * (1 - FEE)
                trades += 1; trade_pnls.append(cur)
                in_pos = False; qty = 0.0; entry = 0.0; hold = 0
        if in_pos: hold += 1
        if cd > 0: cd -= 1
    if in_pos:
        cash = qty * sigs[-1]['close'] * (1 - FEE)
        trade_pnls.append((sigs[-1]['close'] / entry - 1.0) * 100.0)
        trades += 1
    pnl = (cash / 1000.0 - 1.0) * 100.0
    wr = sum(1 for p in trade_pnls if p > 0) / max(1, len(trade_pnls)) * 100
    return dict(pnl_pct=pnl, trades=trades, win_rate=wr, skipped=skipped)


def sweep_g_on_window(sigs, asset_cfg, label, current_prod_gate=None):
    """Run Mode G sweep on a single window. Returns DataFrame of all configs."""
    rr_dict = build_rr(sigs, HORIZONS)
    base = simulate(sigs, {}, 8, 36, 9999, 9999, 0, asset_cfg)
    # Also score current production gate for comparison
    if current_prod_gate:
        prod = simulate(sigs, rr_dict,
                       int(current_prod_gate['h_short']), int(current_prod_gate['h_long']),
                       float(current_prod_gate['t_short_pct']), float(current_prod_gate['t_long_pct']),
                       int(current_prod_gate['cd_hours']), asset_cfg)
        prod_pnl = prod['pnl_pct']
    else:
        prod_pnl = base['pnl_pct']

    pairs = [(a, b) for i, a in enumerate(HORIZONS) for b in HORIZONS[i+1:]]
    results = []
    for h_s, h_l in pairs:
        for t_s in thr_for(h_s):
            for t_l in thr_for(h_l):
                for cd_h in CD_GRID:
                    r = simulate(sigs, rr_dict, h_s, h_l, t_s, t_l, cd_h, asset_cfg)
                    results.append({
                        'h_s': h_s, 'h_l': h_l, 't_s': t_s, 't_l': t_l, 'cd_h': cd_h,
                        'pnl_pct': r['pnl_pct'], 'trades': r['trades'],
                        'wr': r['win_rate'], 'skipped': r['skipped'],
                        'delta_vs_base': r['pnl_pct'] - base['pnl_pct'],
                        'delta_vs_prod': r['pnl_pct'] - prod_pnl,
                    })
    df = pd.DataFrame(results)
    df['key'] = df.apply(lambda r: (int(r['h_s']), int(r['h_l']), float(r['t_s']),
                                     float(r['t_l']), int(r['cd_h'])), axis=1)
    df['beats_base'] = df['delta_vs_base'] > 0
    df['beats_prod'] = df['delta_vs_prod'] > 0
    return df, base['pnl_pct'], prod_pnl


def main():
    print("Loading 90d signals...")
    sigs = load_signals()
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']
    bull_gate = asset_cfg.get('bull', {}).get('rally_cooldown', {}) or asset_cfg.get('rally_cooldown', {})
    print(f"  Total signals: {len(sigs)}")
    print(f"  Date range: {sigs[0]['datetime']} to {sigs[-1]['datetime']}")
    print(f"  Current PROD bull gate: {bull_gate}")
    print()

    # Define windows
    windows = {
        'cum_30d (last 30)':       slice_window(sigs, 0, 30),
        'cum_60d (last 60)':       slice_window(sigs, 0, 60),
        'cum_90d (full)':          slice_window(sigs, 0, 90),
        'disj_recent (0-30)':      slice_window(sigs, 0, 30),
        'disj_mid    (30-60)':     slice_window(sigs, 30, 60),
        'disj_old    (60-90)':     slice_window(sigs, 60, 90),
    }
    print("Window sizes:")
    for name, w in windows.items():
        if w:
            print(f"  {name:<28}: {len(w):>5} sigs ({w[0]['datetime'].date()} to {w[-1]['datetime'].date()})")
        else:
            print(f"  {name:<28}: EMPTY")
    print()

    # Run Mode G sweep on each unique window (cum_30d == disj_recent)
    unique_windows = {
        'cum_30d':      windows['cum_30d (last 30)'],
        'cum_60d':      windows['cum_60d (last 60)'],
        'cum_90d':      windows['cum_90d (full)'],
        'disj_mid':     windows['disj_mid    (30-60)'],
        'disj_old':     windows['disj_old    (60-90)'],
    }

    results = {}
    for name, w in unique_windows.items():
        print(f"\n{'='*100}")
        print(f"  Mode G sweep on {name} ({len(w)} signals)")
        print(f"{'='*100}")
        df, base_pnl, prod_pnl = sweep_g_on_window(w, asset_cfg, name, current_prod_gate=bull_gate)
        n_strict = df['beats_base'].sum()
        print(f"  Baseline (no gate): {base_pnl:+.2f}%   PROD gate: {prod_pnl:+.2f}%")
        print(f"  Total configs: {len(df)}, beats baseline: {n_strict} ({n_strict/len(df)*100:.0f}%)")
        # Top 5 by delta_vs_base
        top = df.sort_values('delta_vs_base', ascending=False).head(5)
        print(f"  TOP 5 by delta vs no-gate:")
        for _, r in top.iterrows():
            print(f"    rr{int(r['h_s'])}>={r['t_s']}% OR rr{int(r['h_l'])}>={r['t_l']}% cd={int(r['cd_h'])}h  "
                  f"-> ret {r['pnl_pct']:+.2f}% (delta {r['delta_vs_base']:+.2f}pp)")
        results[name] = df

    # CROSS-WINDOW INTERSECTION ANALYSIS
    print(f"\n{'='*100}")
    print(f"  CROSS-WINDOW INTERSECTION ANALYSIS")
    print(f"{'='*100}")

    # CUMULATIVE: STRICT on 30d AND 60d AND 90d
    cum_keys = (set(results['cum_30d'][results['cum_30d']['beats_base']]['key'])
                & set(results['cum_60d'][results['cum_60d']['beats_base']]['key'])
                & set(results['cum_90d'][results['cum_90d']['beats_base']]['key']))
    print(f"\n  CUMULATIVE windows STRICT intersection (30d AND 60d AND 90d): {len(cum_keys)} configs")

    # DISJOINT: STRICT on recent AND mid AND old
    disj_keys = (set(results['cum_30d'][results['cum_30d']['beats_base']]['key'])
                 & set(results['disj_mid'][results['disj_mid']['beats_base']]['key'])
                 & set(results['disj_old'][results['disj_old']['beats_base']]['key']))
    print(f"  DISJOINT windows STRICT intersection (recent AND mid AND old): {len(disj_keys)} configs")

    # Cumulative-only winners (cum_90d top)
    print(f"\n  CUM_90D top winner (single-window):")
    cum_top = results['cum_90d'].sort_values('delta_vs_base', ascending=False).head(1).iloc[0]
    print(f"    rr{int(cum_top['h_s'])}>={cum_top['t_s']}% OR rr{int(cum_top['h_l'])}>={cum_top['t_l']}% cd={int(cum_top['cd_h'])}h")
    print(f"    cum_90d delta: {cum_top['delta_vs_base']:+.2f}pp")
    cum_top_key = cum_top['key']
    cum_top_in_cum_inter = cum_top_key in cum_keys
    cum_top_in_disj_inter = cum_top_key in disj_keys
    print(f"    Also passes CUMULATIVE STRICT: {cum_top_in_cum_inter}")
    print(f"    Also passes DISJOINT STRICT:   {cum_top_in_disj_inter}")

    # Cumulative-strict winner — dict lookups to avoid tuple-key pandas bug
    if cum_keys:
        cum30_d = {row['key']: row['delta_vs_base'] for _, row in results['cum_30d'].iterrows()}
        cum60_d = {row['key']: row['delta_vs_base'] for _, row in results['cum_60d'].iterrows()}
        cum90_d = {row['key']: row['delta_vs_base'] for _, row in results['cum_90d'].iterrows()}
        cum_strict_rows = []
        for k in cum_keys:
            cum_strict_rows.append({
                'key': k, 'd30': cum30_d[k], 'd60': cum60_d[k], 'd90': cum90_d[k],
                'min_delta': min(cum30_d[k], cum60_d[k], cum90_d[k]),
                'avg_delta': (cum30_d[k] + cum60_d[k] + cum90_d[k]) / 3,
            })
        cum_strict_df = pd.DataFrame(cum_strict_rows).sort_values('min_delta', ascending=False)
        winner = cum_strict_df.iloc[0]
        k = winner['key']
        print(f"\n  CUMULATIVE-STRICT winner (best worst-window across cum 30/60/90):")
        print(f"    rr{k[0]}>={k[2]}% OR rr{k[1]}>={k[3]}% cd={k[4]}h")
        print(f"    deltas: 30d {winner['d30']:+.2f}pp | 60d {winner['d60']:+.2f}pp | 90d {winner['d90']:+.2f}pp")
        print(f"    min_delta = {winner['min_delta']:+.2f}pp, avg_delta = {winner['avg_delta']:+.2f}pp")

    # Disjoint-strict winner — dict lookups
    if disj_keys:
        all_dicts = {n: {row['key']: (row['delta_vs_base'], row['pnl_pct'])
                          for _, row in results[n].iterrows()}
                     for n in unique_windows}
        rows = []
        for k in disj_keys:
            r = {'key': k}
            for n in unique_windows:
                r[f'delta_{n}'] = all_dicts[n][k][0]
                r[f'pnl_{n}'] = all_dicts[n][k][1]
            r['min_delta_disj'] = min(r['delta_cum_30d'], r['delta_disj_mid'], r['delta_disj_old'])
            r['avg_delta_disj'] = (r['delta_cum_30d'] + r['delta_disj_mid'] + r['delta_disj_old']) / 3
            rows.append(r)
        disj_df = pd.DataFrame(rows).sort_values('min_delta_disj', ascending=False)
        disj_winner = disj_df.iloc[0]
        k = disj_winner['key']
        print(f"\n  DISJOINT-STRICT winner (best worst-window across 3 disjoint 30d periods):")
        print(f"    rr{k[0]}>={k[2]}% OR rr{k[1]}>={k[3]}% cd={k[4]}h")
        print(f"    Per-window deltas (also showing cum windows for context):")
        for n in unique_windows:
            print(f"      {n:<12}: delta {disj_winner[f'delta_{n}']:+6.2f}pp  pnl {disj_winner[f'pnl_{n}']:+6.2f}%")
        print(f"    min_delta (across 3 disjoint) = {disj_winner['min_delta_disj']:+.2f}pp")
        print(f"    avg_delta (across 3 disjoint) = {disj_winner['avg_delta_disj']:+.2f}pp")

    # AGREEMENT CHECK: how often do cum_90d top configs survive the disjoint test?
    print(f"\n{'='*100}")
    print(f"  90D AGGREGATION HONESTY CHECK")
    print(f"{'='*100}")
    cum_top_n = results['cum_90d'].sort_values('delta_vs_base', ascending=False).head(50)
    n_pass_disj = sum(k in disj_keys for k in cum_top_n['key'])
    print(f"\n  Of TOP 50 cum_90d winners, {n_pass_disj} ({n_pass_disj/50*100:.0f}%) pass disjoint STRICT-3-of-3.")
    n_pass_cum = sum(k in cum_keys for k in cum_top_n['key'])
    print(f"  Of TOP 50 cum_90d winners, {n_pass_cum} ({n_pass_cum/50*100:.0f}%) pass cumulative STRICT-3-of-3.")
    print()
    if n_pass_disj < 10:
        print(f"  WARNING: cum_90d top configs RARELY survive the disjoint test.")
        print(f"  -> 90d aggregate hides regime variance. Disjoint slicing is more honest.")
    elif n_pass_disj > 30:
        print(f"  GOOD: cum_90d top configs USUALLY survive the disjoint test.")
        print(f"  -> 90d aggregate is reasonably honest. Cumulative window is fine.")
    else:
        print(f"  MIXED: ~half of cum_90d winners survive disjoint check.")
        print(f"  -> 90d gives some signal but disjoint adds real filtering value.")

    # Save full results
    out = os.path.join(ENGINE, 'output',
                       f'tg_window_decomp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Combined summary
    combined = []
    for name, df in results.items():
        df_copy = df.copy()
        df_copy['window'] = name
        combined.append(df_copy)
    pd.concat(combined).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
