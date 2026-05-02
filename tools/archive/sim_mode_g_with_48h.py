"""
sim_mode_g_with_48h.py — replicates Mode G's _sweep_rally_cooldown EXACTLY
but with HORIZONS extended to include 48h. Tests whether Mode G — given
the longer lookback in its search space — would find a config close to
the rally_48h>=4% gate we identified manually.

CRITICAL: Mode G's mechanism = trigger + N-hour cooldown timer. Our
finding's mechanism = continuous current-state check. Same lookback (48h)
in different frameworks WILL give different gates. This script tests what
Mode G's framework picks WITH 48h available.

Reads: cache + regime_config_ed.json. Writes nothing (pure simulation).
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

# Match Mode G's existing parameters EXACTLY, with 48h appended
HORIZONS_PROD = [8, 10, 12, 14, 16, 18, 20, 24, 30, 36]
HORIZONS_EXTENDED = [8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 48]
CD_GRID = [6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 48]
PLATEAU_THR = 0.7


def thr_for(h):
    if h in (8, 10, 12):       return [round(2.0 + 0.5*i, 2) for i in range(9)]
    if h in (14, 16, 18, 20):  return [round(3.0 + 0.5*i, 2) for i in range(9)]
    if h in (24, 30, 36):      return [round(4.0 + 0.5*i, 2) for i in range(11)]
    if h == 48:                return [round(4.0 + 0.5*i, 2) for i in range(11)]  # 4.0 .. 9.0
    raise ValueError(h)


def load_signals():
    with open(CACHE_PKL, 'rb') as f:
        sigs = pickle.load(f)
    for s in sigs:
        t = pd.Timestamp(s['datetime'])
        s['datetime'] = t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    sigs.sort(key=lambda s: s['datetime'])
    return sigs


def simulate(sigs, rr_dict, h_s, h_l, t_s, t_l, cd_h, asset_cfg, regime_filter='all',
             other_regime_gate=None):
    """Replicates Mode G's simulate() exactly."""
    bull_shield = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    bull_conf = float(asset_cfg.get('bull', {}).get('min_confidence', 75.0))
    bear_conf = float(asset_cfg.get('bear', {}).get('min_confidence', 65.0))
    min_sell_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.5))
    max_hold = int(asset_cfg.get('max_hold_hours', 10))

    other_regime = 'bear' if regime_filter == 'bull' else ('bull' if regime_filter == 'bear' else None)
    o_hs = o_hl = 0
    o_ts = o_tl = 0.0
    o_cd_h = 0
    o_rs_arr = o_rl_arr = None
    if other_regime is not None and other_regime_gate is not None:
        try:
            o_hs = int(other_regime_gate['h_short']); o_hl = int(other_regime_gate['h_long'])
            o_ts = float(other_regime_gate['t_short_pct']); o_tl = float(other_regime_gate['t_long_pct'])
            o_cd_h = int(other_regime_gate['cd_hours'])
            o_rs_arr = rr_dict.get(o_hs); o_rl_arr = rr_dict.get(o_hl)
        except (KeyError, TypeError, ValueError):
            o_cd_h = 0

    cash = 1000.0; qty = 0.0; in_pos = False; entry = 0.0
    hold = 0; trades = 0; skipped = 0; cd = 0; other_cd = 0
    ec = [1000.0]
    n = len(sigs)
    rs_arr = rr_dict.get(h_s); rl_arr = rr_dict.get(h_l)
    for i in range(n):
        s = sigs[i]; price = s['close']
        regime = s.get('regime', 'bull')
        conf_thr = bull_conf if regime == 'bull' else bear_conf
        if cd_h > 0 and rs_arr is not None and rl_arr is not None:
            if regime_filter == 'all' or regime_filter == regime:
                rs = rs_arr[i]; rl = rl_arr[i]
                if (rs == rs and rs >= t_s) or (rl == rl and rl >= t_l):
                    cd = max(cd, cd_h)
        if o_cd_h > 0 and o_rs_arr is not None and regime == other_regime:
            rs = o_rs_arr[i]; rl = o_rl_arr[i]
            if (rs == rs and rs >= o_ts) or (rl == rl and rl >= o_tl):
                other_cd = max(other_cd, o_cd_h)
        ec.append(cash + qty * price if in_pos else cash)
        if s['signal'] == 'BUY' and s['confidence'] >= conf_thr and not in_pos:
            active_cd = cd if (regime_filter == 'all' or regime == regime_filter) else other_cd
            if active_cd > 0:
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
        if other_cd > 0: other_cd -= 1
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


def run_mode_g_sweep(sigs, asset_cfg, replay_h, horizons_set, label):
    """Run a single regime_filter='all' sweep on the given horizons set."""
    days = replay_h / 24.0
    half_days = days / 2.0

    end_t = sigs[-1]['datetime']
    t_h1_lo = end_t - pd.Timedelta(days=half_days)
    t_h2_lo = end_t - pd.Timedelta(days=days)
    sigs_h1 = [s for s in sigs if s['datetime'] >= t_h1_lo]
    sigs_h2 = [s for s in sigs if t_h2_lo <= s['datetime'] < t_h1_lo]
    sigs_ref = [s for s in sigs if s['datetime'] >= t_h2_lo]

    rr_h1 = build_rr(sigs_h1, horizons_set)
    rr_h2 = build_rr(sigs_h2, horizons_set)
    rr_ref = build_rr(sigs_ref, horizons_set)

    # Baselines (no gate)
    b_h1 = simulate(sigs_h1, {}, 8, 36, 9999, 9999, 0, asset_cfg)
    b_h2 = simulate(sigs_h2, {}, 8, 36, 9999, 9999, 0, asset_cfg)
    b_ref = simulate(sigs_ref, {}, 8, 36, 9999, 9999, 0, asset_cfg)

    print(f"\n[{label}]  baselines:  H1={b_h1['pnl_pct']:+.2f}%  "
          f"H2={b_h2['pnl_pct']:+.2f}%  REF={b_ref['pnl_pct']:+.2f}%")

    pairs = [(a, b) for i, a in enumerate(horizons_set) for b in horizons_set[i+1:]]
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
    df['score_recent'] = df['delta_h1'] - 0.5 * abs(df['worst_dd'])

    # STRICT filter: beats_3of3 + plateau ≥ 0.7
    print(f"  Total configs: {len(df)}, beats_3of3: {df['beats_3of3'].sum()}")

    strict = df[df['beats_3of3']].copy()
    if len(strict) == 0:
        print(f"  NO STRICT winners. Best beats_2of3 by ref delta:")
        any_beats_ref = df[df['delta_ref'] > 0].copy().sort_values('delta_ref', ascending=False)
        print(any_beats_ref.head(10).to_string(index=False))
        return None

    # Plateau check: for each candidate, count fraction of ±1 STRICT neighbors also passing STRICT
    # Mode G uses neighbor density on the (h_s, h_l, t_s, t_l, cd_h) grid.
    # Simplified plateau: for each STRICT row, count how many other STRICT rows share same (h_s, h_l)
    # with thresholds within ±1 step and cd within ±2 step. Plateau = neighbors / max_neighbors.
    # Approximation; close enough for ranking.
    strict_idx = set()
    for r in strict.itertuples():
        strict_idx.add((r.h_s, r.h_l, r.t_s, r.t_l, r.cd_h))
    plateau = []
    for r in strict.itertuples():
        neighbors_total = 0
        neighbors_strict = 0
        for d_ts in [-0.5, 0, 0.5]:
            for d_tl in [-0.5, 0, 0.5]:
                for d_cd in [-2, 0, 2]:
                    if d_ts == 0 and d_tl == 0 and d_cd == 0:
                        continue
                    nt_s = round(r.t_s + d_ts, 2)
                    nt_l = round(r.t_l + d_tl, 2)
                    n_cd = r.cd_h + d_cd
                    if nt_s in thr_for(r.h_s) and nt_l in thr_for(r.h_l) and n_cd in CD_GRID:
                        neighbors_total += 1
                        if (r.h_s, r.h_l, nt_s, nt_l, n_cd) in strict_idx:
                            neighbors_strict += 1
        plateau.append(neighbors_strict / max(1, neighbors_total))
    strict['plateau_score'] = plateau

    # Top-15 by score_dd_aware
    top15 = strict.sort_values('score_dd_aware', ascending=False).head(15)
    print(f"\n  Top-15 STRICT (sorted by score_dd_aware):")
    show_cols = ['h_s', 'h_l', 't_s', 't_l', 'cd_h', 'h1', 'h2', 'ref',
                 'delta_h1', 'delta_h2', 'delta_ref', 'worst_dd', 'plateau_score', 'skipped_ref']
    print(top15[show_cols].round(2).to_string(index=False))

    # Filter to plateau ≥ PLATEAU_THR
    plateau_pass = strict[strict['plateau_score'] >= PLATEAU_THR].sort_values('score_dd_aware', ascending=False)
    print(f"\n  STRICT + plateau >= {PLATEAU_THR}: {len(plateau_pass)} configs")
    if len(plateau_pass) == 0:
        print("  NO plateau-passing winner.")
        return None

    winner = plateau_pass.iloc[0].to_dict()
    print(f"\n  WINNER: h_s={int(winner['h_s'])} h_l={int(winner['h_l'])} "
          f"t_s={winner['t_s']} t_l={winner['t_l']} cd={int(winner['cd_h'])}h")
    print(f"          REF return={winner['ref']:.2f}% (delta {winner['delta_ref']:+.2f}pp), "
          f"H1 delta={winner['delta_h1']:+.2f}, H2 delta={winner['delta_h2']:+.2f}, "
          f"plateau={winner['plateau_score']:.2f}, "
          f"skipped/REF={int(winner['skipped_ref'])}")
    return winner, top15, df


def main():
    print("Loading cache + config...")
    sigs = load_signals()
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    # Use 60d (1440h) — match standard Mode G replay
    replay_h = 1440

    print(f"\n{'='*100}")
    print(f"  MODE G COMPARISON: PROD HORIZONS vs HORIZONS+48h on ETH 60d")
    print(f"{'='*100}")
    print(f"  PROD HORIZONS:     {HORIZONS_PROD}")
    print(f"  EXTENDED HORIZONS: {HORIZONS_EXTENDED}")
    print(f"  CD_GRID: {CD_GRID}")

    print("\n--- PROD search space (Mode G as it currently runs) ---")
    pairs_prod = sum(1 for i, a in enumerate(HORIZONS_PROD) for _ in HORIZONS_PROD[i+1:])
    n_prod = sum(len(thr_for(a)) * len(thr_for(b)) for i, a in enumerate(HORIZONS_PROD) for b in HORIZONS_PROD[i+1:]) * len(CD_GRID)
    print(f"  {pairs_prod} pairs, {n_prod:,} total configs")
    res_prod = run_mode_g_sweep(sigs, asset_cfg, replay_h, HORIZONS_PROD, "PROD")

    print("\n--- EXTENDED search space (PROD + 48h) ---")
    pairs_ext = sum(1 for i, a in enumerate(HORIZONS_EXTENDED) for _ in HORIZONS_EXTENDED[i+1:])
    n_ext = sum(len(thr_for(a)) * len(thr_for(b)) for i, a in enumerate(HORIZONS_EXTENDED) for b in HORIZONS_EXTENDED[i+1:]) * len(CD_GRID)
    print(f"  {pairs_ext} pairs, {n_ext:,} total configs")
    res_ext = run_mode_g_sweep(sigs, asset_cfg, replay_h, HORIZONS_EXTENDED, "PROD+48h")

    print("\n" + "=" * 100)
    print("  COMPARISON SUMMARY")
    print("=" * 100)
    if res_prod is not None and res_ext is not None:
        wp, _, _ = res_prod
        we, _, _ = res_ext
        print(f"\n  PROD winner:     h_s={int(wp['h_s'])}h>={wp['t_s']}% OR h_l={int(wp['h_l'])}h>={wp['t_l']}% "
              f"cd={int(wp['cd_h'])}h  ->  REF {wp['ref']:.2f}% (Δ{wp['delta_ref']:+.2f}pp)")
        print(f"  EXTENDED winner: h_s={int(we['h_s'])}h>={we['t_s']}% OR h_l={int(we['h_l'])}h>={we['t_l']}% "
              f"cd={int(we['cd_h'])}h  ->  REF {we['ref']:.2f}% (Δ{we['delta_ref']:+.2f}pp)")
        print(f"\n  Did EXTENDED choose 48h? "
              f"{'YES' if we['h_s'] == 48 or we['h_l'] == 48 else 'NO (still picked from old set)'}")
        print(f"  Mechanism reminder: Mode G is TRIGGER + N-hour COOLDOWN TIMER (cd={int(we['cd_h'])}h)")
        print(f"  Our manual finding: block_rally_48h>=4% — CONTINUOUS condition (no timer)")

    print("\nDone. No production files modified.")


if __name__ == '__main__':
    main()
