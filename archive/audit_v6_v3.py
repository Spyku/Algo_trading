"""V6 fine-tuning sweep — DENSE grid + plateau (robustness) score.

Same structure as v2 (OR gate on two rolling returns, block BUYs for C hours),
but denser on every axis and with a plateau score that rewards configs sitting
inside a ridge of other winners rather than isolated spikes.

Grid:
  horizons     : {8, 10, 12, 14, 16, 18, 20, 24, 30, 36}    (10 values, 45 pairs)
  thresholds   : step 0.5%, horizon-scaled
                   8/10/12h    -> 2.0 .. 6.0   (9 values)
                   14/16/18/20 -> 3.0 .. 7.0   (9 values)
                   24/30/36h   -> 4.0 .. 9.0   (11 values)
  cooldowns    : {6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 48}  (12 values)

Windows + scoring = same as v2:
  H1 = newest 30d, H2 = prior 30d (non-overlapping), 60d reference.
  Winner = beats V0 baseline PnL on BOTH H1 AND H2.

Plateau score (robustness):
  For each config, look at its 10 cross-neighbors (±1 grid-step on each of the
  5 dims: Hshort, Hlong, Tshort, Tlong, cd). Count how many of those neighbors
  also beat baseline on both halves. High = config sits on a ridge.
  Also report `nbr_avg_pnl` = mean avg_pnl_halves across the 10 neighbors.

Final pick = highest plateau_score among configs that `beat_both_halves`,
tie-broken by avg_pnl_halves.
"""
from __future__ import annotations

import os
import pickle
import time
import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ENGINE_DIR, 'data')
SIG_90D    = os.path.join(DATA_DIR, 'eth_sl_signals_90d.pkl')

TRADING_FEE      = 0.0005  # 5 bps/leg realistic maker blend (see ed.py BACKTEST_FEE_PER_LEG)
MIN_SELL_PNL_PCT = 0.005
MAX_HOLD_HOURS   = 10

HORIZONS = [8, 10, 12, 14, 16, 18, 20, 24, 30, 36]

def thresholds_for(h):
    if h in (8, 10, 12):
        return [round(2.0 + 0.5 * i, 2) for i in range(9)]          # 2.0..6.0
    if h in (14, 16, 18, 20):
        return [round(3.0 + 0.5 * i, 2) for i in range(9)]          # 3.0..7.0
    if h in (24, 30, 36):
        return [round(4.0 + 0.5 * i, 2) for i in range(11)]         # 4.0..9.0
    raise ValueError(f"no threshold grid for horizon {h}")

CD_GRID = [6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 48]


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


def simulate(signals, rr_cols_arr, h_short, h_long, t_short, t_long, cd_hours):
    """rr_cols_arr: dict {h: numpy array aligned with signals order}.
    Precomputing this speeds things up massively vs DataFrame lookup in inner loop."""
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

    rs_arr = rr_cols_arr[h_short]
    rl_arr = rr_cols_arr[h_long]

    for i in range(n):
        s = signals[i]
        price = s['close']
        sig = s['signal']; conf = s['confidence']; thr = s['conf_threshold']

        if cd_hours > 0:
            rs = rs_arr[i]; rl = rl_arr[i]
            hit_s = (rs == rs) and rs >= t_short  # NaN-safe
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
    return total_pnl, max_dd, trades, skipped


def baseline(signals, rr_cols_arr):
    pnl, dd, tr, sk = simulate(signals, rr_cols_arr, 8, 36, 9999, 9999, cd_hours=0)
    return dict(pnl_pct=pnl, dd_pct=dd, trades=tr)


def align_rr_arrays(signals, df_hourly):
    """Build {h: np.array} aligned with signals order (so we don't re-lookup per sim)."""
    rr_map = df_hourly.set_index('datetime')
    out = {}
    for h in HORIZONS:
        col = f'rr_{h}h'
        arr = np.empty(len(signals), dtype=float)
        for i, s in enumerate(signals):
            dt = s['datetime']
            if dt in rr_map.index:
                arr[i] = rr_map.at[dt, col]
            else:
                arr[i] = np.nan
        out[h] = arr
    return out


def slice_window(signals, df_hourly, start_days_back, end_days_back):
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
    print("  V6 FINE-TUNING SWEEP v3 (DENSE grid + plateau score)")
    print("=" * 100)
    signals = load_signals()
    df = build_hourly(signals)
    print(f"  signals={len(signals)}  span: {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    win_h1  = slice_window(signals, df, 30, 0)
    win_h2  = slice_window(signals, df, 60, 30)
    win_60  = slice_tail(signals, df, 60)
    windows = [('H1_30d', win_h1), ('H2_30d', win_h2), ('REF_60d', win_60)]

    for label, (ws, wd) in windows:
        print(f"  {label}: {len(ws)} signals  {wd['datetime'].iloc[0]} -> {wd['datetime'].iloc[-1]}")

    # Pre-align rr arrays per window
    rr_arrs = {label: align_rr_arrays(ws, wd) for label, (ws, wd) in windows}

    base = {}
    for label, (ws, wd) in windows:
        b = baseline(ws, rr_arrs[label])
        base[label] = b
        print(f"  baseline {label}: PnL={b['pnl_pct']:+.2f}%  DD={b['dd_pct']:.2f}%  trades={b['trades']}")

    pairs = [(a, b) for i, a in enumerate(HORIZONS) for b in HORIZONS[i + 1:]]
    total_configs = 0
    for (a, b) in pairs:
        total_configs += len(thresholds_for(a)) * len(thresholds_for(b))
    total_configs *= len(CD_GRID)
    total_sims = total_configs * len(windows)
    print(f"\n  Sweeping {len(pairs)} horizon pairs -> {total_configs} configs x "
          f"{len(windows)} windows = {total_sims:,} sims")

    t0 = time.time()
    rows = []
    progress = 0
    for (h_s, h_l) in pairs:
        ts_grid = thresholds_for(h_s)
        tl_grid = thresholds_for(h_l)
        for t_s in ts_grid:
            for t_l in tl_grid:
                for cd in CD_GRID:
                    row = {'h_short': h_s, 'h_long': h_l,
                           't_short': t_s, 't_long': t_l, 'cd': cd}
                    for label, (ws, wd) in windows:
                        pnl, dd_, tr, sk = simulate(ws, rr_arrs[label], h_s, h_l, t_s, t_l, cd)
                        row[f'pnl_{label}'] = pnl
                        row[f'dd_{label}']  = dd_
                        row[f'tr_{label}']  = tr
                        row[f'sk_{label}']  = sk
                    row['beats_H1']   = row['pnl_H1_30d']  > base['H1_30d']['pnl_pct']
                    row['beats_H2']   = row['pnl_H2_30d']  > base['H2_30d']['pnl_pct']
                    row['beats_60d']  = row['pnl_REF_60d'] > base['REF_60d']['pnl_pct']
                    row['beats_both_halves'] = row['beats_H1'] and row['beats_H2']
                    row['beats_3of3']        = row['beats_H1'] and row['beats_H2'] and row['beats_60d']
                    row['avg_pnl_halves']    = (row['pnl_H1_30d'] + row['pnl_H2_30d']) / 2.0
                    row['worst_dd']          = max(abs(row['dd_H1_30d']), abs(row['dd_H2_30d']))
                    # DD-aware composite score (Calmar-ish, half weight on DD)
                    row['score_dd_aware']    = row['avg_pnl_halves'] - 0.5 * row['worst_dd']
                    rows.append(row)
                    progress += 1
        elapsed = time.time() - t0
        rate = progress / elapsed if elapsed > 0 else 0
        eta = (total_configs - progress) / rate if rate > 0 else 0
        print(f"    pair ({h_s},{h_l}) done  {progress}/{total_configs}  "
              f"elapsed={elapsed:.0f}s  rate={rate:.0f}/s  eta={eta:.0f}s")

    out = pd.DataFrame(rows)
    out_path = os.path.join(ENGINE_DIR, 'audit_v6_v3_summary.csv')
    out.to_csv(out_path, index=False)
    print(f"\n  Wrote {out_path}  ({len(out)} configs, {time.time()-t0:.0f}s total)")

    # ----- Plateau score (cross-neighbors: ±1 step on each of the 5 dims) -----
    print("\n  Computing plateau scores...")
    h_idx = {h: i for i, h in enumerate(HORIZONS)}
    cd_idx = {c: i for i, c in enumerate(CD_GRID)}

    key_to_row = {}
    for r in rows:
        key = (r['h_short'], r['h_long'], r['t_short'], r['t_long'], r['cd'])
        key_to_row[key] = r

    def neighbor(r, dim, step):
        hs, hl, ts, tl, cd = r['h_short'], r['h_long'], r['t_short'], r['t_long'], r['cd']
        if dim == 'hs':
            i = h_idx[hs] + step
            if not (0 <= i < len(HORIZONS)): return None
            new_hs = HORIZONS[i]
            if new_hs >= hl: return None
            key = (new_hs, hl, ts, tl, cd)
        elif dim == 'hl':
            i = h_idx[hl] + step
            if not (0 <= i < len(HORIZONS)): return None
            new_hl = HORIZONS[i]
            if new_hl <= hs: return None
            # threshold grid for new hl may differ — reject if tl not on new grid
            if tl not in thresholds_for(new_hl): return None
            key = (hs, new_hl, ts, tl, cd)
        elif dim == 'ts':
            new_ts = round(ts + 0.5 * step, 2)
            if new_ts not in thresholds_for(hs): return None
            key = (hs, hl, new_ts, tl, cd)
        elif dim == 'tl':
            new_tl = round(tl + 0.5 * step, 2)
            if new_tl not in thresholds_for(hl): return None
            key = (hs, hl, ts, new_tl, cd)
        elif dim == 'cd':
            i = cd_idx[cd] + step
            if not (0 <= i < len(CD_GRID)): return None
            key = (hs, hl, ts, tl, CD_GRID[i])
        return key_to_row.get(key)

    for r in rows:
        nbrs = []
        for dim in ('hs', 'hl', 'ts', 'tl', 'cd'):
            for step in (-1, 1):
                nb = neighbor(r, dim, step)
                if nb is not None:
                    nbrs.append(nb)
        r['n_neighbors'] = len(nbrs)
        if nbrs:
            r['plateau_score']   = sum(1 for nb in nbrs if nb['beats_both_halves']) / len(nbrs)
            r['plateau_3of3']    = sum(1 for nb in nbrs if nb['beats_3of3']) / len(nbrs)
            r['nbr_avg_pnl']     = float(np.mean([nb['avg_pnl_halves'] for nb in nbrs]))
            winners = [nb for nb in nbrs if nb['beats_both_halves']]
            r['plateau_pnl_winners'] = float(np.mean([nb['avg_pnl_halves'] for nb in winners])) if winners else 0.0
        else:
            r['plateau_score'] = r['plateau_3of3'] = r['nbr_avg_pnl'] = r['plateau_pnl_winners'] = 0.0

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)  # rewrite with plateau cols
    print(f"  Wrote plateau scores to {out_path}")

    cols = ['h_short', 'h_long', 't_short', 't_long', 'cd',
            'pnl_H1_30d', 'pnl_H2_30d', 'avg_pnl_halves', 'pnl_REF_60d',
            'dd_H1_30d', 'dd_H2_30d', 'worst_dd',
            'tr_H1_30d', 'tr_H2_30d', 'sk_H1_30d', 'sk_H2_30d',
            'score_dd_aware', 'plateau_pnl_winners']
    num_fmt = {c: '{:+.2f}'.format for c in cols
               if c.startswith(('pnl_', 'dd_', 'avg_', 'nbr_', 'score_', 'worst_', 'plateau_pnl'))}
    num_fmt['worst_dd']      = '{:.2f}'.format

    n_3of3   = int(out['beats_3of3'].sum())
    n_2of2   = int(out['beats_both_halves'].sum())
    print(f"\n  Filter counts: beats_2of2={n_2of2}/{len(out)}   "
          f"beats_3of3={n_3of3}/{len(out)}   "
          f"3of3 + plateau_score>=0.7: {int(((out['beats_3of3']) & (out['plateau_score']>=0.7)).sum())}")

    # ---------- VIEW 1: peak PnL ----------
    print("\n" + "=" * 130)
    print("  VIEW 1 — TOP 15 BY avg_pnl_halves  (peak-chaser, ignores DD and robustness)")
    print("=" * 130)
    print(out.sort_values('avg_pnl_halves', ascending=False).head(15)[cols]
            .to_string(index=False, formatters=num_fmt))

    # ---------- VIEW 2: DD-aware composite ----------
    print("\n" + "=" * 130)
    print("  VIEW 2 — TOP 15 BY score_dd_aware = avg_pnl_halves - 0.5 * worst_dd  "
          "(rewards both PnL and shallow DD)")
    print("=" * 130)
    print(out.sort_values('score_dd_aware', ascending=False).head(15)[cols]
            .to_string(index=False, formatters=num_fmt))

    # ---------- VIEW 3: plateau_score (binary count) ----------
    print("\n" + "=" * 130)
    print("  VIEW 3 — TOP 15 BY plateau_score (among beats_both_halves), tie=avg_pnl_halves")
    print("  Robust to small grid perturbations; ignores DD")
    print("=" * 130)
    v3_set = out[out['beats_both_halves']].copy()
    if len(v3_set):
        print(v3_set.sort_values(['plateau_score','avg_pnl_halves'], ascending=[False, False])
                    .head(15)[cols].to_string(index=False, formatters=num_fmt))
    else:
        print("  NONE")

    # ---------- VIEW 4: STRICT — 3of3 + plateau gate + DD-aware ----------
    print("\n" + "=" * 130)
    print("  VIEW 4 — STRICT: beats_3of3 AND plateau_score>=0.7, ranked by score_dd_aware")
    print("  tie-break: plateau_pnl_winners (height of the surrounding ridge)")
    print("=" * 130)
    strict = out[(out['beats_3of3']) & (out['plateau_score'] >= 0.7)].copy()
    if len(strict):
        print(f"  ({len(strict)} configs survive)")
        print(strict.sort_values(['score_dd_aware','plateau_pnl_winners'], ascending=[False, False])
                    .head(15)[cols].to_string(index=False, formatters=num_fmt))
    else:
        print("  NONE survive — relax the filter.")

    # ---------- V6 reference for comparison ----------
    v6 = out[(out['h_short'] == 10) & (out['h_long'] == 24) &
             (out['t_short'] == 5.0) & (out['t_long'] == 7.0) & (out['cd'] == 24)]
    if len(v6):
        print("\n  V6 reference row (rr10>=5 OR rr24>=7, cd=24):")
        print(v6[cols].to_string(index=False, formatters=num_fmt))


if __name__ == '__main__':
    main()
