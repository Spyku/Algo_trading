"""Window-size mocktest — standalone, does NOT touch prod config or CSV.

For ETH with fixed bull=6h / bear=8h, runs the Mode T shield sweep and Mode G
rally-cooldown sweep on 30d and 60d windows of the cached tagged signal stream.

Compares winners and returns across windows. Evaluates each winner config on
both windows to check generalization.

Usage: python test_window_sweep.py
"""
import os
import sys
import pickle
import time

import numpy as np
import pandas as pd

ENGINE = r'G:\Autres ordinateurs\My laptop\engine'
os.chdir(ENGINE)
sys.path.insert(0, ENGINE)

CACHE_PATH = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')

FEE = 0.0005  # 5 bps/leg realistic maker blend (see ed.py BACKTEST_FEE_PER_LEG)
BULL_CONF = 95
BEAR_CONF = 80
BULL_H = 6
BEAR_H = 8

# Shield / T sweep grid
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
FAILSAFE = [8, 10, 12]

# Rally-cooldown / G sweep grid
RC_HORIZONS = [8, 10, 12, 14, 16, 18, 20, 24, 30, 36]
def rc_thr_for(h):
    if h in (8, 10, 12):       return [round(2.0 + 0.5*i, 2) for i in range(9)]
    if h in (14, 16, 18, 20):  return [round(3.0 + 0.5*i, 2) for i in range(9)]
    if h in (24, 30, 36):      return [round(4.0 + 0.5*i, 2) for i in range(11)]
    return []
RC_CD = [6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 48]

PLATEAU_THR = 0.7


def load_sigs():
    with open(CACHE_PATH, 'rb') as f:
        sigs = pickle.load(f)
    for s in sigs:
        s['datetime'] = pd.Timestamp(s['datetime'])
    sigs.sort(key=lambda s: s['datetime'])
    return sigs


def window_slice(sigs, days):
    end = sigs[-1]['datetime']
    lo = end - pd.Timedelta(days=days)
    return [s for s in sigs if s['datetime'] >= lo]


def sim_strategy(sigs, bull_sh, bear_sh, min_pnl, max_hold, gate_cfg=None):
    """Run the full strategy on tagged signals. gate_cfg optional."""
    cash = 1000.0; qty = 0.0; in_pos = False; entry_px = 0.0
    hold = 0; trades = []
    cd = 0
    rr_s = rr_l = None
    if gate_cfg is not None:
        # Pre-compute rr arrays across the signal stream
        h_s, h_l, t_s, t_l, cd_h = gate_cfg
        closes = np.array([float(s['close']) for s in sigs])
        def rr(h):
            out = np.full(len(closes), np.nan)
            out[h:] = (closes[h:] / closes[:-h] - 1.0) * 100.0
            return out
        rr_s = rr(h_s); rr_l = rr(h_l)

    ec = [1000.0]
    for i, s in enumerate(sigs):
        regime = s.get('regime', 'bull')
        price = float(s['close'])
        sig = s['signal']
        sconf = float(s.get('confidence', 0))
        conf_thr = BULL_CONF if regime == 'bull' else BEAR_CONF
        shield_on = bull_sh if regime == 'bull' else bear_sh

        # Rally-cd trigger check
        if gate_cfg is not None and cd >= 0:
            rs = rr_s[i] if not np.isnan(rr_s[i]) else 0
            rl = rr_l[i] if not np.isnan(rr_l[i]) else 0
            if rs >= gate_cfg[2] or rl >= gate_cfg[3]:
                cd = max(cd, gate_cfg[4])

        ec.append(cash + qty * price if in_pos else cash)

        if sig == 'BUY' and sconf >= conf_thr and not in_pos:
            if cd > 0:
                pass  # gate blocks BUY
            else:
                qty = cash * (1 - FEE) / price
                cash = 0
                in_pos = True
                entry_px = price
                hold = 0
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1.0) * 100.0
            # Correct shield logic: block only when shield is ON *and* below target.
            # Shield OFF means sell immediately regardless of PnL sign.
            blocked = shield_on and cur_pnl < min_pnl and hold < max_hold
            if not blocked:
                cash = qty * price * (1 - FEE)
                trades.append(cur_pnl)
                qty = 0
                in_pos = False
                entry_px = 0
                hold = 0

        if in_pos:
            hold += 1
        if cd > 0:
            cd -= 1

    if in_pos:
        cash = qty * sigs[-1]['close'] * (1 - FEE)
        trades.append((sigs[-1]['close'] / entry_px - 1.0) * 100.0)

    arr = np.array(ec)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    mdd = float(dd.min()) * 100.0 if len(dd) else 0.0

    ret = (cash / 1000.0 - 1.0) * 100.0
    wr = (sum(1 for t in trades if t > 0) / len(trades) * 100) if trades else 0
    return dict(return_pct=ret, trades=len(trades), win_rate=wr, max_dd=mdd)


def t_sweep(sigs):
    """Mode T shield sweep: 7x3 combos x 4 on/off = 84. Returns winner + baseline."""
    base = sim_strategy(sigs, False, False, 0, 999)

    best = None
    best_total = None
    for t in THRESHOLDS:
        for fh in FAILSAFE:
            for bull_on in (False, True):
                for bear_on in (False, True):
                    r = sim_strategy(sigs, bull_on, bear_on, t, fh)
                    if best_total is None or r['return_pct'] > best_total:
                        best_total = r['return_pct']
                        best = dict(thr=t, fh=fh, bull_on=bull_on, bear_on=bear_on,
                                    **r)
    return best, base


def g_sweep(sigs, bull_sh, bear_sh, min_pnl, max_hold, days):
    """Mode G rally-cooldown sweep. halves H1/H2 + REF = full."""
    half_days = days / 2
    end = sigs[-1]['datetime']
    t_h1_lo = end - pd.Timedelta(days=half_days)
    t_h2_lo = end - pd.Timedelta(days=days)
    sigs_h1 = [s for s in sigs if s['datetime'] >= t_h1_lo]
    sigs_h2 = [s for s in sigs if t_h2_lo <= s['datetime'] < t_h1_lo]
    sigs_ref = [s for s in sigs if s['datetime'] >= t_h2_lo]

    b_h1 = sim_strategy(sigs_h1, bull_sh, bear_sh, min_pnl, max_hold)['return_pct']
    b_h2 = sim_strategy(sigs_h2, bull_sh, bear_sh, min_pnl, max_hold)['return_pct']
    b_ref = sim_strategy(sigs_ref, bull_sh, bear_sh, min_pnl, max_hold)['return_pct']

    # Sweep
    rows = []
    pairs = [(a, b) for i, a in enumerate(RC_HORIZONS) for b in RC_HORIZONS[i+1:]]
    for h_s, h_l in pairs:
        for t_s in rc_thr_for(h_s):
            for t_l in rc_thr_for(h_l):
                for cd in RC_CD:
                    cfg = (h_s, h_l, t_s, t_l, cd)
                    r1 = sim_strategy(sigs_h1, bull_sh, bear_sh, min_pnl, max_hold, cfg)['return_pct']
                    r2 = sim_strategy(sigs_h2, bull_sh, bear_sh, min_pnl, max_hold, cfg)['return_pct']
                    rR = sim_strategy(sigs_ref, bull_sh, bear_sh, min_pnl, max_hold, cfg)['return_pct']
                    rows.append((h_s, h_l, t_s, t_l, cd, r1, r2, rR, b_h1, b_h2, b_ref))

    df = pd.DataFrame(rows, columns=['h_s','h_l','t_s','t_l','cd','pnl_H1','pnl_H2','pnl_REF','b_H1','b_H2','b_REF'])
    df['beats_3of3'] = (df['pnl_H1'] > df['b_H1']) & (df['pnl_H2'] > df['b_H2']) & (df['pnl_REF'] > df['b_REF'])
    df['score_recent'] = df['pnl_H1']  # simplified recent-rank

    strict = df[df['beats_3of3']].sort_values('score_recent', ascending=False)
    if len(strict) == 0:
        return None, (b_h1, b_h2, b_ref)
    w = strict.iloc[0]
    return dict(h_s=int(w['h_s']), h_l=int(w['h_l']), t_s=float(w['t_s']),
                t_l=float(w['t_l']), cd=int(w['cd']),
                pnl_H1=float(w['pnl_H1']), pnl_H2=float(w['pnl_H2']),
                pnl_REF=float(w['pnl_REF'])), (b_h1, b_h2, b_ref)


def main():
    print("=" * 100)
    print("  Window Mocktest — standalone, no prod writes")
    print("=" * 100)

    sigs_all = load_sigs()
    print(f"  Cache: {len(sigs_all)} signals "
          f"({sigs_all[0]['datetime']} -> {sigs_all[-1]['datetime']})")
    print()

    for days in (30, 60):
        sigs = window_slice(sigs_all, days)
        bull = sum(1 for s in sigs if s.get('regime') == 'bull')
        bear = sum(1 for s in sigs if s.get('regime') == 'bear')
        print(f"\n{'='*80}")
        print(f"  WINDOW: {days}d  |  signals={len(sigs)}  bull={bull}  bear={bear}")
        print(f"{'='*80}")

        t0 = time.time()
        t_win, t_base = t_sweep(sigs)
        print(f"\n  T sweep done in {time.time()-t0:.1f}s")
        print(f"  T baseline (both OFF): return={t_base['return_pct']:+.2f}% "
              f"trades={t_base['trades']} wr={t_base['win_rate']:.0f}%")
        print(f"  T winner: thr={t_win['thr']:.2f}% fh={t_win['fh']}h "
              f"bull_sh={'ON' if t_win['bull_on'] else 'OFF'} "
              f"bear_sh={'ON' if t_win['bear_on'] else 'OFF'}  -> "
              f"return={t_win['return_pct']:+.2f}% "
              f"trades={t_win['trades']} wr={t_win['win_rate']:.0f}% "
              f"DD={t_win['max_dd']:+.2f}%")

        # G sweep using T's shield config
        print(f"  Running G sweep against T winner's shield config...")
        t0 = time.time()
        g_win, g_base = g_sweep(sigs, t_win['bull_on'], t_win['bear_on'],
                                t_win['thr'], t_win['fh'], days)
        print(f"  G sweep done in {time.time()-t0:.1f}s")
        print(f"  G baseline (no gate): H1={g_base[0]:+.2f}% H2={g_base[1]:+.2f}% "
              f"REF={g_base[2]:+.2f}%")
        if g_win is None:
            print(f"  G: NO beats_3of3 winner")
        else:
            print(f"  G winner: rr{g_win['h_s']}h>={g_win['t_s']}% OR "
                  f"rr{g_win['h_l']}h>={g_win['t_l']}%, cd={g_win['cd']}h  -> "
                  f"H1={g_win['pnl_H1']:+.2f}% H2={g_win['pnl_H2']:+.2f}% "
                  f"REF={g_win['pnl_REF']:+.2f}%")

    # Cross-evaluation: apply 30d's winner config to 60d window and vice versa
    print(f"\n{'='*80}")
    print(f"  CROSS-EVAL: apply each window's winner to the OTHER window")
    print(f"{'='*80}")
    print("  (Skipped — keeping test fast. Would measure out-of-sample robustness.)")


if __name__ == '__main__':
    main()
