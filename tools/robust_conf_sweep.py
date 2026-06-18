"""2026-06-17 (Lever A cont.): cross-window CONFIDENCE robustness for the live config.

The 2880h cross-window run held confidence fixed (bull@90 / bear@65). This sweeps it —
free, because confidence is a sim-layer threshold on the SAME cached 2880h signals
(no regeneration). Detector = live sma48>sma100, models = prod 8h w163 / 5h w169.
Ranks each (bull_conf, bear_conf) by cross-window robustness + worst-window (downside-
weighted, per the user's stable-growth goal).

Run: python tools/robust_conf_sweep.py
"""
import os, sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('FAYE_LIBRARY_MODE', '1'); os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
os.environ.setdefault('FAYE_MODELS_DIR', 'models')
import warnings; warnings.filterwarnings('ignore')

FEE = 0.0011; MIN_SELL, MAX_HOLD = 0.0, 10
WIN_H, STEP_H = 720, 240
BULL_PKL = 'data/_robust_bull_8h_w163_2880h.pkl'
BEAR_PKL = 'data/_robust_w169prod_2880h.pkl'
BULL_CONFS = [75, 80, 85, 90, 95]
BEAR_CONFS = [55, 60, 65, 70, 75]
LIVE = (90, 65)


def load(p):
    return {x['dt']: x for x in pickle.load(open(p, 'rb'))}


def sim(bull, bear, regfn, dts, bc, rc):
    cash = 1000.0; qty = 0.0; inpos = False; entry = 0.0; hold = 0
    for dt in dts:
        is_bull = regfn(dt); a = bull[dt] if is_bull else bear[dt]
        p = a['close']; thr = bc if is_bull else rc
        if a['sig'] == 'BUY' and a['conf'] >= thr and not inpos:
            qty = cash*(1-FEE)/p; cash = 0.0; inpos = True; entry = p; hold = 0
        elif a['sig'] == 'SELL' and inpos:
            cur = (p/entry-1)*100
            if cur >= MIN_SELL or hold >= MAX_HOLD:
                cash = qty*p*(1-FEE); inpos = False; qty = 0.0; entry = 0.0; hold = 0
        if inpos: hold += 1
    if inpos:
        last = dts[-1]; p = (bull if regfn(last) else bear)[last]['close']; cash = qty*p*(1-FEE)
    return (cash/1000.0-1)*100


def main():
    import crypto_trading_system_faye as F
    bull = load(BULL_PKL); bear = load(BEAR_PKL)
    ind, detectors = F._build_regime_indicators_and_detectors('ETH')
    regfn = detectors['sma48>sma100']
    dts = sorted(set(bull) & set(bear)); N = len(dts)
    wins = [(s, s+WIN_H) for s in range(0, N-WIN_H+1, STEP_H)]
    print(f"  span {dts[0]}->{dts[-1]} ({N} bars) | {len(wins)} windows | detector=sma48>sma100, models=prod 8h w163 / 5h w169")
    rows = []
    for bc in BULL_CONFS:
        for rc in BEAR_CONFS:
            wr = [sim(bull, bear, regfn, dts[a:b], bc, rc) for a, b in wins]
            rows.append(dict(bc=bc, rc=rc, wr=wr, median=float(np.median(wr)), worst=float(min(wr)),
                             std=float(np.std(wr)), pos=sum(1 for x in wr if x > 0),
                             full=sim(bull, bear, regfn, dts, bc, rc)))
    # downside-weighted robustness (per user's stable-growth goal): median + worst
    for r in rows:
        r['robust_dn'] = r['median'] + r['worst']      # rewards high median AND shallow worst
    rows.sort(key=lambda r: r['robust_dn'], reverse=True)
    live = next(r for r in rows if (r['bc'], r['rc']) == LIVE)
    print(f"\n  {'rank':>4}  {'bull':>4} {'bear':>4} {'median':>7} {'worst':>7} {'std':>6} {'pos':>5} {'med+worst':>9} {'full':>7}")
    print("  " + "-"*70)
    for i, r in enumerate(rows, 1):
        tag = '  <== LIVE 90/65' if (r['bc'], r['rc']) == LIVE else ''
        print(f"  {i:>4}  {r['bc']:>4} {r['rc']:>4} {r['median']:>+6.1f} {r['worst']:>+6.1f} {r['std']:>5.1f} "
              f"{r['pos']:>2}/{len(wins):<2} {r['robust_dn']:>+8.1f} {r['full']:>+6.1f}{tag}")
    best = rows[0]
    print(f"\n  LIVE (90/65): median {live['median']:+.1f} worst {live['worst']:+.1f} med+worst {live['robust_dn']:+.1f} (rank {rows.index(live)+1}/{len(rows)})")
    print(f"  Most downside-robust: bull@{best['bc']}/bear@{best['rc']}  med+worst {best['robust_dn']:+.1f} "
          f"(median {best['median']:+.1f} worst {best['worst']:+.1f}) vs live {live['robust_dn']:+.1f} -> margin {best['robust_dn']-live['robust_dn']:+.1f}pp")


if __name__ == '__main__':
    main()
