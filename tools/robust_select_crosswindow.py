"""2026-06-17 (Lever A): cross-window robustness selection + promotion hysteresis.

The problem: Mode H/S/T pick the single-window PEAK -> the winner shifts as the window
moves -> the live config churns, no stable growth. Lever A fixes the SELECTION CRITERION:
score each candidate across MANY rolling windows, prefer the consistently-good one
(median / worst-case / low-variance), and only "promote" a challenger over the incumbent
(live config) if it clears a HYSTERESIS margin across windows.

Sweeps detector x bear-model (bull held at prod 8h w163) using the REAL engine detectors
(_build_regime_indicators_and_detectors), over rolling windows. Generates its own signal
streams per --replay (cached), so it works on any span.

  python tools/robust_select_crosswindow.py --replay 1440   # quick (3 windows)
  python tools/robust_select_crosswindow.py --replay 2880   # rigorous (~10 windows; ~80 min gen)
"""
import os, sys, pickle, math, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
os.environ.setdefault('FAYE_MODELS_DIR', 'models')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

ASSET = 'ETH'
FEE = 0.0011
BULL_CONF, BEAR_CONF = 90.0, 65.0
MIN_SELL, MAX_HOLD = 0.0, 10          # live policy (min_sell=0 -> shields inert; no rally gate here)
WIN_H, STEP_H = 720, 240              # rolling 30d windows, stepped 15d
HYST_MARGIN = 5.0                     # challenger must beat incumbent median by >= this (pp)
# candidates: bull held at prod 8h w163; bear in {w169 prod, w300 alt}
CONFIGS = {
    'bull_8h_w163': dict(h=8, combo='XGB+LGBM', w=163, g=0.9985),
    'w169 (prod)':  dict(h=5, combo='RF+LGBM',  w=169, g=0.9998),
    'w300 (alt)':   dict(h=5, combo='RF+LGBM',  w=300, g=0.9998),
}
BULL_KEY = 'bull_8h_w163'
BEAR_KEYS = ['w169 (prod)', 'w300 (alt)']
INCUMBENT = ('sma48>sma100', 'w169 (prod)')


def _prod_feats():
    df = pd.read_csv('models/crypto_ed_production.csv')
    return {h: df[(df.coin == ASSET) & (df.horizon == h)].sort_values('combined_score', ascending=False).iloc[0]['optimal_features'].split(',')
            for h in (5, 8)}


def _pkl(name, replay):
    safe = name.replace(' ', '').replace('(', '').replace(')', '')
    return f'data/_robust_{safe}_{replay}h.pkl'


def _gen_all(replay, F):
    feats = _prod_feats()
    for name, c in CONFIGS.items():
        p = _pkl(name, replay)
        if os.path.exists(p):
            print(f"  {name}: cached"); continue
        t0 = time.time()
        sigs = F.generate_signals(ASSET, c['combo'].split('+'), c['w'], replay,
                                  feature_override=feats[c['h']], horizon=c['h'], gamma=c['g'])
        out = [dict(dt=pd.to_datetime(s['datetime']), close=float(s['close']),
                    sig=s['signal'], conf=float(s['confidence'])) for s in sigs]
        pickle.dump(out, open(p, 'wb'))
        print(f"  {name}: {len(out)} sigs in {time.time()-t0:.0f}s -> {p}", flush=True)


def _load(name, replay):
    return {x['dt']: x for x in pickle.load(open(_pkl(name, replay), 'rb'))}


def sim(bull_sigs, bear_sigs, regfn, dts):
    cash = 1000.0; qty = 0.0; inpos = False; entry = 0.0; hold = 0; trades = 0; wins = 0
    for dt in dts:
        is_bull = regfn(dt)
        a = bull_sigs[dt] if is_bull else bear_sigs[dt]
        p = a['close']; thr = BULL_CONF if is_bull else BEAR_CONF
        if a['sig'] == 'BUY' and a['conf'] >= thr and not inpos:
            qty = cash*(1-FEE)/p; cash = 0.0; inpos = True; entry = p; hold = 0
        elif a['sig'] == 'SELL' and inpos:
            cur = (p/entry - 1)*100
            if cur >= MIN_SELL or hold >= MAX_HOLD:
                cash = qty*p*(1-FEE); trades += 1; wins += 1 if cur > 0 else 0
                inpos = False; qty = 0.0; entry = 0.0; hold = 0
        if inpos: hold += 1
    if inpos:
        last = dts[-1]; p = (bull_sigs if regfn(last) else bear_sigs)[last]['close']
        cash = qty*p*(1-FEE); trades += 1; wins += 1 if (p/entry-1) > 0 else 0
    return (cash/1000.0 - 1)*100, trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay', type=int, default=2880)
    args = ap.parse_args()
    replay = args.replay
    import crypto_trading_system_faye as F
    _gen_all(replay, F)

    bull = _load(BULL_KEY, replay)
    bears = {k: _load(k, replay) for k in BEAR_KEYS}
    ind, detectors = F._build_regime_indicators_and_detectors(ASSET)
    dts = sorted(set(bull).intersection(*[set(b) for b in bears.values()]))
    N = len(dts)
    wins_idx = [(s, s + WIN_H) for s in range(0, N - WIN_H + 1, STEP_H)]
    print(f"\n  span: {dts[0]} -> {dts[-1]} ({N} bars) | {len(wins_idx)} rolling {WIN_H}h windows (step {STEP_H}h)")
    print("  detector bull%:", {d: f"{100*sum(fn(t) for t in dts)/N:.0f}%" for d, fn in detectors.items()})

    rows = []
    for det, regfn in detectors.items():
        for bm in BEAR_KEYS:
            wr = [sim(bull, bears[bm], regfn, dts[a:b])[0] for (a, b) in wins_idx]
            full, ntr = sim(bull, bears[bm], regfn, dts)
            rows.append(dict(det=det, bear=bm, wr=wr, full=full, ntr=ntr,
                             median=float(np.median(wr)), worst=float(min(wr)),
                             mean=float(np.mean(wr)), std=float(np.std(wr)),
                             pos=sum(1 for x in wr if x > 0)))
    inc = next(r for r in rows if (r['det'], r['bear']) == INCUMBENT)
    for r in rows:
        r['robust'] = r['median'] - 0.5*r['std']
        r['beats_inc'] = sum(1 for a, b in zip(r['wr'], inc['wr']) if a > b)
    rows.sort(key=lambda r: r['robust'], reverse=True)

    print(f"\n  INCUMBENT (live) = {INCUMBENT[0]} + bear {INCUMBENT[1]}  median={inc['median']:+.1f} worst={inc['worst']:+.1f} std={inc['std']:.1f}")
    print(f"\n  {'rank':>4}  {'detector':<14} {'bear':<11} {'median':>7} {'worst':>7} {'std':>6} {'pos':>5} {'robust':>7} {'full':>7} {'>inc':>5}")
    print("  " + "-"*88)
    for i, r in enumerate(rows, 1):
        tag = '  <== INCUMBENT' if (r['det'], r['bear']) == INCUMBENT else ''
        print(f"  {i:>4}  {r['det']:<14} {r['bear']:<11} {r['median']:>+6.1f} {r['worst']:>+6.1f} "
              f"{r['std']:>5.1f} {r['pos']:>2}/{len(wins_idx):<2} {r['robust']:>+6.1f} {r['full']:>+6.1f} "
              f"{r['beats_inc']:>2}/{len(wins_idx):<2}{tag}")

    nwin = len(wins_idx); need = math.ceil(2*nwin/3)
    print(f"\n  HYSTERESIS: challenger replaces incumbent only if it beats it in >={need}/{nwin} windows AND median margin >= +{HYST_MARGIN:.0f}pp")
    chal = [r for r in rows if (r['det'], r['bear']) != INCUMBENT
            and r['beats_inc'] >= need and (r['median'] - inc['median']) >= HYST_MARGIN]
    if chal:
        print(f"  -> {len(chal)} config(s) CLEAR the bar:")
        for r in chal:
            print(f"     {r['det']} + {r['bear']}: {r['beats_inc']}/{nwin} windows, median {r['median']:+.1f} vs {inc['median']:+.1f} (+{r['median']-inc['median']:.1f}pp)")
    else:
        rb = max(rows, key=lambda r: r['robust'])
        print(f"  -> NO challenger clears the bar -> KEEP incumbent, DON'T churn.")
        print(f"     (most-robust: {rb['det']} + {rb['bear']} robust={rb['robust']:+.1f} vs incumbent {inc['robust']:+.1f}; "
              f"margin {rb['median']-inc['median']:+.1f}pp, beats {rb['beats_inc']}/{nwin})")


if __name__ == '__main__':
    main()
