"""2026-06-16 (Idea 2, Step 1): per-regime candidate scoring with a PINNED detector.

Question (user): the Mode-H filter picks the best model per horizon on the FULL
picture (regime-blind) because the detector isn't known yet. If we PIN the detector
to the current best (`sma48>sma100`) up front, label bars bull/bear, and give each
candidate a BULL subscore and a BEAR subscore, do the best-bull / best-bear / best-
overall candidates come out RADICALLY DIFFERENT or the SAME?

This is the cheap standalone proxy (no engine edit). For each horizon it sweeps a
candidate grid (combo x window x gamma; features fixed to the live prod set for that
horizon), regenerates each candidate's per-bar signal stream via generate_signals,
splits bars by the pinned detector, and scores each candidate 3 ways:
  - overall : trade ALL bars, best conf of {65,75,90}, shield ON     (full-picture proxy)
  - bull    : trade BULL bars only, conf 90, shield ON               (bull deployment)
  - bear    : trade BEAR bars only, conf 65, shield OFF              (bear deployment)
score = return_pct * (win_rate/100)  (the engine's production scoring, Rule 7).

Resumable: appends each candidate to a CSV, skips ones already done.

Run:
  python tools/regime_candidate_scan.py --probe          # time ONE candidate, validate end-to-end
  python tools/regime_candidate_scan.py                  # full grid (size after probe)
  python tools/regime_candidate_scan.py --parse-only     # just re-print rankings from the CSV
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # engine root
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
os.environ.setdefault('FAYE_MODELS_DIR', 'models')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

ASSET = 'ETH'
REPLAY = 960
HORIZONS = [5, 8]
FEE = 0.0011
BULL_CONF, BEAR_CONF = 90.0, 65.0
MIN_SELL, MAX_HOLD = 0.0, 10
OUT_CSV = 'output/regime_candidate_scan_ETH.csv'

# candidate grid (combo x window); features FIXED to live prod set per horizon,
# gamma FIXED to each horizon's prod value (probe = 6.5 min/candidate on GPU, so the
# full combo x window x gamma sweep = ~19h; trimmed to combo x window to fit ~2.5-3h.
# combo + window are the structural levers; gamma is a finer knob to add only if this reorders).
COMBOS       = ['RF+LGBM', 'XGB+LGBM', 'RF+XGB']
WINDOWS_BY_H = {5: [100, 169, 250, 300], 8: [100, 163, 250, 300]}  # incl. each prod window
GAMMA_BY_H   = {5: 0.9998, 8: 0.9985}                              # each horizon's prod gamma


def _load_prod_feats():
    df = pd.read_csv('models/crypto_ed_production.csv')
    out = {}
    for h in HORIZONS:
        r = df[(df.coin == ASSET) & (df.horizon == h)].sort_values('combined_score', ascending=False)
        out[h] = r.iloc[0]['optimal_features'].split(',') if len(r) else None
    return out


def _regime_series():
    df = pd.read_csv('data/ETH_hourly_data.csv'); df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').set_index('datetime')
    sma48 = df['close'].rolling(48).mean(); sma100 = df['close'].rolling(100).mean()
    reg = {}
    for t in df.index:
        a, b = sma48.loc[t], sma100.loc[t]
        reg[t] = 'bull' if (a != a or b != b or a > b) else 'bear'  # cold-start -> bull
    return reg


def _parse_sigs(sigs, regmap):
    """Normalize generate_signals output to a list of (dt, close, sig, conf, regime)."""
    out = []
    for s in sigs:
        dt = s['datetime']
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        else:
            dt = pd.to_datetime(dt)
        out.append(dict(dt=dt, close=float(s['close']), sig=s['signal'],
                        conf=float(s['confidence']), regime=regmap.get(dt, 'bull')))
    return out


def _sim(bars, conf_thr, shield_on, regime=None):
    """Trade `bars`. If regime is 'bull'/'bear', trade ONLY that regime's bars and
    force-close at the last in-regime bar before a flip (the other model takes over).
    Returns (return_pct, trades, win_rate)."""
    cash = 1000.0; qty = 0.0; in_pos = False; entry = 0.0; hold = 0
    trades = 0; wins = 0
    n = len(bars)
    for i, b in enumerate(bars):
        p = b['close']
        active = (regime is None) or (b['regime'] == regime)
        if active:
            if b['sig'] == 'BUY' and b['conf'] >= conf_thr and not in_pos:
                qty = cash * (1 - FEE) / p; cash = 0.0; in_pos = True; entry = p; hold = 0
            elif b['sig'] == 'SELL' and in_pos:
                cur = (p / entry - 1) * 100
                if cur >= (MIN_SELL if shield_on else 0.0) or hold >= MAX_HOLD:
                    cash = qty * p * (1 - FEE); trades += 1; wins += 1 if cur > 0 else 0
                    in_pos = False; qty = 0.0; entry = 0.0; hold = 0
            if in_pos: hold += 1
        # force-close at the last in-regime bar before a flip (or final bar)
        if regime is not None and in_pos:
            nxt_out = (i + 1 >= n) or (bars[i + 1]['regime'] != regime)
            if nxt_out:
                cur = (p / entry - 1) * 100
                cash = qty * p * (1 - FEE); trades += 1; wins += 1 if cur > 0 else 0
                in_pos = False; qty = 0.0; entry = 0.0; hold = 0
    if in_pos:
        p = bars[-1]['close']; cash = qty * p * (1 - FEE); trades += 1; wins += 1 if (p / entry - 1) > 0 else 0
    ret = (cash / 1000.0 - 1) * 100
    return ret, trades, (100 * wins / trades if trades else 0.0)


def _score(bars):
    """3 deployment-faithful scores for one candidate's signal stream."""
    # overall: all bars, best conf of {65,75,90}, shield ON (full-picture proxy)
    best = None
    for c in (65, 75, 90):
        r, t, w = _sim(bars, c, True, regime=None)
        s = r * (w / 100.0)
        if best is None or s > best[0]:
            best = (s, r, t, w, c)
    ov_score, ov_ret, ov_tr, ov_wr, ov_conf = best
    # bull deployment: bull bars, conf90, shield ON
    br, bt, bw = _sim(bars, BULL_CONF, True, regime='bull')
    # bear deployment: bear bars, conf65, shield OFF
    er, et, ew = _sim(bars, BEAR_CONF, False, regime='bear')
    return dict(
        ov_score=ov_score, ov_ret=ov_ret, ov_tr=ov_tr, ov_wr=ov_wr, ov_conf=ov_conf,
        bull_score=br * (bw / 100.0), bull_ret=br, bull_tr=bt, bull_wr=bw,
        bear_score=er * (ew / 100.0), bear_ret=er, bear_tr=et, bear_wr=ew,
    )


def _ranking_report(csv=None):
    csv = csv or OUT_CSV
    df = pd.read_csv(csv)
    print("\n" + "=" * 84)
    print(f"  RANKING REPORT — {len(df)} candidates scored | detector pinned = sma48>sma100")
    print("=" * 84)
    for h in sorted(df.horizon.unique()):
        d = df[df.horizon == h].copy()
        print(f"\n  ===== {h}h ({len(d)} candidates) =====")
        for key, lab in [('ov_score', 'OVERALL (full-picture)'),
                         ('bull_score', 'BULL subscore'),
                         ('bear_score', 'BEAR subscore')]:
            top = d.sort_values(key, ascending=False).head(5)
            print(f"\n  top-5 by {lab}:")
            for _, r in top.iterrows():
                tag = ' <-PROD' if r.get('is_prod') else ''
                print(f"     {r['combo']:<9} w{int(r['window']):<3} g{r['gamma']:<7} | "
                      f"ov={r['ov_ret']:+6.2f}%/{int(r['ov_tr'])}t/{r['ov_wr']:.0f}%@{int(r['ov_conf'])}  "
                      f"bull={r['bull_ret']:+6.2f}%/{int(r['bull_tr'])}t  "
                      f"bear={r['bear_ret']:+6.2f}%/{int(r['bear_tr'])}t{tag}")
        # divergence check
        ov1 = d.sort_values('ov_score', ascending=False).iloc[0]
        bl1 = d.sort_values('bull_score', ascending=False).iloc[0]
        br1 = d.sort_values('bear_score', ascending=False).iloc[0]
        def cfg(r): return f"{r['combo']} w{int(r['window'])} g{r['gamma']}"
        print(f"\n  WINNERS: overall={cfg(ov1)} | bull={cfg(bl1)} | bear={cfg(br1)}")
        print(f"  bull-best == overall-best ? {cfg(bl1)==cfg(ov1)}   "
              f"bear-best == overall-best ? {cfg(br1)==cfg(ov1)}   "
              f"bull-best == bear-best ? {cfg(bl1)==cfg(br1)}")
        try:
            from scipy.stats import spearmanr
            rho_bb, _ = spearmanr(d['bull_score'], d['bear_score'])
            rho_bo, _ = spearmanr(d['bull_score'], d['ov_score'])
            rho_eo, _ = spearmanr(d['bear_score'], d['ov_score'])
            print(f"  Spearman rank-corr: bull~bear={rho_bb:+.2f}  bull~overall={rho_bo:+.2f}  bear~overall={rho_eo:+.2f}")
            print("  (low bull~bear => regimes want different models => specialization headroom)")
        except Exception:
            pass


def main():
    global REPLAY, OUT_CSV
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true', help='time ONE candidate then exit')
    ap.add_argument('--parse-only', action='store_true', help='re-print rankings from CSV')
    ap.add_argument('--replay', type=int, default=REPLAY, help='backtest replay window in hours (default 960)')
    args = ap.parse_args()

    REPLAY = args.replay
    OUT_CSV = f'output/regime_candidate_scan_ETH_{REPLAY}h.csv'

    if args.parse_only:
        _ranking_report(); return

    import crypto_trading_system_faye as F
    prod_feats = _load_prod_feats()
    regmap = _regime_series()
    prod_cfg = {5: ('RF+LGBM', 169, 0.9998), 8: ('XGB+LGBM', 163, 0.9985)}

    done = set()
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        done = {(int(r.horizon), r.combo, int(r.window), float(r.gamma)) for _, r in prev.iterrows()}
        print(f"  resume: {len(done)} candidates already scored")

    # build candidate list (combo x window per horizon; gamma fixed per horizon)
    cands = []
    for h in HORIZONS:
        for combo in COMBOS:
            for w in WINDOWS_BY_H[h]:
                cands.append((h, combo, w, GAMMA_BY_H[h]))
    if args.probe:
        cands = [(8, 'XGB+LGBM', 163, 0.9985)]  # the live 8h config
        print("  PROBE: scoring ONE candidate (live 8h) to measure wall time")

    print(f"  grid: {len(COMBOS)} combos x ~{len(WINDOWS_BY_H[HORIZONS[0]])} windows "
          f"x {len(HORIZONS)} horizons = {len(cands)} candidates (REPLAY={REPLAY}h, gamma fixed per horizon)")
    t_all = time.time()
    for k, (h, combo, w, g) in enumerate(cands, 1):
        if (h, combo, w, float(g)) in done:
            continue
        t0 = time.time()
        try:
            sigs = F.generate_signals(ASSET, combo.split('+'), w, REPLAY,
                                      feature_override=prod_feats[h], horizon=h, gamma=g)
        except Exception as e:
            print(f"  [{k}/{len(cands)}] {h}h {combo} w{w} g{g} -> ERROR {e}")
            continue
        bars = _parse_sigs(sigs, regmap)
        sc = _score(bars)
        is_prod = (combo, w) == (prod_cfg[h][0], prod_cfg[h][1] if h != 8 else 163) or \
                  (h == 5 and combo == 'RF+LGBM' and w == 169) or \
                  (h == 8 and combo == 'XGB+LGBM' and w == 163)
        row = dict(horizon=h, combo=combo, window=w, gamma=g, n_bars=len(bars),
                   is_prod=is_prod, **sc)
        hdr = not os.path.exists(OUT_CSV)
        pd.DataFrame([row]).to_csv(OUT_CSV, mode='a', header=hdr, index=False)
        dt = time.time() - t0
        print(f"  [{k}/{len(cands)}] {h}h {combo:<9} w{w:<3} g{g:<7} {len(bars)}bars {dt:5.1f}s | "
              f"ov={sc['ov_ret']:+6.2f}% bull={sc['bull_ret']:+6.2f}% bear={sc['bear_ret']:+6.2f}%", flush=True)
        if args.probe:
            n_full = sum(len(COMBOS) * len(WINDOWS_BY_H[hh]) for hh in HORIZONS)
            print(f"\n  PROBE done in {dt:.1f}s/candidate -> full grid ({n_full} cands) ~= {dt*n_full/60:.0f} min")
            return
    print(f"\n  ALL DONE in {(time.time()-t_all)/60:.1f} min")
    _ranking_report()


if __name__ == '__main__':
    main()
