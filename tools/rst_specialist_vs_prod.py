"""2026-06-16 (Idea 2): head-to-head regime-switch backtest — SPECIALIST (longer
per-regime training windows, from the regime_candidate_scan) vs PRODUCTION, over
1 week / 1 month / 2 months. Strategy (detector + per-regime conf/shield/gate) is
held to the CURRENT LIVE config for BOTH arms, so only the MODELS differ -> the
delta isolates the window effect (not a re-optimized detector/gate).

  PROD : bull 8h XGB+LGBM w163 g0.9985 | bear 5h RF+LGBM w169 g0.9998   (live models)
  SPEC : bull 8h XGB+LGBM w250 g0.9985 | bear 5h RF+LGBM w300 g0.9998   (scan winners)
  (features fixed to the live prod set per horizon for both arms; only window differs)

Caches each config's signal stream to pkl, so the (instant) regime-switch sims can be
re-run under a different gate config without regenerating signals.

  python tools/rst_specialist_vs_prod.py            # generate (if needed) + compare
  python tools/rst_specialist_vs_prod.py --report    # re-run sims from cached pkls only
"""
import os, sys, time, argparse, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
os.environ.setdefault('FAYE_MODELS_DIR', 'models')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

ASSET = 'ETH'
REPLAY = 1440
FEE = 0.0011
WINDOWS = {'1wk': 168, '1mo': 720, '2mo': 1440}
CONFIGS = {
    'prod_5h': dict(h=5, combo='RF+LGBM',  w=169, g=0.9998),
    'prod_8h': dict(h=8, combo='XGB+LGBM', w=163, g=0.9985),
    'spec_5h': dict(h=5, combo='RF+LGBM',  w=300, g=0.9998),
    'spec_8h': dict(h=8, combo='XGB+LGBM', w=250, g=0.9985),
}
PKL = 'data/_rst_cmp_{name}.pkl'


def _prod_feats():
    df = pd.read_csv('models/crypto_ed_production.csv')
    out = {}
    for h in (5, 8):
        r = df[(df.coin == ASSET) & (df.horizon == h)].sort_values('combined_score', ascending=False).iloc[0]
        out[h] = r['optimal_features'].split(',')
    return out


def _regmap():
    df = pd.read_csv('data/ETH_hourly_data.csv'); df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').set_index('datetime')
    s48 = df['close'].rolling(48).mean(); s100 = df['close'].rolling(100).mean()
    return {t: ('bull' if (s48.loc[t] != s48.loc[t] or s100.loc[t] != s100.loc[t] or s48.loc[t] > s100.loc[t]) else 'bear')
            for t in df.index}


def _gen_all():
    import crypto_trading_system_faye as F
    feats = _prod_feats()
    for name, c in CONFIGS.items():
        p = PKL.format(name=name)
        if os.path.exists(p):
            print(f"  {name}: cached, skip"); continue
        t0 = time.time()
        sigs = F.generate_signals(ASSET, c['combo'].split('+'), c['w'], REPLAY,
                                  feature_override=feats[c['h']], horizon=c['h'], gamma=c['g'])
        out = []
        for s in sigs:
            dt = pd.to_datetime(s['datetime'])
            out.append(dict(dt=dt, close=float(s['close']), sig=s['signal'], conf=float(s['confidence'])))
        pickle.dump(out, open(p, 'wb'))
        print(f"  {name}: {len(out)} sigs in {time.time()-t0:.0f}s -> {p}", flush=True)


def _load(name):
    return {x['dt']: x for x in pickle.load(open(PKL.format(name=name), 'rb'))}


def _live_gate():
    e = json.load(open('config/regime_config_ed.json'))['ETH']
    g = {}
    for r in ('bull', 'bear'):
        rc = e[r].get('rally_cooldown') or {}
        g[r] = dict(on=bool(rc.get('enabled')), hs=int(rc.get('h_short', 8)), hl=int(rc.get('h_long', 36)),
                    ts=float(rc.get('t_short_pct', 9999)), tl=float(rc.get('t_long_pct', 9999)),
                    cd=int(rc.get('cd_hours', 0)),
                    conf=float(e[r]['min_confidence']), shield=bool(e[r].get('hold_shield')))
    g['min_sell'] = float(e.get('min_sell_pnl_pct', 0)); g['max_hold'] = int(e.get('max_hold_hours', 10))
    return g


def _sim(bull_sigs, bear_sigs, regmap, gate, dts):
    """Regime-switch over `dts`: bull bars use bull_sigs (8h), bear bars use bear_sigs (5h).
    Per-regime conf/shield + per-regime rally-cd gate. Fresh start (flat, cd=0)."""
    price = np.array([(bear_sigs if regmap.get(dt, 'bull') == 'bear' else bull_sigs)[dt]['close'] for dt in dts])
    rr = {h: np.concatenate([np.full(h, np.nan), (price[h:] / price[:-h] - 1) * 100]) for h in (8, 30, 36)}
    cash = 1000.0; qty = 0.0; inpos = False; entry = 0.0; hold = 0; trades = 0; wins = 0; blocked = 0
    cd = {'bull': 0, 'bear': 0}
    for i, dt in enumerate(dts):
        reg = regmap.get(dt, 'bull'); g = gate[reg]
        a = (bear_sigs if reg == 'bear' else bull_sigs)[dt]; p = price[i]
        if g['on']:
            rs, rl = rr.get(g['hs'], rr[8])[i], rr.get(g['hl'], rr[36])[i]
            if (rs == rs and rs >= g['ts']) or (rl == rl and rl >= g['tl']):
                cd[reg] = max(cd[reg], g['cd'])
        if a['sig'] == 'BUY' and a['conf'] >= g['conf'] and not inpos:
            if cd[reg] > 0:
                blocked += 1
            else:
                qty = cash * (1 - FEE) / p; cash = 0.0; inpos = True; entry = p; hold = 0
        elif a['sig'] == 'SELL' and inpos:
            cur = (p / entry - 1) * 100
            if cur >= (gate['min_sell'] if g['shield'] else 0.0) or hold >= gate['max_hold']:
                cash = qty * p * (1 - FEE); trades += 1; wins += 1 if cur > 0 else 0
                inpos = False; qty = 0.0; entry = 0.0; hold = 0
        if inpos: hold += 1
        if cd['bull'] > 0: cd['bull'] -= 1
        if cd['bear'] > 0: cd['bear'] -= 1
    if inpos:
        p = price[-1]; cash = qty * p * (1 - FEE); trades += 1; wins += 1 if (p / entry - 1) > 0 else 0
    return dict(ret=(cash / 1000 - 1) * 100, tr=trades, wr=(100 * wins / trades if trades else 0), blk=blocked)


def _bh(sigs, dts):
    p0 = sigs[dts[0]]['close']; p1 = sigs[dts[-1]]['close']
    return (p1 / p0 - 1) * 100


def _report():
    regmap = _regmap()
    gate = _live_gate()
    P8, P5 = _load('prod_8h'), _load('prod_5h')
    S8, S5 = _load('spec_8h'), _load('spec_5h')
    dts_all = sorted(set(P8) & set(P5) & set(S8) & set(S5))
    print(f"\n  full span: {dts_all[0]} -> {dts_all[-1]} ({len(dts_all)} bars)")
    print(f"  gate (live, both arms): bull on={gate['bull']['on']} (rr{gate['bull']['hs']}>={gate['bull']['ts']} OR "
          f"rr{gate['bull']['hl']}>={gate['bull']['tl']} cd{gate['bull']['cd']}) @{gate['bull']['conf']:.0f} shield={gate['bull']['shield']} | "
          f"bear on={gate['bear']['on']} (rr{gate['bear']['hs']}>={gate['bear']['ts']} OR rr{gate['bear']['hl']}>={gate['bear']['tl']} "
          f"cd{gate['bear']['cd']}) @{gate['bear']['conf']:.0f} shield={gate['bear']['shield']} | min_sell={gate['min_sell']} max_hold={gate['max_hold']}")
    print(f"\n  PROD  = bull 8h w163 / bear 5h w169   (production)")
    print(f"  MIXED = bull 8h w163 / bear 5h w300   (SURGICAL: only the bear window changed = the real finding)")
    print(f"  SPEC  = bull 8h w250 / bear 5h w300   (960h-overfit specialist; bull w250 was the overfit leg)\n")
    print(f"  {'window':<7} {'B&H':>7} | {'PROD':>8} {'tr':>4} | {'MIXED':>8} {'tr':>4} {'vsPROD':>7} | {'SPEC':>8} {'vsPROD':>7}")
    print("  " + "-" * 80)
    for wlab, wbars in WINDOWS.items():
        dts = dts_all[-wbars:]
        bh = _bh(P8, dts)
        rp = _sim(P8, P5, regmap, gate, dts)
        rm = _sim(P8, S5, regmap, gate, dts)   # MIXED: prod bull (w163) + spec bear (w300) — the clean test
        rs = _sim(S8, S5, regmap, gate, dts)
        print(f"  {wlab:<7} {bh:>+6.2f}% | {rp['ret']:>+7.2f}% {rp['tr']:>4} | "
              f"{rm['ret']:>+7.2f}% {rm['tr']:>4} {rm['ret']-rp['ret']:>+6.2f} | "
              f"{rs['ret']:>+7.2f}% {rs['ret']-rp['ret']:>+6.2f}")
    print("\n  vsPROD = arm - PROD (pp). MIXED isolates the 5h bear-window change w169->w300 (the clean test).")
    print("  SPEC's overfit bull (w250) is what sank it on 2mo earlier; MIXED keeps prod's bull.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', action='store_true', help='re-run sims from cached pkls only (no generation)')
    args = ap.parse_args()
    if not args.report:
        _gen_all()
    _report()


if __name__ == '__main__':
    main()
