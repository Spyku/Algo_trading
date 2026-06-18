"""2026-06-16 (Idea 1): is the DOUBLE-condition rally-cooldown gate worth its
complexity, or would a single condition do just as well?

The production gate (`faye._sweep_rally_cooldown`) is STRUCTURALLY a 2-window
double-condition per regime: it fires when `rr[h_s] >= t_s OR rr[h_l] >= t_l`,
and the sweep only ever enumerates horizon PAIRS — a single-window gate is never
even considered. Live (sma48>sma100):
    bull  rr30 >= 4.0  OR  rr36 >= 6.5   cd=24h
    bear  rr8  >= 2.5  OR  rr30 >= 7.0   cd=14h

This study (faithful clone of tools/test_cooldown_revert.py's simulator, same
cached 5h/8h signals + live regime/policy) answers it two ways:

  PART A  Leg-ablation — drop one leg of the live double at a time (per regime)
          and see if the result moves. If short-only == both, the long leg is
          dead weight (and vice versa).
  PART B  Best single-condition sweep — for each regime sweep a SINGLE
          (window, threshold, cd) over the same grid the production sweep uses,
          hold the OTHER regime at its live double, score on H1/H2/REF the same
          way production does, and compare the best simple gate to the live
          double. If a single gate matches the double's score, the second leg
          (and the whole 2-window machinery) isn't earning its keep.

Run: python tools/test_gate_simplicity.py
"""
import pickle, numpy as np, pandas as pd

P5, P8 = 'data/_detbt_sig_5h.pkl', 'data/_detbt_sig_8h.pkl'
FEE = 0.0011
# live config (sma48>sma100) — verified against config/regime_config_ed.json 2026-06-16
BULL_CONF, BEAR_CONF = 90.0, 65.0
BULL_SHIELD, BEAR_SHIELD = True, False
MIN_SELL, MAX_HOLD = 0.0, 10
LIVE = {'bull': dict(conds=[(30, 4.0), (36, 6.5)], cd=24),
        'bear': dict(conds=[(8, 2.5), (30, 7.0)], cd=14)}
OFF = dict(conds=[], cd=0)

# production sweep grid (faye.py:7797-7803) — mirrored exactly
HORIZONS = [8, 10, 12, 14, 16, 18, 20, 24, 30, 36]
def thr_for(h):
    if h in (8, 10, 12):       return [round(2.0 + 0.5*i, 2) for i in range(9)]
    if h in (14, 16, 18, 20):  return [round(3.0 + 0.5*i, 2) for i in range(9)]
    if h in (24, 30, 36):      return [round(4.0 + 0.5*i, 2) for i in range(11)]
    raise ValueError(h)
CD_GRID = [6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 48]

# ---- merged stream: regime = sma48>sma100, active sig = 8h(bull)/5h(bear) ----
s5 = {x['dt']: x for x in pickle.load(open(P5, 'rb'))}
s8 = {x['dt']: x for x in pickle.load(open(P8, 'rb'))}
dts = sorted(set(s5) & set(s8))
df = pd.read_csv('data/ETH_hourly_data.csv'); df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').set_index('datetime')
sma48 = df['close'].rolling(48).mean(); sma100 = df['close'].rolling(100).mean()
def regime_at(dt):
    t = pd.to_datetime(dt)
    if t not in sma48.index or sma48.loc[t] != sma48.loc[t] or sma100.loc[t] != sma100.loc[t]:
        return 'bull'   # cold-start default (trader convention)
    return 'bull' if sma48.loc[t] > sma100.loc[t] else 'bear'

stream = []
for dt in dts:
    reg = regime_at(dt)
    a = s8[dt] if reg == 'bull' else s5[dt]
    stream.append(dict(dt=dt, close=a['close'], sig=a['sig'], conf=a['conf'], regime=reg))
N = len(stream)


def build_rr(bars):
    p = np.array([b['close'] for b in bars], dtype=float)
    return {h: np.concatenate([np.full(h, np.nan), (p[h:]/p[:-h] - 1)*100]) for h in HORIZONS}


def simulate(bars, rr, gates):
    """gates = {'bull':dict(conds=[(h,t),...],cd), 'bear':...}. A regime's cd
    fires when ANY of its conds is met (rr[h]>=t) — empty conds = never fires."""
    cash = 1000.0; qty = 0.0; in_pos = False; entry = 0.0; hold = 0
    trades = 0; wins = 0; skipped = 0
    cd = {'bull': 0, 'bear': 0}
    ec = []
    for i, b in enumerate(bars):
        reg = b['regime']; g = gates[reg]; price = b['close']
        for (h, t) in g['conds']:
            v = rr[h][i]
            if v == v and v >= t:
                cd[reg] = max(cd[reg], g['cd']); break
        conf_thr = BULL_CONF if reg == 'bull' else BEAR_CONF
        if b['sig'] == 'BUY' and b['conf'] >= conf_thr and not in_pos:
            if cd[reg] > 0:
                skipped += 1
            else:
                qty = cash*(1-FEE)/price; cash = 0.0; in_pos = True; entry = price; hold = 0
        elif b['sig'] == 'SELL' and in_pos:
            cur = (price/entry - 1)*100
            shield_on = BULL_SHIELD if reg == 'bull' else BEAR_SHIELD
            shield_min = MIN_SELL if shield_on else 0.0
            if cur >= shield_min or hold >= MAX_HOLD:
                cash = qty*price*(1-FEE); trades += 1; wins += 1 if cur > 0 else 0
                in_pos = False; qty = 0.0; entry = 0.0; hold = 0
        if in_pos: hold += 1
        if cd['bull'] > 0: cd['bull'] -= 1
        if cd['bear'] > 0: cd['bear'] -= 1
        ec.append(cash + (qty*price if in_pos else 0.0))
    if in_pos:
        price = bars[-1]['close']; cash = qty*price*(1-FEE); trades += 1; wins += 1 if (price/entry-1) > 0 else 0
        ec[-1] = cash
    ret = (cash/1000.0 - 1)*100
    arr = np.array(ec); peak = np.maximum.accumulate(arr); dd = (arr-peak)/peak
    mdd = float(dd.min())*100 if len(dd) else 0.0
    return dict(ret=ret, trades=trades, wr=(100*wins/trades if trades else 0),
                skipped=skipped, dd=mdd)


# windows: REF = all bars; H1 = recent half; H2 = earlier half (production convention)
mid = N // 2
WIN = {'REF': stream, 'H1': stream[mid:], 'H2': stream[:mid]}
RR = {k: build_rr(v) for k, v in WIN.items()}

print(f"  window: {stream[0]['dt']} -> {stream[-1]['dt']}  ({N} bars, "
      f"bull={sum(1 for s in stream if s['regime']=='bull')} bear={sum(1 for s in stream if s['regime']=='bear')})")
print(f"  H1 (recent half) = {len(WIN['H1'])} bars | H2 (earlier half) = {len(WIN['H2'])} bars\n")

# ===================== PART A — leg ablation =====================
print("=" * 78)
print("  PART A - leg ablation (REF = full window). Other regime held at LIVE double.")
print("=" * 78)
print(f"  {'variant':<40} {'return':>8} {'trades':>6} {'WR':>5} {'maxDD':>7} {'blocked':>8}")
print("  " + "-"*78)

def leg(reg, mode):
    """Return a one-regime gate spec: 'both'/'short'/'long'/'off'."""
    s, l = LIVE[reg]['conds']; cd = LIVE[reg]['cd']
    return {'both': dict(conds=[s, l], cd=cd), 'short': dict(conds=[s], cd=cd),
            'long': dict(conds=[l], cd=cd), 'off': OFF}[mode]

variants_A = [
    ('V0  no gate (both off)',           {'bull': OFF, 'bear': OFF}),
    ('LIVE  bull=both  bear=both',       {'bull': leg('bull','both'),  'bear': leg('bear','both')}),
    ('bull SHORT-only (rr30>=4.0)',      {'bull': leg('bull','short'), 'bear': leg('bear','both')}),
    ('bull LONG-only  (rr36>=6.5)',      {'bull': leg('bull','long'),  'bear': leg('bear','both')}),
    ('bull OFF',                         {'bull': leg('bull','off'),   'bear': leg('bear','both')}),
    ('bear SHORT-only (rr8>=2.5)',       {'bull': leg('bull','both'),  'bear': leg('bear','short')}),
    ('bear LONG-only  (rr30>=7.0)',      {'bull': leg('bull','both'),  'bear': leg('bear','long')}),
    ('bear OFF',                         {'bull': leg('bull','both'),  'bear': leg('bear','off')}),
]
for label, gates in variants_A:
    r = simulate(WIN['REF'], RR['REF'], gates)
    print(f"  {label:<40} {r['ret']:>+7.2f}% {r['trades']:>6} {r['wr']:>4.0f}% "
          f"{r['dd']:>+6.2f}% {r['skipped']:>8}")

# ===================== PART B — best single-condition sweep =====================
print("\n" + "=" * 78)
print("  PART B - best SINGLE-condition gate per regime vs the LIVE double.")
print("  (sweep 1 window x 1 thr x cd; other regime held LIVE; rank = score_recent")
print("   = ret_H1 - 0.5*|dd_H1|; keep only configs that beat 'this-regime-off' on")
print("   H1 AND H2 AND REF, i.e. 3-of-3, the production STRICT filter.)")
print("=" * 78)

def score_config(gates):
    out = {}
    for k in ('REF', 'H1', 'H2'):
        out[k] = simulate(WIN[k], RR[k], gates)
    out['score_recent'] = out['H1']['ret'] - 0.5*abs(out['H1']['dd'])
    return out

for reg in ('bull', 'bear'):
    other = 'bear' if reg == 'bull' else 'bull'
    other_gate = leg(other, 'both')
    # baseline = this regime OFF, other LIVE
    base = score_config({reg: OFF, other: other_gate})
    live = score_config({reg: leg(reg, 'both'), other: other_gate})

    rows = []
    for h in HORIZONS:
        for t in thr_for(h):
            for cd in CD_GRID:
                g = {reg: dict(conds=[(h, t)], cd=cd), other: other_gate}
                sc = score_config(g)
                b3 = (sc['REF']['ret'] > base['REF']['ret'] and
                      sc['H1']['ret']  > base['H1']['ret'] and
                      sc['H2']['ret']  > base['H2']['ret'])
                rows.append(dict(h=h, t=t, cd=cd, beats3=b3,
                                 ref=sc['REF']['ret'], h1=sc['H1']['ret'],
                                 h2=sc['H2']['ret'], score=sc['score_recent'],
                                 tr=sc['REF']['trades'], wr=sc['REF']['wr'],
                                 dd=sc['REF']['dd'], sk=sc['REF']['skipped']))
    rdf = pd.DataFrame(rows)
    strict = rdf[rdf['beats3']].sort_values('score', ascending=False)

    print(f"\n  --- {reg.upper()} (other={other} held LIVE) ---")
    print(f"  baseline ({reg} OFF):        REF={base['REF']['ret']:+.2f}%  "
          f"H1={base['H1']['ret']:+.2f}%  H2={base['H2']['ret']:+.2f}%")
    lf = LIVE[reg]
    print(f"  LIVE double rr{lf['conds'][0][0]}>={lf['conds'][0][1]} OR "
          f"rr{lf['conds'][1][0]}>={lf['conds'][1][1]} cd{lf['cd']}:  "
          f"REF={live['REF']['ret']:+.2f}%  H1={live['H1']['ret']:+.2f}%  "
          f"H2={live['H2']['ret']:+.2f}%  score={live['score_recent']:+.2f}  "
          f"(beats3={'Y' if (live['REF']['ret']>base['REF']['ret'] and live['H1']['ret']>base['H1']['ret'] and live['H2']['ret']>base['H2']['ret']) else 'N'})")
    print(f"  single-condition configs passing 3-of-3: {len(strict)} / {len(rdf)}")
    if len(strict):
        print(f"  {'rank':>4}  {'single gate':<22} {'REF':>8} {'H1':>8} {'H2':>8} {'score':>7} {'tr':>4} {'WR':>4} {'blk':>4}")
        for i, (_, r) in enumerate(strict.head(8).iterrows(), 1):
            g = f"rr{int(r['h'])}>={r['t']} cd{int(r['cd'])}"
            print(f"  {i:>4}  {g:<22} {r['ref']:>+7.2f}% {r['h1']:>+7.2f}% "
                  f"{r['h2']:>+7.2f}% {r['score']:>+6.2f} {int(r['tr']):>4} "
                  f"{r['wr']:>3.0f}% {int(r['sk']):>4}")
    else:
        print("  (no single-condition gate beats baseline on all 3 windows)")

print("\n  Read: if a single-condition gate matches the LIVE double's REF/score,")
print("  the 2nd leg (and the 2-window sweep machinery) isn't earning its keep.")
