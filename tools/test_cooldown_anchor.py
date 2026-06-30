"""2026-06-30 — Mode-T gate logic test: EXTEND (current/live) vs FIRE-ONCE cooldown anchoring.

The live + backtest gate both ANCHOR the cooldown to the LATEST hot bar (cd resets to cd_h on
EVERY bar with rr>=t), so a sustained rally holds the cooldown until cd_h AFTER it last looked
hot. FIRE-ONCE anchors to the FIRST trigger (rising edge) only — cd_h from the rally START,
unblocking even if still hot. This re-runs ONLY the Mode-T gate layer (faithful clone of
test_gate_simplicity.py's simulate) on the CURRENT live config, two ways:

  PART 1  CURRENT live gate (bull rr8>=2.5 cd6 / bear rr8>=2.5 cd8) under EXTEND vs FIRE-ONCE
          -> "is fire-once better with the current numbers?"  (EXTEND must ~= HRST +65.14%)
  PART 2  Re-sweep the production single-gate grid under EACH mode, 3-of-3 STRICT, and compare
          the best gate FIRE-ONCE finds to the best EXTEND finds -> "is it better overall?"

Prior art: C87 rally-cooldown REVERT-ABORT (2026-06-16) — a different cd-shortening — HURT.
Run: python tools/test_cooldown_anchor.py
"""
import os, sys, pickle
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
os.environ.setdefault('FAYE_CONFIG_DIR', 'config_faye_bt')
os.environ.pop('FAYE_EMBARGO_OVERRIDE', None)
HERE = r'c:\Users\Alex\algo_trading\engine'
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'tools')); os.chdir(HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from compare_prod_vs_4mo import _gen  # noqa: E402

CSV = 'models/crypto_ed_production.csv'          # the just-promoted live config
REPLAY = int(os.environ.get('CDA_REPLAY', '1440'))   # CDA_REPLAY=60 for a quick smoke test
FEE = 0.0011
BULL_H, BEAR_H = 6, 4                             # live: detector sma168>sma480
BULL_CONF, BEAR_CONF = 80.0, 65.0
BULL_SHIELD, BEAR_SHIELD = False, False
MIN_SELL, MAX_HOLD = 0.0, 10
LIVE_GATE = {'bull': dict(conds=[(8, 2.5)], cd=6), 'bear': dict(conds=[(8, 2.5)], cd=8)}
OFF = dict(conds=[], cd=0)
HORIZONS = [8, 10, 12, 14, 16, 18, 20, 24, 30, 36]


def thr_for(h):
    if h in (8, 10, 12):       return [round(2.0 + 0.5*i, 2) for i in range(9)]
    if h in (14, 16, 18, 20):  return [round(3.0 + 0.5*i, 2) for i in range(9)]
    if h in (24, 30, 36):      return [round(4.0 + 0.5*i, 2) for i in range(11)]
    raise ValueError(h)


CD_GRID = [6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 48]


def get_sig(h):
    cf = f'data/_cdanchor_sig_{h}h_{REPLAY}.pkl'
    if os.path.exists(cf):
        return pickle.load(open(cf, 'rb'))
    s = {str(pd.Timestamp(d)): v for d, v in _gen(CSV, h, REPLAY).items()}
    pickle.dump(s, open(cf, 'wb'))
    return s


s6, s4 = get_sig(BULL_H), get_sig(BEAR_H)

df = pd.read_csv('data/eth_hourly_data.csv'); df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').set_index('datetime')
sma_s = df['close'].rolling(168).mean(); sma_l = df['close'].rolling(480).mean()


def regime_at(dt):
    t = pd.to_datetime(dt)
    if t not in sma_s.index or sma_s.loc[t] != sma_s.loc[t] or sma_l.loc[t] != sma_l.loc[t]:
        return 'bull'
    return 'bull' if sma_s.loc[t] > sma_l.loc[t] else 'bear'


dts = sorted(set(s6) & set(s4))
stream = []
for dt in dts:
    reg = regime_at(dt)
    a = s6[dt] if reg == 'bull' else s4[dt]
    stream.append(dict(dt=dt, close=a['close'], sig=a['signal'],
                       conf=(a.get('confidence') or 0), regime=reg))
N = len(stream)


def build_rr(bars):
    p = np.array([b['close'] for b in bars], float)
    return {h: np.concatenate([np.full(h, np.nan), (p[h:]/p[:-h] - 1)*100]) for h in HORIZONS}


def simulate(bars, rr, gates, mode='extend'):
    cash = 1000.0; qty = 0.0; in_pos = False; entry = 0.0; hold = 0
    trades = 0; wins = 0; skipped = 0
    cd = {'bull': 0, 'bear': 0}; prev_hot = {'bull': False, 'bear': False}; ec = []
    for i, b in enumerate(bars):
        reg = b['regime']; g = gates[reg]; price = b['close']
        hot = False
        for (h, t) in g['conds']:
            v = rr[h][i]
            if v == v and v >= t:
                hot = True; break
        if hot:
            if mode == 'extend':
                cd[reg] = max(cd[reg], g['cd'])           # reset every hot bar (live behaviour)
            elif not prev_hot[reg]:
                cd[reg] = g['cd']                          # fire-once: only on the rising edge
        prev_hot[reg] = hot
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
        price = bars[-1]['close']; cash = qty*price*(1-FEE); trades += 1
        wins += 1 if (price/entry - 1) > 0 else 0; ec[-1] = cash
    ret = (cash/1000.0 - 1)*100
    arr = np.array(ec); peak = np.maximum.accumulate(arr); dd = (arr-peak)/peak
    return dict(ret=ret, trades=trades, wr=(100*wins/trades if trades else 0),
                skipped=skipped, dd=float(dd.min())*100 if len(dd) else 0)


mid = N // 2
WIN = {'REF': stream, 'H1': stream[mid:], 'H2': stream[:mid]}
RR = {k: build_rr(v) for k, v in WIN.items()}
print(f"window {stream[0]['dt']} -> {stream[-1]['dt']}  ({N} bars, "
      f"bull={sum(1 for s in stream if s['regime']=='bull')} bear={sum(1 for s in stream if s['regime']=='bear')})")

print("\n=== PART 1 — CURRENT live gate (bull rr8>=2.5 cd6 / bear rr8>=2.5 cd8) ===")
print(f"  {'mode':<11} {'REF':>9} {'H1':>9} {'H2':>9} {'WR':>5} {'trades':>6} {'blocked':>7}")
res = {}
for mode in ('extend', 'fire_once'):
    r = simulate(WIN['REF'], RR['REF'], LIVE_GATE, mode); res[mode] = r
    r1 = simulate(WIN['H1'], RR['H1'], LIVE_GATE, mode); r2 = simulate(WIN['H2'], RR['H2'], LIVE_GATE, mode)
    print(f"  {mode:<11} {r['ret']:>+8.2f}% {r1['ret']:>+8.2f}% {r2['ret']:>+8.2f}% "
          f"{r['wr']:>4.0f}% {r['trades']:>6} {r['skipped']:>7}")
print(f"  sanity: EXTEND REF should ~= HRST Mode-T baseline +65.14% / 86% WR / 94 tr")
print(f"  => Q2 (current numbers): fire_once - extend = {res['fire_once']['ret']-res['extend']['ret']:+.2f}pp REF")


def score(gates, mode):
    out = {k: simulate(WIN[k], RR[k], gates, mode) for k in ('REF', 'H1', 'H2')}
    out['score'] = out['H1']['ret'] - 0.5*abs(out['H1']['dd']); return out


print("\n=== PART 2 — best gate per regime: EXTEND sweep vs FIRE-ONCE sweep (3-of-3 STRICT) ===")
for reg in ('bull', 'bear'):
    other = 'bear' if reg == 'bull' else 'bull'
    og = LIVE_GATE[other]
    for mode in ('extend', 'fire_once'):
        base = score({reg: OFF, other: og}, mode)
        strict = []
        for h in HORIZONS:
            for t in thr_for(h):
                for cd in CD_GRID:
                    sc = score({reg: dict(conds=[(h, t)], cd=cd), other: og}, mode)
                    if (sc['REF']['ret'] > base['REF']['ret'] and sc['H1']['ret'] > base['H1']['ret']
                            and sc['H2']['ret'] > base['H2']['ret']):
                        strict.append((sc['score'], sc['REF']['ret'], h, t, cd,
                                       sc['REF']['trades'], sc['REF']['wr']))
        strict.sort(key=lambda r: -r[0])
        if strict:
            b = strict[0]
            print(f"  {reg:<4} {mode:<10} winner rr{b[2]}>={b[3]} cd{b[4]}  "
                  f"REF={b[1]:+.2f}% tr={b[5]} WR={b[6]:.0f}%  ({len(strict)} strict)")
        else:
            print(f"  {reg:<4} {mode:<10} NO STRICT WINNER (no gate beats no-gate)")

print("\nRead: Q1 (overall) = does fire_once's best gate beat extend's best per regime?")
print("      Q2 (current)  = PART 1 fire_once vs extend on the live gate. C87 prior: cd-shortening HURT.")
