"""2026-06-16: test a REVERT-ABORT for the rally-cooldown on the LIVE sma48>sma100
setup. Idea: the fixed cd (bull 24h / bear 14h) is blunt — if the rally that
triggered the cooldown reverts back to its origin price, cancel the cooldown early
(re-enable BUYs) instead of waiting out the full window.

Faithfully replicates faye._sweep_rally_cooldown.simulate (per-regime cd counters,
shield/min_sell/max_hold), on the cached 5h/8h signals + the live regime/gates.

  V0 = no gate        V1 = production fixed-cd gate (live)      V2 = gate + revert-abort
Run: python tools/test_cooldown_revert.py
"""
import pickle, numpy as np, pandas as pd

P5, P8 = 'data/_detbt_sig_5h.pkl', 'data/_detbt_sig_8h.pkl'
FEE = 0.0011
# live config (sma48>sma100)
BULL_CONF, BEAR_CONF = 90.0, 65.0
BULL_SHIELD, BEAR_SHIELD = True, False
MIN_SELL, MAX_HOLD = 0.0, 10
GATE = {'bull': dict(h_s=30, h_l=36, t_s=4.0, t_l=6.5, cd=24),
        'bear': dict(h_s=8,  h_l=30, t_s=2.5, t_l=7.0, cd=14)}

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
n = len(stream)
price = np.array([s['close'] for s in stream])
# rr[h][i] = pct return over last h hours on the merged price series
rr = {h: np.concatenate([np.full(h, np.nan), (price[h:]/price[:-h]-1)*100]) for h in (8, 30, 36)}

def simulate(revert_abort):
    cash = 1000.0; qty = 0.0; in_pos = False; entry = 0.0; hold = 0
    trades = 0; wins = 0; skipped = 0
    cd = {'bull': 0, 'bear': 0}; origin = {'bull': np.nan, 'bear': np.nan}
    for i in range(n):
        s = stream[i]; p = price[i]; reg = s['regime']; g = GATE[reg]
        # trigger on this regime's bars
        rs, rl = rr[g['h_s']][i], rr[g['h_l']][i]
        if (rs == rs and rs >= g['t_s']) or (rl == rl and rl >= g['t_l']):
            cd[reg] = max(cd[reg], g['cd'])
            j = i - g['h_s']
            origin[reg] = price[j] if j >= 0 else np.nan   # rally start price
        # revert-abort: cancel cooldown if price fell back to the rally origin
        if revert_abort and cd[reg] > 0 and origin[reg] == origin[reg] and p <= origin[reg]:
            cd[reg] = 0
        # BUY (blocked by the active regime's cd)
        conf_thr = BULL_CONF if reg == 'bull' else BEAR_CONF
        if s['sig'] == 'BUY' and s['conf'] >= conf_thr and not in_pos:
            if cd[reg] > 0:
                skipped += 1
            else:
                qty = cash*(1-FEE)/p; cash = 0.0; in_pos = True; entry = p; hold = 0
        elif s['sig'] == 'SELL' and in_pos:
            cur = (p/entry-1)*100
            shield_on = BULL_SHIELD if reg == 'bull' else BEAR_SHIELD
            shield_min = MIN_SELL if shield_on else 0.0
            if cur >= shield_min or hold >= MAX_HOLD:
                cash = qty*p*(1-FEE); trades += 1; wins += 1 if cur > 0 else 0
                in_pos = False; qty = 0.0; entry = 0.0; hold = 0
        if in_pos: hold += 1
        if cd['bull'] > 0: cd['bull'] -= 1
        if cd['bear'] > 0: cd['bear'] -= 1
    if in_pos:
        p = price[-1]; cash = qty*p*(1-FEE); trades += 1; wins += 1 if (p/entry-1) > 0 else 0
    ret = (cash/1000.0-1)*100
    return ret, trades, (100*wins/trades if trades else 0), skipped

def simulate_nogate():
    cash = 1000.0; qty = 0.0; in_pos = False; entry = 0.0; hold = 0; trades = 0; wins = 0
    for i in range(n):
        s = stream[i]; p = price[i]; reg = s['regime']
        conf_thr = BULL_CONF if reg == 'bull' else BEAR_CONF
        if s['sig'] == 'BUY' and s['conf'] >= conf_thr and not in_pos:
            qty = cash*(1-FEE)/p; cash = 0.0; in_pos = True; entry = p; hold = 0
        elif s['sig'] == 'SELL' and in_pos:
            cur = (p/entry-1)*100
            shield_on = BULL_SHIELD if reg == 'bull' else BEAR_SHIELD
            shield_min = MIN_SELL if shield_on else 0.0
            if cur >= shield_min or hold >= MAX_HOLD:
                cash = qty*p*(1-FEE); trades += 1; wins += 1 if cur > 0 else 0
                in_pos = False; qty = 0.0; entry = 0.0; hold = 0
        if in_pos: hold += 1
    if in_pos:
        p = price[-1]; cash = qty*p*(1-FEE); trades += 1; wins += 1 if (p/entry-1) > 0 else 0
    return (cash/1000.0-1)*100, trades, (100*wins/trades if trades else 0), 0

print(f"  window: {stream[0]['dt']} -> {stream[-1]['dt']}  ({n} bars, bull={sum(1 for s in stream if s['regime']=='bull')} bear={sum(1 for s in stream if s['regime']=='bear')})")
print(f"\n  {'variant':<34} {'return':>8} {'trades':>6} {'WR':>5} {'BUYs blocked':>13}")
print("  " + "-"*72)
for label, fn in [('V0  no gate', lambda: simulate_nogate()),
                  ('V1  production fixed-cd (LIVE)', lambda: simulate(False)),
                  ('V2  gate + revert-abort', lambda: simulate(True))]:
    r, t, w, sk = fn()
    print(f"  {label:<34} {r:>+7.2f}% {t:>6} {w:>4.0f}% {sk:>13}")
print("\n  (live sma48>sma100: bull 8h@90 cd24[30/36,4.0/6.5], bear 5h@65 cd14[8/30,2.5/7.0], min_sell=0, max_hold=10)")
