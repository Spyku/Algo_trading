"""2026-06-15: fix the 8h/6h horizon pair, sweep EVERY regime detector over a
1440h window THROUGH TODAY, to see which switcher would have been best (incl. the
06-14/15 rally). Mode-S semantics (pre-rally-cooldown), embargo=horizon (matches
the production Mode S sweep). Live configs read from models/crypto_ed_production.csv.

  --gen 8     generate one horizon's walk-forward signals -> pickle  (run 2 in parallel)
  --gen 6
  --sweep     read both pickles, build all detectors, sweep conf pairs, rank
"""
import os, sys, argparse, pickle, csv
os.environ['FAYE_LIBRARY_MODE']='1'; os.environ['_FAYE_WARNINGS_BAKED']='1'
# embargo left at default (=horizon) to match the production Mode S methodology.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

REPLAY = 1440
PKL = lambda h: f"data/_detbt_sig_{h}h.pkl"
CONF_GRID = [65,70,75,80,85,90]
FEE = 0.0011  # per leg (TRADING_FEE)
MAX_HOLD = 10

def load_cfg(h):
    with open('models/crypto_ed_production.csv') as f:
        for r in csv.DictReader(f):
            if r['coin']=='ETH' and int(r['horizon'])==h:
                return (r['best_combo'].split('+'), int(r['best_window']),
                        float(r['gamma']), r['optimal_features'].split(','))
    raise SystemExit(f"no ETH {h}h row")

def gen(h):
    import crypto_trading_system_faye as F
    models, win, gamma, feats = load_cfg(h)
    sigs = F.generate_signals('ETH', models, win, REPLAY, feature_override=feats, horizon=h, gamma=gamma)
    out = [{'dt':s['datetime'],'close':s['close'],'sig':s['signal'],'conf':s['confidence']} for s in sigs]
    with open(PKL(h),'wb') as f: pickle.dump(out, f)
    print(f"  {h}h: wrote {len(out)} signals -> {PKL(h)}")

def build_indicators():
    df = pd.read_csv('data/ETH_hourly_data.csv')
    df['datetime']=pd.to_datetime(df['datetime']); df=df.sort_values('datetime').set_index('datetime')
    c=df['close']
    for w in (6,12,24,48,72,100,168,200,480): df[f'sma{w}']=c.rolling(w).mean()
    for s in (9,12,21,26): df[f'ema{s}']=c.ewm(span=s).mean()
    df['tsmom168']=np.log(c/c.shift(168)); df['tsmom72']=np.log(c/c.shift(72))
    # rsi14
    d=c.diff(); up=d.clip(lower=0).rolling(14).mean(); dn=(-d.clip(upper=0)).rolling(14).mean()
    df['rsi14']=100-100/(1+up/dn.replace(0,np.nan))
    df['dd48']=(c/c.rolling(48).max()-1)*100; df['dd72']=(c/c.rolling(72).max()-1)*100
    df['bounce48']=(c/c.rolling(48).min()-1)*100
    df['macd']=c.ewm(span=12).mean()-c.ewm(span=26).mean()
    # vol_calm (engine formula)
    lr=np.log(c/c.shift(1)); al=lr.abs(); hour=df.index.hour
    sf=al.groupby(hour).transform(lambda s:s.rolling(30,min_periods=10).mean())
    ald=al/sf.replace(0,np.nan); v=ald.rolling(24).std()
    df['vol_calm_flag']=v < v.rolling(720,min_periods=240).quantile(0.70)
    df['tsmom672']=np.log(c/c.shift(672))
    return df

def detectors(df):
    D={}
    D['sma24>sma100']=df['sma24']>df['sma100']
    D['sma168>sma480']=df['sma168']>df['sma480']
    D['price>sma72']=df['close']>df['sma72']
    D['vol_calm']=df['vol_calm_flag']
    D['tsmom_672h']=df['tsmom672']>0
    # FASTER-than-sma24>sma100 candidates (2026-06-15 user request)
    D['sma12>sma48']=df['sma12']>df['sma48']
    D['sma12>sma24']=df['sma12']>df['sma24']
    D['sma6>sma24']=df['sma6']>df['sma24']
    D['price>sma24']=df['close']>df['sma24']
    D['ema9>ema21']=df['ema9']>df['ema21']
    D['ema12>ema26']=df['ema12']>df['ema26']
    D['tsmom_168h']=df['tsmom168']>0
    D['tsmom_72h']=df['tsmom72']>0
    # legacy menu
    D['sma24>sma72']=df['sma24']>df['sma72']
    D['sma48>sma100']=df['sma48']>df['sma100']
    D['sma48>sma200']=df['sma48']>df['sma200']
    D['price>sma48']=df['close']>df['sma48']
    D['price>sma100']=df['close']>df['sma100']
    D['rsi>55']=df['rsi14']>55; D['rsi>50']=df['rsi14']>50; D['rsi>45']=df['rsi14']>45
    D['dd48>-2%']=df['dd48']>-2; D['dd48>-3%']=df['dd48']>-3; D['dd48>-5%']=df['dd48']>-5
    D['dd72>-3%']=df['dd72']>-3; D['dd72>-5%']=df['dd72']>-5
    D['bounce48>2%']=df['bounce48']>2; D['bounce48>3%']=df['bounce48']>3
    D['macd>0']=df['macd']>0
    D['sma24>100+dd48>-3']=(df['sma24']>df['sma100'])&(df['dd48']>-3)
    D['sma24>100+rsi>45']=(df['sma24']>df['sma100'])&(df['rsi14']>45)
    D['dd48>-3%+rsi>45']=(df['dd48']>-3)&(df['rsi14']>45)
    D['price>sma72+rsi>50']=(df['close']>df['sma72'])&(df['rsi14']>50)
    return D

def simulate(dts, s8, s6, is_bull, bullC, bearC):
    pos='cash'; entry=0; ei=0; eq=1.0; trades=0; wins=0; bull_hours=0
    for i,dt in enumerate(dts):
        bull = is_bull.get(dt, True); bull_hours += 1 if bull else 0
        act = s8.get(dt) if bull else s6.get(dt)
        if act is None: continue
        gate = bullC if bull else bearC
        if pos=='cash':
            if act['sig']=='BUY' and act['conf']>=gate:
                pos='long'; entry=act['close']; ei=i
        else:
            held=i-ei
            if act['sig']=='SELL' or held>=MAX_HOLD:
                pnl=(act['close']/entry-1)-2*FEE; eq*=(1+pnl); trades+=1; wins+=1 if pnl>0 else 0; pos='cash'
    if pos=='long':
        last=s8.get(dts[-1]) or s6.get(dts[-1])
        pnl=(last['close']/entry-1)-2*FEE; eq*=(1+pnl); trades+=1; wins+=1 if pnl>0 else 0
    ret=(eq-1)*100; wr=100*wins/trades if trades else 0
    return ret, trades, wr, 100*bull_hours/max(len(dts),1)

def sweep():
    s8={x['dt']:x for x in pickle.load(open(PKL(8),'rb'))}
    s6={x['dt']:x for x in pickle.load(open(PKL(6),'rb'))}
    dts=sorted(set(s8)&set(s6))
    print(f"\n  Window: {dts[0]} -> {dts[-1]}  ({len(dts)} hours)")
    df=build_indicators(); D=detectors(df)
    # detector flags keyed by the same dt strings
    dtidx={d.strftime('%Y-%m-%d %H:%M'):d for d in df.index}
    def flagmap(series):
        m={}
        for ds in dts:
            ts=pd.to_datetime(ds)
            m[ds]= bool(series.get(ts, True)) if ts in series.index else True
        return m
    # baselines
    print("\n  --- single-horizon baselines (no switching) ---")
    for h,sig in (('8h_only',s8),('6h_only',s6)):
        for C in [65,70,75,80,85,90]:
            pass
    for name,sig in (('8h_only',s8),('6h_only',s6)):
        best=None
        for C in CONF_GRID:
            r,t,w,_=simulate(dts, sig, sig, {d:True for d in dts}, C, C)
            sc=r*w/100 if r>0 else r
            if best is None or sc>best[0]: best=(sc,r,t,w,C)
        print(f"    {name:14s} best@{best[4]}%: {best[1]:+.2f}%  tr={best[2]} wr={best[3]:.0f}%")
    # detector sweep
    rows=[]
    for name,series in D.items():
        fm=flagmap(series)
        best=None
        for bC in CONF_GRID:
            for rC in CONF_GRID:
                r,t,w,bp=simulate(dts, s8, s6, fm, bC, rC)
                if t<5: continue
                sc=r*w/100 if r>0 else r
                if best is None or sc>best[0]: best=(sc,r,t,w,bp,bC,rC)
        if best: rows.append((name,)+best)
    rows.sort(key=lambda x:-x[1])  # by score
    print(f"\n  {'='*92}")
    print(f"  8h/6h PAIR — ALL DETECTORS (best conf-pair each), ranked by score   [* = engine-native/promotable]")
    print(f"  {'='*92}")
    print(f"  {'detector':<22} {'bullC':>5} {'bearC':>5} {'return':>9} {'trades':>6} {'WR':>5} {'bull%':>6}")
    native={'sma24>sma100','sma168>sma480','price>sma72','vol_calm','tsmom_672h'}
    for name,sc,r,t,w,bp,bC,rC in rows:
        star='*' if name in native else ' '
        print(f" {star}{name:<22} {bC:>5} {rC:>5} {r:>+8.2f}% {t:>6} {w:>4.0f}% {bp:>5.0f}%")

def halves():
    s8={x['dt']:x for x in pickle.load(open(PKL(8),'rb'))}
    s6={x['dt']:x for x in pickle.load(open(PKL(6),'rb'))}
    dts=sorted(set(s8)&set(s6)); m=len(dts)//2
    H2=dts[:m]; H1=dts[m:]   # engine convention: H1=recent, H2=earlier
    df=build_indicators(); D=detectors(df)
    key=['sma48>sma100','sma24>sma100','vol_calm','tsmom_672h']
    def flagmap(series, sub):
        return {ds:(bool(series.get(pd.to_datetime(ds),True)) if pd.to_datetime(ds) in series.index else True) for ds in sub}
    print(f"\n  Temporal stability @ conf 90/70 (full-window-best for all 4)")
    print(f"  H2 (earlier/chop): {H2[0]}->{H2[-1]}   H1 (recent/rally): {H1[0]}->{H1[-1]}")
    print(f"\n  {'detector':<16} {'FULL':>18} {'H2 earlier':>18} {'H1 recent':>18}")
    for name in key:
        out=[]
        for sub in (dts,H2,H1):
            r,t,w,_=simulate(sub, s8, s6, flagmap(D[name],sub), 90, 70)
            out.append(f"{r:+6.2f}% ({t}t,{w:.0f}%)")
        star='*' if name in {'sma24>sma100','vol_calm','tsmom_672h'} else ' '
        print(f" {star}{name:<16} {out[0]:>18} {out[1]:>18} {out[2]:>18}")

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--gen',type=int); ap.add_argument('--sweep',action='store_true'); ap.add_argument('--halves',action='store_true')
    a=ap.parse_args()
    if a.gen: gen(a.gen)
    elif a.sweep: sweep()
    elif a.halves: halves()
