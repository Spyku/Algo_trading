"""2026-06-15: full audit of all 45 swept detectors — group into families by
mechanism AND by actual regime agreement (redundancy), with nervousness (flips),
bull%, and the Mode-S return/WR. Model-free (price series only).

Run:  python tools/detector_audit.py
"""
import numpy as np, pandas as pd, re

LOG = 'logs/hrst/faye_run_20260615_214325.log'
HURST_WIN = 240

# ---- indicators ----
df = pd.read_csv('data/ETH_hourly_data.csv'); df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
c, h, l = df['close'], df['high'], df['low']
def sma(w): return c.rolling(w).mean()
def ema(s): return c.ewm(span=s).mean()
def wil(s, n): return s.ewm(alpha=1/n, adjust=False).mean()
rsi_d = c.diff(); rsi = 100 - 100/(1 + rsi_d.clip(lower=0).rolling(14).mean()/(-rsi_d.clip(upper=0)).rolling(14).mean().replace(0, np.nan))
dd48 = (c/c.rolling(48).max()-1)*100; dd72 = (c/c.rolling(72).max()-1)*100
bnc48 = (c/c.rolling(48).min()-1)*100
pc = c.shift(1); tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
up, dn = h.diff(), -l.diff(); pdm = ((up>dn)&(up>0))*up; mdm = ((dn>up)&(dn>0))*dn
atr = wil(tr,14); pdi = 100*wil(pdm,14)/atr; mdi = 100*wil(mdm,14)/atr
adx = wil(100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan),14)
def _hrs(p):
    p=np.asarray(p,float); lr=np.diff(np.log(p)); N=len(lr)
    if N<20: return np.nan
    pts=[]
    for k in (kk for kk in (10,20,40,80) if kk<N):
        nch=N//k; v=[]
        for i in range(nch):
            s=lr[i*k:(i+1)*k]; sd=s.std()
            if sd>0: d=np.cumsum(s-s.mean()); v.append((d.max()-d.min())/sd)
        if v: pts.append((k,np.mean(v)))
    return float(np.polyfit(np.log([x[0] for x in pts]),np.log([x[1] for x in pts]),1)[0]) if len(pts)>=2 else np.nan
hurst = np.log(c).rolling(HURST_WIN).apply(_hrs, raw=True)

# ---- 45 detector boolean series + mechanism family ----
S, FAM = {}, {}
def add(name, series, fam): S[name]=series; FAM[name]=fam
# SMA crosses
for nm,(a,b) in {'sma6>sma24':(6,24),'sma12>sma24':(12,24),'sma12>sma48':(12,48),'sma24>sma72':(24,72),
                 'sma24>sma100':(24,100),'sma48>sma100':(48,100),'sma48>sma200':(48,200),'sma168>sma480':(168,480)}.items():
    add(nm, sma(a)>sma(b), 'SMA-cross')
# price vs SMA
for w in (24,48,72,100,200): add(f'price>sma{w}', c>sma(w), 'price-vs-SMA')
# EMA crosses
add('ema9>ema21', ema(9)>ema(21), 'EMA-cross'); add('ema12>ema26', ema(12)>ema(26), 'EMA-cross')
# momentum
for nm,lag in {'tsmom_672h':672,'tsmom_168h':168,'tsmom_72h':72}.items(): add(nm, np.log(c/c.shift(lag))>0, 'momentum')
add('macd>0', (ema(12)-ema(26))>0, 'momentum')
# rsi
for t in (45,50,55): add(f'rsi>{t}', rsi>t, 'RSI')
# drawdown
for t in (2,3,5): add(f'dd48>-{t}%', dd48>-t, 'drawdown')
for t in (3,5): add(f'dd72>-{t}%', dd72>-t, 'drawdown')
# bounce
for t in (2,3): add(f'bounce48>{t}%', bnc48>t, 'bounce')
# vol / strength / hurst
v = (np.log(c/c.shift(1)).abs()); sf=v.groupby(df['datetime'].dt.hour).transform(lambda s:s.rolling(30,min_periods=10).mean())
vd=(v/sf.replace(0,np.nan)).rolling(24).std(); add('vol_calm', vd < vd.rolling(720,min_periods=240).quantile(0.70), 'volatility')
add('adx>25', adx>25, 'trend-strength'); add('adx>40', adx>40, 'trend-strength')
add('hurst>0.5', hurst>0.5, 'trend/mean-revert')
# composites
add('sma24>100+dd48>-3', (sma(24)>sma(100))&(dd48>-3), 'composite')
add('sma24>100+rsi>45', (sma(24)>sma(100))&(rsi>45), 'composite')
add('dd48>-3%+rsi>45', (dd48>-3)&(rsi>45), 'composite')
add('price>sma72+rsi>50', (c>sma(72))&(rsi>50), 'composite')
add('sma24>sma100 & adx>25', (sma(24)>sma(100))&(adx>25), 'composite')
add('sma24>sma100 & adx>40', (sma(24)>sma(100))&(adx>40), 'composite')
add('sma24>sma100 & hurst>0.5', (sma(24)>sma(100))&(hurst>0.5), 'composite')
add('sma48>sma100 & adx>25', (sma(48)>sma(100))&(adx>25), 'composite')
add('sma48>sma100 & adx>40', (sma(48)>sma(100))&(adx>40), 'composite')
add('sma48>sma100 & hurst>0.5', (sma(48)>sma(100))&(hurst>0.5), 'composite')
add('tsmom_672h & adx>25', (np.log(c/c.shift(672))>0)&(adx>25), 'composite')
add('tsmom_672h & hurst>0.5', (np.log(c/c.shift(672))>0)&(hurst>0.5), 'composite')

# ---- parse Mode-S best ret/WR per detector from the log ----
RET, WR = {}, {}
for ln in open(LOG, encoding='utf-8', errors='replace').readlines()[339:9162]:
    t=ln.split()
    if len(t)<11 or not t[0].isdigit(): continue
    try:
        det=' '.join(t[1:-10]); r=float(t[-6].replace('%','').replace('+','')); w=float(t[-4].replace('%',''))
        if det not in RET or r>RET[det]: RET[det]=r; WR[det]=w
    except: continue

# ---- window: last 1440h, valid ----
W = pd.DataFrame(S).tail(1440).dropna()
names = list(S)
flips = {k: int((W[k].astype(int).diff().abs()==1).sum()) for k in names}
bullpct = {k: W[k].mean()*100 for k in names}

print(f"AUDIT of {len(names)} detectors  |  window {len(W)}h\n")
fams = ['SMA-cross','price-vs-SMA','EMA-cross','momentum','RSI','drawdown','bounce','volatility','trend-strength','trend/mean-revert','composite']
for fam in fams:
    mem=[k for k in names if FAM[k]==fam]
    print(f"== {fam} ({len(mem)}) ==")
    for k in sorted(mem, key=lambda x: flips[x]):
        rr = f"{RET.get(k,float('nan')):+.1f}%" if k in RET else "  n/a"
        ww = f"{WR.get(k,0):.0f}%" if k in WR else " n/a"
        print(f"   {k:<26} flips={flips[k]:>3}  bull={bullpct[k]:>3.0f}%  ret={rr:>7}  WR={ww:>4}")
    print()

# ---- agreement clusters across ALL detectors (>=85% same regime) ----
order = sorted(names, key=lambda k: -RET.get(k,-99))  # best-return first = cluster rep
groups=[]
for k in order:
    placed=False
    for g in groups:
        if (W[k]==W[g[0]]).mean()>=0.85: g.append(k); placed=True; break
    if not placed: groups.append([k])
print(f"==== AGREEMENT CLUSTERS (>=85% same regime) : {len(groups)} distinct behaviors ====")
for gi,g in enumerate(sorted(groups,key=lambda g:-RET.get(g[0],-99)),1):
    rep=g[0]
    print(f"  C{gi:<2} rep=[{rep}] ret={RET.get(rep,float('nan')):+.1f}% flips={flips[rep]} | {len(g)} members: {', '.join(g)}")
