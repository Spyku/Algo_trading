"""2026-06-15: 1wk / 1mo / 2mo return + WR for the 3 top detector configs from the
detector RST run, on a consistent pre-gate sim (matches Mode S level).

Reads cached 5h + 8h walk-forward signals (data/_detbt_sig_{5,8}h.pkl, produced by
detector_sweep_86.py --gen), computes the needed detectors, and simulates each
config over the last 168h (1wk), 720h (1mo), 1440h (2mo).

  --sim    read pickles, compute, print the window table
"""
import os, sys, pickle, argparse
import numpy as np, pandas as pd

FEE = 0.0011         # per leg
MAX_HOLD = 10
HURST_WIN = 240
WINDOWS = [('1wk', 168), ('1mo', 720), ('2mo', 1440)]
P5, P8 = 'data/_detbt_sig_5h.pkl', 'data/_detbt_sig_8h.pkl'

# 3 configs: (label, detector_fn_builder, bullC, bearC)  bull=8h, bear=5h
def _hurst_rs(prices):
    p = np.asarray(prices, float); lr = np.diff(np.log(p)); N = len(lr)
    if N < 20: return np.nan
    pts = []
    for k in (kk for kk in (10,20,40,80) if kk < N):
        nch = N//k; vals=[]
        for i in range(nch):
            seg = lr[i*k:(i+1)*k]; s = seg.std()
            if s>0:
                dev = np.cumsum(seg-seg.mean()); vals.append((dev.max()-dev.min())/s)
        if vals: pts.append((k, np.mean(vals)))
    if len(pts)<2: return np.nan
    return float(np.polyfit(np.log([x[0] for x in pts]), np.log([x[1] for x in pts]),1)[0])

def indicators():
    df = pd.read_csv('data/ETH_hourly_data.csv')
    df['datetime']=pd.to_datetime(df['datetime']); df=df.sort_values('datetime').set_index('datetime').tail(1440+800)
    c=df['close']
    X=pd.DataFrame(index=df.index)
    X['close']=c; X['sma48']=c.rolling(48).mean(); X['sma100']=c.rolling(100).mean(); X['sma72']=c.rolling(72).mean()
    X['hurst']=np.log(c).rolling(HURST_WIN).apply(_hurst_rs, raw=True)
    X['tsmom672']=np.log(c/c.shift(672))
    return X.to_dict('index')

def simulate(dts, s8, s5, is_bull, bullC, bearC):
    pos='cash'; e=0; ei=0; eq=1.0; tr=0; wins=0
    for i,dt in enumerate(dts):
        bull = is_bull(dt)
        act = s8.get(dt) if bull else s5.get(dt)
        if act is None: continue
        gate = bullC if bull else bearC
        if pos=='cash':
            if act['sig']=='BUY' and act['conf']>=gate:
                pos='long'; e=act['close']; ei=i
        else:
            if act['sig']=='SELL' or (i-ei)>=MAX_HOLD:
                pnl=(act['close']/e-1)-2*FEE; eq*=(1+pnl); tr+=1; wins+=1 if pnl>0 else 0; pos='cash'
    if pos=='long':
        last=s8.get(dts[-1]) or s5.get(dts[-1]); pnl=(last['close']/e-1)-2*FEE; eq*=(1+pnl); tr+=1; wins+=1 if pnl>0 else 0
    return (eq-1)*100, tr, (100*wins/tr if tr else 0)

def main():
    s5={x['dt']:x for x in pickle.load(open(P5,'rb'))}
    s8={x['dt']:x for x in pickle.load(open(P8,'rb'))}
    dts=sorted(set(s5)&set(s8))
    XBD=indicators()
    def g(dt,k):
        r=XBD.get(pd.to_datetime(dt));
        return r[k] if r else np.nan
    # detector builders (return is_bull(dt))
    def det_combo(dt):   # sma48>sma100 & hurst>0.5
        a=g(dt,'sma48')>g(dt,'sma100'); h=g(dt,'hurst')
        return bool(a and (h>0.5 if h==h else True))
    def det_sma48(dt):
        return bool(g(dt,'sma48')>g(dt,'sma100'))
    def det_price72(dt):
        return bool(g(dt,'close')>g(dt,'sma72'))
    def det_tsmom(dt):   # tsmom_672h (LIVE production detector)
        v=g(dt,'tsmom672'); return bool(v>0) if v==v else True
    configs=[
        ('tsmom_672h LIVE (8h@80/5h@70)',           det_tsmom, 80, 70),
        ('sma48>sma100 & hurst>0.5 (combo, 90/65)', det_combo, 90, 65),
        ('sma48>sma100 (single, 90/65)',            det_sma48, 90, 65),
        ('price>sma72 (single, 70/65)',             det_price72, 70, 65),
    ]
    print(f"\n  window data: {dts[0]} -> {dts[-1]}  ({len(dts)} hrs)")
    print(f"\n  {'config':<42} " + "  ".join(f"{w[0]:>16}" for w in WINDOWS))
    print("  " + "-"*92)
    for label, det, bc, rc in configs:
        cells=[]
        for _,n in WINDOWS:
            sub=dts[-n:] if len(dts)>=n else dts
            ret,tr,wr=simulate(sub, s8, s5, det, bc, rc)
            cells.append(f"{ret:+6.1f}% {wr:3.0f}%WR {tr:2d}t")
        print(f"  {label:<42} " + "  ".join(f"{c:>16}" for c in cells))
    print("\n  (pre-gate, fee 0.11%/leg, max_hold 10h; 1wk=168h 1mo=720h 2mo=1440h)")
    print("  [ref] engine Mode-S 2mo: combo +55.98%/80%, sma48>sma100 +55.68%/80%, price>sma72 +57.95%/76%")

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--sim',action='store_true'); a=ap.parse_args()
    main()
