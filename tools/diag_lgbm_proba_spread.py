"""
diag_lgbm_proba_spread.py — WHY does the regularized LGBM (mc30/reg5) score worse?
=================================================================================
Walk-forwards GB+LGBM on ETH 8h (w169, gamma0.9998, 24 feats) over the last 1440h at
STEP=12, with LGBM at CURRENT (mc20/reg0) vs TUNED (mc30/reg5) params (via the engine
env hook). Replicates generate_signals' exact model logic (vote + averaged-proba
confidence) and captures, per bar: LGBM proba, GB proba, ensemble confidence, signal.

Tests the hypothesis: heavy regularization COMPRESSES LGBM's probability spread toward
0.5 -> fewer confident calls -> fewer BUYs clear the live conf gate (bull@65) -> misses
the up-moves. If true, TUNED shows lower proba std/IQR, fewer |p-0.5|>0.15 bars, and
fewer BUY>=65 than CURRENT.

Run:  python tools/diag_lgbm_proba_spread.py
"""
import os
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import crypto_trading_system_faye as F  # noqa: E402

ASSET, HORIZON, REPLAY, STEP, GATE = "ETH", 8, 1440, 12, 65

prod = pd.read_csv("models/crypto_ed_production.csv")
r = prod[(prod.coin == ASSET) & (prod.horizon == HORIZON)].iloc[0]
window, gamma = int(r["best_window"]), float(r["gamma"])
feats = [f.strip() for f in str(r["optimal_features"]).split(",") if f.strip() and f.strip() != "nan"]

df_raw = F.load_data(ASSET)
df, fcols = F._build_features(df_raw, ASSET, feature_override=feats, horizon=HORIZON)
n = len(df)
start = max(window + 50, n - REPLAY)
print(f"  ETH {HORIZON}h  w={window} gamma={gamma}  {len(fcols)} feats  |  bars {start}..{n} step {STEP}\n")


def run(env):
    for k in ("LGBM_MIN_CHILD", "LGBM_REG_LAMBDA"):
        os.environ.pop(k, None)
    os.environ.update(env)
    rows = []
    for i in range(start, n, STEP):
        ts = max(0, i - window)
        te = max(ts, i - HORIZON)
        tr = df.iloc[ts:te]
        X, y = tr[fcols], tr["label"].values
        if len(np.unique(y)) < 2:
            continue
        Xte = df.iloc[i:i + 1][fcols]
        if X.isnull().any().any() or Xte.isnull().any().any():
            continue
        sc = StandardScaler()
        Xs = pd.DataFrame(sc.fit_transform(X), columns=fcols)
        Xts = pd.DataFrame(sc.transform(Xte), columns=fcols)
        sw = F.get_decay_weights(len(y), gamma)
        ps, preds = {}, {}
        for m in ("GB", "LGBM"):
            mdl = F.ALL_MODELS[m]()
            mdl.fit(Xs, y, sample_weight=sw)
            preds[m] = int(mdl.predict(Xts)[0])
            ps[m] = float(mdl.predict_proba(Xts)[0][1])
        avg = np.mean(list(ps.values()))
        br = sum(preds.values()) / 2
        sig = "BUY" if br > 0.5 else ("SELL" if br == 0 else "HOLD")
        conf = round(avg * 100) if sig != "SELL" else round((1 - avg) * 100)
        rows.append(dict(lgbm_p=ps["LGBM"], gb_p=ps["GB"], lgbm_pred=preds["LGBM"],
                         gb_pred=preds["GB"], avg=avg, sig=sig, conf=conf))
    return pd.DataFrame(rows)


def summarize(tag, d):
    lp = d["lgbm_p"]
    iqr = lp.quantile(0.75) - lp.quantile(0.25)
    conf_frac = (lp.sub(0.5).abs() > 0.15).mean() * 100
    buys = d[d.sig == "BUY"]
    buys_gate = buys[buys.conf >= GATE]
    print(f"  {tag:8} | LGBM_p mean={lp.mean():.3f} std={lp.std():.3f} IQR={iqr:.3f} "
          f"min={lp.min():.3f} max={lp.max():.3f} | |p-.5|>.15: {conf_frac:4.0f}% | pred=1: {d.lgbm_pred.mean()*100:3.0f}%")
    print(f"  {'':8} | signals: BUY={ (d.sig=='BUY').sum():3d}  HOLD={(d.sig=='HOLD').sum():3d}  SELL={(d.sig=='SELL').sum():3d}"
          f"  | BUY>={GATE}: {len(buys_gate):3d}  meanConf(BUY)={buys.conf.mean() if len(buys) else float('nan'):.0f}")
    print(f"  {'':8} | GB_p std={d.gb_p.std():.3f} (partner, unchanged)  | LGBM-GB agree on BUY: {((d.lgbm_pred==1)&(d.gb_pred==1)).sum():3d}\n")
    return iqr


cur = run({})
tun = run({"LGBM_MIN_CHILD": "30", "LGBM_REG_LAMBDA": "5"})
for k in ("LGBM_MIN_CHILD", "LGBM_REG_LAMBDA"):
    os.environ.pop(k, None)
print("=" * 96)
print(f"  {len(cur)} eval bars (step {STEP})  —  LGBM proba spread & gate clearance: CURRENT vs TUNED")
print("=" * 96)
summarize("CURRENT", cur)
summarize("TUNED", tun)
print(f"  spread shrink (IQR):  current {cur['lgbm_p'].quantile(.75)-cur['lgbm_p'].quantile(.25):.3f} "
      f"-> tuned {tun['lgbm_p'].quantile(.75)-tun['lgbm_p'].quantile(.25):.3f}")
