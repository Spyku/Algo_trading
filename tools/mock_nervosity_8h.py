"""
mock_nervosity_8h.py — Forensic on MISSED up-moves + mock engine with order-flow features.
=========================================================================================
Two phases, one walk-forward, ETH 8h, over the window where 1-min data exists
(eth_1m_data.csv, ~Mar 4 - May 3). Tests whether intra-hour / order-flow "nervosity"
features would catch the fast up-moves the hourly model misses.

Nervosity features (per hour, from 1m, ALL known at the hour's close = decision time):
  nv_taker_imb   = (2*taker_buy_base - volume)/volume   net aggressive-buy ratio [-1,1]
  nv_taker_imb_z = 24h z-score of nv_taker_imb          reactive flow deviation
  nv_rvol        = std of 1m logrets within the hour     intra-hour realized vol
  nv_runup       = max intra-hour close/cummin - 1       biggest intra-hour spike

PHASE 1 (forensic): baseline RF+LGBM (prod 24 feats). A "miss" = bar with fwd-8h
return >= UP_THRESH where the model did NOT issue a gate-clearing BUY. Compare nv-feature
means on missed vs caught up-move bars vs all bars -> did flow/spike light up on the misses?

PHASE 2 (mock): refit RF+LGBM with nv feats added. Gated-sim baseline vs mock, how many
previously-missed up-moves the mock now catches, and LGBM importance of the nv feats.

Run:  python tools/mock_nervosity_8h.py            # replay = full 1m overlap
      python tools/mock_nervosity_8h.py --up 0.015
NOTE: hold-shield not modeled (equal both arms). Window is Mar-May (only 1m coverage),
      not the literal last month. Mechanism is window-agnostic: no signal here -> none later.
"""
import os
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import crypto_trading_system_faye as F  # noqa: E402

ASSET, HORIZON = "ETH", 8
FEE = F.BACKTEST_FEE_PER_LEG
BULL_CONF, BEAR_CONF = 65, 70
NV_FEATS = ["nv_taker_imb", "nv_taker_imb_z", "nv_rvol", "nv_runup"]


def prod_ctx():
    p = pd.read_csv("models/crypto_ed_production.csv")
    r = p[(p.coin == ASSET) & (p.horizon == HORIZON)].iloc[0]
    feats = [f.strip() for f in str(r["optimal_features"]).split(",")
             if f.strip() and f.strip() != "nan"]
    return int(r["best_window"]), float(r["gamma"]), feats


def build_nv(path="data/eth_1m_data.csv"):
    m = pd.read_csv(path, parse_dates=["datetime"]).sort_values("datetime")
    m["hour"] = m["datetime"].dt.floor("h")
    m["lr"] = np.log(m["close"] / m["close"].shift(1))

    def per_hour(x):
        vol = x["volume"].sum()
        tb = x["taker_buy_base_volume"].sum()
        c = x["close"].values
        runup = float(np.max(c / np.minimum.accumulate(c) - 1)) if len(c) else 0.0
        return pd.Series({
            "nv_taker_imb": (2 * tb - vol) / (vol + 1e-9),
            "nv_rvol": float(x["lr"].std()),
            "nv_runup": runup,
        })

    nv = m.groupby("hour").apply(per_hour)
    nv["nv_taker_imb_z"] = ((nv["nv_taker_imb"] - nv["nv_taker_imb"].rolling(24, min_periods=6).mean())
                            / (nv["nv_taker_imb"].rolling(24, min_periods=6).std() + 1e-9))
    nv = nv.reset_index().rename(columns={"hour": "datetime"})
    return nv


def vote(preds, probas):
    br = sum(preds) / len(preds)
    sig = "BUY" if br > 0.5 else ("SELL" if br == 0 else "HOLD")
    avg = float(np.mean(probas))
    conf = round((1 - avg) * 100) if sig == "SELL" else round(avg * 100)
    return sig, conf


def gated(sig, conf, bull):
    mc = BULL_CONF if bull else BEAR_CONF
    return "HOLD" if (sig == "BUY" and conf < mc) else sig


def sim(rows, key):
    cash, held, in_pos, entry = 1000.0, 0.0, False, 0.0
    trades = wins = 0
    first = last = None
    for r in rows:
        px = r["close"]; last = px
        if first is None:
            first = px
        act = gated(*r[key], r["bull"])
        if act == "BUY" and not in_pos:
            held, cash, in_pos, entry = cash * (1 - FEE) / px, 0.0, True, px
            trades += 1
        elif act == "SELL" and in_pos:
            cash = held * px * (1 - FEE); wins += int(px > entry); held, in_pos = 0.0, False
    if in_pos and last:
        cash = held * last * (1 - FEE); wins += int(last > entry)
    return {"ret": (cash / 1000 - 1) * 100, "trades": trades,
            "wr": (wins / trades * 100) if trades else 0.0,
            "bh": (last / first - 1) * 100 if first and last else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--up", type=float, default=0.02, help="fwd-8h return that defines an up-move")
    args = ap.parse_args()

    window, gamma, feats = prod_ctx()
    df_raw = F.load_data(ASSET)
    df, _ = F._build_features(df_raw, ASSET, feature_override=feats, horizon=HORIZON)
    nv = build_nv()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.merge(nv, on="datetime", how="left")

    close = df["close"].values
    dtv = df["datetime"].values
    bull = (np.log(df["close"] / df["close"].shift(672)) > 0).values
    n = len(df)
    have_nv = df["nv_taker_imb"].notna().values
    cov = np.where(have_nv)[0]
    start = max(window + 50, cov.min() if len(cov) else n)
    end = (cov.max() + 1) if len(cov) else n
    mock_feats = feats + NV_FEATS

    print("=" * 96)
    print(f"  NERVOSITY MOCK — ETH {HORIZON}h  baseline(prod {len(feats)}) vs mock(+{len(NV_FEATS)} flow/intrahour)")
    print(f"  window={window} gamma={gamma} | up-move = fwd{HORIZON}h >= {args.up:.1%} | maker {FEE*100:.2f}%/leg "
          f"bull@{BULL_CONF} bear@{BEAR_CONF}")
    print(f"  nv feats: {', '.join(NV_FEATS)}")
    print("=" * 96)

    rows, nvrec = [], []
    imp_accum = {f: [] for f in NV_FEATS}
    for i in range(start, end):
        if not have_nv[i]:
            continue
        ts, te = max(0, i - window), max(0, i - HORIZON)
        tr = df.iloc[ts:te]
        y = tr["label"].values
        if len(np.unique(y[~pd.isnull(y)])) < 2 or pd.isnull(y).any():
            continue
        if tr[mock_feats].isnull().any().any() or df.iloc[i:i + 1][mock_feats].isnull().any().any():
            continue
        sw = F.get_decay_weights(len(y), gamma)
        fwd8 = (close[i + HORIZON] / close[i] - 1) if i + HORIZON < n else np.nan
        rec = {"close": close[i], "bull": bool(bull[i]), "fwd8": fwd8}
        for tag, fl in (("base", feats), ("mock", mock_feats)):
            sc = StandardScaler()
            Xs = pd.DataFrame(sc.fit_transform(tr[fl]), columns=fl)
            Xts = pd.DataFrame(sc.transform(df.iloc[i:i + 1][fl]), columns=fl)
            preds, probas = [], []
            for mname in ("RF", "LGBM"):
                mdl = F.ALL_MODELS[mname]()
                mdl.fit(Xs, y, sample_weight=sw)
                preds.append(int(mdl.predict(Xts)[0]))
                probas.append(float(mdl.predict_proba(Xts)[0][1]))
                if tag == "mock" and mname == "LGBM":
                    imp = pd.Series(mdl.feature_importances_, index=fl)
                    tot = imp.sum() or 1
                    for f in NV_FEATS:
                        imp_accum[f].append(imp.get(f, 0) / tot)
            rec[tag] = vote(preds, probas)
        rows.append(rec)
        nvrec.append({f: float(df.iloc[i][f]) for f in NV_FEATS})

    if not rows:
        print("  no clean eval bars (1m coverage / NaN)"); return
    d = pd.DataFrame(rows)
    nvd = pd.DataFrame(nvrec)
    print(f"  eval bars: {len(d)}  ({pd.Timestamp(dtv[start])} .. {pd.Timestamp(dtv[end-1])})\n")

    # ---- PHASE 1: forensic on missed up-moves ----
    d["up"] = d["fwd8"] >= args.up
    d["base_buy"] = [gated(s, c, b) == "BUY" for (s, c), b in zip(d["base"], d["bull"])]
    d["mock_buy"] = [gated(s, c, b) == "BUY" for (s, c), b in zip(d["mock"], d["bull"])]
    up = d["up"]
    caught = up & d["base_buy"]
    missed = up & ~d["base_buy"]
    print("  PHASE 1 — forensic (baseline RF+LGBM):")
    print(f"    up-moves (fwd8h>={args.up:.1%}): {up.sum()}   caught(BUY): {caught.sum()}   "
          f"MISSED(flat): {missed.sum()}   capture rate: {caught.sum()/max(1,up.sum())*100:.0f}%")
    print(f"    {'feature':<16}{'missed-up':>12}{'caught-up':>12}{'all-bars':>12}{'  signal?':>10}")
    for f in NV_FEATS:
        mm = nvd.loc[missed.values, f].mean() if missed.sum() else np.nan
        cm = nvd.loc[caught.values, f].mean() if caught.sum() else np.nan
        am = nvd[f].mean()
        sd = nvd[f].std() + 1e-9
        flag = "yes" if abs(mm - am) / sd > 0.3 else "no"   # missed-up deviates >0.3sd from baseline
        print(f"    {f:<16}{mm:>12.3f}{cm:>12.3f}{am:>12.3f}{flag:>10}")

    # ---- PHASE 2: mock engine ----
    print("\n  PHASE 2 — mock engine (baseline vs +nv):")
    b, mo = sim(rows, "base"), sim(rows, "mock")
    print(f"    {'arm':<10}{'ret':>9}{'win%':>7}{'trades':>8}")
    print(f"    {'baseline':<10}{b['ret']:>+8.1f}%{b['wr']:>6.0f}%{b['trades']:>8}")
    print(f"    {'mock':<10}{mo['ret']:>+8.1f}%{mo['wr']:>6.0f}%{mo['trades']:>8}")
    print(f"    {'Buy&Hold':<10}{b['bh']:>+8.1f}%")
    recovered = (missed & d["mock_buy"]).sum()
    print(f"    previously-missed up-moves now caught by mock: {recovered}/{missed.sum()}")
    print(f"    nv-feature importance in mock LGBM (mean share): "
          + ", ".join(f"{f}={np.mean(imp_accum[f])*100:.1f}%" for f in NV_FEATS))
    print("=" * 96)
    print("  VERDICT: nv helps only if Phase-1 shows signal on misses AND mock beats baseline on")
    print("           the gated sim AND LGBM actually weights the nv feats. Else order flow is not a lever.")


if __name__ == "__main__":
    main()
