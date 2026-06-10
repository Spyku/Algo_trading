"""
mock_fast_features_8h.py — MOCK: add faster-window analogues of the live 24 features.
====================================================================================
Does NOT touch the engine or prod. Builds faster variants (window F, default 6) of the
technical features the live ETH 8h model favors, adds all 9 on top of the live 24,
runs RF+LGBM at w=169 / gamma=0.9998, and reports (a) which fast features LGBM weights
(importance), and (b) gated-sim backtest mock vs the live-24 baseline.

Fast features for window F (faithful analogues of build_hourly_features):
  kama_F, price_to_sma{F}h, adx_{F}h, plus_di_{F}h, vol_of_vol_{F}h,
  vol_ratio_{F}_{2F}, spread_{2F}h_{F}h, price_accel_{F}h, logret_{F}h
Whichever already exist in the engine universe are reused (reported), the rest created.
All are price-derived, causal/trailing (0-lag bucket) — verify with the truncation test.

Run:  python tools/mock_fast_features_8h.py --fast-window 6     # 6h variants
      python tools/mock_fast_features_8h.py --fast-window 5     # 5h/4h-ish variants
NOTE: hold-shield not modeled (equal both arms). Importance on the full 33-feat model
      is a proxy for "would Mode D select it"; a real add needs engine wiring + HRST.
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


def fast_names(W):
    return [f"kama_{W}", f"price_to_sma{W}h", f"adx_{W}h", f"plus_di_{W}h",
            f"vol_of_vol_{W}h", f"vol_ratio_{W}_{2*W}", f"spread_{2*W}h_{W}h",
            f"price_accel_{W}h", f"logret_{W}h"]


def prod_ctx():
    p = pd.read_csv("models/crypto_ed_production.csv")
    r = p[(p.coin == ASSET) & (p.horizon == HORIZON)].iloc[0]
    feats = [f.strip() for f in str(r["optimal_features"]).split(",")
             if f.strip() and f.strip() != "nan"]
    return int(r["best_window"]), float(r["gamma"]), feats


def kama(c, er, fast=2.0/3, slow=2.0/31):
    c = np.asarray(c, float); n = len(c); out = np.zeros(n)
    if n == 0:
        return out
    w = min(er, n); out[:w] = c[:w]
    ad = np.abs(np.diff(c, prepend=c[0]))
    for i in range(er, n):
        d = abs(c[i] - c[i - er]); v = ad[i - er + 1:i + 1].sum()
        e = d / v if v > 0 else 0.0
        sc = (e * (fast - slow) + slow) ** 2
        out[i] = out[i - 1] + sc * (c[i] - out[i - 1])
    return out


def add_fast(df, W):
    c, h, lo = df["close"], df["high"], df["low"]
    lr1 = df["logret_1h"] if "logret_1h" in df else np.log(c / c.shift(1))
    lF = df[f"logret_{W}h"] if f"logret_{W}h" in df else np.log(c / c.shift(W))
    l2F = df[f"logret_{2*W}h"] if f"logret_{2*W}h" in df else np.log(c / c.shift(2 * W))
    df[f"kama_{W}"] = kama(c.values, er=W)
    df[f"price_to_sma{W}h"] = c / c.rolling(W).mean() - 1
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    plus_dm = (h - h.shift(1)).clip(lower=0); minus_dm = (lo.shift(1) - lo).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0); minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr = tr.rolling(W).mean()
    pdi = 100 * (plus_dm.rolling(W).mean() / (atr + 1e-10))
    mdi = 100 * (minus_dm.rolling(W).mean() / (atr + 1e-10))
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    df[f"adx_{W}h"] = dx.rolling(W).mean(); df[f"plus_di_{W}h"] = pdi
    base = max(W + 2, round(1.5 * W))
    df[f"vol_of_vol_{W}h"] = lr1.rolling(base).std().rolling(W).std()
    df[f"vol_ratio_{W}_{2*W}"] = lr1.rolling(W).std() / (lr1.rolling(2 * W).std() + 1e-10)
    df[f"spread_{2*W}h_{W}h"] = l2F - lF
    df[f"price_accel_{W}h"] = lF.diff(W)
    df[f"logret_{W}h"] = lF
    return df


def vote(preds, probas):
    br = sum(preds) / len(preds)
    sig = "BUY" if br > 0.5 else ("SELL" if br == 0 else "HOLD")
    avg = float(np.mean(probas))
    return sig, (round((1 - avg) * 100) if sig == "SELL" else round(avg * 100))


def sim(rows, key, window_h=None):
    rs = rows
    if window_h:
        cut = rs[-1]["dt"] - pd.Timedelta(hours=window_h)
        rs = [r for r in rs if r["dt"] >= cut]
    cash, held, in_pos, entry = 1000.0, 0.0, False, 0.0
    trades = wins = 0; first = last = None
    for r in rs:
        px = r["close"]; last = px
        if first is None:
            first = px
        sig, conf = r[key]
        mc = BULL_CONF if r["bull"] else BEAR_CONF
        act = "HOLD" if (sig == "BUY" and conf < mc) else sig
        if act == "BUY" and not in_pos:
            held, cash, in_pos, entry = cash * (1 - FEE) / px, 0.0, True, px; trades += 1
        elif act == "SELL" and in_pos:
            cash = held * px * (1 - FEE); wins += int(px > entry); held, in_pos = 0.0, False
    if in_pos and last:
        cash = held * last * (1 - FEE); wins += int(last > entry)
    return {"ret": (cash / 1000 - 1) * 100, "trades": trades,
            "wr": (wins / trades * 100) if trades else 0.0,
            "bh": (last / first - 1) * 100 if first and last else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast-window", type=int, default=6)
    ap.add_argument("--replay", type=int, default=1440)
    ap.add_argument("--windows", type=int, nargs="+", default=[1440, 720, 336])
    args = ap.parse_args()
    W = args.fast_window
    windows = [w for w in args.windows if w <= args.replay]

    window, gamma, prod = prod_ctx()
    df_raw = F.load_data(ASSET)
    df, _ = F._build_features(df_raw, ASSET, feature_override=prod, horizon=HORIZON)
    FAST = fast_names(W)
    existed = [f for f in FAST if f in df.columns]
    df = add_fast(df, W)
    created = [f for f in FAST if f not in existed]
    mock = prod + FAST
    close = df["close"].values
    dtv = pd.to_datetime(df["datetime"]).values
    bull = (np.log(df["close"] / df["close"].shift(672)) > 0).values
    n = len(df)
    start = max(window + 50, n - args.replay)

    print("=" * 96)
    print(f"  MOCK FAST FEATURES (W={W}h) — ETH {HORIZON}h  baseline(24) vs mock(+{len(FAST)}) | RF+LGBM w={window} g={gamma}")
    print(f"  created: {', '.join(created)}")
    print(f"  existed: {', '.join(existed) if existed else '(none)'}")
    print("=" * 96)

    rows = []
    imp = {f: [] for f in mock}
    for i in range(start, n):
        ts, te = max(0, i - window), max(0, i - HORIZON)
        tr = df.iloc[ts:te]; y = tr["label"].values
        if len(np.unique(y[~pd.isnull(y)])) < 2 or pd.isnull(y).any():
            continue
        if tr[mock].isnull().any().any() or df.iloc[i:i + 1][mock].isnull().any().any():
            continue
        sw = F.get_decay_weights(len(y), gamma)
        rec = {"dt": pd.Timestamp(dtv[i]), "close": close[i], "bull": bool(bull[i])}
        for tag, fl in (("base", prod), ("mock", mock)):
            sc = StandardScaler()
            Xs = pd.DataFrame(sc.fit_transform(tr[fl]), columns=fl)
            Xts = pd.DataFrame(sc.transform(df.iloc[i:i + 1][fl]), columns=fl)
            preds, probas = [], []
            for m in ("RF", "LGBM"):
                mdl = F.ALL_MODELS[m]()
                mdl.fit(Xs, y, sample_weight=sw)
                preds.append(int(mdl.predict(Xts)[0])); probas.append(float(mdl.predict_proba(Xts)[0][1]))
                if tag == "mock" and m == "LGBM":
                    s = pd.Series(mdl.feature_importances_, index=fl); t = s.sum() or 1
                    for f in fl:
                        imp[f].append(s[f] / t)
            rec[tag] = vote(preds, probas)
        rows.append(rec)

    if not rows:
        print("  no clean eval bars"); return
    print(f"  eval bars: {len(rows)}  ({rows[0]['dt']} .. {rows[-1]['dt']})\n")

    mean_imp = {f: (np.mean(imp[f]) if imp[f] else 0.0) for f in mock}
    rank = {f: r for r, (f, _) in enumerate(sorted(mean_imp.items(), key=lambda x: -x[1]), 1)}
    print(f"  FEATURE DETECTION — {W}h fast features in the {len(mock)}-feat mock (LGBM importance):")
    print(f"    {'fast feature':<18}{'imp%':>8}{'rank':>7}   (vs prod median imp "
          f"{np.median([mean_imp[f] for f in prod])*100:.1f}%)")
    for f in sorted(FAST, key=lambda x: -mean_imp[x]):
        det = "DETECTED" if rank[f] <= 24 else "ignored"
        print(f"    {f:<18}{mean_imp[f]*100:>7.1f}%{rank[f]:>6}   {det}")
    top = sorted(mean_imp.items(), key=lambda x: -x[1])[:8]
    print("    top-8 overall: " + ", ".join(f"{f}({v*100:.1f}%)" for f, v in top))

    print("\n  BACKTEST (gated regime sim) — ret% / win% / trades:")
    head = f"  {'arm':<10}" + "".join(f"{str(w)+'h':>18}" for w in windows)
    print(head + "\n  " + "-" * (len(head) - 2))
    for tag in ("base", "mock"):
        res = {w: sim(rows, tag, w) for w in windows}
        label = "baseline" if tag == "base" else f"mock+{W}h"
        print(f"  {label:<10}" + "".join(
            f"{res[w]['ret']:+6.1f}% {res[w]['wr']:3.0f}% n{res[w]['trades']:>3}".rjust(18) for w in windows))
    bh = {w: sim(rows, 'base', w) for w in windows}
    print(f"  {'Buy&Hold':<10}" + "".join(f"{bh[w]['bh']:+.1f}%".rjust(18) for w in windows))
    print("=" * 96)
    print("  VERDICT: fast features help only if some rank top-24 (detected) AND mock beats baseline.")


if __name__ == "__main__":
    main()
