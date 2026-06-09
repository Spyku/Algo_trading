"""
bt_multiview_features_8h.py — Does splitting features ACROSS two models help? (ETH 8h)
=====================================================================================
DIAGNOSTIC for the multi-view ensemble idea: instead of two correlated models on the
SAME features (the prod RF+LGBM case — generate_signals binds all members to one
feature_override), give each model a DIFFERENT feature view and combine them.

Partition (engine's own taxonomy, _cat @ crypto_trading_system_faye.py:2127):
  FAST view  = technical + pysr_ + deriv_       (hourly, price/microstructure)
  SLOW view  = m_ + xa_ + fg_ + vix_ + oc_      (daily, macro/context)

Replicates generate_signals' exact model logic (LGBM-solo: pred = proba>0.5,
conf = round(proba*100) on BUY) at the walk-forward level so each model can use its
own columns. Then runs the SAME maker-fee regime sim as bt_lgbm_tune_8h.py.

Four arms, all LGBM-solo, all prod-identical params (mc20/reg0, n300 GPU), same bars:
  FULL  : prod's 24 features (the incumbent feature set)        <- anchor
  FAST  : top-K fast-view features (leak-free selection)
  SLOW  : top-K slow-view features (leak-free selection)
  BLEND : average(FAST proba, SLOW proba)  <- the different-features ensemble

The two gating questions this answers:
  (1) CONDITION 1 — is each view individually good?   -> FAST & SLOW standalone sim ret
  (2) CONDITION 2 — are the views decorrelated?        -> error-correlation(FAST, SLOW)
A multi-view ensemble only helps if BOTH hold. If SLOW is garbage or err-corr is high,
the answer is "no, your features don't support a split" — and we learned it in one run.

Leak-free feature selection: top-K per view by LGBM importance fit ONLY on bars before
the eval window (df[:start]); the sim never sees those bars. Mirrors prod (select on the
6mo Mode D window, deploy forward).

Run:  python tools/bt_multiview_features_8h.py                 # replay 1440, step 1, K 15
      python tools/bt_multiview_features_8h.py --replay 720 --step 6 --k 12   # quick look
CAVEATS: hold-shield NOT modeled (applies equally to all arms — relative ranking holds).
         step>1 undersamples the sim (fewer decision points); use step 1 for the verdict.
"""
import os
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import crypto_trading_system_faye as F  # noqa: E402

ASSET, HORIZON = "ETH", 8
FEE = F.BACKTEST_FEE_PER_LEG
BULL_CONF, BEAR_CONF = 65, 70   # config/regime_config_ed.json ETH


def _cat(name):
    """Engine's own feature taxonomy (faye:2127)."""
    if name.startswith('oc_'): return 'on-chain'
    if name.startswith('m_'): return 'macro'
    if name.startswith('xa_'): return 'cross-asset'
    if name.startswith('deriv_'): return 'derivatives'
    if name.startswith(('fg_', 'vix_')): return 'sentiment'
    if name.startswith('gp_'): return 'geopolitical'
    if name.startswith('stable_'): return 'stablecoin'
    if name.startswith('pysr_'): return 'pysr'
    return 'technical'


FAST_CATS = {'technical', 'pysr', 'derivatives'}
SLOW_CATS = {'macro', 'cross-asset', 'sentiment', 'on-chain'}


def prod_ctx():
    p = pd.read_csv("models/crypto_ed_production.csv")
    r = p[(p.coin == ASSET) & (p.horizon == HORIZON)].iloc[0]
    feats = [f.strip() for f in str(r["optimal_features"]).split(",")
             if f.strip() and f.strip() != "nan"]
    return int(r["best_window"]), float(r["gamma"]), feats


def select_topk(df, cols, end, gamma, k):
    """Top-k cols by LGBM importance, fit ONLY on df[:end] (leak-free)."""
    pre = df.iloc[:end]
    y = pre["label"].values
    m = ~pd.isnull(y)
    X, y = pre[cols][m], y[m].astype(int)
    if len(np.unique(y)) < 2 or len(X) < 100:
        return cols[:k]
    mdl = F.ALL_MODELS['LGBM']()
    mdl.fit(X, y, sample_weight=F.get_decay_weights(len(y), gamma))
    imp = pd.Series(mdl.feature_importances_, index=cols).sort_values(ascending=False)
    return imp.head(k).index.tolist()


def fit_proba(Xtr, y, sw, Xte):
    sc = StandardScaler()
    Xs = pd.DataFrame(sc.fit_transform(Xtr), columns=Xtr.columns)
    Xts = pd.DataFrame(sc.transform(Xte), columns=Xtr.columns)
    mdl = F.ALL_MODELS['LGBM']()
    mdl.fit(Xs, y, sample_weight=sw)
    return float(mdl.predict_proba(Xts)[0][1])


def to_signal(p):
    """LGBM-solo logic, mirrors generate_signals / diag_lgbm_proba_spread."""
    if p > 0.5:
        return "BUY", round(p * 100)
    if p < 0.5:
        return "SELL", round((1 - p) * 100)
    return "HOLD", 50


def sim(rows, pkey, window_h=None):
    """Maker-fee regime sim, identical gate logic to bt_lgbm_tune_8h.py."""
    rs = rows
    if window_h:
        cut = rs[-1]["dt"] - pd.Timedelta(hours=window_h)
        rs = [r for r in rs if r["dt"] >= cut]
    cash, held, in_pos, entry = 1000.0, 0.0, False, 0.0
    trades = wins = 0
    first = last = None
    for r in rs:
        px = r["close"]
        last = px
        if first is None:
            first = px
        sig, conf = to_signal(r[pkey])
        mc = BULL_CONF if r["bull"] else BEAR_CONF
        act = "HOLD" if (sig == "BUY" and conf < mc) else sig
        if act == "BUY" and not in_pos:
            held, cash, in_pos, entry = cash * (1 - FEE) / px, 0.0, True, px
            trades += 1
        elif act == "SELL" and in_pos:
            cash = held * px * (1 - FEE)
            wins += int(px > entry)
            held, in_pos = 0.0, False
    if in_pos and last:
        cash = held * last * (1 - FEE)
        wins += int(last > entry)
    return {"ret": (cash / 1000 - 1) * 100, "trades": trades,
            "wr": (wins / trades * 100) if trades else 0.0,
            "bh": (last / first - 1) * 100 if first and last else 0.0, "n": len(rs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=int, default=1440)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--k", type=int, default=15, help="features per view")
    ap.add_argument("--windows", type=int, nargs="+", default=[1440, 720, 336])
    args = ap.parse_args()
    windows = [w for w in args.windows if w <= args.replay]

    window, gamma, prod_feats = prod_ctx()
    df_raw = F.load_data(ASSET)
    df, all_cols = F._build_features(df_raw, ASSET, feature_override=None, horizon=HORIZON)
    close = df["close"].values
    dt = pd.to_datetime(df["datetime"]).values
    bull = (np.log(df["close"] / df["close"].shift(672)) > 0).values
    n = len(df)
    start = max(window + 50, n - args.replay)

    fast_all = [c for c in all_cols if _cat(c) in FAST_CATS]
    slow_all = [c for c in all_cols if _cat(c) in SLOW_CATS]
    fast = select_topk(df, fast_all, start, gamma, args.k)
    slow = select_topk(df, slow_all, start, gamma, args.k)
    views = {"FULL": prod_feats, "FAST": fast, "SLOW": slow}

    print("=" * 96)
    print(f"  MULTI-VIEW feature-split diagnostic — ETH {HORIZON}h  (LGBM-solo, prod params, n300)")
    print(f"  window={window} gamma={gamma} | maker {FEE*100:.2f}%/leg | bull@{BULL_CONF} bear@{BEAR_CONF} (tsmom_672h)")
    print(f"  replay={args.replay}h step={args.step} | universe={len(all_cols)} "
          f"(fast-pool={len(fast_all)} slow-pool={len(slow_all)})")
    print("=" * 96)
    print(f"  FULL ({len(prod_feats)}): {', '.join(prod_feats)}")
    print(f"  FAST ({len(fast)}): {', '.join(fast)}")
    print(f"  SLOW ({len(slow)}): {', '.join(slow)}")
    print("=" * 96)

    rows = []
    allv = list(dict.fromkeys(sum(views.values(), [])))
    for i in range(start, n, args.step):
        ts, te = max(0, i - window), max(0, i - HORIZON)
        tr = df.iloc[ts:te]
        y = tr["label"].values
        if len(np.unique(y[~pd.isnull(y)])) < 2:
            continue
        te_row = df.iloc[i:i + 1]
        # require all view columns clean on this bar (aligned streams across arms)
        if tr[allv].isnull().any().any() or te_row[allv].isnull().any().any():
            continue
        if pd.isnull(y).any():
            continue
        sw = F.get_decay_weights(len(y), gamma)
        row = {"dt": pd.Timestamp(dt[i]), "close": close[i],
               "bull": bool(bull[i]), "label": int(df.iloc[i]["label"])}
        for name, cols in views.items():
            row[name] = fit_proba(tr[cols], y, sw, te_row[cols])
        row["BLEND"] = 0.5 * (row["FAST"] + row["SLOW"])
        rows.append(row)

    if not rows:
        print("  no clean eval bars — check feature NaN coverage / replay window")
        return
    d = pd.DataFrame(rows)
    print(f"\n  eval bars: {len(d)}  ({d.dt.min()} .. {d.dt.max()})\n")

    # ---- CONDITION 2: decorrelation ----
    pred_corr = d["FAST"].corr(d["SLOW"])
    err_corr = (d["FAST"] - d["label"]).corr(d["SLOW"] - d["label"])
    print("  CONDITION 2 — diversity (FAST vs SLOW):")
    print(f"    prediction corr = {pred_corr:+.3f}   error corr = {err_corr:+.3f}   "
          f"({'DECORRELATED (good)' if err_corr < 0.6 else 'too correlated — split buys little'})")

    # ---- CONDITION 1: each view individually good ----
    print("\n  CONDITION 1 — standalone signal quality (full eval window):")
    for name in ("FULL", "FAST", "SLOW", "BLEND"):
        p = d[name]
        acc = ((p > 0.5).astype(int) == d["label"]).mean() * 100
        try:
            auc = roc_auc_score(d["label"], p)
        except ValueError:
            auc = float("nan")
        print(f"    {name:6} acc={acc:4.1f}%  auc={auc:.3f}  p[mean={p.mean():.3f} std={p.std():.3f}]")

    # ---- gated regime sim ----
    print("\n" + "=" * 96 + "\n  GATED REGIME SIM  (ret% / win% / n trades)")
    head = f"  {'Arm':<8}" + "".join(f"{str(w)+'h':>18}" for w in windows)
    print(head + "\n  " + "-" * (len(head) - 2))
    res = {name: {w: sim(rows, name, w) for w in windows}
           for name in ("FULL", "FAST", "SLOW", "BLEND")}
    for name in ("FULL", "FAST", "SLOW", "BLEND"):
        print(f"  {name:<8}" + "".join(
            f"{res[name][w]['ret']:+6.1f}% {res[name][w]['wr']:3.0f}% n{res[name][w]['trades']:>3}".rjust(18)
            for w in windows))
    print(f"  {'Buy&Hold':<8}" + "".join(f"{res['FULL'][w]['bh']:+.1f}%".rjust(18) for w in windows))
    print("=" * 96)
    print("  VERDICT: BLEND must beat FULL *and* both views must be individually positive *and*")
    print("           err-corr must be low. If BLEND<FULL or SLOW is garbage -> features don't split.")
    print("  NOTE: hold-shield not modeled (equal across arms). step>1 undersamples — use step 1.")


if __name__ == "__main__":
    main()
