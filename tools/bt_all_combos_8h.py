"""
bt_all_combos_8h.py — Horserace ALL model combos (size 2..5) on ETH 8h.
=======================================================================
Tests every ensemble combination of the 5 base models {RF, GB, XGB, LR, LGBM}
with >=2 members (10 pairs + 10 triples + 5 quads + 1 quintuple = 26 combos),
through the PRODUCTION model logic + the maker-fee regime sim.

EFFICIENT + FAITHFUL: each base model's per-bar (pred, proba) is independent of the
combo (same features, same training window), so we fit all 5 models ONCE per bar
(~5*1440 fits) and assemble every combo analytically. Reproduces generate_signals
exactly (faye:3104-3119):
    buy_ratio = sum(votes)/len(votes)
    BUY if buy_ratio>0.5 ; SELL if buy_ratio==0 ; else HOLD
    confidence = mean(member probas)   (1-mean for SELL)
All combos evaluate the SAME bars (skip depends only on the feature set) -> fair race.

Context: prod ETH 8h (window, gamma, 24 feats from crypto_ed_production.csv), same
StandardScaler + decay weights + embargo=horizon as generate_signals. Regime: tsmom_672h,
bull@65 bear@70, maker fee = BACKTEST_FEE_PER_LEG.

Run:  python tools/bt_all_combos_8h.py                  # replay 1440, windows 1440 720 336
      python tools/bt_all_combos_8h.py --replay 720
CAVEATS: hold-shield NOT modeled (equal across combos -> ranking holds). Solo (size-1)
rows are shown as reference only (prod removes them via MIN_COMBO_SIZE=2).
"""
import os
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import crypto_trading_system_faye as F  # noqa: E402

ASSET, HORIZON = "ETH", 8
FEE = F.BACKTEST_FEE_PER_LEG
BULL_CONF, BEAR_CONF = 65, 70
MODELS = ["RF", "GB", "XGB", "LR", "LGBM"]
PROD_COMBO = ("RF", "LGBM")   # current live 8h combo (marked in output)


def prod_ctx():
    p = pd.read_csv("models/crypto_ed_production.csv")
    r = p[(p.coin == ASSET) & (p.horizon == HORIZON)].iloc[0]
    feats = [f.strip() for f in str(r["optimal_features"]).split(",")
             if f.strip() and f.strip() != "nan"]
    return int(r["best_window"]), float(r["gamma"]), feats


def combo_sig(row, members):
    votes = [row["pred"][m] for m in members]
    probas = [row["proba"][m] for m in members]
    br = sum(votes) / len(votes)
    sig = "BUY" if br > 0.5 else ("SELL" if br == 0 else "HOLD")
    avg = float(np.mean(probas))
    conf = round((1 - avg) * 100) if sig == "SELL" else round(avg * 100)
    return sig, conf


def sim(rows, members, window_h=None):
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
        sig, conf = combo_sig(r, members)
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
            "bh": (last / first - 1) * 100 if first and last else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=int, default=1440)
    ap.add_argument("--windows", type=int, nargs="+", default=[1440, 720, 336])
    args = ap.parse_args()
    windows = [w for w in args.windows if w <= args.replay]

    window, gamma, feats = prod_ctx()
    df_raw = F.load_data(ASSET)
    df, _ = F._build_features(df_raw, ASSET, feature_override=feats, horizon=HORIZON)
    close = df["close"].values
    dt = pd.to_datetime(df["datetime"]).values
    bull = (np.log(df["close"] / df["close"].shift(672)) > 0).values
    n = len(df)
    start = max(window + 50, n - args.replay)

    print("=" * 100)
    print(f"  ALL-COMBOS horserace — ETH {HORIZON}h  (prod model logic, n300, "
          f"{len(feats)} feats w={window} gamma={gamma})")
    print(f"  replay={args.replay}h | maker {FEE*100:.2f}%/leg | bull@{BULL_CONF} bear@{BEAR_CONF} "
          f"(tsmom_672h) | models={'/'.join(MODELS)}")
    print("=" * 100)

    # ---- walk-forward: fit all 5 base models once per bar, cache pred+proba ----
    rows = []
    for i in range(start, n):
        ts, te = max(0, i - window), max(0, i - HORIZON)
        tr = df.iloc[ts:te]
        y = tr["label"].values
        if len(np.unique(y[~pd.isnull(y)])) < 2:
            continue
        Xtr, Xte = tr[feats], df.iloc[i:i + 1][feats]
        if Xtr.isnull().any().any() or Xte.isnull().any().any() or pd.isnull(y).any():
            continue
        sc = StandardScaler()
        Xs = pd.DataFrame(sc.fit_transform(Xtr), columns=feats)
        Xts = pd.DataFrame(sc.transform(Xte), columns=feats)
        sw = F.get_decay_weights(len(y), gamma)
        pred, proba = {}, {}
        ok = True
        for m in MODELS:
            try:
                mdl = F.ALL_MODELS[m]()
                mdl.fit(Xs, y, sample_weight=sw)
                pred[m] = int(mdl.predict(Xts)[0])
                proba[m] = float(mdl.predict_proba(Xts)[0][1])
            except Exception:
                ok = False
                break
        if not ok:
            continue
        rows.append({"dt": pd.Timestamp(dt[i]), "close": close[i],
                     "bull": bool(bull[i]), "pred": pred, "proba": proba})

    if not rows:
        print("  no clean eval bars"); return
    print(f"  eval bars: {len(rows)}  ({rows[0]['dt']} .. {rows[-1]['dt']})\n")

    # ---- enumerate combos (size 2..5), plus size-1 solos as reference ----
    combos = []
    for k in range(2, len(MODELS) + 1):
        combos += list(itertools.combinations(MODELS, k))
    solos = [(m,) for m in MODELS]

    def row_for(c):
        r = {w: sim(rows, c, w) for w in windows}
        return {"combo": "+".join(c), "k": len(c), "r": r,
                "prod": tuple(c) == PROD_COMBO}

    results = [row_for(c) for c in combos]
    solo_res = [row_for(c) for c in solos]
    results.sort(key=lambda x: x["r"][windows[0]]["ret"], reverse=True)

    def line(x):
        mark = " <- PROD" if x["prod"] else ""
        cells = "".join(
            f"{x['r'][w]['ret']:+6.1f}% {x['r'][w]['wr']:3.0f}% n{x['r'][w]['trades']:>3}".rjust(18)
            for w in windows)
        return f"  k{x['k']} {x['combo']:<18}" + cells + mark

    head = f"  {'combo':<24}" + "".join(f"{str(w)+'h':>18}" for w in windows)
    print(head + "\n  " + "-" * (len(head) + 6))
    for x in results:
        print(line(x))
    print("  " + "-" * (len(head) + 6))
    print("  (reference — solos, removed in prod by MIN_COMBO_SIZE=2):")
    for x in sorted(solo_res, key=lambda z: z["r"][windows[0]]["ret"], reverse=True):
        print(line(x))
    bh = results[0]["r"]
    print("  " + "-" * (len(head) + 6))
    print(f"  {'Buy & Hold':<24}" + "".join(f"{bh[w]['bh']:+.1f}%".rjust(18) for w in windows))
    print("=" * 100)
    best = results[0]
    prod = next(x for x in results if x["prod"])
    w0 = windows[0]
    print(f"  WINNER ({w0}h): {best['combo']}  {best['r'][w0]['ret']:+.1f}%  "
          f"(prod RF+LGBM {prod['r'][w0]['ret']:+.1f}%, delta {best['r'][w0]['ret']-prod['r'][w0]['ret']:+.1f}pp)")
    print("  NOTE: hold-shield not modeled (equal across combos). Promotion needs the gain to")
    print("        survive across windows AND a fresh out-of-sample replay — one window = window-shopping.")


if __name__ == "__main__":
    main()
