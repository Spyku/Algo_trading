"""
forensic_misses_capture_8h.py — WHY are up-moves missed, and can a different CAPTURE
mechanism (not a new feature) catch them? ETH 8h, recent 2-month window.
=====================================================================================
Replaces the crude mean-comparison forensic. For every up-move (fwd8h >= UP) the
baseline RF+LGBM doesn't BUY-through-gate, classify the FAILURE MODE from the model's
own output, then test capture mechanisms and show recovered-misses vs return tradeoff.

Failure modes (2-model vote: BUY iff both predict up; SELL iff both down; else HOLD):
  GATE-BLOCKED : both models up, but confidence < gate  -> model SAW it, threshold killed it
  SPLIT (HOLD) : one up one down                          -> borderline
  WRONG (SELL) : both models down on a +UP move           -> signal genuinely absent

Capture methods tested (all reuse the same per-bar probas; SELL = vote-SELL throughout):
  baseline      BUY iff both-up & conf>=gate           (live logic)
  gate55        baseline vote, bull gate 55 / bear 60
  gate50        baseline vote, gate 50 / 55
  relaxed_vote  BUY iff >=1 model up & conf>=gate       (catches splits)
  proba055      BUY iff avg_proba>=0.55 (ignore vote unanimity)

Read: a method is interesting only if it recovers misses AND keeps gated return >= baseline.
If every recovering method loses return, the misses are correct skips and capture is a mirage.

Run:  python tools/forensic_misses_capture_8h.py            # replay 1440
      python tools/forensic_misses_capture_8h.py --up 0.015 --replay 1440
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


def prod_ctx():
    p = pd.read_csv("models/crypto_ed_production.csv")
    r = p[(p.coin == ASSET) & (p.horizon == HORIZON)].iloc[0]
    feats = [f.strip() for f in str(r["optimal_features"]).split(",")
             if f.strip() and f.strip() != "nan"]
    return int(r["best_window"]), float(r["gamma"]), feats


def decide(r, method):
    """Return ('BUY'|'SELL'|'HOLD') for a capture method, given per-bar record r."""
    pu = r["pred_up"]          # # models predicting up (0,1,2)
    avg = r["avg"]
    gate_b, gate_e = {"baseline": (BULL_CONF, BEAR_CONF), "gate55": (55, 60),
                      "gate50": (50, 55)}.get(method, (BULL_CONF, BEAR_CONF))
    gate = gate_b if r["bull"] else gate_e
    conf = round(avg * 100)
    if pu == 0:                # both down -> SELL (unchanged across methods)
        return "SELL"
    if method in ("baseline", "gate55", "gate50"):
        return "BUY" if (pu == 2 and conf >= gate) else "HOLD"
    if method == "relaxed_vote":
        return "BUY" if (pu >= 1 and conf >= (BULL_CONF if r["bull"] else BEAR_CONF)) else "HOLD"
    if method == "proba055":
        return "BUY" if avg >= 0.55 else "HOLD"
    return "HOLD"


def sim(rows, method):
    cash, held, in_pos, entry = 1000.0, 0.0, False, 0.0
    trades = wins = 0
    first = last = None
    bought_idx = set()
    for k, r in enumerate(rows):
        px = r["close"]; last = px
        if first is None:
            first = px
        act = decide(r, method)
        if act == "BUY" and not in_pos:
            held, cash, in_pos, entry = cash * (1 - FEE) / px, 0.0, True, px
            trades += 1; bought_idx.add(k)
        elif act == "SELL" and in_pos:
            cash = held * px * (1 - FEE); wins += int(px > entry); held, in_pos = 0.0, False
    if in_pos and last:
        cash = held * last * (1 - FEE); wins += int(last > entry)
    return {"ret": (cash / 1000 - 1) * 100, "trades": trades,
            "wr": (wins / trades * 100) if trades else 0.0,
            "bh": (last / first - 1) * 100 if first and last else 0.0,
            "bought": bought_idx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--up", type=float, default=0.02)
    ap.add_argument("--replay", type=int, default=1440)
    args = ap.parse_args()

    window, gamma, feats = prod_ctx()
    df_raw = F.load_data(ASSET)
    df, _ = F._build_features(df_raw, ASSET, feature_override=feats, horizon=HORIZON)
    close = df["close"].values
    dtv = pd.to_datetime(df["datetime"]).values
    bull = (np.log(df["close"] / df["close"].shift(672)) > 0).values
    n = len(df)
    start = max(window + 50, n - args.replay)

    print("=" * 92)
    print(f"  MISSED-UP-MOVE FORENSIC + CAPTURE TEST — ETH {HORIZON}h  (RF+LGBM, w={window})")
    print(f"  replay={args.replay}h | up-move = fwd{HORIZON}h >= {args.up:.1%} | bull@{BULL_CONF} bear@{BEAR_CONF}")
    print("=" * 92)

    rows = []
    for i in range(start, n):
        ts, te = max(0, i - window), max(0, i - HORIZON)
        tr = df.iloc[ts:te]
        y = tr["label"].values
        if len(np.unique(y[~pd.isnull(y)])) < 2 or pd.isnull(y).any():
            continue
        if tr[feats].isnull().any().any() or df.iloc[i:i + 1][feats].isnull().any().any():
            continue
        sc = StandardScaler()
        Xs = pd.DataFrame(sc.fit_transform(tr[feats]), columns=feats)
        Xts = pd.DataFrame(sc.transform(df.iloc[i:i + 1][feats]), columns=feats)
        sw = F.get_decay_weights(len(y), gamma)
        preds, probas = [], []
        for m in ("RF", "LGBM"):
            mdl = F.ALL_MODELS[m]()
            mdl.fit(Xs, y, sample_weight=sw)
            preds.append(int(mdl.predict(Xts)[0]))
            probas.append(float(mdl.predict_proba(Xts)[0][1]))
        fwd8 = (close[i + HORIZON] / close[i] - 1) if i + HORIZON < n else np.nan
        rows.append({"close": close[i], "bull": bool(bull[i]), "fwd8": fwd8,
                     "pred_up": int(sum(preds)), "avg": float(np.mean(probas)),
                     "label": int(df.iloc[i]["label"])})

    d = pd.DataFrame(rows)
    print(f"  eval bars: {len(d)}  ({pd.Timestamp(dtv[start])} .. {pd.Timestamp(dtv[n-1])})\n")

    up = d["fwd8"] >= args.up
    base = sim(rows, "baseline")
    base_buy = np.zeros(len(d), bool)
    base_buy[list(base["bought"])] = True
    # for failure-mode classification use per-bar decision (not position state)
    acts = np.array([decide(r, "baseline") for r in rows])
    miss = up.values & (acts != "BUY")
    conf = (d["avg"] * 100).round()

    print("  PHASE 1 — why are up-moves missed (per-bar decision):")
    print(f"    up-moves (fwd8h>={args.up:.1%}): {up.sum()}   baseline BUY-decisions on them: {(up.values&(acts=='BUY')).sum()}   MISSED: {miss.sum()}")
    gate_blocked = miss & (d["pred_up"].values == 2)             # both up but conf<gate
    split = miss & (d["pred_up"].values == 1)
    wrong = miss & (d["pred_up"].values == 0)
    print(f"    GATE-BLOCKED (both up, conf<gate): {gate_blocked.sum():3d}   "
          f"med conf {conf[gate_blocked].median() if gate_blocked.sum() else float('nan'):.0f} (gate {BULL_CONF}/{BEAR_CONF})")
    print(f"    SPLIT (1 up 1 down, ->HOLD):       {split.sum():3d}   med avg-proba {d['avg'][split].median() if split.sum() else float('nan'):.3f}")
    print(f"    WRONG (both down, ->SELL):         {wrong.sum():3d}   med avg-proba {d['avg'][wrong].median() if wrong.sum() else float('nan'):.3f}")
    print(f"    label=1 share on missed bars: {d['label'][miss].mean()*100 if miss.sum() else float('nan'):.0f}%  "
          f"(model's fee-aware target; high = model 'should' have been long)")

    print("\n  PHASE 2 — capture methods (recovered misses vs gated return):")
    print(f"    {'method':<14}{'ret':>9}{'win%':>7}{'trades':>8}{'recov/'+str(miss.sum()):>10}")
    for mth in ("baseline", "gate55", "gate50", "relaxed_vote", "proba055"):
        s = sim(rows, mth)
        bought = np.zeros(len(d), bool); bought[list(s["bought"])] = True
        # recovered = missed up-bars this method now BUY-decides on
        macts = np.array([decide(r, mth) for r in rows])
        recov = (miss & (macts == "BUY")).sum()
        tag = "  <- live" if mth == "baseline" else ""
        print(f"    {mth:<14}{s['ret']:>+8.1f}%{s['wr']:>6.0f}%{s['trades']:>8}{recov:>10}{tag}")
    print(f"    {'Buy&Hold':<14}{base['bh']:>+8.1f}%")
    print("=" * 92)
    print("  READ: if every method that recovers misses also LOSES return, the misses are correct")
    print("        skips — capture is a mirage. A winner needs recovered>0 AND ret >= baseline.")


if __name__ == "__main__":
    main()
