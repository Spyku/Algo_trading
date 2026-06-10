"""
bt_intrahour_tp_8h.py — Does an INTRA-HOUR take-profit beat ride-to-signal? ETH 8h.
==================================================================================
Prior TP/SL tests were hourly-close resolution and all lost (capping winners surrenders
the fat tail). This tests the untested variant: an intra-hour TP that fires the moment
price wicks to the target MID-candle — catching spikes the hourly close gives back.

No 1m data needed: the hourly candle's `high` IS the intra-hour peak. While long, if
high[k] >= entry*(1+TP), a limit-sell fills at the target that hour; otherwise exit on
the normal gated SELL. Same RF+LGBM entries across all arms.

Sweep TP levels vs the no-TP baseline over the recent window. A TP "wins" only if total
return >= baseline. Reports avg win and TP-exit share so you can see the fat-tail cost.

Run:  python tools/bt_intrahour_tp_8h.py                 # replay 1440, default TP grid
      python tools/bt_intrahour_tp_8h.py --tps 0.02 0.03 0.05
NOTE: hold-shield not modeled (equal across arms). Assumes the TP limit fills at target
      when the bar's high reaches it (realistic for a resting maker sell).
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


def gated(sig, conf, bull):
    mc = BULL_CONF if bull else BEAR_CONF
    return "HOLD" if (sig == "BUY" and conf < mc) else sig


def sim(rows, tp=None):
    cash, held, in_pos, entry = 1000.0, 0.0, False, 0.0
    trades = wins = tp_exits = 0
    win_rets = []
    first = last = None
    for r in rows:
        px, hi = r["close"], r["high"]
        last = px
        if first is None:
            first = px
        if in_pos and tp is not None and hi >= entry * (1 + tp):   # intra-hour TP fills mid-candle
            sellpx = entry * (1 + tp)
            cash = held * sellpx * (1 - FEE)
            wins += 1; tp_exits += 1; win_rets.append(sellpx / entry - 1)
            held, in_pos = 0.0, False
            continue
        act = gated(r["sig"], r["conf"], r["bull"])
        if act == "BUY" and not in_pos:
            held, cash, in_pos, entry = cash * (1 - FEE) / px, 0.0, True, px
            trades += 1
        elif act == "SELL" and in_pos:
            cash = held * px * (1 - FEE)
            w = px > entry; wins += int(w)
            if w:
                win_rets.append(px / entry - 1)
            held, in_pos = 0.0, False
    if in_pos and last:
        cash = held * last * (1 - FEE)
        if last > entry:
            wins += 1; win_rets.append(last / entry - 1)
    return {"ret": (cash / 1000 - 1) * 100, "trades": trades,
            "wr": (wins / trades * 100) if trades else 0.0,
            "tp_exits": tp_exits, "avg_win": (np.mean(win_rets) * 100) if win_rets else 0.0,
            "bh": (last / first - 1) * 100 if first and last else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=int, default=1440)
    ap.add_argument("--tps", type=float, nargs="+", default=[0.015, 0.02, 0.03, 0.04, 0.05, 0.07])
    args = ap.parse_args()

    window, gamma, feats = prod_ctx()
    df_raw = F.load_data(ASSET)
    df, _ = F._build_features(df_raw, ASSET, feature_override=feats, horizon=HORIZON)
    if "high" not in df.columns:
        print("  ERROR: no 'high' column"); return
    close = df["close"].values
    high = df["high"].values
    dtv = pd.to_datetime(df["datetime"]).values
    bull = (np.log(df["close"] / df["close"].shift(672)) > 0).values
    n = len(df)
    start = max(window + 50, n - args.replay)

    print("=" * 88)
    print(f"  INTRA-HOUR TAKE-PROFIT — ETH {HORIZON}h  (RF+LGBM entries, w={window}, intra-hour high)")
    print(f"  replay={args.replay}h | maker {FEE*100:.2f}%/leg | bull@{BULL_CONF} bear@{BEAR_CONF}")
    print("=" * 88)

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
        br = sum(preds) / 2
        sig = "BUY" if br > 0.5 else ("SELL" if br == 0 else "HOLD")
        avg = float(np.mean(probas))
        conf = round((1 - avg) * 100) if sig == "SELL" else round(avg * 100)
        rows.append({"close": close[i], "high": high[i], "bull": bool(bull[i]),
                     "sig": sig, "conf": conf})

    print(f"  eval bars: {len(rows)}  ({pd.Timestamp(dtv[start])} .. {pd.Timestamp(dtv[n-1])})\n")
    base = sim(rows, None)
    print(f"  {'arm':<14}{'ret':>9}{'win%':>7}{'trades':>8}{'tp-exits':>10}{'avg-win':>9}")
    print(f"  {'baseline':<14}{base['ret']:>+8.1f}%{base['wr']:>6.0f}%{base['trades']:>8}{0:>10}{base['avg_win']:>8.2f}%  <- no TP")
    for tp in args.tps:
        s = sim(rows, tp)
        d = s["ret"] - base["ret"]
        print(f"  {'TP '+format(tp,'.1%'):<14}{s['ret']:>+8.1f}%{s['wr']:>6.0f}%{s['trades']:>8}"
              f"{s['tp_exits']:>10}{s['avg_win']:>8.2f}%   ({d:+.1f}pp)")
    print(f"  {'Buy&Hold':<14}{base['bh']:>+8.1f}%")
    print("=" * 88)
    print("  READ: a TP wins only if ret >= baseline. Watch avg-win collapse as TP tightens —")
    print("        that's the fat tail being surrendered. tp-exits = how often it capped a winner.")


if __name__ == "__main__":
    main()
