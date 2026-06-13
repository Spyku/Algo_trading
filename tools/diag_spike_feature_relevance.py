"""
diag_spike_feature_relevance.py — Which features (if any) anticipate fast spikes?
=================================================================================
RANKS the existing + tested feature universe by association with the onset of FAST
SPIKES (the 73 up-moves <=15min, >=1.2% from diag_fast_spikes_resolution). MODEL-FREE,
no engine changes, no backtest — pure feature/target correlation.

Setup (strictly causal, no look-ahead):
  - Fast spike = price up >= THRESH within <= WINDOW minutes (non-overlapping), on 1m data.
  - Hourly bar `datetime` is the candle OPEN; its features are known at close = datetime+1h.
  - A spike whose onset floors to hour H is "predicted" by the bar with datetime == H-1h
    (its features are known exactly at H, before the spike starts). That bar is labeled 1.
  - Eval is restricted to hourly bars inside the 1m-data coverage (only there can a
    no-spike bar be labeled 0 with confidence).

Candidate universe:
  prod live features  +  nv_* (intra-hour/order-flow)  +  fast-window analogues for
  W in {3,4,5,6}h (the 3/4/5h features tested 2026-06-10, plus 6h).

For each feature: point-biserial corr r vs the spike-onset label, |r|, two-sided p,
and ROC-AUC (>0.5 => higher feature value precedes a spike). Ranked by |r|.
Noise floor |r|* = 1.96/sqrt(N): below it, indistinguishable from chance.

Run:  python tools/diag_spike_feature_relevance.py
      python tools/diag_spike_feature_relevance.py --window 15 --thresh 0.012 --top 15
"""
import os
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import crypto_trading_system_faye as F  # noqa: E402
from tools.mock_nervosity_8h import build_nv, NV_FEATS  # noqa: E402
from tools.mock_fast_features_8h import add_fast, fast_names  # noqa: E402

ASSET, HORIZON = "ETH", 8
MIN_NS = 60 * 1_000_000_000
FAST_WINDOWS = [3, 4, 5, 6]


def prod_feats():
    p = pd.read_csv("models/crypto_ed_production.csv")
    r = p[(p.coin == ASSET) & (p.horizon == HORIZON)].iloc[0]
    return [f.strip() for f in str(r["optimal_features"]).split(",")
            if f.strip() and f.strip() != "nan"]


def find_spikes(path, window_min, thr):
    """Non-overlapping up-spikes (>=thr within <=window_min). Returns onset Timestamps."""
    m = pd.read_csv(path, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    ts = m["datetime"].values.astype("datetime64[ns]").astype(np.int64)
    close = m["close"].values.astype(float)
    n = len(m)
    win_ns = window_min * MIN_NS
    onsets = []
    i = 0
    while i < n - 1:
        j_end = int(np.searchsorted(ts, ts[i] + win_ns, side="right"))
        seg = close[i:j_end]
        if len(seg) < 2:
            i += 1
            continue
        k = int(np.argmax(seg))
        if seg[k] / close[i] - 1.0 >= thr and k > 0:
            onsets.append(pd.Timestamp(m["datetime"].iloc[i]))
            i = i + k + 1
        else:
            i += 1
    cov = (pd.Timestamp(m["datetime"].iloc[0]), pd.Timestamp(m["datetime"].iloc[-1]))
    return onsets, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/eth_1m_data.csv")
    ap.add_argument("--window", type=int, default=15, help="max minutes for a fast spike")
    ap.add_argument("--thresh", type=float, default=0.012)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    feats = prod_feats()
    df_raw = F.load_data(ASSET)
    df, _ = F._build_features(df_raw, ASSET, feature_override=feats, horizon=HORIZON)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # intra-hour / order-flow features
    nv = build_nv(args.file)
    df = df.merge(nv, on="datetime", how="left")

    # fast-window analogues
    fast_all = []
    for W in FAST_WINDOWS:
        df = add_fast(df, W)
        fast_all += fast_names(W)

    universe = feats + NV_FEATS + fast_all
    universe = [f for f in dict.fromkeys(universe) if f in df.columns]

    # spike onsets -> causal decision bar (datetime == spike_hour - 1h)
    onsets, cov = find_spikes(args.file, args.window, args.thresh)
    dec_bars = {pd.Timestamp(t).floor("h") - pd.Timedelta(hours=1) for t in onsets}
    df["spike"] = df["datetime"].isin(dec_bars).astype(int)

    # eval = hourly bars fully inside 1m coverage (so a 0 is a real no-spike bar)
    lo = cov[0].floor("h")
    hi = cov[1].floor("h") - pd.Timedelta(hours=1)
    ev = df[(df["datetime"] >= lo) & (df["datetime"] <= hi)].copy()

    N = len(ev)
    npos = int(ev["spike"].sum())
    noise = 1.96 / np.sqrt(N) if N else np.nan
    print("=" * 92)
    print("  SPIKE-FEATURE RELEVANCE — ETH  (model-free, strictly causal lead-1)")
    print(f"  fast spike = up >= {args.thresh:.1%} within <= {args.window}min (non-overlapping)")
    print(f"  1m coverage: {cov[0]} .. {cov[1]}   |  spikes found: {len(onsets)}")
    print(f"  eval hourly bars: {N}   spike-onset bars (label=1): {npos}  "
          f"(base rate {npos/max(1,N)*100:.1f}%)")
    print(f"  noise floor |r|* = 1.96/sqrt(N) = {noise:.3f}  (|r| below this == chance)")
    print("=" * 92)

    y = ev["spike"].values
    res = []
    for f in universe:
        x = ev[f].values.astype(float)
        ok = ~np.isnan(x)
        if ok.sum() < 50 or len(np.unique(x[ok])) < 3 or y[ok].sum() < 3:
            continue
        r, p = stats.pointbiserialr(y[ok], x[ok])
        try:
            auc = roc_auc_score(y[ok], x[ok])
        except ValueError:
            auc = np.nan
        kind = ("nv" if f in NV_FEATS else
                "fast" if f in fast_all else "prod")
        res.append((f, kind, r, abs(r), p, auc))

    res.sort(key=lambda t: -t[3])
    print(f"  {'rank':>4}  {'feature':<20}{'kind':>6}{'r':>9}{'|r|':>8}{'p':>9}{'AUC':>8}{'sig?':>6}")
    print("  " + "-" * 86)
    for rank, (f, kind, r, ar, p, auc) in enumerate(res[:args.top], 1):
        sig = "yes" if ar > noise else "no"
        print(f"  {rank:>4}  {f:<20}{kind:>6}{r:>+9.3f}{ar:>8.3f}{p:>9.3f}{auc:>8.3f}{sig:>6}")

    print("  " + "-" * 86)
    sig_feats = [t for t in res if t[3] > noise]
    fast_sig = [t for t in sig_feats if t[1] == "fast"]
    nv_sig = [t for t in sig_feats if t[1] == "nv"]
    print(f"  features above noise floor: {len(sig_feats)} / {len(res)}   "
          f"(prod {sum(1 for t in sig_feats if t[1]=='prod')}, nv {len(nv_sig)}, fast {len(fast_sig)})")
    if sig_feats:
        best = sig_feats[0]
        print(f"  strongest association: {best[0]} ({best[1]})  |r|={best[3]:.3f}  AUC={best[5]:.3f}")
    # explicit verdict on the tested 3/4/5h fast features
    print("  3/4/5h fast features ranking (the ones tested 2026-06-10):")
    rank_of = {t[0]: i + 1 for i, t in enumerate(res)}
    for W in (3, 4, 5):
        names = [n for n in fast_names(W) if n in rank_of]
        best_n = min(names, key=lambda n: rank_of[n]) if names else None
        if best_n:
            t = next(x for x in res if x[0] == best_n)
            print(f"    {W}h: best = {best_n} rank {rank_of[best_n]}/{len(res)}, "
                  f"|r|={t[3]:.3f} (noise {noise:.3f}) -> {'signal' if t[3]>noise else 'NOISE'}")
    print("=" * 92)
    print("  READ: 'most relevant' = top by |r|. If even the best |r| barely clears the noise")
    print("        floor and AUC ~0.5, NO feature anticipates fast spikes — onset is unpredictable")
    print("        from the hourly state, consistent with the closed reactivity book.")


if __name__ == "__main__":
    main()
