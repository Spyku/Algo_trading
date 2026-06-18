"""
diag_psych_floors_eth.py — Do psychological FLOORS predict bounces on ETH? (model-free)
========================================================================================
PHASE-1 GATE for the "local psychological floor" idea (TA support levels). MODEL-FREE:
no model, no fitting — just measures an unconditional market property, so there is no
overfitting risk and we can use long history. It answers ONE question before we spend a
Mode-D cycle: when ETH sits just above a floor, is the forward 5h/8h return higher than
baseline (a bounce), NET of regime? If not → STOP, don't build the feature (Rule 21:
the gated sim, never a screen — but first, the cheap diagnostic before even that).

The literature (Osler 2000/2003; arXiv 2101.07410; crypto clustering Urquhart'17/Hu'19)
says floors are REAL via order-clustering, BUT:
  * the ROUND-NUMBER reversal is SHORT-LIVED ("gone within an hour" in FX) — so at our
    hourly bar / 5-8h horizon it may already be spent. Flavor A tests exactly that.
  * the LOCAL-SUPPORT (prior-bounce) effect decays over DAYS and STRENGTHENS with prior
    bounce count — a better fit for our timescale. Flavor B is the recommended primary.

Two flavors, both vs a baseline of "not near any floor":
  A) ROUND NUMBERS  — bucket each hour by position in the $G grid: ON-FLOOR (just above a
     round level = support) / MID (baseline) / UNDER-CEILING (just below the next round =
     resistance). Tests support>resistance asymmetry. $-absolute → use a coherent price
     band (default recent 365d); spacing is reported as %-of-price so you can judge it.
  B) LOCAL SUPPORT — confirmed pivot-lows (causal: a pivot at k is only known at k+half).
     "near support" = a confirmed pivot-low sits within delta% below price. Stratified by
     PRIOR-BOUNCE COUNT {0, 1, 2+} — the literature's headline finding. %-relative → scale
     free.

EDGE = mean(fwd | event) - mean(fwd | baseline). Positive + |t|>~2 = floors predict bounces.
Reported overall AND split by regime (sma48>sma100, the live detector) — a floor edge that
only exists in one regime still matters to a regime-switched engine.

CAVEAT printed below: forward windows OVERLAP → the naive t-stat is anti-conservative
(autocorrelation inflates it). Treat |t| as a screen, not a p-value.

Run:  python tools/diag_psych_floors_eth.py
      python tools/diag_psych_floors_eth.py --days 0            # all history (mixes price bands for flavor A)
      python tools/diag_psych_floors_eth.py --delta 0.0075 --horizons 5 8 12
      python tools/diag_psych_floors_eth.py --grids 100 250 500 1000 --pivot-half 12
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def welch_t(ev, base):
    """t-stat for difference of means (unequal var). NOTE: overlapping fwd windows make
    this anti-conservative — use as a screen, not a p-value."""
    ev, base = np.asarray(ev), np.asarray(base)
    if len(ev) < 2 or len(base) < 2:
        return 0.0
    se2 = ev.var(ddof=1) / len(ev) + base.var(ddof=1) / len(base)
    return float((ev.mean() - base.mean()) / np.sqrt(se2)) if se2 > 0 else 0.0


def pivot_lows(low, half):
    """Boolean mask: pivot low at k if low[k] is the strict min of low[k-half : k+half+1].
    A pivot at k is only CONFIRMED (knowable) at index k+half — callers must respect that."""
    n = len(low)
    piv = np.zeros(n, dtype=bool)
    for k in range(half, n - half):
        seg = low[k - half:k + half + 1]
        if low[k] == seg.min() and np.argmin(seg) == half:
            piv[k] = True
    return piv


def summarize(mask, fwd, base_mask, regime_bull):
    """Return rows [(label, n, mean%, win%, t_vs_base)] for ALL / BULL / BEAR."""
    out = []
    base = fwd[base_mask & ~np.isnan(fwd)]
    for lab, rmask in (("all ", np.ones_like(regime_bull)),
                       ("bull", regime_bull), ("bear", ~regime_bull)):
        m = mask & rmask & ~np.isnan(fwd)
        ev = fwd[m]
        if len(ev) == 0:
            out.append((lab, 0, float("nan"), float("nan"), 0.0))
            continue
        b = fwd[base_mask & rmask & ~np.isnan(fwd)]
        out.append((lab, len(ev), ev.mean() * 100, (ev > 0).mean() * 100,
                    welch_t(ev, b if len(b) else base)))
    return out


def print_block(title, rows_by_h, horizons, base_n):
    print(f"\n  {title}   (baseline n={base_n})")
    print(f"    {'regime':>6} {'n':>6} " +
          " ".join(f"{'fwd'+str(h)+'h':>8} {'win':>5} {'dVbase':>7} {'t':>5}" for h in horizons))
    # rows_by_h: dict h -> list of (lab,n,mean,win,t); also need baseline means for delta
    labels = [r[0] for r in rows_by_h[horizons[0]]]
    for i, lab in enumerate(labels):
        cells = []
        for h in horizons:
            _, n, mean, win, t = rows_by_h[h][i]
            d = mean - BASE_MEAN[h].get(lab.strip(), float("nan"))
            if n == 0:
                cells.append(f"{'-':>8} {'-':>5} {'-':>7} {'-':>5}")
            else:
                cells.append(f"{mean:>7.2f}% {win:>4.0f}% {d:>+6.2f}% {t:>5.1f}")
        n0 = rows_by_h[horizons[0]][i][1]
        print(f"    {lab:>6} {n0:>6} " + " ".join(cells))


BASE_MEAN = {}  # h -> {regime_label: baseline mean%} filled per flavor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/eth_hourly_data.csv")
    ap.add_argument("--days", type=int, default=365,
                    help="recent window in days (round-number spacing is price-band dependent); 0 = all history")
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 8])
    ap.add_argument("--delta", type=float, default=0.005,
                    help="proximity band: 'near a floor' = within delta fraction of price (default 0.5%%)")
    ap.add_argument("--grids", type=float, nargs="+", default=[100, 250, 500],
                    help="round-number $ spacings to test (flavor A)")
    ap.add_argument("--pivot-half", type=int, default=12,
                    help="pivot-low half-window in hours (flavor B); pivot known only at k+half (causal)")
    ap.add_argument("--bounce-tol", type=float, default=0.01,
                    help="prior-bounce clustering tolerance: prior pivots within this %% of a level count as bounces")
    args = ap.parse_args()

    df = pd.read_csv(args.file, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    if args.days > 0:
        cutoff = df["datetime"].iloc[-1] - pd.Timedelta(days=args.days)
        df = df[df["datetime"] >= cutoff].reset_index(drop=True)
    close = df["close"].values.astype(float)
    low = df["low"].values.astype(float)
    n = len(df)
    sma48 = pd.Series(close).rolling(48).mean().values
    sma100 = pd.Series(close).rolling(100).mean().values
    regime_bull = sma48 > sma100  # live detector

    H = max(args.horizons)
    fwd = {h: np.full(n, np.nan) for h in args.horizons}
    for h in args.horizons:
        fwd[h][:n - h] = close[h:] / close[:n - h] - 1.0

    # valid decision rows: have all horizons forward + sma100 + pivot confirmation lookback
    valid = np.zeros(n, dtype=bool)
    lo = max(100, args.pivot_half)
    valid[lo:n - H] = True
    valid &= ~np.isnan(sma100)

    px_lo, px_hi = close[valid].min(), close[valid].max()
    span_days = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).days
    print("=" * 100)
    print("  PSYCHOLOGICAL FLOORS on ETH hourly — Phase-1 event study (MODEL-FREE)")
    print(f"  {df['datetime'].iloc[0]:%Y-%m-%d} .. {df['datetime'].iloc[-1]:%Y-%m-%d}  "
          f"({n:,} hrs, {span_days}d) | price band ${px_lo:,.0f}-${px_hi:,.0f} | "
          f"near=within {args.delta:.2%} | horizons={args.horizons}h")
    print(f"  regime: bull={regime_bull[valid].mean():.0%} of decision hours (sma48>sma100)")
    print("=" * 100)

    # =================== FLAVOR A: ROUND NUMBERS ===================
    print("\n" + "-" * 100)
    print("  FLAVOR A — ROUND NUMBERS (Osler: take-profit clusters AT round = floor; stops just below)")
    print("  ON-FLOOR = price within delta ABOVE a round level (resting on support) -> expect bounce (fwd > baseline)")
    print("  UNDER-CEIL = price within delta BELOW the next round (resistance overhead) -> expect drag (fwd < baseline)")
    print("-" * 100)
    for G in args.grids:
        dist_below = close % G                  # $ above nearest round below
        dist_above = G - dist_below             # $ below nearest round above
        band = args.delta * close               # $ band = delta% of price
        on_floor = valid & (dist_below <= band)
        under_ceil = valid & (dist_above <= band)
        mid = valid & ~on_floor & ~under_ceil   # baseline = between floors
        spacing_pct = G / np.median(close[valid]) * 100
        print(f"\n  $ grid = ${G:,.0f}  (~{spacing_pct:.1f}% of median price)   "
              f"on-floor={on_floor.sum()}  under-ceil={under_ceil.sum()}  mid(base)={mid.sum()}")
        BASE_MEAN.clear()
        for h in args.horizons:
            b = fwd[h][mid & ~np.isnan(fwd[h])]
            BASE_MEAN[h] = {"all": b.mean() * 100,
                            "bull": fwd[h][mid & regime_bull & ~np.isnan(fwd[h])].mean() * 100,
                            "bear": fwd[h][mid & ~regime_bull & ~np.isnan(fwd[h])].mean() * 100}
        rows_floor = {h: summarize(on_floor, fwd[h], mid, regime_bull) for h in args.horizons}
        rows_ceil = {h: summarize(under_ceil, fwd[h], mid, regime_bull) for h in args.horizons}
        print_block("ON-FLOOR (support test)", rows_floor, args.horizons, mid.sum())
        print_block("UNDER-CEILING (resistance test)", rows_ceil, args.horizons, mid.sum())

    # =================== FLAVOR B: LOCAL SUPPORT (pivot-lows, prior-bounce) ===================
    print("\n" + "-" * 100)
    print("  FLAVOR B — LOCAL SUPPORT (confirmed pivot-lows, causal) stratified by PRIOR-BOUNCE COUNT")
    print("  arXiv 2101.07410: more prior bounces at a level -> higher next-bounce probability. RECOMMENDED primary.")
    print("-" * 100)
    piv = pivot_lows(low, args.pivot_half)
    piv_k = np.where(piv)[0]
    piv_price = low[piv_k]
    confirm_at = piv_k + args.pivot_half  # index at which each pivot becomes knowable

    near_support = np.zeros(n, dtype=bool)
    bounce_ct = np.zeros(n, dtype=int)
    for i in np.where(valid)[0]:
        # pivots confirmed strictly before i
        m = confirm_at < i
        if not m.any():
            continue
        lv = piv_price[m]
        below = lv[lv <= close[i]]
        if len(below) == 0:
            continue
        nearest = below.max()                      # closest support below price
        if (close[i] - nearest) / close[i] <= args.delta:
            near_support[i] = True
            bounce_ct[i] = int((np.abs(lv - nearest) / nearest <= args.bounce_tol).sum())

    base_b = valid & ~near_support
    BASE_MEAN.clear()
    for h in args.horizons:
        BASE_MEAN[h] = {"all": fwd[h][base_b & ~np.isnan(fwd[h])].mean() * 100,
                        "bull": fwd[h][base_b & regime_bull & ~np.isnan(fwd[h])].mean() * 100,
                        "bear": fwd[h][base_b & ~regime_bull & ~np.isnan(fwd[h])].mean() * 100}
    print(f"\n  pivot half-window={args.pivot_half}h  pivots={len(piv_k)}  near-support hrs={near_support.sum()}  "
          f"baseline hrs={base_b.sum()}")
    for ct_lab, ct_mask in (("ALL near-support", near_support),
                            ("prior-bounce 0 (fresh)", near_support & (bounce_ct <= 1)),
                            ("prior-bounce 1", near_support & (bounce_ct == 2)),
                            ("prior-bounce 2+", near_support & (bounce_ct >= 3))):
        rows = {h: summarize(ct_mask, fwd[h], base_b, regime_bull) for h in args.horizons}
        print_block(ct_lab, rows, args.horizons, base_b.sum())
    print("  (bounce-count buckets: the level itself counts as 1 pivot; '0 fresh'=only this touch, '2+'=>=3 pivots stacked)")

    print("\n" + "=" * 100)
    print("  READ:")
    print("   * deltavbase = mean fwd return in the event MINUS baseline. POSITIVE on ON-FLOOR / near-support")
    print("     = floors predict bounces. The trading edge lives in delta, not the raw level.")
    print("   * |t| > ~2 screens as 'real' BUT forward windows overlap (autocorrelated) -> t is")
    print("     anti-conservative. A borderline t with a small delta is noise.")
    print("   * Flavor A round-number bounce fading at 5-8h (delta~0, |t|<2) would CONFIRM the resolution")
    print("     caveat (Osler's <1h reversal is gone by our horizon) -> round numbers are not our lever.")
    print("   * Flavor B delta rising with prior-bounce count is the literature's signature -> THE thing to see.")
    print("   * GATE: if no event class clears delta>~0.3% with |t|>2 in at least one regime, STOP — a")
    print("     gated feature A/B (Phase 2) would only lose (importance != performance).")
    print("=" * 100)


if __name__ == "__main__":
    main()
