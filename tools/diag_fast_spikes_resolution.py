"""
diag_fast_spikes_resolution.py — Does the HOURLY clock structurally miss fast spikes?
=====================================================================================
MODEL-FREE diagnostic for "the model misses fast spikes entirely." Isolates the
RESOLUTION CEILING from model quality: a strategy that decides on hourly closes can,
at the earliest, enter at the first hourly close AFTER a spike begins. By then some
fraction of the move is already gone ("front-loaded"). This measures that haircut on
ETH 1-minute data, and how much 30m / 15m bars would recover.

For each fast up-spike (price rises >=THRESH within <=WINDOW minutes, non-overlapping):
  total      = peak/start - 1
  reachable_B = (peak / first_B-min_close_after_start) - 1, clipped to [0, total]
              = the part still capturable if you can only act on B-minute closes
  reachable fraction = reachable_B / total   (1.0 = fully catchable, 0 = spike spent before you can act)

Headline = value-weighted reachable fraction at 60 / 30 / 15 min. The 15m-minus-60m
gap is exactly what moving to sub-hourly (Ein/Eli) would buy. If 60m reachable is
already high, it's a MODEL latency problem, not resolution — sub-hourly won't help.

Run:  python tools/diag_fast_spikes_resolution.py
      python tools/diag_fast_spikes_resolution.py --window 90 --thresholds 0.015 0.02 0.03
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIN_NS = 60 * 1_000_000_000


def first_close_after(ts, close, t0_ts, step_min):
    """Close at the first B-minute boundary strictly after t0_ts (wall-clock aligned)."""
    step = step_min * MIN_NS
    nb = ((t0_ts // step) + 1) * step           # next boundary > t0 (epoch-aligned = :00/:15/:30)
    j = int(np.searchsorted(ts, nb, side="left"))
    if j >= len(ts):
        return None, None
    return close[j], j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/eth_1m_data.csv")
    ap.add_argument("--window", type=int, default=120, help="max minutes for a spike to complete")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.015, 0.02, 0.03])
    ap.add_argument("--bars", type=int, nargs="+", default=[60, 30, 15])
    args = ap.parse_args()

    df = pd.read_csv(args.file, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    ts = df["datetime"].values.astype("datetime64[ns]").astype(np.int64)
    close = df["close"].values.astype(float)
    n = len(df)
    span_days = (ts[-1] - ts[0]) / (MIN_NS * 60 * 24)

    print("=" * 92)
    print("  FAST-SPIKE RESOLUTION CEILING — ETH 1-minute  (model-free)")
    print(f"  {df['datetime'].iloc[0]} .. {df['datetime'].iloc[-1]}  ({n:,} min, {span_days:.0f} days)")
    print(f"  spike = up-move >= THRESH within <= {args.window} min (non-overlapping) | act on closes: {args.bars} min")
    print("=" * 92)
    print(f"  {'thr':>5} {'#spk':>5} {'medMove':>8} {'medMinToPk':>11} {'<30m':>6} "
          + "".join(f"reach@{b}m".rjust(11) for b in args.bars))
    print("  " + "-" * 88)

    win_ns = args.window * MIN_NS
    for thr in args.thresholds:
        spikes = []
        i = 0
        while i < n - 1:
            j_end = int(np.searchsorted(ts, ts[i] + win_ns, side="right"))
            seg = close[i:j_end]
            if len(seg) < 2:
                i += 1
                continue
            k = int(np.argmax(seg))
            gain = seg[k] / close[i] - 1.0
            if gain >= thr and k > 0:
                spikes.append((i, i + k))
                i = i + k + 1
            else:
                i += 1

        if not spikes:
            print(f"  {thr:>5.1%} {0:>5}  (none)")
            continue

        totals, mins_to_pk, reach = [], [], {b: [] for b in args.bars}
        for i0, ip in spikes:
            p0, ppk = close[i0], close[ip]
            total = ppk / p0 - 1.0
            totals.append(total)
            mins_to_pk.append((ts[ip] - ts[i0]) / MIN_NS)
            for b in args.bars:
                pc, j = first_close_after(ts, close, ts[i0], b)
                if pc is None or j >= ip:          # boundary at/after the peak -> nothing left
                    reach[b].append(0.0)
                else:
                    r = (ppk / pc - 1.0)
                    reach[b].append(max(0.0, min(r, total)) / total if total > 0 else 0.0)

        totals = np.array(totals)
        w = totals / totals.sum()                  # value-weight by move size
        vw = {b: float(np.dot(w, reach[b])) for b in args.bars}
        frac_fast = float(np.mean(np.array(mins_to_pk) < 30)) * 100
        print(f"  {thr:>5.1%} {len(spikes):>5} {np.median(totals):>7.2%} "
              f"{np.median(mins_to_pk):>9.0f}m {frac_fast:>5.0f}% "
              + "".join(f"{vw[b]:>9.0%}" for b in args.bars))

    print("=" * 92)
    print("  reach@Bm = value-weighted fraction of fast-spike upside still capturable if you can")
    print("             only enter on B-minute closes. 100% = fully catchable; low = spent before you act.")
    print("  READ: reach@60m is the HOURLY ceiling. (reach@15m - reach@60m) = what sub-hourly bars buy.")
    print("        High reach@60m -> it's MODEL latency, not resolution (sub-hourly won't help).")
    print("        Low reach@60m AND big 15m gain -> resolution IS the wall; Ein/Eli justified.")


if __name__ == "__main__":
    main()
