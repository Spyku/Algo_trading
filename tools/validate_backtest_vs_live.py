"""validate_backtest_vs_live.py — the tripwire that would have caught the 2026-06-29
backtest!=live divergence (and DID NOT exist then).

WHY: snapshot-replay tests live self-consistency; refit-replay uses live's FROZEN training
matrix (bypassing generate_signals' OWN window construction); the shadow compares
live-vs-core (both the live path). NONE compared the BACKTEST engine (generate_signals,
faye) to what the live trader actually emitted. A 1-2 row training-window edge mismatch
sat at 75% agreement, undetected, and tainted every config decision.

WHAT: run generate_signals (the backtest path, leakage-free since 2026-06-29) for the live
config's horizon(s) over the window covered by output/inference_snapshots.jsonl, and compare
its signal to what the live trader logged each hour. Exit 1 if agreement < --min (default 95).

This is a faithfulness gate, NOT a correctness gate -- run it after ANY change to the
training-window edge / inference-row selection / generate_signals, and before trusting a
backtest-based config decision (HRST / head-to-head).

Usage:
  python tools/validate_backtest_vs_live.py                 # ETH, live horizons, min 95%
  python tools/validate_backtest_vs_live.py --min 95 --asset ETH --replay 200
"""
import os
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
os.environ.setdefault('FAYE_CONFIG_DIR', 'config_faye_bt')
os.environ.pop('FAYE_FAITHFUL_WINDOW', None)      # rely on the leakage-free DEFAULT
os.environ.pop('FAYE_EMBARGO_OVERRIDE', None)
import sys, json, argparse
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'tools'))
os.chdir(HERE)
import pandas as pd
import bt_regime as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', default='ETH')
    ap.add_argument('--min', type=float, default=95.0, help='min %% agreement to PASS')
    ap.add_argument('--replay', type=int, default=200)
    ap.add_argument('--csv', default='models/crypto_ed_production.csv')
    ap.add_argument('--since', default=None,
                    help='Only compare live hours >= this UTC ISO ts. Default: the production '
                         'CSV mtime (= last promotion). Excludes pre-promotion hours, which were '
                         'logged by a DIFFERENT model so the current backtest legitimately '
                         'disagrees — the config-straddle false FAIL (54%%/59%% on 06-30/07-01).')
    ap.add_argument('--min-recent', type=int, default=8,
                    help='Min post-promotion hours needed to judge; below this, report WARMING UP '
                         '(PASS) instead of failing on a thin sample. Window self-fills in ~1 day.')
    ap.add_argument('--all-hours', action='store_true',
                    help='Disable the recent-only filter (compare the full window, incl. '
                         'pre-promotion hours). For deliberate cross-config forensics only.')
    a = ap.parse_args()

    snap_path = os.path.join(HERE, 'output', 'inference_snapshots.jsonl')
    if not os.path.exists(snap_path):
        print('WARN: no inference_snapshots.jsonl yet (trader needs to run) — cannot validate.')
        sys.exit(2)
    recs = [json.loads(l) for l in open(snap_path)]
    live_all = [r for r in recs if r['asset'] == a.asset]
    if not live_all:
        print(f'WARN: no {a.asset} snapshots — cannot validate.')
        sys.exit(2)
    # live horizons actually logged (the active config's bull/bear)
    horizons = sorted({r['horizon'] for r in live_all})
    start = min(r['inference_row_dt'] for r in live_all)
    print(f"{a.asset} live horizons in snapshots: {horizons}  (since {start[:16]})")

    # Recent-only boundary: the production CSV mtime = the last time the live model changed
    # (= last promotion). Comparing hours logged BEFORE that against the CURRENT backtest model
    # is apples-to-oranges (a different model produced them) -> the config-straddle false FAIL
    # (54%/59% seen 06-30/07-01, which flips the whole daily sanity verdict via [5]). os.path.
    # getmtime is a POSIX epoch -> pd.Timestamp(unit='s') gives tz-naive UTC, matching the UTC
    # inference_row_dt. --all-hours disables the filter for deliberate cross-config forensics.
    if a.all_hours:
        since_ts = None
        print("  [--all-hours] recent-only filter OFF — comparing the full window (may straddle a config change)")
    else:
        if a.since:
            since_ts = pd.Timestamp(a.since)
            src = '--since'
        else:
            since_ts = pd.Timestamp(os.path.getmtime(os.path.join(HERE, a.csv)), unit='s')
            src = 'prod-CSV mtime = last promotion'
        print(f"  recent-only boundary: >= {since_ts} UTC ({src}); pre-boundary hours excluded "
              f"(logged under a prior model).")

    overall_ok = overall_tot = 0
    worst = 100.0
    for h in horizons:
        live = {r['inference_row_dt']: r for r in live_all if r['horizon'] == h}
        sig = {str(pd.Timestamp(k)): v for k, v in B._gen(a.csv, h, a.replay).items()}
        common_all = set(live) & set(sig)
        common = sorted(k for k in common_all if since_ts is None or pd.Timestamp(k) >= since_ts)
        excluded = len(common_all) - len(common)
        if not common:
            print(f"  {h}h: 0 post-boundary hours to compare ({excluded} pre-boundary excluded)")
            continue
        ok = sum(1 for k in common if live[k]['signal'] == sig[k]['signal'])
        pct = 100.0 * ok / len(common)
        worst = min(worst, pct)
        overall_ok += ok; overall_tot += len(common)
        flips = [(k, live[k]['signal'], sig[k]['signal']) for k in common
                 if {live[k]['signal'], sig[k]['signal']} <= {'BUY', 'SELL'} and live[k]['signal'] != sig[k]['signal']]
        print(f"  {h}h: backtest<->live {ok}/{len(common)} = {pct:.1f}%  | real BUY<->SELL flips: {len(flips)}"
              f"  ({excluded} pre-boundary hrs excluded)")
        for f in flips[:5]:
            print(f"      {f[0]}  live={f[1]} backtest={f[2]}")

    # Thin-sample guard: right after a promotion there may be too few post-boundary hours to
    # judge -> report WARMING UP (PASS), never FAIL on noise. Self-fills within a day (24/hr).
    if not a.all_hours and overall_tot < a.min_recent:
        print(f"\nBACKTEST-vs-LIVE faithfulness: WARMING UP — only {overall_tot} post-promotion "
              f"hour(s) since {since_ts} UTC (< --min-recent={a.min_recent}); not enough to judge, "
              f"recovers within a day. -> PASS")
        sys.exit(0)

    overall = 100.0 * overall_ok / max(1, overall_tot)
    status = 'PASS' if worst >= a.min else 'FAIL'
    print(f"\nBACKTEST-vs-LIVE faithfulness: overall {overall:.1f}% | worst-horizon {worst:.1f}% "
          f"(threshold {a.min:.0f}%) -> {status}")
    print("(<95% = generate_signals does NOT predict the live trader -> a training-window/"
          "inference-path divergence; backtest config decisions are untrustworthy. Compares only "
          "post-promotion hours — see the boundary above; --all-hours to override.)")
    sys.exit(0 if worst >= a.min else 1)


if __name__ == '__main__':
    main()
