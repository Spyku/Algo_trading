"""
afternoon_run.py — Statistically robust counterfactual for the cache-fix question
=================================================================================

Run this on Laptop this afternoon. Takes ~30-40 min.

What it does:
  Phase 1 — Pre-flight: verify env (no env-var leak, archeology present)
  Phase 2 — Wider counterfactual: hourly inference over the FULL clean PIT
            window (2026-05-07 → today) with the CURRENT G_narrow live config.
            ~500 hours of inference, ~30 min runtime.
  Phase 3 — Statistical analysis: bootstrap 95% confidence intervals on
            total return and win rate (which the 5-day, 4-trade smoke result
            could not produce). Sub-window breakdown (week-by-week) to
            detect regime-dependent effects.
  Phase 4 — Verdict: clear human-readable interpretation.

What you do:
  1. Open a FRESH PowerShell window on Laptop (no leftover env vars)
  2. cd to engine dir + activate venv
  3. Run:  python tools/afternoon_run.py
  4. Walk away for ~35 min
  5. Paste the final "VERDICT" block back to me when done

What it does NOT do:
  - Does NOT touch the Desktop trader. Read-only, runs entirely on Laptop.
  - Does NOT promote anything. No config changes.
  - Does NOT need internet. All data is in data/_archeology and data/.

Output files (under output/):
  - counterfactual_signals_<ts>.csv    — every hour's signal
  - counterfactual_trades_<ts>.csv     — closed counterfactual trades
  - actual_trades_<ts>.csv             — actual signal_log trades (same window)
  - afternoon_summary_<ts>.md          — human-readable report (paste this)
"""

import os
import sys
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

# ============================================================================
# Pre-flight checks
# ============================================================================
def preflight():
    print("=" * 78)
    print("  AFTERNOON RUN — pre-flight checks")
    print("=" * 78)

    fail = False

    # 1. Working directory
    print(f"\n  [1] Working dir: {Path.cwd()}")
    if Path.cwd() != REPO_ROOT:
        print(f"      FAIL — expected {REPO_ROOT}")
        fail = True
    else:
        print(f"      OK")

    # 2. Env var leak check
    print(f"\n  [2] Env var leak check:")
    leaky = []
    for v in ("V2_DATA_SNAPSHOT", "H_STRICT_MODELS_DIR", "H_STRICT_CONFIG_DIR",
              "H75_WIDE_MODELS_DIR", "H75_WIDE_CONFIG_DIR",
              "G_NARROW_MODELS_DIR", "G_NARROW_CONFIG_DIR"):
        val = os.environ.get(v)
        if val:
            leaky.append((v, val))
    if leaky:
        print(f"      WARN — env vars set in this shell (will be cleared per-subprocess):")
        for k, v in leaky:
            print(f"        {k}={v}")
        print(f"      counterfactual_backtest.py will re-set V2_DATA_SNAPSHOT to _pit_workdir/")
    else:
        print(f"      OK — clean shell")

    # 3. Counterfactual script present
    print(f"\n  [3] tools/counterfactual_backtest.py:")
    script_path = REPO_ROOT / "tools" / "counterfactual_backtest.py"
    if not script_path.exists():
        print(f"      FAIL — not found")
        fail = True
    else:
        sz = script_path.stat().st_size
        print(f"      OK ({sz / 1024:.0f} KB)")

    # 4. Archeology coverage
    print(f"\n  [4] Archeology coverage:")
    arch_root = REPO_ROOT / "data" / "_archeology"
    if not arch_root.exists():
        print(f"      FAIL — {arch_root} not found")
        fail = True
    else:
        needed = ["eth_hourly_data.csv", "derivatives_eth.csv", "onchain_eth.csv",
                  "macro_daily.csv", "cross_asset.csv", "fear_greed.csv",
                  "stablecoin_flows.csv", "btc_hourly_data.csv"]
        missing = []
        for fn in needed:
            sub = arch_root / fn
            if not sub.exists() or not any(sub.iterdir()):
                missing.append(fn)
        if missing:
            print(f"      WARN — missing or empty archeology for: {missing}")
            print(f"      counterfactual will fall back to current data/ for those")
        else:
            print(f"      OK — all 8 files have archeology")

    # 5. Production CSV + regime config
    print(f"\n  [5] Production config:")
    prod = REPO_ROOT / "models" / "crypto_ed_production.csv"
    cfg = REPO_ROOT / "config" / "regime_config_ed.json"
    for p, label in [(prod, "models/crypto_ed_production.csv"),
                     (cfg, "config/regime_config_ed.json")]:
        if p.exists():
            mt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
            print(f"      OK  {label}  mtime={mt}")
        else:
            print(f"      FAIL — {label} missing")
            fail = True

    # 6. Signal log present (needed for actual comparison)
    print(f"\n  [6] Signal log:")
    sl = REPO_ROOT / "config" / "signal_log.csv"
    if sl.exists():
        mt = datetime.fromtimestamp(sl.stat().st_mtime, tz=timezone.utc).isoformat()
        print(f"      OK  config/signal_log.csv  mtime={mt}")
    else:
        print(f"      WARN — signal_log.csv missing, actual-trade comparison will be empty")

    print()
    if fail:
        print("  PRE-FLIGHT FAILED — fix the issues above before continuing")
        sys.exit(1)
    print("  PRE-FLIGHT PASSED")
    print()


# ============================================================================
# Run counterfactual backtest as subprocess
# ============================================================================
def run_counterfactual(start_iso: str, end_iso: str):
    print("=" * 78)
    print("  PHASE 2 — Wider counterfactual backtest")
    print("=" * 78)
    print(f"\n  Window: {start_iso} -> {end_iso}")
    print(f"  Step:   1h (hourly)")
    print(f"  Est:    ~30-40 min on Laptop\n")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "counterfactual_backtest.py"),
        "--start", start_iso,
        "--end", end_iso,
        "--hourly-step", "1",
    ]

    t0 = time.time()
    # Stream output so user sees progress
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        bufsize=1,  # line-buffered
    )
    last_output_lines = []
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            print(f"  {line.rstrip()}")
            last_output_lines.append(line.rstrip())
            if len(last_output_lines) > 200:
                last_output_lines = last_output_lines[-200:]
    rc = proc.wait()
    elapsed = (time.time() - t0) / 60.0
    print(f"\n  Counterfactual done in {elapsed:.1f} min (exit code {rc})")
    return rc, last_output_lines


# ============================================================================
# Locate newest output files
# ============================================================================
def find_latest_outputs():
    out = REPO_ROOT / "output"
    cf_signals = sorted(out.glob("counterfactual_signals_*.csv"), key=lambda p: p.stat().st_mtime)
    cf_trades = sorted(out.glob("counterfactual_trades_*.csv"), key=lambda p: p.stat().st_mtime)
    actual_trades = sorted(out.glob("actual_trades_*.csv"), key=lambda p: p.stat().st_mtime)
    return (
        cf_signals[-1] if cf_signals else None,
        cf_trades[-1] if cf_trades else None,
        actual_trades[-1] if actual_trades else None,
    )


# ============================================================================
# Bootstrap confidence intervals
# ============================================================================
def bootstrap_ci(values, n_iter=10000, alpha=0.05, agg="mean"):
    """Bootstrap CI for a statistic on `values`.

    agg = "mean" -> mean return per trade
    agg = "compound" -> compounded total return: prod(1+r) - 1
    agg = "wr" -> win rate (values must be 0/1)
    """
    import numpy as np
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (None, None, None)
    rng = np.random.default_rng(42)
    stats = []
    for _ in range(n_iter):
        sample = rng.choice(arr, size=n, replace=True)
        if agg == "mean":
            stats.append(sample.mean())
        elif agg == "compound":
            stats.append(np.prod(1 + sample) - 1)
        elif agg == "wr":
            stats.append(sample.mean() * 100)
    stats = np.sort(stats)
    lo = float(stats[int(n_iter * alpha / 2)])
    hi = float(stats[int(n_iter * (1 - alpha / 2))])
    return (lo, hi, n)


# ============================================================================
# Sub-window breakdown
# ============================================================================
def weekly_breakdown(trades_df):
    """Group trades by ISO week of entry_dt. Return DataFrame with WR/return per week."""
    import pandas as pd
    if len(trades_df) == 0:
        return pd.DataFrame()
    df = trades_df.copy()
    df["entry_dt"] = pd.to_datetime(df["entry_dt"], errors="coerce")
    df["week"] = df["entry_dt"].dt.strftime("%Y-W%V")
    weeks = []
    for w, sub in df.groupby("week"):
        weeks.append({
            "week": w,
            "n_trades": len(sub),
            "wr_pct": 100 * sub["win"].sum() / len(sub),
            "total_return_pct": 100 * ((sub["net_return"] + 1).prod() - 1),
            "avg_net_pct": 100 * sub["net_return"].mean(),
        })
    return pd.DataFrame(weeks)


# ============================================================================
# Phase 3 — statistical analysis + verdict
# ============================================================================
def analyze_and_report(cf_signals_path, cf_trades_path, actual_trades_path):
    import pandas as pd
    import numpy as np

    print("\n" + "=" * 78)
    print("  PHASE 3 — Statistical analysis")
    print("=" * 78)

    cf_trades = pd.read_csv(cf_trades_path) if (cf_trades_path and cf_trades_path.exists()) else pd.DataFrame()
    actual_trades = pd.read_csv(actual_trades_path) if (actual_trades_path and actual_trades_path.exists()) else pd.DataFrame()
    signals = pd.read_csv(cf_signals_path) if (cf_signals_path and cf_signals_path.exists()) else pd.DataFrame()

    print(f"\n  Loaded:")
    print(f"    Counterfactual trades:  {len(cf_trades)}")
    print(f"    Actual trades:          {len(actual_trades)}")
    print(f"    Signal rows:            {len(signals)}")

    # Summary stats per condition with bootstrap CI
    def _summary(trades, label):
        if len(trades) == 0:
            return {
                "label": label, "n": 0,
                "wr": None, "wr_ci": (None, None),
                "compound": None, "compound_ci": (None, None),
                "avg": None, "avg_ci": (None, None),
            }
        wins = trades["win"].astype(int).values
        rets = trades["net_return"].astype(float).values
        wr = 100 * wins.mean()
        compound = (rets + 1).prod() - 1
        avg = rets.mean()
        wr_lo, wr_hi, _ = bootstrap_ci(wins, agg="wr")
        c_lo, c_hi, _ = bootstrap_ci(rets, agg="compound")
        a_lo, a_hi, _ = bootstrap_ci(rets, agg="mean")
        return {
            "label": label, "n": len(trades),
            "wr": wr, "wr_ci": (wr_lo, wr_hi),
            "compound": compound, "compound_ci": (c_lo, c_hi),
            "avg": avg, "avg_ci": (a_lo, a_hi),
        }

    s_actual = _summary(actual_trades, "Actual (broken cache, signal_log)")
    s_cf = _summary(cf_trades, "Counterfactual (cache fix applied retroactively)")

    print("\n  ----- Headline stats (with bootstrap 95% CIs) -----\n")
    for s in [s_actual, s_cf]:
        print(f"    {s['label']}:")
        print(f"      Trades: {s['n']}")
        if s["n"] == 0:
            print()
            continue
        print(f"      WR:           {s['wr']:.1f}%   95% CI: [{s['wr_ci'][0]:.1f}%, {s['wr_ci'][1]:.1f}%]")
        print(f"      Compound ret: {s['compound']*100:+.2f}%  95% CI: [{s['compound_ci'][0]*100:+.2f}%, {s['compound_ci'][1]*100:+.2f}%]")
        print(f"      Avg/trade:    {s['avg']*100:+.3f}% 95% CI: [{s['avg_ci'][0]*100:+.3f}%, {s['avg_ci'][1]*100:+.3f}%]")
        print()

    # Delta — does counterfactual significantly beat actual?
    print("  ----- Delta (counterfactual - actual) -----\n")
    if s_actual["n"] and s_cf["n"]:
        print(f"      Trade count: {s_cf['n'] - s_actual['n']:+d}")
        print(f"      WR delta:    {s_cf['wr'] - s_actual['wr']:+.1f}pp")
        print(f"      Return delta: {(s_cf['compound'] - s_actual['compound'])*100:+.2f}pp")
        print(f"      Avg/trade delta: {(s_cf['avg'] - s_actual['avg'])*100:+.3f}pp")

        # Simple bootstrap of the delta itself
        rets_a = actual_trades["net_return"].astype(float).values
        rets_c = cf_trades["net_return"].astype(float).values
        rng = np.random.default_rng(42)
        delta_dist = []
        for _ in range(10000):
            sa = rng.choice(rets_a, size=len(rets_a), replace=True)
            sc = rng.choice(rets_c, size=len(rets_c), replace=True)
            comp_a = (sa + 1).prod() - 1
            comp_c = (sc + 1).prod() - 1
            delta_dist.append(comp_c - comp_a)
        delta_dist = np.sort(delta_dist)
        d_lo = delta_dist[250]
        d_hi = delta_dist[9750]
        prob_pos = (np.asarray(delta_dist) > 0).mean() * 100
        print(f"\n      Bootstrap 95% CI on RETURN DELTA: [{d_lo*100:+.2f}pp, {d_hi*100:+.2f}pp]")
        print(f"      P(counterfactual > actual): {prob_pos:.1f}%")

    # Weekly breakdown
    print("\n  ----- Weekly breakdown -----\n")
    print("    Counterfactual:")
    wb_cf = weekly_breakdown(cf_trades)
    if len(wb_cf):
        for _, row in wb_cf.iterrows():
            print(f"      {row['week']}  n={row['n_trades']:>2}  WR={row['wr_pct']:5.1f}%  return={row['total_return_pct']:+.2f}%  avg={row['avg_net_pct']:+.3f}%")
    else:
        print("      (no trades)")

    print("\n    Actual:")
    wb_actual = weekly_breakdown(actual_trades)
    if len(wb_actual):
        for _, row in wb_actual.iterrows():
            print(f"      {row['week']}  n={row['n_trades']:>2}  WR={row['wr_pct']:5.1f}%  return={row['total_return_pct']:+.2f}%  avg={row['avg_net_pct']:+.3f}%")
    else:
        print("      (no trades)")

    return s_actual, s_cf


# ============================================================================
# Phase 4 — Verdict
# ============================================================================
def print_verdict(s_actual, s_cf):
    print("\n" + "=" * 78)
    print("  PHASE 4 — VERDICT (paste this back to assistant)")
    print("=" * 78)

    if s_actual["n"] == 0 or s_cf["n"] == 0:
        print("\n  NO VERDICT POSSIBLE — one or both condition has 0 trades.")
        print("  Check that signal_log.csv covers the window and counterfactual ran cleanly.")
        return

    n_min = min(s_actual["n"], s_cf["n"])

    # Sample-size category
    if n_min < 10:
        size_label = "STILL TOO SMALL — sample below statistical significance threshold"
    elif n_min < 25:
        size_label = "BORDERLINE — directional signal possible but treat numbers as suggestive"
    else:
        size_label = "STATISTICALLY MEANINGFUL — enough trades to trust the delta"

    print(f"\n  Sample size:    actual={s_actual['n']}, counterfactual={s_cf['n']}  ->  {size_label}")

    # Direction
    return_delta_pp = (s_cf["compound"] - s_actual["compound"]) * 100
    wr_delta = s_cf["wr"] - s_actual["wr"]

    if return_delta_pp > 1.0:
        direction = "COUNTERFACTUAL OUTPERFORMS"
    elif return_delta_pp > 0.2:
        direction = "COUNTERFACTUAL SLIGHTLY OUTPERFORMS"
    elif return_delta_pp > -0.2:
        direction = "ESSENTIALLY TIED"
    elif return_delta_pp > -1.0:
        direction = "ACTUAL SLIGHTLY OUTPERFORMS"
    else:
        direction = "ACTUAL OUTPERFORMS"

    print(f"  Return delta:   {return_delta_pp:+.2f}pp  ->  {direction}")
    print(f"  WR delta:       {wr_delta:+.1f}pp  (note: lower WR + higher return = more selective)")

    print(f"\n  Top-line interpretation:")
    if return_delta_pp > 0.5 and n_min >= 25:
        print(f"    Cache-fix delivers measurable improvement in P&L over 3 weeks of data.")
        print(f"    Direction matches the original hypothesis (TODO 0527 root cause).")
    elif return_delta_pp > 0.5 and n_min < 25:
        print(f"    Cache-fix shows positive return delta in the right direction, but the")
        print(f"    sample is still small. Wait for 2-4 weeks of forward live data to confirm.")
    elif abs(return_delta_pp) <= 0.2:
        print(f"    Effectively no measurable difference. Either the cache bug was less")
        print(f"    impactful than expected, or this window's market conditions don't")
        print(f"    showcase the effect. Watch forward live data carefully.")
    else:
        print(f"    Counterfactual UNDERPERFORMS — unexpected. Possible causes:")
        print(f"    - Sample variance (most likely if n<25)")
        print(f"    - Bear-market window where being MORE selective hurts when downside")
        print(f"      catches you not-in-cash on a HOLD signal")
        print(f"    - Another bug we haven't found")
        print(f"    Inspect the trade logs by hand for the loss cases.")


# ============================================================================
# Main
# ============================================================================
def main():
    preflight()

    # Pick widest clean PIT window:
    # archeology covers May 7 onward; we go through ~1h before now
    start_iso = "2026-05-07T00:00"
    end_dt = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    end_iso = end_dt.strftime("%Y-%m-%dT%H:%M")

    rc, _ = run_counterfactual(start_iso, end_iso)
    if rc != 0:
        print(f"\n  Counterfactual subprocess exited with code {rc}. Inspect output above.")
        sys.exit(rc)

    cf_signals, cf_trades, actual_trades = find_latest_outputs()
    if cf_signals:
        print(f"\n  Latest output files:")
        print(f"    {cf_signals}")
        if cf_trades:
            print(f"    {cf_trades}")
        if actual_trades:
            print(f"    {actual_trades}")
    else:
        print(f"\n  WARN — no output files found under output/. Cannot do analysis.")
        sys.exit(1)

    s_actual, s_cf = analyze_and_report(cf_signals, cf_trades, actual_trades)
    print_verdict(s_actual, s_cf)

    # Save a markdown summary so user can paste it back easily
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = REPO_ROOT / "output" / f"afternoon_summary_{ts}.md"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"# Afternoon counterfactual run — {ts}\n\n")
            f.write(f"Window: `{start_iso}` -> `{end_iso}`\n\n")
            f.write(f"## Results\n\n")
            for s in [s_actual, s_cf]:
                f.write(f"### {s['label']}\n")
                f.write(f"- Trades: **{s['n']}**\n")
                if s["n"]:
                    f.write(f"- WR: **{s['wr']:.1f}%** (95% CI [{s['wr_ci'][0]:.1f}%, {s['wr_ci'][1]:.1f}%])\n")
                    f.write(f"- Compound return: **{s['compound']*100:+.2f}%** (95% CI [{s['compound_ci'][0]*100:+.2f}%, {s['compound_ci'][1]*100:+.2f}%])\n")
                    f.write(f"- Avg/trade: **{s['avg']*100:+.3f}%** (95% CI [{s['avg_ci'][0]*100:+.3f}%, {s['avg_ci'][1]*100:+.3f}%])\n")
                f.write("\n")
        print(f"\n  Markdown summary saved: {summary_path}")
        print(f"  Paste the contents of this file back to the assistant.")
    except Exception as e:
        print(f"\n  WARN — could not write summary: {e}")

    print("\n" + "=" * 78)
    print("  DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
