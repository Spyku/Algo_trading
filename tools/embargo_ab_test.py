"""
embargo_ab_test.py — A/B test: FAYE backtest WITH embargo vs WITHOUT embargo
================================================================================

Updated 2026-05-31 for FAYE.

Purpose
-------
Quantify how much the embargo asymmetry contributes to the LIVE vs BACKTEST
divergence.

FAYE backtest (Mode V refine + Mode T signal cache + Mode D holdout) uses
`train_end = i - horizon` (embargo = horizon hours) to prevent label-overlap
leakage in walk-forward evaluation. Live trader uses `train_end = i - 1`
(no embargo) because labels for the recent `horizon` hours are not yet known
at inference time — there's no leakage risk.

These two training-window choices produce different models, which produce
different predictions. This script measures HOW different by comparing
hour-by-hour signals from Mode T in both configurations.

Implementation (2026-05-31 patch)
---------------------------------
FAYE source now reads `FAYE_EMBARGO_OVERRIDE` env var in 4 walk-forward
training cutoff lines:
  - generate_signals          (Mode T signal cache, Mode V Step 1/3 backtests)
  - train_signals_ed          (Ed legacy backtest path)
  - _deku_eval_with_pruning_inner  (K=5 inner loop — Mode D holdout, Mode V refine)
  - _deku_eval_inner          (single-seed inner for non-K=5 callers)

Default behavior (env unset or var = horizon): same as before (honest backtest).
Set FAYE_EMBARGO_OVERRIDE=0 to force live-equivalent training cutoff.

Isolation
---------
1. Each mode writes to its own FAYE_MODELS_DIR / FAYE_CONFIG_DIR
   (`models_embargo_ab_baseline/` vs `models_embargo_ab_no_embargo/`).
2. Uses --no-persist + a frozen data snapshot when available.
3. Live trader is unaffected; can continue running.

Run modes
---------
  --mode=baseline    Run Mode T on ETH 5h with CURRENT embargo (= horizon). ~70 min.
  --mode=no_embargo  Run Mode T with embargo=0 (live-equivalent). ~70 min.
  --mode=both        Run both sequentially (~2.5h total) and diff results.

Recommended: launch with --mode=both on Laptop. Cleanly separated from
Desktop's trader + any in-flight HRST.

Output
------
output/embargo_ab_<timestamp>/
  - baseline_subprocess.log
  - no_embargo_subprocess.log
  - baseline_signals.csv         per-hour signals from baseline run
  - no_embargo_signals.csv       per-hour signals from no-embargo run
  - signal_diff.csv              hour-by-hour comparison
  - report.md                    summary

Decision criterion
------------------
match_rate ≥ 95% → embargo asymmetry contributes little. Trust backtests.
match_rate 80-95% → modest effect. Apply ~5pp discount to backtest projections.
match_rate < 80% → MAJOR effect. Backtest is significantly inflating live
  performance. Step 6 engine refactor (unify backtest + live signal paths)
  becomes urgent.
"""

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Force UTF-8 stdout for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

SNAPSHOT_DIR = "data/_reliability_hrst_snapshot_laptop_20260522_0139"
LIVE_PRODUCTION_CSV = "models/crypto_ed_production.csv"
LIVE_REGIME_CONFIG = "config/regime_config_ed.json"
FAYE_SCRIPT = "crypto_trading_system_faye.py"


# ============================================================================
# Subprocess runner
# ============================================================================
def run_subprocess(mode, out_dir, log_path):
    """Run Mode T ETH 5h with either baseline embargo or no-embargo.

    The ONLY difference between the two modes is the FAYE_EMBARGO_OVERRIDE
    env var. Both invoke crypto_trading_system_faye.py with identical args.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONWARNINGS"] = "ignore:X does not have valid feature names:UserWarning"

    # FAYE re-execs itself with `python -W ignore` via os.execv (warning
    # suppression, gated by the _FAYE_WARNINGS_BAKED sentinel). On Windows
    # os.execv does NOT replace the process — it spawns a DETACHED child and
    # the original process exits immediately (code 0). subprocess.run() then
    # returns in ~0.2s having captured only the 2 early setup lines, while the
    # real ~70 min Mode T run continues orphaned and uncaptured (parse → 0
    # signals). Pre-setting the sentinel makes FAYE skip the re-exec and run
    # the work in-process, so subprocess.run waits for it and captures stdout.
    # Warning-suppression layers 2-4 inside FAYE still apply.
    env["_FAYE_WARNINGS_BAKED"] = "1"

    if Path(SNAPSHOT_DIR).exists():
        env["V2_DATA_SNAPSHOT"] = SNAPSHOT_DIR
        print(f"  Using snapshot: {SNAPSHOT_DIR}")
    else:
        print(f"  No snapshot — using live data with --no-data-update")

    # Per-mode output dirs (FAYE honors FAYE_MODELS_DIR / FAYE_CONFIG_DIR)
    env["FAYE_MODELS_DIR"] = f"models_embargo_ab_{mode}"
    env["FAYE_CONFIG_DIR"] = f"config_embargo_ab_{mode}"
    os.makedirs(env["FAYE_MODELS_DIR"], exist_ok=True)
    os.makedirs(env["FAYE_CONFIG_DIR"], exist_ok=True)

    # The actual A/B switch:
    if mode == "no_embargo":
        env["FAYE_EMBARGO_OVERRIDE"] = "0"
        print("  FAYE_EMBARGO_OVERRIDE=0 (live-equivalent training cutoff)")
    elif mode == "baseline":
        env.pop("FAYE_EMBARGO_OVERRIDE", None)
        print("  FAYE_EMBARGO_OVERRIDE not set (default = horizon, honest backtest)")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Seed the per-mode dirs from LIVE so Mode T has the production CSV +
    # regime config to read (it doesn't write to these since --no-persist).
    if Path(LIVE_REGIME_CONFIG).exists():
        shutil.copy(LIVE_REGIME_CONFIG,
                    Path(env["FAYE_CONFIG_DIR"]) / "regime_config_faye.json")
    if Path(LIVE_PRODUCTION_CSV).exists():
        shutil.copy(LIVE_PRODUCTION_CSV,
                    Path(env["FAYE_MODELS_DIR"]) / "crypto_faye_production.csv")

    cmd = [
        sys.executable,
        FAYE_SCRIPT,
        "T", "ETH",
        "--replay", "1440",
        "--no-persist",
        "--no-data-update",
    ]
    print(f"  Launching: {' '.join(cmd)}")
    print(f"  Subprocess log: {log_path}")

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"  Subprocess finished in {elapsed:.1f}s (exit={proc.returncode})")

    return {"elapsed_s": elapsed, "exit_code": proc.returncode, "log": str(log_path)}


# ============================================================================
# Parse signals from subprocess log
# ============================================================================
def parse_signals(log_path):
    """Extract '[N] 2026-...: BUY/SELL/HOLD (conf%) | price=$X' lines."""
    signals = []
    current_horizon = None
    sig_re = re.compile(
        r"\[\s*\d+\]\s+(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}):\s+(?P<action>BUY|SELL|HOLD)\s+\((?P<conf>[\d.]+)%\)\s+\|\s+price=\$([\d,.]+)"
    )
    horizon_re = re.compile(r"Generating (\d+)h-ahead signals")
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            mh = horizon_re.search(line)
            if mh:
                current_horizon = int(mh.group(1))
                continue
            m = sig_re.search(line)
            if m:
                signals.append({
                    "timestamp": m.group("ts"),
                    "action": m.group("action"),
                    "confidence": float(m.group("conf")),
                    "price": float(m.group(4).replace(",", "")),
                    "horizon": current_horizon,
                })
    return signals


# ============================================================================
# Compare two signal streams
# ============================================================================
def diff_signals(baseline_signals, no_embargo_signals, out_csv):
    """Match on (timestamp, horizon) and tally action / confidence differences."""
    def key(s):
        return (s["timestamp"], s["horizon"])

    a = {key(s): s for s in baseline_signals}
    b = {key(s): s for s in no_embargo_signals}
    common = sorted(set(a) & set(b))

    rows = []
    n_match = n_diff = 0
    abs_conf_diffs = []
    for k in common:
        sa, sb = a[k], b[k]
        match = sa["action"] == sb["action"]
        if match:
            n_match += 1
        else:
            n_diff += 1
        abs_conf_diffs.append(abs(sa["confidence"] - sb["confidence"]))
        rows.append({
            "timestamp": k[0],
            "horizon": k[1],
            "baseline_action": sa["action"],
            "baseline_conf": sa["confidence"],
            "no_embargo_action": sb["action"],
            "no_embargo_conf": sb["confidence"],
            "match": "MATCH" if match else "DIFF",
            "conf_delta": round(sb["confidence"] - sa["confidence"], 1),
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    total = n_match + n_diff
    if total == 0:
        return {"error": "no common signals"}

    return {
        "common_signals": total,
        "match": n_match,
        "diff": n_diff,
        "match_rate_pct": round(100 * n_match / total, 1),
        "avg_abs_conf_delta": round(sum(abs_conf_diffs) / len(abs_conf_diffs), 2),
        "max_abs_conf_delta": round(max(abs_conf_diffs), 1),
        "diff_csv": str(out_csv),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "no_embargo", "both"], default="both")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"output/embargo_ab_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Embargo A/B Test (FAYE) ===")
    print(f"Output dir: {out_dir}")
    print(f"Mode: {args.mode}")
    print(f"Engine: {FAYE_SCRIPT}")

    results = {"mode": args.mode, "started": datetime.now().isoformat()}

    if args.mode in ("baseline", "both"):
        print("\n--- Phase 1: BASELINE (FAYE_EMBARGO_OVERRIDE unset, embargo = horizon) ---")
        baseline_log = out_dir / "baseline_subprocess.log"
        results["baseline"] = run_subprocess("baseline", out_dir, baseline_log)
        baseline_sigs = parse_signals(baseline_log)
        baseline_csv = out_dir / "baseline_signals.csv"
        pd.DataFrame(baseline_sigs).to_csv(baseline_csv, index=False)
        results["baseline"]["signals_parsed"] = len(baseline_sigs)
        print(f"  Parsed {len(baseline_sigs)} baseline signals → {baseline_csv.name}")

    if args.mode in ("no_embargo", "both"):
        print("\n--- Phase 2: NO_EMBARGO (FAYE_EMBARGO_OVERRIDE=0, live-equivalent) ---")
        no_emb_log = out_dir / "no_embargo_subprocess.log"
        results["no_embargo"] = run_subprocess("no_embargo", out_dir, no_emb_log)
        no_emb_sigs = parse_signals(no_emb_log)
        no_emb_csv = out_dir / "no_embargo_signals.csv"
        pd.DataFrame(no_emb_sigs).to_csv(no_emb_csv, index=False)
        results["no_embargo"]["signals_parsed"] = len(no_emb_sigs)
        print(f"  Parsed {len(no_emb_sigs)} no_embargo signals → {no_emb_csv.name}")

    # Diff
    if args.mode == "both":
        print("\n--- Phase 3: DIFF baseline vs no_embargo ---")
        diff = diff_signals(baseline_sigs, no_emb_sigs, out_dir / "signal_diff.csv")
        results["diff"] = diff
        print(f"  Common signals: {diff.get('common_signals')}")
        print(f"  MATCH:           {diff.get('match')}")
        print(f"  DIFF:            {diff.get('diff')}")
        print(f"  Match rate:      {diff.get('match_rate_pct')}%")
        print(f"  Avg |Δ conf|:    {diff.get('avg_abs_conf_delta')}")
        print(f"  Max |Δ conf|:    {diff.get('max_abs_conf_delta')}")

    # Report
    report_path = out_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Embargo A/B Test Report (FAYE)\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n")
        f.write(f"**Mode**: {args.mode}\n\n")
        f.write("## Subprocess results\n\n")
        if "baseline" in results:
            r = results["baseline"]
            f.write(f"### Baseline (embargo = horizon, default)\n")
            f.write(f"- Elapsed: {r.get('elapsed_s', 0):.1f}s\n")
            f.write(f"- Exit code: {r.get('exit_code')}\n")
            f.write(f"- Signals parsed: {r.get('signals_parsed', 0)}\n")
            f.write(f"- Log: `{Path(r['log']).name}`\n\n")
        if "no_embargo" in results:
            r = results["no_embargo"]
            f.write(f"### No-embargo (FAYE_EMBARGO_OVERRIDE=0)\n")
            f.write(f"- Elapsed: {r.get('elapsed_s', 0):.1f}s\n")
            f.write(f"- Exit code: {r.get('exit_code')}\n")
            f.write(f"- Signals parsed: {r.get('signals_parsed', 0)}\n")
            f.write(f"- Log: `{Path(r['log']).name}`\n\n")
        if "diff" in results:
            d = results["diff"]
            f.write("## Diff: baseline vs no_embargo\n\n")
            f.write(f"- Common signals compared: **{d.get('common_signals')}**\n")
            f.write(f"- Match: **{d.get('match')}**\n")
            f.write(f"- Differ: **{d.get('diff')}**\n")
            f.write(f"- Match rate: **{d.get('match_rate_pct')}%**\n")
            f.write(f"- Avg |Δ confidence|: {d.get('avg_abs_conf_delta')}\n")
            f.write(f"- Max |Δ confidence|: {d.get('max_abs_conf_delta')}\n\n")
            mr = d.get("match_rate_pct")
            if mr is None:
                # No common (timestamp, horizon) keys — diff could not be computed.
                # Do NOT default to 100 (that falsely reads as "minimal effect").
                # The embargo shift moves the walk-forward eval grid, so the
                # every-50th sampled signal lines never align between runs.
                f.write("### Verdict\n\n**INCONCLUSIVE** — match rate could not be computed "
                        "(no common (timestamp, horizon) keys between the two runs).\n")
                f.write("The embargo change shifts the walk-forward evaluation grid, so the "
                        "sparsely-sampled signal lines never align. Compare the **Mode T REF** "
                        "returns in the two subprocess logs instead "
                        "(`grep 'baselines V0' *_subprocess.log`).\n")
            elif mr >= 95:
                f.write("### Verdict\n\n**EMBARGO HAS MINIMAL EFFECT** — match rate ≥95%.\n")
                f.write("Embargo asymmetry is NOT a major source of LIVE-vs-BACKTEST divergence.\n")
                f.write("Trust FAYE backtest numbers; Step 6 refactor is low-urgency.\n")
            elif mr >= 80:
                f.write("### Verdict\n\n**EMBARGO HAS MODEST EFFECT** — match rate 80-95%.\n")
                f.write("Apply a ~5pp discount when projecting live performance from FAYE backtest returns.\n")
                f.write("Step 6 refactor would close the gap entirely.\n")
            else:
                f.write("### Verdict\n\n**EMBARGO HAS MAJOR EFFECT** — match rate <80%.\n")
                f.write("Embargo asymmetry is a SIGNIFICANT cause of LIVE-vs-BACKTEST divergence.\n")
                f.write("**FAYE backtest returns are likely inflated.** Consider:\n")
                f.write("- (a) Switch FAYE to FAYE_EMBARGO_OVERRIDE=0 for promotion-gate runs (matches live)\n")
                f.write("- (b) Prioritize Step 6 engine refactor (unify backtest + live signal paths)\n")
                f.write("- (c) Document the gap so backtest results are interpreted accordingly\n")

    # Save JSON
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nReport: {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
