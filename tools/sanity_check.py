"""sanity_check.py — live-correctness gate (shadow match-rate + engine-vs-trader parity).

Built 2026-06-01. Two roles:
  * DAILY scheduled task (run_sanity_check.bat in Task Scheduler) — surfaces live drift.
  * PRE-PROMOTION gate — run before any config/model swap; exit!=0 means "look before
    you promote".

Checks (the first two are DETERMINISTIC & reproducible → they drive the verdict;
the third is an informational offline re-run that is NON-reproducible by design):
  1. SHADOW match-rate — config/shadow_signal_diff.csv (trader writes it every cycle):
     the live trader's signal vs compute_signal_core must agree. <99% = real codepath
     divergence -> FAIL.
  2. SNAPSHOT REPLAY — recompute (signal, confidence) from the trader's OWN frozen
     point-in-time intermediates (output/inference_snapshots.jsonl) and assert == logged.
     Pure bookkeeping on frozen numbers: immune to data revision, reproducible run-to-run,
     MUST be 100%. Any mismatch is a real logic bug -> FAIL. ~instant (no model fit).
  3. ENGINE-vs-TRADER parity — re-runs the engine over the recent signal_log. INFORMATIONAL
     ONLY: it rebuilds features from CURRENT data, so it is NON-reproducible (the tail-N
     window shifts every hour AND the freshest training rows get revised). Real BUY-vs-SELL
     flips here are almost always point-in-time data-revision artifacts, not bugs — they are
     listed so a human can eyeball them, but they NO LONGER set the verdict (checks 1+2 plus
     the real-time shadow already catch any genuine divergence deterministically).

Exit 0 = clean. Exit 1 = needs attention (shadow <99% OR a snapshot logic mismatch).
Parity flips do NOT change the exit code. Use as a gate.

Usage:
  python tools/sanity_check.py                 # shadow + snapshot + parity (30 samples, ~15 min)
  python tools/sanity_check.py --quick         # shadow + snapshot only (instant, deterministic)
  python tools/sanity_check.py --samples 15    # lighter parity
"""
import sys, os, glob, subprocess, argparse, time
ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENG)
import pandas as pd

SHADOW_MIN = 0.99


def _truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


SHADOW_RECENT_N = 48  # verdict window (hours) — reflects CURRENT correctness, not history


def shadow_check():
    f = os.path.join(ENG, "config", "shadow_signal_diff.csv")
    if not os.path.exists(f):
        return "WARN", "shadow_signal_diff.csv not found (trader writing it?)"
    d = pd.read_csv(f)
    if "match" not in d.columns or len(d) == 0:
        return "WARN", "shadow file empty / no 'match' column"
    recent = d.tail(SHADOW_RECENT_N)
    age_h = (time.time() - os.path.getmtime(f)) / 3600
    # ALERT-GAP FIX (2026-06-18): a shadow that can't even RUN (shadow_error set, e.g.
    # 'import_failed') is the MONITOR being DOWN — categorically different from a real
    # BUY/SELL mismatch. Previously errored rows had a blank 'match' → counted as False
    # → looked like a generic "0% match"; a fully-down monitor went unnamed (the 2.5-day
    # import_failed outage that fired no clear alert). Now we detect + NAME it explicitly.
    err = (recent["shadow_error"].astype(str).str.strip()
           if "shadow_error" in recent.columns else pd.Series([""] * len(recent)))
    errored = err.ne("") & err.str.lower().ne("nan")
    n_err = int(errored.sum())
    valid = recent[~errored.values]
    if n_err >= max(1, int(len(recent) * 0.5)) or len(valid) == 0:
        top = err[errored].value_counts()
        top_err = top.index[0] if len(top) else "?"
        return "FAIL", (f"SHADOW DOWN (monitor not running) - {n_err}/{len(recent)} recent rows "
                        f"errored ['{top_err}']. RESTART the trader. This is the MONITOR, not a "
                        f"confirmed trader bug - see [2] SNAPSHOT REPLAY for the live signal path. "
                        f"file age {age_h:.1f}h")
    m_valid = _truthy(valid["match"])
    recent_rate = m_valid.mean() if len(m_valid) else 0.0  # verdict driver (valid rows only)
    m_all = _truthy(d["match"]); rate = m_all.mean()        # all-time context
    msg = (f"recent-valid {len(m_valid)} {recent_rate*100:.1f}% match (verdict) | "
           f"{n_err} errored | all-time {int(m_all.sum())}/{len(m_all)} = {rate*100:.1f}% | file age {age_h:.1f}h")
    status = "PASS" if recent_rate >= SHADOW_MIN else "FAIL"
    if age_h > 6:
        msg += "  [!] stale — trader may not be writing"
    return status, msg


def parity_check(samples):
    print(f"  running engine-vs-trader parity ({samples} recent samples, ~{samples*0.5:.0f} min)...", flush=True)
    r = subprocess.run(
        [sys.executable, os.path.join(ENG, "tools", "validate_core_against_signal_log.py"),
         "--samples", str(samples), "--recent-only"],  # GPU (Rule 24): mirrors live device='gpu', collapses the -4.6 CPU conf bias -> ~97%
        cwd=ENG, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    csvs = glob.glob(os.path.join(ENG, "output", "core_validation_*.csv"))
    if not csvs:
        return "WARN", "validator produced no output CSV", []
    d = pd.read_csv(max(csvs, key=os.path.getmtime))
    ev = d[d["error"].isna()] if "error" in d.columns else d
    diffs = ev[~_truthy(ev["match"])] if "match" in ev.columns else ev.iloc[0:0]
    dirset = {"BUY", "SELL"}
    real = diffs[diffs["live_action"].isin(dirset) & diffs["core_action"].isin(dirset)
                 & (diffs["live_action"] != diffs["core_action"])]
    n_eval, n_diff, n_real = len(ev), len(diffs), len(real)
    flips = [f"{row['timestamp']} live={row['live_action']}({row['live_confidence']}) "
             f"core={row['core_action']}({row['core_confidence']})" for _, row in real.iterrows()]
    # Honest split of the non-flip diffs (2026-06-20): a HOLD<->directional disagreement
    # with a large |dConf| is NOT a near-gate "boundary" wobble — it's decisive drift from
    # the model refitting on later-settled current-period data (verified this session: device
    # only ~3pt, within-device deterministic, historical data frozen 0/185k cells). Surface it
    # instead of hiding it under "boundary"; parity is recent-only + non-reproducible by design.
    rest = diffs.drop(real.index)
    _dc = (pd.to_numeric(rest.get("core_confidence"), errors="coerce")
           - pd.to_numeric(rest.get("live_confidence"), errors="coerce")).abs()
    n_drift = int((_dc >= 10).sum())          # decisive HOLD<->dir (refit/data-settle drift)
    n_bound = int(len(rest) - n_drift)        # genuine near-gate wobble (|dConf|<10)
    msg = (f"{n_eval-n_diff}/{n_eval} match | {n_real} real BUY<->SELL | "
           f"{n_drift} decisive HOLD<->dir (|dConf|>=10, data-settle drift) | "
           f"{n_bound} near-gate wobble  "
           f"[recent-window, NON-reproducible: informational; verdict=shadow+snapshot]")
    status = "PASS" if n_real == 0 else "ATTENTION"
    return status, msg, flips


def snapshot_check():
    """Deterministic replay of the trader's OWN frozen point-in-time intermediates
    (output/inference_snapshots.jsonl): recompute (signal, confidence) and assert ==
    logged. Immune to data revision, reproducible run-to-run, ~instant (no model fit).
    PASS only at 100% — any logic mismatch is a real bookkeeping bug. WARN if the
    snapshot file isn't there yet."""
    script = os.path.join(ENG, "tools", "validate_snapshot_replay.py")
    if not os.path.exists(script):
        return "WARN", "validate_snapshot_replay.py not found"
    r = subprocess.run([sys.executable, script], cwd=ENG, capture_output=True, text=True)
    out = r.stdout or ""
    if r.returncode == 2:
        return "WARN", "no inference_snapshots.jsonl yet (trader writing it?)"
    # headline line: "BOTH    match : 202/202  (100.00%)   <- headline"
    head = next((ln for ln in out.splitlines() if "BOTH" in ln and "match" in ln), "")
    frac = head.split(":", 1)[1].split("<-")[0].strip() if ":" in head else "?"
    status = "PASS" if r.returncode == 0 else "FAIL"
    return status, f"{frac} bookkeeping match (frozen replay, reproducible)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="deterministic checks only (shadow + snapshot); skip the ~15min parity")
    ap.add_argument("--samples", type=int, default=30, help="parity sample count (default 30)")
    args = ap.parse_args()

    print("=" * 64)
    print("  LIVE SANITY CHECK | [1] engine-vs-trader LIVE + [2] snapshot (deterministic) + [3] engine offline re-run (info)")
    print("=" * 64)

    bad = False

    # [1] SHADOW — deterministic, drives verdict
    s_status, s_msg = shadow_check()
    print(f"\n[1] ENGINE-vs-TRADER live (shadow) : {s_status}")
    print(f"    {s_msg}")
    if s_status == "FAIL":
        bad = True

    # [2] SNAPSHOT REPLAY — deterministic, reproducible 100%, drives verdict
    sn_status, sn_msg = snapshot_check()
    print(f"\n[2] SNAPSHOT REPLAY   : {sn_status}")
    print(f"    {sn_msg}")
    if sn_status == "FAIL":
        bad = True

    # [3] ENGINE-vs-TRADER parity — INFORMATIONAL ONLY. Non-reproducible (data revision +
    #     shifting tail-N window); real flips are listed but do NOT set the verdict.
    if args.quick:
        print("\n[3] ENGINE re-run offline : SKIPPED (--quick; [1] above is the live engine-vs-trader check)")
    else:
        p_status, p_msg, flips = parity_check(args.samples)
        print(f"\n[3] ENGINE-vs-TRADER offline : {p_status}  (re-runs the engine over the last N hours; informational, non-reproducible)")
        print(f"    {p_msg}")
        for fl in flips:
            print(f"      REAL FLIP (likely data-revision artifact): {fl}")

    print("\n" + "=" * 64)
    print(f"  RESULT: {'NEEDS ATTENTION' if bad else 'CLEAN'}  (verdict = shadow + snapshot replay; parity is informational)")
    print("=" * 64)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
