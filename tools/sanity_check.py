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


def refit_check(samples):
    """[3] Engine REFIT from the trader's FROZEN training matrices
    (output/inference_train_snapshots.jsonl) — NON-revised, point-in-time, and
    reproducible by construction (the matrices are what the trader ACTUALLY trained
    on). This REPLACES the old rebuild-from-CURRENT-data parity
    (validate_core_against_signal_log), which rebuilt features from now-revised data
    and false-alarmed: on 2026-06-21 it reported "19/30 real BUY<->SELL" that was
    NOT a live error — partly a validator regime bug (sma24 vs live sma48, fixed
    8947862) and partly the revised-data rebuild + cross-machine snapshot diffs.
    Refit-from-frozen on the same hours was 30/30. Per the owner's directive:
    the validator MUST compute on non-revised data. (The backtester already does —
    all sources route through _dedup_preserve_history / _merge_preserve_history.)"""
    import re
    print(f"  running engine REFIT-from-frozen replay ({samples} samples)...", flush=True)
    r = subprocess.run(
        [sys.executable, os.path.join(ENG, "tools", "validate_refit_replay.py"),
         "--samples", str(samples)],
        cwd=ENG, capture_output=True, text=True,
    )
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 2 or "not found yet" in out:
        return "WARN", ("no frozen training snapshots yet "
                        "(output/inference_train_snapshots.jsonl) — trader needs the "
                        "patched code + a restart; falls back to shadow+snapshot verdict"), []
    m = re.search(r"signal match:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)", out)
    if not m:
        return "WARN", "refit-replay produced no parseable result", []
    n_ok, n_tot, pct = int(m.group(1)), int(m.group(2)), float(m.group(3))
    # A real BUY<->SELL on FROZEN data is a genuine non-reproducibility (real bug),
    # since revision/cross-machine noise is eliminated by using the frozen matrix.
    flips = []
    for ln in out.splitlines():
        fm = re.search(r"refit (BUY|SELL)\([^)]*\) vs logged (BUY|SELL)\(", ln)
        if fm and fm.group(1) != fm.group(2):
            flips.append(ln.strip())
    n_real = len(flips)
    status = "PASS" if (n_real == 0 and pct >= 99.0) else "ATTENTION"
    msg = (f"{n_ok}/{n_tot} refit==logged ({pct:.0f}%) | {n_real} real BUY<->SELL "
           f"[frozen training matrix — non-revised, REPRODUCIBLE]")
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
    print("  LIVE SANITY CHECK | [1] engine-vs-trader LIVE + [2] snapshot + [3] engine REFIT from frozen (all non-revised / reproducible)")
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

    # [3] ENGINE REFIT from FROZEN training matrices — non-revised + reproducible, so
    #     unlike the old rebuild-from-now parity it CAN drive the verdict: a real
    #     BUY<->SELL here is genuine non-reproducibility, not revision noise. WARN
    #     (no frozen snapshots yet) does NOT fail — falls back to shadow+snapshot.
    if args.quick:
        print("\n[3] ENGINE REFIT (frozen) : SKIPPED (--quick; [1]+[2] are deterministic and drive the verdict)")
    else:
        p_status, p_msg, flips = refit_check(args.samples)
        print(f"\n[3] ENGINE REFIT (frozen, non-revised) : {p_status}  (refits from the trader's own frozen training matrices — reproducible)")
        print(f"    {p_msg}")
        for fl in flips:
            print(f"      REAL non-reproducibility (frozen data — investigate): {fl}")
        if p_status == "ATTENTION":
            bad = True

    print("\n" + "=" * 64)
    print(f"  RESULT: {'NEEDS ATTENTION' if bad else 'CLEAN'}  (verdict = shadow + snapshot + frozen-refit; all non-revised)")
    print("=" * 64)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
