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
    # NOTE: keep the substring "match" here — the trader's Telegram output filter
    # (crypto_revolut_ed_v2._run_sanity_and_alert) greps for SHADOW/SNAPSHOT/ENGINE/
    # REAL FLIP/RESULT/match to decide which lines to show.
    msg = (f"{n_ok}/{n_tot} signal match (refit==logged, {pct:.0f}%) | {n_real} real BUY<->SELL "
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


# Critical historical CSVs and their sane floors. A HISTORY CLOBBER (the 2026-06-28
# cross_asset 1637->9 silent truncation: a concurrent reader caught the file mid-write,
# read it as empty, and _yf_merge_with_existing returned the tiny incremental slice over
# the full history) ALWAYS collapses the date span — so (min_rows, min_span_days) catches
# it deterministically without hardcoding absolute dates. Add a file here when wiring a
# new long-history source. Path is relative to data/macro_data/.
_INTEGRITY_REGISTRY = {
    # file                  (min_rows, min_span_days)
    "cross_asset.csv":      (1400, 1200),   # daily, since 2022 — the file that clobbered
    "macro_daily.csv":      (1000, 1200),   # daily, since 2022
    "fear_greed.csv":       (2500, 2500),   # daily, since 2018
    "onchain_eth.csv":      (1200, 1000),   # daily on-chain (live ETH)
    "onchain_btc.csv":      (1200, 1000),
    "derivatives_eth.csv":  (30000, 1000),  # hourly (live ETH)
}


def data_integrity_check():
    """[4] DATA INTEGRITY — deterministic & instant. Verifies critical historical CSVs
    have NOT been truncated/collapsed (the 2026-06-28 cross_asset 1637->9 clobber, which
    silently NaN'd every xa_*_corr* feature and zeroed the 8h PySR). A clobber collapses
    both the row count and the date span, so we floor both. FAIL = a real data-loss event
    -> restore via the source's full re-pull (e.g. download_cross_asset(full=True))."""
    mdir = os.path.join(ENG, "data", "macro_data")
    fails, oks = [], 0
    for fname, (min_rows, min_span_days) in _INTEGRITY_REGISTRY.items():
        path = os.path.join(mdir, fname)
        if not os.path.exists(path):
            fails.append(f"{fname}: MISSING")
            continue
        if os.path.getsize(path) == 0:
            fails.append(f"{fname}: EMPTY (0 bytes)")
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            fails.append(f"{fname}: UNREADABLE ({type(e).__name__})")
            continue
        n = len(df)
        dts = pd.to_datetime(df[df.columns[0]], errors="coerce").dropna()
        span = (dts.max() - dts.min()).days if len(dts) else 0
        if n < min_rows or span < min_span_days:
            fails.append(f"{fname}: COLLAPSED rows={n}(min {min_rows}) span={span}d(min {min_span_days}d) "
                         f"start={str(dts.min())[:10] if len(dts) else '?'}")
        else:
            oks += 1
    if fails:
        return "FAIL", f"{len(fails)} file(s) collapsed/missing: " + " | ".join(fails)
    return "PASS", f"{oks}/{len(_INTEGRITY_REGISTRY)} critical CSVs intact (row+span floors)"


def backtest_vs_live_check(min_pct=95.0):
    """[5] BACKTEST-vs-LIVE faithfulness — the tripwire that was MISSING when the
    2026-06-29 training-window-edge divergence sat at 75% backtest-vs-live, undetected
    (snapshot/refit/shadow all test the LIVE path or live's frozen matrix — none ran
    generate_signals' own window construction vs live). Runs generate_signals (the backtest
    path, leakage-free default) for the live config's horizon(s) over the snapshot window
    and compares to what the trader logged. <min_pct => the backtest does NOT predict live
    => a train-window/inference divergence that taints every backtest config decision.
    ~3min (one retrain per live horizon). WARN if no snapshots yet."""
    import re
    script = os.path.join(ENG, "tools", "validate_backtest_vs_live.py")
    if not os.path.exists(script):
        return "WARN", "validate_backtest_vs_live.py not found"
    r = subprocess.run([sys.executable, script, "--min", str(min_pct)],
                       cwd=ENG, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 2 or "cannot validate" in out:
        return "WARN", "no live snapshots yet (trader needs to run)"
    if "WARMING UP" in out:                                    # thin post-promotion sample — don't cry wolf
        return "PASS", "warming up — thin post-promotion sample (recovers within a day)"
    m = re.search(r"overall ([\d.]+)% \| worst-horizon ([\d.]+)%", out)
    if not m:
        return "WARN", "backtest-vs-live produced no parseable result"
    overall, worst = float(m.group(1)), float(m.group(2))
    status = "PASS" if worst >= min_pct else "FAIL"
    return status, f"BACKTEST<->live overall {overall:.0f}% / worst-horizon {worst:.0f}% (threshold {min_pct:.0f}%)"


def trader_health_check():
    """[6] TRADER HEALTHY — from output/cycle_metrics.csv: is the live trader cycling, are
    downloads fast, are inference features clean? Liveness uses file mtime (tz-safe). Hard
    FAIL only on a hung trader (>90min since last cycle); slow-download / NaN-features /
    skipped-cycles are surfaced as warnings without failing the verdict."""
    import time
    path = os.path.join(ENG, "output", "cycle_metrics.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return "SKIP", "no cycle_metrics.csv yet"
    try:
        df = pd.read_csv(path).tail(12)
    except Exception as e:
        return "SKIP", f"cycle_metrics unreadable ({type(e).__name__})"
    if len(df) == 0:
        return "SKIP", "cycle_metrics empty"
    age_min = (time.time() - os.path.getmtime(path)) / 60.0
    maxp1 = float(df["p1_duration_sec"].max()) if "p1_duration_sec" in df.columns else 0.0
    maxnan = int(df["n_features_nan_inference"].max()) if "n_features_nan_inference" in df.columns else 0
    nskip = 0
    if "skip_reason" in df.columns:
        sr = df["skip_reason"].astype(str).str.strip()
        nskip = int(((sr != "") & (sr.str.lower() != "nan") & (sr != "None")).sum())
    warn = []
    if maxp1 > 20:
        warn.append(f"slow download {maxp1:.0f}s")
    if maxnan > 0:
        warn.append(f"{maxnan} NaN feature(s) at inference")
    if nskip:
        warn.append(f"{nskip} skipped cycle(s)")
    if age_min > 90:
        return "FAIL", f"trader STALE — last cycle {age_min:.0f}min ago (not cycling?)"
    base = f"cycling ({age_min:.0f}min ago) | download {maxp1:.1f}s | {maxnan} NaN | {nskip} skips"
    return "PASS", base + (("  warn: " + "; ".join(warn)) if warn else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="deterministic checks only (shadow + snapshot); skip the ~15min parity")
    ap.add_argument("--samples", type=int, default=30, help="parity sample count (default 30)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")   # emoji-safe on Windows cp1252 consoles
    except Exception:
        pass

    # --- run all checks (logic unchanged; captured here, printed synthetically below) ---
    di_status, di_msg = data_integrity_check()      # [4] history files intact
    s_status, s_msg = shadow_check()                # [1] live == engine (shadow)
    sn_status, sn_msg = snapshot_check()            # [2] signals reproduce (snapshot)
    th_status, th_msg = trader_health_check()       # [6] trader healthy (operational)
    if args.quick:
        p_status, p_msg, flips = "SKIPPED", "skipped (--quick; [1]+[2] drive the verdict)", []
        bl_status, bl_msg = "SKIPPED", "skipped (--quick; ~3min retrain)"
    else:
        p_status, p_msg, flips = refit_check(args.samples)   # [3] engine refit (frozen)
        bl_status, bl_msg = backtest_vs_live_check()         # [5] backtest == live

    bad = any(st in ("FAIL", "ATTENTION") for st in
              (di_status, s_status, sn_status, th_status, p_status, bl_status))

    # --- synthetic, verdict-first output. Colour: 🔵 good · 🔴 bad · ⚪ neutral/skipped.
    #     Mode keywords (ENGINE/SNAPSHOT/intact/BACKTEST/RESULT/HEALTHY/REAL FLIP) are kept
    #     in the labels so the trader's Telegram grep still forwards every line.
    def _ic(st):
        return {"PASS": "🔵", "FAIL": "🔴", "ATTENTION": "🔴"}.get(st, "⚪")
    vic = "🔴" if bad else "🔵"
    sep = "  " + "─" * 58
    print("\n" + "=" * 64)
    print(f"  {vic}  SANITY RESULT: {'NEEDS ATTENTION' if bad else 'CLEAN'}   (ETH live · daily verdict)")
    print(sep)
    print(f"  {_ic(s_status)} LIVE SIGNALS == ENGINE (shadow) : {s_msg}")
    print(f"  {_ic(sn_status)} SIGNALS REPRODUCE (SNAPSHOT)    : {sn_msg}")
    print(f"  {_ic(di_status)} HISTORY FILES INTACT           : {di_msg}")
    print(f"  {_ic(th_status)} TRADER HEALTHY                 : {th_msg}")
    print(f"  {_ic(p_status)} ENGINE REFIT (frozen)          : {p_msg}")
    for fl in flips:
        print(f"        REAL FLIP (frozen-data non-reproducibility — investigate): {fl}")
    print(f"  {_ic(bl_status)} BACKTEST == LIVE               : {bl_msg}")
    print(sep)
    print("  ⓘ the SHADOW 'all-time %' is pre-fix history — the VERDICT is the recent-48 number above.")
    print("=" * 64)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
