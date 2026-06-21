"""bt_15m_vs_1h.py — same-window head-to-head: 15' fork vs 1h faye.

Per owner directive (2026-06-21): compare both systems over the SAME calendar
window (last MONTH + last WEEK), with the 15' at 4x the 1h period count (so the
calendar span matches), 15' stepped at EVERY period (generate_signals is step=1).

Each system is backtested with its OWN engine's regime sim (_modec_sim) on its
OWN selected config (detector + bull/bear horizon + models + per-regime conf gate),
shields OFF (both configs), NO rally-cooldown gate (clean; the 15' gate was the
skipped-sweep bug anyway). max_hold scaled 4x for the 15' so the wall-clock hold
cap matches (1h: 10 bars=10h; 15': 40 bars=10h).

generate_signals is SERIAL (no worker pool) so there is no deadlock — but it is
GPU/CPU heavy. RUN ON A FREE MACHINE (no concurrent HRST), e.g. the desktop.

USAGE (one command — runs everything, prints a final table):
    python tools/bt_15m_vs_1h.py
Single engine/window (used internally + for manual runs):
    python tools/bt_15m_vs_1h.py --engine fuji15 --window week
"""
import os, sys, json, argparse
os.environ.setdefault("FAYE_LIBRARY_MODE", "1")   # skip the os.execv re-exec + run-as-main side effects
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="all", choices=["all", "faye", "fuji15"])
ap.add_argument("--window", default="both", choices=["week", "month", "both"])
args = ap.parse_args()


# ── Driver: run both engines (each in its own process) + print a combined table ──
def _driver():
    import subprocess, re
    res = {}   # (tag, window) -> (ret, trades, wr)
    pat = re.compile(r"^\[(1h |15')\]\s*(week|month)\s*:\s*bars=\s*\d+.*return=\s*([+-][\d.]+)%\s+trades=\s*(\d+)\s+WR=\s*([\d.]+)%")
    for eng in ("faye", "fuji15"):
        print(f"\n{'='*72}\n  RUNNING {eng}  (week then month — serial, full resolution)\n{'='*72}", flush=True)
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--engine", eng, "--window", "both"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for ln in proc.stdout:
            print(ln, end="", flush=True)          # stream live so progress is visible
            m = pat.match(ln)
            if m:
                res[(m.group(1), m.group(2))] = (m.group(3), m.group(4), m.group(5))
        proc.wait()
        if proc.returncode != 0:
            print(f"  [WARN] {eng} exited with code {proc.returncode}", flush=True)
    print("\n" + "=" * 72)
    print("  15' vs 1h  |  SAME calendar window  |  conf-gates ON, shields OFF, NO rally-gate")
    print("=" * 72)
    print(f"  {'window':6s} | {'1h:  return / trades / WR':30s} | {'15:  return / trades / WR':30s}")
    print("  " + "-" * 70)
    for w in ("week", "month"):
        a = res.get(("1h ", w)); b = res.get(("15'", w))
        af = f"{a[0]}% / {a[1]}t / {a[2]}%" if a else "(missing)"
        bf = f"{b[0]}% / {b[1]}t / {b[2]}%" if b else "(missing)"
        print(f"  {w:6s} | {af:30s} | {bf:30s}")
    print("=" * 72)


def _one_engine():
    import pandas as pd
    if args.engine == "faye":
        import crypto_trading_system_faye as ENG
        cfg = json.load(open("config/regime_config_ed.json"))["ETH"]
        prod = pd.read_csv("models/crypto_ed_production.csv")
        windows = {"week": 168, "month": 720}
        maxhold_bars = 10
        tag = "1h "
    else:
        import crypto_trading_system_fujiwara_15 as ENG
        cfg = json.load(open("config_fujiwara_15/regime_config_fujiwara_15.json"))["ETH"]
        prod = pd.read_csv("models_fujiwara_15/crypto_fujiwara_15_production.csv")
        windows = {"week": 672, "month": 2880}   # 4x periods = same calendar span
        maxhold_bars = 40                         # 4x = 10h wall-clock
        tag = "15'"
    if args.window != "both":
        windows = {args.window: windows[args.window]}

    det = cfg["regime_detector"]["params"]["name"]
    specs = {}
    for reg in ("bull", "bear"):
        h = int(cfg[reg]["horizon"]); conf = float(cfg[reg]["min_confidence"])
        rr = prod[(prod["coin"] == "ETH") & (prod["horizon"] == h)].sort_values("combined_score", ascending=False)
        r = rr.iloc[0]
        combo = str(r["models"]) if "models" in r and pd.notna(r["models"]) else str(r["best_combo"])
        feats = r["optimal_features"].split(",") if pd.notna(r.get("optimal_features")) else None
        specs[reg] = dict(h=h, conf=conf, combo=combo, window=int(r["best_window"]),
                          gamma=float(r["gamma"]), features=feats)

    ind, detectors = ENG._build_regime_indicators_and_detectors("ETH")
    detfn = detectors[det]
    print(f"[{tag}] detector={det} | bull {specs['bull']['h']}@{specs['bull']['conf']:.0f} "
          f"{specs['bull']['combo']} w{specs['bull']['window']} | bear {specs['bear']['h']}@{specs['bear']['conf']:.0f} "
          f"{specs['bear']['combo']} w{specs['bear']['window']} | max_hold={maxhold_bars} bars", flush=True)

    for wname, replay in windows.items():
        sig = {}
        for reg in ("bull", "bear"):
            s = specs[reg]
            sg = ENG.generate_signals("ETH", s["combo"].split("+"), s["window"], replay,
                                      feature_override=s["features"], horizon=s["h"], gamma=s["gamma"])
            sig[reg] = {pd.Timestamp(x["datetime"]): x for x in sg}
        common = sorted(set(sig["bull"]) & set(sig["bear"]))
        if not common:
            print(f"[{tag}] {wname}: no common bars", flush=True); continue
        tagged = []
        for dt in common:
            is_bull = bool(detfn(dt))
            x = sig["bull" if is_bull else "bear"][dt]
            tagged.append(dict(close=float(x["close"]), signal=x["signal"],
                               confidence=float(x["confidence"]),
                               regime="bull" if is_bull else "bear",
                               conf_threshold=specs["bull"]["conf"] if is_bull else specs["bear"]["conf"]))
        ret, ntr, wr = ENG._modec_sim(tagged, 0.0, 0.0, maxhold_bars, None, None)
        bullpct = 100.0 * sum(1 for dt in common if bool(detfn(dt))) / len(common)
        span_days = (common[-1] - common[0]).total_seconds() / 86400.0
        print(f"[{tag}] {wname:5s}: bars={len(common):5d} span={span_days:4.1f}d  "
              f"return={ret:+8.2f}%  trades={ntr:4d}  WR={wr:5.1f}%  bull%={bullpct:4.0f}", flush=True)


if args.engine == "all":
    _driver()
else:
    _one_engine()
