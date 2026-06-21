"""bt_15m_vs_1h_validate.py — validation gate for the 15' edge before building a trader.

Answers the two open questions from the same-window comparison (15' ~2.7x 1h):
  (1) OUT-OF-SAMPLE: does 15' still beat 1h in the PRIOR month (not the recent one
      the models were ~selected on)?  -> recency check.
  (2) EXECUTION COST: does the 15' edge survive a higher per-leg cost (proxy for
      slippage / missed maker fills at 3x the trade count)?  -> fee curve 5/9/15 bps.

Method: generate signals ONCE over a 2-MONTH window per engine (1h=1440h, 15'=5760
periods — 4x), then slice into non-overlapping RECENT month + PRIOR month (by real
datetimes, candle-agnostic), and sim each at fees {5,9,15} bps/leg (5=maker blend,
9=Revolut-X taker, 15=taker+slippage). Each engine's OWN regime sim (_modec_sim),
OWN config, conf-gates ON, shields OFF, NO rally-gate. Serial inside each process
(no worker pool); the two engines run as concurrent subprocesses.

RUN ON A FREE MACHINE (desktop):  python tools/bt_15m_vs_1h_validate.py
"""
import os, sys, json, argparse
os.environ.setdefault("FAYE_LIBRARY_MODE", "1")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="all", choices=["all", "faye", "fuji15"])
args = ap.parse_args()
FEES_BPS = [5, 9, 15]


def _one_engine(engine):
    import pandas as pd
    if engine == "faye":
        import crypto_trading_system_faye as ENG
        cfg = json.load(open("config/regime_config_ed.json"))["ETH"]
        prod = pd.read_csv("models/crypto_ed_production.csv")
        replay_2mo = 1440        # hours = bars
        maxhold = 10
        tag = "1h "
    else:
        import crypto_trading_system_fujiwara_15 as ENG
        cfg = json.load(open("config_fujiwara_15/regime_config_fujiwara_15.json"))["ETH"]
        prod = pd.read_csv("models_fujiwara_15/crypto_fujiwara_15_production.csv")
        replay_2mo = 5760        # 4x periods = 2 calendar months
        maxhold = 40
        tag = "15'"

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
    print(f"[{tag}] det={det} | bull {specs['bull']['h']}@{specs['bull']['conf']:.0f} | "
          f"bear {specs['bear']['h']}@{specs['bear']['conf']:.0f} | 2mo replay={replay_2mo} maxhold={maxhold}", flush=True)

    sig = {}
    for reg in ("bull", "bear"):
        s = specs[reg]
        sg = ENG.generate_signals("ETH", s["combo"].split("+"), s["window"], replay_2mo,
                                  feature_override=s["features"], horizon=s["h"], gamma=s["gamma"])
        sig[reg] = {pd.Timestamp(x["datetime"]): x for x in sg}
    common = sorted(set(sig["bull"]) & set(sig["bear"]))
    if not common:
        print(f"[{tag}] no common bars", flush=True); return
    end_t = common[-1]
    halves = {
        "recent": [dt for dt in common if dt >= end_t - pd.Timedelta(days=30)],
        "prior_OOS": [dt for dt in common if end_t - pd.Timedelta(days=60) <= dt < end_t - pd.Timedelta(days=30)],
    }
    for hname, dts in halves.items():
        if len(dts) < 50:
            print(f"[{tag}] {hname}: only {len(dts)} bars — SKIP", flush=True); continue
        tagged = []
        for dt in dts:
            is_bull = bool(detfn(dt))
            x = sig["bull" if is_bull else "bear"][dt]
            tagged.append(dict(close=float(x["close"]), signal=x["signal"], confidence=float(x["confidence"]),
                               regime="bull" if is_bull else "bear",
                               conf_threshold=specs["bull"]["conf"] if is_bull else specs["bear"]["conf"]))
        span = (dts[-1] - dts[0]).total_seconds() / 86400.0
        for fb in FEES_BPS:
            ENG.BACKTEST_FEE_PER_LEG = fb / 10000.0
            ret, ntr, wr = ENG._modec_sim(tagged, 0.0, 0.0, maxhold, None, None)
            print(f"[{tag}] {hname:9s} fee{fb:2d}bp: bars={len(dts):5d} span={span:4.1f}d "
                  f"return={ret:+8.2f}% trades={ntr:4d} WR={wr:5.1f}%", flush=True)


def _driver():
    import subprocess, re
    res = {}   # (tag, half, fee) -> (ret, trades, wr)
    pat = re.compile(r"^\[(1h |15')\]\s*(recent|prior_OOS)\s*fee\s*(\d+)bp:\s*bars=\s*\d+.*return=\s*([+-][\d.]+)%\s*trades=\s*(\d+)\s*WR=\s*([\d.]+)%")
    procs = []
    for eng in ("fuji15", "faye"):     # 15' (heavier) first
        lf = open(f"logs/_btv_{eng}.log", "w", encoding="utf-8")
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--engine", eng],
                             stdout=lf, stderr=subprocess.STDOUT, text=True,
                             env=dict(os.environ, FAYE_LIBRARY_MODE="1"))
        procs.append((eng, p, lf)); print(f"  launched {eng} PID={p.pid} -> {lf.name}", flush=True)
    for eng, p, lf in procs:
        p.wait(); lf.close()
        for ln in open(lf.name, encoding="utf-8"):
            m = pat.match(ln)
            if m:
                res[(m.group(1), m.group(2), int(m.group(3)))] = (m.group(4), m.group(5), m.group(6))
        if p.returncode != 0:
            print(f"  [WARN] {eng} exit {p.returncode}", flush=True)
    print("\n" + "=" * 86)
    print("  15' vs 1h VALIDATION  |  OOS (prior month) + fee/slippage curve  |  conf-gate ON, no rally-gate")
    print("=" * 86)
    print(f"  {'half':10s} {'fee':5s} | {'1h:  return / trades / WR':28s} | {'15:  return / trades / WR':28s}")
    print("  " + "-" * 84)
    for half in ("recent", "prior_OOS"):
        for fb in FEES_BPS:
            a = res.get(("1h ", half, fb)); b = res.get(("15'", half, fb))
            af = f"{a[0]}% / {a[1]}t / {a[2]}%" if a else "(missing)"
            bf = f"{b[0]}% / {b[1]}t / {b[2]}%" if b else "(missing)"
            print(f"  {half:10s} {fb:2d}bp | {af:28s} | {bf:28s}")
        print("  " + "-" * 84)
    print("=" * 86)


if args.engine == "all":
    _driver()
else:
    _one_engine(args.engine)
