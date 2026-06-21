"""bt_15m_vs_1h_parallel.py — PARALLEL driver for the 15' vs 1h same-window compare.

Runs all 4 (engine x window) combos CONCURRENTLY on a FREE machine (the desktop).
Each child is the SERIAL single-combo worker (bt_15m_vs_1h.py --engine X --window W),
so there is NO worker-pool inside any process -> no deadlock; the OS just runs the 4
side by side. The 15' MONTH (the long pole) is launched FIRST so it gets the head
start; the other three finish well before it.

USE ONLY ON A FREE MACHINE (no concurrent HRST) — 4 heavy serial backtests at once
will saturate a busy box. The desktop (RTX 4080 / 26 cores) handles it fine.

Each combo streams to its own log (logs/_btp_<engine>_<window>.log) so you can tail
any of them live. Results print here as each finishes; a final table at the end.

USAGE (on the desktop):
    python tools/bt_15m_vs_1h_parallel.py
"""
import os, sys, subprocess, re, time
os.environ.setdefault("FAYE_LIBRARY_MODE", "1")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "bt_15m_vs_1h.py")
os.makedirs("logs", exist_ok=True)

# 15' month FIRST (long pole), then the rest — all run concurrently.
JOBS = [("fuji15", "month"), ("faye", "month"), ("fuji15", "week"), ("faye", "week")]
PAT = re.compile(r"^\[(1h |15')\]\s*(week|month)\s*:\s*bars=\s*\d+.*return=\s*([+-][\d.]+)%\s+trades=\s*(\d+)\s+WR=\s*([\d.]+)%")

procs = []
for eng, w in JOBS:
    lf = open(f"logs/_btp_{eng}_{w}.log", "w", encoding="utf-8")
    p = subprocess.Popen([sys.executable, WORKER, "--engine", eng, "--window", w],
                         stdout=lf, stderr=subprocess.STDOUT, text=True,
                         env=dict(os.environ, FAYE_LIBRARY_MODE="1"))
    procs.append([eng, w, p, lf, False])   # last = done-flag
    print(f"  launched {eng:6s} {w:5s}  PID={p.pid}  -> {lf.name}", flush=True)
    time.sleep(3)   # small stagger: 15' month grabs resources first; avoids 4 feature-builds hitting disk at the same instant

print(f"\n  {len(JOBS)} combos running concurrently. Results appear below as each finishes "
      f"(15' month last). Tail any logs/_btp_*.log to watch live.\n", flush=True)

res = {}
while not all(j[4] for j in procs):
    for j in procs:
        eng, w, p, lf, done = j
        if done:
            continue
        if p.poll() is not None:        # finished
            lf.flush()
            try:
                txt = open(lf.name, encoding="utf-8").read()
            except Exception:
                txt = ""
            hit = None
            for ln in txt.splitlines():
                m = PAT.match(ln)
                if m:
                    res[(m.group(1), m.group(2))] = (m.group(3), m.group(4), m.group(5))
                    hit = ln.strip()
            tagn = "1h " if eng == "faye" else "15'"
            if hit:
                print(f"  DONE {eng:6s} {w:5s}: {hit}", flush=True)
            else:
                tail = "\n".join(txt.splitlines()[-6:])
                print(f"  DONE {eng:6s} {w:5s}: NO RESULT (exit {p.returncode}). tail:\n{tail}", flush=True)
            j[4] = True
    time.sleep(10)

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
