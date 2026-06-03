"""
build_data_snapshot.py — truncated copy of data/ as-of a past date
==================================================================
Creates data_asof_<YYYYMMDD>/ containing every data/ CSV cut to rows STRICTLY
BEFORE the cutoff (00:00 UTC of that date). Keeps all history before the cutoff,
removes the future. Lets the engine run "as of" a past date via --data-dir,
WITHOUT touching the live data/ files the running trader reads.

Time column auto-detected per file: 'datetime' | 'date' | 'timestamp'(ms).
Files with no time column are copied whole.

Usage:  python tools/build_data_snapshot.py 2026-05-03
"""
import os
import sys
import shutil
import pandas as pd

if len(sys.argv) < 2:
    print("usage: python tools/build_data_snapshot.py YYYY-MM-DD")
    sys.exit(1)

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = "data"
cutoff_str = sys.argv[1]
cutoff = pd.Timestamp(cutoff_str)
dest = f"data_asof_{cutoff_str.replace('-', '')}"
os.makedirs(os.path.join(dest, "macro_data"), exist_ok=True)

print(f"Building {dest}/  (keep rows with time < {cutoff})")


def _time_col(df):
    for c in ("datetime", "date", "timestamp"):
        if c in df.columns:
            return c
    return None


def _process(src_path, dst_path, label):
    try:
        df = pd.read_csv(src_path)
    except Exception as e:
        shutil.copy2(src_path, dst_path)
        print(f"  {label}: copied whole (unparseable: {str(e)[:40]})")
        return
    tc = _time_col(df)
    if tc is None:
        shutil.copy2(src_path, dst_path)
        print(f"  {label}: copied whole (no time column)")
        return
    kw = {"unit": "ms"} if tc == "timestamp" else {}
    # utc=True + tz_localize(None) -> uniform naive-UTC, tolerant of tz-aware cols
    t = pd.to_datetime(df[tc], errors="coerce", utc=True, **kw).dt.tz_localize(None)
    keep = df[t < cutoff]
    keep.to_csv(dst_path, index=False)
    last = t[t < cutoff].max()
    print(f"  {label}: {len(df)} -> {len(keep)} rows  (last kept: {last})  [{tc}]")


# root CSVs
for f in sorted(os.listdir(SRC)):
    sp = os.path.join(SRC, f)
    if os.path.isfile(sp) and f.endswith(".csv"):
        _process(sp, os.path.join(dest, f), f)
# macro_data CSVs
md = os.path.join(SRC, "macro_data")
for f in sorted(os.listdir(md)):
    sp = os.path.join(md, f)
    if os.path.isfile(sp) and f.endswith(".csv"):
        _process(sp, os.path.join(dest, "macro_data", f), f"macro_data/{f}")

print(f"\nDONE -> {dest}/")
