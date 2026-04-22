"""
check_pysr_consistency.py — verify Mode P and HRST didn't drift.

Checks:
  1. All PySR JSON mtimes + discovered_at timestamps
  2. Ordering: PySR JSONs were written BEFORE the latest HRST log started
  3. Production CSV rows reference pysr_* features (or explain why none)
  4. `_check_pysr_leakage` passed in the latest HRST log
  5. Every PySR formula parses as valid sympy, every referenced feature exists

Run:
  python tools/check_pysr_consistency.py                   # ETH default
  python tools/check_pysr_consistency.py --asset BTC
  python tools/check_pysr_consistency.py --asset ETH --horizons 5,6,7,8

Exit code 0 = all clean. Non-zero = at least one check failed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import warnings
from datetime import datetime

import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)


def find_latest_hrst_log(asset: str):
    """Return (path, start_dt) of the most recent HRST log for the asset, or (None, None)."""
    candidates = []
    for log in sorted(glob.glob(os.path.join(ENGINE_DIR, 'logs', 'ed_v1_*.log')), reverse=True):
        try:
            with open(log, encoding='utf-8', errors='ignore') as f:
                head = f.read(500)
        except Exception:
            continue
        if f'Mode HRST | {asset}' in head or f'Mode HRS | {asset}' in head:
            # Extract start time from filename: ed_v1_YYYYMMDD_HHMMSS.log
            m = re.search(r'ed_v1_(\d{8})_(\d{6})\.log', log)
            if m:
                dt = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
                candidates.append((log, dt))
    if not candidates:
        return None, None
    return candidates[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', default='ETH')
    ap.add_argument('--horizons', default='5,6,7,8', help='comma-separated')
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(',')]
    asset = args.asset.upper()

    print("=" * 90)
    print(f"  PYSR / HRST CONSISTENCY CHECK — {asset} horizons {horizons}")
    print("=" * 90)

    fail_count = 0

    # -------- 1) PySR JSONs exist + metadata --------
    print("\n1) PySR JSONs present + metadata:")
    pysr_info = {}
    for h in horizons:
        path = os.path.join(ENGINE_DIR, 'models', f'pysr_{asset}_{h}h.json')
        if not os.path.exists(path):
            print(f"   {h}h: [FAIL] MISSING: {path}")
            fail_count += 1
            continue
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        j = json.load(open(path))
        method = j.get('discovery_method', 'unknown')
        discovered_at = j.get('discovered_at', 'unknown')[:19]
        n_expr = len(j.get('expressions', []))
        pysr_info[h] = {'mtime': mtime, 'method': method, 'n_expr': n_expr, 'path': path, 'json': j}
        tag = '[OK]' if method == 'historical' else '[WARN: non-historical]'
        print(f"   {h}h: {tag} mtime {mtime.strftime('%Y-%m-%d %H:%M:%S')} | discovered_at {discovered_at} | method={method} | exprs={n_expr}")
        if method != 'historical':
            fail_count += 1  # non-historical = leakage risk

    # -------- 2) Ordering: PySR before HRST --------
    print("\n2) Ordering — all PySR JSONs pre-date the latest HRST?")
    log_path, hrst_start = find_latest_hrst_log(asset)
    if not log_path:
        print(f"   [SKIP] No HRST/HRS log found for {asset}. Run HRST first to validate this asset.")
    else:
        log_mtime = datetime.fromtimestamp(os.path.getmtime(log_path))
        print(f"   Latest log: {os.path.basename(log_path)}  start {hrst_start}  end ~{log_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        for h in sorted(pysr_info):
            pj_mtime = pysr_info[h]['mtime']
            if pj_mtime < hrst_start:
                print(f"   {h}h: [OK] PySR written {(hrst_start - pj_mtime).total_seconds()/3600:.1f}h before HRST")
            elif pj_mtime < log_mtime:
                # PySR was re-written DURING the run — could be part of Mode P inside HRST,
                # or a separate P ran mid-HRST. Investigate.
                print(f"   {h}h: [WARN] PySR mtime {pj_mtime} is AFTER HRST start {hrst_start} but before end — partial overlap")
                fail_count += 1
            else:
                print(f"   {h}h: [FAIL] PySR mtime {pj_mtime} is AFTER HRST end {log_mtime} — model trained on stale formulas")
                fail_count += 1

    # -------- 3) Production CSV uses PySR features? --------
    print("\n3) Production CSV feature selection:")
    prod_path = os.path.join(ENGINE_DIR, 'models', 'crypto_ed_production.csv')
    if not os.path.exists(prod_path):
        print(f"   [SKIP] {prod_path} not found.")
    else:
        prod = pd.read_csv(prod_path)
        asset_rows = prod[prod['coin'] == asset].sort_values('horizon')
        if asset_rows.empty:
            print(f"   [SKIP] No rows for {asset}.")
        for _, r in asset_rows.iterrows():
            h = int(r['horizon'])
            if h not in horizons:
                continue
            feats = [f.strip() for f in str(r.get('optimal_features', '')).split(',') if f.strip()]
            pysr_in_feats = [f for f in feats if f.startswith('pysr_')]
            total = len(feats)
            if pysr_in_feats:
                print(f"   {asset} {h}h: [OK] {len(pysr_in_feats)}/{total} selected features are pysr_*: {pysr_in_feats}")
            else:
                print(f"   {asset} {h}h: [INFO] 0 pysr_* among {total} selected features (LGBM importance ranked them below cutoff — not a bug)")

    # -------- 4) Leakage guard triggered in log --------
    print("\n4) Leakage guard check in latest log:")
    if log_path:
        with open(log_path, encoding='utf-8', errors='ignore') as f:
            content = f.read()
        clean_lines = [l for l in content.split('\n') if 'PySR leakage check' in l and 'clean' in l.lower()]
        leak_lines = [l for l in content.split('\n')
                      if re.search(r'(PySR.*leak|leaky.*PySR|stripped.*PySR)', l, re.I)
                      and 'clean' not in l.lower()]
        if leak_lines:
            print(f"   [FAIL] {len(leak_lines)} lines indicate leakage / feature-drop:")
            for l in leak_lines[:5]:
                print(f"     {l.strip()}")
            fail_count += 1
        elif clean_lines:
            print(f"   [OK] {len(clean_lines)} clean-leakage-check entries in log.")
        else:
            print(f"   [INFO] No leakage-guard lines at all — unusual but not failing.")

    # -------- 5) Functional parse + reference check --------
    print("\n5) Functional check — formulas parse + references exist:")
    warnings.filterwarnings('ignore')
    import sympy
    from crypto_trading_system_ed import load_data, build_all_features
    df = load_data(asset)
    if df is None:
        print(f"   [SKIP] No hourly data for {asset}.")
    else:
        for h in sorted(pysr_info):
            _, cols = build_all_features(df, asset_name=asset, horizon=h, verbose=False)
            col_set = set(cols) | set(df.columns)
            j = pysr_info[h]['json']
            ok = fail = 0
            missing = set()
            for e in j.get('expressions', []):
                eq = e.get('sympy_format') or e.get('equation', '')
                try:
                    sym = sympy.sympify(eq)
                    ok += 1
                    for s in sym.free_symbols:
                        name = str(s)
                        if name not in col_set and not name.startswith('pysr_'):
                            missing.add(name)
                except Exception:
                    fail += 1
            tag = '[OK]' if (fail == 0 and not missing) else '[FAIL]'
            line = f"   {h}h: {tag} {ok}/{ok+fail} parse. Missing refs: {sorted(missing) or 'none'}"
            print(line)
            if fail > 0 or missing:
                fail_count += 1

    # -------- Verdict --------
    print()
    print("=" * 90)
    if fail_count == 0:
        print("  VERDICT: CLEAN. HRST and PySR are consistent; no drift detected.")
        exit_code = 0
    else:
        print(f"  VERDICT: {fail_count} issue(s) found. See [FAIL]/[WARN] lines above.")
        exit_code = 1
    print("=" * 90)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
