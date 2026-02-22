"""
Post-Reorganization Test
=========================
Verifies that all file paths, imports, and data loading work
after the folder restructure. Does NOT run full diagnostics.

Usage:
  python test_reorg.py
"""

import os
import sys
import time
import traceback

PASS = 0
FAIL = 0
WARN = 0

def ok(msg):
    global PASS
    PASS += 1
    print(f"  [PASS] {msg}")

def fail(msg):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {msg}")

def warn(msg):
    global WARN
    WARN += 1
    print(f"  [WARN] {msg}")


def main():
    global PASS, FAIL, WARN
    print("=" * 60)
    print("  POST-REORGANIZATION TEST")
    print("=" * 60)

    # ----------------------------------------------------------
    # TEST 1: Directory structure exists
    # ----------------------------------------------------------
    print("\n[1/7] Checking directory structure...")
    expected_dirs = [
        "data", "data/indices", "data/crypto",
        "output", "output/charts", "output/dashboards",
        "output/diagnostics", "output/backtests",
        "docs", "macro_data", "models", "venv",
    ]
    for d in expected_dirs:
        if os.path.isdir(d):
            ok(f"{d}/")
        else:
            fail(f"{d}/ NOT FOUND")

    # ----------------------------------------------------------
    # TEST 2: Data files exist at new locations
    # ----------------------------------------------------------
    print("\n[2/7] Checking data files at new paths...")
    data_files = {
        # Index data
        "data/indices/smi_hourly_data.csv":   "SMI hourly data",
        "data/indices/dax_hourly_data.csv":   "DAX hourly data",
        "data/indices/cac40_hourly_data.csv": "CAC40 hourly data",
        # Model configs
        "data/hourly_best_models.csv":        "Hourly best models",
        # Charts
        "output/charts/hourly_chart_data_v2.json": "Chart data v2",
        "output/charts/hourly_chart_data_v3.json": "Chart data v3",
    }
    # Optional files (may or may not exist)
    optional_files = {
        "data/crypto/btc_hourly_data.csv":    "BTC hourly data",
        "data/crypto/eth_hourly_data.csv":    "ETH hourly data",
        "data/best_models.csv":               "Daily best models",
        "data/broly_positions.json":          "Broly positions",
        "output/charts/chart_data.json":      "Chart data (daily)",
    }

    for path, desc in data_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            ok(f"{desc}: {path} ({size:,} bytes)")
        else:
            fail(f"{desc}: {path} NOT FOUND")

    for path, desc in optional_files.items():
        if os.path.exists(path):
            ok(f"{desc}: {path}")
        else:
            warn(f"{desc}: {path} (not found, may not exist yet)")

    # Check nothing was LEFT behind in root
    print("\n  Checking for stale files left in root...")
    stale = []
    for f in os.listdir("."):
        if f.endswith(".csv") and not f.startswith("."):
            stale.append(f)
        if f.endswith(".json") and f not in (".gitignore",):
            stale.append(f)
        if f.endswith(".html"):
            stale.append(f)
    if stale:
        for s in stale:
            warn(f"Stale file in root: {s} (should it be in data/ or output/?)")
    else:
        ok("No stale CSV/JSON/HTML files left in root")

    # ----------------------------------------------------------
    # TEST 3: Python imports work
    # ----------------------------------------------------------
    print("\n[3/7] Testing Python imports...")

    try:
        from hardware_config import get_config
        hw = get_config()
        ok(f"hardware_config: {hw['machine']}, {hw['cpu_cores']} cores")
    except Exception as e:
        fail(f"hardware_config: {e}")

    try:
        from hourly_trading_system import load_data, build_hourly_features
        ok("hourly_trading_system: load_data, build_hourly_features imported")
    except Exception as e:
        fail(f"hourly_trading_system import: {e}")

    try:
        from features_v2 import build_features_v2_hourly
        ok("features_v2: build_features_v2_hourly imported")
    except Exception as e:
        fail(f"features_v2 import: {e}")

    # ----------------------------------------------------------
    # TEST 4: Data loading from new paths
    # ----------------------------------------------------------
    print("\n[4/7] Testing data loading from new paths...")

    try:
        from hourly_trading_system import load_data
        for asset in ['SMI', 'DAX', 'CAC40']:
            df = load_data(asset)
            if df is not None and len(df) > 100:
                ok(f"load_data('{asset}'): {len(df)} rows loaded")
            elif df is not None:
                warn(f"load_data('{asset}'): only {len(df)} rows (expected 4000+)")
            else:
                fail(f"load_data('{asset}'): returned None")
    except Exception as e:
        fail(f"Data loading: {e}")

    # ----------------------------------------------------------
    # TEST 5: Best models CSV readable
    # ----------------------------------------------------------
    print("\n[5/7] Testing best_models loading...")

    try:
        import pandas as pd
        if os.path.exists("data/hourly_best_models.csv"):
            df_best = pd.read_csv("data/hourly_best_models.csv")
            ok(f"hourly_best_models.csv: {len(df_best)} rows, cols={list(df_best.columns)[:5]}")
        else:
            fail("data/hourly_best_models.csv not found")
    except Exception as e:
        fail(f"best_models loading: {e}")

    # ----------------------------------------------------------
    # TEST 6: Output directories writable
    # ----------------------------------------------------------
    print("\n[6/7] Testing output directories are writable...")

    test_dirs = [
        "output/charts", "output/dashboards",
        "output/diagnostics", "output/backtests", "data",
    ]
    for d in test_dirs:
        test_file = os.path.join(d, "_write_test.tmp")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            ok(f"{d}/ is writable")
        except Exception as e:
            fail(f"{d}/ NOT writable: {e}")

    # ----------------------------------------------------------
    # TEST 7: Quick feature build test
    # ----------------------------------------------------------
    print("\n[7/7] Quick feature build test (DAX)...")

    try:
        from hourly_trading_system import load_data, build_hourly_features
        df_raw = load_data('DAX')
        if df_raw is not None:
            df_feat, cols = build_hourly_features(df_raw)
            n_valid = df_feat.dropna(subset=cols + ['label']).shape[0]
            ok(f"V1 features: {len(cols)} features, {n_valid} valid rows")
        else:
            fail("Could not load DAX data for feature test")
    except Exception as e:
        fail(f"Feature build: {e}")

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    total = PASS + FAIL + WARN
    if FAIL == 0:
        print(f"  ALL CLEAR: {PASS} passed, {WARN} warnings, 0 failures")
        print("  Your reorganized setup is working!")
    else:
        print(f"  ISSUES FOUND: {PASS} passed, {FAIL} FAILED, {WARN} warnings")
        print("  Fix the failures above before running trading scripts.")
    print("=" * 60)

    # Remind about unchecked files
    print("\n  MANUAL CHECKS STILL NEEDED:")
    print("  - broly.py (check broly_positions.json path)")
    print("  - crypto_trading_system.py (check crypto CSV paths)")
    print("    Both may still use old root paths.\n")

    return FAIL == 0


if __name__ == "__main__":
    main()
