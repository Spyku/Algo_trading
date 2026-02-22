"""
Patch: Upgrade hourly_trading_system.py to use optimal 15 V2 features.

Run:
  python patch_v2_features.py

What it changes:
  1. Adds 'from features_v2 import build_features_v2_hourly' import
  2. Adds OPTIMAL_V2_FEATURES constant (the 15 winning features)
  3. Updates generate_signals() to use V2 features
  4. Updates run_full_diagnostic() to use V2 features
  5. Does NOT touch file paths (reorganization already done)
"""

import re
import shutil
from datetime import datetime

TARGET = 'hourly_trading_system.py'
BACKUP = f'hourly_trading_system_v1_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.py'


def patch():
    print("=" * 60)
    print("  PATCH: Upgrade to V2 Optimal Features")
    print("=" * 60)

    # Read current file
    with open(TARGET, 'r', encoding='utf-8') as f:
        code = f.read()

    # Backup
    shutil.copy2(TARGET, BACKUP)
    print(f"  Backup saved: {BACKUP}")

    changes = 0

    # ---------------------------------------------------------------
    # 1. Add V2 import after hardware_config import block
    # ---------------------------------------------------------------
    if 'from features_v2 import' not in code:
        old = "from hardware_config import ("
        new = ("from features_v2 import build_features_v2_hourly\n"
               "from hardware_config import (")
        if old in code:
            code = code.replace(old, new, 1)
            changes += 1
            print("  [1/5] Added features_v2 import")
        else:
            print("  [1/5] SKIP - hardware_config import not found (add manually)")
    else:
        print("  [1/5] SKIP - features_v2 already imported")

    # ---------------------------------------------------------------
    # 2. Add OPTIMAL_V2_FEATURES constant after PREDICTION_HORIZON
    # ---------------------------------------------------------------
    if 'OPTIMAL_V2_FEATURES' not in code:
        v2_block = """
# ============================================================
# OPTIMAL V2 FEATURES (from feature_analysis_v2.py)
# 15 features -> 76.1% accuracy (best subset of 101)
# ============================================================
OPTIMAL_V2_FEATURES = [
    'logret_240h',          # BASE - 10-day momentum
    'm_sp500_vol20d',       # MACRO - S&P500 volatility
    'm_vix_zscore',         # MACRO - VIX normalized
    'm_sp500_zscore',       # MACRO - S&P500 normalized
    'logret_24h',           # BASE - 1-day return
    'volatility_48h',       # BASE - 2-day volatility
    'xa_sp500_relstr5d',    # CROSS-ASSET - relative strength vs S&P
    'atr_pct_14h',          # BASE - ATR as % of price
    'sma20_to_sma50h',      # BASE - MA crossover ratio
    'zscore_50h',           # BASE - price z-score
    'xa_sp500_corr30d',     # CROSS-ASSET - 30d correlation with S&P
    'spread_120h_8h',       # BASE - fast/slow spread
    'xa_nasdaq_corr10d',    # CROSS-ASSET - 10d correlation with Nasdaq
    'm_gold_vol20d',        # MACRO - gold volatility
    'fg_zscore',            # SENTIMENT - Fear & Greed normalized
]
"""
        # Insert after PREDICTION_HORIZON line
        match = re.search(r'(PREDICTION_HORIZON\s*=\s*\d+[^\n]*\n)', code)
        if match:
            insert_pos = match.end()
            code = code[:insert_pos] + v2_block + code[insert_pos:]
            changes += 1
            print("  [2/5] Added OPTIMAL_V2_FEATURES constant")
        else:
            print("  [2/5] SKIP - PREDICTION_HORIZON not found (add manually)")
    else:
        print("  [2/5] SKIP - OPTIMAL_V2_FEATURES already exists")

    # ---------------------------------------------------------------
    # 3. Update generate_signals() to use V2 features
    # ---------------------------------------------------------------
    old_gen = "    df_features, feature_cols = build_hourly_features(df_raw)"
    new_gen = """    # Build V2 features (all 101), then use only optimal 15
    df_v2_all, _all_v2_cols = build_features_v2_hourly(df_raw, original_builder=build_hourly_features)
    feature_cols = [c for c in OPTIMAL_V2_FEATURES if c in df_v2_all.columns]
    if len(feature_cols) < len(OPTIMAL_V2_FEATURES):
        missing = [c for c in OPTIMAL_V2_FEATURES if c not in df_v2_all.columns]
        print(f"    WARNING: Missing V2 features: {missing}")
    df_features = df_v2_all.dropna(subset=feature_cols + ['label']).reset_index(drop=True)"""

    # This appears in generate_signals() — replace only the FIRST occurrence
    if old_gen in code:
        code = code.replace(old_gen, new_gen, 1)
        changes += 1
        print("  [3/5] Updated generate_signals() to use V2 features")
    else:
        print("  [3/5] SKIP - generate_signals build line not found")

    # ---------------------------------------------------------------
    # 4. Update run_full_diagnostic() to use V2 features
    # ---------------------------------------------------------------
    # This is the SECOND occurrence of build_hourly_features(df_raw)
    old_diag = "        df_features, feature_cols = build_hourly_features(df_raw)"
    new_diag = """        # Build V2 features, use optimal 15
        df_v2_all, _all_v2_cols = build_features_v2_hourly(df_raw, original_builder=build_hourly_features)
        feature_cols = [c for c in OPTIMAL_V2_FEATURES if c in df_v2_all.columns]
        df_features = df_v2_all.dropna(subset=feature_cols + ['label']).reset_index(drop=True)"""

    if old_diag in code:
        code = code.replace(old_diag, new_diag, 1)
        changes += 1
        print("  [4/5] Updated run_full_diagnostic() to use V2 features")
    else:
        print("  [4/5] SKIP - diagnostic build line not found")

    # ---------------------------------------------------------------
    # 5. Update the signal dict to handle missing V1 display columns
    # ---------------------------------------------------------------
    # The signal dict references rsi_14h, bb_position_20h, spread_24h_4h
    # These still exist in df_features (V2 builds on V1), but let's be safe
    old_signal_rsi = "            'rsi': round(float(row['rsi_14h']), 1),"
    new_signal_rsi = "            'rsi': round(float(row.get('rsi_14h', 0)), 1),"

    old_signal_bb = "            'bb_position': round(float(row['bb_position_20h']), 3),"
    new_signal_bb = "            'bb_position': round(float(row.get('bb_position_20h', 0)), 3),"

    old_signal_spread = "            'spread_24h_4h': round(float(row['spread_24h_4h'] * 100), 2),"
    new_signal_spread = "            'spread_24h_4h': round(float(row.get('spread_24h_4h', 0) * 100), 2),"

    safe_changes = 0
    for old, new in [(old_signal_rsi, new_signal_rsi),
                     (old_signal_bb, new_signal_bb),
                     (old_signal_spread, new_signal_spread)]:
        if old in code:
            code = code.replace(old, new)
            safe_changes += 1

    if safe_changes > 0:
        changes += 1
        print(f"  [5/5] Made {safe_changes} display columns safe (row.get)")
    else:
        print("  [5/5] SKIP - display columns already safe or not found")

    # ---------------------------------------------------------------
    # Write patched file
    # ---------------------------------------------------------------
    if changes > 0:
        with open(TARGET, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"\n  DONE: {changes} patches applied to {TARGET}")
        print(f"  Backup: {BACKUP}")
        print(f"\n  NEXT STEPS:")
        print(f"  1. Run Mode A to re-run diagnostic with V2 features:")
        print(f"     python hourly_trading_system.py")
        print(f"  2. Then Mode B to generate signals with the new best config")
    else:
        print(f"\n  No changes needed — already patched?")


if __name__ == '__main__':
    patch()
