"""
migrate_folders.py — One-time folder restructure
===================================================
Run this ONCE to move files into the new folder structure:
  data/         ← hourly CSVs + macro_data/
  charts/       ← backtest PNGs
  models/       ← best_models CSV, feature analysis CSVs, chart data JSON
  config/       ← telegram_config.json, revolut_x_config.json, private.pem, position JSON

Usage:
  python migrate_folders.py          # Preview changes
  python migrate_folders.py --go     # Execute migration
"""

import os
import shutil
import sys

DRY_RUN = '--go' not in sys.argv

folders = ['data', 'data/macro_data', 'charts', 'models', 'config']

moves = {
    # Data files → data/
    'btc_hourly_data.csv': 'data/',
    'eth_hourly_data.csv': 'data/',
    'sol_hourly_data.csv': 'data/',
    'xrp_hourly_data.csv': 'data/',
    'doge_hourly_data.csv': 'data/',
    'smi_hourly_data.csv': 'data/',
    'dax_hourly_data.csv': 'data/',
    'cac40_hourly_data.csv': 'data/',
    
    # Charts → charts/
    'BTC_backtest.png': 'charts/',
    'BTC_1h_backtest.png': 'charts/',
    'BTC_2h_backtest.png': 'charts/',
    'BTC_4h_backtest.png': 'charts/',
    'BTC_8h_backtest.png': 'charts/',
    'ETH_backtest.png': 'charts/',
    'ETH_8h_backtest.png': 'charts/',
    'SOL_backtest.png': 'charts/',
    'XRP_backtest.png': 'charts/',
    'DOGE_backtest.png': 'charts/',
    
    # Models → models/
    'crypto_hourly_best_models.csv': 'models/',
    'crypto_hourly_best_models_mode_d.csv': 'models/',
    'crypto_hourly_chart_data.json': 'models/',
    'crypto_feature_set_comparison.csv': 'models/',
    'crypto_feature_analysis_btc_auto.csv': 'models/',
    'crypto_feature_analysis_eth_auto.csv': 'models/',
    'crypto_feature_analysis_sol_auto.csv': 'models/',
    'crypto_feature_analysis_xrp_auto.csv': 'models/',
    'crypto_feature_analysis_doge_auto.csv': 'models/',
    
    # Config → config/
    'telegram_config.json': 'config/',
    'revolut_x_config.json': 'config/',
    'private.pem': 'config/',
    'revolut_position.json': 'config/',
}

# Also move macro_data contents
macro_moves = {
    'macro_data/macro_daily.csv': 'data/macro_data/',
    'macro_data/fear_greed.csv': 'data/macro_data/',
    'macro_data/cross_asset.csv': 'data/macro_data/',
    'macro_data/macro_hourly.csv': 'data/macro_data/',
}

print("=" * 60)
print(f"  FOLDER MIGRATION {'(PREVIEW)' if DRY_RUN else '(EXECUTING)'}")
print("=" * 60)

# Create folders
for folder in folders:
    if not os.path.exists(folder):
        print(f"  📁 Create: {folder}/")
        if not DRY_RUN:
            os.makedirs(folder, exist_ok=True)

# Move files
moved = 0
skipped = 0
for src, dst_dir in {**moves, **macro_moves}.items():
    if os.path.exists(src):
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.exists(dst):
            print(f"  ⚠️  Skip (already exists): {src} → {dst}")
            skipped += 1
        else:
            print(f"  📄 Move: {src} → {dst}")
            if not DRY_RUN:
                shutil.move(src, dst)
            moved += 1

# Clean up old macro_data if empty
if not DRY_RUN and os.path.exists('macro_data'):
    remaining = os.listdir('macro_data')
    if not remaining:
        os.rmdir('macro_data')
        print("  🗑️  Removed empty macro_data/")

print(f"\n  Moved: {moved} | Skipped: {skipped}")

if DRY_RUN:
    print(f"\n  This was a PREVIEW. Run with --go to execute:")
    print(f"    python migrate_folders.py --go")
else:
    print(f"\n  ✓ Migration complete!")
    print(f"  Files are now in: data/ charts/ models/ config/")
