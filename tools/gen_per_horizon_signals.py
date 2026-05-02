"""
gen_per_horizon_signals.py — generate signal stream per ETH horizon (5,6,7,8h)
for the last 90 days, save to a single pickle indexed by (datetime, horizon).
Used by T1b multi-horizon ensemble test.

Each horizon's signals come from its production model (read from
crypto_ed_production.csv). NO production files are modified.

Output: data/eth_per_horizon_signals_90d.pkl
"""
from __future__ import annotations

import os
import pickle
import sys

import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)

PROD_CSV = os.path.join(ENGINE, 'models', 'crypto_ed_production.csv')
OUT_PKL = os.path.join(ENGINE, 'data', 'eth_per_horizon_signals_90d.pkl')

REPLAY_HOURS = 91 * 24

HORIZONS = [5, 6, 7, 8]

# Lazy import to avoid heavy startup
from crypto_trading_system_ed import generate_signals  # noqa: E402

print(f"Generating per-horizon ETH signals over {REPLAY_HOURS}h replay...")
df_models = pd.read_csv(PROD_CSV)

results = {}
for h in HORIZONS:
    rows = df_models[(df_models['coin'] == 'ETH') & (df_models['horizon'] == h)]
    if len(rows) == 0:
        print(f"  h={h}: no production model — skipping")
        continue
    row = rows.sort_values('combined_score', ascending=False).iloc[0]
    feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
    gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0

    print(f"  h={h}: {row['models']} w={int(row['best_window'])}h gamma={gamma:.4f} feats={len(feats)}...")
    sigs = generate_signals('ETH', row['models'].split('+'),
                            int(row['best_window']), REPLAY_HOURS,
                            feature_override=feats, horizon=h, gamma=gamma)
    print(f"    -> {len(sigs)} signals")
    results[h] = sigs

with open(OUT_PKL, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved to: {OUT_PKL}")
print(f"Horizons: {sorted(results.keys())}")
