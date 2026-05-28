"""
Merge OLD PySR expressions (currently in pysr_ETH_<H>h.json) with NEW
expressions (in pysr_ETH_<H>h_new.json) into a single combined JSON
with 10 total expressions:
  - Slots pysr_1..5 = OLD formulas (preserves existing models' references)
  - Slots pysr_6..10 = NEW formulas (additional candidates for new training)

Rationale: instead of swapping OLD -> NEW (which requires retraining all
production models to avoid silent feature drift), we keep BOTH as
candidate features. LGBM gain-based ranking + Optuna's n_features
selection then decides which are most useful. Correlated OLD/NEW pairs
will split the gain budget; usually only one survives the top-N cut.

Backward compatibility:
  - Existing 5h/6h production models reference pysr_1..5 by name. With
    OLD preserved in those slots, their inference behavior is unchanged.
  - New v3-trained 7h+8h models can pick from all 10 slots; LGBM picks
    by gain importance.

Run once before launching v3 HRST. Backs up OLD JSONs as
*_pre_20260528_old_only.json before overwriting.
"""

import json
import os
import shutil
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'models')
HORIZONS = [5, 6, 7, 8]
ASSET = 'ETH'
BACKUP_SUFFIX = '_pre_20260528_old_only'

print(f"PySR merge tool — ASSET={ASSET}, HORIZONS={HORIZONS}")
print(f"  models dir: {MODELS}")
print(f"  backup suffix: {BACKUP_SUFFIX}")
print()

for h in HORIZONS:
    old_path = os.path.join(MODELS, f'pysr_{ASSET}_{h}h.json')
    new_path = os.path.join(MODELS, f'pysr_{ASSET}_{h}h_new.json')
    backup_path = os.path.join(MODELS, f'pysr_{ASSET}_{h}h{BACKUP_SUFFIX}.json')

    if not os.path.exists(old_path):
        print(f"[{h}h] SKIP — OLD file not found: {old_path}")
        continue
    if not os.path.exists(new_path):
        print(f"[{h}h] SKIP — NEW file not found: {new_path}")
        continue

    with open(old_path) as f:
        old_data = json.load(f)
    with open(new_path) as f:
        new_data = json.load(f)

    old_expressions = old_data.get('expressions', [])
    new_expressions = new_data.get('expressions', [])

    if len(old_expressions) != 5 or len(new_expressions) != 5:
        print(f"[{h}h] WARN — expected 5+5 expressions, got OLD={len(old_expressions)}, NEW={len(new_expressions)}")

    # Use NEW's feature_names as the canonical list (it's the up-to-date
    # one that respects current disabled_features.json). OLD's references
    # to disabled features will silently no-op via _compute_pysr_features'
    # missing-dependency handling.
    canonical_feature_names = new_data.get('feature_names', old_data.get('feature_names', []))

    merged = {
        'asset': old_data.get('asset', ASSET),
        'horizon': h,
        'discovered_at': f"MERGED {datetime.now().strftime('%Y-%m-%d %H:%M')} | OLD={old_data.get('discovered_at', '?')} + NEW={new_data.get('discovered_at', '?')}",
        'discovery_method': 'historical',
        'pysr_data_rows': max(old_data.get('pysr_data_rows', 0), new_data.get('pysr_data_rows', 0)),
        'n_expressions': len(old_expressions) + len(new_expressions),
        'feature_names': canonical_feature_names,
        'expressions': list(old_expressions) + list(new_expressions),
        'merge_metadata': {
            'old_n': len(old_expressions),
            'new_n': len(new_expressions),
            'old_source': os.path.basename(old_path),
            'new_source': os.path.basename(new_path),
            'old_slots': [f'pysr_{i+1}' for i in range(len(old_expressions))],
            'new_slots': [f'pysr_{i+1+len(old_expressions)}' for i in range(len(new_expressions))],
        },
    }

    # Backup OLD before overwriting
    if not os.path.exists(backup_path):
        shutil.copyfile(old_path, backup_path)
        print(f"[{h}h] backup created: {os.path.basename(backup_path)}")
    else:
        print(f"[{h}h] backup already exists: {os.path.basename(backup_path)} (not overwriting)")

    # Write merged JSON
    with open(old_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"[{h}h] merged: {len(old_expressions)} OLD + {len(new_expressions)} NEW = {merged['n_expressions']} total")
    print(f"       OLD slots: {merged['merge_metadata']['old_slots']}")
    print(f"       NEW slots: {merged['merge_metadata']['new_slots']}")
    print()

print("DONE.")
print()
print("Verification:")
print("  - Production models trained on pysr_1..5 (OLD) get IDENTICAL features at inference.")
print("  - v3-trained 7h+8h models will pick top-N from pysr_1..10; LGBM gain decides.")
print("  - Rollback: replace pysr_ETH_<H>h.json with pysr_ETH_<H>h_pre_20260528_old_only.json")
print()
print("Recommended next step:")
print('  python crypto_trading_system_ed_g_narrow_d_parallel_nearlive_v3.py H "ETH," 7h,8h --replay 1440 --no-persist --no-data-update')
