"""
Generate config/feature_manifest.json from the production CSV + PySR JSONs.

The manifest is a contract: per (asset, horizon), it lists EVERY feature name
the inference path will read. Downstream tools (preflight, audit) use this
instead of re-parsing crypto_ed_production.csv every time.

Usage:
  python tools/generate_feature_manifest.py
  python tools/generate_feature_manifest.py --out config/feature_manifest.json
"""
import argparse
import csv
import json
import os
import re
import sys

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD_CSV = os.path.join(ENGINE_DIR, 'models', 'crypto_ed_production.csv')
DEFAULT_OUT = os.path.join(ENGINE_DIR, 'config', 'feature_manifest.json')

PYSR_BUILTINS = {
    'tanh', 'sqrt', 'abs', 'Abs', 'log', 'sin', 'cos', 'exp', 'pow',
    'max', 'min', 'floor', 'ceil', 'round', 'sign',
    'PySRFunction', 'X', 'e', 'pi', 'sinh', 'cosh',
}


def pysr_inputs(asset, horizon):
    """Return list of feature names that appear inside any PySR formula for this (asset, horizon)."""
    path = os.path.join(ENGINE_DIR, 'models', f'pysr_{asset}_{horizon}h.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        j = json.load(f)
    inputs = set()
    for expr in j.get('expressions', []):
        eq = expr.get('sympy_format') or expr.get('equation') or ''
        for tok in re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', eq):
            if tok in PYSR_BUILTINS:
                continue
            if tok.replace('_', '').isalpha() and len(tok) <= 2:
                continue  # single-letter sympy vars
            inputs.add(tok)
    return sorted(inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--prod-csv', default=PROD_CSV)
    args = ap.parse_args()

    manifest = {'_generated_at': '', '_source_csv': args.prod_csv, 'assets': {}}

    from datetime import datetime, timezone
    manifest['_generated_at'] = datetime.now(timezone.utc).isoformat()

    with open(args.prod_csv) as f:
        for row in csv.DictReader(f):
            asset = row['coin']
            horizon = int(row['horizon'])
            feats_direct = [x.strip() for x in row['optimal_features'].split(',') if x.strip() and x.strip() != 'nan']
            # Union with PySR-input features: if optimal_features contains pysr_*,
            # the actual inputs needed also include every name referenced inside each formula.
            pysr_needed = any(f.startswith('pysr_') for f in feats_direct)
            indirect = pysr_inputs(asset, horizon) if pysr_needed else []
            all_needed = sorted(set(feats_direct) | set(indirect))
            manifest['assets'].setdefault(asset, {})[str(horizon)] = {
                'direct': feats_direct,
                'pysr_inputs': indirect,
                'union': all_needed,
                'combo': row['best_combo'],
                'window': int(row['best_window']),
                'gamma': float(row['gamma']),
            }

    # Build a reverse index: feature → list of (asset, horizon) that need it
    reverse = {}
    for asset, horizons in manifest['assets'].items():
        for h, info in horizons.items():
            for feat in info['union']:
                reverse.setdefault(feat, []).append(f'{asset}{h}h')
    manifest['feature_to_models'] = {f: sorted(ms) for f, ms in sorted(reverse.items())}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'Manifest written: {args.out}')
    print(f'  Assets: {sorted(manifest["assets"].keys())}')
    total_feats = len(manifest['feature_to_models'])
    print(f'  Unique features referenced: {total_feats}')
    for asset, horizons in sorted(manifest['assets'].items()):
        for h, info in sorted(horizons.items(), key=lambda kv: int(kv[0])):
            print(f'  {asset} {h}h: {len(info["union"])} total ({len(info["direct"])} direct + {len(info["pysr_inputs"])} PySR inputs)')


if __name__ == '__main__':
    main()
