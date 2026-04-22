"""
Feature audit: counts usage of every feature across current production models
and PySR formulas. Outputs a grade (5=top, 1=dead) for each feature.

Usage:
  python tools/audit_features.py                    # full audit across ETH
  python tools/audit_features.py --asset BTC        # audit against BTC's feature universe
  python tools/audit_features.py --export out.csv   # save per-feature table

Grading (based on selection rate across N models):
  5 = selected in ≥60% of models  (top-tier signal)
  4 = 30-60%
  3 = 10-30%
  2 = 1-10%
  1 = 0% (never selected) + not in any PySR formula

Reads:
  - models/crypto_ed_production.csv (optimal_features column)
  - models/pysr_*.json (equation strings)
  - Current feature universe from build_all_features('ETH')
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict, Counter

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

import pandas as pd


PYSR_BUILTINS = {
    'tanh', 'sqrt', 'abs', 'Abs', 'log', 'sin', 'cos', 'exp', 'pow',
    'max', 'min', 'floor', 'ceil', 'round', 'sign',
    'PySRFunction', 'X', 'e', 'pi',
}


def parse_pysr_json(path):
    """Return set of feature identifiers used in any expression of this JSON."""
    with open(path) as f:
        j = json.load(f)
    features = set()
    for expr in j.get('expressions', []):
        eq = expr.get('sympy_format') or expr.get('equation') or ''
        # match identifier tokens but skip builtins/constants
        for tok in re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', eq):
            if tok not in PYSR_BUILTINS:
                features.add(tok)
    return {'asset': j.get('asset'), 'horizon': j.get('horizon'),
            'method': j.get('discovery_method'), 'features': features,
            'n_expr': len(j.get('expressions', []))}


def grade_from_rate(rate_pct):
    if rate_pct >= 60: return 5
    if rate_pct >= 30: return 4
    if rate_pct >= 10: return 3
    if rate_pct >= 1:  return 2
    return 1


def feature_category(name):
    if name.startswith('pysr_'): return 'pysr'
    if name.startswith('oc_'): return 'on-chain'
    if name.startswith('deriv_'): return 'derivatives'
    if name.startswith('m_'): return 'macro'
    if name.startswith('xa_'): return 'cross-asset'
    if name.startswith('fg_'): return 'sentiment'
    if name.startswith('gp_'): return 'geopolitical'
    if name.startswith('stable_'): return 'stablecoin'
    if name.startswith('ob_') or name == 'spread_bps': return 'orderbook'
    if name.startswith('avg_iv') or name.startswith('iv_skew'): return 'options'
    if name.startswith('whale_'): return 'whale'
    return 'technical'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', default='ETH', help='Asset for feature universe (default ETH)')
    ap.add_argument('--export', default='', help='CSV path to save full table')
    args = ap.parse_args()

    # 1) Prod-CSV usage
    prod_csv = os.path.join(ENGINE_DIR, 'models', 'crypto_ed_production.csv')
    df = pd.read_csv(prod_csv)
    n_models = len(df)
    print(f'Production CSV: {prod_csv}')
    print(f'  {n_models} models ({df["coin"].nunique()} assets, horizons {sorted(df["horizon"].unique())})')
    print()

    prod_usage = defaultdict(list)
    prod_ranks = defaultdict(list)
    for _, r in df.iterrows():
        feats = [f.strip() for f in str(r['optimal_features']).split(',') if f.strip()]
        for idx, f in enumerate(feats):
            prod_usage[f].append((r['coin'], int(r['horizon'])))
            prod_ranks[f].append(idx + 1)  # 1-based importance rank

    # 2) PySR usage
    models_dir = os.path.join(ENGINE_DIR, 'models')
    pysr_files = sorted(f for f in os.listdir(models_dir) if f.startswith('pysr_') and f.endswith('.json'))
    pysr_feature_usage = defaultdict(list)  # feat → [(asset,horizon)]
    pysr_summary = []
    for pf in pysr_files:
        info = parse_pysr_json(os.path.join(models_dir, pf))
        pysr_summary.append((pf, info))
        for feat in info['features']:
            pysr_feature_usage[feat].append((info['asset'], info['horizon']))

    print(f'PySR JSONs: {len(pysr_files)}')
    for pf, info in pysr_summary:
        print(f'  {pf}: {info["asset"]} {info["horizon"]}h | method={info["method"]} | n_expr={info["n_expr"]} | uses {len(info["features"])} features')
    print()

    # 3) Current feature universe
    print(f'Loading current feature universe via build_all_features({args.asset!r})...')
    from crypto_trading_system_ed import load_data, build_all_features
    raw = load_data(args.asset)
    _, universe_cols = build_all_features(raw, asset_name=args.asset, horizon=5, verbose=False)
    universe = set(universe_cols)
    print(f'  {len(universe)} features in current build')
    print()

    # 4) Build unified feature list
    all_features = universe | set(prod_usage.keys()) | set(pysr_feature_usage.keys())

    table = []
    for feat in sorted(all_features):
        prod_count = len(prod_usage.get(feat, []))
        prod_rate = 100 * prod_count / max(n_models, 1)
        pysr_count = len(pysr_feature_usage.get(feat, []))
        grade = grade_from_rate(prod_rate)
        # boost grade if feature appears in PySR (it's being used indirectly)
        if grade == 1 and pysr_count > 0:
            grade = 2  # resurrected via PySR
        in_universe = feat in universe
        avg_rank = (sum(prod_ranks.get(feat, [0])) / max(len(prod_ranks.get(feat, [])), 1)) if prod_count > 0 else None
        table.append({
            'feature': feat,
            'category': feature_category(feat),
            'prod_count': prod_count,
            'prod_rate_pct': round(prod_rate, 1),
            'avg_rank': round(avg_rank, 1) if avg_rank is not None else None,
            'pysr_count': pysr_count,
            'in_current_universe': in_universe,
            'grade': grade,
        })

    t = pd.DataFrame(table).sort_values(['grade', 'prod_rate_pct'], ascending=[False, False])

    # Summary by grade
    print('=' * 90)
    print(f'  Feature grades (across {n_models} production models)')
    print('=' * 90)
    for grade in [5, 4, 3, 2, 1]:
        rows = t[t['grade'] == grade]
        if grade >= 4:
            print(f'\n-- Grade {grade}: {len(rows)} features --')
            for _, row in rows.iterrows():
                pysr_tag = f" +PySR×{row['pysr_count']}" if row['pysr_count'] else ""
                rank_tag = f" avg_rank={row['avg_rank']}" if row['avg_rank'] else ""
                univ_tag = "" if row['in_current_universe'] else " [NOT in current build]"
                print(f'  {row["feature"]:<35} {row["category"]:<12} {row["prod_count"]:>2}/{n_models} ({row["prod_rate_pct"]:>4.1f}%){rank_tag}{pysr_tag}{univ_tag}')
        else:
            # Compact output for 1-3
            by_cat = rows.groupby('category').size().sort_values(ascending=False)
            print(f'\n-- Grade {grade}: {len(rows)} features --')
            for cat, n in by_cat.items():
                examples = list(rows[rows['category'] == cat]['feature'].head(4))
                extra = f' ...+{n-len(examples)} more' if n > len(examples) else ''
                print(f'  {cat:<12} ({n:>3}): {", ".join(examples)}{extra}')

    # Dead feature identification
    print()
    print('=' * 90)
    print('  DEAD FEATURES (never selected + not in any PySR formula)')
    print('=' * 90)
    dead = t[(t['grade'] == 1) & (t['pysr_count'] == 0) & (t['in_current_universe'])]
    print(f'  {len(dead)} features are in the current build but never selected AND never used by PySR.')
    print(f'  Disabling these would not affect any trained model — free compute + smaller matrix.')
    if len(dead) > 0:
        by_cat = dead.groupby('category').size().sort_values(ascending=False)
        print(f'\n  Dead-feature breakdown by category:')
        for cat, n in by_cat.items():
            print(f'    {cat:<12}: {n}')
        print(f'\n  All dead features (copy-paste ready for config/disabled_features.json):')
        for f in sorted(dead['feature']):
            print(f'    "{f}",')

    # Category summary
    print()
    print('=' * 90)
    print('  CATEGORY SUMMARY (grade distribution)')
    print('=' * 90)
    cat_summary = t.groupby(['category', 'grade']).size().unstack(fill_value=0)
    # Ensure all grade columns
    for g in [5,4,3,2,1]:
        if g not in cat_summary.columns:
            cat_summary[g] = 0
    cat_summary = cat_summary[[5,4,3,2,1]]
    cat_summary['TOTAL'] = cat_summary.sum(axis=1)
    cat_summary = cat_summary.sort_values('TOTAL', ascending=False)
    print(cat_summary.to_string())

    # Orphans: used historically but missing from current build
    orphans = t[(t['prod_count'] > 0) & (~t['in_current_universe'])]
    if len(orphans):
        print()
        print('=' * 90)
        print(f'  ORPHANS — used by trained models but NOT in current build ({len(orphans)} features)')
        print('=' * 90)
        print('  These models will SILENTLY FALL BACK to FEATURE_SET_A at inference time.')
        print('  Worth retraining those models or restoring features.')
        for _, row in orphans.iterrows():
            print(f'    {row["feature"]:<35} used by {row["prod_count"]} model(s)')

    if args.export:
        t.to_csv(args.export, index=False)
        print()
        print(f'Saved full table: {args.export}')


if __name__ == '__main__':
    main()
