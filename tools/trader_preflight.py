"""
Trader pre-flight integrity check.

Validates that for every currently-enabled (asset, horizon) pair in
regime_config_ed.json:
  1. The feature manifest lists its required features.
  2. Every required feature maps to a known data source (feature_sources.json).
  3. Every source file exists and is fresh per its max_age_hours SLA.
  4. Every required feature is present in the current build_all_features output.
  5. No required feature has NaN in the last 48 bars (except sparse-by-design).

Output: prints a report. Exit code 0 if all good, 1 if any required check fails.
Unused features/assets are NOT validated (they can't freeze the trader).

Usage:
  python tools/trader_preflight.py                  # scan enabled assets
  python tools/trader_preflight.py --asset ETH      # force-check one asset
  python tools/trader_preflight.py --json           # machine-readable output
  python tools/trader_preflight.py --regenerate-manifest  # regen manifest first
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

# Force UTF-8 output even on Windows cp1252 consoles
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

MANIFEST_PATH = os.path.join(ENGINE_DIR, 'config', 'feature_manifest.json')
SOURCES_PATH = os.path.join(ENGINE_DIR, 'config', 'feature_sources.json')
REGIME_CONFIG = os.path.join(ENGINE_DIR, 'config', 'regime_config_ed.json')


def _resolve_source(feature_name, asset, horizon, sources_cfg):
    """Find the source-file rule that matches a given feature name."""
    for rule in sources_cfg['rules']:
        for prefix in rule.get('match_prefix', []):
            if feature_name.startswith(prefix):
                return _materialize_rule(rule, asset, horizon)
    return _materialize_rule(sources_cfg['default'], asset, horizon)


def _materialize_rule(rule, asset, horizon):
    r = dict(rule)
    asset_lower = asset.lower() if asset else ''
    asset_upper = asset.upper() if asset else ''
    if 'source_file' in r:
        r['source_file'] = r['source_file'].replace('{asset}', asset_lower).replace('{ASSET}', asset_upper).replace('{H}', str(horizon))
    return r


def _file_age_hours(path):
    """Return age of last row for a CSV, else file mtime. None if missing."""
    full = os.path.join(ENGINE_DIR, path) if not os.path.isabs(path) else path
    if not os.path.exists(full):
        return None, 'missing'
    if full.endswith('.csv'):
        try:
            import pandas as pd
            df = pd.read_csv(full, nrows=None, usecols=lambda c: c in ('datetime', 'date', 'timestamp'))
            if 'datetime' in df.columns:
                last_dt = pd.to_datetime(df['datetime'].iloc[-1])
                if last_dt.tzinfo is None:
                    last_dt = last_dt.tz_localize('UTC')
                now = datetime.now(timezone.utc)
                age = (now - last_dt.to_pydatetime()).total_seconds() / 3600
                return age, 'csv_last_row'
            elif 'date' in df.columns:
                last_dt = pd.to_datetime(df['date'].iloc[-1])
                if last_dt.tzinfo is None:
                    last_dt = last_dt.tz_localize('UTC')
                now = datetime.now(timezone.utc)
                age = (now - last_dt.to_pydatetime()).total_seconds() / 3600
                return age, 'csv_last_row'
        except Exception:
            pass
    # Fallback to mtime
    mtime = datetime.fromtimestamp(os.path.getmtime(full), tz=timezone.utc)
    now = datetime.now(timezone.utc)
    age = (now - mtime).total_seconds() / 3600
    return age, 'file_mtime'


def _rows_last_24h(path):
    full = os.path.join(ENGINE_DIR, path) if not os.path.isabs(path) else path
    if not os.path.exists(full):
        return None
    try:
        import pandas as pd
        df = pd.read_csv(full)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            now = datetime.now(timezone.utc)
            if df['datetime'].iloc[-1].tzinfo is None:
                cutoff = pd.Timestamp.now() - pd.Timedelta(hours=24)
            else:
                cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=24)
            return int((df['datetime'] >= cutoff).sum())
    except Exception:
        pass
    return None


def preflight(asset=None, verbose=True):
    report = {'ok': True, 'checks': [], 'failures': [], 'warnings': []}

    # Load configs
    if not os.path.exists(MANIFEST_PATH):
        report['failures'].append(f'MANIFEST MISSING: {MANIFEST_PATH}. Run tools/generate_feature_manifest.py first.')
        report['ok'] = False
        return report
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    with open(SOURCES_PATH) as f:
        sources_cfg = json.load(f)

    # Determine which (asset, horizon) pairs are enabled
    enabled_pairs = []
    if asset:
        # Check all horizons in manifest for this asset
        horizons = sorted(manifest['assets'].get(asset, {}).keys(), key=int)
        for h in horizons:
            enabled_pairs.append((asset, int(h)))
    else:
        # Read regime config, find enabled assets, extract their active horizons
        with open(REGIME_CONFIG) as f:
            regime_cfg = json.load(f)
        for a, cfg in regime_cfg.items():
            if a.startswith('_') or not isinstance(cfg, dict):
                continue
            if not cfg.get('enabled'):
                continue
            # Collect bull + bear horizons
            bull = cfg.get('bull', {}).get('horizon')
            bear = cfg.get('bear', {}).get('horizon')
            for h in (bull, bear):
                if h is not None:
                    enabled_pairs.append((a, int(h)))

    if not enabled_pairs:
        report['warnings'].append('No enabled (asset, horizon) pairs found — nothing to validate.')
        return report

    if verbose:
        print(f'\n{"="*72}')
        print(f'  TRADER PRE-FLIGHT — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        print(f'{"="*72}')
        print(f'\n  Enabled (asset, horizon) pairs: {enabled_pairs}')

    # Lazy-load build_all_features only once if needed
    df_cache = {}
    def get_built_df(a):
        if a not in df_cache:
            from crypto_trading_system_ed import load_data, build_all_features
            try:
                from crypto_live_trader_ed import _compute_pysr_features
            except ImportError:
                _compute_pysr_features = None
            raw = load_data(a)
            if raw is None:
                df_cache[a] = None
                return None
            df, cols = build_all_features(raw, asset_name=a, horizon=5, verbose=False)
            if _compute_pysr_features:
                try:
                    _compute_pysr_features(df, cols, a, 5, verbose=False)
                except Exception:
                    pass
            df_cache[a] = df
        return df_cache[a]

    # Validate each pair
    for a, h in enabled_pairs:
        pair_label = f'{a} {h}h'
        if verbose:
            print(f'\n  --- {pair_label} ---')

        info = manifest['assets'].get(a, {}).get(str(h))
        if not info:
            msg = f'{pair_label}: NO MANIFEST ENTRY (not in crypto_ed_production.csv?)'
            report['failures'].append(msg)
            if verbose:
                print(f'    ✗ {msg}')
            continue

        required = info['union']
        direct = info['direct']
        if verbose:
            print(f'    Required: {len(required)} features ({len(direct)} direct + {len(info["pysr_inputs"])} PySR inputs)')

        # 1) Check each required feature's source file
        sources_seen = {}
        for feat in required:
            rule = _resolve_source(feat, a, h, sources_cfg)
            src = rule.get('source_file', '')
            sources_seen.setdefault(src, []).append((feat, rule))

        for src, feats_rules in sources_seen.items():
            rule = feats_rules[0][1]
            feats = [fr[0] for fr in feats_rules]
            if rule.get('source_kind') == 'pysr_json':
                if not os.path.exists(os.path.join(ENGINE_DIR, src)):
                    msg = f'{pair_label}: PySR JSON missing: {src} (needed for {len(feats)} pysr_* features)'
                    report['failures'].append(msg)
                    if verbose:
                        print(f'    ✗ {msg}')
                else:
                    if verbose:
                        print(f'    ✓ {src} (pysr formulas exist)')
                continue

            if rule.get('source_kind') == 'snapshot_ring':
                rows = _rows_last_24h(src)
                min_rows = rule.get('min_rows_last_24h', 20)
                if rows is None:
                    msg = f'{pair_label}: {src} missing (needed for {feats[:3]}...)'
                    if rule.get('optional'):
                        report['warnings'].append(msg)
                        if verbose:
                            print(f'    ⚠ {msg}')
                    else:
                        report['failures'].append(msg)
                        if verbose:
                            print(f'    ✗ {msg}')
                elif rows < min_rows:
                    msg = f'{pair_label}: {src} only {rows}/{min_rows} rows in last 24h'
                    report['warnings'].append(msg)
                    if verbose:
                        print(f'    ⚠ {msg}')
                else:
                    if verbose:
                        print(f'    ✓ {src} ({rows} rows in last 24h)')
                continue

            # age-gated check
            age_h, kind = _file_age_hours(src)
            max_age = rule.get('max_age_hours')
            optional = rule.get('optional', False)
            if age_h is None:
                msg = f'{pair_label}: {src} MISSING (needed for: {feats[:3]}{"..." if len(feats)>3 else ""})'
                if optional:
                    report['warnings'].append(msg)
                    if verbose:
                        print(f'    ⚠ {msg}')
                else:
                    report['failures'].append(msg)
                    if verbose:
                        print(f'    ✗ {msg}')
                continue
            if max_age and age_h > max_age:
                msg = f'{pair_label}: {src} {age_h:.1f}h stale (>{max_age}h SLA); features: {feats[:3]}{"..." if len(feats)>3 else ""}'
                if optional:
                    report['warnings'].append(msg)
                    if verbose:
                        print(f'    ⚠ {msg}')
                else:
                    report['failures'].append(msg)
                    if verbose:
                        print(f'    ✗ {msg}')
            else:
                age_str = f'{age_h:.1f}h old' if age_h is not None else '?'
                max_str = f'≤{max_age}h' if max_age else 'no SLA'
                if verbose:
                    print(f'    ✓ {src} ({age_str}, {max_str}) — {len(feats)} feat(s)')

        # 2) Check each required feature is in build_all_features output
        df = get_built_df(a)
        if df is None:
            msg = f'{pair_label}: build_all_features returned None (data load failed)'
            report['failures'].append(msg)
            if verbose:
                print(f'    ✗ {msg}')
            continue

        missing = [f for f in required if f not in df.columns]
        if missing:
            msg = f'{pair_label}: {len(missing)} required feature(s) NOT in current build: {missing[:5]}{"..." if len(missing)>5 else ""}'
            report['failures'].append(msg)
            if verbose:
                print(f'    ✗ {msg}')

        # 3) NaN-in-tail check (last 48 bars)
        tail_window = 48
        nan_feats = []
        for f in required:
            if f not in df.columns:
                continue
            nan_count = int(df[f].tail(tail_window).isna().sum())
            if nan_count > 0:
                nan_feats.append((f, nan_count))
        if nan_feats:
            # Only fail for direct features (pysr inputs already imputed in practice)
            direct_nan = [(f, n) for f, n in nan_feats if f in direct]
            other_nan = [(f, n) for f, n in nan_feats if f not in direct]
            if direct_nan:
                msg = f'{pair_label}: {len(direct_nan)} direct feature(s) NaN in tail: {[(f, n) for f, n in direct_nan[:3]]}'
                report['failures'].append(msg)
                if verbose:
                    print(f'    ✗ {msg}')
            if other_nan and verbose:
                print(f'    ⚠ {len(other_nan)} PySR-input feature(s) NaN in tail (will be imputed): {[(f, n) for f, n in other_nan[:3]]}')

        report['checks'].append({'pair': pair_label, 'required': len(required), 'missing': len(missing), 'nan_tail_direct': len([f for f, n in nan_feats if f in direct])})

    report['ok'] = len(report['failures']) == 0

    if verbose:
        print(f'\n{"="*72}')
        if report['ok']:
            print(f'  ✅ PRE-FLIGHT PASSED — {len(enabled_pairs)} pair(s) healthy')
        else:
            print(f'  ❌ PRE-FLIGHT FAILED — {len(report["failures"])} failure(s)')
            for msg in report['failures']:
                print(f'     ✗ {msg}')
        if report['warnings']:
            print(f'  ⚠ {len(report["warnings"])} warning(s)')
            for msg in report['warnings']:
                print(f'     ⚠ {msg}')
        print(f'{"="*72}')

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', default=None, help='Force-check a specific asset (ignores regime_config enabled flag)')
    ap.add_argument('--json', action='store_true', help='Machine-readable output')
    ap.add_argument('--regenerate-manifest', action='store_true', help='Regen manifest from prod CSV first')
    args = ap.parse_args()

    if args.regenerate_manifest:
        subprocess.check_call([sys.executable, os.path.join(ENGINE_DIR, 'tools', 'generate_feature_manifest.py')])

    report = preflight(asset=args.asset, verbose=not args.json)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
