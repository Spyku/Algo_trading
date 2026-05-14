"""tools/feature_stability_test.py — does adding a feature break the D/V pipeline?

Methodological test: if the D/V pipeline is feature-stable, adding an irrelevant
feature should produce ~zero change in backtest return. If σ(return) across
trials is large, the pipeline is over-fitting to feature noise.

What it does:
  - Takes the current production 8h ETH config (RF+LGBM, w=150, gamma=0.995, 17 features)
  - Backtests 11 perturbed variants of that EXACT config on the same 1440h replay:
      * 1 baseline (no perturbation, sanity check)
      * 5 trials adding 1 random-noise feature (different seeds)
      * 5 trials permuting 1 existing feature (information-destroyed)
  - Reports σ(return_pct) across trials at the production confidence threshold (65%)

Verdict thresholds (rule of thumb):
  σ < 2pp   STABLE     — D/V pipeline is feature-robust
  σ < 5pp   MARGINAL   — moderate feature sensitivity
  σ ≥ 5pp   UNSTABLE   — pipeline over-fits to feature noise; structural fix needed

ETA: ~15-30 min total on Desktop (11 subprocesses, ~1-3 min each).

Usage:
  python tools/feature_stability_test.py             default (production cfg, conf=65)
  python tools/feature_stability_test.py --conf 80   use a different confidence
  python tools/feature_stability_test.py --trials 5  fewer noise+permute trials each
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

ENGINE = Path(__file__).resolve().parent.parent
os.chdir(ENGINE)
PY = sys.executable

# Default PROD_CFG = production 8h ETH winner (from models/crypto_ed_production.csv).
# Override via `--csv path/to/x.csv --asset ETH --horizon 8` to test a different
# config CSV (e.g. crypto_ed_production_robust.csv from the robust engine fork).
DEFAULT_PROD_CFG = {
    'combo': 'RF+LGBM',
    'window': 150,
    'gamma': 0.995,
    'features': [
        'pysr_2', 'pysr_1', 'price_to_sma100h', 'deriv_funding_chg1d',
        'oc_mvrv_chg1d', 'adx_14h', 'hour_cos', 'deriv_basis_chg1d',
        'deriv_basis_zscore', 'spread_48h_12h', 'sma20_to_sma50h',
        'deriv_funding_rate', 'pysr_3', 'vol_ratio_12_48',
        'volatility_12h', 'logret_72h', 'logret_8h',
    ],
}

# Module-level placeholder, set by main() after parsing args.
PROD_CFG = DEFAULT_PROD_CFG

# ETH bear horizon=8 confidence per production regime config
DEFAULT_CONF = 65
REPLAY_HOURS = 1440


def load_cfg_from_csv(csv_path: str, asset: str, horizon: int) -> dict:
    """Read coin/horizon row from a crypto_ed_production*.csv and return a
    PROD_CFG-shaped dict {combo, window, gamma, features}. Raises if no
    matching row is found. If multiple rows match, uses the one with the
    highest combined_score."""
    import csv as _csv
    candidates = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if (row.get('coin', '').strip().upper() == asset.upper() and
                    int(row.get('horizon', 0)) == int(horizon)):
                candidates.append(row)
    if not candidates:
        raise SystemExit(f"  [csv] no row for asset={asset} horizon={horizon} in {csv_path}")
    candidates.sort(key=lambda r: float(r.get('combined_score', 0)), reverse=True)
    best = candidates[0]
    feats = [f.strip() for f in best['optimal_features'].split(',') if f.strip()]
    cfg = {
        'combo': best['best_combo'].strip(),
        'window': int(best['best_window']),
        'gamma': float(best['gamma']),
        'features': feats,
    }
    print(f"  [csv] loaded {asset} {horizon}h from {csv_path}:")
    print(f"        combo={cfg['combo']} window={cfg['window']} gamma={cfg['gamma']:.4f} "
          f"n_features={len(cfg['features'])} (sampler={best.get('sampler', '?')}, "
          f"return={best.get('return_pct', '?')}%)")
    return cfg

# Features to permute (test "information destruction" of important features)
PERMUTE_TARGETS = ['pysr_1', 'pysr_2', 'deriv_basis_chg1d', 'hour_cos', 'logret_8h']

# Runner: subprocess-callable code. Imports engine, applies one perturbation,
# backtests the production cfg, prints JSON result.
RUNNER_CODE = r'''
import sys, json, os, numpy as np
sys.path.insert(0, r"{engine}")
os.chdir(r"{engine}")

PERTURB = os.environ.get("STAB_PERTURB", "none")
TARGET  = os.environ.get("STAB_TARGET", "")
CONF    = int(os.environ.get("STAB_CONF", "65"))
ASSET   = os.environ.get("STAB_ASSET", "ETH")
HORIZON = int(os.environ.get("STAB_HORIZON", "8"))

import crypto_trading_system_ed as eng

_orig_build_features = eng._build_features

def _patched_build_features(df_raw, asset_name="BTC", feature_override=None, horizon=None):
    df, cols = _orig_build_features(df_raw, asset_name, feature_override, horizon)
    if PERTURB == "noise":
        seed = int(TARGET)
        rng = np.random.default_rng(seed)
        df["_noise_test"] = rng.standard_normal(len(df))
        if "_noise_test" not in cols:
            cols.append("_noise_test")
        print(f"[STABILITY] noise feature added (seed={{seed}}); selected={{len(cols)}}")
    elif PERTURB == "permute":
        feat = TARGET
        if feat in df.columns:
            rng = np.random.default_rng(42)
            df[feat] = rng.permutation(df[feat].values)
            print(f"[STABILITY] permuted feature {{feat}}")
        else:
            print(f"[STABILITY] WARN: target {{feat}} not in df.columns")
    else:
        print(f"[STABILITY] baseline (no perturbation)")
    return df, cols

eng._build_features = _patched_build_features

cfg = json.loads(os.environ["STAB_CFG_JSON"])
try:
    result = eng._backtest_one_config(ASSET, HORIZON, "STAB_" + PERTURB + "_" + str(TARGET), cfg, replay_hours=int(os.environ.get("STAB_REPLAY", "{replay}")))
except Exception as e:
    print("STAB_RESULT:" + json.dumps({{"error": str(e)}}))
    sys.exit(0)

if result is None:
    print("STAB_RESULT:" + json.dumps({{"error": "no_signals"}}))
else:
    sim = result.get(f"conf_{{CONF}}", {{}})
    out = {{
        "return_pct": sim.get("return_pct"),
        "trades": sim.get("trades"),
        "win_rate": sim.get("win_rate"),
        "round_trips": sim.get("round_trips"),
        "signals": result.get("signals"),
        "buy_hold": result.get("buy_hold"),
    }}
    print("STAB_RESULT:" + json.dumps(out))
'''.format(engine=str(ENGINE), replay=REPLAY_HOURS)


def run_trial(label: str, perturb: str, target: str, conf: int,
              asset: str = 'ETH', horizon: int = 8) -> dict | None:
    env = os.environ.copy()
    env['STAB_PERTURB'] = perturb
    env['STAB_TARGET'] = str(target)
    env['STAB_CONF'] = str(conf)
    env['STAB_ASSET'] = asset
    env['STAB_HORIZON'] = str(horizon)
    env['STAB_CFG_JSON'] = json.dumps(PROD_CFG)
    env['PYTHONIOENCODING'] = 'utf-8'
    print(f'\n=== {label}  (perturb={perturb}, target={target!r}) ===', flush=True)
    out = subprocess.run(
        [PY, '-c', RUNNER_CODE], env=env, cwd=str(ENGINE),
        capture_output=True, text=True,
    )
    result = None
    for line in out.stdout.splitlines():
        if line.startswith('STAB_RESULT:'):
            result = json.loads(line[len('STAB_RESULT:'):])
            break
    if result is None:
        print('  [FAIL] no STAB_RESULT in subprocess stdout')
        print('  --- tail of stdout ---')
        print(out.stdout[-500:])
        if out.stderr.strip():
            print('  --- tail of stderr ---')
            print(out.stderr[-500:])
        return None
    if 'error' in result:
        print(f'  [FAIL] runner error: {result["error"]}')
        return None
    print(f'  return={result.get("return_pct"):+.2f}%  trades={result.get("trades")}  WR={result.get("win_rate")}%')
    return result


def main():
    global PROD_CFG, PERMUTE_TARGETS
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', type=int, default=DEFAULT_CONF, help=f'confidence threshold (default {DEFAULT_CONF})')
    ap.add_argument('--trials', type=int, default=5, help='number of noise + permute trials each (default 5)')
    ap.add_argument('--csv', default=None, help='alternate production CSV (e.g. models/crypto_ed_production_robust.csv)')
    ap.add_argument('--asset', default='ETH', help='asset symbol (default ETH)')
    ap.add_argument('--horizon', type=int, default=8, help='prediction horizon in hours (default 8)')
    args = ap.parse_args()

    # If --csv provided, override the default PROD_CFG with the matching row.
    if args.csv:
        PROD_CFG = load_cfg_from_csv(args.csv, args.asset, args.horizon)
        # Refresh permute targets to features that actually exist in the loaded
        # config; fall back to first 5 features if the original PERMUTE_TARGETS
        # don't intersect (different feature universe across robust vs prod).
        live_perms = [f for f in PERMUTE_TARGETS if f in PROD_CFG['features']]
        if len(live_perms) < args.trials:
            extra = [f for f in PROD_CFG['features'] if f not in live_perms][:args.trials - len(live_perms)]
            live_perms = (live_perms + extra)[:args.trials]
        PERMUTE_TARGETS = live_perms

    trials = [('baseline', 'none', '')]
    for i in range(args.trials):
        trials.append((f'noise_seed_{42+i}', 'noise', str(42 + i)))
    for feat in PERMUTE_TARGETS[:args.trials]:
        trials.append((f'permute_{feat}', 'permute', feat))

    src_label = f'CSV={args.csv}' if args.csv else 'default (prod 8h ETH)'
    print('=' * 76)
    print(f'  FEATURE STABILITY TEST — {args.asset} {args.horizon}h '
          f'({PROD_CFG["combo"]}, w={PROD_CFG["window"]})')
    print(f'  source: {src_label}')
    print(f'  conf threshold: {args.conf}%   replay: {REPLAY_HOURS}h   trials: {len(trials)}')
    print('=' * 76)
    print(f'  Production features ({len(PROD_CFG["features"])}): {", ".join(PROD_CFG["features"])}')
    print('=' * 76)

    results: dict[str, dict] = {}
    for label, perturb, target in trials:
        r = run_trial(label, perturb, target, args.conf, args.asset, args.horizon)
        if r is not None:
            results[label] = r

    if not results:
        print('\n[ABORT] all trials failed.')
        sys.exit(1)

    print()
    print('=' * 76)
    print('  RESULTS')
    print('=' * 76)
    returns = [r['return_pct'] for r in results.values() if r.get('return_pct') is not None]
    trades = [r['trades'] for r in results.values() if r.get('trades') is not None]
    if not returns:
        print('No returns parsed.')
        sys.exit(1)

    print(f'  Trials with valid results: {len(results)}/{len(trials)}')
    print(f'  Returns: mean={mean(returns):+.2f}%  σ={stdev(returns) if len(returns) > 1 else 0:.2f}pp  '
          f'range=[{min(returns):+.2f}%, {max(returns):+.2f}%]')
    print(f'  Trades : mean={mean(trades):.1f}  range=[{min(trades)}, {max(trades)}]')
    print()

    if 'baseline' in results:
        b = results['baseline']['return_pct']
        print(f'  Baseline (no perturbation): return={b:+.2f}%  trades={results["baseline"]["trades"]}')
        deltas = [(label, r['return_pct'] - b) for label, r in results.items() if label != 'baseline']
        print()
        print('  Δreturn vs baseline (sorted by |Δ|):')
        for label, delta in sorted(deltas, key=lambda x: abs(x[1]), reverse=True):
            print(f'    {label:24s}  Δ = {delta:+6.2f}pp')
        print()

    sigma = stdev(returns) if len(returns) > 1 else 0
    if sigma < 2.0:
        verdict = f'STABLE (σ={sigma:.2f}pp < 2pp) — D/V is feature-robust'
        meaning = 'Adding/removing features changes return by <2pp on average — pipeline is methodologically sound.'
    elif sigma < 5.0:
        verdict = f'MARGINAL (2pp ≤ σ={sigma:.2f}pp < 5pp) — moderate feature sensitivity'
        meaning = ('Pipeline somewhat noise-amplifying. Real signal can still emerge but is hard to '
                   'distinguish from feature-perturbation noise. Consider fixes A/B/C.')
    else:
        verdict = f'UNSTABLE (σ={sigma:.2f}pp ≥ 5pp) — D/V over-fits to feature noise'
        meaning = ('Pipeline is amplifying feature noise. Adding/removing features randomly shifts results '
                   'by >5pp. The D/V methodology cannot reliably distinguish good features from luck — '
                   'structural fix needed before any feature-related decision.')

    print('=' * 76)
    print(f'  VERDICT: {verdict}')
    print('=' * 76)
    print(f'  {meaning}')
    print()
    print('  Reference fixes if UNSTABLE:')
    print('    A. Drop the n_features hard cap in N_FEATURES_RANGE (engine line 431-435)')
    print('    B. Bootstrap-aggregate the importance ranking (purged_cv_split_count partial)')
    print('    C. Replace combined_score with sample-size-robust metric (rolling Sharpe)')
    print('=' * 76)


if __name__ == '__main__':
    main()
