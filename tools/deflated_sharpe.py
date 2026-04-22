"""
Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) audit tool.

Corrects nominal Sharpe Ratio for multiple testing (we tested N configs, so the best
observed Sharpe is biased upward). Outputs:
  - Nominal Sharpe (SR)
  - Expected max Sharpe under the null (E[SR_max]) given N trials
  - Deflated Sharpe Ratio (DSR): probability that true SR > 0 after multiple-testing correction
  - Probabilistic Sharpe Ratio (PSR): P(true SR > SR_benchmark) at a user-chosen benchmark (default 0)

Interpretation:
  - DSR > 0.95 = strong evidence of real skill (survives multi-testing correction)
  - DSR 0.5 - 0.95 = mixed; could be skill or luck, consider more data
  - DSR < 0.5 = likely noise / overfit to the sweep grid

Usage:
  # Audit today's ETH Mode S winner
  python tools/deflated_sharpe.py --position-file config/position_ed_v2_ETH.json \\
                                   --trials 3920 \\
                                   --days 60

  # Or pass raw Sharpe directly
  python tools/deflated_sharpe.py --sharpe 2.3 --n-observations 39 --trials 3920
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import norm


EULER_GAMMA = 0.5772156649015329


def expected_max_sr(n_trials, sigma_sr=1.0):
    """
    Expected value of the maximum of N i.i.d. standard-normal Sharpe estimates.
    Closed-form approximation (Bailey & Lopez de Prado 2014 eq. 7).
    """
    if n_trials <= 1:
        return 0.0
    z_hi = norm.ppf(1 - 1.0 / n_trials)
    z_lo = norm.ppf(1 - 1.0 / (n_trials * math.e))
    return sigma_sr * ((1 - EULER_GAMMA) * z_hi + EULER_GAMMA * z_lo)


def probabilistic_sr(sr, n_obs, skew=0.0, kurt=3.0, sr_benchmark=0.0):
    """
    Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012).
    P(true SR > sr_benchmark | observed SR = sr, n_obs samples).
    """
    if n_obs < 2:
        return float('nan')
    variance_sr = (1 - skew * sr + ((kurt - 1) / 4.0) * sr**2) / (n_obs - 1)
    if variance_sr <= 0:
        return float('nan')
    z = (sr - sr_benchmark) / math.sqrt(variance_sr)
    return norm.cdf(z)


def deflated_sr(sr, n_obs, n_trials, skew=0.0, kurt=3.0):
    """
    Deflated Sharpe Ratio: PSR with the benchmark set to E[SR_max] from N trials.
    DSR > 0.95 => strong evidence of real skill after multi-testing correction.
    """
    sigma_sr = math.sqrt((1 - skew * sr + ((kurt - 1) / 4.0) * sr**2) / max(n_obs - 1, 1))
    sr_max = expected_max_sr(n_trials, sigma_sr=sigma_sr)
    return probabilistic_sr(sr, n_obs, skew=skew, kurt=kurt, sr_benchmark=sr_max), sr_max


def extract_trade_returns(position_file, days=60):
    """
    Pull per-trade returns from a position_ed_v2_<ASSET>.json file.
    Filters to trades completed within the last `days` days.
    Returns list of pnl_pct values (already in percent).
    """
    with open(position_file) as f:
        p = json.load(f)
    trades = p.get('trades', [])
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    def parse_time(s):
        if not s:
            return None
        s = s.replace('Z', '+00:00').split(' (')[0]
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            try:
                dt = datetime.strptime(s, '%Y-%m-%d %H:%M')
            except ValueError:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    returns = []
    for t in trades:
        if t.get('action') != 'SELL':
            continue
        if t.get('pnl_pct') is None:
            continue
        dt = parse_time(t.get('time'))
        if dt is None or dt < cutoff:
            continue
        returns.append(t['pnl_pct'] / 100.0)  # convert % to fraction
    return returns


def sharpe_from_returns(returns, periods_per_year=None):
    """
    Compute Sharpe from a list of per-trade returns.
    If periods_per_year is given, annualise; otherwise return per-trade Sharpe.
    """
    if len(returns) < 2:
        return 0.0, 0.0, 0.0
    arr = np.asarray(returns)
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0, 0.0, 0.0
    sr = mean / std
    if periods_per_year:
        sr *= math.sqrt(periods_per_year)
    # skew and excess kurtosis (Fisher)
    skew = ((arr - mean) ** 3).mean() / std**3 if std > 0 else 0.0
    kurt = ((arr - mean) ** 4).mean() / std**4 if std > 0 else 3.0  # Pearson kurtosis
    return sr, skew, kurt


def main():
    ap = argparse.ArgumentParser(description='Deflated Sharpe audit (Bailey-Lopez de Prado 2014)')
    ap.add_argument('--position-file', help='Path to position_ed_v2_<ASSET>.json')
    ap.add_argument('--days', type=int, default=60, help='Window for trade filter (default 60)')
    ap.add_argument('--trials', type=int, required=True, help='Number of configs tested in the sweep (e.g. 3920 for ETH Mode S)')
    ap.add_argument('--sharpe', type=float, help='Pre-computed Sharpe (skip position-file parsing)')
    ap.add_argument('--n-observations', type=int, help='Number of observations behind the Sharpe (trades or days)')
    ap.add_argument('--skew', type=float, default=0.0, help='Skew of returns (default 0)')
    ap.add_argument('--kurt', type=float, default=3.0, help='Pearson kurtosis (default 3.0 = Normal)')
    ap.add_argument('--benchmark-sr', type=float, default=0.0, help='Benchmark SR for PSR (default 0)')
    args = ap.parse_args()

    # Gather Sharpe + higher moments
    if args.sharpe is not None:
        if args.n_observations is None:
            print('ERROR: --sharpe requires --n-observations')
            sys.exit(1)
        sr = args.sharpe
        n_obs = args.n_observations
        skew = args.skew
        kurt = args.kurt
        print(f'Using inputs: SR={sr:.3f}, n_obs={n_obs}, skew={skew:.3f}, kurt={kurt:.3f}')
    elif args.position_file:
        returns = extract_trade_returns(args.position_file, days=args.days)
        n_obs = len(returns)
        if n_obs < 2:
            print(f'ERROR: only {n_obs} trades in last {args.days} days; need >= 2')
            sys.exit(1)
        # For per-trade Sharpe (no annualisation). Multi-testing correction is
        # on the same scale regardless.
        sr, skew, kurt = sharpe_from_returns(returns)
        mean_pct = 100.0 * np.mean(returns)
        vol_pct = 100.0 * np.std(returns, ddof=1)
        print(f'Loaded {n_obs} trades from {args.position_file} (last {args.days}d)')
        print(f'  Per-trade: mean={mean_pct:+.3f}%, vol={vol_pct:.3f}%, skew={skew:+.2f}, kurt={kurt:.2f}')
        print(f'  Per-trade Sharpe: {sr:+.3f}')
    else:
        print('ERROR: pass either --position-file or --sharpe')
        sys.exit(1)

    print()
    print('=' * 70)
    print('  DEFLATED SHARPE RATIO AUDIT')
    print('=' * 70)
    print(f'  Sweep size (N trials tested):  {args.trials:,}')
    print(f'  Nominal Sharpe (observed):     {sr:+.3f}')

    psr = probabilistic_sr(sr, n_obs, skew=skew, kurt=kurt, sr_benchmark=args.benchmark_sr)
    dsr, sr_max = deflated_sr(sr, n_obs, args.trials, skew=skew, kurt=kurt)

    print(f'  E[SR_max] under null:          {sr_max:+.3f}   (expected best of {args.trials} random configs)')
    print(f'  Probabilistic SR (vs {args.benchmark_sr}):    {psr:.3f}   (P true SR > {args.benchmark_sr} | observed)')
    print(f'  Deflated SR (multi-test adj):  {dsr:.3f}   (P true SR > E[SR_max] | observed)')
    print()

    # Interpretation
    if dsr >= 0.95:
        verdict = 'STRONG evidence of real skill (survives multi-testing correction)'
    elif dsr >= 0.8:
        verdict = 'MODERATE evidence of skill; worth paper-trading to confirm'
    elif dsr >= 0.5:
        verdict = 'WEAK / ambiguous; could be luck from a big sweep'
    else:
        verdict = 'LIKELY NOISE / overfit; the sweep inflated the best-case result'
    print(f'  Verdict: {verdict}')
    print()

    # Also show: what would DSR be if only 10 configs had been tested?
    if args.trials > 10:
        dsr10, srmax10 = deflated_sr(sr, n_obs, 10, skew=skew, kurt=kurt)
        print(f'  For reference: if only 10 configs tested, DSR would be {dsr10:.3f} (E[SR_max]={srmax10:+.3f})')
        print(f'  => the {args.trials}x sweep costs you ~{100*(psr-dsr):.1f}pp of claimed confidence')


if __name__ == '__main__':
    main()
