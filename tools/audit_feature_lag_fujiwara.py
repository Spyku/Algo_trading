"""audit_feature_lag_fujiwara.py — per-column lag verification for the SUB-HOURLY
Fujiwara forks (15-min / 30-min candles).

Same idea as tools/audit_feature_lag.py, but the lag is a CALENDAR-DAY shift, so on
sub-hourly bars one day = (24 * PERIODS_PER_HOUR) ROWS, not 24:
    15-min: 1 day = 96 candles   |  30-min: 1 day = 48 candles
Daily sources must show a 1-day (=candles/day) shift; on-chain a 2-day shift; hourly /
price / technical no shift. A violation means a merge key is wrong for sub-hourly bars.

Usage:  python tools/audit_feature_lag_fujiwara.py --candle 15|30 [--asset ETH] [--horizon 8]
Exit 0 = clean, 1 = violations.
"""
import sys, os, argparse, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np


def load_fork(candle):
    os.environ['FAYE_LIBRARY_MODE'] = '1'
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        f'crypto_trading_system_fujiwara_{candle}.py')
    spec = importlib.util.spec_from_file_location(f'fuji{candle}', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build(eng, asset, horizon, daily, oc):
    eng.DAILY_MERGE_LAG_DAYS = daily
    eng.ONCHAIN_MERGE_LAG_DAYS = oc
    raw = eng.load_data(asset)
    df, cols = eng.build_all_features(raw, asset_name=asset, horizon=horizon, verbose=False)
    eng._compute_pysr_features(df, cols, asset, horizon, verbose=False)
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candle', type=int, required=True, choices=(15, 30))
    ap.add_argument('--asset', default='ETH')
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    eng = load_fork(args.candle)
    CPD = 24 * eng.PERIODS_PER_HOUR        # candles per calendar day
    SHIFTS = [0, CPD, 2 * CPD]             # 0d / 1d / 2d in candle rows
    print(f"FUJIWARA_{args.candle}: candles/day={CPD}  shifts(rows)={SHIFTS}")
    print(f"Building {args.asset} {args.horizon}p with lag (1/2 days) and without (0/0)...")
    L = build(eng, args.asset, args.horizon, 1, 2)
    U = build(eng, args.asset, args.horizon, 0, 0)

    def expected_days(col):
        if col.startswith('oc_'): return 2
        if col.startswith(('m_', 'vix_', 'fg_', 'stable_')): return 1
        if col.startswith('xa_'): return 0 if '_lag' in col else 1
        if col.startswith('deriv_'): return 0
        if col in ('ob_imbalance', 'spread_bps', 'avg_iv', 'iv_skew'): return 0
        if col.startswith('pysr_'): return None
        return 0

    skip = {'datetime', 'open', 'high', 'low', 'close', 'volume', 'label', 'future_return',
            '_forward_return', '_merge_date', '_merge_dt', '_merge_date_oc',
            '_funding_rate', '_led_logret_1h'}
    cols = [c for c in L.columns if c not in skip and c in U.columns and L[c].dtype != 'O']

    def best_shift(col):
        a = L[col]; out = {}
        for d, N in zip((0, 1, 2), SHIFTS):
            b = U[col].shift(N)
            m = a.notna() & b.notna()
            out[d] = float((a[m] == b[m]).mean()) if m.sum() >= 200 else None
        valid = {k: v for k, v in out.items() if v is not None}
        det = max(valid, key=valid.get) if valid else None
        return det, valid, out

    violations = []
    for c in sorted(cols):
        det, valid, out = best_shift(c)
        e = expected_days(c)
        trip = "/".join(f"{(out[d] if out[d] is not None else float('nan')):.2f}" for d in (0, 1, 2))
        if not valid:
            status = "SPARSE(n<200)"
        elif e is None:
            inherits = valid.get(0, 1.0) < 0.999
            status = "ok(pysr-inherits-daily)" if inherits else "ok(pysr-price-only)"
        else:
            ok = (det == e and valid[det] > 0.95)
            status = f"ok({det}d)" if ok else f"VIOLATION(det={det}d exp={e}d)"
            if not ok: violations.append(c)
        if args.verbose or status.startswith("VIOLATION"):
            print(f"  {c:34s} exp={str(e)+'d':4s} 0/1/2d={trip:17s} {status}")

    print(f"\n{len(cols)} columns audited. VIOLATIONS: {len(violations)}")
    if violations:
        print("  " + ", ".join(violations))
        sys.exit(1)
    print("  All columns carry their expected lag (in candle-day units). Clean.")
    sys.exit(0)


if __name__ == '__main__':
    main()
