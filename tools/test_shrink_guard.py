"""Regression test for the _atomic_to_csv history shrink-guard (2026-06-29).
Proves the cross_asset(1637->9) / orderbook(1351->34) clobber class is structurally blocked
at the single write chokepoint — and that legitimate writes (append, small dedup, explicit
shrink, new file) still pass.  Run: python tools/test_shrink_guard.py
"""
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402
import download_macro_data as D  # noqa: E402


def _df(n):
    return pd.DataFrame({'datetime': range(n), 'v': range(n)})


def main():
    fails = 0
    alerts = []
    D._alert_partial_download = lambda *a, **k: alerts.append((a, k))  # stub: no Telegram in tests
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'hist.csv')
        D._atomic_to_csv(_df(200), f, index=False)                 # seed 200 rows
        assert D._count_csv_data_rows(f) == 200, 'seed row-count'

        cases = []
        # (label, df_rows, allow_shrink, expect_written, expect_rows_after)
        cases.append(('clobber 10<<200 REFUSED',        10,  False, False, 200))
        cases.append(('append 210>200 written',         210, False, True,  210))
        cases.append(('small dedup 205 (2%) allowed',   205, False, True,  205))
        cases.append(('allow_shrink=True 5 written',    5,   True,  True,  5))
        for label, n, allow, exp_written, exp_rows in cases:
            got = D._atomic_to_csv(_df(n), f, allow_shrink=allow, index=False)
            rows = D._count_csv_data_rows(f)
            ok = (bool(got) == exp_written) and (rows == exp_rows)
            print(f"{'PASS' if ok else 'FAIL'}: {label}  (written={got}, rows_now={rows})")
            fails += (not ok)

        g = os.path.join(d, 'new.csv')                              # new file (no history)
        got = D._atomic_to_csv(_df(3), g, index=False)
        ok = (got is True) and (D._count_csv_data_rows(g) == 3)
        print(f"{'PASS' if ok else 'FAIL'}: new file (no history) written  (written={got})")
        fails += (not ok)

    ok = len(alerts) == 1                                          # only the clobber should alert
    print(f"{'PASS' if ok else 'FAIL'}: exactly 1 critical alert fired (the clobber)  (alerts={len(alerts)})")
    fails += (not ok)

    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
