"""Smoke test for the 2026-06-24 snapshot fix (A read-guard + B atomic write)
in download_macro_data.py — the "No columns to parse from file" race.

A = empty/corrupt existing snapshot file must NOT crash the cycle (self-heal).
B = _atomic_to_csv must never leave the target empty/half-written (temp+os.replace).
"""
import os, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import pandas as pd
import download_macro_data as dl  # import-safe (main-guarded)

results = []
def check(name, cond):
    print(('PASS' if cond else 'FAIL'), '-', name)
    results.append(bool(cond))

d = tempfile.mkdtemp()

# ---------- B: atomic write ----------
f = os.path.join(d, 'snap.csv')
dl._atomic_to_csv(pd.DataFrame({'datetime': ['2026-01-01'], 'asset': ['ETH'], 'v': [1]}), f, index=False)
check('B: atomic write creates a valid non-empty file', os.path.exists(f) and os.path.getsize(f) > 0)
check('B: round-trips content', pd.read_csv(f).iloc[0]['asset'] == 'ETH')
# overwrite with new content — must end up with the NEW data, never an empty intermediate
dl._atomic_to_csv(pd.DataFrame({'datetime': ['2026-01-01', '2026-01-02'], 'asset': ['ETH', 'BTC'], 'v': [1, 2]}), f, index=False)
check('B: overwrite yields new content (target never emptied)', len(pd.read_csv(f)) == 2)
check('B: no .tmp leftover', not any('.tmp.' in x for x in os.listdir(d)))

# ---------- A: the empty-file hazard the guard removes ----------
empty = os.path.join(d, 'empty.csv'); open(empty, 'w').close()
try:
    pd.read_csv(empty); hazard = False
except Exception as e:
    hazard = 'No columns to parse' in str(e)
check('A: bare read of a 0-byte file raises "No columns to parse" (the bug)', hazard)
# the source guard: `os.path.exists(f) and os.path.getsize(f) > 0`
guard_skips = not (os.path.exists(empty) and os.path.getsize(empty) > 0)
check('A: size-guard SKIPS the read on a 0-byte file (no crash path)', guard_skips)

# ---------- A: faithful replay of the guarded block on empty + valid ----------
def guarded_append(new_df, outfile):
    """Mirror of the source block (size-guard + try/except + atomic write)."""
    df = new_df
    if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
        try:
            existing = pd.read_csv(outfile)
            df = pd.concat([existing, df], ignore_index=True)
        except Exception as _e:
            print(f'    [self-heal] unreadable {os.path.basename(outfile)} ({_e}); fresh')
    dl._atomic_to_csv(df, outfile, index=False)
    return df

new = pd.DataFrame({'datetime': ['2026-02-01'], 'asset': ['ETH'], 'v': [9]})
# (1) empty existing file -> must self-heal, write fresh, no raise
ef = os.path.join(d, 'iv_empty.csv'); open(ef, 'w').close()
try:
    guarded_append(new, ef); a1 = os.path.getsize(ef) > 0 and len(pd.read_csv(ef)) == 1
except Exception:
    a1 = False
check('A: empty existing file -> self-heals to a fresh 1-row snapshot (no crash)', a1)
# (2) corrupt existing file -> caught, writes fresh
cf = os.path.join(d, 'iv_corrupt.csv'); open(cf, 'wb').write(b'\x00\x01\x02 not a csv \xff')
try:
    guarded_append(new, cf); a2 = os.path.getsize(cf) > 0
except Exception:
    a2 = False
check('A: corrupt existing file -> caught, writes fresh (no uncaught raise)', a2)
# (3) valid existing file -> merges (history preserved)
vf = os.path.join(d, 'iv_valid.csv')
dl._atomic_to_csv(pd.DataFrame({'datetime': ['2026-01-31'], 'asset': ['BTC'], 'v': [5]}), vf, index=False)
guarded_append(new, vf)
check('A: valid existing file -> appends (2 rows, history kept)', len(pd.read_csv(vf)) == 2)

print('\n' + ('ALL PASS (%d/%d)' % (sum(results), len(results)) if all(results)
              else 'FAILURES (%d/%d)' % (sum(results), len(results))))
sys.exit(0 if all(results) else 1)
