"""
Merge OLD PySR expressions (currently in pysr_ETH_<H>h.json) with NEW
expressions (in pysr_ETH_<H>h_new.json) into a single combined JSON.

  - Slots pysr_1..N_old = OLD formulas (preserves existing models' references)
  - Slots pysr_(N_old+1).. = NEW formulas that are NOT duplicates of an OLD
    (or already-kept NEW) formula.

Rationale: instead of swapping OLD -> NEW (which requires retraining all
production models to avoid silent feature drift), we keep OLD as-is and ADD
only the NEW formulas that carry *new information*. LGBM gain ranking + Optuna
n_features then decide which survive the top-N cut.

  *** DEDUP (added 2026-06-06) ***
The original version appended ALL new formulas blindly. That produced an exact
functional duplicate on ETH 5h: pysr_2 ≡ pysr_8 (value-correlation r=1.0000;
their coefficients differ only at the 5th decimal, so a *string* compare misses
it). A duplicate wastes a candidate slot and splits the LGBM gain budget across
two identical columns, which can demote a genuine signal below the top-N cut
(the engine's own merge note even predicted this — "correlated OLD/NEW pairs
will split the gain budget"). We now evaluate every formula's actual values on
real feature data (same sympy.sympify -> lambdify path the engine uses in
_compute_pysr_features) and DROP any NEW formula whose |correlation| with an
already-kept formula is >= DEDUP_CORR_THRESHOLD. OLD formulas are never dropped
(they back existing model references); only redundant NEW ones are skipped.

SAFETY: report-only by default. It prints exactly what it WOULD write and which
NEW formulas it would drop, but does NOT touch any file unless you pass --apply.
This prevents an accidental overwrite of the LIVE models/ PySR files (the live
trader reads them; renumbering a referenced slot = silent feature drift, see
CLAUDE.md Critical Rule 14). Run --apply only as part of a deliberate clean
PySR regen + retrain cycle, with the trader flat.

Backward compatibility:
  - Existing models reference pysr_1..N_old by name; OLD preserved -> unchanged.
  - New training can pick from all surviving slots; LGBM gain decides.

Usage:
  python tools/merge_pysr_old_new.py            # report-only (no writes)
  python tools/merge_pysr_old_new.py --apply     # actually write (use with care)

Backs up OLD JSONs as *_pre_<date>_old_only.json before overwriting (only on --apply).
"""

import json
import os
import shutil
import sys
from datetime import datetime

# Library-mode import so pulling in the engine has zero run-as-main side effects.
os.environ.setdefault('FAYE_LIBRARY_MODE', '1')
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')
os.environ.setdefault('FAYE_MODELS_DIR', 'models')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

MODELS = os.path.join(ROOT, 'models')
HORIZONS = [5, 6, 7, 8]
ASSET = 'ETH'
BACKUP_SUFFIX = f"_pre_{datetime.now().strftime('%Y%m%d')}_dedup_merge"
DEDUP_CORR_THRESHOLD = 0.95   # |corr| >= this vs an already-kept formula -> drop the NEW one
EVAL_TAIL_ROWS = 4000         # window of real feature data used to measure correlation
APPLY = '--apply' in sys.argv


def _expr_series(sympy_str, df):
    """Evaluate one PySR formula on real feature data, mirroring the engine's
    _compute_pysr_features eval path (sympify -> lambdify over numpy). Returns a
    1-D float array aligned to df, or None if a dependency column is missing /
    eval fails (caller treats None as 'cannot compare -> keep')."""
    import numpy as np
    import sympy
    try:
        e = sympy.sympify(sympy_str)
        syms = sorted(e.free_symbols, key=lambda s: s.name)
        names = [s.name for s in syms]
        if any(n not in df.columns for n in names):
            return None
        if not names:  # constant expression
            return np.full(len(df), float(e), dtype=float)
        fn = sympy.lambdify(syms, e, modules=['numpy'])
        vals = np.asarray(fn(*[df[n].values.astype(float) for n in names]), dtype=float)
        if vals.ndim == 0:
            vals = np.full(len(df), float(vals), dtype=float)
        return vals
    except Exception:
        return None


def _build_eval_frame(horizon):
    """Build a real feature frame so formula values (and thus correlations) are
    measured on the same data the engine sees. Returns a DataFrame or None."""
    try:
        import crypto_trading_system_faye as FAYE
        df_raw = FAYE.load_data(ASSET)
        if df_raw is None:
            return None
        df_full, _cols = FAYE.build_all_features(df_raw.tail(EVAL_TAIL_ROWS + 800),
                                                 asset_name=ASSET, horizon=horizon)
        return df_full.tail(EVAL_TAIL_ROWS).reset_index(drop=True)
    except Exception as e:
        print(f"    [warn] could not build feature frame for dedup ({type(e).__name__}: {e}).")
        print(f"           Falling back to NO dedup for this horizon (keeps all NEW).")
        return None


def _max_abs_corr(series, kept_series_list):
    """Max |Pearson corr| of `series` against any series in kept_series_list."""
    import numpy as np
    import pandas as pd
    if series is None:
        return -1.0, None
    best, best_i = -1.0, None
    s = pd.Series(series)
    for i, k in enumerate(kept_series_list):
        if k is None:
            continue
        pair = pd.concat([s, pd.Series(k)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 30:
            continue
        if pair.iloc[:, 0].std() == 0 or pair.iloc[:, 1].std() == 0:
            continue
        r = abs(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        if r > best:
            best, best_i = r, i
    return best, best_i


print(f"PySR merge tool (dedup) — ASSET={ASSET}, HORIZONS={HORIZONS}")
print(f"  models dir   : {MODELS}")
print(f"  dedup thresh : |corr| >= {DEDUP_CORR_THRESHOLD}")
print(f"  mode         : {'APPLY (will write files)' if APPLY else 'REPORT-ONLY (no files written) — pass --apply to write'}")
print()

for h in HORIZONS:
    import glob as _glob
    live_path = os.path.join(MODELS, f'pysr_{ASSET}_{h}h.json')   # WRITE target (the live file)
    new_path = os.path.join(MODELS, f'pysr_{ASSET}_{h}h_new.json')
    backup_path = os.path.join(MODELS, f'pysr_{ASSET}_{h}h{BACKUP_SUFFIX}.json')

    # OLD SOURCE = the true pre-merge original (the *_old_only backup) if present.
    # The live file may ALREADY be a merged 10-formula file, so reading it as
    # "OLD" would double-merge. Prefer the old_only backup as the OLD source.
    old_only = sorted(_glob.glob(os.path.join(MODELS, f'pysr_{ASSET}_{h}h_pre_*old_only*.json')))
    old_src = old_only[-1] if old_only else live_path

    if not os.path.exists(old_src):
        print(f"[{h}h] SKIP — OLD source not found: {old_src}")
        continue
    if not os.path.exists(new_path):
        print(f"[{h}h] SKIP — NEW file not found: {new_path}")
        continue
    print(f"[{h}h] OLD source = {os.path.basename(old_src)}"
          f"{' (true pre-merge original)' if old_only else ' (no old_only backup; using current live file!)'}")

    with open(old_src) as f:
        old_data = json.load(f)
    with open(new_path) as f:
        new_data = json.load(f)

    old_expr = old_data.get('expressions', [])
    new_expr = new_data.get('expressions', [])
    print(f"[{h}h] OLD={len(old_expr)} NEW={len(new_expr)} candidates")

    # --- evaluate formulas on real data for value-based dedup ---
    eval_df = _build_eval_frame(h)
    old_series = [(_expr_series(e.get('sympy_format', e.get('equation', '')), eval_df)
                   if eval_df is not None else None) for e in old_expr]

    kept_expr = list(old_expr)          # OLD always kept (preserves model refs)
    kept_series = list(old_series)
    kept_labels = [f'pysr_{i+1}(OLD)' for i in range(len(old_expr))]
    dropped = []

    for j, e in enumerate(new_expr):
        s = _expr_series(e.get('sympy_format', e.get('equation', '')), eval_df) if eval_df is not None else None
        r, ki = _max_abs_corr(s, kept_series)
        if eval_df is not None and s is not None and r >= DEDUP_CORR_THRESHOLD and ki is not None:
            dropped.append({
                'new_index': j + 1,
                'sympy_format': e.get('sympy_format', e.get('equation', ''))[:90],
                'duplicate_of': kept_labels[ki],
                'corr': round(float(r), 4),
            })
            print(f"       DROP new#{j+1}  (|corr|={r:.4f} vs {kept_labels[ki]})  {e.get('sympy_format','')[:60]}")
        else:
            kept_expr.append(e)
            kept_series.append(s)
            kept_labels.append(f'pysr_{len(kept_expr)}(NEW)')

    n_survivors = len(kept_expr) - len(old_expr)
    canonical_feature_names = new_data.get('feature_names', old_data.get('feature_names', []))
    merged = {
        'asset': old_data.get('asset', ASSET),
        'horizon': h,
        'discovered_at': f"MERGED+DEDUP {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                         f"OLD={old_data.get('discovered_at', '?')} + NEW={new_data.get('discovered_at', '?')}",
        'discovery_method': 'historical',
        'pysr_data_rows': max(old_data.get('pysr_data_rows', 0), new_data.get('pysr_data_rows', 0)),
        'n_expressions': len(kept_expr),
        'feature_names': canonical_feature_names,
        'expressions': kept_expr,
        'merge_metadata': {
            'old_n': len(old_expr),
            'new_n_candidates': len(new_expr),
            'new_n_kept': n_survivors,
            'new_n_dropped': len(dropped),
            'dedup_corr_threshold': DEDUP_CORR_THRESHOLD,
            'old_source': os.path.basename(old_src),
            'new_source': os.path.basename(new_path),
            'old_slots': [f'pysr_{i+1}' for i in range(len(old_expr))],
            'new_slots': [f'pysr_{len(old_expr)+i+1}' for i in range(n_survivors)],
            'dropped_as_duplicate': dropped,
        },
    }

    print(f"       -> OLD {len(old_expr)} kept + NEW {n_survivors} kept "
          f"({len(dropped)} dropped as duplicate) = {len(kept_expr)} total")

    if APPLY:
        # Back up the CURRENT live file before overwriting it.
        if os.path.exists(live_path) and not os.path.exists(backup_path):
            shutil.copyfile(live_path, backup_path)
            print(f"       backup of current live file: {os.path.basename(backup_path)}")
        with open(live_path, 'w') as f:
            json.dump(merged, f, indent=2)
        print(f"       WROTE {os.path.basename(live_path)}")
    else:
        print(f"       (report-only — not written; pass --apply to write)")
    print()

print("DONE.")
if not APPLY:
    print()
    print("REPORT-ONLY: no files changed. Review the DROP lines above, then re-run with --apply")
    print("ONLY as part of a deliberate PySR regen + retrain (trader flat — Critical Rule 19).")
