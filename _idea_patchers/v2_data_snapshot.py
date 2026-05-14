"""V2 data snapshot patcher — isolate test reads from live trader writes.

Mechanism:
  V2 orchestrator copies data/ to data/_v2_snapshot_<CID>/ at campaign start.
  Each subprocess sets V2_DATA_SNAPSHOT env var and imports THIS patcher
  BEFORE any mode patcher. The patcher monkey-patches pd.read_csv so reads
  hitting 'data/*' get redirected to 'data/_v2_snapshot_<CID>/*'.

  Trader: completely unaffected. Continues reading/writing live data/ files.
  Test:   reads frozen snapshot via the redirect. No drift, no collision.

Edge cases:
  - If subprocess reads a path already inside the snapshot, no redirect
    happens (prevents infinite recursion / double-redirect).
  - If snapshot file doesn't exist (e.g., trader created a new file after
    snapshot was taken), pass through to live data/. This is rare but
    handled.
  - Non-CSV files (JSON config, .pkl caches) are not redirected — engine
    config loading paths (models/, config/) are handled by --no-persist
    separately.

Env vars:
  V2_DATA_SNAPSHOT  absolute path to the snapshot directory (REQUIRED)
"""
import os
import pandas as pd
from pathlib import Path

SNAPSHOT_DIR = os.environ.get('V2_DATA_SNAPSHOT')

if SNAPSHOT_DIR:
    _snapshot = Path(SNAPSHOT_DIR).resolve()
    # Infer engine root = parent of 'data' which is parent of snapshot
    _data_dir = _snapshot.parent.resolve()  # snapshot is INSIDE data/, so parent is data/
    _orig_read_csv = pd.read_csv

    def _redirected_read_csv(filepath_or_buffer, *args, **kwargs):
        try:
            if isinstance(filepath_or_buffer, (str, os.PathLike)):
                p = Path(filepath_or_buffer).resolve()
                try:
                    rel = p.relative_to(_data_dir)
                except ValueError:
                    return _orig_read_csv(filepath_or_buffer, *args, **kwargs)
                rel_str = str(rel)
                # Skip if already inside snapshot (recursion guard)
                if rel_str.startswith('_v2_snapshot_'):
                    return _orig_read_csv(filepath_or_buffer, *args, **kwargs)
                redirected = _snapshot / rel
                if redirected.exists():
                    return _orig_read_csv(str(redirected), *args, **kwargs)
        except Exception:
            pass
        return _orig_read_csv(filepath_or_buffer, *args, **kwargs)

    pd.read_csv = _redirected_read_csv
    print(f'[V2_SNAPSHOT] pd.read_csv redirected: data/<file> -> {_snapshot.name}/<file>')
else:
    print('[V2_SNAPSHOT] V2_DATA_SNAPSHOT not set — no redirect (test running in legacy mode)')
