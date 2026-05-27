"""
drive_archeology.py — Download all historical revisions of a file from Google Drive
====================================================================================

Created 2026-05-27 for TODO 0526 data drift archeology.

PURPOSE
-------
Pull every available revision of a Drive-synced file (e.g. derivatives_eth.csv)
so we can reconstruct what the live trader actually saw at past decision times.

Drive's web UI shows up to ~100 versions per file (30-day retention OR 100
versions, whichever expires first). This script automates downloading all of them.

PREREQUISITES
-------------
1. Install packages:
   pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

2. Set up OAuth credentials in Google Cloud Console:
   - Create a project, enable Drive API
   - Configure OAuth consent screen (External + add yourself as test user)
   - Create OAuth 2.0 Client ID (Desktop app type)
   - Download the JSON file
   - Save it to: config/drive_oauth_credentials.json

USAGE
-----
  # Download ALL important files (Priority 1 + 2 from TODO 0526) — default behavior
  python tools/drive_archeology.py

  # Single file:
  python tools/drive_archeology.py --file derivatives_eth.csv

  # Multiple files explicitly:
  python tools/drive_archeology.py --file derivatives_eth.csv --file eth_hourly_data.csv

  # Use a preset:
  python tools/drive_archeology.py --preset important   # default — LIVE model deps
  python tools/drive_archeology.py --preset all         # also includes macro/cross-asset/etc

  # Quick test (5 newest per file):
  python tools/drive_archeology.py --limit 5 --newest-first

  # Just list what's available, don't download:
  python tools/drive_archeology.py --list-only

FIRST RUN
---------
Opens browser for OAuth consent. After "Allow", a token is cached to
config/drive_oauth_token.json. Subsequent runs are fully silent.

If you see "Google hasn't verified this app" → click "Advanced" → "Go to
algo-trading-drive (unsafe)". This is YOUR app accessing YOUR data, perfectly
safe.

OUTPUT STRUCTURE
----------------
data/_archeology/
  derivatives_eth.csv/
    snap_2026-05-27_01-15-23/derivatives_eth.csv  ← revision modified at this UTC time
    snap_2026-05-26_22-00-12/derivatives_eth.csv
    snap_2026-05-26_18-00-45/derivatives_eth.csv
    ...
    _manifest.csv  ← lists all revisions with metadata
"""

import argparse
import io
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

CREDENTIALS_FILE = Path("config/drive_oauth_credentials.json")
TOKEN_FILE = Path("config/drive_oauth_token.json")
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# ============================================================
# File presets (which files to archeology)
# Per TODO 0526 priority list — see docs/STEP_6_ENGINE_REFACTOR.md and TODO.md
# ============================================================
PRESET_IMPORTANT = [
    # Priority 1 — LIVE 5h model dependencies (data drift highest impact)
    "derivatives_eth.csv",
    "eth_hourly_data.csv",
    "onchain_eth.csv",
    # Priority 2 — full reconstruction (model config + regime + PySR)
    "crypto_ed_production.csv",
    "regime_config_ed.json",
    "pysr_ETH_5h.json",
    "pysr_ETH_8h.json",
]

PRESET_ALL = PRESET_IMPORTANT + [
    # Priority 3 — completeness / other models / longer-window analysis
    "macro_daily.csv",
    "cross_asset.csv",
    "stablecoin_flows.csv",
    "fear_greed.csv",
    # Other asset hourly data (for cross-asset reconstruction)
    "btc_hourly_data.csv",
    "sol_hourly_data.csv",
    "link_hourly_data.csv",
    "bnb_hourly_data.csv",
    # Other derivatives files
    "derivatives_btc.csv",
    "derivatives_sol.csv",
    "derivatives_link.csv",
    "derivatives_bnb.csv",
    # Other onchain
    "onchain_btc.csv",
    # Position file (so we can reconstruct trade decisions)
    "position_ed_v2_ETH.json",
    # Signal log (mostly append-only but worth snapshotting in case)
    "signal_log.csv",
]


def _import_google_libs():
    """Lazy import so import errors give a clear message."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        from googleapiclient.errors import HttpError
        return Request, Credentials, InstalledAppFlow, build, MediaIoBaseDownload, HttpError
    except ImportError as e:
        print(f"  [ERROR] Missing package: {e}")
        print("  Install with:")
        print('    "C:/Users/Alex/algo_trading/venv/Scripts/python.exe" -m pip install '
              'google-api-python-client google-auth-httplib2 google-auth-oauthlib')
        sys.exit(1)


def get_drive_service():
    """OAuth flow + Drive API service builder. Caches token to TOKEN_FILE."""
    Request, Credentials, InstalledAppFlow, build, _, _ = _import_google_libs()

    if not CREDENTIALS_FILE.exists():
        print(f"  [ERROR] OAuth credentials not found: {CREDENTIALS_FILE}")
        print("  Run Google Cloud Console setup first (see script header).")
        sys.exit(1)

    creds = None
    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        except Exception as e:
            print(f"  [warn] Cached token unreadable ({e}); will re-auth")

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("  Refreshing expired token...")
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"  [warn] Refresh failed ({e}); will re-auth")
                creds = None
        if not creds:
            print("  Starting OAuth flow — your browser will open.")
            print("  If you see 'Google hasn't verified this app', click")
            print("  Advanced > Go to <project> (unsafe). It's YOUR app, safe.")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        print(f"  Token cached to {TOKEN_FILE}")

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def find_file_by_name(service, filename):
    """Search Drive for files with exact name match. Returns list of file metadata dicts,
    sorted by canonical-path score (production paths first, diagnostic snapshot paths last)."""
    safe_name = filename.replace("'", "\\'")
    query = f"name = '{safe_name}' and trashed = false"
    results = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name, parents, mimeType, modifiedTime, size)",
        pageSize=100,
    ).execute()
    matches = results.get("files", [])

    # Resolve paths and score each match. Production files (data/macro_data/,
    # models/, config/) score positive; diagnostic/snapshot/test-output dirs
    # score negative. Higher score = more likely the canonical production file.
    scored = []
    path_cache = {}
    for m in matches:
        try:
            path = get_file_path(service, m["id"], cache=path_cache)
        except Exception:
            path = m.get("name", "")
        score = _score_path_for_canonicality(path)
        scored.append((score, path, m))
    # Sort by score descending — best (most canonical) match first
    scored.sort(key=lambda x: -x[0])
    return [m for (_, _, m) in scored], [(s, p) for (s, p, _) in scored]


def _score_path_for_canonicality(path):
    """Higher score = more likely the canonical production file.
    Heavily penalize diagnostic snapshot and test-output directories.
    """
    s = 0
    p = path.lower().replace("\\", "/")
    # Heavy penalties: anything that looks like a snapshot/diagnostic/test output
    bad_substrings = [
        "_snapshot_", "_diagnostic_snapshots", "_reliability_",
        "/snap_", "_archeology", "_v2_snapshot",
        "models_g_", "config_g_", "models_h75", "config_h75",
        "models_diag_", "config_diag_", "models_embargo_", "config_embargo_",
        "_h75_wide_", "models_h75_wide", "config_h75_wide",
        "archive/", "_pre_merge", "_pre_h75",
    ]
    for bad in bad_substrings:
        if bad in p:
            s -= 100
    # Strong bonuses for canonical production paths
    canonical_patterns = [
        "/data/macro_data/",   # macro/derivatives/onchain CSVs
        "/data/",              # hourly data CSVs (data/ root)
        "/models/",            # models/crypto_ed_production.csv, models/pysr_*.json
        "/config/",            # config/regime_config_ed.json
    ]
    for good in canonical_patterns:
        if good in p:
            s += 10
    # Penalty for paths with more than 3 underscores in directory names (test dirs tend to be like models_g_desktop_0524_h75)
    if p.count("_") > 4:
        s -= 5
    return s


def get_file_path(service, file_id, cache=None):
    """Walk parent chain to produce a /-separated path. Caches lookups."""
    if cache is None:
        cache = {}
    try:
        meta = service.files().get(fileId=file_id, fields="name, parents").execute()
    except Exception:
        return file_id
    name = meta.get("name", file_id)
    parents = meta.get("parents", [])
    if not parents:
        return name
    parent_id = parents[0]
    if parent_id in cache:
        return f"{cache[parent_id]}/{name}"
    parent_path = get_file_path(service, parent_id, cache)
    cache[parent_id] = parent_path
    return f"{parent_path}/{name}"


def list_revisions(service, file_id):
    """List all revisions of a file. Returns list sorted oldest-first."""
    revisions = []
    page_token = None
    while True:
        resp = service.revisions().list(
            fileId=file_id,
            fields="nextPageToken, revisions(id, modifiedTime, size, mimeType, keepForever)",
            pageSize=200,
            pageToken=page_token,
        ).execute()
        revisions.extend(resp.get("revisions", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    revisions.sort(key=lambda r: r.get("modifiedTime", ""))
    return revisions


def download_revision(service, file_id, revision_id, dest_path):
    """Download a specific revision via direct HTTP using the cached credentials.
    The discovery-based get_media path is unreliable for revisions in v3, so we
    construct the URL manually and use the HTTP authenticator the client built.
    """
    _, _, _, _, MediaIoBaseDownload, HttpError = _import_google_libs()
    try:
        # Method A — try the discovery-based path first
        request = service.revisions().get_media(fileId=file_id, revisionId=revision_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        content = fh.getvalue()
    except (AttributeError, HttpError) as e:
        # Method B — fall back to raw HTTP. The credentials object can authorize requests.
        import requests
        creds = service._http.credentials  # the auth context bound to the service
        if hasattr(creds, "refresh") and not creds.valid:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}/revisions/{revision_id}?alt=media"
        r = requests.get(url, headers={"Authorization": f"Bearer {creds.token}"}, timeout=120)
        r.raise_for_status()
        content = r.content

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(content)
    return len(content)


def process_file(service, filename, output_root, limit=None, newest_first=False, list_only=False, skip_existing=True):
    """Process one file: find it, list revisions, download them.

    Returns a summary dict: {'filename', 'found', 'revisions', 'downloaded', 'skipped', 'failed', 'bytes'}.
    """
    summary = {
        "filename": filename, "found": False, "revisions": 0,
        "downloaded": 0, "skipped": 0, "failed": 0, "bytes": 0,
    }

    print(f"\n{'─' * 70}")
    print(f"  FILE: {filename}")
    print(f"{'─' * 70}")

    matches, scored = find_file_by_name(service, filename)
    if not matches:
        print(f"  [SKIP] No file named '{filename}' found in your Drive.")
        return summary
    summary["found"] = True

    if len(matches) > 1:
        # Heuristic-based selection — prefer canonical paths
        chosen_score = scored[0][0]
        print(f"  {len(matches)} matches for '{filename}'. Picking highest-scored canonical path:")
        for i, (score, path) in enumerate(scored, 1):
            marker = "  ← USING (canonical)" if i == 1 else ""
            if i == 1 and score < 0:
                marker = "  ← USING (best of bad options — no canonical match!)"
            print(f"    [{i}] score={score:+4d}  path={path}{marker}")
        if chosen_score < 0:
            print(f"  [WARNING] All matches are in non-canonical paths. "
                  f"Result may not be the production file.")

    file_id = matches[0]["id"]
    revisions = list_revisions(service, file_id)
    summary["revisions"] = len(revisions)

    if not revisions:
        print(f"  No revisions for '{filename}' (file has no exposed history).")
        return summary

    total_bytes = sum(int(r.get("size", 0)) for r in revisions)
    print(f"  {len(revisions)} revisions, total {total_bytes / 1024 / 1024:.1f} MB if all downloaded")
    print(f"  Range: {revisions[0].get('modifiedTime', '?')} -> {revisions[-1].get('modifiedTime', '?')}")

    if list_only:
        for r in revisions:
            print(f"    {r.get('modifiedTime', '?')}  id={r['id']}  size={int(r.get('size', 0)):,}  "
                  f"keepForever={r.get('keepForever', False)}")
        return summary

    order = revisions if not newest_first else list(reversed(revisions))
    if limit:
        order = order[:limit]
        print(f"  --limit {limit}: downloading {len(order)} of {len(revisions)} revisions")

    out_root = output_root / filename
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for i, rev in enumerate(order, 1):
        rev_id = rev["id"]
        mtime = rev.get("modifiedTime", "unknown")
        stamp = mtime.replace(":", "-").replace("T", "_").split(".")[0]
        dest = out_root / f"snap_{stamp}" / filename

        if skip_existing and dest.exists():
            print(f"    [{i}/{len(order)}] {mtime}  SKIP (already downloaded)")
            summary["skipped"] += 1
            manifest_rows.append({
                "revision_id": rev_id, "modified_time_utc": mtime,
                "size_bytes": dest.stat().st_size, "local_path": str(dest),
                "keep_forever": rev.get("keepForever", False), "status": "skip_existing",
            })
            continue

        try:
            size = download_revision(service, file_id, rev_id, dest)
            print(f"    [{i}/{len(order)}] {mtime}  ({size:,} B) -> {dest.relative_to(output_root)}")
            manifest_rows.append({
                "revision_id": rev_id, "modified_time_utc": mtime,
                "size_bytes": size, "local_path": str(dest),
                "keep_forever": rev.get("keepForever", False), "status": "downloaded",
            })
            summary["downloaded"] += 1
            summary["bytes"] += size
        except Exception as e:
            print(f"    [{i}/{len(order)}] {mtime}  FAILED: {str(e)[:100]}")
            manifest_rows.append({
                "revision_id": rev_id, "modified_time_utc": mtime,
                "size_bytes": 0, "local_path": None,
                "keep_forever": rev.get("keepForever", False),
                "status": "failed", "error": str(e),
            })
            summary["failed"] += 1

    # Write per-file manifest
    import csv
    manifest_path = out_root / "_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        if manifest_rows:
            keys = list({k for r in manifest_rows for k in r.keys()})
            keys.sort(key=lambda k: ("revision_id", "modified_time_utc", "size_bytes",
                                     "local_path", "keep_forever", "status", "error").index(k)
                      if k in ("revision_id", "modified_time_utc", "size_bytes",
                               "local_path", "keep_forever", "status", "error") else 99)
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(manifest_rows)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", action="append", default=None,
                        help="Exact filename to look up (can be repeated). "
                             "If not given, uses --preset (default 'important').")
    parser.add_argument("--preset", choices=["important", "all"], default="important",
                        help="File preset: 'important' = LIVE-model deps only (default), "
                             "'all' = include macro/cross-asset/other-asset files.")
    parser.add_argument("--output", default="data/_archeology",
                        help="Local output root directory (default: data/_archeology)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max revisions per file (default: all). Useful for first test.")
    parser.add_argument("--newest-first", action="store_true",
                        help="Download newest revisions first (default: oldest first)")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list revisions, don't download anything")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Re-download even if a snapshot file already exists locally")
    args = parser.parse_args()

    # Determine file list
    if args.file:
        files = args.file
    elif args.preset == "all":
        files = PRESET_ALL
    else:
        files = PRESET_IMPORTANT

    print("=" * 70)
    print(f"  Drive Archeology — {len(files)} file(s) to process")
    if not args.file:
        print(f"  Preset: {args.preset}")
    print(f"  Output root: {args.output}/")
    if args.limit:
        print(f"  Per-file limit: {args.limit} revisions")
    if args.list_only:
        print(f"  Mode: LIST ONLY (no downloads)")
    print("=" * 70)

    print("\n[1/2] Authenticating with Drive...")
    service = get_drive_service()
    print("  Authenticated OK")

    print(f"\n[2/2] Processing {len(files)} file(s)...")
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for fname in files:
        try:
            s = process_file(
                service, fname, output_root,
                limit=args.limit, newest_first=args.newest_first,
                list_only=args.list_only,
                skip_existing=not args.no_skip_existing,
            )
            summaries.append(s)
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")
            summaries.append({"filename": fname, "found": False, "error": str(e)})

    # Final summary
    print("\n" + "=" * 70)
    print("  OVERALL SUMMARY")
    print("=" * 70)
    print(f"  {'file':<35} {'rev':>5} {'dl':>5} {'skip':>5} {'fail':>5} {'MB':>8}")
    print(f"  {'─' * 35} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 8}")
    total = {"revisions": 0, "downloaded": 0, "skipped": 0, "failed": 0, "bytes": 0, "files_found": 0}
    for s in summaries:
        if not s.get("found"):
            print(f"  {s['filename']:<35}  not found in Drive")
            continue
        total["files_found"] += 1
        total["revisions"] += s.get("revisions", 0)
        total["downloaded"] += s.get("downloaded", 0)
        total["skipped"] += s.get("skipped", 0)
        total["failed"] += s.get("failed", 0)
        total["bytes"] += s.get("bytes", 0)
        print(f"  {s['filename']:<35} {s['revisions']:>5} {s['downloaded']:>5} "
              f"{s['skipped']:>5} {s['failed']:>5} {s['bytes']/1024/1024:>8.1f}")
    print(f"  {'─' * 35} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 8}")
    print(f"  {'TOTAL ('+str(total['files_found'])+' files)':<35} {total['revisions']:>5} "
          f"{total['downloaded']:>5} {total['skipped']:>5} {total['failed']:>5} "
          f"{total['bytes']/1024/1024:>8.1f}")
    print(f"\n  Output: {output_root.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
