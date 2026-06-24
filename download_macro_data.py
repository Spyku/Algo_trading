"""
Download Macro, Sentiment & On-Chain Data (One-Off)
====================================================
Downloads all freely available macro, sentiment, cross-asset, and on-chain data.
Saves to data/macro_data/ folder as CSV files.

Sources:
  - yfinance: VIX, DXY, S&P500, Nasdaq, Gold, US10Y, EUR/USD, BTC correlation pairs
  - alternative.me: Crypto Fear & Greed Index (free, no API key)
  - CoinMetrics Community API: active addresses, hash rate, tx count, MVRV,
    fees, exchange flows (free, no API key)
  - BGeometrics: SOPR (free, no API key, 8 req/hour)

Usage:
  python download_macro_data.py

Adding a new download? See CLAUDE.md "Adding a New Data Source — Checklist".
KEY GOTCHAS: (1) overwrite-safe — route writes through _dedup_preserve_history /
_merge_preserve_history (Rule 22), never a bare to_csv after a full-history pull;
(2) note the cadence (1/day vs hourly) + publish lag — the FEATURE side in
build_all_features must merge daily on _merge_date (lagged) vs hourly on _merge_dt.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create output folder
# Fix #11 (2026-05-29): resolve relative to script location, not CWD.
# Workers spawned via ProcessPoolExecutor occasionally land in a different
# CWD; the relative path 'data/macro_data' then resolves to the wrong place
# and the worker reports "macro_data/ not found". Absolute paths via
# __file__ bypass the CWD-sensitivity. See crypto_trading_system_ed.py Fix #11.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_DIR = os.path.join(_SCRIPT_DIR, 'data', 'macro_data')
os.makedirs(MACRO_DIR, exist_ok=True)

# Freshness threshold — skip re-download if file updated within this many seconds.
# 6h (21600s) is safe because:
#  - macro features are 1d/5d/10d log-changes — within-day staleness invisible to LGBM
#  - on-chain CoinMetrics is daily-frequency anyway
#  - derivatives funding rate updates every 8h
#  - F&G index updates daily
# Bumped 2026-05-02 from 1h. Saves ~3 min per HRST cycle when files are still fresh.
# Live trader's data-freshness gate (Critical Rule 9) is independent and stays at 2h.
FRESHNESS_SECONDS = 21600  # 6 hours


# ============================================================
# Atomic CSV write (B — 2026-06-24, the "No columns to parse" race fix)
# ============================================================
# PROBLEM: a bare `df.to_csv(outfile)` truncates the target to 0 bytes, then
# streams the new contents. A concurrent reader (the other trader instance, or
# Drive mid-sync) that opens the file during that window sees an EMPTY file ->
# pandas raises EmptyDataError "No columns to parse from file", which broke a
# live cycle on 2026-06-24 (IV/orderbook snapshot reads).
#
# FIX: write to a temp file in the same directory, then os.replace() it over the
# target. os.replace is atomic on the same filesystem (Windows + POSIX), so a
# reader sees either the OLD complete file or the NEW complete file — never an
# empty/half-written one. Pair this with the read-side guards (A) so a truly
# corrupt/empty pre-existing file still self-heals instead of crashing.
def _atomic_to_csv(df, outfile, **to_csv_kwargs):
    """Write df to outfile atomically (temp file + os.replace)."""
    tmp = f"{outfile}.tmp.{os.getpid()}"
    try:
        df.to_csv(tmp, **to_csv_kwargs)
        os.replace(tmp, outfile)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise


# ============================================================
# Point-in-time dedup helper (added 2026-05-27 for TODO 0526 data drift fix)
# ============================================================
# PROBLEM: every call site that used `drop_duplicates(keep='last')` or
# `index.duplicated(keep='last')` would SILENTLY OVERWRITE historical rows
# when upstream APIs returned revised values for past dates. This caused
# data drift between live-time and backtest-time — backtest sees corrected
# values that the live trader never saw at decision time, inflating its
# results vs reality.
#
# FIX: this helper splits the dataframe into HISTORICAL rows (older than the
# current hour/day) and CURRENT rows. Historical rows get keep='first' (the
# originally-observed value is frozen). Current rows get keep='last' (allows
# in-progress updates within the current period — e.g. derivatives open
# interest updating multiple times within the current hour before close).
#
# Used by: _yf_merge_with_existing (macro_daily), stablecoin mcap merge,
# options IV snapshot append, orderbook snapshot append.
def _dedup_preserve_history(df, freq='1h', subset=None):
    """Dedup with point-in-time preservation.

    Historical rows (datetime < start of current period) keep='first'.
    Current rows (datetime >= start of current period) keep='last'.

    df: DataFrame to dedup.
        If subset=None, dedup by df.index (must be DatetimeIndex).
        If subset=['col', ...], uses subset[0] as the datetime column.
    freq: '1h' for hourly cutoff, 'D' for daily cutoff.
    """
    import datetime as _dt
    try:
        cutoff = pd.Timestamp(_dt.datetime.utcnow()).floor(freq)
    except Exception:
        # Fallback: treat everything as historical (safest — never overwrites)
        cutoff = pd.Timestamp('2099-01-01')

    if subset is None:
        if not isinstance(df.index, pd.DatetimeIndex):
            # Index isn't datetime — fall back to original keep='last' behavior
            return df[~df.index.duplicated(keep='last')]
        hist_mask = df.index < cutoff
        hist = df.loc[hist_mask]
        curr = df.loc[~hist_mask]
        hist = hist[~hist.index.duplicated(keep='first')]
        curr = curr[~curr.index.duplicated(keep='last')]
        return pd.concat([hist, curr]).sort_index()
    else:
        time_col = subset[0]
        if time_col not in df.columns:
            return df.drop_duplicates(subset=subset, keep='last')
        # Fix (2026-05-29): coerce time_col in-place. Without this, after
        # `pd.concat([read_csv_existing, new_df])`, time_col holds mixed
        # types (strings from CSV + Timestamps from new rows). sort_values
        # then crashes with "'<' not supported between instances of
        # 'Timestamp' and 'str'" — observed killing options_iv and orderbook
        # downloads since 2026-05-27 (stale for 2 days). Coercing the column
        # itself makes the subsequent sort_values pure-Timestamp.
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        hist_mask = df[time_col] < cutoff  # NaT comparisons → False, NaT rows go to current
        hist = df.loc[hist_mask].drop_duplicates(subset=subset, keep='first')
        curr = df.loc[~hist_mask].drop_duplicates(subset=subset, keep='last')
        return pd.concat([hist, curr], ignore_index=True).sort_values(time_col).reset_index(drop=True)


def _merge_preserve_history(new_df, existing_df, freq='1h'):
    """Cell-level merge of new_df with existing_df preserving historical values.

    For rows BEFORE current period: existing values WIN (frozen). Any NaN cells
    in existing get filled from new (covers schema additions — e.g. first time
    a new metric column appears, its historical values come from new).

    For rows IN current period: new values WIN (allows in-progress updates).
    Any NaN cells in new get filled from existing (covers sub-source failures
    where today's download lacks a column).

    Schema-safe: handles columns present in one but not the other via
    pandas combine_first semantics (union of columns + cell-level fill).

    Both DataFrames must have datetime-indexed rows.
    freq: '1h' (derivatives/snapshots) or 'D' (onchain/macro daily).

    Added 2026-05-27 (TODO 0526 data drift fix) — fixes full-file-replace
    downloads (derivatives, onchain) that silently overwrote historical rows
    when upstream APIs returned revised values.
    """
    import datetime as _dt
    try:
        cutoff = pd.Timestamp(_dt.datetime.utcnow()).floor(freq)
    except Exception:
        cutoff = pd.Timestamp('2099-01-01')  # safest fallback: treat all as historical

    if not isinstance(existing_df.index, pd.DatetimeIndex) or not isinstance(new_df.index, pd.DatetimeIndex):
        # Indices not datetime — fall back to plain concat + keep='last' dedup
        combined = pd.concat([existing_df, new_df])
        return combined[~combined.index.duplicated(keep='last')]

    hist_e = existing_df[existing_df.index < cutoff]
    hist_n = new_df[new_df.index < cutoff]
    # Historical: existing values frozen; new only fills cells where existing was NaN
    hist_merged = hist_e.combine_first(hist_n) if len(hist_e) else hist_n

    curr_e = existing_df[existing_df.index >= cutoff]
    curr_n = new_df[new_df.index >= cutoff]
    # Current: new values win; existing only fills cells where new was NaN
    curr_merged = curr_n.combine_first(curr_e) if len(curr_n) else curr_e

    return pd.concat([hist_merged, curr_merged]).sort_index()


def _binance_get(url, ctx, timeout=30, retries=4, backoff=(1, 2, 5, 10), verbose=False):
    """Resilient Binance GET. Retries transient HTTP 400/429/5xx + connection
    errors with exponential backoff. Captures response body on every HTTP
    error so root-cause is visible on persistent failures.

    The trader's morning download burst (funding-rate pagination → OI →
    klines) periodically gets a transient 400 on the first OI request — same
    URL works fine seconds later. Without retry, the entire OI fetch was
    silently failing every day. Retry handles it; body capture makes any
    persistent 4xx (e.g. delisted symbol, schema change) self-explanatory.
    """
    import urllib.request, urllib.error, time as _t
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode('utf-8', errors='replace')[:200]
            except Exception:
                body = ''
            last_err = f'HTTP {e.code} body={body!r}'
            # Retry on transient codes; fail fast on permanent (401/403/404)
            transient = e.code in (400, 408, 425, 429, 500, 502, 503, 504)
            if not transient or attempt == retries - 1:
                raise RuntimeError(last_err) from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = f'{type(e).__name__}: {e}'
            if attempt == retries - 1:
                raise RuntimeError(last_err) from e
        sleep_s = backoff[min(attempt, len(backoff) - 1)]
        if verbose:
            print(f'      [retry {attempt+1}/{retries} after {sleep_s}s] {last_err}')
        _t.sleep(sleep_s)
    raise RuntimeError(last_err or 'unknown error')


def _alert_partial_download(key, msg, severity='warn'):
    """Fix #8 (2026-04-24): lazy-imported Telegram alert for partial-download
    scenarios. Kept as a separate helper to avoid a hard dependency on
    crypto_live_trader_ed at module load (download_macro_data is also used
    standalone as a CLI script)."""
    try:
        from crypto_live_trader_ed import _rate_limited_telegram_lt
        _rate_limited_telegram_lt(key, msg, severity=severity)
    except Exception:
        # Standalone script use / import cycle fallback — at least print it.
        print(f"  [{severity.upper()}] {key}: {msg}")


# ────────────────────────────────────────────────────────────────────────
# Source-health circuit breaker (2026-05-06)
#
# Recurring pain pattern: an upstream provider (Binance, yfinance, Deribit,
# CoinMetrics) silently changes a URL/param/schema. Every cycle until manual
# patch, the trader logs HTTP 400 retry storms. User finds out hours later.
#
# Three-layer defense:
#   (1) fetch_with_fallback() tries a list of URL-variants — if Binance
#       tightens param validation, we already pre-cached alternative forms
#       and the next variant succeeds without code change.
#   (2) Circuit breaker tracks per-source consecutive failures. After
#       _CB_FAIL_THRESHOLD failures the source is MUTED for _CB_MUTE_SEC,
#       so subsequent cycles skip the call entirely (no log noise, no
#       per-cycle retry storm).
#   (3) On state transitions (OK→MUTED, MUTED→OK), send ONE Telegram
#       alert each. Eliminates per-cycle alert spam during outages and
#       gives a clear "recovered" signal when the upstream comes back.
#
# State persists in data/source_health.json so trader restarts don't lose
# tracking. Atomic write via tmp+rename to survive partial writes.
# ────────────────────────────────────────────────────────────────────────

_SOURCE_HEALTH_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'source_health.json'
)
_CB_FAIL_THRESHOLD = 3       # N consecutive failures → MUTE
_CB_MUTE_SEC = 60 * 60       # 1 hour mute on circuit-break


class SourceMuted(Exception):
    """Raised when a source is in the MUTED circuit state.
    Caller should skip the fetch silently for this cycle."""
    pass


def _load_source_health():
    try:
        with open(_SOURCE_HEALTH_FILE, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_source_health(state):
    """Atomic write so a crash mid-write can't leave a corrupt JSON file."""
    try:
        os.makedirs(os.path.dirname(_SOURCE_HEALTH_FILE), exist_ok=True)
        tmp = _SOURCE_HEALTH_FILE + f'.tmp.{os.getpid()}'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp, _SOURCE_HEALTH_FILE)
    except Exception:
        pass  # best-effort; never let health-tracking failure break the fetch


def _is_circuit_muted(name):
    """Returns True if the source is currently MUTED. Auto-clears expired mutes."""
    state = _load_source_health()
    s = state.get(name, {})
    if s.get('circuit_state') != 'MUTED':
        return False
    if time.time() < s.get('mute_until', 0):
        return True
    # Mute window expired — clear it so the next call re-tries the source.
    s['circuit_state'] = 'OK'
    s.pop('mute_until', None)
    state[name] = s
    _save_source_health(state)
    return False


def _record_source_success(name, used_url=None):
    """Record success. If source was previously MUTED, send a recovery alert."""
    state = _load_source_health()
    s = state.get(name, {})
    was_muted = s.get('circuit_state') == 'MUTED'
    had_failures = s.get('consecutive_failures', 0) > 0
    s.update({
        'last_success': time.time(),
        'last_success_iso': datetime.now().isoformat(timespec='seconds'),
        'consecutive_failures': 0,
        'circuit_state': 'OK',
    })
    s.pop('mute_until', None)
    s.pop('last_error', None)
    if used_url:
        s['last_used_url'] = used_url
    state[name] = s
    _save_source_health(state)
    if was_muted:
        _alert_partial_download(
            f'source_recovery_{name}',
            f'✅ Data source RECOVERED: <b>{name}</b>',
            severity='info',
        )
    elif had_failures:
        # Silent recovery from <threshold failures — no alert, just log
        print(f"    [{name}] recovered after transient failure(s)")


def _record_source_failure(name, err):
    """Record failure. At threshold, MUTE the source and send ONE alert."""
    state = _load_source_health()
    s = state.get(name, {})
    s['consecutive_failures'] = s.get('consecutive_failures', 0) + 1
    s['last_error'] = str(err)[:300]
    s['last_failure'] = time.time()
    s['last_failure_iso'] = datetime.now().isoformat(timespec='seconds')
    n_fails = s['consecutive_failures']
    just_muted = (n_fails >= _CB_FAIL_THRESHOLD
                  and s.get('circuit_state') != 'MUTED')
    if just_muted:
        s['circuit_state'] = 'MUTED'
        s['mute_until'] = time.time() + _CB_MUTE_SEC
    state[name] = s
    _save_source_health(state)
    if just_muted:
        mute_min = _CB_MUTE_SEC // 60
        _alert_partial_download(
            f'source_muted_{name}',
            f'🔇 Data source MUTED: <b>{name}</b>\n'
            f'Failures: {n_fails} consecutive\n'
            f'Last error: <code>{s["last_error"][:200]}</code>\n'
            f'Will auto-retry in {mute_min} min',
            severity='warning',
        )


def fetch_with_fallback(name, url_fns, ctx, retries_per_url=2, verbose=False):
    """Robust data fetch with multi-URL fallback + circuit breaker.

    Args:
      name: stable source identifier (e.g. 'oi_ETHUSDT'). Used as the
            circuit-breaker key and Telegram alert key.
      url_fns: ordered list of (callable returning URL string) OR plain URL
            strings. The first that returns data wins. Callables let the
            caller capture fresh timestamps per invocation.
      ctx: ssl context passed to _binance_get.
      retries_per_url: per-URL retries (lower than _binance_get default
            because the outer loop tries multiple URLs anyway).

    Returns:
      (parsed_data, url_used) tuple. parsed_data is the JSON-decoded body.

    Raises:
      SourceMuted: if the source is in MUTED state. Caller should skip.
      RuntimeError: if all URL variants failed (after which the source's
            failure counter is incremented and may transition to MUTED).
    """
    if _is_circuit_muted(name):
        raise SourceMuted(name)
    last_err = None
    for i, fn_or_url in enumerate(url_fns):
        try:
            url = fn_or_url() if callable(fn_or_url) else fn_or_url
            data = _binance_get(url, ctx, retries=retries_per_url,
                                verbose=verbose and i == 0)
            _record_source_success(name, used_url=url)
            if i > 0 and verbose:
                print(f"    [{name}] succeeded on fallback variant {i+1}/{len(url_fns)}")
            return data, url
        except Exception as e:
            last_err = e
            if verbose:
                print(f"    [{name}] variant {i+1}/{len(url_fns)} failed: {str(e)[:120]}")
            continue
    # All variants failed — record + propagate
    _record_source_failure(name, last_err)
    raise RuntimeError(f'{name}: all {len(url_fns)} URL variants failed; '
                       f'last error: {last_err}') from last_err


def get_source_health_summary():
    """Return per-source health for /reload Telegram command + manual inspection."""
    state = _load_source_health()
    out = []
    now = time.time()
    for name, s in sorted(state.items()):
        circuit = s.get('circuit_state', 'OK')
        n_fails = s.get('consecutive_failures', 0)
        if circuit == 'MUTED':
            mute_left = max(0, int((s.get('mute_until', 0) - now) / 60))
            status = f'🔇 MUTED ({mute_left}min left, {n_fails} fails)'
        elif n_fails > 0:
            status = f'⚠️ {n_fails} recent fail(s)'
        else:
            status = '✅ OK'
        last_ok = s.get('last_success_iso', 'never')
        out.append(f'{name:<28} {status:<32} last_ok={last_ok}')
    return '\n'.join(out) if out else '(no sources tracked yet)'


def _is_fresh(filepath, max_age_seconds=FRESHNESS_SECONDS):
    """Return True if file's LAST-ROW datetime is within max_age_seconds.
    Fix #7 (2026-04-24): content-aware, not mtime-aware. Catches partial
    downloads that bumped mtime but didn't advance data, and write-on-empty-
    response cases where the CSV was rewritten with the same last row.

    Looks for a 'datetime' / 'date' / 'timestamp' column (any one suffices).
    Falls back to mtime if no time column is recognizable, so edge-case
    files (snapshot-ring files without a datetime column) still have some
    staleness gate."""
    if not os.path.exists(filepath):
        return False
    try:
        # Efficient: only parse the column we care about
        df = pd.read_csv(filepath, usecols=lambda c: c in ('datetime', 'date', 'timestamp'))
        if len(df) == 0:
            return False
        if 'datetime' in df.columns:
            last = pd.to_datetime(df['datetime'].iloc[-1])
        elif 'date' in df.columns:
            last = pd.to_datetime(df['date'].iloc[-1])
        elif 'timestamp' in df.columns:
            # Binance OHLCV uses ms-epoch timestamp
            last = pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')
        else:
            # No recognizable time column — fall back to mtime
            age = time.time() - os.path.getmtime(filepath)
            return age < max_age_seconds

        # Make both sides tz-aware at UTC for a clean subtraction
        if last.tzinfo is None:
            last = last.tz_localize('UTC')
        age_sec = (pd.Timestamp.now(tz='UTC') - last).total_seconds()
        return age_sec < max_age_seconds
    except Exception:
        # CSV unreadable (corrupted, locked, etc.) — fall back to mtime
        try:
            age = time.time() - os.path.getmtime(filepath)
            return age < max_age_seconds
        except Exception:
            return False


# ============================================================
# 1. YFINANCE MACRO DATA
# ============================================================
# Incremental tail-update window: how many days of overlap to re-pull. Buys
# correction headroom for any prior trailing-NaN ghost that the trim+ffill
# logic might have left behind in the last refresh.
_YF_INCREMENTAL_OVERLAP_DAYS = 7


def _yf_resolve_window(outfile, full):
    """Decide start/end for a yfinance batch.

    full=True  → 2022-01-01 → tomorrow (legacy 3-year pull).
    full=False → max(last_csv_date - 7d, 2022-01-01) → tomorrow when CSV
                 exists and is parseable. Falls back to full pull otherwise.
    Returns (start_date_str, end_date_str, mode_label).
    """
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')  # yfinance `end` is EXCLUSIVE
    if full or not os.path.exists(outfile):
        return '2022-01-01', end_date, 'full'
    try:
        existing = pd.read_csv(outfile, index_col=0, parse_dates=True)
        if len(existing) == 0:
            return '2022-01-01', end_date, 'full'
        last = existing.index.max()
        start = (last - pd.Timedelta(days=_YF_INCREMENTAL_OVERLAP_DAYS)).strftime('%Y-%m-%d')
        return start, end_date, f'incremental (overlap from {start})'
    except Exception as e:
        print(f"  [!] Could not read {outfile} for incremental tail update ({e}); falling back to full pull.")
        return '2022-01-01', end_date, 'full'


def _yf_merge_with_existing(new_df, outfile, mode_label):
    """If running in incremental mode and the existing CSV is present, drop
    rows from existing whose date is in the new slice's range, then concat
    so new fully replaces overlap (including any prior poisoned rows).
    """
    if mode_label == 'full' or not os.path.exists(outfile):
        return new_df
    try:
        existing = pd.read_csv(outfile, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"  [!] Existing {outfile} unreadable during merge ({e}); writing new slice only.")
        return new_df
    if len(existing) == 0 or len(new_df) == 0:
        return new_df if len(new_df) else existing
    # Align column sets — if upstream tickers were added/removed since the
    # existing file was written, fall back to full pull semantics (the new
    # slice alone won't cover history but at least won't smear schemas).
    if set(existing.columns) != set(new_df.columns):
        print(f"  [!] Column mismatch between existing CSV and new slice "
              f"({set(existing.columns) ^ set(new_df.columns)}); writing new slice only.")
        return new_df
    keep = existing.loc[existing.index < new_df.index.min()]
    merged = pd.concat([keep[new_df.columns], new_df]).sort_index()
    # TODO 0526 data drift fix (2026-05-27): was `keep='last'` which let yfinance
    # revisions silently overwrite historical macro values. Now preserves the
    # originally-observed value for past dates; today's row can still update.
    merged = _dedup_preserve_history(merged, freq='D')
    return merged


def download_yfinance_data(full=False):
    """Download macro indicators via yfinance batch download.

    Uses yfinance's native multi-ticker download (`yf.download([list,...])` with
    threads=True default) — yfinance handles internal parallelism safely; an
    external ThreadPoolExecutor caused 2D-shape race conditions in its state.

    full=False (default): incremental tail update — pulls only the last
      ~7 days from yfinance and merges with the existing CSV. ~150× less
      bandwidth than the full pull and dramatically lower rate-limit risk.
    full=True: 3-year refresh from 2022-01-01. Use after corruption is
      detected beyond the 7-day overlap window, or for first-time setup.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  Installing yfinance...")
        os.system(f"{sys.executable} -m pip install yfinance --quiet")
        import yfinance as yf

    # Tickers to download
    tickers = {
        'VIX':      '^VIX',       # CBOE Volatility Index
        'DXY':      'DX-Y.NYB',    # US Dollar Index (DX=F delisted 2026-03-20)
        'SP500':    '^GSPC',       # S&P 500
        'NASDAQ':   '^IXIC',      # Nasdaq Composite
        'GOLD':     'GC=F',       # Gold Futures
        'US10Y':    '^TNX',       # US 10-Year Treasury Yield
        'EURUSD':   'EURUSD=X',   # EUR/USD exchange rate
        'USDJPY':   'JPY=X',      # USD/JPY
        'OIL':      'CL=F',       # Crude Oil Futures
    }

    outfile = os.path.join(MACRO_DIR, 'macro_daily.csv')
    start_date, end_date, mode_label = _yf_resolve_window(outfile, full)

    print(f"\n  Downloading {len(tickers)} macro indicators from yfinance (batch)...")
    print(f"  Period: {start_date} to {end_date} [{mode_label}]")

    # Single batch call — yfinance downloads all tickers internally with threads=True
    ticker_list = list(tickers.values())
    name_for_ticker = {v: k for k, v in tickers.items()}
    raw = yf.download(ticker_list, start=start_date, end=end_date, progress=False, group_by='ticker')

    all_data = {}
    for name, ticker in tickers.items():
        try:
            # yf.download with multiple tickers returns MultiIndex columns: (ticker, field)
            if isinstance(raw.columns, pd.MultiIndex):
                # MultiIndex: get the (ticker, 'Close') column
                if (ticker, 'Close') in raw.columns:
                    series = raw[(ticker, 'Close')].dropna()
                else:
                    print(f"    {name:10s} ({ticker})... NO DATA (ticker not in response)")
                    continue
            else:
                # Single-ticker fallback (shouldn't happen with list input but be safe)
                series = raw['Close'].dropna() if 'Close' in raw.columns else None
                if series is None:
                    print(f"    {name:10s} ({ticker})... NO DATA")
                    continue

            if len(series) == 0:
                print(f"    {name:10s} ({ticker})... NO DATA (empty)")
                continue

            all_data[name] = series
            print(f"    {name:10s} ({ticker})... {len(series)} days")
        except Exception as e:
            print(f"    {name:10s} ({ticker})... ERROR: {e}")

    if not all_data:
        print("  [!!] No macro data downloaded!")
        _alert_partial_download(
            'macro_total_fail',
            "🚨 yfinance macro: ALL 9 tickers failed — macro_daily.csv NOT updated. Check yfinance / network.",
            severity='critical',
        )
        return None

    # Fix #8 (2026-04-24): partial-success alerting
    expected = set(tickers.keys())
    got = set(all_data.keys())
    missing = sorted(expected - got)
    if missing:
        print(f"  [!] Macro partial — missing {missing}. CSV will have fewer columns; m_{{ticker}}_* features absent.")
        _alert_partial_download(
            f'macro_partial_{"_".join(missing)}',
            f"⚠ macro_daily partial download — missing tickers: {missing}. "
            f"Features like m_{{{missing[0].lower()}}}_chg1d etc. will be absent. yfinance outage?",
            severity='warn',
        )

    # Combine into single DataFrame
    macro_df = pd.DataFrame(all_data)
    macro_df.index.name = 'date'
    macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)

    # Merge with existing CSV (incremental mode replaces only the overlap window).
    macro_df = _yf_merge_with_existing(macro_df, outfile, mode_label)

    # Trim trailing rows where US equity tickers (which only print on trading days)
    # are ALL missing — otherwise 24/7 forex tickers extend the index past the last
    # real close and the subsequent ffill duplicates Friday's values onto Sat/Sun/Mon,
    # which then masquerades as fresh data to _file_is_fresh(). ffill internal gaps
    # only (e.g. one ticker missing on a day others have it).
    equity_cols = [c for c in ('SP500', 'NASDAQ', 'VIX') if c in macro_df.columns]
    if equity_cols:
        has_close = macro_df[equity_cols].notna().any(axis=1)
        if has_close.any():
            macro_df = macro_df.loc[:has_close[has_close].index[-1]]
    macro_df = macro_df.ffill()

    # Save
    macro_df.to_csv(outfile)
    print(f"\n  Saved: {outfile} ({len(macro_df)} rows, {len(macro_df.columns)} columns)")
    print(f"  Columns: {list(macro_df.columns)}")
    print(f"  Date range: {macro_df.index[0].date()} to {macro_df.index[-1].date()}")
    print(f"  NaN summary:")
    for col in macro_df.columns:
        nans = macro_df[col].isna().sum()
        if nans > 0:
            print(f"    {col}: {nans} NaN rows")

    return macro_df


# ============================================================
# 2. CRYPTO FEAR & GREED INDEX
# ============================================================
def download_fear_greed():
    """Download Crypto Fear & Greed Index from alternative.me (free, no key)."""
    import urllib.request

    print(f"\n  Downloading Crypto Fear & Greed Index...")
    url = "https://api.alternative.me/fng/?limit=0&format=json"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())

        records = data.get('data', [])
        if not records:
            print("  No Fear & Greed data!")
            return None

        rows = []
        for r in records:
            rows.append({
                'date': pd.to_datetime(int(r['timestamp']), unit='s'),
                'fear_greed': int(r['value']),
                'fear_greed_label': r['value_classification'],
            })

        fg_df = pd.DataFrame(rows)
        fg_df = fg_df.sort_values('date').set_index('date')
        fg_df.index = fg_df.index.tz_localize(None)

        # TODO 0526 data drift fix (2026-05-27): merge with existing file to
        # preserve historical Fear & Greed values. The alternative.me API
        # returns all-history each call; without this merge, any upstream
        # revision to a past day silently overwrites the originally-observed
        # value. Cell-level merge keeps existing historical values frozen and
        # only allows the current day to update.
        outfile = os.path.join(MACRO_DIR, 'fear_greed.csv')
        if os.path.exists(outfile):
            try:
                existing = pd.read_csv(outfile, parse_dates=['date'], index_col='date')
                if existing.index.tz is not None:
                    existing.index = existing.index.tz_localize(None)
                fg_df = _merge_preserve_history(fg_df, existing, freq='D')
            except Exception as e:
                print(f"  [!] Could not merge with existing fear_greed.csv ({e}); writing fresh.")
        fg_df.to_csv(outfile)
        print(f"  Saved: {outfile} ({len(fg_df)} rows)")
        print(f"  Date range: {fg_df.index[0].date()} to {fg_df.index[-1].date()}")
        print(f"  Current: {fg_df.iloc[-1]['fear_greed']} ({fg_df.iloc[-1]['fear_greed_label']})")

        return fg_df

    except Exception as e:
        print(f"  ERROR downloading Fear & Greed: {e}")
        return None


# ============================================================
# 4. CROSS-ASSET PAIRS (for correlation features)
# ============================================================
def download_cross_asset(full=False):
    """Download BTC and major indices for cross-correlation features.

    full=False (default): incremental tail update — last ~7 days only,
      merged with the existing CSV (corrects up to 7 days of past poisoning).
    full=True: 3-year refresh. Use after corruption beyond the overlap window.
    """
    try:
        import yfinance as yf
    except ImportError:
        os.system(f"{sys.executable} -m pip install yfinance --quiet")
        import yfinance as yf

    pairs = {
        'BTC_USD':  'BTC-USD',
        'ETH_USD':  'ETH-USD',
        'NASDAQ':   '^IXIC',
        'SP500':    '^GSPC',
        'DAX':      '^GDAXI',
    }

    outfile = os.path.join(MACRO_DIR, 'cross_asset.csv')
    start_date, end_date, mode_label = _yf_resolve_window(outfile, full)

    print(f"\n  Downloading cross-asset pairs for correlation features (batch)...")
    print(f"  Period: {start_date} to {end_date} [{mode_label}]")
    ticker_list = list(pairs.values())
    raw = yf.download(ticker_list, start=start_date, end=end_date, progress=False, group_by='ticker')

    all_data = {}
    for name, ticker in pairs.items():
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if (ticker, 'Close') in raw.columns:
                    series = raw[(ticker, 'Close')].dropna()
                else:
                    print(f"    {name:10s} ({ticker})... NO DATA (ticker not in response)")
                    continue
            else:
                series = raw['Close'].dropna() if 'Close' in raw.columns else None
                if series is None:
                    print(f"    {name:10s} ({ticker})... NO DATA")
                    continue

            if len(series) == 0:
                print(f"    {name:10s} ({ticker})... NO DATA (empty)")
                continue

            all_data[name] = series
            print(f"    {name:10s} ({ticker})... {len(series)} days")
        except Exception as e:
            print(f"    {name:10s} ({ticker})... ERROR: {e}")

    if all_data:
        # Fix #8 (2026-04-24): partial-success alerting
        expected = set(pairs.keys())
        got = set(all_data.keys())
        missing = sorted(expected - got)
        if missing:
            print(f"  [!] Cross-asset partial — missing {missing}. xa_{{pair}}_* features will be absent.")
            _alert_partial_download(
                f'cross_asset_partial_{"_".join(missing)}',
                f"⚠ cross_asset partial download — missing pairs: {missing}. "
                f"Correlation/relstr features against those pairs will be absent.",
                severity='warn',
            )

        cross_df = pd.DataFrame(all_data)
        cross_df.index.name = 'date'
        cross_df.index = pd.to_datetime(cross_df.index).tz_localize(None)

        # Merge with existing CSV (incremental mode replaces only the overlap window).
        cross_df = _yf_merge_with_existing(cross_df, outfile, mode_label)

        # Trim trailing rows where US equity tickers are ALL NaN — 24/7 BTC/ETH would
        # otherwise extend the index past last real equity close, and the subsequent
        # ffill would duplicate Friday's values forward, masquerading as fresh data
        # to _file_is_fresh(). Internal ffill (per-ticker gaps) still applied.
        equity_cols = [c for c in ('SP500', 'NASDAQ', 'DAX') if c in cross_df.columns]
        if equity_cols:
            has_close = cross_df[equity_cols].notna().any(axis=1)
            if has_close.any():
                cross_df = cross_df.loc[:has_close[has_close].index[-1]]
        cross_df = cross_df.ffill()

        cross_df.to_csv(outfile)
        print(f"\n  Saved: {outfile} ({len(cross_df)} rows)")
        return cross_df

    # Total failure
    _alert_partial_download(
        'cross_asset_total_fail',
        "🚨 cross_asset: ALL pairs failed — file not updated. Check yfinance / network.",
        severity='critical',
    )
    return None


# ============================================================
# 5. ON-CHAIN DATA (Bitcoin blockchain metrics)
# ============================================================
def download_onchain_data(asset='btc'):
    """
    Download on-chain metrics from free APIs (no API key needed).
    Sources:
      - CoinMetrics Community API: active addresses, hash rate, tx count, MVRV,
        fees (native units), exchange inflow/outflow
      - BGeometrics: SOPR (BTC only on free tier — skipped for other assets)
    Saves to data/macro_data/onchain_{asset}.csv (daily frequency).
    """
    import urllib.request
    import urllib.error
    import ssl
    import time

    asset = asset.lower()
    print(f"\n  Downloading {asset.upper()} on-chain data...")

    # SSL fix for Windows
    ctx = ssl._create_unverified_context()

    start_date = '2022-01-01'
    all_data = {}

    # --- CoinMetrics Community API (free, no key) ---
    cm_metrics = 'AdrActCnt,HashRate,TxCnt,CapMVRVCur,FeeTotNtv,FlowInExNtv,FlowOutExNtv'
    cm_url = (
        f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
        f"?assets={asset}&metrics={cm_metrics}&frequency=1d"
        f"&start_time={start_date}&page_size=10000"
    )

    print(f"    CoinMetrics: {cm_metrics.count(',') + 1} metrics...", end=' ')
    try:
        req = urllib.request.Request(cm_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
            data = json.loads(resp.read().decode())

        rows = data.get('data', [])
        if rows:
            cm_df = pd.DataFrame(rows)
            cm_df['date'] = pd.to_datetime(cm_df['time']).dt.tz_localize(None)
            cm_df = cm_df.set_index('date').drop(columns=['time', 'asset'], errors='ignore')

            # Rename columns to clean names
            rename_map = {
                'AdrActCnt': 'active_addresses',
                'HashRate': 'hashrate',
                'TxCnt': 'tx_count',
                'CapMVRVCur': 'mvrv',
                'FeeTotNtv': 'fees_native',
                'FlowInExNtv': 'exchange_inflow',
                'FlowOutExNtv': 'exchange_outflow',
            }
            cm_df = cm_df.rename(columns=rename_map)

            # Drop CoinMetrics status columns (e.g. FlowInExNtv-status)
            status_cols = [c for c in cm_df.columns if '-status' in c]
            cm_df = cm_df.drop(columns=status_cols, errors='ignore')

            # Convert to numeric
            for col in cm_df.columns:
                cm_df[col] = pd.to_numeric(cm_df[col], errors='coerce')

            # Compute net flow
            if 'exchange_inflow' in cm_df.columns and 'exchange_outflow' in cm_df.columns:
                cm_df['exchange_netflow'] = cm_df['exchange_inflow'] - cm_df['exchange_outflow']

            for col in cm_df.columns:
                all_data[col] = cm_df[col]

            print(f"{len(cm_df)} days, {len(cm_df.columns)} columns")
        else:
            print("NO DATA")
    except urllib.error.HTTPError as e:
        # SOL/BNB and other non-BTC/ETH assets return 403 from the community
        # API (free tier covers BTC + ETH only). Log as SKIPPED so it stops
        # showing up as a hard failure on every macro download cycle.
        if e.code == 403:
            print(f"SKIPPED (free-tier 403 — {asset.upper()} not in CoinMetrics community plan)")
        else:
            print(f"ERROR: HTTP {e.code} {e.reason}")
    except Exception as e:
        print(f"ERROR: {e}")

    # --- BGeometrics: SOPR (free, no key, 8 req/hour) — BTC only ---
    if asset != 'btc':
        print(f"    BGeometrics: SOPR skipped (BTC-only on free tier)")
    else:
        print(f"    BGeometrics: SOPR...", end=' ')
        try:
            time.sleep(1)  # polite rate limiting
            bg_url = "https://bitcoin-data.com/v1/sopr"
            req = urllib.request.Request(bg_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                data = json.loads(resp.read().decode())

            if data:
                bg_df = pd.DataFrame(data)
                bg_df['date'] = pd.to_datetime(bg_df['d'])
                bg_df = bg_df.set_index('date').sort_index()
                bg_df['sopr'] = pd.to_numeric(bg_df['sopr'], errors='coerce')

                # Filter to start_date
                bg_df = bg_df[bg_df.index >= start_date]
                all_data['sopr'] = bg_df['sopr']
                print(f"{len(bg_df)} days")
            else:
                print("NO DATA")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_data:
        print("  No on-chain data downloaded!")
        return None

    # Combine into single DataFrame
    onchain_df = pd.DataFrame(all_data)
    onchain_df.index.name = 'date'
    onchain_df = onchain_df.sort_index()
    onchain_df = onchain_df.ffill()  # fill gaps

    # Save
    outfile = os.path.join(MACRO_DIR, f'onchain_{asset}.csv')

    # TODO 0526 data drift fix (2026-05-27): merge with existing CSV preserving
    # historical CoinMetrics values. Without this, every download replaces the
    # whole file — historical metric values could be silently revised by upstream
    # API corrections, drifting from what the live trader saw at decision time.
    if os.path.exists(outfile):
        try:
            existing_oc = pd.read_csv(outfile, parse_dates=[0], index_col=0)
            if existing_oc.index.tz is not None:
                existing_oc.index = existing_oc.index.tz_localize(None)
            n_before = len(onchain_df)
            onchain_df = _merge_preserve_history(onchain_df, existing_oc, freq='D')
            n_after = len(onchain_df)
            if n_after != n_before:
                print(f"  [drift-fix] merged with existing: {n_before} -> {n_after} rows "
                      f"(historical preserved from existing, today's row may update)")
        except Exception as _e:
            print(f"  [drift-fix] could not merge with existing onchain CSV ({_e}); "
                  f"writing fresh download (historical drift NOT prevented this cycle)")

    onchain_df.to_csv(outfile)
    print(f"\n  Saved: {outfile} ({len(onchain_df)} rows, {len(onchain_df.columns)} columns)")
    print(f"  Columns: {list(onchain_df.columns)}")
    print(f"  Date range: {onchain_df.index[0].date()} to {onchain_df.index[-1].date()}")
    for col in onchain_df.columns:
        nans = onchain_df[col].isna().sum()
        if nans > 0:
            print(f"    {col}: {nans} NaN rows")

    return onchain_df


# ============================================================
# 6. DERIVATIVES DATA (funding rate + open interest from Binance)
# ============================================================
def download_derivatives_data(assets=None):
    """
    Download derivatives data from Binance Futures public API (free, no key).
    - Funding rate: every 8h, paginated from 2022-01-01
    - Open interest: hourly, max 30 days per request, paginated
    Saves to data/macro_data/derivatives_{asset}.csv per asset (hourly frequency).
    """
    import urllib.request
    import ssl
    import time

    if assets is None:
        assets = ['BTC', 'ETH']

    all_results = {}
    ctx = ssl._create_unverified_context()
    start_date = '2022-01-01'
    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)

    for asset in assets:
        symbol = f"{asset}USDT"
        print(f"\n  Downloading {asset} derivatives data from Binance...")

        # --- Funding Rate (8h intervals, paginate with limit=1000) ---
        print(f"    Funding rate (8h intervals)...", end=' ')
        all_funding = []
        cursor_ms = start_ms
        try:
            while True:
                url = (f"https://fapi.binance.com/fapi/v1/fundingRate"
                       f"?symbol={symbol}&startTime={cursor_ms}&limit=1000")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                    data = json.loads(resp.read().decode())
                if not data:
                    break
                all_funding.extend(data)
                last_ts = data[-1]['fundingTime']
                if len(data) < 1000:
                    break
                cursor_ms = last_ts + 1
                time.sleep(0.2)

            if all_funding:
                fr_df = pd.DataFrame(all_funding)
                fr_df['datetime'] = pd.to_datetime(fr_df['fundingTime'], unit='ms')
                fr_df['funding_rate'] = pd.to_numeric(fr_df['fundingRate'], errors='coerce')
                fr_df = fr_df[['datetime', 'funding_rate']].set_index('datetime').sort_index()
                fr_hourly = fr_df.resample('1h').ffill()
                print(f"{len(all_funding)} records -> {len(fr_hourly)} hourly rows")
            else:
                fr_hourly = None
                print("NO DATA")
        except Exception as e:
            fr_hourly = None
            print(f"ERROR: {e}")

        # --- Open Interest (hourly, paginate backwards using endTime) ---
        # Uses _binance_get for retries on transient 400/429/5xx — first
        # request after the funding-rate pagination burst was failing daily
        # with a transient 400 (Binance rate-limit-adjacent), no retry,
        # silent OI loss. Retry resolves the transient; body capture surfaces
        # the cause on any persistent failure (delisted symbol, schema change).
        # Brief pre-call sleep also gives the funding-rate burst time to
        # clear Binance's per-IP request window.
        time.sleep(1.0)
        print(f"    Open interest (hourly)...", end=' ')
        all_oi = []
        try:
            # 2026-05-06: fetch_with_fallback tries 3 URL variants in order.
            # If Binance tightens param validation again, we already know
            # which variants used to work. Circuit breaker mutes the source
            # for 1h after 3 consecutive failures across all variants — no
            # more per-cycle log spam during outages, and one Telegram
            # alert on circuit-break (not one per cycle).
            def _oi_url_v1():  # explicit endTime (current Binance preference)
                now_ms = int(time.time() * 1000) - 60_000
                return (f"https://fapi.binance.com/futures/data/openInterestHist"
                        f"?symbol={symbol}&period=1h&endTime={now_ms}&limit=500")
            def _oi_url_v2():  # no endTime (worked pre-2026-05-03)
                return (f"https://fapi.binance.com/futures/data/openInterestHist"
                        f"?symbol={symbol}&period=1h&limit=500")
            def _oi_url_v3():  # smaller limit (in case 500 trips a new cap)
                now_ms = int(time.time() * 1000) - 60_000
                return (f"https://fapi.binance.com/futures/data/openInterestHist"
                        f"?symbol={symbol}&period=1h&endTime={now_ms}&limit=200")
            data, _ = fetch_with_fallback(
                f'oi_{symbol}', [_oi_url_v1, _oi_url_v2, _oi_url_v3],
                ctx, retries_per_url=2, verbose=True,
            )
            if data:
                all_oi.extend(data)

            # Pagination: backfill older OI rows. Binance's 1h OI history
            # is bounded (~30d); requesting endTime older than that returns
            # HTTP 400 "endTime is invalid" (not an empty array), which used
            # to bubble up and kill the entire fetch — losing the fresh data
            # we already got from the initial fetch_with_fallback.
            # Catch HTTP 400 here, treat as "end of available history",
            # break gracefully. retries=1 keeps the log quiet on the
            # expected boundary failure.
            while data and len(data) > 1:
                earliest_ts = data[0]['timestamp']
                if earliest_ts <= start_ms:
                    break
                end_ms = earliest_ts - 1
                url = (f"https://fapi.binance.com/futures/data/openInterestHist"
                       f"?symbol={symbol}&period=1h&endTime={end_ms}&limit=500")
                try:
                    data = _binance_get(url, ctx, retries=1, verbose=False)
                except RuntimeError as pe:
                    msg = str(pe)
                    if 'HTTP 400' in msg and 'endTime' in msg:
                        # Past Binance's 1h OI retention window — stop cleanly
                        break
                    # Other errors (5xx, network) — log once and stop
                    print(f"      [pagination stopped: {msg[:120]}]")
                    break
                if data:
                    all_oi.extend(data)
                time.sleep(0.2)

            if all_oi:
                oi_df = pd.DataFrame(all_oi)
                oi_df['datetime'] = pd.to_datetime(oi_df['timestamp'], unit='ms')
                oi_df['open_interest'] = pd.to_numeric(oi_df['sumOpenInterest'], errors='coerce')
                oi_df['open_interest_usd'] = pd.to_numeric(oi_df['sumOpenInterestValue'], errors='coerce')
                oi_df = oi_df[['datetime', 'open_interest', 'open_interest_usd']]
                oi_df = oi_df.drop_duplicates(subset='datetime').set_index('datetime').sort_index()
                print(f"{len(oi_df)} hourly rows")
            else:
                oi_df = None
                print("NO DATA")
        except Exception as e:
            oi_df = None
            print(f"ERROR: {e}")

        # --- Perp hourly klines (for perp-spot basis) ---
        print(f"    Perp hourly klines...", end=' ')
        all_klines = []
        try:
            cursor_ms = start_ms
            while True:
                url = (f"https://fapi.binance.com/fapi/v1/klines"
                       f"?symbol={symbol}&interval=1h&startTime={cursor_ms}&limit=1000")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                    data = json.loads(resp.read().decode())
                if not data:
                    break
                all_klines.extend(data)
                last_ts = data[-1][0]
                if len(data) < 1000:
                    break
                cursor_ms = last_ts + 1
                time.sleep(0.2)

            if all_klines:
                perp_df = pd.DataFrame(all_klines, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                perp_df['datetime'] = pd.to_datetime(perp_df['open_time'], unit='ms')
                perp_df['perp_close'] = pd.to_numeric(perp_df['close'], errors='coerce')
                perp_df = perp_df[['datetime', 'perp_close']]
                perp_df = perp_df.drop_duplicates(subset='datetime').set_index('datetime').sort_index()
                print(f"{len(perp_df)} hourly rows")
            else:
                perp_df = None
                print("NO DATA")
        except Exception as e:
            perp_df = None
            print(f"ERROR: {e}")

        # Combine
        # Fix #8 (2026-04-24): detect partial-success and alert loudly. File
        # is still written with whatever we got (option B: visibility over
        # safety), so the feature build continues — but a Telegram alert
        # names the missing sub-source so the user knows features are degraded.
        # M-26 fix (2026-04-25): suppress partial-download alerts for
        # sub-sources whose features are ALL in always_disabled_exact. The
        # user doesn't want noise about features they intentionally
        # quarantined. Each sub-source maps to its produced feature names.
        _SUB_FEATURES = {
            'funding_rate': {'deriv_funding_chg1d', 'deriv_funding_chg3d', 'deriv_funding_zscore'},
            'open_interest': {'deriv_oi_chg1d', 'deriv_oi_chg3d', 'deriv_oi_zscore'},
            'perp_klines': {'deriv_basis', 'deriv_basis_chg1d', 'deriv_basis_zscore'},
        }
        try:
            from crypto_trading_system_ed import _load_disabled_features
            _, _, _, _always_disabled = _load_disabled_features()
            _always_disabled = set(_always_disabled or [])
        except Exception:
            _always_disabled = set()

        def _is_sub_fully_disabled(sub_name):
            feats = _SUB_FEATURES.get(sub_name)
            return bool(feats) and feats.issubset(_always_disabled)

        missing = []
        if fr_hourly is None and not _is_sub_fully_disabled('funding_rate'):
            missing.append('funding_rate')
        elif fr_hourly is None:
            print(f"    [info] funding_rate failed for {asset} — features always-disabled, alert suppressed")
        if oi_df is None and not _is_sub_fully_disabled('open_interest'):
            missing.append('open_interest')
        elif oi_df is None:
            print(f"    [info] open_interest failed for {asset} — features always-disabled, alert suppressed")
        if perp_df is None and not _is_sub_fully_disabled('perp_klines'):
            missing.append('perp_klines')
        elif perp_df is None:
            print(f"    [info] perp_klines failed for {asset} — features always-disabled, alert suppressed")
        dfs = [df for df in [fr_hourly, oi_df, perp_df] if df is not None]
        if not dfs:
            print(f"  [!!] No derivatives data for {asset}!")
            _alert_partial_download(
                f'deriv_total_fail_{asset}',
                f"🚨 {asset} derivatives: ALL THREE sub-sources failed (funding, OI, perp_klines). No data written. Check Binance API status.",
                severity='critical',
            )
            continue
        if missing:
            print(f"  [!] {asset} derivatives: PARTIAL — missing {missing}. CSV will have fewer columns; downstream features degraded.")
            _alert_partial_download(
                f'deriv_partial_{asset}_{"_".join(missing)}',
                f"⚠ {asset} derivatives partial download — missing: {missing}. "
                f"CSV written with {len(dfs)}/3 sub-sources; deriv_* features that depend on missing sources will be absent.",
                severity='warn',
            )

        deriv_df = pd.concat(dfs, axis=1).sort_index()
        deriv_df = deriv_df.ffill()

        outfile = os.path.join(MACRO_DIR, f'derivatives_{asset.lower()}.csv')

        # ─────────────────────────────────────────────────────────────────
        # 2026-05-06 fix: when a sub-source fails THIS cycle, splice in
        # the columns from the previous CSV so the file always has the full
        # column set. Otherwise we'd serve the live trader a column-shape
        # mismatch (model trained on N features, sees N-3 features at
        # inference) — that's the real cause of "trader was disabled":
        # silent inference degradation, not log noise.
        #
        # Stale OI is much better than missing OI:
        #   - The model's `optimal_features` may or may not reference OI
        #     columns; when it does, missing column = NaN imputation = bad
        #     predictions until /reload manually heals the file.
        #   - Stale OI just gets ffill'd from last-known-good values, which
        #     is what the existing ffill already does within a single CSV.
        # ─────────────────────────────────────────────────────────────────
        spliced_from_lkg = []
        if missing and os.path.exists(outfile):
            try:
                lkg_df = pd.read_csv(outfile, parse_dates=[0], index_col=0)
                if lkg_df.index.tz is not None:
                    lkg_df.index = lkg_df.index.tz_localize(None)
                # Identify columns from missing sub-sources that exist in
                # the LKG file but not in the new deriv_df.
                missing_cols_by_sub = {
                    'funding_rate': ['funding_rate'],
                    'open_interest': ['open_interest', 'open_interest_usd'],
                    'perp_klines': ['perp_open', 'perp_high', 'perp_low', 'perp_close'],
                }
                for sub in missing:
                    for col in missing_cols_by_sub.get(sub, []):
                        if col in lkg_df.columns and col not in deriv_df.columns:
                            # Reindex LKG col to new index, ffill stale values
                            deriv_df[col] = lkg_df[col].reindex(deriv_df.index).ffill()
                            spliced_from_lkg.append(col)
                if spliced_from_lkg:
                    print(f"  [LKG] Spliced {len(spliced_from_lkg)} stale column(s) "
                          f"from previous {os.path.basename(outfile)}: {spliced_from_lkg}")
            except Exception as _e:
                print(f"  [LKG] Could not splice from previous CSV ({_e}); "
                      f"writing partial CSV — model may see column shape mismatch.")

        # TODO 0526 data drift fix (2026-05-27): merge with existing CSV preserving
        # historical rows. Without this, every download replaces the whole file —
        # Binance returns paginated history every cycle, and any minor differences
        # in OI/funding/perp values for past hours silently overwrote what the live
        # trader saw at decision time. This caused the bulk of the derivatives drift
        # we measured (239 cells diff between May 22 snapshot and current data).
        #
        # This runs AFTER the LKG splicing block above — so even when a sub-source
        # fails, the LKG-spliced columns are also preserved through the merge.
        if os.path.exists(outfile):
            try:
                existing_dv = pd.read_csv(outfile, parse_dates=[0], index_col=0)
                if existing_dv.index.tz is not None:
                    existing_dv.index = existing_dv.index.tz_localize(None)
                n_before = len(deriv_df)
                deriv_df = _merge_preserve_history(deriv_df, existing_dv, freq='1h')
                n_after = len(deriv_df)
                if n_after != n_before:
                    print(f"  [drift-fix] merged with existing: {n_before} -> {n_after} rows "
                          f"(historical preserved from existing, current hour may update)")
            except Exception as _e:
                print(f"  [drift-fix] could not merge with existing derivatives CSV ({_e}); "
                      f"writing fresh download (historical drift NOT prevented this cycle)")

        deriv_df.to_csv(outfile)
        print(f"  Saved: {outfile} ({len(deriv_df)} rows, {len(deriv_df.columns)} columns)")
        print(f"  Date range: {deriv_df.index[0]} to {deriv_df.index[-1]}")
        all_results[asset] = deriv_df

    return all_results


# ============================================================
# 7. GDELT GEOPOLITICAL DATA (free, no API key)
# ============================================================
def download_gdelt_geopolitical():
    """
    Download geopolitical tension data from GDELT DOC 2.0 API (free, no key).
    Queries for Iran/conflict/war/sanctions articles — returns:
      - volume: % of all GDELT articles matching query (15-min resolution)
      - tone:   average tone score (negative=bad news, positive=good news)
    GDELT DOC API limits: max 3-month lookback, rate limit ~5 req/min.
    We download in 3-month chunks going back to 2022, sleeping between requests.
    Saves to data/macro_data/gdelt_geopolitical.csv (hourly, aggregated from 15-min).
    """
    import urllib.request
    import urllib.parse
    import ssl
    import time

    print(f"\n  Downloading GDELT geopolitical data...")
    ctx = ssl._create_unverified_context()

    # Queries: Iran-specific + broader geopolitical conflict terms
    queries = {
        'iran': 'Iran (war OR conflict OR sanctions OR ceasefire OR Hormuz OR nuclear OR attack OR strike OR missile)',
        'geopolitical': '(geopolitical risk OR trade war OR tariff OR sanctions OR military conflict OR escalation)',
    }

    # GDELT DOC API max lookback is 3 months; we paginate in 3-month chunks
    from datetime import datetime, timedelta
    end_date = datetime.utcnow()
    start_date = datetime(2024, 1, 1)  # GDELT DOC API practical limit ~2y back

    all_rows = []

    for qname, query in queries.items():
        print(f"    Query '{qname}'...")
        cursor = end_date
        chunk_count = 0

        while cursor > start_date:
            chunk_start = max(cursor - timedelta(days=89), start_date)
            days_back = (end_date - chunk_start).days
            timespan = f"{days_back * 24}h"

            # Cap at 3 months (2160h)
            hours_back = min(days_back * 24, 2160)
            timespan = f"{hours_back}h"

            for mode in ['timelinevol', 'timelinetone']:
                params = urllib.parse.urlencode({
                    'query': query,
                    'mode': mode,
                    'format': 'json',
                    'timespan': timespan,
                })
                url = f'https://api.gdeltproject.org/api/v2/doc/doc?{params}'

                for attempt in range(3):
                    try:
                        time.sleep(12)  # respect rate limit
                        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                            raw = resp.read().decode()

                        if not raw or raw.startswith('The specified'):
                            print(f"      {mode}: empty response, skipping")
                            break

                        data = json.loads(raw)
                        tl = data.get('timeline', [])
                        if not tl:
                            break

                        pts = tl[0].get('data', [])
                        value_key = 'vol' if mode == 'timelinevol' else 'tone'

                        for p in pts:
                            dt = pd.to_datetime(p['date'])
                            all_rows.append({
                                'datetime': dt,
                                'query': qname,
                                'metric': value_key,
                                'value': float(p['value']),
                            })

                        print(f"      {mode}: {len(pts)} points (back to {pts[0]['date'][:10] if pts else '?'})")
                        break  # success

                    except urllib.error.HTTPError as e:
                        if e.code == 429:
                            wait = 30 * (attempt + 1)
                            print(f"      Rate limited, waiting {wait}s...")
                            time.sleep(wait)
                        else:
                            print(f"      HTTP {e.code}: {e.reason}")
                            break
                    except Exception as e:
                        print(f"      Error: {e}")
                        break

            # Move cursor back; GDELT returns max 3 months so one chunk is enough per query
            cursor = chunk_start - timedelta(days=1)
            chunk_count += 1
            if chunk_count >= 1:
                break  # GDELT DOC API only allows ~3 months lookback anyway

    if not all_rows:
        print("  No GDELT data downloaded!")
        return None

    # Pivot: rows → columns (iran_vol, iran_tone, geopolitical_vol, geopolitical_tone)
    raw_df = pd.DataFrame(all_rows)
    raw_df['col'] = raw_df['query'] + '_' + raw_df['metric']
    pivot = raw_df.pivot_table(index='datetime', columns='col', values='value', aggfunc='mean')

    # Resample 15-min → hourly (mean vol, mean tone)
    pivot.index = pd.to_datetime(pivot.index)
    hourly = pivot.resample('1h').mean()

    # Forward-fill small gaps (up to 4h)
    hourly = hourly.ffill(limit=4)

    # Strip timezone so _load_macro_csv().tz_localize(None) won't fail
    if hourly.index.tz is not None:
        hourly.index = hourly.index.tz_localize(None)

    # Save
    outfile = os.path.join(MACRO_DIR, 'gdelt_geopolitical.csv')
    hourly.to_csv(outfile)
    print(f"  Saved: {outfile} ({len(hourly)} rows, {list(hourly.columns)})")
    print(f"  Date range: {hourly.index[0]} to {hourly.index[-1]}")

    return hourly


# ============================================================
# 8. STABLECOIN FLOWS (USDT + USDC market cap from CoinGecko)
# ============================================================
def download_stablecoin_flows():
    """
    Download USDT and USDC market cap history from CoinGecko (free, no key).
    Daily data — forward-filled to hourly. Computes daily change as feature.
    """
    import urllib.request
    import ssl
    import time

    print(f"\n  Downloading stablecoin market cap data...")
    ctx = ssl._create_unverified_context()

    stables = {'tether': 'usdt', 'usd-coin': 'usdc'}
    all_dfs = []

    for coin_id, label in stables.items():
        print(f"    {label.upper()} market cap...", end=' ')
        try:
            url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                   f"?vs_currency=usd&days=365&interval=daily")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                data = json.loads(resp.read().decode())
            if 'market_caps' in data and data['market_caps']:
                mc = pd.DataFrame(data['market_caps'], columns=['timestamp', f'{label}_mcap'])
                mc['datetime'] = pd.to_datetime(mc['timestamp'], unit='ms')
                mc = mc[['datetime', f'{label}_mcap']].set_index('datetime').sort_index()
                # TODO 0526 data drift fix (2026-05-27): preserve historical mcap values
                mc = _dedup_preserve_history(mc, freq='D')
                all_dfs.append(mc)
                print(f"{len(mc)} days")
            else:
                print("NO DATA")
            time.sleep(1.5)  # CoinGecko rate limit
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_dfs:
        print("  No stablecoin data downloaded!")
        return None

    combined = pd.concat(all_dfs, axis=1).sort_index().ffill()
    combined['total_stable_mcap'] = combined.sum(axis=1)

    outfile = os.path.join(MACRO_DIR, 'stablecoin_flows.csv')
    combined.to_csv(outfile)
    print(f"  Saved: {outfile} ({len(combined)} rows)")
    return combined


# ============================================================
# 9. OPTIONS IV SKEW (Deribit — free public API)
# ============================================================
def download_options_iv_skew():
    """
    Download ETH and BTC options implied volatility from Deribit public API (free, no key).
    Gets current IV surface — ATM IV, 25-delta put/call IV, skew.
    Historical IV requires paid data; we'll snapshot and accumulate over time.
    """
    import urllib.request
    import ssl

    print(f"\n  Downloading options IV data from Deribit...")
    ctx = ssl._create_unverified_context()

    rows = []
    for asset in ['ETH', 'BTC']:
        try:
            # Get instruments list
            url = (f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
                   f"?currency={asset}&kind=option")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                data = json.loads(resp.read().decode())

            if 'result' in data:
                options = data['result']
                # Filter to near-term options (< 30 days)
                now_ts = pd.Timestamp.now().timestamp() * 1000
                near_term = [o for o in options if o.get('creation_timestamp', 0) > 0]

                # Compute aggregate IV stats
                ivs = [o.get('mark_iv', 0) for o in near_term if o.get('mark_iv', 0) > 0]
                put_ivs = [o.get('mark_iv', 0) for o in near_term
                           if o.get('instrument_name', '').endswith('P') and o.get('mark_iv', 0) > 0]
                call_ivs = [o.get('mark_iv', 0) for o in near_term
                            if o.get('instrument_name', '').endswith('C') and o.get('mark_iv', 0) > 0]

                import numpy as np
                avg_iv = np.mean(ivs) if ivs else 0
                put_iv = np.mean(put_ivs) if put_ivs else 0
                call_iv = np.mean(call_ivs) if call_ivs else 0
                skew = put_iv - call_iv  # positive = more fear

                rows.append({
                    'datetime': pd.Timestamp.now().floor('h'),
                    'asset': asset,
                    'avg_iv': avg_iv,
                    'put_iv': put_iv,
                    'call_iv': call_iv,
                    'iv_skew': skew,
                    'n_options': len(near_term),
                })
                print(f"    {asset}: IV={avg_iv:.1f}% skew={skew:.1f}% ({len(near_term)} options)")
            else:
                print(f"    {asset}: no data")
        except Exception as e:
            print(f"    {asset}: ERROR {e}")

    if not rows:
        print("  No options data!")
        return None

    df = pd.DataFrame(rows)
    outfile = os.path.join(MACRO_DIR, 'options_iv_snapshot.csv')
    # Append to existing file. A (2026-06-24): guard the read — an empty/corrupt
    # existing file (interrupted write / Drive-sync race) must self-heal to a
    # fresh snapshot, not crash the cycle with EmptyDataError.
    if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
        try:
            existing = pd.read_csv(outfile)
            # TODO 0526 data drift fix (2026-05-27): preserve historical IV snapshots —
            # was keep='last' which let a re-snapshot within the same hour overwrite
            # the first reading. Now first reading per (hour, asset) is frozen.
            df = _dedup_preserve_history(
                pd.concat([existing, df], ignore_index=True),
                freq='1h', subset=['datetime', 'asset'],
            )
        except Exception as _e:
            print(f"  [snapshot] could not merge existing {os.path.basename(outfile)} "
                  f"({_e}); writing fresh snapshot this cycle")
    _atomic_to_csv(df, outfile, index=False)
    print(f"  Saved: {outfile} ({len(df)} rows)")
    return df


# ============================================================
# 10. ORDERBOOK IMBALANCE SNAPSHOT
# ============================================================
def download_orderbook_snapshot(assets=None):
    """
    Snapshot orderbook bid/ask depth from Binance (free, no key).
    Computes bid_vol/ask_vol imbalance ratio at top 20 levels.
    Appends to CSV — call hourly to build history.
    """
    import urllib.request
    import ssl

    if assets is None:
        assets = ['ETH', 'BTC']

    print(f"\n  Snapshotting orderbook depth...")
    ctx = ssl._create_unverified_context()
    rows = []

    for asset in assets:
        symbol = f"{asset}USDT"
        try:
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                data = json.loads(resp.read().decode())

            bid_vol = sum(float(b[1]) for b in data.get('bids', []))
            ask_vol = sum(float(a[1]) for a in data.get('asks', []))
            imbalance = bid_vol / (ask_vol + 1e-10)
            best_bid = float(data['bids'][0][0]) if data.get('bids') else 0
            best_ask = float(data['asks'][0][0]) if data.get('asks') else 0
            spread_bps = (best_ask - best_bid) / best_bid * 10000 if best_bid > 0 else 0

            rows.append({
                'datetime': pd.Timestamp.now().floor('h'),
                'asset': asset,
                'bid_vol': bid_vol,
                'ask_vol': ask_vol,
                'ob_imbalance': imbalance,
                'spread_bps': spread_bps,
            })
            print(f"    {asset}: imbalance={imbalance:.3f} spread={spread_bps:.1f}bps")
        except Exception as e:
            print(f"    {asset}: ERROR {e}")

    if not rows:
        return None

    df = pd.DataFrame(rows)
    outfile = os.path.join(MACRO_DIR, 'orderbook_snapshots.csv')
    # A (2026-06-24): guard the read — empty/corrupt file self-heals to fresh.
    if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
        try:
            existing = pd.read_csv(outfile)
            # TODO 0526 data drift fix (2026-05-27): preserve historical orderbook snapshots
            df = _dedup_preserve_history(
                pd.concat([existing, df], ignore_index=True),
                freq='1h', subset=['datetime', 'asset'],
            )
        except Exception as _e:
            print(f"  [snapshot] could not merge existing {os.path.basename(outfile)} "
                  f"({_e}); writing fresh snapshot this cycle")
    _atomic_to_csv(df, outfile, index=False)
    print(f"  Saved: {outfile} ({len(df)} rows)")
    return df


# ============================================================
# 11. WHALE WALLET FLOWS (Arkham free endpoint)
# ============================================================
def download_whale_flows():
    """
    Download large ETH/BTC transfers from Arkham Intelligence or Whale Alert.
    Note: Arkham requires signup for API; Whale Alert has a free tier (10 req/min).
    This uses the Whale Alert free API for recent large transfers.
    """
    import urllib.request
    import ssl

    print(f"\n  Downloading whale flow data...")
    ctx = ssl._create_unverified_context()

    # Whale Alert free API — requires API key (free tier: 10 req/min, 100/day)
    # If no key, skip gracefully
    api_key = os.environ.get('WHALE_ALERT_API_KEY', '')
    if not api_key:
        # Try config file
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'whale_alert_config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                api_key = json.load(f).get('api_key', '')

    if not api_key:
        print("  Skipped — no WHALE_ALERT_API_KEY env var or config/whale_alert_config.json")
        print("  Get free key at: https://whale-alert.io/")
        return None

    try:
        # Last 1 hour of large transactions (>$1M)
        import time
        since = int(time.time()) - 3600
        url = (f"https://api.whale-alert.io/v1/transactions"
               f"?api_key={api_key}&min_value=1000000&start={since}")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            data = json.loads(resp.read().decode())

        txns = data.get('transactions', [])
        if txns:
            rows = []
            for tx in txns:
                rows.append({
                    'datetime': pd.Timestamp.now().floor('h'),
                    'symbol': tx.get('symbol', '').upper(),
                    'amount_usd': tx.get('amount_usd', 0),
                    'from_type': tx.get('from', {}).get('owner_type', 'unknown'),
                    'to_type': tx.get('to', {}).get('owner_type', 'unknown'),
                })
            df = pd.DataFrame(rows)

            # Aggregate: net exchange flow per hour
            exchange_in = df[df['to_type'] == 'exchange']['amount_usd'].sum()
            exchange_out = df[df['from_type'] == 'exchange']['amount_usd'].sum()
            print(f"    {len(txns)} large txns | exchange_in=${exchange_in/1e6:.1f}M | exchange_out=${exchange_out/1e6:.1f}M")

            outfile = os.path.join(MACRO_DIR, 'whale_flows.csv')
            # A (2026-06-24): guard the read — empty/corrupt file self-heals to fresh.
            if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
                try:
                    existing = pd.read_csv(outfile)
                    df = pd.concat([existing, df], ignore_index=True)
                except Exception as _e:
                    print(f"  [snapshot] could not read existing {os.path.basename(outfile)} "
                          f"({_e}); writing fresh")
            _atomic_to_csv(df, outfile, index=False)
            print(f"  Saved: {outfile}")
            return df
        else:
            print("  No large transactions in last hour")
            return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ============================================================
# 12. KALSHI PREDICTION MARKET DATA
# ============================================================
def download_kalshi_data():
    """
    Download crypto-related prediction market data from Kalshi.
    Note: Kalshi requires account + API key. Skip if not configured.
    """
    print(f"\n  Kalshi prediction market data...")
    api_key = os.environ.get('KALSHI_API_KEY', '')
    if not api_key:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'kalshi_config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                api_key = json.load(f).get('api_key', '')

    if not api_key:
        print("  Skipped — no KALSHI_API_KEY env var or config/kalshi_config.json")
        print("  Get API access at: https://kalshi.com/")
        return None

    # TODO: implement when API key available
    print("  API key found but download not yet implemented")
    return None


# ============================================================
# MAIN
# ============================================================
def _wait_for_safe_download_window(safe_start_minute=6, max_wait=400):
    """Avoid colliding with the live trader's hourly download cycle at xx:00.

    Trader downloads typically run xx:00-xx:02. If current time is in the
    xx:00 to xx:{safe_start_minute-1} window, sleep until xx:{safe_start_minute}.
    This prevents concurrent CSV writes (orderbook_snapshots, derivatives_*,
    macro_daily) from two processes corrupting each other's files.

    Added 2026-05-29 after user observed v3 + trader could collide if launched
    at xx:00 (risk small but non-zero — both processes append to the same CSVs).
    """
    now = datetime.now()
    if now.minute >= safe_start_minute:
        return
    target_seconds_into_hour = safe_start_minute * 60
    current_seconds_into_hour = now.minute * 60 + now.second
    wait_seconds = max(1, target_seconds_into_hour - current_seconds_into_hour + 1)
    wait_seconds = min(wait_seconds, max_wait)
    print(f"  [SAFE-WINDOW] minute={now.minute:02d} — waiting {wait_seconds}s until xx:{safe_start_minute:02d} "
          f"to avoid trader's hourly download collision", flush=True)
    time.sleep(wait_seconds)


def main(full=False):
    _wait_for_safe_download_window()
    print("=" * 60)
    print("  MACRO & SENTIMENT DATA DOWNLOAD" + ("  [--full]" if full else ""))
    print("=" * 60)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 1. Macro indicators
    if not full and _is_fresh(os.path.join(MACRO_DIR, 'macro_daily.csv')):
        print("\n  Macro data: fresh (< 6h) — skipping")
        macro_df = None
    else:
        macro_df = download_yfinance_data(full=full)

    # 2. Fear & Greed
    if _is_fresh(os.path.join(MACRO_DIR, 'fear_greed.csv')):
        print("  Fear & Greed: fresh — skipping")
    else:
        fg_df = download_fear_greed()

    # 4. Cross-asset pairs
    if not full and _is_fresh(os.path.join(MACRO_DIR, 'cross_asset.csv')):
        print("  Cross-asset: fresh — skipping")
    else:
        cross_df = download_cross_asset(full=full)

    # 5. On-chain data (BTC + ETH + XRP + SOL + LINK + BNB) — parallel across stale assets
    onchain_assets = ['btc', 'eth', 'xrp', 'sol', 'link', 'bnb']
    stale_onchain = [a for a in onchain_assets if not _is_fresh(os.path.join(MACRO_DIR, f'onchain_{a}.csv'))]
    fresh_onchain = [a for a in onchain_assets if a not in stale_onchain]
    for a in fresh_onchain:
        print(f"  On-chain {a.upper()}: fresh — skipping")
    if stale_onchain:
        with ThreadPoolExecutor(max_workers=min(6, len(stale_onchain))) as ex:
            futs = {ex.submit(download_onchain_data, asset=a): a for a in stale_onchain}
            for fut in as_completed(futs):
                # Surface any unhandled exception now rather than swallowing
                fut.result()

    # 6. Derivatives data (funding rate + open interest + perp klines — parallel across stale assets)
    deriv_assets = ['BTC', 'ETH', 'XRP', 'SOL', 'LINK', 'BNB']
    stale_assets = [a for a in deriv_assets if not _is_fresh(os.path.join(MACRO_DIR, f'derivatives_{a.lower()}.csv'))]
    if stale_assets:
        # download_derivatives_data() loops internally; parallelize that loop by
        # calling it once per stale asset in a thread pool. Each call is independent
        # (writes its own file). Binance public API rate limit (2400 req/min/IP)
        # comfortably handles 6 concurrent worker threads × ~15-50 requests each.
        with ThreadPoolExecutor(max_workers=min(6, len(stale_assets))) as ex:
            futs = {ex.submit(download_derivatives_data, assets=[a]): a for a in stale_assets}
            for fut in as_completed(futs):
                fut.result()
    else:
        print("  Derivatives: fresh — skipping")

    # 7. GDELT geopolitical data — DISABLED (0/33 models ever selected, slow rate-limited download)
    # gdelt_df = download_gdelt_geopolitical()

    # 8. Stablecoin flows
    if _is_fresh(os.path.join(MACRO_DIR, 'stablecoin_flows.csv')):
        print("  Stablecoins: fresh — skipping")
    else:
        stable_df = download_stablecoin_flows()

    # 9. Options IV snapshot (always append — accumulates)
    iv_df = download_options_iv_skew()

    # 10. Orderbook snapshot (always append — accumulates)
    ob_df = download_orderbook_snapshot()

    # 11. Whale flows (needs API key)
    whale_df = download_whale_flows()

    # 12. Kalshi (needs API key)
    kalshi_df = download_kalshi_data()

    # Summary
    print(f"\n{'='*60}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Files saved in {MACRO_DIR}/:")
    for f in sorted(os.listdir(MACRO_DIR)):
        size = os.path.getsize(os.path.join(MACRO_DIR, f))
        print(f"    {f:30s} {size:>10,} bytes")

    print(f"\n  Next step: run diagnostic with V2 features:")
    print(f"    python features_v2.py --test")
    print(f"    python hourly_trading_system.py  (will auto-detect V2)")


if __name__ == '__main__':
    import argparse
    _ap = argparse.ArgumentParser(description='Download macro / sentiment / cross-asset data.')
    _ap.add_argument('--full', action='store_true',
                     help='Re-pull full 3-year history for yfinance sources '
                          '(macro_daily, cross_asset). Default is incremental tail '
                          'update (~7 days). Use --full after detecting corruption '
                          'beyond the 7-day overlap window or for first-time setup.')
    _args = _ap.parse_args()
    main(full=_args.full)
