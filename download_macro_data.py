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
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create output folder
MACRO_DIR = 'data/macro_data'
os.makedirs(MACRO_DIR, exist_ok=True)

# Freshness threshold — skip re-download if file updated within this many seconds
FRESHNESS_SECONDS = 3600  # 1 hour


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
def download_yfinance_data():
    """Download macro indicators via yfinance."""
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

    # Download 3 years of daily data (more than enough for our models)
    start_date = '2022-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\n  Downloading {len(tickers)} macro indicators from yfinance...")
    print(f"  Period: {start_date} to {end_date}")

    all_data = {}
    for name, ticker in tickers.items():
        try:
            print(f"    {name:10s} ({ticker})...", end=' ')
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) == 0:
                print("NO DATA")
                continue

            # Keep only Close column, rename
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            series = df['Close'].dropna()
            all_data[name] = series
            print(f"{len(series)} days")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_data:
        print("  No macro data downloaded!")
        return None

    # Combine into single DataFrame
    macro_df = pd.DataFrame(all_data)
    macro_df.index.name = 'date'
    macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)

    # Forward-fill gaps (weekends, holidays)
    macro_df = macro_df.ffill()

    # Save
    outfile = os.path.join(MACRO_DIR, 'macro_daily.csv')
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

        # Save
        outfile = os.path.join(MACRO_DIR, 'fear_greed.csv')
        fg_df.to_csv(outfile)
        print(f"  Saved: {outfile} ({len(fg_df)} rows)")
        print(f"  Date range: {fg_df.index[0].date()} to {fg_df.index[-1].date()}")
        print(f"  Current: {fg_df.iloc[-1]['fear_greed']} ({fg_df.iloc[-1]['fear_greed_label']})")

        return fg_df

    except Exception as e:
        print(f"  ERROR downloading Fear & Greed: {e}")
        return None


# ============================================================
# 3. HOURLY MACRO PROXY (for DAX hourly system)
# ============================================================
def create_hourly_macro(macro_df):
    """
    Create hourly-frequency macro data by forward-filling daily data.
    This gives the DAX hourly system access to daily macro indicators
    (the latest known value at each hour).
    """
    if macro_df is None:
        return

    print(f"\n  Creating hourly macro proxy (forward-fill daily into hourly)...")

    # Create hourly index spanning the full date range
    start = macro_df.index[0]
    end = macro_df.index[-1] + timedelta(days=1)
    hourly_idx = pd.date_range(start=start, end=end, freq='h')

    # Reindex daily data to hourly and forward-fill
    hourly_df = macro_df.reindex(hourly_idx, method='ffill')
    hourly_df.index.name = 'datetime'

    outfile = os.path.join(MACRO_DIR, 'macro_hourly.csv')
    hourly_df.to_csv(outfile)
    print(f"  Saved: {outfile} ({len(hourly_df)} rows)")


# ============================================================
# 4. CROSS-ASSET PAIRS (for correlation features)
# ============================================================
def download_cross_asset():
    """Download BTC and major indices for cross-correlation features."""
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

    start_date = '2022-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\n  Downloading cross-asset pairs for correlation features...")
    all_data = {}
    for name, ticker in pairs.items():
        try:
            print(f"    {name:10s} ({ticker})...", end=' ')
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            series = df['Close'].dropna()
            all_data[name] = series
            print(f"{len(series)} days")
        except Exception as e:
            print(f"ERROR: {e}")

    if all_data:
        cross_df = pd.DataFrame(all_data)
        cross_df.index.name = 'date'
        cross_df.index = pd.to_datetime(cross_df.index).tz_localize(None)
        cross_df = cross_df.ffill()

        outfile = os.path.join(MACRO_DIR, 'cross_asset.csv')
        cross_df.to_csv(outfile)
        print(f"\n  Saved: {outfile} ({len(cross_df)} rows)")
        return cross_df

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
        print(f"    Open interest (hourly)...", end=' ')
        all_oi = []
        try:
            url = (f"https://fapi.binance.com/futures/data/openInterestHist"
                   f"?symbol={symbol}&period=1h&limit=500")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                data = json.loads(resp.read().decode())
            if data:
                all_oi.extend(data)

            while data and len(data) > 1:
                earliest_ts = data[0]['timestamp']
                if earliest_ts <= start_ms:
                    break
                end_ms = earliest_ts - 1
                url = (f"https://fapi.binance.com/futures/data/openInterestHist"
                       f"?symbol={symbol}&period=1h&endTime={end_ms}&limit=500")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                    data = json.loads(resp.read().decode())
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
        dfs = [df for df in [fr_hourly, oi_df, perp_df] if df is not None]
        if not dfs:
            print(f"  No derivatives data for {asset}!")
            continue

        deriv_df = pd.concat(dfs, axis=1).sort_index()
        deriv_df = deriv_df.ffill()

        outfile = os.path.join(MACRO_DIR, f'derivatives_{asset.lower()}.csv')
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
                mc = mc[~mc.index.duplicated(keep='last')]
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
    # Append to existing file
    if os.path.exists(outfile):
        existing = pd.read_csv(outfile)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(subset=['datetime', 'asset'], keep='last')
    df.to_csv(outfile, index=False)
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
    if os.path.exists(outfile):
        existing = pd.read_csv(outfile)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(subset=['datetime', 'asset'], keep='last')
    df.to_csv(outfile, index=False)
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
            if os.path.exists(outfile):
                existing = pd.read_csv(outfile)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_csv(outfile, index=False)
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
def main():
    print("=" * 60)
    print("  MACRO & SENTIMENT DATA DOWNLOAD")
    print("=" * 60)

    # 1. Macro indicators
    if _is_fresh(os.path.join(MACRO_DIR, 'macro_daily.csv')):
        print("\n  Macro data: fresh (< 1h) — skipping")
        macro_df = None
    else:
        macro_df = download_yfinance_data()

    # 2. Fear & Greed
    if _is_fresh(os.path.join(MACRO_DIR, 'fear_greed.csv')):
        print("  Fear & Greed: fresh — skipping")
    else:
        fg_df = download_fear_greed()

    # 3. Hourly proxy for DAX system
    if macro_df is not None:
        create_hourly_macro(macro_df)

    # 4. Cross-asset pairs
    if _is_fresh(os.path.join(MACRO_DIR, 'cross_asset.csv')):
        print("  Cross-asset: fresh — skipping")
    else:
        cross_df = download_cross_asset()

    # 5. On-chain data (BTC + ETH + XRP + SOL + LINK)
    for _asset in ['btc', 'eth', 'xrp', 'sol', 'link']:
        if _is_fresh(os.path.join(MACRO_DIR, f'onchain_{_asset}.csv')):
            print(f"  On-chain {_asset.upper()}: fresh — skipping")
        else:
            download_onchain_data(asset=_asset)

    # 6. Derivatives data (funding rate + open interest — BTC + ETH + XRP + SOL + LINK)
    stale_assets = [a for a in ['BTC', 'ETH', 'XRP', 'SOL', 'LINK'] if not _is_fresh(os.path.join(MACRO_DIR, f'derivatives_{a.lower()}.csv'))]
    if stale_assets:
        deriv_df = download_derivatives_data(assets=stale_assets)
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
    main()
