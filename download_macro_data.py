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
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create output folder
MACRO_DIR = 'data/macro_data'
os.makedirs(MACRO_DIR, exist_ok=True)


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
        'DXY':      'DX=F',        # US Dollar Index Futures (DX-Y.NYB unreliable)
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
def download_onchain_data():
    """
    Download Bitcoin on-chain metrics from free APIs (no API key needed).
    Sources:
      - CoinMetrics Community API: active addresses, hash rate, tx count, MVRV,
        fees (native BTC), exchange inflow/outflow
      - BGeometrics: SOPR
    Saves to data/macro_data/onchain_btc.csv (daily frequency).
    """
    import urllib.request
    import ssl
    import time

    print(f"\n  Downloading Bitcoin on-chain data...")

    # SSL fix for Windows
    ctx = ssl._create_unverified_context()

    start_date = '2022-01-01'
    all_data = {}

    # --- CoinMetrics Community API (free, no key) ---
    cm_metrics = 'AdrActCnt,HashRate,TxCnt,CapMVRVCur,FeeTotNtv,FlowInExNtv,FlowOutExNtv'
    cm_url = (
        f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
        f"?assets=btc&metrics={cm_metrics}&frequency=1d"
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
                'FeeTotNtv': 'fees_btc',
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

    # --- BGeometrics: SOPR (free, no key, 8 req/hour) ---
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
    outfile = os.path.join(MACRO_DIR, 'onchain_btc.csv')
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
def download_derivatives_data():
    """
    Download BTC derivatives data from Binance Futures public API (free, no key).
    - Funding rate: every 8h, paginated from 2022-01-01
    - Open interest: hourly, max 30 days per request, paginated
    Saves to data/macro_data/derivatives_btc.csv (hourly frequency).
    """
    import urllib.request
    import ssl
    import time

    print(f"\n  Downloading BTC derivatives data from Binance...")

    ctx = ssl._create_unverified_context()
    start_date = '2022-01-01'
    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)

    # --- Funding Rate (8h intervals, paginate with limit=1000) ---
    print(f"    Funding rate (8h intervals)...", end=' ')
    all_funding = []
    cursor_ms = start_ms
    try:
        while True:
            url = (f"https://fapi.binance.com/fapi/v1/fundingRate"
                   f"?symbol=BTCUSDT&startTime={cursor_ms}&limit=1000")
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
            time.sleep(0.2)  # polite rate limiting

        if all_funding:
            fr_df = pd.DataFrame(all_funding)
            fr_df['datetime'] = pd.to_datetime(fr_df['fundingTime'], unit='ms')
            fr_df['funding_rate'] = pd.to_numeric(fr_df['fundingRate'], errors='coerce')
            fr_df = fr_df[['datetime', 'funding_rate']].set_index('datetime').sort_index()
            # Resample to hourly (forward-fill 8h values)
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
        # First request: latest 500 hours
        url = (f"https://fapi.binance.com/futures/data/openInterestHist"
               f"?symbol=BTCUSDT&period=1h&limit=500")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            data = json.loads(resp.read().decode())
        if data:
            all_oi.extend(data)

        # Paginate backwards until we reach start_date or run out of data
        while data and len(data) > 1:
            earliest_ts = data[0]['timestamp']
            if earliest_ts <= start_ms:
                break
            end_ms = earliest_ts - 1
            url = (f"https://fapi.binance.com/futures/data/openInterestHist"
                   f"?symbol=BTCUSDT&period=1h&endTime={end_ms}&limit=500")
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

    # Combine
    dfs = [df for df in [fr_hourly, oi_df] if df is not None]
    if not dfs:
        print("  No derivatives data downloaded!")
        return None

    deriv_df = pd.concat(dfs, axis=1).sort_index()
    deriv_df = deriv_df.ffill()

    outfile = os.path.join(MACRO_DIR, 'derivatives_btc.csv')
    deriv_df.to_csv(outfile)
    print(f"\n  Saved: {outfile} ({len(deriv_df)} rows, {len(deriv_df.columns)} columns)")
    print(f"  Columns: {list(deriv_df.columns)}")
    print(f"  Date range: {deriv_df.index[0]} to {deriv_df.index[-1]}")

    return deriv_df


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  MACRO & SENTIMENT DATA DOWNLOAD")
    print("=" * 60)

    # 1. Macro indicators
    macro_df = download_yfinance_data()

    # 2. Fear & Greed
    fg_df = download_fear_greed()

    # 3. Hourly proxy for DAX system
    create_hourly_macro(macro_df)

    # 4. Cross-asset pairs
    cross_df = download_cross_asset()

    # 5. On-chain data (Bitcoin)
    onchain_df = download_onchain_data()

    # 6. Derivatives data (funding rate + open interest)
    deriv_df = download_derivatives_data()

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
