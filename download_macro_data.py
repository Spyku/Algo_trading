"""
Download Macro & Sentiment Data (One-Off)
==========================================
Downloads all freely available macro, sentiment, and cross-asset data.
Saves to macro_data/ folder as CSV files.

Sources:
  - yfinance: VIX, DXY, S&P500, Nasdaq, Gold, US10Y, EUR/USD, BTC correlation pairs
  - alternative.me: Crypto Fear & Greed Index (free, no API key)

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
MACRO_DIR = 'macro_data'
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
        'DXY':      'DX-Y.NYB',   # US Dollar Index
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
