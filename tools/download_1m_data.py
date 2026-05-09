"""
download_1m_data.py — fetch 1-minute OHLCV from Binance for the take-profit
overlay backtest. Defaults to ETH spot, last 60 days.

Output: data/{asset}_1m_data.csv with columns datetime, open, high, low, close, volume

Usage:
  python tools/download_1m_data.py
  python tools/download_1m_data.py --asset ETH --days 60
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ENGINE, 'data')

SYMBOL_MAP = {'ETH': 'ETHUSDT', 'BTC': 'BTCUSDT', 'SOL': 'SOLUSDT', 'BNB': 'BNBUSDT'}


def fetch_klines_1m(symbol: str, days: int) -> pd.DataFrame:
    ctx = ssl.create_default_context()
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000

    all_rows = []
    cursor = start_ms
    page = 0
    while True:
        url = (f"https://api.binance.com/api/v3/klines"
               f"?symbol={symbol}&interval=1m&startTime={cursor}&limit=1000")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"    Page {page} ERROR: {e} — retrying in 3s")
            time.sleep(3)
            continue
        if not data:
            break
        all_rows.extend(data)
        last_ts = data[-1][0]
        page += 1
        if page % 20 == 0:
            print(f"    page {page}: {len(all_rows)} candles, latest "
                  f"{datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).isoformat()}")
        if len(data) < 1000:
            break
        cursor = last_ts + 60_000
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_vol', 'trades', 'tb_base', 'tb_quote', 'ignore'
    ])
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms', utc=True).dt.tz_localize(None)
    for c in ('open', 'high', 'low', 'close', 'volume', 'tb_base'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # tb_base = taker buy base asset volume = aggressive BUY volume per minute.
    # Used by VPIN proxy: aggressive_sell = volume - tb_base.
    df = df.rename(columns={'tb_base': 'taker_buy_base_volume'})
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base_volume']]
    df = df.drop_duplicates(subset='datetime').sort_values('datetime').reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', default='ETH')
    ap.add_argument('--days', type=int, default=60)
    args = ap.parse_args()

    asset = args.asset.upper()
    if asset not in SYMBOL_MAP:
        print(f"Unknown asset {asset}; supported: {list(SYMBOL_MAP)}")
        sys.exit(1)

    sym = SYMBOL_MAP[asset]
    out = os.path.join(DATA_DIR, f"{asset.lower()}_1m_data.csv")

    print(f"Downloading {sym} 1m candles, last {args.days} days...")
    t0 = time.time()
    df = fetch_klines_1m(sym, args.days)
    if df.empty:
        print("NO DATA — abort")
        sys.exit(1)

    df.to_csv(out, index=False)
    elapsed = time.time() - t0
    print(f"\n  Saved {len(df):,} candles to {out}")
    print(f"  Range: {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
