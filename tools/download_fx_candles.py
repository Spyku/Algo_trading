"""
download_fx_candles.py — FX hourly candle downloader for the Gren FX engine.

BOOTSTRAP source = yfinance (zero-setup, but ~730d cap and the `=X` FX tickers can be gappy).
For production/deep history switch to OANDA v20 (free demo token, H1 back to 2005) or IBKR
reqHistoricalData (your execution venue) — see the FX plan. This gets a FIRST candle set NOW
so we can test whether the engine + the macro data we ALREADY have find any FX edge.

Writes data/{pair}_hourly_data.csv in the Gren/Faye schema (datetime,open,high,low,close,volume).
FX spot has no real volume → volume=0 (engine guards volume.sum()==0 → vol features go inert).

  python tools/download_fx_candles.py                  # EURUSD
  python tools/download_fx_candles.py EURUSD GBPUSD USDJPY
"""
import sys
import os
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)

YF = {'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'JPY=X', 'AUDUSD': 'AUDUSD=X',
      'USDCHF': 'CHF=X', 'USDCAD': 'CAD=X', 'NZDUSD': 'NZDUSD=X'}


def fetch(pair):
    tk = YF.get(pair, pair + '=X')
    try:
        h = yf.Ticker(tk).history(period='730d', interval='1h', auto_adjust=False)
    except Exception as e:
        print(f"  {pair}: yfinance error ({tk}): {e}")
        return None
    if h is None or h.empty:
        print(f"  {pair}: NO DATA from yfinance ({tk})")
        return None
    h = h.reset_index()
    dtcol = next((c for c in ('Datetime', 'Date', 'index') if c in h.columns), h.columns[0])
    dt = pd.to_datetime(h[dtcol])
    try:
        dt = dt.dt.tz_localize(None)
    except (TypeError, AttributeError):
        dt = dt.dt.tz_convert(None)
    out = pd.DataFrame({
        'datetime': dt,
        'open': h['Open'].values, 'high': h['High'].values,
        'low': h['Low'].values, 'close': h['Close'].values, 'volume': 0,
    }).dropna(subset=['close']).reset_index(drop=True)
    out = out[out['close'] > 0].reset_index(drop=True)
    # clean + single consistent datetime format (drop NaT / epoch-1970 junk) so the engine's
    # load_data parses with one strptime format instead of choking on mixed microsecond rows.
    out['datetime'] = pd.to_datetime(out['datetime'], errors='coerce')
    out = out[out['datetime'].notna() & (out['datetime'].dt.year >= 2000)].reset_index(drop=True)
    out['datetime'] = out['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return out


def quality(pair, df):
    n = len(df)
    span = (df['datetime'].max() - df['datetime'].min()).days or 1
    per_day = n / span
    d = df['datetime'].sort_values()
    gh = d.diff().dt.total_seconds() / 3600.0
    weekday_gaps = int(((gh > 3) & (d.dt.dayofweek < 5)).sum())
    print(f"  {pair}: {n} bars | {df['datetime'].min().date()} -> {df['datetime'].max().date()} "
          f"({span}d) | ~{per_day:.1f} bars/day | weekday >3h gaps: {weekday_gaps}")
    return per_day


def main():
    pairs = sys.argv[1:] or ['EURUSD']
    for p in pairs:
        df = fetch(p)
        if df is None:
            continue
        ppd = quality(p, df)
        path = f"data/{p.lower()}_hourly_data.csv"
        df.to_csv(path, index=False)
        verdict = ("DENSE — fine for a first read" if ppd >= 18 else
                   "THIN but usable" if ppd >= 10 else
                   "TOO SPARSE — needs OANDA/IBKR for a trustworthy read")
        print(f"    saved -> {path} | verdict: {verdict}")


if __name__ == '__main__':
    main()
