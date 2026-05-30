"""tools/test_c32_to_c40_batch.py — Sequential Desktop runner for C32, C33, C34,
C36, C37, C38, C39, C40 (8 untested ideas from the canonical scoreboard).

Per-idea verdicts via Mode D smoke test (ETH 5,6,7,8h --replay 1440 --no-persist
--no-data-update --grid-tag <CID>). Each idea writes its features via a
monkey-patched build_all_features (same harness pattern as test_14_ideas.py +
test_c04_to_c08_runner.py), runs Mode D, compares the tagged grid winner to
the untagged baseline grid winner.

PRE-FLIGHT before the Mode D loop:
  - C32 (liquidation cascade): no download — uses existing OI + returns + vol
    to PROXY cascade events. Direct liquidation feed (CoinGlass) needs paid API.
  - C36 (CryptoPanic + Santiment): STUB — both APIs need keys this codebase
    doesn't have. Marks idea as BLOCKED, skips Mode D.
  - C38 (Korea/Coinbase premium): downloads Bithumb ETH/KRW daily candles +
    Coinbase ETH/USD daily candles + yfinance KRW/USD. Daily, not hourly —
    fine for smoke test feasibility.
  - C39 (Binance long/short ratio): downloads /futures/data/topLongShortAccountRatio
    (public, 30d retention).
  - All others (C33, C34, C37, C40): use existing cached data.

Decision per idea: avg APF delta vs baseline ≥ +5pp → PASS (escalate to HRST);
within ±5pp → MARGINAL (note in scoreboard, low promote prior); ≤ -5pp → FAIL
(mark DEAD in scoreboard).

Output:
  logs/c32_to_c40_summary_<ts>.txt   — verdicts per idea
  logs/c32_to_c40_<idea>_<h>h_<ts>.log — per-idea per-horizon Mode D log
  models/crypto_ed_grid_ETH_<h>h_<CID>.csv — tagged grid CSV per idea
  data/macro_data/binance_long_short_ratio.csv — fetched if missing
  data/macro_data/korea_premium.csv — fetched if missing

Run (Desktop, ~5-7h overnight):
  python tools/test_c32_to_c40_batch.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Optional

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)
sys.path.insert(0, os.path.join(ENGINE, 'tools'))

from test_14_ideas import (
    write_patcher, run_mode_d, load_grid_csv, load_baseline_grid,
    compare_grid_winners, LOGS_DIR, MODELS_DIR, TS,
)

ASSET = 'ETH'
HORIZONS = [5, 6, 7, 8]
REPLAY = 1440
MACRO_DIR = os.path.join(ENGINE, 'data', 'macro_data')
SUMMARY_PATH = os.path.join(LOGS_DIR, f'c32_to_c40_summary_{TS}.txt')


def _summary_write(line: str):
    print(line, flush=True)
    with open(SUMMARY_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# =============================================================================
# DATA FETCHERS — run before the Mode D loop so failures surface early
# =============================================================================

def fetch_c39_long_short_ratio(asset: str = 'ETHUSDT', period: str = '1h',
                                limit: int = 500) -> bool:
    """C39 — Binance Futures long/short account ratio. Public endpoint, ~30d
    retention. Saves to data/macro_data/binance_long_short_<asset>.csv.
    Returns True on success, False otherwise.
    """
    out_path = os.path.join(MACRO_DIR, f'binance_long_short_{asset.lower()}.csv')
    url = ('https://fapi.binance.com/futures/data/topLongShortAccountRatio'
           f'?symbol={asset}&period={period}&limit={limit}')
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        _summary_write(f'  [C39] long/short download FAILED: {e}')
        return False
    if not data:
        _summary_write('  [C39] long/short returned empty')
        return False
    import pandas as pd
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(None)
    df = df[['datetime', 'longShortRatio', 'longAccount', 'shortAccount']].copy()
    df.columns = ['datetime', 'long_short_ratio', 'long_pct', 'short_pct']
    df['long_short_ratio'] = df['long_short_ratio'].astype(float)
    df['long_pct'] = df['long_pct'].astype(float)
    df['short_pct'] = df['short_pct'].astype(float)
    os.makedirs(MACRO_DIR, exist_ok=True)
    df.sort_values('datetime').to_csv(out_path, index=False)
    _summary_write(f'  [C39] long/short saved: {len(df)} rows -> {out_path}')
    return True


def fetch_c38_korea_premium() -> bool:
    """C38 — Korea/Coinbase premium spread. Daily candles, not hourly.
    Bithumb ETH/KRW + Coinbase ETH/USD + yfinance KRW/USD. Saves to
    data/macro_data/korea_premium.csv with columns:
      date, eth_krw, eth_usd_coinbase, krw_per_usd, premium_pct
    Returns True on success, False otherwise.
    """
    out_path = os.path.join(MACRO_DIR, 'korea_premium.csv')
    import pandas as pd
    # Bithumb: /public/candlestick/{symbol}/24h
    try:
        url = 'https://api.bithumb.com/public/candlestick/ETH_KRW/24h'
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode())
        if payload.get('status') != '0000':
            _summary_write(f'  [C38] Bithumb status={payload.get("status")} — abort')
            return False
        rows = payload.get('data', [])
        # rows: [timestamp_ms, open, close, high, low, volume]
        bdf = pd.DataFrame(rows, columns=['ts', 'o', 'c', 'h', 'l', 'v'])
        bdf['date'] = pd.to_datetime(bdf['ts'].astype(int), unit='ms', utc=True).dt.date
        bdf['eth_krw'] = bdf['c'].astype(float)
        bdf = bdf[['date', 'eth_krw']].drop_duplicates('date').sort_values('date')
    except Exception as e:
        _summary_write(f'  [C38] Bithumb download FAILED: {e}')
        return False
    # Coinbase Exchange: /products/ETH-USD/candles?granularity=86400
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=300)
        url = ('https://api.exchange.coinbase.com/products/ETH-USD/candles'
               f'?granularity=86400&start={start.strftime("%Y-%m-%dT%H:%M:%SZ")}'
               f'&end={end.strftime("%Y-%m-%dT%H:%M:%SZ")}')
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            rows = json.loads(resp.read().decode())
        # Coinbase: [time, low, high, open, close, volume]
        cdf = pd.DataFrame(rows, columns=['ts', 'low', 'high', 'open', 'close', 'volume'])
        cdf['date'] = pd.to_datetime(cdf['ts'].astype(int), unit='s', utc=True).dt.date
        cdf['eth_usd_coinbase'] = cdf['close'].astype(float)
        cdf = cdf[['date', 'eth_usd_coinbase']].drop_duplicates('date').sort_values('date')
    except Exception as e:
        _summary_write(f'  [C38] Coinbase download FAILED: {e}')
        return False
    # yfinance KRW/USD daily
    try:
        import yfinance as yf
        krw = yf.download('KRW=X', period='1y', interval='1d', progress=False, auto_adjust=False)
        if krw is None or len(krw) == 0:
            _summary_write('  [C38] yfinance KRW=X empty')
            return False
        krw = krw.reset_index()
        # yfinance Close column might be MultiIndex
        if isinstance(krw.columns, pd.MultiIndex):
            krw.columns = [c[0] if c[1] == '' or c[1] == 'KRW=X' else c[0] for c in krw.columns]
        krw['date'] = pd.to_datetime(krw['Date']).dt.date
        krw['krw_per_usd'] = krw['Close'].astype(float)
        krw = krw[['date', 'krw_per_usd']]
    except Exception as e:
        _summary_write(f'  [C38] yfinance KRW=X FAILED: {e}')
        return False
    merged = bdf.merge(cdf, on='date', how='inner').merge(krw, on='date', how='inner')
    if len(merged) < 30:
        _summary_write(f'  [C38] only {len(merged)} aligned days — too sparse')
        return False
    merged['eth_krw_in_usd'] = merged['eth_krw'] / merged['krw_per_usd']
    merged['premium_pct'] = ((merged['eth_krw_in_usd'] - merged['eth_usd_coinbase'])
                              / merged['eth_usd_coinbase']) * 100
    os.makedirs(MACRO_DIR, exist_ok=True)
    merged.to_csv(out_path, index=False)
    _summary_write(f'  [C38] korea premium saved: {len(merged)} days -> {out_path}')
    return True


# =============================================================================
# PATCHER CODE — each idea adds features via a monkey-patched build_all_features
# =============================================================================

C32_PATCHER = '''
"""C32 — Liquidation cascade event PROXY (no direct liquidation feed).

True liquidation data needs CoinGlass paid API. Proxy a cascade event from
existing signals: large negative 1h return AND 24h vol > rolling p90 AND OI
1h change < -3%. Adds 3 features:
  liq_cascade_event_h     binary (1 if cascade in current hour)
  liq_cascade_count_24h   count of cascade events in last 24h
  liq_cascade_dist_h      hours since most recent cascade (capped at 168)
"""
import os
import numpy as np
import pandas as pd
import crypto_trading_system_ed as eng

_orig_build = eng.build_all_features


def _compute_cascade(df: pd.DataFrame) -> dict:
    if 'close' not in df.columns:
        return {}
    out = {}
    n = len(df)
    logret_1h = np.log(df['close'].astype(float)).diff().fillna(0).values
    vol_24h = pd.Series(logret_1h).rolling(24).std().values
    vol_24h_p90 = pd.Series(vol_24h).rolling(168, min_periods=24).quantile(0.90).values
    # OI 1h change — pull from derivatives_eth.csv if available
    oi_chg_1h = np.zeros(n)
    deriv_path = os.path.join(os.path.dirname(eng.__file__), 'data', 'macro_data', 'derivatives_eth.csv')
    if os.path.exists(deriv_path):
        try:
            ddf = pd.read_csv(deriv_path)
            if 'datetime' in ddf.columns and 'open_interest' in ddf.columns:
                ddf['datetime'] = pd.to_datetime(ddf['datetime'])
                ddf = ddf.set_index('datetime').sort_index()
                ddf['oi_chg'] = ddf['open_interest'].pct_change().fillna(0)
                df_dt = pd.to_datetime(df['datetime']) if 'datetime' in df.columns else None
                if df_dt is not None:
                    aligned = ddf['oi_chg'].reindex(df_dt, method='ffill').values
                    oi_chg_1h = np.nan_to_num(aligned, nan=0.0)
        except Exception:
            pass
    cascade_thresh_ret = -0.015  # -1.5% 1h return
    cascade_thresh_oi = -0.03    # -3% 1h OI drop
    is_cascade = (
        (logret_1h < cascade_thresh_ret)
        & (vol_24h > vol_24h_p90)
        & (oi_chg_1h < cascade_thresh_oi)
    ).astype(np.float64)
    out['liq_cascade_event_h'] = is_cascade
    out['liq_cascade_count_24h'] = pd.Series(is_cascade).rolling(24, min_periods=1).sum().values
    # hours since most recent cascade, capped at 168
    dist = np.full(n, 168.0)
    last = -1
    for i in range(n):
        if is_cascade[i] > 0.5:
            last = i
        if last >= 0:
            dist[i] = min(168.0, float(i - last))
    out['liq_cascade_dist_h'] = dist
    return out


def _patched_build(*args, **kwargs):
    result = _orig_build(*args, **kwargs)
    if isinstance(result, tuple) and len(result) == 3:
        df, all_cols, lead_cols = result
    elif isinstance(result, tuple) and len(result) == 2:
        df, all_cols = result
        lead_cols = None
    else:
        return result
    add = _compute_cascade(df)
    n_added = 0
    for name, vals in add.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            n_added += 1
    if n_added:
        print(f'[C32] liquidation cascade features added: +{n_added}')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    return df, all_cols


eng.build_all_features = _patched_build
print('[C32] build_all_features patched')
'''


C33_PATCHER = '''
"""C33 — Bid-ask spread compression z-score. Distinct from quarantined raw
spread_bps (which is in always_disabled_exact). Reads
data/macro_data/orderbook_snapshots.csv, computes hourly spread (bps), then
rolling 168h z-score. Adds 1 feature: ob_spread_compression_zscore.
"""
import os
import numpy as np
import pandas as pd
import crypto_trading_system_ed as eng

_orig_build = eng.build_all_features


def _compute_spread_z(df: pd.DataFrame) -> dict:
    if 'datetime' not in df.columns:
        return {}
    ob_path = os.path.join(os.path.dirname(eng.__file__), 'data', 'macro_data', 'orderbook_snapshots.csv')
    if not os.path.exists(ob_path):
        return {}
    try:
        ob = pd.read_csv(ob_path)
        if 'datetime' not in ob.columns or 'spread_bps' not in ob.columns:
            return {}
        ob['datetime'] = pd.to_datetime(ob['datetime'])
        # ETH only
        if 'symbol' in ob.columns:
            ob = ob[ob['symbol'].str.upper().str.startswith('ETH')]
        ob = ob.sort_values('datetime').drop_duplicates('datetime')
        ob = ob.set_index('datetime')
        spread = ob['spread_bps'].astype(float)
        # rolling z-score
        m = spread.rolling(168, min_periods=24).mean()
        s = spread.rolling(168, min_periods=24).std()
        z = (spread - m) / s.replace(0, np.nan)
        z = z.fillna(0)
        df_dt = pd.to_datetime(df['datetime'])
        aligned = z.reindex(df_dt, method='ffill').fillna(0).values
        return {'ob_spread_compression_zscore': aligned}
    except Exception as e:
        print(f'[C33] FAILED: {e}')
        return {}


def _patched_build(*args, **kwargs):
    result = _orig_build(*args, **kwargs)
    if isinstance(result, tuple) and len(result) == 3:
        df, all_cols, lead_cols = result
    elif isinstance(result, tuple) and len(result) == 2:
        df, all_cols = result
        lead_cols = None
    else:
        return result
    add = _compute_spread_z(df)
    n_added = 0
    for name, vals in add.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            n_added += 1
    if n_added:
        print(f'[C33] spread compression z-score added: +{n_added}')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    return df, all_cols


eng.build_all_features = _patched_build
print('[C33] build_all_features patched')
'''


C34_PATCHER = '''
"""C34 — ETH-BTC cointegration residual (Engle-Granger). Rolling 168h regression
of log(ETH) on log(BTC); residual = log(ETH) - (alpha + beta*log(BTC)).
Distinct from existing xa_btc_lag1h/2h/3h (intraday lead-lag). Cointegration
captures equilibrium-deviation, not short-term lead-lag.
Adds 1 feature: xa_eth_btc_coint_resid_168h.
"""
import os
import numpy as np
import pandas as pd
import crypto_trading_system_ed as eng

_orig_build = eng.build_all_features


def _compute_coint_resid(df: pd.DataFrame) -> dict:
    if 'datetime' not in df.columns or 'close' not in df.columns:
        return {}
    btc_path = os.path.join(os.path.dirname(eng.__file__), 'data', 'btc_hourly_data.csv')
    if not os.path.exists(btc_path):
        return {}
    try:
        btc = pd.read_csv(btc_path)
        btc['datetime'] = pd.to_datetime(btc['datetime'])
        btc = btc.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
        df_dt = pd.to_datetime(df['datetime'])
        btc_close = btc['close'].reindex(df_dt, method='ffill').values.astype(float)
        eth_close = df['close'].astype(float).values
        n = len(df)
        log_eth = np.log(eth_close)
        log_btc = np.log(btc_close)
        window = 168
        resid = np.zeros(n)
        for i in range(window, n):
            x = log_btc[i - window:i]
            y = log_eth[i - window:i]
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            # OLS y = a + b*x
            x_mean = x.mean(); y_mean = y.mean()
            cov = ((x - x_mean) * (y - y_mean)).sum()
            var = ((x - x_mean) ** 2).sum()
            if var == 0:
                continue
            beta = cov / var
            alpha = y_mean - beta * x_mean
            resid[i] = log_eth[i] - (alpha + beta * log_btc[i])
        return {'xa_eth_btc_coint_resid_168h': resid}
    except Exception as e:
        print(f'[C34] FAILED: {e}')
        return {}


def _patched_build(*args, **kwargs):
    result = _orig_build(*args, **kwargs)
    if isinstance(result, tuple) and len(result) == 3:
        df, all_cols, lead_cols = result
    elif isinstance(result, tuple) and len(result) == 2:
        df, all_cols = result
        lead_cols = None
    else:
        return result
    add = _compute_coint_resid(df)
    n_added = 0
    for name, vals in add.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            n_added += 1
    if n_added:
        print(f'[C34] cointegration residual added: +{n_added}')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    return df, all_cols


eng.build_all_features = _patched_build
print('[C34] build_all_features patched')
'''


C36_STUB_NOTE = """C36 — News/social sentiment (CryptoPanic + Santiment).
SKIPPED: both APIs require API keys this codebase doesn't have.
To revive:
  1. Sign up for CryptoPanic free tier (limited rate)
  2. Sign up for Santiment Sanbase free tier (limited metrics)
  3. Add fetcher to download_macro_data.py
  4. Replace this STUB with a real patcher that reads sentiment_polarity.csv
     and adds features sentiment_polarity_4h, sentiment_polarity_chg24h, etc.
This is distinct from C28 GDELT (DEAD) — C36 is asset-specific crypto sentiment,
not geopolitical-event-volume.
"""


C37_PATCHER = '''
"""C37 — Stablecoin issuance velocity (1st difference of mcap).

Existing stable_mcap_chg1d / chg7d / zscore are LEVEL-based. Velocity is the
1st difference, captures issuance/redemption rate. CLAUDE.md notes the existing
stablecoin_flows.csv is already loaded. Adds:
  stable_mcap_velocity_1d   1st diff of mcap_total (1-day)
  stable_mcap_velocity_7d   1st diff of mcap_total (7-day rolling)
"""
import os
import numpy as np
import pandas as pd
import crypto_trading_system_ed as eng

_orig_build = eng.build_all_features


def _compute_velocity(df: pd.DataFrame) -> dict:
    if 'datetime' not in df.columns:
        return {}
    sf_path = os.path.join(os.path.dirname(eng.__file__), 'data', 'macro_data', 'stablecoin_flows.csv')
    if not os.path.exists(sf_path):
        return {}
    try:
        sf = pd.read_csv(sf_path)
        # column may be 'date' or 'datetime'
        date_col = 'date' if 'date' in sf.columns else ('datetime' if 'datetime' in sf.columns else None)
        if date_col is None:
            return {}
        sf['date'] = pd.to_datetime(sf[date_col]).dt.date
        # use total mcap (USDT + USDC if present)
        mcap_col = None
        for c in ['mcap_total', 'total_mcap', 'usdt_usdc_mcap']:
            if c in sf.columns:
                mcap_col = c; break
        if mcap_col is None:
            cands = [c for c in sf.columns if 'mcap' in c.lower()]
            if cands:
                sf['_mcap'] = sf[cands].sum(axis=1)
                mcap_col = '_mcap'
            else:
                return {}
        sf = sf.sort_values('date').drop_duplicates('date')
        sf['velocity_1d'] = sf[mcap_col].diff()
        sf['velocity_7d'] = sf[mcap_col].diff(7)
        sf = sf.set_index('date')
        df_dt = pd.to_datetime(df['datetime']).dt.date
        v1 = pd.Series(df_dt).map(sf['velocity_1d']).fillna(0).values
        v7 = pd.Series(df_dt).map(sf['velocity_7d']).fillna(0).values
        return {
            'stable_mcap_velocity_1d': v1,
            'stable_mcap_velocity_7d': v7,
        }
    except Exception as e:
        print(f'[C37] FAILED: {e}')
        return {}


def _patched_build(*args, **kwargs):
    result = _orig_build(*args, **kwargs)
    if isinstance(result, tuple) and len(result) == 3:
        df, all_cols, lead_cols = result
    elif isinstance(result, tuple) and len(result) == 2:
        df, all_cols = result
        lead_cols = None
    else:
        return result
    add = _compute_velocity(df)
    n_added = 0
    for name, vals in add.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            n_added += 1
    if n_added:
        print(f'[C37] stablecoin velocity features added: +{n_added}')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    return df, all_cols


eng.build_all_features = _patched_build
print('[C37] build_all_features patched')
'''


C38_PATCHER = '''
"""C38 — Korea/Coinbase premium spread feature.
Reads data/macro_data/korea_premium.csv (downloaded by the runner pre-flight).
Daily resolution forward-filled to hourly. Adds 1 feature:
  korea_premium_pct_lagged   (last available daily premium aligned to hour)
"""
import os
import numpy as np
import pandas as pd
import crypto_trading_system_ed as eng

_orig_build = eng.build_all_features


def _compute_korea_premium(df: pd.DataFrame) -> dict:
    if 'datetime' not in df.columns:
        return {}
    kp_path = os.path.join(os.path.dirname(eng.__file__), 'data', 'macro_data', 'korea_premium.csv')
    if not os.path.exists(kp_path):
        return {}
    try:
        kp = pd.read_csv(kp_path)
        kp['date'] = pd.to_datetime(kp['date']).dt.date
        kp = kp.sort_values('date').drop_duplicates('date').set_index('date')
        df_dt = pd.to_datetime(df['datetime']).dt.date
        aligned = pd.Series(df_dt).map(kp['premium_pct']).fillna(method='ffill').fillna(0).values
        return {'korea_premium_pct_lagged': aligned}
    except Exception as e:
        print(f'[C38] FAILED: {e}')
        return {}


def _patched_build(*args, **kwargs):
    result = _orig_build(*args, **kwargs)
    if isinstance(result, tuple) and len(result) == 3:
        df, all_cols, lead_cols = result
    elif isinstance(result, tuple) and len(result) == 2:
        df, all_cols = result
        lead_cols = None
    else:
        return result
    add = _compute_korea_premium(df)
    n_added = 0
    for name, vals in add.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            n_added += 1
    if n_added:
        print(f'[C38] korea premium features added: +{n_added}')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    return df, all_cols


eng.build_all_features = _patched_build
print('[C38] build_all_features patched')
'''


C39_PATCHER = '''
"""C39 — Binance Futures long/short account ratio.
Reads data/macro_data/binance_long_short_ethusdt.csv (downloaded by runner
pre-flight). Adds:
  ls_ratio                   raw ratio
  ls_ratio_chg24h            24h change
  ls_ratio_zscore_168h       rolling z-score
"""
import os
import numpy as np
import pandas as pd
import crypto_trading_system_ed as eng

_orig_build = eng.build_all_features


def _compute_ls_ratio(df: pd.DataFrame) -> dict:
    if 'datetime' not in df.columns:
        return {}
    ls_path = os.path.join(os.path.dirname(eng.__file__), 'data', 'macro_data', 'binance_long_short_ethusdt.csv')
    if not os.path.exists(ls_path):
        return {}
    try:
        ls = pd.read_csv(ls_path)
        ls['datetime'] = pd.to_datetime(ls['datetime'])
        ls = ls.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
        ratio = ls['long_short_ratio'].astype(float)
        chg24 = ratio.pct_change(24).fillna(0)
        m = ratio.rolling(168, min_periods=24).mean()
        s = ratio.rolling(168, min_periods=24).std()
        z = ((ratio - m) / s.replace(0, np.nan)).fillna(0)
        df_dt = pd.to_datetime(df['datetime'])
        out = {
            'ls_ratio': ratio.reindex(df_dt, method='ffill').fillna(0).values,
            'ls_ratio_chg24h': chg24.reindex(df_dt, method='ffill').fillna(0).values,
            'ls_ratio_zscore_168h': z.reindex(df_dt, method='ffill').fillna(0).values,
        }
        return out
    except Exception as e:
        print(f'[C39] FAILED: {e}')
        return {}


def _patched_build(*args, **kwargs):
    result = _orig_build(*args, **kwargs)
    if isinstance(result, tuple) and len(result) == 3:
        df, all_cols, lead_cols = result
    elif isinstance(result, tuple) and len(result) == 2:
        df, all_cols = result
        lead_cols = None
    else:
        return result
    add = _compute_ls_ratio(df)
    n_added = 0
    for name, vals in add.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            n_added += 1
    if n_added:
        print(f'[C39] long/short ratio features added: +{n_added}')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    return df, all_cols


eng.build_all_features = _patched_build
print('[C39] build_all_features patched')
'''


C40_PATCHER = '''
"""C40 — Skewness + kurtosis features (3rd / 4th moments). Currently absent.
Computed on rolling logret window. Adds:
  ret_skew_24h, ret_skew_72h
  ret_kurt_24h, ret_kurt_72h
"""
import numpy as np
import pandas as pd
import crypto_trading_system_ed as eng

_orig_build = eng.build_all_features


def _compute_moments(df: pd.DataFrame) -> dict:
    if 'close' not in df.columns:
        return {}
    logret = np.log(df['close'].astype(float)).diff()
    out = {}
    out['ret_skew_24h'] = logret.rolling(24).skew().fillna(0).values
    out['ret_skew_72h'] = logret.rolling(72).skew().fillna(0).values
    out['ret_kurt_24h'] = logret.rolling(24).kurt().fillna(0).values
    out['ret_kurt_72h'] = logret.rolling(72).kurt().fillna(0).values
    return out


def _patched_build(*args, **kwargs):
    result = _orig_build(*args, **kwargs)
    if isinstance(result, tuple) and len(result) == 3:
        df, all_cols, lead_cols = result
    elif isinstance(result, tuple) and len(result) == 2:
        df, all_cols = result
        lead_cols = None
    else:
        return result
    add = _compute_moments(df)
    n_added = 0
    for name, vals in add.items():
        df[name] = vals
        if all_cols is not None and name not in all_cols:
            all_cols.append(name)
            n_added += 1
    if n_added:
        print(f'[C40] skewness/kurtosis features added: +{n_added}')
    if lead_cols is not None:
        return df, all_cols, lead_cols
    return df, all_cols


eng.build_all_features = _patched_build
print('[C40] build_all_features patched')
'''


# =============================================================================
# IDEA RUNNER
# =============================================================================

IDEAS = [
    ('C32', 'liquidation_cascade',         C32_PATCHER, True),
    ('C33', 'spread_compression',          C33_PATCHER, True),
    ('C34', 'eth_btc_cointegration',       C34_PATCHER, True),
    ('C36', 'news_sentiment',              None,         False),  # STUB
    ('C37', 'stablecoin_velocity',         C37_PATCHER, True),
    ('C38', 'korea_premium',               C38_PATCHER, True),
    ('C39', 'long_short_ratio',            C39_PATCHER, True),
    ('C40', 'skew_kurt',                   C40_PATCHER, True),
]


def run_idea(cid: str, name: str, patcher_code: Optional[str], active: bool) -> dict:
    _summary_write('=' * 100)
    _summary_write(f'  {cid} — {name}')
    _summary_write('=' * 100)

    if not active or patcher_code is None:
        if cid == 'C36':
            _summary_write(C36_STUB_NOTE)
        else:
            _summary_write(f'  {cid}: STUB / not active')
        return {'cid': cid, 'name': name, 'verdict': 'SKIPPED', 'avg_delta': float('nan'),
                'per_horizon': []}

    write_patcher(name, patcher_code)
    _summary_write(f'  Patcher: _idea_patchers/{name}.py')

    deltas = []
    per_horizon = []
    for h in HORIZONS:
        log_path = os.path.join(LOGS_DIR, f'c32_to_c40_{cid}_{name}_{h}h_{TS}.log')
        _summary_write(f'  >> {cid} Mode D ETH {h}h (replay={REPLAY}h) -> {log_path}')
        rc = run_mode_d(ASSET, h, REPLAY, cid,
                        f'_idea_patchers.{name}', log_path)
        if rc != 0:
            _summary_write(f'     ERROR rc={rc} (see log)')
            per_horizon.append((h, None, None, None))
            continue
        test_df, _ = load_grid_csv(ASSET, h, cid)
        base_df, _ = load_baseline_grid(ASSET, h)
        if test_df is None or base_df is None:
            _summary_write(f'     ERROR loading grid CSVs')
            per_horizon.append((h, None, None, None))
            continue
        tw, bw, delta = compare_grid_winners(test_df, base_df)
        if delta is None:
            _summary_write(f'     ERROR comparison failed')
            per_horizon.append((h, None, None, None))
            continue
        per_horizon.append((h, tw, bw, delta))
        deltas.append(delta)
        _summary_write(f'     {h}h: test_apf={tw:.3f}  base_apf={bw:.3f}  delta={delta:+.3f}')

    if not deltas:
        verdict = 'ERROR'
        avg = float('nan')
    else:
        avg = sum(deltas) / len(deltas)
        if avg >= 5.0:
            verdict = 'PASS'
        elif avg >= -5.0:
            verdict = 'MARGINAL'
        else:
            verdict = 'FAIL'
    _summary_write(f'  {cid} verdict: avg APF delta = {avg:+.3f} -> {verdict}')
    _summary_write('')
    return {'cid': cid, 'name': name, 'verdict': verdict, 'avg_delta': avg,
            'per_horizon': per_horizon}


def main():
    _summary_write('=' * 100)
    _summary_write(f'  C32-C40 BATCH RUNNER — {datetime.now().isoformat()}')
    _summary_write(f'  Asset: {ASSET}  Horizons: {HORIZONS}  Replay: {REPLAY}h --no-persist --no-data-update')
    _summary_write(f'  Summary: {SUMMARY_PATH}')
    _summary_write('=' * 100)

    # --- PRE-FLIGHT: download data for ideas that need new sources ---
    _summary_write('')
    _summary_write('  PRE-FLIGHT — fetching data for ideas with new sources')
    _summary_write('-' * 100)
    fetch_c39_long_short_ratio()
    fetch_c38_korea_premium()
    _summary_write('')
    _summary_write('  Pre-flight complete; starting Mode D loop.')
    _summary_write('')

    # --- IDEA LOOP ---
    results = []
    for cid, name, patcher, active in IDEAS:
        try:
            res = run_idea(cid, name, patcher, active)
        except Exception as e:
            import traceback
            _summary_write(f'  {cid} EXCEPTION: {e!r}')
            _summary_write(traceback.format_exc())
            res = {'cid': cid, 'name': name, 'verdict': 'EXCEPTION',
                   'avg_delta': float('nan'), 'per_horizon': []}
        results.append(res)

    # --- FINAL SUMMARY ---
    _summary_write('=' * 100)
    _summary_write('  FINAL SUMMARY — C32-C40')
    _summary_write('=' * 100)
    for r in results:
        avg = r['avg_delta']
        avg_s = f'{avg:+.3f}' if avg == avg else 'n/a'
        _summary_write(f"  {r['cid']:<5} {r['name']:<28} verdict={r['verdict']:<10} avg_delta={avg_s}")
        for h, tw, bw, d in r['per_horizon']:
            if d is None:
                _summary_write(f'    {h}h: ERROR')
            else:
                _summary_write(f'    {h}h: test={tw:.3f} base={bw:.3f} delta={d:+.3f}')
    _summary_write('')
    _summary_write('  Decision rule (per CLAUDE.md):')
    _summary_write('    avg APF delta ≥ +5.0  → PASS  → escalate to HRST validation')
    _summary_write('    -5.0 < avg < +5.0     → MARGINAL → record + low promote prior')
    _summary_write('    avg ≤ -5.0            → FAIL  → mark DEAD in canonical scoreboard')


if __name__ == '__main__':
    main()
