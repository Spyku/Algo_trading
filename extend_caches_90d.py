"""Extend ETH caches to 90 days for multi-window rally-cooldown backtest.

- 5m candles: paginates Binance REST backwards to cover 91 days.
- Hourly signals: re-runs production model inference over 91 days worth of
  hours and merges with existing cache.

Outputs (independent of the 30d caches):
  data/eth_5m_backtest_90d.csv
  data/eth_sl_signals_90d.pkl
"""
from __future__ import annotations

import os
import sys
import json
import time
import pickle
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ENGINE_DIR)
sys.path.insert(0, ENGINE_DIR)

FIVE_MIN_90D_CSV   = os.path.join(ENGINE_DIR, 'data', 'eth_5m_backtest_90d.csv')
SIG_90D_PKL        = os.path.join(ENGINE_DIR, 'data', 'eth_sl_signals_90d.pkl')
STANDARD_PROD_CSV  = os.path.join(ENGINE_DIR, 'models', 'crypto_ed_production.csv')
REGIME_CONFIG_PATH = os.path.join(ENGINE_DIR, 'config', 'regime_config_ed.json')

ASSET = 'ETH'
DAYS  = 91
REPLAY_HOURS = DAYS * 24

BINANCE_BASE = 'https://api.binance.com/api/v3/klines'
BINANCE_SYMBOL = 'ETHUSDT'
BINANCE_INTERVAL = '5m'
BINANCE_LIMIT = 1000


def _binance_fetch(start_ms: int, end_ms: int):
    url = (
        f"{BINANCE_BASE}?symbol={BINANCE_SYMBOL}&interval={BINANCE_INTERVAL}"
        f"&startTime={start_ms}&endTime={end_ms}&limit={BINANCE_LIMIT}"
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'backtest/1.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def download_5m_90d():
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=DAYS + 1)
    start_ms = int(start.timestamp() * 1000)
    end_ms   = int(now.timestamp() * 1000)

    if os.path.exists(FIVE_MIN_90D_CSV):
        try:
            ex = pd.read_csv(FIVE_MIN_90D_CSV)
            ex['datetime'] = pd.to_datetime(ex['datetime'], utc=True)
            want_start = pd.Timestamp(start) + pd.Timedelta(hours=2)
            want_end   = pd.Timestamp(now)   - pd.Timedelta(hours=2)
            if ex['datetime'].min() <= want_start and ex['datetime'].max() >= want_end:
                print(f"  [5m] reuse cache {len(ex)} rows "
                      f"{ex['datetime'].min()} -> {ex['datetime'].max()}")
                return ex
        except Exception as e:
            print(f"  [5m] cache unusable ({e}); re-downloading")

    print(f"  [5m] downloading {DAYS}d from Binance ({start} -> {now})...")
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _binance_fetch(cursor, end_ms)
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        cursor = last_open + 5 * 60 * 1000
        if len(batch) < BINANCE_LIMIT:
            break
        time.sleep(0.25)

    if not rows:
        raise RuntimeError("Binance returned 0 candles")

    df = pd.DataFrame(rows, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore',
    ])
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df = df.drop_duplicates(subset='datetime').sort_values('datetime').reset_index(drop=True)
    df.to_csv(FIVE_MIN_90D_CSV, index=False)
    print(f"  [5m] saved {len(df)} rows -> {FIVE_MIN_90D_CSV}")
    return df


def pick_model_row(df_models, asset, horizon):
    rows = df_models[(df_models['coin'] == asset) & (df_models['horizon'] == horizon)]
    if rows.empty:
        raise ValueError(f"No model for {asset} {horizon}h")
    return rows.sort_values('combined_score', ascending=False).iloc[0]


def generate_horizon_signals(asset, row, replay_hours):
    from crypto_trading_system_ed import generate_signals, _suppress_stderr
    feats = None
    if 'optimal_features' in row and pd.notna(row['optimal_features']):
        feats = [f.strip() for f in str(row['optimal_features']).split(',') if f.strip()]
    gamma = float(row['gamma']) if pd.notna(row.get('gamma', None)) else 1.0
    models = str(row['models']).split('+')
    window = int(row['best_window'])
    horizon = int(row['horizon'])
    print(f"    {asset} {horizon}h | models={row['models']} | w={window} | gamma={gamma}")
    with _suppress_stderr():
        sigs = generate_signals(asset, models, window, replay_hours,
                                feature_override=feats, horizon=horizon, gamma=gamma)
    out = {}
    for s in sigs:
        dt = s['datetime']
        if isinstance(dt, str):
            dt = datetime.strptime(dt, '%Y-%m-%d %H:%M')
        out[pd.Timestamp(dt)] = s
    return out


def merge_signals(asset, bull_sigs, bull_thr, bull_h, bear_sigs, bear_thr, bear_h, detector_name='tsmom_672h'):
    from crypto_trading_system_ed import _build_regime_indicators_and_detectors
    ind, detectors = _build_regime_indicators_and_detectors(asset)
    if detector_name not in detectors:
        print(f"  [sig] WARN: detector '{detector_name}' not found, using tsmom_672h")
        detector_name = 'tsmom_672h'
    detector = detectors[detector_name]
    all_dts = sorted(set(bull_sigs.keys()) | set(bear_sigs.keys()))
    merged = []
    for dt in all_dts:
        is_bull = bool(detector(dt))
        regime = 'bull' if is_bull else 'bear'
        src = bull_sigs if is_bull else bear_sigs
        thr = bull_thr if is_bull else bear_thr
        h = bull_h if is_bull else bear_h
        s = src.get(dt)
        if s is None:
            other = (bear_sigs if is_bull else bull_sigs).get(dt)
            if other is None:
                continue
            merged.append({
                'datetime': dt, 'close': float(other['close']),
                'signal': 'HOLD', 'confidence': 0.0,
                'conf_threshold': thr, 'regime': regime, 'horizon': h,
            })
            continue
        merged.append({
            'datetime': dt, 'close': float(s['close']),
            'signal': s['signal'], 'confidence': float(s['confidence']),
            'conf_threshold': thr, 'regime': regime, 'horizon': h,
        })
    return merged


def generate_signals_90d():
    if os.path.exists(SIG_90D_PKL):
        with open(SIG_90D_PKL, 'rb') as f:
            sigs = pickle.load(f)
        print(f"  [sig] reuse cache {len(sigs)} signals "
              f"{sigs[0]['datetime']} -> {sigs[-1]['datetime']}")
        return sigs

    print(f"  [sig] generating signals for {DAYS}d ({REPLAY_HOURS}h)...")
    # Read regime config for horizons + conf thresholds (set by HRS + Mode T)
    with open(REGIME_CONFIG_PATH) as f:
        regime_cfg = json.load(f)
    asset_cfg = regime_cfg.get(ASSET, {})
    bull_h = int(asset_cfg.get('bull', {}).get('horizon', 6))
    bear_h = int(asset_cfg.get('bear', {}).get('horizon', 8))
    bull_thr = float(asset_cfg.get('bull', {}).get('min_confidence', 85))
    bear_thr = float(asset_cfg.get('bear', {}).get('min_confidence', 65))
    detector_name = (asset_cfg.get('regime_detector') or {}).get('params', {}).get('name', 'tsmom_672h')
    print(f"  [sig] per regime_config: bull={bull_h}h@{bull_thr}% | bear={bear_h}h@{bear_thr}% | detector={detector_name}")

    dfm = pd.read_csv(STANDARD_PROD_CSV)
    bull_row = pick_model_row(dfm, ASSET, bull_h)
    bear_row = pick_model_row(dfm, ASSET, bear_h)

    bull_sigs = generate_horizon_signals(ASSET, bull_row, REPLAY_HOURS)
    bear_sigs = generate_horizon_signals(ASSET, bear_row, REPLAY_HOURS)
    merged = merge_signals(ASSET, bull_sigs, bull_thr, bull_h, bear_sigs, bear_thr, bear_h, detector_name=detector_name)
    print(f"  [sig] merged {len(merged)} hourly signals "
          f"{merged[0]['datetime']} -> {merged[-1]['datetime']}")
    with open(SIG_90D_PKL, 'wb') as f:
        pickle.dump(merged, f)
    print(f"  [sig] saved -> {SIG_90D_PKL}")
    return merged


if __name__ == '__main__':
    print("=" * 80)
    print("  Extending ETH caches to 90 days")
    print("=" * 80)
    df5 = download_5m_90d()
    sigs = generate_signals_90d()
    print(f"\n  DONE. 5m rows={len(df5)}  signals={len(sigs)}")
