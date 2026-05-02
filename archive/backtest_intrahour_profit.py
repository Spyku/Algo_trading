"""Backtest: intra-hour profit-take on ETH prod trader.

Question: over the last 30 days, how much PnL would an intra-hour +0.5%
profit-take have added to the live ETH trader?

Baseline  = current prod (hourly signal at top-of-hour + Hold Shield:
            sell only if PnL >= 0.5% or held >= 10h).
Intra-hr  = same entries, but while invested scan each 5-min HIGH inside
            the hour. If HIGH >= entry * 1.005 exit at entry * 1.005
            (limit order assumption). Otherwise fall back to hourly
            Shield/failsafe.

Run:  python backtest_intrahour_profit.py
"""
from __future__ import annotations

import os
import sys
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ENGINE_DIR)
sys.path.insert(0, ENGINE_DIR)

FIVE_MIN_CSV         = os.path.join(ENGINE_DIR, 'data', 'eth_5m_backtest.csv')
DOOHAN_CSV           = os.path.join(ENGINE_DIR, 'models', 'crypto_doohan_v1_6_production.csv')
STANDARD_PROD_CSV    = os.path.join(ENGINE_DIR, 'models', 'crypto_ed_production.csv')
REGIME_CFG_PATH      = os.path.join(ENGINE_DIR, 'config', 'regime_config_ed.json')

ASSET                = 'ETH'
DAYS                 = 30
REPLAY_HOURS         = DAYS * 24                  # 720
PROFIT_TARGET_PCT    = 0.005                      # +0.5%
MAX_HOLD_HOURS       = 10
MIN_SELL_PNL_PCT     = 0.005                      # 0.5% shield floor
TRADING_FEE          = 0.0005                     # 5 bps/leg realistic maker blend (see ed.py BACKTEST_FEE_PER_LEG)

BINANCE_BASE = 'https://api.binance.com/api/v3/klines'
BINANCE_SYMBOL = 'ETHUSDT'
BINANCE_INTERVAL = '5m'
BINANCE_LIMIT = 1000


# ----------------------------------------------------------------
# Step 1: 5m candles from Binance REST
# ----------------------------------------------------------------
def _binance_fetch(start_ms: int, end_ms: int) -> list[list]:
    url = (
        f"{BINANCE_BASE}?symbol={BINANCE_SYMBOL}&interval={BINANCE_INTERVAL}"
        f"&startTime={start_ms}&endTime={end_ms}&limit={BINANCE_LIMIT}"
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'backtest/1.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def download_5m_candles(days: int = DAYS) -> pd.DataFrame:
    """Download last <days> of 5m ETH/USDT candles from Binance."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=days + 1)        # 1 day buffer
    start_ms = int(start.timestamp() * 1000)
    end_ms   = int(now.timestamp() * 1000)

    # Reuse existing CSV if it fully covers the window
    if os.path.exists(FIVE_MIN_CSV):
        try:
            existing = pd.read_csv(FIVE_MIN_CSV)
            existing['datetime'] = pd.to_datetime(existing['datetime'], utc=True)
            if not existing.empty:
                have_start = existing['datetime'].min()
                have_end   = existing['datetime'].max()
                want_start = pd.Timestamp(start) + pd.Timedelta(hours=2)
                want_end   = pd.Timestamp(now)   - pd.Timedelta(hours=2)
                if have_start <= want_start and have_end >= want_end:
                    print(f"  [5m] reusing cache: {FIVE_MIN_CSV} "
                          f"({have_start} -> {have_end}, {len(existing)} rows)")
                    return existing
        except Exception as e:
            print(f"  [5m] cache unusable ({e}); re-downloading")

    print(f"  [5m] downloading {days}d from Binance ({start} -> {now})...")
    all_rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _binance_fetch(cursor, end_ms)
        if not batch:
            break
        all_rows.extend(batch)
        last_open = batch[-1][0]
        # advance cursor by one step to avoid infinite loops if same bucket returned
        cursor = last_open + 5 * 60 * 1000
        if len(batch) < BINANCE_LIMIT:
            break
        time.sleep(0.25)         # polite pacing

    if not all_rows:
        raise RuntimeError("Binance returned 0 candles")

    df = pd.DataFrame(all_rows, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore',
    ])
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df = df.drop_duplicates(subset='datetime').sort_values('datetime').reset_index(drop=True)

    os.makedirs(os.path.dirname(FIVE_MIN_CSV), exist_ok=True)
    df.to_csv(FIVE_MIN_CSV, index=False)
    print(f"  [5m] saved {len(df)} rows -> {FIVE_MIN_CSV}")
    return df


# ----------------------------------------------------------------
# Step 2 & 3: Replay hourly signals via production models
# ----------------------------------------------------------------
def pick_model_row(df_models: pd.DataFrame, asset: str, horizon: int) -> pd.Series:
    rows = df_models[(df_models['coin'] == asset) & (df_models['horizon'] == horizon)]
    if rows.empty:
        raise ValueError(f"No model for {asset} {horizon}h in provided CSV")
    return rows.sort_values('combined_score', ascending=False).iloc[0]


def generate_horizon_signals(asset: str, row: pd.Series, replay_hours: int) -> dict:
    from crypto_trading_system_ed import generate_signals, _suppress_stderr
    feats = None
    if 'optimal_features' in row and pd.notna(row['optimal_features']):
        feats = [f.strip() for f in str(row['optimal_features']).split(',') if f.strip()]
    gamma = float(row['gamma']) if pd.notna(row.get('gamma', None)) else 1.0
    models = str(row['models']).split('+')
    window = int(row['best_window'])
    horizon = int(row['horizon'])
    print(f"    {asset} {horizon}h | models={row['models']} | w={window}h | gamma={gamma}")
    with _suppress_stderr():
        sigs = generate_signals(asset, models, window, replay_hours,
                                feature_override=feats, horizon=horizon, gamma=gamma)
    # index by pandas Timestamp (UTC-naive, matching raw data)
    out = {}
    for s in sigs:
        dt = s['datetime']
        if isinstance(dt, str):
            dt = datetime.strptime(dt, '%Y-%m-%d %H:%M')
        out[pd.Timestamp(dt)] = s
    return out


def merge_signals_via_regime(asset: str,
                             bull_sigs: dict, bull_conf_thr: float, bull_h: int,
                             bear_sigs: dict, bear_conf_thr: float, bear_h: int):
    """Combine 6h (bull) and 8h (bear) signals per-hour using tsmom_672h.

    Returns list of dicts with datetime (Timestamp), close, signal, confidence,
    conf_threshold, regime.
    """
    from crypto_trading_system_ed import _build_regime_indicators_and_detectors
    ind, detectors = _build_regime_indicators_and_detectors(asset)
    detector = detectors['tsmom_672h']

    all_dts = sorted(set(bull_sigs.keys()) | set(bear_sigs.keys()))
    merged = []
    for dt in all_dts:
        is_bull = bool(detector(dt))
        regime = 'bull' if is_bull else 'bear'
        src = bull_sigs if is_bull else bear_sigs
        thr = bull_conf_thr if is_bull else bear_conf_thr
        s = src.get(dt)
        if s is None:
            # no signal for active regime horizon at this timestamp
            # fall back to other horizon's close (price is shared anyway) but mark HOLD
            other = (bear_sigs if is_bull else bull_sigs).get(dt)
            if other is None:
                continue
            merged.append({
                'datetime': dt, 'close': float(other['close']),
                'signal': 'HOLD', 'confidence': 0.0,
                'conf_threshold': thr, 'regime': regime,
                'horizon': bull_h if is_bull else bear_h,
            })
            continue
        merged.append({
            'datetime': dt,
            'close': float(s['close']),
            'signal': s['signal'],
            'confidence': float(s['confidence']),
            'conf_threshold': thr,
            'regime': regime,
            'horizon': bull_h if is_bull else bear_h,
        })
    return merged


# ----------------------------------------------------------------
# Step 4: Simulate strategies
# ----------------------------------------------------------------
def _five_min_slice(df5: pd.DataFrame, hour_start: pd.Timestamp) -> pd.DataFrame:
    """5m candles falling strictly INSIDE [hour_start, hour_start+1h).

    df5 index is UTC timezone-aware. hour_start may be tz-naive; assume UTC.
    """
    if hour_start.tzinfo is None:
        hour_start = hour_start.tz_localize('UTC')
    hour_end = hour_start + pd.Timedelta(hours=1)
    return df5[(df5.index >= hour_start) & (df5.index < hour_end)]


def simulate(signals: list, df5: pd.DataFrame, strategy: str) -> dict:
    """Simulate 'baseline' or 'intrahour' on merged hourly signals.

    Returns dict of per-trade log + summary stats. Entries execute at the
    NEXT hour's open price (matches live trader behavior).
    """
    assert strategy in ('baseline', 'intrahour')
    df5 = df5.set_index('datetime') if df5.index.name != 'datetime' else df5
    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry_px = 0.0
    entry_time = None
    hold_hours = 0
    trades = []
    intrahour_trigs = 0
    equity_curve = [1000.0]

    n = len(signals)
    for i, s in enumerate(signals):
        dt = s['datetime']
        price = s['close']
        sig = s['signal']
        conf = s['confidence']
        thr = s['conf_threshold']

        # record equity at top-of-hour (mark-to-close)
        equity = cash + qty * price if in_pos else cash
        equity_curve.append(equity)

        # ----- INTRA-HOUR EXIT (strategy=intrahour) -----
        if strategy == 'intrahour' and in_pos:
            # scan 5m candles inside THIS hour (after entry)
            bars = _five_min_slice(df5, dt)
            if entry_time is not None and entry_time == dt:
                # entry just happened at THIS hour's open; scan bars inside this hour
                scan_bars = bars
            elif entry_time is not None and dt > entry_time:
                scan_bars = bars
            else:
                scan_bars = bars
            target = entry_px * (1.0 + PROFIT_TARGET_PCT)
            hit_row = None
            for ts, row in scan_bars.iterrows():
                if row['high'] >= target:
                    hit_row = (ts, row)
                    break
            if hit_row is not None:
                ts, _row = hit_row
                exit_px = target         # limit order assumption
                cash = qty * exit_px * (1 - TRADING_FEE)
                pnl_pct = (exit_px / entry_px - 1.0) * 100.0
                trades.append({
                    'strategy': strategy,
                    'entry_time': entry_time, 'entry_price': entry_px,
                    'exit_time': ts, 'exit_price': exit_px,
                    'pnl_pct': pnl_pct, 'hold_hours': hold_hours,
                    'exit_reason': 'intrahour_target',
                })
                intrahour_trigs += 1
                in_pos = False
                qty = 0.0
                entry_px = 0.0
                hold_hours = 0
                entry_time = None
                # fall through: still process this hour's signal for potential re-entry

        # ----- HOURLY DECISION -----
        # BUY: emit signal at hour i, fill at open of hour i+1
        if sig == 'BUY' and conf >= thr and not in_pos:
            if i + 1 < n:
                next_sig = signals[i + 1]
                fill_dt = next_sig['datetime']
                fill_px = next_sig['close']
            else:
                fill_dt = dt
                fill_px = price
            qty = cash * (1 - TRADING_FEE) / fill_px
            cash = 0.0
            in_pos = True
            entry_px = fill_px
            entry_time = fill_dt
            hold_hours = 0

        elif sig == 'SELL' and in_pos:
            # hourly sell: fill at NEXT hour's open (matches prod)
            if i + 1 < n:
                next_sig = signals[i + 1]
                fill_dt = next_sig['datetime']
                fill_px = next_sig['close']
            else:
                fill_dt = dt
                fill_px = price
            cur_pnl_pct = (fill_px / entry_px - 1.0) * 100.0
            shield_ok = cur_pnl_pct >= (MIN_SELL_PNL_PCT * 100.0)
            failsafe  = hold_hours >= MAX_HOLD_HOURS
            if shield_ok or failsafe:
                cash = qty * fill_px * (1 - TRADING_FEE)
                trades.append({
                    'strategy': strategy,
                    'entry_time': entry_time, 'entry_price': entry_px,
                    'exit_time': fill_dt,    'exit_price': fill_px,
                    'pnl_pct': cur_pnl_pct,  'hold_hours': hold_hours + 1,
                    'exit_reason': 'failsafe' if (failsafe and not shield_ok) else 'hourly_shield_ok',
                })
                in_pos = False
                qty = 0.0
                entry_px = 0.0
                entry_time = None
                hold_hours = 0

        # advance hold counter at end of hour
        if in_pos:
            hold_hours += 1

    # close out open position at last available price
    if in_pos:
        final_px = signals[-1]['close']
        cash = qty * final_px * (1 - TRADING_FEE)
        pnl_pct = (final_px / entry_px - 1.0) * 100.0
        trades.append({
            'strategy': strategy,
            'entry_time': entry_time, 'entry_price': entry_px,
            'exit_time': signals[-1]['datetime'], 'exit_price': final_px,
            'pnl_pct': pnl_pct, 'hold_hours': hold_hours,
            'exit_reason': 'end_of_window',
        })
        in_pos = False
        qty = 0.0

    # stats
    tot_pnl_pct = (cash / 1000.0 - 1.0) * 100.0
    wins = sum(1 for t in trades if t['pnl_pct'] > 0)
    avg_hold = (sum(t['hold_hours'] for t in trades) / max(len(trades), 1))
    # max drawdown on equity curve
    ec = np.array(equity_curve)
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / peak
    max_dd = float(dd.min()) * 100.0 if len(dd) else 0.0

    return {
        'strategy': strategy,
        'trades': trades,
        'n_trades': len(trades),
        'wins': wins,
        'total_pnl_pct': tot_pnl_pct,
        'avg_hold': avg_hold,
        'max_drawdown_pct': max_dd,
        'intrahour_triggers': intrahour_trigs,
        'final_cash': cash,
    }


# ----------------------------------------------------------------
# Step 5: report
# ----------------------------------------------------------------
def report(base: dict, intra: dict):
    print()
    print("=" * 80)
    print(f"  ETH intra-hour profit-take backtest  ({DAYS}d, target +{PROFIT_TARGET_PCT*100:.2f}%)")
    print("=" * 80)
    hdr = f"  {'Metric':<22}{'Baseline':>12}{'Intra-hour':>14}{'Delta':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    def row(label, b, i, fmt='{:.2f}', dfmt=None):
        d = i - b
        dfmt = dfmt or fmt
        sign = '+' if d >= 0 else ''
        print(f"  {label:<22}{fmt.format(b):>12}{fmt.format(i):>14}{(sign + dfmt.format(d)):>12}")
    row('Total trades',   base['n_trades'],        intra['n_trades'],        '{:d}')
    row('Winning trades', base['wins'],            intra['wins'],            '{:d}')
    row('Total PnL %',    base['total_pnl_pct'],   intra['total_pnl_pct'])
    row('Avg hold (h)',   base['avg_hold'],        intra['avg_hold'])
    row('Max drawdown %', base['max_drawdown_pct'], intra['max_drawdown_pct'])
    print(f"  {'Intra-hour triggers':<22}{'-':>12}{intra['intrahour_triggers']:>14}")

    # "missed-by-hourly" = intrahour exits where baseline's matching trade
    # (same entry_time) exited at a LOWER price (or later in time).
    base_by_entry = {t['entry_time']: t for t in base['trades']}
    missed = 0
    for t in intra['trades']:
        if t['exit_reason'] != 'intrahour_target':
            continue
        bt = base_by_entry.get(t['entry_time'])
        if bt is None:
            missed += 1
            continue
        if bt['exit_price'] < t['exit_price'] or bt['exit_time'] > t['exit_time']:
            missed += 1
    print(f"  {'Missed-by-hourly':<22}{'-':>12}{missed:>14}")

    print()
    print("=" * 80)
    print("  TRADE LOG")
    print("=" * 80)
    fmt_row = "  {:<10} {:<20} {:>10} {:<20} {:>10} {:>8} {:>5}  {}"
    print(fmt_row.format('strategy', 'entry_time', 'entry', 'exit_time', 'exit', 'pnl%', 'hold', 'reason'))
    print("  " + "-" * 100)
    for t in (base['trades'] + intra['trades']):
        et = str(t['entry_time']).replace('+00:00', '')[:19]
        xt = str(t['exit_time']).replace('+00:00', '')[:19]
        print(fmt_row.format(
            t['strategy'], et, f"{t['entry_price']:.2f}",
            xt, f"{t['exit_price']:.2f}",
            f"{t['pnl_pct']:+.2f}%", t['hold_hours'], t['exit_reason']))


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    print("=" * 80)
    print(f"  Intra-hour profit-take backtest  ({ASSET}, {DAYS}d)")
    print("=" * 80)

    # Step 1 — 5m candles
    df5 = download_5m_candles(DAYS)
    # normalize index for later
    df5['datetime'] = pd.to_datetime(df5['datetime'], utc=True)

    # Step 2 — regime config
    with open(REGIME_CFG_PATH) as f:
        regime_cfg = json.load(f)
    asset_cfg = regime_cfg[ASSET]
    bull_h   = int(asset_cfg['bull']['horizon'])
    bear_h   = int(asset_cfg['bear']['horizon'])
    bull_thr = float(asset_cfg['bull']['min_confidence'])
    bear_thr = float(asset_cfg['bear']['min_confidence'])
    print(f"  regime: tsmom_672h | bull={bull_h}h@{bull_thr}% | bear={bear_h}h@{bear_thr}%")

    # Step 3 — model metadata (doohan v1.6 for 8h; standard prod for 6h)
    doohan = pd.read_csv(DOOHAN_CSV)
    standard = pd.read_csv(STANDARD_PROD_CSV)

    bull_row_source = doohan if bull_h in doohan[doohan['coin']==ASSET]['horizon'].values else standard
    bear_row_source = doohan if bear_h in doohan[doohan['coin']==ASSET]['horizon'].values else standard
    bull_row = pick_model_row(bull_row_source, ASSET, bull_h)
    bear_row = pick_model_row(bear_row_source, ASSET, bear_h)
    print(f"  bull model source = "
          f"{'doohan_v1_6' if bull_row_source is doohan else 'crypto_ed_production'}")
    print(f"  bear model source = "
          f"{'doohan_v1_6' if bear_row_source is doohan else 'crypto_ed_production'}")

    # Step 4 — replay signals for both horizons
    print()
    print(f"  Generating signals (replay={REPLAY_HOURS}h)...")
    bull_sigs = generate_horizon_signals(ASSET, bull_row, REPLAY_HOURS)
    bear_sigs = generate_horizon_signals(ASSET, bear_row, REPLAY_HOURS)

    # Step 5 — merge via regime
    merged = merge_signals_via_regime(ASSET, bull_sigs, bull_thr, bull_h,
                                      bear_sigs, bear_thr, bear_h)
    print(f"  merged: {len(merged)} hourly signals "
          f"({sum(1 for m in merged if m['regime']=='bull')} bull / "
          f"{sum(1 for m in merged if m['regime']=='bear')} bear)")

    # Trim to strict last 30 days (safety — generate_signals may include extras)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(days=DAYS)
    merged = [m for m in merged if m['datetime'] >= pd.Timestamp(cutoff)]
    print(f"  after 30d window trim: {len(merged)} hourly signals")
    if not merged:
        print("  ERROR: no signals in window — aborting")
        return

    # Step 6 — prepare 5m data (indexed, UTC-aware)
    df5 = df5.set_index('datetime').sort_index()

    # Convert merged 'datetime' from tz-naive -> tz-aware UTC for 5m slicing
    for m in merged:
        if m['datetime'].tzinfo is None:
            m['datetime'] = m['datetime'].tz_localize('UTC')

    # Step 7 — simulate both strategies
    base = simulate(merged, df5.copy(), 'baseline')
    intra = simulate(merged, df5.copy(), 'intrahour')

    # Step 8 — report
    report(base, intra)


if __name__ == '__main__':
    main()
