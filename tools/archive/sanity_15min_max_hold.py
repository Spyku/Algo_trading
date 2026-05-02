"""
sanity_15min_max_hold.py — Standalone sanity check for the question:
"Could 15-min candle data have improved max_hold decisions?"

For each MAX_HOLD-failsafe trade in the 60d backtest, pull 15-min ETH data
covering the trade window + the 2h LEADING UP to the failsafe exit. Compute:
  - last 8 x 15-min log returns (the 2h before exit)
  - trend score: count of negative returns in that window
  - max drawdown within the last 2h
  - simple classification: MONOTONIC_DOWN / MONOTONIC_UP / CHOPPY

Output:
  - table per MAX_HOLD trade
  - aggregate count of "predictable" trades (where 15-min trend would have
    cleanly told us 'cut now' or 'extend')

Touches:
  - reads data/eth_15m_data.csv  (read-only)
  - reads data/eth_sl_signals_90d.pkl  (read-only)
  - reads config/regime_config_ed.json  (read-only)
  - writes output/sanity_15min_max_hold_<timestamp>.csv

Does NOT touch any production file.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PKL = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
REGIME_CFG = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
DATA_15M = os.path.join(ENGINE, 'data', 'eth_15m_data.csv')
FEE = 0.0005


def load_sigs():
    with open(CACHE_PKL, 'rb') as f:
        sigs = pickle.load(f)
    for s in sigs:
        s['datetime'] = pd.Timestamp(s['datetime'])
    sigs.sort(key=lambda s: s['datetime'])
    return sigs


def window_slice(sigs, days):
    end = sigs[-1]['datetime']
    lo = end - pd.Timedelta(days=days)
    return [s for s in sigs if s['datetime'] >= lo]


def get_rally_tuple(asset_cfg, regime):
    block = asset_cfg.get(regime, {})
    rc = block.get('rally_cooldown') if isinstance(block, dict) else None
    rc = rc or asset_cfg.get('rally_cooldown')
    if not rc or not rc.get('enabled'):
        return None
    try:
        return (int(rc['h_short']), int(rc['h_long']),
                float(rc['t_short_pct']), float(rc['t_long_pct']),
                int(rc['cd_hours']))
    except (KeyError, TypeError, ValueError):
        return None


def find_max_hold_trades(sigs, asset_cfg):
    """Re-run the regime-switched simulator with the live config to extract
    the exact MAX_HOLD-tagged trades. Same logic as test_per_regime_max_hold.py
    baseline, just collects per-trade detail with exit_reason classification."""
    bull_conf = float(asset_cfg.get('bull', {}).get('min_confidence', 80))
    bear_conf = float(asset_cfg.get('bear', {}).get('min_confidence', 65))
    bull_shield = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    min_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.5))
    max_hold_h = int(asset_cfg.get('max_hold_hours', 10))

    bull_rally = get_rally_tuple(asset_cfg, 'bull')
    bear_rally = get_rally_tuple(asset_cfg, 'bear')

    closes = np.array([float(s['close']) for s in sigs])

    def _rr(h):
        out = np.full(len(closes), np.nan)
        if h and h < len(closes):
            out[h:] = (closes[h:] / closes[:-h] - 1.0) * 100.0
        return out

    bh_s = bh_l = 0
    bt_s = bt_l = 0.0
    bcd_h = 0
    bull_rs = bull_rl = None
    if bull_rally is not None:
        bh_s, bh_l, bt_s, bt_l, bcd_h = bull_rally
        bull_rs = _rr(bh_s)
        bull_rl = _rr(bh_l)
    rh_s = rh_l = 0
    rt_s = rt_l = 0.0
    rcd_h = 0
    bear_rs = bear_rl = None
    if bear_rally is not None:
        rh_s, rh_l, rt_s, rt_l, rcd_h = bear_rally
        bear_rs = _rr(rh_s)
        bear_rl = _rr(rh_l)

    cash = 1000.0
    held = 0.0
    in_pos = False
    entry_px = 0.0
    entry_dt = None
    entry_regime = None
    hold_since_entry = 0
    bull_cd = bear_cd = 0
    trade_log = []

    for i, s in enumerate(sigs):
        price = float(s['close'])
        regime = s.get('regime', 'bull')
        sig = s['signal']
        sconf = float(s.get('confidence', 0))
        conf_thr = bull_conf if regime == 'bull' else bear_conf
        shield_on = bull_shield if regime == 'bull' else bear_shield

        if in_pos:
            hold_since_entry += 1

        if bcd_h > 0 and bull_rs is not None:
            rs = bull_rs[i] if not np.isnan(bull_rs[i]) else 0
            rl = bull_rl[i] if not np.isnan(bull_rl[i]) else 0
            if rs >= bt_s or rl >= bt_l:
                bull_cd = max(bull_cd, bcd_h)
        if rcd_h > 0 and bear_rs is not None:
            rs = bear_rs[i] if not np.isnan(bear_rs[i]) else 0
            rl = bear_rl[i] if not np.isnan(bear_rl[i]) else 0
            if rs >= rt_s or rl >= rt_l:
                bear_cd = max(bear_cd, rcd_h)
        active_cd = bull_cd if regime == 'bull' else bear_cd

        if sig == 'BUY' and sconf >= conf_thr and not in_pos and active_cd <= 0:
            held = cash * (1 - FEE) / price
            cash = 0
            in_pos = True
            entry_px = price
            entry_dt = s['datetime']
            entry_regime = regime
            hold_since_entry = 0
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = shield_on and min_pnl > 0 and cur_pnl < min_pnl
            if not shield_blocks or override_expired:
                exit_reason = 'MAX_HOLD' if (shield_on and override_expired and cur_pnl < min_pnl) else 'MODEL'
                cash = held * price * (1 - FEE)
                trade_log.append({
                    'entry_dt': entry_dt, 'exit_dt': s['datetime'],
                    'entry_px': entry_px, 'exit_px': price,
                    'pnl_pct': cur_pnl, 'hold_h': hold_since_entry,
                    'regime': entry_regime, 'exit_reason': exit_reason,
                })
                held = 0
                in_pos = False
                hold_since_entry = 0

        if bull_cd > 0:
            bull_cd -= 1
        if bear_cd > 0:
            bear_cd -= 1

    return [t for t in trade_log if t['exit_reason'] == 'MAX_HOLD']


def load_15min_data():
    df = pd.read_csv(DATA_15M)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()
    return df


def analyze_trade(trade, df_15m, lookback_hours=2):
    """For a single MAX_HOLD trade, look at the 15-min candles in the
    `lookback_hours` window leading up to the exit time. Return classification."""
    exit_dt = trade['exit_dt']
    if exit_dt.tzinfo is None:
        exit_dt = exit_dt.tz_localize('UTC')
    else:
        exit_dt = exit_dt.tz_convert('UTC')
    lookback_start = exit_dt - timedelta(hours=lookback_hours)

    window = df_15m[(df_15m.index >= lookback_start) & (df_15m.index <= exit_dt)]
    if len(window) < 4:
        return None  # insufficient data
    closes = window['close'].values
    # Log returns of 15-min candles
    logrets = np.diff(np.log(closes))  # n-1 returns
    n_neg = int((logrets < 0).sum())
    n_pos = int((logrets > 0).sum())
    n_total = len(logrets)
    cum_15m_ret = 100 * (closes[-1] / closes[0] - 1)
    drawdown = 100 * (closes.min() / closes[0] - 1)  # max DD within window
    runup = 100 * (closes.max() / closes[0] - 1)

    # Classification:
    # MONOTONIC_DOWN: ≥ 75% of 15-min returns are negative AND cum return < -0.3%
    # MONOTONIC_UP:   ≥ 75% of returns positive AND cum return > +0.3%
    # CHOPPY:         everything else
    pct_neg = n_neg / n_total if n_total else 0
    pct_pos = n_pos / n_total if n_total else 0
    if pct_neg >= 0.65 and cum_15m_ret < -0.2:
        cls = 'MONO_DOWN'
    elif pct_pos >= 0.65 and cum_15m_ret > 0.2:
        cls = 'MONO_UP'
    else:
        cls = 'CHOPPY'

    return {
        'lookback_h': lookback_hours,
        'n_15m_candles': n_total,
        'pct_neg': round(100 * pct_neg, 0),
        'pct_pos': round(100 * pct_pos, 0),
        'cum_15m_ret': round(cum_15m_ret, 3),
        'window_runup': round(runup, 3),
        'window_dd': round(drawdown, 3),
        'classification': cls,
    }


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    sigs = load_sigs()
    win = window_slice(sigs, 60)
    print(f"60d window: {len(win)} signals, "
          f"{win[0]['datetime']} -> {win[-1]['datetime']}\n")

    mh_trades = find_max_hold_trades(win, asset_cfg)
    print(f"Found {len(mh_trades)} MAX_HOLD-failsafe trades in the 60d window.\n")

    df_15m = load_15min_data()
    print(f"15-min data: {len(df_15m)} rows, "
          f"{df_15m.index.min()} -> {df_15m.index.max()}\n")

    rows = []
    for i, t in enumerate(mh_trades, 1):
        # Two windows: last 2h and last 1h
        a2 = analyze_trade(t, df_15m, lookback_hours=2)
        a1 = analyze_trade(t, df_15m, lookback_hours=1)
        if a2 is None:
            continue
        rows.append({
            'trade': i,
            'entry_dt': t['entry_dt'].strftime('%m-%d %H:%M'),
            'exit_dt': t['exit_dt'].strftime('%m-%d %H:%M'),
            'hold_h': t['hold_h'],
            'regime': t['regime'][:4],
            'pnl_pct': round(t['pnl_pct'], 2),
            # Last 2h
            'L2h_neg%': a2['pct_neg'],
            'L2h_cum%': a2['cum_15m_ret'],
            'L2h_class': a2['classification'],
            # Last 1h
            'L1h_neg%': a1['pct_neg'] if a1 else None,
            'L1h_cum%': a1['cum_15m_ret'] if a1 else None,
            'L1h_class': a1['classification'] if a1 else None,
        })

    df = pd.DataFrame(rows)
    print("=" * 130)
    print(f"  PER-TRADE 15-MIN ANALYSIS — {len(df)} MAX_HOLD trades, last 2h before exit")
    print("=" * 130)
    print(df.to_string(index=False))

    print(f"\n{'='*130}")
    print(f"  AGGREGATE PATTERNS")
    print(f"{'='*130}")

    print(f"\n[Last 2h before MAX_HOLD exit]")
    print(df['L2h_class'].value_counts().to_string())
    print(f"\n[Last 1h before MAX_HOLD exit]")
    print(df['L1h_class'].value_counts().to_string())

    # If trend was MONO_DOWN, exiting earlier (at hour 8 instead of 10) would have saved %
    print(f"\n[Counterfactual: if trend filter detected MONO_DOWN at L1h, exit would have been earlier]")
    print(f"  (Exit at hour-2: avg PnL recovery = avoiding the L2h cumulative drop)")
    if 'L2h_class' in df.columns:
        mono_down = df[df['L2h_class'] == 'MONO_DOWN']
        choppy = df[df['L2h_class'] == 'CHOPPY']
        mono_up = df[df['L2h_class'] == 'MONO_UP']
        print(f"  Trades where 2h trend was MONO_DOWN: {len(mono_down)}/{len(df)}")
        if len(mono_down):
            avg_drop = mono_down['L2h_cum%'].mean()
            avg_pnl = mono_down['pnl_pct'].mean()
            print(f"    Avg 2h pre-exit drop on these: {avg_drop:+.2f}%")
            print(f"    Avg realized PnL on these: {avg_pnl:+.2f}%")
            print(f"    Theoretical save if exited 2h earlier: {-avg_drop:+.2f}pp per trade")
            print(f"    Total saved across these {len(mono_down)} trades: "
                  f"{-mono_down['L2h_cum%'].sum():+.2f}pp")
        print(f"  Trades where 2h trend was MONO_UP (extending would have been right): "
              f"{len(mono_up)}/{len(df)}")
        if len(mono_up):
            print(f"    Avg 2h pre-exit RUNUP on these: {mono_up['L2h_cum%'].mean():+.2f}%")
            print(f"    Avg realized PnL: {mono_up['pnl_pct'].mean():+.2f}%")
        print(f"  Trades where 2h was CHOPPY (15-min filter wouldn't help): "
              f"{len(choppy)}/{len(df)}")

    out = os.path.join(ENGINE, 'output',
                       f'sanity_15min_max_hold_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nFull table saved to: {out}")


if __name__ == '__main__':
    main()
