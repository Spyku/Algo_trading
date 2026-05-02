"""
test_t2c_vs_gates.py — does the T2c drawdown-from-7d-high filter REPLACE
the existing rally-cooldown gates, or COMPLEMENT them?

Current production gates:
  bull: rr20>=4.0% OR rr24>=4.5%, cd=12h
  bear: rr30>=9.0% OR rr36>=9.0%, cd=48h

Both block BUYs for N hours after a rally trigger fires. T2c instead
checks the CURRENT drawdown from 7d high — a continuous condition rather
than a timeout-based block.

Tested variants per regime:
  - GATE ON, T2C OFF (current production)
  - GATE OFF, T2C ON (replacement)
  - GATE ON, T2C ON (combined)
  - GATE OFF, T2C OFF (no entry filter at all)

Also: T2c can be applied per-regime (only block bull buys, only block bear,
or both). All combinations swept.

Read-only on production files.
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timedelta
from itertools import product
import copy

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PKL = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
REGIME_CFG = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
ETH_HOURLY = os.path.join(ENGINE, 'data', 'eth_hourly_data.csv')
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


def precompute_dd_from_7d_high(sigs, eth_df):
    n = len(sigs)
    dd = np.full(n, np.nan)
    eth_high = eth_df['high']
    for i, s in enumerate(sigs):
        dt = s['datetime']
        if dt.tzinfo is None: dt = dt.tz_localize('UTC')
        else: dt = dt.tz_convert('UTC')
        sub7d = eth_high[(eth_high.index < dt) & (eth_high.index >= dt - timedelta(days=7))]
        if len(sub7d) > 24:
            cur_close = float(s['close'])
            high_7d = sub7d.max()
            dd[i] = 100 * (cur_close / high_7d - 1)
    return dd


def sim(sigs, asset_cfg, dd_arr,
        bull_gate_on=True, bear_gate_on=True,
        bull_dd_min=None, bear_dd_min=None,
        bull_conf=None, bear_conf=None):
    """Simulator with toggleable gates and per-regime dd filter.
    bull/bear_dd_min: if set, BUY in that regime requires dd from 7d high >=
                     this threshold (e.g. 3.0 means must be 3% below high).
    bull/bear_conf: optional override for confidence threshold.
    """
    bull_conf_thr = bull_conf if bull_conf is not None else float(asset_cfg.get('bull', {}).get('min_confidence', 80))
    bear_conf_thr = bear_conf if bear_conf is not None else float(asset_cfg.get('bear', {}).get('min_confidence', 65))
    bull_shield = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    min_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.5))
    max_hold_h = int(asset_cfg.get('max_hold_hours', 10))

    bull_rally = get_rally_tuple(asset_cfg, 'bull') if bull_gate_on else None
    bear_rally = get_rally_tuple(asset_cfg, 'bear') if bear_gate_on else None

    closes = np.array([float(s['close']) for s in sigs])

    def _rr(h):
        out = np.full(len(closes), np.nan)
        if h and h < len(closes):
            out[h:] = (closes[h:] / closes[:-h] - 1.0) * 100.0
        return out

    bh_s = bh_l = 0; bt_s = bt_l = 0.0; bcd_h = 0
    bull_rs = bull_rl = None
    if bull_rally is not None:
        bh_s, bh_l, bt_s, bt_l, bcd_h = bull_rally
        bull_rs = _rr(bh_s); bull_rl = _rr(bh_l)
    rh_s = rh_l = 0; rt_s = rt_l = 0.0; rcd_h = 0
    bear_rs = bear_rl = None
    if bear_rally is not None:
        rh_s, rh_l, rt_s, rt_l, rcd_h = bear_rally
        bear_rs = _rr(rh_s); bear_rl = _rr(rh_l)

    cash = 1000.0; held = 0.0; in_pos = False; entry_px = 0.0
    hold_since_entry = 0; bull_cd = bear_cd = 0
    trade_log = []

    for i, s in enumerate(sigs):
        price = float(s['close'])
        regime = s.get('regime', 'bull')
        sig = s['signal']
        sconf = float(s.get('confidence', 0))
        conf_thr = bull_conf_thr if regime == 'bull' else bear_conf_thr
        shield_on = bull_shield if regime == 'bull' else bear_shield

        if in_pos:
            hold_since_entry += 1

        if bcd_h > 0 and bull_rs is not None:
            rs = bull_rs[i] if not np.isnan(bull_rs[i]) else 0
            rl = bull_rl[i] if not np.isnan(bull_rl[i]) else 0
            if rs >= bt_s or rl >= bt_l: bull_cd = max(bull_cd, bcd_h)
        if rcd_h > 0 and bear_rs is not None:
            rs = bear_rs[i] if not np.isnan(bear_rs[i]) else 0
            rl = bear_rl[i] if not np.isnan(bear_rl[i]) else 0
            if rs >= rt_s or rl >= rt_l: bear_cd = max(bear_cd, rcd_h)
        active_cd = bull_cd if regime == 'bull' else bear_cd

        if sig == 'BUY' and sconf >= conf_thr and not in_pos and active_cd <= 0:
            # Apply per-regime dd filter
            block = False
            dd = dd_arr[i]
            if regime == 'bull' and bull_dd_min is not None:
                if not np.isnan(dd) and dd > -abs(bull_dd_min):
                    block = True
            if regime == 'bear' and bear_dd_min is not None:
                if not np.isnan(dd) and dd > -abs(bear_dd_min):
                    block = True
            if not block:
                held = cash * (1 - FEE) / price
                cash = 0; in_pos = True; entry_px = price
                hold_since_entry = 0
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = shield_on and min_pnl > 0 and cur_pnl < min_pnl
            if not shield_blocks or override_expired:
                cash = held * price * (1 - FEE)
                trade_log.append({'pnl_pct': cur_pnl, 'regime': regime})
                held = 0; in_pos = False; hold_since_entry = 0

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        cash = held * sigs[-1]['close'] * (1 - FEE)
        trade_log.append({'pnl_pct': (sigs[-1]['close'] / entry_px - 1) * 100,
                          'regime': sigs[-1].get('regime', 'bull')})

    ret = (cash / 1000.0 - 1) * 100
    n = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    wr = (wins / n * 100) if n else 0
    n_bull = sum(1 for t in trade_log if t['regime'] == 'bull')
    n_bear = n - n_bull
    return ret, n, wr, n_bull, n_bear


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    sigs = load_sigs()
    eth = pd.read_csv(ETH_HOURLY)
    eth['datetime'] = pd.to_datetime(eth['datetime'], utc=True)
    eth = eth.set_index('datetime').sort_index()

    windows = {'30d': window_slice(sigs, 30), '60d': window_slice(sigs, 60),
               '90d': sigs}

    print(f"{'window':<6} {'config':<55} {'return':>8} {'delta':>8} {'tr':>4} {'WR':>4} {'B':>3} {'b':>3}")
    print('-' * 110)

    for w_name, w_sigs in windows.items():
        dd_arr = precompute_dd_from_7d_high(w_sigs, eth)
        # Baseline = current production (both gates ON, no dd, default conf)
        base, base_n, base_wr, base_b, base_br = sim(
            w_sigs, asset_cfg, dd_arr, bull_gate_on=True, bear_gate_on=True)
        print(f"{w_name:<6} {'BASELINE (prod: bull_gate ON, bear_gate ON)':<55} "
              f"{base:>+7.2f}% {0:>+7.2f}pp {base_n:>4} {base_wr:>3.0f}% {base_b:>3} {base_br:>3}")

        # Test grid: each gate ON/OFF × per-regime dd ON/OFF
        tests = [
            ('REPLACE bull gate with T2c bull_dd=3%',
             dict(bull_gate_on=False, bear_gate_on=True, bull_dd_min=3.0, bear_dd_min=None)),
            ('REPLACE bear gate with T2c bear_dd=3%',
             dict(bull_gate_on=True, bear_gate_on=False, bull_dd_min=None, bear_dd_min=3.0)),
            ('REPLACE BOTH gates with T2c on both regimes',
             dict(bull_gate_on=False, bear_gate_on=False, bull_dd_min=3.0, bear_dd_min=3.0)),
            ('Drop bull gate, no T2c (raw bull)',
             dict(bull_gate_on=False, bear_gate_on=True)),
            ('Drop bear gate, no T2c (raw bear)',
             dict(bull_gate_on=True, bear_gate_on=False)),
            ('Drop both gates, no T2c (no filtering)',
             dict(bull_gate_on=False, bear_gate_on=False)),
            ('KEEP gates, ADD bull_dd=3% on top',
             dict(bull_gate_on=True, bear_gate_on=True, bull_dd_min=3.0, bear_dd_min=None)),
            ('KEEP gates, ADD bear_dd=3% on top',
             dict(bull_gate_on=True, bear_gate_on=True, bull_dd_min=None, bear_dd_min=3.0)),
            ('KEEP gates, ADD bull+bear dd=3% (T2 combo)',
             dict(bull_gate_on=True, bear_gate_on=True, bull_dd_min=3.0, bear_dd_min=3.0)),
            ('REPLACE bull gate with T2c bull_dd=3% + bull_conf=90',
             dict(bull_gate_on=False, bear_gate_on=True, bull_dd_min=3.0,
                  bear_dd_min=None, bull_conf=90)),
        ]

        for name, kw in tests:
            ret, n, wr, b, br = sim(w_sigs, asset_cfg, dd_arr, **kw)
            d = ret - base
            print(f"{w_name:<6} {name:<55} {ret:>+7.2f}% {d:>+7.2f}pp {n:>4} {wr:>3.0f}% {b:>3} {br:>3}")
        print('-' * 110)


if __name__ == '__main__':
    main()
