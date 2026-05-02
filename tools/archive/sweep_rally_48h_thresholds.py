"""
sweep_rally_48h_thresholds.py — focused sweep of rally_48h thresholds
{3.0, 3.5, 4.0, 4.5, 5.0}% on 30d + 60d. PROD as-is + only this gate added.
No other changes (shields ON, gates ON, conf unchanged).
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timedelta

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
        if s['datetime'].tzinfo is None:
            s['datetime'] = s['datetime'].tz_localize('UTC')
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


def precompute_rally_48h(sigs, eth_df):
    n = len(sigs)
    closes = eth_df['close']
    out = np.full(n, np.nan)
    for i, s in enumerate(sigs):
        dt = s['datetime']
        if dt.tzinfo is None: dt = dt.tz_localize('UTC')
        else: dt = dt.tz_convert('UTC')
        cur_idx = eth_df.index[eth_df.index <= dt]
        if len(cur_idx) == 0: continue
        cur_close = closes.loc[cur_idx[-1]]
        prior_dt = dt - pd.Timedelta(hours=48)
        prior_idx = eth_df.index[eth_df.index <= prior_dt]
        if len(prior_idx) > 0:
            pp = closes.loc[prior_idx[-1]]
            if pp > 0:
                out[i] = (cur_close / pp - 1) * 100
    return out


def sim(sigs, asset_cfg, rally_arr, block_thr=None):
    bull_conf_thr = float(asset_cfg.get('bull', {}).get('min_confidence', 80))
    bear_conf_thr = float(asset_cfg.get('bear', {}).get('min_confidence', 65))
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
    blocked = 0
    max_dd_pct = 0.0
    peak_eq = cash

    for i, s in enumerate(sigs):
        price = float(s['close'])
        regime = s.get('regime', 'bull')
        sig = s['signal']
        sconf = float(s.get('confidence', 0))
        conf_thr = bull_conf_thr if regime == 'bull' else bear_conf_thr
        shield_on = bull_shield if regime == 'bull' else bear_shield

        if in_pos: hold_since_entry += 1
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
            block = False
            if block_thr is not None:
                v = rally_arr[i]
                if not np.isnan(v) and v >= block_thr:
                    block = True
            if block:
                blocked += 1
            else:
                held = cash * (1 - FEE) / price
                cash = 0; in_pos = True; entry_px = price; hold_since_entry = 0
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = shield_on and min_pnl > 0 and cur_pnl < min_pnl
            if not shield_blocks or override_expired:
                cash = held * price * (1 - FEE)
                trade_log.append({'pnl_pct': cur_pnl, 'regime': regime})
                held = 0; in_pos = False; hold_since_entry = 0

        cur_eq = cash if not in_pos else held * price
        peak_eq = max(peak_eq, cur_eq)
        dd = (cur_eq / peak_eq - 1) * 100
        if dd < max_dd_pct: max_dd_pct = dd

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        cash = held * sigs[-1]['close'] * (1 - FEE)
        trade_log.append({'pnl_pct': (sigs[-1]['close'] / entry_px - 1) * 100, 'regime': 'bull'})

    ret = (cash / 1000.0 - 1) * 100
    n = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    wr = (wins / n * 100) if n else 0
    avg = float(np.mean([t['pnl_pct'] for t in trade_log])) if trade_log else 0.0
    return ret, n, wr, blocked, max_dd_pct, avg


def main():
    print("Loading...")
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']
    sigs = load_sigs()
    eth = pd.read_csv(ETH_HOURLY)
    eth['datetime'] = pd.to_datetime(eth['datetime'], utc=True)
    eth = eth.set_index('datetime').sort_index()

    thresholds = [3.0, 3.5, 4.0, 4.5, 5.0]

    rows = []
    for w_name, w_days in [('30d', 30), ('60d', 60)]:
        w_sigs = window_slice(sigs, w_days)
        rally_arr = precompute_rally_48h(w_sigs, eth)
        # Baseline (PROD, no gate)
        b_ret, b_n, b_wr, _, b_dd, b_avg = sim(w_sigs, asset_cfg, rally_arr)
        rows.append({
            'window': w_name, 'config': 'PROD baseline (no rally_48h gate)',
            'return_pct': round(b_ret, 2), 'delta_vs_prod': 0.00,
            'n_trades': b_n, 'win_rate': round(b_wr, 1),
            'avg_pnl_pct': round(b_avg, 2),
            'max_dd_pct': round(b_dd, 2),
            'blocked': 0,
        })
        for thr in thresholds:
            ret, n, wr, blk, dd, avg = sim(w_sigs, asset_cfg, rally_arr, block_thr=thr)
            rows.append({
                'window': w_name,
                'config': f'PROD + block_rally_48h>={thr}%',
                'return_pct': round(ret, 2),
                'delta_vs_prod': round(ret - b_ret, 2),
                'n_trades': n,
                'win_rate': round(wr, 1),
                'avg_pnl_pct': round(avg, 2),
                'max_dd_pct': round(dd, 2),
                'blocked': blk,
            })

    df = pd.DataFrame(rows)

    print(f"\n{'='*120}")
    print(f"  RALLY_48H THRESHOLD SWEEP — PROD + block_rally_48h>=X% only")
    print(f"{'='*120}")
    for w in ['30d', '60d']:
        sub = df[df['window'] == w].copy()
        print(f"\n--- {w} window ---")
        cols = ['config', 'return_pct', 'delta_vs_prod', 'n_trades', 'win_rate',
                'avg_pnl_pct', 'max_dd_pct', 'blocked']
        print(sub[cols].to_string(index=False))

    # Cross-window summary table
    print(f"\n{'='*120}")
    print(f"  CROSS-WINDOW: 30d Δ vs 60d Δ vs blocks per 60d")
    print(f"{'='*120}")
    pivot_d = df.pivot(index='config', columns='window', values='delta_vs_prod')
    pivot_n = df.pivot(index='config', columns='window', values='n_trades')
    pivot_b = df.pivot(index='config', columns='window', values='blocked')
    pivot_dd = df.pivot(index='config', columns='window', values='max_dd_pct')
    pivot_wr = df.pivot(index='config', columns='window', values='win_rate')
    full = pd.concat([
        pivot_d.add_prefix('d_'),
        pivot_n.add_prefix('n_'),
        pivot_b.add_prefix('blk_'),
        pivot_dd.add_prefix('dd_'),
        pivot_wr.add_prefix('wr_'),
    ], axis=1)
    # Sort: baseline first then by 60d delta
    sort_key = full.index.to_series().apply(lambda s: (-100 if 'baseline' in s else float(s.split('>=')[1].rstrip('%'))))
    full = full.assign(_sk=sort_key.values).sort_values('_sk').drop(columns=['_sk'])
    print(full.to_string())

    out = os.path.join(ENGINE, 'output',
                       f'sweep_rally_48h_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
