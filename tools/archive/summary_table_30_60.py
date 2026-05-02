"""
summary_table_30_60.py — clean summary of where we stand and what shipping
each suggested option would project to. 30d + 60d.

Rows:
  1. Buy-and-hold ETH (passive)
  2. Pure model signals (no shield, no rally cooldown gate)
  3. Current production (A: shields ON, gates ON)
  4. Option U: PROD + bull_conf=90 + block_rally_72h>=7%
  5. Option O: bear_shield=OFF + block_rally_48h>=4%
  6. Option F: PROD + block_rally_48h>=3%

Reads cache + regime_config + eth_hourly. Read-only. Prints clean tables.
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


def precompute_prior_rallies(sigs, eth_df):
    n = len(sigs)
    closes = eth_df['close']
    rally_windows = [24, 48, 72]
    out = {h: np.full(n, np.nan) for h in rally_windows}
    for i, s in enumerate(sigs):
        dt = s['datetime']
        if dt.tzinfo is None: dt = dt.tz_localize('UTC')
        else: dt = dt.tz_convert('UTC')
        cur_idx = eth_df.index[eth_df.index <= dt]
        if len(cur_idx) == 0: continue
        cur_close = closes.loc[cur_idx[-1]]
        for h in rally_windows:
            prior_dt = dt - pd.Timedelta(hours=h)
            prior_idx = eth_df.index[eth_df.index <= prior_dt]
            if len(prior_idx) > 0:
                pp = closes.loc[prior_idx[-1]]
                if pp > 0:
                    out[h][i] = (cur_close / pp - 1) * 100
    return out


def sim(sigs, asset_cfg, prior_rallies,
        bull_shield_override=None, bear_shield_override=None,
        bull_conf=None,
        block_rally_h=None, block_rally_pct=None,
        disable_gates=False):
    bull_conf_thr = bull_conf if bull_conf is not None else float(asset_cfg.get('bull', {}).get('min_confidence', 80))
    bear_conf_thr = float(asset_cfg.get('bear', {}).get('min_confidence', 65))
    bull_shield_default = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield_default = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    bull_shield_on = bull_shield_default if bull_shield_override is None else bull_shield_override
    bear_shield_on = bear_shield_default if bear_shield_override is None else bear_shield_override
    min_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.5))
    max_hold_h = int(asset_cfg.get('max_hold_hours', 10))
    bull_rally = None if disable_gates else get_rally_tuple(asset_cfg, 'bull')
    bear_rally = None if disable_gates else get_rally_tuple(asset_cfg, 'bear')

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
        shield_on = bull_shield_on if regime == 'bull' else bear_shield_on

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
            if block_rally_h is not None and block_rally_pct is not None:
                v = prior_rallies[block_rally_h][i]
                if not np.isnan(v) and v >= block_rally_pct:
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


def buy_and_hold(sigs):
    p0 = float(sigs[0]['close'])
    p1 = float(sigs[-1]['close'])
    # one buy + one sell -> two fees
    return ((p1 / p0) * (1 - FEE) * (1 - FEE) - 1) * 100


def main():
    print("Loading...")
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']
    sigs = load_sigs()
    eth = pd.read_csv(ETH_HOURLY)
    eth['datetime'] = pd.to_datetime(eth['datetime'], utc=True)
    eth = eth.set_index('datetime').sort_index()

    windows = {'30d': window_slice(sigs, 30),
               '60d': window_slice(sigs, 60)}

    results = {}
    for w_name, w_sigs in windows.items():
        pr = precompute_prior_rallies(w_sigs, eth)
        bh = buy_and_hold(w_sigs)
        # 1. Pure model — no shield, no gate
        pure_ret, pure_n, pure_wr, _, pure_dd, pure_avg = sim(
            w_sigs, asset_cfg, pr,
            bull_shield_override=False, bear_shield_override=False,
            disable_gates=True)
        # 2. Pure model + gates (no shield)
        gate_ret, gate_n, gate_wr, _, gate_dd, gate_avg = sim(
            w_sigs, asset_cfg, pr,
            bull_shield_override=False, bear_shield_override=False)
        # 3. Current PRODUCTION
        prod_ret, prod_n, prod_wr, _, prod_dd, prod_avg = sim(
            w_sigs, asset_cfg, pr)
        # 4. Option U: PROD + bull_conf=90 + block_rally_72h>=7
        u_ret, u_n, u_wr, u_blk, u_dd, u_avg = sim(
            w_sigs, asset_cfg, pr,
            bull_conf=90, block_rally_h=72, block_rally_pct=7.0)
        # 5. Option O: bear_shield=OFF + block_rally_48h>=4
        o_ret, o_n, o_wr, o_blk, o_dd, o_avg = sim(
            w_sigs, asset_cfg, pr,
            bear_shield_override=False, block_rally_h=48, block_rally_pct=4.0)
        # 6. Option F: PROD + block_rally_48h>=3
        f_ret, f_n, f_wr, f_blk, f_dd, f_avg = sim(
            w_sigs, asset_cfg, pr,
            block_rally_h=48, block_rally_pct=3.0)

        results[w_name] = {
            'B&H ETH (passive)':                                  (bh, 1, 0, 0, 0, 0, 0),
            'Pure model signals (no shield, no gate)':            (pure_ret, pure_n, pure_wr, 0, pure_dd, pure_avg, 0),
            'Pure model + gates (no shield)':                     (gate_ret, gate_n, gate_wr, 0, gate_dd, gate_avg, 0),
            'CURRENT PRODUCTION (shields ON, gates ON)':          (prod_ret, prod_n, prod_wr, 0, prod_dd, prod_avg, 0),
            'Option U: PROD + bull_conf=90 + block_rally_72h>=7%': (u_ret, u_n, u_wr, u_blk, u_dd, u_avg, 0),
            'Option O: bear_shield=OFF + block_rally_48h>=4%':     (o_ret, o_n, o_wr, o_blk, o_dd, o_avg, 0),
            'Option F: PROD + block_rally_48h>=3%':                (f_ret, f_n, f_wr, f_blk, f_dd, f_avg, 0),
        }

    # Print combined table
    rows = []
    for label in list(results['30d'].keys()):
        r30 = results['30d'][label]
        r60 = results['60d'][label]
        prod30 = results['30d']['CURRENT PRODUCTION (shields ON, gates ON)'][0]
        prod60 = results['60d']['CURRENT PRODUCTION (shields ON, gates ON)'][0]
        rows.append({
            'config': label,
            'ret_30d': round(r30[0], 2),
            'd_vs_prod_30d': round(r30[0] - prod30, 2),
            'ret_60d': round(r60[0], 2),
            'd_vs_prod_60d': round(r60[0] - prod60, 2),
            'tr_30d': r30[1],
            'tr_60d': r60[1],
            'wr_30d': round(r30[2], 0),
            'wr_60d': round(r60[2], 0),
            'avg_pnl_30d': round(r30[5], 2),
            'avg_pnl_60d': round(r60[5], 2),
            'maxdd_30d': round(r30[4], 2),
            'maxdd_60d': round(r60[4], 2),
            'blocked_30d': r30[3],
            'blocked_60d': r60[3],
        })
    df = pd.DataFrame(rows)

    print(f"\n{'='*150}")
    print(f"  CURRENT STATE + PROJECTED OPTIONS — 30d & 60d (ETH only)")
    print(f"{'='*150}\n")

    # Compact display
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 60)
    cols_show = ['config', 'ret_30d', 'd_vs_prod_30d', 'ret_60d', 'd_vs_prod_60d',
                 'tr_30d', 'tr_60d', 'wr_30d', 'wr_60d', 'maxdd_30d', 'maxdd_60d']
    print(df[cols_show].to_string(index=False))

    print(f"\n  Trade-level detail (avg PnL per trade, blocks if any):")
    cols2 = ['config', 'avg_pnl_30d', 'avg_pnl_60d', 'blocked_30d', 'blocked_60d']
    print(df[cols2].to_string(index=False))

    out = os.path.join(ENGINE, 'output',
                       f'summary_30_60_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
