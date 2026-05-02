"""
compare_shield_vs_rally_block.py — apples-to-apples on 30d + 60d.

Test matrix:
  Current production = bull_shield ON, bear_shield ON, no rally-block
  - Toggle each shield ON/OFF
  - Add rally-block entry filter at various thresholds
  - Combine: replace shields with rally-block

Reads: cache + regime_config + eth_hourly. Writes output/cmp_shield_vs_block_*.csv.
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
        block_rally_h=None, block_rally_pct=None):
    bull_conf_thr = bull_conf if bull_conf is not None else float(asset_cfg.get('bull', {}).get('min_confidence', 80))
    bear_conf_thr = float(asset_cfg.get('bear', {}).get('min_confidence', 65))
    bull_shield_default = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield_default = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    bull_shield_on = bull_shield_default if bull_shield_override is None else bull_shield_override
    bear_shield_on = bear_shield_default if bear_shield_override is None else bear_shield_override
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

        # Track equity drawdown
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
    avg_pnl = float(np.mean([t['pnl_pct'] for t in trade_log])) if trade_log else 0.0
    return ret, n, wr, blocked, max_dd_pct, avg_pnl


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

    rows = []
    for w_name, w_sigs in windows.items():
        print(f"\n[{w_name}: {len(w_sigs)} sigs]")
        pr = precompute_prior_rallies(w_sigs, eth)

        # The matrix
        configs = [
            # === Reference baselines ===
            ('A. PROD (current): bull_shield=ON, bear_shield=ON, NO block',
             dict()),
            ('B. shields OFF (no block, no shield)',
             dict(bull_shield_override=False, bear_shield_override=False)),
            ('C. bull_shield=OFF only, bear_shield=ON',
             dict(bull_shield_override=False, bear_shield_override=True)),
            ('D. bull_shield=ON, bear_shield=OFF only',
             dict(bull_shield_override=True, bear_shield_override=False)),
            # === Add rally-block ON TOP of current shields ===
            ('E. PROD shields + block_rally_24h>=3%',
             dict(block_rally_h=24, block_rally_pct=3.0)),
            ('F. PROD shields + block_rally_48h>=3%',
             dict(block_rally_h=48, block_rally_pct=3.0)),
            ('G. PROD shields + block_rally_48h>=4%',
             dict(block_rally_h=48, block_rally_pct=4.0)),
            ('H. PROD shields + block_rally_48h>=5%',
             dict(block_rally_h=48, block_rally_pct=5.0)),
            ('I. PROD shields + block_rally_72h>=5%',
             dict(block_rally_h=72, block_rally_pct=5.0)),
            ('J. PROD shields + block_rally_72h>=7%',
             dict(block_rally_h=72, block_rally_pct=7.0)),
            # === REPLACE bull shield with rally-block ===
            ('K. bull_shield=OFF + block_rally_48h>=3%',
             dict(bull_shield_override=False, block_rally_h=48, block_rally_pct=3.0)),
            ('L. bull_shield=OFF + block_rally_48h>=4%',
             dict(bull_shield_override=False, block_rally_h=48, block_rally_pct=4.0)),
            ('M. bull_shield=OFF + block_rally_72h>=5%',
             dict(bull_shield_override=False, block_rally_h=72, block_rally_pct=5.0)),
            # === REPLACE bear shield with rally-block ===
            ('N. bear_shield=OFF + block_rally_48h>=3%',
             dict(bear_shield_override=False, block_rally_h=48, block_rally_pct=3.0)),
            ('O. bear_shield=OFF + block_rally_48h>=4%',
             dict(bear_shield_override=False, block_rally_h=48, block_rally_pct=4.0)),
            # === REPLACE BOTH shields with rally-block ===
            ('P. shields=OFF + block_rally_48h>=3%',
             dict(bull_shield_override=False, bear_shield_override=False,
                  block_rally_h=48, block_rally_pct=3.0)),
            ('Q. shields=OFF + block_rally_48h>=4%',
             dict(bull_shield_override=False, bear_shield_override=False,
                  block_rally_h=48, block_rally_pct=4.0)),
            ('R. shields=OFF + block_rally_72h>=5%',
             dict(bull_shield_override=False, bear_shield_override=False,
                  block_rally_h=72, block_rally_pct=5.0)),
            # === Combos with bull_conf=90 ===
            ('S. PROD shields + bull_conf=90 + block_rally_48h>=4%',
             dict(bull_conf=90, block_rally_h=48, block_rally_pct=4.0)),
            ('T. PROD shields + bull_conf=90 + block_rally_72h>=5%',
             dict(bull_conf=90, block_rally_h=72, block_rally_pct=5.0)),
            ('U. PROD shields + bull_conf=90 + block_rally_72h>=7%',
             dict(bull_conf=90, block_rally_h=72, block_rally_pct=7.0)),
            ('V. shields=OFF + bull_conf=90 + block_rally_48h>=4%',
             dict(bull_shield_override=False, bear_shield_override=False,
                  bull_conf=90, block_rally_h=48, block_rally_pct=4.0)),
            ('W. shields=OFF + bull_conf=90 + block_rally_72h>=5%',
             dict(bull_shield_override=False, bear_shield_override=False,
                  bull_conf=90, block_rally_h=72, block_rally_pct=5.0)),
        ]

        for name, kwargs in configs:
            ret, n, wr, blk, dd, avg = sim(w_sigs, asset_cfg, pr, **kwargs)
            rows.append({
                'window': w_name,
                'config': name,
                'return_pct': round(ret, 2),
                'n_trades': n,
                'win_rate': round(wr, 1),
                'avg_pnl_pct': round(avg, 2),
                'max_dd_pct': round(dd, 2),
                'blocked': blk,
            })

    df = pd.DataFrame(rows)

    for w in ['30d', '60d']:
        sub = df[df['window'] == w].copy()
        base_ret = sub.iloc[0]['return_pct']
        sub['delta_vs_prod'] = (sub['return_pct'] - base_ret).round(2)
        sub = sub.sort_values('return_pct', ascending=False)
        print(f"\n{'='*120}")
        print(f"  {w} — sorted by return_pct (vs PROD baseline = {base_ret:+.2f}%)")
        print(f"{'='*120}")
        cols = ['config', 'return_pct', 'delta_vs_prod', 'n_trades', 'win_rate',
                'avg_pnl_pct', 'max_dd_pct', 'blocked']
        print(sub[cols].to_string(index=False))

    out = os.path.join(ENGINE, 'output',
                       f'cmp_shield_vs_block_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
