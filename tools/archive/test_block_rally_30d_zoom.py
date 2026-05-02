"""
test_block_rally_30d_zoom.py — focused 30d test of `block_rally_Xh >= Y%`.

The earlier sweep showed `block_rally_48h >= 3%` is +6.16pp on 60d but
-0.69pp on 30d. Drill into 30d:

1. Finer threshold sweep (rally_24h/48h/72h × thresholds 2..6 in 0.5 steps)
2. For top winners, list every blocked trade — entry dt, entry px, prior
   rally values, what the ACTUAL outcome was if we hadn't blocked
3. Bucketed 30d-only diagnostic
4. Combos with bull_conf=90 in 30d

Reads: cache + regime_config + eth_hourly. Writes output/block_rally_30d_zoom_*.csv.
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
    rally_windows = [12, 18, 24, 36, 48, 60, 72, 96, 168]
    out = {h: np.full(n, np.nan) for h in rally_windows}
    for i, s in enumerate(sigs):
        dt = s['datetime']
        if dt.tzinfo is None: dt = dt.tz_localize('UTC')
        else: dt = dt.tz_convert('UTC')
        cur_idx = eth_df.index[eth_df.index <= dt]
        if len(cur_idx) == 0:
            continue
        cur_close = closes.loc[cur_idx[-1]]
        for h in rally_windows:
            prior_dt = dt - pd.Timedelta(hours=h)
            prior_idx = eth_df.index[eth_df.index <= prior_dt]
            if len(prior_idx) > 0:
                pp = closes.loc[prior_idx[-1]]
                if pp > 0:
                    out[h][i] = (cur_close / pp - 1) * 100
    return out


def sim_with_block(sigs, asset_cfg, prior_rallies,
                   block_rally_h=None, block_rally_pct=None,
                   bull_conf=None, also_block_24h_pct=None):
    bull_conf_thr = bull_conf if bull_conf is not None else float(asset_cfg.get('bull', {}).get('min_confidence', 80))
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
    blocked_log = []
    blocked_count = 0

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
            block_reason = None
            if block_rally_h is not None and block_rally_pct is not None:
                v = prior_rallies[block_rally_h][i]
                if not np.isnan(v) and v >= block_rally_pct:
                    block = True
                    block_reason = f'rally_{block_rally_h}h={v:.2f}%'
            if not block and also_block_24h_pct is not None:
                v24 = prior_rallies[24][i]
                if not np.isnan(v24) and v24 >= also_block_24h_pct:
                    block = True
                    block_reason = f'rally_24h={v24:.2f}%'
            if block:
                blocked_count += 1
                blocked_log.append({
                    'idx': i, 'entry_dt': s['datetime'], 'entry_px': price,
                    'regime': regime, 'conf': sconf, 'reason': block_reason,
                    'r24h': prior_rallies[24][i],
                    'r48h': prior_rallies[48][i],
                    'r72h': prior_rallies[72][i],
                })
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

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        cash = held * sigs[-1]['close'] * (1 - FEE)
        trade_log.append({'pnl_pct': (sigs[-1]['close'] / entry_px - 1) * 100, 'regime': 'bull'})

    ret = (cash / 1000.0 - 1) * 100
    n = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    return ret, n, (wins / n * 100 if n else 0), blocked_count, blocked_log


def collect_buys_with_outcome(sigs, asset_cfg, prior_rallies, bull_conf=None):
    """Run baseline + record EVERY BUY with its prior rally + outcome."""
    bull_conf_thr = bull_conf if bull_conf is not None else float(asset_cfg.get('bull', {}).get('min_confidence', 80))
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

    in_pos = False; entry_px = 0.0
    hold_since_entry = 0; bull_cd = bear_cd = 0
    trades = []
    open_buy = None

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
            open_buy = {
                'entry_dt': s['datetime'], 'entry_px': price,
                'regime': regime, 'conf': sconf,
                'r12h': prior_rallies[12][i], 'r18h': prior_rallies[18][i],
                'r24h': prior_rallies[24][i], 'r36h': prior_rallies[36][i],
                'r48h': prior_rallies[48][i], 'r60h': prior_rallies[60][i],
                'r72h': prior_rallies[72][i], 'r96h': prior_rallies[96][i],
                'r168h': prior_rallies[168][i],
            }
            in_pos = True; entry_px = price; hold_since_entry = 0
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = shield_on and min_pnl > 0 and cur_pnl < min_pnl
            if not shield_blocks or override_expired:
                if open_buy is not None:
                    open_buy['pnl_pct'] = cur_pnl
                    open_buy['exit_dt'] = s['datetime']
                    open_buy['hold_h'] = hold_since_entry
                    trades.append(open_buy)
                    open_buy = None
                in_pos = False; hold_since_entry = 0

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos and open_buy is not None:
        open_buy['pnl_pct'] = (sigs[-1]['close'] / entry_px - 1) * 100
        open_buy['exit_dt'] = sigs[-1]['datetime']
        open_buy['hold_h'] = hold_since_entry
        trades.append(open_buy)
    return trades


def main():
    print("Loading...")
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']
    sigs = load_sigs()
    eth = pd.read_csv(ETH_HOURLY)
    eth['datetime'] = pd.to_datetime(eth['datetime'], utc=True)
    eth = eth.set_index('datetime').sort_index()

    win30 = window_slice(sigs, 30)
    print(f"\n30d window: {len(win30)} sigs ({win30[0]['datetime']} - {win30[-1]['datetime']})")
    pr = precompute_prior_rallies(win30, eth)

    # === Per-trade outcomes on 30d ===
    print("\n=== EVERY BUY IN LAST 30d WITH PRIOR-RALLY CONTEXT + OUTCOME ===")
    trades = collect_buys_with_outcome(win30, asset_cfg, pr)
    df_t = pd.DataFrame(trades)
    print(f"Baseline trades: {len(df_t)}, avg PnL = {df_t['pnl_pct'].mean():.2f}%, "
          f"WR = {(df_t['pnl_pct'] > 0).mean()*100:.0f}%, "
          f"sum PnL = {df_t['pnl_pct'].sum():.2f}%")
    print(df_t[['entry_dt', 'entry_px', 'regime', 'conf', 'r24h', 'r48h', 'r72h',
                'pnl_pct', 'hold_h']].round(2).to_string())

    # === Bucket 30d trades by rally_48h ===
    print("\n--- 30d-only bucketed by rally_48h ---")
    bucket_edges = [-100, -3, -1, 0, 1, 2, 3, 4, 6, 100]
    bucket_lbls = ['<-3', '-3..-1', '-1..0', '0..1', '1..2', '2..3', '3..4', '4..6', '>=6']
    df_t['r48h_bucket'] = pd.cut(df_t['r48h'], bucket_edges, labels=bucket_lbls)
    agg = df_t.groupby('r48h_bucket', observed=False).agg(
        n=('pnl_pct', 'count'),
        wr=('pnl_pct', lambda s: (s > 0).mean() * 100),
        mean_pnl=('pnl_pct', 'mean'),
        sum_pnl=('pnl_pct', 'sum'),
    ).round(2)
    print(agg.to_string())

    print("\n--- 30d-only bucketed by rally_24h ---")
    df_t['r24h_bucket'] = pd.cut(df_t['r24h'], bucket_edges, labels=bucket_lbls)
    print(df_t.groupby('r24h_bucket', observed=False).agg(
        n=('pnl_pct', 'count'),
        wr=('pnl_pct', lambda s: (s > 0).mean() * 100),
        mean_pnl=('pnl_pct', 'mean'),
        sum_pnl=('pnl_pct', 'sum'),
    ).round(2).to_string())

    # === Finer threshold sweep on 30d ===
    print("\n=== 30d FINER SWEEP ===")
    base_ret, base_n, base_wr, _, _ = sim_with_block(win30, asset_cfg, pr)
    print(f"30d baseline: {base_ret:+.2f}% / {base_n} tr / WR {base_wr:.0f}%")

    rows = []
    for h in [18, 24, 36, 48, 60, 72, 96]:
        for thr in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]:
            ret, n, wr, blk, _ = sim_with_block(
                win30, asset_cfg, pr,
                block_rally_h=h, block_rally_pct=thr)
            rows.append({
                'config': f'block_rally_{h}h>={thr}%',
                'return_pct': round(ret, 2),
                'delta_30d': round(ret - base_ret, 2),
                'n_trades': n,
                'win_rate': round(wr, 1),
                'blocked': blk,
            })
    # Combos with bull_conf=90
    for h in [24, 48, 72]:
        for thr in [2.5, 3.0, 4.0, 5.0]:
            ret, n, wr, blk, _ = sim_with_block(
                win30, asset_cfg, pr,
                block_rally_h=h, block_rally_pct=thr, bull_conf=90)
            rows.append({
                'config': f'block_rally_{h}h>={thr}% + bull_conf=90',
                'return_pct': round(ret, 2),
                'delta_30d': round(ret - base_ret, 2),
                'n_trades': n,
                'win_rate': round(wr, 1),
                'blocked': blk,
            })

    df = pd.DataFrame(rows).sort_values('delta_30d', ascending=False)
    print("\n--- TOP 30 by 30d delta ---")
    print(df.head(30).to_string(index=False))
    print("\n--- BOTTOM 10 by 30d delta ---")
    print(df.tail(10).to_string(index=False))

    # === Show what the WINNER blocks ===
    if df.iloc[0]['delta_30d'] > 0:
        winner = df.iloc[0]['config']
        # Parse h and thr from config string
        parts = winner.split('+')[0].strip()
        # block_rally_Xh>=Y%
        h_str = parts.split('block_rally_')[1].split('h>=')[0]
        thr_str = parts.split('h>=')[1].rstrip('%').strip()
        bull_conf_arg = 90 if 'bull_conf=90' in winner else None
        h = int(h_str); thr = float(thr_str)
        ret, n, wr, blk, blocked_log = sim_with_block(
            win30, asset_cfg, pr,
            block_rally_h=h, block_rally_pct=thr, bull_conf=bull_conf_arg)
        print(f"\n=== WINNER: {winner} -> {ret:+.2f}% / {n}tr / WR {wr:.0f}% / blocked {blk} ===")
        if blocked_log:
            df_blk = pd.DataFrame(blocked_log)
            print(df_blk[['entry_dt', 'entry_px', 'regime', 'conf', 'r24h', 'r48h', 'r72h', 'reason']].round(2).to_string())

    out = os.path.join(ENGINE, 'output',
                       f'block_rally_30d_zoom_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved sweep: {out}")


if __name__ == '__main__':
    main()
