"""
test_anti_chase_filter.py — sweep an "anti-chase entry filter" against the
60d signal cache to see if blocking late-rally / calm-vol BUYs reduces the
MAX_HOLD drag without surrendering too many winners.

Filter logic at BUY time:
  Block BUY if ALL of:
    - ETH 24h return > eth_thr  (modest rally already in play)
    - vol_24h / vol_30d < vol_thr  (below-average vol)
    - (optional) BTC 24h return > btc_thr

Reads (read-only):
  data/eth_sl_signals_90d.pkl
  data/eth_hourly_data.csv
  data/btc_hourly_data.csv
  config/regime_config_ed.json

Writes:
  output/anti_chase_filter_<timestamp>.csv
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PKL = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
REGIME_CFG = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
ETH_HOURLY = os.path.join(ENGINE, 'data', 'eth_hourly_data.csv')
BTC_HOURLY = os.path.join(ENGINE, 'data', 'btc_hourly_data.csv')
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


def precompute_filter_features(sigs, eth_df, btc_df):
    """For each signal, compute the 3 anti-chase features:
    eth_24h_ret, vol_ratio_24h_30d, btc_24h_ret. Return parallel arrays."""
    n = len(sigs)
    eth_24h = np.full(n, np.nan)
    vol_ratio = np.full(n, np.nan)
    btc_24h = np.full(n, np.nan)

    eth_close = eth_df['close']
    btc_close = btc_df['close'] if btc_df is not None else None

    for i, s in enumerate(sigs):
        dt = s['datetime']
        if dt.tzinfo is None:
            dt = dt.tz_localize('UTC')
        else:
            dt = dt.tz_convert('UTC')

        # ETH 24h return
        sub24 = eth_close[(eth_close.index < dt) &
                          (eth_close.index >= dt - timedelta(hours=24))]
        if len(sub24) > 4:
            eth_24h[i] = 100 * (sub24.iloc[-1] / sub24.iloc[0] - 1)

        # Vol ratio
        sub30d = eth_close[(eth_close.index < dt) &
                           (eth_close.index >= dt - timedelta(days=30))]
        if len(sub24) > 4 and len(sub30d) > 100:
            v24 = np.log(sub24).diff().std()
            v30d = np.log(sub30d).diff().std()
            if v30d > 0:
                vol_ratio[i] = v24 / v30d

        # BTC 24h return
        if btc_close is not None:
            bsub = btc_close[(btc_close.index < dt) &
                             (btc_close.index >= dt - timedelta(hours=24))]
            if len(bsub) > 4:
                btc_24h[i] = 100 * (bsub.iloc[-1] / bsub.iloc[0] - 1)

    return eth_24h, vol_ratio, btc_24h


def sim_with_filter(sigs, asset_cfg, eth_24h_arr, vol_ratio_arr, btc_24h_arr,
                    eth_thr, vol_thr, btc_thr, use_btc=False):
    """Same regime-switched simulator as before, with one addition: at BUY
    decision time, if the filter conditions are met, BLOCK the BUY (treat as
    if signal were HOLD).
    """
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
    blocked_by_filter = 0

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
            if rs >= bt_s or rl >= bt_l: bull_cd = max(bull_cd, bcd_h)
        if rcd_h > 0 and bear_rs is not None:
            rs = bear_rs[i] if not np.isnan(bear_rs[i]) else 0
            rl = bear_rl[i] if not np.isnan(bear_rl[i]) else 0
            if rs >= rt_s or rl >= rt_l: bear_cd = max(bear_cd, rcd_h)
        active_cd = bull_cd if regime == 'bull' else bear_cd

        if sig == 'BUY' and sconf >= conf_thr and not in_pos and active_cd <= 0:
            # ANTI-CHASE FILTER
            eth_r = eth_24h_arr[i]
            vr = vol_ratio_arr[i]
            btc_r = btc_24h_arr[i]
            blocked = False
            if not np.isnan(eth_r) and not np.isnan(vr):
                eth_block = eth_r > eth_thr
                vol_block = vr < vol_thr
                if use_btc:
                    btc_block = (not np.isnan(btc_r)) and btc_r > btc_thr
                    blocked = eth_block and vol_block and btc_block
                else:
                    blocked = eth_block and vol_block
            if blocked:
                blocked_by_filter += 1
            else:
                held = cash * (1 - FEE) / price
                cash = 0; in_pos = True; entry_px = price
                hold_since_entry = 0
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = shield_on and min_pnl > 0 and cur_pnl < min_pnl
            if not shield_blocks or override_expired:
                exit_reason = 'MAX_HOLD' if (shield_on and override_expired and cur_pnl < min_pnl) else 'MODEL'
                cash = held * price * (1 - FEE)
                trade_log.append({'pnl_pct': cur_pnl, 'exit_reason': exit_reason})
                held = 0; in_pos = False; hold_since_entry = 0

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        cash = held * sigs[-1]['close'] * (1 - FEE)
        trade_log.append({'pnl_pct': (sigs[-1]['close'] / entry_px - 1) * 100,
                          'exit_reason': 'OPEN_AT_END'})

    ret = (cash / 1000.0 - 1) * 100
    n_tr = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    wr = (wins / n_tr * 100) if n_tr else 0
    n_mh = sum(1 for t in trade_log if t['exit_reason'] == 'MAX_HOLD')
    mh_pnl = sum(t['pnl_pct'] for t in trade_log if t['exit_reason'] == 'MAX_HOLD')
    return ret, n_tr, wr, n_mh, mh_pnl, blocked_by_filter


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    sigs = load_sigs()
    win = window_slice(sigs, 60)
    print(f"60d window: {len(win)} signals\n")

    eth = pd.read_csv(ETH_HOURLY)
    eth['datetime'] = pd.to_datetime(eth['datetime'], utc=True)
    eth = eth.set_index('datetime').sort_index()
    btc = pd.read_csv(BTC_HOURLY) if os.path.exists(BTC_HOURLY) else None
    if btc is not None:
        btc['datetime'] = pd.to_datetime(btc['datetime'], utc=True)
        btc = btc.set_index('datetime').sort_index()

    print("Precomputing filter features for each signal...")
    eth_24h_arr, vol_ratio_arr, btc_24h_arr = precompute_filter_features(win, eth, btc)
    print(f"  ETH 24h returns:    {pd.Series(eth_24h_arr).describe()[['mean','50%','min','max']].to_dict()}")
    print(f"  Vol ratio 24h/30d:  {pd.Series(vol_ratio_arr).describe()[['mean','50%','min','max']].to_dict()}")
    print(f"  BTC 24h returns:    {pd.Series(btc_24h_arr).describe()[['mean','50%','min','max']].to_dict()}\n")

    # Baseline (no filter)
    base_ret, base_n, base_wr, base_mh, base_mhp, _ = sim_with_filter(
        win, asset_cfg, eth_24h_arr, vol_ratio_arr, btc_24h_arr,
        eth_thr=999, vol_thr=-999, btc_thr=999)  # impossible thresholds = no filtering
    print(f"BASELINE (no anti-chase filter): "
          f"{base_ret:+.2f}% | {base_n} trades | WR {base_wr:.0f}% | "
          f"MAX_HOLD: {base_mh} fires, sum {base_mhp:+.2f}%\n")

    # Sweep
    eth_thrs = [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    vol_thrs = [0.6, 0.7, 0.8, 0.9, 1.0]
    use_btc_options = [False, True]
    btc_thrs = [0.0, 0.5]

    rows = []
    for eth_t in eth_thrs:
        for vol_t in vol_thrs:
            for use_btc in use_btc_options:
                for btc_t in btc_thrs if use_btc else [None]:
                    ret, n_tr, wr, n_mh, mh_pnl, blocked = sim_with_filter(
                        win, asset_cfg, eth_24h_arr, vol_ratio_arr, btc_24h_arr,
                        eth_t, vol_t, btc_t if btc_t is not None else 0,
                        use_btc=use_btc)
                    rows.append({
                        'eth_thr': eth_t,
                        'vol_thr': vol_t,
                        'use_btc': use_btc,
                        'btc_thr': btc_t if use_btc else None,
                        'return_pct': round(ret, 2),
                        'delta_vs_base': round(ret - base_ret, 2),
                        'n_trades': n_tr,
                        'win_rate': round(wr, 1),
                        'max_hold_fires': n_mh,
                        'max_hold_pnl_sum': round(mh_pnl, 2),
                        'blocked_by_filter': blocked,
                    })

    df = pd.DataFrame(rows).sort_values('return_pct', ascending=False)

    print(f"\n{'='*130}")
    print(f"  TOP 15 by return")
    print(f"{'='*130}")
    print(df.head(15).to_string(index=False))

    print(f"\n{'='*130}")
    print(f"  WORST 5")
    print(f"{'='*130}")
    print(df.tail(5).to_string(index=False))

    print(f"\n{'='*130}")
    print(f"  BEST CONFIGS WHERE MAX_HOLD ≤ 10 (cleanest reduction)")
    print(f"{'='*130}")
    clean = df[df['max_hold_fires'] <= 10].sort_values('return_pct', ascending=False).head(10)
    print(clean.to_string(index=False))

    out = os.path.join(ENGINE, 'output',
                       f'anti_chase_filter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nFull sweep saved to: {out}")


if __name__ == '__main__':
    main()
