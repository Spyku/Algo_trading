"""
T1b PROXY — confidence-threshold sweep as a stand-in for true multi-horizon
ensemble vote (full version requires per-horizon signal regen, ~3h).

Rationale: ensemble agreement and signal confidence are correlated. When
multiple horizons agree, the active model fires at higher confidence. So
raising the bull confidence threshold (80 -> 85 -> 90 -> 95) is a cheap
proxy for "require more model agreement before BUY."

Sweep:
  bull_conf_thr in {80, 82, 85, 88, 90, 92, 95}
  bear_conf_thr in {65, 70, 75, 80, 85, 90}

Reads (read-only):
  data/eth_sl_signals_90d.pkl
  config/regime_config_ed.json

Writes:
  output/t1b_proxy_conf_<timestamp>.csv
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PKL = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
REGIME_CFG = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
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


def sim_with_conf(sigs, asset_cfg, bull_thr, bear_thr):
    """Standard sim but with custom per-regime confidence thresholds."""
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

    for i, s in enumerate(sigs):
        price = float(s['close'])
        regime = s.get('regime', 'bull')
        sig = s['signal']
        sconf = float(s.get('confidence', 0))
        conf_thr = bull_thr if regime == 'bull' else bear_thr
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
    win = window_slice(sigs, 60)
    print(f"60d window: {len(win)} signals\n")

    cur_bull = float(asset_cfg['bull']['min_confidence'])
    cur_bear = float(asset_cfg['bear']['min_confidence'])
    print(f"Current production: bull={cur_bull}% / bear={cur_bear}%\n")

    base_ret, base_n, base_wr, _, _ = sim_with_conf(win, asset_cfg, cur_bull, cur_bear)
    print(f"BASELINE ({cur_bull}/{cur_bear}): {base_ret:+.2f}% | {base_n} trades | WR {base_wr:.0f}%\n")

    bull_thrs = [70, 75, 78, 80, 82, 85, 88, 90, 92, 95]
    bear_thrs = [55, 60, 65, 70, 75, 80, 85, 90, 95]

    rows = []
    for bt in bull_thrs:
        for rt in bear_thrs:
            ret, n, wr, nb, nbr = sim_with_conf(win, asset_cfg, bt, rt)
            rows.append({
                'bull_thr': bt, 'bear_thr': rt,
                'return_pct': round(ret, 2),
                'delta_vs_base': round(ret - base_ret, 2),
                'n_trades': n, 'win_rate': round(wr, 1),
                'n_bull_trades': nb, 'n_bear_trades': nbr,
            })

    df = pd.DataFrame(rows).sort_values('return_pct', ascending=False)

    print(f"{'='*110}")
    print(f"  T1b PROXY — confidence-threshold sweep ({len(rows)} combos)")
    print(f"{'='*110}\n")

    print("TOP 15 by return:")
    print(df.head(15).to_string(index=False))

    print("\nBOTTOM 5 by return:")
    print(df.tail(5).to_string(index=False))

    # Pivot for visualization
    pivot = df.pivot(index='bull_thr', columns='bear_thr', values='return_pct')
    print(f"\n{'='*110}")
    print(f"  RETURN HEATMAP (rows=bull_thr, cols=bear_thr). Baseline at "
          f"({int(cur_bull)},{int(cur_bear)})={base_ret:.2f}%")
    print(f"{'='*110}")
    print(pivot.round(1).to_string())

    out = os.path.join(ENGINE, 'output',
                       f't1b_proxy_conf_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nFull sweep saved to: {out}")


if __name__ == '__main__':
    main()
