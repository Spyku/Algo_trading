"""
T1a — Consecutive-SELL exit filter (proxy for asymmetric exit horizons).

Hypothesis: model SELL at hour H is sometimes noise; if the rally continues,
the model will keep saying SELL each subsequent hour. Requiring K consecutive
SELLs before exiting filters out single-bar noise and lets winners run.

This is a CHEAP proxy for "use a longer horizon model for exits" — the cache
only has the production horizon's signals, so we simulate the longer-horizon
behavior by demanding consensus across consecutive bars.

Sweep:
  K in {1, 2, 3, 4}: number of consecutive SELL signals required to exit
  (K=1 == current behavior == baseline)

PLUS a separate sweep:
  min_hold_h in {0, 2, 4, 6, 8}: ignore model SELL for the first N hours
  after entry. Forces winners to run at least N hours before exit considered.

Reads (read-only):
  data/eth_sl_signals_90d.pkl
  config/regime_config_ed.json
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


def sim_with_filters(sigs, asset_cfg, k_consecutive=1, min_hold_h=0):
    """Same regime-switched simulator with two new exit filters:
      - k_consecutive: require K consecutive SELLs before exit
      - min_hold_h: ignore SELL signals during first N hours after entry
                    (max_hold failsafe still applies)
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
    consecutive_sells = 0
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
            consecutive_sells = 0
        elif sig == 'SELL' and in_pos:
            # Track consecutive SELL signals (only updated on SELL bars)
            consecutive_sells += 1

            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = shield_on and min_pnl > 0 and cur_pnl < min_pnl
            min_hold_active = hold_since_entry < min_hold_h
            sells_confirmed = consecutive_sells >= k_consecutive

            # Exit logic mirrors canonical sim:
            #   exit if (shield doesn't block OR max_hold expired) AND
            #          (min_hold passed) AND (k consecutive SELLs met)
            # max_hold fires only on SELL bars (matches canonical sim).
            allowed_by_shield_or_failsafe = (not shield_blocks) or override_expired
            allowed_by_min_hold = not min_hold_active
            allowed_by_consensus = sells_confirmed

            if allowed_by_shield_or_failsafe and allowed_by_min_hold and allowed_by_consensus:
                exit_reason = 'MAX_HOLD' if (shield_on and override_expired and cur_pnl < min_pnl) else 'MODEL'
                cash = held * price * (1 - FEE)
                trade_log.append({
                    'pnl_pct': cur_pnl,
                    'hold_h': hold_since_entry,
                    'exit_reason': exit_reason,
                    'regime': s.get('regime', 'bull'),
                })
                held = 0; in_pos = False; hold_since_entry = 0
                consecutive_sells = 0
        elif in_pos:
            # Non-SELL signal: reset consecutive counter (signal stream broken)
            consecutive_sells = 0

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        cash = held * sigs[-1]['close'] * (1 - FEE)
        trade_log.append({
            'pnl_pct': (sigs[-1]['close'] / entry_px - 1) * 100,
            'hold_h': hold_since_entry, 'exit_reason': 'OPEN_AT_END',
            'regime': sigs[-1].get('regime', 'bull')})

    ret = (cash / 1000.0 - 1) * 100
    n = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    wr = (wins / n * 100) if n else 0
    n_mh = sum(1 for t in trade_log if t['exit_reason'] == 'MAX_HOLD')
    mh_pnl = sum(t['pnl_pct'] for t in trade_log if t['exit_reason'] == 'MAX_HOLD')
    avg_hold = np.mean([t['hold_h'] for t in trade_log]) if trade_log else 0
    return ret, n, wr, n_mh, mh_pnl, avg_hold


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    sigs = load_sigs()
    win = window_slice(sigs, 60)
    print(f"60d window: {len(win)} signals\n")

    # Baseline (k=1, min_hold=0 — current behavior)
    base_ret, base_n, base_wr, base_mh, base_mhp, base_ah = sim_with_filters(
        win, asset_cfg, k_consecutive=1, min_hold_h=0)
    print(f"BASELINE (k=1, min_hold=0): {base_ret:+.2f}% | {base_n} trades | "
          f"WR {base_wr:.0f}% | MAX_HOLD {base_mh} fires (sum {base_mhp:+.2f}%) | "
          f"avg hold {base_ah:.1f}h\n")

    print("=" * 110)
    print("  SWEEP: k_consecutive (require K consecutive SELLs) X min_hold_h")
    print("=" * 110)
    print()

    rows = []
    for k, mh in product([1, 2, 3, 4], [0, 2, 4, 6, 8]):
        ret, n, wr, n_mh, mhp, ah = sim_with_filters(
            win, asset_cfg, k_consecutive=k, min_hold_h=mh)
        rows.append({
            'k_consecutive': k, 'min_hold_h': mh,
            'return_pct': round(ret, 2),
            'delta_vs_base': round(ret - base_ret, 2),
            'n_trades': n, 'win_rate': round(wr, 1),
            'max_hold_fires': n_mh,
            'max_hold_pnl_sum': round(mhp, 2),
            'avg_hold_h': round(ah, 1),
        })

    df = pd.DataFrame(rows).sort_values('return_pct', ascending=False)
    print(df.to_string(index=False))

    out = os.path.join(ENGINE, 'output',
                       f't1a_consecutive_sell_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nFull sweep saved to: {out}")


if __name__ == '__main__':
    main()
