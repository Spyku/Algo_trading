"""
validate_t1a_extended.py — extended-window validation of the T1a winner.

Runs the smart-exit logic from smart_exit.py against:
  - Full 90d cache (vs the 60d window the original sweep used)
  - Multiple (k_consecutive, min_hold_hours) configurations near the winner
  - Per-window decomposition (60d / 90d / first half / second half)

Goal: confirm whether the T1a +6.34pp win on 60d holds on a longer window
and isn't an artifact of the specific 60d window chosen.

Reads (read-only):
  data/eth_sl_signals_90d.pkl
  config/regime_config_ed.json

Writes:
  output/t1a_validation_<timestamp>.csv

Production untouched.
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
import sys

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smart_exit import SmartExitState

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


def slice_by_days(sigs, days):
    end = sigs[-1]['datetime']
    lo = end - pd.Timedelta(days=days)
    return [s for s in sigs if s['datetime'] >= lo]


def slice_first_half(sigs):
    n = len(sigs)
    return sigs[:n // 2]


def slice_second_half(sigs):
    n = len(sigs)
    return sigs[n // 2:]


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


def sim(sigs, asset_cfg, k_consecutive=1, min_hold_hours=0):
    """Standard regime-switched simulator with the smart-exit filters layered
    in. k_consecutive=1, min_hold_hours=0 reproduces the canonical baseline.
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
    state = SmartExitState(k_consecutive=k_consecutive, min_hold_hours=min_hold_hours)
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
            state.observe_signal(sig, sconf, conf_thr)

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
            state.on_entry()
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = shield_on and min_pnl > 0 and cur_pnl < min_pnl
            smart_blocks = state.should_block_sell(sig, sconf, conf_thr, hold_since_entry)

            # max_hold ALWAYS overrides smart_exit (failsafe is failsafe)
            # smart_exit applies BEFORE shield (so it can suppress noise even
            # at high PnL where shield wouldn't block)
            if override_expired:
                # Failsafe path — fire regardless of smart_exit
                if (not shield_blocks) or override_expired:
                    exit_reason = 'MAX_HOLD' if (shield_on and cur_pnl < min_pnl) else 'MODEL_AT_MAX'
                    cash = held * price * (1 - FEE)
                    trade_log.append({'pnl_pct': cur_pnl, 'exit_reason': exit_reason})
                    held = 0; in_pos = False; hold_since_entry = 0
            elif not smart_blocks and not shield_blocks:
                # Normal model exit
                cash = held * price * (1 - FEE)
                trade_log.append({'pnl_pct': cur_pnl, 'exit_reason': 'MODEL'})
                held = 0; in_pos = False; hold_since_entry = 0

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        cash = held * sigs[-1]['close'] * (1 - FEE)
        trade_log.append({'pnl_pct': (sigs[-1]['close'] / entry_px - 1) * 100,
                          'exit_reason': 'OPEN_AT_END'})

    ret = (cash / 1000.0 - 1) * 100
    n = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    wr = (wins / n * 100) if n else 0
    n_mh = sum(1 for t in trade_log if t['exit_reason'] == 'MAX_HOLD')
    mh_pnl = sum(t['pnl_pct'] for t in trade_log if t['exit_reason'] == 'MAX_HOLD')
    return ret, n, wr, n_mh, mh_pnl


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    sigs = load_sigs()
    print(f"Cache: {len(sigs)} signals, "
          f"{sigs[0]['datetime']} -> {sigs[-1]['datetime']} "
          f"({(sigs[-1]['datetime'] - sigs[0]['datetime']).days}d span)\n")

    windows = {
        'last_30d': slice_by_days(sigs, 30),
        'last_60d': slice_by_days(sigs, 60),
        'full_90d': sigs,
        'first_half': slice_first_half(sigs),
        'second_half': slice_second_half(sigs),
    }

    # Configs to compare. Winner from 60d sweep is k=2, mh=4.
    # Test neighboring values to map robustness.
    configs = [
        ('baseline (k=1, mh=0)', 1, 0),
        ('k=2, mh=0',            2, 0),
        ('k=2, mh=2',            2, 2),
        ('WINNER (k=2, mh=4)',   2, 4),
        ('k=2, mh=6',            2, 6),
        ('k=1, mh=4',            1, 4),
        ('k=3, mh=0',            3, 0),
        ('k=3, mh=4',            3, 4),
    ]

    rows = []
    for win_name, win_sigs in windows.items():
        if not win_sigs:
            continue
        # Compute baseline first for this window
        base_ret, _, _, _, _ = sim(win_sigs, asset_cfg, k_consecutive=1, min_hold_hours=0)
        for cfg_name, k, mh in configs:
            ret, n, wr, n_mh, mhp = sim(win_sigs, asset_cfg,
                                         k_consecutive=k, min_hold_hours=mh)
            rows.append({
                'window': win_name,
                'config': cfg_name,
                'k': k, 'min_hold_h': mh,
                'return_pct': round(ret, 2),
                'delta_vs_base': round(ret - base_ret, 2),
                'n_trades': n,
                'win_rate': round(wr, 1),
                'max_hold_fires': n_mh,
                'max_hold_pnl_sum': round(mhp, 2),
            })

    df = pd.DataFrame(rows)

    print("=" * 130)
    print("  T1a EXTENDED VALIDATION: smart-exit (k_consecutive, min_hold_hours)")
    print("  across multiple windows. Each row's delta is vs baseline IN THAT WINDOW.")
    print("=" * 130)
    for win_name in windows.keys():
        sub = df[df['window'] == win_name].copy()
        if sub.empty:
            continue
        print(f"\n[{win_name.upper()}]  n_signals={len(windows[win_name])}")
        print(sub.drop(columns=['window']).to_string(index=False))

    # Pivot: WINNER config across all windows
    print("\n" + "=" * 130)
    print("  WINNER (k=2, mh=4) ACROSS ALL WINDOWS")
    print("=" * 130)
    win_only = df[df['config'] == 'WINNER (k=2, mh=4)']
    print(win_only[['window', 'return_pct', 'delta_vs_base', 'n_trades',
                    'win_rate', 'max_hold_fires']].to_string(index=False))

    out = os.path.join(ENGINE, 'output',
                       f't1a_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nFull validation saved to: {out}")

    # Decision summary
    print("\n" + "=" * 130)
    print("  DECISION SUMMARY")
    print("=" * 130)
    deltas = win_only.set_index('window')['delta_vs_base'].to_dict()
    if all(deltas.get(w, -999) >= 3 for w in ['last_30d', 'last_60d', 'full_90d']):
        print(f"  T1a robust across 30/60/90 day windows. Recommend promotion.")
    elif deltas.get('full_90d', -999) >= 3 and deltas.get('last_60d', -999) >= 3:
        print(f"  T1a robust on 60d and 90d. Validate further before promotion.")
    elif any(deltas.get(w, -999) < 0 for w in ['last_30d', 'last_60d', 'full_90d']):
        print(f"  T1a NEGATIVE on at least one window. Likely overfitting — DO NOT PROMOTE.")
    else:
        print(f"  T1a marginal across windows. Reconsider.")
    print(f"  Per-window deltas: {deltas}")


if __name__ == '__main__':
    main()
