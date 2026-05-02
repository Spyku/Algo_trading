"""
T1c — Time-of-day execution window filter.

Hypothesis: certain UTC hours produce systematically worse trades. Block BUYs
during those hours.

Three sweeps:
  S1. Single-hour exclusion: skip BUYs at hour H, sweep H ∈ [0..23]
  S2. Window exclusion: skip BUYs during a contiguous N-hour window
       (Asian session 22-08, EU session 06-16, US session 13-22)
  S3. Per-hour PnL profile: just report which hours produced winners vs losers

All standalone, read-only on production.
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime

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


def sim(sigs, asset_cfg, blocked_hours=None):
    """Standard regime-switched sim with optional set of UTC hours where
    BUY signals are blocked."""
    if blocked_hours is None:
        blocked_hours = set()

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
        utc_hour = s['datetime'].hour

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
            if utc_hour in blocked_hours:
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
                cash = held * price * (1 - FEE)
                trade_log.append({
                    'pnl_pct': cur_pnl,
                    'hold_h': hold_since_entry,
                    'entry_hour': None,  # filled when computing per-hour profile
                    'regime': regime,
                })
                held = 0; in_pos = False; hold_since_entry = 0

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        cash = held * sigs[-1]['close'] * (1 - FEE)

    ret = (cash / 1000.0 - 1) * 100
    n = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    wr = (wins / n * 100) if n else 0
    return ret, n, wr, blocked_by_filter


def per_hour_pnl_profile(sigs, asset_cfg):
    """Walk the sim once tracking entry hour for each trade."""
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
    entry_hour = None
    hold_since_entry = 0; bull_cd = bear_cd = 0
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
            entry_hour = s['datetime'].hour
            hold_since_entry = 0
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = shield_on and min_pnl > 0 and cur_pnl < min_pnl
            if not shield_blocks or override_expired:
                cash = held * price * (1 - FEE)
                trade_log.append({'pnl_pct': cur_pnl, 'entry_hour': entry_hour})
                held = 0; in_pos = False; hold_since_entry = 0

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    return trade_log


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    sigs = load_sigs()
    win = window_slice(sigs, 60)
    print(f"60d window: {len(win)} signals\n")

    base_ret, base_n, base_wr, _ = sim(win, asset_cfg)
    print(f"BASELINE (no filter): {base_ret:+.2f}% | {base_n} trades | WR {base_wr:.0f}%\n")

    print("=" * 100)
    print("  S3: PER-HOUR PnL PROFILE (live config, no filter)")
    print("=" * 100)
    trades = per_hour_pnl_profile(win, asset_cfg)
    df = pd.DataFrame(trades)
    grp = df.groupby('entry_hour').agg(
        n=('pnl_pct', 'count'),
        wr=('pnl_pct', lambda x: (x > 0).mean() * 100),
        avg_pnl=('pnl_pct', 'mean'),
        sum_pnl=('pnl_pct', 'sum'),
    ).round(2)
    print(grp.to_string())

    print("\n" + "=" * 100)
    print("  S1: SINGLE-HOUR EXCLUSION SWEEP")
    print("=" * 100)
    rows = []
    for h in range(24):
        ret, n_tr, wr, blocked = sim(win, asset_cfg, blocked_hours={h})
        rows.append({
            'blocked_hour': h, 'return_pct': round(ret, 2),
            'delta': round(ret - base_ret, 2),
            'n_trades': n_tr, 'win_rate': round(wr, 1),
            'blocked': blocked,
        })
    s1 = pd.DataFrame(rows).sort_values('return_pct', ascending=False)
    print(s1.to_string(index=False))

    print("\n" + "=" * 100)
    print("  S2: SESSION-WINDOW EXCLUSION")
    print("=" * 100)
    sessions = {
        'Asian (22-07 UTC)':   set(range(22, 24)) | set(range(0, 8)),
        'European (06-16 UTC)': set(range(6, 16)),
        'US (13-22 UTC)':      set(range(13, 22)),
        'Off-hours (00-05 UTC)': set(range(0, 6)),
        'Asian-only narrow (00-05 UTC)': set(range(0, 6)),
        'Late-day (16-22 UTC)': set(range(16, 22)),
    }
    rows = []
    for name, hours in sessions.items():
        ret, n_tr, wr, blocked = sim(win, asset_cfg, blocked_hours=hours)
        rows.append({
            'session': name, 'return_pct': round(ret, 2),
            'delta': round(ret - base_ret, 2),
            'n_trades': n_tr, 'win_rate': round(wr, 1),
            'blocked': blocked,
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 100)
    print("  S2b: BEST 2-HOUR-AT-A-TIME EXCLUSION (top 5 by return)")
    print("=" * 100)
    rows = []
    for h in range(24):
        h2 = (h + 1) % 24
        ret, n_tr, wr, blocked = sim(win, asset_cfg, blocked_hours={h, h2})
        rows.append({
            'block_hours': f'{h:02d}-{h2:02d}', 'return_pct': round(ret, 2),
            'delta': round(ret - base_ret, 2),
            'n_trades': n_tr, 'win_rate': round(wr, 1),
            'blocked': blocked,
        })
    s2b = pd.DataFrame(rows).sort_values('return_pct', ascending=False).head(5)
    print(s2b.to_string(index=False))

    out = os.path.join(ENGINE, 'output',
                       f't1c_time_of_day_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    s1.to_csv(out, index=False)
    print(f"\nSingle-hour sweep saved to: {out}")


if __name__ == '__main__':
    main()
