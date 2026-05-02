"""
test_t6_triple_barrier.py — triple-barrier exit logic OVERLAY on cached signals.

Lopez de Prado Ch. 3 framing: instead of holding to a fixed-horizon SELL or
shield+max_hold timeout, exit on the FIRST of three barriers:
  - upper barrier  (profit target, vol-adaptive)
  - lower barrier  (stop-loss, vol-adaptive)
  - vertical (time horizon)

The vol-adaptive barriers use 24h realized stdev × multiplier. We sweep
upper_mult, lower_mult, vertical_h, comparing to baseline (model-driven exit
+ shield + max_hold). Reads cache + eth_hourly. No production touched.

Note: this PARTIALLY ignores model SELL signals while in position — the
triple barrier replaces the SELL+shield+max_hold exit logic. Entry still
follows model BUY at conf >= threshold.

Reads: cache + regime_config + eth_hourly. Writes output/t6_triple_barrier_*.csv
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


def precompute_vol(sigs, eth_df):
    """24h realized stdev of hourly returns at each signal's timestamp."""
    n = len(sigs)
    vol_24h = np.full(n, np.nan)  # std as decimal (e.g., 0.01 = 1%)
    eth_close = eth_df['close']
    for i, s in enumerate(sigs):
        dt = s['datetime']
        if dt.tzinfo is None: dt = dt.tz_localize('UTC')
        else: dt = dt.tz_convert('UTC')
        sub = eth_close[(eth_close.index < dt) & (eth_close.index >= dt - timedelta(hours=24))]
        if len(sub) >= 12:
            rets = sub.pct_change().dropna()
            vol_24h[i] = rets.std()
    return vol_24h


def sim_baseline(sigs, asset_cfg, bull_conf=None, bear_conf=None):
    """Reference: model-driven SELL + shield + max_hold."""
    bull_conf_thr = bull_conf if bull_conf is not None else float(asset_cfg.get('bull', {}).get('min_confidence', 80))
    bear_conf_thr = bear_conf if bear_conf is not None else float(asset_cfg.get('bear', {}).get('min_confidence', 65))
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
            held = cash * (1 - FEE) / price
            cash = 0; in_pos = True; entry_px = price; hold_since_entry = 0
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = shield_on and min_pnl > 0 and cur_pnl < min_pnl
            if not shield_blocks or override_expired:
                cash = held * price * (1 - FEE)
                trade_log.append({'pnl_pct': cur_pnl, 'exit': 'MODEL'})
                held = 0; in_pos = False; hold_since_entry = 0

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        cash = held * sigs[-1]['close'] * (1 - FEE)
        trade_log.append({'pnl_pct': (sigs[-1]['close'] / entry_px - 1) * 100, 'exit': 'OPEN'})
    ret = (cash / 1000.0 - 1) * 100
    n = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    return ret, n, (wins / n * 100 if n else 0)


def sim_triple_barrier(sigs, asset_cfg, vol_arr,
                       upper_mult, lower_mult, vertical_h,
                       bull_conf=None):
    """Replace SELL/shield/max_hold with triple barrier:
       upper = entry × (1 + upper_mult × vol_24h)
       lower = entry × (1 - lower_mult × vol_24h)
       vertical = vertical_h hours of holding."""
    bull_conf_thr = bull_conf if bull_conf is not None else float(asset_cfg.get('bull', {}).get('min_confidence', 80))
    bear_conf_thr = float(asset_cfg.get('bear', {}).get('min_confidence', 65))
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
    upper_px = lower_px = 0.0; vertical_end = 0
    bull_cd = bear_cd = 0
    trade_log = []
    upper_fires = lower_fires = vertical_fires = 0

    for i, s in enumerate(sigs):
        price = float(s['close'])
        regime = s.get('regime', 'bull')
        sig = s['signal']
        sconf = float(s.get('confidence', 0))
        conf_thr = bull_conf_thr if regime == 'bull' else bear_conf_thr

        # cooldown updates
        if bcd_h > 0 and bull_rs is not None:
            rs = bull_rs[i] if not np.isnan(bull_rs[i]) else 0
            rl = bull_rl[i] if not np.isnan(bull_rl[i]) else 0
            if rs >= bt_s or rl >= bt_l: bull_cd = max(bull_cd, bcd_h)
        if rcd_h > 0 and bear_rs is not None:
            rs = bear_rs[i] if not np.isnan(bear_rs[i]) else 0
            rl = bear_rl[i] if not np.isnan(bear_rl[i]) else 0
            if rs >= rt_s or rl >= rt_l: bear_cd = max(bear_cd, rcd_h)
        active_cd = bull_cd if regime == 'bull' else bear_cd

        # Check exits while in position
        if in_pos:
            exit_reason = None
            if price >= upper_px:
                exit_reason = 'UPPER'; upper_fires += 1
            elif price <= lower_px:
                exit_reason = 'LOWER'; lower_fires += 1
            elif i >= vertical_end:
                exit_reason = 'VERTICAL'; vertical_fires += 1
            if exit_reason:
                cur_pnl = (price / entry_px - 1) * 100
                cash = held * price * (1 - FEE)
                trade_log.append({'pnl_pct': cur_pnl, 'exit': exit_reason, 'regime': regime})
                held = 0; in_pos = False
                if bull_cd > 0: bull_cd -= 1
                if bear_cd > 0: bear_cd -= 1
                continue

        if sig == 'BUY' and sconf >= conf_thr and not in_pos and active_cd <= 0:
            v = vol_arr[i]
            if np.isnan(v) or v <= 0:
                v = 0.01  # 1% fallback
            held = cash * (1 - FEE) / price
            cash = 0; in_pos = True; entry_px = price
            upper_px = entry_px * (1 + upper_mult * v)
            lower_px = entry_px * (1 - lower_mult * v)
            vertical_end = i + vertical_h

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        cash = held * sigs[-1]['close'] * (1 - FEE)
        trade_log.append({'pnl_pct': (sigs[-1]['close'] / entry_px - 1) * 100,
                          'exit': 'OPEN', 'regime': 'bull'})
    ret = (cash / 1000.0 - 1) * 100
    n = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    wr = (wins / n * 100) if n else 0
    return ret, n, wr, upper_fires, lower_fires, vertical_fires


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    print("Loading...")
    sigs = load_sigs()
    eth = pd.read_csv(ETH_HOURLY)
    eth['datetime'] = pd.to_datetime(eth['datetime'], utc=True)
    eth = eth.set_index('datetime').sort_index()

    windows = {'30d': window_slice(sigs, 30),
               '60d': window_slice(sigs, 60),
               '90d': sigs}

    rows = []
    for w_name, w_sigs in windows.items():
        print(f"\n[{w_name}: {len(w_sigs)} sigs]")
        vol = precompute_vol(w_sigs, eth)

        # Baselines: prod (bull_conf=80) and bull_conf=90
        b80, n80, wr80 = sim_baseline(w_sigs, asset_cfg, bull_conf=80)
        b90, n90, wr90 = sim_baseline(w_sigs, asset_cfg, bull_conf=90)
        print(f"  baseline_80: {b80:+.2f}% / {n80} tr / WR {wr80:.0f}%")
        print(f"  baseline_90: {b90:+.2f}% / {n90} tr / WR {wr90:.0f}%")

        # TB sweep
        configs = []
        for um in [2, 3, 4, 5, 6]:
            for lm in [1, 2, 3, 4]:
                for v in [12, 24, 48]:
                    configs.append((um, lm, v))

        for um, lm, v in configs:
            for conf in [80, 90]:
                ret, n, wr, uf, lf, vf = sim_triple_barrier(
                    w_sigs, asset_cfg, vol, um, lm, v, bull_conf=conf)
                base = b80 if conf == 80 else b90
                rows.append({
                    'window': w_name,
                    'config': f'TB up={um}s lo={lm}s vert={v}h conf={conf}',
                    'return_pct': round(ret, 2),
                    'delta_vs_base': round(ret - base, 2),
                    'n_trades': n,
                    'win_rate': round(wr, 1),
                    'upper_fires': uf,
                    'lower_fires': lf,
                    'vertical_fires': vf,
                })

    df = pd.DataFrame(rows)
    pivot = df.pivot(index='config', columns='window', values='delta_vs_base')
    pivot.columns = [f'd_{c}' for c in pivot.columns]
    pivot_n = df.pivot(index='config', columns='window', values='n_trades')
    pivot_n.columns = [f'n_{c}' for c in pivot_n.columns]
    full = pd.concat([pivot, pivot_n], axis=1).sort_values('d_60d', ascending=False)

    print(f"\n{'='*120}")
    print(f"  T6 TRIPLE BARRIER — sorted by 60d delta vs same-conf baseline")
    print(f"{'='*120}")
    print(full.head(25).to_string())

    print(f"\n{'='*120}")
    print(f"  POSITIVE ON ALL 3 WINDOWS")
    print(f"{'='*120}")
    pos = full[(full['d_30d'] > 0) & (full['d_60d'] > 0) & (full['d_90d'] > 0)]
    if pos.empty:
        print("  None.")
    else:
        print(pos.head(25).to_string())

    out = os.path.join(ENGINE, 'output',
                       f't6_triple_barrier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
