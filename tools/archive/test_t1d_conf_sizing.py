"""
T1d — Confidence-based position sizing.

Hypothesis: high-conviction signals should get larger positions; barely-above-
threshold signals should get smaller. Currently the trader is binary (all-in
at fixed $12k regardless of confidence).

Sizing rules tested:
  R0. Binary (baseline, current behavior)
  R1. Linear:    size = full × (conf - thr) / (100 - thr), clipped to [min, 1.0]
  R2. Step:      conf 80-89 → 50% size, 90-94 → 80% size, 95+ → 100% size
  R3. Power 2:   size = full × ((conf - thr) / (100 - thr))^2  (heavily weighted to high conf)
  R4. Reverse-binary: skip if conf < 90% (effectively raises threshold)

Metric: total return after compounding sized positions, vs binary baseline.
A win means: smaller average loss on weaker signals + similar wins on strong ones.
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


def sim_with_sizing(sigs, asset_cfg, sizing_fn, min_size_frac=0.3):
    """sizing_fn(conf, conf_thr) -> fraction of cash to deploy on BUY (in [0,1]).
    The remaining cash is held in reserve and not invested.

    The reserve does NOT compound (held flat). PnL on the active fraction
    determines the trade contribution. After SELL, reserve is added back to
    cash and full bankroll is available next BUY.

    Returns (return_pct, n_trades, win_rate, avg_size_used).
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

    bankroll = 1000.0
    reserve = 0.0       # cash held back when sizing < 1.0
    held = 0.0
    in_pos = False
    entry_px = 0.0
    hold_since_entry = 0
    bull_cd = bear_cd = 0
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
            size_frac = sizing_fn(sconf, conf_thr)
            size_frac = max(min_size_frac, min(1.0, size_frac))
            if size_frac <= 0:
                continue  # skip
            cash_total = bankroll  # all cash is in `bankroll` between trades
            invested = cash_total * size_frac
            reserve = cash_total - invested
            held = invested * (1 - FEE) / price
            bankroll = 0.0
            in_pos = True
            entry_px = price
            hold_since_entry = 0
            trade_log.append({'size_frac': size_frac, 'pnl_pct': None})
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = shield_on and min_pnl > 0 and cur_pnl < min_pnl
            if not shield_blocks or override_expired:
                proceeds = held * price * (1 - FEE)
                bankroll = proceeds + reserve
                reserve = 0.0
                held = 0
                in_pos = False
                hold_since_entry = 0
                trade_log[-1]['pnl_pct'] = cur_pnl

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        bankroll = held * sigs[-1]['close'] * (1 - FEE) + reserve
        trade_log[-1]['pnl_pct'] = (sigs[-1]['close'] / entry_px - 1) * 100

    ret = (bankroll / 1000.0 - 1) * 100
    closed = [t for t in trade_log if t['pnl_pct'] is not None]
    n = len(closed)
    wins = sum(1 for t in closed if t['pnl_pct'] > 0)
    wr = (wins / n * 100) if n else 0
    avg_size = np.mean([t['size_frac'] for t in closed]) if closed else 0
    return ret, n, wr, avg_size


# ---- Sizing functions ----
def binary(conf, thr): return 1.0
def linear(conf, thr): return (conf - thr) / max(1, 100 - thr)
def linear_min50(conf, thr): return max(0.5, (conf - thr) / max(1, 100 - thr))
def step(conf, thr):
    if conf >= 95: return 1.0
    if conf >= 90: return 0.8
    if conf >= 85: return 0.6
    return 0.5
def power2(conf, thr):
    x = (conf - thr) / max(1, 100 - thr)
    return x * x
def power_half(conf, thr):
    x = (conf - thr) / max(1, 100 - thr)
    return x ** 0.5  # sqrt: weights MORE toward weaker signals (compresses range)


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    sigs = load_sigs()
    win = window_slice(sigs, 60)
    print(f"60d window: {len(win)} signals\n")

    # Show entry-confidence distribution at the cache's signals
    confs = [float(s.get('confidence', 0)) for s in win
             if s['signal'] == 'BUY' and s.get('confidence', 0) >=
                (asset_cfg['bull']['min_confidence'] if s.get('regime') == 'bull'
                 else asset_cfg['bear']['min_confidence'])]
    if confs:
        print(f"Entry-confidence distribution: median={np.median(confs):.1f} "
              f"mean={np.mean(confs):.1f} min={min(confs):.1f} max={max(confs):.1f} n={len(confs)}\n")

    sizing_fns = [
        ('R0_binary',       binary,         0.0),
        ('R1_linear_min30', linear,         0.3),
        ('R1_linear_min50', linear,         0.5),
        ('R2_step',         step,           0.5),
        ('R3_power2',       power2,         0.3),
        ('R4_power_half',   power_half,     0.3),
    ]

    rows = []
    for name, fn, min_size in sizing_fns:
        ret, n, wr, avg_size = sim_with_sizing(win, asset_cfg, fn, min_size_frac=min_size)
        rows.append({
            'rule': name, 'min_size': min_size,
            'return_pct': round(ret, 2),
            'n_trades': n, 'win_rate': round(wr, 1),
            'avg_size_frac': round(avg_size, 2),
        })

    df = pd.DataFrame(rows)
    base_ret = df[df['rule'] == 'R0_binary']['return_pct'].iloc[0]
    df['delta_vs_base'] = df['return_pct'] - base_ret

    print("=" * 100)
    print("  T1d: CONFIDENCE-BASED POSITION SIZING")
    print("=" * 100)
    print(df.to_string(index=False))

    out = os.path.join(ENGINE, 'output',
                       f't1d_conf_sizing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved to: {out}")


if __name__ == '__main__':
    main()
