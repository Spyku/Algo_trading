"""
test_t4_batch.py — four genuinely new ideas not yet tested.

T4a. MULTI-BAR BUY CONSENSUS (entry-side of T1a)
   T1a tested K consecutive SELLs. Now test K consecutive BUYs:
   require K consecutive BUY signals at conf >= threshold before entering.
   Hypothesis: "wait for confirmation" before entry.

T4b. ANTI-TILT FILTER (post-loss cool-off)
   Pause BUYs for N hours after a loss (PnL <= 0%). Anti-recency / anti-tilt.

T4c. REGIME-CHANGE-ONLY ENTRY
   Block BUY unless the regime has been "new" within last K bars (i.e., a
   regime transition happened recently). Captures the "fresh regime
   opportunity" idea.

T4d. BEAR CONFIDENCE THRESHOLD SWEEP
   T1b raised bull conf to 90% (winner). Now sweep BEAR conf threshold.
   Current production: bull=80, bear=85. Test bear in {65, 70, 75, 80, 85, 90, 95}.

Reads: cache + regime_config (read-only). Writes: output/t4_batch.csv
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


def sim(sigs, asset_cfg,
        k_consecutive_buy=1,
        post_loss_cooloff_h=0,
        regime_fresh_within=None,
        bull_conf=None, bear_conf=None):
    """All filters None/0 = no-op (matches baseline)."""
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
    consecutive_buys = 0
    last_loss_idx = -10**9
    prev_regime = None
    last_regime_change_idx = -10**9
    trade_log = []
    blocked = 0

    for i, s in enumerate(sigs):
        price = float(s['close'])
        regime = s.get('regime', 'bull')
        sig = s['signal']
        sconf = float(s.get('confidence', 0))
        conf_thr = bull_conf_thr if regime == 'bull' else bear_conf_thr
        shield_on = bull_shield if regime == 'bull' else bear_shield

        if regime != prev_regime:
            last_regime_change_idx = i
            prev_regime = regime

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

        # Track consecutive BUY signals (above conf threshold) regardless of in_pos
        if sig == 'BUY' and sconf >= conf_thr:
            consecutive_buys += 1
        else:
            consecutive_buys = 0

        if sig == 'BUY' and sconf >= conf_thr and not in_pos and active_cd <= 0:
            block = False
            if k_consecutive_buy > 1 and consecutive_buys < k_consecutive_buy:
                block = True
            if not block and post_loss_cooloff_h > 0:
                if (i - last_loss_idx) < post_loss_cooloff_h:
                    block = True
            if not block and regime_fresh_within is not None:
                if (i - last_regime_change_idx) > regime_fresh_within:
                    block = True
            if block:
                blocked += 1
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
                trade_log.append({'pnl_pct': cur_pnl, 'regime': regime})
                if cur_pnl <= 0:
                    last_loss_idx = i
                held = 0; in_pos = False; hold_since_entry = 0

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        cash = held * sigs[-1]['close'] * (1 - FEE)
        cur_pnl = (sigs[-1]['close'] / entry_px - 1) * 100
        trade_log.append({'pnl_pct': cur_pnl, 'regime': sigs[-1].get('regime', 'bull')})

    ret = (cash / 1000.0 - 1) * 100
    n = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    wr = (wins / n * 100) if n else 0
    return ret, n, wr, blocked


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    sigs = load_sigs()
    windows = {'30d': window_slice(sigs, 30), '60d': window_slice(sigs, 60),
               '90d': sigs}

    rows = []
    for w_name, w_sigs in windows.items():
        base, base_n, base_wr, _ = sim(w_sigs, asset_cfg)

        tests = []
        # T4a multi-bar BUY consensus
        for k in [2, 3, 4]:
            tests.append((f'T4a buy_consensus k={k}', dict(k_consecutive_buy=k)))
        # T4b anti-tilt
        for h in [4, 8, 12, 24, 48]:
            tests.append((f'T4b post_loss_cool={h}h', dict(post_loss_cooloff_h=h)))
        # T4c regime-fresh
        for w in [4, 8, 12, 24]:
            tests.append((f'T4c regime_fresh<={w}h', dict(regime_fresh_within=w)))
        # T4d bear conf sweep
        for bc in [60, 65, 70, 75, 80, 85, 90, 95]:
            tests.append((f'T4d bear_conf={bc}', dict(bear_conf=bc)))
        # Combine winner from each
        tests.append(('combo: bull_conf=90 + bear_conf=80',
                      dict(bull_conf=90, bear_conf=80)))
        tests.append(('combo: bull_conf=90 + bear_conf=70',
                      dict(bull_conf=90, bear_conf=70)))
        tests.append(('combo: bull_conf=90 + post_loss=8h',
                      dict(bull_conf=90, post_loss_cooloff_h=8)))

        for name, kw in tests:
            ret, n, wr, blk = sim(w_sigs, asset_cfg, **kw)
            rows.append({
                'window': w_name, 'test': name,
                'return_pct': round(ret, 2),
                'delta': round(ret - base, 2),
                'n_trades': n, 'win_rate': round(wr, 1),
            })

    df = pd.DataFrame(rows)
    pivot = df.pivot(index='test', columns='window', values='delta').round(2)
    pivot.columns = [f'd_{c}' for c in pivot.columns]
    counts = df.pivot(index='test', columns='window', values='n_trades')
    counts.columns = [f'n_{c}' for c in counts.columns]
    full = pd.concat([pivot, counts], axis=1).sort_values('d_60d', ascending=False)

    print(f"\n{'='*110}")
    print(f"  T4 RESULTS (delta vs baseline 49.34/22.06/36.93)")
    print(f"{'='*110}")
    print(full.to_string())

    # Positive on all 3 windows
    print(f"\n{'='*110}")
    print(f"  CONFIGS POSITIVE ON ALL 3 WINDOWS")
    print(f"{'='*110}")
    pos = full[(full['d_30d'] > 0) & (full['d_60d'] > 0) & (full['d_90d'] > 0)]
    if pos.empty:
        print("  None.")
    else:
        print(pos.to_string())

    out = os.path.join(ENGINE, 'output',
                       f't4_batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved to: {out}")


if __name__ == '__main__':
    main()
