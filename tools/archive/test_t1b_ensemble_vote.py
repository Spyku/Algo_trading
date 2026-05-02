"""
test_t1b_ensemble_vote.py — TRUE multi-horizon ensemble vote (replaces T1b PROXY).

The earlier T1b PROXY (test_t1b_proxy_conf_threshold.py) used a confidence-
threshold sweep as a stand-in for "require multi-horizon agreement before
trading". This test does the real thing using the freshly-generated
per-horizon signal stream (data/eth_per_horizon_signals_90d.pkl, made by
tools/gen_per_horizon_signals.py — h=5,6,7,8).

Vote rules tested per (asset bar, regime):
  - K_BUY  ∈ {1,2,3,4} horizons must say BUY  with conf >= conf_thr
  - K_SELL ∈ {1,2,3,4} horizons must say SELL with conf >= conf_thr
  - SUBSET ∈ {all, just 5+6, just 6+7, just 7+8, just 5+8}
  - Bull conf threshold ∈ {80, 85, 90}
  - Bear conf threshold ∈ {65, 75, 85}

Anchor horizon for execution = current production bull/bear (6h bull, 7h bear).
The other horizons supply votes only.

Reads:
  - data/eth_per_horizon_signals_90d.pkl  (per-horizon signal cache)
  - config/regime_config_ed.json          (for shield/max_hold/gates)

No production touched. Writes output/t1b_ensemble_vote_<timestamp>.csv.
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
PER_HORIZON_PKL = os.path.join(ENGINE, 'data', 'eth_per_horizon_signals_90d.pkl')
SINGLE_CACHE_PKL = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
REGIME_CFG = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
FEE = 0.0005


def load_per_horizon():
    with open(PER_HORIZON_PKL, 'rb') as f:
        data = pickle.load(f)
    # Index each horizon's signals by datetime for fast lookup
    indexed = {}
    for h, sigs in data.items():
        d = {}
        for s in sigs:
            dt = pd.Timestamp(s['datetime'])
            d[dt] = s
        indexed[h] = d
    return indexed


def load_baseline_sigs():
    """Single-stream signals from production (used for regime + execution)."""
    with open(SINGLE_CACHE_PKL, 'rb') as f:
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


def sim_baseline(sigs, asset_cfg, bull_conf=None):
    """Reference: production single-stream baseline."""
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
    return ret, n, (wins / n * 100 if n else 0)


def vote_signal(per_h, dt, horizons_subset, vote_conf_thr):
    """Tally votes from horizons_subset at this datetime."""
    buy_votes = 0
    sell_votes = 0
    available = 0
    for h in horizons_subset:
        s = per_h[h].get(dt)
        if s is None:
            continue
        available += 1
        sig = s['signal']
        conf = float(s.get('confidence', 0))
        if conf < vote_conf_thr:
            continue
        if sig == 'BUY':
            buy_votes += 1
        elif sig == 'SELL':
            sell_votes += 1
    return buy_votes, sell_votes, available


def sim_vote(sigs, per_h, asset_cfg,
             horizons_subset, k_buy, k_sell, vote_conf_thr,
             bull_conf=None, bear_conf=None):
    """Use per-horizon ensemble vote for entry/exit decisions.
    Anchor execution price = baseline cache's close at each tick.
    Vote requires k_buy / k_sell horizons in subset to agree at conf >= thr.
    """
    bull_conf_thr = bull_conf if bull_conf is not None else vote_conf_thr
    bear_conf_thr = bear_conf if bear_conf is not None else vote_conf_thr
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
    bars_with_no_votes = 0

    for i, s in enumerate(sigs):
        price = float(s['close'])
        regime = s.get('regime', 'bull')
        dt = s['datetime']

        # per-regime conf thr applied to vote check
        thr_for_regime = bull_conf_thr if regime == 'bull' else bear_conf_thr

        # collect ensemble votes
        buy_v, sell_v, avail = vote_signal(per_h, dt, horizons_subset, thr_for_regime)
        if avail == 0:
            bars_with_no_votes += 1

        # effective signal
        if buy_v >= k_buy and buy_v >= sell_v:
            ens_sig = 'BUY'
        elif sell_v >= k_sell and sell_v > buy_v:
            ens_sig = 'SELL'
        else:
            ens_sig = 'HOLD'

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

        if ens_sig == 'BUY' and not in_pos and active_cd <= 0:
            held = cash * (1 - FEE) / price
            cash = 0; in_pos = True; entry_px = price; hold_since_entry = 0
        elif ens_sig == 'SELL' and in_pos:
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
    return ret, n, (wins / n * 100 if n else 0), bars_with_no_votes


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    print("Loading...")
    per_h = load_per_horizon()
    base_sigs = load_baseline_sigs()
    print(f"  per-horizon: {sorted(per_h.keys())}, sizes: {[len(per_h[h]) for h in sorted(per_h.keys())]}")
    print(f"  baseline cache: {len(base_sigs)} sigs, {base_sigs[0]['datetime']} - {base_sigs[-1]['datetime']}")

    windows = {'30d': window_slice(base_sigs, 30),
               '60d': window_slice(base_sigs, 60),
               '90d': base_sigs}

    # Subsets to test
    subsets = {
        'all_5678': (5, 6, 7, 8),
        '56_only':  (5, 6),
        '67_only':  (6, 7),
        '78_only':  (7, 8),
        '58_only':  (5, 8),
        '567_only': (5, 6, 7),
        '678_only': (6, 7, 8),
    }

    rows = []
    for w_name, w_sigs in windows.items():
        print(f"\n[{w_name}: {len(w_sigs)} sigs]")
        for conf in [80, 90]:
            base_ret, base_n, base_wr = sim_baseline(w_sigs, asset_cfg, bull_conf=conf)
            print(f"  baseline_{conf}: {base_ret:+.2f}% / {base_n} tr / WR {base_wr:.0f}%")

            for sub_name, sub_h in subsets.items():
                max_k = len(sub_h)
                for k_buy in range(1, max_k + 1):
                    for k_sell in range(1, max_k + 1):
                        for vote_thr in [70, 80, 85, 90]:
                            ret, n, wr, no_v = sim_vote(
                                w_sigs, per_h, asset_cfg,
                                horizons_subset=sub_h,
                                k_buy=k_buy, k_sell=k_sell,
                                vote_conf_thr=vote_thr,
                                bull_conf=conf, bear_conf=vote_thr)
                            rows.append({
                                'window': w_name,
                                'config': f'{sub_name} k_buy={k_buy} k_sell={k_sell} thr={vote_thr} (base_conf={conf})',
                                'subset': sub_name,
                                'k_buy': k_buy,
                                'k_sell': k_sell,
                                'vote_thr': vote_thr,
                                'base_conf': conf,
                                'return_pct': round(ret, 2),
                                'delta': round(ret - base_ret, 2),
                                'n_trades': n,
                                'win_rate': round(wr, 1),
                                'bars_no_votes': no_v,
                            })

    df = pd.DataFrame(rows)
    pivot = df.pivot(index='config', columns='window', values='delta')
    pivot.columns = [f'd_{c}' for c in pivot.columns]
    pivot_n = df.pivot(index='config', columns='window', values='n_trades')
    pivot_n.columns = [f'n_{c}' for c in pivot_n.columns]
    full = pd.concat([pivot, pivot_n], axis=1).sort_values('d_60d', ascending=False)

    print(f"\n{'='*130}")
    print(f"  T1b ENSEMBLE VOTE — sorted by 60d delta vs same-base-conf baseline")
    print(f"{'='*130}")
    print(full.head(30).to_string())

    print(f"\n{'='*130}")
    print(f"  POSITIVE ON ALL 3 WINDOWS")
    print(f"{'='*130}")
    pos = full[(full['d_30d'] > 0) & (full['d_60d'] > 0) & (full['d_90d'] > 0)]
    print(pos.head(30).to_string() if not pos.empty else "  None.")

    print(f"\n{'='*130}")
    print(f"  POSITIVE ON 60d AND 90d (most relevant for current regime)")
    print(f"{'='*130}")
    pos60_90 = full[(full['d_60d'] > 0) & (full['d_90d'] > 0)]
    print(pos60_90.head(20).to_string() if not pos60_90.empty else "  None.")

    out = os.path.join(ENGINE, 'output',
                       f't1b_ensemble_vote_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
