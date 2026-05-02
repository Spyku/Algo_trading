"""
test_t1b_on_top_of_mix.py — re-test T1b ensemble vote with MIX gate as baseline.

The original T1b test (test_t1b_ensemble_vote.py) was run BEFORE MIX gate
was promoted, so its delta of +19.84pp on 60d was measured against the OLD
PROD bull gate (rr20>=4.0% OR rr24>=4.5% cd=12h).

Now MIX gate is live (rr12>=2.5% OR rr20>=4.0% cd=24h). We need to know:
  - Does T1b STILL add meaningful alpha on top of MIX?
  - Is the marginal gain large enough to justify the implementation cost
    (live trader needs to run all 4 horizon models, not just 2)?
  - Does T1b survive Tier 1 OOS validation (held-out first 30d of cache)?

Reads:
  - data/eth_per_horizon_signals_90d.pkl (per-horizon signal cache, h=5,6,7,8)
  - data/eth_sl_signals_90d.pkl (single-stream signals for baseline)
  - config/regime_config_ed.json (already has MIX gate as the bull rally_cooldown)

Read-only. Writes output/t1b_on_mix_<ts>.csv
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
    indexed = {}
    for h, sigs in data.items():
        d = {}
        for s in sigs:
            dt = pd.Timestamp(s['datetime'])
            d[dt] = s
        indexed[h] = d
    return indexed


def load_baseline_sigs():
    with open(SINGLE_CACHE_PKL, 'rb') as f:
        sigs = pickle.load(f)
    for s in sigs:
        s['datetime'] = pd.Timestamp(s['datetime'])
    sigs.sort(key=lambda s: s['datetime'])
    return sigs


def window_slice(sigs, days_back_from, days_back_to):
    """Slice signals where (now - days_back_to) <= dt < (now - days_back_from)."""
    end = sigs[-1]['datetime']
    lo = end - pd.Timedelta(days=days_back_to)
    hi = end - pd.Timedelta(days=days_back_from)
    return [s for s in sigs if lo <= s['datetime'] <= hi]


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


def vote_signal(per_h, dt, horizons_subset, vote_conf_thr):
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


def sim(sigs, asset_cfg, per_h=None, vote_subset=None,
        k_buy=1, k_sell=1, vote_conf_thr=0,
        bull_conf=None, bear_conf=None):
    """Simulate strategy with MIX gate (read from asset_cfg) PLUS optional T1b ensemble.
    If per_h + vote_subset given: use multi-horizon vote for entry/exit (T1b).
    Otherwise: use the baseline single-stream signal (PROD-with-MIX baseline)."""
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
        dt = s['datetime']

        if per_h is not None and vote_subset is not None:
            thr_for_regime = bull_conf_thr if regime == 'bull' else bear_conf_thr
            buy_v, sell_v, _avail = vote_signal(per_h, dt, vote_subset, vote_conf_thr or thr_for_regime)
            if buy_v >= k_buy and buy_v >= sell_v:
                ens_sig = 'BUY'
            elif sell_v >= k_sell and sell_v > buy_v:
                ens_sig = 'SELL'
            else:
                ens_sig = 'HOLD'
            sig = ens_sig
            sconf = 100.0  # ensemble already filtered by conf threshold
        else:
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


def main():
    print("Loading...")
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']
    bg = asset_cfg['bull']['rally_cooldown']
    print(f"  Bull gate (MIX): rr{bg['h_short']}h>={bg['t_short_pct']}% OR rr{bg['h_long']}h>={bg['t_long_pct']}% cd={bg['cd_hours']}h")
    per_h = load_per_horizon()
    base_sigs = load_baseline_sigs()
    print(f"  per-horizon: {sorted(per_h.keys())}")
    print(f"  baseline cache: {len(base_sigs)} sigs ({base_sigs[0]['datetime']} - {base_sigs[-1]['datetime']})")

    # Define windows — including OOS held-out for Tier 1
    windows = {
        '30d (in-sample)':    window_slice(base_sigs, 0, 30),
        '60d (in-sample)':    window_slice(base_sigs, 0, 60),
        '90d (full cache)':   base_sigs,
        'OOS held-out (60-90d ago)': window_slice(base_sigs, 60, 90),
    }

    # Subsets to test (per the original T1b winner logic)
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
        if not w_sigs:
            print(f"\n[{w_name}: EMPTY — skipping]")
            continue
        print(f"\n[{w_name}: {len(w_sigs)} sigs ({w_sigs[0]['datetime'].date()} to {w_sigs[-1]['datetime'].date()})]")

        # Baseline = MIX gate already in asset_cfg, no T1b ensemble
        for base_conf in [80, 90]:
            base_ret, base_n, base_wr = sim(w_sigs, asset_cfg, bull_conf=base_conf)
            print(f"  baseline (MIX, base_conf={base_conf}): {base_ret:+.2f}% / {base_n} tr / WR {base_wr:.0f}%")

            # Test T1b ensemble configs on top of MIX
            for sub_name, sub_h in subsets.items():
                max_k = len(sub_h)
                for k_buy in range(1, max_k + 1):
                    for k_sell in range(1, max_k + 1):
                        for vote_thr in [70, 80, 85, 90]:
                            ret, n, wr = sim(
                                w_sigs, asset_cfg, per_h=per_h,
                                vote_subset=sub_h,
                                k_buy=k_buy, k_sell=k_sell,
                                vote_conf_thr=vote_thr,
                                bull_conf=base_conf, bear_conf=vote_thr)
                            rows.append({
                                'window': w_name,
                                'config': f'{sub_name} k_buy={k_buy} k_sell={k_sell} thr={vote_thr} (base_conf={base_conf})',
                                'subset': sub_name, 'k_buy': k_buy, 'k_sell': k_sell,
                                'vote_thr': vote_thr, 'base_conf': base_conf,
                                'return_pct': round(ret, 2),
                                'delta': round(ret - base_ret, 2),
                                'n_trades': n, 'win_rate': round(wr, 1),
                                'baseline_ret': round(base_ret, 2),
                            })

    df = pd.DataFrame(rows)

    # Headline: top configs across all windows
    pivot_d = df.pivot_table(index='config', columns='window', values='delta')
    pivot_d.columns = [f'd_{c.split(" ")[0]}' for c in pivot_d.columns]
    pivot_n = df.pivot_table(index='config', columns='window', values='n_trades')
    pivot_n.columns = [f'n_{c.split(" ")[0]}' for c in pivot_n.columns]

    # Use 30d/60d/OOS for ranking (90d redundant with 60d)
    rank_cols = [c for c in pivot_d.columns if 'OOS' in c or '60d' in c or '30d' in c]
    full = pd.concat([pivot_d, pivot_n], axis=1)
    full = full.sort_values(rank_cols[0] if 'OOS' in rank_cols[0] else 'd_60d', ascending=False)

    print(f"\n{'='*120}")
    print(f"  T1b ON TOP OF MIX — top 30 by OOS delta vs MIX-baseline")
    print(f"{'='*120}")
    print(full.head(30).to_string())

    # Specifically — the original T1b winner config
    print(f"\n{'='*120}")
    print(f"  ORIGINAL T1b WINNER (58_only k_buy=1 k_sell=2 thr=85 base_conf=80) — performance vs MIX baseline")
    print(f"{'='*120}")
    target = '58_only k_buy=1 k_sell=2 thr=85 (base_conf=80)'
    if target in full.index:
        row = full.loc[target]
        print(row.to_string())
    else:
        print("  Not found in results")

    # Positive-on-OOS-AND-in-sample filter
    print(f"\n{'='*120}")
    print(f"  CONFIGS POSITIVE ON OOS + 60d (true robustness signal)")
    print(f"{'='*120}")
    oos_col = next((c for c in full.columns if c.startswith('d_OOS')), None)
    if oos_col and 'd_60d' in full.columns:
        pos = full[(full[oos_col] > 0) & (full['d_60d'] > 0)].sort_values(oos_col, ascending=False)
        print(pos.head(20).to_string() if not pos.empty else '  None.')

    out = os.path.join(ENGINE, 'output',
                       f't1b_on_mix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
