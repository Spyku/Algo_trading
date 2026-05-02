"""
test_entry_after_rally.py — flip the question. Instead of "exit faster on
crash", ask: "given prior rally context, was the BUY entry itself a good idea?"

Phase 1 (diagnostic): for every BUY taken by the baseline strategy in the
last 60d/90d, compute the price's prior rally over windows {5h, 12h, 24h,
48h, 72h, 168h (=7d)}. Bucket trades by prior-rally magnitude. For each
bucket: win rate, avg PnL, avg max favorable, avg max adverse.

Phase 2 (intervention): block BUY when any prior-rally indicator >= X%.
Sweep X. Measure delta vs baseline. Includes COMBO with bull_conf=90 and
T5b winner.

Reads: cache + regime_config + eth_hourly. Writes output/entry_after_rally_*.csv.
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
        if s['datetime'].tzinfo is None:
            s['datetime'] = s['datetime'].tz_localize('UTC')
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


def precompute_prior_rallies(sigs, eth_df):
    """For each signal, prior_rally_Xh = pct gain over last Xh ending at signal time."""
    n = len(sigs)
    closes = eth_df['close']
    rally_windows = [5, 12, 24, 48, 72, 168]
    out = {h: np.full(n, np.nan) for h in rally_windows}
    out['eth_close'] = np.full(n, np.nan)

    for i, s in enumerate(sigs):
        dt = s['datetime']
        if dt.tzinfo is None: dt = dt.tz_localize('UTC')
        else: dt = dt.tz_convert('UTC')
        # current close from eth_df at-or-just-before dt
        cur_idx = eth_df.index[eth_df.index <= dt]
        if len(cur_idx) == 0:
            continue
        cur_close = closes.loc[cur_idx[-1]]
        out['eth_close'][i] = cur_close
        for h in rally_windows:
            prior_dt = dt - pd.Timedelta(hours=h)
            prior_idx = eth_df.index[eth_df.index <= prior_dt]
            if len(prior_idx) > 0:
                prior_close = closes.loc[prior_idx[-1]]
                if prior_close > 0:
                    out[h][i] = (cur_close / prior_close - 1) * 100
    return out


def collect_buy_outcomes(sigs, asset_cfg, prior_rallies, bull_conf=None):
    """Run baseline + record every BUY's prior_rally context + outcome (PnL)."""
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

    in_pos = False; entry_px = 0.0
    hold_since_entry = 0; bull_cd = bear_cd = 0
    trades = []
    open_buy = None

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
            open_buy = {
                'entry_dt': s['datetime'],
                'entry_px': price,
                'regime': regime,
                'conf': sconf,
                'rally_5h': prior_rallies[5][i],
                'rally_12h': prior_rallies[12][i],
                'rally_24h': prior_rallies[24][i],
                'rally_48h': prior_rallies[48][i],
                'rally_72h': prior_rallies[72][i],
                'rally_7d':  prior_rallies[168][i],
            }
            in_pos = True; entry_px = price; hold_since_entry = 0
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = shield_on and min_pnl > 0 and cur_pnl < min_pnl
            if not shield_blocks or override_expired:
                if open_buy is not None:
                    open_buy['pnl_pct'] = cur_pnl
                    open_buy['exit_dt'] = s['datetime']
                    open_buy['hold_h'] = hold_since_entry
                    trades.append(open_buy)
                    open_buy = None
                in_pos = False; hold_since_entry = 0

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos and open_buy is not None:
        open_buy['pnl_pct'] = (sigs[-1]['close'] / entry_px - 1) * 100
        open_buy['exit_dt'] = sigs[-1]['datetime']
        open_buy['hold_h'] = hold_since_entry
        trades.append(open_buy)
    return trades


def sim_with_block(sigs, asset_cfg, prior_rallies,
                   block_rally_5h_pct=None,
                   block_rally_12h_pct=None,
                   block_rally_24h_pct=None,
                   block_rally_48h_pct=None,
                   block_rally_72h_pct=None,
                   block_rally_7d_pct=None,
                   bull_conf=None,
                   require_dd_from_7d_high_pct=None,
                   bear_dd_from_7d_high_pct=None,
                   eth_high_arr=None):
    """Baseline strategy + block BUY when any specified prior_rally indicator >= threshold."""
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
    blocked = 0

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
            block = False
            checks = [
                (block_rally_5h_pct,  prior_rallies[5][i]),
                (block_rally_12h_pct, prior_rallies[12][i]),
                (block_rally_24h_pct, prior_rallies[24][i]),
                (block_rally_48h_pct, prior_rallies[48][i]),
                (block_rally_72h_pct, prior_rallies[72][i]),
                (block_rally_7d_pct,  prior_rallies[168][i]),
            ]
            for thr, val in checks:
                if thr is not None and not np.isnan(val) and val >= thr:
                    block = True
                    break
            # Combo: require dd from 7d high (T5b winner mechanism) when in bull
            if not block and require_dd_from_7d_high_pct is not None and eth_high_arr is not None:
                dd = eth_high_arr[i]
                regime_thr = require_dd_from_7d_high_pct if regime == 'bull' else (bear_dd_from_7d_high_pct or require_dd_from_7d_high_pct)
                if not np.isnan(dd) and dd > -abs(regime_thr):
                    block = True
            if block:
                blocked += 1
            else:
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
    return ret, n, (wins / n * 100 if n else 0), blocked


def precompute_dd_from_7d_high(sigs, eth_df):
    n = len(sigs)
    out = np.full(n, np.nan)
    eth_high = eth_df['high']
    for i, s in enumerate(sigs):
        dt = s['datetime']
        if dt.tzinfo is None: dt = dt.tz_localize('UTC')
        else: dt = dt.tz_convert('UTC')
        sub = eth_high[(eth_high.index <= dt) & (eth_high.index >= dt - timedelta(days=7))]
        if len(sub) > 24:
            cur_idx = eth_df.index[eth_df.index <= dt]
            if len(cur_idx) > 0:
                out[i] = (eth_df.loc[cur_idx[-1], 'close'] / sub.max() - 1) * 100
    return out


def main():
    print("Loading...")
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']
    sigs = load_sigs()
    eth = pd.read_csv(ETH_HOURLY)
    eth['datetime'] = pd.to_datetime(eth['datetime'], utc=True)
    eth = eth.set_index('datetime').sort_index()

    # PHASE 1 — diagnostic on 90d
    print("\n=== PHASE 1: DIAGNOSTIC — collect every BUY's prior-rally context + outcome ===")
    win90 = sigs
    print("Precomputing prior rallies (this may take a moment)...")
    pr_90 = precompute_prior_rallies(win90, eth)
    trades_baseline = collect_buy_outcomes(win90, asset_cfg, pr_90)
    df_t = pd.DataFrame(trades_baseline)
    if 'pnl_pct' not in df_t.columns or len(df_t) == 0:
        print("  No trades found.")
        return

    # Bucket by prior_rally_24h
    print(f"\nTotal baseline trades (90d): {len(df_t)}, avg PnL = {df_t['pnl_pct'].mean():.2f}%, "
          f"WR = {(df_t['pnl_pct'] > 0).mean()*100:.0f}%")

    for col in ['rally_5h', 'rally_12h', 'rally_24h', 'rally_48h', 'rally_72h', 'rally_7d']:
        print(f"\n--- Bucketed by {col} ---")
        bucket_edges = [-100, -2, 0, 2, 4, 6, 8, 10, 100]
        bucket_lbls = ['<-2', '-2..0', '0..2', '2..4', '4..6', '6..8', '8..10', '>=10']
        df_t[f'{col}_bucket'] = pd.cut(df_t[col], bucket_edges, labels=bucket_lbls)
        agg = df_t.groupby(f'{col}_bucket', observed=False).agg(
            n=('pnl_pct', 'count'),
            wr=('pnl_pct', lambda s: (s > 0).mean() * 100),
            mean_pnl=('pnl_pct', 'mean'),
            median_pnl=('pnl_pct', 'median'),
            sum_pnl=('pnl_pct', 'sum'),
        ).round(2)
        print(agg.to_string())

    # PHASE 2 — backtest entry-blocking filters
    print("\n=== PHASE 2: BACKTEST — block BUY if prior rally above threshold ===")
    eth_high_30 = precompute_dd_from_7d_high(window_slice(sigs, 30), eth)
    eth_high_60 = precompute_dd_from_7d_high(window_slice(sigs, 60), eth)
    eth_high_90 = precompute_dd_from_7d_high(sigs, eth)

    windows = {
        '30d': (window_slice(sigs, 30), eth_high_30),
        '60d': (window_slice(sigs, 60), eth_high_60),
        '90d': (sigs, eth_high_90),
    }

    rows = []
    for w_name, (w_sigs, w_eth_high) in windows.items():
        print(f"\n[{w_name}: {len(w_sigs)} sigs]")
        pr = precompute_prior_rallies(w_sigs, eth)
        b_ret, b_n, b_wr, _ = sim_with_block(w_sigs, asset_cfg, pr)
        print(f"  baseline: {b_ret:+.2f}% / {b_n} tr / WR {b_wr:.0f}%")

        tests = []
        # Single-window blocks
        for h_lbl, key in [('5h', 'block_rally_5h_pct'),
                           ('12h', 'block_rally_12h_pct'),
                           ('24h', 'block_rally_24h_pct'),
                           ('48h', 'block_rally_48h_pct'),
                           ('72h', 'block_rally_72h_pct'),
                           ('7d',  'block_rally_7d_pct')]:
            for thr in [3.0, 5.0, 7.0, 10.0]:
                tests.append((f'block_rally_{h_lbl}>={thr}%', {key: thr}))
        # Combo with bull_conf=90
        for h_lbl, key in [('24h', 'block_rally_24h_pct'),
                           ('48h', 'block_rally_48h_pct'),
                           ('72h', 'block_rally_72h_pct')]:
            for thr in [5.0, 7.0, 10.0]:
                tests.append((f'block_rally_{h_lbl}>={thr}% + bull_conf=90',
                              {key: thr, 'bull_conf': 90}))
        # T5b combo: dd_from_7d_high + bull_conf=90 + rally_block
        for h_lbl, key in [('72h', 'block_rally_72h_pct'),
                           ('48h', 'block_rally_48h_pct')]:
            for r_thr in [5.0, 7.0]:
                tests.append((f'block_rally_{h_lbl}>={r_thr}% + bull_conf=90 + bull_dd>=3% + bear_dd>=5%',
                              {key: r_thr, 'bull_conf': 90,
                               'require_dd_from_7d_high_pct': 3.0,
                               'bear_dd_from_7d_high_pct': 5.0}))

        for name, kwargs in tests:
            ret, n, wr, blk = sim_with_block(w_sigs, asset_cfg, pr, eth_high_arr=w_eth_high, **kwargs)
            rows.append({
                'window': w_name,
                'config': name,
                'return_pct': round(ret, 2),
                'delta': round(ret - b_ret, 2),
                'n_trades': n,
                'win_rate': round(wr, 1),
                'blocked': blk,
            })

    df = pd.DataFrame(rows)
    pivot = df.pivot(index='config', columns='window', values='delta')
    pivot.columns = [f'd_{c}' for c in pivot.columns]
    pivot_n = df.pivot(index='config', columns='window', values='n_trades')
    pivot_n.columns = [f'n_{c}' for c in pivot_n.columns]
    pivot_b = df.pivot(index='config', columns='window', values='blocked')
    pivot_b.columns = [f'blk_{c}' for c in pivot_b.columns]
    full = pd.concat([pivot, pivot_n, pivot_b], axis=1).sort_values('d_60d', ascending=False)

    print(f"\n{'='*130}")
    print(f"  ENTRY-AFTER-RALLY BLOCK — sorted by 60d delta vs baseline")
    print(f"{'='*130}")
    print(full.to_string())

    print(f"\n{'='*130}")
    print(f"  POSITIVE ON ALL 3 WINDOWS")
    print(f"{'='*130}")
    pos = full[(full['d_30d'] > 0) & (full['d_60d'] > 0) & (full['d_90d'] > 0)]
    print(pos.to_string() if not pos.empty else "  None.")

    out = os.path.join(ENGINE, 'output',
                       f'entry_after_rally_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Save per-trade diagnostic too
    diag_out = os.path.join(ENGINE, 'output',
                            f'entry_after_rally_diag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    df_t.to_csv(diag_out, index=False)
    print(f"Per-trade diagnostic: {diag_out}")


if __name__ == '__main__':
    main()
