"""
test_t2_batch.py — batch test of 6 new ideas, all standalone, all read-only.

T2a. PROFIT-TARGET EXIT — exit at +X% PnL even before model SELL fires.
     Different from prior trailing-stop tests: this is a hard target, not a
     trail. Tests {2%, 3%, 5%}.

T2b. BTC CROSS-ASSET ENTRY VETO — skip ETH BUY if BTC 24h return < threshold
     (BTC trending down). Different from prior anti-chase: this is one-sided
     (just BTC negative blocks).

T2c. DRAWDOWN-FROM-RECENT-HIGH FILTER — only BUY ETH if currently down ≥X%
     from 7d high (mean-reversion entry filter). Tests {1%, 2%, 3%}.

T2d. TIME-SINCE-LAST-TRADE FILTER — don't BUY within N hours of last SELL.
     Anti-overtrading filter. Tests {2h, 4h, 8h}.

T2e. VOLUME-CONFIRMATION ENTRY — only BUY if last 4h cumulative volume >
     median 7d cumulative 4h volume × multiplier. Tests {1.0x, 1.5x, 2.0x}.

T2f. MULTI-HORIZON DISAGREEMENT FILTER (PROXY) — don't BUY if confidence
     gap between model's BUY conf and threshold is large (low-conviction =
     proxy for likely disagreement with other horizons). Tests by requiring
     conf gap >= {5pp, 10pp, 15pp} above threshold.

Reads (read-only): cache + regime_config + eth_hourly + btc_hourly
Writes: output/t2_batch_<timestamp>.csv
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
BTC_HOURLY = os.path.join(ENGINE, 'data', 'btc_hourly_data.csv')
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


def precompute_context(sigs, eth_df, btc_df):
    """For each signal, compute supporting context arrays:
       eth_24h_ret, btc_24h_ret, dd_from_7d_high_pct,
       vol_4h_to_7d_median_ratio."""
    n = len(sigs)
    eth_24h = np.full(n, np.nan)
    btc_24h = np.full(n, np.nan)
    dd_from_7d = np.full(n, np.nan)
    vol_ratio = np.full(n, np.nan)

    eth_close = eth_df['close']
    eth_high = eth_df['high']
    eth_volume = eth_df['volume'] if 'volume' in eth_df.columns else None
    btc_close = btc_df['close'] if btc_df is not None else None

    for i, s in enumerate(sigs):
        dt = s['datetime']
        if dt.tzinfo is None:
            dt = dt.tz_localize('UTC')
        else:
            dt = dt.tz_convert('UTC')

        sub24 = eth_close[(eth_close.index < dt) & (eth_close.index >= dt - timedelta(hours=24))]
        if len(sub24) > 4:
            eth_24h[i] = 100 * (sub24.iloc[-1] / sub24.iloc[0] - 1)

        if btc_close is not None:
            bsub = btc_close[(btc_close.index < dt) & (btc_close.index >= dt - timedelta(hours=24))]
            if len(bsub) > 4:
                btc_24h[i] = 100 * (bsub.iloc[-1] / bsub.iloc[0] - 1)

        sub7d = eth_high[(eth_high.index < dt) & (eth_high.index >= dt - timedelta(days=7))]
        if len(sub7d) > 24 and len(sub24) > 0:
            high_7d = sub7d.max()
            cur_close = sub24.iloc[-1]
            dd_from_7d[i] = 100 * (cur_close / high_7d - 1)  # negative if below high

        if eth_volume is not None:
            sub4h = eth_volume[(eth_volume.index < dt) & (eth_volume.index >= dt - timedelta(hours=4))]
            sub7d_vol = eth_volume[(eth_volume.index < dt) & (eth_volume.index >= dt - timedelta(days=7))]
            if len(sub4h) >= 3 and len(sub7d_vol) >= 24:
                cur_4h_total = sub4h.sum()
                # median rolling 4h sum over 7d
                rolled = sub7d_vol.rolling(4).sum().dropna()
                if len(rolled) > 0:
                    median_4h = rolled.median()
                    if median_4h > 0:
                        vol_ratio[i] = cur_4h_total / median_4h

    return eth_24h, btc_24h, dd_from_7d, vol_ratio


def sim(sigs, asset_cfg, eth_24h_arr, btc_24h_arr, dd_arr, vol_arr,
        profit_target_pct=None,
        btc_veto_thr=None,
        dd_min_pct=None,
        time_since_sell_h=0,
        vol_mult_min=None,
        conf_gap_pp=0):
    """Simulator with all 6 idea filters layered in. Defaults = no-op."""
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
    last_sell_idx = -10**9
    trade_log = []
    blocked = 0
    pt_fires = 0

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

        # Profit target check (T2a) — applies before shield/max_hold
        if in_pos and profit_target_pct is not None:
            cur_pnl = (price / entry_px - 1) * 100
            if cur_pnl >= profit_target_pct:
                cash = held * price * (1 - FEE)
                trade_log.append({'pnl_pct': cur_pnl, 'exit': 'PROFIT_TARGET'})
                held = 0; in_pos = False; hold_since_entry = 0
                last_sell_idx = i
                pt_fires += 1
                if bull_cd > 0: bull_cd -= 1
                if bear_cd > 0: bear_cd -= 1
                continue

        if sig == 'BUY' and sconf >= conf_thr and not in_pos and active_cd <= 0:
            # Apply entry filters (T2b/T2c/T2d/T2e/T2f)
            block = False
            if btc_veto_thr is not None:
                br = btc_24h_arr[i]
                if not np.isnan(br) and br < btc_veto_thr:
                    block = True
            if not block and dd_min_pct is not None:
                dd = dd_arr[i]
                if not np.isnan(dd) and dd > -abs(dd_min_pct):  # dd is negative
                    block = True
            if not block and time_since_sell_h > 0:
                if (i - last_sell_idx) < time_since_sell_h:
                    block = True
            if not block and vol_mult_min is not None:
                vr = vol_arr[i]
                if not np.isnan(vr) and vr < vol_mult_min:
                    block = True
            if not block and conf_gap_pp > 0:
                if (sconf - conf_thr) < conf_gap_pp:
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
                trade_log.append({'pnl_pct': cur_pnl, 'exit': 'MODEL'})
                held = 0; in_pos = False; hold_since_entry = 0
                last_sell_idx = i

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        cash = held * sigs[-1]['close'] * (1 - FEE)
        trade_log.append({'pnl_pct': (sigs[-1]['close'] / entry_px - 1) * 100,
                          'exit': 'OPEN_AT_END'})

    ret = (cash / 1000.0 - 1) * 100
    n = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    wr = (wins / n * 100) if n else 0
    return ret, n, wr, blocked, pt_fires


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    print("Loading data...")
    sigs = load_sigs()
    eth = pd.read_csv(ETH_HOURLY)
    eth['datetime'] = pd.to_datetime(eth['datetime'], utc=True)
    eth = eth.set_index('datetime').sort_index()
    btc = pd.read_csv(BTC_HOURLY) if os.path.exists(BTC_HOURLY) else None
    if btc is not None:
        btc['datetime'] = pd.to_datetime(btc['datetime'], utc=True)
        btc = btc.set_index('datetime').sort_index()

    # Test on 60d AND 90d
    windows = {'60d': window_slice(sigs, 60), '90d': sigs}

    all_rows = []
    for win_name, win_sigs in windows.items():
        print(f"\n[Window: {win_name}, {len(win_sigs)} signals]")
        print("Precomputing context...")
        eth_24h, btc_24h, dd_arr, vol_arr = precompute_context(win_sigs, eth, btc)

        # Baseline
        base_ret, base_n, base_wr, _, _ = sim(win_sigs, asset_cfg, eth_24h, btc_24h, dd_arr, vol_arr)
        print(f"  BASELINE: {base_ret:+.2f}% / {base_n} trades / WR {base_wr:.0f}%")

        tests = []
        # T2a profit target
        for pt in [2.0, 3.0, 5.0]:
            tests.append((f'T2a profit_target={pt}%', dict(profit_target_pct=pt)))
        # T2b BTC veto
        for bt in [-1.0, -0.5, 0.0, 0.5]:
            tests.append((f'T2b btc_veto<{bt}%', dict(btc_veto_thr=bt)))
        # T2c drawdown filter
        for dd in [1.0, 2.0, 3.0]:
            tests.append((f'T2c dd_min={dd}%', dict(dd_min_pct=dd)))
        # T2d time since last SELL
        for th in [2, 4, 8, 12]:
            tests.append((f'T2d wait_after_sell={th}h', dict(time_since_sell_h=th)))
        # T2e volume multiplier
        for vm in [1.0, 1.5, 2.0]:
            tests.append((f'T2e vol_mult>={vm}x', dict(vol_mult_min=vm)))
        # T2f confidence gap
        for cg in [5, 10, 15]:
            tests.append((f'T2f conf_gap>={cg}pp', dict(conf_gap_pp=cg)))
        # Combos
        tests.append(('combo: btc_veto<0 + dd_min=2%', dict(btc_veto_thr=0.0, dd_min_pct=2.0)))
        tests.append(('combo: profit_target=3% + dd_min=2%', dict(profit_target_pct=3.0, dd_min_pct=2.0)))

        for name, kwargs in tests:
            ret, n, wr, blocked, pt_fires = sim(
                win_sigs, asset_cfg, eth_24h, btc_24h, dd_arr, vol_arr, **kwargs)
            all_rows.append({
                'window': win_name,
                'test': name,
                'return_pct': round(ret, 2),
                'delta': round(ret - base_ret, 2),
                'n_trades': n,
                'win_rate': round(wr, 1),
                'blocked': blocked,
                'pt_fires': pt_fires,
            })

    df = pd.DataFrame(all_rows)
    print("\n" + "=" * 120)
    print("  RESULTS — sorted by 60d delta")
    print("=" * 120)
    pivot = df.pivot(index='test', columns='window', values='delta').round(2)
    pivot.columns = [f'delta_{c}' for c in pivot.columns]
    counts = df.pivot(index='test', columns='window', values='n_trades')
    counts.columns = [f'n_{c}' for c in counts.columns]
    wrs = df.pivot(index='test', columns='window', values='win_rate')
    wrs.columns = [f'wr_{c}' for c in wrs.columns]
    full = pd.concat([pivot, counts, wrs], axis=1)
    full = full.sort_values('delta_60d', ascending=False)
    print(full.to_string())

    out = os.path.join(ENGINE, 'output',
                       f't2_batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved to: {out}")


if __name__ == '__main__':
    main()
