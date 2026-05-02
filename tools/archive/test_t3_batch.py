"""
test_t3_batch.py — three new untested ideas.

T3a. VOLATILITY-REGIME ENTRY FILTER
   H3 finding: MAX_HOLD trades had vol_ratio_24h_30d median 0.67 vs MODEL
   trades 0.89. MAX_HOLD trades cluster in CALM conditions. Block BUYs when
   24h vol < X * 30d vol.

T3b. CONFIDENCE MOMENTUM FILTER
   Don't just look at current confidence — look at confidence trend.
   Block BUY if last 3 model bars show DECLINING confidence (model losing
   conviction). Conversely, prioritize BUYs where confidence is rising.

T3c. DAY-OF-WEEK FILTER
   Block BUYs on specific weekdays (Sat/Sun underperformance hypothesis from
   crypto literature).

T3d. RECENT-SELL-DENSITY FILTER
   If model said SELL on >X% of last N bars (in cash periods), current BUY
   is regime-flipping noise. Block.

Reads (read-only): cache + regime_config + eth_hourly
Writes: output/t3_batch_<timestamp>.csv
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


def precompute_vol_ratio(sigs, eth_df):
    """24h realized vol / 30d realized vol per signal bar."""
    n = len(sigs)
    out = np.full(n, np.nan)
    closes = eth_df['close']
    for i, s in enumerate(sigs):
        dt = s['datetime']
        if dt.tzinfo is None: dt = dt.tz_localize('UTC')
        else: dt = dt.tz_convert('UTC')
        sub24 = closes[(closes.index < dt) & (closes.index >= dt - timedelta(hours=24))]
        sub30d = closes[(closes.index < dt) & (closes.index >= dt - timedelta(days=30))]
        if len(sub24) > 4 and len(sub30d) > 100:
            v24 = np.log(sub24).diff().std()
            v30d = np.log(sub30d).diff().std()
            if v30d > 0:
                out[i] = v24 / v30d
    return out


def precompute_conf_momentum(sigs, lookback=3):
    """For each bar, average confidence change over last `lookback` bars
    of the same signal type. Positive = building conviction."""
    n = len(sigs)
    out = np.full(n, np.nan)
    confs = np.array([float(s.get('confidence', 0)) for s in sigs])
    for i in range(lookback, n):
        # Compute conf delta over last `lookback` bars regardless of signal type
        recent = confs[i - lookback:i + 1]
        # Linear slope (not avg diff)
        x = np.arange(len(recent))
        if recent.std() > 0:
            slope = np.polyfit(x, recent, 1)[0]
        else:
            slope = 0
        out[i] = slope
    return out


def precompute_recent_sell_density(sigs, lookback=12):
    """Fraction of last N signals that were SELL. High = bear-flipping zone."""
    n = len(sigs)
    out = np.full(n, np.nan)
    is_sell = np.array([1 if s['signal'] == 'SELL' else 0 for s in sigs])
    for i in range(lookback, n):
        out[i] = is_sell[i - lookback:i].sum() / lookback
    return out


def sim(sigs, asset_cfg, vol_ratio_arr, conf_mom_arr, sell_density_arr,
        vol_min_ratio=None, vol_max_ratio=None,
        conf_mom_min=None,
        block_weekdays=None,
        sell_density_max=None):
    """Apply filters at BUY time. None = no-op."""
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
    blocked_count = 0

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
            block = False
            # T3a vol regime
            if vol_min_ratio is not None:
                vr = vol_ratio_arr[i]
                if not np.isnan(vr) and vr < vol_min_ratio:
                    block = True
            if not block and vol_max_ratio is not None:
                vr = vol_ratio_arr[i]
                if not np.isnan(vr) and vr > vol_max_ratio:
                    block = True
            # T3b confidence momentum
            if not block and conf_mom_min is not None:
                cm = conf_mom_arr[i]
                if not np.isnan(cm) and cm < conf_mom_min:
                    block = True
            # T3c day-of-week
            if not block and block_weekdays is not None:
                dt = s['datetime']
                if dt.tzinfo is None: dt = dt.tz_localize('UTC')
                if dt.weekday() in block_weekdays:
                    block = True
            # T3d sell density
            if not block and sell_density_max is not None:
                sd = sell_density_arr[i]
                if not np.isnan(sd) and sd > sell_density_max:
                    block = True

            if block:
                blocked_count += 1
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
                held = 0; in_pos = False; hold_since_entry = 0

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        cash = held * sigs[-1]['close'] * (1 - FEE)
        trade_log.append({'pnl_pct': (sigs[-1]['close'] / entry_px - 1) * 100,
                          'regime': sigs[-1].get('regime', 'bull')})

    ret = (cash / 1000.0 - 1) * 100
    n = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    wr = (wins / n * 100) if n else 0
    return ret, n, wr, blocked_count


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    sigs = load_sigs()
    eth = pd.read_csv(ETH_HOURLY)
    eth['datetime'] = pd.to_datetime(eth['datetime'], utc=True)
    eth = eth.set_index('datetime').sort_index()

    print("Precomputing context...")
    vol_full = precompute_vol_ratio(sigs, eth)
    conf_mom_full = precompute_conf_momentum(sigs, lookback=3)
    sell_dens_full = precompute_recent_sell_density(sigs, lookback=12)
    print(f"  vol_ratio: median={np.nanmedian(vol_full):.2f}, "
          f"mean={np.nanmean(vol_full):.2f}, "
          f"min/max=({np.nanmin(vol_full):.2f}, {np.nanmax(vol_full):.2f})")
    print(f"  conf_mom: median={np.nanmedian(conf_mom_full):.2f}")
    print(f"  sell_density: median={np.nanmedian(sell_dens_full):.2f}")

    windows = {'30d': 30, '60d': 60, '90d': 90}
    full_rows = []

    for w_name, days in windows.items():
        w_sigs = window_slice(sigs, days) if days < 90 else sigs
        # Slice the precomputed arrays accordingly
        # full_sig_idx_offset = where this window starts in `sigs`
        offset = len(sigs) - len(w_sigs)
        vol_w = vol_full[offset:]
        cm_w = conf_mom_full[offset:]
        sd_w = sell_dens_full[offset:]

        base, base_n, base_wr, _ = sim(w_sigs, asset_cfg, vol_w, cm_w, sd_w)

        tests = []
        # T3a vol regime
        for vmin in [0.6, 0.7, 0.8, 0.9, 1.0]:
            tests.append((f'T3a vol_min={vmin}', dict(vol_min_ratio=vmin)))
        # T3a vol max (block volatile entries)
        for vmax in [1.0, 1.2, 1.5]:
            tests.append((f'T3a vol_max={vmax}', dict(vol_max_ratio=vmax)))
        # T3b conf momentum
        for cm in [0.0, 1.0, 2.0, 5.0]:
            tests.append((f'T3b conf_mom_min={cm}', dict(conf_mom_min=cm)))
        # T3c day of week
        tests.append(('T3c block_weekend (Sat+Sun)', dict(block_weekdays={5, 6})))
        tests.append(('T3c block_monday', dict(block_weekdays={0})))
        tests.append(('T3c block_friday', dict(block_weekdays={4})))
        # T3d sell density
        for sd in [0.3, 0.5, 0.7]:
            tests.append((f'T3d sell_density_max={sd}', dict(sell_density_max=sd)))
        # Combo with already-known winner (bull conf at 90% via min_confidence override)
        # Skip — would require sim signature change.

        for name, kw in tests:
            ret, n, wr, blk = sim(w_sigs, asset_cfg, vol_w, cm_w, sd_w, **kw)
            full_rows.append({
                'window': w_name, 'test': name,
                'return_pct': round(ret, 2),
                'delta': round(ret - base, 2),
                'n_trades': n, 'win_rate': round(wr, 1),
                'blocked': blk,
            })

    df = pd.DataFrame(full_rows)
    print("\n" + "=" * 110)
    print("  T3 RESULTS")
    print("=" * 110)
    pivot = df.pivot(index='test', columns='window', values='delta').round(2)
    pivot.columns = [f'd_{c}' for c in pivot.columns]
    counts = df.pivot(index='test', columns='window', values='n_trades')
    counts.columns = [f'n_{c}' for c in counts.columns]
    full = pd.concat([pivot, counts], axis=1)
    # Sort by 60d delta
    full = full.sort_values('d_60d', ascending=False)
    print(full.to_string())

    out = os.path.join(ENGINE, 'output',
                       f't3_batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved to: {out}")

    # Highlight any 3/3 positive
    print("\n" + "=" * 110)
    print("  CONFIGS POSITIVE ON ALL 3 WINDOWS")
    print("=" * 110)
    pos = full[(full['d_30d'] > 0) & (full['d_60d'] > 0) & (full['d_90d'] > 0)]
    if pos.empty:
        print("  None.")
    else:
        print(pos.to_string())


if __name__ == '__main__':
    main()
