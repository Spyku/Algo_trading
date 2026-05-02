"""
test_t8_gdelt_overlay.py — GDELT geopolitical features as ENTRY OVERLAY filter.

Earlier audit (2026-04-19) found GDELT features were never selected by any of
33 production models. But that was during pre-Iran-rally regime. Today's
2026-04-26 rally is broadly Iran/geopolitically driven, so GDELT may carry
real signal NOW. Test: don't retrain — instead overlay GDELT-based filters
on the existing baseline signal stream and measure delta.

Variants tested:
  - Block BUY when geopolitical_vol > Yth pctile of 30d (chaos avoidance)
  - Allow BUY only when geopolitical_vol > Yth pctile (chaos opportunity)
  - Block BUY when iran_tone < threshold (escalation)
  - Block BUY when geopolitical_tone falls > X over last 24h
  - Allow BUY only when iran_tone has improved over 24h

If ANY variant beats baseline by >+5pp on 60d AND is positive on all 3
windows, we have evidence GDELT carries signal in current regime →
schedule re-download + Mode D test on full feature set.

Reads: cache + regime_config + eth_hourly + gdelt_geopolitical.
Writes: output/t8_gdelt_overlay_*.csv
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
GDELT_CSV = os.path.join(ENGINE, 'data', 'macro_data', 'gdelt_geopolitical.csv')
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


def precompute_gdelt(sigs, gd):
    """For each signal, compute GDELT features at that timestamp."""
    n = len(sigs)
    geo_vol = np.full(n, np.nan)
    geo_tone = np.full(n, np.nan)
    iran_tone = np.full(n, np.nan)
    geo_vol_pctile_30d = np.full(n, np.nan)
    geo_tone_chg24h = np.full(n, np.nan)
    iran_tone_chg24h = np.full(n, np.nan)

    for i, s in enumerate(sigs):
        dt = s['datetime']
        if dt.tzinfo is not None:
            dt = dt.tz_convert(None)

        # Latest GDELT row at-or-before dt
        gsub = gd[gd.index <= dt]
        if len(gsub) == 0:
            continue
        last = gsub.iloc[-1]
        geo_vol[i] = last['geopolitical_vol']
        geo_tone[i] = last['geopolitical_tone']
        iran_tone[i] = last['iran_tone']

        # 30-day percentile of geopolitical_vol
        win = gsub.tail(30 * 24)['geopolitical_vol'].dropna()
        if len(win) > 24 and not np.isnan(geo_vol[i]):
            geo_vol_pctile_30d[i] = (win < geo_vol[i]).sum() / len(win) * 100

        # 24h changes
        prior = gd[gd.index <= dt - timedelta(hours=24)]
        if len(prior) > 0:
            p = prior.iloc[-1]
            if not np.isnan(geo_tone[i]) and not np.isnan(p['geopolitical_tone']):
                geo_tone_chg24h[i] = geo_tone[i] - p['geopolitical_tone']
            if not np.isnan(iran_tone[i]) and not np.isnan(p['iran_tone']):
                iran_tone_chg24h[i] = iran_tone[i] - p['iran_tone']

    return geo_vol, geo_tone, iran_tone, geo_vol_pctile_30d, geo_tone_chg24h, iran_tone_chg24h


def sim(sigs, asset_cfg, gv_pctile, geo_tone, iran_tone, gt_chg24, it_chg24,
        block_above_geovol_pctile=None,
        only_above_geovol_pctile=None,
        block_below_iran_tone=None,
        only_above_iran_tone=None,
        block_below_geo_tone_chg24=None,
        only_iran_improving_24h=False,
        bull_conf=None):
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
            if block_above_geovol_pctile is not None:
                v = gv_pctile[i]
                if not np.isnan(v) and v > block_above_geovol_pctile:
                    block = True
            if not block and only_above_geovol_pctile is not None:
                v = gv_pctile[i]
                if np.isnan(v) or v < only_above_geovol_pctile:
                    block = True
            if not block and block_below_iran_tone is not None:
                v = iran_tone[i]
                if not np.isnan(v) and v < block_below_iran_tone:
                    block = True
            if not block and only_above_iran_tone is not None:
                v = iran_tone[i]
                if np.isnan(v) or v < only_above_iran_tone:
                    block = True
            if not block and block_below_geo_tone_chg24 is not None:
                v = gt_chg24[i]
                if not np.isnan(v) and v < block_below_geo_tone_chg24:
                    block = True
            if not block and only_iran_improving_24h:
                v = it_chg24[i]
                if np.isnan(v) or v <= 0:
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


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    print("Loading...")
    sigs = load_sigs()
    eth = pd.read_csv(ETH_HOURLY)
    eth['datetime'] = pd.to_datetime(eth['datetime'], utc=True)
    eth = eth.set_index('datetime').sort_index()
    eth.index = eth.index.tz_convert(None)
    gd = pd.read_csv(GDELT_CSV)
    gd['datetime'] = pd.to_datetime(gd['datetime'])
    gd = gd.set_index('datetime').sort_index()
    print(f"GDELT range: {gd.index[0]} to {gd.index[-1]} ({len(gd)} rows)")

    windows = {'30d': window_slice(sigs, 30),
               '60d': window_slice(sigs, 60),
               '90d': sigs}

    rows = []
    for w_name, w_sigs in windows.items():
        print(f"\n[{w_name}: {len(w_sigs)} sigs]")
        gv, gt, it, gv_p, gt_c, it_c = precompute_gdelt(w_sigs, gd)
        print(f"  GDELT coverage: geo_vol={(~np.isnan(gv)).sum()}/{len(gv)} sigs")

        for conf in [80, 90]:
            base_ret, base_n, base_wr, _ = sim(w_sigs, asset_cfg,
                                                gv_p, gt, it, gt_c, it_c,
                                                bull_conf=conf)
            print(f"  baseline_{conf}: {base_ret:+.2f}% / {base_n} tr / WR {base_wr:.0f}%")

            tests = []
            for y in [50, 70, 80, 90]:
                tests.append((f'block_above_geovol_pctile={y}', dict(block_above_geovol_pctile=y)))
                tests.append((f'only_above_geovol_pctile={y}', dict(only_above_geovol_pctile=y)))
            for thr in [-5, -4, -3, -2]:
                tests.append((f'block_below_iran_tone<{thr}', dict(block_below_iran_tone=thr)))
                tests.append((f'only_above_iran_tone>{thr}', dict(only_above_iran_tone=thr)))
            for thr in [-1.0, -0.5, 0.0, 0.5]:
                tests.append((f'block_below_geo_tone_chg24<{thr}', dict(block_below_geo_tone_chg24=thr)))
            tests.append(('only_iran_improving_24h', dict(only_iran_improving_24h=True)))

            for name, kwargs in tests:
                ret, n, wr, blk = sim(w_sigs, asset_cfg,
                                       gv_p, gt, it, gt_c, it_c,
                                       bull_conf=conf, **kwargs)
                rows.append({
                    'window': w_name,
                    'config': f'{name} (conf={conf})',
                    'return_pct': round(ret, 2),
                    'delta': round(ret - base_ret, 2),
                    'n_trades': n,
                    'win_rate': round(wr, 1),
                    'blocked': blk,
                })

    df = pd.DataFrame(rows)
    pivot = df.pivot(index='config', columns='window', values='delta')
    pivot.columns = [f'd_{c}' for c in pivot.columns]
    pivot_n = df.pivot(index='config', columns='window', values='n_trades')
    pivot_n.columns = [f'n_{c}' for c in pivot_n.columns]
    full = pd.concat([pivot, pivot_n], axis=1).sort_values('d_60d', ascending=False)

    print(f"\n{'='*120}")
    print(f"  T8 GDELT OVERLAY — sorted by 60d delta")
    print(f"{'='*120}")
    print(full.head(25).to_string())

    print(f"\n{'='*120}")
    print(f"  POSITIVE ON ALL 3 WINDOWS")
    print(f"{'='*120}")
    pos = full[(full['d_30d'] > 0) & (full['d_60d'] > 0) & (full['d_90d'] > 0)]
    print(pos.to_string() if not pos.empty else "  None.")

    out = os.path.join(ENGINE, 'output',
                       f't8_gdelt_overlay_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
