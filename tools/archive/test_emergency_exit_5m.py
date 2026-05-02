"""
test_emergency_exit_5m.py — Phase 2: backtest emergency-exit overlay on
top of the current hourly strategy, using 5-minute price data for the
trigger logic.

Trigger family (default): EMERG fires when in_pos AND ret_15m < -1.0%
on the 5-minute series. Variants tested:
  - threshold ∈ {-0.7%, -1.0%, -1.2%, -1.5%, -2.0%}
  - lookback ∈ {15min, 20min, 30min}
  - compound: ret_15m < X AND dd_from_60m_high < Y
  - regime gating: all / bull-only / bear-only
  - re-entry cooldown after emergency: {0, 60, 120, 180, 360 min}
  - shield bypass mode: always-bypass vs bypass-only-if-pnl>=0

Hourly strategy = current production cache (data/eth_sl_signals_90d.pkl,
detector + bull/bear horizons + shield + max_hold + rally cooldowns).
At each hour: same buy/sell decision as baseline. Within each hour: scan
the 12 × 5-minute bars; if EMERG triggers while in_pos, force-sell at
that 5m close (with fee) and start re-entry cooldown.

Reads: cache + regime_config + eth_5m_backtest_90d.csv + fresh Binance.
Writes: output/emerg_exit_5m_<timestamp>.csv.
"""
from __future__ import annotations

import json
import os
import pickle
import ssl
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PKL = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
REGIME_CFG = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
ETH_5M_CSV = os.path.join(ENGINE, 'data', 'eth_5m_backtest_90d.csv')
FEE = 0.0005


def load_combined_5m():
    df_cache = pd.read_csv(ETH_5M_CSV)
    df_cache['datetime'] = pd.to_datetime(df_cache['datetime'], utc=True)
    df_cache = df_cache.set_index('datetime').sort_index()
    last_cache = df_cache.index[-1]

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    fresh_rows = []
    start_ms = int((last_cache + timedelta(minutes=5)).timestamp() * 1000)
    while True:
        url = (f'https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=5m'
               f'&startTime={start_ms}&limit=1000')
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=10) as r:
                batch = json.loads(r.read())
        except Exception:
            break
        if not batch:
            break
        for k in batch:
            fresh_rows.append({
                'datetime': pd.Timestamp(k[0], unit='ms', tz='UTC'),
                'open': float(k[1]), 'high': float(k[2]),
                'low': float(k[3]), 'close': float(k[4]),
                'volume': float(k[5]),
            })
        if len(batch) < 1000:
            break
        start_ms = batch[-1][0] + 5 * 60 * 1000
    if fresh_rows:
        df_fresh = pd.DataFrame(fresh_rows).set_index('datetime')
        df = pd.concat([df_cache, df_fresh])
        df = df[~df.index.duplicated(keep='last')].sort_index()
        return df
    return df_cache


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


def precompute_5m_indicators(df_5m):
    closes = df_5m['close']
    highs = df_5m['high']
    out = pd.DataFrame(index=df_5m.index)
    # 15/20/30m returns kept for old variants
    for win_min in [15, 20, 30]:
        b = win_min // 5
        out[f'ret_{win_min}m'] = (closes / closes.shift(b) - 1) * 100
    # Short-window drops (5m, 10m) for new I-variants
    for win_min in [5, 10]:
        b = win_min // 5
        out[f'ret_{win_min}m'] = (closes / closes.shift(b) - 1) * 100
    out['dd_60m'] = (closes / highs.rolling(12).max() - 1) * 100
    # Preceding rally over various hour windows ending 5min ago (for short drops)
    for prior_h in [1, 2, 4, 5, 6, 7, 8, 9, 10, 12]:
        prior_bars = prior_h * 12
        ret_prior = (closes.shift(1) / closes.shift(1 + prior_bars) - 1) * 100
        out[f'rally_prior_{prior_h}h'] = ret_prior
    # 1st derivative: rate of change of price (= ret_5m essentially)
    # 2nd derivative: change in 5m return between adjacent bars
    out['d1_5m'] = out['ret_5m']  # 5m % change
    out['d2_5m'] = out['ret_5m'] - out['ret_5m'].shift(1)  # 5m accel
    # Same for 10m
    out['d1_10m'] = out['ret_10m']
    out['d2_10m'] = out['ret_10m'] - out['ret_10m'].shift(1)
    return out


def sim(sigs, asset_cfg, df_5m, ind5,
        emerg_enabled=False,
        emerg_lookback_min=15,
        emerg_threshold_pct=-1.0,
        emerg_compound_dd60_pct=None,
        emerg_regime='all',
        emerg_cooldown_min=120,
        emerg_bypass_only_if_profit=False,
        emerg_min_pnl_to_arm=None,
        emerg_require_prior_rally_h=None,
        emerg_require_prior_rally_pct=None,
        emerg_require_d2_below=None):
    """Simulate hourly strategy with optional 5-minute emergency-exit overlay."""
    bull_conf_thr = float(asset_cfg.get('bull', {}).get('min_confidence', 80))
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
    emerg_cd_until = pd.Timestamp(0, tz='UTC')
    trade_log = []
    emerg_fires = 0
    emerg_pnl_at_fire = []
    last_emerg_dt = None

    # Index 5m by datetime for fast slicing
    ind5_idx = ind5.index
    closes5_arr = df_5m['close'].values

    ind_col = f'ret_{emerg_lookback_min}m'

    for i, s in enumerate(sigs):
        price = float(s['close'])
        regime = s.get('regime', 'bull')
        sig = s['signal']
        sconf = float(s.get('confidence', 0))
        conf_thr = bull_conf_thr if regime == 'bull' else bear_conf_thr
        shield_on = bull_shield if regime == 'bull' else bear_shield
        hr_dt = s['datetime']

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

        # === Emergency-exit scan within this hour (5-min granularity) ===
        if emerg_enabled and in_pos:
            apply_to_regime = (emerg_regime == 'all' or
                               (emerg_regime == 'bull' and regime == 'bull') or
                               (emerg_regime == 'bear' and regime == 'bear'))
            if apply_to_regime:
                # Look at 5m bars within this hour [hr_dt, hr_dt+1h)
                bar_lo = hr_dt
                bar_hi = hr_dt + pd.Timedelta(hours=1)
                mask = (ind5_idx >= bar_lo) & (ind5_idx < bar_hi)
                ind_window = ind5.loc[mask]
                px_window_idx = np.where(mask)[0]
                for j_local, (j_global, dt_5m) in enumerate(zip(px_window_idx, ind_window.index)):
                    ret_val = ind_window[ind_col].iloc[j_local]
                    dd_val = ind_window['dd_60m'].iloc[j_local]
                    if np.isnan(ret_val):
                        continue
                    if ret_val >= emerg_threshold_pct:
                        continue
                    if emerg_compound_dd60_pct is not None:
                        if np.isnan(dd_val) or dd_val >= emerg_compound_dd60_pct:
                            continue
                    if emerg_require_prior_rally_h is not None and emerg_require_prior_rally_pct is not None:
                        rally_col = f'rally_prior_{emerg_require_prior_rally_h}h'
                        if rally_col in ind_window.columns:
                            r_val = ind_window[rally_col].iloc[j_local]
                            if np.isnan(r_val) or r_val < emerg_require_prior_rally_pct:
                                continue
                        else:
                            continue
                    if emerg_require_d2_below is not None:
                        d2_col = 'd2_5m' if emerg_lookback_min == 5 else 'd2_10m'
                        if d2_col in ind_window.columns:
                            d2_val = ind_window[d2_col].iloc[j_local]
                            if np.isnan(d2_val) or d2_val >= emerg_require_d2_below:
                                continue
                        else:
                            continue
                    # Trigger candidate
                    cur_5m_px = closes5_arr[j_global]
                    cur_pnl_5m = (cur_5m_px / entry_px - 1) * 100
                    if emerg_bypass_only_if_profit and cur_pnl_5m < 0:
                        continue
                    if emerg_min_pnl_to_arm is not None and cur_pnl_5m < emerg_min_pnl_to_arm:
                        continue
                    # Fire!
                    cash = held * cur_5m_px * (1 - FEE)
                    trade_log.append({
                        'pnl_pct': cur_pnl_5m,
                        'regime': regime,
                        'exit': 'EMERG',
                    })
                    emerg_pnl_at_fire.append(cur_pnl_5m)
                    held = 0; in_pos = False; hold_since_entry = 0
                    emerg_cd_until = dt_5m + pd.Timedelta(minutes=emerg_cooldown_min)
                    last_emerg_dt = dt_5m
                    emerg_fires += 1
                    break  # one fire per hour

        # === Hourly logic continues only if still in_pos OR considering BUY ===
        if sig == 'BUY' and sconf >= conf_thr and not in_pos and active_cd <= 0:
            if hr_dt < emerg_cd_until:
                pass  # blocked by emergency cooldown
            else:
                held = cash * (1 - FEE) / price
                cash = 0; in_pos = True; entry_px = price; hold_since_entry = 0
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = shield_on and min_pnl > 0 and cur_pnl < min_pnl
            if not shield_blocks or override_expired:
                cash = held * price * (1 - FEE)
                trade_log.append({'pnl_pct': cur_pnl, 'regime': regime, 'exit': 'MODEL'})
                held = 0; in_pos = False; hold_since_entry = 0

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        cash = held * sigs[-1]['close'] * (1 - FEE)
        trade_log.append({'pnl_pct': (sigs[-1]['close'] / entry_px - 1) * 100,
                          'regime': 'bull', 'exit': 'OPEN'})

    ret = (cash / 1000.0 - 1) * 100
    n = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    wr = (wins / n * 100) if n else 0
    avg_emerg_pnl = float(np.mean(emerg_pnl_at_fire)) if emerg_pnl_at_fire else 0.0
    return ret, n, wr, emerg_fires, avg_emerg_pnl


def main():
    print("Loading...")
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']
    sigs = load_sigs()
    df_5m = load_combined_5m()
    print(f"  Hourly: {len(sigs)} sigs ({sigs[0]['datetime']} - {sigs[-1]['datetime']})")
    print(f"  5-min:  {len(df_5m)} bars ({df_5m.index[0]} - {df_5m.index[-1]})")
    ind5 = precompute_5m_indicators(df_5m)

    windows = {'30d': window_slice(sigs, 30),
               '60d': window_slice(sigs, 60),
               '90d': sigs}

    rows = []
    for w_name, w_sigs in windows.items():
        print(f"\n[{w_name}: {len(w_sigs)} sigs]")
        # Baseline (no emergency exit)
        b_ret, b_n, b_wr, _, _ = sim(w_sigs, asset_cfg, df_5m, ind5,
                                      emerg_enabled=False)
        print(f"  baseline: {b_ret:+.2f}% / {b_n} tr / WR {b_wr:.0f}%")

        tests = []
        # Threshold sweep at 15min lookback, 120min cooldown, all regimes, always bypass
        for thr in [-0.7, -1.0, -1.2, -1.5, -2.0]:
            tests.append((f'A.thr={thr} lb=15m cd=120 reg=all byp=always',
                          dict(emerg_lookback_min=15, emerg_threshold_pct=thr,
                               emerg_cooldown_min=120, emerg_regime='all',
                               emerg_bypass_only_if_profit=False)))
        # Lookback sweep at -1.0%
        for lb in [15, 20, 30]:
            tests.append((f'B.thr=-1.0 lb={lb}m cd=120 reg=all byp=always',
                          dict(emerg_lookback_min=lb, emerg_threshold_pct=-1.0,
                               emerg_cooldown_min=120, emerg_regime='all',
                               emerg_bypass_only_if_profit=False)))
        # Cooldown sweep at -1.0% / 15min
        for cd in [0, 60, 120, 180, 360]:
            tests.append((f'C.thr=-1.0 lb=15m cd={cd} reg=all byp=always',
                          dict(emerg_lookback_min=15, emerg_threshold_pct=-1.0,
                               emerg_cooldown_min=cd, emerg_regime='all',
                               emerg_bypass_only_if_profit=False)))
        # Regime gating
        for rg in ['bull', 'bear']:
            tests.append((f'D.thr=-1.0 lb=15m cd=120 reg={rg} byp=always',
                          dict(emerg_lookback_min=15, emerg_threshold_pct=-1.0,
                               emerg_cooldown_min=120, emerg_regime=rg,
                               emerg_bypass_only_if_profit=False)))
        # Bypass only if profit
        tests.append(('E.thr=-1.0 lb=15m cd=120 reg=all byp=if_profit',
                      dict(emerg_lookback_min=15, emerg_threshold_pct=-1.0,
                           emerg_cooldown_min=120, emerg_regime='all',
                           emerg_bypass_only_if_profit=True)))
        # Compound triggers
        for thr, dd60 in [(-1.0, -1.5), (-1.0, -2.0), (-1.2, -2.0), (-0.8, -1.5)]:
            tests.append((f'F.thr={thr} lb=15m + dd60<{dd60} cd=120 reg=all',
                          dict(emerg_lookback_min=15, emerg_threshold_pct=thr,
                               emerg_compound_dd60_pct=dd60,
                               emerg_cooldown_min=120, emerg_regime='all',
                               emerg_bypass_only_if_profit=False)))
        # G — armed only when current unrealized PnL >= X (protect rally gains)
        for arm_pnl in [1.0, 2.0, 3.0, 4.0, 5.0]:
            for thr in [-1.0, -1.5, -2.0]:
                tests.append((f'G.thr={thr} lb=15m armed_at_pnl>={arm_pnl}% cd=120',
                              dict(emerg_lookback_min=15, emerg_threshold_pct=thr,
                                   emerg_cooldown_min=120, emerg_regime='all',
                                   emerg_min_pnl_to_arm=arm_pnl)))
        # H — POST-RALLY give-back: -X% in 15m AFTER prior_h rally of >= prior_pct
        for prior_h in [1, 2, 4, 8, 12]:
            for prior_pct in [1.0, 2.0, 3.0, 4.0]:
                for thr in [-0.7, -1.0, -1.5]:
                    tests.append((f'H.thr={thr} lb=15m + prior_{prior_h}h>={prior_pct}% cd=120',
                                  dict(emerg_lookback_min=15, emerg_threshold_pct=thr,
                                       emerg_cooldown_min=120, emerg_regime='all',
                                       emerg_require_prior_rally_h=prior_h,
                                       emerg_require_prior_rally_pct=prior_pct)))
        # I — DERIVATIVE-BASED: prior_h rally (5-10h) >= +3%, recent drop (5-10m) <= Y,
        #     2nd derivative (acceleration) <= Z — all three required.
        for prior_h in [5, 6, 7, 8, 10]:
            for prior_pct in [2.0, 3.0, 4.0]:
                for lb_min in [5, 10]:
                    for drop_thr in [-0.3, -0.5, -0.7]:
                        for d2_thr in [None, -0.3, -0.5, -0.8]:
                            d2_lbl = f'd2<={d2_thr}' if d2_thr is not None else 'd2=OFF'
                            lbl = (f'I.prior_{prior_h}h>={prior_pct}% '
                                   f'+ drop_{lb_min}m<={drop_thr}% + {d2_lbl} cd=120')
                            tests.append((lbl,
                                          dict(emerg_lookback_min=lb_min,
                                               emerg_threshold_pct=drop_thr,
                                               emerg_cooldown_min=120,
                                               emerg_regime='all',
                                               emerg_require_prior_rally_h=prior_h,
                                               emerg_require_prior_rally_pct=prior_pct,
                                               emerg_require_d2_below=d2_thr)))

        for name, kwargs in tests:
            ret, n, wr, fires, avg_emerg_pnl = sim(
                w_sigs, asset_cfg, df_5m, ind5, emerg_enabled=True, **kwargs)
            rows.append({
                'window': w_name,
                'config': name,
                'return_pct': round(ret, 2),
                'delta': round(ret - b_ret, 2),
                'n_trades': n,
                'win_rate': round(wr, 1),
                'emerg_fires': fires,
                'avg_emerg_pnl_pct': round(avg_emerg_pnl, 2),
            })

    df = pd.DataFrame(rows)
    pivot = df.pivot(index='config', columns='window', values='delta')
    pivot.columns = [f'd_{c}' for c in pivot.columns]
    pivot_n = df.pivot(index='config', columns='window', values='n_trades')
    pivot_n.columns = [f'n_{c}' for c in pivot_n.columns]
    pivot_f = df.pivot(index='config', columns='window', values='emerg_fires')
    pivot_f.columns = [f'fires_{c}' for c in pivot_f.columns]
    full = pd.concat([pivot, pivot_n, pivot_f], axis=1).sort_values('d_60d', ascending=False)

    print(f"\n{'='*130}")
    print(f"  EMERGENCY-EXIT 5m OVERLAY -- sorted by 60d delta vs no-emerg baseline")
    print(f"{'='*130}")
    print(full.to_string())

    print(f"\n{'='*130}")
    print(f"  POSITIVE ON ALL 3 WINDOWS")
    print(f"{'='*130}")
    pos = full[(full['d_30d'] > 0) & (full['d_60d'] > 0) & (full['d_90d'] > 0)]
    print(pos.to_string() if not pos.empty else "  None.")

    out = os.path.join(ENGINE, 'output',
                       f'emerg_exit_5m_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
