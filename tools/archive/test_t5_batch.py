"""
test_t5_batch.py — 10 untested ideas, all standalone, read-only.
Default window: 60d (canonical). Also reports 30d/90d for context.

T5a. ASYMMETRIC EXIT HORIZON (PROXY) — entry uses bull_conf=90 (current
     6h@90% winner). For SELL after entry, raise required SELL conf to
     {85, 90, 95}. Slower exit ≈ longer effective horizon. (True multi-
     horizon swap requires regenerating signals at a different h, ~3hr.)

T5b. PER-REGIME T2c THRESHOLD SWEEP — refine T2c+bull_conf=90 winner by
     sweeping bull_dd and bear_dd independently in {1, 2, 3, 4, 5}.

T5c. TRAILING-PROFIT LOCK (peak-relative give-back) — once unrealized PnL
     reaches +X%, force SELL if PnL falls back to +X×retain%. Tests
     {peak=2%/retain=0.5}, {peak=3%/retain=0.5}, {peak=4%/retain=0.6},
     {peak=5%/retain=0.6}, {peak=2%/retain=0.7}.

T5d. PER-REGIME min_sell_pnl — sweep bull_min_pnl ∈ {0.3, 0.5, 0.7, 1.0}
     × bear_min_pnl ∈ {0.3, 0.5, 1.0, 1.5}. Currently 0.5/0.5 shared.

T5e. RALLY-MOMENTUM EXIT OVERRIDE — if rr_K hours ≥ X% within the last
     M hours of being in position, force SELL bypass shield. Mirror of
     bull rally-cooldown but on EXIT side. Tests rr_8h≥{3,4,5}, rr_12h≥
     {4,5,6}.

T5f. ENTRY ON MULTIPLE-DAYS-DOWN — only BUY if last 3 daily closes are
     each lower than 3 days ago (deeper pullback). Tests {2 of 3, 3 of 3}.

T5g. CONF-WEIGHTED MAX_HOLD — extend max_hold from 10h to 14h, BUT only
     if entry_conf ≥ 95%. Tests {extend by 4h@95%, 6h@95%, 4h@90%}.

T5h. SHIELD AUTO-OFF AT HIGH PROFIT — disable shield when in_pos PnL ≥
     X%. (Lock in your wins.) Tests X ∈ {3, 5, 7, 10}.

T5i. ENTRY VOLATILITY GATE — block BUY if 24h realized vol > Yth pctile
     of 30d distribution (avoid entering during chaos). Tests Y ∈
     {70, 80, 90, 95}.

T5j. SELL-CONF DECAY — accept SELL at lower conf as time-in-position
     grows. Tests: at h>=4, accept SELL if conf >= base_conf - {5pp,10pp,
     15pp}.

Reads: cache + regime_config + eth_hourly. Writes output/t5_batch_*.csv.
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


def precompute_context(sigs, eth_df):
    """For each signal compute: dd_from_7d_high, vol_24h_pctile_30d,
    days_down_count_3, eth_24h_ret."""
    n = len(sigs)
    dd_from_7d = np.full(n, np.nan)
    vol_pctile = np.full(n, np.nan)
    days_down_count = np.zeros(n, dtype=int)
    eth_24h = np.full(n, np.nan)

    eth_close = eth_df['close']
    eth_high = eth_df['high']

    # Daily closes resampled
    daily = eth_close.resample('1D').last().dropna()

    for i, s in enumerate(sigs):
        dt = s['datetime']
        if dt.tzinfo is None:
            dt = dt.tz_localize('UTC')
        else:
            dt = dt.tz_convert('UTC')

        sub24 = eth_close[(eth_close.index < dt) & (eth_close.index >= dt - timedelta(hours=24))]
        if len(sub24) > 4:
            eth_24h[i] = 100 * (sub24.iloc[-1] / sub24.iloc[0] - 1)

        sub7d = eth_high[(eth_high.index < dt) & (eth_high.index >= dt - timedelta(days=7))]
        if len(sub7d) > 24 and len(sub24) > 0:
            high_7d = sub7d.max()
            cur_close = sub24.iloc[-1]
            dd_from_7d[i] = 100 * (cur_close / high_7d - 1)

        # 24h realized vol percentile vs 30d
        win30d = eth_close[(eth_close.index < dt) & (eth_close.index >= dt - timedelta(days=30))]
        if len(win30d) > 100:
            rets = win30d.pct_change().dropna()
            # rolling 24h std
            rolling_vol = rets.rolling(24).std().dropna()
            cur_vol = rets.tail(24).std()
            if len(rolling_vol) > 10 and not np.isnan(cur_vol):
                pctile = (rolling_vol < cur_vol).sum() / len(rolling_vol) * 100
                vol_pctile[i] = pctile

        # Last 3 daily closes vs the day before
        days_priorto = daily[daily.index < dt]
        if len(days_priorto) >= 4:
            last4 = days_priorto.iloc[-4:].values
            cnt = 0
            for k in range(1, 4):
                if last4[k] < last4[k - 1]:
                    cnt += 1
            days_down_count[i] = cnt

    return dd_from_7d, vol_pctile, days_down_count, eth_24h


def sim(sigs, asset_cfg, dd_arr, vol_pctile, days_down, eth_24h_arr,
        # T5a
        sell_conf_uplift_pp=0,
        # T5b
        bull_dd_min=None, bear_dd_min=None,
        # T5c
        trailing_peak_pct=None, trailing_retain_pct=None,
        # T5d
        bull_min_pnl_override=None, bear_min_pnl_override=None,
        # T5e
        rally_exit_h=None, rally_exit_pct=None,
        # T5f
        require_days_down_min=None,
        # T5g
        max_hold_extend_h=0, max_hold_extend_min_conf=0,
        # T5h
        shield_off_above_pnl=None,
        # T5i
        block_above_vol_pctile=None,
        # T5j: at hold>=4, accept SELL even if conf below threshold by N pp
        sell_conf_decay_pp=0,
        # T5b convenience: also bull_conf override
        bull_conf=None, bear_conf=None):

    bull_conf_thr = bull_conf if bull_conf is not None else float(asset_cfg.get('bull', {}).get('min_confidence', 80))
    bear_conf_thr = bear_conf if bear_conf is not None else float(asset_cfg.get('bear', {}).get('min_confidence', 65))
    bull_shield = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    base_min_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.5))
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

    # T5e — rally-exit indicator
    rally_exit_arr = _rr(rally_exit_h) if rally_exit_h else None

    cash = 1000.0; held = 0.0; in_pos = False; entry_px = 0.0
    entry_regime = None; entry_conf = 0.0
    hold_since_entry = 0; bull_cd = bear_cd = 0
    peak_pnl = 0.0
    trade_log = []
    blocked = 0
    trailing_fires = 0; rally_exit_fires = 0; shield_off_fires = 0

    for i, s in enumerate(sigs):
        price = float(s['close'])
        regime = s.get('regime', 'bull')
        sig = s['signal']
        sconf = float(s.get('confidence', 0))
        conf_thr = bull_conf_thr if regime == 'bull' else bear_conf_thr
        shield_on = bull_shield if regime == 'bull' else bear_shield

        # T5d — per-regime min_sell_pnl override
        if regime == 'bull' and bull_min_pnl_override is not None:
            min_pnl = bull_min_pnl_override
        elif regime == 'bear' and bear_min_pnl_override is not None:
            min_pnl = bear_min_pnl_override
        else:
            min_pnl = base_min_pnl

        # T5g — max_hold extension
        eff_max_hold = max_hold_h
        if max_hold_extend_h > 0 and entry_conf >= max_hold_extend_min_conf:
            eff_max_hold = max_hold_h + max_hold_extend_h

        if in_pos:
            hold_since_entry += 1
            cur_pnl = (price / entry_px - 1) * 100
            peak_pnl = max(peak_pnl, cur_pnl)

            # T5h — shield auto-off at high profit
            local_shield_on = shield_on
            if shield_off_above_pnl is not None and cur_pnl >= shield_off_above_pnl:
                local_shield_on = False
                shield_off_fires += 1

            # T5c — trailing profit lock
            if trailing_peak_pct is not None and trailing_retain_pct is not None:
                if peak_pnl >= trailing_peak_pct:
                    floor_pnl = peak_pnl * trailing_retain_pct
                    if cur_pnl <= floor_pnl:
                        cash = held * price * (1 - FEE)
                        trade_log.append({'pnl_pct': cur_pnl, 'exit': 'TRAILING_LOCK',
                                          'regime': regime})
                        held = 0; in_pos = False; hold_since_entry = 0
                        peak_pnl = 0; entry_conf = 0
                        trailing_fires += 1
                        if bull_cd > 0: bull_cd -= 1
                        if bear_cd > 0: bear_cd -= 1
                        continue

            # T5e — rally-momentum exit (only fires while in_pos within bound)
            if rally_exit_arr is not None and rally_exit_pct is not None:
                rr = rally_exit_arr[i]
                if not np.isnan(rr) and rr >= rally_exit_pct:
                    cash = held * price * (1 - FEE)
                    trade_log.append({'pnl_pct': cur_pnl, 'exit': 'RALLY_EXIT',
                                      'regime': regime})
                    held = 0; in_pos = False; hold_since_entry = 0
                    peak_pnl = 0; entry_conf = 0
                    rally_exit_fires += 1
                    if bull_cd > 0: bull_cd -= 1
                    if bear_cd > 0: bear_cd -= 1
                    continue
        else:
            local_shield_on = shield_on

        # cooldown trigger updates
        if bcd_h > 0 and bull_rs is not None:
            rs = bull_rs[i] if not np.isnan(bull_rs[i]) else 0
            rl = bull_rl[i] if not np.isnan(bull_rl[i]) else 0
            if rs >= bt_s or rl >= bt_l: bull_cd = max(bull_cd, bcd_h)
        if rcd_h > 0 and bear_rs is not None:
            rs = bear_rs[i] if not np.isnan(bear_rs[i]) else 0
            rl = bear_rl[i] if not np.isnan(bear_rl[i]) else 0
            if rs >= rt_s or rl >= rt_l: bear_cd = max(bear_cd, rcd_h)
        active_cd = bull_cd if regime == 'bull' else bear_cd

        # ENTRY
        if sig == 'BUY' and sconf >= conf_thr and not in_pos and active_cd <= 0:
            block = False
            # T5b per-regime dd
            if regime == 'bull' and bull_dd_min is not None:
                dd = dd_arr[i]
                if not np.isnan(dd) and dd > -abs(bull_dd_min):
                    block = True
            if not block and regime == 'bear' and bear_dd_min is not None:
                dd = dd_arr[i]
                if not np.isnan(dd) and dd > -abs(bear_dd_min):
                    block = True
            # T5f require N of 3 days down
            if not block and require_days_down_min is not None:
                if days_down[i] < require_days_down_min:
                    block = True
            # T5i — block above volatility percentile
            if not block and block_above_vol_pctile is not None:
                vp = vol_pctile[i]
                if not np.isnan(vp) and vp > block_above_vol_pctile:
                    block = True

            if block:
                blocked += 1
            else:
                held = cash * (1 - FEE) / price
                cash = 0; in_pos = True; entry_px = price
                entry_regime = regime; entry_conf = sconf
                hold_since_entry = 0; peak_pnl = 0.0

        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= eff_max_hold

            # T5a — uplift required SELL conf above base
            sell_conf_thr = conf_thr + sell_conf_uplift_pp

            # T5j — accept lower SELL conf as hold grows
            if sell_conf_decay_pp > 0 and hold_since_entry >= 4:
                sell_conf_thr = max(0, sell_conf_thr - sell_conf_decay_pp)

            sell_conf_ok = sconf >= sell_conf_thr

            shield_blocks = local_shield_on and min_pnl > 0 and cur_pnl < min_pnl

            if sell_conf_ok and (not shield_blocks or override_expired):
                cash = held * price * (1 - FEE)
                trade_log.append({'pnl_pct': cur_pnl, 'exit': 'MODEL', 'regime': regime})
                held = 0; in_pos = False; hold_since_entry = 0
                peak_pnl = 0; entry_conf = 0

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        cash = held * sigs[-1]['close'] * (1 - FEE)
        trade_log.append({'pnl_pct': (sigs[-1]['close'] / entry_px - 1) * 100,
                          'exit': 'OPEN_AT_END', 'regime': entry_regime or 'bull'})

    ret = (cash / 1000.0 - 1) * 100
    n = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    wr = (wins / n * 100) if n else 0
    return ret, n, wr, blocked, trailing_fires, rally_exit_fires, shield_off_fires


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    print("Loading data...")
    sigs = load_sigs()
    eth = pd.read_csv(ETH_HOURLY)
    eth['datetime'] = pd.to_datetime(eth['datetime'], utc=True)
    eth = eth.set_index('datetime').sort_index()

    windows = {'30d': window_slice(sigs, 30),
               '60d': window_slice(sigs, 60),
               '90d': sigs}

    all_rows = []
    for win_name, win_sigs in windows.items():
        print(f"\n[Window: {win_name}, {len(win_sigs)} signals]")
        print("Precomputing context...")
        dd_arr, vol_pctile, days_down, eth_24h = precompute_context(win_sigs, eth)

        # Baseline
        base_ret, base_n, base_wr, _, _, _, _ = sim(win_sigs, asset_cfg,
                                                    dd_arr, vol_pctile, days_down, eth_24h)
        print(f"  BASELINE: {base_ret:+.2f}% / {base_n} trades / WR {base_wr:.0f}%")

        tests = []
        # T5a sell-conf uplift (asymmetric exit)
        for u in [5, 10, 15]:
            tests.append((f'T5a sell_uplift={u}pp (bull_conf=90)',
                          dict(bull_conf=90, sell_conf_uplift_pp=u)))
        # T5b per-regime dd sweep (with bull_conf=90 already winner)
        for bull_dd in [2, 3, 4, 5]:
            for bear_dd in [None, 2, 3, 5]:
                lbl = f'T5b bull_dd={bull_dd} bear_dd={bear_dd if bear_dd is not None else "OFF"} (bull_conf=90)'
                tests.append((lbl, dict(bull_conf=90, bull_dd_min=bull_dd, bear_dd_min=bear_dd)))
        # T5c trailing profit lock
        for peak, retain in [(2.0, 0.5), (3.0, 0.5), (4.0, 0.6), (5.0, 0.6),
                             (2.0, 0.7), (3.0, 0.7)]:
            tests.append((f'T5c trail peak={peak}% retain={int(retain*100)}%',
                          dict(trailing_peak_pct=peak, trailing_retain_pct=retain)))
        # T5d per-regime min_sell_pnl
        for bp in [0.3, 0.5, 0.7, 1.0]:
            for rp in [0.3, 0.5, 1.0, 1.5]:
                if bp == 0.5 and rp == 0.5:
                    continue  # baseline
                tests.append((f'T5d bull_pnl={bp}% bear_pnl={rp}%',
                              dict(bull_min_pnl_override=bp, bear_min_pnl_override=rp)))
        # T5e rally-momentum exit
        for h, t in [(8, 3.0), (8, 4.0), (8, 5.0), (12, 4.0), (12, 5.0), (12, 6.0)]:
            tests.append((f'T5e rally_exit rr{h}h>={t}%',
                          dict(rally_exit_h=h, rally_exit_pct=t)))
        # T5f days-down
        for n in [2, 3]:
            tests.append((f'T5f days_down>={n}', dict(require_days_down_min=n)))
        # T5g extend max_hold conditional on entry_conf
        for ext, mc in [(4, 95), (6, 95), (4, 90), (8, 95)]:
            tests.append((f'T5g extend_h={ext} entry_conf>={mc}',
                          dict(max_hold_extend_h=ext, max_hold_extend_min_conf=mc)))
        # T5h shield auto-off at profit
        for x in [3, 5, 7, 10]:
            tests.append((f'T5h shield_off_above_pnl={x}%', dict(shield_off_above_pnl=x)))
        # T5i vol gate
        for y in [70, 80, 90, 95]:
            tests.append((f'T5i block_above_vol_pctile={y}', dict(block_above_vol_pctile=y)))
        # T5j sell-conf decay
        for d in [5, 10, 15]:
            tests.append((f'T5j sell_decay@h>=4 = -{d}pp (bull_conf=90)',
                          dict(bull_conf=90, sell_conf_decay_pp=d)))
        # Big combo: T5b winner candidates + T5h
        tests.append(('combo: bull_conf=90 + bull_dd=3 + shield_off>=5%',
                      dict(bull_conf=90, bull_dd_min=3, shield_off_above_pnl=5)))
        tests.append(('combo: bull_conf=90 + bull_dd=3 + trail peak=4%/r=60%',
                      dict(bull_conf=90, bull_dd_min=3, trailing_peak_pct=4, trailing_retain_pct=0.6)))
        tests.append(('combo: bull_conf=90 + bull_dd=3 + rally_exit rr12h>=5%',
                      dict(bull_conf=90, bull_dd_min=3, rally_exit_h=12, rally_exit_pct=5)))

        for name, kwargs in tests:
            ret, n, wr, blk, tfires, refires, sofires = sim(
                win_sigs, asset_cfg, dd_arr, vol_pctile, days_down, eth_24h, **kwargs)
            all_rows.append({
                'window': win_name,
                'test': name,
                'return_pct': round(ret, 2),
                'delta': round(ret - base_ret, 2),
                'n_trades': n,
                'win_rate': round(wr, 1),
                'blocked': blk,
                'trailing_fires': tfires,
                'rally_exit_fires': refires,
                'shield_off_fires': sofires,
            })

    df = pd.DataFrame(all_rows)

    # Pivot to compare windows side-by-side
    pivot_d = df.pivot(index='test', columns='window', values='delta')
    pivot_d.columns = [f'd_{c}' for c in pivot_d.columns]
    pivot_n = df.pivot(index='test', columns='window', values='n_trades')
    pivot_n.columns = [f'n_{c}' for c in pivot_n.columns]
    full = pd.concat([pivot_d, pivot_n], axis=1)
    full = full.sort_values('d_60d', ascending=False)

    print(f"\n{'='*120}")
    print(f"  T5 RESULTS — sorted by 60d delta (canonical)")
    print(f"{'='*120}")
    print(full.to_string())

    # Positive-on-all-three filter
    pos_all = full[(full['d_30d'] > 0) & (full['d_60d'] > 0) & (full['d_90d'] > 0)]
    print(f"\n{'='*120}")
    print(f"  POSITIVE ON ALL 3 WINDOWS ({len(pos_all)} configs)")
    print(f"{'='*120}")
    if pos_all.empty:
        print("  None.")
    else:
        print(pos_all.head(20).to_string())

    out = os.path.join(ENGINE, 'output',
                       f't5_batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nFull results saved to: {out}")


if __name__ == '__main__':
    main()
