"""
max_hold_pattern_scan.py — for the 21 MAX_HOLD-failsafe trades, scan for
patterns at ENTRY time that might predict the failure. Tests several
hypotheses cheaply, all reading from existing data + signal cache only.

Hypotheses tested:
  H1: MAX_HOLD trades have lower entry confidence than MODEL exits
  H2: MAX_HOLD trades cluster at specific times of day (UTC hour)
  H3: MAX_HOLD trades happen in higher-vol regimes than MODEL exits
  H4: MAX_HOLD trades are preceded by losing-streak (last 3 trades all losses)
  H5: MAX_HOLD trades happen after a recent rally (entry into reversal)
  H6: BTC was trending DOWN at the entry hour (leader-lag warning)
  H7: ETH 15-min vol-of-vol was elevated at entry (chop signal)

For each hypothesis, compare MAX_HOLD trade distribution vs MODEL trade
distribution. If a hypothesis is real, the two distributions should differ.
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import timedelta

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PKL = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
REGIME_CFG = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
ETH_HOURLY = os.path.join(ENGINE, 'data', 'eth_hourly_data.csv')
BTC_HOURLY = os.path.join(ENGINE, 'data', 'btc_hourly_data.csv')
ETH_15M = os.path.join(ENGINE, 'data', 'eth_15m_data.csv')
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


def find_all_trades(sigs, asset_cfg):
    """Re-run the regime-switched simulator and return ALL trades (MODEL +
    MAX_HOLD) with entry context recorded."""
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
    entry_dt = None; entry_regime = None; entry_conf = 0
    hold_since_entry = 0; bull_cd = bear_cd = 0
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
            held = cash * (1 - FEE) / price
            cash = 0; in_pos = True; entry_px = price
            entry_dt = s['datetime']; entry_regime = regime
            entry_conf = sconf
            hold_since_entry = 0
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = shield_on and min_pnl > 0 and cur_pnl < min_pnl
            if not shield_blocks or override_expired:
                exit_reason = 'MAX_HOLD' if (shield_on and override_expired and cur_pnl < min_pnl) else 'MODEL'
                cash = held * price * (1 - FEE)
                trade_log.append({
                    'entry_dt': entry_dt, 'exit_dt': s['datetime'],
                    'entry_px': entry_px, 'exit_px': price,
                    'pnl_pct': cur_pnl, 'hold_h': hold_since_entry,
                    'regime': entry_regime, 'entry_conf': entry_conf,
                    'exit_reason': exit_reason,
                })
                held = 0; in_pos = False; hold_since_entry = 0

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    return trade_log


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    sigs = load_sigs()
    win = window_slice(sigs, 60)
    trades = find_all_trades(win, asset_cfg)

    df = pd.DataFrame(trades)
    df['entry_dt'] = pd.to_datetime(df['entry_dt'], utc=True)
    df['exit_dt'] = pd.to_datetime(df['exit_dt'], utc=True)
    print(f"\n60d trades: {len(df)} total | "
          f"MAX_HOLD: {(df['exit_reason']=='MAX_HOLD').sum()} | "
          f"MODEL: {(df['exit_reason']=='MODEL').sum()}\n")

    mh = df[df['exit_reason'] == 'MAX_HOLD'].copy()
    mod = df[df['exit_reason'] == 'MODEL'].copy()

    # Load OHLCV for context
    eth = pd.read_csv(ETH_HOURLY)
    eth['datetime'] = pd.to_datetime(eth['datetime'], utc=True)
    eth = eth.set_index('datetime').sort_index()
    btc = pd.read_csv(BTC_HOURLY) if os.path.exists(BTC_HOURLY) else None
    if btc is not None:
        btc['datetime'] = pd.to_datetime(btc['datetime'], utc=True)
        btc = btc.set_index('datetime').sort_index()
    eth_15m = pd.read_csv(ETH_15M)
    eth_15m['datetime'] = pd.to_datetime(eth_15m['datetime'], utc=True)
    eth_15m = eth_15m.set_index('datetime').sort_index()

    # Compute per-trade features at ENTRY
    def entry_features(trades_df):
        rows = []
        for _, t in trades_df.iterrows():
            ed = t['entry_dt']
            row = {'entry_dt': ed, 'pnl_pct': t['pnl_pct'],
                   'entry_conf': t['entry_conf'], 'regime': t['regime']}
            # H2: hour of day
            row['utc_hour'] = ed.hour
            # H3: ETH realized vol over last 24h vs 30d (entry-time)
            eth_24h = eth[(eth.index < ed) & (eth.index >= ed - timedelta(hours=24))]['close']
            eth_30d = eth[(eth.index < ed) & (eth.index >= ed - timedelta(days=30))]['close']
            if len(eth_24h) > 4 and len(eth_30d) > 100:
                vol_24h = np.log(eth_24h).diff().std() * np.sqrt(24)
                vol_30d = np.log(eth_30d).diff().std() * np.sqrt(24)
                row['vol_ratio_24h_30d'] = vol_24h / vol_30d if vol_30d > 0 else np.nan
            else:
                row['vol_ratio_24h_30d'] = np.nan
            # H5: ETH 24h return into entry
            if len(eth_24h) > 4:
                row['ret_24h_into_entry'] = 100 * (eth_24h.iloc[-1] / eth_24h.iloc[0] - 1)
            else:
                row['ret_24h_into_entry'] = np.nan
            # H6: BTC 24h return into entry
            if btc is not None:
                btc_24h = btc[(btc.index < ed) & (btc.index >= ed - timedelta(hours=24))]['close']
                if len(btc_24h) > 4:
                    row['btc_ret_24h_into_entry'] = 100 * (btc_24h.iloc[-1] / btc_24h.iloc[0] - 1)
                else:
                    row['btc_ret_24h_into_entry'] = np.nan
            else:
                row['btc_ret_24h_into_entry'] = np.nan
            # H7: ETH 15-min vol-of-vol over last 4h
            sub = eth_15m[(eth_15m.index >= ed - timedelta(hours=4)) & (eth_15m.index < ed)]
            if len(sub) > 8:
                rets = np.log(sub['close']).diff().dropna()
                # vol of vol = std of rolling-window stds
                if len(rets) > 8:
                    rolling_std = rets.rolling(window=4).std().dropna()
                    if len(rolling_std) > 2:
                        row['vov_15m_4h'] = rolling_std.std()
                    else:
                        row['vov_15m_4h'] = np.nan
                else:
                    row['vov_15m_4h'] = np.nan
            else:
                row['vov_15m_4h'] = np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    print("Computing entry features for all trades...")
    feat_mh = entry_features(mh)
    feat_mod = entry_features(mod)

    print("\n" + "=" * 100)
    print("  H1: Entry confidence — do MAX_HOLD trades have lower confidence?")
    print("=" * 100)
    print(f"  MAX_HOLD median entry conf: {feat_mh['entry_conf'].median():.1f}%, "
          f"mean: {feat_mh['entry_conf'].mean():.1f}%")
    print(f"  MODEL    median entry conf: {feat_mod['entry_conf'].median():.1f}%, "
          f"mean: {feat_mod['entry_conf'].mean():.1f}%")
    print(f"  Distribution of MAX_HOLD entry conf bins:")
    bins = [0, 80, 85, 90, 95, 100]
    for lo, hi in zip(bins[:-1], bins[1:]):
        n_mh = ((feat_mh['entry_conf'] >= lo) & (feat_mh['entry_conf'] < hi)).sum()
        n_mod = ((feat_mod['entry_conf'] >= lo) & (feat_mod['entry_conf'] < hi)).sum()
        rate = n_mh / (n_mh + n_mod) if (n_mh + n_mod) else 0
        print(f"    [{lo:>3}-{hi:>3}%): MAX_HOLD={n_mh:>2}  MODEL={n_mod:>2}  "
              f"failsafe rate={100*rate:.0f}%")

    print("\n" + "=" * 100)
    print("  H2: Time of day — do MAX_HOLD trades cluster at specific UTC hours?")
    print("=" * 100)
    mh_hours = feat_mh['utc_hour'].value_counts().sort_index()
    mod_hours = feat_mod['utc_hour'].value_counts().sort_index()
    print(f"  MAX_HOLD entry hours (UTC): {dict(mh_hours)}")
    print(f"  MODEL entry hours (UTC):    {dict(mod_hours)}")

    print("\n" + "=" * 100)
    print("  H3: Vol regime — are MAX_HOLD trades entered in higher vol?")
    print("=" * 100)
    print(f"  MAX_HOLD median vol_24h/vol_30d: {feat_mh['vol_ratio_24h_30d'].median():.2f}, "
          f"mean: {feat_mh['vol_ratio_24h_30d'].mean():.2f}")
    print(f"  MODEL    median vol_24h/vol_30d: {feat_mod['vol_ratio_24h_30d'].median():.2f}, "
          f"mean: {feat_mod['vol_ratio_24h_30d'].mean():.2f}")

    print("\n" + "=" * 100)
    print("  H5: 24h ETH return INTO entry — do MAX_HOLD trades enter into recent rallies?")
    print("=" * 100)
    print(f"  MAX_HOLD median 24h ETH return: {feat_mh['ret_24h_into_entry'].median():+.2f}%, "
          f"mean: {feat_mh['ret_24h_into_entry'].mean():+.2f}%")
    print(f"  MODEL    median 24h ETH return: {feat_mod['ret_24h_into_entry'].median():+.2f}%, "
          f"mean: {feat_mod['ret_24h_into_entry'].mean():+.2f}%")

    print("\n" + "=" * 100)
    print("  H6: BTC 24h return INTO entry — was BTC trending opposite at entry?")
    print("=" * 100)
    if not feat_mh['btc_ret_24h_into_entry'].isna().all():
        print(f"  MAX_HOLD median BTC 24h return: "
              f"{feat_mh['btc_ret_24h_into_entry'].median():+.2f}%, "
              f"mean: {feat_mh['btc_ret_24h_into_entry'].mean():+.2f}%")
        print(f"  MODEL    median BTC 24h return: "
              f"{feat_mod['btc_ret_24h_into_entry'].median():+.2f}%, "
              f"mean: {feat_mod['btc_ret_24h_into_entry'].mean():+.2f}%")
    else:
        print("  No BTC data available")

    print("\n" + "=" * 100)
    print("  H7: 15-min vol-of-vol over last 4h before entry — chop signal?")
    print("=" * 100)
    print(f"  MAX_HOLD median vov_15m_4h: "
          f"{feat_mh['vov_15m_4h'].median():.5f}")
    print(f"  MODEL    median vov_15m_4h: "
          f"{feat_mod['vov_15m_4h'].median():.5f}")

    print("\n" + "=" * 100)
    print("  H4: Loss-streak — does MAX_HOLD follow a string of losses?")
    print("=" * 100)
    df_sorted = df.sort_values('entry_dt').reset_index(drop=True)
    df_sorted['prev_pnl_1'] = df_sorted['pnl_pct'].shift(1)
    df_sorted['prev_pnl_2'] = df_sorted['pnl_pct'].shift(2)
    df_sorted['prev_pnl_3'] = df_sorted['pnl_pct'].shift(3)
    df_sorted['prev_3_losers'] = (
        (df_sorted['prev_pnl_1'] <= 0) &
        (df_sorted['prev_pnl_2'] <= 0) &
        (df_sorted['prev_pnl_3'] <= 0)
    ).fillna(False)
    n_mh_after_streak = ((df_sorted['exit_reason'] == 'MAX_HOLD') &
                         df_sorted['prev_3_losers']).sum()
    n_mh_total = (df_sorted['exit_reason'] == 'MAX_HOLD').sum()
    n_streak_total = df_sorted['prev_3_losers'].sum()
    print(f"  MAX_HOLD trades that followed 3 prior losses: "
          f"{n_mh_after_streak}/{n_mh_total} = "
          f"{100*n_mh_after_streak/n_mh_total if n_mh_total else 0:.0f}%")
    print(f"  All trades that followed 3 prior losses: {n_streak_total}/{len(df_sorted)} = "
          f"{100*n_streak_total/len(df_sorted):.0f}%")

    print("\n" + "=" * 100)
    print("  H_REGIME: Do MAX_HOLD trades cluster in bear regime?")
    print("=" * 100)
    print(f"  MAX_HOLD by regime: {dict(mh['regime'].value_counts())}")
    print(f"  MODEL    by regime: {dict(mod['regime'].value_counts())}")
    bear_mh = (mh['regime'] == 'bear').sum()
    bear_mod = (mod['regime'] == 'bear').sum()
    print(f"  Bear failsafe rate: {bear_mh}/{bear_mh + bear_mod} = "
          f"{100*bear_mh/(bear_mh+bear_mod) if (bear_mh+bear_mod) else 0:.0f}%")
    bull_mh = (mh['regime'] == 'bull').sum()
    bull_mod = (mod['regime'] == 'bull').sum()
    print(f"  Bull failsafe rate: {bull_mh}/{bull_mh + bull_mod} = "
          f"{100*bull_mh/(bull_mh+bull_mod) if (bull_mh+bull_mod) else 0:.0f}%")


if __name__ == '__main__':
    main()
