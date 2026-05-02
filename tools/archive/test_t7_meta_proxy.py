"""
test_t7_meta_proxy.py — meta-labeling PROXY (Lopez de Prado Ch. 3).

True meta-labeling trains a secondary model to predict whether the primary
model's BUY signal will be correct. Full implementation lives in
crypto_trading_system_meta.py. Here we test a CHEAP PROXY: build features
from the cached signal stream itself (without retraining anything) and
filter BUYs through a logistic regression that's trained walk-forward on
realized trade outcomes.

Features per BUY candidate (all available without retraining):
  - signal confidence
  - 24h ETH return
  - 7d drawdown from high
  - hours since last SELL
  - regime (bull=1, bear=0)
  - confidence-percentile-rank over last 7d
  - 24h realized vol percentile (vs 30d)

Walk-forward: train on past N trades, predict next N, slide.
Filter: drop BUY if predicted P(profitable) < threshold.

Reads: cache + regime_config + eth_hourly. Writes output/t7_meta_proxy_*.csv.
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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


def precompute_meta_features(sigs, eth_df):
    """For every signal, compute meta features."""
    n = len(sigs)
    eth_close = eth_df['close']
    eth_high = eth_df['high']

    eth_24h = np.full(n, np.nan)
    dd_7d = np.full(n, np.nan)
    vol_pctile = np.full(n, np.nan)
    conf_pctile_7d = np.full(n, np.nan)

    confs = np.array([float(s.get('confidence', 0)) for s in sigs])

    for i, s in enumerate(sigs):
        dt = s['datetime']
        if dt.tzinfo is None: dt = dt.tz_localize('UTC')
        else: dt = dt.tz_convert('UTC')

        sub24 = eth_close[(eth_close.index < dt) & (eth_close.index >= dt - timedelta(hours=24))]
        if len(sub24) > 4:
            eth_24h[i] = 100 * (sub24.iloc[-1] / sub24.iloc[0] - 1)

        sub7d = eth_high[(eth_high.index < dt) & (eth_high.index >= dt - timedelta(days=7))]
        if len(sub7d) > 24 and len(sub24) > 0:
            dd_7d[i] = 100 * (sub24.iloc[-1] / sub7d.max() - 1)

        win30d = eth_close[(eth_close.index < dt) & (eth_close.index >= dt - timedelta(days=30))]
        if len(win30d) > 100:
            rets = win30d.pct_change().dropna()
            rolling_vol = rets.rolling(24).std().dropna()
            cur_vol = rets.tail(24).std()
            if len(rolling_vol) > 10 and not np.isnan(cur_vol):
                vol_pctile[i] = (rolling_vol < cur_vol).sum() / len(rolling_vol) * 100

        # confidence percentile rank over last 7d (168 hours)
        lo_idx = max(0, i - 168)
        prior_confs = confs[lo_idx:i]
        if len(prior_confs) > 24:
            conf_pctile_7d[i] = (prior_confs < confs[i]).sum() / len(prior_confs) * 100

    return eth_24h, dd_7d, vol_pctile, conf_pctile_7d


def collect_buy_outcomes(sigs, asset_cfg, eth_24h, dd_7d, vol_pctile, conf_pctile,
                          bull_conf=None):
    """Run baseline strategy, record meta-features at each BUY + outcome."""
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
    last_sell_idx = -10**9
    buys = []  # list of (idx, features dict, outcome)
    open_buy_idx = None

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
            features = {
                'conf': sconf,
                'eth_24h': eth_24h[i] if not np.isnan(eth_24h[i]) else 0,
                'dd_7d': dd_7d[i] if not np.isnan(dd_7d[i]) else 0,
                'vol_pctile': vol_pctile[i] if not np.isnan(vol_pctile[i]) else 50,
                'conf_pctile_7d': conf_pctile[i] if not np.isnan(conf_pctile[i]) else 50,
                'regime_bull': 1 if regime == 'bull' else 0,
                'hours_since_sell': min(168, i - last_sell_idx),
            }
            in_pos = True; entry_px = price; hold_since_entry = 0
            open_buy_idx = len(buys)
            buys.append({'idx': i, 'features': features, 'pnl_pct': None})
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = shield_on and min_pnl > 0 and cur_pnl < min_pnl
            if not shield_blocks or override_expired:
                if open_buy_idx is not None:
                    buys[open_buy_idx]['pnl_pct'] = cur_pnl
                    open_buy_idx = None
                in_pos = False; hold_since_entry = 0
                last_sell_idx = i

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos and open_buy_idx is not None:
        buys[open_buy_idx]['pnl_pct'] = (sigs[-1]['close'] / entry_px - 1) * 100

    return buys


def sim_with_meta(sigs, asset_cfg, eth_24h, dd_7d, vol_pctile, conf_pctile,
                  meta_train_n, meta_threshold, bull_conf=None):
    """Walk-forward meta filter: train on past N trades, filter next ones."""
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
    last_sell_idx = -10**9
    trade_log = []
    closed_trades = []  # historical features+outcome buffer for meta training
    blocked = 0

    feat_cols = ['conf', 'eth_24h', 'dd_7d', 'vol_pctile', 'conf_pctile_7d',
                 'regime_bull', 'hours_since_sell']
    open_buy_features = None

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
            cur_features = {
                'conf': sconf,
                'eth_24h': eth_24h[i] if not np.isnan(eth_24h[i]) else 0,
                'dd_7d': dd_7d[i] if not np.isnan(dd_7d[i]) else 0,
                'vol_pctile': vol_pctile[i] if not np.isnan(vol_pctile[i]) else 50,
                'conf_pctile_7d': conf_pctile[i] if not np.isnan(conf_pctile[i]) else 50,
                'regime_bull': 1 if regime == 'bull' else 0,
                'hours_since_sell': min(168, i - last_sell_idx),
            }
            allow = True
            if len(closed_trades) >= meta_train_n:
                # Train on last meta_train_n closed trades
                recent = closed_trades[-meta_train_n:]
                X = np.array([[t['features'][k] for k in feat_cols] for t in recent])
                y = np.array([1 if t['pnl_pct'] > 0 else 0 for t in recent])
                if y.sum() > 2 and (len(y) - y.sum()) > 2:
                    try:
                        scaler = StandardScaler()
                        Xs = scaler.fit_transform(X)
                        clf = LogisticRegression(max_iter=300, random_state=42)
                        clf.fit(Xs, y)
                        x_cur = np.array([[cur_features[k] for k in feat_cols]])
                        x_cur_s = scaler.transform(x_cur)
                        prob = clf.predict_proba(x_cur_s)[0][1]
                        if prob < meta_threshold:
                            allow = False
                    except Exception:
                        pass

            if allow:
                held = cash * (1 - FEE) / price
                cash = 0; in_pos = True; entry_px = price
                hold_since_entry = 0
                open_buy_features = cur_features
            else:
                blocked += 1

        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = shield_on and min_pnl > 0 and cur_pnl < min_pnl
            if not shield_blocks or override_expired:
                cash = held * price * (1 - FEE)
                trade_log.append({'pnl_pct': cur_pnl, 'regime': regime})
                if open_buy_features is not None:
                    closed_trades.append({'features': open_buy_features, 'pnl_pct': cur_pnl})
                    open_buy_features = None
                held = 0; in_pos = False; hold_since_entry = 0
                last_sell_idx = i

        if bull_cd > 0: bull_cd -= 1
        if bear_cd > 0: bear_cd -= 1

    if in_pos:
        cash = held * sigs[-1]['close'] * (1 - FEE)
        cur_pnl = (sigs[-1]['close'] / entry_px - 1) * 100
        trade_log.append({'pnl_pct': cur_pnl, 'regime': 'bull'})

    ret = (cash / 1000.0 - 1) * 100
    n = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    wr = (wins / n * 100) if n else 0
    return ret, n, wr, blocked


def sim_baseline_simple(sigs, asset_cfg, bull_conf=None):
    """Reference baseline matching collect_buy_outcomes logic."""
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


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    print("Loading...")
    sigs = load_sigs()
    eth = pd.read_csv(ETH_HOURLY)
    eth['datetime'] = pd.to_datetime(eth['datetime'], utc=True)
    eth = eth.set_index('datetime').sort_index()

    windows = {'30d': window_slice(sigs, 30),
               '60d': window_slice(sigs, 60),
               '90d': sigs}

    rows = []
    for w_name, w_sigs in windows.items():
        print(f"\n[{w_name}: {len(w_sigs)} sigs]")
        eth_24h, dd_7d, vol_pctile, conf_pctile = precompute_meta_features(w_sigs, eth)

        for conf in [80, 90]:
            base_ret, base_n, base_wr = sim_baseline_simple(w_sigs, asset_cfg, bull_conf=conf)
            print(f"  baseline_{conf}: {base_ret:+.2f}% / {base_n} tr / WR {base_wr:.0f}%")

            for train_n in [10, 15, 20]:
                for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
                    ret, n, wr, blk = sim_with_meta(
                        w_sigs, asset_cfg, eth_24h, dd_7d, vol_pctile, conf_pctile,
                        meta_train_n=train_n, meta_threshold=thr, bull_conf=conf)
                    rows.append({
                        'window': w_name,
                        'config': f'meta train={train_n} thr={thr} bull_conf={conf}',
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
    print(f"  T7 META PROXY — sorted by 60d delta vs same-conf baseline")
    print(f"{'='*120}")
    print(full.head(25).to_string())

    print(f"\n{'='*120}")
    print(f"  POSITIVE ON ALL 3 WINDOWS")
    print(f"{'='*120}")
    pos = full[(full['d_30d'] > 0) & (full['d_60d'] > 0) & (full['d_90d'] > 0)]
    print(pos.to_string() if not pos.empty else "  None.")

    out = os.path.join(ENGINE, 'output',
                       f't7_meta_proxy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
