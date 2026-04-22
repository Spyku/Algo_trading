"""
test_meta_strategy_impact.py — Strategy-aware A/B test of meta-labeling.

Runs the FULL Ed strategy (regime switching + shield + gates + min_sell_pnl + max_hold)
on the same primary signal stream twice:
  1. Baseline: strategy unmodified
  2. Meta-filtered: BUYs with meta_prob < threshold downgraded to HOLD
Same regime config, same gates, same shield — only the BUY-gate from meta differs.

Does NOT write production files. Reads current prod config to match live behavior.

Usage:
  python tools/test_meta_strategy_impact.py ETH --replay 2880
  python tools/test_meta_strategy_impact.py ETH --replay 2880 --p-thresholds 0.45,0.50,0.55,0.60
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

warnings.filterwarnings('ignore')

from crypto_trading_system_ed import (
    load_data,
    build_all_features,
    generate_signals,
    _merge_tagged_signals,
    _build_regime_indicators_and_detectors,
    _build_detector_from_cfg,
    BACKTEST_FEE_PER_LEG,
)
from crypto_trading_system_meta import (
    load_primary_config,
    build_meta_dataset,
    walk_forward_meta_train,
)


# ---- Strategy simulator (extracted from _sweep_rally_cooldown.simulate) ----

def simulate_strategy(sigs, asset_cfg, rr_dict=None, gate_cfg=None, fee=BACKTEST_FEE_PER_LEG):
    """Full strategy simulation matching _sweep_rally_cooldown.simulate().

    sigs: list of tagged signal dicts (regime, signal, confidence, close, datetime)
    rr_dict: optional {horizon: np.array} of relative returns for gate trigger
    gate_cfg: optional dict {h_short, h_long, t_short, t_long, cd_hours, regime_filter}

    Returns dict(pnl_pct, dd_pct, trades, skipped, buy_count, sell_count, hold_count).
    """
    bull_shield = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    bull_conf = float(asset_cfg.get('bull', {}).get('min_confidence', 75.0))
    bear_conf = float(asset_cfg.get('bear', {}).get('min_confidence', 65.0))
    min_sell_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.5))
    max_hold = int(asset_cfg.get('max_hold_hours', 10))
    qr_cfg = asset_cfg.get('shield_quick_release', {})
    qr_enabled = bool(qr_cfg.get('enabled', False))
    qr_min_conf = float(qr_cfg.get('min_sell_conf', 95))
    qr_max_hours = float(qr_cfg.get('max_hours', 3))

    if gate_cfg:
        h_s = int(gate_cfg['h_short']); h_l = int(gate_cfg['h_long'])
        t_s = float(gate_cfg['t_short']); t_l = float(gate_cfg['t_long'])
        cd_h = int(gate_cfg['cd_hours']); regime_filter = gate_cfg.get('regime_filter', 'all')
        rs_arr = rr_dict.get(h_s) if rr_dict else None
        rl_arr = rr_dict.get(h_l) if rr_dict else None
    else:
        cd_h = 0; rs_arr = None; rl_arr = None; regime_filter = 'all'
        h_s = h_l = 0; t_s = t_l = 0.0

    cash = 1000.0; qty = 0.0; in_pos = False; entry = 0.0
    hold = 0; trades = 0; skipped = 0; cd = 0
    buy_count = sell_count = hold_count = 0
    ec = [1000.0]
    n = len(sigs)

    for i in range(n):
        s = sigs[i]; price = s['close']
        regime = s.get('regime', 'bull')
        conf_thr = bull_conf if regime == 'bull' else bear_conf

        # Gate trigger check
        if cd_h > 0 and rs_arr is not None and rl_arr is not None:
            if regime_filter == 'all' or regime_filter == regime:
                rs = rs_arr[i] if i < len(rs_arr) else float('nan')
                rl = rl_arr[i] if i < len(rl_arr) else float('nan')
                if (rs == rs and rs >= t_s) or (rl == rl and rl >= t_l):
                    cd = max(cd, cd_h)

        ec.append(cash + qty * price if in_pos else cash)

        sig = s['signal']
        if sig == 'BUY': buy_count += 1
        elif sig == 'SELL': sell_count += 1
        else: hold_count += 1

        if sig == 'BUY' and s['confidence'] >= conf_thr and not in_pos:
            if cd > 0:
                skipped += 1
            else:
                fp = sigs[i + 1]['close'] if i + 1 < n else price
                qty = cash * (1 - fee) / fp
                cash = 0.0; in_pos = True; entry = fp; hold = 0
        elif sig == 'SELL' and in_pos:
            fp = sigs[i + 1]['close'] if i + 1 < n else price
            cur = (fp / entry - 1.0) * 100.0
            shield_on = bull_shield if regime == 'bull' else bear_shield
            shield_min = min_sell_pnl if shield_on else 0.0
            quick_release = (qr_enabled and shield_on
                             and hold <= qr_max_hours
                             and float(s.get('confidence', 0)) >= qr_min_conf)
            if cur >= shield_min or hold >= max_hold or quick_release:
                cash = qty * fp * (1 - fee)
                trades += 1; in_pos = False; qty = 0.0; entry = 0.0; hold = 0

        if in_pos: hold += 1
        if cd > 0: cd -= 1

    if in_pos:
        cash = qty * sigs[-1]['close'] * (1 - fee); trades += 1

    pnl = (cash / 1000.0 - 1.0) * 100.0
    arr = np.array(ec)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    mdd = float(dd.min()) * 100.0 if len(dd) else 0.0
    start_p = sigs[0]['close']; end_p = sigs[-1]['close']
    bh = (end_p / start_p - 1.0) * 100.0
    return dict(pnl_pct=pnl, dd_pct=mdd, trades=trades, skipped=skipped,
                buy_sig=buy_count, sell_sig=sell_count, hold_sig=hold_count,
                bh_pct=bh, alpha_pp=pnl - bh)


# ---- Meta prediction lookup ----

def build_meta_lookup(asset, horizon, replay_hours, primary_cfg, signals=None):
    """Train meta for one horizon, return dict[(naive_datetime)] -> meta_prob for primary BUYs."""
    meta_df, feature_cols = build_meta_dataset(asset, horizon, replay_hours, primary_cfg, signals=signals)
    if len(meta_df) == 0:
        return {}
    preds = walk_forward_meta_train(meta_df, feature_cols, horizon=horizon,
                                    min_train=40, step=10)
    lookup = {}
    for _, row in preds.iterrows():
        dt = pd.Timestamp(row['datetime'])
        if dt.tzinfo is not None:
            dt = dt.tz_convert('UTC').tz_localize(None)
        lookup[dt] = float(row['meta_prob'])
    return lookup


def apply_meta_filter(sigs, meta_bull, meta_bear, bull_h, bear_h, threshold):
    """Return new signal list with BUYs downgraded to HOLD if meta_prob < threshold.
    Uses the regime tag on each signal to pick the right meta lookup."""
    out = []
    kept = filtered = no_pred = 0
    for s in sigs:
        s2 = dict(s)
        if s2['signal'] == 'BUY':
            dt = s2['datetime']
            if isinstance(dt, str):
                dt = pd.Timestamp(dt)
            if dt.tzinfo is not None:
                dt = dt.tz_convert('UTC').tz_localize(None)
            regime = s2.get('regime', 'bull')
            lookup = meta_bull if regime == 'bull' else meta_bear
            p = lookup.get(dt, None)
            if p is None:
                no_pred += 1
                # Keep BUY (no meta prediction available — don't filter defensively)
            elif p < threshold:
                s2['signal'] = 'HOLD'
                filtered += 1
            else:
                kept += 1
        out.append(s2)
    return out, kept, filtered, no_pred


# ---- Main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('asset', help='Asset symbol (ETH, BTC, ...)')
    ap.add_argument('--replay', type=int, default=1440, help='Replay window in hours (1440=2mo, 2880=4mo)')
    ap.add_argument('--p-thresholds', default='0.45,0.50,0.55,0.60',
                    help='Comma-separated meta probability thresholds to test')
    ap.add_argument('--config', default='config/regime_config_ed.json',
                    help='Path to regime config (read-only)')
    args = ap.parse_args()

    thresholds = [float(t) for t in args.p_thresholds.split(',')]

    config_path = os.path.join(ENGINE_DIR, args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        full_cfg = json.load(f)
    asset_cfg = full_cfg.get(args.asset)
    if asset_cfg is None:
        print(f'ERROR: {args.asset} not in {config_path}')
        return

    bull_h = int(asset_cfg['bull']['horizon'])
    bear_h = int(asset_cfg['bear']['horizon'])

    print('=' * 90)
    print(f'  STRATEGY-AWARE META-LABELING A/B TEST — {args.asset}')
    print(f'  Replay: {args.replay}h ({args.replay/720:.1f}mo)')
    print(f'  Bull horizon: {bull_h}h @ conf>={asset_cfg["bull"]["min_confidence"]}%')
    print(f'  Bear horizon: {bear_h}h @ conf>={asset_cfg["bear"]["min_confidence"]}%')
    print(f'  Shield: bull={asset_cfg["bull"].get("hold_shield", True)} bear={asset_cfg["bear"].get("hold_shield", True)}')
    print(f'  min_sell_pnl={asset_cfg.get("min_sell_pnl_pct", 0.5)}% max_hold={asset_cfg.get("max_hold_hours", 10)}h')
    print(f'  Regime detector: {asset_cfg.get("regime_detector", {}).get("params", {}).get("name", "?")}')
    print('=' * 90)

    # Load primary configs for both horizons
    print(f'\n  [1/4] Loading primary configs...')
    bull_cfg = load_primary_config(args.asset, bull_h)
    bear_cfg = load_primary_config(args.asset, bear_h)
    print(f'    Bull {bull_h}h: {bull_cfg["combo"]} w={bull_cfg["window"]} gamma={bull_cfg["gamma"]} n_feat={len(bull_cfg["features"])}')
    print(f'    Bear {bear_h}h: {bear_cfg["combo"]} w={bear_cfg["window"]} gamma={bear_cfg["gamma"]} n_feat={len(bear_cfg["features"])}')

    # Generate primary signals for both horizons
    print(f'\n  [2/4] Generating primary signals for both horizons ({args.replay}h replay)...')
    bull_sigs = generate_signals(
        asset_name=args.asset, model_names=bull_cfg['combo'].split('+'),
        window_size=bull_cfg['window'], replay_hours=args.replay,
        feature_override=bull_cfg['features'], horizon=bull_h, gamma=bull_cfg['gamma'],
    )
    bear_sigs = generate_signals(
        asset_name=args.asset, model_names=bear_cfg['combo'].split('+'),
        window_size=bear_cfg['window'], replay_hours=args.replay,
        feature_override=bear_cfg['features'], horizon=bear_h, gamma=bear_cfg['gamma'],
    )
    print(f'    Bull signals: {len(bull_sigs)} ({sum(1 for s in bull_sigs if s["signal"]=="BUY")} BUYs)')
    print(f'    Bear signals: {len(bear_sigs)} ({sum(1 for s in bear_sigs if s["signal"]=="BUY")} BUYs)')

    # Regime-tag via _merge_tagged_signals
    tagged = _merge_tagged_signals(args.asset, bull_sigs, bear_sigs, asset_cfg)
    n_bull = sum(1 for s in tagged if s['regime'] == 'bull')
    n_bear = sum(1 for s in tagged if s['regime'] == 'bear')
    print(f'    Merged: {len(tagged)} bars | bull={n_bull} bear={n_bear}')

    # Train meta for BOTH horizons (reuse pre-generated signals — no regen)
    print(f'\n  [3/4] Training meta (walk-forward) for each horizon...')
    print(f'    --- Bull {bull_h}h meta ---')
    meta_bull = build_meta_lookup(args.asset, bull_h, args.replay, bull_cfg, signals=bull_sigs)
    print(f'    --- Bear {bear_h}h meta ---')
    meta_bear = build_meta_lookup(args.asset, bear_h, args.replay, bear_cfg, signals=bear_sigs)

    # Run baseline
    print(f'\n  [4/4] Running strategy simulations...')
    print(f'\n    --- BASELINE (prod conf thresholds, no meta filter) ---')
    baseline = simulate_strategy(tagged, asset_cfg)
    print(f'    PnL={baseline["pnl_pct"]:+.2f}%  BH={baseline["bh_pct"]:+.2f}%  '
          f'alpha={baseline["alpha_pp"]:+.2f}pp  trades={baseline["trades"]}  '
          f'MDD={baseline["dd_pct"]:.2f}%  skipped={baseline["skipped"]}')

    # Confidence-sweep baseline: what if we just raised primary confidence threshold?
    # This is the orthogonality test — does meta add value beyond simply being stricter on primary conf?
    print(f'\n    --- CONFIDENCE-SWEEP BASELINE (no meta, only raise primary conf thresholds) ---')
    print(f'    {"bull":>5} {"bear":>5} {"kept":>5} {"pnl":>9} {"alpha":>9} {"trades":>7} {"MDD":>7} {"d_vs_base":>11}')
    conf_rows = []
    for bull_c, bear_c in [(70, 75), (75, 80), (80, 85), (85, 90), (90, 90), (95, 95)]:
        cfg_tight = json.loads(json.dumps(asset_cfg))
        cfg_tight['bull']['min_confidence'] = bull_c
        cfg_tight['bear']['min_confidence'] = bear_c
        # Count kept BUYs (primary BUY with conf >= regime threshold)
        kept = sum(1 for s in tagged
                   if s['signal'] == 'BUY'
                   and s['confidence'] >= (bull_c if s['regime'] == 'bull' else bear_c))
        r = simulate_strategy(tagged, cfg_tight)
        delta = r['pnl_pct'] - baseline['pnl_pct']
        print(f'    {bull_c:>5} {bear_c:>5} {kept:>5} {r["pnl_pct"]:>+8.2f}% '
              f'{r["alpha_pp"]:>+8.2f}pp {r["trades"]:>7} {r["dd_pct"]:>6.2f}% {delta:>+10.2f}pp')
        conf_rows.append(dict(bull_conf=bull_c, bear_conf=bear_c, kept=kept,
                              pnl_pct=r['pnl_pct'], alpha_pp=r['alpha_pp'],
                              trades=r['trades'], dd_pct=r['dd_pct'],
                              delta_vs_baseline=delta))

    # Per-threshold runs
    print(f'\n    --- META-FILTERED ---')
    print(f'    {"p":>5} {"kept":>5} {"filt":>5} {"no_p":>5} {"pnl":>9} {"alpha":>9} {"trades":>7} {"MDD":>7} {"d_vs_base":>11}')
    rows = []
    for thr in thresholds:
        filtered_sigs, kept, filtered, no_pred = apply_meta_filter(
            tagged, meta_bull, meta_bear, bull_h, bear_h, thr)
        r = simulate_strategy(filtered_sigs, asset_cfg)
        delta = r['pnl_pct'] - baseline['pnl_pct']
        print(f'    {thr:>5.2f} {kept:>5} {filtered:>5} {no_pred:>5} '
              f'{r["pnl_pct"]:>+8.2f}% {r["alpha_pp"]:>+8.2f}pp {r["trades"]:>7} '
              f'{r["dd_pct"]:>6.2f}% {delta:>+10.2f}pp')
        rows.append(dict(threshold=thr, kept=kept, filtered=filtered,
                         no_pred=no_pred, pnl_pct=r['pnl_pct'],
                         alpha_pp=r['alpha_pp'], trades=r['trades'],
                         dd_pct=r['dd_pct'], delta_vs_baseline=delta))

    # Save results
    os.makedirs('output', exist_ok=True)
    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    meta_rows = [dict(label=f'meta_p={r["threshold"]}', kept=r['kept'],
                      pnl_pct=r['pnl_pct'], alpha_pp=r['alpha_pp'],
                      trades=r['trades'], dd_pct=r['dd_pct'],
                      delta_vs_baseline=r['delta_vs_baseline']) for r in rows]
    conf_out = [dict(label=f'conf_bull{c["bull_conf"]}_bear{c["bear_conf"]}',
                     kept=c['kept'], pnl_pct=c['pnl_pct'], alpha_pp=c['alpha_pp'],
                     trades=c['trades'], dd_pct=c['dd_pct'],
                     delta_vs_baseline=c['delta_vs_baseline']) for c in conf_rows]
    df_out = pd.DataFrame([
        dict(label='baseline', kept=baseline['buy_sig'],
             pnl_pct=baseline['pnl_pct'], alpha_pp=baseline['alpha_pp'],
             trades=baseline['trades'], dd_pct=baseline['dd_pct'],
             delta_vs_baseline=0.0),
    ] + conf_out + meta_rows)
    out_path = f'output/meta_strategy_impact_{args.asset}_{args.replay}h_{tag}.csv'
    df_out.to_csv(out_path, index=False)
    print(f'\n  Saved: {out_path}')
    print(f'\n  No production files modified.')


if __name__ == '__main__':
    main()
