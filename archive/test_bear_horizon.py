"""Bear horizon test — standalone, no prod writes.

For each bear_h in [5, 6, 7, 8] with bull fixed at 6h, generate signals,
merge with the current regime detector (tsmom_672h), simulate the full
strategy (shield + current rally-cd gate), compare returns.

Limitation: uses bear.min_confidence=80% across all candidates (current config).
A proper test would re-run Mode S per candidate, but that's hours of work.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ENGINE = r'G:\Autres ordinateurs\My laptop\engine'
os.chdir(ENGINE)
sys.path.insert(0, ENGINE)

import json
from crypto_trading_system_ed import (
    generate_signals, _suppress_stderr, _merge_tagged_signals
)

REGIME_CFG_PATH = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
PROD_CSV = os.path.join(ENGINE, 'models', 'crypto_ed_production.csv')
ASSET = 'ETH'
BULL_H = 6
BEAR_CANDIDATES = [5, 6, 7, 8]
REPLAY = 1440

FEE = 0.0005  # 5 bps/leg realistic maker blend (see ed.py BACKTEST_FEE_PER_LEG)


def load_cfg():
    with open(REGIME_CFG_PATH) as f:
        return json.load(f)


def pick_model_row(df_models, asset, h):
    rows = df_models[(df_models['coin'] == asset) & (df_models['horizon'] == h)]
    if rows.empty:
        return None
    return rows.sort_values('combined_score', ascending=False).iloc[0]


def gen_signals_for(asset, h, replay_hours):
    df_models = pd.read_csv(PROD_CSV)
    row = pick_model_row(df_models, asset, h)
    if row is None:
        return None
    feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
    gamma = float(row.get('gamma', 1.0))
    with _suppress_stderr():
        sigs = generate_signals(asset, row['models'].split('+'),
                                int(row['best_window']), replay_hours,
                                feature_override=feats, horizon=h, gamma=gamma)
    return sigs


def sim_strategy(sigs, asset_cfg):
    """Simulate with current config's shield + single gate, per-regime tags."""
    bull_shield = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    bull_conf = float(asset_cfg.get('bull', {}).get('min_confidence', 95))
    bear_conf = float(asset_cfg.get('bear', {}).get('min_confidence', 80))
    min_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.6))
    max_hold = int(asset_cfg.get('max_hold_hours', 12))
    rc = asset_cfg.get('rally_cooldown') or {}
    if rc.get('enabled'):
        h_s = int(rc['h_short']); h_l = int(rc['h_long'])
        t_s = float(rc['t_short_pct']); t_l = float(rc['t_long_pct'])
        cd_h = int(rc['cd_hours'])
    else:
        h_s = h_l = 0; t_s = t_l = 9999; cd_h = 0

    closes = np.array([float(s['close']) for s in sigs])
    def _rr(h):
        out = np.full(len(closes), np.nan)
        if 0 < h < len(closes):
            out[h:] = (closes[h:] / closes[:-h] - 1.0) * 100.0
        return out
    rs_arr = _rr(h_s) if h_s else None
    rl_arr = _rr(h_l) if h_l else None

    cash = 1000.0; qty = 0.0; in_pos = False; entry_px = 0.0
    hold = 0; cd = 0; trades = []

    for i, s in enumerate(sigs):
        regime = s.get('regime', 'bull')
        price = float(s['close'])
        sig = s['signal']
        sconf = float(s.get('confidence', 0))
        conf_thr = bull_conf if regime == 'bull' else bear_conf
        shield_on = bull_shield if regime == 'bull' else bear_shield

        if cd_h > 0 and rs_arr is not None and rl_arr is not None:
            rs = rs_arr[i] if not np.isnan(rs_arr[i]) else 0
            rl = rl_arr[i] if not np.isnan(rl_arr[i]) else 0
            if rs >= t_s or rl >= t_l:
                cd = max(cd, cd_h)

        if sig == 'BUY' and sconf >= conf_thr and not in_pos:
            if cd <= 0:
                qty = cash * (1 - FEE) / price
                cash = 0
                in_pos = True
                entry_px = price
                hold = 0
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1.0) * 100.0
            shield_min = min_pnl if shield_on else 0.0
            blocked = shield_on and cur_pnl < shield_min and hold < max_hold
            if not blocked:
                cash = qty * price * (1 - FEE)
                trades.append(cur_pnl)
                qty = 0
                in_pos = False
                entry_px = 0
                hold = 0

        if in_pos: hold += 1
        if cd > 0: cd -= 1

    if in_pos:
        cash = qty * sigs[-1]['close'] * (1 - FEE)
        trades.append((sigs[-1]['close'] / entry_px - 1.0) * 100.0)

    ret = (cash / 1000.0 - 1.0) * 100.0
    wr = (sum(1 for t in trades if t > 0) / len(trades) * 100) if trades else 0
    worst = min(trades, default=0)
    return dict(return_pct=ret, trades=len(trades), win_rate=wr, worst=worst)


def main():
    print("=" * 100)
    print(f"  Bear Horizon Test — {ASSET}, bull fixed at {BULL_H}h, 60d window, no prod writes")
    print("=" * 100)

    cfg = load_cfg()
    asset_cfg = cfg[ASSET]
    print(f"  Current: bull={BULL_H}h@{asset_cfg['bull']['min_confidence']}% | "
          f"bear={asset_cfg['bear']['horizon']}h@{asset_cfg['bear']['min_confidence']}%")
    print(f"  Shield: bull={asset_cfg['bull'].get('hold_shield')} / bear={asset_cfg['bear'].get('hold_shield')}, "
          f"thr={asset_cfg.get('min_sell_pnl_pct')}%/{asset_cfg.get('max_hold_hours')}h")
    rc = asset_cfg.get('rally_cooldown', {})
    print(f"  Gate: rr{rc.get('h_short')}h>={rc.get('t_short_pct')}% OR "
          f"rr{rc.get('h_long')}h>={rc.get('t_long_pct')}%, cd={rc.get('cd_hours')}h")
    print()

    # Generate bull signals once
    print(f"  Generating bull {BULL_H}h signals (shared across tests)...")
    t0 = time.time()
    bull_sigs = gen_signals_for(ASSET, BULL_H, REPLAY)
    print(f"    {BULL_H}h: {len(bull_sigs)} candles ({time.time()-t0:.1f}s)")

    # Generate each bear candidate + test
    results = []
    for bear_h in BEAR_CANDIDATES:
        print(f"\n  Generating bear {bear_h}h signals...")
        t0 = time.time()
        bear_sigs = gen_signals_for(ASSET, bear_h, REPLAY)
        print(f"    {bear_h}h: {len(bear_sigs)} candles ({time.time()-t0:.1f}s)")

        # Override asset_cfg bear.horizon for this test (in-memory only)
        test_cfg = json.loads(json.dumps(asset_cfg))  # deep copy
        test_cfg['bear']['horizon'] = bear_h

        tagged = _merge_tagged_signals(ASSET, bull_sigs, bear_sigs, test_cfg)
        bull_count = sum(1 for s in tagged if s['regime'] == 'bull')
        bear_count = sum(1 for s in tagged if s['regime'] == 'bear')
        print(f"    merged: bull={bull_count} bear={bear_count}")

        r = sim_strategy(tagged, test_cfg)
        r['bear_h'] = bear_h
        results.append(r)

    # Summary
    print()
    print("=" * 100)
    print(f"  Results — bull={BULL_H}h fixed, bear varying")
    print("=" * 100)
    print(f"  {'bear_h':<8}{'Return%':>10}{'Trades':>9}{'WR%':>6}{'Worst%':>10}")
    print(f"  {'-'*8}{'-'*10}{'-'*9}{'-'*6}{'-'*10}")
    for r in results:
        marker = ' <- CURRENT' if r['bear_h'] == asset_cfg['bear']['horizon'] else ''
        print(f"  {r['bear_h']}h      {r['return_pct']:>+9.2f}%{r['trades']:>9}"
              f"{r['win_rate']:>5.0f}%{r['worst']:>+9.2f}%{marker}")

    # Find winner
    best = max(results, key=lambda r: r['return_pct'])
    current = next(r for r in results if r['bear_h'] == asset_cfg['bear']['horizon'])
    if best['bear_h'] != current['bear_h']:
        delta = best['return_pct'] - current['return_pct']
        print(f"\n  >>> Better bear horizon found: {best['bear_h']}h beats current "
              f"{current['bear_h']}h by {delta:+.2f}pp")
    else:
        print(f"\n  >>> Current bear={current['bear_h']}h is already the winner.")


if __name__ == '__main__':
    main()
