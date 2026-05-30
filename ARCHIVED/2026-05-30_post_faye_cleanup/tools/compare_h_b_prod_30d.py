"""tools/compare_h_b_prod_30d.py
30-day head-to-head: LIVE production vs B_multi_seed vs H75.

For each variant, generates fresh walk-forward signals over the last 720h of ETH data
using its own per-horizon (window, gamma, features, combo), tags by detector,
applies the variant's per-regime gate + shield + max_hold policy, simulates trades,
reports return / trades / max DD.

Run from the engine root:  python tools/compare_h_b_prod_30d.py

CAVEAT — in-sample bias in MODEL SELECTION (not in prediction):
  Each variant's Mode V picked its (w, gamma, features, combo) tuple by maximizing
  return over the variant's HRST replay window. Predictions within that window are
  walk-forward (no leakage at the bar level), but the SELECTION had visibility of
  the entire window. So variants whose HRST ended more recently have an unfair
  advantage on a "last 30 days" comparison.

    LIVE_prod   HRST = May 6  → 12 days of the test window are OOS for selection
    B_multi_seed HRST = May 15 → 3 days OOS
    H75         HRST = May 18 → 0 days OOS (fully in-sample)

  Newer variants will look artificially better. For a true OOS comparison, compare
  each variant's H2 from its own HRST log (CLAUDE.md Critical Rule 17), but those
  windows don't align in calendar time.
"""

import csv
import json
import os
import sys
import time
import numpy as np
import pandas as pd

# Make engine root importable when this script is launched from tools/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Prevent engine from parsing our CLI args during import
_argv_save = sys.argv
sys.argv = [sys.argv[0]]
import crypto_trading_system_ed as eng  # noqa: E402
sys.argv = _argv_save

REPLAY_HOURS = 720  # 30 days
FEE = eng.BACKTEST_FEE_PER_LEG  # 0.0005 = 5 bps/leg

# B_multi_seed Mode T winner (reconstructed from
# logs/run_reliability_hrst_desktop_20260515_154801_B_multi_seed.log):
#   detector = sma24>sma100
#   bull = 6h @ 65%  shield=OFF  gate rr12h>=2.0% OR rr24h>=4.0%  cd=8h
#   bear = 8h @ 65%  shield=OFF  gate rr8h>=2.0%  OR rr16h>=3.0%  cd=8h
#   min_sell_pnl = 0.50 (irrelevant — shields off)  max_hold = 10h
B_REGIME_CFG = {
    'enabled': True,
    'symbol': 'ETH-USD',
    'regime_detector': {'type': 'named', 'params': {'name': 'sma24>sma100'}},
    'bull': {
        'horizon': 6,
        'min_confidence': 65,
        'hold_shield': False,
        'rally_cooldown': {'enabled': True, 'h_short': 12, 'h_long': 24,
                            't_short_pct': 2.0, 't_long_pct': 4.0, 'cd_hours': 8},
    },
    'bear': {
        'horizon': 8,
        'min_confidence': 65,
        'hold_shield': False,
        'rally_cooldown': {'enabled': True, 'h_short': 8, 'h_long': 16,
                            't_short_pct': 2.0, 't_long_pct': 3.0, 'cd_hours': 8},
    },
    'min_sell_pnl_pct': 0.5,
    'max_hold_hours': 10,
}

VARIANTS = [
    {
        'label': 'LIVE_prod',
        'prod_csv': 'models/crypto_ed_production.csv',
        'regime_json': 'config/regime_config_ed.json',
    },
    {
        'label': 'B_multi_seed',
        'prod_csv': 'models/crypto_ed_production_reliability_B_multi_seed.csv',
        'regime_json': None,  # use B_REGIME_CFG inline
    },
    {
        'label': 'H75',
        'prod_csv': 'models_h75/crypto_ed_production_noprod.csv',
        'regime_json': 'config_h75/regime_config_ed_noprod.json',
    },
    {
        # G_narrow_d Desktop, finished 2026-05-21 10:28. Fully in-sample
        # against the 1440h replay it was tuned on (60d) — selection bias
        # is largest here; treat the comparison number cautiously.
        'label': 'G_narrow_desktop',
        'prod_csv': 'models_g_desktop/crypto_ed_production_noprod.csv',
        'regime_json': 'config_g_desktop/regime_config_ed_noprod.json',
    },
]


def load_variant(v):
    cfg = B_REGIME_CFG if v['regime_json'] is None else json.load(open(v['regime_json']))['ETH']
    models = {}
    with open(v['prod_csv']) as f:
        for row in csv.DictReader(f):
            if row['coin'] != 'ETH':
                continue
            h = int(row['horizon'])
            models[h] = {
                'combo': row['best_combo'],
                'window': int(row['best_window']),
                'gamma': float(row['gamma']),
                'features': [x.strip() for x in row['optimal_features'].split(',')],
            }
    return cfg, models


def make_signals(models, h):
    m = models[h]
    print(f"    signal-gen h={h}h w={m['window']} {m['combo']} γ={m['gamma']:.4f} nf={len(m['features'])}", flush=True)
    t0 = time.time()
    sigs = eng.generate_signals(
        'ETH', m['combo'].split('+'), m['window'],
        replay_hours=REPLAY_HOURS,
        feature_override=m['features'],
        horizon=h, gamma=m['gamma'],
    )
    print(f"    -> {len(sigs)} bars in {time.time()-t0:.1f}s", flush=True)
    return sigs


def compute_rr_dict(sigs):
    """Rolling-return arrays at horizons used by gates."""
    closes = np.array([s['close'] for s in sigs])
    n = len(closes)
    rr = {}
    for h in [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 48]:
        arr = np.full(n, np.nan)
        for i in range(h, n):
            arr[i] = (closes[i] / closes[i - h] - 1) * 100
        rr[h] = arr
    return rr


def simulate(sigs, rr_dict, asset_cfg):
    """Port of the inner simulate() from _sweep_rally_cooldown (regime_filter='all').
    Independent per-regime gates: each regime's gate fires only on its own bars."""
    bull_shield = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    bull_conf = float(asset_cfg.get('bull', {}).get('min_confidence', 75))
    bear_conf = float(asset_cfg.get('bear', {}).get('min_confidence', 65))
    min_sell_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.5))
    max_hold = int(asset_cfg.get('max_hold_hours', 10))

    def get_gate(regime):
        g = asset_cfg.get(regime, {}).get('rally_cooldown') or {}
        if not g.get('enabled'):
            return None
        try:
            return (int(g['h_short']), int(g['h_long']),
                    float(g['t_short_pct']), float(g['t_long_pct']),
                    int(g['cd_hours']))
        except (KeyError, TypeError, ValueError):
            return None

    bull_g = get_gate('bull')
    bear_g = get_gate('bear')

    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry = 0.0
    hold = 0
    trades = 0
    skipped = 0
    bull_cd = 0
    bear_cd = 0
    ec = [1000.0]
    trade_log = []
    n_buys_bull = n_buys_bear = 0

    for i, s in enumerate(sigs):
        price = s['close']
        regime = s.get('regime', 'bull')
        conf_thr = bull_conf if regime == 'bull' else bear_conf
        shield_on = bull_shield if regime == 'bull' else bear_shield

        # Fire gate on its own regime's bars
        if bull_g and regime == 'bull':
            h_s, h_l, t_s, t_l, cd_h = bull_g
            rs = rr_dict.get(h_s, [np.nan]*len(sigs))[i]
            rl = rr_dict.get(h_l, [np.nan]*len(sigs))[i]
            if (rs == rs and rs >= t_s) or (rl == rl and rl >= t_l):
                bull_cd = max(bull_cd, cd_h)
        if bear_g and regime == 'bear':
            h_s, h_l, t_s, t_l, cd_h = bear_g
            rs = rr_dict.get(h_s, [np.nan]*len(sigs))[i]
            rl = rr_dict.get(h_l, [np.nan]*len(sigs))[i]
            if (rs == rs and rs >= t_s) or (rl == rl and rl >= t_l):
                bear_cd = max(bear_cd, cd_h)

        ec.append(cash + qty * price if in_pos else cash)
        active_cd = bull_cd if regime == 'bull' else bear_cd

        if s['signal'] == 'BUY' and s['confidence'] >= conf_thr and not in_pos:
            if active_cd > 0:
                skipped += 1
            else:
                qty = cash * (1 - FEE) / price
                cash = 0.0
                in_pos = True
                entry = price
                hold = 0
                if regime == 'bull':
                    n_buys_bull += 1
                else:
                    n_buys_bear += 1
        elif s['signal'] == 'SELL' and in_pos:
            cur_pnl = (price / entry - 1.0) * 100.0
            shield_min = min_sell_pnl if shield_on else 0.0
            if cur_pnl >= shield_min or hold >= max_hold:
                cash = qty * price * (1 - FEE)
                trades += 1
                trade_log.append(cur_pnl)
                in_pos = False
                qty = 0.0
                entry = 0.0
                hold = 0

        if in_pos:
            hold += 1
        if bull_cd > 0:
            bull_cd -= 1
        if bear_cd > 0:
            bear_cd -= 1

    if in_pos:
        cash = qty * sigs[-1]['close'] * (1 - FEE)
        trades += 1
        trade_log.append((sigs[-1]['close'] / entry - 1.0) * 100.0)

    pnl = (cash / 1000.0 - 1.0) * 100.0
    arr = np.array(ec)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    mdd = float(dd.min()) * 100.0 if len(dd) else 0.0
    wins = sum(1 for t in trade_log if t > 0)
    wr = (wins / len(trade_log) * 100.0) if trade_log else 0.0
    avg_win = np.mean([t for t in trade_log if t > 0]) if wins else 0.0
    avg_loss = np.mean([t for t in trade_log if t <= 0]) if (len(trade_log) - wins) else 0.0
    return dict(pnl_pct=pnl, mdd_pct=mdd, trades=trades, round_trips=len(trade_log),
                wr=wr, skipped=skipped, avg_win=avg_win, avg_loss=avg_loss,
                n_buys_bull=n_buys_bull, n_buys_bear=n_buys_bear)


def main():
    print(f"=== 30-day backtest (last {REPLAY_HOURS}h, fee={FEE*100:.2f}%/leg) ===")
    print("    bias caveat: H75 fully in-sample, B 3d OOS, LIVE 12d OOS\n")

    # Buy-and-hold baseline for the same window
    df = pd.read_csv('data/eth_hourly_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    start_close = float(df.iloc[-REPLAY_HOURS]['close'])
    end_close = float(df.iloc[-1]['close'])
    bh_ret = (end_close / start_close - 1.0) * 100.0
    print(f"    B&H ETH: {start_close:.2f} -> {end_close:.2f} = {bh_ret:+.2f}%  "
          f"({df.iloc[-REPLAY_HOURS]['datetime']} -> {df.iloc[-1]['datetime']})\n")

    # Cache signals by (horizon, window, combo, gamma, features_hash) — variants
    # often share horizons but with different models, so we can't dedupe blindly.
    results = []
    for v in VARIANTS:
        print(f"--- {v['label']} ---")
        cfg, models = load_variant(v)
        bull_h = cfg['bull']['horizon']
        bear_h = cfg['bear']['horizon']
        det = cfg['regime_detector']['params']['name']
        print(f"  detector={det}  bull={bull_h}h@{cfg['bull']['min_confidence']}%  "
              f"bear={bear_h}h@{cfg['bear']['min_confidence']}%  "
              f"shields=bull:{cfg['bull'].get('hold_shield')}/bear:{cfg['bear'].get('hold_shield')}")

        bull_sig = make_signals(models, bull_h)
        bear_sig = make_signals(models, bear_h) if bear_h != bull_h else bull_sig

        merged = eng._merge_tagged_signals('ETH', bull_sig, bear_sig, cfg)
        n_bull = sum(1 for s in merged if s['regime'] == 'bull')
        n_bear = len(merged) - n_bull
        rr = compute_rr_dict(merged)
        res = simulate(merged, rr, cfg)

        res['label'] = v['label']
        res['n_signals'] = len(merged)
        res['n_bull_bars'] = n_bull
        res['n_bear_bars'] = n_bear
        res['alpha_vs_bh'] = res['pnl_pct'] - bh_ret
        results.append(res)
        print(f"  signals: {len(merged)} bars (bull={n_bull}/bear={n_bear})")
        print(f"  -> return={res['pnl_pct']:+.2f}%  alpha={res['alpha_vs_bh']:+.2f}pp  "
              f"trades={res['round_trips']} wr={res['wr']:.0f}%  mdd={res['mdd_pct']:.2f}%  "
              f"skipped={res['skipped']}\n")

    # Summary
    print("=" * 80)
    print(f"SUMMARY — last {REPLAY_HOURS}h ({REPLAY_HOURS/24:.0f}d), fee={FEE*100:.2f}%/leg")
    print("=" * 80)
    print(f"{'Variant':<15} {'Return':>10} {'Alpha':>10} {'Trades':>8} {'WR':>6} {'MaxDD':>10} {'Skip':>6}")
    for r in sorted(results, key=lambda x: -x['pnl_pct']):
        print(f"{r['label']:<15} {r['pnl_pct']:>+9.2f}% {r['alpha_vs_bh']:>+9.2f}pp "
              f"{r['round_trips']:>8} {r['wr']:>5.0f}% {r['mdd_pct']:>+9.2f}% {r['skipped']:>6}")
    print(f"{'B&H ETH':<15} {bh_ret:>+9.2f}%")
    print()

    # Save
    import datetime as _dt
    ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = f'output/compare_h_b_prod_30d_{ts}.csv'
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == '__main__':
    main()
