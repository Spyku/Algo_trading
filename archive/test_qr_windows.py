"""Test shield quick-release variants on 30d and 60d windows.

Does our chosen 95%/4h help, harm, or neutral vs OFF on real data?
Uses the current config's shield + bear gate (bull gate currently missing).
Applies the realistic 5 bps/leg fee.

Usage: python test_qr_windows.py
"""
import os
import sys
import pickle
import json

import numpy as np
import pandas as pd

ENGINE = r'G:\Autres ordinateurs\My laptop\engine'
os.chdir(ENGINE)
sys.path.insert(0, ENGINE)

CACHE = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
REGIME_CFG = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
FEE = 0.0005

VARIANTS = [
    ('OFF',            False, 95, 999),
    ('95%/3h',         True,  95, 3),
    ('95%/4h (chosen)', True,  95, 4),
    ('95%/5h',         True,  95, 5),
    ('90%/5h',         True,  90, 5),
    ('90%/6h',         True,  90, 6),
]


def load_sigs():
    with open(CACHE, 'rb') as f:
        sigs = pickle.load(f)
    for s in sigs:
        s['datetime'] = pd.Timestamp(s['datetime'])
    sigs.sort(key=lambda s: s['datetime'])
    return sigs


def window_slice(sigs, days):
    end = sigs[-1]['datetime']
    lo = end - pd.Timedelta(days=days)
    return [s for s in sigs if s['datetime'] >= lo]


def sim(sigs, asset_cfg, qr_enabled, qr_conf, qr_hours):
    """Full strategy sim with per-regime shield + per-regime gate + QR + new fee."""
    bull_shield = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    bull_conf = float(asset_cfg.get('bull', {}).get('min_confidence', 95))
    bear_conf = float(asset_cfg.get('bear', {}).get('min_confidence', 80))
    min_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.6))
    max_hold = int(asset_cfg.get('max_hold_hours', 12))

    def _gate_for(regime):
        block = asset_cfg.get(regime, {})
        rc = block.get('rally_cooldown') if isinstance(block, dict) else None
        rc = rc or asset_cfg.get('rally_cooldown')
        if not rc or not rc.get('enabled'):
            return None
        return (int(rc['h_short']), int(rc['h_long']),
                float(rc['t_short_pct']), float(rc['t_long_pct']),
                int(rc['cd_hours']))
    bull_gate = _gate_for('bull')
    bear_gate = _gate_for('bear')

    closes = np.array([float(s['close']) for s in sigs])
    def _rr(h):
        out = np.full(len(closes), np.nan)
        if 0 < h < len(closes):
            out[h:] = (closes[h:] / closes[:-h] - 1.0) * 100.0
        return out
    rr = {}
    for g in (bull_gate, bear_gate):
        if g is not None:
            for h in (g[0], g[1]):
                if h not in rr:
                    rr[h] = _rr(h)

    cash = 1000.0; qty = 0.0; in_pos = False; entry_px = 0.0
    hold = 0; cd = 0
    trades = []
    qr_fires = 0
    shield_blocks = 0

    for i, s in enumerate(sigs):
        regime = s.get('regime', 'bull')
        price = float(s['close'])
        sig = s['signal']
        sconf = float(s.get('confidence', 0))
        conf_thr = bull_conf if regime == 'bull' else bear_conf
        shield_on = bull_shield if regime == 'bull' else bear_shield
        gate = bull_gate if regime == 'bull' else bear_gate

        if gate is not None:
            h_s, h_l, t_s, t_l, cd_h = gate
            rs = rr[h_s][i] if not np.isnan(rr[h_s][i]) else 0
            rl = rr[h_l][i] if not np.isnan(rr[h_l][i]) else 0
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
            # Quick-release check (before shield)
            quick_release = (qr_enabled and shield_on
                             and hold <= qr_hours and sconf >= qr_conf)
            if shield_on and cur_pnl < min_pnl and hold < max_hold and not quick_release:
                shield_blocks += 1
            else:
                if quick_release and shield_on and cur_pnl < min_pnl:
                    qr_fires += 1
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
    return dict(return_pct=ret, trades=len(trades), win_rate=wr,
                worst=worst, qr_fires=qr_fires, shield_blocks=shield_blocks)


def main():
    print("=" * 100)
    print("  Shield Quick-Release Variant Test — current config + 5 bps fee")
    print("=" * 100)

    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']
    print(f"  Shield: bull={asset_cfg['bull'].get('hold_shield')}, "
          f"bear={asset_cfg['bear'].get('hold_shield')}")
    print(f"  Thresholds: {asset_cfg['min_sell_pnl_pct']}% / {asset_cfg['max_hold_hours']}h")
    print(f"  Bull gate: {asset_cfg['bull'].get('rally_cooldown', 'MISSING')}")
    print(f"  Bear gate: {asset_cfg['bear'].get('rally_cooldown', 'MISSING')}")
    print()

    sigs_all = load_sigs()

    for days in (30, 60):
        sigs = window_slice(sigs_all, days)
        print(f"\n{'='*80}")
        print(f"  {days}d window — {len(sigs)} signals")
        print(f"{'='*80}")
        print(f"  {'QR variant':<20}{'Ret%':>9}{'Trades':>8}{'WR%':>6}{'Worst':>9}"
              f"{'QR fires':>10}{'Shield blocks':>15}")
        print(f"  {'-'*20}{'-'*9}{'-'*8}{'-'*6}{'-'*9}{'-'*10}{'-'*15}")
        for name, en, conf, hr in VARIANTS:
            r = sim(sigs, asset_cfg, en, conf, hr)
            print(f"  {name:<20}{r['return_pct']:>+8.2f}%{r['trades']:>8}"
                  f"{r['win_rate']:>5.0f}%{r['worst']:>+8.2f}%"
                  f"{r['qr_fires']:>10}{r['shield_blocks']:>15}")


if __name__ == '__main__':
    main()
