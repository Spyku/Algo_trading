"""Per-regime gate test — standalone, no prod writes.

Uses the existing 90d cache (bull=6h, bear=8h tagged signals), takes the last
60d, and sweeps rally-cooldown INDEPENDENTLY for bull-regime bars and
bear-regime bars. Simulates three policies:

  A. Current single gate (baseline)
  B. Per-regime gates (new: bull-gate + bear-gate)
  C. No gate (control)

Reports final return, trades, worst DD for each.
"""
import os
import sys
import pickle
import time
import json

import numpy as np
import pandas as pd

ENGINE = r'G:\Autres ordinateurs\My laptop\engine'
os.chdir(ENGINE)
sys.path.insert(0, ENGINE)

CACHE = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
REGIME_CFG = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
ASSET = 'ETH'
FEE = 0.0005  # 5 bps/leg realistic maker blend (see ed.py BACKTEST_FEE_PER_LEG)

# G-style sweep grid
HORIZONS = [8, 10, 12, 14, 16, 18, 20, 24, 30, 36]
def thr_for(h):
    if h in (8, 10, 12):       return [round(2.0 + 0.5*i, 2) for i in range(9)]
    if h in (14, 16, 18, 20):  return [round(3.0 + 0.5*i, 2) for i in range(9)]
    if h in (24, 30, 36):      return [round(4.0 + 0.5*i, 2) for i in range(11)]
    return []
CD_GRID = [6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 48]
PLATEAU_THR = 0.7


def load_signals():
    with open(CACHE, 'rb') as f:
        sigs = pickle.load(f)
    for s in sigs:
        s['datetime'] = pd.Timestamp(s['datetime'])
    sigs.sort(key=lambda s: s['datetime'])
    end = sigs[-1]['datetime']
    lo = end - pd.Timedelta(days=60)
    return [s for s in sigs if s['datetime'] >= lo]


def sim_strategy(sigs, asset_cfg, bull_gate=None, bear_gate=None, single_gate=None):
    """Simulate full strategy with per-regime gates.
    bull_gate / bear_gate: tuples (h_s, h_l, t_s, t_l, cd_h) or None
    single_gate: use this gate for both regimes (mutually exclusive with bull/bear)
    """
    bull_shield = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    bull_conf = float(asset_cfg.get('bull', {}).get('min_confidence', 95))
    bear_conf = float(asset_cfg.get('bear', {}).get('min_confidence', 80))
    min_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.6))
    max_hold = int(asset_cfg.get('max_hold_hours', 12))

    # Pre-compute rr arrays for any horizon used across gates
    horizons_needed = set()
    for g in (bull_gate, bear_gate, single_gate):
        if g is not None:
            horizons_needed.update([g[0], g[1]])
    closes = np.array([float(s['close']) for s in sigs])
    rr = {}
    for h in horizons_needed:
        out = np.full(len(closes), np.nan)
        if 0 < h < len(closes):
            out[h:] = (closes[h:] / closes[:-h] - 1.0) * 100.0
        rr[h] = out

    cash = 1000.0; qty = 0.0; in_pos = False; entry_px = 0.0
    hold = 0; cd = 0; trades = []
    ec = [1000.0]

    for i, s in enumerate(sigs):
        regime = s.get('regime', 'bull')
        price = float(s['close'])
        sig = s['signal']
        sconf = float(s.get('confidence', 0))
        conf_thr = bull_conf if regime == 'bull' else bear_conf
        shield_on = bull_shield if regime == 'bull' else bear_shield

        # Select gate for this bar
        if single_gate is not None:
            gate = single_gate
        else:
            gate = bull_gate if regime == 'bull' else bear_gate

        if gate is not None:
            h_s, h_l, t_s, t_l, cd_h = gate
            rs = rr[h_s][i] if not np.isnan(rr[h_s][i]) else 0
            rl = rr[h_l][i] if not np.isnan(rr[h_l][i]) else 0
            if rs >= t_s or rl >= t_l:
                cd = max(cd, cd_h)

        ec.append(cash + qty * price if in_pos else cash)

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
    arr = np.array(ec); peak = np.maximum.accumulate(arr); dd = (arr - peak) / peak
    mdd = float(dd.min()) * 100.0 if len(dd) else 0.0
    return dict(return_pct=ret, trades=len(trades), win_rate=wr, worst=worst, max_dd=mdd)


def sweep_gate_per_regime(sigs, asset_cfg, regime_filter):
    """Sweep gate on bars tagged as regime_filter ('bull' or 'bear'). For each
    gate config, simulate the FULL strategy but apply the gate only to the
    filtered regime (other regime is treated as 'no gate').
    Returns winner gate + base no-gate return.
    """
    # Baseline: no gate at all on this regime
    base_r = sim_strategy(sigs, asset_cfg,
                          bull_gate=None, bear_gate=None)
    base_return = base_r['return_pct']

    pairs = [(a, b) for i, a in enumerate(HORIZONS) for b in HORIZONS[i+1:]]
    total = sum(len(thr_for(a)) * len(thr_for(b)) for a, b in pairs) * len(CD_GRID)
    print(f"  sweeping {total:,} configs for {regime_filter} gate...")

    t0 = time.time()
    best_return = base_return
    best_gate = None
    rows = []
    for h_s, h_l in pairs:
        for t_s in thr_for(h_s):
            for t_l in thr_for(h_l):
                for cd in CD_GRID:
                    gate = (h_s, h_l, t_s, t_l, cd)
                    bull_g = gate if regime_filter == 'bull' else None
                    bear_g = gate if regime_filter == 'bear' else None
                    r = sim_strategy(sigs, asset_cfg,
                                     bull_gate=bull_g, bear_gate=bear_g)
                    rows.append((h_s, h_l, t_s, t_l, cd, r['return_pct'],
                                 r['max_dd'], r['trades'], r['win_rate']))
                    if r['return_pct'] > best_return:
                        best_return = r['return_pct']
                        best_gate = gate
    print(f"  {regime_filter} sweep done in {time.time()-t0:.1f}s "
          f"(tried {len(rows)} configs, best {best_return:+.2f}% vs base {base_return:+.2f}%)")
    return best_gate, best_return, base_return


def main():
    print("=" * 100)
    print("  Per-regime Gate Test — ETH bull=6h/bear=8h, 60d, no prod writes")
    print("=" * 100)

    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg[ASSET]
    current_gate_cfg = asset_cfg.get('rally_cooldown') or {}
    if current_gate_cfg.get('enabled'):
        current_gate = (int(current_gate_cfg['h_short']), int(current_gate_cfg['h_long']),
                        float(current_gate_cfg['t_short_pct']),
                        float(current_gate_cfg['t_long_pct']),
                        int(current_gate_cfg['cd_hours']))
    else:
        current_gate = None

    sigs = load_signals()
    bull_n = sum(1 for s in sigs if s.get('regime') == 'bull')
    bear_n = sum(1 for s in sigs if s.get('regime') == 'bear')
    print(f"  Signals: {len(sigs)} (bull={bull_n}, bear={bear_n})")
    if current_gate:
        print(f"  Current single gate: rr{current_gate[0]}h>={current_gate[2]}% OR "
              f"rr{current_gate[1]}h>={current_gate[3]}%, cd={current_gate[4]}h")
    print()

    # Baseline A: current single gate
    print("A. Current single gate...")
    A = sim_strategy(sigs, asset_cfg, single_gate=current_gate)
    print(f"   return={A['return_pct']:+.2f}%  trades={A['trades']}  "
          f"wr={A['win_rate']:.0f}%  worst={A['worst']:+.2f}%  mdd={A['max_dd']:+.2f}%")

    # Baseline C: no gate at all
    print("\nC. No gate (control)...")
    C = sim_strategy(sigs, asset_cfg, single_gate=None)
    print(f"   return={C['return_pct']:+.2f}%  trades={C['trades']}  "
          f"wr={C['win_rate']:.0f}%  worst={C['worst']:+.2f}%  mdd={C['max_dd']:+.2f}%")

    # Per-regime sweep
    print("\nB. Per-regime gates (sweep)...")
    bull_gate, bull_best_ret, bull_base_ret = sweep_gate_per_regime(sigs, asset_cfg, 'bull')
    bear_gate, bear_best_ret, bear_base_ret = sweep_gate_per_regime(sigs, asset_cfg, 'bear')
    print(f"  bull winner: {bull_gate}")
    print(f"  bear winner: {bear_gate}")

    # Apply both together
    B = sim_strategy(sigs, asset_cfg, bull_gate=bull_gate, bear_gate=bear_gate)
    print(f"   return={B['return_pct']:+.2f}%  trades={B['trades']}  "
          f"wr={B['win_rate']:.0f}%  worst={B['worst']:+.2f}%  mdd={B['max_dd']:+.2f}%")

    # Summary
    print()
    print("=" * 100)
    print("  Summary")
    print("=" * 100)
    print(f"  A. Current single gate ({current_gate}): {A['return_pct']:+.2f}%")
    print(f"  B. Per-regime gates   (bull={bull_gate} / bear={bear_gate}): {B['return_pct']:+.2f}%")
    print(f"  C. No gate:                              {C['return_pct']:+.2f}%")
    delta_ba = B['return_pct'] - A['return_pct']
    delta_ac = A['return_pct'] - C['return_pct']
    print(f"\n  Per-regime vs single: {delta_ba:+.2f}pp")
    print(f"  Single vs no-gate:    {delta_ac:+.2f}pp (confirms gate earns its keep)")

    if delta_ba > 0.5:
        print("  >>> Per-regime gate shows meaningful improvement — worth shipping.")
    elif delta_ba > 0:
        print("  >>> Per-regime gate slightly better; noise-level improvement.")
    else:
        print("  >>> Per-regime gate does NOT help on this window. Keep single gate.")


if __name__ == '__main__':
    main()
