"""
C02 — Multi-horizon emergency-exit ensemble overlay.

Hypothesis: force exit when h5 AND h8 both flip SELL within window W (1h, 2h, 3h),
bypassing shield + max_hold. Captures model-recognized regime change earlier
than the live single-horizon signal cycle.

Distinct from:
  - C24 (5-min price-action overlay) — that was REJECTED 2026-04-27 (35 variants)
  - C14 (triple-barrier exit overlay) — different mechanism
  - T1b ensemble vote (entry-side) — already shelved 2026-04-27

Method:
  1. Use eth_sl_signals_90d.pkl as baseline (matches C09/C10 baseline at +147.59%)
  2. Look up h5 and h8 signals from eth_per_horizon_signals_90d.pkl by timestamp
  3. Sweep: window W ∈ {0h, 1h, 2h, 3h}, conf threshold for SELL ∈ {0, 70, 80, 90}
     Only count h5/h8 SELL if conf >= threshold AND within last W hours.

Decision rule: any variant beats baseline by >=+5pp -> PROMISING (escalate),
positive -> MARGINAL, negative -> DEAD.
"""
import os, sys, json, pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)

CFG = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
SL_CACHE = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
PH_CACHE = os.path.join(ENGINE, 'data', 'eth_per_horizon_signals_90d.pkl')
FEE_PER_LEG = 0.0005


def load_cfg():
    with open(CFG) as f:
        cfg = json.load(f)
    eth = cfg['ETH']
    return {
        'bull_thr': float(eth['bull']['min_confidence']),
        'bear_thr': float(eth['bear']['min_confidence']),
        'bull_shield': bool(eth['bull'].get('hold_shield', True)),
        'bear_shield': bool(eth['bear'].get('hold_shield', True)),
        'min_sell_pnl_pct': float(eth.get('min_sell_pnl_pct', 0.5)),
        'max_hold_hours': float(eth.get('max_hold_hours', 10)),
        'bull_gate': eth['bull'].get('rally_cooldown', {'enabled': False}),
        'bear_gate': eth['bear'].get('rally_cooldown', {'enabled': False}),
    }


def _gate_blocks_buy(gate_cfg, ts, i, sigs):
    if not gate_cfg.get('enabled', False):
        return False
    h_short = gate_cfg.get('h_short', 0)
    h_long = gate_cfg.get('h_long', 0)
    t_s = gate_cfg.get('t_short_pct', 999)
    t_l = gate_cfg.get('t_long_pct', 999)
    cd_h = gate_cfg.get('cd_hours', 0)
    p_now = float(sigs[i]['close'])
    fired = False
    for h, t in ((h_short, t_s), (h_long, t_l)):
        if h <= 0 or t >= 999:
            continue
        j = i - h
        if j < 0:
            continue
        if (p_now / float(sigs[j]['close']) - 1.0) * 100.0 >= t:
            fired = True
            break
    if not fired:
        return False
    for k in range(max(0, i - int(cd_h)), i + 1):
        for h, t in ((h_short, t_s), (h_long, t_l)):
            if h <= 0 or t >= 999:
                continue
            j2 = k - h
            if j2 < 0:
                continue
            rrk = (float(sigs[k]['close']) / float(sigs[j2]['close']) - 1.0) * 100.0
            if rrk >= t and (i - k) < int(cd_h):
                return True
    return False


def simulate(sigs, cfg, h5_idx=None, h8_idx=None, window=0, conf_thr=0):
    """Baseline simulator + optional multi-horizon emergency-exit overlay.

    Overlay fires when in_pos AND (h5 had SELL with conf>=conf_thr within last
    `window+1` hours) AND (h8 had SELL with conf>=conf_thr within last `window+1`
    hours). Bypasses shield + max_hold.
    """
    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry_px = 0.0
    hold = 0
    trades = []
    emergency_exits = 0

    for i, s in enumerate(sigs):
        regime = s['regime']
        sig = s['signal']
        sconf = float(s.get('confidence', 0))
        price = float(s['close'])
        thr = cfg['bull_thr'] if regime == 'bull' else cfg['bear_thr']
        shield_on = cfg['bull_shield'] if regime == 'bull' else cfg['bear_shield']
        gate_cfg = cfg['bull_gate'] if regime == 'bull' else cfg['bear_gate']

        if in_pos:
            hold += 1

        # ----- Multi-horizon emergency-exit overlay -----
        emergency_fire = False
        if in_pos and h5_idx is not None and h8_idx is not None:
            ts = s['datetime']
            h5_hit = False
            h8_hit = False
            for w in range(window + 1):
                tcheck = ts - timedelta(hours=w)
                h5s = h5_idx.get(tcheck)
                h8s = h8_idx.get(tcheck)
                if h5s and h5s['signal'] == 'SELL' and float(h5s.get('confidence', 0)) >= conf_thr:
                    h5_hit = True
                if h8s and h8s['signal'] == 'SELL' and float(h8s.get('confidence', 0)) >= conf_thr:
                    h8_hit = True
                if h5_hit and h8_hit:
                    break
            emergency_fire = h5_hit and h8_hit

        if sig == 'BUY' and sconf >= thr and not in_pos:
            if _gate_blocks_buy(gate_cfg, s['datetime'], i, sigs):
                continue
            qty = cash * (1 - FEE_PER_LEG) / price
            cash = 0.0
            in_pos = True
            entry_px = price
            hold = 0
        elif in_pos and emergency_fire:
            cur_pnl = (price / entry_px - 1.0) * 100.0
            cash = qty * price * (1 - FEE_PER_LEG)
            trades.append({'pnl_pct': cur_pnl, 'hold_hours': hold, 'reason': 'emergency'})
            qty = 0.0
            in_pos = False
            emergency_exits += 1
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1.0) * 100.0
            do_sell = False
            if not shield_on:
                do_sell = True
            elif cur_pnl >= cfg['min_sell_pnl_pct']:
                do_sell = True
            elif hold >= cfg['max_hold_hours']:
                do_sell = True
            if do_sell:
                cash = qty * price * (1 - FEE_PER_LEG)
                trades.append({'pnl_pct': cur_pnl, 'hold_hours': hold, 'reason': 'model'})
                qty = 0.0
                in_pos = False

    if in_pos:
        cash = qty * sigs[-1]['close'] * (1 - FEE_PER_LEG)
        trades.append({
            'pnl_pct': (sigs[-1]['close'] / entry_px - 1.0) * 100.0,
            'hold_hours': hold, 'reason': 'eod'
        })
    return (cash / 1000.0 - 1.0) * 100.0, trades, emergency_exits


def fmt_trades(trades):
    if not trades:
        return "0 trades"
    n = len(trades)
    wr = sum(1 for t in trades if t['pnl_pct'] > 0) / n * 100
    avg = np.mean([t['pnl_pct'] for t in trades])
    em = sum(1 for t in trades if t.get('reason') == 'emergency')
    return f"{n} trades, WR={wr:.0f}%, avg={avg:+.2f}%, em={em}"


def main():
    cfg = load_cfg()
    print(f"Live config: bull@{cfg['bull_thr']}% sh={cfg['bull_shield']} / "
          f"bear@{cfg['bear_thr']}% sh={cfg['bear_shield']} / "
          f"shield {cfg['min_sell_pnl_pct']}%/{cfg['max_hold_hours']}h")
    print()

    with open(SL_CACHE, 'rb') as f:
        sl = pickle.load(f)
    sigs = sl if isinstance(sl, list) else sl['signals']
    for s in sigs:
        if not isinstance(s['datetime'], (pd.Timestamp, datetime)):
            s['datetime'] = pd.to_datetime(s['datetime'])

    with open(PH_CACHE, 'rb') as f:
        ph = pickle.load(f)
    h5_idx = {pd.to_datetime(s['datetime']): s for s in ph[5]}
    h8_idx = {pd.to_datetime(s['datetime']): s for s in ph[8]}

    # Restrict baseline signals to the per-horizon overlap window
    ph_min = min(min(h5_idx.keys()), min(h8_idx.keys()))
    ph_max = max(max(h5_idx.keys()), max(h8_idx.keys()))
    sigs = [s for s in sigs if ph_min <= s['datetime'] <= ph_max]
    print(f"Aligned window: {sigs[0]['datetime']} -> {sigs[-1]['datetime']}  ({len(sigs)} bars)")
    print()

    # ----- Baseline (no overlay, on aligned window) -----
    base_ret, base_trades, _ = simulate(sigs, cfg, h5_idx=None, h8_idx=None)
    print(f"BASELINE (aligned window, no overlay):  return={base_ret:+7.2f}%  {fmt_trades(base_trades)}")
    print()

    # ----- C02 sweep -----
    print(f"=== C02 multi-horizon emergency-exit sweep ===")
    print(f"{'window':>8}  {'conf_thr':>9}  {'em_fires':>9}  {'return':>8}  {'delta':>9}  trades")
    rows = []
    for window in (0, 1, 2, 3):
        for conf_thr in (0, 70, 80, 90):
            ret, trades, em = simulate(sigs, cfg, h5_idx=h5_idx, h8_idx=h8_idx,
                                        window=window, conf_thr=conf_thr)
            d = ret - base_ret
            rows.append((window, conf_thr, em, ret, d, trades))
            print(f"  W={window}h     thr>={conf_thr:>3}     {em:>4}        {ret:+7.2f}%   {d:+7.2f}pp  {fmt_trades(trades)}")
    print()

    rows.sort(key=lambda r: -r[4])
    print(f"Best variant: W={rows[0][0]}h thr>={rows[0][1]} delta={rows[0][4]:+.2f}pp")
    print()
    print("Verdict: ship if best delta >= +5pp on baseline (HRST run-to-run noise band).")


if __name__ == '__main__':
    main()
