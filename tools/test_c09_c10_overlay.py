"""
C09 + C10 PnL overlay simulation.

Runs baseline (current live config, ETH 90d cache) then applies:
  C09: skip BUYs where rv_24h > p90 of 30d window
  C10: skip BUYs where BTC 24h logret < 0

Plus a small sweep over alternate thresholds since the binary versions look
extreme on skip rate (C09: 3% too thin, C10: 54% too aggressive).
"""
import os, sys, json, pickle
from datetime import datetime
import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)

CACHE = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
BTC_CSV = os.path.join(ENGINE, 'data', 'btc_hourly_data.csv')
CFG = os.path.join(ENGINE, 'config', 'regime_config_ed.json')

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
    """Replicate trader rally-cooldown gate. Returns True if BUY blocked."""
    if not gate_cfg.get('enabled', False):
        return False
    h_short = gate_cfg.get('h_short', 0)
    h_long  = gate_cfg.get('h_long', 0)
    t_s_pct = gate_cfg.get('t_short_pct', 999)
    t_l_pct = gate_cfg.get('t_long_pct', 999)
    cd_h    = gate_cfg.get('cd_hours', 0)
    p_now = float(sigs[i]['close'])
    fired_at = None
    for h, t in ((h_short, t_s_pct), (h_long, t_l_pct)):
        if h <= 0 or t >= 999:
            continue
        j = i - h
        if j < 0:
            continue
        p_then = float(sigs[j]['close'])
        rr = (p_now / p_then - 1.0) * 100.0
        if rr >= t:
            fired_at = i
            break
    if fired_at is None:
        return False
    # Walk back to find earliest trigger within cd_h
    for k in range(max(0, i - int(cd_h)), i + 1):
        for h, t in ((h_short, t_s_pct), (h_long, t_l_pct)):
            if h <= 0 or t >= 999:
                continue
            j2 = k - h
            if j2 < 0:
                continue
            p_a = float(sigs[k]['close'])
            p_b = float(sigs[j2]['close'])
            rrk = (p_a / p_b - 1.0) * 100.0
            if rrk >= t:
                if (i - k) < int(cd_h):
                    return True
    return False

def simulate(sigs, cfg, skip_mask=None):
    """All-in/all-out simulator. skip_mask = same length as sigs, True=skip BUY."""
    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry_px = 0.0
    hold = 0
    trades = []
    skipped = 0
    for i, s in enumerate(sigs):
        regime = s['regime']
        sig    = s['signal']
        sconf  = float(s.get('confidence', 0))
        price  = float(s['close'])
        thr    = cfg['bull_thr'] if regime == 'bull' else cfg['bear_thr']
        shield_on = cfg['bull_shield'] if regime == 'bull' else cfg['bear_shield']
        gate_cfg  = cfg['bull_gate']   if regime == 'bull' else cfg['bear_gate']
        if in_pos:
            hold += 1
        if sig == 'BUY' and sconf >= thr and not in_pos:
            if _gate_blocks_buy(gate_cfg, s['datetime'], i, sigs):
                continue
            if skip_mask is not None and skip_mask[i]:
                skipped += 1
                continue
            qty = cash * (1 - FEE_PER_LEG) / price
            cash = 0.0
            in_pos = True
            entry_px = price
            hold = 0
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
                trades.append({'pnl_pct': cur_pnl, 'hold_hours': hold})
                qty = 0.0
                in_pos = False
    if in_pos:
        cash = qty * sigs[-1]['close'] * (1 - FEE_PER_LEG)
        trades.append({'pnl_pct': (sigs[-1]['close']/entry_px - 1.0)*100.0, 'hold_hours': hold})
    return (cash / 1000.0 - 1.0) * 100.0, trades, skipped

def fmt_trades(trades):
    if not trades:
        return "0 trades"
    n = len(trades)
    wr = sum(1 for t in trades if t['pnl_pct'] > 0) / n * 100
    avg = np.mean([t['pnl_pct'] for t in trades])
    return f"{n} trades, WR={wr:.0f}%, avg_pnl={avg:+.2f}%"

def main():
    cfg = load_cfg()
    print(f"Live config: bull@{cfg['bull_thr']}% sh={cfg['bull_shield']} / "
          f"bear@{cfg['bear_thr']}% sh={cfg['bear_shield']} / "
          f"shield {cfg['min_sell_pnl_pct']}%/{cfg['max_hold_hours']}h")
    print(f"             bear_gate enabled={cfg['bear_gate'].get('enabled', False)}")
    print()

    with open(CACHE, 'rb') as f:
        sigs_raw = pickle.load(f)
    sigs = sigs_raw if isinstance(sigs_raw, list) else sigs_raw['signals']
    for s in sigs:
        if not isinstance(s['datetime'], (pd.Timestamp, datetime)):
            s['datetime'] = pd.to_datetime(s['datetime'])
    print(f"Cache: {len(sigs)} signals, {sigs[0]['datetime']} -> {sigs[-1]['datetime']}")
    print()

    # -------- Baseline --------
    base_ret, base_trades, _ = simulate(sigs, cfg)
    print(f"BASELINE: return={base_ret:+7.2f}%  {fmt_trades(base_trades)}")
    print()

    # -------- C09 vol_entry_filter sweep --------
    df = pd.DataFrame(sigs)
    df['logret'] = np.log(df['close']).diff()
    df['rv_24h'] = df['logret'].rolling(24).std()
    print("=== C09 vol_entry_filter sweep ===")
    print(f"{'pctile':>8}  {'window':>8}  {'skip_n':>7}  {'return':>8}  {'delta':>8}  trades")
    for win in (24*30, 24*7):
        for pctile in (0.95, 0.90, 0.85, 0.80, 0.70):
            df['rv_pct'] = df['rv_24h'].rolling(win).quantile(pctile)
            mask = (df['rv_24h'] > df['rv_pct']).fillna(False).values
            ret, trades, skipped = simulate(sigs, cfg, skip_mask=mask)
            d = ret - base_ret
            print(f"  p{int(pctile*100):2d}    {win:>5}h    {skipped:>4}     {ret:+7.2f}%  {d:+7.2f}pp  {fmt_trades(trades)}")
    print()

    # -------- C10 btc_momentum_gate sweep --------
    btc = pd.read_csv(BTC_CSV)
    btc['datetime'] = pd.to_datetime(btc['datetime'])
    print("=== C10 btc_momentum_gate sweep ===")
    print(f"{'window':>8}  {'thresh':>10}  {'skip_n':>7}  {'return':>8}  {'delta':>8}  trades")
    for win_h in (6, 12, 24, 48, 72):
        btc[f'btc_lret_{win_h}h'] = np.log(btc['close']).diff(win_h)
        merged = pd.merge(df[['datetime']], btc[['datetime', f'btc_lret_{win_h}h']],
                          on='datetime', how='left')
        col = merged[f'btc_lret_{win_h}h'].values
        for thr_pct in (0.0, -0.005, -0.01, -0.02, -0.03):
            # Skip BUY when btc lret < thr (more negative threshold = harder to skip)
            mask = (col < thr_pct)
            mask = pd.Series(mask).fillna(False).values
            ret, trades, skipped = simulate(sigs, cfg, skip_mask=mask)
            d = ret - base_ret
            print(f"  {win_h:>3}h     {thr_pct:+.3f}     {skipped:>4}     {ret:+7.2f}%  {d:+7.2f}pp  {fmt_trades(trades)}")
    print()

    print("Verdict guide: ship if delta ≥ +5pp on baseline (HRST run-to-run noise band).")

if __name__ == '__main__':
    main()
