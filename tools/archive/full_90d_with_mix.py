"""
full_90d_with_mix.py — replay the FULL 90d cache with MIX gate. Show
cumulative dollar performance starting from $12,000 on Jan 27, 2026.
Compare current PROD vs MIX gate vs B&H.
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
FEE = 0.0005
START_CAPITAL = 12000.0
START_DATE = pd.Timestamp('2026-01-27', tz='UTC')


def load_signals():
    with open(CACHE_PKL, 'rb') as f:
        sigs = pickle.load(f)
    for s in sigs:
        t = pd.Timestamp(s['datetime'])
        s['datetime'] = t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    sigs.sort(key=lambda s: s['datetime'])
    return sigs


def build_rr(sigs, horizons):
    df = pd.DataFrame([{'datetime': s['datetime'], 'close': s['close']} for s in sigs])
    df = df.sort_values('datetime').reset_index(drop=True)
    return {h: ((df['close'] / df['close'].shift(h) - 1.0) * 100.0).values for h in horizons}


def simulate(sigs, asset_cfg, gate=None, start_capital=START_CAPITAL):
    bull_shield = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    bull_conf = float(asset_cfg.get('bull', {}).get('min_confidence', 75.0))
    bear_conf = float(asset_cfg.get('bear', {}).get('min_confidence', 65.0))
    min_sell_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.5))
    max_hold = int(asset_cfg.get('max_hold_hours', 10))

    h_s = h_l = 0; t_s = t_l = 9999.0; cd_h = 0
    rs_arr = rl_arr = None
    if gate is not None:
        h_s = int(gate['h_short']); h_l = int(gate['h_long'])
        t_s = float(gate['t_short_pct']); t_l = float(gate['t_long_pct'])
        cd_h = int(gate['cd_hours'])
        rr_dict = build_rr(sigs, [h_s, h_l])
        rs_arr = rr_dict.get(h_s); rl_arr = rr_dict.get(h_l)

    cash = start_capital; qty = 0.0; in_pos = False; entry = 0.0
    hold = 0; cd = 0
    trade_pnls = []
    trade_log = []
    eq_curve = []
    skipped = 0

    for i in range(len(sigs)):
        s = sigs[i]; price = s['close']; dt = s['datetime']
        regime = s.get('regime', 'bull')
        conf_thr = bull_conf if regime == 'bull' else bear_conf
        shield_on = bull_shield if regime == 'bull' else bear_shield
        if cd_h > 0 and rs_arr is not None and rl_arr is not None:
            rs = rs_arr[i]; rl = rl_arr[i]
            if (rs == rs and rs >= t_s) or (rl == rl and rl >= t_l):
                cd = max(cd, cd_h)
        cur_eq = cash + qty * price if in_pos else cash
        eq_curve.append({'datetime': dt, 'equity': cur_eq})
        if s['signal'] == 'BUY' and s['confidence'] >= conf_thr and not in_pos:
            if cd > 0:
                skipped += 1
            else:
                qty = cash * (1 - FEE) / price
                cash = 0.0; in_pos = True; entry = price; hold = 0
                trade_log.append({'dt': dt, 'side': 'BUY', 'price': price, 'qty': qty})
        elif s['signal'] == 'SELL' and in_pos:
            cur = (price / entry - 1.0) * 100.0
            shield_min = min_sell_pnl if shield_on else 0.0
            if cur >= shield_min or hold >= max_hold:
                cash_recv = qty * price * (1 - FEE)
                trade_pnls.append(cur)
                trade_log.append({'dt': dt, 'side': 'SELL', 'price': price,
                                  'pnl_pct': cur, 'pnl_usd': cash_recv - (qty * entry)})
                cash = cash_recv
                in_pos = False; qty = 0.0; entry = 0.0; hold = 0
        if in_pos: hold += 1
        if cd > 0: cd -= 1
    if in_pos:
        cash = qty * sigs[-1]['close'] * (1 - FEE)
        cur_pnl = (sigs[-1]['close'] / entry - 1.0) * 100.0
        trade_pnls.append(cur_pnl)
        trade_log.append({'dt': sigs[-1]['datetime'], 'side': 'SELL_FINAL',
                          'price': sigs[-1]['close'], 'pnl_pct': cur_pnl,
                          'pnl_usd': cash - (qty * entry)})

    final_eq = cash
    pnl_pct = (final_eq / start_capital - 1.0) * 100.0
    pnl_usd = final_eq - start_capital
    n = len(trade_pnls)
    wins = sum(1 for p in trade_pnls if p > 0)
    wr = wins / n * 100 if n else 0
    eq_df = pd.DataFrame(eq_curve)
    peak = eq_df['equity'].cummax()
    dd = (eq_df['equity'] - peak) / peak * 100
    mdd = float(dd.min()) if len(dd) else 0
    return {
        'final_equity': final_eq,
        'pnl_pct': pnl_pct,
        'pnl_usd': pnl_usd,
        'n_trades': n,
        'win_rate': wr,
        'max_dd_pct': mdd,
        'skipped': skipped,
        'trade_log': trade_log,
        'eq_curve': eq_df,
    }


def main():
    print("Loading...")
    sigs = load_signals()
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    # Slice to start from 2026-01-27
    sigs = [s for s in sigs if s['datetime'] >= START_DATE]
    print(f"  Cache slice from {START_DATE.date()}: {len(sigs)} signals "
          f"({sigs[0]['datetime']} to {sigs[-1]['datetime']})")
    days_in_period = (sigs[-1]['datetime'] - sigs[0]['datetime']).days
    print(f"  Total span: {days_in_period} days")

    # B&H
    bh_pnl = (sigs[-1]['close'] / sigs[0]['close'] * (1 - FEE) * (1 - FEE) - 1) * 100
    bh_usd = START_CAPITAL * (sigs[-1]['close'] / sigs[0]['close']) * (1 - FEE) * (1 - FEE)

    # Setups
    current_prod_gate = {'h_short': 20, 'h_long': 24,
                         't_short_pct': 4.0, 't_long_pct': 4.5, 'cd_hours': 12}
    mix_gate = {'h_short': 12, 'h_long': 20,
                't_short_pct': 2.5, 't_long_pct': 4.0, 'cd_hours': 24}

    r_no_gate = simulate(sigs, asset_cfg, gate=None)
    r_prod = simulate(sigs, asset_cfg, gate=current_prod_gate)
    r_mix = simulate(sigs, asset_cfg, gate=mix_gate)

    print(f"\n{'='*100}")
    print(f"  $12,000 on 2026-01-27 -> 2026-04-18 ({days_in_period}d, ETH cache)")
    print(f"{'='*100}\n")
    print(f"  Starting capital:               $12,000.00")
    print(f"  ETH price 2026-01-27 12:00 UTC: ${sigs[0]['close']:.2f}")
    print(f"  ETH price 2026-04-18 12:00 UTC: ${sigs[-1]['close']:.2f}")
    print()

    print(f"{'Setup':<55} {'Final $':>13} {'Return %':>10} {'PnL $':>13} {'Trades':>7} {'WR':>5} {'Max DD':>8}")
    print("-" * 113)
    print(f"{'Buy & hold ETH':<55} ${bh_usd:>11,.2f} {bh_pnl:>+9.2f}% ${bh_usd-START_CAPITAL:>+11,.2f} {'1':>7} {'—':>5} {'—':>8}")
    print(f"{'No gate (pure model + shields)':<55} ${r_no_gate['final_equity']:>11,.2f} {r_no_gate['pnl_pct']:>+9.2f}% ${r_no_gate['pnl_usd']:>+11,.2f} {r_no_gate['n_trades']:>7} {r_no_gate['win_rate']:>4.0f}% {r_no_gate['max_dd_pct']:>+7.2f}%")
    print(f"{'Current PROD (rr20>=4 OR rr24>=4.5 cd=12)':<55} ${r_prod['final_equity']:>11,.2f} {r_prod['pnl_pct']:>+9.2f}% ${r_prod['pnl_usd']:>+11,.2f} {r_prod['n_trades']:>7} {r_prod['win_rate']:>4.0f}% {r_prod['max_dd_pct']:>+7.2f}%")
    print(f"{'**MIX gate (rr12>=2.5 OR rr20>=4 cd=24)**':<55} ${r_mix['final_equity']:>11,.2f} {r_mix['pnl_pct']:>+9.2f}% ${r_mix['pnl_usd']:>+11,.2f} {r_mix['n_trades']:>7} {r_mix['win_rate']:>4.0f}% {r_mix['max_dd_pct']:>+7.2f}%")
    print()
    print(f"  MIX vs PROD delta:    +${r_mix['final_equity'] - r_prod['final_equity']:,.2f}  ({r_mix['pnl_pct'] - r_prod['pnl_pct']:+.2f}pp)")
    print(f"  MIX vs B&H delta:     +${r_mix['final_equity'] - bh_usd:,.2f}  ({r_mix['pnl_pct'] - bh_pnl:+.2f}pp)")
    print(f"  MIX vs no-gate delta: +${r_mix['final_equity'] - r_no_gate['final_equity']:,.2f}  ({r_mix['pnl_pct'] - r_no_gate['pnl_pct']:+.2f}pp)")

    # Per-month breakdown
    print(f"\n{'='*100}")
    print(f"  Monthly equity curve (MIX gate, $12k starting capital)")
    print(f"{'='*100}\n")
    eq = r_mix['eq_curve'].copy()
    eq['month'] = eq['datetime'].dt.to_period('M')
    monthly_last = eq.groupby('month').last().reset_index()
    monthly_first = eq.groupby('month').first().reset_index()
    prev_eq = START_CAPITAL
    for i, row in monthly_last.iterrows():
        first_eq = monthly_first.iloc[i]['equity']
        last_eq = row['equity']
        m_pnl = (last_eq / prev_eq - 1) * 100
        m_usd = last_eq - prev_eq
        print(f"  End of {str(row['month'])}: equity ${last_eq:>11,.2f}  | "
              f"month {m_pnl:>+6.2f}%  ${m_usd:>+9,.2f}  | "
              f"cum {(last_eq/START_CAPITAL-1)*100:>+6.2f}%")
        prev_eq = last_eq

    # Show MIX trades summary
    print(f"\n{'='*100}")
    print(f"  MIX gate trade summary ({r_mix['n_trades']} trades, {r_mix['skipped']} blocks)")
    print(f"{'='*100}\n")
    sells = [t for t in r_mix['trade_log'] if t['side'] in ('SELL', 'SELL_FINAL')]
    print(f"  Best trade:    +{max(t['pnl_pct'] for t in sells):.2f}%")
    print(f"  Worst trade:   {min(t['pnl_pct'] for t in sells):+.2f}%")
    print(f"  Median trade:  {np.median([t['pnl_pct'] for t in sells]):+.2f}%")
    print(f"  Avg trade:     {np.mean([t['pnl_pct'] for t in sells]):+.2f}%")
    print(f"  Win rate:      {r_mix['win_rate']:.1f}%")
    print(f"  Max DD:        {r_mix['max_dd_pct']:+.2f}%")


if __name__ == '__main__':
    main()
