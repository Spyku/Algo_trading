"""
decompose_mix_gain.py — break down the +$9,189 MIX gate gain.
Show what drove it: gates, shields, big trades, specific months, skipped entries.
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
    return [s for s in sigs if s['datetime'] >= START_DATE]


def build_rr(sigs, horizons):
    df = pd.DataFrame([{'datetime': s['datetime'], 'close': s['close']} for s in sigs])
    df = df.sort_values('datetime').reset_index(drop=True)
    return {h: ((df['close'] / df['close'].shift(h) - 1.0) * 100.0).values for h in horizons}


def simulate(sigs, asset_cfg, gate=None, bull_shield_override=None,
             bear_shield_override=None, capital=START_CAPITAL, log_skipped=False):
    bull_shield_d = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield_d = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    bull_shield = bull_shield_d if bull_shield_override is None else bull_shield_override
    bear_shield = bear_shield_d if bear_shield_override is None else bear_shield_override
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

    cash = capital; qty = 0.0; in_pos = False; entry = 0.0
    entry_dt = None; entry_regime = None
    hold = 0; cd = 0
    trades = []  # full trade records
    skipped = []  # skipped entry records (price/dt/conf/regime)
    eq_curve = []

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
        eq_curve.append({'datetime': dt, 'equity': cur_eq, 'in_pos': in_pos})
        if s['signal'] == 'BUY' and s['confidence'] >= conf_thr and not in_pos:
            if cd > 0:
                if log_skipped:
                    skipped.append({'dt': dt, 'price': price, 'conf': s['confidence'],
                                    'regime': regime})
            else:
                qty = cash * (1 - FEE) / price
                cash = 0.0; in_pos = True; entry = price; hold = 0
                entry_dt = dt; entry_regime = regime
        elif s['signal'] == 'SELL' and in_pos:
            cur = (price / entry - 1.0) * 100.0
            shield_min = min_sell_pnl if shield_on else 0.0
            if cur >= shield_min or hold >= max_hold:
                cash_recv = qty * price * (1 - FEE)
                pnl_usd = cash_recv - capital + (capital - qty * entry)  # actually proper $ pnl
                pnl_usd = qty * (price - entry) - (FEE * qty * price) - (FEE * qty * entry)
                trades.append({
                    'entry_dt': entry_dt, 'exit_dt': dt,
                    'entry_px': entry, 'exit_px': price,
                    'pnl_pct': cur, 'pnl_usd_in_capital': pnl_usd,
                    'hold_h': hold, 'regime': entry_regime,
                    'capital_at_entry': qty * entry,
                })
                cash = cash_recv
                in_pos = False; qty = 0.0; entry = 0.0; hold = 0
        if in_pos: hold += 1
        if cd > 0: cd -= 1
    if in_pos:
        cash_recv = qty * sigs[-1]['close'] * (1 - FEE)
        cur = (sigs[-1]['close'] / entry - 1.0) * 100.0
        pnl_usd = qty * (sigs[-1]['close'] - entry) - (FEE * qty * sigs[-1]['close']) - (FEE * qty * entry)
        trades.append({
            'entry_dt': entry_dt, 'exit_dt': sigs[-1]['datetime'],
            'entry_px': entry, 'exit_px': sigs[-1]['close'],
            'pnl_pct': cur, 'pnl_usd_in_capital': pnl_usd,
            'hold_h': hold, 'regime': entry_regime,
            'capital_at_entry': qty * entry,
        })
        cash = cash_recv

    eq_df = pd.DataFrame(eq_curve)
    return {
        'final_equity': cash,
        'pnl_pct': (cash / capital - 1) * 100,
        'pnl_usd': cash - capital,
        'trades': trades,
        'skipped': skipped,
        'eq_curve': eq_df,
    }


def main():
    print("Loading...")
    sigs = load_signals()
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    current_prod_gate = {'h_short': 20, 'h_long': 24,
                         't_short_pct': 4.0, 't_long_pct': 4.5, 'cd_hours': 12}
    mix_gate = {'h_short': 12, 'h_long': 20,
                't_short_pct': 2.5, 't_long_pct': 4.0, 'cd_hours': 24}

    # === COMPONENT DECOMPOSITION ===
    print("\n" + "="*100)
    print("  COMPONENT DECOMPOSITION (incremental contribution to MIX gate's $9,189)")
    print("="*100)

    setups = [
        ('1. Pure model only (NO shields, NO gate)', None, False, False),
        ('2. + Shields ON (no gate)', None, True, True),
        ('3. + Current PROD gate', current_prod_gate, True, True),
        ('4. + MIX gate (replace PROD gate)', mix_gate, True, True),
    ]

    prev_eq = START_CAPITAL
    print(f"\n  {'Step':<48} {'Final $':>11} {'Step PnL $':>12} {'Cumul %':>8}")
    print("  " + "-"*90)
    for label, gate, bs, brs in setups:
        r = simulate(sigs, asset_cfg, gate=gate,
                     bull_shield_override=bs, bear_shield_override=brs)
        delta = r['final_equity'] - prev_eq
        print(f"  {label:<48} ${r['final_equity']:>10,.0f} ${delta:>+10,.0f} {r['pnl_pct']:>+7.2f}%")
        prev_eq = r['final_equity']

    # === FULL DECOMPOSITION OF MIX TRADES ===
    print("\n" + "="*100)
    print("  TRADE-LEVEL ANALYSIS (MIX gate, 49 trades)")
    print("="*100)
    r_mix = simulate(sigs, asset_cfg, gate=mix_gate,
                      bull_shield_override=True, bear_shield_override=True,
                      log_skipped=True)
    df = pd.DataFrame(r_mix['trades'])
    df['exit_dt'] = pd.to_datetime(df['exit_dt'])
    df['month'] = df['exit_dt'].dt.tz_convert(None).dt.to_period('M')

    # Trade size distribution
    print("\n  Trade size distribution (by % PnL):")
    bins = [-100, -2, -0.5, 0, 0.5, 1, 2, 3, 5, 100]
    labels = ['<-2%', '-2..-0.5%', '-0.5..0%', '0..0.5%', '0.5..1%', '1..2%', '2..3%', '3..5%', '>=5%']
    df['pct_bucket'] = pd.cut(df['pnl_pct'], bins, labels=labels)
    bucket = df.groupby('pct_bucket', observed=False).agg(
        count=('pnl_pct', 'count'),
        sum_pnl_pct=('pnl_pct', 'sum'),
    ).round(2)
    print(bucket.to_string())

    # Per-month $ contribution
    print("\n  PnL contribution by month (each trade's $ PnL on capital at entry):")
    monthly = df.groupby('month', observed=False).agg(
        n_trades=('pnl_pct', 'count'),
        winners=('pnl_pct', lambda s: (s > 0).sum()),
        wr=('pnl_pct', lambda s: (s > 0).mean() * 100),
        avg_pnl_pct=('pnl_pct', 'mean'),
        sum_pnl_pct=('pnl_pct', 'sum'),
        sum_pnl_usd=('pnl_usd_in_capital', 'sum'),
    ).round(2)
    print(monthly.to_string())

    # Top 10 winners and losers
    print("\n  Top 10 BIGGEST WINNERS:")
    top_wins = df.sort_values('pnl_pct', ascending=False).head(10)
    print(top_wins[['entry_dt', 'exit_dt', 'pnl_pct', 'hold_h', 'regime',
                     'capital_at_entry', 'pnl_usd_in_capital']].round(2).to_string(index=False))

    print("\n  Top 10 BIGGEST LOSERS:")
    top_loss = df.sort_values('pnl_pct').head(10)
    print(top_loss[['entry_dt', 'exit_dt', 'pnl_pct', 'hold_h', 'regime',
                     'capital_at_entry', 'pnl_usd_in_capital']].round(2).to_string(index=False))

    # Where does the alpha really come from
    print("\n  Concentration analysis:")
    sorted_df = df.sort_values('pnl_usd_in_capital', ascending=False)
    top5_usd = sorted_df.head(5)['pnl_usd_in_capital'].sum()
    top10_usd = sorted_df.head(10)['pnl_usd_in_capital'].sum()
    bot5_usd = sorted_df.tail(5)['pnl_usd_in_capital'].sum()
    total_usd = df['pnl_usd_in_capital'].sum()
    print(f"  Top 5 trades contributed ${top5_usd:>8,.0f} ({top5_usd/total_usd*100:.0f}% of total $ PnL)")
    print(f"  Top 10 trades contributed ${top10_usd:>8,.0f} ({top10_usd/total_usd*100:.0f}% of total $ PnL)")
    print(f"  Bottom 5 trades cost     ${bot5_usd:>8,.0f}")
    print(f"  Total $ PnL:             ${total_usd:>8,.0f}")

    # Bull vs bear regime
    print("\n  Bull vs Bear regime breakdown:")
    by_regime = df.groupby('regime', observed=False).agg(
        n=('pnl_pct', 'count'),
        wr=('pnl_pct', lambda s: (s > 0).mean() * 100),
        avg_pnl=('pnl_pct', 'mean'),
        sum_pnl_pct=('pnl_pct', 'sum'),
        sum_pnl_usd=('pnl_usd_in_capital', 'sum'),
    ).round(2)
    print(by_regime.to_string())

    # Skipped entries — were they bad entries (justified) or good entries (cost us)?
    print("\n  SKIPPED ENTRIES analysis (gate refused 187 entries):")
    print(f"  Total skipped: {len(r_mix['skipped'])}")
    if r_mix['skipped']:
        # Compare what would have happened: take the entry, hold for max_hold or until next SELL
        # Approximate: forward 6h return after each skipped entry
        sk_df = pd.DataFrame(r_mix['skipped'])
        sk_df['dt'] = pd.to_datetime(sk_df['dt'])
        sk_df['month'] = sk_df['dt'].dt.tz_convert(None).dt.to_period('M')
        # Compute forward 6h return at each skipped point
        sigs_idx = {s['datetime']: i for i, s in enumerate(sigs)}
        sk_df['fwd_6h_pct'] = sk_df.apply(
            lambda r: ((sigs[min(sigs_idx[r['dt']]+6, len(sigs)-1)]['close']
                        / sigs[sigs_idx[r['dt']]]['close']) - 1) * 100, axis=1)
        sk_df['fwd_12h_pct'] = sk_df.apply(
            lambda r: ((sigs[min(sigs_idx[r['dt']]+12, len(sigs)-1)]['close']
                        / sigs[sigs_idx[r['dt']]]['close']) - 1) * 100, axis=1)
        sk_df['fwd_24h_pct'] = sk_df.apply(
            lambda r: ((sigs[min(sigs_idx[r['dt']]+24, len(sigs)-1)]['close']
                        / sigs[sigs_idx[r['dt']]]['close']) - 1) * 100, axis=1)
        print(f"  Avg forward 6h return at skipped points:  {sk_df['fwd_6h_pct'].mean():+.2f}%")
        print(f"  Avg forward 12h return at skipped points: {sk_df['fwd_12h_pct'].mean():+.2f}%")
        print(f"  Avg forward 24h return at skipped points: {sk_df['fwd_24h_pct'].mean():+.2f}%")
        print(f"  Median forward 6h at skipped:  {sk_df['fwd_6h_pct'].median():+.2f}%")
        print(f"  Median forward 12h at skipped: {sk_df['fwd_12h_pct'].median():+.2f}%")
        print(f"  Median forward 24h at skipped: {sk_df['fwd_24h_pct'].median():+.2f}%")
        print(f"  Skipped where forward 24h was POSITIVE: {(sk_df['fwd_24h_pct'] > 0).sum()}/{len(sk_df)}")
        print(f"  Skipped where forward 24h was NEGATIVE: {(sk_df['fwd_24h_pct'] < 0).sum()}/{len(sk_df)}")
        print(f"  Skipped where forward 24h was < -1%: {(sk_df['fwd_24h_pct'] < -1).sum()}/{len(sk_df)}")
        print(f"  Skipped where forward 24h was > +1%: {(sk_df['fwd_24h_pct'] > 1).sum()}/{len(sk_df)}")


if __name__ == '__main__':
    main()
