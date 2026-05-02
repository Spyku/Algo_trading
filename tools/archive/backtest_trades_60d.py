"""
backtest_trades_60d.py — emit per-trade detail of the 60d backtest using the
current production config. Reads the regime_config_ed.json (live) and the
signal cache (eth_sl_signals_90d.pkl). Same simulation logic as
report_production.py but logs every trade with: entry/exit time + price,
PnL%, hold hours, regime, exit reason.
"""
import os
import sys
import pickle
import json

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)

CACHE = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
REGIME_CFG = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
PROD_CSV = os.path.join(ENGINE, 'models', 'crypto_ed_production.csv')
FEE = 0.0005


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


def sim_per_trade(sigs, asset_cfg, fee=FEE):
    bull_shield = bool(asset_cfg.get('bull', {}).get('hold_shield', True))
    bear_shield = bool(asset_cfg.get('bear', {}).get('hold_shield', True))
    bull_conf = float(asset_cfg.get('bull', {}).get('min_confidence', 95))
    bear_conf = float(asset_cfg.get('bear', {}).get('min_confidence', 80))
    min_pnl = float(asset_cfg.get('min_sell_pnl_pct', 0.6))
    max_hold = int(asset_cfg.get('max_hold_hours', 12))
    brake_pct = float(asset_cfg.get('disaster_brake_pct', 0))

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

    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry_px = 0.0
    entry_dt = None
    entry_regime = None
    entry_idx = 0
    hold = 0
    cd = 0
    trades = []
    blocked_buys = 0
    blocked_sells = 0

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

        # Disaster brake
        force_sell = False
        if in_pos and brake_pct > 0 and entry_px > 0:
            cur_pnl = (price / entry_px - 1.0) * 100.0
            if cur_pnl <= -brake_pct:
                force_sell = True

        if force_sell and in_pos:
            cur_pnl = (price / entry_px - 1.0) * 100.0
            cash = qty * price * (1 - fee)
            trades.append({
                'entry_dt': entry_dt, 'exit_dt': s['datetime'],
                'entry_px': entry_px, 'exit_px': price,
                'pnl_pct': cur_pnl, 'hold_h': hold,
                'regime': entry_regime, 'exit_reason': 'BRAKE',
                'entry_conf': float(sigs[entry_idx].get('confidence', 0)),
                'exit_conf': sconf,
            })
            qty = 0
            in_pos = False
            entry_px = 0
            hold = 0
        elif sig == 'BUY' and sconf >= conf_thr and not in_pos:
            if cd <= 0:
                qty = cash * (1 - fee) / price
                cash = 0
                in_pos = True
                entry_px = price
                entry_dt = s['datetime']
                entry_regime = regime
                entry_idx = i
                hold = 0
            else:
                blocked_buys += 1
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1.0) * 100.0
            blocked = shield_on and cur_pnl < min_pnl and hold < max_hold
            if not blocked:
                cash = qty * price * (1 - fee)
                exit_reason = 'MODEL'
                if shield_on and hold >= max_hold and cur_pnl < min_pnl:
                    exit_reason = 'MAX_HOLD'
                trades.append({
                    'entry_dt': entry_dt, 'exit_dt': s['datetime'],
                    'entry_px': entry_px, 'exit_px': price,
                    'pnl_pct': cur_pnl, 'hold_h': hold,
                    'regime': entry_regime, 'exit_reason': exit_reason,
                    'entry_conf': float(sigs[entry_idx].get('confidence', 0)),
                    'exit_conf': sconf,
                })
                qty = 0
                in_pos = False
                entry_px = 0
                hold = 0
            else:
                blocked_sells += 1

        if in_pos:
            hold += 1
        if cd > 0:
            cd -= 1

    if in_pos:
        cash = qty * sigs[-1]['close'] * (1 - fee)
        trades.append({
            'entry_dt': entry_dt, 'exit_dt': sigs[-1]['datetime'],
            'entry_px': entry_px, 'exit_px': sigs[-1]['close'],
            'pnl_pct': (sigs[-1]['close'] / entry_px - 1.0) * 100.0,
            'hold_h': hold, 'regime': entry_regime,
            'exit_reason': 'OPEN_AT_END',
            'entry_conf': float(sigs[entry_idx].get('confidence', 0)),
            'exit_conf': float(sigs[-1].get('confidence', 0)),
        })

    return trades, cash, blocked_buys, blocked_sells


def main():
    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    sigs = load_sigs()
    print(f"Cache: {len(sigs)} signals, {sigs[0]['datetime']} -> {sigs[-1]['datetime']}")
    win = window_slice(sigs, 60)
    print(f"60d window: {len(win)} signals\n")

    print(f"Config: bull={asset_cfg['bull']['horizon']}h@{asset_cfg['bull']['min_confidence']}% "
          f"shield={'ON' if asset_cfg['bull'].get('hold_shield') else 'OFF'} | "
          f"bear={asset_cfg['bear']['horizon']}h@{asset_cfg['bear']['min_confidence']}% "
          f"shield={'ON' if asset_cfg['bear'].get('hold_shield') else 'OFF'}")
    print(f"Shared: min_sell_pnl={asset_cfg['min_sell_pnl_pct']}%, "
          f"max_hold={asset_cfg['max_hold_hours']}h, "
          f"brake={asset_cfg.get('disaster_brake_pct', 0)}%")
    print(f"Bull gate: {asset_cfg['bull'].get('rally_cooldown', 'NONE')}")
    print(f"Bear gate: {asset_cfg['bear'].get('rally_cooldown', 'NONE')}\n")

    trades, final_cash, blocked_b, blocked_s = sim_per_trade(win, asset_cfg)

    print("=" * 130)
    print(f"  PER-TRADE RESULTS (60d, fee={FEE*1e4:.1f} bps/leg)")
    print("=" * 130)
    print(f"\n{'#':<3}{'entry_time':<18}{'exit_time':<18}{'hold':>5}{'reg':>5}{'reason':>10}{'entry_$':>10}{'exit_$':>10}{'PnL%':>8}{'cum%':>9}")
    print('-' * 130)

    cum_mult = 1.0
    for i, t in enumerate(trades, 1):
        cum_mult *= (1 + t['pnl_pct'] / 100) * (1 - FEE) * (1 - FEE)
        # better: account for fee already applied in sim — recompute purely
        pass

    cum_mult = 1.0
    cum_pct = 0.0
    for i, t in enumerate(trades, 1):
        # gross trade PnL with fee
        net = (1 + t['pnl_pct'] / 100) * (1 - FEE) ** 2 - 1
        cum_mult *= (1 + net)
        cum_pct = (cum_mult - 1) * 100
        ed = t['entry_dt'].strftime('%m-%d %H:%M')
        xd = t['exit_dt'].strftime('%m-%d %H:%M')
        flag = '+' if t['pnl_pct'] > 0 else ('-' if t['pnl_pct'] < 0 else '0')
        print(f"{i:<3}{ed:<18}{xd:<18}{t['hold_h']:>4}h{t['regime'][:4]:>5}"
              f"{t['exit_reason']:>10}{t['entry_px']:>10.2f}{t['exit_px']:>10.2f}"
              f"{t['pnl_pct']:>+7.2f}%{cum_pct:>+8.2f}%")

    print("=" * 130)

    # Summary stats
    n = len(trades)
    pnl_pcts = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p <= 0]
    wr = len(wins) / n * 100 if n else 0
    avg_w = np.mean(wins) if wins else 0
    avg_l = np.mean(losses) if losses else 0
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
    expectancy = np.mean(pnl_pcts)
    total_strat = (cum_mult - 1) * 100
    bh_pct = (win[-1]['close'] / win[0]['close'] - 1.0) * 100

    print(f"\nSUMMARY")
    print(f"  Trades:         {n}  (wins {len(wins)} / losses {len(losses)})")
    print(f"  Strategy ret:   {total_strat:+.2f}%")
    print(f"  Buy & hold:     {bh_pct:+.2f}%")
    print(f"  Alpha:          {total_strat - bh_pct:+.2f}pp")
    print(f"  Win rate:       {wr:.1f}%")
    print(f"  Avg win:        {avg_w:+.2f}%   Avg loss: {avg_l:+.2f}%")
    print(f"  Best:           {max(pnl_pcts):+.2f}%   Worst: {min(pnl_pcts):+.2f}%")
    print(f"  Expectancy/tr:  {expectancy:+.3f}%")
    print(f"  Profit factor:  {pf:.2f}")
    print(f"  Blocked BUYs (gate): {blocked_b}")
    print(f"  Blocked SELLs (shield): {blocked_s}")

    # By exit reason
    print(f"\nBY EXIT REASON")
    df = pd.DataFrame(trades)
    if not df.empty:
        grp = df.groupby('exit_reason').agg(
            n=('pnl_pct', 'count'),
            avg_pnl=('pnl_pct', 'mean'),
            sum_pnl=('pnl_pct', 'sum'),
            avg_hold=('hold_h', 'mean'),
        ).round(2)
        print(grp.to_string())

    print(f"\nBY REGIME")
    if not df.empty:
        grp = df.groupby('regime').agg(
            n=('pnl_pct', 'count'),
            wr=('pnl_pct', lambda x: (x > 0).mean() * 100),
            avg_pnl=('pnl_pct', 'mean'),
            sum_pnl=('pnl_pct', 'sum'),
            avg_hold=('hold_h', 'mean'),
        ).round(2)
        print(grp.to_string())


if __name__ == '__main__':
    main()
