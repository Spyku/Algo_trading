"""Production ETH report — shows the exact live config + comprehensive backtest
stats on 30d and 60d windows.

Metrics:
  - Total return (strategy vs buy-and-hold)
  - Trade count, win rate
  - Avg win %, avg loss %, expectancy per trade
  - Max drawdown, best trade, worst trade
  - Profit factor (sum wins / sum losses)
  - Sharpe-like (mean / std of trade returns)
  - Time in market (%)
  - Model signal accuracy (SELL prediction correctness at horizon)
"""
import os
import sys
import pickle
import json
import math

import numpy as np
import pandas as pd

ENGINE = r'G:\Autres ordinateurs\My laptop\engine'
os.chdir(ENGINE)
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


def sim(sigs, asset_cfg):
    """Full strategy sim with per-regime shield + per-regime gate + brake + 5 bps fee."""
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

    cash = 1000.0; qty = 0.0; in_pos = False; entry_px = 0.0
    hold = 0; cd = 0
    trades = []  # list of pnl_pct
    brake_fires = 0
    bars_in_pos = 0
    equity = [1000.0]

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

        equity.append(cash + qty * price if in_pos else cash)

        # Disaster brake check (before model action)
        force_sell = False
        if in_pos and brake_pct > 0 and entry_px > 0:
            cur_pnl = (price / entry_px - 1.0) * 100.0
            if cur_pnl <= -brake_pct:
                force_sell = True
                brake_fires += 1

        if force_sell and in_pos:
            cur_pnl = (price / entry_px - 1.0) * 100.0
            cash = qty * price * (1 - FEE)
            trades.append(cur_pnl)
            qty = 0
            in_pos = False
            entry_px = 0
            hold = 0
        elif sig == 'BUY' and sconf >= conf_thr and not in_pos:
            if cd <= 0:
                qty = cash * (1 - FEE) / price
                cash = 0
                in_pos = True
                entry_px = price
                hold = 0
        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1.0) * 100.0
            blocked = shield_on and cur_pnl < min_pnl and hold < max_hold
            if not blocked:
                cash = qty * price * (1 - FEE)
                trades.append(cur_pnl)
                qty = 0
                in_pos = False
                entry_px = 0
                hold = 0

        if in_pos:
            hold += 1
            bars_in_pos += 1
        if cd > 0: cd -= 1

    if in_pos:
        cash = qty * sigs[-1]['close'] * (1 - FEE)
        trades.append((sigs[-1]['close'] / entry_px - 1.0) * 100.0)

    arr = np.array(equity)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    mdd = float(dd.min()) * 100.0 if len(dd) else 0.0

    ret = (cash / 1000.0 - 1.0) * 100.0
    n = len(trades)
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    wr = (len(wins) / n * 100) if n else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    best = max(trades, default=0)
    worst = min(trades, default=0)
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
    # Expectancy per trade
    exp_per_trade = (wr/100) * avg_win + (1 - wr/100) * avg_loss if n else 0
    # Trade-level "sharpe" = mean / std of trade returns (not annualized)
    sharpe_like = np.mean(trades) / np.std(trades) if n > 1 and np.std(trades) > 0 else 0
    time_in_market = bars_in_pos / len(sigs) * 100 if sigs else 0

    # Buy-and-hold
    bh_pct = (sigs[-1]['close'] / sigs[0]['close'] - 1.0) * 100.0

    return dict(
        return_pct=ret, buyhold_pct=bh_pct, alpha=ret - bh_pct,
        n_trades=n, win_rate=wr,
        avg_win=avg_win, avg_loss=avg_loss,
        best=best, worst=worst,
        profit_factor=pf, expectancy=exp_per_trade, sharpe_like=sharpe_like,
        max_dd=mdd, time_in_market=time_in_market,
        brake_fires=brake_fires,
    )


def model_accuracy(sigs, horizon_hours=6):
    """Strict: compute what fraction of actionable signals (BUY + SELL only,
    above bull/bear conf threshold) were correct at their horizon."""
    bull_conf = 95; bear_conf = 80
    correct = 0; total = 0
    n = len(sigs)
    for i, s in enumerate(sigs):
        regime = s.get('regime', 'bull')
        conf_thr = bull_conf if regime == 'bull' else bear_conf
        sconf = float(s.get('confidence', 0))
        if sconf < conf_thr:
            continue
        sig = s['signal']
        if sig not in ('BUY', 'SELL'):
            continue
        j = i + horizon_hours
        if j >= n:
            break
        price_now = float(s['close'])
        price_then = float(sigs[j]['close'])
        ret = (price_then / price_now - 1.0)
        if sig == 'BUY' and ret > 0: correct += 1
        elif sig == 'SELL' and ret <= 0: correct += 1
        total += 1
    return correct / total * 100 if total else 0, total


def main():
    print("=" * 100)
    print("  ETH PRODUCTION REPORT — current config + comprehensive backtest stats")
    print("=" * 100)

    with open(REGIME_CFG) as f:
        cfg = json.load(f)
    asset_cfg = cfg['ETH']

    # Config summary
    print(f"\n[CONFIG — live on disk]")
    print(f"  Regime detector: {asset_cfg['regime_detector']['type']}:"
          f"{asset_cfg['regime_detector'].get('params', {}).get('name', '?')}")
    print(f"  Bull: {asset_cfg['bull']['horizon']}h @ {asset_cfg['bull']['min_confidence']}% "
          f"| ${asset_cfg['bull'].get('max_position_usd'):,} | "
          f"shield={'ON' if asset_cfg['bull'].get('hold_shield') else 'OFF'}")
    print(f"  Bear: {asset_cfg['bear']['horizon']}h @ {asset_cfg['bear']['min_confidence']}% "
          f"| ${asset_cfg['bear'].get('max_position_usd'):,} | "
          f"shield={'ON' if asset_cfg['bear'].get('hold_shield') else 'OFF'}")
    print(f"  Shield thresholds: >={asset_cfg['min_sell_pnl_pct']}% PnL OR "
          f">={asset_cfg['max_hold_hours']}h held")
    bull_g = asset_cfg['bull'].get('rally_cooldown', None)
    bear_g = asset_cfg['bear'].get('rally_cooldown', None)
    asset_g = asset_cfg.get('rally_cooldown', None)
    print(f"  Bull gate: {bull_g if bull_g else ('asset-level ' + str(asset_g) if asset_g else 'NONE')}")
    print(f"  Bear gate: {bear_g if bear_g else ('asset-level ' + str(asset_g) if asset_g else 'NONE')}")
    print(f"  Disaster brake: {asset_cfg.get('disaster_brake_pct', 0)}% "
          f"({'active' if asset_cfg.get('disaster_brake_pct', 0) > 0 else 'OFF'})")
    print(f"  Quick-release: {asset_cfg.get('shield_quick_release', 'OFF (no config)')}")
    print(f"  Backtest fee: {FEE*1e4:.1f} bps/leg")

    # Model details from production CSV
    df = pd.read_csv(PROD_CSV)
    eth = df[df['coin'] == 'ETH'].sort_values('horizon')
    print(f"\n[MODELS in crypto_ed_production.csv]")
    print(f"  {'Horizon':<8}{'Combo':<12}{'Acc%':<8}{'Gamma':<8}{'Ret%':<8}{'Feats':<6}")
    for _, r in eth.iterrows():
        print(f"  {r['horizon']}h      {r['best_combo']:<12}{r['accuracy']:<7.1f}%"
              f"{r['gamma']:<8.4f}{r['return_pct']:<7.2f}%{r['n_features']}")

    # Signal-level accuracy
    sigs_all = load_sigs()
    acc_6h, n_acc_6h = model_accuracy(sigs_all, horizon_hours=6)
    acc_8h, n_acc_8h = model_accuracy(sigs_all, horizon_hours=8)
    print(f"\n[SIGNAL ACCURACY — 90d cache]")
    print(f"  High-conf signals correct at 6h horizon: {acc_6h:.1f}% (n={n_acc_6h})")
    print(f"  High-conf signals correct at 8h horizon: {acc_8h:.1f}% (n={n_acc_8h})")

    # Strategy stats per window
    for days in (30, 60):
        sigs = window_slice(sigs_all, days)
        r = sim(sigs, asset_cfg)
        print(f"\n[BACKTEST {days}d — {len(sigs)} bars, "
              f"{sigs[0]['datetime'].date()} -> {sigs[-1]['datetime'].date()}]")
        print(f"  Total return (strategy) : {r['return_pct']:+.2f}%")
        print(f"  Buy-and-hold ETH        : {r['buyhold_pct']:+.2f}%")
        print(f"  Alpha                   : {r['alpha']:+.2f}pp")
        print(f"  Trade count             : {r['n_trades']}")
        print(f"  Win rate                : {r['win_rate']:.1f}%")
        print(f"  Avg win per trade       : {r['avg_win']:+.2f}%")
        print(f"  Avg loss per trade      : {r['avg_loss']:+.2f}%")
        print(f"  Expectancy per trade    : {r['expectancy']:+.3f}%")
        print(f"  Profit factor           : {r['profit_factor']:.2f} "
              f"({'inf' if math.isinf(r['profit_factor']) else ''})")
        print(f"  Sharpe-like (trade)     : {r['sharpe_like']:.2f}")
        print(f"  Best trade              : {r['best']:+.2f}%")
        print(f"  Worst trade             : {r['worst']:+.2f}%")
        print(f"  Max drawdown (equity)   : {r['max_dd']:+.2f}%")
        print(f"  Time in market          : {r['time_in_market']:.1f}%")
        print(f"  Disaster brake fires    : {r['brake_fires']}")


if __name__ == '__main__':
    main()
