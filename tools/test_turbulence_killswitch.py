"""test_turbulence_killswitch.py — Smoke test for Idea #4:
Turbulence Index as cross-asset risk-off kill-switch.

Kritzman & Li (2010, Financial Analysts Journal) "Skulls, Financial Turbulence,
and Risk Management". State Street uses for ~$3T AUM.

  turbulence(t) = (r(t) - mu)^T * Sigma^-1 * (r(t) - mu)

where r(t) is a vector of daily returns across uncorrelated risk factors,
mu and Sigma are computed on a rolling window. High = stressed market.

Risk-vector (avoiding ETH itself since it's the trading asset):
  BTC (from cross_asset.csv), VIX, DXY, SP500, GOLD, US10Y (from macro_daily.csv)

Test: replay 60d baseline trades; skip BUYs when turbulence z-score >= threshold
on that day. Sweep z-thresholds {1.0, 1.5, 2.0, 2.5, 3.0}.

NO production change. Read-only.

Usage:
  python tools/test_turbulence_killswitch.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ENGINE)
sys.path.insert(0, ENGINE)

CACHE_PKL = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
CROSS_CSV = os.path.join(ENGINE, 'data', 'macro_data', 'cross_asset.csv')
MACRO_CSV = os.path.join(ENGINE, 'data', 'macro_data', 'macro_daily.csv')
REGIME_CFG = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
ts_run = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_CSV = os.path.join(ENGINE, 'output', f"turbulence_killswitch_{ts_run}.csv")

REPLAY_DAYS = 60
FEE_PER_LEG = 0.0005

COV_WINDOW_DAYS = 252       # rolling 1-year covariance window
ZSCORE_WINDOW_DAYS = 252    # rolling 1-year window for z-scoring turbulence

ZSCORE_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]
RISK_VECTOR = ['BTC_USD', 'VIX', 'DXY', 'SP500', 'GOLD', 'US10Y']


def load_cfg():
    with open(REGIME_CFG) as f:
        cfg = json.load(f).get('ETH', {})
    return {
        'bull_thr':  float(cfg.get('bull', {}).get('min_confidence', 65)),
        'bear_thr':  float(cfg.get('bear', {}).get('min_confidence', 75)),
        'bull_shield': bool(cfg.get('bull', {}).get('hold_shield', True)),
        'bear_shield': bool(cfg.get('bear', {}).get('hold_shield', True)),
        'min_sell_pnl_pct': float(cfg.get('min_sell_pnl_pct', 0.5)),
        'max_hold_hours':   int(cfg.get('max_hold_hours', 10)),
        'bear_gate':        cfg.get('bear', {}).get('rally_cooldown', {}),
        'bull_gate':        cfg.get('bull', {}).get('rally_cooldown', {}),
    }


def load_signals():
    with open(CACHE_PKL, 'rb') as f:
        sigs = pickle.load(f)
    for s in sigs:
        s['datetime'] = pd.Timestamp(s['datetime'])
    sigs.sort(key=lambda s: s['datetime'])
    end = sigs[-1]['datetime']
    lo = end - pd.Timedelta(days=REPLAY_DAYS)
    return [s for s in sigs if s['datetime'] >= lo]


def build_turbulence_series() -> pd.Series:
    """Compute daily turbulence z-score from cross_asset + macro_daily."""
    cross = pd.read_csv(CROSS_CSV)
    cross['date'] = pd.to_datetime(cross['date']).dt.normalize()
    cross = cross.set_index('date').sort_index()

    macro = pd.read_csv(MACRO_CSV)
    macro['date'] = pd.to_datetime(macro['date']).dt.normalize()
    macro = macro.set_index('date').sort_index()

    merged = pd.DataFrame(index=sorted(set(cross.index) | set(macro.index)))
    for col in RISK_VECTOR:
        if col in cross.columns:
            merged[col] = cross[col].reindex(merged.index)
        elif col in macro.columns:
            merged[col] = macro[col].reindex(merged.index)
        else:
            print(f"  WARNING: {col} not found in either CSV")

    # Forward-fill across weekends (equities don't trade Sat/Sun, crypto does)
    merged = merged.ffill()
    merged = merged.dropna(how='all').sort_index()
    print(f"  Risk-vector DataFrame: {len(merged)} rows {merged.index.min()} -> {merged.index.max()}")
    print(f"  Coverage by column:")
    for c in RISK_VECTOR:
        if c in merged.columns:
            print(f"    {c:<10} valid={merged[c].notna().sum()}/{len(merged)}")

    # Daily log returns
    rets = np.log(merged).diff().dropna(how='all')
    rets = rets.dropna()  # any row with even one NaN -> drop (need full vector for Mahalanobis)
    print(f"  Daily-returns matrix after dropna: {len(rets)} rows {rets.index.min()} -> {rets.index.max()}")

    # Rolling Mahalanobis turbulence
    turb = pd.Series(index=rets.index, dtype=float)
    for i in range(COV_WINDOW_DAYS, len(rets)):
        window = rets.iloc[i - COV_WINDOW_DAYS : i]   # past data only (causal)
        mu = window.mean().values
        Sigma = window.cov().values
        try:
            inv_S = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            continue
        diff = rets.iloc[i].values - mu
        turb.iloc[i] = float(diff @ inv_S @ diff)

    turb = turb.dropna()
    # Z-score over rolling 252d
    rolling_mean = turb.rolling(ZSCORE_WINDOW_DAYS, min_periods=60).mean()
    rolling_std  = turb.rolling(ZSCORE_WINDOW_DAYS, min_periods=60).std()
    z = (turb - rolling_mean) / (rolling_std + 1e-10)
    z = z.dropna()
    print(f"  Turbulence z-score series: {len(z)} days {z.index.min()} -> {z.index.max()}")
    print(f"  Z-score stats: mean={z.mean():.3f} std={z.std():.3f} "
          f"p50={z.median():.3f} p90={z.quantile(0.9):.3f} p99={z.quantile(0.99):.3f} max={z.max():.3f}")
    return z


def turb_at_signal_time(z_series: pd.Series, signal_dt: pd.Timestamp) -> float:
    """Look up turbulence z at the signal day (most recent <= signal day)."""
    day = pd.Timestamp(signal_dt).normalize()
    try:
        sub = z_series.loc[:day]
        if len(sub) == 0:
            return float('nan')
        return float(sub.iloc[-1])
    except Exception:
        return float('nan')


def _gate_blocks_buy(gate_cfg, signal_dt, sig_history_idx, signals):
    if not gate_cfg or not gate_cfg.get('enabled', False):
        return False
    h_short = int(gate_cfg.get('h_short', 30))
    h_long  = int(gate_cfg.get('h_long', 36))
    t_short = float(gate_cfg.get('t_short_pct', 9.0))
    t_long  = float(gate_cfg.get('t_long_pct', 9.0))
    cd_h    = int(gate_cfg.get('cd_hours', 48))
    cutoff = signal_dt - pd.Timedelta(hours=cd_h)
    for j in range(sig_history_idx - 1, -1, -1):
        s = signals[j]
        bar_dt = s['datetime']
        if bar_dt < cutoff:
            return False
        bar_close = s['close']
        for k_back, thr in ((h_short, t_short), (h_long, t_long)):
            target_dt = bar_dt - pd.Timedelta(hours=k_back)
            ref = None
            for kk in range(j, -1, -1):
                if signals[kk]['datetime'] <= target_dt:
                    ref = signals[kk]['close']
                    break
            if ref is None:
                continue
            rr = (bar_close / ref - 1.0) * 100.0
            if rr >= thr:
                return True
    return False


def simulate(sigs, cfg, turb_z=None, z_threshold=None):
    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry_px = 0.0
    hold = 0
    trades = []
    skipped = 0
    skipped_at = []  # (datetime, z) tuples

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
            if turb_z is not None and z_threshold is not None:
                v = turb_at_signal_time(turb_z, s['datetime'])
                if not np.isnan(v) and v >= z_threshold:
                    skipped += 1
                    skipped_at.append((s['datetime'], v))
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
        trades.append({'pnl_pct': (sigs[-1]['close']/entry_px - 1.0) * 100.0, 'hold_hours': hold})
    return (cash / 1000.0 - 1.0) * 100.0, trades, skipped, skipped_at


def main():
    print("=" * 100)
    print(f"  ETH TURBULENCE KILL-SWITCH SMOKE TEST -- {REPLAY_DAYS}d, fee={FEE_PER_LEG*100:.2f}%/leg")
    print("=" * 100)

    cfg = load_cfg()
    print(f"  Live config: bull@{cfg['bull_thr']}% (shield={cfg['bull_shield']}), "
          f"bear@{cfg['bear_thr']}% (shield={cfg['bear_shield']})")
    print(f"  Risk-vector: {RISK_VECTOR} | cov_window={COV_WINDOW_DAYS}d | z_window={ZSCORE_WINDOW_DAYS}d")

    print(f"\n  Building turbulence series...")
    turb_z = build_turbulence_series()

    sigs = load_signals()
    print(f"\n  Hourly signals: {len(sigs)}  range {sigs[0]['datetime']} -> {sigs[-1]['datetime']}")

    # Diagnostic: turbulence z at each baseline BUY signal
    buy_signal_dts = [s['datetime'] for s in sigs
                      if s['signal'] == 'BUY' and s.get('confidence', 0) >=
                      (cfg['bull_thr'] if s['regime'] == 'bull' else cfg['bear_thr'])]
    z_at_buys = [turb_at_signal_time(turb_z, dt) for dt in buy_signal_dts]
    z_at_buys = [v for v in z_at_buys if not np.isnan(v)]
    if z_at_buys:
        arr = np.array(z_at_buys)
        print(f"\n  Turbulence z-score at BASELINE BUY signal times: n={len(arr)} "
              f"mean={arr.mean():.3f} p50={np.median(arr):.3f} "
              f"p90={np.percentile(arr,90):.3f} max={arr.max():.3f}")
        for thr in ZSCORE_THRESHOLDS:
            n_above = int((arr >= thr).sum())
            print(f"    >= {thr}: {n_above} BUYs ({n_above/len(arr)*100:.0f}%)")

    print(f"\n  --- BASELINE simulation (no turbulence gate) ---")
    base_ret, base_trades, _, _ = simulate(sigs, cfg)
    base_wr = (sum(1 for t in base_trades if t['pnl_pct'] > 0) / len(base_trades) * 100) if base_trades else 0
    base_avg = np.mean([t['pnl_pct'] for t in base_trades]) if base_trades else 0
    print(f"  Baseline: return={base_ret:+.2f}%  trades={len(base_trades)}  WR={base_wr:.0f}%  avg_pnl={base_avg:+.2f}%")
    if not base_trades:
        return

    print(f"\n  --- TURBULENCE KILL-SWITCH sweep ({len(ZSCORE_THRESHOLDS)} variants) ---")
    print(f"  {'Variant':<28}{'Ret%':>9}{'d_vs_base':>11}{'Trades':>8}"
          f"{'WR%':>6}{'AvgPnL':>9}{'Skipped':>9}")
    print(f"  {'-'*28}{'-'*9}{'-'*11}{'-'*8}{'-'*6}{'-'*9}{'-'*9}")

    rows = []
    for thr in ZSCORE_THRESHOLDS:
        ret, trades, skipped, skipped_at = simulate(sigs, cfg, turb_z, thr)
        wr = (sum(1 for t in trades if t['pnl_pct'] > 0) / len(trades) * 100) if trades else 0
        avg = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        delta = ret - base_ret
        name = f"turb_z>={thr}"
        print(f"  {name:<28}{ret:>+8.2f}%{delta:>+10.2f}{len(trades):>8}"
              f"{wr:>5.0f}%{avg:>+8.2f}%{skipped:>9}")
        rows.append({'variant': name, 'z_threshold': thr,
                     'return_pct': round(ret, 4), 'delta_vs_base': round(delta, 4),
                     'trades': len(trades), 'win_rate': round(wr, 2),
                     'avg_pnl': round(avg, 4), 'skipped_buys': skipped})

    rows.append({'variant': 'BASELINE', 'z_threshold': None,
                 'return_pct': round(base_ret, 4), 'delta_vs_base': 0.0,
                 'trades': len(base_trades), 'win_rate': round(base_wr, 2),
                 'avg_pnl': round(base_avg, 4), 'skipped_buys': 0})
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n  Summary CSV: {OUT_CSV}")

    print("\n  --- TOP 5 by delta_vs_base ---")
    top = df_out[df_out['variant'] != 'BASELINE'].sort_values('delta_vs_base', ascending=False).head(5)
    for _, r in top.iterrows():
        print(f"    {r['variant']:<28}  ret={r['return_pct']:+.2f}%  delta={r['delta_vs_base']:+.2f}pp  "
              f"trades={r['trades']}  WR={r['win_rate']:.0f}%  skipped={r['skipped_buys']}")
    print(f"\n    BASELINE                    ret={base_ret:+.2f}%  trades={len(base_trades)}  WR={base_wr:.0f}%")

    print("\n" + "=" * 100)
    print("  VERDICT GUIDE:")
    print("  - >=1 variant with delta >= +5pp AND skipped > 0 -> kill-switch helps. Move to feature integration test.")
    print("  - All variants with skipped > 0 in (-3pp, +3pp) -> noise. Shelve.")
    print("  - All variants HURT -> turbulence not informative for ETH at this window. Shelve.")
    print("  - skipped=0 across all -> turbulence never triggered (no stress days in 60d window). Inconclusive.")
    print("=" * 100)


if __name__ == '__main__':
    main()
