"""test_vpin_filter.py — Smoke test for Idea #3: VPIN entry filter.

Easley, Lopez de Prado, O'Hara (2012, RFS) "Flow Toxicity and Liquidity in a
High-Frequency World". Detects toxic order flow that precedes adverse moves.

VPIN proxy used here (Binance 1m klines have taker_buy_base_volume):
  per_minute_imbalance = |2 * taker_buy_vol - total_vol| / total_vol
                       = |buy_vol - sell_vol| / total_vol  (with buy_vol = taker_buy)
  vpin_Nh = volume-weighted average of per_minute_imbalance over last N minutes

Test: replay current 60d baseline trades, skip BUYs when VPIN >= threshold at
signal time. Sweep VPIN window and threshold.

NO production change. Read-only on engine. Auto-downloads 1m data with
taker_buy_base_volume if missing.

Usage:
  python tools/test_vpin_filter.py
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

CACHE_PKL    = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
ONE_MIN_CSV  = os.path.join(ENGINE, 'data', 'eth_1m_data.csv')
REGIME_CFG   = os.path.join(ENGINE, 'config', 'regime_config_ed.json')
ts_run = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_CSV = os.path.join(ENGINE, 'output', f"vpin_filter_{ts_run}.csv")

REPLAY_DAYS = 60
FEE_PER_LEG = 0.0005

# VPIN sweep grid
VPIN_LOOKBACK_MIN = [30, 60, 120]
VPIN_THRESHOLDS   = [0.30, 0.40, 0.50, 0.60, 0.70]
# 15 variants


def ensure_1m_data_with_taker_buy():
    """If eth_1m_data.csv is missing or doesn't have taker_buy_base_volume,
    re-download via the existing downloader (which now keeps that field)."""
    needs_download = False
    if not os.path.exists(ONE_MIN_CSV):
        needs_download = True
        print(f"  1m CSV missing — downloading...")
    else:
        df = pd.read_csv(ONE_MIN_CSV, nrows=1)
        if 'taker_buy_base_volume' not in df.columns:
            needs_download = True
            print(f"  1m CSV exists but missing taker_buy_base_volume column — re-downloading...")
    if needs_download:
        from tools.download_1m_data import fetch_klines_1m, SYMBOL_MAP
        df = fetch_klines_1m(SYMBOL_MAP['ETH'], REPLAY_DAYS)
        df.to_csv(ONE_MIN_CSV, index=False)
        print(f"  Saved {len(df):,} candles -> {ONE_MIN_CSV}")
    else:
        print(f"  1m CSV present with taker_buy_base_volume column.")


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


def load_1m():
    df = pd.read_csv(ONE_MIN_CSV)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    return df


def compute_vpin_series(df_1m: pd.DataFrame, lookback_min: int) -> pd.Series:
    """For each minute, compute the VPIN proxy = volume-weighted avg of
    |2*taker_buy - total_vol| / total_vol over the previous `lookback_min`
    minutes (excluding the current minute, so it's strictly causal).
    """
    vol = df_1m['volume'].astype(float).values
    tb  = df_1m['taker_buy_base_volume'].astype(float).values
    abs_imb = np.abs(2 * tb - vol)  # = |buy - sell|
    # VPIN per minute = volume-weighted avg of (abs_imb_per_minute / vol_per_minute)
    # Numerically: rolling sum of abs_imb / rolling sum of vol over lookback_min.
    s_abs = pd.Series(abs_imb, index=df_1m.index).rolling(lookback_min, min_periods=lookback_min).sum()
    s_vol = pd.Series(vol,     index=df_1m.index).rolling(lookback_min, min_periods=lookback_min).sum()
    vpin = s_abs / s_vol.replace(0, np.nan)
    return vpin


def vpin_at_signal_time(vpin_series: pd.Series, signal_dt: pd.Timestamp) -> float:
    """VPIN value at the minute closest to (and at or before) signal_dt."""
    try:
        sub = vpin_series.loc[:signal_dt]
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


def simulate(sigs, cfg, vpin_series=None, vpin_threshold=None):
    """Replicate baseline strategy. If vpin_series + threshold given, skip BUY
    when VPIN at signal time >= threshold."""
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
            # VPIN gate
            if vpin_series is not None and vpin_threshold is not None:
                v = vpin_at_signal_time(vpin_series, s['datetime'])
                if not np.isnan(v) and v >= vpin_threshold:
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
        trades.append({'pnl_pct': (sigs[-1]['close']/entry_px - 1.0) * 100.0, 'hold_hours': hold})
    return (cash / 1000.0 - 1.0) * 100.0, trades, skipped


def main():
    print("=" * 100)
    print(f"  ETH VPIN ENTRY FILTER SMOKE TEST -- {REPLAY_DAYS}d, fee={FEE_PER_LEG*100:.2f}%/leg")
    print("=" * 100)

    print("\n  Ensuring 1m data has taker_buy_base_volume...")
    ensure_1m_data_with_taker_buy()

    cfg = load_cfg()
    print(f"\n  Live config: bull@{cfg['bull_thr']}% (shield={cfg['bull_shield']}), "
          f"bear@{cfg['bear_thr']}% (shield={cfg['bear_shield']}), "
          f"shield params {cfg['min_sell_pnl_pct']}%/{cfg['max_hold_hours']}h")
    print(f"               bear_gate.enabled={cfg['bear_gate'].get('enabled', False)}")

    sigs = load_signals()
    df_1m = load_1m()
    overlap_start = max(sigs[0]['datetime'], df_1m.index.min())
    overlap_end   = min(sigs[-1]['datetime'], df_1m.index.max())
    sigs = [s for s in sigs if overlap_start <= s['datetime'] <= overlap_end]
    df_1m = df_1m.loc[overlap_start:overlap_end]
    print(f"\n  Window: {overlap_start} -> {overlap_end}")
    print(f"  Hourly signals: {len(sigs)}  |  1m candles: {len(df_1m):,}")

    print(f"\n  --- BASELINE simulation (no VPIN gate) ---")
    base_ret, base_trades, _ = simulate(sigs, cfg)
    base_wr = (sum(1 for t in base_trades if t['pnl_pct'] > 0) / len(base_trades) * 100) if base_trades else 0
    base_avg = np.mean([t['pnl_pct'] for t in base_trades]) if base_trades else 0
    print(f"  Baseline: return={base_ret:+.2f}%  trades={len(base_trades)}  WR={base_wr:.0f}%  avg_pnl={base_avg:+.2f}%")
    if not base_trades:
        return

    # Pre-compute VPIN series for each lookback window
    print(f"\n  Computing VPIN series for lookbacks {VPIN_LOOKBACK_MIN}min...")
    vpin_by_lb = {}
    for lb in VPIN_LOOKBACK_MIN:
        v = compute_vpin_series(df_1m, lb)
        valid = v.dropna()
        print(f"    lb={lb:3d}m: valid {len(valid):,}/{len(v):,} "
              f"  mean={valid.mean():.3f}  p50={valid.median():.3f}  "
              f"p90={valid.quantile(0.90):.3f}  p99={valid.quantile(0.99):.3f}")
        vpin_by_lb[lb] = v

    # Diagnostic: VPIN at each baseline BUY signal time
    print(f"\n  VPIN distribution at BASELINE BUY entry times (per lookback):")
    base_buy_dts = [pd.Timestamp(t.get('entry_dt') or sigs[0]['datetime']) for t in base_trades]
    # entry_dt isn't in our trade dict; recompute from simulation
    # Easier: just describe vpin distribution at each BUY in raw signals
    buy_signal_dts = [s['datetime'] for s in sigs
                      if s['signal'] == 'BUY' and s.get('confidence', 0) >=
                      (cfg['bull_thr'] if s['regime'] == 'bull' else cfg['bear_thr'])]
    for lb in VPIN_LOOKBACK_MIN:
        vals = [vpin_at_signal_time(vpin_by_lb[lb], dt) for dt in buy_signal_dts]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            arr = np.array(vals)
            print(f"    lb={lb:3d}m at BUY signals: n={len(arr)} mean={arr.mean():.3f} "
                  f"p50={np.median(arr):.3f} p90={np.percentile(arr,90):.3f} max={arr.max():.3f}")

    print(f"\n  --- VPIN ENTRY FILTER sweep ({len(VPIN_LOOKBACK_MIN)*len(VPIN_THRESHOLDS)} variants) ---")
    print(f"  {'Variant':<28}{'Ret%':>9}{'d_vs_base':>11}{'Trades':>8}"
          f"{'WR%':>6}{'AvgPnL':>9}{'Skipped':>9}")
    print(f"  {'-'*28}{'-'*9}{'-'*11}{'-'*8}{'-'*6}{'-'*9}{'-'*9}")

    rows = []
    for lb in VPIN_LOOKBACK_MIN:
        for thr in VPIN_THRESHOLDS:
            ret, trades, skipped = simulate(sigs, cfg, vpin_by_lb[lb], thr)
            wr = (sum(1 for t in trades if t['pnl_pct'] > 0) / len(trades) * 100) if trades else 0
            avg = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
            delta = ret - base_ret
            name = f"vpin_lb={lb}m_thr={thr}"
            print(f"  {name:<28}{ret:>+8.2f}%{delta:>+10.2f}{len(trades):>8}"
                  f"{wr:>5.0f}%{avg:>+8.2f}%{skipped:>9}")
            rows.append({'variant': name, 'lookback_min': lb, 'threshold': thr,
                         'return_pct': round(ret, 4), 'delta_vs_base': round(delta, 4),
                         'trades': len(trades), 'win_rate': round(wr, 2),
                         'avg_pnl': round(avg, 4), 'skipped_buys': skipped})

    rows.append({'variant': 'BASELINE', 'lookback_min': None, 'threshold': None,
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
    print("  - >=2 variants with delta >= +5pp AND skipped > 0 -> VPIN gate helps. Promote candidate.")
    print("  - All variants with skipped > 0 in (-3pp, +3pp) -> noise. Shelve.")
    print("  - All variants HURT or skipped=0 -> shelve.")
    print("=" * 100)


if __name__ == '__main__':
    main()
