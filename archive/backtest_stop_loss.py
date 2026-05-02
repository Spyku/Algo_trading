"""Backtest: disaster-brake stop-loss levels on ETH prod trader.

Question: over the last 30 days, could a hard server-side stop-loss at
-2/-3/-4/-5/-7/-10% cap baseline losers without killing overall PnL?

Baseline  = current prod (hourly signal at top-of-hour + Hold Shield:
            sell only if PnL >= 0.5% or held >= 10h).
SL mode   = same entries; while invested, if any 5-min LOW inside the
            holding window <= entry * (1 - SL_pct/100), force-sell at the
            trigger price. Shield/failsafe only apply if SL doesn't fire.

Signals are cached to disk (pickle) so we only run model inference once.
Delete data/eth_sl_signals.pkl to force a refresh.

Run:  python backtest_stop_loss.py
"""
from __future__ import annotations

import os
import sys
import json
import pickle
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

# Reuse the existing backtest's plumbing
ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ENGINE_DIR)
sys.path.insert(0, ENGINE_DIR)

from backtest_intrahour_profit import (  # noqa: E402
    FIVE_MIN_CSV,
    DOOHAN_CSV,
    STANDARD_PROD_CSV,
    REGIME_CFG_PATH,
    TRADING_FEE,
    MAX_HOLD_HOURS,
    MIN_SELL_PNL_PCT,
    download_5m_candles,
    pick_model_row,
    generate_horizon_signals,
    merge_signals_via_regime,
    _five_min_slice,
)

ASSET            = 'ETH'
DAYS             = 30
REPLAY_HOURS     = DAYS * 24
SL_LEVELS_PCT    = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]      # positive numbers
SIG_CACHE_PATH   = os.path.join(ENGINE_DIR, 'data', 'eth_sl_signals.pkl')


# ----------------------------------------------------------------
# Signal cache
# ----------------------------------------------------------------
def load_or_build_signals() -> list[dict]:
    """Return merged hourly signal list, trimmed to last DAYS days.

    Cached so we don't re-run model inference every iteration.
    Cache is invalidated if it's older than 6h or doesn't cover 'now'.
    """
    if os.path.exists(SIG_CACHE_PATH):
        try:
            with open(SIG_CACHE_PATH, 'rb') as f:
                cached = pickle.load(f)
            if cached and isinstance(cached, list):
                last_dt = max(m['datetime'] for m in cached)
                age_h = (datetime.now(timezone.utc) - last_dt.to_pydatetime()).total_seconds() / 3600
                if age_h < 6:
                    print(f"  [sig] reusing cache: {SIG_CACHE_PATH} "
                          f"({len(cached)} rows, last={last_dt}, age={age_h:.1f}h)")
                    return cached
                print(f"  [sig] cache stale (age {age_h:.1f}h) — rebuilding")
        except Exception as e:
            print(f"  [sig] cache unusable ({e}); rebuilding")

    # Build
    with open(REGIME_CFG_PATH) as f:
        regime_cfg = json.load(f)
    asset_cfg = regime_cfg[ASSET]
    bull_h   = int(asset_cfg['bull']['horizon'])
    bear_h   = int(asset_cfg['bear']['horizon'])
    bull_thr = float(asset_cfg['bull']['min_confidence'])
    bear_thr = float(asset_cfg['bear']['min_confidence'])
    print(f"  regime: tsmom_672h | bull={bull_h}h@{bull_thr}% | bear={bear_h}h@{bear_thr}%")

    doohan = pd.read_csv(DOOHAN_CSV)
    standard = pd.read_csv(STANDARD_PROD_CSV)
    bull_row_source = doohan if bull_h in doohan[doohan['coin']==ASSET]['horizon'].values else standard
    bear_row_source = doohan if bear_h in doohan[doohan['coin']==ASSET]['horizon'].values else standard
    bull_row = pick_model_row(bull_row_source, ASSET, bull_h)
    bear_row = pick_model_row(bear_row_source, ASSET, bear_h)

    print(f"  Generating signals (replay={REPLAY_HOURS}h)...")
    bull_sigs = generate_horizon_signals(ASSET, bull_row, REPLAY_HOURS)
    bear_sigs = generate_horizon_signals(ASSET, bear_row, REPLAY_HOURS)
    merged = merge_signals_via_regime(ASSET, bull_sigs, bull_thr, bull_h,
                                      bear_sigs, bear_thr, bear_h)

    # Trim to last 30 days
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(days=DAYS)
    merged = [m for m in merged if m['datetime'] >= pd.Timestamp(cutoff)]

    # Make tz-aware for downstream 5m slicing
    for m in merged:
        if m['datetime'].tzinfo is None:
            m['datetime'] = m['datetime'].tz_localize('UTC')

    with open(SIG_CACHE_PATH, 'wb') as f:
        pickle.dump(merged, f)
    print(f"  [sig] cached {len(merged)} rows -> {SIG_CACHE_PATH}")
    return merged


# ----------------------------------------------------------------
# Simulation with optional stop-loss
# ----------------------------------------------------------------
def simulate_sl(signals: list, df5: pd.DataFrame, sl_pct: float | None) -> dict:
    """Simulate baseline (sl_pct=None) or with a hard SL at `sl_pct` percent.

    SL fills at entry*(1-sl_pct/100) the moment a 5m LOW breaches it.
    """
    if df5.index.name != 'datetime':
        df5 = df5.set_index('datetime')
    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry_px = 0.0
    entry_time = None
    hold_hours = 0
    trades = []
    sl_trigs = 0
    equity_curve = [1000.0]

    n = len(signals)
    for i, s in enumerate(signals):
        dt = s['datetime']
        price = s['close']
        sig = s['signal']
        conf = s['confidence']
        thr = s['conf_threshold']

        equity = cash + qty * price if in_pos else cash
        equity_curve.append(equity)

        # ----- INTRA-HOUR SL CHECK -----
        if sl_pct is not None and in_pos:
            trigger_px = entry_px * (1.0 - sl_pct / 100.0)
            bars = _five_min_slice(df5, dt)
            hit_row = None
            for ts, row in bars.iterrows():
                if row['low'] <= trigger_px:
                    hit_row = (ts, row)
                    break
            if hit_row is not None:
                ts, _row = hit_row
                exit_px = trigger_px   # assume stop triggers at level
                cash = qty * exit_px * (1 - TRADING_FEE)
                pnl_pct = (exit_px / entry_px - 1.0) * 100.0
                trades.append({
                    'entry_time': entry_time, 'entry_price': entry_px,
                    'exit_time': ts, 'exit_price': exit_px,
                    'pnl_pct': pnl_pct, 'hold_hours': hold_hours,
                    'exit_reason': f'sl_{sl_pct:.0f}',
                })
                sl_trigs += 1
                in_pos = False
                qty = 0.0
                entry_px = 0.0
                hold_hours = 0
                entry_time = None
                # fall through to hourly decision for potential re-entry

        # ----- HOURLY DECISION -----
        if sig == 'BUY' and conf >= thr and not in_pos:
            if i + 1 < n:
                next_sig = signals[i + 1]
                fill_dt = next_sig['datetime']
                fill_px = next_sig['close']
            else:
                fill_dt = dt
                fill_px = price
            qty = cash * (1 - TRADING_FEE) / fill_px
            cash = 0.0
            in_pos = True
            entry_px = fill_px
            entry_time = fill_dt
            hold_hours = 0

        elif sig == 'SELL' and in_pos:
            if i + 1 < n:
                next_sig = signals[i + 1]
                fill_dt = next_sig['datetime']
                fill_px = next_sig['close']
            else:
                fill_dt = dt
                fill_px = price
            cur_pnl_pct = (fill_px / entry_px - 1.0) * 100.0
            shield_ok = cur_pnl_pct >= (MIN_SELL_PNL_PCT * 100.0)
            failsafe  = hold_hours >= MAX_HOLD_HOURS
            if shield_ok or failsafe:
                cash = qty * fill_px * (1 - TRADING_FEE)
                trades.append({
                    'entry_time': entry_time, 'entry_price': entry_px,
                    'exit_time': fill_dt, 'exit_price': fill_px,
                    'pnl_pct': cur_pnl_pct, 'hold_hours': hold_hours + 1,
                    'exit_reason': 'failsafe' if (failsafe and not shield_ok) else 'hourly_shield_ok',
                })
                in_pos = False
                qty = 0.0
                entry_px = 0.0
                entry_time = None
                hold_hours = 0

        if in_pos:
            hold_hours += 1

    # close out open position
    if in_pos:
        final_px = signals[-1]['close']
        cash = qty * final_px * (1 - TRADING_FEE)
        pnl_pct = (final_px / entry_px - 1.0) * 100.0
        trades.append({
            'entry_time': entry_time, 'entry_price': entry_px,
            'exit_time': signals[-1]['datetime'], 'exit_price': final_px,
            'pnl_pct': pnl_pct, 'hold_hours': hold_hours,
            'exit_reason': 'end_of_window',
        })

    tot_pnl = (cash / 1000.0 - 1.0) * 100.0
    wins = sum(1 for t in trades if t['pnl_pct'] > 0)
    ec = np.array(equity_curve)
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / peak
    max_dd = float(dd.min()) * 100.0 if len(dd) else 0.0

    return {
        'sl_pct': sl_pct,
        'trades': trades,
        'n_trades': len(trades),
        'wins': wins,
        'total_pnl_pct': tot_pnl,
        'max_drawdown_pct': max_dd,
        'sl_triggers': sl_trigs,
    }


# ----------------------------------------------------------------
# False-alarm analysis
# ----------------------------------------------------------------
def false_alarm_count(base_trades: list, sl_result: dict) -> int:
    """How many SL triggers would NOT have needed to fire?

    A trigger is a "false alarm" if, in the baseline (no SL), the same
    entry (matched by entry_time) ultimately exited at a better PnL than
    the SL-capped -sl_pct%. I.e., the baseline recovered from the dip.
    """
    base_by_entry = {t['entry_time']: t for t in base_trades}
    false = 0
    sl_pct = sl_result['sl_pct']
    for t in sl_result['trades']:
        if not t['exit_reason'].startswith('sl_'):
            continue
        bt = base_by_entry.get(t['entry_time'])
        if bt is None:
            continue
        # SL locked in -sl_pct%. Baseline's actual pnl for this entry:
        if bt['pnl_pct'] > -sl_pct:
            false += 1
    return false


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    print("=" * 80)
    print(f"  Stop-loss backtest  ({ASSET}, {DAYS}d, levels={SL_LEVELS_PCT})")
    print("=" * 80)

    # 5m candles
    df5 = download_5m_candles(DAYS)
    df5['datetime'] = pd.to_datetime(df5['datetime'], utc=True)
    df5 = df5.set_index('datetime').sort_index()

    # Signals (cached)
    merged = load_or_build_signals()
    print(f"  replay window: {len(merged)} hourly signals "
          f"({merged[0]['datetime']} -> {merged[-1]['datetime']})")

    # Baseline
    base = simulate_sl(merged, df5.copy(), sl_pct=None)

    # Each SL level
    results = []
    for sl in SL_LEVELS_PCT:
        r = simulate_sl(merged, df5.copy(), sl_pct=sl)
        r['false_alarms'] = false_alarm_count(base['trades'], r)
        results.append(r)

    # -------- Report --------
    print()
    print("=" * 88)
    print(f"  ETH stop-loss backtest  ({DAYS}d)")
    print("=" * 88)
    hdr = f"  {'SL level':<12}{'Trades':>8}{'Wins':>7}{'Total PnL%':>13}{'Max DD%':>10}{'SL trigs':>11}{'False alarms':>15}{'vs Baseline':>14}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    base_pnl = base['total_pnl_pct']
    print(f"  {'None (base)':<12}{base['n_trades']:>8}{base['wins']:>7}"
          f"{base['total_pnl_pct']:>+13.2f}{base['max_drawdown_pct']:>+10.2f}"
          f"{'-':>11}{'-':>15}{'-':>14}")
    for r in results:
        delta = r['total_pnl_pct'] - base_pnl
        sign = '+' if delta >= 0 else ''
        print(f"  {'-' + str(int(r['sl_pct'])) + '%':<12}"
              f"{r['n_trades']:>8}{r['wins']:>7}"
              f"{r['total_pnl_pct']:>+13.2f}{r['max_drawdown_pct']:>+10.2f}"
              f"{r['sl_triggers']:>11}{r['false_alarms']:>15}"
              f"{(sign + f'{delta:.2f}'):>14}")

    # Trade log per level (only SL triggers and the matching baseline trade)
    print()
    print("=" * 88)
    print("  SL trigger detail (which entries triggered, baseline recovery?)")
    print("=" * 88)
    base_by_entry = {t['entry_time']: t for t in base['trades']}
    for r in results:
        sl = r['sl_pct']
        trigs = [t for t in r['trades'] if t['exit_reason'].startswith('sl_')]
        if not trigs:
            continue
        print(f"\n  -- SL -{int(sl)}% ({len(trigs)} triggers) --")
        for t in trigs:
            et = str(t['entry_time']).replace('+00:00', '')[:16]
            bt = base_by_entry.get(t['entry_time'])
            if bt:
                recovered = 'RECOVERED' if bt['pnl_pct'] > -sl else 'still_loss'
                print(f"    entry {et}  entry_px={t['entry_price']:.2f}  "
                      f"sl_pnl={-sl:+.2f}%  "
                      f"baseline_pnl={bt['pnl_pct']:+.2f}%  ({recovered})")
            else:
                print(f"    entry {et}  entry_px={t['entry_price']:.2f}  "
                      f"sl_pnl={-sl:+.2f}%  (no baseline match)")

    # Summary
    print()
    print("=" * 88)
    print("  Baseline trade losers (where SL could help)")
    print("=" * 88)
    losers = [t for t in base['trades'] if t['pnl_pct'] < 0]
    for t in losers:
        et = str(t['entry_time']).replace('+00:00', '')[:16]
        xt = str(t['exit_time']).replace('+00:00', '')[:16]
        print(f"    {et} -> {xt}  entry={t['entry_price']:.2f} exit={t['exit_price']:.2f}  "
              f"pnl={t['pnl_pct']:+.2f}% hold={t['hold_hours']}h reason={t['exit_reason']}")


if __name__ == '__main__':
    main()
