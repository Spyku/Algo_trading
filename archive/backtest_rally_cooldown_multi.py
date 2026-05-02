"""Rally-cooldown validation across 30d / 60d / 90d windows.

Requires caches produced by extend_caches_90d.py:
  data/eth_5m_backtest_90d.csv
  data/eth_sl_signals_90d.pkl

Runs V0..V6 variants on each window and writes per-window + stability CSVs.
"""
from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ENGINE_DIR, 'data')
SIG_90D  = os.path.join(DATA_DIR, 'eth_sl_signals_90d.pkl')
FIVE_90D = os.path.join(DATA_DIR, 'eth_5m_backtest_90d.csv')

TRADING_FEE = 0.0005  # 5 bps/leg realistic maker blend (see ed.py BACKTEST_FEE_PER_LEG)
MIN_SELL_PNL_PCT = 0.005
MAX_HOLD_HOURS = 10

VARIANTS = [
    dict(code='V0', name='Baseline', gate=None),
    dict(code='V1', name='rr24>=5% gate',          gate='gate', cond=('rr_24h', 5.0)),
    dict(code='V2', name='rr24>=5% cooldown 12h',  gate='cool', cond=('rr_24h', 5.0), hours=12),
    dict(code='V3', name='rr24>=7% cooldown 24h',  gate='cool', cond=('rr_24h', 7.0), hours=24),
    dict(code='V4', name='rr10>=3% cooldown 6h',   gate='cool', cond=('rr_10h', 3.0), hours=6),
    dict(code='V5', name='rr10>=5% cooldown 10h',  gate='cool', cond=('rr_10h', 5.0), hours=10),
    dict(code='V6', name='(rr10>=5 or rr24>=7) cooldown 24h',
         gate='combo', conds=[('rr_10h', 5.0), ('rr_24h', 7.0)], hours=24),
]


def load_caches():
    with open(SIG_90D, 'rb') as f:
        signals = pickle.load(f)
    # Normalize signal datetimes to tz-aware UTC
    for s in signals:
        t = pd.Timestamp(s['datetime'])
        s['datetime'] = t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    df5 = pd.read_csv(FIVE_90D)
    df5['datetime'] = pd.to_datetime(df5['datetime'], utc=True)
    df5 = df5.set_index('datetime').sort_index()
    return signals, df5


def build_hourly_frame(signals):
    rows = [{'datetime': s['datetime'], 'close': s['close'],
             'signal': s['signal'], 'confidence': s['confidence'],
             'conf_threshold': s['conf_threshold']} for s in signals]
    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    for h in (6, 10, 24, 48):
        df[f'rr_{h}h'] = (df['close'] / df['close'].shift(h) - 1.0) * 100.0
    return df


def trigger_hit(row, cfg):
    if cfg['gate'] in ('gate', 'cool'):
        col, thr = cfg['cond']
        return pd.notna(row[col]) and row[col] >= thr
    if cfg['gate'] == 'combo':
        for col, thr in cfg['conds']:
            if pd.notna(row[col]) and row[col] >= thr:
                return True
        return False
    return False


def simulate(signals, df_hourly, cfg):
    rr_map = df_hourly.set_index('datetime')
    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry_px = 0.0
    entry_time = None
    hold_hours = 0
    trades = []
    shield_fires = 0
    failsafe_fires = 0
    skipped_buys = 0
    cooldown_remaining = 0
    equity_curve = [1000.0]

    n = len(signals)
    for i, s in enumerate(signals):
        dt = s['datetime']
        price = s['close']
        sig = s['signal']
        conf = s['confidence']
        thr = s['conf_threshold']

        if dt in rr_map.index:
            rr_row = rr_map.loc[dt]
        else:
            rr_row = None

        if cfg['gate'] in ('cool', 'combo') and rr_row is not None:
            if trigger_hit(rr_row, cfg):
                cooldown_remaining = max(cooldown_remaining, cfg['hours'])

        equity = cash + qty * price if in_pos else cash
        equity_curve.append(equity)

        if sig == 'BUY' and conf >= thr and not in_pos:
            block = False
            if cfg['gate'] == 'gate' and rr_row is not None and trigger_hit(rr_row, cfg):
                block = True
            elif cfg['gate'] in ('cool', 'combo') and cooldown_remaining > 0:
                block = True
            if block:
                skipped_buys += 1
            else:
                if i + 1 < n:
                    fill_dt = signals[i + 1]['datetime']
                    fill_px = signals[i + 1]['close']
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
                fill_dt = signals[i + 1]['datetime']
                fill_px = signals[i + 1]['close']
            else:
                fill_dt = dt
                fill_px = price
            cur_pnl_pct = (fill_px / entry_px - 1.0) * 100.0
            shield_ok = cur_pnl_pct >= (MIN_SELL_PNL_PCT * 100.0)
            failsafe = hold_hours >= MAX_HOLD_HOURS
            if shield_ok or failsafe:
                cash = qty * fill_px * (1 - TRADING_FEE)
                reason = 'failsafe' if (failsafe and not shield_ok) else 'shield_release'
                if reason == 'failsafe':
                    failsafe_fires += 1
                else:
                    shield_fires += 1
                trades.append({
                    'entry_time': entry_time, 'entry_price': entry_px,
                    'exit_time': fill_dt, 'exit_price': fill_px,
                    'pnl_pct': cur_pnl_pct, 'hold_hours': hold_hours + 1,
                    'exit_reason': reason,
                })
                in_pos = False
                qty = 0.0
                entry_px = 0.0
                entry_time = None
                hold_hours = 0

        if in_pos:
            hold_hours += 1
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

    if in_pos:
        final_px = signals[-1]['close']
        cash = qty * final_px * (1 - TRADING_FEE)
        pnl_pct = (final_px / entry_px - 1.0) * 100.0
        trades.append({
            'entry_time': entry_time, 'entry_price': entry_px,
            'exit_time': signals[-1]['datetime'], 'exit_price': final_px,
            'pnl_pct': pnl_pct, 'hold_hours': hold_hours,
            'exit_reason': 'eow_flatten',
        })

    total_pnl = (cash / 1000.0 - 1.0) * 100.0
    wins = sum(1 for t in trades if t['pnl_pct'] > 0)
    ec = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / peak
    max_dd = float(dd.min()) * 100.0 if len(dd) else 0.0
    return dict(
        code=cfg['code'], name=cfg['name'],
        trades=len(trades), wins=wins,
        total_pnl_pct=total_pnl, max_dd_pct=max_dd,
        skipped_buys=skipped_buys,
    )


def run_window(signals, df_hourly, days, label):
    end_t = df_hourly['datetime'].iloc[-1]
    start_t = end_t - pd.Timedelta(days=days)
    def _t(x):
        t = pd.Timestamp(x)
        return t.tz_localize('UTC') if t.tzinfo is None else t
    win_sigs = [s for s in signals if _t(s['datetime']) >= start_t]
    win_df = df_hourly[df_hourly['datetime'] >= start_t].reset_index(drop=True)
    print(f"\n  === {label} ({len(win_sigs)} signals, "
          f"{win_df['datetime'].iloc[0]} -> {win_df['datetime'].iloc[-1]}) ===")
    results = []
    for cfg in VARIANTS:
        r = simulate(win_sigs, win_df, cfg)
        r['window'] = label
        results.append(r)
    base_pnl = results[0]['total_pnl_pct']
    print(f"  {'Variant':<40}{'Trades':>8}{'Wins':>6}{'PnL%':>10}"
          f"{'MaxDD%':>10}{'Skip':>8}{'vs V0':>10}")
    print("  " + "-" * 92)
    for r in results:
        delta = r['total_pnl_pct'] - base_pnl
        delta_s = '-' if r['code'] == 'V0' else f"{delta:+.2f}"
        print(f"  {r['code']+' '+r['name']:<40}"
              f"{r['trades']:>8}{r['wins']:>6}"
              f"{r['total_pnl_pct']:>+10.2f}{r['max_dd_pct']:>+10.2f}"
              f"{r['skipped_buys']:>8}{delta_s:>10}")
    return results


def main():
    print("=" * 92)
    print("  ETH Rally-Cooldown Stability Validation (30d / 60d / 90d)")
    print("=" * 92)
    signals, df5 = load_caches()
    df_hourly = build_hourly_frame(signals)
    print(f"  full: {len(signals)} signals "
          f"{df_hourly['datetime'].iloc[0]} -> {df_hourly['datetime'].iloc[-1]}")

    all_rows = []
    for days, label in [(30, '30d'), (60, '60d'), (90, '90d')]:
        rs = run_window(signals, df_hourly, days, label)
        all_rows.extend(rs)
        pd.DataFrame(rs).to_csv(
            os.path.join(ENGINE_DIR, f'rally_cooldown_summary_{label}.csv'),
            index=False)

    # Stability pivot
    df = pd.DataFrame(all_rows)
    piv_pnl = df.pivot(index=['code', 'name'], columns='window',
                       values='total_pnl_pct').reset_index()
    piv_dd  = df.pivot(index=['code', 'name'], columns='window',
                       values='max_dd_pct').reset_index()
    piv_pnl.columns = ['code', 'name', 'pnl_30d', 'pnl_60d', 'pnl_90d']
    piv_dd.columns  = ['code', 'name', 'dd_30d', 'dd_60d', 'dd_90d']
    stab = piv_pnl.merge(piv_dd[['code', 'dd_30d', 'dd_60d', 'dd_90d']], on='code')
    # wins_vs_baseline: how many windows beat V0?
    base = stab[stab['code'] == 'V0'].iloc[0]
    def wins_vs_base(row):
        return sum(1 for w in ('30d','60d','90d') if row[f'pnl_{w}'] > base[f'pnl_{w}'])
    stab['wins_vs_baseline'] = stab.apply(wins_vs_base, axis=1)
    stab['avg_pnl'] = (stab['pnl_30d'] + stab['pnl_60d'] + stab['pnl_90d']) / 3.0
    stab['avg_dd']  = (stab['dd_30d']  + stab['dd_60d']  + stab['dd_90d'])  / 3.0

    print("\n  === STABILITY (PnL% per window) ===")
    print(stab.to_string(index=False, formatters={
        'pnl_30d': '{:+.2f}'.format, 'pnl_60d': '{:+.2f}'.format,
        'pnl_90d': '{:+.2f}'.format, 'dd_30d': '{:+.2f}'.format,
        'dd_60d': '{:+.2f}'.format, 'dd_90d': '{:+.2f}'.format,
        'avg_pnl': '{:+.2f}'.format, 'avg_dd': '{:+.2f}'.format,
    }))
    stab.to_csv(os.path.join(ENGINE_DIR, 'rally_cooldown_stability.csv'), index=False)
    print("\n  Artifacts: rally_cooldown_summary_{30d,60d,90d}.csv, "
          "rally_cooldown_stability.csv")


if __name__ == '__main__':
    main()
