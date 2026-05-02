"""Rally-cooldown research backtest for ETH (30d cached window).

Does NOT modify production code. Reuses cached signals + 5m candles.
Simulates baseline + 6 cooldown variants (V0..V6) with rolling-return
entry gates, and produces a rally table + variant comparison.

Run:  python backtest_rally_cooldown.py
"""
from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ENGINE_DIR, 'data')
SIG_CACHE = os.path.join(DATA_DIR, 'eth_sl_signals.pkl')
FIVE_MIN = os.path.join(DATA_DIR, 'eth_5m_backtest.csv')

TRADING_FEE = 0.0005  # 5 bps/leg realistic maker blend (see ed.py BACKTEST_FEE_PER_LEG)
MIN_SELL_PNL_PCT = 0.005
MAX_HOLD_HOURS = 10


def load_caches():
    with open(SIG_CACHE, 'rb') as f:
        signals = pickle.load(f)
    df5 = pd.read_csv(FIVE_MIN)
    df5['datetime'] = pd.to_datetime(df5['datetime'], utc=True)
    df5 = df5.set_index('datetime').sort_index()
    return signals, df5


def five_min_slice(df5, hour_start):
    if hour_start.tzinfo is None:
        hour_start = hour_start.tz_localize('UTC')
    hour_end = hour_start + pd.Timedelta(hours=1)
    return df5.loc[(df5.index >= hour_start) & (df5.index < hour_end)]


def build_hourly_frame(signals):
    """One row per signal, with rolling-return columns."""
    rows = [{'datetime': s['datetime'], 'close': s['close'],
             'signal': s['signal'], 'confidence': s['confidence'],
             'conf_threshold': s['conf_threshold']} for s in signals]
    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    # rolling returns: (close / close[-N hours] - 1)*100
    for h in (6, 10, 24, 48):
        df[f'rr_{h}h'] = (df['close'] / df['close'].shift(h) - 1.0) * 100.0
    return df


# ----------------------------------------------------------------
# Step 1: top rallies
# ----------------------------------------------------------------
def rally_table(df):
    """Top-20 rallies by 24h rolling return (non-overlapping by 24h window)."""
    tmp = df.dropna(subset=['rr_24h']).copy()
    # sort by rr_24h descending and greedily pick non-overlapping peaks
    tmp = tmp.sort_values('rr_24h', ascending=False).reset_index(drop=True)
    picked = []
    used = []
    for _, r in tmp.iterrows():
        peak_t = r['datetime']
        if any(abs((peak_t - u).total_seconds()) < 24 * 3600 for u in used):
            continue
        picked.append(r)
        used.append(peak_t)
        if len(picked) >= 20:
            break
    # enrich each with start/end/post
    out = []
    close_by_t = df.set_index('datetime')['close']
    for r in picked:
        peak_t = r['datetime']
        start_t = peak_t - pd.Timedelta(hours=24)
        # nearest available bar for start price
        try:
            start_px = close_by_t.asof(start_t)
        except Exception:
            start_px = np.nan
        peak_px = r['close']
        post_t = peak_t + pd.Timedelta(hours=24)
        post_px = close_by_t.asof(post_t) if post_t <= close_by_t.index.max() else np.nan
        post_48 = peak_t + pd.Timedelta(hours=48)
        post_48_px = close_by_t.asof(post_48) if post_48 <= close_by_t.index.max() else np.nan
        ret24 = (post_px / peak_px - 1.0) * 100.0 if pd.notna(post_px) else np.nan
        ret48 = (post_48_px / peak_px - 1.0) * 100.0 if pd.notna(post_48_px) else np.nan
        out.append({
            'start_time': start_t,
            'end_time': peak_t,
            'start_price': start_px,
            'peak_price': peak_px,
            'peak_return_pct': r['rr_24h'],
            'price_at_peak_plus_24h': post_px,
            'post_peak_24h_return_pct': ret24,
            'post_peak_48h_return_pct': ret48,
            'reverted_24h': bool(pd.notna(ret24) and ret24 < 0),
            'reverted_48h': bool(pd.notna(ret48) and ret48 < 0),
        })
    return pd.DataFrame(out)


# ----------------------------------------------------------------
# Step 2: baseline trades labeled vs rallies
# ----------------------------------------------------------------
def label_baseline_trades(df):
    trades_path = os.path.join(ENGINE_DIR, 'backtest_sl_variants_trades.csv')
    tr = pd.read_csv(trades_path)
    tr = tr[tr['variant'] == 'A'].copy()
    tr['entry_time'] = pd.to_datetime(tr['entry_time'], utc=True)
    close_by_t = df.set_index('datetime')['close']
    def rr_at(t, hours):
        past = t - pd.Timedelta(hours=hours)
        now_px = close_by_t.asof(t)
        past_px = close_by_t.asof(past)
        if pd.isna(now_px) or pd.isna(past_px):
            return np.nan
        return (now_px / past_px - 1.0) * 100.0
    tr['rr_6h'] = tr['entry_time'].apply(lambda t: rr_at(t, 6))
    tr['rr_10h'] = tr['entry_time'].apply(lambda t: rr_at(t, 10))
    tr['rr_24h'] = tr['entry_time'].apply(lambda t: rr_at(t, 24))
    tr['post_rally'] = tr['rr_24h'] >= 5.0
    return tr


# ----------------------------------------------------------------
# Step 3: simulate cooldown variants
# ----------------------------------------------------------------
# Cooldown rule spec:
#   type: 'gate_only' (V1) — no clock, just skip BUY if condition now true
#   type: 'cooldown'  — if trigger fires, block BUY for `cool_hours` after
#       The trigger looks at rolling returns at each hour; once triggered,
#       cooldown timer counts down. For V2: "any point in last 24h" is
#       equivalent to: trigger each hour rr_24h>=5%, then 12h block.
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


def trigger_hit(row, cfg):
    if cfg['gate'] == 'gate':
        col, thr = cfg['cond']
        return pd.notna(row[col]) and row[col] >= thr
    if cfg['gate'] == 'cool':
        col, thr = cfg['cond']
        return pd.notna(row[col]) and row[col] >= thr
    if cfg['gate'] == 'combo':
        for col, thr in cfg['conds']:
            if pd.notna(row[col]) and row[col] >= thr:
                return True
        return False
    return False


def simulate(signals, df5, df_hourly, cfg):
    """Replicates backtest_sl_variants A logic with an added BUY-gate."""
    # map datetime -> rolling returns row
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

        # get rolling-return row for this hour
        if dt in rr_map.index:
            rr_row = rr_map.loc[dt]
        else:
            rr_row = None

        # update cooldown trigger detection (cool/combo): if condition hit now,
        # reset cooldown_remaining to cfg['hours']
        if cfg['gate'] in ('cool', 'combo') and rr_row is not None:
            if trigger_hit(rr_row, cfg):
                cooldown_remaining = max(cooldown_remaining, cfg['hours'])

        equity = cash + qty * price if in_pos else cash
        equity_curve.append(equity)

        # no intra-hour SL (baseline A)
        # ----- hourly decision -----
        if sig == 'BUY' and conf >= thr and not in_pos:
            # gate check
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
        trade_list=trades,
    )


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    print("=" * 92)
    print("  ETH Rally-Cooldown Research (30d)")
    print("=" * 92)
    signals, df5 = load_caches()
    df_hourly = build_hourly_frame(signals)
    print(f"  signals: {len(signals)}  window: {df_hourly['datetime'].iloc[0]} -> {df_hourly['datetime'].iloc[-1]}")
    print()

    # --- Step 1 ---
    rallies = rally_table(df_hourly)
    print("  === STEP 1: Top-20 24h rallies ===")
    cols = ['start_time', 'end_time', 'start_price', 'peak_price',
            'peak_return_pct', 'price_at_peak_plus_24h',
            'post_peak_24h_return_pct', 'reverted_24h', 'reverted_48h']
    with pd.option_context('display.max_rows', None, 'display.width', 200,
                           'display.max_columns', None):
        print(rallies[cols].to_string(index=False,
              formatters={'peak_return_pct': '{:+.2f}'.format,
                          'post_peak_24h_return_pct': '{:+.2f}'.format,
                          'start_price': '{:.2f}'.format,
                          'peak_price': '{:.2f}'.format,
                          'price_at_peak_plus_24h': '{:.2f}'.format}))
    rev24 = int(rallies['reverted_24h'].sum())
    rev48 = int(rallies['reverted_48h'].sum())
    print()
    print(f"  Reverted within 24h: {rev24}/{len(rallies)}")
    print(f"  Reverted within 48h: {rev48}/{len(rallies)}")
    print()

    # --- Step 2 ---
    labeled = label_baseline_trades(df_hourly)
    print("  === STEP 2: baseline trades — post-rally vs not ===")
    print(labeled[['entry_time', 'rr_6h', 'rr_10h', 'rr_24h',
                   'pnl_pct', 'post_rally']].to_string(index=False,
                   formatters={'rr_6h': '{:+.2f}'.format,
                               'rr_10h': '{:+.2f}'.format,
                               'rr_24h': '{:+.2f}'.format,
                               'pnl_pct': '{:+.2f}'.format}))
    pr = labeled[labeled['post_rally']]
    npr = labeled[~labeled['post_rally']]
    print()
    print(f"  post-rally (rr24>=+5%) trades: n={len(pr)}  mean_pnl={pr['pnl_pct'].mean():+.3f}%  "
          f"wins={(pr['pnl_pct']>0).sum()}/{len(pr)}")
    print(f"  non-post-rally trades:         n={len(npr)}  mean_pnl={npr['pnl_pct'].mean():+.3f}%  "
          f"wins={(npr['pnl_pct']>0).sum()}/{len(npr)}")
    print()

    # --- Step 3: variants ---
    print("  === STEP 3: Variant simulation ===")
    results = []
    for cfg in VARIANTS:
        r = simulate(signals, df5, df_hourly, cfg)
        results.append(r)

    baseline = results[0]
    print(f"  {'Variant':<36}{'Trades':>8}{'Wins':>6}{'Total PnL%':>12}"
          f"{'Max DD%':>10}{'Skipped':>10}{'vs Base':>10}")
    print("  " + "-" * 92)
    for r in results:
        delta = r['total_pnl_pct'] - baseline['total_pnl_pct']
        delta_s = '-' if r['code'] == 'V0' else f"{delta:+.2f}"
        print(f"  {r['code']+' '+r['name']:<36}"
              f"{r['trades']:>8}{r['wins']:>6}"
              f"{r['total_pnl_pct']:>+12.2f}{r['max_dd_pct']:>+10.2f}"
              f"{r['skipped_buys']:>10}{delta_s:>10}")

    # save artifacts
    rallies.to_csv(os.path.join(ENGINE_DIR, 'rally_top20.csv'), index=False)
    labeled.to_csv(os.path.join(ENGINE_DIR, 'baseline_trades_labeled.csv'), index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k != 'trade_list'}
                  for r in results]).to_csv(
        os.path.join(ENGINE_DIR, 'rally_cooldown_summary.csv'), index=False)
    print()
    print("  wrote rally_top20.csv, baseline_trades_labeled.csv, rally_cooldown_summary.csv")


if __name__ == '__main__':
    main()
