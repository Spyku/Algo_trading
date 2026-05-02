"""
Full Month Backtest — Mar 6 to Apr 6, 2026
============================================================
Regenerates ML signals from scratch using walk-forward training,
applies regime detection (SMA24/100), and simulates trades.

BTC: bull=7h@85%, bear=6h@80%, $6k per trade
ETH: bull=5h@85%, bear=7h@85%, $6k per trade
============================================================
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import from the trading system
from crypto_trading_system_ed import (
    load_data, build_all_features, _compute_pysr_features,
    ALL_MODELS, get_decay_weights, FEATURE_SET_A, FEATURE_SET_B,
)
from sklearn.preprocessing import StandardScaler

# ── Config ──────────────────────────────────────────────────
BACKTEST_START = pd.Timestamp('2026-03-06 00:00')
BACKTEST_END   = pd.Timestamp('2026-04-06 13:00')
TRADING_FEE    = 0.0010  # round-trip (0.10%) — 2 × 5 bps/leg realistic maker blend (see ed.py BACKTEST_FEE_PER_LEG)
MAX_POS        = 6000    # USD per trade

# Production model configs (from crypto_ed_production.csv)
CONFIGS = {
    'BTC': {
        5: {'models': 'RF+LGBM', 'window': 300, 'gamma': 0.997,
            'features': 'volatility_48h,price_to_sma100h,volatility_12h,adx_14h,logret_72h,volume_ratio_h,plus_di_14h,pysr_5,gk_volatility_48h,logret_240h,xa_dax_corr10d,logret_6h,vol_ratio_12_48,minus_di_14h,sma20_to_sma50h,hour_cos,xa_dax_relstr5d,xa_sp500_relstr5d,logret_48h,price_accel_24h'},
        6: {'models': 'XGB+LGBM', 'window': 163, 'gamma': 0.9995,
            'features': 'pysr_5,pysr_4,price_to_sma100h,hour_cos,m_sp500_chg1d,m_nasdaq_chg1d,m_dxy_chg1d,dow_sin,pysr_3,xa_sp500_corr10d,volatility_12h,fg_chg10d,logret_72h,intraday_range,volatility_48h,m_gold_chg5d,m_dxy_chg5d,logret_2h,xa_eth_usd_corr30d,logret_24h,zscore_50h,hour_sin,xa_sp500_relstr5d,minus_di_14h'},
        7: {'models': 'XGB+LGBM', 'window': 101, 'gamma': 0.9974,
            'features': 'pysr_3,m_sp500_chg1d,pysr_2,hour_cos,price_to_sma100h,m_nasdaq_chg1d,m_sp500_chg5d'},
        8: {'models': 'RF+LGBM', 'window': 160, 'gamma': 0.9967,
            'features': 'pysr_2,m_sp500_chg1d,hour_cos,price_to_sma100h,pysr_4,m_nasdaq_chg1d,m_sp500_chg5d,pysr_5'},
    },
    'ETH': {
        5: {'models': 'RF+XGB', 'window': 200, 'gamma': 0.995,
            'features': 'pysr_5,pysr_4,volatility_48h,pysr_2,pysr_1,hour_cos,logret_5h,logret_240h,intraday_range,minus_di_14h,hour_sin,adx_14h,vol_ratio_12_48'},
        6: {'models': 'RF+LGBM', 'window': 159, 'gamma': 0.9971,
            'features': 'pysr_5,m_sp500_chg1d,hour_cos,logret_24h,hour_sin,m_sp500_chg5d,pysr_4,m_nasdaq_chg1d,m_vix_chg5d,pysr_3,volatility_48h,m_vix_chg1d,m_nasdaq_chg5d,pysr_1,dow_sin'},
        7: {'models': 'RF+LGBM', 'window': 200, 'gamma': 0.997,
            'features': 'pysr_5,pysr_1,vol_ratio_12_48,volatility_48h,pysr_4,hour_cos,adx_14h,price_to_sma20h,price_to_sma100h,m_nasdaq_chg1d,pysr_2,volatility_12h,logret_8h,xa_btc_usd_relstr5d,m_vix_chg5d,plus_di_14h,dow_sin,m_sp500_chg1d,hour_sin,spread_240h_24h'},
        8: {'models': 'XGB+LGBM', 'window': 100, 'gamma': 0.999,
            'features': 'pysr_4,volatility_48h,hour_cos,vol_ratio_12_48,m_eurusd_chg1d,price_to_sma20h,m_sp500_chg1d,adx_14h,logret_6h,volatility_12h'},
    },
}

# Regime config: which horizon + confidence per regime
REGIME_RULES = {
    'BTC': {
        'sma_fast': 24, 'sma_slow': 100,
        'bull': {'horizon': 7, 'min_conf': 85},
        'bear': {'horizon': 6, 'min_conf': 80},
    },
    'ETH': {
        'sma_fast': 24, 'sma_slow': 100,
        'bull': {'horizon': 5, 'min_conf': 85},
        'bear': {'horizon': 7, 'min_conf': 85},
    },
}


def generate_signals_for_horizon(asset, horizon, replay_hours):
    """Walk-forward signal generation for a single asset+horizon."""
    cfg = CONFIGS[asset][horizon]
    model_names = cfg['models'].split('+')
    window = cfg['window']
    gamma = cfg['gamma']
    feature_list = [f.strip() for f in cfg['features'].split(',') if f.strip()]

    print(f"\n  [{asset} {horizon}h] Loading data & building features...")
    df_raw = load_data(asset)
    if df_raw is None:
        print(f"  ERROR: No data for {asset}")
        return {}

    df_full, all_cols = build_all_features(df_raw, asset_name=asset, horizon=horizon, verbose=False)
    _compute_pysr_features(df_full, all_cols, asset, horizon, verbose=False)

    feature_cols = [f for f in feature_list if f in all_cols]
    missing = [f for f in feature_list if f not in all_cols]
    if missing:
        print(f"    WARNING: missing features: {missing}")
    if not feature_cols:
        print(f"    ERROR: no valid features!")
        return {}

    df = df_full.dropna(subset=feature_cols + ['label']).reset_index(drop=True)
    n = len(df)

    # Find start index for our backtest period
    start_idx = None
    for idx in range(n):
        if df.iloc[idx]['datetime'] >= BACKTEST_START:
            start_idx = idx
            break
    if start_idx is None or start_idx < window + 50:
        start_idx = max(window + 50, n - replay_hours)

    print(f"    Data: {n} rows | Features: {len(feature_cols)} | "
          f"Window: {window}h | Gamma: {gamma}")
    print(f"    Backtesting from idx {start_idx} to {n-1} "
          f"({n - start_idx} hours)")
    print(f"    Period: {df.iloc[start_idx]['datetime']} to {df.iloc[n-1]['datetime']}")

    signals = {}
    count = 0

    for i in range(start_idx, n):
        row = df.iloc[i]
        dt = row['datetime']

        train_start = max(0, i - window)
        train_end = max(train_start, i - horizon)  # embargo
        train = df.iloc[train_start:train_end]
        X_train = train[feature_cols]
        y_train = train['label'].values
        X_test = df.iloc[i:i+1][feature_cols]

        if len(np.unique(y_train)) < 2:
            continue
        if X_train.isnull().any().any() or X_test.isnull().any().any():
            continue

        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
        X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

        sw = get_decay_weights(len(y_train), gamma)
        votes, probas = [], []

        for model_name in model_names:
            try:
                model = ALL_MODELS[model_name]()
                model.fit(X_train_s, y_train, sample_weight=sw)
                votes.append(model.predict(X_test_s)[0])
                probas.append(model.predict_proba(X_test_s)[0][1])
            except Exception:
                continue

        if not votes:
            continue

        buy_votes = sum(votes)
        buy_ratio = buy_votes / len(votes)

        if buy_ratio > 0.5:
            signal = 'BUY'
        elif buy_ratio == 0:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        avg_proba = np.mean(probas)
        if signal == 'SELL':
            confidence = round((1 - avg_proba) * 100)
        else:
            confidence = round(avg_proba * 100)

        hour_key = pd.Timestamp(dt).floor('h')
        signals[hour_key] = {
            'signal': signal,
            'confidence': confidence,
            'price': float(row['close']),
            'datetime': dt,
        }

        count += 1
        if count % 100 == 0:
            print(f"    [{count}] {dt}: {signal} ({confidence}%) ${row['close']:,.2f}")

    print(f"    Generated {len(signals)} signals for {asset} {horizon}h")
    return signals


def compute_regime_map(asset, rule):
    """Compute bull/bear regime for every hour in backtest period."""
    df_raw = load_data(asset)
    fast = rule['sma_fast']
    slow = rule['sma_slow']

    lookback = BACKTEST_START - timedelta(hours=slow + 50)
    df = df_raw[df_raw['datetime'] >= lookback].copy().sort_values('datetime').reset_index(drop=True)
    df['sma_fast'] = df['close'].rolling(fast).mean()
    df['sma_slow'] = df['close'].rolling(slow).mean()
    df['regime'] = np.where(df['sma_fast'] > df['sma_slow'], 'bull', 'bear')
    df['hour'] = df['datetime'].dt.floor('h')

    return dict(zip(df['hour'], zip(df['regime'], df['close'])))


def simulate_regime_trades(asset, all_signals, regime_map, rule):
    """Simulate trades using regime-aware horizon selection."""
    # Get all unique hours, sorted
    hours = sorted(set(
        list(all_signals.get(rule['bull']['horizon'], {}).keys()) +
        list(all_signals.get(rule['bear']['horizon'], {}).keys())
    ))

    # Filter to backtest period
    hours = [h for h in hours if BACKTEST_START <= h <= BACKTEST_END]

    trades = []
    position = None
    signals_used = []

    for h in hours:
        regime_info = regime_map.get(h)
        if regime_info is None:
            continue
        regime, price = regime_info

        # Pick horizon + confidence threshold based on regime
        regime_cfg = rule[regime]
        horizon = regime_cfg['horizon']
        min_conf = regime_cfg['min_conf']

        # Get signal for this horizon at this hour
        sig = all_signals.get(horizon, {}).get(h)
        if sig is None:
            continue

        signal = sig['signal']
        conf = sig['confidence']
        price = sig['price']

        signals_used.append({
            'hour': h, 'regime': regime, 'horizon': horizon,
            'signal': signal, 'confidence': conf, 'price': price,
        })

        if signal == 'BUY' and conf >= min_conf and position is None:
            position = {
                'entry_price': price, 'entry_time': h,
                'regime': regime, 'horizon': horizon,
            }
        elif signal == 'SELL' and position is not None:
            pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
            pnl_usd = MAX_POS * pnl_pct / 100
            fee = MAX_POS * TRADING_FEE
            pnl_usd -= fee

            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': h,
                'entry_price': position['entry_price'],
                'exit_price': price,
                'regime': position['regime'],
                'horizon': position['horizon'],
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'hold_hours': (h - position['entry_time']).total_seconds() / 3600,
            })
            position = None

    unrealized = None
    if position is not None:
        last_sig = signals_used[-1] if signals_used else None
        if last_sig:
            last_price = last_sig['price']
            pnl_pct = (last_price - position['entry_price']) / position['entry_price'] * 100
            pnl_usd = MAX_POS * pnl_pct / 100
            unrealized = {
                'entry_price': position['entry_price'],
                'entry_time': position['entry_time'],
                'current_price': last_price,
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'regime': position['regime'],
            }

    return trades, unrealized, signals_used


def print_report(asset, trades, unrealized, signals_used, rule):
    """Print detailed trade report."""
    print(f"\n{'='*75}")
    print(f"  {asset} — ${MAX_POS:,} per trade | "
          f"Bull: {rule['bull']['horizon']}h@{rule['bull']['min_conf']}% | "
          f"Bear: {rule['bear']['horizon']}h@{rule['bear']['min_conf']}%")
    print(f"{'='*75}")

    if not trades and not unrealized:
        print("  No trades")
        return

    # Weekly breakdown
    week_starts = pd.date_range(BACKTEST_START, BACKTEST_END, freq='7D')
    for wi in range(len(week_starts)):
        ws = week_starts[wi]
        we = week_starts[wi + 1] if wi + 1 < len(week_starts) else BACKTEST_END + timedelta(days=1)
        label = f"Week {wi+1} ({ws.strftime('%b %d')} - {(we - timedelta(days=1)).strftime('%b %d')})"

        wk_trades = [t for t in trades if ws <= t['entry_time'] < we]
        if not wk_trades:
            print(f"\n  --- {label} ---")
            print(f"  No trades")
            continue

        wins = [t for t in wk_trades if t['pnl_usd'] > 0]
        losses = [t for t in wk_trades if t['pnl_usd'] <= 0]
        total = sum(t['pnl_usd'] for t in wk_trades)
        wr = len(wins) / len(wk_trades) * 100 if wk_trades else 0

        print(f"\n  --- {label} ---")
        print(f"  Trades: {len(wk_trades)} ({len(wins)}W / {len(losses)}L) | "
              f"WR: {wr:.0f}% | PnL: ${total:+.2f}")

        for i, t in enumerate(wk_trades, 1):
            r = t['regime'][0].upper()
            print(f"    #{i} [{r}/{t['horizon']}h] "
                  f"{t['entry_time'].strftime('%m/%d %H:%M')} -> "
                  f"{t['exit_time'].strftime('%m/%d %H:%M')} | "
                  f"${t['entry_price']:,.2f} -> ${t['exit_price']:,.2f} | "
                  f"{t['pnl_pct']:+.2f}% | ${t['pnl_usd']:+.2f} | "
                  f"{t['hold_hours']:.0f}h")

    if unrealized:
        r = unrealized['regime'][0].upper()
        print(f"\n  [OPEN] [{r}] since {unrealized['entry_time'].strftime('%m/%d %H:%M')} | "
              f"${unrealized['entry_price']:,.2f} -> ${unrealized['current_price']:,.2f} | "
              f"{unrealized['pnl_pct']:+.2f}% | ${unrealized['pnl_usd']:+.2f} (unrealized)")

    # Totals
    total_realized = sum(t['pnl_usd'] for t in trades)
    total_unrealized = unrealized['pnl_usd'] if unrealized else 0
    all_wins = [t for t in trades if t['pnl_usd'] > 0]
    all_losses = [t for t in trades if t['pnl_usd'] <= 0]
    wr_all = len(all_wins) / len(trades) * 100 if trades else 0
    avg_hold = np.mean([t['hold_hours'] for t in trades]) if trades else 0
    avg_win = np.mean([t['pnl_usd'] for t in all_wins]) if all_wins else 0
    avg_loss = np.mean([t['pnl_usd'] for t in all_losses]) if all_losses else 0

    # Regime breakdown
    bull_trades = [t for t in trades if t['regime'] == 'bull']
    bear_trades = [t for t in trades if t['regime'] == 'bear']
    bull_pnl = sum(t['pnl_usd'] for t in bull_trades)
    bear_pnl = sum(t['pnl_usd'] for t in bear_trades)
    bull_wins = sum(1 for t in bull_trades if t['pnl_usd'] > 0)
    bear_wins = sum(1 for t in bear_trades if t['pnl_usd'] > 0)

    print(f"\n  {'─'*70}")
    print(f"  TOTAL: {len(trades)} trades ({len(all_wins)}W/{len(all_losses)}L) | WR: {wr_all:.0f}%")
    print(f"  Realized: ${total_realized:+.2f} | Unrealized: ${total_unrealized:+.2f}")
    print(f"  Avg Win: ${avg_win:+.2f} | Avg Loss: ${avg_loss:+.2f} | Avg Hold: {avg_hold:.1f}h")

    if trades:
        best = max(trades, key=lambda t: t['pnl_usd'])
        worst = min(trades, key=lambda t: t['pnl_usd'])
        print(f"  Best: ${best['pnl_usd']:+.2f} ({best['entry_time'].strftime('%m/%d')}) | "
              f"Worst: ${worst['pnl_usd']:+.2f} ({worst['entry_time'].strftime('%m/%d')})")

    print(f"\n  Regime breakdown:")
    print(f"    BULL: {len(bull_trades)} trades ({bull_wins}W) | ${bull_pnl:+.2f}")
    print(f"    BEAR: {len(bear_trades)} trades ({bear_wins}W) | ${bear_pnl:+.2f}")

    # Signal distribution
    buy_sigs = [s for s in signals_used if s['signal'] == 'BUY']
    sell_sigs = [s for s in signals_used if s['signal'] == 'SELL']
    hold_sigs = [s for s in signals_used if s['signal'] == 'HOLD']
    print(f"\n  Signal distribution: {len(buy_sigs)} BUY | {len(sell_sigs)} SELL | {len(hold_sigs)} HOLD")
    if buy_sigs:
        confs = [s['confidence'] for s in buy_sigs]
        print(f"    BUY confidence: min={min(confs)}% | avg={np.mean(confs):.0f}% | max={max(confs)}%")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("#" * 75)
    print("  FULL MONTH BACKTEST — ML Signal Regeneration")
    print(f"  Period: {BACKTEST_START.strftime('%Y-%m-%d')} to {BACKTEST_END.strftime('%Y-%m-%d')}")
    print(f"  Fee: {TRADING_FEE*100:.2f}% round-trip | Position: ${MAX_POS:,}")
    print("#" * 75)

    replay_hours = int((BACKTEST_END - BACKTEST_START).total_seconds() / 3600) + 200  # extra buffer

    all_trades = {}
    all_unrealized = {}
    all_signals_used = {}

    for asset in ['BTC', 'ETH']:
        rule = REGIME_RULES[asset]
        needed_horizons = set([rule['bull']['horizon'], rule['bear']['horizon']])

        print(f"\n\n{'#'*75}")
        print(f"  {asset}: Generating signals for horizons {sorted(needed_horizons)}")
        print(f"{'#'*75}")

        # Generate signals for each needed horizon
        asset_signals = {}
        for h in sorted(needed_horizons):
            asset_signals[h] = generate_signals_for_horizon(asset, h, replay_hours)

        # Compute regime map
        print(f"\n  Computing regime map (SMA{rule['sma_fast']}/{rule['sma_slow']})...")
        regime_map = compute_regime_map(asset, rule)

        # Count regimes in backtest period
        bt_regimes = {h: r for h, (r, p) in regime_map.items()
                      if BACKTEST_START <= h <= BACKTEST_END}
        bull_hrs = sum(1 for r in bt_regimes.values() if r == 'bull')
        bear_hrs = sum(1 for r in bt_regimes.values() if r == 'bear')
        print(f"    Regime: {bull_hrs} bull hours / {bear_hrs} bear hours")

        # Simulate trades
        trades, unrealized, signals_used = simulate_regime_trades(
            asset, asset_signals, regime_map, rule
        )

        all_trades[asset] = trades
        all_unrealized[asset] = unrealized
        all_signals_used[asset] = signals_used

        print_report(asset, trades, unrealized, signals_used, rule)

    # ── Portfolio Summary ──────────────────────────────────────
    print(f"\n\n{'#'*75}")
    print(f"  PORTFOLIO SUMMARY — ${MAX_POS * 2:,} ($6k BTC + $6k ETH)")
    print(f"  {BACKTEST_START.strftime('%Y-%m-%d')} to {BACKTEST_END.strftime('%Y-%m-%d')}")
    print(f"{'#'*75}")

    btc_realized = sum(t['pnl_usd'] for t in all_trades['BTC'])
    eth_realized = sum(t['pnl_usd'] for t in all_trades['ETH'])
    btc_unr = all_unrealized['BTC']['pnl_usd'] if all_unrealized['BTC'] else 0
    eth_unr = all_unrealized['ETH']['pnl_usd'] if all_unrealized['ETH'] else 0

    # Weekly breakdown
    week_starts = pd.date_range(BACKTEST_START, BACKTEST_END, freq='7D')
    print(f"\n  {'':30s} {'BTC':>10s} {'ETH':>10s} {'TOTAL':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    for wi in range(len(week_starts)):
        ws = week_starts[wi]
        we = week_starts[wi + 1] if wi + 1 < len(week_starts) else BACKTEST_END + timedelta(days=1)
        label = f"Wk{wi+1} ({ws.strftime('%b %d')}-{(we-timedelta(days=1)).strftime('%b %d')})"

        btc_wk = sum(t['pnl_usd'] for t in all_trades['BTC'] if ws <= t['entry_time'] < we)
        eth_wk = sum(t['pnl_usd'] for t in all_trades['ETH'] if ws <= t['entry_time'] < we)
        print(f"  {label:30s} {'${:+.0f}'.format(btc_wk):>10s} {'${:+.0f}'.format(eth_wk):>10s} {'${:+.0f}'.format(btc_wk+eth_wk):>10s}")

    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Realized':30s} {'${:+.0f}'.format(btc_realized):>10s} {'${:+.0f}'.format(eth_realized):>10s} {'${:+.0f}'.format(btc_realized+eth_realized):>10s}")
    print(f"  {'Unrealized':30s} {'${:+.0f}'.format(btc_unr):>10s} {'${:+.0f}'.format(eth_unr):>10s} {'${:+.0f}'.format(btc_unr+eth_unr):>10s}")
    print(f"  {'='*30} {'='*10} {'='*10} {'='*10}")
    grand = btc_realized + eth_realized + btc_unr + eth_unr
    print(f"  {'TOTAL PnL':30s} {'':>10s} {'':>10s} {'${:+.0f}'.format(grand):>10s}")
    print(f"  {'Return on $12k':30s} {'':>10s} {'':>10s} {'{:+.2f}%'.format(grand/12000*100):>10s}")

    # Win rates
    def wr(trades):
        if not trades: return 'N/A'
        w = sum(1 for t in trades if t['pnl_usd'] > 0)
        return f"{w}/{len(trades)} ({w/len(trades)*100:.0f}%)"

    print(f"\n  Win Rates:")
    print(f"    BTC: {wr(all_trades['BTC'])}")
    print(f"    ETH: {wr(all_trades['ETH'])}")
    print(f"    Combined: {wr(all_trades['BTC'] + all_trades['ETH'])}")

    # Price context
    print(f"\n  Price Context:")
    for asset in ['BTC', 'ETH']:
        sigs = all_signals_used[asset]
        if sigs:
            first_p = sigs[0]['price']
            last_p = sigs[-1]['price']
            all_p = [s['price'] for s in sigs]
            print(f"    {asset}: ${first_p:,.0f} -> ${last_p:,.0f} "
                  f"({(last_p/first_p-1)*100:+.1f}%) "
                  f"| Low: ${min(all_p):,.0f} | High: ${max(all_p):,.0f}")

    print(f"\n{'='*75}")
    print(f"  Done. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*75}")
