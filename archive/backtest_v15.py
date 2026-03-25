"""
Backtest comparison: V1.5 holdout winners (CURRENT vs A vs B) vs Production
Tests BTC 8h over the last full week, every hour.
Simulates trades at confidence thresholds 70%, 80%, 90%.

Reads configs from V1.5 CSV files + production best_models.csv.
"""
import sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from crypto_trading_system_deku import (
    generate_signals, TRADING_FEE,
)

REPLAY_HOURS = 168  # 1 full week
CONF_THRESHOLDS = [70, 80, 90]
ASSET = 'BTC'
HORIZON = 8


def load_config_from_csv(csv_path, asset, horizon):
    """Load best model config from a CSV file."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    mask = (df['coin'] == asset) & (df['horizon'] == horizon)
    rows = df[mask]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        'combo': row['best_combo'],
        'window': int(row['best_window']),
        'gamma': float(row['gamma']),
        'features': row['optimal_features'],
        'n_features': int(row['n_features']),
    }


def simulate_with_threshold(signals, conf_threshold):
    """Simulate trades with a confidence threshold filter."""
    cash = 1000.0
    qty = 0
    position = 'cash'
    trades = 0
    trade_log = []
    entry_price = 0

    for sig in signals:
        price = sig['close']
        conf = sig['confidence']

        if conf < conf_threshold:
            continue

        if sig['signal'] == 'BUY' and position == 'cash':
            qty = cash * (1 - TRADING_FEE) / price
            entry_price = price
            cash = 0
            position = 'invested'
            trades += 1
        elif sig['signal'] == 'SELL' and position == 'invested':
            cash = qty * price * (1 - TRADING_FEE)
            pnl_pct = (price / entry_price - 1) * 100
            trade_log.append(pnl_pct)
            qty = 0
            position = 'cash'
            trades += 1

    final = cash if position == 'cash' else qty * signals[-1]['close']
    ret = (final / 1000 - 1) * 100
    winners = sum(1 for t in trade_log if t > 0)
    win_rate = (winners / len(trade_log) * 100) if trade_log else 0

    return {
        'return_pct': ret,
        'final_value': final,
        'trades': trades,
        'round_trips': len(trade_log),
        'win_rate': win_rate,
        'trade_returns': trade_log,
        'still_invested': position == 'invested',
    }


def main():
    # ── Load configs from CSV files ──
    configs = {}

    # Production model
    prod_csv = os.path.join('models', 'crypto_deku_best_models.csv')
    prod_cfg = load_config_from_csv(prod_csv, ASSET, HORIZON)
    if prod_cfg:
        configs['Production'] = prod_cfg

    # V1.5 holdout winners
    for mode in ['current', 'A', 'B']:
        csv_path = os.path.join('models', f'crypto_deku_v1_5_best_models_{mode}.csv')
        cfg = load_config_from_csv(csv_path, ASSET, HORIZON)
        if cfg:
            configs[f'V1.5 {mode.upper()}'] = cfg

    if not configs:
        print("No configs found. Run V1.5 first.")
        return

    print("=" * 80)
    print(f"  BACKTEST: V1.5 HOLDOUT COMPARISON — {ASSET} {HORIZON}h")
    print(f"  Period: last {REPLAY_HOURS} hours (1 week), every hour")
    print(f"  Confidence thresholds: {CONF_THRESHOLDS}")
    print(f"  Models loaded: {', '.join(configs.keys())}")
    print("=" * 80)

    results = {}

    for label, cfg in configs.items():
        models = cfg['combo'].split('+')
        features = cfg['features'].split(',')
        window = cfg['window']
        gamma = cfg['gamma']

        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"  {cfg['combo']}  w={window}h  g={gamma:.4f}  f={len(features)}")
        print(f"{'─' * 70}")

        signals = generate_signals(
            asset_name=ASSET,
            model_names=models,
            window_size=window,
            replay_hours=REPLAY_HOURS,
            feature_override=features,
            horizon=HORIZON,
            gamma=gamma,
        )

        if not signals:
            print(f"  [!] No signals generated")
            continue

        bh_ret = (signals[-1]['close'] / signals[0]['close'] - 1) * 100
        results[label] = {'cfg': cfg, 'signals': len(signals), 'buy_hold': bh_ret}

        for conf in CONF_THRESHOLDS:
            sim = simulate_with_threshold(signals, conf)
            results[label][f'conf_{conf}'] = sim
            print(f"  Conf>={conf}%: return={sim['return_pct']:+.2f}%, "
                  f"trades={sim['trades']}, round_trips={sim['round_trips']}, "
                  f"win_rate={sim['win_rate']:.0f}%"
                  f"{' [still in]' if sim['still_invested'] else ''}")

        print(f"  Buy&Hold: {bh_ret:+.2f}%")

    # ── Summary table ──
    print(f"\n\n{'=' * 80}")
    print(f"  SUMMARY: {ASSET} {HORIZON}h — Last 7 days")
    print(f"{'=' * 80}")

    if results:
        first = list(results.values())[0]
        print(f"\n  Buy & Hold: {first['buy_hold']:+.2f}%\n")

    header = f"  {'Model':<25} | {'Combo':22s} | {'W':>4} | {'G':>6} | {'F':>3} | {'Conf':>4} | {'Return':>8} | {'Tr':>3} | {'RT':>3} | {'WR':>4}"
    print(header)
    print(f"  {'─' * len(header)}")

    for label, r in results.items():
        cfg = r['cfg']
        for conf in CONF_THRESHOLDS:
            sim = r[f'conf_{conf}']
            inv = '*' if sim['still_invested'] else ' '
            print(f"  {label:<25} | {cfg['combo']:22s} | {cfg['window']:>3}h | {cfg['gamma']:>.3f} | {len(cfg['features'].split(',')):>3} | "
                  f"{conf:>3}% | {sim['return_pct']:>+7.2f}% | {sim['trades']:>3} | {sim['round_trips']:>3} | {sim['win_rate']:>3.0f}%{inv}")

    print(f"\n  * = still invested at end of period")
    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()
