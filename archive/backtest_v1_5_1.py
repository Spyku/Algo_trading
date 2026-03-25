"""
Backtest: V1.5.1 holdout winners vs Production
Tests BTC 8h over the last full week, every hour.
Simulates trades at confidence thresholds 70%, 80%, 90%.

Compares APF-scored and Return-scored V1.5.1 winners against production.
Uses LGBM feature ranking to reconstruct feature lists.
"""
import sys, os, time
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from crypto_trading_system_deku import (
    generate_signals, TRADING_FEE, load_data, build_all_features,
    _test_lgbm_importance,
)

REPLAY_HOURS = 168  # 1 full week
CONF_THRESHOLDS = [70, 80, 90]
ASSET = 'BTC'
HORIZON = 8

# APF-scored V1.5.1 winners (holdout ranked by AVG_APF)
CONFIGS_APF = {
    'APF #1 (AVG_APF=4.34)': {
        'combo': 'GB+XGB+LR+LGBM', 'window': 48, 'gamma': 0.997, 'n_features': 30,
    },
    'APF #2 (AVG_APF=2.88)': {
        'combo': 'GB+XGB+LGBM', 'window': 72, 'gamma': 0.999, 'n_features': 13,
    },
}

# Return-scored V1.5.1 winners (holdout ranked by AVG_Return)
CONFIGS_RET = {
    'RET #1 (avg=+9.6%)': {
        'combo': 'RF+LGBM', 'window': 72, 'gamma': 0.998, 'n_features': 40,
    },
    'RET #2 (avg=+9.4%)': {
        'combo': 'RF+GB+LGBM', 'window': 72, 'gamma': 0.999, 'n_features': 40,
    },
    'RET #3 (avg=+9.1%)': {
        'combo': 'GB+XGB+LGBM', 'window': 48, 'gamma': 0.999, 'n_features': 40,
    },
}

CONFIGS = {**CONFIGS_APF, **CONFIGS_RET}


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
    print("=" * 80)
    print(f"  BACKTEST: V1.5.1 APF vs RETURN vs PRODUCTION — {ASSET} {HORIZON}h")
    print(f"  Period: last {REPLAY_HOURS} hours (1 week), every hour")
    print(f"  Confidence thresholds: {CONF_THRESHOLDS}")
    print("=" * 80)

    # Step 1: LGBM feature ranking
    print(f"\n  Computing LGBM feature ranking for {ASSET}...")
    t0 = time.time()
    df_raw = load_data(ASSET)
    df_full, all_cols = build_all_features(df_raw, asset_name=ASSET, horizon=HORIZON)
    df_clean = df_full.dropna(subset=all_cols + ['label']).reset_index(drop=True)
    importance_df = _test_lgbm_importance(df_clean, all_cols, gamma=1.0)
    ranked_features = importance_df['feature'].tolist()
    print(f"  [Ranking done: {time.time()-t0:.1f}s] — {len(ranked_features)} features ranked")

    # Step 2: Load production config
    prod_csv = os.path.join('models', 'crypto_deku_best_models.csv')
    prod_features = None
    if os.path.exists(prod_csv):
        df_prod = pd.read_csv(prod_csv)
        mask = (df_prod['coin'] == ASSET) & (df_prod['horizon'] == HORIZON)
        rows = df_prod[mask]
        if not rows.empty:
            row = rows.iloc[0]
            prod_features = row['optimal_features'].split(',')

    # Step 3: Build configs
    all_configs = {}

    if prod_features:
        df_prod_row = df_prod[mask].iloc[0]
        all_configs['Production'] = {
            'combo': df_prod_row['best_combo'],
            'window': int(df_prod_row['best_window']),
            'gamma': float(df_prod_row['gamma']),
            'features': prod_features,
            'n_features': len(prod_features),
        }

    for label, cfg in CONFIGS.items():
        n_feat = cfg['n_features']
        features = ranked_features[:n_feat]
        all_configs[label] = {
            'combo': cfg['combo'],
            'window': cfg['window'],
            'gamma': cfg['gamma'],
            'features': features,
            'n_features': n_feat,
        }

    results = {}

    for label, cfg in all_configs.items():
        models = cfg['combo'].split('+')
        features = cfg['features']
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
            print(f"  {label:<25} | {cfg['combo']:22s} | {cfg['window']:>3}h | {cfg['gamma']:>.3f} | {cfg['n_features']:>3} | "
                  f"{conf:>3}% | {sim['return_pct']:>+7.2f}% | {sim['trades']:>3} | {sim['round_trips']:>3} | {sim['win_rate']:>3.0f}%{inv}")

    print(f"\n  * = still invested at end of period")
    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()
