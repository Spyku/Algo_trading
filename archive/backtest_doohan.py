"""
Backtest: Doohan V1.3 top 5 vs Deku Production
Tests BTC 8h over the last full week, every hour.
Simulates trades at confidence thresholds 70%, 80%, 90%.

Uses LGBM feature ranking to reconstruct feature lists for Doohan configs.
V1.3 winner (#1) is loaded from CSV; #2-#5 from holdout results.
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

# Doohan V1.3 holdout top 5 (multi-seed, 3×150 trials)
# #1 loaded from CSV, #2-#5 use LGBM-ranked features
DOOHAN_V13_CONFIGS = {
    'BTC': [
        # #1 (XGB+LGBM w=150 g=0.995 f=13) loaded from CSV
        {'label': 'V1.3 #2 (APF=5.2)', 'combo': 'RF+GB+XGB+LGBM', 'window': 100, 'gamma': 0.997, 'n_features': 25},
        {'label': 'V1.3 #3 (APF=4.8)', 'combo': 'GB+XGB+LGBM', 'window': 100, 'gamma': 0.998, 'n_features': 25},
        {'label': 'V1.3 #4 (APF=4.7)', 'combo': 'XGB+LR+LGBM', 'window': 150, 'gamma': 0.995, 'n_features': 20},
        {'label': 'V1.3 #5 (APF=3.3)', 'combo': 'GB+XGB+LR+LGBM', 'window': 150, 'gamma': 0.999, 'n_features': 17},
    ],
}

ASSETS = ['BTC']
HORIZON = 8


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
    print(f"  BACKTEST: DOOHAN V1.3 TOP 5 vs DEKU PRODUCTION — {','.join(ASSETS)} {HORIZON}h")
    print(f"  Period: last {REPLAY_HOURS} hours (1 week), every hour")
    print(f"  Confidence thresholds: {CONF_THRESHOLDS}")
    print("=" * 80)

    prod_csv = os.path.join('models', 'crypto_deku_best_models.csv')
    df_prod = pd.read_csv(prod_csv) if os.path.exists(prod_csv) else None

    v13_csv = os.path.join('models', 'crypto_doohan_v1_3_best_models.csv')
    df_v13 = pd.read_csv(v13_csv) if os.path.exists(v13_csv) else None

    all_results = {}

    for asset in ASSETS:
        print(f"\n{'#' * 70}")
        print(f"  {asset} {HORIZON}h")
        print(f"{'#' * 70}")

        # LGBM feature ranking for this asset
        print(f"\n  Computing LGBM feature ranking for {asset}...")
        t0 = time.time()
        df_raw = load_data(asset)
        df_full, all_cols = build_all_features(df_raw, asset_name=asset, horizon=HORIZON)
        df_clean = df_full.dropna(subset=all_cols + ['label']).reset_index(drop=True)
        importance_df = _test_lgbm_importance(df_clean, all_cols, gamma=1.0)
        ranked_features = importance_df['feature'].tolist()
        print(f"  [Ranking done: {time.time()-t0:.1f}s] — {len(ranked_features)} features ranked")

        configs = {}

        # Deku Production
        if df_prod is not None:
            mask = (df_prod['coin'] == asset) & (df_prod['horizon'] == HORIZON)
            rows = df_prod[mask]
            if not rows.empty:
                row = rows.iloc[0]
                configs['Deku Prod'] = {
                    'combo': row['best_combo'],
                    'window': int(row['best_window']),
                    'gamma': float(row['gamma']),
                    'features': row['optimal_features'].split(','),
                    'n_features': int(row['n_features']),
                }

        # Doohan V1.3 #1 winner (multi-seed) — loaded from CSV with its own features
        if df_v13 is not None:
            mask = (df_v13['coin'] == asset) & (df_v13['horizon'] == HORIZON)
            rows = df_v13[mask]
            if not rows.empty:
                row = rows.iloc[0]
                configs['V1.3 #1 (APF=5.7)'] = {
                    'combo': row['best_combo'],
                    'window': int(row['best_window']),
                    'gamma': float(row['gamma']),
                    'features': row['optimal_features'].split(','),
                    'n_features': int(row['n_features']),
                }

        # Doohan V1.3 #2-#5 holdout runners-up (use LGBM-ranked features)
        if asset in DOOHAN_V13_CONFIGS:
            for dcfg in DOOHAN_V13_CONFIGS[asset]:
                n_feat = dcfg['n_features']
                configs[dcfg['label']] = {
                    'combo': dcfg['combo'],
                    'window': dcfg['window'],
                    'gamma': dcfg['gamma'],
                    'features': ranked_features[:n_feat],
                    'n_features': n_feat,
                }

        results = {}

        for label, cfg in configs.items():
            models = cfg['combo'].split('+')
            features = cfg['features']
            window = cfg['window']
            gamma = cfg['gamma']

            print(f"\n{'─' * 70}")
            print(f"  {label}: {asset}")
            print(f"  {cfg['combo']}  w={window}h  g={gamma:.4f}  f={len(features)}")
            print(f"{'─' * 70}")

            signals = generate_signals(
                asset_name=asset,
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

        all_results[asset] = results

    # ── Summary ──
    print(f"\n\n{'=' * 80}")
    print(f"  SUMMARY: DOOHAN vs DEKU PRODUCTION — Last 7 days")
    print(f"{'=' * 80}")

    for asset, results in all_results.items():
        if results:
            first = list(results.values())[0]
            print(f"\n  {asset} {HORIZON}h — Buy & Hold: {first['buy_hold']:+.2f}%\n")

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
