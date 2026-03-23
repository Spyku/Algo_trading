"""
Backtest: Doohan V1.3 top 6 candidates vs Deku Production
Loads all candidates from crypto_doohan_v1_3_best_models.csv (ranked by holdout).
Runs 1-week live replay at confidence thresholds 70%, 80%, 90%.
Saves the best live performer to crypto_doohan_v1_3_production.csv.

Usage:
    python backtest_doohan_v1_3.py              # backtest + save best
    python backtest_doohan_v1_3.py --no-save    # backtest only, don't save
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
PRIMARY_CONF = 80   # confidence threshold used to rank live performance
HORIZON = 8

CANDIDATES_CSV = os.path.join('models', 'crypto_doohan_v1_3_best_models.csv')
PRODUCTION_CSV = os.path.join('models', 'crypto_doohan_v1_3_production.csv')
DEKU_CSV = os.path.join('models', 'crypto_deku_best_models.csv')


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
    save_best = '--no-save' not in sys.argv

    # Load candidates
    if not os.path.exists(CANDIDATES_CSV):
        print(f"  ERROR: {CANDIDATES_CSV} not found. Run V1.3 Mode D first.")
        return
    df_candidates = pd.read_csv(CANDIDATES_CSV)

    # Load Deku production for comparison
    df_deku = pd.read_csv(DEKU_CSV) if os.path.exists(DEKU_CSV) else None

    # Get unique assets from candidates
    assets = df_candidates['coin'].unique().tolist()

    print("=" * 80)
    print(f"  BACKTEST: DOOHAN V1.3 CANDIDATES vs DEKU — {','.join(assets)} {HORIZON}h")
    print(f"  Period: last {REPLAY_HOURS} hours (1 week), every hour")
    print(f"  Ranking by: conf>={PRIMARY_CONF}% return")
    print(f"  Save best: {'YES → ' + PRODUCTION_CSV if save_best else 'NO (--no-save)'}")
    print("=" * 80)

    all_results = {}
    production_models = []

    for asset in assets:
        print(f"\n{'#' * 70}")
        print(f"  {asset} {HORIZON}h")
        print(f"{'#' * 70}")

        # LGBM feature ranking for this asset (needed for candidates that use ranked features)
        print(f"\n  Computing LGBM feature ranking for {asset}...")
        t0 = time.time()
        df_raw = load_data(asset)
        df_full, all_cols = build_all_features(df_raw, asset_name=asset, horizon=HORIZON)
        df_clean = df_full.dropna(subset=all_cols + ['label']).reset_index(drop=True)
        importance_df = _test_lgbm_importance(df_clean, all_cols, gamma=1.0)
        ranked_features = importance_df['feature'].tolist()
        print(f"  [Ranking done: {time.time()-t0:.1f}s] — {len(ranked_features)} features ranked")

        configs = {}

        # Deku Production (baseline)
        if df_deku is not None:
            mask = (df_deku['coin'] == asset) & (df_deku['horizon'] == HORIZON)
            rows = df_deku[mask]
            if not rows.empty:
                row = rows.iloc[0]
                configs['Deku Prod'] = {
                    'combo': row['best_combo'],
                    'window': int(row['best_window']),
                    'gamma': float(row['gamma']),
                    'features': row['optimal_features'].split(','),
                    'n_features': int(row['n_features']),
                    'source': 'deku',
                }

        # Doohan V1.3 candidates
        mask = (df_candidates['coin'] == asset) & (df_candidates['horizon'] == HORIZON)
        asset_candidates = df_candidates[mask].sort_values('rank' if 'rank' in df_candidates.columns else 'combined_score',
                                                            ascending='rank' in df_candidates.columns)

        for idx, row in asset_candidates.iterrows():
            rank = int(row.get('rank', idx + 1))
            ho_apf = float(row.get('combined_score', 0))
            label = f"V1.3 #{rank} (APF={ho_apf:.1f})"

            # Use features from CSV if available, otherwise use LGBM-ranked
            if pd.notna(row.get('optimal_features', None)) and row['optimal_features']:
                features = row['optimal_features'].split(',')
            else:
                features = ranked_features[:int(row['n_features'])]

            configs[label] = {
                'combo': row['best_combo'],
                'window': int(row['best_window']),
                'gamma': float(row['gamma']),
                'features': features,
                'n_features': int(row['n_features']),
                'source': 'doohan_v1.3',
                'rank': rank,
                'ho_apf': ho_apf,
                'csv_row': row.to_dict(),
            }

        # Run backtests
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

        # Find best Doohan candidate at PRIMARY_CONF
        doohan_results = [(label, r) for label, r in results.items()
                          if r['cfg'].get('source') == 'doohan_v1.3'
                          and f'conf_{PRIMARY_CONF}' in r]

        if doohan_results:
            best_label, best_r = max(doohan_results,
                                     key=lambda x: x[1][f'conf_{PRIMARY_CONF}']['return_pct'])
            best_sim = best_r[f'conf_{PRIMARY_CONF}']
            best_cfg = best_r['cfg']

            print(f"\n  {'='*70}")
            print(f"  LIVE BEST: {best_label}  →  {asset} {HORIZON}h")
            print(f"  {best_cfg['combo']}  w={best_cfg['window']}h  g={best_cfg['gamma']:.4f}  f={best_cfg['n_features']}")
            print(f"  Return (conf>={PRIMARY_CONF}%): {best_sim['return_pct']:+.2f}%  "
                  f"WR={best_sim['win_rate']:.0f}%  trades={best_sim['trades']}")

            # Compare vs Deku
            if 'Deku Prod' in results and f'conf_{PRIMARY_CONF}' in results['Deku Prod']:
                deku_ret = results['Deku Prod'][f'conf_{PRIMARY_CONF}']['return_pct']
                diff = best_sim['return_pct'] - deku_ret
                print(f"  vs Deku: {diff:+.2f}% {'BETTER' if diff > 0 else 'WORSE'}")
            print(f"  {'='*70}")

            if save_best:
                prod_row = best_cfg['csv_row'].copy()
                production_models.append(prod_row)

    # ── Summary ──
    print(f"\n\n{'=' * 80}")
    print(f"  SUMMARY: DOOHAN V1.3 CANDIDATES vs DEKU — Last 7 days")
    print(f"{'=' * 80}")

    for asset, results in all_results.items():
        if results:
            first = list(results.values())[0]
            print(f"\n  {asset} {HORIZON}h — Buy & Hold: {first['buy_hold']:+.2f}%\n")

        header = (f"  {'Model':<25} | {'Combo':22s} | {'W':>4} | {'G':>6} | {'F':>3} | "
                  f"{'Conf':>4} | {'Return':>8} | {'Tr':>3} | {'RT':>3} | {'WR':>4}")
        print(header)
        print(f"  {'─' * len(header)}")

        for label, r in results.items():
            cfg = r['cfg']
            for conf in CONF_THRESHOLDS:
                if f'conf_{conf}' not in r:
                    continue
                sim = r[f'conf_{conf}']
                inv = '*' if sim['still_invested'] else ' '
                print(f"  {label:<25} | {cfg['combo']:22s} | {cfg['window']:>3}h | {cfg['gamma']:>.3f} | "
                      f"{cfg['n_features']:>3} | {conf:>3}% | {sim['return_pct']:>+7.2f}% | "
                      f"{sim['trades']:>3} | {sim['round_trips']:>3} | {sim['win_rate']:>3.0f}%{inv}")

    print(f"\n  * = still invested at end of period")

    # Save production models
    if save_best and production_models:
        df_prod = pd.DataFrame(production_models)
        # Remove rank column — production model is the live-validated best
        if 'rank' in df_prod.columns:
            df_prod = df_prod.drop(columns=['rank'])

        # Merge with existing production CSV (other assets/horizons)
        if os.path.exists(PRODUCTION_CSV):
            df_existing = pd.read_csv(PRODUCTION_CSV)
            for m in production_models:
                mask = (df_existing['coin'] == m['coin']) & (df_existing['horizon'] == HORIZON)
                df_existing = df_existing[~mask]
            df_prod = pd.concat([df_existing, df_prod], ignore_index=True)

        df_prod.to_csv(PRODUCTION_CSV, index=False)
        print(f"\n  Production model saved: {PRODUCTION_CSV}")
        for m in production_models:
            print(f"    {m['coin']} {HORIZON}h: {m['best_combo']}  w={m['best_window']}h  "
                  f"g={m['gamma']}  f={m['n_features']}")

    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()
