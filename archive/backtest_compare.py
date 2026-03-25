"""
Backtest comparison: V1.3.1 A (gamma search) vs V1.4 (gamma=1.0) vs Production
Tests BTC 8h and LINK 8h over the last full week, every hour.
Simulates trades at confidence thresholds 70%, 80%, 90%.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from crypto_trading_system_deku import (
    generate_signals, TRADING_FEE, PREDICTION_HORIZON
)

# ── Model configs to test ──────────────────────────────────────────
CONFIGS = {
    # BTC 8h
    'BTC_8h_v131a': {
        'asset': 'BTC', 'horizon': 8,
        'window': 200, 'combo': 'GB+LGBM', 'gamma': 0.9967,
        'features': 'volatility_48h,logret_72h,vol_ratio_12_48,volume_ratio_h,bb_position_20h,logret_8h,stoch_k_14h,hour_cos,volatility_12h,logret_240h,gk_volatility_48h,xa_nasdaq_relstr5d,spread_48h_12h,sma20_to_sma50h,xa_dax_corr10d,m_vix_chg1d,price_to_sma100h,xa_dax_relstr5d,minus_di_14h,spread_240h_24h,m_us10y_chg5d,logret_24h,logret_6h,price_accel_12h,adx_14h,price_to_sma20h,spread_48h_4h,gk_volatility_14h,logret_120h,xa_eth_usd_corr10d,m_gold_vol20d,plus_di_14h,atr_pct_14h,m_nasdaq_vol5d,xa_eth_usd_relstr5d,rsi_14h,m_usdjpy_zscore,logret_2h,fg_zscore,m_vix_chg10d,xa_sp500_corr10d',
        'label': 'V1.3.1-A (gamma=0.997, PBO=0.33)',
    },
    'BTC_8h_v14': {
        'asset': 'BTC', 'horizon': 8,
        'window': 48, 'combo': 'XGB+LR+LGBM', 'gamma': 1.0,
        'features': 'volatility_48h,logret_72h,logret_240h,volatility_12h,spread_48h_12h,xa_nasdaq_relstr5d,price_accel_12h,xa_dax_corr10d,hour_cos,logret_24h,volume_ratio_h,vol_ratio_12_48,price_to_sma100h,sma20_to_sma50h,logret_8h',
        'label': 'V1.4 (gamma=1.0, PBO=0.33)',
    },
    'BTC_8h_prod': {
        'asset': 'BTC', 'horizon': 8,
        'window': 200, 'combo': 'XGB+LR+LGBM', 'gamma': 0.9956,
        'features': 'volatility_48h,logret_72h,volume_ratio_h,xa_dax_corr10d,adx_14h,stoch_k_14h,logret_240h,gk_volatility_48h,hour_cos,vol_ratio_12_48,xa_nasdaq_relstr5d,bb_position_20h,spread_48h_12h',
        'label': 'V1.3 Prod (gamma=0.996, no CPCV)',
    },

    # LINK 8h
    'LINK_8h_v131a': {
        'asset': 'LINK', 'horizon': 8,
        'window': 200, 'combo': 'RF+GB+LR+LGBM', 'gamma': 0.9963,
        'features': 'logret_72h,vol_ratio_12_48,price_to_sma100h,logret_120h,adx_14h,xa_nasdaq_relstr5d,volatility_12h,bb_position_20h,sma20_to_sma50h,spread_24h_4h,hour_cos,atr_pct_14h,xa_eth_usd_relstr5d,m_sp500_chg1d,gk_volatility_48h,logret_48h,spread_240h_24h,logret_240h,spread_120h_8h,minus_di_14h',
        'label': 'V1.3.1-A (gamma=0.996, PBO=0.20)',
    },
    'LINK_8h_v14': {
        'asset': 'LINK', 'horizon': 8,
        'window': 150, 'combo': 'GB+XGB+LR', 'gamma': 1.0,
        'features': 'price_to_sma100h,logret_72h,logret_120h,adx_14h,bb_position_20h,xa_nasdaq_relstr5d,volatility_12h,hour_cos,vol_ratio_12_48,sma20_to_sma50h,spread_24h_4h,logret_48h,m_dxy_chg1d,gk_volatility_48h,xa_eth_usd_relstr5d,spread_48h_12h,m_sp500_chg1d,price_to_sma20h,minus_di_14h,xa_dax_relstr5d,xa_dax_corr10d,gk_volatility_14h,spread_120h_8h,atr_pct_14h,volatility_48h,plus_di_14h,xa_sp500_relstr5d,price_to_sma50h,price_accel_24h,spread_240h_24h,logret_240h,fg_zscore,m_vix_chg1d,logret_8h',
        'label': 'V1.4 (gamma=1.0, PBO=0.00)',
    },
    'LINK_8h_prod': {
        'asset': 'LINK', 'horizon': 8,
        'window': 200, 'combo': 'RF+GB+LR+LGBM', 'gamma': 0.9963,
        'features': 'logret_72h,vol_ratio_12_48,price_to_sma100h,logret_120h,adx_14h,xa_nasdaq_relstr5d,volatility_12h,bb_position_20h,sma20_to_sma50h,spread_24h_4h,hour_cos,atr_pct_14h,xa_eth_usd_relstr5d,m_sp500_chg1d,gk_volatility_48h,logret_48h,spread_240h_24h,logret_240h,spread_120h_8h,minus_di_14h',
        'label': 'V1.3 Prod (gamma=0.996, no CPCV)',
    },
}

REPLAY_HOURS = 168  # 1 full week
CONF_THRESHOLDS = [70, 80, 90]


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
            continue  # skip low-confidence signals

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

    # Final value
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
    print("  BACKTEST COMPARISON: V1.3.1-A vs V1.4 vs Production")
    print(f"  Period: last {REPLAY_HOURS} hours (1 week), every hour")
    print(f"  Confidence thresholds: {CONF_THRESHOLDS}")
    print("=" * 80)

    results = {}

    for key, cfg in CONFIGS.items():
        asset = cfg['asset']
        horizon = cfg['horizon']
        models = cfg['combo'].split('+')
        features = cfg['features'].split(',')
        gamma = cfg['gamma']
        window = cfg['window']
        label = cfg['label']

        print(f"\n{'─' * 70}")
        print(f"  {asset} {horizon}h — {label}")
        print(f"  Window={window}, Models={cfg['combo']}, Features={len(features)}, Gamma={gamma}")
        print(f"{'─' * 70}")

        signals = generate_signals(
            asset_name=asset,
            model_names=models,
            window_size=window,
            replay_hours=REPLAY_HOURS,
            feature_override=features,
            horizon=horizon,
            gamma=gamma,
        )

        if not signals:
            print(f"  [!] No signals generated for {key}")
            continue

        # Buy & hold
        bh_ret = (signals[-1]['close'] / signals[0]['close'] - 1) * 100

        results[key] = {'label': label, 'signals': len(signals), 'buy_hold': bh_ret}

        for conf in CONF_THRESHOLDS:
            sim = simulate_with_threshold(signals, conf)
            results[key][f'conf_{conf}'] = sim
            print(f"  Conf>={conf}%: return={sim['return_pct']:+.2f}%, "
                  f"trades={sim['trades']}, round_trips={sim['round_trips']}, "
                  f"win_rate={sim['win_rate']:.0f}%"
                  f"{' [still in]' if sim['still_invested'] else ''}")

        print(f"  Buy&Hold: {bh_ret:+.2f}%")

    # ── Summary table ──────────────────────────────────────────────
    print("\n")
    print("=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)

    for asset in ['BTC', 'LINK']:
        print(f"\n  {'─' * 40}")
        print(f"  {asset} 8h — Last 7 days")
        print(f"  {'─' * 40}")

        keys = [k for k in results if k.startswith(f'{asset}_8h_')]
        if not keys:
            continue

        bh = results[keys[0]]['buy_hold']
        print(f"  Buy & Hold: {bh:+.2f}%\n")

        header = f"  {'Model':<40} | {'Conf':>4} | {'Return':>8} | {'Trades':>6} | {'Trips':>5} | {'WinR':>5}"
        print(header)
        print(f"  {'─' * len(header)}")

        for key in keys:
            r = results[key]
            for conf in CONF_THRESHOLDS:
                sim = r[f'conf_{conf}']
                inv = '*' if sim['still_invested'] else ' '
                print(f"  {r['label']:<40} | {conf:>3}% | {sim['return_pct']:>+7.2f}% | {sim['trades']:>6} | {sim['round_trips']:>5} | {sim['win_rate']:>4.0f}%{inv}")

    print(f"\n  * = still invested at end of period")
    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()
