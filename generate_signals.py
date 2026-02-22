"""
Generate Signals — Backtest + Dashboard
=========================================
Reads config from daily_setup.py, generates hourly signals,
simulates leveraged portfolios, and exports HTML dashboard.

Prerequisites:
  Run daily_setup.py first to generate:
    - data/setup_config.json (features, best models)
    - data/hourly_best_models.csv
    - data/indices/*.csv (hourly OHLCV)

Usage:
  python generate_signals.py
"""

import sys
import os

os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from features_v2 import build_features_v2_hourly
from hardware_config import get_all_models

try:
    import sklearn.utils.parallel
    sklearn.utils.parallel.warnings.warn = lambda *a, **kw: None
except Exception:
    pass

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# LOAD SETUP CONFIG
# ============================================================
def load_setup_config():
    """Load config exported by daily_setup.py."""
    config_path = 'data/setup_config.json'
    csv_path = 'data/hourly_best_models.csv'

    if not os.path.exists(config_path):
        print(f"ERROR: {config_path} not found!")
        print("Run daily_setup.py first.")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = json.load(f)

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found!")
        sys.exit(1)

    df_best = pd.read_csv(csv_path)
    config['best_models_df'] = df_best

    return config


# ============================================================
# ASSET CONFIG (must match daily_setup.py paths)
# ============================================================
ASSETS = {
    'SMI':   {'ticker': '^SSMI',  'file': 'data/indices/smi_hourly_data.csv'},
    'DAX':   {'ticker': '^GDAXI', 'file': 'data/indices/dax_hourly_data.csv'},
    'CAC40': {'ticker': '^FCHI',  'file': 'data/indices/cac40_hourly_data.csv'},
}


# ============================================================
# USER INPUT
# ============================================================
def get_user_input():
    """Ask user for cash amount."""
    print("\n" + "=" * 60)
    print("  BACKTEST CONFIGURATION")
    print("=" * 60)

    # Cash input
    while True:
        raw = input("\n  Starting cash (e.g. 10000): ").strip().replace(',', '').replace("'", '')
        try:
            cash = float(raw)
            if cash > 0:
                break
            print("  Must be positive.")
        except ValueError:
            print("  Enter a number.")

    # Fixed leverage: always 1:1, 1:5, 1:10 on same chart
    leverage_list = [1, 5, 10]

    # Backtest: always last month (last week marker added on chart)
    replay_hours = 200

    print(f"\n  Cash: {cash:,.0f}")
    print(f"  Leverage: 1:1, 1:5, 1:10 (all on same chart, click legend to toggle)")
    print(f"  Backtest: last month (~{replay_hours}h) with last-week marker")

    return cash, leverage_list, replay_hours


# ============================================================
# DATA + FEATURES
# ============================================================
def load_data(asset_name):
    """Load hourly CSV."""
    filepath = ASSETS[asset_name]['file']
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found. Skipping {asset_name}.")
        return None
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)


def build_hourly_features(df_hourly, prediction_horizon=4):
    """Build base features from hourly OHLCV (same as daily_setup.py)."""
    df = df_hourly.copy()

    for period in [1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 120, 240]:
        df[f'logret_{period}h'] = np.log(df['close'] / df['close'].shift(period))

    df['spread_24h_4h']   = df['logret_24h']  - df['logret_4h']
    df['spread_48h_4h']   = df['logret_48h']  - df['logret_4h']
    df['spread_120h_8h']  = df['logret_120h'] - df['logret_8h']
    df['spread_240h_24h'] = df['logret_240h'] - df['logret_24h']
    df['spread_48h_12h']  = df['logret_48h']  - df['logret_12h']
    df['spread_120h_12h'] = df['logret_120h'] - df['logret_12h']

    df['sma20h']  = df['close'].rolling(20).mean()
    df['sma50h']  = df['close'].rolling(50).mean()
    df['sma100h'] = df['close'].rolling(100).mean()
    df['sma200h'] = df['close'].rolling(200).mean()
    df['price_to_sma20h']  = df['close'] / df['sma20h'] - 1
    df['price_to_sma50h']  = df['close'] / df['sma50h'] - 1
    df['price_to_sma100h'] = df['close'] / df['sma100h'] - 1
    df['sma20_to_sma50h']  = df['sma20h'] / df['sma50h'] - 1

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14h'] = 100 - (100 / (1 + rs))

    low14  = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['stoch_k_14h'] = 100 * (df['close'] - low14) / (high14 - low14)

    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position_20h'] = (df['close'] - (bb_mid - 2 * bb_std)) / (4 * bb_std)

    roll_mean = df['close'].rolling(50).mean()
    roll_std  = df['close'].rolling(50).std()
    df['zscore_50h'] = (df['close'] - roll_mean) / roll_std

    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    df['atr_pct_14h'] = tr.rolling(14).mean() / df['close']

    df['intraday_range'] = (df['high'] - df['low']) / df['close']

    df['volatility_12h'] = df['logret_1h'].rolling(12).std()
    df['volatility_48h'] = df['logret_1h'].rolling(48).std()
    df['vol_ratio_12_48'] = df['volatility_12h'] / df['volatility_48h']

    if df['volume'].sum() == 0 or df['volume'].isna().all():
        df['volume_ratio_h'] = 1.0
    else:
        df['volume'] = df['volume'].replace(0, np.nan).ffill().bfill()
        vol_sma = df['volume'].rolling(20).mean()
        df['volume_ratio_h'] = df['volume'] / vol_sma
        df['volume_ratio_h'] = df['volume_ratio_h'].fillna(1.0)

    hour = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    dow = df['datetime'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    future_return = df['close'].shift(-prediction_horizon) / df['close'] - 1
    rolling_median = future_return.rolling(200, min_periods=50).median().shift(prediction_horizon)
    df['label'] = (future_return > rolling_median).astype(int)

    feature_cols = [
        'logret_1h', 'logret_2h', 'logret_3h', 'logret_6h',
        'logret_12h', 'logret_24h', 'logret_48h', 'logret_72h',
        'logret_240h', 'spread_120h_8h', 'sma20_to_sma50h',
        'rsi_14h', 'bb_position_20h', 'zscore_50h', 'atr_pct_14h',
        'intraday_range', 'volatility_48h',
        'hour_sin', 'dow_sin', 'dow_cos',
    ]

    display_cols = ['spread_24h_4h']
    keep_cols = ['datetime', 'close', 'high', 'low', 'volume'] + feature_cols + display_cols + ['label']
    df = df[keep_cols].copy()

    nan_counts = df[feature_cols + ['label']].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"    Rows before dropna: {len(df)}")
        top_nan = sorted({k: v for k, v in dict(nan_cols).items() if v > 0}.items(), key=lambda x: -x[1])[:5]
        print(f"    Top NaN columns: {dict(top_nan)}")

    df = df.dropna().reset_index(drop=True)
    print(f"    Rows after dropna: {len(df)}")
    return df, feature_cols


# ============================================================
# SIGNAL GENERATION (walk-forward)
# ============================================================
def generate_signals(asset_name, model_names, window_size, feature_cols,
                     replay_hours, prediction_horizon):
    """Generate hourly signals using walk-forward training."""
    ALL_MODELS = get_all_models()

    print(f"\n  Generating signals for {asset_name} "
          f"(models={'+'.join(model_names)}, window={window_size}h, "
          f"replay={replay_hours}h)...")

    df_raw = load_data(asset_name)
    if df_raw is None:
        return []

    # Build V2 features, use optimal subset
    df_v2, _all_cols = build_features_v2_hourly(df_raw, original_builder=build_hourly_features)
    use_cols = [c for c in feature_cols if c in df_v2.columns]
    missing = [c for c in feature_cols if c not in df_v2.columns]
    if missing:
        print(f"    WARNING: Missing features: {missing}")

    df_features = df_v2.dropna(subset=use_cols + ['label']).reset_index(drop=True)

    n = len(df_features)
    start_idx = max(window_size + 50, n - replay_hours)

    signals = []
    count = 0

    for i in range(start_idx, n):
        row = df_features.iloc[i]
        dt_str = row['datetime'].strftime('%Y-%m-%d %H:%M')

        train_start = max(0, i - window_size)
        train = df_features.iloc[train_start:i]
        X_train = train[use_cols]
        y_train = train['label'].values
        X_test = df_features.iloc[i:i+1][use_cols]

        if len(np.unique(y_train)) < 2:
            continue
        if X_train.isnull().any().any() or X_test.isnull().any().any():
            continue

        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=use_cols, index=X_train.index)
        X_test_s  = pd.DataFrame(scaler.transform(X_test), columns=use_cols, index=X_test.index)

        votes, probas = [], []
        for model_name in model_names:
            try:
                model = ALL_MODELS[model_name]()
                model.fit(X_train_s, y_train)
                votes.append(model.predict(X_test_s)[0])
                probas.append(model.predict_proba(X_test_s)[0][1])
            except Exception:
                continue

        if not votes:
            continue

        buy_votes = sum(votes)
        total_votes = len(votes)
        buy_ratio = buy_votes / total_votes
        avg_proba = np.mean(probas)

        # Signal classification
        if buy_ratio > 0.5:
            signal = 'BUY'
            confidence = avg_proba * 100
        elif buy_ratio == 0:
            signal = 'SELL'
            confidence = (1 - avg_proba) * 100
        else:
            signal = 'HOLD'
            confidence = max(avg_proba, 1 - avg_proba) * 100

        # Check actual outcome
        actual = None
        if i + prediction_horizon < n:
            future_close = df_features.iloc[i + prediction_horizon]['close']
            actual_return = (future_close / row['close'] - 1) * 100
            actual = 'UP' if actual_return > 0 else 'DOWN'

        signals.append({
            'datetime': dt_str,
            'close': round(float(row['close']), 2),
            'signal': signal,
            'confidence': round(float(confidence), 1),
            'buy_votes': int(buy_votes),
            'total_votes': int(total_votes),
            'avg_proba': round(float(avg_proba), 4),
            'rsi': round(float(row.get('rsi_14h', 0)), 1),
            'bb_position': round(float(row.get('bb_position_20h', 0)), 3),
            'hourly_change': round(float(row.get('logret_1h', 0) * 100), 3),
            'spread_120h_8h': round(float(row.get('spread_120h_8h', 0) * 100), 2),
            'actual': actual,
        })

        count += 1
        if count % 50 == 0:
            print(f"    [{count}] {dt_str}: {signal} ({confidence:.0f}%) "
                  f"| price={row['close']:,.2f}")

    print(f"  Generated {len(signals)} signals for {asset_name}")
    return signals


# ============================================================
# PORTFOLIO SIMULATION (with leverage)
# ============================================================
def simulate_portfolio(signals, cash, leverage, prediction_horizon):
    """
    Simulate a portfolio with leverage on hourly signals.

    BUY  -> go LONG with full leveraged position
    SELL -> go SHORT with full leveraged position
    HOLD -> maintain current position

    Leverage amplifies both gains AND losses.
    Liquidation if equity drops to 0.

    Returns list of portfolio snapshots.
    """
    if not signals:
        return []

    equity = cash
    position = 'flat'       # flat / long / short
    entry_price = None
    start_price = signals[0]['close']
    trades = 0
    wins = 0
    max_equity = cash
    max_drawdown = 0
    liquidated = False

    results = []

    for sig in signals:
        price = sig['close']
        hold_value = cash * (price / start_price)

        if liquidated:
            results.append({
                **sig,
                'equity': 0,
                'hold_value': round(hold_value, 2),
                'position': 'LIQUIDATED',
                'pnl_pct': -100.0,
                'drawdown_pct': -100.0,
            })
            continue

        # Update unrealized P&L
        if position == 'long' and entry_price:
            price_change_pct = (price / entry_price - 1)
            unrealized_pnl = equity_at_entry * leverage * price_change_pct
            equity = equity_at_entry + unrealized_pnl
        elif position == 'short' and entry_price:
            price_change_pct = (entry_price / price - 1)
            unrealized_pnl = equity_at_entry * leverage * price_change_pct
            equity = equity_at_entry + unrealized_pnl

        # Check liquidation
        if equity <= 0:
            equity = 0
            liquidated = True
            position = 'flat'
            results.append({
                **sig,
                'equity': 0,
                'hold_value': round(hold_value, 2),
                'position': 'LIQUIDATED',
                'pnl_pct': -100.0,
                'drawdown_pct': -100.0,
            })
            continue

        # Signal actions
        new_signal = sig['signal']

        if new_signal == 'BUY':
            if position == 'short':
                # Close short, realize P&L
                trades += 1
                if equity > equity_at_entry:
                    wins += 1
            if position != 'long':
                position = 'long'
                entry_price = price
                equity_at_entry = equity
        elif new_signal == 'SELL':
            if position == 'long':
                # Close long, realize P&L
                trades += 1
                if equity > equity_at_entry:
                    wins += 1
            if position != 'short':
                position = 'short'
                entry_price = price
                equity_at_entry = equity
        # HOLD: do nothing

        # Track drawdown
        if equity > max_equity:
            max_equity = equity
        dd = (equity / max_equity - 1) * 100 if max_equity > 0 else 0
        if dd < max_drawdown:
            max_drawdown = dd

        pnl_pct = (equity / cash - 1) * 100

        results.append({
            **sig,
            'equity': round(equity, 2),
            'hold_value': round(hold_value, 2),
            'position': position.upper(),
            'pnl_pct': round(pnl_pct, 2),
            'drawdown_pct': round(dd, 2),
        })

    # Summary stats
    final_equity = results[-1]['equity'] if results else cash
    total_return = (final_equity / cash - 1) * 100
    hold_return = (results[-1]['hold_value'] / cash - 1) * 100 if results else 0
    win_rate = (wins / trades * 100) if trades > 0 else 0

    summary = {
        'starting_cash': cash,
        'leverage': leverage,
        'final_equity': round(final_equity, 2),
        'total_return_pct': round(total_return, 2),
        'hold_return_pct': round(hold_return, 2),
        'alpha_pct': round(total_return - hold_return, 2),
        'max_drawdown_pct': round(max_drawdown, 2),
        'trades': trades,
        'win_rate_pct': round(win_rate, 1),
        'liquidated': liquidated,
    }

    return results, summary


# ============================================================
# HTML DASHBOARD GENERATION
# ============================================================
def generate_dashboard(all_results, cash, leverage_list, config):
    """Generate self-contained HTML dashboard."""

    generated_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pred_h = config['prediction_horizon']
    n_features = config['n_features']

    # Build JSON data for JS
    chart_data = {
        'generated': generated_ts,
        'prediction_horizon': pred_h,
        'n_features': n_features,
        'cash': cash,
        'leverage_list': leverage_list,
        'assets': {},
    }

    for asset_name, lev_data in all_results.items():
        chart_data['assets'][asset_name] = {}
        for lev, (results, summary) in lev_data.items():
            chart_data['assets'][asset_name][str(lev)] = {
                'signals': results,
                'summary': summary,
            }

    data_json = json.dumps(chart_data)

    lev_labels = ', '.join(f'1:{l}' for l in leverage_list)

    # ---- Build the HTML ----
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Signal Backtest Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #0a0e17; --card: #111827; --card-h: #1a2332; --border: #1e293b;
  --t1: #e2e8f0; --t2: #94a3b8; --t3: #64748b;
  --green: #22c55e; --red: #ef4444; --blue: #3b82f6; --amber: #f59e0b;
  --cyan: #06b6d4; --purple: #a78bfa; --pink: #f472b6;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--t1); font-family:'DM Sans',sans-serif; min-height:100vh; }}
.container {{ max-width:1440px; margin:0 auto; padding:24px 20px; }}
.header {{
  display:flex; justify-content:space-between; align-items:flex-start;
  margin-bottom:24px; padding-bottom:18px; border-bottom:1px solid var(--border);
}}
.header h1 {{
  font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:600;
  letter-spacing:-0.3px;
}}
.header h1 span {{ color:var(--blue); }}
.header-meta {{
  font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--t3);
  text-align:right; line-height:1.8;
}}
.header-meta .dot {{
  display:inline-block; width:6px; height:6px; background:var(--green);
  border-radius:50%; margin-right:4px; animation:pulse 2s infinite;
}}
@keyframes pulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.3;}} }}

/* Asset tabs */
.asset-tabs {{
  display:flex; gap:2px; margin-bottom:20px; background:var(--card);
  border-radius:6px; padding:3px; border:1px solid var(--border); width:fit-content;
}}
.asset-tab {{
  font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:500; padding:8px 18px;
  background:none; border:none; color:var(--t3); cursor:pointer; border-radius:4px; transition:all 0.2s;
}}
.asset-tab:hover {{ color:var(--t2); }}
.asset-tab.active {{ background:var(--blue); color:#fff; }}

/* Summary cards */
.summary-grid {{
  display:grid; grid-template-columns:repeat(auto-fill, minmax(200px, 1fr));
  gap:12px; margin-bottom:24px;
}}
.sum-card {{
  background:var(--card); border:1px solid var(--border); border-radius:8px;
  padding:14px 16px;
}}
.sum-card .lbl {{
  font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--t3);
  text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px;
}}
.sum-card .val {{
  font-family:'JetBrains Mono',monospace; font-size:18px; font-weight:600;
}}
.sum-card .sub {{
  font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--t3); margin-top:4px;
}}
.pos {{ color:var(--green); }} .neg {{ color:var(--red); }}

/* Charts */
.chart-panel {{
  background:var(--card); border:1px solid var(--border); border-radius:8px;
  padding:20px; margin-bottom:16px;
}}
.chart-panel h3 {{
  font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:500;
  color:var(--t2); margin-bottom:14px; letter-spacing:0.5px;
}}
.chart-wrap {{ position:relative; width:100%; }}
.chart-wrap.big {{ height:380px; }}
.chart-wrap.med {{ height:220px; }}
.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }}

/* Table */
.tbl-wrap {{
  background:var(--card); border:1px solid var(--border); border-radius:8px;
  padding:16px 18px; max-height:450px; overflow-y:auto; margin-bottom:16px;
}}
.tbl-wrap h3 {{
  font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:500;
  color:var(--t2); margin-bottom:12px;
}}
table {{ width:100%; border-collapse:collapse; }}
th {{
  font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:500;
  color:var(--t3); text-align:left; padding:6px 8px;
  border-bottom:1px solid var(--border); position:sticky; top:0;
  background:var(--card); text-transform:uppercase; letter-spacing:0.5px;
}}
td {{
  font-family:'JetBrains Mono',monospace; font-size:11px; padding:5px 8px;
  border-bottom:1px solid rgba(30,41,59,0.5); color:var(--t2);
}}
tr:hover td {{ background:rgba(59,130,246,0.03); }}
.correct {{ background:rgba(34,197,94,0.06); }}
.wrong {{ background:rgba(239,68,68,0.06); }}

::-webkit-scrollbar {{ width:5px; }}
::-webkit-scrollbar-track {{ background:var(--card); }}
::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:3px; }}

@media (max-width:900px) {{
  .summary-grid {{ grid-template-columns:1fr 1fr; }}
  .two-col {{ grid-template-columns:1fr; }}
}}
.zoom-toolbar {{
  display:flex; gap:6px; align-items:center; margin-bottom:10px;
}}
.zoom-toolbar .hint {{
  font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--t3);
}}
.btn-reset {{
  font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:500;
  background:var(--border); color:var(--t2); border:1px solid var(--t3);
  border-radius:4px; padding:4px 10px; cursor:pointer; transition:all 0.2s;
}}
.btn-reset:hover {{ background:var(--blue); color:#fff; border-color:var(--blue); }}
.view-btn {{
  font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:600;
  background:var(--border); color:var(--t2); border:1px solid var(--t3);
  border-radius:4px; padding:5px 14px; cursor:pointer; transition:all 0.2s;
}}
.view-btn:hover {{ border-color:var(--blue); color:var(--blue); }}
.view-btn.active {{ background:var(--blue); color:#fff; border-color:var(--blue); }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1><span>//</span> SIGNAL BACKTEST</h1>
      <div style="font-size:12px;color:var(--t3);margin-top:4px;">
        {pred_h}h prediction &middot; {n_features} V2 features &middot; Last month + week view &middot; Leverage: {lev_labels}
      </div>
    </div>
    <div class="header-meta">
      <div><span class="dot"></span>Generated {generated_ts}</div>
      <div>Starting cash: {cash:,.0f}</div>
    </div>
  </div>
  <div class="asset-tabs" id="assetTabs"></div>
  <div id="summaryGrid" class="summary-grid"></div>
  <div id="charts"></div>
</div>

<script>
const D = {data_json};
const LEV_COLORS = {{1:'#3b82f6', 5:'#22c55e', 10:'#f59e0b', 20:'#ef4444'}};
const LEV_WIDTHS = {{1:2.5, 5:2, 10:2, 20:1.5}};
const LEV_DASH   = {{1:[], 5:[6,3], 10:[2,2], 20:[8,4,2,4]}};
const LEV_LABELS = {{1:'1:1', 5:'1:5', 10:'1:10', 20:'1:20'}};

let activeAsset = Object.keys(D.assets)[0];
let charts = {{}};

// ---- Asset Tabs ----
function renderAssetTabs() {{
  const el = document.getElementById('assetTabs');
  el.innerHTML = '';
  for (const name of Object.keys(D.assets)) {{
    const btn = document.createElement('button');
    btn.className = 'asset-tab' + (name === activeAsset ? ' active' : '');
    btn.textContent = name;
    btn.onclick = () => {{ activeAsset = name; renderAll(); }};
    el.appendChild(btn);
  }}
}}

// ---- Summary Cards ----
function computeStats(sigs, cash) {{
  if (!sigs.length) return null;
  const first = sigs[0], last = sigs[sigs.length - 1];
  // Rebase: treat first equity as if it were 'cash'
  const ratio = cash / first.equity;
  const finalEq = last.equity * ratio;
  const ret = ((finalEq / cash) - 1) * 100;
  const holdRatio = cash / first.hold_value;
  const finalHold = last.hold_value * holdRatio;
  const holdRet = ((finalHold / cash) - 1) * 100;
  let maxEq = first.equity, maxDD = 0;
  sigs.forEach(s => {{
    if (s.equity > maxEq) maxEq = s.equity;
    const dd = (s.equity / maxEq - 1) * 100;
    if (dd < maxDD) maxDD = dd;
  }});
  const trades = sigs.filter(s => s.signal !== 'HOLD').length;
  const wins = sigs.filter(s => (s.signal==='BUY' && s.actual==='UP') || (s.signal==='SELL' && s.actual==='DOWN')).length;
  const acted = sigs.filter(s => s.signal !== 'HOLD' && s.actual).length;
  const winRate = acted > 0 ? (wins / acted * 100) : 0;
  return {{ ret, holdRet, alpha: ret - holdRet, maxDD, finalEq, trades, winRate }};
}}

function renderSummary() {{
  const el = document.getElementById('summaryGrid');
  el.innerHTML = '';
  const levData = D.assets[activeAsset];
  const viewLabel = currentView === 'week' ? 'Week' : 'Month';

  for (const lev of D.leverage_list) {{
    const allSigs = levData[String(lev)].signals;
    const [startIdx, sigs] = getViewSlice(allSigs);
    const stats = computeStats(sigs, D.cash);
    if (!stats) continue;

    const rSign = stats.ret >= 0 ? '+' : '';
    const rClass = stats.ret >= 0 ? 'pos' : 'neg';
    el.innerHTML += `
      <div class="sum-card">
        <div class="lbl">1:${{lev}} ${{viewLabel}} Return</div>
        <div class="val ${{rClass}}">${{rSign}}${{stats.ret.toFixed(1)}}%</div>
        <div class="sub">${{stats.finalEq.toLocaleString('en',{{minimumFractionDigits:0}})}}</div>
      </div>`;

    const aSign = stats.alpha >= 0 ? '+' : '';
    const aClass = stats.alpha >= 0 ? 'pos' : 'neg';
    el.innerHTML += `
      <div class="sum-card">
        <div class="lbl">1:${{lev}} Alpha</div>
        <div class="val ${{aClass}}">${{aSign}}${{stats.alpha.toFixed(1)}}%</div>
        <div class="sub">vs B&H ${{stats.holdRet >= 0 ? '+' : ''}}${{stats.holdRet.toFixed(1)}}%</div>
      </div>`;

    el.innerHTML += `
      <div class="sum-card">
        <div class="lbl">1:${{lev}} Max DD</div>
        <div class="val neg">${{stats.maxDD.toFixed(1)}}%</div>
        <div class="sub">${{stats.trades}} trades | ${{stats.winRate.toFixed(0)}}% win</div>
      </div>`;
  }}
}}

// ---- Charts ----
function destroyCharts() {{
  Object.values(charts).forEach(c => c.destroy());
  charts = {{}};
}}

const zoomPlugin = {{
  zoom: {{
    wheel: {{ enabled: true, modifierKey: null }},
    pinch: {{ enabled: true }},
    drag: {{ enabled: true, backgroundColor: 'rgba(59,130,246,0.1)', borderColor: '#3b82f6', borderWidth: 1 }},
    mode: 'x',
  }},
  pan: {{
    enabled: true,
    mode: 'x',
    modifierKey: 'shift',
  }},
}};

const chartBase = {{
  responsive:true, maintainAspectRatio:false,
  plugins: {{
    legend: {{
      display:true, position:'top',
      labels: {{
        color:'#94a3b8', font:{{ family:'JetBrains Mono',size:11 }},
        boxWidth:14, boxHeight:3, padding:16, usePointStyle:false,
        generateLabels: function(chart) {{
          const original = Chart.defaults.plugins.legend.labels.generateLabels(chart);
          original.forEach(label => {{
            if (label.hidden) {{
              label.fontColor = '#475569';
              label.strokeStyle = '#475569';
            }}
          }});
          return original;
        }}
      }},
      onHover: function(e) {{ e.native.target.style.cursor = 'pointer'; }},
      onLeave: function(e) {{ e.native.target.style.cursor = 'default'; }},
    }},
    tooltip: {{
      backgroundColor:'#1e293b', titleColor:'#e2e8f0', bodyColor:'#94a3b8',
      titleFont:{{ family:'JetBrains Mono',size:11 }}, bodyFont:{{ family:'JetBrains Mono',size:10 }},
      borderColor:'#334155', borderWidth:1, padding:10, cornerRadius:4,
    }},
    zoom: zoomPlugin,
  }},
  scales: {{
    x: {{ ticks:{{ color:'#475569', font:{{ family:'JetBrains Mono',size:9 }}, maxTicksLimit:14, maxRotation:0 }}, grid:{{ color:'rgba(30,41,59,0.5)' }} }},
    y: {{ ticks:{{ color:'#475569', font:{{ family:'JetBrains Mono',size:9 }} }}, grid:{{ color:'rgba(30,41,59,0.5)' }} }}
  }}
}};

function renderCharts() {{
  destroyCharts();
  const el = document.getElementById('charts');
  const levData = D.assets[activeAsset];

  // Use first leverage level for labels/price/signals
  const firstLev = String(D.leverage_list[0]);
  const allSignals = levData[firstLev].signals;
  const [startIdx, signals] = getViewSlice(allSignals);
  const labels = signals.map(s => s.datetime.substring(5));
  const viewLabel = currentView === 'week' ? 'Week' : 'Month';

  // ---- EQUITY CURVES (rebased to cash) ----
  let datasets = [];
  for (const lev of D.leverage_list) {{
    const allSigs = levData[String(lev)].signals;
    const [si, sigs] = getViewSlice(allSigs);
    const rawEq = sigs.map(s => s.equity);
    const rebased = rebaseValues(rawEq, D.cash);
    datasets.push({{
      label: 'ML 1:' + lev,
      data: rebased,
      borderColor: LEV_COLORS[lev],
      backgroundColor: LEV_COLORS[lev] + '15',
      borderDash: LEV_DASH[lev] || [],
      fill: false,
      borderWidth: LEV_WIDTHS[lev] || 2,
      pointRadius: 0,
      tension: 0.3,
    }});
  }}
  // Buy & hold (rebased)
  const rawHold = signals.map(s => s.hold_value);
  const rebasedHold = rebaseValues(rawHold, D.cash);
  datasets.push({{
    label: 'Buy & Hold',
    data: rebasedHold,
    borderColor: '#64748b',
    borderDash: [4,4],
    borderWidth: 1.5,
    pointRadius: 0,
    fill: false,
    tension: 0.3,
  }});

  // Week boundary line index (only meaningful in month view)
  const weekBoundary = currentView === 'month' ? signals.length - 40 : -1;

  el.innerHTML = `
    <div class="chart-panel">
      <h3>EQUITY CURVES — ${{activeAsset}} · ${{viewLabel}} · starting ${{D.cash.toLocaleString()}}</h3>
      <div class="zoom-toolbar">
        <button class="view-btn ${{currentView==='month'?'active':''}}" data-view="month" onclick="setView('month')">Last Month</button>
        <button class="view-btn ${{currentView==='week'?'active':''}}" data-view="week" onclick="setView('week')">Last Week</button>
        <span style="margin-left:8px"></span>
        <button class="btn-reset" onclick="resetZoom('equity')">Reset Zoom</button>
        <span class="hint">Click legend to show/hide · Drag to zoom · Scroll to zoom</span>
      </div>
      <div class="chart-wrap big"><canvas id="cEquity"></canvas></div>
    </div>
    <div class="chart-panel">
      <h3>PRICE ACTION + SIGNALS · ${{viewLabel}}</h3>
      <div class="zoom-toolbar">
        <button class="btn-reset" onclick="resetZoom('price')">Reset Zoom</button>
      </div>
      <div class="chart-wrap big"><canvas id="cPrice"></canvas></div>
    </div>
    <div class="two-col">
      <div class="chart-panel">
        <h3>DRAWDOWN · ${{viewLabel}}</h3>
        <div class="zoom-toolbar">
          <button class="btn-reset" onclick="resetZoom('dd')">Reset Zoom</button>
        </div>
        <div class="chart-wrap med"><canvas id="cDD"></canvas></div>
      </div>
      <div class="chart-panel">
        <h3>SIGNAL CONFIDENCE · ${{viewLabel}}</h3>
        <div class="zoom-toolbar">
          <button class="btn-reset" onclick="resetZoom('conf')">Reset Zoom</button>
        </div>
        <div class="chart-wrap med"><canvas id="cConf"></canvas></div>
      </div>
    </div>
    <div class="chart-panel">
      <h3>SIGNAL DISTRIBUTION · ${{viewLabel}}</h3>
      <div class="chart-wrap med"><canvas id="cDist"></canvas></div>
    </div>
    <div id="signalTable"></div>
  `;

  // Week boundary plugin - draws vertical dashed line (only in month view)
  const weekLinePlugin = {{
    id: 'weekLine',
    beforeDraw(chart) {{
      const weekIdx = chart.config._weekIdx;
      if (weekIdx === undefined || weekIdx <= 0) return;
      const {{ ctx, chartArea: {{top, bottom}}, scales: {{x}} }} = chart;
      const xPos = x.getPixelForValue(weekIdx);
      if (xPos < x.left || xPos > x.right) return;
      ctx.save();
      ctx.strokeStyle = '#f59e0b80';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(xPos, top);
      ctx.lineTo(xPos, bottom);
      ctx.stroke();
      ctx.fillStyle = '#f59e0b';
      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText('\\u2190 LAST WEEK \\u2192', xPos, top - 6);
      ctx.restore();
    }}
  }};

  // Equity chart
  const equityCfg = {{
    type:'line', data:{{ labels, datasets }},
    options:{{ ...chartBase, interaction:{{ intersect:false, mode:'index' }} }},
    plugins: [weekLinePlugin],
  }};
  equityCfg._weekIdx = weekBoundary;
  charts.equity = new Chart(document.getElementById('cEquity').getContext('2d'), equityCfg);

  // Price + signals (sliced)
  const buyPts = signals.map(s => s.signal === 'BUY' ? s.close : null);
  const sellPts = signals.map(s => s.signal === 'SELL' ? s.close : null);
  const priceCfg = {{
    type:'line',
    data:{{ labels, datasets:[
      {{ label:'Close', data:signals.map(s=>s.close), borderColor:'#a78bfa', borderWidth:1.5, pointRadius:0, tension:0.2, fill:false }},
      {{ label:'BUY', data:buyPts, borderColor:'transparent', backgroundColor:'#22c55e', pointRadius:4, pointStyle:'triangle', showLine:false }},
      {{ label:'SELL', data:sellPts, borderColor:'transparent', backgroundColor:'#ef4444', pointRadius:4, pointStyle:'rect', pointRotation:45, showLine:false }},
    ] }},
    options:{{ ...chartBase, interaction:{{ intersect:false, mode:'index' }} }},
    plugins: [weekLinePlugin],
  }};
  priceCfg._weekIdx = weekBoundary;
  charts.price = new Chart(document.getElementById('cPrice').getContext('2d'), priceCfg);

  // Drawdown (sliced, recomputed from slice start)
  let ddDatasets = [];
  for (const lev of D.leverage_list) {{
    const allSigs = levData[String(lev)].signals;
    const [si, sigs] = getViewSlice(allSigs);
    // Recompute drawdown from slice start
    let peak = sigs[0].equity;
    const dd = sigs.map(s => {{
      if (s.equity > peak) peak = s.equity;
      return (s.equity / peak - 1) * 100;
    }});
    ddDatasets.push({{
      label: '1:' + lev,
      data: dd,
      borderColor: LEV_COLORS[lev],
      backgroundColor: LEV_COLORS[lev] + '20',
      borderDash: LEV_DASH[lev] || [],
      fill: true,
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
    }});
  }}
  const ddCfg = {{
    type:'line', data:{{ labels, datasets:ddDatasets }},
    options:{{ ...chartBase }},
    plugins: [weekLinePlugin],
  }};
  ddCfg._weekIdx = weekBoundary;
  charts.dd = new Chart(document.getElementById('cDD').getContext('2d'), ddCfg);

  // Confidence (sliced)
  const confColors = signals.map(s => s.signal === 'BUY' ? '#22c55e60' : s.signal === 'SELL' ? '#ef444460' : '#f59e0b60');
  const confBorders = signals.map(s => s.signal === 'BUY' ? '#22c55e' : s.signal === 'SELL' ? '#ef4444' : '#f59e0b');
  charts.conf = new Chart(document.getElementById('cConf').getContext('2d'), {{
    type:'bar',
    data:{{ labels, datasets:[{{ label:'Confidence %', data:signals.map(s=>s.confidence), backgroundColor:confColors, borderColor:confBorders, borderWidth:1 }}] }},
    options:{{ ...chartBase, plugins:{{ ...chartBase.plugins, legend:{{ display:false }} }}, scales:{{ ...chartBase.scales, y:{{ ...chartBase.scales.y, min:40, max:100 }} }} }}
  }});

  // Distribution (sliced)
  const counts = {{}};
  signals.forEach(s => {{ counts[s.signal] = (counts[s.signal]||0) + 1; }});
  const distLabels = Object.keys(counts);
  const distData = Object.values(counts);
  const distColors = distLabels.map(l => l === 'BUY' ? '#22c55e40' : l === 'SELL' ? '#ef444440' : '#f59e0b40');
  const distBorders = distLabels.map(l => l === 'BUY' ? '#22c55e' : l === 'SELL' ? '#ef4444' : '#f59e0b');
  charts.dist = new Chart(document.getElementById('cDist').getContext('2d'), {{
    type:'doughnut',
    data:{{ labels:distLabels, datasets:[{{ data:distData, backgroundColor:distColors, borderColor:distBorders, borderWidth:2 }}] }},
    options:{{ responsive:true, maintainAspectRatio:false, plugins:{{ legend:{{ position:'right', labels:{{ color:'#94a3b8', font:{{ family:'JetBrains Mono',size:11 }}, padding:12 }} }} }}, cutout:'65%' }}
  }});

  // Signal table
  renderTable(signals);
}}

function renderTable(signals) {{
  const el = document.getElementById('signalTable');
  const recent = [...signals].reverse().slice(0, 60);
  let rows = recent.map(s => {{
    const sigClass = s.signal === 'BUY' ? 'color:var(--green);font-weight:600' :
                     s.signal === 'SELL' ? 'color:var(--red);font-weight:600' : 'color:var(--amber)';
    const actClass = s.actual === 'UP' ? 'color:var(--green)' : s.actual === 'DOWN' ? 'color:var(--red)' : '';
    const isCorrect = (s.signal === 'BUY' && s.actual === 'UP') || (s.signal === 'SELL' && s.actual === 'DOWN');
    const isWrong = s.actual && !isCorrect && s.signal !== 'HOLD';
    const rowClass = isCorrect ? 'correct' : isWrong ? 'wrong' : '';
    return `<tr class="${{rowClass}}">
      <td>${{s.datetime}}</td>
      <td style="text-align:right">${{s.close.toLocaleString('en',{{minimumFractionDigits:2}})}}</td>
      <td style="${{sigClass}}">${{s.signal}}</td>
      <td style="text-align:right">${{s.confidence}}%</td>
      <td style="text-align:center">${{s.buy_votes}}/${{s.total_votes}}</td>
      <td style="${{actClass}};text-align:center">${{s.actual||'-'}}</td>
      <td style="text-align:right">${{s.equity.toLocaleString('en',{{minimumFractionDigits:2}})}}</td>
      <td style="text-align:right;color:${{s.pnl_pct >= 0 ? 'var(--green)' : 'var(--red)'}}">${{s.pnl_pct >= 0 ? '+' : ''}}${{s.pnl_pct.toFixed(1)}}%</td>
    </tr>`;
  }}).join('');

  el.innerHTML = `
    <div class="tbl-wrap">
      <h3>SIGNAL LOG — ${{activeAsset}} (last 60)</h3>
      <table><thead><tr>
        <th>Datetime</th><th style="text-align:right">Close</th><th>Signal</th>
        <th style="text-align:right">Conf</th><th style="text-align:center">Votes</th>
        <th style="text-align:center">Actual</th>
        <th style="text-align:right">Equity</th><th style="text-align:right">P&L</th>
      </tr></thead><tbody>${{rows}}</tbody></table>
    </div>`;
}}

let currentView = 'month';  // 'month' or 'week'

function setView(view) {{
  currentView = view;
  document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll(`.view-btn[data-view="${{view}}"]`).forEach(b => b.classList.add('active'));
  renderSummary();
  renderCharts();
}}

function resetZoom(chartKey) {{
  const chart = charts[chartKey];
  if (chart) chart.resetZoom();
}}

function getViewSlice(signals) {{
  // Returns [startIdx, slicedSignals] based on currentView
  if (currentView === 'week') {{
    const startIdx = Math.max(0, signals.length - 40);
    return [startIdx, signals.slice(startIdx)];
  }}
  return [0, signals];
}}

function rebaseValues(values, targetStart) {{
  // Rebase array so first value = targetStart, rest scaled proportionally
  if (!values.length) return values;
  const ratio = targetStart / values[0];
  return values.map(v => v * ratio);
}}

function renderAll() {{ renderAssetTabs(); renderSummary(); renderCharts(); }}
renderAll();
</script>
</body>
</html>'''

    # Write dashboard
    os.makedirs('output/dashboards', exist_ok=True)
    filepath = 'output/dashboards/backtest_dashboard.html'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)

    return filepath


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  GENERATE SIGNALS + BACKTEST")
    print("=" * 60)

    # Load config from daily_setup
    config = load_setup_config()
    features = config['optimal_features']
    pred_h = config['prediction_horizon']
    df_best = config['best_models_df']

    print(f"\n  Config loaded (generated {config['generated']})")
    print(f"  Prediction: {pred_h}h ahead")
    print(f"  Features: {config['n_features']} optimal V2")
    print(f"\n  Best models:")
    for _, row in df_best.iterrows():
        print(f"    {row['coin']:6s} | w={row['best_window']:4d}h | {row['best_combo']:20s} | {row['accuracy']:.1f}%")

    # User input
    cash, leverage_list, replay_hours = get_user_input()

    # Generate signals per asset
    print("\n" + "=" * 60)
    print("  GENERATING SIGNALS")
    print("=" * 60)

    all_results = {}

    for _, row in df_best.iterrows():
        asset_name = row['coin']
        model_names = row['models'].split('+')
        window = int(row['best_window'])

        signals = generate_signals(
            asset_name, model_names, window, features,
            replay_hours, pred_h,
        )

        if not signals:
            print(f"  WARNING: No signals for {asset_name}")
            continue

        # Simulate for each leverage level
        all_results[asset_name] = {}
        for lev in leverage_list:
            results, summary = simulate_portfolio(signals, cash, lev, pred_h)
            all_results[asset_name][lev] = (results, summary)

            sign = '+' if summary['total_return_pct'] >= 0 else ''
            liq = ' ** LIQUIDATED **' if summary['liquidated'] else ''
            print(f"    {asset_name} 1:{lev:2d} | "
                  f"Return: {sign}{summary['total_return_pct']:.1f}% | "
                  f"DD: {summary['max_drawdown_pct']:.1f}% | "
                  f"Trades: {summary['trades']} | "
                  f"Win: {summary['win_rate_pct']:.0f}%{liq}")

    if not all_results:
        print("\nNo results to display.")
        return

    # Generate dashboard
    print("\n" + "=" * 60)
    print("  GENERATING DASHBOARD")
    print("=" * 60)

    filepath = generate_dashboard(all_results, cash, leverage_list, config)
    print(f"\n  Dashboard saved: {filepath}")
    print(f"  Open in browser to view.")

    # Print final summary
    print(f"\n{'='*60}")
    print("  BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Cash: {cash:,.0f}")
    for asset_name, lev_data in all_results.items():
        print(f"\n  {asset_name}:")
        for lev, (results, summary) in lev_data.items():
            sign = '+' if summary['total_return_pct'] >= 0 else ''
            a_sign = '+' if summary['alpha_pct'] >= 0 else ''
            print(f"    1:{lev:2d} | Final: {summary['final_equity']:>10,.0f} | "
                  f"Return: {sign}{summary['total_return_pct']:>6.1f}% | "
                  f"Alpha: {a_sign}{summary['alpha_pct']:>6.1f}% | "
                  f"MaxDD: {summary['max_drawdown_pct']:>6.1f}%")

    print(f"\n{'='*60}")
    print(f"  DONE — open {filepath}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
