"""
Leveraged CFD Backtest
========================
Simulates leveraged CFD trading on index signals.
Uses the hourly ML trading system's signals with real margin mechanics.

Usage:
  python leveraged_backtest.py
"""

import os, sys, json, warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import numpy as np
from datetime import datetime

# Import from hourly system
from hourly_trading_system import (
    load_data, build_hourly_features, ALL_MODELS, PREDICTION_HORIZON,
    classify_signal_v2, update_all_data
)
from sklearn.preprocessing import StandardScaler
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================
ASSETS_CONFIG = {
    'DAX': {
        'starting_capital': 10000,  # CHF
        'currency': 'CHF',
        'margin_rate': 0.05,        # 5% margin = 20:1 leverage
        'stop_loss_pct': 2.0,       # 2% stop on notional
    },
}

REPLAY_HOURS = 200  # last 200 hourly signals


# ============================================================
# SIGNAL GENERATION (reuses hourly system)
# ============================================================
def generate_signals_for_asset(asset_name, model_names, window_size, replay_hours):
    """Generate V2 signals using the hourly system's logic."""
    from hourly_trading_system import load_data, build_hourly_features

    df_raw = load_data(asset_name)
    if df_raw is None:
        print(f"  ERROR: No data for {asset_name}")
        return []

    df_features, feature_cols = build_hourly_features(df_raw)
    n = len(df_features)
    start_idx = max(window_size + 50, n - replay_hours)

    signals = []
    for i in range(start_idx, n):
        row = df_features.iloc[i]
        dt_str = row['datetime'].strftime('%Y-%m-%d %H:%M')

        train_start = max(0, i - window_size)
        train = df_features.iloc[train_start:i]
        X_train = train[feature_cols]
        y_train = train['label'].values
        X_test = df_features.iloc[i:i+1][feature_cols]

        if len(np.unique(y_train)) < 2:
            continue
        if X_train.isnull().any().any() or X_test.isnull().any().any():
            continue

        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train),
                                 columns=feature_cols, index=X_train.index)
        X_test_s = pd.DataFrame(scaler.transform(X_test),
                                columns=feature_cols, index=X_test.index)

        votes = []
        probas = []
        for mn in model_names:
            try:
                model = ALL_MODELS[mn]()
                model.fit(X_train_s, y_train)
                pred = model.predict(X_test_s)[0]
                proba = model.predict_proba(X_test_s)[0]
                votes.append(pred)
                probas.append(proba[1])
            except:
                pass

        if not votes:
            continue

        buy_votes = sum(votes)
        total_votes = len(votes)
        avg_proba = np.mean(probas)
        signal, confidence = classify_signal_v2(buy_votes, total_votes, avg_proba)

        # Actual outcome
        actual = None
        if i + PREDICTION_HORIZON < n:
            future_close = df_features.iloc[i + PREDICTION_HORIZON]['close']
            actual_return = (future_close / row['close'] - 1) * 100
            actual = 'UP' if actual_return > 0 else 'DOWN'

        signals.append({
            'datetime': dt_str,
            'close': round(float(row['close']), 2),
            'signal': signal,
            'confidence': round(float(confidence), 1),
            'actual': actual,
        })

        if len(signals) % 50 == 0:
            print(f"    [{len(signals)}] {dt_str}: {signal} ({confidence:.0f}%) | price={row['close']:,.2f}")

    return signals


# ============================================================
# LEVERAGED PORTFOLIO SIMULATIONS
# ============================================================
def simulate_leveraged(signals, capital, margin_rate, stop_loss_pct):
    """
    Simulates leveraged CFD trading.

    When BUY:
      - units = floor(capital / (price * margin_rate))
      - margin_used = units * price * margin_rate
      - exposure = units * price

    P&L is on full exposure, but capital at risk is the margin.
    Stop-loss is on notional (e.g., 2% of exposure).

    Returns 4 portfolio series:
      - v1_leveraged: all-in/all-out with leverage
      - v2_leveraged: graduated (50%/100%) with leverage
      - hold_leveraged: buy & hold with leverage (buy once, hold)
      - hold_unleveraged: buy & hold without leverage (reference)
    """
    if not signals:
        return signals

    start_price = signals[0]['close']
    leverage = 1 / margin_rate

    # -- V1 Leveraged (all-in / all-out) --
    v1_capital = capital
    v1_units = 0
    v1_entry = None
    v1_stopped = False

    # -- V2 Leveraged (graduated) --
    v2_capital = capital
    v2_units = 0
    v2_alloc = 0.0  # 0, 0.5, or 1.0
    v2_entry = None

    # -- Buy & Hold Leveraged --
    hold_lev_units = int(capital / (start_price * margin_rate))
    hold_lev_margin = hold_lev_units * start_price * margin_rate
    hold_lev_cash = capital - hold_lev_margin
    hold_lev_entry = start_price

    # -- Buy & Hold Unleveraged --
    # Just tracks index % change applied to capital
    hold_unlev = capital

    for sig in signals:
        price = sig['close']
        signal = sig['signal']

        # === Buy & Hold Unleveraged ===
        hold_unlev = capital * (price / start_price)

        # === Buy & Hold Leveraged ===
        hold_lev_pnl = hold_lev_units * (price - hold_lev_entry)
        hold_lev_value = hold_lev_cash + hold_lev_margin + hold_lev_pnl

        # === V1 Leveraged: all-in / all-out ===
        v1_signal = 'BUY' if signal in ('STRONG BUY', 'BUY') else \
                    'SELL' if signal in ('STRONG SELL', 'SELL') else 'HOLD'

        if v1_signal == 'BUY' and v1_units == 0:
            v1_units = int(v1_capital / (price * margin_rate))
            v1_entry = price
            v1_stopped = False
        elif v1_signal == 'SELL' and v1_units > 0:
            pnl = v1_units * (price - v1_entry)
            v1_capital += pnl
            v1_units = 0
            v1_entry = None
        elif v1_units > 0 and v1_entry:
            # Check stop-loss
            loss_pct = (price - v1_entry) / v1_entry * 100
            if loss_pct <= -stop_loss_pct:
                pnl = v1_units * (price - v1_entry)
                v1_capital += pnl
                v1_units = 0
                v1_entry = None
                v1_stopped = True

        if v1_units > 0 and v1_entry:
            v1_value = v1_capital + v1_units * (price - v1_entry)
        else:
            v1_value = v1_capital

        # === V2 Leveraged: graduated ===
        if signal == 'STRONG BUY':
            target = 1.0
        elif signal == 'BUY':
            target = 0.5
        elif signal == 'HOLD':
            target = v2_alloc
        else:
            target = 0.0

        if target != v2_alloc:
            # Close current position P&L
            if v2_units > 0 and v2_entry:
                pnl = v2_units * (price - v2_entry)
                v2_capital += pnl
                v2_units = 0
                v2_entry = None

            # Open new position at target allocation
            if target > 0:
                budget_for_trade = v2_capital * target
                v2_units = int(budget_for_trade / (price * margin_rate))
                v2_entry = price
            v2_alloc = target
        elif v2_units > 0 and v2_entry:
            # Check stop-loss
            loss_pct = (price - v2_entry) / v2_entry * 100
            if loss_pct <= -stop_loss_pct:
                pnl = v2_units * (price - v2_entry)
                v2_capital += pnl
                v2_units = 0
                v2_entry = None
                v2_alloc = 0.0

        if v2_units > 0 and v2_entry:
            v2_value = v2_capital + v2_units * (price - v2_entry)
        else:
            v2_value = v2_capital

        # Store all
        sig['v1_leveraged'] = round(v1_value, 2)
        sig['v2_leveraged'] = round(v2_value, 2)
        sig['hold_leveraged'] = round(hold_lev_value, 2)
        sig['hold_unleveraged'] = round(hold_unlev, 2)
        sig['v2_allocation'] = v2_alloc

    return signals


# ============================================================
# HTML DASHBOARD
# ============================================================
def export_leveraged_dashboard(asset_name, signals, config):
    """Generate self-contained HTML dashboard for leveraged backtest."""
    import json

    capital = config['starting_capital']
    currency = config['currency']
    margin_rate = config['margin_rate']
    leverage = int(1 / margin_rate)

    last = signals[-1]
    data_json = json.dumps(signals)
    generated_ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Calculate summary stats
    v1_pnl = last['v1_leveraged'] - capital
    v2_pnl = last['v2_leveraged'] - capital
    hold_lev_pnl = last['hold_leveraged'] - capital
    hold_unlev_pnl = last['hold_unleveraged'] - capital

    v1_pct = v1_pnl / capital * 100
    v2_pct = v2_pnl / capital * 100
    hold_lev_pct = hold_lev_pnl / capital * 100
    hold_unlev_pct = hold_unlev_pnl / capital * 100

    # Count trades
    v1_trades = 0
    v2_trades = 0
    prev_v1 = 'HOLD'
    prev_v2 = 0.0
    for s in signals:
        sig = s['signal']
        v1_sig = 'BUY' if sig in ('STRONG BUY', 'BUY') else 'SELL' if sig in ('STRONG SELL', 'SELL') else 'HOLD'
        if v1_sig == 'BUY' and prev_v1 != 'BUY':
            v1_trades += 1
        prev_v1 = v1_sig

        alloc = s.get('v2_allocation', 0)
        if alloc > 0 and prev_v2 == 0:
            v2_trades += 1
        prev_v2 = alloc

    # Accuracy
    with_actual = [s for s in signals if s['actual'] in ('UP', 'DOWN')]
    correct = [s for s in with_actual if
               (s['signal'] in ('BUY', 'STRONG BUY') and s['actual'] == 'UP') or
               (s['signal'] in ('SELL', 'STRONG SELL') and s['actual'] == 'DOWN')]
    acc = (len(correct) / len(with_actual) * 100) if with_actual else 0

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{asset_name} Leveraged CFD Backtest</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg: #0a0a0f; --surface: #12121a; --border: #1e1e2e;
  --text: #e2e8f0; --text-muted: #64748b; --text-secondary: #94a3b8;
  --green: #22c55e; --red: #ef4444; --blue: #3b82f6; --purple: #a78bfa;
  --cyan: #06b6d4; --orange: #f97316; --yellow: #eab308;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'SF Mono','Fira Code',monospace; font-size:13px; padding:20px; }}
.container {{ max-width:1400px; margin:0 auto; }}
.header {{ display:flex; justify-content:space-between; align-items:flex-end; border-bottom:1px solid var(--border); padding-bottom:16px; margin-bottom:20px; }}
.header h1 {{ font-size:20px; font-weight:600; }} .header h1 span {{ color:var(--blue); }}
.header-meta {{ text-align:right; font-size:11px; color:var(--text-muted); }}
.cards {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-bottom:20px; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px; }}
.card-label {{ font-size:10px; text-transform:uppercase; color:var(--text-muted); letter-spacing:1px; margin-bottom:6px; }}
.card-value {{ font-size:22px; font-weight:700; }}
.card-sub {{ font-size:11px; color:var(--text-muted); margin-top:4px; }}
.positive {{ color:var(--green); }} .negative {{ color:var(--red); }}
.chart-panel {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:16px; }}
.chart-panel h3 {{ font-size:13px; color:var(--text-secondary); margin-bottom:12px; text-transform:uppercase; letter-spacing:1px; }}
.chart-wrapper {{ position:relative; height:350px; }}
.chart-wrapper.small {{ height:200px; }}
.stats-grid {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-bottom:20px; }}
.stat {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:12px; text-align:center; }}
.stat-label {{ font-size:10px; text-transform:uppercase; color:var(--text-muted); letter-spacing:1px; }}
.stat-val {{ font-size:16px; font-weight:600; margin-top:4px; }}
.legend {{ display:flex; gap:20px; margin-bottom:12px; flex-wrap:wrap; }}
.legend-item {{ display:flex; align-items:center; gap:6px; font-size:11px; color:var(--text-secondary); }}
.legend-dot {{ width:12px; height:3px; border-radius:2px; }}
.signal-table {{ width:100%; border-collapse:collapse; font-size:11px; }}
.signal-table th {{ text-align:left; padding:8px; border-bottom:1px solid var(--border); color:var(--text-muted); text-transform:uppercase; font-size:10px; }}
.signal-table td {{ padding:6px 8px; border-bottom:1px solid var(--border)22; }}
.tag {{ padding:2px 6px; border-radius:3px; font-size:10px; font-weight:600; }}
.tag-buy {{ background:#22c55e22; color:var(--green); }}
.tag-sell {{ background:#ef444422; color:var(--red); }}
.tag-hold {{ background:#64748b22; color:var(--text-muted); }}
.warning {{ background:#f9731622; border:1px solid #f97316; border-radius:8px; padding:12px; margin-bottom:20px; font-size:12px; color:var(--orange); }}
</style></head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1><span>//</span> {asset_name} LEVERAGED CFD BACKTEST</h1>
      <div style="font-size:12px;color:var(--text-muted);margin-top:4px;">
        {currency} {capital:,} capital &middot; {margin_rate*100:.0f}% margin &middot; {leverage}:1 leverage &middot; {len(signals)} signals
      </div>
    </div>
    <div class="header-meta">
      <div>Generated {generated_ts}</div>
      <div>Stop-loss: {config['stop_loss_pct']}% &middot; Signal accuracy: {acc:.1f}%</div>
    </div>
  </div>

  <div class="warning">
    &#9888; LEVERAGED TRADING: {leverage}:1 leverage amplifies both gains AND losses.
    A {config['stop_loss_pct']}% index move = {config['stop_loss_pct'] * leverage:.0f}% capital impact.
    Max risk per trade: {currency} {capital * config['stop_loss_pct'] * leverage / 100:,.0f}
  </div>

  <div class="cards">
    <div class="card">
      <div class="card-label">V1 All-in/out (leveraged)</div>
      <div class="card-value {'positive' if v1_pnl >= 0 else 'negative'}">{'+' if v1_pnl >= 0 else ''}{currency} {v1_pnl:,.0f}</div>
      <div class="card-sub">{'+' if v1_pct >= 0 else ''}{v1_pct:.1f}% &middot; {v1_trades} trades</div>
    </div>
    <div class="card">
      <div class="card-label">V2 Graduated (leveraged)</div>
      <div class="card-value {'positive' if v2_pnl >= 0 else 'negative'}">{'+' if v2_pnl >= 0 else ''}{currency} {v2_pnl:,.0f}</div>
      <div class="card-sub">{'+' if v2_pct >= 0 else ''}{v2_pct:.1f}% &middot; {v2_trades} trades</div>
    </div>
    <div class="card">
      <div class="card-label">Buy & Hold (leveraged)</div>
      <div class="card-value {'positive' if hold_lev_pnl >= 0 else 'negative'}">{'+' if hold_lev_pnl >= 0 else ''}{currency} {hold_lev_pnl:,.0f}</div>
      <div class="card-sub">{'+' if hold_lev_pct >= 0 else ''}{hold_lev_pct:.1f}% &middot; {leverage}x exposure</div>
    </div>
    <div class="card">
      <div class="card-label">Buy & Hold (no leverage)</div>
      <div class="card-value" style="color:var(--text-secondary)">{'+' if hold_unlev_pnl >= 0 else ''}{currency} {hold_unlev_pnl:,.0f}</div>
      <div class="card-sub">{'+' if hold_unlev_pct >= 0 else ''}{hold_unlev_pct:.1f}% &middot; 1x exposure</div>
    </div>
  </div>

  <div class="chart-panel">
    <h3>Portfolio Value -- {currency} {capital:,} starting capital</h3>
    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:var(--blue)"></div>V1 All-in/out ({leverage}:1)</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--green)"></div>V2 Graduated ({leverage}:1)</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--purple)"></div>Hold Leveraged ({leverage}:1)</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--text-muted);border-top:1px dashed var(--text-muted)"></div>Hold Unleveraged (1:1)</div>
    </div>
    <div class="chart-wrapper"><canvas id="chartPortfolio"></canvas></div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
    <div class="chart-panel">
      <h3>Signal Distribution</h3>
      <div class="chart-wrapper small"><canvas id="chartDist"></canvas></div>
    </div>
    <div class="chart-panel">
      <h3>Confidence per Signal</h3>
      <div class="chart-wrapper small"><canvas id="chartConf"></canvas></div>
    </div>
  </div>

  <div class="chart-panel">
    <h3>Price + Signals -- {asset_name}</h3>
    <div class="chart-wrapper"><canvas id="chartPrice"></canvas></div>
  </div>

  <div class="chart-panel">
    <h3>Signal Log (last 60)</h3>
    <div style="max-height:400px;overflow-y:auto;">
      <table class="signal-table">
        <thead><tr>
          <th>Time</th><th>Price</th><th>Signal</th><th>Conf</th>
          <th>V1 Portfolio</th><th>V2 Portfolio</th><th>Actual</th>
        </tr></thead>
        <tbody id="signalLog"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const DATA = {data_json};
const CAPITAL = {capital};

const labels = DATA.map(s => s.datetime);
const chartDefaults = {{
  responsive: true, maintainAspectRatio: false,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    x: {{ ticks: {{ color:'#64748b', maxTicksLimit:12, font:{{ size:10 }} }}, grid:{{ color:'#1e1e2e' }} }},
    y: {{ ticks: {{ color:'#64748b', font:{{ size:10 }} }}, grid:{{ color:'#1e1e2e' }} }}
  }}
}};

// Portfolio chart
new Chart(document.getElementById('chartPortfolio').getContext('2d'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{ label:'V1 Leveraged', data:DATA.map(s=>s.v1_leveraged), borderColor:'#3b82f6', borderWidth:2, pointRadius:0, tension:0.3, fill:false }},
      {{ label:'V2 Leveraged', data:DATA.map(s=>s.v2_leveraged), borderColor:'#22c55e', borderWidth:2, pointRadius:0, tension:0.3, fill:false }},
      {{ label:'Hold Leveraged', data:DATA.map(s=>s.hold_leveraged), borderColor:'#a78bfa', borderWidth:1.5, pointRadius:0, tension:0.3, fill:false }},
      {{ label:'Hold Unleveraged', data:DATA.map(s=>s.hold_unleveraged), borderColor:'#64748b', borderDash:[4,4], borderWidth:1.5, pointRadius:0, tension:0.3, fill:false }},
      {{ label:'Starting Capital', data:DATA.map(()=>CAPITAL), borderColor:'#ffffff22', borderDash:[2,6], borderWidth:1, pointRadius:0, fill:false }},
    ]
  }},
  options: {{ ...chartDefaults, interaction:{{ intersect:false, mode:'index' }},
    plugins: {{ legend:{{ display:true, labels:{{ color:'#94a3b8', font:{{ size:10 }}, usePointStyle:true, pointStyle:'line' }} }} }}
  }}
}});

// Signal distribution pie
const sigCounts = {{}};
DATA.forEach(s => {{ sigCounts[s.signal] = (sigCounts[s.signal]||0)+1; }});
const pieLabels = Object.keys(sigCounts);
const pieColors = pieLabels.map(l => l.includes('BUY') ? '#22c55e' : l.includes('SELL') ? '#ef4444' : '#64748b');
new Chart(document.getElementById('chartDist').getContext('2d'), {{
  type: 'doughnut',
  data: {{ labels:pieLabels, datasets:[{{ data:Object.values(sigCounts), backgroundColor:pieColors.map(c=>c+'80'), borderColor:pieColors, borderWidth:1 }}] }},
  options: {{ responsive:true, maintainAspectRatio:false, plugins:{{ legend:{{ position:'right', labels:{{ color:'#94a3b8', font:{{ size:10 }} }} }} }} }}
}});

// Confidence chart
const confColors = DATA.map(s => s.signal.includes('BUY') ? '#22c55e60' : s.signal.includes('SELL') ? '#ef444460' : '#64748b40');
new Chart(document.getElementById('chartConf').getContext('2d'), {{
  type: 'bar',
  data: {{ labels, datasets:[{{ data:DATA.map(s=>s.confidence), backgroundColor:confColors, borderWidth:0 }}] }},
  options: {{ ...chartDefaults, plugins:{{ legend:{{ display:false }} }} }}
}});

// Price + signals chart
const buyPts = DATA.map((s,i) => (s.signal.includes('BUY') ? s.close : null));
const sellPts = DATA.map((s,i) => (s.signal.includes('SELL') ? s.close : null));
new Chart(document.getElementById('chartPrice').getContext('2d'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{ label:'Price', data:DATA.map(s=>s.close), borderColor:'#94a3b8', borderWidth:1.5, pointRadius:0, tension:0.3, fill:false }},
      {{ label:'BUY', data:buyPts, borderColor:'transparent', pointBackgroundColor:'#22c55e', pointRadius:4, pointStyle:'triangle', showLine:false }},
      {{ label:'SELL', data:sellPts, borderColor:'transparent', pointBackgroundColor:'#ef4444', pointRadius:4, pointStyle:'triangle', rotation:180, showLine:false }},
    ]
  }},
  options: {{ ...chartDefaults, interaction:{{ intersect:false, mode:'index' }} }}
}});

// Signal log table
const tbody = document.getElementById('signalLog');
const last60 = DATA.slice(-60);
last60.forEach(s => {{
  const tagClass = s.signal.includes('BUY') ? 'tag-buy' : s.signal.includes('SELL') ? 'tag-sell' : 'tag-hold';
  const actualOk = s.actual ? ((s.signal.includes('BUY') && s.actual==='UP') || (s.signal.includes('SELL') && s.actual==='DOWN')) : null;
  const actualStyle = actualOk === true ? 'color:#22c55e' : actualOk === false ? 'color:#ef4444' : 'color:#64748b';
  const row = document.createElement('tr');
  row.innerHTML =
    '<td>' + s.datetime + '</td>' +
    '<td>' + s.close.toLocaleString() + '</td>' +
    '<td><span class="tag ' + tagClass + '">' + s.signal + '</span></td>' +
    '<td>' + s.confidence + '%</td>' +
    '<td style="' + (s.v1_leveraged >= CAPITAL ? 'color:#22c55e' : 'color:#ef4444') + '">' + s.v1_leveraged.toLocaleString() + '</td>' +
    '<td style="' + (s.v2_leveraged >= CAPITAL ? 'color:#22c55e' : 'color:#ef4444') + '">' + s.v2_leveraged.toLocaleString() + '</td>' +
    '<td style="' + actualStyle + '">' + (s.actual || '-') + '</td>';
  tbody.appendChild(row);
}});
</script>
</body></html>"""

    filename = f'leveraged_backtest_{asset_name.lower()}.html'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n  Dashboard saved to {filename}")
    return filename


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  LEVERAGED CFD BACKTEST")
    print("=" * 60)

    # Load best models
    if not os.path.exists('hourly_best_models.csv'):
        print("ERROR: hourly_best_models.csv not found. Run hourly diagnostic first.")
        return

    df_best = pd.read_csv('hourly_best_models.csv')

    for asset_name, config in ASSETS_CONFIG.items():
        row = df_best[df_best['coin'] == asset_name]
        if len(row) == 0:
            print(f"  {asset_name}: No model config found, skipping")
            continue

        model_names = row.iloc[0]['models'].split('+')
        window = int(row.iloc[0]['best_window'])
        leverage = int(1 / config['margin_rate'])

        print(f"\n--- {asset_name} ---")
        print(f"  Capital: {config['currency']} {config['starting_capital']:,}")
        print(f"  Margin: {config['margin_rate']*100:.0f}% ({leverage}:1 leverage)")
        print(f"  Models: {'+'.join(model_names)}, window={window}h")
        print(f"  Generating {REPLAY_HOURS} signals...")

        # Update data first
        try:
            update_all_data([asset_name])
        except:
            pass

        signals = generate_signals_for_asset(asset_name, model_names, window, REPLAY_HOURS)
        if not signals:
            print(f"  No signals generated for {asset_name}")
            continue

        print(f"  {len(signals)} signals generated")

        # Run leveraged simulation
        signals = simulate_leveraged(
            signals,
            capital=config['starting_capital'],
            margin_rate=config['margin_rate'],
            stop_loss_pct=config['stop_loss_pct'],
        )

        last = signals[-1]
        capital = config['starting_capital']
        ccy = config['currency']
        print(f"\n  Results ({len(signals)} hours):")
        print(f"    V1 Leveraged:     {ccy} {last['v1_leveraged']:>10,.2f}  ({(last['v1_leveraged']/capital-1)*100:+.1f}%)")
        print(f"    V2 Leveraged:     {ccy} {last['v2_leveraged']:>10,.2f}  ({(last['v2_leveraged']/capital-1)*100:+.1f}%)")
        print(f"    Hold Leveraged:   {ccy} {last['hold_leveraged']:>10,.2f}  ({(last['hold_leveraged']/capital-1)*100:+.1f}%)")
        print(f"    Hold Unleveraged: {ccy} {last['hold_unleveraged']:>10,.2f}  ({(last['hold_unleveraged']/capital-1)*100:+.1f}%)")

        # Export dashboard
        export_leveraged_dashboard(asset_name, signals, config)

    print("\n" + "=" * 60)
    print("  BACKTEST COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
