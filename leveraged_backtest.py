"""
Leveraged CFD Backtest -- Unit Accumulation Strategies
======================================================
10:1 leverage on DAX, CHF 10,000 capital.
Compares two strategies that accumulate units on BUY signals:

  S1 (Cautious):    BUY/STRONG BUY -> +1 unit    | SELL/STRONG SELL -> sell all
  S2 (Aggressive):  STRONG BUY -> +2, BUY -> +1  | SELL/STRONG SELL -> sell all

Plus buy & hold (10:1) and unleveraged reference.

Usage:
  python leveraged_backtest.py
"""

import os, sys, json, warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pandas as pd

from hourly_trading_system import (
    load_data, build_hourly_features, ALL_MODELS, PREDICTION_HORIZON,
    classify_signal_v2, update_all_data
)


# ============================================================
# CONFIGURATION
# ============================================================
ASSET = 'DAX'
STARTING_CAPITAL = 10000   # CHF
CURRENCY = 'CHF'
LEVERAGE = 10              # 10:1
STOP_LOSS_PCT = 2.0        # 2% stop-loss on notional exposure

REPLAY_HOURS = 200         # last 200 hourly signals

# Strategies to compare
STRATEGIES = [
    {
        'key': 's1',
        'name': 'S1: Cautious (+1 unit)',
        'short': 'S1 Cautious',
        'color': '#3b82f6',       # blue
        'desc': 'BUY/STRONG BUY -> +1 unit, SELL -> close all',
        'strong_buy_units': 1,
        'buy_units': 1,
    },
    {
        'key': 's2',
        'name': 'S2: Aggressive (+1/+2 units)',
        'short': 'S2 Aggressive',
        'color': '#f97316',       # orange
        'desc': 'STRONG BUY -> +2 units, BUY -> +1, SELL -> close all',
        'strong_buy_units': 2,
        'buy_units': 1,
    },
]


# ============================================================
# SIGNAL GENERATION (reuses hourly system)
# ============================================================
def generate_signals(asset_name, model_names, window_size, replay_hours):
    """Generate V2 5-tier signals for an asset."""
    df_raw = load_data(asset_name)
    if df_raw is None:
        return []

    df_features, feature_cols = build_hourly_features(df_raw)
    n = len(df_features)
    start_idx = max(0, n - replay_hours)

    train_end = start_idx
    if train_end < 100:
        print(f"  Not enough training data")
        return []

    X_train = df_features.iloc[:train_end][feature_cols].values
    y_train = df_features.iloc[:train_end]['label'].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {}
    for name in model_names:
        cls = ALL_MODELS.get(name)
        if cls:
            try:
                m = cls()
                m.fit(X_train_scaled, y_train)
                models[name] = m
            except:
                pass

    signals = []
    for i in range(start_idx, n):
        row = df_features.iloc[i]
        dt_str = str(row.get('datetime', f'bar_{i}'))
        X_i = row[feature_cols].values.reshape(1, -1)
        X_i_scaled = scaler.transform(X_i)

        votes, probas = [], []
        for name, m in models.items():
            try:
                pred = m.predict(X_i_scaled)[0]
                proba = m.predict_proba(X_i_scaled)[0]
                votes.append(int(pred))
                probas.append(float(max(proba)))
            except:
                pass

        if not votes:
            continue

        buy_votes = sum(votes)
        total_votes = len(votes)
        avg_proba = np.mean(probas)
        signal, confidence = classify_signal_v2(buy_votes, total_votes, avg_proba)

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
# UNIT ACCUMULATION SIMULATION
# ============================================================
def simulate_strategies(signals, capital, leverage, stop_loss_pct, strategies):
    """
    Simulates unit-accumulation strategies at fixed leverage.

    Each strategy defines how many units to add per signal type.
    Max units constrained by available margin.
    Weighted average entry price tracked across accumulations.
    Stop-loss: if unrealised loss on total position >= stop_loss_pct of notional.
    """
    if not signals:
        return signals

    start_price = signals[0]['close']
    margin_per_unit = start_price / leverage  # CHF per unit margin

    # Max units at start (based on starting capital & starting price)
    max_units = int(capital / margin_per_unit)

    print(f"\n  Position sizing @ DAX {start_price:,.0f}:")
    print(f"    Margin/unit: {CURRENCY} {margin_per_unit:,.0f}")
    print(f"    Max units:   {max_units}")
    print(f"    Max exposure: {CURRENCY} {max_units * start_price:,.0f}")
    print(f"    2% stop on max: {CURRENCY} {max_units * start_price * stop_loss_pct / 100:,.0f} "
          f"({max_units * start_price * stop_loss_pct / 100 / capital * 100:.0f}% of capital)")

    # Buy & hold leveraged: buy max units at start
    hold_units = max_units
    hold_entry = start_price

    # State per strategy
    state = {}
    for strat in strategies:
        state[strat['key']] = {
            'capital': capital,         # realised capital (cash)
            'units': 0,                 # units currently held
            'avg_entry': 0.0,           # weighted average entry price
            'trades': 0,                # total round-trip trades (each close = 1 trade)
            'wins': 0,
            'losses': 0,
            'peak': capital,
            'max_drawdown': 0.0,
            'stopped_out': 0,
            'max_units_held': 0,
        }

    for sig in signals:
        price = sig['close']
        signal = sig['signal']

        # Unleveraged buy & hold (1:1)
        sig['hold_1x'] = round(capital * (price / start_price), 2)

        # Leveraged buy & hold (10:1)
        hold_pnl = hold_units * (price - hold_entry)
        sig['hold_10x'] = round(capital + hold_pnl, 2)

        for strat in strategies:
            key = strat['key']
            st = state[key]

            # Determine units to add based on signal
            add_units = 0
            close_position = False

            if signal == 'STRONG BUY':
                add_units = strat['strong_buy_units']
            elif signal == 'BUY':
                add_units = strat['buy_units']
            elif signal in ('SELL', 'STRONG SELL'):
                close_position = True
            # HOLD -> do nothing

            # Close entire position on SELL
            if close_position and st['units'] > 0:
                pnl = st['units'] * (price - st['avg_entry'])
                st['capital'] += pnl
                st['trades'] += 1
                if pnl >= 0:
                    st['wins'] += 1
                else:
                    st['losses'] += 1
                st['units'] = 0
                st['avg_entry'] = 0.0

            # Add units (if BUY/STRONG BUY and we have room)
            if add_units > 0 and not close_position:
                # Max units we can currently hold (based on current capital & current price)
                current_margin = price / leverage
                current_max = int(st['capital'] / current_margin)
                # But also consider margin already used by existing units
                available = current_max - st['units']
                actual_add = min(add_units, max(0, available))

                if actual_add > 0:
                    # Update weighted average entry
                    if st['units'] == 0:
                        st['avg_entry'] = price
                        st['units'] = actual_add
                    else:
                        total_cost = st['avg_entry'] * st['units'] + price * actual_add
                        st['units'] += actual_add
                        st['avg_entry'] = total_cost / st['units']

                    if st['units'] > st['max_units_held']:
                        st['max_units_held'] = st['units']

            # Check stop-loss on total position
            if st['units'] > 0 and st['avg_entry'] > 0:
                loss_pct = (price - st['avg_entry']) / st['avg_entry'] * 100
                if loss_pct <= -stop_loss_pct:
                    pnl = st['units'] * (price - st['avg_entry'])
                    st['capital'] += pnl
                    st['trades'] += 1
                    st['losses'] += 1
                    st['stopped_out'] += 1
                    st['units'] = 0
                    st['avg_entry'] = 0.0

            # Calculate current portfolio value
            if st['units'] > 0:
                unrealised = st['units'] * (price - st['avg_entry'])
                portfolio_val = st['capital'] + unrealised
            else:
                portfolio_val = st['capital']

            # Track drawdown
            if portfolio_val > st['peak']:
                st['peak'] = portfolio_val
            dd = (st['peak'] - portfolio_val) / st['peak'] * 100
            if dd > st['max_drawdown']:
                st['max_drawdown'] = dd

            sig[key] = round(portfolio_val, 2)
            sig[f'{key}_units'] = st['units']

    # Attach stats
    for strat in strategies:
        strat['stats'] = state[strat['key']]

    return signals


# ============================================================
# HTML DASHBOARD
# ============================================================
def export_dashboard(asset_name, signals, capital, currency, leverage, stop_loss_pct, strategies):
    last = signals[-1]
    start_price = signals[0]['close']
    data_json = json.dumps(signals)
    generated_ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    margin_per_unit = start_price / leverage
    max_units = int(capital / margin_per_unit)

    # Accuracy
    with_actual = [s for s in signals if s['actual'] in ('UP', 'DOWN')]
    correct = [s for s in with_actual if
               (s['signal'] in ('BUY', 'STRONG BUY') and s['actual'] == 'UP') or
               (s['signal'] in ('SELL', 'STRONG SELL') and s['actual'] == 'DOWN')]
    acc = (len(correct) / len(with_actual) * 100) if with_actual else 0

    # Strategy cards
    cards_html = ''
    for strat in strategies:
        key = strat['key']
        st = strat['stats']
        final = last[key]
        pnl = final - capital
        pct = pnl / capital * 100
        cls = 'positive' if pnl >= 0 else 'negative'
        wr = (st['wins'] / st['trades'] * 100) if st['trades'] > 0 else 0

        cards_html += f"""
    <div class="card" style="border-left:3px solid {strat['color']}">
      <div class="card-label">{strat['short']}</div>
      <div class="card-value {cls}">{'+'if pnl>=0 else ''}{currency} {pnl:,.0f} ({pct:+.1f}%)</div>
      <div class="card-sub">
        {strat['desc']}<br>
        Trades: {st['trades']} (W:{st['wins']} L:{st['losses']} WR:{wr:.0f}%)<br>
        Max units held: {st['max_units_held']} / {max_units}<br>
        Stop-outs: {st['stopped_out']}<br>
        Max drawdown: {st['max_drawdown']:.1f}%
      </div>
    </div>"""

    # Buy & hold cards
    for label, key_name, col in [
        (f'Buy & Hold {leverage}:1', 'hold_10x', '#22c55e'),
        ('Buy & Hold 1:1', 'hold_1x', '#64748b')
    ]:
        val = last[key_name]
        pnl = val - capital
        pct = pnl / capital * 100
        cls = 'positive' if pnl >= 0 else 'negative'
        extra = f'{max_units} units, no trading' if '10' in key_name else 'No leverage, no trades'
        cards_html += f"""
    <div class="card" style="border-left:3px solid {col}">
      <div class="card-label">{label}</div>
      <div class="card-value {cls}">{'+'if pnl>=0 else ''}{currency} {pnl:,.0f} ({pct:+.1f}%)</div>
      <div class="card-sub">{extra}</div>
    </div>"""

    # Build JS datasets for portfolio chart
    ds_js = ''
    for strat in strategies:
        ds_js += f"{{ label:'{strat['short']}', data:DATA.map(s=>s['{strat['key']}']), borderColor:'{strat['color']}', borderWidth:2.5, pointRadius:0, tension:0.3, fill:false }},"
    ds_js += f"{{ label:'Hold {leverage}:1', data:DATA.map(s=>s.hold_10x), borderColor:'#22c55e', borderDash:[6,3], borderWidth:1.5, pointRadius:0, tension:0.3, fill:false }},"
    ds_js += f"{{ label:'Hold 1:1', data:DATA.map(s=>s.hold_1x), borderColor:'#64748b', borderDash:[3,6], borderWidth:1, pointRadius:0, tension:0.3, fill:false }},"
    ds_js += f"{{ label:'Capital', data:DATA.map(()=>{capital}), borderColor:'#ffffff15', borderWidth:1, borderDash:[2,8], pointRadius:0, fill:false }},"

    # Units chart datasets
    units_ds = ''
    for strat in strategies:
        units_ds += f"{{ label:'{strat['short']}', data:DATA.map(s=>s['{strat['key']}_units']), borderColor:'{strat['color']}', backgroundColor:'{strat['color']}30', borderWidth:1.5, pointRadius:0, tension:0.3, fill:true }},"

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{asset_name} Strategy Comparison -- {leverage}:1</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{ --bg:#0a0a0f; --surface:#12121a; --border:#1e1e2e; --text:#e2e8f0; --text-muted:#64748b; --text-secondary:#94a3b8; --green:#22c55e; --red:#ef4444; --blue:#3b82f6; --orange:#f97316; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'SF Mono','Fira Code',monospace; font-size:13px; padding:20px; }}
.container {{ max-width:1400px; margin:0 auto; }}
.header {{ display:flex; justify-content:space-between; align-items:flex-end; border-bottom:1px solid var(--border); padding-bottom:16px; margin-bottom:20px; }}
.header h1 {{ font-size:20px; font-weight:600; }} .header h1 span {{ color:var(--blue); }}
.header-meta {{ text-align:right; font-size:11px; color:var(--text-muted); }}
.cards {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-bottom:20px; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px; }}
.card-label {{ font-size:10px; text-transform:uppercase; color:var(--text-muted); letter-spacing:1px; margin-bottom:6px; }}
.card-value {{ font-size:20px; font-weight:700; }}
.card-sub {{ font-size:11px; color:var(--text-muted); margin-top:6px; line-height:1.7; }}
.positive {{ color:var(--green); }} .negative {{ color:var(--red); }}
.chart-panel {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:16px; }}
.chart-panel h3 {{ font-size:13px; color:var(--text-secondary); margin-bottom:12px; text-transform:uppercase; letter-spacing:1px; }}
.chart-wrapper {{ position:relative; height:400px; }}
.chart-wrapper.small {{ height:240px; }}
.info-box {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:16px; font-size:12px; line-height:1.8; }}
.info-box h4 {{ font-size:11px; text-transform:uppercase; letter-spacing:1px; color:var(--text-secondary); margin-bottom:8px; }}
.info-box .row {{ display:flex; justify-content:space-between; border-bottom:1px solid var(--border); padding:4px 0; }}
.info-box .label {{ color:var(--text-muted); }}
.signal-table {{ width:100%; border-collapse:collapse; font-size:11px; }}
.signal-table th {{ text-align:left; padding:8px; border-bottom:1px solid var(--border); color:var(--text-muted); text-transform:uppercase; font-size:10px; }}
.signal-table td {{ padding:6px 8px; border-bottom:1px solid var(--border)22; }}
.tag {{ padding:2px 6px; border-radius:3px; font-size:10px; font-weight:600; }}
.tag-sbuy {{ background:#22c55e33; color:#4ade80; }}
.tag-buy {{ background:#22c55e22; color:var(--green); }}
.tag-sell {{ background:#ef444422; color:var(--red); }}
.tag-ssell {{ background:#ef444433; color:#f87171; }}
.tag-hold {{ background:#64748b22; color:var(--text-muted); }}
</style></head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1><span>//</span> {asset_name} STRATEGY COMPARISON</h1>
      <div style="font-size:12px;color:var(--text-muted);margin-top:4px;">
        {currency} {capital:,} &middot; {leverage}:1 leverage &middot; {stop_loss_pct}% stop-loss &middot; {len(signals)} signals &middot; {acc:.1f}% accuracy
      </div>
    </div>
    <div class="header-meta"><div>Generated {generated_ts}</div></div>
  </div>

  <div class="info-box">
    <h4>Position Sizing @ DAX {start_price:,.0f}</h4>
    <div class="row"><span class="label">Margin per unit</span><span>{currency} {margin_per_unit:,.0f} (10% of {start_price:,.0f})</span></div>
    <div class="row"><span class="label">Max units ({currency} {capital:,} budget)</span><span><strong>{max_units} units</strong></span></div>
    <div class="row"><span class="label">Max exposure</span><span>{currency} {max_units * start_price:,.0f}</span></div>
    <div class="row"><span class="label">1 unit x 2% stop</span><span style="color:var(--red)">-{currency} {start_price * stop_loss_pct / 100:,.0f} ({start_price * stop_loss_pct / 100 / capital * 100:.1f}% of capital)</span></div>
    <div class="row"><span class="label">{max_units} units x 2% stop</span><span style="color:var(--red)">-{currency} {max_units * start_price * stop_loss_pct / 100:,.0f} ({max_units * start_price * stop_loss_pct / 100 / capital * 100:.0f}% of capital)</span></div>
  </div>

  <div class="cards">{cards_html}</div>

  <div class="chart-panel">
    <h3>Portfolio Value -- Strategy Comparison</h3>
    <div class="chart-wrapper"><canvas id="chartMain"></canvas></div>
  </div>

  <div class="chart-panel">
    <h3>Units Held Over Time</h3>
    <div class="chart-wrapper small"><canvas id="chartUnits"></canvas></div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
    <div class="chart-panel"><h3>Signal Distribution</h3><div class="chart-wrapper small"><canvas id="chartDist"></canvas></div></div>
    <div class="chart-panel"><h3>Price + Signals</h3><div class="chart-wrapper small"><canvas id="chartPrice"></canvas></div></div>
  </div>

  <div class="chart-panel">
    <h3>Signal Log (last 80)</h3>
    <div style="max-height:500px;overflow-y:auto;">
      <table class="signal-table">
        <thead><tr><th>Time</th><th>Price</th><th>Signal</th><th>Conf</th><th>S1 Val</th><th>S1 Units</th><th>S2 Val</th><th>S2 Units</th><th>Hold 10x</th><th>Actual</th></tr></thead>
        <tbody id="signalLog"></tbody>
      </table>
    </div>
  </div>
</div>
<script>
const DATA = {data_json};
const CAPITAL = {capital};
const labels = DATA.map(s => s.datetime);
const cDef = {{ responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{ display:true, labels:{{ color:'#94a3b8', font:{{ size:10 }}, usePointStyle:true, pointStyle:'line' }} }} }},
  scales:{{ x:{{ ticks:{{ color:'#64748b', maxTicksLimit:12, font:{{ size:10 }} }}, grid:{{ color:'#1e1e2e' }} }},
            y:{{ ticks:{{ color:'#64748b', font:{{ size:10 }} }}, grid:{{ color:'#1e1e2e' }} }} }}
}};

// Portfolio value chart
new Chart(document.getElementById('chartMain').getContext('2d'), {{
  type:'line', data:{{ labels, datasets:[{ds_js}] }},
  options:{{ ...cDef, interaction:{{ intersect:false, mode:'index' }} }}
}});

// Units chart
new Chart(document.getElementById('chartUnits').getContext('2d'), {{
  type:'line', data:{{ labels, datasets:[{units_ds}] }},
  options:{{ ...cDef, interaction:{{ intersect:false, mode:'index' }},
    scales:{{ x:{{ ticks:{{ color:'#64748b', maxTicksLimit:12, font:{{ size:10 }} }}, grid:{{ color:'#1e1e2e' }} }},
              y:{{ ticks:{{ color:'#64748b', font:{{ size:10 }}, stepSize:1 }}, grid:{{ color:'#1e1e2e' }}, min:0, max:{max_units + 1} }} }} }}
}});

// Signal distribution
const sc = {{}}; DATA.forEach(s => {{ sc[s.signal]=(sc[s.signal]||0)+1; }});
const sigOrder = ['STRONG BUY','BUY','HOLD','SELL','STRONG SELL'];
const sigColors = {{'STRONG BUY':'#4ade80','BUY':'#22c55e','HOLD':'#64748b','SELL':'#ef4444','STRONG SELL':'#f87171'}};
const pl = sigOrder.filter(s => sc[s]);
new Chart(document.getElementById('chartDist').getContext('2d'), {{
  type:'doughnut',
  data:{{ labels:pl, datasets:[{{ data:pl.map(l=>sc[l]), backgroundColor:pl.map(l=>sigColors[l]+'80'), borderColor:pl.map(l=>sigColors[l]), borderWidth:1 }}] }},
  options:{{ responsive:true, maintainAspectRatio:false, plugins:{{ legend:{{ position:'right', labels:{{ color:'#94a3b8', font:{{ size:10 }} }} }} }} }}
}});

// Price + signals
const bp=DATA.map(s=>s.signal==='BUY'?s.close:null);
const sbp=DATA.map(s=>s.signal==='STRONG BUY'?s.close:null);
const sp=DATA.map(s=>s.signal==='SELL'?s.close:null);
const ssp=DATA.map(s=>s.signal==='STRONG SELL'?s.close:null);
new Chart(document.getElementById('chartPrice').getContext('2d'), {{
  type:'line', data:{{ labels, datasets:[
    {{ label:'Price', data:DATA.map(s=>s.close), borderColor:'#94a3b8', borderWidth:1.5, pointRadius:0, tension:0.3, fill:false }},
    {{ label:'STRONG BUY', data:sbp, borderColor:'transparent', pointBackgroundColor:'#4ade80', pointRadius:5, pointStyle:'triangle', showLine:false }},
    {{ label:'BUY', data:bp, borderColor:'transparent', pointBackgroundColor:'#22c55e', pointRadius:4, pointStyle:'triangle', showLine:false }},
    {{ label:'SELL', data:sp, borderColor:'transparent', pointBackgroundColor:'#ef4444', pointRadius:4, pointStyle:'triangle', rotation:180, showLine:false }},
    {{ label:'STRONG SELL', data:ssp, borderColor:'transparent', pointBackgroundColor:'#f87171', pointRadius:5, pointStyle:'triangle', rotation:180, showLine:false }},
  ] }}, options:{{ ...cDef, interaction:{{ intersect:false, mode:'index' }} }}
}});

// Signal log
const tbody=document.getElementById('signalLog');
const vc = (v) => '<td style="color:'+(v>=CAPITAL?'#22c55e':'#ef4444')+'">'+v.toLocaleString()+'</td>';
const tagMap = {{'STRONG BUY':'tag-sbuy','BUY':'tag-buy','SELL':'tag-sell','STRONG SELL':'tag-ssell','HOLD':'tag-hold'}};
DATA.slice(-80).forEach(s => {{
  const tc = tagMap[s.signal] || 'tag-hold';
  const ok = s.actual ? ((s.signal.includes('BUY')&&s.actual==='UP')||(s.signal.includes('SELL')&&s.actual==='DOWN')) : null;
  const as = ok===true ? 'color:#22c55e' : ok===false ? 'color:#ef4444' : 'color:#64748b';
  const r = document.createElement('tr');
  r.innerHTML =
    '<td>'+s.datetime+'</td><td>'+s.close.toLocaleString()+'</td>' +
    '<td><span class="tag '+tc+'">'+s.signal+'</span></td><td>'+s.confidence+'%</td>' +
    vc(s.s1)+'<td>'+s.s1_units+'</td>' +
    vc(s.s2)+'<td>'+s.s2_units+'</td>' +
    vc(s.hold_10x) +
    '<td style="'+as+'">'+(s.actual||'-')+'</td>';
  tbody.appendChild(r);
}});
</script></body></html>"""

    filename = f'output/backtests/leveraged_backtest_{asset_name.lower()}.html'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n  Dashboard: {filename}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  LEVERAGED CFD BACKTEST -- UNIT ACCUMULATION")
    print(f"  {ASSET} | {CURRENCY} {STARTING_CAPITAL:,} | {LEVERAGE}:1 | {STOP_LOSS_PCT}% stop")
    print("=" * 60)

    if not os.path.exists('data/hourly_best_models.csv'):
        print("ERROR: hourly_best_models.csv not found.")
        return

    df_best = pd.read_csv('data/hourly_best_models.csv')
    row = df_best[df_best['coin'] == ASSET]
    if len(row) == 0:
        print(f"  {ASSET}: No model config found")
        return

    model_names = row.iloc[0]['models'].split('+')
    window = int(row.iloc[0]['best_window'])

    print(f"\n  Models: {'+'.join(model_names)}, window={window}h")
    print(f"\n  Strategies:")
    for strat in STRATEGIES:
        print(f"    {strat['name']:35s} | {strat['desc']}")

    print(f"\n  Updating data...")
    try:
        update_all_data([ASSET])
    except:
        pass

    print(f"  Generating {REPLAY_HOURS} signals...")
    signals = generate_signals(ASSET, model_names, window, REPLAY_HOURS)
    if not signals:
        print("  No signals generated")
        return
    print(f"  {len(signals)} signals generated")

    signals = simulate_strategies(signals, STARTING_CAPITAL, LEVERAGE, STOP_LOSS_PCT, STRATEGIES)

    # Print results
    last = signals[-1]
    print(f"\n  {'Strategy':35s} {'Final':>12s} {'P&L':>10s} {'Return':>8s} {'Trades':>7s} {'WR':>6s} {'MaxDD':>7s}")
    print(f"  {'-'*88}")
    for strat in STRATEGIES:
        st = strat['stats']
        final = last[strat['key']]
        pnl = final - STARTING_CAPITAL
        pct = pnl / STARTING_CAPITAL * 100
        wr = (st['wins'] / st['trades'] * 100) if st['trades'] > 0 else 0
        print(f"  {strat['name']:35s} {CURRENCY} {final:>9,.0f} {pnl:>+9,.0f} {pct:>+7.1f}% {st['trades']:>5d}   {wr:>4.0f}%  {st['max_drawdown']:>5.1f}%")

    hold_10x = last['hold_10x']
    hold_1x = last['hold_1x']
    print(f"  {'Buy & Hold 10:1':35s} {CURRENCY} {hold_10x:>9,.0f} {hold_10x-STARTING_CAPITAL:>+9,.0f} {(hold_10x/STARTING_CAPITAL-1)*100:>+7.1f}%")
    print(f"  {'Buy & Hold 1:1':35s} {CURRENCY} {hold_1x:>9,.0f} {hold_1x-STARTING_CAPITAL:>+9,.0f} {(hold_1x/STARTING_CAPITAL-1)*100:>+7.1f}%")

    export_dashboard(ASSET, signals, STARTING_CAPITAL, CURRENCY, LEVERAGE, STOP_LOSS_PCT, STRATEGIES)

    print(f"\n{'='*60}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
