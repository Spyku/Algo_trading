"""
TESTING: Deku — Multi-Timeframe Fusion Backtest
============================================================
Combines Deku hourly (1h candles) and Deku V15 (15-min candles)
into a single decision engine. Tests all cross-timeframe strategies.

4 independent signals per decision point:
  - 1h  4h  (hourly candle, 4-hour horizon)
  - 1h  8h  (hourly candle, 8-hour horizon)
  - 15' s4  (15-min candle, 4-candle horizon = 60 min)
  - 15' s8  (15-min candle, 8-candle horizon = 120 min)

Evaluation cadences:
  - Hourly:  all 4 signals checked once per hour
  - 15-min:  V15 signals fresh every 15 min, 1h carried forward

Also compares Deku V15 vs CASCA V15 (Optuna vs grid search).

Usage:
  python testing_deku_fusion.py              # BTC, hourly cadence (default)
  python testing_deku_fusion.py ETH          # Test ETH
  python testing_deku_fusion.py --15min      # Evaluate on 15-min cadence
  python testing_deku_fusion.py --compare    # Deku V15 vs CASCA V15 comparison only

Results: models/testing_deku_fusion_results.csv
"""

import sys
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import json

# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_ASSET = 'BTC'
REPLAY_HOURS = 400  # enough overlap for meaningful comparison
RESULTS_CSV = 'models/testing_deku_fusion_results.csv'

# Trading costs (same as production)
TRADING_FEE = 0.0011

# ============================================================
# Import both Deku systems
# ============================================================
import importlib
deku_1h = importlib.import_module('crypto_trading_system_deku')
deku_15m = importlib.import_module('crypto_trading_system_deku_15m')


# ============================================================
# STRATEGY DEFINITIONS
# ============================================================
# Each strategy is a function(sig_1h_4h, conf_1h_4h, sig_1h_8h, conf_1h_8h,
#                              sig_15m_s4, conf_15m_s4, sig_15m_s8, conf_15m_s8,
#                              min_conf) -> signal
# Returns 'BUY', 'SELL', or 'HOLD'

def _any_sell(*sigs):
    """True if any signal is SELL."""
    return any(s == 'SELL' for s in sigs)


STRATEGIES = {}

def _register(name):
    def decorator(fn):
        STRATEGIES[name] = fn
        return fn
    return decorator


# --- 1h-only baselines ---
@_register('1h_both_agree')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    if s14 == 'SELL' or s18 == 'SELL': return 'SELL'
    if s14 == 'BUY' and s18 == 'BUY' and c14 >= mc and c18 >= mc: return 'BUY'
    return 'HOLD'

@_register('1h_either_agree')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    if s14 == 'SELL' or s18 == 'SELL': return 'SELL'
    if (s14 == 'BUY' and c14 >= mc) or (s18 == 'BUY' and c18 >= mc): return 'BUY'
    return 'HOLD'

@_register('1h_4h_only')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    if s14 == 'SELL': return 'SELL'
    if s14 == 'BUY' and c14 >= mc: return 'BUY'
    return 'HOLD'

@_register('1h_8h_only')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    if s18 == 'SELL': return 'SELL'
    if s18 == 'BUY' and c18 >= mc: return 'BUY'
    return 'HOLD'

# --- 15'-only baselines ---
@_register('15m_s4_only')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    if s15_4 == 'SELL': return 'SELL'
    if s15_4 == 'BUY' and c15_4 >= mc: return 'BUY'
    return 'HOLD'

@_register('15m_s8_only')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    if s15_8 == 'SELL': return 'SELL'
    if s15_8 == 'BUY' and c15_8 >= mc: return 'BUY'
    return 'HOLD'

@_register('15m_both_agree')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    if s15_4 == 'SELL' or s15_8 == 'SELL': return 'SELL'
    if s15_4 == 'BUY' and s15_8 == 'BUY' and c15_4 >= mc and c15_8 >= mc: return 'BUY'
    return 'HOLD'

@_register('15m_either_agree')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    if s15_4 == 'SELL' or s15_8 == 'SELL': return 'SELL'
    if (s15_4 == 'BUY' and c15_4 >= mc) or (s15_8 == 'BUY' and c15_8 >= mc): return 'BUY'
    return 'HOLD'

# --- Cross-timeframe strategies ---
@_register('all_4_agree')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    """All 4 signals must say BUY. Any SELL triggers SELL."""
    if _any_sell(s14, s18, s15_4, s15_8): return 'SELL'
    if (s14 == 'BUY' and s18 == 'BUY' and s15_4 == 'BUY' and s15_8 == 'BUY'
            and c14 >= mc and c18 >= mc and c15_4 >= mc and c15_8 >= mc):
        return 'BUY'
    return 'HOLD'

@_register('any_4_buy')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    """Any of 4 signals says BUY. Any SELL triggers SELL."""
    if _any_sell(s14, s18, s15_4, s15_8): return 'SELL'
    if ((s14 == 'BUY' and c14 >= mc) or (s18 == 'BUY' and c18 >= mc) or
            (s15_4 == 'BUY' and c15_4 >= mc) or (s15_8 == 'BUY' and c15_8 >= mc)):
        return 'BUY'
    return 'HOLD'

@_register('majority_3of4')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    """3 out of 4 must say BUY. Any SELL triggers SELL."""
    if _any_sell(s14, s18, s15_4, s15_8): return 'SELL'
    buys = sum([
        s14 == 'BUY' and c14 >= mc,
        s18 == 'BUY' and c18 >= mc,
        s15_4 == 'BUY' and c15_4 >= mc,
        s15_8 == 'BUY' and c15_8 >= mc,
    ])
    if buys >= 3: return 'BUY'
    return 'HOLD'

@_register('1h_8h_and_15m_s8')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    """Both long horizons agree (1h 8h + 15' s8)."""
    if s18 == 'SELL' or s15_8 == 'SELL': return 'SELL'
    if s18 == 'BUY' and s15_8 == 'BUY' and c18 >= mc and c15_8 >= mc: return 'BUY'
    return 'HOLD'

@_register('1h_8h_or_15m_s8')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    """Either long horizon (1h 8h OR 15' s8)."""
    if s18 == 'SELL' or s15_8 == 'SELL': return 'SELL'
    if (s18 == 'BUY' and c18 >= mc) or (s15_8 == 'BUY' and c15_8 >= mc): return 'BUY'
    return 'HOLD'

@_register('1h_4h_and_15m_s4')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    """Both short horizons agree (1h 4h + 15' s4)."""
    if s14 == 'SELL' or s15_4 == 'SELL': return 'SELL'
    if s14 == 'BUY' and s15_4 == 'BUY' and c14 >= mc and c15_4 >= mc: return 'BUY'
    return 'HOLD'

@_register('1h_8h_confirmed_15m')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    """1h 8h makes the call, 15' must confirm (either 15' horizon)."""
    if s18 == 'SELL': return 'SELL'
    if (s18 == 'BUY' and c18 >= mc and
            ((s15_4 == 'BUY' and c15_4 >= mc) or (s15_8 == 'BUY' and c15_8 >= mc))):
        return 'BUY'
    return 'HOLD'

@_register('15m_fast_entry_1h_exit')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    """15' triggers BUY (fast entry), 1h triggers SELL (slow exit)."""
    if s14 == 'SELL' or s18 == 'SELL': return 'SELL'
    if (s15_4 == 'BUY' and c15_4 >= mc) or (s15_8 == 'BUY' and c15_8 >= mc): return 'BUY'
    return 'HOLD'

@_register('1h_trend_15m_timing')
def _(s14, c14, s18, c18, s15_4, c15_4, s15_8, c15_8, mc):
    """1h 8h (trend) + 15' s4 (timing) must agree."""
    if s18 == 'SELL' or s15_4 == 'SELL': return 'SELL'
    if s18 == 'BUY' and s15_4 == 'BUY' and c18 >= mc and c15_4 >= mc: return 'BUY'
    return 'HOLD'


# ============================================================
# SIGNAL GENERATION
# ============================================================
def _load_1h_signals(asset):
    """Generate Deku hourly signals for 4h and 8h."""
    csv_path = deku_1h._get_models_csv_path()
    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found. Run Deku Mode D first.")
        return None, None

    df = pd.read_csv(csv_path)
    signals_4h, signals_8h = None, None

    for horizon in [4, 8]:
        cfg = df[(df['coin'] == asset) & (df['horizon'] == horizon)]
        if cfg.empty:
            print(f"  WARNING: No Deku 1h {horizon}h model for {asset}")
            continue

        row = cfg.iloc[0]
        feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
        gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0

        print(f"  Generating 1h {horizon}h signals ({row['models']}, gamma={gamma:.4f})...")
        sigs = deku_1h.generate_signals(asset, row['models'].split('+'),
                                        int(row['best_window']), REPLAY_HOURS,
                                        feature_override=feats, horizon=horizon, gamma=gamma)
        if horizon == 4:
            signals_4h = sigs
        else:
            signals_8h = sigs

    return signals_4h, signals_8h


def _load_15m_signals(asset):
    """Generate Deku V15 signals for s4 (60') and s8 (120')."""
    csv_path = deku_15m._get_models_csv_path()
    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found. Run Deku V15 Mode D first.")
        return None, None

    df = pd.read_csv(csv_path)
    signals_s4, signals_s8 = None, None

    # V15 replay: convert hours to 15-min candles
    replay_candles = REPLAY_HOURS * 4  # 400h = 1600 candles

    for horizon in [4, 8]:
        cfg = df[(df['coin'] == asset) & (df['horizon'] == horizon)]
        if cfg.empty:
            print(f"  WARNING: No Deku V15 s{horizon} model for {asset}")
            continue

        row = cfg.iloc[0]
        feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
        gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0

        print(f"  Generating 15' s{horizon} signals ({row['models']}, gamma={gamma:.4f})...")
        sigs = deku_15m.generate_signals(asset, row['models'].split('+'),
                                         int(row['best_window']), replay_candles,
                                         feature_override=feats, horizon=horizon, gamma=gamma)
        if horizon == 4:
            signals_s4 = sigs
        else:
            signals_s8 = sigs

    return signals_s4, signals_s8


# ============================================================
# MERGE SIGNALS ON COMMON TIMELINE
# ============================================================
def _merge_signals_hourly(h_4h, h_8h, m15_s4, m15_s8):
    """
    Merge all 4 signal streams on hourly timestamps.
    15' signals are snapped to the nearest hour (use the :00 candle).
    """
    map_1h_4h = {s['datetime']: s for s in (h_4h or [])}
    map_1h_8h = {s['datetime']: s for s in (h_8h or [])}

    # 15' signals: keep only the :00 candles (top of hour)
    map_15m_s4 = {}
    for s in (m15_s4 or []):
        dt = pd.Timestamp(s['datetime'])
        if dt.minute == 0:
            map_15m_s4[s['datetime']] = s

    map_15m_s8 = {}
    for s in (m15_s8 or []):
        dt = pd.Timestamp(s['datetime'])
        if dt.minute == 0:
            map_15m_s8[s['datetime']] = s

    # Common hourly timestamps (where at least 1h system has data)
    h_times = set(list(map_1h_4h.keys()) + list(map_1h_8h.keys()))
    all_times = sorted(h_times)

    merged = []
    for dt in all_times:
        s14 = map_1h_4h.get(dt)
        s18 = map_1h_8h.get(dt)
        s15_4 = map_15m_s4.get(dt)
        s15_8 = map_15m_s8.get(dt)

        price = (s14 or s18 or s15_4 or s15_8)['close']

        merged.append({
            'datetime': dt,
            'close': price,
            'v5_4h_sig': s14['signal'] if s14 else 'HOLD',
            'v5_4h_conf': s14['confidence'] if s14 else 50,
            'v5_8h_sig': s18['signal'] if s18 else 'HOLD',
            'v5_8h_conf': s18['confidence'] if s18 else 50,
            'v15_4c_sig': s15_4['signal'] if s15_4 else 'HOLD',
            'v15_4c_conf': s15_4['confidence'] if s15_4 else 50,
            'v15_8c_sig': s15_8['signal'] if s15_8 else 'HOLD',
            'v15_8c_conf': s15_8['confidence'] if s15_8 else 50,
        })

    return merged


def _merge_signals_15min(h_4h, h_8h, m15_s4, m15_s8):
    """
    Merge all 4 signal streams on 15-min timestamps.
    1h signals are carried forward (latest hourly signal applies until next hour).
    """
    map_1h_4h = {s['datetime']: s for s in (h_4h or [])}
    map_1h_8h = {s['datetime']: s for s in (h_8h or [])}
    map_15m_s4 = {s['datetime']: s for s in (m15_s4 or [])}
    map_15m_s8 = {s['datetime']: s for s in (m15_s8 or [])}

    # Use 15' timestamps as the cadence
    v15_times = sorted(set(list(map_15m_s4.keys()) + list(map_15m_s8.keys())))

    if not v15_times:
        return []

    # Carry-forward state for 1h signals
    last_1h_4h = {'signal': 'HOLD', 'confidence': 50}
    last_1h_8h = {'signal': 'HOLD', 'confidence': 50}

    merged = []
    for dt in v15_times:
        # Update 1h signals if we have a new hourly reading
        if dt in map_1h_4h:
            last_1h_4h = map_1h_4h[dt]
        if dt in map_1h_8h:
            last_1h_8h = map_1h_8h[dt]

        s15_4 = map_15m_s4.get(dt)
        s15_8 = map_15m_s8.get(dt)

        price = (s15_4 or s15_8)['close']

        merged.append({
            'datetime': dt,
            'close': price,
            'v5_4h_sig': last_1h_4h.get('signal', 'HOLD'),
            'v5_4h_conf': last_1h_4h.get('confidence', 50),
            'v5_8h_sig': last_1h_8h.get('signal', 'HOLD'),
            'v5_8h_conf': last_1h_8h.get('confidence', 50),
            'v15_4c_sig': s15_4['signal'] if s15_4 else 'HOLD',
            'v15_4c_conf': s15_4['confidence'] if s15_4 else 50,
            'v15_8c_sig': s15_8['signal'] if s15_8 else 'HOLD',
            'v15_8c_conf': s15_8['confidence'] if s15_8 else 50,
        })

    return merged


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest_strategy(merged_signals, strategy_name, min_conf):
    """Run a single strategy over merged signals. Returns result dict."""
    strat_fn = STRATEGIES[strategy_name]
    cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
    trades, wins = 0, 0
    peak = 1000.0
    max_dd = 0.0

    for m in merged_signals:
        signal = strat_fn(
            m['v5_4h_sig'], m['v5_4h_conf'],
            m['v5_8h_sig'], m['v5_8h_conf'],
            m['v15_4c_sig'], m['v15_4c_conf'],
            m['v15_8c_sig'], m['v15_8c_conf'],
            min_conf
        )

        if signal == 'BUY' and not in_pos:
            held = cash * (1 - TRADING_FEE) / m['close']
            cash = 0
            in_pos = True
            entry_px = m['close']
            trades += 1
        elif signal == 'SELL' and in_pos:
            cash = held * m['close'] * (1 - TRADING_FEE)
            if m['close'] > entry_px:
                wins += 1
            held = 0
            in_pos = False

        # Max drawdown
        current_val = (held * m['close']) if in_pos else cash
        if current_val > peak:
            peak = current_val
        dd = (peak - current_val) / peak
        if dd > max_dd:
            max_dd = dd

    # Close open position at end
    if in_pos and merged_signals:
        cash = held * merged_signals[-1]['close'] * (1 - TRADING_FEE)

    cum_ret = (cash / 1000.0 - 1) * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0
    # Buy & hold
    if merged_signals:
        bh_ret = (merged_signals[-1]['close'] / merged_signals[0]['close'] - 1) * 100
    else:
        bh_ret = 0

    return {
        'strategy': strategy_name,
        'min_conf': min_conf,
        'return_pct': round(cum_ret, 2),
        'trades': trades,
        'win_rate': round(win_rate, 1),
        'max_dd_pct': round(max_dd * 100, 1),
        'buy_hold_pct': round(bh_ret, 2),
        'alpha': round(cum_ret - bh_ret, 2),
    }


# ============================================================
# DEKU V15 vs CASCA V15 COMPARISON
# ============================================================
def compare_deku_vs_casca_v15(asset):
    """Compare Deku V15 (Optuna) vs CASCA V15 (grid search) results from CSVs."""
    print(f"\n{'='*80}")
    print(f"  DEKU V15 vs CASCA V15 — {asset} (Optuna vs Grid Search)")
    print(f"{'='*80}")

    deku_csv = deku_15m._get_models_csv_path()
    casca_csv = f'{deku_15m.MODELS_DIR}/crypto_15m_best_models.csv'

    has_deku = os.path.exists(deku_csv)
    has_casca = os.path.exists(casca_csv)

    if not has_deku:
        print(f"  No Deku V15 results ({deku_csv}). Run Deku V15 Mode D first.")
        return
    if not has_casca:
        print(f"  No CASCA V15 results ({casca_csv}). Cannot compare.")
        print(f"  (Deku V15 results available — comparison skipped)")
        return

    df_deku = pd.read_csv(deku_csv)
    df_casca = pd.read_csv(casca_csv)

    print(f"\n  {'System':<14} {'Horizon':<9} {'Models':<24} {'Window':<8} {'Acc%':<8} {'Return%':<10} {'APF':<8} {'Feats':<7} {'Gamma':<8}")
    print(f"  {'-'*100}")

    for horizon in [4, 8]:
        # Deku
        d_row = df_deku[(df_deku['coin'] == asset) & (df_deku['horizon'] == horizon)]
        if not d_row.empty:
            r = d_row.iloc[0]
            apf = r.get('combined_score', 0)
            print(f"  {'Deku (Optuna)':<14} s{horizon:<8} {str(r['models']):<24} {int(r['best_window']):<8} "
                  f"{r['accuracy']:<8.1f} {r.get('return_pct', 0):<+10.1f} {apf:<8.3f} "
                  f"{int(r.get('n_features', 0)):<7} {r.get('gamma', 1.0):<8.4f}")

        # CASCA
        c_row = df_casca[(df_casca['coin'] == asset) & (df_casca['horizon'] == horizon)]
        if not c_row.empty:
            r = c_row.iloc[0]
            apf = r.get('combined_score', 0)
            print(f"  {'CASCA (Grid)':<14} s{horizon:<8} {str(r['models']):<24} {int(r['best_window']):<8} "
                  f"{r['accuracy']:<8.1f} {r.get('return_pct', 0):<+10.1f} {apf:<8.3f} "
                  f"{int(r.get('n_features', 0)):<7} {r.get('gamma', 1.0):<8.4f}")

        # Delta
        if not d_row.empty and not c_row.empty:
            d = d_row.iloc[0]
            c = c_row.iloc[0]
            d_ret = d.get('return_pct', 0)
            c_ret = c.get('return_pct', 0)
            d_apf = d.get('combined_score', 0)
            c_apf = c.get('combined_score', 0)
            delta_ret = d_ret - c_ret
            delta_apf = d_apf - c_apf
            winner = "Deku" if d_ret > c_ret else "CASCA"
            print(f"  {'  -> Delta':<14} s{horizon:<8} {'':24} {'':8} "
                  f"{'':8} {delta_ret:<+10.1f} {delta_apf:<+8.3f} {'':7} {'':8} {winner}")

        print(f"  {'-'*100}")

    print()


# ============================================================
# MAIN
# ============================================================
def main():
    asset = DEFAULT_ASSET
    cadence = 'hourly'
    compare_only = False

    for arg in sys.argv[1:]:
        if arg.upper() in ('BTC', 'ETH', 'XRP', 'DOGE'):
            asset = arg.upper()
        elif arg == '--15min':
            cadence = '15min'
        elif arg == '--compare':
            compare_only = True

    os.makedirs('models', exist_ok=True)

    # Always show Deku V15 vs CASCA V15 comparison
    compare_deku_vs_casca_v15(asset)

    if compare_only:
        return

    print("=" * 80)
    print("  DEKU: Multi-Timeframe Fusion Backtest")
    print(f"  Asset: {asset}")
    print(f"  Cadence: {cadence}")
    print(f"  Replay: {REPLAY_HOURS}h")
    print(f"  Strategies: {len(STRATEGIES)}")
    print("=" * 80)

    t0 = time.time()

    # Step 1: Generate all signals
    print("\n--- Generating Deku hourly signals ---")
    h_4h, h_8h = _load_1h_signals(asset)

    print("\n--- Generating Deku V15 signals ---")
    m15_s4, m15_s8 = _load_15m_signals(asset)

    # Check what we have
    n_1h_4h = len(h_4h) if h_4h else 0
    n_1h_8h = len(h_8h) if h_8h else 0
    n_15m_s4 = len(m15_s4) if m15_s4 else 0
    n_15m_s8 = len(m15_s8) if m15_s8 else 0

    print(f"\n  Signal counts: 1h 4h={n_1h_4h}, 1h 8h={n_1h_8h}, "
          f"15' s4={n_15m_s4}, 15' s8={n_15m_s8}")

    if n_1h_4h == 0 and n_1h_8h == 0:
        print("  ERROR: No hourly Deku signals. Run Deku Mode D first.")
        return
    if n_15m_s4 == 0 and n_15m_s8 == 0:
        print("  ERROR: No 15' Deku signals. Run Deku V15 Mode D first.")
        return

    # Step 2: Merge signals
    print(f"\n--- Merging signals ({cadence} cadence) ---")
    if cadence == '15min':
        merged = _merge_signals_15min(h_4h, h_8h, m15_s4, m15_s8)
    else:
        merged = _merge_signals_hourly(h_4h, h_8h, m15_s4, m15_s8)

    print(f"  Merged timeline: {len(merged)} data points")

    if len(merged) < 10:
        print("  ERROR: Not enough overlapping data points.")
        return

    # Step 3: Test all strategies with confidence sweep
    print(f"\n--- Testing {len(STRATEGIES)} strategies × 8 confidence levels ---")
    confidence_levels = [55, 60, 65, 70, 75, 80, 85, 90]

    all_results = []
    for strat_name in STRATEGIES:
        best_result = None
        best_return = -999

        for mc in confidence_levels:
            result = backtest_strategy(merged, strat_name, mc)
            result['cadence'] = cadence
            result['asset'] = asset
            result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M')
            result['n_datapoints'] = len(merged)

            if result['return_pct'] > best_return:
                best_return = result['return_pct']
                best_result = result

        all_results.append(best_result)

    # Sort by return
    all_results.sort(key=lambda x: x['return_pct'], reverse=True)

    # Step 4: Print results table
    print(f"\n{'='*100}")
    print(f"  DEKU FUSION RESULTS — {asset} ({cadence} cadence, {len(merged)} points)")
    print(f"{'='*100}")
    print(f"  {'Strategy':<28} {'Conf%':<7} {'Return%':<10} {'Trades':<8} {'WinRate':<9} {'MaxDD':<8} {'B&H%':<9} {'Alpha':<8}")
    print(f"  {'-'*96}")

    for r in all_results:
        marker = ' ***' if r == all_results[0] else ''
        # Color-code category
        if r['strategy'].startswith('1h_'):
            cat = '[1h]'
        elif r['strategy'].startswith('15m_'):
            cat = '[15]'
        else:
            cat = '[MIX]'
        print(f"  {cat} {r['strategy']:<24} {r['min_conf']:<7} {r['return_pct']:<+10.1f} "
              f"{r['trades']:<8} {r['win_rate']:<9.1f} {r['max_dd_pct']:<8.1f} {r['buy_hold_pct']:<+9.1f} "
              f"{r['alpha']:<+8.1f}{marker}")

    print(f"  {'='*96}")

    # Highlight
    best = all_results[0]
    print(f"\n  BEST OVERALL: {best['strategy']} @{best['min_conf']}% -> "
          f"{best['return_pct']:+.1f}% return, {best['trades']} trades, "
          f"alpha={best['alpha']:+.1f}% vs B&H")

    # Compare categories
    cat_1h = [r for r in all_results if r['strategy'].startswith('1h_')]
    cat_15m = [r for r in all_results if r['strategy'].startswith('15m_')]
    cat_mix = [r for r in all_results if not r['strategy'].startswith('1h_') and not r['strategy'].startswith('15m_')]

    best_1h = max(cat_1h, key=lambda x: x['return_pct']) if cat_1h else None
    best_15m = max(cat_15m, key=lambda x: x['return_pct']) if cat_15m else None
    best_mix = max(cat_mix, key=lambda x: x['return_pct']) if cat_mix else None

    print(f"\n  Category winners:")
    if best_1h:
        print(f"    1h-only:   {best_1h['strategy']:24s} -> {best_1h['return_pct']:+.1f}%")
    if best_15m:
        print(f"    15'-only:  {best_15m['strategy']:24s} -> {best_15m['return_pct']:+.1f}%")
    if best_mix:
        print(f"    Cross-TF:  {best_mix['strategy']:24s} -> {best_mix['return_pct']:+.1f}%")

    # Fusion advantage
    if best_1h and best_mix:
        delta = best_mix['return_pct'] - best_1h['return_pct']
        print(f"\n  Fusion advantage vs 1h-only: {delta:+.1f}% {'(BETTER)' if delta > 0 else '(WORSE)'}")
    if best_15m and best_mix:
        delta = best_mix['return_pct'] - best_15m['return_pct']
        print(f"  Fusion advantage vs 15'-only: {delta:+.1f}% {'(BETTER)' if delta > 0 else '(WORSE)'}")

    # Step 5: Save results
    df_results = pd.DataFrame(all_results)
    if os.path.exists(RESULTS_CSV):
        df_existing = pd.read_csv(RESULTS_CSV)
        # Remove old results for same asset+cadence
        mask = (df_existing['asset'] == asset) & (df_existing['cadence'] == cadence)
        df_existing = df_existing[~mask]
        df_results = pd.concat([df_existing, df_results], ignore_index=True)
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"\n  Results saved to {RESULTS_CSV}")

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
