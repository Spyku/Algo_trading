"""
Master regime-switching backtest.
Tests all horizon combinations × all regime detectors.

Usage:
  python backtest_regime_master.py                         # 2-month default, all horizons
  python backtest_regime_master.py --months 4              # 4-month backtest
  python backtest_regime_master.py --horizons 6,8          # only test 6h and 8h (fast)
  python backtest_regime_master.py --bull 6 --bear 8       # fix pair, compare regimes only
  python backtest_regime_master.py --regimes sma,rsi       # filter regime families
  python backtest_regime_master.py --no-combos             # single-horizon baselines only
  python backtest_regime_master.py --top 20                # show top 20 results
  python backtest_regime_master.py --conf 80               # confidence threshold (default 90)
"""
import sys, os, argparse, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
from itertools import permutations

from crypto_trading_system_doohan import (
    generate_signals, simulate_portfolio, load_data,
    TRADING_FEE, _suppress_stderr
)

PROD_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'models', 'crypto_doohan_v1_7_1_production.csv')
ALL_HORIZONS = [4, 5, 6, 7, 8, 10, 12, 14]
MONTHS_TO_HOURS = {1: 720, 2: 1440, 3: 2160, 4: 2880, 6: 4320}

# ─── Argument parsing ───────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Master regime-switching backtest')
    p.add_argument('--months', type=int, default=2, choices=[1,2,3,4,6])
    p.add_argument('--horizons', type=str, default=None, help='Comma-separated horizons (e.g. 6,8,10)')
    p.add_argument('--bull', type=int, default=None, help='Fix bull horizon')
    p.add_argument('--bear', type=int, default=None, help='Fix bear horizon')
    p.add_argument('--regimes', type=str, default=None, help='Filter regime families (e.g. sma,rsi,dd)')
    p.add_argument('--no-combos', action='store_true', help='Skip regime combos, baselines only')
    p.add_argument('--top', type=int, default=15, help='Show top N results')
    p.add_argument('--conf', type=int, default=90, help='Min confidence threshold')
    p.add_argument('--asset', type=str, default='BTC', help='Asset to test')
    return p.parse_args()

# ─── Signal generation ───────────────────────────────────────────────────────
def generate_horizon_signals(asset, horizon, replay, df_models):
    rows = df_models[(df_models['coin'] == asset) & (df_models['horizon'] == horizon)]
    if len(rows) == 0:
        return None
    row = rows.iloc[0]
    feats = row['optimal_features'].split(',') if pd.notna(row.get('optimal_features', '')) else None
    gamma = float(row.get('gamma', 1.0)) if pd.notna(row.get('gamma', 1.0)) else 1.0
    model_desc = f"{row['models']} w={int(row['best_window'])} g={gamma} f={int(row['n_features'])} [{row.get('sampler','?')}]"

    with _suppress_stderr():
        sigs = generate_signals(asset, row['models'].split('+'),
                                int(row['best_window']), replay,
                                feature_override=feats, horizon=horizon, gamma=gamma)
        sigs = simulate_portfolio(sigs)

    result = {}
    for s in sigs:
        dt = s['datetime']
        if isinstance(dt, str):
            dt = datetime.strptime(dt, '%Y-%m-%d %H:%M')
            s['datetime'] = dt
        result[dt] = s
    return result, model_desc

# ─── Regime indicators ───────────────────────────────────────────────────────
def build_indicators(asset):
    df = load_data(asset)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()

    # SMAs
    for w in [24, 48, 72, 100, 200]:
        df[f'sma{w}'] = df['close'].rolling(w).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi14'] = 100 - (100 / (1 + rs))

    # Drawdown from rolling high
    for w in [24, 48, 72]:
        df[f'high_{w}'] = df['high'].rolling(w).max()
        df[f'dd_{w}'] = (df['close'] / df[f'high_{w}'] - 1) * 100

    # Bounce from rolling low
    for w in [24, 48]:
        df[f'low_{w}'] = df['low'].rolling(w).min()
        df[f'bounce_{w}'] = (df['close'] / df[f'low_{w}'] - 1) * 100

    # EMA cross
    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    df['macd_line'] = df['ema12'] - df['ema26']

    return df

def get_regime_detectors(df_ind, regime_filter=None):
    """Return dict of {name: function(dt) -> bool (True=bull)}."""
    ind = df_ind.to_dict('index')

    def safe(dt, fn, default=True):
        if dt not in ind:
            return default
        try:
            return fn(ind[dt])
        except (KeyError, TypeError):
            return default

    detectors = {
        # SMA crossovers
        'sma24>sma72':   lambda dt: safe(dt, lambda r: r['sma24'] > r['sma72']),
        'sma24>sma100':  lambda dt: safe(dt, lambda r: r['sma24'] > r['sma100']),
        'sma48>sma100':  lambda dt: safe(dt, lambda r: r['sma48'] > r['sma100']),
        'sma48>sma200':  lambda dt: safe(dt, lambda r: r['sma48'] > r['sma200']),

        # Price vs SMA
        'price>sma48':   lambda dt: safe(dt, lambda r: r['close'] > r['sma48']),
        'price>sma72':   lambda dt: safe(dt, lambda r: r['close'] > r['sma72']),
        'price>sma100':  lambda dt: safe(dt, lambda r: r['close'] > r['sma100']),

        # RSI
        'rsi>55':        lambda dt: safe(dt, lambda r: r['rsi14'] > 55),
        'rsi>50':        lambda dt: safe(dt, lambda r: r['rsi14'] > 50),
        'rsi>45':        lambda dt: safe(dt, lambda r: r['rsi14'] > 45),

        # Drawdown
        'dd48>-2%':      lambda dt: safe(dt, lambda r: r['dd_48'] > -2),
        'dd48>-3%':      lambda dt: safe(dt, lambda r: r['dd_48'] > -3),
        'dd48>-5%':      lambda dt: safe(dt, lambda r: r['dd_48'] > -5),
        'dd72>-3%':      lambda dt: safe(dt, lambda r: r['dd_72'] > -3),
        'dd72>-5%':      lambda dt: safe(dt, lambda r: r['dd_72'] > -5),

        # Bounce
        'bounce48>2%':   lambda dt: safe(dt, lambda r: r['bounce_48'] > 2),
        'bounce48>3%':   lambda dt: safe(dt, lambda r: r['bounce_48'] > 3),

        # MACD
        'macd>0':        lambda dt: safe(dt, lambda r: r['macd_line'] > 0),

        # Combos
        'sma24>100+dd48>-3':  lambda dt: safe(dt, lambda r: r['sma24'] > r['sma100'] and r['dd_48'] > -3),
        'sma24>100+rsi>45':   lambda dt: safe(dt, lambda r: r['sma24'] > r['sma100'] and r['rsi14'] > 45),
        'dd48>-3%+rsi>45':    lambda dt: safe(dt, lambda r: r['dd_48'] > -3 and r['rsi14'] > 45),
        'price>sma72+rsi>50': lambda dt: safe(dt, lambda r: r['close'] > r['sma72'] and r['rsi14'] > 50),
    }

    if regime_filter:
        families = [f.strip().lower() for f in regime_filter.split(',')]
        filtered = {}
        for name, fn in detectors.items():
            for fam in families:
                if fam in name.lower():
                    filtered[name] = fn
                    break
        return filtered

    return detectors

# ─── Simulation engine ───────────────────────────────────────────────────────
def simulate(all_dts, signals_cache, horizon_picker, conf):
    """Fast simulation — horizon_picker(dt) returns horizon int."""
    cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
    trades, wins = 0, 0
    switches = 0
    last_h = None
    first_price = None
    last_price = None

    for dt in all_dts:
        h = horizon_picker(dt)
        if last_h is not None and h != last_h:
            switches += 1
        last_h = h

        sigs = signals_cache.get(h)
        if sigs is None:
            continue
        s = sigs.get(dt)
        if s is None:
            # Track price from any available horizon for B&H
            for oh in signals_cache:
                os = signals_cache[oh].get(dt)
                if os:
                    last_price = os['close']
                    if first_price is None:
                        first_price = last_price
                    break
            continue

        price = s['close']
        last_price = price
        if first_price is None:
            first_price = price
        sig = s['signal']
        c = s['confidence']

        if sig == 'BUY' and c >= conf and not in_pos:
            held = cash * (1 - TRADING_FEE) / price
            cash = 0
            in_pos = True
            entry_px = price
            trades += 1
        elif sig == 'SELL' and in_pos:
            cash = held * price * (1 - TRADING_FEE)
            if price > entry_px:
                wins += 1
            held = 0
            in_pos = False

    # Close open position
    if in_pos and last_price:
        cash = held * last_price * (1 - TRADING_FEE)
        if last_price > entry_px:
            wins += 1

    ret = (cash / 1000.0 - 1) * 100
    wr = (wins / trades * 100) if trades > 0 else 0
    bh = ((last_price / first_price - 1) * 100) if first_price and last_price else 0
    alpha = ret - bh
    return ret, trades, wr, switches, bh, alpha

# ─── Monthly breakdown simulation ────────────────────────────────────────────
def simulate_monthly(all_dts, signals_cache, horizon_picker, conf):
    """Simulate and return monthly breakdown."""
    cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
    trades, wins = 0, 0
    monthly = {}  # {(year,month): {'start_eq': ..., 'end_eq': ..., 'trades': 0, 'wins': 0}}

    for dt in all_dts:
        h = horizon_picker(dt)
        sigs = signals_cache.get(h)
        if sigs is None:
            continue
        s = sigs.get(dt)
        if s is None:
            continue

        price = s['close']
        ym = (dt.year, dt.month)

        eq = cash + held * price if in_pos else cash

        if ym not in monthly:
            monthly[ym] = {'start_eq': eq, 'end_eq': eq, 'trades': 0, 'wins': 0}
        monthly[ym]['end_eq'] = eq

        sig = s['signal']
        c = s['confidence']

        if sig == 'BUY' and c >= conf and not in_pos:
            held = cash * (1 - TRADING_FEE) / price
            cash = 0
            in_pos = True
            entry_px = price
            trades += 1
            monthly[ym]['trades'] += 1
        elif sig == 'SELL' and in_pos:
            cash = held * price * (1 - TRADING_FEE)
            w = price > entry_px
            if w:
                wins += 1
                monthly[ym]['wins'] += 1
            held = 0
            in_pos = False
            monthly[ym]['end_eq'] = cash

    # Update final equity for last month
    if in_pos and all_dts:
        for h_sigs in signals_cache.values():
            s = h_sigs.get(all_dts[-1])
            if s:
                last_price = s['close']
                cash = held * last_price * (1 - TRADING_FEE)
                ym = (all_dts[-1].year, all_dts[-1].month)
                if ym in monthly:
                    monthly[ym]['end_eq'] = cash
                break

    return monthly

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    replay = MONTHS_TO_HOURS[args.months]
    df_models = pd.read_csv(PROD_CSV)

    # Determine horizons to test
    if args.horizons:
        horizons = [int(h) for h in args.horizons.split(',')]
    elif args.bull is not None and args.bear is not None:
        horizons = list(set([args.bull, args.bear]))
    else:
        horizons = ALL_HORIZONS

    # Check which horizons have production models
    available = sorted(df_models[df_models['coin'] == args.asset]['horizon'].unique())
    horizons = [h for h in horizons if h in available]
    missing = [h for h in (ALL_HORIZONS if not args.horizons else [int(h) for h in args.horizons.split(',')]) if h not in available]

    print(f"\n{'='*80}")
    print(f"  REGIME-SWITCHING BACKTEST")
    print(f"  Asset: {args.asset} | Period: {args.months}mo ({replay}h) | Conf: >={args.conf}%")
    print(f"  Horizons: {horizons}" + (f" (missing models: {missing})" if missing else ""))
    print(f"{'='*80}")

    # ── Step 1: Generate signals for all horizons ──
    signals_cache = {}
    model_descs = {}
    t0 = time.time()

    for h in horizons:
        print(f"\n  [{time.time()-t0:.0f}s] Generating {h}h signals ({replay}h replay)...")
        result = generate_horizon_signals(args.asset, h, replay, df_models)
        if result is None:
            print(f"    No model for {args.asset} {h}h — skipping")
            continue
        sigs, desc = result
        signals_cache[h] = sigs
        model_descs[h] = desc
        print(f"    -> {len(sigs)} signals | {desc}")

    # Filter to horizons with signals
    horizons = [h for h in horizons if h in signals_cache]
    if not horizons:
        print("\n  ERROR: No valid horizons with production models.")
        return

    # Build unified timeline
    all_dts = set()
    for sigs in signals_cache.values():
        all_dts.update(sigs.keys())
    all_dts = sorted(all_dts)
    print(f"\n  [{time.time()-t0:.0f}s] Signal generation complete. {len(all_dts)} candles.")

    # ── Step 2: Build regime indicators ──
    print(f"  Building regime indicators...")
    df_ind = build_indicators(args.asset)
    detectors = get_regime_detectors(df_ind, args.regimes)
    print(f"  {len(detectors)} regime detectors loaded.")

    # ── Step 3: Baselines — single-horizon strategies ──
    print(f"\n  {'─'*78}")
    print(f"  SINGLE-HORIZON BASELINES")
    print(f"  {'─'*78}")
    print(f"  {'Horizon':<10} {'Return':>8} {'B&H':>8} {'Alpha':>8} {'Trades':>7} {'WR':>6}")
    print(f"  {'─'*52}")

    baselines = []
    bh_ref = None
    for h in horizons:
        ret, tr, wr, _, bh, alpha = simulate(all_dts, signals_cache, lambda dt, hh=h: hh, args.conf)
        baselines.append((f'{h}h_only', ret, tr, wr, 0, bh, alpha, h, h))
        if bh_ref is None:
            bh_ref = bh
        print(f"  {h:>4}h     {ret:>+7.2f}% {bh:>+7.2f}% {alpha:>+7.2f}% {tr:>7d} {wr:>5.0f}%")

    if args.no_combos:
        baselines.sort(key=lambda x: -x[1])
        print(f"\n  >>> BEST: {baselines[0][0]} -> {baselines[0][1]:+.2f}%")
        return

    # ── Step 4: Regime-switching combos ──
    # Build all (bull_h, bear_h) pairs
    if args.bull is not None and args.bear is not None:
        pairs = [(args.bull, args.bear)]
    else:
        pairs = [(b, r) for b in horizons for r in horizons if b != r]

    print(f"\n  [{time.time()-t0:.0f}s] Testing {len(detectors)} regimes × {len(pairs)} horizon pairs = {len(detectors)*len(pairs)} combos...")

    print(f"\n  {'─'*78}")
    print(f"  REGIME-SWITCHING RESULTS")
    print(f"  {'─'*78}")

    all_results = list(baselines)  # include baselines in ranking

    for regime_name, is_bull_fn in detectors.items():
        regime_results = []
        for bull_h, bear_h in pairs:
            picker = lambda dt, fn=is_bull_fn, bh=bull_h, brh=bear_h: bh if fn(dt) else brh
            ret, tr, wr, sw, bh, alpha = simulate(all_dts, signals_cache, picker, args.conf)
            regime_results.append((regime_name, ret, tr, wr, sw, bh, alpha, bull_h, bear_h))

        # Sort this regime's results by return
        regime_results.sort(key=lambda x: -x[1])
        best = regime_results[0]
        all_results.append(best)

    # ── Step 5: Print results sorted by return ──
    all_results.sort(key=lambda x: -x[1])

    print(f"\n  {'─'*90}")
    print(f"  TOP {args.top} STRATEGIES (sorted by return)")
    print(f"  {'─'*90}")
    print(f"  {'#':>3} {'Strategy':<28} {'Bull→Bear':>10} {'Return':>8} {'B&H':>8} {'Alpha':>8} {'Trades':>7} {'WR':>6} {'Sw':>5}")
    print(f"  {'─'*90}")

    for i, (name, ret, tr, wr, sw, bh, alpha, bull_h, bear_h) in enumerate(all_results[:args.top]):
        if bull_h == bear_h:
            pair_str = f"{bull_h}h"
        else:
            pair_str = f"{bull_h}h→{bear_h}h"
        marker = " ***" if i == 0 else ""
        print(f"  {i+1:>3} {name:<28} {pair_str:>10} {ret:>+7.2f}% {bh:>+7.2f}% {alpha:>+7.2f}% {tr:>7d} {wr:>5.0f}% {sw:>5d}{marker}")

    # ── Step 6: Best per regime (show which horizon pair each regime prefers) ──
    print(f"\n  {'─'*90}")
    print(f"  BEST HORIZON PAIR PER REGIME")
    print(f"  {'─'*90}")
    print(f"  {'Regime':<28} {'Best Pair':>10} {'Return':>8} {'Alpha':>8} {'Trades':>7} {'WR':>6} {'Sw':>5}")
    print(f"  {'─'*90}")

    # Re-run to get best pair per regime
    regime_bests = []
    for regime_name, is_bull_fn in detectors.items():
        best_ret = -999
        best_row = None
        for bull_h, bear_h in pairs:
            picker = lambda dt, fn=is_bull_fn, bh=bull_h, brh=bear_h: bh if fn(dt) else brh
            ret, tr, wr, sw, bh, alpha = simulate(all_dts, signals_cache, picker, args.conf)
            if ret > best_ret:
                best_ret = ret
                best_row = (regime_name, ret, tr, wr, sw, bh, alpha, bull_h, bear_h)
        regime_bests.append(best_row)

    regime_bests.sort(key=lambda x: -x[1])
    for name, ret, tr, wr, sw, bh, alpha, bull_h, bear_h in regime_bests:
        pair_str = f"{bull_h}h→{bear_h}h"
        print(f"  {name:<28} {pair_str:>10} {ret:>+7.2f}% {alpha:>+7.2f}% {tr:>7d} {wr:>5.0f}% {sw:>5d}")

    # ── Step 7: Summary ──
    best = all_results[0]
    print(f"\n  {'='*90}")
    if best[7] == best[8]:
        print(f"  WINNER: {best[0]} ({best[7]}h) -> {best[1]:+.2f}% return, {best[5]:+.2f}% B&H, {best[6]:+.2f}% alpha")
    else:
        print(f"  WINNER: {best[0]} (bull={best[7]}h, bear={best[8]}h) -> {best[1]:+.2f}% return, {best[5]:+.2f}% B&H, {best[6]:+.2f}% alpha")
    print(f"          {best[2]} trades, {best[3]:.0f}% WR, {best[4]} switches")
    print(f"  {'='*90}")

    # Monthly breakdown for winner
    best_name, _, _, _, _, _, _, best_bull, best_bear = best
    if best_bull == best_bear:
        winner_picker = lambda dt, hh=best_bull: hh
    else:
        # Find the regime detector for the winner
        winner_detector = detectors.get(best_name)
        if winner_detector:
            winner_picker = lambda dt, fn=winner_detector, bh=best_bull, brh=best_bear: bh if fn(dt) else brh
        else:
            winner_picker = lambda dt, hh=best_bull: hh

    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly = simulate_monthly(all_dts, signals_cache, winner_picker, args.conf)

    if monthly:
        print(f"\n  MONTHLY BREAKDOWN:")
        print(f"  {'Month':<14} {'Return':>8} {'Trades':>7} {'WR':>6}")
        print(f"  {'─'*38}")
        for ym in sorted(monthly.keys()):
            m = monthly[ym]
            m_ret = (m['end_eq'] / m['start_eq'] - 1) * 100 if m['start_eq'] > 0 else 0
            m_wr = (m['wins'] / m['trades'] * 100) if m['trades'] > 0 else 0
            wr_str = f"{m_wr:.0f}%" if m['trades'] > 0 else "  -"
            print(f"  {month_names[ym[1]]} {ym[0]}    {m_ret:>+7.2f}% {m['trades']:>7d} {wr_str:>6}")
        print(f"  {'─'*38}")

    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
