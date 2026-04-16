"""One-off: does it matter whether V7 gate is added AFTER strategy selection vs DURING?

Path A (current production order):  pick best strategy params WITHOUT gate, then bolt on V7.
Path B (joint optimisation):         pick best strategy params WITH gate active throughout.

Both evaluated on the same OOS test slice.
Sweep happens at the post-model layer (the only knobs we can move on a fixed signal cache):
    min_sell_pnl_pct, max_hold_hours, conf_threshold offset.

Run with the trader's venv:
    "C:\\algo_trading\\venv\\Scripts\\python.exe" compare_chain_order.py
"""
from __future__ import annotations

import os
import pickle
import itertools
import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
SIG_90D    = os.path.join(ENGINE_DIR, 'data', 'eth_sl_signals_90d.pkl')

TRADING_FEE = 0.0011
V7 = dict(h_short=8, h_long=36, t_short=3.0, t_long=5.5, cd_hours=30)

TRAIN_DAYS = 60
TEST_DAYS  = 30

# Walk-forward params (rolling): fold k uses last (TRAIN+TEST)-(k*STEP) days
WF_TRAIN_DAYS = 30
WF_TEST_DAYS  = 10
WF_STEP_DAYS  = 10
WF_MAX_FOLDS  = 6

# Sweep grid — small enough to run in <1 min
GRID_MIN_SELL = [0.0, 0.25, 0.5, 0.75, 1.0]
GRID_MAX_HOLD = [4, 6, 8, 10, 14, 20]
GRID_CONF_OFF = [-10, -5, 0, 5, 10]


def load_signals():
    with open(SIG_90D, 'rb') as f:
        signals = pickle.load(f)
    for s in signals:
        t = pd.Timestamp(s['datetime'])
        s['datetime'] = t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    return signals


def build_rr(signals, horizons):
    closes = np.array([s['close'] for s in signals], dtype=float)
    out = {}
    for h in horizons:
        rr = np.full(len(signals), np.nan)
        rr[h:] = (closes[h:] / closes[:-h] - 1.0) * 100.0
        out[h] = rr
    return out


def simulate(signals, rr_arrs, *, min_sell_pnl_pct, max_hold_hours, conf_offset, gate_on):
    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry_px = 0.0
    hold_hours = 0
    trades = 0
    skipped = 0
    cd = 0
    equity_curve = [1000.0]
    n = len(signals)

    h_s, h_l = V7['h_short'], V7['h_long']
    t_s, t_l = V7['t_short'], V7['t_long']
    cd_h     = V7['cd_hours']
    rs_arr = rr_arrs.get(h_s) if gate_on else None
    rl_arr = rr_arrs.get(h_l) if gate_on else None

    for i in range(n):
        s = signals[i]
        price = s['close']
        sig = s['signal']
        conf = s['confidence']
        thr = s['conf_threshold'] + conf_offset

        # gate trigger fires every tick (matches production)
        if gate_on:
            rs = rs_arr[i]
            rl = rl_arr[i]
            if (rs == rs and rs >= t_s) or (rl == rl and rl >= t_l):
                cd = max(cd, cd_h)

        equity = cash + qty * price if in_pos else cash
        equity_curve.append(equity)

        if sig == 'BUY' and conf >= thr and not in_pos:
            if cd > 0:
                skipped += 1
            else:
                fill_px = signals[i + 1]['close'] if i + 1 < n else price
                qty = cash * (1 - TRADING_FEE) / fill_px
                cash = 0.0
                in_pos = True
                entry_px = fill_px
                hold_hours = 0

        elif sig == 'SELL' and in_pos:
            fill_px = signals[i + 1]['close'] if i + 1 < n else price
            cur_pnl_pct = (fill_px / entry_px - 1.0) * 100.0
            if cur_pnl_pct >= min_sell_pnl_pct or hold_hours >= max_hold_hours:
                cash = qty * fill_px * (1 - TRADING_FEE)
                trades += 1
                in_pos = False
                qty = 0.0
                entry_px = 0.0
                hold_hours = 0

        if in_pos:
            hold_hours += 1
        if cd > 0:
            cd -= 1

    if in_pos:
        cash = qty * signals[-1]['close'] * (1 - TRADING_FEE)
        trades += 1

    pnl = (cash / 1000.0 - 1.0) * 100.0
    ec = np.array(equity_curve)
    peak = np.maximum.accumulate(ec)
    dd = ((ec - peak) / peak).min() * 100.0 if len(ec) else 0.0
    return dict(pnl=pnl, dd=dd, trades=trades, skipped=skipped)


def sweep(signals, rr_arrs, *, gate_on):
    best = None
    for ms, mh, co in itertools.product(GRID_MIN_SELL, GRID_MAX_HOLD, GRID_CONF_OFF):
        r = simulate(signals, rr_arrs,
                     min_sell_pnl_pct=ms, max_hold_hours=mh,
                     conf_offset=co, gate_on=gate_on)
        # rank by PnL, tiebreak by less-negative DD
        score = (r['pnl'], -r['dd'])
        if best is None or score > best['score']:
            best = dict(score=score, ms=ms, mh=mh, co=co, **r)
    return best


def slice_by_index(signals, start_idx, end_idx, horizons):
    sub = signals[start_idx:end_idx]
    rr  = build_rr(sub, horizons)
    return sub, rr


def run_one_split(signals, horizons, train_start, test_start, end_t, label=''):
    """Run Path A and Path B on one (train, test) split. Returns dict per path."""
    train_idx = next((i for i, s in enumerate(signals) if s['datetime'] >= train_start), 0)
    test_idx  = next((i for i, s in enumerate(signals) if s['datetime'] >= test_start), 0)
    end_idx   = next((i for i, s in enumerate(signals) if s['datetime'] >  end_t), len(signals))

    train_sigs, train_rr = slice_by_index(signals, train_idx, test_idx, horizons)
    test_sigs,  test_rr  = slice_by_index(signals, test_idx,  end_idx,  horizons)
    if len(train_sigs) < 50 or len(test_sigs) < 20:
        return None

    a_pick = sweep(train_sigs, train_rr, gate_on=False)
    a_test = simulate(test_sigs, test_rr,
                      min_sell_pnl_pct=a_pick['ms'], max_hold_hours=a_pick['mh'],
                      conf_offset=a_pick['co'], gate_on=True)
    b_pick = sweep(train_sigs, train_rr, gate_on=True)
    b_test = simulate(test_sigs, test_rr,
                      min_sell_pnl_pct=b_pick['ms'], max_hold_hours=b_pick['mh'],
                      conf_offset=b_pick['co'], gate_on=True)
    return dict(label=label,
                train_n=len(train_sigs), test_n=len(test_sigs),
                train_start=train_sigs[0]['datetime'], test_start=test_sigs[0]['datetime'],
                test_end=test_sigs[-1]['datetime'],
                a_pick=(a_pick['ms'], a_pick['mh'], a_pick['co']),
                b_pick=(b_pick['ms'], b_pick['mh'], b_pick['co']),
                a_test=a_test, b_test=b_test)


def walk_forward(signals, horizons):
    print("=" * 78)
    print("  ROLLING WALK-FORWARD (Path A vs Path B)")
    print(f"  Train={WF_TRAIN_DAYS}d, Test={WF_TEST_DAYS}d, Step={WF_STEP_DAYS}d, MaxFolds={WF_MAX_FOLDS}")
    print("=" * 78)

    end_t = signals[-1]['datetime']
    folds = []
    for k in range(WF_MAX_FOLDS):
        test_end   = end_t - pd.Timedelta(days=k * WF_STEP_DAYS)
        test_start = test_end - pd.Timedelta(days=WF_TEST_DAYS)
        train_start = test_start - pd.Timedelta(days=WF_TRAIN_DAYS)
        if train_start < signals[0]['datetime']:
            break
        folds.append((train_start, test_start, test_end))
    folds.reverse()  # chronological order

    print(f"\n  {'fold':<5}{'train_start':<22}{'test':<43}"
          f"{'Apick':<18}{'Bpick':<18}{'A.pnl':>8}{'B.pnl':>8}{'dPnL':>8}{'dDD':>8}")
    print("  " + "-" * 138)

    results = []
    for k, (ts, vs, ve) in enumerate(folds, 1):
        r = run_one_split(signals, horizons, ts, vs, ve, label=f'F{k}')
        if r is None:
            continue
        results.append(r)
        ap, bp = r['a_pick'], r['b_pick']
        a, b = r['a_test'], r['b_test']
        dpnl = b['pnl'] - a['pnl']; ddd = b['dd'] - a['dd']
        print(f"  {r['label']:<5}{str(r['train_start'])[:19]:<22}"
              f"{str(r['test_start'])[:19]+'..'+str(r['test_end'])[5:10]:<43}"
              f"{str(ap):<18}{str(bp):<18}"
              f"{a['pnl']:>+8.2f}{b['pnl']:>+8.2f}{dpnl:>+8.2f}{ddd:>+8.2f}")

    if not results:
        print("\n  no usable folds"); return

    a_pnls = np.array([r['a_test']['pnl'] for r in results])
    b_pnls = np.array([r['b_test']['pnl'] for r in results])
    a_dds  = np.array([r['a_test']['dd']  for r in results])
    b_dds  = np.array([r['b_test']['dd']  for r in results])
    same   = sum(1 for r in results if r['a_pick'] == r['b_pick'])
    b_wins = int((b_pnls > a_pnls).sum())
    a_wins = int((a_pnls > b_pnls).sum())
    ties   = len(results) - b_wins - a_wins

    print("\n" + "=" * 78)
    print("  AGGREGATE (n={})".format(len(results)))
    print("=" * 78)
    print(f"  Path A test PnL: mean={a_pnls.mean():+.2f}%  median={np.median(a_pnls):+.2f}%  "
          f"sum={a_pnls.sum():+.2f}%  worst_dd={a_dds.min():+.2f}%")
    print(f"  Path B test PnL: mean={b_pnls.mean():+.2f}%  median={np.median(b_pnls):+.2f}%  "
          f"sum={b_pnls.sum():+.2f}%  worst_dd={b_dds.min():+.2f}%")
    print(f"  Folds where same pick: {same}/{len(results)}")
    print(f"  B wins: {b_wins}   A wins: {a_wins}   ties: {ties}")

    if same == len(results):
        print("\n  -> ordering does NOT matter: same picks every fold.")
    elif b_wins >= a_wins + 2 and b_pnls.mean() - a_pnls.mean() > 0.5:
        print("\n  -> Path B (joint) consistently better -> consider promoting joint search.")
    elif a_wins >= b_wins + 2:
        print("\n  -> Path A (current G-last) better -> keep current order.")
    else:
        print("\n  -> mixed -> noise-level, keep current G-last (cheaper).")


def main():
    print("=" * 78)
    print("  CHAIN-ORDER ONE-OFF: G-last (current) vs G-during")
    print(f"  Sweep: min_sell {GRID_MIN_SELL}, max_hold {GRID_MAX_HOLD}, conf_off {GRID_CONF_OFF}")
    print(f"  Train = {TRAIN_DAYS}d, Test = {TEST_DAYS}d (chronological)")
    print("=" * 78)

    signals = load_signals()
    horizons = [V7['h_short'], V7['h_long']]
    end_t = signals[-1]['datetime']
    test_start  = end_t - pd.Timedelta(days=TEST_DAYS)
    train_start = end_t - pd.Timedelta(days=TRAIN_DAYS + TEST_DAYS)

    train_idx = next((i for i, s in enumerate(signals) if s['datetime'] >= train_start), 0)
    test_idx  = next((i for i, s in enumerate(signals) if s['datetime'] >= test_start), 0)
    end_idx   = len(signals)

    train_sigs, train_rr = slice_by_index(signals, train_idx, test_idx, horizons)
    test_sigs,  test_rr  = slice_by_index(signals, test_idx,  end_idx,  horizons)

    print(f"\n  train: {train_sigs[0]['datetime']} -> {train_sigs[-1]['datetime']}  ({len(train_sigs)} sigs)")
    print(f"  test:  {test_sigs[0]['datetime']} -> {test_sigs[-1]['datetime']}  ({len(test_sigs)} sigs)")

    # ---------- Path A: pick best WITHOUT gate, then evaluate WITH gate on test ----------
    print("\n  --- Path A (current): pick on train w/o gate, test w/ gate ---")
    a_pick = sweep(train_sigs, train_rr, gate_on=False)
    print(f"    train pick: ms={a_pick['ms']} mh={a_pick['mh']} co={a_pick['co']:+d}  "
          f"pnl={a_pick['pnl']:+.2f}% dd={a_pick['dd']:+.2f}% trades={a_pick['trades']}")
    a_test = simulate(test_sigs, test_rr,
                      min_sell_pnl_pct=a_pick['ms'], max_hold_hours=a_pick['mh'],
                      conf_offset=a_pick['co'], gate_on=True)
    a_test_nogate = simulate(test_sigs, test_rr,
                      min_sell_pnl_pct=a_pick['ms'], max_hold_hours=a_pick['mh'],
                      conf_offset=a_pick['co'], gate_on=False)
    print(f"    test (no gate): pnl={a_test_nogate['pnl']:+.2f}% dd={a_test_nogate['dd']:+.2f}% "
          f"trades={a_test_nogate['trades']}")
    print(f"    test (+gate):   pnl={a_test['pnl']:+.2f}% dd={a_test['dd']:+.2f}% "
          f"trades={a_test['trades']} skipped={a_test['skipped']}")

    # ---------- Path B: pick best WITH gate, evaluate WITH gate on test ----------
    print("\n  --- Path B (joint): pick on train w/ gate, test w/ gate ---")
    b_pick = sweep(train_sigs, train_rr, gate_on=True)
    print(f"    train pick: ms={b_pick['ms']} mh={b_pick['mh']} co={b_pick['co']:+d}  "
          f"pnl={b_pick['pnl']:+.2f}% dd={b_pick['dd']:+.2f}% trades={b_pick['trades']} "
          f"skipped={b_pick['skipped']}")
    b_test = simulate(test_sigs, test_rr,
                      min_sell_pnl_pct=b_pick['ms'], max_hold_hours=b_pick['mh'],
                      conf_offset=b_pick['co'], gate_on=True)
    print(f"    test (+gate):   pnl={b_test['pnl']:+.2f}% dd={b_test['dd']:+.2f}% "
          f"trades={b_test['trades']} skipped={b_test['skipped']}")

    # ---------- Verdict ----------
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    same_params = (a_pick['ms'], a_pick['mh'], a_pick['co']) == (b_pick['ms'], b_pick['mh'], b_pick['co'])
    delta_pnl = b_test['pnl'] - a_test['pnl']
    delta_dd  = b_test['dd']  - a_test['dd']
    print(f"  Same train pick? {same_params}")
    print(f"  Test PnL d(B-A): {delta_pnl:+.2f}%   Test DD d(B-A): {delta_dd:+.2f}%")
    if same_params:
        print("  -> ordering does NOT matter on this data: gate is independent of strategy pick.")
    elif delta_pnl > 0.5 and delta_dd >= -0.5:
        print("  -> Path B (joint) wins meaningfully -> consider promoting joint search.")
    elif delta_pnl < -0.5:
        print("  -> Path A (current) wins -> keep G-last.")
    else:
        print("  -> noise-level difference -> keep current G-last (less search overhead).")


if __name__ == '__main__':
    import sys
    if '--single' in sys.argv:
        main()
    else:
        signals = load_signals()
        horizons = [V7['h_short'], V7['h_long']]
        walk_forward(signals, horizons)
