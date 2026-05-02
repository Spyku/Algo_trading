"""Shield variant comparison — standalone, does NOT modify prod code or config.

Loads the 60d tagged ETH signal cache and simulates 7 shield variants. Per-regime:
bull shield ON, bear shield OFF (as current config). Prints metrics for each.

Usage: python test_shield_variants.py
"""
import os
import sys
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

ENGINE = r'G:\Autres ordinateurs\My laptop\engine'
os.chdir(ENGINE)
sys.path.insert(0, ENGINE)

CACHE_PATH = os.path.join(ENGINE, 'data', 'eth_sl_signals_90d.pkl')
FEE = 0.0005  # 5 bps/leg realistic maker blend (see ed.py BACKTEST_FEE_PER_LEG)
MIN_SELL_PNL = 0.6  # %
MAX_HOLD = 12  # hours
BULL_CONF = 95
BEAR_CONF = 80

# Variants to test
VARIANTS = [
    # (name, kind, params)
    ('A_no_shield',              'off',         {}),
    ('B_current_shield',         'shield',      {}),
    ('C_QR_95_3h',               'shield_qr',   {'conf': 95, 'hours': 3}),
    ('D_QR_90_5h',               'shield_qr',   {'conf': 90, 'hours': 5}),
    ('E_persist_N2_C90',         'shield_pers', {'N': 2, 'conf': 90}),
    ('F_persist_N2_C95',         'shield_pers', {'N': 2, 'conf': 95}),
    ('G_persist_N3_C90',         'shield_pers', {'N': 3, 'conf': 90}),
    ('H_persist_N1_C95',         'shield_pers', {'N': 1, 'conf': 95}),  # "any single 95%+ SELL"
]


def load_signals():
    with open(CACHE_PATH, 'rb') as f:
        sigs = pickle.load(f)
    # Normalize + take last 60d
    for s in sigs:
        s['datetime'] = pd.Timestamp(s['datetime'])
    sigs.sort(key=lambda s: s['datetime'])
    end = sigs[-1]['datetime']
    lo = end - pd.Timedelta(days=60)
    return [s for s in sigs if s['datetime'] >= lo]


def simulate(sigs, kind, params):
    """Simulate the full strategy with a given shield variant.
    Returns dict with metrics."""
    cash = 1000.0
    qty = 0.0
    in_pos = False
    entry_px = 0.0
    hold = 0
    trades = []
    blocked = 0
    released_pnl = 0          # shield released by PnL target
    released_failsafe = 0     # released by 12h
    released_variant = 0      # released by the variant-specific rule
    sell_streak = 0           # consecutive high-conf SELL cycles

    conf_for_variant = params.get('conf', BULL_CONF)
    persist_N = params.get('N', 1)
    qr_hours = params.get('hours', 999)

    for i, s in enumerate(sigs):
        regime = s.get('regime', 'bull')
        price = float(s['close'])
        sig = s['signal']
        sconf = float(s.get('confidence', 0))
        conf_thr = BULL_CONF if regime == 'bull' else BEAR_CONF

        # Shield is per-regime: ON for bull, OFF for bear (current prod config)
        shield_active = (regime == 'bull')

        if in_pos:
            hold += 1
            # Track SELL streak
            if sig == 'SELL' and sconf >= conf_for_variant:
                sell_streak += 1
            elif sig == 'SELL':  # SELL but lower confidence
                pass  # don't reset — weak SELL doesn't break streak
            else:
                sell_streak = 0

        if sig == 'BUY' and sconf >= conf_thr and not in_pos:
            qty = cash * (1 - FEE) / price
            cash = 0
            in_pos = True
            entry_px = price
            hold = 0
            sell_streak = 0

        elif sig == 'SELL' and in_pos:
            cur_pnl = (price / entry_px - 1.0) * 100.0
            do_sell = False
            release_reason = None

            if not shield_active:
                # Shield OFF (bear) — sell immediately
                do_sell = True
                release_reason = 'shield_off'
            elif kind == 'off':
                # Variant A: no shield at all
                do_sell = True
                release_reason = 'no_shield'
            elif cur_pnl >= MIN_SELL_PNL:
                do_sell = True; release_reason = 'pnl_target'
                released_pnl += 1
            elif hold >= MAX_HOLD:
                do_sell = True; release_reason = 'failsafe'
                released_failsafe += 1
            else:
                # Variant-specific secondary release conditions
                if kind == 'shield_qr':
                    if hold <= qr_hours and sconf >= conf_for_variant:
                        do_sell = True; release_reason = 'qr'
                        released_variant += 1
                elif kind == 'shield_pers':
                    if sell_streak >= persist_N:
                        do_sell = True; release_reason = 'persistence'
                        released_variant += 1

                if not do_sell:
                    blocked += 1

            if do_sell:
                cash = qty * price * (1 - FEE)
                trades.append({
                    'pnl_pct': cur_pnl,
                    'hold_hours': hold,
                    'reason': release_reason,
                    'regime': regime,
                })
                qty = 0
                in_pos = False
                entry_px = 0
                hold = 0
                sell_streak = 0

    # Close any open position at the end at final price
    if in_pos:
        cash = qty * sigs[-1]['close'] * (1 - FEE)
        trades.append({
            'pnl_pct': (sigs[-1]['close'] / entry_px - 1.0) * 100.0,
            'hold_hours': hold,
            'reason': 'end_of_window',
            'regime': 'unknown',
        })

    ret_pct = (cash / 1000.0 - 1.0) * 100.0
    n_trades = len(trades)
    winners = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = (winners / n_trades * 100) if n_trades else 0
    avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
    worst = min([t['pnl_pct'] for t in trades], default=0)
    best = max([t['pnl_pct'] for t in trades], default=0)

    return dict(
        return_pct=ret_pct,
        trades=n_trades,
        win_rate=wr,
        avg_pnl=avg_pnl,
        worst_trade=worst,
        best_trade=best,
        blocked=blocked,
        rel_pnl=released_pnl,
        rel_failsafe=released_failsafe,
        rel_variant=released_variant,
    )


def main():
    print("=" * 100)
    print("  ETH 60d Shield Variant Comparison — standalone, no config writes")
    print("=" * 100)

    sigs = load_signals()
    bull = sum(1 for s in sigs if s.get('regime') == 'bull')
    bear = sum(1 for s in sigs if s.get('regime') == 'bear')
    print(f"  Signals: {len(sigs)}  (bull={bull}, bear={bear})")
    print(f"  Window:  {sigs[0]['datetime']} -> {sigs[-1]['datetime']}")
    print(f"  Policy:  bull@{BULL_CONF}% shield=ON | bear@{BEAR_CONF}% shield=OFF | "
          f"MIN_SELL_PNL={MIN_SELL_PNL}% | MAX_HOLD={MAX_HOLD}h | FEE=0%")
    print()

    rows = []
    for name, kind, params in VARIANTS:
        r = simulate(sigs, kind, params)
        r['name'] = name
        rows.append(r)

    # Print summary table
    print(f"  {'Variant':<22}{'Ret%':>8}{'Trades':>8}{'WR%':>6}"
          f"{'AvgPnL':>9}{'Worst':>8}{'Best':>8}"
          f"{'Blocked':>9}{'RelPnl':>8}{'RelFS':>7}{'RelVar':>8}")
    print(f"  {'-'*22}{'-'*8}{'-'*8}{'-'*6}{'-'*9}{'-'*8}{'-'*8}{'-'*9}{'-'*8}{'-'*7}{'-'*8}")
    for r in rows:
        print(f"  {r['name']:<22}{r['return_pct']:>+7.2f}%{r['trades']:>8}"
              f"{r['win_rate']:>5.0f}%{r['avg_pnl']:>+8.2f}%{r['worst_trade']:>+7.2f}%"
              f"{r['best_trade']:>+7.2f}%{r['blocked']:>9}{r['rel_pnl']:>8}"
              f"{r['rel_failsafe']:>7}{r['rel_variant']:>8}")

    print()
    print("  Legend:")
    print("    Blocked  = shield refused SELL (no variant-specific release fired either)")
    print("    RelPnl   = shield released because PnL target hit")
    print("    RelFS    = shield released by 12h failsafe")
    print("    RelVar   = shield released by the variant's additional rule "
          "(QR or persistence)")


if __name__ == '__main__':
    main()
