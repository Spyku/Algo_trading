"""Decompose where Mode T's pipeline loses alpha vs raw single-horizon model.

Background: Variant A_floorON_trimOFF (clean matrix, 2026-04-24) shows:
  - 5h model alone (refine #2):  +96.87%
  - 6h model alone (refine #1):  +72.34%
  - Mode T pipeline total:       +18.62%   (regime-switched + shield + gate)

Where do the missing ~78pp go? Run 6 isolated configurations to attribute.

Tests (each 60d sim, 5bps/leg fee):
  T1: 5h alone @ 90% conf, no regime/shield/gate          (re-validates +96.87%)
  T2: 5h alone @ 65% conf, no regime/shield/gate          (= conf-drop effect)
  T3: 5h alone @ 65% conf + bull shield ON                (= shield effect)
  T4: 5h alone @ 65% conf + bull rally gate               (= gate effect)
  T5: regime-switch 5h@65% (bull) / 6h@65% (bear), no shield/gate  (= regime gating)
  T6: full Mode T config (= matches matrix +18.62%)

Each test reuses generate_signals + a custom trade simulator.
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import csv

from crypto_trading_system_ed import generate_signals, BACKTEST_FEE_PER_LEG
from crypto_live_trader_ed import _evaluate_named_detector

# ─── load variant A's trained models from its tagged CSV ────────────────
PROD_CSV = 'models/crypto_ed_production_noprod_A_floorON_trimOFF.csv'

def _load_model_cfg(asset, horizon):
    with open(PROD_CSV) as f:
        for row in csv.DictReader(f):
            if row['coin']==asset and row['horizon']==str(horizon):
                return row
    return None

m5 = _load_model_cfg('ETH', 5)
m6 = _load_model_cfg('ETH', 6)
print(f"Loaded ETH 5h: combo={m5['best_combo']} w={m5['best_window']} g={m5['gamma']} ret_csv={m5['return_pct']}%")
print(f"Loaded ETH 6h: combo={m6['best_combo']} w={m6['best_window']} g={m6['gamma']} ret_csv={m6['return_pct']}%")

# ─── generate signals (1440h = 60d) for each horizon ────────────────────
def gen_for(model_row, horizon):
    feats = [f.strip() for f in model_row['optimal_features'].split(',') if f.strip()]
    return generate_signals(
        'ETH',
        model_row['best_combo'].split('+'),
        int(model_row['best_window']),
        replay_hours=1440,
        feature_override=feats,
        horizon=horizon,
        gamma=float(model_row['gamma']),
    )

print("\nGenerating 5h signals (~3-5 min)...")
sigs5 = gen_for(m5, 5)
print(f"  Got {len(sigs5)} 5h signals")
print("\nGenerating 6h signals (~3-5 min)...")
sigs6 = gen_for(m6, 6)
print(f"  Got {len(sigs6)} 6h signals")

# ─── trade simulator: configurable filters per test variant ─────────────
FEE = BACKTEST_FEE_PER_LEG  # 0.0005

def _simulate(sigs, conf_thr, shield_min_pnl=0, max_hold_h=999,
              rally_cfg=None, regime_other_sigs=None, regime_name=None):
    """Walk through signals, BUY/SELL/HOLD with config filters.
    Returns (final_return_pct, n_trades, win_rate_pct).

    regime_other_sigs/regime_name: if both provided, switch between this and
    other based on detector. Used for regime-switching variants.
    """
    cash = 100.0  # start with $100
    held = 0.0
    in_pos = False
    entry_px = 0
    hold_since_entry = 0
    cd = 0  # rally cooldown remaining
    trades_pnl = []

    closes = np.array([s['close'] for s in sigs])
    # Pre-compute rally returns for gate
    if rally_cfg:
        h_s, h_l, t_s, t_l, cd_h = rally_cfg
        def _rr(h):
            out = np.full(len(closes), np.nan)
            if h < len(closes):
                out[h:] = (closes[h:] / closes[:-h] - 1.0) * 100
            return out
        rs_arr = _rr(h_s); rl_arr = _rr(h_l)

    # Pre-compute regime per-bar if regime switching
    is_bull_per_bar = None
    if regime_other_sigs is not None:
        is_bull_per_bar = []
        # For each bar, evaluate sma24>sma100 detector on the price history up to that bar
        for i, s in enumerate(sigs):
            # Build a mini-df for the detector
            sub = pd.DataFrame({'close': closes[:i+1]})
            ib = _evaluate_named_detector('sma24>sma100', sub)
            is_bull_per_bar.append(bool(ib) if ib is not None else True)

    for i, s in enumerate(sigs):
        price = s['close']
        # Pick which signal to use this bar
        if is_bull_per_bar is not None:
            active_sig = sigs[i] if is_bull_per_bar[i] else regime_other_sigs[i]
        else:
            active_sig = s

        if in_pos:
            hold_since_entry += 1

        # Rally trigger check
        if rally_cfg and cd_h > 0:
            rs = rs_arr[i] if not np.isnan(rs_arr[i]) else 0
            rl = rl_arr[i] if not np.isnan(rl_arr[i]) else 0
            if rs >= t_s or rl >= t_l:
                cd = max(cd, cd_h)

        sig_action = active_sig.get('signal', 'HOLD')
        sig_conf = active_sig.get('confidence', 0) or 0

        if sig_action == 'BUY' and sig_conf >= conf_thr and not in_pos:
            if cd > 0:
                pass  # gate blocks BUY
            else:
                held = cash * (1 - FEE) / price
                cash = 0
                in_pos = True
                entry_px = price
                hold_since_entry = 0
        elif sig_action == 'SELL' and in_pos:
            cur_pnl_pct = (price / entry_px - 1) * 100
            override_expired = hold_since_entry >= max_hold_h
            shield_blocks = (shield_min_pnl > 0
                             and cur_pnl_pct < shield_min_pnl
                             and not override_expired)
            if not shield_blocks:
                cash = held * price * (1 - FEE)
                trades_pnl.append(cur_pnl_pct)
                held = 0
                in_pos = False
                hold_since_entry = 0

        if cd > 0:
            cd -= 1

    final = cash if not in_pos else held * sigs[-1]['close']
    ret_pct = (final / 100 - 1) * 100
    n = len(trades_pnl)
    wr = sum(1 for p in trades_pnl if p > 0) / max(n, 1) * 100
    return ret_pct, n, wr

# ─── run the 6 tests ──────────────────────────────────────────────────────
print("\n" + "="*76)
print("  PIPELINE DECOMPOSITION — 60d sim, 5bps/leg fee")
print("="*76)

# Bull gate config from variant A: rr8>=4.5% OR rr18>=4.5% cd=18h
GATE_A_BULL = (8, 18, 4.5, 4.5, 18)

tests = [
    ("T1: 5h@90% alone, no filters",
     dict(sigs=sigs5, conf_thr=90)),
    ("T2: 5h@65% alone, no filters",
     dict(sigs=sigs5, conf_thr=65)),
    ("T3: 5h@65% + shield (min_sell=0.5%, max_hold=10h)",
     dict(sigs=sigs5, conf_thr=65, shield_min_pnl=0.5, max_hold_h=10)),
    ("T4: 5h@65% + bull rally gate",
     dict(sigs=sigs5, conf_thr=65, rally_cfg=GATE_A_BULL)),
    ("T5: regime-switch 5h@65% (bull) / 6h@65% (bear), no shield/gate",
     dict(sigs=sigs5, conf_thr=65, regime_other_sigs=sigs6, regime_name='sma24>sma100')),
    ("T6: full Mode T config (regime + shield + gate)",
     dict(sigs=sigs5, conf_thr=65, shield_min_pnl=0.5, max_hold_h=10,
          rally_cfg=GATE_A_BULL, regime_other_sigs=sigs6, regime_name='sma24>sma100')),
]

print(f"\n{'Test':<60} {'Return':>10} {'Trades':>8} {'WinRate':>9}")
print("-"*88)
for label, kwargs in tests:
    ret, n, wr = _simulate(**kwargs)
    print(f"{label:<60} {ret:>+9.2f}% {n:>8} {wr:>8.1f}%")

print("\n" + "="*76)
print("  Reference points from matrix CSV (variant A_floorON_trimOFF, clean):")
print("    h5_return (refine#2 backtest):  +96.87%")
print("    h6_return (refine#1 backtest):  +72.34%")
print("    Mode T t_ref:                   +18.62%")
print("="*76)
