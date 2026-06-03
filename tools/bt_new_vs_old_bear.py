"""
bt_new_vs_old_bear.py — 1-month backtest: NEW bear model (5h@80%) vs OLD (8h@65%)
================================================================================
One-off comparison driver. Reuses the engine's OWN walk-forward signal generator
(generate_signals) + maker-fee simulator (_simulate_with_threshold via
_backtest_one_config), so the only custom code is loading the two production
configs and reading the result at each one's production confidence.

Faithful to: model combo/window/gamma/features, horizon, BACKTEST_FEE_PER_LEG
(0.0005/leg maker), and closed-bar inference (backtests inherently use complete
bars — the forming-bar bug was live-only and cannot be reproduced here).
NOT modelled: hold-shields, rally cooldown, regime switching (single horizon —
valid because the period is bear-dominated and bull 6h@65 is identical in both
configs). This isolates the ONE lever that changed: the bear model.

Run:  python tools/bt_new_vs_old_bear.py            # default 720h = 1 month
      python tools/bt_new_vs_old_bear.py 1440        # 2 months
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading_system_ed import _backtest_one_config, BACKTEST_FEE_PER_LEG  # noqa: E402

REPLAY = int(sys.argv[1]) if len(sys.argv) > 1 else 720  # 720h = 30d
ASSET = "ETH"

# NEW = bear 5h @ 80% conf ; OLD = bear 8h @ 65% conf  (bull 6h@65% identical in both)
PLAN = [
    ("OLD bear 8h@65%", 8, 65),
    ("NEW bear 5h@80%", 5, 80),
]

prod = pd.read_csv("models/crypto_ed_production.csv")
prod = prod[prod["coin"] == ASSET]

print("=" * 72)
print(f"  1-MONTH BACKTEST — NEW vs OLD bear model  ({ASSET}, {REPLAY}h, "
      f"maker fee {BACKTEST_FEE_PER_LEG*100:.2f}%/leg)")
print("=" * 72)

summary = []
for label, horizon, prod_conf in PLAN:
    sub = prod[prod["horizon"] == horizon]
    if sub.empty:
        print(f"\n  [!] {label}: no {horizon}h config in production CSV — SKIP")
        continue
    row = sub.iloc[0]
    feats = [f.strip() for f in str(row["optimal_features"]).split(",")
             if f.strip() and f.strip() != "nan"]
    cfg = {
        "combo": row["best_combo"],
        "features": feats,
        "window": int(row["best_window"]),
        "gamma": float(row["gamma"]),
    }
    res = _backtest_one_config(ASSET, horizon, label, cfg, replay_hours=REPLAY)
    if res:
        sim = res.get(f"conf_{prod_conf}")
        summary.append((label, prod_conf, sim, res.get("buy_hold")))

print("\n" + "=" * 72)
print("  SUMMARY @ each config's PRODUCTION confidence")
print("=" * 72)
for label, conf, sim, bh in summary:
    if sim is None:
        print(f"  {label}: conf_{conf} not in sweep")
        continue
    print(f"  {label:<18} (conf>={conf}%): return={sim['return_pct']:+.2f}%  "
          f"trades={sim['trades']}  round_trips={sim['round_trips']}  "
          f"win_rate={sim['win_rate']:.0f}%"
          f"{'  [still in position]' if sim.get('still_invested') else ''}")
if summary and summary[0][2] and summary[-1][2]:
    delta = summary[-1][2]["return_pct"] - summary[0][2]["return_pct"]
    print(f"\n  NEW − OLD = {delta:+.2f} pp   (Buy&Hold over window: {summary[0][3]:+.2f}%)")
print("=" * 72)
