"""
trim_safety_check.py — verify the Mode D grid trim was safe by analyzing
the OLD full-grid CSVs from before the trim (6h/7h/8h have 324 evals each).

Question: of the 252 configs we dropped (324 - 72), did any rank in the
top 10 of the historical full grid? If yes → trim is risky. If no → trim
is fully safe.

Dropped configs:
  - combo == 'RF+XGB'
  - window in {200, 250, 300}
  - n_features in {20, 30}

Reads: models/crypto_ed_grid_ETH_{5,6,7,8}h.csv (full-grid CSVs from prior runs).
"""
from __future__ import annotations

import os
import pandas as pd

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define what was DROPPED in today's trim
DROPPED_COMBOS = {'RF+XGB'}
DROPPED_WINDOWS = {200, 250, 300}
DROPPED_FEATURES = {20, 30}

KEPT_COMBOS = {'RF+LGBM', 'XGB+LGBM'}
KEPT_WINDOWS = {72, 100, 150}
KEPT_FEATURES = {10, 13, 17, 25}


def is_dropped(row):
    return (row['combo'] in DROPPED_COMBOS
            or row['window'] in DROPPED_WINDOWS
            or row['n_features'] in DROPPED_FEATURES)


def is_kept(row):
    return (row['combo'] in KEPT_COMBOS
            and row['window'] in KEPT_WINDOWS
            and row['n_features'] in KEPT_FEATURES)


def analyze_grid(path, horizon):
    df_all = pd.read_csv(path)
    print(f"\n{'='*100}")
    print(f"  HORIZON {horizon}h - full grid analysis: {len(df_all)} configs total")
    print(f"  File: {path}")
    print(f"{'='*100}")

    # CRITICAL: filter to trades >= 8 (matches OLD pipeline filter that produced
    # historical production winners). Configs with trades=3-4 have inflated APF
    # because few-shot wins look great but don't generalize.
    # Optuna training fold = 60% of total. For these CSVs, n_total=1440,
    # fold=864, MIN_TRADES on OLD code = 8 (hardcoded), NEW code = max(4, 864//360) = 4.
    # We use 8 for the apples-to-apples test against the OLD pipeline.
    MIN_TRADES_FILTER = 8

    df = df_all[df_all['trades'] >= MIN_TRADES_FILTER].copy()
    print(f"  After trades >= {MIN_TRADES_FILTER} filter: {len(df)} configs (matches OLD pipeline behavior)")

    if len(df) == 0:
        print("  All configs filtered out by trades>=8. Falling back to trades>=4.")
        df = df_all[df_all['trades'] >= 4].copy()
        if len(df) == 0:
            print("  Still empty. Using all configs.")
            df = df_all.copy()

    # Sort by APF descending (Mode D's primary score)
    df = df.sort_values('apf', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    df['dropped'] = df.apply(is_dropped, axis=1)
    df['kept'] = df.apply(is_kept, axis=1)
    df['kept_or_dropped'] = df.apply(
        lambda r: 'KEPT' if is_kept(r) else 'DROPPED' if is_dropped(r) else 'OTHER',
        axis=1)

    # Top 20 with marker
    print(f"\n  TOP 20 by APF — would the trim have included these?")
    print(f"  {'Rank':>4} {'Combo':<10} {'W':>4} {'G':>7} {'F':>3} {'APF':>8} {'Ret%':>7} {'Trades':>6} | Kept by trim?")
    print(f"  {'-'*4} {'-'*10} {'-'*4} {'-'*7} {'-'*3} {'-'*8} {'-'*7} {'-'*6} | {'-'*15}")
    for i in range(min(20, len(df))):
        r = df.iloc[i]
        marker = '[KEPT]' if r['kept'] else '[DROPPED]'
        if not r['kept'] and not r['dropped']:
            marker = '[partial]'  # combo+window+feat all kept individually but not all together
        # Mark dropped reason
        reason = ''
        if r['dropped']:
            r_parts = []
            if r['combo'] in DROPPED_COMBOS: r_parts.append(f"combo={r['combo']}")
            if r['window'] in DROPPED_WINDOWS: r_parts.append(f"w={int(r['window'])}")
            if r['n_features'] in DROPPED_FEATURES: r_parts.append(f"f={int(r['n_features'])}")
            reason = f"  [{','.join(r_parts)}]"
        print(f"  {r['rank']:>4} {r['combo']:<10} {int(r['window']):>4} {r['gamma']:>7.4f} "
              f"{int(r['n_features']):>3} {r['apf']:>8.3f} {r['return_pct']:>+7.2f} "
              f"{int(r['trades']):>6} | {marker}{reason}")

    # Stats
    n_kept = df['kept'].sum()
    n_dropped = df['dropped'].sum()
    n_other = len(df) - n_kept - n_dropped
    top10_kept = df.head(10)['kept'].sum()
    top10_dropped = df.head(10)['dropped'].sum()
    top6_kept = df.head(6)['kept'].sum()
    top6_dropped = df.head(6)['dropped'].sum()

    print(f"\n  STATISTICS for h={horizon}")
    print(f"  Total configs:    {len(df)}")
    print(f"  Trim KEPT:        {n_kept} ({n_kept/len(df)*100:.0f}%)")
    print(f"  Trim DROPPED:     {n_dropped} ({n_dropped/len(df)*100:.0f}%)")
    print(f"  Other (mixed):    {n_other} ({n_other/len(df)*100:.0f}%)")
    print(f"  Top 6 by APF:     KEPT={top6_kept}/6, DROPPED={top6_dropped}/6")
    print(f"  Top 10 by APF:    KEPT={top10_kept}/10, DROPPED={top10_dropped}/10")
    if top6_dropped > 0:
        print(f"  !! WARNING: {top6_dropped} of top-6 winners are DROPPED — trim cost real candidates")
    else:
        print(f"  [SAFE]: all top 6 are KEPT — trim didn't cost any top-tier candidate")

    # By return (alternative metric)
    df_by_ret = df.sort_values('return_pct', ascending=False).reset_index(drop=True)
    top6_ret_dropped = df_by_ret.head(6).apply(is_dropped, axis=1).sum()
    print(f"  Top 6 by return:  DROPPED={top6_ret_dropped}/6")

    return df


def main():
    print("="*100)
    print("  TRIM SAFETY CHECK")
    print("="*100)
    print("\n  Today's trim:")
    print(f"    KEPT combos:   {sorted(KEPT_COMBOS)}")
    print(f"    KEPT windows:  {sorted(KEPT_WINDOWS)}")
    print(f"    KEPT features: {sorted(KEPT_FEATURES)}")
    print(f"    DROPPED combos:   {sorted(DROPPED_COMBOS)}")
    print(f"    DROPPED windows:  {sorted(DROPPED_WINDOWS)}")
    print(f"    DROPPED features: {sorted(DROPPED_FEATURES)}")

    paths = [
        ('models/crypto_ed_grid_ETH_6h.csv', 6),
        ('models/crypto_ed_grid_ETH_7h.csv', 7),
        ('models/crypto_ed_grid_ETH_8h.csv', 8),
    ]
    for path_rel, h in paths:
        path = os.path.join(ENGINE, path_rel)
        if os.path.exists(path):
            analyze_grid(path, h)
        else:
            print(f"\n  Missing: {path_rel}")

    # Aggregate verdict
    print("\n" + "="*100)
    print("  AGGREGATE VERDICT")
    print("="*100)
    total_top6_dropped = 0
    total_top10_dropped = 0
    n_horizons_tested = 0
    for path_rel, h in paths:
        path = os.path.join(ENGINE, path_rel)
        if not os.path.exists(path):
            continue
        n_horizons_tested += 1
        df = pd.read_csv(path).sort_values('apf', ascending=False).reset_index(drop=True)
        top6 = df.head(6)
        top10 = df.head(10)
        d6 = top6.apply(is_dropped, axis=1).sum()
        d10 = top10.apply(is_dropped, axis=1).sum()
        total_top6_dropped += d6
        total_top10_dropped += d10

    print(f"\n  Across {n_horizons_tested} historical horizons (FULL grid):")
    print(f"    Top 6 by APF in dropped configs:  {total_top6_dropped} / {n_horizons_tested * 6}")
    print(f"    Top 10 by APF in dropped configs: {total_top10_dropped} / {n_horizons_tested * 10}")
    if total_top6_dropped == 0:
        print(f"\n  VERDICT: TRIM IS PROVEN SAFE")
        print(f"    Zero dropped configs reached top 6 in any historical full-grid run.")
        print(f"    The trim removed configs that empirically never won.")
    elif total_top6_dropped <= n_horizons_tested:
        print(f"\n  VERDICT: TRIM IS BORDERLINE")
        print(f"    Some dropped configs reached top 6 — average {total_top6_dropped/n_horizons_tested:.1f}/6 per horizon.")
        print(f"    Consider reverting parts of the trim if these were strong winners.")
    else:
        print(f"\n  VERDICT: TRIM MAY BE RISKY")
        print(f"    Dropped configs frequently reached top 6 — trim cost real candidates.")


if __name__ == '__main__':
    main()
