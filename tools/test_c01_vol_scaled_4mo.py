"""tools/test_c01_vol_scaled_4mo.py — C01 4-month validation of vol-scaled horizons.

Background:
  The 2-month run (2026-04-19) of `tools/test_vol_scaled_horizon.py` produced:
    Best variant: vol_2band low→8h high→6h @90% = +33.82% / 46 trades / 65% WR
    Beat tsmom_672h regime baseline by +5.02pp (and beat all single-horizon
    baselines).
  CLAUDE.md research queue notes the only outstanding blocker is a 4-month
  confirmation; if it passes, this is a config-only ship (replace tsmom_672h
  with vol_2band as the regime detector).

What this does:
  - Reuses the existing `test_vol_scaled_horizon.main()` logic verbatim.
  - Monkey-patches REPLAY from 1440 (2mo) to 2880 (4mo) before invocation.
  - The vol-scaled simulation, baselines, and printout all run on the longer
    window. No model retraining (uses current prod models from
    crypto_ed_production.csv).

Decision rule:
  Best vol-scaled variant beats the live regime baseline (currently
  `tsmom_672h bull=6h shield=OFF / bear=8h shield=OFF` per 2026-05-06
  promotion) by ≥+5pp on the 4-month return AND keeps WR ≥ 60% AND
  doesn't materially worsen drawdown → ship-candidate. Otherwise null.

Run:
  python tools/test_c01_vol_scaled_4mo.py

Runtime: ~30-45 min on laptop (4× more bars to simulate than 2mo).
"""
from __future__ import annotations

import os
import sys

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)
sys.path.insert(0, os.path.join(ENGINE, 'tools'))

import test_vol_scaled_horizon as vsh


def main():
    # Monkey-patch the REPLAY constant in the imported module BEFORE main() runs.
    # The simulator reads vsh.REPLAY for both signal generation and B&H window.
    print('=' * 70)
    print('  C01 — VOLATILITY-SCALED HORIZONS: 4-MONTH VALIDATION')
    print('=' * 70)
    print(f'  Original 2mo (REPLAY=1440) result: vol_2band low→8h high→6h @90%')
    print(f'                                     +33.82% / 46 trades / 65% WR')
    print(f'                                     beat tsmom_672h baseline by +5.02pp')
    print(f'  Now testing REPLAY=2880 (4mo) for structural-consistency confirmation.')
    print(f'  Decision: best vol-scaled variant ≥+5pp over current live baseline AND')
    print(f'            WR ≥ 60% → ship as detector. Otherwise null.')
    print('=' * 70)
    print()

    _orig = vsh.REPLAY
    vsh.REPLAY = 2880
    print(f'  [override] REPLAY: {_orig}h -> {vsh.REPLAY}h (4 months)')
    print()
    try:
        vsh.main()
    finally:
        vsh.REPLAY = _orig

    print()
    print('=' * 70)
    print('  C01 COMPLETE — apply decision rule above to the printed results.')
    print('=' * 70)
    print('  Reference: current LIVE config (per regime_config_ed.json after 2026-05-06')
    print('  promotion) is `tsmom_672h bull=6h@75% shield=OFF / bear=8h@65% shield=OFF`')
    print('  with bull/bear rally-cooldown gates active. The simulator above does NOT')
    print('  apply gates — its baseline labelled "tsmom_672h bull=6h bear=8h @90%" uses')
    print('  conf=90, no shield, no gate. Compare against that reference, not against')
    print('  full live performance numbers (which include shield+gate alpha).')


if __name__ == '__main__':
    main()
