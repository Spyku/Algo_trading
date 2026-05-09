"""tools/test_c05_c06_only.py — Standalone runner for the C05/C06 post-hoc
rerank, isolated from the C04-C08 / desktop-5ideas wrappers.

C05 = CVaR_5% objective scoring. C06 = Sortino objective scoring. Both are
post-hoc reranks of the existing baseline grid CSV's top-15 APF candidates,
re-ran with per-trade-return capture so we can compute CVaR + Sortino.

Why this file:
  - test_c04_to_c08_runner.run_c05_c06() is the actual implementation but its
    main() runs C04 + C05 + C06 + C07 + C08 sequentially. We just want C05/C06.
  - On the 2026-05-07 overnight run, C05/C06 raised OSError(22) "device doesn't
    exist" — likely GPU contention with the concurrent C50 HRST. Now that C50
    has finished and the script has been hardened to force CPU LGBM, this
    wrapper kicks off just the rerank in a clean process.

Run:
  python tools/test_c05_c06_only.py

Output:
  output/c05_c06_rerank_<ts>.csv             — joint rerank table
  output/c05_c06_per_trade_<ts>.csv          — per-trade audit data
  logs/c04_to_c08_summary_<ts>.txt           — verdict summary

Decision (apply manually after run completes):
  - Does CVaR_5% or Sortino rerank pick a materially DIFFERENT top-3 from APF
    AND with similar returns AND lower tail-risk (max_dd, cvar_5pct)?
    YES across ≥2 horizons -> ship-candidate (1-line scoring change in engine).
    NO / identical to APF -> null. Mark C05/C06 SHELVED-by-pattern alongside
    C13 + C50.
"""
from __future__ import annotations

import os
import sys

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)
sys.path.insert(0, os.path.join(ENGINE, 'tools'))

from test_c04_to_c08_runner import run_c05_c06


def main():
    print('=' * 100)
    print('  C05 + C06 STANDALONE RUNNER')
    print('=' * 100)
    print('  CPU LGBM forced inside backtester (defensive vs OSError 22 GPU contention).')
    print('  Walk-forward per-trade backtest of top-15 APF candidates per horizon (5,6,7,8).')
    print('  Estimated runtime: 60-90 min on Desktop CPU.')
    print()
    result = run_c05_c06()
    print()
    print('=' * 100)
    print('  C05/C06 STANDALONE: COMPLETE')
    print('=' * 100)
    print(f'  Verdict: {result.get("verdict", "ERROR")}')
    if 'rerank_path' in result:
        print(f'  Rerank table:    {result["rerank_path"]}')
        print(f'  Per-trade audit: {result["audit_path"]}')


if __name__ == '__main__':
    main()
