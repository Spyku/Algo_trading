"""One-shot C57 retest — Markov-switching AR(1) regime probability as feature.

Imports the C57_PATCHER + run_idea machinery from the 9-idea Desktop runner
and runs only C57 on ETH 5,6,7,8h. ~50 min on Desktop (statsmodels MS-AR
fit is heavy — slowest of the Tier-A retests).

Prerequisite: statsmodels (with regime_switching submodule) must be importable.
  python -m pip install statsmodels

Decision rule (per CLAUDE.md TODO):
  avg APF delta >= +5pp -> PASS, escalate to HRST
  avg in (0, +5)        -> MARGINAL
  avg <= 0              -> FAIL (close out as DEAD with reliability)

Run:
  python tools/test_c57_only.py
"""
import os, sys
from datetime import datetime

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)
sys.path.insert(0, os.path.join(ENGINE, 'tools'))

from test_desktop_5ideas_runner import C57_PATCHER, run_idea, ASSET, HORIZONS, REPLAY
from test_14_ideas import LOGS_DIR, TS

if __name__ == '__main__':
    # Pre-flight dep check — fail loudly instead of silently no-opping like C56 did
    try:
        from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression  # noqa
    except ImportError:
        print('FATAL: statsmodels MS-AR not importable. Install with:')
        print('  python -m pip install statsmodels')
        print('Refusing to run C57 against a no-op patcher (would yield meaningless delta).')
        sys.exit(1)

    print('=' * 100)
    print(f'  C57 ONLY (Markov-switching AR(1) state probability) — {datetime.now().isoformat()}')
    print(f'  Asset: {ASSET}  Horizons: {HORIZONS}  Replay: {REPLAY}h')
    print('=' * 100)
    summary = []
    run_idea('c57_msar_regime', 'C57MS', C57_PATCHER, summary)

    out_summary = os.path.join(LOGS_DIR, f'c57_only_summary_{TS}.txt')
    with open(out_summary, 'w', encoding='utf-8') as f:
        f.write(f'C57 only retest {TS}\n')
        f.write(f'Asset={ASSET} horizons={HORIZONS} replay={REPLAY}h\n\n')
        for r in summary:
            f.write(f"{r['name']}: verdict={r['verdict']} avg_delta={r['avg_delta']:+.3f}\n")
            for h, tw, bw, d in r['per_horizon']:
                f.write(f"  {h}h: test={tw:.3f} base={bw:.3f} delta={d:+.3f}\n")
    print(f'\nSummary: {out_summary}')
