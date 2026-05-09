"""One-shot C56 retest — validates harness fix (runpy → main() call).

Imports the C56_PATCHER + run_idea machinery from the 9-idea Desktop runner
and runs only C56 on ETH 5,6,7,8h. ~35 min on Desktop.

Decision rule (per CLAUDE.md TODO): if avg APF delta materially differs
from overnight FAIL Δ−5.34 (e.g. trends positive or near-zero), the
runpy → main() fix is genuinely active and we unlock the 13-SUSPECT
retest queue. If still ~Δ−5.34, fix is broken or C56 is genuinely DEAD.

Run:
  python tools/test_c56_only.py
"""
import os, sys
from datetime import datetime

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)
sys.path.insert(0, os.path.join(ENGINE, 'tools'))

from test_desktop_5ideas_runner import C56_PATCHER, run_idea, ASSET, HORIZONS, REPLAY
from test_14_ideas import LOGS_DIR, TS

if __name__ == '__main__':
    print('=' * 100)
    print(f'  C56 ONLY (HMM 2-state regime as feature) — {datetime.now().isoformat()}')
    print(f'  Asset: {ASSET}  Horizons: {HORIZONS}  Replay: {REPLAY}h')
    print('=' * 100)
    summary = []
    run_idea('c56_hmm_regime', 'C56HMM', C56_PATCHER, summary)

    out_summary = os.path.join(LOGS_DIR, f'c56_only_summary_{TS}.txt')
    with open(out_summary, 'w', encoding='utf-8') as f:
        f.write(f'C56 only retest {TS}\n')
        f.write(f'Asset={ASSET} horizons={HORIZONS} replay={REPLAY}h\n\n')
        for r in summary:
            f.write(f"{r['name']}: verdict={r['verdict']} avg_delta={r['avg_delta']:+.3f}\n")
            for h, tw, bw, d in r['per_horizon']:
                f.write(f"  {h}h: test={tw:.3f} base={bw:.3f} delta={d:+.3f}\n")
    print(f'\nSummary: {out_summary}')
