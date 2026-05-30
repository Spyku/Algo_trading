# 2026-05-30 post-FAYE cleanup

Archived after FAYE Phases 1-7 shipped (commits `8c122ef` through `4ab34d5`).
The single-file FAYE consolidation supersedes the variant scripts that lived
at the engine root; the v3 monkey-patch chain (g_narrow_d + parallel_nearlive
+ step6_nearlive + signal_core_nearlive) is what FAYE inlines.

**These files were NOT inputs to FAYE — they were dead experimental code or
old date-stamped snapshots that survived as clutter.**

## How to restore something

```bash
git mv "ARCHIVED/2026-05-30_post_faye_cleanup/<subdir>/<filename>" .
```

Or for untracked files (most of these):
```bash
mv "ARCHIVED/2026-05-30_post_faye_cleanup/<subdir>/<filename>" .
```

## What's here

### `variants/` — experimental engine forks
Dead variant scripts that lived at engine root. None were imported by active
code paths; only doc/history mentions remain in `CLAUDE.md`, `TODO.md`, and
`ARCHIVED_LOG.md`.

- `crypto_trading_system_ed_cdar.py` — CDaR (Conditional Drawdown at Risk)
  risk-metric experiment
- `crypto_trading_system_ed_cvar.py` — CVaR (Conditional Value at Risk) experiment
- `crypto_trading_system_ed_cpcv.py` — CPCV (Combinatorial Purged CV) experiment
- `crypto_trading_system_ed_robust.py` — robust optimization variant
- `crypto_trading_system_ed_h_strict_family.py` — H_STRICT_FAMILY (K=5 + dedup)
  experimental fork; the bits that survived (K=5, dedup by `(combo, w)`) are
  now inline in FAYE as native code
- `crypto_trading_system_ed_noprod.py` — small noprod variant
- `crypto_trading_system_ed_pre_macro_cache_fix_20260527_112231.py` —
  date-stamped engine backup from before the macro-cache fix
- `_launch_h_strict_family.bat` — launcher for h_strict_family variant

### `tools/` — variant-driving test runners
Tools that ONLY existed to drive the archived variants above. Each cluster
links to the variant it tested.

CDaR/idea-test cluster:
- `test_c04_to_c08_runner.py`, `test_desktop_5ideas_runner.py`,
  `test_c05_c06_only.py`, `test_c32_to_c40_batch.py`

CVaR + CPCV cluster:
- `run_cvar_hrst_resumable.py`, `run_cpcv_hrst_resumable.py`,
  `run_c67_hrst_resumable.py`, `rerun_v2_full.py`

Robust + reliability cluster:
- `run_reliability_test.py`, `run_reliability_hrst.py`

step6 + h_strict_family + path-fix cluster:
- `test_step6_regression.py` — step6 (pre-nearlive) regression harness
- `smoke_test_path_resolution_matrix.py` — sub-task from Fix #11 work, orphaned
- `run_locked_detector_hrst.py` — orphan (0 external refs)
- `compare_h_b_prod_30d.py` — h vs b production comparison (h75 era)

### `docs/` — stale doc drafts
- `CLAUDE_NEW.md` — May 19 draft of CLAUDE.md; was never promoted, current
  `CLAUDE.md` evolved separately
- `TODO_TEST.md` — tiny stale TODO scratchpad

### `snapshots/` — old `models_*/` and `config_*/` output dirs
Pre-`_nearlive` artifact snapshots from the H75 / G_desktop / 0524 / 0525
experimental runs. All file mtimes are pre-2026-05-29 (FAYE's predecessor
v3 first ran 2026-05-27). The ACTIVE v3 output lives in `models_g_desktop_nearlive/`
and `config_g_desktop_nearlive/` — those are NOT here.

7 model dirs × 7 config dirs = 14 snapshot dirs, ~440 KB total.

## What was NOT archived (and why)

- `crypto_trading_system_ed.py`, `_g_narrow_d.py`,
  `_g_narrow_d_parallel_nearlive.py`, `_g_narrow_d_parallel_nearlive_v3.py`,
  `_step6.py`, `_step6_nearlive.py` — actively imported by the v3 chain that
  is still running on Desktop. Will be safe to archive AFTER FAYE replaces
  v3 in production.
- `crypto_signal_core.py`, `crypto_signal_core_nearlive.py` — actively imported.
- `crypto_revolut_ed_v2.py` — the LIVE trader.
- `_idea_patchers/` — referenced by `tools/test_14_ideas.py` (still in TODO.md).
- `_pit_workdir/` — used by 4 active debugging tools (validate_core_point_in_time,
  counterfactual_*, afternoon_run).
- `models_g_desktop_nearlive/` and `config_g_desktop_nearlive/` — active v3
  output, written-to right now.
