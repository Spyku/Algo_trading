# ARCHIVED 2026-06-05 — Ed v3 monkey-patch chain (retired by FAYE)

These files are the **Ed v3 production stack** — a 4-layer monkey-patch chain that
`crypto_trading_system_faye.py` consolidated into one self-contained, zero-monkey-patch
file. FAYE was **promoted to production 2026-05-31**, which retired this entire chain.
They are kept for git history / forensic reference only.

## What's here

| File | Was | Imported by (pre-archive) |
|---|---|---|
| `crypto_trading_system_ed_g_narrow_d.py` | base narrow-grid fork | only v2 + v3 |
| `crypto_trading_system_ed_g_narrow_d_parallel_nearlive.py` | v2 fork (NEAR_LIVE semantics) | only v3 |
| `crypto_trading_system_ed_g_narrow_d_parallel_nearlive_v3.py` | v3 fork (true Mode-D outer parallelism) | only the v3 smoke test |
| `crypto_trading_system_ed_step6.py` | Step 6 engine-refactor fork | nothing (0 importers) |
| `crypto_trading_system_ed_step6_nearlive.py` | Step 6 near-live companion | only v2 + v3 |
| `crypto_signal_core_nearlive.py` | near-live signal core (v3-era) | only `step6_nearlive` |
| `tools/smoke_test_v3_worker_guard.py` | test of the v3 ProcessPool worker | nothing |

## Why it was safe to archive (verification, 2026-06-05)

Import graph is a **closed cluster** — these files import only each other. Verified that
**no production file imports any of them**:
- `crypto_trading_system_faye.py`, `crypto_trading_system_ed.py`, `crypto_revolut_ed_v2.py`,
  `crypto_live_trader_ed.py`, `crypto_optimizer_bot.py`, `crypto_live_shadow.py` — none import the chain.
- String references in `faye.py` / `faye_refineopt.py` / `faye_reftest.py` to these names are
  all **docstrings / "inlined from" comments**, NOT imports (FAYE inlined the near-live semantics).
- **KEPT in root:** `crypto_signal_core.py` (no `_nearlive`) — the *production* signal core that
  `crypto_live_shadow.py` imports (`import crypto_signal_core as _core_mod`). Do not confuse it
  with the archived `crypto_signal_core_nearlive.py`.

Post-archive: `tools/smoke_test_faye.py` → FAYE `✓ imports without error`; all production files
`py_compile` clean. (The smoke test's one ✗ `MODE_D_OUTER_WORKERS==8 got 3` is the laptop's
worker count vs the Desktop-authored assertion — unrelated to this archive.)

## Restoration recipe

```powershell
# from engine root — restore everything
git mv ARCHIVED\2026-06-05_v3_chain_retired\*.py .
git mv ARCHIVED\2026-06-05_v3_chain_retired\tools\smoke_test_v3_worker_guard.py tools\
```
(Or `Move-Item` if not using git.) The chain only re-runs if you intentionally revive the
v3 monkey-patch path; FAYE is the canonical engine and needs none of these.
