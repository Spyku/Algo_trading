# crypto_trading_system_ed.py — RETIRED 2026-07-01

The Ed V1.0 engine. **Superseded by `crypto_trading_system_faye.py`** (THE engine).
Serving migrated to faye 2026-06-05; the last live/operational users were repointed to
faye on 2026-07-01, then this file was archived:

- `crypto_live_shadow.py` — feature-building ed → faye (Rule-23 mirror gap closed; commit c734642)
- `crypto_optimizer_bot.py` — `SCRIPT_PATH` ed → faye (+ `_FAYE_WARNINGS_BAKED=1` in the spawn env)
- `crypto_trading_system_meta.py` — `import` ed → faye
- `tools/validate_core_against_signal_log.py` — dropped the `--engine ed` branch

**Reviving a research tool that imports `crypto_trading_system_ed`:** repoint it to
`crypto_trading_system_faye` (same functions: `load_data`, `build_all_features`,
`generate_signals`, `_compute_pysr_features`, `get_decay_weights`, …). faye needs
`FAYE_LIBRARY_MODE=1` + `_FAYE_WARNINGS_BAKED=1` set before import to skip its os.execv re-exec.
The ~48 `tools/` + `_idea_patchers/` scripts that still `import crypto_trading_system_ed` are
closed C-scoreboard experiments — they will ImportError if run until repointed.
