---
name: promote-check
description: >
  Use BEFORE any write to live production — copying a config/model into models/ or
  config/, upserting crypto_ed_production.csv, promoting a FAYE result, or otherwise
  changing what the live trader serves. The F1 promotion gates. STOP if any fails.
---

# F1 — Promotion gates (before any live config / model / CSV / engine change)

*Mirrors CLAUDE.md "## Critical Rules" F1 — gates 1–5.*

The live trader imports its engine by exact filename — currently `crypto_trading_system_faye.py`
([crypto_live_trader_ed.py:61](../../../crypto_live_trader_ed.py#L61)). FAYE generates models; the v2 trader consumes them.

1. **Trader FLAT first.** Verify `state=cash` in `config/position_ed_v2_<ASSET>.json` before any config/prod-CSV swap. Promoting mid-position corrupts entry/exit bookkeeping. User may override per-promotion. [[feedback-flat-before-promotion]]
2. **Leakage check.** Before writing production CSV, confirm no leakage — for PySR, `discovery_method == "historical"`. Enforced by `_check_pysr_leakage`, which BLOCKS the write on a non-historical formula.
3. **Regen PySR ⇒ retrain.** After any Mode P (rewrites `pysr_*.json`), run DV/HRS for the same horizon or you get silent feature drift (production stores feature NAMES; inference re-evaluates whatever formula is in the JSON now).
4. **CSV upsert by coin AND horizon.** Mask on BOTH before delete-then-append; filtering by coin alone wipes every horizon's row for that coin.
5. **Promote FAYE atomically + verify staged freshness.** Copy `models_faye/crypto_faye_production.csv` AND `config_faye/regime_config_faye.json` into live in ONE step; verify both unchanged (mtime) since the HRST that produced them — a later HRST overwriting the staged path yields a frankenstein. Auto-promote is disabled ([crypto_revolut_ed_v2.py](../../../crypto_revolut_ed_v2.py)) after exactly that fired.

**Forbidden here:** never promote mid-position without explicit override; never write production from a research/fork path (that's /fork-engine's wall).

**After it passes:** finish per /rules §6 — update README/CLAUDE.md/TODO.md (+ ARCHIVED_LOG.md if an arc closed), commit, and push.
