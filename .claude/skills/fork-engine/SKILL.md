---
name: fork-engine
description: >
  Use when forking/cloning the engine or adding a NEW asset / horizon / timeframe
  (e.g. the Fujiwara 15m/30m forks). The Chinese-wall + research-validity recipe:
  total isolation from production, no monkey-patching, PySR discovered for THIS
  timeframe first. Combines F1 + F6 + the wall.
---

# Engine fork / new asset / horizon / timeframe

*Cross-cutting — combines CLAUDE.md F1 (promotion/leakage) + F6 (research validity) + the Chinese-wall invariant (FORBIDDEN #1–#2).*

**Order matters: Mode P (PySR) is STEP 0 — never run HRST on a new timeframe/asset without discovering PySR for THAT timeframe first.**

## Chinese wall (no production infection)
1. **Isolated outputs** — write only to fork dirs (`models_fujiwara_15/`, `config_fujiwara_15/`, …); NEVER `models/`, `config/`, `crypto_ed_production.csv`, `regime_config_ed.json`.
2. **No shared-config writes** — redirect anything that writes a shared file (e.g. Mode F → `disabled_features.json`) to the fork's own dir.
3. **Disable `--promote`** in the fork — it must be physically unable to write live.
4. **Don't read production either** — seed config from a fork template, not `config/regime_config_ed.json`; point Mode C incumbent at fork paths.
5. **Never imported by the live trader** (trader imports faye only).
6. Reading shared INPUTS (`data/macro_data/*`) is fine — the wall forbids WRITES to production.

## No monkey-patching
- Every change is first-class code in the fork (mirror faye's philosophy). Don't mutate an imported module's functions/globals at runtime. If a shared helper needs a fork variant, give the fork its OWN copy (e.g. `pysr_discover_features_fujiwara.py`) — leave the shared one byte-identical so production is provably unbroken.

## PySR per timeframe (F6 + leakage)
- Discover on the fork's OWN data (inject the fork's `load_data`/`build_all_features`), candle-scaled 6-month historical window, written to the fork dir.
- **Distinct filenames per timeframe** — `pysr_ETH_5h.json` (hourly) vs `pysr_ETH_5p_15m.json` vs `pysr_ETH_5p_30m.json`. No collision. Update BOTH write and read sites.
- **Distinct COLUMN labels per timeframe** — the feature columns are tagged too: `pysr_1_15`/`pysr_2_15` (15m), `pysr_1_30` (30m); hourly stays `pysr_1`. Set in `_compute_pysr_features` (`col_name = f'pysr_{i+1}_{CANDLE_MINUTES}'`). They still `startswith('pysr_')` so the feature-floor / exclusion / grade logic is unaffected. This makes even a bare feature name in a model's `optimal_features` unambiguous.

## Verify before handing back
- Production files md5 unchanged after running the fork; shared modules untouched; fork imports clean and points only to fork dirs; lag audit clean (`tools/audit_feature_lag_fujiwara.py`). Then validate any result through the GATED sim (see /validate-research). Don't auto-launch the long HRST — ask.
