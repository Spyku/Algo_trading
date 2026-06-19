---
name: rules
description: >
  MASTER index of Alex's working rules for the algo-trading engine. Holds the Query
  Pre-Flight (run every message), the universal FORBIDDEN list, and routing to the
  focused workflow skills. Invoke /rules at the start of any non-trivial task or when
  unsure which gate applies. For a specific workflow load the focused skill instead:
  /promote-check (live writes), /inference-change (live signal path), /data-source
  (new/changed data), /fork-engine (engine fork or new asset/horizon/timeframe),
  /validate-research (judging a feature/model/hyperparam).
---

# Working Rules — MASTER

> Apply the Pre-Flight on **every** message — including "build", "quick", "research".
> Skipping it on a build task was the 2026-06-19 failure. A UserPromptSubmit hook
> injects a short reminder each message; this skill is the full index.

## 0. Prime directive — run the PRE-FLIGHT every message

1. **Verify, don't guess (Rule 0) — when the user asks ANYTHING, CHECK THE PROGRAMS/code/
   data, never answer from your head.** Read the actual source — code (`file:line`), prod
   CSV, `regime_config_ed.json`, logs, data — and answer **with the numbers/values you
   found, never memory/gut.** If you can't verify, say "unverified." (This applies even to
   rule content: I once asserted TODO numbering was "DDMM" from memory — it's MMDD. Check.)
2. **Classify → fire the matching family → STOP if a gate fails.**
3. **Universal checks:** verified with numbers? · flat-before-promote? · leakage clean? ·
   isolated from production? · liveness by mtimes not processes? · gated-sim not screen?

This skill set MIRRORS the canonical classification in [CLAUDE.md](../../../CLAUDE.md) "## Critical Rules" (2026-06-14: Rule 0 + 21 gates in 6 families + 8 demoted-to-reference). The "Gates" column below is the authoritative CLAUDE.md gate numbers each skill covers — keep them in sync.

| If the request… | Family (CLAUDE.md gates) | Load |
|---|---|---|
| writes live config / models / prod CSV / live engine | F1 Promotion (1–5) | **/promote-check** |
| changes live inference / signal generation | F2 Inference (6–11) | **/inference-change** |
| touches live trader config / orders / display | F3 Execution (12–15) | §3 below |
| downloads / merges a data source | F4 Data (16–17 + checklist) | **/data-source** |
| adds/changes a computed FEATURE (from existing data) | F4+F6 | **/add-feature** |
| judges a job / compares backtests / emits a user command | F5 Process (18–20) | §4 below |
| judges/PROPOSES a feature, model, hyperparam, or idea | F6 Research (21) | **/validate-research** |
| forks/clones the engine or adds a new asset/horizon/timeframe | F1+F6+wall | **/fork-engine** |

## 1. FORBIDDEN (never — without explicit user OK) — applies universally

1. **No production infection — Chinese wall.** Research/forks NEVER write `models/`, `config/`, `crypto_ed_production.csv`, `regime_config_ed.json`, or shared configs (`disabled_features.json`). Forks write only their own dirs; may READ shared inputs, never WRITE production.
2. **No monkey-patching.** First-class code only; don't mutate imported modules at runtime to change behavior.
3. **Never modify/flag** `trading_config.json` / live config as wrong **without asking first.** [[feedback-trading-config]]
4. **Never promote mid-position** (`state=invested`) unless the user explicitly overrides.
5. **Never trade/enable** an `enabled:false` asset.
6. **Never assert from memory/gut** — verify against the source first (Rule 0).
7. **Never judge a feature/model/hyperparam by importance / AUC / ungated screen** — gated sim only.
8. **Never run HRST on a new timeframe/asset without discovering PySR for THAT timeframe first** (distinct per-timeframe PySR files).
9. **Never add embargo to live signal generation.**
10. **Never auto-launch long/expensive runs** (HRST, Mode P) or **irreversible/outward-facing actions** without asking.
11. **Never skip the Pre-Flight** — not for "build", "quick", or "just".
12. **Never pursue a NEW idea without a prior-art check first.** Before proposing, scoping, or starting any new idea/feature/experiment, search the history: ARCHIVED_LOG.md (C01–C88 scoreboard + DEAD/SHELVED verdicts), TODO.md (backlog + "untested"), CLAUDE.md (What works / doesn't / promising / untested), README.md. If it was already tested → surface the prior verdict + numerical evidence + (for SHELVED) its revival conditions, and do NOT silently redo it.
13. **Never `cp` a fresh mock from production.** All mock/experiment runs use the ONE canonical `crypto_trading_system_faye_mock.py` (hardened, fail-closed `_mock_guard`, env isolation). Spinning up a new mock by copying `crypto_trading_system_faye.py` proliferates stale copies and loses the isolation guards. [[feedback-canonical-mock-file]]

## 2. ALWAYS (the spine; full procedures live in the focused skills)

- Verify with numbers, cite `file:line` (Rule 0). [[feedback-verify-with-numbers]]
- **Prior-art check before any new idea** — grep ARCHIVED_LOG.md / TODO.md / CLAUDE.md / README.md for it; report DEAD/SHELVED/done status before spending effort (full procedure: /validate-research step 0).
- **Opening CLAUDE.md implicitly asks for current state** — whenever the user says "open/show/review CLAUDE.md" (or similar), ALSO read TODO.md and lead with the ⚡ ACTIVE block + in-flight job status, before addressing the CLAUDE.md question. [[feedback-open-claude-md-shows-todo]]
- Test first — never change the live engine/config without testing on faye/a fork. [[feedback-testing-routine]]
- Save/update a memory when the user gives feedback or a rule is learned (check for an existing one first).
- **TODO.md entries are numbered MMDD (month+day), not sequential** — today's entry is today's MMDD (e.g. 0619), with letter suffixes for same-day extras. [[feedback-todo-numbering]]
- **Validated code change ⇒ document + push (definition of done).** Once a code change is tested/validated, update the docs and commit+push — never leave a validated change undocumented or unpushed. Full steps: §6. [[feedback-validated-change-doc-push]]

## 3. F3 — Live trader execution (reference; no dedicated skill)

- Regime config shape: per-asset `regime_detector` + `bull`/`bear` blocks; per-regime `min_confidence` is the live gate; module `MIN_CONFIDENCE=75` is only a fallback.
- Maker orders: `post_only`, BUY rests `bid+0.01`; cancel stale first, reprice each cycle, market-fallback on timeout.
- Clock drift via independent NTP query, not the 409 echo.
- Telegram display timestamps Europe/Zurich; all order/internal logic UTC.

## 4. F5 — Operating process (reference; no dedicated skill)

- **Liveness by file mtimes**, two-machine (Drive-shared). Ask "is it running on the other machine?" before declaring a job dead. Mode R runs silent up to ~2h. [[feedback-drive-sync-check]]
- **Window-scale backtest comparisons** — returns scale with replay length; compare the recent half (H1) or annualize; check the `Replay: NNNNh` header first.
- **Emit user commands as bare** `python tools/<script>.py` (venv already active; no absolute interpreter paths).

## 5. Demoted to reference (CLAUDE.md 2026-06-14 — non-gating; mirror)

The actionable demoted items were promoted into skills: **Windows SSL** → /inference-change #8; **Set-D `feature_override`** → /inference-change #7; **asset-universe invariant** ("never trade `enabled:false`") → FORBIDDEN #5. The rest stay pure lookup (consult CLAUDE.md, don't gate on them):
- **Production scoring:** Mode V/H picks the model by `score = return × (win_rate/100)` for positive returns, raw return for negatives (≥5-trade configs). Describes selection, gates nothing.
- **Mode T chains rally-cooldown:** Mode T merges bull/bear signals then auto-runs the rally-cooldown sweep — no need to run Mode G after T (standalone G is a cache-fed fast-iteration fallback).
- **signal_log schema:** `timestamp,asset,price,action,confidence,h_1,sig_1,conf_1,h_2,sig_2,conf_2` — chart-rendering only; write errors swallowed.

## 6. Definition of done — a VALIDATED code change

When a code change has been tested/validated (gated sim / smoke test / shadow / parity — per /validate-research), finish it; don't leave it half-shipped:
1. **Update the docs to match reality:**
   - **README.md** + **CLAUDE.md** — stable reference (engine card, architecture, version table, commands) if the change touches any of it.
   - **TODO.md** — active state / in-flight runs / live state (per the existing doc-routing in CLAUDE.md "Pending Work").
   - **ARCHIVED_LOG.md** — if a research arc closed, move its entry here with the verdict + numerical evidence (append-only; don't rewrite closed entries).
2. **Commit** with a clear message describing what changed + the validation evidence.
3. **Push git** (`git_push.bat` from the engine folder, or `git push`).

**Standing authorization + guardrails (user-set 2026-06-19):** push IS authorized for validated changes — this overrides the general "ask before outward-facing actions" gate *for this case*. Still: never commit `config/` (gitignored — secrets/keys), don't bypass hooks/signing, and end commit messages with the required `Co-Authored-By` trailer. If the change touches **live production** (models/config/trader), the doc+push step comes AFTER /promote-check passes, not instead of it.

Full family gate details: [CLAUDE.md](../../../CLAUDE.md) "## Critical Rules".
