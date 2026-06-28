---
name: validate-research
description: >
  Use when PROPOSING/considering a new idea OR judging whether a feature, model combo,
  hyperparam, or detector is worth promoting. Two F6 rules: (0) prior-art check the idea
  against the history BEFORE spending effort, and (1) validate ONLY through the real
  gated sim, never an ungated screen (importance / AUC / ungated grid). Importance !=
  performance; and don't re-run something already DEAD/SHELVED.
---

# F6 — Research validity (how you measure)

*Mirrors CLAUDE.md "## Critical Rules" F6 — gate 21 (gated-sim-not-screen); step 0 (prior-art) + step 4 (PySR-ablation) added from session feedback + memories.*

0. **Prior-art check FIRST (before any effort on a new idea).** Search the history and report status before proposing or building:
   - **ARCHIVED_LOG.md** — canonical scoreboard **C01–C88** + DEAD/SHELVED verdicts (each SHELVED entry carries its revival conditions; each DEAD its root cause). Primary check.
   - **TODO.md** — active backlog + "What's untested (queued)" + idea queue.
   - **CLAUDE.md** — "What works / What doesn't work (tested and abandoned) / What's promising / What's untested".
   - **README.md** — high-level pointers.
   Grep the idea's keywords across all four. If already tested → surface the prior verdict + numbers + (SHELVED) revival conditions; do NOT silently retest. Only proceed if it's genuinely new, or a revival condition is now met (say which).

1. **Gated sim, never a screen.** Validate through the real gated backtest (Mode V/HRS, `tools/bt_*.py`, or the live shadow) — never LGBM importance, AUC, or an ungated grid score. Ungated screens are systematically optimistic by **7–27pp** because they skip the live confidence gate.
   - Evidence: importance-ranked fast-window features (ranked #2–#4) cost −22 to −27.5pp gated; GB+LGBM won the ungated bear screen but lost the real engine (+43.5% vs +51.2%); a 2-model RF+LGBM pair beats every triple/quad/quintuple gated while LR poisons every combo. **Importance ≠ performance — only the gated sim decides.**
2. **Same gated path, both arms.** Generate signals ONCE; apply the change as a transform so both arms trade identical entries/exits (any CPU/GPU proba bias cancels). Pattern: `tools/bt_vol_target.py`.
3. **Window-scaling / window-shop guard.** Returns scale with replay length — compare the recent half (H1) or annualize. A 2-month "winner" must hold out-of-sample (recent-window overfit is the classic trap — C01 vol-scaled, the 960h specialist scan). Check sub-period consistency: a deployable result should beat baseline in EVERY chunk, not just one regime.
4. **Feature-removal tests must re-run PySR FIRST.** Disabling a feature family via `disabled_prefixes` is NOT a clean removal: PySR symbolic features (`pysr_*`) embed raw features inside their formulas, so a `xa_` prefix-disable doesn't touch `pysr_1` — the signal stays in the model. A prefix-disable measures **Trim A** ("drop the raw duplicate, PySR keeps the signal" ⇒ raw redundant given PySR), NOT **Trim B** ("remove the family entirely"). To truly test removal: re-run Mode P with the family excluded from the input pool, then disable the direct features, then HRST. [[feedback-pysr-embeds-raw-features]]
5. **Don't auto-promote a win.** A passing result goes through /promote-check (flat, leakage, atomic) — and only if the user agrees. Don't launch the expensive confirming run unasked.
6. **Backtest fills are zero-lag idealized; live execution isn't (CLAUDE.md F6 gate 22, added 2026-06-29).** The backtest transacts instantly at the inference-bar **close**; live executes ~1–2 min later at the real **bid**, which in fast moves diverges sharply (the 2026-06-26 ETH crash filled SELLs ~3% below the decision-bar close in ~60s — **clean maker fills at the live bid, 0% fee, NOT slippage**). So a backtest **overstates** returns by the decision-to-execution lag — worst for **frequent-trading horizons** (4h) and **volatile** regimes. Confirm any thin edge with the **lag ladder** (re-fill the same signals at +0/+1/+2 bars): live 4h/4h was **+3.85% @ +0 lag → −2.89% @ +1 bar ≈ the live −3.78%**. An edge that survives only at +0 lag is a fill-timing artifact; the **live realized PnL (ledger) is the honest number**.
   - **When a backtest beats live, localize the gap (don't guess) with the isolation ladder** — see CLAUDE.md "Backtest-vs-Live Fidelity — diagnosis playbook": (1) snapshot replay = bookkeeping, (2) refit replay = engine-vs-data, (3) re-infer on current data = data-revision hindsight, (4) feature diff = culprit source, (5) lag ladder = execution lag.
   - **Traps (each cost real time):** ledger bare timestamps are **CEST**, price bars + `inference_snapshots.logged_at` are **UTC** (ISO-`Z` ledger times = UTC) — align before comparing or you invent phantom 1–2% discrepancies; the theoretical fill = the **decision-bar close** (`inference_snapshots.close`), never the trade-hour bar close (that's 59 min in the future); `eth_hourly_data.csv` high/low ranges are collapsed (~0.03% vs 0.57% baseline) so **closes are usable but high/low are not** (intra-hour TP/SL backtests invalid); maker fills are clean (order log) — the gap is the decision→execution price MOVE, not slippage.
