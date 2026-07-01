---
name: todo
description: >
  Use whenever the user says "ask todo" / asks about "the todo" / "what's left", wants to
  LOG a new task or in-flight run, or marks a task DONE. Active work lives in TODO.md
  (check here AND log here); a finished/closed arc MOVES to ARCHIVED_LOG.md with its
  verdict; CLAUDE.md holds only stable reference. Prior-art check before logging; always
  preserve the verdict + numbers when archiving.
---

# TODO workflow — log active in TODO.md, archive done in ARCHIVED_LOG.md

*Three files by role (see CLAUDE.md "## Pending Work"). Mirrors [[feedback_open_claude_md_shows_todo]], [[feedback_todo_numbering]], [[feedback_prior_art_check]], [[feedback_color_coding_blue_good]], [[feedback_validated_change_doc_push]].*

**The three files — never mix their roles:**
- **TODO.md** — **ACTIVE work only.** The `## ⚡ TODO — Active work only` block (P0s) + the `## 📊 At-a-glance` dashboard table (`| Pri | Item | Where | Status |`). **Check here AND log new work here.**
- **ARCHIVED_LOG.md** — historical audit trail: canonical scoreboard **C01–C88**, closed / DEAD / SHELVED verdicts (SHELVED carries revival conditions, DEAD carries root cause). **Append-only.** Check here for prior art; log CLOSED arcs here.
- **CLAUDE.md** — stable reference only (engine card, What-works/doesn't, critical rules). **Never** put volatile TODO state here.

## 1. When the user "asks the todo" (check / report)
Read **TODO.md** and surface the **⚡ ACTIVE block + any in-flight job status** in the reply — don't make them ask for it separately ([[feedback_open_claude_md_shows_todo]]). Verify liveness of any "running" item by file mtimes, not memory (Rule 0). If they ask "what's left", separate **genuinely open/actionable** from **done-but-not-pruned** (offer to prune the stale rows).

## 2. Logging a NEW todo (→ TODO.md)
1. **Prior-art check FIRST** ([[feedback_prior_art_check]]) — grep ARCHIVED_LOG.md (C01–C88) + TODO.md + CLAUDE.md for the idea's keywords. If already DEAD/SHELVED/done → report the prior verdict instead of re-logging (unless a SHELVED revival condition is now met — say which).
2. Add a **dashboard row** (`| Pri | Item | Where | Status |`); if it's an in-flight run or a P0, also add/refresh the `⚡ ACTIVE` block with backup paths + a one-line rollback for anything live.
3. **Conventions:**
   - **ID = MMDD** (month+day, e.g. `0701`), letter suffix for same-day extras (`0701b`) — NOT sequential ([[feedback_todo_numbering]]).
   - **Priority:** 🔥 P0 · 🔵 P1/P2 · 🟡 parked-active · 🚀 P3 · ⚪ P4.
   - **Color:** 🔵 = good/pass, 🔴 = bad/fail — NEVER green; grey/white = neutral/skipped ([[feedback_color_coding_blue_good]]).
   - Convert relative dates to absolute; fill **Where** (machine) + **Status**.

## 3. Marking a todo DONE (TODO.md → ARCHIVED_LOG.md)
- **Move, don't delete.** Cut the entry from TODO.md and append it to ARCHIVED_LOG.md **with its verdict + the numbers** — the evidence IS the point (future prior-art checks read it). Assign/keep its **C-number** on the scoreboard for a research idea.
- ARCHIVED_LOG is **append-only** — don't rewrite a closed entry; add a new one referencing the old if a verdict later changes.
- A **validated engine truth** (not just "task done") also updates CLAUDE.md's What-works/What-doesn't or engine card, then commit + push ([[feedback_validated_change_doc_push]]).
- Prune the now-stale TODO row so the dashboard stays "active only."

## Don't
- Don't log volatile state in CLAUDE.md, or leave a research verdict only in TODO.md (it gets pruned → lost; archive it).
- Don't silently delete a done item without preserving its verdict in ARCHIVED_LOG.
- Don't re-log a DEAD/SHELVED idea without a met revival condition.
