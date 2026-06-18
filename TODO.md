## ⚡ TODO — Active work only

**Companion file**: [ARCHIVED_LOG.md](ARCHIVED_LOG.md) — historical audit trail, canonical scoreboard C01-C82, MERGED TOPICS, IDEA QUEUE drop-list (closed/shipped/STUB items with verdicts).

## 📊 At-a-glance — active TODO dashboard (2026-06-05)

| Pri | Item | Where | Status |
|---|---|---|---|
| 📌 | **FAYE LIVE** (promoted 2026-05-31 14:22 CEST; **bear swapped 8h@65%→5h@80% on 2026-06-02 23:09**) — `tsmom_672h` detector / bull 6h@65% / **bear 5h@80%** / shields OFF / rally cooldowns ON (custom per-regime params) / min_sell_pnl_pct=0 / max_hold_hours=10 / maker orders ON. Models: ETH 5h/6h/7h/8h RF+LGBM from FAYE H. Rollback in `archive/g_narrow_live_20260531_142202/`. **⚠️ CONFIG-DRIFT INCIDENT (found 2026-06-12):** the live `regime_config_ed.json` had silently DRIFTED on June-8 23:18 to **bull 8h / bear 8h@70 / both rally-cooldown gates `enabled:false` / bear shield ON / min_sell_pnl 0.5** — from an "inconclusive" June-8 8h promotion (`logs/ed_v1_20260602_215119.log` "Promoting ETH 8h → production return=8.98%") that left NO `_pre_` backup and dropped the validated gates. So the trader ran ~4 days with NO cooldown gates + wrong horizons. **NOT caused by the 0611/0612 feature testing** (all `--no-persist`; config mtime was June-8, pre-testing). **RESTORED 2026-06-12 08:52** to the June-2 RST winner (bull 6h@65 / bear 5h@80 / shields OFF both / gates ON [bull rr8h≥2.0 OR rr14h≥6.0 cd6h; bear rr10h≥5.5 OR rr12h≥2.0 cd8h] / min_sell_pnl 0) — gate params recovered from `config/regime_config_ed_pre_RST_bear5h_20260602.json` + winner log `ed_v1_20260602_210324.log` (line 478/611). Backup of the drifted config: `config/regime_config_ed_pre_restore_6h5h_20260612_085249.json`. Applied while trader was INVESTED per explicit user override of Rule 19. | **Desktop** (always) | 🟢 running |
| 🔥 **SHADOW DOWN 2.5d — FIXED (0618)** | **Shadow monitor `import_failed` since 2026-06-15 22:37 (silent, no alert) — ROOT-CAUSED + fixed; needs trader restart.** Live-correctness shadow (`crypto_live_shadow.py`) logged `import_failed` every cycle for ~2.5 days. **Root cause:** the P0-0613 logging tee `_TeeStream` ([crypto_live_trader_ed.py:128](crypto_live_trader_ed.py#L128)) had `write`/`flush` but **no `reconfigure`**; the shadow's `import crypto_trading_system_ed` (ed.py imported ONLY by the shadow, not the live path) hits ed.py:383 `sys.stdout.reconfigure('utf-8')` at import → AttributeError under the tee → import dies. **NOT caused by Mode C / my 06-16/17 edits** (predates them; coincides with the tee logging going live 06-15). **Live trader UNAFFECTED** — only the monitor (live path = revolut→live_trader→faye.compute_signal_core, never imports ed.py; trader traded normally: bought 06:00, signals 18:00 BUY95.5/19:00 SELL). **Cost:** 2.5d with the tripwire blind (trading ran unverified by the cross-check). **FIX applied:** added `_TeeStream.reconfigure()` proxy — compiles + reproduced-and-verified ("SHADOW IMPORT UNDER TEE: OK"). **Activate: restart the trader (FLAT now = cash, safe moment)** → shadow_signal_diff.csv returns to real comparisons. **Why no alert (2 gaps):** (1) shadow swallows errors by design (no Telegram); (2) daily sanity doesn't escalate `import_failed` (counts BUY↔SELL mismatches, not "monitor down") → silent outage. **✅ SANITY ALERT-GAP FIXED 2026-06-18** (`tools/sanity_check.py` `shadow_check`): now detects `shadow_error` rows (import_failed) → distinct loud **"SHADOW DOWN (monitor not running) — RESTART"** FAIL (was buried as a generic 0% match → no clear alert). Rerun confirmed: [1] SHADOW=FAIL (48/48 import_failed, now NAMED) + [2] SNAPSHOT REPLAY=PASS **368/368 = 100%** → trader signals verified correct *throughout* the 2.5-day outage. **✅ VALIDATOR FIXED** (`tools/validate_core_against_signal_log.py`): it was testing the WRONG engine/config — `--engine ed` (old engine) or `--engine faye`→**stale `config_faye`** (tsmom_672h/bull-6h vs live sma48/bull-8h → all-STALE/inconclusive). 3 edits: both engines read the **LIVE** config (`config/regime_config_ed.json`+`models/`); `FAYE_MODELS_DIR=models` (mirror live pysr); default `--engine faye`. **Result (laptop, faye+live, 30 samples): 25/30 = 83.3% match, avg conf delta −1.22 (was −12.23 on ed → ENGINE ALIGNMENT CONFIRMED).** 5 DIFFs are recompute artifacts (09:00 pure HOLD/SELL boundary @ conf 56.0 vs 55.68; 08:00/17:00/19:00 18–28pp gaps — recent ones likely the ~12h-stale laptop data). **Bit-exact "100% / no drift" = SNAPSHOT REPLAY (368/368); parity re-computes features so it can't be bit-100% by design.** To push parity→100%: rerun on DESKTOP (current data) + chase 08:00/17:00/19:00. **FOLLOW-UPS:** trader restart (shadow `_TeeStream.reconfigure` activate); the daily sanity now uses `--engine faye` (live engine) via the new default. UNCOMMITTED (shadow fix, sanity alert-gap, validator fix). | **Desktop restart** | ✅ fixed; ⏳ restart to activate |
| 🟢 **MODE C "Choice" (0617)** | **Native promotion gate added to `crypto_trading_system_faye.py` (additive, NO monkey-patch).** Lever-A Phase 1: compares **LIVE production** (`config/regime_config_ed.json` + `models/crypto_ed_production.csv`) vs **last-computed FAYE** (`config_faye/regime_config_faye.json` + `models_faye/crypto_faye_production.csv`) by **cross-window + hysteresis** — backtests BOTH full configs (detector+horizons+confs+gates+models) across rolling 720h windows over `--replay`, verdict **PROMOTE iff challenger beats incumbent in ≥⅔ windows AND ≥5pp (downside-weighted median+worst); else HOLD**. **Report-only** (never writes live — manual promote, trader flat, Rule 19). CLI: `python crypto_trading_system_faye.py C ETH [--replay 2880] [--chal-config ... --chal-models ...]`. **Build:** 3 additive edits — `run_mode_c`+`_modec_sim` (module-level, before `__main__`), `C` dispatch branch, `--chal-config/--chal-models` flags; + `'C'` added to `VALID_MODES` (first smoke hit the menu — fixed). `_modec_sim` is a documented mirror of the nested `_sim_regime_switched` (can't reuse — it closes over Mode-T-local QR vars; consolidate later). Byte-compiles clean; touches ZERO existing code paths. **Smoke RUNNING** (live vs MIXED-RST challenger tsmom_168h/5h-w300, `--replay 960`, ~40min) — expect HOLD (cross-checks the manual finding that w300/tsmom_168h don't robustly beat live). **Follow-ups:** add `C` to the `Choose mode:` menu text (cosmetic); future consolidation of the dup sim. **UNCOMMITTED.** ⚠️ os.execv re-exec misfires the bg-completion notification on CLI runs — poll the log. **✅ SMOKE PASSED 2026-06-17 (960h, live vs MIXED-RST):** ran end-to-end, dedup worked (3 unique models — shared 8h w163 not regenerated), detectors tagged sane (sma48 33% / tsmom_168h 20% bull), clean verdict emitted. **BUT exposed a real flaw:** `--replay 960` → only **1 window** → trivial hysteresis → misleading **PROMOTE** (the exact single-window overfit C is meant to block!). **FIXED: added `MIN_ROBUST_WINDOWS=5` guard** — never PROMOTE on <5 windows; warns to use `--replay >= 2880`. Re-compiled. **Real 2880h test RUNNING** (`bch3n72ko`, 9 windows, ~1.6h) — expect HOLD (challenger doesn't robustly beat live, per the manual cross-window analysis). **Lesson: C is only meaningful with a long --replay (≥2880).** | **Laptop (faye.py)** | 🟢 2880h test running |
| 🟢 **TRADER UX (0616)** | **Gate UX simplified + live-setup in /help — CODE DONE (laptop), needs DESKTOP RESTART to activate.** Per user (gate ops were confusing): (1) **`/gate off` now ALSO wipes the active timer** (off = fully off; both-regimes `/gate [ASSET] off` + all-asset `/gate off` paths via new `_wipe_cooldown_timer`; per-regime `bull/bear off` deliberately does NOT wipe — the timer is shared, the other regime may own it). (2) **`/gate clear` now stamps `rally_cd_cleared_at`** so the catch-up scan ([crypto_revolut_ed_v2.py:1497](crypto_revolut_ed_v2.py#L1497)) **won't re-arm the just-cleared rally** — only a FRESH rally (trigger_time > stamp) re-arms; gate stays on for future. Fixes the "cleared but it restarted" confusion (the catch-up was re-detecting the same rally). (3) **`/help` appends a live "Current setup" block** (`_current_setup_text`: per-asset detector, bull/bear h@conf, shield, gate, min_sell, max_hold) so the actual config is always visible after button toggles. Byte-compiles (+70/−13). Execution-layer only — no signal-gen/shadow impact (Rule 7 N/A). **Activate: let Drive sync the .py to the desktop → restart the trader there. UNCOMMITTED.** | **Desktop restart** | ✅ code done; ⏳ restart to activate |
| 🔵 **STUDY (0616)** | **Idea 1 — is the double-condition rally-cooldown gate worth its complexity, or would a single condition do?** The gate (`_sweep_rally_cooldown`, [faye.py:7780](crypto_trading_system_faye.py#L7780)) is STRUCTURALLY a 2-window double-condition per regime: it fires when `rr[h_s]≥t_s OR rr[h_l]≥t_l` and only ever enumerates horizon **pairs** ([:7954](crypto_trading_system_faye.py#L7954)) — a single-window gate is never considered by the sweep. Live (sma48>sma100): **bull** `rr30≥4.0 OR rr36≥6.5 cd24`, **bear** `rr8≥2.5 OR rr30≥7.0 cd14`. **Quick study** (`tools/test_gate_simplicity.py`, modeled faithfully on `tools/test_cooldown_revert.py`; cached 5h/8h sigs `data/_detbt_sig_{5,8}h.pkl` + live regime/policy): per-regime **leg-ablation** (both=live / short-only / long-only / off) + a **best single-condition sweep** (1 window × 1 thr × cd) vs the double winner. Read: if short-only ≈ both → 2nd leg is dead weight → simplify; if a best-single ≈ the double's REF/3-of-3 → the complexity isn't earning its keep. | **Laptop** | ✅ **DONE 2026-06-16 — the double condition is NOT pulling its weight (on this window the live 4-leg config is BEATEN by a single condition).** `tools/test_gate_simplicity.py`, window 04-16→06-15 (1437 bars, 564 bull/873 bear — the recent ~60d the gate was tuned on). **Leg ablation (other regime held LIVE):** of the 4 active legs, exactly ONE carries the entire benefit — **bear SHORT `rr8≥2.5`**: bear-short-only == bear-both = **+28.84%**, bear-OFF = **+21.17%** → that one leg = **+7.67pp** (the whole rally-cooldown edge). The other 3 legs are dead-or-harmful: **bear LONG `rr30≥7.0`** is dead weight (long-only +20.64% ≈ off +21.17%); **bull LONG `rr36≥6.5`** NEVER fires (long-only == bull-off, both +30.70%); **bull SHORT `rr30≥4.0`** is net-HARMFUL −1.86pp (blocks 12 profitable BUYs → bull-ON +28.84% < bull-OFF +30.70%). **Simplest config = single bear `rr8≥2.5 cd14` + NO bull gate = +30.70%**, beating the current double-double LIVE (+28.84%) by +1.86pp and no-gate (+22.91%) by +7.79pp. **PART B single sweep:** 0/1152 bull singles beat baseline (no bull gate of any kind helps this window); bear strict-3of3 is fragile (H2 earlier-half has no bear rallies → any useful gate TIES baseline in H2 → fails the production STRICT `>` filter — exposes a window-fragility in `_sweep_rally_cooldown`'s STRICT too). **CAVEATS (Rule 19/F1):** ONE recent window, low H2 power, bull-harmful is likely window-specific (no chase-rally occurred this window — the bull gate's purpose). Needs a 2nd-window / forward confirm before simplifying live. **Recommendation:** move toward a single-condition gate (bear short leg is the only one earning its keep); re-run on a 2880/cross-window before flipping live. **⚡ FLIPPED LIVE 2026-06-16 (forward experiment — user-chosen, trader FLAT at flip):** `bull.rally_cooldown.enabled true→false` + `bear.rally_cooldown.t_long_pct 7.0→999` ⇒ single live gate **bear rr8≥2.5 cd14, bull OFF**. Atomic edit + hot-reload (no restart). Backup `config/regime_config_ed_pre_single_gate_20260616.json`; **rollback** `cp config/regime_config_ed_pre_single_gate_20260616.json config/regime_config_ed.json` (hot-reloads ≤5 min). Treating live as the 2nd window in lieu of the 2880 confirm. **Watch:** live WR/return vs the double's +28.84% study baseline; revert if it underperforms. NB an active `rally_cooldown_until 2026-06-17T04:00Z` (set by the old double gate) persists — trader only extends, never shortens. **🛠 ENGINE CODE CHANGED 2026-06-16 (user override, applied mid-HRST):** `_sweep_rally_cooldown` ([faye.py:7951](crypto_trading_system_faye.py#L7951)) now sweeps **SINGLE-window** gates only (`rr[h]≥t`, 1152 cfg = 96 thr × 12 cd) instead of horizon **pairs** → so "single gate" is now what the engine SEARCHES & PRODUCES (sticky across future HRSTs). **5-key config schema preserved** for trader/downstream compat: winner written with `h_long=h_short`, `t_long_pct=9999` (the never-fires sentinel the V0 baseline already uses) — trader's `_update_rally_cooldown`, T↔G convergence cmp, and `/gate` all consume it unchanged. Plateau/`nb_idx` reduced to 3 live dims (h,t,cd). git: `crypto_trading_system_faye.py` +41/−40, all inside the sweep fn. **Verified:** byte-compiles clean; smoke-test on cached sigs (regime_filter `all`+`bear`, write_config=False) → 1152 single cfg, all `h_short==h_long`/`t_long==9999`, top single = **rr8≥2.5 cd14 (= the live deployed gate)**; `NO STRICT WINNER` this window = expected (H2-no-rally strict fragility, same as Part B), not a regression. ⚠️ **Desktop HRST was RUNNING at edit time** (user overrode P0-0613 worker-corruption guard) → **that HRST's result may be inconsistent → re-run it on the new code.** Cosmetic follow-up: trader `/gate`+`/regime` Telegram display ([crypto_revolut_ed_v2.py:4038](crypto_revolut_ed_v2.py#L4038), `:2527`) will render the sentinel as "OR rr{h}h≥9999%" — harmless, fix on next trader touch to detect `t_long≥9999`→show single. NOT git-committed (no user request). **⚠️ CONFIG DRIFT 2026-06-16 13:42 + RESTORED 14:42:** after the 08:28 single-gate flip, `config/regime_config_ed.json` was rewritten at 13:42 re-enabling the **bull** double gate (orig params 30/36/4.0/6.5/24) while bear kept the `t_long=999` neutering → hybrid state. NOT the engine (FAYE writes isolated `config_faye/`; auto-promote off) and NO `/gate`/`setup` event in the trader log → a **direct/manual file edit**, cause unconfirmed. User chose RESTORE → re-disabled bull gate at 14:42 (trader flat, atomic). Drifted state backed up `config/regime_config_ed_pre_restore_singlegate_20260616_144242.json`. Live now = single gate (bull OFF, bear rr8≥2.5 cd14). |
| 🟣 **ARCH (0616)** | **Idea 2 — per-regime model search: is the H→RS pipeline order optimal?** (deep, much more testing). Current order: Mode **H** picks the best model per horizon **regime-blind** (trained/selected on ALL bars), THEN Mode **RS** picks the best bull/bear horizon pairing + detector — so the bull and bear "best" are chosen at the S step from a pool that was never optimised per-regime. Idea: search for **bull-specific** and **bear-specific** models (train/select each on its OWN regime's bars), refine the **top-X of each**, THEN run Mode S over those specialists. Hypothesis: a regime-blind model is a compromise; regime-specialised candidates may beat it once S selects among them. **Test plan (compute-bounded):** start on **`--replay 960` + 5,8h ONLY**; if it looks worth it → extend to **`--replay 1440` + 4 horizons (5,6,7,8h)**. Needs a code path to label bars by regime pre-training and a top-X-per-regime refine handoff into S — non-trivial. | **Laptop** | 🟢 **IN PROGRESS (Step 1, 2026-06-16).** **Design locked w/ user:** the chicken-and-egg — Mode H runs BEFORE the detector exists (detector chosen later in R/S), so H *can't* label bars bull/bear → forced to filter on the full picture. **Fix = PIN the detector to the current best (`sma48>sma100`) up front** → label bars → give each candidate a **bull subscore + bear subscore** → take **top-10 bull + top-10 bear** (not 1 full-picture winner) → Mode S sweeps `bull∈bull_pool × bear∈bear_pool`. Interpretation (b) = regime-conditional SELECTION (train on full window, specialize the *scoring*), NOT (a) regime-filtered training (deferred — breaks walk-forward contiguity). Hook: add bull/bear subscores to Mode-D candidate scoring ([faye.py:7331](crypto_trading_system_faye.py#L7331)); S handoff is the bigger piece. **Precondition diagnostic DONE** (cached 5h/8h sigs, sma48>sma100): regime-matching is a STRONG lever — `8h/5h +22.91%` (live) vs `5h/8h swapped +3.50%` (−19pp), `8h-only +16.90%`, `5h-only +7.94%` → **8h carries bull, 5h's value is concentrated in bear** → real headroom for per-regime model choice. **Step 1 = cheap standalone proxy** (no engine edit) `tools/regime_candidate_scan.py`: pin detector, sweep combo×window×gamma (features fixed to prod set), `generate_signals` per candidate over `--replay 960`, score overall/bull/bear (ret×WR), rank 3 ways → **does bull-best ≠ bear-best ≠ overall-best (radical or marginal)?** Both 5h+8h, ~1.5-2.5h (user-approved scope). Probe-timing first to size grid. If it reorders meaningfully → build the Mode-D subscore engine change + S pool-handoff. **960h SCAN DONE 2026-06-16** (24 cand, combo×window, gamma+features fixed to prod): per-regime winners DO differ — 5h overall XGB+LGBM w300 / bull XGB+LGBM w250 / **bear RF+LGBM w300** (bull~bear Spearman **−0.07**); 8h **bull XGB+LGBM w250** / bear RF+XGB w300 (corr +0.25). Reorder real, both regimes want LONGER windows than prod (5h w169→w300, 8h w163→w250). **BUT the gated head-to-head FAILED to generalize** (`tools/rst_specialist_vs_prod.py`, specialist-longer-windows vs prod, same live gate both arms): 1wk **+4.12pp** (7-9t=noise), 1mo +0.94pp (~40t), **2mo −9.17pp** (84-86t=reliable: PROD +27.68% vs SPEC +18.51%). **Root cause (user-spotted): scan SELECTED on `--replay 960` (recent ~40d)** → 1wk/1mo fall INSIDE the selection window (SPEC wins by construction); 2mo extends 480h before it → out-of-sample → SPEC loses. Recent-window overfit (echoes C01 vol-scaled). **NOW RE-TESTING `--replay 1440`** (`bnd6wwmfc`, ~4.5-5h, separate `_1440h.csv`): select on the FULL window (apples-to-apples w/ prod's 1440h pipeline) — if per-regime winners STILL differ from prod → real full-window headroom (→ build Mode-D subscore + S handoff); if they converge to prod-like windows → the reorder was pure 960h overfit, Idea 2 shelved. **⚠️ live config drifting EXTERNALLY 2026-06-16 (13:42, 15:02 writes — desktop/Drive, NOT local; laptop restores not sticking; bull.shield drifted OFF). Manage live config from the DESKTOP.** |
| 🔥 **P0 (0613)** | **Engine-folder cleanup — DONE today + QUEUED logging code-change (blocked on engine-idle).** ✅ **DONE 2026-06-13**: archived 10 orphaned forks → `ARCHIVED/2026-06-13_engine_forks/` (`crypto_trading_system_faye_refineopt.py`, `_reftest.py` + `tools/refine_only.py`, `_feattest.py` + `feattest_features.py`, `hardware_config_pre_faye_workers_20260603_000436.py` + their `models/config_faye_reftest` + `models/config_feattest` dirs); deleted ~26M stale scratch (`_pit_workdir`, `charts`, `models/config_g_desktop_nearlive`, `models/config_embargo_ab_*`); pruned `logs/` **1.5G→552M** (621 stale-empty + 1619 `ed_v1` >14d); created empty `logs/{trader,hrst,misc}/`. KEPT `data_asof_20260503` (user) + `__pycache__` (active/regenerating). Mock convention locked: every mock uses `crypto_trading_system_faye_mock.py` (memory `feedback-canonical-mock-file`). **⏳ QUEUED — blocked until engine IDLE (no fee-A/B / no desktop-RST importing `faye.py`; editing mid-import corrupts MP workers):** **(1) LOGGING CODE-CHANGE (route new logs + kill empty-turds).** Adversarial-verified gotchas — MUST: ① add **top-level `import re` to BOTH `crypto_trading_system_faye.py`** (currently only local @:2072) **AND `crypto_trading_system_ed.py`** (absent) — else the `re.sub` in the new log block is a **NameError at import** that kills every engine run AND silently disables the shadow (Rule 23). ② gate the log `open()` in faye with the **existing** idiom `not _FAYE_LIBRARY_MODE and _FAYE_IS_MAIN_PROCESS` (NOT a new `__name__` check — that was the buggy draft); in ed add `_ED_IS_MAIN = multiprocessing.current_process().name=='MainProcess'` then gate `_ED_IS_MAIN and __name__=='__main__'`. ③ route: engine→`logs/hrst/faye_<MODE>_<ASSET>_<ts>.log`; trader/optimizer/funding (`start_ed_v2.bat`/`start_optimizer.bat`/`start_funding_carry.bat` + `crypto_live_trader_ed.py:156`)→`logs/trader/`; research harnesses→`logs/misc/`. ④ TOPOLOGY: live trader imports **faye** (`crypto_live_trader_ed.py:61`), ed is the **shadow's** dep (`crypto_live_shadow.py:77`) → **smoke-test a short engine run + trader start + shadow cycle**, not just an HRST, before relying on it. Then **USER restarts bots** to adopt new paths. **(2)** delete `__pycache__` (trivial, once idle). **(3)** move ~1194 existing flat log survivors into `trader/hrst/misc` (deferred — some held open by running jobs; do at restart). **(4) optional/low:** archive closed `_idea_patchers/` **per-file** vs the ARCHIVED_LOG C-scoreboard (⚠️ 35 untracked C60-C82 are live inputs to `tools/test_v3_lit_batch_C60_to_C82.py` → do NOT blanket-move). Full read-only audit: workflow `wf_71d075d9-629` (2 adversarial passes; DELETE/ARCHIVE lists confirmed safe, logging draft caught + corrected above). | **Laptop edit → Desktop sync; USER restarts bots** | ⏳ **QUEUED** — blocked on engine-idle (fee-A/B LT0005 finishing). Apply the code-change the moment `faye.py` has no importer running, smoke-test, then user restarts. |
| 🔵 **P1 (0613)** | **BUG — standalone Mode V refine-dispatch DEADLOCKS (not concurrency, not parallel-horizons); HRST is the workaround.** Symptom: standalone `V ETH 5,6,7,8h` hangs at STEP 2 `_refine_top_configs` → `dispatching N task(s), pool=6, K_parallel=5`, on the FIRST horizon (5h), and never returns. **Reproduced on TWO independent runs — frozen at the identical line:** concurrent-with-RST (`logs/ed_v1_20260613_133223.log`) AND **SOLO (`logs/ed_v1_20260613_203037.log`, 20:30, run completely alone)**. STEP 1 (loky/joblib `_run_parallel_backtests`) COMPLETES both times → the hang is purely the STEP-2 refine ProcessPool. **Ruled out:** (a) concurrency — solo hangs too; (b) parallel-horizon fan-out — `run_mode_v` ([faye.py:9087](crypto_trading_system_faye.py#L9087)) loops horizons SEQUENTIALLY and dies on 5h, never reaching 6/7/8h (the "all 4 at once" perception was the hang). **PRIME SUSPECT (code):** `_refine_top_configs` creates a fresh `ProcessPoolExecutor` at [faye.py:10145](crypto_trading_system_faye.py#L10145) and blocks on `as_completed` ([:10169](crypto_trading_system_faye.py#L10169)) with **NO loky shutdown between STEP 1's joblib pool and this ProcessPool** → on Windows a live loky reusable-executor + a new 6-worker ProcessPool (workers re-import faye.py) deadlocks. HRST avoids it because its Mode-D phase runs+cleans a ProcessPool BEFORE the refine; standalone V's refine ProcessPool is the FIRST in the process, on top of a live loky pool. **REPRO (cheap, when idle):** solo `python crypto_trading_system_faye.py V ETH 5h --replay 1440 --no-persist` → expect hang at "dispatching" if confirmed. **FIX candidates (when engine IDLE — do NOT edit faye.py mid-run):** (1) kill loky right before [:10145](crypto_trading_system_faye.py#L10145): `from joblib.externals.loky import get_reusable_executor; get_reusable_executor().shutdown(wait=True, kill_workers=True)` (mirror the cleanups at 3619/3720/10331); (2) add a per-future timeout to the `as_completed` loop → fall back to `_refine_top_configs_serial` on hang (no silent forever-block); (3) simplest safe: route standalone Mode V to the serial refine. ⚠️ **Possible prod link:** the 2026-06-13 audit flagged the live w150 models as "Grid-sampler, never Refined" — if this refine intermittently hangs/falls-back under HRST too, "refined" models may silently be Grid; verify when fixing. Workaround in use: **HRST** (user ran it) — its refine completes. | **Desktop (fix when idle)** | 📅 LOGGED 2026-06-13 — root-caused to the refine ProcessPool dispatch (reproduced solo); fix deferred until engine idle. |
| 🔵 **TESTING (0613)** | **Feature-family ablation (Experiment 2) — does removing suspect external families HELP the gated sim?** Tests whether dropping the model-NOT-selected "external/slow" families IMPROVES Mode D. 7 arms = BASE + remove-each-of {macro `m_`, onchain `oc_`, sentiment `fg_`+`vix_`, cross_asset `xa_`, stablecoin `stable_mcap_`+`whale_`} + ALL_external. The 5 model-SELECTED families (technical/temporal, deriv, volatility, logret, pysr) are NOT ablated (core signal — removing them would gut the edge). FAYE **mock** engine (`crypto_trading_system_faye_mock.py`), `D ETH 5h --replay 1440 --no-persist`, **snapshot-pinned** (`data/_v2_snapshot_ablation_20260613_112045` — drift-free, every arm identical data), MOCK_DISABLED_PATH per arm (enabled=true + disabled_prefixes=family; Grade-1 trim OFF so it doesn't confound; 7 sparse always-off). Robust metric: top-5-median return_pct among ≥20-trade configs; **HARMFUL (clean it) if removing gives Δ≥+5pp**. **Mode D only → immune to the P1 V-refine deadlock.** Driver `tools/feature_family_ablation.py` (resume-safe: skips arms that already have a grid CSV; UTF-8-safe prints). **State 2026-06-13:** 3 arms DONE + snapshot-pinned on desktop (BASE/macro/onchain), 4 remaining (sentiment/cross_asset/stablecoin/ALL_external) → ~5h. **CAVEATS:** 1-horizon + same-window deltas → a HARMFUL family MUST be confirmed on a 2nd horizon / forward window before any production removal; and this is the Mode-D **screen**, not the full gated regime sim — a non-DEAD result needs gated confirmation ([[feedback_screen_vs_gated_engine]]). Parse anytime: `python tools/feature_family_ablation.py --parse-only`. | **Laptop** | ⚠️ **NOT A CLEAN REMOVAL — caught by user 2026-06-14 (potential pysr leak, UNVERIFIED — earlier "fully contaminated" was an over-claim).** Arm-log check: the raw DIRECT `fg_`/`vix_`/`xa_` WERE dropped, but all 5 PySR were still COMPUTED + in the candidate pool (`PySR: 5 features added`, incl pysr_1 `xa_btc..` / pysr_4 `xa_nasdaq_corr10d` — **none skipped**, since the raw xa columns stay in df for eval) → xa was **reachable via pysr**. BUT the importance ranking shows only the *macro* `pysr_5` (1.5%), NOT the xa-pysr → whether the winning models actually leaned on the leaked xa is **UNCONFIRMED** (grid CSV stores only `n_features`, not names). So: not a *clean* "remove xa" test → the +7.55pp can't be trusted as such, but it is NOT proven contaminated either. So +7.86/+6.08/**+7.55pp** measured **"drop the RAW direct duplicates while PySR keeps the signal"** (⇒ the raw direct features are *redundant given pysr*), NOT "remove xa/fg/vix." **Removing them ENTIRELY (incl re-discovering PySR without them) is UNTESTED — could help OR hurt** (the model did well *with* the pysr-encoded signal present → full removal might cost return). `ALL_external` −6.07pp = macro/onchain/stablecoin are useful context (KEEP). The real test of full removal = the NEXT-row HRST — a **genuine open experiment, NOT +7.55pp-backed**. |
| 🎯 **NEXT (0614)** | **Re-run full HRST with `cross_asset` (`xa_`) + `sentiment` (`fg_`/`vix_`) DISABLED from the feature matrix — but KEEP downloading the data.** This is the FIRST VALID test of FULL removal (Trim B): the prior Mode-D ablation was **NOT a clean removal** (xa stayed reachable via PySR — *not proven* contaminated, but not a valid "remove xa" test either; see ablation row ↑), so the +7.55pp does **NOT** pre-validate this. **Genuine open experiment — could help OR hurt** (full removal also strips the pysr-encoded xa/fg/vix signal). The HRST's H→R→S→T→G is the gated test. **Production implementation:** **(1) downloads UNCHANGED** — `download_macro_data.py` keeps pulling `cross_asset.csv` + `fear_greed.csv` + VIX (data stays current → re-enable = a one-line config flip; kept available for carry/portfolio). **(1b) ⚠️ CRITICAL — re-run Mode P (PySR) WITHOUT xa/fg/vix FIRST.** The CURRENT PySR formulas EMBED these features (expressions referencing them: **5h 3/5, 6h 2/5, 7h 3/5, 8h 2/5** — e.g. 5h pysr_1 uses `xa_btc_usd_corr30d`+`xa_nasdaq_corr10d`, 8h pysr_2 uses `fg_zscore`), so disabling ONLY the direct features lets them **leak straight back through `pysr_N`**. Re-discover PySR for ETH 5,6,7,8h with `xa_`/`fg_`/`vix_` excluded from the input pool → clean symbolic features (Rule 14: regen pysr ⇒ retrain, which the HRST does; keep Rule 8 leakage check `discovery_method=historical`). macro-using pysr (`m_vix_*`/`m_dxy`) are FINE — macro stays. **(2) Exclude `xa_`/`fg_`/`vix_` from the FEATURE pool via an UNCONDITIONAL disable** — ⚠️ **NOT** by flipping `config/disabled_features.json` `enabled:true` (that ALSO re-activates the 65 Grade-1 `disabled_exact` = **−21.4pp**). Cleanest = add an `always_disabled_prefixes:["xa_","fg_","vix_"]` block (small mirror of the existing `always_disabled_exact` handling in `_load_disabled_features`/`_apply_feature_disable`); no-code alt = enumerate the ~20 `xa_`/`fg_`/`vix_` names into `always_disabled_exact`. NB `vix_` hits only `vix_normal` — the **useful macro `m_vix_*` is a different prefix → kept**. **(3) The HRST then trains/selects models whose `optimal_features` EXCLUDE them → at promote the live trader auto-excludes them (Rule 3, `feature_override=optimal_features`) — no live code change; they're computed-but-never-fed.** **Promote** only if trimmed-HRST ≥ current prod on the gated Mode-S/T REF (trader flat, Rule 19); verify live `optimal_features` contain no `xa_`/`fg_`/`vix_`. If worse → the screen didn't survive the gate → keep them. **BLOCKED on:** (a) current desktop HRST finishing (8h, home stretch); (b) ideally the **P1 refine fix first** (the HRST *completes* but runs slow — fix speeds it up, not required for correctness). **Don't run yet (user).** | **Desktop** (~7h, after P1 fix) | 📋 QUEUED — don't launch until current HRST done + P1 fixed |
| 🔵 **P1 (0609)** | **Model-combo + GB/LGBM hyperparam sweeps — GB+LGBM added to FAYE grid; tune LGBM, leave GB scrappy.** Model-combo retest: **LGBM+GB #1 of 25 combos** (+15.8%/APF 2.06, step=36) → **added to FAYE `GRID_COMBOS`, commit `048aada`** (on origin/main). GB sweep (640 cfg): **keep `min_samples_leaf=1`** — more reg HURTS the pair (diversity 0/10: best-solo GB = worst partner); only `lr 0.1→0.03` marginal (+7pp). LGBM sweep (288 cfg): **regularize** — `min_child 20→30` + `reg_λ 0→5` (+22pp; current LGBM ranks #276/288). **Lesson: tune the backbone (LGBM) for quality, leave the diversifier (GB) scrappy.** Detail ↓ "Logged 2026-06-09". | **Desktop/Laptop** | ❌ **VALIDATED & REJECTED 2026-06-09** — real-engine backtest (`tools/bt_lgbm_tune_8h.py`, ETH 8h, 2mo regime sim): GB+LGBM LOST to live RF+LGBM (**+43.5% vs +51.2%** @1440h) and the LGBM-reg tune REVERSED (tuned mc30/reg5 **+11.0%** vs current +43.5%). **GB+LGBM REMOVED from FAYE `GRID_COMBOS`** (back to RF+LGBM + XGB+LGBM). Screen rewarded regularization that erases conviction under the live conf gate ([[feedback_screen_vs_gated_engine]]). Re-add only after a step=1 / multi-window test beats RF+LGBM. |
| 🟢 **NEW STRATEGY (0610)** | **Funding-carry sleeve (delta-neutral) — backtested WINNER, paper bot live.** Pivot after exhausting the single-asset directional model (every feature/model/label/decision tweak fails the gated sim). Thorough crypto-quant landscape research → tested 3 orthogonal strategy classes: **(1) basis/funding carry WINS** (long spot + short perp; ETH Sharpe 15-24 friction-free / ~5 net per lit, APY 5-8%, maxDD <2%, funding +81% of time; BTC/LINK/XRP similar; BNB negative-funding→reverse; uncorrelated to directional ETH). **(2) stat-arb ETH-BTC FAILS** (−100%, neg Sharpe — pair trends not mean-reverts, ADF~0.05). **(3) cross-sectional momentum WEAK** (best Sharpe 0.73 @14d lookback, 6-coin universe too thin; needs ~30-50 coins). Paper bot `tools/funding_carry_eth.py` built (FREE Binance public data: premiumIndex+fundingRate+spot, NO keys, restart-safe, **single-instance lock** for the Drive-synced config/) via 2 workflows (empirical-API design + adversarial leakage/lag/accounting/robustness review — all fixes applied incl. entry phantom-credit, pagination, lag-1 backtest leak). **SEPARATE parallel sleeve, NOT embedded in trader** (diff venue, opposite ETH-spot intent, blast-radius). ⚠️ ETH funding currently NEGATIVE (bear) → carry PAYS now; earns when funding turns +. Detail ↓ "Logged 2026-06-10". ✅ **ARCHIVED_LOG scoreboard updated 2026-06-11** (C83 carry-active / C84 stat-arb DEAD / C85 xsec SHELVED / C86 trend-scanning DEAD); CLAUDE.md carry-sleeve note still deferred until paper validates. | **Desktop** (paper `--loop`) | 🟢 PAPER-TRADING — ✅ **correctness verified 2026-06-11** (ledger audit clean + **data-fidelity 3/3 settlements match Binance `fundingRate` exactly**, rate→8dp / mark→cent; pre-inception jittered boundary correctly skipped → re-confirms phantom-credit fix). NEXT: ~2-4wk forward record (ideally spanning a +funding stretch); live needs a perp venue (Coinbase Advanced eligibility / Binance / Bybit). |
| 🟢 **MONITOR (0611)** | **Sanity check made deterministic & reproducible — fixes the wobbling daily Telegram.** Root cause of "sanity gives different results / send is wrong": the engine-vs-trader **PARITY** check rebuilds features from CURRENT data → inherently non-reproducible (the `tail(30)` window shifts every hour **and** the freshest training rows get revised). **NOT a send bug** — the trader faithfully forwards each run's result. **Fix**: built `tools/validate_snapshot_replay.py` (replays the trader's OWN frozen point-in-time intermediates from `output/inference_snapshots.jsonl`: recompute signal+conf, assert == logged → **202/202 = 100%, deterministic, ~instant, immune to data revision**). Rewired `tools/sanity_check.py` to 3 checks: **[1] SHADOW + [2] SNAPSHOT REPLAY drive the verdict** (both deterministic); **[3] PARITY demoted to INFORMATIONAL** (real flips listed but no longer escalate — almost always data-revision artifacts). Also fixed the **SHADOW verdict to key off recent-48** (current correctness) not all-time cumulative — it was stuck FAIL 91.2% forever, dragged down by the already-fixed Jun 3-4 episode (Rule 23), while `last-20 fails=0`. `crypto_revolut_ed_v2.py` daily send: added `'SNAPSHOT'` to the Telegram grep. Detail ↓ "Logged 2026-06-11". | **Laptop edit → Desktop (Drive sync)** | 🟢 DONE — verdict fix is live on the **next daily run** (sanity_check spawned fresh, no restart); the `'SNAPSHOT'` Telegram header line needs a **trader restart** to appear. |
| 🟦 **FRONTIER (0611)** | **Portfolio layer — the "real frontier" (turn 2 sleeves into a book).** From the 2026-06-11 model-vs-field analysis: the next Sharpe comes NOT from a better RF+LGBM (directional core is a proven local optimum — every lever fails the gated sim) but from **breadth + allocation**. Build (a) a **risk-budget allocator** (rolling Ledoit-Wolf cov + Equal-Risk-Contribution) across the directional sleeve + carry sleeve, and (b) a pod-shop-style **BOOK-level capital drawdown governor** (cut a sleeve's allocation after a rolling-DD threshold, reallocate toward what's working — on EQUITY CURVES, **NOT** overriding individual SELL timing, which is the directional edge). Math: book Sharpe ≈ IC·√(independent bets) (Grinold); carry's marginal value is **almost entirely its near-zero correlation** to directional → only pays if you size by RISK contribution → combined Sharpe can exceed either sleeve alone. Detail ↓ "Logged 2026-06-11 — model-vs-field". | **—** | 🔒 **GATED on carry going live** (C83, currently paper). Effort medium (light compute; cost is process + monitoring). The strategic destination, not an immediate task. |
| 🟡 **P2 PARKED-ACTIVE (0611)** | **Vol-targeted position sizing — gated A/B (the "free Sharpe" item).** From the 2026-06-11 model-vs-field analysis. Size the long leg 0→100% by **inverse 24-48h realized vol** (constant ex-ante risk). Most evidence-backed single-asset sizing tool (Harvey/Man 2018; Moreira-Muir J.Finance 2017): **~+0.4 Sharpe on a single crypto asset** via the leverage effect ETH strongly exhibits. **CRITICAL — do NOT let Kelly (C22) pre-kill this**: vol-targeting sizes by **RISK** (inverse realized vol), a DIFFERENT axis from Kelly's confidence gradient → needs **NO calibration** (your proven-dead axis) and does **NOT touch the directional conf gate** (only modulates notional GIVEN a BUY). Free orthogonal Sharpe from rolling realized-vol already on disk. **Blocker: requires relaxing binary all-in/out on the long leg only.** Must be judged by the GATED sim, not a screen ([[feedback_screen_vs_gated_engine]]). Detail ↓ "Logged 2026-06-11 — model-vs-field". | **Laptop/Desktop** | 🟡 **TESTED 2026-06-11 — PROMISING, PARKED (kept active, may become handy).** Tool: `tools/bt_vol_target.py` (faithful gated A/B — live signals generated ONCE, vol-sizing applied as a transform so both arms trade identical entries/exits; causal `size = clip(trailing-720h-median-vol / realized-vol, 0, cap)`). **2mo standard window (B&H −24%):** spot cap=1.0 **+25.95% vs +22.79% baseline** (+3.15pp), **Sharpe 5.55 vs 4.17** (+1.38), **maxDD 9.93% vs 13.72%** (−3.8pp), avgExpo 0.88 → **Pareto-better on every axis**; 90d same shape. Leverage cap=2.0 → **+38% / Sh 6.57** (needs a margin venue). **KEY NUANCE — it's a RISK tool, not a RETURN tool on no-leverage spot.** Sub-period split (3×20d): the return gain is **DOWNSIDE-DRIVEN** — concentrated in the deepest-DD chunk (**+2.97pp**); in the up/winning chunks it **TIED (+0.03pp)** or **gave up a little (−0.70pp)**. So in a sustained BULL it would likely give up some return (88% avg expo) while still cutting DD. **OPEN (do before any deploy): bull-window test** — add `--end-offset` to the tool, run a +30-50% ETH 2mo window to confirm the bull-market return give-up. **Why KEPT ACTIVE:** (a) clean drawdown-reducer as-is; (b) becomes a real return-adder IF a margin venue appears (e.g. the carry perp venue → leverage cap>1 added +13-21pp in tests); (c) handy any time you want a smoother equity curve. **NOT a screen — this is the real gated live path.** Detail ↓ "Logged 2026-06-11 — model-vs-field". |
| 🔵 **TESTING (0611)** | **C61/C62/C67/C75 feature gated-A/B — RUNNING on laptop.** From the 2026-06-11 feature-scout (thorough web research → deduped vs C01-C86 → the 4 highest-prior *queued* features). `python tools/test_v3_lit_batch_C60_to_C82.py --only C61,C62,C67,C75` (Mode D ETH 5,6,7,8h `--replay 1440 --no-persist`, top-APF delta vs a freshly-refreshed baseline; PASS ≥+5pp / MARGINAL / FAIL / DEAD). **C61** vol-of-vol (cleanest shot — conditioning version of vol feats the model already trusts), **C62** DXY accel, **C67** Connors RSI, **C75** stablecoin-supply RATIO (⚠️ minor same-date stablecoin merge → a PASS would need a daily-lag re-test). **C73 dispersion-ratio HELD** (needs the standby universe live). Prior near-zero (the C32-C40 batch went 0 PASS) — value is closing them definitively + catching any surprise. Patchers reviewed causal; harness smoke-confirmed on the post-lag-fix engine. Started 23:16 CEST 2026-06-11. | **Laptop** | ⚠️ **FIRST RUN INVALID (2026-06-12) — reported "4/4 PASS" was a FALSE POSITIVE from 2 harness bugs, NOT signal.** (1) `refresh_baselines()` ran `D ETH 5h 6h 7h 8h` as ONE cmd → engine only ran the LAST horizon (8h) → 5/6/7h baselines stale (June-8 data) → patched runs "beat" a stale baseline on 3/4 horizons. (2) **C61 CLOSED as MOOT** — `vol_of_vol_8h` + `vol_of_vol_24h` are ALREADY native engine features ([ed.py:1018-1019](crypto_trading_system_ed.py#L1018)); the patcher crashed on the duplicate (read ancient May-10 grids → fake +10.88). The C60-C82 queue mislabeled C61 "untested." **Harness FIXED 2026-06-12**: refresh now per-horizon + patcher template dedups feature_cols (native-dup → skip, not crash). Stale C61 grids deleted. Baselines rebuilt clean (06-12 08:32-08:56). ✅ **CLOSED 2026-06-12 — NO SIGNAL (screen-noise), not promoted (user chose option b).** Valid recompute vs fresh baseline: C62 top1 **+4.16**/top5 +5.34, C67 top1 **+8.12**/top5 +7.10, C75 top1 **+6.51**/top5 **+1.10** — but every "PASS" rides on ONE horizon's overfit APF outlier (C62 8h=**28.6**, C67 6h=**31.3** vs realistic 1-3), per-horizon deltas swing **+20 to −5** (a real feature is consistent), and C75 collapses on the robust top-5. **No clean, consistent signal — consistent with the feature-add family being exhausted (0 real PASS).** **C61 closed MOOT** (vol_of_vol_8h/24h already native, ed.py:1018). **HARNESS LESSON: the v3_lit top-1-APF screen is structurally too noisy for feature screening** (MIN_TRADES=8 lets overfit few-trade configs inflate APF to ~30 and dominate the top-1) — **filter configs to ≥~20 trades or use top-5-median next time** (noted in `tools/test_v3_lit_batch_C60_to_C82.py` header). |
| ✅ **DONE 0612** | **Restart trader to load the shadow closed-bar fix** (`crypto_live_shadow.py:220`). Shadow match crashed 100%→48% Jun 3 because fix #2 (closed-bar) wasn't mirrored into the shadow monitor — proven 23/23 (+1h forming-vs-closed). Code fixed; the live number only recovers on trader restart. **After restart, confirm `config/shadow_signal_diff.csv` returns to ~100%.** NOT a model change, NOT a trading bug — monitor only. | **Desktop** | ✅ **DONE 2026-06-12** — trader restarted (user); shadow should return to ~100% on the next cycle (confirm via `config/shadow_signal_diff.csv`). |
| ✅ **DONE 0612** | **Verify ~0.078 BTC: real or phantom** — BTC position file has a malformed April SELL (price=0) that set `state:cash` without zeroing `base_amount=0.0785`. `get_balances()` only echoes the exchange, so the `/status` display fired on a real reported balance → **~$5k BTC may be untracked on Revolut X.** Run `python crypto_revolut_ed_v2.py --balance`. If ~0.078 BTC present → sell it; if 0 → zero the stale position file. (Display already fixed to hide non-enabled coins.) | **Desktop** | ✅ **DONE 2026-06-12 — PHANTOM confirmed** (user: no BTC invested; the April 6 "manual market sell - disabling BTC" did happen, only the bookkeeping was stale). Zeroed `base_amount` (0.0785→0), `usd_invested` (5452.55→0), `entry_price`, `entry_time` in `config/position_ed_v2_BTC.json`; trade history kept as audit trail; `state` was already `cash`. Backup: `config/position_ed_v2_BTC_pre_zero_20260612_215054.json`. Safe (BTC `enabled:false`, file untouched by trader since April). |
| 🔵 **P1 (0605)** | **2mo vs 4mo HRST decision** — `python tools/compare_prod_vs_4mo.py` (running on Desktop). Production (bull 6h@65/bear 5h@80) vs 4mo-HRST (`ed_v1_20260604_075223`: bull 5h@70/bear 5h@65) over 720h+168h. Read 720h as signal, 168h as tiebreaker (~10-25 trades). Verdict decides whether future HRST uses 2mo or 4mo window. | **Desktop** (~45-90 min) | 🟢 running |
| 🔵 **P1 (0605)** | **Embargo-sensitivity sweep on 1h (+4h)** to settle short-horizon viability — `FAYE_EMBARGO_OVERRIDE=4/8 python crypto_trading_system_faye.py D ETH 1h --replay 1440 --no-persist`. WR collapses with bigger embargo → leak/overfit (kill 1-3h); WR holds at embargo=8 → real edge. DV WR gradient: 1h 92-98% / 2h 88% / 3h 85% / **4h 76% (in-band, plausible)** / 5-8h 74-83%. 4h worth pursuing, 2-3h borderline, 1h inflated. | **Laptop** | ❌ **DEAD 2026-06-11 — forward dry-run settled it** (`tools/dryrun_1_4h.py`, 5d / 75 cycles, live core, no embargo sweep needed): **1h −1.99%** (3/17W = 18% WR), **2h −0.54%** (3/7W = 43%), **3h +0.98%** (1/5W = 20%, one lucky trade = noise), **4h −1.05%** (3/8W = 38%). The high DV WR (1h 92-98% / 4h 76%) was **LEAK/OVERFIT** — leak-free forward paper collapses to losing. **Whole 1-4h band dead; production 5-8h unaffected. 4h's "worth pursuing" hope did NOT survive forward testing.** Dry-run loop can be stopped. |
| 🔥 **P0** | **Live WR/P&L monitor on new FAYE config** — first 1-3 days = sanity window, 2-4 weeks = real validation | **Desktop** (passive) | 🕐 STARTED 2026-05-31 14:22. Watch: signal cycle at next hourly tick using bull=6h+bear=8h (was 5h+8h); WR tracking close to Mode V Step 3 predictions (79-83%); total return tracking the +55%/+37%/+46% scale on similar period. **Rollback trigger**: live WR <60% over 7+ trades or persistent negative trades week 1 → revert to G_narrow archive. |
| 🔥 **P0 (this week)** | **FAYE engine-vs-trader parity verification on the NEW config** — confirm trader produces identical signals to FAYE engine for ETH 6h+8h after the May 31 promotion | **Desktop** (~15 min) | 📅 BLOCKED on 24+ hours of trader signal_log accumulation (trader started 14:19, ETA ready ~14:00 June 1). Then run `python tools/validate_core_against_signal_log.py --samples 30 --recent-only --cpu-lgbm`. Acceptance: 0/N real BUY↔SELL flips (HOLD-threshold DIFFs OK). Failure modes to watch: (a) PySR feature drift — verify `optimal_features` columns in `models/crypto_ed_production.csv` only reference PySR features that exist in `models/pysr_ETH_{6,7,8}h.json`; (b) embargo handling — trader must NOT use embargo (no future labels in live), backtest uses `embargo=horizon`; (c) regime detector parity — trader's `tsmom_672h = log(close/close.shift(672))` must compute correctly from live data history (no gaps in the last 672h). **If divergence found**, treat current FAYE production as suspect, investigate before next promotion. Until verified, today's parity (May 30 G_narrow test, 0/8 flips) only proves the OLD config's parity, NOT the new FAYE picks. Step 6 engine refactor (P2 below) would formally unify backtest+live codepaths but is pending — until then, parity is by-construction-not-formally-proven. |
| ✅ DONE 2026-05-31 | **FAYE promoted to production** (was P0 BLOCKED) | — | ✅ Promoted 14:22 CEST. Archive: `archive/g_narrow_live_20260531_142202/`. Trader hot-reloaded and picked up new config. Gates (a)+(c) deferred to post-promotion validation; gate (b) hard A/B remains queued in P3 as nice-to-have not blocker. |
| ✅ DONE 2026-05-31 | **FAYE RST ETH 6h,7h,8h --replay 1440** | — | ✅ DONE 14:14 — Mode R picked tsmom_672h × 6h/8h regime pair (REF +72.75% / +76.05% / B&H +61.88%). Mode S optimized confs at 65%/65%. Mode T converged shields=OFF + rally cooldowns + min_sell_pnl=0 + max_hold=10h. Total RST wall ~80 min. |
| ✅ DONE 2026-05-31 | **FAYE H ETH 6h,7h,8h --replay 1440** | — | ✅ DONE ~14:30 — Winners: 6h RF+LGBM w=150 γ=0.996 f=15 **+55.30%** WR=79.3% (Grid); 7h RF+LGBM w=151 γ=0.9992 f=15 **+37.54%** WR=82.4% (Refined); 8h RF+LGBM w=155 γ=0.9997 f=11 **+46.37%** WR=83.0% (Refined). 5 bugs found+fixed mid-run (#15-18 + sleep-guard). |
| ✅ DONE 2026-05-31 | **FAYE DV ETH 5h --replay 1440** (evening) — adds 5h to FAYE production CSV | — | ✅ DONE 18:03 — 5h winner: ETH RF+LGBM w=150 γ=0.999 f=15 **+41.93%** WR=79.7% (Grid). **Now lives in `models_faye/crypto_faye_production.csv` as the 4th row** alongside 6h/7h/8h. **NOT promoted to LIVE `models/crypto_ed_production.csv`** — LIVE still has only 6h/7h/8h. Reason: subsequent RST evaluation (next row) confirmed 5h is not selected by Mode R's regime detector for any bull/bear pair under the current data window, so promoting 5h to LIVE would be dead weight (regime config never references it). 5h is "validated alternative" kept on file in case future market data shifts make 5h-inclusive regimes optimal. To promote later: surgical splice via `tools/promote_v3_5h_winner.py` (or similar) — or just copy the 5h row from `models_faye/crypto_faye_production.csv` into `models/crypto_ed_production.csv`. |
| ✅ DONE 2026-05-31 | **FAYE RST ETH 5h,6h,7h,8h --replay 1440 --no-persist** (evening, validation of LIVE config) | — | ✅ DONE 19:38 — Picked EXACTLY the same config currently LIVE: tsmom_672h × bull=6h@65% × bear=8h@65%, shields=OFF, rally_cd=ON, min_sell_pnl=0, max_hold=10h. Mode S top: **+70.18% / 88 trades / 84% WR** (B&H -5.38%, alpha +75.56pp). 5h/8h regime pairing was ranked outside top-10 (rank 11 for tsmom_672h × 5h/8h with +8.33% on 8 trades). **Conclusion**: adding 5h candidate did NOT shift the optimal regime — current LIVE FAYE config is validated against a wider candidate pool. Output in `models_faye/crypto_faye_production_noprod.csv` + `config_faye/regime_config_faye_noprod.json` (research files, not promoted). |
| 🔥 **P0** | **Shadow mode continuous match-rate check** — primary live correctness gate | **Desktop** (passive observation) | 🕐 IN PROGRESS — every 1-2 days run the match-rate query; any drop below ~99% = NEW bug to investigate. |
| ✅ CLOSED 2026-05-30 | **Counterfactual backtest on wider window** — superseded by engine-vs-trader parity test | — | ✅ CLOSED — original goal was statistical proof of macro_cache fix's economic impact. Yesterday's parity test (`tools/validate_core_against_signal_log.py --samples 30 --recent-only`) gave a more direct answer: 22/30 direct MATCH, **0/8 real BUY↔SELL flips**, all DIFFs HOLD-threshold boundary cases. Engine and trader codepaths agree on direction whenever both produce one. Wider counterfactual would only re-measure what the parity test already confirmed. Live WR/P&L monitor continues to validate economic impact in real time. |
| ✅ **DONE 0531** | **Embargo A/B test** (`tools/embargo_ab_test.py --mode=both`) | **Laptop** | ✅ DONE 2026-05-31 — same ETH models, Mode T: embargo=horizon **REF +69.09%** vs embargo=0 **REF +174.20%** = **+105pp leakage lever**. +174% is leakage-inflated (embargo=0 in a *backtest* reintroduces label-overlap leakage), NOT a live target. Confirms embargo essential in backtest/selection + live correctly uses NONE (Rule 9) + embargo is NOT the live-vs-backtest gap source → Step 6 still warranted. See detail section below. |
| ✅ **DONE 0611** | **Daily-data availability lag fix** (5th live-vs-backtest divergence; see [[project_daily_data_lag_fix]] memory) — daily features were same-date-merged → backtest clairvoyance; diagnosed 2026-06-01 via the 07:00 ETH parity flip (`oc_mvrv_chg1d` +0.53%→−0.59% once May-31 on-chain published ~14:01). **FIXED in FAYE**: `DAILY_MERGE_LAG_DAYS=1` (on-chain `ONCHAIN_MERGE_LAG_DAYS=2`); PySR decoupled ed→faye + isolated to `models_faye/`; verified 202 cols / 0 violations. **Cascade pending**: PySR 5/6/7/8h all discovered ✓ → `DV ETH 6h,8h` on lagged feats → `RST` validate → **promote: MUST mirror lag into `crypto_trading_system_ed.py` build_all_features + `copy models_faye\pysr_*.json models\`, trader flat, else live drift (Rule 14)**. ed.py deliberately NOT lagged until then. **Lagged-parity check**: `validate_core_against_signal_log.py --engine faye` added; watcher auto-runs it when DV 8h rewrites the 8h config (does the 07:00 flip resolve under the lag?). | **Desktop** | ✅ **DONE — row was stale.** ed.py serving path **already lagged since 2026-06-04** (commit `a7cb7c9` "ed engine: mirror FAYE daily-merge lag onto serving path (trader+validator)"): `DAILY_MERGE_LAG_DAYS=1` / `ONCHAIN_MERGE_LAG_DAYS=2` at `crypto_trading_system_ed.py:501-502`, used in `build_all_features` (`:1345` / `:1393` / `:1504`). Live models are FAYE-trained (lagged, promoted 05-31) → **train+serve consistent**; pysr copied to `models/` 06-08. The 07:00 `oc_mvrv_chg1d` BUY→SELL clairvoyance flip is resolved on the live serving path. The "DV ETH 6h/8h → mirror to ed.py" cascade was effectively completed by `a7cb7c9`. **Memory `project_daily_data_lag_fix` updated 2026-06-11** (was "ed.py NOT lagged"). |
| 🔥 **P1** | **TODO 0519B-G1** — `deriv_oi_*` re-enable A/B test | **Desktop** (~6h, off-hours) | 📅 **PARKED until ~2026-06-25** — OI data starts 2026-04-26 (~36 days as of 06-01, near-complete hourly). Needs 60 days for full 2-month (`--replay 1440`) coverage → un-park 2026-06-25. LGBM-NaN-safe to run earlier but OI signal too thin <60d (treat a null result as "still thin," not dead). Keep parked; run when 2 months of OI exist. |
| ✅ **IMPL DONE (FAYE)** / ⚠️ near-live ≠ exact | **Step 6 engine refactor** — make backtest match live semantics so HRST predicts live | — | ✅ **SEMANTICS implemented natively in FAYE NEAR_LIVE_MODE** ([crypto_trading_system_faye.py:863-869](crypto_trading_system_faye.py#L863), always-on): `step=1`, `signal_mode=ternary`, `na_policy=mean_last_10`, `return_probas=True`. So FAYE HRST WR is far closer to live than ed.py's inflated ~85%. **BUT backtest ≠ live bit-identically — 2 residuals remain:** (1) `na_policy=mean_last_10` vs live `ffill` ("near-equivalent," not identical); (2) backtest keeps `embargo=horizon` (correct, anti-leakage) while live uses NONE → backtest model trains on `horizon` fewer recent rows than live, so its WR is a close PROXY, not an exact predictor — and no refactor removes this (removing backtest embargo = re-introduce leakage). Empirically the `--engine faye` parity match is ~73-87%, **not 100%** — that gap IS the residual. **VERDICT 2026-06-02: ACCEPTED, do not reopen** — embargo is *theoretically unfixable* (López de Prado purge/embargo is definitional to leakage-free OOS eval; live correctly omits it) AND *conservative* (live's fresher window only helps → backtest WR is a lower bound); na_policy is negligible (NaN rare at the frontier). Old `docs/STEP_6_ENGINE_REFACTOR.md` was the legacy ed.py plan. |
| 📋 **P2** | **Re-run HRST on refactored engine** to get realistic backtest WR | **Desktop** (one HRST, ~7h) | ⏸ BLOCKED — depends on Step 6 done |
| 📋 **P2** | **TODO 0519B-G2** — orderbook + IV re-enable A/B test | **Desktop** (~6h) | 📋 **READY ~2026-06-18** — orderbook+IV start 2026-04-19 (~43 days as of 06-01, ~85% hourly density), so 60 days lands 2026-06-18 — a week BEFORE G1 (06-25). "Depends on G1 outcome" is a soft prior, NOT a data block: can run G2 standalone 06-18, or batch with G1 after 06-25 and use G1's result to decide. **See the 0612 clean-window row below — run these on `--replay 960`, not 1440.** |
| 🔵 **P2 (0612)** | **Clean-window re-enable A/B for the 7 quarantined sparse features** (`deriv_oi_chg1d/3d/zscore`, `ob_imbalance`, `spread_bps`, `avg_iv`, `iv_skew`; the `always_disabled_exact` block in [config/disabled_features.json:72](config/disabled_features.json#L72)). **Data-issue RE-CHECKED 2026-06-12: the original sparsity has largely resolved** — now 22% NaN (OI) / 10% (orderbook+IV) over the last 1440h, all comfortably under the 0.70 auto-filter (`SPARSE_NAN_THRESHOLD`); and **live/forward = ~0% NaN** (feeds captured hourly now). The remaining NaN is pure history-before-feed-start, **time-correlated at the OLD end** of any window (feed starts: OI 2026-04-26, orderbook/IV 2026-04-19) → real risk is LGBM latching "missing → early-era regime," not the raw count. **FIX: restrict the A/B to the covered span — `--replay 960` (~40d) is NaN-free for all 7 at once** (orderbook/IV clean ≤1297h, OI-raw ≤1146h, OI-zscore-168h-warmup ≤978h). Run both arms (feat ON/OFF) on the SAME short window (Rule 17; ~40d = fewer trades, less power, can't compare abs return to a 1440h baseline). Backfill is DEAD (Binance OI hist ≈30d cap, already exceeded; orderbook/IV ephemeral trader-captured → forward-only). ⚠️ **Promotion gotcha**: over a 6-month training window (`MAX_DIAG_HOURS=4320`) OI hits ~73% NaN → auto-dropped by the 0.70 filter → would need a shorter train window or more history to actually carry in prod. A truly clean `--replay 1440` isn't available until ~2026-07-12. **This answers "would they break training" (no); whether they HELP is still the gated-A/B question.** **★ PURPOSE = early go/no-go on the WAIT.** The feeds don't reach a full clean 1440h until ~07-12 (G2 60d ~06-18 / G1 ~06-25). Rather than park until then, run the 40d clean A/B NOW: if 960h shows **zero signal → don't wait, close G1/G2**; if **promising → confirm later** at the fuller window with more trades/power. Asymmetric read: a clear positive at 40d is encouraging; a clear null is low-power but low-prior → likely not worth the wait. So this row effectively **supersedes the "park G1/G2 until the data matures" plan** — the 960h read makes that call cheaply, today. | **Laptop/Desktop** (~40d A/B per group) | 📅 **READY NOW** on the clean window — the decision gate for G1/G2. Parked behind BTC Mode P (resource order only). |
| 🔵 **NEW (0612)** | **BTC directional retry — Mode P running, HRST next (to settle the stale 2026-04-06 disable).** Hypothesis (user): BTC less volatile → maybe more predictable. **Data check 2026-06-12**: BTC ann-vol **38.5%** vs ETH 53% (true, ~27% lower) BUT bars clearing the 0.22% fee-aware label = **32.5% vs ETH 52.3%** → lower vol is a **HEADWIND** in this fee-aware/all-in-out engine (thinner tradeable-move set) — the likely cause of the April-6 disable (45% WR, avg loss>win). "More predictable" is unsupported — BTC is the most efficient/arbitraged crypto market (harder, not easier). ⚠️ **Diversification value LOW**: BTC ~0.8 correlated to the ETH sleeve → "more depth in the same directional bet," not breadth (carry/vol-targeting are the real orthogonal adds). BUT the April-6 verdict is **STALE** (pre-FAYE, pre-daily-lag-fix, pre-K5, pre-200-feature set) → a clean re-test is defensible + cheap. **IN MOTION**: `python crypto_trading_system_faye.py P BTC 5,6,7,8h` running (BTC PySR was stale Mar–Apr → refreshing to `models_faye/pysr_BTC_*.json`; 5/6/7h written, 8h dispatching as of ~21:43). **NEXT**: `HRST BTC 5,6,7,8h --replay 1440` on FAYE → the **GATED WR/return is the real test** (importance ≠ performance). Temper expectations per the fee-clearing headwind; the gated sim decides PROMOTE vs re-shelve. | **Desktop/Laptop** | 🟢 **Mode P running** |
| ✅ **DONE 0612** | **PROD BUG FIXED — FAYE grid-CSV escaped isolation into LIVE `models/`.** Found 2026-06-12 by the mock-isolation contamination audit (while building the 960 A/B mock). [crypto_trading_system_faye.py:5589](crypto_trading_system_faye.py#L5589) hardcoded `os.path.join('models', 'crypto_ed_grid_*.csv')` instead of `FAYE_MODELS_DIR` — a leftover from the ed.py→FAYE consolidation (line 1510 `MODELS_DIR = FAYE_MODELS_DIR  # was 'models'` shows the intended substitution; THIS line was missed). So every FAYE Mode D wrote its grid-dump CSV into the **live `models/`** dir, not the isolated `models_faye/` (seen live: `models/crypto_ed_grid_ETH_7h/8h.csv` written by today's HRST). **Severity LOW** — write-only inspection dump, **NEVER read** by FAYE (grep: only 1 write + comment + help, 0 reads) or the trader (trader reads ONLY `models/crypto_ed_production.csv` by exact name — [crypto_live_trader_ed.py:492](crypto_live_trader_ed.py#L492); the grid is never touched) → cosmetic isolation-leak/clutter, **zero live-trading risk**. **FIX**: `'models'` → `FAYE_MODELS_DIR` (1-line, git diff = 1 line, byte-compiles). Next FAYE run isolates correctly; the **running HRST is unaffected** (in-memory code). ed.py's identical line (4665) is correct-by-design (its `MODELS_DIR` literally IS `models`) — not touched. Leftover `models/crypto_ed_grid_*.csv` are inert clutter, deletable separately. | **shared engine** | ✅ fixed + verified |
| 📋 **P2** | **Verify feature importances stable** after cache fix — re-run Mode V importance ranking, compare to pre-fix | **Laptop** (~30 min) | 📋 OPTIONAL — sanity check that the same features still rank high once they actually vary across time |
| 🚀 P3 | **Continuous macro archeology** — capture daily snapshots so future PIT validation has clean coverage | **Desktop** (cron, 5 min/day) | 📅 NEW — set up nightly `python tools/drive_archeology.py --preset all` so the next time we need PIT, drift is bounded |
| 🚀 P3 **AFTER FAYE IN PROD** | **Counterfactual: ffill vs mean_last_10 on trader's actual May 1-28 hours** — measure exact signal-flip count, not estimate | **Desktop or Laptop** (~1-2h dev + ~15-30 min run) | 📅 DEFERRED 2026-05-29 — analytical estimate was 5-15 hours of 723 (1-3%) would emit different action under mean_last_10; net economic impact estimated ±0.5-1.5pp/month (in noise range). Exact count needs full counterfactual: bypass engine's auto-ffill, build features WITH NaN intact for each hour, call `compute_signal_core` with both `na_policy='ffill'` and `na_policy='mean_last_10'`, run trader's actual model, diff predictions. Existing `tools/counterfactual_backtest.py` is the framework but is built for cache-bug testing, not fill-policy testing — needs adaptation. Outputs: exact flip count, signal-distance histogram, per-action breakdown (BUY/HOLD/SELL transitions). Only run AFTER FAYE in production so we have validated mean_last_10 behavior to compare against. |
| 🚀 P3 | **P4** — C14 vol-conditional triple-barrier retest | Laptop (~2.5h) | open |
| 🚀 P3 | **P5** — C11 VPIN at 5-min cadence | Laptop (~1 day eng) | open |
| 🚀 P3 | **P6** — C15 meta-labeling SOL/BTC | blocked (assets shelved, ~6h) | open |
| 🚀 P3 | **IDEA QUEUE Tier A** — Untested clean (C13-narrow, C54, C55, C58, C59) | research backlog | 5 items open |
| 🚀 P3 | **IDEA QUEUE Tier B** — V3-lit C60-C82 (16 of 23 patcher-ready) | research backlog | 23 items open |
| ✅ **DONE 0601** | **Validate FAYE bug #16 perf patch HARD A/B** — soft validation DONE 2026-05-31; **hard A/B DONE 2026-06-01** | **Desktop** | ✅ **PASS 2026-06-01 11:07** — legacy refine (`FAYE_REFINE_TRIAL_SPLIT=1` + `EARLYSTOP_PATIENCE=0`) on ETH 6h --replay 1440 --no-persist (ran ~23:06 May 31 → 11:07 Jun 1, ~12h — legacy is slow w/o chunking). Result: legacy winner **XGB+LGBM w=150 γ=0.996 20f → 55.24%** vs chunked H winner **RF+LGBM w=150 γ=0.996 15f → +55.30%** = **Δ0.06pp, IDENTICAL window (150) + gamma (0.996)** → PASS on ±2pp criterion. Combo/feature differ (plateau has multiple ~equivalent optima ~55.2-55.3%); chunked refine did NOT degrade quality (marginally higher). **Promotion gate CLEARED + live 6h pick reassured.** Log `logs/ed_v1_20260531_230612.log`; output `models_faye/crypto_faye_production_noprod.csv`. Queue (optimization, not blocking): TPE-shared study via RDB storage so chunks share history; auto-scale WORKERS by `cpu_count()`. |
| 🚀 P3 | **Verify trade count for 6h winner at best_conf** — the +55.3% return at WR=79.3% might be from very few trades at a high conf threshold (small-sample cherry-pick from Step 3's 6-conf scan [65/70/75/80/85/90]) | **Desktop or Laptop** (~5-15 min) | 📅 NEXT — re-run single backtest of the production winner cfg, expose per-conf trade counts. If best_conf was 85% or 90% with only 5-10 trades, +55.3% is statistically weak. Easier alternative: search the H run's terminal scrollback for "OVERALL BEST: ... → ETH 6h" line — it prints conf and trade count there. |
| 🚀 P3 | **Investigate 8h Mode D survivor count** — only 2 candidates survived 3-fold rolling holdout vs 10 for 6h. **NOT A BUG**, identified as data-driven: 8h labels noisier → models less confident → 0-trade holdout filter prunes most candidates | **Desktop or Laptop** (~30 min) | 📅 LOW — could (a) lower holdout conf threshold for harder horizons, or (b) loosen 0-trade filter to "0 trades only if ALL 3 folds 0". Either is a behavior change that needs care. Not blocking. |
| ⚪ P4 | **TODO 0519C** — CPCV HRST diagnostic | trigger-based re-run | available, no plan |
| ⚪ P4 | **Kalshi** — prediction-market data integration | needs API key + impl | backlog |

### Logged 2026-06-11 — model-vs-field analysis (your system vs the serious crypto-quant landscape)

**Context**: revisited the 2026-06-10 crypto-quant research as a structured head-to-head — your system vs the field — via a workflow (codebase audit + 4 parallel web sweeps: predictive ML/DL/RL · relative-value/carry/factors · market-neutral/microstructure · risk-sizing/multi-strategy/alt-data + synthesis). Sourced (Grinsztajn NeurIPS 2022, Harvey/Man 2018, Moreira-Muir J.Finance 2017, Grinold Fundamental Law, BIS WP 1087, Borri-Shakhnov, etc.).

**Headline**: your **directional core is academically correct AND maxed out** — GBDT > DL/RL on engineered tabular crypto (your RF+LGBM verdict = the literature), and your leakage/eval rigor (embargo, gated-sim, PIT lag, GPU determinism) **exceeds typical practice** and matches serious-desk de Prado. Every directional lever now fails your gated sim → marginal return on a better *model* ≈ 0. **The frontier is breadth, not depth: uncorrelated sleeves + a portfolio risk layer.** The one self-imposed structural lock is **binary all-in/out**, which makes vol-targeting (the field's best-evidenced sizing tool) inert.

**3 buckets**:
- 🟢 **Settled** (you match the canon / already tested — don't reinvest): directional GBDT ✓, deep learning ✗ (wins only on raw sub-second LOB), RL-signal/sizing ✗, execution/maker ✓ (you lead the reachable slice), TSMOM ✓ (already your regime detector — but beta-correlated, NOT a new sleeve), on-chain ✓ (your grading matches lit flows>valuation>noise), stat-arb DEAD (C84, matches lit OOS failure), per-trade stops DEAD.
- 🔴 **Structurally blocked at your scale**: cross-sectional momentum (C85 — needs 30-50 coins + shorting; N=6 → effective breadth ≈ 0, lit predicts your Sharpe 0.73 exactly), market-making / AMM LP (inventory infra; LP is short-gamma/LVR), HFT/latency/MEV (colocation/capital), Kelly (binary + needs calibration).
- 🟦 **Frontier (untapped)**: **carry** (C83, in motion ✓), **vol-targeting** (the surprise — see below), **portfolio layer** (the strategic frame), options vol-risk-premium (orthogonal 3rd sleeve, heavy infra; cheap first step = skew/IV-RV as features).

**Two findings worth acting on (now on the dashboard)**:
1. **Vol-targeting ≠ Kelly** → new dashboard row P2 (0611). Kelly sizes by *confidence* (needs calibration — proven dead); vol-targeting sizes by *risk* (inverse realized vol) — a DIFFERENT axis: no calibration, doesn't touch the gate, only modulates size given a BUY. ETH's strong leverage effect → ~+0.4 Sharpe from data on disk. Cost: relax binary on the long leg only. **The highest-ROI single-asset item; don't let Kelly's death pre-kill it.**
2. **The real frontier = the portfolio layer** → new dashboard row FRONTIER (0611). Book Sharpe ≈ IC·√(independent bets); carry's value is its near-zero correlation, which only pays if sized by risk contribution. Risk-budget allocator (ERC + Ledoit-Wolf) + book-level DD governor, once carry is live.

**Where you LEAD**: leakage/PIT rigor (most published crypto Sharpes are clairvoyant/fee-light by comparison — *this is why your negative results are trustworthy*), model choice, screen-vs-gated lesson, 0-fee maker, GPU determinism, on-chain feature grading. **Where you LAG**: breadth (N=1, ETH-beta), no portfolio/allocation layer, binary sizing leaves free Sharpe on the table, thin universe, single spot venue.

**Carry caveat the lit hammers** (apply your own rigor to it): every headline Sharpe (carry 6-15, stat-arb 2.45) is gross/in-sample/decaying — survey carry Sharpe 6.45→4.06→**negative** (2024-25); basis 25%→4.5%. **Size on CURRENT realized funding > fee breakeven, not the historical Sharpe** (ETH funding negative/thin right now — consistent with the paper bot earning ~0 so far). Plus an FTX-style counterparty tail your spot book never carried → needs a hard kill-switch.

**Other prioritized next** (logged for the record, lower than the two dashboard items): −5/−7% disaster-brake gated A/B (low prior, the one price-based DD item your notes left open); wire options skew/IV-RV/term-structure as conditioning *features* (exploratory, uses IV snapshots already captured); add USDT-exchange-flow + ETH-netflow 1-6h features (low prior, Chi-Chu-Hao 2024); multi-lookback TSMOM regime-sign ensembling (low — improves existing sleeve, not a new stream); 30-50 coin PIT universe + perp shorts BEFORE any cross-sectional retry (exploratory, high effort).

### Logged 2026-06-11 — deterministic sanity check (snapshot-replay validator + verdict rewire)

**Context**: user reported "sanity check does not give me the same results → the daily sending from the trader is not correct." Investigated `_run_sanity_and_alert` (`crypto_revolut_ed_v2.py:4813`) + `tools/sanity_check.py` + `validate_core_against_signal_log.py`.

**Diagnosis — the send is NOT wrong; the offline validator is non-reproducible by design.** `_run_sanity_and_alert` faithfully runs `sanity_check.py`, captures its stdout verbatim, greps the summary lines, and Telegrams them — no parsing/send bug. The wobble comes from the engine-vs-trader **PARITY** check, which re-runs the engine over `sig_log.tail(30)` on CURRENT data → two structural non-reproducibilities: (1) `tail(30)` = a different 30-hour window every run (the 08:30 send vs any later re-run cover different hours); (2) the freshest training rows (deriv/on-chain) get backfilled, so even the same hour re-runs to a different proba/boundary call. So consecutive runs legitimately disagree — that's the file-rebuild validator's noise, same root cause as the 83%-not-100% parity gap. RF is bit-deterministic; the diffs are data revision + CPU-vs-GPU boundary, never a real BUY↔SELL bug (shadow already ~100% real-time).

**Fix 1 — snapshot-replay validator** (`tools/validate_snapshot_replay.py`, built+adversarially verified via workflow `wteuovmt5`, 2 agents). Offline analogue of the shadow but **immune to data revision**: replays the trader's OWN frozen point-in-time intermediates from `output/inference_snapshots.jsonl` (written by `crypto_live_trader_ed._log_inference_snapshot`, line 926 — fields: buy_ratio, avg_proba, per-model probas, signal, confidence, all PRE-gate). Per row: reconstruct buy_ratio from probas (vote = proba≥0.5), recompute mean(probas)==avg_proba, then ASSERT recomputed (signal, confidence) from the frozen buy_ratio/avg_proba == logged, using the exact live ternary + confidence math (`crypto_signal_core.compute_signal_core` / `generate_live_signal` ~891-917). **Result: 202/202 = 100.00% on signal AND confidence + both intermediate cross-checks 100%, deterministic, ~instant (no model fit, stdlib-only).** Verifier verdict SOUND (mutation-tested: corrupting a logged signal/conf drops the match + exits 1 → not a tautology). Scope caveat: proves trader-internal bookkeeping consistency, not engine-vs-trader proba parity (that's the shadow's + parity's job). Production-config assumption: `disagree_filter` + `funding_gate` are OFF (absent from `crypto_ed_production.csv`) → buy_ratio∈{0,0.5,1.0}, signal a pure function of it; if either is ever turned ON the snapshot must log the flag.

**Fix 2 — `sanity_check.py` rewired to 3 checks, verdict driven only by the deterministic two:**
- **[1] SHADOW** (deterministic, reads `shadow_signal_diff.csv`) — **verdict now keys off recent-48** (`SHADOW_RECENT_N=48`), not the all-time cumulative mean. The cumulative was stuck **FAIL 91.2%** permanently because the fixed Jun 3-4 shadow episode (Rule 23) drags it below 99% forever, even though `last-20 fails=0` / recent-48 = 100%. All-time still printed as context.
- **[2] SNAPSHOT REPLAY** (NEW, deterministic, reproducible 100%) — drives verdict; FAIL on any logic mismatch.
- **[3] ENGINE-vs-TRADER parity** — **demoted to INFORMATIONAL ONLY.** Still runs + prints the % and lists any real flips (labeled "likely data-revision artifact"), but no longer sets NEEDS ATTENTION. Justification: the parity is the noisy/non-reproducible re-run; the real-time shadow [1] + frozen replay [2] already catch any genuine divergence deterministically, so a benign data-revision flip should not raise a ⚠️. `--quick` now = shadow + snapshot (both instant + deterministic), was shadow-only.
- Verified `--quick`: [1] PASS recent-48 100% / [2] PASS 202/202 / RESULT CLEAN (exit 0). Both files byte-compile clean.

**Fix 3 — `crypto_revolut_ed_v2.py` daily send**: added `'SNAPSHOT'` to the Telegram grep keyword tuple (`_run_sanity_and_alert`, line 4825) so the `[2] SNAPSHOT REPLAY : PASS` header reaches the phone (the detail line already contains "match"). The helper is INFORMATIONAL-only + try/except-wrapped → zero trading-path risk (not the inference path, so Rule 23 N/A).

**Rollout**: the verdict fix takes effect on the **next daily 08:30 run** (sanity_check.py is spawned fresh as a subprocess, reads the new file — no restart needed). The `'SNAPSHOT'` Telegram-header keyword needs a **trader restart** (the running process holds the old grep tuple in memory). All edits on the laptop → Drive-sync to the desktop trader.

**NEXT (optional)**: could fold the snapshot-replay headline directly into the daily Telegram body as the lead line; could truncate/rotate `shadow_signal_diff.csv` so the all-time number eventually reflects only post-fix history.

### Logged 2026-06-10 — crypto-quant landscape research + 3 orthogonal strategies + funding-carry paper bot

**Context**: the single-asset directional ML model is a proven local optimum — every lever (features, model tuning, decision layer, target/label) now fails the gated sim. So the frontier is **diversifying into orthogonal strategy classes**, not more directional-prediction sophistication. Did a thorough internet sweep of serious crypto algo trading (cross-sectional momentum, basis/funding carry, stat-arb, market-making, DL/transformers, RL, on-chain alpha, vol-targeting — academic + institutional sources) → mapped vs the system → tested the 3 most feasible.

**Negative-result A/Bs earlier tonight (all gated, ETH 8h, scoreboard-bound):**
- **Trend-scanning label (López de Prado) → DEAD** (`tools/bt_trend_scanning_8h.py`, gated 60d): −8.64pp (more trades 89 vs 80, lower WR 58 vs 68). Permissive target dilutes selectivity. Fee-aware label confirmed well-matched.
- **ETF-flow proxy → NEUTRAL** (`tools/bt_etf_proxy_8h.py`, yfinance ETF $-volume, gated 60d): +0.38pp (within noise) — the FIRST non-fatal new feature (everything else −4 to −44pp). Weakly supports orthogonal-info thesis. ⚠️ proxy = gross volume ≠ net flow; the REAL net-flow test still pending (user to provide data; Farside 403s scrapers).
- **LGBM-tuning gated A/B → WORSE** (`tools/bt_lgbm_tuned_vs_current.py`, 60d): regularized LGBM −4.72pp gated despite +22pp on the raw screen — gate-starvation (compressed probas clear 65/70 less). Reconfirms [[feedback_screen_vs_gated_engine]].

**3-strategy backtest (`tools/bt_basis_carry.py`, `bt_statarb_eth_btc.py`, `bt_xsec_momentum.py`):**
1. **Basis/funding carry → WINNER.** Delta-neutral (long spot + short perp), harvest funding every 8h. ETH always-on (unleveraged, frictionless): **Sharpe 15.08, APY +6.17%, maxDD 1.97%, funding + 81% of time**; BTC Sharpe 23.8, LINK 24.1, XRP 13.5; SOL/BNB weak (BNB +funding only 18% → reverse carry). always-on > positive-only for majors. Caveat: Sharpe inflated by frictionless/perfect-neutral assumptions (lit ~4.8 net); still far above the directional Sharpe and **uncorrelated**. → pursue.
2. **Stat-arb ETH-BTC → FAILS.** −100%, Sharpe −3 to −6, WR 39-49% across configs. Spread ADF p≈0.05 (borderline) — ETH/BTC trends (dominance regimes) more than it mean-reverts. (Crude per-bar sizing; sign robust.) → out.
3. **Cross-sectional momentum → WEAK/inconclusive.** Best Sharpe **0.73** (14d lookback — matches the academic 2-4wk crypto-momentum finding), APY +34% but maxDD 36-63%; most configs negative. 6-coin universe too thin (lit uses ~50). → maybe later with ~30+ coins.

**Funding-carry paper bot** (`tools/funding_carry_eth.py`): PAPER-only, FREE Binance public data (premiumIndex live rate+mark+nextFundingTime, fundingRate realized history, spot ticker — no keys). Restart-safe (idempotent by `fundingTime`, replays missed settlements once), single-instance lockfile (Drive-synced `config/` → blocks dual-machine corruption), per-cycle CSV + settlements audit CSV + JSON state. Built+reviewed via 2 dynamic workflows (empirical API verification + 4-dimension adversarial review). Fixes applied: **entry phantom-credit** (anchored funding clock to actual realized `fundingTime`, ms-jitter-safe), pagination (limit 1000 + loop for >33d downtime), defensive markPrice parse, replay-from-epoch clamp, dropped local-clock staleness gate (NTP-drift Rule 11), rebalance dust-gate-before-mutate, strptime guard, **single-instance lock**. Backtest leak fixed (positive-only now lag-1, not same-period realized).

**ARCHITECTURE**: carry = **separate parallel sleeve**, NOT embedded in the directional trader — different venue (Revolut X spot-only vs perp venue), opposite intent on ETH spot (trader wants exposure, carry wants it hedged), and blast-radius (don't risk validated prod with experimental code). Eventual unification = thin supervisor/launcher, not merged logic.

**⚠️ CURRENT REALITY**: ETH funding is NEGATIVE right now (bearish regime) — the carry PAYS, doesn't earn, until funding turns positive (~81% of history but regime-dependent). Paper-trading validates correctness + net-of-frictions now; the edge shows in a +funding stretch.

**VERIFIED 2026-06-11 (~18h in, 3 settlements)**: ledger audit clean (funding math exact, delta-neutral held @ net −1.97 bps, fees clean $14 entry-only, survived Modern-Standby sleep gaps) + **data-fidelity check — all 3 captured settlements match Binance `fundingRate` history EXACTLY** (rate to 8dp, mark to cent: 00:00 −0.00003684 / 08:00 +0.00000754 / 16:00 −0.00002432; the pre-inception 06-10 16:00 jittered boundary `...009` was correctly skipped → independently re-confirms the entry phantom-credit + ms-jitter-anchor fixes). Data is correctness-verified but **NOT decision-grade** yet (n=3, negative-funding tail = worst case, $14 entry fee dominates P&L). Edge estimate stays with the backtest; paper run's job = correctness (done) + catch live surprises (needs weeks).

**NEXT**: (a) paper-trade forward ~2-4 weeks on Desktop (`python tools/funding_carry_eth.py --loop`); (b) decide a perp venue (Coinbase Advanced eligibility check — owner has Coinbase; else Binance/Bybit) — gates live; (c) Phase-2 live-execution build (authenticated orders + auto-rebalance + margin/liquidation guard) once venue chosen; (d) real ETF-flow test pending owner's net-flow data. **Doc follow-ups**: ✅ **DONE 2026-06-11** — logged to ARCHIVED_LOG scoreboard as **C83** (carry, 🟢 active-paper) / **C84** (stat-arb ETH-BTC, 🔴 DEAD, cross-refs C34) / **C85** (cross-sectional momentum, ⚪ SHELVED, revive ≥30-50 coins) / **C86** (trend-scanning label, 🔴 DEAD, C20 family); dated pointer added at top of ARCHIVED_LOG. ⏳ Still deferred: "Strategy diversification / carry sleeve" note to CLAUDE.md — **gated until the paper run validates net-of-frictions**.

### Logged 2026-06-09 — GB/LGBM hyperparameter sweeps (model-combo investigation)

**Context**: re-tested the historical "trim to LGBM-containing pairs" decision on the current de-leaked/lagged FAYE pipeline, then swept GB and LGBM hyperparameters in the FIXED bear context (ETH 8h, window 169, γ 0.9998, the 24 production bear features), each scored as the **GB+LGBM ensemble** (primary) **+ solo** (diversity check). Tools: `tools/model_combo_retest_desktop.py`, `tools/gb_hyperparam_sweep.py`, `tools/lgbm_hyperparam_sweep.py`. CSVs in `output/gb_hyperparam_sweep_*` / `lgbm_hyperparam_sweep_*`.

1. **Model-combo retest (25 combos, step=36, ETH 6h w150 17f g0.997 6mo):** **LGBM+GB ranked #1 of all 25** (singles+pairs+triples): +15.8% / APF 2.06 / tight seed spread, beating both prod pairs (RF+LGBM #3, XGB+LGBM #7; RF+XGB #4 beat XGB+LGBM). No triple beat the best pair (3 models dilute — best triple LGBM+RF+GB +11.6%). **LR toxic everywhere** (RF+LR worst combo overall). → **GB+LGBM added to FAYE `GRID_COMBOS` (commit `048aada`).**

2. **GB hyperparam sweep (640 cfg, GB+LGBM ensemble, 6mo):** current-default GB (leaf=1) already **#66/640**; best only **+7pp**. **`min_samples_leaf=1` is BEST for the pair** (more regularization HURTS); `lr=0.03` marginally best; `max_depth`/`min_samples_split`/`subsample` irrelevant. **Diversity overlap = 0/10** — best-SOLO GB (leaf 13-15, solo −15.7%) is among the WORST partners (ens −32.4%); best-PARTNER GB (leaf=1, solo ~−37%) is the best partner (ens −22.9%). Empirically confirms: **tuning GB for solo destroys the pair. Keep GB scrappy.**

3. **LGBM hyperparam sweep (288 cfg, GB+LGBM ensemble, 6mo):** current LGBM (lr0.05/d4/min_child20/reg0/sub1.0) ranks **#276/288** (near worst). Best (lr0.07 / min_child30 / reg_λ5; depth non-binding because min_child caps leaf count) lifts the ensemble **+22pp** (−29.9 → −7.6, APF 0.59 → 1.28). Wants **more** regularization. Diversity 4/10 (partly agrees — LGBM is the dominant predictor, so individual quality helps the pair).

**LESSON**: GB and LGBM tune in OPPOSITE directions — regularize the **backbone** (LGBM) for quality, leave the **diversifier** (GB) scrappy. Their value split is decorrelation, by design.

**CAVEATS**: raw walk-forward eval (NO confidence gate / shields / rally-cooldown → the all-negative numbers are NOT live P&L; it's a downtrend window; only the RELATIVE ranking is valid). SCREEN at **step=36 / n_est=100** — `lr` interacts with `n_estimators`, so re-check at 300 trees. Single 6mo window (could be drawdown-fit).

**NEXT (was, now superseded)**: validate the COMBINED config — scrappy GB + regularized LGBM — at step=1/300/K=5 on a 2nd window vs current RF+LGBM.

**✅ VALIDATION DONE 2026-06-09 — screen REVERSED under the live gate; GB+LGBM removed.** Built a default-safe LGBM env hook (`LGBM_MIN_CHILD`/`LGBM_REG_LAMBDA`/`LGBM_LR`/`LGBM_MAX_DEPTH` → `_lgbm_hyperparam_overrides()` in faye; unset = byte-identical to prod, wired into the 3 deku factories) and `tools/bt_lgbm_tune_8h.py`, which runs a pinned config through the REAL engine (`generate_signals` n300 GPU walk-forward) + the maker-fee regime sim. **2mo (1440h):** PROD live RF+LGBM **+51.2%** > GB+LGBM current **+43.5%** ≫ GB+LGBM tuned mc30/reg5 **+11.0%** (ordering consistent across 1440/720/336/168h). So BOTH the GB-partner swap AND the LGBM-reg tune LOSE. `tools/diag_lgbm_proba_spread.py` showed why: the tune compressed LGBM proba spread (IQR 0.84→0.61, range 0..1→.015..988, confident-bar share 88%→74%) AND drifted bullish-mushy (pred=1 44%→52%, SELL/cash 50→45, meanConf 90→84) → held long through a −21.7% tape instead of sitting in cash. **Regularization erased conviction = the edge** (the ungated screen rewarded exactly that). → **GB+LGBM REMOVED from FAYE `GRID_COMBOS` 2026-06-09** (back to RF+LGBM + XGB+LGBM); lesson logged [[feedback_screen_vs_gated_engine]]. Re-add only after a proper step=1 / multi-window test beats RF+LGBM. Re-run: `python tools/bt_lgbm_tune_8h.py --replay 1440`.

### Recently CLOSED (2026-06-06)

| Item | Status |
|---|---|
| **PySR merge created a functional duplicate** (`models/pysr_ETH_*.json`) | ✅ ROOT CAUSE FIXED + proven inert. `merge_pysr_old_new.py` (run 2026-05-28/29) appended all 5 NEW formulas blindly → the `xa_nasdaq_relstr5d − logret_120h` formula got re-discovered by NEW and stacked next to OLD: **5h pysr_2≡pysr_8 r=1.0000**, 6h r=0.9980, 7h r=0.9993 (8h clean). **Diagnostic verdict: INERT** — that signal ranks #80–117/194, never near the top-15 cut; removing pysr_8 changes the live top-15 by **0 features** on both 5h+6h; neither live model even references pysr_2/8. So **no live impact, no retrain needed.** **FIX**: rewrote `merge_pysr_old_new.py` with value-based dedup (sympy.sympify→lambdify eval on real data, drop NEW with \|corr\|≥0.95 vs any kept; report-only by default; reads the true `_pre_*_old_only` backup as OLD so re-runs don't double-merge). Verified report-only catches all 3 dups (5h→9, 6h→9, 7h→9, 8h→10); live files untouched. **DEFERRED (not urgent, dup is inert)**: (a) actually removing the dup from LIVE needs `--apply` + an HRST retrain — applying renumbers slots and `pysr_9` (live-referenced) would shift meaning (Rule 14), so only do it inside a clean PySR regen with trader flat; (b) **`models/` vs `models_faye/` slot-name collision** — same `pysr_N` names hold *different* formulas (live reads `models/`, FAYE defaults `models_faye/`); a future FAYE PySR-using winner promoted to live without aligning dirs would silently change feature meaning → reconcile both dirs to one canonical deduped set at the next regen. |

### Recently CLOSED (2026-06-05)

| Item | Status |
|---|---|
| **Engine-vs-trader parity (the old P0)** | ✅ DONE — 96.7% (29/30) on GPU, avg conf delta −0.93 (was 90% / −5.71 on CPU). GPU is the right device (Rule 24). 1 DIFF = recent unsettled hour, 0 BUY↔SELL flips. |
| **Shadow-monitor closed-bar bug** | ✅ FIXED in code — `crypto_live_shadow.py:220` mirrors fix #2. Root cause: fix #2 closed-bar applied to live, not shadow → +1h forming-vs-closed (23/23). Live number pending trader restart (P0 above). NOT a model change. |
| **GPU cross-machine determinism probe** | ✅ DONE — Laptop==Desktop bit-identical (`RF=0.51463784 LGBM=0.05573897`). Device decision: use GPU for sanity, drop `--cpu-lgbm` (Rule 24, TODO0604.md). |
| **Training-window data-revision** | ✅ CONFIRMED (was hypothesis) — 0/15 inference-row drift but probas differ ⟹ training rows revised (deriv/on-chain backfill). Revision-flipped hours are PERMANENT reproduction mismatches; 100% live-replay not achievable without PIT snapshots. The ~3% sanity residual is this, not a leak. |
| **Bear-config doc drift** | ✅ CORRECTED — live is bull 6h@65/bear 5h@80 (user-confirmed intended); LIVE STATE fixed in TODO+CLAUDE. Bear-swap backtests were on un-lagged engine → re-confirm on lagged engine (open, low pri). |
| **1-4h DV + leak analysis** | ✅ ANALYZED — WR gradient 1h 92-98%/2h 88%/3h 85%/4h 76%/5-8h 74-83%. Embargo protects all horizons equally (user was right); gradient is the inflation signature. 4h plausible, 2-3h borderline, 1h inflated. Decision gated on embargo sweep (P1 above). |
| **Trader BTC-display bug** | ✅ DISPLAY FIXED — `/status` no longer lists non-enabled coins (`crypto_revolut_ed_v2.py:2544`). Real-vs-phantom BTC check still pending (P0 above). |

### Recently CLOSED (2026-05-31)

| Item | Status |
|---|---|
| **Bug #15** — refine ignored `--replay` (4320h hardcoded `MAX_DIAG_HOURS`) | ✅ FIXED commit `7fad9bf` — threaded `replay_hours` through `_refine_top_configs` + `_refine_top_configs_serial` + 3 call sites. Validated end-to-end on DV ETH 7h --replay 528. Was inherited from v3 chain (parallel_nearlive.py:480). |
| **Bug #16** — refine perf: trial chunking + 6-worker pool + early-stop callback | ✅ SHIPPED commits `9f286f8` (opt-in) + `3b2426b` (defaults ON). Env-controlled: `FAYE_REFINE_TRIAL_SPLIT` (default 3), `FAYE_REFINE_WORKERS` (default 6), `FAYE_REFINE_EARLYSTOP_PATIENCE` (default 15). Soft validation done (chunks find variance, max-aggregation captures it). **Hard A/B still pending → P2 above.** |
| **Bug #17** — refine worker pool halved when n_cfgs<3 | ✅ FIXED commit `7ce0320` — formula `trial_split * max(1, n_cfgs-1)` gave 3 instead of 6 when only 2 candidates passed Mode D's holdout filter (8h case). Changed to `min(trial_split * n_cfgs, 6)`. My initial "Windows kernel handle leak" hypothesis was wrong — actual root cause was a one-line arithmetic bug. |
| **Bug #18** — early-stop callback dormant in chunked mode | ✅ FIXED commit `e16853e` — hardcoded `n_complete < 20` warm-up combined with chunked 25-trial budget meant earliest trigger was trial 35, exceeding chunk size. Callback NEVER fired in production chunked runs (bug #16's claimed "30-50% savings on convergence" never materialized). Now scales: `warm_up = max(5, n_trials//4)`, `patience = min(env, n_trials//3)`. Will activate on next H run. |
| **Sleep-guard** — prevent Windows system sleep during H/HRST runs | ✅ SHIPPED commit `b5cbe23` — `SetThreadExecutionState(ES_CONTINUOUS \| ES_SYSTEM_REQUIRED)` at `__main__` startup; atexit restore. Display sleep still allowed. No-op on non-Windows. Triggered by 2026-05-31 overnight ~18min loss when Desktop slept mid-run. |
| **Banner UX** — refine output shows `[chunk X/Y, seed=Z]` when chunking active | ✅ SHIPPED commit `8875934` — disambiguates 3× repeated "Refining #1" / "Refining #2" output lines that confused real-time tailing. Suffix only when n_chunks>1; legacy output unchanged. |
| **FAYE H ETH 6h,7h,8h --replay 1440** — first production H run on FAYE | ✅ DONE 2026-05-31 — winners: 6h +55.3% Grid, 7h +37.54% Refined, 8h +46.37% Refined. All RF+LGBM, window 150-155, gamma 0.996-0.9997. Strong cross-horizon coherence. Bugs #15/16/17/18 all discovered+fixed during this run; #17 baked into 8h's slower wall time but didn't affect correctness. Production CSV: `models_faye/crypto_faye_production.csv`. |
| **Investigation: Step 1 → Step 3 gap on 6h (+0.7% → +55.3%)** | ✅ NOT A BUG — Step 1 ranks at PRIMARY_CONF=80% only; Step 3 scans 6 conf levels [65/70/75/80/85/90] and picks best return×WR with ≥5 trades min. Different metrics → different winners. Statistical robustness still requires verifying trade count at best_conf (P3 above). |
| **Investigation: 8h Mode D only 2 candidates vs 6h's 10** | ✅ NOT A BUG — 3-fold rolling holdout filter at line 5592 (`if ho_entry[4] == 0: continue`) drops candidates with 0 trades in holdout folds. 8h labels noisier → models less confident → more 0-trade results → fewer survivors. Possible loosening as P3 above. |

### Recently CLOSED (2026-05-30)

| Item | Status |
|---|---|
| **v3 HRST on Desktop** (ETH 5h,6h,7h,8h --replay 1440) | ✅ DONE 2026-05-30 — completed all 4 horizons. 5h winner ETH RF+LGBM w=200 γ=0.999 10f Grid +49.56%. **NOT promoted** — superseded by FAYE H run on 2026-05-31 with bugs #15-18 fixed. |
| **FAYE single-file consolidation** (7 phases, commits `8c122ef` → `4ab34d5`) | ✅ DONE 2026-05-30 — `crypto_trading_system_faye.py` (~9100 lines) collapses the Ed v3 4-layer monkey-patch chain into one file with ZERO monkey-patches. Native K=5 + 8-worker Mode D + 3-worker hybrid refine + NEAR_LIVE defaults + isolated `models_faye/`+`config_faye/` outputs. Smoke test `tools/smoke_test_faye.py` 38/38 ✓. CLI identical to v3. Full writeup in ARCHIVED_LOG.md. **Not yet in production** — first FAYE HRST to validate equivalence with v3 still pending. |
| **Engine-vs-trader parity test** on G_narrow LIVE config (30 recent ETH hours) | ✅ DONE 2026-05-30 — `tools/validate_core_against_signal_log.py --samples 30 --recent-only --cpu-lgbm`. 30/30 evaluated, 0 errors, 22/30 direct MATCH (73.3%), **0/8 real BUY↔SELL flips** — all DIFFs are HOLD-threshold boundary cases (5 live=HOLD because conf<95% bear threshold; 3 core=HOLD because probability<50%). Engine and trader codepaths agree on direction every time both produce one. **The original "bug between live trader and crypto trading" is gone.** Output: `output/core_validation_20260530_015454.csv`. |
| **Post-FAYE archive cleanup** (`ARCHIVED/2026-05-30_post_faye_cleanup/`, commit `1fce7f8`) | ✅ DONE 2026-05-30 — Tier A (8 variant scripts: `_cdar`, `_cvar`, `_cpcv`, `_robust`, `_h_strict_family`, `_noprod`, `_pre_macro_cache_fix`, `_launch_h_strict_family.bat`) + Tier B (14 variant-driving tools: CDaR/CVaR/CPCV/robust/h_strict_family clusters) + Tier C (14 old `models_g_desktop_*/` + `config_g_desktop_*/` snapshot dirs ~440KB) + docs (`CLAUDE_NEW.md`, `TODO_TEST.md`). Root file count 38→29, root dir count 35→21. Per-item grep-checked first per memory rule; NOT archived = anything still imported by v3 chain. Smoke test still PASS after move. Restoration recipe in `ARCHIVED/2026-05-30_post_faye_cleanup/INDEX.md`. |
| **NEAR_LIVE_MODE HRST (v3 fork research run, 2026-05-27 → 2026-05-30)** | ✅ DONE 2026-05-30 — superseded by FAYE consolidation. The v3 run on Desktop completed 5h and is in 6h (Mode V Step 2 refine, ETA full HRST done ~17:30 May 30). When done + trader flat, copy v3's winners into `models/`+`config/` (Critical Rule 19). 5h winner: ETH RF+LGBM w=200 γ=0.999 10f Grid +49.56%. |

### Recently CLOSED (2026-05-27)

| Item | Status |
|---|---|
| **TODO 0527** — `_macro_cache` mtime bug | ✅ DONE — root cause of live-vs-backtest gap. Cache fix shipped; shadow mode 100% match. See ARCHIVED_LOG.md for full writeup. |
| **TODO 0526** — LIVE vs BACKTEST divergence investigation | ✅ CLOSED — superseded by TODO 0527 root-cause discovery. "4 semantic divergences" framing was directionally right but cache was the dominant cause. Step 6 refactor still pending to address residual backtest-vs-live semantic gap. |
| **Audit of sig_1/sig_2 + data drift fix + cache fix** | ✅ DONE — 3 audits across multiple angles each. Found + fixed 4 additional latent HORIZON_SHORT/LONG bugs (sig_short Telegram, asset preflight, gamma fallback, _log_signal edge case) + 1 missing safe-merge call site (`download_fear_greed`). |
| **Counterfactual backtest tool** (`tools/counterfactual_backtest.py`) | ✅ NEW — runs hourly inference with fresh data via oldest-wins archeology + simulates trades. 5-day result on May 22-27: +0.50pp return, 2× per-trade edge, smaller drawdowns vs broken-cache actual (4 vs 7 trades, sample too small to be definitive). |
| **CLAUDE.md stale-reference cleanup + ETH legacy-key strip** | ✅ DONE — CLAUDE.md now reflects G_narrow live state; regime_config_ed.json ETH block stripped of inert top-level `horizon: 8` + `min_confidence: 85` legacy keys (no behavior change). |
| **TODO 0525** — G_narrow_d HRST with extended grid (May 25-26) | ✅ DONE — REF +83.85% lost to LIVE +91.01% by 7pp. Triggered TODO 0526 architecture analysis which led to TODO 0527 discovery. Underlying hypothesis (extended grid unlocks high-window basin) rejected. |
| **TODO 0524** — Top-5 HRST clean rerun (May 24-25) | ✅ DONE — REF +80.56% lost to LIVE +91.01%. Parallel fork validated (~8× refine speedup retained). |
| **TODO 0522** — Parallel refine speedup fork | ✅ DONE — Stage 1 passed; Stage 2 verdict invalidated by grid bug, bug-fixed, superseded by TODO 0524. |
| **TODO 0519** — G_narrow_d relaunch on Desktop | ✅ DONE — REF +89.14%, no STRICT winner but per-horizon V winners drove G_narrow promote 2026-05-21. |

### Machine allocation summary

| Machine | Active load | What runs here next |
|---|---|---|
| **Desktop** | Trader (live) + v3 HRST (started 2026-05-29 19:10, currently 6h Mode V Step 2 refine, ETA done ~17:30 May 30) | When v3 done: promote winners + restart trader; then first FAYE HRST as the next validation run |
| **Laptop** | Currently idle (this is where FAYE was built) | Wider counterfactual backtest; embargo A/B test; idea-queue research. Optionally first FAYE HRST here if Desktop is busy with trader-only after promo. |

**Honest top-of-mind (2026-05-30 ~02:00)**: FAYE single-file consolidation shipped today — `crypto_trading_system_faye.py` collapses the v3 4-layer monkey-patch chain into native code, smoke-tested (38/38 ✓). Engine-vs-trader parity test on 30 recent ETH hours showed **0 real BUY↔SELL flips** — the major signal-divergence bug is gone, codepath-wise. The v3 HRST on Desktop is still running (5h done, 6h in Mode V Step 2 refine, ETA ~17:30 May 30). **Next 12 hours: v3 HRST finishes → promote winners to `models/`+`config/` (Critical Rule 19: trader flat) → restart trader.** **Next 1-2 days**: first FAYE HRST on same May data to validate it produces equivalent winners to v3. If equivalent ±2pp, FAYE replaces v3 as canonical engine path. **No FAYE-to-prod promotion until that validates.** The realistic backtest WR target after FAYE is in production is still ~65-75% (per Step 6 thesis) — anything higher is the old binary-step backtest math, not live-equivalent.

---

**Layout (priority-ordered, top → bottom)**:
- 📌 LIVE STATE (always visible — current production config + rollback)
- 🔥 **P1** — Act this week (in-flight + imminent)
- 📋 **P2** — Scheduled next month
- 🚀 **P3** — Research backlog (when capacity allows)
- ⚪ **P4** — Low priority / Diagnostics / Backlog

---

## 📌 LIVE STATE — FAYE models (promoted 2026-05-31 14:22 CEST)

> ✅ **CURRENT LIVE — 2026-06-16 00:25 CEST: detector `tsmom_672h` → `sma48>sma100`** (trader rebooted, verified end-to-end). ETH only. **Bull 8h@90% shield ON** (rally-cd rr30h≥4.0%/rr36h≥6.5% cd24h) / **bear 5h@65% shield OFF** (rally-cd rr8h≥2.5%/rr30h≥7.0% cd14h), min_sell_pnl **0**, max_hold 10h, max_position $14,300, maker ON. Models unchanged (`models/crypto_ed_production.csv`: ETH 8h XGB+LGBM w163 γ0.9985 + 5h RF+LGBM w169 γ0.9998). **Rollback:** `cp config/regime_config_ed_pre_sma48_20260616_002531.json config/regime_config_ed.json` + restart `start_ed_v2.bat`.
> **Provenance:** detector RST 2026-06-15 (`logs/hrst/faye_run_20260615_214325.log`, 45-detector menu, 5h/8h pair, 1440h). `sma48>sma100` Mode-S #2 (+55.68%/80% WR) vs incumbent `tsmom_672h` #118 (+49.70%); beat it across 1wk/1mo/2mo (+2.3/+4.4/+10.4pp). **Engine (commit `744112a`):** ENABLED_DETECTORS 2→6 (`tsmom_672h, sma168>sma480, sma48>sma100, tsmom_168h, price>sma72, vol_calm`) in faye+ed; trader `_evaluate_named_detector` wired for sma48>sma100 + tsmom_168h. **`min_sell_pnl` Mode-T default 0.5→0** (commit `00df184`). **⚠️ Validated on ONE window** (recent 2mo incl. the 06-14/15 rally — favors faster detectors); the **`2880` cross-window pass is the open tiebreaker** (tsmom won the rally-free 06-13 HRST window). Everything below is now HISTORICAL.

**Engine**: live trader (`crypto_revolut_ed_v2.py` → `crypto_live_trader_ed.py`) is UNCHANGED — inference still runs through `compute_signal_core()` in `crypto_trading_system_ed.py` (macro_cache mtime fix from 2026-05-27 intact). The live MODELS + regime config were *generated* by `crypto_trading_system_faye.py` (FAYE H + RST run 2026-05-31, with bugs #15–18 fixed) and spliced into `models/crypto_ed_production.csv` + `config/regime_config_ed.json`.

> ⚠️ **DRIFT FLAG — updated 2026-06-09.** `config/regime_config_ed.json` no longer matches the 6h/5h block below. The LIVE file now reads: detector **`tsmom_672h`**, **bull 8h@65% (shield OFF, rally-cd OFF)** / **bear 8h@70% (shield ON, rally-cd OFF)**, min_sell_pnl **0.5%**, max_hold 10h, max_position $14,300, maker ON, ETH only. Both regimes select **horizon 8 = RF+LGBM w=169 γ=0.9998 24f** (`models/crypto_ed_production.csv` 8h row). The 6h/5h detail below is HISTORICAL. **Provenance + rollback snapshot for the 8h/8h promotion are NOT yet recorded** — confirm which run/date promoted it and stash its backup before trusting the rollback ladder.

**Models + regime config — PRIOR 6h/5h STATE (superseded by the 8h/8h DRIFT FLAG above; kept for history):**
- Detector: **`tsmom_672h`** (named) — CHANGED from `sma24>sma100`
- Bull = **6h@65%** RF+LGBM w=150 γ=0.996 15f (FAYE H winner +55.30% / WR 79.3%) — was 5h pre-FAYE
- Bear = **5h@80%** RF+LGBM w=150 γ=0.999 15f (FAYE H 5h winner +41.93% / WR 79.7%, Grid) — **CHANGED 2026-06-02 from 8h@65%** (RF+LGBM w=155 γ=0.9997 11f). 5h-standalone backtested better on the recent month (Jun-3 `am1_prod_recentmonth.csv`: 5h +39.31% vs 8h +31.96%); bear conf raised 65%→80%. Backtest drivers: commits `f580d36` (NEW 5h@80 vs OLD 8h@65), `af1c0d5` (recent-month a-m1). Confirmed intended by user 2026-06-04.
- Shields OFF (both regimes)
- **Rally cooldown ON** (both regimes). Bull: rr8h≥2.0% OR rr14h≥6.0% cd=6h. Bear: rr10h≥5.5% OR rr12h≥2.0% cd=8h.
- min_sell_pnl=0%, max_hold=10h, max_position_usd=$14,300, maker orders ON

**Asset universe**: ETH live; BTC/SOL/LINK/BNB `enabled: false`; XRP removed from trader data pipeline 2026-05-23 (silent-crash mitigation).

**Promotion source**: FAYE H + RST ETH 6h/7h/8h --replay 1440 on 2026-05-31. Mode R picked `tsmom_672h × 6h/8h` (REF +72.75%/+76.05%, B&H +61.88%); Mode S optimized confs to 65%/65%; Mode T converged shields OFF + rally cooldowns + min_sell_pnl=0 + max_hold=10h. A wider 5h-inclusive RST (research, `--no-persist`, not promoted) re-confirmed the 6h/8h regime pair *at that time*; the bear was subsequently moved to **5h@80%** on 2026-06-02 (see Bear note above). **Parity verified on the new config 2026-06-04 22:06**: `validate_core_against_signal_log.py` → 25/30 = 83.3% current-config match, 0 errors, **0 real BUY↔SELL flips** (5 DIFFs all HOLD-threshold boundary cases). The 5h FAYE winner (+41.93%) is now the LIVE bear model and sits in `models/crypto_ed_production.csv` as the 5h row.

**Rollback ladder (one-command each, hot-reloads within 5 min):**

```powershell
# One level back — to G_narrow (sma24>sma100 / bull 5h / bear 8h; live 2026-05-21 → 2026-05-31, the config FAYE replaced)
copy archive\g_narrow_live_20260531_142202\regime_config_ed.json    config\regime_config_ed.json
copy archive\g_narrow_live_20260531_142202\crypto_ed_production.csv  models\crypto_ed_production.csv

# Two levels back — to pre-G_narrow / H75-fresh promote state (live 2026-05-20 09:04 → 2026-05-21 21:56)
copy config\regime_config_ed_pre_G_narrow_20260521.json    config\regime_config_ed.json
copy models\crypto_ed_production_pre_G_narrow_20260521.csv models\crypto_ed_production.csv

# Three levels back — to H75-snapshot (live 2026-05-18 22:02 → 2026-05-20 09:04)
copy config\regime_config_ed_pre_H75fresh_20260520.json    config\regime_config_ed.json
copy models\crypto_ed_production_pre_H75fresh_20260520.csv models\crypto_ed_production.csv

# Four levels back — to pre-H75 baseline (live before 2026-05-18)
copy config\regime_config_ed_pre_H75_20260518.json    config\regime_config_ed.json
copy models\crypto_ed_production_pre_H75_20260518.csv models\crypto_ed_production.csv
```

**Promotion source**: Desktop G_narrow_d HRST run 2026-05-20 11:05 → 2026-05-21 10:28 (wall ~23h 22m, log `logs/ed_v1_20260520_110556.log`). Mode T REF +89.14% (converged iter 2, no STRICT rally-cooldown winner).

**Promotion timeline**:
1. 2026-05-18 22:02 — H75 promoted (snapshot: `pre_H75_20260518`)
2. 2026-05-20 09:04 — H75-fresh promoted (snapshot: `pre_H75fresh_20260520`)
3. **2026-05-21 21:56 — G_narrow promoted (current)** (snapshot: `pre_G_narrow_20260521`)
4. 2026-05-22 19:51 — G_fresh promoted (content-identical ETH rows; snapshot: `pre_G_fresh_20260522`)
5. 2026-05-23 22:21 — manual: rally_cooldown enabled → disabled
6. **2026-05-27 11:22 — engine: macro_cache mtime fix patched** (TODO 0527 root cause; snapshot: `crypto_trading_system_ed_pre_macro_cache_fix_20260527_112231.py`)
7. **2026-05-27 13:18 — manual: rally_cooldown disabled → enabled** (reverted to Mode T optimal; backup: `regime_config_ed_pre_rc_reenable_20260527_131848.json`)

Full promotion events in ARCHIVED_LOG.md.

---

# 🔥 P0 — Step 6 fork ready to test on Desktop (action items)

**Search anchor**: `STEP6-DESKTOP-RUN`

**Status**: Step 6a + 6b + 6c shipped in research fork `crypto_trading_system_ed_step6.py` (commit `50a63ab`, 2026-05-27). Smoke regression (1 grid point) confirmed BIT-IDENTICAL vs production engine. Wider regression + LIVE_EQUIVALENT_MODE diagnostic pending — should run on Desktop (faster).

**What Step 6 is**: refactor of `_deku_eval_with_pruning` to delegate to `crypto_signal_core.compute_signal_core()`. With default parameters, output is bit-identical to current engine (regression-safe). New params (`embargo`, `na_policy`, `signal_mode`, `return_probas`, `eval_step`) let callers opt into live-trader semantics. `LIVE_EQUIVALENT_MODE=1` env var overrides all 5 to live conventions.

**Why it matters**: lets HRST runs produce a REALISTIC backtest projection that should predict live performance. Current backtest math (`_deku_eval_with_pruning` original) uses different semantics from live trader (binary signals, step=36, embargo=horizon, NaN skip) → it overstates live WR. Step 6c diagnostic mode tells us by how much.

## Desktop run instructions

### Step 1 — Confirm Drive sync (~1 min)

Fresh PowerShell on Desktop, venv activated:

```powershell
cd G:\engine
Test-Path crypto_trading_system_ed_step6.py
Test-Path tools\test_step6_regression.py
```

Both should be True.

### Step 2 — Wider regression test (~10 min, builds confidence beyond the 1-point smoke)

```powershell
# Clear any leftover env vars
$env:V2_DATA_SNAPSHOT = ""
$env:LIVE_EQUIVALENT_MODE = ""

# Capture baseline with the ORIGINAL engine (12 grid points)
python tools\test_step6_regression.py --engine ed --save baseline_wide

# Compare fork against that baseline
python tools\test_step6_regression.py --engine step6 --compare baseline_wide
```

Expected output: `[OK] BIT-IDENTICAL` on all 12 grid points.

If any point diverges, paste output to assistant for debug. If all 12 match, Phase 6a/6b verified production-grade.

### Step 3 — LIVE_EQUIVALENT_MODE diagnostic (~10 min)

The payoff — see what the model's realistic backtest WR looks like with live-trader semantics:

```powershell
# Enable LIVE_EQUIVALENT_MODE — overrides all 4 backtest semantics to live
$env:LIVE_EQUIVALENT_MODE = "1"

# Run the same 12-point grid; the fork's _deku_eval_with_pruning will now
# use embargo=1, na_policy='ffill', signal_mode='ternary', eval_step=1
python tools\test_step6_regression.py --engine step6 --save liveeq_diag

# Disable so subsequent runs are normal
$env:LIVE_EQUIVALENT_MODE = ""
```

Paste per-grid-point cum_return / accuracy / trades back to assistant. Comparing against `baseline_wide`:
- LIVE_EQUIVALENT cum_return **dramatically lower** than baseline → confirms backtest math was overoptimistic, Step 6 thesis validated
- LIVE_EQUIVALENT cum_return **similar** → either (a) backtest math wasn't the dominant issue or (b) small per-feature drift was already dominating

### Step 4 — Full HRST in LIVE_EQUIVALENT_MODE (optional, ~7h, the real test)

If Steps 2+3 look good, run a full HRST with the fork and live-equivalent mode to get the REAL Mode T REF projection:

```powershell
$env:LIVE_EQUIVALENT_MODE = "1"
$env:V2_DATA_SNAPSHOT = "data\_reliability_hrst_snapshot_desktop_20260515_154801"

python crypto_trading_system_ed_step6.py HRST ETH 5,8h --replay 1440 --no-persist --no-data-update
```

This gives a Mode T REF number that should approximate what live performance actually delivers. Probably much lower than the current overoptimistic ~89%. That number becomes the **realistic target** — any future HRST should be evaluated against it instead of the current backtest WR.

### Decision after Steps 2+3 (+ optionally 4)

| Step 2 result | Step 3 result | What it means | Next action |
|---|---|---|---|
| All 12 BIT-IDENTICAL | LIVE_EQUIVALENT much lower than baseline | Refactor correct + backtest was overoptimistic | Promote fork to production engine. Future HRST runs use LIVE_EQUIVALENT_MODE. |
| All 12 BIT-IDENTICAL | LIVE_EQUIVALENT ≈ baseline | Refactor correct + backtest math wasn't the dominant gap source | Keep fork as research tool only. Look elsewhere for model improvements. |
| Any DIFF in Step 2 | — | Refactor broke bit-identical | Paste DIFF output to assistant; debug before doing Step 3 or 4. |

---

# 🔥 P0 — Live monitoring (passive, Desktop)

## ⚡ G_narrow LIVE OOS monitoring — RESTARTED 2026-05-27 post-fixes

**Search anchor**: `G_NARROW-OOS-MONITOR`

The original OOS window (2026-05-21 21:56 → ~2026-06-04) is effectively a different system now after today's fixes:
- `_macro_cache` mtime fix at 11:22 (TODO 0527 root cause)
- `_log_signal` schema rename to sig_1/sig_2 + 3 latent HORIZON_SHORT bugs fixed
- `download_fear_greed` safe-merge fix
- `rally_cooldown.enabled` reverted to True (Mode T optimal) at 13:18

**Pre-fix closed trades under broken-cache G_narrow** (now historical context, not directly informative of forward performance):

| # | Open | Close | Entry | Exit | PnL |
|---|---|---|---|---|---|
| 1 | 2026-05-22 18:22Z (manual BUY) | 2026-05-23 22:00 CEST (auto SELL) | $2120.96 | $2075.26 | −2.15% / −$298.81 |
| 2 | 2026-05-24 18:02Z (auto BUY) | 2026-05-24 22:00 CEST (auto SELL) | $2100.60 | $2098.32 | −0.11% / −$14.73 |
| 3 | 2026-05-27 (today, post-cache-fix) | 2026-05-27 same day | — | — | LOSS (user noted "felt healthier" qualitatively despite outcome) |

**Forward monitoring window** (starts 2026-05-27 13:18, the moment all fixes are in place):
- 3-5 days: first qualitative read on trader behavior
- 10 trades (~7-14 days): first WR/return signal vs sample-size noise
- 30+ trades (~3-4 weeks): statistically meaningful estimate

**Rollback triggers under the now-fixed setup** (any one fires → discuss with user):
- Cumulative realized alpha < +5% after first 10 trades
- MaxDD exceeds −10% in any 14-day window
- First 10 trades WR < 50%
- Trade count vastly different from counterfactual prediction (counterfactual showed ≈4 trades / 5 days = 0.8/day; if live is dramatically higher or lower, investigate)

**Decision tree after 2-4 weeks**:
| Outcome | Action |
|---|---|
| Holds OOS (≥0 triggers, WR trending above broken-cache 50.9%) | Continue. Document cache-fix impact as confirmed. |
| Underperforms (>1 trigger OR alpha < +5% after 10 trades) | Investigate: shadow log first (any DIFFs?), then execution layer (slippage, partial fills) |
| Borderline | Watch another 1-2 weeks; don't act on small samples |

---

## 🔥 Shadow mode continuous match-rate check (Desktop)

**Tripwire**: any drop below ~99% match in `config/shadow_signal_diff.csv` = NEW bug to investigate. Currently at 10/10 (100%) since 2026-05-27 fixes.

**Periodic query** (every 1-2 days):
```powershell
Import-Csv config\shadow_signal_diff.csv | Group-Object match | Select Count,Name
```

When the broader gate from Step 6 (engine↔trader shared core) is in place, shadow mode can be retired. Until then it's the primary live-correctness gate.

---

# 🔥 P1 — Re-evaluate disqualified items under LIVE_EQUIVALENT_MODE (CONDITIONAL on Step 6 verdict)

**Search anchor**: `STEP6-REEVAL`

**Status**: 📅 PLANNED, conditional on Step 6 LIVE_EQUIVALENT_MODE results showing a meaningful gap vs current backtest baseline. If gap is >10pp, prior "DEAD" / "SHELVED" verdicts in the ideas scoreboard may have been methodology artifacts and worth re-testing.

**Why this exists**: every "DEAD"/"SHELVED" verdict in CLAUDE.md / ARCHIVED_LOG.md was reached using the SAME `_deku_eval_with_pruning` math that Step 6 is exposing as overoptimistic. The 4 semantic divergences (embargo, NaN policy, step size, signal mode) hurt different candidates asymmetrically — meaning relative rankings under broken backtest may not predict relative rankings under live.

**Decision gate**: re-test only if Step 6's LIVE_EQUIVALENT_MODE shows cum_return materially below baseline. If LIVE_EQUIVALENT ≈ baseline, prior verdicts hold and this whole block is moot.

## Priority list (ordered by recovery potential under LIVE_EQUIVALENT_MODE)

### Priority 1 — NaN-sensitive (sparse-history features quarantined by 'skip' policy)

These were filtered out because backtest's `na_policy='skip'` dropped any row with NaN in their column. Under `na_policy='ffill'` (live), they'd contribute.

| Item | Reason originally quarantined | Re-test action |
|---|---|---|
| `deriv_oi_*` family | Sparse OI data (30-day history only) → most training rows had NaN → skip removed them | TODO 0519B-G1 already queued; now reframed under Step 6 lens |
| Orderbook (`ob_imbalance`, `spread_bps`) | Hourly snapshots, gaps | Part of TODO 0519B-G2 |
| IV (`avg_iv`, `iv_skew`) | Sparse data | Part of TODO 0519B-G2 |
| Stablecoin mcap (3 features) | Currently Grade 1 (all importance <1%) | Re-test with ffill |

### Priority 2 — Step-size sensitive (hourly-cadence ideas)

Backtest evaluates every 36h; live every 1h. Anything responsive to short-term changes was undersampled.

| Item | Re-test action |
|---|---|
| C11 VPIN at 5-min cadence | Re-test with `eval_step=1` |

### Priority 3 — Embargo-sensitive (short-window logrets)

Backtest uses `embargo=horizon` (5-8h gap before test); live uses 0. Recent-momentum features lose their freshest data.

| Item | Re-test action |
|---|---|
| `logret_2h`, `logret_5h`, `logret_8h` (currently Grade 1) | Re-test with `embargo=0` |
| Any `chg1d` feature | Re-test with `embargo=0` |

### Priority 4 — Signal-mode sensitive (HOLD-aware strategies)

Backtest forces BUY-or-not on every step; live ternary allows HOLD. High-confidence-only strategies were penalized.

| Item | Re-test action |
|---|---|
| C14 triple-barrier overlay | Re-test with `signal_mode='ternary'` |
| C15 meta-labeling | Re-test with `signal_mode='ternary'` |
| Asymmetric loss (`scale_pos_weight`) | Re-test with `signal_mode='ternary'` |
| C56/C57 regime detectors | Re-test with `signal_mode='ternary'` |

### Priority 5 — HYPERPARAMETER RE-GRID (the biggest recovery surface)

**This is bigger than the idea/feature re-tests.** The current production winners — `RF+LGBM w=281 γ=0.9981 12f` (5h) and `RF+LGBM w=293 γ=0.9990 16f` (8h) — were selected by the same broken `_deku_eval_with_pruning` math. EVERY hyperparameter chosen (combo, window, gamma, feature count) was scored under backtest semantics that don't predict live performance.

If LIVE_EQUIVALENT_MODE shows a meaningful gap, the "best config" rankings can shift entirely. Different combos may win. Different windows. Different gammas. Different feature counts.

| Hyperparameter | Current production grid | Expanded re-grid (live-equivalent) | Recovery hypothesis |
|---|---|---|---|
| **Model combos** | 3 viable: RF+LGBM, XGB+LGBM, RF+XGB | Add back: **LR**, **GB**, **single-model LGBM**, **RF+GB**, **RF+LR**, **GB+LR** | Solo models may work in ternary HOLD mode (less overfitting risk); LR may benefit from ffill giving more usable rows |
| **Windows** | [72, 100, 150, 200, 250, 300] | [30, 50, 72, 100, 150, 200, 250, 300, 350, 400, 500, 720] | Shorter windows benefit from `eval_step=1` (more recent regime weighting); longer windows benefit from less embargo cutting away signal |
| **Gammas** | [0.995, 0.997, 0.999] | [0.99, 0.995, 0.997, 0.998, 0.999, 0.9995, 1.0] | Different time-decay weights under live semantics; `gamma=1.0` (no decay) might suddenly be viable when step=1 makes recent data more informative |
| **Feature counts** | [5, 10, 15, 20, 25, 30] + Optuna range [4, 40-80] | [3, 5, 8, 10, 15, 20, 25, 30, 40, 60, 100] | With ffill, sparse features become viable → more features can pass selection; with ternary HOLD, fewer features may suffice |
| **MIN_COMBO_SIZE** | 2 (solo removed) | 1 (solo allowed) | Solo LGBM or solo XGB may dominate when the ternary HOLD semantics provide their own "abstain" risk filter |

**Compute estimate**: current grid = 324 configs × 1 horizon ≈ 30 min on Desktop. Expanded grid = ~5,000 configs × 2 horizons ≈ 5-7h on Desktop. Plus Mode V refine for top candidates: +3-4h.

**Plan** (conditional on Step 6 showing meaningful gap):
1. Expand `GRID_COMBOS`, `GRID_WINDOWS`, `GRID_GAMMAS`, `GRID_FEATURES`, `MIN_COMBO_SIZE` in fork
2. Run Mode D ETH 5h + 8h with `LIVE_EQUIVALENT_MODE=1` on expanded grid
3. Compare top-10 winners under live-equivalent vs current production
4. If a meaningfully better config emerges → promote (after appropriate validation)
5. If current production is still in the top-10 under live-equivalent → it was selected correctly despite the broken backtest math; no change

This is **the highest-value single experiment** after Step 6 validates.

### Priority 6 — Dead model combos (subset of P5 — broken out for clarity)

| Item | Re-test action |
|---|---|
| GB+LR, RF+GB, RF+LR combos (dropped from `GRID_COMBOS`) | Covered by P5 expanded re-grid |

### Priority 7 — Disabled feature families

| Item | Re-test action |
|---|---|
| GDELT 21 features (disabled 2026-04-19, 0/33 selection) | Re-enable temporarily, run Mode V live-equivalent |

### Priority 8 — Full IDEA QUEUE Tier A/B sweep

Only if Step 6 shows >15pp gap AND P5 expanded re-grid produces meaningfully different winners — then ALL prior verdicts are suspect and a sweep is justified.

## How re-testing will work

Once Step 6 verdict is in, if gap is meaningful:

1. **Build `tools/re_evaluate_dead_ideas.py`** — takes a config + identifier list, runs each under both `backtest` mode (legacy) and `LIVE_EQUIVALENT_MODE`, outputs side-by-side delta table.
2. **Stage candidates from Priority 1 first** — already-queued G1/G2 work becomes the proof-of-concept for the methodology.
3. **If P1 produces a flipped verdict** (item revives), expand to P2-P4.
4. **Update scoreboard** in ARCHIVED_LOG.md with new verdicts. Old verdicts kept as audit trail, new entries reference them.

**Budget cap**: stop after spending 3 days of compute on re-testing. If nothing flips, prior verdicts hold; if many flip, the engine has a fundamentally different ranking under live semantics and a full HRST re-run + retraining is warranted.

---

# 🔥 P1 — Act this week (Laptop preferred unless noted)

## 📝 Wider counterfactual backtest (Laptop, ~30 min, RUNNING NOW)

**Search anchor**: `COUNTERFACTUAL-WIDE`

Running as of 2026-05-27 afternoon via `python tools/afternoon_run.py`. Output expected at `output/afternoon_summary_<ts>.md`. Will produce bootstrap CIs on:
- WR / compound return / avg per trade (both conditions)
- Return delta (counterfactual − actual)
- P(counterfactual > actual)
- Weekly breakdown

Decision gate on completion:
- P(counterfactual > actual) > 80% → cache-fix value confirmed; ride forward live data with confidence
- 50% < P < 80% → direction looks right; wait for forward live
- P < 50% → unexpected; re-examine execution layer or another bug

---

## ✅ Embargo A/B test — DONE 2026-05-31

**Search anchor**: `TODO-0526-EMBARGO-AB`

**Command**: `python tools/embargo_ab_test.py --mode=both` (Laptop)

**Result** (ETH Mode T, replay=1440h, identical models 6h RF+LGBM w=150 15f γ=0.996 / 8h RF+LGBM w=155 11f γ=0.9997 — only embargo varied):

| Training cutoff | H1 | H2 | REF |
|---|---|---|---|
| `i − horizon` (embargo, honest backtest) | +12.00% | +50.31% | **+69.09%** |
| `i − 0` (no embargo, live-equivalent cutoff) | +49.46% | +82.06% | **+174.20%** |

**+105pp gap from embargo alone.** Interpretation: the +174.20% is **leakage-inflated** — setting `embargo=0` in a *backtest* reintroduces label-overlap leakage (training rows in the last `horizon` hours carry labels that peek into the test window). It is NOT a live target. The honest, live-realistic number is the embargo'd **+69.09%**. Conclusions:
1. **Keep embargo in backtest/selection** (Mode D/V/H/T, HRST) — it is essential; the +105pp is the size of the leakage it removes.
2. **Live trader correctly uses NO embargo** (`train_end = i`, all data) — Critical Rule 9 / [[feedback_no_live_embargo]]. Nothing changes there.
3. **Embargo is NOT the source of the live-vs-backtest gap** (that was the macro_cache bug + signal-path semantics, TODO 0526/0527) → **Step 6 refactor still warranted** for an honest live projection (you cannot get one by flipping embargo off — that leaks).

**Output**: `output/embargo_ab_20260531_171552/` (report.md, baseline/no_embargo subprocess logs + signal CSVs).

**Two harness bugs found + fixed** (`tools/embargo_ab_test.py`):
- Windows `os.execv` re-exec in FAYE spawns a *detached* child and the launcher process exits in 0.2s → `subprocess.run` captured nothing (parsed 0 signals). Fixed by pre-setting `_FAYE_WARNINGS_BAKED=1` in the subprocess env so FAYE skips the re-exec and runs in-process. **General gotcha: any harness that subprocess-launches `crypto_trading_system_faye.py` and waits on it needs this env var on Windows.**
- Error-path verdict defaulted `match_rate` None→100, printing "EMBARGO HAS MINIMAL EFFECT" (backwards) → now prints **INCONCLUSIVE**.
- Match-rate itself remains uncomputable by design (harness parses only every-50th signal-cache line; embargo shifts the walk-forward grid +5h so samples never align). **Use Mode T REF as the comparison metric**, not the harness match rate. (Phase times 4.9h vs 25min were laptop sleep during standalone Mode T, not compute.)

---

## 🔥 TODO 0519B-G1 — `deriv_oi_*` re-enable A/B test (Desktop off-hours, ~6h)

**Search anchor**: `TODO 0519B-G1`

**Status**: pending; deferred multiple times. Newly relevant after the macro_cache fix because `deriv_oi_*` features (when re-enabled) will now actually vary across trader cycles instead of staying frozen at startup values. The decision criterion may swing differently than the pre-fix expectation.

**Procedure**: A/B compare Mode V refine output with `deriv_oi_*` features in vs out of the disabled-feature quarantine list. Detailed steps in `archive/disabled_features_pre_g1_<DATE>.json` backup procedure.

**Don't run while a P1 Laptop job is going** — wait for capacity.

---

# 📝 P2 — Pending design work (Laptop, ~1.5 calendar days + 12h compute)

## 📖 Step 6 engine refactor

**Search anchor**: `STEP-6-REFACTOR`

**Design doc**: [docs/STEP_6_ENGINE_REFACTOR.md](docs/STEP_6_ENGINE_REFACTOR.md)

**Goal**: make Mode V / Mode T backtest call the same `compute_signal_core()` that the live trader uses (after TODO 0527 fixes). After this, HRST results predict live performance. Expected outcome: backtest WR projections drop from ~85% (overoptimistic) to a realistic ~65-75%, but with the property that live performance should approach that realistic projection.

**Required before next promotion** — promoting on the current overoptimistic backtest will keep producing live-vs-backtest gaps.

**4 phases**:
- 6a: regression-safe refactor (bit-identical Mode D output)
- 6b: expose embargo/NaN-policy/signal-mode as explicit parameters
- 6c: live-equivalent diagnostic mode in the engine
- 6d: cross-validate against shadow data

---

## ⏸ Re-run HRST on refactored engine (Desktop, ~7h, BLOCKED on Step 6)

Once Step 6 ships, re-run the canonical HRST so the recommended config reflects realistic live expectations. Validation: Mode T REF should match counterfactual backtest within ±5pp.

---

# 📋 P2 — Scheduled (next month)

## 📅 TODO 0519B-G2 — orderbook + IV re-enable A/B test (2026-06-18, ~30 days)

**Search anchor**: `TODO 0519B-G2`

**Features**: `ob_imbalance`, `spread_bps`, `avg_iv`, `iv_skew`. ~60d coverage from 2026-04-19 live trader snapshot writes.

**Procedure**: identical to G1 with these substitutions:
- Backup filename: `disabled_features_pre_ob_iv_20260618.json`
- Log filenames: `g2_ob_iv_OFF_*.log` / `g2_ob_iv_ON_*.log`
- 4 features to remove from `always_disabled_exact` (not 3)

Full G2 details: [ARCHIVED_LOG.md:3553](ARCHIVED_LOG.md).

**Don't auto-run on the date** — decide based on G1 outcome. If G1 ships → prior on G2 up. If G1 fails (matches "feature-add family exhausted" pattern) → consider skipping G2.

---

# 🚀 P3 — Research backlog (when capacity allows)

## P-Queue items (from 2026-05-10 priority list — kept open, lower than P1/P2)

### P4 — C14 vol-conditional triple-barrier retest (~2.5h overlay sim)

**What**: Only apply triple-barrier exit when realized vol > p70. Original C14 SHELVED 2026-04-26 (+10.48pp 60d but +1.24pp 90d — gain didn't survive). Hypothesis: barriers help only in high-vol regimes; current 60d-test averaged across vol regimes diluted the win.
**How**: Standalone overlay sim script reusing `data/eth_per_horizon_signals_90d.pkl` + realized-vol percentile filter on entries.
**Decision**: if vol-conditional gain ≥ +5pp on 90d → engine integration. Within ±5pp → SHELVE final. Worse → kill the angle.

### P5 — C11 VPIN at 5-min cadence (~1 day engineering)

**What**: Move VPIN entry filter from hourly to 5-min sub-loop in trader. Original C11 SHELVED 2026-05-03 (+3.83pp on 60d, below +5pp ship). Literature uses 1-min cadence — hourly was too slow.
**How**: Real engineering — needs 5-min OHLCV download in `download_macro_data.py`, sub-loop in `crypto_revolut_ed_v2.py`, threshold sweep.
**Decision**: not actionable until someone has a clear day for engineering. Lower priority than P4 because higher effort × similar expected payoff.

### P6 — C15 meta-labeling on SOL/BTC (~6h, blocked on assets shelved)

**What**: Retest meta-labeling on SOL/BTC primaries (current production is ETH-only). Original C15 SHELVED 2026-04-27 (lost on strong ETH primary by −2.12pp). Door explicitly left open for weaker-primary assets per CLAUDE.md.
**Blocker**: SOL/BTC/XRP/LINK all `enabled: false` in `regime_config_ed.json` for diversification/correlation reasons. Re-test requires re-enabling at least one asset.
**Decision**: deprioritized — only relevant if (a) ETH live performance forces asset diversification, OR (b) cross-asset thesis revives.

---

## 🚀 IDEA QUEUE — Tier A: Untested clean (5 items)

| CID | Idea | Effort |
|---|---|---|
| **C13-narrow** | Single-horizon CDaR variant (no regime split) | ~2h |
| **C54** | Time-decay sell threshold | ~1h |
| **C55** | Liquidity-aware entry timing | ~1.5h |
| **C58** | Yield-curve macro regime detector (depends on C41) | ~2h |
| **C59** | K-means cluster regime (multi-dim macro+vol) | ~3h |

C13-narrow has positive prior (C13's 8h Refined #1 +67.03% was strongest CDaR result). C54/C55 are execution-side, distinct from feature-add family. C58/C59 are regime-detection — prior LOW (C56 HMM DEAD Δ−0.93, C57 MS-AR FAIL Δ−1.574).

## 🚀 IDEA QUEUE — Tier B: V3-lit archive-recovered (23 ideas, 16 with ready patchers)

Pulled into the scoreboard 2026-05-04 from `archive/literature_v3_ideas.md`. All 23 still untested.

**Patcher-ready (16)** — already exist in `_idea_patchers/C*_v3lit.py`, launchable via existing harness:
C60, C61, C62, C63, C64, C65, C67, C69, C70, C71, C73, C75, C79, C80, C81, C82

**Patcher-missing (7)** — need writing first:
C66, C68, C72, C74, C76, C77, C78

**Top 5 cheap patcher-ready picks** (by effort × V3 priority):

| CID | Idea | Effort | V3 # |
|---|---|---|---|
| **C62** | DXY Acceleration (2nd derivative of DXY) | ~30 min | #6 |
| **C60** | US Market Hours Flag (binary NYSE 14:30-21:00 UTC) | ~1h | #3 |
| **C61** | Volatility of Volatility | ~1h | #5 |
| **C63** | KAMA Slope (Kaufman Adaptive MA) | ~1.5h | #7 |
| **C64** | Ehlers Fisher Transform | ~1.5h | #8 |

Full C60-C82 list: [ARCHIVED_LOG.md "ARCHIVE-RECOVERED IDEAS C60-C82"](ARCHIVED_LOG.md).

**Pattern caveat**: feature-add family has consistently failed (C32-C40 batch 0 PASS / 1 FAIL / 6 MARGINAL; C03/C12/C23/C29b/C31/C35/C42/C44/C47/C56/C57 all DEAD or marginal). C60-C82 are mostly feature-adds — prior LOW. Allocate ≤1h per first attempt; if smoke shows MARGINAL like the others, the family ceiling is real.

## 🚀 IDEA QUEUE — What was dropped (see [ARCHIVED_LOG.md "IDEA QUEUE drop-list"](ARCHIVED_LOG.md))

18 ideas that lived briefly on the IDEA QUEUE have been closed with verdicts. Curated drop-list lives in ARCHIVED_LOG.md as a quick-lookup section. One-line summary:
- **6 Tier 1 ideas** (C35, C42, C43, C44, C47, C57) — DEAD on 2026-05-07 fixed-harness retest
- **5 Tier 2 ideas** (C03, C12, C23, C29, C31) — DEAD on 2026-05-10 batch
- **4 Tier 3 ideas** (C16-narrow shipped, C48/C52/C53 DEAD)
- **3 STUB-blocked** (C45/C46/C49 — architectural prerequisites)

For revival, check the verdict + re-add to TODO.md only if evidence overturns the closure.

---

# ⚪ P4 — Low priority / Diagnostics / Backlog

## ⚪ TODO 0519C — CPCV HRST diagnostic (trigger-based re-run)

**Search anchor**: `TODO 0519C`

**Status**: ⚪ LOW PRIORITY. Tested 2026-05-11 → matched current method (no Mode T re-rank, no headline win). Kept because the PBO diagnostic remains useful intel — periodic re-runs would catch if a future engine change introduces overfit configs that current 3-fold rolling holdout misses.

**Trigger to re-run** (any of):
- Major engine architecture change (like H75 → H_STRICT_FAMILY merge 2026-05-18) — re-run on new top-6 candidates to verify the new arch isn't producing overfit configs
- Suspicious Mode T win on a new HRST (>+15pp over current production) — use PBO as overfit sanity check before promoting
- Quarterly hygiene check (~2026-08 next)

**Run command** (resumable, single-instance lock):
```powershell
python tools/run_cpcv_hrst_resumable.py
```

ETA ~5-7h Desktop. Engine fork `crypto_trading_system_ed_cpcv.py` + launcher already in place.

Full closure: [ARCHIVED_LOG.md "Closed 2026-05-11 — P3 CPCV HRST"](ARCHIVED_LOG.md).

## ⚪ Kalshi prediction-market integration (backlog — needs API key + impl)

**Source**: [download_macro_data.py:1501](download_macro_data.py#L1501) — `# TODO: implement when API key available`

**What**: download crypto-related prediction market data from Kalshi (https://kalshi.com/). Currently a stub: function exists, exits early if no `KALSHI_API_KEY` env var or `config/kalshi_config.json`. Implementation never written.

**Why low priority**:
- Requires user to register for Kalshi API access
- New macro feature — same family as GDELT (DEAD), stablecoin mcap (DEAD), C32-C40 batch (mostly MARGINAL/FAIL)
- Feature-add ceiling on this engine is at-or-near zero per the 2026-05-09 retro
- Not actionable until both (a) API key obtained, (b) someone has time to write `download_kalshi_data()`

**Trigger to action**: only if user obtains API key AND has a specific hypothesis about prediction-market data beating VIX/equity-1d-change for macro fear signal.

---

**Honest expectation across the backlog**: per the 2026-05-09 batch retro + 2026-05-19 audit, the H75 production engine is at-or-near its alpha ceiling from feature/scoring tweaks. **8 of 11 originally-queued ideas DEAD on fixed harness**. Future meaningful gains likely come from **execution-gap research** (~17pp untouched alpha per ARCHIVED_LOG.md:1060). C54/C55 in Tier A and P5 VPIN-5min are the only execution-side candidates currently scoped.
