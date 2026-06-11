"""
validate_snapshot_replay.py — Offline DECISION-LOGIC validator (frozen snapshots)
=================================================================================
Built 2026-06-11. The offline analogue of crypto_live_shadow.py, but immune to
data revision: it replays the trader's signal+confidence aggregation FROM the
point-in-time intermediate values the trader itself recorded — no feature
rebuild, no model retrain, no data download.

INPUT  : output/inference_snapshots.jsonl  (written by
         crypto_live_trader_ed._log_inference_snapshot inside generate_live_signal)
         Each row carries the FROZEN values: buy_ratio, avg_proba, per-model
         probas, signal, confidence (pre-min-confidence-gate aggregation output).

WHAT IT PROVES
--------------
For each snapshot, recompute (signal, confidence) FROM the stored probas /
buy_ratio using the EXACT aggregation + confidence math the live path runs
(crypto_live_trader_ed.generate_live_signal lines ~891-917, mirrored in
crypto_signal_core.compute_signal_core), then ASSERT it equals the logged
(signal, confidence). This is pure bookkeeping on frozen numbers → it MUST be
~100%. Any mismatch is a REAL logic bug (a divergence between the math the
trader claims to run and the values it persisted), not a data artifact.

PRE-GATE NOTE
-------------
The snapshot's `signal`/`confidence` are logged INSIDE generate_live_signal,
BEFORE the regime min_confidence gate (which lives one level up in
generate_regime_signal / compute_signal). So we replay the PRE-gate aggregation
only. The min_confidence gate is a trivial `conf >= min_conf` threshold applied
to this same `signal`; it does not alter the signal/confidence computed here.

PRODUCTION CONFIG ASSUMPTION
----------------------------
The disagreement filter (`disagree_filter`) and funding-rate gate
(`funding_gate`) are OPTIONAL config enhancements absent from
models/crypto_ed_production.csv → they default OFF and never fire. With a
2-model ensemble that makes buy_ratio in {0.0, 0.5, 1.0} and signal a pure
function of buy_ratio. If a future config turns either enhancement ON, the
snapshot would need to log the flag for a faithful replay — flagged below.

Run:  python tools/validate_snapshot_replay.py
"""
import json
import os
import sys

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT_PATH = os.path.join(ENGINE_DIR, 'output', 'inference_snapshots.jsonl')
SIGNAL_LOG_PATH = os.path.join(ENGINE_DIR, 'config', 'signal_log.csv')

# Tolerance for the confidence float compare. Both sides round to 2 decimals,
# so this only absorbs the last-bit float noise of re-deriving the same round().
CONF_TOL = 0.005


def _recompute_buy_ratio_from_probas(probas):
    """Reconstruct the ensemble vote from per-model probas.

    The live trader builds buy_ratio from model.predict() (a hard vote), not from
    predict_proba. For sklearn/LGBM binary classifiers predict() == (proba >= 0.5),
    so we reconstruct votes that way and compare against the logged buy_ratio. A
    disagreement here would mean predict() and predict_proba() disentangle around
    0.5 — surfaced as a 'vote_recon' mismatch rather than silently trusted.
    """
    votes = [1 if p >= 0.5 else 0 for p in probas]
    return sum(votes) / len(votes), votes


def _signal_from_buy_ratio(buy_ratio):
    """EXACT live ternary semantics (crypto_live_trader_ed.py lines 899-904),
    with disagree_filter / funding_gate OFF (production config default).
        buy_ratio > 0.5  -> BUY
        buy_ratio == 0   -> SELL
        else             -> HOLD
    """
    if buy_ratio > 0.5:
        return 'BUY'
    elif buy_ratio == 0:
        return 'SELL'
    else:
        return 'HOLD'


def _confidence(avg_proba, signal):
    """EXACT live confidence formula (crypto_live_trader_ed.py line 917 /
    crypto_signal_core.py line 251): avg_proba*100 for non-SELL, (1-avg_proba)*100
    for SELL, rounded to 2 decimals."""
    return round(avg_proba * 100, 2) if signal != 'SELL' else round((1 - avg_proba) * 100, 2)


def _load_signal_log_index():
    """Map (asset, 'YYYY-MM-DD HH') logged-hour -> list of signal_log rows.
    Best-effort cross-check only; never fatal."""
    idx = {}
    if not os.path.exists(SIGNAL_LOG_PATH):
        return idx
    try:
        import csv
        with open(SIGNAL_LOG_PATH, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                ts = (r.get('timestamp') or '')[:13]  # 'YYYY-MM-DD HH'
                idx.setdefault((r.get('asset'), ts), []).append(r)
    except Exception:
        pass
    return idx


def _populated_slot(sl_row):
    """Return (horizon, signal, confidence) for whichever signal_log slot is
    populated this hour. Per Critical Rule 21 the slot is regime-anchored and
    only ONE regime fires per cycle: bull populates sig_1, bear populates sig_2.
    So we read the filled slot rather than matching on horizon."""
    for slot in ('1', '2'):
        if sl_row.get(f'sig_{slot}'):
            return sl_row.get(f'h_{slot}'), sl_row.get(f'sig_{slot}'), sl_row.get(f'conf_{slot}')
    return None


def main():
    if not os.path.exists(SNAPSHOT_PATH):
        print(f"[X] snapshot file not found: {SNAPSHOT_PATH}")
        return 2

    rows = []
    bad_json = 0
    with open(SNAPSHOT_PATH, encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad_json += 1
    if not rows:
        print("[X] no snapshot rows parsed.")
        return 2

    n = len(rows)
    sig_match = conf_match = both_match = 0
    vote_recon_match = avgproba_match = 0
    mismatches = []   # logic mismatches (the valuable findings)
    notes = []        # non-fatal observations (vote_recon / avg_proba drift)

    for i, r in enumerate(rows):
        dt = r.get('inference_row_dt')
        probas_d = r.get('probas') or {}
        models = r.get('models') or list(probas_d.keys())
        # Preserve model order from `models`; fall back to dict order.
        probas = [probas_d[m] for m in models if m in probas_d] or list(probas_d.values())

        logged_signal = r.get('signal')
        logged_conf = r.get('confidence')
        logged_buy_ratio = r.get('buy_ratio')
        logged_avg_proba = r.get('avg_proba')

        if not probas:
            mismatches.append((i, dt, "NO PROBAS in snapshot", logged_signal, None, logged_conf, None))
            continue

        # --- 1. reconstruct buy_ratio from probas, cross-check vs logged ---
        recon_ratio, votes = _recompute_buy_ratio_from_probas(probas)
        ratio_ok = (logged_buy_ratio is not None
                    and abs(recon_ratio - logged_buy_ratio) < 1e-9)
        if ratio_ok:
            vote_recon_match += 1
        else:
            notes.append(f"  row {i} {dt}: vote-from-proba buy_ratio={recon_ratio} != logged buy_ratio={logged_buy_ratio} "
                         f"(probas={[round(p,4) for p in probas]}) -> predict()/predict_proba() disagree near 0.5")

        # --- 2. recompute avg_proba from probas, cross-check vs logged ---
        recon_avg = sum(probas) / len(probas)
        if logged_avg_proba is not None and abs(recon_avg - logged_avg_proba) < 1e-9:
            avgproba_match += 1
        else:
            notes.append(f"  row {i} {dt}: mean(probas)={recon_avg!r} != logged avg_proba={logged_avg_proba!r}")

        # --- 3. THE ASSERTIONS: replay signal + confidence from FROZEN values ---
        # Use the LOGGED buy_ratio / avg_proba as the source of truth for the
        # aggregation step (these are exactly what the live path fed into the
        # signal/confidence formulas). This isolates the DECISION LOGIC from the
        # vote-reconstruction cross-check above.
        src_ratio = logged_buy_ratio if logged_buy_ratio is not None else recon_ratio
        src_avg = logged_avg_proba if logged_avg_proba is not None else recon_avg

        recomp_signal = _signal_from_buy_ratio(src_ratio)
        recomp_conf = _confidence(src_avg, recomp_signal)

        s_ok = (recomp_signal == logged_signal)
        c_ok = (logged_conf is not None and abs(recomp_conf - logged_conf) <= CONF_TOL)
        sig_match += s_ok
        conf_match += c_ok
        both_match += (s_ok and c_ok)

        if not (s_ok and c_ok):
            mismatches.append((i, dt, "LOGIC MISMATCH", logged_signal, recomp_signal,
                               logged_conf, recomp_conf))

    # --- optional signal_log.csv cross-check (best-effort) ---
    # The snapshot's inference_row_dt is the BAR time; the trader infers on the
    # last CLOSED bar so the signal_log timestamp is ~1h LATER (closed-bar fix #2,
    # 2026-06-03). Align on bar+1h. The slot is regime-anchored (Critical Rule 21),
    # so we read the populated slot, not the horizon-matched one.
    from datetime import datetime as _dt, timedelta as _td
    sl_idx = _load_signal_log_index()
    sl_checked = sl_cross_ok = 0
    sl_notes = []
    if sl_idx:
        for i, r in enumerate(rows):
            dt = r.get('inference_row_dt') or ''
            try:
                logged_hour = (_dt.fromisoformat(dt) + _td(hours=1)).strftime('%Y-%m-%d %H')
            except Exception:
                continue
            hits = sl_idx.get((r.get('asset'), logged_hour))
            if not hits:
                continue
            for row_sl in hits:
                p = _populated_slot(row_sl)
                if not p:
                    continue
                sl_checked += 1
                if p[1] == r.get('signal'):
                    sl_cross_ok += 1
                else:
                    sl_notes.append(f"  row {i} bar={dt} (logged~{logged_hour}:00): snapshot sig={r.get('signal')} "
                                    f"!= signal_log h={p[0]} sig={p[1]} conf={p[2]} "
                                    f"(separate inference cycle / startup retry — non-fatal)")

    # ----------------------------- REPORT -----------------------------
    pct = lambda a: 100.0 * a / n
    print("=" * 70)
    print("SNAPSHOT REPLAY VALIDATION (frozen point-in-time decision logic)")
    print("=" * 70)
    print(f"snapshot file : {SNAPSHOT_PATH}")
    print(f"rows parsed   : {n}" + (f"  ({bad_json} unparseable lines skipped)" if bad_json else ""))
    print(f"signal  match : {sig_match}/{n}  ({pct(sig_match):.2f}%)")
    print(f"confid. match : {conf_match}/{n}  ({pct(conf_match):.2f}%)")
    print(f"BOTH    match : {both_match}/{n}  ({pct(both_match):.2f}%)   <- headline")
    print("-" * 70)
    print(f"buy_ratio reconstructed from probas == logged : {vote_recon_match}/{n} ({pct(vote_recon_match):.2f}%)")
    print(f"mean(probas)           == logged avg_proba    : {avgproba_match}/{n} ({pct(avgproba_match):.2f}%)")
    if sl_idx:
        print(f"signal_log.csv cross-check (same hour+horizon): {sl_cross_ok}/{sl_checked} matched")

    if notes:
        print("-" * 70)
        print(f"NON-FATAL NOTES ({len(notes)}):")
        for ln_ in notes[:20]:
            print(ln_)
        if len(notes) > 20:
            print(f"  ... +{len(notes) - 20} more")
    if sl_notes:
        print("-" * 70)
        print(f"signal_log.csv disagreements ({len(sl_notes)}):")
        for ln_ in sl_notes[:20]:
            print(ln_)

    if mismatches:
        print("-" * 70)
        print(f"LOGIC MISMATCHES ({len(mismatches)}) -- REAL BUGS:")
        print(f"  {'row':>4} {'inference_row_dt':<22} {'kind':<14} "
              f"{'logged_sig':<10} {'recomp_sig':<10} {'logged_conf':>11} {'recomp_conf':>11}")
        for (i, dt, kind, ls, rs, lc, rc) in mismatches[:40]:
            print(f"  {i:>4} {str(dt):<22} {kind:<14} {str(ls):<10} {str(rs):<10} "
                  f"{str(lc):>11} {str(rc):>11}")
        if len(mismatches) > 40:
            print(f"  ... +{len(mismatches) - 40} more")
    else:
        print("-" * 70)
        print("[OK] 0 logic mismatches — recomputed (signal, confidence) == logged for every row.")

    print("=" * 70)
    return 0 if not mismatches else 1


if __name__ == '__main__':
    sys.exit(main())
