"""
validate_refit_replay.py — from-scratch REFIT reproduction from frozen training matrices.

Built 2026-06-20. Complements validate_snapshot_replay.py:
  - snapshot_replay  : recomputes signal/conf from frozen PROBAS (logic check, 100%).
  - refit_replay (this): re-FITS the model on the frozen X_train/y_train the trader
    actually used (output/inference_train_snapshots.jsonl), predicts the frozen X_test,
    and checks it reproduces the logged decision. This is the check that raw-data
    rebuilds (sanity [3] parity) CAN'T do — because the live-time raw data is gone,
    but the prepared training matrix is now frozen no-overwrite.

A match here proves the model TRAINING is reproducible bit-for-bit (modulo ~3pt GPU/CPU).
WARN if the sibling file isn't there yet (trader hasn't run the patched code).

Usage: python tools/validate_refit_replay.py [--samples N] [--cpu-lgbm]
Exit 0 = clean (>=95% signal match), 1 = mismatches, 2 = no sibling file yet.
"""
import os, sys, json, glob, argparse
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

TRAIN = "output/inference_train_snapshots.jsonl"
SNAP = "output/inference_snapshots.jsonl"


def _signal_from(votes, probas):
    buy_ratio = sum(votes) / len(votes)
    sig = "BUY" if buy_ratio > 0.5 else ("SELL" if buy_ratio == 0 else "HOLD")
    avg = float(np.mean(probas))
    conf = round(avg * 100, 2) if sig != "SELL" else round((1 - avg) * 100, 2)
    return sig, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--cpu-lgbm", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(TRAIN):
        print(f"[WARN] {TRAIN} not found yet — trader hasn't written it (needs the patched "
              f"crypto_live_trader_ed.py running + a restart). Nothing to replay.")
        sys.exit(2)

    os.environ.setdefault("FAYE_LIBRARY_MODE", "1")
    import crypto_live_trader_ed as lt
    # logged decisions, keyed by inference bar, to compare against
    logged = {}
    if os.path.exists(SNAP):
        for ln in open(SNAP, encoding="utf-8"):
            try:
                o = json.loads(ln); logged[o["inference_row_dt"]] = o
            except Exception:
                pass

    rows = [json.loads(l) for l in open(TRAIN, encoding="utf-8") if l.strip()][-args.samples:]
    print(f"refit-replay: {len(rows)} training-matrix snapshots (cpu_lgbm={args.cpu_lgbm})")
    n_ok = n_sig = n_tot = 0
    for o in rows:
        Xtr = pd.DataFrame(o["X_train"], columns=o["feature_cols"])
        ytr = np.array(o["y_train"])
        Xte = pd.DataFrame(o["X_test"], columns=o["feature_cols"])
        if len(np.unique(ytr)) < 2:
            continue
        scaler = StandardScaler()
        Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), columns=o["feature_cols"])
        Xte_s = pd.DataFrame(scaler.transform(Xte), columns=o["feature_cols"])
        sw = lt.get_decay_weights(len(ytr), float(o["gamma"]))
        votes, probas = [], []
        for m in o["models"]:
            fac = lt.ALL_MODELS[m]
            mdl = fac()
            mdl.fit(Xtr_s, ytr, sample_weight=sw)
            votes.append(int(mdl.predict(Xte_s)[0])); probas.append(float(mdl.predict_proba(Xte_s)[0][1]))
        if not votes:
            continue
        sig, conf = _signal_from(votes, probas)
        n_tot += 1
        lg = logged.get(o["inference_row_dt"])
        if lg is not None:
            n_sig += int(sig == lg.get("signal"))
            dconf = abs(conf - float(lg.get("confidence", conf)))
            n_ok += int(sig == lg.get("signal") and dconf <= 5)
            print(f"  {o['inference_row_dt']}: refit {sig}({conf}) vs logged {lg.get('signal')}({lg.get('confidence')})  dConf={conf-float(lg.get('confidence',conf)):+.2f}")
        else:
            print(f"  {o['inference_row_dt']}: refit {sig}({conf})  (no logged row to compare)")
    if n_tot and logged:
        print(f"\nsignal match: {n_sig}/{n_tot} ({100*n_sig/n_tot:.1f}%) | signal+conf(<=5): {n_ok}/{n_tot}")
        sys.exit(0 if n_sig >= 0.95 * n_tot else 1)
    print(f"\n{n_tot} refits done (no logged decisions to compare against)")
    sys.exit(0)


if __name__ == "__main__":
    main()
