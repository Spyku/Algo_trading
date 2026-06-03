"""
validate_core_against_signal_log.py — Path A validation for TODO 0526
======================================================================

Created 2026-05-27 (after midnight of 0526 session).

PURPOSE
-------
Validate that `compute_signal_core()` produces the same signals as the live
trader produced at historical hours. Uses signal_log.csv (1474 ETH entries
since 2026-03-20) as ground truth.

For each sampled hour H:
  1. Truncate data to <= H (simulate "as of hour H")
  2. Determine regime via sma24>sma100 (live trader's detector)
  3. Pick horizon (bull=5h, bear=8h per current LIVE config)
  4. Run compute_signal_core() with that horizon's production config
  5. Apply the regime's min_confidence gate to the core signal (sub-threshold
     directional calls → HOLD, matching the live trader's action mapping)
  6. Compare to signal_log entry at hour H

Caveat: data files have CURRENT values, not the values live trader saw at
hour H. Where data has drifted, results will differ. For LIVE 5h features
we already measured drift is small (~1 cell of derivatives, no macro).

Usage
-----
  python tools/validate_core_against_signal_log.py
  python tools/validate_core_against_signal_log.py --samples 50
  python tools/validate_core_against_signal_log.py --samples 200 --recent-only

Output
------
  output/core_validation_<ts>.csv      — per-hour comparison
  Console summary with verdict (CORE VALIDATED / PARTIAL / NEEDS DEBUG)
"""

import argparse
import io
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# UTF-8 output for Windows — use reconfigure (gentler than wrapping buffer),
# avoids the import-hang we hit when wrapping sys.stdout.buffer with TextIOWrapper.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass  # older Python, console will replace non-cp1252 chars with ? — cosmetic only

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

# Make sure no leftover env var redirects the engine
for v in ("V2_DATA_SNAPSHOT", "H_STRICT_MODELS_DIR", "H_STRICT_CONFIG_DIR",
          "H75_WIDE_MODELS_DIR", "H75_WIDE_CONFIG_DIR"):
    if os.environ.get(v):
        print(f"  [warning] env var {v}={os.environ[v]!r} — UNSET to avoid data redirect", flush=True)
        del os.environ[v]


def _load_with_imports(engine_name="ed"):
    """Import everything lazily so any import error has a clear message.
    engine_name='faye' uses the lagged FAYE engine (build_all_features +
    _compute_pysr_features reading models_faye/) — tests the daily-lag fix."""
    import crypto_signal_core as csc
    if engine_name == "faye":
        import crypto_trading_system_faye as engine
    else:
        import crypto_trading_system_ed as engine
    import crypto_live_trader_ed as lt
    return csc, engine, lt


def _cpu_lgbm_factory(seed=42):
    """LGBM factory forced to CPU device. Used when --cpu-lgbm is set, to isolate
    GPU non-determinism (different GPUs / different machines produce slightly
    different LGBM probabilities, which can flip signals near 0.5 boundary)."""
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        class_weight='balanced', verbose=-1, random_state=seed,
        device='cpu',
    )


def replay_hour(df_raw_truncated, config, asset, csc, engine, lt, verbose=False, cpu_lgbm=False):
    """Replicate the live trader's data prep + call compute_signal_core.

    Returns a dict with: signal, confidence, horizon, error (or None).
    If cpu_lgbm=True, overrides the LGBM factory to use CPU device."""
    try:
        model_names = config["models"].split("+")
        window = int(config["best_window"])
        fs = config.get("feature_set", "A")
        horizon = int(config.get("horizon", lt.HORIZON_SHORT))
        opt_features = str(config.get("optimal_features", "") or "")
        gamma = float(config.get("gamma", 1.0))

        if fs in ("D", "E2", "E3") and opt_features and opt_features.strip() and opt_features.strip() != "nan":
            feature_list = [f.strip() for f in opt_features.split(",") if f.strip() and f.strip() != "nan"]
        elif fs == "B":
            feature_list = list(lt.FEATURE_SET_B)
        else:
            feature_list = list(lt.FEATURE_SET_A)

        df_full, all_cols = engine.build_all_features(
            df_raw_truncated, asset_name=asset, horizon=horizon,
            verbose=False, keep_label_nan_tail=True,
        )
        engine._compute_pysr_features(df_full, all_cols, asset, horizon, verbose=False)

        feature_cols = [f for f in feature_list if f in all_cols]
        if not feature_cols:
            return {"error": "no_features_available", "horizon": horizon}

        df_train = df_full.dropna(subset=["label"]).reset_index(drop=True)
        df = df_full.reset_index(drop=True).copy()
        df[feature_cols] = df[feature_cols].ffill().fillna(0.0)
        df_train[feature_cols] = df_train[feature_cols].ffill().fillna(0.0)

        n_train = len(df_train)
        if n_train < window + 100:
            return {"error": f"insufficient_training_data(n_train={n_train},window={window})", "horizon": horizon}

        train_start = max(0, n_train - window)
        X_train = df_train.iloc[train_start:][feature_cols].values
        y_train = df_train.iloc[train_start:]["label"].values
        i = len(df) - 1
        X_test = df.iloc[i:i + 1][feature_cols].values

        factories = {name: lt.ALL_MODELS[name] for name in model_names if name in lt.ALL_MODELS}
        if not factories:
            return {"error": f"no_models_available:{model_names}", "horizon": horizon}
        # CPU override: replace LGBM factory with CPU-device version to eliminate
        # cross-GPU non-determinism from the comparison
        if cpu_lgbm and "LGBM" in factories:
            factories["LGBM"] = _cpu_lgbm_factory

        result = csc.compute_signal_core(
            X_train=X_train, y_train=y_train, X_test=X_test,
            model_factories=factories, gamma=gamma,
            na_policy="ffill", return_probas=True, binary_signal=False,
        )

        if result is None or result.get("signal") is None:
            return {"error": f"core_skipped:{result.get('skipped_reason') if result else 'none'}",
                    "horizon": horizon}

        return {
            "signal": result["signal"],
            "confidence": result["confidence"],
            "horizon": horizon,
            "n_features": len(feature_cols),
            "n_train": len(y_train),
            "error": None,
        }
    except Exception as e:
        import traceback
        return {"error": f"{type(e).__name__}: {str(e)[:200]}", "traceback": traceback.format_exc()}


def detect_regime_simple(df_truncated, detector_name="sma24>sma100"):
    """Replicate the live trader's named regime detector
    (crypto_live_trader_ed.py::_evaluate_named_detector). The active detector is
    read from regime_config_ed.json, NOT hardcoded — picking the wrong regime
    selects the wrong horizon config and produces false DIFFs.
    df_truncated must have a 'close' column sorted by time. Returns 'bull'/'bear'."""
    import numpy as _np
    closes = df_truncated["close"].astype(float)

    if detector_name == "tsmom_672h":
        # 28-day time-series momentum: bull if log(close_now / close_672h_ago) > 0
        if len(closes) < 680:
            return "bull"  # live default for insufficient history
        return "bull" if _np.log(closes.iloc[-1] / closes.iloc[-672]) > 0 else "bear"

    # default: sma24 > sma100
    if len(closes) < 100:
        return "bull"  # default
    sma24 = closes.rolling(24).mean().iloc[-1]
    sma100 = closes.rolling(100).mean().iloc[-1]
    return "bull" if sma24 > sma100 else "bear"


def apply_confidence_gate(signal, confidence, min_conf):
    """Replicate the live trader's min_confidence gate exactly (crypto_revolut_ed_v2
    compute_asset_signal, Xh_only path):
        SELL              -> SELL              (NEVER confidence-gated — exits pass through)
        BUY, conf >= min  -> BUY
        BUY, conf <  min  -> HOLD              (the "BUY(57%) -> HOLD [low_conf]" mapping)
        HOLD              -> HOLD
    compute_signal_core returns the UNGATED signal, so without this every sub-threshold
    BUY registers as a false DIFF against live's gated HOLD. The gate is asymmetric:
    only BUY is gated, never SELL."""
    if signal == "BUY" and confidence is not None and min_conf is not None and confidence < float(min_conf):
        return "HOLD"
    return signal


def live_horizons_from_row(row):
    """Horizons the live trader actually ran at this hour, read from signal_log's
    per-slot columns (h_1/h_2). Used to bucket each comparison:
      - 'current': the core's horizon was among the live horizons → apples-to-apples
      - 'stale':   live ran a different horizon (row predates a config/horizon change)
      - 'unknown': signal_log logged no horizon (very old rows) → treated as current
    Returns a set of ints (empty → unknown)."""
    hs = set()
    for col in ("h_1", "h_2"):
        v = row.get(col) if hasattr(row, "get") else None
        if v is not None and pd.notna(v):
            try:
                hs.add(int(float(v)))
            except (ValueError, TypeError):
                pass
    return hs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100,
                        help="How many hours to sample from signal_log (default 100)")
    parser.add_argument("--recent-only", action="store_true",
                        help="Sample only the most recent N hours (default: random sample across all)")
    parser.add_argument("--asset", default="ETH")
    parser.add_argument("--engine", default="ed", choices=["ed", "faye"],
                        help="Engine whose build_all_features/PySR + production config to use. "
                             "'faye' = lagged FAYE engine + models_faye/ config (tests the daily-lag fix).")
    parser.add_argument("--cpu-lgbm", action="store_true",
                        help="Force LGBM to use CPU device. Use to isolate GPU non-determinism "
                             "(different GPUs produce slightly different probabilities → flipped signals).")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  Core Validation vs signal_log — {args.samples} samples ({'recent' if args.recent_only else 'random'})")
    print("=" * 70)

    # Imports
    print("\n[1/4] Importing engine + live trader + core...")
    t0 = time.time()
    csc, engine, lt = _load_with_imports(args.engine)
    print(f"  Done ({time.time() - t0:.1f}s)  [engine={args.engine}]")

    # Load signal_log
    print("\n[2/4] Loading signal_log + production configs...")
    sig_log = pd.read_csv("config/signal_log.csv")
    sig_log["timestamp"] = pd.to_datetime(sig_log["timestamp"], errors="coerce")
    sig_log = sig_log[sig_log["asset"] == args.asset].dropna(subset=["timestamp"]).copy()
    sig_log = sig_log.sort_values("timestamp").reset_index(drop=True)
    sig_log["hour"] = sig_log["timestamp"].dt.floor("h")
    print(f"  signal_log: {len(sig_log)} {args.asset} entries from {sig_log['timestamp'].min()} to {sig_log['timestamp'].max()}")

    # Read bull/bear horizons + regime detector from the config — NOT hardcoded.
    # (Was hardcoded 5h/8h + sma24>sma100, which broke when FAYE promoted bull=6h + tsmom_672h.)
    # --engine faye reads the isolated FAYE config + production CSV (lagged models).
    if args.engine == "faye":
        regime_path, prod_csv = "config_faye/regime_config_faye.json", "models_faye/crypto_faye_production.csv"
    else:
        regime_path, prod_csv = "config/regime_config_ed.json", "models/crypto_ed_production.csv"
    with open(regime_path) as f:
        asset_cfg = json.load(f).get(args.asset, {})
    bull_h = asset_cfg.get("bull", {}).get("horizon")
    bear_h = asset_cfg.get("bear", {}).get("horizon")
    bull_minconf = asset_cfg.get("bull", {}).get("min_confidence")
    bear_minconf = asset_cfg.get("bear", {}).get("min_confidence")
    detector_name = asset_cfg.get("regime_detector", {}).get("params", {}).get("name", "sma24>sma100")
    if bull_h is None or bear_h is None:
        print(f"  ERROR: missing bull/bear horizon in {regime_path} for {args.asset}")
        return

    # Load production configs for those horizons
    prod = pd.read_csv(prod_csv)
    prod = prod[prod["coin"] == args.asset]
    cfg_bull = prod[prod["horizon"] == bull_h].iloc[0].to_dict() if not prod[prod["horizon"] == bull_h].empty else None
    cfg_bear = prod[prod["horizon"] == bear_h].iloc[0].to_dict() if not prod[prod["horizon"] == bear_h].empty else None
    if cfg_bull is None or cfg_bear is None:
        print(f"  ERROR: missing bull({bull_h}h) or bear({bear_h}h) config for {args.asset} in production CSV")
        return
    print(f"  detector: {detector_name}")
    print(f"  bull {bull_h}h config: w={cfg_bull['best_window']} {cfg_bull['best_combo']} γ={cfg_bull['gamma']} min_conf={bull_minconf}")
    print(f"  bear {bear_h}h config: w={cfg_bear['best_window']} {cfg_bear['best_combo']} γ={cfg_bear['gamma']} min_conf={bear_minconf}")

    # Load full raw data
    print("\n[3/4] Loading hourly price data...")
    df_raw_full = pd.read_csv(f"data/{args.asset.lower()}_hourly_data.csv", parse_dates=["datetime"])
    df_raw_full = df_raw_full.sort_values("datetime").reset_index(drop=True)
    print(f"  {len(df_raw_full)} hours of {args.asset} price data from {df_raw_full['datetime'].min()} to {df_raw_full['datetime'].max()}")

    # Sample hours
    if args.recent_only:
        sampled = sig_log.tail(args.samples).reset_index(drop=True)
    else:
        sampled = sig_log.sample(n=min(args.samples, len(sig_log)), random_state=42).sort_values("timestamp").reset_index(drop=True)

    # Filter to hours where we have enough price data (need ~500h before for training window)
    earliest_usable = df_raw_full["datetime"].min() + pd.Timedelta(hours=500)
    sampled = sampled[sampled["timestamp"] >= earliest_usable].reset_index(drop=True)
    print(f"  After filtering for sufficient price history: {len(sampled)} samples")

    # Closed-bar realignment (2026-06-03): after the trader's fix #2, live inference
    # uses the last CLOSED bar (the bar that closed at the cycle hour), not the
    # forming current-hour bar. So for entries logged on/after the fix, the core
    # must infer on the bar BEFORE hour_floor (truncate < hour_floor); legacy
    # entries used the forming bar (<= hour_floor). Self-calibrate the cutoff from
    # the first inference_snapshots.jsonl entry (only the patched trader writes it),
    # so we never hardcode a restart time and the transition is handled exactly.
    closed_bar_cutoff = None
    try:
        snap_path = os.path.join("output", "inference_snapshots.jsonl")
        if os.path.exists(snap_path):
            with open(snap_path, encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line:
                        continue
                    _la = pd.Timestamp(json.loads(_line).get("logged_at"))
                    if _la.tzinfo is not None:
                        _la = _la.tz_convert(None)  # -> naive UTC (signal_log is UTC)
                    closed_bar_cutoff = _la
                    break  # first line = earliest = when the patched trader started
    except Exception:
        closed_bar_cutoff = None
    if closed_bar_cutoff is not None:
        n_closed = int((sampled["timestamp"] >= closed_bar_cutoff).sum())
        print(f"  Closed-bar realignment: {n_closed}/{len(sampled)} entries >= {closed_bar_cutoff} UTC "
              f"use bar < hour_floor (rest use <= hour_floor)")

    # Run validation
    print(f"\n[4/4] Running {len(sampled)} replays (~10s each, ~{len(sampled) * 10 // 60} min total)...")
    results = []
    n_match = n_diff = n_error = 0

    for idx, row in sampled.iterrows():
        ts = row["timestamp"]
        hour_floor = row["hour"]
        live_action = row["action"]
        live_conf = float(row["confidence"]) if pd.notna(row["confidence"]) else None

        # Truncate price data to the bar the live trader actually inferred on:
        # closed bar (< hour_floor) for post-fix entries, forming bar (<= hour_floor)
        # for legacy entries. See closed_bar_cutoff above.
        if closed_bar_cutoff is not None and ts >= closed_bar_cutoff:
            df_truncated = df_raw_full[df_raw_full["datetime"] < hour_floor].copy()
            bar_mode = "closed"
        else:
            df_truncated = df_raw_full[df_raw_full["datetime"] <= hour_floor].copy()
            bar_mode = "forming"
        if len(df_truncated) < 500:
            print(f"  [{idx + 1}/{len(sampled)}] {ts} SKIP (insufficient history)")
            n_error += 1
            continue

        regime = detect_regime_simple(df_truncated, detector_name)
        config = cfg_bull if regime == "bull" else cfg_bear

        # Config-aware bucket: did the live trader run the same horizon the core
        # is about to use? If not, this row predates a config/horizon change and
        # an apples-to-apples comparison is impossible — bucket it as 'stale' so
        # it does not pollute the verdict (which keys off 'current' only).
        live_hs = live_horizons_from_row(row)
        core_h = int(config["horizon"])
        if not live_hs:
            bucket = "unknown"
        elif core_h in live_hs:
            bucket = "current"
        else:
            bucket = "stale"

        t0 = time.time()
        core_result = replay_hour(df_truncated, config, args.asset, csc, engine, lt, cpu_lgbm=args.cpu_lgbm)
        elapsed = time.time() - t0

        if core_result.get("error"):
            n_error += 1
            print(f"  [{idx + 1}/{len(sampled)}] {ts} {regime[:4]} h={config['horizon']} live={live_action}({live_conf}) "
                  f"ERROR: {core_result['error'][:60]} ({elapsed:.1f}s)")
            results.append({
                "timestamp": str(ts), "hour_floor": str(hour_floor),
                "regime": regime, "horizon_used": config["horizon"],
                "live_action": live_action, "live_confidence": live_conf,
                "core_action": None, "core_confidence": None,
                "match": None, "conf_delta": None, "config_bucket": bucket, "bar_mode": bar_mode,
                "error": core_result["error"], "elapsed_s": round(elapsed, 2),
            })
            continue

        # Gate the core signal with the same min_confidence the live trader applies,
        # so we compare like-for-like (live `action` is already gated).
        min_conf = bull_minconf if regime == "bull" else bear_minconf
        core_raw = core_result["signal"]
        core_gated = apply_confidence_gate(core_raw, core_result["confidence"], min_conf)
        match = core_gated == live_action
        if match:
            n_match += 1
        else:
            n_diff += 1
        delta = core_result["confidence"] - live_conf if live_conf is not None else None

        stale_tag = "  [STALE: live h={}]".format("/".join(map(str, sorted(live_hs)))) if bucket == "stale" else ""
        gate_tag = f"→{core_gated}" if core_gated != core_raw else ""
        print(f"  [{idx + 1}/{len(sampled)}] {ts} {regime[:4]} h={config['horizon']} "
              f"live={live_action}({live_conf}) core={core_raw}{gate_tag}({core_result['confidence']}) "
              f"{'MATCH' if match else 'DIFF'} ({elapsed:.1f}s){stale_tag}")

        results.append({
            "timestamp": str(ts), "hour_floor": str(hour_floor),
            "regime": regime, "horizon_used": config["horizon"], "config_bucket": bucket,
            "bar_mode": bar_mode,
            "live_action": live_action, "live_confidence": live_conf,
            "core_action": core_gated, "core_action_raw": core_raw,
            "core_confidence": core_result["confidence"],
            "match": match, "conf_delta": round(delta, 2) if delta is not None else None,
            "n_features": core_result.get("n_features"),
            "n_train": core_result.get("n_train"),
            "error": None, "elapsed_s": round(elapsed, 2),
        })

    # Save
    df_res = pd.DataFrame(results)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path("output") / f"core_validation_{ts_str}.csv"
    out_path.parent.mkdir(exist_ok=True)
    df_res.to_csv(out_path, index=False)

    # Summary — config-aware two-bucket report.
    n_valid = n_match + n_diff
    df_valid = df_res[df_res["error"].isna()].copy()
    if "config_bucket" not in df_valid.columns:
        df_valid["config_bucket"] = "unknown"

    def _rate(sub):
        n = len(sub)
        m = int(sub["match"].sum()) if n else 0
        return m, n, (100.0 * m / n if n else 0.0)

    cur = df_valid[df_valid["config_bucket"] == "current"]
    stl = df_valid[df_valid["config_bucket"] == "stale"]
    unk = df_valid[df_valid["config_bucket"] == "unknown"]
    # Verdict bucket = everything we can fairly compare (current + unknown).
    # 'stale' rows ran a different horizon than the live config (they predate a
    # config/horizon change) — reported for transparency but EXCLUDED from the verdict.
    verdict_sub = df_valid[df_valid["config_bucket"] != "stale"]
    cm, cn, cr = _rate(cur)
    sm, sn, sr = _rate(stl)
    um, un, ur = _rate(unk)
    vm, vn, vr = _rate(verdict_sub)

    print("\n" + "=" * 70)
    print("  VALIDATION RESULT")
    print("=" * 70)
    print(f"  Sampled hours:           {len(sampled)}")
    print(f"  Successfully evaluated:  {n_valid}")
    print(f"  Errors (skipped):        {n_error}")
    print()
    print("  CONFIG-AWARE BUCKETS (verdict keys off current-config only):")
    print(f"    current-config (core horizon == live horizon): {cm}/{cn} = {cr:.1f}%")
    if sn:
        print(f"    stale-config   (live ran a different horizon): {sm}/{sn} = {sr:.1f}%   [EXCLUDED from verdict]")
    if un:
        print(f"    unknown        (no horizon in signal_log):     {um}/{un} = {ur:.1f}%   [counted as current]")
    if "conf_delta" in df_valid.columns and df_valid["conf_delta"].notna().any():
        print(f"  Avg conf delta (core-live), all buckets: {df_valid['conf_delta'].mean():+.2f}")
    print(f"\n  Per-horizon match rate (non-stale):")
    for h in sorted(verdict_sub["horizon_used"].dropna().unique()):
        sub = verdict_sub[verdict_sub["horizon_used"] == h]
        if len(sub):
            mr = 100 * sub["match"].sum() / len(sub)
            print(f"    horizon={int(h)}: {int(sub['match'].sum())}/{len(sub)} = {mr:.1f}%")
    print()
    print("  VERDICT (current-config + unknown; stale excluded):")
    if vn < 20:
        print(f"    [INCONCLUSIVE] Only {vn} comparable sample(s)"
              + (f" ({sn} stale row(s) excluded — likely a recent config/horizon change)." if sn else "."))
        print(f"                   Re-run once more post-change hours accumulate (or --samples 200).")
    elif vr >= 90:
        print(f"    [OK] CORE VALIDATED — current-config match {vr:.1f}% >= 90%.")
    elif vr >= 70:
        print(f"    [PARTIAL] current-config match {vr:.1f}% (70-90%). Inspect mismatches in {out_path.name}.")
    else:
        print(f"    [FAIL] current-config match {vr:.1f}% < 70%. CORE NEEDS DEBUG. Inspect {out_path.name}.")
    print(f"\n  Full results: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
