"""
crypto_live_shadow.py — Shadow signal generation for LIVE vs CORE validation
=============================================================================

Created 2026-05-26 for TODO 0526 Phase 2.

PURPOSE
-------
The shadow path runs `compute_signal_core()` in PARALLEL with the existing
`generate_live_signal()` on the SAME inputs at the SAME hour, then logs both
results to `config/shadow_signal_diff.csv`. Trading still uses the OLD signal —
zero behavior change.

After 7-14 days of shadow data, we can confirm whether the core matches the
live trader's output before switching live to call the core (Step 4 of
TODO 0526 migration).

DESIGN
------
- Self-contained module — only imports from crypto_signal_core + the live
  trader's helpers (build_all_features, _compute_pysr_features, ALL_MODELS)
- The shadow function `compute_signal_via_core()` REPLICATES the data prep
  from `generate_live_signal()` line-by-line (read it carefully — any
  divergence in prep is itself a bug worth catching)
- Every call writes ONE row to shadow_signal_diff.csv
- All errors are swallowed — shadow path MUST NEVER crash the live trader

ACTIVATION
----------
The live trader checks `os.environ.get('SHADOW_MODE') == '1'` or a config flag.
When OFF (default), this module is not called. When ON, it runs after every
`generate_live_signal()` call.

LOG SCHEMA — config/shadow_signal_diff.csv
-------------------------------------------
  timestamp_utc           When the comparison was run
  asset                   ETH / BTC / etc.
  horizon                 5 / 6 / 7 / 8
  live_signal             BUY / SELL / HOLD (from generate_live_signal)
  live_confidence         float 0-100
  core_signal             BUY / SELL / HOLD (from compute_signal_core, ternary mode)
  core_confidence         float 0-100
  match                   True if live_signal == core_signal
  conf_delta              core_confidence - live_confidence
  shadow_error            None or error message (any prep/compute failure)
  shadow_elapsed_ms       runtime of shadow path in ms
  n_features              count of features actually used (after intersection)
  n_train                 training-window row count (post-dropna)
  inference_row_dt        timestamp of the row predicted on

When the file doesn't exist, the header is written automatically on first row.
"""

import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Lazy import — only at first use, so import errors don't crash live trader on startup
_engine = None
_live_trader = None
_core = None


def _lazy_import():
    """Import dependencies on first use. Returns True if successful."""
    global _engine, _live_trader, _core
    if _core is not None:
        return True
    try:
        import crypto_signal_core as _core_mod
        import crypto_trading_system_ed as _engine_mod
        import crypto_live_trader_ed as _lt_mod
        _core = _core_mod
        _engine = _engine_mod
        _live_trader = _lt_mod
        return True
    except Exception as e:
        # Log once, don't keep retrying
        sys.stderr.write(f"[SHADOW] Could not import dependencies: {e}\n")
        return False


SHADOW_LOG_PATH = "config/shadow_signal_diff.csv"

LOG_COLUMNS = [
    "timestamp_utc",
    "asset",
    "horizon",
    "live_signal",
    "live_confidence",
    "core_signal",
    "core_confidence",
    "match",
    "conf_delta",
    "shadow_error",
    "shadow_elapsed_ms",
    "n_features",
    "n_train",
    "inference_row_dt",
]


def shadow_compare(
    asset: str,
    config: dict,
    df_raw: Optional[pd.DataFrame],
    live_signal_result: Optional[dict],
) -> None:
    """Called by live trader AFTER generate_live_signal() returns.

    Replicates the live trader's data prep, calls compute_signal_core() on
    equivalent inputs, compares the result to live_signal_result, and appends
    one row to config/shadow_signal_diff.csv.

    NEVER raises. NEVER blocks the live trader. NEVER modifies any state
    other than appending a CSV row.

    Parameters
    ----------
    asset : str
        e.g. 'ETH'
    config : dict
        The config dict that was passed to generate_live_signal()
        (must contain 'models', 'best_window', 'gamma', 'horizon', 'optimal_features', etc.)
    df_raw : pd.DataFrame or None
        Raw OHLCV passed to generate_live_signal()
    live_signal_result : dict or None
        Whatever generate_live_signal() returned (or None if it refused/failed)
    """
    t_start = time.time()
    row: Dict[str, Any] = {col: None for col in LOG_COLUMNS}
    row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    row["asset"] = asset

    try:
        if not _lazy_import():
            row["shadow_error"] = "import_failed"
            _write_row(row)
            return

        # Extract live signal fields
        if live_signal_result is not None and isinstance(live_signal_result, dict):
            row["live_signal"] = live_signal_result.get("signal")
            row["live_confidence"] = live_signal_result.get("confidence")
            row["horizon"] = live_signal_result.get("horizon") or config.get("horizon")
        else:
            row["live_signal"] = "REFUSED"
            row["horizon"] = config.get("horizon")

        # Skip shadow compute if live refused — we can't know what core would have done
        # without going through staleness checks ourselves, and that's complexity we don't
        # need for validation (refusals will show up in the data as gaps).
        if live_signal_result is None:
            row["shadow_error"] = "live_refused_no_compare"
            row["shadow_elapsed_ms"] = round((time.time() - t_start) * 1000, 1)
            _write_row(row)
            return

        # ----- Replicate data prep from generate_live_signal -----
        model_names = config["models"].split("+")
        window = config["best_window"]
        fs = config.get("feature_set", "A")
        horizon = config.get("horizon", _live_trader.HORIZON_SHORT)
        opt_features = config.get("optimal_features", "")
        gamma = config.get("gamma", 1.0)

        if fs in ("D", "E2", "E3") and opt_features and opt_features.strip() and opt_features.strip() != "nan":
            feature_list = [f.strip() for f in opt_features.split(",") if f.strip() and f.strip() != "nan"]
        elif fs == "B":
            feature_list = list(_live_trader.FEATURE_SET_B)
        else:
            feature_list = list(_live_trader.FEATURE_SET_A)

        if df_raw is None:
            row["shadow_error"] = "df_raw_none"
            row["shadow_elapsed_ms"] = round((time.time() - t_start) * 1000, 1)
            _write_row(row)
            return

        df_full, all_cols = _engine.build_all_features(
            df_raw, asset_name=asset, horizon=horizon, verbose=False, keep_label_nan_tail=True
        )
        _engine._compute_pysr_features(df_full, all_cols, asset, horizon, verbose=False)

        feature_cols = [f for f in feature_list if f in all_cols]
        if not feature_cols:
            row["shadow_error"] = "zero_features_available"
            row["n_features"] = 0
            row["shadow_elapsed_ms"] = round((time.time() - t_start) * 1000, 1)
            _write_row(row)
            return

        row["n_features"] = len(feature_cols)

        # The label-NaN tail split and the ffill — same as live trader
        df_train = df_full.dropna(subset=["label"]).reset_index(drop=True)
        df = df_full.reset_index(drop=True).copy()
        df[feature_cols] = df[feature_cols].ffill().fillna(0.0)
        df_train[feature_cols] = df_train[feature_cols].ffill().fillna(0.0)

        n_train = len(df_train)
        if n_train < window + 100:
            row["shadow_error"] = f"insufficient_training_data(n_train={n_train},window={window})"
            row["n_train"] = n_train
            row["shadow_elapsed_ms"] = round((time.time() - t_start) * 1000, 1)
            _write_row(row)
            return

        # Train window: live trader uses [n_train-window, n_train] — NO embargo
        train_start = max(0, n_train - window)
        X_train = df_train.iloc[train_start:][feature_cols].values
        y_train = df_train.iloc[train_start:]["label"].values

        # Inference row: live trader uses the freshest row of df (which has all rows,
        # including the label-NaN tail)
        i = len(df) - 1
        X_test = df.iloc[i : i + 1][feature_cols].values
        row["inference_row_dt"] = str(df.iloc[i].get("datetime", ""))
        row["n_train"] = len(y_train)

        # ----- Build model factories matching live trader -----
        model_factories = {}
        for name in model_names:
            if name in _live_trader.ALL_MODELS:
                model_factories[name] = _live_trader.ALL_MODELS[name]
            else:
                row["shadow_error"] = f"model_not_in_ALL_MODELS:{name}"
                _write_row(row)
                return

        # ----- Call shared core (live convention: ffill + ternary + probas) -----
        core_result = _core.compute_signal_core(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            model_factories=model_factories,
            gamma=gamma,
            na_policy="ffill",      # match live trader
            return_probas=True,     # match live trader
            binary_signal=False,    # match live trader (ternary)
        )

        if core_result is None or core_result.get("signal") is None:
            row["shadow_error"] = f"core_skipped:{core_result.get('skipped_reason') if core_result else 'none'}"
            row["shadow_elapsed_ms"] = round((time.time() - t_start) * 1000, 1)
            _write_row(row)
            return

        row["core_signal"] = core_result["signal"]
        row["core_confidence"] = core_result["confidence"]
        row["match"] = row["core_signal"] == row["live_signal"]
        if row["live_confidence"] is not None and row["core_confidence"] is not None:
            row["conf_delta"] = round(row["core_confidence"] - row["live_confidence"], 2)

    except Exception as e:
        # CATCH ALL — never let shadow crash live trader
        row["shadow_error"] = f"exception: {type(e).__name__}: {str(e)[:200]}"
        try:
            tb = traceback.format_exc()
            # Save full traceback to a separate file (don't pollute the CSV)
            os.makedirs("config", exist_ok=True)
            with open("config/shadow_signal_diff_errors.log", "a", encoding="utf-8") as f:
                f.write(f"\n--- {datetime.now(timezone.utc).isoformat()} {asset} ---\n")
                f.write(tb)
        except Exception:
            pass

    row["shadow_elapsed_ms"] = round((time.time() - t_start) * 1000, 1)
    _write_row(row)


def _write_row(row: Dict[str, Any]) -> None:
    """Append one row to SHADOW_LOG_PATH. Creates header if file doesn't exist.
    Swallows any IO error."""
    try:
        os.makedirs(os.path.dirname(SHADOW_LOG_PATH), exist_ok=True)
        df = pd.DataFrame([{k: row.get(k) for k in LOG_COLUMNS}])
        write_header = not os.path.exists(SHADOW_LOG_PATH)
        df.to_csv(SHADOW_LOG_PATH, mode="a", header=write_header, index=False)
    except Exception as e:
        try:
            sys.stderr.write(f"[SHADOW] CSV write failed: {e}\n")
        except Exception:
            pass


def is_shadow_enabled() -> bool:
    """Single source of truth for whether shadow mode is on.
    Reads env var SHADOW_MODE. Set to '1' to enable."""
    return os.environ.get("SHADOW_MODE", "").strip() == "1"
