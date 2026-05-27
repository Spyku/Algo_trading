"""
counterfactual_backtest.py — "What if the cache bug had been fixed?"
====================================================================

Created 2026-05-27 evening as follow-up to TODO 0527 (`_macro_cache` mtime fix).

For each hour in the validation window:
  1. Stage oldest-wins merged archeology, truncated to <= signal_dt (PIT data
     reconstruction — see tools/validate_core_point_in_time.py for the merge
     rationale).
  2. Use the CURRENT LIVE config (G_narrow models on H75 engine, stable since
     2026-05-21 21:56). For the May 21-27 window this is the only config that
     was active; pre-May-21 the same config would still be a valid "what if
     we had this model + fresh features" question, just with different model
     parameters than what the trader actually used.
  3. Run compute_signal_core() to get the COUNTERFACTUAL signal — i.e., what
     the trader would have produced AT THAT HOUR if the cache had been fresh.
  4. Apply min_confidence threshold to get the final action (BUY/SELL/HOLD).
  5. Run a stateful trade simulator that walks the hourly bar series and
     enters / exits / times-out positions based on the counterfactual signals.

Then run the SAME trade simulator on the actual signal_log.csv signals (which
were generated with the broken-cache trader) to produce a "simulated-from-
actual" trade set, and compare.

OUTPUT
------
- output/counterfactual_trades_<ts>.csv  — full trade log (counterfactual)
- output/actual_trades_<ts>.csv          — trade log derived from signal_log
- Summary printed to stdout: WR / total return / trade count / per-trade stats

USAGE
-----
    python tools/counterfactual_backtest.py
    python tools/counterfactual_backtest.py --start 2026-05-22 --end 2026-05-27
    python tools/counterfactual_backtest.py --hourly-step 1   # default
    python tools/counterfactual_backtest.py --hourly-step 4   # subsample for speed

LIMITATIONS
-----------
- Archeology coverage starts 2026-05-07. Hours before that fall back to current
  data/, which has been drift-corrupted by months of keep='last' overwrites.
- The trade simulator is a SIMPLIFIED model: BUY at next-hour open, SELL at next-
  hour open, max_hold-hours forced exit. It does NOT model rally-cooldown gates,
  hold_shield, partial fills, maker-order timing, or position-size dynamics —
  all of which are OFF or simplified in the current LIVE config anyway. Per-leg
  cost is hardcoded at 11 bps (TRADING_FEE_BASE 9bps + SLIPPAGE 2bps) to match
  the engine.
- This is a counterfactual SIGNAL comparison, not a perfect execution simulator.
  Interpret WR/return deltas as DIRECTIONAL evidence of cache-bug impact, not
  precise dollar amounts.
"""

import argparse
import io
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# UTF-8 stdout
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

# Working staging dir for the engine to read PIT-truncated archeology
WORKING_SNAPSHOT_DIR = REPO_ROOT / "_pit_workdir"
WORKING_DATA_DIR = WORKING_SNAPSHOT_DIR / "data"
WORKING_DATA_DIR.mkdir(parents=True, exist_ok=True)
(WORKING_DATA_DIR / "macro_data").mkdir(parents=True, exist_ok=True)

# Clear conflicting env vars then set V2_DATA_SNAPSHOT before engine import
for v in ("H_STRICT_MODELS_DIR", "H_STRICT_CONFIG_DIR",
          "H75_WIDE_MODELS_DIR", "H75_WIDE_CONFIG_DIR"):
    if os.environ.get(v):
        del os.environ[v]
os.environ["V2_DATA_SNAPSHOT"] = str(WORKING_SNAPSHOT_DIR.relative_to(REPO_ROOT)).replace("\\", "/")

import pandas as pd
import numpy as np


# ============================================================================
# Constants — matched to current live config
# ============================================================================
TRADING_FEE_BASE = 0.0009       # 0.09% Revolut X taker fee
SLIPPAGE = 0.0002               # 0.02% slippage assumption
TRADING_FEE = TRADING_FEE_BASE + SLIPPAGE  # 11 bps per leg
MAX_HOLD_HOURS = 10              # from current regime config

ARCHEOLOGY_FILES = [
    ("eth_hourly_data.csv", ""),
    ("derivatives_eth.csv", "macro_data"),
    ("onchain_eth.csv", "macro_data"),
    ("macro_daily.csv", "macro_data"),
    ("cross_asset.csv", "macro_data"),
    ("fear_greed.csv", "macro_data"),
    ("stablecoin_flows.csv", "macro_data"),
    ("btc_hourly_data.csv", ""),
]

ARCHEOLOGY_ROOT = REPO_ROOT / "data" / "_archeology"


# ============================================================================
# Snapshot index — copied from validate_core_point_in_time.py
# ============================================================================
def parse_snap_ts(folder_name):
    m = re.match(r"snap_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", folder_name)
    if m:
        y, mo, d, h, mi, s = map(int, m.groups())
        return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)
    return None


def build_snapshot_index():
    """Return {(filename, subdir): [(ts, path), ...]} sorted by ts."""
    index = {(f, sd): [] for f, sd in ARCHEOLOGY_FILES}
    for filename, subdir in ARCHEOLOGY_FILES:
        arch_file_root = ARCHEOLOGY_ROOT / filename
        if not arch_file_root.exists():
            continue
        for snap_folder in arch_file_root.iterdir():
            if not snap_folder.is_dir():
                continue
            ts = parse_snap_ts(snap_folder.name)
            if ts is None:
                continue
            csv_path = snap_folder / filename
            if csv_path.exists():
                index[(filename, subdir)].append((ts, csv_path))
    for key in index:
        index[key].sort(key=lambda x: x[0])
    return index


def build_merged_archeology(index_entries, filename):
    """Oldest-wins reconstruction — for each row, keep value from oldest snapshot containing it."""
    if not index_entries:
        return None
    sample_path = index_entries[0][1]
    sample_df = pd.read_csv(sample_path, nrows=5)
    key_col = None
    for cand in ("datetime", "date", "timestamp"):
        if cand in sample_df.columns:
            key_col = cand
            break
    if key_col is None:
        return None

    merged = None
    seen_keys = set()
    for ts, path in index_entries:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if key_col not in df.columns:
            continue
        parsed = pd.to_datetime(df[key_col], errors="coerce", utc=True)
        df = df[parsed.notna()].copy()
        parsed = parsed[parsed.notna()]
        new_mask = ~parsed.isin(seen_keys)
        new_rows = df[new_mask.values]
        if len(new_rows):
            if merged is None:
                merged = new_rows.copy()
            else:
                merged = pd.concat([merged, new_rows], ignore_index=True)
            seen_keys.update(parsed[new_mask.values].tolist())
    if merged is None:
        return None
    sort_keys = pd.to_datetime(merged[key_col], errors="coerce", utc=True)
    merged = merged.iloc[sort_keys.argsort().values].reset_index(drop=True)
    return merged


def _truncate_and_write(source_df, dest, signal_dt):
    """Filter rows of an already-loaded merged DataFrame to <= signal_dt and write to dest."""
    if source_df is None or source_df.empty:
        return 0
    time_col = None
    for cand in ("datetime", "date", "timestamp"):
        if cand in source_df.columns:
            time_col = cand
            break
    if time_col is None:
        source_df.to_csv(dest, index=False)
        return len(source_df)
    parsed = pd.to_datetime(source_df[time_col], errors="coerce", utc=True)
    keep_mask = parsed.notna() & (parsed <= signal_dt)
    df_trunc = source_df.loc[keep_mask.values].copy()
    df_trunc.to_csv(dest, index=False)
    return len(df_trunc)


def stage_merged_for(signal_dt, merged_cache):
    """Write each merged DataFrame, truncated to <= signal_dt, into _pit_workdir/data/<file>."""
    (WORKING_DATA_DIR / "macro_data").mkdir(parents=True, exist_ok=True)
    for (filename, subdir), merged_df in merged_cache.items():
        if subdir:
            dest = WORKING_DATA_DIR / subdir / filename
        else:
            dest = WORKING_DATA_DIR / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                if dest.exists():
                    dest.unlink()
                _truncate_and_write(merged_df, dest, signal_dt)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.5)


# ============================================================================
# Regime detection (sma24 > sma100 on price)
# ============================================================================
def detect_regime_simple(price_csv_path, as_of_dt):
    df = pd.read_csv(price_csv_path)
    if "datetime" not in df.columns:
        return "bull"
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    target_naive = as_of_dt.astimezone(timezone.utc).replace(tzinfo=None)
    df["dt_naive"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[df["dt_naive"] <= target_naive].sort_values("dt_naive")
    if len(df) < 100:
        return "bull"
    closes = df["close"].astype(float)
    sma24 = closes.rolling(24).mean().iloc[-1]
    sma100 = closes.rolling(100).mean().iloc[-1]
    return "bull" if sma24 > sma100 else "bear"


# ============================================================================
# Single-hour inference
# ============================================================================
def run_inference(asset, config, csc, engine, lt):
    """Given an asset + model config dict + staged data, produce a raw signal.

    Returns {'signal', 'confidence', 'horizon', 'error'} where signal is
    'BUY'/'SELL'/'HOLD' from the live-trader convention.
    """
    try:
        model_names = config["models"].split("+")
        window = int(config["best_window"])
        fs = config.get("feature_set", "A")
        horizon = int(config.get("horizon", lt.HORIZON_SHORT))
        opt_features = str(config.get("optimal_features", "") or "")
        gamma = float(config.get("gamma", 1.0))

        if fs in ("D", "E2", "E3") and opt_features.strip() and opt_features.strip() != "nan":
            feature_list = [f.strip() for f in opt_features.split(",") if f.strip() and f.strip() != "nan"]
        elif fs == "B":
            feature_list = list(lt.FEATURE_SET_B)
        else:
            feature_list = list(lt.FEATURE_SET_A)

        df_raw = lt.load_data(asset)
        if df_raw is None or len(df_raw) == 0:
            return {"error": "df_raw_empty", "horizon": horizon}

        df_full, all_cols = engine.build_all_features(
            df_raw, asset_name=asset, horizon=horizon,
            verbose=False, keep_label_nan_tail=True,
        )
        engine._compute_pysr_features(df_full, all_cols, asset, horizon, verbose=False)

        feature_cols = [f for f in feature_list if f in all_cols]
        if not feature_cols:
            return {"error": "no_features", "horizon": horizon}

        df_train = df_full.dropna(subset=["label"]).reset_index(drop=True)
        df = df_full.reset_index(drop=True).copy()
        df[feature_cols] = df[feature_cols].ffill().fillna(0.0)
        df_train[feature_cols] = df_train[feature_cols].ffill().fillna(0.0)

        n_train = len(df_train)
        if n_train < window + 100:
            return {"error": f"insufficient_training({n_train}<{window+100})", "horizon": horizon}

        train_start = max(0, n_train - window)
        X_train = df_train.iloc[train_start:][feature_cols].values
        y_train = df_train.iloc[train_start:]["label"].values
        i = len(df) - 1
        X_test = df.iloc[i:i+1][feature_cols].values

        factories = {name: lt.ALL_MODELS[name] for name in model_names if name in lt.ALL_MODELS}
        if not factories:
            return {"error": f"no_models:{model_names}", "horizon": horizon}

        result = csc.compute_signal_core(
            X_train=X_train, y_train=y_train, X_test=X_test,
            model_factories=factories, gamma=gamma,
            na_policy="ffill", return_probas=True, binary_signal=False,
        )

        if result is None or result.get("signal") is None:
            return {"error": f"core_skipped:{result.get('skipped_reason') if result else None}",
                    "horizon": horizon}

        return {
            "signal": result["signal"],
            "confidence": result["confidence"],
            "horizon": horizon,
            "error": None,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


# ============================================================================
# Trade simulator — stateful walk through hourly bars
# ============================================================================
def simulate_trades(signals_df, price_df, max_hold_h=MAX_HOLD_HOURS, fee_per_leg=TRADING_FEE):
    """Walk hour-by-hour. If cash + BUY signal -> enter at next bar's open.
    If invested + SELL signal -> exit at next bar's open.
    Forced exit after max_hold_h.

    signals_df: DataFrame with columns ['signal_dt' (naive UTC), 'action', 'confidence']
                where action is 'BUY'/'SELL'/'HOLD'.
    price_df:   DataFrame with columns ['datetime' (naive UTC), 'open', 'close'].

    Returns list of trade dicts.
    """
    # Build a hourly-indexed lookup for fast open-price lookup
    pdf = price_df.copy()
    pdf["datetime"] = pd.to_datetime(pdf["datetime"], errors="coerce")
    if pdf["datetime"].dt.tz is not None:
        pdf["datetime"] = pdf["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    pdf = pdf.set_index("datetime").sort_index()

    # Signals sorted by time
    sdf = signals_df.copy()
    sdf["signal_dt"] = pd.to_datetime(sdf["signal_dt"], errors="coerce")
    if sdf["signal_dt"].dt.tz is not None:
        sdf["signal_dt"] = sdf["signal_dt"].dt.tz_convert("UTC").dt.tz_localize(None)
    sdf = sdf.sort_values("signal_dt").reset_index(drop=True)

    trades = []
    state = "cash"
    entry_price = None
    entry_dt = None

    def _open_at(dt):
        """Return open price for the hour bar starting at dt (or first bar after dt)."""
        floor_hour = pd.Timestamp(dt).floor("h")
        # Try several next-hour candidates to skip gaps
        for offset in range(0, 25):
            cand = floor_hour + pd.Timedelta(hours=offset)
            if cand in pdf.index:
                return float(pdf.loc[cand, "open"]), cand
        return None, None

    for _, row in sdf.iterrows():
        sig_dt = row["signal_dt"]
        action = str(row["action"]).upper()
        conf = float(row["confidence"]) if pd.notna(row["confidence"]) else None

        # Force exit if max_hold reached (before considering current signal)
        if state == "invested" and entry_dt is not None:
            hold_h = (sig_dt - entry_dt).total_seconds() / 3600.0
            if hold_h >= max_hold_h:
                exit_price, exit_dt = _open_at(sig_dt + pd.Timedelta(hours=1))
                if exit_price is not None:
                    gross = (exit_price / entry_price) - 1.0
                    net = gross - 2 * fee_per_leg
                    trades.append({
                        "entry_dt": entry_dt, "entry_price": entry_price,
                        "exit_dt": exit_dt, "exit_price": exit_price,
                        "hold_hours": (exit_dt - entry_dt).total_seconds() / 3600.0,
                        "gross_return": gross, "net_return": net,
                        "win": net > 0, "exit_reason": "max_hold",
                    })
                    state = "cash"
                    entry_price = entry_dt = None

        # Process this signal
        if state == "cash" and action == "BUY":
            entry_price, entry_dt = _open_at(sig_dt + pd.Timedelta(hours=1))
            if entry_price is not None:
                state = "invested"
        elif state == "invested" and action == "SELL":
            exit_price, exit_dt = _open_at(sig_dt + pd.Timedelta(hours=1))
            if exit_price is not None and entry_price is not None:
                gross = (exit_price / entry_price) - 1.0
                net = gross - 2 * fee_per_leg
                trades.append({
                    "entry_dt": entry_dt, "entry_price": entry_price,
                    "exit_dt": exit_dt, "exit_price": exit_price,
                    "hold_hours": (exit_dt - entry_dt).total_seconds() / 3600.0,
                    "gross_return": gross, "net_return": net,
                    "win": net > 0, "exit_reason": "signal_sell",
                })
                state = "cash"
                entry_price = entry_dt = None

    return trades


def trade_summary(trades, label=""):
    n = len(trades)
    if n == 0:
        return {"label": label, "n_trades": 0, "wr": None, "total_return": None,
                "avg_return": None, "best": None, "worst": None}
    wins = sum(1 for t in trades if t["win"])
    total_net = 1.0
    for t in trades:
        total_net *= (1.0 + t["net_return"])
    total_net -= 1.0
    nets = [t["net_return"] for t in trades]
    return {
        "label": label,
        "n_trades": n,
        "wr": 100 * wins / n,
        "total_return": total_net,
        "avg_return": np.mean(nets),
        "best": max(nets),
        "worst": min(nets),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-05-22T00:00",
                        help="Start of counterfactual window (ISO, UTC). Default: 2026-05-22T00:00")
    parser.add_argument("--end", default=None,
                        help="End of window. Default: 1h before now")
    parser.add_argument("--hourly-step", type=int, default=1,
                        help="Step in hours between inference points (default 1 = every hour)")
    parser.add_argument("--asset", default="ETH")
    args = parser.parse_args()

    end_dt = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    if args.end:
        end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)

    print("=" * 78)
    print("  Counterfactual backtest — what if cache had been fresh?")
    print("=" * 78)
    print(f"  Asset:  {args.asset}")
    print(f"  Window: {start_dt.isoformat()} -> {end_dt.isoformat()}")
    print(f"  Step:   {args.hourly_step}h")
    print(f"  Fees:   {TRADING_FEE * 1e4:.0f} bps/leg (TRADING_FEE_BASE + SLIPPAGE)")
    print(f"  Max hold: {MAX_HOLD_HOURS}h")
    print("=" * 78)

    # [1] Build snapshot index + merged archeology
    print("\n[1/5] Building snapshot index + oldest-wins merged archeology...")
    index = build_snapshot_index()
    merged_cache = {}
    for (filename, subdir), entries in index.items():
        if not entries:
            print(f"  {filename}: NO archeology — will use current data/")
            continue
        merged = build_merged_archeology(entries, filename)
        if merged is not None:
            merged_cache[(filename, subdir)] = merged
            print(f"  {filename}: merged {len(merged)} rows from {len(entries)} snapshots")

    # [2] Import engine + core
    print("\n[2/5] Importing engine + core...")
    import crypto_signal_core as csc
    import crypto_trading_system_ed as engine
    import crypto_live_trader_ed as lt
    print("  OK")

    # [3] Load CURRENT live config + price history for trade simulation
    print("\n[3/5] Loading current production config + price history...")
    prod = pd.read_csv("models/crypto_ed_production.csv")
    prod_eth = prod[prod["coin"] == args.asset]
    cfg_bull = prod_eth[prod_eth["horizon"] == 5].iloc[0].to_dict() if not prod_eth[prod_eth["horizon"] == 5].empty else None
    cfg_bear = prod_eth[prod_eth["horizon"] == 8].iloc[0].to_dict() if not prod_eth[prod_eth["horizon"] == 8].empty else None
    if cfg_bull is None or cfg_bear is None:
        print(f"  ERROR: missing 5h or 8h config for {args.asset}")
        return
    with open("config/regime_config_ed.json", encoding="utf-8") as f:
        regime_cfg_full = json.load(f)
    bull_conf = regime_cfg_full[args.asset]["bull"].get("min_confidence", 65)
    bear_conf = regime_cfg_full[args.asset]["bear"].get("min_confidence", 65)
    print(f"  Bull: 5h @ {bull_conf}% — {cfg_bull['best_combo']} w={cfg_bull['best_window']} γ={cfg_bull['gamma']}")
    print(f"  Bear: 8h @ {bear_conf}% — {cfg_bear['best_combo']} w={cfg_bear['best_window']} γ={cfg_bear['gamma']}")

    # Price history (untruncated, for trade execution at next-hour open)
    price_df = pd.read_csv(f"data/{args.asset.lower()}_hourly_data.csv")

    # [4] Loop over hourly grid, build counterfactual signals
    print(f"\n[4/5] Running counterfactual inference at every {args.hourly_step}h step...")
    counterfactual_rows = []
    cur = start_dt
    step = timedelta(hours=args.hourly_step)
    total = int((end_dt - start_dt) / step) + 1
    idx = 0
    while cur <= end_dt:
        idx += 1
        # Stage truncated archeology for this hour
        stage_merged_for(cur, merged_cache)

        # Determine regime from staged price file
        price_path = WORKING_DATA_DIR / "eth_hourly_data.csv"
        regime = detect_regime_simple(price_path, cur) if price_path.exists() else "bull"
        cfg = cfg_bull if regime == "bull" else cfg_bear
        min_conf = bull_conf if regime == "bull" else bear_conf

        # Run inference
        result = run_inference(args.asset, cfg, csc, engine, lt)

        if result.get("error"):
            counterfactual_rows.append({
                "signal_dt": cur.replace(tzinfo=None), "regime": regime, "horizon": cfg["horizon"],
                "raw_signal": None, "raw_conf": None, "action": "SKIP",
                "confidence": None, "error": result["error"],
            })
            if idx % 10 == 0 or idx == total:
                print(f"  [{idx}/{total}] {cur.isoformat()} {regime[:4]} h={cfg['horizon']}  ERROR: {result['error'][:50]}")
            cur += step
            continue

        raw_signal = result["signal"]
        raw_conf = result["confidence"]
        # Apply min_confidence threshold to get final action
        if raw_signal == "BUY" and raw_conf is not None and raw_conf >= min_conf:
            action = "BUY"
        elif raw_signal == "SELL" and raw_conf is not None and raw_conf >= min_conf:
            action = "SELL"
        else:
            action = "HOLD"

        counterfactual_rows.append({
            "signal_dt": cur.replace(tzinfo=None), "regime": regime, "horizon": cfg["horizon"],
            "raw_signal": raw_signal, "raw_conf": raw_conf,
            "action": action, "confidence": raw_conf, "error": None,
        })
        if idx % 10 == 0 or idx == total:
            print(f"  [{idx}/{total}] {cur.isoformat()} {regime[:4]} h={cfg['horizon']}  "
                  f"raw={raw_signal}({raw_conf}) -> {action}")
        cur += step

    counterfactual_df = pd.DataFrame(counterfactual_rows)
    print(f"\n  Counterfactual signals: {len(counterfactual_df)} hours")
    print(f"  BUY: {(counterfactual_df['action'] == 'BUY').sum()}  "
          f"SELL: {(counterfactual_df['action'] == 'SELL').sum()}  "
          f"HOLD: {(counterfactual_df['action'] == 'HOLD').sum()}  "
          f"SKIP: {(counterfactual_df['action'] == 'SKIP').sum()}")

    # [5] Simulate trades from counterfactual signals AND from actual signal_log
    print("\n[5/5] Simulating trades on counterfactual and actual signals...")

    cf_signals = counterfactual_df[counterfactual_df["action"].isin(["BUY", "SELL", "HOLD"])].copy()
    cf_trades = simulate_trades(cf_signals.rename(columns={"signal_dt": "signal_dt"}), price_df)

    # Actual: read signal_log, filter to window
    actual = pd.read_csv("config/signal_log.csv")
    actual["signal_dt"] = pd.to_datetime(actual["timestamp"], errors="coerce")
    actual = actual[(actual["asset"] == args.asset)
                    & (actual["signal_dt"] >= pd.Timestamp(start_dt.replace(tzinfo=None)))
                    & (actual["signal_dt"] <= pd.Timestamp(end_dt.replace(tzinfo=None)))].copy()
    actual_trades = simulate_trades(actual.rename(columns={"action": "action"}), price_df)

    # Write trade logs
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    cf_out = out_dir / f"counterfactual_trades_{ts_str}.csv"
    actual_out = out_dir / f"actual_trades_{ts_str}.csv"
    if cf_trades:
        pd.DataFrame(cf_trades).to_csv(cf_out, index=False)
    if actual_trades:
        pd.DataFrame(actual_trades).to_csv(actual_out, index=False)

    sigs_out = out_dir / f"counterfactual_signals_{ts_str}.csv"
    counterfactual_df.to_csv(sigs_out, index=False)

    cf_sum = trade_summary(cf_trades, "Counterfactual (cache fix applied retroactively)")
    actual_sum = trade_summary(actual_trades, "Actual (broken cache, signal_log)")

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("  COUNTERFACTUAL RESULTS")
    print("=" * 78)
    for s in [actual_sum, cf_sum]:
        print(f"\n  {s['label']}:")
        if s["n_trades"] == 0:
            print(f"    No trades closed in window.")
            continue
        print(f"    Trades:        {s['n_trades']}")
        print(f"    Win rate:      {s['wr']:.1f}%")
        print(f"    Total return:  {s['total_return'] * 100:+.2f}%")
        print(f"    Avg / trade:   {s['avg_return'] * 100:+.3f}%")
        print(f"    Best trade:    {s['best'] * 100:+.2f}%")
        print(f"    Worst trade:   {s['worst'] * 100:+.2f}%")

    # Delta
    if actual_sum["n_trades"] and cf_sum["n_trades"]:
        print(f"\n  DELTA (counterfactual - actual):")
        print(f"    Trade count:   {cf_sum['n_trades'] - actual_sum['n_trades']:+d}")
        print(f"    Win rate:      {cf_sum['wr'] - actual_sum['wr']:+.1f}pp")
        print(f"    Total return:  {(cf_sum['total_return'] - actual_sum['total_return']) * 100:+.2f}pp")

    print(f"\n  Output files:")
    print(f"    {sigs_out}")
    if cf_trades:
        print(f"    {cf_out}")
    if actual_trades:
        print(f"    {actual_out}")
    print("=" * 78)


if __name__ == "__main__":
    main()
