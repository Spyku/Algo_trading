"""
counterfactual_ab.py — PIT-config counterfactual + G_narrow vs H75-fresh A/B
============================================================================

Created 2026-05-27 evening as follow-up to the counterfactual_backtest.py
result (13pp underperformance) which was confounded by multi-config history.

WHAT THIS TOOL ANSWERS
----------------------
Two distinct questions, side by side:

  Q1 (PIT mode): For each historical hour, what would have happened if the
      ACTUAL LIVE CONFIG at that moment had had fresh data? Compare to the
      actual broken-cache signal log. Removes the "current G_narrow vs
      historical multi-config mix" confound.

  Q2 (A/B mode): Run the SAME hourly grid TWICE — once with G_narrow config,
      once with H75-fresh config. Same data, same simulator, same fees. Direct
      head-to-head with fresh inputs.

Both modes use oldest-wins merged archeology for the data (recovers as close
to "originally-observed" feature values as we have).

USAGE
-----
    python tools/counterfactual_ab.py
    python tools/counterfactual_ab.py --start 2026-05-07T00:00 --end 2026-05-27T10:00
    python tools/counterfactual_ab.py --asset ETH --hourly-step 1

OUTPUTS
-------
- output/cf_ab_signals_pit_<ts>.csv      — PIT signals (each hour, live-config-at-time)
- output/cf_ab_signals_gnarrow_<ts>.csv  — All hours with G_narrow config
- output/cf_ab_signals_h75fresh_<ts>.csv — All hours with H75-fresh config
- output/cf_ab_trades_pit_<ts>.csv       — PIT mode trades
- output/cf_ab_trades_gnarrow_<ts>.csv   — G_narrow trades
- output/cf_ab_trades_h75fresh_<ts>.csv  — H75-fresh trades
- output/cf_ab_trades_actual_<ts>.csv    — Same simulator on signal_log
- output/cf_ab_summary_<ts>.md           — Human-readable report

RUNTIME
-------
About 3× the runtime of counterfactual_backtest.py (3 inferences per hour
instead of 1). Expect ~90-100 minutes on a 491-hour window with hourly step.
"""

import argparse
import os
import re
import sys
import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

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

# Working dir for engine snapshot redirect
WORKING_SNAPSHOT_DIR = REPO_ROOT / "_pit_workdir"
WORKING_DATA_DIR = WORKING_SNAPSHOT_DIR / "data"
WORKING_DATA_DIR.mkdir(parents=True, exist_ok=True)
(WORKING_DATA_DIR / "macro_data").mkdir(parents=True, exist_ok=True)

for v in ("H_STRICT_MODELS_DIR", "H_STRICT_CONFIG_DIR",
          "H75_WIDE_MODELS_DIR", "H75_WIDE_CONFIG_DIR",
          "G_NARROW_MODELS_DIR", "G_NARROW_CONFIG_DIR"):
    if os.environ.get(v):
        del os.environ[v]
os.environ["V2_DATA_SNAPSHOT"] = str(WORKING_SNAPSHOT_DIR.relative_to(REPO_ROOT)).replace("\\", "/")

import pandas as pd
import numpy as np

# ============================================================================
# Constants
# ============================================================================
TRADING_FEE_BASE = 0.0009
SLIPPAGE = 0.0002
TRADING_FEE = TRADING_FEE_BASE + SLIPPAGE
MAX_HOLD_HOURS = 10

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

# Production CSV snapshots for PIT config lookup
PROD_CSV_NAMES = ["crypto_ed_production.csv"]
PROD_CSV_PATH_CURRENT = REPO_ROOT / "models" / "crypto_ed_production.csv"

# G_narrow promotion event — for partitioning when which config was live
G_NARROW_PROMOTE_UTC = datetime(2026, 5, 21, 21, 56, tzinfo=timezone.utc)

# Backup files for A/B model loading
G_NARROW_PROD_CSV = REPO_ROOT / "models" / "crypto_ed_production.csv"  # current = G_narrow
H75_FRESH_PROD_CSV = REPO_ROOT / "models" / "crypto_ed_production_pre_G_narrow_20260521.csv"


# ============================================================================
# Archeology + oldest-wins merge — copied from counterfactual_backtest.py
# ============================================================================
def parse_snap_ts(folder_name):
    m = re.match(r"snap_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", folder_name)
    if m:
        y, mo, d, h, mi, s = map(int, m.groups())
        return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)
    return None


def build_snapshot_index():
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


def build_merged_archeology(index_entries):
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
# PIT config lookup — pick whichever live config was active at signal_dt
# ============================================================================
def build_prod_index():
    """Build sorted (timestamp, prod_csv_path) entries from archeology of
    crypto_ed_production.csv + the current file (with its mtime)."""
    arch_root = ARCHEOLOGY_ROOT / "crypto_ed_production.csv"
    entries = []
    if arch_root.exists():
        for snap in arch_root.iterdir():
            if not snap.is_dir():
                continue
            ts = parse_snap_ts(snap.name)
            if ts is None:
                continue
            f = snap / "crypto_ed_production.csv"
            if f.exists():
                entries.append((ts, f))
    # Add current file
    if PROD_CSV_PATH_CURRENT.exists():
        mt = datetime.fromtimestamp(PROD_CSV_PATH_CURRENT.stat().st_mtime, tz=timezone.utc)
        entries.append((mt, PROD_CSV_PATH_CURRENT))
    entries.sort(key=lambda x: x[0])
    return entries


def load_config_at(prod_index, signal_dt, asset, horizon):
    """Look up the config row matching (asset, horizon) in the prod-CSV snapshot
    that was LIVE at signal_dt (most recent snapshot taken AT OR BEFORE signal_dt).

    Falls back to the earliest snapshot if signal_dt is before any snapshot.
    """
    eligible = [(ts, p) for ts, p in prod_index if ts <= signal_dt]
    if eligible:
        ts, path = max(eligible, key=lambda x: x[0])
    elif prod_index:
        ts, path = prod_index[0]
    else:
        return None, None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, None
    match = df[(df["coin"] == asset) & (df["horizon"] == horizon)]
    if match.empty:
        return None, None
    cfg = match.iloc[0].to_dict()
    return cfg, ts


def load_config_from_csv(csv_path, asset, horizon):
    """Load a single config row from a specific CSV (used for A/B mode)."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    match = df[(df["coin"] == asset) & (df["horizon"] == horizon)]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


# ============================================================================
# Regime detection
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
# Trade simulator
# ============================================================================
def simulate_trades(signals_df, price_df, max_hold_h=MAX_HOLD_HOURS, fee_per_leg=TRADING_FEE):
    pdf = price_df.copy()
    pdf["datetime"] = pd.to_datetime(pdf["datetime"], errors="coerce")
    if pdf["datetime"].dt.tz is not None:
        pdf["datetime"] = pdf["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    pdf = pdf.set_index("datetime").sort_index()

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
        floor_hour = pd.Timestamp(dt).floor("h")
        for offset in range(0, 25):
            cand = floor_hour + pd.Timedelta(hours=offset)
            if cand in pdf.index:
                return float(pdf.loc[cand, "open"]), cand
        return None, None

    for _, row in sdf.iterrows():
        sig_dt = row["signal_dt"]
        action = str(row["action"]).upper()

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
        return {"label": label, "n": 0, "wr": None, "compound": None,
                "avg": None, "best": None, "worst": None, "ci_wr": (None,None),
                "ci_compound": (None,None), "ci_avg": (None,None)}
    wins = sum(1 for t in trades if t["win"])
    rets = [t["net_return"] for t in trades]
    total_net = 1.0
    for t in trades:
        total_net *= (1.0 + t["net_return"])
    total_net -= 1.0
    # Bootstrap CIs
    rng = np.random.default_rng(42)
    bs_wr, bs_c, bs_a = [], [], []
    for _ in range(5000):
        idx = rng.choice(len(rets), size=len(rets), replace=True)
        sample = np.array(rets)[idx]
        wins_s = np.array([t["win"] for t in trades])[idx]
        bs_wr.append(100 * wins_s.mean())
        bs_c.append((sample + 1).prod() - 1)
        bs_a.append(sample.mean())
    return {
        "label": label, "n": n,
        "wr": 100 * wins / n, "compound": total_net, "avg": np.mean(rets),
        "best": max(rets), "worst": min(rets),
        "ci_wr": (np.percentile(bs_wr, 2.5), np.percentile(bs_wr, 97.5)),
        "ci_compound": (np.percentile(bs_c, 2.5), np.percentile(bs_c, 97.5)),
        "ci_avg": (np.percentile(bs_a, 2.5), np.percentile(bs_a, 97.5)),
    }


def delta_significance(rets_a, rets_b, n_iter=5000):
    """Return P(B > A) bootstrap on compound return delta."""
    if len(rets_a) == 0 or len(rets_b) == 0:
        return None, None, None
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(n_iter):
        sa = rng.choice(rets_a, size=len(rets_a), replace=True)
        sb = rng.choice(rets_b, size=len(rets_b), replace=True)
        deltas.append(((sb + 1).prod() - 1) - ((sa + 1).prod() - 1))
    deltas = np.sort(deltas)
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    prob_b_wins = float((np.asarray(deltas) > 0).mean() * 100)
    return lo, hi, prob_b_wins


# ============================================================================
# Main inference loop — runs 3 conditions per hour
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-05-07T00:00")
    parser.add_argument("--end", default=None)
    parser.add_argument("--hourly-step", type=int, default=1)
    parser.add_argument("--asset", default="ETH")
    args = parser.parse_args()

    end_dt = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    if args.end:
        end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)

    print("=" * 78)
    print("  PIT-config + G_narrow/H75-fresh A/B counterfactual")
    print("=" * 78)
    print(f"  Asset:  {args.asset}")
    print(f"  Window: {start_dt.isoformat()} -> {end_dt.isoformat()}")
    print(f"  Step:   {args.hourly_step}h")
    print(f"  G_narrow promote: {G_NARROW_PROMOTE_UTC.isoformat()}")
    print("=" * 78)

    # Build oldest-wins archeology
    print("\n[1/6] Building oldest-wins merged archeology...")
    index = build_snapshot_index()
    merged_cache = {}
    for (filename, subdir), entries in index.items():
        if not entries:
            continue
        merged = build_merged_archeology(entries)
        if merged is not None:
            merged_cache[(filename, subdir)] = merged
            print(f"  {filename}: merged {len(merged)} rows from {len(entries)} snapshots")

    # Import engine
    print("\n[2/6] Importing engine + core...")
    import crypto_signal_core as csc
    import crypto_trading_system_ed as engine
    import crypto_live_trader_ed as lt
    print("  OK")

    # Build PIT prod-CSV index + load both A/B configs
    print("\n[3/6] Loading configs for the 3 conditions...")
    prod_index = build_prod_index()
    print(f"  prod-CSV PIT index: {len(prod_index)} entries")
    for ts, p in prod_index:
        print(f"    {ts.isoformat()}  <- {p.name}")

    # Load A/B configs (G_narrow current vs H75-fresh backup)
    cfg_gnarrow_5h = load_config_from_csv(G_NARROW_PROD_CSV, args.asset, 5)
    cfg_gnarrow_8h = load_config_from_csv(G_NARROW_PROD_CSV, args.asset, 8)
    cfg_h75fresh_5h = load_config_from_csv(H75_FRESH_PROD_CSV, args.asset, 5)
    cfg_h75fresh_8h = load_config_from_csv(H75_FRESH_PROD_CSV, args.asset, 8)

    if not all([cfg_gnarrow_5h, cfg_gnarrow_8h, cfg_h75fresh_5h, cfg_h75fresh_8h]):
        print("  ERROR — missing one or more config files for A/B comparison")
        return

    print(f"  G_narrow  5h: {cfg_gnarrow_5h['best_combo']} w={cfg_gnarrow_5h['best_window']} γ={cfg_gnarrow_5h['gamma']}")
    print(f"  G_narrow  8h: {cfg_gnarrow_8h['best_combo']} w={cfg_gnarrow_8h['best_window']} γ={cfg_gnarrow_8h['gamma']}")
    print(f"  H75-fresh 5h: {cfg_h75fresh_5h['best_combo']} w={cfg_h75fresh_5h['best_window']} γ={cfg_h75fresh_5h['gamma']}")
    print(f"  H75-fresh 8h: {cfg_h75fresh_8h['best_combo']} w={cfg_h75fresh_8h['best_window']} γ={cfg_h75fresh_8h['gamma']}")

    # Min-confidence threshold (we use the LIVE one — 65 for both regimes, current setup)
    bull_conf = 65
    bear_conf = 65

    # [4] Loop hourly — generate signals for all three conditions
    print(f"\n[4/6] Hourly inference for PIT + G_narrow + H75-fresh ({3} runs per hour)...")
    print(f"  Estimated time: {3 * 3 * (int((end_dt - start_dt).total_seconds() / 3600) // args.hourly_step) / 60:.0f} minutes")

    pit_rows, gnarrow_rows, h75fresh_rows = [], [], []
    cur = start_dt
    step = timedelta(hours=args.hourly_step)
    total = int((end_dt - start_dt) / step) + 1
    idx = 0
    while cur <= end_dt:
        idx += 1
        stage_merged_for(cur, merged_cache)

        # Regime
        price_path = WORKING_DATA_DIR / "eth_hourly_data.csv"
        regime = detect_regime_simple(price_path, cur) if price_path.exists() else "bull"
        h = 5 if regime == "bull" else 8
        min_conf = bull_conf if regime == "bull" else bear_conf

        # --- Condition 1: PIT config (whatever was live at cur) ---
        cfg_pit, cfg_ts = load_config_at(prod_index, cur, args.asset, h)
        # --- Condition 2: G_narrow ---
        cfg_g = cfg_gnarrow_5h if regime == "bull" else cfg_gnarrow_8h
        # --- Condition 3: H75-fresh ---
        cfg_h = cfg_h75fresh_5h if regime == "bull" else cfg_h75fresh_8h

        results = {}
        for name, cfg in [("pit", cfg_pit), ("gnarrow", cfg_g), ("h75fresh", cfg_h)]:
            if cfg is None:
                results[name] = {"error": "no_config", "horizon": h}
                continue
            results[name] = run_inference(args.asset, cfg, csc, engine, lt)

        def _row(name, res, cfg_used_ts=None):
            row = {
                "signal_dt": cur.replace(tzinfo=None), "regime": regime, "horizon": h,
                "raw_signal": None, "raw_conf": None, "action": "SKIP",
                "confidence": None, "error": None,
                "cfg_used_ts": cfg_used_ts.isoformat() if cfg_used_ts else None,
            }
            if res.get("error"):
                row["error"] = res["error"]
                return row
            raw_signal = res["signal"]
            raw_conf = res["confidence"]
            if raw_signal == "BUY" and raw_conf is not None and raw_conf >= min_conf:
                action = "BUY"
            elif raw_signal == "SELL" and raw_conf is not None and raw_conf >= min_conf:
                action = "SELL"
            else:
                action = "HOLD"
            row.update({"raw_signal": raw_signal, "raw_conf": raw_conf,
                        "action": action, "confidence": raw_conf})
            return row

        pit_rows.append(_row("pit", results["pit"], cfg_ts))
        gnarrow_rows.append(_row("gnarrow", results["gnarrow"]))
        h75fresh_rows.append(_row("h75fresh", results["h75fresh"]))

        if idx % 10 == 0 or idx == total:
            p_act = results["pit"].get("signal") or "ERR"
            g_act = results["gnarrow"].get("signal") or "ERR"
            h_act = results["h75fresh"].get("signal") or "ERR"
            print(f"  [{idx}/{total}] {cur.isoformat()} {regime[:4]} h={h}  "
                  f"PIT={p_act} G={g_act} H={h_act}")
        cur += step

    df_pit = pd.DataFrame(pit_rows)
    df_g = pd.DataFrame(gnarrow_rows)
    df_h = pd.DataFrame(h75fresh_rows)

    print(f"\n  PIT signals:      {len(df_pit)} hours  (BUY={(df_pit['action']=='BUY').sum()} SELL={(df_pit['action']=='SELL').sum()} HOLD={(df_pit['action']=='HOLD').sum()} SKIP={(df_pit['action']=='SKIP').sum()})")
    print(f"  G_narrow signals: {len(df_g)} hours    (BUY={(df_g['action']=='BUY').sum()} SELL={(df_g['action']=='SELL').sum()} HOLD={(df_g['action']=='HOLD').sum()} SKIP={(df_g['action']=='SKIP').sum()})")
    print(f"  H75-fresh signals:{len(df_h)} hours    (BUY={(df_h['action']=='BUY').sum()} SELL={(df_h['action']=='SELL').sum()} HOLD={(df_h['action']=='HOLD').sum()} SKIP={(df_h['action']=='SKIP').sum()})")

    # [5] Trade simulation
    print(f"\n[5/6] Simulating trades for all 4 conditions...")
    price_df = pd.read_csv(f"data/{args.asset.lower()}_hourly_data.csv")

    cf_pit_trades = simulate_trades(df_pit[df_pit['action'].isin(['BUY','SELL','HOLD'])], price_df)
    cf_g_trades   = simulate_trades(df_g[df_g['action'].isin(['BUY','SELL','HOLD'])], price_df)
    cf_h_trades   = simulate_trades(df_h[df_h['action'].isin(['BUY','SELL','HOLD'])], price_df)

    actual = pd.read_csv("config/signal_log.csv")
    actual["signal_dt"] = pd.to_datetime(actual["timestamp"], errors="coerce")
    actual = actual[(actual["asset"] == args.asset)
                    & (actual["signal_dt"] >= pd.Timestamp(start_dt.replace(tzinfo=None)))
                    & (actual["signal_dt"] <= pd.Timestamp(end_dt.replace(tzinfo=None)))].copy()
    actual_trades = simulate_trades(actual, price_df)

    # Outputs
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO_ROOT / "output"
    out.mkdir(exist_ok=True)
    df_pit.to_csv(out / f"cf_ab_signals_pit_{ts_str}.csv", index=False)
    df_g.to_csv(out / f"cf_ab_signals_gnarrow_{ts_str}.csv", index=False)
    df_h.to_csv(out / f"cf_ab_signals_h75fresh_{ts_str}.csv", index=False)
    if cf_pit_trades: pd.DataFrame(cf_pit_trades).to_csv(out / f"cf_ab_trades_pit_{ts_str}.csv", index=False)
    if cf_g_trades: pd.DataFrame(cf_g_trades).to_csv(out / f"cf_ab_trades_gnarrow_{ts_str}.csv", index=False)
    if cf_h_trades: pd.DataFrame(cf_h_trades).to_csv(out / f"cf_ab_trades_h75fresh_{ts_str}.csv", index=False)
    if actual_trades: pd.DataFrame(actual_trades).to_csv(out / f"cf_ab_trades_actual_{ts_str}.csv", index=False)

    # [6] Analysis
    print(f"\n[6/6] Analysis (bootstrap 5000 iters)...")
    sums = {
        "ACTUAL (broken-cache signal_log)": trade_summary(actual_trades, "ACTUAL"),
        "PIT (live-config-at-time + fresh data)": trade_summary(cf_pit_trades, "PIT"),
        "G_narrow throughout + fresh data": trade_summary(cf_g_trades, "G_narrow"),
        "H75-fresh throughout + fresh data": trade_summary(cf_h_trades, "H75-fresh"),
    }

    print("\n" + "=" * 78)
    print("  HEAD-TO-HEAD RESULTS  (bootstrap 95% CIs)")
    print("=" * 78)

    for label, s in sums.items():
        print(f"\n  {label}:")
        if s["n"] == 0:
            print(f"    no trades"); continue
        print(f"    n_trades:     {s['n']}")
        print(f"    WR:           {s['wr']:.1f}%   95% CI [{s['ci_wr'][0]:.1f}%, {s['ci_wr'][1]:.1f}%]")
        print(f"    Compound ret: {s['compound']*100:+.2f}%  95% CI [{s['ci_compound'][0]*100:+.2f}%, {s['ci_compound'][1]*100:+.2f}%]")
        print(f"    Avg/trade:    {s['avg']*100:+.3f}% 95% CI [{s['ci_avg'][0]*100:+.3f}%, {s['ci_avg'][1]*100:+.3f}%]")

    # Pairwise probabilities — focus on the questions that matter
    print("\n" + "=" * 78)
    print("  PAIRWISE PROBABILITY OF OUTPERFORMANCE  (bootstrap)")
    print("=" * 78)

    rets_actual = [t["net_return"] for t in actual_trades]
    rets_pit    = [t["net_return"] for t in cf_pit_trades]
    rets_g      = [t["net_return"] for t in cf_g_trades]
    rets_h      = [t["net_return"] for t in cf_h_trades]

    pairs = [
        ("PIT vs ACTUAL", rets_actual, rets_pit, "Does cache fix help when keeping the live config history fixed?"),
        ("G_narrow vs H75-fresh", rets_h, rets_g, "Is current G_narrow promotion justified vs prior H75-fresh?"),
        ("H75-fresh vs G_narrow", rets_g, rets_h, "(same comparison, other direction)"),
        ("H75-fresh vs ACTUAL", rets_actual, rets_h, "Would running H75-fresh all along (with fresh data) have beaten the actual broken-cache trades?"),
    ]
    for name, rets_a, rets_b, question in pairs:
        lo, hi, p_b = delta_significance(rets_a, rets_b)
        print(f"\n  {name}:")
        print(f"    {question}")
        if lo is None:
            print(f"    (insufficient data)"); continue
        print(f"    Return delta CI: [{lo*100:+.2f}pp, {hi*100:+.2f}pp]")
        print(f"    P(2nd outperforms 1st): {p_b:.1f}%")

    # Markdown summary
    md_path = out / f"cf_ab_summary_{ts_str}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# A/B + PIT counterfactual — {ts_str}\n\n")
        f.write(f"Window: `{start_dt.isoformat()}` → `{end_dt.isoformat()}`\n\n")
        f.write(f"## Headline results\n\n| Condition | n | WR | Compound | Avg/trade |\n|---|---|---|---|---|\n")
        for label, s in sums.items():
            if s["n"] == 0:
                f.write(f"| {label} | 0 | — | — | — |\n"); continue
            f.write(f"| {label} | {s['n']} | {s['wr']:.1f}% [{s['ci_wr'][0]:.1f}-{s['ci_wr'][1]:.1f}] | "
                    f"{s['compound']*100:+.2f}% [{s['ci_compound'][0]*100:+.2f}-{s['ci_compound'][1]*100:+.2f}] | "
                    f"{s['avg']*100:+.3f}% [{s['ci_avg'][0]*100:+.3f}-{s['ci_avg'][1]*100:+.3f}] |\n")
        f.write(f"\n## Pairwise probabilities\n\n")
        for name, rets_a, rets_b, question in pairs:
            lo, hi, p_b = delta_significance(rets_a, rets_b)
            f.write(f"### {name}\n*{question}*\n\n")
            if lo is None:
                f.write("Insufficient data\n\n"); continue
            f.write(f"- Return delta 95% CI: **[{lo*100:+.2f}pp, {hi*100:+.2f}pp]**\n")
            f.write(f"- P(2nd outperforms 1st): **{p_b:.1f}%**\n\n")

    print(f"\n  Markdown summary: {md_path}")
    print(f"  Paste back to assistant.")
    print("\n" + "=" * 78)
    print("  DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
