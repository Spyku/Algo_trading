#!/usr/bin/env python
"""
_build_fujiwara.py — deterministically derive the two sub-hourly research forks
(Fujiwara_15, Fujiwara_30) from crypto_trading_system_faye.py.

Chinese-wall transform: applies a fixed set of TARGETED edits (each asserted to
match exactly once) + GLOBAL filename/dir token swaps, so the fork:
  * writes only to models_fujiwara_{15,30}/ + config_fujiwara_{15,30}/,
  * reads price data from data/{asset}_{15m,30m}_data.csv,
  * downloads {15m,30m} candles,
  * CANNOT promote to live production (--promote disabled),
  * never reads production config/CSV.
build_all_features (incl. the daily/onchain availability lag and the hourly
floor('h') merges) is left BYTE-IDENTICAL to faye — lag inherited unchanged.

Run:  python tools/_build_fujiwara.py
"""
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'crypto_trading_system_faye.py')

def _rep(text, old, new, label, n=1):
    c = text.count(old)
    assert c == n, f"[{label}] expected {n} match(es), found {c}"
    return text.replace(old, new)

def build(MIN, TF):
    TAG = TF                                    # '15m' / '30m'
    PPH = 60 // MIN                             # periods per hour
    MODELS = f'models_fujiwara_{MIN}'
    CONFIG = f'config_fujiwara_{MIN}'
    PROD = f'crypto_fujiwara_{MIN}_production.csv'
    CFG = f'regime_config_fujiwara_{MIN}.json'
    NAME = f'FUJIWARA_{MIN}'
    ENVM = f'FUJI{MIN}_MODELS_DIR'
    ENVC = f'FUJI{MIN}_CONFIG_DIR'
    mins = ", ".join(str(p * MIN) for p in (5, 6, 7, 8))

    t = open(SRC, encoding='utf-8').read()

    # ── 1. Header docstring ──
    old_hdr = '''"""
FAYE — ML Trading System (next-generation Ed, 2026-05-29 / 2026-05-30)
============================================================
Standalone single-file evolution of the Ed engine with NEAR_LIVE_MODE
behavior as the new default. Built by inlining the v3 architecture
(parallel_nearlive + step6_nearlive + signal_core_nearlive + g_narrow_d)
into one file with ZERO monkey-patches.

ML trading system for BTC, ETH, XRP, SOL, LINK (5 assets post-2026-04-19 prune).
130+ features -> walk-forward ML -> BUY/SELL/HOLD signals (ternary by default).
Variable horizon per asset (5h, 6h, 7h, 8h, etc.) with regime-switching.'''
    new_hdr = f'''"""
{NAME} — {MIN}-MINUTE-candle ML Trading System (research fork of FAYE, 2026-06-19)
====================================================================================
Self-contained {MIN}-minute-candle fork of crypto_trading_system_faye.py. Built to
re-run the early-project "does a sub-hourly trader work?" test on the modern FAYE
engine. Standalone single file, ZERO monkey-patches (mirrors faye's philosophy).

CHINESE WALL — NO INFECTION OF PRODUCTION. Research-only, physically walled:
  * Output -> {MODELS}/ + {CONFIG}/ (never models/ or config/).
  * Price data -> data/<asset>_{TAG}_data.csv (never *_hourly_data.csv).
  * --promote is DISABLED (returns early) — can NEVER write live production.
  * Never reads production config/CSV (config seed + Mode C redirected to fork).
  * Not imported by the live trader (live imports faye only).
Shared READ-ONLY inputs: data/macro_data/* — reading shared inputs is not
infection; the wall forbids WRITES to production paths.

HORIZONS ARE IN CANDLE (PERIOD) UNITS: 5,6,7,8 periods = {mins} minutes.
The daily/on-chain availability LAG is inherited UNCHANGED from faye
(DAILY_MERGE_LAG_DAYS / ONCHAIN_MERGE_LAG_DAYS in *days*, broadcast across all
intraday bars; hourly-cadence sources keep floor('h')). build_all_features is
byte-identical to faye — lag everywhere, identical discipline.

Original FAYE header (engine genealogy) follows:
---
FAYE — ML Trading System (next-generation Ed, 2026-05-29 / 2026-05-30)
============================================================
Standalone single-file evolution of the Ed engine with NEAR_LIVE_MODE
behavior as the new default. Built by inlining the v3 architecture
(parallel_nearlive + step6_nearlive + signal_core_nearlive + g_narrow_d)
into one file with ZERO monkey-patches.

ML trading system for BTC, ETH, XRP, SOL, LINK (5 assets post-2026-04-19 prune).
130+ features -> walk-forward ML -> BUY/SELL/HOLD signals (ternary by default).
Variable horizon per asset (in CANDLE units) with regime-switching.'''
    t = _rep(t, old_hdr, new_hdr, 'header')

    # ── 2. Output dirs (env keys + defaults -> fork; var NAMES kept) ──
    old_dir = """FAYE_MODELS_DIR = _resolve_dir('FAYE_MODELS_DIR', 'models_faye')
FAYE_CONFIG_DIR = _resolve_dir('FAYE_CONFIG_DIR', 'config_faye')
_FAYE_MODELS_RAW = os.environ.get('FAYE_MODELS_DIR', 'models_faye')
_FAYE_CONFIG_RAW = os.environ.get('FAYE_CONFIG_DIR', 'config_faye')"""
    new_dir = f"""FAYE_MODELS_DIR = _resolve_dir('{ENVM}', '{MODELS}')
FAYE_CONFIG_DIR = _resolve_dir('{ENVC}', '{CONFIG}')
_FAYE_MODELS_RAW = os.environ.get('{ENVM}', '{MODELS}')
_FAYE_CONFIG_RAW = os.environ.get('{ENVC}', '{CONFIG}')"""
    t = _rep(t, old_dir, new_dir, 'dir-block')

    # ── 3. Config seed source: NEVER read production; seed from fork template ──
    t = _rep(t,
             "    _faye_live_regime_cfg = os.path.join(_SCRIPT_DIR, 'config', 'regime_config_ed.json')",
             f"    _faye_live_regime_cfg = os.path.join(FAYE_CONFIG_DIR, 'regime_config_fujiwara_{MIN}_template.json')  # WALL: never read production",
             'seed-src')

    # ── 4. Resume dir ──
    t = _rep(t, "RESUME_DIR = f'{FAYE_MODELS_DIR}/.resume_hourly'",
             f"RESUME_DIR = f'{{FAYE_MODELS_DIR}}/.resume_{TAG}'", 'resume')

    # ── 5. Candle constants + _ASSET_DEFS (sub-hourly filenames) ──
    old_defs = """_ASSET_DEFS = [
    # 2026-05-02: BNB added after Revolut X listing verified;
    # DOGE/ADA/AVAX/DOT remain pruned (weak priors, no diversification).
    ('BTC',   'binance',  'BTC/USDT',  'btc_hourly_data.csv',  '2017-08-01T00:00:00Z'),
    ('ETH',   'binance',  'ETH/USDT',  'eth_hourly_data.csv',  '2017-08-01T00:00:00Z'),
    ('XRP',   'binance',  'XRP/USDT',  'xrp_hourly_data.csv',  '2018-05-01T00:00:00Z'),
    ('SOL',   'binance',  'SOL/USDT',  'sol_hourly_data.csv',  '2020-08-01T00:00:00Z'),
    ('LINK',  'binance',  'LINK/USDT', 'link_hourly_data.csv', '2019-01-01T00:00:00Z'),
    ('BNB',   'binance',  'BNB/USDT',  'bnb_hourly_data.csv',  '2017-12-01T00:00:00Z'),
    ('SMI',   'yfinance', '^SSMI',     'smi_hourly_data.csv',  None),
    ('DAX',   'yfinance', '^GDAXI',    'dax_hourly_data.csv',  None),
    ('CAC40', 'yfinance', '^FCHI',     'cac40_hourly_data.csv', None),
]"""
    new_defs = f"""# ── Candle granularity ({NAME} fork) ──────────────────────────────────────────
CANDLE_MINUTES = {MIN}
CANDLE_TIMEFRAME = '{TF}'        # ccxt timeframe for download_binance
CANDLE_TAG = '{TAG}'             # data-file tag
PERIODS_PER_HOUR = 60 // CANDLE_MINUTES   # {PPH}
# Horizons, windows, embargo, gamma are all in CANDLE (period) units — faye is
# candle-agnostic. Daily/on-chain merge lag stays in DAYS (broadcasts to intraday
# bars); hourly-cadence sources keep floor('h'). Crypto starts pinned to 2024-01-01
# to keep sub-hourly downloads manageable (only ETH is used in this first test).

_ASSET_DEFS = [
    ('BTC',   'binance',  'BTC/USDT',  'btc_{TAG}_data.csv',  '2024-01-01T00:00:00Z'),
    ('ETH',   'binance',  'ETH/USDT',  'eth_{TAG}_data.csv',  '2024-01-01T00:00:00Z'),
    ('XRP',   'binance',  'XRP/USDT',  'xrp_{TAG}_data.csv',  '2024-01-01T00:00:00Z'),
    ('SOL',   'binance',  'SOL/USDT',  'sol_{TAG}_data.csv',  '2024-01-01T00:00:00Z'),
    ('LINK',  'binance',  'LINK/USDT', 'link_{TAG}_data.csv', '2024-01-01T00:00:00Z'),
    ('BNB',   'binance',  'BNB/USDT',  'bnb_{TAG}_data.csv',  '2024-01-01T00:00:00Z'),
    ('SMI',   'yfinance', '^SSMI',     'smi_hourly_data.csv',  None),
    ('DAX',   'yfinance', '^GDAXI',    'dax_hourly_data.csv',  None),
    ('CAC40', 'yfinance', '^FCHI',     'cac40_hourly_data.csv', None),
]"""
    t = _rep(t, old_defs, new_defs, 'asset-defs')

    # ── 6. Download timeframe ──
    t = _rep(t, "    timeframe = '1h'", "    timeframe = CANDLE_TIMEFRAME", 'timeframe')

    # ── 7. Window/cap constants: KEEP BASE PERIOD COUNTS (do NOT scale by PERIODS_PER_HOUR).
    # The engine's caps are period counts (the "maximum"), not a fixed calendar. Keeping them
    # at base means the same COMPUTE per fit + the max training window = 4320 periods (= 3
    # months @30m, 1.5mo @15m) — fast and "linked to the maximum", per the owner. Earlier I
    # PPH-scaled these to preserve 6-month *calendar* -> 2-4x bigger windows + slow. Reverted.
    # (MODE_G_REPLAY_HOURS / MAX_DIAG fallback left at faye base, untouched.)

    # ── 8. Do NOT create production models/ + config/ dirs (wall) ──
    t = _rep(t, """    os.path.join(_SCRIPT_DIR, 'models'),
    os.path.join(_SCRIPT_DIR, 'config'),""",
             "    # (production models/ + config/ intentionally NOT created — Chinese wall)",
             'no-prod-mkdir')

    # ── 9. --promote DISABLED ──
    old_promote = """    if '--promote' in sys.argv:
        import shutil
        _faye_csv = os.path.join(FAYE_MODELS_DIR, 'crypto_faye_production.csv')
        _faye_cfg = os.path.join(FAYE_CONFIG_DIR, 'regime_config_faye.json')
        _np_csv = _faye_csv.replace('.csv', '_noprod.csv')
        _np_cfg = _faye_cfg.replace('.json', '_noprod.json')
        _trader_csv = os.path.join(_SCRIPT_DIR, 'models', 'crypto_ed_production.csv')
        _trader_cfg = os.path.join(_SCRIPT_DIR, 'config', 'regime_config_ed.json')"""
    assert t.count(old_promote) == 1, "promote-head not found uniquely"
    # Replace from the if-head through its terminating `return` (the WARNING block).
    head_idx = t.index("    if '--promote' in sys.argv:")
    ret_marker = "        print(\"  *** was FLAT (no open position) before promoting — a mid-trade swap mismatches\")"
    assert t.count(ret_marker) == 1, "promote-tail marker missing"
    tail_idx = t.index(ret_marker)
    end_idx = t.index("        return", tail_idx) + len("        return")
    new_promote = f"""    if '--promote' in sys.argv:
        print("\\n  [{NAME}] --promote is DISABLED on this research fork (Chinese wall).")
        print("  This fork can NEVER write live production (models/ or config/).")
        print("  Results live in {MODELS}/ + {CONFIG}/ only.")
        return"""
    t = t[:head_idx] + new_promote + t[end_idx:]

    # ── 9b. disabled_features.json -> fork config dir (Mode F WRITES it; prod reads it) ──
    t = _rep(t, "    cfg_path = os.path.join(here, 'config', 'disabled_features.json')",
             "    cfg_path = os.path.join(FAYE_CONFIG_DIR, 'disabled_features.json')  # WALL: fork-local, never the shared production file",
             'disabled-feat-write')
    t = _rep(t, "    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'disabled_features.json')",
             "    path = os.path.join(FAYE_CONFIG_DIR, 'disabled_features.json')  # WALL: fork-local",
             'disabled-feat-read')

    # ── 10. Mode C incumbent -> fork paths (no production read) ──
    t = _rep(t,
             "    inc_cfg, inc_csv = 'config/regime_config_ed.json', 'models/crypto_ed_production.csv'",
             f"    inc_cfg, inc_csv = f'{{FAYE_CONFIG_DIR}}/{CFG}', f'{{FAYE_MODELS_DIR}}/{PROD}'  # WALL: fork incumbent, not production",
             'mode-c-inc')

    # ── 11. Mode P: drive the FORK-LOCAL PySR discovery (pysr_discover_features_fujiwara),
    #        on the fork's sub-hourly data, with candle-scaled window + timeframe-distinct
    #        filenames (pysr_ETH_5p_15m.json / _30m). Shared pysr_discover_features.py untouched. ──
    t = _rep(t, "        import pysr_discover_features as _pysr_mod",
             "        import pysr_discover_features_fujiwara as _pysr_mod", 'pysr-import-worker')
    t = _rep(t, "    import pysr_discover_features as pysr_mod",
             "    import pysr_discover_features_fujiwara as pysr_mod", 'pysr-import-parallel')
    t = _rep(t, "        from pysr_discover_features import save_results",
             "        from pysr_discover_features_fujiwara import save_results", 'pysr-import-save')

    # _discover_features_parallel: add max_diag_hours param + thread it through
    t = _rep(t,
             "                                load_data_fn=None, build_features_fn=None,\n"
             "                                horizon_suffix='h'):",
             "                                load_data_fn=None, build_features_fn=None,\n"
             "                                horizon_suffix='h', max_diag_hours=None):",
             'parallel-sig')
    t = _rep(t,
             "    X, y, all_cols, pysr_rows = pysr_mod._prepare_data(\n"
             "        asset, horizon,\n"
             "        load_data_fn=load_data_fn,\n"
             "        build_features_fn=build_features_fn\n"
             "    )",
             "    X, y, all_cols, pysr_rows = pysr_mod._prepare_data(\n"
             "        asset, horizon,\n"
             "        load_data_fn=load_data_fn,\n"
             "        build_features_fn=build_features_fn,\n"
             "        max_diag_hours=max_diag_hours\n"
             "    )",
             'parallel-prepare-call')
    t = _rep(t,
             "            load_data_fn=load_data_fn,\n"
             "            build_features_fn=build_features_fn,\n"
             "            horizon_suffix=horizon_suffix,\n"
             "        )",
             "            load_data_fn=load_data_fn,\n"
             "            build_features_fn=build_features_fn,\n"
             "            horizon_suffix=horizon_suffix,\n"
             "            max_diag_hours=max_diag_hours,\n"
             "        )",
             'parallel-fallback-call')

    # run_mode_p: inject fork's loaders, distinct name, candle-scaled 6mo window, fork out_dir
    t = _rep(t,
             "            results, pysr_rows = _discover_features_parallel(asset, h)",
             "            results, pysr_rows = _discover_features_parallel(\n"
             "                asset, h, load_data_fn=load_data, build_features_fn=build_all_features,\n"
             "                horizon_suffix=f'p_{CANDLE_TAG}', max_diag_hours=6 * 30 * 24)",
             'runmodep-discover')
    t = _rep(t,
             "                save_results(asset, h, results, all_cols, pysr_rows=pysr_rows)",
             "                save_results(asset, h, results, all_cols, pysr_rows=pysr_rows,\n"
             "                             out_dir=FAYE_MODELS_DIR, horizon_suffix=f'p_{CANDLE_TAG}')",
             'runmodep-save')
    t = _rep(t,
             '                print(f"  python crypto_trading_system_ed.py DV {asset} {h}h")',
             '                print(f"  python crypto_trading_system_fujiwara_{CANDLE_MINUTES}.py DV {asset} {h}h")',
             'runmodep-msg')

    # PySR COLUMN labels: timeframe-tag the feature names so they can't be mixed up
    # across engines (pysr_1_15 / pysr_2_15 on 15m; pysr_1_30 on 30m; hourly stays pysr_1).
    # startswith('pysr_') still matches, so floor/exclusion/grade logic is unaffected.
    t = _rep(t, "        col_name = f'pysr_{i+1}'",
             "        col_name = f'pysr_{i+1}_{CANDLE_MINUTES}'", 'pysr-col-label')

    # PySR READ sites: load the timeframe-distinct name (pysr_<asset>_<h>p_<tag>.json)
    t = _rep(t, "    pysr_path = os.path.join(models_dir, f'pysr_{asset_name}_{horizon}h.json')",
             "    pysr_path = os.path.join(models_dir, f'pysr_{asset_name}_{horizon}p_{CANDLE_TAG}.json')",
             'pysr-read-1')
    t = _rep(t, "    pysr_path = os.path.join(models_dir, f'pysr_{asset}_{horizon}h.json')",
             "    pysr_path = os.path.join(models_dir, f'pysr_{asset}_{horizon}p_{CANDLE_TAG}.json')",
             'pysr-read-2')

    # ── GLOBAL token swaps (filenames + dir literals everywhere else) ──
    t = t.replace('crypto_faye_production.csv', PROD)
    t = t.replace('regime_config_faye.json', CFG)
    t = t.replace('models_faye', MODELS)
    t = t.replace('config_faye', CONFIG)

    out = os.path.join(REPO, f'crypto_trading_system_fujiwara_{MIN}.py')
    open(out, 'w', encoding='utf-8').write(t)

    # ── post-build wall verification ──
    leaks = []
    for bad in ('crypto_ed_production.csv', "regime_config_ed.json",
                "'models'", "'config'", 'models_faye', 'config_faye',
                'crypto_faye_production.csv', 'regime_config_faye.json'):
        # allow the two production strings ONLY where they no longer write:
        if bad in t:
            # count occurrences for the report
            leaks.append((bad, t.count(bad)))
    print(f"  built {out}  ({len(t):,} chars)")
    return out, leaks

if __name__ == '__main__':
    for MIN, TF in ((15, '15m'), (30, '30m')):
        out, leaks = build(MIN, TF)
        if leaks:
            print(f"    residual production/faye tokens (verify each is harmless):")
            for s, c in leaks:
                print(f"      {s!r}: {c}")
    print("done.")
