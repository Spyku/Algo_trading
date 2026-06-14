"""
hrst_ablate_xa_fg_vix_58h.py — full FAYE HRST on ETH 5,8h with the cross_asset + sentiment
RAW feature families removed (xa_ / fg_ / vix_), but KEPT inside PySR. *** RUN ON THE DESKTOP ***

WHAT THIS TESTS
  Whether dropping cross_asset + sentiment as RAW model inputs changes the production 5h/8h
  regime — vs the HRST you promoted 2026-06-14 (live: detector tsmom_672h, bull=8h@80 / bear=5h@70).
  Live 5h-bear actually selected a RAW cross_asset feature (xa_btc_lag1h); 8h-bull carries
  cross_asset only via pysr_1 (kept). So this measures "are the RAW families redundant given pysr".

DELIBERATE Trim A (your call): xa_/fg_/vix_ removed only as RAW features; the pysr_* columns that
  embed them are UNTOUCHED. NOTE m_vix_* is MACRO (prefix m_) and is NOT removed.

NO RISK ON PRODUCTION — triple-isolated:
  1) mock engine  crypto_trading_system_faye_mock.py  (never the live engine)
  2) isolated output dirs  models_faye_ablate58/ + config_faye_ablate58/  (the mock's production
     paths DERIVE from FAYE_MODELS_DIR / FAYE_CONFIG_DIR, so they can't be models/ or config/)
  3) --no-persist  (redirects any production write to *_noprod within the isolated dirs)
  Data is a FROZEN snapshot of current data/ (drift-proof across the multi-hour run) + --no-data-update.

RUN (desktop, venv active, from the engine root):
  python tools/hrst_ablate_xa_fg_vix_58h.py

OUTPUT
  log:     tools/ablate58_hrst.log
  models:  models_faye_mock_ablate58/crypto_faye_production.csv   (the families-off 5h/8h models)
  regime:  config_faye_mock_ablate58/regime_config_faye*.json     (the families-off bull/bear pick)
  Compare against the production HRST that produced live: logs/ed_v1_20260613_214253.log
  Expected runtime: ~3-5h on the Desktop (2 horizons, --replay 1440, full H->R->S->T->G).
"""
import os
os.environ.setdefault('_FAYE_WARNINGS_BAKED', '1')   # Windows: skip the os.execv re-exec (else the subprocess detaches)
import sys
import re
import json
import glob
import time
import shutil
import datetime
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
try:                                                  # UTF-8-safe console (Windows cp1252 would crash on the table)
    sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

MOCK       = 'crypto_trading_system_faye_mock.py'     # THE canonical mock (never cp a fresh one)
HORIZONS   = '5,8h'                                    # the live regime pair (bull=8h, bear=5h)
REPLAY     = '1440'                                    # match the production HRST (2 months)
PY         = sys.executable

MODELS_DIR = 'models_faye_mock_ablate58'              # isolated -> mock PRODUCTION_CSV derives here ('mock' REQUIRED by guard)
CONFIG_DIR = 'config_faye_mock_ablate58'              # isolated -> mock REGIME_CONFIG_PATH derives here ('mock' REQUIRED by guard)
LOG        = 'tools/ablate58_hrst.log'
PROD_LOG   = 'logs/ed_v1_20260613_214253.log'         # the HRST that produced the live 06-14 config
PYSR_SRC   = 'models_faye'                             # what the production HRST actually read its pysr from

DISABLED_PREFIXES = ['xa_', 'fg_', 'vix_']            # cross_asset + sentiment; m_vix_* (macro) is NOT matched


def _fail(msg):
    print(f"\n[ABORT] {msg}", flush=True)
    sys.exit(1)


def make_snapshot():
    """Freeze current data/ into an isolated copy so the trader / Drive sync can't drift the data
    mid-run. Same file-set the engine reads: top-level *_hourly_data.csv + the whole macro_data/ dir."""
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    snap = os.path.join('data', f'_v2_snapshot_ablate58_{ts}')
    os.makedirs(os.path.join(snap, 'macro_data'), exist_ok=True)
    n = 0
    for f in glob.glob(os.path.join('data', '*_hourly_data.csv')):
        shutil.copy2(f, os.path.join(snap, os.path.basename(f))); n += 1
    md = os.path.join('data', 'macro_data')
    if os.path.isdir(md):
        for f in glob.glob(os.path.join(md, '*')):
            if os.path.isfile(f):
                shutil.copy2(f, os.path.join(snap, 'macro_data', os.path.basename(f))); n += 1
    if n < 5:
        _fail(f"snapshot only captured {n} files from data/ — expected ~40. Are you in the engine root with data/?")
    print(f"[snapshot] froze {n} data files -> {snap}", flush=True)
    return snap


def write_disabled_cfg():
    """Remove the 2 RAW families; keep production's trim state (OFF) and its 7 always-off sparse feats."""
    always = json.load(open('config/disabled_features.json')).get('always_disabled_exact', [])
    cfg = {
        '_note': 'ABLATION ETH 5,8h — remove RAW xa_/fg_/vix_ (cross_asset+sentiment); pysr KEEPS them. '
                 'Trim OFF (disabled_exact empty == production).',
        'enabled': True,            # activate disabled_prefixes ...
        'disabled_exact': [],       # ... but Grade-1 trim stays OFF (production has it OFF; trim-ON costs -21.4pp)
        'disabled_prefixes': DISABLED_PREFIXES,
        'always_disabled_exact': always,
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    p = os.path.join(MODELS_DIR, 'disabled_features_ablate58.json')
    json.dump(cfg, open(p, 'w'), indent=2)
    return p, len(always)


def seed_pysr():
    """Copy the SAME pysr formulas the production HRST used into the isolated models dir, so the
    families' pysr-encoded signal is IDENTICAL to production (we keep pysr; only RAW is removed)."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    seeded = []
    for h in ('5', '8'):
        for src_dir in (PYSR_SRC, 'models'):
            src = os.path.join(src_dir, f'pysr_ETH_{h}h.json')
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(MODELS_DIR, f'pysr_ETH_{h}h.json'))
                seeded.append(h); break
    return seeded


def last_baseline_ref(path):
    """Pull the no-gate regime baseline 'baselines V0 ... REF=<x>%' — a stable, comparable headline number."""
    try:
        txt = open(path, encoding='utf-8', errors='ignore').read()
        m = re.findall(r'baselines V0.*?REF=([+\-]?\d+\.\d+)%', txt)
        return m[-1] if m else None
    except Exception:
        return None


def main():
    print("=" * 80)
    print("  FAYE HRST ABLATION — ETH 5,8h — remove RAW xa_/fg_/vix_ (kept in pysr)  [NOT production]")
    print("=" * 80, flush=True)

    if not os.path.exists(MOCK):
        _fail(f"{MOCK} not found in {HERE}. Run from the engine root on the Desktop.")

    snap = make_snapshot()
    cfgp, n_always = write_disabled_cfg()
    seeded = seed_pysr()
    print(f"[setup] remove RAW prefixes={DISABLED_PREFIXES} | trim OFF | {n_always} sparse always-off | "
          f"pysr seeded for {seeded}", flush=True)
    if set(seeded) != {'5', '8'}:
        _fail("pysr NOT seeded for both 5h & 8h — without it the families wouldn't be in pysr either "
              "(would silently become a different test). Check models_faye/pysr_ETH_{5,8}h.json.")

    env = dict(os.environ)
    env['_FAYE_WARNINGS_BAKED'] = '1'
    env['PYTHONWARNINGS']       = 'ignore'
    env['FAYE_MODELS_DIR']      = MODELS_DIR
    env['FAYE_CONFIG_DIR']      = CONFIG_DIR
    env['MOCK_DISABLED_PATH']   = cfgp
    env['V2_DATA_SNAPSHOT']     = snap

    cmd = [PY, MOCK, 'HRST', 'ETH', HORIZONS, '--replay', REPLAY, '--no-persist', '--no-data-update']
    print(f"[run] {' '.join(cmd)}")
    print(f"[run] FAYE_MODELS_DIR={MODELS_DIR}  FAYE_CONFIG_DIR={CONFIG_DIR}")
    print(f"[run] V2_DATA_SNAPSHOT={snap}")
    print(f"[run] tee -> {LOG}   (~3-5h on Desktop — leave it running)\n", flush=True)

    t0 = time.time()
    with open(LOG, 'w', encoding='utf-8') as lf:
        r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    mins = (time.time() - t0) / 60.0
    print(f"[done] rc={r.returncode} in {mins:.0f} min   (full log: {LOG})", flush=True)

    # ---- surface the ablated result + the production reference (regime is reliable; returns: verify in logs) ----
    print("\n" + "=" * 80)
    print("  RESULT — ablated (RAW xa/fg/vix removed, pysr kept)  vs  PRODUCTION (live, 2026-06-14)")
    print("=" * 80)
    cfgs = glob.glob(os.path.join(CONFIG_DIR, 'regime_config*.json'))
    if cfgs:
        eth = json.load(open(max(cfgs, key=os.path.getmtime))).get('ETH', {})
        det = (eth.get('regime_detector') or {}).get('params', {}).get('name', '?')
        b, r2 = eth.get('bull', {}), eth.get('bear', {})
        print(f"  ABLATED     detector={det}  bull={b.get('horizon')}h@{b.get('min_confidence')}  "
              f"bear={r2.get('horizon')}h@{r2.get('min_confidence')}")
    else:
        print(f"  ABLATED     [no regime_config in {CONFIG_DIR}/ — read {LOG} for the Mode T result]")
    print(f"  PRODUCTION  detector=tsmom_672h  bull=8h@80  bear=5h@70   (live, promoted 2026-06-14 13:11)")

    abl, prod = last_baseline_ref(LOG), last_baseline_ref(PROD_LOG)
    print(f"\n  regime baseline (no-gate, V0) Mode T REF:")
    print(f"    ablated    = {abl if abl else 'n/a — read the log'} %")
    print(f"    production = {prod if prod else 'n/a — read ' + PROD_LOG} %   ({PROD_LOG})")
    if abl and prod:
        print(f"    delta      = {float(abl) - float(prod):+.2f} pp   (RAW families removed vs kept)")
    print("    (headline baseline only — for the GATED winner read the Mode T / STRICT table in each log)")

    print("\n  ablated 5h/8h models: " + os.path.join(MODELS_DIR, 'crypto_faye_production.csv'))
    print(f"  ISOLATION: wrote ONLY to {MODELS_DIR}/ + {CONFIG_DIR}/ — live models/ + config/ untouched.")
    print("=" * 80)


if __name__ == '__main__':
    main()
