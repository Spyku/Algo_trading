"""2026-06-15 — DESKTOP QUERY: re-select ETH regime detector (5h/8h pair), FULL menu.

Runs Mode R -> S -> T over a window THROUGH TODAY (incl. the 06-14/15 rally),
letting Mode S choose the best detector + confidences from the COMPLETE detector
menu (~45): SMA/EMA crosses, price-vs-SMA, momentum/MACD, RSI, drawdown, bounce,
volatility, ADX, Hurst, and 2-family combinations.

SAFE / ISOLATED:
  * Injects every detector IN-PROCESS via monkeypatch — edits NO shared engine
    file (faye.py/ed.py untouched on disk; a running BTC job is safe).
  * All indicators computed in-script (Wilder ADX, R/S Hurst, SMAs/EMAs/RSI/DD/
    bounce/MACD/tsmom) and keyed to the same naive timestamps the engine uses.
    vol_calm is taken from the engine (deseasonalized vol).
  * Writes ONLY to config_faye_detsel/. live config/ + models_faye/ untouched.
  * FULL rankings logged (every combo prints).

LAUNCH ON DESKTOP (after BTC finishes):
    python tools/detector_reselect_rst.py            # 1440h (2mo, incl. rally)
    python tools/detector_reselect_rst.py 2880        # 4mo robustness check
Tip: capture everything incl. workers ->  ... > logs/detsel.out 2>&1

NOTE: standalone adx/hurst/rsi/dd/bounce are direction-AGNOSTIC or mean-revert
flavored — included for completeness; the COMBOS are the intended "direction AND
strength" use. PROMOTE deliberately; wire the winner into crypto_live_trader_ed.py
first if it isn't a built-in named detector, then restart the trader.
"""
import os, sys
import numpy as np
import pandas as pd

REPLAY   = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1440
HORIZONS = [5, 8]                       # user: assume 5h/8h is the best pair
HURST_WIN = 240                         # ~10 days lookback for rolling Hurst

os.environ['_FAYE_WARNINGS_BAKED'] = '1'
os.environ.setdefault('FAYE_CONFIG_DIR', 'config_faye_detsel')   # ISOLATED config writes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import crypto_trading_system_faye as eng

# ───────────────────────── helpers ─────────────────────────
def _wilder(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()

def _hurst_rs(prices):
    """Rescaled-range Hurst on a window of prices. >0.5 trending, <0.5 mean-reverting."""
    p = np.asarray(prices, dtype=float)
    lr = np.diff(np.log(p))
    N = len(lr)
    if N < 20:
        return np.nan
    pts = []
    for k in (kk for kk in (10, 20, 40, 80) if kk < N):
        nch = N // k
        vals = []
        for i in range(nch):
            seg = lr[i * k:(i + 1) * k]
            s = seg.std()
            if s > 0:
                dev = np.cumsum(seg - seg.mean())
                vals.append((dev.max() - dev.min()) / s)
        if vals:
            pts.append((k, np.mean(vals)))
    if len(pts) < 2:
        return np.nan
    return float(np.polyfit(np.log([x[0] for x in pts]), np.log([x[1] for x in pts]), 1)[0])

# ───────────────────────── compute ALL indicators (keyed by datetime) ─────────────────────────
_dfp = eng.load_data('ETH').copy()
_dfp['datetime'] = pd.to_datetime(_dfp['datetime'])
_dfp = _dfp.set_index('datetime').sort_index().tail(REPLAY + 800)   # 800 buffer covers tsmom672/sma480/hurst lookback
c, h, l = _dfp['close'], _dfp['high'], _dfp['low']

X = pd.DataFrame(index=_dfp.index)
X['close'] = c
for w in (6, 12, 24, 48, 72, 100, 168, 200, 480):
    X[f'sma{w}'] = c.rolling(w).mean()
for s in (9, 12, 21, 26):
    X[f'ema{s}'] = c.ewm(span=s).mean()
_d = c.diff()
_up14 = _d.clip(lower=0).rolling(14).mean()
_dn14 = (-_d.clip(upper=0)).rolling(14).mean()
X['rsi14'] = 100 - 100 / (1 + _up14 / _dn14.replace(0, np.nan))
X['dd48'] = (c / c.rolling(48).max() - 1) * 100
X['dd72'] = (c / c.rolling(72).max() - 1) * 100
X['bounce48'] = (c / c.rolling(48).min() - 1) * 100
X['macd'] = c.ewm(span=12).mean() - c.ewm(span=26).mean()
X['tsmom72'] = np.log(c / c.shift(72))
X['tsmom168'] = np.log(c / c.shift(168))
X['tsmom672'] = np.log(c / c.shift(672))
# ADX(14, Wilder)
_pc = c.shift(1)
_tr = pd.concat([h - l, (h - _pc).abs(), (l - _pc).abs()], axis=1).max(axis=1)
_up, _dn = h.diff(), -l.diff()
_pdm = ((_up > _dn) & (_up > 0)) * _up
_mdm = ((_dn > _up) & (_dn > 0)) * _dn
_atr = _wilder(_tr, 14)
_pdi = 100 * _wilder(_pdm, 14) / _atr
_mdi = 100 * _wilder(_mdm, 14) / _atr
X['adx'] = _wilder(100 * (_pdi - _mdi).abs() / (_pdi + _mdi).replace(0, np.nan), 14)
X['hurst'] = np.log(c).rolling(HURST_WIN).apply(_hurst_rs, raw=True)
_XBD = X.to_dict('index')
print(f"[detsel] indicators computed on {len(X)} bars "
      f"(adx non-nan={sum(1 for r in _XBD.values() if r['adx']==r['adx'])}, "
      f"hurst non-nan={sum(1 for r in _XBD.values() if r['hurst']==r['hurst'])})")

def _D(fn):
    """Wrap a row->bool detector with safe NaN/missing handling (default bull, engine convention)."""
    def _f(dt):
        r = _XBD.get(dt)
        if r is None:
            return True
        try:
            v = fn(r)
            return True if v != v else bool(v)
        except (KeyError, TypeError):
            return True
    return _f

# ───────────────────────── full detector menu (~45) ─────────────────────────
_MENU = {
    # SMA crosses (fast>slow = bull)
    'sma6>sma24':   _D(lambda r: r['sma6']  > r['sma24']),
    'sma12>sma24':  _D(lambda r: r['sma12'] > r['sma24']),
    'sma12>sma48':  _D(lambda r: r['sma12'] > r['sma48']),
    'sma24>sma72':  _D(lambda r: r['sma24'] > r['sma72']),
    'sma24>sma100': _D(lambda r: r['sma24'] > r['sma100']),
    'sma48>sma100': _D(lambda r: r['sma48'] > r['sma100']),
    'sma48>sma200': _D(lambda r: r['sma48'] > r['sma200']),
    'sma168>sma480':_D(lambda r: r['sma168']> r['sma480']),
    # price vs SMA
    'price>sma24':  _D(lambda r: r['close'] > r['sma24']),
    'price>sma48':  _D(lambda r: r['close'] > r['sma48']),
    'price>sma72':  _D(lambda r: r['close'] > r['sma72']),
    'price>sma100': _D(lambda r: r['close'] > r['sma100']),
    'price>sma200': _D(lambda r: r['close'] > r['sma200']),
    # EMA crosses
    'ema9>ema21':   _D(lambda r: r['ema9']  > r['ema21']),
    'ema12>ema26':  _D(lambda r: r['ema12'] > r['ema26']),
    # momentum
    'tsmom_672h':   _D(lambda r: r['tsmom672'] > 0),
    'tsmom_168h':   _D(lambda r: r['tsmom168'] > 0),
    'tsmom_72h':    _D(lambda r: r['tsmom72']  > 0),
    'macd>0':       _D(lambda r: r['macd'] > 0),
    # RSI
    'rsi>45':       _D(lambda r: r['rsi14'] > 45),
    'rsi>50':       _D(lambda r: r['rsi14'] > 50),
    'rsi>55':       _D(lambda r: r['rsi14'] > 55),
    # drawdown (within X% of recent high)
    'dd48>-2%':     _D(lambda r: r['dd48'] > -2),
    'dd48>-3%':     _D(lambda r: r['dd48'] > -3),
    'dd48>-5%':     _D(lambda r: r['dd48'] > -5),
    'dd72>-3%':     _D(lambda r: r['dd72'] > -3),
    'dd72>-5%':     _D(lambda r: r['dd72'] > -5),
    # bounce (up X% off recent low)
    'bounce48>2%':  _D(lambda r: r['bounce48'] > 2),
    'bounce48>3%':  _D(lambda r: r['bounce48'] > 3),
    # trend-strength / mean-revert
    'adx>25':       _D(lambda r: r['adx'] > 25),
    'adx>40':       _D(lambda r: r['adx'] > 40),
    'hurst>0.5':    _D(lambda r: r['hurst'] > 0.5),
    # 2-family composites (direction AND strength/quality)
    'sma24>100+dd48>-3':    _D(lambda r: r['sma24'] > r['sma100'] and r['dd48'] > -3),
    'sma24>100+rsi>45':     _D(lambda r: r['sma24'] > r['sma100'] and r['rsi14'] > 45),
    'dd48>-3%+rsi>45':      _D(lambda r: r['dd48'] > -3 and r['rsi14'] > 45),
    'price>sma72+rsi>50':   _D(lambda r: r['close'] > r['sma72'] and r['rsi14'] > 50),
    'sma24>sma100 & adx>25':    _D(lambda r: r['sma24'] > r['sma100'] and r['adx'] > 25),
    'sma24>sma100 & adx>40':    _D(lambda r: r['sma24'] > r['sma100'] and r['adx'] > 40),
    'sma24>sma100 & hurst>0.5': _D(lambda r: r['sma24'] > r['sma100'] and r['hurst'] > 0.5),
    'sma48>sma100 & adx>25':    _D(lambda r: r['sma48'] > r['sma100'] and r['adx'] > 25),
    'sma48>sma100 & adx>40':    _D(lambda r: r['sma48'] > r['sma100'] and r['adx'] > 40),
    'sma48>sma100 & hurst>0.5': _D(lambda r: r['sma48'] > r['sma100'] and r['hurst'] > 0.5),
    'tsmom_672h & adx>25':      _D(lambda r: r['tsmom672'] > 0 and r['adx'] > 25),
    'tsmom_672h & hurst>0.5':   _D(lambda r: r['tsmom672'] > 0 and r['hurst'] > 0.5),
}

# ───────────────────────── inject into the Mode-S/R/T sweep (in-process) ─────────────────────────
_orig_build = eng._build_regime_indicators_and_detectors
def _patched_build(asset):
    ind, dets = _orig_build(asset)     # dets == {'vol_calm'} since ENABLED is trimmed below
    dets.update(_MENU)                 # add the full menu (vol_calm stays from the engine)
    return ind, dets
eng._build_regime_indicators_and_detectors = _patched_build
eng.ENABLED_DETECTORS = {'vol_calm'}  # engine provides only vol_calm; everything else from _MENU

_ind0, _dets0 = eng._build_regime_indicators_and_detectors('ETH')
print("=" * 80)
print(f"  DETECTOR RE-SELECTION (R->S->T)  ETH  replay={REPLAY}h  horizons={HORIZONS}")
print(f"  sweeping {len(_dets0)} detectors")
print(f"  config dir : {eng.FAYE_CONFIG_DIR}  (ISOLATED — live config/ untouched)")
print(f"  models dir : {eng.FAYE_MODELS_DIR}  (read-only)")
print("=" * 80)

if os.environ.get('FAYE_DETSEL_DRY') == '1':
    _dts = sorted(d for d in _XBD if _XBD[d]['adx'] == _XBD[d]['adx'])[-3:]
    print("\n[DRY] full detector list:")
    for k in sorted(_dets0):
        print(f"     - {k}")
    print("\n[DRY] sample evals on last 3 bars:")
    for _dt in _dts:
        flags = {k: _dets0[k](_dt) for k in ('sma48>sma100', 'adx>25', 'hurst>0.5',
                                             'rsi>50', 'dd48>-3%', 'sma48>sma100 & adx>25')}
        print(f"  {_dt} ADX={_XBD[_dt]['adx']:.0f} Hurst={_XBD[_dt]['hurst']:.2f} RSI={_XBD[_dt]['rsi14']:.0f} -> {flags}")
    print("[DRY] OK — exiting before modes.")
    sys.exit(0)

# ───────────────────────── replicate engine RST dispatch, FULL rankings ─────────────────────────
class _Args:
    pass
a = _Args()
a.replay, a.conf, a.top, a.max_iter = REPLAY, 0, 9999, 0

r_results = eng._run_mode_r(['ETH'], HORIZONS, a)
eng._apply_mode_r_to_config(r_results)
eng.run_mode_s(['ETH'], HORIZONS, a)
eng.run_mode_t(['ETH'], a)

print("\n" + "=" * 80)
print(f"  DONE -> {eng.FAYE_CONFIG_DIR}/regime_config_faye.json")
print("  Full S ranking printed above. Promote deliberately — do NOT auto-promote.")
print("=" * 80)
