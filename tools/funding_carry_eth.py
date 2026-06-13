"""
funding_carry_eth.py — PAPER-TRADE delta-neutral funding-carry bot for ETH.
================================================================================
Simulates LONG $X spot ETH + SHORT $X ETHUSDT perp (delta-neutral). Price moves
cancel; the SHORT perp COLLECTS funding when the rate is positive (longs pay shorts)
and PAYS when negative. Net carry = realized funding - frictions. NOT prediction.

NO REAL TRADES, NO API KEYS. Free Binance PUBLIC market data only:
  - fapi.binance.com/fapi/v1/premiumIndex  (live mark + predicted funding + nextFundingTime)
  - fapi.binance.com/fapi/v1/fundingRate   (REALIZED settled funding, the accounting source of truth)
  - api.binance.com/api/v3/ticker/price    (spot; NOTE: api. host, not fapi.)

Correctness (per design review — funding accounting is easy to get wrong):
  * SIGN: short receives +rate*notional when rate>0 (verify: --selftest).
  * UNITS: rate is a per-8h DECIMAL (0.0001 = 0.01%); used raw in cash math, x100 only for display.
  * TIMING: funding credited ONLY at the realized 8h settlement (fundingTime), never per poll.
  * REALIZED vs PREDICTED: credit from /fundingRate history (realized); premiumIndex.lastFundingRate
    is the live PREDICTED rate, used only for display + the --positive-only gate.
  * RESTART: last_settled_funding_time guards against double-count; missed settlements are replayed once.
  * MARK vs SPOT: perp valued at mark, spot at spot; entry basis booked as a separate line item.
  * ANNUALIZE: 8h periods -> x1095/yr (APY), x sqrt(1095) (Sharpe).

CURRENT REALITY (June 2026): ETH funding is NEGATIVE right now (bearish regime) — a short carry
PAYS at the moment. The carry harvests in positive-funding (bull) regimes (~81% of history). Run it
forward and watch net_yield; or use --positive-only to sit out negative periods.

Run (venv active, from engine root):
  python tools/funding_carry_eth.py --status          # show state, exit
  python tools/funding_carry_eth.py --selftest        # accounting unit checks, exit
  python tools/funding_carry_eth.py                   # one cycle (enter if flat), exit
  python tools/funding_carry_eth.py --loop            # run forever (Ctrl-C to stop)
  python tools/funding_carry_eth.py --loop --positive-only --notional 10000
"""
import os
import sys
import csv as _csv
import json
import time
import ssl
import socket
import atexit
import argparse
import threading
import urllib.request
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PREMIUM_URL = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=ETHUSDT"
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate?symbol=ETHUSDT"
SPOT_URL = "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT"
FUNDING_INTERVAL_MS = 8 * 3600 * 1000          # ETHUSDT settles every 8h (00/08/16 UTC)
PERIODS_PER_YEAR = 365 * 3                       # 8h funding periods
STATE_FILE = "config/funding_carry_eth_state.json"
LOCK_FILE = "config/funding_carry_eth.lock"      # cross-machine single-instance guard (config/ is Drive-synced)
LOCK_STALE_MIN = 20                               # heartbeat older than this = previous instance is dead
OUTDIR = "output/funding_carry_eth"
CSV_PATH = os.path.join(OUTDIR, "funding_carry.csv")
SETTLE_CSV = os.path.join(OUTDIR, "funding_settlements.csv")
MAX_STALE_MS = 120_000                           # refuse data older than 2 min
os.makedirs(OUTDIR, exist_ok=True)

_CTX = ssl._create_unverified_context()          # Critical Rule 6 (Windows SSL)
_state_lock = threading.Lock()

try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("Europe/Zurich")          # Critical Rule 4
except Exception:
    LOCAL_TZ = None


def _stamps():
    now = datetime.now(timezone.utc)
    utc_s = now.strftime("%Y-%m-%d %H:%M:%S")
    local_s = (now.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z") if LOCAL_TZ else utc_s + " UTC")
    return utc_s, local_s


def _keep_awake():
    """Prevent Windows sleep/Modern Standby for the loop's life (mirror of dryrun_1_4h)."""
    try:
        import ctypes
        import atexit
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_AWAYMODE_REQUIRED = 0x80000000, 0x00000001, 0x00000040
        k32 = ctypes.windll.kernel32
        prev = k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
        if prev == 0:
            prev = k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        if prev != 0:
            atexit.register(lambda: k32.SetThreadExecutionState(ES_CONTINUOUS))
            return True
    except Exception:
        pass
    return False


def _get(url, retries=4, backoff=(1, 2, 5, 10)):
    """Resilient public GET (pattern from download_macro_data._binance_get). Raises on final failure."""
    last = None
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for i in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30, context=_CTX) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            last = e
            if i < retries:
                time.sleep(backoff[min(i, len(backoff) - 1)])
    raise RuntimeError(f"GET failed after {retries} retries: {url} :: {last}")


def fetch_premium_index():
    d = _get(PREMIUM_URL)
    return dict(mark=float(d["markPrice"]), index=float(d["indexPrice"]),
               pred_rate=float(d["lastFundingRate"]), next_funding=int(d["nextFundingTime"]),
               server_ms=int(d["time"]))


def fetch_spot():
    return float(_get(SPOT_URL)["price"])


def fetch_funding_history(start_ms=None, limit=1000):
    url = FUNDING_URL + f"&limit={limit}" + (f"&startTime={int(start_ms)}" if start_ms else "")
    out = []
    for e in _get(url):
        try:
            mark = float(e.get("markPrice") or 0)   # old (2019-era) records have markPrice='' -> 0
        except (TypeError, ValueError):
            mark = 0.0
        out.append(dict(ft=int(e["fundingTime"]), rate=float(e["fundingRate"]), mark=mark))
    return out


# ---------------- state ----------------
def default_state():
    return dict(schema_version=1, asset="ETH", state="flat",
                notional_usd=0.0, spot_units=0.0, spot_entry_px=0.0,
                perp_units=0.0, perp_entry_px=0.0, perp_mark_px=0.0,
                entry_basis_usd=0.0, cash_usd=0.0,
                cum_funding_usd=0.0, cum_fees_usd=0.0, funding_events=0,
                last_settled_funding_time=0, next_funding_time=0,
                inception_utc="", last_poll_utc="", ret_samples=[])


def load_state():
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
        return {**default_state(), **s}
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        if os.path.exists(STATE_FILE):
            print(f"  [warn] state file unreadable ({e}) — starting flat")
        return default_state()


def _atomic_write_json(path, obj, attempts=6, base_delay=0.4):
    """Atomic JSON write (temp + os.replace). config/ is Google Drive-synced and the Drive
    virtual FS sporadically raises OSError (EINVAL/EACCES) on create/replace mid-sync — a single
    such hiccup must NOT crash a multi-day loop. Retry with exponential backoff (~25s total);
    on persistent failure log and return False rather than raise. Safe to skip a write: the next
    cycle re-persists and funding replay is idempotent by last_settled_funding_time."""
    tmp = f"{path}.{os.getpid()}.tmp"
    last = None
    for i in range(attempts):
        try:
            with open(tmp, "w") as f:
                json.dump(obj, f, indent=2)
            os.replace(tmp, path)
            return True
        except OSError as e:
            last = e
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            time.sleep(base_delay * (2 ** i))
    print(f"  [warn] could not persist {path} after {attempts} tries ({last}) — continuing "
          f"(re-persists next cycle; funding replay is idempotent)")
    return False


def save_state(state):
    with _state_lock:
        _atomic_write_json(STATE_FILE, state)


def write_lock():
    """Heartbeat lockfile. config/ is Drive-synced, so this prevents a 2nd instance (e.g. the OTHER
    machine) from running the bot and corrupting the shared state ledger."""
    _atomic_write_json(LOCK_FILE, dict(host=socket.gethostname(), pid=os.getpid(), heartbeat=_stamps()[0]))


def acquire_lock(force=False):
    """Refuse to start if another instance's heartbeat is recent (< LOCK_STALE_MIN). Returns True if acquired."""
    if os.path.exists(LOCK_FILE):
        try:
            lk = json.load(open(LOCK_FILE))
            hb = datetime.strptime(lk.get("heartbeat", ""), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            age_min = (datetime.now(timezone.utc) - hb).total_seconds() / 60
            same_host = lk.get("host") == socket.gethostname()
            if age_min < LOCK_STALE_MIN and not same_host and not force:
                print(f"  [REFUSING TO START] another carry instance looks live: host={lk.get('host')} "
                      f"pid={lk.get('pid')} heartbeat {age_min:.1f} min ago.")
                print(f"  config/ is Drive-synced — two instances would corrupt the shared ledger. "
                      f"If you're SURE the other is dead, re-run with --force.")
                return False
            # recent lock from THIS host but a different pid = a crashed/previous instance we're
            # restarting (the .bat enforces one instance per machine) — take it over silently.
            if age_min < LOCK_STALE_MIN and same_host and lk.get("pid") != os.getpid():
                print(f"  [lock] taking over recent same-host lock (pid {lk.get('pid')}, "
                      f"{age_min:.1f} min ago) — assuming previous instance exited.")
        except Exception:
            pass
    write_lock()
    atexit.register(lambda: os.path.exists(LOCK_FILE) and os.remove(LOCK_FILE))
    return True


# ---------------- carry mechanics ----------------
def enter(state, spot_px, mark_px, next_funding, notional, friction):
    """flat -> carry. Long spot + short perp at `notional` each. Charge friction on BOTH legs."""
    state["state"] = "carry"
    state["notional_usd"] = notional
    state["spot_units"] = notional / spot_px
    state["spot_entry_px"] = spot_px
    state["perp_units"] = notional / mark_px            # short size (positive units, short implied)
    state["perp_entry_px"] = mark_px
    state["perp_mark_px"] = mark_px
    state["entry_basis_usd"] = (mark_px - spot_px) / spot_px * notional   # one-time basis line item
    cost = 2 * notional * friction
    state["cash_usd"] -= cost
    state["cum_fees_usd"] += cost
    # funding clock: set to the ACTUAL most-recent realized settlement fundingTime (robust to
    # Binance's ms jitter — an exact `next - 8h` boundary would let the just-passed, pre-entry
    # settlement slip through and credit a period we never held). Only settlements strictly
    # AFTER this fundingTime are ever credited.
    try:
        latest = fetch_funding_history(limit=1)
        state["last_settled_funding_time"] = latest[-1]["ft"] if latest else next_funding - FUNDING_INTERVAL_MS
    except Exception:
        state["last_settled_funding_time"] = next_funding - FUNDING_INTERVAL_MS
    state["next_funding_time"] = next_funding
    if not state["inception_utc"]:
        state["inception_utc"] = _stamps()[0]
    return cost


def flatten(state, spot_px, mark_px, friction):
    """carry -> flat. Charge friction on both legs. Keeps cumulative ledger."""
    notional = state["notional_usd"]
    cost = 2 * notional * friction
    state["cash_usd"] -= cost
    state["cum_fees_usd"] += cost
    state["state"] = "flat"
    state["spot_units"] = state["perp_units"] = 0.0
    return cost


def settle_funding(state):
    """Credit REALIZED funding (from /fundingRate history) for every settlement past
    last_settled_funding_time. Idempotent by fundingTime. Returns (credited_usd, settle_rows)."""
    if state["state"] != "carry":
        return 0.0, []
    # clamp: never replay from epoch on a corrupt carry state (last_settled=0) — Binance startTime=1
    # would return 2019-era records and march forward forever.
    if state["last_settled_funding_time"] <= 0:
        floor = (state["next_funding_time"] or int(datetime.now(timezone.utc).timestamp() * 1000)) - FUNDING_INTERVAL_MS
        state["last_settled_funding_time"] = floor
    credited, rows = 0.0, []
    # paginate: Binance returns the OLDEST `limit` from startTime, so loop until caught up
    # (handles the bot being down for >1000 settlements). Idempotent by fundingTime.
    while True:
        try:
            hist = fetch_funding_history(start_ms=state["last_settled_funding_time"] + 1, limit=1000)
        except Exception as e:
            print(f"  [warn] funding history fetch failed ({e}) — skipping credit (retry next cycle)")
            break
        hist = sorted((e for e in hist if e["ft"] > state["last_settled_funding_time"]), key=lambda e: e["ft"])
        if not hist:
            break
        for ev in hist:
            # SIGN: short receives +rate*notional when rate>0. NOTIONAL at settlement = perp_units * mark_at_settlement.
            mark = ev["mark"] if ev["mark"] > 0 else state["perp_mark_px"]   # fallback if record's mark is blank
            fund_usd = ev["rate"] * (state["perp_units"] * mark)
            state["cash_usd"] += fund_usd
            state["cum_funding_usd"] += fund_usd
            state["funding_events"] += 1
            state["last_settled_funding_time"] = ev["ft"]
            state["ret_samples"].append(fund_usd / state["notional_usd"] if state["notional_usd"] else 0.0)
            state["ret_samples"] = state["ret_samples"][-500:]
            credited += fund_usd
            rows.append((ev["ft"], ev["rate"], mark, fund_usd))
            save_state(state)   # crash-safe: persist immediately after each credit
        if len(hist) < 1000:
            break
    return credited, rows


def metrics(state):
    nz = state["notional_usd"] or 1.0
    net = state["cum_funding_usd"] - state["cum_fees_usd"]
    incept = state.get("inception_utc") or _stamps()[0]
    try:
        incept_dt = datetime.strptime(incept, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        incept_dt = datetime.now(timezone.utc)
    days = max((datetime.now(timezone.utc) - incept_dt).total_seconds() / 86400, 1e-9)
    yield_pct = net / nz * 100
    # only annualize after >= one 8h settlement (~0.33d); below that the divisor is tiny -> garbage APY
    apy = yield_pct * 365 / days if days >= 0.33 else float("nan")
    rs = state["ret_samples"]
    import statistics
    sharpe = (statistics.mean(rs) / statistics.pstdev(rs) * (PERIODS_PER_YEAR ** 0.5)) if len(rs) >= 2 and statistics.pstdev(rs) > 0 else float("nan")
    return dict(net=net, yield_pct=yield_pct, apy=apy, sharpe=sharpe, days=days)


def log_row(state, pi, spot_px, policy, action, settled_usd):
    m = metrics(state)
    utc_s, local_s = _stamps()
    basis_bps = (pi["mark"] - spot_px) / spot_px * 1e4
    net_delta_bps = (state["spot_units"] * spot_px - state["perp_units"] * pi["mark"]) / (state["notional_usd"] or 1.0) * 1e4
    row = dict(logged_utc=utc_s, logged_local=local_s, policy=policy, state=state["state"],
               pred_funding_rate=f'{pi["pred_rate"]:.8f}', mark_px=f'{pi["mark"]:.2f}', spot_px=f'{spot_px:.2f}',
               basis_bps=f"{basis_bps:.2f}", notional_usd=f'{state["notional_usd"]:.2f}',
               spot_units=f'{state["spot_units"]:.6f}', perp_units=f'{state["perp_units"]:.6f}',
               net_delta_bps=f"{net_delta_bps:.2f}", action=action,
               settled_funding_usd=(f"{settled_usd:.4f}" if settled_usd else ""),
               cum_funding_usd=f'{state["cum_funding_usd"]:.4f}', cum_fees_usd=f'{state["cum_fees_usd"]:.4f}',
               net_carry_usd=f'{m["net"]:.4f}', funding_events=state["funding_events"],
               days_live=f'{m["days"]:.3f}', net_yield_pct=f'{m["yield_pct"]:.4f}',
               annualized_pct=(f'{m["apy"]:.2f}' if m["apy"] == m["apy"] else ""),
               running_sharpe=(f'{m["sharpe"]:.2f}' if m["sharpe"] == m["sharpe"] else ""))
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    return m, basis_bps, net_delta_bps


def log_settlements(rows):
    if not rows:
        return
    new = not os.path.exists(SETTLE_CSV)
    with open(SETTLE_CSV, "a", newline="") as f:
        w = _csv.writer(f)
        if new:
            w.writerow(["funding_time_utc", "rate", "mark", "funding_usd"])
        for ft, rate, mark, usd in rows:
            ts = datetime.fromtimestamp(ft / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            w.writerow([ts, f"{rate:.8f}", f"{mark:.2f}", f"{usd:.4f}"])


# ---------------- cycle ----------------
def cycle(state, args):
    policy = "positive_only" if args.positive_only else "always_on"
    try:
        pi = fetch_premium_index()
        spot_px = fetch_spot()
    except Exception as e:
        print(f"  [warn] price fetch failed ({e}) — skipping cycle")
        return
    # NOTE: no local-clock staleness gate — these machines have real NTP drift (Critical Rule 11),
    # so local_now vs server_ms is unreliable. A successful live fetch is inherently fresh, AND
    # funding crediting is event-driven off the REALIZED /fundingRate history (not this snapshot),
    # so a momentarily stale premiumIndex can't corrupt the ledger anyway.
    state["next_funding_time"] = pi["next_funding"]
    state["perp_mark_px"] = pi["mark"]

    # 1) settle any realized funding (timing/double-count safe)
    settled_usd, settle_rows = settle_funding(state)
    if settle_rows:
        log_settlements(settle_rows)

    action = "HOLD"
    # 2) policy / position management
    if state["state"] == "flat":
        # always-on: enter. positive-only: enter only if predicted funding >= 0.
        if (not args.positive_only) or pi["pred_rate"] >= 0:
            enter(state, spot_px, pi["mark"], pi["next_funding"], args.notional, args.fric)
            action = "ENTER"
    else:  # carry
        if args.positive_only and pi["pred_rate"] < 0:
            flatten(state, spot_px, pi["mark"], args.fric)
            action = "FLATTEN_NEG"
        else:
            # 3) delta rebalance
            net_delta = state["spot_units"] * spot_px - state["perp_units"] * pi["mark"]
            if abs(net_delta) / (state["notional_usd"] or 1.0) * 1e4 > args.rebalance_bps:
                # compute the prospective trade BEFORE mutating, so the dust floor is real
                new_spot_u = state["notional_usd"] / spot_px
                new_perp_u = state["notional_usd"] / pi["mark"]
                traded = (abs(new_spot_u * spot_px - state["spot_units"] * spot_px)
                          + abs(new_perp_u * pi["mark"] - state["perp_units"] * pi["mark"]))
                if traded > 5.0:   # dust floor: only rebalance if the trade is material
                    state["spot_units"], state["perp_units"] = new_spot_u, new_perp_u
                    cost = traded * args.fric
                    state["cash_usd"] -= cost
                    state["cum_fees_usd"] += cost
                    action = "REBAL"

    if settled_usd and action == "HOLD":
        action = "SETTLE"
    state["last_poll_utc"] = _stamps()[0]
    save_state(state)
    write_lock()   # heartbeat the single-instance lock
    m, basis_bps, net_delta_bps = log_row(state, pi, spot_px, policy, action, settled_usd)

    _, local_s = _stamps()
    fr_disp = pi["pred_rate"] * 100
    print(f"  {local_s} | {state['state']:5} {action:11} | pred_fund {fr_disp:+.4f}%/8h ({fr_disp*1095/100*100:+.1f}%/yr) "
          f"mark {pi['mark']:.2f} spot {spot_px:.2f} basis {basis_bps:+.1f}bp delta {net_delta_bps:+.1f}bp")
    if settled_usd:
        print(f"           +SETTLED funding ${settled_usd:+.4f} ({len(settle_rows)} event(s))")
    print(f"           net carry ${m['net']:+.2f}  (funding ${state['cum_funding_usd']:+.2f} - fees ${state['cum_fees_usd']:.2f})  "
          f"| {m['days']:.2f}d  yield {m['yield_pct']:+.3f}%  APY {m['apy']:+.1f}%  Sharpe {m['sharpe'] if m['sharpe']==m['sharpe'] else float('nan'):.2f}")


def selftest():
    """Accounting unit checks (pitfalls 1,2,7)."""
    s = default_state()
    s.update(state="carry", notional_usd=10000.0, perp_units=10000.0 / 2000.0)  # short 5 ETH at $2000
    # rate=+0.0001 (0.01%/8h), mark 2000 -> short receives +10000*0.0001 = +$1.00
    fund = 0.0001 * (s["perp_units"] * 2000.0)
    assert abs(fund - 1.0) < 1e-9, f"sign/units FAIL: got {fund}, expected +1.00"
    # rate=-0.0001 -> short PAYS -$1.00
    fund_neg = -0.0001 * (s["perp_units"] * 2000.0)
    assert abs(fund_neg + 1.0) < 1e-9, f"negative FAIL: got {fund_neg}, expected -1.00"
    # units: 0.0001 is 0.01%, annualized x1095 = 10.95%
    assert abs(0.0001 * 1095 - 0.1095) < 1e-9
    print("  SELFTEST PASS: short +rate=credit, -rate=debit; units per-8h decimal; APY=x1095.")
    print("    rate +0.0001 -> +$1.00 on $10k short  |  rate -0.0001 -> -$1.00  |  0.0001/8h = 10.95%/yr")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notional", type=float, default=10000.0, help="per-leg target USD notional")
    ap.add_argument("--leverage", type=float, default=1.0, help="perp leverage (reporting only)")
    ap.add_argument("--rebalance-bps", dest="rebalance_bps", type=float, default=50.0)
    ap.add_argument("--fee-bps", dest="fee_bps", type=float, default=5.0, help="per-leg fee bps (maker~5, taker~10)")
    ap.add_argument("--slippage-bps", dest="slip_bps", type=float, default=2.0)
    ap.add_argument("--positive-only", action="store_true", help="flatten when predicted funding<0; default OFF (always-on)")
    ap.add_argument("--loop", action="store_true", help="run forever, aligned to :02 past each hour")
    ap.add_argument("--poll-secs", dest="poll_secs", type=int, default=0, help="override loop cadence in seconds")
    ap.add_argument("--reset", action="store_true", help="wipe state to flat before starting")
    ap.add_argument("--force", action="store_true", help="override the single-instance lock (only if the other instance is dead)")
    ap.add_argument("--status", action="store_true", help="print state + ledger and exit")
    ap.add_argument("--selftest", action="store_true", help="run accounting unit checks and exit")
    args = ap.parse_args()
    args.fric = (args.fee_bps + args.slip_bps) / 1e4

    if args.selftest:
        selftest()
        return
    if args.reset and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print("  state reset to flat.")
    state = load_state()
    if args.status:
        m = metrics(state)
        print(json.dumps(state, indent=2))
        print(f"\n  net carry ${m['net']:+.2f}  yield {m['yield_pct']:+.3f}%  APY {m['apy']:+.1f}%  "
              f"Sharpe {m['sharpe'] if m['sharpe']==m['sharpe'] else float('nan'):.2f}  ({state['funding_events']} settlements)")
        return

    print("=" * 92)
    print("  FUNDING-CARRY PAPER BOT — ETH (long spot + short perp), NO REAL TRADES, free Binance data")
    print("=" * 92)
    print(f"  notional ${args.notional:,.0f}/leg  policy {'positive-only' if args.positive_only else 'always-on'}  "
          f"fric {args.fee_bps+args.slip_bps:.0f}bps/leg  rebalance>{args.rebalance_bps:.0f}bp")
    print(f"  state file {STATE_FILE}  |  log {CSV_PATH}")
    print("  NOTE: ETH funding is currently NEGATIVE (bearish) — a short carry PAYS now; it harvests when funding turns positive.\n")

    if not acquire_lock(args.force):
        return

    if not args.loop and not args.poll_secs:
        cycle(state, args)
        return

    _keep_awake()
    try:
        while True:
            cycle(state, args)
            if args.poll_secs:
                time.sleep(args.poll_secs)
            else:  # align to :02 past each hour (mirror dryrun)
                now = datetime.now(timezone.utc)
                nxt = now.replace(minute=2, second=0, microsecond=0)
                if nxt <= now:
                    nxt += timedelta(hours=1)
                time.sleep(max(5, (nxt - now).total_seconds()))
    except KeyboardInterrupt:
        print("\n  stopped (Ctrl-C). State saved; re-run to resume.")


if __name__ == "__main__":
    main()
