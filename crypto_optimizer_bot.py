"""
Crypto Optimizer Bot — Remote Telegram interface for model optimization
======================================================================
Separate from the live trading bot. Allows remote triggering of Mode D/V/H/P/S
runs via Telegram inline keyboard menus.

Usage:
  python crypto_optimizer_bot.py              # Start the bot
  python crypto_optimizer_bot.py --setup      # Configure Telegram

Commands:
  /optimize  — Start optimization flow (Mode → Assets → Horizons → Confirm)
  /queue     — Show running/pending/completed jobs
  /cancel    — Cancel current job or menu flow
  /status    — Show current production models
  /results   — Show last results for an asset
  /help      — List commands
  /stop      — Stop the bot
"""

import os
import sys
import ssl
import json
import time
import re
import uuid
import threading
import subprocess
import urllib.request
from datetime import datetime
from dataclasses import dataclass, field

# ── SSL fix for Windows ──────────────────────────────────────────────
_ssl_ctx = ssl._create_unverified_context()

# ── Machine detection & paths ────────────────────────────────────────
_logical_cores = os.cpu_count() or 8
ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))

if _logical_cores >= 24:
    MACHINE = 'DESKTOP'
    PYTHON_EXE = r'C:\algo_trading\venv\Scripts\python.exe'
else:
    MACHINE = 'LAPTOP'
    PYTHON_EXE = r'C:\Users\alexa\algo_trading\venv\Scripts\python.exe'

# Fallback to current interpreter if venv path doesn't exist
if not os.path.exists(PYTHON_EXE):
    PYTHON_EXE = sys.executable

SCRIPT_PATH = os.path.join(ENGINE_DIR, 'crypto_trading_system_ed.py')
PRODUCTION_CSV = os.path.join(ENGINE_DIR, 'models', 'crypto_ed_production.csv')
TRADING_CONFIG = os.path.join(ENGINE_DIR, 'config', 'regime_config_ed.json')
OPTIMIZER_CONFIG_FILE = os.path.join(ENGINE_DIR, 'config', 'telegram_optimizer_config.json')

ASSETS = ['BTC', 'ETH', 'SOL', 'LINK', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT']
HORIZONS = [5, 6, 7, 8, 10, 12, 14]
MODES = {
    'P':    'PySR Discovery',
    'D':    'Grid Search',
    'V':    'Validate + Refine',
    'DV':   'Grid + Validate',
    'H':    'Horizon Sweep',
    'R':    'Regime Backtest',
    'S':    'Regime Confidence',
    'RS':   'Regime + Confidence',
    'HRS':  'Full Ed Pipeline',
    'DVRS': 'DV + Regime + Conf',
}

# Time estimates per (asset, horizon) in minutes
MODE_TIME_EST = {'P': 60, 'D': 25, 'V': 30, 'DV': 55, 'H': 55, 'R': 30, 'S': 30, 'RS': 60, 'HRS': 120, 'DVRS': 120}

# ── Telegram config ──────────────────────────────────────────────────
TELEGRAM_CONFIG = {'token': '', 'chat_id': ''}

def _load_telegram_config():
    global TELEGRAM_CONFIG
    if os.path.exists(OPTIMIZER_CONFIG_FILE):
        with open(OPTIMIZER_CONFIG_FILE) as f:
            TELEGRAM_CONFIG.update(json.load(f))
    # Env vars override
    if os.environ.get('OPTIMIZER_TELEGRAM_TOKEN'):
        TELEGRAM_CONFIG['token'] = os.environ['OPTIMIZER_TELEGRAM_TOKEN']
    if os.environ.get('OPTIMIZER_TELEGRAM_CHAT_ID'):
        TELEGRAM_CONFIG['chat_id'] = os.environ['OPTIMIZER_TELEGRAM_CHAT_ID']


# ── Telegram API ─────────────────────────────────────────────────────
_last_update_id = 0


def send_telegram(message, parse_mode='HTML'):
    token = TELEGRAM_CONFIG.get('token', '')
    chat_id = TELEGRAM_CONFIG.get('chat_id', '')
    if not token or not chat_id:
        print(f"  [Telegram not configured] {message}")
        return None
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        'chat_id': chat_id, 'text': message, 'parse_mode': parse_mode
    }).encode('utf-8')
    try:
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as resp:
            result = json.loads(resp.read().decode())
            if result.get('ok'):
                return result['result'].get('message_id')
    except Exception as e:
        print(f"  [!] Telegram send error: {e}")
    return None


def send_telegram_with_buttons(message, buttons, parse_mode='HTML'):
    """Send message with inline keyboard.
    buttons: list of rows, each row is list of (text, callback_data) tuples.
    """
    token = TELEGRAM_CONFIG.get('token', '')
    chat_id = TELEGRAM_CONFIG.get('chat_id', '')
    if not token or not chat_id:
        print(f"  [Telegram not configured] {message}")
        return None
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    keyboard = {
        'inline_keyboard': [
            [{'text': text, 'callback_data': cb} for text, cb in row]
            for row in buttons
        ]
    }
    payload = json.dumps({
        'chat_id': chat_id, 'text': message,
        'parse_mode': parse_mode, 'reply_markup': keyboard
    }).encode('utf-8')
    try:
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as resp:
            result = json.loads(resp.read().decode())
            if result.get('ok'):
                return result['result'].get('message_id')
    except Exception as e:
        print(f"  [!] Telegram send error: {e}")
    return None


def edit_telegram_message(message_id, text, parse_mode='HTML'):
    """Edit an existing message (for progress updates)."""
    token = TELEGRAM_CONFIG.get('token', '')
    chat_id = TELEGRAM_CONFIG.get('chat_id', '')
    if not token or not chat_id or not message_id:
        return False
    url = f"https://api.telegram.org/bot{token}/editMessageText"
    payload = json.dumps({
        'chat_id': chat_id, 'message_id': message_id,
        'text': text, 'parse_mode': parse_mode
    }).encode('utf-8')
    try:
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as resp:
            result = json.loads(resp.read().decode())
            return result.get('ok', False)
    except Exception:
        return False


def _answer_callback_query(callback_query_id):
    token = TELEGRAM_CONFIG.get('token', '')
    if not token:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
        payload = json.dumps({'callback_query_id': callback_query_id}).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=5, context=_ssl_ctx)
    except Exception:
        pass


def check_telegram_updates():
    """Poll for new messages/callbacks. Returns list of (text, callback_query_id_or_None)."""
    global _last_update_id
    token = TELEGRAM_CONFIG.get('token', '')
    if not token:
        return []
    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates?offset={_last_update_id + 1}&timeout=0"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=5, context=_ssl_ctx) as resp:
            data = json.loads(resp.read().decode())
        if not data.get('ok') or not data.get('result'):
            return []
        updates = []
        for update in data['result']:
            _last_update_id = update['update_id']
            # Text message
            text = update.get('message', {}).get('text', '').strip()
            if text:
                updates.append((text, None))
            # Callback query (inline button press)
            cb = update.get('callback_query')
            if cb:
                _answer_callback_query(cb['id'])
                cb_data = cb.get('data', '').strip()
                if cb_data:
                    updates.append((cb_data, cb['id']))
        return updates
    except Exception:
        return []


def _flush_old_updates():
    global _last_update_id
    token = TELEGRAM_CONFIG.get('token', '')
    if not token:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates?offset=-1"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=5, context=_ssl_ctx) as resp:
            data = json.loads(resp.read().decode())
        if data.get('ok') and data.get('result'):
            _last_update_id = data['result'][-1]['update_id']
    except Exception:
        pass


# ── Job Queue ────────────────────────────────────────────────────────
@dataclass
class OptJob:
    id: str
    mode: str
    assets: list
    horizons: list
    replay: int = 0              # --replay hours (0 = default)
    status: str = 'queued'       # queued, running, done, failed, cancelled
    created_at: str = ''
    started_at: str = ''
    finished_at: str = ''
    process: object = field(default=None, repr=False)
    progress_msg_id: int = None  # Telegram message ID for progress updates
    progress_text: str = ''
    result_summary: str = ''
    output_lines: list = field(default_factory=list, repr=False)

    def label(self):
        assets_str = ','.join(self.assets)
        h_str = ','.join(f'{h}h' for h in self.horizons)
        replay_str = f" {self.replay}h" if self.replay else ""
        return f"{self.mode} {assets_str} {h_str}{replay_str}"


_job_queue = []          # pending jobs
_current_job = None      # running job
_job_history = []        # completed jobs (last 10)
_queue_lock = threading.Lock()
_stop_event = threading.Event()


# ── Menu State Machine ───────────────────────────────────────────────
REPLAY_OPTIONS = {
    '2w': (336, '2 weeks'),
    '1m': (720, '1 month'),
    '2m': (1440, '2 months'),
    '3m': (2160, '3 months'),
    '4m': (2880, '4 months'),
    '6m': (4320, '6 months'),
}
REPLAY_MODES = {'R', 'RS', 'HRS', 'DVRS'}  # modes that support --replay

_menu_state = {
    'step': None,              # None, 'mode', 'assets', 'horizons', 'replay', 'confirm'
    'mode': None,
    'selected_assets': set(),
    'selected_horizons': set(),
    'selected_replay': 2880,   # default 4 months
    'last_activity': 0,
}
MENU_TIMEOUT = 300  # 5 minutes


def _menu_reset():
    _menu_state.update({
        'step': None, 'mode': None,
        'selected_assets': set(), 'selected_horizons': set(),
        'selected_replay': 2880, 'last_activity': 0,
    })


def _menu_is_active():
    if _menu_state['step'] is None:
        return False
    if time.time() - _menu_state['last_activity'] > MENU_TIMEOUT:
        _menu_reset()
        return False
    return True


def _menu_touch():
    _menu_state['last_activity'] = time.time()


# ── Menu Handlers ────────────────────────────────────────────────────
def _show_mode_menu():
    _menu_state['step'] = 'mode'
    _menu_touch()
    buttons = [
        [('D - Grid', 'opt_mode_D'), ('V - Validate', 'opt_mode_V')],
        [('DV - Grid+Val', 'opt_mode_DV'), ('H - Horizon', 'opt_mode_H')],
        [('R - Regime', 'opt_mode_R'), ('S - Confidence', 'opt_mode_S')],
        [('RS - Regime+Conf', 'opt_mode_RS'), ('HRS - Full', 'opt_mode_HRS')],
        [('P - PySR', 'opt_mode_P')],
        [('Cancel', 'opt_cancel')],
    ]
    send_telegram_with_buttons("<b>Select optimization mode:</b>", buttons)


def _show_asset_menu():
    _menu_state['step'] = 'assets'
    _menu_touch()
    selected = _menu_state['selected_assets']
    # Build asset buttons in rows of 3
    asset_rows = []
    for i in range(0, len(ASSETS), 3):
        row = []
        for a in ASSETS[i:i+3]:
            check = '[x]' if a in selected else '[ ]'
            row.append((f"{check} {a}", f"opt_asset_{a}"))
        asset_rows.append(row)
    asset_rows.append([('Select All', 'opt_asset_all'), ('Clear', 'opt_asset_clear')])
    asset_rows.append([('Next ->', 'opt_asset_next'), ('Cancel', 'opt_cancel')])
    mode_name = MODES.get(_menu_state['mode'], _menu_state['mode'])
    send_telegram_with_buttons(
        f"<b>Mode: {mode_name}</b>\nSelect assets (tap to toggle):",
        asset_rows
    )


def _show_horizon_menu():
    _menu_state['step'] = 'horizons'
    _menu_touch()
    selected = _menu_state['selected_horizons']
    buttons = []
    row = []
    for h in HORIZONS:
        check = '[x]' if h in selected else '[ ]'
        row.append((f"{check} {h}h", f"opt_h_{h}"))
        if len(row) == 4:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([('Next ->', 'opt_h_next'), ('Cancel', 'opt_cancel')])
    send_telegram_with_buttons("<b>Select horizons:</b>", buttons)


def _show_replay_menu():
    _menu_state['step'] = 'replay'
    _menu_touch()
    current = _menu_state['selected_replay']
    buttons = []
    row = []
    for key, (hours, label) in REPLAY_OPTIONS.items():
        mark = ">> " if hours == current else ""
        row.append((f"{mark}{label}", f"opt_replay_{key}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([('Next ->', 'opt_replay_next'), ('Cancel', 'opt_cancel')])
    send_telegram_with_buttons("<b>Select backtest period:</b>", buttons)


def _show_confirm():
    _menu_state['step'] = 'confirm'
    _menu_touch()
    mode = _menu_state['mode']
    assets = sorted(_menu_state['selected_assets'])
    horizons = sorted(_menu_state['selected_horizons'])
    mode_name = MODES.get(mode, mode)
    assets_str = ', '.join(assets)
    h_str = ', '.join(f'{h}h' for h in horizons)

    # Time estimate
    per_combo = MODE_TIME_EST.get(mode, 30)
    if mode == 'H':
        total_min = per_combo * len(horizons) * len(assets)
    else:
        total_min = per_combo * len(assets) * len(horizons)

    if total_min < 60:
        time_est = f"~{total_min} min"
    else:
        time_est = f"~{total_min/60:.1f} hours"

    replay_str = ""
    if mode in REPLAY_MODES:
        replay_hours = _menu_state['selected_replay']
        for key, (hours, label) in REPLAY_OPTIONS.items():
            if hours == replay_hours:
                replay_str = f"\nReplay: {label} ({replay_hours}h)"
                break

    msg = (
        f"<b>Confirm optimization:</b>\n\n"
        f"Mode: {mode_name} ({mode})\n"
        f"Assets: {assets_str}\n"
        f"Horizons: {h_str}{replay_str}\n"
        f"Machine: {MACHINE}\n"
        f"Est. time: {time_est}"
    )
    buttons = [[('Confirm', 'opt_confirm'), ('Cancel', 'opt_cancel')]]
    send_telegram_with_buttons(msg, buttons)


def _handle_menu_callback(data):
    """Route inline button callbacks for the optimization menu."""
    if data == 'opt_cancel':
        _menu_reset()
        send_telegram("Cancelled.")
        return

    # Mode selection
    if data.startswith('opt_mode_'):
        mode = data.replace('opt_mode_', '')
        _menu_state['mode'] = mode
        # Default asset selection
        _menu_state['selected_assets'] = {'BTC'}
        # Default horizon selection based on mode
        if mode in ('H', 'HRS', 'DVRS', 'R', 'RS'):
            _menu_state['selected_horizons'] = set(HORIZONS)
        else:
            _menu_state['selected_horizons'] = {6}
        _show_asset_menu()
        return

    # Asset toggle
    if data.startswith('opt_asset_'):
        asset = data.replace('opt_asset_', '')
        if asset == 'all':
            _menu_state['selected_assets'] = set(ASSETS)
        elif asset == 'clear':
            _menu_state['selected_assets'] = set()
        elif asset == 'next':
            if not _menu_state['selected_assets']:
                send_telegram("Select at least one asset.")
                return
            # Skip horizon menu for modes that don't need it
            if _menu_state['mode'] in ('P',):
                _show_horizon_menu()
            else:
                _show_horizon_menu()
            return
        elif asset in ASSETS:
            s = _menu_state['selected_assets']
            if asset in s:
                s.discard(asset)
            else:
                s.add(asset)
        _show_asset_menu()
        return

    # Replay selection
    if data.startswith('opt_replay_'):
        val = data.replace('opt_replay_', '')
        if val == 'next':
            _show_confirm()
            return
        if val in REPLAY_OPTIONS:
            _menu_state['selected_replay'] = REPLAY_OPTIONS[val][0]
        _show_replay_menu()
        return

    # Horizon toggle
    if data.startswith('opt_h_'):
        h = data.replace('opt_h_', '')
        if h == 'next':
            if not _menu_state['selected_horizons']:
                send_telegram("Select at least one horizon.")
                return
            # Show replay menu for modes that support it
            if _menu_state['mode'] in REPLAY_MODES:
                _show_replay_menu()
            else:
                _show_confirm()
            return
        h_int = int(h)
        s = _menu_state['selected_horizons']
        if h_int in s:
            s.discard(h_int)
        else:
            s.add(h_int)
        _show_horizon_menu()
        return

    # Confirm
    if data == 'opt_confirm':
        _enqueue_job()
        _menu_reset()
        return


def _enqueue_job():
    """Create a job from menu state and add to queue."""
    mode = _menu_state['mode']
    assets = sorted(_menu_state['selected_assets'])
    horizons = sorted(_menu_state['selected_horizons'])
    replay = _menu_state.get('selected_replay', 0) if mode in REPLAY_MODES else 0
    job = OptJob(
        id=uuid.uuid4().hex[:8],
        mode=mode,
        assets=assets,
        horizons=horizons,
        replay=replay,
        created_at=datetime.now().strftime('%H:%M:%S'),
    )
    with _queue_lock:
        _job_queue.append(job)
    send_telegram(
        f"<b>Job queued:</b> {job.label()}\n"
        f"ID: <code>{job.id}</code>\n"
        f"Position: #{len(_job_queue)}"
    )
    print(f"  JOB QUEUED: {job.id} {job.label()}")


# ── Job Worker ───────────────────────────────────────────────────────
def _job_worker():
    """Background thread: runs jobs sequentially from the queue."""
    global _current_job
    while not _stop_event.is_set():
        job = None
        with _queue_lock:
            if _job_queue:
                job = _job_queue.pop(0)

        if job is None:
            _stop_event.wait(timeout=3)
            continue

        # Run the job
        with _queue_lock:
            _current_job = job

        job.status = 'running'
        job.started_at = datetime.now().strftime('%H:%M:%S')
        print(f"  JOB START: {job.id} {job.label()}")

        # Send progress message (we'll edit this in-place)
        msg_id = send_telegram(
            f"<b>Running:</b> {job.label()}\n"
            f"Started: {job.started_at}\n"
            f"Phase: Starting..."
        )
        job.progress_msg_id = msg_id

        try:
            _run_job(job)
        except Exception as e:
            job.status = 'failed'
            job.result_summary = str(e)
            print(f"  JOB ERROR: {job.id} {e}")

        job.finished_at = datetime.now().strftime('%H:%M:%S')

        # Send completion message
        if job.status == 'done':
            summary = job.result_summary or 'Completed successfully'
            send_telegram(
                f"<b>Job complete:</b> {job.label()}\n"
                f"Duration: {job.started_at} - {job.finished_at}\n\n"
                f"{summary}"
            )
        elif job.status == 'cancelled':
            send_telegram(f"<b>Job cancelled:</b> {job.label()}")
        elif job.status == 'failed':
            tail = '\n'.join(job.output_lines[-10:]) if job.output_lines else 'No output'
            send_telegram(
                f"<b>Job failed:</b> {job.label()}\n"
                f"<pre>{tail}</pre>"
            )

        # Move to history
        with _queue_lock:
            _current_job = None
            _job_history.append(job)
            if len(_job_history) > 10:
                _job_history.pop(0)

        print(f"  JOB END: {job.id} status={job.status}")


def _run_job(job):
    """Execute optimization subprocess and parse progress."""
    # Build command
    assets_arg = ','.join(job.assets)
    h_arg = ','.join(str(h) for h in job.horizons) + 'h'
    cmd = [PYTHON_EXE, '-u', SCRIPT_PATH, job.mode, assets_arg, h_arg]
    if hasattr(job, 'replay') and job.replay:
        cmd.extend(['--replay', str(job.replay)])

    print(f"  CMD: {' '.join(cmd)}")

    # Run at below-normal priority on Windows
    creation_flags = 0
    if sys.platform == 'win32':
        creation_flags = 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
        cwd=ENGINE_DIR,
    )
    job.process = proc

    # Read stdout in a thread
    last_progress_update = 0
    progress_interval = 30  # seconds between Telegram progress edits
    start_time = time.time()

    # State for progress parsing
    phase = 'starting'
    grid_current = 0
    grid_total = 324
    best_apf = 0
    best_ret = ''
    current_asset = job.assets[0] if job.assets else ''
    current_horizon = job.horizons[0] if job.horizons else 6

    for raw_line in proc.stdout:
        if job.status == 'cancelled':
            proc.terminate()
            break

        line = raw_line.decode('utf-8', errors='replace').rstrip()
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode('ascii', errors='replace').decode('ascii'))
        job.output_lines.append(line)
        # Keep buffer manageable
        if len(job.output_lines) > 500:
            job.output_lines = job.output_lines[-300:]

        # Parse progress patterns
        # Grid eval: [120/324] RF+LGBM w=200 ... | apf=1.23 ret=+2.1%
        m = re.search(r'\[\s*(\d+)/(\d+)\].*apf=([\d.]+).*ret=([+-][\d.]+)%', line)
        if m:
            phase = 'grid'
            grid_current = int(m.group(1))
            grid_total = int(m.group(2))
            apf = float(m.group(3))
            if apf > best_apf:
                best_apf = apf
                best_ret = m.group(4) + '%'

        # Batch summary: [108/324] 33% | ... | best APF=1.52 | ETA 6m
        m = re.search(r'\[\s*(\d+)/(\d+)\]\s+(\d+)%.*best APF=([\d.]+).*ETA\s+([\d.]+)', line)
        if m:
            phase = 'grid'
            grid_current = int(m.group(1))
            grid_total = int(m.group(2))
            best_apf = float(m.group(4))

        # Phase changes
        if 'EXHAUSTIVE GRID' in line:
            phase = 'grid'
            grid_current = 0
            m2 = re.search(r'(\w+)\s+(\d+)h', line)
            if m2:
                current_asset = m2.group(1)
                current_horizon = m2.group(2)
        elif 'STEP 1: BACKTEST MODE D' in line:
            phase = 'backtest D candidates'
        elif 'STEP 2: OPTUNA REFINE' in line:
            phase = 'refining'
        elif 'STEP 3: BACKTEST REFINED' in line:
            phase = 'backtest refined'
        elif 'GRID RESULTS' in line:
            phase = 'grid complete'
        elif 'OVERALL BEST' in line:
            phase = 'done'
            job.result_summary = line.strip()
        elif 'HORIZON SWEEP' in line:
            phase = 'horizon sweep'
        elif re.search(r'HORIZON \d+h \(\d+/\d+\)', line):
            m3 = re.search(r'(\d+)h \((\d+)/(\d+)\)', line)
            if m3:
                current_horizon = m3.group(1)
                phase = f"horizon {m3.group(1)}h ({m3.group(2)}/{m3.group(3)})"

        # Refine new best
        if 'NEW BEST' in line and 'Refin' not in phase:
            m4 = re.search(r'apf=([\d.]+).*ret=([+-][\d.]+)%', line)
            if m4:
                phase = 'refining'

        # Production save
        if 'Production model saved' in line:
            # Extract next line for details
            pass
        if 'Trading config updated' in line:
            pass

        # Extract summary from Mode V
        m5 = re.search(r'OVERALL BEST:.*?→\s+(\w+\s+\d+h)', line)
        if m5:
            job.result_summary = line.strip()

        # Extract production write summary
        m6 = re.search(r'(\w+)\s+(\d+)h:\s+(\S+)\s+w=(\d+)h\s+g=([\d.]+)\s+f=(\d+)\s+conf>=(\d+)%', line)
        if m6:
            asset, h, combo, w, g, f, conf = m6.groups()
            job.result_summary = (
                f"{asset} {h}h: {combo} w={w}h g={g} f={f} conf>={conf}%"
            )

        # Done!
        if line.strip() == 'Done!':
            phase = 'done'

        # Update progress text
        elapsed = (time.time() - start_time) / 60
        if phase == 'grid':
            pct = (grid_current / grid_total * 100) if grid_total > 0 else 0
            job.progress_text = (
                f"Phase: Grid {current_asset} {current_horizon}h\n"
                f"Progress: [{grid_current}/{grid_total}] {pct:.0f}%\n"
                f"Best: APF={best_apf:.3f} ret={best_ret}\n"
                f"Elapsed: {elapsed:.1f} min"
            )
        else:
            job.progress_text = (
                f"Phase: {phase}\n"
                f"Elapsed: {elapsed:.1f} min"
            )

        # Send Telegram progress update (rate limited)
        now = time.time()
        if now - last_progress_update >= progress_interval and job.progress_msg_id:
            edit_telegram_message(
                job.progress_msg_id,
                f"<b>Running:</b> {job.label()}\n\n{job.progress_text}"
            )
            last_progress_update = now

    # Wait for process to finish
    proc.wait()

    if job.status == 'cancelled':
        return

    if proc.returncode == 0:
        job.status = 'done'
        # Try to extract final summary from last 50 lines
        if not job.result_summary:
            for line in reversed(job.output_lines[-50:]):
                if 'OVERALL BEST' in line or 'Production model saved' in line:
                    job.result_summary = line.strip()
                    break
            else:
                job.result_summary = 'Completed successfully'
    else:
        job.status = 'failed'
        job.result_summary = f'Exit code: {proc.returncode}'


# ── Command Handlers ─────────────────────────────────────────────────
def _handle_optimize():
    if _menu_is_active():
        send_telegram("Menu already active. /cancel to reset.")
        return
    _show_mode_menu()


def _handle_queue():
    with _queue_lock:
        lines = []
        if _current_job:
            j = _current_job
            lines.append(f"<b>Running:</b> {j.label()} ({j.id})")
            if j.progress_text:
                lines.append(j.progress_text)
        else:
            lines.append("No job running.")

        if _job_queue:
            lines.append(f"\n<b>Queued ({len(_job_queue)}):</b>")
            for i, j in enumerate(_job_queue, 1):
                lines.append(f"  #{i}: {j.label()} ({j.id})")
        else:
            lines.append("\nNo jobs queued.")

        if _job_history:
            lines.append(f"\n<b>Recent ({len(_job_history)}):</b>")
            for j in reversed(_job_history[-3:]):
                status_icon = {'done': 'OK', 'failed': 'FAIL', 'cancelled': 'X'}.get(j.status, '?')
                lines.append(f"  [{status_icon}] {j.label()} ({j.finished_at})")

    send_telegram('\n'.join(lines))


def _handle_cancel(job_id=None):
    with _queue_lock:
        # Cancel menu if active
        if _menu_is_active() and not job_id:
            _menu_reset()
            send_telegram("Menu cancelled.")
            return

        # Cancel specific queued job
        if job_id:
            for i, j in enumerate(_job_queue):
                if j.id == job_id:
                    j.status = 'cancelled'
                    _job_queue.pop(i)
                    send_telegram(f"Removed from queue: {j.label()} ({j.id})")
                    return
            send_telegram(f"Job {job_id} not found in queue.")
            return

        # Cancel running job
        if _current_job and _current_job.process:
            _current_job.status = 'cancelled'
            try:
                _current_job.process.terminate()
            except Exception:
                pass
            send_telegram(f"Cancelling: {_current_job.label()}")
            return

        send_telegram("Nothing to cancel.")


def _handle_status():
    """Show current production models."""
    try:
        import pandas as pd
        df = pd.read_csv(PRODUCTION_CSV)

        # Load trading config for enabled/active info
        tc = {}
        if os.path.exists(TRADING_CONFIG):
            with open(TRADING_CONFIG) as f:
                tc = json.load(f)

        lines = ['<b>Production Models</b>\n']
        # Group by active trading config
        for _, row in df.iterrows():
            coin = row['coin']
            h = int(row['horizon'])
            combo = row['best_combo']
            w = int(row['best_window'])
            g = float(row['gamma'])
            f_count = int(row['n_features'])
            ret = row.get('return_pct', 0)
            wr = row.get('accuracy', '')
            sampler = row.get('sampler', '')

            # Check if this is the active config for this coin
            active_h = tc.get(coin, {}).get('horizon')
            active_conf = tc.get(coin, {}).get('min_confidence', '')
            enabled = tc.get(coin, {}).get('enabled', False)
            is_active = (active_h == h and enabled)

            marker = ' [LIVE]' if is_active else ''
            ret_str = f"+{ret:.1f}%" if ret > 0 else f"{ret:.1f}%"
            wr_str = f" WR={wr:.0f}%" if wr and wr != '' else ''

            lines.append(
                f"{'*' if is_active else ' '}{coin} {h}h | {combo} w={w}h g={g:.4f} f={f_count}\n"
                f"  ret={ret_str}{wr_str} conf={active_conf}%{marker} [{sampler}]"
            )

        send_telegram('\n'.join(lines))
    except Exception as e:
        send_telegram(f"Error reading production CSV: {e}")


def _handle_results(asset=None):
    """Show last grid/production results for an asset."""
    if not asset:
        send_telegram("Usage: /results BTC")
        return
    asset = asset.upper()
    try:
        import pandas as pd
        # Try grid CSV first
        for h in HORIZONS:
            grid_path = os.path.join(ENGINE_DIR, 'models', f'crypto_ed_grid_{asset}_{h}h.csv')
            if os.path.exists(grid_path):
                df = pd.read_csv(grid_path)
                df = df.sort_values('apf', ascending=False).head(10)
                lines = [f'<b>Grid Results: {asset} {h}h</b> ({len(df)} shown)\n']
                lines.append('<pre>')
                lines.append(f'{"#":>2} {"APF":>6} {"Combo":<10} {"W":>4} {"G":>6} {"F":>2} {"Ret":>7}')
                for i, (_, r) in enumerate(df.iterrows(), 1):
                    ret = f"+{r['return_pct']:.1f}%" if r.get('return_pct', 0) > 0 else f"{r.get('return_pct', 0):.1f}%"
                    lines.append(f"{i:>2} {r['apf']:>6.3f} {r['combo']:<10} {int(r['window']):>4} {r['gamma']:>6.4f} {int(r['n_features']):>2} {ret:>7}")
                lines.append('</pre>')
                send_telegram('\n'.join(lines))
                return
        send_telegram(f"No grid results found for {asset}.")
    except Exception as e:
        send_telegram(f"Error: {e}")


def _handle_help():
    send_telegram(
        "<b>Optimizer Bot Commands</b>\n\n"
        "/optimize — Start optimization (menu)\n"
        "/queue — Show job queue\n"
        "/cancel — Cancel current job/menu\n"
        "/status — Production models\n"
        "/results BTC — Grid results for asset\n"
        "/stop — Stop the bot\n"
        "/help — This message"
    )


# ── Setup ────────────────────────────────────────────────────────────
def _setup():
    """Interactive setup to configure Telegram bot token and chat_id."""
    print("\n  Optimizer Bot Setup")
    print("  " + "=" * 40)
    print("\n  1. Create a new bot via @BotFather on Telegram")
    print("  2. Copy the bot token\n")

    token = input("  Bot token: ").strip()
    if not token:
        print("  Cancelled.")
        return

    # Send a test message to detect chat_id
    print("\n  Now send any message to your bot on Telegram...")
    print("  Waiting for message", end='', flush=True)

    chat_id = None
    for _ in range(60):
        time.sleep(2)
        print('.', end='', flush=True)
        try:
            url = f"https://api.telegram.org/bot{token}/getUpdates"
            req = urllib.request.Request(url, headers={'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=5, context=_ssl_ctx) as resp:
                data = json.loads(resp.read().decode())
            if data.get('ok') and data.get('result'):
                for update in data['result']:
                    msg = update.get('message', {})
                    if msg.get('chat', {}).get('id'):
                        chat_id = str(msg['chat']['id'])
                        break
            if chat_id:
                break
        except Exception:
            pass

    if not chat_id:
        print("\n  Timeout. No message received.")
        return

    print(f"\n  Chat ID detected: {chat_id}")

    # Save config
    config = {'token': token, 'chat_id': chat_id}
    os.makedirs(os.path.dirname(OPTIMIZER_CONFIG_FILE), exist_ok=True)
    with open(OPTIMIZER_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {OPTIMIZER_CONFIG_FILE}")

    # Send test message
    TELEGRAM_CONFIG.update(config)
    send_telegram("Optimizer Bot configured! Use /help for commands.")
    print("  Setup complete!")


# ── Main Loop ────────────────────────────────────────────────────────
def main():
    if '--setup' in sys.argv:
        _setup()
        return

    _load_telegram_config()

    if not TELEGRAM_CONFIG.get('token'):
        print("  ERROR: No Telegram token configured.")
        print("  Run: python crypto_optimizer_bot.py --setup")
        return

    print(f"\n  {'='*50}")
    print(f"  CRYPTO OPTIMIZER BOT")
    print(f"  Machine: {MACHINE}")
    print(f"  Python: {PYTHON_EXE}")
    print(f"  Engine: {ENGINE_DIR}")
    print(f"  {'='*50}\n")

    # Flush old updates
    _flush_old_updates()

    # Start job worker thread
    worker = threading.Thread(target=_job_worker, daemon=True)
    worker.start()

    send_telegram(
        f"<b>Optimizer Bot started</b>\n"
        f"Machine: {MACHINE}\n"
        f"/optimize to begin, /help for commands"
    )

    print("  Bot running. Polling for commands...")

    try:
        while not _stop_event.is_set():
            updates = check_telegram_updates()
            for text, cb_id in updates:
                text_lower = text.lower().strip()

                # Inline button callbacks
                if text.startswith('opt_'):
                    _handle_menu_callback(text)
                    continue

                # Text commands
                if text_lower in ('/optimize', '/start'):
                    _handle_optimize()
                elif text_lower == '/queue':
                    _handle_queue()
                elif text_lower.startswith('/cancel'):
                    parts = text.split()
                    job_id = parts[1] if len(parts) > 1 else None
                    _handle_cancel(job_id)
                elif text_lower == '/status':
                    _handle_status()
                elif text_lower.startswith('/results'):
                    parts = text.split()
                    asset = parts[1] if len(parts) > 1 else None
                    _handle_results(asset)
                elif text_lower == '/help':
                    _handle_help()
                elif text_lower == '/stop':
                    send_telegram("Optimizer Bot stopping...")
                    _stop_event.set()
                    # Cancel running job
                    with _queue_lock:
                        if _current_job and _current_job.process:
                            _current_job.status = 'cancelled'
                            try:
                                _current_job.process.terminate()
                            except Exception:
                                pass
                    break
                elif _menu_is_active():
                    # Unknown input during menu — ignore
                    pass
                else:
                    _handle_help()

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n  Shutting down...")
        _stop_event.set()
        with _queue_lock:
            if _current_job and _current_job.process:
                _current_job.status = 'cancelled'
                try:
                    _current_job.process.terminate()
                except Exception:
                    pass

    worker.join(timeout=10)
    print("  Bot stopped.")


if __name__ == '__main__':
    main()
