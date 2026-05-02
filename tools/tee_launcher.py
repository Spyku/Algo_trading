"""
Tee launcher: run a command, show output in terminal AND save to log file.
Handles Unicode, Ctrl+C, and Windows encoding properly.

Usage: python tee_launcher.py <logfile> <command> [args...]
"""
import sys
import os
import subprocess
import signal
from datetime import datetime

if len(sys.argv) < 3:
    print("Usage: python tee_launcher.py <logfile> <command> [args...]")
    sys.exit(1)

logfile = sys.argv[1]
cmd = sys.argv[2:]

os.makedirs(os.path.dirname(logfile), exist_ok=True)

proc = None

def handle_sigint(sig, frame):
    if proc:
        proc.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

with open(logfile, 'a', encoding='utf-8') as f:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    try:
        for raw_line in proc.stdout:
            line = raw_line.decode('utf-8', errors='replace')
            # Write to log (UTF-8, always works)
            f.write(line)
            f.flush()
            # Write to terminal (may need ASCII fallback on Windows)
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except UnicodeEncodeError:
                sys.stdout.write(line.encode('ascii', errors='replace').decode('ascii'))
                sys.stdout.flush()
    except KeyboardInterrupt:
        proc.terminate()

    proc.wait()
    sys.exit(proc.returncode)
