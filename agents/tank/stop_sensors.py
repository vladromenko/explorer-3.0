"""Stop only the matching read-only observer; never touch the motor agent."""
import os
from pathlib import Path
import signal
import sys

mode=sys.argv[1]
if mode not in ('0','1'):raise ValueError('Invalid observer')
root=Path(__file__).resolve().parent
file=root/f'sensors-{mode}.pid'
if file.exists():
    pid=int(file.read_text());proc=Path('/proc')/str(pid)
    if proc.exists() and (proc/'cwd').resolve()==root:
        args=(proc/'cmdline').read_bytes().split(b'\0')
        if b'sensors.py' in args and b'--localhost' in args and mode.encode() in args:
            os.kill(pid,signal.SIGTERM)
    file.unlink(missing_ok=True)
