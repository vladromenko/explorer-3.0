"""Terminate only this agent; systemd then runs the guarded neutral-command fallback."""
import os,signal,time
from pathlib import Path
root=Path(__file__).resolve().parent
file=root/'agent.pid'
if file.exists():
    pid=int(file.read_text())
    proc=Path('/proc')/str(pid)
    if proc.exists() and (proc/'cwd').resolve()==root and b'uvicorn' in (proc/'cmdline').read_bytes():
        os.kill(pid,signal.SIGTERM)
        deadline=time.monotonic()+8
        while proc.exists() and time.monotonic()<deadline:time.sleep(.1)
        if proc.exists():os.kill(pid,signal.SIGKILL)
    file.unlink(missing_ok=True)
