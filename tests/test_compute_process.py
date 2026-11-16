import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest

from explorer3 import compute_process


def test_limits_are_required_and_cannot_be_injected(monkeypatch):
    monkeypatch.setattr(compute_process.sys, 'platform', 'linux')
    for limit in (None, True, 0, 100000, '6G'):
        with pytest.raises(ValueError):
            compute_process.command('a' * 32, ['python'], {'memory_limit_mib': limit})
    with pytest.raises(ValueError):
        compute_process.command('--anything', ['python'], {})
    args, unit = compute_process.command('a' * 32, ['python', 'train.py'], {})
    assert 'MemoryMax=6144M' in args and args[-2:] == ['python', 'train.py']
    assert unit.endswith('.scope')
    monkeypatch.setattr(compute_process.sys, 'platform', 'darwin')
    with pytest.raises(RuntimeError):
        compute_process.command('a' * 32, ['python'], {})


@pytest.mark.skipif(os.environ.get('EXPLORER_TEST_SYSTEMD') != '1', reason='requires Linux user systemd')
def test_real_scope_limits_descendants_and_stop_kills_new_session(tmp_path):
    marker = tmp_path / 'child.json'
    child = """
import json, os, pathlib, time
group = pathlib.Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1]
limit = (pathlib.Path('/sys/fs/cgroup') / group.lstrip('/') / 'memory.max').read_text().strip()
pathlib.Path(MARKER).write_text(json.dumps({'pid': os.getpid(), 'limit': limit, 'group': group}))
time.sleep(60)
""".replace('MARKER', repr(str(marker)))
    parent = f'import subprocess,sys; subprocess.run([sys.executable,"-c",{child!r}], start_new_session=True)'
    args, unit = compute_process.command(uuid.uuid4().hex, [sys.executable, '-c', parent], {'memory_limit_mib': 1024})
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        end = time.monotonic() + 10
        while not marker.exists() and proc.poll() is None and time.monotonic() < end:
            time.sleep(.05)
        assert marker.exists(), proc.communicate(timeout=1)
        state = json.loads(marker.read_text())
        assert state['limit'] == str(1024 ** 3)
        assert state['group'].endswith('/' + unit)
        compute_process.stop(unit)
        proc.wait(timeout=5)
        status = Path(f'/proc/{state["pid"]}/status')
        assert not status.exists() or '\nState:\tZ' in status.read_text()
    finally:
        compute_process.stop(unit)
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=5)
