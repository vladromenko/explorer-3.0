"""Stall recorder: when the Explorer event loop or a watched block (dataset write) is stuck > 1 s,
write every thread's stack to DATA/logs/stalls.log, so the cause is visible instead of guessed."""
import asyncio
import contextlib
import sys
import threading
import time
import traceback

from .paths import DATA

LOG = DATA / 'logs/stalls.log'
THRESHOLD_S = 0.5
_loop_beat = [time.monotonic()]
_watched = {}
_lock = threading.Lock()
_started = False


def _dump(reason):
    frames = sys._current_frames()
    names = {t.ident: t.name for t in threading.enumerate()}
    lines = [f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} {reason}']
    for ident, frame in frames.items():
        lines.append(f'--- thread {names.get(ident, ident)}')
        lines.extend(l.rstrip() for l in traceback.format_stack(frame)[-14:])
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as f:
        f.write('\n'.join(lines) + '\n')


def note(text):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as f:
        f.write(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} NOTE {text}\n')


@contextlib.contextmanager
def watch(name):
    key = (threading.get_ident(), name)
    with _lock:
        _watched[key] = [time.monotonic(), False]
    try:
        yield
    finally:
        with _lock:
            _watched.pop(key, None)


def _watchdog():
    loop_dumped = False
    while True:
        time.sleep(.2)
        now = time.monotonic()
        stuck = now - _loop_beat[0]
        if stuck > THRESHOLD_S and not loop_dumped:
            loop_dumped = True
            try:
                _dump(f'event loop blocked {stuck:.1f} s')
            except Exception:
                pass
        elif stuck <= THRESHOLD_S:
            loop_dumped = False
        with _lock:
            items = list(_watched.items())
        for key, entry in items:
            if not entry[1] and now - entry[0] > THRESHOLD_S:
                entry[1] = True
                try:
                    _dump(f'{key[1]} running {now - entry[0]:.1f} s')
                except Exception:
                    pass


async def _beat():
    while True:
        _loop_beat[0] = time.monotonic()
        await asyncio.sleep(.1)


def start():
    """Call once from the running event loop (app startup)."""
    global _started
    if _started:
        return
    _started = True
    asyncio.get_running_loop().create_task(_beat())
    threading.Thread(target=_watchdog, name='stall-watch', daemon=True).start()
