"""SO-101 Leader (read-only, LeRobot) -> ROSMASTER Console 8090, started on demand from that console.

Runs inside the Explorer service on Homebrain. The Leader is opened only while 8090 is connected
and the stream is requested; Console 8090 maps it with its own MirrorMapper (absolute mode).
CubeBot's absolute gripper is 0 = open, the Leader's calibrated 0 = closed, so it is inverted here.
"""
import json
import threading
import time
from fastapi import Request
from nicegui import app
from .hardware import JOINTS
from .leader import LeaderReader
from .paths import DATA


class Leader8090:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.done = threading.Event()
        self.state = {'running': False, 'frames': 0, 'error': None, 'since': None}

    def start(self, host):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return dict(self.state)
            self.done.clear()
            self.state = {'running': True, 'frames': 0, 'error': None, 'since': time.time()}
            self.thread = threading.Thread(target=self.run, args=(host,), daemon=True)
            self.thread.start()
            return dict(self.state)

    def stop(self):
        self.done.set()
        thread = self.thread
        if thread:
            thread.join(3)
        return dict(self.state)

    def run(self, host):
        from websockets.sync.client import connect
        reader = None
        try:
            with connect(f'ws://{host}:8090/mirror', open_timeout=3, close_timeout=1) as ws:
                reader = LeaderReader()
                reader.start()
                while not self.done.is_set():
                    tick = time.monotonic()
                    values = dict(zip(JOINTS, reader.sample()['positions']))
                    values['gripper'] = 100. - values['gripper']
                    ws.send(json.dumps({'leader': values, 't': time.time()}, allow_nan=False))
                    self.state['frames'] += 1
                    self.done.wait(max(0., 1/30 - (time.monotonic() - tick)))
        except Exception as exc:
            self.state['error'] = f'{type(exc).__name__}: {exc}'[:300]
        finally:
            if reader is not None:
                reader.close()
            self.state['running'] = False


leader8090 = Leader8090()


def robot_host():
    return json.loads((DATA/'configs/rosmaster-agent.json').read_text())['host']


@app.post('/api/rosmaster/leader8090/start')
def start_8090(request: Request):
    return leader8090.start(robot_host())


@app.post('/api/rosmaster/leader8090/stop')
def stop_8090(request: Request):
    return leader8090.stop()


@app.get('/api/rosmaster/leader8090')
def status_8090(request: Request):
    return dict(leader8090.state)


@app.on_shutdown
def shutdown_8090():
    leader8090.done.set()
