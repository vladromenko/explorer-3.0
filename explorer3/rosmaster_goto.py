"""«Подъехать к …» on ROSMASTER: the same controller as Tank (explorer3.goto), driving the base
through the existing leased operator session with source='policy' (agent clamps and STOP
apply). The camera is on the arm, so its pose is recomputed from the measured servos."""
import threading
import time

import cv2
import numpy as np

from . import rosmaster_control
from .goto import GoTo, rosmaster_camera

OWNER = 'goto'


class RosmasterGoTo:
    def __init__(self):
        self.goal = None
        self.thread = None
        self.error = None
        self.stop_event = threading.Event()

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def status(self):
        if self.goal is None:
            return {'message': self.error or '', 'active': False}
        return {**self.goal.status(), 'message': self.error or self.goal.status()['message']}

    def start(self, text=None, task=None):
        """text: «подъехать к …»; task: any goal object with command(rgb) (e.g. explore.Explore)."""
        operator = rosmaster_control.operator
        if self.running():
            raise RuntimeError('ROSMASTER уже выполняет задание')
        if operator.lease:
            raise RuntimeError('Сначала выключите ручное управление ROSMASTER (STOP в его панели)')
        rosmaster_control.ensure_agent()
        goal = task if task is not None else GoTo('rosmaster', text)
        operator.input(OWNER, rosmaster_control.ZERO)
        try:
            operator.enable(OWNER, 'rosmaster_dock', 'cartesian', True, source='policy')
        except Exception:
            goal.cancel('Не удалось включить управление')
            raise
        self.goal, self.error = goal, None
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return goal

    def cancel(self, reason='Отменено оператором'):
        self.stop_event.set()
        if self.goal is not None:
            self.goal.cancel(reason)

    def _run(self):
        operator, goal = rosmaster_control.operator, self.goal
        client = None
        last_pose = 0.
        try:
            client = rosmaster_control.connection(1)
            while goal.active and not self.stop_event.is_set():
                started = time.monotonic()
                operator.input(OWNER, rosmaster_control.ZERO)          # keeps the session alive
                if operator.owner != OWNER or not operator.lease:
                    goal.cancel(operator.error or 'Сессия ROSMASTER завершена')
                    break
                if started - last_pose > 1. and hasattr(goal, 'set_camera'):
                    servo = (client.get('/health').json().get('observation') or {}).get('arm_measured')
                    if servo and len(servo) == 6:
                        goal.set_camera(rosmaster_camera(servo))
                    last_pose = started
                frame = client.get('/frame.jpg')
                if frame.status_code != 200:
                    operator.set_direct({'base': [0., 0., 0.]})
                    time.sleep(.1)
                    continue
                rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                v, w = goal.command(rgb)
                operator.set_direct({'base': [float(v), 0., float(w)]})
                time.sleep(max(0., 1 / 15 - (time.monotonic() - started)))
        except Exception as exc:
            self.error = 'ROSMASTER: ' + str(exc)
            goal.cancel(self.error)
        finally:
            operator.set_direct({'base': [0., 0., 0.]})
            if client is not None:
                client.close()
            if operator.owner == OWNER:
                operator.stop('Подъезд завершён: ' + goal.status()['message'])


runner = RosmasterGoTo()
