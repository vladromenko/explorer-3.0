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
        blocked_since = None
        last_cmd = (0., 0.)
        last_depth = 0.
        camera_block = False
        from .sensors import SensorClient
        sensors = SensorClient(robot='rosmaster')
        alive = threading.Event()

        def heartbeat():
            # The session watchdog needs a sign of life every <3 s. The control loop can stall (the
            # first SAM 3 load takes ~20 s): direct commands then expire after 0.3 s and the base
            # stands still, but the session is kept.
            while not alive.wait(.1):
                operator.input(OWNER, rosmaster_control.ZERO)
        threading.Thread(target=heartbeat, daemon=True, name='rosmaster-goto-heartbeat').start()
        try:
            client = rosmaster_control.connection(1)
            while goal.active and not self.stop_event.is_set():
                started = time.monotonic()
                operator.input(OWNER, rosmaster_control.ZERO)          # keeps the session alive
                if operator.owner != OWNER or not operator.lease:
                    goal.cancel(operator.error or 'Сессия ROSMASTER завершена')
                    break
                if started - last_pose > 1.:
                    health = client.get('/health').json()
                    servo = (health.get('observation') or {}).get('arm_measured')
                    if servo and len(servo) == 6 and hasattr(goal, 'set_camera'):
                        goal.set_camera(rosmaster_camera(servo))
                    last_pose = started
                    # The robot's own lidar guard can pin it (a wall along one side blocks both the
                    # turn and the straight drive). Mecanum wheels can still slide away sideways.
                    st = health.get('status') or {}
                    pinned = bool(st.get('guard')) and not any(abs(v) > 1e-3 for v in (st.get('base_command') or [0.]))
                    blocked_since = (blocked_since or started) if pinned and any(abs(v) > 1e-3 for v in last_cmd) else None
                    if blocked_since is not None and started - blocked_since > 2.:
                        self._slide_away(operator)
                        blocked_since = None
                frame = client.get('/frame.jpg')
                if frame.status_code != 200:
                    operator.set_direct({'base': [0., 0., 0.]})
                    time.sleep(.1)
                    continue
                rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                if started - last_depth > .4:
                    last_depth = started
                    near = None
                    try:
                        depth = sensors.snapshot(only=['depth'])['arrays'].get('depth')
                        if depth is not None:
                            metres = depth.astype(np.float32) / 1000.
                            goal.depth = (time.monotonic(), metres)
                            goal.depth_rgb_size = (rgb.shape[1], rgb.shape[0])
                            near = front_clearance(metres)
                    except Exception:
                        near = None
                    camera_block = near is not None and near < CAMERA_STOP_M
                v, w = goal.command(rgb)
                if camera_block and v > 0:
                    # the arm and its camera stick out ahead of the base, above the lidar plane
                    # (cabinet doors, shelves): forward is refused, turning stays allowed
                    v = 0.
                    if hasattr(goal, 'message'):
                        goal.message = f'Перед камерой препятствие в {near:.2f} м: вперёд не еду'
                last_cmd = (v, w)
                operator.set_direct({'base': [float(v), 0., float(w)]})
                time.sleep(max(0., 1 / 15 - (time.monotonic() - started)))
        except Exception as exc:
            self.error = 'ROSMASTER: ' + str(exc)
            goal.cancel(self.error)
        finally:
            alive.set()
            operator.set_direct({'base': [0., 0., 0.]})
            if client is not None:
                client.close()
            if operator.owner == OWNER:
                operator.stop('Подъезд завершён: ' + goal.status()['message'])


CAMERA_STOP_M = .30


def front_clearance(depth_m):
    """Nearest surface in the central part of the arm camera view (3rd percentile: a small near
    object still counts, single noisy pixels do not), metres."""
    h, w = depth_m.shape
    centre = depth_m[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    valid = centre[(centre > .05) & (centre < 4.)]
    return float(np.percentile(valid, 3)) if valid.size > 200 else None


def slide_side(points, reach=.35):
    """+1 = slide left, -1 = right: away from the side with the nearest returns beside the body."""
    if points is None or len(points) == 0:
        return 0
    near = points[(np.abs(points[:, 0]) < reach) & (np.abs(points[:, 1]) < reach)]
    if len(near) == 0:
        return 0
    left = near[near[:, 1] > 0]
    right = near[near[:, 1] < 0]
    dl = float(np.abs(left[:, 1]).min()) if len(left) else 9.
    dr = float(np.abs(right[:, 1]).min()) if len(right) else 9.
    return 1 if dr < dl else -1


def _slide(self, operator, seconds=1.5, speed=.08):
    from .fleet_map import FLEET
    side = slide_side(FLEET.scan('rosmaster'))
    if not side:
        return
    end = time.monotonic() + seconds
    while time.monotonic() < end and not self.stop_event.is_set():
        operator.set_direct({'base': [0., side * speed, 0.]})          # the robot's guard still applies
        time.sleep(.05)
    operator.set_direct({'base': [0., 0., 0.]})


RosmasterGoTo._slide_away = _slide
runner = RosmasterGoTo()
