"""ROSMASTER arm skills for the fleet: pick a named object from the floor and place what it holds.

Perception: SAM 3 box on the arm camera + the aligned depth image -> 3D point -> robot base frame
through the manufacturer arm model (camera pose from the measured servos). Grasp: CubeBot's
recipe (open, hover above, descend, close past the contact angle, lift, carry pose, in-hand check
from the gripper stall). Motion goes through the existing leased agent session: the base in the
'rosmaster_dock' profile, the arm in 'rosmaster_pick_place' (servo targets, agent rate limits).
All geometry is pure functions so it is tested without the robot.
"""
import math
import threading
import time

import numpy as np

from .sim.vendor_rosmaster.arm_model import ArmModel

OWNER = 'fleet-arm'
SEARCH_NEAR = [90, 130, 0, 5, 90, 30]          # camera ~45° down: floor from ~0.25 m (CubeBot)
SEARCH_FAR = [90, 130, 0, 25, 90, 30]
GRIP_OPEN, GRIP_CLOSE, GRIP_MAX, OVERDRIVE = 30, 125, 165, 55
OPEN_WIDTH_M = .11
STEP_DEG = 8.                                    # per command, under the agent's 12° tracking limit
REACH_X = (.16, .26)                             # floor grasp window in front of the base (IK, measured)
ARM = ArmModel()


# ------------------------------------------------------------------ geometry (pure)
def object_in_base(depth_m, box, servo, K):
    """3D grasp point in the base frame from a detection box and the aligned depth image.
    Returns dict(point, top_z, width_m) or None when the box has no valid depth."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = depth_m.shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    # central part of the box: the object, not the floor around it
    cx1, cx2 = x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4
    cy1, cy2 = y1 + (y2 - y1) // 4, y2 - (y2 - y1) // 4
    patch = depth_m[cy1:cy2, cx1:cx2]
    vs, us = np.mgrid[cy1:cy2, cx1:cx2]
    ok = np.isfinite(patch) & (patch > .1) & (patch < 2.)
    if ok.sum() < 10:
        return None
    z = patch[ok]
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    pts_cam = np.stack([(us[ok] - cx) / fx * z, (vs[ok] - cy) / fy * z, z, np.ones_like(z)], 0)
    T = np.asarray(ARM.camera_pose([float(v) for v in servo]))
    pts = (T @ pts_cam)[:3].T
    centre = np.median(pts, 0)
    top = float(np.percentile(pts[:, 2], 90))
    width = float((x2 - x1) * np.median(z) / fx)
    return {'point': centre, 'top_z': top, 'width_m': width}


def grasp_target(obj):
    """Where the fingers close: object centre, at half its height (at least 1.5 cm above the floor)."""
    x, y, _ = obj['point']
    return np.array([x, y, max(.015, min(.08, obj['top_z'] / 2))])


def grip_command(width_m):
    """Gripper angle that clamps an object of this width (contact angle + overdrive, CubeBot)."""
    width = min(max(width_m, .01), OPEN_WIDTH_M)
    contact = GRIP_CLOSE - width / OPEN_WIDTH_M * (GRIP_CLOSE - GRIP_OPEN)
    return int(round(min(GRIP_MAX, contact + OVERDRIVE)))


def plan_grasp(target, seed):
    """(hover, grasp) servo poses, or (None, reason) when the arm cannot reach from here."""
    seed = [float(v) for v in list(seed)[:5]] + [90.]
    sol, pitch = ARM.grasp_ik(list(target), seed)
    if sol is None:
        return None, 'вне досягаемости руки'
    hover_target = np.array(target) + [0, 0, .06]
    hover, _ = ARM.grasp_ik(list(hover_target), list(sol), pitches=[pitch, pitch - 10, pitch + 10])
    if hover is None:
        hover = list(sol)
    grasp = [float(v) for v in list(sol)[:5]] + [GRIP_OPEN]
    hover = [float(v) for v in list(hover)[:5]] + [GRIP_OPEN]
    return (hover, grasp), None


def base_correction(target):
    """How the base should move so the object lands in the reach window: (forward m, turn rad)."""
    x, y = float(target[0]), float(target[1])
    turn = math.atan2(y, x) if abs(y) > .04 else 0.
    reach = math.hypot(x, y)
    mid = sum(REACH_X) / 2
    forward = reach - mid if not (REACH_X[0] <= reach <= REACH_X[1]) else 0.
    return forward, turn


def holding(command_grip, measured_grip):
    """The servo stalls short of the command on an object and reaches it when empty (CubeBot)."""
    return measured_grip < command_grip - 6


def learned_pick_job():
    """The newest ROSMASTER floor-pick training whose «Допуск к роботу» passed for its current
    weights, or None (then the geometric grasp is used)."""
    try:
        from . import catalog, deploy_gate
        from .training_drafts import checkpoint_hash
        for job in catalog.rows('jobs'):
            if job['profile'] != 'rosmaster_pick_place' or job['status'] != 'completed' or not job['checkpoint']:
                continue
            report = deploy_gate.latest(job['id'])
            if report and report['verdict'] == 'ready' and report['checkpoint_bundle_sha256'] == checkpoint_hash(job['checkpoint']):
                return job['id']
    except Exception:
        return None
    return None


# ------------------------------------------------------------------ live skills
class ArmSkills:
    """Pick/place on the real robot. Runs synchronously in the caller's thread; stop() aborts."""

    def __init__(self, log=None):
        from . import rosmaster_control
        self.rc = rosmaster_control
        self.op = rosmaster_control.operator
        self.log = log or (lambda text: None)
        self.stop_event = threading.Event()
        self.profile = None
        self.client = None

    def stop(self):
        self.stop_event.set()

    def _check(self):
        if self.stop_event.is_set():
            raise InterruptedError('Остановлено')
        if self.profile and (self.op.owner != OWNER or not self.op.lease):
            raise RuntimeError(self.op.error or 'Сессия ROSMASTER завершена агентом')

    def use(self, profile):
        if self.profile == profile:
            return
        self.release()
        if self.op.lease:
            raise RuntimeError('ROSMASTER занят другим управлением')
        self.rc.ensure_agent()
        if not self.rc.request('/health', timeout=4).get('hardware_open'):
            self.rc.request('/acquire', {}, 45)          # take the serial port from the native bridge
        self.op.input(OWNER, self.rc.ZERO)
        self.op.enable(OWNER, profile, 'joint' if profile == 'rosmaster_pick_place' else 'cartesian', True, source='policy')
        self.profile = profile
        if getattr(self, 'beat', None) is None or not self.beat.is_alive():
            # sign of life for the session watchdog while SAM 3 loads/recognizes (no motion implied)
            def beat():
                while self.profile is not None and not self.stop_event.is_set():
                    self.op.input(OWNER, self.rc.ZERO)
                    time.sleep(.1)
            self.beat = threading.Thread(target=beat, daemon=True, name='rosmaster-arm-heartbeat')
            self.beat.start()
        if self.client is None:
            self.client = self.rc.connection(2)

    def release(self):
        if self.profile is not None:
            try:
                self.op.set_direct(None)
                if self.op.owner == OWNER:
                    self.op.stop('Навык руки завершён')
            finally:
                self.profile = None
                time.sleep(.3)

    def close(self):
        self.release()
        if self.client is not None:
            self.client.close()
            self.client = None

    def measured(self):
        obs = (self.client.get('/health').json().get('observation') or {})
        arm = obs.get('arm_measured')
        if not arm or len(arm) != 6:
            raise RuntimeError('Нет измерений руки')
        return [float(v) for v in arm]

    def _wait(self, seconds, tick=None):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._check()
            self.op.input(OWNER, self.rc.ZERO)          # heartbeat
            if tick is not None:
                tick()
            time.sleep(.1)

    def move(self, servo, timeout=8., tol=3.):
        self.use('rosmaster_pick_place')
        target = [float(v) for v in servo]
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self._check()
            self.op.input(OWNER, self.rc.ZERO)
            m = self.measured()
            if max(abs(a - b) for a, b in zip(m[:5], target[:5])) < tol and abs(m[5] - target[5]) < 3 * tol:
                return m
            # the agent refuses policy targets more than 12° from the measured arm: walk there
            step = [mv + max(-STEP_DEG, min(STEP_DEG, tv - mv)) for mv, tv in zip(m, target)]
            self.op.set_direct({'arm': step})
            time.sleep(.1)
        return self.measured()

    def drive(self, forward_m=0., turn_rad=0., v=.06, w=.4):
        self.use('rosmaster_dock')
        for value, speed, cmd in ((turn_rad, w, lambda s: [0., 0., s]), (forward_m, v, lambda s: [s, 0., 0.])):
            if abs(value) < 1e-3:
                continue
            duration = abs(value) / speed
            sign = 1. if value > 0 else -1.
            end = time.monotonic() + duration
            while time.monotonic() < end:
                self._check()
                self.op.input(OWNER, self.rc.ZERO)
                self.op.set_direct({'base': cmd(sign * speed)})
                time.sleep(.05)
            self.op.set_direct({'base': [0., 0., 0.]})
            time.sleep(.4)

    def look(self, prompt, poses=(SEARCH_NEAR, SEARCH_FAR)):
        """Arm camera + depth -> object in the base frame, or None."""
        from .goto import Detector
        from .sensors import SensorClient
        sensors = SensorClient(robot='rosmaster')
        try:
            for pose in poses:
                self.move(pose)
                self._wait(.5)
                packet = sensors.snapshot()
                rgb, depth = packet['arrays'].get('rgb'), packet['arrays'].get('depth')
                if rgb is None or depth is None:
                    continue
                info = packet['meta']['streams'].get('rgb_info') or {}
                K = np.array(info.get('k') or [480.7, 0, 322.3, 0, 480.7, 245.6, 0, 0, 1], float).reshape(3, 3)
                boxes, _ = Detector.shared().detect(rgb, prompt, conf=.4)
                if not boxes:
                    continue
                best = max(boxes, key=lambda b: b[4])
                obj = object_in_base(depth.astype(np.float32) / 1000., best[:4], self.measured(), K)
                if obj is not None:
                    obj['score'] = best[4]
                    obj['rgb'] = rgb
                    obj['box'] = best[:4]
                    return obj
            return None
        finally:
            sensors.close()

    def pick(self, prompt, attempts=3):
        """Grasp `prompt`: with a learned policy that passed «Допуск к роботу» when there is one
        (explorer3/deploy_gate.py), otherwise the geometric grasp. Returns a result dict."""
        job = learned_pick_job()
        if job is not None:
            return self.policy_pick(prompt, job)
        return self.geometric_pick(prompt, attempts)

    def policy_pick(self, prompt, job, seconds=25.):
        """The recording-block-3 setting, then the learned policy through the robot runtime:
        the arm camera puts the object 5–10 cm ahead (as in the demonstrations), the policy runs
        from the observation pose at the profile rate, the gripper stall decides «holding»."""
        from . import deploy_gate
        from .policy_runtime import PolicyCore
        from .profiles import profile
        report = deploy_gate.latest(job)
        cfg = profile('rosmaster_pick_place')
        obj = self.look(prompt)
        if obj is None:
            return {'ok': False, 'reason': 'предмет не виден камерой руки'}
        forward, turn = base_correction(grasp_target(obj))
        if abs(forward) > .01 or abs(turn) > .02:
            self.log(f'захват моделью: ставлю предмет в зону показов, подъезжаю {forward:+.2f} м')
            self.drive(forward, turn)
        self.move(SEARCH_FAR)
        core = PolicyCore(report['checkpoint'], 'cuda', instruction=f'pick up the {prompt}')
        self.log(f'захват моделью {report["policy"].upper()} {job[:8]} (допуск пройден)')
        import cv2
        end = time.monotonic() + seconds
        last = None
        while time.monotonic() < end:
            started = time.monotonic()
            self._check()
            self.op.input(OWNER, self.rc.ZERO)
            frame = self.client.get('/frame.jpg')
            rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            state = self.measured()
            action = core.act(state, {cfg['camera']: rgb}, cfg['task'])
            self.op.set_direct({'arm': [float(v) for v in action]})      # the agent rate-limits every joint
            last = action
            time.sleep(max(0., 1 / cfg['fps'] - (time.monotonic() - started)))
        measured = self.measured()
        grip = int(last[5]) if last is not None else GRIP_CLOSE
        if holding(grip, measured[5]) and measured[5] > GRIP_OPEN + 10:
            return {'ok': True, 'grip': grip, 'policy_job': job}
        carry = list(SEARCH_FAR)
        carry[5] = GRIP_OPEN
        self.move(carry)
        return {'ok': False, 'reason': f'модель {job[:8]} не удержала предмет', 'policy_job': job}

    def geometric_pick(self, prompt, attempts=3):
        """Find `prompt` in front of the robot and grasp it (CubeBot recipe). Returns a result dict."""
        extra = 0
        for attempt in range(1, attempts + 1):
            self.log(f'захват: попытка {attempt}, ищу предмет камерой руки')
            obj = self.look(prompt)
            if obj is None:
                return {'ok': False, 'reason': 'предмет не виден камерой руки'}
            target = grasp_target(obj)
            plan, why = plan_grasp(target, self.measured())
            if plan is None:
                forward, turn = base_correction(target)
                self.log(f'захват: {why}; подъезжаю {forward:+.2f} м, поворот {math.degrees(turn):+.0f}°')
                self.drive(forward, turn)
                continue
            hover, grasp = plan
            self.move(hover)
            self.move(grasp, timeout=6., tol=4.)
            grip = min(GRIP_MAX, grip_command(obj['width_m']) + extra)
            closing = list(grasp)
            for value in range(GRIP_OPEN, grip + 1, 5):          # close in steps, like CubeBot
                closing[5] = value
                self.op.set_direct({'arm': closing})
                self._wait(.18)
            closing[5] = grip
            self.op.set_direct({'arm': closing})
            self._wait(.5)
            lift = list(hover)
            lift[5] = grip
            self.move(lift)
            carry = list(SEARCH_FAR)
            carry[5] = grip
            measured = self.move(carry)
            if holding(grip, measured[5]):
                return {'ok': True, 'grip': grip, 'object': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                                            for k, v in obj.items() if k not in ('rgb',)}}
            self.log('захват: пустой захват, повторяю сильнее')
            extra += 5
            carry[5] = GRIP_OPEN
            self.move(carry)
        return {'ok': False, 'reason': f'не удалось взять за {attempts} попытки'}

    def place(self, place_pose, grip):
        """Put the held object down at a taught arm pose and open the hand."""
        pose = [float(v) for v in place_pose]
        above = list(pose)
        above[1] = max(0., above[1] - 15)                         # come from slightly higher
        above[5] = pose[5] = grip
        self.move(above)
        self.move(pose, timeout=6., tol=4.)
        opened = list(pose)
        opened[5] = GRIP_OPEN
        self.move(opened)
        self._wait(.6)
        retreat = list(SEARCH_FAR)
        self.move(retreat)
        return {'ok': True}
