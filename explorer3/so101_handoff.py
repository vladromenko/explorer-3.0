"""SO-101 hand-over: take the object ROSMASTER put on the hand-over pad and hold it out to the
person. Poses are taught once with the leader arm (Explorer teleoperation): «над площадкой»,
«взять» (gripper open), «сжать» (only the gripper value), «отдать человеку», «исходная».
The wrist camera confirms the object (SAM 3) and the arm self-calibrates its pan to centre it:
a small trial pan shows how the object moves in the image, then the pan is corrected — no
camera-to-arm calibration needed. Runs through HardwareSource (all joint/speed limits apply).
"""
import json
import math
import threading
import time

from .paths import DATA

CONFIG = DATA / 'configs' / 'handoff.json'
POSES = ('above', 'grasp', 'grip', 'give', 'home')
NAMES = {'above': 'над площадкой', 'grasp': 'взять (захват открыт)', 'grip': 'сжать (только захват)',
         'give': 'отдать человеку', 'home': 'исходная'}


def load():
    try:
        return json.loads(CONFIG.read_text())
    except (OSError, ValueError):
        return {}


def save(value):
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    tmp.replace(CONFIG)


def teach(key, value):
    cfg = load()
    if key.startswith('so101.'):
        name = key.split('.', 1)[1]
        if name not in POSES:
            raise ValueError('Неизвестная поза')
        if len(value) != 6 or not all(math.isfinite(float(v)) for v in value):
            raise ValueError('Нужны шесть суставов')
        cfg.setdefault('so101', {})[name] = [round(float(v), 2) for v in value]
    else:
        cfg[key] = value
    cfg.setdefault('taught_at', {})[key] = time.time()
    save(cfg)
    return cfg


def missing():
    cfg = load()
    out = [f'SO-101: {NAMES[p]}' for p in POSES if p not in (cfg.get('so101') or {})]
    if 'rosmaster_spot' not in cfg:
        out.append('ROSMASTER: место передачи на карте')
    if 'rosmaster_place' not in cfg:
        out.append('ROSMASTER: поза руки «положить на площадку»')
    return out


def pan_correction(err_before, err_after, trial_deg, limit=15.):
    """Pan (deg) that centres the object, from a trial pan: image error is ~linear in pan."""
    slope = (err_after - err_before) / trial_deg
    if abs(slope) < 1e-3:
        return 0.
    return max(-limit, min(limit, -err_after / slope))


def gripper_holding(command, measured, tol=4.):
    """A position-servo gripper stops short of its command when something is between the jaws."""
    return abs(measured - command) > tol


class Handoff:
    """One hand-over run on the real SO-101 (source=HardwareSource-like for tests)."""

    def __init__(self, prompt, source=None, detector=None, log=None, speed_deg_s=25., wait_take_s=20.):
        self.prompt = prompt
        self.cfg = load()
        self.poses = self.cfg.get('so101') or {}
        if any(p not in self.poses for p in POSES):
            raise RuntimeError('SO-101: сначала запомните позы передачи (' + ', '.join(missing()) + ')')
        if source is None:
            from .hardware import HardwareSource
            source = HardwareSource(teleoperate=False)
        self.source = source
        self.detector = detector
        self.log = log or (lambda text: None)
        self.speed = speed_deg_s
        self.wait_take_s = wait_take_s
        self.taken = threading.Event()
        self.stop_event = threading.Event()
        self.started = False

    def stop(self):
        self.stop_event.set()

    def _check(self):
        if self.stop_event.is_set():
            raise InterruptedError('Остановлено')

    def move(self, target, settle=.4):
        """Straight joint-space move at self.speed deg/s (HardwareSource clamps each step too)."""
        obs = self.source.observe(require_images=False)
        start = list(obs['observation.state'])
        span = max(abs(a - b) for a, b in zip(start[:5], target[:5]))
        steps = max(1, int(span / self.speed * 30))
        for i in range(1, steps + 1):
            self._check()
            goal = [a + (b - a) * i / steps for a, b in zip(start, target)]
            obs = self.source.observe(require_images=False)
            self.source.send(goal, obs)
            time.sleep(1 / 30)
        end = time.monotonic() + settle
        while time.monotonic() < end:
            self._check()
            obs = self.source.observe(require_images=False)
            self.source.send(list(target), obs)
            time.sleep(1 / 30)
        return self.source.observe(require_images=False)['observation.state']

    def find(self):
        """(normalised horizontal error of the object in the wrist image, score) or None."""
        if self.detector is None:
            from .goto import Detector
            self.detector = Detector.shared()
        for _ in range(3):
            self._check()
            obs = self.source.observe()
            frame = obs['images'].get('wrist') or next(iter(obs['images'].values()), None)
            rgb = frame['image'] if isinstance(frame, dict) else frame
            boxes, _ = self.detector.detect(rgb, self.prompt, conf=.4)
            if boxes:
                b = max(boxes, key=lambda x: x[4])
                w = rgb.shape[1]
                return ((b[0] + b[2]) / 2 - w / 2) / (w / 2), b[4]
            time.sleep(.7)
        return None

    def run(self):
        P = {k: list(v) for k, v in self.poses.items()}
        if not self.started:
            self.source.start(explicit_user_start=True, stop=self.stop_event.is_set)
            self.started = True
        try:
            self.log('SO-101: смотрю на площадку')
            self.move(P['above'])
            seen = self.find()
            if seen is None:
                self.move(P['home'])
                return {'ok': False, 'reason': 'на площадке передачи предмет не виден'}
            err, _ = seen
            pan = 0.
            if abs(err) > .08:                                # self-calibrated pan centring
                trial = 4. if err < 0 else -4.
                probe = list(P['above'])
                probe[0] += trial
                self.move(probe)
                again = self.find()
                if again is not None:
                    pan = trial + pan_correction(err, again[0], trial)
            for key in ('above', 'grasp'):
                P[key][0] += pan
            self.log(f'SO-101: беру (поправка поворота {pan:+.1f}°)')
            self.move(P['above'])
            self.move(P['grasp'])
            closed = list(P['grasp'])
            closed[5] = P['grip'][5]
            state = self.move(closed, settle=.8)
            if not gripper_holding(closed[5], state[5]):
                opened = list(P['grasp'])
                self.move(opened)
                self.move(P['home'])
                return {'ok': False, 'reason': 'SO-101 не взял предмет (захват сомкнулся пустым)'}
            lift = list(P['above'])
            lift[5] = closed[5]
            self.move(lift)
            give = list(P['give'])
            give[5] = closed[5]
            self.move(give)
            self.log('SO-101: держу — заберите предмет')
            end = time.monotonic() + self.wait_take_s
            while time.monotonic() < end and not self.taken.is_set():
                self._check()
                obs = self.source.observe(require_images=False)
                self.source.send(give, obs)
                time.sleep(1 / 30)
            opened = list(P['give'])
            opened[5] = P['grasp'][5]
            self.move(opened, settle=1.2)
            self.move(P['home'])
            if not self.taken.is_set():
                # holding it out and opening on a timeout is not a hand-over: nobody confirmed it
                return {'ok': False, 'reason': 'получение не подтверждено кнопкой «Забрал»', 'pan_correction': pan,
                        'taken_by_button': False}
            return {'ok': True, 'pan_correction': pan, 'taken_by_button': True}
        finally:
            try:
                self.source.close()
            except Exception:
                pass
