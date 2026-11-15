"""«Подъедь к …»: drive a mobile base to an object named in words, using the robot camera.

YOLOE (open vocabulary) finds the object; distance comes from the bottom edge of its box on
the floor plane (camera height/pitch from the manufacturer model); a simple controller turns
toward it, drives, and stops in front. Search by turning in place when the object is not
seen. Motion only while the operator's Explorer session is active; STOP and the robot's own
limits apply as usual. No training required.
"""
import base64
import json
import math
import subprocess
import threading
import time

import cv2
import numpy as np

from .paths import DATA, ROOT


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

# Russian -> English prompts for the detector's text encoder, matched by word stem so any case
# form works («к носку», «носок», «носки»). Unknown words (and English input) pass through.
STEMS = [
    ('печень', 'cookie'), ('печен', 'cookie'), ('конфет', 'candy'), ('яблок', 'apple'), ('банан', 'banana'),
    ('шоколад', 'chocolate bar'), ('орех', 'nut'), ('мандарин', 'tangerine'), ('апельсин', 'orange fruit'),
    ('носк', 'sock'), ('носок', 'sock'), ('чемодан', 'suitcase'), ('флакон', 'bottle'), ('порошк', 'detergent bottle'), ('кубик', 'cube'), ('куб', 'cube'), ('мяч', 'ball'), ('коробк', 'box'),
    ('коробок', 'box'), ('ящик', 'box'), ('стул', 'chair'), ('кресл', 'armchair'), ('табурет', 'stool'),
    ('бутыл', 'bottle'), ('кружк', 'mug'), ('чашк', 'cup'), ('стакан', 'glass'), ('игрушк', 'toy'),
    ('тап', 'slipper'), ('ботин', 'shoe'), ('туфл', 'shoe'), ('обув', 'shoe'), ('кроссов', 'sneaker'), ('кед', 'sneaker'),
    ('сумк', 'bag'), ('пакет', 'bag'), ('рюкзак', 'backpack'), ('книг', 'book'), ('телефон', 'phone'), ('банк', 'can'),
    ('двер', 'door'), ('стол', 'table'), ('диван', 'sofa'), ('кроват', 'bed'), ('шкаф', 'cabinet'), ('полк', 'shelf'),
    ('кот', 'cat'), ('кошк', 'cat'), ('собак', 'dog'), ('пёс', 'dog'), ('робот', 'robot'), ('человек', 'person'),
    ('джинс', 'jeans'), ('штан', 'trousers'), ('брюк', 'trousers'), ('футболк', 't-shirt'), ('рубашк', 'shirt'),
    ('куртк', 'jacket'), ('кофт', 'sweater'), ('шапк', 'hat'), ('кепк', 'cap'), ('перчатк', 'glove'),
    ('колес', 'wheel'), ('колёс', 'wheel'), ('провод', 'cable'), ('кабел', 'cable'), ('зарядк', 'charger'),
    ('ноутбук', 'laptop'), ('монитор', 'monitor'), ('клавиатур', 'keyboard'), ('мыш', 'computer mouse'),
    ('пульт', 'remote control'), ('лампа', 'lamp'), ('ламп', 'lamp'), ('цвет', 'flower pot'), ('горшк', 'flower pot'),
    ('ведр', 'bucket'), ('корзин', 'basket'), ('мусор', 'trash can'), ('подушк', 'pillow'), ('одеял', 'blanket'),
    ('ковр', 'rug'), ('ковёр', 'rug'), ('полотенц', 'towel'), ('тарелк', 'plate'), ('ложк', 'spoon'), ('ключ', 'keys'),
    ('кошел', 'wallet'), ('очк', 'glasses'), ('часы', 'watch'), ('ручк', 'pen'), ('карандаш', 'pencil'),
    ('ножниц', 'scissors'), ('скотч', 'tape'), ('батаре', 'radiator'), ('окн', 'window'), ('стен', 'wall'),
    ('щит', 'board'), ('метк', 'marker sign'), ('конус', 'cone'), ('кирпич', 'brick'), ('деталь', 'part'),
    ('красн', 'red'), ('син', 'blue'), ('голуб', 'light blue'), ('зелён', 'green'), ('зелен', 'green'),
    ('жёлт', 'yellow'), ('желт', 'yellow'), ('бел', 'white'), ('чёрн', 'black'), ('черн', 'black'),
    ('оранжев', 'orange'), ('розов', 'pink'), ('фиолет', 'purple'), ('сер', 'gray'), ('коричнев', 'brown'),
    ('больш', 'big'), ('маленьк', 'small'), ('деревянн', 'wooden'), ('пластик', 'plastic'),
]
STEMS.sort(key=lambda p: -len(p[0]))      # longest stem wins («коробк» before «кот»)
SKIP = {'подъедь', 'подъехать', 'подъезжай', 'езжай', 'едь', 'поезжай', 'двигайся', 'к', 'ко', 'до', 'на', 'в',
        'возьми', 'найди', 'и', 'the', 'to', 'go', 'a', 'принеси', 'привези', 'подай', 'дай', 'мне', 'найдите',
        'осмотрите', 'ищи', 'ищите', 'где', 'пожалуйста', 'please', 'bring', 'find', 'me'}


# Camera geometry (MentorPi T1 manufacturer model; replace when measured).
CAMERAS = {'tank': {'height': .139, 'forward': .079, 'lateral': 0.0, 'yaw': 0.0, 'pitch_down': 0.0,
                    'fx': 415.3, 'fy': 416.1, 'cx': 320, 'cy': 200},
           # ROSMASTER: camera on the arm; pose recomputed from measured servos (rosmaster_camera)
           'rosmaster': {'height': .35, 'forward': .123, 'lateral': 0.0, 'yaw': 0.0, 'pitch_down': 25.0,
                         'fx': 480.71, 'fy': 480.71, 'cx': 320, 'cy': 240}}


def rosmaster_camera(servo):
    """Camera pose from the measured arm servos with the manufacturer arm model (matches the
    MuJoCo model to the millimetre). Returns the CAMERAS-style dict."""
    from .sim.vendor_rosmaster.arm_model import ArmModel
    P = np.asarray(ArmModel().camera_pose([float(v) for v in servo]))
    fwd = P[:3, 2]                                     # optical z = viewing direction
    cam = dict(CAMERAS['rosmaster'])
    cam.update(height=float(P[2, 3]), forward=float(P[0, 3]), lateral=float(P[1, 3]),
               yaw=math.atan2(fwd[1], fwd[0]), pitch_down=math.degrees(math.asin(max(-1., min(1., -fwd[2])))))
    return cam


def to_prompt(text):
    out = []
    for raw in text.split():
        w = raw.strip(' ,.!?«»"').lower()
        if not w or w in SKIP:
            continue
        for stem, english in STEMS:
            if w.startswith(stem) and len(w) - len(stem) <= 4:
                out.append(english)
                break
        else:
            out.append(w)
    return ' '.join(out).strip() or text.strip()


class Detector:
    """Client for scripts/vision_worker.py (one persistent process per Explorer process)."""

    _instance = None

    @classmethod
    def shared(cls):
        if cls._instance is None or cls._instance.process.poll() is not None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.lock = threading.Lock()
        # SAM 3 (transformers, main env): ~1 s per frame on the Orin GPU but reliable on low,
        # dark floor-level views where the small YOLOE model finds almost nothing.
        self.process = subprocess.Popen([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/vision_worker.py'), 'sam3'],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        text=True, bufsize=1, start_new_session=True)
        ready = json.loads(self.process.stdout.readline() or '{}')
        if not ready.get('ready'):
            raise RuntimeError('Распознаватель не запустился')
        self.device = ready.get('device')

    def detect(self, rgb, prompt, conf=.35):
        ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self.lock:
            self.process.stdin.write(json.dumps({'prompt': prompt, 'conf': conf,
                                                 'jpeg_b64': base64.b64encode(jpeg.tobytes()).decode()}) + '\n')
            self.process.stdin.flush()
            reply = json.loads(self.process.stdout.readline() or '{"error":"распознаватель остановился"}')
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        return reply['boxes'], reply['ms']

    def detect_many(self, rgb, prompts, conf=.45):
        """{prompt: boxes} for many prompts in one vision pass (room inventory)."""
        ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self.lock:
            self.process.stdin.write(json.dumps({'prompts': list(prompts), 'conf': conf,
                                                 'jpeg_b64': base64.b64encode(jpeg.tobytes()).decode()}) + '\n')
            self.process.stdin.flush()
            reply = json.loads(self.process.stdout.readline() or '{"error":"распознаватель остановился"}')
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        return reply['results'], reply['ms']


def ground_geometry(c, box, height):
    """Object foot point on the floor -> (distance, bearing from the robot centre, touches frame bottom)."""
    u = (box[0] + box[2]) / 2
    cam_bearing = math.atan2(-(u - c['cx']), c['fx'])                # left of centre = positive (CCW)
    below = math.atan2(box[3] - c['cy'], c['fy']) + math.radians(c['pitch_down'])
    ground = c['height'] / math.tan(below) if below > math.radians(1.5) else 9.9
    a = c.get('yaw', 0.) + cam_bearing
    x = c['forward'] + ground * math.cos(a)
    y = c.get('lateral', 0.) + ground * math.sin(a)
    return math.hypot(x, y), math.atan2(y, x), box[3] >= height * .97


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class GoTo:
    """Controller state for one «подъедь к …» request. command(rgb) -> [v, w] for the base."""

    def set_camera(self, cam):
        self.cam = cam

    def __init__(self, robot, text, *, stop_m=.30, v_max=.18, w_max=.8, search_w=.6):
        if robot not in CAMERAS:
            raise ValueError('Камера этого робота ещё не описана')
        self.robot, self.text, self.prompt = robot, text, to_prompt(text)
        self.cam = CAMERAS[robot]
        self.stop_m, self.v_max, self.w_max, self.search_w = stop_m, v_max, w_max, search_w
        self.detector = Detector.shared()
        self.box, self.box_time, self.score = None, 0., 0.
        self.frame = None
        self.state, self.message = 'search', f'Ищу «{text}» ({self.prompt})'
        self.started = time.monotonic()
        self.searched = 0.
        self.active = True
        self.distance = self.bearing = None
        self.detect_ms = None
        self.tracker = None
        self.fresh = False
        from . import world_memory
        self.memory = world_memory
        self.pose = world_memory.POSES[robot]
        self.localizer = world_memory.BeaconLocalizer()
        self.last_tick = time.monotonic()
        self.last_cmd = (0., 0.)
        self.last_locate = 0.
        self.last_record = 0.
        self.path = None
        self.planned_at = 0.
        self.roam = None
        self.roams = 0
        from .episodes import Episode
        self.episode = Episode(robot, 'goto', text)
        self.episode.extra['prompt'] = self.prompt
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._detect_loop, daemon=True)
        self.thread.start()

    # ---- perception (background, as fast as the detector allows)
    def _detect_loop(self):
        while self.active:
            with self.lock:
                frame = self.frame
            if frame is None:
                time.sleep(.02)
                continue
            try:
                boxes, ms = self.detector.detect(frame, self.prompt)
            except Exception as exc:
                self.state, self.message, self.active = 'error', 'Распознавание: ' + str(exc), False
                return
            self.detect_ms = ms
            if boxes:
                if self.box is not None and time.monotonic() - self.box_time < 2:
                    pick = max(boxes, key=lambda b: iou(b, self.box) + .3 * b[4])
                else:
                    pick = max(boxes, key=lambda b: b[4])
                with self.lock:
                    self.box, self.box_time, self.score = pick[:4], time.monotonic(), pick[4]
                    self.fresh = True

    depth = None                       # (time, metres HxW) from a depth camera aligned with rgb, if any

    def geometry(self, box, height):
        d, b, at_bottom = ground_geometry(self.cam, box, height)
        # The floor-contact estimate assumes the object stands on the floor; on a shelf it reads too
        # far. A fresh aligned depth image inside the box gives the real distance (camera ->
        # object); the nearer of the two is used.
        if self.depth is not None and time.monotonic() - self.depth[0] < 1.5:
            img = self.depth[1]
            h, w = img.shape
            x1, y1, x2, y2 = [int(v) for v in box]
            sx, sy = w / max(1, self.depth_rgb_size[0]), h / max(1, self.depth_rgb_size[1])
            cx1, cx2 = int((x1 + (x2 - x1) / 4) * sx), int((x2 - (x2 - x1) / 4) * sx)
            cy1, cy2 = int((y1 + (y2 - y1) / 4) * sy), int((y2 - (y2 - y1) / 4) * sy)
            patch = img[max(0, cy1):max(0, cy2), max(0, cx1):max(0, cx2)]
            valid = patch[(patch > .08) & (patch < 4.)]
            if valid.size > 20:
                d = min(d, float(np.median(valid)))
        return d, b, at_bottom

    depth_rgb_size = (640, 480)

    # ---- control (called every frame by the session loop)
    def _bookkeeping(self, rgb):
        """Dead reckoning from the previous command, beacon fix twice a second."""
        now = time.monotonic()
        dt, self.last_tick = min(.3, now - self.last_tick), now
        self.pose.move(self.last_cmd[0] * .85, 0., self.last_cmd[1] * .85, dt)
        if now - self.last_locate > .5:
            self.last_locate = now
            fix = self.localizer.locate(rgb, self.cam)
            if fix:
                self.pose.fix(fix['x'], fix['y'], fix['theta'])

    def _out(self, cmd):
        self.last_cmd = (cmd[0], cmd[1])
        return cmd

    def _memory_drive(self):
        """Head for a position another robot (or this one earlier) reported; None if not possible."""
        item = self.memory.target_for(self.robot, self.prompt)
        if item is None:
            return None
        who = {'tank': 'Tank', 'rosmaster': 'ROSMASTER'}.get(item['robot'], item['robot'])
        if not self.pose.known():
            self.message = f'{who} видел «{self.text}», но я пока не вижу маяков и не знаю, где я: ищу сам'
            return None
        tx, ty = item['room_xy']
        dx, dy = tx - self.pose.x, ty - self.pose.y
        dist = math.hypot(dx, dy)
        if dist < .6:
            return None                                  # close: look for it with the camera
        b = wrap(math.atan2(dy, dx) - self.pose.theta)
        self.state = 'memory'
        pose = (self.pose.x, self.pose.y, self.pose.theta)
        drive = self._map_drive(pose, (tx, ty))
        if drive is not None:
            v, w, remaining, why = drive
            self.message = (f'Объезжаю: {why}' if why else
                            f'Еду по карте к месту, где {who} видел «{self.text}»: {remaining:.1f} м')
            return [v, w]
        self.message = f'Еду к месту, где {who} видел «{self.text}»: {dist:.1f} м, {math.degrees(b):+.0f}°'
        w = max(-self.w_max, min(self.w_max, 1.4 * b))
        v = 0. if abs(b) > .4 else min(self.v_max, .5 * dist)
        return self._guarded([v, w])

    def _map_drive(self, pose, target):
        """Path around furniture on the shared lidar map; None when there is no map here."""
        from . import navigation
        from .fleet_map import FLEET
        if FLEET.grid.empty():
            return None
        now = time.monotonic()
        if self.path is None or now - self.planned_at > 2.:
            self.planned_at = now
            self.path = navigation.plan(FLEET.grid, self.robot, pose[:2], target, allow_unknown=True)
            FLEET.track.setdefault(self.robot, {})['path'] = None if self.path is None else list(self.path)
        if self.path is None:
            return None
        v, w, remaining = navigation.follow(pose, self.path, self.v_max, self.w_max)
        v, w, why = navigation.guard(FLEET.scan(self.robot), self.robot, v, w)
        return v, w, remaining, why

    def _guarded(self, cmd):
        from . import navigation
        from .fleet_map import FLEET
        v, w, why = navigation.guard(FLEET.scan(self.robot), self.robot, cmd[0], cmd[1])
        if why:
            self.message = 'Объезжаю: ' + why
        return [v, w]

    def _roam(self):
        """Nothing around: drive to the next unexplored place on the map and look again."""
        from .explore import Explore
        from .fleet_map import FLEET
        if self.roams >= 4 or FLEET.grid.empty() or not self.pose.known():
            return None
        if self.roam is None:
            self.roam = Explore(self.robot, record=False, v_max=self.v_max)
        cmd = self.roam.command(None)
        if not self.roam.active or self.roam.targets_done:
            done_all = not self.roam.active
            self.roam, self.roams, self.searched = None, self.roams + 1, 0.
            if done_all:
                self.roams = 99                          # whole room seen
            return [0., 0.]
        self.state = 'roam'
        self.message = f'Вокруг «{self.text}» нет: еду осмотреть другое место ({self.roams + 1}/4)'
        return cmd

    def command(self, rgb):
        if not self.active:
            return [0., 0.]
        self._bookkeeping(rgb)
        cmd = self._out(self._command(rgb))
        pose = (self.pose.x, self.pose.y, self.pose.theta) if self.pose.known() else None
        self.episode.step(pose, cmd, self.state, rgb, distance=None if self.distance is None else round(self.distance, 2))
        return cmd

    def _command(self, rgb):
        with self.lock:
            self.frame = rgb
            fresh, self.fresh = self.fresh, False
            box = self.box
        # A fast tracker follows the box between the (≈1 s) detections so the motion is smooth.
        if fresh and box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            if x2 - x1 > 4 and y2 - y1 > 4:
                self.tracker = cv2.TrackerMIL_create()
                self.tracker.init(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (x1, y1, x2 - x1, y2 - y1))
        elif self.tracker is not None and box is not None:
            ok, (x, y, w, h) = self.tracker.update(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if ok:
                with self.lock:
                    self.box = [x, y, x + w, y + h]
                    self.box_time = max(self.box_time, time.monotonic() - .6)   # tracker keeps it alive, detection confirms
                    box = self.box
        with self.lock:
            age = time.monotonic() - self.box_time
        if box is not None and age < 1.2:
            d, b, at_bottom = self.geometry(box, rgb.shape[0])
            self.distance, self.bearing = d, b
            if time.monotonic() - self.last_record > 2. and age < .5:
                self.last_record = time.monotonic()
                try:
                    self.memory.record(self.robot, self.text, self.prompt, d, b, rgb, box, self.score)
                except Exception:
                    pass
            if d <= self.stop_m or at_bottom:
                self.finish('arrived', f'Приехал к «{self.text}»: ~{max(d, .2):.2f} м')
                return [0., 0.]
            self.state = 'approach'
            self.message = f'Вижу «{self.text}»: {d:.2f} м, {math.degrees(b):+.0f}°'
            w = max(-self.w_max, min(self.w_max, 1.6 * b))
            v = 0. if abs(b) > .35 else min(self.v_max, .6 * (d - self.stop_m) + .05)
            self.searched = 0.
            self.roam = None
            return [v, w]
        # not seen: wait briefly for the detector, then turn in place in steps
        if box is not None and age < 2.5:
            self.state, self.message = 'lost', f'Потерял «{self.text}», жду…'
            return [0., 0.]
        drive = self._memory_drive()
        if drive is not None:
            return drive
        self.state = 'search'
        if self.roam is not None or self.searched > 2 * math.pi + .5:
            roam = self._roam()
            if roam is not None:
                return roam
            self.finish('not_found', f'Не вижу «{self.text}» ни вокруг себя, ни в осмотренных местах'
                        if self.roams else f'Не вижу «{self.text}» вокруг себя')
            return [0., 0.]
        phase = (time.monotonic() - self.started) % 1.6                 # turn 0.8 s, hold 0.8 s for a sharp frame
        if phase < .8:
            self.searched += self.search_w * .05
            self.message = f'Ищу «{self.text}» ({self.prompt}): поворачиваюсь'
            return [0., self.search_w]
        self.message = f'Ищу «{self.text}» ({self.prompt}): смотрю'
        return [0., 0.]

    def finish(self, state, message):
        self.state, self.message, self.active = state, message, False
        try:
            self.episode.extra.update(roams=self.roams, final_distance=self.distance)
            self.episode.finish(state, message)
            from .fleet_map import FLEET
            FLEET.track.setdefault(self.robot, {})['path'] = None
        except Exception:
            pass
        try:
            (DATA / 'logs').mkdir(parents=True, exist_ok=True)
            with (DATA / 'logs' / 'goto.jsonl').open('a') as log:
                log.write(json.dumps({'at': time.time(), 'robot': self.robot, 'text': self.text, 'prompt': self.prompt,
                                      'state': state, 'message': message, 'seconds': round(time.monotonic() - self.started, 1)},
                                     ensure_ascii=False) + '\n')
        except OSError:
            pass

    def cancel(self, reason='Отменено'):
        if self.active:
            self.finish('cancelled', reason)

    def status(self):
        return {'state': self.state, 'message': self.message, 'prompt': self.prompt, 'active': self.active,
                'distance': self.distance, 'bearing': self.bearing, 'box': self.box, 'score': self.score,
                'detect_ms': self.detect_ms}
