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

# Russian -> English prompts for the detector's text encoder, matched by word stem so any case
# form works («к носку», «носок», «носки»). Unknown words (and English input) pass through.
STEMS = [
    ('носк', 'sock'), ('носок', 'sock'), ('кубик', 'cube'), ('куб', 'cube'), ('мяч', 'ball'), ('коробк', 'box'),
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
        'возьми', 'найди', 'и', 'the', 'to', 'go', 'a'}


# Camera geometry (MentorPi T1 manufacturer model; replace when measured).
CAMERAS = {'tank': {'height': .139, 'forward': .079, 'pitch_down': 0.0, 'fx': 415.3, 'fy': 416.1, 'cx': 320, 'cy': 200}}


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


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class GoTo:
    """Controller state for one «подъедь к …» request. command(rgb) -> [v, w] for the base."""

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

    def geometry(self, box, height):
        c = self.cam
        u = (box[0] + box[2]) / 2
        bearing = math.atan2(-(u - c['cx']), c['fx'])                    # left of centre = positive (CCW)
        below = math.atan2(box[3] - c['cy'], c['fy']) + math.radians(c['pitch_down'])
        distance = c['height'] / math.tan(below) + c['forward'] if below > math.radians(1.5) else 9.9
        return distance, bearing, box[3] >= height * .97

    # ---- control (called every frame by the session loop)
    def command(self, rgb):
        if not self.active:
            return [0., 0.]
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
            if d <= self.stop_m or at_bottom:
                self.finish('arrived', f'Приехал к «{self.text}»: ~{max(d, .2):.2f} м')
                return [0., 0.]
            self.state = 'approach'
            self.message = f'Вижу «{self.text}»: {d:.2f} м, {math.degrees(b):+.0f}°'
            w = max(-self.w_max, min(self.w_max, 1.6 * b))
            v = 0. if abs(b) > .35 else min(self.v_max, .6 * (d - self.stop_m) + .05)
            self.searched = 0.
            return [v, w]
        # not seen: wait briefly for the detector, then turn in place in steps
        if box is not None and age < 2.5:
            self.state, self.message = 'lost', f'Потерял «{self.text}», жду…'
            return [0., 0.]
        self.state = 'search'
        if self.searched > 2 * math.pi + .5:
            self.finish('not_found', f'Не вижу «{self.text}» вокруг себя')
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
