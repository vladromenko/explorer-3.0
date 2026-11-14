"""Room inventory while driving: every robot's camera frames go through SAM 3 with a list of
household objects (configs/inventory.yaml, several prompts per vision pass). Everything found is
written to the shared fleet memory with its room position, so «принеси печенье» already knows
where the cookie is. Runs in the background next to any task; never commands motion."""
import threading
import time

import yaml

from .paths import ROOT
from . import world_memory

DEFAULT = [('cookie', 'печенье'), ('cup', 'кружка'), ('bottle', 'бутылка'), ('sock', 'носок'), ('shoe', 'обувь'),
           ('toy', 'игрушка'), ('box', 'коробка'), ('cube', 'кубик')]


def inventory():
    """[(prompt, russian name)]"""
    try:
        items = yaml.safe_load((ROOT / 'configs' / 'inventory.yaml').read_text()).get('objects') or []
        out = [(str(i['prompt']), str(i.get('name') or i['prompt'])) for i in items if i.get('prompt')]
        return out or DEFAULT
    except (OSError, ValueError, AttributeError, TypeError):
        return DEFAULT


class Scout:
    """Feed frames with feed(rgb); a background thread inventories the newest one."""

    BATCH = 6               # prompts per vision pass (~2.3 s on the Orin)

    def __init__(self, robot, cam=None, priority=None, on_found=None, min_score=.5):
        from .goto import CAMERAS, Detector
        self.robot = robot
        self.cam = dict(cam or CAMERAS[robot])
        self.priority = priority            # prompt checked on every pass (the thing being searched for)
        self.on_found = on_found
        self.min_score = min_score
        self.items = inventory()
        self.names = dict(self.items)
        self.cursor = 0
        self.frame = None
        self.frame_at = 0.
        self.seen = {}                       # prompt -> last record time
        self.found = []                      # (prompt, item) found during this run
        self.passes = 0
        self.error = None
        self.detector = Detector.shared()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f'scout-{robot}')
        self.thread.start()

    def set_camera(self, cam):
        self.cam = dict(cam)

    def feed(self, rgb):
        if rgb is not None:
            with self.lock:
                self.frame, self.frame_at = rgb, time.monotonic()

    def stop(self):
        self.done.set()

    def next_prompts(self):
        prompts = [p for p, _ in self.items]
        batch = []
        if self.priority:
            batch.append(self.priority)
        while len(batch) < min(self.BATCH, len(prompts) + (1 if self.priority else 0)):
            p = prompts[self.cursor % len(prompts)]
            self.cursor += 1
            if p not in batch:
                batch.append(p)
        return batch

    def _run(self):
        from .goto import ground_geometry
        while not self.done.is_set():
            with self.lock:
                frame, at = self.frame, self.frame_at
                self.frame = None
            if frame is None or time.monotonic() - at > 1.5:
                self.done.wait(.1)
                continue
            try:
                results, _ = self.detector.detect_many(frame, self.next_prompts(), conf=self.min_score)
                self.error = None
            except Exception as exc:
                self.error = str(exc)
                self.done.wait(2)
                continue
            self.passes += 1
            for prompt, boxes in results.items():
                boxes = [b for b in boxes if b[4] >= self.min_score]
                if not boxes:
                    continue
                best = max(boxes, key=lambda b: b[4])
                d, b, _ = ground_geometry(self.cam, best[:4], frame.shape[0])
                if d > 5:
                    continue                         # far away / above the horizon: position unreliable
                if time.monotonic() - self.seen.get(prompt, 0) < 4 and prompt != self.priority:
                    continue
                self.seen[prompt] = time.monotonic()
                name = self.names.get(prompt, prompt)
                try:
                    item = world_memory.record(self.robot, 'к ' + name, prompt, d, b, frame, best[:4], best[4])
                except Exception:
                    continue
                self.found.append((prompt, item))
                if self.on_found is not None:
                    try:
                        self.on_found(self.robot, prompt, item)
                    except Exception:
                        pass
