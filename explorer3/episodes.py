"""Autonomy episodes: every autonomous task (go to an object, explore the room) is written
down by itself — task, robot, outcome, time, path, commands, a few camera frames and the map —
so success rates for the article come straight from the robots' own work.

  DATA/episodes/<id>/episode.json   summary + 5 Hz timeline (pose, command, state)
  DATA/episodes/<id>/frame_XXX.jpg  camera every 2 s (start and end always)
  DATA/episodes/<id>/map.png        shared map at the end
  DATA/episodes/index.jsonl         one summary line per episode
"""
import json
import math
import threading
import time

import cv2

from .paths import DATA

ROOT = DATA / 'episodes'
LOCK = threading.Lock()
SUCCESS = {'arrived', 'explored'}
OPERATOR = {'cancelled', 'intervened'}


class Episode:
    def __init__(self, robot, kind, task):
        self.robot, self.kind, self.task = robot, kind, task
        self.started = time.time()
        self.id = time.strftime('%Y%m%d-%H%M%S', time.localtime(self.started)) + f'-{robot}-{kind}'
        self.dir = ROOT / self.id
        self.timeline = []
        self.frames = 0
        self.last_step = self.last_frame = 0.
        self.closed = False
        self.extra = {}

    def step(self, pose, cmd, state, rgb=None, **info):
        now = time.time()
        if now - self.last_step >= .2:
            self.last_step = now
            self.timeline.append({'t': round(now - self.started, 2),
                                  'pose': None if pose is None else [round(float(v), 3) for v in pose],
                                  'cmd': [round(float(v), 3) for v in cmd], 'state': state,
                                  **{k: v for k, v in info.items() if v is not None}})
        if rgb is not None and (self.frames == 0 or now - self.last_frame >= 2.) and self.frames < 120:
            self._frame(rgb)
        self.last_rgb = rgb

    def _frame(self, rgb):
        self.dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.dir / f'frame_{self.frames:03d}.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        self.frames += 1
        self.last_frame = time.time()

    def finish(self, outcome, message):
        if self.closed:
            return None
        self.closed = True
        if getattr(self, 'last_rgb', None) is not None:
            self._frame(self.last_rgb)
        poses = [s['pose'] for s in self.timeline if s['pose'] is not None]
        path = sum(math.dist(a[:2], b[:2]) for a, b in zip(poses, poses[1:]))
        summary = {'id': self.id, 'robot': self.robot, 'kind': self.kind, 'task': self.task,
                   'outcome': outcome, 'success': outcome in SUCCESS, 'operator_stopped': outcome in OPERATOR,
                   'message': message, 'started': self.started, 'seconds': round(time.time() - self.started, 1),
                   'path_m': round(path, 2), 'localized_share': round(len(poses) / max(1, len(self.timeline)), 2),
                   'frames': self.frames, **self.extra}
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            from .fleet_map import FLEET
            img = FLEET.image(3)
            if img is not None:
                cv2.imwrite(str(self.dir / 'map.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        except Exception:
            pass
        (self.dir / 'episode.json').write_text(json.dumps({**summary, 'timeline': self.timeline}, ensure_ascii=False))
        with LOCK:
            with (ROOT / 'index.jsonl').open('a') as f:
                f.write(json.dumps(summary, ensure_ascii=False) + '\n')
        return summary


def load(limit=None):
    try:
        lines = (ROOT / 'index.jsonl').read_text().splitlines()
    except OSError:
        return []
    items = []
    for line in lines:
        try:
            items.append(json.loads(line))
        except ValueError:
            continue
    items.sort(key=lambda i: -i['started'])
    return items[:limit] if limit else items


def stats():
    """Per robot and task kind: episodes, success rate, operator stops, median time."""
    groups = {}
    for item in load():
        groups.setdefault((item['robot'], item['kind']), []).append(item)
    out = []
    for (robot, kind), items in sorted(groups.items()):
        times = sorted(i['seconds'] for i in items if i['success'])
        out.append({'robot': robot, 'kind': kind, 'episodes': len(items),
                    'success_rate': round(sum(i['success'] for i in items) / len(items), 2),
                    'operator_stops': sum(i['operator_stopped'] for i in items),
                    'median_success_s': times[len(times) // 2] if times else None})
    return out
