"""Shared lidar map: the robots find each other through the room itself, no markers.

Both robots carry 360° lidars (ROSMASTER two MS200 front-right/rear-left, Tank one MS200 in the
centre). Every scan is matched against one occupancy map that both robots build together:

  * ROSMASTER tracks itself with wheel odometry, corrected by scan matching;
  * Tank has no wheel encoders: it finds itself in the same map by global scan matching
    (whole map, all headings) and then tracks locally from scan to scan;
  * a robot's pose in this map is its pose in the shared room frame (world_memory.POSES),
    so «Tank saw a sock 1 m ahead» becomes a point the other robot can drive to.

Wi-Fi only carries the scans to Homebrain, where the map lives. Radio signal strength
(Wi-Fi/Bluetooth RSSI) is not used: it gives ±2–3 m and no direction indoors.
"""
import math
import threading
import time

import cv2
import numpy as np

from .paths import DATA
from . import world_memory

MAP_DIR = DATA / 'world' / 'map'
RES = .05                         # m per cell
SIZE = 400                        # 20 x 20 m, origin in the middle
SIGMA = .06                       # match tolerance (m)

# Lidar mounting in the robot base frame (x forward, y left): (x, y, yaw, direction).
# direction -1: the scan angle runs clockwise.
#   ROSMASTER: 'scan' = /scan0 = laser0 (rear-left), 'scan_rear' = /scan1 = laser1 (front-right); positions
#   from the CubeBot URDF. Orientation measured on the robot (2026-11-14): both MS200 scan clockwise,
#   laser1 is turned by 180°. Chosen by (a) both lidars seeing the same walls and (b) agreement with
#   the depth camera at lidar height; the URDF's yaw 0 / counter-clockwise put a phantom wall in front.
#   Tank: tank.xacro, counter-clockwise (checked against the camera: radiator on the left).
MOUNTS = {
    'rosmaster': {'scan': (-.11617, .09156, math.radians(-2.), -1), 'scan_rear': (.10766, -.09078, math.radians(178.), -1)},
    'tank': {'scan': (-.00168, -.00135, 0., 1)},
}
# Points inside the robot's own body (arm, tracks, posts) are dropped: (x_min, x_max, y_min, y_max).
BODY = {'rosmaster': (-.20, .20, -.18, .18), 'tank': (-.17, .17, -.14, .14)}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def scan_points(robot, packet, max_range=8.):
    """All valid lidar returns of one sensor packet as (N, 2) points in the robot base frame."""
    out = []
    for name, (mx, my, myaw, direction) in MOUNTS[robot].items():
        ranges = packet['arrays'].get(name)
        meta = packet['meta']['streams'].get(name)
        if ranges is None or meta is None:
            continue
        ranges = np.asarray(ranges, np.float64)
        ang = direction * (float(meta['angle_min']) + np.arange(len(ranges)) * float(meta['angle_increment'])) + myaw
        ok = np.isfinite(ranges) & (ranges > max(.05, float(meta.get('range_min') or 0))) & (ranges < max_range)
        r, a = ranges[ok], ang[ok]
        out.append(np.stack([mx + r * np.cos(a), my + r * np.sin(a)], 1))
    if not out:
        return np.zeros((0, 2))
    pts = np.concatenate(out)
    x0, x1, y0, y1 = BODY[robot]
    inside = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
    return pts[~inside]


def thin(pts, cell=.04, limit=360):
    """One point per small cell, at most `limit` points (keeps matching fast and even)."""
    if len(pts) == 0:
        return pts
    _, idx = np.unique(np.floor(pts / cell).astype(np.int64), axis=0, return_index=True)
    pts = pts[np.sort(idx)]
    if len(pts) > limit:
        pts = pts[np.linspace(0, len(pts) - 1, limit).astype(int)]
    return pts


def transform(pts, pose):
    x, y, t = pose
    c, s = math.cos(t), math.sin(t)
    return np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], 1)


class Grid:
    """Hit counts per cell plus a likelihood field (how close each cell is to a wall)."""

    def __init__(self):
        self.hits = np.zeros((SIZE, SIZE), np.float32)
        self.free = np.zeros((SIZE, SIZE), np.float32)     # how often a beam passed through the cell
        self.field = np.zeros((SIZE, SIZE), np.float32)
        self.coarse = {}
        self.scans = 0

    def cells(self, pts):
        return np.floor(pts / RES).astype(np.int64) + SIZE // 2

    def add(self, pts_world, origin=None):
        c = self.cells(pts_world)
        ok = (c >= 0).all(1) & (c < SIZE).all(1)
        if origin is not None:
            o = tuple(int(v) for v in self.cells(np.array([origin[:2]]))[0])
            beams = np.zeros((SIZE, SIZE), np.uint8)
            for x, y in c[ok]:
                cv2.line(beams, o, (int(x), int(y)), 1, 1)
            beams[c[ok, 1], c[ok, 0]] = 0                     # the end cell is the wall itself
            self.free += beams
        np.add.at(self.hits, (c[ok, 1], c[ok, 0]), 1.)
        self.scans += 1
        self.refresh()

    def occupied(self):
        """Walls and furniture: hit often enough compared with beams passing through
        (a person who walked by is cleared by later scans)."""
        return (self.hits >= 1) & (self.hits * 4 >= self.free)

    def known_free(self):
        return (self.free >= 1) & ~self.occupied()

    def refresh(self):
        occupied = self.occupied().astype(np.uint8)
        dist = cv2.distanceTransform(1 - occupied, cv2.DIST_L2, 5) * RES
        self.field = np.exp(-(dist ** 2) / (2 * SIGMA ** 2)).astype(np.float32)
        self.coarse = {}

    def empty(self):
        return self.scans == 0

    def pooled(self, factor):
        """Max-pooled field: an upper bound for coarse global search."""
        if factor not in self.coarse:
            k = 2 * factor + 1
            self.coarse[factor] = cv2.dilate(self.field, np.ones((k, k), np.uint8))
        return self.coarse[factor]

    def score(self, pts, x, y, thetas, field=None):
        """Mean likelihood of `pts` placed at (x, y) for every theta -> (len(thetas),)."""
        field = self.field if field is None else field
        out = np.empty(len(thetas))
        for i, t in enumerate(thetas):
            c = self.cells(transform(pts, (x, y, t)))
            ok = (c >= 0).all(1) & (c < SIZE).all(1)
            out[i] = field[c[ok, 1], c[ok, 0]].sum() / max(1, len(pts))
        return out

    def search(self, pts, center, half_xy, step_xy, thetas, field=None):
        """Brute force over a translation window for each theta. Returns sorted (score, x, y, theta)."""
        field = self.field if field is None else field
        offs = np.arange(-half_xy, half_xy + 1e-9, step_xy)
        gx, gy = np.meshgrid(offs + center[0], offs + center[1])
        gx, gy = gx.ravel(), gy.ravel()
        best = []
        for t in thetas:
            c, s = math.cos(t), math.sin(t)
            rx = c * pts[:, 0] - s * pts[:, 1]
            ry = s * pts[:, 0] + c * pts[:, 1]
            cx = np.floor((gx[:, None] + rx[None]) / RES).astype(np.int64) + SIZE // 2
            cy = np.floor((gy[:, None] + ry[None]) / RES).astype(np.int64) + SIZE // 2
            ok = (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
            vals = np.where(ok, field[np.clip(cy, 0, SIZE - 1), np.clip(cx, 0, SIZE - 1)], 0.).sum(1) / len(pts)
            j = int(vals.argmax())
            best.append((float(vals[j]), float(gx[j]), float(gy[j]), float(t)))
        return sorted(best, reverse=True)

    def refine(self, pts, pose, rounds=((.10, .02, math.radians(6), math.radians(1.5)),
                                         (.03, .01, math.radians(1.5), math.radians(.5)))):
        x, y, t = pose
        best = None
        for half, step, half_t, step_t in rounds:
            thetas = t + np.arange(-half_t, half_t + 1e-9, step_t)
            best = self.search(pts, (x, y), half, step, thetas)[0]
            _, x, y, t = best
        return best[0], (x, y, wrap(t))

    def inliers(self, pts, pose, tol=.08):
        c = self.cells(transform(pts, pose))
        ok = (c >= 0).all(1) & (c < SIZE).all(1)
        return float((self.field[c[ok, 1], c[ok, 0]] > math.exp(-(tol ** 2) / (2 * SIGMA ** 2))).sum() / max(1, len(pts)))

    def conflicts(self, pts, pose, clear=.15):
        """Share of points that land where the map has repeatedly seen free space, far from any
        wall: a pose that puts the scan's walls into known empty floor is wrong."""
        c = self.cells(transform(pts, pose))
        ok = (c >= 0).all(1) & (c < SIZE).all(1)
        cy, cx = c[ok, 1], c[ok, 0]
        far = self.field[cy, cx] < math.exp(-(clear ** 2) / (2 * SIGMA ** 2))
        return float((far & (self.free[cy, cx] >= 2)).sum() / max(1, len(pts)))

    def locate_global(self, pts):
        """Where in the whole map could this scan have been taken? -> (quality, pose, ambiguous)."""
        ys, xs = np.nonzero(self.hits)
        if len(xs) < 20:
            return 0., None, True
        lo = (np.array([xs.min(), ys.min()]) - SIZE // 2) * RES - .5
        hi = (np.array([xs.max(), ys.max()]) - SIZE // 2) * RES + .5
        center = (lo + hi) / 2
        half = float(max(hi - lo) / 2)
        sparse = thin(pts, .1, 120)
        coarse = self.pooled(3)                          # +-3 cells = 15 cm bound
        cands = []
        thetas = np.radians(np.arange(0, 360, 4.))
        for t in thetas:
            cands += self.search(sparse, center, half, .15, [t], coarse)[:1]
        cands.sort(reverse=True)
        results = []
        for sc, x, y, t in cands[:6]:
            q, pose = self.refine(pts, (x, y, t), rounds=((.2, .05, math.radians(5), math.radians(1.)),
                                                         (.06, .015, math.radians(1.5), math.radians(.5)),
                                                         (.02, .005, math.radians(.5), math.radians(.25))))
            inl = self.inliers(pts, pose)
            # Points in known empty floor count against the pose (twice): it rejects «the partition
            # looks like the left wall» kind of matches that a plain overlap score accepts.
            results.append((inl - 2 * self.conflicts(pts, pose), q, pose))
        results.sort(key=lambda r: -r[0])
        best = results[0]
        # Ambiguous when another, clearly different pose explains the scan almost as well
        # (symmetric rooms: rectangle rotated by 180°).
        ambiguous = any(r[0] > best[0] - .08 and (math.dist(r[2][:2], best[2][:2]) > .4 or abs(wrap(r[2][2] - best[2][2])) > .35)
                        for r in results[1:])
        return best[0], best[2], ambiguous


class FleetMap:
    """One shared map; each robot is tracked in it. Thread-safe; fed by FleetLocalizer."""

    def __init__(self):
        self.grid = Grid()
        self.lock = threading.Lock()
        self.track = {}      # robot -> {'pose', 'quality', 'at', 'odom', 'state', 'message'}
        self.last_key = {}
        self.subs = {}       # robot -> own map while it has not found itself in the shared one
        self.contrib = {}    # robot -> scans it added to the shared map

    def status(self):
        with self.lock:
            return {'scans': self.grid.scans, 'contrib': dict(self.contrib), 'robots': {r: {k: v for k, v in s.items() if k not in HIDDEN} for r, s in self.track.items()}}

    def pose(self, robot, max_age=3.):
        """Latest confident pose and the time it was measured, or (None, None)."""
        with self.lock:
            s = self.track.get(robot) or {}
            if s.get('pose') is None or s.get('state') not in ('mapping', 'tracking') or time.time() - s['at'] > max_age:
                return None, None
            return s['pose'], s['at']

    def scan(self, robot, max_age=1.):
        """Latest scan points in the robot base frame (for the collision guard)."""
        with self.lock:
            s = self.track.get(robot) or {}
            if s.get('scan') is None or time.time() - s['at'] > max_age:
                return None
            return s['scan']

    def update(self, robot, packet, odom=None):
        """odom: (x, y, yaw) from wheel encoders if the robot has them. Returns the tracked state."""
        pts = scan_points(robot, packet)
        if len(pts) < 40:
            return self._set(robot, state='no_scan', scan=None, message='Лидар не передаёт данные')
        fine = thin(pts, .04, 360)
        with self.lock:
            s = self.track.get(robot) or {}
            if self.grid.empty():
                pose = (0., 0., 0.)                       # the first robot that scans defines the room frame
                self.grid.add(transform(pts, pose), pose)
                self.last_key[robot] = pose
                self.contrib[robot] = self.contrib.get(robot, 0) + 1
                return self._set(robot, pose=pose, quality=1., state='mapping', odom=odom, scan=fine,
                                 message='Карта начата этим роботом')
            prior = None
            if s.get('pose') is not None and time.time() - s.get('at', 0) < 10:
                prior = s['pose']
                if odom is not None and s.get('odom') is not None:
                    prior = compose(prior, delta(s['odom'], odom))
            if prior is not None:
                # Without wheel encoders (Tank) the robot may have moved/turned more between scans.
                first = (.25, .05, math.radians(12), math.radians(3)) if odom is not None else (.4, .05, math.radians(40), math.radians(4))
                q, pose = self.grid.refine(fine, prior, rounds=(first,
                                                                (.06, .015, math.radians(3), math.radians(.75)),
                                                                (.02, .005, math.radians(.75), math.radians(.25))))
                inl = self.grid.inliers(fine, pose)
                if inl < .45 and odom is None:
                    prior = None                          # lost: fall back to global search
            if prior is None:
                # Global search costs ~1 s of CPU: at most every 4 s per robot. Meanwhile the robot
                # keeps its own small map, which is stitched into the shared one once it is found.
                found = False
                if time.time() - s.get('global_at', 0) >= 4:
                    s['global_at'] = time.time()
                    inl, pose, ambiguous = self.grid.locate_global(fine)
                    found = pose is not None and inl >= .55 and not ambiguous
                    s['global_note'] = ' (комната симметрична, нужно подъехать)' if ambiguous else ''
                    # Two independent global fixes must agree before the robot joins the shared map
                    # (relative to its own map when it has one, else in place).
                    sub = self.subs.get(robot)
                    if found:
                        ref = compose(pose, inverse(sub['pose'])) if sub and sub['pose'] is not None else pose
                        prev = s.get('candidate')
                        agree = prev is not None and math.dist(prev[:2], ref[:2]) < .2 and abs(wrap(prev[2] - ref[2])) < .1
                        s['candidate'] = ref
                        if not agree:
                            found = False
                            s['global_note'] = ' (похоже, узнал место: проверяю ещё раз)'
                    else:
                        s['candidate'] = None
                if not found:
                    return self._submap(robot, pts, fine, odom)
                merged = self._merge_submap(robot, pose)
                if merged:
                    s['merged_note'] = f'своя карта вшита в общую: {merged} сканов'
            # Grow the map only from confident poses, and only after the robot moved.
            moved = self.last_key.get(robot)
            if inl > .6 and (moved is None or math.dist(moved[:2], pose[:2]) > .2 or abs(wrap(moved[2] - pose[2])) > .2):
                self.grid.add(self._without_robots(robot, transform(pts, pose)), pose)
                self.last_key[robot] = pose
            state = 'tracking' if inl > .45 else 'weak'
            note = s.pop('merged_note', '') if time.time() - s.get('global_at', 0) > 8 else s.get('merged_note', '')
            out = self._set(robot, pose=pose, quality=round(inl, 2), state=state, odom=odom, scan=fine, sub_pose=None,
                            message=f'На карте: {inl:.0%} точек совпало' + (f' · {note}' if note else ''))
            self.contrib[robot] = self.contrib.get(robot, 0) + (1 if self.last_key.get(robot) == pose else 0)
        if state == 'tracking':
            world_memory.POSES[robot].fix(*pose, source='lidar')
        return out

    def _without_robots(self, robot, pts_world, radius=.35):
        """Drop returns from the other robots: a robot is not furniture, and leaving it in the
        map would block passages for everybody after it drove on. Caller holds the lock."""
        keep = np.ones(len(pts_world), bool)
        for other, s in self.track.items():
            p = s.get('pose')
            if other != robot and p is not None and time.time() - s.get('at', 0) < 3:
                keep &= np.hypot(pts_world[:, 0] - p[0], pts_world[:, 1] - p[1]) > radius
        return pts_world[keep]

    def _submap(self, robot, pts, fine, odom):
        """Track the robot in its own map (until it recognises the shared one). Caller holds the lock."""
        sub = self.subs.get(robot)
        if sub is None:
            sub = self.subs[robot] = {'grid': Grid(), 'pose': None, 'odom': None, 'keys': [], 'at': 0.}
        if sub['grid'].empty():
            pose = (0., 0., 0.)
        else:
            prior = sub['pose'] if sub['pose'] is not None and time.time() - sub['at'] < 10 else None
            if prior is None:
                sub['grid'] = Grid()                         # lost in its own map too: start it again
                sub['keys'] = []
                pose = (0., 0., 0.)
            else:
                if odom is not None and sub['odom'] is not None:
                    prior = compose(prior, delta(sub['odom'], odom))
                first = (.25, .05, math.radians(12), math.radians(3)) if odom is not None else (.4, .05, math.radians(40), math.radians(4))
                _, pose = sub['grid'].refine(fine, prior, rounds=(first, (.06, .015, math.radians(3), math.radians(.75)),
                                                                  (.02, .005, math.radians(.75), math.radians(.25))))
                if sub['grid'].inliers(fine, pose) < .45:
                    pose = prior
        last = sub['keys'][-1][0] if sub['keys'] else None
        if last is None or math.dist(last[:2], pose[:2]) > .2 or abs(wrap(last[2] - pose[2])) > .2:
            sub['grid'].add(transform(pts, pose), pose)
            sub['keys'].append((pose, thin(pts, .03, 600)))
        sub.update(pose=pose, odom=odom, at=time.time())
        s = self.track.get(robot) or {}
        return self._set(robot, state='submap', pose=None, sub_pose=pose, odom=odom, scan=fine, quality=None,
                         message=f'Строит свою карту ({len(sub["keys"])} сканов), пока не узнает общие стены'
                                 + s.get('global_note', ''))

    def _merge_submap(self, robot, main_pose):
        """The robot was found in the shared map: move its own map into it. Caller holds the lock."""
        sub = self.subs.pop(robot, None)
        if not sub or sub['pose'] is None or not sub['keys']:
            return 0
        T = compose(main_pose, inverse(sub['pose']))      # own-map frame -> shared frame
        for key_pose, pts in sub['keys']:
            pose = compose(T, key_pose)
            self.grid.add(transform(pts, pose), pose)
        self.contrib[robot] = self.contrib.get(robot, 0) + len(sub['keys'])
        return len(sub['keys'])

    def nav(self, robot, max_age=3.):
        """Where the robot can navigate right now: ('main', grid, pose, at) on the shared map, or
        ('sub:<n>', own grid, pose, at) while it only knows its own map; None if neither."""
        with self.lock:
            s = self.track.get(robot) or {}
            if time.time() - s.get('at', 0) > max_age:
                return None
            if s.get('pose') is not None and s.get('state') in ('mapping', 'tracking'):
                return 'main', self.grid, s['pose'], s['at']
            sub = self.subs.get(robot)
            if sub is not None and sub['pose'] is not None:
                return f'sub:{id(sub["grid"])}', sub['grid'], sub['pose'], sub['at']
            return None

    def _set(self, robot, **values):
        s = self.track.setdefault(robot, {})
        s.update(values, at=time.time())
        return {k: v for k, v in s.items() if k not in HIDDEN}

    def relative(self, a, b):
        """Pose of robot b as robot a sees it: (distance, bearing, b's heading relative to a)."""
        pa, pb = (self.track.get(r, {}).get('pose') for r in (a, b))
        if pa is None or pb is None:
            return None
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        return {'distance': math.hypot(dx, dy), 'bearing': wrap(math.atan2(dy, dx) - pa[2]),
                'heading': wrap(pb[2] - pa[2])}

    def image(self, scale=2):
        """Map with both robots drawn, as an RGB array (for the panel)."""
        with self.lock:
            hits = self.grid.occupied().astype(np.float32)
            free = self.grid.known_free()
            track = {r: dict(s) for r, s in self.track.items()}
        ys, xs = np.nonzero(hits + free)
        if len(xs) == 0:
            return None
        x0, x1 = max(0, xs.min() - 20), min(SIZE, xs.max() + 21)
        y0, y1 = max(0, ys.min() - 20), min(SIZE, ys.max() + 21)
        img = np.full((y1 - y0, x1 - x0, 3), (17, 24, 33), np.uint8)
        img[free[y0:y1, x0:x1]] = (31, 44, 58)
        img[(hits[y0:y1, x0:x1] > 0)] = (180, 200, 215)
        for robot, s in track.items():
            for x, y in s.get('path') or []:
                p = (int((x / RES + SIZE // 2) - x0), int((y / RES + SIZE // 2) - y0))
                if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]:
                    img[p[1], p[0]] = (99, 224, 184) if robot == 'tank' else (242, 182, 90)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        colors = {'tank': (95, 212, 168), 'rosmaster': (242, 182, 90)}
        for robot, s in track.items():
            if s.get('pose') is None:
                continue
            x, y, t = s['pose']
            cx = ((x / RES + SIZE // 2) - x0) * scale
            cy = ((y / RES + SIZE // 2) - y0) * scale
            p = (int(cx), int(cy))
            cv2.circle(img, p, int(.15 / RES * scale), colors[robot], 2, cv2.LINE_AA)
            tip = (int(cx + math.cos(t) * .3 / RES * scale), int(cy + math.sin(t) * .3 / RES * scale))
            cv2.arrowedLine(img, p, tip, colors[robot], 2, cv2.LINE_AA, tipLength=.35)
        for item in world_memory.known_objects():
            if item.get('room_xy'):
                x, y = item['room_xy']
                p = (int(((x / RES + SIZE // 2) - x0) * scale), int(((y / RES + SIZE // 2) - y0) * scale))
                cv2.drawMarker(img, p, (255, 123, 114), cv2.MARKER_TILTED_CROSS, 12, 2)
        return np.ascontiguousarray(img[::-1])           # +y up, like a floor plan seen from above

    def save(self):
        MAP_DIR.mkdir(parents=True, exist_ok=True)
        with self.lock:
            tmp = MAP_DIR / 'map.tmp.npz'                  # atomic: a stop mid-write never corrupts the map
            np.savez_compressed(tmp, hits=self.grid.hits, free=self.grid.free, scans=self.grid.scans)
            tmp.replace(MAP_DIR / 'map.npz')

    def load(self):
        try:
            data = np.load(MAP_DIR / 'map.npz')
            data['hits']
        except Exception:                                # missing or damaged file: start a new map
            return False
        with self.lock:
            self.grid.hits = data['hits'].astype(np.float32)
            self.grid.free = data['free'].astype(np.float32) if 'free' in data else np.zeros_like(self.grid.hits)
            self.grid.scans = int(data['scans'])
            self.grid.refresh()
        return True

    def reset(self):
        with self.lock:
            self.grid = Grid()
            self.track.clear()
            self.last_key.clear()
            self.subs.clear()
            self.contrib.clear()
        (MAP_DIR / 'map.npz').unlink(missing_ok=True)


HIDDEN = ('odom', 'scan', 'global_at', 'path', 'global_note', 'merged_note', 'candidate')


def delta(a, b):
    """Motion from odometry pose a to b, expressed in a's frame."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    c, s = math.cos(a[2]), math.sin(a[2])
    return (c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2]))


def inverse(p):
    c, s = math.cos(p[2]), math.sin(p[2])
    return (-c * p[0] - s * p[1], s * p[0] - c * p[1], wrap(-p[2]))


def compose(p, d):
    c, s = math.cos(p[2]), math.sin(p[2])
    return (p[0] + c * d[0] - s * d[1], p[1] + s * d[0] + c * d[1], wrap(p[2] + d[2]))


def odom_pose(packet):
    v = packet['arrays'].get('odometry')
    if v is None or len(v) != 9 or not np.isfinite(v).all():
        return None
    qx, qy, qz, qw = (float(x) for x in v[2:6])
    return (float(v[0]), float(v[1]), math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


FLEET = FleetMap()


class FleetLocalizer:
    """Background reader: both robots' lidars -> FLEET, twice a second. Read-only."""

    SOURCES = {'rosmaster': False, 'tank': True}         # SensorClient(isolated=...) that carries the scan

    def __init__(self, interval=.4):
        self.interval = interval
        self.threads = []
        self.done = threading.Event()
        self.errors = {}

    def running(self):
        return any(t.is_alive() for t in self.threads)

    def start(self):
        if self.running():
            return
        FLEET.load()
        self.done.clear()
        self.threads = [threading.Thread(target=self._run, args=(robot, isolated), daemon=True, name=f'fleet-{robot}')
                        for robot, isolated in self.SOURCES.items()]
        self.threads.append(threading.Thread(target=self._save_loop, daemon=True, name='fleet-save'))
        for t in self.threads:
            t.start()

    def stop(self):
        self.done.set()

    def _save_loop(self):
        while not self.done.wait(30):
            FLEET.save()
        FLEET.save()

    def _run(self, robot, isolated):
        """One reader per robot: a slow Wi-Fi link to one robot does not delay the other."""
        from .sensors import SensorClient
        client = None
        while not self.done.is_set():
            started = time.monotonic()
            try:
                if client is None:
                    client = SensorClient(isolated=isolated, robot=robot)
                packet = client.snapshot(only=('scan', 'scan_rear', 'odometry'))
                FLEET.update(robot, packet, odom_pose(packet) if robot == 'rosmaster' else None)
                self.errors.pop(robot, None)
            except Exception as exc:                      # robot off / out of Wi-Fi: retry, keep the other one going
                self.errors[robot] = str(exc)[:200]
                if client is not None:
                    client.close()
                client = None
                self.done.wait(2)
            self.done.wait(max(.05, self.interval - (time.monotonic() - started)))
        if client is not None:
            client.close()


LOCALIZER = FleetLocalizer()
