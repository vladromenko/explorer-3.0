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

# Lidar mounting in the robot base frame (x forward, y left): (x, y, yaw).
MOUNTS = {
    'rosmaster': {'scan': (.10766, -.09078, 0.), 'scan_rear': (-.11617, .09156, 0.)},   # CubeBot URDF, laser1/laser0
    'tank': {'scan': (-.00168, -.00135, 0.)},                                           # mentorpi tank.xacro
}
# Points inside the robot's own body (arm, tracks, posts) are dropped: (x_min, x_max, y_min, y_max).
BODY = {'rosmaster': (-.24, .24, -.18, .18), 'tank': (-.17, .17, -.14, .14)}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def scan_points(robot, packet, max_range=8.):
    """All valid lidar returns of one sensor packet as (N, 2) points in the robot base frame."""
    out = []
    for name, (mx, my, myaw) in MOUNTS[robot].items():
        ranges = packet['arrays'].get(name)
        meta = packet['meta']['streams'].get(name)
        if ranges is None or meta is None:
            continue
        ranges = np.asarray(ranges, np.float64)
        ang = float(meta['angle_min']) + np.arange(len(ranges)) * float(meta['angle_increment']) + myaw
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
        self.field = np.zeros((SIZE, SIZE), np.float32)
        self.coarse = {}
        self.scans = 0

    def cells(self, pts):
        return np.floor(pts / RES).astype(np.int64) + SIZE // 2

    def add(self, pts_world):
        c = self.cells(pts_world)
        ok = (c >= 0).all(1) & (c < SIZE).all(1)
        np.add.at(self.hits, (c[ok, 1], c[ok, 0]), 1.)
        self.scans += 1
        self.refresh()

    def refresh(self):
        occupied = (self.hits >= 1).astype(np.uint8)
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
            results.append((self.inliers(pts, pose), q, pose))
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

    def status(self):
        with self.lock:
            return {'scans': self.grid.scans, 'robots': {r: {k: v for k, v in s.items() if k != 'odom'} for r, s in self.track.items()}}

    def update(self, robot, packet, odom=None):
        """odom: (x, y, yaw) from wheel encoders if the robot has them. Returns the tracked state."""
        pts = scan_points(robot, packet)
        if len(pts) < 40:
            return self._set(robot, state='no_scan', message='Лидар не передаёт данные')
        fine = thin(pts, .04, 360)
        with self.lock:
            s = self.track.get(robot) or {}
            if self.grid.empty():
                pose = (0., 0., 0.)                       # the first robot that scans defines the room frame
                self.grid.add(transform(fine, pose))
                self.last_key[robot] = pose
                return self._set(robot, pose=pose, quality=1., state='mapping', odom=odom,
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
                # Global search costs ~1 s of CPU: at most every 4 s per robot.
                if time.time() - s.get('global_at', 0) < 4:
                    return {k: v for k, v in s.items() if k != 'odom'}
                s['global_at'] = time.time()
                inl, pose, ambiguous = self.grid.locate_global(fine)
                if pose is None or inl < .55 or ambiguous:
                    return self._set(robot, state='searching', quality=round(inl, 2), odom=odom, pose=None,
                                     message='Ищет себя на общей карте' + (' (комната симметрична, нужно подъехать)' if ambiguous else ''))
            # Grow the map only from confident poses, and only after the robot moved.
            moved = self.last_key.get(robot)
            if inl > .6 and (moved is None or math.dist(moved[:2], pose[:2]) > .2 or abs(wrap(moved[2] - pose[2])) > .2):
                self.grid.add(transform(fine, pose))
                self.last_key[robot] = pose
            state = 'tracking' if inl > .45 else 'weak'
            out = self._set(robot, pose=pose, quality=round(inl, 2), state=state, odom=odom,
                            message=f'На карте: {inl:.0%} точек совпало')
        if state == 'tracking':
            world_memory.POSES[robot].fix(*pose, source='lidar')
        return out

    def _set(self, robot, **values):
        s = self.track.setdefault(robot, {})
        s.update(values, at=time.time())
        return {k: v for k, v in s.items() if k != 'odom'}

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
            hits = self.grid.hits.copy()
            track = {r: dict(s) for r, s in self.track.items()}
        ys, xs = np.nonzero(hits)
        if len(xs) == 0:
            return None
        x0, x1 = max(0, xs.min() - 20), min(SIZE, xs.max() + 21)
        y0, y1 = max(0, ys.min() - 20), min(SIZE, ys.max() + 21)
        img = np.full((y1 - y0, x1 - x0, 3), (17, 24, 33), np.uint8)
        img[(hits[y0:y1, x0:x1] > 0)] = (180, 200, 215)
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
            np.savez_compressed(MAP_DIR / 'map.npz', hits=self.grid.hits, scans=self.grid.scans)

    def load(self):
        try:
            data = np.load(MAP_DIR / 'map.npz')
        except (OSError, ValueError):
            return False
        with self.lock:
            self.grid.hits = data['hits'].astype(np.float32)
            self.grid.scans = int(data['scans'])
            self.grid.refresh()
        return True

    def reset(self):
        with self.lock:
            self.grid = Grid()
            self.track.clear()
            self.last_key.clear()
        (MAP_DIR / 'map.npz').unlink(missing_ok=True)


def delta(a, b):
    """Motion from odometry pose a to b, expressed in a's frame."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    c, s = math.cos(a[2]), math.sin(a[2])
    return (c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2]))


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

    def __init__(self, interval=.5):
        self.interval = interval
        self.thread = None
        self.done = threading.Event()
        self.errors = {}

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        if self.running():
            return
        FLEET.load()
        self.done.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name='fleet-localizer')
        self.thread.start()

    def stop(self):
        self.done.set()

    def _run(self):
        from .sensors import SensorClient
        clients, last_save = {}, time.monotonic()
        while not self.done.is_set():
            started = time.monotonic()
            for robot, isolated in self.SOURCES.items():
                try:
                    if robot not in clients:
                        clients[robot] = SensorClient(isolated=isolated, robot=robot)
                    packet = clients[robot].snapshot()
                    FLEET.update(robot, packet, odom_pose(packet) if robot == 'rosmaster' else None)
                    self.errors.pop(robot, None)
                except Exception as exc:                  # robot off / out of Wi-Fi: keep the other one going
                    self.errors[robot] = str(exc)[:200]
                    client = clients.pop(robot, None)
                    if client is not None:
                        client.close()
            if time.monotonic() - last_save > 30:
                FLEET.save()
                last_save = time.monotonic()
            self.done.wait(max(.05, self.interval - (time.monotonic() - started)))
        for client in clients.values():
            client.close()
        FLEET.save()


LOCALIZER = FleetLocalizer()
