"""Tank speed limits and an optional obstacle stop. No I/O, ROS or hardware construction.

Only two things always act: speed clamps (factory recommendation |v|<=0.6 m/s,
|w|<=2 rad/s, MentorPi T1 manual ch. 3.4) and the command TTL in the agent (the factory
manual: closing a program does not stop the chassis, a zero command is required).
Everything else is optional and off by default.
"""
import json
import math
from pathlib import Path
import time

DEFAULTS = {
    'version': 2,
    'manual': {'v': .3, 'w': 1.0},
    'policy': {'v': .3, 'w': 1.0},
    'absolute': {'v': .6, 'w': 2.0},
    'acceleration': {'v': 1.5, 'w': 6.0},
    'command_ttl_s': .5,
    'obstacle_stop': False,          # optional: stop when the lidar sees something closer than lidar_stop_m
    'lidar_stop_m': .15,
    'autonomous_stop_m': .25,        # swarm/policy sessions always stop: 0.17 m lidar->nose + 0.08 m
    'sector_half_angle_deg': 30.,
    'battery_warning_v': 7.0,        # warning only
    'battery_stop_v': 6.2,           # deep-discharge protection (factory: recharge below 6.4 V)
}


def load(root):
    value = json.loads(json.dumps(DEFAULTS))
    path = Path(root) / 'configs/tank-limits.json'
    if path.exists():
        try:
            user = json.loads(path.read_text())
            if user.get('version') == 2:
                for key, default in DEFAULTS.items():
                    if key in user and type(user[key]) is type(default):
                        value[key] = user[key]
        except (OSError, ValueError):
            pass
    value['absolute']['v'] = min(value['absolute']['v'], .6)
    value['absolute']['w'] = min(value['absolute']['w'], 2.)
    value['command_ttl_s'] = min(max(value['command_ttl_s'], .2), 1.)
    return value


def clip(value, v, w, mode='manual'):
    if not all(math.isfinite(x) for x in (v, w)):
        raise ValueError('Нечисловая скорость Tank')
    group = value['manual'] if mode == 'manual' else value['policy']
    caps = [min(group[k], value['absolute'][k]) for k in ('v', 'w')]
    return tuple(max(-c, min(c, x)) for x, c in zip((v, w), caps))


class Ramp:
    """Mild acceleration limit (smooths current spikes); zero is applied immediately."""

    def __init__(self, value, clock=time.monotonic):
        self.value = value
        self.clock = clock
        self.reset()

    def reset(self):
        self.previous = (0., 0.)
        self.updated = self.clock()

    def apply(self, v, w):
        now = self.clock()
        dt = max(0., min(.1, now - self.updated))
        self.updated = now
        self.previous = tuple(0. if target == 0. else old + max(-a * dt, min(a * dt, target - old))
                              for target, old, a in zip((v, w), self.previous,
                                                        (self.value['acceleration']['v'], self.value['acceleration']['w'])))
        return self.previous


def sector_distance(ranges, angle_min, increment, range_min, range_max, *, rear=False, half_angle=30.):
    selected = []
    center = math.pi if rear else 0.
    for index, distance in enumerate(ranges):
        angle = angle_min + increment * index
        delta = math.atan2(math.sin(angle - center), math.cos(angle - center))
        if abs(delta) <= math.radians(half_angle) and math.isfinite(distance) and range_min <= distance <= range_max:
            selected.append(float(distance))
    return min(selected) if selected else None


def tilt_degrees(acceleration):
    if acceleration is None or len(acceleration) != 3 or not all(math.isfinite(v) for v in acceleration):
        return None
    x, y, z = acceleration
    if math.sqrt(x * x + y * y + z * z) < 1.:
        return None
    return math.degrees(math.atan2(math.hypot(x, y), z))


def guard(value, v, w, *, battery=None, front=None, rear=None, autonomous=False, **_):
    """Returns (command, reason). Blocks only on deep discharge or, if enabled, a close obstacle
    in the direction of travel. Missing sensors never block driving."""
    if v == 0. and w == 0.:
        return (0., 0.), None
    if battery is not None and math.isfinite(battery) and 0 < battery < value['battery_stop_v']:
        return (0., 0.), f"Батарея {battery:.2f} В: зарядите Tank"
    # Manual driving: optional (off by default, the operator sees the robot). Autonomous sessions
    # (swarm, trained policies): always, at a distance measured from the lidar to beyond the nose.
    if (value['obstacle_stop'] or autonomous) and v != 0.:
        distance = front if v > 0 else rear
        limit = max(value['lidar_stop_m'], value['autonomous_stop_m']) if autonomous else value['lidar_stop_m']
        if distance is not None and distance < limit:
            return (0., w), 'Препятствие впереди' if v > 0 else 'Препятствие сзади'
    return (v, w), None


def depth_obstacle(points, *, half_width=.13, min_height=.02, max_height=.30, max_range=1.0, min_points=12, rng=None):
    """Nearest obstacle ahead from the depth camera's metric cloud (optical frame: x right,
    y down, z forward). The floor plane is fitted on every cloud (RANSAC), so the camera tilt does
    not need to be known: points 2–30 cm above the floor inside the robot's width are obstacles.
    Low things the 2D lidar misses (chair legs and wheels, bags) are what this is for.
    Returns (forward distance in m or None, floor_found)."""
    import numpy as np
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(1) & (pts[:, 2] > .05) & (pts[:, 2] < 3.)]
    if len(pts) < 50:
        return None, False
    rng = rng or np.random.default_rng(0)
    best, best_n = None, 0
    near = pts[pts[:, 2] < 2.]
    lower = near[near[:, 1] >= np.percentile(near[:, 1], 50)] if len(near) > 30 else pts  # floor: lower half of the view
    for _ in range(120):
        a, b, c = lower[rng.choice(len(lower), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        if n[1] < 0:
            n = -n
        if n[1] < math.cos(math.radians(50)):                       # must look like a floor (y down)
            continue
        inliers = int((np.abs((pts - a) @ n) < .012).sum())
        if inliers > best_n:
            best, best_n = (a, n), inliers
    if best is None or best_n < max(60, .05 * len(pts)):
        # no floor in view: whatever is inside the robot's corridor this close is above the floor
        sel = (np.abs(pts[:, 0]) < half_width) & (pts[:, 2] < max_range)
        if sel.sum() < min_points:
            return None, False
        return float(np.sort(pts[sel, 2])[min_points - 1]), False
    a, n = best
    height = -((pts - a) @ n)                                        # above the floor (y is down)
    # forward / lateral in the floor plane
    fwd = np.array([0., 0., 1.]) - n * n[2]
    fwd /= np.linalg.norm(fwd)
    side = np.cross(n, fwd)
    f, s = pts @ fwd, pts @ side
    sel = (height > min_height) & (height < max_height) & (np.abs(s) < half_width) & (f > 0) & (f < max_range)
    if sel.sum() < min_points:
        return None, True
    return float(np.sort(f[sel])[min_points - 1]), True
