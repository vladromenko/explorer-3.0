"""Shared world memory for the fleet: what was seen, by whom, when and — when the robot knows
where it is — where in the room. Room frame comes from printed ArUco «beacons» on the walls
(configs/room_beacons.yaml); between beacon sightings each robot dead-reckons (ROSMASTER from
wheel odometry, Tank from its own commands). One robot's sighting becomes another robot's goal.
"""
import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .paths import DATA, ROOT

LOCK = threading.Lock()
MEMORY = DATA / 'world' / 'memory.json'
SNAPSHOTS = DATA / 'world' / 'snapshots'


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------ beacons and robot poses
def beacons():
    """{id: {'x','y','z','facing'}} — facing = direction the printed side looks (radians, room frame)."""
    path = ROOT / 'configs' / 'room_beacons.yaml'
    try:
        value = yaml.safe_load(path.read_text()) or {}
    except OSError:
        return {}
    out = {}
    for item in value.get('beacons') or []:
        try:
            out[int(item['id'])] = {'x': float(item['x']), 'y': float(item['y']), 'z': float(item.get('z', .3)),
                                    'facing': math.radians(float(item['facing_deg'])), 'side': float(item.get('side', .20))}
        except (KeyError, TypeError, ValueError):
            continue
    return out


class Pose:
    """Robot pose in the room frame: beacon fixes + dead reckoning between them."""

    def __init__(self, robot):
        self.robot = robot
        self.x = self.y = self.theta = None
        self.fixed_at = None
        self.source = 'unknown'
        self.drift_m = 0.

    def known(self):
        return self.x is not None

    def fix(self, x, y, theta):
        self.x, self.y, self.theta = x, y, theta
        self.fixed_at, self.source, self.drift_m = time.time(), 'beacon', 0.

    def move(self, vx, vy, w, dt):
        if not self.known():
            return
        c, s = math.cos(self.theta), math.sin(self.theta)
        self.x += (c * vx - s * vy) * dt
        self.y += (s * vx + c * vy) * dt
        self.theta = wrap(self.theta + w * dt)
        self.drift_m += math.hypot(vx, vy) * dt * .1 + abs(w) * dt * .05
        self.source = 'dead_reckoning'

    def as_dict(self):
        return {'robot': self.robot, 'x': self.x, 'y': self.y, 'theta': self.theta, 'source': self.source,
                'fixed_at': self.fixed_at, 'drift_m': round(self.drift_m, 2)}


POSES = {'tank': Pose('tank'), 'rosmaster': Pose('rosmaster')}


class BeaconLocalizer:
    """Every ArUco beacon in view gives the camera pose; the robot centre follows from the
    camera mounting (same camera dicts as explorer3.goto)."""

    def __init__(self):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)

    def locate(self, rgb, cam):
        table = beacons()
        if not table:
            return None
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if ids is None:
            return None
        K = np.array([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1.]])
        fixes = []
        for c, i in zip(corners, ids.ravel()):
            b = table.get(int(i))
            if b is None:
                continue
            h = b['side'] / 2
            obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, c.reshape(4, 2).astype(np.float64), K, np.zeros(5),
                                                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not n:
                continue
            best = None
            for rvec, tvec, err in zip(rvecs, tvecs, np.ravel(errs)):
                R, _ = cv2.Rodrigues(rvec)
                into = -R[:, 2]
                score = abs(into[1]) + (0. if into @ tvec.ravel() > 0 else 10.) + .01 * float(err)
                if best is None or score < best[0]:
                    best = (score, R, tvec.ravel())
            _, R, t = best
            into = -R[:, 2]
            d = math.hypot(t[0], t[2])                     # camera -> beacon, ground plane
            bearing_cam = math.atan2(-t[0], t[2])           # beacon left of the optical axis = +
            facing_into_cam = math.atan2(-into[0], into[2])  # direction into the wall, camera frame
            # Room frame: «into the wall» is facing + pi. Camera heading follows.
            cam_heading = wrap(b['facing'] + math.pi - facing_into_cam)
            cx = b['x'] - d * math.cos(cam_heading + bearing_cam)
            cy = b['y'] - d * math.sin(cam_heading + bearing_cam)
            theta = wrap(cam_heading - cam.get('yaw', 0.))
            x = cx - (cam['forward'] * math.cos(theta) - cam.get('lateral', 0.) * math.sin(theta))
            y = cy - (cam['forward'] * math.sin(theta) + cam.get('lateral', 0.) * math.cos(theta))
            fixes.append((d, x, y, theta, int(i)))
        if not fixes:
            return None
        d, x, y, theta, beacon = min(fixes)                  # the closest beacon is the most accurate
        return {'x': x, 'y': y, 'theta': theta, 'beacon': beacon, 'distance': d}


# ------------------------------------------------------------------ object memory
def _load():
    try:
        return json.loads(MEMORY.read_text())
    except (OSError, ValueError):
        return []


def _save(items):
    MEMORY.parent.mkdir(parents=True, exist_ok=True)
    tmp = MEMORY.with_suffix('.tmp')
    tmp.write_text(json.dumps(items[-500:], ensure_ascii=False, indent=1))
    tmp.replace(MEMORY)


def record(robot, text, prompt, distance, bearing, rgb=None, box=None, score=None):
    """A robot saw `prompt`; store the sighting (room position when the robot's pose is known)."""
    pose = POSES[robot]
    item = {'at': time.time(), 'robot': robot, 'text': text, 'prompt': prompt, 'distance': round(distance, 2),
            'bearing': round(bearing, 3), 'score': None if score is None else round(float(score), 2),
            'room_xy': None, 'robot_pose': pose.as_dict() if pose.known() else None}
    if pose.known():
        a = pose.theta + bearing
        item['room_xy'] = [round(pose.x + distance * math.cos(a), 2), round(pose.y + distance * math.sin(a), 2)]
        item['uncertainty_m'] = round(.05 + .1 * distance + pose.drift_m, 2)
    if rgb is not None:
        SNAPSHOTS.mkdir(parents=True, exist_ok=True)
        name = f"{int(item['at'] * 1000)}-{robot}.jpg"
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (120, 230, 180), 3)
        cv2.imwrite(str(SNAPSHOTS / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        item['snapshot'] = name
    with LOCK:
        items = _load()
        items.append(item)
        _save(items)
    return item


def known_objects(max_age_s=3600):
    """Latest sighting per object name (newest first)."""
    latest = {}
    now = time.time()
    for item in _load():
        if now - item['at'] > max_age_s:
            continue
        key = item['prompt']
        if key not in latest or item['at'] > latest[key]['at'] or (item['room_xy'] and not latest[key]['room_xy']):
            latest[key] = item
    return sorted(latest.values(), key=lambda i: -i['at'])


def forget(prompt=None):
    with LOCK:
        _save([] if prompt is None else [i for i in _load() if i['prompt'] != prompt])


def target_for(robot, prompt):
    """Room position of `prompt` if any robot placed it in the room frame, else None."""
    for item in known_objects():
        if item['prompt'] == prompt and item['room_xy']:
            return item
    return None
