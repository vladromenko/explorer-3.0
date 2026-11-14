"""Shared fleet memory: beacon localization on a synthetic camera view, sightings in the room
frame, and a second robot heading for a place the first one reported."""
import math
import time

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
from explorer3 import goto, world_memory

CAM = dict(goto.CAMERAS['tank'])


def render_beacon(robot_xyt, beacon, cam=CAM, marker_id=1, size=(400, 640)):
    x, y, t = robot_xyt
    cx = x + cam['forward'] * math.cos(t) - cam['lateral'] * math.sin(t)
    cy = y + cam['forward'] * math.sin(t) + cam['lateral'] * math.cos(t)
    tc = t + cam['yaw']
    K = np.array([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1.]])

    def to_cam(px, py, pz):
        rx, ry = px - cx, py - cy
        f = rx * math.cos(tc) + ry * math.sin(tc)
        left = -rx * math.sin(tc) + ry * math.cos(tc)
        return np.array([-left, -(pz - cam['height']), f])
    h = beacon['side'] / 2
    phi = beacon['facing']
    right = np.array([-math.sin(phi), math.cos(phi)])          # viewer's right along the wall, seen from the front
    corners = []
    for sx, sz in ((-h, h), (h, h), (h, -h), (-h, -h)):
        px, py = beacon['x'] + sx * right[0], beacon['y'] + sx * right[1]
        corners.append(to_cam(px, py, beacon['z'] + sz))
    px = (K @ np.array(corners).T).T
    px = px[:, :2] / px[:, 2:3]
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id, 200)
    canvas = np.full((300, 300), 255, np.uint8)
    canvas[50:250, 50:250] = marker
    H = cv2.getPerspectiveTransform(np.float32([[50, 50], [250, 50], [250, 250], [50, 250]]), px.astype(np.float32))
    gray = cv2.warpPerspective(canvas, H, (size[1], size[0]), borderValue=170)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


@pytest.fixture
def room(monkeypatch, tmp_path):
    table = {1: {'x': 0.0, 'y': 1.0, 'z': .14, 'facing': 0.0, 'side': .15}}
    monkeypatch.setattr(world_memory, 'beacons', lambda: table)
    monkeypatch.setattr(world_memory, 'MEMORY', tmp_path / 'memory.json')
    monkeypatch.setattr(world_memory, 'SNAPSHOTS', tmp_path / 'snap')
    from explorer3 import episodes
    monkeypatch.setattr(episodes, 'ROOT', tmp_path / 'episodes')
    for p in world_memory.POSES.values():
        p.x = p.y = p.theta = None
    return table


@pytest.mark.parametrize('pose', [(1.2, 1.0, math.pi), (1.5, 0.7, math.pi - .3), (1.0, 1.4, math.pi + .25)])
def test_beacon_gives_robot_pose(room, pose):
    fix = world_memory.BeaconLocalizer().locate(render_beacon(pose, room[1]), CAM)
    assert fix is not None
    # Single planar marker: heading is the weak part head-on (~10 deg), position follows (~0.2 m).
    # Enough to drive to where an object was seen and finish with the camera.
    assert abs(fix['x'] - pose[0]) < .25 and abs(fix['y'] - pose[1]) < .25
    assert abs(world_memory.wrap(fix['theta'] - pose[2])) < .2


def test_sighting_is_stored_in_room_frame(room):
    world_memory.POSES['tank'].fix(1.0, 1.0, math.pi / 2)          # facing +y
    item = world_memory.record('tank', 'к носку', 'sock', 1.0, 0.0)
    assert item['room_xy'] == pytest.approx([1.0, 2.0], abs=1e-6)
    assert world_memory.target_for('rosmaster', 'sock')['robot'] == 'tank'


class NoDetections:
    def detect(self, rgb, prompt, conf=.35):
        time.sleep(.01)
        return [], 1.0


def test_helper_heads_for_reported_place(room, monkeypatch):
    world_memory.POSES['tank'].fix(1.0, 1.0, math.pi / 2)
    world_memory.record('tank', 'к носку', 'sock', 1.0, 0.0)        # sock at (1, 2)
    world_memory.POSES['rosmaster'].fix(1.0, 0.0, 0.0)              # ROSMASTER at (1, 0) facing +x
    monkeypatch.setattr(goto.Detector, 'shared', classmethod(lambda cls: NoDetections()))
    g = goto.GoTo('rosmaster', 'к носку')
    g.localizer.locate = lambda rgb, cam: None
    v, w = g.command(np.zeros((480, 640, 3), np.uint8))
    assert g.state == 'memory' and w > 0                           # turns left toward (1, 2)
    g.cancel()
