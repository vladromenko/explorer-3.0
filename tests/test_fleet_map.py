"""Shared lidar map: ROSMASTER maps a synthetic room, Tank (no encoders) finds itself in it
by global matching, both are tracked, and the relative pose between them is right."""
import math

import numpy as np
import pytest

pytest.importorskip('cv2')
from explorer3 import fleet_map as fm, world_memory

# Room 5 x 4 m with a sofa, a table and a cupboard (asymmetric, like a real room).
SEGMENTS = [((-2, -1.5), (3, -1.5)), ((3, -1.5), (3, 2.5)), ((3, 2.5), (-2, 2.5)), ((-2, 2.5), (-2, -1.5)),
            ((-1.6, 1.6), (-0.2, 1.6)), ((-0.2, 1.6), (-0.2, 2.5)),                 # sofa
            ((1.2, -0.4), (1.9, -0.4)), ((1.9, -0.4), (1.9, 0.3)), ((1.9, 0.3), (1.2, 0.3)), ((1.2, 0.3), (1.2, -0.4)),
            ((2.4, 1.0), (3, 1.0)), ((2.4, 1.0), (2.4, 2.0)), ((2.4, 2.0), (3, 2.0))]  # cupboard


def raycast(pose, n=450, mount=(0., 0., 0., 1), noise=.01, rng=np.random.default_rng(0)):
    x, y, t = pose
    mx, my, myaw, direction = mount
    ox, oy = x + math.cos(t) * mx - math.sin(t) * my, y + math.sin(t) * mx + math.cos(t) * my
    ranges = np.full(n, np.inf)
    for i in range(n):
        a = t + myaw + direction * 2 * math.pi * i / n
        d = np.array([math.cos(a), math.sin(a)])
        for (ax, ay), (bx, by) in SEGMENTS:
            e = np.array([bx - ax, by - ay])
            m = np.array([[d[0], -e[0]], [d[1], -e[1]]])
            if abs(np.linalg.det(m)) < 1e-9:
                continue
            r, u = np.linalg.solve(m, [ax - ox, ay - oy])
            if r > .05 and 0 <= u <= 1:
                ranges[i] = min(ranges[i], r)
    ranges += rng.normal(0, noise, n)
    return ranges.astype(np.float32)


def packet(robot, pose):
    arrays, streams = {}, {}
    for name, mount in fm.MOUNTS[robot].items():
        arrays[name] = raycast(pose, 450, mount)
        streams[name] = {'angle_min': 0., 'angle_increment': 2 * math.pi / 450, 'range_min': .05}
    return {'arrays': arrays, 'meta': {'streams': streams}}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(fm, 'MAP_DIR', tmp_path / 'map')
    monkeypatch.setattr(world_memory, 'MEMORY', tmp_path / 'memory.json')
    from explorer3 import episodes
    monkeypatch.setattr(episodes, 'ROOT', tmp_path / 'episodes')
    for p in world_memory.POSES.values():
        p.x = p.y = p.theta = None


def locate(fleet, robot, pk, tries=3):
    """A robot joins the shared map after two agreeing global fixes (4 s apart live)."""
    for _ in range(tries):
        s = fleet.update(robot, pk)
        fleet.track[robot]['global_at'] = 0.
        if s['state'] == 'tracking':
            return s
    return s


def close(a, b, xy=.08, th=.06):
    return math.dist(a[:2], b[:2]) < xy and abs(fm.wrap(a[2] - b[2])) < th


def test_scan_points_use_mounting_and_drop_own_body():
    pk = packet('rosmaster', (0., 0., 0.))
    pk['arrays']['scan'][:5] = .1                     # returns from the robot's own arm
    pts = fm.scan_points('rosmaster', pk)
    assert len(pts) > 700
    assert not ((abs(pts[:, 0]) < .2) & (abs(pts[:, 1]) < .15)).any()
    # table front at x = 1.2 m ahead, wall at x = -2 m behind (rear lidar sees it)
    row = pts[np.abs(pts[:, 1]) < .05][:, 0]
    assert abs(row.max() - 1.2) < .05 and abs(row.min() + 2.) < .05


def test_rosmaster_maps_and_tank_finds_itself():
    fleet = fm.FleetMap()
    path = [(0., 0., 0.), (.3, .1, .2), (.6, .5, .6), (.4, 1.0, 1.4), (-.3, .9, 2.4), (-1., .3, 3.0), (-1.2, -.6, -2.0)]
    for i, pose in enumerate(path):
        odom = (pose[0] + .03 * i, pose[1] - .02 * i, pose[2] + .02 * i)   # drifting encoders
        s = fleet.update('rosmaster', packet('rosmaster', pose), odom)
        assert s['state'] in ('mapping', 'tracking')
        assert close(s['pose'], pose), (i, s['pose'], pose)
    # Tank appears somewhere in the room facing an arbitrary direction: global search.
    tank = (2.2, -.9, 2.5)
    s = locate(fleet, 'tank', packet('tank', tank))
    assert s['state'] == 'tracking' and close(s['pose'], tank), s
    assert world_memory.POSES['tank'].source == 'lidar'
    # Tank drives and turns without encoders; it is tracked scan to scan.
    for tank in [(2.0, -.7, 2.8), (1.8, -.4, -3.0), (1.5, -.6, -2.5)]:
        s = fleet.update('tank', packet('tank', tank))
        assert close(s['pose'], tank), (s, tank)
    rel = fleet.relative('tank', 'rosmaster')
    ros = path[-1]
    assert rel['distance'] == pytest.approx(math.dist(ros[:2], tank[:2]), abs=.1)
    assert abs(fm.wrap(rel['bearing'] - fm.wrap(math.atan2(ros[1] - tank[1], ros[0] - tank[0]) - tank[2]))) < .08


def test_sighting_by_one_robot_is_a_goal_for_the_other():
    fleet = fm.FleetMap()
    fleet.update('rosmaster', packet('rosmaster', (0., 0., 0.)), (0., 0., 0.))
    locate(fleet, 'tank', packet('tank', (1.0, 1.5, -1.57)))
    item = world_memory.record('tank', 'к носку', 'sock', 1.0, 0.0)       # 1 m straight ahead of Tank
    assert item['room_xy'] == pytest.approx([1.0, .5], abs=.1)
    assert world_memory.target_for('rosmaster', 'sock')['robot'] == 'tank'


def test_unrelated_place_is_not_forced_into_the_map():
    fleet = fm.FleetMap()
    fleet.update('rosmaster', packet('rosmaster', (0., 0., 0.)), (0., 0., 0.))
    corridor = {'arrays': {'scan': np.where(np.arange(450) % 225 < 112, .3, 1.5).astype(np.float32)},
                'meta': {'streams': {'scan': {'angle_min': 0., 'angle_increment': 2 * math.pi / 450}}}}
    s = fleet.update('tank', corridor)
    assert s['state'] == 'submap' and s['pose'] is None                 # keeps its own map meanwhile


def test_map_survives_restart(tmp_path):
    fleet = fm.FleetMap()
    fleet.update('rosmaster', packet('rosmaster', (0., 0., 0.)), (0., 0., 0.))
    fleet.save()
    again = fm.FleetMap()
    assert again.load() and again.grid.scans == 1
    s = locate(again, 'tank', packet('tank', (2.0, 1.5, 1.0)))
    assert close(s['pose'], (2.0, 1.5, 1.0))
    assert again.image() is not None


def test_frozen_lidar_is_detected_and_not_used(monkeypatch):
    """A ROS stream that stops changing (ROSMASTER's bridge stopped) must not keep a robot «localized»."""
    p = {'meta': {'streams': {'scan': {'sequence': 5, 'source_timestamp_ns': 9}}}}
    assert fm.fresh_marker(p) == fm.fresh_marker({'meta': {'streams': {'scan': {'sequence': 5, 'source_timestamp_ns': 9}}}})
    assert fm.fresh_marker(p) != fm.fresh_marker({'meta': {'streams': {'scan': {'sequence': 6, 'source_timestamp_ns': 9}}}})
    f = fm.FleetMap()
    f.track['rosmaster'] = {'pose': (0., 0., 0.), 'state': 'tracking', 'at': 0., 'scan': 1}
    f.stale('rosmaster', 'Лидар не обновляется')
    assert f.track['rosmaster']['state'] == 'no_scan' and f.scan('rosmaster') is None and f.nav('rosmaster') is None
