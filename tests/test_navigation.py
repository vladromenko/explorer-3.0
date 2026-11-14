"""Autonomy on the shared map in a simulated room: planning around furniture, the collision
guard, full autonomous exploration (robot driven only by the Explore task, lidar ray-cast),
and every task written down as an episode."""
import math

import numpy as np
import pytest

pytest.importorskip('cv2')
from explorer3 import episodes, explore, fleet_map as fm, navigation, world_memory

SEGMENTS = np.array([((-2, -1.5), (3, -1.5)), ((3, -1.5), (3, 2.5)), ((3, 2.5), (-2, 2.5)), ((-2, 2.5), (-2, -1.5)),
                     ((-1.6, 1.6), (-0.2, 1.6)), ((-0.2, 1.6), (-0.2, 2.5)),
                     ((1.2, -0.4), (1.9, -0.4)), ((1.9, -0.4), (1.9, 0.3)), ((1.9, 0.3), (1.2, 0.3)), ((1.2, 0.3), (1.2, -0.4)),
                     ((2.4, 1.0), (3, 1.0)), ((2.4, 1.0), (2.4, 2.0)), ((2.4, 2.0), (3, 2.0)),
                     ((0.4, -1.5), (0.4, 0.6))], float)            # a partition: the room has two halves


def raycast(pose, mount, n=450):
    x, y, t = pose
    mx, my, myaw, direction = mount
    o = np.array([x + math.cos(t) * mx - math.sin(t) * my, y + math.sin(t) * mx + math.cos(t) * my])
    a = t + myaw + direction * 2 * np.pi * np.arange(n) / n
    d = np.stack([np.cos(a), np.sin(a)], 1)[:, None, :]             # (n,1,2)
    p, e = SEGMENTS[None, :, 0], SEGMENTS[None, :, 1] - SEGMENTS[None, :, 0]
    den = d[..., 0] * -e[..., 1] + d[..., 1] * e[..., 0]
    q = p - o
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (q[..., 0] * -e[..., 1] + q[..., 1] * e[..., 0]) / den
        u = (d[..., 0] * q[..., 1] - d[..., 1] * q[..., 0]) / den
    r = np.where((abs(den) > 1e-9) & (r > .05) & (u >= 0) & (u <= 1), r, np.inf).min(1)
    return (r + np.random.default_rng(abs(int(1e3 * (x + y)))).normal(0, .01, n)).astype(np.float32)


def packet(robot, pose):
    arrays, streams = {}, {}
    for name, mount in fm.MOUNTS[robot].items():
        arrays[name] = raycast(pose, mount)
        streams[name] = {'angle_min': 0., 'angle_increment': 2 * math.pi / 450, 'range_min': .05}
    return {'arrays': arrays, 'meta': {'streams': streams}}


def wall_distance(xy):
    a, b = SEGMENTS[:, 0], SEGMENTS[:, 1]
    ab = b - a
    t = np.clip(((xy - a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
    return float(np.linalg.norm(a + t[:, None] * ab - xy, axis=1).min())


@pytest.fixture
def fleet(monkeypatch, tmp_path):
    f = fm.FleetMap()
    monkeypatch.setattr(fm, 'FLEET', f)
    monkeypatch.setattr(explore, 'FLEET', f)
    monkeypatch.setattr(fm, 'MAP_DIR', tmp_path / 'map')
    monkeypatch.setattr(episodes, 'ROOT', tmp_path / 'episodes')
    monkeypatch.setattr(world_memory, 'MEMORY', tmp_path / 'memory.json')
    for p in world_memory.POSES.values():
        p.x = p.y = p.theta = None
    return f


def test_plan_goes_around_the_partition(fleet):
    for pose in [(0., 0., 0.), (-1., -.5, 0.), (-1., .8, 0.), (1., 1.2, 0.), (2.2, -1., 0.), (1., -1., 0.)]:
        fleet.update('rosmaster', packet('rosmaster', pose), pose)
    path = navigation.plan(fleet.grid, 'rosmaster', (-1., -1.), (1.0, -1.0))
    assert path is not None
    assert max(p[1] for p in path) > .6                # must pass above the partition end (y = 0.6)
    for a, b in zip(path, path[1:]):
        for t in np.linspace(0, 1, 10):
            xy = np.array(a) + t * (np.array(b) - np.array(a))
            assert wall_distance(xy) > .2


def test_guard_stops_before_a_wall():
    pts = np.array([[.40, y] for y in np.linspace(-.3, .3, 20)])     # wall 0.40 m ahead of the centre
    v, w, why = navigation.guard(pts, 'rosmaster', .15, 0.)
    assert v == 0 and why and abs(w) > 0
    v, w, why = navigation.guard(pts + [1.5, 0], 'rosmaster', .15, 0.)
    assert v == .15 and why is None


@pytest.mark.parametrize('robot', ['rosmaster', 'tank'])
def test_robot_explores_the_room_by_itself(fleet, robot):
    true = (-1.2, -.8, 0.3)
    if robot == 'tank':                                   # Tank arrives into a map ROSMASTER started
        fleet.update('rosmaster', packet('rosmaster', (-1.5, -1., 0.)), (-1.5, -1., 0.))
    task = explore.Explore(robot)
    dt, closest = .25, 9.
    for _ in range(1600):
        fleet.update(robot, packet(robot, true), true if robot == 'rosmaster' else None)
        fleet.track[robot]['global_at'] = 0.
        v, w = task.command(None)
        if not task.active:
            break
        true = (true[0] + v * dt * math.cos(true[2]), true[1] + v * dt * math.sin(true[2]), fm.wrap(true[2] + w * dt))
        closest = min(closest, wall_distance(np.array(true[:2])))
    half_width = {'rosmaster': .18, 'tank': .14}[robot]
    assert task.state == 'explored', task.message
    assert closest > half_width + .02                    # never touched furniture with its side
    free = fleet.grid.known_free()
    ys, xs = np.nonzero(free)
    origin = (-1.5, -1., 0.) if robot == 'tank' else (-1.2, -.8, 0.3)   # the map starts at the first robot's pose
    pts = fm.transform(np.stack([(xs - fm.SIZE // 2 + .5) * fm.RES, (ys - fm.SIZE // 2 + .5) * fm.RES], 1), origin)
    room = pts[:, 0], pts[:, 1]
    seen_right = ((room[0] > .6) & (room[0] < 2.8) & (room[1] > -1.3) & (room[1] < 0.)).sum() * fm.RES ** 2
    assert seen_right > .8                               # the half behind the partition was mapped
    item = episodes.load()[0]
    assert item['robot'] == robot and item['kind'] == 'explore' and item['success'] and item['path_m'] > 1
    assert episodes.stats()[0]['success_rate'] == 1.0


def test_goto_drives_around_furniture_to_a_reported_place(fleet, monkeypatch):
    from explorer3 import goto
    for pose in [(0., 0., 0.), (-1., -.5, 0.), (-1., .8, 0.), (1., 1.2, 0.), (2.2, -1., 0.)]:
        fleet.update('rosmaster', packet('rosmaster', pose), pose)
    world_memory.POSES['tank'].fix(2.2, -1., math.pi)
    world_memory.record('tank', 'к носку', 'sock', .3, 0.)          # sock at (1.9, -1.0), behind the partition
    true = (-1., -.8, 0.)
    world_memory.POSES['rosmaster'].fix(*true)

    class Blind:
        def detect(self, rgb, prompt, conf=.35):
            import time
            time.sleep(.01)
            return [], 1.

    monkeypatch.setattr(goto.Detector, 'shared', classmethod(lambda cls: Blind()))
    g = goto.GoTo('rosmaster', 'к носку')
    g.localizer.locate = lambda rgb, cam: None
    frame = np.zeros((480, 640, 3), np.uint8)
    for _ in range(600):
        fleet.update('rosmaster', packet('rosmaster', true), true)
        g.planned_at = 0.                                  # re-plan every step in the fast test loop
        v, w = g.command(frame)
        if g.state != 'memory' and not (v or w):
            break
        true = (true[0] + v * .25 * math.cos(true[2]), true[1] + v * .25 * math.sin(true[2]), fm.wrap(true[2] + w * .25))
        assert wall_distance(np.array(true[:2])) > .19                  # half-width .18: no side contact
    assert math.dist(true[:2], (1.9, -1.0)) < .8          # arrived near the reported place
    g.cancel()
    assert episodes.load()[0]['kind'] == 'goto'


def test_tank_own_map_is_stitched_into_the_shared_one(fleet):
    # ROSMASTER knows only the left half; Tank starts behind the partition (right half).
    for pose in [(0., 0., 0.), (-1., -.5, 0.), (-1., .8, 0.)]:
        fleet.update('rosmaster', packet('rosmaster', pose), pose)
    right_before = fleet.grid.known_free()[int(-1.0 / fm.RES) + fm.SIZE // 2, int(1.0 / fm.RES) + fm.SIZE // 2]
    route = [(1.0, -1.0, 1.57), (1.0, -.7, 1.57), (1.0, -.3, 1.57), (1.0, .1, 1.57), (1.0, .5, 1.8), (.9, .9, 2.2),
             (.6, 1.1, 2.6), (.2, 1.2, 3.0), (-.2, 1.0, 3.1)]
    states = []
    for true in route:
        s = fleet.update('tank', packet('tank', true))
        fleet.track['tank']['global_at'] = 0.                 # no 4 s throttle in the test
        states.append(s['state'])
    assert states[0] == 'submap' and states[-1] == 'tracking', states
    assert math.dist(s['pose'][:2], route[-1][:2]) < .1 and abs(fm.wrap(s['pose'][2] - route[-1][2])) < .08
    assert fleet.contrib.get('tank', 0) >= 3                   # Tank's own scans went into the shared map
    assert not right_before and fleet.grid.known_free()[int(-1.0 / fm.RES) + fm.SIZE // 2, int(1.0 / fm.RES) + fm.SIZE // 2]


def test_two_robots_split_the_frontiers(fleet):
    for pose in [(0., 0., 0.)]:
        fleet.update('rosmaster', packet('rosmaster', pose), pose)
    for _ in range(3):
        fleet.update('tank', packet('tank', (-1.0, -.6, 0.)))
        fleet.track['tank']['global_at'] = 0.
    assert fleet.track['tank']['state'] == 'tracking'
    a = explore.Explore('rosmaster')
    b = explore.Explore('tank')
    a.command(None)
    b.command(None)
    assert a.target is not None and b.target is not None
    assert math.dist(a.target, b.target) > .5                 # different parts of the boundary
    a.cancel(); b.cancel()
