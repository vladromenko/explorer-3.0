"""The swarm end to end on stand-in robots: command parsing, team search ending for everybody
on the first sighting, «принеси печенье» through every stage (locate -> approach -> pick ->
deliver -> place -> SO-101 hand-over), SO-101 pan self-calibration on a simulated wrist camera,
and Tank's session opened by the swarm itself."""
import math
import threading
import time

import numpy as np
import pytest

from explorer3 import episodes, so101_handoff, swarm, world_memory


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(world_memory, 'MEMORY', tmp_path / 'memory.json')
    monkeypatch.setattr(world_memory, 'SNAPSHOTS', tmp_path / 'snap')
    monkeypatch.setattr(episodes, 'ROOT', tmp_path / 'episodes')
    monkeypatch.setattr(so101_handoff, 'CONFIG', tmp_path / 'handoff.json')
    from explorer3 import knowledge
    monkeypatch.setattr(knowledge, 'STORE', tmp_path / 'knowledge.json')
    from explorer3 import catalog
    from explorer3.cognitive import store
    monkeypatch.setattr(catalog, 'DATA', tmp_path)
    monkeypatch.setattr(store, 'DATA', tmp_path)
    store.initialize()
    for p in world_memory.POSES.values():
        p.x = p.y = p.theta = None


def test_parse_commands():
    assert swarm.parse('Осмотрите комнату') == ('explore', None)
    assert swarm.parse('принеси печеньку') == ('fetch', 'cookie')
    assert swarm.parse('найдите носок') == ('find', 'sock')
    assert swarm.parse('где кружка') == ('find', 'mug')
    assert swarm.parse('подъедь к кубику') == ('goto', 'cube')


class Goal:
    """Stand-in task: finishes with `outcome` after `steps` commands; `effect` runs on the way."""

    def __init__(self, robot, outcome='arrived', steps=5, effect=None, stop_when=None):
        self.robot, self.outcome, self.left, self.effect, self.stop_when = robot, outcome, steps, effect, stop_when
        self.active, self.state, self.message = True, 'run', 'еду'

    def command(self, rgb):
        if self.stop_when is not None and self.stop_when():
            self.cancel('team_done')
            return [0., 0.]
        self.left -= 1
        if self.effect is not None:
            self.effect(self)
        if self.left <= 0:
            self.state, self.active, self.message = self.outcome, False, self.outcome
        return [.1, 0.]

    def cancel(self, reason):
        if self.active:
            self.state, self.active, self.message = 'cancelled', False, reason

    def status(self):
        return {'message': self.message}


class Link:
    def __init__(self):
        self.started, self.released = [], 0

    def start(self, goal):
        self.started.append(goal)

        def drive():
            while goal.active:
                goal.command(None)
                time.sleep(.005)
        threading.Thread(target=drive, daemon=True).start()

    def cancel(self, reason):
        for g in self.started:
            g.cancel(reason)

    def release(self):
        self.released += 1


class Arm:
    def __init__(self, log):
        self.calls = []

    def pick(self, prompt, knowledge=None):
        self.calls.append(('pick', prompt))
        return {'ok': True, 'grip': 90}

    def release(self):
        self.calls.append(('release',))

    def place(self, pose, grip):
        self.calls.append(('place', tuple(pose), grip))

    def drive(self, forward, turn):
        self.calls.append(('drive', forward))

    def close(self):
        self.calls.append(('close',))

    def stop(self):
        pass


class Hand:
    def __init__(self, prompt, log):
        self.prompt, self.taken = prompt, threading.Event()

    def run(self):
        return {'ok': True}

    def stop(self):
        pass


def teach_everything():
    for name in so101_handoff.POSES:
        so101_handoff.teach('so101.' + name, [0., 10., 20., 30., 0., 40.])
    so101_handoff.teach('rosmaster_spot', [1.0, 2.0, 1.57])
    so101_handoff.teach('rosmaster_place', [90, 60, 40, 30, 90, 90])


def make(arm_log, explore_effect=None):
    ros, tank = Link(), Link()

    def explore(robot, **kw):
        return Goal(robot, 'explored', 30, explore_effect, kw.get('stop_when'))
    s = swarm.Swarm(tank=tank, ros=ros, explore_factory=explore,
                    goto_factory=lambda robot, text: Goal(robot, 'arrived'),
                    drive_factory=lambda robot, pose, text: Goal(robot, 'arrived'),
                    arm_factory=lambda log: arm_log.append(Arm(log)) or arm_log[-1],
                    so101_factory=lambda prompt, log: Hand(prompt, log))
    return s, ros, tank


def wait(s, timeout=10):
    end = time.time() + timeout
    while s.thread.is_alive() and time.time() < end:
        time.sleep(.02)
    assert not s.thread.is_alive()


def test_fetch_known_object_runs_every_stage():
    teach_everything()
    world_memory.POSES['tank'].fix(0., 0., 0.)
    world_memory.record('tank', 'к печенью', 'cookie', 1.0, 0.)          # Tank saw it earlier
    arms = []
    s, ros, tank = make(arms)
    s.command('принеси печеньку')
    wait(s)
    assert s.state['stage'] == 'done', s.state
    stages = [x[1] for x in s.state['log']]
    for name in ('approach', 'pick', 'deliver', 'place', 'handoff'):
        assert name in stages
    assert arms[0].calls[0] == ('pick', 'cookie') and ('place', (90, 60, 40, 30, 90, 90), 90) in arms[0].calls
    assert ('drive', -.15) in arms[0].calls                               # backs off before SO-101 moves
    ep = episodes.load()[0]
    # ROSMASTER had no grasp knowledge for the cookie: it asked the SO-101 station first ('learn');
    # the station is not running in the test, so it went on with its own estimate
    assert ep['kind'] == 'fetch' and ep['success'] and [x['stage'] for x in ep['stages']] == ['learn', 'approach', 'pick', 'deliver', 'place', 'handoff']
    assert ep['stages'][0]['ok'] is False


def test_fetch_unknown_object_team_search_stops_both():
    teach_everything()
    world_memory.POSES['rosmaster'].fix(0., 0., 0.)

    def scout(goal):                                                      # ROSMASTER's camera sees it
        if goal.robot == 'rosmaster' and goal.left == 20:
            world_memory.record('rosmaster', 'к печенью', 'cookie', .8, .2)
    arms = []
    s, ros, tank = make(arms, scout)
    s.command('принеси печенье')
    wait(s)
    assert s.state['stage'] == 'done', s.state
    searches = [g for g in ros.started + tank.started if g.outcome == 'explored']
    assert len(searches) == 2 and all(g.state == 'cancelled' for g in searches)   # first sighting ended both
    assert [x['stage'] for x in episodes.load()[0]['stages']][0] == 'search'


def test_fetch_requires_taught_handover():
    s, _, _ = make([])
    s.command('принеси печенье')
    wait(s)
    assert s.state['stage'] == 'error' and 'покажите места передачи' in s.state['message']


def test_stop_cancels_everybody():
    s, ros, tank = make([])
    s.explore_factory = lambda robot, **kw: Goal(robot, 'explored', 10 ** 9)
    s.command('осмотрите комнату')
    time.sleep(.2)
    s.stop()
    wait(s)
    assert s.state['stage'] == 'cancelled'
    assert all(not g.active for g in ros.started + tank.started)


# ------------------------------------------------------------------ SO-101 hand-over
class FakeSO101:
    """Joints follow commands; the wrist camera sees the object shifted by the pan error."""

    def __init__(self, object_pan=7.0, holds=True):
        self.q = [0., 10., 20., 30., 0., 40.]
        self.object_pan, self.holds = object_pan, holds
        self.sent = []

    def start(self, **kw):
        pass

    def observe(self, require_images=True):
        img = np.zeros((480, 640, 3), np.uint8)
        q = list(self.q)
        if self.holds and q[5] < 20:          # jaws stop on the object
            q[5] = 20.
        return {'observation.state': q, 'images': {'wrist': {'image': img}}}

    def send(self, goal, obs):
        self.q = list(goal)
        self.sent.append(list(goal))

    def close(self):
        pass


class WristDetector:
    def __init__(self, robot):
        self.robot = robot

    def detect(self, rgb, prompt, conf=.4):
        # object pixel moves 25 px per degree of pan error (sign unknown to the controller)
        u = 320 - 25 * (self.robot.object_pan - self.robot.q[0])
        return [[u - 30, 200, u + 30, 260, .9]], 1.


def test_handoff_centres_pan_and_detects_grip():
    so101_handoff.teach('so101.above', [0., 10., 20., 30., 0., 60.])
    so101_handoff.teach('so101.grasp', [0., 30., 40., 30., 0., 60.])
    so101_handoff.teach('so101.grip', [0., 0., 0., 0., 0., 5.])
    so101_handoff.teach('so101.give', [40., 20., 20., 10., 0., 60.])
    so101_handoff.teach('so101.home', [0., 0., 0., 0., 0., 60.])
    robot = FakeSO101(object_pan=7.)
    h = so101_handoff.Handoff('cookie', source=robot, detector=WristDetector(robot), speed_deg_s=400, wait_take_s=.1)
    h.taken.set()
    result = h.run()
    assert result['ok'] and abs(result['pan_correction'] - 7.) < .5
    grasp_cmds = [c for c in robot.sent if abs(c[1] - 30.) < 1e-6 and abs(c[2] - 40.) < 1e-6]
    assert grasp_cmds and abs(grasp_cmds[-1][0] - 7.) < .5                       # grasped where the object is
    empty = FakeSO101(object_pan=0., holds=False)
    h2 = so101_handoff.Handoff('cookie', source=empty, detector=WristDetector(empty), speed_deg_s=400, wait_take_s=.1)
    assert not h2.run()['ok']


def test_pan_correction_any_sign():
    assert abs(so101_handoff.pan_correction(.5, .3, 4.) - 6.) < 1e-6
    assert abs(so101_handoff.pan_correction(.5, .7, -4.) - 14.) < 1e-6           # moved the wrong way: go back further
    assert so101_handoff.pan_correction(.5, .5, 4.) == 0.                            # no response: no correction


# ------------------------------------------------------------------ Tank link
class FakeControl:
    def __init__(self):
        self.robot, self.owner, self.goal, self.beats, self.stopped = 'so101', None, None, 0, None

    def expired(self):
        return self.owner is None

    def select(self, robot):
        self.robot = robot

    def heartbeat(self, owner):
        self.beats += 1

    def stop(self, reason):
        self.owner, self.stopped = None, reason

    def cancel_goal(self, reason):
        if self.goal is not None:
            self.goal.cancel(reason)


class FakeSession:
    def __init__(self, control):
        self.control, self.thread, self.phase, self.error = control, None, 'idle', None

    def start(self, owner, name, scene, record=True):
        assert name == 'tank_dock' and record is False
        self.control.owner = owner
        self.phase = 'preparing'
        self.thread = threading.Thread(target=lambda: time.sleep(5), daemon=True)
        self.thread.start()

        def ready():
            time.sleep(.1)
            self.phase = 'running'
        threading.Thread(target=ready, daemon=True).start()


def test_swarm_opens_tank_itself_and_releases_it():
    control = FakeControl()
    link = swarm.TankLink(control, FakeSession(control))
    goal = Goal('tank')
    link.start(goal)
    assert control.robot == 'tank' and control.owner == 'swarm' and control.goal is goal
    time.sleep(.5)
    assert control.beats >= 1
    link.release()
    assert control.owner is None and 'Рой' in control.stopped


def test_solo_run_uses_one_robot_and_is_marked_for_the_article():
    s, ros, tank = make([])
    s.command('осмотрите комнату', robots=['rosmaster'])
    wait(s)
    assert ros.started and not tank.started
    ep = episodes.load()[0]
    assert ep['condition'] == 'solo' and ep['robots'] == ['rosmaster']
    csv = episodes.to_csv()
    assert 'condition' in csv.splitlines()[0] and ',solo,' in csv
