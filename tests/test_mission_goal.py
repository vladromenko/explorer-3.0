import time
from types import SimpleNamespace

import numpy as np
import pytest

from explorer3 import mission_goal, swarm_sim, world_memory
from test_swarm import isolated, make, wait, Arm


def zone(domain='sim'):
    return {**mission_goal.first_floor_goal(domain), 'center_xy_m': [1., 1.],
            'approach_pose': [.78, 1., 0.], 'place_pose': [90, 100, 0, 0, 90, 30]}


def measurement(**updates):
    return {'domain': 'sim', 'object_class': 'cube', 'frame': 'fleet_map', 'xy_m': [1., 1.],
            'position_sigma_m': 0., 'object_half_extent_m': .02,
            'at': time.time(), 'released': True, 'inside_height_band': True,
            'source': 'sim2d_object_state', **updates}


def test_delivery_needs_release_fresh_observation_and_whole_object_inside():
    since = time.time() - 1
    def valid(**updates):
        return mission_goal.delivery_verified(zone(), measurement(**updates), domain='sim', object_class='cube', since=since)
    assert valid()
    assert not valid(released=False)
    assert not valid(at=since-1)
    assert not valid(xy_m=[1.19, 1.])
    assert not valid(position_sigma_m=.19)
    assert not valid(object_half_extent_m=None)
    assert not valid(domain='real')
    assert not valid(source='commanded_pose')
    assert mission_goal.preparation_reason(zone(), 'sim') is None
    assert mission_goal.preparation_reason(zone('real'), 'real')
    assert mission_goal.preparation_reason(zone(), 'real')


@pytest.mark.parametrize('verified', [True, False])
def test_floor_mission_never_calls_so101_and_does_not_count_unverified_delivery(isolated, verified):
    world_memory.POSES['rosmaster'].fix(0., 0., 0.)
    world_memory.record('rosmaster', 'cube', 'cube', 1., 0.)
    arms = []
    team, _, _ = make(arms)
    team.agents.domain = lambda: 'sim'
    team.helpers = []  # autonomous recipient; no live donor measurement
    class FloorArm(Arm):
        def observe_delivery(self, prompt):
            return measurement(released=verified)
    team.arm_factory = lambda log: arms.append(FloorArm(log)) or arms[-1]
    team.so101_factory = lambda *a: pytest.fail('SO-101 must not receive this mission')
    team.command('принеси кубик', executor='geometric_baseline', task_context={'terminal_goal': zone()})
    wait(team)
    assert team.state['stage'] == ('done' if verified else 'delivery_unverified')
    assert all(step['robot'] != 'so101' for step in team.agents.plan)
    assert any(call[0] == 'place' for call in arms[0].calls)


def test_real_zone_blocks_before_motion_and_pending_skill_remembers_goal(isolated):
    team, ros, tank = make([])
    team.command('принеси кубик', executor='geometric_baseline', task_context={'terminal_goal': zone('real')})
    wait(team)
    assert team.state['stage'] == 'awaiting_delivery_setup'
    assert not ros.started and not tank.started
    team.command('принеси кубик')
    wait(team)
    assert team.state['stage'] == 'awaiting_skill'
    from explorer3.cognitive import store
    goal = store.mission(team.agents.mission)['terminal_goal']
    assert goal['kind'] == 'floor_zone' and goal['center_xy_m'] is None


def test_two_dimensional_arm_cannot_execute_learned_policy_or_teleport_release(monkeypatch):
    world = SimpleNamespace(true={'rosmaster': (2., 3., 0.)}, objects={'cube': np.array([0., 0.])},
                            carried='cube', sizes={'cube': .04}, to_map=lambda x,y: (x,y,0))
    arm = swarm_sim.SimArm(world, lambda *a: None)
    arm.execution_selection = 'learned-job'
    assert arm.pick('cube')['ok'] is False
    arm.terminal_goal = zone()
    monkeypatch.setattr(swarm_sim.time, 'sleep', lambda seconds: None)
    arm.place(zone()['place_pose'], 40)
    assert world.objects['cube'] == pytest.approx([2.22, 3.])
    assert world.carried is None
    assert not mission_goal.delivery_verified(zone(), arm.observe_delivery('cube'),
        domain='sim', object_class='cube', since=0)


def test_virtual_floor_zone_is_separate_from_donor_and_object():
    world = swarm_sim.SimWorld(seed=17)
    world.true0 = dict(world.true)
    goal = world.floor_goal('cube')
    assert mission_goal.preparation_reason(goal, 'sim') is None
    source_xy = world.to_map(*world.objects['cube'])[:2]
    assert np.linalg.norm(np.array(goal['center_xy_m']) - source_xy) == pytest.approx(.7)
