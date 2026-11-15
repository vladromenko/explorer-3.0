"""MuJoCo tasks that mirror the real recording blocks: same contract as the robot, solvable,
start from the real rest/observation pose, no debug markers in the policy camera."""
import numpy as np
import pytest

from explorer3.profiles import profile
from explorer3.training_drafts import semantic_contract


def _contract(name):
    c = semantic_contract(profile(name))
    c.pop('task')
    return c


@pytest.mark.parametrize('sim,real', [('so101_handover_sim', 'so101_pick_place'), ('rosmaster_floor_pick_sim', 'rosmaster_pick_place')])
def test_same_channels_camera_and_rate_as_the_robot(sim, real):
    assert _contract(sim) == _contract(real)
    assert profile(sim)['real_profile'] == real and profile(sim)['domain'] == 'sim'


mujoco = pytest.importorskip('mujoco')


@pytest.mark.parametrize('name', ['so101_handover_sim', 'rosmaster_floor_pick_sim'])
def test_expert_solves_the_task_from_the_real_start_pose(name):
    from explorer3.sim.aligned import alignment
    from explorer3.sim.factory import make, EXPERTS
    env = make(name)
    start = alignment('so101_handover' if name.startswith('so101') else 'rosmaster_floor_pick')
    start = np.array(start.get('rest_pose') or start['start_pose'])
    ok = 0
    for seed in range(4):
        obs, _ = env.reset(seed=500 + seed)
        assert np.abs(obs['state'][:4] - start[:4]).max() < 6.
        expert = EXPERTS[name](env)
        while True:
            _, _, terminated, truncated, info = env.step(expert.action())
            if terminated or truncated:
                break
        ok += info['outcome'] == 'success'
    assert ok >= 3


def test_no_debug_markers_in_policy_camera():
    from explorer3.sim.factory import make
    env = make('rosmaster_floor_pick_sim', images=True)
    env.reset(seed=1)
    im = env.render()
    assert int(((im[:, :, 0] > 200) & (im[:, :, 1] < 40) & (im[:, :, 2] < 40)).sum()) == 0
    env.close()
