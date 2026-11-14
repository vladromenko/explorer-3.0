"""Simlab dock: contract, GPU/CPU batch env, MuJoCo validation env, numpy policy runtime."""
import math

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('mujoco')

from explorer3.simlab import contract
from explorer3.simlab.dock_gpu import DockBatch, wrap
from explorer3.simlab.dock_mujoco import DockMujoco


def scripted(env):
    x, y, yaw = env.pose.unbind(1)
    dx, dy = env.standoff - x, -y
    dist = torch.hypot(dx, dy)
    head = wrap(torch.atan2(dy, dx) - yaw)
    yerr = wrap(yaw - math.pi)
    if env.robot == 'tank':
        v = torch.where(dist < .04, torch.zeros_like(dist), (0.9 * dist).clamp(max=.3) * torch.cos(head).clamp(min=0))
        w = torch.where(dist < .04, -2 * yerr, 2 * head)
        return torch.stack([v / .3, w / 1.0], 1)
    c, s = torch.cos(yaw), torch.sin(yaw)
    a = torch.stack([(0.9 * (c * dx + s * dy)) / .3, (0.9 * (-s * dx + c * dy)) / .25, -2 * yerr / .8], 1)
    a[:, :2] *= (dist > .03).float()[:, None]
    return a


@pytest.mark.parametrize('robot', ['tank', 'rosmaster'])
def test_observation_matches_contract(robot):
    env = DockBatch(robot, n=8, device='cpu', seed=1)
    obs, priv = env.reset_all()
    assert obs.shape == (8, contract.size(robot))
    assert torch.isfinite(obs).all() and torch.isfinite(priv).all()
    # the real-robot builder produces the same layout from the same quantities
    vec = contract.build(robot, {'distance': 1.0, 'bearing': 0.2, 'yaw': 0.1, 'visible': 1, 'age_s': 0.0},
                         np.full(36, 2.0), (0.1, 0.0, 0.0), np.zeros(len(contract.ROBOTS[robot]['action'])))
    assert vec.shape == (contract.size(robot),)


def test_scripted_controller_docks_on_easy_starts():
    env = DockBatch('tank', n=64, device='cpu', seed=2, curriculum=0.0)
    env.reset_all()
    successes = 0
    for _ in range(300):
        *_, info = env.step(scripted(env))
        successes += int(info['success'].sum())
    assert successes >= 60


def test_agent_lidar_stop_limits_forward_speed():
    env = DockBatch('tank', n=1, device='cpu', seed=0)
    lidar = torch.full((1, 36), 3.0)
    lidar[0, 0] = 0.10
    env.agent_out[:] = torch.tensor([[0.1, 0., 0.]])
    cmd = env._agent_layer(torch.tensor([[0.3, 0., 0.]]), lidar)
    assert cmd[0, 0] <= 0.1 + 1e-6
    lidar[0, 0] = 0.04
    cmd = env._agent_layer(torch.tensor([[0.3, 0., 0.]]), lidar)
    assert cmd[0, 0] == 0
    back = env._agent_layer(torch.tensor([[-0.2, 0., 0.]]), lidar)
    assert back[0, 0] < 0                         # reversing away is allowed


def test_mujoco_lidar_agrees_with_analytic_casting():
    env = DockMujoco('tank', seed=5)
    for seed in range(5):
        env.g.manual_seed(seed)
        env.reset(torch.ones(1, dtype=torch.bool))
        analytic = DockBatch.lidar_bins(env, noisy=False)
        mj = env.lidar_bins(noisy=False)
        both = (analytic > 0) & (mj > 0)
        assert both.sum() > 20
        assert torch.allclose(analytic[both], mj[both], atol=0.03)


def test_policy_artifact_roundtrip(tmp_path):
    from explorer3.simlab.ppo import ActorCritic
    from explorer3.simlab import artifact
    model = ActorCritic(contract.size('tank') * 3, 53, 2)
    x = torch.randn(4, contract.size('tank') * 3)
    model.actor_norm.update(x)
    meta = {'robot': 'tank', 'history': 3}
    artifact.export(model, meta, tmp_path, scenario='dock_v1', scenario_fingerprint='x')
    pol = artifact.Policy(tmp_path)
    obs = x[0].view(3, -1)
    for row in obs:
        out = pol.act(row.numpy())
    expected = model.actor(model.actor_norm(x[0:1])).clamp(-1, 1)[0].detach().numpy()
    assert np.allclose(out, expected, atol=1e-5)
    info = (tmp_path / 'policy.json').read_text().replace(contract.contract_hash('tank'), 'bad')
    (tmp_path / 'policy.json').write_text(info)
    with pytest.raises(ValueError):
        artifact.Policy(tmp_path)


def test_mujoco_env_scripted_rollout_runs():
    env = DockMujoco('tank', seed=3, curriculum=0.0)
    env.g.manual_seed(11)
    env.reset(torch.ones(1, dtype=torch.bool))
    done = False
    for _ in range(400):
        a = scripted(env)
        action = env.command(a, env.lidar_bins())
        collided = env.integrate()
        _, term, trunc, info = env.finish(action, collided, auto_reset=False)
        if bool(term[0] or trunc[0]):
            done = True
            break
        env.observe()
    assert done and bool(info['success'][0])
