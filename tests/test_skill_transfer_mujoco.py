import copy

import numpy as np
import pytest

pytest.importorskip('mujoco')
from explorer3.sim.factory import make
from explorer3.sim.skill_transfer import RetargetedGrasp, RosmasterTransferInterface
from explorer3.skill_packets import relative_pose
from test_skill_packets import packet_fixture


def test_hold_reference_keeps_the_lift_and_recipient_native_schema():
    env = make('rosmaster_floor_pick_sim')
    try:
        packet = packet_fixture()
        packet['measurement']['closing_axis_object'] = [1, 0, 0]
        hold = packet['phases'][-1]
        hold['world_object_at_start'][2][3] += .12
        sample = hold['samples'][0]
        sample['object_tcp'] = relative_pose(hold['world_object_at_start'], sample['world_tcp']).tolist()
        expert = RetargetedGrasp(env, packet)
        assert expert.samples[-1]['pose'][2, 3] == pytest.approx(env.data.body('task_cube').xpos[2] + .12)
        previous = expert.interface.ik.command.copy()
        action = expert.action()
        assert action.shape == (6,)
        assert np.all(np.abs(action - previous) <= np.array([45] * 5 + [120]) * env.dt + 1e-8)
        assert expert.report['orientation_relaxation']
        assert not expert.report['physical_verified']
    finally:
        env.close()


def test_incompatible_object_and_floor_target_are_rejected_before_motion():
    env = make('rosmaster_floor_pick_sim')
    try:
        packet = packet_fixture()
        uncertain = copy.deepcopy(packet)
        uncertain['measurement']['position_sigma_m'] = .02
        with pytest.raises(ValueError, match='uncertainty'):
            RetargetedGrasp(env, uncertain)
        packet['object']['size_m'] = [.08] * 3
        with pytest.raises(ValueError, match='dimensions'):
            RetargetedGrasp(env, packet)
        interface = RosmasterTransferInterface(env)
        with pytest.raises(ValueError, match='floor clearance'):
            interface.target_pose_to_action(np.eye(4), .04, 'approach', [1, 0, 0])
        assert env.steps == 0
    finally:
        env.close()


def test_stalled_variant_is_rejected_without_object_chasing(monkeypatch):
    from explorer3.sim.skill_transfer import TransferRejected
    env = make('rosmaster_floor_pick_sim')
    try:
        packet = packet_fixture()
        packet['measurement']['closing_axis_object'] = [1, 0, 0]
        expert = RetargetedGrasp(env, packet)
        expert.approach_done = True
        target = expert.samples[0]['pose'].copy()
        target[0, 3] += .1
        monkeypatch.setattr(expert.interface, 'get_robot_eef_pose', lambda: target)
        monkeypatch.setattr(expert.interface, 'target_pose_to_action', lambda *a, **kw: np.zeros(6))
        with pytest.raises(TransferRejected, match='stalled'):
            for _ in range(41):
                expert.action()
        assert expert.path_time == 0 and env.steps == 0
    finally:
        env.close()


def test_existing_policy_evaluator_attaches_independent_physics_measurements(tmp_path, monkeypatch):
    from explorer3.sim import evaluate
    from explorer3.sim.aligned import ScriptedFloorPick
    from explorer3 import training_drafts
    env = make('rosmaster_floor_pick_sim', images=False)
    monkeypatch.setattr(evaluate, 'make', lambda *a, **kw: env)
    monkeypatch.setattr(training_drafts, 'checkpoint_hash', lambda p: 'a'*64)
    class ScriptedFixture:
        # Test transport only: this deliberately is not a trained policy or accepted skill.
        ready = {'policy_type':'scripted_fixture'}
        def __init__(self, *a, **kw):
            self.expert = None
        def request(self, command):
            self.expert = ScriptedFloorPick(env)
        def action(self, obs, cfg):
            return self.expert.action()
        def close(self):
            pass
    monkeypatch.setattr(evaluate, 'ACTClient', ScriptedFixture)
    checkpoint = tmp_path/'checkpoint';checkpoint.mkdir()
    (checkpoint/'model.safetensors').write_bytes(b'fixture only')
    result = evaluate.evaluate(checkpoint, tmp_path/'result.json', [42000,42001,42002],
                               profile='rosmaster_floor_pick_sim', verify_grasp=True)
    assert result['independent_grasp_verification'] and not result['physical_skill_verified']
    assert result['checkpoint_bundle_sha256'] == 'a'*64
    assert len(result['trials']) == 3
    for trial in result['trials']:
        m = trial['grasp_verification']
        assert m['method'].startswith('every MuJoCo physics substep')
        assert m['held_s'] >= 0 and not trial['donor_online'] and not trial['geometric_correction']
        assert trial['object']['name'] == 'cube'
