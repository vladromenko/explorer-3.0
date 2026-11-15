"""ROSMASTER grasp geometry without the robot: a box on the floor rendered into a depth image
through the real arm-camera model comes back at the right base-frame point; reach window,
gripper clamp angle, base correction and in-hand check."""
import numpy as np
import pytest

from explorer3 import rosmaster_arm as ra

K = np.array([[480.7, 0, 322.3], [0, 480.7, 245.6], [0, 0, 1.]])
SERVO = ra.SEARCH_NEAR


def render_box(centre, size=.05, shape=(480, 640)):
    """Depth (m) of an axis-aligned box standing on the floor, seen by the arm camera; floor elsewhere."""
    T = np.asarray(ra.ARM.camera_pose(SERVO))
    R, t = T[:3, :3], T[:3, 3]
    v, u = np.mgrid[0:shape[0], 0:shape[1]]
    rays_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(shape)], -1)
    rays = rays_cam @ R.T                                   # base-frame directions (z = optical depth 1)
    depth = np.full(shape, np.nan)
    floor = -t[2] / rays[..., 2]
    depth[rays[..., 2] < 0] = floor[rays[..., 2] < 0]
    lo = np.array([centre[0] - size / 2, centre[1] - size / 2, 0.])
    hi = np.array([centre[0] + size / 2, centre[1] + size / 2, size])
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (lo - t) / rays
        t2 = (hi - t) / rays
    near = np.nanmax(np.minimum(t1, t2), -1)
    far = np.nanmin(np.maximum(t1, t2), -1)
    hit = (near <= far) & (near > 0)
    depth[hit] = np.minimum(depth[hit], near[hit])
    pix = (hit.nonzero())
    box = [pix[1].min(), pix[0].min(), pix[1].max(), pix[0].max()]
    return depth, box


def test_box_on_floor_comes_back_in_base_frame():
    depth, box = render_box((.30, .05))
    obj = ra.object_in_base(depth, box, SERVO, K)
    assert obj is not None
    assert abs(obj['point'][0] - .30) < .03 and abs(obj['point'][1] - .05) < .03
    assert abs(obj['top_z'] - .05) < .015
    target = ra.grasp_target(obj)
    assert .015 <= target[2] <= .04


def test_reach_window_and_base_correction():
    assert ra.plan_grasp(np.array([.22, 0., .025]), SERVO)[0] is not None
    plan, why = ra.plan_grasp(np.array([.40, 0., .025]), SERVO)
    assert plan is None and 'досягаемости' in why
    forward, turn = ra.base_correction(np.array([.40, 0., .025]))
    assert forward == pytest.approx(.40 - sum(ra.REACH_X) / 2) and turn == 0.
    forward, turn = ra.base_correction(np.array([.20, .10, .025]))
    assert turn > .3


def test_gripper_angle_and_holding():
    assert ra.grip_command(.11) == ra.GRIP_OPEN + ra.OVERDRIVE              # widest object: contact at open
    assert ra.grip_command(.03) > ra.grip_command(.08)
    assert ra.grip_command(.001) <= ra.GRIP_MAX
    assert ra.holding(120, 100) and not ra.holding(120, 118)


def test_learned_pick_runs_only_an_admitted_policy_and_checks_holding(monkeypatch):
    """With an admitted checkpoint the swarm grasp is the policy through the robot runtime;
    the result is the gripper stall, not the policy's word."""
    import types
    import numpy as np
    import explorer3.rosmaster_arm as ra
    from explorer3 import deploy_gate, policy_runtime

    sent = []

    class Core:
        def __init__(self, checkpoint, device, instruction=None):
            assert checkpoint == 'ck' and instruction == 'pick up the cookie'

        def act(self, state, images, task):
            assert set(images) == {'front'} and len(state) == 6
            return [90., 120., 10., 30., 90., 125.]

    class Client:
        def get(self, path):
            import cv2
            ok, jpg = cv2.imencode('.jpg', np.zeros((480, 640, 3), np.uint8))
            return types.SimpleNamespace(content=jpg.tobytes(), status_code=200)

    arm = ra.ArmSkills.__new__(ra.ArmSkills)
    arm.log, arm.stop_event, arm.profile, arm.client = (lambda m: None), ra.threading.Event(), 'rosmaster_pick_place', Client()
    arm.op = types.SimpleNamespace(input=lambda *a: None, set_direct=lambda c: sent.append(c), owner=ra.OWNER, lease='x', error=None)
    arm.rc = types.SimpleNamespace(ZERO={})
    arm.look = lambda prompt: {'point': np.array([.21, 0., .02]), 'top_z': .04, 'width_m': .04}
    arm.drive = lambda *a: None
    arm.move = lambda pose, **k: [90., 130., 0., 25., 90., 100.]
    arm.measured = lambda: [90., 130., 0., 25., 90., 100.]          # gripper stalled at 100 < 125
    monkeypatch.setattr(policy_runtime, 'PolicyCore', Core)
    monkeypatch.setattr(deploy_gate, 'latest', lambda job: {'checkpoint': 'ck', 'policy': 'smolvla', 'verdict': 'ready'})
    result = arm.policy_pick('cookie', 'job1', seconds=.3)
    assert result['ok'] and result['policy_job'] == 'job1' and sent and sent[0] == {'arm': [90., 120., 10., 30., 90., 125.]}
    arm.measured = lambda: [90., 130., 0., 25., 90., 125.]          # closed fully: nothing in the hand
    assert not arm.policy_pick('cookie', 'job1', seconds=.2)['ok']
