"""One runtime everywhere: the robot trial path (trials.PolicyActions) and the MuJoCo evaluation
worker (sim.policy_worker via ACTClient) give the same action for the same observation."""
import glob
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('lerobot')
from explorer3.paths import DATA

CANDIDATES = sorted(glob.glob(str(DATA / 'checkpoints' / '*-b*/checkpoints/*/pretrained_model')))
ACT = next((Path(c) for c in CANDIDATES if json.loads((Path(c) / 'config.json').read_text()).get('type') == 'act'
            and json.loads((Path(c) / 'config.json').read_text())['input_features'].get('observation.state', {}).get('shape') == [6]), None)


@pytest.mark.skipif(ACT is None, reason='no ACT checkpoint on this machine')
def test_robot_path_and_mujoco_path_agree(tmp_path):
    from explorer3.policy_runtime import PolicyCore
    from explorer3.act_client import ACTClient
    from explorer3 import trials
    cfg = json.loads((ACT / 'config.json').read_text())
    key = next(k for k in cfg['input_features'] if k.startswith('observation.images.'))
    h, w = cfg['input_features'][key]['shape'][1:]
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    state = [5., 40., -30., 60., 0., 30.]
    task = 'Pick up the cube, place it in the box, and withdraw the arm.'
    core = PolicyCore(ACT, 'cuda' if torch.cuda.is_available() else 'cpu')
    direct = core.act(state, {key: rgb}, task)
    # robot trial path
    pa = trials.PolicyActions.__new__(trials.PolicyActions)
    pa.core, pa.policy, pa.environment_shape, pa.task, pa.decision_info = core, core.policy, core.environment, task, {}
    core.reset()
    robot = pa.next({'observation.state': state, 'images': {key.split('.')[-1]: {'image': rgb}}})
    assert np.allclose(direct, robot, atol=1e-4)
    # MuJoCo evaluation worker (separate process, same runtime)
    client = ACTClient(ACT, tmp_path / 'inference.log', device='cuda' if torch.cuda.is_available() else 'cpu', response_timeout=300)
    try:
        sim = client.action({'state': np.array(state), 'rgb': rgb}, {'camera': key.split('.')[-1], 'task': task})
    finally:
        client.close()
    assert np.allclose(direct, sim, atol=2e-2)            # PNG transport is lossless; GPU nondeterminism only
