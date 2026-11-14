"""M10/M11: policy loops — shadow sends nothing, operator sends scaled commands, failures stop."""
import json
import hashlib

import numpy as np
import pytest

pytest.importorskip('httpx')
from explorer3.sensors import CHANNELS
from explorer3.simlab import contract
from explorer3.simlab.runtime import Fleet, PolicyLoop


def make_policy(path, robot, bias):
    path.mkdir()
    dim = contract.size(robot) * 3
    a = len(contract.ROBOTS[robot]['action'])
    np.savez(path / 'policy.npz', norm_mean=np.zeros(dim), norm_var=np.ones(dim), w0=np.zeros((a, dim)), b0=np.full(a, bias))
    (path / 'policy.json').write_text(json.dumps({
        'format': 'explorer3.simlab.mlp_elu.v1', 'robot': robot, 'contract_version': contract.VERSION,
        'contract_hash': contract.contract_hash(robot), 'history': 3, 'layers': 1, 'scenario': 'dock_v1',
        'scenario_fingerprint': 'x', 'weights_sha256': hashlib.sha256((path / 'policy.npz').read_bytes()).hexdigest()}))
    return path


class Source:
    def __init__(self, fail_after=None):
        self.n, self.fail_after = 0, fail_after

    def read(self):
        self.n += 1
        if self.fail_after and self.n > self.fail_after:
            raise RuntimeError('sensor stream lost')
        return {'features': np.zeros(len(CHANNELS), np.float32), 'image': None}


class Sink:
    def __init__(self):
        self.sent, self.stopped = [], None

    def send(self, a):
        self.sent.append(np.asarray(a))

    def stop(self, reason):
        self.stopped = reason


def test_shadow_sends_nothing(tmp_path):
    sink = Sink()
    loop = PolicyLoop(make_policy(tmp_path / 'p', 'tank', 0.5), Source(), sink, mode='shadow')
    stats = loop.run(duration_s=0.3)
    assert stats.steps >= 3 and sink.sent == [] and sink.stopped == 'duration'


def test_operator_scales_and_parallel_fleet(tmp_path):
    sinks = {'tank': Sink(), 'rosmaster': Sink()}
    fleet = Fleet({r: PolicyLoop(make_policy(tmp_path / r, r, 2.0), Source(), sinks[r], mode='operator') for r in sinks})
    fleet.start(duration_s=0.5)
    summary = fleet.join(5)
    assert all(s['steps'] >= 5 for s in summary.values())
    assert np.allclose(sinks['tank'].sent[0], contract.ROBOTS['tank']['policy_limits'])   # clipped to +1 then scaled
    assert all(s.stopped == 'duration' for s in sinks.values())


def test_sensor_failure_stops_robot(tmp_path):
    sink = Sink()
    loop = PolicyLoop(make_policy(tmp_path / 'p', 'tank', 0.1), Source(fail_after=2), sink, mode='operator')
    stats = loop.run(duration_s=2)
    assert stats.stopped_reason.startswith('error') and sink.stopped.startswith('error')
