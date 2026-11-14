"""M8.2: shadow mode on a recording never sends commands and reports agreement."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip('av')
pd = pytest.importorskip('pandas')
pytest.importorskip('httpx')

ROOT = Path(__file__).resolve().parents[1]


def fake_policy(path):
    from explorer3.simlab import contract
    dim = contract.size('tank') * 3
    np.savez(path / 'policy.npz', norm_mean=np.zeros(dim), norm_var=np.ones(dim), w0=np.zeros((2, dim)), b0=np.array([0.5, 0.0]))
    import hashlib
    info = {'format': 'explorer3.simlab.mlp_elu.v1', 'robot': 'tank', 'contract_version': contract.VERSION,
            'contract_hash': contract.contract_hash('tank'), 'history': 3, 'layers': 1, 'scenario': 'dock_v1',
            'scenario_fingerprint': 'x', 'weights_sha256': hashlib.sha256((path / 'policy.npz').read_bytes()).hexdigest()}
    (path / 'policy.json').write_text(json.dumps(info))


def fake_dataset(path, n=12):
    from explorer3.sensors import CHANNELS
    (path / 'meta').mkdir(parents=True)
    (path / 'meta/info.json').write_text(json.dumps({'fps': 20, 'features': {'observation.environment_state': {}, 'action': {}}}))
    env = np.zeros((n, len(CHANNELS)), np.float32)
    env[:, :36] = 2.0
    pd.DataFrame({'episode_index': [0] * n, 'observation.environment_state': list(env),
                  'action': [np.array([0.15, 0.0], np.float32)] * n}).to_parquet((path / 'data').mkdir(parents=True) or path / 'data/file-000.parquet')
    video = path / 'videos/observation.images.front/chunk-000'
    video.mkdir(parents=True)
    with av.open(str(video / 'file-000.mp4'), 'w') as c:
        s = c.add_stream('mpeg4', rate=20)
        s.width, s.height = 640, 400
        for _ in range(n):
            for p in s.encode(av.VideoFrame.from_ndarray(np.full((400, 640, 3), 128, np.uint8), format='bgr24')):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def test_shadow_report(tmp_path):
    pol, ds = tmp_path / 'policy', tmp_path / 'ds'
    pol.mkdir()
    fake_policy(pol)
    fake_dataset(ds)
    out = tmp_path / 'report.json'
    r = subprocess.run([sys.executable, 'scripts/simlab_shadow.py', str(pol), str(ds), '--out', str(out)],
                       cwd=ROOT, env={'PYTHONPATH': str(ROOT), 'PATH': '/usr/bin:/bin'}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    s = json.loads(out.read_text())['summary']
    assert s['frames'] == 12 and s['commands_sent'] == 0
    assert s['marker_visible_fraction'] == 0.0
    assert s['sign_agreement_when_moving'][0] == 1.0      # constant forward proposal vs forward human action
    assert s['proposed_beyond_limits'] == 0
