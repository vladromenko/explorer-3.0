"""M8.2: shadow mode on real recordings — the policy sees the real observations frame by
frame and proposes actions; nothing is sent to a robot.

Needs a recording of a base profile with `observation.environment_state` (the
`*_rgbd_lidar` profiles) and the front camera with the target marker in view.
Reports agreement with the operator's actions, out-of-distribution inputs and latency.

Usage (main environment, decodes AV1 with PyAV):
  PYTHONPATH=. .venv/bin/python scripts/simlab_shadow.py <policy_dir> <dataset_dir> [--out report.json]
"""
import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd

from explorer3.simlab import contract
from explorer3.simlab.artifact import Policy
from explorer3.simlab.perception import MarkerDetector, TargetTracker
from explorer3.simlab.scenario import load_measurements


def frames(dataset, key='observation.images.front'):
    for path in sorted((Path(dataset) / 'videos' / key).rglob('*.mp4')):
        with av.open(str(path)) as c:
            for f in c.decode(video=0):
                yield f.to_ndarray(format='bgr24')


def main():
    a = argparse.ArgumentParser()
    a.add_argument('policy')
    a.add_argument('dataset')
    a.add_argument('--out')
    args = a.parse_args()
    policy = Policy(args.policy)
    robot = policy.robot
    info = json.loads((Path(args.dataset) / 'meta/info.json').read_text())
    if 'observation.environment_state' not in info['features']:
        raise SystemExit('Recording has no environment features: record with the "+ глубина и лидар" profile')
    evidence = json.loads((Path(__file__).resolve().parents[1] / 'configs/tank-hardware-evidence.json').read_text())
    K = np.asarray(evidence['installed_camera']['intrinsics']['k']).reshape(3, 3)   # Tank; ROSMASTER: camera_info
    meas = load_measurements()
    detector = MarkerDetector(K, side=meas['dock_target']['marker']['side'], marker_id=meas['dock_target']['marker']['id'])
    data = pd.concat([pd.read_parquet(p) for p in sorted((Path(args.dataset) / 'data').rglob('*.parquet'))])
    limits = np.asarray(contract.ROBOTS[robot]['policy_limits'])
    rows, tracker, prev, latency, episode = [], TargetTracker(), np.zeros(len(limits)), [], None
    for (_, row), image in zip(data.iterrows(), frames(args.dataset)):
        if row['episode_index'] != episode:
            episode, tracker, prev = row['episode_index'], TargetTracker(), np.zeros(len(limits))
            policy.reset()
        target = tracker.update(detector.detect(image), 1 / info['fps'])
        obs = contract.from_real_features(robot, row['observation.environment_state'], target, prev,
                                          meas[robot]['lidar'])
        t = time.perf_counter()
        proposed = policy.act(obs)
        latency.append(time.perf_counter() - t)
        human = np.clip(np.asarray(row['action'], float) / limits, -1, 1)
        z = (np.repeat(obs[None], policy.k, 0).ravel() - policy.mean) / policy.std
        rows.append({'episode': int(episode), 'visible': target['visible'], 'proposed': proposed.tolist(),
                     'human': human.tolist(), 'ood': bool(np.abs(z).max() > 6)})
        prev = human
    P, H = np.array([r['proposed'] for r in rows]), np.array([r['human'] for r in rows])
    moving = np.abs(H).max(1) > 0.05
    report = {'policy': args.policy, 'dataset': args.dataset, 'frames': len(rows),
              'marker_visible_fraction': float(np.mean([r['visible'] for r in rows])),
              'mae_normalized': np.abs(P - H).mean(0).round(3).tolist(),
              'sign_agreement_when_moving': (np.sign(P[moving]) == np.sign(H[moving])).mean(0).round(3).tolist() if moving.any() else None,
              'out_of_distribution_fraction': float(np.mean([r['ood'] for r in rows])),
              'proposed_beyond_limits': int((np.abs(P) > 1.0 + 1e-6).sum()),
              'latency_ms_p99': float(np.percentile(latency, 99) * 1000), 'commands_sent': 0}
    out = Path(args.out) if args.out else Path(args.policy) / f'shadow-{Path(args.dataset).name}.json'
    out.write_text(json.dumps({'summary': report, 'frames': rows}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
