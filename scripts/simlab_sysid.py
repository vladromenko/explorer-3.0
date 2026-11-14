"""M4: identify base drive parameters from real `*_rgbd_lidar` recordings.

Yaw channel: commanded yaw rate (action) vs IMU gyro z (observation.environment_state).
Forward channel: commanded forward speed vs measured speed — ROSMASTER odometry (odom_vx,
wheel encoders) when present, otherwise the rate of change of the front lidar sector range
while the robot faces the target (Tank has no wheel feedback).
Writes ~/.local/share/explorer3/simlab/sysid/<robot>.json; the GPU training environment uses
the identified ranges only for fits accepted on hold-out data (> 10 % better than nominal).

Usage: PYTHONPATH=. .venv-sim/bin/python scripts/simlab_sysid.py <robot> <dataset_dir> [<dataset_dir> ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from explorer3.paths import DATA
from explorer3.simlab import sysid_base

GYRO_Z, IMU_VALID = 36 + 36 + 48 + 48 + 2, 36 + 36 + 48 + 48 + 6
FRONT = [0, 1, 2, 34, 35]


def episodes(dataset, robot):
    frames = pd.concat([pd.read_parquet(p) for p in sorted(Path(dataset).joinpath('data').rglob('*.parquet'))])
    fps = 20.0
    yaw, fwd = [], []
    for _, ep in frames.groupby('episode_index'):
        env = np.stack(ep['observation.environment_state'].to_numpy())
        act = np.stack(ep['action'].to_numpy())
        if env[:, IMU_VALID].mean() > 0.9:
            yaw.append((act[:, -1], env[:, GYRO_Z]))
        if robot == 'rosmaster' and env.shape[1] > 200:
            from explorer3.sensors import ROSMASTER_CHANNELS
            i = ROSMASTER_CHANNELS.index('odom_vx')
            if env[:, ROSMASTER_CHANNELS.index('odom_valid')].mean() > 0.9:
                fwd.append((act[:, 0], env[:, i]))
        else:
            front = np.where(env[:, FRONT] > 0, env[:, FRONT], np.nan)
            r = np.nanmedian(front, axis=1)
            speed = -np.gradient(r) * fps
            ok = np.isfinite(speed)
            if ok.mean() > 0.9:
                fwd.append((act[ok, 0], np.clip(speed[ok], -0.6, 0.6)))
    return yaw, fwd


def main():
    robot, datasets = sys.argv[1], sys.argv[2:]
    yaw, fwd = [], []
    for d in datasets:
        y, f = episodes(d, robot)
        yaw += y
        fwd += f
    fits = []
    if yaw:
        fits.append(sysid_base.fit_channel(yaw, channel='yaw'))
    if fwd:
        fits.append(sysid_base.fit_channel(fwd, channel='forward'))
    if not fits:
        raise SystemExit('No usable episodes: record with the "+ глубина и лидар" profile')
    out = DATA / 'simlab/sysid'
    out.mkdir(parents=True, exist_ok=True)
    payload = sysid_base.save(out / f'{robot}.json', fits, {'datasets': datasets})
    for f in fits:
        print(f.as_dict())
    print('ranges:', payload['ranges'])


if __name__ == '__main__':
    main()
