"""Render rollout videos of a simlab policy in full MuJoCo into <policy>/videos/.

Usage: MUJOCO_GL=egl PYTHONPATH=. .venv-sim/bin/python scripts/simlab_video.py <policy_dir> [--episodes 4] [--curriculum 1.0]
"""
import argparse
import json
from pathlib import Path

from explorer3.simlab import video
from explorer3.simlab.scenario import load_scenario


def main():
    a = argparse.ArgumentParser()
    a.add_argument('policy')
    a.add_argument('--episodes', type=int, default=4)
    a.add_argument('--curriculum', type=float, default=1.0)
    args = a.parse_args()
    info = json.loads((Path(args.policy) / 'policy.json').read_text())
    lo, _ = load_scenario(info['scenario'])['evaluation']['frozen_seeds']
    seeds = list(range(lo + 500, lo + 500 + args.episodes))          # not the evaluation seeds
    out = Path(args.policy) / 'videos'
    if info['robot'] in ('tank', 'rosmaster'):
        index = video.dock_videos(args.policy, seeds, out, args.curriculum)
    else:
        index = video.arm_videos(args.policy, seeds, out)
    print(json.dumps(index, ensure_ascii=False))


if __name__ == '__main__':
    main()
