"""Train a dock policy on the batched GPU environment and export the artifact.

Usage: PYTHONPATH=. .venv-sim/bin/python scripts/simlab_train.py tank [--minutes 15] [--steps N] [--seed 0]
Short runs (<= 15 min) are pipeline checks; full runs need the user's explicit go-ahead.
"""
import argparse
import json
from functools import partial

from explorer3.paths import DATA
from explorer3.simlab import artifact, scenario
from explorer3.simlab.dock_gpu import DockBatch
from explorer3.simlab.ppo import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument('robot', choices=['tank', 'rosmaster'])
    p.add_argument('--minutes', type=float, default=15)
    p.add_argument('--steps', type=int, default=200_000_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('--name', default=None)
    a = p.parse_args()
    name = a.name or f'{a.robot}_dock_s{a.seed}'
    out = DATA / 'simlab/runs' / name
    make = partial(DockBatch, a.robot)
    model, meta = train(make, robot=a.robot, out_dir=out, total_steps=a.steps, n=a.n, seed=a.seed,
                        max_minutes=a.minutes)
    best = out / 'best.pt'
    if best.exists():
        import torch
        model.load_state_dict(torch.load(best))
    info = artifact.export(model, meta, DATA / 'simlab/policies' / name, scenario='dock_v1',
                           scenario_fingerprint=scenario.fingerprint('dock_v1'),
                           extra={'assumed_measurements': scenario.assumed_fields(),
                                  'dynamics': 'randomized priors; not yet identified from real logs'})
    print(json.dumps({'artifact': str(DATA / 'simlab/policies' / name), 'meta': meta}, indent=1))


if __name__ == '__main__':
    main()
