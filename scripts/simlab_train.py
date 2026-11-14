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
    p.add_argument('robot', choices=['tank', 'rosmaster', 'so101'])
    p.add_argument('--minutes', type=float, default=15)
    p.add_argument('--steps', type=int, default=200_000_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n', type=int, default=None)
    p.add_argument('--name', default=None)
    p.add_argument('--demos', type=int, default=0, help='arm: scripted-teacher episodes for behaviour cloning')
    a = p.parse_args()
    task = 'pick_place' if a.robot == 'so101' else 'dock'
    name = a.name or f'{a.robot}_{task}_s{a.seed}'
    out = DATA / 'simlab/runs' / name
    if a.robot == 'so101':
        from explorer3.simlab.arm_warp import ArmBatch
        make = lambda n, seed, curriculum: ArmBatch(n=n, seed=seed, curriculum=curriculum)
    else:
        make = partial(DockBatch, a.robot)
    extra = {}
    if a.robot == 'so101' and a.demos:
        from explorer3.simlab.arm_demos import collect
        import numpy as np
        cache = DATA / f'simlab/demos/so101_{a.demos}.npz'
        if cache.exists():
            z = np.load(cache); X, Y = z['X'], z['Y']
        else:
            X, Y, kept = collect(a.demos)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, X=X, Y=Y)
        extra = {'bc': (X, Y), 'init_log_std': -1.6}
    model, meta = train(make, robot=a.robot, out_dir=out, total_steps=a.steps, n=a.n or (1024 if a.robot == 'so101' else 4096), seed=a.seed,
                        max_minutes=a.minutes, **extra)
    best = out / 'best.pt'
    if best.exists():
        import torch
        model.load_state_dict(torch.load(best))
    if a.robot == 'so101':
        torch_save = out / 'final_model.pt'
        import torch
        torch.save(model.state_dict(), torch_save)
        print(json.dumps({'checkpoint': str(torch_save), 'meta': meta}, indent=1))
        return
    info = artifact.export(model, meta, DATA / 'simlab/policies' / name, scenario='dock_v1',
                           scenario_fingerprint=scenario.fingerprint('dock_v1'),
                           extra={'assumed_measurements': scenario.assumed_fields(),
                                  'dynamics': 'randomized priors; not yet identified from real logs'})
    print(json.dumps({'artifact': str(DATA / 'simlab/policies' / name), 'meta': meta}, indent=1))


if __name__ == '__main__':
    main()
