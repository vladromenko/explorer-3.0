"""Arm policy from the scripted teacher: demos (cached) -> behaviour cloning -> numpy artifact ->
evaluation of the artifact (numpy runtime) in CPU MuJoCo on frozen seeds.

Usage: MUJOCO_GL=egl PYTHONPATH=. .venv-sim/bin/python scripts/simlab_arm_bc.py [--demos 200] [--episodes 50] [--name so101_bc_v1]
"""
import argparse
import json

import numpy as np
import torch

from explorer3.paths import DATA
from explorer3.simlab import artifact, scenario
from explorer3.simlab.arm_demos import collect
from explorer3.simlab.arm_warp import MAX_STEP
from explorer3.simlab.ppo import ActorCritic


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--demos', type=int, default=200)
    a.add_argument('--episodes', type=int, default=50)
    a.add_argument('--epochs', type=int, default=200)
    a.add_argument('--name', default='so101_bc_v1')
    args = a.parse_args()
    cache = DATA / f'simlab/demos/so101_{args.demos}.npz'
    if cache.exists():
        z = np.load(cache); X, Y = z['X'], z['Y']
    else:
        X, Y, _ = collect(args.demos)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=X, Y=Y)
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ActorCritic(X.shape[1], 1, 6, (512, 512, 512)).to(dev)
    Xg, Yg = torch.tensor(X, device=dev), torch.tensor(Y, device=dev)
    model.actor_norm.update(Xg)
    opt = torch.optim.Adam(model.actor.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for e in range(args.epochs):
        perm = torch.randperm(len(Xg), device=dev)
        for j in range(0, len(Xg), 1024):
            i = perm[j:j + 1024]
            loss = ((model.actor(model.actor_norm(Xg[i])) - Yg[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    meta = {'robot': 'so101', 'history': 3, 'method': 'behaviour cloning of the scripted IK teacher (DART demos)',
            'demos_frames': int(len(X)), 'bc_mse': float(loss), 'epochs': args.epochs}
    out = DATA / 'simlab/policies' / args.name
    artifact.export(model.cpu(), meta, out, scenario='pick_place_v1', scenario_fingerprint=scenario.fingerprint('pick_place_v1'),
                    extra={'assumed_measurements': scenario.assumed_fields(), 'action': 'absolute_normalized_joint_target'})
    # evaluate the exported artifact through the numpy runtime, exactly as on the robot
    from explorer3.sim.environment import SO101PickPlace
    from explorer3.simlab import contract
    pol = artifact.Policy(out)
    env = SO101PickPlace()
    lo, hi = env.ranges[:, 0], env.ranges[:, 1]
    box = env.data.body('target_box').xpos.copy()
    lo_s, _ = scenario.load_scenario('pick_place_v1')['evaluation']['frozen_seeds']
    rng = np.random.default_rng(1)
    records = []
    for k in range(args.episodes):
        env.reset(seed=lo_s + k)
        pol.reset()
        command = env.data.ctrl[:6].copy()
        steps = int(env.deadline_s * env.fps)
        for t in range(steps):
            cube = env.data.body('task_cube').xpos.copy() + rng.normal(0, 0.005, 3)
            obs = contract.build_arm(2 * (env.data.qpos[:6] - lo) / (hi - lo) - 1, env.data.qvel[:6], cube, 1, 0.0, box, t / steps)
            target = lo + (pol.act_target(obs) + 1) / 2 * (hi - lo)
            command = command + np.clip(target - command, -MAX_STEP, MAX_STEP)
            _, _, te, tr, info = env.step(env.from_radians(command))
            if te or tr:
                break
        records.append({'seed': lo_s + k, 'outcome': info['outcome'], 'steps': t + 1})
    k = sum(r['outcome'] == 'success' for r in records)
    summary = {'artifact': str(out), 'engine': 'mujoco_cpu', 'episodes': len(records), 'success': k / len(records),
               'curriculum': 'easy (cube +-2 cm, as the teacher demos)'}
    (out / 'eval-cpu.json').write_text(json.dumps({'summary': summary, 'episodes': records}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
