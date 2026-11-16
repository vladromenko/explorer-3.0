"""Bounded diagnosis of a failed floor-grasp policy in MuJoCo (why 0/20?).

Runs the checkpoint on a few evaluation seeds of a knowledge profile (same per-episode task text as
sim/evaluate.py) and records per step: policy action, measured joints, gripper, TCP and object
positions. Summarises per episode: closest TCP-object distance and when, whether/where the jaws
closed, object displacement and lift, and the action range against the scripted teacher's range on
the same seed. Run AFTER training jobs (uses the GPU):

  MUJOCO_GL=egl .venv-sim/bin/python scripts/diagnose_policy_grasp.py <checkpoint_dir> \
      --profile rosmaster_floor_pick_knowledge_sim --seeds 10000 10001 10002 --out DATA/diagnosis/<name>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def teacher_actions(profile, seed):
    from explorer3.sim.factory import make, EXPERTS
    env = make(profile)
    env.reset(seed=seed)
    ex = EXPERTS[profile](env)
    acts = []
    while True:
        a = ex.action()
        acts.append(a)
        _, _, t, tr, info = env.step(a)
        if t or tr:
            break
    env.close()
    return np.array(acts), info['outcome']


def run(checkpoint, profile, seed, client):
    from explorer3.sim.factory import make
    from explorer3.profiles import profile as profile_config
    env = make(profile, images=True)
    cfg = profile_config(profile)
    obs, _ = env.reset(seed=seed)
    client.request('reset')
    task = getattr(env, 'task_text', None) or cfg['task']
    rows = []
    obj0 = env.data.body('task_cube').xpos.copy()
    while True:
        a = client.action(obs, {**cfg, 'task': task})
        obs, _, t, tr, info = env.step(a)
        ee = env.data.site('gripperframe').xpos.copy()
        ob = env.data.body('task_cube').xpos.copy()
        rows.append({'action': [round(float(v), 2) for v in a], 'state': [round(float(v), 2) for v in obs['state']],
                     'tcp': ee.round(4).tolist(), 'object': ob.round(4).tolist(),
                     'dist': round(float(np.linalg.norm(ee - ob)), 4), 'held': bool(info.get('held'))})
        if t or tr:
            break
    env.close()
    acts = np.array([r['action'] for r in rows])
    d = np.array([r['dist'] for r in rows])
    grip = acts[:, 5]
    i_min = int(d.argmin())
    closed = np.where(grip > 100)[0]          # native gripper degrees: ~30 open, >100 closing
    objs = np.array([r['object'] for r in rows])
    return {'seed': seed, 'task': task, 'outcome': info['outcome'], 'reason': info['termination_reason'],
            'min_tcp_object_m': float(d.min()), 'min_at_step': i_min, 'steps': len(rows),
            'first_close_step': int(closed[0]) if len(closed) else None,
            'dist_at_first_close_m': float(d[closed[0]]) if len(closed) else None,
            'object_moved_m': float(np.linalg.norm(objs[-1, :2] - obj0[:2])), 'object_max_lift_m': float(objs[:, 2].max() - obj0[2]),
            'action_min': acts.min(0).round(1).tolist(), 'action_max': acts.max(0).round(1).tolist(),
            'action_std_last_100': acts[-100:].std(0).round(2).tolist()}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('--profile', default='rosmaster_floor_pick_knowledge_sim')
    ap.add_argument('--seeds', type=int, nargs='+', default=[10000, 10001, 10002])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from explorer3.act_client import ACTClient
    client = ACTClient(a.checkpoint, a.out / 'inference.log', device=a.device)
    summary = []
    try:
        for seed in a.seeds:
            s, rows = run(a.checkpoint, a.profile, seed, client)
            t_acts, t_out = teacher_actions(a.profile, seed)
            s.update(teacher_outcome=t_out, teacher_action_min=t_acts.min(0).round(1).tolist(),
                     teacher_action_max=t_acts.max(0).round(1).tolist(),
                     teacher_first_close_step=int(np.where(t_acts[:, 5] > 100)[0][0]) if (t_acts[:, 5] > 100).any() else None)
            (a.out / f'seed{seed}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
            summary.append(s)
            print(json.dumps(s, ensure_ascii=False), flush=True)
    finally:
        client.close()
    (a.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
