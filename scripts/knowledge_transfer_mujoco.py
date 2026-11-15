"""Root experiment at the MuJoCo level: does grasp knowledge from a helper make ROSMASTER's floor
grasp succeed more often on objects it has not seen?

Same seeds (object, position, yaw, start pose) for every condition:
  privileged  teacher with the true object centre (upper bound)
  none        ROSMASTER knows where, not what: assumes the 32 mm cube (no help)
  so101       knowledge from the SO-101 station: true size with its measurement error
              (±1.5 mm, ~ the copied overview fit: 30.1 mm vs 29.5 mm)
Object sizes are modelled household objects (RosmasterFloorPick.OBJECTS), not measurements.
  .venv-sim/bin/python scripts/knowledge_transfer_mujoco.py --episodes 60
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(seed, condition):
    from explorer3.sim.aligned import RosmasterFloorPick, ScriptedFloorPick
    env = RosmasterFloorPick(vary_objects=True)
    objs = sorted(RosmasterFloorPick.OBJECTS)
    env.reset(seed=seed, options={'object': objs[seed % len(objs)]})
    half = env.object_half()
    if condition == 'privileged':
        k = None
    elif condition == 'none':
        k = 'none'
    else:
        err = np.random.default_rng(seed + 99).normal(0, .0015, 2)
        k = {'width_m': float(2 * min(half[:2]) + err[0]), 'height_m': float(2 * half[2] + err[1])}
    try:
        expert = ScriptedFloorPick(env, knowledge=k)
        info = {}
        while True:
            _, _, term, trunc, info = env.step(expert.action())
            if term or trunc:
                break
        outcome = info['outcome']
    except RuntimeError as exc:
        outcome = 'ik_failure: ' + str(exc)
    env.close()
    return {'seed': seed, 'condition': condition, 'object': env.object_name, 'size_m': (2 * half).round(4).tolist(),
            'success': outcome == 'success', 'outcome': outcome}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=60)
    ap.add_argument('--out', default=str(ROOT / 'docs/results/knowledge_transfer_mujoco'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(a.episodes):
        for c in ('privileged', 'none', 'so101'):
            r = run(9000 + i, c)
            rows.append(r)
            print(c, r['seed'], r['object'], r['success'], r['outcome'], flush=True)
    (out / 'results.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    summary = {}
    for c in ('privileged', 'none', 'so101'):
        sel = [r for r in rows if r['condition'] == c]
        by = {}
        for r in sel:
            by.setdefault(r['object'], []).append(r['success'])
        summary[c] = {'n': len(sel), 'success_rate': round(sum(r['success'] for r in sel) / len(sel), 3),
                      'by_object': {o: f'{sum(v)}/{len(v)}' for o, v in sorted(by.items())}}
    summary['note'] = 'MuJoCo, scripted grasp with the console IK; modelled object sizes; level of evidence: simulation'
    summary['created'] = time.strftime('%Y-%m-%d %H:%M')
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
