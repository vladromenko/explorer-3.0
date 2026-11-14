"""M8: evaluate a dock policy artifact in full MuJoCo on the frozen evaluation seeds.

Usage: MUJOCO_GL=egl PYTHONPATH=. .venv-sim/bin/python scripts/simlab_eval.py <artifact_dir> [--curriculum 1.0] [--episodes 100]
Writes <artifact_dir>/eval-<curriculum>.json with Wilson 95 % interval.
"""
import argparse
import json
import math
from pathlib import Path

from explorer3.simlab.artifact import Policy
from explorer3.simlab.dock_mujoco import evaluate
from explorer3.simlab.scenario import load_scenario


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('artifact')
    a.add_argument('--curriculum', type=float, default=1.0)
    a.add_argument('--episodes', type=int, default=100)
    args = a.parse_args()
    policy = Policy(args.artifact)
    lo, hi = load_scenario(policy.info['scenario'])['evaluation']['frozen_seeds']
    seeds = list(range(lo, hi))[:args.episodes]
    records = evaluate(policy, policy.robot, seeds, curriculum=args.curriculum)
    k = sum(r['outcome'] == 'success' for r in records)
    summary = {'artifact': args.artifact, 'engine': 'mujoco_full', 'curriculum': args.curriculum, 'episodes': len(records),
               'success': k / len(records), 'success_ci95': wilson(k, len(records)),
               'collision': sum(r['outcome'] == 'collision' for r in records) / len(records),
               'timeout': sum(r['outcome'] == 'timeout' for r in records) / len(records),
               'safety_stops_per_episode': sum(r['safety_stops'] for r in records) / len(records),
               'mean_steps_success': (sum(r['steps'] for r in records if r['outcome'] == 'success') / max(1, k))}
    (Path(args.artifact) / f'eval-{args.curriculum}.json').write_text(json.dumps({'summary': summary, 'episodes': records}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
