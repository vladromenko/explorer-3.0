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
import hashlib
import json
import os
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
            'success': outcome == 'success', 'outcome': outcome, 'model_hash': env.manifest['xml_sha256'],
            'geometry': 'box_proxy', 'knowledge': k}


def summarize(rows):
    from scripts.benchmark_stats import wilson
    summary = {}
    for c in ('privileged', 'none', 'so101'):
        sel = [r for r in rows if r['condition'] == c]
        by = {}
        for r in sel:
            by.setdefault(r['object'], []).append(r['success'])
        good = sum(r['success'] for r in sel)
        summary[c] = {'n': len(sel), 'success_rate': round(good / len(sel), 3) if sel else None,
                      'success_ci95': wilson(good, len(sel)),
                      'by_object': {o: f'{sum(v)}/{len(v)}' for o, v in sorted(by.items())}}
    summary['note'] = ('MuJoCo, scripted grasp, rigid box proxies. Privileged teacher knows orientation; '
                       'SO-101 condition supplies noisy size only. No trained policy or hardware evidence.')
    return summary


def save_summary(out, rows, expected):
    summary = summarize(rows)
    summary.update(completed=len(rows) == expected, expected_trials=expected)
    tmp = out / 'summary.tmp'
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    tmp.replace(out / 'summary.json')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=60)
    ap.add_argument('--out', default=str(ROOT / 'docs/results/knowledge_transfer_mujoco'))
    ap.add_argument('--start-seed', type=int, default=9000)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    if a.episodes < 1 or a.start_seed < 0:
        ap.error('Positive episode count and nonnegative seed required')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = ['scripts/knowledge_transfer_mujoco.py', 'explorer3/sim/aligned.py',
               'explorer3/sim/models.py', 'explorer3/sim/vendor_rosmaster/arm_model.py',
               'configs/sim-alignment.json']
    def hashes():
        return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    protocol = {'version': 2, 'episodes': a.episodes, 'start_seed': a.start_seed,
                'conditions': ['privileged', 'none', 'so101'], 'sources': hashes()}
    result_file = out / 'results.jsonl'
    if a.resume:
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            ap.error('Resume requires the same code and parameters')
        rows = [json.loads(line) for line in result_file.read_text().splitlines()] if result_file.exists() else []
    else:
        if result_file.exists():
            ap.error('Preserve existing results: choose a new --out or use --resume')
        (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        rows = []
    done = {(r['seed'], r['condition']) for r in rows}
    for i in range(a.episodes):
        for c in ('privileged', 'none', 'so101'):
            seed = a.start_seed + i
            if (seed, c) in done:
                continue
            if hashes() != protocol['sources']:
                raise RuntimeError('Source changed during the experiment; use a frozen checkout')
            r = run(seed, c)
            rows.append(r)
            with result_file.open('a') as stream:
                stream.write(json.dumps(r) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            save_summary(out, rows, 3 * a.episodes)
            print(c, r['seed'], r['object'], r['success'], r['outcome'], flush=True)
    summary = save_summary(out, rows, 3 * a.episodes)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
