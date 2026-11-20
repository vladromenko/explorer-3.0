"""E3 diagnostic (review item "execution horizon"): same trained checkpoint, replan more often.

The v4 checkpoints execute the whole predicted chunk open-loop (ACT 100 steps = 5 s, SmolVLA 50 = 2.5 s),
so the approach to the object is planned from one frame. This re-evaluates the SAME weights on the SAME
seeds with n_action_steps = 1/10/25/... (no retraining). Inference only (GPU, light).

  MUJOCO_GL=egl .venv-sim/bin/python scripts/e3_horizon_sweep.py <checkpoint> \
      --profile rosmaster_floor_pick_noknowledge_sim --steps 10 25 --seeds 10000-10019 --out DATA/diagnosis/e3_horizon_act
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def seeds(text):
    a, _, b = text.partition('-')
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('--profile', required=True)
    ap.add_argument('--steps', type=int, nargs='+', default=[10, 25])
    ap.add_argument('--seeds', type=seeds, default=seeds('10000-10019'))
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    from explorer3.sim.evaluate import evaluate
    a.out.mkdir(parents=True, exist_ok=True)
    table = {}
    for n in a.steps:
        os.environ['EXPLORER_N_ACTION_STEPS'] = str(n)   # read by the policy worker (policy_runtime)
        out = a.out / f'n{n}.json'
        if not out.exists() or json.loads(out.read_text()).get('status') != 'completed':
            evaluate(a.checkpoint, out, a.seeds, profile=a.profile)
        r = json.loads(out.read_text())
        table[n] = {'success': sum(t['outcome'] == 'success' for t in r['trials']), 'of': len(r['trials']),
                    'success_seeds': [t['seed'] for t in r['trials'] if t['outcome'] == 'success']}
        print(n, table[n], flush=True)
    (a.out / 'summary.json').write_text(json.dumps({'checkpoint': str(a.checkpoint), 'profile': a.profile,
                                                     'n_action_steps': table}, indent=1))


if __name__ == '__main__':
    main()
