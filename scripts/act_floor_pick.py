"""ACT on the same K v4 floor-pick pairs (no language), to test whether the v4 SmolVLA failure
(2/20 and 0/20, jaws closing 3-6 cm from the object) is under-training rather than missing data.

Uses the existing Jobs queue (training + sim evaluation on the same 20 evaluation seeds).
Run detached, one heavy job at a time:
  nohup .venv/bin/python scripts/act_floor_pick.py > DATA/runs/.../act.log 2>&1 &
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explorer3 import training_drafts  # noqa: E402
from explorer3.jobs import Jobs  # noqa: E402
from explorer3.paths import DATA  # noqa: E402

V4 = DATA / 'runs/knowledge-policy-v3-20261116/full-v4-lookpose'
OUT = Path(os.environ.get('ACT_OUT', str(V4.parent / 'act-v4')))
PROFILE = os.environ.get('ACT_PROFILE', 'rosmaster_floor_pick_noknowledge_sim')
STEPS = int(os.environ.get('ACT_STEPS', '30000'))
BATCH = int(os.environ.get('ACT_BATCH', '16'))


def save(state):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=1))


def wait(ident):
    while True:
        row = Jobs(None).get(ident)
        if row['status'] == 'completed':
            return row
        if row['status'] in ('failed', 'cancelled', 'interrupted'):
            raise RuntimeError(f"Job {ident}: {row['status']}; {row.get('error')}")
        time.sleep(10)


def main():
    state = json.loads((OUT / 'state.json').read_text()) if (OUT / 'state.json').exists() else {}
    if not state.get('train'):
        v4 = json.loads((V4 / 'state.json').read_text())
        good = [json.loads(p.read_text()) for p in sorted(V4.glob('[0-9][0-9][0-9][0-9][0-9].json'))]
        ids = [r['imports'][PROFILE]['episode_id'] for r in good if r['outcome'] == 'success']
        if len(ids) != 200:
            raise ValueError(f'Expected the 200 v4 pairs, got {len(ids)}')
        draft, config = training_drafts.prepare(PROFILE, 'act', ids, steps=STEPS, batch=BATCH, seed=1000, device='cuda')
        state = {'started': time.time(), 'profile': PROFILE, 'policy': 'act', 'steps': STEPS, 'batch': BATCH,
                 'episodes': len(ids), 'v4_smolvla_scenes': v4['plans'][PROFILE]['scenes'], 'draft': draft,
                 'plan_id': config['plan_id']}
        state['train'] = Jobs(None).start_prepared_training(draft)
        state['stage'] = 'training'
        save(state)
    row = wait(state['train'])
    state['checkpoint'] = row['checkpoint']
    if not state.get('evaluation'):
        state['evaluation'] = Jobs(None).sim_evaluate(state['train'])
        state['stage'] = 'evaluating'
        save(state)
    wait(state['evaluation'])
    result = json.loads((DATA / 'jobs' / state['evaluation'] / 'result.json').read_text())
    state.update(stage='completed', finished=time.time(), success_fraction=result.get('success_fraction'),
                 trials=[{k: t[k] for k in ('seed', 'outcome', 'reason')} for t in result.get('trials', [])])
    save(state)
    print(json.dumps({'success_fraction': state['success_fraction']}))


if __name__ == '__main__':
    main()
