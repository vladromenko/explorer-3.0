"""E3 diagnostic (review item "overfit"): can ACT, with the same recipe, memorise a handful of scenes?

Trains ACT on 8 successful v4 scenes (no-knowledge profile, same recipe as v4 except steps), then
evaluates on the TRAIN scenes' own seeds. If it cannot pick even the scenes it was trained on, the
training/inference pipeline is wrong; if it can, v4's failure is data quantity / localisation.

  .venv/bin/python scripts/e3_overfit.py prepare            # freeze plan + draft, no GPU
  .venv/bin/python scripts/e3_overfit.py train              # GPU, ~2 h, loud fan: ask the owner first
  MUJOCO_GL=egl .venv-sim/bin/python scripts/e3_overfit.py evaluate
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explorer3.paths import DATA  # noqa: E402

PAIRS = DATA / 'runs/knowledge-policy-v3-20261116/full-v4-lookpose'
PROFILE = 'rosmaster_floor_pick_noknowledge_sim'
OUT = DATA / 'diagnosis/e3_overfit'
STATE = OUT / 'state.json'


def save(state):
    OUT.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))


def prepare():
    from explorer3 import training_drafts, datasets
    good = [json.loads(p.read_text()) for p in sorted(PAIRS.glob('[0-9]' * 5 + '.json'))]
    good = [r for r in good if r['outcome'] == 'success'][:8]
    ids = [r['imports'][PROFILE]['episode_id'] for r in good]
    draft, config = training_drafts.prepare(PROFILE, 'act', ids, steps=3000, batch=16, seed=1000, device='cuda')
    plan = datasets.load_plan(config['plan_id'])
    by_id = {r['imports'][PROFILE]['episode_id']: r['seed'] for r in good}
    train_seeds = sorted(by_id[e['id']] for e in plan['splits']['train'] if e['id'] in by_id)
    save({'draft': draft, 'plan_id': config['plan_id'], 'episodes': ids, 'train_seeds': train_seeds,
          'split_counts': {k: len(v) for k, v in plan['splits'].items()}})
    print(json.dumps(json.loads(STATE.read_text()), indent=1))


def train():
    from explorer3.jobs import Jobs
    state = json.loads(STATE.read_text())
    state['job'] = Jobs(None).start_prepared_training(state['draft'])
    save(state)
    print('job', state['job'])


def evaluate():
    from explorer3.jobs import Jobs
    from explorer3.sim.evaluate import evaluate as run
    state = json.loads(STATE.read_text())
    row = Jobs(None).get(state['job'])
    if row['status'] != 'completed':
        raise SystemExit('training job: ' + row['status'])
    r = run(Path(row['checkpoint']), OUT / 'eval_train_seeds.json', state['train_seeds'], profile=PROFILE)
    state['train_seed_success'] = sum(t['outcome'] == 'success' for t in r['trials'])
    state['of'] = len(r['trials'])
    save(state)
    print('overfit check: %d/%d train scenes picked' % (state['train_seed_success'], state['of']))


if __name__ == '__main__':
    {'prepare': prepare, 'train': train, 'evaluate': evaluate}[sys.argv[1]]()
