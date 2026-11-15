"""Root experiment: does help chosen by the recipient's own knowledge gaps make the fleet succeed
more often / faster than the recipient alone or a fixed single helper?

Conditions (same virtual rooms and objects, paired):
  K0  ROSMASTER alone: searches itself, grasps with its own single-view estimate
  K1  + Tank only     : help for WHERE (Tank searches too), no grasp knowledge
  K2  + SO-101 only   : help for HOW (the SO-101 station measures the object), ROSMASTER searches alone
  K3  gap-driven      : ROSMASTER asks for what it is missing — Tank for WHERE, SO-101 for HOW (rules)
  K4  gap-driven, the local language model (Qwen3-VL-2B) names the gaps; validated by code (~36 s/episode)
Writes results.jsonl + summary.json. Level of evidence: the 2D swarm simulation (modelled sensing and
grasp-size dependence) — a check of the coordination logic, not of MuJoCo physics or the real robots.

  .venv/bin/python scripts/knowledge_gap_experiment.py --rooms 10
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONDITIONS = {'K0': (['rosmaster'], set(), 'off'), 'K1': (['rosmaster', 'tank'], set(), 'off'),
              'K2': (['rosmaster'], {'so101'}, 'off'), 'K3': (['rosmaster', 'tank'], {'so101'}, 'off'),
              'K4': (['rosmaster', 'tank'], {'so101'}, 'decide')}
OBJECTS = {'cookie': 'печенье', 'sock': 'носок', 'cube': 'кубик', 'ball': 'мяч'}


def episode(seed, condition, prompt):
    from explorer3 import catalog, fleet_map, world_memory, episodes, so101_handoff, swarm_sim
    from explorer3.cognitive import store
    tmp = Path(tempfile.mkdtemp())
    catalog.DATA = store.DATA = tmp
    store.initialize()
    swarm_sim.SIM, fleet_map.MAP_DIR = tmp / 'sim', tmp / 'map'
    world_memory.MEMORY, episodes.ROOT, so101_handoff.CONFIG = tmp / 'm.json', tmp / 'ep', tmp / 'h.json'
    team = swarm_sim.enter(seed=seed)
    robots, helpers, advisor = CONDITIONS[condition]
    team.helpers, team.advisor = helpers, advisor
    started = time.time()
    team.command('принеси ' + OBJECTS[prompt], robots)
    while team.thread.is_alive() and time.time() - started < 600:
        time.sleep(1)
    st = team.status()
    truth = swarm_sim.truth_overlay()
    swarm_sim.leave()
    return {'seed': seed, 'condition': condition, 'object': prompt, 'outcome': st['stage'], 'message': st['message'][:200],
            'seconds': round(time.time() - started, 1), 'collisions': truth['collisions']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rooms', type=int, default=10)
    ap.add_argument('--conditions', default='K0,K1,K2,K3,K4')
    ap.add_argument('--out', default=str(Path.home() / '.local/share/explorer3/simlab/knowledge-gap'))
    a = ap.parse_args()
    out = Path(a.out) / time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    objs = list(OBJECTS)
    for i in range(a.rooms):
        for c in a.conditions.split(','):
            r = episode(7000 + i, c, objs[i % len(objs)])
            rows.append(r)
            with (out / 'results.jsonl').open('a') as f:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(c, r['seed'], r['object'], r['outcome'], r['seconds'], flush=True)
    summary = {c: {'episodes': len(sel), 'delivered': sum(r['outcome'] == 'done' for r in sel),
                   'median_seconds': sorted(r['seconds'] for r in sel)[len(sel) // 2] if sel else None}
               for c in a.conditions.split(',') for sel in [[r for r in rows if r['condition'] == c]]}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
