"""Help between robots, the article's comparison (RESEARCH_PROTOCOL «AG2»), in the virtual room.

Every episode: «принеси <предмет>»; the object is moved by a «person» 5 s after it was first seen,
so ROSMASTER arrives at a stale place and needs help. Conditions differ only in how the
AssistanceBroker uses the help memory:
  M0  no memory: fixed baseline order;
  M2  memory without the recipient-compatibility filter;
  M3  memory only from cases compatible with the recipient's current skill version.
Each condition starts from the same frozen memory snapshot (optional --seed-memory file) and the
same rooms/seeds (paired). Writes results.jsonl + summary.json; the agents' own outcome verifier
decides helper/recipient/task outcomes.

  .venv/bin/python scripts/swarm_help_experiment.py --rooms 10 --out DATA/simlab/help-experiment
Real time: about 2-4 min per episode.
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def copy_db(src, dst):
    """SQLite copy including the WAL contents (a plain file copy can lose them)."""
    import sqlite3
    with sqlite3.connect(src) as a, sqlite3.connect(dst) as b:
        a.backup(b)


OLD_VERSION = 'e' * 64          # «an earlier version of ROSMASTER's grasp»: memory scope that no longer applies


def episode(seed, comparison, prompt, memory_db, *, frozen=True, in_place=False, old_skill=False, vanish=False):
    from explorer3 import catalog, fleet_map, world_memory, episodes, so101_handoff, swarm_sim, fleet_agents
    from explorer3.cognitive import store
    tmp = Path(tempfile.mkdtemp())
    if in_place:                        # building the memory: write into the shared database itself
        tmp = Path(memory_db).parent
    else:
        copy_db(memory_db, tmp / 'catalog.sqlite3')
    real_skills = getattr(fleet_agents, '_real_fleet_skills', fleet_agents.fleet_skills)
    fleet_agents._real_fleet_skills = real_skills
    if old_skill:
        def patched():
            cards = real_skills()
            k = 'rosmaster.fleet_pick_floor.v1'
            cards[k] = cards[k].model_copy(update={'version': OLD_VERSION, 'policy_sha256': OLD_VERSION})
            return cards
        fleet_agents.fleet_skills = patched
    else:
        fleet_agents.fleet_skills = real_skills
    catalog.DATA = store.DATA = tmp
    swarm_sim.SIM, fleet_map.MAP_DIR = tmp / 'sim', tmp / 'map'
    world_memory.MEMORY, episodes.ROOT, so101_handoff.CONFIG = tmp / 'm.json', tmp / 'ep', tmp / 'h.json'
    team = swarm_sim.enter(seed=seed)
    team.agents.comparison = comparison
    team.agents.frozen = frozen
    (swarm_sim.WORLD.vanish_after_seen if vanish else swarm_sim.WORLD.relocate_after_seen).add(prompt)
    started = time.time()
    team.command('принеси ' + {'cookie': 'печенье', 'cube': 'кубик', 'ball': 'мяч'}[prompt], executor='geometric_baseline')
    while team.thread.is_alive() and time.time() - started < 600:
        time.sleep(1)
    st = team.status()
    cases = [c for c in store.cases() if c['id'].startswith('fleet-case:')]
    truth = swarm_sim.truth_overlay()
    swarm_sim.leave()
    return {'seed': seed, 'comparison': comparison, 'object': prompt, 'outcome': st['stage'], 'message': st['message'],
            'seconds': round(time.time() - started, 1), 'helps': st['agents']['helps'],
            'help_choices': [r['state'] for r in st['agents']['roles'] if r['role'] == 'AssistanceBroker'],
            'new_cases': [{k: c[k] for k in ('assistance_type', 'helper_outcome', 'recipient_outcome', 'task_outcome', 'status')} for c in cases],
            'collisions': truth['collisions'], 'delivered': truth['delivered']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rooms', type=int, default=10)
    ap.add_argument('--conditions', default='M0,M2,M3')
    ap.add_argument('--seed-memory', default=None, help='catalog.sqlite3 with the frozen help memory to start from')
    ap.add_argument('--out', default=str(Path.home() / '.local/share/explorer3/simlab/help-experiment'))
    ap.add_argument('--build', type=int, default=8, help='memory-building episodes per kind (useful / outdated)')
    a = ap.parse_args()
    out = Path(a.out) / time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    from explorer3.cognitive import store
    from explorer3 import catalog
    base = out / 'memory-start.sqlite3'
    if a.seed_memory:
        copy_db(a.seed_memory, base)
    else:
        # 1) memory from separate training rooms: useful help for today's grasp, and failed help
        #    recorded for an older grasp version (should not steer today's recipient: M3 vs M2)
        build = out / 'build'
        build.mkdir()
        catalog.DATA = store.DATA = build
        store.initialize()
        prompts = ('cookie', 'cube', 'ball')
        for i in range(a.build):
            r = episode(9000 + i, 'M3', prompts[i % 3], build / 'catalog.sqlite3', frozen=False, in_place=True)
            print('build useful', r['seed'], r['outcome'], r['help_choices'], flush=True)
            r = episode(9500 + i, 'M3', prompts[i % 3], build / 'catalog.sqlite3', frozen=False, in_place=True,
                        old_skill=True, vanish=True)
            print('build outdated', r['seed'], r['outcome'], r['help_choices'], flush=True)
        copy_db(build / 'catalog.sqlite3', base)
    # 2) test rooms never used above; every condition starts from the same frozen memory
    rows = []
    for i in range(a.rooms):
        prompt = ('cookie', 'cube', 'ball')[i % 3]
        for comparison in a.conditions.split(','):
            r = episode(5000 + i, comparison, prompt, base, frozen=True)
            rows.append(r)
            with (out / 'results.jsonl').open('a') as f:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(comparison, r['seed'], r['outcome'], r['seconds'], r['help_choices'], flush=True)
    summary = {}
    for c in a.conditions.split(','):
        sel = [r for r in rows if r['comparison'] == c]
        summary[c] = {'episodes': len(sel), 'delivered': sum(r['outcome'] == 'done' for r in sel),
                      'median_seconds': sorted(r['seconds'] for r in sel)[len(sel) // 2] if sel else None,
                      'helps': sum(r['helps'] for r in sel), 'collisions': sum(r['collisions'] for r in sel)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
