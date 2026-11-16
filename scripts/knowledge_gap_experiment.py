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
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONDITIONS = {'K0': (['rosmaster'], set(), 'off'), 'K1': (['rosmaster', 'tank'], set(), 'off'),
              'K2': (['rosmaster'], {'so101'}, 'off'), 'K3': (['rosmaster', 'tank'], {'so101'}, 'off'),
              'K4': (['rosmaster', 'tank'], {'so101'}, 'decide')}
OBJECTS = {'cookie': 'печенье', 'sock': 'носок', 'cube': 'кубик', 'ball': 'мяч'}
SOURCES = ['scripts/knowledge_gap_experiment.py', 'explorer3/swarm_sim.py', 'explorer3/swarm.py',
           'explorer3/so101_station.py', 'explorer3/knowledge.py', 'explorer3/fleet_agents.py',
           'explorer3/explore.py', 'explorer3/drive.py', 'explorer3/fleet_map.py',
           'explorer3/navigation.py', 'explorer3/sim2d.py', 'explorer3/world_memory.py',
           'explorer3/cognitive/gap_advisor.py', 'configs/cognitive-assets.lock.json']


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def source_hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}


def summarize(rows):
    from scripts.benchmark_stats import wilson
    out = {}
    for c in CONDITIONS:
        sel = [r for r in rows if r['condition'] == c]
        if not sel:
            continue
        good = [r for r in sel if r['delivered']]
        out[c] = {'episodes': len(sel), 'delivered': len(good), 'success_rate': len(good) / len(sel),
                  'success_ci95': wilson(len(good), len(sel)),
                  'median_seconds_all': statistics.median(r['seconds'] for r in sel),
                  'median_seconds_delivered': statistics.median(r['seconds'] for r in good) if good else None,
                  'timeouts': sum(r['outcome'] == 'timeout' for r in sel),
                  'errors': sum(r['outcome'] in ('worker_error', 'error') for r in sel),
                  'advisor_fallbacks': sum(bool(r.get('advisor_fallback')) for r in sel),
                  'knowledge_requests': sum(r.get('knowledge_requests', 0) for r in sel),
                  'recovery_requests': sum(r.get('recovery_requests', 0) for r in sel),
                  'collisions': sum(r.get('collisions', 0) for r in sel)}
    pairs = {}
    solo = {(r['seed'], r['object']): r for r in rows if r['condition'] == 'K0'}
    for c in out:
        if c == 'K0':
            continue
        matched = [(solo[(r['seed'], r['object'])], r) for r in rows if r['condition'] == c
                   and (r['seed'], r['object']) in solo]
        pairs[c] = {'pairs': len(matched),
                    'help_only_successes': sum(not a['delivered'] and b['delivered'] for a, b in matched),
                    'solo_only_successes': sum(a['delivered'] and not b['delivered'] for a, b in matched),
                    'both_successes': sum(a['delivered'] and b['delivered'] for a, b in matched)}
    return {'conditions': out, 'paired_vs_K0': pairs,
            'evidence': '2D simulation: modelled perception and width-dependent grasp; no physical grasp validation',
            'timing': 'wall seconds, including advice; failed/censored runs are not delivery times',
            'handover': 'SO-101 delivery is common to all conditions; K0/K1 exclude its knowledge assistance only'}


def episode(seed, condition, prompt, deadline_s=600):
    """Called only in a fresh worker with EXPLORER3_DATA set before importing Explorer.

    Retain all evidence under that worker's directory. Even on timeout or an exception the
    parent terminates the entire worker process group before starting the next episode.
    """
    from explorer3 import catalog, swarm_sim, episodes, knowledge
    from explorer3.cognitive import store
    catalog.initialize()
    store.initialize()
    robots, helpers, advisor = CONDITIONS[condition]
    team = swarm_sim.enter(seed=seed, robots=robots)
    team.helpers, team.advisor = helpers, advisor
    started = time.monotonic()
    timed_out = False
    try:
        team.command('принеси ' + OBJECTS[prompt], robots, executor='geometric_baseline')
        team.thread.join(timeout=deadline_s)
        timed_out = team.thread.is_alive()
        elapsed = time.monotonic() - started
        if timed_out:
            team.stop('Лимит времени эксперимента')
            team.thread.join(timeout=5)
        st = team.status()
        truth = swarm_sim.truth_overlay()
        recorded = episodes.load()
        fetch = next((e for e in recorded if e['kind'] == 'fetch'), {})
        stages = fetch.get('stages', [])
        advice = [s for s in stages if s['stage'] == 'advise']
        cases = [{k: c[k] for k in ('assistance_type', 'helper_outcome', 'recipient_outcome', 'task_outcome', 'status')}
                 for c in store.cases()]
        # A completed pick (or an interrupted handover) is not delivery to a person.
        delivered = not timed_out and st['stage'] == 'done' and prompt in truth['delivered']
        row = {'seed': seed, 'condition': condition, 'object': prompt,
               'outcome': 'timeout' if timed_out else st['stage'], 'delivered': delivered,
               'message': st['message'], 'seconds': round(min(elapsed, deadline_s), 2),
               'deadline_s': deadline_s, 'robots': robots, 'collisions': truth['collisions'],
               'knowledge_requests': len(knowledge._load()), 'recovery_requests': st['agents']['helps'],
               'advisor_fallback': any(not s['ok'] for s in advice), 'stages': stages,
               'agents': st['agents'], 'help_cases': cases, 'truth': truth,
               'knowledge': knowledge._load()}
        return row
    finally:
        swarm_sim.leave()


def run_worker(request):
    """No network connections from the simulator; Qwen is a separate offline worker."""
    import socket
    original = socket.socket.connect
    def denied(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise PermissionError('Knowledge-gap experiment is simulation-only')
        return original(sock, address)
    socket.socket.connect = denied
    result = episode(request['seed'], request['condition'], request['object'], request['deadline_s'])
    write_json(request['result'], result)


def isolated_episode(out, seed, condition, prompt, deadline_s, model_data):
    directory = out / 'episodes' / f'{seed}-{condition}'
    # An interrupted parent may leave evidence without a committed result row. Preserve it,
    # but never reuse that episode's mutable knowledge/catalogue on resume.
    attempt = 1
    while directory.exists():
        attempt += 1
        directory = out / 'episodes' / f'{seed}-{condition}-attempt{attempt}'
    directory.mkdir(parents=True)
    data = directory / 'data'
    data.mkdir(exist_ok=True)
    # Read the pinned local model from its cache, and honour the shared GPU lock.
    # Catalogues, observations, maps, knowledge and model advice remain private to the episode.
    for relative in ('cache/cognitive', 'locks'):
        source = model_data / relative
        target = data / relative
        if source.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=True)
    result_path = directory / 'result.json'
    request = {'seed': seed, 'condition': condition, 'object': prompt, 'deadline_s': deadline_s,
               'result': str(result_path)}
    write_json(directory / 'request.json', request)
    started = time.monotonic()
    env = {**os.environ, 'EXPLORER3_DATA': str(data), 'PYTHONUNBUFFERED': '1',
           'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'}
    with (directory / 'worker.log').open('w') as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(directory / 'request.json')],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=deadline_s + 45)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # Include any advice child; no late writes/sightings may contaminate the next run.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    if result_path.exists() and proc.returncode == 0:
        row = json.loads(result_path.read_text())
    else:
        row = {**{k: request[k] for k in ('seed', 'condition', 'object', 'deadline_s')},
               'outcome': 'worker_error', 'delivered': False,
               'seconds': round(min(time.monotonic() - started, deadline_s), 2),
               'message': f'Worker exited without a completed result (exit {proc.returncode}); see worker.log'}
    row['artifact_dir'] = str(directory.relative_to(out))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rooms', type=int, default=10)
    ap.add_argument('--conditions', default='K0,K1,K2,K3')
    ap.add_argument('--deadline', type=float, default=600)
    ap.add_argument('--start-seed', type=int, default=7000)
    ap.add_argument('--out', default=str(Path.home() / '.local/share/explorer3/simlab/knowledge-gap'))
    ap.add_argument('--resume', type=Path, help='Continue the exact frozen protocol, skipping completed rows')
    ap.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        run_worker(json.loads(a.worker.read_text()))
        return
    conditions = a.conditions.split(',')
    if a.rooms < 1 or not 0 < a.deadline <= 3600 or not conditions or len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        ap.error('Positive rooms/deadline and unique conditions K0,K1,K2,K3,K4 required')
    protocol = {'version': 2, 'rooms': a.rooms, 'conditions': conditions, 'start_seed': a.start_seed,
                'deadline_s': a.deadline, 'objects': list(OBJECTS), 'sources': source_hashes()}
    out = (a.resume or (Path(a.out) / time.strftime('%Y%m%d-%H%M%S'))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.resume:
        previous = json.loads((out / 'protocol.json').read_text())
        if previous != protocol:
            ap.error('Resume requires the same code, rooms, conditions, seed and deadline')
    else:
        if (out / 'protocol.json').exists():
            ap.error('Output already exists; use --resume')
        write_json(out / 'protocol.json', protocol)
    result_file = out / 'results.jsonl'
    rows = [json.loads(line) for line in result_file.read_text().splitlines()] if result_file.exists() else []
    done = {(r['seed'], r['condition']) for r in rows}
    model_data = Path(os.environ.get('EXPLORER3_DATA', str(Path.home() / '.local/share/explorer3'))).resolve()
    print('Results:', out, flush=True)
    for i in range(a.rooms):
        for c in conditions:
            if (a.start_seed + i, c) in done:
                continue
            if source_hashes() != protocol['sources']:
                raise RuntimeError('Experiment sources changed; keep this run and start a new protocol')
            r = isolated_episode(out, a.start_seed + i, c, list(OBJECTS)[i % len(OBJECTS)], a.deadline, model_data)
            rows.append(r)
            with result_file.open('a') as f:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            write_json(out / 'summary.json', summarize(rows))
            print(c, r['seed'], r['object'], r['outcome'], r['seconds'], flush=True)
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
