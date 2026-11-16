"""Bounded paired K experiment; collection script plus the existing Jobs queue.
No replacement trainer or hardware execution. Touch OUT/pause to stop between stages.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explorer3.paths import DATA, offline_environment
from explorer3.resources import ResourceLease, ExistingTrainingLease
from explorer3 import datasets, training_drafts
from explorer3.jobs import Jobs

OUT = DATA / 'runs/knowledge-policy-v3-20261116/full'
PROTOCOL = ROOT / 'configs/knowledge-policy-v3.json'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def rows():
    return [json.loads(p.read_text()) for p in sorted(OUT.glob('5????.json'))]


def provenance():
    from explorer3.policy_catalog import smolvla_paths
    paths = ('explorer3/sim/aligned.py', 'explorer3/sim/capture.py', 'explorer3/sim/import_episode.py',
             'explorer3/sim/paired_capture.py', 'explorer3/training.py', 'explorer3/datasets.py',
             'explorer3/policy_runtime.py', 'explorer3/sim/evaluate.py', 'explorer3/sim/factory.py',
             'explorer3/sim/environment.py', 'configs/sim-profiles.json', 'scripts/knowledge_paired_training.py')
    return {'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
            'pretrained_sha256': [training_drafts.checkpoint_hash(p) for p in smolvla_paths()]}


def collect(protocol, state):
    from explorer3.sim.paired_capture import KNOWN, CONTROL
    for seed in range(protocol['start_seed'], protocol['start_seed'] + protocol['max_attempts']):
        records = rows()
        if sum(r['outcome'] == 'success' for r in records) >= protocol['successful_pairs']:
            state['stage'] = 'collected'
            save(OUT/'state.json', state)
            return
        if (OUT/'pause').exists():
            state['stage'] = 'paused_between_attempts'
            save(OUT/'state.json', state)
            return
        output = OUT/f'{seed}.json'
        if not output.exists() or not all(name in json.loads(output.read_text()).get('imports', {}) for name in (KNOWN, CONTROL)):
            health = json.load(urllib.request.urlopen('http://127.0.0.1:8030/api/health', timeout=5))
            if health['armed'] or health['session']['thread_active']:
                raise RuntimeError('Physical session is active; collection stopped before next attempt')
            with ResourceLease('compute'), ResourceLease('physical-activity'), ExistingTrainingLease():
                started = time.time()
                args = ['systemd-run', '--user', '--scope', '--collect', '--unit', f'explorer-k-pair-{seed}',
                        '-p', 'MemoryMax=6144M', '-p', 'MemorySwapMax=2048M', '-p', 'OOMPolicy=kill',
                        '/usr/bin/timeout', '600', str(ROOT/'.venv-sim/bin/python'), '-m', 'explorer3.sim.paired_capture',
                        '--seed', str(seed), '--output', str(output), '--external-spool']
                with (OUT/f'{seed}.log').open('a') as log:
                    result = subprocess.run(args, cwd=ROOT, env={**offline_environment(), 'MUJOCO_GL': 'egl'},
                                            stdout=log, stderr=subprocess.STDOUT)
                cost = {'seed': seed, 'exit_code': result.returncode, 'wall_s': time.time()-started}
                with (OUT/'costs.jsonl').open('a') as f:
                    f.write(json.dumps(cost)+'\n')
                if result.returncode:
                    raise RuntimeError(f'Technical capture error for {seed}; original spool/log retained')
        record = json.loads(output.read_text())
        if record['model'] != protocol['model_sha256'] or record['object']['geometry_version'] != 3:
            raise ValueError('Changed simulation geometry')
        if any(record['imports'][name]['readback'] != 'PASS' for name in (KNOWN, CONTROL)):
            raise ValueError('Pair import/readback failed')
        state.update(stage='collecting', last_seed=seed, attempts=len(rows()),
                     successes=sum(r['outcome']=='success' for r in rows()), updated=time.time())
        save(OUT/'state.json', state)
        print(json.dumps({k:state[k] for k in ('stage','last_seed','attempts','successes')}), flush=True)
    if sum(r['outcome']=='success' for r in rows()) == protocol['successful_pairs']:
        state['stage']='collected'
        save(OUT/'state.json',state)
    else:
        raise RuntimeError('Declared generation budget exhausted; no automatic extension')


def wait_job(identifier):
    start = time.time()
    while time.time()-start < 12*3600:
        row = Jobs(None).get(identifier)
        if row['status'] == 'completed':
            return row
        if row['status'] in ('failed','cancelled','interrupted'):
            raise RuntimeError(f"Job {identifier}: {row['status']}; {row.get('error')}")
        time.sleep(5)
    raise RuntimeError('Job monitoring deadline reached; inspect existing Jobs before further execution')


def train(protocol, state):
    good = [r for r in rows() if r['outcome']=='success']
    if len(good) != protocol['successful_pairs']:
        raise ValueError('Exactly the declared completed pairs are required')
    plans = {}
    for name in protocol['profiles']:
        ids = [r['imports'][name]['episode_id'] for r in good]
        draft, config = training_drafts.prepare(name, protocol['policy'], ids, steps=protocol['steps'],
            batch=protocol['batch'], seed=protocol['seed'], device=protocol['device'])
        plan = datasets.load_plan(config['plan_id'])
        plans[name] = {'draft':draft, 'scenes':{role:sorted(e['scene'] for e in group) for role,group in plan['splits'].items()}}
    if plans[protocol['profiles'][0]]['scenes'] != plans[protocol['profiles'][1]]['scenes']:
        raise ValueError('Pair train/validation/test scenes differ')
    state['plans'] = plans
    save(OUT/'state.json',state)
    for name in protocol['profiles']:
        if (OUT/'pause').exists():
            state['stage']='paused_between_jobs'
            save(OUT/'state.json',state)
            return
        if provenance() != state['provenance']:
            raise ValueError('Frozen implementation or initial weights changed')
        run=state.setdefault('jobs',{}).setdefault(name,{})
        if not run.get('train'):
            run['train']=Jobs(None).start_prepared_training(plans[name]['draft'])
            state['stage']='training'
            save(OUT/'state.json',state)
        row=wait_job(run['train'])
        run['checkpoint']=row['checkpoint']
        if not run.get('evaluation'):
            if (OUT/'pause').exists():
                state['stage']='paused_before_evaluation'
                save(OUT/'state.json',state)
                return
            run['evaluation']=Jobs(None).sim_evaluate(run['train'])
            state['stage']='evaluating'
            save(OUT/'state.json',state)
        wait_job(run['evaluation'])
        run['result']=json.loads((DATA/'jobs'/run['evaluation']/'result.json').read_text())
        save(OUT/'state.json',state)
    state.update(stage='completed',finished=time.time())
    save(OUT/'state.json',state)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('collect','train'))
    args=parser.parse_args()
    protocol=json.loads(PROTOCOL.read_text())
    probe=json.loads((OUT.parent/'memory-smoke.json').read_text())
    if probe['status']!='completed' or Jobs(None).get(probe['job'])['status']!='completed':
        raise ValueError('Successful actual trainer memory probe required first')
    OUT.mkdir(parents=True,exist_ok=True)
    state=json.loads((OUT/'state.json').read_text()) if (OUT/'state.json').exists() else {
        'started':time.time(), 'protocol':protocol, 'provenance':provenance(), 'probe_job':probe['job']}
    if state['protocol']!=protocol or state['provenance']!=provenance():
        raise ValueError('Existing run uses a different protocol, source or initial model')
    save(OUT/'state.json',state)
    try:
        {'collect':collect,'train':train}[args.phase](protocol,state)
    except Exception as exc:
        state.update(stage='stopped',error=f'{type(exc).__name__}: {exc}',updated=time.time())
        save(OUT/'state.json',state)
        raise


if __name__=='__main__':
    main()
