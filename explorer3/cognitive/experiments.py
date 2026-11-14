"""Finite paired M0-M3 plans. Preparation freezes inputs and never queues work."""
import hashlib
import json
from explorer3.paths import DATA
from explorer3.training_drafts import save_content
from . import registry,store,protocol

METHODS=('M0','M1','M2','M3')


def prepare(seeds,scene='workshop',split='validation'):
    from explorer3.sim.room import PRESETS
    if scene not in PRESETS or split not in ('validation','test'):raise ValueError('Unknown scene or split')
    if not seeds or len(seeds)>100 or len(set(seeds))!=len(seeds):raise ValueError('One to 100 distinct seeds required')
    if any(type(s) is not int or not 0<=s<2**31 for s in seeds):raise ValueError('Invalid seed')
    snapshot=store.memory_snapshot()
    plan={'schema':'agent_comparison_v1','scenario':'S1','domain':'sim','scene':scene,'seeds':seeds,'split':split,
        'methods':list(METHODS),'memory_mode':'frozen','memory_updates':False,'memory_snapshot':snapshot,
        'source_prefix':[c['id'] for c in snapshot['cases']],
        'skill_versions':{k:c.version for k,c in registry.cards().items()},
        'protocol_version':protocol.VERSION,'choices':list(protocol.CHOICES),
        'source_hashes':{p:registry.file_hash(p) for p in ('explorer3/cognitive/protocol.py',
            'explorer3/cognitive/run.py','explorer3/cognitive/model.py','explorer3/cognitive/store.py',
            'explorer3/sim/cooperation.py','configs/cognitive-model.json','configs/cognitive-assets.lock.json')},
        'budgets':{'model_steps':4,'format_repairs':1,'replans_per_reason':2,'memory_cases':5,
            'helper_sim_seconds':5,'recipient_sim_seconds':30,'model_wall_seconds':100},
        'primary_metric':'paired complete-task success; unknown/cancelled/hardware_error reported separately',
        'unit_of_replication':'scene seed, never camera frame',
        'initial_state_rule':'same scene, seed, model and initial physics digest within each pair',
        'training_started':False,'weights_frozen':True,'physical_access':False,
        'cells':[{'id':f'{seed}-{method}','seed':seed,'comparison':method} for seed in seeds for method in METHODS]}
    return save_content(DATA/'cognitive/comparisons',plan),plan


def load(ident,*,verify_current=True):
    if len(ident)!=64 or any(c not in '0123456789abcdef' for c in ident):raise ValueError('Invalid comparison plan')
    raw=(DATA/'cognitive/comparisons'/(ident+'.json')).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ident:raise ValueError('Comparison plan changed')
    plan=json.loads(raw)
    if store.digest(plan['memory_snapshot']['cases'])!=plan['memory_snapshot']['sha256']:raise ValueError('Memory snapshot changed')
    if verify_current:
        for path,sha in plan['source_hashes'].items():
            if registry.file_hash(path)!=sha:raise ValueError('Implementation changed; prepare a new comparison')
        if {k:c.version for k,c in registry.cards().items()}!=plan['skill_versions']:raise ValueError('Skill contracts changed')
    return plan


def cell_config(ident,cell_id):
    plan=load(ident);cell=next((c for c in plan['cells'] if c['id']==cell_id),None)
    if cell is None:raise ValueError('Unknown comparison cell')
    return {'kind':'cognitive_simulation','comparison_plan':ident,'comparison_cell':cell_id,
        'seed':cell['seed'],'comparison':cell['comparison'],'scene':plan['scene'],
        'device':'cuda','training_requested':False,'real_authorized':False}


def report(plan_id,runs):
    """Every planned cell is visible, including missing and interrupted ones."""
    plan=load(plan_id,verify_current=False);rows=[];seen={};problems=[]
    for run in runs:
        if run.get('comparison_plan')!=plan_id:raise ValueError('Result belongs to another comparison')
        cell=run.get('comparison_cell')
        if cell in seen:raise ValueError('Duplicate comparison result; choose the preregistered attempt')
        seen[cell]=run
    if set(seen)-{c['id'] for c in plan['cells']}:raise ValueError('Unknown result cell')
    for seed in plan['seeds']:
        hashes={seen[c['id']].get('initial_state_sha256') for c in plan['cells'] if c['seed']==seed and c['id'] in seen}
        if len(hashes)>1 or None in hashes:problems.append(f'{seed}: initial states not verified equal')
    for method in METHODS:
        selected=[seen[c['id']] for c in plan['cells'] if c['comparison']==method and c['id'] in seen]
        rows.append({'method':method,'planned':len(plan['seeds']),'reported':len(selected),
            'success':sum(r.get('outcome')=='success' for r in selected),
            'failure':sum(r.get('outcome')=='failure' for r in selected),
            'unknown_or_technical':sum(r.get('outcome') not in ('success','failure') for r in selected),
            'wall_time_s':sum(r.get('wall_time_s',0) for r in selected),
            'harmful_help':sum(r.get('helper')=='success' and r.get('recipient')=='failure' for r in selected)})
    return {'plan_id':plan_id,'rows':rows,'problems':problems,
        'complete':len(seen)==len(plan['cells']) and not problems,'paired_by':'scene seed',
        'scientific_effect_established':False,'physical_results':False}


def saved_report(plan_id):
    from explorer3 import catalog
    runs=[]
    for row in catalog.rows('jobs'):
        config=json.loads(row['config'])
        if config.get('comparison_plan')==plan_id:
            path=DATA/'jobs'/row['id']/'result.json'
            if row['status']=='completed' and path.is_file():
                result=json.loads(path.read_text())
            else:
                result={'comparison_plan':plan_id,'comparison_cell':config['comparison_cell'],
                    'outcome':'cancelled' if row['status']=='cancelled' else 'unknown',
                    'job_status':row['status'],'error':row.get('error')}
            runs.append(result)
    return report(plan_id,runs)
