"""Explicit sim -> real initialization on REAL-only data, no robot or training access."""
import json
from pathlib import Path
from . import datasets
from .paths import DATA
from .profiles import profile
from .policy_catalog import validate_recipe
from .training_drafts import save_content,checkpoint_hash,validate_budget

TARGETS=('rosmaster_pick_place','rosmaster_dock')


def contract(cfg):
    return {'task':cfg['task'],'fps':cfg['fps'],'camera':cfg['camera'],'image_shape':cfg.get('image_shape',[480,640,3]),
        'actions':cfg['channels'],'action_units':cfg['units'],
        'state':cfg.get('state_channels',cfg['channels']),'state_units':cfg.get('state_units',cfg['units'])}


def check_source(source,target):
    if target not in TARGETS:raise ValueError('Выберите задачу ROSMASTER')
    cfg=json.loads(source['config'])
    if source['status']!='completed' or cfg.get('kind','train')!='train' or not source.get('checkpoint'):
        raise ValueError('Нужны завершённые веса из симуляции')
    sim=profile(source['profile']);real=profile(target)
    if sim.get('domain')!='sim' or sim.get('real_profile')!=target:raise ValueError('Политика относится к другой задаче')
    # This is offline weight initialization, not permission to deploy a new policy.
    validate_recipe(source['policy'],source['profile'])
    plan=datasets.load_plan(cfg['plan_id'])
    if plan['domain']!='sim' or plan['profile']!=source['profile']:raise ValueError('Исходный план не соответствует весам')
    if contract(plan['profile_contract'])!=contract(real) or contract(sim)!=contract(real):
        raise ValueError('Камера, частота, единицы или порядок каналов различаются; нужен явный адаптер')
    root=Path(source['checkpoint']);weights=json.loads((root/'config.json').read_text())
    state=weights['input_features'].get('observation.state',{})
    camera=weights['input_features'].get('observation.images.'+real['camera'],{})
    h,w,c=real.get('image_shape',[480,640,3])
    if weights.get('type')!=source['policy'] or state.get('shape')!=[len(real.get('state_channels',real['channels']))] or camera.get('shape')!=[c,h,w] or weights['output_features']['action']['shape']!=[len(real['channels'])]:
        raise ValueError('Фактические входы/выходы checkpoint несовместимы с ROSMASTER')
    return plan,contract(real),checkpoint_hash(root)


def prepare(source,target,episode_ids,*,steps,batch,seed,device):
    validate_budget(steps,batch,seed,device)
    old,channels,fingerprint=check_source(source,target)
    plan_id,plan=datasets.freeze(target,False,seed=seed,selected_ids=episode_ids,min_real_episodes=3)
    source_plan_id=json.loads(source['config'])['plan_id']
    frames={role:sum(e['frames'] for e in rows) for role,rows in plan['splits'].items()}
    common={'version':1,'kind':'train','profile':target,'policy':source['policy'],'smoke':False,
        'plan_id':plan_id,'steps':steps,'batch_size':batch,'seed':seed,'device':device,
        'source_job':source['id'],'source_plan_id':source_plan_id,'source_sim_model_hash':old.get('sim_model_hash'),
        'transfer_contract':channels,'normalization':'real train only; official non-resume processor statistics override',
        'sampling':'real train frames only; no simulator samples relabelled as physical',
        'split_counts':{role:len(rows) for role,rows in plan['splits'].items()},'frames':frames,
        'mixture':{'sim_train_frames':0,'real_train_frames':frames['train']},'training_started':False,
        'preparation_only':True,'physical_deployment_authorized':False,'pilot':True,
        'limitation':'Three independent groups are a pipeline pilot, not proof of transfer or adequate sample size'}
    transfer={**common,'mode':'sim_to_real_new_optimizer','initial_checkpoint':source['checkpoint'],
        'initial_checkpoint_sha256':fingerprint,'source_smoke':bool(old['smoke'])}
    scratch={**common,'mode':'matched_real_only_from_scratch','initial_checkpoint':None,
        'initial_checkpoint_sha256':None,'source_job':None,'source_plan_id':None}
    transfer_id=save_content(DATA/'training-drafts',transfer);scratch_id=save_content(DATA/'training-drafts',scratch)
    comparison={'version':1,'target':target,'transfer_draft':transfer_id,'scratch_draft':scratch_id,
        'target_plan_id':plan_id,'shared_budget':{'steps':steps,'batch':batch,'seed':seed,'device':device},
        'source_job':source['id'],'training_started':False,'physical_deployment_authorized':False,
        'comparison_rule':'identical real splits and update budget; only initialization differs'}
    ident=save_content(DATA/'transfer-plans',comparison)
    return ident,comparison


def inventory(target):
    """Local provenance audit, no camera/agent probes and no artificial examples."""
    if target not in TARGETS:raise ValueError('Unknown target')
    from . import catalog
    result=[]
    for entry in catalog.rows('episodes'):
        if entry['profile']!=target:continue
        reason=None
        try:
            meta=json.loads((Path(entry['dataset'])/'provenance.json').read_text())
            if meta.get('domain','real')!='real' or meta.get('synthetic') is not False:reason='Не реальная запись'
            elif entry['excluded']:reason='Исключена оператором'
            elif meta.get('hardware',{}).get('pipeline_check'):reason='Диагностический клип, не показ'
            elif meta.get('hardware',{}).get('training_eligible') is False:reason='Не принята для обучения'
            elif entry['category'] not in {'demonstration','correction'}:reason='Испытание, не отдельный показ'
            elif entry['outcome']!='success':reason='Нет успешного исхода'
            elif meta.get('command_source')!='human':reason='Не человеческий показ'
        except (OSError,ValueError,KeyError):reason='Нет читаемого происхождения записи'
        result.append({**entry,'eligible':reason is None,'reason':reason or 'Пригодна для проверки чтения'})
    return result


def load_comparison(ident):
    import hashlib
    from .training_drafts import load
    if len(ident)!=64 or any(c not in '0123456789abcdef' for c in ident):raise ValueError('Invalid comparison ID')
    raw=(DATA/'transfer-plans'/(ident+'.json')).read_text()
    if hashlib.sha256(raw.encode()).hexdigest()!=ident:raise ValueError('План сравнения изменён')
    value=json.loads(raw)
    for key in ('transfer_draft','scratch_draft'):load(value[key])
    return value
