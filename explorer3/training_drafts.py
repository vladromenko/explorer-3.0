"""Immutable, reviewable sim training drafts. Preparation never queues a trainer."""
import hashlib
import json
from pathlib import Path
from .paths import DATA
from . import datasets
from .policy_catalog import validate_recipe
from .profiles import profile


def save_content(directory,value):
    raw=json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)
    ident=hashlib.sha256(raw.encode()).hexdigest();directory.mkdir(parents=True,exist_ok=True)
    path=directory/(ident+'.json')
    if not path.exists():path.write_text(raw)
    return ident


def checkpoint_hash(root):
    root=Path(root)
    if not (root/'config.json').is_file() or not (root/'model.safetensors').is_file():raise ValueError('Checkpoint неполный')
    value=hashlib.sha256()
    for path in sorted(p for p in root.iterdir() if p.is_file()):
        value.update(path.name.encode())
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def semantic_contract(value):
    if value is None:return None
    return {key:value.get(key) for key in ('robot','fps','task','camera','image_shape','channels','units','state_channels','state_units','environment_channels','sensor_schema')}


def merge_plans(old,new):
    """Keep prior hold-outs and prevent old training scenes leaking into new test."""
    if old['domain']!='sim' or new['domain']!='sim' or old['profile']!=new['profile'] or old['sim_model_hash']!=new['sim_model_hash'] or semantic_contract(old.get('profile_contract'))!=semantic_contract(new.get('profile_contract')):
        raise ValueError('Дообучение требует тот же профиль и версию модели; sim→real — отдельный протокол')
    anchored={e['id']:role for role,entries in old['splits'].items() for e in entries}
    roles={e['id']:role for role,entries in new['splits'].items() for e in entries}
    entries={e['id']:e for plan in (old,new) for group in plan['splits'].values() for e in group}
    for e in (e for g in old['splits'].values() for e in g):
        if e['sha256']!=entries[e['id']]['sha256']:raise ValueError('Исходный показ изменился')
    splits={role:[] for role in ('train','validation','test')}
    for group in datasets.connected_groups(list(entries.values())):
        fixed={anchored[e['id']] for e in group if e['id'] in anchored}
        if len(fixed)>1:raise ValueError('Новые данные связывают прежние train/hold-out; такая смесь недопустима')
        chosen=fixed or {roles[e['id']] for e in group}
        if len(chosen)!=1:raise ValueError('Связанные сцены оказались в разных частях плана')
        splits[chosen.pop()].extend(group)
    if any(not entries for entries in splits.values()):raise ValueError('Нужны независимые train/validation/test после объединения')
    plan={**new,'smoke':False,'splits':splits,'statistics':'train_only',
          'split_rule':'prior connected groups retain their roles; no prior train scene becomes hold-out'}
    return plan


def validate_budget(steps,batch,seed,device):
    if type(steps) is not int or not 50<=steps<=1_000_000:raise ValueError('Число обновлений: от 50 до 1 000 000')
    if type(batch) is not int or not 1<=batch<=32:raise ValueError('Размер пакета: от 1 до 32')
    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Seed должен быть целым неотрицательным числом')
    if device not in {'cpu','cuda'}:raise ValueError('Выберите CPU или GPU')


def prepare(name,policy,episode_ids,*,steps,batch,seed,device,source=None):
    validate_budget(steps,batch,seed,device)
    if profile(name).get('domain')!='sim':raise ValueError('Этот конструктор предназначен для симуляции')
    validate_recipe(policy,name)
    plan_id,plan=datasets.freeze(name,False,seed=seed,selected_ids=episode_ids,min_sim_episodes=3)
    # A draft is a new run; resume is a separate existing job action.
    initial=None;prior=None;source_plan_id=None;fingerprint=None
    if source:
        source_config=json.loads(source['config'])
        if source['status']!='completed' or source['profile']!=name or source['policy']!=policy or not source['checkpoint']:
            raise ValueError('Нужны завершённые веса той же модели и задачи')
        if source_config.get('kind','train')!='train':raise ValueError('Нужен исходный результат обучения')
        source_plan_id=source_config.get('plan_id')
        if not source_plan_id:raise ValueError('Нет неизменяемого исходного плана данных')
        prior=datasets.load_plan(source_plan_id)
        initial=source['checkpoint'];fingerprint=checkpoint_hash(initial)
        if json.loads((Path(initial)/'config.json').read_text()).get('type')!=policy:
            raise ValueError('Тип checkpoint не совпадает с выбранной моделью')
        plan=merge_plans(prior,plan)
        plan['parent_plan_id']=source_plan_id
        plan_id=save_content(DATA/'plans',plan)
    old_ids={e['id'] for e in prior['splits']['train']} if prior else set()
    frames={role:sum(e['frames'] for e in entries) for role,entries in plan['splits'].items()}
    old_frames=sum(e['frames'] for e in plan['splits']['train'] if e['id'] in old_ids)
    draft={'version':1,'kind':'train','profile':name,'policy':policy,'smoke':False,'plan_id':plan_id,
        'steps':steps,'batch_size':batch,'seed':seed,'device':device,
        'initial_checkpoint':initial,'initial_checkpoint_sha256':fingerprint,'source_job':source['id'] if source else None,
        'source_plan_id':source_plan_id,'mode':'fine_tune_new_optimizer' if source else 'new_training',
        'normalization':'new train partition only; official LeRobot non-resume statistics override',
        'sampling':'uniform merged train frames; no oversampling or hidden class weights',
        'mixture':{'old_train_frames':old_frames,'new_train_frames':frames['train']-old_frames},
        'split_counts':{role:len(entries) for role,entries in plan['splits'].items()},'frames':frames,
        'test_rule':'prior hold-outs retained; test not used for training or validation selection',
        'training_started':False,'preparation_only':True,
        'limitation':'Three independent groups only make a valid split; they do not guarantee an adequate learning dataset'}
    return save_content(DATA/'training-drafts',draft),draft


def load(ident):
    if len(ident)!=64 or any(c not in '0123456789abcdef' for c in ident):raise ValueError('Invalid draft ID')
    raw=(DATA/'training-drafts'/(ident+'.json')).read_text()
    if hashlib.sha256(raw.encode()).hexdigest()!=ident:raise ValueError('План запуска изменился')
    draft=json.loads(raw);validate_budget(draft['steps'],draft['batch_size'],draft['seed'],draft['device'])
    datasets.load_plan(draft['plan_id'])
    if draft.get('source_plan_id'):datasets.load_plan(draft['source_plan_id'])
    if draft['initial_checkpoint'] and checkpoint_hash(draft['initial_checkpoint'])!=draft['initial_checkpoint_sha256']:
        raise ValueError('Исходные веса изменились после подготовки')
    return draft
