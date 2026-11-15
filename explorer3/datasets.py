"""Immutable training selections; original recordings are never rewritten."""
import hashlib
import json
import random
from pathlib import Path
from . import catalog
from .paths import DATA
from .profiles import profile


def fingerprint(root):
    root = Path(root).resolve()
    if not root.is_relative_to(DATA / 'datasets'):
        raise ValueError('Dataset outside Explorer storage')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in {'.json', '.jsonl', '.parquet', '.mp4', '.npz'})
    digest = hashlib.sha256()
    for path in files:
        if not path.resolve().is_relative_to(root):raise ValueError('Dataset file points outside its recording')
        digest.update(str(path.relative_to(root)).encode())
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def eligible(name):
    profile(name)
    return [e for e in catalog.rows('episodes') if e['profile'] == name and not e['excluded']
            and e['category'] in {'demonstration', 'correction'} and e['outcome'] == 'success']


def connected_groups(episodes):
    # Connected components: sharing EITHER a session OR a scene prevents leakage,
    # including transitive connections between several recording sessions.
    groups = []
    for episode in sorted(episodes, key=lambda e: e['id']):
        linked = [g for g in groups if any(
            episode['session'] == e['session'] or episode['scene'] == e['scene'] for e in g)]
        merged = [episode] + [e for g in linked for e in g]
        groups = [g for g in groups if g not in linked] + [merged]
    return groups


def grouped_split(episodes, seed=1000):
    groups=connected_groups(episodes)
    if len(groups) < 3:
        raise ValueError('Нужны ≥3 независимые группы: общие сессии ИЛИ расстановки нельзя разделять между train/validation/test')
    random.Random(seed).shuffle(groups)
    hold = max(1, round(len(groups) * .1))
    return {key: [e for group in part for e in group] for key, part in
            [('test', groups[:hold]), ('validation', groups[hold:2*hold]), ('train', groups[2*hold:])]}


def simulation_model_hash(episode):
    provenance=json.loads((Path(episode['dataset'])/'provenance.json').read_text())
    value=provenance.get('hardware',{}).get('model',{}).get('xml_sha256')
    if not isinstance(value,str) or len(value)!=64:
        raise ValueError('У симуляционной записи отсутствует версия физической модели')
    return value


def select_simulation_revision(episodes, expected=None):
    """Never pool old proxy geometry and corrected contact dynamics implicitly."""
    versions={e['id']:simulation_model_hash(e) for e in episodes}
    selected=[e for e in episodes if expected is None or versions[e['id']]==expected]
    revisions={versions[e['id']] for e in selected}
    if len(revisions)>1:
        raise ValueError('Записи относятся к разным версиям физики. Выберите одну версию модели')
    return selected, next(iter(revisions),expected)


# real recording block -> the MuJoCo task that mirrors it (explorer3/sim/aligned.py)
SIM_FOR_REAL = {'so101_pick_place': 'so101_handover_sim', 'rosmaster_pick_place': 'rosmaster_floor_pick_sim'}


def cotraining_episodes(name, real_train_frames, ratio):
    """Sim-and-real co-training (Maddukuri et al., 2025, arXiv:2503.24361): demonstrations of the
    MuJoCo task that mirrors the recording block join the TRAIN split only, at most ratio x the real
    training frames, newest physics version only. Validation and test stay real."""
    from .training_drafts import semantic_contract
    sim_name = SIM_FOR_REAL.get(name)
    if sim_name is None:
        raise ValueError('Для этой задачи нет выровненной сцены MuJoCo')
    sims = sorted(eligible(sim_name), key=lambda e: -e['created'])
    if not sims:
        raise ValueError('Нет показов MuJoCo для совместного обучения: сначала соберите их')
    newest = simulation_model_hash(sims[0])
    sims = [e for e in sims if simulation_model_hash(e) == newest]
    def contract(p):
        c = semantic_contract(p) or {}
        c.pop('task', None)
        return c
    if contract(profile(sim_name)) != contract(profile(name)):
        raise ValueError('Каналы, камера или частота MuJoCo-задачи отличаются от реальной')
    chosen, frames = [], 0
    for e in sims:
        if frames + e['frames'] > ratio * real_train_frames:
            break
        sources = {json.loads(line)['command_source'] for line in (Path(e['dataset']) / 'research.jsonl').read_text().splitlines()}
        if not sources.issubset({'scripted_expert', 'human', 'ppo_expert'}):
            raise ValueError('Недопустимый источник команд в показе MuJoCo')
        chosen.append({**e, 'sha256': fingerprint(e['dataset']), 'domain': 'sim'})
        frames += e['frames']
    return chosen, {'sim_profile': sim_name, 'sim_model_hash': newest, 'sim_episodes': len(chosen), 'sim_frames': frames,
                    'real_train_frames': real_train_frames, 'ratio_limit': ratio, 'splits_with_sim': ['train'],
                    'reference': 'Sim-and-Real Co-Training, arXiv:2503.24361'}


def freeze(name, smoke, seed=1000, sim_model_hash=None, *, selected_ids=None, min_sim_episodes=None,min_real_episodes=None,cotrain_sim=None):
    domain=profile(name).get('domain','real')
    episodes = eligible(name)
    if selected_ids is not None:
        ids=set(selected_ids)
        if not ids or not ids.issubset({e['id'] for e in episodes}):raise ValueError('Выбранный показ отсутствует или не допущен')
        episodes=[e for e in episodes if e['id'] in ids]
    if min_sim_episodes is not None and (domain!='sim' or type(min_sim_episodes) is not int or min_sim_episodes<3):
        raise ValueError('At least 3 simulation episodes required')
    if min_real_episodes is not None and (domain!='real' or type(min_real_episodes) is not int or min_real_episodes<3 or smoke):
        raise ValueError('Real pilot requires at least 3 episodes and independent splits')
    if domain=='sim':episodes,sim_model_hash=select_simulation_revision(episodes,sim_model_hash)
    elif sim_model_hash is not None:raise ValueError('Версия симулятора не относится к физическим данным')
    minimum=min_real_episodes if min_real_episodes is not None else min_sim_episodes if min_sim_episodes is not None else 3 if smoke else 50
    if len(episodes) < minimum:
        raise ValueError(f'Нужно минимум {minimum} пригодных показов одной версии модели; для оценки нужны 3 независимые группы')
    for episode in episodes:
        root = Path(episode['dataset'])
        provenance = json.loads((root / 'provenance.json').read_text())
        if provenance.get('hardware',{}).get('pipeline_check') or provenance.get('hardware',{}).get('training_eligible') is False:
            raise ValueError('Диагностическая или не принятая запись не является обучающим показом')
        if provenance.get('domain','real')!=domain or provenance.get('synthetic') is not (domain!='real'):
            raise ValueError('Синтетические данные запрещены в физическом наборе')
        from .training_drafts import semantic_contract
        if semantic_contract(provenance.get('profile'))!=semantic_contract(profile(name)):
            raise ValueError('Камера, частота, единицы или каналы записи отличаются от текущего профиля: '+episode['id'])
        validation = DATA / 'validation' / f"episode-{episode['id']}.json"
        digest = fingerprint(root)
        checked = json.loads(validation.read_text()) if validation.exists() else {}
        if checked.get('readback') != 'PASS' or checked.get('dataset_sha256') != digest:
            from .validation import validate
            checked = validate(episode['id'],domain=domain)
            if checked['dataset_sha256'] != digest:
                raise ValueError('Dataset changed during validation')
        sources = set(json.loads(line)['command_source'] for line in (root / 'research.jsonl').read_text().splitlines())
        if not sources or not sources.issubset({'human'} if domain=='real' else {'human','scripted_expert','ppo_expert'}):
            raise ValueError('Behavior cloning принимает только отдельные сегменты human без границ переключения')
        episode['sha256'] = digest
    splits = {'train': episodes, 'validation': [], 'test': []} if smoke else grouped_split(episodes, seed)
    plan = {'version': 1, 'domain':domain, 'profile': name, 'profile_contract': profile(name), 'smoke': bool(smoke),
            'seed': seed, 'statistics': 'train_only', 'splits': splits,'sim_model_hash':sim_model_hash,
            'limitation': 'Smoke использует все показы; это не независимая оценка качества' if smoke else
                          'Split заморожен; test не используется для выбора модели'}
    if min_real_episodes is not None:plan.update(pilot=True,minimum_episodes=min_real_episodes)
    if cotrain_sim:
        if domain!='real' or smoke:raise ValueError('Совместное обучение — для основного обучения на настоящих показах')
        extra,info=cotraining_episodes(name,sum(e['frames'] for e in splits['train']),float(cotrain_sim))
        splits['train']=splits['train']+extra
        plan['cotraining']=info
    blob = json.dumps(plan, sort_keys=True, ensure_ascii=False, indent=2)
    ident = hashlib.sha256(blob.encode()).hexdigest()
    directory = DATA / 'plans'
    directory.mkdir(exist_ok=True)
    path = directory / f'{ident}.json'
    if not path.exists():
        path.write_text(blob)
    return ident, plan


def load_plan(ident, verify=True):
    if len(ident) != 64 or any(c not in '0123456789abcdef' for c in ident):
        raise ValueError('Invalid plan ID')
    blob = (DATA / 'plans' / f'{ident}.json').read_text()
    if hashlib.sha256(blob.encode()).hexdigest() != ident:
        raise ValueError('Frozen plan changed')
    plan = json.loads(blob)
    if verify:
        for entries in plan['splits'].values():
            for entry in entries:
                if fingerprint(entry['dataset']) != entry['sha256']:
                    raise ValueError('Frozen dataset changed: ' + entry['id'])
    return plan
