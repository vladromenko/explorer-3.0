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
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in {'.json', '.jsonl', '.parquet', '.mp4'})
    digest = hashlib.sha256()
    for path in files:
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


def freeze(name, smoke, seed=1000, sim_model_hash=None, *, selected_ids=None, min_sim_episodes=None):
    domain=profile(name).get('domain','real')
    episodes = eligible(name)
    if selected_ids is not None:
        if domain!='sim':raise ValueError('Explicit simulation selection only')
        ids=set(selected_ids)
        if not ids or not ids.issubset({e['id'] for e in episodes}):raise ValueError('Выбранный показ отсутствует или не допущен')
        episodes=[e for e in episodes if e['id'] in ids]
    if min_sim_episodes is not None and (domain!='sim' or type(min_sim_episodes) is not int or min_sim_episodes<3):
        raise ValueError('At least 3 simulation episodes required')
    if domain=='sim':episodes,sim_model_hash=select_simulation_revision(episodes,sim_model_hash)
    elif sim_model_hash is not None:raise ValueError('Версия симулятора не относится к физическим данным')
    minimum=min_sim_episodes if min_sim_episodes is not None else 3 if smoke else 50
    if len(episodes) < minimum:
        raise ValueError(f'Нужно минимум {minimum} пригодных показов одной версии модели; для оценки нужны 3 независимые группы')
    for episode in episodes:
        root = Path(episode['dataset'])
        provenance = json.loads((root / 'provenance.json').read_text())
        if provenance.get('domain','real')!=domain or provenance.get('synthetic') is not (domain!='real'):
            raise ValueError('Синтетические данные запрещены в физическом наборе')
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
