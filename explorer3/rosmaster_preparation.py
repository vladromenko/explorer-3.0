"""Local ROSMASTER readiness and immutable identification plans, no robot client."""
import json
from pathlib import Path
from . import catalog,datasets,transfer_preparation
from .paths import DATA
from .training_drafts import save_content
from .sim.rosmaster_sysid import series


def motion_sources(target):
    accepted=[];rejected=[]
    for e in catalog.rows('episodes'):
        if e['profile']!=target:continue
        reason=None
        try:
            root=Path(e['dataset']);meta=json.loads((root/'provenance.json').read_text())
            if e['excluded']:raise ValueError('Исключена оператором')
            if meta.get('domain','real')!='real' or meta.get('synthetic') is not False:raise ValueError('Не физическая запись')
            if meta.get('hardware',{}).get('pipeline_check'):raise ValueError('Диагностический клип')
            rows=[json.loads(line) for line in (root/'research.jsonl').read_text().splitlines()]
            values=series(rows,target)
            if values is None:raise ValueError('Недостаточно команд и измерений отклика')
            import numpy as np
            threshold=.005 if target=='rosmaster_dock' else .5
            if not np.any(np.ptp(values[1],axis=0)>threshold):raise ValueError('Нет различимого изменения команд')
            if not np.any(np.ptp(values[3],axis=0)>threshold):raise ValueError('Нет различимого измеренного движения')
            accepted.append({**e,'sha256':datasets.fingerprint(root)})
        except (OSError,ValueError,KeyError,TypeError) as exc:reason=str(exc)
        if reason:rejected.append({'id':e['id'],'reason':reason})
    return sorted(accepted,key=lambda e:e['created']),rejected


def report():
    result={'hardware_access':False,'training_started':False,'profiles':{}}
    for target in transfer_preparation.TARGETS:
        episodes=transfer_preparation.inventory(target);good=[e for e in episodes if e['eligible']]
        motion,rejected=motion_sources(target);sources=[]
        for j in catalog.rows('jobs'):
            if j['profile']!=target+'_sim' or j['status']!='completed' or not j['checkpoint'] or json.loads(j['config']).get('kind','train')!='train':continue
            try:
                transfer_preparation.check_source(j,target);reason=None
            except (ValueError,OSError,KeyError,TypeError) as exc:reason=str(exc)
            sources.append({'id':j['id'],'policy':j['policy'],'eligible':reason is None,'reason':reason})
        result['profiles'][target]={'episodes':episodes,'eligible_shows':len(good),
            'independent_groups':len(datasets.connected_groups(good)),
            'motion_episodes':motion,'motion_rejected':rejected,
            'motion_groups':len(datasets.connected_groups(motion)),'sources':sources}
    return result


def prepare_identification(target,selected_ids):
    entries,_=motion_sources(target);ids=set(selected_ids)
    if not ids or not ids.issubset({e['id'] for e in entries}):raise ValueError('Выберите пригодные записи измеренного движения')
    entries=[e for e in entries if e['id'] in ids]
    if len(datasets.connected_groups(entries))<2:raise ValueError('Нужны минимум две независимые группы для настройки и проверки')
    plan={'version':1,'kind':'rosmaster_identification','profile':target,'entries':entries,
        'holdout_fraction':.3,'physical_access':False,'apply_automatically':False,
        'split_rule':'most recent connected session/scene groups held out','fixed_grid':True}
    return save_content(DATA/'identification-plans',plan),plan


def load_identification(ident):
    import hashlib
    if len(ident)!=64 or any(c not in '0123456789abcdef' for c in ident):raise ValueError('Invalid identification plan')
    raw=(DATA/'identification-plans'/(ident+'.json')).read_text()
    if hashlib.sha256(raw.encode()).hexdigest()!=ident:raise ValueError('План идентификации изменён')
    plan=json.loads(raw)
    for entry in plan['entries']:
        if datasets.fingerprint(entry['dataset'])!=entry['sha256']:raise ValueError('Запись изменилась после подготовки')
    return plan
