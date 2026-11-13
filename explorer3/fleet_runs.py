"""Prepare bounded common-world inference jobs; no hardware or training imports."""
import json
import math
from pathlib import Path
from . import catalog
from .paths import DATA
from .datasets import load_plan,select_simulation_revision
from .profiles import profile
from .training_drafts import checkpoint_hash
from .vision_assets import digest

PROFILES={'so101':'so101_pick_place_sim','tank':'tank_dock_sim','rosmaster':'rosmaster_pick_place_sim'}


def selections():
    return [j for j in catalog.rows('jobs') if j['status']=='completed' and j['profile'] in PROFILES.values()
        and j['checkpoint'] and j['policy'] in {'act','smolvla','diffusion'} and json.loads(j['config']).get('kind','train')=='train']


def prepare(selected,*,horizon_s=2.,seed=1,software_probe=False,method='fleet_thrifty',delay_s=.1,help_s=1.,support=None,gain_job=None):
    if not isinstance(horizon_s,(int,float)) or not math.isfinite(horizon_s) or not .2<=horizon_s<=20:
        raise ValueError('Длительность проверки: 0,2–20 секунд симуляции')
    if method not in {'random','fleet_thrifty'}:raise ValueError('Unknown allocator')
    if not 0<=delay_s<=10 or not 0<help_s<=10:raise ValueError('Invalid assistance budget')
    if not selected or not set(selected).issubset(PROFILES):raise ValueError('Выберите роботов общей сцены')
    jobs={j['id']:j for j in catalog.rows('jobs')};specs={};support=support or {}
    for name,ident in selected.items():
        if ident=='fixture':specs[name]={'kind':'fixture'}
        else:
            source=jobs.get(ident)
            if source not in selections() or source['profile']!=PROFILES[name]:raise ValueError('Checkpoint не соответствует выбранной задаче')
            plan_id=json.loads(source['config']).get('plan_id')
            if not plan_id:raise ValueError('Нет зафиксированного плана данных')
            plan=load_plan(plan_id)
            if plan.get('domain')!='sim':raise ValueError('Нужен симуляционный checkpoint')
            _,model_hash=select_simulation_revision([e for group in plan['splits'].values() for e in group])
            if model_hash is None:raise ValueError('Нет версии физики обучающих данных')
            checkpoint=Path(source['checkpoint']);config=json.loads((checkpoint/'config.json').read_text())
            cfg=profile(PROFILES[name]);expected=2 if name=='tank' else 6
            if config['output_features']['action']['shape']!=[expected] or 'observation.images.'+cfg['camera'] not in config['input_features']:
                raise ValueError('Несовместимые каналы или камера политики')
            specs[name]={'kind':'checkpoint','source_job':ident,'checkpoint':str(checkpoint),'device':'cpu',
                'checkpoint_sha256':checkpoint_hash(checkpoint),'training_model_sha256':model_hash,
                'software_probe':bool(software_probe),'source_plan':plan_id,'smoke':bool(plan['smoke'])}
            if support.get(name):
                item=jobs[support[name]]
                if item['status']!='completed' or json.loads(item['config']).get('kind')!='support_calibration':raise ValueError('Нужна завершённая калибровка оценщика')
                result=json.loads((DATA/'jobs'/item['id']/'result.json').read_text())
                if result['source_job']!=ident:raise ValueError('Калибровка получена для другого checkpoint')
                paths=[Path(result['memory']),Path(result['memory']).with_suffix('.json'),Path(result['thresholds'])]
                specs[name].update(memory=result['memory'],thresholds=result['thresholds'],support_job=item['id'],
                    support_files={str(p):digest(p) for p in paths})
    gain=None
    if gain_job:
        job=jobs[gain_job]
        if job['status']!='completed' or json.loads(job['config']).get('kind')!='branch_gain_train':raise ValueError('Нужен готовый оценщик пользы')
        path=DATA/'jobs'/gain_job/'model.json';artifact=json.loads(path.read_text())
        if artifact['protocol']['correction_s']!=help_s:raise ValueError('Бюджет помощи не совпадает с оценщиком')
        gain={'path':str(path),'sha256':digest(path),'source_job':gain_job}
    return {'gain':gain,'kind':'fleet_execution','domain':'sim','specs':specs,'seed':int(seed),'horizon_s':float(horizon_s),
        'method':method,'delay_s':float(delay_s),'help_s':float(help_s),'device':'cpu','training_started':False,
        'transfer_protocol':'individual_to_shared_nominal; not physical deployment',
        'software_probe':bool(software_probe),'replay_steps':8}
