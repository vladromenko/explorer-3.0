"""Matched sim branches -> immutable common-estimator plan. Never fits a model.

This is a simulation estimand for bounded scripted assistance, not human causal
labels. Repeated branches and repeated scene seeds remain one split group.
"""
import hashlib
import json
import math
from pathlib import Path
from .paths import DATA
from .research import FEATURES,features,matrix


def checksum(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def sample(profile,state,action,seconds):
    """Only controller-visible state; never task phase, object pose or dynamics."""
    timestamp=round(seconds*1e9)+1
    state=list(map(float,state));action=list(map(float,action))
    kind='previous_command_estimate_no_wheel_feedback'
    if 'pick_place' in profile:kind='servo_position_readback'
    elif profile=='rosmaster_dock_sim':state=state[:len(action)];kind='wheel_encoder_velocity'
    return {'observation.state':state,'action':action,'observation_timestamp_ns':timestamp,
        'action_send_start_ns':timestamp,'action_send_end_ns':timestamp,'action_available_ns':timestamp,
        'action_time_kind':'simulation_control_submission','state_valid':True,'measured_response_kind':kind,
        'camera':{'timestamp_ns':timestamp},'domain':'sim','response_source':'simulation_observable_proxy'}


def observation(history,seconds,scales,delay):
    if not history:
        return {name:(float(delay) if name=='assigned_delay_s' else None) for name in FEATURES}
    return features(history,round(seconds*1e9)+1,scales,delay_s=delay)


def candidate(report):
    if report.get('domain')!='sim' or not report.get('replay_verified'):
        raise ValueError('Нужны проверенные симуляционные ветви')
    if report.get('policy',{}).get('kind')!='frozen_policy':
        raise ValueError('Диагностическое удержание не обучает оценщик помощи')
    context=report.get('decision_context')
    if not context or set(context.get('features',{}))!=set(FEATURES):
        raise ValueError('Нет сохранённых признаков до решения; повторите сравнение')
    if context.get('source')!='controller_visible_history' or context.get('truth_features') is not False:
        raise ValueError('Нет проверенного источника наблюдаемых признаков')
    for value in context['features'].values():
        if value is not None and (not isinstance(value,(int,float)) or not math.isfinite(value)):
            raise ValueError('Недопустимые признаки решения')
    outcomes={r['branch']:r for r in report['outcomes'] if r['repeat']==0}
    if set(outcomes)!={'continue','hold_only','correction'}:raise ValueError('Нужны три сопоставимые ветви')
    if any(r['outcome'] not in {'success','failure'} for r in outcomes.values()):
        raise ValueError('Незавершённые ветви не являются неуспехом')
    cfg=report['budget'];group=checksum({'profile':cfg['profile'],'seed':cfg['seed'],'model':report['model']['xml_sha256']})
    if context['features']['assigned_delay_s']!=report['actual_windows']['delay_s']:
        raise ValueError('Признаки относятся к другой задержке помощи')
    return {'lineage':report['lineage'],'scene_group':group,'profile':cfg['profile'],
        'policy_id':report['policy']['policy_id'],'model':report['model']['xml_sha256'],
        'features':context['features'],'outcomes':{k:int(v['outcome']=='success') for k,v in outcomes.items()},
        'protocol':{'correction_s':report['actual_windows']['correction_s'],
            'horizon_s':report['actual_windows']['horizon_s'],'reset_rule':report['reset_rule']},
        'delay_s':report['actual_windows']['delay_s'],'domain':'sim','supervision':'scripted_expert'}


def prepare(reports,*,include_physical=True,seed=1000):
    """At least five independent scenes, transitive lineage/scene grouping."""
    sources=[];rows=[]
    for source in reports:
        path=Path(source).resolve();raw=path.read_bytes();report=json.loads(raw)
        rows.append(candidate(report));sources.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()})
    if not rows:raise ValueError('Нет завершённых сопоставимых сравнений')
    if len({r['lineage'] for r in rows})!=len(rows):raise ValueError('Один lineage выбран несколько раз')
    if len({checksum(r['protocol']) for r in rows})!=1:raise ValueError('Бюджеты и правила коррекции должны совпадать')
    # Same scene seeds are grouped across delays/checkpoints and cannot leak.
    groups=sorted({r['scene_group'] for r in rows},key=lambda x:checksum([seed,x]))
    if len(groups)<5:raise ValueError('Нужны минимум пять независимых сцен для train/validation/test')
    validation=max(1,len(groups)//5);test=max(1,len(groups)//5)
    roles={g:('validation' if i<validation else 'test' if i<validation+test else 'train') for i,g in enumerate(groups)}
    for row in rows:row['split']=roles[row['scene_group']]
    train=[r for r in rows if r['split']=='train']
    if any(len({r['outcomes'][branch] for r in train})!=2 for branch in ('continue','correction')):
        raise ValueError('Для обеих ветвей в train нужны успехи и неуспехи')
    _,names=matrix(rows,include_physical)
    plan={'schema':'branch_gain_plan_v1','rows':rows,'sources':sources,'feature_names':names,
        'include_physical':bool(include_physical),'seed':int(seed),'max_iter':1000,
        'training_started':False,'domain':'sim','supervision':'scripted_expert',
        'estimand':'bounded_scripted_correction_minus_continue','protocol':rows[0]['protocol'],
        'limitations':['not human intervention effect','uncalibrated nominal physics','small plans are pilots, not research evidence']}
    ident=checksum(plan);root=DATA/'gain-plans';root.mkdir(parents=True,exist_ok=True)
    path=root/(ident+'.json');value=json.dumps(plan,indent=2,allow_nan=False)
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Immutable plan collision')
    if not path.exists():path.write_text(value)
    return ident,plan


def load(ident):
    if len(ident)!=64 or any(c not in '0123456789abcdef' for c in ident):raise ValueError('Invalid gain plan')
    plan=json.loads((DATA/'gain-plans'/(ident+'.json')).read_text())
    if checksum(plan)!=ident:raise ValueError('Gain plan changed')
    for source in plan['sources']:
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('Branch report changed after preparation')
    return plan
