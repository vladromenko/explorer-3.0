"""R4 preparation: bounded noise from measured validation error, never training.

This is DART-inspired data collection, not DART covariance optimization. Original
executed actions remain in research.jsonl; the separate expert target is explicit.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from . import catalog,datasets
from .paths import DATA,ROOT
from .profiles import profile
from .training_drafts import checkpoint_hash,save_content

METHOD='noise-augmented demonstrations (DART-inspired)'
SOURCES=('explorer3/noise_demonstrations.py','explorer3/sim/capture.py','explorer3/sim/expert.py',
    'explorer3/sim/mobile.py','explorer3/recording.py','explorer3/validation.py','explorer3/sim/import_episode.py')


def prepare(source_job,seeds,*,horizon_s=30.):
    row=next((j for j in catalog.rows('jobs') if j['id']==source_job),None)
    if row is None or row['status']!='completed' or row['policy']!='act' or not row['checkpoint']:
        raise ValueError('Нужен завершённый ACT и измеренная ошибка на validation')
    cfg=json.loads(row['config']);contract=profile(row['profile'])
    if contract.get('domain')!='sim' or cfg.get('smoke') or cfg.get('kind','train')!='train':
        raise ValueError('Только отдельный симуляционный ACT с независимой validation')
    split=datasets.load_plan(cfg['plan_id'])
    path=DATA/'jobs'/source_job/'evaluation.json';report=json.loads(path.read_text())
    if report.get('role')!='held_out_validation' or Path(report['checkpoint']).resolve()!=Path(row['checkpoint']).resolve():
        raise ValueError('Ошибка должна относиться к validation именно этого checkpoint')
    model_hash=checkpoint_hash(row['checkpoint'])
    if report.get('checkpoint_bundle_sha256')!=model_hash:raise ValueError('Оценка не фиксирует текущие веса; нужна новая validation после разрешения запуска')
    if not 1<=len(seeds)<=20 or len(set(seeds))!=len(seeds) or any(type(s) is not int or not 0<=s<2**31 for s in seeds):
        raise ValueError('Нужно от 1 до 20 разных расстановок')
    if not isinstance(horizon_s,(int,float)) or not np.isfinite(horizon_s) or not 1<=horizon_s<=60:
        raise ValueError('Горизонт: от 1 до 60 секунд симуляции')
    old_seeds=set()
    for entries in split['splits'].values():
        for entry in entries:
            provenance=json.loads((Path(entry['dataset'])/'provenance.json').read_text())
            old_seeds.add(provenance['hardware']['seed'])
    if set(seeds)&old_seeds:raise ValueError('Новые показы не должны повторять исходные train/validation/test seeds')
    expected={str(Path(e['dataset']).resolve()) for e in split['splits']['validation']}
    if {str(Path(d['dataset']).resolve()) for d in report['datasets']}!=expected:
        raise ValueError('Список validation не совпадает с замороженным планом')
    counts=np.asarray([d['samples'] for d in report['datasets']],float)
    errors=np.asarray([d['rmse_by_channel'] for d in report['datasets']],float)
    if errors.shape!=(len(counts),len(contract['channels'])) or not np.isfinite(errors).all() or np.any(errors<0) or not np.isfinite(counts).all() or np.any(counts<1):
        raise ValueError('Некорректная поканальная ошибка ученика')
    if any(d['channels']!=contract['channels'] for d in report['datasets']):raise ValueError('Порядок каналов изменился')
    rmse=np.sqrt(np.average(errors**2,axis=0,weights=counts)).tolist()
    value={'schema':'noise_demonstrations_v1','method':METHOD,'domain':'sim','profile':row['profile'],
        'source_job':source_job,'source_plan_id':cfg['plan_id'],'checkpoint_sha256':model_hash,
        'evaluation_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rmse_by_channel':rmse,
        'channels':contract['channels'],'units':contract['units'],'model_sha256':split['sim_model_hash'],
        'seeds':seeds,'horizon_s':float(horizon_s),'noise_rule':'independent Gaussian; sigma=min(validation RMSE, 0.05 action range); clip to action limits',
        'label_rule':'expert intention before noise at visited observation; executed command preserved separately',
        'arms':['clean','noisy'],'paired_budget':'same seeds and maximum steps; terminated episodes remain terminated',
        'training_comparison':'same ACT recipe and frame budget; new independent test seeds; no fit in this job',
        'training_started':False,'physical_access':False,'full_dart_covariance_optimization':False}
    value['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}
    return save_content(DATA/'noise-plans',value),value


def load(ident):
    if len(ident)!=64 or any(c not in '0123456789abcdef' for c in ident):raise ValueError('Invalid noise plan')
    raw=(DATA/'noise-plans'/(ident+'.json')).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ident:raise ValueError('Noise plan changed')
    value=json.loads(raw);source=next(j for j in catalog.rows('jobs') if j['id']==value['source_job'])
    for p,sha in value['source_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=sha:raise ValueError('Collection implementation changed; prepare a new plan')
    if checkpoint_hash(source['checkpoint'])!=value['checkpoint_sha256']:raise ValueError('Student checkpoint changed')
    if hashlib.sha256((DATA/'jobs'/source['id']/'evaluation.json').read_bytes()).hexdigest()!=value['evaluation_sha256']:
        raise ValueError('Measured validation error changed')
    datasets.load_plan(value['source_plan_id'])
    return value


def perturb(intent,rmse,low,high,rng,*,noisy):
    intent=np.asarray(intent,float);low=np.asarray(low,float);high=np.asarray(high,float);rmse=np.asarray(rmse,float)
    if intent.shape!=low.shape or high.shape!=low.shape or rmse.shape!=low.shape or not all(np.isfinite(v).all() for v in (intent,rmse,low,high)) or np.any(high<=low) or np.any(rmse<0):
        raise ValueError('Invalid noise action contract')
    if np.any(intent<low) or np.any(intent>high):raise ValueError('Expert target exceeds action limits')
    sigma=np.minimum(rmse,.05*(high-low))
    noise=rng.normal(size=intent.shape)*sigma if noisy else np.zeros_like(intent)
    return np.clip(intent+noise,low,high),sigma


def training_target(sample,*,domain,label):
    if label is None:return sample['action']
    if domain!='sim' or label!='expert_intent_before_noise' or sample.get('command_source')!='scripted_expert':
        raise ValueError('Expert target annotation is only admitted in the controlled simulation profile')
    target=np.asarray(sample['expert_action'],float);executed=np.asarray(sample['action'],float)
    if target.shape!=executed.shape or not np.isfinite(target).all() or not np.isfinite(executed).all():raise ValueError('Invalid separate expert target')
    return target.tolist()


def validate_review(root,review):
    from .branch_corrections import source_hash
    meta=json.loads((root/'manifest.json').read_text())
    if review.get('decision')!='accept_noise_demonstration' or review.get('source_sha256')!=source_hash(root):raise ValueError('Просмотр относится к другим данным')
    if meta['domain']!='sim' or meta['outcome']!='success' or meta.get('noise_plan')!=review.get('plan_id') or meta.get('training_label')!='expert_intent_before_noise':
        raise ValueError('Нельзя принять этот восстановительный показ')
    for line in (root/'samples.jsonl').read_text().splitlines():training_target(json.loads(line),domain='sim',label=meta['training_label'])


def run(config,work):
    """Executed only by an explicitly queued simulation job, never at import."""
    import math
    from .sim.factory import make,EXPERTS
    from .sim.capture import SimCapture
    plan=load(config['plan_id']);results=[]
    for seed in plan['seeds']:
        for arm in plan['arms']:
            env=make(plan['profile']);capture=None
            try:
                env.reset(seed=seed,options={'variation':'nominal'})
                if env.manifest['xml_sha256']!=plan['model_sha256']:raise ValueError('Current physics differs from the student dataset')
                expert=EXPERTS[plan['profile']](env);rng=np.random.default_rng(seed)
                capture=SimCapture(env,seed,'scripted_expert')
                capture.meta.update(noise_plan=config['plan_id'],arm=arm,method=METHOD,
                    training_label='expert_intent_before_noise',training_eligible=False,expert_privileged=True,
                    segment_role='noise_demonstration',session=config['plan_id']+'-'+str(seed),
                    lineage=config['plan_id']+'-'+str(seed))
                outcome='aborted';reason='bounded_horizon';response=[]
                for step in range(math.ceil(plan['horizon_s']/env.dt)):
                    intent=np.clip(np.asarray(expert.action(),float),env.action_space.low,env.action_space.high)
                    action,sigma=perturb(intent,plan['rmse_by_channel'],env.action_space.low,env.action_space.high,rng,noisy=arm=='noisy')
                    capture.add(env,action,annotations={'expert_action':intent.tolist(),'noise_sigma':sigma.tolist(),
                        'noise_delta':(action-intent).tolist(),'training_label':'expert_intent_before_noise'})
                    _,_,terminated,truncated,info=env.step(action)
                    response.append({'step':step,'state_after':env.observation()['state'].tolist(),
                        'applied_command':np.asarray(env.applied_command()).tolist(),'outcome':info['outcome']})
                    if terminated or truncated:
                        outcome=info['outcome'] if info['outcome'] in ('success','failure') else 'aborted';reason=info['termination_reason'];break
                root=capture.finish(outcome,reason)
                (root/'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in response))
                results.append({'seed':seed,'arm':arm,'spool':str(root),'frames':capture.count,'outcome':outcome,'review_required':True})
            finally:
                if capture and not capture.finished:capture.finish('aborted','execution_error')
                env.close()
    result={'method':METHOD,'plan_id':config['plan_id'],'domain':'sim','episodes':results,
        'weights_updated':False,'training_started':False,'scientific_effect_established':False}
    (work/'result.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    import sys
    run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
