"""Prepare real ACT support memory from a frozen plan; no optimization steps."""
import hashlib
import json
from pathlib import Path
import numpy as np
from . import catalog
from .act_client import ACTClient
from .autointervene import Memory,Settings,calibrate
from .datasets import load_plan
from .paths import DATA
from .profiles import profile


def source_plan(job):
    config=json.loads(job['config'])
    if job['status']!='completed' or job['policy']!='act' or not job['checkpoint'] or config.get('kind','train')!='train':
        raise ValueError('Нужен завершённый ACT с замороженным набором')
    if not config.get('plan_id'):raise ValueError('Отсутствует frozen plan обучения')
    plan=load_plan(config['plan_id'])
    if plan['smoke'] or not plan['splits']['validation']:
        raise ValueError('После smoke нет отложенных данных для калибровки; нужен основной запуск с train/validation/test')
    if plan['profile']!=job['profile']:raise ValueError('Profile/plan mismatch')
    return config,plan


def prepare(job_id,output):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    found=[j for j in catalog.rows('jobs') if j['id']==job_id]
    if len(found)!=1:raise ValueError('ACT job not found')
    job=found[0];config,plan=source_plan(job);settings=Settings()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cfg=profile(job['profile']);client=ACTClient(Path(job['checkpoint']),output/'inference.log')
    cache=DATA/'cache/act-support';cache.mkdir(exist_ok=True)
    try:
        policy_id=client.ready['policy_id']
        if client.ready['chunk_size']<settings.prefix:raise ValueError('ACT chunk shorter than support prefix')
        def extract(entry,role):
            identity={'dataset_sha256':entry['sha256'],'policy_id':policy_id,'profile':job['profile'],
                      'role':role,'extractor':'frozen-act-layer4-mean-v1','prefix':settings.prefix}
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            archive=cache/(key+'.npz')
            if archive.exists():
                with np.load(archive,allow_pickle=False) as arrays:features=arrays['features'];actions=arrays['actions'];proposals=arrays['proposals']
            else:
                ds=LeRobotDataset('local/act_support',root=Path(entry['dataset']),video_backend='pyav')
                features=[];actions=[];proposals=[];client.request('reset')
                for index in range(len(ds)):
                    row=ds[index];actions.append(row['action'].numpy())
                    obs={'state':row['observation.state'].numpy(),
                         'rgb':np.clip(row['observation.images.'+cfg['camera']].numpy().transpose(1,2,0)*255,0,255).round().astype(np.uint8)}
                    result=client.observation_request('propose' if role=='calibration' else 'embed',obs,cfg)
                    if result['policy_id']!=policy_id:raise ValueError('Inference checkpoint changed')
                    features.append(result['features'])
                    if role=='calibration':proposals.append(result['chunk'][:settings.prefix])
                    if index%50==0:(output/'progress.json').write_text(json.dumps({'episode':entry['id'],'role':role,'frame':index,'total':len(ds),'training':False}))
                features=np.asarray(features,dtype=np.float32);actions=np.asarray(actions,dtype=np.float32)
                proposals=np.asarray(proposals,dtype=np.float32)
                temporary=archive.with_suffix('.tmp')
                with temporary.open('wb') as stream:np.savez_compressed(stream,features=features,actions=actions,proposals=proposals)
                temporary.replace(archive)
            value={**{k:entry[k] for k in ('id','session','scene','outcome')},'role':role,'profile':job['profile'],
                'policy_id':policy_id,'features':features,'actions':actions,'data_kind':plan['domain']+'_recorded_execution'}
            if role=='calibration':
                valid=len(actions)-settings.prefix+1
                if valid<1:raise ValueError('Calibration trajectory too short')
                value.update(features=features[:valid],proposals=proposals[:valid],proposal_kind='policy_prediction')
            return value
        references=[extract(entry,'memory') for entry in plan['splits']['train']]
        memory=Memory(references,profile=job['profile'],policy_id=policy_id,settings=settings)
        held=[extract(entry,'calibration') for entry in plan['splits']['validation']]
        thresholds=calibrate(memory,held)
        memory.save(output/'memory.npz')
        (output/'thresholds.json').write_text(json.dumps(thresholds,indent=2))
        result={'method':'AutoIntervene-adapted','source_job':job_id,'plan_id':config['plan_id'],
                'policy_id':policy_id,'memory_id':memory.ident,'profile':job['profile'],
                'memory':str(output/'memory.npz'),'thresholds':str(output/'thresholds.json'),
                'training_started':False,'test_used':False,'physical_skill_verified':False,
                'limitation':'Calibrated support monitor, not a causal human-help value estimator'}
        (output/'result.json').write_text(json.dumps(result,indent=2));return result
    finally:client.close()
