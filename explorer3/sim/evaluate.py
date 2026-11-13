"""Fresh-process LeRobot policy evaluation on actual MuJoCo dynamics and unseen seeds."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .factory import make
from explorer3.profiles import profile as profile_config


from explorer3.act_client import ACTClient


def evaluate(checkpoint,output,seeds,profile='so101_pick_place_sim',expected_model=None):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    env=make(profile,images=True)
    if expected_model is not None and env.manifest['xml_sha256']!=expected_model:
        env.close()
        raise ValueError('Физика сцены изменилась после записи обучающего набора. Нужны новые показы; перенос между моделями требует отдельного эксперимента')
    cfg=profile_config(profile)
    client=None
    trials=[];latencies=[]
    try:
        client=ACTClient(checkpoint,output.with_suffix('.inference.log'))
        for seed in seeds:
            obs,_=env.reset(seed=seed);client.request('reset')
            done=False
            while not done:
                started=time.monotonic();action=client.action(obs,cfg);latencies.append(time.monotonic()-started)
                obs,reward,terminated,truncated,info=env.step(action)
                done=terminated or truncated
            trials.append({'seed':seed,'outcome':info['outcome'],'reason':info['termination_reason'],
                           'steps':env.steps,'sim_seconds':env.elapsed})
            output.write_text(json.dumps({'status':'running','trials':trials},indent=2))
    finally:
        if client is not None:client.close()
        env.close()
    result={'status':'completed','domain':'sim','policy':client.ready['policy_type'],'checkpoint':str(checkpoint),
            'checkpoint_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
            'model':env.manifest,'new_process_reload':True,'privileged_input':False,'trials':trials,
            'success_fraction':sum(t['outcome']=='success' for t in trials)/len(trials),
            'sensor_to_action_seconds_p50_p95_max':np.percentile(latencies,[50,95,100]).tolist(),
            'physical_skill_verified':False}
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--start',type=int,default=10000);parser.add_argument('--count',type=int,default=3)
    parser.add_argument('--profile',default='so101_pick_place_sim')
    parser.add_argument('--expected-model-sha256')
    args=parser.parse_args();evaluate(args.checkpoint,args.output,range(args.start,args.start+args.count),args.profile,args.expected_model_sha256)
