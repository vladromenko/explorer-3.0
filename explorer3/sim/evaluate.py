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


def evaluate(checkpoint,output,seeds,profile='so101_pick_place_sim',expected_model=None,device='cuda',verify_grasp=False,inference_seed=1000):
    if type(inference_seed) is not int or not 0 <= inference_seed < 2**32:
        raise ValueError('Inference seed must be an integer in [0, 2**32)')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if verify_grasp and not profile.startswith('rosmaster_floor_pick'):
        raise ValueError('Independent grasp verification requires a ROSMASTER floor task')
    from explorer3.training_drafts import checkpoint_hash
    bundle = checkpoint_hash(checkpoint) if verify_grasp else None
    env=make(profile,images=True)
    if expected_model is not None and env.manifest['xml_sha256']!=expected_model:
        env.close()
        raise ValueError('Физика сцены изменилась после записи обучающего набора. Нужны новые показы; перенос между моделями требует отдельного эксперимента')
    cfg=profile_config(profile)
    client=None
    trials=[];latencies=[]
    try:
        client=ACTClient(checkpoint,output.with_suffix('.inference.log'),device=device)
        for seed in seeds:
            env.physics_observer = None
            obs,_=env.reset(seed=seed)
            policy_seed = (inference_seed + int(seed)) % 2**32
            sampling = client.request('reset', seed=policy_seed)
            if sampling.get('seed') != policy_seed:
                raise RuntimeError('Policy worker did not confirm the declared sampling seed')
            verifier = None
            if verify_grasp:
                from .skill_transfer import TransferVerifier
                verifier = TransferVerifier(env)
                env.physics_observer = verifier
            # per-episode task text (knowledge-gap profiles carry the object's measured size)
            episode_cfg={**cfg,'task':getattr(env,'task_text',None) or cfg['task']}
            done=False
            while not done:
                started=time.monotonic();action=client.action(obs,episode_cfg);latencies.append(time.monotonic()-started)
                obs,reward,terminated,truncated,info=env.step(action)
                done=terminated or truncated
            trials.append({'seed':seed,'outcome':info['outcome'],'reason':info['termination_reason'],
                           'steps':env.steps,'sim_seconds':env.elapsed, 'policy_sampling':sampling,
                           'task':episode_cfg['task'], 'object':dict(env.episode_parameters.get('object',{}))})
            if verifier is not None:
                trials[-1].update(grasp_verification=verifier.report(), object=dict(env.episode_parameters['object']),
                                  donor_online=False, geometric_correction=False)
            output.write_text(json.dumps({'status':'running','trials':trials},indent=2))
    finally:
        if client is not None:client.close()
        env.close()
    if verify_grasp and checkpoint_hash(checkpoint) != bundle:
        raise ValueError('Checkpoint changed during independent evaluation')
    result={'status':'completed','domain':'sim','policy':client.ready['policy_type'],'checkpoint':str(checkpoint),
            'checkpoint_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
            'model':env.manifest,'new_process_reload':True,'privileged_input':False,'trials':trials,
            'success_fraction':sum(t['outcome']=='success' for t in trials)/len(trials),
            'sensor_to_action_seconds_p50_p95_max':np.percentile(latencies,[50,95,100]).tolist(),
            'physical_skill_verified':False,
            'sampling':{'method':'torch.manual_seed per scene', 'base_seed':inference_seed,
                        'scene_seed_rule':'(base_seed + scene_seed) mod 2**32',
                        'limitation':'Fixed stochastic inputs; bitwise equality across devices/libraries is not promised'}}
    if verify_grasp:
        result.update(checkpoint_bundle_sha256=bundle, independent_grasp_verification=True,
                      verifier_sha256=hashlib.sha256(Path(__file__).with_name('skill_transfer.py').read_bytes()).hexdigest())
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--start',type=int,default=10000);parser.add_argument('--count',type=int,default=3)
    parser.add_argument('--profile',default='so101_pick_place_sim')
    parser.add_argument('--expected-model-sha256')
    parser.add_argument('--verify-grasp',action='store_true')
    parser.add_argument('--inference-seed',type=int,default=1000)
    args=parser.parse_args();evaluate(args.checkpoint,args.output,range(args.start,args.start+args.count),args.profile,args.expected_model_sha256,verify_grasp=args.verify_grasp,inference_seed=args.inference_seed)
