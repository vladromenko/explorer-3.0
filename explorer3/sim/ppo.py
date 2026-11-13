"""Standard SB3 PPO on native physics; state policy is simulation-only."""
import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from explorer3.paths import DATA,ROOT,offline_environment
from .environment import StateTeacher
from .factory import make,ENVIRONMENTS
from . import checkpoints


class SaveProgress(BaseCallback):
    def __init__(self,work,total):
        super().__init__();self.work=work;self.total=total;self.cancelled=False;self.started=time.monotonic()

    def _on_step(self):
        if self.num_timesteps%256==0 or self.cancelled:
            (self.work/'progress.json').write_text(json.dumps({'steps':self.num_timesteps,'budget':self.total,
                'elapsed_s':time.monotonic()-self.started,'device':'cpu','domain':'sim'}))
        if self.num_timesteps%1024==0 or self.cancelled:
            checkpoints.publish(self.model,self.work)
        return not self.cancelled


def evaluate(checkpoint,output,profile='so101_pick_place_sim',expected_model=None):
    model=PPO.load(checkpoint,device='cpu')
    env=StateTeacher(make(profile))
    if expected_model and env.unwrapped.manifest['xml_sha256']!=expected_model:
        env.close();raise ValueError('PPO evaluation model differs from training')
    normalizer=VecNormalize.load(checkpoints.verify(checkpoint),DummyVecEnv([lambda:env]))
    normalizer.training=False;normalizer.norm_reward=False
    trials=[]
    try:
        for seed in range(10000,10020):
            obs,_=env.reset(seed=seed);done=False
            while not done:
                action,_=model.predict(normalizer.normalize_obs(obs),deterministic=True)
                obs,_,terminated,truncated,info=env.step(action);done=terminated or truncated
            trials.append({'seed':seed,'outcome':info['outcome'],'reason':info['termination_reason']})
    finally:
        normalizer.close()
    result={'domain':'sim','policy':'SB3 PPO state teacher','privileged_state':True,'hardware_deployable':False,
            'model':env.unwrapped.manifest,'observation_normalization':'frozen training VecNormalize',
            'new_process_reload':True,'trials':trials,'success_fraction':sum(t['outcome']=='success' for t in trials)/len(trials)}
    Path(output).write_text(json.dumps(result,indent=2));return result


def train(path):
    cfg=json.loads(path.read_text());work=DATA/'jobs'/path.stem;work.mkdir(parents=True,exist_ok=True)
    if cfg['profile'] not in ENVIRONMENTS:raise ValueError('Unsupported simulator profile')
    budget=int(cfg['steps'])
    if not 256<=budget<=1000000:raise ValueError('Invalid finite training budget')
    profile=cfg['profile']
    raw=Monitor(StateTeacher(make(profile,variation=cfg.get('variation','nominal'))),filename=str(work/'episodes.csv'))
    manifest={**raw.unwrapped.manifest,'teacher_state_version':2,'normalization':'VecNormalize-v1','variation':cfg.get('variation','nominal')}
    env=DummyVecEnv([lambda:raw])
    if cfg.get('expected_model_hash') and cfg['expected_model_hash']!=manifest['xml_sha256']:
        env.close();raise ValueError('Scene changed after the training request; create a new explicit request')
    manifest_path=work/'model.json'
    if cfg.get('resume_checkpoint'):
        if not manifest_path.exists() or json.loads(manifest_path.read_text())!=manifest:
            env.close();raise ValueError('Cannot resume PPO across unknown or changed physics')
        env=VecNormalize.load(checkpoints.verify(cfg['resume_checkpoint']),env)
        model=PPO.load(cfg['resume_checkpoint'],env=env,device='cpu')
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2))
        env=VecNormalize(env,norm_obs=True,norm_reward=True,clip_obs=10.,gamma=.99)
        model=PPO('MlpPolicy',env,device='cpu',seed=cfg['seed'],n_steps=256,batch_size=64,n_epochs=4,
                  policy_kwargs={'net_arch':[128,128]},verbose=1)
    callback=SaveProgress(work,budget)
    signal.signal(signal.SIGTERM,lambda *_:setattr(callback,'cancelled',True))
    remaining=max(0,budget-model.num_timesteps)
    try:
        if remaining:
            model.learn(remaining,callback=callback,reset_num_timesteps=False)
        checkpoint=checkpoints.publish(model,work)
    finally:env.close()
    if callback.cancelled:return
    subprocess.run([sys.executable,'-m','explorer3.sim.ppo','--evaluate',str(checkpoint),
                    '--output',str(work/'evaluation.json'),'--profile',profile,'--expected-model-sha256',manifest['xml_sha256']],check=True,env=offline_environment())
    (work/'result.json').write_text(json.dumps({'checkpoint':str(checkpoint),'domain':'sim',
        'steps':model.num_timesteps,'budget':budget,'physical_skill_verified':False,'model':manifest,
        'evaluation':json.loads((work/'evaluation.json').read_text())},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path);parser.add_argument('--evaluate');parser.add_argument('--output')
    parser.add_argument('--profile',default='so101_pick_place_sim')
    parser.add_argument('--expected-model-sha256')
    args=parser.parse_args()
    if args.evaluate:evaluate(args.evaluate,args.output,args.profile,args.expected_model_sha256)
    else:train(args.job)
