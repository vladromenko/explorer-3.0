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
from explorer3.paths import DATA,ROOT,offline_environment
from .environment import StateTeacher
from .factory import make,ENVIRONMENTS


class SaveProgress(BaseCallback):
    def __init__(self,work,total):
        super().__init__();self.work=work;self.total=total;self.cancelled=False;self.started=time.monotonic()

    def _on_step(self):
        if self.num_timesteps%256==0 or self.cancelled:
            (self.work/'progress.json').write_text(json.dumps({'steps':self.num_timesteps,'budget':self.total,
                'elapsed_s':time.monotonic()-self.started,'device':'cpu','domain':'sim'}))
        if self.num_timesteps%1024==0 or self.cancelled:
            self.model.save(self.work/'latest')
        return not self.cancelled


def evaluate(checkpoint,output,profile='so101_pick_place_sim'):
    model=PPO.load(checkpoint,device='cpu')
    env=StateTeacher(make(profile))
    trials=[]
    try:
        for seed in range(10000,10020):
            obs,_=env.reset(seed=seed);done=False
            while not done:
                action,_=model.predict(obs,deterministic=True)
                obs,_,terminated,truncated,info=env.step(action);done=terminated or truncated
            trials.append({'seed':seed,'outcome':info['outcome'],'reason':info['termination_reason']})
    finally:
        env.close()
    result={'domain':'sim','policy':'SB3 PPO state teacher','privileged_state':True,'hardware_deployable':False,
            'new_process_reload':True,'trials':trials,'success_fraction':sum(t['outcome']=='success' for t in trials)/len(trials)}
    Path(output).write_text(json.dumps(result,indent=2));return result


def train(path):
    cfg=json.loads(path.read_text());work=DATA/'jobs'/path.stem;work.mkdir(parents=True,exist_ok=True)
    if cfg['profile'] not in ENVIRONMENTS:raise ValueError('Unsupported simulator profile')
    budget=int(cfg['steps'])
    if not 256<=budget<=1000000:raise ValueError('Invalid finite training budget')
    profile=cfg['profile']
    env=Monitor(StateTeacher(make(profile)),filename=str(work/'episodes.csv'))
    if cfg.get('resume_checkpoint'):
        model=PPO.load(cfg['resume_checkpoint'],env=env,device='cpu')
    else:
        model=PPO('MlpPolicy',env,device='cpu',seed=cfg['seed'],n_steps=256,batch_size=64,n_epochs=4,
                  policy_kwargs={'net_arch':[128,128]},verbose=1)
    callback=SaveProgress(work,budget)
    signal.signal(signal.SIGTERM,lambda *_:setattr(callback,'cancelled',True))
    remaining=max(0,budget-model.num_timesteps)
    try:
        if remaining:
            model.learn(remaining,callback=callback,reset_num_timesteps=False)
        model.save(work/'latest')
    finally:env.close()
    if callback.cancelled:return
    subprocess.run([sys.executable,'-m','explorer3.sim.ppo','--evaluate',str(work/'latest.zip'),
                    '--output',str(work/'evaluation.json'),'--profile',profile],check=True,env=offline_environment())
    (work/'result.json').write_text(json.dumps({'checkpoint':str(work/'latest.zip'),'domain':'sim',
        'steps':model.num_timesteps,'budget':budget,'physical_skill_verified':False,
        'evaluation':json.loads((work/'evaluation.json').read_text())},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path);parser.add_argument('--evaluate');parser.add_argument('--output')
    parser.add_argument('--profile',default='so101_pick_place_sim')
    args=parser.parse_args()
    if args.evaluate:evaluate(args.evaluate,args.output,args.profile)
    else:train(args.job)
