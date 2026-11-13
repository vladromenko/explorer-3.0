"""Fresh-process ACT evaluation on actual MuJoCo dynamics and unseen seeds."""
import argparse
import base64
import hashlib
import json
import subprocess
import time
import selectors
import os
from pathlib import Path
import cv2
import numpy as np
from explorer3.paths import ROOT,offline_environment
from .factory import make
from explorer3.profiles import profile as profile_config


class ACTClient:
    def __init__(self,checkpoint,log):
        self.log=log.open('w')
        self.process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m','explorer3.sim.policy_worker',str(checkpoint)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0,env=offline_environment())
        self.buffer=b'';self.selector=selectors.DefaultSelector();self.selector.register(self.process.stdout,selectors.EVENT_READ)
        os.set_blocking(self.process.stdin.fileno(),False)
        try:self.ready=self.read(timeout=90)
        except Exception:self.close();raise

    def read(self,timeout=30):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if b'\n' in self.buffer:
                line,self.buffer=self.buffer.split(b'\n',1)
                if line.startswith(b'EXPLORER_JSON '):return json.loads(line[len(b'EXPLORER_JSON '):])
            elif self.selector.select(max(0.,deadline-time.monotonic())):
                chunk=os.read(self.process.stdout.fileno(),65536)
                if not chunk:raise RuntimeError('ACT worker exited; inspect inference log')
                self.buffer+=chunk
        raise TimeoutError('ACT inference response deadline exceeded')

    def request(self,command,**kwargs):
        data=(json.dumps({'command':command,**kwargs})+'\n').encode()
        view=memoryview(data)
        deadline=time.monotonic()+30
        with selectors.DefaultSelector() as writable:
            writable.register(self.process.stdin,selectors.EVENT_WRITE)
            while view:
                if time.monotonic()>=deadline or not writable.select(max(0.,deadline-time.monotonic())):
                    raise TimeoutError('ACT request write deadline exceeded')
                try:
                    count=os.write(self.process.stdin.fileno(),view)
                    if not count:raise BrokenPipeError('ACT worker stopped accepting input')
                    view=view[count:]
                except BlockingIOError:pass
        return self.read()

    def action(self,obs,cfg):
        return self.observation_request('action',obs,cfg)['action']

    def observation_request(self,command,obs,cfg):
        ok,png=cv2.imencode('.png',cv2.cvtColor(obs['rgb'],cv2.COLOR_RGB2BGR))
        if not ok:raise RuntimeError('Camera encode failed')
        return self.request(command,state=obs['state'].tolist(),rgb=base64.b64encode(png).decode(),camera_key='observation.images.'+cfg['camera'],task=cfg['task'])

    def close(self):
        self.process.stdin.close()
        try:self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        self.selector.close();self.process.stdout.close()
        self.log.close()


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
    result={'status':'completed','domain':'sim','policy':'ACT','checkpoint':str(checkpoint),
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
