"""Bounded local stdio interface to frozen official LeRobot inference, no hardware."""
import base64
import json
import os
import selectors
import subprocess
import time
import cv2
from .paths import ROOT,offline_environment


class PolicyClient:
    def __init__(self,checkpoint,log,*,device="cpu",response_timeout=30):
        self.response_timeout=response_timeout
        self.log=log.open('w')
        self.process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m','explorer3.sim.policy_worker',str(checkpoint),'--device',device],
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
                if not chunk:raise RuntimeError('Policy worker exited; inspect inference log')
                self.buffer+=chunk
        raise TimeoutError('Policy inference response deadline exceeded')

    def request(self,command,**kwargs):
        data=(json.dumps({'command':command,**kwargs})+'\n').encode()
        view=memoryview(data)
        deadline=time.monotonic()+30
        with selectors.DefaultSelector() as writable:
            writable.register(self.process.stdin,selectors.EVENT_WRITE)
            while view:
                if time.monotonic()>=deadline or not writable.select(max(0.,deadline-time.monotonic())):
                    raise TimeoutError('Policy request write deadline exceeded')
                try:
                    count=os.write(self.process.stdin.fileno(),view)
                    if not count:raise BrokenPipeError('Policy worker stopped accepting input')
                    view=view[count:]
                except BlockingIOError:pass
        return self.read(timeout=self.response_timeout)

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


# Retain the validated ACT-specific caller API. Ready metadata identifies type.
ACTClient=PolicyClient
