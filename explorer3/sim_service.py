"""UI bridge to the isolated simulator; no hardware commands or imports."""
import json
import os
import subprocess
import threading
import time
import urllib.request
from .paths import ROOT,DATA,offline_environment

API_PORT=int(os.environ.get('EXPLORER3_SIM_API_PORT','8032'))
VIEWER_PORT=int(os.environ.get('EXPLORER3_SIM_VIEWER_PORT','8031'))
class SimulatorProcess:
    def __init__(self,module,api_port,viewer_port,label):
        self.module=module;self.api_port=api_port;self.viewer_port=viewer_port;self.label=label
        self.process=None;self.lock=threading.Lock()

    def request(self,command=None,**kwargs):
        data=json.dumps({'command':command,**kwargs}).encode() if command else None
        req=urllib.request.Request(f'http://127.0.0.1:{self.api_port}/',data=data,headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=5) as response:return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(json.load(exc).get('error','Simulator request failed')) from exc

    def start(self):
        with self.lock:
            try:return self.request()
            except urllib.error.URLError:pass
            if self.process is None or self.process.poll() is not None:
                (DATA/'logs').mkdir(parents=True,exist_ok=True)
                with (DATA/'logs'/f'{self.label}.log').open('a') as log:
                    self.process=subprocess.Popen([str(ROOT/'.venv-sim/bin/python'),'-m',self.module,
                        '--api-port',str(self.api_port),'--viewer-port',str(self.viewer_port)],cwd=ROOT,
                        env={**offline_environment(),'MUJOCO_GL':'egl','OPENBLAS_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            try:return self.request()
            except urllib.error.URLError:
                if self.process.poll() is not None:raise RuntimeError(f'Симулятор не запустился; журнал logs/{self.label}.log')
                time.sleep(.2)
        raise TimeoutError('Запуск симулятора дольше 30 секунд')

    def shutdown(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            # Worker finalizes the current spool then waits up to 60 s for its
            # official dataset writer. Never kill it at the former 20 s mark.
            try:self.process.wait(timeout=70)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)


service=SimulatorProcess('explorer3.sim.worker',API_PORT,VIEWER_PORT,'simulator')
request=service.request
start=service.start
shutdown=service.shutdown
