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
process=None
lock=threading.Lock()


def request(command=None,**kwargs):
    data=json.dumps({'command':command,**kwargs}).encode() if command else None
    req=urllib.request.Request(f'http://127.0.0.1:{API_PORT}/',data=data,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=5) as response:
            result=json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(json.load(exc).get('error','Simulator request failed')) from exc
    return result


def start():
    global process
    with lock:
        try:
            return request()
        except urllib.error.URLError:
            pass
        if process is None or process.poll() is not None:
            (DATA/'logs').mkdir(parents=True,exist_ok=True)
            with (DATA/'logs/simulator.log').open('a') as log:
                process=subprocess.Popen([str(ROOT/'.venv-sim/bin/python'),'-m','explorer3.sim.worker',
                    '--api-port',str(API_PORT),'--viewer-port',str(VIEWER_PORT)],cwd=ROOT,
                    env={**offline_environment(),'MUJOCO_GL':'egl'},stdout=log,stderr=subprocess.STDOUT)
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        try:
            return request()
        except urllib.error.URLError:
            if process.poll() is not None:
                raise RuntimeError('Симулятор не запустился; журнал logs/simulator.log')
            time.sleep(.2)
    raise TimeoutError('Запуск симулятора дольше 30 секунд')


def shutdown():
    if process is not None and process.poll() is None:
        process.terminate()
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
