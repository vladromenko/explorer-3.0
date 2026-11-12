"""Explorer manual operator channel; UI presence never auto-enables a robot."""
import asyncio
import json
import subprocess
from pathlib import Path
import threading
import time
import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse
from nicegui import app, ui
from .paths import DATA

ZERO={'drive':[0,0,0],'arm':[0,0,0],'pitch':0,'roll':0,'grip':0,'slow':False}


def connection(timeout=2):
    cfg=json.loads((DATA/'configs/rosmaster-agent.json').read_text())
    return httpx.Client(base_url=f"http://{cfg['host']}:{cfg['port']}",
                        headers={'X-Explorer-Token':cfg['token']},timeout=timeout,trust_env=False)


def enabled():
    path=DATA/'configs/rosmaster-agent.json'
    return path.exists() and json.loads(path.read_text()).get('manual_control_enabled') is True


def request(path,value=None,timeout=2):
    if path in {'/acquire','/session','/command'} and not enabled():raise PermissionError('Ручное управление ROSMASTER ещё не разрешено')
    with connection(timeout) as client:
        result=client.get(path) if value is None else client.post(path,json=value)
        if not result.is_success:
            raise RuntimeError(result.json().get('detail','ROSMASTER request failed'))
        return result.json()


def ensure_agent(wait=40):
    """Start the inert agent unit after a robot reboot; it opens nothing until /acquire."""
    try:
        request('/health');return False
    except httpx.TransportError:pass
    cfg=json.loads((DATA/'configs/rosmaster-agent.json').read_text())
    key=Path.home()/'.ssh/rosmaster-collector'
    result=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','-o','ConnectTimeout=5',
        f"{cfg.get('ssh_user','vlad')}@{cfg['host']}",'systemctl --user start explorer3-rosmaster-agent.service'],
        capture_output=True,text=True,timeout=40)
    if result.returncode:raise RuntimeError('Агент ROSMASTER не запущен: '+(result.stderr.strip() or 'ssh error'))
    end=time.monotonic()+wait
    while time.monotonic()<end:
        try:
            request('/health');return True
        except httpx.TransportError:time.sleep(1)
    raise RuntimeError('Агент ROSMASTER запущен, но не отвечает')


class Operator:
    def __init__(self):
        self.lock=threading.RLock();self.owner=None;self.lease=None;self.generation=0
        self.inputs={};self.sequence=0;self.error=None;self.status={}
        self.leader=None;self.leader_sample=None;self.leader_thread=None;self.leader_done=threading.Event()
        self.mirror=False;self.done=threading.Event();self.operation=threading.Lock()
        self.leader_count=0;self.leader_owner=None;self.leader_error=None
        self.direct=None;self.source=None
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def input(self,owner,value):
        with self.lock:self.inputs[owner]=(time.monotonic(),value)

    def enable(self,owner,profile,arm_mode,observing,source='human'):
        if not enabled():raise PermissionError('Ручное управление ROSMASTER ещё не разрешено')
        with self.operation:
            with self.lock:
                if self.lease:
                    if self.owner==owner:return 'already'
                    raise RuntimeError('Управление уже включено в другой вкладке или на другом устройстве. Нажмите STOP и включите здесь')
                stamp,intent=self.inputs.get(owner,(0,ZERO))
                if time.monotonic()-stamp>.3:raise RuntimeError('Вернитесь в активную вкладку управления')
                if any(intent['drive']+intent['arm']+[intent['pitch'],intent['roll'],intent['grip']]):
                    raise RuntimeError('Отпустите клавиши и джойстик перед включением')
                generation=self.generation
            response=request('/session',{'profile':profile,'arm_mode':arm_mode,'observing':observing,'source':source})
            with self.lock:
                if generation!=self.generation:
                    request('/stop',{})
                    raise RuntimeError('STOP во время включения')
                self.owner=owner;self.lease=response['lease'];self.sequence=0;self.error=None;self.direct=None;self.source=None
                self.status={'profile':profile,'armed':True,'source':source}

    def stop(self,reason='STOP'):
        with self.lock:
            stamp,_=self.inputs.get(self.owner,(0,ZERO))
            if self.lease is not None:
                self.record({'reason':reason,'source':'homebrain','browser_age_s':round(time.monotonic()-stamp,3) if stamp else None,
                             **getattr(self,'stats',{})})
            self.generation+=1;self.lease=None;self.owner=None;self.mirror=False;self.direct=None;self.source=None
            self.error=reason
        try:request('/stop',{},timeout=1)
        except (OSError,ValueError,httpx.HTTPError,RuntimeError) as exc:
            self.error=reason+'; подтверждение STOP не получено: '+str(exc)

    def lost(self,owner):
        with self.lock:
            self.inputs.pop(owner,None)
            active=self.owner==owner
            leader=self.leader_owner==owner
        if active:self.stop('Вкладка скрыта / связь потеряна')
        if leader:self.stop_leader()

    # The Leader is read-only (no torque, no writes), so it stays connected across STOP and
    # new sessions; only the mirror flag is per session. Short bus glitches are retried.
    LEADER_HZ=30
    LEADER_GLITCH_S=.5

    def start_leader(self,owner):
        with self.lock:
            if self.leader_thread and self.leader_thread.is_alive():
                self.leader_owner=owner;return 'Leader уже подключён'
            self.leader_done.clear();self.leader_sample=None;self.leader_count=0;self.leader_error=None
            self.leader_owner=owner
            self.leader_thread=threading.Thread(target=self.read_leader,daemon=True);self.leader_thread.start()
        return 'Подключение Leader'

    def stop_leader(self):
        with self.lock:self.mirror=False;self.leader_done.set();thread=self.leader_thread
        if thread and thread is not threading.current_thread():thread.join(2)
        with self.lock:self.leader_owner=None

    def read_leader(self):
        from .leader import LeaderReader
        from .hardware import JOINTS
        reader=LeaderReader()
        try:
            reader.start()
            last_ok=time.monotonic()
            while not self.leader_done.is_set():
                try:
                    sample=reader.sample()
                except Exception as exc:
                    if time.monotonic()-last_ok>self.LEADER_GLITCH_S:raise
                    self.leader_glitches=getattr(self,'leader_glitches',0)+1
                else:
                    last_ok=time.monotonic()
                    with self.lock:
                        self.leader_sample=(last_ok,dict(zip(JOINTS,sample['positions'])));self.leader_count+=1
                self.leader_done.wait(1/self.LEADER_HZ)
        except Exception as exc:
            # The agent's mirror times out on its own and returns the arm to manual mode.
            with self.lock:self.leader_error='Leader: '+str(exc);self.mirror=False
        finally:
            reader.close()
            with self.lock:self.leader_sample=None

    def enable_mirror(self,owner):
        with self.lock:
            if owner!=self.owner or not self.lease:raise RuntimeError('Сначала включите управление рукой')
            if self.leader_sample is None or time.monotonic()-self.leader_sample[0]>.2:raise RuntimeError('Нет свежих данных Leader: нажмите «Подключить Leader»')
            self.mirror=True

    def set_source(self,value):
        """'policy' or 'human' inside a policy session (pilot correction hand-over)."""
        with self.lock:self.source=value;self.direct=None

    def set_direct(self,value):
        """Policy/replay action for the current single-task session; None clears it."""
        with self.lock:self.direct=(time.monotonic(),value) if value is not None else None

    def disable_mirror(self,owner):
        with self.lock:self.mirror=False

    # Browser input older than INPUT_HOLD_S is replaced by zero intent (motion stops at once,
    # session stays); after BROWSER_LOST_S without input the session is stopped. The agent's own
    # 0.4 s heartbeat and the controller's 240 ms lease remain the hard limits, so one late HTTP
    # reply is retried instead of ending the session.
    INPUT_HOLD_S=.25
    BROWSER_LOST_S=1.
    COMMAND_TIMEOUT_S=.25
    COMMAND_LOST_S=.35

    def record(self,event):
        path=DATA/'logs/rosmaster-control.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as output:output.write(json.dumps({'at':time.time(),**event},ensure_ascii=False)+'\n')

    def agent_reason(self):
        try:
            with connection(.5) as client:
                status=client.get('/health').json().get('status') or {}
            return status.get('reason')
        except Exception:return None

    def ended_by_agent(self,generation,detail):
        """The agent ended the session itself; show its own reason, do not mask it with STOP."""
        reason=self.agent_reason() or detail
        with self.lock:
            if generation!=self.generation:return
            self.generation+=1;self.lease=None;self.owner=None;self.mirror=False
            self.error='Робот остановил сессию: '+str(reason)
        self.record({'reason':self.error,'source':'agent','detail':detail,**self.stats})
        try:request('/stop',{},timeout=1)
        except Exception:pass

    def run(self):
        client=None;last=time.monotonic()
        self.stats={}
        while not self.done.wait(max(.001,1/30-(time.monotonic()-last))):
            last=time.monotonic()
            with self.lock:
                lease=self.lease;generation=self.generation
                stamp,intent=self.inputs.get(self.owner,(0,ZERO))
                if lease:self.sequence+=1
                sample=self.leader_sample
                age=last-stamp
                if age>self.INPUT_HOLD_S:intent={**ZERO,'free':False}
                packet={'lease':lease,'sequence':self.sequence,'intent':intent,'mirror':self.mirror}
                if sample is not None and last-sample[0]<.2:packet['leader']=sample[1]
                if self.direct is not None and last-self.direct[0]<.3:packet['direct']=self.direct[1]
                if self.source:packet['source']=self.source
            if not lease:
                self.stats={};continue
            stats=self.stats
            if not stats:stats.update(started=last,last_ok=last,commands=0,timeouts=0,held=0,max_rtt=0.,max_browser_gap=0.)
            stats['max_browser_gap']=max(stats['max_browser_gap'],age)
            if age<=self.INPUT_HOLD_S and any(abs(v)>1e-3 for v in [*intent.get('drive',[]),*intent.get('arm',[]),intent.get('pitch',0),intent.get('roll',0),intent.get('grip',0)]):
                stats['active_inputs']=stats.get('active_inputs',0)+1
            if age>self.INPUT_HOLD_S:stats['held']+=1
            if age>self.BROWSER_LOST_S:
                self.stop(f'Нет heartbeat активной вкладки ({age:.2f} с)');continue
            try:
                if client is None:client=connection(self.COMMAND_TIMEOUT_S)
                sent=time.monotonic()
                response=client.post('/command',json=packet)
                rtt=time.monotonic()-sent
            except httpx.TransportError as exc:
                stats['timeouts']+=1
                if client is not None:client.close();client=None
                if time.monotonic()-stats['last_ok']>self.COMMAND_LOST_S:
                    with self.lock:current=generation==self.generation
                    if current:self.stop(f'Нет ответа агента {time.monotonic()-stats["last_ok"]:.2f} с: {type(exc).__name__}')
                continue
            if not response.is_success:
                try:detail=response.json().get('detail','Команда отклонена')
                except ValueError:detail=f'HTTP {response.status_code}'
                self.ended_by_agent(generation,detail);continue
            stats['last_ok']=time.monotonic();stats['commands']+=1;stats['max_rtt']=max(stats['max_rtt'],rtt)
            with self.lock:
                if generation==self.generation:
                    self.status=response.json()
                    if not self.status['armed']:
                        reason=self.status.get('reason')
                        self.lease=None;self.owner=None;self.mirror=False;self.generation+=1
                        self.error='Сессия завершена: '+str(reason)
                        done=True
                    else:done=False
                else:done=False
            if done:self.record({'reason':self.error,'source':'agent-status',**stats})
        if client is not None:client.close()


operator=Operator()


@app.on_shutdown
def shutdown():
    operator.done.set()
    if operator.lease:operator.stop('Homebrain shutting down')


def brighten(jpeg,level):
    """Display-only enhancement for a dark room; recordings and the agent keep raw frames."""
    import cv2
    import numpy as np
    image=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
    if image is None:return jpeg
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB);light=lab[:,:,0]
    mean=max(float(light.mean()),1.)/255
    target=(.35,.45,.55)[level-1]
    gamma=min(1.,max(.25,np.log(target)/np.log(mean))) if mean<target else 1.
    table=(np.linspace(0,1,256)**gamma*255).astype(np.uint8)
    light=cv2.LUT(light,table)
    light=cv2.createCLAHE(clipLimit=(1.5,2.5,3.5)[level-1],tileGridSize=(8,8)).apply(light)
    if level==3:light=cv2.GaussianBlur(light,(3,3),0)
    lab[:,:,0]=light
    ok,data=cv2.imencode('.jpg',cv2.cvtColor(lab,cv2.COLOR_LAB2BGR),[cv2.IMWRITE_JPEG_QUALITY,80])
    return data.tobytes() if ok else jpeg


@app.get('/api/rosmaster/camera.mjpg')
async def camera(request_:Request,light:int=0):
    cfg=json.loads((DATA/'configs/rosmaster-agent.json').read_text())
    async def frames():
        async with httpx.AsyncClient(base_url=f"http://{cfg['host']}:{cfg['port']}",
                headers={'X-Explorer-Token':cfg['token']},trust_env=False,timeout=1) as client:
            while not await request_.is_disconnected():
                start=time.monotonic()
                try:
                    result=await client.get('/frame.jpg');result.raise_for_status()
                    content=result.content
                    if light in (1,2,3):content=await asyncio.to_thread(brighten,content,light)
                    yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(content)).encode()+b'\r\n\r\n'+content+b'\r\n'
                except httpx.HTTPError:await asyncio.sleep(.2)
                await asyncio.sleep(max(0,.1-(time.monotonic()-start)))
    return StreamingResponse(frames(),media_type='multipart/x-mixed-replace; boundary=frame',headers={'Cache-Control':'no-store'})


