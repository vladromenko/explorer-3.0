"""Explorer manual operator channel; UI presence never auto-enables a robot."""
import asyncio
import json
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


class Operator:
    def __init__(self):
        self.lock=threading.RLock();self.owner=None;self.lease=None;self.generation=0
        self.inputs={};self.sequence=0;self.error=None;self.status={}
        self.leader=None;self.leader_sample=None;self.leader_thread=None;self.leader_done=threading.Event()
        self.mirror=False;self.done=threading.Event();self.operation=threading.Lock()
        self.leader_count=0
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def input(self,owner,value):
        with self.lock:self.inputs[owner]=(time.monotonic(),value)

    def enable(self,owner,profile,arm_mode,observing):
        if not enabled():raise PermissionError('Ручное управление ROSMASTER ещё не разрешено')
        with self.operation:
            with self.lock:
                if self.lease:raise RuntimeError('Сначала STOP: управление уже занято')
                stamp,intent=self.inputs.get(owner,(0,ZERO))
                if time.monotonic()-stamp>.3:raise RuntimeError('Вернитесь в активную вкладку управления')
                if any(intent['drive']+intent['arm']+[intent['pitch'],intent['roll'],intent['grip']]):
                    raise RuntimeError('Отпустите клавиши и джойстик перед включением')
                generation=self.generation
            response=request('/session',{'profile':profile,'arm_mode':arm_mode,'observing':observing})
            with self.lock:
                if generation!=self.generation:
                    request('/stop',{})
                    raise RuntimeError('STOP во время включения')
                self.owner=owner;self.lease=response['lease'];self.sequence=0;self.error=None

    def stop(self,reason='STOP'):
        with self.lock:
            stamp,_=self.inputs.get(self.owner,(0,ZERO))
            record={'at':time.time(),'reason':reason,'had_session':self.lease is not None,
                    'browser_age_s':time.monotonic()-stamp if stamp else None}
            path=DATA/'logs/rosmaster-control.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a') as output:output.write(json.dumps(record,ensure_ascii=False)+'\n')
            self.generation+=1;self.lease=None;self.owner=None;self.mirror=False
            self.error=reason;self.leader_done.set()
        try:request('/stop',{},timeout=1)
        except (OSError,ValueError,httpx.HTTPError,RuntimeError) as exc:
            self.error=reason+'; подтверждение STOP не получено: '+str(exc)

    def lost(self,owner):
        with self.lock:
            self.inputs.pop(owner,None)
            active=self.owner==owner
        if active:self.stop('Вкладка скрыта / связь потеряна')

    def start_leader(self,owner):
        with self.lock:
            if owner!=self.owner or not self.lease:raise RuntimeError('Сначала включите управление рукой')
            if self.leader_thread and self.leader_thread.is_alive():raise RuntimeError('Leader уже подключён')
            self.leader_done.clear();self.leader_sample=None;self.leader_count=0
            self.leader_thread=threading.Thread(target=self.read_leader,daemon=True);self.leader_thread.start()

    def read_leader(self):
        from .leader import LeaderReader
        from .hardware import JOINTS
        reader=LeaderReader()
        try:
            reader.start()
            while not self.leader_done.is_set():
                sample=reader.sample()
                with self.lock:
                    self.leader_sample=(time.monotonic(),dict(zip(JOINTS,sample['positions'])));self.leader_count+=1
                self.leader_done.wait(1/30)
        except Exception as exc:self.stop('Leader: '+str(exc))
        finally:
            reader.close()
            with self.lock:self.leader_sample=None

    def enable_mirror(self,owner):
        with self.lock:
            if owner!=self.owner or not self.lease:raise RuntimeError('Нет вашей активной сессии')
            if self.leader_sample is None or time.monotonic()-self.leader_sample[0]>.2:raise RuntimeError('Нет свежих данных Leader')
            self.mirror=True

    def run(self):
        client=None;last=time.monotonic()
        while not self.done.wait(max(.001,1/30-(time.monotonic()-last))):
            last=time.monotonic()
            with self.lock:
                lease=self.lease;generation=self.generation
                stamp,intent=self.inputs.get(self.owner,(0,ZERO))
                if lease:self.sequence+=1
                sample=self.leader_sample
                packet={'lease':lease,'sequence':self.sequence,'intent':intent,'mirror':self.mirror}
                if sample is not None and time.monotonic()-sample[0]<.2:packet['leader']=sample[1]
            if lease:
                if time.monotonic()-stamp>.3:
                    self.stop('Нет heartbeat активной вкладки')
                else:
                    try:
                        if client is None:client=connection(.2)
                        response=client.post('/command',json=packet)
                        if not response.is_success:raise RuntimeError(response.json().get('detail','Команда отклонена'))
                        with self.lock:
                            if generation==self.generation:
                                self.status=response.json()
                                if not self.status['armed']:
                                    self.lease=None;self.owner=None;self.mirror=False;self.leader_done.set()
                    except Exception as exc:
                        with self.lock:
                            if generation==self.generation:self.stop(str(exc))
                        if client is not None:client.close();client=None
        if client is not None:client.close()


operator=Operator()


@app.on_shutdown
def shutdown():
    operator.done.set()
    if operator.lease:operator.stop('Homebrain shutting down')


@app.get('/api/rosmaster/camera.mjpg')
async def camera(request_:Request):
    cfg=json.loads((DATA/'configs/rosmaster-agent.json').read_text())
    async def frames():
        async with httpx.AsyncClient(base_url=f"http://{cfg['host']}:{cfg['port']}",
                headers={'X-Explorer-Token':cfg['token']},trust_env=False,timeout=1) as client:
            while not await request_.is_disconnected():
                start=time.monotonic()
                try:
                    result=await client.get('/frame.jpg');result.raise_for_status()
                    yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(result.content)).encode()+b'\r\n\r\n'+result.content+b'\r\n'
                except httpx.HTTPError:await asyncio.sleep(.2)
                await asyncio.sleep(max(0,.1-(time.monotonic()-start)))
    return StreamingResponse(frames(),media_type='multipart/x-mixed-replace; boundary=frame',headers={'Cache-Control':'no-store'})


