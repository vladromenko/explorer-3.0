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
            self.leader_done.clear();self.leader_sample=None
            self.leader_thread=threading.Thread(target=self.read_leader,daemon=True);self.leader_thread.start()

    def read_leader(self):
        from .leader import LeaderReader
        from .hardware import JOINTS
        reader=LeaderReader()
        try:
            reader.start()
            while not self.leader_done.is_set():
                sample=reader.sample()
                with self.lock:self.leader_sample=(time.monotonic(),dict(zip(JOINTS,sample['positions'])))
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
        client=None
        while not self.done.wait(.05):
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
                        self.stop(str(exc))
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


@ui.page('/rosmaster/control')
def page():
    client=ui.context.client;owner=client.id
    ui.dark_mode().enable()
    ui.link('← Explorer 3.0','/')
    ui.label('ROSMASTER · управление Explorer').style('font-size:26px')
    message=ui.label('Наблюдение. Включение процесса само по себе не двигает робот.')
    live_state=ui.label('Связь проверяется').style('font-size:20px')
    arm_state=ui.label()
    async def action(fn,*args):
        try:
            result=await asyncio.to_thread(fn,*args)
            message.set_text(str(result) if result else 'Команда принята')
        except Exception as exc:message.set_text(str(exc));ui.notify(str(exc),type='negative')
    async def restore():
        await action(operator.stop,'Возврат прежнего управления')
        await action(request,'/restore',{},45)
    with ui.row():
        ui.button('Передать управление Explorer',on_click=lambda:action(request,'/acquire',{},45))
        ui.button('Прежнее управление',on_click=restore)
        ui.button('STOP',color='negative',on_click=lambda:action(operator.stop,'STOP из интерфейса')).props('size=lg')
    ui.label('Перед движением: зарядник отключён, пространство свободно, оператор рядом. Другая панель закрыта.')
    observing=ui.checkbox('Я рядом, зарядник отключён, пространство свободно, захват пуст')
    ui.button('Подготовить руку · исходная поза за 4 секунды',
        on_click=lambda:action(operator.enable,owner,'prepare','joint',observing.value))
    ui.label('Подготовка — прежняя поза 90° × 6. Она не запускается автоматически. Держите эту вкладку активной; STOP прервёт команду.')
    with ui.row():
        profile=ui.select({'rosmaster_pick_place':'Рука · шасси неподвижно','rosmaster_dock':'Шасси · рука неподвижна'},value='rosmaster_pick_place',label='Управление')
        mode=ui.select({'cartesian':'1 · Захват XYZ','joint':'2 · Суставы'},value='cartesian',label='Режим руки')
        ui.button('Включить управление',on_click=lambda:action(operator.enable,owner,profile.value,mode.value,observing.value))
    ui.label('XYZ: стрелки — вперёд/назад и влево/вправо, R/F — выше/ниже. Суставы: ←/→ основание, ↑/↓ плечо, R/F локоть.')
    ui.label('В обоих режимах: ,/. — наклон кисти, [/] — вращение, N/M — открыть/закрыть. Shift — медленнее. Пробел/Esc — STOP.')
    ui.label('Шасси: W/S вперёд/назад, A/D боком, Q/E поворот. Переключение профиля или режима требует STOP и нового включения.')
    ui.label('Джойстик: левый стик — шасси, LB/RB — поворот; правый стик — рука, крестовина — высота/наклон, LT/RT — захват, B — STOP. Перед включением отпустите все органы управления.')
    with ui.row():
        ui.button('Подключить Leader',on_click=lambda:action(operator.start_leader,owner))
        ui.button('Включить зеркало',on_click=lambda:action(operator.enable_mirror,owner))
    ui.label('Зеркало использует сохранённый адаптер :8090; знаки и пределы не изменены. Leader подключается только этой кнопкой.')
    ui.html('<img src="/api/rosmaster/camera.mjpg" style="width:640px;max-width:100%" alt="Камера ROSMASTER">')
    with ui.expansion('Технические данные').classes('w-full'):
        state=ui.code('',language='json').style('width:100%;max-height:400px;overflow:auto')
    ui.on('rosmaster_input',lambda event:operator.input(owner,event.args))
    ui.on('rosmaster_stop',lambda:action(operator.lost,owner))
    client.on_disconnect(lambda:operator.lost(owner))
    async def change():
        if operator.owner==owner:await action(operator.stop,'Режим изменён: включите управление заново')
    profile.on_value_change(change);mode.on_value_change(change)
    ui.on('rosmaster_mode',lambda event:mode.set_value(event.args))
    async def refresh():
        try:
            health=await asyncio.to_thread(request,'/health')
            runtime=health.get('status')
            if runtime is None:live_state.set_text('Наблюдение · нажмите «Передать управление Explorer»')
            elif runtime['armed']:
                live_state.set_text('Подготовка руки · не закрывайте вкладку' if runtime['preparing'] else
                    ('Следование за Leader' if runtime['mode']=='mirror' else 'Управление готово'))
            elif runtime['reason'] not in (None,'Explicit enable required'):live_state.set_text('Остановлено · '+runtime['reason'])
            else:live_state.set_text('Контроллер готов · управление выключено' if runtime['controller_ready'] else 'Контроллер готовится')
            measured=runtime.get('arm_measured') if runtime else (health.get('observation') or {}).get('arm_measured')
            arm_state.set_text('Суставы: '+', '.join(str(round(v,1)) for v in measured) if measured and all(v is not None for v in measured) else 'Нет свежих измерений всех суставов')
            state.set_content(json.dumps({'runtime':health.get('status'),'homebrain':operator.error,
                'leader_fresh':operator.leader_sample is not None and time.monotonic()-operator.leader_sample[0]<.2},ensure_ascii=False,indent=2))
        except Exception as exc:state.set_content(str(exc));live_state.set_text('Нет связи с ROSMASTER')
    ui.timer(1,refresh)
    ui.add_body_html(CONTROLS_JS)


CONTROLS_JS='''<script>
(()=>{
const keys=new Set();
let padBlocked=true;
const map={KeyW:['drive',0,1],KeyS:['drive',0,-1],KeyA:['drive',1,1],KeyD:['drive',1,-1],KeyQ:['drive',2,1],KeyE:['drive',2,-1],ArrowUp:['arm',0,1],ArrowDown:['arm',0,-1],ArrowLeft:['arm',1,1],ArrowRight:['arm',1,-1],KeyI:['arm',0,1],KeyK:['arm',0,-1],KeyJ:['arm',1,1],KeyL:['arm',1,-1],KeyR:['arm',2,1],KeyF:['arm',2,-1]};
const scalars={Comma:['pitch',-1],Period:['pitch',1],BracketLeft:['roll',-1],BracketRight:['roll',1],KeyN:['grip',-1],KeyM:['grip',1]};
const stop=()=>{keys.clear();padBlocked=true;emitEvent('rosmaster_stop');};
window.addEventListener('keydown',e=>{
 if(['Space','Escape'].includes(e.code)){e.preventDefault();stop();return;}
 if(['INPUT','TEXTAREA'].includes(e.target.tagName)||e.target.isContentEditable)return;
 if(['Digit1','Digit2'].includes(e.code)){if(!e.repeat){stop();emitEvent('rosmaster_mode',e.code==='Digit1'?'cartesian':'joint');}return;}
 if(e.repeat)return;
 if(map[e.code]||scalars[e.code]||e.code.startsWith('Shift')){e.preventDefault();keys.add(e.code);}
});
window.addEventListener('keyup',e=>keys.delete(e.code));
window.addEventListener('blur',stop);window.addEventListener('pagehide',stop);
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
setInterval(()=>{
 if(document.hidden||!document.hasFocus())return;
 const x={drive:[0,0,0],arm:[0,0,0],pitch:0,roll:0,grip:0,slow:keys.has('ShiftLeft')||keys.has('ShiftRight')};
 for(const k of keys){if(map[k]){const [g,i,s]=map[k];x[g][i]+=s;}if(scalars[k]){const [g,s]=scalars[k];x[g]+=s;}}
 const pad=Array.from(navigator.getGamepads?navigator.getGamepads():[]).find(p=>p&&p.mapping==='standard'&&p.axes.length>=4&&p.buttons.length>=16);
 if(pad){
  if(pad.buttons[1].pressed){stop();return;}
  if(padBlocked&&pad.axes.every(v=>Math.abs(v)<.12)&&pad.buttons.every(b=>!b.pressed&&b.value<.12))padBlocked=false;
  if(!padBlocked){const z=v=>Math.abs(v)<.12?0:v;
   x.drive[0]+=-z(pad.axes[1]);x.drive[1]+=-z(pad.axes[0]);
   if(pad.buttons[4].pressed)x.drive[2]+=1;if(pad.buttons[5].pressed)x.drive[2]-=1;
   x.arm[1]+=-z(pad.axes[2]);x.arm[0]+=-z(pad.axes[3]);
   if(pad.buttons[12].pressed)x.arm[2]+=1;if(pad.buttons[13].pressed)x.arm[2]-=1;
   if(pad.buttons[14].pressed)x.pitch-=1;if(pad.buttons[15].pressed)x.pitch+=1;
   x.grip+=(pad.buttons[7].value||0)-(pad.buttons[6].value||0);
  }
 }
 for(const g of ['drive','arm'])x[g]=x[g].map(v=>Math.max(-1,Math.min(1,v)));
 for(const g of ['pitch','roll','grip'])x[g]=Math.max(-1,Math.min(1,x[g]));
 emitEvent('rosmaster_input',x);
},100);
})();</script>'''
