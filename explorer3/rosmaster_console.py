"""Serve the selected Console 8090 frontend over Explorer's own agent protocol."""
import asyncio
import json
import time
import uuid
from pathlib import Path
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnected
from nicegui import app
from .rosmaster_control import operator, request, connection, ensure_agent
from .rosmaster_session import RosmasterSession, PROFILES
from .paths import DATA
from . import local_work

# Actions that open the robot (serial/camera/motors). STOP and status reads are never gated.
HARDWARE_ACTIONS={'acquire','enable','prepare','mirror_on','mirror_toggle','leader','record_start','trial_start','preset','pair_capture'}

recorder=RosmasterSession(operator)
_plans={'at':0,'items':[]}


def plans():
    """Prepared ROSMASTER trial plans (replay/policy), refreshed at most every 2 s."""
    if time.monotonic()-_plans['at']>2:
        items=[]
        for path in sorted((DATA/'trials').glob('*.json')):
            plan=json.loads(path.read_text())
            if plan.get('profile') in PROFILES:
                items.append({'id':plan['id'],'label':plan['profile']+' · '+plan['kind']+' · '+plan['id'][:8]})
        from . import research
        research.initialize()
        studies=[{'id':st['id'],'label':st['profile']+' · пилот '+st['id'][:8]+(' · точка '+str(st.get('decision_after_s'))+' с' if st.get('timing_accepted') else ' · время не подтверждено')}
                 for st in research.studies() if st.get('profile') in PROFILES]
        _plans.update(at=time.monotonic(),items=items,studies=studies)
    return _plans['items']


@app.get('/rosmaster/control',response_class=HTMLResponse)
def console():
    return (Path(__file__).parent/'static/rosmaster-console.html').read_text()


@app.post('/api/rosmaster/stop')
async def stop_http():
    """Independent STOP path when the browser's WebSocket has stalled."""
    await asyncio.to_thread(operator.stop,'Браузер: кнопка STOP (HTTP)')
    return {'armed':False,'notice':operator.error,'physical_stop_confirmed':False}


JOINT_NAMES=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')


def mirror_check(s):
    """Per-joint Leader vs robot movement since the mirror was enabled (degrees)."""
    sample=operator.leader_sample;ref=operator.mirror_ref
    measured=s.get('arm_measured')
    rows=[]
    for i,name in enumerate(JOINT_NAMES):
        leader=robot=None
        if sample and ref:leader=round(sample[1][name]-ref[0][name],1)
        if measured and ref and ref[1]:robot=round(measured[i]-ref[1][i],1)
        rows.append({'leader':leader,'robot':robot})
    return {'rows':rows,'sign':s.get('mirror_sign'),'gripper_map':s.get('mirror_gripper_map'),
            'leader_gripper':round(sample[1]['gripper'],1) if sample else None,
            'pair':s.get('mirror_ref_pair'),'aligning':s.get('mirror_aligning')}


REASONS={'Explicit enable required':'Управление выключено. Нажмите «▶ Я рядом — включить»',
         'Operator STOP':'Остановлено (STOP)','Operator heartbeat expired':'Связь с браузером прервалась — включите снова',
         'Browser / Homebrain heartbeat lost':'Связь с браузером прервалась — включите снова'}


def russian(text):
    return REASONS.get(text,text) if text else text


def telemetry(health,mode,notice):
    s=health.get('status') or {};observation=health.get('observation') or {}
    measured=s.get('arm_measured') or observation.get('arm_measured')
    arm=s.get('arm_command') or measured or []
    sample=operator.leader_sample
    age=time.monotonic()-sample[0] if sample else None
    return {'arm_mode':mode,'dry_run':False,'enabled':s.get('armed',False),
        'notice':notice or russian(s.get('reason')) or ('Управление включено' if s.get('armed') else 'Нажмите «▶ Я рядом — включить»'),
        'mode':s.get('mode','manual'),'battery':{'soc':None,'minutes_left':None,'volts':s.get('battery_voltage',observation.get('battery_voltage'))},
        'battery_stale':False,'arm':[round(v,1) if v is not None else '?' for v in arm],
        'arm_measured':measured,'tcp':s.get('tcp') or [],'jog_blocked':s.get('arm_blocked'),
        'mirror':{'on':s.get('mode')=='mirror','requested':bool(operator.mirror),'blocked':s.get('mirror_blocked') or operator.leader_error,'frames':operator.leader_count,
                  'leader':'ошибка' if operator.leader_error else 'подключён' if sample else 'не подключён',
                  'age':round(age,2) if age is not None else None,'target':s.get('mirror_target') or []},
        'guard':s.get('guard'),'lidar_manual_override':s.get('lidar_manual_override',False),'mission':'не запущена','mission_result':None,
        'mirror_check':mirror_check(s),'recording':recorder.status(),'plans':plans(),'studies':_plans.get('studies',[]),'session_profile':s.get('profile'),
        'log':(s.get('logs',[])+([operator.error] if operator.error else [])+([notice] if notice else []))[-14:]}


def log_action(owner,action,value,error):
    """Every panel action and its result, so failures seen only in a browser are diagnosable."""
    try:
        path=DATA/'logs/rosmaster-panel-actions.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        record={'at':time.time(),'tab':owner[:8],'action':action,'error':f'{type(error).__name__}: {error}' if error else None,
                'value':{k:v for k,v in value.items() if k in ('profile','mode','on','joint','which','outcome','plan','study')}}
        with path.open('a') as output:output.write(json.dumps(record,ensure_ascii=False)+'\n')
    except OSError:pass


@app.websocket('/api/rosmaster/console/ws')
async def websocket(socket:WebSocket):
    await socket.accept()
    owner=uuid.uuid4().hex
    state={'mode':'cartesian','notice':'','grip':{}}
    async def push():
        client=None
        try:
            while True:
                try:
                    if not local_work.hardware_allowed('rosmaster'):
                        if client is not None:client.close();client=None
                        payload=telemetry({},state['mode'],'Локальная подготовка: робот не опрашивается. Откройте «Подготовка без робота».')
                        payload['offline']=True
                    else:
                        if client is None:client=connection(1)
                        response=await asyncio.to_thread(client.get,'/health');response.raise_for_status()
                        health=response.json()
                        payload=telemetry(health,state['mode'],state['notice'])
                except Exception as exc:payload=telemetry({},state['mode'],'Нет связи: '+str(exc))
                await socket.send_json(payload)
                await asyncio.sleep(.1)
        finally:
            if client is not None:client.close()
    action_lock=asyncio.Lock();pending=[]
    async def act(action,value):
        try:
            if action in HARDWARE_ACTIONS and not local_work.hardware_allowed('rosmaster'):
                raise PermissionError('Подключение робота сейчас закрыто: идёт локальная подготовка. Физическая проверка выполняется отдельно.')
            if action=='stop':
                reason='Браузер: '+str(value.get('reason','кнопка STOP'))[:100]
                await asyncio.to_thread(operator.stop,reason);state['notice']='Остановлено · '+reason
            elif action=='acquire':
                state['notice']='Запуск агента ROSMASTER…' if await asyncio.to_thread(ensure_agent) else state['notice']
                await asyncio.to_thread(request,'/acquire',{},45);state['notice']='Explorer получил управление. Нажмите «Включить управление»'
            elif action=='restore':
                await asyncio.to_thread(operator.stop,'Возврат прежнего управления')
                await asyncio.to_thread(request,'/restore',{},45);state['notice']='Прежнее управление восстановлено'
            elif action=='prepare' and operator.owner==owner and operator.lease:
                # Already enabled: raise the arm with a preset instead of a new session.
                await asyncio.to_thread(request,'/preset',{'lease':operator.lease,'name':'up'})
                state['notice']='Рука выпрямляется'
            elif action in ('enable','prepare'):
                generation=operator.generation
                profile='prepare' if action=='prepare' else value.get('profile','rosmaster_manual')
                if profile not in {'prepare','rosmaster_manual','rosmaster_dock','rosmaster_pick_place'}:raise ValueError('Неизвестный режим управления')
                # Connect automatically: start the inert agent and take serial/camera if needed.
                health=await asyncio.to_thread(lambda:(ensure_agent(),request('/health'))[1])
                if not health.get('hardware_open'):
                    state['notice']='Подключение ROSMASTER…'
                    await asyncio.to_thread(request,'/acquire',{},45)
                if generation!=operator.generation:raise RuntimeError('Включение отменено кнопкой STOP; нажмите «Включить» заново')
                result=await asyncio.to_thread(operator.enable,owner,profile,state['mode'],value.get('observing') is True)
                state['notice']='Управление уже включено: двигайте клавишами (окно должно быть в фокусе)' if result=='already' else ''
            elif action in ('record_start','trial_start'):
                if recorder.active():raise RuntimeError('Сначала закончите текущую запись')
                plan=None
                if action=='trial_start':
                    from . import trials
                    plan=trials.load(str(value.get('plan','')))
                    trials.require_physical(plan)
                    name=plan['profile']
                    study=None
                    if value.get('study'):
                        from . import research
                        study=next((x for x in research.studies() if x['id']==value.get('study')),None)
                        if study is None:raise ValueError('Исследование не найдено')
                else:name=value.get('profile');study=None
                if name not in PROFILES:raise ValueError('Выберите задачу ROSMASTER')
                source='policy' if plan else 'human'
                current=operator.status or {}
                if not (operator.owner==owner and operator.lease and current.get('profile')==name and current.get('source','human')==source):
                    if operator.lease:await asyncio.to_thread(operator.stop,'Переключение на задачу '+name)
                    health=await asyncio.to_thread(lambda:(ensure_agent(),request('/health'))[1])
                    if not health.get('hardware_open'):await asyncio.to_thread(request,'/acquire',{},45)
                    await asyncio.to_thread(operator.enable,owner,name,state['mode'],value.get('observing') is True,source)
                recorder.start(owner,name,str(value.get('scene','scene-01'))[:60],
                               'correction' if value.get('category')=='correction' else 'demonstration',
                               str(value.get('collection') or time.strftime('%Y-%m-%d')+'-'+name)[:80],plan=plan,study=study)
                state['notice']=('Испытание' if plan else 'Запись')+' идёт: '+name+'. Затем «Успех», «Неуспех» или «Прервать»'
            elif action=='record_finish':
                recorder.finish(value.get('outcome'));state['notice']='Сохранение эпизода: '+str(value.get('outcome'))
            elif action in ('mirror_flip','grip_capture','grip_save','grip_reset','pair_capture','pair_reset'):
                current=(operator.status or {})
                sign=list(current.get('mirror_sign') or [1,-1,1,1,1]);grip=current.get('mirror_gripper_map')
                pair=current.get('mirror_ref_pair')
                if action=='grip_capture':
                    sample=operator.leader_sample
                    if not sample:raise RuntimeError('Сначала «Подключить Leader»')
                    state['grip'][value.get('which')]=sample[1]['gripper']
                    state['notice']='Захват Leader: '+str(value.get('which'))+' = '+str(round(sample[1]['gripper'],1))
                else:
                    if operator.mirror:raise RuntimeError('Сначала выключите зеркало')
                    if action=='mirror_flip':
                        index=int(value.get('joint'));sign[index]=-sign[index]
                    elif action=='pair_capture':
                        sample=operator.leader_sample;measured=current.get('arm_measured')
                        if not sample or time.monotonic()-sample[0]>.3:raise RuntimeError('Сначала подключите Leader (кнопка «Включить зеркало» или «Подключить Leader»)')
                        if not measured:raise RuntimeError('Нет измерений руки ROSMASTER')
                        pair={'leader':[round(sample[1][n],2) for n in JOINT_NAMES],'servo':[round(v,2) for v in measured[:5]]}
                    elif action=='pair_reset':pair=None
                    elif action=='grip_save':
                        if 'open' not in state['grip'] or 'closed' not in state['grip']:raise ValueError('Снимите обе точки: Leader открыт и закрыт')
                        grip=[state['grip']['open'],state['grip']['closed']]
                    elif action=='grip_reset':grip=None
                    if not operator.lease:raise RuntimeError('Сначала «▶ Включить»')
                    await asyncio.to_thread(request,'/mirror_config',{'sign':sign,'gripper_map':grip,'ref_pair':pair})
                    state['notice']='Зеркало: настройка сохранена на роботе'
            elif action=='use8090':
                if operator.lease:await asyncio.to_thread(operator.stop,'Переход в консоль 8090')
                await asyncio.to_thread(operator.stop_leader)
                await asyncio.to_thread(request,'/console8090',{},60)
                state['notice']='Консоль 8090 запущена: http://192.168.0.177:8090 (возврат — «▶ Я рядом — включить» здесь)'
            elif action=='correction_done':
                recorder.end_correction();state['notice']='Коррекция завершена, управление у политики'
            elif action=='emergency':
                recorder.mark_emergency()  # before STOP, so the pilot record and provenance say 'emergency'
                await asyncio.to_thread(operator.stop,'Аварийное вмешательство оператора')
                recorder.finish('aborted',emergency=True);state['notice']='Аварийное вмешательство: STOP, эпизод помечен'
            elif action=='arm_mode':
                if value.get('mode') not in {'cartesian','joint'}:raise ValueError('Неизвестный режим руки')
                state['mode']=value['mode']
                if operator.owner==owner:operator.arm_mode=value['mode']  # switched live, no stop
                state['notice']='Режим руки: '+('суставы' if value['mode']=='joint' else 'захват XYZ')
            elif action in ('mirror_on','mirror_off','mirror_toggle'):
                # Idempotent on/off: a double click must never switch the mirror on and straight off.
                wanted=(not operator.mirror) if action=='mirror_toggle' else action=='mirror_on'
                if wanted and operator.mirror and operator.owner==owner:
                    state['notice']='Зеркало уже включено'
                elif not wanted:
                    operator.disable_mirror(owner);state['notice']='Зеркало выключено, ручное управление продолжается'
                else:
                    if operator.owner!=owner or not operator.lease:
                        # One button: enable control first (same checks as «▶ Включить»).
                        if operator.lease:await asyncio.to_thread(operator.stop,'Переход к зеркалу')
                        health=await asyncio.to_thread(lambda:(ensure_agent(),request('/health'))[1])
                        if not health.get('hardware_open'):await asyncio.to_thread(request,'/acquire',{},45)
                        await asyncio.to_thread(operator.enable,owner,'rosmaster_manual',state['mode'],True)
                    if operator.leader_sample is None or time.monotonic()-operator.leader_sample[0]>.2:
                        await asyncio.to_thread(operator.start_leader,owner)
                        end=time.monotonic()+5
                        while time.monotonic()<end and (operator.leader_sample is None or time.monotonic()-operator.leader_sample[0]>.2):
                            if operator.leader_error:raise RuntimeError(operator.leader_error)
                            await asyncio.sleep(.1)
                    operator.enable_mirror(owner);state['notice']='Зеркало включено: двигайте Leader понемногу'
            elif action=='leader':
                state['notice']=await asyncio.to_thread(operator.start_leader,owner)
            elif action=='leader_off':
                await asyncio.to_thread(operator.stop_leader);state['notice']='Leader отключён'
            elif action=='mirror':
                if value.get('on'):
                    operator.enable_mirror(owner);state['notice']=''
                else:
                    operator.disable_mirror(owner);state['notice']='Зеркало выключено, ручное управление продолжается'
            elif action=='preset':
                if operator.owner!=owner:raise PermissionError('Сначала включите управление')
                await asyncio.to_thread(request,'/preset',{'lease':operator.lease,'name':value.get('name')})
                state['notice']='Переход в выбранную позу'
            else:raise ValueError('Эта функция ещё не перенесена в Explorer')
        except Exception as exc:
            state['notice']=str(exc)
            log_action(owner,action,value,exc)
        else:log_action(owner,action,value,None)

    async def serial(action,value):
        async with action_lock:
            await act(action,value)

    task=asyncio.create_task(push())
    try:
        while True:
            value=await socket.receive_json()
            action=value.get('action')
            if action is None:
                operator.input(owner,{key:value[key] for key in ('drive','arm','pitch','roll','grip','slow','free') if key in value})
            elif action=='stop':
                # STOP never waits behind another action.
                await act(action,value)
            else:
                # Long actions (Leader start, acquire) run aside, so 'tab alive' inputs keep flowing.
                pending.append(asyncio.create_task(serial(action,value)))
                pending[:]=[t for t in pending if not t.done()]
    except (WebSocketDisconnect,WebSocketDisconnected):pass
    finally:
        task.cancel()
        try:await task
        except (asyncio.CancelledError,WebSocketDisconnect,RuntimeError):pass
        await asyncio.to_thread(operator.lost,owner)
