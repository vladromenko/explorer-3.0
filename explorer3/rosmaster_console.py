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


@app.get('/rosmaster/control',response_class=HTMLResponse)
def console():
    return (Path(__file__).parent/'static/rosmaster-console.html').read_text()


def telemetry(health,mode,notice):
    s=health.get('status') or {};observation=health.get('observation') or {}
    measured=s.get('arm_measured') or observation.get('arm_measured')
    arm=s.get('arm_command') or measured or []
    sample=operator.leader_sample
    age=time.monotonic()-sample[0] if sample else None
    return {'arm_mode':mode,'dry_run':False,'enabled':s.get('armed',False),
        'notice':notice or s.get('reason') or ('Управление включено' if s.get('armed') else 'Включите управление'),
        'mode':s.get('mode','manual'),'battery':{'soc':None,'minutes_left':None,'volts':s.get('battery_voltage',observation.get('battery_voltage'))},
        'battery_stale':False,'arm':[round(v,1) if v is not None else '?' for v in arm],
        'arm_measured':measured,'tcp':s.get('tcp') or [],'jog_blocked':s.get('arm_blocked'),
        'mirror':{'on':s.get('mode')=='mirror','blocked':s.get('mirror_blocked') or operator.leader_error,'frames':operator.leader_count,
                  'leader':'ошибка' if operator.leader_error else 'подключён' if sample else 'не подключён',
                  'age':round(age,2) if age is not None else None,'target':s.get('mirror_target') or []},
        'guard':s.get('guard'),'mission':'не запущена','mission_result':None,
        'log':(s.get('logs',[])+([operator.error] if operator.error else [])+([notice] if notice else []))[-14:]}


@app.websocket('/api/rosmaster/console/ws')
async def websocket(socket:WebSocket):
    await socket.accept()
    owner=uuid.uuid4().hex
    state={'mode':'cartesian','notice':''}
    async def push():
        with connection(1) as client:
            while True:
                try:
                    response=await asyncio.to_thread(client.get,'/health');response.raise_for_status()
                    health=response.json()
                    payload=telemetry(health,state['mode'],state['notice'])
                except Exception as exc:payload=telemetry({},state['mode'],'Нет связи: '+str(exc))
                await socket.send_json(payload)
                await asyncio.sleep(.1)
    task=asyncio.create_task(push())
    try:
        while True:
            value=await socket.receive_json()
            action=value.get('action')
            if action is None:
                operator.input(owner,{key:value[key] for key in ('drive','arm','pitch','roll','grip','slow','free') if key in value})
            else:
                try:
                    if action=='stop':
                        reason='Браузер: '+str(value.get('reason','кнопка STOP'))[:100]
                        await asyncio.to_thread(operator.stop,reason);state['notice']='Остановлено · '+reason
                    elif action=='acquire':
                        state['notice']='Запуск агента ROSMASTER…' if await asyncio.to_thread(ensure_agent) else state['notice']
                        await asyncio.to_thread(request,'/acquire',{},45);state['notice']='Explorer получил управление. Нажмите «Включить управление»'
                    elif action=='restore':
                        await asyncio.to_thread(operator.stop,'Возврат прежнего управления')
                        await asyncio.to_thread(request,'/restore',{},45);state['notice']='Прежнее управление восстановлено'
                    elif action in ('enable','prepare'):
                        result=await asyncio.to_thread(operator.enable,owner,'prepare' if action=='prepare' else 'rosmaster_manual',state['mode'],value.get('observing') is True)
                        state['notice']='Управление уже включено: двигайте клавишами (окно должно быть в фокусе)' if result=='already' else ''
                    elif action=='arm_mode':
                        if value.get('mode') not in {'cartesian','joint'}:raise ValueError('Неизвестный режим руки')
                        if operator.owner==owner:await asyncio.to_thread(operator.stop,'Смена режима руки')
                        state['mode']=value['mode'];state['notice']='Режим изменён. Нажмите «Включить управление»'
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
                except Exception as exc:state['notice']=str(exc)
    except (WebSocketDisconnect,WebSocketDisconnected):pass
    finally:
        task.cancel()
        try:await task
        except (asyncio.CancelledError,WebSocketDisconnect,RuntimeError):pass
        await asyncio.to_thread(operator.lost,owner)
