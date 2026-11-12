"""ROSMASTER agent entrypoint. Starts inert and cannot acquire hardware without D acceptance."""
import hmac
import json
import os
from pathlib import Path
import threading
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import Response

ROOT = Path(os.environ.get('EXPLORER3_ROOT', '/home/vlad/explorer3'))
lock = threading.RLock()
runtime = None
observer = None
observer_error = None


@asynccontextmanager
async def lifespan(app):
    global observer, observer_error, runtime
    if config().get('observation_enabled') is True:
        from observer import Observer
        observer = Observer()
        try:
            observer.start()
        except Exception as exc:
            observer_error = str(exc)
    try:
        yield
    finally:
        if runtime is not None:
            runtime.close()
            runtime=None
        if observer is not None:
            observer.close()
            observer = None


app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)


def config():
    return json.loads((ROOT/'config/agent.json').read_text())


@app.middleware('http')
async def authenticate(request: Request, call_next):
    from fastapi.responses import JSONResponse
    cfg = config()
    if request.client.host not in {'127.0.0.1', cfg['homebrain']} or not hmac.compare_digest(
            request.headers.get('X-Explorer-Token', ''), cfg['token']):
        return JSONResponse({'detail': 'Unauthorized'}, status_code=403)
    return await call_next(request)


@app.get('/health')
def health():
    observation=observer.snapshot() if observer else None
    state=runtime.status() if runtime else None
    if state is not None and observation is not None:
        observation={**observation,'source':'native_serial_owner','read_only':False,
            'motor_commands_sent':state['motion_commands_sent']>0,
            'arm_measured':state['arm_measured'],'arm_complete':state['arm_measured'] is not None,
            'arm_received_ns':None,'battery_received_ns':None,'odom_received_ns':None,
            'battery_voltage':state['battery_voltage'],'measured_velocity':state['measured_velocity'],
            'controller_command_readiness':state['controller_ready'],
            'arm_outside_native_model':None}
    return {'robot': 'rosmaster', 'hardware_open': runtime is not None,
            'armed': False if runtime is None else runtime.armed,
            'monotonic_ns': time.monotonic_ns(), 'physical_acceptance': False,
            'observation': observation,
            'observation_error': observer_error,
            'status': state,
            'prepared': (ROOT/'native-manifest.json').exists(),
            'reason': 'Separate physical block D not accepted' if not (ROOT/'stage-d-authorized.json').exists() else 'Explicit start required'}


@app.get('/frame.jpg')
def frame():
    if observer is None:
        raise HTTPException(503, 'Read-only observer is not running')
    try:
        data, state = observer.jpeg()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return Response(data, media_type='image/jpeg', headers={
        'Cache-Control': 'no-store',
        'X-Explorer-Camera': json.dumps(state['camera']),
        'X-Explorer-Observation': json.dumps(state),
    })


@app.get('/sample.jpg')
def sample():
    """Camera frame first, then the control snapshot: the frame is never newer than the state."""
    current=runtime
    if observer is None:raise HTTPException(503,'Camera observer is not running')
    if current is None:raise HTTPException(409,'Runtime not acquired')
    try:data,camera_state=observer.jpeg()
    except RuntimeError as exc:raise HTTPException(503,str(exc)) from exc
    state=current.sample_state()
    return Response(data,media_type='image/jpeg',headers={'Cache-Control':'no-store',
        'X-Explorer-Camera':json.dumps(camera_state['camera']),'X-Explorer-Sample':json.dumps(state)})


@app.post('/session')
def session(value:dict=Body(default={})):
    with lock:
        if not (ROOT/'stage-d-authorized.json').is_file():
            raise HTTPException(409, 'Нужна отдельная физическая проверка ROSMASTER')
        if runtime is None:raise HTTPException(409,'Сначала передайте управление Explorer')
        try:return runtime.enable(value.get('profile'),value.get('arm_mode','cartesian'),value.get('observing') is True,
                                  value.get('source','human'))
        except (ValueError,RuntimeError,PermissionError) as exc:raise HTTPException(409,str(exc)) from exc


@app.post('/acquire')
def acquire():
    global runtime
    with lock:
        if not (ROOT/'stage-d-authorized.json').is_file():raise HTTPException(409,'Физический блок ROSMASTER не разрешён')
        if runtime is not None:return {'acquired':True,'status':runtime.status()}
        from handoff import Handoff
        from runtime import Runtime
        handoff=Handoff(ROOT)
        try:
            handoff.enter()
            candidate=Runtime(ROOT)
            try:candidate.start()
            except Exception:
                handoff.restore()
                raise
            runtime=candidate
            return {'acquired':True,'armed':False}
        except Exception as exc:raise HTTPException(409,str(exc)) from exc


@app.post('/command')
def command(value:dict=Body(...)):
    current=runtime
    if current is None:raise HTTPException(409,'Runtime not acquired')
    try:return current.command(value.get('lease',''),value.get('sequence'),value.get('intent',{}),
                                    value.get('leader'),value.get('mirror') is True,value.get('direct'))
    except (ValueError,RuntimeError,PermissionError,TypeError) as exc:
        if value.get('lease')==current.lease:current.stop('Command rejected: '+str(exc))
        raise HTTPException(409,str(exc)) from exc


@app.post('/preset')
def preset(value:dict=Body(...)):
    if runtime is None:raise HTTPException(409,'Runtime not acquired')
    try:return runtime.preset(value.get('lease',''),value.get('name'))
    except (ValueError,RuntimeError,PermissionError) as exc:raise HTTPException(409,str(exc)) from exc


@app.post('/restore')
def restore():
    global runtime
    with lock:
        from handoff import Handoff
        try:
            if runtime is not None:runtime.close();runtime=None
            return Handoff(ROOT).restore()
        except Exception as exc:raise HTTPException(409,str(exc)) from exc


@app.get('/ownership')
def ownership():
    from handoff import Handoff
    return Handoff(ROOT).status()


@app.post('/stop')
def stop():
    current=runtime
    if current:current.stop('Operator STOP')
    return {'armed': False, 'physical_stop_confirmed': False}


@app.get('/controls')
def controls():
    return {'source': 'CubeBot Console 8090', 'prepared_only': False,
            'arm_modes': ['cartesian', 'joint'], 'leader_mapper': 'cubebot.mirror.MirrorMapper',
            'physical_acceptance':False,'reason': 'Explicit ownership and control session required'}
