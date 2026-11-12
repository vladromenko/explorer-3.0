"""ROSMASTER agent entrypoint. Starts inert and cannot acquire hardware without D acceptance."""
import hmac
import json
import os
from pathlib import Path
import threading
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

ROOT = Path(os.environ.get('EXPLORER3_ROOT', '/home/vlad/explorer3'))
lock = threading.RLock()
runtime = None
observer = None
observer_error = None


@asynccontextmanager
async def lifespan(app):
    global observer, observer_error
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
    return {'robot': 'rosmaster', 'hardware_open': runtime is not None,
            'armed': False if runtime is None else runtime.armed,
            'monotonic_ns': time.monotonic_ns(), 'physical_acceptance': False,
            'observation': observer.snapshot() if observer else None,
            'observation_error': observer_error,
            'status': runtime.status() if runtime else None,
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


@app.post('/session')
def session():
    with lock:
        if not (ROOT/'stage-d-authorized.json').is_file():
            raise HTTPException(409, 'Нужна отдельная физическая проверка ROSMASTER')
        # The working keyboard scheme has been copied. Hardware lifecycle is still under
        # acceptance; a marker alone must never make incomplete control reachable.
        raise HTTPException(409, 'Передача ресурсов и драйвер Explorer 3.0 ещё не приняты')


@app.post('/stop')
def stop():
    with lock:
        if runtime:
            runtime.stop('Operator STOP')
        return {'armed': False, 'physical_stop_confirmed': False}


@app.get('/controls')
def controls():
    return {'source': 'CubeBot Console 8090', 'prepared_only': True,
            'arm_modes': ['cartesian', 'joint'], 'leader_mapper': 'cubebot.mirror.MirrorMapper',
            'reason': 'Native control copied; Explorer runtime integration pending'}
