"""ROSMASTER agent entrypoint. Starts inert and cannot acquire hardware without D acceptance."""
import hmac
import json
import os
from pathlib import Path
import threading
import time
from fastapi import FastAPI, HTTPException, Request

ROOT = Path(os.environ.get('EXPLORER3_ROOT', '/home/vlad/explorer3'))
app = FastAPI(docs_url=None, redoc_url=None)
lock = threading.RLock()
runtime = None


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
            'status': runtime.status() if runtime else None,
            'prepared': (ROOT/'native-manifest.json').exists(),
            'reason': 'Separate physical block D not accepted' if not (ROOT/'stage-d-authorized.json').exists() else 'Explicit start required'}


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
