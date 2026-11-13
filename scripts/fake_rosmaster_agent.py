"""Fake ROSMASTER agent HTTP server for browser acceptance (no robot, no serial). Records every request."""
import json, sys, time, threading
import cv2, numpy as np, uvicorn
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import Response

app = FastAPI()
S = {'acquired': False, 'armed': False, 'lease': None, 'profile': None, 'seq': -1, 'step': 0, 'reason': 'Explicit enable required',
     'last_intent': None, 'requests': []}
LOCK = threading.Lock()


def log(kind, value=None):
    S['requests'].append({'t': time.time(), 'kind': kind, 'value': value})


def status():
    return {'armed': S['armed'], 'profile': S['profile'], 'reason': S['reason'], 'error': None, 'mode': 'manual',
            'arm_measured': [90., 90., 90., 90., 90., 30.], 'arm_command': [90, 90, 90, 90, 90, 30], 'arm_blocked': None,
            'arm_mode': 'joint', 'tcp': [0.03, 0., 0.54], 'guard': None, 'mirror_target': [90.] * 6, 'mirror_blocked': None,
            'battery_voltage': 12.1, 'measured_velocity': [0., 0., 0.], 'controller_ready': True, 'logs': [],
            'motion_commands_sent': 0, 'mirror_sign': [1, -1, -1, -1, 1], 'mirror_gripper_map': [100., 0.],
            'mirror_ref_pair': None, 'mirror_aligning': False, 'preparing': False}


@app.get('/health')
def health():
    return {'robot': 'rosmaster', 'hardware_open': S['acquired'], 'armed': S['armed'], 'monotonic_ns': time.monotonic_ns(),
            'status': status() if S['acquired'] else None, 'observation': None}


@app.get('/_requests')
def requests():
    return S['requests']


@app.post('/acquire')
def acquire():
    log('acquire'); S['acquired'] = True; return {'acquired': True}


@app.post('/session')
def session(value: dict = Body(default={})):
    log('session', value)
    if not S['acquired']: raise HTTPException(409, 'Сначала передайте управление Explorer')
    S.update(armed=True, lease='L' + str(time.time()), profile=value.get('profile'), seq=-1, reason=None)
    return {'lease': S['lease'], 'profile': S['profile']}


@app.post('/command')
def command(value: dict = Body(...)):
    if not S['armed'] or value.get('lease') != S['lease']: raise HTTPException(409, 'Сессия остановлена')
    S['seq'] = value['sequence']; S['last_intent'] = value.get('intent'); log('command', value.get('intent'))
    return status()


@app.post('/stop')
def stop():
    log('stop'); S.update(armed=False, lease=None, reason='Operator STOP'); return {'armed': False}


def jpeg():
    S['step'] += 1
    return cv2.imencode('.jpg', np.full((480, 640, 3), S['step'] % 255, np.uint8))[1].tobytes()


@app.get('/frame.jpg')
def frame():
    return Response(jpeg(), media_type='image/jpeg')


@app.get('/sample.jpg')
def sample():
    now = time.monotonic_ns(); data = jpeg(); intent = S['last_intent'] or {}
    state = {'agent_monotonic_ns': now, 'control_step': S['step'], 'armed': S['armed'], 'profile': S['profile'], 'preparing': False,
             'mode': 'manual', 'command_source': 'human', 'operator_intent': intent, 'direct_requested': None, 'direct_received_ns': None,
             'base_requested': [0, 0, 0], 'base_after_limits': [0, 0, 0], 'base_guard': None, 'base_sent': [0., 0., 0.], 'base_sent_ns': None,
             'arm_sent': None, 'arm_sent_ns': None, 'arm_blocked': None, 'arm_target': [90., 90., 90., 90., 90., 30.],
             'arm_measured': [90., 90., 90., 90., 90., 30.], 'arm_measured_ages_s': [.01] * 6, 'arm_measured_valid': [True] * 6,
             'base_measured': [0., 0., 0.], 'base_measured_age_s': .02, 'battery_voltage': 12.1, 'battery_age_s': .1,
             'controller_ready': True, 'arm_cancel_recoveries': 0, 'motion_commands_sent': 0, 'clock_resyncs': 0}
    camera = {'available': True, 'received_ns': now - 20_000_000, 'source_timestamp_ns': 1, 'source_clock': 'fake', 'sequence': S['step']}
    return Response(data, media_type='image/jpeg', headers={'X-Explorer-Sample': json.dumps(state), 'X-Explorer-Camera': json.dumps(camera)})


@app.post('/preset')
def preset(value: dict = Body(...)):
    log('preset', value); return {}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=int(sys.argv[1]), log_level='warning')
