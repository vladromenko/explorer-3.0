"""Isolated native physics + official Studio WebViewer; localhost control API only."""
import argparse
import base64
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
from mujoco.studio import launch_web, viewer_protocol
from .factory import make,ENVIRONMENTS,EXPERTS
from .capture import SimCapture
from .studio import TaskCamera
from .ownership import OperatorLease
from explorer3.paths import ROOT, DATA, offline_environment
import subprocess


def run(api_port=8032, viewer_port=8031):
    running=threading.Event();running.set()
    signal.signal(signal.SIGTERM,lambda *_:running.clear())
    env = make('so101_pick_place_sim')
    guard = threading.RLock()
    lease=OperatorLease()
    state = {'paused':True, 'error':None, 'target':env.neutral_action(),
             'last_input':0.,'seed':0,'rgb':None,'frames':0,'outcome':None,'recording':False,'finalizing':False,'episode':None,
             'profile':env.profile,'controller':'manual','requested_profile':None,'limits':[env.action_space.low.tolist(),env.action_space.high.tolist()]}
    handle = launch_web.launch(viewer_protocol.ViewerConfig(title='Explorer · MuJoCo',
        gfx='web', http_port=viewer_port, open_browser=False), plugins=[TaskCamera()])
    handle.set_model(env.model,data=env.data)

    capture = None
    teacher = None
    saving=[]
    last_render = 0.

    def finish_capture(outcome, reason):
        nonlocal capture
        if capture is None:
            return
        pending=capture
        capture = None
        try:root=pending.finish(outcome, reason)
        except Exception as exc:
            state.update(recording=False,finalizing=False,paused=True,error=f'Запись не сохранена: {exc}')
            return
        state.update(recording=False, finalizing=True)
        def save():
            try:
                with (root/'import.log').open('w') as log:
                    subprocess.run([str(ROOT/'.venv/bin/python'),'-m','explorer3.sim.import_episode',str(root)],
                                   env=offline_environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
                with guard:
                    state['episode'] = json.loads((root/'imported.json').read_text())['episode_id']
            except Exception as exc:
                with guard:
                    state['error'] = str(exc)+'; '+str(root/'import.log')
            finally:
                with guard:
                    state['finalizing'] = False
        thread=threading.Thread(target=save,daemon=False)
        saving.append(thread);thread.start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def respond(self,code,value):
            body = json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            with guard:
                self.respond(200,{**state, 'owner':lease.owner, 'observation':env.observation()['state'].tolist(), 'task':env.info()})

        def do_POST(self):
            nonlocal capture,teacher
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0<length<10000:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(length))
                with guard:
                    command = data['command']
                    token=data.get('operator')
                    if command not in {'claim','pause','release'}:lease.require(token)
                    if state['requested_profile'] is not None and command not in {'claim','pause','release','heartbeat'}:
                        raise ValueError('Модель загружается; дождитесь готовности сцены')
                    if command=='claim':
                        lease.claim(token)
                    elif command=='heartbeat':
                        pass
                    elif command=='release':
                        if lease.owner==token:
                            if teacher is not None:finish_capture('aborted','scripted_operator_left')
                            state.update(paused=True,controller='manual');teacher=None
                        lease.release(token)
                    elif command == 'profile':
                        if data['profile'] not in ENVIRONMENTS:raise ValueError('Среда ещё не готова')
                        finish_capture('aborted','profile_changed')
                        state.update(paused=True,controller='manual',requested_profile=data['profile']);teacher=None
                    elif command=='demonstrate':
                        if capture is not None or state['finalizing']:raise ValueError('Сначала закончите запись')
                        env.reset(seed=int(data['seed']),options={'variation':data.get('variation',env.variation_mode)})
                        teacher=EXPERTS[env.profile](env)
                        state.update(paused=False,controller='scripted_expert',last_input=time.monotonic(),seed=int(data['seed']),
                            outcome=None,error=None,frames=0,target=env.neutral_action(),episode=None)
                        handle.set_model(env.model,data=env.data)
                    elif command == 'record':
                        if capture is not None or state['finalizing']:
                            raise ValueError('Запись уже идёт или сохраняется')
                        capture=SimCapture(env,state['seed'],'scripted_expert' if teacher is not None else 'human')
                        state.update(recording=True,episode=None)
                    elif command == 'finish':
                        finish_capture('success' if env.info()['success'] else 'failure','operator_finished')
                        state.update(paused=True,controller='manual');teacher=None
                    elif command == 'reset':
                        finish_capture('aborted','scene_reset')
                        env.reset(seed=int(data['seed']),options={'variation':data.get('variation',env.variation_mode)})
                        handle.set_model(env.model,data=env.data)
                        teacher=None
                        state.update(seed=int(data['seed']),paused=True,controller='manual',error=None,outcome=None,
                                     target=env.neutral_action(),frames=0)
                    elif command == 'target':
                        if teacher is not None:raise ValueError('Идёт заданная проверка; сначала нажмите STOP')
                        import numpy as np
                        target = np.asarray(data['target'],dtype=float)
                        if target.shape!=env.action_space.shape or not np.isfinite(target).all():
                            raise ValueError('Invalid control channel count or value')
                        if state['outcome']:
                            raise ValueError('Эпизод завершён; сбросьте сцену')
                        state.update(target=np.clip(target,env.action_space.low,env.action_space.high).tolist(),
                                     last_input=time.monotonic(),paused=False)
                    elif command == 'pause':
                        if teacher is not None:finish_capture('aborted','scripted_check_stopped')
                        state.update(paused=True,controller='manual');teacher=None
                        lease.release(force=True)
                    else:
                        raise ValueError('Unknown simulation command')
                    self.respond(200,{'ok':True,'paused':state['paused']})
            except (ValueError,KeyError,OSError,RuntimeError) as exc:
                self.respond(400,{'error':str(exc)})

    server = ThreadingHTTPServer(('127.0.0.1',api_port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        while running.is_set() and handle.is_running():
            started = time.monotonic()
            with guard:
                if state['requested_profile'] is not None:
                    chosen=state['requested_profile']
                    variation=env.variation_mode;env.close();env=make(chosen,variation=variation)
                    state.update(profile=chosen,requested_profile=None,target=env.neutral_action(),
                        limits=[env.action_space.low.tolist(),env.action_space.high.tolist()],frames=0,rgb=None,outcome=None,episode=None)
                    handle.set_model(env.model,data=env.data)
                if lease.expire() or (teacher is None and started-state['last_input'] > .75):
                    if teacher is not None:finish_capture('aborted','scripted_operator_expired')
                    state.update(paused=True,controller='manual');teacher=None
                if not state['paused']:
                    try:
                        if teacher is not None:state['target']=teacher.action().tolist()
                        if capture is not None:
                            capture.add(env,state['target'])
                        _,_,terminated,truncated,info = env.step(state['target'])
                        state['frames'] += 1
                        if terminated or truncated:
                            state.update(paused=True,outcome=info['outcome'])
                            teacher=None;state['controller']='manual'
                            finish_capture(info['outcome'],info['termination_reason'])
                    except Exception as exc:
                        teacher=None
                        state.update(paused=True,controller='manual',outcome='aborted',error=f'Сцена остановлена: {exc}')
                        lease.release(force=True)
                        finish_capture('aborted',f'technical_error:{type(exc).__name__}:{exc}')
                if started-last_render>.15 or state['rgb'] is None:
                    last_render=started
                    rgb = env.render()
                    ok,jpg = cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,75])
                    if not ok:
                        raise RuntimeError('Camera encoding failed')
                    state['rgb'] = base64.b64encode(jpg).decode()
                handle.sync(env.model,env.data)
            time.sleep(max(0.,1/env.fps-(time.monotonic()-started)))
    finally:
        finish_capture('aborted','worker_shutdown')
        server.shutdown();handle.close();env.close()
        for thread in saving:thread.join(timeout=60)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--api-port',type=int,default=8032)
    parser.add_argument('--viewer-port',type=int,default=8031)
    args=parser.parse_args()
    run(args.api_port,args.viewer_port)
