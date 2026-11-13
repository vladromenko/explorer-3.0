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
from .environment import SO101PickPlace
from .capture import SimCapture
from explorer3.paths import ROOT, DATA, offline_environment
import subprocess


def run(api_port=8032, viewer_port=8031):
    running=threading.Event();running.set()
    signal.signal(signal.SIGTERM,lambda *_:running.clear())
    env = SO101PickPlace()
    guard = threading.RLock()
    state = {'paused':True, 'error':None, 'target':env.observation()['state'].tolist(),
             'last_input':0.,'seed':0,'rgb':None,'frames':0,'outcome':None,'recording':False,'finalizing':False,'episode':None}
    handle = launch_web.launch(viewer_protocol.ViewerConfig(title='Explorer · MuJoCo · SO-101',
        gfx='web', http_port=viewer_port, open_browser=False), plugins=[])
    handle.set_model(env.model,data=env.data)

    capture = None
    last_render = 0.

    def finish_capture(outcome, reason):
        nonlocal capture
        if capture is None:
            return
        root = capture.finish(outcome, reason)
        capture = None
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
        threading.Thread(target=save,daemon=True).start()

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
                self.respond(200,{**state, 'observation':env.observation()['state'].tolist(), 'task':env.info()})

        def do_POST(self):
            nonlocal capture
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0<length<10000:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(length))
                with guard:
                    command = data['command']
                    if command == 'record':
                        if capture is not None or state['finalizing']:
                            raise ValueError('Запись уже идёт или сохраняется')
                        capture=SimCapture(env,state['seed'])
                        state.update(recording=True,episode=None)
                    elif command == 'finish':
                        finish_capture('success' if env.info()['success'] else 'failure','operator_finished')
                        state['paused']=True
                    elif command == 'reset':
                        finish_capture('aborted','scene_reset')
                        env.reset(seed=int(data['seed']))
                        state.update(seed=int(data['seed']),paused=True,error=None,outcome=None,
                                     target=env.observation()['state'].tolist(),frames=0)
                    elif command == 'target':
                        import numpy as np
                        target = np.asarray(data['target'],dtype=float)
                        if target.shape!=(6,) or not np.isfinite(target).all():
                            raise ValueError('Six finite joint targets required')
                        if state['outcome']:
                            raise ValueError('Эпизод завершён; сбросьте сцену')
                        state.update(target=np.clip(target,env.action_space.low,env.action_space.high).tolist(),
                                     last_input=time.monotonic(),paused=False)
                    elif command == 'pause':
                        state['paused'] = True
                    else:
                        raise ValueError('Unknown simulation command')
                    self.respond(200,{'ok':True,'paused':state['paused']})
            except (ValueError,KeyError) as exc:
                self.respond(400,{'error':str(exc)})

    server = ThreadingHTTPServer(('127.0.0.1',api_port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        while running.is_set() and handle.is_running():
            started = time.monotonic()
            with guard:
                if started-state['last_input'] > .75:
                    state['paused'] = True
                if not state['paused']:
                    if capture is not None:
                        capture.add(env,state['target'])
                    _,_,terminated,truncated,info = env.step(state['target'])
                    state['frames'] += 1
                    if terminated or truncated:
                        state.update(paused=True,outcome=info['outcome'])
                        finish_capture(info['outcome'],info['termination_reason'])
                if started-last_render>.15 or state['rgb'] is None:
                    last_render=started
                    rgb = env.render()
                    ok,jpg = cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,75])
                    if not ok:
                        raise RuntimeError('Camera encoding failed')
                    state['rgb'] = base64.b64encode(jpg).decode()
                handle.sync(env.model,env.data)
            time.sleep(max(0.,1/30-(time.monotonic()-started)))
    finally:
        finish_capture('aborted','worker_shutdown')
        server.shutdown();handle.close();env.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--api-port',type=int,default=8032)
    parser.add_argument('--viewer-port',type=int,default=8031)
    args=parser.parse_args()
    run(args.api_port,args.viewer_port)
