"""Bounded interactive shared physics. Demonstrations are scripted, not trained."""
import argparse
import base64
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import cv2
from mujoco.studio import launch_web,viewer_protocol
from .fleet import FleetWorld
from .ownership import OperatorLease
from .studio import TaskCamera
from explorer3.fleet_allocation import FleetAllocator


def run(api_port,viewer_port):
    world=FleetWorld();lease=OperatorLease();guard=threading.RLock();alive=threading.Event();alive.set()
    signal.signal(signal.SIGTERM,lambda *_:alive.clear())
    state={'paused':True,'error':None,'owner':None,'rgb':None,'selected':'so101','snapshot_available':False,
           'frames':0,'mode':'manual','last_target':0.}
    saved=None;manual=None
    handle=launch_web.launch(viewer_protocol.ViewerConfig(title='Explorer · общий стенд',gfx='web',http_port=viewer_port,open_browser=False),plugins=[TaskCamera()])
    handle.set_model(world.model,data=world.data)
    def pause(reason="operator_stop"):
        nonlocal manual
        state.update(paused=True,mode='manual',stop_reason=reason);manual=None;world.stop();lease.release(force=True)
    def status():return {**state,'owner':lease.owner,'world':world.info()}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def respond(self,code,value):
            body=json.dumps(value).encode();self.send_response(code);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            with guard:self.respond(200,status())
        def do_POST(self):
            nonlocal saved,manual,world
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<10000:raise ValueError('Invalid request size')
                payload=json.loads(self.rfile.read(size));cmd=payload['command'];token=payload.get('operator')
                with guard:
                    if cmd not in {'claim','pause','release'}:lease.require(token)
                    if cmd=='claim':lease.claim(token)
                    elif cmd=='heartbeat':pass
                    elif cmd=='pause':pause()
                    elif cmd=='release':
                        if lease.owner==token:pause()
                    elif cmd=='select':
                        pause();lease.claim(token);world.assign_operator(payload['agent']);state['selected']=payload['agent']
                    elif cmd=='target':
                        name=state['selected']
                        if state['mode']=='scripted_expert':raise ValueError('Сначала остановите показ')
                        if world.views[name].done:raise ValueError('Задача завершена; сбросьте выбранного робота')
                        world.views[name].env.prepare_action(payload['target'])
                        world.assign_operator(name);manual={name:payload['target']}
                        state.update(paused=False,last_target=time.monotonic())
                    elif cmd=='demonstrate':
                        world.reset(int(payload.get('seed',0)));world.demonstrate();manual=None
                        state.update(paused=False,mode='scripted_expert',error=None,frames=0)
                        handle.set_model(world.model,data=world.data)
                    elif cmd=='scene':
                        from .room import PRESETS
                        if payload.get('scene') not in PRESETS:raise ValueError('Unknown scene')
                        pause();lease.claim(token)
                        replacement=FleetWorld(int(payload.get('seed',0)),payload['scene'])
                        world.close();world=replacement;saved=None
                        state.update(error=None,snapshot_available=False,frames=0)
                        handle.set_model(world.model,data=world.data)
                    elif cmd=='reset':
                        pause();lease.claim(token);world.reset(int(payload.get('seed',0)),payload.get('agent'))
                        state.update(error=None);handle.set_model(world.model,data=world.data)
                    elif cmd=='snapshot':
                        world.lease_state={'owner':lease.owner,'selected':state['selected'],'mode':state['mode']}
                        saved=world.snapshot();state['snapshot_available']=True
                    elif cmd=='restore':
                        if saved is None:raise ValueError('Сначала сохраните состояние')
                        pause();world.restore(saved)
                        # Offline branch state is exact; interactive restore must
                        # never resume old commands or reanimate a departed owner.
                        world.stop();state['selected']=saved['lease_state']['selected']
                    elif cmd=='allocation_check':
                        if not state['paused']:raise ValueError('Сначала STOP')
                        allocator=FleetAllocator(3,payload.get('method','fleet_thrifty'),seed=1000,min_steps=2)
                        selected=allocator.allocate([.8,.9,.1],[.8,.7,.1])
                        world.scheduler_state=allocator.snapshot()
                        world.assign_operator(list(world.views)[selected] if selected is not None else None)
                        state['allocation']={'assignment':world.operator,'simultaneous_requests':['so101','rosmaster'],
                            'source':'declared numerical fixture, not calibrated risk','concurrent_operators':int(allocator.assignments.sum())}
                    else:raise ValueError('Unknown fleet command')
                    self.respond(200,{'ok':True})
            except (ValueError,KeyError,RuntimeError,OSError) as exc:self.respond(400,{'error':str(exc)})
    server=ThreadingHTTPServer(('127.0.0.1',api_port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start();last_render=0.
    try:
        while alive.is_set() and handle.is_running():
            started=time.monotonic()
            with guard:
                if lease.expire():pause('operator_heartbeat_expired')
                if not state['paused'] and state['mode']=='manual' and started-state['last_target']>.75:pause('target_expired')
                if not state['paused']:
                    try:
                        info=world.step(manual);state['frames']+=1
                        if all(v['done'] for v in info['robots'].values()):pause()
                    except Exception as exc:pause();state['error']=str(exc)
                if started-last_render>.2:
                    ok,jpg=cv2.imencode('.jpg',cv2.cvtColor(world.render(state['selected']),cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,70])
                    if ok:state['rgb']=base64.b64encode(jpg).decode()
                    last_render=started
                handle.sync(world.model,world.data)
            time.sleep(max(.001,.05-(time.monotonic()-started)))
    finally:pause();server.shutdown();handle.close();world.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--api-port',type=int,default=8038);p.add_argument('--viewer-port',type=int,default=8039)
    args=p.parse_args();run(args.api_port,args.viewer_port)
