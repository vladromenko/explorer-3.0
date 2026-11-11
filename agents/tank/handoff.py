"""Host-side fixed handoff actions. Run as root; no caller-supplied shell or device paths."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path('/home/pi/explorer3/control')
REMOTE='/home/ubuntu/explorer3'
STATE=ROOT/'handoff.json'
BOOT=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
SERVICES=('start_node.service','wifi.service')
AGENT='explorer3-tank-agent.service'


def run(*args,check=True,timeout=25):
    return subprocess.run(args,text=True,capture_output=True,check=check,timeout=timeout)


def docker(*args):return run('docker','exec','MentorPi',*args)


def active(service):return run('systemctl','is-active',service,check=False).returncode==0


def remote_json(name):
    return json.loads(docker('cat',REMOTE+'/'+name).stdout)


def api(path):
    cfg=remote_json('config.json')
    request=urllib.request.Request('http://127.0.0.1:8041'+path,
        headers={'X-Explorer-Token':cfg['token']},method='GET' if path=='/health' else 'POST')
    with urllib.request.urlopen(request,timeout=12) as response:return json.load(response)


def owners():
    found=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            hits=[]
            for fd in (proc/'fd').iterdir():
                try:
                    device=os.readlink(fd)
                    if device=='/dev/ttyACM0' or device.startswith('/dev/bus/usb/'):
                        hits.append(device)
                except FileNotFoundError:pass  # Descriptor closed during inspection.
            if hits:found.append({'pid':int(proc.name),'devices':sorted(set(hits)),
                                  'command':(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()})
        except (FileNotFoundError,ProcessLookupError):pass
    return found


def save(state):
    temporary=STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,indent=2));temporary.chmod(0o600);temporary.replace(STATE)


def marker(enabled):
    if enabled:
        value=json.dumps({'boot_id':BOOT})
        docker('python3','-c',"from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])",REMOTE+'/handoff-active.json',value)
    else:docker('rm','-f',REMOTE+'/handoff-active.json')


def factory_launches():
    result=docker('python3','-c',r"""import json
from pathlib import Path
found=[]
for proc in Path('/proc').glob('[0-9]*'):
    try:
        args=(proc/'cmdline').read_bytes().split(b'\0')
        if args[:5]==[b'/usr/bin/python3',b'/opt/ros/humble/bin/ros2',b'launch',b'bringup',b'bringup.launch.py']:
            found.append(int(proc.name))
    except (FileNotFoundError,ProcessLookupError):pass
print(json.dumps(found))
""")
    return json.loads(result.stdout)


def enter():
    authorization=remote_json('physical-block-authorized.json')
    if authorization.get('stage')!='C':raise PermissionError('Physical block C authorization missing')
    if api('/health')['hardware_open']:raise RuntimeError('First STOP and close the Explorer session')
    if STATE.exists():
        previous=json.loads(STATE.read_text())
        if not previous['restored']:
            if previous['boot_id']!=BOOT:raise RuntimeError('Unrestored handoff from another boot; restore first')
            if owners():raise RuntimeError('Hardware still owned after previous handoff; restore first')
            marker(True)
            return previous
    launches=factory_launches()
    if len(launches)>1:raise RuntimeError('Multiple factory launches; ownership ambiguous')
    state={'boot_id':BOOT,'services':{s:active(s) for s in SERVICES},'restored':False,'phase':'releasing','created_at':time.time()}
    save(state)
    # ROS launch propagates SIGINT to its own children; the stock board node sends neutral on shutdown.
    if launches:
        docker('kill','-INT',str(launches[0]))
        deadline=time.monotonic()+16
        while factory_launches() and time.monotonic()<deadline:time.sleep(.2)
        if factory_launches():raise RuntimeError('Factory launch did not stop; no Explorer authorization issued')
    for service in SERVICES:
        if state['services'][service]:run('systemctl','stop',service)
    held=owners()
    if held:raise RuntimeError('Remaining device owners: '+json.dumps(held))
    marker(True)
    state['phase']='explorer_ready';save(state)
    return state


def restore():
    if not STATE.exists():return {'restored':True,'reason':'No handoff performed'}
    state=json.loads(STATE.read_text())
    if set(state['services'])!=set(SERVICES):raise ValueError('Invalid saved service state')
    if state['restored']:return state
    if active(AGENT):
        status=api('/stop')
        if status['hardware_open'] or status['armed']:raise RuntimeError('Explorer still owns hardware')
    marker(False)
    # Stop own SDK process before the stock launcher invokes its broad stop_ros script.
    run('systemctl','stop',AGENT)
    held=owners()
    # A partially failed handoff can leave original services up. Never kill unrelated processes.
    if held and not factory_launches():raise RuntimeError('Device owners remain: '+json.dumps(held))
    for service in SERVICES:
        if state['services'][service] and not active(service):run('systemctl','start',service)
    state['restored']=True;state['phase']='factory';state['restored_at']=time.time();save(state)
    run('systemctl','start',AGENT)
    return state


def status():
    return {'services':{s:active(s) for s in SERVICES},'agent_active':active(AGENT),
            'owners':owners(),'factory_launches':factory_launches(),
            'handoff':json.loads(STATE.read_text()) if STATE.exists() else None}


if __name__=='__main__':
    if os.geteuid()!=0:raise PermissionError('Host root required for complete owner inspection')
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'handoff.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        action=sys.argv[1] if len(sys.argv)==2 else None
        if action not in ('status','enter','restore'):raise ValueError('Expected status, enter or restore')
        print(json.dumps({'status':status,'enter':enter,'restore':restore}[action]()))
